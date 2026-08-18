//! Is prefill chunking actually output-equivalent, and what does the chunk
//! size cost?
//!
//! `forward_chunked` carries a careful argument for why chunking cannot change
//! the result — RoPE and the causal sentinel key off `cache.offset()`, and the
//! linear-attention layers carry their conv/SSM state through the cache, so a
//! gated-delta scan over a chunk starting from the prior chunk's state equals a
//! scan over the concatenation. That argument is reasoned, not tested. This
//! example tests it, because `LUMEN_QWEN35_PREFILL_CHUNK` is the first-order
//! control on prefill peak memory (5.7x across 256..4096 on an 8K prompt) and
//! nobody should retune it on the strength of a comment.
//!
//! The A/B is exact rather than approximate: `qwen35_prefill_chunk()` reads the
//! environment on every call rather than caching it in a `OnceLock`, so a
//! single process can prefill the same prompt at several chunk sizes against
//! the same loaded weights. Same model, same tokens, same greedy decode — the
//! only variable is the chunk. Token ids must match exactly; a single differing
//! id is a correctness bug, not a rounding artifact.
//!
//! Reference chunk is the first in the list (the current default unless
//! `--chunks` overrides), and every other size is compared against it.
//!
//! ```text
//! MODEL_ID=~/models/Qwen3.5-9B-MTPLX-Speed \
//!   cargo run --release -p lumen-mlx --features mlx-native \
//!   --example prefill_chunk_equivalence -- --prompt-tokens 8000 --gen 32
//!
//! # tighter sweep, more generated tokens for a stronger equality check
//! ... -- --chunks 2048,1024,512,256,128 --gen 64
//! ```
//!
//! ## Result on record (Qwen3.5-9B, M3 Max)
//!
//! **Output is chunk-invariant** — bit-identical token ids at 256/512/1024/2048
//! across every run, at both 8K and 20K prompts. The equivalence argument
//! holds.
//!
//! **The latency price is not flat, and that is the reason the default stays at
//! 2048.** At an 8,007-token prompt (4 to 16 chunks) the time deltas over five
//! runs average ~0 against a ±11% run-to-run noise floor, while memory after
//! prefill goes 2,055 → 1,250 → 849 → 646 MB. At 20,000 tokens (10 to 40
//! chunks) the same reduction costs **+9 to +17% at 1024 and +54 to +118% at
//! 512**. Reversing the sweep order reproduces it (512 first: 113 s, 2048
//! second: 52 s), so it is not thermal drift or ordering — the cost tracks
//! chunk *count*, because each chunk is `eval`'d before the next and every one
//! of those is a pipeline serialization point.
//!
//! Measuring at one prompt length would have produced the wrong answer here.
//! It did, once: an 8K-only sweep read as "free" and was briefly written into
//! the docs as a recommendation to lower the default.

use std::time::Instant;

use anyhow::{Result, anyhow};
use lumen_mlx::metal_memory::{clear_cache, get_active_memory};
use lumen_mlx::{MlxBackend, MlxBatchedSeqDriver};

const CHUNK_ENV: &str = "LUMEN_QWEN35_PREFILL_CHUNK";
const DEFAULT_CHUNKS: &[usize] = &[2048, 1024, 512, 256];

/// Filler vocabulary — same bank as `kv_concurrency_ab`, so a prompt of a given
/// target length is comparable between the two harnesses.
const WORDS: &[&str] = &[
    "system", "value", "record", "matrix", "signal", "buffer", "window", "kernel", "vector",
    "planet", "garden", "silver", "market", "reason", "letter", "number", "moment", "figure",
    "branch", "circle", "stream", "camera", "ticket", "island", "friend", "orange", "bridge",
    "packet", "column", "target", "sample", "region", "handle", "author", "series", "corner",
    "policy", "device", "格子", "회로", "réseau",
];

fn filler(n_words: usize) -> String {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut out = String::with_capacity(n_words * 7);
    for w in 0..n_words {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let pick = (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as usize;
        if w > 0 {
            out.push(' ');
        }
        out.push_str(WORDS[pick % WORDS.len()]);
    }
    out
}

fn build_prompt(driver: &dyn MlxBatchedSeqDriver, target: usize) -> Result<Vec<u32>> {
    let mut words = target.max(8);
    let mut best: Option<Vec<u32>> = None;
    for _ in 0..8 {
        let msgs = vec![(
            "user".to_string(),
            format!("Summarize the following log:\n{}", filler(words)),
        )];
        let ids = driver.build_chat_input(&msgs, false, None)?;
        let n = ids.len();
        if n >= target {
            if n <= target + target / 20 {
                return Ok(ids);
            }
            best = Some(ids);
            words = ((words as f64) * (target as f64 / n as f64)).ceil() as usize + 1;
        } else {
            let grow = ((words as f64) * (target as f64 / n.max(1) as f64)).ceil() as usize;
            words = grow.max(words + 4);
        }
    }
    best.ok_or_else(|| anyhow!("could not reach {target} tokens in 8 rounds"))
}

struct Run {
    chunk: usize,
    tokens: Vec<u32>,
    prefill_ms: f64,
    peak: usize,
}

/// Prefill `prompt` at `chunk`, then greedily decode `n_gen` tokens. The
/// sequence is removed and the pool drained afterwards so each run starts from
/// the same floor.
fn run_at_chunk(
    driver: &mut dyn MlxBatchedSeqDriver,
    prompt: &[u32],
    chunk: usize,
    n_gen: usize,
    seq_id: u64,
    baseline: usize,
) -> Result<Run> {
    // SAFETY: single-threaded example, and the value is read by
    // `qwen35_prefill_chunk()` on the next prefill call rather than cached.
    // This is the whole reason the A/B can share one loaded model.
    unsafe { std::env::set_var(CHUNK_ENV, chunk.to_string()) };

    let mut peak = get_active_memory().unwrap_or(0);
    let t0 = Instant::now();
    let (first, mut pos) = driver.prefill(seq_id, prompt)?;
    let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
    peak = peak.max(get_active_memory().unwrap_or(0));

    let mut tokens = vec![first];
    let mut last = first;
    for _ in 1..n_gen {
        let out = driver.decode_step_batch(&[seq_id], &[last], &[pos])?;
        let (tok, p) = out[0];
        tokens.push(tok);
        last = tok;
        pos = p;
    }

    driver.remove_seq(seq_id)?;
    let _ = clear_cache();

    Ok(Run {
        chunk,
        tokens,
        prefill_ms,
        peak: peak.saturating_sub(baseline),
    })
}

fn parse_usize_list(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(|p| {
            p.parse::<usize>()
                .map_err(|_| anyhow!("expected comma-separated positive integers, got {s:?}"))
        })
        .collect()
}

fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args().collect();
    let model_id = std::env::var("MODEL_ID")
        .map_err(|_| anyhow!("set MODEL_ID to a local model directory or an HF repo id"))?;
    let mut chunks = DEFAULT_CHUNKS.to_vec();
    let mut prompt_tokens = 8000usize;
    let mut n_gen = 32usize;

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--chunks" => {
                chunks = parse_usize_list(argv.get(i + 1).map(String::as_str).unwrap_or(""))?;
                i += 2;
            }
            "--prompt-tokens" => {
                prompt_tokens = argv
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| anyhow!("--prompt-tokens needs a positive integer"))?;
                i += 2;
            }
            "--gen" => {
                n_gen = argv
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| anyhow!("--gen needs a positive integer"))?;
                i += 2;
            }
            other => return Err(anyhow!("unknown argument {other:?}")),
        }
    }
    if chunks.len() < 2 {
        return Err(anyhow!("need at least two chunk sizes to compare"));
    }
    if n_gen == 0 {
        return Err(anyhow!("--gen must be at least 1"));
    }

    println!("--- prefill chunk equivalence + cost ---");
    println!("model         = {model_id}");
    println!("prompt target = {prompt_tokens} tokens");
    println!("generated     = {n_gen} tokens (greedy)");
    println!("chunks        = {chunks:?}  (reference = {})", chunks[0]);

    let mut backend = MlxBackend::load(&model_id)?;
    let driver = backend.batched_seq_driver_mut();

    // Prime so the baseline is weights-resident, not zero — MLX loads lazily.
    let warm = build_prompt(driver, 64)?;
    run_at_chunk(driver, &warm, chunks[0], 2, 900_001, 0)?;
    let _ = clear_cache();
    let baseline = get_active_memory().unwrap_or(0);
    println!("baseline      = {:.2} GB\n", baseline as f64 / 1e9);

    let prompt = build_prompt(driver, prompt_tokens)?;
    println!("prompt        = {} tokens\n", prompt.len());

    let mut runs = Vec::new();
    for (k, &chunk) in chunks.iter().enumerate() {
        let run = run_at_chunk(driver, &prompt, chunk, n_gen, 1000 + k as u64, baseline)?;
        println!(
            "  chunk={:<5} prefill={:>8.1} ms  peak={:>7.1} MB  first 8 tokens={:?}",
            run.chunk,
            run.prefill_ms,
            run.peak as f64 / 1e6,
            &run.tokens[..run.tokens.len().min(8)],
        );
        runs.push(run);
    }

    println!("\n--- equivalence vs chunk={} ---", runs[0].chunk);
    let reference = &runs[0];
    let mut all_match = true;
    for run in &runs[1..] {
        match run
            .tokens
            .iter()
            .zip(&reference.tokens)
            .position(|(a, b)| a != b)
        {
            None => println!(
                "  chunk={:<5} IDENTICAL ({} tokens)",
                run.chunk,
                run.tokens.len()
            ),
            Some(at) => {
                all_match = false;
                println!(
                    "  chunk={:<5} DIVERGES at token {at}: {} vs reference {}",
                    run.chunk, run.tokens[at], reference.tokens[at],
                );
            }
        }
    }

    println!("\n--- cost of the chunk ---");
    println!(
        "{:>7}  {:>12}  {:>10}  {:>14}",
        "chunk", "prefill ms", "peak MB", "vs reference"
    );
    for run in &runs {
        println!(
            "{:>7}  {:>12.1}  {:>10.1}  {:>13.1}%",
            run.chunk,
            run.prefill_ms,
            run.peak as f64 / 1e6,
            (run.prefill_ms - reference.prefill_ms) / reference.prefill_ms * 100.0,
        );
    }

    if all_match {
        println!(
            "\nEvery chunk size produced byte-identical output. The chunk is a pure \
             memory/time knob, and `forward_chunked`'s equivalence argument holds on this model."
        );
        Ok(())
    } else {
        Err(anyhow!(
            "chunk size changed the generated tokens — prefill chunking is NOT \
             output-equivalent here, and the default must not be retuned until that is understood"
        ))
    }
}
