//! bf16 KV storage vs the f32 default — memory, throughput and output, across
//! several operating points.
//!
//! Task 007 measured the full-attention KV cache at ~67 KB per allocated slot
//! on Qwen3.5-9B, matching f32 arithmetic exactly (8 full-attn layers × 4 KV
//! heads × 256 head_dim × 2 for K+V × 4 bytes = 64.0 KB) and ruling out bf16's
//! 32.0 KB. Storing it in bf16 is the largest memory lever left after that task
//! ruled out PagedAttention. Unlike paging, it is **not** free: the cast lands
//! after k_norm and RoPE, so attention runs in bf16 end-to-end and the output
//! can change.
//!
//! ## Why several operating points, not one
//!
//! Twice in one session a single-operating-point measurement produced a
//! confident, wrong recommendation on this exact subsystem: an 8K-only prefill
//! sweep read "chunking is free" when at 20K the same change costs +54 to
//! +118%. Memory savings here scale with cached tokens and batch width, and
//! quality degradation may scale with context length — neither is visible from
//! one point. So the default sweep crosses short/long prompts with single and
//! batched decode, and the summary refuses to give a verdict unless the points
//! agree.
//!
//! ## What is compared
//!
//! Both conditions run against the **same loaded weights in one process** —
//! `set_kv_store_bf16` flips an atomic, so there is no per-process env read to
//! work around and no second model load to confound the memory numbers.
//!
//!   * **First-token match** — a clean single-step signal. Both conditions see
//!     byte-identical inputs, so a divergence here is purely the bf16 rounding,
//!     with no autoregressive cascade mixed in.
//!   * **Sequence match rate** — what a user actually sees under greedy decode.
//!     Expect this to be lower than the first-token signal: once one argmax
//!     flips, everything after it is a different continuation rather than a
//!     worse one, so a low rate is not by itself evidence of bad quality.
//!   * **Resident KV** — active memory over baseline after decode, pool
//!     drained. This is the number the lever exists for; bf16 should roughly
//!     halve it.
//!   * **Decode throughput** — bf16 attention moves half the bytes, so this
//!     should not regress. If it does, the cast is costing more than it saves.
//!
//! ```text
//! MODEL_ID=~/models/Qwen3.5-9B-MTPLX-Speed \
//!   cargo run --release -p lumen-mlx --features mlx-native \
//!   --example kv_bf16_ab
//!
//! # custom operating points, as N:prompt_tokens
//! ... --example kv_bf16_ab -- --points 1:2000,1:8000,4:1500 --gen 48
//! ```

use std::time::Instant;

use anyhow::{Result, anyhow};
use lumen_mlx::metal_memory::{clear_cache, get_active_memory};
use lumen_mlx::{MlxBackend, MlxBatchedSeqDriver, set_kv_store_bf16};

/// `(batch width, prompt tokens)`. Crosses short/long context with single and
/// batched decode so a saving that only exists at one shape cannot masquerade
/// as a general one.
const DEFAULT_POINTS: &[(usize, usize)] = &[(1, 2000), (1, 8000), (8, 800), (8, 2500)];

const WORDS: &[&str] = &[
    "system", "value", "record", "matrix", "signal", "buffer", "window", "kernel", "vector",
    "planet", "garden", "silver", "market", "reason", "letter", "number", "moment", "figure",
    "branch", "circle", "stream", "camera", "ticket", "island", "friend", "orange", "bridge",
    "packet", "column", "target", "sample", "region", "handle", "author", "series", "corner",
    "policy", "device", "格子", "회로", "réseau",
];

fn filler(seq_ix: usize, n_words: usize) -> String {
    let mut state =
        0x9E37_79B9_7F4A_7C15u64 ^ ((seq_ix as u64 + 1).wrapping_mul(0x517C_C1B7_2722_0A95));
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

fn build_prompt(
    driver: &dyn MlxBatchedSeqDriver,
    seq_ix: usize,
    target: usize,
) -> Result<Vec<u32>> {
    let mut words = target.max(8);
    let mut best: Option<Vec<u32>> = None;
    for _ in 0..8 {
        let msgs = vec![(
            "user".to_string(),
            format!("Summarize the following log:\n{}", filler(seq_ix, words)),
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
    best.ok_or_else(|| anyhow!("could not reach {target} tokens for seq {seq_ix}"))
}

struct Cond {
    /// One generated sequence per batch row, in row order.
    tokens: Vec<Vec<u32>>,
    prefill_ms: f64,
    decode_ms: f64,
    resident_kv: usize,
}

/// Active memory once the reusable pool has actually stopped shrinking.
///
/// A single `clear_cache()` is not enough: it frees already-released buffers,
/// and MLX does not always release everything by the time the call returns.
/// Measured, this shows up as a bimodal reading — the *same* condition at the
/// *same* operating point reports either 190.6 MB or 317.9 MB, a fixed ~127 MB
/// quantum that is present or absent. Pairing a clean reading of one condition
/// with a contaminated reading of the other manufactures a 32 KB/slot saving
/// into a 29 KB/slot regression, which is exactly the sort of artifact that
/// reads as a real result.
///
/// Looping until two consecutive readings agree collapses it. The minimum is
/// returned because the contaminant only ever adds.
fn settled_active() -> usize {
    // The wait is load-bearing, not defensive. `LUMEN_NATIVE_DEFER_CLEAR_CACHE`
    // is on by default, so `remove_seq` hands the clear to a background worker
    // (~45 ms). Two back-to-back readings taken before that worker runs are
    // equal to each other and both wrong, which is how a settle loop without a
    // sleep exits early on the contaminated value.
    let mut best = usize::MAX;
    let mut prev = usize::MAX;
    let mut stable = 0;
    for _ in 0..12 {
        let _ = clear_cache();
        let now = get_active_memory().unwrap_or(0);
        best = best.min(now);
        stable = if now == prev { stable + 1 } else { 0 };
        if stable >= 2 {
            break;
        }
        prev = now;
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    best
}

/// Prefill `prompts` as one batch, then greedily decode `n_gen` tokens per row
/// with the bf16 flag pinned to `bf16`. Sequences are released and the pool
/// drained afterwards, so the next condition starts from the same floor.
fn run_condition(
    driver: &mut dyn MlxBatchedSeqDriver,
    prompts: &[Vec<u32>],
    n_gen: usize,
    bf16: bool,
    first_seq_id: u64,
    baseline: usize,
) -> Result<Cond> {
    set_kv_store_bf16(bf16);

    let ids: Vec<u64> = (0..prompts.len() as u64)
        .map(|i| first_seq_id + i)
        .collect();
    let mut last = Vec::with_capacity(ids.len());
    let mut pos = Vec::with_capacity(ids.len());

    let t_prefill = Instant::now();
    for (row, &id) in ids.iter().enumerate() {
        let (tok, p) = driver.prefill(id, &prompts[row])?;
        last.push(tok);
        pos.push(p);
    }
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

    let mut tokens: Vec<Vec<u32>> = last.iter().map(|&t| vec![t]).collect();

    // Settle step, untimed: drops the prefill activations MLX still holds live
    // so the resident number below is KV rather than a leftover graph.
    let settle = driver.decode_step_batch(&ids, &last, &pos)?;
    for (row, (tok, p)) in settle.into_iter().enumerate() {
        tokens[row].push(tok);
        last[row] = tok;
        pos[row] = p;
    }
    let _ = clear_cache();

    let t_decode = Instant::now();
    for _ in 0..n_gen {
        let out = driver.decode_step_batch(&ids, &last, &pos)?;
        for (row, (tok, p)) in out.into_iter().enumerate() {
            tokens[row].push(tok);
            last[row] = tok;
            pos[row] = p;
        }
    }
    let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;

    let resident_kv = settled_active().saturating_sub(baseline);

    for &id in &ids {
        driver.remove_seq(id)?;
    }
    let _ = clear_cache();

    Ok(Cond {
        tokens,
        prefill_ms,
        decode_ms,
        resident_kv,
    })
}

/// `KV_CACHE_STEP`-rounded slot count, the quantity the per-slot cost is
/// charged against. Re-exported from the crate so this does not hard-code 256.
fn alloc_slots(prompts: &[Vec<u32>], n_gen: usize) -> usize {
    prompts
        .iter()
        // +1 for the untimed settle step, which caches a real token.
        .map(|p| {
            (p.len() + 1 + n_gen).div_ceil(lumen_mlx::KV_CACHE_STEP) * lumen_mlx::KV_CACHE_STEP
        })
        .sum()
}

struct Row {
    n: usize,
    prompt_len: usize,
    slots: usize,
    first_token_match: usize,
    seq_match: usize,
    seq_total: usize,
    first_divergence: Option<usize>,
    kv_f32: usize,
    kv_bf16: usize,
    tps_f32: f64,
    tps_bf16: f64,
}

fn parse_points(s: &str) -> Result<Vec<(usize, usize)>> {
    s.split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (n, len) = p
                .split_once(':')
                .ok_or_else(|| anyhow!("expected N:tokens pairs, got {p:?}"))?;
            Ok((
                n.trim()
                    .parse()
                    .map_err(|_| anyhow!("bad batch width in {p:?}"))?,
                len.trim()
                    .parse()
                    .map_err(|_| anyhow!("bad prompt length in {p:?}"))?,
            ))
        })
        .collect()
}

fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args().collect();
    let model_id = std::env::var("MODEL_ID")
        .map_err(|_| anyhow!("set MODEL_ID to a local model directory or an HF repo id"))?;
    let mut points = DEFAULT_POINTS.to_vec();
    let mut n_gen = 32usize;
    // Control mode: run BOTH conditions in f32. Any mismatch it reports is
    // scheduling nondeterminism in MLX, not the dtype — without it, a 97%
    // sequence match reads as bf16 quality loss when it may be the floor.
    let mut control = false;

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--points" => {
                points = parse_points(argv.get(i + 1).map(String::as_str).unwrap_or(""))?;
                i += 2;
            }
            "--gen" => {
                n_gen = argv
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| anyhow!("--gen needs a positive integer"))?;
                i += 2;
            }
            "--control" => {
                control = true;
                i += 1;
            }
            other => return Err(anyhow!("unknown argument {other:?}")),
        }
    }
    if points.is_empty() {
        return Err(anyhow!("--points produced an empty list"));
    }

    println!("--- bf16 KV storage A/B ---");
    println!("model  = {model_id}");
    println!("points = {points:?}  (batch width : prompt tokens)");
    println!("gen    = {n_gen} tokens per sequence, greedy");
    if control {
        println!(
            "MODE   = CONTROL — both conditions run f32. Any mismatch below is MLX \n\
             \x20        nondeterminism, and is the floor the real A/B must be read against."
        );
    }
    println!();

    let mut backend = MlxBackend::load(&model_id)?;
    let driver = backend.batched_seq_driver_mut();

    // Prime: MLX loads weights lazily, so the baseline must be taken after a
    // real forward or the whole weight set lands in the first KV measurement.
    let warm = vec![build_prompt(driver, 0, 64)?];
    run_condition(driver, &warm, 2, false, 900_001, 0)?;
    let baseline = settled_active();
    println!(
        "baseline (weights + runtime) = {:.2} GB\n",
        baseline as f64 / 1e9
    );

    let mut rows = Vec::new();
    let mut next_id: u64 = 1;

    for &(n, prompt_len) in &points {
        let prompts: Vec<Vec<u32>> = (0..n)
            .map(|i| build_prompt(driver, i, prompt_len))
            .collect::<Result<_>>()?;
        let real_len: usize = prompts.iter().map(Vec::len).sum();

        let f32_run = run_condition(driver, &prompts, n_gen, false, next_id, baseline)?;
        next_id += n as u64;
        let bf16_run = run_condition(driver, &prompts, n_gen, !control, next_id, baseline)?;
        next_id += n as u64;

        let first_token_match = f32_run
            .tokens
            .iter()
            .zip(&bf16_run.tokens)
            .filter(|(a, b)| a[0] == b[0])
            .count();
        let mut seq_match = 0usize;
        let mut seq_total = 0usize;
        let mut first_divergence: Option<usize> = None;
        for (a, b) in f32_run.tokens.iter().zip(&bf16_run.tokens) {
            for (k, (x, y)) in a.iter().zip(b).enumerate() {
                seq_total += 1;
                if x == y {
                    seq_match += 1;
                } else if first_divergence.is_none_or(|d| k < d) {
                    first_divergence = Some(k);
                }
            }
        }

        let steps = (n * n_gen) as f64;
        let row = Row {
            n,
            prompt_len: real_len / n,
            slots: alloc_slots(&prompts, n_gen),
            first_token_match,
            seq_match,
            seq_total,
            first_divergence,
            kv_f32: f32_run.resident_kv,
            kv_bf16: bf16_run.resident_kv,
            tps_f32: steps / (f32_run.decode_ms / 1000.0),
            tps_bf16: steps / (bf16_run.decode_ms / 1000.0),
        };

        println!(
            "  N={:<2} prompt≈{:<6} KV {:>7.1} → {:>7.1} MB ({:+.1}%)   decode {:>6.1} → {:>6.1} tok/s ({:+.1}%)   \
             prefill {:>8.1} → {:>8.1} ms",
            row.n,
            row.prompt_len,
            row.kv_f32 as f64 / 1e6,
            row.kv_bf16 as f64 / 1e6,
            (row.kv_bf16 as f64 - row.kv_f32 as f64) / row.kv_f32.max(1) as f64 * 100.0,
            row.tps_f32,
            row.tps_bf16,
            (row.tps_bf16 - row.tps_f32) / row.tps_f32 * 100.0,
            f32_run.prefill_ms,
            bf16_run.prefill_ms,
        );
        println!(
            "        first-token match {}/{}   sequence match {}/{} ({:.0}%)   first divergence at {}",
            row.first_token_match,
            row.n,
            row.seq_match,
            row.seq_total,
            row.seq_match as f64 / row.seq_total.max(1) as f64 * 100.0,
            match row.first_divergence {
                Some(d) => format!("token {d}"),
                None => "never".into(),
            },
        );
        rows.push(row);
    }

    println!("\n--- summary ---");
    println!(
        "{:>3}  {:>8}  {:>10}  {:>10}  {:>9}  {:>11}  {:>11}  {:>9}",
        "N", "prompt", "KV f32 MB", "KV bf16 MB", "saved", "1st-tok", "seq match", "tok/s Δ"
    );
    for r in &rows {
        println!(
            "{:>3}  {:>8}  {:>10.1}  {:>10.1}  {:>8.1}%  {:>7}/{:<3}  {:>10.0}%  {:>8.1}%",
            r.n,
            r.prompt_len,
            r.kv_f32 as f64 / 1e6,
            r.kv_bf16 as f64 / 1e6,
            (r.kv_f32 as f64 - r.kv_bf16 as f64) / r.kv_f32.max(1) as f64 * 100.0,
            r.first_token_match,
            r.n,
            r.seq_match as f64 / r.seq_total.max(1) as f64 * 100.0,
            (r.tps_bf16 - r.tps_f32) / r.tps_f32 * 100.0,
        );
    }

    // Total resident includes a per-sequence, length-independent term (the
    // linear-attention conv/SSM state — ~53 MB/seq on this model per task 007)
    // that no KV dtype change can touch, so the headline percentage is always
    // below 50 and moves with batch width. Bytes saved per allocated slot is
    // the shape-independent quantity: it should land near the difference
    // between the f32 and bf16 per-slot costs, ~33.7 KB.
    println!("\n--- saving per allocated cache slot (shape-independent) ---");
    for r in &rows {
        println!(
            "  N={:<2} prompt≈{:<6} slots={:<7} saved {:>6.1} MB = {:>5.1} KB/slot",
            r.n,
            r.prompt_len,
            r.slots,
            (r.kv_f32 as f64 - r.kv_bf16 as f64) / 1e6,
            (r.kv_f32 as f64 - r.kv_bf16 as f64) / r.slots.max(1) as f64 / 1e3,
        );
    }

    // A verdict is only meaningful if the operating points agree. Reporting the
    // spread rather than an average is the whole reason this sweeps more than
    // one shape.
    let saved: Vec<f64> = rows
        .iter()
        .map(|r| (r.kv_f32 as f64 - r.kv_bf16 as f64) / r.slots.max(1) as f64 / 1e3)
        .collect();
    let lo = saved.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = saved.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let tps: Vec<f64> = rows
        .iter()
        .map(|r| (r.tps_bf16 - r.tps_f32) / r.tps_f32 * 100.0)
        .collect();
    let tlo = tps.iter().cloned().fold(f64::INFINITY, f64::min);
    let thi = tps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let all_first_match = rows.iter().all(|r| r.first_token_match == r.n);

    println!("\n--- verdict ---");
    println!(
        "saved per slot across points: {lo:.1} .. {hi:.1} KB   (expect ~33.7 KB — the f32 minus \
         bf16 per-slot cost — if the cast reaches the cache)"
    );
    println!("decode throughput Δ:          {tlo:+.1}% .. {thi:+.1}%");
    println!(
        "first-token agreement:  {}",
        if all_first_match {
            "every point matched — bf16 rounding did not flip a single first argmax"
        } else {
            "DIVERGED at at least one point — bf16 rounding flips the first argmax, which is the \
             cleanest evidence of real quality loss"
        }
    );
    if hi - lo > 5.0 {
        println!(
            "NOTE: the per-slot saving spans more than 5 KB across operating shapes. It should be \
             a constant of the model, so a spread this wide means something other than the cache \
             dtype is moving — investigate before quoting any single number."
        );
    }
    Ok(())
}
