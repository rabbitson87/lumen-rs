//! Track A2.4 A/B measurement — N-gram K=2 spec decode vs baseline.
//!
//! Loads model ONCE, then runs the same prompt twice:
//!   (1) baseline: LUMEN_MLX_SPEC unset → `chat_streaming` non-spec path
//!   (2) spec-on: LUMEN_MLX_SPEC=ngram → `chat_streaming_spec_ngram`
//!
//! Reports for each: token sequence, tok/s. Then computes:
//!   - tok/s ratio
//!   - bit-identical match rate (strict gate, expected to FAIL per A2.0)
//!   - top-1 match rate at common-prefix length (relaxed gate)
//!   - first divergence step
//!
//! The bench accepts a single chat-style user message and runs greedy decode.
//!
//! Usage:
//!   USE_MLX=1 cargo run --release -p lumen-mlx --example bench_spec_ab -- \
//!     --model mlx-community/Qwen3.6-35B-A3B-mxfp4 --max-tokens 64

use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Result, anyhow};
use lumen_mlx::MlxBackend;

fn parse_args() -> (String, String, usize) {
    let mut model =
        std::env::var("MODEL_ID").unwrap_or_else(|_| "mlx-community/Qwen3.6-35B-A3B-mxfp4".into());
    let mut max_tokens = 64usize;
    let mut prompt = String::from(
        "List five interesting facts about the history of computing, with one short sentence each.",
    );
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model = args[i + 1].clone();
                i += 2;
            }
            "--max-tokens" => {
                max_tokens = args[i + 1].parse().unwrap_or(64);
                i += 2;
            }
            "--prompt" => {
                prompt = args[i + 1].clone();
                i += 2;
            }
            _ => i += 1,
        }
    }
    (model, prompt, max_tokens)
}

fn run_one(
    backend: &Mutex<MlxBackend>,
    label: &str,
    user_prompt: &str,
    max_tokens: usize,
    spec_on: bool,
) -> Result<(Vec<u32>, f64, f64, String)> {
    // SAFETY: this bench is single-threaded; we toggle env between runs.
    unsafe {
        if spec_on {
            std::env::set_var("LUMEN_MLX_SPEC", "ngram");
        } else {
            std::env::remove_var("LUMEN_MLX_SPEC");
        }
    }

    let messages = vec![("user".to_string(), user_prompt.to_string())];

    let mut be = backend.lock().unwrap();
    let qwen = be
        .as_qwen35_mut()
        .ok_or_else(|| anyhow!("bench requires Qwen35-family backend"))?;
    let seq_id = qwen.alloc_seq_id();
    let t_total = Instant::now();
    // Inner per-call prefill time is logged by the backend itself; we
    // approximate decode-only time by subtracting the eprintln-traced prefill
    // ms from the total. Simpler approach: call prefill manually first.
    let _t_prefill = Instant::now();
    // Mirror chat_streaming's prompt builder (we don't expose it directly).
    // Instead, just do the full chat_streaming and compute decode-only via
    // (total - prefill_ms_observed). For this bench we use the "cheap" trick:
    // run twice (warmup + measure) so prefill is amortized; report total tok/s.
    let text = qwen.chat_streaming(&messages, max_tokens, false, seq_id, |_| {})?;
    let total_elapsed = t_total.elapsed().as_secs_f64();
    drop(be);

    // We don't have direct access to the token sequence — re-encode the text.
    let be = backend.lock().unwrap();
    let tokens = be.encode(&text)?;
    drop(be);

    let total_tok_s = tokens.len() as f64 / total_elapsed;
    eprintln!(
        "[{label}] tokens={} total_elapsed={:.2}s total_tok/s={:.2}",
        tokens.len(),
        total_elapsed,
        total_tok_s
    );
    Ok((tokens, total_tok_s, total_elapsed, text))
}

fn main() -> Result<()> {
    let (model, prompt, max_tokens) = parse_args();
    eprintln!("--- Track A2.4 spec decode A/B ---");
    eprintln!("model      = {model}");
    eprintln!("max_tokens = {max_tokens}");
    eprintln!("prompt     = {prompt:?}");

    let backend = MlxBackend::load(&model)?;
    let backend = Mutex::new(backend);

    // Warm up: discard first run so prefill cold-start (model file paging,
    // Metal kernel JIT) doesn't poison the baseline measurement.
    eprintln!("\n[warmup] (discarded)...");
    let _ = run_one(&backend, "warmup", &prompt, 16, false)?;

    eprintln!("\n[run 1] baseline (no spec)...");
    let (tokens_a, tps_a, _ela_a, text_a) =
        run_one(&backend, "baseline", &prompt, max_tokens, false)?;

    eprintln!("\n[run 2] spec-on (LUMEN_MLX_SPEC=ngram)...");
    let (tokens_b, tps_b, _ela_b, text_b) =
        run_one(&backend, "spec   ", &prompt, max_tokens, true)?;

    // Reverse order: run spec first, then baseline — averages out thermal/
    // Metal-cache warmup asymmetry between A vs B.
    eprintln!("\n[run 3] spec-on (reverse)...");
    let (_tokens_c, tps_c, _ela_c, _text_c) =
        run_one(&backend, "spec  R", &prompt, max_tokens, true)?;

    eprintln!("\n[run 4] baseline (reverse)...");
    let (_tokens_d, tps_d, _ela_d, _text_d) =
        run_one(&backend, "base  R", &prompt, max_tokens, false)?;

    let baseline_avg = (tps_a + tps_d) / 2.0;
    let spec_avg = (tps_b + tps_c) / 2.0;

    // ── Comparison ──
    let common = tokens_a
        .iter()
        .zip(tokens_b.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let n_a = tokens_a.len();
    let n_b = tokens_b.len();
    let n_min = n_a.min(n_b);
    let bit_ident_match_rate = if n_min == 0 {
        0.0
    } else {
        let matches = tokens_a
            .iter()
            .zip(tokens_b.iter())
            .filter(|(a, b)| a == b)
            .count();
        matches as f64 / n_min as f64
    };

    println!("\n--- comparison ---");
    println!("baseline tok/s (run 1) = {tps_a:.2}");
    println!("baseline tok/s (run 4) = {tps_d:.2}");
    println!("baseline avg           = {baseline_avg:.2}");
    println!("spec-on  tok/s (run 2) = {tps_b:.2}");
    println!("spec-on  tok/s (run 3) = {tps_c:.2}");
    println!("spec-on  avg           = {spec_avg:.2}");
    println!(
        "ratio (spec_avg / baseline_avg) = {:.3}×",
        spec_avg / baseline_avg.max(1e-9)
    );
    println!("baseline tokens = {n_a}");
    println!("spec-on  tokens = {n_b}");
    println!("common prefix   = {common} tokens");
    println!(
        "bit-ident match = {:.1}% over {n_min} positions (relaxed gate cares about top-1; \
         strict baseline drift expected per A2.0)",
        bit_ident_match_rate * 100.0
    );
    println!("\n--- baseline text ---\n{text_a}");
    println!("\n--- spec-on text ---\n{text_b}");

    Ok(())
}
