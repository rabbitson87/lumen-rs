//! Continuous-batching throughput bench for Qwen3.5-MoE.
//!
//! Prefills N sequences with synthetic token prompts, then runs
//! `decode_step_batch` for DECODE_STEPS iterations measuring:
//!   - per-step wall-clock latency
//!   - aggregate tokens/s (N tokens produced per step)
//!   - per-seq tokens/s (aggregate / N)
//!
//! Compares N=1 vs N=K to quantify CB overhead ratio.
//!
//! Run:
//!   MODEL_ID=mlx-community/Qwen3.6-35B-A3B-mxfp4 \
//!   LUMEN_QWEN35_SHARDS=/path/to/shards \
//!   N=2 DECODE_STEPS=64 WARMUP=4 \
//!     cargo run --release --features turboquant-gpu \
//!     -p lumen-model --example bench_cb_qwen35
//!
//! Env:
//!   N             batch size (default 1)
//!   DECODE_STEPS  decode iterations (default 64)
//!   WARMUP        warm-up steps before timing (default 4)
//!   PROMPT_LEN    synthetic prompt length per seq (default 8)

#![cfg(feature = "turboquant-gpu")]

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use lumen_metal::mxfp4_gpu::MxFp4Context;
use lumen_model::qwen3_5_moe::backend::Qwen35MoeBackend;

/// Anti-pattern #36 mitigation: surface perf-gate env state at bench start
/// so the log records which fusion / dispatch paths were active during the
/// measurement. The Candle Qwen3.5-MoE path uses DISABLE_* opt-out style
/// (unset = fusion active) and ENABLE_* opt-in style (unset = inactive).
fn log_lumen_env_state() {
    const DEFAULT_ON_OPT_OUTS: &[&str] = &[
        "LUMEN_DISABLE_FLASH_ATTN",
        "LUMEN_DISABLE_RESIDUAL_FUSION",
        "LUMEN_DISABLE_RMSNORM_FUSION",
        "LUMEN_DISABLE_INPUT_RMSNORM_FUSION",
        "LUMEN_DISABLE_DENSE_MLP_RESIDUAL_FUSION",
        "LUMEN_DISABLE_MOE_GATE_UP_SILU_MUL_FUSION",
        "LUMEN_DISABLE_MOE_WSUM_FUSION",
        "LUMEN_DISABLE_MOE_Y_POOL",
        "LUMEN_DISABLE_QKNORM_ROPE",
        "LUMEN_DISABLE_ROPE_CACHE",
    ];
    const DEFAULT_OFF_OPT_INS: &[&str] = &[
        "LUMEN_ENABLE_GATE_UP_SILU_MUL_FUSION",
        "LUMEN_ENABLE_INPUT_RMSNORM_FUSION",
        "LUMEN_ENABLE_MOE_BF16_CHAIN",
        "LUMEN_ENABLE_MOE_MATMUL_WSUM_FUSION",
        "LUMEN_ENABLE_ROUTING_TOPK_FUSION",
        "LUMEN_ENABLE_DENSE_POST_ATTN_FUSION",
        "LUMEN_BF16_RESIDUAL",
        "LUMEN_BF16_RMSNORM",
        "LUMEN_BF16_OUT",
    ];
    for var in DEFAULT_ON_OPT_OUTS {
        match std::env::var(var) {
            Ok(v) => eprintln!(
                "[cb-qwen35 env] {var}={v} (explicit; fusion {})",
                if v == "1" { "DISABLED" } else { "active" }
            ),
            Err(_) => eprintln!("[cb-qwen35 env] {var}=(unset → fusion active)"),
        }
    }
    for var in DEFAULT_OFF_OPT_INS {
        match std::env::var(var) {
            Ok(v) => eprintln!("[cb-qwen35 env] {var}={v} (explicit)"),
            Err(_) => eprintln!("[cb-qwen35 env] {var}=(unset → off)"),
        }
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn run_bench(
    backend: &mut Qwen35MoeBackend,
    n: usize,
    decode_steps: usize,
    warmup: usize,
    prompt_len: usize,
) -> Result<()> {
    println!("\n--- N={n}, decode_steps={decode_steps}, prompt_len={prompt_len} ---");

    // Synthetic prompts: each sequence gets a distinct token stream.
    // Token IDs are in [10, 200] (safe vocab range for any tokenizer).
    let seq_ids: Vec<u64> = (1..=n as u64).collect();
    let prompts: Vec<Vec<u32>> = (0..n)
        .map(|i| {
            (0..prompt_len)
                .map(|j| 10 + ((i * prompt_len + j) % 190) as u32)
                .collect()
        })
        .collect();

    // Prefill all sequences.
    let t_prefill = std::time::Instant::now();
    let mut last_tokens: Vec<u32> = Vec::with_capacity(n);
    let mut positions: Vec<usize> = Vec::with_capacity(n);
    for (i, &seq_id) in seq_ids.iter().enumerate() {
        backend.init_sequence(seq_id);
        let (first_tok, pos) = backend
            .prefill_sequence(seq_id, &prompts[i])
            .with_context(|| format!("prefill seq {seq_id}"))?;
        last_tokens.push(first_tok);
        positions.push(pos);
    }
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    let total_prefill_toks: usize = prompts.iter().map(|p| p.len()).sum();
    println!(
        "prefill: {total_prefill_toks} tokens across {n} seqs in {prefill_ms:.0}ms  ({:.1} tok/s)",
        total_prefill_toks as f64 / (prefill_ms / 1000.0)
    );

    // Warm-up (not timed).
    for _ in 0..warmup {
        let logits = backend.decode_step_batch(&seq_ids, &last_tokens, &positions)?;
        let flat: Vec<f32> = logits
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let vocab = flat.len() / n;
        for i in 0..n {
            last_tokens[i] = argmax(&flat[i * vocab..(i + 1) * vocab]);
            positions[i] += 1;
        }
    }

    // Timed decode.
    let mut step_latencies: Vec<f64> = Vec::with_capacity(decode_steps);
    for _ in 0..decode_steps {
        let t0 = std::time::Instant::now();
        let logits = backend.decode_step_batch(&seq_ids, &last_tokens, &positions)?;
        let flat: Vec<f32> = logits
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        step_latencies.push(elapsed_ms);

        let vocab = flat.len() / n;
        for i in 0..n {
            last_tokens[i] = argmax(&flat[i * vocab..(i + 1) * vocab]);
            positions[i] += 1;
        }
    }

    // Clean up.
    for &seq_id in &seq_ids {
        backend.remove_sequence(seq_id);
    }

    // Stats.
    let total_ms: f64 = step_latencies.iter().sum();
    let mean_ms = total_ms / decode_steps as f64;
    step_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = step_latencies[decode_steps / 2];
    let p95 = step_latencies[(decode_steps as f64 * 0.95) as usize];
    let total_tokens = decode_steps * n;
    let agg_tps = total_tokens as f64 / (total_ms / 1000.0);
    let per_seq_tps = decode_steps as f64 / (total_ms / 1000.0);

    println!("decode: {decode_steps} steps × N={n} = {total_tokens} tokens in {total_ms:.0}ms");
    println!("  step latency:  mean={mean_ms:.1}ms  p50={p50:.1}ms  p95={p95:.1}ms");
    println!("  aggregate tps: {agg_tps:.1} tok/s");
    println!("  per-seq tps:   {per_seq_tps:.1} tok/s");

    Ok(())
}

fn main() -> Result<()> {
    let model_id =
        std::env::var("MODEL_ID").unwrap_or_else(|_| "mlx-community/Qwen3.6-35B-A3B-mxfp4".into());
    let shard_dir = PathBuf::from(
        std::env::var("LUMEN_QWEN35_SHARDS")
            .context("LUMEN_QWEN35_SHARDS must point to the shard directory")?,
    );
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let decode_steps: usize = std::env::var("DECODE_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let warmup: usize = std::env::var("WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let prompt_len: usize = std::env::var("PROMPT_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    // Anti-pattern #36 mitigation: dump perf-gate env state before timing
    // so a later log reader can tell whether canonical fusion paths were
    // active. The Candle Qwen3.5 path defaults most fusions ON via
    // `DISABLE_*` opt-outs (= unset means active); the `ENABLE_*` family
    // is opt-in default-off.
    log_lumen_env_state();

    println!("Loading {model_id} from {}", shard_dir.display());
    let gpu_ctx = Arc::new(MxFp4Context::new()?);
    let mut backend = Qwen35MoeBackend::load(&model_id, &shard_dir, gpu_ctx)
        .context("Failed to load Qwen35MoeBackend")?;
    println!("Model loaded.");

    // N=1 baseline first, then requested N.
    run_bench(&mut backend, 1, decode_steps, warmup, prompt_len)?;
    if n > 1 {
        run_bench(&mut backend, n, decode_steps, warmup, prompt_len)?;

        // Print ratio summary.
        println!("\nNote: N={n} runs the same GPU compute N× sequentially per step.");
        println!(
            "Per-seq throughput degradation = CB scheduling overhead (queue drain, CPU sampling)."
        );
    }

    Ok(())
}
