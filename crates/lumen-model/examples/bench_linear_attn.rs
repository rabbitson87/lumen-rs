//! Decode tok/s bench (Phase A.8-C.5 measurement).
//!
//! Loads `mlx-community/Qwen3.6-35B-A3B-mxfp4`, prefills a short prompt, then
//! runs `--decode-steps` autoregressive decode steps. Reports per-step ms +
//! end-to-end tok/s.
//!
//! Usage:
//!   LUMEN_QWEN35_SHARDS=/path/to/shards \
//!     cargo run --release --example bench_linear_attn \
//!     --features turboquant-gpu -- --decode-steps 50
//!
//! Run with `LUMEN_LINEAR_ATTN_NATIVE=1` to exercise the fused native
//! linear-attn dispatcher; baseline (no env var) uses the Candle path.

#![cfg(feature = "turboquant-gpu")]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{Tensor, D};
use lumen_metal::mxfp4_gpu::MxFp4Context;
use lumen_model::qwen3_5_moe::backend::Qwen35MoeBackend;

const PROMPT: &str =
    "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n";

fn main() -> Result<()> {
    let mut decode_steps = 50;
    let mut prefill_only = false;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--decode-steps" => {
                decode_steps = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(50);
                i += 2;
            }
            "--prefill-only" => {
                prefill_only = true;
                i += 1;
            }
            _ => i += 1,
        }
    }

    let shard_dir = std::env::var("LUMEN_QWEN35_SHARDS")
        .context("LUMEN_QWEN35_SHARDS required")?;
    let shard_dir = PathBuf::from(shard_dir);
    let model_id = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| "mlx-community/Qwen3.6-35B-A3B-mxfp4".into());
    let native_flag = std::env::var("LUMEN_LINEAR_ATTN_NATIVE")
        .map(|v| v == "1")
        .unwrap_or(false);
    eprintln!(
        "[bench] model={model_id} steps={decode_steps} native_linear_attn={native_flag}"
    );

    let gpu_ctx = std::sync::Arc::new(MxFp4Context::new()?);
    let load_start = Instant::now();
    let mut backend = Qwen35MoeBackend::load(&model_id, &shard_dir, gpu_ctx)?;
    eprintln!(
        "[bench] model loaded in {:.1}s",
        load_start.elapsed().as_secs_f64()
    );

    let ids = backend.encode(PROMPT)?;
    let device = backend.device().clone();
    eprintln!("[bench] prompt tokens: {}", ids.len());

    // ── Prefill (warmup) ─────────────────────────────────────────────
    let prompt_tensor = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
    backend.model_mut().reset_cache();
    let prefill_start = Instant::now();
    let logits = backend.model_mut().forward_with_offset(&prompt_tensor, 0)?;
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[bench] prefill {} tokens in {:.1}ms",
        ids.len(),
        prefill_ms
    );

    // Pick first decode token = argmax of last prefill logit.
    let last = logits.narrow(D::Minus2, ids.len() - 1, 1)?;
    let last_vec = last
        .flatten_all()?
        .to_dtype(candle_core::DType::F32)?
        .to_vec1::<f32>()?;
    let mut next_tok = last_vec
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();

    if prefill_only {
        return Ok(());
    }

    // ── Decode loop ──────────────────────────────────────────────────
    let mut step_ms_log: Vec<f64> = Vec::with_capacity(decode_steps);
    let mut offset = ids.len();
    let decode_start = Instant::now();
    for step in 0..decode_steps {
        let t0 = Instant::now();
        let dec_input = Tensor::new(&[next_tok], &device)?.unsqueeze(0)?;
        let dec_logits = backend.model_mut().forward_with_offset(&dec_input, offset)?;
        let dec_last = dec_logits
            .narrow(D::Minus2, 0, 1)?
            .flatten_all()?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1::<f32>()?;
        next_tok = dec_last
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        offset += 1;
        let step_ms = t0.elapsed().as_secs_f64() * 1000.0;
        step_ms_log.push(step_ms);
        if step < 3 || step % 10 == 0 {
            eprintln!("[bench] step {step:>3}: {step_ms:6.1}ms  next_tok={next_tok}");
        }
    }
    let total_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    let mean_ms = step_ms_log.iter().sum::<f64>() / step_ms_log.len() as f64;
    let median_ms = {
        let mut v = step_ms_log.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let tok_per_sec = decode_steps as f64 / (total_ms / 1000.0);

    eprintln!("──────────────────────────────────────────────");
    eprintln!("[bench] decode summary ({} steps):", decode_steps);
    eprintln!("  mean  step: {mean_ms:6.1}ms");
    eprintln!("  median step: {median_ms:6.1}ms");
    eprintln!("  total: {total_ms:.1}ms");
    eprintln!("  tok/s: {tok_per_sec:.2}");
    eprintln!("  native_linear_attn={native_flag}");
    Ok(())
}
