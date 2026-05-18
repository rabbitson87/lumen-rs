//! generate_with_opts 경로로 walltime 측정.
//!
//! Decode warm step time을 isolate해서 5.53 tok/s 체크포인트와 비교한다.
//! `LUMEN_MOE_TIMING=1` / `LUMEN_LINEAR_ATTN_TIMING=1` / `LUMEN_MXFP4_KERNEL_VERSION=v2`
//! 등 환경변수 조합을 외부에서 주입해서 회귀 원인을 isolate.
//!
//! 기존 `generate_with_opts`는 `eprintln!`으로 step별 ms를 이미 출력함;
//! 본 probe는 그 출력을 그대로 받고 마지막에 평균/중앙값을 집계한다.
//!
//! Run:
//! ```sh
//! LUMEN_QWEN35_SHARDS="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-mxfp4/snapshots/<sha>" \
//! cargo run --release -p lumen-model --example moe_timing_probe \
//!   --features turboquant-gpu
//! ```

#![cfg(feature = "turboquant-gpu")]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use lumen_metal::mxfp4_gpu::MxFp4Context;
use lumen_model::qwen3_5_moe::backend::Qwen35MoeBackend;

const PROMPT: &str = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n";
const MAX_NEW_TOKENS: usize = 8;

fn main() -> Result<()> {
    let shard_dir = std::env::var("LUMEN_QWEN35_SHARDS")
        .context("LUMEN_QWEN35_SHARDS required")?;
    let shard_dir = PathBuf::from(shard_dir);
    let model_id = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| "mlx-community/Qwen3.6-35B-A3B-mxfp4".into());

    eprintln!(
        "env: MXFP4_VER={} MOE_GROUPED={} MOE_BATCHED={} MOE_LEGACY={}",
        std::env::var("LUMEN_MXFP4_KERNEL_VERSION").unwrap_or_else(|_| "(unset=v1)".into()),
        std::env::var("LUMEN_MOE_GROUPED").unwrap_or_else(|_| "(unset=on)".into()),
        std::env::var("LUMEN_MOE_BATCHED").unwrap_or_else(|_| "(unset=on)".into()),
        std::env::var("LUMEN_MOE_LEGACY").unwrap_or_else(|_| "(unset=off)".into()),
    );

    let gpu_ctx = std::sync::Arc::new(MxFp4Context::new()?);
    eprintln!("MXFP4 v2 active: {}", gpu_ctx.uses_v2());

    eprintln!("loading {model_id} from {}", shard_dir.display());
    let mut backend = Qwen35MoeBackend::load(&model_id, &shard_dir, gpu_ctx)?;
    let ids = backend.encode(PROMPT)?;
    eprintln!("prompt tokens ({}): {:?}", ids.len(), ids);

    eprintln!("\n=== generate_with_opts (max_new={MAX_NEW_TOKENS}) ===");
    let t0 = Instant::now();
    // Greedy fast path: temperature=0 + top_k=0 + repeat_penalty=1.0 → GPU argmax.
    // Set `LUMEN_GREEDY=0` (or any non-empty other value) to fall back to CPU sampler.
    let greedy = std::env::var("LUMEN_GREEDY")
        .map(|v| v != "0")
        .unwrap_or(true);
    let (top_k, repeat_penalty) = if greedy { (0usize, 1.0f32) } else { (20usize, 1.1f32) };
    eprintln!("greedy={greedy} (top_k={top_k}, repeat_penalty={repeat_penalty})");
    let out = backend.generate_with_opts(&ids, MAX_NEW_TOKENS, 0.0, 1.0, top_k, repeat_penalty)?;
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "generated {} tokens in {:.0} ms total ({:.2} tok/s wallclock incl prefill)",
        out.len(),
        total_ms,
        out.len() as f64 / (total_ms / 1000.0)
    );
    Ok(())
}
