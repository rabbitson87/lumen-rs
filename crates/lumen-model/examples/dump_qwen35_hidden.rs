//! Runs a single prefill on the same fixed prompt used by
//! `scripts/dump_mlx_hidden.py` and, with `LUMEN_DUMP_HIDDEN=/some/dir` set,
//! writes `embed.bin`, `L00.bin`..`L39.bin`, `final_norm.bin`, `logits.bin` there.
//!
//! Intended to be diffed against the MLX reference to find the first layer
//! whose output diverges.
//!
//! Usage:
//!   LUMEN_QWEN35_SHARDS=... \
//!   LUMEN_DUMP_HIDDEN=/tmp/rust_hidden \
//!   cargo run -p lumen-model --features turboquant-gpu \
//!     --release --example dump_qwen35_hidden

#![cfg(feature = "turboquant-gpu")]

use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{IndexOp, Tensor};
use lumen_metal::mxfp4_gpu::MxFp4Context;
use lumen_model::qwen3_5_moe::backend::Qwen35MoeBackend;

const PROMPT: &str = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n";

fn main() -> Result<()> {
    let dump_dir = std::env::var("LUMEN_DUMP_HIDDEN").unwrap_or_else(|_| "/tmp/rust_hidden".into());
    std::fs::create_dir_all(&dump_dir).with_context(|| format!("mkdir {dump_dir}"))?;
    // Ensure the forward code sees the env var set to this exact value.
    // Safe: single-threaded at this point and the value originates from this process.
    unsafe { std::env::set_var("LUMEN_DUMP_HIDDEN", &dump_dir) };

    let shard_dir = std::env::var("LUMEN_QWEN35_SHARDS")
        .context("LUMEN_QWEN35_SHARDS must point at the model snapshot dir")?;
    let shard_dir = PathBuf::from(shard_dir);

    let model_id =
        std::env::var("MODEL_ID").unwrap_or_else(|_| "mlx-community/Qwen3.6-35B-A3B-mxfp4".into());

    let gpu_ctx = std::sync::Arc::new(MxFp4Context::new().context("Metal context")?);

    let mut backend = Qwen35MoeBackend::load(&model_id, &shard_dir, gpu_ctx)?;

    let ids = backend.encode(PROMPT)?;
    eprintln!("prompt tokens ({}): {:?}", ids.len(), ids);

    // Single prefill. The model's forward_with_offset picks up LUMEN_DUMP_HIDDEN and
    // serializes every intermediate tensor into dump_dir.
    let device = backend.device().clone();
    let prompt_tensor = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
    backend.model_mut().reset_cache();
    let logits = backend.model_mut().forward_with_offset(&prompt_tensor, 0)?;
    eprintln!("logits shape: {:?}", logits.dims());

    // Print top-5 argmax of the last token from the dumped logits for a quick sanity check.
    let last = logits
        .i((.., ids.len() - 1, ..))?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut indexed: Vec<(u32, f32)> = last
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Rust top-5 last-token logits:");
    for (tok, score) in indexed.iter().take(5) {
        let s = backend.decode(&[*tok]).unwrap_or_else(|_| "??".into());
        println!("  {tok:>6}: {s:?}  logit={score:+.4}");
    }
    println!("wrote tensors to {dump_dir}");
    Ok(())
}
