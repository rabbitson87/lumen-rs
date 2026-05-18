//! Same as `dump_qwen35_hidden`, but also runs one decode step after the prefill
//! so we can diff Rust's decode-path hidden states against MLX.
//!
//! Outputs:
//!   /tmp/rust_hidden_prefill/    — embed + L00..L39 + final_norm + logits (prefill)
//!   /tmp/rust_hidden_decode/     — same, for the first decode step (input = argmax token)

#![cfg(feature = "turboquant-gpu")]

use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{IndexOp, Tensor, D};
use lumen_metal::mxfp4_gpu::MxFp4Context;
use lumen_model::qwen3_5_moe::backend::Qwen35MoeBackend;

const PROMPT: &str = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n";

fn set_dump_dir(path: &str) {
    unsafe { std::env::set_var("LUMEN_DUMP_HIDDEN", path) };
}

fn clear_dump_dir() {
    unsafe { std::env::remove_var("LUMEN_DUMP_HIDDEN") };
}

fn main() -> Result<()> {
    let prefill_dir = "/tmp/rust_hidden_prefill";
    let decode_dir = "/tmp/rust_hidden_decode";
    for d in [prefill_dir, decode_dir] {
        std::fs::create_dir_all(d).with_context(|| format!("mkdir {d}"))?;
    }

    let shard_dir = std::env::var("LUMEN_QWEN35_SHARDS")
        .context("LUMEN_QWEN35_SHARDS required")?;
    let shard_dir = PathBuf::from(shard_dir);
    let model_id = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| "mlx-community/Qwen3.6-35B-A3B-mxfp4".into());
    let gpu_ctx = std::sync::Arc::new(MxFp4Context::new()?);

    let mut backend = Qwen35MoeBackend::load(&model_id, &shard_dir, gpu_ctx)?;
    let ids = backend.encode(PROMPT)?;
    eprintln!("prompt tokens ({}): {:?}", ids.len(), ids);

    let device = backend.device().clone();

    // ── Prefill ──
    set_dump_dir(prefill_dir);
    let prompt_tensor = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
    backend.model_mut().reset_cache();
    let logits = backend.model_mut().forward_with_offset(&prompt_tensor, 0)?;
    let last = logits.narrow(D::Minus2, ids.len() - 1, 1)?;
    let last_vec = last.flatten_all()?.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?;
    let mut indexed: Vec<(u32, f32)> = last_vec.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let first_tok = indexed[0].0;
    println!("prefill top-5 (after {} tokens):", ids.len());
    for (tok, score) in indexed.iter().take(5) {
        let s = backend.decode(&[*tok]).unwrap_or_else(|_| "??".into());
        println!("  {tok:>6}: {s:?}  logit={score:+.4}");
    }

    // ── Decode 1 step with input = first_tok ──
    set_dump_dir(decode_dir);
    let dec_input = Tensor::new(&[first_tok], &device)?.unsqueeze(0)?;
    let offset = ids.len();
    let dec_logits = backend.model_mut().forward_with_offset(&dec_input, offset)?;
    // dec_logits shape [1, 1, vocab] — last token is the only one.
    let dec_last = dec_logits.i((.., 0, ..))?.flatten_all()?
        .to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?;
    let mut idx2: Vec<(u32, f32)> = dec_last.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
    idx2.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("decode step 1 top-5 (input tok={first_tok}, offset={offset}):");
    for (tok, score) in idx2.iter().take(5) {
        let s = backend.decode(&[*tok]).unwrap_or_else(|_| "??".into());
        println!("  {tok:>6}: {s:?}  logit={score:+.4}");
    }
    clear_dump_dir();

    println!("wrote prefill to {prefill_dir}, decode to {decode_dir}");
    Ok(())
}
