//! Batched decode throughput benchmark.
//!
//! Prefills K sequences with distinct prompts, then runs `forward_batched_decode_v2`
//! for DECODE_STEPS iterations, measuring tokens/sec across the batch.
//!
//! Run:
//!   MODEL_ID=.models/google_gemma-4-E4B-it-Q4_K_M.gguf \
//!   N=4 DECODE_STEPS=64 \
//!     cargo run --release --features paged-kv -p lumen-model \
//!     --example batched_throughput_bench

#![cfg(feature = "paged-kv")]

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_gemma4::ModelWeights;
use lumen_model::paged_kv::PagedKVBackend;

const N_LAYERS: u32 = 42;
const N_KV_HEADS: u32 = 2;
const HEAD_DIM_SLIDING: u32 = 256;
const HEAD_DIM_GLOBAL: u32 = 512;
const GLOBAL_EVERY: u32 = 6;

fn load(path: &str, device: &Device) -> Result<ModelWeights> {
    let mut file = std::fs::File::open(path).with_context(|| format!("open {path}"))?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    let mut model = ModelWeights::from_gguf(content, &mut file, device)?;
    let backend = PagedKVBackend::new(
        device.clone(),
        512,
        16,
        N_LAYERS,
        N_KV_HEADS,
        HEAD_DIM_SLIDING,
        HEAD_DIM_GLOBAL,
        GLOBAL_EVERY,
    )?;
    model.set_compressed_kv(Box::new(backend));
    Ok(model)
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best as u32
}

fn main() -> Result<()> {
    unsafe {
        std::env::set_var("TQ_THRESHOLD", "1");
    }
    let path = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| ".models/google_gemma-4-E4B-it-Q4_K_M.gguf".into());
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let decode_steps: usize = std::env::var("DECODE_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    let device = Device::new_metal(0)?;
    println!("Config: N={n}, decode_steps={decode_steps}");

    // K distinct prompts (simple variations on a base).
    let base: Vec<u32> = vec![2, 105, 2364, 107, 9259, 106, 107];
    let prompts: Vec<Vec<u32>> = (0..n)
        .map(|i| {
            let mut p = base.clone();
            p.push(100 + i as u32);
            p
        })
        .collect();

    let mut model = load(&path, &device)?;

    // Prefill each seq independently.
    let seq_ids: Vec<u64> = (100..100 + n as u64).collect();
    let t_prefill = std::time::Instant::now();
    for (i, prompt) in prompts.iter().enumerate() {
        model.set_current_seq_id(seq_ids[i]);
        let tok = Tensor::new(prompt.as_slice(), &device)?.unsqueeze(0)?;
        let _ = model.forward(&tok, 0)?;
    }
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    let total_prefill_toks: usize = prompts.iter().map(|p| p.len()).sum();
    println!(
        "prefill: {} tokens across {} seqs in {:.0}ms ({:.1} tok/s)",
        total_prefill_toks,
        n,
        prefill_ms,
        total_prefill_toks as f64 / (prefill_ms / 1000.0)
    );

    // Seed decode with the last prompt token for each seq.
    let mut last_tokens: Vec<u32> = prompts.iter().map(|p| *p.last().unwrap()).collect();
    let mut positions: Vec<usize> = prompts.iter().map(|p| p.len()).collect();

    // Warmup step.
    let tok_tensor = Tensor::new(last_tokens.as_slice(), &device)?.reshape((n, 1))?;
    let _ = model.forward_batched_decode_v2(&tok_tensor, &seq_ids, &positions)?;

    let t_decode = std::time::Instant::now();
    for step in 0..decode_steps {
        let tok_tensor = Tensor::new(last_tokens.as_slice(), &device)?.reshape((n, 1))?;
        let logits = model.forward_batched_decode_v2(&tok_tensor, &seq_ids, &positions)?;
        // Greedy per seq.
        let logits_cpu: Vec<f32> = logits
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let vocab = logits_cpu.len() / n;
        for i in 0..n {
            let row = &logits_cpu[i * vocab..(i + 1) * vocab];
            last_tokens[i] = argmax(row);
            positions[i] += 1;
        }
        let _ = step;
    }
    let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
    let total_decode_toks = decode_steps * n;
    println!(
        "decode: {} steps × N={} = {} tokens in {:.0}ms",
        decode_steps, n, total_decode_toks, decode_ms
    );
    println!(
        "  per-step latency: {:.1}ms",
        decode_ms / decode_steps as f64
    );
    println!(
        "  aggregate throughput: {:.1} tok/s",
        total_decode_toks as f64 / (decode_ms / 1000.0)
    );
    println!(
        "  per-seq throughput: {:.1} tok/s",
        decode_steps as f64 / (decode_ms / 1000.0)
    );

    Ok(())
}
