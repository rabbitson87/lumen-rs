//! Stage 1.5 smoke: exercises `ModelWeights::forward_batched_decode` with N=1
//! and compares logits against a parallel `forward` call on a second model
//! instance loaded from the same GGUF. Should produce bit-identical outputs
//! (same code path underneath) — failure indicates a regression in the
//! batched skeleton.
//!
//! Run: `MODEL_ID=.models/google_gemma-4-E4B-it-Q4_K_M.gguf \
//!       cargo run --release -p lumen-model --example batched_decode_smoke`

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_gemma4::ModelWeights;

fn load_model(path: &str, device: &Device) -> Result<ModelWeights> {
    let mut file = std::fs::File::open(path).with_context(|| format!("open {path}"))?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    let model = ModelWeights::from_gguf(content, &mut file, device)?;
    Ok(model)
}

fn main() -> Result<()> {
    let path = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| ".models/google_gemma-4-E4B-it-Q4_K_M.gguf".into());
    let device = Device::new_metal(0)?;

    let mut m_ref = load_model(&path, &device)?;
    let mut m_bat = load_model(&path, &device)?;

    // Tiny prefill so we have a non-empty KV state.
    let prefill: Vec<u32> = vec![2, 105, 2364, 107, 9259, 106, 107];
    let pre_tensor = Tensor::new(prefill.as_slice(), &device)?.unsqueeze(0)?;

    let _ = m_ref.forward(&pre_tensor, 0)?;
    let _ = m_bat.forward(&pre_tensor, 0)?;

    let offset = prefill.len();
    let next_tok: u32 = 105;

    // Reference: regular forward with [1, 1] token.
    let tok = Tensor::new(&[next_tok], &device)?.unsqueeze(0)?; // [1, 1]
    let logits_ref = m_ref.forward(&tok, offset)?; // [1, vocab]
    let v_ref: Vec<f32> = logits_ref
        .squeeze(0)?
        .to_dtype(candle_core::DType::F32)?
        .to_vec1()?;

    // Batched N=1.
    let tok_bat = Tensor::new(&[next_tok], &device)?.unsqueeze(0)?; // [1, 1]
    let logits_bat = m_bat.forward_batched_decode(&tok_bat, &[0u64], &[offset])?; // [1, vocab]
    let v_bat: Vec<f32> = logits_bat
        .squeeze(0)?
        .to_dtype(candle_core::DType::F32)?
        .to_vec1()?;

    assert_eq!(v_ref.len(), v_bat.len());
    let mut max_abs = 0.0f32;
    for (a, b) in v_ref.iter().zip(v_bat.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    println!(
        "vocab={} max|Δ|={:.3e} ref[0..4]={:?} bat[0..4]={:?}",
        v_ref.len(),
        max_abs,
        &v_ref[..4],
        &v_bat[..4]
    );

    if max_abs > 1e-3 {
        anyhow::bail!("batched N=1 logits diverge from forward: max|Δ|={max_abs}");
    }
    println!("OK: Stage 1.5 N=1 equivalence holds.");
    Ok(())
}
