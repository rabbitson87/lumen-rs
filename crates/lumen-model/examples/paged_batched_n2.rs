//! Stage 2b-full-γ: paged-backed N=2 test.
//!
//! Exercises `forward_attn_batched_compressed` via the PagedKVBackend. Loads
//! two independent paged models for reference, and one shared paged model
//! for the batched run. Compares each batched row to its reference.
//!
//! Run (requires paged-kv feature + Metal):
//!   MODEL_ID=.models/google_gemma-4-E4B-it-Q4_K_M.gguf \
//!     cargo run --release --features paged-kv -p lumen-model \
//!     --example paged_batched_n2

#![cfg(feature = "paged-kv")]

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_gemma4::ModelWeights;
use lumen_model::paged_kv::PagedKVBackend;

// Gemma 4 E4B config.
const N_LAYERS: u32 = 42;
const N_KV_HEADS: u32 = 2;
const HEAD_DIM_SLIDING: u32 = 256;
const HEAD_DIM_GLOBAL: u32 = 512;
const GLOBAL_EVERY: u32 = 6;

fn load_model_with_paged(path: &str, device: &Device) -> Result<ModelWeights> {
    let mut file = std::fs::File::open(path).with_context(|| format!("open {path}"))?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    let mut model = ModelWeights::from_gguf(content, &mut file, device)?;

    let backend = PagedKVBackend::new(
        device.clone(),
        512, // MB
        16,  // block_size
        N_LAYERS,
        N_KV_HEADS,
        HEAD_DIM_SLIDING,
        HEAD_DIM_GLOBAL,
        GLOBAL_EVERY,
    )?;
    model.set_compressed_kv(Box::new(backend));
    Ok(model)
}

fn prefill_and_decode(
    m: &mut ModelWeights,
    device: &Device,
    seq_id: u64,
    prefill: &[u32],
    next: u32,
) -> Result<Vec<f32>> {
    m.set_current_seq_id(seq_id);
    let pre = Tensor::new(prefill, device)?.unsqueeze(0)?;
    let _ = m.forward(&pre, 0)?;
    let tok = Tensor::new(&[next], device)?.unsqueeze(0)?;
    let logits = m.forward(&tok, prefill.len())?;
    Ok(logits
        .squeeze(0)?
        .to_dtype(candle_core::DType::F32)?
        .to_vec1()?)
}

fn main() -> Result<()> {
    let path = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| ".models/google_gemma-4-E4B-it-Q4_K_M.gguf".into());
    // TQ_THRESHOLD=1 so compressed path activates from the first decode token.
    // SAFETY: single-threaded example, no concurrent env access.
    unsafe {
        std::env::set_var("TQ_THRESHOLD", "1");
    }
    let device = Device::new_metal(0)?;

    let prefill_a: Vec<u32> = vec![2, 105, 2364, 107, 9259, 106, 107];
    let prefill_b: Vec<u32> = vec![2, 105, 2364, 107, 1000, 2000, 3000, 4000];
    let next_a: u32 = 105;
    let next_b: u32 = 42;

    println!("=== Paged reference A ===");
    let mut m_a = load_model_with_paged(&path, &device)?;
    let ref_a = prefill_and_decode(&mut m_a, &device, 0, &prefill_a, next_a)?;
    drop(m_a);

    println!("=== Paged reference B ===");
    let mut m_b = load_model_with_paged(&path, &device)?;
    let ref_b = prefill_and_decode(&mut m_b, &device, 0, &prefill_b, next_b)?;
    drop(m_b);

    println!("=== Paged batched N=2 (v2) ===");
    let mut m_ab = load_model_with_paged(&path, &device)?;
    m_ab.set_current_seq_id(100);
    let pre_a = Tensor::new(prefill_a.as_slice(), &device)?.unsqueeze(0)?;
    let _ = m_ab.forward(&pre_a, 0)?;
    m_ab.set_current_seq_id(200);
    let pre_b = Tensor::new(prefill_b.as_slice(), &device)?.unsqueeze(0)?;
    let _ = m_ab.forward(&pre_b, 0)?;

    let next_tokens = Tensor::new(&[next_a, next_b], &device)?.reshape((2, 1))?;
    let logits_bat = m_ab.forward_batched_decode_v2(
        &next_tokens,
        &[100u64, 200u64],
        &[prefill_a.len(), prefill_b.len()],
    )?;

    let bat_a: Vec<f32> = logits_bat
        .narrow(0, 0, 1)?
        .squeeze(0)?
        .to_dtype(candle_core::DType::F32)?
        .to_vec1()?;
    let bat_b: Vec<f32> = logits_bat
        .narrow(0, 1, 1)?
        .squeeze(0)?
        .to_dtype(candle_core::DType::F32)?
        .to_vec1()?;

    let d_a = ref_a
        .iter()
        .zip(bat_a.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let d_b = ref_b
        .iter()
        .zip(bat_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("vocab={}", ref_a.len());
    println!("  seq A: max|Δ|={:.3e}  ref[0..4]={:?}", d_a, &ref_a[..4]);
    println!("                        bat[0..4]={:?}", &bat_a[..4]);
    println!("  seq B: max|Δ|={:.3e}  ref[0..4]={:?}", d_b, &ref_b[..4]);
    println!("                        bat[0..4]={:?}", &bat_b[..4]);

    // Paged v2 uses batched Metal matmul + batched paged kernel → more
    // tile-order drift than non-paged. Allow 5e-2 abs tolerance.
    let worst = d_a.max(d_b);
    let limit = 5e-2;
    if worst > limit {
        anyhow::bail!("paged batched N=2 drift {worst:.3e} exceeds limit {limit:.0e}");
    }
    println!(
        "OK: paged Stage 2b-full-γ batched attention holds (worst {worst:.3e}, limit {limit:.0e})."
    );
    Ok(())
}
