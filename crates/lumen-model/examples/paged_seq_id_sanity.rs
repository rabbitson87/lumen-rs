//! Sanity: does a single paged seq with a non-zero seq_id match seq_id=0?
//!
//! Load two identical paged models. On A: run everything under seq_id=0.
//! On B: run everything under seq_id=42. Compare logits of one decode token.
//! If they diverge, the multi-seq plumbing has a bug independent of N>1.

#![cfg(feature = "paged-kv")]

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_gemma4::ModelWeights;
use lumen_model::paged_kv::PagedKVBackend;

fn load(path: &str, device: &Device) -> Result<ModelWeights> {
    let mut file = std::fs::File::open(path).with_context(|| format!("open {path}"))?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    let mut model = ModelWeights::from_gguf(content, &mut file, device)?;
    let backend = PagedKVBackend::new(device.clone(), 512, 16, 42, 2, 256, 512, 6)?;
    model.set_compressed_kv(Box::new(backend));
    Ok(model)
}

fn run(
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
    unsafe {
        std::env::set_var("TQ_THRESHOLD", "1");
    }
    let path = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| ".models/google_gemma-4-E4B-it-Q4_K_M.gguf".into());
    let device = Device::new_metal(0)?;
    let prefill: Vec<u32> = vec![2, 105, 2364, 107, 9259, 106, 107];
    let next: u32 = 105;

    let mut m0 = load(&path, &device)?;
    let r0 = run(&mut m0, &device, 0, &prefill, next)?;

    let mut m42 = load(&path, &device)?;
    let r42 = run(&mut m42, &device, 42, &prefill, next)?;

    let d = r0
        .iter()
        .zip(r42.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("seq_id=0  r[0..4]={:?}", &r0[..4]);
    println!("seq_id=42 r[0..4]={:?}", &r42[..4]);
    println!("max|Δ| = {d:.3e}");
    if d > 1e-3 {
        anyhow::bail!("non-zero seq_id diverges from seq_id=0 on single-seq paged path");
    }
    println!("OK: non-zero seq_id matches seq_id=0.");
    Ok(())
}
