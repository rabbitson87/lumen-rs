//! Two seqs prefilled, then FULL single-seq forward decoded in turn
//! (no per-layer interleave). If THIS passes but paged_batched_n2 fails,
//! the bug is specifically in layer-by-layer seq interleaving.

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

fn solo(path: &str, device: &Device, seq_id: u64, prefill: &[u32], next: u32) -> Result<Vec<f32>> {
    let mut m = load(path, device)?;
    m.set_current_seq_id(seq_id);
    let pre = Tensor::new(prefill, device)?.unsqueeze(0)?;
    let _ = m.forward(&pre, 0)?;
    let tok = Tensor::new(&[next], device)?.unsqueeze(0)?;
    let l = m.forward(&tok, prefill.len())?;
    Ok(l.squeeze(0)?.to_dtype(candle_core::DType::F32)?.to_vec1()?)
}

fn main() -> Result<()> {
    unsafe {
        std::env::set_var("TQ_THRESHOLD", "1");
    }
    let path = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| ".models/google_gemma-4-E4B-it-Q4_K_M.gguf".into());
    let device = Device::new_metal(0)?;
    let pa: Vec<u32> = vec![2, 105, 2364, 107, 9259, 106, 107];
    let pb: Vec<u32> = vec![2, 105, 2364, 107, 1000, 2000, 3000, 4000];
    let na: u32 = 105;
    let nb: u32 = 42;

    // References (fresh model each).
    let ra = solo(&path, &device, 0, &pa, na)?;
    let rb = solo(&path, &device, 0, &pb, nb)?;

    // Shared model: prefill both under different seq_ids, then decode each
    // via full forward (no batched, no layer interleave).
    let mut m = load(&path, &device)?;
    m.set_current_seq_id(100);
    let _ = m.forward(&Tensor::new(pa.as_slice(), &device)?.unsqueeze(0)?, 0)?;
    m.set_current_seq_id(200);
    let _ = m.forward(&Tensor::new(pb.as_slice(), &device)?.unsqueeze(0)?, 0)?;

    // Decode seq A (switch back).
    m.set_current_seq_id(100);
    let la = m.forward(&Tensor::new(&[na], &device)?.unsqueeze(0)?, pa.len())?;
    let va: Vec<f32> = la
        .squeeze(0)?
        .to_dtype(candle_core::DType::F32)?
        .to_vec1()?;

    // Decode seq B.
    m.set_current_seq_id(200);
    let lb = m.forward(&Tensor::new(&[nb], &device)?.unsqueeze(0)?, pb.len())?;
    let vb: Vec<f32> = lb
        .squeeze(0)?
        .to_dtype(candle_core::DType::F32)?
        .to_vec1()?;

    let da = ra
        .iter()
        .zip(va.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let db = rb
        .iter()
        .zip(vb.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("seq A max|Δ|={da:.3e}  ref[0..4]={:?}", &ra[..4]);
    println!("                       got[0..4]={:?}", &va[..4]);
    println!("seq B max|Δ|={db:.3e}  ref[0..4]={:?}", &rb[..4]);
    println!("                       got[0..4]={:?}", &vb[..4]);
    if da.max(db) > 1e-3 {
        anyhow::bail!("two-seq sequential-decode plumbing diverges");
    }
    println!("OK: two-seq sequential decode holds.");
    Ok(())
}
