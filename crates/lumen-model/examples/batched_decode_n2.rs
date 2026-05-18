//! Stage 2b-full-α: N=2 multi-seq KV isolation test.
//!
//! Loads two independent model instances (A, B) and one shared model (AB).
//! A and B each prefill a different prompt and run one decode step.
//! AB prefills both prompts (each under its own seq_id via set_current_seq_id)
//! and runs a single `forward_batched_decode` with both next-tokens.
//!
//! If per-seq KV state works, AB's logits row 0 must match A's logits, and
//! AB's logits row 1 must match B's logits (bit-identical, since the
//! underlying compute path is still sequential per seq inside the loop).
//!
//! Run: `MODEL_ID=.models/google_gemma-4-E4B-it-Q4_K_M.gguf \
//!       cargo run --release -p lumen-model --example batched_decode_n2`

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_gemma4::ModelWeights;

fn load_model(path: &str, device: &Device) -> Result<ModelWeights> {
    let mut file = std::fs::File::open(path).with_context(|| format!("open {path}"))?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    Ok(ModelWeights::from_gguf(content, &mut file, device)?)
}

fn run_single(
    m: &mut ModelWeights,
    device: &Device,
    prefill: &[u32],
    next: u32,
) -> Result<Vec<f32>> {
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
    let device = Device::new_metal(0)?;

    // Two distinct prefill prompts → exercises shared_kv_states per-seq.
    let prefill_a: Vec<u32> = vec![2, 105, 2364, 107, 9259, 106, 107];
    let prefill_b: Vec<u32> = vec![2, 105, 2364, 107, 1000, 2000, 3000, 4000];
    let next_a: u32 = 105;
    let next_b: u32 = 42;

    // Independent references.
    let mut m_a = load_model(&path, &device)?;
    let ref_a = run_single(&mut m_a, &device, &prefill_a, next_a)?;
    drop(m_a);

    let mut m_b = load_model(&path, &device)?;
    let ref_b = run_single(&mut m_b, &device, &prefill_b, next_b)?;
    drop(m_b);

    // Shared model: two seqs under seq_id 100 and 200.
    let mut m_ab = load_model(&path, &device)?;

    // Seq A prefill.
    m_ab.set_current_seq_id(100);
    let pre_a = Tensor::new(prefill_a.as_slice(), &device)?.unsqueeze(0)?;
    let _ = m_ab.forward(&pre_a, 0)?;

    // Seq B prefill.
    m_ab.set_current_seq_id(200);
    let pre_b = Tensor::new(prefill_b.as_slice(), &device)?.unsqueeze(0)?;
    let _ = m_ab.forward(&pre_b, 0)?;

    // Batched decode: [N=2, 1].
    let next_tokens = Tensor::new(&[next_a, next_b], &device)?.reshape((2, 1))?;
    // Use v2 (batched non-attn, per-seq attn) if MODE=v2 else Stage 1.
    let mode = std::env::var("MODE").unwrap_or_else(|_| "v1".to_string());
    let logits_bat = if mode == "v2" {
        println!("=== forward_batched_decode_v2 ===");
        m_ab.forward_batched_decode_v2(
            &next_tokens,
            &[100u64, 200u64],
            &[prefill_a.len(), prefill_b.len()],
        )?
    } else {
        println!("=== forward_batched_decode (Stage 1) ===");
        m_ab.forward_batched_decode(
            &next_tokens,
            &[100u64, 200u64],
            &[prefill_a.len(), prefill_b.len()],
        )?
    };

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

    let worst = d_a.max(d_b);
    // v1 (Stage 1) is bit-identical (per-seq full forward path). v2 uses
    // batched non-attn kernels whose Metal matmul tile ordering differs from
    // b=1, so we tolerate ~1e-2 absolute drift (< 1e-3 relative on typical
    // top logits).
    let limit = if mode == "v2" { 2e-2 } else { 1e-3 };
    if worst > limit {
        anyhow::bail!("mode={mode} max|Δ|={worst:.3e} exceeds limit {limit:.0e}");
    }
    println!(
        "OK [{mode}]: per-seq KV isolation holds (worst max|Δ|={worst:.3e}, limit {limit:.0e})."
    );
    Ok(())
}
