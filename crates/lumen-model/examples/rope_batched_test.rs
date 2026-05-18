//! Stage 2b-lite: numerical validation of batched per-seq RoPE.
//!
//! Compares `apply_rotary_emb_qkv_batched` against N separate single-seq
//! `apply_rotary_emb_qkv` calls on the same `[N, h, 1, d]` tensors.
//!
//! Run: `cargo run --release -p lumen-model --example rope_batched_test`

use anyhow::Result;
use candle_core::Device;
use candle_transformers::models::quantized_gemma4::test_rope_batched_equivalence;

fn main() -> Result<()> {
    let device = Device::new_metal(0)
        .or_else(|_| Device::new_cuda(0))
        .unwrap_or(Device::Cpu);
    println!("device = {device:?}");

    // Gemma 4 E4B sliding: head_dim=256, n_head=8, rope_freq=10000.
    let cases: &[(usize, usize, &[usize], f32)] = &[
        (256, 8, &[0], 10_000.0),                    // N=1 sanity
        (256, 8, &[0, 5, 128, 999], 10_000.0),       // N=4 varied
        (256, 8, &[4095, 0, 1, 2048], 10_000.0),     // edges + mid
        (512, 8, &[17, 93, 256, 1024], 1_000_000.0), // global head_dim=512
        (256, 4, &[42, 42, 42, 42], 10_000.0),       // all same pos
    ];

    let mut worst = 0.0f32;
    for (head_dim, n_head, positions, freq) in cases {
        let (dq, dk) =
            test_rope_batched_equivalence(&device, *head_dim, *n_head, positions, *freq)?;
        worst = worst.max(dq.max(dk));
        println!(
            "  head_dim={:>3} n_head={} N={} positions={:?} freq={:>8.0}  max|Δq|={:.3e} max|Δk|={:.3e}",
            head_dim,
            n_head,
            positions.len(),
            positions,
            freq,
            dq,
            dk
        );
    }

    if worst > 1e-4 {
        anyhow::bail!("batched RoPE diverges from per-seq reference: worst max|Δ|={worst:.3e}");
    }
    println!("OK: Stage 2b-lite batched RoPE equivalence holds (worst max|Δ|={worst:.3e}).");
    Ok(())
}
