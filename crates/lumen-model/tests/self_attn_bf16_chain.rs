//! Workstream B Phase 3 — bf16 chain forward parity test.
//!
//! Verifies that `SelfAttention::forward` with a bf16 input (LUMEN_BF16_RMSNORM
//! gate ON in production) produces output numerically equivalent to the f32
//! reference within bf16 rounding tolerance.
//!
//! Path exercised:
//!   bf16 input → forward_bf16_in_bf16_out (Dense fallback in this test —
//!   composes f32 matmul + cast) → q_norm/k_norm/transpose/RoPE in bf16 →
//!   flash_attn_bf16 (Workstream C kernel) → bf16 output → cast to f32 at
//!   the o_proj boundary → o_proj f32.
//!
//! Tolerance: relative L2 < 2e-2. The MLX bf16↔f32 reference gap is ~5.5e-3
//! (per `self_attn_fixture.rs` doc); we allow ~4× headroom because the bf16
//! chain widens the rounding surface (multiple ops in bf16 instead of one).
//!
//! Production target: when this passes, the full self_attn forward is safe
//! to enable end-to-end on the bf16 chain.

#![cfg(feature = "turboquant-gpu")]

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, RmsNorm};
use lumen_model::qwen3_5_moe::self_attn::{SelfAttention, SelfAttnDims, SelfAttnRuntime};

fn metal_device() -> Option<Device> {
    Device::new_metal(0).ok()
}

fn random_f32(shape: &[usize], seed: u64, dev: &Device) -> Tensor {
    let n: usize = shape.iter().product();
    let mut s = seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493);
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = ((s >> 32) as u32) as f32 / u32::MAX as f32;
        data.push((bits - 0.5) * 0.1); // small magnitude — keeps softmax stable
    }
    Tensor::from_vec(data, shape, dev).unwrap()
}

fn relative_l2(a: &Tensor, b: &Tensor) -> f32 {
    let a32 = a.to_dtype(DType::F32).unwrap();
    let b32 = b.to_dtype(DType::F32).unwrap();
    let diff = (&a32 - &b32).unwrap();
    let num = diff
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt();
    let den = b32
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt()
        .max(1e-12);
    num / den
}

fn cosine_sim(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
    let b = b.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
    let av = a.to_vec1::<f32>().unwrap();
    let bv = b.to_vec1::<f32>().unwrap();
    let dot: f32 = av.iter().zip(bv.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = av.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = bv.iter().map(|y| y * y).sum::<f32>().sqrt();
    dot / (na * nb + 1e-30)
}

fn build_self_attn(dev: &Device) -> SelfAttention {
    // Qwen3.6-27B Dense layer-3 self_attn shape — same constants as
    // `self_attn_fixture.rs::layer3_dims`.
    let dims = SelfAttnDims {
        hidden_size: 2048,
        num_heads: 16,
        num_kv_heads: 2,
        head_dim: 256,
        attn_output_gate: true,
        rotary_dim: 64,
    };
    let rt = SelfAttnRuntime {
        dims,
        rope_theta: 10_000_000.0,
        rms_norm_eps: 1e-6,
    };

    // QKV combined projection (matches production layout):
    //   q_out = num_heads * head_dim * 2 (gated) = 16 * 256 * 2 = 8192
    //   kv_out = num_kv_heads * head_dim = 2 * 256 = 512
    //   total qkv_out = 8192 + 512 + 512 = 9216
    let qkv_w = random_f32(&[9216, dims.hidden_size], 11, dev) * 0.02f64;
    let o_w = random_f32(&[dims.hidden_size, dims.num_heads * dims.head_dim], 12, dev) * 0.02f64;
    let q_norm_w = Tensor::ones((dims.head_dim,), DType::F32, dev).unwrap();
    let k_norm_w = Tensor::ones((dims.head_dim,), DType::F32, dev).unwrap();

    let qkv = Linear::new(qkv_w.unwrap(), None);
    let o = Linear::new(o_w.unwrap(), None);
    let q_norm = RmsNorm::new(q_norm_w, rt.rms_norm_eps);
    let k_norm = RmsNorm::new(k_norm_w, rt.rms_norm_eps);

    SelfAttention::new(rt, qkv.into(), o.into(), q_norm, k_norm)
}

#[test]
fn bf16_chain_forward_matches_f32_within_tolerance() {
    let dev = match metal_device() {
        Some(d) => d,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    // Same input fed through both paths.
    let x_f32 = random_f32(&[1, 4, 2048], 42, &dev);
    let x_bf16 = x_f32.to_dtype(DType::BF16).unwrap().contiguous().unwrap();

    // Two independent self_attn instances with identical weights — needed
    // because forward(...) mutates per-call state (KV cache, RoPE cache).
    let mut attn_f32 = build_self_attn(&dev);
    let mut attn_bf16 = build_self_attn(&dev);

    // f32 reference path.
    let y_f32 = attn_f32.forward(&x_f32, 0, None).expect("f32 forward");
    assert_eq!(y_f32.dtype(), DType::F32);

    // bf16 chain path (B.3): input bf16 → propagates bf16 through the chain →
    // cast to f32 at o_proj boundary → final output f32.
    let y_b16 = attn_bf16.forward(&x_bf16, 0, None).expect("bf16 forward");
    assert_eq!(
        y_b16.dtype(),
        DType::F32,
        "self_attn must return f32 (residual contract)"
    );

    assert_eq!(y_f32.dims(), y_b16.dims(), "shape parity");

    let cos = cosine_sim(&y_f32, &y_b16);
    let rel_l2 = relative_l2(&y_b16, &y_f32);
    let max_abs = (&y_f32 - &y_b16)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    eprintln!("B.3 bf16 chain parity: cos={cos:.6} rel_L2={rel_l2:.4e} max_abs={max_abs:.4e}");

    assert!(cos > 0.998, "cosine sim {cos} must exceed 0.998");
    assert!(
        rel_l2 < 2e-2,
        "relative L2 {rel_l2:.4e} must be < 2e-2 \
         (~4× MLX bf16↔f32 floor)"
    );
}

/// Determinism check: the bf16 chain itself must be bit-stable on repeat
/// calls — same protocol as Workstream C's flash_attn_bf16 determinism test.
/// This confirms the chain integration didn't introduce non-determinism above
/// the kernel level.
#[test]
fn bf16_chain_determinism_repeat_call() {
    let dev = match metal_device() {
        Some(d) => d,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    let x = random_f32(&[1, 4, 2048], 7, &dev)
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();

    let mut runs: Vec<Vec<f32>> = Vec::with_capacity(5);
    for _ in 0..5 {
        let mut attn = build_self_attn(&dev);
        let y = attn.forward(&x, 0, None).expect("bf16 forward");
        runs.push(y.flatten_all().unwrap().to_vec1::<f32>().unwrap());
    }

    let n = runs[0].len();
    let mut differing = 0usize;
    for i in 1..runs.len() {
        for j in 0..n {
            if runs[0][j].to_bits() != runs[i][j].to_bits() {
                differing += 1;
            }
        }
    }
    eprintln!(
        "B.3 chain determinism: {differing} differing bits over {} comparisons",
        4 * n
    );
    assert_eq!(
        differing, 0,
        "bf16 chain self_attn forward must be bit-deterministic across calls"
    );
}
