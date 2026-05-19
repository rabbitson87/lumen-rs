//! Workstream B Phase 4 — bf16 chain forward parity test for `GatedDeltaNet`.
//!
//! Verifies that `GatedDeltaNet::forward` with a bf16 input (LUMEN_BF16_RMSNORM
//! gate ON in production) routed through the bf16 `tq_gated_delta_step_bf16`
//! kernel produces output numerically equivalent to the f32 reference within
//! bf16 rounding tolerance.
//!
//! Path exercised:
//!   bf16 input → forward_bf16_in_bf16_out (Dense fallback in this test —
//!   composes f32 matmul + cast) → conv1d weight cast to bf16 →
//!   q/k/v/g/beta in bf16 → tq_gated_delta_step_bf16 kernel (f32 state,
//!   Escape #3) → bf16 y → RMSNormGated (norm_weight cast to bf16 on the fly,
//!   silu(z)*y in f32 = Escape #1) → bf16 → cast to f32 at o_proj boundary →
//!   o_proj f32.
//!
//! Tolerance: relative L2 < 2e-2. The MLX bf16↔f32 reference gap on the
//! linear-attn block is ≈5.5e-3 (per `linear_attn_fixture.rs` doc); we allow
//! ~4× headroom because the bf16 chain widens the rounding surface.
//!
//! Native path is bypassed via a non-None mask (an all-ones mask is a no-op
//! on the SSM ops but flips the `mask.is_none()` precondition that gates the
//! native path), so the test exercises the GDN kernel + Candle path.

#![cfg(feature = "turboquant-gpu")]

use candle_core::{DType, Device, Tensor};
use candle_nn::Linear;
use lumen_model::qwen3_5_moe::linear_attn::{
    GatedDeltaNet, GatedDeltaNetRuntime, LinearAttnDims, conv1d_from_mlx_weight,
};
use std::sync::Mutex;

/// Serializes both tests in this file: `lumen_metal::gated_delta::set_enabled`
/// flips a process-wide atomic. With cargo's default parallel test scheduler,
/// one test's restore-to-`was_enabled` can land mid-loop in the other test
/// and silently flip the GDN kernel off, producing non-deterministic output.
/// A simple guard mutex is enough — both tests acquire before touching the
/// toggle.
static GDN_TEST_GUARD: Mutex<()> = Mutex::new(());

fn metal_device() -> Option<Device> {
    Device::new_metal(0).ok()
}

fn random_f32(shape: &[usize], seed: u64, scale: f32, dev: &Device) -> Tensor {
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
        data.push((bits - 0.5) * scale);
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

fn dims() -> LinearAttnDims {
    // Shape echoing Qwen3.6-27B Dense layer 0 GatedDeltaNet.
    LinearAttnDims {
        hidden_size: 2048,
        num_k_heads: 16,
        num_v_heads: 32,
        head_dim: 128,
        conv_kernel: 4,
    }
}

fn runtime() -> GatedDeltaNetRuntime {
    GatedDeltaNetRuntime {
        dims: dims(),
        rms_norm_eps: 1e-6,
    }
}

fn build_net(dev: &Device) -> GatedDeltaNet {
    let d = dims();
    let qkv_dim = d.qkv_dim();
    let v_dim = d.v_dim();
    let hv = d.num_v_heads;
    let combined_out = qkv_dim + v_dim + hv + hv;

    let in_proj_w = random_f32(&[combined_out, d.hidden_size], 11, 0.02, dev);
    let in_proj_combined = Linear::new(in_proj_w, None);

    let conv_w = random_f32(&[d.qkv_dim(), d.conv_kernel, 1], 12, 0.1, dev);
    let conv1d = conv1d_from_mlx_weight(conv_w, d.conv_kernel).unwrap();

    // A_log: small negative values so exp(-exp(A_log)*softplus(...)) decays gently.
    let a_log = random_f32(&[d.num_v_heads], 13, 0.5, dev)
        .affine(1.0, -1.0)
        .unwrap();
    let dt_bias = random_f32(&[d.num_v_heads], 14, 0.1, dev);
    let norm_w = Tensor::ones((d.head_dim,), DType::F32, dev).unwrap();
    let out_proj_w = random_f32(&[d.hidden_size, d.v_dim()], 15, 0.02, dev);
    let out_proj = Linear::new(out_proj_w, None);

    GatedDeltaNet::new(
        runtime(),
        in_proj_combined.into(),
        conv1d,
        a_log,
        dt_bias,
        norm_w,
        out_proj.into(),
    )
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

    // Same input and same RNG-seeded weights for both nets.
    let x_f32 = random_f32(&[1, 4, 2048], 42, 0.1, &dev);
    let x_bf16 = x_f32.to_dtype(DType::BF16).unwrap().contiguous().unwrap();

    // All-ones mask of len S — bypasses the native path's `mask.is_none()`
    // precondition without changing the SSM math (multiplicative identity).
    let mask = Tensor::ones((1, 4), DType::F32, &dev).unwrap();

    // Force the GDN kernel ON for the duration of this test.
    let _guard = GDN_TEST_GUARD.lock().unwrap();
    let was_enabled = lumen_metal::gated_delta::is_enabled();
    lumen_metal::gated_delta::set_enabled(true);

    let mut net_f32 = build_net(&dev);
    let mut net_bf16 = build_net(&dev);

    let y_f32 = net_f32.forward(&x_f32, Some(&mask)).expect("f32 forward");
    assert_eq!(y_f32.dtype(), DType::F32);

    let y_bf16 = net_bf16
        .forward(&x_bf16, Some(&mask))
        .expect("bf16 forward");
    assert_eq!(
        y_bf16.dtype(),
        DType::F32,
        "linear_attn must return f32 (residual contract)"
    );

    lumen_metal::gated_delta::set_enabled(was_enabled);

    assert_eq!(y_f32.dims(), y_bf16.dims(), "shape parity");

    let cos = cosine_sim(&y_f32, &y_bf16);
    let rel_l2 = relative_l2(&y_bf16, &y_f32);
    let max_abs = (&y_f32 - &y_bf16)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    eprintln!("B.4 bf16 chain parity: cos={cos:.6} rel_L2={rel_l2:.4e} max_abs={max_abs:.4e}");

    assert!(cos > 0.998, "cosine sim {cos} must exceed 0.998");
    assert!(
        rel_l2 < 2e-2,
        "relative L2 {rel_l2:.4e} must be < 2e-2 (~4× MLX bf16↔f32 floor)"
    );
}

/// Determinism: bf16 chain through GDN kernel must be bit-stable on repeat
/// calls (same protocol as Workstream B Phase 3 + Workstream C). Confirms
/// chain integration didn't introduce non-determinism above the kernel level.
#[test]
fn bf16_chain_determinism_repeat_call() {
    let dev = match metal_device() {
        Some(d) => d,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    let x = random_f32(&[1, 4, 2048], 7, 0.1, &dev)
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();
    let mask = Tensor::ones((1, 4), DType::F32, &dev).unwrap();

    let _guard = GDN_TEST_GUARD.lock().unwrap();
    let was_enabled = lumen_metal::gated_delta::is_enabled();
    lumen_metal::gated_delta::set_enabled(true);

    let mut runs: Vec<Vec<f32>> = Vec::with_capacity(5);
    for _ in 0..5 {
        let mut net = build_net(&dev);
        let y = net.forward(&x, Some(&mask)).expect("bf16 forward");
        runs.push(y.flatten_all().unwrap().to_vec1::<f32>().unwrap());
    }

    lumen_metal::gated_delta::set_enabled(was_enabled);

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
        "B.4 chain determinism: {differing} differing bits over {} comparisons",
        4 * n
    );
    assert_eq!(
        differing, 0,
        "bf16 chain GatedDeltaNet forward must be bit-deterministic across calls"
    );
}
