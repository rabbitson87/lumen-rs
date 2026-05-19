//! Workstream B Phase 5 — bf16 chain forward parity test for the
//! `post_attention_layernorm → MLP` boundary.
//!
//! What this test proves:
//!   - `DenseMlp::forward_bf16_in(x_bf16)` yields output numerically
//!     equivalent to `DenseMlp::forward(x_f32)` within bf16 rounding
//!     tolerance.
//!   - `DenseMlp::forward_with_residual_bf16_in(x_bf16, residual)` yields
//!     output numerically equivalent to `forward_with_residual(x_f32, residual)`.
//!   - Repeat calls on the same input are bit-deterministic (same protocol
//!     as B.3 / B.4 / Workstream C).
//!
//! Path exercised (Dense ProjLinear arm — cast-and-defer fallback):
//!   bf16 input → ProjLinear::forward_bf16_in (Dense arm casts internally
//!   to f32) → existing f32 SharedExpert forward → f32 output.
//!
//! The Affine4 native fast path (`affine4_qmv_fast_bf16in_bf16out` on
//! gate_up) is not exercised here because synthesizing a quantized Affine4
//! checkpoint inside an integration test is heavy. The fast path is
//! covered by `affine4_qmv_fast_bf16_parity` upstream in lumen-metal,
//! and end-to-end on the user's external `scripts/run_bf16_chain_ab.sh`.

#![cfg(feature = "turboquant-gpu")]

use candle_core::{DType, Device, Tensor};
use candle_nn::Linear;
use lumen_model::qwen3_5_moe::moe::{DenseMlp, MlpBlock};

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

fn build_dense_mlp(dev: &Device, hidden: usize, intermediate: usize) -> DenseMlp {
    let gate_up_w = random_f32(&[2 * intermediate, hidden], 11, 0.02, dev);
    let down_w = random_f32(&[hidden, intermediate], 12, 0.02, dev);
    let gate_up = Linear::new(gate_up_w, None);
    let down = Linear::new(down_w, None);
    DenseMlp::new(gate_up.into(), down.into(), intermediate)
}

#[test]
fn dense_mlp_forward_bf16_in_matches_f32_within_tolerance() {
    let dev = match metal_device() {
        Some(d) => d,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    let hidden = 2048;
    let intermediate = 5120; // Qwen3.6-27B Dense layer
    let mlp = build_dense_mlp(&dev, hidden, intermediate);

    let x_f32 = random_f32(&[1, 4, hidden], 42, 0.1, &dev);
    let x_bf16 = x_f32.to_dtype(DType::BF16).unwrap().contiguous().unwrap();

    let y_f32 = mlp.forward(&x_f32).expect("f32 forward");
    let y_b16 = mlp.forward_bf16_in(&x_bf16).expect("bf16-in forward");

    assert_eq!(
        y_b16.dtype(),
        DType::F32,
        "MLP output stays f32 (residual contract)"
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

    eprintln!(
        "B.5 forward_bf16_in (Dense arm) parity: cos={cos:.6} rel_L2={rel_l2:.4e} max_abs={max_abs:.4e}"
    );

    assert!(cos > 0.998, "cosine sim {cos} must exceed 0.998");
    assert!(
        rel_l2 < 1e-2,
        "relative L2 {rel_l2:.4e} must be < 1e-2 (bf16↔f32 input rounding)"
    );
}

#[test]
fn dense_mlp_forward_with_residual_bf16_in_matches_f32_within_tolerance() {
    let dev = match metal_device() {
        Some(d) => d,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    let hidden = 2048;
    let intermediate = 5120;
    let mlp = build_dense_mlp(&dev, hidden, intermediate);

    let x_f32 = random_f32(&[1, 4, hidden], 42, 0.1, &dev);
    let residual = random_f32(&[1, 4, hidden], 9, 0.1, &dev);
    let x_bf16 = x_f32.to_dtype(DType::BF16).unwrap().contiguous().unwrap();

    let y_f32 = mlp
        .forward_with_residual(&x_f32, &residual)
        .expect("f32 fwd+res");
    let y_b16 = mlp
        .forward_with_residual_bf16_in(&x_bf16, &residual)
        .expect("bf16-in fwd+res");

    assert_eq!(y_b16.dtype(), DType::F32);
    assert_eq!(y_f32.dims(), y_b16.dims());

    let cos = cosine_sim(&y_f32, &y_b16);
    let rel_l2 = relative_l2(&y_b16, &y_f32);
    eprintln!(
        "B.5 forward_with_residual_bf16_in (Dense arm) parity: cos={cos:.6} rel_L2={rel_l2:.4e}"
    );
    assert!(cos > 0.998);
    assert!(rel_l2 < 1e-2);
}

#[test]
fn mlp_block_dispatch_returns_bf16_in_for_dense_and_none_for_moe_intent() {
    // Just sanity-check the MlpBlock dispatch surface — Dense arm should
    // accept the bf16-in call (returns Some(_)). MoE arm returns None
    // (B.5 scope is Dense only; MoE bf16 wiring is B.7+).
    let dev = match metal_device() {
        Some(d) => d,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };
    let hidden = 256;
    let intermediate = 512;
    let dense = build_dense_mlp(&dev, hidden, intermediate);
    let mlp = MlpBlock::Dense(dense);

    let x_bf16 = random_f32(&[1, 1, hidden], 1, 0.1, &dev)
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();

    let y = mlp.forward_bf16_in(&x_bf16);
    assert!(y.is_some(), "Dense arm must accept bf16-in dispatch");
    assert_eq!(y.unwrap().unwrap().dtype(), DType::F32);

    // supports_bf16_in_fast_path should be false here (Dense ProjLinear is
    // not Affine4) — the dispatch still works, just via the cast fallback.
    assert!(
        !mlp.supports_bf16_in_fast_path(),
        "Dense ProjLinear is not Affine4 → no native fast path"
    );
}

/// Workstream B Phase 8 — `forward_bf16_in_bf16_out` parity. Same composition
/// as B.5's `forward_bf16_in`, but `down_proj` returns bf16. Output bf16
/// must round-trip to f32 within bf16 rounding tolerance vs the f32 reference.
#[test]
fn dense_mlp_forward_bf16_in_bf16_out_matches_f32_within_tolerance() {
    let dev = match metal_device() {
        Some(d) => d,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    let hidden = 2048;
    let intermediate = 5120;
    let mlp = build_dense_mlp(&dev, hidden, intermediate);

    let x_f32 = random_f32(&[1, 4, hidden], 42, 0.1, &dev);
    let x_bf16 = x_f32.to_dtype(DType::BF16).unwrap().contiguous().unwrap();

    let y_f32 = mlp.forward(&x_f32).expect("f32 forward");
    let y_bf16 = mlp
        .forward_bf16_in_bf16_out(&x_bf16)
        .expect("bf16-in-bf16-out forward");

    assert_eq!(
        y_bf16.dtype(),
        DType::BF16,
        "B.8 forward_bf16_in_bf16_out must return BF16 (residual stream contract)"
    );
    assert_eq!(y_f32.dims(), y_bf16.dims());

    let cos = cosine_sim(&y_f32, &y_bf16);
    let rel_l2 = relative_l2(&y_bf16, &y_f32);
    let max_abs = (&y_f32.to_dtype(DType::F32).unwrap() - &y_bf16.to_dtype(DType::F32).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    eprintln!(
        "B.8 forward_bf16_in_bf16_out (Dense arm) parity: cos={cos:.6} rel_L2={rel_l2:.4e} max_abs={max_abs:.4e}"
    );

    // Same envelope as B.5: bf16 rounding on the input + a single output
    // narrowing on down_proj. cos > 0.998 is the bf16-chain floor used
    // across B.3/B.4/B.5/B.6.
    assert!(cos > 0.998, "cosine sim {cos} must exceed 0.998");
    assert!(
        rel_l2 < 2e-2,
        "relative L2 {rel_l2:.4e} must be < 2e-2 (bf16 input + bf16 output rounding)"
    );
}

/// Workstream B Phase 8 — `forward_with_residual_bf16_in_bf16_out` parity.
/// bf16 input, bf16 residual, bf16 output. Output must match the f32
/// reference (`forward_with_residual` on f32 with the same residual cast
/// to f32) within bf16 rounding tolerance.
#[test]
fn dense_mlp_forward_with_residual_bf16_in_bf16_out_matches_f32_within_tolerance() {
    let dev = match metal_device() {
        Some(d) => d,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    let hidden = 2048;
    let intermediate = 5120;
    let mlp = build_dense_mlp(&dev, hidden, intermediate);

    let x_f32 = random_f32(&[1, 4, hidden], 42, 0.1, &dev);
    let residual_f32 = random_f32(&[1, 4, hidden], 9, 0.1, &dev);
    let x_bf16 = x_f32.to_dtype(DType::BF16).unwrap().contiguous().unwrap();
    let residual_bf16 = residual_f32
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();

    let y_f32 = mlp
        .forward_with_residual(&x_f32, &residual_f32)
        .expect("f32 fwd+res");
    let y_bf16 = mlp
        .forward_with_residual_bf16_in_bf16_out(&x_bf16, &residual_bf16)
        .expect("bf16-in-bf16-out fwd+res");

    assert_eq!(y_bf16.dtype(), DType::BF16);
    assert_eq!(y_f32.dims(), y_bf16.dims());

    let cos = cosine_sim(&y_f32, &y_bf16);
    let rel_l2 = relative_l2(&y_bf16, &y_f32);
    eprintln!(
        "B.8 forward_with_residual_bf16_in_bf16_out (Dense arm) parity: cos={cos:.6} rel_L2={rel_l2:.4e}"
    );
    assert!(cos > 0.998);
    assert!(rel_l2 < 2e-2);
}

/// Workstream B Phase 8 — residual dtype contract. The bf16-in-bf16-out
/// residual variant requires the residual to be BF16. Surface a loud error
/// rather than silently casting (caller is `layer.rs` which must keep the
/// residual stream consistent).
#[test]
fn dense_mlp_forward_with_residual_bf16_in_bf16_out_rejects_f32_residual() {
    let dev = match metal_device() {
        Some(d) => d,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };
    let hidden = 256;
    let intermediate = 512;
    let mlp = build_dense_mlp(&dev, hidden, intermediate);

    let x_bf16 = random_f32(&[1, 1, hidden], 1, 0.1, &dev)
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();
    // Residual is intentionally f32 — should be rejected.
    let residual_f32 = random_f32(&[1, 1, hidden], 2, 0.1, &dev);

    let result = mlp.forward_with_residual_bf16_in_bf16_out(&x_bf16, &residual_f32);
    let msg = match result {
        Ok(_) => panic!("f32 residual must be rejected — Ok returned"),
        Err(e) => format!("{e}"),
    };
    assert!(
        msg.contains("residual dtype") && msg.contains("BF16"),
        "expected dtype-contract error, got: {msg}"
    );
}

/// Workstream B Phase 8 — MlpBlock dispatch surface for bf16-in-bf16-out.
/// Dense returns `Some`; MoE returns `None` (bf16 wiring lands in B.7+).
/// `supports_bf16_in_bf16_out_fast_path()` matches the same Dense+Affine4
/// gate as B.5's `supports_bf16_in_fast_path` (the test build's Dense
/// ProjLinear isn't Affine4 → returns false).
#[test]
fn mlp_block_dispatch_bf16_in_bf16_out_surface() {
    let dev = match metal_device() {
        Some(d) => d,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };
    let hidden = 256;
    let intermediate = 512;
    let dense = build_dense_mlp(&dev, hidden, intermediate);
    let mlp = MlpBlock::Dense(dense);

    let x_bf16 = random_f32(&[1, 1, hidden], 1, 0.1, &dev)
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();

    let y = mlp.forward_bf16_in_bf16_out(&x_bf16);
    assert!(
        y.is_some(),
        "Dense arm must accept bf16-in-bf16-out dispatch"
    );
    assert_eq!(y.unwrap().unwrap().dtype(), DType::BF16);

    let residual_bf16 = random_f32(&[1, 1, hidden], 2, 0.1, &dev)
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();
    let y_res = mlp.forward_with_residual_bf16_in_bf16_out(&x_bf16, &residual_bf16);
    assert!(
        y_res.is_some(),
        "Dense arm must accept residual bf16-in-bf16-out"
    );
    assert_eq!(y_res.unwrap().unwrap().dtype(), DType::BF16);

    // Dense ProjLinear (test build) is not Affine4 → no native fast path.
    assert!(
        !mlp.supports_bf16_in_bf16_out_fast_path(),
        "Dense ProjLinear is not Affine4 → no native bf16-in-bf16-out fast path"
    );
}

#[test]
fn dense_mlp_forward_bf16_in_determinism_repeat_call() {
    let dev = match metal_device() {
        Some(d) => d,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    let hidden = 2048;
    let intermediate = 5120;
    let mlp = build_dense_mlp(&dev, hidden, intermediate);

    let x = random_f32(&[1, 4, hidden], 7, 0.1, &dev)
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();

    let mut runs: Vec<Vec<f32>> = Vec::with_capacity(5);
    for _ in 0..5 {
        let y = mlp.forward_bf16_in(&x).expect("bf16-in forward");
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
        "B.5 forward_bf16_in determinism: {differing} differing bits over {} comparisons",
        4 * n
    );
    assert_eq!(
        differing, 0,
        "DenseMlp::forward_bf16_in must be bit-deterministic across calls"
    );
}
