//! Gated Delta-Net SSM step kernel parity test.
//!
//! Compares `tq_gated_delta_step` Metal kernel against a Candle ops reference
//! implementation that mirrors `mlx_lm.models.gated_delta._gated_delta_step_ops`
//! (decode T=1 case).
//!
//! Tolerance: max_abs ≤ 1e-3 (the kernel uses simdgroup-reduced f32 sums; Candle
//! uses sequential f32 sums via `sum(D::Minus1)` — order-of-rounding differences
//! ≤ ULP-tier).
//!
//! Shapes covered: (B=1, T=1, Hk=4, Hv=8, Dk=64, Dv=64) — minimal.
//!                 (B=1, T=1, Hk=16, Hv=48, Dk=128, Dv=128) — Qwen3.6-27B Dense.

#![cfg(feature = "model-integration")]

use candle_core::{D, DType, Device, Tensor};
use lumen_metal::gated_delta::{gated_delta_step_candle, set_enabled};

fn metal_device() -> Option<Device> {
    Device::new_metal(0).ok()
}

/// Reference path: one timestep of gated-delta SSM via Candle ops, identical
/// to the body of the inner loop in `qwen3_5_moe::linear_attn::forward_inner`
/// (lines 768-794).
fn candle_ref_step(
    q_t: &Tensor,    // [B, Hv, Dk]
    k_t: &Tensor,    // [B, Hv, Dk]
    v_t: &Tensor,    // [B, Hv, Dv]
    g_t: &Tensor,    // [B, Hv]
    beta_t: &Tensor, // [B, Hv]
    state: &Tensor,  // [B, Hv, Dv, Dk]
) -> (Tensor, Tensor) {
    let decay = g_t
        .unsqueeze(D::Minus1)
        .unwrap()
        .unsqueeze(D::Minus1)
        .unwrap();
    let state = state.broadcast_mul(&decay).unwrap();
    let k_bc = k_t.unsqueeze(D::Minus2).unwrap();
    let kv_mem = state.broadcast_mul(&k_bc).unwrap().sum(D::Minus1).unwrap();
    let delta = (v_t - kv_mem)
        .unwrap()
        .broadcast_mul(&beta_t.unsqueeze(D::Minus1).unwrap())
        .unwrap();
    let outer = k_bc
        .broadcast_mul(&delta.unsqueeze(D::Minus1).unwrap())
        .unwrap();
    let state = (state + outer).unwrap();
    let q_bc = q_t.unsqueeze(D::Minus2).unwrap();
    let y_t = state.broadcast_mul(&q_bc).unwrap().sum(D::Minus1).unwrap();
    (y_t, state)
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.to_device(&Device::Cpu).unwrap().flatten_all().unwrap();
    let b = b.to_device(&Device::Cpu).unwrap().flatten_all().unwrap();
    let av = a.to_vec1::<f32>().unwrap();
    let bv = b.to_vec1::<f32>().unwrap();
    let mut m = 0.0f32;
    for (x, y) in av.iter().zip(bv.iter()) {
        let d = (x - y).abs();
        if d > m {
            m = d;
        }
    }
    m
}

fn make_input(
    device: &Device,
    b: usize,
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
    // Deterministic test inputs — sin/cos based, range [-0.5, 0.5].
    let make = |elems: usize, phase: f32, scale: f32| -> Vec<f32> {
        (0..elems)
            .map(|i| (i as f32 * 0.013 + phase).sin() * scale)
            .collect()
    };
    let q = Tensor::from_vec(make(b * hk * dk, 0.0, 0.5), (b, hk, dk), device).unwrap();
    let k = Tensor::from_vec(make(b * hk * dk, 1.7, 0.5), (b, hk, dk), device).unwrap();
    let v = Tensor::from_vec(make(b * hv * dv, 3.3, 0.5), (b, hv, dv), device).unwrap();
    // g should be in (0, 1] — use sigmoid-like values.
    let g = Tensor::from_vec(
        (0..b * hv)
            .map(|i| 0.5 + 0.4 * (i as f32 * 0.05).sin())
            .collect(),
        (b, hv),
        device,
    )
    .unwrap();
    let beta = Tensor::from_vec(
        (0..b * hv)
            .map(|i| 0.5 + 0.3 * (i as f32 * 0.07).cos())
            .collect(),
        (b, hv),
        device,
    )
    .unwrap();
    let state =
        Tensor::from_vec(make(b * hv * dv * dk, 5.5, 0.1), (b, hv, dv, dk), device).unwrap();
    (q, k, v, g, beta, state)
}

#[test]
fn gated_delta_kernel_matches_candle_ops() {
    let Some(device) = metal_device() else {
        eprintln!("Metal device unavailable; skipping");
        return;
    };

    // Repeat-heads helper: q/k come from the SSM at [B, Hk, Dk] and need to be
    // expanded to [B, Hv, Dk] (Hv = h_ratio * Hk) for the kernel input.
    fn repeat_heads(t: &Tensor, h_ratio: usize) -> Tensor {
        // [B, Hk, Dk] -> [B, Hk, h_ratio, Dk] -> [B, Hv, Dk]
        let dims = t.dims();
        let (b, hk, dk) = (dims[0], dims[1], dims[2]);
        let hv = hk * h_ratio;
        t.unsqueeze(2)
            .unwrap()
            .expand(&[b, hk, h_ratio, dk])
            .unwrap()
            .reshape(&[b, hv, dk])
            .unwrap()
            .contiguous()
            .unwrap()
    }

    let cases = [
        // (B, Hk, Hv, Dk, Dv)
        (1usize, 4, 8, 64, 64), // small smoke
        (1, 16, 48, 128, 128),  // Qwen3.6-27B Dense GDN shape
    ];

    set_enabled(true);

    for &(b, hk, hv, dk, dv) in cases.iter() {
        let h_ratio = hv / hk;
        let (q_hk, k_hk, v_hv, g, beta, state) = make_input(&device, b, hk, hv, dk, dv);

        // Candle reference: SSM expects q/k expanded to Hv heads.
        let q_hv = repeat_heads(&q_hk, h_ratio);
        let k_hv = repeat_heads(&k_hk, h_ratio);
        let (y_ref, state_ref) = candle_ref_step(&q_hv, &k_hv, &v_hv, &g, &beta, &state);

        // Kernel input: [B, T=1, Hk, Dk] for q/k, [B, T=1, Hv, Dv] for v,
        // [B, T=1, Hv] for g/beta, [B, Hv, Dv, Dk] for state.
        let q_in = q_hk.unsqueeze(1).unwrap().contiguous().unwrap();
        let k_in = k_hk.unsqueeze(1).unwrap().contiguous().unwrap();
        let v_in = v_hv.unsqueeze(1).unwrap().contiguous().unwrap();
        let g_in = g.unsqueeze(1).unwrap().contiguous().unwrap();
        let beta_in = beta.unsqueeze(1).unwrap().contiguous().unwrap();

        let (y_kernel_4d, state_kernel) =
            gated_delta_step_candle(&q_in, &k_in, &v_in, &g_in, &beta_in, &state)
                .expect("kernel returned None")
                .expect("kernel failed");
        // Squeeze the T=1 axis.
        let y_kernel = y_kernel_4d.squeeze(1).unwrap();

        let y_diff = max_abs_diff(&y_kernel, &y_ref);
        let s_diff = max_abs_diff(&state_kernel, &state_ref);

        eprintln!(
            "case B={b} Hk={hk} Hv={hv} Dk={dk} Dv={dv}: \
             y_max_abs={y_diff:.3e}, state_max_abs={s_diff:.3e}"
        );

        assert!(
            y_diff < 1e-3,
            "y mismatch: {y_diff:.3e} (B={b},Hv={hv},Dk={dk})"
        );
        assert!(
            s_diff < 1e-3,
            "state mismatch: {s_diff:.3e} (B={b},Hv={hv},Dk={dk})"
        );
    }

    set_enabled(false);
}
