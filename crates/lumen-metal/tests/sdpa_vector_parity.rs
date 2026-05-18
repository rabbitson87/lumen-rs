//! SDPA vector kernel parity test.
//!
//! Compares the MLX-style `tq_sdpa_vector` decode kernel against:
//!   1. Existing `tq_flash_attn` (FA2) kernel — both online softmax, same f32
//!      arithmetic in different orders → max_abs ≤ 1e-3 tolerance
//!   2. CPU reference SDPA — establishes ground truth → max_abs ≤ 5e-3
//!
//! Decode shape used: B=1, H=40, Sq=1, D=256, H_kv=8 (group=5), Skv ∈ {64, 256, 1024}.
//! Mask: causal-style (full visible during decode, additive bias = 0).

#![cfg(feature = "model-integration")]

use candle_core::{DType, Device, Tensor};
use lumen_metal::flash_attn::{flash_attn_candle, set_disabled, set_sdpa_vector_enabled};

fn metal_device() -> Option<Device> {
    Device::new_metal(0).ok()
}

/// Reference scaled dot-product attention on CPU at f32. GQA is handled via
/// `repeat` (kv_head[i / group] for q_head[i]).
fn cpu_sdpa_reference(
    q: &Tensor,    // [B, H, Sq, D]
    k: &Tensor,    // [B, H_kv, Skv, D]
    v: &Tensor,    // [B, H_kv, Skv, D]
    scale: f32,
) -> Tensor {
    let cpu = Device::Cpu;
    let q = q.to_device(&cpu).unwrap();
    let k = k.to_device(&cpu).unwrap();
    let v = v.to_device(&cpu).unwrap();

    let dq = q.dims();
    let dk = k.dims();
    let (b, h, sq, d) = (dq[0], dq[1], dq[2], dq[3]);
    let h_kv = dk[1];
    let group = h / h_kv;

    // Expand K/V from H_kv → H by repeating each kv head `group` times.
    let k_full = k
        .unsqueeze(2)
        .unwrap()
        .expand(&[b, h_kv, group, dk[2], d])
        .unwrap()
        .reshape(&[b, h, dk[2], d])
        .unwrap()
        .contiguous()
        .unwrap();
    let v_full = v
        .unsqueeze(2)
        .unwrap()
        .expand(&[b, h_kv, group, dk[2], d])
        .unwrap()
        .reshape(&[b, h, dk[2], d])
        .unwrap()
        .contiguous()
        .unwrap();

    let scores = q
        .matmul(&k_full.transpose(candle_core::D::Minus2, candle_core::D::Minus1).unwrap())
        .unwrap();
    let scores = (scores * (scale as f64)).unwrap();
    let weights = candle_nn::ops::softmax_last_dim(&scores).unwrap();
    let out = weights.matmul(&v_full).unwrap();

    let _ = sq;
    out
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let cpu = Device::Cpu;
    let a = a.to_device(&cpu).unwrap().flatten_all().unwrap();
    let b = b.to_device(&cpu).unwrap().flatten_all().unwrap();
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

fn make_qkv(
    device: &Device,
    b: usize,
    h: usize,
    sq: usize,
    skv: usize,
    h_kv: usize,
    d: usize,
) -> (Tensor, Tensor, Tensor) {
    // Deterministic test data — sin/cos based, range [-1, 1].
    let make = |elems: usize, phase: f32| -> Vec<f32> {
        (0..elems)
            .map(|i| ((i as f32 * 0.013 + phase).sin() * 0.5))
            .collect()
    };
    let q_data = make(b * h * sq * d, 0.0);
    let k_data = make(b * h_kv * skv * d, 1.7);
    let v_data = make(b * h_kv * skv * d, 3.3);

    let q = Tensor::from_vec(q_data, (b, h, sq, d), device).unwrap();
    let k = Tensor::from_vec(k_data, (b, h_kv, skv, d), device).unwrap();
    let v = Tensor::from_vec(v_data, (b, h_kv, skv, d), device).unwrap();
    (q, k, v)
}

#[test]
fn sdpa_vector_matches_flash_attn() {
    let Some(device) = metal_device() else {
        eprintln!("Metal device unavailable; skipping");
        return;
    };

    let cases = [
        // (B, H, Sq, Skv, H_kv, D)
        (1usize, 8, 1, 64, 2, 256),    // small Skv, small H
        (1, 40, 1, 256, 8, 256),       // Qwen3.6-27B Dense decode shape, mid Skv
        (1, 40, 1, 1024, 8, 256),      // long context decode
        (1, 16, 1, 128, 4, 256),       // mid case
    ];

    for &(b, h, sq, skv, h_kv, d) in cases.iter() {
        let (q, k, v) = make_qkv(&device, b, h, sq, skv, h_kv, d);
        let scale = (d as f32).powf(-0.5);

        // CPU ref
        let cpu_out = cpu_sdpa_reference(&q, &k, &v, scale);

        // FA path
        set_sdpa_vector_enabled(false);
        set_disabled(false);
        let fa_out = flash_attn_candle(&q, &k, &v, None, scale)
            .expect("FA returned None")
            .expect("FA failed");

        // SDPA vector path
        set_sdpa_vector_enabled(true);
        let sdpav_out = flash_attn_candle(&q, &k, &v, None, scale)
            .expect("SDPAV returned None")
            .expect("SDPAV failed");
        set_sdpa_vector_enabled(false);

        let fa_vs_cpu = max_abs_diff(&fa_out, &cpu_out);
        let sdpav_vs_cpu = max_abs_diff(&sdpav_out, &cpu_out);
        let sdpav_vs_fa = max_abs_diff(&sdpav_out, &fa_out);

        eprintln!(
            "case B={b} H={h} Sq={sq} Skv={skv} H_kv={h_kv} D={d}: \
             FA-vs-CPU={fa_vs_cpu:.3e} SDPAV-vs-CPU={sdpav_vs_cpu:.3e} \
             SDPAV-vs-FA={sdpav_vs_fa:.3e}"
        );

        assert!(
            sdpav_vs_cpu < 5e-3,
            "SDPA vector vs CPU mismatch too large: {sdpav_vs_cpu:.3e} \
             (case Skv={skv} H={h})"
        );
        assert!(
            sdpav_vs_fa < 1e-3,
            "SDPA vector vs FA mismatch too large: {sdpav_vs_fa:.3e} \
             (case Skv={skv} H={h})"
        );
    }
}
