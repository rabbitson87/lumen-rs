//! Flash Attention 2 bf16-I/O kernel parity + determinism tests.
//!
//! Validates `tq_flash_attn_bf16` (Workstream C — MLX dtype policy alignment)
//! against:
//!   1. The existing f32 `tq_flash_attn` kernel — establishes that switching
//!      I/O dtype only (compute stays f32) preserves the math within bf16
//!      rounding noise (target cos ≥ 0.999, max|Δ| ≤ 8e-3).
//!   2. CPU f32 reference SDPA — sanity floor.
//!   3. Self-determinism — 10 repeat calls produce bit-identical bf16 output
//!      (no MPSGraph-style reduction-order non-determinism — same fix
//!      protocol as `RmsNormBf16Out` from Phase 2).
//!
//! Decode/prefill shapes mirror Qwen3.6-27B Dense self_attn:
//!   B=1, H=16, H_kv=2 (group=8), D=256.

#![cfg(feature = "model-integration")]

use candle_core::{DType, Device, Tensor};
use lumen_metal::flash_attn::{flash_attn_candle, set_disabled};

fn metal_device() -> Option<Device> {
    Device::new_metal(0).ok()
}

fn cpu_sdpa_reference_f32(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Tensor {
    let cpu = Device::Cpu;
    let q = q.to_device(&cpu).unwrap();
    let k = k.to_device(&cpu).unwrap();
    let v = v.to_device(&cpu).unwrap();

    let dq = q.dims();
    let dk = k.dims();
    let (b, h, _sq, d) = (dq[0], dq[1], dq[2], dq[3]);
    let h_kv = dk[1];
    let group = h / h_kv;

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
        .matmul(
            &k_full
                .transpose(candle_core::D::Minus2, candle_core::D::Minus1)
                .unwrap(),
        )
        .unwrap();
    let scores = (scores * (scale as f64)).unwrap();
    let weights = candle_nn::ops::softmax_last_dim(&scores).unwrap();
    weights.matmul(&v_full).unwrap()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
    dot / (na * nb + 1e-30)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn random_tensor(shape: &[usize], seed: u64, dev: &Device) -> Tensor {
    // Deterministic linear-congruential filler — keeps the tests reproducible
    // without pulling in `rand`. Values in roughly [-0.5, 0.5].
    let n: usize = shape.iter().product();
    let mut state = seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493);
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = ((state >> 32) as u32) as f32 / u32::MAX as f32;
        data.push(bits - 0.5);
    }
    Tensor::from_vec(data, shape, dev).unwrap()
}

fn run_bf16(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Option<Tensor> {
    set_disabled(false);
    let q_b = q.to_dtype(DType::BF16).unwrap().contiguous().unwrap();
    let k_b = k.to_dtype(DType::BF16).unwrap().contiguous().unwrap();
    let v_b = v.to_dtype(DType::BF16).unwrap().contiguous().unwrap();
    flash_attn_candle(&q_b, &k_b, &v_b, None, scale).map(|r| r.unwrap())
}

fn run_f32(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Option<Tensor> {
    set_disabled(false);
    flash_attn_candle(q, k, v, None, scale).map(|r| r.unwrap())
}

#[test]
fn parity_decode_h16_skv256() {
    let dev = match metal_device() {
        Some(d) => d,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    // Qwen3.6-27B Dense self_attn decode shape (Sq=1).
    let (b, h, sq, d, h_kv, skv) = (1usize, 16, 1, 256, 2, 256);
    let scale = (d as f64).powf(-0.5) as f32;

    let q = random_tensor(&[b, h, sq, d], 11, &dev);
    let k = random_tensor(&[b, h_kv, skv, d], 22, &dev);
    let v = random_tensor(&[b, h_kv, skv, d], 33, &dev);

    let out_f32 = run_f32(&q, &k, &v, scale).expect("f32 flash_attn");
    let out_bf16 = run_bf16(&q, &k, &v, scale).expect("bf16 flash_attn");

    assert_eq!(out_bf16.dtype(), DType::BF16, "output must be bf16");

    let f32_v = out_f32.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let bf16_v = out_bf16
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let cos = cosine_similarity(&f32_v, &bf16_v);
    let max_d = max_abs_diff(&f32_v, &bf16_v);
    eprintln!("decode_h16_skv256 vs f32: cos={cos:.6} max|Δ|={max_d:.4e}");
    assert!(cos > 0.999, "cosine sim {cos} must exceed 0.999");
    assert!(max_d < 8e-3, "max|Δ| {max_d} must be < 8e-3");

    // Also compare to CPU reference (looser tolerance — bf16 rounding stacks).
    let out_cpu = cpu_sdpa_reference_f32(&q, &k, &v, scale);
    let cpu_v = out_cpu.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let cos_cpu = cosine_similarity(&cpu_v, &bf16_v);
    eprintln!("decode_h16_skv256 vs cpu: cos={cos_cpu:.6}");
    assert!(cos_cpu > 0.999, "vs cpu cosine {cos_cpu} must exceed 0.999");
}

#[test]
fn parity_prefill_h16_sq8_skv64() {
    let dev = match metal_device() {
        Some(d) => d,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    let (b, h, sq, d, h_kv, skv) = (1usize, 16, 8, 256, 2, 64);
    let scale = (d as f64).powf(-0.5) as f32;

    let q = random_tensor(&[b, h, sq, d], 100, &dev);
    let k = random_tensor(&[b, h_kv, skv, d], 200, &dev);
    let v = random_tensor(&[b, h_kv, skv, d], 300, &dev);

    let out_f32 = run_f32(&q, &k, &v, scale).expect("f32 flash_attn");
    let out_bf16 = run_bf16(&q, &k, &v, scale).expect("bf16 flash_attn");

    let f32_v = out_f32.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let bf16_v = out_bf16
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let cos = cosine_similarity(&f32_v, &bf16_v);
    let max_d = max_abs_diff(&f32_v, &bf16_v);
    eprintln!("prefill_h16_sq8_skv64 vs f32: cos={cos:.6} max|Δ|={max_d:.4e}");
    assert!(cos > 0.999, "cosine sim {cos} must exceed 0.999");
    assert!(max_d < 8e-3, "max|Δ| {max_d} must be < 8e-3");
}

#[test]
fn parity_with_mask_decode() {
    let dev = match metal_device() {
        Some(d) => d,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    let (b, h, sq, d, h_kv, skv) = (1usize, 16, 1, 256, 2, 128);
    let scale = (d as f64).powf(-0.5) as f32;

    let q = random_tensor(&[b, h, sq, d], 1, &dev);
    let k = random_tensor(&[b, h_kv, skv, d], 2, &dev);
    let v = random_tensor(&[b, h_kv, skv, d], 3, &dev);

    // Half the keys masked out (additive -1e4 ≈ -∞ in bf16).
    let mask_data: Vec<f32> = (0..(sq * skv))
        .map(|i| if i % 2 == 0 { 0.0 } else { -1e4 })
        .collect();
    let mask_f32 = Tensor::from_vec(mask_data, (sq, skv), &dev).unwrap();
    let mask_bf16 = mask_f32
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();

    set_disabled(false);
    let out_f32 = flash_attn_candle(&q, &k, &v, Some(&mask_f32), scale)
        .map(|r| r.unwrap())
        .expect("f32 flash_attn with mask");

    let q_b = q.to_dtype(DType::BF16).unwrap().contiguous().unwrap();
    let k_b = k.to_dtype(DType::BF16).unwrap().contiguous().unwrap();
    let v_b = v.to_dtype(DType::BF16).unwrap().contiguous().unwrap();
    let out_bf16 = flash_attn_candle(&q_b, &k_b, &v_b, Some(&mask_bf16), scale)
        .map(|r| r.unwrap())
        .expect("bf16 flash_attn with mask");

    let f32_v = out_f32.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let bf16_v = out_bf16
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let cos = cosine_similarity(&f32_v, &bf16_v);
    let max_d = max_abs_diff(&f32_v, &bf16_v);
    eprintln!("masked_decode_h16_skv128 vs f32: cos={cos:.6} max|Δ|={max_d:.4e}");
    assert!(cos > 0.999, "cosine sim {cos} must exceed 0.999");
    assert!(max_d < 8e-3, "max|Δ| {max_d} must be < 8e-3");
}

/// Workstream C determinism gate: same inputs → bit-identical bf16 output
/// across repeat calls. Same protocol as `RmsNormBf16Out` from Phase 2 —
/// we explicitly require this because the Phase 2 work showed that MPSGraph
/// reductions on Apple Silicon are *not* bit-deterministic (5119/5120 bits
/// flipped per call). A native Metal kernel with fixed reduction order
/// (simd_sum within fixed lane layout) MUST be bit-stable, otherwise it
/// breaks the chain's `R1↔R2` token determinism.
#[test]
fn determinism_repeat_call_bit_identical() {
    let dev = match metal_device() {
        Some(d) => d,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    let (b, h, sq, d, h_kv, skv) = (1usize, 16, 1, 256, 2, 256);
    let scale = (d as f64).powf(-0.5) as f32;

    let q = random_tensor(&[b, h, sq, d], 7, &dev)
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();
    let k = random_tensor(&[b, h_kv, skv, d], 8, &dev)
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();
    let v = random_tensor(&[b, h_kv, skv, d], 9, &dev)
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();

    set_disabled(false);
    let mut runs: Vec<Vec<u16>> = Vec::with_capacity(10);
    for _ in 0..10 {
        let out = flash_attn_candle(&q, &k, &v, None, scale)
            .map(|r| r.unwrap())
            .expect("bf16 flash_attn");
        // Read raw bf16 bits via byte view to detect any bit drift.
        let cpu = out.to_device(&Device::Cpu).unwrap().flatten_all().unwrap();
        let bytes = cpu.to_dtype(DType::U8).ok();
        // Fallback: compare via f32 vector — if any bit differs, the f32
        // reconstruction will too. (DType::U8 cast on bf16 is not available.)
        let _ = bytes;
        let bf16_bits: Vec<u16> = cpu
            .to_vec1::<half::bf16>()
            .unwrap()
            .into_iter()
            .map(|b| b.to_bits())
            .collect();
        runs.push(bf16_bits);
    }

    let n = runs[0].len();
    let mut differing = 0usize;
    for i in 1..runs.len() {
        for j in 0..n {
            if runs[0][j] != runs[i][j] {
                differing += 1;
            }
        }
    }
    eprintln!(
        "determinism: {differing} differing bits over 9 × {n} = {} positions",
        9 * n
    );
    assert_eq!(
        differing, 0,
        "bf16 flash_attn must be bit-deterministic across calls"
    );
}

#[test]
fn dtype_mismatch_returns_none() {
    let dev = match metal_device() {
        Some(d) => d,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    let (b, h, sq, d, h_kv, skv) = (1usize, 16, 1, 256, 2, 64);
    let scale = (d as f64).powf(-0.5) as f32;

    let q = random_tensor(&[b, h, sq, d], 1, &dev);
    let k_bf16 = random_tensor(&[b, h_kv, skv, d], 2, &dev)
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();
    let v = random_tensor(&[b, h_kv, skv, d], 3, &dev);

    set_disabled(false);
    let result = flash_attn_candle(&q, &k_bf16, &v, None, scale);
    assert!(
        result.is_none(),
        "mixed dtype must return None (caller falls back)"
    );
}
