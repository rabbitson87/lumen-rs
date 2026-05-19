//! Native Metal `RmsNormBf16Out` parity + determinism gate.
//!
//! Background: the predecessor `MpsRmsNormBf16Out` (MPSGraph-based) passed
//! scalar parity vs CPU (≤6e-3 bf16 mantissa floor) but produced bit-different
//! output across calls on identical inputs (5119/5120 bits flipped per call —
//! Apple-internal MPSGraph reduction-order optimization). That broke the
//! 27B Dense bf16 chain's `R1↔R2` token bit-identical contract (12/12 → 0/12).
//!
//! Acceptance for the native replacement:
//!   - Parity vs CPU: ≤6e-3 absolute (same bf16 mantissa budget as
//!     `MpsRmsNormBf16Out` parity test).
//!   - Determinism: bit-identical output across two calls on identical inputs.
//!     **This is the gate that distinguishes native from MPSGraph.**
//!   - Microbench: standalone cost ≤ 200% of MPSGraph version (best-effort —
//!     we trade some perf for determinism, but it shouldn't be catastrophic).

#![cfg(feature = "model-integration")]

use candle_core::{DType, Device, Tensor};
use lumen_metal::rms_norm::RmsNormBf16Out;

fn cpu_reference(x: &[f32], weight: &[f32], m: usize, hidden: usize, eps: f32) -> Vec<f32> {
    let mut y = vec![0.0_f32; m * hidden];
    for row in 0..m {
        let off = row * hidden;
        let mean_sq: f32 = x[off..off + hidden].iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv = (mean_sq + eps).sqrt().recip();
        for c in 0..hidden {
            y[off + c] = x[off + c] * inv * weight[c];
        }
    }
    y
}

fn synth_inputs(m: usize, hidden: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
    let mut s = seed;
    let x: Vec<f32> = (0..m * hidden)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (((s >> 8) & 0xFFFF) as f32 / 32768.0 - 1.0) * 1.5
        })
        .collect();
    let weight: Vec<f32> = (0..hidden).map(|i| 0.5 + 0.01 * (i as f32).sin()).collect();
    (x, weight)
}

fn parity_check(m: usize, hidden: usize) {
    let device = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    let eps = 1e-6_f32;
    let (x_data, w_data) = synth_inputs(m, hidden, 0xC0FFEE);

    let x = Tensor::from_vec(x_data.clone(), (m, hidden), &device).unwrap();
    let weight = Tensor::from_vec(w_data.clone(), hidden, &device).unwrap();

    let runtime = RmsNormBf16Out::new(eps).expect("init");
    let y = runtime.forward(&x, &weight).expect("forward");
    assert_eq!(y.dtype(), DType::BF16);
    assert_eq!(y.dims(), &[m, hidden]);

    let y_back: Vec<f32> = y
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let cpu = cpu_reference(&x_data, &w_data, m, hidden, eps);

    let max_abs = y_back
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    eprintln!("  rms_norm native shape=({m},{hidden}) max|bf16-cpu|={max_abs:.2e}");
    assert!(
        max_abs < 6e-3,
        "native bf16 vs CPU drift {max_abs:.2e} exceeds 6e-3 budget"
    );
}

#[test]
fn native_rms_norm_parity_decode_m1_h2048() {
    parity_check(1, 2048);
}

#[test]
fn native_rms_norm_parity_decode_m1_h5120() {
    parity_check(1, 5120); // 27B Dense hidden
}

#[test]
fn native_rms_norm_parity_prefill_m8_h2048() {
    parity_check(8, 2048);
}

/// **THE determinism gate.** This is the test that the MPSGraph predecessor
/// failed (5119/5120 bits flipped per repeat call). Native must produce
/// bit-identical output across calls on identical inputs.
#[test]
fn native_rms_norm_determinism_repeat_call_bit_identical() {
    let device = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    let eps = 1e-6_f32;
    let m = 1usize;
    let hidden = 5120usize;
    let (x_data, w_data) = synth_inputs(m, hidden, 0xDEADBEEF);
    let x = Tensor::from_vec(x_data, (m, hidden), &device).unwrap();
    let weight = Tensor::from_vec(w_data, hidden, &device).unwrap();
    let runtime = RmsNormBf16Out::new(eps).expect("init");

    let y1 = runtime.forward(&x, &weight).expect("call 1");
    let y2 = runtime.forward(&x, &weight).expect("call 2");

    let bits1: Vec<u16> = y1
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|f| half::bf16::from_f32(*f).to_bits())
        .collect();
    let bits2: Vec<u16> = y2
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|f| half::bf16::from_f32(*f).to_bits())
        .collect();

    let mismatches = bits1
        .iter()
        .zip(bits2.iter())
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "native rms_norm repeat-call bits: {mismatches}/{} differ",
        bits1.len()
    );
    assert_eq!(
        mismatches,
        0,
        "native RmsNormBf16Out is non-deterministic — {mismatches}/{} bits differ",
        bits1.len()
    );
}

/// Cross-validation: native and MPSGraph versions should agree within bf16
/// mantissa precision (both target the same math, so output should be near
/// bit-equal; structural divergence would indicate a kernel bug).
#[test]
#[cfg(feature = "mpsgraph")]
fn native_vs_mpsgraph_parity() {
    use lumen_metal::mpsgraph::MpsRmsNormBf16Out;

    let device = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    let eps = 1e-6_f32;
    let m = 1usize;
    let hidden = 5120usize;
    let (x_data, w_data) = synth_inputs(m, hidden, 0x1234_5678);
    let x = Tensor::from_vec(x_data, (m, hidden), &device).unwrap();
    let weight = Tensor::from_vec(w_data, hidden, &device).unwrap();

    let native = RmsNormBf16Out::new(eps).expect("native init");
    let mpsg = MpsRmsNormBf16Out::new(eps).expect("mpsg init");

    let y_native: Vec<f32> = native
        .forward(&x, &weight)
        .expect("native fwd")
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let y_mpsg: Vec<f32> = mpsg
        .forward(&x, &weight)
        .expect("mpsg fwd")
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let max_abs = y_native
        .iter()
        .zip(y_mpsg.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("native vs MPSGraph max|Δ|={max_abs:.2e}");
    assert!(
        max_abs < 6e-3,
        "native vs MPSGraph drift {max_abs:.2e} exceeds 6e-3 — possible kernel divergence"
    );
}
