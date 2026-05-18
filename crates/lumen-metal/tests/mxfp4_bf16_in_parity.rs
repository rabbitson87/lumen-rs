//! Lever B L.2 acceptance gate — `mxfp4_matmul_bf16in_f32out_v3` parity +
//! cost POC.
//!
//! The bf16-input kernel widens `bfloat` activations to f32 once during the
//! cooperative threadgroup-memory staging step. From the `threadgroup_barrier`
//! onward the math is bit-identical to `mxfp4_matmul_f32_v3`, so the only
//! source of drift vs the f32-in path is the input-side bf16 mantissa
//! truncation (≤ 7.8e-3 relative per element). After the matmul reduction
//! collapses `in_features` partial products into a single output value the
//! observable cosine drift is much tighter than the per-element bound.
//!
//! Acceptance (per `.outline/lever_b_bf16_rmsnorm_plan.md` §L.2):
//!   - cosine similarity ≥ 0.9999 on production shapes (qkv_proj out=9216,
//!     o_proj out=2048×8192, etc.) — looser would suggest a structural bug in
//!     the kernel (e.g. bad widening, dispatch arity mismatch).
//!   - microbench cost within ±5% of f32-in on the decode shape (m=1).
//!
//! The microbench is best-effort under the dev profile; we keep a 20% ceiling
//! to guard against structural regressions (extra dispatch, accidental sync)
//! while leaving headroom for scheduler jitter.

use std::sync::Arc;
use std::time::Instant;

use lumen_metal::mxfp4_gpu::{Mxfp4Weight, MxFp4Context};

fn synth_packed(out: usize, ins: usize, seed: u32) -> Vec<u32> {
    let n = out * ins / 8;
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            s
        })
        .collect()
}

fn synth_scales(out: usize, ins: usize, seed: u32) -> Vec<u8> {
    let n = out * ins / 32;
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            120u8.saturating_add(((s >> 8) & 0x0F) as u8)
        })
        .collect()
}

fn synth_x(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.013).sin() * 1.7).collect()
}

fn f32_vec_to_bf16_bits(xs: &[f32]) -> Vec<u16> {
    xs.iter().map(|x| half::bf16::from_f32(*x).to_bits()).collect()
}

/// Round-trip f32 → bf16 → f32 to recover the exact value the bf16-in kernel
/// will widen back. Comparing against this (instead of the original f32) lets
/// us isolate the matmul reduction error from the input quantization error.
fn f32_through_bf16(xs: &[f32]) -> Vec<f32> {
    xs.iter()
        .map(|x| half::bf16::from_f32(*x).to_f32())
        .collect()
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn rel_max_err(a: &[f32], b: &[f32]) -> f32 {
    let max_mag = a
        .iter()
        .chain(b.iter())
        .map(|v| v.abs())
        .fold(0.0f32, f32::max);
    if max_mag == 0.0 {
        return 0.0;
    }
    max_abs_err(a, b) / max_mag
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na * nb)) as f32
}

fn assert_bf16_in_parity(name: &str, ref_f32: &[f32], bf16_in: &[f32]) {
    let cos = cosine_similarity(ref_f32, bf16_in);
    let rel = rel_max_err(ref_f32, bf16_in);
    let abs = max_abs_err(ref_f32, bf16_in);
    eprintln!(
        "  {name}: cos={cos:.6}  rel_max={rel:.3e}  abs_max={abs:.3e}"
    );
    assert!(
        cos >= 0.9999,
        "{name}: cosine {cos} below 0.9999 (abs={abs}, rel={rel})"
    );
}

/// Raw kernel parity: `mxfp4_matmul_bf16in_f32out_v3` vs the f32 v3 path on
/// the same activation. The bf16 buffer is the bit-truncation of the f32
/// input; the reference is recomputed against the round-tripped f32 to
/// isolate matmul reduction error from input quantization.
#[test]
fn mxfp4_bf16_in_v3_parity_production_shapes() {
    let ctx = match MxFp4Context::new() {
        Ok(c) => Arc::new(c),
        Err(_) => return, // No Metal device — skip (CI without GPU).
    };

    // Shapes representative of Qwen3.6-35B-A3B-mxfp4 hot path.
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("qkv-small", 256, 256, 1),
        ("gate_up", 1024, 2048, 1),
        ("down", 2048, 512, 1),
        ("o_proj", 2048, 8192, 1),
        ("qkv-prod", 9216, 2048, 1),
        ("batch4", 64, 256, 4),
    ];

    for &(name, out, ins, batch) in shapes {
        let packed = synth_packed(out, ins, 0xC0FFEE);
        let scales = synth_scales(out, ins, 0xBADCAFE);
        let x_f32 = synth_x(batch * ins);
        let weight = Mxfp4Weight::from_host(&ctx.ctx, &packed, &scales, out, ins).unwrap();

        // f32-in reference: feed the bf16-truncated activation through the
        // f32 path to isolate the matmul reduction from input quantization.
        let x_round_trip = f32_through_bf16(&x_f32);
        let x_ref_buf = ctx.ctx.buffer_with_data(&x_round_trip);
        let y_ref_buf = ctx.ctx.buffer_for::<f32>(batch * out);
        ctx.matmul_zero_copy(&weight, &x_ref_buf, 0, &y_ref_buf, 0, batch)
            .unwrap();
        let y_ref = ctx.ctx.read_buffer::<f32>(&y_ref_buf, batch * out);

        // bf16-in path: bind the raw bf16 bits as a `bfloat*` buffer.
        let x_bf16_bits = f32_vec_to_bf16_bits(&x_f32);
        let x_bf16_buf = ctx.ctx.buffer_with_data(&x_bf16_bits);
        let y_bf16in_buf = ctx.ctx.buffer_for::<f32>(batch * out);
        ctx.matmul_zero_copy_bf16_in(&weight, &x_bf16_buf, 0, &y_bf16in_buf, 0, batch)
            .unwrap();
        let y_bf16in = ctx.ctx.read_buffer::<f32>(&y_bf16in_buf, batch * out);

        assert_bf16_in_parity(
            &format!("dense `{name}` ({out}x{ins} batch={batch})"),
            &y_ref,
            &y_bf16in,
        );
    }
}

/// Tensor-level parity: `Mxfp4Linear::forward` (f32-in, post round-trip to
/// bf16) vs `Mxfp4Linear::forward_bf16_in` (bf16 Tensor input). Exercises the
/// full wrapper including Candle DType handling and Metal buffer extraction.
#[test]
fn mxfp4_linear_forward_bf16_in_tensor_parity() {
    use candle_core::{DType, Device, Tensor};
    use lumen_metal::mxfp4_linear::Mxfp4Linear;

    let device = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    let ctx = match MxFp4Context::new() {
        Ok(c) => Arc::new(c),
        Err(_) => return,
    };

    let shapes: &[(&str, usize, usize, usize)] = &[
        ("decode-qkv", 9216, 2048, 1),
        ("decode-o", 2048, 8192, 1),
        ("prefill-batch", 2048, 2048, 8),
    ];

    for &(name, out, ins, batch) in shapes {
        let packed = synth_packed(out, ins, 0xCAFEBABE);
        let scales = synth_scales(out, ins, 0xFACEFEED);
        let weight = Mxfp4Weight::from_host(&ctx.ctx, &packed, &scales, out, ins).unwrap();
        let linear = Mxfp4Linear::new(weight, None, ctx.clone());

        let x_vec = synth_x(batch * ins);
        let x_f32 = Tensor::from_vec(x_vec.clone(), (batch, ins), &device).unwrap();

        // f32-in reference fed the round-tripped activation.
        let x_round_trip_vec = f32_through_bf16(&x_vec);
        let x_round_trip =
            Tensor::from_vec(x_round_trip_vec, (batch, ins), &device).unwrap();
        let y_ref = linear.forward(&x_round_trip).unwrap();
        assert_eq!(y_ref.dtype(), DType::F32);
        let y_ref_vec = y_ref.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // bf16-in path. The wrapper accepts f32 too (auto-casts) but the
        // production wiring will pass a bf16 tensor produced by RmsNormBf16Out.
        let x_bf16 = x_f32.to_dtype(DType::BF16).unwrap();
        let y_bf16in = linear.forward_bf16_in(&x_bf16).unwrap();
        assert_eq!(
            y_bf16in.dtype(),
            DType::F32,
            "{name}: forward_bf16_in must produce f32 tensor"
        );
        let y_bf16in_vec = y_bf16in.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_bf16_in_parity(
            &format!("Mxfp4Linear `{name}` ({out}x{ins} batch={batch})"),
            &y_ref_vec,
            &y_bf16in_vec,
        );
    }
}

/// Microbench: per-call latency of f32-in vs bf16-in dispatch on the decode
/// qkv-proj shape (out=9216, in=2048, batch=1). Acceptance per L.2:
/// bf16-in within ±5% of f32-in. Generous 20% ceiling absorbs dev-profile
/// jitter while still catching structural regressions.
#[test]
fn mxfp4_bf16_in_microbench_within_baseline() {
    let ctx = match MxFp4Context::new() {
        Ok(c) => Arc::new(c),
        Err(_) => return,
    };

    let out = 9216usize;
    let ins = 2048usize;
    let batch = 1usize;

    let packed = synth_packed(out, ins, 0xBEEFCAFE);
    let scales = synth_scales(out, ins, 0xFADEBEEF);
    let x_f32 = synth_x(batch * ins);
    let weight = Mxfp4Weight::from_host(&ctx.ctx, &packed, &scales, out, ins).unwrap();

    let x_f32_buf = ctx.ctx.buffer_with_data(&x_f32);
    let x_bf16_bits = f32_vec_to_bf16_bits(&x_f32);
    let x_bf16_buf = ctx.ctx.buffer_with_data(&x_bf16_bits);

    let y_f32_buf = ctx.ctx.buffer_for::<f32>(batch * out);
    let y_bf16in_buf = ctx.ctx.buffer_for::<f32>(batch * out);

    // Warmup.
    for _ in 0..16 {
        ctx.matmul_zero_copy(&weight, &x_f32_buf, 0, &y_f32_buf, 0, batch)
            .unwrap();
        ctx.matmul_zero_copy_bf16_in(&weight, &x_bf16_buf, 0, &y_bf16in_buf, 0, batch)
            .unwrap();
    }

    let iters = 200;
    let t0 = Instant::now();
    for _ in 0..iters {
        ctx.matmul_zero_copy(&weight, &x_f32_buf, 0, &y_f32_buf, 0, batch)
            .unwrap();
    }
    let f32_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let t1 = Instant::now();
    for _ in 0..iters {
        ctx.matmul_zero_copy_bf16_in(&weight, &x_bf16_buf, 0, &y_bf16in_buf, 0, batch)
            .unwrap();
    }
    let bf16in_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    eprintln!(
        "qkv-prod ({out}x{ins} batch={batch}):  f32-in: {f32_ms:.4} ms  \
         bf16-in: {bf16in_ms:.4} ms  ratio: {:.3}",
        bf16in_ms / f32_ms
    );

    assert!(
        bf16in_ms <= f32_ms * 1.20,
        "bf16-in {bf16in_ms:.4}ms exceeds f32-in {f32_ms:.4}ms × 1.20"
    );
}
