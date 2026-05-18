//! Parity check: `mxfp4_matmul_f32in_bf16out_v3` (Phase A.0) vs the
//! production f32 v3 path on identical synthetic inputs.
//!
//! The bf16-output kernel is byte-for-byte identical to v3 in the inner loop
//! (FMA in f32 inside each simdgroup, simd_sum reduction). The only difference
//! is the final `bfloat()` narrow before the device store. Acceptable drift
//! is bounded by bf16's 7-bit mantissa (≈ 2^-7 ≈ 7.8e-3 relative per element)
//! plus accumulated rounding from RTNE.
//!
//! Acceptance:
//!   - cosine similarity ≥ 0.999  (structural drift would break this)
//!   - relative max err ≤ 1e-2  (1 LSB of bf16 ≈ 7.8e-3, allow some headroom)

use std::sync::Arc;

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

/// Decode bf16 storage bits → f32. bf16 is the high 16 bits of an IEEE-754
/// binary32, so re-flooding the low 16 bits with zero recovers the value.
fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn read_bf16_buffer_as_f32(
    ctx: &lumen_metal::device::MetalContext,
    buf: &lumen_metal::metal::Buffer,
    count: usize,
) -> Vec<f32> {
    let raw: Vec<u16> = ctx.read_buffer::<u16>(buf, count);
    raw.into_iter().map(bf16_bits_to_f32).collect()
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

fn assert_bf16_parity(name: &str, ref_f32: &[f32], bf16_dec: &[f32]) {
    let cos = cosine_similarity(ref_f32, bf16_dec);
    let rel = rel_max_err(ref_f32, bf16_dec);
    let abs = max_abs_err(ref_f32, bf16_dec);
    assert!(
        cos >= 0.999,
        "{name}: cosine {cos} below 0.999 (abs={abs}, rel={rel})"
    );
    assert!(
        rel <= 1e-2,
        "{name}: relative max err {rel} exceeds 1e-2 (cos={cos}, abs={abs})"
    );
}

#[test]
fn mxfp4_bf16_out_v3_parity_production_shapes() {
    let ctx = match MxFp4Context::new() {
        Ok(c) => Arc::new(c),
        Err(_) => return, // No Metal device — skip (CI without GPU).
    };

    // Shapes representative of Qwen3.6-35B-A3B-mxfp4 hot path:
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
        let x = synth_x(batch * ins);
        let weight = Mxfp4Weight::from_host(&ctx.ctx, &packed, &scales, out, ins).unwrap();

        let x_buf = ctx.ctx.buffer_with_data(&x);

        // f32 reference path.
        let y_f32_buf = ctx.ctx.buffer_for::<f32>(batch * out);
        ctx.matmul_zero_copy(&weight, &x_buf, 0, &y_f32_buf, 0, batch)
            .unwrap();
        let y_f32 = ctx.ctx.read_buffer::<f32>(&y_f32_buf, batch * out);

        // bf16 path — output buffer holds 2 bytes/elem.
        let y_bf16_buf = ctx.ctx.buffer_for::<u16>(batch * out);
        ctx.matmul_zero_copy_bf16_out(&weight, &x_buf, 0, &y_bf16_buf, 0, batch)
            .unwrap();
        let y_bf16 = read_bf16_buffer_as_f32(&ctx.ctx, &y_bf16_buf, batch * out);

        assert_bf16_parity(
            &format!("dense `{name}` ({out}x{ins} batch={batch})"),
            &y_f32,
            &y_bf16,
        );
    }
}

/// Tensor-level parity: `Mxfp4Linear::forward` (f32) vs
/// `Mxfp4Linear::forward_bf16_out` (bf16) with the same Candle Tensor input.
/// Validates the full Tensor wrapper path including Metal buffer extraction
/// and Candle DType handling, not just the raw kernel.
#[test]
fn mxfp4_linear_forward_bf16_out_tensor_parity() {
    use candle_core::{DType, Device, Tensor};
    use lumen_metal::mxfp4_linear::Mxfp4Linear;

    let device = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => return, // No Metal device — skip.
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
        let x = Tensor::from_vec(x_vec, (batch, ins), &device).unwrap();

        let y_f32 = linear.forward(&x).unwrap();
        assert_eq!(y_f32.dtype(), DType::F32);
        let y_f32_vec = y_f32.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let y_bf16 = linear.forward_bf16_out(&x).unwrap();
        assert_eq!(
            y_bf16.dtype(),
            DType::BF16,
            "{name}: forward_bf16_out must produce bf16 tensor"
        );
        // Convert bf16 → f32 via Candle's own promotion to validate end-to-end
        // (this exercises the same dtype-cast path downstream model code uses).
        let y_bf16_as_f32 = y_bf16
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_bf16_parity(
            &format!("Mxfp4Linear `{name}` ({out}x{ins} batch={batch})"),
            &y_f32_vec,
            &y_bf16_as_f32,
        );
    }
}

/// Edge case: `out_features` not a multiple of 8. Threadgroups whose simdgroup
/// row index lands past `out_features` must skip the bf16 store the same way
/// the f32 v3 path does.
#[test]
fn mxfp4_bf16_out_v3_parity_non_multiple_of_8() {
    let ctx = match MxFp4Context::new() {
        Ok(c) => Arc::new(c),
        Err(_) => return,
    };
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("out=63", 63, 256, 1),
        ("out=129", 129, 512, 1),
        ("out=251", 251, 256, 1),
    ];
    for &(name, out, ins, batch) in shapes {
        let packed = synth_packed(out, ins, 0xDEADBEEF);
        let scales = synth_scales(out, ins, 0xFEEDFACE);
        let x = synth_x(batch * ins);
        let weight = Mxfp4Weight::from_host(&ctx.ctx, &packed, &scales, out, ins).unwrap();

        let x_buf = ctx.ctx.buffer_with_data(&x);

        let y_f32_buf = ctx.ctx.buffer_for::<f32>(batch * out);
        ctx.matmul_zero_copy(&weight, &x_buf, 0, &y_f32_buf, 0, batch)
            .unwrap();
        let y_f32 = ctx.ctx.read_buffer::<f32>(&y_f32_buf, batch * out);

        let y_bf16_buf = ctx.ctx.buffer_for::<u16>(batch * out);
        ctx.matmul_zero_copy_bf16_out(&weight, &x_buf, 0, &y_bf16_buf, 0, batch)
            .unwrap();
        let y_bf16 = read_bf16_buffer_as_f32(&ctx.ctx, &y_bf16_buf, batch * out);

        assert_bf16_parity(
            &format!("dense `{name}` ({out}x{ins} batch={batch})"),
            &y_f32,
            &y_bf16,
        );
    }
}
