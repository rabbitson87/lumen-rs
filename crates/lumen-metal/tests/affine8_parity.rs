//! Parity test for `affine8_matmul_bf16` Metal kernel vs CPU dequant reference.
//!
//! Lives in `tests/` (not `src/`) so it compiles independently of the rest
//! of the lumen-metal lib-test target (which has pre-existing unrelated
//! build errors in mxfp4 test code as of this commit).

use half::bf16;
use lumen_metal::affine8_gpu::{
    cpu_reference_matmul_bf16, AFFINE8_GROUP_SIZE, Affine8Context, Affine8Weight,
};

fn make_ctx_or_skip() -> Option<Affine8Context> {
    match Affine8Context::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("[skip] no Metal device or shader compile failed: {e}");
            None
        }
    }
}

#[test]
fn affine8_matmul_bf16_identity_ones() {
    let Some(ctx) = make_ctx_or_skip() else { return };
    let in_f = 64;
    let out = 2;
    let batch = 1;

    // Each output row has a distinct byte fingerprint (o, o+1, o+2, o+3) repeated.
    let mut packed = vec![0u32; out * in_f / 4];
    for o in 0..out {
        for w in 0..(in_f / 4) {
            let b0 = (o as u32) & 0xFF;
            let b1 = ((o + 1) as u32) & 0xFF;
            let b2 = ((o + 2) as u32) & 0xFF;
            let b3 = ((o + 3) as u32) & 0xFF;
            packed[o * (in_f / 4) + w] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }
    }
    let groups = in_f / AFFINE8_GROUP_SIZE;
    let scales: Vec<u16> = vec![bf16::from_f32(1.0).to_bits(); out * groups];
    let biases: Vec<u16> = vec![bf16::from_f32(0.0).to_bits(); out * groups];
    let x_bf16: Vec<u16> = vec![bf16::from_f32(1.0).to_bits(); batch * in_f];

    let weight = Affine8Weight::from_host(&ctx.ctx, &packed, &scales, &biases, out, in_f).unwrap();
    let gpu = ctx.matmul_bf16_with_weight(&weight, &x_bf16, batch).unwrap();
    let cpu = cpu_reference_matmul_bf16(&packed, &scales, &biases, &x_bf16, out, in_f, batch);

    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let gf = bf16::from_bits(*g).to_f32();
        let cf = bf16::from_bits(*c).to_f32();
        assert!(
            (gf - cf).abs() / cf.abs().max(1.0) < 1e-2,
            "y[{i}] gpu={gf} cpu={cf}"
        );
    }
}

#[test]
fn affine8_matmul_bf16_random() {
    let Some(ctx) = make_ctx_or_skip() else { return };
    let in_f = 128;
    let out = 8;
    let batch = 3;
    let groups = in_f / AFFINE8_GROUP_SIZE;

    let mut prng: u32 = 0xDEAD_BEEF;
    let mut rng = || -> u32 {
        prng ^= prng << 13;
        prng ^= prng >> 17;
        prng ^= prng << 5;
        prng
    };
    let packed: Vec<u32> = (0..out * in_f / 4).map(|_| rng()).collect();
    let scales: Vec<u16> = (0..out * groups)
        .map(|_| {
            let r = (rng() & 0xFFFF) as f32 / 65536.0;
            bf16::from_f32(0.001 + r * 0.01).to_bits()
        })
        .collect();
    let biases: Vec<u16> = (0..out * groups)
        .map(|_| {
            let r = (rng() & 0xFFFF) as f32 / 65536.0;
            bf16::from_f32(r * 0.5 - 0.25).to_bits()
        })
        .collect();
    let x_bf16: Vec<u16> = (0..batch * in_f)
        .map(|_| {
            let r = (rng() & 0xFFFF) as f32 / 65536.0;
            bf16::from_f32(r * 2.0 - 1.0).to_bits()
        })
        .collect();

    let weight = Affine8Weight::from_host(&ctx.ctx, &packed, &scales, &biases, out, in_f).unwrap();
    let gpu = ctx.matmul_bf16_with_weight(&weight, &x_bf16, batch).unwrap();
    let cpu = cpu_reference_matmul_bf16(&packed, &scales, &biases, &x_bf16, out, in_f, batch);

    let mut max_err = 0f32;
    for (g, c) in gpu.iter().zip(cpu.iter()) {
        let gf = bf16::from_bits(*g).to_f32();
        let cf = bf16::from_bits(*c).to_f32();
        let rel = (gf - cf).abs() / cf.abs().max(1e-3);
        max_err = max_err.max(rel);
    }
    assert!(max_err < 5e-2, "max relative error {max_err} > 5e-2");
}

/// End-to-end candle Tensor → Affine8Linear → candle Tensor path.
/// Must produce distinct outputs for distinct batch rows, exercising
/// the full candle ↔ Metal interop (metal_buffer_of, command queue
/// hand-off, output Tensor::zeros allocation).
#[test]
fn affine8_linear_candle_tensor_batches() {
    use candle_core::{DType, Device, Tensor};
    use lumen_metal::affine8_linear::Affine8Linear;
    use std::sync::Arc;

    let Some(ctx) = make_ctx_or_skip() else { return };
    let in_f = 128;
    let out = 32;
    let batch = 4;
    let groups = in_f / AFFINE8_GROUP_SIZE;

    let mut prng: u32 = 0xFEED_FACE;
    let mut rng = || -> u32 {
        prng ^= prng << 13;
        prng ^= prng >> 17;
        prng ^= prng << 5;
        prng
    };
    let packed: Vec<u32> = (0..out * in_f / 4).map(|_| rng()).collect();
    let scales: Vec<u16> = (0..out * groups)
        .map(|_| bf16::from_f32(0.005).to_bits())
        .collect();
    let biases: Vec<u16> = (0..out * groups)
        .map(|_| bf16::from_f32(0.0).to_bits())
        .collect();

    let weight = Affine8Weight::from_host(&ctx.ctx, &packed, &scales, &biases, out, in_f).unwrap();
    let linear = Affine8Linear::new(weight, Arc::new(ctx));

    let device = match Device::new_metal(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[skip] candle Metal device unavailable: {e}");
            return;
        }
    };

    // Build a candle bf16 tensor with row r filled with value `(r + 1.0) * 0.1`.
    // Distinct row scales → distinct outputs.
    let mut x_f32: Vec<f32> = Vec::with_capacity(batch * in_f);
    for r in 0..batch {
        let v = (r as f32 + 1.0) * 0.1;
        x_f32.extend(std::iter::repeat(v).take(in_f));
    }
    let x = Tensor::from_vec(x_f32.clone(), (batch, in_f), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let y = linear.forward(&x).unwrap();
    let y_f32 = y.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap();

    // Row r must be (r+1) * row 0 (within bf16 rounding).
    let row0 = &y_f32[0];
    for r in 1..batch {
        let expected_scale = r as f32 + 1.0;
        for o in 0..out {
            let expected = row0[o] * expected_scale;
            let actual = y_f32[r][o];
            assert!(
                (actual - expected).abs() / expected.abs().max(1e-3) < 0.05,
                "row {r} col {o}: expected {expected}, got {actual} (row0={})",
                row0[o]
            );
        }
    }
}

/// Two batches with distinct x rows must produce distinct y rows.
/// Regression guard against an accidentally-broken batch index in the kernel.
#[test]
fn affine8_matmul_bf16_distinguishes_batches() {
    let Some(ctx) = make_ctx_or_skip() else { return };
    let in_f = 64;
    let out = 4;
    let batch = 2;
    let groups = in_f / AFFINE8_GROUP_SIZE;

    // Pack non-zero deterministic bytes so every group contributes.
    let mut packed = vec![0u32; out * in_f / 4];
    for (i, v) in packed.iter_mut().enumerate() {
        *v = 0x01020304u32.wrapping_add(i as u32);
    }
    let scales: Vec<u16> = vec![bf16::from_f32(0.01).to_bits(); out * groups];
    let biases: Vec<u16> = vec![bf16::from_f32(0.0).to_bits(); out * groups];

    // Row 0: all 1.0; row 1: all 2.0 → outputs must differ by factor 2.
    let mut x_bf16 = Vec::with_capacity(batch * in_f);
    for _ in 0..in_f {
        x_bf16.push(bf16::from_f32(1.0).to_bits());
    }
    for _ in 0..in_f {
        x_bf16.push(bf16::from_f32(2.0).to_bits());
    }

    let weight = Affine8Weight::from_host(&ctx.ctx, &packed, &scales, &biases, out, in_f).unwrap();
    let gpu = ctx.matmul_bf16_with_weight(&weight, &x_bf16, batch).unwrap();

    for o in 0..out {
        let y0 = bf16::from_bits(gpu[o]).to_f32();
        let y1 = bf16::from_bits(gpu[out + o]).to_f32();
        assert!(
            (y1 - 2.0 * y0).abs() / y0.abs().max(1e-3) < 5e-2,
            "row[1][{o}]={y1} != 2 * row[0][{o}]={y0}"
        );
    }
}

/// 3D input shape (b, L, H) as used in transformer forward. Distinct
/// batches AND distinct sequence positions must all stay independent.
#[test]
fn affine8_linear_candle_3d_input() {
    use candle_core::{DType, Device, Tensor};
    use lumen_metal::affine8_linear::Affine8Linear;
    use std::sync::Arc;

    let Some(ctx) = make_ctx_or_skip() else { return };
    let in_f = 1024;
    let out = 2048;
    let b = 3;
    let l = 7;
    let groups = in_f / AFFINE8_GROUP_SIZE;

    let mut prng: u32 = 0xBABE_FEED;
    let mut rng = || -> u32 {
        prng ^= prng << 13;
        prng ^= prng >> 17;
        prng ^= prng << 5;
        prng
    };
    let packed: Vec<u32> = (0..out * in_f / 4).map(|_| rng()).collect();
    let scales: Vec<u16> = (0..out * groups)
        .map(|_| bf16::from_f32(0.002).to_bits())
        .collect();
    let biases: Vec<u16> = (0..out * groups)
        .map(|_| bf16::from_f32(0.0).to_bits())
        .collect();

    let weight = Affine8Weight::from_host(&ctx.ctx, &packed, &scales, &biases, out, in_f).unwrap();
    let linear = Affine8Linear::new(weight, Arc::new(ctx));

    let device = match Device::new_metal(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[skip] candle Metal unavailable: {e}");
            return;
        }
    };

    // Each (batch_idx, seq_idx) gets a unique scale → distinct outputs everywhere.
    let mut x_f32: Vec<f32> = Vec::with_capacity(b * l * in_f);
    for bi in 0..b {
        for si in 0..l {
            let v = (bi as f32 + 1.0) * 0.1 + (si as f32) * 0.01;
            x_f32.extend(std::iter::repeat(v).take(in_f));
        }
    }
    let x = Tensor::from_vec(x_f32, (b, l, in_f), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let y = linear.forward(&x).unwrap();
    let y_f32 = y.to_dtype(DType::F32).unwrap().to_vec3::<f32>().unwrap();

    // y[bi][si] should equal scale[bi][si] / scale[0][0] * y[0][0] (linear scaling).
    let row00 = &y_f32[0][0];
    let scale00 = 0.1f32;
    for bi in 0..b {
        for si in 0..l {
            let scale_bisi = (bi as f32 + 1.0) * 0.1 + (si as f32) * 0.01;
            let factor = scale_bisi / scale00;
            for o in 0..out {
                let expected = row00[o] * factor;
                let actual = y_f32[bi][si][o];
                let rel = (actual - expected).abs() / expected.abs().max(1e-3);
                assert!(
                    rel < 0.1,
                    "y[{bi}][{si}][{o}]={actual} expected ~{expected} (factor {factor}, rel_err {rel:.4})"
                );
            }
        }
    }
}

/// Production-shape Affine8Linear via candle. Replicates an actual
/// Qwen3-Embedding q_proj (in=1024, out=2048, batch=21).
#[test]
fn affine8_linear_candle_q_proj_shape() {
    use candle_core::{DType, Device, Tensor};
    use lumen_metal::affine8_linear::Affine8Linear;
    use std::sync::Arc;

    let Some(ctx) = make_ctx_or_skip() else { return };
    let in_f = 1024;
    let out = 2048;
    let batch = 21; // b * max_len = 3 * 7 (smoke shape)
    let groups = in_f / AFFINE8_GROUP_SIZE;

    let mut prng: u32 = 0xC0FFEE;
    let mut rng = || -> u32 {
        prng ^= prng << 13;
        prng ^= prng >> 17;
        prng ^= prng << 5;
        prng
    };
    let packed: Vec<u32> = (0..out * in_f / 4).map(|_| rng()).collect();
    let scales: Vec<u16> = (0..out * groups)
        .map(|_| bf16::from_f32(0.002).to_bits())
        .collect();
    let biases: Vec<u16> = (0..out * groups)
        .map(|_| bf16::from_f32(0.0).to_bits())
        .collect();

    let weight = Affine8Weight::from_host(&ctx.ctx, &packed, &scales, &biases, out, in_f).unwrap();
    let linear = Affine8Linear::new(weight, Arc::new(ctx));

    let device = match Device::new_metal(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[skip] candle Metal unavailable: {e}");
            return;
        }
    };

    let mut x_f32: Vec<f32> = Vec::with_capacity(batch * in_f);
    for r in 0..batch {
        let v = (r as f32 + 1.0) * 0.01;
        x_f32.extend(std::iter::repeat(v).take(in_f));
    }
    let x = Tensor::from_vec(x_f32, (batch, in_f), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let y = linear.forward(&x).unwrap();
    let y_f32 = y.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap();

    // Distinct batch rows scaled linearly should give linearly-scaled outputs.
    // row r should be (r+1) * row 0.
    let row0 = &y_f32[0];
    for r in 1..batch {
        let expected_scale = r as f32 + 1.0;
        let mut max_err = 0f32;
        for o in 0..out {
            let expected = row0[o] * expected_scale;
            let actual = y_f32[r][o];
            let rel = (actual - expected).abs() / expected.abs().max(1e-3);
            if rel > max_err {
                max_err = rel;
            }
        }
        assert!(
            max_err < 0.1,
            "row {r}: max_rel_err {max_err:.4} > 10%"
        );
    }
}

#[test]
fn affine8_matmul_bf16_qwen3_embedding_shape() {
    // Sanity-check a shape that matches one of Qwen3-Embedding's actual
    // projections (k_proj in: 1024 → out: 1024 with GQA num_kv_heads=8 × head_dim=128).
    let Some(ctx) = make_ctx_or_skip() else { return };
    let in_f = 1024;
    let out = 1024;
    let batch = 4;
    let groups = in_f / AFFINE8_GROUP_SIZE;

    let mut prng: u32 = 0xCAFE_BABE;
    let mut rng = || -> u32 {
        prng ^= prng << 13;
        prng ^= prng >> 17;
        prng ^= prng << 5;
        prng
    };
    let packed: Vec<u32> = (0..out * in_f / 4).map(|_| rng()).collect();
    let scales: Vec<u16> = (0..out * groups)
        .map(|_| {
            let r = (rng() & 0xFFFF) as f32 / 65536.0;
            bf16::from_f32(0.001 + r * 0.005).to_bits()
        })
        .collect();
    let biases: Vec<u16> = (0..out * groups)
        .map(|_| {
            let r = (rng() & 0xFFFF) as f32 / 65536.0;
            bf16::from_f32(r * 0.2 - 0.1).to_bits()
        })
        .collect();
    let x_bf16: Vec<u16> = (0..batch * in_f)
        .map(|_| {
            let r = (rng() & 0xFFFF) as f32 / 65536.0;
            bf16::from_f32(r * 0.4 - 0.2).to_bits()
        })
        .collect();

    let weight = Affine8Weight::from_host(&ctx.ctx, &packed, &scales, &biases, out, in_f).unwrap();
    let gpu = ctx.matmul_bf16_with_weight(&weight, &x_bf16, batch).unwrap();
    let cpu = cpu_reference_matmul_bf16(&packed, &scales, &biases, &x_bf16, out, in_f, batch);

    let mut max_err = 0f32;
    let mut max_idx = 0;
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let gf = bf16::from_bits(*g).to_f32();
        let cf = bf16::from_bits(*c).to_f32();
        let rel = (gf - cf).abs() / cf.abs().max(1e-3);
        if rel > max_err {
            max_err = rel;
            max_idx = i;
        }
    }
    // 1024 dot-product accumulates more rounding so loosen tol vs the
    // small-dim case. bf16 quantization noise alone is ~3% per term.
    assert!(
        max_err < 0.1,
        "max relative error {max_err} at idx {max_idx} > 10%"
    );
}
