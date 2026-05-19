//! Bit-parity check between the GPU-resident `Affine4Linear` and the CPU
//! `dequant_int4_affine` + dense matmul reference. Validates that the new
//! kernel produces results equivalent to dequantizing
//! to f32 first and then doing the matmul on host — within f32 fast-math drift.
//!
//! The kernel uses fast-math + reordered FMAs, so we tolerate small relative
//! drift (max abs ≤ 1e-3 on these synthetic shapes). For exact arithmetic
//! parity, set `LUMEN_AFFINE4_FORCE_CPU=1` (routes through `matmul_with_weight`
//! which is the same f32 reference).

#![cfg(feature = "turboquant-gpu")]

use std::sync::Arc;

use candle_core::{Device, Tensor};
use lumen_metal::affine4_gpu::{AFFINE4_GROUP_SIZE, Affine4Context, Affine4Weight};
use lumen_metal::affine4_linear::Affine4Linear;
use lumen_model::qwen3_5_moe::loader::debug_dequant_int4_affine;

fn bf16_from_f32(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

/// Hand-built fixture: random nibbles + small bf16 scales + small bf16 biases.
/// Reference path: dequant to f32 → CPU dense matmul.
/// Kernel path: build `Affine4Linear`, run forward on a Metal-resident input.
#[test]
fn affine4_gpu_matches_cpu_dequant_matmul() -> anyhow::Result<()> {
    let ctx = match Affine4Context::new() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("skip: no Metal device or Affine4Context init failed: {e}");
            return Ok(());
        }
    };

    // Shape large enough to exercise multiple groups + multiple rows.
    let out_features = 16;
    let in_features = 256; // 4 groups per row
    let groups_per_row = in_features / AFFINE4_GROUP_SIZE;

    let total_words = out_features * in_features / 8;
    let total_groups = out_features * groups_per_row;

    // Deterministic pseudo-random nibbles via xorshift32. We avoid pulling in `rand`
    // here since this is a leaf integration test.
    let mut state: u32 = 0xCAFEBABE;
    let mut next_u32 = || -> u32 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };

    let packed: Vec<u32> = (0..total_words).map(|_| next_u32()).collect();

    // Scales + biases in a small range so f32 accumulator stays well-behaved.
    let scales: Vec<u16> = (0..total_groups)
        .map(|i| {
            let s = 0.01 + 0.005 * ((i % 7) as f32);
            bf16_from_f32(s)
        })
        .collect();
    let biases: Vec<u16> = (0..total_groups)
        .map(|i| {
            let b = -0.05 + 0.01 * ((i % 5) as f32);
            bf16_from_f32(b)
        })
        .collect();

    // Reference: CPU dequant to f32, manual matmul.
    let w_dequant =
        debug_dequant_int4_affine(&packed, &scales, &biases, AFFINE4_GROUP_SIZE).unwrap();
    assert_eq!(w_dequant.len(), out_features * in_features);

    // Random input (small range to keep magnitudes bounded).
    let x: Vec<f32> = (0..in_features)
        .map(|_| {
            let r = (next_u32() as f32) / (u32::MAX as f32);
            -1.0 + 2.0 * r
        })
        .collect();

    let mut y_ref = vec![0f32; out_features];
    for r in 0..out_features {
        let row = &w_dequant[r * in_features..(r + 1) * in_features];
        let mut acc = 0.0f32;
        for k in 0..in_features {
            acc += row[k] * x[k];
        }
        y_ref[r] = acc;
    }

    // GPU path through `Affine4Linear`.
    let weight = Affine4Weight::from_host(
        &ctx.ctx,
        &packed,
        &scales,
        &biases,
        out_features,
        in_features,
    )?;
    let lin = Affine4Linear::new(weight, None, Arc::clone(&ctx));

    // Host-staged path (CPU fallback in `Affine4Linear::forward`): build x as a
    // CPU-side Candle tensor. The forward routes through `matmul_with_weight`
    // which dispatches the GPU kernel via a fresh command buffer.
    let device = Device::Cpu;
    let x_t = Tensor::from_vec(x.clone(), (1, in_features), &device)?;
    let y_t = lin.forward(&x_t)?;
    let y_gpu_2d = y_t.to_vec2::<f32>()?;
    assert_eq!(y_gpu_2d.len(), 1);
    let y_gpu = &y_gpu_2d[0];
    assert_eq!(y_gpu.len(), out_features);

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for r in 0..out_features {
        let a = y_ref[r];
        let g = y_gpu[r];
        let abs_err = (a - g).abs();
        max_abs = max_abs.max(abs_err);
        let denom = a.abs().max(1e-6);
        max_rel = max_rel.max(abs_err / denom);
    }

    println!(
        "affine4 GPU vs CPU-dequant: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}, \
         out=[{out_features}], in=[{in_features}]"
    );

    // Tolerance: kernel uses fast-math + reduction reordering. f32 ops on this
    // shape (in=256, ~256 FMAs per output) drift bounded by ~1e-3 absolute.
    assert!(
        max_abs < 1e-3,
        "max_abs={max_abs:.3e} exceeds 1e-3 tolerance"
    );
    Ok(())
}

/// R1 sanity: the fused RmsNorm kernel produces results equivalent to manual
/// (Candle) RmsNorm followed by the standard f32 v3 matmul. Validates the new
/// `affine4_matmul_f32_v3_rmsnorm` kernel before wiring it into model code.
#[test]
fn affine4_rmsnorm_fused_matches_manual_rmsnorm_then_matmul() -> anyhow::Result<()> {
    use lumen_metal::affine4_linear::Affine4Linear;

    let ctx = match Affine4Context::new() {
        Ok(c) => Arc::new(c),
        Err(_) => return Ok(()),
    };

    // 27B-Dense-shaped: in=5120, out=128 (small for test). 80 groups per row.
    let out_features = 128;
    let in_features = 5120;
    let groups_per_row = in_features / AFFINE4_GROUP_SIZE;
    let total_words = out_features * in_features / 8;
    let total_groups = out_features * groups_per_row;

    let mut state: u32 = 0xBEEF_F00D;
    let mut next_u32 = || -> u32 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let packed: Vec<u32> = (0..total_words).map(|_| next_u32()).collect();
    let scales: Vec<u16> = (0..total_groups)
        .map(|i| {
            let s = 0.005 + 0.002 * ((i % 11) as f32);
            bf16_from_f32(s)
        })
        .collect();
    let biases: Vec<u16> = (0..total_groups)
        .map(|i| {
            let b = -0.02 + 0.005 * ((i % 7) as f32);
            bf16_from_f32(b)
        })
        .collect();

    let weight = Affine4Weight::from_host(
        &ctx.ctx,
        &packed,
        &scales,
        &biases,
        out_features,
        in_features,
    )?;
    let lin = Affine4Linear::new(weight, None, Arc::clone(&ctx));

    // Build x_raw and rms_weight as Candle CPU tensors (forward path will
    // upload via Metal-aware fallback). Both are 1-row activations to mimic
    // single-token decode.
    let x_raw_vec: Vec<f32> = (0..in_features)
        .map(|_| {
            let r = (next_u32() as f32) / (u32::MAX as f32);
            -1.0 + 2.0 * r
        })
        .collect();
    let rms_w_vec: Vec<f32> = (0..in_features)
        .map(|i| 0.95 + 0.05 * ((i % 13) as f32) / 13.0)
        .collect();
    let rms_eps = 1e-6f32;

    let device = candle_core::Device::Cpu;
    let x_raw = candle_core::Tensor::from_vec(x_raw_vec.clone(), (1, in_features), &device)?;
    let rms_w = candle_core::Tensor::from_vec(rms_w_vec.clone(), in_features, &device)?;

    // Reference: manual RmsNorm + standard forward.
    let sq = x_raw.sqr()?;
    let mean_sq = sq.mean_keepdim(1)?;
    let inv_rms = (mean_sq + rms_eps as f64)?.sqrt()?.recip()?;
    let x_normed = x_raw.broadcast_mul(&inv_rms)?.broadcast_mul(&rms_w)?;
    let y_ref = lin.forward(&x_normed)?;
    let y_ref_vec = y_ref.to_vec2::<f32>()?[0].clone();

    // Fused: forward_with_rmsnorm.
    let y_fused = lin.forward_with_rmsnorm(&x_raw, &rms_w, rms_eps)?;
    let y_fused_vec = y_fused.to_vec2::<f32>()?[0].clone();

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for i in 0..out_features {
        let a = y_ref_vec[i];
        let g = y_fused_vec[i];
        let abs_err = (a - g).abs();
        max_abs = max_abs.max(abs_err);
        let denom = a.abs().max(1e-6);
        max_rel = max_rel.max(abs_err / denom);
    }
    println!(
        "affine4 R1 fused vs manual: max_abs={max_abs:.3e} max_rel={max_rel:.3e}, in={in_features}, out={out_features}"
    );
    assert!(
        max_abs < 5e-3,
        "R1 fused kernel max_abs={max_abs:.3e} exceeds 5e-3 tolerance — kernel bug suspected"
    );
    Ok(())
}

/// R4 parity: the v3-tiled kernel (used when `in_features` exceeds the v3
/// single-shot TG memory budget) produces results equivalent to the CPU
/// dequant + matmul reference. Uses `in=16384 > 8192` so `pick_tile_for_in`
/// engages the tiled path (n_chunks=2, tile=8192).
#[test]
fn affine4_tiled_matches_cpu_dequant_matmul() -> anyhow::Result<()> {
    use lumen_metal::affine4_gpu::pick_tile_for_in;

    let ctx = match Affine4Context::new() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("skip: no Metal device or Affine4Context init failed: {e}");
            return Ok(());
        }
    };

    // Shape that forces the tiled path.
    let out_features = 20;
    let in_features = 16384; // 256 groups per row, > 8192 → tiled path
    let (tile_in, n_chunks) = pick_tile_for_in(in_features).expect("tile expected for in=16384");
    assert_eq!(tile_in * n_chunks, in_features);
    assert!(tile_in <= 8192);

    let groups_per_row = in_features / AFFINE4_GROUP_SIZE;
    let total_words = out_features * in_features / 8;
    let total_groups = out_features * groups_per_row;

    let mut state: u32 = 0xDEAD_F00D;
    let mut next_u32 = || -> u32 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let packed: Vec<u32> = (0..total_words).map(|_| next_u32()).collect();

    let scales: Vec<u16> = (0..total_groups)
        .map(|i| {
            let s = 0.003 + 0.001 * ((i % 7) as f32);
            bf16_from_f32(s)
        })
        .collect();
    let biases: Vec<u16> = (0..total_groups)
        .map(|i| {
            let b = -0.01 + 0.002 * ((i % 5) as f32);
            bf16_from_f32(b)
        })
        .collect();

    // Reference: CPU dequant → manual matmul.
    let w_dequant =
        debug_dequant_int4_affine(&packed, &scales, &biases, AFFINE4_GROUP_SIZE).unwrap();
    assert_eq!(w_dequant.len(), out_features * in_features);

    let x: Vec<f32> = (0..in_features)
        .map(|_| {
            let r = (next_u32() as f32) / (u32::MAX as f32);
            -1.0 + 2.0 * r
        })
        .collect();

    let mut y_ref = vec![0f32; out_features];
    for r in 0..out_features {
        let row = &w_dequant[r * in_features..(r + 1) * in_features];
        let mut acc = 0.0f32;
        for k in 0..in_features {
            acc += row[k] * x[k];
        }
        y_ref[r] = acc;
    }

    let weight = Affine4Weight::from_host(
        &ctx.ctx,
        &packed,
        &scales,
        &biases,
        out_features,
        in_features,
    )?;
    let lin = Affine4Linear::new(weight, None, Arc::clone(&ctx));

    let device = Device::Cpu;
    let x_t = Tensor::from_vec(x.clone(), (1, in_features), &device)?;
    let y_t = lin.forward(&x_t)?;
    let y_gpu_2d = y_t.to_vec2::<f32>()?;
    assert_eq!(y_gpu_2d.len(), 1);
    let y_gpu = &y_gpu_2d[0];
    assert_eq!(y_gpu.len(), out_features);

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for r in 0..out_features {
        let a = y_ref[r];
        let g = y_gpu[r];
        let abs_err = (a - g).abs();
        max_abs = max_abs.max(abs_err);
        let denom = a.abs().max(1e-6);
        max_rel = max_rel.max(abs_err / denom);
    }
    println!(
        "affine4 tiled vs CPU-dequant: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}, \
         in={in_features} (tile={tile_in}, n_chunks={n_chunks}), out={out_features}"
    );

    // Tolerance: 16384-wide reduction → ~16x more terms vs the 256-wide test.
    // f32 fast-math drift bounded by ~5e-2 absolute on this shape.
    assert!(
        max_abs < 5e-2,
        "tiled max_abs={max_abs:.3e} exceeds 5e-2 tolerance"
    );
    Ok(())
}

/// MLX-pattern `affine4_qmv_fast` parity vs CPU dequant matmul.
/// Different reduction order and pre-scaling math → fast-math drift,
/// so allow somewhat looser tolerance than v3 (still well within model
/// quality range).
#[test]
fn affine4_qmv_fast_matches_cpu_dequant() -> anyhow::Result<()> {
    let ctx = match Affine4Context::new() {
        Ok(c) => Arc::new(c),
        Err(_) => return Ok(()),
    };

    // Shapes must satisfy qmv_fast constraints: in % 512 == 0, out % 8 == 0.
    // Use 27B-Dense-like shapes.
    let cases = [(8u32, 512u32), (16, 1024), (40, 5120), (24, 22528)];

    for (out_features, in_features) in cases {
        let out_features = out_features as usize;
        let in_features = in_features as usize;
        assert!(Affine4Context::qmv_fast_supports(in_features, out_features));

        let groups_per_row = in_features / AFFINE4_GROUP_SIZE;
        let total_words = out_features * in_features / 8;
        let total_groups = out_features * groups_per_row;

        let mut state: u32 = 0x12345 ^ (out_features as u32);
        let mut next_u32 = || -> u32 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let packed: Vec<u32> = (0..total_words).map(|_| next_u32()).collect();
        let scales: Vec<u16> = (0..total_groups)
            .map(|i| bf16_from_f32(0.005 + 0.001 * ((i % 11) as f32)))
            .collect();
        let biases: Vec<u16> = (0..total_groups)
            .map(|i| bf16_from_f32(-0.02 + 0.005 * ((i % 7) as f32)))
            .collect();

        let w_dequant =
            debug_dequant_int4_affine(&packed, &scales, &biases, AFFINE4_GROUP_SIZE).unwrap();

        let x: Vec<f32> = (0..in_features)
            .map(|_| {
                let r = (next_u32() as f32) / (u32::MAX as f32);
                -1.0 + 2.0 * r
            })
            .collect();

        let mut y_ref = vec![0f32; out_features];
        for r in 0..out_features {
            let row = &w_dequant[r * in_features..(r + 1) * in_features];
            let mut acc = 0.0f32;
            for k in 0..in_features {
                acc += row[k] * x[k];
            }
            y_ref[r] = acc;
        }

        let weight = Affine4Weight::from_host(
            &ctx.ctx,
            &packed,
            &scales,
            &biases,
            out_features,
            in_features,
        )?;
        let lin = Affine4Linear::new(weight, None, Arc::clone(&ctx));

        // Force qmv_fast path via Metal-resident input.
        let device = candle_core::Device::new_metal(0)?;
        let x_t = Tensor::from_vec(x.clone(), (1, in_features), &device)?;
        let y = lin.forward(&x_t)?;
        let y_gpu = y.to_vec2::<f32>()?[0].clone();

        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for r in 0..out_features {
            let a = y_ref[r];
            let g = y_gpu[r];
            let abs_err = (a - g).abs();
            max_abs = max_abs.max(abs_err);
            let denom = a.abs().max(1e-6);
            max_rel = max_rel.max(abs_err / denom);
        }
        println!(
            "qmv_fast vs CPU @ out={out_features}, in={in_features}: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}"
        );
        // Tolerance: in=22528 means ~22K FMAs per output → f32 fast-math
        // drift bounded ~ 1e-1 absolute on synthetic inputs.
        assert!(
            max_abs < 1.0,
            "qmv_fast max_abs={max_abs:.3e} exceeds 1.0 tolerance for shape (out={out_features}, in={in_features})"
        );
    }

    Ok(())
}

/// R4+residual: tiled forward_with_residual_f32 must equal manual
/// (forward + broadcast_add) within fast-math drift. Validates the
/// `affine4_reduce_chunks_f32_residual` kernel under in > 8192 path.
#[test]
fn affine4_tiled_residual_matches_manual() -> anyhow::Result<()> {
    use lumen_metal::affine4_gpu::pick_tile_for_in;

    let ctx = match Affine4Context::new() {
        Ok(c) => Arc::new(c),
        Err(_) => return Ok(()),
    };

    // Force tiled path with a clean tile.
    let out_features = 32;
    let in_features = 16384;
    assert!(pick_tile_for_in(in_features).is_some());

    let groups_per_row = in_features / AFFINE4_GROUP_SIZE;
    let total_words = out_features * in_features / 8;
    let total_groups = out_features * groups_per_row;

    let mut state: u32 = 0xBADD_F00D;
    let mut next_u32 = || -> u32 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let packed: Vec<u32> = (0..total_words).map(|_| next_u32()).collect();
    let scales: Vec<u16> = (0..total_groups)
        .map(|i| bf16_from_f32(0.002 + 0.001 * ((i % 7) as f32)))
        .collect();
    let biases: Vec<u16> = (0..total_groups)
        .map(|i| bf16_from_f32(-0.005 + 0.001 * ((i % 5) as f32)))
        .collect();

    let weight = Affine4Weight::from_host(
        &ctx.ctx,
        &packed,
        &scales,
        &biases,
        out_features,
        in_features,
    )?;
    let lin = Affine4Linear::new(weight, None, Arc::clone(&ctx));

    // Build x and residual on Metal.
    let device = candle_core::Device::new_metal(0)?;
    let x_vec: Vec<f32> = (0..in_features)
        .map(|_| {
            let r = (next_u32() as f32) / (u32::MAX as f32);
            -1.0 + 2.0 * r
        })
        .collect();
    let res_vec: Vec<f32> = (0..out_features)
        .map(|_| {
            let r = (next_u32() as f32) / (u32::MAX as f32);
            -0.5 + r
        })
        .collect();
    let x_t = Tensor::from_vec(x_vec, (1, in_features), &device)?;
    let res_t = Tensor::from_vec(res_vec, (1, out_features), &device)?;

    // Manual: forward + broadcast_add.
    let y_manual = lin.forward(&x_t)?.broadcast_add(&res_t)?;
    let y_manual_v = y_manual.to_vec2::<f32>()?[0].clone();

    // Fused: forward_with_residual_f32 → tiled+residual reduction kernel.
    let y_fused = lin.forward_with_residual_f32(&x_t, &res_t)?;
    let y_fused_v = y_fused.to_vec2::<f32>()?[0].clone();

    let mut max_abs = 0.0f32;
    for i in 0..out_features {
        max_abs = max_abs.max((y_manual_v[i] - y_fused_v[i]).abs());
    }
    println!(
        "affine4 tiled+residual vs manual: max_abs={max_abs:.3e}, in={in_features}, out={out_features}"
    );
    assert!(
        max_abs < 1e-4,
        "tiled+residual max_abs={max_abs:.3e} exceeds 1e-4 tolerance"
    );
    Ok(())
}

/// Smaller-scale check: verify a single group with bias-only weights produces
/// `y[r] = bias * sum(x)` exactly. No nibble contribution → no rounding drift
/// from the dequant arithmetic, only the f32 accumulator drift.
#[test]
fn affine4_bias_only_dequant_matches_analytic() -> anyhow::Result<()> {
    let ctx = match Affine4Context::new() {
        Ok(c) => Arc::new(c),
        Err(_) => return Ok(()),
    };

    let out_features = 1;
    let in_features = 64;
    let packed = vec![0xFFFF_FFFFu32; out_features * in_features / 8];
    let scales = vec![bf16_from_f32(0.0); out_features];
    let biases = vec![bf16_from_f32(0.5); out_features];

    let w = Affine4Weight::from_host(
        &ctx.ctx,
        &packed,
        &scales,
        &biases,
        out_features,
        in_features,
    )?;
    let lin = Affine4Linear::new(w, None, Arc::clone(&ctx));

    let x = vec![1.0f32; in_features];
    let device = Device::Cpu;
    let x_t = Tensor::from_vec(x.clone(), (1, in_features), &device)?;
    let y = lin.forward(&x_t)?.to_vec2::<f32>()?;
    let got = y[0][0];

    // bias-only: each weight = 0 * nibble + 0.5 = 0.5. y = sum(0.5 * x_k) = 0.5 * 64 = 32.
    let expected = 0.5 * (in_features as f32);
    assert!(
        (got - expected).abs() < 1e-4,
        "bias-only: expected {expected}, got {got}"
    );
    Ok(())
}
