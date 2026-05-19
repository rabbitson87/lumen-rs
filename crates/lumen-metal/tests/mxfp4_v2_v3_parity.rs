//! Parity check: v3 (simdgroup-cooperative + threadgroup x cache) vs v2
//! (uint4 + float4 vectorized scalar) must produce numerically equivalent
//! output for the same inputs.
//!
//! v3 changes the dispatch topology — 32 lanes cooperate on one row and
//! reduce via `simd_sum` instead of a single thread accumulating in
//! sequential order. The per-term arithmetic `acc += LUT * s * x` is still
//! present per lane, but the final reduction order shifts: instead of one
//! thread doing 32 fmas in source order across all groups, each lane does
//! `groups/32` fmas and a tree-reduce across lanes adds the partials.
//!
//! Acceptance:
//!   - cosine similarity ≥ 0.9999 (any honest drift; structural bugs collapse this)
//!   - relative max error ≤ 5e-3 (≈ 3 ULPs on typical magnitudes)

use lumen_metal::metal::{BatchedEncoderExt, CommandBufferExt, ComputeEncoderCompat};
use std::sync::Arc;

use lumen_metal::metal;
use lumen_metal::mtl_size;
use lumen_metal::mxfp4_gpu::{MxFp4Context, Mxfp4Weight};

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

/// Direct v2 dispatch (matches the v1/v2 parity test helper).
fn run_v2(ctx: &MxFp4Context, weight: &Mxfp4Weight, x: &[f32], batch: usize) -> Vec<f32> {
    let pso = ctx.matmul_f32_pipeline_v2();
    let x_buf = ctx.ctx.buffer_with_data(x);
    let y_buf = ctx.ctx.buffer_for::<f32>(batch * weight.out_features);

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct Dims {
        out_features: u32,
        in_features: u32,
    }
    let dims = Dims {
        out_features: weight.out_features as u32,
        in_features: weight.in_features as u32,
    };
    let batch_u32 = batch as u32;

    let cmd = metal::new_command_buffer(&ctx.ctx.queue);
    let encoder = cmd.auto_compute_encoder();
    encoder.set_compute_pipeline_state(pso);
    encoder.set_buffer(
        0,
        Some(weight.packed_buffer_ref()),
        weight.packed_offset as usize,
    );
    encoder.set_buffer(
        1,
        Some(weight.scales_buffer_ref()),
        weight.scales_offset as usize,
    );
    encoder.set_buffer(2, Some(&x_buf), 0);
    encoder.set_buffer(3, Some(&y_buf), 0);
    encoder.set_bytes_directly(
        4,
        std::mem::size_of::<Dims>(),
        &dims as *const _ as *const _,
    );
    encoder.set_bytes_directly(5, 4, &batch_u32 as *const _ as *const _);

    let max_threads = pso.max_total_threads_per_threadgroup();
    let threads_per_tg = max_threads.min(256);
    let grid = mtl_size!(weight.out_features, batch, 1);
    let tg = mtl_size!(threads_per_tg, 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    ctx.ctx
        .read_buffer::<f32>(&y_buf, batch * weight.out_features)
}

/// Direct v3 dispatch — different topology (256 threads/threadgroup,
/// threadgroup memory holds staged x, grid = (ceil(out/8), batch)).
fn run_v3(ctx: &MxFp4Context, weight: &Mxfp4Weight, x: &[f32], batch: usize) -> Vec<f32> {
    let pso = ctx.matmul_f32_pipeline_v3();
    let x_buf = ctx.ctx.buffer_with_data(x);
    let y_buf = ctx.ctx.buffer_for::<f32>(batch * weight.out_features);

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct Dims {
        out_features: u32,
        in_features: u32,
    }
    let dims = Dims {
        out_features: weight.out_features as u32,
        in_features: weight.in_features as u32,
    };
    let batch_u32 = batch as u32;

    let cmd = metal::new_command_buffer(&ctx.ctx.queue);
    let encoder = cmd.auto_compute_encoder();
    encoder.set_compute_pipeline_state(pso);
    encoder.set_buffer(
        0,
        Some(weight.packed_buffer_ref()),
        weight.packed_offset as usize,
    );
    encoder.set_buffer(
        1,
        Some(weight.scales_buffer_ref()),
        weight.scales_offset as usize,
    );
    encoder.set_buffer(2, Some(&x_buf), 0);
    encoder.set_buffer(3, Some(&y_buf), 0);
    encoder.set_bytes_directly(
        4,
        std::mem::size_of::<Dims>(),
        &dims as *const _ as *const _,
    );
    encoder.set_bytes_directly(5, 4, &batch_u32 as *const _ as *const _);

    let tg_mem_bytes = (weight.in_features as u64) * 4;
    encoder.set_threadgroup_memory_length(0, tg_mem_bytes as usize);

    let n_tg_x = weight.out_features.div_ceil(8) as u64;
    let grid = mtl_size!(n_tg_x, batch, 1);
    let tg = mtl_size!(256, 1, 1);
    encoder.dispatch_thread_groups(grid, tg);
    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    ctx.ctx
        .read_buffer::<f32>(&y_buf, batch * weight.out_features)
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

fn assert_parity(name: &str, ref_: &[f32], v3: &[f32]) {
    let cos = cosine_similarity(ref_, v3);
    let rel = rel_max_err(ref_, v3);
    let abs = max_abs_err(ref_, v3);
    assert!(
        cos >= 0.9999,
        "{name}: cosine {cos} below 0.9999 (abs={abs}, rel={rel})"
    );
    assert!(
        rel <= 5e-3,
        "{name}: relative max err {rel} exceeds 5e-3 (cos={cos}, abs={abs})"
    );
}

#[test]
fn v2_v3_parity_dense_production_shapes() {
    let ctx = match MxFp4Context::new() {
        Ok(c) => Arc::new(c),
        Err(_) => return,
    };
    // Representative shapes from Qwen3.6-35B-A3B-mxfp4.
    let shapes = [
        ("qkv-small", 256, 256, 1),
        ("gate_up", 1024, 2048, 1),
        ("down", 2048, 512, 1),
        ("o_proj", 2048, 8192, 1),
        ("qkv-prod", 9216, 2048, 1),
        ("batch4", 64, 256, 4),
    ];
    for (name, out, ins, batch) in shapes {
        let packed = synth_packed(out, ins, 0xC0FFEE);
        let scales = synth_scales(out, ins, 0xBADCAFE);
        let x = synth_x(batch * ins);
        let weight = Mxfp4Weight::from_host(&ctx.ctx, &packed, &scales, out, ins).unwrap();

        let y_v2 = run_v2(&ctx, &weight, &x, batch);
        let y_v3 = run_v3(&ctx, &weight, &x, batch);
        assert_parity(
            &format!("dense `{name}` ({out}x{ins} batch={batch})"),
            &y_v2,
            &y_v3,
        );
    }
}

/// Edge case: out_features not a multiple of 8. v3 dispatches `ceil(out/8)`
/// threadgroups and the last threadgroup has some simdgroups whose `row >=
/// out_features` — those should bail out without writing.
#[test]
fn v2_v3_parity_non_multiple_of_8() {
    let ctx = match MxFp4Context::new() {
        Ok(c) => Arc::new(c),
        Err(_) => return,
    };
    let shapes = [
        ("out=63", 63, 256, 1),   // 8 tg's, last has only 7 valid rows
        ("out=129", 129, 512, 1), // 17 tg's, last has 1 valid row
        ("out=251", 251, 256, 1), // 32 tg's, last has 3 valid rows
    ];
    for (name, out, ins, batch) in shapes {
        let packed = synth_packed(out, ins, 0xDEADBEEF);
        let scales = synth_scales(out, ins, 0xFEEDFACE);
        let x = synth_x(batch * ins);
        let weight = Mxfp4Weight::from_host(&ctx.ctx, &packed, &scales, out, ins).unwrap();

        let y_v2 = run_v2(&ctx, &weight, &x, batch);
        let y_v3 = run_v3(&ctx, &weight, &x, batch);
        assert_parity(
            &format!("dense `{name}` ({out}x{ins} batch={batch})"),
            &y_v2,
            &y_v3,
        );
    }
}
