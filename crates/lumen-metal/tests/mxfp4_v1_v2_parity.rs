//! Parity check: v2 (uint4 + float4 vectorized loads) vs v1 (scalar loads)
//! must produce numerically equivalent output for the same inputs.
//!
//! v2 only changes the load width — the per-term
//! arithmetic `acc += LUT * s * x` is preserved literally in source. However,
//! the Metal compiler treats vector loads as a hint for aggressive FMA fusion
//! across consecutive terms, so the floating-point result can drift by a few
//! ULPs of magnitude relative to the inner accumulator.
//!
//! Acceptance criterion (matches the Phase A roadmap's logits cos > 0.99 for
//! new kernels):
//!   - cosine similarity ≥ 0.9999 (any honest drift; structural bugs collapse this)
//!   - relative max error ≤ 5e-3 (≈ 3 ULPs on typical magnitudes)
//! A genuine bug — e.g. a wrong nibble decode — would push cosine below 0.9
//! and relative error to 0.5+, well beyond these bounds.

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
    // Avoid 0xFF (NaN/skip path); centre around bias 127 (= 1.0 scale).
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

/// Run a single-call kernel using the explicitly-selected pipeline (v1 or v2),
/// bypassing the env-driven `matmul_zero_copy` path so both versions can be
/// measured in the same process.
fn run_dense(
    ctx: &MxFp4Context,
    pso: &metal::ComputePipelineStateRef,
    weight: &Mxfp4Weight,
    x: &[f32],
    batch: usize,
) -> Vec<f32> {
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

fn assert_parity(name: &str, v1: &[f32], v2: &[f32]) {
    let cos = cosine_similarity(v1, v2);
    let rel = rel_max_err(v1, v2);
    let abs = max_abs_err(v1, v2);
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
fn v1_v2_parity_dense_qkv_shape() {
    let ctx = match MxFp4Context::new() {
        Ok(c) => Arc::new(c),
        Err(_) => return,
    };
    // Representative shapes from Qwen3.6-35B-A3B-mxfp4. Keep them small enough
    // for the test to run quickly but large enough to exercise multiple
    // threadgroups + multiple groups per row.
    let shapes = [
        ("qkv-tiny", 256, 256, 1),
        ("gate_up", 1024, 2048, 1),
        ("o_proj", 2048, 512, 1),
        ("batch4", 64, 256, 4),
    ];
    for (name, out, ins, batch) in shapes {
        let packed = synth_packed(out, ins, 0xBAD_C0DE);
        let scales = synth_scales(out, ins, 0xCAFE_BABE);
        let x = synth_x(batch * ins);
        let weight = Mxfp4Weight::from_host(&ctx.ctx, &packed, &scales, out, ins).unwrap();

        let y_v1 = run_dense(&ctx, ctx.matmul_f32_pipeline_v1(), &weight, &x, batch);
        let y_v2 = run_dense(&ctx, ctx.matmul_f32_pipeline_v2(), &weight, &x, batch);
        assert_parity(
            &format!("dense `{name}` ({out}x{ins} batch={batch})"),
            &y_v1,
            &y_v2,
        );
    }
}

#[test]
fn v1_v2_parity_moe_grouped() {
    let ctx = match MxFp4Context::new() {
        Ok(c) => Arc::new(c),
        Err(_) => return,
    };
    let num_experts = 8;
    let out = 1024;
    let ins = 2048;
    let batch = 1;
    let k = 4;

    // Build packed/scales for `num_experts` experts contiguous in one buffer.
    let packed_per_expert = out * ins / 8;
    let scales_per_expert = out * ins / 32;
    let mut packed_all = Vec::with_capacity(num_experts * packed_per_expert);
    let mut scales_all = Vec::with_capacity(num_experts * scales_per_expert);
    for e in 0..num_experts {
        packed_all.extend(synth_packed(out, ins, 0x1000 + e as u32));
        scales_all.extend(synth_scales(out, ins, 0x2000 + e as u32));
    }
    let packed_buf = ctx.ctx.buffer_with_data(&packed_all);
    let scales_buf = ctx.ctx.buffer_with_data(&scales_all);

    let x = synth_x(batch * ins);
    let x_buf = ctx.ctx.buffer_with_data(&x);
    let expert_indices: Vec<u32> = vec![0, 2, 5, 7];

    let y_v1 = ctx.ctx.buffer_for::<f32>(k * batch * out);
    let y_v2 = ctx.ctx.buffer_for::<f32>(k * batch * out);

    fn dispatch_moe(
        ctx: &MxFp4Context,
        pso: &metal::ComputePipelineStateRef,
        packed: &metal::Buffer,
        scales: &metal::Buffer,
        indices: &[u32],
        x: &metal::Buffer,
        y: &metal::Buffer,
        out: usize,
        ins: usize,
        batch: usize,
    ) {
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MoeDims {
            out_features: u32,
            in_features: u32,
            batch: u32,
            broadcast_x: u32,
        }
        let dims = MoeDims {
            out_features: out as u32,
            in_features: ins as u32,
            batch: batch as u32,
            broadcast_x: 1,
        };
        let idx_buf = ctx.ctx.buffer_with_data(indices);
        let cmd = metal::new_command_buffer(&ctx.ctx.queue);
        let enc = cmd.auto_compute_encoder();
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(packed), 0);
        enc.set_buffer(1, Some(scales), 0);
        enc.set_buffer(2, Some(&idx_buf), 0);
        enc.set_buffer(3, Some(x), 0);
        enc.set_buffer(4, Some(y), 0);
        enc.set_bytes_directly(
            5,
            std::mem::size_of::<MoeDims>(),
            &dims as *const _ as *const _,
        );

        let max_threads = pso.max_total_threads_per_threadgroup();
        let threads_per_tg = max_threads.min(256);
        let grid = mtl_size!(out, batch, indices.len());
        let tg = mtl_size!(threads_per_tg, 1, 1);
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let _ = idx_buf; // outlives the wait
    }

    dispatch_moe(
        &ctx,
        ctx.matmul_moe_f32_pipeline_v1(),
        &packed_buf,
        &scales_buf,
        &expert_indices,
        &x_buf,
        &y_v1,
        out,
        ins,
        batch,
    );
    dispatch_moe(
        &ctx,
        ctx.matmul_moe_f32_pipeline_v2(),
        &packed_buf,
        &scales_buf,
        &expert_indices,
        &x_buf,
        &y_v2,
        out,
        ins,
        batch,
    );

    let v1: Vec<f32> = ctx.ctx.read_buffer(&y_v1, k * batch * out);
    let v2: Vec<f32> = ctx.ctx.read_buffer(&y_v2, k * batch * out);
    assert_parity("MoE grouped (k=4 of 8)", &v1, &v2);
}
