//! Microbenchmark for MXFP4 Metal kernels (Step 1A: Path B profiling baseline).
//!
//! Measures per-call GPU time for representative shapes used by Qwen3.6-35B-A3B-mxfp4
//! and computes effective memory bandwidth (% of M3 Max LPDDR5 peak 400 GB/s) to
//! determine whether kernels are memory-bound (→ tiling will help) or compute-bound.
//!
//! Methodology:
//! - Encode N back-to-back dispatches into a SINGLE serial compute encoder + cmd buf.
//! - Wall-clock total / N = per-call GPU time (encoder overhead amortized).
//! - Compare against single-cmdbuf baseline to isolate dispatch/commit/wait cost.
//!
//! Run: `cargo run --release --example bench_mxfp4_kernel -p lumen-metal`

use anyhow::Result;
use lumen_metal::device::MetalContext;
use lumen_metal::metal;
use lumen_metal::metal::{BatchedEncoderExt, CommandBufferExt, ComputeEncoderCompat};
use lumen_metal::mtl_size;
use lumen_metal::mxfp4_gpu::{MxFp4Context, Mxfp4Weight};
use std::time::Instant;

const PEAK_BW_GBS: f64 = 400.0; // M3 Max LPDDR5

#[derive(Clone)]
struct DenseShape {
    name: &'static str,
    out_features: usize,
    in_features: usize,
    batch: usize,
}

#[derive(Clone)]
struct MoeShape {
    name: &'static str,
    out_features: usize,
    in_features: usize,
    batch: usize,
    k_slots: usize,
    num_experts_total: usize,
}

fn random_packed(out_f: usize, in_f: usize) -> Vec<u32> {
    // Deterministic non-zero pattern. Real values don't matter for timing — what matters
    // is that no nibble is zero (else the inner-loop short-circuits would skew timing).
    let n = out_f * in_f / 8;
    (0..n)
        .map(|i| 0x12345678u32.wrapping_add(i as u32))
        .collect()
}

fn random_scales(out_f: usize, in_f: usize) -> Vec<u8> {
    // E8M0=127 means scale=1.0 (no skip). 0xFF would be NaN/skip — explicitly avoid.
    vec![127u8; out_f * in_f / 32]
}

fn random_activation(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.001).sin()).collect()
}

fn print_header(label: &str) {
    println!("M3 Max LPDDR5 peak bandwidth: {:.0} GB/s", PEAK_BW_GBS);
    println!("Kernel version: {label}");
    println!();
    println!(
        "{:<40} {:>10} {:>10} {:>10} {:>10}",
        "Shape", "ms/call", "GB/s", "% peak", "comp/mem"
    );
    println!("{}", "-".repeat(82));
}

fn report(name: &str, ms_per_call: f64, bytes_per_call: usize, flops_per_call: usize) {
    let secs_per_call = ms_per_call / 1000.0;
    let bw_gbs = (bytes_per_call as f64 / 1e9) / secs_per_call;
    let pct = bw_gbs / PEAK_BW_GBS * 100.0;
    let arith_intensity = flops_per_call as f64 / bytes_per_call as f64; // flops/byte
    println!(
        "{:<40} {:>10.4} {:>10.2} {:>9.1}% {:>10.3}",
        name, ms_per_call, bw_gbs, pct, arith_intensity
    );
}

fn bench_dense(ctx: &MxFp4Context, shape: &DenseShape) -> Result<()> {
    let DenseShape {
        name,
        out_features,
        in_features,
        batch,
    } = *shape;

    let packed = random_packed(out_features, in_features);
    let scales = random_scales(out_features, in_features);
    let weight = Mxfp4Weight::from_host(&ctx.ctx, &packed, &scales, out_features, in_features)?;
    let x = random_activation(batch * in_features);
    let x_buf = ctx.ctx.buffer_with_data(&x);
    let y_buf = ctx.ctx.buffer_for::<f32>(batch * out_features);

    // Warmup with the currently-selected pipeline (v1 default, v2 if env)
    for _ in 0..5 {
        ctx.matmul_zero_copy(&weight, &x_buf, 0, &y_buf, 0, batch)?;
    }

    // Single cmd buffer with N back-to-back dispatches via the env-selected pipeline.
    const N: usize = 200;
    let t0 = Instant::now();
    encode_n_dense(ctx, &weight, &x_buf, &y_buf, batch, N)?;
    let total = t0.elapsed();
    let ms_per_call = total.as_secs_f64() * 1000.0 / N as f64;

    // Bytes per call:
    //   packed:  out × in / 8 × 4 bytes
    //   scales:  out × in / 32 bytes
    //   x:       batch × in × 4 bytes
    //   y:       batch × out × 4 bytes (write)
    let packed_b = out_features * in_features / 8 * 4;
    let scales_b = out_features * in_features / 32;
    let x_b = batch * in_features * 4;
    let y_b = batch * out_features * 4;
    let bytes = packed_b + scales_b + x_b + y_b;

    // Effective FLOPs: 2 × out × in × batch (multiply + accumulate per element)
    let flops = 2 * out_features * in_features * batch;

    report(name, ms_per_call, bytes, flops);
    Ok(())
}

fn encode_n_dense(
    ctx: &MxFp4Context,
    weight: &Mxfp4Weight,
    x_buf: &metal::Buffer,
    y_buf: &metal::Buffer,
    batch: usize,
    n: usize,
) -> Result<()> {
    // Build dims/batch as inline bytes
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
    let pso = ctx.matmul_f32_pipeline();
    let max_threads = pso.max_total_threads_per_threadgroup();
    let threads_per_tg = max_threads.min(256);

    for _ in 0..n {
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
        encoder.set_buffer(2, Some(x_buf), 0);
        encoder.set_buffer(3, Some(y_buf), 0);
        encoder.set_bytes_directly(
            4,
            std::mem::size_of::<Dims>(),
            &dims as *const _ as *const _,
        );
        encoder.set_bytes_directly(5, 4, &batch_u32 as *const _ as *const _);

        let grid = mtl_size!(weight.out_features, batch, 1);
        let tg = mtl_size!(threads_per_tg, 1, 1);
        encoder.dispatch_threads(grid, tg);
        encoder.end_encoding();
    }
    cmd.commit();
    cmd.wait_until_completed();
    Ok(())
}

fn bench_moe(ctx: &MxFp4Context, shape: &MoeShape) -> Result<()> {
    let MoeShape {
        name,
        out_features,
        in_features,
        batch,
        k_slots,
        num_experts_total,
    } = *shape;

    let packed_per_expert = out_features * in_features / 8;
    let scales_per_expert = out_features * in_features / 32;
    let total_packed: Vec<u32> = (0..num_experts_total * packed_per_expert)
        .map(|i| 0x11223344u32.wrapping_add(i as u32))
        .collect();
    let total_scales = vec![127u8; num_experts_total * scales_per_expert];

    let packed_buf = ctx.ctx.buffer_with_data(&total_packed);
    let scales_buf = ctx.ctx.buffer_with_data(&total_scales);
    let x = random_activation(batch * in_features);
    let x_buf = ctx.ctx.buffer_with_data(&x);
    let y_buf = ctx.ctx.buffer_for::<f32>(k_slots * batch * out_features);
    let expert_indices: Vec<u32> = (0..k_slots as u32).collect();

    // Warmup
    for _ in 0..5 {
        ctx.matmul_moe_zero_copy(
            &packed_buf,
            &scales_buf,
            &expert_indices,
            &x_buf,
            0,
            &y_buf,
            0,
            out_features,
            in_features,
            batch,
            true, // broadcast_x: same x for all slots
        )?;
    }

    const N: usize = 200;
    let t0 = Instant::now();
    encode_n_moe(
        ctx,
        &packed_buf,
        &scales_buf,
        &expert_indices,
        &x_buf,
        &y_buf,
        out_features,
        in_features,
        batch,
        N,
    )?;
    let total = t0.elapsed();
    let ms_per_call = total.as_secs_f64() * 1000.0 / N as f64;

    // For MoE: k slots × (packed + scales + y) but x is broadcast (read once total).
    let packed_b = k_slots * out_features * in_features / 8 * 4;
    let scales_b = k_slots * out_features * in_features / 32;
    let x_b = batch * in_features * 4; // broadcast: read once
    let y_b = k_slots * batch * out_features * 4;
    let bytes = packed_b + scales_b + x_b + y_b;
    let flops = 2 * k_slots * out_features * in_features * batch;

    report(name, ms_per_call, bytes, flops);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_n_moe(
    ctx: &MxFp4Context,
    packed_all: &metal::Buffer,
    scales_all: &metal::Buffer,
    expert_indices: &[u32],
    x_buf: &metal::Buffer,
    y_buf: &metal::Buffer,
    out_features: usize,
    in_features: usize,
    batch: usize,
    n: usize,
) -> Result<()> {
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct MoeDims {
        out_features: u32,
        in_features: u32,
        batch: u32,
        broadcast_x: u32,
    }
    let dims = MoeDims {
        out_features: out_features as u32,
        in_features: in_features as u32,
        batch: batch as u32,
        broadcast_x: 1,
    };
    let k = expert_indices.len();
    let indices_buf = ctx.ctx.buffer_with_data(expert_indices);

    let cmd = metal::new_command_buffer(&ctx.ctx.queue);
    let pso = ctx.matmul_moe_f32_pipeline();
    let max_threads = pso.max_total_threads_per_threadgroup();
    let threads_per_tg = max_threads.min(256);

    for _ in 0..n {
        let encoder = cmd.auto_compute_encoder();
        encoder.set_compute_pipeline_state(pso);
        encoder.set_buffer(0, Some(packed_all), 0);
        encoder.set_buffer(1, Some(scales_all), 0);
        encoder.set_buffer(2, Some(&indices_buf), 0);
        encoder.set_buffer(3, Some(x_buf), 0);
        encoder.set_buffer(4, Some(y_buf), 0);
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<MoeDims>(),
            &dims as *const _ as *const _,
        );

        let grid = mtl_size!(out_features, batch, k);
        let tg = mtl_size!(threads_per_tg, 1, 1);
        encoder.dispatch_threads(grid, tg);
        encoder.end_encoding();
    }
    cmd.commit();
    cmd.wait_until_completed();
    let _ = indices_buf; // ensure lifetime to wait
    Ok(())
}

fn main() -> Result<()> {
    let ctx = MxFp4Context::new()?;
    let label = if ctx.uses_v3() {
        "v3 (simdgroup cooperative + tg x cache)"
    } else if ctx.uses_v2() {
        "v2 (uint4+float4 vectorized)"
    } else {
        "v1 (scalar baseline)"
    };
    print_header(label);

    let dense_shapes = [
        DenseShape {
            name: "self_attn qkv  [9216 x 2048]",
            out_features: 9216,
            in_features: 2048,
            batch: 1,
        },
        DenseShape {
            name: "self_attn o    [2048 x 8192]",
            out_features: 2048,
            in_features: 8192,
            batch: 1,
        },
        DenseShape {
            name: "shared gate_up [1024 x 2048]",
            out_features: 1024,
            in_features: 2048,
            batch: 1,
        },
        DenseShape {
            name: "shared down    [2048 x  512]",
            out_features: 2048,
            in_features: 512,
            batch: 1,
        },
        DenseShape {
            name: "linear_attn in [4480 x 2048]",
            out_features: 4480,
            in_features: 2048,
            batch: 1,
        },
        DenseShape {
            name: "lm_head        [vocab x 2048] proxy",
            out_features: 32768,
            in_features: 2048,
            batch: 1,
        },
    ];
    for s in &dense_shapes {
        bench_dense(&ctx, s)?;
    }

    println!();
    println!("--- MoE grouped (k=8 expert slots) ---");
    let moe_shapes = [
        MoeShape {
            name: "MoE gate_up [1024 x 2048] x8",
            out_features: 1024,
            in_features: 2048,
            batch: 1,
            k_slots: 8,
            num_experts_total: 256,
        },
        MoeShape {
            name: "MoE down    [2048 x  512] x8",
            out_features: 2048,
            in_features: 512,
            batch: 1,
            k_slots: 8,
            num_experts_total: 256,
        },
    ];
    for s in &moe_shapes {
        bench_moe(&ctx, s)?;
    }

    println!();
    println!("Notes:");
    println!("- ms/call: per-dispatch GPU time (N=200 amortized in single cmd buffer)");
    println!("- GB/s: effective bandwidth (weight + scales + x + y bytes / time)");
    println!("- comp/mem: arithmetic intensity (FLOPs / byte). M3 Max ~ peak fp32 8 GFLOPs/byte;");
    println!("  fp4 weight-only is memory-bound when intensity << ridge point.");

    let _ = std::any::type_name::<MetalContext>();
    Ok(())
}
