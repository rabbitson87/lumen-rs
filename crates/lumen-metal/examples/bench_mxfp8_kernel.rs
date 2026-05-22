//! Microbenchmark for the MXFP8 Metal kernel at Qwen3-Embedding-4B shapes.
//!
//! Two questions answered here, with no host data movement on the hot path:
//!
//!   1. **Naive vs qmv_fast** — every projection in Qwen3-Embedding-4B
//!      satisfies the cooperative kernel's alignment (`in % 512 == 0 &&
//!      out % 8 == 0`), but we still want to quantify the win so future
//!      regressions are caught.
//!   2. **Bandwidth utilization vs M3 Max peak** — packed-weight load is
//!      the dominant memory traffic; if we're nowhere near peak BW, the
//!      kernel has room to grow (and inversely, hitting peak means
//!      further optimization needs algorithmic change, not micro-tuning).
//!
//! Run:
//!   ```
//!   cargo run --release -p lumen-metal --example bench_mxfp8_kernel
//!   ```
//!
//! Optional knobs:
//!   `LUMEN_BENCH_ITERS=N`   override the per-shape iteration count (default 200).
//!   `LUMEN_MXFP8_NAIVE=1`   force the naive kernel for every dispatch
//!                           (overrides the alignment check; useful to
//!                           sanity-check the fallback path on aligned shapes).

use anyhow::Result;
use half::bf16;
use lumen_metal::metal::{self, CommandBufferExt};
use lumen_metal::mxfp8_gpu::{MXFP8_GROUP_SIZE, Mxfp8Context, Mxfp8Weight};
use std::time::Instant;

const PEAK_BW_GBS: f64 = 400.0; // M3 Max LPDDR5

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    out_features: usize,
    in_features: usize,
    batch: usize,
}

/// Projection shapes for `mlx-community/Qwen3-Embedding-4B-mxfp8`.
///   hidden = 2560, intermediate = 9728, q_heads = 32, kv_heads = 8,
///   head_dim = 128. (See HF config.json.)
/// Embedding workload is `batch = max_seq_len` (single forward, no
/// autoregressive decode) — we bench at three representative batch sizes:
///   batch=1     — minimum, simdgroup utilization floor
///   batch=32    — typical short sentence
///   batch=512   — long context / sentence pair re-ranking
const SHAPES: &[Shape] = &[
    // Attention projections
    Shape {
        name: "q_proj  2560→4096",
        out_features: 4096,
        in_features: 2560,
        batch: 1,
    },
    Shape {
        name: "q_proj  2560→4096 b=32",
        out_features: 4096,
        in_features: 2560,
        batch: 32,
    },
    Shape {
        name: "q_proj  2560→4096 b=512",
        out_features: 4096,
        in_features: 2560,
        batch: 512,
    },
    Shape {
        name: "k_proj  2560→1024",
        out_features: 1024,
        in_features: 2560,
        batch: 1,
    },
    Shape {
        name: "v_proj  2560→1024",
        out_features: 1024,
        in_features: 2560,
        batch: 1,
    },
    Shape {
        name: "o_proj  1024→2560",
        out_features: 2560,
        in_features: 1024,
        batch: 1,
    },
    // MLP projections
    Shape {
        name: "gate    2560→9728",
        out_features: 9728,
        in_features: 2560,
        batch: 1,
    },
    Shape {
        name: "gate    2560→9728 b=32",
        out_features: 9728,
        in_features: 2560,
        batch: 32,
    },
    Shape {
        name: "up      2560→9728",
        out_features: 9728,
        in_features: 2560,
        batch: 1,
    },
    Shape {
        name: "down    9728→2560",
        out_features: 2560,
        in_features: 9728,
        batch: 1,
    },
    Shape {
        name: "down    9728→2560 b=32",
        out_features: 2560,
        in_features: 9728,
        batch: 32,
    },
];

fn bench_iters() -> usize {
    std::env::var("LUMEN_BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200)
}

fn random_packed(out_f: usize, in_f: usize) -> Vec<u32> {
    // Deterministic mixed-bit pattern. E4M3 NaN encoding is 0x7F / 0xFF —
    // mask those out so the kernel doesn't short-circuit zero contributions
    // and skew the timing.
    let n = out_f * in_f / 4;
    (0..n)
        .map(|i| {
            let mut w = 0x3848_5868u32.wrapping_mul(i as u32 + 1);
            for byte_idx in 0..4u32 {
                let b = (w >> (byte_idx * 8)) & 0xFF;
                if b == 0x7F || b == 0xFF {
                    w &= !(0xFF << (byte_idx * 8));
                    w |= 0x38u32 << (byte_idx * 8);
                }
            }
            w
        })
        .collect()
}

fn random_scales(out_f: usize, in_f: usize) -> Vec<u8> {
    // E8M0 = 127 → scale = 2^0 = 1.0; avoids the 0xFF NaN sentinel.
    vec![127u8; out_f * in_f / MXFP8_GROUP_SIZE]
}

fn random_activation_bf16(n: usize) -> Vec<u16> {
    (0..n)
        .map(|i| bf16::from_f32(((i as f32) * 0.001).sin()).to_bits())
        .collect()
}

fn print_header(label: &str) {
    println!("M3 Max LPDDR5 peak bandwidth: {:.0} GB/s", PEAK_BW_GBS);
    println!("Kernel: {label}");
    println!();
    println!(
        "{:<32} {:>10} {:>10} {:>10} {:>10}",
        "Shape", "ms/call", "GB/s", "% peak", "GFLOP/s"
    );
    println!("{}", "-".repeat(78));
}

fn report(name: &str, ms_per_call: f64, bytes_per_call: usize, flops_per_call: usize) {
    let secs = ms_per_call / 1000.0;
    let bw_gbs = (bytes_per_call as f64 / 1e9) / secs;
    let pct = bw_gbs / PEAK_BW_GBS * 100.0;
    let gflops = (flops_per_call as f64 / 1e9) / secs;
    println!(
        "{:<32} {:>10.4} {:>10.2} {:>9.1}% {:>10.2}",
        name, ms_per_call, bw_gbs, pct, gflops
    );
}

/// Encode N back-to-back dispatches into a SINGLE command buffer so per-call
/// encoder + commit overhead is amortized. Mirrors `bench_mxfp4_kernel.rs`'s
/// methodology; comparable numbers across the two kernels.
fn encode_n(
    ctx: &Mxfp8Context,
    weight: &Mxfp8Weight,
    x_buf: &metal::Buffer,
    y_buf: &metal::Buffer,
    batch: usize,
    n: usize,
    force_naive: bool,
) -> Result<()> {
    let cmd = metal::new_command_buffer(&ctx.ctx.queue);
    for _ in 0..n {
        let encoder = cmd.auto_compute_encoder();
        encoder.set_label("lumen:mxfp8_bench");
        if force_naive {
            ctx.encode_naive_bf16_dispatch(
                encoder.as_ref(),
                weight,
                x_buf,
                0,
                y_buf,
                0,
                batch,
            );
        } else {
            ctx.encode_matmul_bf16_dispatch(
                encoder.as_ref(),
                weight,
                x_buf,
                0,
                y_buf,
                0,
                batch,
            );
        }
        encoder.end_encoding();
    }
    cmd.commit();
    cmd.wait_until_completed();
    Ok(())
}

fn bench_one(ctx: &Mxfp8Context, shape: &Shape, force_naive: bool) -> Result<()> {
    let Shape {
        name,
        out_features,
        in_features,
        batch,
    } = *shape;
    let packed = random_packed(out_features, in_features);
    let scales = random_scales(out_features, in_features);
    let weight = Mxfp8Weight::from_host(&ctx.ctx, &packed, &scales, out_features, in_features)?;
    let x_bf16 = random_activation_bf16(batch * in_features);
    let x_buf = ctx.ctx.buffer_with_data(&x_bf16);
    let y_buf = ctx.ctx.buffer_for::<u16>(batch * out_features);

    // Warmup (5 dispatches) — primes the cmd-queue + ensures shaders are
    // compiled before the timing window.
    encode_n(ctx, &weight, &x_buf, &y_buf, batch, 5, force_naive)?;

    let iters = bench_iters();
    let t0 = Instant::now();
    encode_n(ctx, &weight, &x_buf, &y_buf, batch, iters, force_naive)?;
    let total = t0.elapsed();
    let ms_per_call = total.as_secs_f64() * 1000.0 / iters as f64;

    // Bytes per call (write-once, read-once for weights):
    //   packed:  out × in / 4  uint32 × 4 bytes  = out × in bytes
    //   scales:  out × in / 32 bytes
    //   x:       batch × in  ushort × 2 bytes
    //   y:       batch × out ushort × 2 bytes
    let packed_b = out_features * in_features;
    let scales_b = out_features * in_features / MXFP8_GROUP_SIZE;
    let x_b = batch * in_features * 2;
    let y_b = batch * out_features * 2;
    let bytes = packed_b + scales_b + x_b + y_b;

    // 2 × out × in × batch (multiply + accumulate per element)
    let flops = 2 * out_features * in_features * batch;
    report(name, ms_per_call, bytes, flops);
    Ok(())
}

fn main() -> Result<()> {
    let ctx = Mxfp8Context::new()?;
    let env_naive = std::env::var("LUMEN_MXFP8_NAIVE")
        .map(|v| v == "1")
        .unwrap_or(false);

    if env_naive {
        println!(
            "[bench] LUMEN_MXFP8_NAIVE=1 — running naive kernel only for all shapes\n"
        );
        print_header("naive (mxfp8_matmul_bf16)");
        for s in SHAPES {
            bench_one(&ctx, s, true)?;
        }
        return Ok(());
    }

    // Default: side-by-side naive vs qmv_fast at each shape.
    print_header("naive (mxfp8_matmul_bf16)");
    for s in SHAPES {
        bench_one(&ctx, s, true)?;
    }
    println!();
    print_header("qmv_fast (mxfp8_qmv_fast_bf16)");
    for s in SHAPES {
        bench_one(&ctx, s, false)?;
    }
    println!(
        "\nNote: qmv_fast requires in%512==0 && out%8==0 — all Qwen3-Embedding-4B\nprojection shapes satisfy this, so the auto-dispatcher picks it in production."
    );

    Ok(())
}
