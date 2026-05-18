//! compare the dedicated small-out matmul kernel
//! against the general v3 kernel on r_gate-shaped projections.
//!
//! r_gate (out=256, in=2048, batch=1) is the routing gate inside `SparseMoeBlock`.
//! v3 schedules `n_groups_x = ceil(256/8) = 32` threadgroups, leaving the M3 Max
//! GPU's ~30 cores under-occupied. The small-out kernel runs 1 TG per output
//! element (256 TGs) with 256 cooperating threads per TG, which trades more
//! cross-simdgroup reduction for restored latency hiding.
//!
//! Run: `cargo run --release --example bench_mxfp4_small_out -p lumen-metal`

use anyhow::Result;
use std::time::Instant;
use lumen_metal::mxfp4_gpu::{Mxfp4Weight, MxFp4Context};

const N_REPS: usize = 200;
const WARMUP: usize = 10;

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    out: usize,
    in_features: usize,
    batch: usize,
}

fn random_packed(out_f: usize, in_f: usize) -> Vec<u32> {
    let n = out_f * in_f / 8;
    (0..n).map(|i| 0x12345678u32.wrapping_add(i as u32)).collect()
}

fn random_scales(out_f: usize, in_f: usize) -> Vec<u8> {
    vec![127u8; out_f * in_f / 32]
}

fn random_activation(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.001).sin()).collect()
}

fn bench_shape(ctx: &MxFp4Context, shape: Shape) -> Result<()> {
    let Shape {
        name,
        out,
        in_features,
        batch,
    } = shape;

    let packed = random_packed(out, in_features);
    let scales = random_scales(out, in_features);
    let weight = Mxfp4Weight::from_host(&ctx.ctx, &packed, &scales, out, in_features)?;
    let x = random_activation(batch * in_features);
    let x_buf = ctx.ctx.buffer_with_data(&x);
    let y_buf = ctx.ctx.buffer_for::<f32>(batch * out);

    // Warmup both kernels (per-call commit+wait).
    for _ in 0..WARMUP {
        ctx.matmul_zero_copy(&weight, &x_buf, 0, &y_buf, 0, batch)?;
        ctx.matmul_small_out_zero_copy(&weight, &x_buf, 0, &y_buf, 0, batch)?;
    }

    // v3
    let t0 = Instant::now();
    for _ in 0..N_REPS {
        ctx.matmul_zero_copy(&weight, &x_buf, 0, &y_buf, 0, batch)?;
    }
    let v3_us = t0.elapsed().as_secs_f64() * 1e6 / N_REPS as f64;

    // small-out
    let t0 = Instant::now();
    for _ in 0..N_REPS {
        ctx.matmul_small_out_zero_copy(&weight, &x_buf, 0, &y_buf, 0, batch)?;
    }
    let so_us = t0.elapsed().as_secs_f64() * 1e6 / N_REPS as f64;

    let speedup = v3_us / so_us;
    let delta_pct = (v3_us - so_us) / v3_us * 100.0;
    println!(
        "{:<36} v3 {:>9.2} us  small {:>9.2} us  speedup {:>5.2}x  delta {:>+6.2}%",
        name, v3_us, so_us, speedup, delta_pct
    );
    Ok(())
}

fn main() -> Result<()> {
    let ctx = MxFp4Context::new()?;
    println!(
        "Phase M.2-A microbench: v3 zero_copy vs small-out zero_copy (N={N_REPS}, warmup={WARMUP})"
    );
    println!("Each call has its own commit + wait — matches the production routing-gate path.");
    println!();

    let shapes = [
        Shape {
            name: "r_gate (256 x 2048, b=1)",
            out: 256,
            in_features: 2048,
            batch: 1,
        },
        Shape {
            name: "r_gate (256 x 2048, b=4)",
            out: 256,
            in_features: 2048,
            batch: 4,
        },
        Shape {
            name: "small (128 x 2048, b=1)",
            out: 128,
            in_features: 2048,
            batch: 1,
        },
        Shape {
            name: "shared gate_up (1024 x 2048)",
            out: 1024,
            in_features: 2048,
            batch: 1,
        },
        Shape {
            name: "lm_head proxy (8192 x 2048)",
            out: 8192,
            in_features: 2048,
            batch: 1,
        },
    ];
    for s in shapes {
        bench_shape(&ctx, s)?;
    }
    Ok(())
}
