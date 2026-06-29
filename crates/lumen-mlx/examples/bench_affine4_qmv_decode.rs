//! Isolation BW benchmark for mlx's affine4 `quantized_matmul` at 27B-dense
//! DECODE shapes (matvec, seq=1). Decides whether the baseline-decode
//! BW-efficiency lever has headroom: if mlx's QMV already hits ~85-90% of the
//! M3 Max 300 GB/s peak on these shapes, a custom kernel can't help (lever dead);
//! if it's ~60%, there's room.
//!
//! Measures GB/s = (packed + scales + biases bytes) * iters / elapsed, evaling
//! each iter (mimics per-layer decode dispatch). Run:
//!   cargo run --release -p lumen-mlx --features mlx-native --example bench_affine4_qmv_decode

use anyhow::Result;
use mlx_rs::{Array, Dtype};
use std::time::Instant;

fn dtype_bytes(d: Dtype) -> usize {
    match d {
        Dtype::Uint32 | Dtype::Int32 | Dtype::Float32 => 4,
        Dtype::Bfloat16 | Dtype::Float16 | Dtype::Uint16 | Dtype::Int16 => 2,
        _ => 4,
    }
}
fn arr_bytes(a: &Array) -> usize {
    a.shape().iter().product::<i32>() as usize * dtype_bytes(a.dtype())
}

fn bench_shape(name: &str, out: i32, inn: i32, iters: usize, peak_gbs: f64) -> Result<()> {
    let gs = 64;
    let bits = 4;
    // Random weight [out, in] → affine4 quantize (packed u32, scales, biases).
    let w =
        mlx_rs::random::normal::<f32>(&[out, inn], None, None, None)?.as_dtype(Dtype::Bfloat16)?;
    let (wq, scales, biases) = mlx_rs::ops::quantize(&w, gs, bits)?;
    wq.eval()?;
    scales.eval()?;
    biases.eval()?;
    // Decode input [1, 1, in] bf16.
    let x =
        mlx_rs::random::normal::<f32>(&[1, 1, inn], None, None, None)?.as_dtype(Dtype::Bfloat16)?;
    x.eval()?;

    let wbytes = arr_bytes(&wq) + arr_bytes(&scales) + arr_bytes(&biases);

    // Warmup.
    for _ in 0..8 {
        let y = mlx_rs::ops::quantized_matmul(&x, &wq, &scales, Some(&biases), true, gs, bits)?;
        y.eval()?;
    }
    // Timed: eval each iter (steady per-dispatch decode pattern).
    let t0 = Instant::now();
    for _ in 0..iters {
        let y = mlx_rs::ops::quantized_matmul(&x, &wq, &scales, Some(&biases), true, gs, bits)?;
        y.eval()?;
    }
    let el = t0.elapsed().as_secs_f64();
    let per_ms = el / iters as f64 * 1000.0;
    let gbs = wbytes as f64 * iters as f64 / el / 1e9;
    println!(
        "{name:<22} out={out:>6} in={inn:>6}  w={:.1}MB  {per_ms:>7.3} ms/call  {gbs:>6.1} GB/s  ({:>4.0}% of {peak_gbs:.0})",
        wbytes as f64 / 1e6,
        gbs / peak_gbs * 100.0,
    );
    Ok(())
}

/// Pipelined MLP chain: gate[h→i] then down[i→h], output feeds next input, ALL
/// in one eval (no per-call sync) — mimics the real forward's command buffer
/// where consecutive matmuls pipeline. This is the TRUE in-decode BW of mlx's
/// QMV (vs the eval-per-call number which serializes & shows dispatch latency).
fn bench_mlp_chain(hidden: i32, interm: i32, pairs: usize, peak: f64) -> Result<()> {
    let (gs, bits) = (64, 4);
    let mk = |out: i32, inn: i32| -> Result<(Array, Array, Array)> {
        let w = mlx_rs::random::normal::<f32>(&[out, inn], None, None, None)?
            .as_dtype(Dtype::Bfloat16)?;
        let q = mlx_rs::ops::quantize(&w, gs, bits)?;
        q.0.eval()?;
        q.1.eval()?;
        q.2.eval()?;
        Ok(q)
    };
    let (gw, gs_, gb) = mk(interm, hidden)?; // gate: hidden -> interm
    let (dw, ds_, db) = mk(hidden, interm)?; // down: interm -> hidden
    let bytes_pair = arr_bytes(&gw)
        + arr_bytes(&gs_)
        + arr_bytes(&gb)
        + arr_bytes(&dw)
        + arr_bytes(&ds_)
        + arr_bytes(&db);
    let mut x = mlx_rs::random::normal::<f32>(&[1, 1, hidden], None, None, None)?
        .as_dtype(Dtype::Bfloat16)?;
    x.eval()?;
    // warmup
    for _ in 0..3 {
        let g = mlx_rs::ops::quantized_matmul(&x, &gw, &gs_, Some(&gb), true, gs, bits)?;
        let d = mlx_rs::ops::quantized_matmul(&g, &dw, &ds_, Some(&db), true, gs, bits)?;
        d.eval()?;
    }
    let t0 = Instant::now();
    let mut cur = x.clone();
    for _ in 0..pairs {
        let g = mlx_rs::ops::quantized_matmul(&cur, &gw, &gs_, Some(&gb), true, gs, bits)?;
        cur = mlx_rs::ops::quantized_matmul(&g, &dw, &ds_, Some(&db), true, gs, bits)?;
    }
    cur.eval()?; // single eval for the whole chain → pipelined
    let _ = &mut x;
    let el = t0.elapsed().as_secs_f64();
    let gbs = bytes_pair as f64 * pairs as f64 / el / 1e9;
    println!(
        "MLP CHAIN (pipelined)  {pairs} pairs  {:.0}MB/pair  {:.2} ms/pair  {:.1} GB/s  ({:.0}% of {peak:.0})",
        bytes_pair as f64 / 1e6,
        el / pairs as f64 * 1000.0,
        gbs,
        gbs / peak * 100.0,
    );
    Ok(())
}

fn main() -> Result<()> {
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(400);
    // M3 Max 36GB = 300 GB/s; override via PEAK_GBS.
    let peak: f64 = std::env::var("PEAK_GBS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300.0);
    println!("=== affine4 QMV decode-shape BW (mlx quantized_matmul, seq=1) ===");
    println!("iters={iters} peak={peak} GB/s  (27B dense: hidden=5120 interm=17408)\n");

    // 27B dense MLP decode matmuls.
    bench_shape("gate_proj", 17408, 5120, iters, peak)?;
    bench_shape("up_proj", 17408, 5120, iters, peak)?;
    bench_shape("down_proj", 5120, 17408, iters, peak)?;
    // FUSED gate_up (concat along out): 1 dispatch instead of 2 — the lever.
    bench_shape("gate_up_FUSED", 34816, 5120, iters, peak)?;
    // Attention proj (q/k/v/o) for reference — smaller.
    bench_shape("o_proj", 5120, 5120, iters, peak)?;
    // lm_head (full vocab) — the draft cost.
    bench_shape("lm_head", 248320, 5120, iters, peak)?;
    println!();
    // PIPELINED chain (true in-forward BW of mlx QMV — the decisive number for
    // "is mlx's matmul already BW-optimal or is there custom-kernel headroom").
    bench_mlp_chain(5120, 17408, 200, peak)?;
    Ok(())
}
