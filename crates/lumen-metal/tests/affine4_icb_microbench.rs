//! micro A/B for `forward_bf16_in_bf16_out` (standard) vs
//! `forward_bf16_in_bf16_out_icb` (ICB path with plural `useResources`).
//!
//! Question this answers: at N=1 (per-Linear ICB granularity), does the
//! useResource → useResources lift bring the ICB path within the standard
//! path's cost envelope?
//!
//! Pre-17.D-A baseline (PoC #1 N=1): -2.87% (ICB worse than standard) due
//! to per-buffer useResource FFI overhead.
//! Post-17.D-A target: 0 ± few percent (WASH or marginal WIN).
//!
//! Run:
//!   cargo test --test affine4_icb_microbench -p lumen-metal \
//!     --features model-integration --release -- --nocapture
//!
//! Note: this is a CPU-time microbench at the per-call level, not the full
//! σ bench. It isolates the dispatch encoding cost from kernel execution by
//! measuring `forward_*` round-trip including command buffer commit + wait.

#![cfg(feature = "model-integration")]

use candle_core::{DType, Device, Tensor};
use std::sync::Arc;
use std::time::Instant;
use lumen_metal::affine4_gpu::{Affine4Context, Affine4Weight};
use lumen_metal::affine4_linear::Affine4Linear;

const OUT: usize = 5120;
const IN: usize = 5120;
const WARMUP: usize = 50;
const ITERS: usize = 200;

fn synth_packed(out: usize, ins: usize, seed: u32) -> Vec<u32> {
    let n = out * ins / 8;
    let mut s = seed;
    (0..n).map(|_| { s = s.wrapping_mul(1103515245).wrapping_add(12345); s }).collect()
}

fn synth_scales_or_biases(out: usize, ins: usize, seed: u32, neg: bool) -> Vec<u16> {
    let n = out * ins / 64;
    let mut s = seed;
    let off = if neg { -0.005 } else { 0.01 };
    (0..n).map(|_| {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        let f = ((s >> 8) & 0xff) as f32 / 256.0 * 0.01 + off;
        (f.to_bits() >> 16) as u16
    }).collect()
}

fn synth_x(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        ((s >> 8) & 0xff) as f32 / 256.0 - 0.5
    }).collect()
}

fn welchs_t(a: &[f64], b: &[f64]) -> f64 {
    let na = a.len() as f64;
    let nb = b.len() as f64;
    let ma = a.iter().sum::<f64>() / na;
    let mb = b.iter().sum::<f64>() / nb;
    let va = a.iter().map(|x| (x - ma).powi(2)).sum::<f64>() / (na - 1.0);
    let vb = b.iter().map(|x| (x - mb).powi(2)).sum::<f64>() / (nb - 1.0);
    let se = (va / na + vb / nb).sqrt();
    if se == 0.0 { 0.0 } else { (ma - mb) / se }
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|x, y| x.partial_cmp(y).unwrap());
    s[s.len() / 2]
}

#[test]
fn forward_icb_vs_standard_micro_at_n1() {
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("[skip] no Metal device");
            return;
        }
    };
    let ctx = match Affine4Context::new() {
        Ok(c) => Arc::new(c),
        Err(_) => {
            eprintln!("[skip] no Affine4 context");
            return;
        }
    };

    let packed = synth_packed(OUT, IN, 0xDEADBEEF);
    let scales = synth_scales_or_biases(OUT, IN, 0xCAFEBABE, false);
    let biases = synth_scales_or_biases(OUT, IN, 0x12345678, true);
    let weight = Affine4Weight::from_host(&ctx.ctx, &packed, &scales, &biases, OUT, IN)
        .expect("weight upload");
    let linear = Affine4Linear::new(weight, None, ctx.clone());

    let x_data = synth_x(IN, 0xFADEFADE);
    let x = Tensor::from_vec(x_data, &[1, 1, IN], &dev).unwrap()
        .to_dtype(DType::BF16).unwrap()
        .contiguous().unwrap();

    use candle_core::backend::BackendDevice as _;

    // ── Warmup both paths ──────────────────────────────────────────────
    unsafe { std::env::set_var("LUMEN_ICB", "0"); }
    for _ in 0..WARMUP {
        let _ = linear.forward_bf16_in_bf16_out(&x).unwrap();
    }
    if let Device::Metal(md) = &dev { let _ = md.synchronize(); }

    unsafe { std::env::set_var("LUMEN_ICB", "1"); }
    for _ in 0..WARMUP {
        let _ = linear.forward_bf16_in_bf16_out_icb(&x).unwrap();
    }
    if let Device::Metal(md) = &dev { let _ = md.synchronize(); }

    // ── Interleaved measurement to neutralise thermal/scheduler bias ──
    let mut t_std: Vec<f64> = Vec::with_capacity(ITERS);
    let mut t_icb: Vec<f64> = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        unsafe { std::env::set_var("LUMEN_ICB", "0"); }
        let t0 = Instant::now();
        let y_std = linear.forward_bf16_in_bf16_out(&x).unwrap();
        if let Device::Metal(md) = y_std.device() { let _ = md.synchronize(); }
        t_std.push(t0.elapsed().as_secs_f64() * 1e6);

        unsafe { std::env::set_var("LUMEN_ICB", "1"); }
        let t1 = Instant::now();
        let y_icb = linear.forward_bf16_in_bf16_out_icb(&x).unwrap();
        if let Device::Metal(md) = y_icb.device() { let _ = md.synchronize(); }
        t_icb.push(t1.elapsed().as_secs_f64() * 1e6);
    }

    let mean_std = t_std.iter().sum::<f64>() / ITERS as f64;
    let mean_icb = t_icb.iter().sum::<f64>() / ITERS as f64;
    let med_std = median(&t_std);
    let med_icb = median(&t_icb);
    // Sign convention: positive σ = ICB faster.
    let sigma = welchs_t(&t_std, &t_icb);
    let pct = (med_icb - med_std) / med_std * 100.0;

    eprintln!();
    eprintln!("=== Phase 17.D-A — forward_*_icb (N=1) vs forward_* (standard) ===");
    eprintln!("Linear shape:    out={OUT} in={IN}, batch=1");
    eprintln!("Iterations:      {ITERS} per variant (interleaved, {WARMUP} warmup)");
    eprintln!("Standard (μ/med): {:.3} / {:.3} µs", mean_std, med_std);
    eprintln!("ICB plural (μ/med): {:.3} / {:.3} µs", mean_icb, med_icb);
    eprintln!("Δ (med):         {:+.2}% (negative = ICB faster than standard)", pct);
    eprintln!("Welch's t σ:     {:+.2}  (positive σ = ICB faster)", sigma);
    eprintln!();
    eprintln!("Reference baselines:");
    eprintln!("  PoC #1 N=1 pre-17.D-A:   -2.87%");
    eprintln!("  Phase 17.D-A target:     0 ± 3% (WASH or marginal WIN)");

    // Soft assertion — we don't fail the test on σ; this is a measurement.
    // The test exists to surface the data, not gate the pipeline.
}

/// More representative microbench: K consecutive forwards within ONE
/// command buffer (synchronize only at the end). This matches real decode
/// where ~256 Affine4 dispatches share a single encoder before commit.
/// Encoder-creation amortisation reveals the ICB validation savings.
#[test]
fn forward_icb_vs_standard_micro_batched_n_calls() {
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("[skip] no Metal device");
            return;
        }
    };
    let ctx = match Affine4Context::new() {
        Ok(c) => Arc::new(c),
        Err(_) => {
            eprintln!("[skip] no Affine4 context");
            return;
        }
    };

    let packed = synth_packed(OUT, IN, 0xDEADBEEF);
    let scales = synth_scales_or_biases(OUT, IN, 0xCAFEBABE, false);
    let biases = synth_scales_or_biases(OUT, IN, 0x12345678, true);
    let weight = Affine4Weight::from_host(&ctx.ctx, &packed, &scales, &biases, OUT, IN)
        .expect("weight upload");
    // 64 distinct Affine4Linear instances mimic 64 layers; each will be hit
    // once per "decode step" so the per-Linear ICB cache fast-path is the
    // common case (not cold-record).
    const N_LINEARS: usize = 64;
    let linears: Vec<Affine4Linear> = (0..N_LINEARS)
        .map(|_| Affine4Linear::new(
            Affine4Weight::from_host(&ctx.ctx, &packed, &scales, &biases, OUT, IN)
                .expect("weight upload"),
            None,
            ctx.clone(),
        ))
        .collect();

    let x_data = synth_x(IN, 0xFADEFADE);
    let x = Tensor::from_vec(x_data, &[1, 1, IN], &dev).unwrap()
        .to_dtype(DType::BF16).unwrap()
        .contiguous().unwrap();

    use candle_core::backend::BackendDevice as _;

    // Warmup both paths.
    unsafe { std::env::set_var("LUMEN_ICB", "0"); }
    for _ in 0..5 {
        for l in &linears { let _ = l.forward_bf16_in_bf16_out(&x).unwrap(); }
        if let Device::Metal(md) = &dev { let _ = md.synchronize(); }
    }
    unsafe { std::env::set_var("LUMEN_ICB", "1"); }
    for _ in 0..5 {
        for l in &linears { let _ = l.forward_bf16_in_bf16_out_icb(&x).unwrap(); }
        if let Device::Metal(md) = &dev { let _ = md.synchronize(); }
    }
    if let Device::Metal(md) = &dev { let _ = md.synchronize(); }

    const STEPS: usize = 50;
    let mut t_std: Vec<f64> = Vec::with_capacity(STEPS);
    let mut t_icb: Vec<f64> = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        unsafe { std::env::set_var("LUMEN_ICB", "0"); }
        let t0 = Instant::now();
        for l in &linears {
            let _ = l.forward_bf16_in_bf16_out(&x).unwrap();
        }
        if let Device::Metal(md) = &dev { let _ = md.synchronize(); }
        t_std.push(t0.elapsed().as_secs_f64() * 1e6);

        unsafe { std::env::set_var("LUMEN_ICB", "1"); }
        let t1 = Instant::now();
        for l in &linears {
            let _ = l.forward_bf16_in_bf16_out_icb(&x).unwrap();
        }
        if let Device::Metal(md) = &dev { let _ = md.synchronize(); }
        t_icb.push(t1.elapsed().as_secs_f64() * 1e6);
    }

    let mean_std = t_std.iter().sum::<f64>() / STEPS as f64;
    let mean_icb = t_icb.iter().sum::<f64>() / STEPS as f64;
    let med_std = median(&t_std);
    let med_icb = median(&t_icb);
    let sigma = welchs_t(&t_std, &t_icb);
    let pct = (med_icb - med_std) / med_std * 100.0;
    let per_call_delta_us = (med_icb - med_std) / N_LINEARS as f64;

    eprintln!();
    eprintln!("=== Phase 17.D-A — batched ({N_LINEARS} forwards × {STEPS} steps) ===");
    eprintln!("This is closer to real decode (~64-256 forwards share one encoder).");
    eprintln!("Standard (μ/med): {:.0} / {:.0} µs / step", mean_std, med_std);
    eprintln!("ICB plural (μ/med): {:.0} / {:.0} µs / step", mean_icb, med_icb);
    eprintln!("Δ (med):           {:+.2}% per step", pct);
    eprintln!("Δ per call:        {:+.2} µs / call", per_call_delta_us);
    eprintln!("Welch's t σ:       {:+.2}  (positive σ = ICB faster)", sigma);
    eprintln!();
    eprintln!("If σ-NEGATIVE here too → useResource lift is not enough,");
    eprintln!("must escalate to per-MLP-block ICB (17.D-B, N=3) or further.");
}
