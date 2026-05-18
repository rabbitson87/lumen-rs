//! measures σ at the
//! `SharedExpert::forward_with_residual_bf16_in_bf16_out` entry point
//! (the call site decoder layers actually use), with `LUMEN_MLP_ICB=0`
//! vs `=1`.
//!
//! Differs from `mlp_block_icb_poc` (which bypassed SharedExpert and
//! invoked the ICB chain directly): this measures the FULL production
//! overhead — env-gate check, shape qualification, Mutex lock, buffer
//! extraction via Storage::Metal match, lazy cache init — to confirm the
//! +3.84% PoC result holds end-to-end through the production wrapper.
//!
//! Run:
//!   cargo test --test mlp_icb_production_microbench -p lumen-model \
//!     --release -- --nocapture --test-threads=1

#![cfg(feature = "turboquant-gpu")]

use candle_core::backend::BackendDevice as _;
use candle_core::{DType, Device, Tensor};
use std::sync::Arc;
use std::time::Instant;
use lumen_metal::affine4_gpu::{Affine4Context, Affine4Weight};
use lumen_metal::affine4_linear::Affine4Linear;
use lumen_model::qwen3_5_moe::moe::DenseMlp;
use lumen_model::qwen3_5_moe::proj::ProjLinear;

// Shape source: mlx-community/Qwen3.6-27B-4bit config.json text_config
//   hidden_size = 5120, intermediate_size = 17408 (was 25600, fictional — anti-pattern #25)
const HIDDEN: usize = 5120;
const INTER: usize = 17408;
const ITERS: usize = 100;
const WARMUP: usize = 30;

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
fn mlp_icb_production_path_microbench() {
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

    // Build a fixture DenseMlp with Affine4 weights matching 27B Dense shapes.
    let gate_up_packed = synth_packed(2 * INTER, HIDDEN, 0xDEAD_BEEF);
    let gate_up_scales = synth_scales_or_biases(2 * INTER, HIDDEN, 0xCAFE_BABE, false);
    let gate_up_biases = synth_scales_or_biases(2 * INTER, HIDDEN, 0x1234_5678, true);
    let gate_up_w = Affine4Weight::from_host(
        &ctx.ctx, &gate_up_packed, &gate_up_scales, &gate_up_biases,
        2 * INTER, HIDDEN,
    ).expect("gate_up weight");
    let down_packed = synth_packed(HIDDEN, INTER, 0xFADE_FADE);
    let down_scales = synth_scales_or_biases(HIDDEN, INTER, 0xBEEF_BEEF, false);
    let down_biases = synth_scales_or_biases(HIDDEN, INTER, 0xC0DE_C0DE, true);
    let down_w = Affine4Weight::from_host(
        &ctx.ctx, &down_packed, &down_scales, &down_biases,
        HIDDEN, INTER,
    ).expect("down weight");

    let gate_up_lin = Affine4Linear::new(gate_up_w, None, ctx.clone());
    let down_lin = Affine4Linear::new(down_w, None, ctx.clone());

    let mlp = DenseMlp::new(
        ProjLinear::Affine4(gate_up_lin),
        ProjLinear::Affine4(down_lin),
        INTER,
    );

    // Inputs.
    let x_data = synth_x(HIDDEN, 0xAAAA_BBBB);
    let x = Tensor::from_vec(x_data, &[1, 1, HIDDEN], &dev).unwrap()
        .to_dtype(DType::BF16).unwrap()
        .contiguous().unwrap();
    let r_data = synth_x(HIDDEN, 0x9999_8888);
    let residual = Tensor::from_vec(r_data, &[1, 1, HIDDEN], &dev).unwrap()
        .to_dtype(DType::BF16).unwrap()
        .contiguous().unwrap();

    // ── Bit-identity gate ────────────────────────────────────────────
    unsafe { std::env::set_var("LUMEN_MLP_ICB", "0"); }
    let y_off = mlp.forward_with_residual_bf16_in_bf16_out(&x, &residual).unwrap();
    unsafe { std::env::set_var("LUMEN_MLP_ICB", "1"); }
    let y_on = mlp.forward_with_residual_bf16_in_bf16_out(&x, &residual).unwrap();

    let bits_off: Vec<u32> = y_off.flatten_all().unwrap()
        .to_dtype(DType::F32).unwrap()
        .to_vec1::<f32>().unwrap()
        .iter().map(|f| f.to_bits()).collect();
    let bits_on: Vec<u32> = y_on.flatten_all().unwrap()
        .to_dtype(DType::F32).unwrap()
        .to_vec1::<f32>().unwrap()
        .iter().map(|f| f.to_bits()).collect();
    let diffs = bits_off.iter().zip(bits_on.iter()).filter(|(a, b)| a != b).count();
    eprintln!();
    eprintln!("=== Production path bit-identity (off ↔ on) ===");
    eprintln!("Compared:  {} elements", bits_off.len());
    eprintln!("Diffs:     {diffs} {}", if diffs == 0 { "✓" } else { "✗ (drift)" });

    // ── Bench (interleaved) ──────────────────────────────────────────
    unsafe { std::env::set_var("LUMEN_MLP_ICB", "0"); }
    for _ in 0..WARMUP {
        let _ = mlp.forward_with_residual_bf16_in_bf16_out(&x, &residual).unwrap();
    }
    if let Device::Metal(md) = &dev { let _ = md.synchronize(); }
    unsafe { std::env::set_var("LUMEN_MLP_ICB", "1"); }
    for _ in 0..WARMUP {
        let _ = mlp.forward_with_residual_bf16_in_bf16_out(&x, &residual).unwrap();
    }
    if let Device::Metal(md) = &dev { let _ = md.synchronize(); }

    let mut t_off: Vec<f64> = Vec::with_capacity(ITERS);
    let mut t_on: Vec<f64> = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        unsafe { std::env::set_var("LUMEN_MLP_ICB", "0"); }
        let t0 = Instant::now();
        let y = mlp.forward_with_residual_bf16_in_bf16_out(&x, &residual).unwrap();
        if let Device::Metal(md) = y.device() { let _ = md.synchronize(); }
        t_off.push(t0.elapsed().as_secs_f64() * 1e6);

        unsafe { std::env::set_var("LUMEN_MLP_ICB", "1"); }
        let t1 = Instant::now();
        let y = mlp.forward_with_residual_bf16_in_bf16_out(&x, &residual).unwrap();
        if let Device::Metal(md) = y.device() { let _ = md.synchronize(); }
        t_on.push(t1.elapsed().as_secs_f64() * 1e6);
    }

    let mean_off = t_off.iter().sum::<f64>() / ITERS as f64;
    let mean_on = t_on.iter().sum::<f64>() / ITERS as f64;
    let med_off = median(&t_off);
    let med_on = median(&t_on);
    let sigma = welchs_t(&t_off, &t_on);
    let pct = (med_on - med_off) / med_off * 100.0;

    eprintln!();
    eprintln!("=== Phase 17.D-1f Production path A/B ===");
    eprintln!("Iterations:           {ITERS} per arm ({WARMUP} warmup)");
    eprintln!("LUMEN_MLP_ICB=0:    µ {mean_off:.0} / med {med_off:.0} µs");
    eprintln!("LUMEN_MLP_ICB=1:    µ {mean_on:.0} / med {med_on:.0} µs");
    eprintln!("Δ (med):              {pct:+.2}%   (negative = ICB faster)");
    eprintln!("Welch's t σ:          {sigma:+.2}  (positive σ = ICB faster)");
    eprintln!();
    eprintln!("Reference: PoC mlp_block_icb_poc disambiguation result");
    eprintln!("           ICB vs standard: -3.84% σ=+2.86");
    eprintln!("           ICB vs fused-no-ICB: -4.37% σ=+3.47");
}
