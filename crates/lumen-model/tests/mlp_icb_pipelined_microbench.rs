//! pipelined (real-decode-mode) ICB-vs-standard.
//!
//! Production decode does NOT synchronize per MLP block. The whole token's
//! 640+ dispatches are pipelined into one command buffer, with one final
//! synchronize at sample time. This microbench measures `LUMEN_MLP_ICB=0`
//! vs `=1` in that mode, exposing what we actually save in real decode.
//!
//! Run:
//!   cargo test --test mlp_icb_pipelined_microbench -p lumen-model \
//!     --release -- --nocapture --test-threads=1

#![cfg(feature = "turboquant-gpu")]

use candle_core::backend::BackendDevice as _;
use candle_core::{DType, Device, Tensor};
use lumen_metal::affine4_gpu::{Affine4Context, Affine4Weight};
use lumen_metal::affine4_linear::Affine4Linear;
use lumen_model::qwen3_5_moe::moe::DenseMlp;
use lumen_model::qwen3_5_moe::proj::ProjLinear;
use std::sync::Arc;
use std::time::Instant;

// Shape source: mlx-community/Qwen3.6-27B-4bit config.json text_config
//   hidden_size = 5120
//   intermediate_size = 17408   (was previously 25600 — fictional, see anti-pattern #25)
//   layer_types = [linear_attention × 3, full_attention × 1] × 16  (HYBRID, not Dense)
const HIDDEN: usize = 5120;
const INTER: usize = 17408;
const ITERS: usize = 1000;
const WARMUP: usize = 50;

fn synth_packed(out: usize, ins: usize, seed: u32) -> Vec<u32> {
    let n = out * ins / 8;
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            s
        })
        .collect()
}

fn synth_scales_or_biases(out: usize, ins: usize, seed: u32, neg: bool) -> Vec<u16> {
    let n = out * ins / 64;
    let mut s = seed;
    let off = if neg { -0.005 } else { 0.01 };
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            let f = ((s >> 8) & 0xff) as f32 / 256.0 * 0.01 + off;
            (f.to_bits() >> 16) as u16
        })
        .collect()
}

fn synth_x(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 8) & 0xff) as f32 / 256.0 - 0.5
        })
        .collect()
}

#[test]
fn mlp_icb_vs_standard_pipelined() {
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

    let gate_up_packed = synth_packed(2 * INTER, HIDDEN, 0xDEAD_BEEF);
    let gate_up_scales = synth_scales_or_biases(2 * INTER, HIDDEN, 0xCAFE_BABE, false);
    let gate_up_biases = synth_scales_or_biases(2 * INTER, HIDDEN, 0x1234_5678, true);
    let gate_up_w = Affine4Weight::from_host(
        &ctx.ctx,
        &gate_up_packed,
        &gate_up_scales,
        &gate_up_biases,
        2 * INTER,
        HIDDEN,
    )
    .expect("gate_up weight");
    let down_packed = synth_packed(HIDDEN, INTER, 0xFADE_FADE);
    let down_scales = synth_scales_or_biases(HIDDEN, INTER, 0xBEEF_BEEF, false);
    let down_biases = synth_scales_or_biases(HIDDEN, INTER, 0xC0DE_C0DE, true);
    let down_w = Affine4Weight::from_host(
        &ctx.ctx,
        &down_packed,
        &down_scales,
        &down_biases,
        HIDDEN,
        INTER,
    )
    .expect("down weight");

    let gate_up_lin = Affine4Linear::new(gate_up_w, None, ctx.clone());
    let down_lin = Affine4Linear::new(down_w, None, ctx.clone());
    let mlp = DenseMlp::new(
        ProjLinear::Affine4(gate_up_lin),
        ProjLinear::Affine4(down_lin),
        INTER,
    );

    let x_data = synth_x(HIDDEN, 0xAAAA_BBBB);
    let x = Tensor::from_vec(x_data, &[1, 1, HIDDEN], &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();
    let r_data = synth_x(HIDDEN, 0x9999_8888);
    let residual = Tensor::from_vec(r_data, &[1, 1, HIDDEN], &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();

    let metal_dev = match x.device() {
        Device::Metal(m) => m,
        _ => unreachable!(),
    };

    // Standard pipelined (no per-call synchronize, 1 final sync).
    unsafe {
        std::env::set_var("LUMEN_MLP_ICB", "0");
    }
    for _ in 0..WARMUP {
        let _ = mlp
            .forward_with_residual_bf16_in_bf16_out(&x, &residual)
            .unwrap();
    }
    let _ = metal_dev.synchronize();
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = mlp
            .forward_with_residual_bf16_in_bf16_out(&x, &residual)
            .unwrap();
    }
    let _ = metal_dev.synchronize();
    let std_pipelined = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    // ICB pipelined.
    unsafe {
        std::env::set_var("LUMEN_MLP_ICB", "1");
    }
    for _ in 0..WARMUP {
        let _ = mlp
            .forward_with_residual_bf16_in_bf16_out(&x, &residual)
            .unwrap();
    }
    let _ = metal_dev.synchronize();
    let t1 = Instant::now();
    for _ in 0..ITERS {
        let _ = mlp
            .forward_with_residual_bf16_in_bf16_out(&x, &residual)
            .unwrap();
    }
    let _ = metal_dev.synchronize();
    let icb_pipelined = t1.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    let pct = (icb_pipelined - std_pipelined) / std_pipelined * 100.0;

    eprintln!();
    eprintln!("=== Pipelined ICB vs Standard (real-decode mode) ===");
    eprintln!("Iterations:           {ITERS} per arm ({WARMUP} warmup)");
    eprintln!("Standard pipelined:   {std_pipelined:.1} µs/MLP-block");
    eprintln!("ICB pipelined:        {icb_pipelined:.1} µs/MLP-block");
    eprintln!("Δ:                    {pct:+.2}%   (negative = ICB faster)");
    eprintln!();
    eprintln!("Reference: per-call-sync mode showed -2.00% σ=+2.60");
    eprintln!("THIS pipelined mode is what real model decode actually does.");
    eprintln!();
    eprintln!("Throughput projection for full token (rough):");
    let saved_per_mlp = std_pipelined - icb_pipelined;
    let saved_per_token_us = saved_per_mlp * 64.0; // 64 layers
    let new_token_ms = (67.0 - saved_per_token_us / 1000.0).max(1.0);
    let new_tps = 1000.0 / new_token_ms;
    eprintln!("  Δ per MLP-block:     {saved_per_mlp:+.2} µs");
    eprintln!(
        "  Δ per token (×64):   {:+.2} µs ({:+.2} ms)",
        saved_per_token_us,
        saved_per_token_us / 1000.0
    );
    eprintln!("  Decode time:         67 ms → {new_token_ms:.2} ms");
    eprintln!("  Throughput:          15.04 → {new_tps:.2} tok/s");
}
