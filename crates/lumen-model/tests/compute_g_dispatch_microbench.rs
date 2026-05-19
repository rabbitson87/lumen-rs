//! compute_g 5-dispatch microbench.
//!
//! Production native fused path (`forward_post_conv_fused`) issues 5 element-wise
//! dispatches for the GatedDeltaNet g/beta chain inside a SINGLE command buffer:
//!   sigmoid → broadcast_add_per_head → softplus → mul_broadcast_per_head → neg_exp
//!
//! MLX's `compute_g` is `@partial(mx.compile, shapeless=True)` — single fused
//! Metal kernel (1 dispatch). This bench measures the actual saving achievable
//! by collapsing 5 → 1, by isolating the chain in its own command buffer and
//! comparing 5-dispatch vs 1-dispatch (sigmoid alone, as a lower bound on the
//! single-fused-kernel time).
//!
//! Shape: Hv = 48 (linear_num_value_heads for 27B-4bit Qwen3.6).
//! Layers: 48 linear-attention layers per token.
//! Production decode baseline: 67 ms/token = 15.04 tok/s.
//!
//! Run:
//!   cargo test --test compute_g_dispatch_microbench -p lumen-model \
//!     --features turboquant-gpu --release -- --nocapture --test-threads=1

#![cfg(feature = "turboquant-gpu")]

use candle_core::Device;
use lumen_metal::metal::CommandBufferExt;
use lumen_model::qwen3_5_moe_native::{NativeDType, shared_native_resources_for};
use std::time::Instant;

const HV: usize = 48;
const N_LIN_LAYERS: usize = 48;
const PRODUCTION_DECODE_MS: f64 = 67.0;
const ITERS: usize = 1000;
const WARMUP: usize = 50;

#[test]
fn compute_g_dispatch_microbench() {
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("[skip] no Metal device");
            return;
        }
    };

    let res_mutex = match shared_native_resources_for(&dev) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[skip] shared_native_resources_for: {e}");
            return;
        }
    };
    let res = res_mutex.lock().unwrap();
    let ctx = &res.ctx;
    let lib = &res.lib;

    // [B=1, S=1, Hv=48] tensors for the chain.
    let b_in = ctx.zeros(vec![1, 1, HV], NativeDType::F32).unwrap();
    let a_in = ctx.zeros(vec![1, 1, HV], NativeDType::F32).unwrap();
    let dt_bias = ctx.zeros(vec![HV], NativeDType::F32).unwrap();
    let exp_a_log = ctx.zeros(vec![HV], NativeDType::F32).unwrap();
    let beta = ctx.zeros(vec![1, 1, HV], NativeDType::F32).unwrap();
    let a_plus_dt = ctx.zeros(vec![1, 1, HV], NativeDType::F32).unwrap();
    let softplus_a = ctx.zeros(vec![1, 1, HV], NativeDType::F32).unwrap();
    let g_pre = ctx.zeros(vec![1, 1, HV], NativeDType::F32).unwrap();
    let g = ctx.zeros(vec![1, 1, HV], NativeDType::F32).unwrap();

    // ── Run #1: full 5-dispatch chain (current production) ─────────────────
    // Single-cmdbuf pattern: ITERS dispatches inside one command buffer,
    // single commit + wait. Per-iter cost = (total - cmdbuf_overhead) / ITERS.
    // Mimics production where 5 dispatches share `forward_post_conv_fused`'s cmdbuf.
    let run_5_loop = || {
        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc = cmd.auto_compute_encoder();
        enc.set_label("compute_g_5dispatch_loop");
        for _ in 0..ITERS {
            lib.encode_sigmoid(&enc, &b_in, &beta).unwrap();
            lib.encode_broadcast_add_per_head(&enc, &a_in, &dt_bias, &a_plus_dt)
                .unwrap();
            lib.encode_softplus(&enc, &a_plus_dt, &softplus_a).unwrap();
            lib.encode_mul_broadcast_per_head(&enc, &softplus_a, &exp_a_log, &g_pre)
                .unwrap();
            lib.encode_neg_exp(&enc, &g_pre, &g).unwrap();
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    };
    run_5_loop(); // warmup
    let t0 = Instant::now();
    run_5_loop();
    let avg_5_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    let run_1_loop = || {
        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc = cmd.auto_compute_encoder();
        enc.set_label("compute_g_1dispatch_loop");
        for _ in 0..ITERS {
            lib.encode_sigmoid(&enc, &b_in, &beta).unwrap();
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    };
    run_1_loop();
    let t0 = Instant::now();
    run_1_loop();
    let avg_1_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    // ── Run #fused: compute_g_full (5→1 fusion) loop in single cmdbuf ───────
    let beta_f = ctx.zeros(vec![1, 1, HV], NativeDType::F32).unwrap();
    let g_f = ctx.zeros(vec![1, 1, HV], NativeDType::F32).unwrap();
    let run_fused_loop = || {
        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc = cmd.auto_compute_encoder();
        enc.set_label("compute_g_fused_loop");
        for _ in 0..ITERS {
            lib.encode_compute_g_full(&enc, &b_in, &a_in, &dt_bias, &exp_a_log, &beta_f, &g_f)
                .unwrap();
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    };
    run_fused_loop();
    let t0 = Instant::now();
    run_fused_loop();
    let avg_fused_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    // Empty cmdbuf — measures commit/wait overhead alone.
    let run_0_loop = || {
        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc = cmd.auto_compute_encoder();
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    };
    for _ in 0..WARMUP {
        run_0_loop();
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        run_0_loop();
    }
    let avg_0_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    // ── Parity check: fused beta/g == reference beta/g ──────────────────────
    // Re-run the 5-dispatch chain into beta/g, then run fused into beta_f/g_f.
    // Compare element-wise.
    {
        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc = cmd.auto_compute_encoder();
        lib.encode_sigmoid(&enc, &b_in, &beta).unwrap();
        lib.encode_broadcast_add_per_head(&enc, &a_in, &dt_bias, &a_plus_dt)
            .unwrap();
        lib.encode_softplus(&enc, &a_plus_dt, &softplus_a).unwrap();
        lib.encode_mul_broadcast_per_head(&enc, &softplus_a, &exp_a_log, &g_pre)
            .unwrap();
        lib.encode_neg_exp(&enc, &g_pre, &g).unwrap();
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let cmd2 = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc2 = cmd2.auto_compute_encoder();
        lib.encode_compute_g_full(&enc2, &b_in, &a_in, &dt_bias, &exp_a_log, &beta_f, &g_f)
            .unwrap();
        enc2.end_encoding();
        cmd2.commit();
        cmd2.wait_until_completed();

        let beta_ref = beta.to_vec_f32().unwrap();
        let beta_new = beta_f.to_vec_f32().unwrap();
        let g_ref = g.to_vec_f32().unwrap();
        let g_new = g_f.to_vec_f32().unwrap();
        let max_db = beta_ref
            .iter()
            .zip(&beta_new)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_dg = g_ref
            .iter()
            .zip(&g_new)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!();
        eprintln!("  Parity check (fused vs reference):");
        eprintln!("    max|Δbeta| = {max_db:.3e}");
        eprintln!("    max|Δg|    = {max_dg:.3e}");
    }

    // ── Analysis ────────────────────────────────────────────────────────────
    // Saving from 5→1 fusion ≈ avg_5_us − avg_1_us per linear-attn layer.
    // Production has 48 linear-attn layers per token.
    let saving_per_layer_us = (avg_5_us - avg_1_us).max(0.0);
    let saving_per_token_ms = saving_per_layer_us * N_LIN_LAYERS as f64 / 1000.0;
    let saving_pct = saving_per_token_ms / PRODUCTION_DECODE_MS * 100.0;

    let dispatch_overhead_us = avg_5_us / 5.0;

    eprintln!();
    eprintln!("=== Phase 19.A.4.1 compute_g dispatch microbench (production-pattern) ===");
    eprintln!("  shape: [B=1, S=1, Hv={HV}] f32, {ITERS} dispatches per cmdbuf");
    eprintln!();
    eprintln!("  Run #1: 5-dispatch loop (production)        : {avg_5_us:.2} µs/dispatch-set");
    eprintln!("  Run #2: 1-dispatch loop (sigmoid alone)     : {avg_1_us:.2} µs/dispatch");
    eprintln!("  Run #fused: compute_g_full loop (NEW 5→1)   : {avg_fused_us:.2} µs/dispatch");
    eprintln!("  Run #3: 0-dispatch (commit/wait only)        : {avg_0_us:.2} µs/cmdbuf-roundtrip");
    let realised_saving_us = (avg_5_us - avg_fused_us).max(0.0);
    let realised_per_token_ms = realised_saving_us * N_LIN_LAYERS as f64 / 1000.0;
    let realised_pct = realised_per_token_ms / PRODUCTION_DECODE_MS * 100.0;
    eprintln!();
    eprintln!(
        "  REALISED 5→1 saving (microbench)         : {realised_saving_us:.2} µs/layer × {N_LIN_LAYERS} = {realised_per_token_ms:.3} ms/token = {realised_pct:.2}%"
    );
    eprintln!();
    eprintln!("  Per-dispatch overhead inside cmdbuf      : {dispatch_overhead_us:.2} µs");
    eprintln!("  5→1 fusion saving per linear-attn layer  : {saving_per_layer_us:.2} µs");
    eprintln!("  Per-token saving (× {N_LIN_LAYERS} layers)        : {saving_per_token_ms:.3} ms");
    eprintln!(
        "  % of {PRODUCTION_DECODE_MS:.0} ms decode                       : {saving_pct:.2}%"
    );
    eprintln!();
}
