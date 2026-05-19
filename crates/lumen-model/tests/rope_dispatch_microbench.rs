//! RoPE dispatch microbench.
//!
//! Production native fused path (`self_attn`) issues RoPE for Q and K inside
//! the same command buffer as RmsNorm + transpose + attention. Each full-attn
//! layer = 2 RoPE dispatches (q + k). 16 full-attn layers per token = 32 RoPE
//! dispatches per token.
//!
//! MLX's `mx.fast.rope` is also single-dispatch — same kernel-level design.
//! This bench measures the actual GPU time + dispatch overhead so we can rule
//! it in or out as a 16.6% gap source.
//!
//! Shape (27B-4bit Qwen3.6):
//!   Q: [1, 1, num_attention_heads=24, head_dim=256], half_d = 32
//!   K: [1, 1, num_key_value_heads=4, head_dim=256], half_d = 32
//!
//! Run:
//!   cargo test --test rope_dispatch_microbench -p lumen-model \
//!     --features turboquant-gpu --release -- --nocapture --test-threads=1

#![cfg(feature = "turboquant-gpu")]

use candle_core::Device;
use lumen_metal::metal::CommandBufferExt;
use lumen_model::qwen3_5_moe_native::{NativeDType, shared_native_resources_for};
use std::time::Instant;

const HEAD_DIM: usize = 256;
const HALF_D: usize = 32; // head_dim * partial_rotary_factor / 2 = 256*0.25/2 = 32
const Q_HEADS: usize = 24;
const K_HEADS: usize = 4;
const N_FUL_LAYERS: usize = 16;
const PRODUCTION_DECODE_MS: f64 = 67.0;
const ITERS: usize = 1000;

#[test]
fn rope_dispatch_microbench() {
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
            eprintln!("[skip] {e}");
            return;
        }
    };
    let res = res_mutex.lock().unwrap();
    let ctx = &res.ctx;
    let lib = &res.lib;

    // Buffers
    let q = ctx
        .zeros(vec![1, 1, Q_HEADS, HEAD_DIM], NativeDType::F32)
        .unwrap();
    let q_out = ctx
        .zeros(vec![1, 1, Q_HEADS, HEAD_DIM], NativeDType::F32)
        .unwrap();
    let k = ctx
        .zeros(vec![1, 1, K_HEADS, HEAD_DIM], NativeDType::F32)
        .unwrap();
    let k_out = ctx
        .zeros(vec![1, 1, K_HEADS, HEAD_DIM], NativeDType::F32)
        .unwrap();
    let cos = ctx.zeros(vec![1, HALF_D], NativeDType::F32).unwrap();
    let sin = ctx.zeros(vec![1, HALF_D], NativeDType::F32).unwrap();

    // ── Production-pattern: ITERS×(q+k) RoPE in single cmdbuf ──────────────
    let run_qk_loop = || {
        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc = cmd.auto_compute_encoder();
        enc.set_label("rope_qk_loop");
        for _ in 0..ITERS {
            lib.encode_rope_partial(&enc, &q, &cos, &sin, &q_out, HALF_D)
                .unwrap();
            lib.encode_rope_partial(&enc, &k, &cos, &sin, &k_out, HALF_D)
                .unwrap();
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    };
    run_qk_loop();
    let t0 = Instant::now();
    run_qk_loop();
    let avg_qk_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    // ── Q-only loop ────────────────────────────────────────────────────────
    let run_q_loop = || {
        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc = cmd.auto_compute_encoder();
        enc.set_label("rope_q_loop");
        for _ in 0..ITERS {
            lib.encode_rope_partial(&enc, &q, &cos, &sin, &q_out, HALF_D)
                .unwrap();
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    };
    run_q_loop();
    let t0 = Instant::now();
    run_q_loop();
    let avg_q_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    let avg_k_us = avg_qk_us - avg_q_us;
    let per_layer_rope_us = avg_qk_us;
    let per_token_rope_ms = per_layer_rope_us * N_FUL_LAYERS as f64 / 1000.0;
    let per_token_rope_pct = per_token_rope_ms / PRODUCTION_DECODE_MS * 100.0;

    eprintln!();
    eprintln!("=== Phase 19.A.4.2 RoPE dispatch microbench (production-pattern) ===");
    eprintln!("  shape: Q=[1,1,{Q_HEADS},{HEAD_DIM}] K=[1,1,{K_HEADS},{HEAD_DIM}] half_d={HALF_D}");
    eprintln!("  {ITERS} dispatches per cmdbuf");
    eprintln!();
    eprintln!("  Q+K per-layer RoPE dispatch time         : {avg_qk_us:.2} µs");
    eprintln!("  Q only                                   : {avg_q_us:.2} µs");
    eprintln!("  K only (derived)                         : {avg_k_us:.2} µs");
    eprintln!();
    eprintln!(
        "  RoPE total per-token (× {N_FUL_LAYERS} full-attn layers): {per_token_rope_ms:.3} ms"
    );
    eprintln!(
        "  % of {PRODUCTION_DECODE_MS:.0} ms decode                       : {per_token_rope_pct:.2}%"
    );
    eprintln!();
    eprintln!(
        "  → Even ELIMINATING RoPE entirely (impossible) saves only {per_token_rope_pct:.2}%."
    );
    eprintln!("  → Any fusion / dispatch optimization would save FRACTION of this.");
}
