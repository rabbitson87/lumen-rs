//! Stage 2 — per-shape BW utilization diagnostic for 27B-4bit hybrid model.
//!
//! For each dominant Affine4 matmul shape in mlx-community/Qwen3.6-27B-4bit
//! decode, measure pipelined µs/forward and effective BW, projecting to the
//! production 67 ms/token decode budget.
//!
//! Goal: identify which kernels are BW-bound (3-bit quant lever candidate)
//! vs compute-bound (kernel arithmetic / occupancy lever candidate).
//!
//! Shape source (verified from config.json text_config):
//!   hidden_size = 5120
//!   intermediate_size = 17408         (gate_up out = 34816 = inter*2)
//!   head_dim = 256
//!   num_attention_heads = 24          (full-attn Q heads)
//!   num_key_value_heads = 4           (full-attn KV heads, GQA 6:1)
//!   linear_num_value_heads = 48       (Mamba V heads)
//!   linear_num_key_heads = 16         (Mamba K heads)
//!   linear_value_head_dim = 128
//!   linear_key_head_dim = 128
//!   layer_types = [linear_attention × 3, full_attention × 1] × 16
//!     → 48 lin layers + 16 ful layers, full_attention_interval = 4
//!
//! Derived projection shapes:
//!   ful qkv    : 5120 → 8192   (24*256 + 4*256 + 4*256 = 6144+1024+1024)
//!   o_proj     : 6144 → 5120   (24*256 = 6144) — same shape ful & lin out_proj
//!   lin in_proj: 5120 → 16480  (Option M fused: qkv_dim 10240 + v_dim 6144 + 2*Hv 96)
//!   gate_up    : 5120 → 34816  (gate + up = inter*2)
//!   down       : 17408 → 5120
//!
//! Run:
//!   cargo test --test stage2_per_shape_bw_diagnostic -p lumen-metal \
//!     --features model-integration --release -- --nocapture --test-threads=1

#![cfg(feature = "model-integration")]

use candle_core::{DType, Device, Tensor, backend::BackendDevice as _};
use lumen_metal::affine4_gpu::{Affine4Context, Affine4Weight};
use lumen_metal::affine4_linear::Affine4Linear;
use std::sync::Arc;
use std::time::Instant;

const HIDDEN: usize = 5120;
const INTER: usize = 17408;

const N_LIN_LAYERS: usize = 48;
const N_FUL_LAYERS: usize = 16;
const N_TOTAL_LAYERS: usize = 64;

const PRODUCTION_DECODE_MS: f64 = 67.0;
const M3_MAX_PEAK_BW_GBS: f64 = 400.0;

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

fn synth_meta(out: usize, ins: usize, seed: u32, neg: bool) -> Vec<u16> {
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

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    out: usize,
    ins: usize,
    /// Forward calls per decode token (sum of layers carrying this kernel).
    calls_per_token: usize,
}

fn measure_shape(ctx: &Arc<Affine4Context>, dev: &Device, shape: Shape) -> (f64, f64, f64, f64) {
    // Build a fixture Affine4 weight + linear wrapper.
    let packed = synth_packed(shape.out, shape.ins, 0xDEAD_BEEF ^ shape.name.len() as u32);
    let scales = synth_meta(shape.out, shape.ins, 0xCAFE_BABE, false);
    let biases = synth_meta(shape.out, shape.ins, 0x1234_5678, true);
    let weight =
        Affine4Weight::from_host(&ctx.ctx, &packed, &scales, &biases, shape.out, shape.ins)
            .expect("affine4 weight");
    let lin = Affine4Linear::new(weight, None, ctx.clone());

    // Activation in bf16 (production decode dtype).
    let x_data = synth_x(shape.ins, 0xAAAA_BBBB);
    let x = Tensor::from_vec(x_data, &[1, 1, shape.ins], dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();

    let metal_dev = match x.device() {
        Device::Metal(m) => m,
        _ => unreachable!(),
    };

    // Warmup.
    for _ in 0..WARMUP {
        let _ = lin.forward_bf16_in_bf16_out(&x).unwrap();
    }
    let _ = metal_dev.synchronize();

    // Pipelined timing: ITERS forwards, single sync at end.
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = lin.forward_bf16_in_bf16_out(&x).unwrap();
    }
    let _ = metal_dev.synchronize();
    let pipelined_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    // Bytes read per forward:
    //   packed (4-bit): out * ins / 2
    //   scales (bf16) + biases (bf16) at group=64: out * ins / 64 * 4
    //   activation (bf16): ins * 2
    //   output (bf16): out * 2  (write, but BW counts both R+W on M3)
    let packed_bytes = shape.out * shape.ins / 2;
    let meta_bytes = (shape.out * shape.ins / 64) * 4;
    let act_bytes = shape.ins * 2;
    let out_bytes = shape.out * 2;
    let total_bytes = packed_bytes + meta_bytes + act_bytes + out_bytes;

    let effective_bw_gbs = total_bytes as f64 / pipelined_us / 1e3;
    let bw_util_pct = effective_bw_gbs / M3_MAX_PEAK_BW_GBS * 100.0;

    let token_contribution_ms = pipelined_us * shape.calls_per_token as f64 / 1000.0;

    (
        pipelined_us,
        effective_bw_gbs,
        bw_util_pct,
        token_contribution_ms,
    )
}

#[test]
fn stage2_per_shape_bw() {
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

    // 27B-4bit dominant Affine4 matmul shapes per decode token.
    //
    // calls_per_token reflects how many times the kernel is executed during
    // ONE token's decode forward (not the count of distinct weights, but the
    // dispatch count this shape contributes to wall-time).
    let shapes = [
        Shape {
            name: "ful qkv (5120 → 8192)",
            out: 8192,
            ins: HIDDEN,
            calls_per_token: N_FUL_LAYERS,
        },
        Shape {
            name: "ful o_proj (6144 → 5120)",
            out: HIDDEN,
            ins: 6144,
            calls_per_token: N_FUL_LAYERS,
        },
        Shape {
            name: "lin in_proj (5120 → 16480)",
            out: 16480,
            ins: HIDDEN,
            calls_per_token: N_LIN_LAYERS,
        },
        Shape {
            name: "lin out_proj (6144 → 5120)",
            out: HIDDEN,
            ins: 6144,
            calls_per_token: N_LIN_LAYERS,
        },
        Shape {
            name: "MLP gate_up (5120 → 34816)",
            out: 2 * INTER,
            ins: HIDDEN,
            calls_per_token: N_TOTAL_LAYERS,
        },
        Shape {
            name: "MLP down (17408 → 5120)",
            out: HIDDEN,
            ins: INTER,
            calls_per_token: N_TOTAL_LAYERS,
        },
    ];

    eprintln!();
    eprintln!("=== Stage 2: 27B-4bit per-shape pipelined BW utilization ===");
    eprintln!("Iterations: {ITERS} per shape ({WARMUP} warmup, single sync at end)");
    eprintln!("Production decode baseline: {PRODUCTION_DECODE_MS} ms/token = 15.04 tok/s");
    eprintln!("M3 Max peak BW: {M3_MAX_PEAK_BW_GBS} GB/s");
    eprintln!();

    eprintln!(
        "{:<32} {:>10} {:>9} {:>9} {:>10} {:>9}",
        "shape", "µs/call", "GB/s", "BW %", "calls/tok", "ms/tok"
    );
    eprintln!("{}", "─".repeat(85));

    let mut total_token_ms: f64 = 0.0;
    let mut shape_results: Vec<(Shape, f64, f64, f64, f64)> = Vec::new();
    for shape in shapes {
        let (us, bw_gbs, bw_pct, ms_tok) = measure_shape(&ctx, &dev, shape);
        eprintln!(
            "{:<32} {:>10.1} {:>9.1} {:>8.1}% {:>10} {:>9.2}",
            shape.name, us, bw_gbs, bw_pct, shape.calls_per_token, ms_tok
        );
        total_token_ms += ms_tok;
        shape_results.push((shape, us, bw_gbs, bw_pct, ms_tok));
    }
    eprintln!("{}", "─".repeat(85));
    eprintln!(
        "{:<32} {:>10} {:>9} {:>9} {:>10} {:>9.2}",
        "Σ matmul / token", "—", "—", "—", "—", total_token_ms
    );

    eprintln!();
    eprintln!("=== Production decode budget breakdown ===");
    eprintln!(
        "Σ Affine4 matmul (all shapes) per token: {:.2} ms ({:.1}% of {} ms)",
        total_token_ms,
        total_token_ms / PRODUCTION_DECODE_MS * 100.0,
        PRODUCTION_DECODE_MS
    );
    eprintln!(
        "Remaining (non-matmul: SSM scan, conv1d, RoPE, SDPA, RmsNorm, residual, sample): {:.2} ms ({:.1}%)",
        PRODUCTION_DECODE_MS - total_token_ms,
        (PRODUCTION_DECODE_MS - total_token_ms) / PRODUCTION_DECODE_MS * 100.0
    );

    eprintln!();
    eprintln!("=== Lever ranking (BW-bound = 3-bit quant candidate) ===");
    eprintln!(
        "{:<32} {:>9} {:>10} {:>10}",
        "shape", "BW %", "ms/tok", "category"
    );
    for (shape, _us, _bw_gbs, bw_pct, ms_tok) in &shape_results {
        let category = if *bw_pct >= 75.0 {
            "BW-bound"
        } else if *bw_pct >= 50.0 {
            "BW-leaning"
        } else if *bw_pct >= 25.0 {
            "mixed"
        } else {
            "compute-bound"
        };
        eprintln!(
            "{:<32} {:>8.1}% {:>9.2} {:>10}",
            shape.name, bw_pct, ms_tok, category
        );
    }

    eprintln!();
    eprintln!("=== 3-bit weight quant projection (Affine3, packed × 0.75) ===");
    eprintln!("(method: ms_tok scaled by total_bytes_3bit / total_bytes_4bit at same BW%)");
    let mut total_savings_ms = 0.0;
    for (shape, _us, _bw_gbs, _bw_pct, ms_tok) in &shape_results {
        let packed4 = shape.out * shape.ins / 2;
        let packed3 = shape.out * shape.ins * 3 / 8; // 3-bit packed
        let meta = (shape.out * shape.ins / 64) * 4;
        let act = shape.ins * 2;
        let out_b = shape.out * 2;
        let total4 = packed4 + meta + act + out_b;
        let total3 = packed3 + meta + act + out_b;
        let new_ms_tok = ms_tok * (total3 as f64) / (total4 as f64);
        let saving_ms = ms_tok - new_ms_tok;
        total_savings_ms += saving_ms;
        eprintln!(
            "  {:<32} {:>6.2} ms → {:>6.2} ms (Δ {:+.2} ms, packed share {:.0}%)",
            shape.name,
            ms_tok,
            new_ms_tok,
            -saving_ms,
            packed4 as f64 / total4 as f64 * 100.0
        );
    }
    let projected_decode_ms = PRODUCTION_DECODE_MS - total_savings_ms;
    let projected_tps = 1000.0 / projected_decode_ms;
    eprintln!();
    eprintln!(
        "Total Δ matmul savings: {:.2} ms/token  →  decode {:.2} → {:.2} ms = 15.04 → {:.2} tok/s ({:+.1}%)",
        total_savings_ms,
        PRODUCTION_DECODE_MS,
        projected_decode_ms,
        projected_tps,
        (projected_tps / 15.04 - 1.0) * 100.0
    );
    eprintln!("(NOTE: assumes 3-bit Affine3 kernel achieves same BW% as 4-bit Affine4)");
    eprintln!("(NOTE: non-matmul 18.83 ms/token unchanged — needs separate lever)");
}
