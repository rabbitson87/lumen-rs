//! Stage 2.5 — non-matmul cost breakdown for 27B-4bit decode.
//!
//! Stage 2 measured Σ matmul = 47.58 ms / 67 ms (71.0%); the remaining
//! 19.42 ms (29.0%) is non-matmul (RmsNorm, silu*mul, Flash-attn, Mamba
//! SSM scan, Conv1D, residual, embed/lm_head/sample).
//!
//! This test isolates the kernels we can directly measure and DERIVES the
//! ones we can't (Mamba SSM scan, Conv1D — bound up in GatedDeltaNet's
//! forward). The derived value comes from
//!   lin_non_matmul ≈ stage1_lin_attn_ms - lin_matmul_ms
//!
//! Run:
//!   cargo test --test stage25_non_matmul_breakdown -p lumen-metal \
//!     --features model-integration --release -- --nocapture --test-threads=1

#![cfg(feature = "model-integration")]

use candle_core::{DType, Device, Tensor, backend::BackendDevice as _};
use lumen_metal::flash_attn::flash_attn_candle;
use lumen_metal::rms_norm::RmsNormBf16InBf16Out;
use lumen_metal::silu_mul::SiluMulBf16InBf16Out;
use std::time::Instant;

const HIDDEN: usize = 5120;
const INTER: usize = 17408;

const N_LIN_LAYERS: usize = 48;
const N_FUL_LAYERS: usize = 16;
const N_TOTAL_LAYERS: usize = 64;

const PRODUCTION_DECODE_MS: f64 = 67.0;

const ITERS: usize = 1000;
const WARMUP: usize = 50;

fn pipelined_us<F>(metal_dev: &candle_core::MetalDevice, f: F) -> f64
where
    F: Fn() -> (),
{
    for _ in 0..WARMUP {
        f();
    }
    let _ = metal_dev.synchronize();
    let t0 = Instant::now();
    for _ in 0..ITERS {
        f();
    }
    let _ = metal_dev.synchronize();
    t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64
}

fn synth(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 8) & 0xff) as f32 / 256.0 - 0.5
        })
        .collect()
}

#[test]
fn stage25_non_matmul_breakdown() {
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("[skip] no Metal device");
            return;
        }
    };
    let metal_dev = match &dev {
        Device::Metal(m) => m.clone(),
        _ => unreachable!(),
    };

    eprintln!();
    eprintln!("=== Stage 2.5: non-matmul cost breakdown (27B-4bit, INTER=17408) ===");
    eprintln!("Iterations: {ITERS} per kernel ({WARMUP} warmup, single sync at end)");
    eprintln!("Production decode baseline: {PRODUCTION_DECODE_MS} ms/token = 15.04 tok/s");
    eprintln!();

    let mut total_ms_tok = 0.0;
    let mut rows: Vec<(String, f64, usize, f64)> = Vec::new();

    // ── 1. RmsNorm (bf16-in/bf16-out) at hidden=5120 ──────────────────────
    {
        let kernel = match RmsNormBf16InBf16Out::new(1e-6) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("[skip] RmsNorm init: {e}");
                return;
            }
        };
        let x_data = synth(HIDDEN, 0xAAAA_1111);
        let x = Tensor::from_vec(x_data, &[1, 1, HIDDEN], &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .contiguous()
            .unwrap();
        let w_data = synth(HIDDEN, 0xBBBB_2222);
        let w = Tensor::from_vec(w_data, &[HIDDEN], &dev)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .contiguous()
            .unwrap();

        let us = pipelined_us(&metal_dev, || {
            let _ = kernel.forward(&x, &w).unwrap();
        });
        // 2 RmsNorms per layer (input_layernorm + post_attention_layernorm)
        let calls = 2 * N_TOTAL_LAYERS;
        let ms_tok = us * calls as f64 / 1000.0;
        total_ms_tok += ms_tok;
        rows.push((
            format!("RmsNorm bf16in/bf16out (hidden={HIDDEN})"),
            us,
            calls,
            ms_tok,
        ));
    }

    // ── 2. silu*mul fused (bf16-in/bf16-out) at INTER=17408 ───────────────
    {
        let kernel = match SiluMulBf16InBf16Out::new() {
            Ok(k) => k,
            Err(e) => {
                eprintln!("[skip] silu_mul init: {e}");
                return;
            }
        };
        // gate_up output is [batch, seq, 2*INTER]; silu*mul → [batch, seq, INTER]
        let combined_data = synth(2 * INTER, 0xCCCC_3333);
        let combined = Tensor::from_vec(combined_data, &[1, 1, 2 * INTER], &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .contiguous()
            .unwrap();

        let us = pipelined_us(&metal_dev, || {
            let _ = kernel.forward(&combined).unwrap();
        });
        let calls = N_TOTAL_LAYERS; // 1 per MLP block
        let ms_tok = us * calls as f64 / 1000.0;
        total_ms_tok += ms_tok;
        rows.push((format!("silu*mul fused (INTER={INTER})"), us, calls, ms_tok));
    }

    // ── 3. Flash-attn (full-attn shape: 24Q heads, 4KV heads, head_dim=256) ──
    // For decode at varying KV cache length. Sample at skv=256 (mid-range)
    // and skv=2048 (long context) for sensitivity check.
    for &skv in &[256usize, 2048] {
        let q_data = synth(1 * 24 * 1 * 256, 0xDDDD_4444 ^ skv as u32);
        let q = Tensor::from_vec(q_data, &[1, 24, 1, 256], &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .contiguous()
            .unwrap();
        let k_data = synth(1 * 4 * skv * 256, 0xEEEE_5555 ^ skv as u32);
        let k = Tensor::from_vec(k_data, &[1, 4, skv, 256], &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .contiguous()
            .unwrap();
        let v_data = synth(1 * 4 * skv * 256, 0xFFFF_6666 ^ skv as u32);
        let v = Tensor::from_vec(v_data, &[1, 4, skv, 256], &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .contiguous()
            .unwrap();
        let scale = 1.0 / (256.0_f32).sqrt();

        // Probe.
        let probe = flash_attn_candle(&q, &k, &v, None, scale);
        if probe.is_none() {
            eprintln!("[skip] flash_attn unavailable for ful shape");
            continue;
        }
        let _ = probe.unwrap().unwrap();

        let us = pipelined_us(&metal_dev, || {
            let _ = flash_attn_candle(&q, &k, &v, None, scale).unwrap().unwrap();
        });
        let calls = N_FUL_LAYERS;
        let ms_tok = us * calls as f64 / 1000.0;
        total_ms_tok += ms_tok;
        rows.push((
            format!("flash_attn ful (24Q/4KV/d=256, skv={skv})"),
            us,
            calls,
            ms_tok,
        ));
    }

    // ── 4. Vector residual add (bf16) at hidden=5120 ──────────────────────
    {
        let a_data = synth(HIDDEN, 0xCAFE_BABE);
        let a = Tensor::from_vec(a_data, &[1, 1, HIDDEN], &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .contiguous()
            .unwrap();
        let b_data = synth(HIDDEN, 0xDEAD_BEEF);
        let b = Tensor::from_vec(b_data, &[1, 1, HIDDEN], &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .contiguous()
            .unwrap();

        let us = pipelined_us(&metal_dev, || {
            let _ = (&a + &b).unwrap();
        });
        // 2 residual adds per layer (post-attn + post-MLP)
        let calls = 2 * N_TOTAL_LAYERS;
        let ms_tok = us * calls as f64 / 1000.0;
        total_ms_tok += ms_tok;
        rows.push((
            format!("residual add bf16 (hidden={HIDDEN})"),
            us,
            calls,
            ms_tok,
        ));
    }

    // ── Report ────────────────────────────────────────────────────────────
    eprintln!(
        "{:<46} {:>10} {:>10} {:>10}",
        "kernel", "µs/call", "calls/tok", "ms/tok"
    );
    eprintln!("{}", "─".repeat(82));
    for (name, us, calls, ms_tok) in &rows {
        eprintln!("{:<46} {:>10.2} {:>10} {:>10.3}", name, us, calls, ms_tok);
    }
    eprintln!("{}", "─".repeat(82));
    eprintln!(
        "{:<46} {:>10} {:>10} {:>10.3}",
        "Σ measured non-matmul / token", "—", "—", total_ms_tok
    );

    // Stage 1+2 derivation:
    //   Stage 1 attn-block (lin + ful) = 26.32 ms
    //   Stage 2 lin matmul = 8.41 + 2.14 = 10.55 ms (in_proj + out_proj)
    //   Stage 2 ful matmul = 1.07 + 0.71 = 1.78 ms (qkv + o_proj)
    //   Σ attn matmul = 12.33 ms
    //   → attn non-matmul (Mamba SSM scan + Conv1D + qknorm + RoPE + SDPA + intra-block norm)
    //     = 26.32 - 12.33 = 13.99 ms
    let stage1_attn_ms = 26.32;
    let stage2_attn_matmul_ms = 8.41 + 2.14 + 1.07 + 0.71;
    let attn_non_matmul_ms = stage1_attn_ms - stage2_attn_matmul_ms;
    eprintln!();
    eprintln!("=== Derived (Stage 1 - Stage 2) ===");
    eprintln!("Stage 1 attn-block total (sync-mode ratio × 67ms): {stage1_attn_ms:.2} ms");
    eprintln!(
        "Stage 2 attn matmul (lin + ful in/out + qkv + o_proj): {stage2_attn_matmul_ms:.2} ms"
    );
    eprintln!(
        "→ Derived attn non-matmul (Mamba SSM scan + Conv1D + qknorm + RoPE + SDPA): {attn_non_matmul_ms:.2} ms ({:.1}% of decode)",
        attn_non_matmul_ms / PRODUCTION_DECODE_MS * 100.0
    );
    eprintln!(
        "  Of which flash_attn ful (measured above) accounts for: ~{:.2} ms",
        rows.iter()
            .find(|r| r.0.contains("flash_attn ful"))
            .map(|r| r.3)
            .unwrap_or(0.0)
    );

    // Final budget reconciliation.
    eprintln!();
    eprintln!("=== Final 67 ms decode budget reconciliation ===");
    eprintln!("Σ matmul (Stage 2):                              47.58 ms (71.0%)");
    eprintln!(
        "Σ measured non-matmul (this test):              {:>5.2} ms ({:.1}%)",
        total_ms_tok,
        total_ms_tok / PRODUCTION_DECODE_MS * 100.0
    );
    let stage2_attn_matmul_total = 12.33;
    let derived_unmeasured = PRODUCTION_DECODE_MS - 47.58 - total_ms_tok;
    eprintln!(
        "Remaining (Mamba SSM scan + Conv1D + qknorm + RoPE + sample):  {:>5.2} ms ({:.1}%)",
        derived_unmeasured,
        derived_unmeasured / PRODUCTION_DECODE_MS * 100.0
    );
    eprintln!();
    eprintln!("Note: 'remaining' includes stages NOT measurable in lumen-metal alone:");
    eprintln!("  - Mamba SSM scan (recurrent state update, f32, in lumen-model)");
    eprintln!("  - Conv1D depthwise (bound to GatedDeltaNet)");
    eprintln!("  - qk_norm + RoPE rotary (in self_attn forward)");
    eprintln!("  - lm_head matmul + sample argmax (single-shot per token)");
    let _ = stage2_attn_matmul_total; // silence
}
