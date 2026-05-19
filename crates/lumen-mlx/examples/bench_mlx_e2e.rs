//! End-to-end MLX bench. Measures decode throughput so we can compare against
//! raw `mlx_lm.generate` on the same machine. M3 Max 36 GB ceilings, single
//! seq, greedy: ~75 tok/s on 35B mxfp4, ~92 tok/s on 4B mxfp4. (Earlier "91
//! tok/s on 35B" comment was wrong — that figure was actually 4B.)
//!
//! Run:
//!   cargo run --release -p lumen-mlx --example bench_mlx_e2e
//!
//! Env:
//!   MODEL_ID    default mlx-community/Qwen3.6-35B-A3B-mxfp4
//!   PROMPT_LEN  synthetic prompt length (default 8)
//!   STEPS       decode steps to time (default 64)
//!   WARMUP      warm-up steps before timing (default 4)

use std::time::Instant;

use anyhow::{Result, anyhow};
use lumen_mlx::MlxBackend;

fn main() -> Result<()> {
    let model_id =
        std::env::var("MODEL_ID").unwrap_or_else(|_| "mlx-community/Qwen3.6-35B-A3B-mxfp4".into());
    let prompt_len: usize = std::env::var("PROMPT_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let steps: usize = std::env::var("STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let warmup: usize = std::env::var("WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    // Anti-pattern #36 mitigation: log perf-gate env state before timing
    // so it's clear whether the measurement reflects the canonical path or
    // an experimental / legacy configuration.
    lumen_mlx::env_state::log_env_state(
        "mlx-e2e",
        &[
            "LUMEN_MLX_BACKEND",
            "LUMEN_NATIVE_KV_STEP_PREALLOC",
            "LUMEN_NATIVE_ALLOC_REUSE",
            "LUMEN_NATIVE_LINEAR_ATTN_SCALE_FUSE",
            "LUMEN_NATIVE_DEFER_CLEAR_CACHE",
            "LUMEN_NATIVE_FUSE_SWIGLU",
            "LUMEN_NATIVE_COMPILE",
            "LUMEN_NATIVE_CACHED_STREAM",
            "LUMEN_NATIVE_STREAM_INCR",
            "LUMEN_QWEN35_ROPE_PRECOMPUTE_FREQS",
        ],
    );

    let mut b = MlxBackend::load(&model_id)?;
    let b = b
        .as_qwen35_mut()
        .ok_or_else(|| anyhow!("bench requires Qwen35-family backend"))?;

    // Synthetic prompt: vocab-safe range.
    let prompt: Vec<u32> = (0..prompt_len).map(|i| 10 + (i as u32 * 7) % 200).collect();

    let t_prefill = Instant::now();
    let (mut last, mut pos) = b.prefill(1, &prompt)?;
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    println!(
        "prefill: {} tokens in {prefill_ms:.0}ms ({:.1} tok/s)",
        prompt.len(),
        prompt.len() as f64 / (prefill_ms / 1000.0)
    );

    // Warmup.
    for _ in 0..warmup {
        let (n, p) = b.decode_step(1, last, pos)?;
        last = n;
        pos = p;
    }

    // Timed decode.
    let mut step_ms: Vec<f64> = Vec::with_capacity(steps);
    for _ in 0..steps {
        let t0 = Instant::now();
        let (n, p) = b.decode_step(1, last, pos)?;
        last = n;
        pos = p;
        step_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    b.remove_seq(1)?;

    // Optional: dump per-step latencies in positional order before we sort.
    if std::env::var("LUMEN_DUMP_STEP_MS")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        for (i, ms) in step_ms.iter().enumerate() {
            println!("[step-ms] {i} {ms:.4}");
        }
    }

    step_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let total: f64 = step_ms.iter().sum();
    let mean = total / steps as f64;
    let p50 = step_ms[steps / 2];
    let p95 = step_ms[(steps as f64 * 0.95) as usize];
    let tps = steps as f64 / (total / 1000.0);

    println!("decode: {steps} steps in {total:.0}ms");
    println!("  step latency: mean={mean:.2}ms p50={p50:.2}ms p95={p95:.2}ms");
    println!("  throughput:   {tps:.1} tok/s");

    // Native runner forward-vs-tail breakdown when LUMEN_NATIVE_TIMING=1.
    if let Some(log) = b.take_native_decode_timing_log() {
        if log.len() >= steps {
            // Skip warmup entries; keep the last `steps` so the breakdown
            // aligns with the timed decode loop above.
            let timed = &log[log.len() - steps..];
            let mut fwd: Vec<f64> = timed.iter().map(|(f, _)| *f).collect();
            let mut tail: Vec<f64> = timed.iter().map(|(_, t)| *t).collect();
            fwd.sort_by(|a, b| a.partial_cmp(b).unwrap());
            tail.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let fwd_total: f64 = fwd.iter().sum();
            let tail_total: f64 = tail.iter().sum();
            let fwd_mean = fwd_total / steps as f64;
            let tail_mean = tail_total / steps as f64;
            let fwd_p50 = fwd[steps / 2];
            let tail_p50 = tail[steps / 2];
            let fwd_p95 = fwd[(steps as f64 * 0.95) as usize];
            let tail_p95 = tail[(steps as f64 * 0.95) as usize];
            let split_total = fwd_mean + tail_mean;
            let fwd_pct = 100.0 * fwd_mean / split_total;
            let tail_pct = 100.0 * tail_mean / split_total;
            println!(
                "  [breakdown] forward: mean={fwd_mean:.2}ms p50={fwd_p50:.2}ms p95={fwd_p95:.2}ms ({fwd_pct:.1}%)"
            );
            println!(
                "  [breakdown] tail:    mean={tail_mean:.2}ms p50={tail_p50:.2}ms p95={tail_p95:.2}ms ({tail_pct:.1}%)"
            );
            println!(
                "  [breakdown] sum (forward+tail) = {split_total:.2}ms vs measured mean={mean:.2}ms"
            );
        } else {
            eprintln!(
                "[breakdown] timing log has {} entries but {steps} steps requested; skipping",
                log.len()
            );
        }
    }

    // PyO3 decode_step stage breakdown when LUMEN_PYO3_DECODE_STAGE_TIMING=1.
    {
        let pyo3_stage = b.take_pyo3_decode_stage_timings().unwrap_or_default();
        if !pyo3_stage.is_empty() && pyo3_stage.len() >= steps {
            // Drop warmup entries to align with the timed decode loop.
            let timed = &pyo3_stage[pyo3_stage.len() - steps..];
            let n = timed.len() as f64;
            let sum_ns = |f: fn(&(u64, u64, u64, u64)) -> u64| -> f64 {
                timed.iter().map(f).sum::<u64>() as f64
            };
            let arr_us = sum_ns(|t| t.0) / 1000.0 / n;
            let fwd_us = sum_ns(|t| t.1) / 1000.0 / n;
            let sync_us = sum_ns(|t| t.2) / 1000.0 / n;
            let tail_us = sum_ns(|t| t.3) / 1000.0 / n;
            let total_us = arr_us + fwd_us + sync_us + tail_us;
            let pct = |v: f64| 100.0 * v / total_us;
            println!(
                "  [pyo3-stage] arr:     {arr_us:>8.2}us ({:>5.1}%)",
                pct(arr_us)
            );
            println!(
                "  [pyo3-stage] forward: {fwd_us:>8.2}us ({:>5.1}%)  (lazy graph build)",
                pct(fwd_us)
            );
            println!(
                "  [pyo3-stage] sync:    {sync_us:>8.2}us ({:>5.1}%)  (argmax + .item() forces GPU)",
                pct(sync_us)
            );
            println!(
                "  [pyo3-stage] tail:    {tail_us:>8.2}us ({:>5.1}%)  (kv quant + state)",
                pct(tail_us)
            );
            println!(
                "  [pyo3-stage] sum = {:.3}ms (measured mean = {mean:.3}ms)",
                total_us / 1000.0
            );
        }
    }

    // Layer-kind fine breakdown (embed / full_attn / linear_attn / moe / lm_head).
    if let Some(fine_log) = b.take_native_decode_fine_timing_log() {
        if fine_log.len() >= steps {
            let timed = &fine_log[fine_log.len() - steps..];
            let n = timed.len() as f64;
            let sum_f64 = |f: fn(&lumen_mlx::FineTimings) -> f64| timed.iter().map(f).sum::<f64>();
            let embed = sum_f64(|t| t.embed_ms) / n;
            let full_attn = sum_f64(|t| t.full_attn_ms) / n;
            let linear_attn = sum_f64(|t| t.linear_attn_ms) / n;
            let moe = sum_f64(|t| t.moe_ms) / n;
            let lm_head = sum_f64(|t| t.lm_head_ms) / n;
            let total = embed + full_attn + linear_attn + moe + lm_head;
            let pct = |v: f64| 100.0 * v / total;
            // Per-layer means (average cost of one layer of each kind).
            let last = &timed[timed.len() - 1];
            let per_full = if last.full_attn_count > 0 {
                full_attn / last.full_attn_count as f64
            } else {
                0.0
            };
            let per_lin = if last.linear_attn_count > 0 {
                linear_attn / last.linear_attn_count as f64
            } else {
                0.0
            };
            let per_moe = if last.moe_count > 0 {
                moe / last.moe_count as f64
            } else {
                0.0
            };
            println!(
                "  [fine] embed:        mean={embed:.3}ms ({:.1}%)",
                pct(embed)
            );
            println!(
                "  [fine] full_attn:    mean={full_attn:.3}ms ({:.1}%) over {} layers (per-layer={per_full:.3}ms)",
                pct(full_attn),
                last.full_attn_count
            );
            println!(
                "  [fine] linear_attn:  mean={linear_attn:.3}ms ({:.1}%) over {} layers (per-layer={per_lin:.3}ms)",
                pct(linear_attn),
                last.linear_attn_count
            );
            println!(
                "  [fine] moe:          mean={moe:.3}ms ({:.1}%) over {} layers (per-layer={per_moe:.3}ms)",
                pct(moe),
                last.moe_count
            );
            println!(
                "  [fine] lm_head:      mean={lm_head:.3}ms ({:.1}%)",
                pct(lm_head)
            );
            println!("  [fine] sum (5 buckets) = {total:.2}ms");

            // ── linear_attn sub-buckets (per-step mean of per-decode totals) ──
            let lin_in_proj = sum_f64(|t| t.linear_in_proj_ms) / n;
            let lin_conv = sum_f64(|t| t.linear_conv_ms) / n;
            let lin_norm_qk = sum_f64(|t| t.linear_norm_qk_ms) / n;
            let lin_ssm = sum_f64(|t| t.linear_ssm_ms) / n;
            let lin_out = sum_f64(|t| t.linear_out_proj_ms) / n;
            let lin_sub_total = lin_in_proj + lin_conv + lin_norm_qk + lin_ssm + lin_out;
            let lin_pct = |v: f64| 100.0 * v / lin_sub_total.max(1e-9);
            let n_lin = last.linear_attn_count.max(1) as f64;
            println!(
                "  [linear_attn sub] in_proj:    mean={lin_in_proj:.3}ms ({:.1}%) per-layer={:.3}ms",
                lin_pct(lin_in_proj),
                lin_in_proj / n_lin
            );
            println!(
                "  [linear_attn sub] conv:       mean={lin_conv:.3}ms ({:.1}%) per-layer={:.3}ms",
                lin_pct(lin_conv),
                lin_conv / n_lin
            );
            println!(
                "  [linear_attn sub] norm_qk:    mean={lin_norm_qk:.3}ms ({:.1}%) per-layer={:.3}ms",
                lin_pct(lin_norm_qk),
                lin_norm_qk / n_lin
            );
            println!(
                "  [linear_attn sub] ssm:        mean={lin_ssm:.3}ms ({:.1}%) per-layer={:.3}ms",
                lin_pct(lin_ssm),
                lin_ssm / n_lin
            );
            println!(
                "  [linear_attn sub] out_proj:   mean={lin_out:.3}ms ({:.1}%) per-layer={:.3}ms",
                lin_pct(lin_out),
                lin_out / n_lin
            );
            println!(
                "  [linear_attn sub] sum = {lin_sub_total:.2}ms vs outer linear_attn={linear_attn:.2}ms"
            );

            // ── MoE sub-buckets ──
            let moe_routing = sum_f64(|t| t.moe_routing_ms) / n;
            let moe_glu = sum_f64(|t| t.moe_switch_glu_ms) / n;
            let moe_shared = sum_f64(|t| t.moe_shared_expert_ms) / n;
            let moe_sub_total = moe_routing + moe_glu + moe_shared;
            let moe_pct = |v: f64| 100.0 * v / moe_sub_total.max(1e-9);
            let n_moe = last.moe_count.max(1) as f64;
            println!(
                "  [moe sub] routing:      mean={moe_routing:.3}ms ({:.1}%) per-layer={:.3}ms",
                moe_pct(moe_routing),
                moe_routing / n_moe
            );
            println!(
                "  [moe sub] switch_glu:   mean={moe_glu:.3}ms ({:.1}%) per-layer={:.3}ms",
                moe_pct(moe_glu),
                moe_glu / n_moe
            );
            println!(
                "  [moe sub] shared_exp:   mean={moe_shared:.3}ms ({:.1}%) per-layer={:.3}ms",
                moe_pct(moe_shared),
                moe_shared / n_moe
            );
            println!("  [moe sub] sum = {moe_sub_total:.2}ms vs outer moe={moe:.2}ms");
        } else {
            eprintln!(
                "[fine] fine_timing_log has {} entries but {steps} steps requested; skipping",
                fine_log.len()
            );
        }
    }

    Ok(())
}
