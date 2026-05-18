//! Step 2 microbench — isolate per-call overhead of `quantized_matmul_with_mode`.
//!
//! Hypothesis (from `notes/native_pyo3_step1_source_inspection.md`):
//! lumen's wrapper allocates a fresh `mlx_default_gpu_stream_new()` and an
//! `OwnedEmptyArray` biases sentinel on every call. mlx-rs standard binding
//! and Python both reuse a singleton stream and avoid the sentinel handle. We
//! suspect this 2-3 alloc round-trip per call is the residual Native↔PyO3
//! `in_proj 3.12×` ratio source.
//!
//! Variants (single MXFP4 matmul, identical inputs):
//!   * **A — current**: per-call `mlx_default_gpu_stream_new()` +
//!     `mlx_array_new()` sentinel + `mlx_array_new()` output, then drops.
//!     Mirrors `native_quant::quantized_matmul_with_mode` exactly.
//!   * **B — cached stream + cached sentinel**: stream + biases sentinel
//!     allocated once (process lifetime), shared across all iterations. Only
//!     output array allocated per call (unavoidable — receives op result).
//!
//! Decision tree:
//!   * **B p50 ≪ A p50** (≥ 2x faster, sub-µs delta) → H4 confirmed; Step 4
//!     binding patch becomes a landing-eligible lever.
//!   * **B p50 ≈ A p50** (within ±10%) → H4 falsified at microbench level;
//!     gap lives further down (graph build / kernel cache / queue scheduling).
//!     Pivot to Step 3 (eval-barrier coalescing) or Step 5 (stream/cache).
//!
//! Run:
//!   cargo run --release -p lumen-mlx --features mlx-native \
//!     --example bench_quantized_matmul_isolated
//!
//! Env:
//!   ITERS    timed iterations per variant (default 10000)
//!   WARMUP   warm-up iterations per variant (default 1000)
//!   IN       in_features (default 2048; must be multiple of 32 for MXFP4)
//!   OUT      out_features (default 2048; must be multiple of 32)
//!   EVAL_EACH=1  force `mlx_array_eval` per call (matches instrumented
//!                breakdown's eval-barrier semantics; default 0 = lazy)

#[cfg(not(feature = "mlx-native"))]
fn main() {
    eprintln!("bench_quantized_matmul_isolated requires --features mlx-native");
    std::process::exit(2);
}

#[cfg(feature = "mlx-native")]
fn main() -> anyhow::Result<()> {
    imp::run()
}

#[cfg(feature = "mlx-native")]
mod imp {
    use anyhow::{Result, anyhow};
    use mlx_rs::Array;
    use std::ffi::CStr;
    use std::time::Instant;

    const MODE_MXFP4: &CStr = c"mxfp4";
    const GROUP_SIZE: i32 = 32;
    const BITS: i32 = 4;

    fn optional_int(value: i32) -> mlx_sys::mlx_optional_int {
        mlx_sys::mlx_optional_int {
            value,
            has_value: true,
        }
    }

    /// Variant A — exact replica of `native_quant::imp::quantized_matmul_with_mode`.
    /// Per call: 1× stream alloc/free + 1× sentinel alloc/free + 1× output alloc.
    fn matmul_variant_a(x: &Array, w: &Array, scales: &Array) -> Result<Array> {
        unsafe {
            let stream_raw = mlx_sys::mlx_default_gpu_stream_new();
            let sentinel = mlx_sys::mlx_array_new();
            let mut out_raw = mlx_sys::mlx_array_new();
            let status = mlx_sys::mlx_quantized_matmul(
                &mut out_raw,
                x.as_ptr(),
                w.as_ptr(),
                scales.as_ptr(),
                sentinel,
                true,
                optional_int(GROUP_SIZE),
                optional_int(BITS),
                MODE_MXFP4.as_ptr(),
                stream_raw,
            );
            // Drop equivalent
            let _ = mlx_sys::mlx_stream_free(stream_raw);
            let _ = mlx_sys::mlx_array_free(sentinel);
            if status != 0 {
                let _ = mlx_sys::mlx_array_free(out_raw);
                return Err(anyhow!("Variant A: mlx_quantized_matmul status {status}"));
            }
            Ok(Array::from_ptr(out_raw))
        }
    }

    /// Variant B — stream + sentinel cached at process scope, only output allocated per call.
    /// Mirrors mlx-rs standard `quantized_matmul_device` (caller-provided stream)
    /// + Python's `biases=None → std::nullopt` semantics (no per-call sentinel).
    fn matmul_variant_b(
        x: &Array,
        w: &Array,
        scales: &Array,
        cached_stream: mlx_sys::mlx_stream,
        cached_sentinel: mlx_sys::mlx_array,
    ) -> Result<Array> {
        unsafe {
            let mut out_raw = mlx_sys::mlx_array_new();
            let status = mlx_sys::mlx_quantized_matmul(
                &mut out_raw,
                x.as_ptr(),
                w.as_ptr(),
                scales.as_ptr(),
                cached_sentinel,
                true,
                optional_int(GROUP_SIZE),
                optional_int(BITS),
                MODE_MXFP4.as_ptr(),
                cached_stream,
            );
            if status != 0 {
                let _ = mlx_sys::mlx_array_free(out_raw);
                return Err(anyhow!("Variant B: mlx_quantized_matmul status {status}"));
            }
            Ok(Array::from_ptr(out_raw))
        }
    }

    fn percentile(sorted_ms: &[f64], p: f64) -> f64 {
        if sorted_ms.is_empty() {
            return 0.0;
        }
        let idx = ((sorted_ms.len() as f64 - 1.0) * p).round() as usize;
        sorted_ms[idx.min(sorted_ms.len() - 1)]
    }

    fn summarize(label: &str, mut times_ns: Vec<u64>, baseline_us: Option<f64>) -> f64 {
        times_ns.sort_unstable();
        let n = times_ns.len() as f64;
        let mean_us = (times_ns.iter().copied().sum::<u64>() as f64 / n) / 1_000.0;
        let times_us: Vec<f64> = times_ns.iter().map(|&t| t as f64 / 1_000.0).collect();
        let p50 = percentile(&times_us, 0.50);
        let p95 = percentile(&times_us, 0.95);
        let p99 = percentile(&times_us, 0.99);
        let speedup = baseline_us
            .map(|base| format!("  ({:.2}× vs A)", base / p50))
            .unwrap_or_default();
        println!(
            "  {label:<14} mean={mean_us:>7.3}μs  p50={p50:>7.3}μs  p95={p95:>7.3}μs  p99={p99:>7.3}μs{speedup}"
        );
        p50
    }

    pub fn run() -> Result<()> {
        let iters: usize = std::env::var("ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10_000);
        let warmup: usize = std::env::var("WARMUP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1_000);
        let in_features: i32 = std::env::var("IN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2048);
        let out_features: i32 = std::env::var("OUT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2048);
        let eval_each: bool = std::env::var("EVAL_EACH")
            .map(|v| v == "1")
            .unwrap_or(false);

        assert_eq!(
            in_features % 32,
            0,
            "in_features must be multiple of 32 for MXFP4"
        );
        assert_eq!(
            out_features % 32,
            0,
            "out_features must be multiple of 32 for MXFP4"
        );

        println!(
            "bench_quantized_matmul_isolated: ITERS={iters} WARMUP={warmup} IN={in_features} OUT={out_features} EVAL_EACH={}",
            eval_each as u8
        );
        println!(
            "  shape: x[1,{}] (f32) × w[{},{}/8] (u32 packed mxfp4) → y[1,{}] (f32)",
            in_features, out_features, in_features, out_features
        );
        println!(
            "  mode={} group_size={GROUP_SIZE} bits={BITS}",
            "mxfp4"
        );

        // Build inputs (shared across both variants).
        // x: [1, in_features] f32 — single token decode shape
        let x_data: Vec<f32> = (0..in_features as usize)
            .map(|i| 0.001 * (i as f32))
            .collect();
        let x = Array::from_slice(&x_data, &[1, in_features]);

        // Synthetic MXFP4 weight: u32 packed [out, in/8], values arbitrary.
        let w_packed_len = (out_features as usize) * (in_features as usize / 8);
        let w_data: Vec<u32> = vec![0x12345678_u32; w_packed_len];
        let w = Array::from_slice(&w_data, &[out_features, in_features / 8]);

        // E8M0 scales: u8 [out, in/32]. Value 127 → exponent 0 → 2^0 = 1.0.
        let s_len = (out_features as usize) * (in_features as usize / 32);
        let s_data: Vec<u8> = vec![127_u8; s_len];
        let scales = Array::from_slice(&s_data, &[out_features, in_features / 32]);

        // ── Variant A warmup + measure ───────────────────────────────
        for _ in 0..warmup {
            let r = matmul_variant_a(&x, &w, &scales)?;
            if eval_each {
                mlx_rs::transforms::eval([&r])?;
            } else {
                drop(r);
            }
        }
        let mut t_a = Vec::with_capacity(iters);
        let mut keep_a: Vec<Array> = Vec::with_capacity(if eval_each { 0 } else { iters });
        for _ in 0..iters {
            let t0 = Instant::now();
            let r = matmul_variant_a(&x, &w, &scales)?;
            if eval_each {
                mlx_rs::transforms::eval([&r])?;
            }
            t_a.push(t0.elapsed().as_nanos() as u64);
            if !eval_each {
                keep_a.push(r);
            }
        }
        if !eval_each {
            // Force lazy graph completion at the end so the queued ops actually run.
            let last = keep_a.last().unwrap();
            mlx_rs::transforms::eval([last])?;
            keep_a.clear();
        }

        // ── Variant B warmup + measure ───────────────────────────────
        // Allocate stream + sentinel ONCE, hold for the entire run.
        let cached_stream = unsafe { mlx_sys::mlx_default_gpu_stream_new() };
        let cached_sentinel = unsafe { mlx_sys::mlx_array_new() };

        for _ in 0..warmup {
            let r = matmul_variant_b(&x, &w, &scales, cached_stream, cached_sentinel)?;
            if eval_each {
                mlx_rs::transforms::eval([&r])?;
            } else {
                drop(r);
            }
        }
        let mut t_b = Vec::with_capacity(iters);
        let mut keep_b: Vec<Array> = Vec::with_capacity(if eval_each { 0 } else { iters });
        for _ in 0..iters {
            let t0 = Instant::now();
            let r = matmul_variant_b(&x, &w, &scales, cached_stream, cached_sentinel)?;
            if eval_each {
                mlx_rs::transforms::eval([&r])?;
            }
            t_b.push(t0.elapsed().as_nanos() as u64);
            if !eval_each {
                keep_b.push(r);
            }
        }
        if !eval_each {
            let last = keep_b.last().unwrap();
            mlx_rs::transforms::eval([last])?;
            keep_b.clear();
        }

        // Cleanup cached handles.
        unsafe {
            let _ = mlx_sys::mlx_stream_free(cached_stream);
            let _ = mlx_sys::mlx_array_free(cached_sentinel);
        }

        // ── Report ───────────────────────────────────────────────────
        println!("\nresults (per-call wall time):");
        let p50_a = summarize("A current  ", t_a, None);
        let p50_b = summarize("B cached   ", t_b, Some(p50_a));

        let delta_us = p50_a - p50_b;
        let ratio = p50_a / p50_b;
        println!("\nΔ p50 = {delta_us:.3}μs  (A/B ratio = {ratio:.2}×)");

        // Per-step extrapolation for in_proj (4 calls × 30 layers = 120/step)
        let in_proj_calls_per_step = 120.0_f64;
        let in_proj_save_ms = delta_us * in_proj_calls_per_step / 1_000.0;
        println!(
            "  extrapolation: in_proj ({:.0} calls/step) × Δ = {:.3} ms/step potential save",
            in_proj_calls_per_step, in_proj_save_ms
        );

        // Decision gate (Step 2 plan)
        println!("\ndecision gate:");
        let h4_confirmed = ratio >= 2.0 && delta_us >= 0.5;
        let h4_falsified = ratio < 1.10 || delta_us < 0.1;
        if h4_confirmed {
            println!(
                "  H4 CONFIRMED — A/B ratio {ratio:.2}× ≥ 2.0× and Δ {delta_us:.3}μs ≥ 0.5μs"
            );
            println!("  → proceed to Step 4: cache stream + sentinel in NativeQwen3_5MoeModel");
        } else if h4_falsified {
            println!(
                "  H4 FALSIFIED — A/B ratio {ratio:.2}× < 1.10× or Δ {delta_us:.3}μs < 0.1μs"
            );
            println!("  → pivot to Step 3 (eval-barrier coalescing) or Step 5 (stream/cache)");
        } else {
            println!(
                "  H4 INCONCLUSIVE — ratio {ratio:.2}×, Δ {delta_us:.3}μs (between thresholds)"
            );
            println!("  → re-run with EVAL_EACH=1 or larger ITERS, or inspect deeper");
        }

        Ok(())
    }
}
