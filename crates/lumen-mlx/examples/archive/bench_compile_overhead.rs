//! task 003 (mlx-rs-compile-persistent-cache).
//!
//! Measures per-call overhead of `mlx_rs::transforms::compile::compile`
//! under two usage patterns:
//!
//!   * **Variant A** — per-call `compile(f, true)((args))?` then drop. This
//!     mirrors the Step A pattern shipped under `LUMEN_NATIVE_COMPILE=1`
//!     in `native_ssm.rs::compute_g`. The hypothesis (Step A NEGATIVE memo)
//!     is that `Compiled<F, G>::Drop` calls `mlx_detail_compile_erase(id)`
//!     each call → wrapper alloc + cache erase = +0.025ms/call observed.
//!
//!   * **Variant B** — keep one `Compiled<F, G>` alive in a local mutable
//!     binding and reuse it across iterations (tuple form). This matches
//!     Python's `@partial(mx.compile, ...)` decorator semantics where the
//!     wrapper is built once at function-definition time.
//!
//! Note on Variant C (`OnceLock<Mutex<Box<dyn FnMut + Send>>>` static cache,
//! the originally-proposed Phase 2 production pattern): NOT achievable with
//! the current mlx-rs API. `Compile::compile<'args>` declares `'args` as a
//! method-level lifetime baked into the returned `impl CallMut<Args<'args>>`;
//! erasing to `Box<dyn FnMut(&[Array]) -> _>` makes args escape the inferred
//! `'args`. To unlock the static-cache path, mlx-rs needs an HRTB-compatible
//! API (see `notes/mlx_rs_compile_hrtb_constraint.md`). This gate measures
//! whether such an upstream change is even worth pursuing.
//!
//! Gate (per task 003 PLAN/CHECKLIST):
//!
//!   * **PASS**  Variant B per-call < 0.005ms AND ≥ 5x faster than Variant A
//!     → mlx-rs PR (HRTB API) becomes worthwhile, or a thread-local /
//!       scoped-cache workaround in lumen-rs is justified
//!   * **FAIL**  → STOP, root-cause is mlx-c side dispatch (not wrapper);
//!                 LAND task 001 at 92.7%
//!
//! Run:
//!
//!   cargo run --release -p lumen-mlx --features mlx-native \
//!     --example bench_compile_overhead
//!
//! Env:
//!
//!   ITERS    timed iterations per variant (default 10000)
//!   WARMUP   warm-up iterations per variant (default 1000)
//!   HV       gating-head dim used for synthetic compute_g shape
//!            (default 4 — small to make wrapper cost dominate)

#[cfg(not(feature = "mlx-native"))]
fn main() {
    eprintln!("bench_compile_overhead requires --features mlx-native");
    std::process::exit(2);
}

#[cfg(feature = "mlx-native")]
fn main() -> anyhow::Result<()> {
    imp::run()
}

#[cfg(feature = "mlx-native")]
mod imp {
    use std::time::Instant;

    use anyhow::Result;
    use mlx_rs::error::Exception;
    use mlx_rs::transforms::compile::{compile, CallMut, Compile};
    use mlx_rs::Array;

    // Mirrors `native_ssm::compute_g_compiled_inner` op-for-op (tuple form,
    // matches Step A signature).
    fn compute_g_inner_tuple(
        args: (&Array, &Array, &Array),
    ) -> std::result::Result<Array, Exception> {
        let (a_log, a, dt_bias) = args;
        let a_log_f32 = a_log.as_dtype(mlx_rs::Dtype::Float32)?;
        let exp_a_log = a_log_f32.exp()?;
        let a_plus_bias = a.add(dt_bias)?;
        let softplus = mlx_rs::nn::softplus(&a_plus_bias)?;
        let prod = exp_a_log.multiply(&softplus)?;
        let neg = prod.negative()?;
        neg.exp()
    }

    fn percentile(sorted_ms: &[f64], p: f64) -> f64 {
        if sorted_ms.is_empty() {
            return 0.0;
        }
        let idx = ((sorted_ms.len() as f64 - 1.0) * p).round() as usize;
        sorted_ms[idx.min(sorted_ms.len() - 1)]
    }

    fn summarize(label: &str, mut times_ns: Vec<u64>, baseline_ms: Option<f64>) -> f64 {
        times_ns.sort_unstable();
        let n = times_ns.len() as f64;
        let mean_ms = (times_ns.iter().copied().sum::<u64>() as f64 / n) / 1_000_000.0;
        let times_ms: Vec<f64> = times_ns.iter().map(|&t| t as f64 / 1_000_000.0).collect();
        let p50 = percentile(&times_ms, 0.50);
        let p95 = percentile(&times_ms, 0.95);
        let p99 = percentile(&times_ms, 0.99);

        let speedup = baseline_ms
            .map(|base| format!(" ({:.2}x vs A)", base / p50))
            .unwrap_or_default();

        println!(
            "  {label:<10} mean={mean_ms:.4}ms p50={p50:.4}ms p95={p95:.4}ms p99={p99:.4}ms{speedup}"
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
        let hv: usize = std::env::var("HV")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);

        println!(
            "bench_compile_overhead: ITERS={iters} WARMUP={warmup} HV={hv}\n\
             compute_g shape: A_log[{hv}] a[1,1,{hv}] dt_bias[{hv}] f32"
        );

        // Build inputs once; both variants share them.
        let a_log = Array::from_slice(&vec![-0.5_f32; hv], &[hv as i32]);
        let a = Array::from_slice(&vec![0.25_f32; hv], &[1, 1, hv as i32]);
        let dt_bias = Array::from_slice(&vec![0.1_f32; hv], &[hv as i32]);

        let eval_each: bool = std::env::var("EVAL_EACH")
            .map(|v| v == "1")
            .unwrap_or(false);
        println!(
            "  eval_each={} ({})",
            eval_each,
            if eval_each {
                "per-call eval — measures wrapper + dispatch + queue sync"
            } else {
                "build-only — measures wrapper + graph build (production-relevant)"
            }
        );

        // ── Variant A: per-call compile() + drop (tuple form) ────────────
        for _ in 0..warmup {
            let mut compiled = compile(compute_g_inner_tuple, true);
            let r = compiled((&a_log, &a, &dt_bias))?;
            if eval_each {
                mlx_rs::transforms::eval([&r])?;
            }
        }
        let mut t_a = Vec::with_capacity(iters);
        let mut keep_a: Vec<Array> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let mut compiled = compile(compute_g_inner_tuple, true);
            let r = compiled((&a_log, &a, &dt_bias))?;
            if eval_each {
                mlx_rs::transforms::eval([&r])?;
            }
            t_a.push(t0.elapsed().as_nanos() as u64);
            if !eval_each {
                keep_a.push(r);
            }
        }
        if !eval_each {
            // Force terminal eval so the lazy graph actually runs and
            // the iters aren't optimized away.
            let last = keep_a.last().unwrap();
            mlx_rs::transforms::eval([last])?;
        }

        // ── Variant B: persistent local Compiled, reused (tuple form) ────
        let mut compiled_b = compute_g_inner_tuple.compile(true);
        for _ in 0..warmup {
            let r = compiled_b.call_mut((&a_log, &a, &dt_bias))?;
            if eval_each {
                mlx_rs::transforms::eval([&r])?;
            }
        }
        let mut t_b = Vec::with_capacity(iters);
        let mut keep_b: Vec<Array> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let r = compiled_b.call_mut((&a_log, &a, &dt_bias))?;
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
        }

        // ── Report ───────────────────────────────────────────────────────
        println!("\nresults (per-call):");
        let p50_a = summarize("A per-call", t_a, None);
        let p50_b = summarize("B cached  ", t_b, Some(p50_a));

        println!("\ngate (Phase 1):");
        let gate_b_lt_5us = p50_b < 0.005;
        let gate_b_5x = p50_a / p50_b >= 5.0;
        println!(
            "  Variant B  per-call <0.005ms: {} ({:.4}ms)  | >=5x vs A: {} ({:.2}x)",
            if gate_b_lt_5us { "PASS" } else { "FAIL" },
            p50_b,
            if gate_b_5x { "PASS" } else { "FAIL" },
            p50_a / p50_b
        );

        let gate_pass = gate_b_lt_5us && gate_b_5x;
        println!(
            "\nPhase 1 gate: {}",
            if gate_pass {
                "PASS — proceed to Phase 2 (compute_g cache adoption); \
                 mlx-rs HRTB-API extension or thread-local workaround needed"
            } else {
                "FAIL — root cause is mlx-c side dispatch; STOP, LAND task 001 at 92.7%"
            }
        );

        Ok(())
    }
}
