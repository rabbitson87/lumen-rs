//! Phase 2 Session 1.5 — synthetic-weight Step B (MTP block forward)
//! latency probe. Validates the K=2 vs K=3 cycle math before the loader +
//! runner integration sessions land.
//!
//! Measures `Qwen35MtpBlock::forward` at the production Qwen3.6-35B-A3B-mxfp4
//! shapes (hidden=2048, num_heads=16, num_kv=2, head_dim=256, intermediate=4304,
//! vocab=248320, MXFP4 group_size=32 bits=4). Weights are random + quantized
//! via `mlx_rs::ops::quantize` — the dispatch graph is bit-identical to the
//! production block; only the weight values differ.
//!
//! Cycle math fed by this bench:
//!   K=2 cycle = 14.4 (trunk_decode) + 2*step_B + 19.2 (trunk_verify_S3) + 3
//!   K=3 cycle = 14.4 (trunk_decode) + 3*step_B + 22.6 (trunk_verify_S4) + 3
//! Break-even emit per cycle = cycle / 14.4 (trunk baseline tok/s).
//! K-max emit = K+1 (trunk's sample + accepted drafts).
//!
//! Usage:
//!   cargo run --release -p lumen-mlx --features mlx-native \
//!     --example bench_qwen35_mtp_step_b -- --runs 11

use anyhow::Result;
use lumen_mlx::run_step_b_synthetic_bench;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut runs: usize = 11; // 1 warmup + 10 measured

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--runs" {
            runs = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(runs);
            i += 2;
        } else {
            i += 1;
        }
    }

    println!("--- Phase 2 Step B synthetic-weight microbench ---");
    println!(
        "shapes  = Qwen3.6-35B-A3B-mxfp4 (hidden=2048, heads=16/2, head_dim=256, intermediate=4304)"
    );
    println!("quant   = MXFP4 (group_size=32, bits=4)");
    println!("runs    = {runs} per T (drop first as warmup)");

    // T values to probe:
    //   T=1 — AR draft step (the cost we multiply by K)
    //   T=2 — verify-batch at K=1 (smallest verify; cheap reference)
    //   T=3 — verify-batch at K=2 (matches mlx-native trunk verify for K=2)
    //   T=4 — verify-batch at K=3 (matches mlx-native trunk verify for K=3)
    let t_values: Vec<i32> = vec![1, 2, 3, 4];
    let report = run_step_b_synthetic_bench(&t_values, runs)?;

    println!();
    println!(
        "{:>3}  {:>10}  {:>10}  {:>10}  {:>10}",
        "T", "min_ms", "med_ms", "max_ms", "med/T"
    );
    println!(
        "{:->3}  {:->10}  {:->10}  {:->10}  {:->10}",
        "", "", "", "", ""
    );
    let mut t1_med: Option<f64> = None;
    for p in &report {
        println!(
            "{:>3}  {:>10.2}  {:>10.2}  {:>10.2}  {:>10.2}",
            p.t,
            p.min_ms,
            p.median_ms,
            p.max_ms,
            p.median_ms / p.t as f64
        );
        if p.t == 1 {
            t1_med = Some(p.median_ms);
        }
    }

    println!();
    println!("--- K=2 vs K=3 cycle math (mlx-native trunk baseline 14.4 ms/step) ---");
    if let Some(step_b) = t1_med {
        let trunk_decode = 14.4_f64;
        let trunk_verify_s3 = 19.2_f64;
        let trunk_verify_s4 = 22.55_f64;
        let overhead = 3.0_f64; // snap + restore + accept (gemma4 pattern)

        // Marginal step_B = T=1 cost. We use this per draft step.
        let cycle_k2 = trunk_decode + 2.0 * step_b + trunk_verify_s3 + overhead;
        let cycle_k3 = trunk_decode + 3.0 * step_b + trunk_verify_s4 + overhead;
        let break_even_k2 = cycle_k2 / trunk_decode;
        let break_even_k3 = cycle_k3 / trunk_decode;
        let k2_max_emit = 3.0_f64; // 1 (trunk sample) + 2 (K=2 accepted)
        let k3_max_emit = 4.0_f64; // 1 + 3
        let k2_margin = k2_max_emit - break_even_k2;
        let k3_margin = k3_max_emit - break_even_k3;

        println!("step_B (T=1 measured)     = {step_b:.2} ms");
        println!();
        println!("K=2  cycle = 14.4 + 2*{step_b:.2} + 19.2 + 3.0 = {cycle_k2:.2} ms");
        println!(
            "K=2  break-even emit = cycle/14.4 = {break_even_k2:.2}  (max emit 3 -> margin {k2_margin:+.2})"
        );
        println!();
        println!("K=3  cycle = 14.4 + 3*{step_b:.2} + 22.55 + 3.0 = {cycle_k3:.2} ms");
        println!(
            "K=3  break-even emit = cycle/14.4 = {break_even_k3:.2}  (max emit 4 -> margin {k3_margin:+.2})"
        );
        println!();
        // Estimated tok/s at typical accept rates.
        let accepts = [0.5_f64, 0.65, 0.75, 0.85];
        println!("{:>10}  {:>12}  {:>12}", "accept", "K=2 tok/s", "K=3 tok/s");
        for &a in &accepts {
            // Expected emit = 1 + K*accept (Bernoulli, ignoring tail-stop).
            let emit_k2 = 1.0 + 2.0 * a;
            let emit_k3 = 1.0 + 3.0 * a;
            let tps_k2 = 1000.0 * emit_k2 / cycle_k2;
            let tps_k3 = 1000.0 * emit_k3 / cycle_k3;
            println!("{a:>10.2}  {tps_k2:>12.1}  {tps_k3:>12.1}");
        }
        let baseline_tps = 1000.0 / trunk_decode;
        println!();
        println!("baseline (no MTP) = {baseline_tps:.1} tok/s");
        println!();
        if k2_margin >= 0.5 && k2_margin >= k3_margin {
            println!("VERDICT: K=2 favoured (margin {k2_margin:+.2} >= 0.5 and >= K=3)");
        } else if k3_margin >= 0.5 {
            println!("VERDICT: K=3 favoured (margin {k3_margin:+.2} >= 0.5)");
        } else if k2_margin > 0.0 || k3_margin > 0.0 {
            println!(
                "VERDICT: MARGINAL (best margin > 0 but < 0.5 — needs high accept rate to win)"
            );
        } else {
            println!("VERDICT: NET LOSS at every K (step_B too high; cycle won't close)");
        }
    } else {
        println!("(T=1 not in t_values; cannot compute cycle math)");
    }

    Ok(())
}
