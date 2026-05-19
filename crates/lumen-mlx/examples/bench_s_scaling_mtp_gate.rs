//! Phase 1 MTP feasibility gate — measure the mlx-native S-scaling factor.
//!
//! For MTP to break even on the mlx-native runner the verify-batch forward at
//! S=K+1 must cost roughly the same as a single S=1 decode step. If the per-S
//! cost scales close to linearly (compute-bound), MTP can never close the math.
//!
//! Methodology per S in {1, 2, 3, 4}:
//!   1. Prefill a fixed prompt → seq at position P.
//!   2. Take a deep snapshot of the seq state.
//!   3. Repeatedly: restore_state → forward_probe([dummy; S]) → record ms.
//!   4. Drop the first run as warm-up; report median of the rest.
//!
//! Gate (Phase 2 entry):
//!   T(S=2) / T(S=1) < 1.8  → proceed with mlx-native MTP port
//!   T(S=2) / T(S=1) ≥ 2.5  → abandon (compute-bound at small S)
//!   1.8 ≤ ratio < 2.5       → marginal; revisit after profiling attention path
//!
//! Usage:
//!   MODEL_ID=mlx-community/Qwen3.6-27B-4bit \
//!   LUMEN_QWEN35_SHARDS=/path/to/snapshot \
//!     cargo run --release -p lumen-mlx --example bench_s_scaling_mtp_gate \
//!       --features mlx-native -- --runs 7

use std::time::Instant;

use anyhow::{Result, anyhow};
use lumen_mlx::MlxBackend;

const DEFAULT_PROMPT: &str = "Speculative decoding lets a fast draft model propose multiple \
candidate tokens which the larger target model verifies in parallel. The key throughput lever \
is acceptance rate: when the draft is accurate, the target's single forward emits many tokens \
at once. Multi-Token Prediction (MTP) heads share the trunk's hidden states and are trained \
jointly, which typically lifts acceptance into the 0.6-0.8 range. The remaining question is \
whether the per-cycle latency closes the math.";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_id =
        std::env::var("MODEL_ID").unwrap_or_else(|_| "mlx-community/Qwen3.6-27B-4bit".into());

    let mut runs: usize = 7;
    let mut s_list: Vec<usize> = vec![1, 2, 3, 4];

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--runs" => {
                runs = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(7);
                i += 2;
            }
            "--s-list" => {
                s_list = args[i + 1]
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                i += 2;
            }
            _ => i += 1,
        }
    }

    println!("--- Phase 1 MTP feasibility gate: mlx-native S-scaling ---");
    println!("model = {model_id}");
    println!("runs  = {runs} per S (drop first as warm-up)");
    println!("S     = {s_list:?}");

    let t_load = Instant::now();
    let mut backend = MlxBackend::load(&model_id)?;
    println!("loaded in {:.1}s", t_load.elapsed().as_secs_f64());

    let prompt_ids = backend.encode(DEFAULT_PROMPT)?;
    println!(
        "prompt: {} chars -> {} tokens",
        DEFAULT_PROMPT.len(),
        prompt_ids.len()
    );

    let seq = backend.alloc_seq_id();
    let t_prefill = Instant::now();
    let (last_tok, position) = backend.prefill(seq, &prompt_ids)?;
    println!(
        "prefill: {} tokens in {:.1} ms, position={position}, last_argmax={last_tok}",
        prompt_ids.len(),
        t_prefill.elapsed().as_secs_f64() * 1000.0,
    );

    // Take a reusable deep snapshot so we can rewind to the same cache state
    // between every forward_probe iteration. snapshot_state_deep is the
    // independent (non-aliased) variant — required because each restore
    // consumes one snapshot id; we re-snapshot after every restore.
    let (mut snap_id, _) = backend.snapshot_state_deep(seq)?;
    println!("snapshot id={snap_id}");

    // dummy token sequence — we use last_tok repeated; the argmax content
    // doesn't matter for latency, only the input shape does.
    let dummy_tokens: Vec<u32> = (0..*s_list.iter().max().unwrap_or(&4))
        .map(|_| last_tok)
        .collect();

    println!();
    println!(
        "{:>3}  {:>10}  {:>10}  {:>10}  {:>10}",
        "S", "min_ms", "med_ms", "max_ms", "med/S"
    );
    println!("{:->3}  {:->10}  {:->10}  {:->10}  {:->10}", "", "", "", "", "");

    let mut s1_median: Option<f64> = None;
    let mut s2_median: Option<f64> = None;

    for &s in &s_list {
        let probe = &dummy_tokens[..s];
        let mut times = Vec::with_capacity(runs);
        for _ in 0..runs {
            // Restore consumes the snapshot; capture a fresh one after.
            let _pos = backend.restore_state(seq, snap_id)?;
            let t0 = Instant::now();
            let _row = backend.forward_probe(seq, probe)?;
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            times.push(ms);
            (snap_id, _) = backend.snapshot_state_deep(seq)?;
        }
        // Drop run 0 (warm-up / JIT-compile outlier).
        let warm = &times[1..];
        let mut sorted: Vec<f64> = warm.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = sorted[0];
        let med = sorted[sorted.len() / 2];
        let max = sorted[sorted.len() - 1];
        println!(
            "{:>3}  {:>10.2}  {:>10.2}  {:>10.2}  {:>10.2}",
            s,
            min,
            med,
            max,
            med / s as f64
        );
        if s == 1 {
            s1_median = Some(med);
        }
        if s == 2 {
            s2_median = Some(med);
        }
    }

    println!();
    println!("--- Phase 1 gate ---");
    match (s1_median, s2_median) {
        (Some(t1), Some(t2)) => {
            let ratio = t2 / t1;
            println!(
                "T(S=1) = {t1:.2} ms, T(S=2) = {t2:.2} ms, ratio = {ratio:.2}x"
            );
            if ratio < 1.8 {
                println!("VERDICT: PASS (ratio < 1.8) -- proceed with mlx-native MTP port");
            } else if ratio >= 2.5 {
                println!("VERDICT: FAIL (ratio >= 2.5) -- compute-bound at small S, abandon");
            } else {
                println!("VERDICT: MARGINAL (1.8 <= ratio < 2.5) -- profile attention path first");
            }
        }
        _ => {
            return Err(anyhow!("missing S=1 or S=2 in run; cannot compute gate"));
        }
    }

    backend.release_snapshot(snap_id).ok();
    backend.remove_seq(seq).ok();
    Ok(())
}
