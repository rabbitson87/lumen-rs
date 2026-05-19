//! Track A1.0 — Prefix caching baseline measurement.
//!
//! Validates that A1 (master-snapshot prefix cache) is worth implementing by
//! measuring how much cheaper `restore_state + extend(suffix)` is compared to
//! a cold `prefill(prefix + suffix)`. The "restore + extend(S)" path is what
//! A1 will pay on a cache hit; the cold prefill path is what we pay today.
//!
//! Protocol per (P, S):
//!   1. Build prefix of P tokens (synthetic) and suffix of S tokens.
//!   2. Path A (cold) — prefill(prefix ++ suffix) on fresh seq, time TTFT.
//!   3. Path B (cached, simulated) —
//!         pre-prefill(prefix), snapshot, ... [now this is the "master" state]
//!         restore + extend(suffix), time both.
//!      The pre-prefill cost is amortized; only restore + extend(S) count.
//!   4. T_savings = T_A - (T_restore + T_extend_S).
//!
//! Notes:
//!   - The current `snapshot/restore` API uses ref-swap, so this benchmark
//!     only measures what A1 will pay if `restore_to_new_seq` is implemented
//!     with the same primitive cost (ArraysCache: list copy; KVCache: trim).
//!     A1.1 will need a deep-copy variant whose cost is bounded above by a
//!     full single forward pass.
//!   - Run with USE_MLX=1. Default model Qwen2.5-0.5B for fast iteration; set
//!     MODEL_ID=mlx-community/Qwen3.6-35B-A3B-mxfp4 for the production model.
//!
//! Usage:
//!   USE_MLX=1 cargo run --release -p lumen-mlx --example bench_prefix_cache_baseline
//!   USE_MLX=1 MODEL_ID=mlx-community/Qwen3.6-35B-A3B-mxfp4 \
//!     cargo run --release -p lumen-mlx --example bench_prefix_cache_baseline

use std::time::Instant;

use anyhow::{Result, anyhow};
use lumen_mlx::{MlxBackend, MlxQwen35Backend};

/// Synthesizes a long, deterministic prompt by repeating a sentence. Returns
/// approximately `target_tokens` tokens worth of text (slightly more in
/// practice; we then truncate to the exact count via tokenizer).
fn synth_prompt(target_tokens: usize) -> String {
    // ~10-12 tokens per repetition for tokenizers in this family.
    let unit = "The system prompt is long and detailed, listing rules and constraints. ";
    let n_repeats = (target_tokens / 10) + 8;
    unit.repeat(n_repeats)
}

fn measure_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[derive(Debug, Clone)]
struct Stats {
    p_tokens: usize,
    s_tokens: usize,
    cold_ms: f64,
    restore_ms: f64,
    extend_ms: f64,
    next_pred_a: u32,
    next_pred_b: u32,
}

impl Stats {
    fn cached_total_ms(&self) -> f64 {
        self.restore_ms + self.extend_ms
    }
    fn savings_ms(&self) -> f64 {
        self.cold_ms - self.cached_total_ms()
    }
    fn savings_pct(&self) -> f64 {
        100.0 * self.savings_ms() / self.cold_ms
    }
    fn match_str(&self) -> &'static str {
        if self.next_pred_a == self.next_pred_b {
            "✓"
        } else {
            "✗"
        }
    }
}

fn run_one(backend: &mut MlxQwen35Backend, p_tokens: usize, s_tokens: usize) -> Result<Stats> {
    let prefix_text = synth_prompt(p_tokens);
    let suffix_text = synth_prompt(s_tokens);

    let prefix_ids_full = backend.encode(&prefix_text)?;
    let suffix_ids_full = backend.encode(&suffix_text)?;

    let prefix: Vec<u32> = prefix_ids_full.into_iter().take(p_tokens).collect();
    let suffix: Vec<u32> = suffix_ids_full.into_iter().take(s_tokens).collect();

    if prefix.len() != p_tokens || suffix.len() != s_tokens {
        return Err(anyhow!(
            "tokenizer didn't produce enough tokens: P={} S={} (got {} / {})",
            p_tokens,
            s_tokens,
            prefix.len(),
            suffix.len()
        ));
    }

    let mut combined = prefix.clone();
    combined.extend_from_slice(&suffix);

    // ── Path A: cold prefill ──
    let seq_a = backend.alloc_seq_id();
    let t = Instant::now();
    let (next_pred_a, _pos_a) = backend.prefill(seq_a, &combined)?;
    let cold_ms = measure_ms(t);
    backend.remove_seq(seq_a).ok();

    // ── Path B: pre-prefill prefix, snapshot, restore, extend(suffix) ──
    let seq_b = backend.alloc_seq_id();
    let _ = backend.prefill(seq_b, &prefix)?;
    let snap = backend.snapshot_state(seq_b)?;

    // Decode 1 step on the prefix to mutate state, so restore has work to do.
    // (Without this, restore is a no-op and the measurement is unrealistically
    // cheap.) Then restore.
    let _ = backend.decode_step(seq_b, *prefix.last().unwrap(), prefix.len())?;

    let t = Instant::now();
    let _ = backend.restore_state(seq_b, snap)?;
    let restore_ms = measure_ms(t);

    let t = Instant::now();
    let (next_pred_b, _pos_b) = backend
        .runner_extend(seq_b, &suffix)
        .map_err(|e| anyhow!("extend failed: {e}"))?;
    let extend_ms = measure_ms(t);

    backend.release_snapshot(snap).ok();
    backend.remove_seq(seq_b).ok();

    Ok(Stats {
        p_tokens,
        s_tokens,
        cold_ms,
        restore_ms,
        extend_ms,
        next_pred_a,
        next_pred_b,
    })
}

fn main() -> Result<()> {
    let model_id = std::env::var("MODEL_ID").unwrap_or_else(|_| "Qwen/Qwen2.5-0.5B".into());

    println!("--- Track A1.0 prefix-cache baseline ---");
    println!("model = {model_id}");

    let t0 = Instant::now();
    let mut backend = MlxBackend::load(&model_id)?;
    println!("loaded in {:.0}ms", t0.elapsed().as_secs_f64() * 1000.0);
    let backend = backend
        .as_qwen35_mut()
        .ok_or_else(|| anyhow!("bench requires Qwen35-family backend"))?;

    // Warmup: small prefill to settle MPS cache + kernel JIT.
    let warm = backend.encode("warm up the engine please")?;
    let warm_seq = backend.alloc_seq_id();
    let _ = backend.prefill(warm_seq, &warm)?;
    backend.remove_seq(warm_seq).ok();

    // Sweep: various (P, S) combos. Suffix is small (typical user query).
    let combos: Vec<(usize, usize)> = vec![
        (256, 32),
        (1024, 32),
        (2048, 32),
        // 4096 is heavy on 35B; uncomment if model supports it.
        // (4096, 32),
    ];

    let runs_per: usize = 3;

    println!(
        "\n{:>6} {:>4} | {:>9} | {:>9} {:>9} {:>10} | {:>9} {:>7} | match",
        "P", "S", "cold(ms)", "restore", "extend(S)", "cached(ms)", "saved(ms)", "save%"
    );
    println!("{}", "-".repeat(96));

    for &(p, s) in &combos {
        let mut runs: Vec<Stats> = Vec::with_capacity(runs_per);
        for _ in 0..runs_per {
            match run_one(&mut *backend, p, s) {
                Ok(st) => runs.push(st),
                Err(e) => {
                    eprintln!("[error] P={p} S={s}: {e}");
                }
            }
        }
        if runs.is_empty() {
            continue;
        }
        // Take median by cold_ms (typical bench convention).
        runs.sort_by(|a, b| a.cold_ms.partial_cmp(&b.cold_ms).unwrap());
        let m = &runs[runs.len() / 2];

        println!(
            "{:>6} {:>4} | {:>9.1} | {:>9.1} {:>9.1} {:>10.1} | {:>9.1} {:>6.1}% | {}",
            m.p_tokens,
            m.s_tokens,
            m.cold_ms,
            m.restore_ms,
            m.extend_ms,
            m.cached_total_ms(),
            m.savings_ms(),
            m.savings_pct(),
            m.match_str(),
        );
    }

    println!(
        "\nNote: 'match' compares predicted next token from cold path vs cached path.\n\
         A1.1 must preserve this when restore_to_new_seq is added."
    );

    Ok(())
}
