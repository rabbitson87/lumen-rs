//! In-process A/B for any env flag that is re-read per call.
//!
//! ## Why this exists
//!
//! A process-per-run A/B of `LUMEN_GEMMA4_CUSTOM_FLASH_ATTN` could not resolve
//! a difference, three times, in three different ways:
//!
//! * blocked ON→OFF said OFF was 20–27% faster;
//! * the reversed order produced a **93.5 s** run against a 1.5–3.5 s norm;
//! * interleaved ABAB on a quiet machine gave 1 win ON, 2 OFF, 1 tie, with a
//!   **2.7× spread inside each side** against a 15% difference between them.
//!
//! Every one of those runs paid for a fresh process: a 26 B model load, cold
//! page cache, fresh Metal pipeline compilation, and whatever else the machine
//! was doing during that particular minute. That variance is charged to the
//! measurement, and it is far larger than any attention-kernel effect.
//!
//! So: load the model **once**, and alternate the flag between `generate`
//! calls. `stats.decode_ms` excludes prefill, adjacent A and B samples sit
//! milliseconds apart in time and share their thermal state, and the model load
//! is paid once instead of 2N times.
//!
//! This works because the flags it targets are read with `std::env::var` at the
//! point of use rather than cached in a `OnceLock` — `gemma4_moe.rs` reads
//! `LUMEN_GEMMA4_CUSTOM_FLASH_ATTN` inside the attention forward, on every
//! call. `prefill_chunk_equivalence.rs` relies on the same property.
//!
//! **Check that before trusting a result here.** A flag behind `OnceLock` (most
//! of the `lumen_flags::flag!` registry) latches on first read, and this
//! harness would then report a beautifully tight "no difference" while
//! measuring one side twice. `lumen_flags` flags have `with()`/`set()` for
//! exactly this; use those instead.
//!
//! ```text
//! MODEL_ID=~/models/mlx-community--gemma-4-26b-a4b-it-4bit \
//!   AB_ENV=LUMEN_GEMMA4_CUSTOM_FLASH_ATTN AB_PAIRS=6 \
//!   PROMPT_LEN=8192 STEPS=32 \
//!   cargo run --release -p lumen-mlx --features mlx-native \
//!   --example bench_env_flag_ab
//! ```

#[cfg(not(feature = "mlx-native"))]
fn main() {
    eprintln!("bench_env_flag_ab requires --features mlx-native");
}

#[cfg(feature = "mlx-native")]
fn main() -> anyhow::Result<()> {
    use anyhow::Context;
    use lumen_mlx::gemma4::{GenerateConfig, NativeGemma4Model};
    use std::path::Path;

    fn env_usize(key: &str, default: usize) -> usize {
        std::env::var(key)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    }

    let model_id = std::env::var("MODEL_ID").context("set MODEL_ID")?;
    let ab_env =
        std::env::var("AB_ENV").unwrap_or_else(|_| "LUMEN_GEMMA4_CUSTOM_FLASH_ATTN".to_string());
    let on_value = std::env::var("AB_ON").unwrap_or_else(|_| "1".to_string());
    let off_value = std::env::var("AB_OFF").unwrap_or_else(|_| "0".to_string());
    let pairs = env_usize("AB_PAIRS", 6);
    let prompt_len = env_usize("PROMPT_LEN", 8192);
    let steps = env_usize("STEPS", 32);
    let warmup = env_usize("WARMUP", 8);

    eprintln!("[ab] loading {model_id}");
    let model = NativeGemma4Model::load(Path::new(&model_id)).context("load")?;
    let vocab = model.vocab_size() as u32;
    let prompt: Vec<u32> = (0..prompt_len)
        .map(|i| 10 + ((i as u32 * 7) % (vocab.saturating_sub(20).max(200))))
        .collect();

    let cfg = |n: usize| GenerateConfig {
        max_new_tokens: n,
        stop_on_eos: false,
        sampling: None,
    };

    eprintln!("[ab] warmup ({warmup} tokens)");
    let _ = model
        .generate(&prompt, &cfg(warmup.max(1)))
        .context("warmup")?;

    // (per-step ms, steps) for each side, in call order.
    let mut on: Vec<(f64, usize)> = Vec::new();
    let mut off: Vec<(f64, usize)> = Vec::new();

    eprintln!("[ab] {ab_env}: {pairs} interleaved pairs, {steps} decode steps each\n");
    for p in 0..pairs {
        for (value, bucket) in [(&on_value, &mut on), (&off_value, &mut off)] {
            // SAFETY: single-threaded here, and the flag under test is read
            // with `env::var` at its point of use, so the change takes effect
            // on the next generate rather than being latched.
            unsafe { std::env::set_var(&ab_env, value) };
            let stats = model.generate(&prompt, &cfg(steps)).context("generate")?;
            let per_step = if stats.decode_steps > 0 {
                stats.decode_ms / stats.decode_steps as f64
            } else {
                f64::NAN
            };
            println!(
                "  pair {p} {ab_env}={value:<3} {:>7.2} ms/step over {} steps",
                per_step, stats.decode_steps
            );
            bucket.push((per_step, stats.decode_steps));
        }
    }

    fn median(v: &mut [f64]) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if v.is_empty() {
            return f64::NAN;
        }
        v[v.len() / 2]
    }

    let mut on_ms: Vec<f64> = on.iter().map(|(m, _)| *m).collect();
    let mut off_ms: Vec<f64> = off.iter().map(|(m, _)| *m).collect();
    let (on_min, off_min) = (
        on_ms.iter().cloned().fold(f64::INFINITY, f64::min),
        off_ms.iter().cloned().fold(f64::INFINITY, f64::min),
    );
    let (on_max, off_max) = (
        on_ms.iter().cloned().fold(0.0, f64::max),
        off_ms.iter().cloned().fold(0.0, f64::max),
    );
    let (on_med, off_med) = (median(&mut on_ms), median(&mut off_ms));

    // Steps must match, or the two sides did different work — the runaway
    // guard can cut a generate short, and that has already invalidated one
    // round of this experiment.
    let steps_differ = on.iter().map(|(_, s)| *s).max() != off.iter().map(|(_, s)| *s).max()
        || on.iter().map(|(_, s)| *s).min() != off.iter().map(|(_, s)| *s).min();

    println!("\n=== {ab_env} ===");
    println!(
        "  {on_value:<3} median {on_med:>7.2} ms/step   min {on_min:>7.2}   max {on_max:>7.2}"
    );
    println!(
        "  {off_value:<3} median {off_med:>7.2} ms/step   min {off_min:>7.2}   max {off_max:>7.2}"
    );

    // The comparison that matters is the difference against the noise, not the
    // difference alone — and the noise estimate has to be robust, or one stall
    // decides the verdict.
    //
    // `max/min` is not robust. The first run of this harness saw a single
    // 1020 ms/step sample against a 22–27 ms cluster (a ~40x stall, and both
    // sides took one), which put "spread" at 4594% and made every result
    // INCONCLUSIVE by construction. A tool that cannot conclude is not
    // measuring anything.
    //
    // So: `min` is the primary estimator, because contention only ever makes a
    // sample **slower** — the fastest run of each side is the one least
    // interfered with. The noise floor is `median/min`, which describes the
    // spread of the clean cluster and ignores the slow tail entirely.
    let diff_med = (on_med - off_med) / off_med * 100.0;
    let diff_min = (on_min - off_min) / off_min * 100.0;
    let spread = ((on_med / on_min).max(off_med / off_min) - 1.0) * 100.0;
    let raw_spread = ((on_max / on_min).max(off_max / off_min) - 1.0) * 100.0;
    let stalls = on_ms
        .iter()
        .chain(off_ms.iter())
        .filter(|m| **m > 3.0 * on_med.min(off_med))
        .count();
    println!("\n  median delta {diff_med:+.1}%   min-vs-min delta {diff_min:+.1}%");
    println!("  noise floor (median/min, robust) {spread:.1}%   raw max/min {raw_spread:.0}%");
    if stalls > 0 {
        println!(
            "  {stalls} sample(s) >3x the median — excluded from the noise floor by\n               construction, but their existence is itself worth knowing"
        );
    }

    if steps_differ {
        println!(
            "\n  INCONCLUSIVE: the two sides ran different step counts, so they did\n  \
             different work. Check for a `[runaway] … aborted` cut."
        );
    } else if spread > diff_min.abs() {
        println!(
            "\n  NO EFFECT ABOVE {spread:.1}%: the min-vs-min difference is {diff_min:+.1}%,\n  \
             inside the noise floor. Whatever the flag does here is smaller than the\n  \
             run-to-run variation of a quiet machine — which is a result, not a\n  \
             failure to measure. Raise AB_PAIRS to tighten the bound."
        );
    } else {
        println!(
            "\n  Signal exceeds the noise floor: {} is faster.",
            if diff_med < 0.0 {
                &on_value
            } else {
                &off_value
            }
        );
    }
    Ok(())
}
