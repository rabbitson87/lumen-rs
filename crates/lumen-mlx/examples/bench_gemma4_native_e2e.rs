//! End-to-end Gemma 4 native bench. Mirrors `bench_mlx_e2e` but routes
//! through our `NativeGemma4Model::generate()` (which uses async-eval
//! pipelined decode) instead of `MlxBackend`. This is the apples-to-apples
//! native counterpart to the PyO3 `bench_mlx_e2e` run for Gemma 4.
//!
//! Env:
//!   MODEL_ID    local directory with config.json + safetensors shards
//!   PROMPT_LEN  synthetic prompt length (default 8)
//!   STEPS       decode steps to time (default 64)
//!   WARMUP      warm-up steps before timing (default 4)
//!
//! Run:
//!   MLX_LOCAL_SOURCE_DIR=/path/to/mlx \
//!   LUMEN_GEMMA4_NO_F32_CAST=1 \
//!   MODEL_ID=/path/to/models/gemma-4-26b-a4b-mlx-4bit \
//!   cargo run --release -p lumen-mlx \
//!       --example bench_gemma4_native_e2e --features mlx-native

use std::path::Path;

use anyhow::{Context, Result};

// lumen-rs Phase 1.6: MLX-side SDPA stage timing dump.
// Implemented in mlx/fast.cpp (extern "C" public symbol so macOS ld
// doesn't dead-strip the internal namespace counters in libmlx.a).
#[cfg(feature = "mlx-native")]
unsafe extern "C" {
    fn mlx_dump_sdpa_timing();
    fn mlx_reset_sdpa_timing();
    // mlx-c wrapper stage timing.
    fn mlxc_dump_sdpa_timing();
    fn mlxc_reset_sdpa_timing();
}

#[cfg(feature = "mlx-native")]
fn main() -> Result<()> {
    use lumen_mlx::gemma4::{GenerateConfig, NativeGemma4Model, take_gemma4_breakdown};

    let model_id = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| "/path/to/models/gemma-4-26b-a4b-mlx-4bit".into());
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

    // Anti-pattern #36 mitigation: log the state of every env gate this
    // bench's measurement depends on. Centralised in `env_state` module so
    // future LANDED gates can be added in one place.
    lumen_mlx::env_state::log_env_state(
        "gemma4-native",
        &[
            "LUMEN_GEMMA4_FUSE_DENSE_MLP",
            "LUMEN_GEMMA4_FUSE_EXPERTS",
            "LUMEN_GEMMA4_FUSE_ROUTER",
            "LUMEN_GEMMA4_CUSTOM_FLASH_ATTN",
            "LUMEN_GEMMA4_PREFILL_SYNC",
        ],
    );

    eprintln!("[gemma4-native] loading {model_id}");
    let model = NativeGemma4Model::load(Path::new(&model_id)).context("NativeGemma4Model::load")?;
    eprintln!("[gemma4-native] loaded");

    let vocab = model.vocab_size() as u32;
    let prompt: Vec<u32> = (0..prompt_len)
        .map(|i| 10 + ((i as u32 * 7) % (vocab.saturating_sub(20).max(200))))
        .collect();

    // Warm-up: run a generate to JIT compile slots + populate caches.
    let warm_cfg = GenerateConfig {
        max_new_tokens: warmup.max(1),
        stop_on_eos: false,
        sampling: None,
    };
    // hide `LUMEN_METAL_CAPTURE` from the warmup generate so
    // the .gputrace bundle only contains the timed run. We re-set after.
    let saved_capture = std::env::var_os("LUMEN_METAL_CAPTURE");
    if saved_capture.is_some() {
        // SAFETY: bench is single-threaded at this point; no concurrent
        // readers of the env block.
        unsafe {
            std::env::remove_var("LUMEN_METAL_CAPTURE");
        }
    }
    let _ = model
        .generate(&prompt, &warm_cfg)
        .context("warmup generate")?;
    if let Some(v) = saved_capture {
        // SAFETY: same — still single-threaded.
        unsafe {
            std::env::set_var("LUMEN_METAL_CAPTURE", v);
        }
    }

    // Reset MLX-side + mlx-c + mlx-rs SDPA stage counters AFTER warmup
    // so the breakdown reflects only the timed run. Counters accumulate
    // across all SDPA calls in the process — without reset, warmup's
    // calls dilute the averages.
    unsafe {
        mlx_reset_sdpa_timing();
        mlxc_reset_sdpa_timing();
    }
    mlx_rs::fast::sdpa_timing::reset();
    lumen_mlx::native_attention::lumen_sdpa_timing::reset();
    // eval_gpu / prim histogram counters — same reset semantic.
    // These track per-primitive `gpu::eval()` encode work inside
    // async_eval / eval. Compares per-step op count + encode wall time
    // between PROMPT_LEN=8 and 4K to localize where the lazy graph
    // density grows in our path vs mlx-lm's.
    let _ = mlx_rs::metal::reset_eval_gpu_stats();
    let _ = mlx_rs::metal::reset_prim_histogram();
    let _ = mlx_rs::metal::reset_prim_histogram_dynamic();
    // CRITICAL: also reset the lumen-side attn/dense/router/experts
    // and sub-stage buckets. Otherwise warmup's prefill work
    // (especially `build_causal_mask` allocating a 4K×4K mask at long
    // prompts) gets attributed to the timed decode steps and inflates
    // attn.sdpa by ~23 ms/step at PROMPT_LEN=4096. `take_*` returns the
    // current values AND resets to zero — we discard the warmup values.
    let _ = take_gemma4_breakdown();

    // per-callsite op-count breakdown to identify the
    // 1.62× ratio source vs mlx-lm Python ops/step. Wire mlx-rs's G6-G
    // instrumentation (`reset_op_counter` + `enable_op_breakdown`) here so
    // the breakdown reflects ONLY the timed decode pass.
    let count_ops = std::env::var("LUMEN_NATIVE_COUNT_OPS")
        .map(|v| v != "0")
        .unwrap_or(false);
    let count_breakdown = std::env::var("LUMEN_NATIVE_COUNT_OPS_BREAKDOWN")
        .map(|v| v != "0")
        .unwrap_or(false);
    if count_ops || count_breakdown {
        mlx_rs::utils::reset_op_counter();
        if count_breakdown {
            mlx_rs::utils::enable_op_breakdown();
        }
    }

    // Timed run: prefill + `steps` decode tokens (no EOS).
    let cfg = GenerateConfig {
        max_new_tokens: steps,
        stop_on_eos: false,
        sampling: None,
    };
    // Marker line for external profilers (e.g. capture_metal_systrace.sh) to
    // sync attach with the start of the timed decode window.
    eprintln!("[timed-decode-start]");
    let stats = model.generate(&prompt, &cfg).context("timed generate")?;

    if count_ops || count_breakdown {
        let total = mlx_rs::utils::read_op_counter();
        let per_step = total as f64 / stats.decode_steps.max(1) as f64;
        println!(
            "  [count-ops] total={total} steps={} per_step={per_step:.1}",
            stats.decode_steps
        );
        if count_breakdown {
            let breakdown = mlx_rs::utils::take_op_breakdown();
            println!("  [count-ops] top 30 callsites:");
            for (site, n) in breakdown.into_iter().take(30) {
                let per_step_site = n as f64 / stats.decode_steps.max(1) as f64;
                println!("    {n:>7} ({per_step_site:>6.1}/step)  {site}");
            }
        }
    }

    println!(
        "prefill: {} tokens in {:.0}ms ({:.1} tok/s)",
        stats.prompt_tokens,
        stats.prefill_ms,
        stats.prompt_tokens as f64 / (stats.prefill_ms / 1000.0)
    );
    let mean = if stats.decode_steps > 0 {
        stats.decode_ms / stats.decode_steps as f64
    } else {
        0.0
    };
    println!(
        "decode: {} steps in {:.0}ms",
        stats.decode_steps, stats.decode_ms
    );
    println!("  step latency: mean={mean:.2}ms");
    println!("  throughput:   {:.1} tok/s", stats.decode_tok_per_sec);

    // Smoke print: dump first 20 generated tokens.
    // Helps detect quality regression — same synthetic prompt should produce
    // very similar token sequences across ON/OFF (quant introduces drift but
    // should not produce out-of-range or pathological IDs).
    if std::env::var("LUMEN_BENCH_PRINT_TOKENS")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        let n = stats.generated_tokens.len().min(20);
        eprintln!(
            "[smoke] first {n} tokens = {:?}",
            &stats.generated_tokens[..n]
        );
    }

    // Per-stage breakdown. Two modes are supported:
    //   - `LUMEN_GEMMA4_BREAKDOWN=1`: legacy mode with eval() barriers
    //     between stages. Each bucket attributes GPU compute correctly but
    //     defeats async pipelined decode → totals inflated 10–17× vs
    //     un-instrumented decode. Read RATIOS, not absolutes.
    //   - `LUMEN_GEMMA4_HONEST_BREAKDOWN=1`: pure Rust-side FFI dispatch
    //     time per stage (no eval barriers). Totals stay close to the
    //     CPU-side share of decode_ms (decode_ms − eval_gpu_us). Reveals
    //     which stage's per-op dispatch cost scales asymmetrically with
    //     context length. Does NOT attribute GPU compute to stages.
    let inflated = std::env::var("LUMEN_GEMMA4_BREAKDOWN")
        .map(|v| v != "0")
        .unwrap_or(false);
    let honest = std::env::var("LUMEN_GEMMA4_HONEST_BREAKDOWN")
        .map(|v| v != "0")
        .unwrap_or(false);
    if inflated || honest {
        // Prefill breakdown — snapshot taken inside generate() right after the
        // prefill GPU drain (only populated when breakdown is enabled).
        if let Some(pb) = stats.prefill_breakdown {
            let label = if honest { "honest" } else { "inflated" };
            println!(
                "  [prefill-breakdown:{label}] total ms (sum 4 buckets) = {:.1}",
                pb.attn_ms + pb.dense_ms + pb.router_ms + pb.experts_ms,
            );
            println!(
                "  [prefill-breakdown:{label}] attn:    {:.1}ms (full={:.1} sliding={:.1})",
                pb.attn_ms, pb.attn_full_ms, pb.attn_sliding_ms,
            );
            println!(
                "  [prefill-breakdown:{label}] attn.qkvo:  {:.1}ms  attn.norms: {:.1}ms  attn.rope: {:.1}ms",
                pb.attn_qkvo_ms, pb.attn_norms_ms, pb.attn_rope_ms,
            );
            println!(
                "  [prefill-breakdown:{label}] attn.cache: {:.1}ms  attn.sdpa:  {:.1}ms  attn.sdpa(tight): {:.1}ms",
                pb.attn_cache_ms, pb.attn_sdpa_ms, pb.attn_sdpa_tight_ms,
            );
            println!(
                "  [prefill-breakdown:{label}] dense:   {:.1}ms  router: {:.1}ms  experts: {:.1}ms",
                pb.dense_ms, pb.router_ms, pb.experts_ms,
            );
        }
        let b = take_gemma4_breakdown();
        let n = stats.decode_steps.max(1) as f64;
        let label = if honest { "honest" } else { "inflated" };
        println!(
            "  [breakdown:{label}] attn:    {:.3}ms/step (full={:.3} sliding={:.3})",
            b.attn_ms / n,
            b.attn_full_ms / n,
            b.attn_sliding_ms / n,
        );
        // Attention sub-stages (honest mode adds a tiny per-substage timer
        // overhead, but it's < 1% of the totals). All 5 sum to ~attn_ms.
        println!(
            "  [breakdown:{label}] attn.qkvo:  {:.3}ms/step  attn.norms: {:.3}ms/step  attn.rope: {:.3}ms/step",
            b.attn_qkvo_ms / n,
            b.attn_norms_ms / n,
            b.attn_rope_ms / n,
        );
        println!(
            "  [breakdown:{label}] attn.cache: {:.3}ms/step  attn.sdpa:  {:.3}ms/step  attn.sdpa(tight): {:.3}ms/step",
            b.attn_cache_ms / n,
            b.attn_sdpa_ms / n,
            b.attn_sdpa_tight_ms / n,
        );
        println!(
            "  [breakdown:{label}] dense:   {:.3}ms/step",
            b.dense_ms / n
        );
        println!(
            "  [breakdown:{label}] router:  {:.3}ms/step",
            b.router_ms / n
        );
        println!(
            "  [breakdown:{label}] experts: {:.3}ms/step",
            b.experts_ms / n
        );
        let sum = (b.attn_ms + b.dense_ms + b.router_ms + b.experts_ms) / n;
        let suffix = if honest {
            "Rust dispatch only; no GPU attribution"
        } else {
            "eval-instrumented; inflated"
        };
        println!("  [breakdown:{label}] sum (4 buckets) = {sum:.3}ms/step ({suffix})");
    }

    // MLX-side SDPA stage breakdown (per-call ns inside the C++ function).
    // Dumps to stderr via the extern "C" entry point in mlx/fast.cpp.
    if std::env::var("LUMEN_SDPA_TIMING_DUMP")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        unsafe {
            mlx_dump_sdpa_timing();
        }
    }

    // mlx-c wrapper stage breakdown (input_extract / sdpa_call / output_wrap).
    // Compared with MLX-side TOTAL, isolates the cost between Rust's
    // `Array::try_from_op` and the C++ entry into mlx::core::SDPA.
    if std::env::var("LUMEN_MLXC_SDPA_TIMING_DUMP")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        unsafe {
            mlxc_dump_sdpa_timing();
        }
    }

    // mlx-rs wrapper stage breakdown (pre_ffi / try_from_op).
    // Compared with mlx-c wrapper TOTAL, isolates the cost in
    // `Array::try_from_op` (Guard alloc + result wrap) vs the
    // pre-FFI mask_mode/mask_arr setup.
    if std::env::var("LUMEN_MLXRS_SDPA_TIMING_DUMP")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        mlx_rs::fast::sdpa_timing::dump();
    }

    // native_attention::sdpa wrapper stage breakdown (pre / call / post).
    // Compared with mlx-rs TOTAL, isolates cost in our local
    // `sdpa()` wrapper (mask enum construction, `.context()` chain).
    if std::env::var("LUMEN_NATIVE_SDPA_TIMING_DUMP")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        lumen_mlx::native_attention::lumen_sdpa_timing::dump();
    }

    // per-primitive gpu::eval encode stats + dynamic histogram.
    // Reveals (a) total #ops encoded in timed run, (b) per-op average ns,
    // and (c) which primitive types are emitted in what proportions. The
    // bench includes prefill + decode in one generate() call; per-step
    // average is contaminated by prefill — read the dynamic histogram for
    // ratios and the calls/ns totals for absolute scale.
    if std::env::var("LUMEN_EVAL_GPU_DUMP")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        if let Ok((calls, ns)) = mlx_rs::metal::eval_gpu_stats() {
            let total_ms = ns as f64 / 1e6;
            let per_call_us = if calls > 0 {
                ns as f64 / 1000.0 / calls as f64
            } else {
                0.0
            };
            let per_step = if stats.decode_steps > 0 {
                calls as f64 / stats.decode_steps as f64
            } else {
                0.0
            };
            eprintln!(
                "[eval-gpu] calls={calls}  ns_total={ns}  ms={:.1}  per_call_us={:.2}  approx_per_step={:.1}",
                total_ms, per_call_us, per_step
            );
        }
        if let Ok(s) = mlx_rs::metal::prim_histogram_dynamic() {
            eprintln!("[prim-hist] dynamic (sorted by count):");
            // Parse "name=count\n..." and sort desc by count.
            let mut rows: Vec<(String, u64)> = s
                .lines()
                .filter_map(|l| {
                    let mut it = l.splitn(2, '=');
                    let name = it.next()?.to_string();
                    let count: u64 = it.next()?.parse().ok()?;
                    Some((name, count))
                })
                .collect();
            rows.sort_by(|a, b| b.1.cmp(&a.1));
            let total: u64 = rows.iter().map(|(_, c)| *c).sum();
            for (name, count) in rows.iter().take(20) {
                let pct = 100.0 * (*count as f64) / (total.max(1) as f64);
                eprintln!("  {:<40} {:>8}  ({:.1}%)", name, count, pct);
            }
            if rows.len() > 20 {
                eprintln!("  ... {} more primitive types", rows.len() - 20);
            }
        }
    }

    Ok(())
}

#[cfg(not(feature = "mlx-native"))]
fn main() -> Result<()> {
    eprintln!("build with --features mlx-native");
    std::process::exit(2);
}
