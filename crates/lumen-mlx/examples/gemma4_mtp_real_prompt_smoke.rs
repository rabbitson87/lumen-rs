//! real-prompt MTP end-to-end with bit-identical-vs-OFF
//! verification + acceptance-rate histogram.
//!
//! Uses the chat-templated token sequence for "한국의 수도는?" (= "What's
//! the capital of Korea?") — the same hard-coded ids that
//! `dump_gemma4_hidden.rs` / `scripts/dump_mlx_gemma4_hidden.py` produce.
//! Bypasses the need for a HuggingFace tokenizer roundtrip at this layer.
//!
//! At temperature=0 (greedy), MTP guarantees byte-identical output vs the
//! standard decode path on real text — assuming cache rollback is correct.
//! The Phase 5a synthetic-prompt smoke diverged at idx 5 (108↔184 attractor
//! state). Real-prompt acceptance-rate should be >0% (lilting.ch reported
//! ~+13% on M1 Max), keeping the cache trajectory aligned with OFF.
//!
//! Run:
//!   MLX_LOCAL_SOURCE_DIR=/path/to/Documents/GitHub/mlx \
//!   MODEL_ID=/path/to/models/gemma-4-26b-a4b-mlx-4bit \
//!   DRAFTER_DIR=/path/to/models/gemma-4-26B-A4B-it-assistant-bf16 \
//!   cargo run --release -p lumen-mlx \
//!       --example gemma4_mtp_real_prompt_smoke --features mlx-native

use std::path::Path;

use anyhow::{Context, Result};

#[cfg(feature = "mlx-native")]
fn main() -> Result<()> {
    use lumen_mlx::gemma4::{GenerateConfig, NativeGemma4Model};

    // Chat-templated token sequence: apply_chat_template(
    //     [{"role":"user","content":"한국의 수도는?"}],
    //     add_generation_prompt=True,
    // ) → 18 ids on the gemma-4-26b-a4b mlx-4bit tokenizer.
    const FIXED_PROMPT: &[u32] = &[
        2, 105, 2364, 107, 114216, 237281, 79301, 237170, 236881, 106, 107, 105, 4368, 107, 100,
        45518, 107, 101,
    ];

    let model_id = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| "/path/to/models/gemma-4-26b-a4b-mlx-4bit".into());
    let drafter_dir = std::env::var("DRAFTER_DIR").unwrap_or_else(|_| {
        "/path/to/models/gemma-4-26B-A4B-it-assistant-bf16".into()
    });
    let max_tokens: usize = std::env::var("MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    eprintln!("[mtp-real-smoke] loading trunk {model_id}");
    let mut model = NativeGemma4Model::load(Path::new(&model_id)).context("trunk load")?;
    let skip_drafter = std::env::var("SKIP_DRAFTER").ok().as_deref() == Some("1");
    if skip_drafter {
        eprintln!("[mtp-real-smoke] SKIP_DRAFTER=1 — drafter NOT loaded (OFF-only baseline)");
    } else {
        eprintln!("[mtp-real-smoke] enabling MTP from {drafter_dir}");
        model
            .try_enable_mtp(Path::new(&drafter_dir))
            .context("try_enable_mtp")?;
    }

    let cfg = GenerateConfig {
        max_new_tokens: max_tokens,
        stop_on_eos: false,
    };

    // Baseline (MTP-off). Run once as warmup, then time the second pass so the
    // result reflects steady-state decode, not 1st-call compile/codegen cost.
    unsafe {
        std::env::set_var("LUMEN_GEMMA4_MTP", "0");
    }
    eprintln!(
        "[mtp-real-smoke] OFF warmup pass — prompt_len={}, max_tokens={}",
        FIXED_PROMPT.len(),
        max_tokens
    );
    let _warm = model
        .generate(FIXED_PROMPT, &cfg)
        .context("generate(OFF warmup)")?;
    eprintln!("[mtp-real-smoke] OFF timed pass (warm)");
    let t0 = std::time::Instant::now();
    let stats_off = model
        .generate(FIXED_PROMPT, &cfg)
        .context("generate(MTP off)")?;
    let off_wall = t0.elapsed().as_secs_f64() * 1e3;

    // MTP on. Same warmup-then-time protocol so OFF vs ON are both steady-state.
    unsafe {
        std::env::set_var("LUMEN_GEMMA4_MTP", "1");
    }
    eprintln!("[mtp-real-smoke] ON warmup pass");
    let _warm_on = model
        .generate(FIXED_PROMPT, &cfg)
        .context("generate(ON warmup)")?;
    eprintln!("[mtp-real-smoke] ON timed pass (warm)");
    let t0 = std::time::Instant::now();
    let stats_on = model
        .generate(FIXED_PROMPT, &cfg)
        .context("generate(MTP on)")?;
    let on_wall = t0.elapsed().as_secs_f64() * 1e3;

    println!("\n=== MTP real-prompt smoke ===");
    println!(
        "  prompt_len={} max_tokens={max_tokens}",
        FIXED_PROMPT.len(),
    );
    println!(
        "  OFF: {} tokens in {:.0} ms (decode {:.1} tok/s, {} steps)",
        stats_off.generated_tokens.len(),
        off_wall,
        stats_off.decode_tok_per_sec,
        stats_off.decode_steps
    );
    println!(
        "  ON : {} tokens in {:.0} ms (decode {:.1} tok/s, {} mtp_steps)",
        stats_on.generated_tokens.len(),
        on_wall,
        stats_on.decode_tok_per_sec,
        stats_on.decode_steps
    );
    let speedup = off_wall / on_wall;
    println!("  wall-clock speedup: {speedup:.2}x");

    // Bit-identical greedy check.
    let n_compare = stats_off
        .generated_tokens
        .len()
        .min(stats_on.generated_tokens.len());
    let mut match_len = 0;
    for i in 0..n_compare {
        if stats_off.generated_tokens[i] == stats_on.generated_tokens[i] {
            match_len += 1;
        } else {
            break;
        }
    }
    println!(
        "\n  bit-identical prefix: {match_len}/{n_compare} tokens",
    );
    println!(
        "  OFF first 32: {:?}",
        &stats_off.generated_tokens[..stats_off.generated_tokens.len().min(32)]
    );
    println!(
        "  ON  first 32: {:?}",
        &stats_on.generated_tokens[..stats_on.generated_tokens.len().min(32)]
    );

    // Acceptance-rate inference: ON has `mtp_steps` mtp_step calls. Each call
    // commits (1 + n_accepted + 1) tokens by contract (next_token + accepted
    // drafts + correction/bonus). So total committed tokens =
    // sum over steps of (2 + n_accepted_k) = 2 * S + sum(n_accepted_k),
    // where S = mtp_steps. Solving for the mean: mean_n_accepted =
    // (total_committed - 2 * S) / S.
    //
    // The smoke's `decode_steps` field in stats_on is the mtp_steps count;
    // `generated_tokens.len()` is the total committed.
    let s = stats_on.decode_steps as f64;
    let total = stats_on.generated_tokens.len() as f64;
    let mean_n_accepted = if s > 0.0 { (total - 2.0 * s) / s } else { 0.0 };
    let n_draft_env: f64 = std::env::var("LUMEN_GEMMA4_MTP_BLOCK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6.0);
    let accept_pct = if n_draft_env > 0.0 {
        mean_n_accepted / n_draft_env * 100.0
    } else {
        0.0
    };
    println!(
        "\n  mtp_steps={s:.0}, total_committed={total:.0}",
    );
    println!(
        "  inferred mean n_accepted/step = {mean_n_accepted:.2} of n_draft={n_draft_env:.0} \
         (acceptance rate ≈ {accept_pct:.0}%)"
    );

    if match_len == n_compare && n_compare > 0 {
        println!("\n=== Phase 5b real-prompt MTP: BIT-IDENTICAL PASS ===");
    } else {
        eprintln!(
            "\nNOTE: tokens diverge at index {match_len} on real prompt. Greedy MTP is \
             supposed to be bit-identical at T=0 — this confirms a cache-correctness \
             gap (Phase 5a memo flagged this as Phase 4 work)."
        );
    }
    Ok(())
}

#[cfg(not(feature = "mlx-native"))]
fn main() -> Result<()> {
    eprintln!("This example requires --features mlx-native");
    Ok(())
}
