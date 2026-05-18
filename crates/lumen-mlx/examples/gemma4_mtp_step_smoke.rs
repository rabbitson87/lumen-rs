//! load trunk + drafter, prefill a small prompt,
//! call `mtp_step` once with `n_draft=4`, verify outputs are sane.
//!
//! This DOES not yet compare bit-identicality vs MTP-off — that requires
//! aligning the generation loop. Phase 5 bench will do the proper A/B.
//!
//! Run:
//!   MLX_LOCAL_SOURCE_DIR=/path/to/Documents/GitHub/mlx \
//!   MODEL_ID=/path/to/models/gemma-4-26b-a4b-mlx-4bit \
//!   DRAFTER_DIR=/path/to/models/gemma-4-26B-A4B-it-assistant-bf16 \
//!   cargo run --release -p lumen-mlx \
//!       --example gemma4_mtp_step_smoke --features mlx-native

use std::path::Path;

use anyhow::{Context, Result};

#[cfg(feature = "mlx-native")]
fn main() -> Result<()> {
    use lumen_mlx::gemma4::{NativeGemma4Model, NativeGemma4PromptCache};

    let model_id = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| "/path/to/models/gemma-4-26b-a4b-mlx-4bit".into());
    let drafter_dir = std::env::var("DRAFTER_DIR").unwrap_or_else(|_| {
        "/path/to/models/gemma-4-26B-A4B-it-assistant-bf16".into()
    });
    let n_draft: usize = std::env::var("N_DRAFT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    eprintln!("[mtp-step-smoke] loading trunk {model_id}");
    let mut model = NativeGemma4Model::load(Path::new(&model_id)).context("trunk load")?;
    eprintln!("[mtp-step-smoke] enabling MTP from {drafter_dir}");
    model
        .try_enable_mtp(Path::new(&drafter_dir))
        .context("try_enable_mtp")?;
    assert!(model.mtp_enabled());

    // Synthetic deterministic prompt (same pattern as bench_gemma4_native_e2e).
    let vocab = model.vocab_size() as u32;
    let prompt: Vec<u32> = (0..128)
        .map(|i| 10 + ((i as u32 * 7) % (vocab.saturating_sub(20).max(200))))
        .collect();

    let mut cache = NativeGemma4PromptCache::for_config(model.text_config());

    eprintln!("[mtp-step-smoke] prefill {} tokens", prompt.len());
    let prefill_logits = model
        .forward_last_token(&prompt, &mut cache)
        .context("prefill")?;
    let first_token = model
        .argmax_last_token(&prefill_logits)
        .context("argmax(prefill)")?;
    eprintln!(
        "[mtp-step-smoke] prefill done; cache offset={}, first decode token={first_token}",
        cache.offset()
    );

    eprintln!("[mtp-step-smoke] calling mtp_step(n_draft={n_draft})");
    let out = model
        .mtp_step(&mut cache, first_token, n_draft)
        .context("mtp_step")?;
    eprintln!(
        "[mtp-step-smoke] mtp_step done; committed.len()={}, n_attempted={}, n_accepted={}, cache_offset={}",
        out.committed.len(),
        out.n_attempted,
        out.n_accepted,
        cache.offset()
    );
    println!("\n=== MTP step output ===");
    println!("  n_attempted = {}", out.n_attempted);
    println!("  n_accepted  = {}", out.n_accepted);
    println!("  committed   = {:?}", out.committed);
    println!("  cache_offset after = {}", cache.offset());

    // Sanity checks:
    // - committed length == 1 (next_token) + n_accepted + 1 (correction/bonus) = n_accepted + 2
    let expected_len = out.n_accepted + 2;
    assert_eq!(
        out.committed.len(),
        expected_len,
        "committed.len() {} != expected {}",
        out.committed.len(),
        expected_len
    );
    // - cache offset advanced by 1 (Step A) + 1 (next_token) + n_accepted (drafts) = n_accepted + 2
    let expected_cache_advance = 1 + 1 + out.n_accepted;
    let actual_advance = cache.offset() - (128); // prompt was 128
    assert_eq!(
        actual_advance, expected_cache_advance,
        "cache advanced by {actual_advance}, expected {expected_cache_advance}"
    );
    // - tokens are all in vocab range
    for t in &out.committed {
        assert!(*t < vocab, "token {t} out of vocab range");
    }

    println!("\n=== Phase 3 mtp_step end-to-end: PASS ===");
    Ok(())
}

#[cfg(not(feature = "mlx-native"))]
fn main() -> Result<()> {
    eprintln!("This example requires --features mlx-native");
    Ok(())
}
