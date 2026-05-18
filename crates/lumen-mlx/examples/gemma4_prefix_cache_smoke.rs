//! Prefix cache smoke for Moltis sports matching workload.
//!
//! Scenario: a batch of 3 matching requests sharing the same ~300-token
//! system+candidates prefix but with different user queries at the end.
//! Without prefix cache, every request pays the full prefill cost
//! (~3-5s on Mac Mini). With prefix cache, only the first request pays;
//! subsequent ones forward only the differing suffix (~3-5 tokens).
//!
//! This smoke measures the per-request prefill wall directly via
//! `forward_last_token` so the cache stays under the smoke's control.
//! No `generate()` refactor needed for this measurement.
//!
//! Run:
//!   MODEL_ID=/path/to/models/gemma-4-26b-a4b-mlx-3bit \
//!   cargo run --release --features mlx-native -p lumen-mlx \
//!       --example gemma4_prefix_cache_smoke

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};

#[cfg(feature = "mlx-native")]
fn main() -> Result<()> {
    use lumen_mlx::gemma4::NativeGemma4Model;

    let model_id = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| "/path/to/models/gemma-4-26b-a4b-mlx-3bit".into());

    eprintln!("[prefix-cache] loading trunk {model_id}");
    let model = NativeGemma4Model::load(Path::new(&model_id)).context("trunk load")?;

    // Sports matching prompt suffix tokens differ from the canonical
    // SPORTS_PROMPT we measured before only in the user query at the very
    // end. Production batch uses the same system + candidates list per
    // 30-min batch, but a different team name to match.
    //
    // We build 3 prompts that share a common ~300-token prefix and diverge
    // only in the last ~5 tokens (the team name). The user query token IDs
    // were extracted via a tokenizer offline.
    //
    // For this smoke we re-use the SPORTS_PROMPT (318 ids) for prompt 1.
    // Prompts 2 and 3 mutate the last 5 tokens (the user query tail) so
    // LCP ≈ 313 vs 318 = 98% of prefix shared.
    const SHARED_PREFIX_LEN: usize = 313;
    const SPORTS_PROMPT_1: &[u32] = &[
        2, 105, 2364, 107, 213229, 78622, 126623, 237223, 42813, 189515, 121876, 237293, 30549,
        242267, 152748, 236761, 43625, 157539, 148583, 42757, 33742, 209116, 121876, 237293, 71207,
        14377, 90918, 10434, 184412, 7246, 149107, 152748, 236761, 108, 238871, 237660, 121876,
        236787, 107, 236772, 3714, 14992, 1083, 623, 74246, 3640, 827, 623, 80043, 1083, 623,
        126350, 9654, 827, 623, 14801, 1083, 623, 76395, 827, 623, 125757, 1083, 11058, 241592,
        238263, 827, 623, 241592, 238597, 107792, 17920, 237610, 237077, 239477, 238001, 827, 623,
        3366, 201285, 99471, 107, 236772, 3714, 14992, 1083, 623, 74246, 4085, 827, 623, 80043,
        1083, 623, 126350, 9654, 827, 623, 14801, 1083, 623, 76395, 827, 623, 125757, 1083, 11058,
        241592, 237462, 827, 623, 241592, 238597, 107792, 9420, 239477, 827, 623, 3366, 4085,
        99471, 107, 236772, 3714, 14992, 1083, 623, 98125, 827, 623, 80043, 1083, 623, 126350,
        9654, 827, 623, 14801, 1083, 623, 76395, 827, 623, 125757, 1083, 11058, 237469, 239029,
        240525, 827, 623, 236798, 11962, 99471, 107, 236772, 3714, 14992, 1083, 623, 100203, 827,
        623, 80043, 1083, 623, 126350, 9654, 827, 623, 14801, 1083, 623, 76395, 827, 623, 125757,
        1083, 11058, 247524, 237462, 827, 623, 236780, 11962, 99471, 107, 236772, 3714, 14992,
        1083, 623, 20235, 19627, 827, 623, 80043, 1083, 623, 4967, 40777, 827, 623, 14801, 1083,
        623, 79932, 827, 623, 125757, 1083, 11058, 238775, 238948, 17814, 190710, 238001, 827,
        623, 238775, 238948, 827, 623, 236794, 4868, 99471, 107, 236772, 3714, 14992, 1083, 623,
        11962, 23498, 827, 623, 80043, 1083, 623, 4967, 40777, 827, 623, 14801, 1083, 623, 79932,
        827, 623, 125757, 1083, 11058, 238505, 238855, 242271, 827, 623, 238505, 238855, 241901,
        131812, 827, 623, 11962, 236799, 99471, 108, 238778, 238862, 184412, 236787, 107, 14937,
        1201, 1083, 623, 10480, 236779, 14992, 827, 623, 33178, 1083, 3714, 14992, 1083, 623,
        49256, 623, 80043, 1083, 623, 49256, 623, 14801, 1083, 623, 24002, 1807, 108, 237586,
        239247, 236787, 165432, 238263, 30549, 242267, 237578, 242332, 106, 107, 105, 4368, 107,
        100, 45518, 107, 101,
    ];
    // Prompt 2 / 3 mutate the user query token (1-2 tokens worth of team
    // name change). For this smoke we synthetically shift the last 5
    // tokens by 1 to simulate a different team query. Real production
    // would re-tokenize a different team name. Conservative shift —
    // doesn't have to be a valid query, only differ at the end so we can
    // measure prefix cache hit accurately.
    let mut prompt_2: Vec<u32> = SPORTS_PROMPT_1.to_vec();
    let n = prompt_2.len();
    prompt_2[n - 5] = 123;
    prompt_2[n - 4] = 456;
    let mut prompt_3: Vec<u32> = SPORTS_PROMPT_1.to_vec();
    let n3 = prompt_3.len();
    prompt_3[n3 - 5] = 789;
    prompt_3[n3 - 4] = 1011;

    let prompts: Vec<&[u32]> = vec![SPORTS_PROMPT_1, &prompt_2, &prompt_3];

    eprintln!(
        "[prefix-cache] 3 prompts, len={} each. Expected LCP across all = {SHARED_PREFIX_LEN}",
        SPORTS_PROMPT_1.len()
    );

    // Warmup pass so 1st measured request doesn't pay Metal kernel compile.
    {
        let mut warm_cache = model.make_cache();
        let logits = model
            .forward_last_token(prompts[0], &mut warm_cache)
            .context("warmup forward")?;
        logits.eval().ok();
    }

    // ─── Scenario A: NO prefix cache (baseline) ───
    // Each request gets a fresh cache and pays full prefill.
    println!("\n=== Scenario A: NO prefix cache (full prefill each request) ===");
    let mut total_a_ms = 0.0_f64;
    for (i, prompt) in prompts.iter().enumerate() {
        let mut cache = model.make_cache();
        let t0 = Instant::now();
        let logits = model
            .forward_last_token(prompt, &mut cache)
            .context("forward_last_token (A)")?;
        logits.eval().context("eval (A)")?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        total_a_ms += ms;
        println!("  request {}: prefill_only = {:.0} ms", i + 1, ms);
    }
    println!("  total prefill (A): {:.0} ms", total_a_ms);

    // ─── Scenario B: WITH prefix cache (LCP + truncate + suffix prefill) ───
    println!("\n=== Scenario B: WITH prefix cache (LCP truncate + suffix forward) ===");
    let mut cache = model.make_cache();
    let mut last_prompt: Vec<u32> = Vec::new();
    let mut total_b_ms = 0.0_f64;
    for (i, prompt) in prompts.iter().enumerate() {
        // Longest common prefix
        let lcp = prompt
            .iter()
            .zip(last_prompt.iter())
            .take_while(|(a, b)| a == b)
            .count();
        // Truncate cache down to LCP if necessary
        if lcp < cache.offset() {
            cache.truncate_to(lcp).context("truncate_to lcp")?;
        }
        // Forward only the suffix
        let suffix = &prompt[lcp..];
        let t0 = Instant::now();
        let logits = model
            .forward_last_token(suffix, &mut cache)
            .context("forward_last_token (B suffix)")?;
        logits.eval().context("eval (B)")?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        total_b_ms += ms;
        println!(
            "  request {}: lcp={} suffix_len={} prefill = {:.0} ms",
            i + 1,
            lcp,
            suffix.len(),
            ms
        );
        last_prompt = prompt.to_vec();
    }
    println!("  total prefill (B): {:.0} ms", total_b_ms);

    let speedup = if total_b_ms > 0.0 {
        total_a_ms / total_b_ms
    } else {
        0.0
    };
    let saved_ms = total_a_ms - total_b_ms;
    println!("\n=== Prefix cache summary ===");
    println!("  scenario A total = {:.0} ms (no cache)", total_a_ms);
    println!("  scenario B total = {:.0} ms (with cache)", total_b_ms);
    println!(
        "  saved: {:.0} ms across 3 requests = {:.2}x speedup",
        saved_ms, speedup
    );
    println!(
        "  per-request after cache hit: {:.0} ms (vs {:.0} ms cold)",
        total_b_ms / 3.0,
        total_a_ms / 3.0
    );
    Ok(())
}

#[cfg(not(feature = "mlx-native"))]
fn main() -> Result<()> {
    eprintln!("This example requires --features mlx-native");
    Ok(())
}
