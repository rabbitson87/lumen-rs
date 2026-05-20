//! Prompt-Lookup Decoding (PLD) smoke on Gemma 4.
//!
//! Same warm-vs-warm A/B protocol as `gemma4_mtp_real_prompt_smoke.rs`
//! but routes through `lookup_step` instead of `mtp_step`. No drafter is
//! loaded — the speculative drafts come from an n-gram match over the
//! generated context.
//!
//! Run:
//!   MODEL_ID=/path/to/models/gemma-4-26b-a4b-mlx-3bit \
//!   cargo run --release --features mlx-native -p lumen-mlx \
//!       --example gemma4_lookup_real_prompt_smoke
//!
//! Env:
//!   MAX_TOKENS         default 128 (longer than MTP smoke to give n-gram
//!                      matches time to appear in the generated stream)
//!   LUMEN_GEMMA4_LOOKUP_N    prefix length (default 3)
//!   LUMEN_GEMMA4_LOOKUP_K    max draft length (default 10)
//!   LUMEN_LOOKUP_DEBUG=1     log per-step drafts/preds

use std::path::Path;

use anyhow::{Context, Result};

#[cfg(feature = "mlx-native")]
fn main() -> Result<()> {
    use lumen_mlx::gemma4::{GenerateConfig, NativeGemma4Model};

    // Chat-templated "한국의 수도는?" — 18 ids on gemma-4-26b-a4b tokenizer.
    const KOREAN_PROMPT: &[u32] = &[
        2, 105, 2364, 107, 114216, 237281, 79301, 237170, 236881, 106, 107, 105, 4368, 107, 100,
        45518, 107, 101,
    ];

    // Chat-templated "다음 Python 코드를 그대로 다시 출력해주세요:\n```python\n
    // def bubble_sort(arr): ... ```\n출력:" — 113 ids. The model's expected
    // output is to echo the same bubble_sort code → maximal PLD match.
    const CODE_PROMPT: &[u32] = &[
        2, 105, 2364, 107, 181310, 17856, 36726, 70289, 114192, 55573, 149107, 219770, 236787, 107,
        2717, 6719, 107, 2063, 24225, 236779, 10479, 236769, 2762, 1473, 107, 140, 236749, 578,
        5980, 236769, 2762, 236768, 107, 140, 1708, 858, 528, 2644, 236769, 236749, 1473, 107, 144,
        1708, 673, 528, 2644, 236769, 236771, 236764, 538, 236772, 236747, 236772, 236770, 1473,
        107, 148, 584, 4617, 236840, 236804, 236842, 1890, 4617, 236840, 236804, 236862, 236770,
        9414, 107, 152, 2762, 236840, 236804, 1604, 4617, 236840, 236804, 236862, 236770, 236842,
        578, 4617, 236840, 236804, 236862, 236770, 1604, 4617, 236840, 236804, 236842, 107, 140,
        2060, 4617, 107, 2717, 107, 238778, 238862, 236787, 106, 107, 105, 4368, 107, 100, 45518,
        107, 101,
    ];

    // Moltis-style sports team matching prompt — chat-templated, 318 ids.
    // RAG-style: retrieved candidate teams in the prompt, expected output
    // is a tool_call JSON block that reuses tokens from the candidate
    // list (team name, league name, country) — maximum PLD match.
    const SPORTS_PROMPT: &[u32] = &[
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
        623, 79932, 827, 623, 125757, 1083, 11058, 238775, 238948, 17814, 190710, 238001, 827, 623,
        238775, 238948, 827, 623, 236794, 4868, 99471, 107, 236772, 3714, 14992, 1083, 623, 11962,
        23498, 827, 623, 80043, 1083, 623, 4967, 40777, 827, 623, 14801, 1083, 623, 79932, 827,
        623, 125757, 1083, 11058, 238505, 238855, 242271, 827, 623, 238505, 238855, 241901, 131812,
        827, 623, 11962, 236799, 99471, 108, 238778, 238862, 184412, 236787, 107, 14937, 1201,
        1083, 623, 10480, 236779, 14992, 827, 623, 33178, 1083, 3714, 14992, 1083, 623, 49256, 623,
        80043, 1083, 623, 49256, 623, 14801, 1083, 623, 24002, 1807, 108, 237586, 239247, 236787,
        165432, 238263, 30549, 242267, 237578, 242332, 106, 107, 105, 4368, 107, 100, 45518, 107,
        101,
    ];

    let prompt_kind = std::env::var("PROMPT_KIND").unwrap_or_else(|_| "code".into());
    let fixed_prompt: &[u32] = match prompt_kind.as_str() {
        "korean" => KOREAN_PROMPT,
        "sports" => SPORTS_PROMPT,
        "code" | _ => CODE_PROMPT,
    };

    let model_id = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| "/path/to/models/gemma-4-26b-a4b-mlx-3bit".into());
    let max_tokens: usize = std::env::var("MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);

    eprintln!("[lookup-smoke] loading trunk {model_id}");
    let model = NativeGemma4Model::load(Path::new(&model_id)).context("trunk load")?;

    // Production-realistic: cap output + stop on EOS (token 106 = <end_of_turn>
    // on Gemma 4). The LLM finishes the JSON tool_call in ~70 tokens; the
    // remaining ~230 tokens of `max_tokens=300` were ramble/self-repetition
    // that production never sees (Moltis cuts at finish_reason="stop").
    let stop_on_eos = std::env::var("STOP_ON_EOS")
        .map(|v| v != "0")
        .unwrap_or(true);
    let cfg = GenerateConfig {
        max_new_tokens: max_tokens,
        stop_on_eos,
        sampling: None,
    };

    // ── OFF: standard decode ──
    unsafe {
        std::env::set_var("LUMEN_GEMMA4_LOOKUP_SPEC", "0");
    }
    eprintln!("[lookup-smoke] OFF warmup pass");
    let _warm = model
        .generate(fixed_prompt, &cfg)
        .context("generate(OFF warmup)")?;
    eprintln!("[lookup-smoke] OFF timed pass (warm)");
    let t0 = std::time::Instant::now();
    let stats_off = model
        .generate(fixed_prompt, &cfg)
        .context("generate(OFF)")?;
    let off_wall = t0.elapsed().as_secs_f64() * 1e3;

    // ── ON: lookup decoding ──
    unsafe {
        std::env::set_var("LUMEN_GEMMA4_LOOKUP_SPEC", "1");
    }
    eprintln!("[lookup-smoke] ON warmup pass");
    let _warm_on = model
        .generate(fixed_prompt, &cfg)
        .context("generate(ON warmup)")?;
    eprintln!("[lookup-smoke] ON timed pass (warm)");
    let t0 = std::time::Instant::now();
    let stats_on = model.generate(fixed_prompt, &cfg).context("generate(ON)")?;
    let on_wall = t0.elapsed().as_secs_f64() * 1e3;

    let n_lookup = std::env::var("LUMEN_GEMMA4_LOOKUP_N")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3);
    let n_draft = std::env::var("LUMEN_GEMMA4_LOOKUP_K")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10);

    println!("\n=== Gemma 4 PLD smoke ===");
    println!(
        "  prompt_kind={prompt_kind} prompt_len={} max_tokens={max_tokens} \
         n_lookup={n_lookup} n_draft={n_draft}",
        fixed_prompt.len(),
    );
    println!(
        "  OFF: {} tokens in {:.0} ms (decode {:.1} tok/s, {} steps)",
        stats_off.generated_tokens.len(),
        off_wall,
        stats_off.decode_tok_per_sec,
        stats_off.decode_steps
    );
    println!(
        "  ON : {} tokens in {:.0} ms (decode {:.1} tok/s, {} lookup_steps)",
        stats_on.generated_tokens.len(),
        on_wall,
        stats_on.decode_tok_per_sec,
        stats_on.decode_steps
    );
    let speedup = off_wall / on_wall;
    println!("  wall-clock speedup: {speedup:.2}x");

    // Bit-identical check (greedy ⇒ MUST match).
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
    println!("\n  bit-identical prefix: {match_len}/{n_compare} tokens");
    if std::env::var("DUMP_TOKENS")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        println!("\nOFF_TOKENS={:?}", stats_off.generated_tokens);
        println!("ON_TOKENS={:?}", stats_on.generated_tokens);
    }

    // Average n_accepted per lookup_step (mean draft tokens accepted).
    // For PLD: total_committed = sum_steps(1 + n_accepted_k + (1 if matched)) so
    // it's not as clean as MTP — use a simpler proxy: total / steps.
    let avg_tok_per_step = if stats_on.decode_steps > 0 {
        stats_on.generated_tokens.len() as f64 / stats_on.decode_steps as f64
    } else {
        0.0
    };
    println!("  avg tokens/lookup_step = {avg_tok_per_step:.2} (vs OFF 1.00)",);

    if match_len == n_compare && n_compare > 0 {
        println!("\n=== PLD: BIT-IDENTICAL PASS ===");
    } else {
        eprintln!(
            "\nNOTE: tokens diverge at index {match_len}. PLD at T=0 should \
             be bit-identical — divergence indicates a verify/rollback gap."
        );
    }
    Ok(())
}

#[cfg(not(feature = "mlx-native"))]
fn main() -> Result<()> {
    eprintln!("This example requires --features mlx-native");
    Ok(())
}
