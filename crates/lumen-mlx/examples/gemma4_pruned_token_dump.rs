//! Dump generated tokens (greedy) from `MODEL_ID` for two fixed prompts.
//!
//! Used to A/B-compare a pruned Gemma 4 build against the source by running
//! this binary twice with different `MODEL_ID` env vars and diffing the
//! printed token IDs.
//!
//! Run:
//!   MODEL_ID=/path/to/model cargo run --release --features mlx-native \
//!     -p lumen-mlx --example gemma4_pruned_token_dump
//!
//! Env:
//!   MAX_TOKENS  default 32

use std::path::Path;

use anyhow::{Context, Result};

#[cfg(feature = "mlx-native")]
fn main() -> Result<()> {
    use lumen_mlx::gemma4::{GenerateConfig, NativeGemma4Model};

    // Same prompts as `gemma4_lookup_real_prompt_smoke.rs`.
    const KOREAN_PROMPT: &[u32] = &[
        2, 105, 2364, 107, 114216, 237281, 79301, 237170, 236881, 106, 107, 105, 4368, 107, 100,
        45518, 107, 101,
    ];
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
        165432, 238263, 30549, 242267, 237578, 242332, 106, 107, 105, 4368, 107, 101,
    ];

    let model_id = std::env::var("MODEL_ID").context("MODEL_ID env required")?;
    let max_tokens: usize = std::env::var("MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    eprintln!("[dump] loading {model_id}");
    let model = NativeGemma4Model::load(Path::new(&model_id)).context("load")?;

    let cfg = GenerateConfig {
        max_new_tokens: max_tokens,
        stop_on_eos: true,
        sampling: None,
    };

    // Disable lookup spec — pure greedy for stable cross-model compare.
    unsafe {
        std::env::set_var("LUMEN_GEMMA4_LOOKUP_SPEC", "0");
    }

    for (kind, prompt) in [("korean", KOREAN_PROMPT), ("sports", SPORTS_PROMPT)] {
        eprintln!("[dump] {kind} prompt_len={}", prompt.len());
        let _warm = model.generate(prompt, &cfg).context("warmup")?;
        let stats = model.generate(prompt, &cfg).context("timed")?;
        let toks = &stats.generated_tokens;
        println!("=== {kind} ({} tokens, {:.1} tok/s) ===", toks.len(), stats.decode_tok_per_sec);
        for (i, t) in toks.iter().enumerate() {
            print!("{t}");
            if i + 1 < toks.len() {
                print!(",");
            }
        }
        println!();
    }
    Ok(())
}

#[cfg(not(feature = "mlx-native"))]
fn main() -> Result<()> {
    Err(anyhow::anyhow!("build with --features mlx-native"))
}
