//! Verify pruned Gemma 4 recognizes English sport team / league names.
//!
//! Probes the Moltis matching workload: Korean query → JSON tool call
//! using English team & league names. The token output is decoded and
//! printed for both source & pruned model comparison.
//!
//! Run:
//!   MODEL_ID=/path/to/model cargo run --release --features mlx-native \
//!     -p lumen-mlx --example gemma4_english_team_recognition

use std::path::Path;

use anyhow::{Context, Result};

#[cfg(feature = "mlx-native")]
fn main() -> Result<()> {
    use lumen_mlx::gemma4::{GenerateConfig, NativeGemma4Model};

    // 4 Moltis-style queries each naming a different English team. The
    // model should produce a JSON tool_call containing the canonical
    // English team name + league name. Each prompt is chat-templated.
    let queries: &[(&str, &[u32])] = &[
        // "Liverpool 이 어느 리그 소속인지 알려줘"
        (
            "Liverpool",
            &[
                2, 105, 2364, 107, 98125, 4214, 107561, 225183, 18004, 238701, 93860, 88440,
                242332, 106, 107, 105, 4368, 107, 100, 45518, 107, 101,
            ],
        ),
        // "Real Madrid 가 어느 리그?"
        (
            "Real Madrid",
            &[
                2, 105, 2364, 107, 20235, 19627, 8486, 107561, 225183, 236881, 106, 107, 105, 4368,
                107, 100, 45518, 107, 101,
            ],
        ),
        // "Manchester City 의 리그명을 말해줘"
        (
            "Manchester City",
            &[
                2, 105, 2364, 107, 74246, 4085, 18132, 225183, 127500, 18906, 237578, 242332, 106,
                107, 105, 4368, 107, 100, 45518, 107, 101,
            ],
        ),
        // "Bayern Munich 어느 리그?"
        (
            "Bayern Munich",
            &[
                2, 105, 2364, 107, 218437, 46566, 107561, 225183, 236881, 106, 107, 105, 4368, 107,
                100, 45518, 107, 101,
            ],
        ),
    ];

    let model_id = std::env::var("MODEL_ID").context("MODEL_ID env required")?;
    eprintln!("[probe] loading {model_id}");
    let model = NativeGemma4Model::load(Path::new(&model_id)).context("load")?;

    let cfg = GenerateConfig {
        max_new_tokens: 24,
        stop_on_eos: true,
        sampling: None,
    };

    unsafe {
        std::env::set_var("LUMEN_GEMMA4_LOOKUP_SPEC", "0");
    }

    // Warm
    let _ = model.generate(queries[0].1, &cfg).context("warm")?;

    for (label, prompt) in queries {
        let stats = model.generate(prompt, &cfg).context("gen")?;
        println!(
            "=== query={label} ({} tok generated) ===",
            stats.generated_tokens.len()
        );
        for (i, t) in stats.generated_tokens.iter().enumerate() {
            print!("{t}");
            if i + 1 < stats.generated_tokens.len() {
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
