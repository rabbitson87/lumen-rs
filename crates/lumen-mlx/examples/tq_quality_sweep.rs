//! Multi-prompt quality sweep for TurboQuant Stage 1 (+ optional QJL).
//!
//! Loads the model ONCE, then runs a fixed prompt suite covering factual,
//! code, math, creative, and technical categories. Used to validate
//! TQ Stage 1's quality at bits=2/3 (the regime where mlx-affine can't
//! reach — TQ's production raison d'être).
//!
//! Workflow: run this example once per (bits, QJL) config; diff the
//! outputs to assess quality degradation vs OFF baseline.
//!
//! Run:
//!   # OFF baseline (one model load, all prompts):
//!   MLX_LOCAL_SOURCE_DIR=/path/to/mlx \
//!   MODEL_ID=/path/to/model \
//!   cargo run --release -p lumen-mlx \
//!       --example tq_quality_sweep --features mlx-native \
//!       > /tmp/sweep_off.txt
//!
//!   # TQ bits=2 + QJL:
//!   LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT=1 \
//!   LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT_BITS=2 \
//!   LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT_QJL=1 \
//!   LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT_QJL_M=1024 \
//!   ...same other env... \
//!   cargo run ... > /tmp/sweep_b2_qjl.txt
//!
//!   diff /tmp/sweep_off.txt /tmp/sweep_b2_qjl.txt

#[cfg(feature = "mlx-native")]
use anyhow::{Context, Result};

#[cfg(feature = "mlx-native")]
const DEFAULT_PROMPTS: &[(&str, &str)] = &[
    (
        "factual",
        "Explain why the sky is blue in two short sentences.",
    ),
    (
        "code",
        "Write a Python function that returns the nth Fibonacci number using dynamic programming.",
    ),
    (
        "math",
        "What is 27 times 43? Show your reasoning step by step.",
    ),
    ("creative", "Write a haiku about morning coffee."),
    (
        "technical",
        "In one sentence, explain the difference between bfloat16 and float16.",
    ),
];

#[cfg(feature = "mlx-native")]
fn main() -> Result<()> {
    use lumen_mlx::gemma4::Gemma4Backend;
    use std::path::PathBuf;

    let model_id = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| "/path/to/models/gemma-4-26b-a4b-mlx-4bit".into());
    let max_tokens: usize = std::env::var("MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let config_label = std::env::var("SWEEP_LABEL").unwrap_or_else(|_| "default".into());

    eprintln!("[tq-sweep] model={model_id}");
    eprintln!("[tq-sweep] config={config_label}");
    eprintln!("[tq-sweep] max_tokens={max_tokens}");

    for key in [
        "LUMEN_GEMMA4_QUANT_KV",
        "LUMEN_GEMMA4_QUANT_KV_SLIDING",
        "LUMEN_GEMMA4_QUANT_KV_SLIDING_BITS",
        "LUMEN_GEMMA4_QUANT_KV_SLIDING_GROUP_SIZE",
        "LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT",
        "LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT_BITS",
        "LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT_QJL",
        "LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT_QJL_M",
        "LUMEN_GEMMA4_TQ_BAKE_R",
        "LUMEN_GEMMA4_TQ_BAKE_R_V",
        "LUMEN_GEMMA4_TQ_BAKE_R_O",
    ] {
        if let Ok(v) = std::env::var(key) {
            eprintln!("[tq-sweep] {key}={v}");
        }
    }

    let model_path = PathBuf::from(&model_id);
    let mut backend =
        Gemma4Backend::from_dir("tq-sweep", &model_path).context("Gemma4Backend::from_dir")?;
    eprintln!("[tq-sweep] model loaded");

    println!("===CONFIG={config_label}===");
    println!("===MAX_TOKENS={max_tokens}===");

    for (category, prompt) in DEFAULT_PROMPTS {
        eprintln!("[tq-sweep] running prompt: {category}");
        let messages: Vec<(String, String)> = vec![("user".to_string(), prompt.to_string())];
        let prompt_ids = backend
            .build_chat_input(&messages, false)
            .with_context(|| format!("build_chat_input for {category}"))?;

        let out_tokens = backend
            .generate(&prompt_ids, max_tokens, 0.0, 1.0, &Default::default())
            .with_context(|| format!("generate for {category}"))?;
        let out_text = backend
            .decode(&out_tokens)
            .unwrap_or_else(|_| "<decode failed>".into());

        println!("---PROMPT---");
        println!("category: {category}");
        println!("text: {prompt}");
        println!("---TOKENS---");
        // Single line, space-separated, for easy diffing.
        let line: Vec<String> = out_tokens.iter().map(|t| t.to_string()).collect();
        println!("{}", line.join(" "));
        println!("---TEXT---");
        println!("{}", out_text.trim_end());
        println!("---END---");
    }

    Ok(())
}

#[cfg(not(feature = "mlx-native"))]
fn main() {
    eprintln!("tq_quality_sweep requires --features mlx-native");
    std::process::exit(0);
}
