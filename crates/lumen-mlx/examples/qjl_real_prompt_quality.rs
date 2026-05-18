//! Real-prompt quality eval for TurboQuant Stage-1 + QJL Stage-2 vs bf16
//! baseline.
//!
//! The bench `bench_gemma4_native_e2e` uses a synthetic arithmetic-
//! progression prompt that is hyper-sensitive to KV cache fidelity:
//! every quant variant (including ones that are mathematically very close
//! to bf16) degenerates within a handful of tokens. This example uses
//! real chat prompts — coherent natural language — which is what
//! production traffic actually looks like, and where the model's
//! attention patterns are robust to small KV perturbations.
//!
//! Workflow: run this example twice, once with no TQ env vars (bf16
//! baseline) and once with the TQ+QJL gates set; diff the printed tokens
//! / text. The output is structured so a simple `diff` reveals where the
//! TQ+QJL path drifts from the baseline.
//!
//! Run:
//!   # baseline:
//!   MLX_LOCAL_SOURCE_DIR=/path/to/mlx \
//!   MODEL_ID=/path/to/gemma-4-26b-a4b-mlx-4bit \
//!   PROMPT="Explain why the sky is blue in two sentences." \
//!   cargo run --release -p lumen-mlx \
//!       --example qjl_real_prompt_quality --features mlx-native \
//!       > /tmp/qjl_off.txt
//!
//!   # TQ + QJL:
//!   LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT=1 \
//!   LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT_QJL=1 \
//!   LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT_BITS=4 \
//!   ...same other env... \
//!   cargo run --release -p lumen-mlx \
//!       --example qjl_real_prompt_quality --features mlx-native \
//!       > /tmp/qjl_on.txt
//!
//!   diff /tmp/qjl_off.txt /tmp/qjl_on.txt

#[cfg(feature = "mlx-native")]
use anyhow::{Context, Result};

#[cfg(feature = "mlx-native")]
fn main() -> Result<()> {
    use lumen_mlx::gemma4::Gemma4Backend;
    use std::path::PathBuf;

    let model_id = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| "/path/to/models/gemma-4-26b-a4b-mlx-4bit".into());
    let prompt = std::env::var("PROMPT")
        .unwrap_or_else(|_| "Explain why the sky is blue in two sentences.".into());
    let max_tokens: usize = std::env::var("MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);

    eprintln!("[qjl-real-prompt] model={model_id}");
    eprintln!("[qjl-real-prompt] prompt={prompt:?}");

    // Surface every env gate that affects KV cache so the diff against
    // another run is unambiguous about what changed.
    for key in [
        "LUMEN_GEMMA4_QUANT_KV",
        "LUMEN_GEMMA4_QUANT_KV_SLIDING",
        "LUMEN_GEMMA4_QUANT_KV_SLIDING_BITS",
        "LUMEN_GEMMA4_QUANT_KV_SLIDING_GROUP_SIZE",
        "LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT",
        "LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT_BITS",
        "LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT_QJL",
        "LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT_QJL_M",
        "LUMEN_GEMMA4_TQ_FUSED_ENCODE",
    ] {
        if let Ok(v) = std::env::var(key) {
            eprintln!("[qjl-real-prompt] {key}={v}");
        }
    }

    let model_path = PathBuf::from(&model_id);
    let mut backend = Gemma4Backend::from_dir("qjl-real-prompt", &model_path)
        .context("Gemma4Backend::from_dir")?;

    let messages: Vec<(String, String)> = vec![("user".to_string(), prompt.clone())];
    let prompt_ids = backend
        .build_chat_input(&messages, /* thinking */ false)
        .context("build_chat_input")?;

    eprintln!(
        "[qjl-real-prompt] prompt tokens (len={}): {:?}",
        prompt_ids.len(),
        &prompt_ids[..prompt_ids.len().min(16)]
    );

    let out_tokens = backend
        .generate(&prompt_ids, max_tokens, 0.0, 1.0)
        .context("generate")?;
    let out_text = backend
        .decode(&out_tokens)
        .unwrap_or_else(|_| "<decode failed>".into());

    println!("---TOKENS---");
    for (i, t) in out_tokens.iter().enumerate() {
        println!("{:>3} {}", i, t);
    }
    println!("---TEXT---");
    println!("{}", out_text.trim_end());

    Ok(())
}

#[cfg(not(feature = "mlx-native"))]
fn main() {
    eprintln!("qjl_real_prompt_quality requires --features mlx-native");
    std::process::exit(0);
}
