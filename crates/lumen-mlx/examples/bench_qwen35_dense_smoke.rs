//! Trunk-only coherence smoke for the Qwen3.5/3.6 DENSE path (9B/27B).
//! Loads the model, encodes a real prompt, greedily decodes, and prints the
//! continuation — confirms the dense SwiGLU MLP + affine trunk produce coherent
//! text (not garbage). No MTP head involved.
//!
//!   LUMEN_MLX_BACKEND=native MODEL_ID=~/models/Qwen3.5-9B-MTPLX-Speed \
//!     cargo run --release -p lumen-mlx --features mlx-native --example bench_qwen35_dense_smoke

use anyhow::{Result, anyhow};
use lumen_mlx::MlxBackend;

fn main() -> Result<()> {
    let model_id =
        std::env::var("MODEL_ID").map_err(|_| anyhow!("set MODEL_ID to the dense model dir"))?;
    let n_gen: usize = std::env::var("GEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(48);
    let prompt_text = std::env::var("PROMPT").unwrap_or_else(|_| {
        "<|im_start|>user\nExplain in one paragraph why the sky is blue.\
         <|im_end|>\n<|im_start|>assistant\n"
            .to_string()
    });

    let mut backend = MlxBackend::load(&model_id)?;
    let prompt = backend.encode(&prompt_text)?;
    println!("model: {model_id}\nprompt tokens: {}", prompt.len());

    let qwen = backend
        .as_qwen35_mut()
        .ok_or_else(|| anyhow!("not a Qwen3.5-family backend"))?;

    let seq: u64 = 1;
    let (mut last, mut pos) = qwen.prefill(seq, &prompt)?;
    let mut out = vec![last];
    for _ in 0..n_gen {
        let (n, p) = qwen.decode_step(seq, last, pos)?;
        out.push(n);
        last = n;
        pos = p;
    }
    qwen.remove_seq(seq)?;
    // `drop(qwen)` used to be here. `qwen` is a borrow, so dropping it freed
    // nothing; NLL already ends the borrow at its last use, which is what the
    // line was actually achieving.

    let text = backend.decode(&out).unwrap_or_default();
    println!("--- continuation ---\n{text}");
    Ok(())
}
