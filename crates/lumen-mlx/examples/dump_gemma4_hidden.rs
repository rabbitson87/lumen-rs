//! Single-prefill dump of Gemma 4 26B-A4B hidden states for divergence debug.
//!
//! With `LUMEN_DUMP_HIDDEN=/some/dir` set, runs one prefill on a fixed token
//! sequence (the same 18 ids `scripts/dump_mlx_gemma4_hidden.py` produces for
//! `"한국의 수도는?"` via the model's chat template) and writes:
//!     {dir}/embed.bin
//!     {dir}/L00.bin .. L29.bin
//!     {dir}/final_norm.bin
//!     {dir}/logits.bin
//!
//! Each blob is `b"TQHD" + rank(u32) + dims(u32 each, LE) + f32 bytes` —
//! consumable by `scripts/compare_gemma4_hidden.py`.
//!
//! Run:
//!   MLX_LOCAL_SOURCE_DIR=/path/to/mlx \
//!   LUMEN_DUMP_HIDDEN=/tmp/rust_gemma4_hidden \
//!   MODEL_ID=/path/to/models/gemma-4-26b-a4b-mlx-4bit \
//!   cargo run --release -p lumen-mlx --features mlx-native \
//!     --example dump_gemma4_hidden

#[cfg(feature = "mlx-native")]
fn main() -> anyhow::Result<()> {
    use anyhow::Context;
    use lumen_mlx::gemma4::{NativeGemma4Model, set_forward_step};
    use std::path::Path;

    // Fixed token sequence: `tokenizer.apply_chat_template([{"role":"user","content":"한국의 수도는?"}], add_generation_prompt=True)`
    // → 18 ids on the mlx-community/gemma-4-26b-a4b-mlx-4bit tokenizer. Hardcoded
    //   so this example doesn't need to know the chat template details; the
    //   Python reference script generates the same ids.
    const FIXED_IDS: &[u32] = &[
        2, 105, 2364, 107, 114216, 237281, 79301, 237170, 236881, 106, 107, 105, 4368, 107, 100,
        45518, 107, 101,
    ];

    let model_id = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| "/path/to/models/gemma-4-26b-a4b-mlx-4bit".into());
    let dump_dir =
        std::env::var("LUMEN_DUMP_HIDDEN").unwrap_or_else(|_| "/tmp/rust_gemma4_hidden".into());
    std::fs::create_dir_all(&dump_dir).with_context(|| format!("mkdir {dump_dir}"))?;
    // SAFETY: bench is single-threaded at this point; gemma4_moe.rs reads
    //         LUMEN_DUMP_HIDDEN once via OnceLock on the first dump call.
    unsafe { std::env::set_var("LUMEN_DUMP_HIDDEN", &dump_dir) };

    eprintln!("[dump-gemma4] loading {model_id}");
    let model = NativeGemma4Model::load(Path::new(&model_id)).context("load model")?;
    eprintln!("[dump-gemma4] vocab={}", model.vocab_size());

    let mut cache = model.make_cache();
    eprintln!("[dump-gemma4] prefilling {} tokens...", FIXED_IDS.len());
    set_forward_step(0);
    let logits = model.forward(FIXED_IDS, &mut cache).context("forward")?;

    let argmax = model.argmax_last_token(&logits).context("argmax")?;
    let dims = logits.shape();
    eprintln!("[dump-gemma4] logits shape={dims:?} prefill_argmax={argmax}");
    eprintln!("[dump-gemma4] wrote tensors to {dump_dir}");

    // Optional N-step greedy decode chain. Used by Step 3.5 of the
    // dispatch-divergence debug to find which decode step the Rust
    // backend first picks a different token from mlx_lm.
    let steps: usize = std::env::var("LUMEN_DECODE_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if steps > 0 {
        let mut decoded: Vec<u32> = Vec::with_capacity(steps);
        decoded.push(argmax);
        let mut next = argmax;
        for step in 1..steps {
            set_forward_step(step);
            let lg = model
                .forward(&[next], &mut cache)
                .with_context(|| format!("decode step {step}"))?;
            next = model.argmax_last_token(&lg)?;
            decoded.push(next);
        }
        // JSON-like line for easy diff. Tokens only — decoding to text is
        // left to the comparison script which has the tokenizer handy.
        let toks_csv = decoded
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",");
        println!("RUST_DECODE_TOKENS=[{toks_csv}]");
    }
    Ok::<(), anyhow::Error>(())
}

#[cfg(not(feature = "mlx-native"))]
fn main() {
    eprintln!("dump_gemma4_hidden requires --features mlx-native");
}
