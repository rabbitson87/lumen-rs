//! End-to-end image → text with the native Gemma 4 vision tower.
//!
//! Builds a Gemma 4 chat turn containing one image block
//! (`<|image>` + N × `<|image|>` + `<image|>`), runs prefill with the vision
//! features spliced over the placeholder rows, and greedily decodes a reply.
//!
//! Run:
//!   LUMEN_VISION=1 MODEL_ID=/path/to/gemma-4-26b-a4b \
//!     cargo run --release --features mlx-native -p lumen-mlx \
//!     --example gemma4_vision_describe -- <image.png> ["prompt text"]

use std::path::Path;

use anyhow::{Context, Result};

#[cfg(feature = "mlx-native")]
fn main() -> Result<()> {
    use lumen_mlx::gemma4::{GenerateConfig, NativeGemma4Model};
    use tokenizers::Tokenizer;

    let mut args = std::env::args().skip(1);
    let image_path = args
        .next()
        .context("usage: gemma4_vision_describe <image> [prompt]")?;
    let user_text = args
        .next()
        .unwrap_or_else(|| "Describe this image in one sentence.".to_string());

    let model_id = std::env::var("MODEL_ID").context("MODEL_ID env required")?;
    let model_dir = Path::new(&model_id);

    if std::env::var("LUMEN_VISION").ok().as_deref() != Some("1") {
        eprintln!("[warn] LUMEN_VISION=1 is not set — the tower will not load");
    }

    // Cap MLX's buffer pool before anything allocates. The 26B weights already
    // sit at ~15 GB, so on a 36 GB machine an uncapped cache plus the vision
    // tower's activations will push the system into swap.
    let mem_gb: f64 = std::env::var("LUMEN_MEM_LIMIT_GB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20.0);
    let cache_gb: f64 = std::env::var("LUMEN_CACHE_LIMIT_GB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2.0);
    lumen_mlx::metal_memory::set_memory_limit((mem_gb * 1024.0 * 1024.0 * 1024.0) as usize);
    lumen_mlx::metal_memory::set_cache_limit((cache_gb * 1024.0 * 1024.0 * 1024.0) as usize);
    eprintln!("[vision] mlx limits: memory {mem_gb} GB, cache {cache_gb} GB");

    eprintln!("[vision] loading {model_id}");
    let model = NativeGemma4Model::load(model_dir).context("load model")?;
    anyhow::ensure!(
        model.vision_available(),
        "vision tower unavailable — set LUMEN_VISION=1 and use a checkpoint that still \
         ships vision_tower.* weights"
    );

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

    let bytes = std::fs::read(&image_path).with_context(|| format!("read {image_path}"))?;

    // Decode + resize once. The prepared image sizes the prompt's placeholder
    // run below and is handed straight to the prefill, so the tower runs once
    // and the image is never decoded twice.
    let prepared = model.prepare_image(&bytes).context("prepare image")?;
    let num_soft_tokens = prepared.num_soft_tokens;
    eprintln!("[vision] image → {num_soft_tokens} soft tokens");

    let image_token = model
        .image_token_id()
        .context("config has no image_token_id")?;
    // Gemma 4 turn scaffolding: <bos> <|turn> "user" \n … <turn|> \n <|turn> "model" \n
    const TOK_BOS: u32 = 2;
    const TOK_TURN_OPEN: u32 = 105;
    const TOK_TURN_CLOSE: u32 = 106;
    const TOK_NEWLINE: u32 = 107;
    const TOK_USER: u32 = 2364;
    const TOK_MODEL: u32 = 4368;
    let boi = 255999u32;
    let eoi = 258882u32;

    let text_ids: Vec<u32> = tokenizer
        .encode(user_text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
        .get_ids()
        .to_vec();

    let mut prompt: Vec<u32> = vec![TOK_BOS, TOK_TURN_OPEN, TOK_USER, TOK_NEWLINE];
    prompt.push(boi);
    prompt.extend(std::iter::repeat_n(image_token, num_soft_tokens));
    prompt.push(eoi);
    prompt.push(TOK_NEWLINE);
    prompt.extend_from_slice(&text_ids);
    prompt.extend_from_slice(&[
        TOK_TURN_CLOSE,
        TOK_NEWLINE,
        TOK_TURN_OPEN,
        TOK_MODEL,
        TOK_NEWLINE,
    ]);

    eprintln!("[vision] prompt = {} tokens", prompt.len());

    let cfg = GenerateConfig {
        max_new_tokens: std::env::var("MAX_NEW_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(96),
        stop_on_eos: true,
        sampling: None,
    };

    let stats = model
        .generate_with_cache_and_images(&prompt, std::slice::from_ref(&prepared), &cfg, None)
        .context("generate")?;

    let text = tokenizer
        .decode(&stats.generated_tokens, true)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;

    println!("--- prompt ---\n{user_text}\n--- reply ---\n{text}");
    let gb = |v: Result<usize, mlx_rs::error::Exception>| {
        v.map(|b| b as f64 / 1.073_741_824e9).unwrap_or(f64::NAN)
    };
    eprintln!(
        "[vision] {} tokens generated | peak {:.2} GB, active {:.2} GB, cache {:.2} GB",
        stats.generated_tokens.len(),
        gb(lumen_mlx::metal_memory::get_peak_memory()),
        gb(lumen_mlx::metal_memory::get_active_memory()),
        gb(lumen_mlx::metal_memory::get_cache_memory()),
    );
    Ok(())
}

#[cfg(not(feature = "mlx-native"))]
fn main() -> Result<()> {
    anyhow::bail!("this example requires --features mlx-native")
}
