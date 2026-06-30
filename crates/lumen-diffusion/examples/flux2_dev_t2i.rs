//! FLUX.2-dev text-to-image end-to-end generation.
//!
//! Loads the dev-32B DiT (mlx affine 4-bit), the shared VAE, and the validated
//! Mistral text embedding, then runs the full denoise loop and writes a PNG.
//!
//! Build + run (Metal GPU required — DO NOT run under the sandbox):
//! ```text
//! cargo build -p lumen-diffusion --features mlx-native-metal --example flux2_dev_t2i
//! LUMEN_FLUX2_STEPS=28 LUMEN_FLUX2_SIZE=512 \
//!   target/debug/examples/flux2_dev_t2i
//! ```
//!
//! Env knobs (all optional):
//! - `LUMEN_FLUX2_SIZE`     image side, default 512.
//! - `LUMEN_FLUX2_STEPS`    denoise steps, default 28.
//! - `LUMEN_FLUX2_SEED`     latent RNG seed, default 0.
//! - `LUMEN_FLUX2_GUIDANCE` guidance scalar, default 4.0.
//! - `LUMEN_FLUX2_OUT`      PNG path, default /tmp/flux2_dev_out.png.
//! - `LUMEN_FLUX2_PROMPT`   arbitrary text prompt. When set it takes priority:
//!   the prompt is tokenized live (FLUX.2 template → Tekken ids, left-padded to
//!   512) and run through the live Mistral encoder. Example:
//!   `LUMEN_FLUX2_PROMPT="a red fox in snow" target/debug/examples/flux2_dev_t2i`.
//! - `LUMEN_FLUX2_EMBED`    text-embed .bin override; used only when
//!   `LUMEN_FLUX2_PROMPT` is unset. If it (or the default
//!   `/tmp/mistral_embed_ref.bin`) exists it is reused, otherwise the live
//!   Mistral encoder runs on `/tmp/mistral_tokens.bin`.

#[cfg(feature = "mlx-native")]
fn main() -> anyhow::Result<()> {
    use lumen_diffusion::dev_pipeline::{
        DevPipeline, ENV_DIT_DIR, ENV_ENCODER_DIR, ENV_VAE_PATH, image_to_rgb8, resolve_path,
    };
    use mlx_rs::Array;

    // Default 4-bit MLX repos, resolved from the local HuggingFace cache;
    // override with LUMEN_FLUX2_{DIT_DIR,ENCODER_DIR,VAE_PATH} to point at the
    // bf16 `black-forest-labs/FLUX.2-dev` repo subdirs.
    use lumen_diffusion::{hf_cache, repos};
    let dit_default = hf_cache::snapshot_path_or_rel(repos::DIT_4BIT, "transformer");
    let vae_default =
        hf_cache::snapshot_path_or_rel(repos::VAE, "vae/diffusion_pytorch_model.safetensors");
    let dit_dir = resolve_path(ENV_DIT_DIR, &dit_default.to_string_lossy());
    let vae_path = resolve_path(ENV_VAE_PATH, &vae_default.to_string_lossy());
    const TXT_SEQ: i32 = 512;
    const JOINT_DIM: i32 = 15360;

    fn env_usize(k: &str, d: usize) -> usize {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    }
    fn env_f32(k: &str, d: f32) -> f32 {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    }
    fn read_i32_bin(path: &str) -> anyhow::Result<Vec<i32>> {
        let bytes = std::fs::read(path)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
    fn read_f32_bin(path: &str) -> anyhow::Result<Vec<f32>> {
        let bytes = std::fs::read(path)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    let size = env_usize("LUMEN_FLUX2_SIZE", 512) as i32;
    let steps = env_usize("LUMEN_FLUX2_STEPS", 28);
    let seed = env_usize("LUMEN_FLUX2_SEED", 0) as u64;
    let guidance = env_f32("LUMEN_FLUX2_GUIDANCE", 4.0);
    let out_path =
        std::env::var("LUMEN_FLUX2_OUT").unwrap_or_else(|_| "/tmp/flux2_dev_out.png".to_string());

    // ── text embedding selection (priority order) ──
    //  1. LUMEN_FLUX2_PROMPT set → live tokenize arbitrary prompt + encode.
    //  2. dumped embedding exists (LUMEN_FLUX2_EMBED / default) → reuse fast-path.
    //  3. otherwise → run the live encoder on the dumped /tmp/mistral_tokens.bin.
    let embed_path = std::env::var("LUMEN_FLUX2_EMBED")
        .unwrap_or_else(|_| "/tmp/mistral_embed_ref.bin".to_string());
    // Encoder dir: env override (bf16 repo) else the default 4-bit snapshot.
    let encoder_dir = resolve_path(
        ENV_ENCODER_DIR,
        &lumen_diffusion::text_encoder::default_weights_dir().to_string_lossy(),
    );
    let prompt_embeds = if let Ok(prompt) = std::env::var("LUMEN_FLUX2_PROMPT") {
        // Live path: arbitrary prompt → FLUX template → token ids → Mistral encoder.
        use lumen_diffusion::text_encoder::MistralEncoder;
        use lumen_diffusion::tokenizer::{first_real_index, tokenize_flux_prompt};
        println!("[embed] live tokenize + encode prompt: {prompt:?}");
        let ids = tokenize_flux_prompt(&prompt)?;
        let first_real = first_real_index(&ids);
        println!(
            "[embed] {} real tokens (first_real={first_real}), padded to {}",
            ids.len() - first_real,
            ids.len()
        );
        let enc = MistralEncoder::load(&encoder_dir)?;
        enc.encode(&ids, first_real)?
    } else if std::path::Path::new(&embed_path).exists() {
        println!("[embed] reusing dumped text embedding: {embed_path}");
        let data = read_f32_bin(&embed_path)?;
        let arr = Array::from_slice(&data, &[1, TXT_SEQ, JOINT_DIM]);
        arr.eval()?;
        arr
    } else {
        println!("[embed] running live Mistral encoder on /tmp/mistral_tokens.bin");
        use lumen_diffusion::text_encoder::MistralEncoder;
        let ids = read_i32_bin("/tmp/mistral_tokens.bin")?;
        let first_real = read_i32_bin("/tmp/mistral_firstreal.bin")?[0] as usize;
        let enc = MistralEncoder::load(&encoder_dir)?;
        enc.encode(&ids, first_real)?
    };
    let txt_ids = vec![0i32; (TXT_SEQ * 4) as usize];

    println!("[gen] size={size} steps={steps} seed={seed} guidance={guidance}");
    let t0 = std::time::Instant::now();
    let pipe = DevPipeline::load(&dit_dir, &vae_path)?.with_guidance(guidance);
    println!(
        "[gen] loaded DiT + VAE in {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    let t1 = std::time::Instant::now();
    let result = pipe.generate(&prompt_embeds, &txt_ids, size, size, steps, seed)?;
    println!("[gen] denoise+decode in {:.1}s", t1.elapsed().as_secs_f32());

    // stats
    let img: &[f32] = result.image.as_slice();
    let (mut mn, mut mx, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
    for &v in img {
        mn = mn.min(v);
        mx = mx.max(v);
        sum += v as f64;
    }
    let mean = sum / img.len() as f64;
    println!(
        "[stats] shape={:?} min={mn:.4} max={mx:.4} mean={mean:.4}",
        result.image.shape()
    );

    let (rgb, w, h) = image_to_rgb8(&result.image)?;
    let buf = image::RgbImage::from_raw(w, h, rgb).expect("rgb buffer size");
    buf.save(&out_path)?;
    println!("[done] wrote {out_path} ({w}x{h})");
    Ok(())
}

#[cfg(not(feature = "mlx-native"))]
fn main() {
    eprintln!("flux2_dev_t2i requires --features mlx-native-metal");
}
