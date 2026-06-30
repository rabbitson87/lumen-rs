//! FLUX.2-dev text-to-image denoise loop (pipeline glue).
//!
//! Mirrors [`crate::pipeline::KleinPipeline`] but targets the dev-32B
//! checkpoint: an mlx affine 4-bit DiT with a guidance branch. The structure is
//! identical — DiT noise prediction → flow-match Euler step → VAE
//! `decode_packed_latents` — the only differences are:
//!
//! - the DiT loads quantized (rather than dense) linears via [`crate::dit`],
//! - the DiT forward is passed a `guidance` scalar (default 4.0),
//! - this module owns the random-latent creation (seed → `[1,128,lh,lw]` →
//!   packed `[1, lh*lw, 128]`) and the `img_ids` grid, mirroring mflux's
//!   `Flux2LatentCreator`.
//!
//! The caller supplies the prompt embedding `[1, txt_seq, 15360]` (from the
//! validated Mistral encoder) and `txt_ids` (zeros `[txt_seq, 4]`).

#[cfg(feature = "mlx-native")]
mod imp {
    use anyhow::{Context, Result, bail};
    use mlx_rs::Array;

    use crate::dit::{Dit, DitConfig};
    use crate::scheduler;
    use crate::vae::VaeDecoder;

    /// Default classifier-free guidance scalar for dev (mflux default).
    pub const DEFAULT_GUIDANCE: f32 = 4.0;

    /// Loaded dev pipeline: quantized DiT + VAE decoder.
    pub struct DevPipeline {
        dit: Dit,
        vae: VaeDecoder,
        guidance: f32,
    }

    /// Result of `generate`: the final decoded image plus its packed-latent
    /// grid dims and a few array stats (for the no-reference gate).
    pub struct DevImage {
        /// Decoded image `[1, H, W, 3]` channels-last, values ~[-1, 1].
        pub image: Array,
        pub lat_h: i32,
        pub lat_w: i32,
    }

    impl DevPipeline {
        /// Load the dev DiT (directory of shards) + the shared VAE.
        pub fn load(dit_dir: &str, vae_path: &str) -> Result<Self> {
            let dit = Dit::load(dit_dir, DitConfig::dev_32b()).context("load dev DiT")?;
            let vae = VaeDecoder::load(vae_path).context("load VAE")?;
            Ok(Self {
                dit,
                vae,
                guidance: DEFAULT_GUIDANCE,
            })
        }

        /// Override the guidance scalar (default [`DEFAULT_GUIDANCE`]).
        pub fn with_guidance(mut self, guidance: f32) -> Self {
            self.guidance = guidance;
            self
        }

        /// Build the initial packed latents + `img_ids` for an `height×width`
        /// image, mirroring mflux `Flux2LatentCreator::prepare_packed_latents`.
        ///
        /// Returns `(packed_latents [1, lh*lw, 128], img_ids [lh*lw*4],
        /// lat_h, lat_w)`. `lat_h = lat_w = dim/16` (vae_scale_factor 8 × patch 2).
        fn prepare_latents(
            &self,
            seed: u64,
            height: i32,
            width: i32,
        ) -> Result<(Array, Vec<i32>, i32, i32)> {
            // height/width rounded to the 16-multiple grid, then /2 patch + /8 vae.
            let h = 2 * (height / 16);
            let w = 2 * (width / 16);
            let lat_h = h / 2;
            let lat_w = w / 2;

            // [1, 128, lat_h, lat_w] standard normal (32 channels × 4 patch = 128).
            let key = mlx_rs::random::key(seed).context("random key")?;
            let latents = mlx_rs::random::normal::<f32>(&[1, 128, lat_h, lat_w], 0.0, 1.0, &key)
                .context("random normal latents")?;

            // pack: [1,128,lh,lw] -> [1,128,lh*lw] -> [1,lh*lw,128].
            let packed = latents
                .reshape(&[1, 128, lat_h * lat_w])
                .and_then(|x| x.transpose_axes(&[0, 2, 1]))
                .context("pack latents")?;
            packed.eval().context("eval packed latents")?;

            // img_ids: [lh*lw, 4] = [t=0, row, col, layer=0] row-major.
            let mut img_ids = Vec::with_capacity((lat_h * lat_w * 4) as usize);
            for row in 0..lat_h {
                for col in 0..lat_w {
                    img_ids.push(0);
                    img_ids.push(row);
                    img_ids.push(col);
                    img_ids.push(0);
                }
            }
            Ok((packed, img_ids, lat_h, lat_w))
        }

        /// Full text-to-image generation.
        ///
        /// - `prompt_embeds`: `[1, txt_seq, 15360]` (Mistral encoder output).
        /// - `txt_ids`: row-major `[txt_seq, 4]` i32 (zeros for text positions).
        /// - `height`/`width`: target image size (multiple of 16).
        /// - `steps`: denoise steps.
        /// - `seed`: latent RNG seed.
        pub fn generate(
            &self,
            prompt_embeds: &Array,
            txt_ids: &[i32],
            height: i32,
            width: i32,
            steps: usize,
            seed: u64,
        ) -> Result<DevImage> {
            if steps == 0 {
                bail!("steps must be >= 1");
            }
            let (mut latents, img_ids, lat_h, lat_w) = self.prepare_latents(seed, height, width)?;
            let img_seq = (lat_h * lat_w) as usize;

            let sched = scheduler::compute(img_seq, steps);

            for t in 0..steps {
                let timestep = sched.timesteps[t]; // already ×1000-scaled
                let noise = self
                    .dit
                    .forward(
                        &latents,
                        prompt_embeds,
                        timestep,
                        Some(self.guidance),
                        &img_ids,
                        txt_ids,
                    )
                    .with_context(|| format!("dev DiT forward step {t}"))?;

                // Euler: latents += (sigmas[t+1]-sigmas[t]) * noise.
                let dt = sched.dt(t);
                let dt_arr = Array::from_slice(&[dt], &[1]);
                latents = noise
                    .multiply(&dt_arr)
                    .and_then(|d| latents.add(&d))
                    .with_context(|| format!("euler step {t}"))?;
                latents
                    .eval()
                    .with_context(|| format!("eval latents step {t}"))?;
            }

            // Pack → channels-first [1,128,lh,lw] → VAE decode.
            let packed_cf = latents
                .reshape(&[1, lat_h, lat_w, 128])
                .and_then(|x| x.transpose_axes(&[0, 3, 1, 2]))
                .context("final pack reshape")?;
            let image = self
                .vae
                .decode_packed_latents(&packed_cf)
                .context("decode_packed_latents")?;
            image.eval().context("eval final image")?;

            Ok(DevImage {
                image,
                lat_h,
                lat_w,
            })
        }
    }

    /// Encode a decoded image `[1, H, W, 3]` directly to PNG bytes (in-memory).
    /// Convenience for servers that need a byte buffer rather than a file —
    /// runs [`image_to_rgb8`] then PNG-encodes via the `image` crate.
    pub fn image_to_png(image: &Array) -> Result<(Vec<u8>, u32, u32)> {
        use image::ImageEncoder;
        let (rgb, w, h) = image_to_rgb8(image)?;
        let buf = image::RgbImage::from_raw(w, h, rgb)
            .context("construct RGB image buffer (size mismatch)")?;
        let mut png: Vec<u8> = Vec::new();
        image::codecs::png::PngEncoder::new(std::io::Cursor::new(&mut png))
            .write_image(buf.as_raw(), w, h, image::ExtendedColorType::Rgb8)
            .context("PNG encode")?;
        Ok((png, w, h))
    }

    /// Map a decoded image `[1, H, W, 3]` in ~[-1,1] to interleaved RGB u8
    /// `(H*W*3)`, applying `(x*0.5+0.5)*255` clamped to `[0,255]`.
    pub fn image_to_rgb8(image: &Array) -> Result<(Vec<u8>, u32, u32)> {
        let sh = image.shape().to_vec();
        let (h, w) = match sh.as_slice() {
            // channels-last [1,H,W,3]
            [1, h, w, 3] => (*h, *w),
            other => bail!("image_to_rgb8 expects [1,H,W,3], got {other:?}"),
        };
        image.eval().context("eval image for rgb8")?;
        let data: &[f32] = image.as_slice();
        let mut out = Vec::with_capacity((h * w * 3) as usize);
        for &v in data.iter() {
            let p = ((v * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
            out.push(p);
        }
        Ok((out, w as u32, h as u32))
    }
}

// ── env-overridable component paths ──────────────────────────────────────────
//
// The DiT dir, text-encoder dir and VAE path default to the 4-bit MLX repos but
// can be redirected to the official bf16 `black-forest-labs/FLUX.2-dev` repo
// subdirs (`<repo>/transformer`, `<repo>/text_encoder`, `<repo>/vae`) for full
// precision on 128 GB machines. Set:
//   - `LUMEN_FLUX2_DIT_DIR`     → transformer shard directory.
//   - `LUMEN_FLUX2_ENCODER_DIR` → text-encoder shard directory.
//   - `LUMEN_FLUX2_VAE_PATH`    → VAE safetensors file.
// Each resolves to the env value if set, else the given 4-bit default.

/// Env var for the DiT (transformer) shard directory override.
pub const ENV_DIT_DIR: &str = "LUMEN_FLUX2_DIT_DIR";
/// Env var for the text-encoder shard directory override.
pub const ENV_ENCODER_DIR: &str = "LUMEN_FLUX2_ENCODER_DIR";
/// Env var for the VAE safetensors path override.
pub const ENV_VAE_PATH: &str = "LUMEN_FLUX2_VAE_PATH";

/// Resolve a component path: the env override if set & non-empty, else `default`.
pub fn resolve_path(env_key: &str, default: &str) -> String {
    match std::env::var(env_key) {
        Ok(v) if !v.trim().is_empty() => v,
        _ => default.to_string(),
    }
}

#[cfg(feature = "mlx-native")]
pub use imp::{DEFAULT_GUIDANCE, DevImage, DevPipeline, image_to_png, image_to_rgb8};

// ── minimal end-to-end smoke test (no numerical reference) ───────────────────
#[cfg(all(test, feature = "mlx-native"))]
mod smoke_tests {
    use super::imp::{DevPipeline, image_to_rgb8};
    use mlx_rs::Array;

    // 4-bit dev DiT + shared VAE, resolved from the local HF cache.
    fn dit_dir() -> String {
        crate::hf_cache::snapshot_path_or_rel(crate::repos::DIT_4BIT, "transformer")
            .to_string_lossy()
            .into_owned()
    }
    fn vae_path() -> String {
        crate::hf_cache::snapshot_path_or_rel(
            crate::repos::VAE,
            "vae/diffusion_pytorch_model.safetensors",
        )
        .to_string_lossy()
        .into_owned()
    }

    fn read_f32_bin(path: &str) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Minimal e2e: 256×256, 4 steps, reusing the validated text embedding.
    /// Gate = runs without error, all weights load, output is finite and
    /// non-degenerate, and a PNG is written.
    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device; 32B forward is slow"]
    fn dev_pipeline_minimal_e2e() {
        const TXT_SEQ: i32 = 512;
        const JOINT_DIM: i32 = 15360;

        let embed = read_f32_bin("/tmp/mistral_embed_ref.bin");
        assert_eq!(embed.len(), (TXT_SEQ * JOINT_DIM) as usize, "embed size");
        let prompt_embeds = Array::from_slice(&embed, &[1, TXT_SEQ, JOINT_DIM]);
        prompt_embeds.eval().unwrap();
        let txt_ids = vec![0i32; (TXT_SEQ * 4) as usize];

        let pipe = DevPipeline::load(&dit_dir(), &vae_path()).expect("load dev pipeline");
        let result = pipe
            .generate(&prompt_embeds, &txt_ids, 256, 256, 4, 0)
            .expect("generate");

        assert_eq!(result.image.shape(), &[1, 256, 256, 3], "image shape");
        let img: &[f32] = result.image.as_slice();
        let (mut mn, mut mx, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
        for &v in img {
            assert!(v.is_finite(), "non-finite pixel: {v}");
            mn = mn.min(v);
            mx = mx.max(v);
            sum += v as f64;
        }
        let mean = sum / img.len() as f64;
        println!("dev minimal e2e: shape=[1,256,256,3] min={mn:.4} max={mx:.4} mean={mean:.4}");
        // non-degenerate: must have real spread, not a flat fill.
        assert!(mx - mn > 0.05, "degenerate image (range {:.4})", mx - mn);

        let (rgb, w, h) = image_to_rgb8(&result.image).expect("rgb8");
        let buf = image::RgbImage::from_raw(w, h, rgb).expect("rgb buffer");
        buf.save("/tmp/flux2_dev_minimal.png").expect("save png");
        println!("dev minimal e2e: wrote /tmp/flux2_dev_minimal.png");
    }
}
