//! Diffusion engine — channel-actor wrapper around the native FLUX.2-dev
//! text-to-image pipeline in `lumen-diffusion`.
//!
//! Mirrors the `EngineHandle` / `EmbeddingService` thread+mpsc pattern used for
//! chat and embeddings: MLX FFI is **not** async-friendly (it drives a Metal
//! command queue and must run on a dedicated OS thread, never a tokio worker),
//! so the heavy generation runs on a blocking worker thread and the async route
//! handler talks to it over an mpsc channel with a oneshot reply.
//!
//! ## Lazy loading
//!
//! The encoder (~13 GB) + DiT/VAE (~18 GB) are loaded **lazily on the first
//! generation request**, not at server startup. This keeps `image`/`hybrid`
//! startup fast and avoids paying the ~30 GB load cost until an image is
//! actually requested. The worker logs `loading diffusion models…` on the first
//! request and reuses the loaded models for all subsequent requests.
//!
//! The full pipeline (matching `examples/flux2_dev_t2i.rs`) is:
//!   tokenize_flux_prompt(prompt) → MistralEncoder.encode → [1,512,15360]
//!     → DevPipeline.generate(embed, size, steps, seed, guidance) → RGB → PNG.

use tokio::sync::{mpsc, oneshot};

/// Parameters for one image generation request handed to the worker thread.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub prompt: String,
    pub width: i32,
    pub height: i32,
    pub steps: usize,
    pub seed: u64,
    pub guidance: f32,
}

/// One generated image: PNG bytes ready for base64 encoding.
pub struct GeneratedImage {
    pub png: Vec<u8>,
}

pub enum DiffusionRequest {
    Generate {
        params: GenerateParams,
        reply: oneshot::Sender<anyhow::Result<GeneratedImage>>,
    },
}

#[derive(Clone)]
pub struct DiffusionHandle {
    tx: mpsc::Sender<DiffusionRequest>,
    /// Image model id surfaced via `/v1/models` and error messages.
    model_id: String,
}

impl DiffusionHandle {
    pub fn new(tx: mpsc::Sender<DiffusionRequest>, model_id: String) -> Self {
        Self { tx, model_id }
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Generate a single image. Returns the PNG bytes or an error string.
    pub async fn generate(&self, params: GenerateParams) -> anyhow::Result<GeneratedImage> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(DiffusionRequest::Generate {
                params,
                reply: reply_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("diffusion service channel closed"))?;
        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("diffusion service dropped reply"))?
    }
}

/// The blocking worker. Holds the lazily-loaded pipeline state and drains the
/// request channel until the sender is dropped. Only compiled with the
/// `mlx-native` feature (the diffusion crate's MLX path); without it the
/// service cannot be constructed and `image`/`hybrid` modes are rejected at
/// startup.
#[cfg(feature = "mlx-native")]
pub struct DiffusionService;

#[cfg(feature = "mlx-native")]
mod imp {
    use super::{DiffusionRequest, GenerateParams, GeneratedImage};
    use anyhow::{Context, Result};
    use lumen_diffusion::dev_pipeline::{
        DevPipeline, ENV_DIT_DIR, ENV_ENCODER_DIR, ENV_VAE_PATH, image_to_png, resolve_path,
    };
    use lumen_diffusion::text_encoder::{MistralEncoder, default_weights_dir};
    use lumen_diffusion::tokenizer::{first_real_index, tokenize_flux_prompt};
    use tokio::sync::mpsc;

    const TXT_IDS_LEN: usize = 512 * 4;

    /// Resolved component paths for one image model.
    struct ComponentDirs {
        dit_dir: String,
        encoder_dir: String,
        vae_path: String,
    }

    /// Resolve the DiT / text-encoder / VAE paths for the active `model_id`.
    ///
    /// Precedence (per component, independently):
    ///   1. explicit env override (`LUMEN_FLUX2_{DIT_DIR,ENCODER_DIR,VAE_PATH}`),
    ///   2. else if `model_id` is the bf16 repo and its snapshot is in the local
    ///      HF cache, that repo's `transformer/` `text_encoder/` `vae/…`,
    ///   3. else the 4-bit MLX defaults (also resolved from the HF cache).
    ///
    /// All HF-cache lookups go through [`lumen_diffusion::hf_cache`] (no
    /// machine-specific paths). Env vars always bypass.
    fn resolve_dirs(model_id: &str) -> ComponentDirs {
        use lumen_diffusion::{hf_cache, repos};

        // bf16 single-repo layout (transformer/ text_encoder/ vae/) when selected.
        let bf16 = (model_id == repos::DEV_BF16)
            .then(|| hf_cache::snapshot_dir(repos::DEV_BF16))
            .flatten();
        let path = |p: &std::path::Path| p.to_string_lossy().into_owned();

        let dit_default = match &bf16 {
            Some(r) => path(&r.join("transformer")),
            None => path(&hf_cache::snapshot_path_or_rel(
                repos::DIT_4BIT,
                "transformer",
            )),
        };
        let enc_default = match &bf16 {
            Some(r) => path(&r.join("text_encoder")),
            None => path(&default_weights_dir()),
        };
        // FLUX.2 ships the VAE as `vae/diffusion_pytorch_model.safetensors`.
        let vae_rel = "vae/diffusion_pytorch_model.safetensors";
        let vae_default = match &bf16 {
            Some(r) => path(&r.join("vae").join("diffusion_pytorch_model.safetensors")),
            None => path(&hf_cache::snapshot_path_or_rel(repos::VAE, vae_rel)),
        };

        ComponentDirs {
            dit_dir: resolve_path(ENV_DIT_DIR, &dit_default),
            encoder_dir: resolve_path(ENV_ENCODER_DIR, &enc_default),
            vae_path: resolve_path(ENV_VAE_PATH, &vae_default),
        }
    }

    /// Models held resident across requests. The text encoder is TRIMMED to the
    /// layers FLUX.2 actually reads (0..=30; layers 31..=40 + lm_head + final
    /// norm are skipped at load), shrinking it ~13 GB → ~10 GB. Combined with
    /// the ~18 GB DiT+VAE that is ~28 GB — within a 36 GB Mac's envelope — so
    /// both can stay hot and repeat requests skip the load entirely (≈denoise
    /// only). If the host has too little memory this load OOM-kills the process;
    /// fall back to a smaller resolution or run on a larger box.
    struct Loaded {
        encoder: MistralEncoder,
        pipeline: Option<DevPipeline>,
        guidance: f32,
        txt_ids: Vec<i32>,
    }

    impl Loaded {
        fn load(model_id: &str) -> Result<Self> {
            let dirs = resolve_dirs(model_id);
            eprintln!(
                "[diffusion] loading models for `{model_id}`\n  \
                 DiT={}\n  encoder={}\n  VAE={}",
                dirs.dit_dir, dirs.encoder_dir, dirs.vae_path
            );
            let t0 = std::time::Instant::now();
            let encoder =
                MistralEncoder::load(&dirs.encoder_dir).context("load Mistral text encoder")?;
            eprintln!(
                "[diffusion] encoder loaded in {:.1}s; loading DiT+VAE…",
                t0.elapsed().as_secs_f32()
            );
            let t1 = std::time::Instant::now();
            let pipeline =
                DevPipeline::load(&dirs.dit_dir, &dirs.vae_path).context("load dev DiT + VAE")?;
            eprintln!(
                "[diffusion] DiT+VAE loaded in {:.1}s (models hot, total {:.1}s)",
                t1.elapsed().as_secs_f32(),
                t0.elapsed().as_secs_f32()
            );
            Ok(Self {
                encoder,
                pipeline: Some(pipeline),
                guidance: lumen_diffusion::dev_pipeline::DEFAULT_GUIDANCE,
                txt_ids: vec![0i32; TXT_IDS_LEN],
            })
        }

        fn generate(&mut self, p: &GenerateParams) -> Result<GeneratedImage> {
            let t0 = std::time::Instant::now();
            let ids = tokenize_flux_prompt(&p.prompt).context("tokenize prompt")?;
            let first_real = first_real_index(&ids);
            let embed = self
                .encoder
                .encode(&ids, first_real)
                .context("encode prompt")?;
            // Re-bind guidance only when it differs (with_guidance consumes self,
            // so take the pipeline out and put it back — no weight reload).
            if (p.guidance - self.guidance).abs() > f32::EPSILON {
                let pipe = self.pipeline.take().expect("pipeline present");
                self.pipeline = Some(pipe.with_guidance(p.guidance));
                self.guidance = p.guidance;
            }
            let pipeline = self.pipeline.as_ref().expect("pipeline present");
            let result = pipeline
                .generate(&embed, &self.txt_ids, p.height, p.width, p.steps, p.seed)
                .context("dev pipeline generate")?;
            let (png, _w, _h) = image_to_png(&result.image).context("encode PNG")?;
            eprintln!(
                "[diffusion] image done in {:.1}s (models stayed hot)",
                t0.elapsed().as_secs_f32()
            );
            Ok(GeneratedImage { png })
        }
    }

    /// Drain the request channel until the sender side is dropped. Loads the
    /// (trimmed-encoder + DiT) models once on the first request and keeps them
    /// hot, so subsequent requests pay only the denoise cost. Dedicated thread.
    pub fn run(mut rx: mpsc::Receiver<DiffusionRequest>, model_id: String) {
        let mut loaded: Option<Loaded> = None;
        while let Some(req) = rx.blocking_recv() {
            match req {
                DiffusionRequest::Generate { params, reply } => {
                    let res = (|| -> Result<GeneratedImage> {
                        if loaded.is_none() {
                            loaded = Some(Loaded::load(&model_id)?);
                        }
                        loaded.as_mut().expect("loaded").generate(&params)
                    })();
                    let _ = reply.send(res);
                }
            }
        }
    }
}

#[cfg(feature = "mlx-native")]
impl DiffusionService {
    /// Spawn the worker on a dedicated OS thread and return its handle.
    /// Loading is deferred to the first request (see module docs).
    pub fn spawn(model_id: String) -> std::io::Result<DiffusionHandle> {
        let (tx, rx) = mpsc::channel(8);
        let worker_model_id = model_id.clone();
        std::thread::Builder::new()
            .name("diffusion-service".into())
            .spawn(move || imp::run(rx, worker_model_id))?;
        Ok(DiffusionHandle::new(tx, model_id))
    }
}
