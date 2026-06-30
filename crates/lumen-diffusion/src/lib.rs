//! `lumen-diffusion` — native FLUX.2 text-to-image for lumen-rs.
//!
//! A standalone subsystem for image generation, kept separate from the
//! autoregressive LLM serving path (`lumen-mlx` / `lumen-model`). Diffusion is
//! iterative denoising over a fixed latent and shares none of the KV-cache /
//! MTP / sampling machinery — only the low-level MLX primitives.
//!
//! ## Pipeline
//!
//! `tokenize_flux_prompt` → [`text_encoder`] (Mistral-Small-3.2, layers 10/20/30
//! stacked → `[1,512,15360]`) → [`dit`] (32B rectified-flow: 8 double + 48 single
//! stream, 4D RoPE, AdaLN) → [`scheduler`] (flow-match Euler) → [`vae`] decode →
//! RGB PNG. See [`dev_pipeline`] for the assembled FLUX.2-dev generator.
//!
//! ## Modules
//!
//! - [`ops`] — net-new MLX ops: Conv2d, GroupNorm, 4D RoPE, SiLU, q/dense linear.
//! - [`vae`] — conv decoder (latent → image).
//! - [`text_encoder`] — Mistral-Small-3.2, text-only (4-bit or bf16, auto-detected).
//! - [`dit`] — double/single-stream rectified-flow transformer (4-bit or bf16).
//! - [`scheduler`] / [`pipeline`] / [`dev_pipeline`] — denoise loop + orchestration.
//! - [`tokenizer`] — FLUX.2 prompt template + Tekken tokenization.
//! - [`hf_cache`] — resolve component repos from the local HuggingFace cache.
//!
//! All MLX-touching code is gated behind the `mlx-native` feature so the crate
//! compiles (as an empty shell) without the heavy mlx-sys fork.

pub mod ops;

/// Resolve FLUX.2 component repos from the local HuggingFace cache (no machine-
/// specific paths). Used as the default for the encoder / DiT / VAE / tokenizer.
pub mod hf_cache;

/// HuggingFace repo ids for the FLUX.2-dev components. The pipeline resolves
/// their local snapshot dirs via [`hf_cache`]; all are overridable by env.
pub mod repos {
    /// 4-bit MLX text encoder + tokenizer (Mistral-Small-3.2-24B).
    pub const ENCODER_4BIT: &str = "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit";
    /// 4-bit MLX DiT + VAE (FLUX.2-dev).
    pub const DIT_4BIT: &str = "AITRADER/FLUX2-dev-mlx-4bit";
    /// Shared VAE (klein/dev are the same FLUX.2 VAE).
    pub const VAE: &str = "black-forest-labs/FLUX.2-klein-4B";
    /// Official non-quantized (bf16) FLUX.2-dev — single repo, all components.
    pub const DEV_BF16: &str = "black-forest-labs/FLUX.2-dev";
}

// Prompt tokenization is pure-CPU (Mistral/Tekken via the `tokenizers` crate);
// always compiled so arbitrary prompts work without the MLX/Metal feature.
pub mod tokenizer;

#[cfg(feature = "mlx-native")]
pub mod vae;

#[cfg(feature = "mlx-native")]
pub mod text_encoder;

#[cfg(feature = "mlx-native")]
pub mod dit;

// Scheduler is pure-scalar f32 (no MLX) — always compiled.
pub mod scheduler;

#[cfg(feature = "mlx-native")]
pub mod pipeline;

#[cfg(feature = "mlx-native")]
pub mod dev_pipeline;
