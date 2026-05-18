//! Qwen3.5-VL-MoE model support (hybrid Mamba2 + GQA + 256-expert MoE + ViT).
//!
//! Upstream reference architecture: `Qwen3_5MoeForConditionalGeneration` in
//! HuggingFace transformers v4.57+.
//!
//! Submodules land incrementally per the Phase 1.5 plan in
//! [`docs/plans/qwen3_5-moe-mxfp4-adoption.md`]. Currently only [`config`] is populated; the
//! router, experts, attention paths, and model forward come in subsequent commits.

#[cfg(feature = "turboquant-gpu")]
pub mod backend;
pub mod config;
pub mod kv_cache_ring;
pub mod layer;
pub mod linear_attn;
pub mod loader;
pub mod model;
pub mod moe;
pub mod mtp;
pub mod mtp_drafter;
#[cfg(feature = "mpsgraph")]
pub(crate) mod mpsgraph_norm;
pub mod proj;
pub mod self_attn;
pub mod weights;
