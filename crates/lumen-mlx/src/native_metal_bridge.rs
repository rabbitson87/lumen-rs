//! canonical custom-kernel path for Gemma 4 decode.
//!
//! All Gemma-4-specific Metal kernels now live in mlx core as registered
//! `mlx::core::Primitive` subclasses (see `mlx/lumen_primitives.cpp` and
//! `mlx/backend/metal/lumen_flash_attn.cpp`). This module is the thin
//! Rust entry-point: each public function constructs a lazy mlx Array node
//! that fires the primitive's `eval_gpu` when the surrounding graph is
//! evaluated.
//!
//! Earlier milestones (M1, M1.5, M2.0-M2.6, M4.6, M4.7) used a bridge-
//! dispatch pattern that pulled mlx's `MTL::Buffer` via mlx-c extensions,
//! encoded our own Metal pipelines, and wrapped the output back as an mlx
//! Array. That pattern was correct but incurred per-call sync cost (M4.6
//! +30 ms regression, M4.7 +12 ms) and was retired when M4.8 demonstrated
//! that primitive registration matches `mlx::fast::sdpa` perf with zero
//! per-call overhead. The mlx-c bridge extensions (`_mlx_array_metal_buffer`,
//! `_mlx_metal_allocator_*`, `_mlx_array_new_from_metal_buffer`,
//! `_mlx_metal_current_command_buffer`) are kept in mlx-rs/mlx-sys for any
//! future kernel that genuinely needs them, but no lumen-rs production
//! path uses them today.

#![cfg(feature = "mlx-native")]

use anyhow::{Result, anyhow};
use mlx_rs::Array;

/// M4.8 — bf16 flash-attention via the registered `lumen_flash_attn_bf16`
/// mlx Primitive. Bit-identical to `mlx::fast::scaled_dot_product_attention`
/// for the bf16 / head_dim=256 / sliding-attention path Gemma 4 uses
/// (max|Δ|=1.95e-3, cos=0.999999).
///
/// Contract:
///   - `q`: `[B, H, Sq, 256]` bf16
///   - `k, v`: `[B, H_kv, Skv, 256]` bf16 (`H = H_kv * group`)
///   - `mask`: optional `[Sq, Skv]` bf16 additive bias
///   - output: `[B, H, Sq, 256]` bf16
pub fn run_flash_attn_bf16(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    mask: Option<&Array>,
) -> Result<Array> {
    let stream = mlx_rs::Stream::gpu();
    mlx_rs::metal::lumen_flash_attn_bf16(q, k, v, mask, scale, &stream)
        .map_err(|e| anyhow!("lumen_flash_attn_bf16: {e}"))
}
