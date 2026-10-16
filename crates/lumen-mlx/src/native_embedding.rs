//! Native MLX embedding lookup.
//!
//! Two paths exist in production:
//! * **Plain f32 / bf16 / f16 weight** — plain `mx.take(weight, ids, axis=0)`,
//!   used when the embedding table is stored unquantized.
//! * **MXFP4-quantized weight** — the production Qwen3.6-35B-A3B-mxfp4
//!   `language_model.model.embed_tokens` is MXFP4. mlx-rs's stock
//!   `nn::QuantizedEmbedding::forward` calls `dequantize` *without* a mode
//!   parameter (hardcodes `affine`), so it would silently corrupt MXFP4
//!   weights. We replicate the upstream forward path manually but route
//!   through `dequantize_with_mode(..., MODE_MXFP4)` instead.
//!
//! See `.ai/memory/active/mlx-rs-native-port/CONTEXT.md` Session 21 for
//! Phase 3b's overall plan.

#[cfg(feature = "mlx-native")]
#[allow(dead_code)] // `embedding_lookup` (plain f32 path) is held alongside `quantized_embedding_lookup_with_mode` for future plain-embedding callsites
mod imp {
    use anyhow::{Context, Result};
    use mlx_rs::Array;
    use std::ffi::CStr;

    use crate::native_quant::dequantize_with_mode;

    /// Plain embedding lookup: `weight[token_ids]` along axis 0.
    ///
    /// `weight`: `[vocab, hidden]` of any float dtype mlx-rs supports.
    /// `token_ids`: `[..., n_tokens]` of int (i32 expected for MLX `take`).
    /// Returns: `[..., n_tokens, hidden]`.
    pub fn embedding_lookup(weight: &Array, token_ids: &Array) -> Result<Array> {
        weight
            .take_axis(token_ids, 0)
            .context("mlx-rs take_axis (plain embedding lookup) failed")
    }

    /// MXFP4-quantized embedding lookup. Mirrors `nn::QuantizedEmbedding::forward`
    /// row-selection-then-dequantize pattern but with explicit `mode="mxfp4"`.
    ///
    /// `packed`: `[vocab, hidden / 8]` u32 (MXFP4 packed nibbles).
    /// `scales`: `[vocab, hidden / 32]` u8 (E8M0 shared exponents).
    /// `token_ids`: `[..., n_tokens]` of int.
    /// Returns: `[..., n_tokens, hidden]` in `bf16` (MLX's mxfp4 dequant
    /// default — caller can `.as_dtype(...)` if needed).
    ///
    /// Indexing the *small* selected slice (rather than dequantizing the full
    /// table) keeps memory and bandwidth proportional to the active token count.
    pub fn quantized_embedding_lookup_with_mode(
        packed: &Array,
        scales: &Array,
        biases: Option<&Array>,
        token_ids: &Array,
        group_size: i32,
        bits: i32,
        mode: &CStr,
    ) -> Result<Array> {
        let selected_packed = packed
            .take_axis(token_ids, 0)
            .context("mlx-rs take_axis (packed rows) failed")?;
        let selected_scales = scales
            .take_axis(token_ids, 0)
            .context("mlx-rs take_axis (scales rows) failed")?;
        // Affine dequant needs per-row biases (3-tensor); mxfp4/nvfp4 pass None.
        let selected_biases = match biases {
            Some(b) => Some(
                b.take_axis(token_ids, 0)
                    .context("mlx-rs take_axis (biases rows) failed")?,
            ),
            None => None,
        };
        dequantize_with_mode(
            &selected_packed,
            &selected_scales,
            selected_biases.as_ref(),
            group_size,
            bits,
            mode,
        )
    }
}

#[cfg(feature = "mlx-native")]
#[allow(unused_imports)] // Consumed by Phase 3b model assembly in runner_native.rs.
pub(crate) use imp::{embedding_lookup, quantized_embedding_lookup_with_mode};
