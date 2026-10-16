//! Native MLX normalization wrappers.
//!
//! Currently exposes `rms_norm` (the only normalization the Qwen3.5-MoE
//! architecture uses for both pre-attention and pre-MLP norms). Routed
//! through `mlx_rs::fast::rms_norm`, which is itself a thin wrapper over
//! `mlx_fast_rms_norm`. The op is deterministic for fixed inputs, so a
//! correct binding produces bit-identical output vs MLX's Python reference.
//!
//! See `.ai/memory/active/mlx-rs-native-port/CONTEXT.md` Session 21+.

#[cfg(feature = "mlx-native")]
mod imp {
    use anyhow::{Context, Result};
    use mlx_rs::Array;

    /// RMSNorm along the last axis: `y = x * rsqrt(mean(x*x, axis=-1) + eps) * weight`.
    pub fn rms_norm(x: &Array, weight: &Array, eps: f32) -> Result<Array> {
        mlx_rs::fast::rms_norm(x, weight, eps).context("mlx-rs fast::rms_norm FFI call failed")
    }
}

#[cfg(feature = "mlx-native")]
#[allow(unused_imports)] // Consumed by Phase 3b model assembly in runner_native.rs.
pub(crate) use imp::rms_norm;
