//! Native MLX Conv1d wrapper (depthwise + dense).
//!
//! The Qwen3.5-MoE linear-attention block uses a depthwise Conv1d
//! (groups == in_channels == out_channels) with kernel_size=4, no padding,
//! followed by SiLU activation. This module wraps `mlx_rs::ops::conv1d`
//! for both the dense and depthwise regimes.
//!
//! MLX layout matches `nn.Conv1d`:
//!     array:   [batch, length, in_channels]
//!     weight:  [out_channels, kernel_size, in_channels / groups]
//!     output:  [batch, out_length, out_channels]
//!
//! See `.ai/memory/active/mlx-rs-native-port/CONTEXT.md` Session 22+.

#[cfg(feature = "mlx-native")]
mod imp {
    use anyhow::{Context, Result};
    use mlx_rs::Array;

    /// 1D convolution. `groups == in_channels` selects the depthwise regime.
    pub fn conv1d(
        x: &Array,
        weight: &Array,
        stride: i32,
        padding: i32,
        groups: i32,
    ) -> Result<Array> {
        mlx_rs::ops::conv1d(x, weight, stride, padding, /* dilation */ 1, groups)
            .context("mlx-rs ops::conv1d FFI call failed")
    }
}

#[cfg(feature = "mlx-native")]
#[allow(unused_imports)] // Consumed by Phase 3b.5 SSM block assembly.
pub(crate) use imp::conv1d;
