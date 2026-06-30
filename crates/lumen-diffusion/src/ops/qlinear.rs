//! Quantized affine linear layer for the mlx-lm 4-bit weights.
//!
//! mlx-lm stores each `Linear` as three tensors:
//! - `weight`: `U32`, shape `[out, in / 8]` (8 packed 4-bit values per u32),
//! - `scales`: `BF16`, shape `[out, in / group_size]`,
//! - `biases`: `BF16`, shape `[out, in / group_size]`.
//!
//! [`mlx_rs::ops::quantized_matmul`] computes `x @ dequant(weight)ᵀ` directly
//! from the packed representation when called with `transpose = true`, so we
//! never materialize the dense weight. This is the Mistral text-encoder hot
//! path (every q/k/v/o and gate/up/down projection across 40 layers).

#[cfg(feature = "mlx-native")]
mod imp {
    use anyhow::{Context, Result};
    use mlx_rs::Array;

    /// mlx-lm affine quantization parameters for the FLUX.2 weights.
    pub const GROUP_SIZE: i32 = 64;
    pub const BITS: i32 = 4;

    /// A quantized linear's three packed tensors (kept in their native packed
    /// layout; never dequantized to a dense matrix).
    pub struct QuantLinear {
        /// `U32` packed weight, `[out, in / 8]`.
        pub weight: Array,
        /// `BF16` scales, `[out, in / group_size]`.
        pub scales: Array,
        /// `BF16` biases, `[out, in / group_size]`.
        pub biases: Array,
    }

    /// `y = x @ dequant(qw.weight)ᵀ` (no bias term — Mistral projections are
    /// bias-free). `x`: `[.., in]` → `y`: `[.., out]`.
    pub fn qlinear(x: &Array, qw: &QuantLinear) -> Result<Array> {
        mlx_rs::ops::quantized_matmul(
            x, &qw.weight, &qw.scales, &qw.biases, true, /* transpose: compute x @ Wᵀ */
            GROUP_SIZE, BITS,
        )
        .context("qlinear: quantized_matmul FFI failed")
    }
}

#[cfg(feature = "mlx-native")]
pub use imp::{BITS, GROUP_SIZE, QuantLinear, qlinear};
