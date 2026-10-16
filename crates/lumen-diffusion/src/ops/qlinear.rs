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

    use crate::ops::linear_nobias::linear_nobias;

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

    impl QuantLinear {
        /// Evaluate the three packed tensors.
        pub fn eval(&self) -> Result<()> {
            for a in [&self.weight, &self.scales, &self.biases] {
                a.eval().context("qlinear: eval packed tensor failed")?;
            }
            Ok(())
        }
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

    /// A bias-free projection as the checkpoint stores it: mlx affine 4-bit
    /// packed (the dev DiT, the 4-bit text encoder) or dense (klein, the bf16
    /// encoder). [`Linear::apply`] hides the difference, so a forward pass is
    /// the same code for both.
    pub enum Linear {
        /// Dense `[in, out]` (already transposed at load) — `matmul(x, w)`.
        Dense(Array),
        /// mlx affine 4-bit packed `{weight,scales,biases}`.
        Quant(QuantLinear),
    }

    impl Linear {
        /// `x @ Wᵀ`: `qlinear` for a packed weight, `linear_nobias` for a
        /// dense one.
        #[inline]
        pub fn apply(&self, x: &Array) -> Result<Array> {
            match self {
                Linear::Dense(w) => linear_nobias(x, w),
                Linear::Quant(q) => qlinear(x, q),
            }
        }
    }
}

#[cfg(feature = "mlx-native")]
pub use imp::{BITS, GROUP_SIZE, Linear, QuantLinear, qlinear};
