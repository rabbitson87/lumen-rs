//! Native MLX Conv2d wrapper for the FLUX.2 VAE.
//!
//! The autoregressive LLM path only ever needed a depthwise Conv1d
//! (`lumen-mlx::native_conv1d`); the diffusion VAE decoder is conv-heavy and
//! needs full 2D convolution. This module wraps `mlx_rs::ops::conv2d` from the
//! patched fork — confirmed present alongside `conv_transpose2d`, so the VAE can
//! be built natively without a matmul-as-conv fallback.
//!
//! MLX layout matches `nn.Conv2d` (channels-last):
//!     array:   [batch, H, W, in_channels]
//!     weight:  [out_channels, kH, kW, in_channels / groups]
//!     output:  [batch, out_H, out_W, out_channels]
//!
//! Note MLX convolution is *cross-correlation* (no kernel flip), matching
//! PyTorch / `mlx.nn.Conv2d` semantics — the VAE weights are trained against
//! that convention, so no flip is applied here.
//!
//! Design docs: `.ai/memory/active/flux2-dev-diffusion-port/`.

#[cfg(feature = "mlx-native")]
mod imp {
    use anyhow::{Context, Result};
    use mlx_rs::Array;

    /// 2D convolution (cross-correlation).
    ///
    /// `stride`, `padding`, and `dilation` are `(height, width)` pairs.
    /// `groups == in_channels` selects the depthwise regime.
    pub fn conv2d(
        x: &Array,
        weight: &Array,
        stride: (i32, i32),
        padding: (i32, i32),
        dilation: (i32, i32),
        groups: i32,
    ) -> Result<Array> {
        mlx_rs::ops::conv2d(x, weight, stride, padding, dilation, groups)
            .context("mlx-rs ops::conv2d FFI call failed")
    }
}

#[cfg(feature = "mlx-native")]
#[allow(unused_imports)] // Consumed by the VAE decoder (Phase 0).
pub use imp::conv2d;

// Self-verifying parity tests.
//
// Unlike the conv1d suite these need no external fixture: the inputs are tiny
// enough to compute the expected cross-correlation by hand, which both proves
// the FFI binding is wired correctly *and* pins MLX's orientation/stride
// semantics (channels-last, no kernel flip).
#[cfg(all(test, feature = "mlx-native"))]
mod parity_tests {
    use super::imp::conv2d;
    use mlx_rs::Array;

    /// 3×3 input, 2×2 all-ones kernel, valid padding → each output is the sum of
    /// the 2×2 window. Hand-computed expected values pin orientation:
    ///   input rows: [1 2 3] / [4 5 6] / [7 8 9]
    ///   out[0,0]=1+2+4+5=12  out[0,1]=2+3+5+6=16
    ///   out[1,0]=4+5+7+8=24  out[1,1]=5+6+8+9=28
    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn ones_kernel_3x3_matches_hand_computed() {
        let x = Array::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[1, 3, 3, 1], // [batch, H, W, C_in]
        );
        let w = Array::from_slice(
            &[1.0f32, 1.0, 1.0, 1.0],
            &[1, 2, 2, 1], // [C_out, kH, kW, C_in]
        );

        let out = conv2d(&x, &w, (1, 1), (0, 0), (1, 1), 1)
            .expect("mlx-rs ops::conv2d FFI call must succeed");

        assert_eq!(
            out.shape(),
            &[1, 2, 2, 1],
            "conv2d output shape mismatch (valid padding, stride 1)"
        );
        out.eval().expect("mlx eval must succeed");

        let observed: &[f32] = out.as_slice();
        let expected = [12.0f32, 16.0, 24.0, 28.0];
        for (i, (&got, &exp)) in observed.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "conv2d[{i}]: got {got}, expected {exp} (orientation/sum mismatch)"
            );
        }
    }

    /// stride-2 over a 4×4 input with a 2×2 kernel → 2×2 output. Verifies the
    /// output-shape arithmetic `(H - kH)/stride + 1` and that strided sampling
    /// lands on the expected windows.
    ///   input rows: [1 2 3 4]/[5 6 7 8]/[9 10 11 12]/[13 14 15 16]
    ///   stride 2 windows top-left at (0,0) and (0,2),(2,0),(2,2):
    ///   out[0,0]=1+2+5+6=14   out[0,1]=3+4+7+8=22
    ///   out[1,0]=9+10+13+14=46 out[1,1]=11+12+15+16=54
    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn ones_kernel_stride2_shape_and_values() {
        let x = Array::from_slice(
            &[
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[1, 4, 4, 1],
        );
        let w = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[1, 2, 2, 1]);

        let out = conv2d(&x, &w, (2, 2), (0, 0), (1, 1), 1)
            .expect("mlx-rs ops::conv2d FFI call must succeed");

        assert_eq!(out.shape(), &[1, 2, 2, 1], "stride-2 output shape mismatch");
        out.eval().expect("mlx eval must succeed");

        let observed: &[f32] = out.as_slice();
        let expected = [14.0f32, 22.0, 46.0, 54.0];
        for (i, (&got, &exp)) in observed.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "conv2d_stride2[{i}]: got {got}, expected {exp}"
            );
        }
    }
}
