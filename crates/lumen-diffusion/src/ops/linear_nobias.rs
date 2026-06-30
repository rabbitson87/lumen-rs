//! Bias-free linear layer (`x @ Wᵀ`) for the FLUX.2 DiT.
//!
//! Every `Linear` in the klein-4B DiT is `bias=False`, so the affine
//! [`crate::ops::linear::linear`] (which always adds a bias) does not fit. The
//! diffusers weights are `[out, in]`; we accept them pre-transposed to
//! `[in, out]` (done once at load) and just `matmul(x, w)`.

#[cfg(feature = "mlx-native")]
mod imp {
    use anyhow::{Context, Result};
    use mlx_rs::Array;

    /// Bias-free linear: `matmul(x, w_in_out)`.
    ///
    /// - `x`: `[.., in]`
    /// - `w_in_out`: `[in, out]` (already transposed from diffusers `[out, in]`)
    pub fn linear_nobias(x: &Array, w_in_out: &Array) -> Result<Array> {
        mlx_rs::ops::matmul(x, w_in_out).context("linear_nobias: matmul failed")
    }
}

#[cfg(feature = "mlx-native")]
pub use imp::linear_nobias;

#[cfg(all(test, feature = "mlx-native"))]
mod parity_tests {
    use super::imp::linear_nobias;
    use mlx_rs::Array;

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn linear_nobias_matches_hand_computed() {
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2]);
        // w_in_out is [in=2, out=3]; x@w = [1,2,3].
        let w = Array::from_slice(&[1.0f32, 0.0, 1.0, 0.0, 1.0, 1.0], &[2, 3]);
        let out = linear_nobias(&x, &w).expect("linear_nobias must succeed");
        assert_eq!(out.shape(), &[1, 3]);
        out.eval().expect("mlx eval must succeed");
        let observed: &[f32] = out.as_slice();
        let expected = [1.0f32, 2.0, 3.0];
        for (i, (&got, &exp)) in observed.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-4, "[{i}]: got {got}, expected {exp}");
        }
    }
}
