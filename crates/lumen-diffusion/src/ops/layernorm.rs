//! Affine-false LayerNorm for the FLUX.2 DiT.
//!
//! Every LayerNorm inside the klein-4B transformer blocks is
//! `nn.LayerNorm(dim, eps=1e-6, affine=False)` — pure normalization over the
//! last axis with no learned weight/bias (the modulation `(1+scale)*x + shift`
//! is applied separately by the block). mlx's `fast::layer_norm` accepts `None`
//! for both weight and bias, giving exactly the affine-false normalization.

#[cfg(feature = "mlx-native")]
mod imp {
    use anyhow::{Context, Result};
    use mlx_rs::Array;

    /// LayerNorm over the last axis with NO affine (weight=bias=None).
    pub fn layer_norm_no_affine(x: &Array, eps: f32) -> Result<Array> {
        mlx_rs::fast::layer_norm(x, None, None, eps)
            .context("layer_norm_no_affine: fast::layer_norm FFI failed")
    }
}

#[cfg(feature = "mlx-native")]
pub use imp::layer_norm_no_affine;

#[cfg(all(test, feature = "mlx-native"))]
mod parity_tests {
    use super::imp::layer_norm_no_affine;
    use mlx_rs::Array;

    /// LayerNorm of [1,2,3,4] over last axis: mean=2.5, var=1.25,
    /// (x-2.5)/sqrt(1.25+eps) ≈ [-1.3416,-0.4472,0.4472,1.3416].
    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn layer_norm_no_affine_matches_hand_computed() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4]);
        let out = layer_norm_no_affine(&x, 1e-6).expect("ln must succeed");
        out.eval().expect("mlx eval must succeed");
        let observed: &[f32] = out.as_slice();
        let expected = [-1.34163f32, -0.44721, 0.44721, 1.34163];
        for (i, (&got, &exp)) in observed.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-3, "[{i}]: got {got}, expected {exp}");
        }
    }
}
