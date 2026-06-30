//! Affine linear layer (`x @ Wᵀ + b`) for the FLUX.2 VAE attention block.
//!
//! diffusers stores `Linear` weights as `[out, in]`; mlx `matmul` wants the
//! contraction axis last, so we accept the weight pre-transposed to `[in, out]`
//! (the caller transposes once at load time) and just do `matmul(x, w) + b`.

#[cfg(feature = "mlx-native")]
mod imp {
    use anyhow::{Context, Result};
    use mlx_rs::Array;

    /// Linear: `matmul(x, w_in_out) + b`.
    ///
    /// - `x`: `[.., in]`
    /// - `w_in_out`: `[in, out]` (already transposed from diffusers `[out, in]`)
    /// - `b`: `[out]`, broadcast over the leading axes.
    pub fn linear(x: &Array, w_in_out: &Array, b: &Array) -> Result<Array> {
        let y = mlx_rs::ops::matmul(x, w_in_out).context("linear: matmul failed")?;
        y.add(b).context("linear: + bias failed")
    }
}

#[cfg(feature = "mlx-native")]
#[allow(unused_imports)] // Consumed by the VAE attention block (Phase 0).
pub use imp::linear;

#[cfg(all(test, feature = "mlx-native"))]
mod parity_tests {
    use super::imp::linear;
    use mlx_rs::Array;

    /// x=[[1,2]], w(out,in)=[[1,0],[0,1],[1,1]] -> w(in,out)=[[1,0,1],[0,1,1]];
    /// x@w_in_out = [1,2,3]; + b[10,20,30] = [11,22,33].
    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn linear_matches_hand_computed() {
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2]);
        // w_in_out is [in=2, out=3].
        let w = Array::from_slice(&[1.0f32, 0.0, 1.0, 0.0, 1.0, 1.0], &[2, 3]);
        let b = Array::from_slice(&[10.0f32, 20.0, 30.0], &[3]);
        let out = linear(&x, &w, &b).expect("linear must succeed");
        assert_eq!(out.shape(), &[1, 3]);
        out.eval().expect("mlx eval must succeed");
        let observed: &[f32] = out.as_slice();
        let expected = [11.0f32, 22.0, 33.0];
        for (i, (&got, &exp)) in observed.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "linear[{i}]: got {got}, expected {exp}"
            );
        }
    }
}
