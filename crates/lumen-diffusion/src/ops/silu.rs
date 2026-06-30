//! SiLU / swish activation (`x * sigmoid(x)`) for the FLUX.2 VAE.
//!
//! The decoder's resnet blocks and the final pre-conv activation use SiLU. mlx
//! exposes `sigmoid`; we compose `x * sigmoid(x)` here so the decoder forward
//! stays a single small helper rather than inlining the multiply everywhere.

#[cfg(feature = "mlx-native")]
mod imp {
    use anyhow::{Context, Result};
    use mlx_rs::Array;

    /// SiLU activation: `x * sigmoid(x)`, elementwise.
    pub fn silu(x: &Array) -> Result<Array> {
        let s = mlx_rs::ops::sigmoid(x).context("silu: sigmoid FFI failed")?;
        x.multiply(&s).context("silu: x * sigmoid(x) failed")
    }
}

#[cfg(feature = "mlx-native")]
#[allow(unused_imports)] // Consumed by the VAE decoder (Phase 0).
pub use imp::silu;

#[cfg(all(test, feature = "mlx-native"))]
mod parity_tests {
    use super::imp::silu;
    use mlx_rs::Array;

    /// silu(0)=0, silu(1)=1/(1+e^-1)=0.7310586, silu(-1)=-0.2689414,
    /// silu(2)=2/(1+e^-2)=1.7615942.
    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn silu_matches_hand_computed() {
        let x = Array::from_slice(&[0.0f32, 1.0, -1.0, 2.0], &[4]);
        let out = silu(&x).expect("silu must succeed");
        out.eval().expect("mlx eval must succeed");
        let observed: &[f32] = out.as_slice();
        let expected = [0.0f32, 0.7310586, -0.26894143, 1.7615942];
        for (i, (&got, &exp)) in observed.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "silu[{i}]: got {got}, expected {exp}"
            );
        }
    }
}
