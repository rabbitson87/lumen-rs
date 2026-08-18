//! Sinusoidal timestep embedding for the FLUX.2 DiT.
//!
//! Mirrors `Flux2TimestepGuidanceEmbeddings._timestep_embedding`:
//! ```text
//! half = dim // 2
//! freqs = exp(-ln(10000) * arange(half) / half)
//! args  = timesteps[:,None] * freqs[None,:]
//! emb   = concat([sin(args), cos(args)], -1)
//! emb   = concat([emb[:, half:], emb[:, :half]], -1)   # flip_sin_to_cos
//! ```
//! The frequency table is constant given `dim`, so we precompute it on the host
//! and feed a small f32 `Array`; only the per-timestep `args`/`sin`/`cos` touch
//! mlx. `dim` is even for the DiT (256), so the odd-`dim` zero-pad branch is
//! unnecessary.

#[cfg(feature = "mlx-native")]
mod imp {
    use anyhow::{Context, Result};
    use mlx_rs::Array;
    use mlx_rs::ops::{concatenate_axis, cos, sin};

    /// Sinusoidal timestep embedding with `flip_sin_to_cos=True`.
    ///
    /// - `timesteps`: `[B]` f32 array (already scaled by 1000 if needed).
    /// - returns `[B, dim]`.
    pub fn timestep_embedding(timesteps: &Array, dim: usize) -> Result<Array> {
        assert!(dim.is_multiple_of(2), "timestep_embedding dim must be even");
        let half = dim / 2;
        // freqs[i] = exp(-ln(10000) * i / half)
        let ln10000 = (10000.0f32).ln();
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-ln10000 * i as f32 / half as f32).exp())
            .collect();
        let freqs = Array::from_slice(&freqs, &[1, half as i32]); // [1, half]

        let b = timesteps.dim(0);
        let t = timesteps
            .reshape(&[b, 1])
            .context("timestep_embedding: reshape timesteps failed")?;
        // args = t[:,None] * freqs[None,:] -> [B, half]
        let args = t
            .multiply(&freqs)
            .context("timestep_embedding: t * freqs failed")?;
        let s = sin(&args).context("timestep_embedding: sin failed")?;
        let c = cos(&args).context("timestep_embedding: cos failed")?;
        // emb = concat([sin, cos]) then flip halves -> [cos, sin]
        let emb = concatenate_axis(&[&c, &s], -1)
            .context("timestep_embedding: flip_sin_to_cos concat failed")?;
        Ok(emb)
    }
}

#[cfg(feature = "mlx-native")]
pub use imp::timestep_embedding;

#[cfg(all(test, feature = "mlx-native"))]
mod parity_tests {
    use super::imp::timestep_embedding;
    use mlx_rs::Array;

    /// dim=4, half=2, freqs=[1, exp(-ln10000/2)=0.01]. t=1 ->
    /// args=[1,0.01]; sin=[0.84147,0.0099998]; cos=[0.54030,0.99995].
    /// flip_sin_to_cos => [cos, sin] = [0.54030,0.99995, 0.84147,0.0099998].
    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn timestep_embedding_matches_hand_computed() {
        let t = Array::from_slice(&[1.0f32], &[1]);
        let out = timestep_embedding(&t, 4).expect("ts embed");
        assert_eq!(out.shape(), &[1, 4]);
        out.eval().unwrap();
        let obs: &[f32] = out.as_slice();
        let exp = [0.54030f32, 0.99995, 0.84147, 0.0099998];
        for (i, (&g, &e)) in obs.iter().zip(exp.iter()).enumerate() {
            assert!((g - e).abs() < 1e-4, "[{i}]: got {g}, expected {e}");
        }
    }
}
