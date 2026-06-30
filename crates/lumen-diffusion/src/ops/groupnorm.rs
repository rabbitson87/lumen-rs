//! Native MLX GroupNorm for the FLUX.2 VAE.
//!
//! A *functional* group normalization that takes the loaded `weight` / `bias`
//! arrays explicitly, rather than the stateful `mlx_rs::nn::GroupNorm` module
//! (which owns its parameters behind `Param` and needs `&mut self`). This fits
//! the VAE's weight-loading flow better and keeps the decoder forward pass
//! allocation-light and stateless.
//!
//! The reduction mirrors mlx-rs's tested `GroupNorm::pytorch_group_norm`: it
//! groups along the **last** axis (channels-last, matching our Conv2d output
//! `[batch, H, W, C]`) using the PyTorch grouping order, layer-norms each group,
//! then applies the affine `weight * x + bias`. FLUX.2's VAE is PyTorch-trained,
//! so `pytorch_compatible = true` is the right default; the VAE parity gate vs
//! the Swift reference will confirm.
//!
//! Design docs: `.ai/memory/active/flux2-dev-diffusion-port/`.

#[cfg(feature = "mlx-native")]
mod imp {
    use anyhow::{Context, Result};
    use mlx_rs::Array;

    /// Group normalization over channels-last input `[batch, ..spatial.., C]`.
    ///
    /// - `num_groups` must divide the channel count `C`.
    /// - `weight` / `bias` are optional affine params of shape `[C]` (broadcast
    ///   over the channels-last axis).
    /// - `pytorch_compatible` selects PyTorch's grouping order (default for
    ///   diffusers-trained VAEs).
    pub fn group_norm(
        x: &Array,
        num_groups: i32,
        weight: Option<&Array>,
        bias: Option<&Array>,
        eps: f32,
        pytorch_compatible: bool,
    ) -> Result<Array> {
        let batch = x.dim(0);
        let dims = x.dim(-1);
        let ndim = x.ndim();
        let rest: Vec<i32> = x.shape()[1..ndim - 1].to_vec();

        if dims % num_groups != 0 {
            anyhow::bail!("group_norm: channels {dims} not divisible by num_groups {num_groups}");
        }

        let normed = if pytorch_compatible {
            let group_size = dims / num_groups;
            // Split into groups → [batch, spatial, num_groups, group_size],
            // bring groups forward, flatten group's (spatial × group_size).
            let g = x
                .reshape(&[batch, -1, num_groups, group_size])
                .context("group_norm: reshape to groups failed")?;
            let g = g
                .transpose_axes(&[0, 2, 1, 3])
                .context("group_norm: transpose failed")?
                .reshape(&[batch, num_groups, -1])
                .context("group_norm: flatten group failed")?;
            // Normalize each group over its last axis (no affine here).
            let g = mlx_rs::fast::layer_norm(&g, None::<&Array>, None::<&Array>, eps)
                .context("group_norm: layer_norm FFI failed")?;
            // Undo the grouping reshape/transpose.
            let g = g
                .reshape(&[batch, num_groups, -1, group_size])
                .context("group_norm: un-flatten failed")?;
            let new_shape: Vec<i32> = std::iter::once(batch)
                .chain(rest.iter().copied())
                .chain(std::iter::once(dims))
                .collect();
            g.transpose_axes(&[0, 2, 1, 3])
                .context("group_norm: un-transpose failed")?
                .reshape(&new_shape)
                .context("group_norm: final reshape failed")?
        } else {
            // Non-PyTorch grouping: flatten to [batch, -1, num_groups] and
            // normalize over the spatial axis per group.
            let g = x
                .reshape(&[batch, -1, num_groups])
                .context("group_norm: reshape (non-pytorch) failed")?;
            let g = mlx_rs::fast::layer_norm(&g, None::<&Array>, None::<&Array>, eps)
                .context("group_norm: layer_norm FFI failed")?;
            let new_shape: Vec<i32> = std::iter::once(batch)
                .chain(rest.iter().copied())
                .chain(std::iter::once(dims))
                .collect();
            g.reshape(&new_shape)
                .context("group_norm: final reshape (non-pytorch) failed")?
        };

        // Affine: weight * x + bias, broadcasting [C] over channels-last.
        match (weight, bias) {
            (Some(w), Some(b)) => w
                .multiply(&normed)
                .and_then(|wx| wx.add(b))
                .context("group_norm: affine weight*x+bias failed"),
            (Some(w), None) => w
                .multiply(&normed)
                .context("group_norm: affine weight*x failed"),
            (None, None) => Ok(normed),
            (None, Some(_)) => anyhow::bail!("group_norm: bias without weight is unsupported"),
        }
    }
}

#[cfg(feature = "mlx-native")]
#[allow(unused_imports)] // Consumed by the VAE decoder (Phase 0).
pub use imp::group_norm;

#[cfg(all(test, feature = "mlx-native"))]
mod parity_tests {
    use super::imp::group_norm;
    use mlx_rs::Array;

    /// num_groups = 1 over 4 channels at a single spatial location reduces to a
    /// plain layer-norm over the channels. Hand-computed (population variance):
    ///   x=[1,2,3,4]  mean=2.5  var=1.25  std=sqrt(1.25)=1.118034
    ///   normalized = (x-2.5)/std = [-1.341641,-0.447214,0.447214,1.341641]
    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn single_group_matches_layernorm() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 1, 4]);

        let out = group_norm(&x, 1, None, None, 1e-5, true).expect("group_norm must succeed");
        assert_eq!(out.shape(), &[1, 1, 1, 4]);
        out.eval().expect("mlx eval must succeed");

        let observed: &[f32] = out.as_slice();
        let expected = [-1.341641f32, -0.447214, 0.447214, 1.341641];
        for (i, (&got, &exp)) in observed.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "group_norm[{i}]: got {got}, expected {exp}"
            );
        }
    }

    /// Same input with affine weight=2, bias=1 → out = 2*normalized + 1.
    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn affine_weight_bias_applied() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 1, 4]);
        let w = Array::from_slice(&[2.0f32, 2.0, 2.0, 2.0], &[4]);
        let b = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4]);

        let out =
            group_norm(&x, 1, Some(&w), Some(&b), 1e-5, true).expect("group_norm must succeed");
        out.eval().expect("mlx eval must succeed");

        let observed: &[f32] = out.as_slice();
        // 2 * [-1.341641,-0.447214,0.447214,1.341641] + 1
        let expected = [-1.683282f32, 0.105572, 1.894428, 3.683282];
        for (i, (&got, &exp)) in observed.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "group_norm_affine[{i}]: got {got}, expected {exp}"
            );
        }
    }

    /// Two groups over 4 channels: groups {0,1} and {2,3} normalize independently.
    /// At a single spatial location each group has 2 elements → normalized to
    /// ±1 (population std of [a,b] is |a-b|/2, so (x-mean)/std = ±1).
    ///   x=[1,2,3,4] → group0 [1,2]→[-1,1], group1 [3,4]→[-1,1]
    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn two_groups_normalize_independently() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 1, 4]);

        let out = group_norm(&x, 2, None, None, 1e-8, true).expect("group_norm must succeed");
        out.eval().expect("mlx eval must succeed");

        let observed: &[f32] = out.as_slice();
        let expected = [-1.0f32, 1.0, -1.0, 1.0];
        for (i, (&got, &exp)) in observed.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "group_norm_2groups[{i}]: got {got}, expected {exp}"
            );
        }
    }
}
