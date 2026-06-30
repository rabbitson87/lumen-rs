//! 4-axis (4D) rotary position embedding for the FLUX.2 DiT.
//!
//! Mirrors `Flux2PosEmbed` + `AttentionUtils.apply_rope_bshd`.
//!
//! `pos_embed(ids[seq,4]) -> (cos[seq, sum(axes/2)], sin[..])`:
//! for each of the 4 axes (dim=32), `omega = 1/theta^(arange(0,32,2)/32)` (16
//! freqs), `out = pos[:,axis,None] * omega[None]`, take cos/sin, concat the 4
//! axes along the last dim → width `4*16 = 64 = head_dim/2`.
//!
//! Since `ids` are integers and `theta`/`axes` are fixed config, the whole
//! cos/sin table is deterministic and cheap to build on the host; we return it
//! as two f32 `Array`s `[seq, 64]`. `apply_rope_bshd` then rotates Q/K.
//!
//! `apply_rope_bshd(x[B,H,S,D], cos/sin[S, D/2])`: view last dim as `[D/2, 2]`
//! pairs `(real, imag)`; `out0 = real*cos - imag*sin`, `out1 = imag*cos +
//! real*sin`; stack back to `[..,D/2,2]` and reshape to `[..,D]`.

#[cfg(feature = "mlx-native")]
mod imp {
    use anyhow::{Context, Result};
    use mlx_rs::Array;
    use mlx_rs::ops::{concatenate_axis, split, stack_axis};

    /// Build the (cos, sin) RoPE tables for a set of position ids.
    ///
    /// - `ids`: row-major `[seq, 4]` integer positions (as i32).
    /// - `axes_dim`: per-axis rope dim (e.g. `[32,32,32,32]`).
    /// - returns `(cos, sin)` each `[seq, sum(axes_dim)/2]` as f32 `Array`s.
    ///
    /// Computed on the host (deterministic given integer ids + config), then
    /// uploaded as f32 arrays — matching the mflux float32 reference exactly.
    pub fn pos_embed(
        ids: &[i32],
        seq: usize,
        axes_dim: &[usize],
        theta: f32,
    ) -> Result<(Array, Array)> {
        let n_axes = axes_dim.len();
        assert_eq!(ids.len(), seq * n_axes, "ids must be [seq, n_axes]");
        let half_total: usize = axes_dim.iter().map(|d| d / 2).sum();

        let mut cos_buf = vec![0.0f32; seq * half_total];
        let mut sin_buf = vec![0.0f32; seq * half_total];

        for s in 0..seq {
            let mut col = 0usize;
            for (ax, &dim) in axes_dim.iter().enumerate() {
                let pos = ids[s * n_axes + ax] as f32;
                let nfreq = dim / 2;
                for j in 0..nfreq {
                    // scale = (2*j)/dim ; omega = 1/theta^scale
                    let scale = (2 * j) as f32 / dim as f32;
                    let omega = 1.0f32 / theta.powf(scale);
                    let angle = pos * omega;
                    let idx = s * half_total + col + j;
                    cos_buf[idx] = angle.cos();
                    sin_buf[idx] = angle.sin();
                }
                col += nfreq;
            }
        }

        let cos = Array::from_slice(&cos_buf, &[seq as i32, half_total as i32]);
        let sin = Array::from_slice(&sin_buf, &[seq as i32, half_total as i32]);
        Ok((cos, sin))
    }

    /// Rotate a single `[B,H,S,D]` tensor by (cos, sin) `[S, D/2]`.
    fn rope_mix(x: &Array, cos_b: &Array, sin_b: &Array) -> Result<Array> {
        let shape = x.shape().to_vec(); // [B,H,S,D]
        let d = *shape.last().unwrap();
        let mut pair_shape = shape.clone();
        let last = pair_shape.len() - 1;
        pair_shape[last] = d / 2;
        pair_shape.push(2);
        // [B,H,S,D] -> [B,H,S,D/2,2]
        let x2 = x
            .reshape(&pair_shape)
            .context("rope: reshape to pairs failed")?;
        let parts = split(&x2, 2, -1).context("rope: split real/imag failed")?;
        // each [B,H,S,D/2,1] -> squeeze last
        let real = parts[0]
            .reshape(&shape_drop_last(&pair_shape))
            .context("rope: reshape real failed")?;
        let imag = parts[1]
            .reshape(&shape_drop_last(&pair_shape))
            .context("rope: reshape imag failed")?;
        // out0 = real*cos - imag*sin ; out1 = imag*cos + real*sin
        let rc = real.multiply(cos_b).context("rope: real*cos failed")?;
        let is = imag.multiply(sin_b).context("rope: imag*sin failed")?;
        let out0 = rc.subtract(&is).context("rope: out0 failed")?;
        let ic = imag.multiply(cos_b).context("rope: imag*cos failed")?;
        let rs = real.multiply(sin_b).context("rope: real*sin failed")?;
        let out1 = ic.add(&rs).context("rope: out1 failed")?;
        // stack along new last axis -> [B,H,S,D/2,2] -> reshape [B,H,S,D]
        let stacked = stack_axis(&[&out0, &out1], -1).context("rope: stack failed")?;
        stacked.reshape(&shape).context("rope: reshape back failed")
    }

    fn shape_drop_last(s: &[i32]) -> Vec<i32> {
        s[..s.len() - 1].to_vec()
    }

    /// Apply 4D RoPE to Q and K, both `[B,H,S,D]`. cos/sin are `[S, D/2]`;
    /// broadcast to `[1,1,S,D/2]`.
    pub fn apply_rope_bshd(
        q: &Array,
        k: &Array,
        cos: &Array,
        sin: &Array,
    ) -> Result<(Array, Array)> {
        let s = cos.dim(0);
        let half = cos.dim(1);
        let cos_b = cos
            .reshape(&[1, 1, s, half])
            .context("rope: cos broadcast")?;
        let sin_b = sin
            .reshape(&[1, 1, s, half])
            .context("rope: sin broadcast")?;
        let qo = rope_mix(q, &cos_b, &sin_b)?;
        let ko = rope_mix(k, &cos_b, &sin_b)?;
        Ok((qo, ko))
    }

    // Keep the concat helper visible for callers building joint [txt;img] tables.
    pub fn concat_seq(a: &Array, b: &Array) -> Result<Array> {
        concatenate_axis(&[a, b], 0).context("rope: concat seq failed")
    }
}

#[cfg(feature = "mlx-native")]
pub use imp::{apply_rope_bshd, concat_seq, pos_embed};

#[cfg(all(test, feature = "mlx-native"))]
mod parity_tests {
    use super::imp::{apply_rope_bshd, pos_embed};
    use mlx_rs::Array;

    /// Single axis pair sanity: pos=1, dim=2 axis only (omega=1), D=2.
    /// cos=cos(1)=0.5403, sin=sin(1)=0.84147.
    /// q=[1,0] -> out0=1*cos-0*sin=0.5403; out1=0*cos+1*sin=0.84147.
    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn rope_single_pair_rotation() {
        // one axis, dim=2 -> half_total=1, head_dim D=2.
        let (cos, sin) = pos_embed(&[1], 1, &[2], 2000.0).unwrap();
        cos.eval().unwrap();
        sin.eval().unwrap();
        let cs: &[f32] = cos.as_slice();
        let sn: &[f32] = sin.as_slice();
        assert!((cs[0] - 0.5403023).abs() < 1e-4);
        assert!((sn[0] - 0.84147096).abs() < 1e-4);

        // q,k = [B=1,H=1,S=1,D=2] = [1,0]
        let q = Array::from_slice(&[1.0f32, 0.0], &[1, 1, 1, 2]);
        let k = Array::from_slice(&[1.0f32, 0.0], &[1, 1, 1, 2]);
        let (qo, _ko) = apply_rope_bshd(&q, &k, &cos, &sin).unwrap();
        qo.eval().unwrap();
        let o: &[f32] = qo.as_slice();
        assert!((o[0] - 0.5403023).abs() < 1e-4, "out0={}", o[0]);
        assert!((o[1] - 0.84147096).abs() < 1e-4, "out1={}", o[1]);
    }
}
