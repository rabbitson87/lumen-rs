//! The arithmetic of a layer visit's regions, written once.
//!
//! Around its matmuls, a layer visit is long chains of small ops: the two short convolutions, the
//! clips, the value norm, the position scaling and the relative-bias matmul after the attention
//! projection; the convolution and the residual add after the attention and after the MLP. For a
//! decode step `fused.rs` compiles each chain into a few kernels. A long prefill chunk, and a run
//! with `LUMEN_INKLING2_FUSE=0`, run the same chain op by op. Both call the functions below (the
//! compiled form traces them), so a compiled region cannot compute anything its plain form does not,
//! and the two cannot drift apart.
//!
//! A region takes the arrays it reads as one slice in a fixed order, documented on each function, and
//! the numbers it builds its graph from as plain parameters. A compiled region is keyed on exactly
//! those parameters (`fused.rs`).

use anyhow::{Result, anyhow, bail};
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Dtype};

use super::ops::{dynamic_conv, l2_clip, scalar, static_conv, unit_rms};

/// What a projection region builds its graph from: the head geometry and the clip thresholds of a
/// layer kind, and the residual scale of its convolution.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) struct ProjCfg {
    pub heads: i32,
    pub kv_heads: i32,
    pub head_dim: i32,
    pub d_rel: i32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub conv_scale: f32,
}

fn wrong_count(region: &str, got: usize) -> anyhow::Error {
    anyhow!("the {region} region was given {got} arrays")
}

/// The tail both layer kinds share, from the convolved k and v: clips, the value norm, the position
/// scaling and the relative-bias matmul. `tail` is `[q_scale, r_scale, rel_proj, tau?]`. Returns
/// `[q, k, v, rel]`, the first three as `[B, H, L, D]`.
fn projection_tail(
    cfg: &ProjCfg,
    (q, k, v, r): (&Array, &Array, &Array, &Array),
    tail: &[&Array],
) -> Result<[Array; 4]> {
    let (q_scale, r_scale, rel_proj, tau) = match tail {
        [q_scale, r_scale, rel_proj] => (q_scale, r_scale, rel_proj, None),
        [q_scale, r_scale, rel_proj, tau] => (q_scale, r_scale, rel_proj, Some(tau)),
        _ => bail!(
            "the projection region was given {} arrays after its pieces",
            tail.len()
        ),
    };
    let (b, length) = (q.shape()[0], q.shape()[1]);
    let (nh, nkv, hd) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
    let one = scalar(1.0);
    let q = l2_clip(
        &mlx_rs::ops::reshape(q, &[b, length, nh, hd])?,
        q_scale,
        cfg.qk_clip,
    )?;
    let k = l2_clip(
        &mlx_rs::ops::reshape(k, &[b, length, nkv, hd])?,
        &one,
        cfg.qk_clip,
    )?;
    // The value norm has no gamma.
    let v = unit_rms(&mlx_rs::ops::reshape(v, &[b, length, nkv, hd])?.as_dtype(Dtype::Float32)?)?
        .as_dtype(q.dtype())?;
    let r = l2_clip(
        &mlx_rs::ops::reshape(r, &[b, length, nh, cfg.d_rel])?,
        r_scale,
        cfg.rel_clip,
    )?;
    let (q, r) = match tau {
        None => (q, r),
        Some(tau) => {
            let scaled = |a: &Array| -> Result<Array> {
                Ok(mlx_rs::ops::multiply(a.as_dtype(Dtype::Float32)?, tau)?.as_dtype(a.dtype())?)
            };
            (scaled(&q)?, scaled(&r)?)
        }
    };
    let rel = mlx_rs::ops::matmul(&r, rel_proj.as_dtype(r.dtype())?)?;
    Ok([
        mlx_rs::ops::transpose_axes(&q, &[0, 2, 1, 3])?,
        mlx_rs::ops::transpose_axes(&k, &[0, 2, 1, 3])?,
        mlx_rs::ops::transpose_axes(&v, &[0, 2, 1, 3])?,
        rel,
    ])
}

/// The projection of a local layer after its matmul. Arrays: `[q, k, v, r, k_state, v_state, k_conv,
/// v_conv, q_scale, r_scale, rel_proj, tau?]`. Returns `[q, k, v, rel, k_window, v_window]`.
pub(super) fn project_local(cfg: &ProjCfg, args: &[&Array]) -> Result<[Array; 6]> {
    let [q, k, v, r, k_state, v_state, k_conv, v_conv, tail @ ..] = args else {
        return Err(wrong_count("local projection", args.len()));
    };
    let (k, k_window) = static_conv(k, k_conv, k_state, cfg.conv_scale)?;
    let (v, v_window) = static_conv(v, v_conv, v_state, cfg.conv_scale)?;
    let [q, k, v, rel] = projection_tail(cfg, (q, &k, &v, r), tail)?;
    Ok([q, k, v, rel, k_window, v_window])
}

/// The projection of a global layer after its matmuls. Arrays: `[q, r, kp, vp, k_state, v_state,
/// k_static, v_static, q_scale, r_scale, rel_proj, tau?]`, `kp` and `vp` the whole k and v
/// projections (the values, then the dynamic taps). Returns `[q, k, v, rel, k_window, v_window]`.
pub(super) fn project_global(cfg: &ProjCfg, args: &[&Array]) -> Result<[Array; 6]> {
    let [
        q,
        r,
        kp,
        vp,
        k_state,
        v_state,
        k_static,
        v_static,
        tail @ ..,
    ] = args
    else {
        return Err(wrong_count("global projection", args.len()));
    };
    let dkv = cfg.kv_heads * cfg.head_dim;
    let conv = |p: &Array, static_w: &Array, state: &Array| -> Result<(Array, Array)> {
        dynamic_conv(
            &p.try_index((.., .., ..dkv))?,
            &p.try_index((.., .., dkv..))?,
            static_w,
            state,
            cfg.kv_heads,
            cfg.conv_scale,
        )
    };
    let (k, k_window) = conv(kp, k_static, k_state)?;
    let (v, v_window) = conv(vp, v_static, v_state)?;
    let [q, k, v, rel] = projection_tail(cfg, (q, &k, &v, r), tail)?;
    Ok([q, k, v, rel, k_window, v_window])
}

/// The convolution of an attention output and the residual it joins, in float32: the stream plus the
/// output, or the output alone at a block boundary. Arrays: `[x, w, state, h?]`. Returns `[residual,
/// window]`, the window being the convolution's padded input.
pub(super) fn conv_residual(conv_scale: f32, args: &[&Array]) -> Result<[Array; 2]> {
    let (x, w, state, h) = match args {
        [x, w, state] => (x, w, state, None),
        [x, w, state, h] => (x, w, state, Some(h)),
        _ => return Err(wrong_count("conv residual", args.len())),
    };
    let (y, window) = static_conv(x, w, state, conv_scale)?;
    let y = y.as_dtype(Dtype::Float32)?;
    let residual = match h {
        Some(h) => mlx_rs::ops::add(h.as_dtype(Dtype::Float32)?, y)?,
        None => y,
    };
    Ok([residual, window])
}

/// The MLP output's convolution added to the residual, in `dtype`. Arrays: `[m, w, state,
/// residual]`. Returns `[next, window]`.
pub(super) fn conv_next(conv_scale: f32, dtype: Dtype, args: &[&Array]) -> Result<[Array; 2]> {
    let [m, w, state, residual] = args else {
        return Err(wrong_count("conv next", args.len()));
    };
    let (y, window) = static_conv(m, w, state, conv_scale)?;
    let next = mlx_rs::ops::add(residual, y.as_dtype(Dtype::Float32)?)?.as_dtype(dtype)?;
    Ok([next, window])
}
