//! Compiled (fused) forms of the elementwise-heavy helpers of the Inkling 2 decode step.
//!
//! A decode step makes about 21,000 MLX primitives and 11,600 kernel dispatches, 1% of them
//! weight kernels (see `profile.rs`); launching them costs more than half of the step. MLX's
//! `compile` turns a chain of elementwise ops into one kernel, so each helper here is the same
//! arithmetic as its plain form in `ops.rs`, in the same order and the same dtypes, wrapped in a
//! persistent compiled closure (`native_compile_cache`). The plain forms stay: `LUMEN_INKLING2_FUSE=0`
//! serves them, and the tests run both on the same inputs.
//!
//! A closure compiled `shapeless` accepts any input shape, so the helpers that hold only
//! elementwise ops and last-axis reductions are used for every row count. Helpers that cut or
//! reshape by the shape they see compile per shape and are used only for decode-sized inputs
//! ([`DECODE_ROWS`]), so a long prompt's chunk lengths never fill the compile cache.

use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};
use mlx_rs::error::Exception;
use mlx_rs::{Array, Dtype};

use crate::native_compile_cache::{CompiledMultiRefs, invoke_compiled_multi_refs};

/// Inputs of at most this many rows take the helpers that compile per shape.
pub(super) const DECODE_ROWS: usize = 16;

lumen_flags::flag! {
    /// Serve the compiled (fused) forms of the Inkling 2 elementwise helpers: the length-1 decode
    /// step makes a third of the kernel dispatches. `=0` restores the plain composed ops.
    pub(crate) fuse {
        env: "LUMEN_INKLING2_FUSE",
        default: true,
        kind: Optimization,
    }
}

/// Whether the fused forms are served.
pub(super) fn enabled() -> bool {
    fuse::get()
}

fn one(
    slot: &'static OnceLock<Mutex<CompiledMultiRefs>>,
    inner: fn(&[Array]) -> std::result::Result<Vec<Array>, Exception>,
    shapeless: bool,
    args: &[&Array],
    what: &str,
) -> Result<Array> {
    let mut out = invoke_compiled_multi_refs(slot, inner, shapeless, args)?;
    out.pop()
        .with_context(|| format!("{what}: the compiled graph returned nothing"))
}

fn f32_of(x: &Array) -> std::result::Result<Array, Exception> {
    x.as_dtype(Dtype::Float32)
}

/// `ops::l2_clip`: `[x, scale, threshold]`, a rank-0 `threshold`.
fn l2_clip_inner(args: &[Array]) -> std::result::Result<Vec<Array>, Exception> {
    let (x, scale, threshold) = (&args[0], &args[1], &args[2]);
    let xf = mlx_rs::ops::multiply(f32_of(x)?, scale)?;
    let squares = mlx_rs::ops::multiply(&xf, &xf)?;
    let norm = mlx_rs::ops::sqrt(mlx_rs::ops::sum_axis(&squares, -1, true)?)?;
    let denom = mlx_rs::ops::add(&norm, Array::from_f32(1e-6))?;
    let factor = mlx_rs::ops::minimum(
        mlx_rs::ops::divide(threshold, &denom)?,
        Array::from_f32(1.0),
    )?;
    Ok(vec![
        mlx_rs::ops::multiply(&xf, &factor)?.as_dtype(x.dtype())?,
    ])
}

static L2_CLIP: OnceLock<Mutex<CompiledMultiRefs>> = OnceLock::new();

pub(super) fn l2_clip(x: &Array, scale: &Array, threshold: &Array) -> Result<Array> {
    one(
        &L2_CLIP,
        l2_clip_inner,
        true,
        &[x, scale, threshold],
        "l2_clip",
    )
}

/// `ops::softcapped_swiglu`: `[gate, up, gate_cap, up_cap]`, the caps rank 0.
fn swiglu_inner(args: &[Array]) -> std::result::Result<Vec<Array>, Exception> {
    let (gate, up, gc, uc) = (&args[0], &args[1], &args[2], &args[3]);
    let g = f32_of(gate)?;
    let u = f32_of(up)?;
    let capped_gate = mlx_rs::ops::multiply(mlx_rs::ops::tanh(mlx_rs::ops::divide(&g, gc)?)?, gc)?;
    let capped_up = mlx_rs::ops::multiply(mlx_rs::ops::tanh(mlx_rs::ops::divide(&u, uc)?)?, uc)?;
    let act = mlx_rs::ops::multiply(
        mlx_rs::ops::multiply(&capped_gate, mlx_rs::ops::sigmoid(&g)?)?,
        &capped_up,
    )?;
    Ok(vec![act.as_dtype(gate.dtype())?])
}

static SWIGLU: OnceLock<Mutex<CompiledMultiRefs>> = OnceLock::new();

pub(super) fn softcapped_swiglu(
    gate: &Array,
    up: &Array,
    gate_cap: &Array,
    up_cap: &Array,
) -> Result<Array> {
    one(
        &SWIGLU,
        swiglu_inner,
        true,
        &[gate, up, gate_cap, up_cap],
        "softcapped_swiglu",
    )
}

// ---- the projection region ----------------------------------------------------------------------
//
// After the projection's matmul, an attention layer runs two short convolutions, four clips, a value
// norm, the position scaling and the relative-bias matmul: about 150 primitives a visit, a third of
// a decode step. As one compiled region they are a handful of kernels. The region takes the
// projection's output pieces and the layer's small weights as arguments; the shapes and the clip
// constants are those of the layer kind (local or global), baked in when the region is first traced.

/// What a projection region bakes in. Every layer of a kind has the same one; a layer that does
/// not (another model's config) is not served by the region.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) struct ProjCfg {
    pub heads: i32,
    pub kv_heads: i32,
    pub head_dim: i32,
    pub d_rel: i32,
    /// `kv_heads * head_dim`: where a global layer's k projection splits.
    pub dkv: i32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    /// The residual scale of the layer kind's convolution.
    pub conv_scale: f32,
}

static LOCAL_CFG: OnceLock<ProjCfg> = OnceLock::new();
static GLOBAL_CFG: OnceLock<ProjCfg> = OnceLock::new();
static PROJECT_LOCAL: OnceLock<Mutex<CompiledMultiRefs>> = OnceLock::new();
static PROJECT_GLOBAL: OnceLock<Mutex<CompiledMultiRefs>> = OnceLock::new();

fn ex<T>(r: Result<T>) -> std::result::Result<T, Exception> {
    r.map_err(|e| Exception::custom(format!("{e:#}")))
}

/// The tail both kinds share, from the convolved k and v: clips, the value norm, the position
/// scaling and the relative-bias matmul. Returns `[q, k, v, rel]`, the first three as `[B, H, L, D]`.
fn projection_tail(
    cfg: &ProjCfg,
    (q, k, v, r): (&Array, Array, Array, &Array),
    (q_scale, r_scale, rel_proj): (&Array, &Array, &Array),
    tau: Option<&Array>,
) -> std::result::Result<Vec<Array>, Exception> {
    use super::ops::{l2_clip_plain, scalar, unit_rms};
    let (b, length) = (q.shape()[0], q.shape()[1]);
    let (nh, nkv, hd) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
    let one = scalar(1.0);
    let q = ex(l2_clip_plain(
        &mlx_rs::ops::reshape(q, &[b, length, nh, hd])?,
        q_scale,
        cfg.qk_clip,
    ))?;
    let k = ex(l2_clip_plain(
        &mlx_rs::ops::reshape(&k, &[b, length, nkv, hd])?,
        &one,
        cfg.qk_clip,
    ))?;
    // The value norm has no gamma.
    let v = ex(unit_rms(
        &mlx_rs::ops::reshape(&v, &[b, length, nkv, hd])?.as_dtype(Dtype::Float32)?,
    ))?
    .as_dtype(q.dtype())?;
    let r = ex(l2_clip_plain(
        &mlx_rs::ops::reshape(r, &[b, length, nh, cfg.d_rel])?,
        r_scale,
        cfg.rel_clip,
    ))?;
    let (q, r) = match tau {
        None => (q, r),
        Some(tau) => {
            let scaled = |a: &Array| -> std::result::Result<Array, Exception> {
                mlx_rs::ops::multiply(a.as_dtype(Dtype::Float32)?, tau)?.as_dtype(a.dtype())
            };
            (scaled(&q)?, scaled(&r)?)
        }
    };
    let rel = mlx_rs::ops::matmul(&r, rel_proj.as_dtype(r.dtype())?)?;
    Ok(vec![
        mlx_rs::ops::transpose_axes(&q, &[0, 2, 1, 3])?,
        mlx_rs::ops::transpose_axes(&k, &[0, 2, 1, 3])?,
        mlx_rs::ops::transpose_axes(&v, &[0, 2, 1, 3])?,
        rel,
    ])
}

/// Arguments: `[q, k, v, r, k_state, v_state, k_conv, v_conv, q_scale, r_scale, rel_proj, tau?]`.
/// Returns `[q, k, v, rel, k_window, v_window]`.
fn project_local_inner(a: &[Array]) -> std::result::Result<Vec<Array>, Exception> {
    let cfg = LOCAL_CFG
        .get()
        .expect("the config is claimed before the first call");
    let (k, k_window) = ex(super::ops::static_conv(&a[1], &a[6], &a[4], cfg.conv_scale))?;
    let (v, v_window) = ex(super::ops::static_conv(&a[2], &a[7], &a[5], cfg.conv_scale))?;
    let mut out = projection_tail(cfg, (&a[0], k, v, &a[3]), (&a[8], &a[9], &a[10]), a.get(11))?;
    out.extend([k_window, v_window]);
    Ok(out)
}

/// Arguments: `[q, r, kp, vp, k_state, v_state, k_static, v_static, q_scale, r_scale, rel_proj,
/// tau?]`, `kp` and `vp` the whole k and v projections (the values, then the dynamic taps).
fn project_global_inner(a: &[Array]) -> std::result::Result<Vec<Array>, Exception> {
    use mlx_rs::ops::indexing::TryIndexOp;
    let cfg = GLOBAL_CFG
        .get()
        .expect("the config is claimed before the first call");
    let dkv = cfg.dkv;
    let conv = |p: &Array,
                static_w: &Array,
                state: &Array|
     -> std::result::Result<(Array, Array), Exception> {
        ex(super::ops::dynamic_conv(
            &p.try_index((.., .., ..dkv))?,
            &p.try_index((.., .., dkv..))?,
            static_w,
            state,
            cfg.kv_heads,
            cfg.conv_scale,
        ))
    };
    let (k, k_window) = conv(&a[2], &a[6], &a[4])?;
    let (v, v_window) = conv(&a[3], &a[7], &a[5])?;
    let mut out = projection_tail(cfg, (&a[0], k, v, &a[1]), (&a[8], &a[9], &a[10]), a.get(11))?;
    out.extend([k_window, v_window]);
    Ok(out)
}

/// The compiled projection region of a local layer, or `None` when this layer's constants are not
/// the ones the region was traced with.
pub(super) fn project_local(cfg: ProjCfg, args: &[&Array]) -> Result<Option<Vec<Array>>> {
    if *LOCAL_CFG.get_or_init(|| cfg) != cfg {
        return Ok(None);
    }
    Ok(Some(invoke_compiled_multi_refs(
        &PROJECT_LOCAL,
        project_local_inner,
        false,
        args,
    )?))
}

/// [`project_local`] for a global layer.
pub(super) fn project_global(cfg: ProjCfg, args: &[&Array]) -> Result<Option<Vec<Array>>> {
    if *GLOBAL_CFG.get_or_init(|| cfg) != cfg {
        return Ok(None);
    }
    Ok(Some(invoke_compiled_multi_refs(
        &PROJECT_GLOBAL,
        project_global_inner,
        false,
        args,
    )?))
}

// ---- the residual regions -----------------------------------------------------------------------
//
// The constants every layer shares: the convolutions' residual scale, the attention residual's
// softmax scale and whether the loop controllers round through bf16. Claimed by the first region
// called; a region is not served for a call whose constants differ from the claimed ones.

#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) struct SharedCfg {
    pub conv_scale: f32,
    pub mix_scale: f32,
    pub round_loop_mix: bool,
}

static SHARED_CFG: OnceLock<SharedCfg> = OnceLock::new();

fn claimed(cfg: SharedCfg) -> bool {
    *SHARED_CFG.get_or_init(|| cfg) == cfg
}

fn shared() -> &'static SharedCfg {
    SHARED_CFG
        .get()
        .expect("the shared config is claimed before the first call")
}

/// Arguments `[x, w, state, h?]`: the convolution of an attention or MLP output and, without `h`,
/// the residual of a block boundary (`x` alone), with `h` the residual `h + x`, in float32.
/// Returns `[residual, window]`.
fn conv_residual_inner(a: &[Array]) -> std::result::Result<Vec<Array>, Exception> {
    let (y, window) = ex(super::ops::static_conv(
        &a[0],
        &a[1],
        &a[2],
        shared().conv_scale,
    ))?;
    let y32 = y.as_dtype(Dtype::Float32)?;
    let residual = match a.get(3) {
        Some(h) => mlx_rs::ops::add(h.as_dtype(Dtype::Float32)?, y32)?,
        None => y32,
    };
    Ok(vec![residual, window])
}

/// Arguments `[m, w, state, residual]`: the MLP output's convolution added to the residual, back in
/// the layer's dtype. Returns `[next, window]`.
fn conv_next_inner(a: &[Array]) -> std::result::Result<Vec<Array>, Exception> {
    let (y, window) = ex(super::ops::static_conv(
        &a[0],
        &a[1],
        &a[2],
        shared().conv_scale,
    ))?;
    let next = mlx_rs::ops::add(&a[3], y.as_dtype(Dtype::Float32)?)?.as_dtype(a[0].dtype())?;
    Ok(vec![next, window])
}

static CONV_RESIDUAL: OnceLock<Mutex<CompiledMultiRefs>> = OnceLock::new();
static CONV_NEXT: OnceLock<Mutex<CompiledMultiRefs>> = OnceLock::new();

/// [`conv_residual_inner`] compiled; `None` when the constants are not the claimed ones.
pub(super) fn conv_residual(cfg: SharedCfg, args: &[&Array]) -> Result<Option<Vec<Array>>> {
    if !claimed(cfg) {
        return Ok(None);
    }
    Ok(Some(invoke_compiled_multi_refs(
        &CONV_RESIDUAL,
        conv_residual_inner,
        false,
        args,
    )?))
}

/// [`conv_next_inner`] compiled; `None` when the constants are not the claimed ones.
pub(super) fn conv_next(cfg: SharedCfg, args: &[&Array]) -> Result<Option<Vec<Array>>> {
    if !claimed(cfg) {
        return Ok(None);
    }
    Ok(Some(invoke_compiled_multi_refs(
        &CONV_NEXT,
        conv_next_inner,
        false,
        args,
    )?))
}

/// Arguments `[x, message, down, gain_forget]`: the loop controllers' mix.
fn mix_messages_inner(a: &[Array]) -> std::result::Result<Vec<Array>, Exception> {
    Ok(vec![ex(super::ops::mix_messages(
        &a[0],
        &a[1],
        &a[2],
        &a[3],
        shared().round_loop_mix,
    ))?])
}

/// Arguments `[partial_n, query, blocks...]`: the attention residual's mix.
fn attn_res_mix_inner(a: &[Array]) -> std::result::Result<Vec<Array>, Exception> {
    Ok(vec![ex(super::ops::attn_res_mix(
        &a[2..],
        &a[0],
        &a[1],
        shared().mix_scale,
    ))?])
}

static MIX_MESSAGES: OnceLock<Mutex<CompiledMultiRefs>> = OnceLock::new();
static ATTN_RES_MIX: OnceLock<Mutex<CompiledMultiRefs>> = OnceLock::new();

/// [`mix_messages_inner`] compiled; `None` when the constants are not the claimed ones.
pub(super) fn mix_messages(cfg: SharedCfg, args: &[&Array]) -> Result<Option<Array>> {
    if !claimed(cfg) {
        return Ok(None);
    }
    one(
        &MIX_MESSAGES,
        mix_messages_inner,
        false,
        args,
        "mix_messages",
    )
    .map(Some)
}

/// [`attn_res_mix_inner`] compiled; `None` when the constants are not the claimed ones.
pub(super) fn attn_res_mix(cfg: SharedCfg, args: &[&Array]) -> Result<Option<Array>> {
    if !claimed(cfg) {
        return Ok(None);
    }
    one(
        &ATTN_RES_MIX,
        attn_res_mix_inner,
        false,
        args,
        "attn_res_mix",
    )
    .map(Some)
}
