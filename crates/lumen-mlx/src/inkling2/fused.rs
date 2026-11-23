//! Compiled (fused) forms of the elementwise-heavy regions of the Inkling 2 decode step.
//!
//! A decode step makes about 21,000 MLX primitives and 11,600 kernel dispatches, 1% of them
//! weight kernels (see `profile.rs`); launching them costs more than half of the step. MLX's
//! `compile` turns a chain of elementwise ops into one kernel. Each function here is the plain
//! function of `regions.rs` or `ops.rs` (same arguments, same result) run through a compiled
//! closure: the closure traces that very function, so the compiled form computes what the plain
//! form computes, in the same order and dtypes. `LUMEN_INKLING2_FUSE=0` serves the plain forms, and
//! the tests run both on the same inputs.
//!
//! A region is keyed on its constants (`keyed_compile.rs`): the numbers the traced function builds
//! its graph from (a head count, a clip threshold, a scale) are stated with every call and handed to
//! the function when it is traced, so a layer with other constants gets a graph of its own, as a
//! layer with other shapes does. Nothing is claimed process-wide and no layer falls back.
//!
//! A region compiled `shapeless` accepts any input shape: the softcapped swiglu is used for every
//! row count. The others reshape and cut by the shape they see, compile per shape, and are used only
//! for decode-sized inputs ([`serves`]), so a long prompt's chunk lengths never fill the compile
//! cache.

use std::sync::LazyLock;

use anyhow::{Context, Result, anyhow, bail};
use mlx_rs::error::Exception;
use mlx_rs::{Array, Dtype};

use super::ops::{self, DECODE_ROWS};
use super::regions::{self, ProjCfg};
use crate::keyed_compile::KeyedCompiled;

lumen_flags::flag! {
    /// Serve the compiled (fused) forms of the Inkling 2 elementwise helpers: the length-1 decode
    /// step makes a third of the kernel dispatches. `=0` restores the plain composed ops.
    pub(crate) fuse {
        env: "LUMEN_INKLING2_FUSE",
        default: true,
        kind: Optimization,
    }
}

/// Whether the compiled forms are on.
pub(super) fn enabled() -> bool {
    fuse::get()
}

/// Whether a pass of `rows` rows takes the compiled forms of the regions that compile per shape: they
/// are on, and it is a decode step.
pub(super) fn serves(rows: usize) -> bool {
    enabled() && rows <= DECODE_ROWS
}

#[cfg(test)]
thread_local! {
    static SERVED: std::cell::RefCell<std::collections::BTreeMap<&'static str, usize>> =
        const { std::cell::RefCell::new(std::collections::BTreeMap::new()) };
}

/// How many times each compiled region has run on this thread (a test's check that each was served).
#[cfg(test)]
pub(super) fn served() -> std::collections::BTreeMap<&'static str, usize> {
    SERVED.with(|s| s.borrow().clone())
}

/// Runs a compiled region on `args` with its `constants`, and checks the count of what it returns.
fn run<const N: usize>(
    name: &'static str,
    region: &KeyedCompiled,
    args: &[&Array],
    constants: &[u64],
) -> Result<[Array; N]> {
    #[cfg(test)]
    SERVED.with(|s| *s.borrow_mut().entry(name).or_default() += 1);
    let out = region
        .call(args, constants)
        .with_context(|| format!("the {name} region"))?;
    let got = out.len();
    <[Array; N]>::try_from(out)
        .map_err(|_| anyhow!("the {name} region returned {got} arrays, not {N}"))
}

/// What a traced function hands back to MLX: its result, or its error as an exception.
fn traced<const N: usize>(
    f: impl FnOnce() -> Result<[Array; N]>,
) -> std::result::Result<Vec<Array>, Exception> {
    f().map(Vec::from)
        .map_err(|e| Exception::custom(format!("{e:#}")))
}

fn refs(arrays: &[Array]) -> Vec<&Array> {
    arrays.iter().collect()
}

// ---- constants -----------------------------------------------------------------------------------
//
// The key of a call is the numbers its traced function reads, as `u64`s: a float as its bits, an
// integer as itself. Each region encodes its constants in one function and decodes them in another
// next to it.

fn float(x: f32) -> u64 {
    u64::from(x.to_bits())
}

fn float_of(k: u64) -> f32 {
    f32::from_bits(k as u32)
}

fn int(x: i32) -> u64 {
    u64::from(x as u32)
}

fn int_of(k: u64) -> i32 {
    k as u32 as i32
}

fn dtype_of(k: u64) -> Result<Dtype> {
    [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16]
        .into_iter()
        .find(|d| *d as u64 == k)
        .with_context(|| format!("a compiled region cannot produce dtype {k}"))
}

fn proj_key(cfg: &ProjCfg) -> [u64; 7] {
    [
        int(cfg.heads),
        int(cfg.kv_heads),
        int(cfg.head_dim),
        int(cfg.d_rel),
        float(cfg.qk_clip),
        float(cfg.rel_clip),
        float(cfg.conv_scale),
    ]
}

fn proj_cfg(key: &[u64]) -> Result<ProjCfg> {
    let &[
        heads,
        kv_heads,
        head_dim,
        d_rel,
        qk_clip,
        rel_clip,
        conv_scale,
    ] = key
    else {
        bail!(
            "a projection region is keyed on 7 constants, got {}",
            key.len()
        );
    };
    Ok(ProjCfg {
        heads: int_of(heads),
        kv_heads: int_of(kv_heads),
        head_dim: int_of(head_dim),
        d_rel: int_of(d_rel),
        qk_clip: float_of(qk_clip),
        rel_clip: float_of(rel_clip),
        conv_scale: float_of(conv_scale),
    })
}

/// The one constant of a region that has one, or an error for a key of another length.
fn single(key: &[u64]) -> Result<u64> {
    match key {
        &[k] => Ok(k),
        _ => bail!("the region is keyed on one constant, got {}", key.len()),
    }
}

// ---- the softcapped swiglu -----------------------------------------------------------------------

fn trace_swiglu(a: &[Array], key: &[u64]) -> std::result::Result<Vec<Array>, Exception> {
    traced(|| {
        let ([gate, up], &[gcap, ucap]) = (a, key) else {
            bail!("the swiglu region takes two arrays and two caps");
        };
        Ok([ops::softcapped_swiglu(
            gate,
            up,
            float_of(gcap),
            float_of(ucap),
        )?])
    })
}

static SWIGLU: LazyLock<KeyedCompiled> = LazyLock::new(|| KeyedCompiled::new(trace_swiglu, true));

/// [`ops::softcapped_swiglu`], compiled.
pub(super) fn softcapped_swiglu(gate: &Array, up: &Array, gcap: f32, ucap: f32) -> Result<Array> {
    let [act] = run("swiglu", &SWIGLU, &[gate, up], &[float(gcap), float(ucap)])?;
    Ok(act)
}

// ---- the projection regions ----------------------------------------------------------------------
//
// After the projection's matmul, an attention layer runs two short convolutions, four clips, a value
// norm, the position scaling and the relative-bias matmul: about 150 primitives a visit, a third of
// a decode step. As one compiled region they are a handful of kernels.

fn trace_project_local(a: &[Array], key: &[u64]) -> std::result::Result<Vec<Array>, Exception> {
    traced(|| regions::project_local(&proj_cfg(key)?, &refs(a)))
}

fn trace_project_global(a: &[Array], key: &[u64]) -> std::result::Result<Vec<Array>, Exception> {
    traced(|| regions::project_global(&proj_cfg(key)?, &refs(a)))
}

static PROJECT_LOCAL: LazyLock<KeyedCompiled> =
    LazyLock::new(|| KeyedCompiled::new(trace_project_local, false));
static PROJECT_GLOBAL: LazyLock<KeyedCompiled> =
    LazyLock::new(|| KeyedCompiled::new(trace_project_global, false));

/// [`regions::project_local`], compiled.
pub(super) fn project_local(cfg: &ProjCfg, args: &[&Array]) -> Result<[Array; 6]> {
    run("local projection", &PROJECT_LOCAL, args, &proj_key(cfg))
}

/// [`regions::project_global`], compiled.
pub(super) fn project_global(cfg: &ProjCfg, args: &[&Array]) -> Result<[Array; 6]> {
    run("global projection", &PROJECT_GLOBAL, args, &proj_key(cfg))
}

// ---- the residual regions ------------------------------------------------------------------------

fn trace_conv_residual(a: &[Array], key: &[u64]) -> std::result::Result<Vec<Array>, Exception> {
    traced(|| regions::conv_residual(float_of(single(key)?), &refs(a)))
}

fn trace_conv_next(a: &[Array], key: &[u64]) -> std::result::Result<Vec<Array>, Exception> {
    traced(|| {
        let &[scale, dtype] = key else {
            bail!(
                "the conv-next region is keyed on two constants, got {}",
                key.len()
            );
        };
        regions::conv_next(float_of(scale), dtype_of(dtype)?, &refs(a))
    })
}

static CONV_RESIDUAL: LazyLock<KeyedCompiled> =
    LazyLock::new(|| KeyedCompiled::new(trace_conv_residual, false));
static CONV_NEXT: LazyLock<KeyedCompiled> =
    LazyLock::new(|| KeyedCompiled::new(trace_conv_next, false));

/// [`regions::conv_residual`], compiled.
pub(super) fn conv_residual(conv_scale: f32, args: &[&Array]) -> Result<[Array; 2]> {
    run("conv residual", &CONV_RESIDUAL, args, &[float(conv_scale)])
}

/// [`regions::conv_next`], compiled.
pub(super) fn conv_next(conv_scale: f32, dtype: Dtype, args: &[&Array]) -> Result<[Array; 2]> {
    run(
        "conv next",
        &CONV_NEXT,
        args,
        &[float(conv_scale), dtype as u64],
    )
}

// ---- the mixes -----------------------------------------------------------------------------------

fn trace_mix_messages(a: &[Array], key: &[u64]) -> std::result::Result<Vec<Array>, Exception> {
    traced(|| {
        let [x, message, down, gain_forget] = a else {
            bail!("the mix-messages region takes four arrays, got {}", a.len());
        };
        Ok([ops::mix_messages(
            x,
            message,
            down,
            gain_forget,
            single(key)? != 0,
        )?])
    })
}

fn trace_attn_res_mix(a: &[Array], key: &[u64]) -> std::result::Result<Vec<Array>, Exception> {
    traced(|| {
        let [partial_n, query, blocks @ ..] = a else {
            bail!("the attention-residual mix takes the partial sum, the query and the blocks");
        };
        Ok([ops::attn_res_mix(
            blocks,
            partial_n,
            query,
            float_of(single(key)?),
        )?])
    })
}

static MIX_MESSAGES: LazyLock<KeyedCompiled> =
    LazyLock::new(|| KeyedCompiled::new(trace_mix_messages, false));
static ATTN_RES_MIX: LazyLock<KeyedCompiled> =
    LazyLock::new(|| KeyedCompiled::new(trace_attn_res_mix, false));

/// [`ops::mix_messages`], compiled.
pub(super) fn mix_messages(
    x: &Array,
    message: &Array,
    down: &Array,
    gain_forget: &Array,
    round_bf16: bool,
) -> Result<Array> {
    let [mixed] = run(
        "mix messages",
        &MIX_MESSAGES,
        &[x, message, down, gain_forget],
        &[u64::from(round_bf16)],
    )?;
    Ok(mixed)
}

/// [`ops::attn_res_mix`], compiled.
pub(super) fn attn_res_mix(
    blocks: &[Array],
    partial_n: &Array,
    query: &Array,
    scale: f32,
) -> Result<Array> {
    let mut args: Vec<&Array> = vec![partial_n, query];
    args.extend(blocks);
    let [mixed] = run(
        "attention-residual mix",
        &ATTN_RES_MIX,
        &args,
        &[float(scale)],
    )?;
    Ok(mixed)
}
