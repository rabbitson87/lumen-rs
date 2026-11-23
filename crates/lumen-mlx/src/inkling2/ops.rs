//! Numerical building blocks of the Inkling 2 text model.
//!
//! Derived from `omlx/patches/inkling2/model.py` and `kernels.py` of oMLX 0.8.2 (Apache-2.0); the
//! comments name the function each one ports. Where the reference casts to float32 or rounds to
//! another dtype, so does the port at the same place: the dtype of every intermediate is part of
//! the contract (see `SPEC-step2-text.md`).
//!
//! The reference runs some of these as Metal kernels on the GPU (`attn_res_mix`, `route_topk`,
//! the flash attention of long prefill blocks) and as plain MLX ops otherwise. The port is the
//! plain-ops form of all of them.

use anyhow::{Result, ensure};
use mlx_rs::ops::indexing::{TryIndexOp, take_along_axis, take_axis};
use mlx_rs::{Array, Dtype};

use super::fused;

pub(super) fn scalar(v: f32) -> Array {
    Array::from_f32(v)
}

fn f32_of(x: &Array) -> Result<Array> {
    Ok(x.as_dtype(Dtype::Float32)?)
}

/// `1 / sqrt(taps + 1)`, the residual scale of the short convolutions.
pub(super) fn static_conv_scale(taps: usize) -> f32 {
    (1.0f64 / ((taps + 1) as f64).sqrt()) as f32
}

/// `1 / sqrt(2 * taps + 1)`, the same for the dynamic convolution of the global layers' k and v.
pub(super) fn dynamic_conv_scale(taps: usize) -> f32 {
    (1.0f64 / ((2 * taps + 1) as f64).sqrt()) as f32
}

/// `_unit_rms`: `mx.fast.rms_norm(x, None, 1e-6)`. The epsilon is the literal one, not the
/// config's `rms_norm_eps`.
pub(super) fn unit_rms(x: &Array) -> Result<Array> {
    let width = *x.shape().last().expect("rank >= 1");
    Ok(mlx_rs::fast::rms_norm(x, &unit_weight(width)?, 1e-6)?)
}

thread_local! {
    /// The all-ones weight of [`unit_rms`] by width: made once, not as a fill kernel per call.
    static UNIT_WEIGHTS: std::cell::RefCell<std::collections::HashMap<i32, Array>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

fn unit_weight(width: i32) -> Result<Array> {
    UNIT_WEIGHTS.with(|cache| {
        if let Some(ones) = cache.borrow().get(&width) {
            return Ok(ones.clone());
        }
        let ones = mlx_rs::ops::ones::<f32>(&[width])?;
        ones.eval()?;
        cache.borrow_mut().insert(width, ones.clone());
        Ok(ones)
    })
}

/// `_softcap`: `tanh(x / cap) * cap`.
pub(super) fn softcap(x: &Array, cap: f32) -> Result<Array> {
    let c = scalar(cap);
    let t = mlx_rs::ops::tanh(mlx_rs::ops::divide(x, &c)?)?;
    Ok(mlx_rs::ops::multiply(&t, &c)?)
}

/// `_l2_clip`: scale `x` by `scale`, then shrink each last-axis row to norm `threshold` if it is
/// longer. Computed in float32, returned in `x`'s dtype.
pub(super) fn l2_clip(x: &Array, scale: &Array, threshold: f32) -> Result<Array> {
    let xf = mlx_rs::ops::multiply(f32_of(x)?, scale)?;
    let squares = mlx_rs::ops::multiply(&xf, &xf)?;
    let norm = mlx_rs::ops::sqrt(mlx_rs::ops::sum_axis(&squares, -1, true)?)?;
    let denom = mlx_rs::ops::add(&norm, scalar(1e-6))?;
    let factor =
        mlx_rs::ops::minimum(mlx_rs::ops::divide(scalar(threshold), &denom)?, scalar(1.0))?;
    Ok(mlx_rs::ops::multiply(&xf, &factor)?.as_dtype(x.dtype())?)
}

/// `_softcapped_swiglu`: with `g` the raw gate and `u` the raw up projection,
/// `tanh(g / gcap) * gcap * sigmoid(g) * (tanh(u / ucap) * ucap)`, in float32, in `gate`'s dtype.
/// The sigmoid reads the raw gate, the first factor the capped one.
pub(super) fn softcapped_swiglu(gate: &Array, up: &Array, gcap: f32, ucap: f32) -> Result<Array> {
    if fused::enabled() {
        return fused::softcapped_swiglu(gate, up, &scalar(gcap), &scalar(ucap));
    }
    let dtype = gate.dtype();
    let g = f32_of(gate)?;
    let u = f32_of(up)?;
    let (gc, uc) = (scalar(gcap), scalar(ucap));
    let capped_gate =
        mlx_rs::ops::multiply(mlx_rs::ops::tanh(mlx_rs::ops::divide(&g, &gc)?)?, &gc)?;
    let capped_up = mlx_rs::ops::multiply(mlx_rs::ops::tanh(mlx_rs::ops::divide(&u, &uc)?)?, &uc)?;
    let act = mlx_rs::ops::multiply(
        mlx_rs::ops::multiply(&capped_gate, mlx_rs::ops::sigmoid(&g)?)?,
        &capped_up,
    )?;
    Ok(act.as_dtype(dtype)?)
}

/// A copy of `a` that owns its storage: a slice would keep the whole chunk-sized buffer it was cut
/// from alive for as long as the state it becomes (`x * 1`, exact for every float).
pub(super) fn owned(a: &Array) -> Result<Array> {
    Ok(mlx_rs::ops::multiply(a, scalar(1.0).as_dtype(a.dtype())?)?)
}

/// The last `taps - 1` rows (axis 1) of a convolution's padded input: its next state.
pub(super) fn next_conv_state(window: &Array, taps: usize) -> Result<Array> {
    let len = window.shape()[1];
    let keep = taps as i32 - 1;
    let state = window.try_index((.., (len - keep)..len, ..))?;
    // A decode-sized window is a few rows: the view pins nothing worth a copy kernel.
    if fused::enabled() && (len - keep) as usize <= fused::DECODE_ROWS {
        return Ok(state);
    }
    owned(&state)
}

/// `_static_conv`: a depthwise causal convolution with fixed taps. `w` is `[channels, taps]` and
/// its last tap meets the current token; `state` is the `taps - 1` rows before `x`. The result
/// is `(round(conv, w.dtype) + x) * scale` in `x`'s dtype, and the padded input window.
pub(super) fn static_conv(
    x: &Array,
    w: &Array,
    state: &Array,
    scale: f32,
) -> Result<(Array, Array)> {
    let xf = f32_of(x)?;
    let taps = w.shape()[1];
    let length = xf.shape()[1];
    let xp = mlx_rs::ops::concatenate_axis(&[state, &xf], 1)?;
    let tap = |j: i32| -> Result<Array> {
        let wj = w.try_index((.., j))?;
        let window = xp.try_index((.., j..j + length, ..))?;
        Ok(mlx_rs::ops::multiply(&wj, &window)?)
    };
    let mut out = tap(0)?;
    for j in 1..taps {
        out = mlx_rs::ops::add(&out, tap(j)?)?;
    }
    // The reference rounds the depthwise result to the weight dtype before the residual add.
    let rounded = f32_of(&out.as_dtype(w.dtype())?)?;
    let y = mlx_rs::ops::multiply(mlx_rs::ops::add(&rounded, &xf)?, scalar(scale))?;
    Ok((y.as_dtype(x.dtype())?, xp))
}

/// `_dynamic_conv`: per-token taps `static_w[lag] + dyn[head, lag]` over the k or v channels of a
/// global layer. `x` is `[B, L, C]`, `dyn_` `[B, L, heads * taps]`, `static_w` `[taps, C]`.
pub(super) fn dynamic_conv(
    x: &Array,
    dyn_: &Array,
    static_w: &Array,
    state: &Array,
    heads: i32,
    scale: f32,
) -> Result<(Array, Array)> {
    let xf = f32_of(x)?;
    let (b, length, c) = (xf.shape()[0], xf.shape()[1], xf.shape()[2]);
    let taps = static_w.shape()[0];
    let xp = mlx_rs::ops::concatenate_axis(&[state, &xf], 1)?;
    let dynamic = mlx_rs::ops::reshape(f32_of(dyn_)?, &[b, length, heads, 1, taps])?;
    let per_head = c / heads;
    let mut out: Option<Array> = None;
    for lag in 0..taps {
        let start = taps - 1 - lag;
        let window = mlx_rs::ops::reshape(
            xp.try_index((.., start..start + length, ..))?,
            &[b, length, heads, per_head],
        )?;
        let fixed = mlx_rs::ops::reshape(static_w.try_index(lag)?, &[heads, per_head])?;
        let tap = mlx_rs::ops::add(&fixed, dynamic.try_index((.., .., .., .., lag))?)?;
        let term = mlx_rs::ops::multiply(&window, &tap)?;
        out = Some(match out {
            None => term,
            Some(acc) => mlx_rs::ops::add(&acc, &term)?,
        });
    }
    let out = mlx_rs::ops::reshape(out.expect("taps >= 1"), &[b, length, c])?;
    let y = mlx_rs::ops::multiply(mlx_rs::ops::add(&out, &xf)?, scalar(scale))?;
    Ok((y.as_dtype(x.dtype())?, xp))
}

/// `attn_res_mix`: mix unit-RMS depth sources (`blocks` and the partial sum `partial_n`, each
/// `[B, L, H]` float32) with `query` (`[Q, H]`): a softmax over the sources of the scaled dots,
/// then the weighted sum. Returns `[B, L, Q, H]` float32.
pub(super) fn attn_res_mix(
    blocks: &[Array],
    partial_n: &Array,
    query: &Array,
    scale: f32,
) -> Result<Array> {
    let mut sources: Vec<&Array> = blocks.iter().collect();
    sources.push(partial_n);
    let src = mlx_rs::ops::stack_axis(&sources, 2)?; // [B, L, sources, H]
    let query_t = mlx_rs::ops::transpose_axes(query, &[1, 0])?;
    let logits = mlx_rs::ops::multiply(mlx_rs::ops::matmul(&src, &query_t)?, scalar(scale))?;
    let weights = mlx_rs::ops::softmax_axis(&logits, 2, Some(true))?; // [B, L, sources, Q]
    let weights_t = mlx_rs::ops::transpose_axes(&weights, &[0, 1, 3, 2])?;
    Ok(mlx_rs::ops::matmul(&weights_t, &src)?)
}

/// Up to this many rows [`mix_messages`] runs its projections one row at a time.
const BY_ROW_MAX: usize = 16;

/// `_mix_messages`: the loop controllers, `(1 - tanh(forget)) * x + tanh(gain) * message`, where
/// gain and forget come from a low-rank projection of the unit-RMS `x` and `message`.
///
/// `x` and `message` are `[N, B, L, H]` float32 (one row per controlled input), `down` is
/// `[N, 2, H, r]` and `gain_forget` `[N, r, 2H]`. The reference rounds the projection's input and
/// output through bf16 whatever the weight dtype. `round_bf16` keeps that (always, in serving); a
/// float32 parity replica turns it off because there the rounding is a staircase that two
/// implementations cross at different places (see the fixture script).
pub(super) fn mix_messages(
    x: &Array,
    message: &Array,
    down: &Array,
    gain_forget: &Array,
    round_bf16: bool,
) -> Result<Array> {
    // `a @ w` with the rows of `a` (axis 1) taken one at a time when there are few (the matrix-
    // vector product a one-token step runs: a block of rows gets from every row what a single step
    // gets, and avoids the matrix kernel a multi-row product takes, which is slower here and sums in
    // another order), together when there are many (a prefill, where the rows are the work).
    let project = |a: &Array, w: &Array| -> Result<Array> {
        let rows = a.shape()[1];
        if rows as usize > BY_ROW_MAX || rows == 1 {
            return Ok(mlx_rs::ops::matmul(a, w)?);
        }
        let parts = (0..rows)
            .map(|r| Ok(mlx_rs::ops::matmul(a.try_index((.., r..r + 1, ..))?, w)?))
            .collect::<Result<Vec<Array>>>()?;
        let refs: Vec<&Array> = parts.iter().collect();
        Ok(mlx_rs::ops::concatenate_axis(&refs, 1)?)
    };
    let (n, b, length, hidden) = {
        let s = x.shape();
        (s[0], s[1], s[2], s[3])
    };
    let rank = down.shape()[3];
    let rows = b * length;
    let through_bf16 = |a: Array| -> Result<Array> {
        if round_bf16 {
            Ok(a.as_dtype(Dtype::Bfloat16)?.as_dtype(Dtype::Float32)?)
        } else {
            Ok(a)
        }
    };
    let z = mlx_rs::ops::stack_axis(&[unit_rms(x)?, unit_rms(message)?], 1)?; // [N, 2, B, L, H]
    let z = mlx_rs::ops::reshape(through_bf16(z)?, &[2 * n, rows, hidden])?;
    // One batched matmul over the halves: a single [rows, 2H] x [2H, r] product gets a row-count
    // dependent kernel, so prefill chunking would change the result.
    let down = mlx_rs::ops::reshape(down, &[2 * n, hidden, rank])?;
    let low = mlx_rs::ops::reshape(project(&z, &down)?, &[n, 2, rows, rank])?;
    let low = through_bf16(mlx_rs::ops::sum_axis(&low, 1, false)?)?;
    let gates = mlx_rs::ops::reshape(project(&low, gain_forget)?, &[n, b, length, 2 * hidden])?;
    let gain = mlx_rs::ops::tanh(gates.try_index((.., .., .., ..hidden))?)?;
    let forget = mlx_rs::ops::tanh(gates.try_index((.., .., .., hidden..))?)?;
    let kept = mlx_rs::ops::multiply(mlx_rs::ops::subtract(scalar(1.0), &forget)?, x)?;
    Ok(mlx_rs::ops::add(
        &kept,
        mlx_rs::ops::multiply(&gain, message)?,
    )?)
}

/// `route_topk` (its plain-ops form): sigmoid scores, the `k` best by `score + bias`, their
/// unbiased scores renormalised to sum to `route_scale`. `logits` is `[N, E]` float32; returns
/// `([N, k] indices, [N, k] float32 weights)`.
pub(super) fn route_topk(
    logits: &Array,
    bias: &Array,
    k: i32,
    route_scale: f32,
) -> Result<(Array, Array)> {
    let scores = mlx_rs::ops::sigmoid(logits)?;
    let keyed = mlx_rs::ops::negative(mlx_rs::ops::add(&scores, bias)?)?;
    let order = mlx_rs::ops::argpartition_axis(&keyed, k - 1, -1)?;
    let inds = order.try_index((.., ..k))?;
    let picked = take_along_axis(&scores, &inds, -1)?;
    let total = mlx_rs::ops::add(mlx_rs::ops::sum_axis(&picked, -1, true)?, scalar(1e-20))?;
    let weights =
        mlx_rs::ops::multiply(mlx_rs::ops::divide(&picked, &total)?, scalar(route_scale))?;
    Ok((inds, weights))
}

/// Queries of a block above this many mask elements are split, so the mask stays below ~2 GiB of
/// float32 (`_MASK_BUDGET`).
const MASK_BUDGET: i64 = 1 << 29;

/// `_decode_attention`: one query (`B == 1`, `L == 1`) against the keys it can see, with MLX's
/// own `sinks` argument carrying the sink logit and an additive mask carrying the relative bias.
///
/// `rel` is `[1, 1, Hq, R]`, `sink` `[1, 1, Hq]`.
fn decode_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    rel: &Array,
    sink: &Array,
    scale: f32,
    window: usize,
) -> Result<Array> {
    let s = k.shape()[2];
    let lo = if window > 0 {
        (s - window as i32).max(0)
    } else {
        0
    };
    let (k, v) = if lo > 0 {
        (
            k.try_index((.., .., lo.., ..))?,
            v.try_index((.., .., lo.., ..))?,
        )
    } else {
        (k.clone(), v.clone())
    };
    let visible = s - lo;
    let r = rel.shape()[3];
    let near = r.min(visible);
    let hq = q.shape()[1];
    // Bias by key position, oldest first: the key at distance d sits d from the end.
    let by_distance = rel.try_index((0, 0, .., ..near))?; // [Hq, near]
    let newest_last: Vec<i32> = (0..near).rev().collect();
    let mut tail = take_axis(&by_distance, Array::from_slice(&newest_last, &[near]), 1)?;
    if near < visible {
        let pad = mlx_rs::ops::zeros_dtype(&[hq, visible - near], tail.dtype())?;
        tail = mlx_rs::ops::concatenate_axis(&[&pad, &tail], -1)?;
    }
    let mask = mlx_rs::ops::reshape(&tail, &[1, hq, 1, visible])?.as_dtype(q.dtype())?;
    let sinks = sink.try_index((0, 0, ..))?.as_dtype(q.dtype())?;
    Ok(mlx_rs::fast::scaled_dot_product_attention(
        q,
        &k,
        &v,
        scale,
        Some(mlx_rs::fast::ScaledDotProductAttentionMask::Array(&mask)),
        Some(&sinks),
    )?)
}

/// What the additive mask of one block of queries is built from before the layer's own bias comes
/// in: where each key sits behind each query, which keys a query may see, which lie inside the
/// bias band, and the constants the mask is filled with. It depends on the shapes and the window
/// only, so every layer visit of a kind in a decode step needs the same one: a decode-sized block
/// keeps the last few, and the 40-odd visits after the first take them instead of building them.
struct BandGeometry {
    /// `[B, Hq, nq, near]`: the position in the bias band each key reads, clipped into it.
    gather_at: Array,
    /// `[nq, near]`: the key is not after the query, and inside the window of a sliding layer.
    visible: Array,
    /// `[nq, near]`: the key lies inside the bias band.
    within: Array,
    /// `-inf` in the query's dtype.
    neg: Array,
    /// `0` in the query's dtype, and in the bias's.
    zero: Array,
    zero_bias: Array,
}

#[derive(Clone, PartialEq)]
struct BandKey {
    shape: [i32; 4],
    shift: i32,
    r: i32,
    window: i32,
    q_dtype: Dtype,
    bias_dtype: Dtype,
}

thread_local! {
    static BAND_GEOMETRY: std::cell::RefCell<Vec<(BandKey, std::rc::Rc<BandGeometry>)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// How many geometries a decode-sized block keeps: a sliding and a global layer's of this step.
const BAND_GEOMETRY_KEPT: usize = 4;

fn band_geometry(key: BandKey, cache: bool) -> Result<std::rc::Rc<BandGeometry>> {
    if cache
        && let Some(hit) = BAND_GEOMETRY.with(|c| {
            c.borrow()
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, g)| g.clone())
        })
    {
        return Ok(hit);
    }
    let [b, hq, nq, near] = key.shape;
    let (r, window) = (key.r, key.window);
    // dist[i, j]: how far key `f + j` lies behind query `i0 + i`.
    let rows = Array::arange::<_, i32>(None, nq, None)?;
    let cols = Array::arange::<_, i32>(None, near, None)?;
    let dist = mlx_rs::ops::subtract(
        mlx_rs::ops::add(
            mlx_rs::ops::reshape(&rows, &[nq, 1])?,
            Array::from_int(key.shift),
        )?,
        mlx_rs::ops::reshape(&cols, &[1, near])?,
    )?;
    let clipped = mlx_rs::ops::minimum(
        mlx_rs::ops::maximum(&dist, Array::from_int(0))?,
        Array::from_int(r - 1),
    )?;
    let mut visible = mlx_rs::ops::ge(&dist, Array::from_int(0))?;
    if window > 0 {
        visible =
            mlx_rs::ops::logical_and(&visible, mlx_rs::ops::lt(&dist, Array::from_int(window))?)?;
    }
    let geometry = std::rc::Rc::new(BandGeometry {
        gather_at: mlx_rs::ops::broadcast_to(&clipped, &[b, hq, nq, near])?,
        visible,
        within: mlx_rs::ops::lt(&dist, Array::from_int(r))?,
        neg: scalar(f32::NEG_INFINITY).as_dtype(key.q_dtype)?,
        zero: scalar(0.0).as_dtype(key.q_dtype)?,
        zero_bias: scalar(0.0).as_dtype(key.bias_dtype)?,
    });
    if cache {
        BAND_GEOMETRY.with(|c| {
            let mut c = c.borrow_mut();
            if c.len() >= BAND_GEOMETRY_KEPT {
                c.remove(0);
            }
            c.push((key, geometry.clone()));
        });
    }
    Ok(geometry)
}

thread_local! {
    static SINK_ROWS: std::cell::RefCell<Vec<((i32, i32, i32, Dtype), Array)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// The zero key or value row that stands for the sink in front of the keys: one per shape and
/// dtype, made once and evaluated, not a fill kernel on every layer visit.
fn sink_row(b: i32, hkv: i32, d: i32, dtype: Dtype) -> Result<Array> {
    SINK_ROWS.with(|rows| {
        let key = (b, hkv, d, dtype);
        if let Some((_, row)) = rows.borrow().iter().find(|(k, _)| *k == key) {
            return Ok(row.clone());
        }
        let row = mlx_rs::ops::zeros_dtype(&[b, hkv, 1, d], dtype)?;
        row.eval()?;
        rows.borrow_mut().push((key, row.clone()));
        Ok(row)
    })
}

/// Drops the kept geometries (the test of the sharing starts from none).
#[cfg(test)]
pub(super) fn forget_band_geometry() {
    BAND_GEOMETRY.with(|c| c.borrow_mut().clear());
}

/// `_masked_attention`: SDPA over an additive mask that carries the relative-bias band and the
/// sinks. A zero key and value row in front of the keys stands for the sink: its logit is the
/// mask's first column.
///
/// `q` is `[B, Hq, L, D]`, `k` and `v` `[B, Hkv, S, D]` (the queries are the last `L` keys), `rel`
/// `[B, L, Hq, R]`, `sink` `[B, L, Hq]`.
fn masked_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    rel: &Array,
    sink: &Array,
    scale: f32,
    window: usize,
) -> Result<Array> {
    let (b, hq, length, d) = {
        let s = q.shape();
        (s[0], s[1], s[2], s[3])
    };
    let s = k.shape()[2];
    let r = rel.shape()[3];
    let window_i = window as i32;
    if window > 0 {
        ensure!(window_i == r, "sliding layers expect rel_extent == window");
    }
    let lo = if window > 0 {
        (s - length - window_i + 1).max(0)
    } else {
        0
    };
    let hkv = k.shape()[1];
    let zero = sink_row(b, hkv, d, k.dtype())?;
    let kz = mlx_rs::ops::concatenate_axis(&[&zero, &k.try_index((.., .., lo.., ..))?], 2)?;
    let vz = mlx_rs::ops::concatenate_axis(&[&zero, &v.try_index((.., .., lo.., ..))?], 2)?;
    let nk = s - lo;
    let budget_rows = MASK_BUDGET / i64::from((b * hq * (nk + 1)).max(1));
    let qb = (budget_rows.min(i64::from(length)) as i32).max(16);
    let mut outs: Vec<Array> = Vec::new();
    let mut i0 = 0;
    while i0 < length {
        let i1 = length.min(i0 + qb);
        let nq = i1 - i0;
        let a0 = s - length + i0;
        let hi = s - length + i1;
        let f = lo.max(a0 - r + 1);
        let near = hi - f;
        let geo = band_geometry(
            BandKey {
                shape: [b, hq, nq, near],
                shift: a0 - f,
                r,
                window: window_i,
                q_dtype: q.dtype(),
                bias_dtype: rel.dtype(),
            },
            nq <= super::fused::DECODE_ROWS as i32,
        )?;
        let neg = &geo.neg;
        let rel_q =
            mlx_rs::ops::transpose_axes(&rel.try_index((.., i0..i1, .., ..))?, &[0, 2, 1, 3])?;
        let bias = take_along_axis(&rel_q, &geo.gather_at, -1)?;
        let in_band = mlx_rs::ops::r#where(&geo.within, &bias, &geo.zero_bias)?;
        let near_mask = mlx_rs::ops::r#where(&geo.visible, &in_band, neg)?;
        let sink_col = mlx_rs::ops::expand_dims(
            mlx_rs::ops::transpose_axes(&sink.try_index((.., i0..i1, ..))?, &[0, 2, 1])?,
            -1,
        )?
        .as_dtype(q.dtype())?;
        let mut parts: Vec<Array> = vec![sink_col];
        if f > lo {
            let fill = if window > 0 {
                neg.clone()
            } else {
                geo.zero.clone()
            };
            parts.push(mlx_rs::ops::broadcast_to(&fill, &[b, hq, nq, f - lo])?);
        }
        parts.push(near_mask.as_dtype(q.dtype())?);
        if hi < s {
            parts.push(mlx_rs::ops::broadcast_to(neg, &[b, hq, nq, s - hi])?);
        }
        let refs: Vec<&Array> = parts.iter().collect();
        let mask = mlx_rs::ops::concatenate_axis(&refs, -1)?;
        let out = mlx_rs::fast::scaled_dot_product_attention(
            q.try_index((.., .., i0..i1, ..))?,
            &kz,
            &vz,
            scale,
            Some(mlx_rs::fast::ScaledDotProductAttentionMask::Array(&mask)),
            None,
        )?;
        if qb < length {
            // A lazy graph would hold every block's mask at once; evaluated block by block, one
            // mask is alive.
            out.eval()?;
        }
        outs.push(out);
        i0 = i1;
    }
    if outs.len() == 1 {
        return Ok(outs.remove(0));
    }
    let refs: Vec<&Array> = outs.iter().collect();
    Ok(mlx_rs::ops::concatenate_axis(&refs, 2)?)
}

/// `banded_sink_attention`: attention with a relative bias band and one sink logit per query.
///
/// `q` is `[B, Hq, L, D]`; `k`, `v` `[B, Hkv, S, D]` with the queries the last `L` keys; `rel`
/// `[B, L, Hq, R]` is the bias by backward distance `d` (added where `d < R`); `sink` `[B, L, Hq]`
/// is the logit of a zero-valued sink key. `window > 0` limits the keys to `d < window`.
/// Returns `[B, Hq, L, D]`.
pub(super) fn banded_sink_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    rel: &Array,
    sink: &Array,
    scale: f32,
    window: usize,
) -> Result<Array> {
    if q.shape()[0] == 1 && q.shape()[2] == 1 {
        decode_attention(q, k, v, rel, sink, scale, window)
    } else {
        masked_attention(q, k, v, rel, sink, scale, window)
    }
}
