//! The norm-and-residual glue of a Gemma 4 decoder layer as two Metal kernels.
//!
//! Between its weight kernels a decoder layer normalises and adds rows of the hidden width a dozen
//! times, each a launch of its own (`rms_norm`, `add`, `multiply`): after the attention, the
//! post-attention norm, the residual add, and the three input norms of the dense MLP, the router
//! and the experts; at the end, the two branches' output norms, their sum, the feed-forward norm,
//! the residual add, the layer scalar, and the next layer's input norm. In a decode step those are
//! 12 of a layer's 46 dispatches, each moving a few kilobytes. [`attn_out`] and [`layer_out`] are
//! the same arithmetic in one launch each.
//!
//! One threadgroup a row, laid out as MLX's `rms_single_row` lays out its launch: a thread for each
//! four consecutive values of the row, rounded up to a whole simdgroup. Every norm reduces as that
//! kernel does (a thread's four squares added one after the other, `acc += x * x` in one statement
//! as it writes it, `simd_sum`, then `simd_sum` over the simdgroups' totals), takes
//! `precise::rsqrt(acc / axis_size + eps)`, and rounds as it does (`w * T(x * inv)`); the adds and
//! the scalar multiply are bfloat16 operations, as MLX's binary kernels compute them. The kernels
//! therefore give MLX's result bit for bit, which the tests compare; that is why the flag is an
//! `Optimization`. They serve bfloat16 rows of at most [`MAX_ROWS`] rows (a decode step, an MTP
//! verify block) and widths `rms_norm` reduces in one threadgroup (a multiple of four up to 4096);
//! a prefill chunk and anything else keep the composed ops.
//!
//! [`serves`] vouches for a pass's input. What a layer computes from it has its shape but takes
//! the type of the weights' scales and of the routing scale, which a checkpoint may store wider
//! than bfloat16, so a caller asks [`serves_type`] of every state it is about to hand a kernel. A
//! wider one goes to [`attn_out_composed`] or [`layer_out_composed`], the ops MLX composes, which
//! promote it.

use std::cell::OnceCell;

use anyhow::{Context, Result, anyhow, ensure};
use mlx_rs::{Array, Dtype};

use crate::metal_kernel::{MetalKernel, MetalKernelConfig};
use crate::native_norm::rms_norm;

lumen_flags::flag! {
    /// Run the norm-and-residual glue of each Gemma 4 decoder layer as two Metal kernels: after
    /// the attention (post-attention norm, residual add, and the input norms of the dense MLP, the
    /// router and the experts) and at the end of the layer (both branches' output norms, their
    /// sum, the feed-forward norm, the residual add, the layer scalar and the next layer's input
    /// norm). Twelve dispatches a layer become two. Bit for bit what the composed ops give.
    /// Decode-sized passes only (an MTP verify block too); a prefill chunk keeps the composed ops.
    ///
    /// On by default: a decode token 0.45 ms (2.8%) faster on gemma-4-26b-a4b NVFP4 (M3 Max,
    /// 12 same-process ABBA blocks, paired standard error 0.017 ms), 301 fewer dispatches a token,
    /// the same continuation in 48 of 48 runs. `=0` restores the composed ops.
    pub(crate) norm_kernels {
        env: "LUMEN_GEMMA4_NORM_KERNELS",
        default: true,
        kind: Optimization,
    }
}

/// Whether the decoder layers' glue runs as kernels.
pub(crate) fn enabled() -> bool {
    norm_kernels::get()
}

/// The most rows a launch serves: decode steps and verify blocks, not prefill chunks.
pub(crate) const MAX_ROWS: i32 = 16;

/// The widest row `rms_norm` reduces in one threadgroup (`rms_single_row`); a wider one it loops,
/// adding in another order.
const MAX_HIDDEN: i32 = 4096;

/// The simdgroup width.
const SIMD: i32 = 32;

/// Threads of a row's threadgroup: one for each four values, rounded up to a simdgroup, as
/// `RMSNorm::eval_gpu` sizes its launch.
fn threads_for(hidden: i32) -> i32 {
    (hidden / 4 + SIMD - 1) / SIMD * SIMD
}

/// Whether the kernels serve a hidden state `x` (`[1, L, H]`): bfloat16, at most [`MAX_ROWS`]
/// rows, a width that is a multiple of four and at most 4096.
pub(crate) fn serves(x: &Array) -> bool {
    matches!(x.shape(), &[1, rows, hidden]
        if (1..=MAX_ROWS).contains(&rows) && hidden % 4 == 0 && (4..=MAX_HIDDEN).contains(&hidden))
        && serves_type(x)
}

/// Whether a state has the one type the kernels read and write. A layer asks it of each state it
/// computed from an input [`serves`] vouched for: the shape is the input's, the type need not be.
pub(crate) fn serves_type(state: &Array) -> bool {
    state.dtype() == Dtype::Bfloat16
}

/// The inverse RMS of a row, as `rms_single_row` reduces it: `acc` is this thread's sum of
/// squares. `sums` (32 floats) and `inv` (one) are threadgroup memory; every thread of the group
/// must call it, and may call it again at once (the barriers order the reuse).
const HEADER: &str = r#"
inline float row_inv_rms(
    float acc,
    uint lane,
    uint sg,
    uint axis_size,
    float eps,
    threadgroup float* sums,
    threadgroup float* inv) {
    acc = simd_sum(acc);
    if (sg == 0) {
        sums[lane] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        sums[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(sums[lane]);
        if (lane == 0) {
            inv[0] = metal::precise::rsqrt(acc / axis_size + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return inv[0];
}
"#;

/// The prologue both kernels share: this thread's place in its row. A width is a multiple of four,
/// so a thread holds all four of its values or none (`live`); one with none adds zeros, as
/// `rms_single_row`'s masked loads do.
const PROLOGUE: &str = r#"
    constexpr int N = 4;
    const uint row = threadgroup_position_in_grid.x;
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint axis_size = uint(r_shape[1]);
    const bool live = lid * N < axis_size;
    const size_t at = size_t(row) * axis_size + lid * N;
    const uint wat = lid * N;
    const float e = eps[0];
    threadgroup float sums[32];
    threadgroup float inv[1];
"#;

/// `a` is the attention's output and `r` the layer's input, both `[rows, H]`; `w_attn`, `w_dense`,
/// `w_router`, `w_moe` are `[H]`. Writes `h = r + rms_norm(a, w_attn)`, then `dense`, `router` and
/// `moe`: `h` normed with each of the three weights.
const ATTN_OUT_BODY: &str = r#"
    float t[N];
    float acc = 0;
    for (int i = 0; i < N; i++) {
        t[i] = live ? float(a[at + i]) : 0.0f;
        acc += t[i] * t[i];
    }
    const float inv_a = row_inv_rms(acc, lane, sg, axis_size, e, sums, inv);

    float th[N];
    float acc_h = 0;
    for (int i = 0; i < N; i++) {
        th[i] = 0.0f;
        if (live) {
            const bfloat16_t normed = w_attn[wat + i] * static_cast<bfloat16_t>(t[i] * inv_a);
            const bfloat16_t sum = r[at + i] + normed;
            h[at + i] = sum;
            th[i] = float(sum);
        }
        acc_h += th[i] * th[i];
    }
    const float inv_h = row_inv_rms(acc_h, lane, sg, axis_size, e, sums, inv);

    if (live) {
        for (int i = 0; i < N; i++) {
            dense[at + i] = w_dense[wat + i] * static_cast<bfloat16_t>(th[i] * inv_h);
            router[at + i] = w_router[wat + i] * static_cast<bfloat16_t>(th[i] * inv_h);
            moe[at + i] = w_moe[wat + i] * static_cast<bfloat16_t>(th[i] * inv_h);
        }
    }
"#;

/// `d` is the dense MLP's output, `m` the experts', `r` the residual stream after the attention,
/// all `[rows, H]`; `w1`, `w2`, `w_ff`, `w_next` are `[H]` and `s` one value. Writes
/// `out = (r + rms_norm(rms_norm(d, w1) + rms_norm(m, w2), w_ff)) * s` and `next`, `out` normed with
/// `w_next` (the next layer's input norm, or the final norm).
const LAYER_OUT_BODY: &str = r#"
    float td[N];
    float tm[N];
    float acc_d = 0;
    float acc_m = 0;
    for (int i = 0; i < N; i++) {
        td[i] = live ? float(d[at + i]) : 0.0f;
        acc_d += td[i] * td[i];
    }
    for (int i = 0; i < N; i++) {
        tm[i] = live ? float(m[at + i]) : 0.0f;
        acc_m += tm[i] * tm[i];
    }
    const float inv_d = row_inv_rms(acc_d, lane, sg, axis_size, e, sums, inv);
    const float inv_m = row_inv_rms(acc_m, lane, sg, axis_size, e, sums, inv);

    float ts[N];
    float acc_s = 0;
    for (int i = 0; i < N; i++) {
        ts[i] = 0.0f;
        if (live) {
            const bfloat16_t h1 = w1[wat + i] * static_cast<bfloat16_t>(td[i] * inv_d);
            const bfloat16_t h2 = w2[wat + i] * static_cast<bfloat16_t>(tm[i] * inv_m);
            const bfloat16_t sum = h1 + h2;
            ts[i] = float(sum);
        }
        acc_s += ts[i] * ts[i];
    }
    const float inv_s = row_inv_rms(acc_s, lane, sg, axis_size, e, sums, inv);

    const bfloat16_t scale = s[0];
    float to[N];
    float acc_o = 0;
    for (int i = 0; i < N; i++) {
        to[i] = 0.0f;
        if (live) {
            const bfloat16_t ff = w_ff[wat + i] * static_cast<bfloat16_t>(ts[i] * inv_s);
            const bfloat16_t with_residual = r[at + i] + ff;
            const bfloat16_t scaled = with_residual * scale;
            out[at + i] = scaled;
            to[i] = float(scaled);
        }
        acc_o += to[i] * to[i];
    }
    const float inv_o = row_inv_rms(acc_o, lane, sg, axis_size, e, sums, inv);

    if (live) {
        for (int i = 0; i < N; i++) {
            next[at + i] = w_next[wat + i] * static_cast<bfloat16_t>(to[i] * inv_o);
        }
    }
"#;

thread_local! {
    /// MLX keeps one stream per thread and the engine decodes on one thread, so each kernel is
    /// built once per thread rather than per call.
    static ATTN_OUT: OnceCell<MetalKernel> = const { OnceCell::new() };
    static LAYER_OUT: OnceCell<MetalKernel> = const { OnceCell::new() };
}

#[cfg(test)]
thread_local! {
    static LAUNCHES: std::cell::Cell<[usize; 2]> = const { std::cell::Cell::new([0; 2]) };
}

/// How many times each kernel (`attn_out`, `layer_out`) has run on this thread: a test's check
/// that both call sites were taken.
#[cfg(test)]
pub(crate) fn launches() -> [usize; 2] {
    LAUNCHES.with(std::cell::Cell::get)
}

/// The rows of a `[1, L, H]` hidden state, checked against the first input's.
fn rows_of(name: &str, x: &Array, like: &[i32]) -> Result<Array> {
    ensure!(
        x.dtype() == Dtype::Bfloat16 && x.shape() == like,
        "{name} must be bfloat16 {like:?}, got {:?} {:?}",
        x.dtype(),
        x.shape()
    );
    Ok(mlx_rs::ops::reshape(x, &[like[1], like[2]])?)
}

fn check_weight(name: &str, w: &Array, hidden: i32) -> Result<()> {
    ensure!(
        w.dtype() == Dtype::Bfloat16 && w.shape() == [hidden],
        "{name} must be bfloat16 [{hidden}], got {:?} {:?}",
        w.dtype(),
        w.shape()
    );
    Ok(())
}

fn check_eps(eps: &Array) -> Result<()> {
    ensure!(
        eps.dtype() == Dtype::Float32 && eps.size() == 1,
        "eps must be one float32, got {:?} {:?}",
        eps.dtype(),
        eps.shape()
    );
    Ok(())
}

/// One launch of `slot`'s kernel over `rows` rows of `hidden`, `outputs` outputs of `[rows,
/// hidden]` bfloat16, reshaped back to `[1, rows, hidden]`.
fn launch(
    slot: &'static std::thread::LocalKey<OnceCell<MetalKernel>>,
    build: impl FnOnce() -> Result<MetalKernel>,
    inputs: &[&Array],
    outputs: usize,
    rows: i32,
    hidden: i32,
) -> Result<Vec<Array>> {
    ensure!(
        (1..=MAX_ROWS).contains(&rows) && hidden % 4 == 0 && (4..=MAX_HIDDEN).contains(&hidden),
        "the norm kernels take 1 to {MAX_ROWS} rows of a width that is a multiple of 4 up to \
         {MAX_HIDDEN}, got {rows} rows of {hidden}"
    );
    let threads = threads_for(hidden);
    let outs = crate::metal_kernel::with_thread_kernel(slot, build, |kernel| {
        let config = MetalKernelConfig::new();
        config.set_grid(threads * rows, 1, 1)?;
        config.set_thread_group(threads, 1, 1)?;
        for _ in 0..outputs {
            config.add_output_arg(&[rows, hidden], Dtype::Bfloat16)?;
        }
        kernel
            .apply(inputs, &config, outputs)
            .context("a Gemma 4 norm kernel failed")
    })?;
    outs.into_iter()
        .map(|o| Ok(mlx_rs::ops::reshape(&o, &[1, rows, hidden])?))
        .collect()
}

/// After the attention, as one launch: `a` is the attention's output and `x` the layer's input
/// (`[1, L, H]` bfloat16 each), `weights` the post-attention norm, then the input norms of the dense
/// MLP, the router (its pre-scaled weight) and the experts (`[H]` bfloat16 each), `eps` one float32.
/// Returns `[h, dense_in, router_in, moe_in]`: `h = x + rms_norm(a, weights[0])`, then `h` normed
/// with each of the other three weights.
pub(crate) fn attn_out(
    a: &Array,
    x: &Array,
    weights: [&Array; 4],
    eps: &Array,
) -> Result<[Array; 4]> {
    let shape = x.shape().to_vec();
    let &[_, rows, hidden] = shape.as_slice() else {
        return Err(anyhow!("a hidden state is [1, L, H], got {shape:?}"));
    };
    let r = rows_of("the layer input", x, &shape)?;
    let a = rows_of("the attention output", a, &shape)?;
    for (name, w) in ["w_attn", "w_dense", "w_router", "w_moe"]
        .iter()
        .zip(weights)
    {
        check_weight(name, w, hidden)?;
    }
    check_eps(eps)?;
    #[cfg(test)]
    LAUNCHES.with(|n| {
        let mut counts = n.get();
        counts[0] += 1;
        n.set(counts);
    });
    let build = || {
        // The name is a literal: `cargo xtask test` reads the names off `MetalKernel::new`.
        MetalKernel::new_with_header(
            "gemma4_attn_out_norms",
            &["a", "r", "w_attn", "w_dense", "w_router", "w_moe", "eps"],
            &["h", "dense", "router", "moe"],
            HEADER,
            &format!("{PROLOGUE}{ATTN_OUT_BODY}"),
            /* ensure_row_contiguous */ true,
            /* atomic_outputs */ false,
        )
    };
    let [w_attn, w_dense, w_router, w_moe] = weights;
    let outs = launch(
        &ATTN_OUT,
        build,
        &[&a, &r, w_attn, w_dense, w_router, w_moe, eps],
        4,
        rows,
        hidden,
    )?;
    outs.try_into()
        .map_err(|_| anyhow!("the attention-out kernel returned the wrong number of outputs"))
}

/// The end of a decoder layer, as one launch: `d` is the dense MLP's output, `m` the experts', `h`
/// the residual stream after the attention (`[1, L, H]` bfloat16 each); `weights` the dense
/// branch's output norm, the experts' output norm, the feed-forward norm and the next norm (the next
/// layer's input norm, or the final norm), `[H]` bfloat16 each; `scalar` the layer scalar (one
/// bfloat16), `eps` one float32. Returns `[out, next]`: `out = (h + rms_norm(rms_norm(d, w1) +
/// rms_norm(m, w2), w_ff)) * scalar`, and `out` normed with the next norm.
pub(crate) fn layer_out(
    d: &Array,
    m: &Array,
    h: &Array,
    weights: [&Array; 4],
    scalar: &Array,
    eps: &Array,
) -> Result<[Array; 2]> {
    let shape = h.shape().to_vec();
    let &[_, rows, hidden] = shape.as_slice() else {
        return Err(anyhow!("a hidden state is [1, L, H], got {shape:?}"));
    };
    let r = rows_of("the residual stream", h, &shape)?;
    let d = rows_of("the dense output", d, &shape)?;
    let m = rows_of("the experts' output", m, &shape)?;
    for (name, w) in ["w1", "w2", "w_ff", "w_next"].iter().zip(weights) {
        check_weight(name, w, hidden)?;
    }
    ensure!(
        scalar.dtype() == Dtype::Bfloat16 && scalar.size() == 1,
        "the layer scalar must be one bfloat16, got {:?} {:?}",
        scalar.dtype(),
        scalar.shape()
    );
    check_eps(eps)?;
    #[cfg(test)]
    LAUNCHES.with(|n| {
        let mut counts = n.get();
        counts[1] += 1;
        n.set(counts);
    });
    let build = || {
        MetalKernel::new_with_header(
            "gemma4_layer_out_norms",
            &["d", "m", "r", "w1", "w2", "w_ff", "s", "w_next", "eps"],
            &["out", "next"],
            HEADER,
            &format!("{PROLOGUE}{LAYER_OUT_BODY}"),
            /* ensure_row_contiguous */ true,
            /* atomic_outputs */ false,
        )
    };
    let [w1, w2, w_ff, w_next] = weights;
    let outs = launch(
        &LAYER_OUT,
        build,
        &[&d, &m, &r, w1, w2, w_ff, scalar, w_next, eps],
        2,
        rows,
        hidden,
    )?;
    outs.try_into()
        .map_err(|_| anyhow!("the layer-out kernel returned the wrong number of outputs"))
}

/// [`attn_out`] as the ops `decoder_layer_forward` composes, for states the kernel does not serve
/// (see the module's note on wider states); they take any float type and promote as MLX does.
pub(crate) fn attn_out_composed(
    a: &Array,
    x: &Array,
    weights: [&Array; 4],
    eps: f32,
) -> Result<[Array; 4]> {
    let [w_attn, w_dense, w_router, w_moe] = weights;
    let normed = rms_norm(a, w_attn, eps).context("layer: post_attention_layernorm")?;
    let h = mlx_rs::ops::add(x, &normed).context("layer: +residual (attn)")?;
    let dense = rms_norm(&h, w_dense, eps).context("layer: pre_feedforward_layernorm")?;
    let router = rms_norm(&h, w_router, eps).context("router: rms_norm")?;
    let moe = rms_norm(&h, w_moe, eps).context("layer: pre_feedforward_layernorm_2")?;
    Ok([h, dense, router, moe])
}

/// [`layer_out`] as the composed ops, then the next norm, for states the kernel does not serve.
pub(crate) fn layer_out_composed(
    d: &Array,
    m: &Array,
    h: &Array,
    weights: [&Array; 4],
    scalar: &Array,
    eps: f32,
) -> Result<[Array; 2]> {
    let [w1, w2, w_ff, w_next] = weights;
    let h1 = rms_norm(d, w1, eps).context("layer: post_feedforward_layernorm_1")?;
    let h2 = rms_norm(m, w2, eps).context("layer: post_feedforward_layernorm_2")?;
    let sum = mlx_rs::ops::add(&h1, &h2).context("layer: h1 + h2")?;
    let ff = rms_norm(&sum, w_ff, eps).context("layer: post_feedforward_layernorm")?;
    let with_residual = mlx_rs::ops::add(h, &ff).context("layer: +residual (ff)")?;
    let out = mlx_rs::ops::multiply(&with_residual, scalar).context("layer: × layer_scalar")?;
    let next = rms_norm(&out, w_next, eps).context("layer: the next norm")?;
    Ok([out, next])
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    /// A reproducible stream of floats in `[-1, 1)`.
    fn noise(seed: u64, n: usize) -> Vec<f32> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 23) as f32) - 1.0
            })
            .collect()
    }

    /// `[shape]` bfloat16 values `offset + gain * noise`, with a zero every eleventh element and
    /// a value a thousand times larger every thirty-first (the roundings have edges to cross).
    fn values(seed: u64, shape: &[i32], offset: f32, gain: f32) -> Array {
        let n: i32 = shape.iter().product();
        let v: Vec<f32> = noise(seed, n as usize)
            .into_iter()
            .enumerate()
            .map(|(i, x)| match i {
                i if i % 11 == 0 => 0.0,
                i if i % 31 == 0 => offset + x * gain * 1000.0,
                _ => offset + x * gain,
            })
            .collect();
        Array::from_slice(&v, shape)
            .as_dtype(Dtype::Bfloat16)
            .expect("cast")
    }

    fn bits(a: &Array) -> Vec<u16> {
        let flat = mlx_rs::ops::reshape(a, &[-1]).unwrap();
        let wide = flat.as_dtype(Dtype::Float32).unwrap();
        wide.eval().unwrap();
        // A bfloat16 is the high half of its float32.
        wide.as_slice::<f32>()
            .iter()
            .map(|x| (x.to_bits() >> 16) as u16)
            .collect()
    }

    fn eps() -> Array {
        Array::from_slice(&[EPS], &[1])
    }

    fn assert_same(what: &str, got: &Array, want: &Array) {
        assert_eq!(got.shape(), want.shape(), "{what}: shape");
        assert_eq!(got.dtype(), want.dtype(), "{what}: dtype");
        let (g, w) = (bits(got), bits(want));
        let differ = g.iter().zip(&w).filter(|(a, b)| a != b).count();
        assert_eq!(differ, 0, "{what}: {differ} of {} values differ", w.len());
    }

    /// The composed ops `decoder_layer_forward` runs after the attention.
    fn attn_out_composed(a: &Array, x: &Array, w: [&Array; 4]) -> [Array; 4] {
        super::attn_out_composed(a, x, w, EPS).expect("the composed attention-out ops")
    }

    /// The composed ops at the end of a layer, then the next layer's input norm.
    fn layer_out_composed(
        d: &Array,
        m: &Array,
        h: &Array,
        w: [&Array; 4],
        scalar: &Array,
    ) -> [Array; 2] {
        super::layer_out_composed(d, m, h, w, scalar, EPS).expect("the composed layer-out ops")
    }

    /// Both kernels are the composed ops to the bit over widths that fill whole simdgroups (2816,
    /// 4096, 64) and ones that leave a simdgroup part empty (2880, 400, 1028), the row counts of a
    /// decode step and of verify blocks, and inputs whose scales differ by orders of magnitude.
    #[test]
    fn the_kernels_are_the_composed_ops_to_the_bit() {
        for hidden in [2816, 2880, 4096, 64, 400, 1028] {
            for rows in [1, 2, 5, 16] {
                for gain in [1.0f32, 0.01, 40.0] {
                    let seed = (hidden * 131 + rows) as u64;
                    let shape = [1, rows, hidden];
                    let x = values(seed, &shape, 0.0, gain);
                    let a = values(seed + 1, &shape, 0.0, gain * 3.0);
                    let ws: Vec<Array> = (0..8)
                        .map(|i| values(seed + 10 + i, &[hidden], 1.0, 0.5))
                        .collect();
                    let w4 = [&ws[0], &ws[1], &ws[2], &ws[3]];
                    let got = attn_out(&a, &x, w4, &eps()).expect("attn_out");
                    let want = attn_out_composed(&a, &x, w4);
                    for (i, name) in ["h", "dense_in", "router_in", "moe_in"].iter().enumerate() {
                        assert_same(
                            &format!("attn_out {name}, {rows} x {hidden}, gain {gain}"),
                            &got[i],
                            &want[i],
                        );
                    }

                    let d = values(seed + 2, &shape, 0.0, gain * 7.0);
                    let m = values(seed + 3, &shape, 0.0, gain * 0.3);
                    let scalar = values(seed + 4, &[1], 0.6, 0.1);
                    let w4 = [&ws[4], &ws[5], &ws[6], &ws[7]];
                    let got = layer_out(&d, &m, &x, w4, &scalar, &eps()).expect("layer_out");
                    let want = layer_out_composed(&d, &m, &x, w4, &scalar);
                    for (i, name) in ["out", "next"].iter().enumerate() {
                        assert_same(
                            &format!("layer_out {name}, {rows} x {hidden}, gain {gain}"),
                            &got[i],
                            &want[i],
                        );
                    }
                }
            }
        }
    }

    /// Thirty layers' worth of both kernels fed into each other, as a decode step chains them,
    /// stay the composed chain to the bit.
    #[test]
    fn a_chain_of_layers_stays_the_composed_ops_to_the_bit() {
        let (hidden, rows) = (2816, 1);
        let shape = [1, rows, hidden];
        let ws: Vec<Array> = (0..8)
            .map(|i| values(500 + i, &[hidden], 1.0, 0.5))
            .collect();
        let scalar = values(600, &[1], 0.8, 0.1);
        let mut x_kernel = values(700, &shape, 0.0, 1.0);
        let mut x_composed = x_kernel.clone();
        for layer in 0..30u64 {
            let a = values(800 + layer, &shape, 0.0, 2.0);
            let d = values(900 + layer, &shape, 0.0, 5.0);
            let m = values(1000 + layer, &shape, 0.0, 0.5);
            let [h_k, ..] = attn_out(&a, &x_kernel, [&ws[0], &ws[1], &ws[2], &ws[3]], &eps())
                .expect("attn_out");
            let [h_c, ..] = attn_out_composed(&a, &x_composed, [&ws[0], &ws[1], &ws[2], &ws[3]]);
            let [out_k, _] = layer_out(
                &d,
                &m,
                &h_k,
                [&ws[4], &ws[5], &ws[6], &ws[7]],
                &scalar,
                &eps(),
            )
            .expect("layer_out");
            let [out_c, _] =
                layer_out_composed(&d, &m, &h_c, [&ws[4], &ws[5], &ws[6], &ws[7]], &scalar);
            x_kernel = out_k;
            x_composed = out_c;
        }
        assert_same("30 chained layers", &x_kernel, &x_composed);
    }

    /// A state wider than bfloat16 (what a layer computes from float32 scales or a float32 routing
    /// scale) is not served, whichever input it is, and the composed ops take it without a launch
    /// and hand back the promoted type, as the layer's own ops would.
    #[test]
    fn a_wider_state_is_not_served_and_the_composed_ops_take_it() {
        let hidden = 2816;
        let ws: Vec<Array> = (0..4)
            .map(|i| values(700 + i, &[hidden], 1.0, 0.5))
            .collect();
        let w4 = [&ws[0], &ws[1], &ws[2], &ws[3]];
        let scalar = values(710, &[1], 0.8, 0.1);
        let x = values(720, &[1, 1, hidden], 0.0, 1.0);
        let wide = values(721, &[1, 1, hidden], 0.0, 3.0)
            .as_dtype(Dtype::Float32)
            .unwrap();
        assert!(serves(&x) && !serves(&wide));
        assert!(serves_type(&x) && !serves_type(&wide));

        let before = launches();
        for (a, r) in [(&wide, &x), (&x, &wide)] {
            assert!(attn_out(a, r, w4, &eps()).is_err(), "the kernel refuses it");
            for out in super::attn_out_composed(a, r, w4, EPS).expect("composed") {
                assert_eq!(out.dtype(), Dtype::Float32);
                assert_eq!(out.shape(), [1, 1, hidden]);
            }
        }
        for (d, m, h) in [(&wide, &x, &x), (&x, &wide, &x), (&x, &x, &wide)] {
            assert!(layer_out(d, m, h, w4, &scalar, &eps()).is_err());
            for out in super::layer_out_composed(d, m, h, w4, &scalar, EPS).expect("composed") {
                assert_eq!(out.dtype(), Dtype::Float32);
                assert_eq!(out.shape(), [1, 1, hidden]);
            }
        }
        assert_eq!(launches(), before, "a refused state launched a kernel");
    }

    /// What the kernels do not serve is an error, never a wrong answer.
    #[test]
    fn what_the_kernels_cannot_take_is_an_error() {
        let w = values(1, &[2816], 1.0, 0.5);
        let ws = [&w, &w, &w, &w];
        let x = values(2, &[1, 1, 2816], 0.0, 1.0);
        let wide = x.as_dtype(Dtype::Float32).unwrap();
        assert!(attn_out(&wide, &x, ws, &eps()).is_err(), "a float32 row");
        let tall = values(3, &[1, MAX_ROWS + 1, 2816], 0.0, 1.0);
        assert!(
            attn_out(&tall, &tall, ws, &eps()).is_err(),
            "a prefill chunk"
        );
        let w_wide = values(4, &[8192], 1.0, 0.5);
        let x_wide = values(5, &[1, 1, 8192], 0.0, 1.0);
        let ws_wide = [&w_wide, &w_wide, &w_wide, &w_wide];
        assert!(
            attn_out(&x_wide, &x_wide, ws_wide, &eps()).is_err(),
            "a looped width"
        );
        let odd = values(6, &[1, 1, 2814], 0.0, 1.0);
        let w_odd = values(7, &[2814], 1.0, 0.5);
        assert!(
            attn_out(&odd, &odd, [&w_odd, &w_odd, &w_odd, &w_odd], &eps()).is_err(),
            "a width that is not a multiple of 4"
        );
        let short = values(8, &[2815], 1.0, 0.5);
        assert!(
            attn_out(&x, &x, [&w, &short, &w, &w], &eps()).is_err(),
            "a wrong weight"
        );
        let scalar_f32 = Array::from_slice(&[1.0f32], &[1]);
        assert!(
            layer_out(&x, &x, &x, ws, &scalar_f32, &eps()).is_err(),
            "a float32 layer scalar"
        );
        assert!(!serves(&wide) && !serves(&tall) && !serves(&x_wide) && !serves(&odd));
        assert!(serves(&x));
    }
}
