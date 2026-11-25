//! The local attention layer's projection region as one Metal kernel.
//!
//! After the quantized matmul of a local layer, [`regions::project_local`](super::regions::project_local)
//! runs the short convolutions of k and v, the L2 clips of q, k and the relative query, the value
//! norm, and the relative-bias matmul, and lays q, k, v out as `[B, H, L, D]`. [`fused`](super::fused)
//! compiles that for a decode step into 24 Metal dispatches, 40 times a token. This does everything
//! but the matmul in one launch; the matmul stays an MLX op, because its summation order is a GEMM's.
//!
//! One simdgroup (32 lanes) for each head row of each token row: a q head, a k head, a v head, and
//! one more for the relative query's heads. The sums are added in the order MLX adds them, so the
//! kernel is bit for bit the composed ops, and [`kernel`] an `Optimization`:
//!
//! * A clip's sum of squares is MLX's row reduction of the squares. A row of 64 or fewer values is
//!   `row_reduce_small`: one thread adds them one after another from zero (lane 0 does it here, and
//!   hands the sum to the others). A longer row (up to 512) is a simdgroup: each lane adds its four
//!   values, then the lanes are `simd_sum`med. The squares are rounded to float32 before they are
//!   added (a statement of their own: MLX's reduction reads them from memory).
//! * The value norm is `rms_norm`'s `rms_single_row`: a lane adds the squares of its four values
//!   (`acc += x * x` in one statement, as that kernel writes it), `simd_sum`, `precise::rsqrt` of the
//!   mean over the width read from the shape plus the epsilon, the row times it.
//! * The elementwise tail of a clip is MLX's: `precise::sqrt`, the epsilon added, the threshold
//!   divided, the minimum with one taking a NaN first, the row times the factor, rounded to the
//!   activation dtype.
//! * The convolution is [`conv_kernel`](super::conv_kernel)'s, for one channel at a time: the taps
//!   added in the composed order, the sum rounded to the weight dtype, the row's dtype for the result.
//!
//! It takes the whole projection output of the layer, as the three arrays q and r, k and v come from
//! (the same array three times when one input feeds all three, the thirds of a stacked projection in
//! a looped pass), and writes the two padded convolution windows beside q, k, v and the clipped
//! relative query.

use std::cell::OnceCell;

use anyhow::{Context, Result, anyhow, ensure};
use mlx_rs::{Array, Dtype};

use super::regions::ProjCfg;
use crate::metal_kernel::{MetalKernel, MetalKernelConfig};

lumen_flags::flag! {
    /// Run a local attention layer's projection region (the short convolutions of k and v, the L2
    /// clips of q, k and the relative query, the value norm, the layout of q, k, v) as one Metal
    /// kernel in place of the compiled composed ops: 24 dispatches become one (and the relative
    /// bias matmul after it). Bit for bit what the composed ops give (the kernel adds every sum in
    /// the order MLX's reductions do, and the tests compare bits). Decode-sized passes only (a verify
    /// block too): a prefill chunk keeps the composed ops.
    ///
    /// A decode token about 0.7 ms (2.5%) faster: 10 same-session runs an arm against the version
    /// without it, 0.73 ms (paired standard error 0.03) by mean wall time and 0.71 ms by the median of
    /// steps; 880 fewer dispatches a token (22 for each of the 40 local layer visits).
    pub(crate) kernel {
        env: "LUMEN_INKLING2_PROJECT_KERNEL",
        default: true,
        kind: Optimization,
    }
}

/// Whether the projection runs as a kernel.
pub(super) fn enabled() -> bool {
    kernel::get()
}

#[cfg(test)]
thread_local! {
    static LAUNCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// How many times the kernel has been launched on this thread (a test's check that it ran).
#[cfg(test)]
pub(super) fn launches() -> usize {
    LAUNCHES.with(std::cell::Cell::get)
}

/// Lanes of a simdgroup: a head row is one.
const SIMD: i32 = 32;

/// The widest head row a simdgroup reduces the way MLX does (one block of 32 lanes by four values) and
/// the value norm's single simdgroup.
const MAX_HEAD_DIM: i32 = 128;

/// The longest row MLX reduces with one thread adding the values one after another.
const MAX_SMALL_ROW: i32 = 64;

/// Threads of a threadgroup: four simdgroups.
const GROUP: i32 = 128;

/// Float conversions and the helpers the kernel body calls: a store that converts to the output's
/// type, a rounding to the type of the array a pointer points into, the clip's factor, and one
/// channel of a short convolution at one token row (the taps added in the composed order).
const HEADER: &str = r#"
template <typename T>
inline void store(device T* p, int i, float v) {
    p[i] = static_cast<T>(v);
}
template <typename T>
inline float round_as(const device T* proxy, float v) {
    return float(static_cast<T>(v));
}
inline float clip_factor(float sum, float threshold) {
    const float norm = metal::precise::sqrt(sum);
    const float denom = norm + 1e-6f;
    const float ratio = threshold / denom;
    if (metal::isnan(ratio)) {
        return ratio;
    }
    return ratio < 1.0f ? ratio : 1.0f;
}
template <typename X, typename W>
inline float conv_y(
    const device X* x, int base, int stride, int row, int channel,
    const device W* w, const device float* state, int channels, int taps, float scale) {
    float acc = 0.0f;
    float current = 0.0f;
    for (int j = 0; j < taps; ++j) {
        const int t = row + j;
        const float v = t < taps - 1
            ? state[t * channels + channel]
            : float(x[(t - (taps - 1)) * stride + base + channel]);
        if (j == taps - 1) {
            current = v;
        }
        const float term = float(w[channel * taps + j]) * v;
        acc = j == 0 ? term : acc + term;
    }
    return round_as(x, (round_as(w, acc) + current) * scale);
}
"#;

/// `sq`, `sk`, `sv` are `[rows, TOTAL]` projection outputs: q and the relative query come from `sq`
/// (columns `0..C0` and `C2..`), k from `sk` (columns `C0..C1`), v from `sv` (columns `C1..C2`).
/// `k_conv`, `v_conv` are `[DKV, TAPS]`, `k_state`, `v_state` `[TAPS - 1, DKV]` float32, `q_scale`
/// `[NH, HD]`, `r_scale` `[NH, DR]`. Outputs: `q_out` `[NH, rows, HD]`, `k_out`, `v_out`
/// `[KVH, rows, HD]`, `r_out` the clipped relative query `[rows, NH, DR]`, `k_window`, `v_window`
/// `[TAPS - 1 + rows, DKV]` float32.
const SOURCE: &str = r#"
    constexpr int DKV = KVH * HD;
    constexpr int TASKS = NH + 2 * KVH + 1;
    const int task = thread_position_in_grid.x / 32;
    const int lane = thread_index_in_simdgroup;
    const int rows = sq_shape[0];
    if (task >= rows * TASKS) {
        return;
    }
    const int r = task / TASKS;
    const int t = task % TASKS;

    // The elements of a head row this lane owns, in the order MLX's row reduction adds them.
    int own[8];
    int n_own = 0;
    if (HD <= 64) {
        for (int e = lane; e < HD; e += 32) {
            own[n_own++] = e;
        }
    } else {
        for (int b = 0; b < HD / 128; ++b) {
            for (int i = 0; i < 4; ++i) {
                own[n_own++] = b * 128 + lane * 4 + i;
            }
        }
        const int extra = HD - (HD / 128) * 128;
        const int index = lane * 4;
        const int base = (HD / 128) * 128 + index;
        if (index + 4 <= extra) {
            for (int i = 0; i < 4; ++i) {
                own[n_own++] = base + i;
            }
        } else {
            for (int i = 0; index + i < extra; ++i) {
                own[n_own++] = base + i;
            }
        }
    }

#define X_Q(E) (float(sq[r * TOTAL + h * HD + (E)]) * float(q_scale[h * HD + (E)]))
#define X_K(E) conv_y(sk, C0, TOTAL, r, h * HD + (E), k_conv, k_state, DKV, TAPS, scale)
#define X_V(E) conv_y(sv, C1, TOTAL, r, h * HD + (E), v_conv, v_state, DKV, TAPS, scale)

// The sum of the squares of a head row, as MLX's row reduction adds them.
#define SUM_OF_SQUARES(XE, S)                  \
    float S = 0.0f;                            \
    if (HD <= 64) {                            \
        if (lane == 0) {                       \
            for (int e = 0; e < HD; ++e) {     \
                const float x = XE(e);         \
                const float sqr = x * x;       \
                S = sqr + S;                   \
            }                                  \
        }                                      \
        S = simd_broadcast_first(S);           \
    } else {                                   \
        for (int m = 0; m < n_own; ++m) {      \
            const float x = XE(own[m]);        \
            const float sqr = x * x;           \
            S = sqr + S;                       \
        }                                      \
        S = simd_sum(S);                       \
    }

    if (t < NH) {
        const int h = t;
        SUM_OF_SQUARES(X_Q, total)
        const float factor = clip_factor(total, qk_clip);
        for (int m = 0; m < n_own; ++m) {
            const int e = own[m];
            store(q_out, (h * rows + r) * HD + e, X_Q(e) * factor);
        }
    } else if (t < NH + KVH) {
        const int h = t - NH;
        SUM_OF_SQUARES(X_K, total)
        const float factor = clip_factor(total, qk_clip);
        for (int m = 0; m < n_own; ++m) {
            const int e = own[m];
            const int ch = h * HD + e;
            store(k_out, (h * rows + r) * HD + e, X_K(e) * factor);
            k_window[(TAPS - 1 + r) * DKV + ch] = float(sk[r * TOTAL + C0 + ch]);
            if (r == 0) {
                for (int tt = 0; tt < TAPS - 1; ++tt) {
                    k_window[tt * DKV + ch] = k_state[tt * DKV + ch];
                }
            }
        }
    } else if (t < NH + 2 * KVH) {
        // The value norm: `rms_norm`'s single row, a lane holding four values.
        const int h = t - NH - KVH;
        float acc = 0.0f;
        float xs[4];
        for (int i = 0; i < 4; ++i) {
            const int e = lane * 4 + i;
            xs[i] = e < HD ? X_V(e) : 0.0f;
            acc += xs[i] * xs[i];
        }
        acc = simd_sum(acc);
        const float inv = metal::precise::rsqrt(acc / float(q_scale_shape[1]) + 1e-6f);
        for (int i = 0; i < 4; ++i) {
            const int e = lane * 4 + i;
            if (e < HD) {
                const int ch = h * HD + e;
                store(v_out, (h * rows + r) * HD + e, xs[i] * inv);
                v_window[(TAPS - 1 + r) * DKV + ch] = float(sv[r * TOTAL + C1 + ch]);
                if (r == 0) {
                    for (int tt = 0; tt < TAPS - 1; ++tt) {
                        v_window[tt * DKV + ch] = v_state[tt * DKV + ch];
                    }
                }
            }
        }
    } else {
        // The relative query's heads, a lane for each head, the row added one value after another.
        for (int h = lane; h < NH; h += 32) {
            float total = 0.0f;
            for (int e = 0; e < DR; ++e) {
                const float x = float(sq[r * TOTAL + C2 + h * DR + e]) * float(r_scale[h * DR + e]);
                const float sqr = x * x;
                total = sqr + total;
            }
            const float factor = clip_factor(total, rel_clip);
            for (int e = 0; e < DR; ++e) {
                const float x = float(sq[r * TOTAL + C2 + h * DR + e]) * float(r_scale[h * DR + e]);
                store(r_out, (r * NH + h) * DR + e, x * factor);
            }
        }
    }
"#;

thread_local! {
    /// MLX keeps one stream per thread and the engine decodes on one thread, so the kernel is
    /// built once per thread rather than per call.
    static KERNEL: OnceCell<MetalKernel> = const { OnceCell::new() };
}

fn with_kernel<R>(f: impl FnOnce(&MetalKernel) -> Result<R>) -> Result<R> {
    KERNEL.with(|cell| {
        if cell.get().is_none() {
            // The name is a literal: `cargo xtask test` reads the names off `MetalKernel::new`.
            let built = MetalKernel::new_with_header(
                "inkling2_project_local",
                &[
                    "sq", "sk", "sv", "k_conv", "v_conv", "k_state", "v_state", "q_scale",
                    "r_scale", "scale", "qk_clip", "rel_clip",
                ],
                &["q_out", "k_out", "v_out", "r_out", "k_window", "v_window"],
                HEADER,
                SOURCE,
                /* ensure_row_contiguous */ true,
                /* atomic_outputs */ false,
            )?;
            let _ = cell.set(built);
        }
        f(cell
            .get()
            .ok_or_else(|| anyhow!("the projection kernel is missing"))?)
    })
}

fn is_float(d: Dtype) -> bool {
    matches!(d, Dtype::Float32 | Dtype::Bfloat16 | Dtype::Float16)
}

/// The arrays a local layer's projection reads besides the three projection outputs, in the order
/// [`project_local`] takes them: `[k_state, v_state, k_conv, v_conv, q_scale, r_scale, rel_proj]`.
pub(super) const FIXED_ARRAYS: usize = 7;

/// Whether the kernel can serve a layer of this geometry and these arrays: a batch of one, head rows it
/// reduces the way MLX does, arrays big enough to be device buffers, float dtypes. A layer it cannot
/// serve keeps the compiled region.
pub(super) fn serves(cfg: &ProjCfg, sources: [&Array; 3], fixed: &[&Array]) -> bool {
    fixed.len() == FIXED_ARRAYS
        && cfg.head_dim % 4 == 0
        && (4..=MAX_HEAD_DIM).contains(&cfg.head_dim)
        && (1..=MAX_SMALL_ROW).contains(&cfg.d_rel)
        && cfg.heads >= 1
        && cfg.kv_heads >= 1
        && sources.iter().all(|s| {
            s.shape().len() == 3 && s.shape()[0] == 1 && is_float(s.dtype()) && s.size() >= 8
        })
        && fixed.iter().all(|a| a.size() >= 8)
}

/// One launch plus the relative-bias matmul: [`regions::project_local`](super::regions::project_local)
/// for a layer the kernel serves. `sources` are `[q and r from, k from, v from]`, each `[1, L, TOTAL]`
/// (the same array three times when one input feeds all three); `cuts` are where the projection
/// splits (`[dq, dq + dkv, dq + 2 dkv, dq + 2 dkv + dr, ..]`); `fixed` is `[k_state, v_state, k_conv,
/// v_conv, q_scale, r_scale, rel_proj]`. Returns `[q, k, v, rel, k_window, v_window]`.
pub(super) fn project_local(
    cfg: &ProjCfg,
    cuts: &[i32],
    sources: [&Array; 3],
    fixed: &[&Array],
) -> Result<[Array; 6]> {
    ensure!(
        serves(cfg, sources, fixed),
        "the projection kernel cannot serve this layer: {cfg:?}"
    );
    let [k_state, v_state, k_conv, v_conv, q_scale, r_scale, rel_proj] = fixed else {
        return Err(anyhow!("the projection takes {FIXED_ARRAYS} fixed arrays"));
    };
    let &[c0, c1, c2, ..] = cuts else {
        return Err(anyhow!("the projection splits at four or more columns"));
    };
    let (nh, nkv, hd, dr) = (cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.d_rel);
    let dkv = nkv * hd;
    let &[_, rows, total] = sources[0].shape() else {
        return Err(anyhow!("a projection output is [1, L, C]"));
    };
    for s in sources {
        ensure!(
            s.shape() == [1, rows, total] && s.dtype() == sources[0].dtype(),
            "the projection outputs are {:?} {:?}, got {:?} {:?}",
            sources[0].dtype(),
            sources[0].shape(),
            s.dtype(),
            s.shape()
        );
    }
    ensure!(
        c0 == nh * hd && c1 == c0 + dkv && c2 == c1 + dkv && c2 + nh * dr <= total,
        "the projection splits at {cuts:?}, which is not {nh} heads of {hd}, {nkv} of k and v, {nh} relative rows of {dr}"
    );
    ensure!(
        k_state.shape() == [1, k_conv.shape()[1] - 1, dkv]
            && v_state.shape() == k_state.shape()
            && k_state.dtype() == Dtype::Float32
            && v_state.dtype() == Dtype::Float32
            && k_conv.shape() == [dkv, k_conv.shape()[1]]
            && v_conv.shape() == k_conv.shape()
            && is_float(k_conv.dtype())
            && is_float(v_conv.dtype())
            && q_scale.shape() == [nh, hd]
            && r_scale.shape() == [nh, dr]
            && is_float(q_scale.dtype())
            && is_float(r_scale.dtype()),
        "the projection's states, taps and scales do not fit {cfg:?}"
    );
    let taps = k_conv.shape()[1];
    ensure!(
        (1..=16).contains(&taps),
        "the projection takes at most 16 taps, got {taps}"
    );

    #[cfg(test)]
    LAUNCHES.with(|n| n.set(n.get() + 1));
    let flat = |a: &Array| mlx_rs::ops::reshape(a, &[rows, total]);
    let (sq, sk, sv) = (flat(sources[0])?, flat(sources[1])?, flat(sources[2])?);
    let k_state2 = mlx_rs::ops::reshape(k_state, &[taps - 1, dkv])?;
    let v_state2 = mlx_rs::ops::reshape(v_state, &[taps - 1, dkv])?;
    let scale = Array::from_f32(cfg.conv_scale);
    let (qk_clip, rel_clip) = (Array::from_f32(cfg.qk_clip), Array::from_f32(cfg.rel_clip));
    let dtype = sources[0].dtype();
    let mut outs = with_kernel(|kernel| {
        let config = MetalKernelConfig::new();
        for (name, value) in [
            ("NH", nh),
            ("KVH", nkv),
            ("HD", hd),
            ("DR", dr),
            ("TAPS", taps),
            ("TOTAL", total),
            ("C0", c0),
            ("C1", c1),
            ("C2", c2),
        ] {
            config.add_template_arg_int(name, value)?;
        }
        let tasks = rows * (nh + 2 * nkv + 1);
        let threads = tasks * SIMD;
        config.set_grid(threads, 1, 1)?;
        config.set_thread_group(GROUP.min(threads), 1, 1)?;
        config.add_output_arg(&[nh, rows, hd], dtype)?;
        config.add_output_arg(&[nkv, rows, hd], dtype)?;
        config.add_output_arg(&[nkv, rows, hd], dtype)?;
        config.add_output_arg(&[rows, nh, dr], dtype)?;
        config.add_output_arg(&[taps - 1 + rows, dkv], Dtype::Float32)?;
        config.add_output_arg(&[taps - 1 + rows, dkv], Dtype::Float32)?;
        kernel
            .apply(
                &[
                    &sq, &sk, &sv, k_conv, v_conv, &k_state2, &v_state2, q_scale, r_scale, &scale,
                    &qk_clip, &rel_clip,
                ],
                &config,
                6,
            )
            .context("the projection kernel failed")
    })?;
    let v_window = outs.swap_remove(5);
    let k_window = outs.swap_remove(4);
    let r = outs.swap_remove(3);
    let v = outs.swap_remove(2);
    let k = outs.swap_remove(1);
    let q = outs.swap_remove(0);
    let rel = mlx_rs::ops::matmul(
        mlx_rs::ops::reshape(r, &[1, rows, nh, dr])?,
        rel_proj.as_dtype(dtype)?,
    )?;
    Ok([
        mlx_rs::ops::reshape(q, &[1, nh, rows, hd])?,
        mlx_rs::ops::reshape(k, &[1, nkv, rows, hd])?,
        mlx_rs::ops::reshape(v, &[1, nkv, rows, hd])?,
        rel,
        mlx_rs::ops::reshape(k_window, &[1, taps - 1 + rows, dkv])?,
        mlx_rs::ops::reshape(v_window, &[1, taps - 1 + rows, dkv])?,
    ])
}

#[cfg(test)]
mod tests {
    use mlx_rs::ops::indexing::TryIndexOp;

    use super::super::regions;
    use super::*;

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

    /// `[shape]` of values around `gain`, in `dtype`: a zero every thirteenth element, and the gain
    /// grown for every third group of eight so that some rows are clipped and some are not.
    fn values(seed: u64, shape: &[i32], gain: f32, dtype: Dtype) -> Array {
        let n: i32 = shape.iter().product();
        let v: Vec<f32> = noise(seed, n as usize)
            .into_iter()
            .enumerate()
            .map(|(i, x)| match i {
                i if i % 13 == 0 => 0.0,
                i if (i / 8) % 3 == 0 => x * gain * 9.0,
                _ => x * gain,
            })
            .collect();
        Array::from_slice(&v, shape).as_dtype(dtype).expect("cast")
    }

    fn bits(a: &Array) -> Vec<u32> {
        let flat = mlx_rs::ops::reshape(a.as_dtype(Dtype::Float32).unwrap(), &[-1]).unwrap();
        flat.eval().unwrap();
        flat.as_slice::<f32>().iter().map(|x| x.to_bits()).collect()
    }

    /// A layer's geometry: heads, kv heads, head dim, relative width, relative extent.
    #[derive(Clone, Copy, Debug)]
    struct Geometry(i32, i32, i32, i32, i32);

    /// Everything a region call reads for `rows` token rows: the three projection outputs (the same
    /// array three times when `stacked` is false), the states, taps and scales.
    struct Case {
        cfg: ProjCfg,
        cuts: Vec<i32>,
        sources: [Array; 3],
        fixed: Vec<Array>,
    }

    fn case(g: Geometry, rows: i32, dtype: Dtype, stacked: bool, seed: u64) -> Case {
        let Geometry(nh, nkv, hd, dr, ext) = g;
        let (dq, dkv, drr) = (nh * hd, nkv * hd, nh * dr);
        let total = dq + 2 * dkv + drr + dq + nh;
        let big = |s: u64| values(seed + s, &[1, rows, total], 1.0, dtype);
        let (a, b, c) = (big(1), big(2), big(3));
        let sources = if stacked {
            [a, b, c]
        } else {
            [a.clone(), a.clone(), a]
        };
        let taps = 4;
        let fixed = vec![
            values(seed + 4, &[1, taps - 1, dkv], 2.0, Dtype::Float32),
            values(seed + 5, &[1, taps - 1, dkv], 2.0, Dtype::Float32),
            values(seed + 6, &[dkv, taps], 0.2, dtype),
            values(seed + 7, &[dkv, taps], 0.2, dtype),
            values(seed + 8, &[nh, hd], 0.4, dtype),
            values(seed + 9, &[nh, dr], 0.4, dtype),
            values(seed + 10, &[dr, ext], 0.3, dtype),
        ];
        Case {
            cfg: ProjCfg {
                heads: nh,
                kv_heads: nkv,
                head_dim: hd,
                d_rel: dr,
                qk_clip: 3.0 * (hd as f32).sqrt() / 2.0,
                rel_clip: 8.0,
                conv_scale: (1.0f64 / 5.0f64.sqrt()) as f32,
            },
            cuts: vec![dq, dq + dkv, dq + 2 * dkv, dq + 2 * dkv + drr, total - nh],
            sources,
            fixed,
        }
    }

    fn reference(c: &Case) -> [Array; 6] {
        let [a, b, k_src] = &c.sources;
        let cuts = &c.cuts;
        let q = a.try_index((.., .., ..cuts[0])).unwrap();
        let k = b.try_index((.., .., cuts[0]..cuts[1])).unwrap();
        let v = k_src.try_index((.., .., cuts[1]..cuts[2])).unwrap();
        let r = a.try_index((.., .., cuts[2]..cuts[3])).unwrap();
        let mut args: Vec<&Array> = vec![&q, &k, &v, &r];
        args.extend(c.fixed.iter());
        regions::project_local(&c.cfg, &args).unwrap()
    }

    fn kernel_run(c: &Case) -> [Array; 6] {
        let [a, b, d] = &c.sources;
        let fixed: Vec<&Array> = c.fixed.iter().collect();
        project_local(&c.cfg, &c.cuts, [a, b, d], &fixed).unwrap()
    }

    /// The kernel is the composed projection to the bit: all six results (q, k, v, the relative
    /// scores, the two windows) for the real model's local layer and for other geometries (head rows of
    /// 4 to 128 values: the single-thread reduction, the simdgroup's, one with idle lanes), one row
    /// and several, bfloat16 and float32, one input feeding the three and three different inputs (a
    /// looped pass's stacked projection).
    #[test]
    fn the_kernel_is_the_composed_projection_to_the_bit() {
        let names = ["q", "k", "v", "rel", "k window", "v window"];
        for g in [
            Geometry(24, 8, 128, 16, 1024),
            Geometry(4, 2, 32, 4, 24),
            Geometry(6, 3, 64, 8, 16),
            Geometry(8, 2, 96, 64, 32),
            Geometry(5, 5, 128, 16, 40),
            Geometry(2, 1, 8, 4, 8),
        ] {
            for rows in [1, 2, 5, 16] {
                for dtype in [Dtype::Bfloat16, Dtype::Float32] {
                    for stacked in [false, true] {
                        let c = case(g, rows, dtype, stacked, 100 * rows as u64);
                        let (want, got) = (reference(&c), kernel_run(&c));
                        for (at, name) in names.iter().enumerate() {
                            assert_eq!(
                                want[at].shape(),
                                got[at].shape(),
                                "{g:?} rows {rows} {dtype:?} stacked {stacked}: {name} shape"
                            );
                            assert_eq!(
                                want[at].dtype(),
                                got[at].dtype(),
                                "{g:?} rows {rows} {dtype:?} stacked {stacked}: {name} dtype"
                            );
                            let (w, k) = (bits(&want[at]), bits(&got[at]));
                            let moved = w.iter().zip(&k).filter(|(a, b)| a != b).count();
                            assert_eq!(
                                moved,
                                0,
                                "{g:?} rows {rows} {dtype:?} stacked {stacked}: {name} differs in {moved} of {} elements",
                                w.len()
                            );
                        }
                    }
                }
            }
        }
    }

    /// A call's windows are the next call's states: chained over many steps the kernel stays the
    /// composed ops'.
    #[test]
    fn a_chain_of_calls_stays_the_composed_projection_to_the_bit() {
        let g = Geometry(24, 8, 128, 16, 64);
        let mut want_case = case(g, 1, Dtype::Bfloat16, false, 7);
        let mut got_case = case(g, 1, Dtype::Bfloat16, false, 7);
        for step in 0..30u64 {
            let (want, got) = (reference(&want_case), kernel_run(&got_case));
            for (at, name) in ["q", "k", "v", "rel", "k window", "v window"]
                .iter()
                .enumerate()
            {
                assert_eq!(bits(&want[at]), bits(&got[at]), "step {step}: {name}");
            }
            let cut = |w: &Array| w.try_index((.., 1..4, ..)).unwrap();
            let fresh = case(g, 1, Dtype::Bfloat16, false, 1000 + step);
            for (c, out) in [(&mut want_case, &want), (&mut got_case, &got)] {
                let (ks, vs) = (cut(&out[4]), cut(&out[5]));
                c.sources = fresh.sources.clone();
                c.fixed[0] = ks;
                c.fixed[1] = vs;
            }
        }
    }

    /// What the kernel cannot serve is an error, and `serves` says so beforehand.
    #[test]
    fn what_the_kernel_cannot_take_is_an_error() {
        let ok = case(Geometry(4, 2, 32, 4, 24), 2, Dtype::Bfloat16, false, 3);
        let fixed: Vec<&Array> = ok.fixed.iter().collect();
        let [a, b, c] = &ok.sources;
        assert!(serves(&ok.cfg, [a, b, c], &fixed), "a plain layer");
        assert!(project_local(&ok.cfg, &ok.cuts, [a, b, c], &fixed).is_ok());

        let wide = case(Geometry(2, 1, 132, 4, 8), 1, Dtype::Bfloat16, false, 3);
        let wf: Vec<&Array> = wide.fixed.iter().collect();
        let [wa, wb, wc] = &wide.sources;
        assert!(!serves(&wide.cfg, [wa, wb, wc], &wf), "a head row of 132");
        assert!(project_local(&wide.cfg, &wide.cuts, [wa, wb, wc], &wf).is_err());

        let long_rel = case(Geometry(2, 1, 32, 72, 8), 1, Dtype::Bfloat16, false, 3);
        let lf: Vec<&Array> = long_rel.fixed.iter().collect();
        let [la, lb, lc] = &long_rel.sources;
        assert!(
            !serves(&long_rel.cfg, [la, lb, lc], &lf),
            "a relative row of 72"
        );

        let batched = values(1, &[2, 2, a.shape()[2]], 1.0, Dtype::Bfloat16);
        assert!(!serves(&ok.cfg, [&batched, b, c], &fixed), "a batch of two");

        let int_src = a.as_dtype(Dtype::Int32).unwrap();
        assert!(
            !serves(&ok.cfg, [&int_src, b, c], &fixed),
            "an integer projection"
        );

        assert!(!serves(&ok.cfg, [a, b, c], &fixed[..6]), "six fixed arrays");

        let bad_cuts = vec![1, 2, 3, 4, 5];
        assert!(
            project_local(&ok.cfg, &bad_cuts, [a, b, c], &fixed).is_err(),
            "cuts that are not the layer's"
        );

        let half_state = ok.fixed[0].as_dtype(Dtype::Bfloat16).unwrap();
        let mut bad: Vec<&Array> = ok.fixed.iter().collect();
        bad[0] = &half_state;
        assert!(
            project_local(&ok.cfg, &ok.cuts, [a, b, c], &bad).is_err(),
            "a bfloat16 state"
        );
    }
}
