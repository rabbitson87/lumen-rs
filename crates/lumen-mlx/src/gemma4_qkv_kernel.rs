//! The Gemma 4 attention's q/k/v norms and RoPE, for a one-row decode pass, as one Metal kernel.
//!
//! After the q, k and v projections, a decode step norms each head of each (`q_norm`, `k_norm`,
//! and a unit-weight norm for v), lays the heads out as `[B, H, L, D]`, and rotates q and k by
//! their position: five Metal dispatches a layer (three `rms_norm`, two `rope`). [`project_tail`]
//! is the same arithmetic in one launch.
//!
//! One threadgroup a head row (the q heads, then the k heads, then the v heads), laid out as MLX's
//! `rms_single_row` lays out a norm over the head width (a thread for each four values), and
//! reducing and rounding as it does (see `gemma4_norm_kernels`). The normed q and k rows then go
//! through threadgroup memory to the rotation, which is MLX's `rope_single` (and
//! `rope_single_freqs`) pair by pair: pair `j` of `D / 2` takes `inv_freq = exp2(-(j / (D / 2)) *
//! log2(base))` (or `1 / freqs[j]`), `theta = offset * inv_freq`, `fast::cos` and `fast::sin`, and
//! the two products of each output in one statement, as that kernel writes them. The outputs are
//! therefore MLX's bit for bit, which the tests compare; that is why the flag is an
//! `Optimization`. One-row passes only (MLX's `rope_single` is the one-row kernel; a verify block
//! runs the strided `rope`): anything else keeps the composed ops.

use std::cell::OnceCell;

use anyhow::{Context, Result, anyhow, ensure};
use mlx_rs::{Array, Dtype};

use crate::metal_kernel::{MetalKernel, MetalKernelConfig};

lumen_flags::flag! {
    /// Run the Gemma 4 attention's q/k/v norms and RoPE as one Metal kernel on a one-row decode
    /// pass: five dispatches a layer become one. Bit for bit what `rms_norm` and `rope` give.
    /// Every KV-cache kind takes its output; a verify block or a prefill chunk keeps the composed
    /// ops.
    ///
    /// On by default: a decode token 0.25 ms (1.6%) faster on gemma-4-26b-a4b NVFP4 (M3 Max,
    /// 20 same-process ABBA blocks, every block faster, paired standard error 0.028 ms), 120 fewer
    /// dispatches a token, the same continuation in 80 of 80 runs. `=0` restores the composed ops.
    pub(crate) qkv_kernel {
        env: "LUMEN_GEMMA4_QKV_KERNEL",
        default: true,
        kind: Optimization,
    }
}

/// Whether the q/k/v norms and RoPE run as a kernel.
pub(crate) fn enabled() -> bool {
    qkv_kernel::get()
}

/// The widest head a launch takes: its normed row is held in threadgroup memory.
const MAX_HEAD_DIM: i32 = 1024;

#[cfg(test)]
thread_local! {
    static LAUNCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// How many times the kernel has been launched on this thread (a test's check that it ran).
#[cfg(test)]
pub(crate) fn launches() -> usize {
    LAUNCHES.with(std::cell::Cell::get)
}

/// How the rotation's frequencies are given.
pub(crate) enum Rope<'a> {
    /// `rope(dims = D, base)`: `log2(base)` as one float32, as `RoPE::eval_gpu` hands it over.
    BaseLog2(&'a Array),
    /// `rope(dims = D, freqs)`: `[D / 2]` float32.
    Freqs(&'a Array),
}

/// The inverse RMS of a row as `rms_single_row` reduces it (see `gemma4_norm_kernels`).
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

/// `q` is `[H * D]`, `k` and `v` `[HKV * D]` (bfloat16); `q_norm`, `k_norm`, `ones` `[D]`
/// bfloat16; `offset` one int32 (the position); `rope` one float32 `log2(base)` (`FREQS` 0) or
/// `[D / 2]` float32 frequencies (`FREQS` 1); `eps` one float32. Writes `q_out` `[H * D]`, `k_out`
/// and `v_out` `[HKV * D]`: q and k normed and rotated, v normed with unit weights.
const SOURCE: &str = r#"
    constexpr int N = 4;
    const uint g = threadgroup_position_in_grid.x;
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint axis_size = uint(q_norm_shape[0]);

    threadgroup float sums[32];
    threadgroup float inv[1];
    threadgroup bfloat16_t normed[D];

    // Which head this group norms: q heads, then k heads, then v heads.
    const bool is_q = g < H;
    const bool is_k = !is_q && g < H + HKV;
    const uint head = is_q ? g : (is_k ? g - H : g - H - HKV);
    const size_t at = size_t(head) * D + lid * N;

    float t[N];
    float acc = 0;
    for (int i = 0; i < N; i++) {
        t[i] = is_q ? float(q[at + i]) : (is_k ? float(k[at + i]) : float(v[at + i]));
        acc += t[i] * t[i];
    }
    const float inv_rms = row_inv_rms(acc, lane, sg, axis_size, eps[0], sums, inv);

    if (!is_q && !is_k) {
        for (int i = 0; i < N; i++) {
            v_out[at + i] = ones[lid * N + i] * static_cast<bfloat16_t>(t[i] * inv_rms);
        }
        return;
    }
    for (int i = 0; i < N; i++) {
        const bfloat16_t w = is_q ? q_norm[lid * N + i] : k_norm[lid * N + i];
        normed[lid * N + i] = w * static_cast<bfloat16_t>(t[i] * inv_rms);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // `rope_single`: pair `j` of the `D / 2` pairs rotates elements `j` and `j + D / 2`.
    const uint half_d = axis_size / 2;
    const float position = 1.0f * static_cast<float>(offset[0]);
    for (uint j = lid; j < half_d; j += D / N) {
        float inv_freq;
        if (FREQS) {
            inv_freq = 1.0 / (rope[j]);
        } else {
            float d = static_cast<float>(j) / static_cast<float>(half_d);
            inv_freq = metal::exp2(-d * rope[0]);
        }
        float theta = position * inv_freq;
        float costheta = metal::fast::cos(theta);
        float sintheta = metal::fast::sin(theta);
        float x1 = static_cast<float>(normed[j]);
        float x2 = static_cast<float>(normed[j + half_d]);
        float rx1 = x1 * costheta - x2 * sintheta;
        float rx2 = x1 * sintheta + x2 * costheta;
        const size_t o = size_t(head) * D;
        if (is_q) {
            q_out[o + j] = static_cast<bfloat16_t>(rx1);
            q_out[o + j + half_d] = static_cast<bfloat16_t>(rx2);
        } else {
            k_out[o + j] = static_cast<bfloat16_t>(rx1);
            k_out[o + j + half_d] = static_cast<bfloat16_t>(rx2);
        }
    }
"#;

thread_local! {
    /// MLX keeps one stream per thread and the engine decodes on one thread, so the kernel is
    /// built once per thread rather than per call.
    static KERNEL: OnceCell<MetalKernel> = const { OnceCell::new() };
}

fn check_bf16(name: &str, a: &Array, shape: &[i32]) -> Result<()> {
    ensure!(
        a.dtype() == Dtype::Bfloat16 && a.shape() == shape,
        "{name} must be bfloat16 {shape:?}, got {:?} {:?}",
        a.dtype(),
        a.shape()
    );
    Ok(())
}

/// The q/k/v norms and RoPE of a one-row decode pass, as one launch. `q` is `[1, 1, H, D]`, `k` and
/// `v` `[1, 1, HKV, D]` (bfloat16; `v` may be `k` itself, for a layer whose values are its keys);
/// `q_norm`, `k_norm` and `ones` are `[D]` bfloat16, `offset` the position as one int32, `eps` one
/// float32. Returns `[q, k, v]` as `[1, H, 1, D]`, `[1, HKV, 1, D]` and `[1, HKV, 1, D]`: `rope(
/// rms_norm(q, q_norm))`, `rope(rms_norm(k, k_norm))` and `rms_norm(v, ones)`, with the heads
/// before the row as the attention reads them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn project_tail(
    q: &Array,
    k: &Array,
    v: &Array,
    q_norm: &Array,
    k_norm: &Array,
    ones: &Array,
    offset: &Array,
    rope: Rope<'_>,
    eps: &Array,
) -> Result<[Array; 3]> {
    let &[1, 1, heads, head_dim] = q.shape() else {
        return Err(anyhow!(
            "the q heads of a one-row pass are [1, 1, H, D], got {:?}",
            q.shape()
        ));
    };
    let &[1, 1, kv_heads, kv_dim] = k.shape() else {
        return Err(anyhow!(
            "the k heads of a one-row pass are [1, 1, HKV, D], got {:?}",
            k.shape()
        ));
    };
    ensure!(
        kv_dim == head_dim && head_dim % 4 == 0 && (128..=MAX_HEAD_DIM).contains(&head_dim),
        "the q/k/v kernel takes heads of one width, a multiple of 4 from 128 to {MAX_HEAD_DIM}, \
         got q {head_dim} and k {kv_dim}"
    );
    check_bf16("q", q, &[1, 1, heads, head_dim])?;
    check_bf16("k", k, &[1, 1, kv_heads, head_dim])?;
    check_bf16("v", v, &[1, 1, kv_heads, head_dim])?;
    for (name, w) in [("q_norm", q_norm), ("k_norm", k_norm), ("ones", ones)] {
        check_bf16(name, w, &[head_dim])?;
    }
    ensure!(
        offset.dtype() == Dtype::Int32 && offset.size() == 1,
        "the position is one int32, got {:?} {:?}",
        offset.dtype(),
        offset.shape()
    );
    ensure!(
        eps.dtype() == Dtype::Float32 && eps.size() == 1,
        "eps is one float32, got {:?} {:?}",
        eps.dtype(),
        eps.shape()
    );
    let (rope, freqs) = match rope {
        Rope::BaseLog2(base) => {
            ensure!(
                base.dtype() == Dtype::Float32 && base.size() == 1,
                "log2(base) is one float32"
            );
            (base, 0)
        }
        Rope::Freqs(f) => {
            ensure!(
                f.dtype() == Dtype::Float32 && f.shape() == [head_dim / 2],
                "the frequencies are [{}] float32, got {:?} {:?}",
                head_dim / 2,
                f.dtype(),
                f.shape()
            );
            (f, 1)
        }
    };
    #[cfg(test)]
    LAUNCHES.with(|n| n.set(n.get() + 1));
    let flat = |a: &Array, h: i32| mlx_rs::ops::reshape(a, &[h * head_dim]);
    let (q1, k1, v1) = (flat(q, heads)?, flat(k, kv_heads)?, flat(v, kv_heads)?);
    let threads = head_dim / 4;
    let groups = heads + 2 * kv_heads;
    let outs = crate::metal_kernel::with_thread_kernel(
        &KERNEL,
        || {
            // The name is a literal: `cargo xtask test` reads the names off `MetalKernel::new`.
            MetalKernel::new_with_header(
                "gemma4_qkv_norm_rope",
                &[
                    "q", "k", "v", "q_norm", "k_norm", "ones", "offset", "rope", "eps",
                ],
                &["q_out", "k_out", "v_out"],
                HEADER,
                SOURCE,
                /* ensure_row_contiguous */ true,
                /* atomic_outputs */ false,
            )
        },
        |kernel| {
            let config = MetalKernelConfig::new();
            config.add_template_arg_int("H", heads)?;
            config.add_template_arg_int("HKV", kv_heads)?;
            config.add_template_arg_int("D", head_dim)?;
            config.add_template_arg_int("FREQS", freqs)?;
            config.set_grid(threads * groups, 1, 1)?;
            config.set_thread_group(threads, 1, 1)?;
            config.add_output_arg(&[heads * head_dim], Dtype::Bfloat16)?;
            config.add_output_arg(&[kv_heads * head_dim], Dtype::Bfloat16)?;
            config.add_output_arg(&[kv_heads * head_dim], Dtype::Bfloat16)?;
            kernel
                .apply(
                    &[&q1, &k1, &v1, q_norm, k_norm, ones, offset, rope, eps],
                    &config,
                    3,
                )
                .context("the Gemma 4 q/k/v kernel failed")
        },
    )?;
    let [q_out, k_out, v_out]: [Array; 3] = outs
        .try_into()
        .map_err(|_| anyhow!("the q/k/v kernel returned the wrong number of outputs"))?;
    Ok([
        mlx_rs::ops::reshape(&q_out, &[1, heads, 1, head_dim])?,
        mlx_rs::ops::reshape(&k_out, &[1, kv_heads, 1, head_dim])?,
        mlx_rs::ops::reshape(&v_out, &[1, kv_heads, 1, head_dim])?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_norm::rms_norm;
    use crate::native_rope::{rope, rope_with_freqs};

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

    fn values(seed: u64, shape: &[i32], offset: f32, gain: f32) -> Array {
        let n: i32 = shape.iter().product();
        let v: Vec<f32> = noise(seed, n as usize)
            .into_iter()
            .enumerate()
            .map(|(i, x)| match i {
                i if i % 13 == 0 => 0.0,
                i if i % 29 == 0 => offset + x * gain * 500.0,
                _ => offset + x * gain,
            })
            .collect();
        Array::from_slice(&v, shape)
            .as_dtype(Dtype::Bfloat16)
            .expect("cast")
    }

    fn bits(a: &Array) -> Vec<u16> {
        let wide = mlx_rs::ops::reshape(a, &[-1])
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        wide.eval().unwrap();
        wide.as_slice::<f32>()
            .iter()
            .map(|x| (x.to_bits() >> 16) as u16)
            .collect()
    }

    /// The frequencies `layer_attention_forward` builds for a partial rotary factor: the first
    /// `rope_dim / 2` from the full head width, the rest infinite.
    fn partial_freqs(head_dim: i32, rope_dim: i32, base: f32) -> Vec<f32> {
        let half_rot = (rope_dim / 2) as usize;
        let half_inf = ((head_dim - rope_dim) / 2) as usize;
        let mut f: Vec<f32> = (0..half_rot)
            .map(|i| base.powf((2 * i) as f32 / head_dim as f32))
            .collect();
        f.extend(std::iter::repeat_n(f32::INFINITY, half_inf));
        f
    }

    /// The composed ops `layer_attention_forward` runs on a one-row pass.
    #[allow(clippy::too_many_arguments)]
    fn composed(
        q: &Array,
        k: &Array,
        v: &Array,
        q_norm: &Array,
        k_norm: &Array,
        ones: &Array,
        offset: i32,
        base: f32,
        freqs: Option<&Array>,
    ) -> [Array; 3] {
        let head_dim = q.shape()[3];
        let t = |a: &Array| mlx_rs::ops::transpose_axes(a, &[0, 2, 1, 3]).unwrap();
        let q_t = t(&rms_norm(q, q_norm, EPS).unwrap());
        let k_t = t(&rms_norm(k, k_norm, EPS).unwrap());
        let v_t = t(&rms_norm(v, ones, EPS).unwrap());
        let (q_r, k_r) = match freqs {
            None => (
                rope(&q_t, head_dim, false, base, 1.0, offset).unwrap(),
                rope(&k_t, head_dim, false, base, 1.0, offset).unwrap(),
            ),
            Some(f) => (
                rope_with_freqs(&q_t, head_dim, false, 1.0, offset, f).unwrap(),
                rope_with_freqs(&k_t, head_dim, false, 1.0, offset, f).unwrap(),
            ),
        };
        [q_r, k_r, v_t]
    }

    /// The kernel is the composed norms and RoPE to the bit: Gemma 4's sliding (16 q heads, 8 kv,
    /// width 256, full rotation) and full layers (16 and 2, width 512, a quarter rotated through
    /// frequencies), a narrower head, positions from the first to past half a million, input
    /// scales across orders of magnitude, and a layer whose values are its keys.
    #[test]
    fn the_kernel_is_the_composed_norms_and_rope_to_the_bit() {
        let geometries = [(16, 8, 256, false), (16, 2, 512, true), (8, 4, 128, false)];
        for (heads, kv_heads, head_dim, partial) in geometries {
            for offset in [0, 1, 7, 1023, 1024, 40_000, 524_287] {
                for gain in [1.0f32, 0.02, 60.0] {
                    let seed = (heads * 1009 + head_dim * 7 + offset) as u64;
                    let q = values(seed, &[1, 1, heads, head_dim], 0.0, gain);
                    let k = values(seed + 1, &[1, 1, kv_heads, head_dim], 0.0, gain * 2.0);
                    let v = values(seed + 2, &[1, 1, kv_heads, head_dim], 0.0, gain * 0.5);
                    let q_norm = values(seed + 3, &[head_dim], 1.0, 0.4);
                    let k_norm = values(seed + 4, &[head_dim], 1.0, 0.4);
                    let ones = values(seed + 5, &[head_dim], 1.0, 0.0);
                    let eps = Array::from_slice(&[EPS], &[1]);
                    let pos = Array::from_slice(&[offset], &[1]);
                    let (base, freqs) = if partial {
                        let f = partial_freqs(head_dim, head_dim / 4, 1_000_000.0);
                        (1_000_000.0f32, Some(Array::from_slice(&f, &[head_dim / 2])))
                    } else {
                        (10_000.0f32, None)
                    };
                    let base_log2 = Array::from_slice(&[base.log2()], &[1]);
                    for keys_are_values in [false, true] {
                        let v = if keys_are_values { &k } else { &v };
                        let rope_arg = match &freqs {
                            Some(f) => Rope::Freqs(f),
                            None => Rope::BaseLog2(&base_log2),
                        };
                        let got =
                            project_tail(&q, &k, v, &q_norm, &k_norm, &ones, &pos, rope_arg, &eps)
                                .expect("kernel");
                        let want = composed(
                            &q,
                            &k,
                            v,
                            &q_norm,
                            &k_norm,
                            &ones,
                            offset,
                            base,
                            freqs.as_ref(),
                        );
                        for (i, name) in ["q", "k", "v"].iter().enumerate() {
                            assert_eq!(got[i].shape(), want[i].shape(), "{name} shape");
                            let (g, w) = (bits(&got[i]), bits(&want[i]));
                            let differ = g.iter().zip(&w).filter(|(a, b)| a != b).count();
                            assert_eq!(
                                differ,
                                0,
                                "{name}: {differ} of {} differ ({heads}/{kv_heads} heads of \
                                 {head_dim}, position {offset}, gain {gain}, keys are values \
                                 {keys_are_values})",
                                w.len()
                            );
                        }
                    }
                }
            }
        }
    }

    /// What the kernel does not serve is an error, never a wrong answer.
    #[test]
    fn what_the_kernel_cannot_take_is_an_error() {
        let q = values(1, &[1, 1, 16, 256], 0.0, 1.0);
        let k = values(2, &[1, 1, 8, 256], 0.0, 1.0);
        let w = values(3, &[256], 1.0, 0.1);
        let eps = Array::from_slice(&[EPS], &[1]);
        let pos = Array::from_slice(&[3i32], &[1]);
        let base = Array::from_slice(&[10_000f32.log2()], &[1]);
        let two_rows = values(4, &[1, 2, 16, 256], 0.0, 1.0);
        assert!(
            project_tail(
                &two_rows,
                &k,
                &k,
                &w,
                &w,
                &w,
                &pos,
                Rope::BaseLog2(&base),
                &eps
            )
            .is_err(),
            "a verify block"
        );
        let wide = q.as_dtype(Dtype::Float32).unwrap();
        assert!(
            project_tail(&wide, &k, &k, &w, &w, &w, &pos, Rope::BaseLog2(&base), &eps).is_err(),
            "float32 heads"
        );
        let k512 = values(5, &[1, 1, 8, 512], 0.0, 1.0);
        assert!(
            project_tail(
                &q,
                &k512,
                &k512,
                &w,
                &w,
                &w,
                &pos,
                Rope::BaseLog2(&base),
                &eps
            )
            .is_err(),
            "heads of two widths"
        );
        let short = Array::from_slice(&[1.0f32; 64], &[64]);
        assert!(
            project_tail(&q, &k, &k, &w, &w, &w, &pos, Rope::Freqs(&short), &eps).is_err(),
            "too few frequencies"
        );
        let pos_f = Array::from_slice(&[3.0f32], &[1]);
        assert!(
            project_tail(&q, &k, &k, &w, &w, &w, &pos_f, Rope::BaseLog2(&base), &eps).is_err(),
            "a float position"
        );
    }
}
