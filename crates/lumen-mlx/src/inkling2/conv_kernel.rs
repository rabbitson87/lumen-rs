//! The short convolution and its residual as one Metal kernel.
//!
//! After the attention and after the MLP, a layer visit convolves the sublayer's output with a few
//! taps per channel (the previous rows are the state, the current row the last tap), rounds the sum
//! to the weight dtype, adds the row, scales, rounds to the activation dtype, and adds the residual
//! stream. [`regions::conv_residual`](super::regions::conv_residual) and
//! [`regions::conv_next`](super::regions::conv_next) write that as composed ops, and
//! [`fused`](super::fused) compiles them for a decode step: still about eight Metal dispatches a
//! call (a copy for each of the state and the row into the padded window, and several fused
//! elementwise kernels), two calls a layer visit. This is the same arithmetic in one launch.
//!
//! A thread for each channel of each row. It adds the taps in the order the composed ops do (the
//! first tap, then each next tap added to the sum), rounds with the same conversions (the weight's
//! dtype for the sum, the row's dtype for the result), and so gives the composed result bit for bit:
//! the tests compare bits, which is why [`kernel`] is an `Optimization`. Each product is a statement
//! of its own and is added to the sum in the next: Metal's default contraction fuses a multiply and
//! an add only inside one statement, and the composed ops' fused kernels emit one op a statement, so
//! a product written into the sum (`acc += w * v`) would be an FMA and round differently.
//!
//! It also writes the padded window (the state, then the rows as float32) that the composed ops
//! build with a concatenate and that the next call's state is cut from.

use std::cell::OnceCell;

use anyhow::{Context, Result, anyhow, ensure};
use mlx_rs::{Array, Dtype};

use crate::metal_kernel::{MetalKernel, MetalKernelConfig};

lumen_flags::flag! {
    /// Run the Inkling 2 short convolution and its residual add (after the attention and after the
    /// MLP of each layer visit) as one Metal kernel in place of the compiled composed ops: about
    /// eight dispatches each become one. Bit for bit what the composed ops give (the kernel adds
    /// the taps and rounds in the same order, and the tests compare bits). Decode-sized passes
    /// only (a verify block too): a prefill chunk keeps the composed ops.
    ///
    /// A decode token about 0.5 to 0.6 ms (1.8%) faster: 20 same-session runs an arm against the
    /// version without it, 0.53 ms by mean wall time (paired standard error 0.12) and 0.64 ms by
    /// the median of steps; 608 fewer dispatches a token.
    pub(crate) kernel {
        env: "LUMEN_INKLING2_CONV_KERNEL",
        default: true,
        kind: Optimization,
    }
}

/// Whether the convolution runs as a kernel.
pub(super) fn enabled() -> bool {
    kernel::get()
}

/// The two places a layer visit convolves.
#[derive(Clone, Copy)]
enum Site {
    /// After the attention.
    Residual,
    /// After the MLP.
    Next,
}

#[cfg(test)]
thread_local! {
    static LAUNCHES: std::cell::Cell<[usize; 2]> = const { std::cell::Cell::new([0; 2]) };
}

/// How many times the kernel has been launched on this thread after the attention and after the MLP
/// (a test's check that each call site ran: one that only counted launches would pass with one of
/// the two never chosen).
#[cfg(test)]
pub(super) fn launches() -> [usize; 2] {
    LAUNCHES.with(std::cell::Cell::get)
}

/// Threads of a threadgroup: a thread for each channel of each row, in groups of this many.
const GROUP: i32 = 256;

/// The device's limit for the taps a launch convolves (the accumulators are registers).
const MAX_TAPS: i32 = 16;

/// A store that converts to the output's type, and a rounding to the type of an array whose pointer
/// is passed (a bfloat16 is not assignable from a float): the conversions the composed ops'
/// `as_dtype` make. The rounding takes either address space an input can be bound in.
const HEADER: &str = r#"
template <typename T>
inline void store(device T* p, int i, float v) {
    p[i] = static_cast<T>(v);
}
template <typename T>
inline float round_as(const device T* proxy, float v) {
    return float(static_cast<T>(v));
}
// An input of under 8 values is bound in the `constant` address space.
template <typename T>
inline float round_as(const constant T* proxy, float v) {
    return float(static_cast<T>(v));
}
"#;

/// `x` is `[rows, C]` (float32, bfloat16 or float16), `w` `[C, TAPS]` (any of those, its last tap
/// meets the current row), `state` `[TAPS - 1, C]` float32, `h` the stream `[rows, C]` (read as it
/// is; not read when `HAS_H` is 0), `scale` one float. `out` is `[rows, C]` in the output's dtype:
/// the stream plus `y` (or `y` alone with no stream), where `y` is `(round_w(sum of the taps) + x) *
/// scale` rounded to `x`'s dtype, as the composed ops round it. `window` is `[TAPS - 1 + rows, C]`
/// float32: the state, then the rows as float32.
const SOURCE: &str = r#"
    const int i = thread_position_in_grid.x;
    const int rows = x_shape[0];
    if (i >= rows * C) {
        return;
    }
    const int c = i % C;
    const int r = i / C;
    float acc = 0.0f;
    float current = 0.0f;
    for (int j = 0; j < TAPS; ++j) {
        const int t = r + j;
        const float v = t < TAPS - 1 ? state[t * C + c] : float(x[(t - (TAPS - 1)) * C + c]);
        if (j == TAPS - 1) {
            current = v;
        }
        const float term = float(w[c * TAPS + j]) * v;
        acc = j == 0 ? term : acc + term;
    }
    const float y = round_as(x, (round_as(w, acc) + current) * scale);
    float result = y;
    if (HAS_H) {
        result = float(h[i]) + y;
    }
    store(out, i, result);
    window[(TAPS - 1 + r) * C + c] = current;
    if (r == 0) {
        for (int t = 0; t < TAPS - 1; ++t) {
            window[t * C + c] = state[t * C + c];
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
                "inkling2_conv_residual",
                &["x", "w", "state", "h", "scale"],
                &["out", "window"],
                HEADER,
                SOURCE,
                /* ensure_row_contiguous */ true,
                /* atomic_outputs */ false,
            )?;
            let _ = cell.set(built);
        }
        f(cell
            .get()
            .ok_or_else(|| anyhow!("the convolution kernel is missing"))?)
    })
}

fn is_float(d: Dtype) -> bool {
    matches!(d, Dtype::Float32 | Dtype::Bfloat16 | Dtype::Float16)
}

/// One launch: `x` `[1, L, C]`, `w` `[C, taps]`, `state` `[1, taps - 1, C]` float32, `stream` the
/// residual stream `[1, L, C]` to add to (or none); `out_dtype` the dtype of the first result.
/// Returns `[out [1, L, C], window [1, taps - 1 + L, C] float32]`.
fn launch(
    site: Site,
    scale: f32,
    out_dtype: Dtype,
    x: &Array,
    w: &Array,
    state: &Array,
    stream: Option<&Array>,
) -> Result<[Array; 2]> {
    ensure!(
        is_float(x.dtype()) && is_float(w.dtype()) && is_float(out_dtype),
        "the convolution kernel takes float32, bfloat16 or float16, got x {:?}, w {:?}, out {out_dtype:?}",
        x.dtype(),
        w.dtype()
    );
    ensure!(
        state.dtype() == Dtype::Float32,
        "the convolution state is float32, got {:?}",
        state.dtype()
    );
    let &[batch, rows, channels] = x.shape() else {
        return Err(anyhow!("a row is [B, L, C], got {:?}", x.shape()));
    };
    ensure!(
        batch == 1,
        "the convolution kernel takes one sequence, got a batch of {batch}"
    );
    let &[w_channels, taps] = w.shape() else {
        return Err(anyhow!("the taps are [C, taps], got {:?}", w.shape()));
    };
    ensure!(
        w_channels == channels && (1..=MAX_TAPS).contains(&taps),
        "the taps are {:?} for rows of {channels} channels (at most {MAX_TAPS} taps)",
        w.shape()
    );
    ensure!(
        state.shape() == [1, taps - 1, channels],
        "the state is {:?}, the convolution needs {:?}",
        state.shape(),
        [1, taps - 1, channels]
    );
    if let Some(h) = stream {
        ensure!(
            is_float(h.dtype()) && h.shape() == x.shape(),
            "the stream is a float {:?}, got {:?} {:?}",
            x.shape(),
            h.dtype(),
            h.shape()
        );
    }

    #[cfg(test)]
    LAUNCHES.with(|n| {
        let mut counts = n.get();
        counts[site as usize] += 1;
        n.set(counts);
    });
    #[cfg(not(test))]
    let _ = site;
    let x2 = mlx_rs::ops::reshape(x, &[rows, channels])?;
    let state2 = mlx_rs::ops::reshape(state, &[taps - 1, channels])?;
    // With no stream the kernel never reads `h`, but it must be a buffer of its own size: the row
    // stands in (an array of under 8 values would be a `constant`).
    let h2 = match stream {
        Some(h) => mlx_rs::ops::reshape(h, &[rows, channels])?,
        None => x2.clone(),
    };
    let scale = Array::from_f32(scale);
    let threads = rows * channels;
    let mut outs = with_kernel(|kernel| {
        let config = MetalKernelConfig::new();
        config.add_template_arg_int("C", channels)?;
        config.add_template_arg_int("TAPS", taps)?;
        config.add_template_arg_int("HAS_H", i32::from(stream.is_some()))?;
        config.set_grid(threads, 1, 1)?;
        config.set_thread_group(GROUP.min(threads), 1, 1)?;
        config.add_output_arg(&[rows, channels], out_dtype)?;
        config.add_output_arg(&[taps - 1 + rows, channels], Dtype::Float32)?;
        kernel
            .apply(&[&x2, w, &state2, &h2, &scale], &config, 2)
            .context("the convolution kernel failed")
    })?;
    let window = mlx_rs::ops::reshape(outs.swap_remove(1), &[1, taps - 1 + rows, channels])?;
    let out = mlx_rs::ops::reshape(outs.swap_remove(0), &[1, rows, channels])?;
    Ok([out, window])
}

/// [`regions::conv_residual`](super::regions::conv_residual) as one launch: `args` are `[x, w,
/// state, h?]`; returns `[residual, window]`, the residual in float32 (the stream plus the
/// convolution, or the convolution alone at a block boundary).
pub(super) fn conv_residual(conv_scale: f32, args: &[&Array]) -> Result<[Array; 2]> {
    let (x, w, state, h) = match args {
        [x, w, state] => (x, w, state, None),
        [x, w, state, h] => (x, w, state, Some(*h)),
        _ => {
            return Err(anyhow!(
                "the conv residual takes three or four arrays, got {}",
                args.len()
            ));
        }
    };
    launch(Site::Residual, conv_scale, Dtype::Float32, x, w, state, h)
}

/// [`regions::conv_next`](super::regions::conv_next) as one launch: `args` are `[m, w, state,
/// residual]`; returns `[next, window]`, `next` in `dtype`.
pub(super) fn conv_next(conv_scale: f32, dtype: Dtype, args: &[&Array]) -> Result<[Array; 2]> {
    let [m, w, state, residual] = args else {
        return Err(anyhow!(
            "the conv next takes four arrays, got {}",
            args.len()
        ));
    };
    launch(Site::Next, conv_scale, dtype, m, w, state, Some(*residual))
}

#[cfg(test)]
mod tests {
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

    /// `[shape]` of values in `[-gain, gain)`, with a zero every eleventh element and one huge value
    /// every thirty-first (the rounding to bfloat16 has edges to cross), as `dtype`.
    fn values(seed: u64, shape: &[i32], gain: f32, dtype: Dtype) -> Array {
        let n: i32 = shape.iter().product();
        let v: Vec<f32> = noise(seed, n as usize)
            .into_iter()
            .enumerate()
            .map(|(i, x)| match i {
                i if i % 11 == 0 => 0.0,
                i if i % 31 == 0 => x * gain * 1000.0,
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

    /// The kernel is the composed ops to the bit: the result, the window, with and without a stream,
    /// for the dtypes the real model (bfloat16) and the float32 replica use, a prefill-free spread
    /// of row counts, and channel counts that are not multiples of the group or of anything.
    #[test]
    fn the_kernel_is_the_composed_convolution_to_the_bit() {
        let scale = (1.0f64 / 5.0f64.sqrt()) as f32;
        for (channels, taps) in [(2304, 4), (64, 4), (100, 4), (257, 3), (1, 4), (512, 2)] {
            for rows in [1, 2, 5, 16] {
                for (xd, wd, hd) in [
                    (Dtype::Bfloat16, Dtype::Bfloat16, Dtype::Bfloat16),
                    (Dtype::Float32, Dtype::Float32, Dtype::Float32),
                    (Dtype::Bfloat16, Dtype::Float32, Dtype::Float32),
                    (Dtype::Float16, Dtype::Bfloat16, Dtype::Float16),
                ] {
                    let x = values(1, &[1, rows, channels], 2.0, xd);
                    let w = values(2, &[channels, taps], 0.7, wd);
                    let state = values(3, &[1, taps - 1, channels], 2.0, Dtype::Float32)
                        .as_dtype(Dtype::Float32)
                        .unwrap();
                    let h = values(4, &[1, rows, channels], 3.0, hd);
                    let label =
                        format!("C={channels} taps={taps} rows={rows} x {xd:?} w {wd:?} h {hd:?}");
                    for stream in [Some(&h), None] {
                        let mut args = vec![&x, &w, &state];
                        args.extend(stream);
                        let want = regions::conv_residual(scale, &args).unwrap();
                        let got = conv_residual(scale, &args).unwrap();
                        assert_eq!(
                            bits(&want[0]),
                            bits(&got[0]),
                            "{label}: residual (stream {})",
                            stream.is_some()
                        );
                        assert_eq!(bits(&want[1]), bits(&got[1]), "{label}: window");
                        assert_eq!(got[0].dtype(), Dtype::Float32);
                        assert_eq!(got[1].shape(), &[1, taps - 1 + rows, channels]);
                    }
                    for dtype in [xd, Dtype::Float32] {
                        let residual = h.as_dtype(Dtype::Float32).unwrap();
                        let args = [&x, &w, &state, &residual];
                        let want = regions::conv_next(scale, dtype, &args).unwrap();
                        let got = conv_next(scale, dtype, &args).unwrap();
                        assert_eq!(bits(&want[0]), bits(&got[0]), "{label}: next as {dtype:?}");
                        assert_eq!(bits(&want[1]), bits(&got[1]), "{label}: window");
                        assert_eq!(got[0].dtype(), dtype);
                    }
                }
            }
        }
    }

    /// A call's window is the next call's state: chained over many steps the kernel stays the
    /// composed ops' (a window written wrong would show a step later).
    #[test]
    fn a_chain_of_calls_stays_the_composed_ops_to_the_bit() {
        let scale = (1.0f64 / 5.0f64.sqrt()) as f32;
        let (channels, taps) = (2304, 4);
        let w = values(2, &[channels, taps], 0.15, Dtype::Bfloat16);
        let h = values(4, &[1, 1, channels], 3.0, Dtype::Bfloat16);
        let mut state_want = Array::zeros::<f32>(&[1, taps - 1, channels]).unwrap();
        let mut state_got = state_want.clone();
        let mut x_want = values(1, &[1, 1, channels], 2.0, Dtype::Bfloat16);
        let mut x_got = x_want.clone();
        for step in 0..40 {
            let want = regions::conv_residual(scale, &[&x_want, &w, &state_want, &h]).unwrap();
            let got = conv_residual(scale, &[&x_got, &w, &state_got, &h]).unwrap();
            assert_eq!(bits(&want[0]), bits(&got[0]), "step {step}: residual");
            assert_eq!(bits(&want[1]), bits(&got[1]), "step {step}: window");
            let cut = |window: &Array| {
                mlx_rs::ops::indexing::TryIndexOp::try_index(window, (.., 1..taps, ..)).unwrap()
            };
            state_want = cut(&want[1]);
            state_got = cut(&got[1]);
            x_want = want[0].as_dtype(Dtype::Bfloat16).unwrap();
            x_got = got[0].as_dtype(Dtype::Bfloat16).unwrap();
        }
    }

    /// What the kernel cannot take is an error, not a wrong answer.
    #[test]
    fn what_the_kernel_cannot_take_is_an_error() {
        let scale = 0.5;
        let x = values(1, &[1, 2, 64], 1.0, Dtype::Bfloat16);
        let w = values(2, &[64, 4], 0.5, Dtype::Bfloat16);
        let state = values(3, &[1, 3, 64], 1.0, Dtype::Float32);
        let h = values(4, &[1, 2, 64], 1.0, Dtype::Bfloat16);
        assert!(
            conv_residual(scale, &[&x, &w, &state, &h]).is_ok(),
            "a plain call"
        );
        let half_state = state.as_dtype(Dtype::Bfloat16).unwrap();
        assert!(
            conv_residual(scale, &[&x, &w, &half_state, &h]).is_err(),
            "a bfloat16 state"
        );
        let short_state = values(3, &[1, 2, 64], 1.0, Dtype::Float32);
        assert!(
            conv_residual(scale, &[&x, &w, &short_state, &h]).is_err(),
            "a state of too few rows"
        );
        let wide_w = values(2, &[32, 4], 0.5, Dtype::Bfloat16);
        assert!(
            conv_residual(scale, &[&x, &wide_w, &state, &h]).is_err(),
            "taps of another channel count"
        );
        let batch = values(1, &[2, 2, 64], 1.0, Dtype::Bfloat16);
        assert!(
            conv_residual(scale, &[&batch, &w, &state, &h]).is_err(),
            "a batch of two"
        );
        let int_x = x.as_dtype(Dtype::Int32).unwrap();
        assert!(
            conv_residual(scale, &[&int_x, &w, &state, &h]).is_err(),
            "an integer row"
        );
        let other_h = values(4, &[1, 3, 64], 1.0, Dtype::Bfloat16);
        assert!(
            conv_residual(scale, &[&x, &w, &state, &other_h]).is_err(),
            "a stream of another length"
        );
        assert!(conv_residual(scale, &[&x, &w]).is_err(), "two arrays");
        assert!(
            conv_next(scale, Dtype::Int32, &[&x, &w, &state, &h]).is_err(),
            "an integer output"
        );
    }
}
