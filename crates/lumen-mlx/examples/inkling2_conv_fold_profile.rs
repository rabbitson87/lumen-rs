//! What replacing a layer visit's "short convolution plus residual" region with one Metal kernel
//! would buy, priced in a sandbox before anything is built into the model.
//!
//! ```text
//! cargo run --release -p lumen-mlx --features mlx-native --example inkling2_conv_fold_profile
//! ```
//!
//! After the attention and after the MLP, a layer visit convolves the sublayer's output with four
//! taps per channel (the previous three rows are the state, the current row the last tap), rounds
//! the sum to the weight dtype (bfloat16), adds the row, scales, rounds to the activation dtype, and
//! adds the residual stream; the padded input (state rows, then the row) is also the next state's
//! window. `inkling2/regions.rs` writes that as composed MLX ops and `fused.rs` compiles it with
//! `mx.compile`: still 6.8 (attention) and 8.0 (MLP) Metal dispatches a call, 712 a token.
//!
//! Three forms are timed as calls each reading the last one's output (so the GPU runs them one after
//! another, as in the model), waited for once:
//!
//! * `plain`: the composed ops as they run for a prefill chunk;
//! * `compiled`: the same ops through `mx.compile`, as a decode step runs them;
//! * `kernel`: one launch, a thread for each channel of each row, that reads the row and the state
//!   and writes both the result and the padded window.
//!
//! The dispatches a call launches are counted with MLX's command-buffer statistics. The kernel's
//! result is compared with the compiled form's to the bit before anything is timed.
//!
//! A sandbox chain over-promises what the model gains: this one says 15 and 19 us a call, 1.6 ms a
//! token for both convolutions; in the model (20 same-session runs an arm against the version without
//! the kernel, paired ABBA blocks) the kernel gained 0.53 ms by mean wall time and 0.64 ms by the
//! median of steps, about a third of it.

use std::time::Instant;

use anyhow::{Result, ensure};
use lumen_mlx::metal_kernel::{MetalKernel, MetalKernelConfig};
use mlx_rs::error::Exception;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::transforms::compile::compile_boxed_slice;
use mlx_rs::{Array, Dtype};

/// Taps of the convolution (`sconv_kernel_size`).
const TAPS: i32 = 4;

/// `x` is `[rows, C]` (bfloat16), `w` `[C, TAPS]` (bfloat16), `state` `[TAPS - 1, C]` float32, `h` the
/// residual stream `[rows, C]` (read as it is; not read when `HAS_H` is 0), `scale` one float.
/// `out` is `[rows, C]`: with `MODE` 0 the residual `h + y` (or `y` alone, no `h`) in float32, with
/// `MODE` 1 the next stream `h + y` in the output's dtype, where `y` is `((round_bf16(sum of the
/// taps)) + x) * scale` rounded to bfloat16, as the composed ops round it. `window` is
/// `[TAPS - 1 + rows, C]` float32: the state, then the rows as float32.
/// A store that converts to the output's type (a bfloat16 output is not assignable from a float).
const HEADER: &str = r#"
template <typename T>
inline void store(device T* p, int i, float v) {
    p[i] = static_cast<T>(v);
}
"#;

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
    const float rounded = float(static_cast<bfloat16_t>(acc));
    const float y = float(static_cast<bfloat16_t>((rounded + current) * scale));
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

#[derive(Clone, Copy, PartialEq)]
enum Region {
    /// `regions::conv_residual` with the stream: `[x, w, state, h]` -> `[residual f32, window]`.
    Residual,
    /// `regions::conv_next`: `[m, w, state, residual f32]` -> `[next bf16, window]`.
    Next,
}

fn scalar(v: f32) -> Array {
    Array::from_f32(v)
}

/// `ops::static_conv`.
fn static_conv(
    x: &Array,
    w: &Array,
    state: &Array,
    scale: f32,
) -> std::result::Result<(Array, Array), Exception> {
    let xf = x.as_dtype(Dtype::Float32)?;
    let length = xf.shape()[1];
    let xp = mlx_rs::ops::concatenate_axis(&[state, &xf], 1)?;
    let mut out: Option<Array> = None;
    for j in 0..TAPS {
        let wj = w.try_index((.., j))?;
        let window = xp.try_index((.., j..j + length, ..))?;
        let tap = mlx_rs::ops::multiply(&wj, &window)?;
        out = Some(match out {
            None => tap,
            Some(acc) => mlx_rs::ops::add(&acc, &tap)?,
        });
    }
    let out = out.expect("taps");
    let rounded = out.as_dtype(w.dtype())?.as_dtype(Dtype::Float32)?;
    let y = mlx_rs::ops::multiply(mlx_rs::ops::add(&rounded, &xf)?, scalar(scale))?;
    Ok((y.as_dtype(x.dtype())?, xp))
}

/// `regions::conv_residual` and `regions::conv_next`.
fn composed(region: Region, args: &[Array]) -> std::result::Result<Vec<Array>, Exception> {
    let [x, w, state, h] = args else {
        unreachable!("four arrays")
    };
    let scale = (1.0f64 / f64::from(TAPS + 1).sqrt()) as f32;
    let (y, window) = static_conv(x, w, state, scale)?;
    let yf = y.as_dtype(Dtype::Float32)?;
    let out = match region {
        Region::Residual => mlx_rs::ops::add(h.as_dtype(Dtype::Float32)?, yf)?,
        Region::Next => mlx_rs::ops::add(h, yf)?.as_dtype(Dtype::Bfloat16)?,
    };
    Ok(vec![out, window])
}

struct Forms {
    plain: fn(Region, &[Array]) -> std::result::Result<Vec<Array>, Exception>,
    residual: Box<dyn FnMut(&[Array]) -> std::result::Result<Vec<Array>, Exception> + Send>,
    next: Box<dyn FnMut(&[Array]) -> std::result::Result<Vec<Array>, Exception> + Send>,
    kernel: MetalKernel,
}

impl Forms {
    fn new() -> Result<Self> {
        Ok(Self {
            plain: composed,
            residual: compile_boxed_slice(|a: &[Array]| composed(Region::Residual, a), false),
            next: compile_boxed_slice(|a: &[Array]| composed(Region::Next, a), false),
            kernel: MetalKernel::new_with_header(
                "inkling2_conv_fold_profile",
                &["x", "w", "state", "h", "scale"],
                &["out", "window"],
                HEADER,
                SOURCE,
                true,
                false,
            )?,
        })
    }

    fn compiled(&mut self, region: Region, args: &[Array]) -> Result<Vec<Array>> {
        let f = match region {
            Region::Residual => &mut self.residual,
            Region::Next => &mut self.next,
        };
        Ok(f(args)?)
    }

    /// The one-launch form. `args` are `[x [1, L, C], w, state [1, 3, C], h [1, L, C]]`.
    fn launch(&self, region: Region, args: &[Array]) -> Result<Vec<Array>> {
        let [x, w, state, h] = args else {
            unreachable!("four arrays")
        };
        let (rows, c) = (x.shape()[1], x.shape()[2]);
        let scale = (1.0f64 / f64::from(TAPS + 1).sqrt()) as f32;
        let x2 = mlx_rs::ops::reshape(x, &[rows, c])?;
        let state2 = mlx_rs::ops::reshape(state, &[TAPS - 1, c])?;
        let h2 = mlx_rs::ops::reshape(h, &[rows, c])?;
        let config = MetalKernelConfig::new();
        config.add_template_arg_int("C", c)?;
        config.add_template_arg_int("TAPS", TAPS)?;
        config.add_template_arg_int("HAS_H", 1)?;
        let threads = rows * c;
        config.set_grid(threads, 1, 1)?;
        config.set_thread_group(256.min(threads), 1, 1)?;
        let out_dtype = match region {
            Region::Residual => Dtype::Float32,
            Region::Next => Dtype::Bfloat16,
        };
        config.add_output_arg(&[rows, c], out_dtype)?;
        config.add_output_arg(&[TAPS - 1 + rows, c], Dtype::Float32)?;
        let mut outs = self
            .kernel
            .apply(&[&x2, w, &state2, &h2, &scalar(scale)], &config, 2)?;
        let window = mlx_rs::ops::reshape(outs.swap_remove(1), &[1, TAPS - 1 + rows, c])?;
        let out = mlx_rs::ops::reshape(outs.swap_remove(0), &[1, rows, c])?;
        Ok(vec![out, window])
    }
}

#[derive(Clone, Copy)]
enum Form {
    Plain,
    Compiled,
    Kernel,
}

impl Form {
    fn name(self) -> &'static str {
        match self {
            Form::Plain => "plain ops (a prefill chunk)",
            Form::Compiled => "compiled (shipped)",
            Form::Kernel => "one kernel",
        }
    }
}

fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect()
}

struct Inputs {
    w: Array,
    state: Array,
    /// The residual stream the call adds to: bfloat16 for the attention's, float32 for the MLP's.
    h: Array,
    x: Array,
}

impl Inputs {
    fn new(region: Region, rows: i32, c: i32) -> Result<Self> {
        let bf16 = |seed: u64, n: i32, shape: &[i32], gain: f32| -> Result<Array> {
            let v: Vec<f32> = noise(seed, n as usize)
                .into_iter()
                .map(|x| x * gain)
                .collect();
            Ok(Array::from_slice(&v, shape).as_dtype(Dtype::Bfloat16)?)
        };
        let h = bf16(5, rows * c, &[1, rows, c], 3.0)?;
        let h = match region {
            Region::Residual => h,
            Region::Next => h.as_dtype(Dtype::Float32)?,
        };
        let inputs = Self {
            w: bf16(1, c * TAPS, &[c, TAPS], 0.15)?,
            state: Array::from_slice(&noise(2, ((TAPS - 1) * c) as usize), &[1, TAPS - 1, c]),
            h,
            x: bf16(3, rows * c, &[1, rows, c], 2.0)?,
        };
        for a in [&inputs.w, &inputs.state, &inputs.h, &inputs.x] {
            a.eval()?;
        }
        Ok(inputs)
    }
}

fn host_bits(a: &Array) -> Result<Vec<u32>> {
    let flat = mlx_rs::ops::reshape(a.as_dtype(Dtype::Float32)?, &[-1])?;
    flat.eval()?;
    Ok(flat.as_slice::<f32>().iter().map(|v| v.to_bits()).collect())
}

fn call(forms: &mut Forms, form: Form, region: Region, args: &[Array]) -> Result<Vec<Array>> {
    match form {
        Form::Plain => Ok((forms.plain)(region, args)?),
        Form::Compiled => forms.compiled(region, args),
        Form::Kernel => forms.launch(region, args),
    }
}

/// The next call's arguments from this call's outputs: the output (as the activation dtype) is the
/// next row, the last three rows of the window the next state.
fn next_args(i: &Inputs, outs: &[Array]) -> Result<Vec<Array>> {
    let window = &outs[1];
    let len = window.shape()[1];
    let state = window.try_index((.., (len - (TAPS - 1))..len, ..))?;
    let x = outs[0].as_dtype(Dtype::Bfloat16)?;
    Ok(vec![x, i.w.clone(), state, i.h.clone()])
}

/// Microseconds a call over a chain of `calls`, the CPU's share of it, and the dispatches a call
/// launches.
fn chain(
    forms: &mut Forms,
    form: Form,
    region: Region,
    i: &Inputs,
    calls: usize,
) -> Result<(f64, f64, f64)> {
    let mut args = vec![i.x.clone(), i.w.clone(), i.state.clone(), i.h.clone()];
    let (_, d0) = mlx_rs::metal::cmd_buffer_stats().unwrap_or_default();
    let start = Instant::now();
    let mut last = Vec::new();
    for _ in 0..calls {
        last = call(forms, form, region, &args)?;
        args = next_args(i, &last)?;
    }
    let build = start.elapsed().as_secs_f64() * 1e6 / calls as f64;
    args[0].eval()?;
    last[1].eval()?;
    let total = start.elapsed().as_secs_f64() * 1e6 / calls as f64;
    let (_, d1) = mlx_rs::metal::cmd_buffer_stats().unwrap_or_default();
    ensure!(
        host_bits(&last[0])?
            .iter()
            .all(|b| f32::from_bits(*b).is_finite()),
        "non-finite output of the {} form after {calls} calls",
        form.name()
    );
    Ok((total, build, d1.saturating_sub(d0) as f64 / calls as f64))
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn section(
    forms: &mut Forms,
    region: Region,
    title: &str,
    rows: i32,
    calls_a_token: f64,
) -> Result<()> {
    const CALLS: usize = 400;
    const ROUNDS: usize = 9;
    let i = Inputs::new(region, rows, 2304)?;
    let args = vec![i.x.clone(), i.w.clone(), i.state.clone(), i.h.clone()];
    // The kernel against the compiled form, to the bit, over a chain of 20 calls (the state and the
    // stream feed back, so an error in the window shows up in the next call's result).
    let (mut a, mut b) = (args.clone(), args);
    let mut moved = 0usize;
    let mut total = 0usize;
    for _ in 0..20 {
        let want = call(forms, Form::Compiled, region, &a)?;
        let got = call(forms, Form::Kernel, region, &b)?;
        for k in 0..2 {
            let (w, g) = (host_bits(&want[k])?, host_bits(&got[k])?);
            ensure!(w.len() == g.len(), "output {k}: shape");
            moved += w.iter().zip(&g).filter(|(p, q)| p != q).count();
            total += w.len();
        }
        a = next_args(&i, &want)?;
        b = next_args(&i, &got)?;
    }
    let forms_list = [Form::Plain, Form::Compiled, Form::Kernel];
    for f in forms_list {
        chain(forms, f, region, &i, 40)?; // compile and warm
    }
    let mut us = vec![Vec::new(); 3];
    let mut build = vec![Vec::new(); 3];
    let mut disp = [0.0f64; 3];
    for _ in 0..ROUNDS {
        for (at, f) in forms_list.iter().enumerate() {
            let (t, bld, d) = chain(forms, *f, region, &i, CALLS)?;
            us[at].push(t);
            build[at].push(bld);
            disp[at] = d;
        }
    }
    println!(
        "\n{title}, {rows} row(s): kernel against compiled, {moved} of {total} output elements differ"
    );
    println!(
        "{:<30} {:>9} {:>9} {:>11} {:>10} {:>10}",
        "", "chain us", "build us", "dispatches", "vs compiled", "us/token"
    );
    let base = median(us[1].clone());
    for (at, f) in forms_list.iter().enumerate() {
        let t = median(us[at].clone());
        println!(
            "{:<30} {:>9.1} {:>9.1} {:>11.1} {:>10.1} {:>10.0}",
            f.name(),
            t,
            median(build[at].clone()),
            disp[at],
            base - t,
            (base - t) * calls_a_token
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut forms = Forms::new()?;
    for rows in [1, 5] {
        section(
            &mut forms,
            Region::Residual,
            "attention conv + residual",
            rows,
            48.0,
        )?;
        section(&mut forms, Region::Next, "MLP conv + residual", rows, 48.0)?;
    }
    println!("\n(us/token: the saving per call times 48 calls; halve it for the model)");
    Ok(())
}
