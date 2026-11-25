//! What the attention projection's elementwise region costs today, and what replacing it with one
//! Metal kernel would buy, priced in a sandbox before anything is built into the model.
//!
//! ```text
//! cargo run --release -p lumen-mlx --features mlx-native --example inkling2_qkv_fold_profile
//! ```
//!
//! After the quantized matmul of a local attention layer, `inkling2/regions.rs::project_local` runs the
//! short convolutions of k and v, the L2 clips of q, k and the relative query, the value norm, and the
//! relative-bias matmul, and lays q, k, v out as `[B, H, L, D]`; `fused.rs` compiles it with
//! `mx.compile`. This example runs a copy of it (the composed ops through `mx.compile`, as a decode
//! step runs it) at the real model's local-layer shapes, one token, as calls each reading the last
//! one's output (so the GPU runs them one after another, as in the model), waited for once, and
//! counts the dispatches a call launches with MLX's command-buffer statistics.
//!
//! Every output of every call is evaluated, as the model uses them all: a chain that only evaluated the
//! outputs the next call reads left most of the region as dead code and priced it at 14 dispatches,
//! where the real region is 24.
//!
//! Two forms besides the shipped one: the plain composed ops, and the k and v convolutions as one
//! kernel launch each (the convolution kernel of `inkling2/conv_kernel.rs`) with the rest of the
//! region compiled. The second saves 6 dispatches and 9 to 18 us a call: the convolutions are a
//! quarter of the region, the clips and the norm the rest. What pays is one kernel for the whole
//! region but the relative-bias matmul, `inkling2/project_kernel.rs`: priced at 2.8 ms a token
//! as an upper bound from this baseline (107 us a call, 40 local visits), 0.73 ms in the model (20 runs
//! an arm against the version without it).

use std::time::Instant;

use anyhow::{Result, ensure};
use lumen_mlx::metal_kernel::{MetalKernel, MetalKernelConfig};
use mlx_rs::error::Exception;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::transforms::compile::compile_boxed_slice;
use mlx_rs::{Array, Dtype};

const TAPS: i32 = 4;
const HEADS: i32 = 24;
const KV_HEADS: i32 = 8;
const HEAD_DIM: i32 = 128;
const D_REL: i32 = 16;
const REL_EXTENT: i32 = 1024;
const QK_CLIP: f32 = 22.627417;
const REL_CLIP: f32 = 8.0;

fn scalar(v: f32) -> Array {
    Array::from_f32(v)
}

/// `ops::l2_clip`.
fn l2_clip(x: &Array, scale: &Array, threshold: f32) -> std::result::Result<Array, Exception> {
    let xf = mlx_rs::ops::multiply(x.as_dtype(Dtype::Float32)?, scale)?;
    let squares = mlx_rs::ops::multiply(&xf, &xf)?;
    let norm = mlx_rs::ops::sqrt(mlx_rs::ops::sum_axis(&squares, -1, true)?)?;
    let denom = mlx_rs::ops::add(&norm, scalar(1e-6))?;
    let factor =
        mlx_rs::ops::minimum(mlx_rs::ops::divide(scalar(threshold), &denom)?, scalar(1.0))?;
    mlx_rs::ops::multiply(&xf, &factor)?.as_dtype(x.dtype())
}

/// `ops::unit_rms`.
fn unit_rms(x: &Array, ones: &Array) -> std::result::Result<Array, Exception> {
    mlx_rs::fast::rms_norm(x, ones, 1e-6)
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

/// `regions::project_local`: arrays `[q, k, v, r, k_state, v_state, k_conv, v_conv, q_scale,
/// r_scale, rel_proj, ones]` -> `[q, k, v, rel, k_window, v_window]`.
fn project_local(args: &[Array]) -> std::result::Result<Vec<Array>, Exception> {
    let [
        q,
        k,
        v,
        r,
        k_state,
        v_state,
        k_conv,
        v_conv,
        q_scale,
        r_scale,
        rel_proj,
        ones,
    ] = args
    else {
        unreachable!("twelve arrays")
    };
    let conv_scale = (1.0f64 / f64::from(TAPS + 1).sqrt()) as f32;
    let (k, k_window) = static_conv(k, k_conv, k_state, conv_scale)?;
    let (v, v_window) = static_conv(v, v_conv, v_state, conv_scale)?;
    let (b, length) = (q.shape()[0], q.shape()[1]);
    let one = scalar(1.0);
    let q = l2_clip(
        &mlx_rs::ops::reshape(q, &[b, length, HEADS, HEAD_DIM])?,
        q_scale,
        QK_CLIP,
    )?;
    let k = l2_clip(
        &mlx_rs::ops::reshape(&k, &[b, length, KV_HEADS, HEAD_DIM])?,
        &one,
        QK_CLIP,
    )?;
    let v = unit_rms(
        &mlx_rs::ops::reshape(&v, &[b, length, KV_HEADS, HEAD_DIM])?.as_dtype(Dtype::Float32)?,
        ones,
    )?
    .as_dtype(q.dtype())?;
    let r = l2_clip(
        &mlx_rs::ops::reshape(r, &[b, length, HEADS, D_REL])?,
        r_scale,
        REL_CLIP,
    )?;
    let rel = mlx_rs::ops::matmul(&r, rel_proj.as_dtype(r.dtype())?)?;
    Ok(vec![
        mlx_rs::ops::transpose_axes(&q, &[0, 2, 1, 3])?,
        mlx_rs::ops::transpose_axes(&k, &[0, 2, 1, 3])?,
        mlx_rs::ops::transpose_axes(&v, &[0, 2, 1, 3])?,
        rel,
        k_window,
        v_window,
    ])
}

/// The convolution kernel of `inkling2/conv_kernel.rs` (a copy), here for the k and v convolutions of
/// the projection, which have no residual stream: it returns the convolved row in the row's dtype and
/// the padded window.
const HEADER: &str = r#"
template <typename T>
inline void store(device T* p, int i, float v) {
    p[i] = static_cast<T>(v);
}
template <typename T>
inline float round_as(const device T* proxy, float v) {
    return float(static_cast<T>(v));
}
template <typename T>
inline float round_as(const constant T* proxy, float v) {
    return float(static_cast<T>(v));
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
    const float y = round_as(x, (round_as(w, acc) + current) * scale);
    store(out, i, y);
    window[(TAPS - 1 + r) * C + c] = current;
    if (r == 0) {
        for (int t = 0; t < TAPS - 1; ++t) {
            window[t * C + c] = state[t * C + c];
        }
    }
"#;

/// `x` `[1, L, C]`, `w` `[C, taps]`, `state` `[1, taps - 1, C]` float32: the convolved row in `x`'s dtype
/// and the window, one launch.
fn kernel_conv(
    kernel: &MetalKernel,
    x: &Array,
    w: &Array,
    state: &Array,
    scale: f32,
) -> Result<(Array, Array)> {
    let (rows, c) = (x.shape()[1], x.shape()[2]);
    let x2 = mlx_rs::ops::reshape(x, &[rows, c])?;
    let state2 = mlx_rs::ops::reshape(state, &[TAPS - 1, c])?;
    let config = MetalKernelConfig::new();
    config.add_template_arg_int("C", c)?;
    config.add_template_arg_int("TAPS", TAPS)?;
    let threads = rows * c;
    config.set_grid(threads, 1, 1)?;
    config.set_thread_group(256.min(threads), 1, 1)?;
    config.add_output_arg(&[rows, c], x.dtype())?;
    config.add_output_arg(&[TAPS - 1 + rows, c], Dtype::Float32)?;
    let mut outs = kernel.apply(&[&x2, w, &state2, &scalar(scale)], &config, 2)?;
    let window = mlx_rs::ops::reshape(outs.swap_remove(1), &[1, TAPS - 1 + rows, c])?;
    let y = mlx_rs::ops::reshape(outs.swap_remove(0), &[1, rows, c])?;
    Ok((y, window))
}

/// `project_local` after the convolutions: arrays `[q, k, v, r, q_scale, r_scale, rel_proj, ones]`,
/// `k` and `v` already convolved -> `[q, k, v, rel]`.
fn project_tail(args: &[Array]) -> std::result::Result<Vec<Array>, Exception> {
    let [q, k, v, r, q_scale, r_scale, rel_proj, ones] = args else {
        unreachable!("eight arrays")
    };
    let (b, length) = (q.shape()[0], q.shape()[1]);
    let one = scalar(1.0);
    let q = l2_clip(
        &mlx_rs::ops::reshape(q, &[b, length, HEADS, HEAD_DIM])?,
        q_scale,
        QK_CLIP,
    )?;
    let k = l2_clip(
        &mlx_rs::ops::reshape(k, &[b, length, KV_HEADS, HEAD_DIM])?,
        &one,
        QK_CLIP,
    )?;
    let v = unit_rms(
        &mlx_rs::ops::reshape(v, &[b, length, KV_HEADS, HEAD_DIM])?.as_dtype(Dtype::Float32)?,
        ones,
    )?
    .as_dtype(q.dtype())?;
    let r = l2_clip(
        &mlx_rs::ops::reshape(r, &[b, length, HEADS, D_REL])?,
        r_scale,
        REL_CLIP,
    )?;
    let rel = mlx_rs::ops::matmul(&r, rel_proj.as_dtype(r.dtype())?)?;
    Ok(vec![
        mlx_rs::ops::transpose_axes(&q, &[0, 2, 1, 3])?,
        mlx_rs::ops::transpose_axes(&k, &[0, 2, 1, 3])?,
        mlx_rs::ops::transpose_axes(&v, &[0, 2, 1, 3])?,
        rel,
    ])
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

fn bf16(seed: u64, shape: &[i32], gain: f32) -> Result<Array> {
    let n: i32 = shape.iter().product();
    let v: Vec<f32> = noise(seed, n as usize)
        .into_iter()
        .map(|x| x * gain)
        .collect();
    Ok(Array::from_slice(&v, shape).as_dtype(Dtype::Bfloat16)?)
}

struct Inputs {
    rows: i32,
    /// The projection output `[1, L, dq + 2 dkv + dr + dq + nh]`, which the region reads q, k, v, r from.
    big: Array,
    k_state: Array,
    v_state: Array,
    k_conv: Array,
    v_conv: Array,
    q_scale: Array,
    r_scale: Array,
    rel_proj: Array,
    ones: Array,
    cuts: [i32; 5],
}

impl Inputs {
    fn new(rows: i32) -> Result<Self> {
        let (dq, dkv, dr) = (HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, HEADS * D_REL);
        let total = dq + 2 * dkv + dr + dq + HEADS;
        let i = Self {
            rows,
            big: bf16(1, &[1, rows, total], 3.0)?,
            k_state: Array::from_slice(&noise(2, (3 * dkv) as usize), &[1, 3, dkv]),
            v_state: Array::from_slice(&noise(3, (3 * dkv) as usize), &[1, 3, dkv]),
            k_conv: bf16(4, &[dkv, TAPS], 0.15)?,
            v_conv: bf16(5, &[dkv, TAPS], 0.15)?,
            q_scale: bf16(6, &[HEADS, HEAD_DIM], 1.0)?,
            r_scale: bf16(7, &[HEADS, D_REL], 1.0)?,
            rel_proj: bf16(8, &[D_REL, REL_EXTENT], 0.3)?,
            ones: mlx_rs::ops::ones::<f32>(&[HEAD_DIM])?,
            cuts: [
                dq,
                dq + dkv,
                dq + 2 * dkv,
                dq + 2 * dkv + dr,
                dq + 2 * dkv + dr + dq,
            ],
        };
        for a in [
            &i.big,
            &i.k_state,
            &i.v_state,
            &i.k_conv,
            &i.v_conv,
            &i.q_scale,
            &i.r_scale,
            &i.rel_proj,
            &i.ones,
        ] {
            a.eval()?;
        }
        Ok(i)
    }

    /// The region's arguments for a projection output `big` and the states: q, k, v, r are slices of it
    /// as the model's split gives them.
    fn args(&self, big: &Array, k_state: &Array, v_state: &Array) -> Result<Vec<Array>> {
        let [c0, c1, c2, c3, _] = self.cuts;
        Ok(vec![
            big.try_index((.., .., ..c0))?,
            big.try_index((.., .., c0..c1))?,
            big.try_index((.., .., c1..c2))?,
            big.try_index((.., .., c2..c3))?,
            k_state.clone(),
            v_state.clone(),
            self.k_conv.clone(),
            self.v_conv.clone(),
            self.q_scale.clone(),
            self.r_scale.clone(),
            self.rel_proj.clone(),
            self.ones.clone(),
        ])
    }
}

#[derive(Clone, Copy)]
enum Form {
    Plain,
    Compiled,
    /// The k and v convolutions as one kernel launch each, the rest of the region compiled.
    ConvKernels,
}

impl Form {
    fn name(self) -> &'static str {
        match self {
            Form::Plain => "plain ops (a prefill chunk)",
            Form::Compiled => "compiled (shipped)",
            Form::ConvKernels => "conv kernels + compiled rest",
        }
    }
}

type Compiled = Box<dyn FnMut(&[Array]) -> std::result::Result<Vec<Array>, Exception> + Send>;

struct Runners {
    full: Compiled,
    tail: Compiled,
    conv: MetalKernel,
}

impl Runners {
    fn new() -> Result<Self> {
        Ok(Self {
            full: compile_boxed_slice(project_local, false),
            tail: compile_boxed_slice(project_tail, false),
            conv: MetalKernel::new_with_header(
                "inkling2_qkv_fold_profile_conv",
                &["x", "w", "state", "scale"],
                &["out", "window"],
                HEADER,
                SOURCE,
                true,
                false,
            )?,
        })
    }
}

fn call(r: &mut Runners, form: Form, args: &[Array]) -> Result<Vec<Array>> {
    Ok(match form {
        Form::Plain => project_local(args)?,
        Form::Compiled => (r.full)(args)?,
        Form::ConvKernels => {
            let [
                q,
                k,
                v,
                rr,
                k_state,
                v_state,
                k_conv,
                v_conv,
                q_scale,
                r_scale,
                rel_proj,
                ones,
            ] = args
            else {
                unreachable!("twelve arrays")
            };
            let scale = (1.0f64 / f64::from(TAPS + 1).sqrt()) as f32;
            let (k, k_window) = kernel_conv(&r.conv, k, k_conv, k_state, scale)?;
            let (v, v_window) = kernel_conv(&r.conv, v, v_conv, v_state, scale)?;
            let mut out = (r.tail)(&[
                q.clone(),
                k,
                v,
                rr.clone(),
                q_scale.clone(),
                r_scale.clone(),
                rel_proj.clone(),
                ones.clone(),
            ])?;
            out.push(k_window);
            out.push(v_window);
            out
        }
    })
}

/// The next projection output and states from a call's outputs: a dependency on q (a one-element
/// slice scaled by zero and added), the windows' last rows as the states.
fn next(i: &Inputs, outs: &[Array]) -> Result<(Array, Array, Array)> {
    let probe = outs[0].try_index((0, 0, 0, ..1))?;
    let zero = mlx_rs::ops::multiply(probe.as_dtype(Dtype::Float32)?, scalar(0.0))?;
    let big = mlx_rs::ops::add(&i.big, zero.as_dtype(Dtype::Bfloat16)?)?;
    let cut = |w: &Array| -> Result<Array> {
        let len = w.shape()[1];
        Ok(w.try_index((.., (len - (TAPS - 1))..len, ..))?)
    };
    Ok((big, cut(&outs[4])?, cut(&outs[5])?))
}

fn bits(a: &Array) -> Result<Vec<u32>> {
    let flat = mlx_rs::ops::reshape(a.as_dtype(Dtype::Float32)?, &[-1])?;
    flat.eval()?;
    Ok(flat.as_slice::<f32>().iter().map(|x| x.to_bits()).collect())
}

/// A form against the compiled one over a chain of 20 calls (each call's windows and a dependency on
/// its q feed the next): the number of output elements that differ in a bit, and the total.
fn compare(r: &mut Runners, form: Form, i: &Inputs) -> Result<(usize, usize)> {
    let (mut ba, mut ksa, mut vsa) = (i.big.clone(), i.k_state.clone(), i.v_state.clone());
    let (mut bb, mut ksb, mut vsb) = (ba.clone(), ksa.clone(), vsa.clone());
    let (mut moved, mut total) = (0, 0);
    for _ in 0..20 {
        let want = call(r, Form::Compiled, &i.args(&ba, &ksa, &vsa)?)?;
        let got = call(r, form, &i.args(&bb, &ksb, &vsb)?)?;
        for (w, g) in want.iter().zip(&got) {
            let (w, g) = (bits(w)?, bits(g)?);
            ensure!(w.len() == g.len(), "shape");
            moved += w.iter().zip(&g).filter(|(a, b)| a != b).count();
            total += w.len();
        }
        (ba, ksa, vsa) = next(i, &want)?;
        (bb, ksb, vsb) = next(i, &got)?;
    }
    Ok((moved, total))
}

fn chain(r: &mut Runners, form: Form, i: &Inputs, calls: usize) -> Result<(f64, f64, f64)> {
    let mut big = i.big.clone();
    let (mut ks, mut vs) = (i.k_state.clone(), i.v_state.clone());
    let (_, d0) = mlx_rs::metal::cmd_buffer_stats().unwrap_or_default();
    let start = Instant::now();
    // Every output of every call is evaluated, as the model uses them all: only the ones the next call
    // reads would otherwise run, and the rest of the region would be dead code.
    let mut all: Vec<Array> = Vec::with_capacity(calls * 6);
    for _ in 0..calls {
        let out = call(r, form, &i.args(&big, &ks, &vs)?)?;
        (big, ks, vs) = next(i, &out)?;
        all.extend(out);
    }
    let build = start.elapsed().as_secs_f64() * 1e6 / calls as f64;
    mlx_rs::transforms::eval(all.iter())?;
    let total = start.elapsed().as_secs_f64() * 1e6 / calls as f64;
    let (_, d1) = mlx_rs::metal::cmd_buffer_stats().unwrap_or_default();
    ensure!(
        bits(&all[all.len() - 6])?
            .iter()
            .all(|b| f32::from_bits(*b).is_finite()),
        "non-finite output of the {} form",
        form.name()
    );
    Ok((total, build, d1.saturating_sub(d0) as f64 / calls as f64))
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<()> {
    const CALLS: usize = 300;
    const ROUNDS: usize = 9;
    let mut r = Runners::new()?;
    for rows in [1, 5] {
        let i = Inputs::new(rows)?;
        let forms = [Form::Plain, Form::Compiled, Form::ConvKernels];
        let (moved, total) = compare(&mut r, Form::ConvKernels, &i)?;
        for f in forms {
            chain(&mut r, f, &i, 30)?; // compile and warm
        }
        let mut us = vec![Vec::new(); forms.len()];
        let mut build = vec![Vec::new(); forms.len()];
        let mut disp = vec![0.0f64; forms.len()];
        for _ in 0..ROUNDS {
            for (at, f) in forms.iter().enumerate() {
                let (t, b, d) = chain(&mut r, *f, &i, CALLS)?;
                us[at].push(t);
                build[at].push(b);
                disp[at] = d;
            }
        }
        println!(
            "\nlocal projection region, {} row(s): conv kernels against compiled, {moved} of {total} output elements differ",
            i.rows
        );
        println!(
            "{:<30} {:>9} {:>9} {:>11} {:>11}",
            "", "chain us", "build us", "dispatches", "vs compiled"
        );
        let base = median(us[1].clone());
        for (at, f) in forms.iter().enumerate() {
            let t = median(us[at].clone());
            println!(
                "{:<30} {:>9.1} {:>9.1} {:>11.1} {:>11.1}",
                f.name(),
                t,
                median(build[at].clone()),
                disp[at],
                base - t
            );
        }
    }
    println!("\n(the saving per call times 40 local visits a token is the upper bound; halve it)");
    Ok(())
}
