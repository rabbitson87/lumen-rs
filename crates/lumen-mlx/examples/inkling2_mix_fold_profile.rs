//! What folding the two ops around the attention-residual mix into its kernel would buy, priced in
//! a sandbox before anything is built into the model.
//!
//! ```text
//! cargo run --release -p lumen-mlx --features mlx-native --example inkling2_mix_fold_profile
//! ```
//!
//! At a visit the model does, around the one-launch mix kernel (`inkling2/mix_kernel.rs`):
//!
//! * `unit_rms` of the partial sum: one `rms_norm` launch;
//! * the stack of the depth sources: `stack_axis` is a concatenate, one copy launch per source
//!   (three to four a visit on average), then a reshape that launches nothing.
//!
//! The four forms below are timed as one call each reading the last one's output (so the GPU runs
//! them one after another, as in the model), waited for once:
//!
//! * `shipped`: `rms_norm`, stack, kernel;
//! * `stack-free`: `rms_norm`, kernel reading the completed blocks from an array stacked once (in
//!   the model it would be re-stacked only when a block is added, a few times a token) and the
//!   partial sum from its own buffer;
//! * `rms in kernel`: stack (the raw partial sum last), kernel that normalises the last source;
//! * `both`: the blocks array and the raw partial sum, normalised in the kernel.
//!
//! The kernel is the shipped kernel's source (`float4` loads, a thread for each `float4`, one source
//! at a time) with two additions switched by template arguments: where the last source comes from
//! (`SPLIT`) and whether it is normalised first (`NORM`: the sum of squares over the hidden axis
//! added in the order `rms_norm` adds it, then `rsqrt`, so the scale is `rms_norm`'s to the bit).
//! Every form is checked against the shipped one before it is timed.
//!
//! A sandbox chain over-promises what the model gains, so the saving is printed per call and for
//! the ~96 calls of a plain decode token (48 of them with a bfloat16 partial sum) as an upper bound.
//! What the fold gained in the model, 20 same-session runs against the version without it: 0.60 ms a
//! plain step (standard error 0.06), about 80% of this chain's 0.6 to 0.7 ms.

use std::time::Instant;

use anyhow::{Result, ensure};
use lumen_mlx::metal_kernel::{MetalKernel, MetalKernelConfig};
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Dtype};

/// The shipped kernel with the last source's origin and normalisation switched by template
/// arguments. `blocks` is `[rows, nb, H]` (`SPLIT`: the completed blocks, the partial sum in `last`
/// `[rows, H]`) or `[rows, sources, H]` (joined: `last` is not read). With `NORM` the last source is
/// scaled by `rsqrt(mean(x^2) + 1e-6)`; `normed` receives it either way when `EMIT` is 1.
const SOURCE: &str = r#"
    constexpr int SIMDS = THREADS / 32;
    constexpr int H4 = H / 4;
    const int sources = SPLIT ? blocks_shape[1] + 1 : blocks_shape[1];
    const int nb = sources - 1;
    const int row = threadgroup_position_in_grid.x;
    const int tid = thread_position_in_threadgroup.x;
    const int sg = simdgroup_index_in_threadgroup;
    const int lane = thread_index_in_simdgroup;
    const device float4* base = (const device float4*)(blocks + (size_t)row * (SPLIT ? nb : sources) * H);
    const device auto* tail = last + (size_t)row * H;
    const device float4* wide_query = (const device float4*)query;
    device float4* mixed = (device float4*)out;
    device float4* emitted = (device float4*)normed;

    threadgroup float partial[SIMDS * TILE * Q];
    threadgroup float weight[MAX_SOURCES * Q];
    threadgroup float squares[SIMDS];

    float last_scale = 1.0f;
    if (NORM) {
        // The sum of squares in the order MLX's `rms_norm` adds them, so the scale is its own to
        // the bit (as in `mix_kernel.rs`).
        float ss = 0.0f;
        for (int h = tid; h < H4; h += THREADS) {
            const float4 v = SPLIT ? load4(tail, h) : base[(size_t)nb * H4 + h];
            ss += v.x * v.x;
            ss += v.y * v.y;
            ss += v.z * v.z;
            ss += v.w * v.w;
        }
        ss = simd_sum(ss);
        if (lane == 0) {
            squares[sg] = ss;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float total = simd_sum(lane < SIMDS ? squares[lane] : 0.0f);
        last_scale = metal::precise::rsqrt(total / float(last_shape[1]) + 1e-6f);
    }
    if (EMIT) {
        for (int h = tid; h < H4; h += THREADS) {
            const float4 v = SPLIT ? load4(tail, h) : base[(size_t)nb * H4 + h];
            emitted[(size_t)row * H4 + h] = v * last_scale;
        }
    }

#define SOURCE_AT(s, h) \
    (SPLIT ? ((s) < nb ? base[(size_t)(s) * H4 + (h)] : load4(tail, (h)) * last_scale) \
           : base[(size_t)(s) * H4 + (h)] * ((s) == nb ? last_scale : 1.0f))

    for (int s0 = 0; s0 < sources; s0 += TILE) {
        const int n = min(TILE, sources - s0);
        float acc[TILE * Q];
        for (int i = 0; i < TILE * Q; ++i) {
            acc[i] = 0.0f;
        }
        for (int h = tid; h < H4; h += THREADS) {
            float4 wanted[Q];
            for (int q = 0; q < Q; ++q) {
                wanted[q] = wide_query[q * H4 + h];
            }
            for (int i = 0; i < TILE; ++i) {
                if (i < n) {
                    const float4 v = SOURCE_AT(s0 + i, h);
                    for (int q = 0; q < Q; ++q) {
                        acc[i * Q + q] += metal::dot(v, wanted[q]);
                    }
                }
            }
        }
        for (int i = 0; i < n * Q; ++i) {
            const float sum = simd_sum(acc[i]);
            if (lane == 0) {
                partial[sg * TILE * Q + i] = sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < n * Q) {
            float total = 0.0f;
            for (int g = 0; g < SIMDS; ++g) {
                total += partial[g * TILE * Q + tid];
            }
            weight[s0 * Q + tid] = total * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < Q) {
        float top = -INFINITY;
        for (int s = 0; s < sources; ++s) {
            top = max(top, weight[s * Q + tid]);
        }
        float total = 0.0f;
        for (int s = 0; s < sources; ++s) {
            const float e = exp(weight[s * Q + tid] - top);
            weight[s * Q + tid] = e;
            total += e;
        }
        for (int s = 0; s < sources; ++s) {
            weight[s * Q + tid] /= total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int h = tid; h < H4; h += THREADS) {
        float4 acc[Q];
        for (int q = 0; q < Q; ++q) {
            acc[q] = float4(0.0f);
        }
        for (int s = 0; s < sources; ++s) {
            const float4 v = SOURCE_AT(s, h);
            for (int q = 0; q < Q; ++q) {
                acc[q] += weight[s * Q + q] * v;
            }
        }
        for (int q = 0; q < Q; ++q) {
            mixed[((size_t)row * Q + q) * H4 + h] = acc[q];
        }
    }
"#;

/// Four values of a row as a `float4`: a vector load from float32, four converted loads otherwise (a
/// bfloat16 partial sum the kernel reads as it is, instead of a cast launch ahead of it).
const HEADER: &str = r#"
inline float4 load4(const device float* p, int i) {
    return ((const device float4*)p)[i];
}
template <typename T>
inline float4 load4(const device T* p, int i) {
    return float4(float(p[4 * i]), float(p[4 * i + 1]), float(p[4 * i + 2]), float(p[4 * i + 3]));
}
"#;

const TILE: i32 = 12;
const MAX_SOURCES: i32 = 32;

#[derive(Clone, Copy)]
struct Form {
    name: &'static str,
    /// The blocks come stacked once, the partial sum in its own buffer.
    split: bool,
    /// The kernel normalises the partial sum.
    norm: bool,
    /// The partial sum arrives as bfloat16 (the attention input's does): a cast launch ahead of
    /// the `rms_norm`, or read as it is by the kernel when that normalises.
    bf16: bool,
}

const FORMS: [Form; 4] = [
    Form {
        name: "shipped (rms_norm, stack, kernel)",
        split: false,
        norm: false,
        bf16: false,
    },
    Form {
        name: "stack-free (rms_norm, kernel)",
        split: true,
        norm: false,
        bf16: false,
    },
    Form {
        name: "rms in kernel (stack, kernel)",
        split: false,
        norm: true,
        bf16: false,
    },
    Form {
        name: "both (kernel)",
        split: true,
        norm: true,
        bf16: false,
    },
];

/// The attention input's partial sum is bfloat16.
const BF16_FORMS: [Form; 2] = [
    Form {
        name: "shipped (cast, rms_norm, stack, kernel)",
        split: false,
        norm: false,
        bf16: true,
    },
    Form {
        name: "both (kernel reads the bfloat16)",
        split: true,
        norm: true,
        bf16: true,
    },
];

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    rows: i32,
    /// Completed blocks (the model's count a visit is 3 or 4 on average, up to 6).
    blocks: i32,
    queries: i32,
    hidden: i32,
}

const THREADS_MIN: i32 = 64;

fn threads_for(hidden: i32) -> i32 {
    ((hidden / 4 + 31) / 32 * 32).clamp(THREADS_MIN, 1024)
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

/// A `[rows, count, hidden]` stack of rows at unit RMS, as the blocks are.
fn unit_rows(seed: u64, rows: i32, count: i32, hidden: i32) -> Array {
    let mut v = noise(seed, (rows * count * hidden) as usize);
    for row in v.chunks_mut(hidden as usize) {
        let rms = (row.iter().map(|x| x * x).sum::<f32>() / hidden as f32).sqrt();
        row.iter_mut().for_each(|x| *x /= rms);
    }
    Array::from_slice(&v, &[rows, count, hidden])
}

/// Everything a call reads, built once.
struct Inputs {
    shape: Shape,
    /// The completed blocks, `[rows, blocks, hidden]`, and one array per block (`[1, rows, hidden]`).
    stacked: Array,
    each: Vec<Array>,
    /// The raw partial sum a call starts from, `[1, rows, hidden]`, not at unit RMS.
    partial: Array,
    query: Array,
    scale: Array,
    ones: Array,
    /// A stand-in for the `last` input of a joined launch (a device buffer: MLX hands a kernel a
    /// one-float input in the `constant` address space).
    unused: Array,
}

impl Inputs {
    fn new(shape: Shape) -> Result<Self> {
        let Shape {
            rows,
            blocks,
            queries,
            hidden,
            ..
        } = shape;
        let stacked = unit_rows(1, rows, blocks, hidden);
        let each: Vec<Array> = (0..blocks)
            .map(|b| {
                stacked
                    .try_index((.., b, ..))
                    .and_then(|a| a.reshape(&[1, rows, hidden]))
            })
            .collect::<std::result::Result<_, _>>()?;
        let raw: Vec<f32> = noise(2, (rows * hidden) as usize)
            .into_iter()
            .map(|x| x * 3.0)
            .collect();
        let inputs = Self {
            shape,
            stacked,
            each,
            partial: Array::from_slice(&raw, &[1, rows, hidden]),
            query: Array::from_slice(&noise(3, (queries * hidden) as usize), &[queries, hidden]),
            scale: Array::from_f32((hidden as f32).powf(-0.5)),
            ones: mlx_rs::ops::ones::<f32>(&[hidden])?,
            unused: mlx_rs::ops::zeros::<f32>(&[rows, hidden])?,
        };
        for a in inputs.each.iter().chain([
            &inputs.stacked,
            &inputs.partial,
            &inputs.query,
            &inputs.ones,
        ]) {
            a.eval()?;
        }
        Ok(inputs)
    }
}

struct Kernel(MetalKernel);

impl Kernel {
    fn new() -> Result<Self> {
        Ok(Self(MetalKernel::new_with_header(
            "inkling2_mix_fold_profile",
            &["blocks", "last", "query", "scale"],
            &["out", "normed"],
            HEADER,
            SOURCE,
            true,
            false,
        )?))
    }

    /// One call of `form` on `partial` (`[1, rows, hidden]`): the mix, `[1, rows, queries, hidden]`.
    fn call(&self, form: Form, i: &Inputs, partial: &Array) -> Result<Array> {
        let Shape {
            rows,
            blocks,
            queries,
            hidden,
            ..
        } = i.shape;
        let sources = blocks + 1;
        let threads = threads_for(hidden);
        // The raw partial sum, or its unit-RMS form from the composed op.
        let tail = if form.norm {
            partial.clone()
        } else if form.bf16 {
            mlx_rs::fast::rms_norm(&partial.as_dtype(Dtype::Float32)?, &i.ones, 1e-6)?
        } else {
            mlx_rs::fast::rms_norm(partial, &i.ones, 1e-6)?
        };
        let joined;
        let (first, last): (&Array, &Array) = if form.split {
            (&i.stacked, &tail)
        } else {
            let mut all: Vec<&Array> = i.each.iter().collect();
            all.push(&tail);
            joined =
                mlx_rs::ops::reshape(mlx_rs::ops::stack_axis(&all, 2)?, &[rows, sources, hidden])?;
            (&joined, &i.unused)
        };
        let last = if form.split {
            mlx_rs::ops::reshape(last, &[rows, hidden])?
        } else {
            last.clone()
        };
        let config = MetalKernelConfig::new();
        config.add_template_arg_int("H", hidden)?;
        config.add_template_arg_int("Q", queries)?;
        config.add_template_arg_int("THREADS", threads)?;
        config.add_template_arg_int("TILE", TILE)?;
        config.add_template_arg_int("MAX_SOURCES", MAX_SOURCES)?;
        config.add_template_arg_int("SPLIT", i32::from(form.split))?;
        config.add_template_arg_int("NORM", i32::from(form.norm))?;
        config.add_template_arg_int("EMIT", 1)?;
        config.set_grid(rows * threads, 1, 1)?;
        config.set_thread_group(threads, 1, 1)?;
        config.add_output_arg(&[rows, queries, hidden], Dtype::Float32)?;
        config.add_output_arg(&[rows, hidden], Dtype::Float32)?;
        let mut outs = self
            .0
            .apply(&[first, &last, &i.query, &i.scale], &config, 2)?;
        let _normed = outs.swap_remove(1);
        let mixed = outs.swap_remove(0);
        Ok(mlx_rs::ops::reshape(&mixed, &[1, rows, queries, hidden])?)
    }
}

fn next_partial(form: Form, mixed: &Array) -> Result<Array> {
    let next = mixed.try_index((.., .., 0, ..))?;
    Ok(if form.bf16 {
        next.as_dtype(Dtype::Bfloat16)?
    } else {
        next
    })
}

/// The partial sum a chain starts from, in the form's dtype.
fn first_partial(form: Form, i: &Inputs) -> Result<Array> {
    Ok(if form.bf16 {
        i.partial.as_dtype(Dtype::Bfloat16)?
    } else {
        i.partial.clone()
    })
}

fn host(a: &Array) -> Result<Vec<f32>> {
    let flat = mlx_rs::ops::reshape(a, &[-1])?;
    flat.eval()?;
    Ok(flat.as_slice::<f32>().to_vec())
}

/// The form's first call against the reference's: the same mix to float32 rounding.
fn check(kernel: &Kernel, i: &Inputs, reference: Form, form: Form) -> Result<f32> {
    let want = host(&kernel.call(reference, i, &first_partial(reference, i)?)?)?;
    let got = host(&kernel.call(form, i, &first_partial(form, i)?)?)?;
    ensure!(want.len() == got.len(), "{}: shape", form.name);
    let worst = want
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    ensure!(
        worst < 2e-5,
        "{}: not the mix, worst {worst:.1e}",
        form.name
    );
    Ok(worst)
}

/// Microseconds a call over a chain of `calls`, each reading the last one's mix; and the CPU's share.
fn chain_us(kernel: &Kernel, i: &Inputs, form: Form, calls: usize) -> Result<(f64, f64)> {
    let start = Instant::now();
    let mut partial = first_partial(form, i)?;
    for _ in 0..calls {
        partial = next_partial(form, &kernel.call(form, i, &partial)?)?;
    }
    let build = start.elapsed().as_secs_f64() * 1e6 / calls as f64;
    partial.eval()?;
    ensure!(
        host(&partial.as_dtype(Dtype::Float32)?)?
            .iter()
            .all(|x| x.is_finite()),
        "non-finite"
    );
    Ok((start.elapsed().as_secs_f64() * 1e6 / calls as f64, build))
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

const CALLS: usize = 400;
const ROUNDS: usize = 9;

/// Times `forms` on `shape` round-robin (so that drift falls on all of them), the first form being
/// the reference, and prints each one's saving against it per call and for `calls_a_token` calls.
fn section(
    kernel: &Kernel,
    shape: Shape,
    title: &str,
    forms: &[Form],
    calls_a_token: f64,
) -> Result<()> {
    let inputs = Inputs::new(shape)?;
    let mut diffs = Vec::new();
    for form in forms {
        diffs.push(check(kernel, &inputs, forms[0], *form)?);
        chain_us(kernel, &inputs, *form, 40)?; // compile and warm
    }
    let mut chain: Vec<Vec<f64>> = vec![Vec::new(); forms.len()];
    let mut build: Vec<Vec<f64>> = vec![Vec::new(); forms.len()];
    for _ in 0..ROUNDS {
        for (at, form) in forms.iter().enumerate() {
            let (c, b) = chain_us(kernel, &inputs, *form, CALLS)?;
            chain[at].push(c);
            build[at].push(b);
        }
    }
    println!("\n{title}: {}", shape.name);
    println!(
        "{:<40} {:>9} {:>9} {:>10} {:>9} {:>10}",
        "", "chain us", "build us", "vs first", "us/token", "vs first"
    );
    let base = median(chain[0].clone());
    for (at, form) in forms.iter().enumerate() {
        let c = median(chain[at].clone());
        println!(
            "{:<40} {:>9.1} {:>9.1} {:>10.1} {:>9.0} {:>10.1e}",
            form.name,
            c,
            median(build[at].clone()),
            base - c,
            (base - c) * calls_a_token,
            diffs[at]
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let shapes = [
        Shape {
            name: "attn_input-like: 3 blocks, 3 queries",
            rows: 1,
            blocks: 3,
            queries: 3,
            hidden: 2304,
        },
        Shape {
            name: "mlp_input-like: 3 blocks, 1 query",
            rows: 1,
            blocks: 3,
            queries: 1,
            hidden: 2304,
        },
        Shape {
            name: "6 blocks, 3 queries",
            rows: 1,
            blocks: 6,
            queries: 3,
            hidden: 2304,
        },
        Shape {
            name: "3 blocks, 3 queries, 5 rows (a verify pass)",
            rows: 5,
            blocks: 3,
            queries: 3,
            hidden: 2304,
        },
    ];
    let kernel = Kernel::new()?;
    for shape in shapes {
        section(&kernel, shape, "float32 partial sum", &FORMS, 96.0)?;
    }
    // Only the attention input's partial sum is bfloat16: 48 calls a token.
    for shape in [shapes[0], shapes[2]] {
        section(&kernel, shape, "bfloat16 partial sum", &BF16_FORMS, 48.0)?;
    }
    println!("\n(us/token: the saving per call times the calls a token; halve it for the model)");
    Ok(())
}
