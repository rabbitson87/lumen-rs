//! Where the time of the attention-residual mix kernel (`inkling2/mix_kernel.rs`) goes, and which
//! changes to it pay.
//!
//! ```text
//! cargo run --release -p lumen-mlx --features mlx-native --example inkling2_mix_kernel_profile
//! ```
//!
//! A decode token's mix is one launch of one threadgroup on one GPU core, about 120 times a token.
//!
//! 1. Phases: copies of the kernel with phases left out (the dot products, the softmax, the weighted
//!    sum) while every launch, dependency and output write stays, so the drop in time against the
//!    full kernel is that phase's work, and the empty kernel is what a launch costs with none.
//! 2. Variants: the three changes that could make one threadgroup read faster, alone and together,
//!    over threads a group: more threads (more loads in flight), `float4` loads (four hidden values
//!    a load), and batched loads (every load of a tile of sources issued before the first multiply,
//!    instead of load, multiply, load, multiply).
//! 3. The best variant against the baseline over the shapes the model reaches.
//!
//! Each case is timed two ways, because the model is GPU-bound and a launch hides behind the CPU
//! only when the GPU is slower than the encoder:
//! * `sync`: one call, then wait for it. The time is encode + submit + GPU + wake-up, the same for
//!   every variant but the GPU part, so differences between variants are GPU time.
//! * `chain`: calls each reading the last one's output (so the GPU runs them one after another, as in
//!   the model), waited for once. `build` is the CPU's share of it (graph building and encoding);
//!   when `chain` is no more than `build` the CPU, not the kernel, was the limit.
//!
//! The kernel source is a copy of `mix_kernel.rs`'s with the phases and the variants switched by
//! template arguments and the output as wide as the input (so a call can read the last one's
//! output); every full variant is checked against a host computation before it is timed. The
//! baseline (`Case::MODEL`) is the kernel as it was before this sweep (scalar loads, 256 threads);
//! `mix_kernel.rs` now holds the sweep's winner (`float4` loads, a thread for each `float4`, one
//! source at a time), so the sweep's best row is the shipped kernel's time.

use std::time::Instant;

use anyhow::{Result, ensure};
use lumen_mlx::metal_kernel::{MetalKernel, MetalKernelConfig};
use mlx_rs::{Array, Dtype};

/// The kernel body, written over `Load` (a `float`, or a `float4` that reads four hidden values a
/// load), `DOT` and `WIDTH`, which the header of each of the two kernels defines. `HV` is the hidden
/// axis in `Load`s. `PHASES` bits: 1 the dot products, 2 the softmax, 4 the weighted sum; a phase left
/// out is replaced by the least that keeps the next one valid (uniform weights; one output write).
/// `BATCH` 0 loads a source and multiplies it, one source at a time; 1 loads a tile's sources before
/// the first multiply.
const BODY: &str = r#"
    constexpr int MAX_SOURCES = 32;
    constexpr int TILE = 12;
    constexpr int SIMDS = THREADS / 32;
    constexpr int HV = H / WIDTH;
    const int sources = src_shape[1];
    const int row = threadgroup_position_in_grid.x;
    const int tid = thread_position_in_threadgroup.x;
    const int sg = simdgroup_index_in_threadgroup;
    const int lane = thread_index_in_simdgroup;
    const device float* base = src + (size_t)row * sources * H;
    const device Load* bv = (const device Load*)base;
    const device Load* qv = (const device Load*)query;
    device Load* ov = (device Load*)out;

    threadgroup float partial[SIMDS * TILE * Q];
    threadgroup float weight[MAX_SOURCES * Q];

    if (PHASES & 1) {
        for (int s0 = 0; s0 < sources; s0 += TILE) {
            const int n = min(TILE, sources - s0);
            float acc[TILE * Q];
            for (int i = 0; i < TILE * Q; ++i) {
                acc[i] = 0.0f;
            }
            for (int h = tid; h < HV; h += THREADS) {
                Load wanted[Q];
                for (int q = 0; q < Q; ++q) {
                    wanted[q] = qv[q * HV + h];
                }
                if (BATCH) {
                    Load v[TILE];
                    for (int i = 0; i < TILE; ++i) {
                        v[i] = i < n ? bv[(size_t)(s0 + i) * HV + h] : Load(0.0f);
                    }
                    for (int i = 0; i < TILE; ++i) {
                        for (int q = 0; q < Q; ++q) {
                            acc[i * Q + q] += DOT(v[i], wanted[q]);
                        }
                    }
                } else {
                    for (int i = 0; i < TILE; ++i) {
                        if (i < n) {
                            const Load v = bv[(size_t)(s0 + i) * HV + h];
                            for (int q = 0; q < Q; ++q) {
                                acc[i * Q + q] += DOT(v, wanted[q]);
                            }
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
    } else {
        if (tid < sources * Q) {
            weight[tid] = 1.0f / sources;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (PHASES & 2) {
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
    }

    if (PHASES & 4) {
        for (int h = tid; h < HV; h += THREADS) {
            Load acc[Q];
            for (int q = 0; q < Q; ++q) {
                acc[q] = Load(0.0f);
            }
            for (int s0 = 0; s0 < sources; s0 += TILE) {
                const int n = min(TILE, sources - s0);
                if (BATCH) {
                    Load v[TILE];
                    for (int i = 0; i < TILE; ++i) {
                        v[i] = i < n ? bv[(size_t)(s0 + i) * HV + h] : Load(0.0f);
                    }
                    for (int i = 0; i < TILE; ++i) {
                        if (i < n) {
                            for (int q = 0; q < Q; ++q) {
                                acc[q] += weight[(s0 + i) * Q + q] * v[i];
                            }
                        }
                    }
                } else {
                    for (int i = 0; i < n; ++i) {
                        const Load v = bv[(size_t)(s0 + i) * HV + h];
                        for (int q = 0; q < Q; ++q) {
                            acc[q] += weight[(s0 + i) * Q + q] * v;
                        }
                    }
                }
            }
            for (int q = 0; q < Q; ++q) {
                ov[((size_t)row * sources + q) * HV + h] = acc[q];
            }
        }
    } else if (tid == 0) {
        out[(size_t)row * sources * H] = weight[0];
    }
"#;

/// Header of the scalar kernel: one hidden value a load.
const SCALAR_HEADER: &str = "using Load = float;\n#define DOT(a, b) ((a) * (b))\n#define WIDTH 1\n";
/// Header of the wide kernel: four hidden values a load (the hidden size must be a multiple of 4).
const WIDE_HEADER: &str =
    "using Load = float4;\n#define DOT(a, b) metal::dot((a), (b))\n#define WIDTH 4\n";

const DOT: i32 = 1;
const SOFTMAX: i32 = 2;
const WEIGHTED_SUM: i32 = 4;
const FULL: i32 = DOT | SOFTMAX | WEIGHTED_SUM;

/// The two compiled forms of [`BODY`].
struct Kernels {
    scalar: MetalKernel,
    wide: MetalKernel,
}

impl Kernels {
    fn new() -> Result<Self> {
        Ok(Self {
            scalar: MetalKernel::new_with_header(
                "inkling2_mix_kernel_profile_scalar",
                &["src", "query", "scale"],
                &["out"],
                SCALAR_HEADER,
                BODY,
                true,
                false,
            )?,
            wide: MetalKernel::new_with_header(
                "inkling2_mix_kernel_profile_wide",
                &["src", "query", "scale"],
                &["out"],
                WIDE_HEADER,
                BODY,
                true,
                false,
            )?,
        })
    }

    fn of(&self, c: Case) -> &MetalKernel {
        if c.wide { &self.wide } else { &self.scalar }
    }
}

#[derive(Clone, Copy)]
struct Case {
    rows: i32,
    sources: i32,
    queries: i32,
    hidden: i32,
    threads: i32,
    phases: i32,
    /// `float4` loads.
    wide: bool,
    /// 1: a tile's loads before its first multiply.
    batch: i32,
}

impl Case {
    /// The model's attention-input mix at one token: six blocks and the partial sum, three queries;
    /// the baseline: the kernel as it was before the sweep (scalar loads, one source at a time, 256 threads).
    const MODEL: Case = Case {
        rows: 1,
        sources: 7,
        queries: 3,
        hidden: 2304,
        threads: 256,
        phases: FULL,
        wide: false,
        batch: 0,
    };
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

/// `[rows, sources, hidden]` at unit RMS per row, and a `[queries, hidden]` table.
fn inputs(c: Case) -> (Vec<f32>, Vec<f32>) {
    let mut src = noise(1, (c.rows * c.sources * c.hidden) as usize);
    for row in src.chunks_mut(c.hidden as usize) {
        let rms = (row.iter().map(|x| x * x).sum::<f32>() / c.hidden as f32).sqrt();
        row.iter_mut().for_each(|x| *x /= rms);
    }
    (src, noise(2, (c.queries * c.hidden) as usize))
}

/// The mix on the host, in f64: `[rows, queries, hidden]`.
fn host_mix(c: Case, src: &[f32], query: &[f32], scale: f32) -> Vec<f32> {
    let (s, q, h) = (c.sources as usize, c.queries as usize, c.hidden as usize);
    let mut out = vec![0.0f32; c.rows as usize * q * h];
    for row in 0..c.rows as usize {
        let at = |i: usize| &src[(row * s + i) * h..][..h];
        for qi in 0..q {
            let logits: Vec<f64> = (0..s)
                .map(|i| {
                    let dot: f64 = at(i)
                        .iter()
                        .zip(&query[qi * h..][..h])
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum();
                    dot * f64::from(scale)
                })
                .collect();
            let top = logits.iter().copied().fold(f64::MIN, f64::max);
            let exps: Vec<f64> = logits.iter().map(|l| (l - top).exp()).collect();
            let total: f64 = exps.iter().sum();
            for hi in 0..h {
                let v: f64 = (0..s).map(|i| exps[i] / total * f64::from(at(i)[hi])).sum();
                out[(row * q + qi) * h + hi] = v as f32;
            }
        }
    }
    out
}

fn launch(kernels: &Kernels, c: Case, src: &Array, query: &Array, scale: &Array) -> Result<Array> {
    ensure!(
        !c.wide || c.hidden % 4 == 0,
        "float4 loads need a hidden size that is a multiple of 4"
    );
    let config = MetalKernelConfig::new();
    config.add_template_arg_int("H", c.hidden)?;
    config.add_template_arg_int("Q", c.queries)?;
    config.add_template_arg_int("THREADS", c.threads)?;
    config.add_template_arg_int("PHASES", c.phases)?;
    config.add_template_arg_int("BATCH", c.batch)?;
    config.set_grid(c.rows * c.threads, 1, 1)?;
    config.set_thread_group(c.threads, 1, 1)?;
    config.add_output_arg(&[c.rows, c.sources, c.hidden], Dtype::Float32)?;
    Ok(kernels
        .of(c)
        .apply(&[src, query, scale], &config, 1)?
        .swap_remove(0))
}

/// Runs a case once. A full kernel is also compared with the host mix; its worst difference is
/// returned (a kernel that is not the mix has no time worth reading).
fn probe(kernels: &Kernels, c: Case) -> Result<Option<f32>> {
    let (src_host, query_host) = inputs(c);
    let scale_value = (c.hidden as f32).powf(-0.5);
    let got = launch(
        kernels,
        c,
        &Array::from_slice(&src_host, &[c.rows, c.sources, c.hidden]),
        &Array::from_slice(&query_host, &[c.queries, c.hidden]),
        &Array::from_f32(scale_value),
    )?;
    got.eval()?;
    if c.phases != FULL {
        return Ok(None);
    }
    let got = got.as_slice::<f32>();
    let want = host_mix(c, &src_host, &query_host, scale_value);
    let (s, q, h) = (c.sources as usize, c.queries as usize, c.hidden as usize);
    let mut worst = 0.0f32;
    for row in 0..c.rows as usize {
        for qi in 0..q {
            for hi in 0..h {
                worst =
                    worst.max((got[(row * s + qi) * h + hi] - want[(row * q + qi) * h + hi]).abs());
            }
        }
    }
    ensure!(worst < 1e-5, "not the mix: worst difference {worst:.1e}");
    Ok(Some(worst))
}

#[derive(Clone, Copy, Default)]
struct Timing {
    sync_us: f64,
    chain_us: f64,
    build_us: f64,
}

fn measure(kernels: &Kernels, c: Case, calls: usize) -> Result<Timing> {
    let (src_host, query_host) = inputs(c);
    let src = Array::from_slice(&src_host, &[c.rows, c.sources, c.hidden]);
    let query = Array::from_slice(&query_host, &[c.queries, c.hidden]);
    let scale = Array::from_f32((c.hidden as f32).powf(-0.5));
    src.eval()?;
    query.eval()?;

    let start = Instant::now();
    for _ in 0..calls {
        launch(kernels, c, &src, &query, &scale)?.eval()?;
    }
    let sync_us = start.elapsed().as_secs_f64() * 1e6 / calls as f64;

    let start = Instant::now();
    let mut cur = launch(kernels, c, &src, &query, &scale)?;
    for _ in 1..calls {
        cur = launch(kernels, c, &cur, &query, &scale)?;
    }
    let build_us = start.elapsed().as_secs_f64() * 1e6 / calls as f64;
    cur.eval()?;
    let chain_us = start.elapsed().as_secs_f64() * 1e6 / calls as f64;
    Ok(Timing {
        sync_us,
        chain_us,
        build_us,
    })
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Times each case `rounds` times, round-robin so that drift of the machine falls on all of them.
/// A case that does not run (the device refuses its threadgroup) is listed as skipped; one that runs
/// and is not the mix stops the example. Returns the timings aligned with `cases`.
fn table(
    title: &str,
    kernels: &Kernels,
    cases: &[(String, Case)],
    calls: usize,
    rounds: usize,
) -> Result<Vec<Option<Timing>>> {
    let checks: Vec<Result<Option<f32>>> = cases.iter().map(|(_, c)| probe(kernels, *c)).collect();
    for ((name, _), check) in cases.iter().zip(&checks) {
        if let Err(e) = check {
            ensure!(!e.to_string().contains("not the mix"), "{name}: {e}");
        }
    }
    let mut samples: Vec<Vec<Timing>> = vec![Vec::new(); cases.len()];
    for _ in 0..rounds {
        for (at, (_, c)) in cases.iter().enumerate() {
            if checks[at].is_ok() {
                samples[at].push(measure(kernels, *c, calls)?);
            }
        }
    }
    println!("\n{title}");
    println!(
        "{:<40} {:>10} {:>10} {:>10} {:>9}",
        "", "sync us", "chain us", "build us", "vs host"
    );
    let mut out = Vec::new();
    for (((name, _), s), check) in cases.iter().zip(&samples).zip(&checks) {
        match check {
            Err(e) => {
                println!("{name:<40} skipped: {e}");
                out.push(None);
            }
            Ok(worst) => {
                let t = Timing {
                    sync_us: median(s.iter().map(|t| t.sync_us).collect()),
                    chain_us: median(s.iter().map(|t| t.chain_us).collect()),
                    build_us: median(s.iter().map(|t| t.build_us).collect()),
                };
                let diff = worst.map_or_else(|| "-".to_string(), |w| format!("{w:.1e}"));
                println!(
                    "{name:<40} {:>10.1} {:>10.1} {:>10.1} {diff:>9}",
                    t.sync_us, t.chain_us, t.build_us
                );
                out.push(Some(t));
            }
        }
    }
    Ok(out)
}

fn variant_name(c: Case) -> String {
    format!(
        "{} loads, {}, {} threads",
        if c.wide { "float4" } else { "scalar" },
        if c.batch == 1 {
            "batched"
        } else {
            "one by one"
        },
        c.threads
    )
}

fn main() -> Result<()> {
    let kernels = Kernels::new()?;

    // Every variant must be the mix before its time means anything: odd shapes first (a hidden size
    // that is not a multiple of the threads, one source, a source count of several tiles).
    for wide in [false, true] {
        for batch in [0, 1] {
            for (sources, queries, hidden, threads) in [
                (1, 1, 100, 128),
                (9, 3, 100, 128),
                (32, 4, 2304, 576),
                (13, 3, 2304, 256),
            ] {
                let c = Case {
                    sources,
                    queries,
                    hidden,
                    threads,
                    wide,
                    batch,
                    ..Case::MODEL
                };
                let worst = probe(&kernels, c)?.unwrap_or(0.0);
                println!(
                    "{}: S={sources} Q={queries} H={hidden}: worst difference from the host mix {worst:.1e}",
                    variant_name(c)
                );
            }
        }
    }

    let calls = 600;
    let rounds = 7;

    // 1. Phases, cumulative: what each adds to the one before.
    let mut cases = Vec::new();
    for (name, phases) in [
        ("empty (launch only)", 0),
        ("+ dot products", DOT),
        ("+ softmax", DOT | SOFTMAX),
        ("+ weighted sum (the full kernel)", FULL),
        ("weighted sum alone (no dot, softmax)", WEIGHTED_SUM),
    ] {
        cases.push((
            name.to_string(),
            Case {
                phases,
                ..Case::MODEL
            },
        ));
    }
    let t = table(
        "phases at the model's shape (7 sources, 3 queries, hidden 2304, 256 threads, 1 row, the baseline kernel)",
        &kernels,
        &cases,
        calls,
        rounds,
    )?;
    let t: Vec<Timing> = t.into_iter().map(Option::unwrap_or_default).collect();
    println!(
        "  added by the dot products: sync {:+.1} us, chain {:+.1} us",
        t[1].sync_us - t[0].sync_us,
        t[1].chain_us - t[0].chain_us
    );
    println!(
        "  added by the softmax:      sync {:+.1} us, chain {:+.1} us",
        t[2].sync_us - t[1].sync_us,
        t[2].chain_us - t[1].chain_us
    );
    println!(
        "  added by the weighted sum: sync {:+.1} us, chain {:+.1} us",
        t[3].sync_us - t[2].sync_us,
        t[3].chain_us - t[2].chain_us
    );
    println!(
        "  the whole kernel over an empty launch: sync {:+.1} us, chain {:+.1} us",
        t[3].sync_us - t[0].sync_us,
        t[3].chain_us - t[0].chain_us
    );

    // 2. The three changes, alone and together, over threads a threadgroup. 2304 hidden values are
    // 576 float4s: 192, 288 and 576 threads divide them evenly, 256 does not.
    let mut cases = Vec::new();
    for wide in [false, true] {
        for batch in [0, 1] {
            for threads in [128, 192, 256, 288, 384, 512, 576, 768, 1024] {
                let c = Case {
                    threads,
                    wide,
                    batch,
                    ..Case::MODEL
                };
                cases.push((variant_name(c), c));
            }
        }
    }
    let variants = table(
        "variants at the model's shape (7 sources, 3 queries, hidden 2304, 1 row)",
        &kernels,
        &cases,
        calls,
        rounds,
    )?;

    let baseline_at = cases
        .iter()
        .position(|(_, c)| !c.wide && c.batch == 0 && c.threads == 256)
        .expect("the baseline is in the sweep");
    let baseline = variants[baseline_at].expect("the baseline runs");
    let (best_at, best) = variants
        .iter()
        .enumerate()
        .filter_map(|(at, t)| t.map(|t| (at, t)))
        .min_by(|a, b| a.1.chain_us.total_cmp(&b.1.chain_us))
        .expect("something ran");
    println!(
        "\nbaseline ({}): chain {:.1} us; best ({}): chain {:.1} us, {:+.1} us a call, {:+.2} ms over the ~120 mixes of a token",
        cases[baseline_at].0,
        baseline.chain_us,
        cases[best_at].0,
        best.chain_us,
        best.chain_us - baseline.chain_us,
        (best.chain_us - baseline.chain_us) * 120.0 / 1000.0,
    );
    println!("  the three alone, at the baseline's 256 threads, against it (chain us):");
    for (name, c) in &cases {
        if c.threads == 256 {
            let at = cases.iter().position(|(n, _)| n == name).unwrap();
            if let Some(t) = variants[at] {
                println!(
                    "    {name:<38} {:>7.1}  ({:+.1})",
                    t.chain_us,
                    t.chain_us - baseline.chain_us
                );
            }
        }
    }

    // 3. The best variant against the baseline over the shapes the model reaches.
    let base_case = Case::MODEL;
    let best_case = cases[best_at].1;
    let mut shape_cases = Vec::new();
    for (sources, queries) in [
        (4, 1),
        (7, 1),
        (7, 3),
        (9, 3),
        (4, 4),
        (7, 4),
        (9, 4),
        (16, 3),
    ] {
        for (label, c) in [("baseline", base_case), ("best", best_case)] {
            shape_cases.push((
                format!("S={sources} Q={queries} {label}"),
                Case {
                    sources,
                    queries,
                    ..c
                },
            ));
        }
    }
    table(
        "baseline against the best variant over sources x queries (attention input Q=3, MLP input Q=1, loop bank Q=4)",
        &kernels,
        &shape_cases,
        calls,
        rounds,
    )?;

    // 4. Rows of a verify block.
    let mut row_cases = Vec::new();
    for rows in [1, 2, 4, 8] {
        for (label, c) in [("baseline", base_case), ("best", best_case)] {
            row_cases.push((format!("{rows} rows {label}"), Case { rows, ..c }));
        }
    }
    table(
        "rows (a verify block; one threadgroup each)",
        &kernels,
        &row_cases,
        calls,
        rounds,
    )?;
    Ok(())
}
