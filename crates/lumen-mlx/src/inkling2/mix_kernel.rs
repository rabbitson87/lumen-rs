//! The attention-residual mix as one Metal kernel.
//!
//! [`ops::attn_res_mix`](super::ops::attn_res_mix) is the plain form: stack the depth sources, a
//! matmul against the queries, a softmax over the sources, a second matmul. Composed that is a dozen
//! kernel launches a call, and the model makes 100-odd calls a decode token; launching them costs
//! more than doing them (see `profile.rs`). This is the same arithmetic in one launch a token row.
//!
//! One threadgroup per row, every thread owning a slice of the hidden axis. Each thread dots its
//! slice of every source with its slice of every query (a tile of sources at a time), `simd_sum` and
//! a threadgroup sum join the slices into the logits, a few threads run the softmax over the sources
//! in threadgroup memory, and all threads write the weighted sum of their slice. The sums run in
//! another order than the composed ops' matmuls, so the result agrees to float32 rounding, not bit
//! for bit; the tests bound the difference, and a greedy continuation can part from the composed
//! form's at a near tie.
//!
//! A threadgroup is one group on one GPU core, so the kernel is bound by the latency of its loads, not
//! by arithmetic or bandwidth: it reads about 90 KB in tens of microseconds. What pays is more bytes in
//! flight per thread and more threads: loads are `float4` (four hidden values), and a launch has one
//! thread for each `float4` of the hidden axis, so each thread issues one load per source. At the
//! model's hidden width of 2304 that is 576 threads and a call costs 13 us, against 33 us for scalar
//! loads on 256 threads (`examples/inkling2_mix_kernel_profile.rs` measures the variants). Loading a
//! tile's sources ahead of the multiplies was slower with `float4`; it is not done.

use std::cell::OnceCell;

use anyhow::{Context, Result, anyhow, ensure};
use mlx_rs::{Array, Dtype};

use crate::metal_kernel::{MetalKernel, MetalKernelConfig};

lumen_flags::flag! {
    /// Mix the Inkling 2 depth sources (the attention residual) in one Metal kernel instead of
    /// the composed ops: a dozen launches a call become one, a decode token about 2 ms (6%) faster.
    /// The kernel sums in another order than the composed ops' matmuls, so the mix differs at
    /// float32 rounding and a greedy continuation can part from the composed form's at a near
    /// tie: a different-but-equivalent numeric regime, not a pure optimization.
    ///
    /// On by default. Teacher-forced over 2,745 positions of 20 prompts it flips the greedy choice
    /// at 0.69% of them, none by more than 2.7 bfloat16 steps, against 0.9 to 1.3% (up to 7 steps)
    /// for the verify blocks that `LUMEN_INKLING2_DRAFT` ships on. `=0` restores the composed ops.
    pub(crate) kernel {
        env: "LUMEN_INKLING2_MIX_KERNEL",
        default: true,
        kind: Behavior,
    }
}

/// Whether the mix runs as a kernel.
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

/// Most sources one launch mixes: the softmax weights live in threadgroup memory.
const MAX_SOURCES: i32 = 32;

/// Sources whose logits one pass of the dot products computes (their accumulators are registers).
const TILE: i32 = 12;

/// The simdgroup width: a launch is a whole number of simdgroups.
const SIMD: i32 = 32;

/// Fewest threads of a launch: the first `TILE * queries` threads each finish one logit, and the
/// softmax uses the first `queries`.
const MIN_THREADS: i32 = 64;

/// Most threads of a launch, the device's limit for a threadgroup.
const MAX_THREADS: i32 = 1024;

/// Threads of a launch: one for each `float4` of the hidden axis, rounded up to a simdgroup and held
/// between [`MIN_THREADS`] and [`MAX_THREADS`] (past that a thread takes several `float4`s, strided).
fn threads_for(hidden: i32) -> i32 {
    ((hidden / 4 + SIMD - 1) / SIMD * SIMD).clamp(MIN_THREADS, MAX_THREADS)
}

/// Rows of `src` are `[sources, hidden]` float32; `query` is `[Q, hidden]`, `scale` one float. The
/// output is `[rows, Q, hidden]`: for each query the softmax over the sources of the scaled dots,
/// then the weighted sum of the sources. `H`, `Q`, `THREADS`, `TILE` and `MAX_SOURCES` are template
/// arguments, the source count is read from the shape so that a model that adds a block does not
/// compile a new kernel. Every access is a `float4` (`H` is a multiple of 4), `H4` of them a row.
const SOURCE: &str = r#"
    constexpr int SIMDS = THREADS / 32;
    constexpr int H4 = H / 4;
    const int sources = src_shape[1];
    const int row = threadgroup_position_in_grid.x;
    const int tid = thread_position_in_threadgroup.x;
    const int sg = simdgroup_index_in_threadgroup;
    const int lane = thread_index_in_simdgroup;
    const device float4* base = (const device float4*)(src + (size_t)row * sources * H);
    const device float4* wide_query = (const device float4*)query;
    device float4* mixed = (device float4*)out;

    threadgroup float partial[SIMDS * TILE * Q];
    threadgroup float weight[MAX_SOURCES * Q];

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
                    const float4 v = base[(size_t)(s0 + i) * H4 + h];
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
            const float4 v = base[(size_t)s * H4 + h];
            for (int q = 0; q < Q; ++q) {
                acc[q] += weight[s * Q + q] * v;
            }
        }
        for (int q = 0; q < Q; ++q) {
            mixed[((size_t)row * Q + q) * H4 + h] = acc[q];
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
            let built = MetalKernel::new(
                "inkling2_attn_res_mix",
                &["src", "query", "scale"],
                &["out"],
                SOURCE,
                /* ensure_row_contiguous */ true,
                /* atomic_outputs */ false,
            )?;
            let _ = cell.set(built);
        }
        f(cell
            .get()
            .ok_or_else(|| anyhow!("the attention-residual mix kernel is missing"))?)
    })
}

/// [`ops::attn_res_mix`](super::ops::attn_res_mix) as one kernel launch: `blocks` and `partial_n`
/// are `[B, L, H]` float32 (the partial sum is the last source), `query` `[Q, H]`; returns
/// `[B, L, Q, H]` float32.
pub(super) fn attn_res_mix(
    blocks: &[Array],
    partial_n: &Array,
    query: &Array,
    scale: f32,
) -> Result<Array> {
    let mut sources: Vec<&Array> = blocks.iter().collect();
    sources.push(partial_n);
    let count = i32::try_from(sources.len()).context("source count")?;
    ensure!(
        count <= MAX_SOURCES,
        "the mix kernel takes at most {MAX_SOURCES} sources, got {count}"
    );
    for source in &sources {
        ensure!(
            source.dtype() == Dtype::Float32 && source.shape() == partial_n.shape(),
            "every source is a float32 {:?}, got {:?} {:?}",
            partial_n.shape(),
            source.dtype(),
            source.shape()
        );
    }
    ensure!(
        query.dtype() == Dtype::Float32,
        "the query is float32, got {:?}",
        query.dtype()
    );
    let &[b, length, hidden] = partial_n.shape() else {
        return Err(anyhow!(
            "a source is [B, L, H], got {:?}",
            partial_n.shape()
        ));
    };
    let &[queries, query_hidden] = query.shape() else {
        return Err(anyhow!("a query is [Q, H], got {:?}", query.shape()));
    };
    ensure!(
        query_hidden == hidden,
        "the query is {query_hidden} wide, the sources {hidden}"
    );
    ensure!(
        hidden % 4 == 0,
        "the mix kernel loads four hidden values at a time, got a hidden width of {hidden}"
    );
    let threads = threads_for(hidden);
    ensure!(
        queries * TILE <= threads,
        "the mix kernel takes at most {} queries at a hidden width of {hidden}, got {queries}",
        threads / TILE
    );

    #[cfg(test)]
    LAUNCHES.with(|n| n.set(n.get() + 1));
    let rows = b * length;
    let src = mlx_rs::ops::reshape(
        mlx_rs::ops::stack_axis(&sources, 2)?,
        &[rows, count, hidden],
    )?;
    let scale = Array::from_f32(scale);
    let mixed = with_kernel(|kernel| {
        let config = MetalKernelConfig::new();
        config.add_template_arg_int("H", hidden)?;
        config.add_template_arg_int("Q", queries)?;
        config.add_template_arg_int("THREADS", threads)?;
        config.add_template_arg_int("TILE", TILE)?;
        config.add_template_arg_int("MAX_SOURCES", MAX_SOURCES)?;
        config.set_grid(rows * threads, 1, 1)?;
        config.set_thread_group(threads, 1, 1)?;
        config.add_output_arg(&[rows, queries, hidden], Dtype::Float32)?;
        Ok(kernel
            .apply(&[&src, query, &scale], &config, 1)
            .context("the attention-residual mix kernel failed")?
            .swap_remove(0))
    })?;
    Ok(mlx_rs::ops::reshape(&mixed, &[b, length, queries, hidden])?)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::super::{fused, ops};
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

    /// A `[B, L, H]` source at unit RMS per row, as the model mixes them.
    fn source(seed: u64, rows: i32, hidden: i32) -> Array {
        let mut v = noise(seed, (rows * hidden) as usize);
        for row in v.chunks_mut(hidden as usize) {
            let rms = (row.iter().map(|x| x * x).sum::<f32>() / hidden as f32).sqrt();
            row.iter_mut().for_each(|x| *x /= rms);
        }
        Array::from_slice(&v, &[1, rows, hidden])
    }

    fn query(seed: u64, queries: i32, hidden: i32, gain: f32) -> Array {
        let v: Vec<f32> = noise(seed, (queries * hidden) as usize)
            .into_iter()
            .map(|x| x * gain)
            .collect();
        Array::from_slice(&v, &[queries, hidden])
    }

    fn host(a: &Array) -> Vec<f32> {
        let a = a.as_dtype(Dtype::Float32).expect("f32");
        a.eval().expect("eval");
        a.as_slice::<f32>().to_vec()
    }

    fn worst(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// The kernel gives the composed ops' mix to float32 rounding: for each source count the real
    /// model reaches, each query count its mixes use, the hidden widths of a launch with one
    /// `float4` a thread (2304: 576 threads), with fewer `float4`s than the least launch (100 and the
    /// replica's 64) and with more than the most (4100: 1025 of them over 1024 threads, one thread
    /// takes two), and queries large enough that the softmax is far from uniform.
    #[test]
    fn the_kernel_gives_the_composed_mix() {
        let mut peak = [0.0f32; 2];
        for (hidden, rows) in [(2304, 1), (2304, 3), (100, 2), (64, 2), (4100, 1)] {
            for queries in [1, 3, 4] {
                for count in [1, 2, 5, 9, 32] {
                    for (at, gain) in [1.0f32, 40.0].into_iter().enumerate() {
                        let blocks: Vec<Array> = (0..count - 1)
                            .map(|i| source(10 + i as u64, rows, hidden))
                            .collect();
                        let partial = source(99, rows, hidden);
                        let q = query(7, queries, hidden, gain);
                        let scale = (hidden as f32).powf(-0.5);
                        let want = host(&ops::attn_res_mix(&blocks, &partial, &q, scale).unwrap());
                        let got = host(&attn_res_mix(&blocks, &partial, &q, scale).unwrap());
                        let diff = worst(&want, &got);
                        peak[at] = peak[at].max(diff);
                        assert!(
                            diff < 2e-5,
                            "H={hidden} rows={rows} Q={queries} S={count} gain={gain}: {diff}"
                        );
                    }
                }
            }
        }
        eprintln!(
            "mix kernel against the composed mix, worst difference: gain 1: {:.2e}, gain 40: {:.2e}",
            peak[0], peak[1]
        );
    }

    /// One source is the whole weight: the mix is that source, exactly.
    #[test]
    fn a_single_source_is_returned_as_it_is() {
        let partial = source(5, 2, 2304);
        let q = query(3, 3, 2304, 1.0);
        let got = host(&attn_res_mix(&[], &partial, &q, 0.02).unwrap());
        let want = host(&partial);
        for query_at in 0..3 {
            let slice: Vec<f32> = (0..2)
                .flat_map(|r| got[(r * 3 + query_at) * 2304..][..2304].to_vec())
                .collect();
            assert_eq!(slice, want, "query {query_at}");
        }
    }

    /// One source far ahead of the rest takes the whole weight: the mix is that source, finite. The
    /// query is the first source scaled up, so its logit is in the thousands, where `exp` overflows
    /// float32 unless the logits are first shifted by their maximum.
    #[test]
    fn a_dominant_source_wins_without_overflow() {
        let hidden = 2304;
        let first = source(1, 1, hidden);
        let second = source(2, 1, hidden);
        let partial = source(3, 1, hidden);
        let big: Vec<f32> = host(&first).into_iter().map(|x| x * 100.0).collect();
        let q = Array::from_slice(&big, &[1, hidden]);
        let scale = (hidden as f32).powf(-0.5);
        let got = host(&attn_res_mix(&[first.clone(), second], &partial, &q, scale).unwrap());
        assert!(got.iter().all(|x| x.is_finite()), "an overflowed softmax");
        assert!(worst(&got, &host(&first)) < 1e-5, "the dominant source");
    }

    /// The kernel is told the sources apart and the partial sum from the blocks: swapping two
    /// blocks leaves the mix alone (a sum over the sources), swapping a block with the partial sum
    /// does too, but dropping a source changes it.
    #[test]
    fn the_kernel_reads_every_source() {
        let a = source(1, 1, 2304);
        let b = source(2, 1, 2304);
        let c = source(3, 1, 2304);
        let q = query(4, 3, 2304, 8.0);
        let mix = |blocks: &[Array], partial: &Array| {
            host(&attn_res_mix(blocks, partial, &q, 0.02).unwrap())
        };
        let base = mix(&[a.clone(), b.clone()], &c);
        let permuted = mix(&[b.clone(), c.clone()], &a);
        assert!(
            worst(&base, &permuted) < 2e-5,
            "a permutation of the sources"
        );
        let dropped = mix(std::slice::from_ref(&a), &c);
        assert!(worst(&base, &dropped) > 1e-3, "a source was not read");
        let other_scale = host(&attn_res_mix(&[a, b], &c, &q, 0.04).unwrap());
        assert!(worst(&base, &other_scale) > 1e-3, "the scale was not read");
    }

    #[test]
    fn what_the_kernel_cannot_take_is_an_error() {
        let partial = source(1, 1, 64);
        let q = query(2, 1, 64, 1.0);
        let many: Vec<Array> = (0..32).map(|i| source(i, 1, 64)).collect();
        assert!(
            attn_res_mix(&many, &partial, &q, 0.1).is_err(),
            "33 sources"
        );
        let wide = query(2, 1, 32, 1.0);
        assert!(
            attn_res_mix(&[], &partial, &wide, 0.1).is_err(),
            "a query of another width"
        );
        let half = partial.as_dtype(Dtype::Float16).unwrap();
        assert!(
            attn_res_mix(&[], &half, &q, 0.1).is_err(),
            "a float16 source"
        );
        let odd = source(1, 1, 66);
        assert!(
            attn_res_mix(&[], &odd, &query(2, 1, 66, 1.0), 0.1).is_err(),
            "a hidden width that is not a multiple of 4"
        );
        // 64 hidden values launch 64 threads, which finish 5 queries' logits of a tile but not 6.
        assert!(
            attn_res_mix(&[], &partial, &query(2, 5, 64, 1.0), 0.1).is_ok(),
            "5 queries"
        );
        assert!(
            attn_res_mix(&[], &partial, &query(2, 6, 64, 1.0), 0.1).is_err(),
            "6 queries"
        );
    }

    /// The launch's thread count: a `float4` a thread, a whole number of simdgroups, never fewer than
    /// two of them nor more than the device allows.
    #[test]
    fn the_launch_has_a_thread_for_each_float4_of_the_hidden_axis() {
        assert_eq!(threads_for(2304), 576);
        assert_eq!(threads_for(64), MIN_THREADS);
        assert_eq!(threads_for(100), MIN_THREADS);
        assert_eq!(threads_for(8192), MAX_THREADS);
        assert_eq!(threads_for(4100), MAX_THREADS);
        for hidden in (4..=5000).step_by(4) {
            let threads = threads_for(hidden);
            assert_eq!(threads % SIMD, 0, "hidden {hidden}");
            assert!(
                (MIN_THREADS..=MAX_THREADS).contains(&threads),
                "hidden {hidden}"
            );
        }
    }

    /// Timing only (prints; asserts nothing about speed): a call of the composed ops, of the
    /// compiled form and of the kernel, each chained on the last so that the GPU cannot overlap
    /// them, at the shape of the real model: six blocks and the partial sum, three queries.
    #[test]
    fn what_a_mix_costs() {
        const CALLS: usize = 300;
        let (hidden, queries) = (2304, 3);
        let blocks: Vec<Array> = (0..6).map(|i| source(20 + i, 1, hidden)).collect();
        let q = query(9, queries, hidden, 1.0);
        let scale = (hidden as f32).powf(-0.5);
        type Mix = fn(&[Array], &Array, &Array, f32) -> Result<Array>;
        let forms: [(&str, Mix); 3] = [
            ("composed", ops::attn_res_mix),
            ("compiled", fused::attn_res_mix),
            ("kernel", attn_res_mix),
        ];
        for (name, run) in forms {
            let chain = |calls: usize| -> Result<(f64, Array)> {
                let mut partial = source(98, 1, hidden);
                let start = Instant::now();
                for _ in 0..calls {
                    let mixed = run(&blocks, &partial, &q, scale)?;
                    partial = mlx_rs::ops::indexing::TryIndexOp::try_index(&mixed, (.., .., 0))?;
                }
                partial.eval()?;
                Ok((start.elapsed().as_secs_f64(), partial))
            };
            chain(20).unwrap(); // build the kernel and the compiled graph
            let (seconds, last) = chain(CALLS).unwrap();
            assert!(host(&last).iter().all(|x| x.is_finite()));
            eprintln!(
                "mix timing: {name:>8}: {:.1} us a call over {CALLS} chained calls",
                seconds * 1e6 / CALLS as f64
            );
        }
    }
}
