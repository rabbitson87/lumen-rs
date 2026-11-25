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
//!
//! The sources come in two buffers, the completed blocks stacked `[rows, blocks, hidden]` and the
//! partial sum (the last source) on its own, so that the model stacks the blocks once, when a block
//! is added, and not once a mix: a stack is a copy launch a source. The kernel can also take the
//! partial sum as the layer left it, bfloat16 or float32 and not yet normalised, and scale it to
//! unit RMS itself, adding the sum of squares in the order `rms_norm` does so that the scale is the
//! same to the bit: that is the cast and the `rms_norm` launch a mix saves ([`attn_res_mix_raw`],
//! behind [`fold`]; `examples/inkling2_mix_fold_profile.rs` prices it).

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

lumen_flags::flag! {
    /// Let the mix kernel read the depth blocks from one stacked array (built when a block is
    /// added, not once a mix) and normalise the partial sum itself, in place of a cast, an
    /// `rms_norm` and a stack of copies ahead of each of the 96 mixes of a decode token. Needs
    /// `LUMEN_INKLING2_MIX_KERNEL` (and is off with it).
    ///
    /// Bit for bit what the mix kernel gives with those ops ahead of it: the kernel adds the sum of
    /// squares in the order `rms_norm` does, and the tests compare the scaled sum and the mix to
    /// the bit. A decode token about 0.7 ms (2.4%) faster (20 same-session runs against the
    /// version without it, standard error 0.13 ms).
    pub(crate) fold {
        env: "LUMEN_INKLING2_MIX_FOLD",
        default: true,
        kind: Optimization,
    }
}

/// Whether the mix runs as a kernel.
pub(super) fn enabled() -> bool {
    kernel::get()
}

/// Whether the mix kernel also stacks the blocks once and normalises the partial sum.
pub(super) fn folds() -> bool {
    kernel::get() && fold::get()
}

#[cfg(test)]
thread_local! {
    static LAUNCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static NORMALISING: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// How many times the kernel has been launched on this thread (a test's check that it ran).
#[cfg(test)]
pub(super) fn launches() -> usize {
    LAUNCHES.with(std::cell::Cell::get)
}

/// How many of those launches normalised the partial sum themselves (the fold's).
#[cfg(test)]
pub(super) fn folded_launches() -> usize {
    NORMALISING.with(std::cell::Cell::get)
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

/// Four values of a row as a `float4`: one vector load from float32, four converted loads from any
/// other type (the bfloat16 partial sum the kernel reads as the layer left it).
const HEADER: &str = r#"
inline float4 load4(const device float* p, int i) {
    return ((const device float4*)p)[i];
}
template <typename T>
inline float4 load4(const device T* p, int i) {
    return float4(float(p[4 * i]), float(p[4 * i + 1]), float(p[4 * i + 2]), float(p[4 * i + 3]));
}
"#;

/// `blocks` is `[rows, B, hidden]` float32, the completed blocks (not read when `HAS_BLOCKS` is 0),
/// `last` the partial sum `[rows, hidden]`, the last source, float32 or bfloat16; `query` is
/// `[Q, hidden]`, `scale` one float. The output is `[rows, Q, hidden]`: for each query the softmax
/// over the `B + 1` sources of the scaled dots, then the weighted sum of the sources. With `NORM` the
/// partial sum is first scaled to unit RMS (`rsqrt(mean(x^2) + 1e-6)`, as `rms_norm` with ones does),
/// and with `EMIT` the scaled sum is also written to `normed`, `[rows, hidden]` (the model keeps it
/// as a block at a block boundary). `H`, `Q`, `THREADS`, `TILE`, `MAX_SOURCES`, `NORM`, `EMIT` and
/// `HAS_BLOCKS` are template arguments, the block count is read from the shape so that a model that
/// adds a block does not compile a new kernel. Every access to the blocks is a `float4` (`H` is a
/// multiple of 4), `H4` of them a row.
const SOURCE: &str = r#"
    constexpr int SIMDS = THREADS / 32;
    constexpr int H4 = H / 4;
    const int nb = HAS_BLOCKS ? blocks_shape[1] : 0;
    const int sources = nb + 1;
    const int row = threadgroup_position_in_grid.x;
    const int tid = thread_position_in_threadgroup.x;
    const int sg = simdgroup_index_in_threadgroup;
    const int lane = thread_index_in_simdgroup;
    const device float4* base = (const device float4*)(blocks + (size_t)row * nb * H);
    const device auto* tail = last + (size_t)row * H;
    const device float4* wide_query = (const device float4*)query;
    device float4* mixed = (device float4*)out;
    device float4* emitted = (device float4*)normed;

    threadgroup float partial[SIMDS * TILE * Q];
    threadgroup float weight[MAX_SOURCES * Q];
    threadgroup float squares[SIMDS];

    float tail_scale = 1.0f;
    if (NORM) {
        // The sum of squares in the order MLX's `rms_norm` adds them (a thread's four values one
        // after the other, `simd_sum`, then `simd_sum` over the simdgroups' totals, which every
        // simdgroup does alike), so that the scale is the one `rms_norm` finds, to the bit.
        float sum_squares = 0.0f;
        for (int h = tid; h < H4; h += THREADS) {
            const float4 v = load4(tail, h);
            sum_squares += v.x * v.x;
            sum_squares += v.y * v.y;
            sum_squares += v.z * v.z;
            sum_squares += v.w * v.w;
        }
        sum_squares = simd_sum(sum_squares);
        if (lane == 0) {
            squares[sg] = sum_squares;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float total = simd_sum(lane < SIMDS ? squares[lane] : 0.0f);
        // The width is read from the shape, as `rms_norm` is handed it: a compile-time constant
        // would let the compiler divide by multiplying with a reciprocal.
        tail_scale = metal::precise::rsqrt(total / float(last_shape[1]) + 1e-6f);
    }
    if (EMIT) {
        for (int h = tid; h < H4; h += THREADS) {
            emitted[(size_t)row * H4 + h] = load4(tail, h) * tail_scale;
        }
    }

#define SOURCE_AT(s, h) \
    ((s) < nb ? base[(size_t)(s) * H4 + (h)] : load4(tail, (h)) * tail_scale)

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
                "inkling2_attn_res_mix",
                &["blocks", "last", "query", "scale"],
                &["out", "normed"],
                HEADER,
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

/// The completed blocks (each `[B, L, H]`) stacked for the kernel: `[B * L, blocks, H]`, `None` when
/// there are none. The stack is a copy launch a block, which is why the model keeps the result
/// rather than stacking once a mix.
pub(super) fn stack_blocks(blocks: &[Array]) -> Result<Option<Array>> {
    let Some(first) = blocks.first() else {
        return Ok(None);
    };
    let &[b, length, hidden] = first.shape() else {
        return Err(anyhow!("a block is [B, L, H], got {:?}", first.shape()));
    };
    let sources: Vec<&Array> = blocks.iter().collect();
    let count = i32::try_from(sources.len()).context("block count")?;
    Ok(Some(mlx_rs::ops::reshape(
        mlx_rs::ops::stack_axis(&sources, 2)?,
        &[b * length, count, hidden],
    )?))
}

/// What a launch computes besides the mix.
#[derive(Clone, Copy)]
struct Fold {
    /// Scale the partial sum to unit RMS first.
    norm: bool,
    /// Also return the scaled partial sum.
    emit: bool,
}

/// One launch. `stacked` is `[B * L, blocks, H]` float32 (`None`: the partial sum is the only
/// source), `partial` the last source `[B, L, H]`, float32 or bfloat16, `query` `[Q, H]` float32.
/// Returns the mix `[B, L, Q, H]` and, with `fold.emit`, the scaled partial sum `[B, L, H]`.
fn launch(
    stacked: Option<&Array>,
    partial: &Array,
    query: &Array,
    scale: f32,
    fold: Fold,
) -> Result<(Array, Option<Array>)> {
    ensure!(
        matches!(partial.dtype(), Dtype::Float32 | Dtype::Bfloat16),
        "the partial sum is float32 or bfloat16, got {:?}",
        partial.dtype()
    );
    ensure!(
        query.dtype() == Dtype::Float32,
        "the query is float32, got {:?}",
        query.dtype()
    );
    let &[b, length, hidden] = partial.shape() else {
        return Err(anyhow!("a source is [B, L, H], got {:?}", partial.shape()));
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
    let rows = b * length;
    let blocks = match stacked {
        Some(s) => {
            ensure!(
                s.dtype() == Dtype::Float32,
                "the blocks are float32, got {:?}",
                s.dtype()
            );
            let &[stacked_rows, count, stacked_hidden] = s.shape() else {
                return Err(anyhow!(
                    "the stacked blocks are [rows, blocks, H], got {:?}",
                    s.shape()
                ));
            };
            ensure!(
                stacked_rows == rows && stacked_hidden == hidden,
                "the blocks are {:?}, the partial sum {:?}",
                s.shape(),
                partial.shape()
            );
            ensure!(
                count < MAX_SOURCES,
                "the mix kernel takes at most {MAX_SOURCES} sources, got {}",
                count + 1
            );
            count
        }
        None => 0,
    };
    let threads = threads_for(hidden);
    ensure!(
        queries * TILE <= threads,
        "the mix kernel takes at most {} queries at a hidden width of {hidden}, got {queries}",
        threads / TILE
    );

    #[cfg(test)]
    {
        LAUNCHES.with(|n| n.set(n.get() + 1));
        if fold.norm {
            NORMALISING.with(|n| n.set(n.get() + 1));
        }
    }
    let partial_rows = mlx_rs::ops::reshape(partial, &[rows, hidden])?;
    let scale = Array::from_f32(scale);
    let mut outs = with_kernel(|kernel| {
        let config = MetalKernelConfig::new();
        config.add_template_arg_int("H", hidden)?;
        config.add_template_arg_int("Q", queries)?;
        config.add_template_arg_int("THREADS", threads)?;
        config.add_template_arg_int("TILE", TILE)?;
        config.add_template_arg_int("MAX_SOURCES", MAX_SOURCES)?;
        config.add_template_arg_int("NORM", i32::from(fold.norm))?;
        config.add_template_arg_int("EMIT", i32::from(fold.emit))?;
        config.add_template_arg_int("HAS_BLOCKS", i32::from(blocks > 0))?;
        config.set_grid(rows * threads, 1, 1)?;
        config.set_thread_group(threads, 1, 1)?;
        config.add_output_arg(&[rows, queries, hidden], Dtype::Float32)?;
        let normed_shape: &[i32] = if fold.emit { &[rows, hidden] } else { &[1, 1] };
        config.add_output_arg(normed_shape, Dtype::Float32)?;
        // With no blocks the kernel never reads the array, but it must be a buffer: the partial
        // sum stands in (an array of under 8 values would be a `constant`).
        let blocks_in = stacked.unwrap_or(&partial_rows);
        kernel
            .apply(&[blocks_in, &partial_rows, query, &scale], &config, 2)
            .context("the attention-residual mix kernel failed")
    })?;
    let normed = outs.swap_remove(1);
    let mixed = outs.swap_remove(0);
    Ok((
        mlx_rs::ops::reshape(&mixed, &[b, length, queries, hidden])?,
        if fold.emit {
            Some(mlx_rs::ops::reshape(&normed, &[b, length, hidden])?)
        } else {
            None
        },
    ))
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
    let count = i32::try_from(blocks.len() + 1).context("source count")?;
    ensure!(
        count <= MAX_SOURCES,
        "the mix kernel takes at most {MAX_SOURCES} sources, got {count}"
    );
    for source in blocks.iter().chain([partial_n]) {
        ensure!(
            source.dtype() == Dtype::Float32 && source.shape() == partial_n.shape(),
            "every source is a float32 {:?}, got {:?} {:?}",
            partial_n.shape(),
            source.dtype(),
            source.shape()
        );
    }
    let stacked = stack_blocks(blocks)?;
    let fold = Fold {
        norm: false,
        emit: false,
    };
    Ok(launch(stacked.as_ref(), partial_n, query, scale, fold)?.0)
}

/// The mix of `stacked` (the completed blocks, from [`stack_blocks`]) and a partial sum the layer
/// left as it was: `partial` is `[B, L, H]` float32 or bfloat16, scaled to unit RMS inside the
/// kernel. Returns the mix `[B, L, Q, H]` and, when `emit`, the scaled partial sum `[B, L, H]`
/// float32 (the block a boundary layer adds).
pub(super) fn attn_res_mix_raw(
    stacked: Option<&Array>,
    partial: &Array,
    query: &Array,
    scale: f32,
    emit: bool,
) -> Result<(Array, Option<Array>)> {
    let fold = Fold { norm: true, emit };
    launch(stacked, partial, query, scale, fold)
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

    /// A partial sum as a layer leaves it, `[1, rows, hidden]`: far from unit RMS (about 3, offset),
    /// float32 or bfloat16.
    fn raw_partial(seed: u64, rows: i32, hidden: i32, dtype: Dtype) -> Array {
        let v: Vec<f32> = noise(seed, (rows * hidden) as usize)
            .into_iter()
            .enumerate()
            .map(|(i, x)| x * 3.0 + 0.5 * ((i % 7) as f32 - 3.0))
            .collect();
        Array::from_slice(&v, &[1, rows, hidden])
            .as_dtype(dtype)
            .expect("cast")
    }

    /// The raw mix scales the partial sum itself and gives the composed mix of the scaled sum, and
    /// returns that scaled sum: for each block count (none, one, a few, the most), query count and
    /// hidden width the plain test uses, float32 and bfloat16 sums, one row and several (a row is a
    /// threadgroup, so several rows exercise the row offsets of the blocks and of the sum).
    #[test]
    fn the_raw_mix_gives_the_composed_mix_of_the_scaled_sum() {
        let (mut worst_mix, mut worst_sum) = (0.0f32, 0.0f32);
        for (hidden, rows) in [(2304, 1), (2304, 3), (100, 2), (64, 2), (4100, 1)] {
            for queries in [1, 3, 4] {
                for count in [0, 1, 4, 31] {
                    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
                        let blocks: Vec<Array> = (0..count)
                            .map(|i| source(10 + i as u64, rows, hidden))
                            .collect();
                        let raw = raw_partial(99, rows, hidden, dtype);
                        let q = query(7, queries, hidden, 3.0);
                        let scale = (hidden as f32).powf(-0.5);
                        let scaled = ops::unit_rms(&raw.as_dtype(Dtype::Float32).unwrap()).unwrap();
                        let want = host(&ops::attn_res_mix(&blocks, &scaled, &q, scale).unwrap());
                        let stacked = stack_blocks(&blocks).unwrap();
                        let (mixed, emitted) =
                            attn_res_mix_raw(stacked.as_ref(), &raw, &q, scale, true).unwrap();
                        let got = host(&mixed);
                        let diff = worst(&want, &got);
                        let sum_diff = worst(
                            &host(&scaled),
                            &host(&emitted.expect("the scaled sum was asked for")),
                        );
                        worst_mix = worst_mix.max(diff);
                        worst_sum = worst_sum.max(sum_diff);
                        let label =
                            format!("H={hidden} rows={rows} Q={queries} blocks={count} {dtype:?}");
                        assert!(diff < 2e-5, "{label}: the mix differs by {diff}");
                        assert!(
                            sum_diff < 5e-6,
                            "{label}: the scaled sum differs by {sum_diff}"
                        );
                    }
                }
            }
        }
        eprintln!(
            "raw mix kernel against the composed mix of the scaled sum: mix {worst_mix:.2e}, scaled sum {worst_sum:.2e}"
        );
    }

    /// With no blocks the partial sum is the whole weight: each query's mix is the scaled sum.
    #[test]
    fn a_raw_partial_sum_alone_is_its_own_mix() {
        let raw = raw_partial(5, 2, 2304, Dtype::Float32);
        let q = query(3, 3, 2304, 1.0);
        let (mixed, _) = attn_res_mix_raw(None, &raw, &q, 0.02, false).unwrap();
        let want = host(&ops::unit_rms(&raw).unwrap());
        let got = host(&mixed);
        for query_at in 0..3 {
            let slice: Vec<f32> = (0..2)
                .flat_map(|r| got[(r * 3 + query_at) * 2304..][..2304].to_vec())
                .collect();
            assert!(worst(&slice, &want) < 5e-6, "query {query_at}");
        }
    }

    /// The scaled sum comes back only when asked for, and the mix is the same either way.
    #[test]
    fn the_scaled_sum_is_returned_only_when_asked() {
        let blocks: Vec<Array> = (0..3).map(|i| source(i, 2, 2304)).collect();
        let stacked = stack_blocks(&blocks).unwrap();
        let raw = raw_partial(9, 2, 2304, Dtype::Bfloat16);
        let q = query(4, 3, 2304, 2.0);
        let (without, none) = attn_res_mix_raw(stacked.as_ref(), &raw, &q, 0.02, false).unwrap();
        assert!(none.is_none());
        let (with, some) = attn_res_mix_raw(stacked.as_ref(), &raw, &q, 0.02, true).unwrap();
        assert_eq!(some.expect("asked for").shape(), &[1, 2, 2304]);
        assert_eq!(host(&without), host(&with));
    }

    /// The stack is the blocks side by side per row, in order, and nothing when there are none.
    #[test]
    fn the_blocks_are_stacked_per_row_in_order() {
        assert!(stack_blocks(&[]).unwrap().is_none());
        let blocks: Vec<Array> = (0..3).map(|i| source(i, 2, 8)).collect();
        let stacked = stack_blocks(&blocks).unwrap().unwrap();
        assert_eq!(stacked.shape(), &[2, 3, 8]);
        let got = host(&stacked);
        for (b, block) in blocks.iter().enumerate() {
            let want = host(block);
            for row in 0..2 {
                assert_eq!(
                    &got[(row * 3 + b) * 8..][..8],
                    &want[row * 8..][..8],
                    "block {b} row {row}"
                );
            }
        }
    }

    /// The raw mix refuses what it cannot take: a sum that is not float32 or bfloat16, blocks of
    /// another row count or width, blocks past the source limit, a width that is not a multiple of 4.
    #[test]
    fn what_the_raw_mix_cannot_take_is_an_error() {
        let raw = raw_partial(1, 1, 64, Dtype::Float32);
        let q = query(2, 1, 64, 1.0);
        let stack = |n: i32, rows: i32, hidden: i32| {
            let blocks: Vec<Array> = (0..n).map(|i| source(i as u64, rows, hidden)).collect();
            stack_blocks(&blocks).unwrap()
        };
        assert!(
            attn_res_mix_raw(stack(2, 1, 64).as_ref(), &raw, &q, 0.1, false).is_ok(),
            "a plain call"
        );
        let half = raw.as_dtype(Dtype::Float16).unwrap();
        assert!(
            attn_res_mix_raw(None, &half, &q, 0.1, false).is_err(),
            "a float16 sum"
        );
        assert!(
            attn_res_mix_raw(stack(2, 2, 64).as_ref(), &raw, &q, 0.1, false).is_err(),
            "blocks of two rows against a sum of one"
        );
        assert!(
            attn_res_mix_raw(stack(2, 1, 32).as_ref(), &raw, &q, 0.1, false).is_err(),
            "blocks of another width"
        );
        assert!(
            attn_res_mix_raw(stack(31, 1, 64).as_ref(), &raw, &q, 0.1, false).is_ok(),
            "32 sources"
        );
        assert!(
            attn_res_mix_raw(stack(32, 1, 64).as_ref(), &raw, &q, 0.1, false).is_err(),
            "33 sources"
        );
        let odd = raw_partial(1, 1, 66, Dtype::Float32);
        assert!(
            attn_res_mix_raw(None, &odd, &query(2, 1, 66, 1.0), 0.1, false).is_err(),
            "a hidden width that is not a multiple of 4"
        );
    }

    /// The raw mix is the composed norm and the kernel mix to the bit, not to a tolerance: the scaled
    /// sum is `rms_norm`'s output and the mix is what the kernel gives that scaled sum. The widths
    /// cover one thread a `float4` (2304: 576 threads), fewer than a simdgroup (64, 100), the
    /// widest single launch (1024, 4096) and a thread taking two (4100), and the sum arrives as
    /// float32 or bfloat16 (a bfloat16 sum is cast, exactly, before `rms_norm` in the plain form).
    #[test]
    fn the_raw_mix_is_the_norm_and_the_kernel_mix_to_the_bit() {
        for (hidden, rows) in [
            (2304, 64),
            (2304, 1),
            (64, 16),
            (100, 8),
            (1024, 8),
            (4096, 4),
            (4100, 4),
        ] {
            for dtype in [Dtype::Float32, Dtype::Bfloat16] {
                for seed in 0..4u64 {
                    let blocks: Vec<Array> = (0..4).map(|i| source(10 + i, rows, hidden)).collect();
                    let raw = raw_partial(100 + seed, rows, hidden, dtype);
                    let q = query(7, 3, hidden, 3.0);
                    let scale = (hidden as f32).powf(-0.5);
                    let scaled = ops::unit_rms(&raw.as_dtype(Dtype::Float32).unwrap()).unwrap();
                    let want_mix = host(&attn_res_mix(&blocks, &scaled, &q, scale).unwrap());
                    let stacked = stack_blocks(&blocks).unwrap();
                    let (mixed, emitted) =
                        attn_res_mix_raw(stacked.as_ref(), &raw, &q, scale, true).unwrap();
                    let label = format!("H={hidden} rows={rows} {dtype:?} seed {seed}");
                    let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<u32>>();
                    assert_eq!(
                        bits(host(&scaled)),
                        bits(host(&emitted.unwrap())),
                        "{label}: the scaled sum is not rms_norm's"
                    );
                    assert_eq!(
                        bits(want_mix),
                        bits(host(&mixed)),
                        "{label}: the mix is not the kernel's mix of the scaled sum"
                    );
                }
            }
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
