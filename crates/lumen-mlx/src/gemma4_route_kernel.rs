//! The Gemma 4 router's top-k and routing weights as one Metal kernel.
//!
//! After the router's projection, a decode step picks each token's experts with an
//! `argpartition` and a slice, then weighs them with a compiled tail: the chosen scores gathered,
//! their softmax, each chosen expert's `per_expert_scale` gathered and multiplied in. Five Metal
//! dispatches a layer. [`route`] is the same arithmetic in one launch a token row.
//!
//! One threadgroup a row, a thread for each expert. The picked experts are MLX's: its GPU
//! `argpartition` is a full merge sort, stable (an odd-even transposition sort that swaps only on
//! strictly less, and a merge that takes from the right only when strictly less), ascending, NaN
//! last (`LessThan`), so the top `k` are the last `k` of that order, lowest score first and ties
//! by index. Each thread ranks its expert by counting the experts that sort before it and writes
//! it at its rank among the last `k`. The softmax is MLX's `softmax_single_row` (precise, float
//! accumulation) as it runs on `k` values: a simdgroup whose lanes each hold four values (the
//! lanes past `k` hold negative infinity), the maximum, `fast::exp(x - max)`, the sum, `1 / sum`,
//! each product rounded to bfloat16, then multiplied by the expert's scale in bfloat16. The
//! indices and weights are therefore MLX's bit for bit, in its order, which the tests compare;
//! that is why the flag is an `Optimization`.

use std::cell::OnceCell;

use anyhow::{Context, Result, anyhow, ensure};
use mlx_rs::{Array, Dtype};

use crate::metal_kernel::{MetalKernel, MetalKernelConfig};

lumen_flags::flag! {
    /// Pick each token's experts and weigh them in one Metal kernel (the top-k, the chosen
    /// scores' softmax and the per-expert scale): five dispatches a layer become one. Bit for bit
    /// what `argpartition` and the routing tail give, in the same order. Runs inside the
    /// decoder layers' kernel path, so it takes effect with `LUMEN_GEMMA4_NORM_KERNELS`;
    /// decode-sized passes only.
    ///
    /// Off: 120 fewer dispatches a token and the same continuation in 80 of 80 runs on
    /// gemma-4-26b-a4b NVFP4, but no measurable speed-up. Over the undisturbed blocks of the ABBA
    /// runs it is about 0.04 ms a token faster, where the q/k/v kernel, removing as many
    /// dispatches, is 0.25 ms. A likely reason, not yet tested: MLX encodes concurrently and the
    /// experts' `gather_qmm` needs only the indices, so the composed tail overlapped the experts,
    /// while the kernel makes them wait for its weights too. Its expert path for affine 4-bit
    /// checkpoints (the experts compile
    /// slot with the combine outside it, instead of the routing-experts slot) has been checked by
    /// the model-free tests only.
    pub(crate) route_kernel {
        env: "LUMEN_GEMMA4_ROUTE_KERNEL",
        default: false,
        kind: Optimization,
    }
}

/// Whether the router's tail runs as a kernel.
pub(crate) fn enabled() -> bool {
    route_kernel::get()
}

/// The simdgroup width.
const SIMD: i32 = 32;

/// The most experts a launch ranks: a thread for each, one threadgroup.
const MAX_EXPERTS: i32 = 1024;

/// The most experts a token picks: the softmax is one simdgroup of four values a lane.
const MAX_K: i32 = 4 * SIMD;

/// The most rows a launch serves (decode steps and verify blocks).
const MAX_ROWS: i32 = 16;

#[cfg(test)]
thread_local! {
    static LAUNCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// How many times the kernel has been launched on this thread (a test's check that it ran).
#[cfg(test)]
pub(crate) fn launches() -> usize {
    LAUNCHES.with(std::cell::Cell::get)
}

/// MLX's `LessThan` on floats: NaN sorts after every number, and two NaNs are equal.
const HEADER: &str = r#"
inline bool sorts_before(float a, float b) {
    const bool an = metal::isnan(a);
    const bool bn = metal::isnan(b);
    if (an | bn) {
        return (!an) & bn;
    }
    return a < b;
}
"#;

/// `scores` is `[rows, E]` bfloat16, `scale` `[E]` bfloat16. Writes `inds` `[rows, K]` uint32 (the
/// last `K` of the row's stable ascending sort) and `weights` `[rows, K]` bfloat16 (their softmax,
/// times each one's scale).
const SOURCE: &str = r#"
    const uint row = threadgroup_position_in_grid.x;
    const int tid = int(thread_position_in_threadgroup.x);
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;

    threadgroup float key[E];
    threadgroup float picked_score[K];
    threadgroup uint picked[K];

    if (tid < E) {
        key[tid] = float(scores[size_t(row) * E + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Where the stable ascending sort puts this expert: after every expert that sorts before it,
    // and after the equal ones of lower index.
    if (tid < E) {
        const float mine = key[tid];
        int rank = 0;
        for (int j = 0; j < E; ++j) {
            const float other = key[j];
            const bool before = sorts_before(other, mine) || (!sorts_before(mine, other) && j < tid);
            rank += before ? 1 : 0;
        }
        if (rank >= E - K) {
            picked[rank - (E - K)] = uint(tid);
            picked_score[rank - (E - K)] = mine;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // `softmax_single_row` over the `K` picked scores, as MLX launches it for an axis of `K`: one
    // simdgroup, four values a lane, the lanes past `K` reading negative infinity.
    if (sg == 0) {
        constexpr int R = 4;
        const int base = int(lane) * R;
        float ld[R];
        if (base + R <= K) {
            for (int i = 0; i < R; i++) {
                ld[i] = picked_score[base + i];
            }
        } else {
            for (int i = 0; i < R; i++) {
                ld[i] = (base + i) < K ? picked_score[base + i] : -INFINITY;
            }
        }
        float maxval = -FLT_MAX;
        for (int i = 0; i < R; i++) {
            maxval = (maxval < ld[i]) ? ld[i] : maxval;
        }
        maxval = simd_max(maxval);
        // The maximum over the simdgroups' maxima, of which there is one.
        maxval = simd_max(lane == 0 ? maxval : -INFINITY);

        float normalizer = 0;
        for (int i = 0; i < R; i++) {
            float exp_x = metal::fast::exp(ld[i] - maxval);
            ld[i] = exp_x;
            normalizer += exp_x;
        }
        normalizer = simd_sum(normalizer);
        // The sum over the simdgroups' sums, of which there is one.
        normalizer = simd_sum(lane == 0 ? normalizer : 0.0f);
        normalizer = 1 / normalizer;

        for (int i = 0; i < R; i++) {
            const int r = base + i;
            if (r < K) {
                // Every slot of `picked` is written only while the ranks are a permutation, which
                // the comparator's total order gives. Clamped, so that an edit to the comparator
                // can at worst pick a wrong expert: `gather_qmm` does not check the ids it is
                // handed, and an unwritten slot would send it past the expert weights.
                const uint e = min(picked[r], uint(E - 1));
                const bfloat16_t soft = static_cast<bfloat16_t>(ld[i] * normalizer);
                weights[size_t(row) * K + r] = soft * scale[e];
                inds[size_t(row) * K + r] = e;
            }
        }
    }
"#;

thread_local! {
    /// MLX keeps one stream per thread and the engine decodes on one thread, so the kernel is
    /// built once per thread rather than per call.
    static KERNEL: OnceCell<MetalKernel> = const { OnceCell::new() };
}

/// The router's top-k and weights of `scores` (`[1, L, E]` bfloat16, the projection's output) with
/// `per_expert_scale` (`[E]` bfloat16), as one launch. Returns `[indices, weights]`, `[1, L, k]`
/// each (uint32 and bfloat16): what `argpartition(scores, -k)[..., E - k:]` and the routing tail
/// give, in the same order.
pub(crate) fn route(scores: &Array, per_expert_scale: &Array, k: i32) -> Result<[Array; 2]> {
    let &[1, rows, experts] = scores.shape() else {
        return Err(anyhow!(
            "the router's scores are [1, L, E], got {:?}",
            scores.shape()
        ));
    };
    ensure!(
        scores.dtype() == Dtype::Bfloat16 && per_expert_scale.dtype() == Dtype::Bfloat16,
        "the route kernel takes bfloat16 scores and scales, got {:?} and {:?}",
        scores.dtype(),
        per_expert_scale.dtype()
    );
    ensure!(
        per_expert_scale.shape() == [experts],
        "the per-expert scale is {:?} for {experts} experts",
        per_expert_scale.shape()
    );
    ensure!(
        (1..=MAX_ROWS).contains(&rows)
            && (1..=MAX_EXPERTS).contains(&experts)
            && (1..=MAX_K.min(experts)).contains(&k),
        "the route kernel takes 1 to {MAX_ROWS} rows, 1 to {MAX_EXPERTS} experts and a top-k of \
         1 to {MAX_K}, got {rows} rows, {experts} experts, top-{k}"
    );
    #[cfg(test)]
    LAUNCHES.with(|n| n.set(n.get() + 1));
    let threads = ((experts + SIMD - 1) / SIMD * SIMD).max(SIMD);
    let scores2 = mlx_rs::ops::reshape(scores, &[rows, experts])?;
    let mut outs = crate::metal_kernel::with_thread_kernel(
        &KERNEL,
        || {
            // The name is a literal: `cargo xtask test` reads the names off `MetalKernel::new`.
            MetalKernel::new_with_header(
                "gemma4_route_topk",
                &["scores", "scale"],
                &["inds", "weights"],
                HEADER,
                SOURCE,
                /* ensure_row_contiguous */ true,
                /* atomic_outputs */ false,
            )
        },
        |kernel| {
            let config = MetalKernelConfig::new();
            config.add_template_arg_int("E", experts)?;
            config.add_template_arg_int("K", k)?;
            config.set_grid(threads * rows, 1, 1)?;
            config.set_thread_group(threads, 1, 1)?;
            config.add_output_arg(&[rows, k], Dtype::Uint32)?;
            config.add_output_arg(&[rows, k], Dtype::Bfloat16)?;
            kernel
                .apply(&[&scores2, per_expert_scale], &config, 2)
                .context("the Gemma 4 route kernel failed")
        },
    )?;
    let weights = mlx_rs::ops::reshape(outs.swap_remove(1), &[1, rows, k])?;
    let indices = mlx_rs::ops::reshape(outs.swap_remove(0), &[1, rows, k])?;
    Ok([indices, weights])
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::indexing::IndexOp;

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

    fn bf16(v: &[f32], shape: &[i32]) -> Array {
        Array::from_slice(v, shape)
            .as_dtype(Dtype::Bfloat16)
            .expect("cast")
    }

    /// The composed ops `router_forward` runs after the projection: `argpartition`, the slice, and
    /// the routing tail (take, precise softmax, take, multiply).
    fn composed(scores: &Array, scale: &Array, k: i32) -> [Array; 2] {
        let experts = scores.shape()[2];
        let partitioned = mlx_rs::ops::argpartition_axis(scores, -k, 2).unwrap();
        let indices = partitioned.index((mlx_rs::ops::indexing::Ellipsis, experts - k..));
        let top = scores.take_along_axis(&indices, 2).unwrap();
        let soft = mlx_rs::ops::softmax_axis(&top, 2, Some(true)).unwrap();
        let per_expert = mlx_rs::ops::indexing::take_axis(scale, &indices, 0).unwrap();
        let weights = mlx_rs::ops::multiply(&soft, &per_expert).unwrap();
        [indices, weights]
    }

    /// The values of `a` in row-major order. The cast makes a fresh array: the slice of a top-1
    /// column reshapes to a strided view, whose buffer `as_slice` would read from its start.
    fn u32s(a: &Array) -> Vec<u32> {
        let flat = mlx_rs::ops::reshape(a, &[-1])
            .unwrap()
            .as_dtype(Dtype::Int64)
            .unwrap();
        flat.eval().unwrap();
        flat.as_slice::<i64>().iter().map(|&i| i as u32).collect()
    }

    fn bf16_bits(a: &Array) -> Vec<u16> {
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

    fn assert_same(what: &str, scores: &Array, scale: &Array, k: i32) {
        let [got_i, got_w] = route(scores, scale, k).expect("route");
        let [want_i, want_w] = composed(scores, scale, k);
        assert_eq!(got_i.shape(), want_i.shape(), "{what}: index shape");
        assert_eq!(got_w.dtype(), want_w.dtype(), "{what}: weight dtype");
        assert_eq!(u32s(&got_i), u32s(&want_i), "{what}: indices");
        assert_eq!(bf16_bits(&got_w), bf16_bits(&want_w), "{what}: weights");
    }

    /// The kernel picks MLX's experts in MLX's order and weighs them to the bit: Gemma 4's
    /// 128 experts top-8 and other sizes (a partly empty simdgroup, a top-k that is not a multiple
    /// of four, a single expert), decode and verify row counts, and score spreads from near-ties to
    /// a winner far ahead.
    #[test]
    fn the_kernel_is_argpartition_and_the_routing_tail_to_the_bit() {
        for (experts, k) in [
            (128, 8),
            (128, 1),
            (64, 6),
            (100, 7),
            (8, 8),
            (256, 16),
            (1000, 4),
        ] {
            for rows in [1, 3, 16] {
                for gain in [8.0f32, 0.05, 300.0] {
                    let seed = (experts * 7919 + k * 31 + rows) as u64;
                    let v: Vec<f32> = noise(seed, (rows * experts) as usize)
                        .into_iter()
                        .map(|x| x * gain)
                        .collect();
                    let scores = bf16(&v, &[1, rows, experts]);
                    let s: Vec<f32> = noise(seed + 1, experts as usize)
                        .into_iter()
                        .map(|x| 1.0 + 0.5 * x)
                        .collect();
                    let scale = bf16(&s, &[experts]);
                    assert_same(
                        &format!("{rows} rows of {experts} experts, top-{k}, gain {gain}"),
                        &scores,
                        &scale,
                        k,
                    );
                }
            }
        }
    }

    /// Ties are where an order that is not MLX's would show: rows of few distinct values, ties at
    /// the boundary of the top-k, a row of one value, infinities, and NaNs (sorted last, so a NaN
    /// is picked first and its softmax is NaN, as MLX's is).
    #[test]
    fn ties_and_non_finite_scores_are_ordered_as_argpartition_orders_them() {
        let experts = 128;
        let scale = bf16(&vec![1.0; experts as usize], &[experts]);
        let mut cases: Vec<(&str, Vec<f32>)> = vec![
            (
                "four distinct values",
                (0..experts).map(|i| (i % 4) as f32).collect(),
            ),
            ("one value", vec![0.5; experts as usize]),
            (
                "a tie across the top-8 boundary",
                (0..experts)
                    .map(|i| if i % 9 == 0 { 3.0 } else { (i % 3) as f32 })
                    .collect(),
            ),
        ];
        let mut inf: Vec<f32> = (0..experts).map(|i| i as f32 * 0.01).collect();
        inf[5] = f32::INFINITY;
        inf[70] = f32::NEG_INFINITY;
        cases.push(("an infinity", inf));
        let mut nan: Vec<f32> = (0..experts).map(|i| (i % 7) as f32).collect();
        nan[40] = f32::NAN;
        nan[90] = f32::NAN;
        cases.push(("two NaNs", nan));
        for (what, v) in cases {
            let scores = bf16(&v, &[1, 1, experts]);
            let [got_i, got_w] = route(&scores, &scale, 8).expect("route");
            let [want_i, want_w] = composed(&scores, &scale, 8);
            assert_eq!(u32s(&got_i), u32s(&want_i), "{what}: indices");
            // NaN weights compare by bits too: both sides are MLX's canonical NaN or not.
            assert_eq!(bf16_bits(&got_w), bf16_bits(&want_w), "{what}: weights");
        }
    }

    /// What the kernel does not serve is an error, never a wrong answer.
    #[test]
    fn what_the_kernel_cannot_take_is_an_error() {
        let scale = bf16(&[1.0; 128], &[128]);
        let scores = bf16(&[0.0; 128], &[1, 1, 128]);
        let wide = scores.as_dtype(Dtype::Float32).unwrap();
        assert!(route(&wide, &scale, 8).is_err(), "float32 scores");
        let tall = bf16(&vec![0.0; 17 * 128], &[1, 17, 128]);
        assert!(route(&tall, &scale, 8).is_err(), "a prefill chunk");
        assert!(route(&scores, &scale, 0).is_err(), "top-0");
        assert!(
            route(&scores, &scale, 129).is_err(),
            "top-k past the experts"
        );
        let short = bf16(&[1.0; 64], &[64]);
        assert!(
            route(&scores, &short, 8).is_err(),
            "a scale of the wrong length"
        );
    }
}
