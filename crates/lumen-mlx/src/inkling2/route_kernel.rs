//! The expert router's top-k as one Metal kernel.
//!
//! [`ops::route_topk`](super::ops::route_topk) is the plain form: a sigmoid, a bias add, a negate, an
//! `argpartition`, a gather, a sum, a divide and a multiply, about ten launches a call and one call
//! for each of the model's 48 layer visits. This is the same arithmetic in one launch a token row.
//!
//! Decode-sized passes only (`fused::serves`): a prefill chunk keeps the composed ops. One
//! threadgroup per row, a thread for each expert (strided past 1024). Every thread scores its
//! expert and keys it by `score + bias`; then ranks it by counting the experts whose key beats it
//! (ties go to the lower index, so the ranks are a permutation); the experts of rank below `k` write
//! their index and score at their rank; `k` threads renormalise the picked scores.
//!
//! Where the plain form returns the `k` best in `argpartition`'s unspecified order, this returns
//! them best first. The set of experts and each one's weight agree to float32 rounding; the order
//! differs, and the experts' weighted sum adds in that order, so a layer's output agrees to rounding,
//! not bit for bit: a `Behavior` flag, like the mix kernel.

use std::cell::OnceCell;

use anyhow::{Context, Result, anyhow, ensure};
use mlx_rs::{Array, Dtype};

use crate::metal_kernel::{MetalKernel, MetalKernelConfig};

lumen_flags::flag! {
    /// Pick each token's experts in one Metal kernel instead of the composed ops (a sigmoid, an
    /// `argpartition`, a gather, a renormalisation: about ten launches a layer visit become one).
    /// The kernel returns the chosen experts best first and not in `argpartition`'s order, so the
    /// experts' weighted sum adds in another order: a different-but-equivalent numeric regime, not a
    /// pure optimization. Decode-sized passes only (a verify block too): prefill keeps the composed ops.
    ///
    /// On by default: a decode token about 0.7 ms (2%) faster (16 same-session runs, standard error
    /// 0.17 ms). Teacher-forced over 2,745 positions of 20 prompts it flips the greedy choice at 1.09%
    /// of them, none by more than 1.7 bfloat16 steps (together with the mix kernel 1.06%, up to
    /// 8.7 steps), against 0.9 to 1.3% (up to 7 steps) for the verify blocks that
    /// `LUMEN_INKLING2_DRAFT` ships on. `=0` restores the composed ops.
    pub(crate) kernel {
        env: "LUMEN_INKLING2_ROUTE_KERNEL",
        default: true,
        kind: Behavior,
    }
}

/// Whether the router's top-k runs as a kernel.
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

/// The simdgroup width: a launch is a whole number of simdgroups.
const SIMD: i32 = 32;

/// Most threads of a launch, the device's limit for a threadgroup.
const MAX_THREADS: i32 = 1024;

/// Most experts one token picks: the first `k` threads renormalise the picked scores.
const MAX_K: i32 = 32;

/// Threads of a launch: one for each expert, rounded up to a simdgroup, between one simdgroup and
/// [`MAX_THREADS`] (past that a thread takes several experts, strided).
fn threads_for(experts: i32) -> i32 {
    ((experts + SIMD - 1) / SIMD * SIMD).clamp(SIMD, MAX_THREADS)
}

/// `logits` is `[N, E]` float32, `bias` `[E]`, `route_scale` one float. Outputs `inds` `[N, K]` uint32
/// and `weights` `[N, K]` float32. `E`, `K` and `THREADS` are template arguments. The sigmoid is
/// MLX's own form (the exponential of `-|x|`, mirrored for negative `x`), the renormalisation adds
/// `1e-20` to the sum as the plain form does.
const SOURCE: &str = r#"
    const int row = threadgroup_position_in_grid.x;
    const int tid = thread_position_in_threadgroup.x;

    threadgroup float score[E];
    threadgroup float key[E];
    threadgroup float picked[K];

    for (int e = tid; e < E; e += THREADS) {
        const float x = logits[(size_t)row * E + e];
        const float y = 1.0f / (1.0f + exp(-abs(x)));
        const float s = x < 0.0f ? 1.0f - y : y;
        const float biased = s + bias[e];
        score[e] = s;
        key[e] = isnan(biased) ? -INFINITY : biased;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = tid; e < E; e += THREADS) {
        const float mine = key[e];
        int rank = 0;
        for (int j = 0; j < E; ++j) {
            const float other = key[j];
            rank += (other > mine || (other == mine && j < e)) ? 1 : 0;
        }
        if (rank < K) {
            picked[rank] = score[e];
            inds[(size_t)row * K + rank] = (uint)e;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < K) {
        float total = 0.0f;
        for (int r = 0; r < K; ++r) {
            total += picked[r];
        }
        total += 1e-20f;
        weights[(size_t)row * K + tid] = picked[tid] / total * route_scale;
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
                "inkling2_route_topk",
                &["logits", "bias", "route_scale"],
                &["inds", "weights"],
                SOURCE,
                /* ensure_row_contiguous */ true,
                /* atomic_outputs */ false,
            )?;
            let _ = cell.set(built);
        }
        f(cell
            .get()
            .ok_or_else(|| anyhow!("the router top-k kernel is missing"))?)
    })
}

/// [`ops::route_topk`](super::ops::route_topk) as one kernel launch: `logits` `[N, E]` float32,
/// `bias` `[E]`; returns `([N, k] uint32 indices, [N, k] float32 weights)`, best first.
pub(super) fn route_topk(
    logits: &Array,
    bias: &Array,
    k: i32,
    route_scale: f32,
) -> Result<(Array, Array)> {
    ensure!(
        logits.dtype() == Dtype::Float32 && bias.dtype() == Dtype::Float32,
        "the router's logits and bias are float32, got {:?} and {:?}",
        logits.dtype(),
        bias.dtype()
    );
    let &[rows, experts] = logits.shape() else {
        return Err(anyhow!("the logits are [N, E], got {:?}", logits.shape()));
    };
    ensure!(
        bias.shape() == [experts],
        "the bias is [{experts}], got {:?}",
        bias.shape()
    );
    ensure!(
        (1..=experts.min(MAX_K)).contains(&k),
        "the router kernel picks 1 to {} experts, got {k} of {experts}",
        experts.min(MAX_K)
    );

    #[cfg(test)]
    LAUNCHES.with(|n| n.set(n.get() + 1));
    let threads = threads_for(experts);
    let scale = Array::from_f32(route_scale);
    let mut outputs = with_kernel(|kernel| {
        let config = MetalKernelConfig::new();
        config.add_template_arg_int("E", experts)?;
        config.add_template_arg_int("K", k)?;
        config.add_template_arg_int("THREADS", threads)?;
        config.set_grid(rows * threads, 1, 1)?;
        config.set_thread_group(threads, 1, 1)?;
        config.add_output_arg(&[rows, k], Dtype::Uint32)?;
        config.add_output_arg(&[rows, k], Dtype::Float32)?;
        kernel
            .apply(&[logits, bias, &scale], &config, 2)
            .context("the router top-k kernel failed")
    })?
    .into_iter();
    let inds = outputs
        .next()
        .ok_or_else(|| anyhow!("the router kernel returned no indices"))?;
    let weights = outputs
        .next()
        .ok_or_else(|| anyhow!("the router kernel returned no weights"))?;
    Ok((inds, weights))
}

#[cfg(test)]
mod tests {
    use super::super::ops;
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

    /// `a`'s values in row-major order. `as_slice` reads the buffer as laid out, so a view (the plain
    /// form's indices are a slice of the `[N, E]` partition) is made contiguous first.
    fn host_f32(a: &Array) -> Vec<f32> {
        let flat = mlx_rs::ops::reshape(a, &[-1]).expect("flatten");
        flat.eval().expect("eval");
        flat.as_slice::<f32>().to_vec()
    }

    fn host_u32(a: &Array) -> Vec<u32> {
        let flat = mlx_rs::ops::reshape(a, &[-1]).expect("flatten");
        flat.eval().expect("eval");
        flat.as_slice::<u32>().to_vec()
    }

    /// Per row, the picked (expert, weight) pairs ordered by expert.
    fn by_expert(inds: &[u32], weights: &[f32], rows: usize, k: usize) -> Vec<Vec<(u32, f32)>> {
        (0..rows)
            .map(|r| {
                let mut pairs: Vec<(u32, f32)> = (0..k)
                    .map(|i| (inds[r * k + i], weights[r * k + i]))
                    .collect();
                pairs.sort_by_key(|p| p.0);
                pairs
            })
            .collect()
    }

    /// The kernel picks the plain form's experts with the plain form's weights to float32 rounding
    /// (the order within a row differs), over the model's router (128 experts, 8 picked), the
    /// replica's (16, 4), one row and many, logits large enough that the sigmoid saturates, and a
    /// bias big enough to reorder the experts.
    #[test]
    fn the_kernel_picks_the_composed_experts() {
        let mut worst = 0.0f32;
        for (experts, k, route_scale) in
            [(128, 8, 2.5f32), (16, 4, 1.0), (1500, 8, 2.5), (8, 8, 1.0)]
        {
            for rows in [1usize, 3, 64] {
                for (spread, bias_gain) in [(1.0f32, 0.2f32), (12.0, 0.2), (1.0, 3.0)] {
                    let logits: Vec<f32> = noise(5, rows * experts as usize)
                        .into_iter()
                        .map(|x| x * spread)
                        .collect();
                    let bias: Vec<f32> = noise(6, experts as usize)
                        .into_iter()
                        .map(|x| x * bias_gain)
                        .collect();
                    let logits = Array::from_slice(&logits, &[rows as i32, experts]);
                    let bias = Array::from_slice(&bias, &[experts]);
                    let (want_inds, want_weights) =
                        ops::route_topk(&logits, &bias, k, route_scale).unwrap();
                    let (inds, weights) = route_topk(&logits, &bias, k, route_scale).unwrap();
                    assert_eq!(inds.dtype(), want_inds.dtype(), "the index dtype");
                    assert_eq!(inds.shape(), [rows as i32, k]);
                    let want = by_expert(
                        &host_u32(&want_inds.as_dtype(Dtype::Uint32).unwrap()),
                        &host_f32(&want_weights),
                        rows,
                        k as usize,
                    );
                    let got = by_expert(&host_u32(&inds), &host_f32(&weights), rows, k as usize);
                    for (r, (w, g)) in want.iter().zip(&got).enumerate() {
                        let experts_w: Vec<u32> = w.iter().map(|p| p.0).collect();
                        let experts_g: Vec<u32> = g.iter().map(|p| p.0).collect();
                        // The host's own top-k, in f64, says which side is right when they differ.
                        let (lg, bs) = (host_f32(&logits), host_f32(&bias));
                        let key = |e: usize| {
                            let x = f64::from(lg[r * experts as usize + e]);
                            1.0 / (1.0 + (-x).exp()) + f64::from(bs[e])
                        };
                        let mut order: Vec<usize> = (0..experts as usize).collect();
                        order.sort_by(|a, b| key(*b).total_cmp(&key(*a)));
                        let mut host: Vec<u32> =
                            order[..k as usize].iter().map(|&e| e as u32).collect();
                        host.sort_unstable();
                        assert_eq!(
                            experts_w, experts_g,
                            "E={experts} rows={rows} spread={spread} bias={bias_gain} row {r}: the experts (host: {host:?})"
                        );
                        for (a, b) in w.iter().zip(g) {
                            worst = worst.max((a.1 - b.1).abs());
                            assert!(
                                (a.1 - b.1).abs() < 1e-5,
                                "E={experts} row {r} expert {}: weight {} against {}",
                                a.0,
                                a.1,
                                b.1
                            );
                        }
                    }
                }
            }
        }
        eprintln!("router kernel against the composed top-k, worst weight difference: {worst:.2e}");
    }

    /// Best first, and the weights of a row sum to the route scale: what the renormalisation is for.
    #[test]
    fn the_picked_experts_come_best_first_and_sum_to_the_route_scale() {
        let (experts, k, rows) = (128, 8, 4usize);
        let logits = Array::from_slice(&noise(1, rows * experts as usize), &[rows as i32, experts]);
        let bias = Array::from_slice(&noise(2, experts as usize), &[experts]);
        let (inds, weights) = route_topk(&logits, &bias, k, 2.5).unwrap();
        let (inds, weights) = (host_u32(&inds), host_f32(&weights));
        let (l, b) = (host_f32(&logits), host_f32(&bias));
        for r in 0..rows {
            let keys: Vec<f32> = (0..k as usize)
                .map(|i| {
                    let e = inds[r * k as usize + i] as usize;
                    1.0 / (1.0 + (-l[r * experts as usize + e]).exp()) + b[e]
                })
                .collect();
            assert!(keys.windows(2).all(|p| p[0] >= p[1]), "row {r}: {keys:?}");
            let sum: f32 = weights[r * k as usize..][..k as usize].iter().sum();
            assert!(
                (sum - 2.5).abs() < 1e-4,
                "row {r}: the weights sum to {sum}"
            );
        }
    }

    /// Equal keys: the lower expert wins, every expert is picked at most once, and nothing is read
    /// out of range. NaN logits still give `k` distinct valid experts.
    #[test]
    fn ties_and_nans_still_give_distinct_valid_experts() {
        let (experts, k) = (128i32, 8i32);
        let flat = Array::from_slice(&vec![0.5f32; experts as usize], &[1, experts]);
        let zero = Array::from_slice(&vec![0.0f32; experts as usize], &[experts]);
        let (inds, _) = route_topk(&flat, &zero, k, 1.0).unwrap();
        assert_eq!(
            host_u32(&inds),
            (0..k as u32).collect::<Vec<_>>(),
            "ties go to the lower index"
        );

        let mut bad = vec![0.1f32; experts as usize];
        bad[3] = f32::NAN;
        bad[40] = f32::NAN;
        let (inds, _) = route_topk(&Array::from_slice(&bad, &[1, experts]), &zero, k, 1.0).unwrap();
        let mut picked = host_u32(&inds);
        assert!(picked.iter().all(|&e| e < experts as u32), "{picked:?}");
        picked.sort_unstable();
        // A NaN key is demoted below every real one: the 8 lowest real experts are picked, the NaN
        // experts (3 and 40) never.
        assert_eq!(
            picked,
            [0, 1, 2, 4, 5, 6, 7, 8],
            "NaN logits are never picked"
        );
    }

    #[test]
    fn what_the_kernel_cannot_take_is_an_error() {
        let logits = Array::from_slice(&noise(1, 16), &[1, 16]);
        let bias = Array::from_slice(&noise(2, 16), &[16]);
        assert!(route_topk(&logits, &bias, 0, 1.0).is_err(), "k = 0");
        assert!(
            route_topk(&logits, &bias, 17, 1.0).is_err(),
            "k above the experts"
        );
        let wide = Array::from_slice(&noise(1, 64), &[1, 64]);
        let wide_bias = Array::from_slice(&noise(2, 64), &[64]);
        assert!(
            route_topk(&wide, &wide_bias, 33, 1.0).is_err(),
            "k above {MAX_K}"
        );
        let short = Array::from_slice(&noise(2, 8), &[8]);
        assert!(
            route_topk(&logits, &short, 4, 1.0).is_err(),
            "a bias of another width"
        );
        let half = logits.as_dtype(Dtype::Float16).unwrap();
        assert!(route_topk(&half, &bias, 4, 1.0).is_err(), "float16 logits");
    }

    #[test]
    fn the_launch_has_a_thread_for_each_expert() {
        assert_eq!(threads_for(128), 128);
        assert_eq!(threads_for(16), SIMD);
        assert_eq!(threads_for(100), 128);
        assert_eq!(threads_for(1500), MAX_THREADS);
    }
}
