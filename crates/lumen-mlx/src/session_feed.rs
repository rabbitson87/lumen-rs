//! How a chat session's prompt is fed so the next turn can resume it from the
//! conversation boundary — without changing a single logit of this one.
//!
//! A Qwen generation prompt ends in a header (`<|im_start|>assistant\n<think>…`)
//! that the next turn does not reproduce: 3.5 and 3.6 templates drop the
//! `<think>` block from replayed assistant turns, and on any checkpoint a
//! client that does not send the trace back drops it too. So a session's own
//! tokens are never a prefix of the next prompt, and exact-extension reuse
//! misses on every turn — the whole conversation is prefilled again. What the
//! next prompt *does* reproduce is everything before that header, and a
//! rollback point there lets the next turn wind the sequence back and feed
//! only what is new.
//!
//! The hybrid models make that point impossible to get after the fact: the
//! linear-attention state at an earlier position cannot be recomputed from a
//! later one, so it has to be captured while the prompt goes through, which
//! means cutting the prefill there. A cut is exact only if every row is still
//! computed exactly as one bulk pass computes it. [`plan_feed`] keeps the bulk
//! pass's chunk grid, and cuts inside a chunk only where the model's rows do
//! not depend on which other rows share the call — and then only with
//! [`MIN_BULK_PIECE`] rows on each side — so the prefill with a rollback point
//! in it is the same computation as the prefill without one.

/// Fewest rows a piece can have and still run through the kernels a bulk
/// prefill uses, on a model whose rows are computed independently (dense).
///
/// MLX picks kernels by row count. Quantized projections switch to a batched
/// vector kernel below a shape- and GPU-dependent limit that tops out at 32
/// (`get_qmv_batch_limit` in the MLX Metal backend), and attention with fewer
/// than nine query rows takes `sdpa_vector`. A different kernel sums in a
/// different order, so a row computed in a short piece need not match the same
/// row computed inside a long one. At or above this size the kernels are the
/// ones a bulk chunk uses, and so are the rows — measured on Qwen3.5-9B: a
/// prefill cut this way, and a turn resumed from the cut, give the same tokens
/// as a bulk prefill in every case the planner distinguishes.
pub(crate) const MIN_BULK_PIECE: usize = 32;

/// The shape of a bulk prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PrefillGrid {
    /// The chunk size the bulk pass runs in.
    pub chunk: usize,
    /// Fewest rows a piece cut inside a chunk can have and still be computed
    /// as the bulk pass computes it — or `None` when no such cut is exact, and
    /// only the chunk boundaries themselves can be cut at.
    ///
    /// `None` is the mixture-of-experts case. MLX's `GatherQMM` sorts the
    /// routed rows by expert and tiles across them, so a row's result depends
    /// on the other rows in its call. Measured on Qwen3.6-35B-A3B: a cut that
    /// left 32 rows changed a greedy answer after about thirty words, and a
    /// turn resumed from a cut differed from a cold prefill even with every
    /// piece above MLX's batching threshold.
    pub min_piece: Option<usize>,
}

/// How to feed `prompt[start..end]`: where each call ends, and which of those
/// ends gets the rollback point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FeedPlan {
    /// End of each call, ascending; the last one is `end`.
    pub cuts: Vec<usize>,
    /// The cut to leave the rollback point at, if one fits at or before the
    /// boundary.
    pub mark: Option<usize>,
}

/// Plan the feed of `prompt[start..end]`, leaving a rollback point as close to
/// `boundary` as exactness allows.
///
/// `grid` is the shape of a bulk prefill of the whole prompt. Positions are
/// absolute, so the chunk grid is the one a cold prefill from position 0 uses
/// — which is also what makes a later resume from the mark line up with a cold
/// prefill of the longer prompt.
///
/// The mark is the latest position at or before `boundary` that is either a
/// grid point, or — when the grid allows cuts inside a chunk at all — inside a
/// chunk with at least `grid.min_piece` rows on each side of it. Every grid
/// point between `start` and `end` stays a cut as well, so a short last chunk
/// — whose rows a bulk pass computes with the small-row kernels — is still fed
/// as exactly that chunk.
///
/// Without a mark there is nothing to cut for, and the plan is one call — what
/// every path did before rollback points existed.
pub(crate) fn plan_feed(start: usize, end: usize, grid: PrefillGrid, boundary: usize) -> FeedPlan {
    let chunk = grid.chunk.max(1);
    let mark = (start < boundary && boundary < end)
        .then(|| {
            // The grid cell holding the boundary, clipped to what is fed.
            let cell_start = (boundary / chunk * chunk).max(start);
            let cell_end = (boundary / chunk + 1).saturating_mul(chunk).min(end);
            let inside = grid.min_piece.map(|floor| {
                let floor = floor.max(1);
                let at = boundary.min(cell_end.saturating_sub(floor));
                (at >= cell_start + floor).then_some(at)
            });
            if let Some(Some(at)) = inside {
                Some(at)
            } else if cell_start > start {
                // A grid point: the call ends where a bulk chunk ends anyway.
                Some(cell_start)
            } else {
                None
            }
        })
        .flatten();
    let Some(mark) = mark else {
        return FeedPlan {
            cuts: vec![end],
            mark: None,
        };
    };
    let mut cuts: Vec<usize> = (start / chunk + 1..)
        .map(|k| k * chunk)
        .take_while(|&c| c < end)
        .chain([mark, end])
        .collect();
    cuts.sort_unstable();
    cuts.dedup();
    FeedPlan {
        cuts,
        mark: Some(mark),
    }
}

#[cfg(test)]
mod tests {
    use super::{FeedPlan, MIN_BULK_PIECE, PrefillGrid, plan_feed};

    fn grid(chunk: usize, min_piece: Option<usize>) -> PrefillGrid {
        PrefillGrid { chunk, min_piece }
    }

    /// A dense model with the default 2048-token chunk.
    const DENSE: PrefillGrid = PrefillGrid {
        chunk: 2048,
        min_piece: Some(MIN_BULK_PIECE),
    };

    /// A mixture-of-experts model: chunk boundaries only.
    const MOE: PrefillGrid = PrefillGrid {
        chunk: 2048,
        min_piece: None,
    };

    /// The pieces a plan feeds, as `(from, to)`.
    fn pieces(start: usize, plan: &FeedPlan) -> Vec<(usize, usize)> {
        let mut from = start;
        plan.cuts
            .iter()
            .map(|&to| {
                let p = (from, to);
                from = to;
                p
            })
            .collect()
    }

    /// The pieces one bulk prefill of `[0, end)` runs, as `(from, to)`.
    fn bulk_chunks(end: usize, chunk: usize) -> Vec<(usize, usize)> {
        (0..end.div_ceil(chunk))
            .map(|k| (k * chunk, ((k + 1) * chunk).min(end)))
            .collect()
    }

    /// Every invariant the exactness argument rests on, checked for one plan.
    fn check(start: usize, end: usize, chunk: usize, min_piece: Option<usize>, boundary: usize) {
        let plan = plan_feed(start, end, grid(chunk, min_piece), boundary);
        let ctx = format!(
            "start={start} end={end} chunk={chunk} min_piece={min_piece:?} boundary={boundary} {plan:?}"
        );

        // The calls cover the fed range exactly, in order.
        assert_eq!(plan.cuts.last(), Some(&end), "{ctx}");
        assert!(plan.cuts.windows(2).all(|w| w[0] < w[1]), "{ctx}");
        assert!(plan.cuts[0] > start, "{ctx}");

        let Some(mark) = plan.mark else {
            assert_eq!(plan.cuts, vec![end], "no mark, no cuts: {ctx}");
            return;
        };
        assert!(mark > start && mark <= boundary, "{ctx}");
        assert!(plan.cuts.contains(&mark), "{ctx}");

        // No call is longer than a bulk chunk, so the runner never re-chunks it.
        let ps = pieces(start, &plan);
        assert!(ps.iter().all(|&(a, b)| b - a <= chunk), "{ctx}");

        // Without cuts inside a chunk, every cut is a chunk boundary, so the
        // pieces are the bulk pass's own chunks.
        let Some(floor) = min_piece else {
            assert!(
                plan.cuts
                    .iter()
                    .all(|&c| c == end || c.is_multiple_of(chunk)),
                "cut inside a chunk: {ctx}"
            );
            return;
        };

        // Each row lands in a piece of the same kind as in the bulk pass: a
        // bulk-sized piece where the bulk chunk was bulk-sized, and exactly the
        // same piece where the bulk chunk was short.
        for &(a, b) in &bulk_chunks(end, chunk) {
            let mine: Vec<_> = ps.iter().filter(|&&(x, y)| x < b && y > a).collect();
            if b - a < floor {
                if a >= start {
                    assert_eq!(mine, vec![&(a, b)], "short bulk chunk re-cut: {ctx}");
                }
            } else {
                for &&(x, y) in &mine {
                    // The first piece of a resume starts where the last turn's
                    // mark left it; this plan cannot move it.
                    if x == start && !start.is_multiple_of(chunk) {
                        continue;
                    }
                    assert!(y - x >= floor, "short piece {x}..{y}: {ctx}");
                }
            }
        }
    }

    #[test]
    fn a_cut_never_changes_which_kernels_a_row_goes_through() {
        for chunk in [64usize, 256, 2048] {
            for min_piece in [Some(MIN_BULK_PIECE), Some(chunk / 2), None] {
                for end in 1..=3 * chunk + 70 {
                    for header in [3, 10, 31, 32, 40] {
                        let Some(boundary) = end.checked_sub(header) else {
                            continue;
                        };
                        check(0, end, chunk, min_piece, boundary);
                        // A resume from a mark an earlier turn left.
                        for start in [1, chunk - 1, chunk, chunk + MIN_BULK_PIECE] {
                            if start < end {
                                check(start, end, chunk, min_piece, boundary);
                            }
                        }
                    }
                }
            }
        }
    }

    /// On a mixture-of-experts model the point goes to the last chunk boundary
    /// before the header — a resume re-feeds more, but feeds it exactly as a
    /// cold prefill would. Shorter than one chunk, there is no such boundary.
    #[test]
    fn a_mixture_of_experts_marks_only_on_chunk_boundaries() {
        assert_eq!(plan_feed(0, 4100, DENSE, 4090).mark, Some(4064));
        // 4096 is past the boundary, so the last chunk boundary before it is
        // 2048 — the next turn re-feeds two thousand tokens, all of them in
        // the chunks a cold prefill would use.
        let moe = plan_feed(0, 4100, MOE, 4090);
        assert_eq!(moe.mark, Some(2048));
        assert_eq!(moe.cuts, vec![2048, 4096, 4100]);
        assert_eq!(plan_feed(0, 11_634, MOE, 11_624).mark, Some(10_240));
        assert_eq!(plan_feed(0, 600, MOE, 590).mark, None);
    }

    /// The case this exists for: an 11.6K-token turn ending in Qwen's 10-token
    /// thinking-off header. The mark lands 32 rows before the end — 22 before
    /// the header — so the next turn re-feeds a few dozen tokens, not 11.6K.
    #[test]
    fn a_long_turn_keeps_all_but_its_last_few_dozen_tokens() {
        let plan = plan_feed(0, 11_634, DENSE, 11_624);
        assert_eq!(plan.mark, Some(11_602));
        assert_eq!(
            plan.cuts,
            vec![2048, 4096, 6144, 8192, 10_240, 11_602, 11_634]
        );
    }

    /// A short last bulk chunk (here 20 rows) is computed by the small-row
    /// kernels, so it must be fed as exactly that chunk — the mark goes to its
    /// grid point instead of inside it.
    #[test]
    fn a_short_last_chunk_is_fed_whole() {
        let plan = plan_feed(0, 4116, DENSE, 4106);
        assert_eq!(plan.mark, Some(4096));
        assert_eq!(plan.cuts, vec![2048, 4096, 4116]);
    }

    /// Shorter than two bulk pieces before the header: no exact place for a
    /// point, so the prompt goes in one call, exactly as before.
    #[test]
    fn a_prompt_too_short_to_cut_is_fed_in_one_call() {
        let plan = plan_feed(0, 40, DENSE, 30);
        assert_eq!(
            plan,
            FeedPlan {
                cuts: vec![40],
                mark: None
            }
        );
    }

    /// A boundary outside the fed range is no boundary at all.
    #[test]
    fn a_boundary_outside_the_feed_places_no_mark() {
        assert_eq!(plan_feed(500, 900, DENSE, 400).mark, None);
        assert_eq!(plan_feed(500, 900, DENSE, 500).mark, None);
        assert_eq!(plan_feed(0, 900, DENSE, 900).mark, None);
    }
}
