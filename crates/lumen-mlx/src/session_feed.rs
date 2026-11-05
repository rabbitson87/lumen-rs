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
//! computed exactly as one bulk pass computes it, and MLX picks kernels — and
//! how they split their sums — by how many rows share a call. So
//! [`plan_feed`] cuts only where a bulk pass already ends a call: on its chunk
//! grid. The pieces are then the bulk pass's own chunks, the same computation
//! row for row, whatever MLX's heuristics do with a given row count.
//!
//! Cutting inside a chunk was exact on dense models once, and is not now:
//!
//! - MLX 0.32's split-K quantized matmul divides the K sum across
//!   `512 / threadgroups` partitions, and the threadgroup count grows with the
//!   rows. Qwen3.5-9B's linear-attention gate projections (`in_proj_a`/`_b`,
//!   32 outputs) split 8 ways for a 2048-row chunk, 16 for ~600 rows and 64
//!   for 32 — every cut inside a chunk re-orders their sums. A turn with a
//!   point 32 rows before its end changed its answer.
//! - On Qwen3.6-35B-A3B a 32-row piece changed a greedy reply after about
//!   thirty words. That was put down to `GatherQMM` sorting the routed rows
//!   by expert, but on synthetic tensors a sorted `gather_qmm` matches from 32
//!   rows on; head-dim-256 attention, which matches no piece longer than one
//!   row, is the likelier cause (inferred).
//!
//! `tests/row_invariance.rs` pins these regimes without weights, so an MLX
//! bump that moves one fails there first.
//!
//! The cost is in the next turn, which re-feeds from the last chunk boundary
//! before the header instead of from the header itself: at most one chunk.

pub(crate) use crate::prefill_budget::PrefillGrid;

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

/// Plan the feed of `prompt[start..end]`, leaving a rollback point at the last
/// cut of the grid at or before `boundary`.
///
/// Positions are absolute, so the cuts are the ones a cold prefill from
/// position 0 makes. The grid depends on position alone, so a cold prefill of
/// any longer prompt makes them too, which is what lets a later turn resume
/// from the mark and still match its own cold prefill. Every cut between
/// `start` and `end` stays a cut, so each call is exactly one of the bulk
/// pass's chunks.
///
/// Without a mark — no grid point after `start` and at or before `boundary` —
/// there is nothing to cut for, and the plan is one call, what every path did
/// before rollback points existed.
pub(crate) fn plan_feed(start: usize, end: usize, grid: PrefillGrid, boundary: usize) -> FeedPlan {
    let mark = (start < boundary && boundary < end)
        .then(|| grid.cut_at_or_before(boundary))
        .filter(|&at| at > start);
    let Some(mark) = mark else {
        return FeedPlan {
            cuts: vec![end],
            mark: None,
        };
    };
    FeedPlan {
        cuts: grid.cuts(start, end),
        mark: Some(mark),
    }
}

#[cfg(test)]
mod tests {
    use super::{FeedPlan, PrefillGrid, plan_feed};

    /// A grid of `chunk`-token chunks whose budget never binds.
    fn uniform(chunk: usize) -> PrefillGrid {
        PrefillGrid::new(chunk, u64::MAX, 1)
    }

    /// The default 2048-token chunk.
    fn grid() -> PrefillGrid {
        uniform(2048)
    }

    /// Every invariant the exactness argument rests on, checked for one plan.
    fn check(start: usize, end: usize, grid: PrefillGrid, boundary: usize) {
        let plan = plan_feed(start, end, grid, boundary);
        let ctx = format!("start={start} end={end} {grid:?} boundary={boundary} {plan:?}");

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
        // The latest such grid point: a later one would fit too.
        assert!(
            grid.cut_after(mark) > boundary,
            "a later grid point was skipped: {ctx}"
        );

        // Every cut is a chunk boundary (or the end), so the pieces are the
        // bulk pass's own chunks — no call is cut inside one — and none is
        // skipped.
        assert!(
            plan.cuts.iter().all(|&c| c == end || grid.is_cut(c)),
            "cut inside a chunk: {ctx}"
        );
        assert!(
            plan.cuts.windows(2).all(|w| grid.cut_after(w[0]) >= w[1]),
            "a call spans a cut: {ctx}"
        );
    }

    #[test]
    fn a_cut_never_changes_which_kernels_a_row_goes_through() {
        for chunk in [64usize, 256, 2048] {
            for end in 1..=3 * chunk + 70 {
                for header in [3, 10, 31, 32, 40] {
                    let Some(boundary) = end.checked_sub(header) else {
                        continue;
                    };
                    check(0, end, uniform(chunk), boundary);
                    // A resume from a mark an earlier turn left.
                    for start in [1, chunk - 1, chunk, chunk + 32] {
                        if start < end {
                            check(start, end, uniform(chunk), boundary);
                        }
                    }
                }
            }
        }
    }

    /// The same on a grid whose chunks shrink with position, as the shipping
    /// budget's do past about 59K tokens at 16 heads; here they shrink from
    /// the fourth chunk on.
    #[test]
    fn a_cut_on_a_shrinking_grid_is_one_of_its_chunks_too() {
        let grid = PrefillGrid::new(1024, 400_000_000, 8);
        for end in (1usize..40_000).step_by(97) {
            for header in [3, 10, 40] {
                let Some(boundary) = end.checked_sub(header) else {
                    continue;
                };
                check(0, end, grid, boundary);
                for start in [grid.cut_at_or_before(boundary), boundary / 2] {
                    if start < end {
                        check(start, end, grid, boundary);
                    }
                }
            }
        }
    }

    /// The point goes to the last chunk boundary before the header — a resume
    /// re-feeds more, but feeds it exactly as a cold prefill would. Shorter
    /// than one chunk, there is no such boundary.
    #[test]
    fn the_point_goes_to_the_last_chunk_boundary_before_the_header() {
        // 4096 is past the boundary, so the last chunk boundary before it is
        // 2048 — the next turn re-feeds two thousand tokens, all of them in
        // the chunks a cold prefill would use.
        let plan = plan_feed(0, 4100, grid(), 4090);
        assert_eq!(plan.mark, Some(2048));
        assert_eq!(plan.cuts, vec![2048, 4096, 4100]);
        assert_eq!(plan_feed(0, 11_634, grid(), 11_624).mark, Some(10_240));
        assert_eq!(plan_feed(0, 4116, grid(), 4106).mark, Some(4096));
        assert_eq!(plan_feed(0, 600, grid(), 590).mark, None);
    }

    /// The case this exists for: an 11.6K-token turn ending in Qwen's 10-token
    /// thinking-off header. The mark lands on the last chunk boundary, so the
    /// next turn re-feeds about 1.4K tokens, not 11.6K.
    #[test]
    fn a_long_turn_keeps_everything_up_to_its_last_chunk() {
        let plan = plan_feed(0, 11_634, grid(), 11_624);
        assert_eq!(plan.mark, Some(10_240));
        assert_eq!(plan.cuts, vec![2048, 4096, 6144, 8192, 10_240, 11_634]);
    }

    /// Shorter than one chunk before the header: no grid point to cut at, so
    /// the prompt goes in one call, exactly as before.
    #[test]
    fn a_prompt_too_short_to_cut_is_fed_in_one_call() {
        let plan = plan_feed(0, 40, grid(), 30);
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
        assert_eq!(plan_feed(500, 900, grid(), 400).mark, None);
        assert_eq!(plan_feed(500, 900, grid(), 500).mark, None);
        assert_eq!(plan_feed(0, 900, grid(), 900).mark, None);
    }
}
