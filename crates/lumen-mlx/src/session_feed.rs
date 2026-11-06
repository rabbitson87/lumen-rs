//! How a chat session's prompt is fed so the next turn can resume it from the
//! conversation boundary, with the same numbers a cold prefill gives it.
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
//! On the grid alone the cost is in the next turn, which re-feeds from the last
//! chunk boundary before the header instead of from the header itself: up to
//! one chunk, seconds at long context. So where a family names its turns,
//! [`plan_turns`] also cuts every prefill where an assistant turn starts, and
//! the next turn feeds only what follows that cut. Those pieces are not the
//! bulk pass's chunks, so a prompt's numbers differ from one bulk pass by
//! rounding. What stays exact is the turn against its own cold prefill: the
//! cuts depend only on the tokens before them, so any longer prompt makes the
//! same calls. Each cut adds a call, and short calls are dear, so
//! [`kept_turns`] skips the turns too close to the cut before them.

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

/// How far past the cut before it a turn must start to get a cut of its own.
pub(crate) const MIN_TURN_GAP: usize = 256;

/// The turn starts of `starts` (ascending) that get a cut: those at least `gap` past the cut
/// before them, the grid's or a kept turn's.
///
/// What this skips is what the next turn re-feeds, so `gap` trades that against a cold prefill: a
/// call costs far more than its rows do inside a bulk chunk. Each choice reads only the positions
/// before it, so a longer prompt keeps the same turns.
pub(crate) fn kept_turns(starts: &[usize], grid: PrefillGrid, gap: usize) -> Vec<usize> {
    let mut last = 0;
    let mut kept = Vec::new();
    for &turn in starts {
        last = last.max(grid.cut_at_or_before(turn));
        if turn - last >= gap {
            kept.push(turn);
            last = turn;
        }
    }
    kept
}

/// Plan the feed of `prompt[start..end]` for a prompt cut at its turns ([`kept_turns`]). Each one
/// is a cut alongside the grid's whether or not a point is wanted, so leaving one changes no call.
/// The point goes on the last cut at or before `boundary`, if this feed makes one.
pub(crate) fn plan_turns(
    start: usize,
    end: usize,
    grid: PrefillGrid,
    boundary: Option<usize>,
    turns: &[usize],
) -> FeedPlan {
    let mut cuts = grid.cuts(start, end);
    cuts.extend(turns.iter().copied().filter(|&t| start < t && t < end));
    cuts.sort_unstable();
    cuts.dedup();
    let mark = boundary
        .filter(|&b| b < end)
        .and_then(|b| cuts.iter().rev().copied().find(|&cut| cut <= b));
    FeedPlan { cuts, mark }
}

/// What names `grid` in the disk key of the state at `at`. A state fed with turn cuts was computed
/// by other calls than one fed on the grid alone, so the two never share a key: a resume would not
/// match the cold prefill.
pub(crate) fn point_grid_key(grid: PrefillGrid, at: usize, turn_cuts: bool) -> u64 {
    grid.key_at(at) ^ if turn_cuts { 1 << 62 } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::{FeedPlan, PrefillGrid, kept_turns, plan_feed, plan_turns, point_grid_key};

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

    /// Every turn of a conversation, warm from the turn before, makes the calls its own cold
    /// prefill makes past that point, and the point is always the conversation boundary.
    #[test]
    fn a_turn_resumes_on_the_cuts_of_its_cold_prefill() {
        // Several per chunk, one on a grid cut, none for a while, a run of very short turns.
        let starts = [300, 1200, 2048, 2900, 7000, 7010, 7020, 11_000];
        let mut from = 0;
        for k in 0..starts.len() {
            let (turns, boundary) = (&starts[..=k], Some(starts[k]));
            let end = starts[k] + 7;
            let cold = plan_turns(0, end, grid(), boundary, turns);
            let warm = plan_turns(from, end, grid(), boundary, turns);
            assert_eq!(cold.mark, boundary);
            assert_eq!(warm.mark, boundary);
            for cut in turns.iter().copied().chain(grid().cuts(0, end)) {
                assert!(cold.cuts.contains(&cut), "{cut} missing from {cold:?}");
            }
            assert!(cold.cuts.windows(2).all(|w| w[0] < w[1]), "{cold:?}");
            let tail: Vec<usize> = cold.cuts.iter().copied().filter(|&c| c > from).collect();
            assert_eq!(warm.cuts, tail, "turn {k}");
            from = starts[k];
        }
    }

    /// Whether a point is wanted changes no call: rollback points stay an optimization.
    #[test]
    fn leaving_a_point_changes_no_cut() {
        let turns = [300, 1200, 4090];
        let with = plan_turns(0, 4100, grid(), Some(4090), &turns);
        let without = plan_turns(0, 4100, grid(), None, &turns);
        assert_eq!(with.cuts, without.cuts);
        assert_eq!(with.mark, Some(4090));
        assert_eq!(without.mark, None);
    }

    /// A prompt that adds nothing past the point still ends its calls where its cold prefill
    /// does, and leaves no new point.
    #[test]
    fn a_retry_of_a_turn_keeps_its_point_and_the_cuts() {
        let plan = plan_turns(4090, 4100, grid(), Some(4090), &[300, 4090]);
        assert_eq!(plan.cuts, vec![4096, 4100]);
        assert_eq!(plan.mark, None);
    }

    /// A turn close to the last cut gets none, whichever kind that cut is, and the choice for
    /// any turn is the same in every longer prompt.
    #[test]
    fn a_turn_is_kept_by_its_distance_from_the_cut_before_it() {
        let starts = [100, 300, 400, 600, 2050, 2100, 2400, 2500];
        let kept = kept_turns(&starts, grid(), 256);
        // 100: under the gap from 0. 400: from 300. 2050: from the grid's 2048. 2500: from 2400.
        assert_eq!(kept, vec![300, 600, 2400]);
        for n in 0..starts.len() {
            let prefix: Vec<usize> = kept.iter().copied().filter(|&k| k <= starts[n]).collect();
            assert_eq!(kept_turns(&starts[..=n], grid(), 256), prefix);
        }
    }

    /// A turn with no cut of its own leaves the point where it was, or on the grid cut it
    /// crossed; a warm turn from there still matches its cold prefill.
    #[test]
    fn a_turn_without_a_cut_keeps_the_last_one_as_its_point() {
        let turns = [300, 600];
        let plan = plan_turns(600, 700, grid(), Some(690), &turns);
        assert_eq!((plan.cuts, plan.mark), (vec![700], None));

        let plan = plan_turns(600, 2100, grid(), Some(2090), &turns);
        assert_eq!((plan.cuts, plan.mark), (vec![2048, 2100], Some(2048)));
    }

    /// States computed on different cuts never answer for each other on disk.
    #[test]
    fn turn_states_and_grid_states_never_share_a_disk_key() {
        let at = 4096;
        assert_eq!(point_grid_key(grid(), at, false), grid().key_at(at));
        assert_ne!(point_grid_key(grid(), at, true), grid().key_at(at));
    }
}
