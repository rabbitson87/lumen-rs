//! Pre-flight allocation budget for chunked prefill (005 Phase 3).
//!
//! SQLite tests out-of-memory by making `malloc()` fail on the Nth call and
//! sweeping N upward. That technique does not port literally here: MLX
//! allocation is not fallible from Rust, and on Apple unified memory an
//! oversized request does not return an error at all — it drives the machine
//! into swap, or aborts inside C++ where no Rust `Err` can be produced.
//!
//! So lumen's defense against a too-large allocation is not to handle the
//! failure but to **never make the allocation**: before prefilling, the chunk
//! size is clamped DOWN so that one chunk's full-attention scores buffer
//! `[heads, q_len, kv_len]` stays under a byte budget. That relocates the whole
//! OOM class into the pure integer decision below — which means it is testable
//! at tier 0, at every boundary, with no GPU and no weights.
//!
//! The sweep shape survives the translation intact: instead of failing
//! allocation 1, 2, 3, …, the sweep lowers the *budget* 1, 2, 3, … and requires
//! the same three properties at every step — a chunk that is never zero (a zero
//! chunk makes the prefill loop spin forever), never larger than requested, and
//! never larger than the budget permits except at the documented floor.
//!
//! Both backends computed this identically save for their env-var names, so the
//! arithmetic lives here once. It is deliberately byte-for-byte the shipping
//! computation: this module is a hoist, not a redesign, and [`MIN_CHUNK`]'s
//! budget-overrun window is pinned by test rather than quietly closed.

/// Bytes per element of the scores accumulator. `4` is the f32 worst case; a
/// bf16 path only halves it, so this keeps the bound conservative regardless of
/// the dtype actually chosen at runtime.
pub const SCORES_BYTES_PER_ELEM: u64 = 4;

/// Default scores budget: 8 GB. Chosen to fit any box that can host the models
/// this serves (Metal `maxBufferLength` is ~16 GB+ there).
pub const DEFAULT_SCORES_BUDGET_BYTES: u64 = 8_000_000_000;

/// Floor on the clamped chunk.
///
/// **This floor can exceed the budget, on purpose.** Each chunk costs an `eval`
/// barrier, and chunks small enough to matter here cost more than the memory
/// they save (measured: 2048 → 512 on a 20K prompt is +54–118% prefill time).
/// So when the budget arithmetic asks for fewer than this many tokens, latency
/// wins and the projected bytes go over. [`ChunkDecision::floored`] reports when
/// that has happened; see `over_budget_bytes` for how far over.
pub const MIN_CHUNK: usize = 256;

/// Parse a `*_PREFILL_SCORES_GB` value into a byte budget, falling back to
/// [`DEFAULT_SCORES_BUDGET_BYTES`].
///
/// Non-positive, unparseable and `NaN` values all fall back (`NaN > 0.0` is
/// false, so the filter rejects it rather than propagating a `NaN` into the
/// cast). `inf` and absurdly large values saturate at `u64::MAX` — Rust's
/// float→int casts saturate, so a huge budget disables the clamp instead of
/// wrapping to a tiny one.
///
/// The **second** filter is not redundant with the first, and the sweep in
/// `tests/prefill_budget_faults.rs` is what found that out: a positive value
/// below `1e-9` GB is under one byte, so the cast floors it to `0` and a value
/// that passed the positivity check arrives as no budget at all. A zero budget
/// silently pins every prompt to [`MIN_CHUNK`] — the guard looks like it is
/// working (it logs a clamp) while actually having been switched off. Treat it
/// as the typo it is and fall back, exactly as `0` and `-1` already do.
pub fn parse_scores_budget(raw: Option<&str>) -> u64 {
    raw.and_then(|v| v.parse::<f64>().ok())
        .filter(|&g| g > 0.0)
        .map(|g| (g * 1e9) as u64)
        .filter(|&b| b > 0)
        .unwrap_or(DEFAULT_SCORES_BUDGET_BYTES)
}

/// [`parse_scores_budget`] applied to an environment variable.
pub fn scores_budget_from_env(var: &str) -> u64 {
    parse_scores_budget(std::env::var(var).ok().as_deref())
}

/// Bytes one chunk's `[heads, q_len, kv_len]` f32 scores buffer occupies.
/// Saturating: an overflowing projection is "definitely too big", which is the
/// same answer a wrapping one would get wrong.
pub fn projected_scores_bytes(heads: u64, q_len: u64, kv_len: u64) -> u64 {
    heads
        .saturating_mul(q_len)
        .saturating_mul(kv_len)
        .saturating_mul(SCORES_BYTES_PER_ELEM)
}

/// Largest chunk whose scores stay under `budget_bytes` at the worst-case
/// `kv_upper` (the full prompt length — chunk *i* attends at most that far),
/// floored at [`MIN_CHUNK`].
///
/// `heads` and `kv_upper` are clamped to at least 1 so a degenerate config
/// cannot divide by zero.
pub fn max_safe_chunk(budget_bytes: u64, heads: usize, kv_upper: usize) -> usize {
    let heads = (heads as u64).max(1);
    let kv_upper = (kv_upper as u64).max(1);
    let per_token = heads
        .saturating_mul(kv_upper)
        .saturating_mul(SCORES_BYTES_PER_ELEM)
        .max(1);
    (budget_bytes / per_token).max(MIN_CHUNK as u64) as usize
}

/// The outcome of clamping a requested chunk against the budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkDecision {
    /// What the caller (env or default) asked for.
    pub requested: usize,
    /// What prefill will actually use. Always ≥ 1, always ≤ `requested`.
    pub chunk: usize,
    /// The budget-derived ceiling before `requested` was applied.
    pub max_safe: usize,
    /// Budget used, in bytes.
    pub budget_bytes: u64,
    pub heads: usize,
    pub kv_upper: usize,
}

impl ChunkDecision {
    /// Whether the budget forced the chunk below what was requested — the
    /// condition the backends log.
    pub fn clamped(&self) -> bool {
        self.chunk < self.requested
    }

    /// Whether [`MIN_CHUNK`] rather than the budget determined `max_safe`,
    /// i.e. the projected allocation is knowingly over budget.
    pub fn floored(&self) -> bool {
        self.max_safe == MIN_CHUNK
            && max_safe_unfloored(self.budget_bytes, self.heads, self.kv_upper) < MIN_CHUNK as u64
    }

    /// Bytes the chosen chunk projects to at the worst-case kv length.
    pub fn projected_bytes(&self) -> u64 {
        projected_scores_bytes(
            (self.heads as u64).max(1),
            self.chunk as u64,
            (self.kv_upper as u64).max(1),
        )
    }

    /// How far the projection exceeds the budget (0 when within it). Non-zero
    /// only when [`floored`](Self::floored) — that is the documented window.
    pub fn over_budget_bytes(&self) -> u64 {
        self.projected_bytes().saturating_sub(self.budget_bytes)
    }
}

/// The ceiling *without* the floor applied — used to tell "the budget chose
/// this" from "the floor did".
fn max_safe_unfloored(budget_bytes: u64, heads: usize, kv_upper: usize) -> u64 {
    let heads = (heads as u64).max(1);
    let kv_upper = (kv_upper as u64).max(1);
    let per_token = heads
        .saturating_mul(kv_upper)
        .saturating_mul(SCORES_BYTES_PER_ELEM)
        .max(1);
    budget_bytes / per_token
}

/// Clamp `requested` against the budget. This is the whole pre-flight decision;
/// both prefill loops call it and then log iff [`ChunkDecision::clamped`].
///
/// `requested` of 0 would make the prefill loop spin forever, so it is raised to
/// 1 here as well as filtered at every env parse — a loop that never terminates
/// is a worse failure than a slow one, and it should take two independent
/// mistakes to reach.
pub fn clamp_chunk(
    requested: usize,
    budget_bytes: u64,
    heads: usize,
    kv_upper: usize,
) -> ChunkDecision {
    let requested = requested.max(1);
    let max_safe = max_safe_chunk(budget_bytes, heads, kv_upper);
    ChunkDecision {
        requested,
        chunk: requested.min(max_safe),
        max_safe,
        budget_bytes,
        heads,
        kv_upper,
    }
}

/// Chunks the grid shrinks to are whole multiples of this many tokens.
pub const GRID_QUANTUM: usize = 256;

/// What a prefill holds at its peak beyond the memory already in use, as the model's shape
/// predicts it: a fixed part for one chunk's activations, what every prompt token keeps (the
/// full-attention K and V, and more), and the largest chunk's attention scores.
///
/// Fitted on three Qwen checkpoints (M3 Max, chunk 2048, prefill high-water mark at 4K-32K):
/// the per-token part is the architecture's to within 1 % (KV plus the scores, stored bf16), and
/// the fixed part tracks the layer count (85-98 MB a layer at a 2048-row chunk); against the
/// measured high-water marks the total is never under and at most 4 % over (the table is in
/// issue #17 and in this module's tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillPeakModel {
    pub layers: usize,
    pub heads: usize,
    /// What one prompt token keeps resident: the full-attention layers' K and V, plus about
    /// `2 * hidden` the measurements found beyond them.
    pub bytes_per_token: u64,
}

/// The fixed part of a prefill's peak, per layer, at a 2048-row chunk (it grows with the chunk).
pub const PEAK_FIXED_BYTES_PER_LAYER: u64 = 100_000_000;
/// Bytes per attention score as the prefill keeps them (bf16), against the f32
/// [`SCORES_BYTES_PER_ELEM`] the scores budget is projected in.
pub const SCORES_STORED_BYTES: u64 = 2;

impl PrefillPeakModel {
    /// The peak a prefill of `len` tokens on `grid` adds to what is in use.
    pub fn transient_bytes(&self, grid: &PrefillGrid, len: usize) -> u64 {
        let fixed =
            PEAK_FIXED_BYTES_PER_LAYER * self.layers as u64 * grid.requested() as u64 / 2048;
        let kept = self.bytes_per_token.saturating_mul(len as u64);
        let mut start = 0usize;
        let mut scores = 0u64;
        for end in grid.pieces(0, len) {
            let q = (end - start) as u64;
            scores = scores.max(
                (self.heads as u64)
                    .saturating_mul(q)
                    .saturating_mul(end as u64)
                    .saturating_mul(SCORES_STORED_BYTES),
            );
            start = end;
        }
        fixed.saturating_add(kept).saturating_add(scores)
    }
}

/// A prefill predicted to need more than the GPU will keep resident (`working_set`, the
/// device's `max_recommended_working_set_size`): past it command buffers fail, or the machine
/// stalls on memory pressure. `resident` is what is in use before the prefill.
pub fn exceeds_working_set(resident: u64, transient: u64, working_set: u64) -> bool {
    resident.saturating_add(transient) > working_set
}

/// Where a bulk prefill cuts its chunks, as a function of position alone.
///
/// The chunk starting at position `at` attends to `at` cached keys plus its
/// own, so its scores buffer is `[heads, q, at + q]`. It is `requested` tokens
/// while that fits the budget, then the largest multiple of [`GRID_QUANTUM`]
/// that does, never below [`MIN_CHUNK`] (whose overrun [`clamp_chunk`]
/// documents). The cuts run from 0, each chunk ending where the next starts.
///
/// Nothing here depends on how long the prompt is. That is the point: when
/// the chunk was sized for a prompt's whole length, a longer prompt was cut
/// somewhere else from its first token on, and a rollback or disk point left
/// on an earlier turn's cut was not a cut of the next turn's cold prefill.
/// Below the position where `requested` stops fitting the cuts are the
/// multiples of `requested`, as they always were.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillGrid {
    requested: usize,
    budget_bytes: u64,
    heads: usize,
}

impl PrefillGrid {
    pub fn new(requested: usize, budget_bytes: u64, heads: usize) -> Self {
        Self {
            requested: requested.max(1),
            budget_bytes,
            heads: heads.max(1),
        }
    }

    /// The chunk size while the budget does not bind.
    pub fn requested(&self) -> usize {
        self.requested
    }

    fn fits(&self, at: usize, q: usize) -> bool {
        projected_scores_bytes(
            self.heads as u64,
            q as u64,
            (at as u64).saturating_add(q as u64),
        ) <= self.budget_bytes
    }

    /// The chunk that starts at `at`.
    pub fn chunk_at(&self, at: usize) -> usize {
        if self.fits(at, self.requested) {
            return self.requested;
        }
        // The largest q with heads·4·q·(at + q) ≤ budget, from the quadratic,
        // then nudged onto the exact integer boundary.
        let cap = self.budget_bytes as f64 / (self.heads as f64 * SCORES_BYTES_PER_ELEM as f64);
        let a = at as f64;
        let mut q = (((a * a + 4.0 * cap).sqrt() - a) / 2.0).max(0.0) as usize;
        while q > 0 && !self.fits(at, q) {
            q -= 1;
        }
        while q + 1 < self.requested && self.fits(at, q + 1) {
            q += 1;
        }
        (q / GRID_QUANTUM * GRID_QUANTUM)
            .max(MIN_CHUNK)
            .min(self.requested)
    }

    /// Where the chunks of `requested` tokens end: the first position whose
    /// chunk is smaller. Every multiple of `requested` up to it is a cut.
    fn uniform_end(&self) -> usize {
        let full_chunks = self.budget_bytes
            / projected_scores_bytes(
                self.heads as u64,
                self.requested as u64,
                self.requested as u64,
            )
            .max(1);
        usize::try_from(full_chunks)
            .unwrap_or(usize::MAX)
            .saturating_mul(self.requested)
    }

    /// The first cut after `pos`.
    pub fn cut_after(&self, pos: usize) -> usize {
        let uniform = self.uniform_end();
        if pos < uniform {
            return (pos / self.requested + 1) * self.requested;
        }
        let mut at = uniform;
        while at <= pos {
            at += self.chunk_at(at);
        }
        at
    }

    /// The last cut at or before `pos` (0 is a cut).
    pub fn cut_at_or_before(&self, pos: usize) -> usize {
        let uniform = self.uniform_end();
        if pos < uniform {
            return pos / self.requested * self.requested;
        }
        let mut at = uniform;
        loop {
            let next = at + self.chunk_at(at);
            if next > pos {
                return at;
            }
            at = next;
        }
    }

    pub fn is_cut(&self, pos: usize) -> bool {
        self.cut_at_or_before(pos) == pos
    }

    /// The cuts in `(start, end)`, ascending, then `end`.
    pub fn cuts(&self, start: usize, end: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut at = self.cut_after(start);
        while at < end {
            out.push(at);
            at += self.chunk_at(at);
        }
        out.push(end);
        out
    }

    /// Where one bulk call over `[start, end)` ends its pieces: from `start`,
    /// each as long as the chunk the grid starts at that position, then `end`.
    /// From a cut these are the grid's own cuts.
    pub fn pieces(&self, start: usize, end: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut at = start;
        while at < end {
            at = (at + self.chunk_at(at)).min(end);
            out.push(at);
        }
        out
    }

    /// What names this grid in a disk key for the state at cut `at`: the
    /// chunk size where every chunk before `at` is a full one, which is what
    /// keys were before the grid shrank by position, so those states keep
    /// theirs. Past that, the whole grid.
    pub fn key_at(&self, at: usize) -> u64 {
        if at <= self.uniform_end() {
            return self.requested as u64;
        }
        let mix = |v: u64| {
            let v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            let v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            v ^ (v >> 31)
        };
        mix(mix(mix(self.requested as u64) ^ self.budget_bytes) ^ self.heads as u64) | 1 << 63
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_parsing_matches_the_shipping_rules() {
        assert_eq!(parse_scores_budget(None), DEFAULT_SCORES_BUDGET_BYTES);
        assert_eq!(parse_scores_budget(Some("8")), 8_000_000_000);
        assert_eq!(parse_scores_budget(Some("0.5")), 500_000_000);
        // Rejected → default, never zero (a zero budget would floor every
        // prompt to MIN_CHUNK regardless of size).
        for bad in ["0", "-1", "", "abc", "NaN"] {
            assert_eq!(
                parse_scores_budget(Some(bad)),
                DEFAULT_SCORES_BUDGET_BYTES,
                "{bad:?} should fall back to the default"
            );
        }
        // Saturating, not wrapping: a huge budget must disable the clamp, not
        // wrap around into a tiny one.
        assert_eq!(parse_scores_budget(Some("inf")), u64::MAX);
        assert_eq!(parse_scores_budget(Some("1e30")), u64::MAX);
    }

    #[test]
    fn projection_saturates_instead_of_wrapping() {
        assert_eq!(projected_scores_bytes(2, 3, 5), 120);
        assert_eq!(projected_scores_bytes(u64::MAX, u64::MAX, 2), u64::MAX);
    }

    /// Every cut from 0 through `end`, by walking the grid one chunk at a time.
    fn walk(grid: &PrefillGrid, end: usize) -> Vec<usize> {
        let mut cuts = vec![0];
        while *cuts.last().unwrap() < end {
            let at = *cuts.last().unwrap();
            cuts.push(at + grid.chunk_at(at));
        }
        cuts
    }

    /// Grids at the shipping budget for 16 and 24 heads, and small ones whose
    /// budget binds within a few chunks so every branch is walked quickly.
    fn grids() -> Vec<PrefillGrid> {
        vec![
            PrefillGrid::new(2048, DEFAULT_SCORES_BUDGET_BYTES, 16),
            PrefillGrid::new(2048, DEFAULT_SCORES_BUDGET_BYTES, 24),
            PrefillGrid::new(4096, 3_000_000_000, 40),
            PrefillGrid::new(1000, 400_000_000, 8),
            PrefillGrid::new(2048, 0, 16),
        ]
    }

    /// Each chunk is `requested` while that fits, then the largest multiple
    /// of the quantum that fits, and only the floor goes over the budget.
    #[test]
    fn each_chunk_is_the_largest_that_fits_where_it_starts() {
        for grid in grids() {
            for at in walk(&grid, 300_000) {
                let q = grid.chunk_at(at);
                let ctx = format!("{grid:?} at={at} q={q}");
                assert!(q >= 1 && q <= grid.requested(), "{ctx}");
                if q == grid.requested() {
                    assert!(grid.fits(at, q), "{ctx}");
                    continue;
                }
                assert!(q % GRID_QUANTUM == 0, "{ctx}");
                assert!(
                    grid.fits(at, q) || q == MIN_CHUNK,
                    "{ctx}: over budget above the floor"
                );
                assert!(
                    !grid.fits(at, q + GRID_QUANTUM),
                    "{ctx}: a larger chunk fits"
                );
            }
        }
    }

    /// Below where `requested` stops fitting, the cuts are its multiples:
    /// the grid every prompt was cut on before, so short and medium prompts
    /// prefill exactly as they did.
    #[test]
    fn below_the_bound_the_cuts_are_multiples_of_the_chunk() {
        let grid = PrefillGrid::new(2048, DEFAULT_SCORES_BUDGET_BYTES, 16);
        assert_eq!(grid.uniform_end(), 29 * 2048);
        let cuts = walk(&grid, 200_000);
        for (k, &c) in cuts.iter().take(30).enumerate() {
            assert_eq!(c, k * 2048);
        }
        assert!(
            cuts[31] - cuts[30] < 2048,
            "the chunk after the bound shrinks"
        );
        // A prompt that ends before the first short chunk would is cut as before.
        assert_eq!(grid.cuts(0, 61_000).len(), 30);
        assert_eq!(*grid.cuts(0, 61_000).last().unwrap(), 61_000);
    }

    /// The lookups agree with walking the grid, at, between and around cuts.
    #[test]
    fn the_lookups_agree_with_walking_the_grid() {
        for grid in grids() {
            let cuts = walk(&grid, 300_000);
            for w in cuts.windows(2) {
                let (a, b) = (w[0], w[1]);
                for pos in [a, a + 1, (a + b) / 2, b - 1] {
                    if pos >= b {
                        continue;
                    }
                    let ctx = format!("{grid:?} pos={pos} between {a} and {b}");
                    assert_eq!(grid.cut_at_or_before(pos), a, "{ctx}");
                    assert_eq!(grid.cut_after(pos), b, "{ctx}");
                    assert_eq!(grid.is_cut(pos), pos == a, "{ctx}");
                }
            }
            let end = cuts[cuts.len() / 2] + 7;
            let inside: Vec<usize> = cuts
                .iter()
                .copied()
                .filter(|&c| c > 0 && c < end)
                .chain([end])
                .collect();
            assert_eq!(grid.cuts(0, end), inside, "{grid:?}");
            assert_eq!(
                grid.pieces(0, end),
                inside,
                "{grid:?}: pieces from a cut are its cuts"
            );
        }
    }

    /// A state's disk key names its grid by the chunk size wherever every
    /// chunk before it is a full one, so states written before the grid
    /// depended on position keep their keys; past that, grids that cut
    /// differently get different keys.
    #[test]
    fn a_disk_key_names_the_grid_a_state_was_cut_on() {
        let a = PrefillGrid::new(2048, DEFAULT_SCORES_BUDGET_BYTES, 16);
        let b = PrefillGrid::new(2048, DEFAULT_SCORES_BUDGET_BYTES, 24);
        assert_eq!(a.key_at(20 * 2048), 2048);
        assert_eq!(b.key_at(10 * 2048), 2048);
        let past = a.cut_after(a.uniform_end());
        assert_ne!(a.key_at(past), 2048);
        assert_ne!(a.key_at(past), b.key_at(past));
        assert_eq!(
            a.key_at(past),
            a.key_at(a.cut_after(past)),
            "one key per grid"
        );
    }

    #[test]
    fn degenerate_inputs_do_not_divide_by_zero() {
        // heads = 0 and kv_upper = 0 are impossible configs, but the shipping
        // code guards them with `.max(1)` and so must this.
        let d = clamp_chunk(2048, DEFAULT_SCORES_BUDGET_BYTES, 0, 0);
        assert!(d.chunk >= 1);
        let d = clamp_chunk(0, DEFAULT_SCORES_BUDGET_BYTES, 32, 4096);
        assert_eq!(d.chunk, 1, "a zero request must not become a zero chunk");
    }
}

#[cfg(test)]
mod peak_model {
    use super::*;

    fn grid(heads: usize) -> PrefillGrid {
        PrefillGrid::new(2048, DEFAULT_SCORES_BUDGET_BYTES, heads)
    }

    /// The prediction against the prefill high-water marks measured on M3 Max (chunk 2048, bf16
    /// KV): `(model, resident GB, [(tokens, high-water GB)])`. It may be over, never under, and
    /// at most 5 % over the total.
    #[test]
    fn the_prediction_covers_the_measured_peaks_closely() {
        // Qwen3.5-9B: 32 layers, 16 heads, 8 full-attention layers x 4 KV heads x 256, hidden 4096.
        // Qwen3.6-27B: 64 layers, 24 heads, 16 full-attention layers x 4 x 256, hidden 5120.
        // Qwen3.6-35B-A3B: 40 layers, 16 heads, 10 full-attention layers x 2 x 256, hidden 2048.
        let kv = |full: u64, kv_heads: u64, hidden: u64| full * 2 * kv_heads * 256 * 2 + 2 * hidden;
        let cases: [(PrefillPeakModel, f64, &[(usize, f64)]); 3] = [
            (
                PrefillPeakModel {
                    layers: 32,
                    heads: 16,
                    bytes_per_token: kv(8, 4, 4096),
                },
                7.35,
                &[
                    (4096, 10.9016),
                    (8192, 11.3380),
                    (16384, 12.2108),
                    (32768, 13.9553),
                ],
            ),
            (
                PrefillPeakModel {
                    layers: 64,
                    heads: 24,
                    bytes_per_token: kv(16, 4, 5120),
                },
                15.24,
                &[
                    (4096, 21.3575),
                    (8192, 22.0616),
                    (16384, 23.4722),
                    (32768, 26.2895),
                ],
            ),
            (
                PrefillPeakModel {
                    layers: 40,
                    heads: 16,
                    bytes_per_token: kv(10, 2, 2048),
                },
                18.47,
                &[(4096, 22.3342), (16384, 23.3068)],
            ),
        ];
        for (model, resident, points) in cases {
            for &(len, measured) in points {
                let predicted =
                    resident + model.transient_bytes(&grid(model.heads), len) as f64 / 1e9;
                let over = predicted / measured - 1.0;
                assert!(
                    (0.0..=0.05).contains(&over),
                    "{model:?} at {len}: predicted {predicted:.2} GB, measured {measured:.2} ({over:+.3})"
                );
            }
        }
    }

    /// Past the uniform chunks the scores stop growing with the prompt (the chunk shrinks to
    /// keep them in budget), so the peak grows only by what each token keeps.
    #[test]
    fn past_the_scores_budget_only_the_kept_bytes_grow() {
        let m = PrefillPeakModel {
            layers: 64,
            heads: 24,
            bytes_per_token: 75_776,
        };
        let g = grid(24);
        let (a, b) = (
            m.transient_bytes(&g, 100_000),
            m.transient_bytes(&g, 110_000),
        );
        let kept = m.bytes_per_token * 10_000;
        assert!(b > a && b - a <= kept + kept / 10, "{a} -> {b}");
    }

    #[test]
    fn a_prefill_past_the_working_set_is_refused_and_one_inside_is_not() {
        let gb = 1_000_000_000u64;
        assert!(exceeds_working_set(15 * gb, 16 * gb, 30 * gb));
        assert!(!exceeds_working_set(15 * gb, 15 * gb, 30 * gb));
        assert!(!exceeds_working_set(u64::MAX, 0, u64::MAX));
    }
}
