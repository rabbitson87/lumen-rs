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
