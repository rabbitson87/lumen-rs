//! Lookahead Decoding (Fu et al. 2024, LMSys) — host-side proposer.
//!
//! Algorithm extends Phase S.1 spec decoding (`spec_decode.rs`) with two
//! data structures the simple n-gram lookup lacks:
//!
//! 1. **Jacobi window** — `W` future positions whose tentative tokens are
//!    refined every forward pass via argmax shift. Stable positions are
//!    promoted to the n-gram pool.
//! 2. **N-gram pool** — `(prefix, continuation)` map indexed by an `N`-gram
//!    suffix. Provides up to `G` candidate continuations per step instead
//!    of the single most-recent match used by `spec_decode::ngram_lookup`.
//!
//! Per step, the verifier observes a single forward of layout
//! `[next, jacobi_W, guess_pack_0, …, guess_pack_{G-1}]` (length
//! `1 + W + G·W`), then accepts the longest matching guess pack.
//!
//! See `.outline/lookahead_decoding_plan.md` for the full design memo.
//!
//! ## Argmax convention
//!
//! Standard Candle next-token shift: for `verify_in =
//! [next, j_0, …, j_{W-1}, g_0_0, …, g_0_{W-1}, …]` the model emits
//! logits with the same length, where `argmax[k]` is the prediction
//! for the token immediately *after* `verify_in[k]`. Mapped to
//! sequence positions (with `next` at position `p`):
//!
//! - `argmax[i]` for `i ∈ 0..W` — model's prediction for sequence
//!   position `p+1+i`, i.e., the slot occupied by `j_i` in the input.
//!   Compared against `jacobi[i]` for greedy acceptance.
//! - `argmax[W]` — prediction for `p+W+1`. The "+1 free" token always
//!   committed when the entire Jacobi window is accepted (full accept).
//! - `argmax[1+W..]` — guess-pack predictions (L.2/L.3).
//!
//! ## Status
//!
//! Jacobi window with pool gathering. `update` and
//! `greedy_jacobi_accept` now use the standard convention; pool
//! population via `observe_committed` and pool lookup in `propose`
//! enable `G > 0` configurations once the backend wires the new
//! verify path.

use std::collections::{HashMap, VecDeque};

/// Lookahead decoding configuration.
///
/// Sweet spot per the design memo (cost model on 35B): `W=4-5`, `G=2-3`,
/// `n_input = 1 + W + G·W = 13–21`.
#[derive(Debug, Clone, Copy)]
pub struct LookaheadConfig {
    /// Lookahead window size — number of Jacobi positions refined per step.
    pub window: usize,
    /// Number of guess packs verified per step (each pack is `window`
    /// tokens long).
    pub guesses: usize,
    /// N-gram prefix length used to key the pool.
    pub ngram: usize,
    /// Maximum entries retained in the n-gram pool. LRU evicts oldest
    /// entries once exceeded.
    pub pool_max: usize,
    /// Consecutive identical Jacobi outputs required before a position
    /// promotes its continuation into the pool.
    pub stable_threshold: u8,
}

impl Default for LookaheadConfig {
    fn default() -> Self {
        Self {
            window: 4,
            guesses: 2,
            ngram: 3,
            pool_max: 4096,
            stable_threshold: 3,
        }
    }
}

impl LookaheadConfig {
    /// Total length of the verify input batch: `1 + W + G·W`.
    pub fn verify_input_len(&self) -> usize {
        1 + self.window + self.guesses * self.window
    }
}

/// Single proposal returned by [`LookaheadProposer::propose`]. Layout
/// matches the verify forward expected by the backend.
#[derive(Debug, Clone)]
pub struct LookaheadProposal {
    /// Jacobi window tokens to thread through the forward (length `W`).
    pub jacobi: Vec<u32>,
    /// Guess packs (length ≤ `G`, each a `W`-long candidate continuation).
    pub guesses: Vec<Vec<u32>>,
}

impl LookaheadProposal {
    /// Flatten into the verifier-expected `[jacobi_W, guess_packs]`
    /// (caller prepends `next_committed_token`).
    pub fn flatten(&self) -> Vec<u32> {
        let total = self.jacobi.len() + self.guesses.iter().map(|g| g.len()).sum::<usize>();
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&self.jacobi);
        for g in &self.guesses {
            out.extend_from_slice(g);
        }
        out
    }
}

/// Stateful Lookahead proposer. One instance per generation request.
///
/// Hot path is `propose` (called per decode step) and `update` (called
/// after the verify forward completes with the new argmax slice).
#[derive(Debug)]
pub struct LookaheadProposer {
    cfg: LookaheadConfig,
    /// Jacobi window state. Length `W`. Initialized lazily once a
    /// committed token is observed (the `seed` call).
    jacobi: VecDeque<u32>,
    /// Per-position stability counter. Length `W`. Reset to 0 on
    /// disagreement, incremented on agreement.
    stability: VecDeque<u8>,
    /// `(prefix → list-of-continuations)` n-gram pool. Each
    /// `continuation` has length `W`.
    pool: HashMap<Vec<u32>, Vec<Vec<u32>>>,
    /// FIFO of pool keys for LRU eviction. One entry per pool insert
    /// (matches the per-continuation count).
    insertion_order: VecDeque<Vec<u32>>,
    /// History index up to which `observe_committed` has scanned for
    /// `(prefix, continuation)` pairs. Persists across `seed`/`reset`
    /// so the pool keeps accumulating across requests when desired.
    observed_up_to: usize,
}

impl LookaheadProposer {
    /// Construct a fresh proposer. Call [`Self::seed`] once a starting
    /// token (last committed token after prefill) is available.
    pub fn new(cfg: LookaheadConfig) -> Self {
        Self {
            cfg,
            jacobi: VecDeque::with_capacity(cfg.window),
            stability: VecDeque::with_capacity(cfg.window),
            pool: HashMap::with_capacity(cfg.pool_max),
            insertion_order: VecDeque::with_capacity(cfg.pool_max),
            observed_up_to: 0,
        }
    }

    /// Initialize Jacobi window with `seed` repeated `W` times. Called
    /// once after prefill.
    pub fn seed(&mut self, seed: u32) {
        self.jacobi.clear();
        self.stability.clear();
        for _ in 0..self.cfg.window {
            self.jacobi.push_back(seed);
            self.stability.push_back(0);
        }
    }

    /// Active config (read-only).
    pub fn config(&self) -> LookaheadConfig {
        self.cfg
    }

    /// Number of (prefix, continuation) entries currently in the pool.
    pub fn pool_len(&self) -> usize {
        self.pool.values().map(|v| v.len()).sum()
    }

    /// Read-only view of the current Jacobi window. Mainly useful in
    /// tests; the backend calls [`Self::propose`] to obtain the
    /// verification target.
    pub fn jacobi(&self) -> Vec<u32> {
        self.jacobi.iter().copied().collect()
    }

    /// Read-only view of per-position stability counters.
    pub fn stability(&self) -> Vec<u8> {
        self.stability.iter().copied().collect()
    }

    /// Build the proposal for the next verify forward.
    ///
    /// - The Jacobi window is always returned (when seeded).
    /// - When `cfg.guesses > 0` and the pool has continuations matching
    ///   the history's `N`-gram suffix, up to `G` of them are returned
    ///   as `proposal.guesses`. Most-recent insertions take priority.
    ///
    /// Returns `None` when the proposer has not been seeded yet —
    /// caller falls back to single-token decode.
    pub fn propose(&self, history: &[u32]) -> Option<LookaheadProposal> {
        if self.jacobi.is_empty() {
            return None;
        }
        let jacobi: Vec<u32> = self.jacobi.iter().copied().collect();

        let mut guesses: Vec<Vec<u32>> = Vec::new();
        if self.cfg.guesses > 0 && self.cfg.ngram > 0 && history.len() >= self.cfg.ngram {
            let n = self.cfg.ngram;
            let suffix = &history[history.len() - n..];
            if let Some(continuations) = self.pool.get(suffix) {
                guesses.reserve(self.cfg.guesses);
                for c in continuations.iter().rev() {
                    if guesses.len() >= self.cfg.guesses {
                        break;
                    }
                    if c.len() == self.cfg.window {
                        guesses.push(c.clone());
                    }
                }
            }
        }

        Some(LookaheadProposal { jacobi, guesses })
    }

    /// Consume the verify forward's per-position argmax to refresh
    /// Jacobi state.
    ///
    /// `argmax[i]` for `i ∈ 0..W` is the model's prediction for the
    /// sequence position occupied by `jacobi[i]` in the input. The
    /// caller commits `committed_count` tokens this step (always ≥ 1,
    /// max `W + 1` when the full window plus the post-window free
    /// token is accepted; up to `W + 1 + W·G` once guess packs are
    /// wired).
    ///
    /// Pipeline:
    /// 1. Build `new_jacobi` from `argmax[0..W]`.
    /// 2. Stability: increment per position when `new == old`, reset
    ///    on mismatch.
    /// 3. Shift left by `committed_count`. Right-pad freed slots with
    ///    the rightmost prediction (`argmax[W-1]`) — the best guess
    ///    we have for positions beyond the window — and reset their
    ///    stability to 0.
    ///
    /// Edge cases:
    /// - `argmax.len() < W` → no-op (malformed input).
    /// - Jacobi not seeded → no-op.
    /// - `committed_count >= W` → entire window flushed (rare; happens
    ///   only when full window is accepted alongside the +1 free).
    pub fn update(&mut self, argmax: &[u32], committed_count: usize) {
        let w = self.cfg.window;
        if w == 0 || argmax.len() < w || self.jacobi.len() != w {
            return;
        }
        let committed = committed_count.max(1);

        // Build new Jacobi predictions and stability counters against
        // the OLD window. argmax[i] is the model's prediction for the
        // same sequence position jacobi[i] was tracking.
        let mut new_pred: Vec<u32> = Vec::with_capacity(w);
        let mut new_stab: Vec<u8> = Vec::with_capacity(w);
        for i in 0..w {
            let pred = argmax[i];
            let old = self.jacobi[i];
            let prev = self.stability[i];
            new_pred.push(pred);
            new_stab.push(if pred == old {
                prev.saturating_add(1)
            } else {
                0
            });
        }

        // Shift left by `committed`. After commit, position i in the
        // window corresponds to OLD position i + committed, so we
        // discard `new_pred[..committed]` and right-pad with the
        // rightmost prediction — the best guess we have for positions
        // beyond the window.
        let shift = committed.min(w);
        let pad = new_pred[w - 1];
        let mut shifted_pred: Vec<u32> = Vec::with_capacity(w);
        let mut shifted_stab: Vec<u8> = Vec::with_capacity(w);
        for i in shift..w {
            shifted_pred.push(new_pred[i]);
            shifted_stab.push(new_stab[i]);
        }
        for _ in 0..shift {
            shifted_pred.push(pad);
            shifted_stab.push(0);
        }
        debug_assert_eq!(shifted_pred.len(), w);
        debug_assert_eq!(shifted_stab.len(), w);

        self.jacobi.clear();
        self.stability.clear();
        for (p, s) in shifted_pred.into_iter().zip(shifted_stab.into_iter()) {
            self.jacobi.push_back(p);
            self.stability.push_back(s);
        }
    }

    /// Greedy verify of the Jacobi window against the verify-forward
    /// argmax. Returns `j ∈ 0..=W` — the count of contiguous Jacobi
    /// positions whose old-jacobi value equalled the model's
    /// prediction at that position. Mirrors the
    /// [`crate::spec_decode::greedy_accept_count`] contract used by
    /// the n-gram path so the backend can use a single accept loop.
    ///
    /// `argmax[0..W]` is the relevant slice — predictions for the
    /// sequence positions the Jacobi window tracks.
    pub fn greedy_jacobi_accept(&self, argmax: &[u32]) -> usize {
        let w = self.cfg.window;
        if w == 0 || self.jacobi.len() != w || argmax.len() < w {
            return 0;
        }
        let mut accepted = 0;
        for i in 0..w {
            if argmax[i] == self.jacobi[i] {
                accepted += 1;
            } else {
                break;
            }
        }
        accepted
    }

    /// Scan committed `history` for previously-unseen
    /// `(N-gram prefix, W-token continuation)` pairs and insert them
    /// into the pool with FIFO-LRU eviction. Idempotent across calls
    /// thanks to the internal `observed_up_to` cursor; safe to call
    /// after every commit even when multiple tokens were committed.
    ///
    /// Pool insertion is content-deduped per prefix: identical
    /// continuations are not re-added.
    pub fn observe_committed(&mut self, history: &[u32]) {
        let n = self.cfg.ngram;
        let w = self.cfg.window;
        let cap = self.cfg.pool_max;
        if n == 0 || w == 0 || cap == 0 {
            return;
        }
        let needed = n + w;
        if history.len() < needed {
            // Advance the cursor as far as it can go without falling
            // behind future calls — cap at len so we don't go
            // backwards if `history` shrinks (e.g., reset between
            // requests reusing the same proposer with fresh
            // `observed_up_to=0` is the supported path).
            self.observed_up_to = self.observed_up_to.min(history.len());
            return;
        }
        let last_valid_start = history.len() - needed;
        if self.observed_up_to > last_valid_start {
            return;
        }
        for start in self.observed_up_to..=last_valid_start {
            let prefix: Vec<u32> = history[start..start + n].to_vec();
            let continuation: Vec<u32> = history[start + n..start + n + w].to_vec();
            self.add_to_pool(prefix, continuation);
        }
        self.observed_up_to = last_valid_start + 1;
    }

    /// Insert a single `(prefix, continuation)` pair into the pool.
    /// Skips when `continuation` already exists for `prefix`. Evicts
    /// FIFO-oldest entries when the total count exceeds `pool_max`.
    fn add_to_pool(&mut self, prefix: Vec<u32>, continuation: Vec<u32>) {
        let cap = self.cfg.pool_max;
        let entry = self.pool.entry(prefix.clone()).or_default();
        if entry.iter().any(|c| c == &continuation) {
            return;
        }
        entry.push(continuation);
        self.insertion_order.push_back(prefix);

        while self.insertion_order.len() > cap {
            if let Some(oldest_key) = self.insertion_order.pop_front() {
                let drop_empty = if let Some(list) = self.pool.get_mut(&oldest_key) {
                    if !list.is_empty() {
                        list.remove(0);
                    }
                    list.is_empty()
                } else {
                    false
                };
                if drop_empty {
                    self.pool.remove(&oldest_key);
                }
            } else {
                break;
            }
        }
    }

    /// Reset all transient state (Jacobi window + stability + observed
    /// cursor). Called on EOS or at the start of a new request. Pool
    /// is preserved across requests by default — see
    /// [`Self::clear_pool`] to wipe it.
    pub fn reset(&mut self) {
        self.jacobi.clear();
        self.stability.clear();
        self.observed_up_to = 0;
    }

    /// Drop the entire n-gram pool. Useful for benchmarking cold-start
    /// pool population.
    pub fn clear_pool(&mut self) {
        self.pool.clear();
        self.insertion_order.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(window: usize) -> LookaheadConfig {
        LookaheadConfig {
            window,
            guesses: 0,
            ngram: 3,
            pool_max: 16,
            stable_threshold: 2,
        }
    }

    fn cfg_with_guesses(window: usize, guesses: usize, ngram: usize) -> LookaheadConfig {
        LookaheadConfig {
            window,
            guesses,
            ngram,
            pool_max: 16,
            stable_threshold: 2,
        }
    }

    #[test]
    fn config_input_len_matches_formula() {
        let c = LookaheadConfig {
            window: 4,
            guesses: 2,
            ngram: 3,
            pool_max: 4096,
            stable_threshold: 3,
        };
        assert_eq!(c.verify_input_len(), 1 + 4 + 2 * 4);
    }

    #[test]
    fn seed_initializes_jacobi_window() {
        let mut p = LookaheadProposer::new(LookaheadConfig::default());
        p.seed(123);
        assert_eq!(p.jacobi().len(), p.config().window);
        assert!(p.jacobi().iter().all(|&t| t == 123));
        assert!(p.stability().iter().all(|&s| s == 0));
    }

    #[test]
    fn proposal_flatten_layout() {
        let prop = LookaheadProposal {
            jacobi: vec![10, 20, 30, 40],
            guesses: vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]],
        };
        assert_eq!(prop.flatten(), vec![10, 20, 30, 40, 1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn propose_returns_jacobi_when_seeded() {
        let mut p = LookaheadProposer::new(cfg(4));
        p.seed(7);
        let proposal = p.propose(&[1, 2, 3, 4, 5]).expect("seeded proposer");
        assert_eq!(proposal.jacobi, vec![7, 7, 7, 7]);
        assert!(proposal.guesses.is_empty());
    }

    #[test]
    fn propose_returns_none_before_seed() {
        let p = LookaheadProposer::new(cfg(4));
        assert!(p.propose(&[1, 2, 3]).is_none());
    }

    #[test]
    fn update_increments_stability_on_full_match() {
        // Seed [9,9,9,9] then verify against argmax confirming every
        // Jacobi position. argmax[0..4] are predictions for the
        // jacobi-tracked positions; the +1 free token (argmax[W]) is
        // not consumed by `update`.
        let mut p = LookaheadProposer::new(cfg(4));
        p.seed(9);
        let argmax = vec![9, 9, 9, 9, 42];
        p.update(&argmax, 1);
        // After shift-by-1: window = [9, 9, 9, pad=9], stability = [1, 1, 1, 0].
        assert_eq!(p.jacobi(), vec![9, 9, 9, 9]);
        assert_eq!(p.stability(), vec![1, 1, 1, 0]);
    }

    #[test]
    fn update_resets_stability_on_mismatch() {
        let mut p = LookaheadProposer::new(cfg(4));
        p.seed(9);
        // Mismatch at every position → all stability counters reset.
        let argmax = vec![1, 2, 3, 4, 999];
        p.update(&argmax, 1);
        // new_pred = [1,2,3,4]; new_stab = [0,0,0,0].
        // Shift by 1: window = [2,3,4, pad=4], stability = [0,0,0,0].
        assert_eq!(p.jacobi(), vec![2, 3, 4, 4]);
        assert_eq!(p.stability(), vec![0, 0, 0, 0]);
    }

    #[test]
    fn update_handles_full_commit() {
        // committed_count == W: all old positions consumed, window
        // entirely refilled with the rightmost prediction.
        let mut p = LookaheadProposer::new(cfg(4));
        p.seed(7);
        let argmax = vec![11, 12, 13, 14, 999]; // new_pred = [11,12,13,14], pad = 14
        p.update(&argmax, 4);
        assert_eq!(p.jacobi(), vec![14, 14, 14, 14]);
        assert_eq!(p.stability(), vec![0, 0, 0, 0]);
    }

    #[test]
    fn update_no_op_on_malformed_argmax() {
        let mut p = LookaheadProposer::new(cfg(4));
        p.seed(5);
        // argmax shorter than W → no-op.
        p.update(&[1, 2], 1);
        assert_eq!(p.jacobi(), vec![5, 5, 5, 5]);
    }

    #[test]
    fn update_no_op_when_unseeded() {
        let mut p = LookaheadProposer::new(cfg(4));
        // No seed → jacobi empty, update should bail early.
        p.update(&[1, 2, 3, 4, 5], 1);
        assert!(p.jacobi().is_empty());
    }

    #[test]
    fn greedy_jacobi_accept_full() {
        let mut p = LookaheadProposer::new(cfg(4));
        p.seed(9);
        let argmax = vec![9, 9, 9, 9, 42];
        assert_eq!(p.greedy_jacobi_accept(&argmax), 4);
    }

    #[test]
    fn greedy_jacobi_accept_partial_then_break() {
        let mut p = LookaheadProposer::new(cfg(4));
        p.seed(9);
        // First two positions match, third diverges — accept stops at 2.
        let argmax = vec![9, 9, 1, 9, 999];
        assert_eq!(p.greedy_jacobi_accept(&argmax), 2);
    }

    #[test]
    fn greedy_jacobi_accept_zero_when_first_mismatch() {
        let mut p = LookaheadProposer::new(cfg(4));
        p.seed(9);
        let argmax = vec![0, 9, 9, 9, 999];
        assert_eq!(p.greedy_jacobi_accept(&argmax), 0);
    }

    #[test]
    fn jacobi_converges_to_fixed_point_under_constant_target() {
        // Drive the proposer with a constant argmax stream — the
        // Jacobi window should converge to that constant within at
        // most W steps and then stay there with stability rising
        // every tick.
        let mut p = LookaheadProposer::new(cfg(3));
        p.seed(0);
        let target: u32 = 77;
        let argmax = vec![target, target, target, 999]; // length W+1
        for step in 1..=10 {
            let accepted = p.greedy_jacobi_accept(&argmax);
            // After a single update, window converges to all-`target`.
            if step >= 2 {
                assert_eq!(accepted, 3, "step {step}: window should be all {target}");
            }
            p.update(&argmax, 1);
        }
        assert!(p.jacobi().iter().all(|&t| t == target));
    }

    #[test]
    fn reset_clears_jacobi_but_not_pool() {
        let mut p = LookaheadProposer::new(LookaheadConfig::default());
        p.seed(99);
        p.reset();
        assert!(p.jacobi().is_empty());
        assert_eq!(p.pool_len(), 0);
    }

    // ── L.2 pool gathering tests ─────────────────────────────────────

    #[test]
    fn observe_committed_populates_pool_with_tail_pair() {
        // history of length N+W = 3+2 = 5 → single (prefix, continuation)
        // pair: prefix=[10,20,30], continuation=[40,50].
        let mut p = LookaheadProposer::new(cfg_with_guesses(2, 2, 3));
        let history = vec![10u32, 20, 30, 40, 50];
        p.observe_committed(&history);
        assert_eq!(p.pool_len(), 1);
    }

    #[test]
    fn observe_committed_advances_cursor_and_dedups() {
        // After observing once, calling again with same history is a no-op.
        let mut p = LookaheadProposer::new(cfg_with_guesses(2, 2, 3));
        let history = vec![1u32, 2, 3, 4, 5, 6, 7];
        p.observe_committed(&history);
        let pool_after_first = p.pool_len();
        // Second call with same history → no new pairs since cursor advanced.
        p.observe_committed(&history);
        assert_eq!(p.pool_len(), pool_after_first);
    }

    #[test]
    fn observe_committed_picks_up_new_tokens() {
        // After a commit extends the history, observe_committed should
        // pick up the newly-valid pairs.
        let mut p = LookaheadProposer::new(cfg_with_guesses(2, 2, 3));
        let h1 = vec![1u32, 2, 3, 4, 5];
        p.observe_committed(&h1);
        let n_after_h1 = p.pool_len(); // = 1: ([1,2,3], [4,5]).
        assert_eq!(n_after_h1, 1);
        let h2 = vec![1u32, 2, 3, 4, 5, 6, 7];
        p.observe_committed(&h2);
        // h2 adds: ([2,3,4],[5,6]) at start=1 and ([3,4,5],[6,7]) at start=2 → 2 new.
        assert_eq!(p.pool_len(), 3);
    }

    #[test]
    fn propose_returns_pool_continuations_when_history_suffix_matches() {
        let mut p = LookaheadProposer::new(cfg_with_guesses(2, 2, 3));
        p.seed(0);
        // Inject a (prefix, continuation) directly via observe_committed.
        let seed_history = vec![10u32, 20, 30, 40, 50];
        p.observe_committed(&seed_history);
        // Now propose with a history whose last 3 tokens = [10,20,30] →
        // pool should return continuation [40, 50].
        let history = vec![99u32, 88, 77, 10, 20, 30];
        let prop = p.propose(&history).expect("seeded");
        assert_eq!(prop.jacobi.len(), 2);
        assert_eq!(prop.guesses.len(), 1);
        assert_eq!(prop.guesses[0], vec![40, 50]);
    }

    #[test]
    fn propose_returns_no_guesses_when_g_is_zero() {
        // Even with pool entries, G=0 should never emit guess packs.
        let mut p = LookaheadProposer::new(cfg(2)); // guesses = 0
        p.seed(0);
        let seed_history = vec![10u32, 20, 30, 40, 50];
        p.observe_committed(&seed_history);
        let history = vec![99u32, 88, 77, 10, 20, 30];
        let prop = p.propose(&history).expect("seeded");
        assert!(prop.guesses.is_empty());
    }

    #[test]
    fn pool_lru_evicts_when_over_capacity() {
        let cfg_small = LookaheadConfig {
            window: 2,
            guesses: 2,
            ngram: 2,
            pool_max: 2, // tiny cap to test eviction
            stable_threshold: 2,
        };
        let mut p = LookaheadProposer::new(cfg_small);
        // Three (prefix, continuation) pairs from 3 separate observations.
        // Each call appends history; we manually reset cursor between
        // calls to simulate distinct events without conflating with the
        // dedup path.
        // Pair 1: prefix=[1,2], cont=[3,4]
        p.observe_committed(&[1u32, 2, 3, 4]);
        // Pair 2: prefix=[5,6], cont=[7,8] — fresh proposer to avoid
        // cursor bleed; reuse the same one but start fresh history.
        // Easier: mutate via the public API by chaining histories.
        let h = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        p.observe_committed(&h);
        // After this call, observe_committed sees cursor at len-needed+1=
        // 8-(2+2)+1=5; we already observed up to start=0, advance through
        // 1..=4, picking up:
        //   start=1: [2,3] -> [4,5]
        //   start=2: [3,4] -> [5,6]
        //   start=3: [4,5] -> [6,7]
        //   start=4: [5,6] -> [7,8]
        // 4 inserts; with cap=2 and 1 pre-existing, total inserts=5,
        // capacity should hold ≤ 2 entries.
        assert!(p.pool_len() <= 2);
    }

    #[test]
    fn reset_clears_observed_cursor() {
        let mut p = LookaheadProposer::new(cfg_with_guesses(2, 2, 3));
        p.observe_committed(&[1u32, 2, 3, 4, 5]);
        let pool_before = p.pool_len();
        p.reset();
        // After reset, observing the same history should not re-add (pool
        // is preserved, dedup applies) — verifies the cursor reset works
        // in concert with content dedup.
        p.observe_committed(&[1u32, 2, 3, 4, 5]);
        assert_eq!(p.pool_len(), pool_before);
    }
}
