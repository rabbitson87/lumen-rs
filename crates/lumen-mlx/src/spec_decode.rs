//! Track A2.4 — N-gram speculative decode for the MLX backend.
//!
//! Greedy K=2 spec decode using a rolling trigram (n=3) table built from the
//! seq's own committed history. Drift-aware: the L.3 / A2.0 lesson that
//! batched-forward (S=2) row-0 logits can differ from single-step (S=1) is
//! accepted as a relaxed bit-identical gate — we measure cosine + top-1 match
//! against baseline rather than strict identity.
//!
//! Algorithm (per spec attempt at state offset M, last_committed = T):
//!
//!   1. Look up trigram(T, history[-1]) → top next-token = d_0.
//!      (When the lookup misses, fall through to a normal decode_step.)
//!   2. If d_0 ≠ T_pred (the target's pre-known prediction at offset M),
//!      no spec attempt: commit T_pred via decode_step.
//!   3. Otherwise look up trigram(history[-1], d_0) → top next = d_1.
//!      If miss, only K=1 spec (commit d_0 + decode_step's next).
//!   4. Snapshot state at offset M.
//!   5. forward_probe([d_0, d_1]) → logits at offsets M+1, M+2.
//!      - row 0 argmax = target's pred for offset M+1 conditioned on cache+d_0.
//!      - row 1 argmax = target's pred for offset M+2 conditioned on cache+d_0+d_1.
//!   6. Acceptance:
//!      - If row 0 == d_1: full accept. Commit [d_0, d_1, row 1 argmax] = 3 tokens.
//!        State already at M+2 (forward consumed d_0,d_1). Next decode starts
//!        from row 1 argmax which we'll feed via decode_step.
//!        -- wait, that overcounts. State at M+2 has d_0..d_1 in cache;
//!        committing 3 tokens means logical sequence advances by 3 (d_0,d_1, +1 corrective).
//!        We need to feed the corrective via decode_step → state M+3, returns next T_pred.
//!      - If row 0 != d_1: partial accept. Commit [d_0, row 0 argmax] = 2 tokens.
//!        Cache currently at M+2 with d_0,d_1; need to roll back to M+1 (cache
//!        has only d_0). Easiest: restore to M, re-feed d_0 via extend.
//!        Then feed corrective via decode_step → state M+2, returns next T_pred.
//!   7. Release / restore the snapshot accordingly.
//!
//! Wins vs single-step:
//!   - Full accept (3 tokens / forward S=2 + decode_step S=1): ~3× per attempt
//!   - Partial accept K=1 (2 tokens / S=2 + decode_step + restore + extend(1) + decode_step): wash to mild loss
//!   - No spec (T_pred ≠ d_0): 1 token / decode_step (=baseline)
//!
//! Env vars:
//!   LUMEN_MLX_SPEC=ngram          enable
//!   LUMEN_MLX_SPEC_K=2            (ignored, K=2 hardcoded for this MVP)
//!   LUMEN_MLX_SPEC_N=3            n-gram order (default 3)

use std::collections::HashMap;

/// Rolling n-gram (n=3 default) table over u32 token ids. Counts next-token
/// frequencies under each (n-1)-gram context. `propose` returns the most-
/// common continuation.
pub(crate) struct NgramTable {
    /// (a, b) → {c: count}. n=3 only for now.
    trigram: HashMap<(u32, u32), HashMap<u32, u32>>,
    n: usize,
}

impl NgramTable {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            trigram: HashMap::new(),
            n: n.max(2),
        }
    }

    /// Update the table after committing `tok` to a sequence whose previous
    /// tokens are in `history`. Adds a single observation to the most recent
    /// n-1 context.
    pub(crate) fn observe(&mut self, history: &[u32], tok: u32) {
        if history.len() < self.n - 1 {
            return;
        }
        let ctx = (history[history.len() - 2], history[history.len() - 1]);
        *self.trigram.entry(ctx).or_default().entry(tok).or_insert(0) += 1;
    }

    /// Propose K next tokens given history. Returns up to K tokens, stopping
    /// at the first context miss. Greedy mode: picks the highest-count
    /// continuation at each step.
    pub(crate) fn propose(&self, history: &[u32], k: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(k);
        if history.len() < self.n - 1 {
            return out;
        }
        let mut a = history[history.len() - 2];
        let mut b = history[history.len() - 1];
        for _ in 0..k {
            let ctx = (a, b);
            let counts = match self.trigram.get(&ctx) {
                Some(c) if !c.is_empty() => c,
                _ => break,
            };
            // Greedy: pick max-count token (break ties by smaller token id).
            let mut best: Option<(u32, u32)> = None;
            for (&tok, &cnt) in counts {
                match best {
                    None => best = Some((tok, cnt)),
                    Some((t0, c0)) if cnt > c0 || (cnt == c0 && tok < t0) => {
                        best = Some((tok, cnt));
                    }
                    _ => {}
                }
            }
            let pick = best.expect("non-empty checked above").0;
            out.push(pick);
            a = b;
            b = pick;
        }
        out
    }
}

/// Counters for one spec-decode session, written to log on completion.
#[derive(Default, Debug, Clone)]
pub(crate) struct SpecStats {
    pub attempts: u64,
    pub fallthrough_no_proposal: u64,
    pub fallthrough_d0_mismatch: u64,
    pub partial_accept: u64,
    pub full_accept: u64,
    pub committed_via_spec: u64,
    pub committed_via_baseline: u64,
}

impl SpecStats {
    pub(crate) fn log_summary(&self, label: &str) {
        let total = self.committed_via_spec + self.committed_via_baseline;
        let speedup = if self.attempts > 0 {
            (self.committed_via_spec as f64 + self.committed_via_baseline as f64)
                / ((self.attempts + self.committed_via_baseline) as f64).max(1.0)
        } else {
            1.0
        };
        eprintln!(
            "[mlx-spec] {label} attempts={} no_prop={} d0_miss={} partial={} full={} \
             tokens={} (spec={} base={}) committed_per_forward≈{:.2}",
            self.attempts,
            self.fallthrough_no_proposal,
            self.fallthrough_d0_mismatch,
            self.partial_accept,
            self.full_accept,
            total,
            self.committed_via_spec,
            self.committed_via_baseline,
            speedup,
        );
    }
}

/// Resolve env-driven spec config. Returns `None` when `LUMEN_MLX_SPEC` is
/// unset or anything other than `ngram`.
#[derive(Clone, Debug)]
pub(crate) struct SpecConfig {
    pub k: usize,
    pub n: usize,
}

pub(crate) fn read_spec_config() -> Option<SpecConfig> {
    let kind = std::env::var("LUMEN_MLX_SPEC").ok()?;
    if kind.trim() != "ngram" {
        return None;
    }
    let k: usize = std::env::var("LUMEN_MLX_SPEC_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let n: usize = std::env::var("LUMEN_MLX_SPEC_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    Some(SpecConfig {
        k: k.max(2),
        n: n.max(2),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_observes_and_proposes_top_continuation() {
        let mut t = NgramTable::new(3);
        // Train: "a b c" three times, "a b d" once → propose after [a,b] = c
        for _ in 0..3 {
            t.observe(&[10, 11], 12);
        }
        t.observe(&[10, 11], 13);
        let p = t.propose(&[10, 11], 2);
        assert_eq!(p[0], 12);
    }

    #[test]
    fn ngram_propose_chains_k_tokens() {
        let mut t = NgramTable::new(3);
        t.observe(&[1, 2], 3);
        t.observe(&[2, 3], 4);
        let p = t.propose(&[1, 2], 2);
        assert_eq!(p, vec![3, 4]);
    }

    #[test]
    fn ngram_propose_stops_on_miss() {
        let mut t = NgramTable::new(3);
        t.observe(&[1, 2], 3);
        // (2,3) → not seen
        let p = t.propose(&[1, 2], 3);
        assert_eq!(p, vec![3]); // stops before next step
    }

    #[test]
    fn ngram_returns_empty_when_history_too_short() {
        let t = NgramTable::new(3);
        let p = t.propose(&[5], 2);
        assert!(p.is_empty());
    }

    #[test]
    fn spec_config_defaults_off() {
        // Don't set env; check None.
        // SAFETY: we control env in unit tests but mid-suite mutation is still
        // racy across threads. cargo test runs each test in its own thread,
        // so we read the var rather than mutating it; this asserts the parser
        // logic when LUMEN_MLX_SPEC happens to be unset in this run.
        if std::env::var("LUMEN_MLX_SPEC").is_err() {
            assert!(read_spec_config().is_none());
        }
    }
}
