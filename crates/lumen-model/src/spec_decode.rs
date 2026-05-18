//! Speculative decoding helpers (Phase S.1).
//!
//! N-gram speculation produces a draft of K candidate tokens by searching the
//! generation history for the longest matching suffix and proposing the
//! tokens that followed it the last time. The draft is then verified by a
//! single batched forward (K+1 tokens) and accepted prefix-wise — see
//! `crates/lumen-model/src/qwen3_5_moe/backend.rs` for the integration
//! site. This file is intentionally self-contained and host-only so it can
//! be unit-tested without any model state.

/// Look up an N-gram-driven draft from `history`. The lookup tries
/// progressively shorter suffixes (`n` down to `min_n`) and, for each,
/// scans the history right-to-left for the most recent occurrence of that
/// suffix. The `k` tokens following the most recent occurrence become the
/// draft.
///
/// Returns `None` when no match is found or when fewer than `k` tokens
/// follow the match (i.e. the match was at the very tail of the history).
///
/// # Why right-to-left
/// Recent code repeats are far more predictive than ancient ones in
/// autoregressive generation. The first hit walking backward is also the
/// closest in distribution.
///
/// # Parameters
/// - `history`: full token sequence so far (prompt + previously generated).
///   Must include at least `n` tokens.
/// - `n`: maximum suffix length to try. Common values: 3-5.
/// - `min_n`: minimum suffix length the lookup will fall back to. `2` keeps
///   the lookup fairly aggressive; `3` is more conservative.
/// - `k`: number of tokens to draft. Common values: 2-4.
pub fn ngram_lookup(history: &[u32], n: usize, min_n: usize, k: usize) -> Option<Vec<u32>> {
    if k == 0 || n == 0 || min_n == 0 || min_n > n {
        return None;
    }
    let h = history.len();
    if h < n + 1 {
        return None;
    }

    // Try the longest suffix first; fall back to shorter suffixes if no match.
    let max_n = n.min(h - 1);
    for cur_n in (min_n..=max_n).rev() {
        let suffix = &history[h - cur_n..];
        // Scan right-to-left over earlier positions where the suffix could fit and
        // still leave `k` tokens after the match.
        // Match position `i` means history[i..i+cur_n] == suffix.
        // Need i + cur_n + k <= h - cur_n  → at least one full draft slot before
        // the trailing suffix. (Equivalent: i <= h - cur_n - k - cur_n.)
        let last_valid_i = h.saturating_sub(2 * cur_n + k);
        let mut i = last_valid_i;
        loop {
            if &history[i..i + cur_n] == suffix {
                let draft = &history[i + cur_n..i + cur_n + k];
                return Some(draft.to_vec());
            }
            if i == 0 {
                break;
            }
            i -= 1;
        }
    }
    None
}

/// Greedy verify: walk `target_argmax` against `draft` and return the count
/// `j ∈ 0..=draft.len()` of accepted tokens. Acceptance stops at the first
/// mismatch, so the call only inspects the first `min(j+1, K)` positions.
///
/// `target_argmax[i]` is the model's argmax for the position immediately
/// after the i-th draft token (or the verify-batch prefix when i==0). This
/// matches the layout of `forward([prefix, d0, ..., d_{K-1}])` whose i-th
/// output position is conditioned on `[prefix, d_0..d_{i-1}]`.
pub fn greedy_accept_count(target_argmax: &[u32], draft: &[u32]) -> usize {
    let mut accepted = 0;
    for (t, d) in target_argmax.iter().zip(draft.iter()) {
        if t == d {
            accepted += 1;
        } else {
            break;
        }
    }
    accepted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_finds_recent_repeat() {
        // history: ... A B C X Y Z A B C (suffix "A B C") → drafted "X Y" from earlier "A B C X Y"
        let h: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 30, 40, 50];
        // suffix len=3 → "30 40 50" → previous occurrence at index 2 → draft starts at 5: [60, 70]
        let d = ngram_lookup(&h, 3, 2, 2).unwrap();
        assert_eq!(d, vec![60, 70]);
    }

    #[test]
    fn ngram_falls_back_to_shorter_suffix() {
        // No match at length 3 ("9 8 7"), but a length-2 match ("8 7") exists earlier.
        let h: Vec<u32> = vec![1, 8, 7, 99, 88, 9, 8, 7];
        // suffix-3 = [9,8,7] → no earlier match
        // suffix-2 = [8,7] → matches at index 1, followed by [99]
        let d = ngram_lookup(&h, 3, 2, 1).unwrap();
        assert_eq!(d, vec![99]);
    }

    #[test]
    fn ngram_returns_none_when_no_match() {
        let h: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        assert!(ngram_lookup(&h, 3, 2, 2).is_none());
    }

    #[test]
    fn ngram_returns_none_when_history_too_short() {
        let h: Vec<u32> = vec![1, 2];
        assert!(ngram_lookup(&h, 3, 2, 2).is_none());
    }

    #[test]
    fn ngram_picks_most_recent_when_multiple_matches() {
        // suffix [5,6] occurs at indices 0 and 4. Most recent is 4 → draft [9, 10]
        let h: Vec<u32> = vec![5, 6, 100, 200, 5, 6, 9, 10, 5, 6];
        let d = ngram_lookup(&h, 2, 2, 2).unwrap();
        assert_eq!(d, vec![9, 10]);
    }

    #[test]
    fn ngram_skips_match_with_insufficient_followers() {
        // suffix [5,6] at index 0, followed by [100, 200] (len 2). Suffix appears only at
        // index 0; with k=4 there are only 8 - 0 - 2 = 6 tokens after, but the
        // history's tail is itself [5,6] and we need at least k tokens AFTER the
        // candidate match BEFORE the trailing suffix begins. Verify the lookup
        // respects the "no overlap with trailing suffix" constraint.
        let h: Vec<u32> = vec![5, 6, 100, 200, 5, 6];
        // history len 6. suffix len 2 = [5,6]. We require i + 2 + 2 <= 6 - 2 = 4 → i <= 0.
        // history[0..2] == [5,6] = suffix → match at 0, draft = history[2..4] = [100, 200].
        let d = ngram_lookup(&h, 2, 2, 2).unwrap();
        assert_eq!(d, vec![100, 200]);
    }

    #[test]
    fn greedy_accept_all() {
        let target = vec![1, 2, 3, 4];
        let draft = vec![1, 2, 3, 4];
        assert_eq!(greedy_accept_count(&target, &draft), 4);
    }

    #[test]
    fn greedy_accept_partial() {
        let target = vec![1, 2, 99, 4];
        let draft = vec![1, 2, 3, 4];
        assert_eq!(greedy_accept_count(&target, &draft), 2);
    }

    #[test]
    fn greedy_accept_none() {
        let target = vec![99, 2, 3];
        let draft = vec![1, 2, 3];
        assert_eq!(greedy_accept_count(&target, &draft), 0);
    }
}
