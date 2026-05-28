//! DRY (Don't Repeat Yourself) sampler — ported from llama.cpp.
//!
//! Detects n-gram repetition in the decode trail and applies an
//! exponential logit penalty to tokens that would extend a repeating
//! sequence beyond `allowed_length`. Soft analogue of [`crate::runaway`]:
//! the runaway detector is a hard kill-switch, DRY gives the model a
//! graceful escape route by lowering — not eliminating — the
//! probability of looping tokens.
//!
//! Pipeline position: applied INSIDE [`crate::sampling::sample_from_logits`]
//! after the classic repeat penalty and before temperature scaling.
//! Operates entirely on a CPU `&mut [f32]` slice.
//!
//! Algorithm (mirrors `llama_sampler_dry_apply` at
//! ggml-org/llama.cpp:src/llama-sampler.cpp:2930-3140):
//!
//! 1. Use a reverse Z-algorithm to compute, for every position `i` in
//!    the trailing window, the longest suffix that *also* occurs
//!    starting at `i`. Linear in the window length.
//! 2. For each position, look up the token that *follows* the matched
//!    suffix in history; that's the token that would extend the
//!    repetition. Record the max match length per such token.
//! 3. Apply `penalty = multiplier × base^(max_repeat − allowed_length)`
//!    to the logit of every such token. Logit penalty is subtracted
//!    (i.e. probability shifts away), not zeroed out — the model can
//!    still escape if the alternative is severely worse.
//!
//! References:
//! - Original DRY proposal: oobabooga/text-generation-webui PR #5677
//! - llama.cpp port: ggml-org/llama.cpp PR #6839 (l3utterfly, 2024-04)
//! - Z-algorithm reference: <https://ivanyu.me/blog/2014/10/15/z-algorithm/>

use std::collections::HashMap;

/// Tuning knobs for one DRY application. All defaults match llama.cpp
/// (multiplier=0 → off by default; `allowed_length=2` permits natural
/// short repetitions like bullet markers or "a a a").
#[derive(Debug, Clone, Copy)]
pub struct DryConfig {
    /// Penalty multiplier. `0.0` disables DRY entirely.
    pub multiplier: f32,
    /// Penalty base (>= 1.0). Higher = more aggressive on long matches.
    pub base: f32,
    /// Repetitions of length `<= allowed_length` are NOT penalised.
    /// llama.cpp default = 2 (allows `A B A B` natural rhythm).
    pub allowed_length: usize,
    /// Trailing window length. `0` disables DRY; `usize::MAX` = entire
    /// history.
    pub penalty_last_n: usize,
}

impl Default for DryConfig {
    fn default() -> Self {
        Self {
            multiplier: 0.0,
            base: 1.75,
            allowed_length: 2,
            penalty_last_n: usize::MAX,
        }
    }
}

impl DryConfig {
    pub fn is_disabled(&self) -> bool {
        self.multiplier == 0.0 || self.base < 1.0 || self.penalty_last_n == 0
    }
}

/// Apply DRY penalty to `logits` in place.
///
/// `recent_tokens` is the full decode trail (prompt tail + everything
/// emitted so far). Only the last `cfg.penalty_last_n` tokens are
/// scanned for n-gram matches.
///
/// No allocation when DRY is disabled. Otherwise allocates one
/// `Vec<i32>` of size `min(last_n, recent.len())` for the Z-array and
/// one `HashMap` keyed by candidate-token id (typically <50 entries).
pub fn apply_dry_penalty(logits: &mut [f32], recent_tokens: &[u32], cfg: &DryConfig) {
    if cfg.is_disabled() {
        return;
    }

    let last_n = cfg.penalty_last_n.min(recent_tokens.len());
    if last_n <= cfg.allowed_length {
        return;
    }
    let window_start = recent_tokens.len() - last_n;
    let window = &recent_tokens[window_start..];

    // Reverse-Z computation: for each k ∈ [1, last_n), dry_repeat[last-k]
    // is the length of the longest match between window[k..] and the
    // trailing suffix window[0..]. By indexing in reverse we get the
    // same effect as llama.cpp's `last_tokens.rat(i)`.
    let n = last_n;
    let last = n.saturating_sub(1);
    let mut z = vec![0i32; n];
    let mut lt = 0usize;
    let mut rt = 0usize;
    let rat = |i: usize| -> u32 { window[n - 1 - i] };

    for k in 1..n {
        if k > rt {
            let mut nn = 0usize;
            while nn + k < n && rat(nn) == rat(nn + k) {
                nn += 1;
            }
            z[last - k] = nn as i32;
            if nn > 0 {
                lt = k;
                rt = k + nn - 1;
            }
        } else {
            let p = k - lt;
            let right_part_remaining = rt - k + 1;
            if (z[last - p] as usize) < right_part_remaining {
                z[last - k] = z[last - p];
            } else {
                let mut nn = right_part_remaining;
                while nn + k < n && rat(nn) == rat(nn + k) {
                    nn += 1;
                }
                z[last - k] = nn as i32;
                lt = k;
                rt = k + nn - 1;
            }
        }
    }

    // For each position with a match of length L, the token that would
    // extend the match is `window[position - 1]` (the token just before
    // the matched suffix in history). Track the max L per such token.
    let mut max_repeat: HashMap<u32, i32> = HashMap::new();
    for i in 0..(n.saturating_sub(1)) {
        let repeat_len = z[i];
        if repeat_len < cfg.allowed_length as i32 + 1 {
            continue;
        }
        // llama.cpp records `rat(last_n_repeat - 2 - i)` as the token
        // that would extend the match. With `rat(j) = window[n-1-j]`,
        // that's `window[n - 1 - (n - 2 - i)] = window[i + 1]` in
        // forward indexing.
        let extending_token = window[i + 1];
        let entry = max_repeat.entry(extending_token).or_insert(0);
        if repeat_len > *entry {
            *entry = repeat_len;
        }
    }

    // Clamp the exponent so `base^exp` never overflows. log(FLT_MAX)
    // ≈ 88.7228 in natural log; divide by ln(base) for safety.
    let max_exponent = if cfg.base > 1.000_001 {
        (88.7228_f32 / cfg.base.ln()).floor() as i32
    } else {
        0
    };
    let allowed = cfg.allowed_length as i32;

    for (token_id, repeat_len) in &max_repeat {
        let idx = *token_id as usize;
        if idx >= logits.len() {
            continue;
        }
        let mut exp = repeat_len - allowed;
        if max_exponent > 0 && exp > max_exponent {
            exp = max_exponent;
        }
        let penalty = cfg.multiplier * cfg.base.powi(exp);
        logits[idx] -= penalty;
    }
}

/// Read DRY config from environment. All four knobs are off by default
/// to keep ship behaviour unchanged unless explicitly enabled.
pub fn dry_config_from_env() -> DryConfig {
    let multiplier = std::env::var("LUMEN_DRY_MULTIPLIER")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .unwrap_or(0.0);
    let base = std::env::var("LUMEN_DRY_BASE")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .unwrap_or(1.75);
    let allowed_length = std::env::var("LUMEN_DRY_ALLOWED_LENGTH")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(2);
    let penalty_last_n = std::env::var("LUMEN_DRY_PENALTY_LAST_N")
        .ok()
        .and_then(|s| {
            let t = s.trim();
            if t == "-1" {
                Some(usize::MAX)
            } else {
                t.parse::<usize>().ok()
            }
        })
        .unwrap_or(usize::MAX);
    DryConfig {
        multiplier,
        base,
        allowed_length,
        penalty_last_n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(mult: f32) -> DryConfig {
        DryConfig {
            multiplier: mult,
            base: 1.75,
            allowed_length: 2,
            penalty_last_n: usize::MAX,
        }
    }

    #[test]
    fn disabled_is_noop() {
        let mut logits = vec![1.0_f32; 10];
        let recent = vec![1u32, 2, 1, 2, 1, 2, 1, 2];
        apply_dry_penalty(&mut logits, &recent, &cfg(0.0));
        assert!(logits.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn short_repetition_under_allowed_length_passes() {
        // "A B A B" — match length 2 == allowed_length, no penalty.
        let mut logits = vec![1.0_f32; 10];
        let recent = vec![1u32, 2, 1, 2];
        apply_dry_penalty(&mut logits, &recent, &cfg(1.0));
        assert!(logits.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn long_cycle_penalizes_extending_token() {
        // "1 2 1 2 1 2 1 2" — trailing suffix "1 2" repeats 4×. The
        // token that would extend the cycle (= 1) should be penalised.
        let mut logits = vec![5.0_f32; 10];
        let recent = vec![1u32, 2, 1, 2, 1, 2, 1, 2];
        apply_dry_penalty(&mut logits, &recent, &cfg(1.0));
        assert!(
            logits[1] < 5.0,
            "extending token 1 should be penalised, got {}",
            logits[1]
        );
        // Unrelated token untouched.
        assert!((logits[3] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn longer_match_gets_stronger_penalty() {
        let mut a = vec![5.0_f32; 10];
        let mut b = vec![5.0_f32; 10];
        let short = vec![1u32, 2, 1, 2, 1, 2]; // 3 matches
        let long = vec![1u32, 2, 1, 2, 1, 2, 1, 2, 1, 2]; // 5 matches
        apply_dry_penalty(&mut a, &short, &cfg(1.0));
        apply_dry_penalty(&mut b, &long, &cfg(1.0));
        assert!(
            b[1] < a[1],
            "longer cycle should produce stronger penalty: short={} long={}",
            a[1],
            b[1]
        );
    }

    #[test]
    fn out_of_vocab_token_does_not_panic() {
        let mut logits = vec![1.0_f32; 4];
        let recent = vec![999u32, 998, 999, 998, 999, 998]; // ids > vocab
        apply_dry_penalty(&mut logits, &recent, &cfg(1.0));
        // No panic; logits unchanged because no token id < 4 was a match.
        assert!(logits.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn varied_sequence_no_penalty() {
        let mut logits = vec![1.0_f32; 100];
        let recent: Vec<u32> = (0..50).collect();
        apply_dry_penalty(&mut logits, &recent, &cfg(1.0));
        assert!(logits.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }
}
