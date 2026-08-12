//! Decode-loop runaway detector.
//!
//! Catches two pathological generation patterns that escape EOS guards:
//!
//! 1. **Single-token repeat** — same token emitted N times in a row
//!    (e.g. `the the the the ...`). Common quantization failure.
//!
//! 2. **N-gram cycle** — short cycle (`A B A B A B ...` or
//!    `당신의 선택은? 당신의 선택은? ...`) within a trailing window.
//!    Caught by counting occurrences of the trailing n-gram in the
//!    last K tokens.
//!
//! Default thresholds are loose enough that legitimate uses (poetry,
//! bulleted lists, tabular output) pass cleanly; they trip only on
//! degenerate loops. Tune via env vars at server start.

/// Detect repetition patterns in the recent decode trail.
///
/// Cheap (no allocation, O(window) per call) so the decode loop can
/// invoke it after every step. Construct once at the top of the
/// generate function and reuse across steps.
pub struct RunawayDetector {
    enabled: bool,
    /// Same token emitted this many times consecutively → trip.
    max_single_repeat: usize,
    /// Trailing window in which to look for n-gram cycles.
    ngram_window: usize,
    /// N-gram size to test (e.g. 4 catches `A B C D A B C D ...`).
    ngram_size: usize,
    /// Number of n-gram occurrences inside the window that constitutes
    /// a cycle. 8 means 8 repeats of the same n-gram in the trailing
    /// window — well outside any legitimate distribution.
    ngram_max_repeats: usize,
}

impl RunawayDetector {
    /// Construct from env. Returns a disabled detector when
    /// `LUMEN_RUNAWAY_DETECT=0` (default is enabled).
    pub fn from_env() -> Self {
        let enabled = std::env::var("LUMEN_RUNAWAY_DETECT")
            .map(|v| {
                let t = v.trim();
                !(t == "0" || t.eq_ignore_ascii_case("false") || t.eq_ignore_ascii_case("off"))
            })
            .unwrap_or(true);
        Self {
            enabled,
            max_single_repeat: env_usize("LUMEN_RUNAWAY_MAX_SINGLE_REPEAT", 64),
            ngram_window: env_usize("LUMEN_RUNAWAY_NGRAM_WINDOW", 128),
            ngram_size: env_usize("LUMEN_RUNAWAY_NGRAM", 4),
            ngram_max_repeats: env_usize("LUMEN_RUNAWAY_NGRAM_MAX_REPEATS", 8),
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Check the trailing decode buffer. Returns a static reason
    /// string if a runaway pattern is detected, `None` otherwise.
    pub fn check(&self, recent: &[u32]) -> Option<&'static str> {
        if !self.enabled {
            return None;
        }
        if self.max_single_repeat > 0 && recent.len() >= self.max_single_repeat {
            let last = *recent.last().unwrap();
            let run = recent.iter().rev().take_while(|&&t| t == last).count();
            if run >= self.max_single_repeat {
                return Some("single-token repeat");
            }
        }
        let n = self.ngram_size;
        if n > 0 && self.ngram_max_repeats > 0 && recent.len() >= n * self.ngram_max_repeats {
            let window_start = recent.len().saturating_sub(self.ngram_window);
            let window = &recent[window_start..];
            if window.len() >= n {
                let tail = &window[window.len() - n..];
                let mut count = 0usize;
                let mut i = 0usize;
                while i + n <= window.len() {
                    if &window[i..i + n] == tail {
                        count += 1;
                        if count >= self.ngram_max_repeats {
                            return Some("n-gram cycle");
                        }
                    }
                    i += 1;
                }
            }
        }
        None
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detector(max_single: usize, window: usize, n: usize, max_rep: usize) -> RunawayDetector {
        RunawayDetector {
            enabled: true,
            max_single_repeat: max_single,
            ngram_window: window,
            ngram_size: n,
            ngram_max_repeats: max_rep,
        }
    }

    #[test]
    fn single_token_repeat_trips() {
        let d = detector(8, 64, 4, 8);
        let tokens = vec![5u32; 8];
        assert_eq!(d.check(&tokens), Some("single-token repeat"));
    }

    #[test]
    fn single_token_repeat_below_threshold_passes() {
        let d = detector(8, 64, 4, 8);
        let tokens = vec![5u32; 7];
        assert_eq!(d.check(&tokens), None);
    }

    #[test]
    fn ngram_cycle_trips() {
        let d = detector(64, 64, 2, 4);
        // 4 repeats of [1, 2] = "1 2 1 2 1 2 1 2"
        let tokens = vec![1u32, 2, 1, 2, 1, 2, 1, 2];
        assert_eq!(d.check(&tokens), Some("n-gram cycle"));
    }

    #[test]
    fn varied_sequence_passes() {
        let d = detector(8, 64, 4, 8);
        let tokens: Vec<u32> = (0..100).collect();
        assert_eq!(d.check(&tokens), None);
    }

    #[test]
    fn disabled_returns_none() {
        let mut d = detector(2, 8, 2, 2);
        d.enabled = false;
        let tokens = vec![7u32; 10];
        assert_eq!(d.check(&tokens), None);
    }

    #[test]
    fn legitimate_short_repetition_passes() {
        // A bulleted list might emit the same bullet token several
        // times; with default max_single_repeat=64 a run of 10
        // identical tokens is fine.
        let d = detector(64, 128, 4, 8);
        let mut tokens = vec![1u32, 2, 3];
        tokens.extend(std::iter::repeat_n(99u32, 10));
        tokens.extend(vec![4u32, 5, 6]);
        assert_eq!(d.check(&tokens), None);
    }
}
