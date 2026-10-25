//! Incremental stop-string matching for streaming generation.
//!
//! Mirrors llama.cpp's server stop handling: emit visible text as it streams,
//! but (1) halt + trim when a full stop string appears, and (2) hold back any
//! trailing fragment that could still grow into a stop string, so a partial
//! match is never streamed to the client only to be retracted.

/// Streaming stop-string matcher. Feed decoded text pieces in order; it
/// returns the prefix that is safe to emit and whether a full stop hit.
#[derive(Debug, Default)]
pub struct StopMatcher {
    stops: Vec<String>,
    /// Bytes seen but not yet emitted (a possible partial-stop tail).
    buf: String,
    /// The stop string that ended the stream, once one has.
    matched: Option<String>,
}

/// Outcome of feeding one piece to a [`StopMatcher`].
#[derive(Debug, PartialEq, Eq)]
pub struct StopStep {
    /// Text confirmed safe to stream now (may be empty while a tail is held).
    pub emit: String,
    /// True once a full stop string was matched — generation should end and
    /// the stop text itself has been trimmed from `emit`.
    pub stopped: bool,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        // Drop empties up front — an empty stop would "match" everywhere.
        let stops = stops.into_iter().filter(|s| !s.is_empty()).collect();
        Self {
            stops,
            buf: String::new(),
            matched: None,
        }
    }

    /// The stop string the stream was cut at, if one completed. Anthropic
    /// reports it back (`stop_sequence`), so "a stop hit" is not enough.
    pub fn matched(&self) -> Option<&str> {
        self.matched.as_deref()
    }

    /// No stops configured — the matcher is a pass-through.
    pub fn is_inert(&self) -> bool {
        self.stops.is_empty()
    }

    /// Feed the next decoded piece. Returns the safe-to-emit prefix and the
    /// stop flag.
    ///
    /// Terminal once a stop has matched: every later piece emits nothing and
    /// reports `stopped` again. A caller is meant to end decoding at the first
    /// `stopped`, but not every decode loop can be ended from its callback, and
    /// one that keeps going must not stream the text after the stop.
    pub fn push(&mut self, piece: &str) -> StopStep {
        if self.matched.is_some() {
            return StopStep {
                emit: String::new(),
                stopped: true,
            };
        }
        if self.stops.is_empty() {
            return StopStep {
                emit: piece.to_string(),
                stopped: false,
            };
        }
        self.buf.push_str(piece);

        // 1) Full stop: emit everything before the earliest match, drop the rest.
        //    Two stops matching at the same byte go to the one listed first —
        //    the same rule as the non-streaming trim, so both name the same one.
        let mut hit: Option<(usize, usize)> = None;
        for (k, s) in self.stops.iter().enumerate() {
            if let Some(idx) = self.buf.find(s.as_str()) {
                if hit.is_none_or(|(b, _)| idx < b) {
                    hit = Some((idx, k));
                }
            }
        }
        if let Some((idx, k)) = hit {
            let emit = self.buf[..idx].to_string();
            self.buf.clear();
            self.matched = Some(self.stops[k].clone());
            return StopStep {
                emit,
                stopped: true,
            };
        }

        // 2) Partial overlap: hold the longest suffix of `buf` that is a strict
        //    prefix of some stop string (it might still complete next piece).
        let mut hold = self.longest_partial_overlap();
        // Never split a multi-byte char.
        let mut split = self.buf.len() - hold;
        while split < self.buf.len() && !self.buf.is_char_boundary(split) {
            split += 1;
        }
        hold = self.buf.len() - split;
        let _ = hold;
        let emit = self.buf[..split].to_string();
        self.buf = self.buf[split..].to_string();
        StopStep {
            emit,
            stopped: false,
        }
    }

    /// At end of generation, release any held tail (no stop completed).
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.buf)
    }

    /// Longest L (1..) such that `buf` ends with the first L bytes of some
    /// stop string AND L < that stop's length (a *partial*, not full, match).
    fn longest_partial_overlap(&self) -> usize {
        let bytes = self.buf.as_bytes();
        let mut best = 0;
        for s in &self.stops {
            let sb = s.as_bytes();
            let max_l = (sb.len() - 1).min(bytes.len());
            for l in (best + 1..=max_l).rev() {
                if bytes.ends_with(&sb[..l]) {
                    best = best.max(l);
                    break;
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(stops: &[&str], pieces: &[&str]) -> (String, bool) {
        let mut m = StopMatcher::new(stops.iter().map(|s| s.to_string()).collect());
        let mut out = String::new();
        let mut stopped = false;
        for p in pieces {
            let step = m.push(p);
            out.push_str(&step.emit);
            if step.stopped {
                stopped = true;
                break;
            }
        }
        if !stopped {
            out.push_str(&m.flush());
        }
        (out, stopped)
    }

    #[test]
    fn no_stops_passes_through() {
        assert_eq!(run(&[], &["a", "bc", "d"]), ("abcd".into(), false));
    }

    #[test]
    fn full_stop_trims_and_halts() {
        // "STOP" appears mid-stream → emit "hello ", halt, drop the rest.
        assert_eq!(
            run(&["STOP"], &["hello ", "STO", "P world"]),
            ("hello ".into(), true)
        );
    }

    #[test]
    fn partial_tail_is_held_then_released() {
        // Ends with "ST" which is a prefix of "STOP" but never completes.
        assert_eq!(run(&["STOP"], &["hi ", "ST"]), ("hi ST".into(), false));
    }

    #[test]
    fn partial_tail_held_not_emitted_midstream() {
        // After "ST" the matcher must NOT have emitted "ST" yet.
        let mut m = StopMatcher::new(vec!["STOP".into()]);
        let a = m.push("hi ");
        assert_eq!(a.emit, "hi ");
        let b = m.push("ST");
        assert_eq!(b.emit, ""); // held
        let c = m.push("X"); // diverges → release held "STX"
        assert_eq!(c.emit, "STX");
    }

    #[test]
    fn multibyte_not_split() {
        // "→END" stop; buffer ends mid multi-byte char of an unrelated emoji.
        assert_eq!(run(&["END"], &["café ", "EN"]), ("café EN".into(), false));
    }

    #[test]
    fn earliest_of_multiple_stops() {
        assert_eq!(
            run(&["world", "lo"], &["hello world"]),
            ("hel".into(), true)
        );
    }

    #[test]
    fn names_the_stop_that_matched() {
        let mut m = StopMatcher::new(vec!["world".into(), "lo".into()]);
        assert_eq!(m.matched(), None);
        assert!(m.push("hel").emit == "he" && m.matched().is_none());
        assert!(m.push("lo world").stopped);
        assert_eq!(
            m.matched(),
            Some("lo"),
            "the earliest match, not the first listed"
        );
        // Same start byte: the one listed first, as the non-streaming trim does.
        let mut m = StopMatcher::new(vec!["bc".into(), "b".into()]);
        assert!(m.push("abc").stopped);
        assert_eq!(m.matched(), Some("bc"));
        // Terminal: a decode loop that runs on past the stop streams nothing more.
        let mut m = StopMatcher::new(vec!["STOP".into()]);
        assert!(m.push("a STOP b").stopped);
        let after = m.push(" more text STOP again");
        assert_eq!((after.emit.as_str(), after.stopped), ("", true));
        assert_eq!(m.flush(), "");
        assert_eq!(m.matched(), Some("STOP"));
        // A stream that never completes a stop names none.
        let mut m = StopMatcher::new(vec!["STOP".into()]);
        m.push("ST");
        m.flush();
        assert_eq!(m.matched(), None);
    }
}
