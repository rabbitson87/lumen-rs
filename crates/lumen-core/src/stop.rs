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
        }
    }

    /// No stops configured — the matcher is a pass-through.
    pub fn is_inert(&self) -> bool {
        self.stops.is_empty()
    }

    /// Feed the next decoded piece. Returns the safe-to-emit prefix and the
    /// stop flag.
    pub fn push(&mut self, piece: &str) -> StopStep {
        if self.stops.is_empty() {
            return StopStep {
                emit: piece.to_string(),
                stopped: false,
            };
        }
        self.buf.push_str(piece);

        // 1) Full stop: emit everything before the earliest match, drop the rest.
        let mut hit: Option<usize> = None;
        for s in &self.stops {
            if let Some(idx) = self.buf.find(s.as_str()) {
                hit = Some(hit.map_or(idx, |b| b.min(idx)));
            }
        }
        if let Some(idx) = hit {
            let emit = self.buf[..idx].to_string();
            self.buf.clear();
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
}
