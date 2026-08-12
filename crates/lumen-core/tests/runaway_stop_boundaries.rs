//! Short-input and boundary coverage for the runaway detector and the stop
//! buffer (005 Phase 4.1).
//!
//! Both of these run on **every decode step**, and both were only ever tested
//! with inputs long enough to trip them. Their untaken branches were therefore
//! all the "not yet" side — the one taken thousands of times per request.
//!
//! The failure modes are opposite and both bad. A detector that trips early
//! truncates a legitimate answer mid-sentence; one that never trips lets a
//! repetition loop run to `max_tokens`. A stop buffer that splits a multi-byte
//! character emits mojibake into the stream — the `is_char_boundary` walk is
//! the only thing preventing that, and it had never been exercised.

use lumen_core::runaway::RunawayDetector;
use lumen_core::stop::StopMatcher;

// ─────────────────────────── runaway detector ───────────────────────────

/// A history shorter than the thresholds must not trip. This is the common
/// case — every request spends its first dozens of steps here — and tripping
/// early truncates a real answer.
#[test]
fn a_short_history_never_trips() {
    let d = RunawayDetector::from_env();
    assert!(d.enabled(), "the default detector is enabled");

    assert_eq!(d.check(&[]), None);
    assert_eq!(d.check(&[7]), None);

    // A repeating 4-gram, but not enough repeats to reach n * max_repeats.
    let few_cycles: Vec<u32> = std::iter::repeat_n([1u32, 2, 3, 4], 7).flatten().collect();
    assert_eq!(
        d.check(&few_cycles),
        None,
        "7 cycles is under the 8-repeat default"
    );

    // A single-token run under BOTH thresholds. Note which one binds: the
    // n-gram detector sees `7 7 7 7` repeated, so a run of 63 identical tokens
    // trips as an "n-gram cycle" at 32 tokens — long before the 64-repeat
    // single-token threshold it looks like it should hit first. The effective
    // ceiling for a single-token run is therefore
    // `ngram_size * ngram_max_repeats` (4 * 8 = 32), not `max_single_repeat`.
    let short_run: Vec<u32> = std::iter::repeat_n(7u32, 31).collect();
    assert_eq!(
        d.check(&short_run),
        None,
        "31 identical tokens is under both thresholds"
    );
}

/// The two detectors overlap, and which one fires first is worth pinning
/// because it is not the one the constant names suggest. A run of identical
/// tokens is also a repeating n-gram, so `ngram_size * ngram_max_repeats`
/// binds at 32 while `max_single_repeat` sits at 64 — anyone tuning the
/// single-repeat knob upward will find it has no effect until the n-gram knobs
/// move too.
#[test]
fn the_ngram_detector_catches_single_token_runs_before_the_single_repeat_one() {
    let d = RunawayDetector::from_env();
    let run: Vec<u32> = std::iter::repeat_n(7u32, 32).collect();
    assert_eq!(
        d.check(&run),
        Some("n-gram cycle"),
        "32 identical tokens is 8 repeats of a 4-gram, so the n-gram rule binds first"
    );
}

/// And it must still trip when the pattern is real, or the test above would
/// pass against a detector that never fires.
#[test]
fn a_genuine_runaway_still_trips() {
    let d = RunawayDetector::from_env();

    let long_run: Vec<u32> = std::iter::repeat_n(7u32, 64).collect();
    assert_eq!(d.check(&long_run), Some("single-token repeat"));

    let cycles: Vec<u32> = std::iter::repeat_n([1u32, 2, 3, 4], 12).flatten().collect();
    assert!(
        d.check(&cycles).is_some(),
        "12 repeats of a 4-gram is a cycle"
    );
}

/// Varied text of the same length must not trip — the length threshold is
/// necessary but not sufficient, and a detector keyed only on length would
/// truncate every long answer.
#[test]
fn length_alone_does_not_trip_the_detector() {
    let d = RunawayDetector::from_env();
    let varied: Vec<u32> = (0..200u32).collect();
    assert_eq!(d.check(&varied), None);

    // A long run that ends in something else: the trailing run is what counts.
    let mut nearly: Vec<u32> = std::iter::repeat_n(7u32, 200).collect();
    *nearly.last_mut().unwrap() = 9;
    assert_eq!(
        d.check(&nearly),
        None,
        "the run is measured from the tail, so one differing final token clears it"
    );
}

// ─────────────────────────── stop buffer ───────────────────────────

/// The `is_char_boundary` walk is the only thing keeping a multi-byte character
/// from being split across two emitted pieces. Reaching it needs a partial stop
/// overlap that lands inside a character — so the stop string and the text must
/// share a multi-byte prefix.
#[test]
fn a_partial_overlap_never_splits_a_multibyte_character() {
    let mut b = StopMatcher::new(vec!["종료".to_string()]);
    // "종" is three UTF-8 bytes and is a strict prefix of the stop string, so
    // the buffer must hold it rather than emit two bytes of it.
    let mut emitted = String::new();
    for piece in ["안녕", "종", "하세요"] {
        let step = b.push(piece);
        emitted.push_str(&step.emit);
        if step.stopped {
            break;
        }
    }
    emitted.push_str(&b.flush());
    assert_eq!(
        emitted, "안녕종하세요",
        "the text must round-trip byte-for-byte through the hold"
    );
    assert!(
        emitted.chars().all(|c| c != '\u{FFFD}'),
        "no replacement characters — a split multi-byte char is mojibake in the stream"
    );
}

/// The same buffer must still stop when the sequence actually completes.
#[test]
fn a_multibyte_stop_string_still_stops() {
    let mut b = StopMatcher::new(vec!["종료".to_string()]);
    let mut out = String::new();
    for piece in ["안녕", "종", "료", "뒤"] {
        let step = b.push(piece);
        out.push_str(&step.emit);
        if step.stopped {
            break;
        }
    }
    assert_eq!(out, "안녕", "everything before the stop, and nothing after");
}

/// An inert buffer (no stop strings) is the common case and must pass text
/// through unchanged, including multi-byte pieces.
#[test]
fn an_inert_buffer_passes_multibyte_text_through() {
    let mut b = StopMatcher::new(vec![]);
    assert!(b.is_inert());
    let mut out = String::new();
    for piece in ["안", "녕", "하세요", "🎉"] {
        out.push_str(&b.push(piece).emit);
    }
    out.push_str(&b.flush());
    assert_eq!(out, "안녕하세요🎉");
}
