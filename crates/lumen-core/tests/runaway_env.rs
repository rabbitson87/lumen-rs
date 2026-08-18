//! The runaway detector's per-rule off-switches (005 Phase 4.1).
//!
//! Own test binary: `RunawayDetector::from_env` reads five process env vars, so
//! a test that sets them races every sibling that constructs a detector — and
//! there, the failure is silent (a sibling asserting "this trips" would see a
//! disabled detector and its assertion would simply stop meaning anything).
//!
//! Each threshold is individually disabled by `0`, and those four guards were
//! the module's remaining uncovered branches. They matter because this detector
//! **terminates generation**: a rule that cannot be switched off leaves an
//! operator with a truncated answer and no recourse, and one that is off when
//! it should be on lets a repetition loop run to `max_tokens`.

use lumen_core::runaway::RunawayDetector;

const VARS: &[&str] = &[
    "LUMEN_RUNAWAY_DETECT",
    "LUMEN_RUNAWAY_MAX_SINGLE_REPEAT",
    "LUMEN_RUNAWAY_NGRAM",
    "LUMEN_RUNAWAY_NGRAM_MAX_REPEATS",
    "LUMEN_RUNAWAY_NGRAM_WINDOW",
];

struct Restore;
impl Drop for Restore {
    fn drop(&mut self) {
        for v in VARS {
            // SAFETY: single-threaded within this binary.
            unsafe { std::env::remove_var(v) };
        }
    }
}

fn set(k: &str, v: &str) {
    // SAFETY: as above; `from_env` reads on the next construction.
    unsafe { std::env::set_var(k, v) };
}

fn clear_all() {
    for v in VARS {
        // SAFETY: as above.
        unsafe { std::env::remove_var(v) };
    }
}

/// A single-token run and an n-gram cycle, long enough to trip every rule at
/// its default threshold.
fn tripping_input() -> Vec<u32> {
    std::iter::repeat_n(7u32, 200).collect()
}

#[test]
fn every_threshold_can_be_switched_off_individually() {
    let _r = Restore;
    clear_all();

    // Baseline: defaults trip. Without this the "off" assertions below could
    // pass against an input that never tripped anything.
    assert!(
        RunawayDetector::from_env()
            .check(&tripping_input())
            .is_some(),
        "the fixture must trip at default thresholds"
    );

    // The master switch.
    for off in ["0", "false", "FALSE", "off", " off "] {
        clear_all();
        set("LUMEN_RUNAWAY_DETECT", off);
        let d = RunawayDetector::from_env();
        assert!(!d.enabled(), "LUMEN_RUNAWAY_DETECT={off:?} must disable");
        assert_eq!(d.check(&tripping_input()), None);
    }

    // Anything else keeps it on — it is opt-OUT, so a typo must not silently
    // remove the only guard against a `max_tokens` runaway.
    for on in ["1", "true", "yes", "", "nope"] {
        clear_all();
        set("LUMEN_RUNAWAY_DETECT", on);
        assert!(
            RunawayDetector::from_env().enabled(),
            "LUMEN_RUNAWAY_DETECT={on:?} is not an off-value"
        );
    }

    // Rule 1 off, rule 2 still on: a single-token run is also an n-gram cycle,
    // so disabling the single-repeat rule alone must NOT stop the detector.
    clear_all();
    set("LUMEN_RUNAWAY_MAX_SINGLE_REPEAT", "0");
    assert_eq!(
        RunawayDetector::from_env().check(&tripping_input()),
        Some("n-gram cycle"),
        "with the single-repeat rule off, the n-gram rule must still catch it"
    );

    // Rule 2 off (two independent ways), rule 1 still on.
    for (k, v) in [
        ("LUMEN_RUNAWAY_NGRAM", "0"),
        ("LUMEN_RUNAWAY_NGRAM_MAX_REPEATS", "0"),
    ] {
        clear_all();
        set(k, v);
        assert_eq!(
            RunawayDetector::from_env().check(&tripping_input()),
            Some("single-token repeat"),
            "{k}=0 must disable only the n-gram rule"
        );
    }

    // Both rules off: nothing trips, even on the worst possible input.
    clear_all();
    set("LUMEN_RUNAWAY_MAX_SINGLE_REPEAT", "0");
    set("LUMEN_RUNAWAY_NGRAM", "0");
    assert_eq!(
        RunawayDetector::from_env().check(&tripping_input()),
        None,
        "with every rule off the detector must be inert"
    );

    // ── A window shorter than the n-gram ──
    //
    // Folded into this test rather than its own: two `#[test]` fns in one
    // binary run concurrently and would see each other's env, which is how the
    // baseline assertion above first failed. One test per env-reading binary.
    //
    // The length check must bail rather than slice `window[len - n..]` with
    // `n > len`. Reachable only through the env, because the default window
    // (128) is far larger than the default n-gram (4).
    {
        clear_all();
        // n = 8, but only 2 tokens of window to look at.
        set("LUMEN_RUNAWAY_NGRAM", "8");
        set("LUMEN_RUNAWAY_NGRAM_MAX_REPEATS", "2");
        set("LUMEN_RUNAWAY_NGRAM_WINDOW", "2");
        set("LUMEN_RUNAWAY_MAX_SINGLE_REPEAT", "0"); // isolate the n-gram path

        let d = RunawayDetector::from_env();
        // 16 tokens clears `n * max_repeats`, so the n-gram block is entered — and
        // then the 2-token window is too short to hold an 8-gram.
        let input: Vec<u32> = std::iter::repeat_n(7u32, 16).collect();
        assert_eq!(
            d.check(&input),
            None,
            "a window narrower than the n-gram must bail, not slice out of range"
        );
    }
}
