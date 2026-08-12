//! Boundary coverage for the DRY penalty (005 Phase 4.1).
//!
//! DRY catches the n-gram cycles the single-token repeat penalty misses — the
//! "…and then, and then, and then" failure that runs a request to
//! `max_tokens`. Its penalty is `multiplier * base^(repeat_len - allowed)`, an
//! **exponential**, and everything uncovered here guarded that exponent:
//!
//! * `is_disabled`'s three operands, only one of which was exercised. A
//!   disabled-check that stops recognising one of its off-switches turns DRY on
//!   for everyone.
//! * The overflow clamp. `base^exp` overflows f32 past ~88.7 in natural log, so
//!   the exponent is capped — and an uncapped one produces `inf`, which
//!   subtracts to `-inf` and removes the token from the vocabulary entirely.
//!   Not a crash: a token that can never be emitted again.
//! * The `base > 1.000001` guard on computing that cap, which is a division by
//!   `ln(base)` — at `base == 1.0`, `ln(base)` is 0.
//!
//! Every one of these fails by producing text rather than an error.

use lumen_core::dry::{DryConfig, apply_dry_penalty};

/// A config with DRY genuinely on, so the `is_disabled` tests below have
/// something to turn off.
fn enabled() -> DryConfig {
    DryConfig {
        multiplier: 1.0,
        base: 1.75,
        allowed_length: 2,
        penalty_last_n: 64,
    }
}

/// `is_disabled` is a three-way `||`. Each operand must be able to disable on
/// its own, and the enabled config must not be disabled by any of them.
#[test]
fn every_off_switch_disables_dry_on_its_own() {
    assert!(!enabled().is_disabled(), "the fixture must be enabled");

    let mut c = enabled();
    c.multiplier = 0.0;
    assert!(c.is_disabled(), "multiplier 0 disables");

    let mut c = enabled();
    c.base = 0.9;
    assert!(c.is_disabled(), "a base below 1.0 disables");
    c.base = 0.0;
    assert!(c.is_disabled());

    let mut c = enabled();
    c.penalty_last_n = 0;
    assert!(c.is_disabled(), "a zero window disables");

    // Boundaries: base exactly 1.0 is NOT below 1.0, so it stays enabled — and
    // it is the value the exponent-cap guard has to survive.
    let mut c = enabled();
    c.base = 1.0;
    assert!(!c.is_disabled(), "base == 1.0 is enabled, not disabled");

    // The default is off, which is what makes DRY opt-in.
    assert!(DryConfig::default().is_disabled());
}

/// A disabled config must not touch the logits, through any of its three
/// off-switches. `apply_dry_penalty` is called unconditionally in the pipeline,
/// so this no-op is the common case on every request.
#[test]
fn a_disabled_config_is_a_no_op() {
    let recent = [1u32, 2, 1, 2, 1, 2];
    let original = vec![1.0_f32; 8];

    for c in [
        DryConfig::default(),
        DryConfig {
            multiplier: 0.0,
            ..enabled()
        },
        DryConfig {
            base: 0.5,
            ..enabled()
        },
        DryConfig {
            penalty_last_n: 0,
            ..enabled()
        },
    ] {
        let mut logits = original.clone();
        apply_dry_penalty(&mut logits, &recent, &c);
        assert_eq!(
            logits, original,
            "disabled config {c:?} must not touch logits"
        );
    }
}

/// The window has to be longer than `allowed_length` for a repeat to be
/// *over* the allowance — with a shorter history there is nothing to penalise,
/// and the early return is what keeps a fresh conversation untouched.
#[test]
fn a_window_no_longer_than_the_allowance_is_a_no_op() {
    let cfg = enabled(); // allowed_length 2
    let original = vec![1.0_f32; 4];

    for recent in [&[][..], &[1][..], &[1, 2][..]] {
        let mut logits = original.clone();
        apply_dry_penalty(&mut logits, recent, &cfg);
        assert_eq!(
            logits, original,
            "history {recent:?} is not longer than allowed_length, nothing to do"
        );
    }

    // `penalty_last_n` clamps the window too, so a short cap has the same
    // effect on a long history.
    let cfg = DryConfig {
        penalty_last_n: 2,
        ..enabled()
    };
    let mut logits = original.clone();
    apply_dry_penalty(&mut logits, &[1, 2, 1, 2, 1, 2], &cfg);
    assert_eq!(
        logits, original,
        "a clamped window shorter than the allowance"
    );
}

/// The thing DRY exists for: a repeating n-gram must push down the token that
/// would continue it, and leave everything else alone.
#[test]
fn a_repeating_ngram_penalises_only_its_continuation() {
    let cfg = enabled();
    // "1 2 3" three times: after the trailing `1 2`, token 3 continues the
    // cycle and is the one to suppress.
    let recent = [1u32, 2, 3, 1, 2, 3, 1, 2];
    let mut logits = vec![0.0_f32; 8];
    apply_dry_penalty(&mut logits, &recent, &cfg);

    assert!(
        logits[3] < 0.0,
        "the continuation must be penalised: {logits:?}"
    );
    for other in [0usize, 4, 5, 6, 7] {
        assert_eq!(
            logits[other], 0.0,
            "token {other} does not continue the cycle: {logits:?}"
        );
    }
}

/// **The overflow clamp.** A long repetition drives `base^(len - allowed)` past
/// f32's range; uncapped it becomes `inf`, the logit becomes `-inf`, and that
/// token is removed from the vocabulary for the rest of the generation. The
/// penalty must stay finite no matter how long the repeat runs.
#[test]
fn a_very_long_repetition_stays_finite() {
    let cfg = DryConfig {
        multiplier: 1.0,
        base: 1.75,
        allowed_length: 2,
        penalty_last_n: 4096,
    };
    // 400 repeats of a 2-gram — far past where 1.75^n overflows f32.
    let recent: Vec<u32> = std::iter::repeat([7u32, 8]).take(400).flatten().collect();
    let mut logits = vec![0.0_f32; 16];
    apply_dry_penalty(&mut logits, &recent, &cfg);

    assert!(
        logits.iter().all(|v| v.is_finite()),
        "an unclamped exponent removes the token from the vocabulary forever: {:?}",
        &logits[..10]
    );
    assert!(
        logits[7] < 0.0 || logits[8] < 0.0,
        "and it must still penalise"
    );
}

/// `base == 1.0` is enabled but makes `ln(base)` zero, so the cap computation
/// is guarded. Unguarded it divides by zero and the cap becomes `inf` or `NaN`,
/// which then flows straight into the penalty.
#[test]
fn a_base_of_exactly_one_does_not_divide_by_zero() {
    let cfg = DryConfig {
        multiplier: 1.0,
        base: 1.0,
        allowed_length: 2,
        penalty_last_n: 4096,
    };
    assert!(
        !cfg.is_disabled(),
        "base 1.0 is enabled, so this path is live"
    );

    let recent: Vec<u32> = std::iter::repeat([7u32, 8]).take(200).flatten().collect();
    let mut logits = vec![0.0_f32; 16];
    apply_dry_penalty(&mut logits, &recent, &cfg);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "base 1.0 must not produce inf/NaN: {:?}",
        &logits[..10]
    );
    // base^n == 1 for every n, so the penalty is exactly the multiplier.
    assert!(
        logits.iter().all(|&v| v > -2.0),
        "with base 1.0 the penalty cannot grow with length: {:?}",
        &logits[..10]
    );

    // Just above the guard's threshold takes the other arm.
    let cfg = DryConfig { base: 1.01, ..cfg };
    let mut logits = vec![0.0_f32; 16];
    apply_dry_penalty(&mut logits, &recent, &cfg);
    assert!(logits.iter().all(|v| v.is_finite()));
}

/// When several repeat lengths point at the same continuation token, the
/// longest wins — the penalty tracks the strongest evidence of a cycle, not the
/// most recent. A max that never updated would under-penalise a long cycle
/// nested inside a short one.
#[test]
fn the_longest_matching_repeat_sets_the_penalty() {
    let cfg = enabled();
    // A short cycle and a longer one both ending in the same continuation.
    let recent = [9u32, 1, 2, 3, 4, 9, 1, 2, 3, 4, 9, 1, 2, 3, 4];
    let mut long_logits = vec![0.0_f32; 16];
    apply_dry_penalty(&mut long_logits, &recent, &cfg);

    let short = [1u32, 2, 3, 1, 2];
    let mut short_logits = vec![0.0_f32; 16];
    apply_dry_penalty(&mut short_logits, &short, &cfg);

    let long_worst = long_logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let short_worst = short_logits.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        long_worst <= short_worst,
        "a longer established cycle must be penalised at least as hard: \
         long {long_worst} vs short {short_worst}"
    );
}

/// `allowed_length` is the free-repeat budget: a cycle at or under it is not
/// penalised, and one token past it is. This is the knob operators actually
/// turn, so both sides of the comparison matter.
#[test]
fn allowed_length_is_the_free_repeat_budget() {
    let recent = [1u32, 2, 3, 1, 2, 3, 1, 2];
    let mut penalised = vec![0.0_f32; 8];
    apply_dry_penalty(
        &mut penalised,
        &recent,
        &DryConfig {
            allowed_length: 2,
            ..enabled()
        },
    );
    assert!(
        penalised.iter().any(|&v| v < 0.0),
        "a 3-gram cycle past allowed=2"
    );

    let mut tolerated = vec![0.0_f32; 8];
    apply_dry_penalty(
        &mut tolerated,
        &recent,
        &DryConfig {
            allowed_length: 64,
            ..enabled()
        },
    );
    assert_eq!(
        tolerated,
        vec![0.0_f32; 8],
        "a generous allowance must tolerate the same cycle"
    );
}
