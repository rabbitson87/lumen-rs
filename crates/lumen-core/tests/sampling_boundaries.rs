//! Boundary coverage for token sampling (005 Phase 4.1).
//!
//! `sampling.rs` had 98 branches with 43 missed, and the misses shared a shape:
//! almost every one was a guard whose **false side had never been taken**. The
//! existing tests check that sampling works; nothing checked what happens at
//! the edges those guards exist for.
//!
//! That distinction matters more here than anywhere else in the crate. A wrong
//! kernel produces a crash or a NaN; a wrong sampler produces *plausible
//! tokens*. Nobody downstream can tell that the nucleus was drawn from a mass
//! of zero, or that an out-of-range token id silently skipped its penalty, or
//! that the accept probability divided by zero — the output still reads like
//! text. So every guard below is exercised on the side that only fires when
//! something has already gone slightly wrong.
//!
//! This is the plan's "boundary checklist" for this module, written as tests
//! rather than as a document: degenerate distributions, exact-equality
//! thresholds, empty and out-of-range inputs, and the fall-through path of
//! every accumulate-until-r loop.

use lumen_core::sampling::{
    SamplingConfig, Xorshift64, apply_presence_frequency_penalty, sample_distribution,
    sample_from_logits, sample_residual, sample_top_p, sampling_distribution, speculative_accept,
};

fn rng() -> Xorshift64 {
    Xorshift64::new(0xDEAD_BEEF)
}

// ───────────────────────────── is_greedy ─────────────────────────────

/// The greedy predicate is a five-way `&&`. Every conjunct has to be able to
/// veto on its own, or a knob silently stops disabling greedy — which would
/// route a temperature-0 request through the sampler, or worse, route a
/// penalised request through argmax and ignore the penalty.
#[test]
fn every_knob_can_veto_greedy_on_its_own() {
    let base = SamplingConfig::default();
    assert!(base.is_greedy(), "the default config is the greedy path");

    let mut c = base.clone();
    c.temperature = 0.7;
    assert!(!c.is_greedy(), "temperature must veto");

    let mut c = base.clone();
    c.repeat_penalty = 1.1;
    assert!(!c.is_greedy(), "repeat_penalty must veto");

    let mut c = base.clone();
    c.presence_penalty = 0.5;
    assert!(!c.is_greedy(), "presence_penalty must veto");

    let mut c = base.clone();
    c.frequency_penalty = 0.5;
    assert!(!c.is_greedy(), "frequency_penalty must veto");

    let mut c = base.clone();
    c.dry.multiplier = 0.8;
    assert!(!c.is_greedy(), "DRY must veto");

    // min_p is excluded deliberately: it can never move the argmax, so it must
    // NOT veto. Pinning this stops someone "fixing" the omission.
    let mut c = base.clone();
    c.min_p = 0.1;
    assert!(
        c.is_greedy(),
        "min_p never changes the argmax, so it must not force the sampling path"
    );

    // Boundary: temperature is `<= 0.0`, so exactly 0.0 is still greedy and the
    // smallest positive value is not.
    let mut c = base.clone();
    c.temperature = 0.0;
    assert!(c.is_greedy());
    c.temperature = f32::MIN_POSITIVE;
    assert!(!c.is_greedy());

    // repeat_penalty uses an epsilon, not equality.
    let mut c = base.clone();
    c.repeat_penalty = 1.0 + 1e-9;
    assert!(
        c.is_greedy(),
        "a sub-epsilon difference is still 'no penalty'"
    );
}

// ────────────────────── presence / frequency penalty ──────────────────────

/// The early return is `presence == 0.0 && frequency == 0.0`, so the second
/// conjunct is only reached when presence is zero. Frequency-only is a real
/// configuration (OpenAI exposes them independently) and it must still apply.
#[test]
fn frequency_only_and_presence_only_both_apply() {
    let window = [1u32, 1, 2];

    let mut logits = vec![0.0_f32; 4];
    apply_presence_frequency_penalty(&mut logits, &window, 0.0, 1.0);
    assert_eq!(logits[1], -2.0, "frequency alone must apply (count 2)");
    assert_eq!(logits[2], -1.0);
    assert_eq!(logits[3], 0.0, "unseen tokens are untouched");

    let mut logits = vec![0.0_f32; 4];
    apply_presence_frequency_penalty(&mut logits, &window, 0.5, 0.0);
    assert_eq!(logits[1], -0.5, "presence is flat, not per-occurrence");
    assert_eq!(logits[2], -0.5);

    // Both zero → untouched, which is the early return.
    let mut logits = vec![1.0_f32; 4];
    apply_presence_frequency_penalty(&mut logits, &window, 0.0, 0.0);
    assert_eq!(logits, vec![1.0; 4]);
}

/// A token id past the end of the logit buffer must be skipped, not panic.
/// This is reachable in production: the penalty window carries ids from the
/// tokenizer, and a grammar-masked or truncated logit slice can be shorter
/// than the vocabulary.
#[test]
fn out_of_range_token_ids_are_skipped_not_panicked_on() {
    let window = [0u32, 99, u32::MAX];
    let mut logits = vec![0.0_f32; 2];
    apply_presence_frequency_penalty(&mut logits, &window, 1.0, 1.0);
    assert_eq!(logits[0], -2.0, "the in-range id is still penalised");
    assert_eq!(logits[1], 0.0, "and the out-of-range ones changed nothing");
}

#[test]
fn an_empty_window_is_a_no_op() {
    let mut logits = vec![1.0_f32, 2.0, 3.0];
    apply_presence_frequency_penalty(&mut logits, &[], 1.0, 1.0);
    assert_eq!(logits, vec![1.0, 2.0, 3.0]);
}

// ───────────────────────────── sample_top_p ─────────────────────────────

/// `top_p` outside `(0, 1)` falls through to a plain categorical draw. Both
/// ends, because `>= 1.0` and `<= 0.0` are separate conditions.
#[test]
fn top_p_outside_the_open_unit_interval_is_a_plain_categorical_draw() {
    let probs = [0.7_f32, 0.2, 0.1];
    for tp in [1.0_f32, 1.5, 0.0, -0.1] {
        let mut r = rng();
        let tok = sample_top_p(&probs, tp, &mut r);
        assert!((tok as usize) < probs.len(), "top_p={tp} produced {tok}");
    }
}

/// A distribution whose mass never reaches `top_p` leaves the cutoff at the
/// full length — the loop's `break` is never taken. An under-normalized
/// distribution is exactly what a masked-and-not-renormalized logit vector
/// looks like, so this is not a synthetic case.
#[test]
fn a_distribution_that_never_reaches_top_p_keeps_everything() {
    // Sums to 0.3, well below any top_p in (0,1).
    let probs = [0.1_f32, 0.1, 0.1];
    let mut counts = [0usize; 3];
    for seed in 0..200u64 {
        let mut r = Xorshift64::new(seed);
        counts[sample_top_p(&probs, 0.9, &mut r) as usize] += 1;
    }
    assert!(
        counts.iter().all(|&c| c > 0),
        "with no cutoff reached every token must stay eligible, got {counts:?}"
    );
}

/// All-zero mass: there is nothing to draw from, and the function must return
/// the top-ranked candidate rather than divide by zero or run off the end.
#[test]
fn an_all_zero_distribution_returns_a_valid_token() {
    let probs = [0.0_f32; 5];
    for seed in 0..20u64 {
        let mut r = Xorshift64::new(seed);
        let tok = sample_top_p(&probs, 0.9, &mut r);
        assert!(
            (tok as usize) < probs.len(),
            "got {tok} from an all-zero vector"
        );
    }
    // Same question of the mass-normalizing helper.
    for seed in 0..20u64 {
        let mut r = Xorshift64::new(seed);
        let tok = sample_distribution(&probs, &mut r);
        assert!((tok as usize) < probs.len());
    }
}

/// The accumulate-until-`r` loops end in a fall-through return, reached when
/// floating-point error leaves `acc` a hair below `r`. Rare per draw and
/// certain over a long run, so it must return a valid token rather than the
/// index-out-of-bounds a `kept[i]` after the loop would give.
#[test]
fn the_accumulator_fall_through_still_returns_a_valid_token() {
    // Mass just under 1.0 makes the final comparison the marginal one.
    let probs = [0.3333333_f32, 0.3333333, 0.3333333];
    for seed in 0..500u64 {
        let mut r = Xorshift64::new(seed);
        let tok = sample_top_p(&probs, 0.999_999, &mut r);
        assert!((tok as usize) < probs.len(), "seed {seed} produced {tok}");
        let mut r = Xorshift64::new(seed);
        let tok = sample_distribution(&probs, &mut r);
        assert!((tok as usize) < probs.len(), "seed {seed} produced {tok}");
    }
}

#[test]
fn a_single_token_distribution_always_returns_it() {
    for tp in [0.1_f32, 0.5, 0.9, 1.0] {
        let mut r = rng();
        assert_eq!(sample_top_p(&[1.0], tp, &mut r), 0);
    }
}

// ──────────────────────── the full pipeline ────────────────────────

/// Temperature is skipped when it is exactly 1.0 (`|t - 1| > 1e-6` guards the
/// scaling loop). The skip has to be a no-op, not a different distribution.
#[test]
fn temperature_exactly_one_skips_scaling_without_changing_the_result() {
    let mut cfg = SamplingConfig {
        temperature: 1.0,
        ..Default::default()
    };
    let logits = vec![2.0_f32, 1.0, 0.5, -1.0];

    let mut a = logits.clone();
    let da = sampling_distribution(&mut a, &[], &cfg);

    // A hair off 1.0 takes the scaling path; the distributions must agree to
    // within float noise, which is what makes the skip a pure optimization.
    cfg.temperature = 1.0 + 1e-5;
    let mut b = logits.clone();
    let db = sampling_distribution(&mut b, &[], &cfg);

    for (x, y) in da.iter().zip(db.iter()) {
        assert!(
            (x - y).abs() < 1e-3,
            "the temperature==1.0 fast path changed the distribution: {da:?} vs {db:?}"
        );
    }
}

/// Both pipeline entry points apply the penalty block, and each has its own
/// copy of the `n > 0` / `repeat_penalty != 1` / `presence|frequency != 0`
/// guards. Covering one says nothing about the other.
#[test]
fn both_pipelines_apply_the_penalty_window() {
    let recent = [1u32, 1, 1];
    let cfg = SamplingConfig {
        temperature: 1.0,
        repeat_penalty: 2.0,
        repeat_penalty_last_n: 8,
        presence_penalty: 0.5,
        frequency_penalty: 0.5,
        ..Default::default()
    };

    // sampling_distribution: token 1 is heavily penalised, so its probability
    // must drop below an equally-scored neighbour that was never seen.
    let mut logits = vec![1.0_f32, 1.0, 1.0];
    let dist = sampling_distribution(&mut logits, &recent, &cfg);
    assert!(
        dist[1] < dist[0] && dist[1] < dist[2],
        "the penalised token should not stay tied: {dist:?}"
    );

    // sample_from_logits: same window, and over many seeds the penalised token
    // must be drawn least often.
    let mut counts = [0usize; 3];
    for seed in 0..300u64 {
        let mut r = Xorshift64::new(seed);
        let mut logits = vec![1.0_f32, 1.0, 1.0];
        counts[sample_from_logits(&mut logits, &recent, &cfg, &mut r) as usize] += 1;
    }
    assert!(
        counts[1] < counts[0] && counts[1] < counts[2],
        "the penalised token was not the rarest draw: {counts:?}"
    );
}

/// `repeat_penalty_last_n` clamps to the available history, and `0` disables
/// the whole block. Both are the `n > 0` guard's two sides.
#[test]
fn the_penalty_window_length_is_clamped_and_zero_disables_it() {
    let cfg_off = SamplingConfig {
        temperature: 1.0,
        repeat_penalty: 4.0,
        repeat_penalty_last_n: 0,
        ..Default::default()
    };
    let mut logits = vec![1.0_f32, 1.0];
    let dist = sampling_distribution(&mut logits, &[0, 0, 0], &cfg_off);
    assert!(
        (dist[0] - dist[1]).abs() < 1e-6,
        "last_n=0 must disable the penalty entirely: {dist:?}"
    );

    // Window longer than the history: clamped, not an out-of-range slice.
    let cfg_long = SamplingConfig {
        repeat_penalty_last_n: 1000,
        ..cfg_off.clone()
    };
    let mut logits = vec![1.0_f32, 1.0];
    let dist = sampling_distribution(&mut logits, &[0], &cfg_long);
    assert!(
        dist[0] < dist[1],
        "a clamped window must still apply: {dist:?}"
    );

    // Empty history with a large window is the same clamp at zero.
    let mut logits = vec![1.0_f32, 1.0];
    let dist = sampling_distribution(&mut logits, &[], &cfg_long);
    assert!((dist[0] - dist[1]).abs() < 1e-6);
}

/// `top_p` in `sampling_distribution` is a mask-and-renormalize, guarded by
/// `top_p < 1.0 && top_p > 0.0`. Both false sides must leave the full
/// distribution intact rather than zeroing it.
#[test]
fn top_p_at_its_disabling_values_leaves_the_distribution_whole() {
    let logits = vec![3.0_f32, 2.0, 1.0, 0.0];
    for top_p in [1.0_f32, 2.0, 0.0, -1.0] {
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_p,
            ..Default::default()
        };
        let mut l = logits.clone();
        let dist = sampling_distribution(&mut l, &[], &cfg);
        assert!(
            dist.iter().all(|&p| p > 0.0),
            "top_p={top_p} must not mask anything, got {dist:?}"
        );
        let sum: f32 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "top_p={top_p} sum={sum}");
    }
}

#[test]
fn top_p_masks_the_tail_and_renormalizes() {
    let cfg = SamplingConfig {
        temperature: 1.0,
        top_p: 0.5,
        ..Default::default()
    };
    let mut logits = vec![10.0_f32, 0.0, -10.0];
    let dist = sampling_distribution(&mut logits, &[], &cfg);
    assert!(
        dist[0] > 0.99,
        "the peak should carry the nucleus: {dist:?}"
    );
    assert_eq!(dist[2], 0.0, "the tail must be masked to exactly zero");
    let sum: f32 = dist.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "must renormalize, sum={sum}");
}

// ─────────────────────── speculative decoding ───────────────────────

/// `q[draft] <= 0` is the divide-by-zero guard, and it has two sub-cases that
/// mean opposite things: the target wants a token the draft never proposed
/// (accept outright) versus neither wants it (reject outright).
#[test]
fn a_zero_draft_probability_resolves_without_dividing() {
    let p = [0.0_f32, 1.0];
    let q = [0.5_f32, 0.0];

    // q[1] == 0 but p[1] > 0 → always accept.
    for seed in 0..50u64 {
        let mut r = Xorshift64::new(seed);
        let out = speculative_accept(&p, &q, 1, &mut r);
        assert!(out.accepted, "p>0 with q==0 must always accept");
        assert_eq!(out.token, 1);
    }

    // q[0] > 0 but p[0] == 0 → never accept, and the correction must be a
    // token the target actually wants.
    for seed in 0..50u64 {
        let mut r = Xorshift64::new(seed);
        let out = speculative_accept(&p, &q, 0, &mut r);
        assert!(!out.accepted, "p==0 must never accept");
        assert_eq!(out.token, 1, "the correction must come from p's support");
    }

    // Both zero → reject (accept_p 0.0), still returning a valid token.
    let p0 = [0.0_f32, 0.0, 1.0];
    let q0 = [0.0_f32, 0.0, 0.0];
    let mut r = rng();
    let out = speculative_accept(&p0, &q0, 1, &mut r);
    assert!(!out.accepted);
    assert!((out.token as usize) < p0.len());
}

/// A draft id past the end of either distribution must be handled by the
/// `get().unwrap_or(0.0)` fallbacks, not panic. Draft heads can propose ids
/// outside a masked slice.
#[test]
fn an_out_of_range_draft_token_does_not_panic() {
    let p = [0.5_f32, 0.5];
    let q = [0.5_f32, 0.5];
    for draft in [2u32, 99, u32::MAX] {
        let mut r = rng();
        let out = speculative_accept(&p, &q, draft, &mut r);
        assert!(
            !out.accepted,
            "an unknown token has p=0 and cannot be accepted"
        );
        assert!((out.token as usize) < p.len());
    }
}

/// When `q` dominates `p` everywhere the residual has no mass, and the
/// documented fallback is `argmax(p)` — the only choice that keeps the
/// committed token in the target's support.
#[test]
fn a_degenerate_residual_falls_back_to_the_target_argmax() {
    let p = [0.2_f32, 0.3, 0.5];
    let q = [0.9_f32, 0.9, 0.9];
    for seed in 0..50u64 {
        let mut r = Xorshift64::new(seed);
        assert_eq!(
            sample_residual(&p, &q, &mut r),
            2,
            "with no residual mass the fallback must be argmax(p)"
        );
    }
}

/// `q` shorter than `p` is treated as zero past its end rather than truncating
/// the residual — otherwise tokens the draft never scored would be unreachable.
#[test]
fn a_shorter_draft_distribution_is_padded_with_zeros() {
    let p = [0.1_f32, 0.1, 0.8];
    let q = [0.5_f32];
    let mut r = rng();
    let tok = sample_residual(&p, &q, &mut r);
    assert!(
        tok == 1 || tok == 2,
        "index 0 is fully covered by q, so the residual should not pick it (got {tok})"
    );
}

/// Identical distributions accept with probability 1 — the `min(1, p/q)` cap.
#[test]
fn identical_distributions_always_accept() {
    let p = [0.25_f32, 0.25, 0.5];
    for seed in 0..100u64 {
        let mut r = Xorshift64::new(seed);
        let out = speculative_accept(&p, &p, 2, &mut r);
        assert!(out.accepted, "p == q must accept every time");
        assert_eq!(out.token, 2);
    }
}

// ───────────── degenerate inputs to the in-place transforms ─────────────
//
// These four guards all handle a logit buffer that is already degenerate —
// every entry masked, nothing summing, a token id past the end. Reaching them
// means something upstream went wrong, and every one of them chooses to
// *recover* rather than fail, which is right for a sampler and is exactly why
// they need pinning: a broken recovery produces tokens, not an error.

/// The repeat penalty's own no-op guard. Its callers already check
/// `repeat_penalty != 1.0`, so this path is only reached by a direct caller —
/// and it must be a true no-op, not a pass through the loop with a factor of 1
/// (which would still perturb signs).
#[test]
fn a_unit_repeat_penalty_leaves_the_logits_untouched() {
    let original = vec![2.0_f32, -1.0, 0.0, 5.0];
    for penalty in [1.0_f32, 1.0 + 1e-9, 1.0 - 1e-9] {
        let mut logits = original.clone();
        lumen_core::sampling::apply_repeat_penalty(&mut logits, &[0, 1, 2, 3], penalty);
        assert_eq!(logits, original, "penalty {penalty} must be a no-op");
    }
}

/// An out-of-range token id in the repeat window must be skipped. Same shape as
/// the presence/frequency case above, different function — and a `logits[i]`
/// without the guard is an index panic on a live request.
#[test]
fn the_repeat_penalty_skips_out_of_range_token_ids() {
    let mut logits = vec![2.0_f32, -2.0];
    lumen_core::sampling::apply_repeat_penalty(&mut logits, &[0, 1, 99, u32::MAX], 2.0);
    // Sign-aware: positive divided, negative multiplied.
    assert!(logits[0] < 2.0, "a positive logit is pushed down");
    assert!(logits[1] < -2.0, "a negative logit is pushed further down");
}

/// Every logit `-inf` — what a grammar mask leaves when nothing is allowed.
/// `min_p` has no finite peak to measure against, so it must bail rather than
/// compute `max + ln(min_p)` on an infinity and mask the buffer into NaN.
#[test]
fn min_p_bails_when_no_logit_is_finite() {
    for fill in [f32::NEG_INFINITY, f32::NAN] {
        let mut logits = vec![fill; 4];
        lumen_core::sampling::apply_min_p(&mut logits, 0.5);
        assert_eq!(logits.len(), 4);
        assert!(
            logits.iter().all(|v| v.is_infinite() || v.is_nan()),
            "the buffer must be left as it was, not turned into something worse"
        );
    }
    // A single finite entry is enough to proceed, and it must survive.
    let mut logits = vec![f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY];
    lumen_core::sampling::apply_min_p(&mut logits, 0.5);
    assert_eq!(logits[1], 1.0, "the peak always survives min_p");
}

/// A softmax whose exponentials all underflow to zero has no distribution to
/// return. Falling back to uniform keeps the sampler able to draw *something*;
/// returning all-zeros would make every downstream mass check fail and the
/// draw degenerate.
#[test]
fn a_softmax_that_underflows_falls_back_to_uniform() {
    // All -inf: every exp() is 0, so the sum is 0.
    let mut logits = vec![f32::NEG_INFINITY; 4];
    lumen_core::sampling::softmax_inplace(&mut logits);
    let sum: f32 = logits.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "must still be a distribution: {logits:?}"
    );
    assert!(
        logits.iter().all(|&p| (p - 0.25).abs() < 1e-6),
        "and a uniform one: {logits:?}"
    );

    // Sanity: a normal buffer is unaffected by the fallback.
    let mut logits = vec![1.0_f32, 2.0, 3.0];
    lumen_core::sampling::softmax_inplace(&mut logits);
    assert!(logits[2] > logits[1] && logits[1] > logits[0]);
}

/// The penalty block's guards have false sides too: a window exists, but the
/// penalty is disabled. Both pipelines, both guards — a config that sets
/// `repeat_penalty_last_n` while leaving the penalties at their defaults is
/// the ordinary case, and it must cost nothing.
#[test]
fn a_window_with_disabled_penalties_changes_nothing() {
    let recent = [1u32, 1, 1];
    let cfg = SamplingConfig {
        temperature: 1.0,
        repeat_penalty: 1.0,   // disabled
        presence_penalty: 0.0, // disabled
        frequency_penalty: 0.0,
        repeat_penalty_last_n: 8, // but the window exists
        ..Default::default()
    };
    let mut a = vec![1.0_f32, 1.0, 1.0];
    let with_window = sampling_distribution(&mut a, &recent, &cfg);
    let mut b = vec![1.0_f32, 1.0, 1.0];
    let without = sampling_distribution(&mut b, &[], &cfg);
    assert_eq!(
        with_window, without,
        "a window with every penalty disabled must be indistinguishable from no window"
    );

    // Frequency-only through the pipeline reaches the second operand of the
    // `presence != 0 || frequency != 0` guard, which presence-only cannot.
    let cfg = SamplingConfig {
        frequency_penalty: 0.5,
        ..cfg
    };
    let mut c = vec![1.0_f32, 1.0, 1.0];
    let dist = sampling_distribution(&mut c, &recent, &cfg);
    assert!(
        dist[1] < dist[0],
        "frequency-only must still apply: {dist:?}"
    );
}

/// `sampling_distribution` has its own top-p loop, separate from
/// `sample_top_p`'s. Its cutoff-never-reached and zero-mass arms are therefore
/// separate branches, and an all-masked distribution must not divide by zero.
#[test]
fn the_distribution_top_p_survives_a_distribution_with_no_mass() {
    let cfg = SamplingConfig {
        temperature: 1.0,
        top_p: 0.9,
        min_p: 0.0,
        ..Default::default()
    };
    // Every logit -inf → softmax falls back to uniform → top_p still applies.
    let mut logits = vec![f32::NEG_INFINITY; 4];
    let dist = sampling_distribution(&mut logits, &[], &cfg);
    assert_eq!(dist.len(), 4);
    assert!(
        dist.iter().all(|p| p.is_finite() && *p >= 0.0),
        "a degenerate input must not produce NaN probabilities: {dist:?}"
    );

    // A distribution whose mass never reaches top_p keeps every entry.
    let cfg = SamplingConfig {
        temperature: 1.0,
        top_p: 0.999_999,
        ..Default::default()
    };
    let mut logits = vec![1.0_f32; 8];
    let dist = sampling_distribution(&mut logits, &[], &cfg);
    assert!(
        dist.iter().all(|&p| p > 0.0),
        "no cutoff reached means nothing is masked: {dist:?}"
    );
}
