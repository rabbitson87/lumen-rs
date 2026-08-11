//! Test-only helpers shared across the lumen crates.
//!
//! The numeric comparisons below were copy-pasted into eighteen test files
//! under three different names (`cosine`, `cosine_sim`, `cosine_similarity`)
//! and two different return types, which meant a parity threshold could not be
//! reasoned about across the suite: two tests asserting "cosine ≥ 0.99" were
//! not necessarily asserting the same thing, and an epsilon fix in one copy
//! never reached the others.
//!
//! Everything takes `&[f32]` and returns `f64`. No framework tensor types —
//! see the note in `Cargo.toml`.
//!
//! This crate also hosts the `arbitrary` input generators and fault-injection
//! fixtures the fuzz targets share; see [`generators`].

pub mod generators;

/// Cosine similarity of two equal-length vectors.
///
/// Accumulates in `f64`. The `f32` accumulators the copies used lose the low
/// bits of a long dot product exactly where these tests are most sensitive —
/// a 4096-element bf16 parity check is comparing values that agree to ~1e-3,
/// so the metric must not itself be the noisiest term.
///
/// Degenerate inputs are defined rather than left to produce `NaN`, because a
/// `NaN` here fails a parity assertion for a reason that has nothing to do
/// with the kernel under test:
///
/// - both vectors zero → `1.0`. They are identical; magnitude is what cosine
///   deliberately ignores.
/// - exactly one zero → `0.0`. Maximally dissimilar, and finite. This matches
///   the `norm < 1e-10 => 0.0` guard the `lumen-core` copy used, which several
///   compressor tests depend on.
///
/// # Panics
/// If the lengths differ — that is a broken test, not a failing assertion.
#[must_use]
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "cosine: length mismatch");
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        dot += f64::from(x) * f64::from(y);
        na += f64::from(x) * f64::from(x);
        nb += f64::from(y) * f64::from(y);
    }
    match (na == 0.0, nb == 0.0) {
        (true, true) => 1.0,
        (true, false) | (false, true) => 0.0,
        (false, false) => dot / (na.sqrt() * nb.sqrt()),
    }
}

/// Largest absolute difference between two equal-length vectors.
///
/// # Panics
/// If the lengths differ.
#[must_use]
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "max_abs_diff: length mismatch");
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
        .fold(0.0, f64::max)
}

/// Largest *relative* difference, scaled by the reference magnitude.
///
/// `eps` guards the denominator. Pass the smallest magnitude that is
/// meaningful for the tensor under test — with `eps` too small, an element
/// that is zero in both inputs reports an enormous relative error and the
/// whole comparison becomes noise.
///
/// # Panics
/// If the lengths differ.
#[must_use]
pub fn max_rel_diff(actual: &[f32], reference: &[f32], eps: f64) -> f64 {
    assert_eq!(
        actual.len(),
        reference.len(),
        "max_rel_diff: length mismatch"
    );
    actual
        .iter()
        .zip(reference)
        .map(|(&x, &r)| {
            let r = f64::from(r);
            (f64::from(x) - r).abs() / (r.abs().max(eps))
        })
        .fold(0.0, f64::max)
}

/// True when every element matches bit-for-bit, `NaN` included.
///
/// `==` on `f32` reports `NaN != NaN`, so a comparison built on it cannot tell
/// "the kernel returned NaN, exactly as the reference did" from "the kernel
/// broke". Bit equality is the right predicate for the A/B checks that claim
/// two code paths are the *same* computation rather than a close one.
#[must_use]
pub fn bit_identical(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Deterministic pseudo-random `f32`s in roughly `[-0.5, 0.5]`.
///
/// A seeded LCG rather than `rand`, so test data is reproducible across
/// machines and runs without a dependency. Not for cryptography or for
/// statistical work — only for filling tensors.
#[must_use]
pub fn deterministic_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(2_862_933_555_777_941_757)
        .wrapping_add(3_037_000_493);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / 16_777_216.0) - 0.5
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_of_identical_vectors_is_one() {
        let v = deterministic_f32(256, 7);
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_of_opposed_vectors_is_minus_one() {
        let v = deterministic_f32(256, 7);
        let neg: Vec<f32> = v.iter().map(|x| -x).collect();
        assert!((cosine(&v, &neg) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_of_orthogonal_vectors_is_zero() {
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-12);
    }

    #[test]
    fn cosine_of_two_zero_vectors_is_one_not_nan() {
        // The f32 copies returned `0/(0+1e-30)` = 0, reading as "completely
        // dissimilar" for two outputs that are in fact identical.
        assert_eq!(cosine(&[0.0, 0.0], &[0.0, 0.0]), 1.0);
    }

    #[test]
    fn cosine_of_one_zero_vector_is_zero_not_nan() {
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 2.0]), 0.0);
        assert_eq!(cosine(&[1.0, 2.0], &[0.0, 0.0]), 0.0);
    }

    #[test]
    fn max_abs_diff_finds_the_largest_gap() {
        assert!((max_abs_diff(&[1.0, 5.0, 2.0], &[1.0, 2.0, 2.0]) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn max_rel_diff_scales_by_the_reference() {
        // Same absolute gap of 1.0, but against references 1000 and 1.
        let far = max_rel_diff(&[1001.0], &[1000.0], 1e-6);
        let near = max_rel_diff(&[2.0], &[1.0], 1e-6);
        assert!(far < near, "far={far} near={near}");
    }

    #[test]
    fn max_rel_diff_eps_keeps_a_zero_reference_finite() {
        assert!(max_rel_diff(&[0.0], &[0.0], 1e-6).is_finite());
    }

    #[test]
    fn bit_identical_is_true_for_matching_nans() {
        let nan = [f32::NAN];
        assert!(bit_identical(&nan, &nan));
        assert!(nan[0] != nan[0], "…which `==` cannot express");
    }

    #[test]
    fn bit_identical_separates_plus_and_minus_zero() {
        assert!(!bit_identical(&[0.0], &[-0.0]));
    }

    #[test]
    fn deterministic_f32_is_reproducible_and_seed_dependent() {
        assert_eq!(deterministic_f32(64, 42), deterministic_f32(64, 42));
        assert_ne!(deterministic_f32(64, 42), deterministic_f32(64, 43));
    }

    #[test]
    fn deterministic_f32_stays_in_range() {
        assert!(
            deterministic_f32(4096, 1)
                .iter()
                .all(|v| (-0.5..=0.5).contains(v))
        );
    }
}
