//! `always!()` / `never!()` — defensive conditions that leave the coverage
//! denominator (005 Phase 4.1).
//!
//! This is SQLite's `ALWAYS()` / `NEVER()`, and it exists to resolve a real
//! conflict rather than to be clever. Defensive code is good practice: a
//! `match` arm for a case that "cannot happen" turns a future refactor's bug
//! into a clean error instead of silent corruption. But a branch that cannot be
//! taken also cannot be covered, so writing defensive code caps branch coverage
//! below 100% — and a target you can never reach is a target nobody defends.
//! Most projects resolve this by abandoning the target. SQLite resolved it by
//! marking the unreachable side so it leaves the *denominator*.
//!
//! Three behaviours from one call site:
//!
//! | build | expands to | why |
//! |---|---|---|
//! | coverage (`--cfg coverage`) | the constant | the branch is never emitted, so it is not counted |
//! | debug | `debug_assert!` + the condition | the claim is checked on every test run |
//! | release | the condition | zero cost; the defensive arm still works |
//!
//! So the assertion is real (debug), the guard is real (release), and the
//! impossible arm does not silently shrink the score into a lie (coverage).
//!
//! ## Measured, not assumed
//!
//! Whether the branch actually leaves the denominator is a claim about LLVM's
//! instrumentation, not about what the macro expands to — so
//! [`__probe_guarded`] and [`__probe_unguarded`] are identical functions
//! differing only in the guard, and `cargo xtask coverage` asserts the
//! difference on every run. Measured on nightly with `cargo llvm-cov --branch`:
//!
//! ```text
//! guarded fn,   real `if xs.is_empty()`   Branch [True: 1, False: 1]  covered
//! unguarded fn, real `if xs.is_empty()`   Branch [True: 1, False: 1]  covered
//! unguarded fn, DEFENSIVE `if`            Branch [True: 1, False: 0]  MISSED
//! guarded fn,   DEFENSIVE `always!(…)`    — no Branch entry emitted —
//! ```
//!
//! (Described by shape rather than by line number on purpose: `cargo fmt`
//! moves these functions, and a doc comment citing stale lines is worse than
//! one citing none.)
//!
//! The unguarded defensive branch is a permanently-uncoverable miss; the
//! guarded one contributes nothing to the denominator. That is the whole
//! mechanism, and the probes exist so a toolchain change that breaks it fails
//! loudly instead of quietly restoring the cap.
//!
//! **Known limit, in scope by decision.** The dead arm's *region* is still
//! emitted (the report marks it `^0`), so LLVM's region and line metrics remain
//! capped. Only the branch metric — and MC/DC, which is what Phase 4 targets —
//! is cleared. Clearing regions too would mean a macro that swallows the whole
//! `if`/`else`, which buys a metric this project does not gate on at the cost of
//! an API that hides control flow.
//!
//! ## When *not* to use these
//!
//! `always!(x)` asserts that `x` is true **by construction** — an invariant a
//! caller cannot violate through the public API. If a bad input can make it
//! false, it is a validation check, not a defensive one: write a real `if` and
//! return an error, and let it be covered like any other branch. Marking a
//! reachable branch with these macros removes it from the denominator and hides
//! exactly the case a test should be pinning.
//!
//! ## Caveat: unused variables under coverage
//!
//! Under `--cfg coverage` the condition is not evaluated, so a binding used
//! *only* inside `always!()` will warn. That warning is usually correct and
//! worth reading rather than silencing: if the sole use of a value is a
//! defensive check, the value is either dead or the check is load-bearing (and
//! therefore not defensive).

/// Asserts a condition that is true by construction, and evaluates to it.
///
/// See the [module docs](self) for the three-build behaviour and for when a
/// plain `if` is the right thing instead.
///
/// ```
/// # use lumen_core::always;
/// fn head(xs: &[u32]) -> Option<u32> {
///     if xs.is_empty() {
///         return None;
///     }
///     // Non-empty was just established, so this cannot be false.
///     if always!(!xs.is_empty()) { Some(xs[0]) } else { None }
/// }
/// assert_eq!(head(&[7, 8]), Some(7));
/// assert_eq!(head(&[]), None);
/// ```
#[macro_export]
macro_rules! always {
    ($cond:expr) => {{
        #[cfg(coverage)]
        {
            true
        }
        #[cfg(all(not(coverage), debug_assertions))]
        {
            let c = $cond;
            // `{}` and not the concat directly: a condition containing a
            // brace — `bytes[i] != b'{'`, any struct literal — would otherwise
            // be parsed as a format specifier and fail to compile at the call
            // site. Found the first time this was used in a parser.
            debug_assert!(
                c,
                "{}",
                concat!(
                    "always!(",
                    stringify!($cond),
                    ") was false — an invariant \
                         held by construction no longer holds"
                )
            );
            c
        }
        #[cfg(all(not(coverage), not(debug_assertions)))]
        {
            $cond
        }
    }};
}

/// Asserts a condition that is false by construction, and evaluates to it.
///
/// The mirror of [`always!`]; see the [module docs](self).
///
/// ```
/// # use lumen_core::never;
/// fn checked_div(a: u32, b: u32) -> Option<u32> {
///     let b = if b == 0 { return None } else { b };
///     // Zero was just excluded.
///     if never!(b == 0) { None } else { Some(a / b) }
/// }
/// assert_eq!(checked_div(10, 2), Some(5));
/// assert_eq!(checked_div(10, 0), None);
/// ```
#[macro_export]
macro_rules! never {
    ($cond:expr) => {{
        #[cfg(coverage)]
        {
            false
        }
        #[cfg(all(not(coverage), debug_assertions))]
        {
            let c = $cond;
            // `{}` and not the concat directly: a condition containing a
            // brace — `bytes[i] != b'{'`, any struct literal — would otherwise
            // be parsed as a format specifier and fail to compile at the call
            // site. Found the first time this was used in a parser.
            debug_assert!(
                !c,
                "{}",
                concat!(
                    "never!(",
                    stringify!($cond),
                    ") was true — a case held \
                         impossible by construction just happened"
                )
            );
            c
        }
        #[cfg(all(not(coverage), not(debug_assertions)))]
        {
            $cond
        }
    }};
}

// ── coverage-denominator probe (005 Phase 4.1) ──
//
// The claim this module rests on — that a defensive branch guarded by
// `always!()` leaves the coverage denominator — is a claim about what LLVM's
// instrumentation does, not about what the macro expands to. These two
// structurally identical functions differ only in the guard, so a coverage run
// over them measures the claim directly: `unguarded` must show a missed branch,
// `guarded` must show none.

#[doc(hidden)]
pub fn __probe_guarded(xs: &[u32]) -> Option<u32> {
    if xs.is_empty() {
        return None;
    }
    if always!(!xs.is_empty()) {
        Some(xs[0])
    } else {
        None
    }
}

#[doc(hidden)]
pub fn __probe_unguarded(xs: &[u32]) -> Option<u32> {
    if xs.is_empty() {
        return None;
    }
    if !xs.is_empty() { Some(xs[0]) } else { None }
}

#[cfg(test)]
mod tests {
    /// Exercises both probes so the coverage report has data for them.
    #[test]
    fn the_probes_run() {
        assert_eq!(super::__probe_guarded(&[1]), Some(1));
        assert_eq!(super::__probe_unguarded(&[1]), Some(1));
        assert_eq!(super::__probe_guarded(&[]), None);
        assert_eq!(super::__probe_unguarded(&[]), None);
    }

    /// The value semantics must be the pass-through the release build relies
    /// on, in whichever configuration the suite runs.
    #[test]
    fn the_macros_evaluate_to_their_condition() {
        assert!(always!(1 + 1 == 2));
        assert!(!never!(1 + 1 == 3));
    }

    /// They must be usable as expressions, not just statements — the whole
    /// point is guarding a branch condition in place.
    #[test]
    fn they_compose_as_expressions() {
        let xs = [3u32, 1, 2];
        let first = if always!(!xs.is_empty()) {
            Some(xs[0])
        } else {
            None
        };
        assert_eq!(first, Some(3));

        let n = if never!(xs.len() > 100) { 0 } else { xs.len() };
        assert_eq!(n, 3);
    }

    /// The condition is evaluated exactly once outside coverage builds.
    /// A macro that evaluated it twice would double any side effect a caller
    /// left in the expression — worth pinning even though the docs ask for
    /// pure conditions.
    #[cfg(not(coverage))]
    #[test]
    fn the_condition_is_evaluated_once() {
        let mut calls = 0;
        let mut probe = || {
            calls += 1;
            true
        };
        assert!(always!(probe()));
        assert_eq!(calls, 1);
    }

    /// A violated `always!` must fail loudly in debug, which is what makes it
    /// an assertion rather than a comment.
    #[cfg(all(not(coverage), debug_assertions))]
    #[test]
    #[should_panic(expected = "always!")]
    fn a_violated_always_panics_in_debug() {
        let _ = always!(1 + 1 == 3);
    }

    #[cfg(all(not(coverage), debug_assertions))]
    #[test]
    #[should_panic(expected = "never!")]
    fn a_violated_never_panics_in_debug() {
        let _ = never!(1 + 1 == 2);
    }

    /// Under a coverage build the macros are constants: the defensive branch is
    /// never emitted, so it cannot appear in the denominator. This is the
    /// entire reason the module exists, so it is asserted rather than assumed.
    #[cfg(coverage)]
    #[test]
    fn coverage_builds_fold_to_constants() {
        // `false` as the condition, `true` as the result — only a constant fold
        // can produce that.
        assert!(always!(1 + 1 == 3));
        assert!(!never!(1 + 1 == 2));
    }
}
