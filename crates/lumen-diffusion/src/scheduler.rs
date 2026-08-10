//! Flow-match Euler-discrete scheduler for FLUX.2 (klein), Phase 3.
//!
//! A scalar-exact port of mflux's `FlowMatchEulerDiscreteScheduler`
//! (`set_image_seq_len` → `get_timesteps_and_sigmas` path, which klein uses
//! because `requires_sigma_shift` is true).
//!
//! ```text
//! sigmas   = linspace(1.0, 1/steps, steps)                # length=steps
//! mu       = empirical_mu(image_seq_len, steps)
//! sigmas   = exp(mu) / (exp(mu) + (1/sigmas - 1))          # time-shift
//! timesteps= sigmas * 1000                                 # length=steps
//! sigmas  += [0.0]                                         # length=steps+1
//! step:  latents += (sigmas[t+1] - sigmas[t]) * noise      # Euler
//! ```
//!
//! All arithmetic is f32 scalar (no MLX) so the schedule is host-computable and
//! independently verifiable against the mflux dump.

/// Computed schedule: `timesteps` (length `steps`, ×1000-scaled) and `sigmas`
/// (length `steps + 1`, terminal 0.0).
#[derive(Clone, Debug)]
pub struct Schedule {
    pub sigmas: Vec<f32>,    // len = steps + 1
    pub timesteps: Vec<f32>, // len = steps
}

/// mflux `_compute_empirical_mu`.
fn empirical_mu(image_seq_len: usize, num_steps: usize) -> f32 {
    let isl = image_seq_len as f64;
    let (a1, b1) = (8.73809524e-05_f64, 1.89833333_f64);
    let (a2, b2) = (0.00016927_f64, 0.45666666_f64);
    if isl > 4300.0 {
        return (a2 * isl + b2) as f32;
    }
    let m_200 = a2 * isl + b2;
    let m_10 = a1 * isl + b1;
    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    (a * num_steps as f64 + b) as f32
}

/// Compute the FLUX.2 flow-match schedule for `image_seq_len` and `num_steps`.
pub fn compute(image_seq_len: usize, num_steps: usize) -> Schedule {
    assert!(num_steps >= 1, "num_steps must be >= 1");
    let n = num_steps as f64;
    let mu = empirical_mu(image_seq_len, num_steps) as f64;
    let exp_mu = mu.exp();

    // linspace(1.0, 1/steps, steps): endpoints inclusive.
    let start = 1.0_f64;
    let end = 1.0_f64 / n;
    let mut sigmas: Vec<f32> = Vec::with_capacity(num_steps + 1);
    for i in 0..num_steps {
        let t = if num_steps == 1 {
            start
        } else {
            start + (end - start) * (i as f64) / ((num_steps - 1) as f64)
        };
        // time-shift: exp(mu) / (exp(mu) + (1/t - 1))
        let s = exp_mu / (exp_mu + (1.0 / t - 1.0));
        sigmas.push(s as f32);
    }
    let timesteps: Vec<f32> = sigmas.iter().map(|&s| s * 1000.0).collect();
    sigmas.push(0.0);
    Schedule { sigmas, timesteps }
}

impl Schedule {
    /// Euler step: `latents + (sigmas[t+1] - sigmas[t]) * noise`.
    /// Returns the per-element delta coefficient `dt = sigmas[t+1] - sigmas[t]`
    /// (negative; the array combine is done by the caller with MLX ops).
    #[inline]
    pub fn dt(&self, t: usize) -> f32 {
        self.sigmas[t + 1] - self.sigmas[t]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// These used to compare against `/tmp/klein_sigmas.bin`, dumped from mflux
    /// during development. The files are long gone, so the tests failed on
    /// every machine — permanently, and without anyone noticing, because a
    /// missing file reads the same as a red test only if you look.
    ///
    /// Re-deriving the expected numbers by calling `compute()` would be
    /// circular. Instead the schedule is pinned by the properties that define
    /// it, which need no external artifact and cannot rot.
    /// The time-shift, stated as an odds ratio.
    ///
    /// `s = e^mu / (e^mu + (1/t - 1))` rearranges exactly to
    /// `s/(1-s) = e^mu · t/(1-t)` — the shift multiplies the odds by `e^mu`.
    /// Checking that form rather than re-running the expression means a typo in
    /// the implementation cannot be reproduced by the test.
    #[test]
    fn shift_multiplies_the_odds_by_exp_mu() {
        for (isl, steps) in [(256usize, 4usize), (1024, 8), (4096, 20), (8192, 28)] {
            let sched = compute(isl, steps);
            let exp_mu = (empirical_mu(isl, steps) as f64).exp();
            let n = steps as f64;
            // i = 0 has t = 1.0, where both sides of the identity are
            // infinite; `schedule_shape_and_monotonicity` pins that endpoint
            // directly as sigma == 1.
            for i in 1..steps {
                let t = 1.0 + (1.0 / n - 1.0) * (i as f64) / ((steps - 1) as f64);
                let s = sched.sigmas[i] as f64;
                let got = s / (1.0 - s);
                let want = exp_mu * t / (1.0 - t);
                let rel = (got - want).abs() / want.abs().max(1e-9);
                assert!(
                    rel < 1e-5,
                    "isl={isl} steps={steps} i={i}: odds {got:.6} != e^mu·odds(t) {want:.6}"
                );
            }
        }
    }

    #[test]
    fn schedule_shape_and_monotonicity() {
        for (isl, steps) in [(256usize, 4usize), (1024, 8), (4096, 20)] {
            let sched = compute(isl, steps);
            assert_eq!(sched.sigmas.len(), steps + 1, "sigmas = steps + 1");
            assert_eq!(sched.timesteps.len(), steps, "timesteps = steps");
            assert_eq!(*sched.sigmas.last().unwrap(), 0.0, "terminal sigma is 0");
            // sigma(1.0) = e^mu / (e^mu + 0) = 1 exactly, for any mu.
            assert!(
                (sched.sigmas[0] - 1.0).abs() < 1e-6,
                "isl={isl}: first sigma {} != 1.0",
                sched.sigmas[0]
            );
            for w in sched.sigmas.windows(2) {
                assert!(w[1] < w[0], "isl={isl}: sigmas must strictly decrease");
            }
            for t in 0..steps {
                assert!(sched.dt(t) < 0.0, "isl={isl}: dt({t}) must be negative");
            }
        }
    }

    #[test]
    fn timesteps_are_sigmas_scaled_by_1000() {
        let sched = compute(1024, 8);
        for (i, (&ts, &sg)) in sched.timesteps.iter().zip(sched.sigmas.iter()).enumerate() {
            assert!(
                (ts - sg * 1000.0).abs() < 1e-3,
                "step {i}: timestep {ts} != sigma {sg} * 1000"
            );
        }
    }

    /// `empirical_mu` is a piecewise fit with a documented switch at
    /// `image_seq_len > 4300`. Pin both branches and the fact that longer
    /// sequences shift harder — that monotonicity is what the schedule relies
    /// on, and a sign slip in the interpolation would invert it.
    #[test]
    fn empirical_mu_branches_and_monotonicity() {
        // Above the cutoff the step count drops out of the formula entirely.
        assert_eq!(
            empirical_mu(5000, 4),
            empirical_mu(5000, 50),
            "isl > 4300 must ignore num_steps"
        );
        // Below it, it does not.
        assert_ne!(
            empirical_mu(1024, 4),
            empirical_mu(1024, 50),
            "isl <= 4300 must depend on num_steps"
        );
        // Monotone *within* each branch. Not across them: at steps=20 the fit
        // steps down from mu(4096) = 2.198 to mu(8192) = 1.843. That
        // discontinuity is mflux's, faithfully ported, and asserting global
        // monotonicity would be asserting a bug into existence.
        for branch in [[256usize, 1024, 4096], [4301, 8192, 16384]] {
            let mut prev = f32::NEG_INFINITY;
            for isl in branch {
                let mu = empirical_mu(isl, 20);
                assert!(
                    mu > prev,
                    "mu must grow with image_seq_len within a branch (isl={isl})"
                );
                prev = mu;
            }
        }
        assert!(
            empirical_mu(4300, 20) > empirical_mu(4301, 20),
            "the documented step down at the 4300 cutoff must survive a refactor"
        );
    }

    #[test]
    fn single_step_schedule_is_degenerate_but_valid() {
        let sched = compute(256, 1);
        assert_eq!(sched.sigmas.len(), 2);
        assert!((sched.sigmas[0] - 1.0).abs() < 1e-6);
        assert_eq!(sched.sigmas[1], 0.0);
        assert!(sched.dt(0) < 0.0);
    }
}
