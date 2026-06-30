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

    fn read_f32_bin(path: &str) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        assert_eq!(bytes.len() % 4, 0, "{path} not f32-aligned");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Pure-scalar test (no MLX) — safe in the sandbox.
    #[test]
    fn scheduler_matches_reference() {
        // 256x256 → image_seq_len = (256/16)*(256/16) = 256, steps = 4.
        let sched = compute(256, 4);
        let exp_sigmas = read_f32_bin("/tmp/klein_sigmas.bin");
        let exp_timesteps = read_f32_bin("/tmp/klein_timesteps.bin");
        assert_eq!(sched.sigmas.len(), exp_sigmas.len(), "sigmas len");
        assert_eq!(sched.timesteps.len(), exp_timesteps.len(), "timesteps len");

        let mut max_abs = 0.0f32;
        for (&g, &e) in sched.sigmas.iter().zip(exp_sigmas.iter()) {
            max_abs = max_abs.max((g - e).abs());
        }
        for (&g, &e) in sched.timesteps.iter().zip(exp_timesteps.iter()) {
            // timesteps are ×1000; compare in sigma units for a fair tol.
            max_abs = max_abs.max(((g - e) / 1000.0).abs());
        }
        println!("scheduler max_abs (sigma units) = {max_abs:.3e}");
        println!("  rust sigmas    = {:?}", sched.sigmas);
        println!("  mflux sigmas   = {exp_sigmas:?}");
        println!("  rust timesteps = {:?}", sched.timesteps);
        assert!(max_abs < 1e-4, "scheduler max_abs {max_abs:.3e} >= 1e-4");
    }
}
