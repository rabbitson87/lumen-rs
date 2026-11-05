//! How many drafts each MTP cycle proposes.
//!
//! A deeper draft pays when the drafts it adds are accepted often enough to outweigh what they
//! cost: the drafter's steps and the wider verify. Both depend on the model, the machine, the
//! text and its context, so the controller measures them on the sequence it serves, as oMLX's
//! does, instead of trusting a constant: per depth `d` the acceptance of draft `d` given that the
//! drafts before it were accepted (`p_d`), and per draft count `k` the wall time of a cycle
//! (`t_k`). Each cycle takes the `k` with the most expected tokens per millisecond,
//! `(1 + p_1 + p_1 p_2 + ... + p_1 ... p_k) / t_k`.
//!
//! Host bookkeeping only: the cycle already waits for its verify, and the controller reads what
//! that wait returned.
#![cfg_attr(not(feature = "mlx-native"), allow(dead_code))]

/// The deepest draft a verify can check exactly: `1 + k` rows go through the row-exact matvecs,
/// which take at most four.
pub(crate) const MAX_EXACT_DRAFTS: usize = 3;

/// Draft counts per cycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MtpDepth {
    /// `k` drafts every cycle (`LUMEN_SPEC_K`).
    Fixed(usize),
    /// Chosen per cycle by [`DepthController`], from 1 to `max`.
    Auto { max: usize },
}

impl MtpDepth {
    /// The deepest count this policy can ask for.
    pub(crate) fn max(self) -> usize {
        match self {
            Self::Fixed(k) => k,
            Self::Auto { max } => max,
        }
    }
}

/// Weight of a new acceptance observation.
const ACCEPT_RATE: f64 = 0.1;
/// Weight of a new cost observation.
const COST_RATE: f64 = 0.2;
/// A cost observation counts for at most this many times the estimate, so one slow cycle (a
/// page fault, another process) does not price a depth out.
const SPIKE: f64 = 2.0;
/// Acceptance assumed for a depth not yet observed.
const PRIOR: f64 = 0.7;
/// Every this many cycles the controller runs a neighbour of its choice, so the estimates of
/// the depths it is not running do not go stale.
const EXPLORE_EVERY: u64 = 16;

pub(crate) struct DepthController {
    depth: MtpDepth,
    /// `accept[d - 1]`: `p_d`.
    accept: Vec<f64>,
    /// `cost[k - 1]`: `t_k` in milliseconds, `None` until measured.
    cost: Vec<Option<f64>>,
    cycles: u64,
    /// `ran[k - 1]`: cycles run with `k` drafts, for the log.
    ran: Vec<u64>,
    explored_up: bool,
}

impl DepthController {
    pub(crate) fn new(depth: MtpDepth) -> Self {
        let max = depth.max().max(1);
        Self {
            depth,
            accept: vec![PRIOR; max],
            cost: vec![None; max],
            cycles: 0,
            ran: vec![0; max],
            explored_up: false,
        }
    }

    /// The draft count of the next cycle.
    pub(crate) fn next(&mut self) -> usize {
        let max = match self.depth {
            MtpDepth::Fixed(k) => return k,
            MtpDepth::Auto { max } => max,
        };
        // The first cycle carries the prefill's tail and is not measured; then each count once.
        if self.cycles == 0 {
            return 1;
        }
        if let Some(k) = (1..=max).find(|&k| self.cost[k - 1].is_none()) {
            return k;
        }
        let best = self.best();
        if !self.cycles.is_multiple_of(EXPLORE_EVERY) || max == 1 {
            return best;
        }
        self.explored_up = !self.explored_up;
        match (self.explored_up, best) {
            (true, k) if k < max => k + 1,
            (_, k) if k > 1 => k - 1,
            (_, k) => k + 1,
        }
    }

    /// What the cycle that ran `k` drafts found: `accepted` of them taken, in `ms`.
    pub(crate) fn observe(&mut self, k: usize, accepted: usize, ms: f64) {
        let first = self.cycles == 0;
        self.cycles += 1;
        if k == 0 || k > self.accept.len() {
            return;
        }
        self.ran[k - 1] += 1;
        // Draft d is seen only if every draft before it was accepted.
        for d in 1..=k {
            let taken = d <= accepted;
            let p = &mut self.accept[d - 1];
            *p += ACCEPT_RATE * (f64::from(u8::from(taken)) - *p);
            if !taken {
                break;
            }
        }
        if first {
            return;
        }
        let c = &mut self.cost[k - 1];
        *c = Some(match *c {
            None => ms,
            Some(t) => t + COST_RATE * (ms.min(SPIKE * t) - t),
        });
    }

    /// Expected tokens per millisecond of `k` drafts.
    fn score(&self, k: usize) -> f64 {
        let Some(t) = self.cost[k - 1] else {
            return 0.0;
        };
        let (mut tokens, mut all) = (1.0, 1.0);
        for p in &self.accept[..k] {
            all *= p;
            tokens += all;
        }
        tokens / t.max(f64::MIN_POSITIVE)
    }

    fn best(&self) -> usize {
        (1..=self.accept.len())
            .max_by(|&a, &b| self.score(a).total_cmp(&self.score(b)))
            .unwrap_or(1)
    }

    /// Cycles run per draft count, `k1:n k2:n ...`.
    pub(crate) fn summary(&self) -> String {
        self.ran
            .iter()
            .enumerate()
            .filter(|(_, n)| **n > 0)
            .map(|(i, n)| format!("k{}:{n}", i + 1))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sequence whose drafts are accepted with probability `p` at every depth, a cycle of `k`
    /// drafts costing `base + per * k` ms; the counts the controller ran, after `n` cycles.
    fn run(p: f64, base: f64, per: f64, n: usize) -> Vec<u64> {
        let mut c = DepthController::new(MtpDepth::Auto { max: 3 });
        // A fixed pseudo-random stream, so the test is the same every run.
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut draw = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..n {
            let k = c.next();
            let accepted = (0..k).take_while(|_| draw() < p).count();
            c.observe(k, accepted, base + per * k as f64);
        }
        c.ran
    }

    /// Drafts almost always accepted and cheap to add: the deepest count wins.
    #[test]
    fn cheap_accepted_drafts_go_deep() {
        let ran = run(0.95, 80.0, 5.0, 400);
        assert!(ran[2] > ran[0] + ran[1], "ran {ran:?}");
    }

    /// Drafts mostly rejected: one draft a cycle.
    #[test]
    fn rejected_drafts_stay_shallow() {
        let ran = run(0.2, 80.0, 5.0, 400);
        assert!(ran[0] > ran[1] + ran[2], "ran {ran:?}");
    }

    /// Accepted drafts whose verify grows steeply: the wider verify is not worth it.
    #[test]
    fn an_expensive_verify_keeps_it_shallow() {
        let ran = run(0.8, 40.0, 60.0, 400);
        assert!(ran[0] > ran[1] + ran[2], "ran {ran:?}");
    }

    /// Acceptance counts draft d only when the drafts before it were accepted, so a depth the
    /// controller rarely reaches keeps its own rate rather than inheriting the rejections of
    /// the depth before it.
    #[test]
    fn acceptance_is_conditional_on_the_drafts_before() {
        let mut c = DepthController::new(MtpDepth::Auto { max: 3 });
        for _ in 0..200 {
            // The first draft is always rejected: the second and third are never seen.
            c.observe(3, 0, 100.0);
        }
        assert!(c.accept[0] < 0.01, "{:?}", c.accept);
        assert_eq!((c.accept[1], c.accept[2]), (PRIOR, PRIOR));
    }

    /// One slow cycle moves a depth's cost by at most `SPIKE` of the estimate.
    #[test]
    fn a_slow_cycle_is_damped() {
        let mut c = DepthController::new(MtpDepth::Auto { max: 1 });
        c.observe(1, 1, 50.0); // the unmeasured first cycle
        c.observe(1, 1, 50.0);
        c.observe(1, 1, 5000.0);
        let t = c.cost[0].expect("measured");
        assert!((t - (50.0 + COST_RATE * 50.0)).abs() < 1e-9, "{t}");
    }

    /// Tokens per millisecond the controller realizes over `n` cycles of the [`run`] sequence.
    fn realized_rate(p: f64, base: f64, per: f64, n: usize) -> f64 {
        let mut c = DepthController::new(MtpDepth::Auto { max: 3 });
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut draw = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut tokens, mut ms) = (0.0, 0.0);
        for _ in 0..n {
            let k = c.next();
            let accepted = (0..k).take_while(|_| draw() < p).count();
            let t = base + per * k as f64;
            c.observe(k, accepted, t);
            tokens += 1.0 + accepted as f64;
            ms += t;
        }
        tokens / ms
    }

    /// Two counts that score within a couple of percent: whichever mix the controller settles
    /// on, it gives up no more than its exploration to the better fixed count. A controller that
    /// flapped on noisy estimates of near-equal scores would pay the worse count's rate and the
    /// switching on top (issue #12's case on one checkpoint).
    #[test]
    fn near_equal_counts_cost_no_more_than_exploration() {
        // p = 0.6, cycles of 100 / 120 / 140 ms: one draft 1.6 / 100 = 0.0160 tokens/ms, two
        // 1.96 / 120 = 0.0163, three 2.18 / 140 = 0.0155.
        let (p, base, per): (f64, f64, f64) = (0.6, 80.0, 20.0);
        let fixed = |k: i32| {
            let tokens: f64 = (0..=k).map(|d| p.powi(d)).sum();
            tokens / (base + per * f64::from(k))
        };
        let best = (1..=3).map(fixed).fold(0.0, f64::max);
        let rate = realized_rate(p, base, per, 4000);
        assert!(
            rate >= 0.97 * best,
            "realized {rate:.5} against the best fixed {best:.5}"
        );
    }

    /// A fixed count is what every cycle runs.
    #[test]
    fn a_fixed_count_is_kept() {
        let mut c = DepthController::new(MtpDepth::Fixed(2));
        for _ in 0..40 {
            assert_eq!(c.next(), 2);
            c.observe(2, 0, 100.0);
        }
    }
}
