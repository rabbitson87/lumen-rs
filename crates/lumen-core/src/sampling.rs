//! Backend-agnostic token sampling.
//!
//! Implements the standard HF / llama.cpp-style sampling pipeline:
//!
//! 1. Repeat penalty over a sliding window of recently emitted tokens
//!    (divide positive logits, multiply negative — pushes mass off
//!    self-reinforcing attractors that cause infinite repetition on
//!    aggressively quantized models).
//! 2. Temperature scaling (`logits /= T`).
//! 3. Softmax with the max-subtraction stability trick.
//! 4. Top-p nucleus filtering.
//! 5. Multinomial draw via a deterministic xorshift64* PRNG.
//!
//! The whole pipeline operates on `&mut [f32]` so it can be shared
//! across backends: the GPU runtime (mlx, candle) is responsible for
//! computing per-step logits and pulling the last-position vector into
//! a CPU buffer; from there `sample_from_logits` returns the next
//! token id.

/// Tuning knobs for one sampling step. `is_greedy()` returns true when
/// the config collapses to argmax (temperature 0 AND no repeat
/// penalty), letting callers skip the CPU pipeline entirely.
#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    /// Softmax temperature. `<= 0` collapses to argmax (greedy).
    pub temperature: f32,
    /// Nucleus sampling cutoff in `(0, 1]`. `>= 1.0` disables top-p
    /// (full distribution).
    pub top_p: f32,
    /// Top-k cutoff: keep only the `k` highest-logit tokens before
    /// top-p / sampling. `0` (or `>= vocab`) disables. Mirrors Ollama's
    /// always-on `top_k=40` — bounds the candidate set so low-rank tail
    /// tokens can't be drawn on quantized weights.
    pub top_k: usize,
    /// HF / llama.cpp-style repeat penalty applied to the last
    /// `repeat_penalty_last_n` tokens. `1.0` = no penalty.
    pub repeat_penalty: f32,
    /// Sliding-window length the penalty applies to. `0` disables.
    pub repeat_penalty_last_n: usize,
    /// PRNG seed. Same prompt + seed → bit-identical output.
    pub seed: u64,
    /// DRY (Don't Repeat Yourself) penalty config. `multiplier=0.0`
    /// (default) disables DRY. Applied after the classic repeat
    /// penalty, before temperature.
    pub dry: crate::dry::DryConfig,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repeat_penalty: 1.0,
            repeat_penalty_last_n: 64,
            seed: 0,
            dry: crate::dry::DryConfig::default(),
        }
    }
}

impl SamplingConfig {
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
            && (self.repeat_penalty - 1.0).abs() < 1e-6
            && self.dry.is_disabled()
    }
}

/// xorshift64* — deterministic, no external crate, perfectly adequate
/// for token sampling. Cryptographic strength is not a goal; cheap
/// seeded reproducibility is.
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    pub fn new(seed: u64) -> Self {
        // Reject the all-zero state — xorshift would lock at 0.
        let state = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform `f32` in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }
}

/// HF-style repeat penalty applied in place: divide positive logits and
/// multiply negative logits of recently-emitted tokens by `penalty`.
/// Pushes probability mass off repeated-token attractors that
/// catastrophically dominate greedy decoding on aggressive 3-bit
/// quantization (the original `karın-karın-...` bug class).
pub fn apply_repeat_penalty(logits: &mut [f32], recent: &[u32], penalty: f32) {
    if (penalty - 1.0).abs() < 1e-6 {
        return;
    }
    for &tok in recent {
        let i = tok as usize;
        if i >= logits.len() {
            continue;
        }
        let v = logits[i];
        logits[i] = if v >= 0.0 { v / penalty } else { v * penalty };
    }
}

/// Top-k logit clamp: keep the `k` highest logits, set the rest to
/// `f32::NEG_INFINITY` (softmax later zeroes them). `k == 0` or
/// `k >= len` is a no-op. Uses an O(n) quickselect to find the keep
/// threshold rather than a full sort. Ties at the threshold are all
/// kept, so the surviving set may slightly exceed `k` — the same
/// lenient behavior as llama.cpp / Ollama.
pub fn apply_top_k(logits: &mut [f32], k: usize) {
    let n = logits.len();
    if k == 0 || k >= n {
        return;
    }
    // Copy values for partial selection — we must not reorder `logits`
    // itself, since positions ARE token ids. Ascending select: the
    // element at index `n - k` is the k-th largest = keep threshold.
    let mut vals: Vec<f32> = logits.to_vec();
    let idx = n - k;
    vals.select_nth_unstable_by(idx, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = vals[idx];
    for v in logits.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// In-place softmax with the standard max-subtraction trick for
/// numerical stability. After this call `logits` is a valid probability
/// distribution summing to ~1.0. Falls back to uniform on degenerate
/// input (all `-inf`) instead of producing NaNs.
pub fn softmax_inplace(logits: &mut [f32]) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        let u = 1.0 / logits.len() as f32;
        for v in logits.iter_mut() {
            *v = u;
        }
        return;
    }
    let mut sum = 0.0_f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum <= 0.0 {
        let u = 1.0 / logits.len() as f32;
        for v in logits.iter_mut() {
            *v = u;
        }
        return;
    }
    let inv = 1.0 / sum;
    for v in logits.iter_mut() {
        *v *= inv;
    }
}

/// Sample a token id from `probs` after applying top-p nucleus
/// filtering. `probs` must sum to ~1 (call `softmax_inplace` first).
/// `top_p >= 1.0` skips the filter and samples from the full
/// distribution. Always keeps at least one token (the argmax) so a
/// degenerate `top_p = 0` doesn't deadlock.
pub fn sample_top_p(probs: &[f32], top_p: f32, rng: &mut Xorshift64) -> u32 {
    debug_assert!(!probs.is_empty());

    if top_p >= 1.0 || top_p <= 0.0 {
        return categorical(probs, rng);
    }

    let mut idx: Vec<u32> = (0..probs.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| {
        probs[b as usize]
            .partial_cmp(&probs[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut cum = 0.0_f32;
    let mut cutoff = idx.len();
    for (rank, &i) in idx.iter().enumerate() {
        cum += probs[i as usize];
        if cum >= top_p {
            cutoff = rank + 1;
            break;
        }
    }
    cutoff = cutoff.max(1);

    let kept = &idx[..cutoff];
    let mass: f32 = kept.iter().map(|&i| probs[i as usize]).sum();
    if mass <= 0.0 {
        return kept[0];
    }
    let r = rng.next_f32() * mass;
    let mut acc = 0.0_f32;
    for &i in kept {
        acc += probs[i as usize];
        if r < acc {
            return i;
        }
    }
    *kept.last().unwrap()
}

fn categorical(probs: &[f32], rng: &mut Xorshift64) -> u32 {
    let r = rng.next_f32();
    let mut acc = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// One-shot helper: run the full pipeline (penalty → temperature →
/// softmax → top-p → sample) on a CPU logit buffer. Mutates `logits`
/// in place (caller may discard or reuse). The caller owns
/// `recent_tokens` (sliding window for the repeat penalty).
pub fn sample_from_logits(
    logits: &mut [f32],
    recent_tokens: &[u32],
    cfg: &SamplingConfig,
    rng: &mut Xorshift64,
) -> u32 {
    // Repeat penalty restricted to the trailing window — older context
    // shouldn't dampen tokens we naturally want to emit again.
    let n = cfg.repeat_penalty_last_n.min(recent_tokens.len());
    if cfg.repeat_penalty != 1.0 && n > 0 {
        let window = &recent_tokens[recent_tokens.len() - n..];
        apply_repeat_penalty(logits, window, cfg.repeat_penalty);
    }

    // DRY (Don't Repeat Yourself): catches n-gram cycles the classic
    // repeat penalty's single-token view misses. No-op when disabled.
    crate::dry::apply_dry_penalty(logits, recent_tokens, &cfg.dry);

    // Top-k clamp BEFORE temperature/top-p so the nucleus is drawn from
    // a bounded candidate set (matches Ollama's always-on top_k=40 and
    // stops low-rank tail-token drift on quantized weights). No-op at 0.
    apply_top_k(logits, cfg.top_k);

    // Temperature scaling before softmax. `<=0` would mean greedy but
    // the caller is responsible for routing greedy elsewhere; clamp to
    // a tiny epsilon as a safety net.
    let t = cfg.temperature.max(1e-5);
    if (t - 1.0).abs() > 1e-6 {
        let inv = 1.0 / t;
        for v in logits.iter_mut() {
            *v *= inv;
        }
    }

    softmax_inplace(logits);
    sample_top_p(logits, cfg.top_p, rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xorshift_deterministic() {
        let mut a = Xorshift64::new(42);
        let mut b = Xorshift64::new(42);
        for _ in 0..16 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut v = vec![1.0_f32, 2.0, 3.0, 4.0];
        softmax_inplace(&mut v);
        let s: f32 = v.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_handles_neg_inf() {
        let mut v = vec![f32::NEG_INFINITY; 4];
        softmax_inplace(&mut v);
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn repeat_penalty_pushes_positive_down() {
        let mut v = vec![2.0_f32, 0.5, -0.5, -2.0];
        apply_repeat_penalty(&mut v, &[0, 2], 1.5);
        assert!(v[0] < 2.0, "positive logit should be divided");
        assert!(v[2] < -0.5, "negative logit should be multiplied");
        assert_eq!(v[1], 0.5);
        assert_eq!(v[3], -2.0);
    }

    #[test]
    fn top_p_keeps_at_least_one() {
        let probs = vec![0.4, 0.3, 0.2, 0.1];
        let mut rng = Xorshift64::new(1);
        let tok = sample_top_p(&probs, 0.0, &mut rng);
        assert!(tok < probs.len() as u32);
    }

    #[test]
    fn top_k_keeps_only_k_highest() {
        let mut logits = vec![1.0_f32, 5.0, 3.0, 2.0, 4.0];
        apply_top_k(&mut logits, 2);
        // Top-2 are 5.0 (idx 1) and 4.0 (idx 4); the rest become -inf.
        assert_eq!(logits[1], 5.0);
        assert_eq!(logits[4], 4.0);
        assert!(logits[0] == f32::NEG_INFINITY);
        assert!(logits[2] == f32::NEG_INFINITY);
        assert!(logits[3] == f32::NEG_INFINITY);
    }

    #[test]
    fn top_k_disabled_is_noop() {
        let orig = vec![1.0_f32, 2.0, 3.0];
        let mut a = orig.clone();
        apply_top_k(&mut a, 0); // 0 = disabled
        assert_eq!(a, orig);
        let mut b = orig.clone();
        apply_top_k(&mut b, 99); // k >= len = disabled
        assert_eq!(b, orig);
    }

    #[test]
    fn greedy_flag() {
        let g = SamplingConfig::default();
        assert!(g.is_greedy());
        let s = SamplingConfig {
            temperature: 0.7,
            ..g
        };
        assert!(!s.is_greedy());
    }

    #[test]
    fn end_to_end_seeded_is_deterministic() {
        let mut probs = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let cfg = SamplingConfig {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 0,
            repeat_penalty: 1.0,
            repeat_penalty_last_n: 0,
            seed: 12345,
            dry: crate::dry::DryConfig::default(),
        };
        let mut rng1 = Xorshift64::new(cfg.seed);
        let t1 = sample_from_logits(&mut probs.clone(), &[], &cfg, &mut rng1);
        let mut rng2 = Xorshift64::new(cfg.seed);
        let t2 = sample_from_logits(&mut probs, &[], &cfg, &mut rng2);
        assert_eq!(t1, t2);
    }
}
