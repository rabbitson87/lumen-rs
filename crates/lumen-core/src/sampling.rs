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
    /// OpenAI presence penalty: subtract this flat amount from any token
    /// seen in the window at least once. `0.0` disables. Applied alongside
    /// the classic repeat penalty over the same window.
    pub presence_penalty: f32,
    /// OpenAI frequency penalty: subtract `count * frequency_penalty` from
    /// each token by its occurrence count in the window. `0.0` disables.
    pub frequency_penalty: f32,
    /// Min-p cutoff: keep only tokens with `prob >= min_p * max_prob`
    /// (i.e. `logit >= max_logit + ln(min_p)`). `0.0` disables. Applied
    /// after top-k, before temperature. The argmax always survives, so this
    /// never changes greedy output.
    pub min_p: f32,
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
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            min_p: 0.0,
            seed: 0,
            dry: crate::dry::DryConfig::default(),
        }
    }
}

impl SamplingConfig {
    pub fn is_greedy(&self) -> bool {
        // min_p never changes the argmax, so it is intentionally excluded.
        self.temperature <= 0.0
            && (self.repeat_penalty - 1.0).abs() < 1e-6
            && self.presence_penalty == 0.0
            && self.frequency_penalty == 0.0
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

/// OpenAI presence/frequency penalties over a token window, applied in
/// place: `logit -= count * frequency + (count > 0) * presence`. Mirrors
/// llama.cpp's penalty sampler. No-op when both are 0.
pub fn apply_presence_frequency_penalty(
    logits: &mut [f32],
    window: &[u32],
    presence: f32,
    frequency: f32,
) {
    if presence == 0.0 && frequency == 0.0 {
        return;
    }
    let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for &tok in window {
        *counts.entry(tok).or_insert(0) += 1;
    }
    for (tok, c) in counts {
        let i = tok as usize;
        if i < logits.len() {
            logits[i] -= frequency * c as f32 + presence;
        }
    }
}

/// Min-p filter applied in place: drop tokens whose logit is below
/// `max_logit + ln(min_p)` (equivalently `prob < min_p * max_prob`) by
/// setting them to `-inf`. `min_p <= 0` is a no-op. The peak token always
/// survives, so this never alters greedy/argmax output.
pub fn apply_min_p(logits: &mut [f32], min_p: f32) {
    if min_p <= 0.0 {
        return;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return;
    }
    let thresh = max + min_p.ln();
    for v in logits.iter_mut() {
        if *v < thresh {
            *v = f32::NEG_INFINITY;
        }
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
    // Unreachable: `max` is finite (guarded above), so `exp(max - max) == 1`
    // is always one of the terms. Kept because the alternative is dividing by
    // a sum nothing proves non-zero.
    if crate::never!(sum <= 0.0) {
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

    let kept = nucleus(probs, top_p);
    let mass: f32 = kept.iter().map(|&i| probs[i as usize]).sum();
    if mass <= 0.0 {
        return kept[0];
    }
    let r = rng.next_f32() * mass;
    let mut acc = 0.0_f32;
    for &i in &kept {
        acc += probs[i as usize];
        if r < acc {
            return i;
        }
    }
    *kept.last().unwrap()
}

/// Bits of an `f32` dropped to name its top-p bucket. What is left is the
/// exponent and the top three mantissa bits, so a bucket spans an eighth of an
/// octave; positive floats order like their bit patterns, so buckets order like
/// the probabilities in them.
const NUCLEUS_BUCKET_SHIFT: u32 = 20;
/// A bucket for every rank key, `0.0` up to infinity (a key never has its sign
/// bit set).
const NUCLEUS_BUCKETS: usize = 1 << (31 - NUCLEUS_BUCKET_SHIFT);

/// What top-p ranks a token by: its probability, with zero, negative and NaN
/// entries all ranked as zero, so the order is total.
#[inline]
fn rank_key(p: f32) -> f32 {
    if p > 0.0 { p } else { 0.0 }
}

#[inline]
fn nucleus_bucket(p: f32) -> usize {
    (rank_key(p).to_bits() >> NUCLEUS_BUCKET_SHIFT) as usize
}

/// Top-p ranks tokens most probable first, the lower id first on a tie.
fn rank_order(probs: &[f32], a: u32, b: u32) -> std::cmp::Ordering {
    rank_key(probs[b as usize])
        .total_cmp(&rank_key(probs[a as usize]))
        .then(a.cmp(&b))
}

/// Where the top-p nucleus ends, as `(id, rank key)` of its last token: the
/// nucleus is the shortest run of the vocabulary in rank order whose mass
/// reaches `top_p`. `None` when no run does, so the nucleus is everything.
///
/// Only one bucket is sorted. A histogram of the mass by bucket names the
/// bucket the run ends in: every token above it is in the nucleus and every
/// token below it out, so the cut is found by ranking that bucket alone. Sums
/// are `f64`, so they differ from exact ones by rounding alone; an `f32` running
/// sum drifts by more than a small token's mass over a long run.
fn nucleus_end(probs: &[f32], top_p: f32) -> Option<(u32, f32)> {
    let mut mass = [0.0_f64; NUCLEUS_BUCKETS];
    for &p in probs {
        mass[nucleus_bucket(p)] += f64::from(rank_key(p));
    }
    let target = f64::from(top_p);
    let mut above = 0.0_f64;
    let mut floor = None;
    for b in (0..NUCLEUS_BUCKETS).rev() {
        if above + mass[b] >= target {
            floor = Some(b);
            break;
        }
        above += mass[b];
    }
    let floor = floor?;
    let mut boundary: Vec<u32> = (0..probs.len() as u32)
        .filter(|&i| nucleus_bucket(probs[i as usize]) == floor)
        .collect();
    boundary.sort_unstable_by(|&a, &b| rank_order(probs, a, b));
    let mut cum = above;
    // The bucket's own sum reached `target` in the histogram; summed again in
    // rank order it can fall short by rounding, and then the whole bucket is in.
    let last = boundary
        .iter()
        .position(|&i| {
            cum += f64::from(rank_key(probs[i as usize]));
            cum >= target
        })
        .map_or(boundary.last().copied(), |r| Some(boundary[r]))?;
    Some((last, rank_key(probs[last as usize])))
}

/// Whether a token with rank key `key` is in the nucleus that ends at `end`.
#[inline]
fn in_nucleus(id: u32, key: f32, (end, end_key): (u32, f32)) -> bool {
    key > end_key || (key == end_key && id <= end)
}

/// The top-p nucleus in rank order (see [`nucleus_end`]).
fn nucleus(probs: &[f32], top_p: f32) -> Vec<u32> {
    let mut kept: Vec<u32> = match nucleus_end(probs, top_p) {
        Some(end) => (0..probs.len() as u32)
            .filter(|&i| in_nucleus(i, rank_key(probs[i as usize]), end))
            .collect(),
        None => (0..probs.len() as u32).collect(),
    };
    kept.sort_unstable_by(|&a, &b| rank_order(probs, a, b));
    kept
}

/// Zero every probability outside the top-p nucleus, in place.
fn mask_outside_nucleus(probs: &mut [f32], top_p: f32) {
    let Some(end) = nucleus_end(probs, top_p) else {
        return;
    };
    for (i, p) in probs.iter_mut().enumerate() {
        if !in_nucleus(i as u32, rank_key(*p), end) {
            *p = 0.0;
        }
    }
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

/// The first index of the largest logit.
fn argmax_lowest(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
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
    // Penalties restricted to the trailing window — older context shouldn't
    // dampen tokens we naturally want to emit again. Repeat (sign-aware) +
    // OpenAI presence/frequency share the same window.
    let n = cfg.repeat_penalty_last_n.min(recent_tokens.len());
    if n > 0 {
        let window = &recent_tokens[recent_tokens.len() - n..];
        if cfg.repeat_penalty != 1.0 {
            apply_repeat_penalty(logits, window, cfg.repeat_penalty);
        }
        if cfg.presence_penalty != 0.0 || cfg.frequency_penalty != 0.0 {
            apply_presence_frequency_penalty(
                logits,
                window,
                cfg.presence_penalty,
                cfg.frequency_penalty,
            );
        }
    }

    // DRY (Don't Repeat Yourself): catches n-gram cycles the classic
    // repeat penalty's single-token view misses. No-op when disabled.
    crate::dry::apply_dry_penalty(logits, recent_tokens, &cfg.dry);

    // Top-k clamp BEFORE temperature/top-p so the nucleus is drawn from
    // a bounded candidate set (matches Ollama's always-on top_k=40 and
    // stops low-rank tail-token drift on quantized weights). No-op at 0.
    apply_top_k(logits, cfg.top_k);

    // Min-p: drop tokens far below the peak. No-op at 0; never removes the
    // argmax, so greedy output is unaffected.
    apply_min_p(logits, cfg.min_p);

    // Temperature 0 is the most likely token — and the same one on every run.
    // Scaling by 1/ε and drawing would still split an exact tie at the top by
    // the RNG, which is seeded from the clock unless the request names a seed:
    // two runs of one temperature-0 request then disagree. Ties go to the
    // lowest id, as MLX's argmax breaks them.
    if cfg.temperature <= 0.0 {
        return argmax_lowest(logits);
    }

    // Temperature scaling before softmax.
    let t = cfg.temperature;
    if (t - 1.0).abs() > 1e-6 {
        let inv = 1.0 / t;
        for v in logits.iter_mut() {
            *v *= inv;
        }
    }

    softmax_inplace(logits);
    sample_top_p(logits, cfg.top_p, rng)
}

// ─────────────────────────────────────────────────────────────────────────
// Speculative (Leviathan/Chen) rejection-sampling acceptance.
//
// Self-speculative decoding (MTP, draft-model, n-gram) proposes draft tokens
// and verifies them in one batched forward. The acceptance rule that keeps the
// output distribution EXACTLY equal to plain sampling at the given temperature
// is the Leviathan/Chen theorem:
//
//   - the draft token `d` must be SAMPLED from the draft distribution `q`
//     (NOT argmax — argmax makes q one-hot and accept degenerates to
//     exact-match, i.e. greedy);
//   - accept `d` with probability `min(1, p[d] / q[d])` where `p` is the
//     target (trunk-verify) distribution;
//   - on reject, emit a correction sampled from the residual `(p - q)_+`
//     renormalized;
//   - after the last accepted position, the bonus token is sampled from `p`.
//
// `p` and `q` MUST be the same post-processed sampling distribution (same
// temperature / top-k / top-p), or the ratio is meaningless. Mirrors MTPLX
// `mtplx/sampling.py` (`acceptance_probability`, `residual_distribution`).
// ─────────────────────────────────────────────────────────────────────────

/// Build the model's true sampling distribution from raw logits: applies the
/// full pipeline (penalty → top-k → min-p → temperature → softmax → top-p
/// nucleus mask) and returns a normalized full-vocab probability vector. Tokens
/// removed by top-k / top-p are exactly `0.0`. Use this for BOTH the target `p`
/// (verify logits) and the draft `q` (drafter logits) so the speculative ratio
/// is well-defined. `recent` feeds the repeat/presence/frequency penalties;
/// pass an empty slice to disable them (recommended for spec drafts, where a
/// per-position penalty window is hard to keep consistent between p and q).
pub fn sampling_distribution(logits: &mut [f32], recent: &[u32], cfg: &SamplingConfig) -> Vec<f32> {
    let n = cfg.repeat_penalty_last_n.min(recent.len());
    if n > 0 {
        let window = &recent[recent.len() - n..];
        if cfg.repeat_penalty != 1.0 {
            apply_repeat_penalty(logits, window, cfg.repeat_penalty);
        }
        if cfg.presence_penalty != 0.0 || cfg.frequency_penalty != 0.0 {
            apply_presence_frequency_penalty(
                logits,
                window,
                cfg.presence_penalty,
                cfg.frequency_penalty,
            );
        }
    }
    crate::dry::apply_dry_penalty(logits, recent, &cfg.dry);
    apply_top_k(logits, cfg.top_k);
    apply_min_p(logits, cfg.min_p);
    let t = cfg.temperature.max(1e-5);
    if (t - 1.0).abs() > 1e-6 {
        let inv = 1.0 / t;
        for v in logits.iter_mut() {
            *v *= inv;
        }
    }
    softmax_inplace(logits);
    // Top-p nucleus as a MASK (zero the tail, renormalize) rather than a draw,
    // so the returned vector is the exact distribution sampled from.
    let top_p = cfg.top_p;
    if top_p < 1.0 && top_p > 0.0 {
        mask_outside_nucleus(logits, top_p);
        let mass: f32 = logits.iter().sum();
        if mass > 0.0 {
            let inv = 1.0 / mass;
            for v in logits.iter_mut() {
                *v *= inv;
            }
        }
    }
    logits.to_vec()
}

/// Sample an index from a probability vector that need not sum to 1
/// (renormalizes by total mass). Falls back to argmax on degenerate (all-zero)
/// input so it never deadlocks.
pub fn sample_distribution(probs: &[f32], rng: &mut Xorshift64) -> u32 {
    let mass: f32 = probs.iter().sum();
    // `!(mass > 0.0)` and NOT `mass <= 0.0`: the two differ on NaN, where
    // the first is true and the second false. NaN mass is exactly the
    // degenerate input this guard exists for — a NaN logit anywhere in the
    // vector poisons the sum — so clippy's suggestion here would reintroduce
    // the fallthrough it is meant to prevent.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    let degenerate = !(mass > 0.0);
    if degenerate {
        return probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }
    let r = rng.next_f32() * mass;
    let mut acc = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// Outcome of one speculative acceptance test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecAccept {
    /// Whether the draft token was accepted.
    pub accepted: bool,
    /// The token to commit: the draft itself if accepted, else a correction
    /// sampled from the residual `(p - q)_+`.
    pub token: u32,
}

/// Leviathan/Chen single-draft acceptance: accept `draft` with probability
/// `min(1, p[draft]/q[draft])`; on reject, return a correction sampled from the
/// residual `(p - q)_+`. `p`/`q` must be the same post-processed distribution
/// (see [`sampling_distribution`]). Equivalent to MTPLX `accept_or_correct`.
pub fn speculative_accept(p: &[f32], q: &[f32], draft: u32, rng: &mut Xorshift64) -> SpecAccept {
    let d = draft as usize;
    let pp = p.get(d).copied().unwrap_or(0.0);
    let qq = q.get(d).copied().unwrap_or(0.0);
    let accept_p = if qq <= 0.0 {
        if pp > 0.0 { 1.0 } else { 0.0 }
    } else {
        (pp / qq).min(1.0)
    };
    if rng.next_f32() <= accept_p {
        return SpecAccept {
            accepted: true,
            token: draft,
        };
    }
    SpecAccept {
        accepted: false,
        token: sample_residual(p, q, rng),
    }
}

/// Sample from the residual distribution `(p - q)_+` renormalized. Falls back
/// to `argmax(p)` if the residual is degenerate (e.g. q dominates p everywhere).
pub fn sample_residual(p: &[f32], q: &[f32], rng: &mut Xorshift64) -> u32 {
    let n = p.len();
    let mut resid = vec![0.0_f32; n];
    let mut mass = 0.0_f32;
    for i in 0..n {
        let r = p[i] - q.get(i).copied().unwrap_or(0.0);
        if r > 0.0 {
            resid[i] = r;
            mass += r;
        }
    }
    // `!(mass > 0.0)` and NOT `mass <= 0.0`: the two differ on NaN, where
    // the first is true and the second false. NaN mass is exactly the
    // degenerate input this guard exists for — a NaN logit anywhere in the
    // vector poisons the sum — so clippy's suggestion here would reintroduce
    // the fallthrough it is meant to prevent.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    let degenerate = !(mass > 0.0);
    if degenerate {
        return p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }
    sample_distribution(&resid, rng)
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
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            min_p: 0.0,
            seed: 12345,
            dry: crate::dry::DryConfig::default(),
        };
        let mut rng1 = Xorshift64::new(cfg.seed);
        let t1 = sample_from_logits(&mut probs.clone(), &[], &cfg, &mut rng1);
        let mut rng2 = Xorshift64::new(cfg.seed);
        let t2 = sample_from_logits(&mut probs, &[], &cfg, &mut rng2);
        assert_eq!(t1, t2);
    }

    /// Gemma 4 temperature-0 requests gave 2-3 distinct outputs across runs of
    /// one binary: penalties keep them off the greedy path, so they came here
    /// and an exact bf16 tie at the top was settled by a clock-seeded draw.
    #[test]
    fn temperature_zero_breaks_a_tie_the_same_way_on_every_seed() {
        let cfg = SamplingConfig {
            temperature: 0.0,
            top_p: 0.95,
            repeat_penalty: 1.1,
            ..Default::default()
        };
        for seed in 0..200u64 {
            let mut rng = Xorshift64::new(seed);
            let mut logits = vec![0.5_f32, 3.0, 3.0, 1.0];
            assert_eq!(
                sample_from_logits(&mut logits, &[], &cfg, &mut rng),
                1,
                "seed {seed}"
            );
        }
    }

    #[test]
    fn temperature_zero_still_applies_the_penalty_first() {
        // Token 1 leads, but it was just emitted and a 1.1 penalty drops it
        // below token 2.
        let cfg = SamplingConfig {
            temperature: 0.0,
            repeat_penalty: 1.1,
            ..Default::default()
        };
        let mut logits = vec![0.0_f32, 3.2, 3.0];
        let mut rng = Xorshift64::new(1);
        assert_eq!(sample_from_logits(&mut logits, &[1], &cfg, &mut rng), 2);
    }

    #[test]
    fn min_p_keeps_peak_prunes_tail() {
        // logits: peak at idx 3. min_p=0.5 → threshold = max + ln(0.5).
        let mut l = vec![0.0_f32, 1.0, 2.0, 5.0];
        apply_min_p(&mut l, 0.5);
        assert!(l[3].is_finite(), "argmax must always survive min_p");
        // ln(0.5) ≈ -0.693, threshold ≈ 4.307 → only idx 3 survives.
        assert_eq!(l[0], f32::NEG_INFINITY);
        assert_eq!(l[2], f32::NEG_INFINITY);
        // min_p = 0 is a no-op.
        let mut l2 = vec![0.0_f32, 1.0];
        apply_min_p(&mut l2, 0.0);
        assert_eq!(l2, vec![0.0, 1.0]);
    }

    #[test]
    fn presence_frequency_penalty_counts() {
        // token 1 appears twice, token 2 once.
        let window = [1u32, 1, 2];
        let mut l = vec![0.0_f32; 4];
        apply_presence_frequency_penalty(&mut l, &window, 0.5, 0.1);
        // token 1: -(0.1*2 + 0.5) = -0.7 ; token 2: -(0.1*1 + 0.5) = -0.6
        assert!((l[1] - (-0.7)).abs() < 1e-5);
        assert!((l[2] - (-0.6)).abs() < 1e-5);
        assert_eq!(l[0], 0.0); // untouched
        assert_eq!(l[3], 0.0);
    }

    // ── Speculative rejection-sampling acceptance ──────────────────────────

    #[test]
    fn spec_accept_certain_when_p_ge_q() {
        // p[d] >= q[d] → accept_p = 1 → always accepted regardless of rng.
        let p = vec![0.6_f32, 0.4];
        let q = vec![0.3_f32, 0.7];
        let mut rng = Xorshift64::new(7);
        for _ in 0..50 {
            let out = speculative_accept(&p, &q, 0, &mut rng);
            assert!(out.accepted && out.token == 0);
        }
    }

    #[test]
    fn spec_reject_correction_from_residual() {
        // q wildly over-proposes token 0 (q0=0.9) vs p0=0.1 → mostly rejected;
        // residual (p-q)+ has all mass on token 1 → correction must be 1.
        let p = vec![0.1_f32, 0.9];
        let q = vec![0.9_f32, 0.1];
        let mut rng = Xorshift64::new(3);
        let mut rejects = 0;
        for _ in 0..200 {
            let out = speculative_accept(&p, &q, 0, &mut rng);
            if !out.accepted {
                rejects += 1;
                assert_eq!(out.token, 1, "residual mass is all on token 1");
            }
        }
        assert!(rejects > 100, "p0/q0=1/9 → ~89% reject, got {rejects}/200");
    }

    /// THE invariant: committed-token distribution == target p. Monte-Carlo a
    /// single spec position (draft sampled from q, accept min(1,p/q), else
    /// residual) and confirm the empirical commit histogram matches p.
    #[test]
    fn spec_acceptance_reproduces_target_distribution() {
        let p = vec![0.5_f32, 0.3, 0.15, 0.05];
        let q = vec![0.1_f32, 0.2, 0.3, 0.4]; // deliberately mismatched draft
        let n = p.len();
        let mut rng = Xorshift64::new(0xC0FFEE);
        let iters = 200_000;
        let mut hist = vec![0u32; n];
        for _ in 0..iters {
            let draft = sample_distribution(&q, &mut rng); // draft ~ q
            let out = speculative_accept(&p, &q, draft, &mut rng);
            hist[out.token as usize] += 1;
        }
        for i in 0..n {
            let emp = hist[i] as f32 / iters as f32;
            assert!(
                (emp - p[i]).abs() < 0.01,
                "token {i}: empirical {emp:.4} vs target {:.4}",
                p[i]
            );
        }
    }

    #[test]
    fn sampling_distribution_top_p_masks_and_renormalizes() {
        // Strong peak; top_p=0.6 should keep only the top token(s) and renorm.
        let mut logits = vec![3.0_f32, 1.0, 0.5, 0.0];
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_p: 0.6,
            ..SamplingConfig::default()
        };
        let dist = sampling_distribution(&mut logits, &[], &cfg);
        let sum: f32 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "renormalized to 1, got {sum}");
        // The peak (idx 0) softmax prob > 0.6 alone, so only idx 0 survives.
        assert!(dist[0] > 0.99, "nucleus collapses to peak: {dist:?}");
        assert_eq!(dist[3], 0.0);
    }
}

#[cfg(test)]
mod nucleus_tests {
    use super::*;

    const TOP_PS: [f32; 8] = [1e-4, 0.1, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999_999];

    /// Every token in the order its definition names: most probable first,
    /// the lower id first on a tie, zero / negative / NaN ranked as zero.
    fn ranked_by_definition(probs: &[f32]) -> Vec<u32> {
        let key = |i: u32| {
            let p = probs[i as usize];
            if p > 0.0 { p } else { 0.0 }
        };
        let mut ids: Vec<u32> = (0..probs.len() as u32).collect();
        ids.sort_by(|&a, &b| key(b).partial_cmp(&key(a)).unwrap().then(a.cmp(&b)));
        ids
    }

    /// The shortest run of `ranked` whose mass reaches `top_p`, or all of it.
    fn cut_by_definition<'a>(probs: &[f32], ranked: &'a [u32], top_p: f32) -> &'a [u32] {
        let mut cum = 0.0_f64;
        let len = ranked
            .iter()
            .position(|&i| {
                cum += f64::from(probs[i as usize].max(0.0));
                cum >= f64::from(top_p)
            })
            .map_or(ranked.len(), |r| r + 1);
        &ranked[..len]
    }

    fn gaussian(rng: &mut Xorshift64) -> f32 {
        let u1 = rng.next_f32().max(1e-7);
        let u2 = rng.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    fn softmax(mut logits: Vec<f32>) -> Vec<f32> {
        softmax_inplace(&mut logits);
        logits
    }

    /// Probability vectors of the shapes a sampler meets: flat to peaked,
    /// logits on a coarse grid (bf16 logits tie constantly), a masked tail,
    /// a power law, and one flat vector.
    fn distributions() -> Vec<(String, Vec<f32>)> {
        let mut rng = Xorshift64::new(7);
        let mut out = Vec::new();
        for n in [1usize, 2, 7, 1000, 50_000] {
            for scale in [0.3_f32, 1.0, 3.0, 8.0] {
                let logits: Vec<f32> = (0..n).map(|_| gaussian(&mut rng) * scale).collect();
                let tied: Vec<f32> = logits.iter().map(|l| (l * 8.0).round() / 8.0).collect();
                let masked: Vec<f32> = logits
                    .iter()
                    .enumerate()
                    .map(|(i, &l)| if i % 3 == 1 { f32::NEG_INFINITY } else { l })
                    .collect();
                out.push((format!("gaussian n={n} scale={scale}"), softmax(logits)));
                out.push((format!("tied n={n} scale={scale}"), softmax(tied)));
                out.push((format!("masked n={n} scale={scale}"), softmax(masked)));
            }
        }
        let zipf: Vec<f32> = (0..20_000)
            .map(|i| -1.1 * ((i * 7919 % 20_000) as f32 + 1.0).ln())
            .collect();
        out.push(("zipf".into(), softmax(zipf)));
        out.push(("flat".into(), vec![1.0 / 4096.0; 4096]));
        out
    }

    #[test]
    fn the_nucleus_is_the_run_its_definition_names() {
        for (name, probs) in distributions() {
            let ranked = ranked_by_definition(&probs);
            for top_p in TOP_PS {
                let kept = cut_by_definition(&probs, &ranked, top_p);
                assert_eq!(nucleus(&probs, top_p), kept, "{name}, top_p={top_p}");
                let mut masked = probs.clone();
                mask_outside_nucleus(&mut masked, top_p);
                let mut expected = vec![0.0; probs.len()];
                for &i in kept {
                    expected[i as usize] = probs[i as usize];
                }
                assert_eq!(masked, expected, "mask: {name}, top_p={top_p}");
            }
        }
    }

    /// Tokens too small for an `f32` running sum still count toward `top_p`.
    /// Each of the hundred is under half an ulp of 0.9, so an `f32` sum stalls
    /// there and would keep the whole vocabulary, the 1e-12 tail included; their
    /// mass passes `top_p` after 41 of them.
    #[test]
    fn small_probabilities_add_up_to_the_cut() {
        let mut probs = vec![0.9_f32];
        probs.extend(std::iter::repeat_n(2.5e-8_f32, 100));
        probs.extend(std::iter::repeat_n(1e-12_f32, 10));
        let top_p = 0.900_001;
        let kept = nucleus(&probs, top_p);
        assert_eq!(kept, (0..=41).collect::<Vec<u32>>());
        assert_eq!(
            kept,
            cut_by_definition(&probs, &ranked_by_definition(&probs), top_p)
        );
    }

    /// The distribution a sampler is meant to draw from: the nucleus by its
    /// definition, renormalized.
    fn intended(probs: &[f32], top_p: f32) -> Vec<f64> {
        let ranked = ranked_by_definition(probs);
        let kept = cut_by_definition(probs, &ranked, top_p);
        let mass: f64 = kept.iter().map(|&i| f64::from(probs[i as usize])).sum();
        let mut q = vec![0.0; probs.len()];
        for &i in kept {
            q[i as usize] = f64::from(probs[i as usize]) / mass;
        }
        q
    }

    /// Kolmogorov–Smirnov distance from uniform of the draws' randomized
    /// probability integral transform: a token `x` maps to the target mass of
    /// every token less probable than it, plus a uniform share of its own.
    /// Draws from `target` make those points uniform on `[0, 1)`; any other
    /// distribution, even one with the same support, does not.
    fn pit_ks_distance(target: &[f64], draws: &[u32]) -> f64 {
        let mut order: Vec<usize> = (0..target.len()).collect();
        order.sort_by(|&a, &b| target[a].total_cmp(&target[b]).then(a.cmp(&b)));
        let mut below = vec![0.0; target.len()];
        let mut acc = 0.0;
        for &i in &order {
            below[i] = acc;
            acc += target[i];
        }
        let mut jitter = Xorshift64::new(0xC0FFEE);
        let mut u: Vec<f64> = draws
            .iter()
            .map(|&t| {
                let v = (jitter.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                below[t as usize] + v * target[t as usize]
            })
            .collect();
        u.sort_by(f64::total_cmp);
        let n = u.len() as f64;
        u.iter()
            .enumerate()
            .map(|(i, &x)| ((i + 1) as f64 / n - x).max(x - i as f64 / n))
            .fold(0.0, f64::max)
    }

    /// Both samplers draw from the nucleus they are meant to, not merely from
    /// inside it: the `sample_from_logits` path (`sample_top_p`) and the
    /// `sampling_distribution` + `sample_distribution` path Qwen decodes with.
    /// The bound is the KS critical value at α = 0.001.
    #[test]
    fn sampled_tokens_follow_the_nucleus_distribution() {
        const DRAWS: usize = 20_000;
        let bound = 1.95 / (DRAWS as f64).sqrt();
        let mut rng = Xorshift64::new(11);
        let wide: Vec<f32> = (0..4096)
            .map(|_| (gaussian(&mut rng) * 3.0 * 8.0).round() / 8.0)
            .collect();
        let cases = [
            // The boundary token holds a sixth of the nucleus.
            (
                "five tokens",
                [0.3_f32, 0.25, 0.2, 0.15, 0.1].map(f32::ln).to_vec(),
                0.8,
            ),
            ("4096 tied logits", wide, 0.9),
        ];
        let cfg = |top_p| SamplingConfig {
            temperature: 1.0,
            top_p,
            ..SamplingConfig::default()
        };
        for (name, logits, top_p) in cases {
            let target = intended(&softmax(logits.clone()), top_p);
            let mut rng = Xorshift64::new(5);
            let from_logits: Vec<u32> = (0..DRAWS)
                .map(|_| sample_from_logits(&mut logits.clone(), &[], &cfg(top_p), &mut rng))
                .collect();
            let p = sampling_distribution(&mut logits.clone(), &[], &cfg(top_p));
            let from_distribution: Vec<u32> = (0..DRAWS)
                .map(|_| sample_distribution(&p, &mut rng))
                .collect();
            for (path, draws) in [
                ("sample_from_logits", from_logits),
                ("sampling_distribution", from_distribution),
            ] {
                assert!(
                    draws.iter().all(|&t| target[t as usize] > 0.0),
                    "{name}, {path}: drew a token outside the nucleus"
                );
                let d = pit_ks_distance(&target, &draws);
                assert!(
                    d < bound,
                    "{name}, {path}: KS distance {d:.4} over the bound {bound:.4}"
                );
            }
        }
    }
}
