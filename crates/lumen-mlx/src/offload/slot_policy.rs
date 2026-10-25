//! Which resident expert gives up its slot when a layer call needs a free one.
//!
//! [`super::slot_map::SlotMap`] owns the bookkeeping (who is in which slot, pins, the free list,
//! the plan → commit protocol). This module owns the *choice*: the per-slot state a policy needs
//! and the victim search. Three policies, picked by `LUMEN_MOE_EVICTION`:
//!
//! * **`clock`** — CLOCK: one reference bit per slot, set by a hit and by a load, cleared by the
//!   hand. Exactly what the pool did before the policies existed.
//! * **`gclock`** — generalised CLOCK: the bit becomes a 2-bit counter. A hit adds one up to 3, the
//!   hand takes one off per visit and evicts at zero, and a load starts at **zero**, so an expert
//!   that was read once and never asked for again is the first to go while one that keeps being
//!   hit survives up to three sweeps. CLOCK is the same code with a cap of 1 and a load that
//!   starts at 1, which is how it is implemented here: one sweep loop, two parameter sets.
//! * **`bias`** (default) — transition-biased retention. A decayed hit count per slot (half-life 8 decode
//!   tokens) plus a per-layer table learned online from decode calls, "expert `e` at token `t`
//!   was followed by expert `f` at token `t + 1`", which says how likely each expert of a layer is
//!   to be asked for next. The victim is the lowest `decayed count + w * P(asked next)` among 64
//!   slots sampled at random, so an expert the table expects to follow the last token is kept
//!   over a colder one. About 14-16% fewer misses than CLOCK on recorded routing traces.
//!
//! All three were chosen and tuned by replaying recorded routing traces (`LUMEN_MOE_TRACE`)
//! offline; the constants below are the ones that simulation evaluated, and `slot_replay` replays
//! the same traces through the real [`super::slot_map::SlotMap`] to show the engine makes the same
//! decisions. The policies never touch the computation: they only decide *which* bytes stay
//! resident, so a different policy changes the hit rate and nothing else.
//!
//! Deterministic by construction: `bias` samples with a PCG64 stream whose starting state is a
//! constant, so the same call sequence gives the same victims on every run.

use anyhow::{Result, bail};

use super::slot_map::{ExpertKey, Slot};

lumen_flags::flag! {
    /// Which resident expert the SSD expert pool evicts when it is full, for Flash-Next style
    /// expert offload. Unset and `0` mean `bias` (the default); `clock` is plain CLOCK, and so is
    /// `1` (the flag switched on, as the flags matrix does: the policy before `bias`). `gclock` is CLOCK with a
    /// 2-bit hit counter in place of the reference bit, and loads start at zero. `bias` keeps a
    /// decayed hit count per expert and a per-layer table of which experts followed which at the
    /// previous token, and evicts the sampled expert it expects least. In trace-driven
    /// simulation `gclock` gains 1.6 to 3.4 and `bias` 1.7 to 6.3 decode hit-rate points over
    /// CLOCK, at pools from 2 to 20 GB. Never changes the output, only which experts are
    /// resident and so how many are read from the SSD; any other value is an error when the
    /// pool is built.
    pub(crate) moe_eviction {
        env: "LUMEN_MOE_EVICTION",
        default: false,
        kind: Optimization,
    }
}

/// The variable's name, for reading its value after the registry flag said it is set.
const EVICTION_ENV: &str = "LUMEN_MOE_EVICTION";

/// Which eviction policy the pool runs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum EvictionKind {
    Clock,
    GClock,
    #[default]
    Bias,
}

impl EvictionKind {
    #[cfg(test)]
    pub(crate) const ALL: [Self; 3] = [Self::Clock, Self::GClock, Self::Bias];

    /// `LUMEN_MOE_EVICTION`: `bias` (default), `clock` or `gclock`.
    pub(crate) fn from_env() -> Result<Self> {
        if !moe_eviction::get() {
            return Ok(Self::default());
        }
        Self::parse(std::env::var(EVICTION_ENV).ok().as_deref())
    }

    pub(crate) fn parse(raw: Option<&str>) -> Result<Self> {
        match raw.map(|r| r.trim().to_ascii_lowercase()).as_deref() {
            None | Some("") => Ok(Self::default()),
            // A bare "on" is the alternate the flags matrix flips to; it means
            // the policy `bias` replaced, so the matrix runs the suite on it.
            Some("clock" | "1" | "true" | "on" | "yes") => Ok(Self::Clock),
            Some("gclock") => Ok(Self::GClock),
            Some("bias") => Ok(Self::Bias),
            Some(other) => bail!("{EVICTION_ENV}={other:?} is not `clock`, `gclock` or `bias`"),
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Clock => "clock",
            Self::GClock => "gclock",
            Self::Bias => "bias",
        }
    }
}

/// What a layer call is, as far as the learning policy is concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CallKind {
    /// A decode step or a speculative verify (a few rows): the routing that repeats from token to
    /// token, which is what the transition table learns from.
    Decode,
    /// A prompt chunk that took the pool path. Its experts are chosen by the whole prompt, not by
    /// the previous token, so it is not learned from.
    Prefill,
}

/// The knobs of `bias`. The defaults are the configuration the trace simulation selected
/// (leave-one-session-out: 17 of 18 folds chose it); only tests change them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BiasParams {
    /// Decode tokens for a slot's hit count to halve.
    pub half_life: f64,
    /// Slots sampled per victim.
    pub sample: usize,
    /// Weight of the predicted-next probability against the decayed hit count.
    pub weight: f64,
    /// Hit count a freshly loaded expert starts with.
    pub admit_score: f64,
    /// A call naming more distinct experts than this is a prompt chunk, not a decode step...
    pub large_call: usize,
    /// ...and samples only this many slots per victim. A chunk misses dozens to hundreds of
    /// experts at once, so the sample is the cost: 8 slots instead of 64 keeps the work of a
    /// 150-expert call near 6 us over CLOCK and costs 0.01 hit-rate points in the simulation.
    /// Set equal to `sample` for the policy exactly as it was first evaluated.
    pub large_sample: usize,
}

impl Default for BiasParams {
    fn default() -> Self {
        Self {
            half_life: 8.0,
            sample: 64,
            weight: 128.0,
            admit_score: 0.5,
            large_call: 48,
            large_sample: 8,
        }
    }
}

/// Everything a policy needs to know about the pool and the model.
#[derive(Debug, Clone)]
pub(crate) struct PolicyConfig {
    pub kind: EvictionKind,
    /// Experts in the widest layer: the side of a layer's transition table.
    pub width: usize,
    /// Trunk MoE layers one decode token passes through (the draft head's layer not counted): the
    /// unit the `bias` decay clock ticks in.
    pub layers_per_pass: usize,
    pub bias: BiasParams,
}

impl PolicyConfig {
    pub(crate) fn new(kind: EvictionKind, width: usize, layers_per_pass: usize) -> Self {
        Self {
            kind,
            width,
            layers_per_pass,
            bias: BiasParams::default(),
        }
    }
}

impl Default for PolicyConfig {
    /// CLOCK, which needs to know nothing about the model.
    fn default() -> Self {
        Self::new(EvictionKind::Clock, 0, 1)
    }
}

/// The policy a [`super::slot_map::SlotMap`] runs. An enum rather than a trait object: the calls
/// are on the per-expert path and the match is perfectly predictable.
pub(super) enum Policy {
    /// CLOCK and GCLOCK.
    Counter(Counter),
    Bias(Box<Bias>),
}

impl Policy {
    pub(super) fn new(n_slots: usize, cfg: &PolicyConfig) -> Self {
        match cfg.kind {
            EvictionKind::Clock => Self::Counter(Counter::new(n_slots, 1, 1)),
            EvictionKind::GClock => Self::Counter(Counter::new(n_slots, 3, 0)),
            EvictionKind::Bias => Self::Bias(Box::new(Bias::new(n_slots, cfg))),
        }
    }

    /// A layer call is about to be planned for `needed`.
    #[inline]
    pub(super) fn begin_call(&mut self, kind: CallKind, needed: &[ExpertKey]) {
        if let Self::Bias(b) = self {
            b.begin_call(kind, needed);
        }
    }

    /// The expert in `slot` was asked for and is resident.
    #[inline]
    pub(super) fn on_hit(&mut self, slot: usize) {
        match self {
            Self::Counter(c) => c.on_hit(slot),
            Self::Bias(b) => b.on_hit(slot),
        }
    }

    /// An expert's bytes were loaded into `slot` and published.
    #[inline]
    pub(super) fn on_admit(&mut self, slot: usize) {
        match self {
            Self::Counter(c) => c.on_admit(slot),
            Self::Bias(b) => b.on_admit(slot),
        }
    }

    /// `slot` no longer holds an expert (evicted, aborted, or lent out): forget what was known.
    #[inline]
    pub(super) fn on_vacate(&mut self, slot: usize) {
        match self {
            Self::Counter(c) => c.heat[slot] = 0,
            Self::Bias(b) => b.score[slot] = 0.0,
        }
    }

    /// The victim: an unpinned, not-loading slot, or `None` when there is none.
    #[inline]
    pub(super) fn victim(&mut self, slots: &[Slot]) -> Option<usize> {
        match self {
            Self::Counter(c) => c.victim(slots),
            Self::Bias(b) => b.victim(slots),
        }
    }

    /// What `slot` is worth to the policy right now, for tests: the CLOCK counter, or the decayed
    /// hit count.
    #[cfg(test)]
    pub(super) fn worth(&self, slot: usize) -> f64 {
        match self {
            Self::Counter(c) => f64::from(c.heat[slot]),
            Self::Bias(b) => b.decayed(slot),
        }
    }

    #[cfg(test)]
    pub(super) fn bias(&self) -> Option<&Bias> {
        match self {
            Self::Bias(b) => Some(b),
            Self::Counter(_) => None,
        }
    }
}

// ---- CLOCK and GCLOCK ---------------------------------------------------------------------------

/// A small counter per slot and a hand. A hit adds one up to `cap`; the hand takes one off each
/// slot it passes and evicts the first slot already at zero; a load starts at `on_load`.
/// CLOCK is `cap = 1, on_load = 1` (a reference bit), GCLOCK `cap = 3, on_load = 0`.
pub(super) struct Counter {
    heat: Vec<u8>,
    hand: usize,
    cap: u8,
    on_load: u8,
}

impl Counter {
    fn new(n_slots: usize, cap: u8, on_load: u8) -> Self {
        Self {
            heat: vec![0; n_slots],
            hand: 0,
            cap,
            on_load,
        }
    }

    #[inline]
    fn on_hit(&mut self, slot: usize) {
        let h = &mut self.heat[slot];
        if *h < self.cap {
            *h += 1;
        }
    }

    #[inline]
    fn on_admit(&mut self, slot: usize) {
        self.heat[slot] = self.on_load;
    }

    fn victim(&mut self, slots: &[Slot]) -> Option<usize> {
        let n = slots.len();
        let mut i = self.hand;
        // A counter at `cap` needs `cap` visits to reach zero and one more to be taken, so
        // `cap + 1` sweeps are enough; a further one would only find pinned slots again.
        for _ in 0..(usize::from(self.cap) + 1) * n {
            let j = i;
            i += 1;
            if i == n {
                i = 0;
            }
            if slots[j].busy() {
                continue;
            }
            let h = &mut self.heat[j];
            if *h > 0 {
                *h -= 1;
                continue;
            }
            self.hand = i;
            return Some(j);
        }
        self.hand = i;
        None
    }
}

// ---- transition-biased retention --------------------------------------------------------------------

/// PCG64 (XSL-RR 128/64), the generator numpy's `default_rng` uses, started from the state
/// `numpy.random.default_rng(7)` starts from. The simulator that selected `bias` sampled victims
/// with exactly that stream, so reproducing it draw for draw (numpy's buffered 32-bit halves and
/// Lemire bounded integers included) lets the engine be checked against the simulator
/// decision by decision instead of statistically.
#[derive(Clone)]
struct Pcg64 {
    state: u128,
    inc: u128,
    /// The high half of the last 64-bit output, if its low half was handed out and it was not.
    spare: Option<u32>,
}

impl Pcg64 {
    const MULT: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;
    /// `default_rng(7).bit_generator.state["state"]` and `["inc"]`.
    const SEED7_STATE: u128 = 0x9d0a_ecc4_e7b1_8891_9ae2_415a_6b29_2038;
    const SEED7_INC: u128 = 0xc475_1896_ce8a_cebf_6688_7f5a_c070_d959;

    fn seeded() -> Self {
        Self {
            state: Self::SEED7_STATE,
            inc: Self::SEED7_INC,
            spare: None,
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(Self::MULT).wrapping_add(self.inc);
        let s = self.state;
        (((s >> 64) as u64) ^ (s as u64)).rotate_right((s >> 122) as u32)
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let v = self.next_u64();
        self.spare = Some((v >> 32) as u32);
        v as u32
    }

    /// Uniform in `0..n` (`n >= 2`), numpy's `Generator.integers(0, n)` for one element.
    #[inline]
    fn below(&mut self, n: u32) -> u32 {
        let mut m = u64::from(self.next_u32()) * u64::from(n);
        let mut leftover = m as u32;
        if leftover < n {
            let threshold = (u32::MAX - (n - 1)) % n;
            while leftover < threshold {
                m = u64::from(self.next_u32()) * u64::from(n);
                leftover = m as u32;
            }
        }
        (m >> 32) as u32
    }
}

/// What one layer has learned about its own routing.
struct Transitions {
    /// `counts[e * width + f]`: how many times expert `f` was asked for at the token after expert
    /// `e` was. Zeroed on allocation, so rows nobody has touched never become resident memory.
    counts: Vec<u16>,
    /// `row_sum[e]` is the sum of row `e` of `counts`.
    row_sum: Vec<u32>,
    /// The experts of this layer's previous decode call, first-seen order.
    prev: Vec<u16>,
}

const NO_LAYER: usize = usize::MAX;

/// Sampling rounds before the victim search gives up on random draws and scans every slot. A round
/// finds only pinned slots with probability `(pinned / slots)^sample`: never, in a pool of
/// hundreds of slots with a few dozen pinned.
const SAMPLE_ROUNDS: usize = 8;

pub(super) struct Bias {
    /// Decayed hit count at the time `seen_at` says, per slot; 0 for an empty slot.
    score: Vec<f64>,
    seen_at: Vec<f64>,
    /// The decay clock, in decode tokens: it advances `1 / layers_per_pass` per layer call.
    tau: f64,
    tau_step: f64,
    /// `ln 2 / half_life`.
    decay: f64,
    admit_score: f64,
    weight: f64,
    sample: usize,
    large_call: usize,
    large_sample: usize,
    /// Slots to sample per victim in the call being planned.
    sample_now: usize,
    rng: Pcg64,

    width: usize,
    /// By layer number; `None` until the layer's first decode call.
    layers: Vec<Option<Box<Transitions>>>,
    /// Probability that each expert of each layer is asked for at that layer's next token, as
    /// of its latest decode call; layer `l` starts at `pred_at[l]` (`NO_LAYER` when unseen).
    pred: Vec<f32>,
    pred_at: Vec<usize>,
    /// Scratch: this call's distinct experts, and a stamp per expert to find them.
    cur: Vec<u16>,
    stamp: Vec<u32>,
    stamp_gen: u32,
}

impl Bias {
    fn new(n_slots: usize, cfg: &PolicyConfig) -> Self {
        let p = cfg.bias;
        Self {
            score: vec![0.0; n_slots],
            seen_at: vec![0.0; n_slots],
            tau: 0.0,
            tau_step: 1.0 / cfg.layers_per_pass.max(1) as f64,
            decay: std::f64::consts::LN_2 / p.half_life,
            admit_score: p.admit_score,
            weight: p.weight,
            sample: p.sample,
            large_call: p.large_call,
            large_sample: p.large_sample,
            sample_now: p.sample,
            rng: Pcg64::seeded(),
            width: cfg.width,
            layers: Vec::new(),
            pred: Vec::new(),
            pred_at: Vec::new(),
            cur: Vec::with_capacity(cfg.width),
            stamp: vec![0; cfg.width],
            stamp_gen: 0,
        }
    }

    fn begin_call(&mut self, kind: CallKind, needed: &[ExpertKey]) {
        self.tau += self.tau_step;
        self.sample_now = if needed.len() > self.large_call {
            self.large_sample
        } else {
            self.sample
        };
        if kind == CallKind::Decode && !needed.is_empty() {
            self.learn(needed);
        }
    }

    /// Fold one decode call into its layer's transition table and refresh the layer's prediction.
    /// A call belongs to the layer of its first expert; the engine never mixes layers in a call.
    fn learn(&mut self, needed: &[ExpertKey]) {
        let layer = usize::from(needed[0].layer);
        let width = self.width;

        // The call's distinct experts, first-seen order.
        self.stamp_gen = self.stamp_gen.wrapping_add(1);
        if self.stamp_gen == 0 {
            self.stamp.fill(0);
            self.stamp_gen = 1;
        }
        self.cur.clear();
        for k in needed {
            let e = usize::from(k.expert);
            if usize::from(k.layer) != layer || e >= width || self.stamp[e] == self.stamp_gen {
                continue;
            }
            self.stamp[e] = self.stamp_gen;
            self.cur.push(k.expert);
        }
        if self.cur.is_empty() {
            return;
        }

        if self.layers.len() <= layer {
            self.layers.resize_with(layer + 1, || None);
            self.pred_at.resize(layer + 1, NO_LAYER);
        }
        let Self {
            layers,
            pred,
            pred_at,
            cur,
            ..
        } = self;
        let t = layers[layer].get_or_insert_with(|| {
            pred_at[layer] = pred.len();
            pred.resize(pred.len() + width, 0.0);
            Box::new(Transitions {
                counts: vec![0; width * width],
                row_sum: vec![0; width],
                prev: Vec::with_capacity(width),
            })
        });
        let at = pred_at[layer];

        // "e at the previous token was followed by f now", for every pair.
        for &p in &t.prev {
            let p = usize::from(p);
            let row = &mut t.counts[p * width..(p + 1) * width];
            let mut halved = false;
            for &f in cur.iter() {
                let c = &mut row[usize::from(f)];
                if *c == u16::MAX {
                    // Keep the ratios, forget the oldest half, and never saturate.
                    row.iter_mut().for_each(|x| *x >>= 1);
                    halved = true;
                }
                row[usize::from(f)] += 1;
            }
            t.row_sum[p] = if halved {
                row.iter().map(|&x| u32::from(x)).sum()
            } else {
                t.row_sum[p] + cur.len() as u32
            };
        }
        t.prev.clear();
        t.prev.extend_from_slice(cur);

        // P(f at the next token) = mean over the experts just used of their rows, normalised.
        let out = &mut pred[at..at + width];
        out.fill(0.0);
        for &e in cur.iter() {
            let e = usize::from(e);
            let denom = (t.row_sum[e] as f32).max(1.0);
            for (o, &c) in out.iter_mut().zip(&t.counts[e * width..(e + 1) * width]) {
                *o += f32::from(c) / denom;
            }
        }
        let n = cur.len() as f32;
        out.iter_mut().for_each(|o| *o /= n);
    }

    #[inline]
    fn on_hit(&mut self, slot: usize) {
        self.score[slot] = self.decayed(slot) + 1.0;
        self.seen_at[slot] = self.tau;
    }

    #[inline]
    fn on_admit(&mut self, slot: usize) {
        self.score[slot] = self.admit_score;
        self.seen_at[slot] = self.tau;
    }

    #[inline]
    fn decayed(&self, slot: usize) -> f64 {
        self.score[slot] * (-(self.tau - self.seen_at[slot]) * self.decay).exp()
    }

    /// The probability the table gives `key` of being asked for at its layer's next token.
    #[inline]
    fn boost(&self, key: Option<ExpertKey>) -> f64 {
        let Some(k) = key else { return 0.0 };
        let at = self
            .pred_at
            .get(usize::from(k.layer))
            .copied()
            .unwrap_or(NO_LAYER);
        if at == NO_LAYER || usize::from(k.expert) >= self.width {
            return 0.0;
        }
        f64::from(self.pred[at + usize::from(k.expert)])
    }

    /// What keeping `slot` is worth: lower goes first.
    #[inline]
    fn retention(&self, slot: usize, key: Option<ExpertKey>) -> f64 {
        self.decayed(slot) + self.weight * self.boost(key)
    }

    fn victim(&mut self, slots: &[Slot]) -> Option<usize> {
        let n = slots.len();
        if self.sample_now < n {
            for _ in 0..SAMPLE_ROUNDS {
                // The first of the lowest wins a tie; a score is never infinite or NaN.
                let (mut best, mut lowest) = (usize::MAX, f64::INFINITY);
                for _ in 0..self.sample_now {
                    let c = self.rng.below(n as u32) as usize;
                    if slots[c].busy() {
                        continue;
                    }
                    let r = self.retention(c, slots[c].key);
                    if r < lowest {
                        (best, lowest) = (c, r);
                    }
                }
                if best != usize::MAX {
                    return Some(best);
                }
            }
        }
        // A pool no larger than the sample, or one nearly all pinned: look at every slot.
        let (mut best, mut lowest) = (None, f64::INFINITY);
        for (c, s) in slots.iter().enumerate() {
            if s.busy() {
                continue;
            }
            let r = self.retention(c, s.key);
            if best.is_none() || r < lowest {
                (best, lowest) = (Some(c), r);
            }
        }
        best
    }

    /// The learned probability that `expert` of `layer` follows the layer's latest decode call.
    #[cfg(test)]
    pub(super) fn predicted(&self, layer: u16, expert: u16) -> f32 {
        match self.pred_at.get(usize::from(layer)) {
            Some(&at) if at != NO_LAYER && usize::from(expert) < self.width => {
                self.pred[at + usize::from(expert)]
            }
            _ => 0.0,
        }
    }

    /// `counts[e][f]` and the row sum of `layer`, for tests.
    #[cfg(test)]
    pub(super) fn transitions(&self, layer: u16, e: u16, f: u16) -> Option<(u16, u32)> {
        let t = self.layers.get(usize::from(layer))?.as_ref()?;
        Some((
            t.counts[usize::from(e) * self.width + usize::from(f)],
            t.row_sum[usize::from(e)],
        ))
    }

    /// Bytes of transition tables allocated so far (the zeroed pages are not resident until
    /// touched).
    #[cfg(test)]
    pub(super) fn table_bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|t| t.counts.len() * 2 + t.row_sum.len() * 4)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_variable_picks_a_policy_and_junk_is_an_error() {
        let ok = |raw: Option<&str>| EvictionKind::parse(raw).unwrap();
        assert_eq!(ok(None), EvictionKind::Bias);
        assert_eq!(ok(Some("")), EvictionKind::Bias);
        assert_eq!(ok(Some(" Clock ")), EvictionKind::Clock);
        assert_eq!(ok(Some("gclock")), EvictionKind::GClock);
        assert_eq!(ok(Some("BIAS")), EvictionKind::Bias);
        // The flags matrix switches every Optimization flag on with `1`: that
        // has to run a policy (the one `bias` replaced), not fail every load.
        assert_eq!(ok(Some("1")), EvictionKind::Clock);
        for bad in ["lru", "2", "bias,gclock", "gclock2"] {
            assert!(EvictionKind::parse(Some(bad)).is_err(), "{bad:?}");
        }
        assert_eq!(EvictionKind::default(), EvictionKind::Bias);
        for kind in EvictionKind::ALL {
            assert_eq!(EvictionKind::parse(Some(kind.name())).unwrap(), kind);
        }
    }

    /// The first draws of `numpy.random.default_rng(7).integers(0, n, size=...)`: the stream
    /// the simulator sampled victims with. Includes a range that forces rejections.
    #[test]
    fn the_sampler_reproduces_numpys_stream_draw_for_draw() {
        let mut rng = Pcg64::seeded();
        let got: Vec<u32> = (0..8).map(|_| rng.below(2423)).collect();
        assert_eq!(got, [2289, 1514, 1657, 2173, 1401, 1879, 2019, 545]);
        let mut rng = Pcg64::seeded();
        let got: Vec<u32> = (0..8).map(|_| rng.below(7161)).collect();
        assert_eq!(got, [6766, 4476, 4899, 6424, 4141, 5554, 5969, 1612]);
    }

    #[test]
    fn the_sampler_is_uniform_enough_and_never_out_of_range() {
        let mut rng = Pcg64::seeded();
        let n = 100u32;
        let mut seen = vec![0u32; n as usize];
        for _ in 0..200_000 {
            let v = rng.below(n);
            assert!(v < n);
            seen[v as usize] += 1;
        }
        // 2000 expected per bucket; a broken Lemire step shows up as a far wider spread.
        assert!(seen.iter().all(|&c| (1700..2300).contains(&c)), "{seen:?}");
    }

    use crate::offload::slot_map::{Plan, SlotMap};

    fn k(layer: u16, expert: u16) -> ExpertKey {
        ExpertKey::new(layer, expert)
    }

    fn map_of(kind: EvictionKind, n_slots: usize, width: usize, layers: usize) -> SlotMap {
        SlotMap::with_policy(n_slots, &PolicyConfig::new(kind, width, layers))
    }

    fn call_as(map: &mut SlotMap, needed: &[ExpertKey], kind: CallKind) -> Plan {
        let plan = map.plan_as(needed, kind).expect("plan");
        map.commit(&plan);
        map.release(&plan);
        plan
    }

    fn decode(map: &mut SlotMap, needed: &[ExpertKey]) -> Plan {
        call_as(map, needed, CallKind::Decode)
    }

    #[test]
    fn gclock_counts_hits_up_to_the_cap_and_a_load_starts_at_zero() {
        let mut map = map_of(EvictionKind::GClock, 4, 8, 1);
        let slot = decode(&mut map, &[k(0, 0)]).slots[0];
        assert_eq!(map.policy().worth(slot), 0.0, "a load starts at zero");
        for want in [1.0, 2.0, 3.0, 3.0, 3.0] {
            decode(&mut map, &[k(0, 0)]);
            assert_eq!(map.policy().worth(slot), want);
        }
        // CLOCK is the same code with a cap of one and a load that starts there.
        let mut clock = map_of(EvictionKind::Clock, 4, 8, 1);
        let slot = decode(&mut clock, &[k(0, 0)]).slots[0];
        assert_eq!(clock.policy().worth(slot), 1.0);
        decode(&mut clock, &[k(0, 0)]);
        assert_eq!(clock.policy().worth(slot), 1.0);
    }

    /// Three slots A B C, A hit until its counter is at the cap of 3. Each miss afterwards sends
    /// the hand round: it takes one off A when it passes and evicts the zeroed neighbour, and A
    /// goes only on the visit after its counter reached zero.
    #[test]
    fn gclock_hand_takes_one_off_per_visit_and_evicts_at_zero() {
        let mut map = map_of(EvictionKind::GClock, 3, 16, 1);
        let (a, b, c) = (k(0, 0), k(0, 1), k(0, 2));
        let slot_a = decode(&mut map, &[a]).slots[0];
        decode(&mut map, &[b]);
        decode(&mut map, &[c]);
        for _ in 0..4 {
            decode(&mut map, &[a]);
        }
        assert_eq!(map.policy().worth(slot_a), 3.0);

        // D: the hand starts at slot 0 (A: 3 -> 2, skipped), slot 1 (B, zero) is the victim.
        decode(&mut map, &[k(0, 3)]);
        assert!(map.peek(b).is_none(), "B was never hit again: first to go");
        assert!(map.peek(a).is_some() && map.peek(c).is_some());
        assert_eq!(map.policy().worth(slot_a), 2.0);
        // E takes C (slot 2, zero). F: A 2 -> 1, then D (slot 1, zero). G: E. H: A 1 -> 0, F.
        decode(&mut map, &[k(0, 4)]);
        assert!(map.peek(c).is_none());
        decode(&mut map, &[k(0, 5)]);
        assert_eq!(map.policy().worth(slot_a), 1.0);
        decode(&mut map, &[k(0, 6)]);
        decode(&mut map, &[k(0, 7)]);
        assert_eq!(map.policy().worth(slot_a), 0.0);
        decode(&mut map, &[k(0, 8)]);
        assert!(map.peek(a).is_some(), "A survived six evictions around it");
        // The hand reaches A a fourth time, finds zero and takes it.
        decode(&mut map, &[k(0, 9)]);
        assert!(map.peek(a).is_none(), "A goes once its counter is spent");
    }

    /// The same sequence under both: a hit on A does nothing for CLOCK, whose sweep clears every
    /// bit and takes A first, while GCLOCK protects A and takes the expert that was never hit.
    #[test]
    fn gclock_keeps_the_hit_expert_that_clock_evicts() {
        let run = |kind| {
            let mut map = map_of(kind, 3, 16, 1);
            for e in 0..3 {
                decode(&mut map, &[k(0, e)]);
            }
            decode(&mut map, &[k(0, 0)]); // A is hit once
            decode(&mut map, &[k(0, 3)]); // D needs a victim
            map
        };
        let clock = run(EvictionKind::Clock);
        assert!(clock.peek(k(0, 0)).is_none(), "CLOCK took A, hit or not");
        let gclock = run(EvictionKind::GClock);
        assert!(gclock.peek(k(0, 0)).is_some(), "GCLOCK kept the hit expert");
        assert!(gclock.peek(k(0, 1)).is_none(), "and took the never-hit B");
    }

    #[test]
    fn a_pinned_slot_is_skipped_however_cold_it_is() {
        for kind in [EvictionKind::GClock, EvictionKind::Bias] {
            let mut map = map_of(kind, 3, 16, 1);
            for e in 0..3 {
                decode(&mut map, &[k(0, e)]);
            }
            // Keep expert 0 pinned in a plan of its own while another call needs a victim.
            let held = map.plan(&[k(0, 0)]).expect("plan");
            let second = map.plan(&[k(1, 0), k(1, 1)]).expect("plan");
            assert!(
                second.loads.iter().all(|&(_, s)| s != held.slots[0]),
                "{kind:?}: the pinned slot was handed out"
            );
            map.commit(&second);
            map.release(&second);
            map.release(&held);
            assert!(map.peek(k(0, 0)).is_some(), "{kind:?}");
        }
    }

    fn bias_map(n_slots: usize, width: usize, layers: usize, weight: f64) -> SlotMap {
        let mut cfg = PolicyConfig::new(EvictionKind::Bias, width, layers);
        cfg.bias.weight = weight;
        SlotMap::with_policy(n_slots, &cfg)
    }

    /// Experts 0,1 then 2,3 then 0,1 again at one layer: after the third call the table says
    /// that at the token after {0,1}, experts 2 and 3 are the ones asked for.
    fn trained() -> SlotMap {
        let mut map = bias_map(4, 4, 1, 128.0);
        decode(&mut map, &[k(0, 0), k(0, 1)]);
        decode(&mut map, &[k(0, 2), k(0, 3)]);
        decode(&mut map, &[k(0, 0), k(0, 1)]);
        map
    }

    #[test]
    fn the_table_counts_what_followed_what_and_predicts_from_the_latest_call() {
        let map = trained();
        let bias = map.policy().bias().expect("bias");
        // After call 2: 0->2, 0->3, 1->2, 1->3 once each. After call 3: 2->0, 2->1, 3->0, 3->1.
        assert_eq!(bias.transitions(0, 0, 2), Some((1, 2)));
        assert_eq!(bias.transitions(0, 1, 3), Some((1, 2)));
        assert_eq!(bias.transitions(0, 2, 1), Some((1, 2)));
        assert_eq!(bias.transitions(0, 0, 1), Some((0, 2)));
        // The latest call was {0,1}: the mean of their rows, [0, 0, 1/2, 1/2].
        assert_eq!(
            [0, 1, 2, 3].map(|e| bias.predicted(0, e)),
            [0.0, 0.0, 0.5, 0.5]
        );
        assert_eq!(
            bias.predicted(1, 0),
            0.0,
            "an unseen layer predicts nothing"
        );
    }

    #[test]
    fn only_decode_calls_are_learned_from() {
        let mut map = trained();
        // Experts 2,3 as a prompt chunk, then 0,1 again as one: neither adds a transition.
        call_as(&mut map, &[k(0, 2), k(0, 3)], CallKind::Prefill);
        let bias = map.policy().bias().expect("bias");
        assert_eq!(bias.transitions(0, 0, 2), Some((1, 2)));
        assert_eq!(bias.transitions(0, 3, 0), Some((1, 2)));
        assert_eq!(bias.predicted(0, 2), 0.5, "the prediction is untouched too");
        // A decode call afterwards pairs with the last *decode* call ({0,1}), not the chunk.
        decode(&mut map, &[k(0, 2)]);
        let bias = map.policy().bias().expect("bias");
        assert_eq!(bias.transitions(0, 0, 2), Some((2, 3)));
    }

    #[test]
    fn the_victim_is_not_the_expert_the_table_expects_next() {
        // Slots hold 0,1 (just hit) and 2,3 (older, never hit). A miss at another layer needs one.
        let mut map = trained();
        decode(&mut map, &[k(1, 0)]);
        assert!(map.peek(k(0, 2)).is_some() && map.peek(k(0, 3)).is_some());
        assert!(
            map.peek(k(0, 0)).is_none(),
            "0 and 1 tie on the count; the table protects 2 and 3, and the first slot of a tie goes"
        );
        assert!(map.peek(k(0, 1)).is_some());

        // The same sequence without the table's say: the colder 2 goes.
        let mut plain = bias_map(4, 4, 1, 0.0);
        decode(&mut plain, &[k(0, 0), k(0, 1)]);
        decode(&mut plain, &[k(0, 2), k(0, 3)]);
        decode(&mut plain, &[k(0, 0), k(0, 1)]);
        decode(&mut plain, &[k(1, 0)]);
        assert!(
            plain.peek(k(0, 2)).is_none(),
            "decayed count alone evicts 2"
        );
        assert!(plain.peek(k(0, 0)).is_some() && plain.peek(k(0, 1)).is_some());
    }

    #[test]
    fn hits_add_one_and_the_count_halves_every_half_life() {
        // 12 layers per token, so one call advances the clock by 1/12 of a token.
        let mut map = bias_map(8, 4, 12, 128.0);
        let slot = decode(&mut map, &[k(0, 0)]).slots[0];
        assert_eq!(map.policy().worth(slot), 0.5, "a load starts at one half");
        decode(&mut map, &[k(0, 0)]);
        let after_hit = map.policy().worth(slot);
        let expected = 0.5 * (-(1.0 / 12.0) * std::f64::consts::LN_2 / 8.0).exp() + 1.0;
        assert!(
            (after_hit - expected).abs() < 1e-12,
            "{after_hit} vs {expected}"
        );
        // Eight tokens (96 calls elsewhere) later it has halved.
        for _ in 0..96 {
            decode(&mut map, &[k(1, 0)]);
        }
        let later = map.policy().worth(slot);
        assert!(
            (later - after_hit / 2.0).abs() < 1e-9,
            "{later} vs {after_hit}"
        );
    }

    #[test]
    fn the_same_calls_give_the_same_victims_even_when_the_choice_is_sampled() {
        // 400 slots is more than the 64 sampled per victim, so this exercises the random path.
        let run = || {
            let mut rng: u64 = 0xD1B5_4A32_D192_ED03;
            let mut next = move || {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng
            };
            let mut map = bias_map(400, 128, 6, 128.0);
            let mut slots = Vec::new();
            for _ in 0..3_000 {
                let layer = (next() % 6) as u16;
                let needed: Vec<ExpertKey> =
                    (0..8).map(|_| k(layer, (next() % 128) as u16)).collect();
                let plan = decode(&mut map, &needed);
                slots.push(plan.slots);
            }
            (slots, map.stats())
        };
        let (a, stats) = run();
        let (b, _) = run();
        assert_eq!(a, b);
        assert!(
            stats.misses > 400,
            "the pool filled up and evicted: {stats:?}"
        );
    }

    #[test]
    fn a_count_that_reaches_the_top_of_u16_halves_its_row_instead_of_saturating() {
        let mut map = bias_map(2, 2, 1, 128.0);
        for _ in 0..70_000 {
            decode(&mut map, &[k(0, 0), k(0, 1)]);
        }
        let bias = map.policy().bias().expect("bias");
        let (count, row_sum) = bias.transitions(0, 0, 0).expect("table");
        assert!((32_768..u16::MAX).contains(&count), "count {count}");
        assert_eq!(
            row_sum,
            2 * u32::from(count),
            "the row sum still equals the sum of the row"
        );
        assert_eq!(bias.transitions(0, 0, 1), Some((count, row_sum)));
        assert_eq!(bias.predicted(0, 0), 0.5);
    }

    #[test]
    fn tables_exist_only_for_bias_only_for_layers_that_decoded_and_cost_width_squared_u16() {
        let width = 512;
        let mut map = bias_map(64, width, 48, 128.0);
        assert_eq!(map.policy().bias().expect("bias").table_bytes(), 0);
        call_as(&mut map, &[k(0, 1), k(0, 2)], CallKind::Prefill);
        assert_eq!(
            map.policy().bias().expect("bias").table_bytes(),
            0,
            "a prompt chunk allocates nothing"
        );
        decode(&mut map, &[k(0, 1), k(0, 2)]);
        let one_layer = width * width * 2 + width * 4;
        assert_eq!(map.policy().bias().expect("bias").table_bytes(), one_layer);
        // The draft head's layer number is far from the trunk's; it gets its own table.
        decode(&mut map, &[k(1000, 5)]);
        assert_eq!(
            map.policy().bias().expect("bias").table_bytes(),
            2 * one_layer
        );
        for kind in [EvictionKind::Clock, EvictionKind::GClock] {
            let map = map_of(kind, 64, width, 48);
            assert!(map.policy().bias().is_none(), "{kind:?} keeps no tables");
        }
    }

    #[test]
    fn a_call_naming_experts_beyond_the_table_width_is_served_without_learning_them() {
        let mut map = bias_map(8, 4, 1, 128.0);
        decode(&mut map, &[k(0, 1), k(0, 9)]);
        decode(&mut map, &[k(0, 1), k(0, 2), k(0, 9)]);
        let bias = map.policy().bias().expect("bias");
        assert_eq!(bias.transitions(0, 1, 2), Some((1, 2)));
        assert_eq!(bias.predicted(0, 9), 0.0);
    }

    #[test]
    fn a_call_naming_many_experts_samples_fewer_slots_per_victim() {
        let mut cfg = PolicyConfig::new(EvictionKind::Bias, 64, 6);
        cfg.bias.large_call = 3;
        cfg.bias.large_sample = 5;
        let mut map = SlotMap::with_policy(200, &cfg);
        let sample_now = |map: &SlotMap| map.policy().bias().expect("bias").sample_now;
        decode(&mut map, &[k(0, 1), k(0, 2), k(0, 3)]);
        assert_eq!(
            sample_now(&map),
            64,
            "three experts is still a decode-sized call"
        );
        decode(&mut map, &[k(0, 1), k(0, 2), k(0, 3), k(0, 4)]);
        assert_eq!(sample_now(&map), 5);
        decode(&mut map, &[k(0, 1)]);
        assert_eq!(sample_now(&map), 64, "and the next small call goes back");
    }
}
