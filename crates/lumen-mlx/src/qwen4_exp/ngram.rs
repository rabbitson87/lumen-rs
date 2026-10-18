//! Hashing of token n-grams into rows of the sharded embedding table.
//!
//! The table has `heads` independent hash heads (`(ngram_size - 1) *
//! heads_per_ngram`, 16 for the shipped model). Head `h` of order `n` maps the
//! last `n` tokens to row `((t0*m0 ^ t1*m1 [^ t2*m2]) mod size[h]) + offset[h]`,
//! where `t0` is the current token, `t1` the one before it and so on, and the
//! multipliers `m`, head sizes and offsets are buffers stored in the
//! checkpoint (they are also derivable from the config; see [`derive_buffers`]).
//!
//! A predecessor that would lie before the start of the text, or before an EOS
//! token, reads as EOS: n-grams never straddle a document boundary.
//!
//! The arithmetic is `u64` wrapping, as in the reference. Multipliers are small
//! enough that `token * multiplier` never exceeds `2^63`, so signed and
//! unsigned interpretations agree.

use anyhow::{Result, bail};

/// Row ids of the n-gram table for a run of tokens.
#[derive(Debug, Clone)]
pub struct NgramHasher {
    ngram_size: usize,
    heads_per_ngram: usize,
    eos: u32,
    multipliers: Vec<u64>,
    sizes: Vec<u64>,
    offsets: Vec<u64>,
}

impl NgramHasher {
    pub fn new(
        ngram_size: usize,
        heads_per_ngram: usize,
        eos: u32,
        multipliers: Vec<u64>,
        sizes: Vec<u64>,
        offsets: Vec<u64>,
    ) -> Result<Self> {
        if ngram_size < 2 {
            bail!("n-gram size {ngram_size} is below 2");
        }
        let heads = (ngram_size - 1) * heads_per_ngram;
        if multipliers.len() != ngram_size {
            bail!(
                "{} multipliers for n-gram size {ngram_size}",
                multipliers.len()
            );
        }
        if sizes.len() != heads || offsets.len() != heads {
            bail!(
                "{} sizes and {} offsets for {heads} hash heads",
                sizes.len(),
                offsets.len()
            );
        }
        if sizes.contains(&0) {
            bail!("a hash head has size 0");
        }
        Ok(Self {
            ngram_size,
            heads_per_ngram,
            eos,
            multipliers,
            sizes,
            offsets,
        })
    }

    /// Hash heads per token.
    pub fn heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// Tokens of context a position needs from before the new tokens.
    pub fn context_len(&self) -> usize {
        self.ngram_size - 1
    }

    /// Total rows addressed by all heads.
    pub fn total_rows(&self) -> u64 {
        self.offsets
            .last()
            .zip(self.sizes.last())
            .map_or(0, |(o, s)| o + s)
    }

    /// `history[t - shift]`, or EOS when that would cross a document start or
    /// an EOS token (the reference's `_shift_right`).
    fn shifted(&self, history: &[u32], shift: usize) -> Vec<u32> {
        if shift == 0 {
            return history.to_vec();
        }
        let mut out = Vec::with_capacity(history.len());
        // Tokens since the last EOS before position t (exclusive of that EOS).
        let mut since_eos: usize = 0;
        for (t, _) in history.iter().enumerate() {
            let ok = since_eos >= shift && t >= shift;
            out.push(if ok { history[t - shift] } else { self.eos });
            // Position t becomes part of the history for t + 1.
            if history[t] == self.eos {
                since_eos = 0;
            } else {
                since_eos += 1;
            }
        }
        out
    }

    /// Row ids for the last `n_new` tokens of `history` (which must hold at
    /// least [`Self::context_len`] tokens of context before them, padded with
    /// EOS at the very start of a text). Token-major: `heads()` ids per token.
    pub fn row_ids(&self, history: &[u32], n_new: usize) -> Vec<u64> {
        assert!(n_new <= history.len(), "n_new exceeds the history");
        let shifted: Vec<Vec<u32>> = (0..self.ngram_size)
            .map(|s| self.shifted(history, s))
            .collect();
        let start = history.len() - n_new;
        let mut out = Vec::with_capacity(n_new * self.heads());
        for t in start..history.len() {
            for order in 2..=self.ngram_size {
                let lo = (order - 2) * self.heads_per_ngram;
                let mut mixed = u64::from(shifted[0][t]).wrapping_mul(self.multipliers[0]);
                for (p, s) in shifted.iter().enumerate().take(order).skip(1) {
                    mixed ^= u64::from(s[t]).wrapping_mul(self.multipliers[p]);
                }
                for h in lo..lo + self.heads_per_ngram {
                    out.push(mixed % self.sizes[h] + self.offsets[h]);
                }
            }
        }
        out
    }
}

// ---- deriving the checkpoint's buffers from the config -------------------------------------

const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX1: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX2: u64 = 0x94D0_49BB_1331_11EB;
const PRIME_1: u64 = 10_007;

fn splitmix64(v: u64) -> u64 {
    let v = v.wrapping_add(GAMMA);
    let v = (v ^ (v >> 30)).wrapping_mul(MIX1);
    let v = (v ^ (v >> 27)).wrapping_mul(MIX2);
    v ^ (v >> 31)
}

fn is_prime(v: u64) -> bool {
    if v < 2 {
        return false;
    }
    if v.is_multiple_of(2) {
        return v == 2;
    }
    let mut d = 3;
    while d * d <= v {
        if v.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

fn nth_prime_after(start: u64, count: u64) -> u64 {
    let mut p = start;
    for _ in 0..count {
        p += 1;
        while !is_prime(p) {
            p += 1;
        }
    }
    p
}

/// The `(multipliers, sizes, offsets)` the reference builds from the config for
/// the n-gram embedding of PLE layer `ple_index`. A checkpoint stores these as
/// buffers; deriving them is the cross-check that the stored ones are the ones
/// the model was trained with.
pub fn derive_buffers(
    ngram_size: usize,
    heads_per_ngram: usize,
    vocab_size: usize,
    ngram_vocab_size_base: usize,
    seed: u64,
    ple_index: usize,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let heads = (ngram_size - 1) * heads_per_ngram;
    let mut sizes = Vec::with_capacity(heads);
    let mut offsets = Vec::with_capacity(heads);
    let mut total = 0u64;
    for h in 0..heads {
        let g = (ple_index * heads + h) as u64;
        let s = nth_prime_after(ngram_vocab_size_base as u64 - 1, g + 1);
        sizes.push(s);
        offsets.push(total);
        total += s;
    }
    let half = (((1u64 << 63) - 1) / vocab_size.max(1) as u64 / 2).max(1);
    let base_seed = seed + PRIME_1 * ple_index as u64;
    let multipliers = (0..ngram_size as u64)
        .map(|i| 2 * (splitmix64(base_seed.wrapping_add(GAMMA.wrapping_mul(i + 1))) % half) + 1)
        .collect();
    (multipliers, sizes, offsets)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped model's buffers, as stored in the checkpoint
    /// (`layer_multipliers`, and the first head sizes: consecutive primes after
    /// 20,000,000), against the formulas that derive them.
    #[test]
    fn derived_buffers_match_the_shipped_checkpoint() {
        let (m, sizes, offsets) = derive_buffers(3, 8, 248_320, 20_000_000, 1234, 0);
        assert_eq!(
            m,
            vec![23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
        );
        assert_eq!(sizes.len(), 16);
        assert_eq!(sizes[0], 20_000_003, "first prime above 20,000,000 - 1");
        assert!(sizes.windows(2).all(|w| w[0] < w[1]), "primes ascend");
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[1], sizes[0]);
        assert_eq!(offsets[15], sizes[..15].iter().sum::<u64>());
    }

    fn tiny() -> NgramHasher {
        let (m, s, o) = derive_buffers(3, 2, 128, 1000, 1234, 0);
        NgramHasher::new(3, 2, 7, m, s, o).unwrap()
    }

    #[test]
    fn a_predecessor_across_an_eos_reads_as_eos() {
        let h = tiny();
        // ids: a b EOS c d. After the EOS, `c` has no predecessors in its text.
        let hist = [1, 2, 7, 3, 4];
        assert_eq!(h.shifted(&hist, 1), vec![7, 1, 2, 7, 3]);
        // c (index 3): since_eos = 0 -> EOS; d (index 4): one token in -> shift 1 ok, shift 2 not.
        assert_eq!(h.shifted(&hist, 2), vec![7, 7, 1, 7, 7]);
    }

    #[test]
    fn only_the_new_tokens_are_hashed_and_they_depend_on_context() {
        let h = tiny();
        let ctx = [7u32, 7]; // start of text: padded with EOS
        let mut full = ctx.to_vec();
        full.extend([5, 6, 9]);
        let all = h.row_ids(&full, 3);
        assert_eq!(all.len(), 3 * h.heads());
        // Hashing the last token alone, with the two before it as context, gives
        // the same rows as the last three entries of the longer call.
        let tail = h.row_ids(&full, 1);
        assert_eq!(tail, all[2 * h.heads()..]);
        // A different predecessor changes the rows of the order-2 heads.
        let mut other = full.clone();
        other[3] = 11;
        assert_ne!(h.row_ids(&other, 1)[..2], tail[..2]);
        // Every id lies inside its head's range.
        for (i, id) in all.iter().enumerate() {
            let head = i % h.heads();
            assert!(*id >= h.offsets[head] && *id < h.offsets[head] + h.sizes[head]);
        }
    }

    #[test]
    fn inconsistent_buffers_are_rejected() {
        let (m, s, o) = derive_buffers(3, 2, 128, 1000, 1234, 0);
        assert!(NgramHasher::new(3, 2, 7, m[..2].to_vec(), s.clone(), o.clone()).is_err());
        assert!(NgramHasher::new(3, 2, 7, m.clone(), s[..3].to_vec(), o.clone()).is_err());
        let mut zero = s.clone();
        zero[0] = 0;
        assert!(NgramHasher::new(3, 2, 7, m, zero, o).is_err());
    }
}
