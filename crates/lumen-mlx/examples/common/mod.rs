//! Helpers shared by the KV and prefill A/B harnesses — `kv_bf16_ab`,
//! `kv_concurrency_ab` and `prefill_chunk_equivalence` — each of which pulls
//! this in with `mod common;`. A directory without a `main.rs` is not an
//! example target of its own.
//!
//! The filler and the length-targeting loop are shared on purpose: a prompt of
//! a given target length is then the same prompt in every harness, so their
//! numbers are comparable.

// Each harness uses a different subset; whatever one of them leaves unused is
// still used by another.
#![allow(dead_code)]

use anyhow::{Result, anyhow};
use lumen_mlx::MlxBatchedSeqDriver;
use lumen_mlx::metal_memory::{clear_cache, get_active_memory};

/// Filler vocabulary. Common short words so the tokenizer yields roughly one
/// token per word, which makes the length-targeting loop converge in 2-3 tries.
pub const WORDS: &[&str] = &[
    "system", "value", "record", "matrix", "signal", "buffer", "window", "kernel", "vector",
    "planet", "garden", "silver", "market", "reason", "letter", "number", "moment", "figure",
    "branch", "circle", "stream", "camera", "ticket", "island", "friend", "orange", "bridge",
    "packet", "column", "target", "sample", "region", "handle", "author", "series", "corner",
    "policy", "device", "格子", "회로", "réseau",
];

/// The filler seed of a single-sequence harness.
pub const FILLER_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// The filler seed of row `seq_ix` in a batch. Distinct per row so two
/// sequences in the same batch diverge at the first word — the point is to
/// *avoid* an accidental shared prefix.
pub fn seq_seed(seq_ix: usize) -> u64 {
    FILLER_SEED ^ ((seq_ix as u64 + 1).wrapping_mul(0x517C_C1B7_2722_0A95))
}

/// `n_words` words drawn deterministically from [`WORDS`], starting from
/// `seed`.
pub fn filler(seed: u64, n_words: usize) -> String {
    let mut state = seed;
    let mut out = String::with_capacity(n_words * 7);
    for w in 0..n_words {
        // xorshift64* — deterministic, no dependency, good enough for filler.
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let pick = (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as usize;
        if w > 0 {
            out.push(' ');
        }
        out.push_str(WORDS[pick % WORDS.len()]);
    }
    out
}

/// Build a prompt from the filler for `seed` whose rendered token count is at
/// least `target`, converging by measuring rather than guessing at the
/// tokenizer's words-per-token ratio.
pub fn build_prompt(
    driver: &dyn MlxBatchedSeqDriver,
    seed: u64,
    target: usize,
) -> Result<Vec<u32>> {
    let mut words = target.max(8);
    let mut best: Option<Vec<u32>> = None;
    for _ in 0..8 {
        let msgs = vec![(
            "user".to_string(),
            format!("Summarize the following log:\n{}", filler(seed, words)),
        )];
        let ids = driver.build_chat_input(&msgs, false, None)?;
        let n = ids.len();
        if n >= target {
            // First length at or above target wins; overshoot past +5% means
            // stepping back down for a tighter fit is worth one more round.
            if n <= target + target / 20 {
                return Ok(ids);
            }
            best = Some(ids);
            words = ((words as f64) * (target as f64 / n as f64)).ceil() as usize + 1;
        } else {
            let grow = ((words as f64) * (target as f64 / n.max(1) as f64)).ceil() as usize;
            words = grow.max(words + 4);
        }
    }
    best.ok_or_else(|| {
        anyhow!("could not reach {target} tokens in 8 rounds (filler seed {seed:#x})")
    })
}

/// Active memory once the reusable pool has actually stopped shrinking.
///
/// A single `clear_cache()` is not enough: it frees already-released buffers,
/// and MLX does not always release everything by the time the call returns.
/// Measured, this shows up as a bimodal reading — the *same* condition at the
/// *same* operating point reports either 190.6 MB or 317.9 MB, a fixed ~127 MB
/// quantum that is present or absent. Pairing a clean reading of one condition
/// with a contaminated reading of the other manufactures a 32 KB/slot saving
/// into a 29 KB/slot regression, which is exactly the sort of artifact that
/// reads as a real result.
///
/// Looping until two consecutive readings agree collapses it. The minimum is
/// returned because the contaminant only ever adds.
pub fn settled_active() -> usize {
    // The wait is load-bearing, not defensive. `LUMEN_NATIVE_DEFER_CLEAR_CACHE`
    // is on by default, so `remove_seq` hands the clear to a background worker
    // (~45 ms). Two back-to-back readings taken before that worker runs are
    // equal to each other and both wrong, which is how a settle loop without a
    // sleep exits early on the contaminated value.
    let mut best = usize::MAX;
    let mut prev = usize::MAX;
    let mut stable = 0;
    for _ in 0..12 {
        let _ = clear_cache();
        let now = get_active_memory().unwrap_or(0);
        best = best.min(now);
        stable = if now == prev { stable + 1 } else { 0 };
        if stable >= 2 {
            break;
        }
        prev = now;
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    best
}

/// A comma-separated list of positive integers, e.g. `--widths 1,2,4,8`.
pub fn parse_usize_list(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(|p| {
            p.parse::<usize>().map_err(|_| {
                anyhow!("expected a comma-separated list of positive integers, got {s:?}")
            })
        })
        .collect()
}
