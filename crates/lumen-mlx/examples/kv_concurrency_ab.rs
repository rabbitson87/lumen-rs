//! 007 Phase 1 — measure the KV path that exists, before porting PagedAttention.
//!
//! Drives N concurrent sequences through the same interface the production
//! batched scheduler uses (`MlxBatchedSeqDriver`, i.e. `lumen-server::engine::
//! run_batched_mlx`), and reports what a paged KV cache would actually have to
//! beat: peak MLX active memory, aggregate tok/s, and per-seq tok/s at N = 1,
//! 2, 4, 8.
//!
//! ## Why this runs before any kernel is written
//!
//! `NativeKvCache` allocates one contiguous buffer per sequence per full-
//! attention layer, grown in `KV_CACHE_STEP` (256) token blocks. So a sequence
//! holding `L` real tokens occupies `ceil(L / 256) * 256` token slots. That
//! rounding is precisely what PagedAttention removes — a paged cache with a
//! 16-token block would occupy `ceil(L / 16) * 16` instead. The waste is
//! bounded at 255 tokens per sequence per layer, which is a large fraction of a
//! short chat turn and a rounding error on a long one.
//!
//! The harness therefore reports two independent things:
//!
//!   * **Analytic** — the rounding overhead, computed from the real token
//!     counts. Layer count, head count and dtype all cancel out of the ratio,
//!     so this number is exact and hardware-free.
//!   * **Measured** — peak `get_active_memory()` over the run, minus the
//!     post-load baseline. This catches what the analytic model cannot: the
//!     transient double-allocation when a buffer crosses a block boundary and
//!     is reallocated by `concatenate_axis`.
//!
//! If the analytic waste is small at realistic lengths *and* measured peak
//! memory tracks it, PagedAttention has little to win here and 007 should stop
//! at this phase with that written down. See `PLAN.md` "Ship criterion".
//!
//! ## Deliberate divergences from production, and why
//!
//!   * **EOS is ignored.** Every sequence generates exactly `--gen` tokens, so
//!     the batch width stays constant for the whole measured window. A seq
//!     retiring early would change the decode kernel's batch shape mid-run and
//!     make the throughput number a blend of two regimes.
//!   * **Prompts diverge at the first word.** Each sequence gets its own
//!     pseudo-random filler, so only the chat template header is shared. This
//!     keeps in-batch shared-prefix dedup (`LUMEN_MLX_SHARED_PREFIX`) from
//!     quietly doing the deduplication that paging is being evaluated for. The
//!     flag's value is printed either way.
//!   * **Decode is warmed per batch width.** MLX specializes kernels per
//!     distinct shape, and batch width N is part of that shape. Without a
//!     warmup at each N, the first measured step at every N pays compilation
//!     and the comparison across N is meaningless. (This is the mistake that
//!     made the 006 embedding port look 1.4x slower than Candle when it was in
//!     fact 4.6x faster.) Warmup uses short prompts at the same width, so
//!     *decode* numbers are warm; *prefill* numbers still include cold cost for
//!     that prompt length and are labelled as such. The warmup rows are also
//!     fed to the fit — see `fit_resident` for why they are load-bearing.
//!   * **One untimed settle step separates prefill from decode.** MLX holds the
//!     prefill logits live until the next eval retires them, so a memory read
//!     taken straight after prefill charges a `[1, prompt_len, vocab]` tensor
//!     to the KV column and reports a *negative* growth over the decode window.
//!   * **Memory is bounded, and the bound is reported.** Active memory is
//!     checked between prefill of each sequence and after each decode step
//!     against `--max-gb` (default 60% of physical RAM). Crossing it unwinds
//!     that width and stops the sweep, naming where it stopped: overshooting
//!     unified memory does not fail an allocation, it swaps, and a swapping
//!     decode loop reports the pager's throughput rather than the cache's.
//!
//! ## Usage
//!
//! ```text
//! MODEL_ID=~/models/Qwen3.5-9B-MTPLX-Speed \
//!   cargo run --release -p lumen-mlx --features mlx-native \
//!   --example kv_concurrency_ab -- --profile mixed --gen 64
//!
//! # short-turn profile — where block rounding hurts most
//! ... --example kv_concurrency_ab -- --profile short
//!
//! # explicit ladder, cycled to fill each batch
//! ... --example kv_concurrency_ab -- --lens 200,1500,400,3000
//!
//! # long context on a machine with room for it
//! ... --example kv_concurrency_ab -- --profile long --gen 32 --max-gb 40
//! ```
//!
//! ## Result on record
//!
//! Run on M3 Max / 36 GiB against Qwen3.5-9B at all three profiles, this said
//! **no** to the PagedAttention port that was going to consume it, and
//! `crates/paged-attention` was deleted rather than revived. Reclaimable
//! block-rounding slack topped out at **0.91% of process memory** (72.2 MB
//! short-turn / 65.8 MB mixed / 35.4 MB long, at N=8); 40-63% of per-sequence
//! residency is linear-attention conv/SSM state that paging cannot compact;
//! and the peak is set by prefill (11.5 GB at N=8 long, against a 3.1 GB decode
//! peak) rather than by the KV. Resident memory fits `53 MB * N + 67 KB * slots`
//! at R² >= 0.998 across all three profiles.
//!
//! That prefill peak is **per-chunk activations**, not a function of prompt
//! length: prefill is always chunked (`LUMEN_QWEN35_PREFILL_CHUNK`, default
//! 2048) and the peak tracks the chunk. Lowering it is a memory/latency trade
//! whose price depends on the prompt: free at 8K, expensive at 20K (+54 to
//! +118% prefill at chunk 512, because the cost tracks chunk *count* and each
//! chunk is `eval`'d before the next). `examples/prefill_chunk_equivalence.rs`
//! measures the exchange rate and proves the output is bit-identical across
//! chunk sizes; `docs/maintainer-workflow.md` §9 has the table. The default is
//! right — do not lower it globally.
//!
//! `docs/maintainer-workflow.md` §9 carries the summary and the commit to
//! recover the deleted crate from. Re-run this harness before reopening the
//! question — the answer would change for a non-hybrid model, where every
//! layer holds full-attention KV, or for very high concurrency on very short
//! turns (`--profile short --n 16,32`).

use std::time::Instant;

use anyhow::{Result, anyhow};
use lumen_mlx::metal_memory::{clear_cache, get_active_memory};
use lumen_mlx::{KV_CACHE_STEP, MlxBackend, MlxBatchedSeqDriver};

/// Block size a paged KV cache would use. vLLM ships 16; it is the comparison
/// point for the rounding-overhead column, not something this crate implements
/// yet.
const PAGED_BLOCK: usize = 16;

/// Batch widths measured, unless `--n` overrides.
const DEFAULT_WIDTHS: &[usize] = &[1, 2, 4, 8];

/// Prompt length ladders. Cycled when N exceeds the ladder length.
const PROFILE_SHORT: &[usize] = &[120, 190, 300, 150, 210, 130, 480, 170];
const PROFILE_MIXED: &[usize] = &[200, 800, 1500, 400, 3000, 260, 650, 1100];
const PROFILE_LONG: &[usize] = &[2000, 4000, 8000, 3000, 6000, 2500, 5000, 7000];

/// Filler vocabulary. Common short words so the tokenizer yields roughly one
/// token per word, which makes the length-targeting loop converge in 2-3 tries.
const WORDS: &[&str] = &[
    "system", "value", "record", "matrix", "signal", "buffer", "window", "kernel", "vector",
    "planet", "garden", "silver", "market", "reason", "letter", "number", "moment", "figure",
    "branch", "circle", "stream", "camera", "ticket", "island", "friend", "orange", "bridge",
    "packet", "column", "target", "sample", "region", "handle", "author", "series", "corner",
    "policy", "device", "格子", "회로", "réseau",
];

struct Args {
    model_id: String,
    widths: Vec<usize>,
    lens: Vec<usize>,
    profile: String,
    n_gen: usize,
    warmup: bool,
    /// Hard ceiling on MLX active memory. Crossing it aborts the current width
    /// instead of pushing the machine into swap.
    budget_bytes: usize,
}

/// Fraction of physical RAM the harness is allowed to reach. Deliberately
/// conservative: MLX allocates through unified memory, so overshooting does not
/// fail an allocation, it swaps — and a swapping decode loop reports a
/// throughput number that measures the pager, not the cache. (Decode collapsing
/// from 35 to 2.5 tok/s under memory pressure is a failure mode this project
/// has already shipped once.)
const DEFAULT_BUDGET_FRACTION: f64 = 0.60;

/// Physical RAM in bytes, or `None` if `sysctl` is unavailable.
fn physical_memory() -> Option<usize> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    String::from_utf8(out.stdout).ok()?.trim().parse().ok()
}

fn parse_args() -> Result<Args> {
    let argv: Vec<String> = std::env::args().collect();
    let mut a = Args {
        model_id: std::env::var("MODEL_ID")
            .map_err(|_| anyhow!("set MODEL_ID to a local model directory or an HF repo id"))?,
        widths: DEFAULT_WIDTHS.to_vec(),
        lens: PROFILE_MIXED.to_vec(),
        profile: "mixed".into(),
        n_gen: 64,
        warmup: true,
        budget_bytes: physical_memory()
            .map(|m| (m as f64 * DEFAULT_BUDGET_FRACTION) as usize)
            // No sysctl: assume the smallest Apple Silicon config rather than
            // the largest, so the fallback errs toward aborting early.
            .unwrap_or(16 * 1_000_000_000),
    };
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                a.model_id = argv
                    .get(i + 1)
                    .ok_or_else(|| anyhow!("--model needs a value"))?
                    .clone();
                i += 2;
            }
            "--n" => {
                a.widths = parse_usize_list(argv.get(i + 1).map(String::as_str).unwrap_or(""))?;
                i += 2;
            }
            "--gen" => {
                a.n_gen = argv
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| anyhow!("--gen needs a positive integer"))?;
                i += 2;
            }
            "--lens" => {
                a.lens = parse_usize_list(argv.get(i + 1).map(String::as_str).unwrap_or(""))?;
                a.profile = "custom".into();
                i += 2;
            }
            "--profile" => {
                let p = argv
                    .get(i + 1)
                    .ok_or_else(|| anyhow!("--profile needs short|mixed|long"))?;
                a.lens = match p.as_str() {
                    "short" => PROFILE_SHORT.to_vec(),
                    "mixed" => PROFILE_MIXED.to_vec(),
                    "long" => PROFILE_LONG.to_vec(),
                    other => return Err(anyhow!("unknown profile {other:?} (short|mixed|long)")),
                };
                a.profile = p.clone();
                i += 2;
            }
            "--max-gb" => {
                let v: f64 = argv
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| anyhow!("--max-gb needs a number of gigabytes"))?;
                a.budget_bytes = (v * 1e9) as usize;
                i += 2;
            }
            "--no-warmup" => {
                a.warmup = false;
                i += 1;
            }
            other => return Err(anyhow!("unknown argument {other:?}")),
        }
    }
    if a.widths.is_empty() {
        return Err(anyhow!("--n produced an empty list of batch widths"));
    }
    if a.lens.is_empty() {
        return Err(anyhow!("--lens produced an empty ladder"));
    }
    if a.n_gen == 0 {
        return Err(anyhow!("--gen must be at least 1"));
    }
    Ok(a)
}

fn parse_usize_list(s: &str) -> Result<Vec<usize>> {
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

/// Deterministic per-sequence filler. Seeded by `seq_ix` so two sequences in the
/// same batch diverge at the first word — the point is to *avoid* an accidental
/// shared prefix.
fn filler(seq_ix: usize, n_words: usize) -> String {
    let mut state =
        0x9E37_79B9_7F4A_7C15u64 ^ ((seq_ix as u64 + 1).wrapping_mul(0x517C_C1B7_2722_0A95));
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

/// Build a prompt for `seq_ix` whose rendered token count is at least
/// `target`, converging by measuring rather than guessing at the tokenizer's
/// words-per-token ratio.
fn build_prompt(
    driver: &dyn MlxBatchedSeqDriver,
    seq_ix: usize,
    target: usize,
) -> Result<Vec<u32>> {
    let mut words = target.max(8);
    let mut best: Option<Vec<u32>> = None;
    for _ in 0..8 {
        let msgs = vec![(
            "user".to_string(),
            format!("Summarize the following log:\n{}", filler(seq_ix, words)),
        )];
        let ids = driver.build_chat_input(&msgs, false)?;
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
    best.ok_or_else(|| anyhow!("could not reach {target} tokens for seq {seq_ix} in 8 rounds"))
}

/// `ceil(len / block) * block`, summed over sequences. Layer count, head count
/// and dtype cancel out of the ratio against `real`, so this is exact.
fn rounded_total(lens: &[usize], block: usize) -> usize {
    lens.iter().map(|&l| l.div_ceil(block) * block).sum()
}

fn active_bytes() -> usize {
    get_active_memory().unwrap_or(0)
}

fn gb(bytes: usize) -> f64 {
    bytes as f64 / 1e9
}

#[derive(Clone)]
struct Row {
    n: usize,
    prompt_tokens: usize,
    real_tokens: usize,
    alloc_step: usize,
    alloc_paged: usize,
    prefill_ms: f64,
    decode_ms: f64,
    agg_tps: f64,
    /// Active memory over baseline once prefill is done and the reusable pool
    /// has been drained. **Not** a KV number: prefill materializes a
    /// `[1, prompt_len, vocab]` logits tensor, which at this vocab size is
    /// larger than the KV it just wrote. Kept because that peak is a real
    /// serving constraint — just not one paging addresses.
    post_prefill: usize,
    /// Active memory over baseline after the decode window, pool drained. This
    /// is the resident KV: decode holds one row of logits, not `prompt_len`.
    kv_final: usize,
    /// High-water mark over the prefill loop.
    peak_prefill: usize,
    /// High-water mark over the decode loop only. What sustained concurrency
    /// actually costs, and the number a paged cache would have to beat.
    peak_decode: usize,
    /// Active memory over baseline after every seq is removed and the pool is
    /// drained. Should return to ~0; anything else is a leak.
    residual: usize,
}

/// What a width attempt produced: a full measurement, or an early stop because
/// continuing would have pushed the machine into swap.
enum Outcome {
    Done(Box<Row>),
    OverBudget { stage: String, active: usize },
}

/// Release every sequence this attempt created and drain the pool. Called on
/// the abort path so a width that ran out of budget does not poison the next
/// one — `remove_seq` tolerates ids that were never prefilled.
fn unwind(driver: &mut dyn MlxBatchedSeqDriver, ids: &[u64]) {
    for &id in ids {
        let _ = driver.remove_seq(id);
    }
    let _ = clear_cache();
}

/// One measured run at batch width `n`. `next_id` is threaded so warmup and
/// measured runs never reuse a seq id — `prefill` rejects ids the runner
/// already knows.
fn run_width(
    driver: &mut dyn MlxBatchedSeqDriver,
    lens: &[usize],
    n: usize,
    n_gen: usize,
    baseline: usize,
    budget: usize,
    next_id: &mut u64,
    label: &str,
) -> Result<Outcome> {
    let targets: Vec<usize> = (0..n).map(|i| lens[i % lens.len()]).collect();

    let mut ids = Vec::with_capacity(n);
    let mut prompts = Vec::with_capacity(n);
    for (i, &t) in targets.iter().enumerate() {
        prompts.push(build_prompt(driver, i, t)?);
        ids.push(*next_id);
        *next_id += 1;
    }

    let mut peak_prefill = active_bytes();

    let t_prefill = Instant::now();
    let mut last_tokens = Vec::with_capacity(n);
    let mut positions = Vec::with_capacity(n);
    for (row, &id) in ids.iter().enumerate() {
        let (tok, pos) = driver.prefill(id, &prompts[row])?;
        last_tokens.push(tok);
        positions.push(pos);
        let active = active_bytes();
        peak_prefill = peak_prefill.max(active);
        // Checked per sequence, not per width: prefill is where the biggest
        // single allocation happens (a `[1, prompt_len, vocab]` logits tensor),
        // so the ceiling has to be enforced between sequences to stop before
        // the next one commits.
        if active > budget {
            unwind(driver, &ids);
            return Ok(Outcome::OverBudget {
                stage: format!(
                    "prefill of seq {}/{n} ({} tokens)",
                    row + 1,
                    prompts[row].len()
                ),
                active,
            });
        }
    }
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

    // One untimed settle step. `clear_cache` alone does not drop the prefill
    // logits — MLX still holds them live until the next eval retires that part
    // of the graph — so measuring straight after prefill charges a
    // `[1, prompt_len, vocab]` tensor to the KV column. Stepping once first,
    // then draining, leaves the actual resident KV.
    let settle = driver.decode_step_batch(&ids, &last_tokens, &positions)?;
    for (row, (tok, pos)) in settle.into_iter().enumerate() {
        last_tokens[row] = tok;
        positions[row] = pos;
    }
    let _ = clear_cache();
    let post_prefill = active_bytes().saturating_sub(baseline);

    // Sampled inside the decode loop only, so it measures decode's own
    // high-water mark rather than inheriting prefill's.
    let mut peak_decode = 0usize;

    // EOS deliberately ignored: a fixed token budget keeps the batch width —
    // and therefore the decode kernel's shape — constant across the window.
    let t_decode = Instant::now();
    for step in 0..n_gen {
        let out = driver.decode_step_batch(&ids, &last_tokens, &positions)?;
        if out.len() != n {
            return Err(anyhow!(
                "decode_step_batch returned {} rows for a batch of {n}",
                out.len()
            ));
        }
        for (row, (tok, pos)) in out.into_iter().enumerate() {
            last_tokens[row] = tok;
            positions[row] = pos;
        }
        let active = active_bytes();
        peak_decode = peak_decode.max(active);
        if active > budget {
            unwind(driver, &ids);
            return Ok(Outcome::OverBudget {
                stage: format!("decode step {}/{n_gen}", step + 1),
                active,
            });
        }
    }
    let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;

    let _ = clear_cache();
    let kv_final = active_bytes().saturating_sub(baseline);

    for &id in &ids {
        driver.remove_seq(id)?;
    }
    let _ = clear_cache();
    let residual = active_bytes().saturating_sub(baseline);

    let prompt_tokens: usize = prompts.iter().map(Vec::len).sum();
    // `+ 1` for the untimed settle step, which is a real cached token.
    let final_lens: Vec<usize> = prompts.iter().map(|p| p.len() + 1 + n_gen).collect();
    let real_tokens: usize = final_lens.iter().sum();
    let row = Row {
        n,
        prompt_tokens,
        real_tokens,
        alloc_step: rounded_total(&final_lens, KV_CACHE_STEP),
        alloc_paged: rounded_total(&final_lens, PAGED_BLOCK),
        prefill_ms,
        decode_ms,
        agg_tps: (n * n_gen) as f64 / (decode_ms / 1000.0),
        post_prefill,
        kv_final,
        peak_prefill: peak_prefill.saturating_sub(baseline),
        peak_decode: peak_decode.saturating_sub(baseline),
        residual,
    };

    println!(
        "{label} N={n:<2} prompt={prompt_tokens:>6} real={real_tokens:>6}  \
         alloc@{KV_CACHE_STEP}={:>6} (+{:>5.1}%)  alloc@{PAGED_BLOCK}={:>6} (+{:>4.1}%)",
        row.alloc_step,
        pct(row.alloc_step, row.real_tokens),
        row.alloc_paged,
        pct(row.alloc_paged, row.real_tokens),
    );
    println!(
        "{:width$} prefill={prefill_ms:>8.1}ms (cold)  decode={decode_ms:>8.1}ms  \
         agg={:>6.1} tok/s  per-seq={:>5.1} tok/s",
        "",
        row.agg_tps,
        row.agg_tps / n as f64,
        width = label.len(),
    );
    println!(
        "{:width$} over baseline: post-prefill={:>7.1} MB (peak {:>7.1})  resident-KV={:>7.1} MB \
         (decode peak {:>7.1})  residual={:>5.1} MB  {:>6.1} KB per allocated slot",
        "",
        row.post_prefill as f64 / 1e6,
        row.peak_prefill as f64 / 1e6,
        row.kv_final as f64 / 1e6,
        row.peak_decode as f64 / 1e6,
        residual as f64 / 1e6,
        row.kv_final as f64 / row.alloc_step.max(1) as f64 / 1e3,
        width = label.len(),
    );
    Ok(Outcome::Done(Box::new(row)))
}

/// Resident memory decomposed as `intercept + per_seq * N + per_slot * slots`.
///
/// A single-predictor fit against slots alone is wrong on a hybrid model and
/// wrong in a way that flatters this task: linear-attention layers hold a
/// fixed-size conv/SSM state **per sequence**, independent of context length,
/// and on a Qwen 3.5/3.6 that state is the larger share at realistic batch
/// widths. Folding it into the per-slot term inflates the slope, and the slope
/// is exactly what converts "block rounding wastes X%" into reclaimable
/// megabytes. Separating the two keeps the answer honest — and the per-seq term
/// is memory PagedAttention does not address at all, since paging is a
/// full-attention KV technique.
struct Fit {
    /// Bytes per sequence, independent of its length.
    per_seq: f64,
    /// Bytes per allocated full-attention cache slot, across every KV layer.
    per_slot: f64,
    r2: f64,
    /// Largest single-point deviation, as a fraction of that point's measured
    /// value. R² alone hides a bad fit at small N behind the large-N points.
    worst_rel_err: f64,
    n_points: usize,
}

/// Least squares of resident bytes on `[N, allocated_slots]`, **with no
/// intercept**.
///
/// Zero sequences holding zero slots cost zero bytes over baseline, so a free
/// constant has no physical referent here. Fitting one anyway produced a
/// spurious ~78 MB term that the solver paid for by bending the other two
/// coefficients, mispredicting every small-N point by 40-100% while R² still
/// read 0.965 off the large-N points alone. Dropping it fits the same data to
/// ~1%.
///
/// Needs points where `slots / N` varies, otherwise the two predictors are
/// collinear and the split between them is arbitrary. The warmup runs supply
/// exactly that leverage: they hold every width at the minimum one block per
/// sequence, so passing them in alongside the measured runs is what makes the
/// decomposition identifiable.
fn fit_resident(rows: &[Row]) -> Option<Fit> {
    if rows.len() < 3 {
        return None;
    }
    let (mut snn, mut sns, mut sss, mut sny, mut ssy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for r in rows {
        let (n, s, y) = (r.n as f64, r.alloc_step as f64, r.kv_final as f64);
        snn += n * n;
        sns += n * s;
        sss += s * s;
        sny += n * y;
        ssy += s * y;
    }
    let det = snn * sss - sns * sns;
    if det.abs() < 1e-6 {
        return None;
    }
    let per_seq = (sny * sss - ssy * sns) / det;
    let per_slot = (ssy * snn - sny * sns) / det;

    let mean_y = rows.iter().map(|r| r.kv_final as f64).sum::<f64>() / rows.len() as f64;
    let ss_tot: f64 = rows
        .iter()
        .map(|r| (r.kv_final as f64 - mean_y).powi(2))
        .sum();
    let mut ss_res = 0.0f64;
    let mut worst_rel_err = 0.0f64;
    for r in rows {
        let pred = per_seq * r.n as f64 + per_slot * r.alloc_step as f64;
        let resid = r.kv_final as f64 - pred;
        ss_res += resid * resid;
        if r.kv_final > 0 {
            worst_rel_err = worst_rel_err.max((resid / r.kv_final as f64).abs());
        }
    }
    let r2 = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };
    Some(Fit {
        per_seq,
        per_slot,
        r2,
        worst_rel_err,
        n_points: rows.len(),
    })
}

fn pct(allocated: usize, real: usize) -> f64 {
    if real == 0 {
        return 0.0;
    }
    (allocated as f64 - real as f64) / real as f64 * 100.0
}

fn env_str(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| "<unset>".into())
}

fn main() -> Result<()> {
    let args = parse_args()?;

    println!("--- 007 Phase 1 — KV concurrency baseline (native_cache) ---");
    println!("model                          = {}", args.model_id);
    println!(
        "profile                        = {} {:?}",
        args.profile, args.lens
    );
    println!("batch widths                   = {:?}", args.widths);
    println!("generated tokens per seq       = {}", args.n_gen);
    println!("KV_CACHE_STEP                  = {KV_CACHE_STEP}");
    println!("paged block (hypothetical)     = {PAGED_BLOCK}");
    println!(
        "LUMEN_MLX_SHARED_PREFIX        = {}",
        env_str("LUMEN_MLX_SHARED_PREFIX")
    );
    println!(
        "LUMEN_NATIVE_KV_STEP_PREALLOC  = {} (default on)",
        env_str("LUMEN_NATIVE_KV_STEP_PREALLOC")
    );
    println!(
        "LUMEN_MLX_KV_QUANT             = {}",
        env_str("LUMEN_MLX_KV_QUANT")
    );
    // First-order control on the prefill peak: prefill is always chunked, and
    // the peak scales with the chunk, not with prompt length.
    println!(
        "LUMEN_QWEN35_PREFILL_CHUNK     = {} (default 2048)",
        env_str("LUMEN_QWEN35_PREFILL_CHUNK")
    );
    match physical_memory() {
        Some(m) => println!(
            "memory budget                  = {:.2} GB of {:.2} GB physical ({:.0}%)",
            gb(args.budget_bytes),
            gb(m),
            args.budget_bytes as f64 / m as f64 * 100.0,
        ),
        None => println!(
            "memory budget                  = {:.2} GB (physical RAM unknown)",
            gb(args.budget_bytes)
        ),
    }

    let t_load = Instant::now();
    let mut backend = MlxBackend::load(&args.model_id)?;
    println!(
        "loaded in {:.1}s (family={:?})",
        t_load.elapsed().as_secs_f64(),
        backend.kind()
    );

    let post_load = active_bytes();
    let driver = backend.batched_seq_driver_mut();
    let mut next_id: u64 = 1;

    // MLX loads weights lazily: nothing is resident until the first forward
    // touches it. Reading the baseline straight after `load()` would report ~0
    // and then charge the entire weight set to the first run's KV column. So
    // prime with one throwaway sequence, then take the floor.
    run_width(
        driver,
        &[64],
        1,
        2,
        0,
        args.budget_bytes,
        &mut next_id,
        "  prime  ",
    )?;
    let _ = clear_cache();
    let baseline = active_bytes();
    println!(
        "active memory: post-load={:.2} GB → post-prime={:.2} GB (baseline: weights + runtime)",
        gb(post_load),
        gb(baseline),
    );
    if baseline >= args.budget_bytes {
        return Err(anyhow!(
            "the model alone holds {:.2} GB, at or above the {:.2} GB budget — raise --max-gb \
             or pick a smaller model",
            gb(baseline),
            gb(args.budget_bytes),
        ));
    }
    println!(
        "headroom for KV within budget    = {:.2} GB",
        gb(args.budget_bytes - baseline)
    );
    println!();

    let mut rows = Vec::new();
    // Measured rows plus warmups. Warmups pin every width at one block per
    // sequence, which is the only reason the per-seq and per-slot terms of the
    // fit can be told apart — see `fit_resident`.
    let mut fit_rows: Vec<Row> = Vec::new();
    let mut skipped: Vec<(usize, String)> = Vec::new();

    for &n in &args.widths {
        if args.warmup {
            // Same batch width, tiny prompts: specializes the decode kernel for
            // this shape without paying a second full prefill.
            let warm_lens = vec![64usize];
            if let Outcome::Done(warm) = run_width(
                driver,
                &warm_lens,
                n,
                3,
                baseline,
                args.budget_bytes,
                &mut next_id,
                "  warmup ",
            )? {
                fit_rows.push(*warm);
            }
            let _ = clear_cache();
        }
        let outcome = run_width(
            driver,
            &args.lens,
            n,
            args.n_gen,
            baseline,
            args.budget_bytes,
            &mut next_id,
            "  measure",
        )?;
        match outcome {
            Outcome::Done(row) => {
                fit_rows.push((*row).clone());
                rows.push(*row);
            }
            Outcome::OverBudget { stage, active } => {
                // Reported, never silently dropped: a truncated sweep that reads
                // as a complete one is exactly the green-by-omission failure this
                // project is trying to remove.
                println!(
                    "  measure N={n:<2} ABORTED at {stage}: active {:.2} GB crossed the \
                     {:.2} GB budget",
                    gb(active),
                    gb(args.budget_bytes),
                );
                skipped.push((n, stage));
                let _ = clear_cache();
                println!();
                println!(
                    "Stopping the sweep here — every larger width costs strictly more. \
                     Raise the ceiling with --max-gb if this machine can take it."
                );
                break;
            }
        }
        let _ = clear_cache();
        println!();
    }

    println!("--- summary ---");
    println!(
        "{:>3}  {:>7}  {:>7}  {:>9}  {:>8}  {:>10}  {:>9}  {:>9}  {:>8}  {:>8}  {:>9}  {:>9}",
        "N",
        "prompt",
        "real",
        "waste@256",
        "waste@16",
        "prefill ms",
        "decode ms",
        "agg tok/s",
        "per-seq",
        "KV MB",
        "KB/slot",
        "peak dec",
    );
    for r in &rows {
        println!(
            "{:>3}  {:>7}  {:>7}  {:>8.1}%  {:>7.1}%  {:>10.1}  {:>9.1}  {:>9.1}  {:>8.1}  {:>8.1}  {:>9.1}  {:>9.1}",
            r.n,
            r.prompt_tokens,
            r.real_tokens,
            pct(r.alloc_step, r.real_tokens),
            pct(r.alloc_paged, r.real_tokens),
            r.prefill_ms,
            r.decode_ms,
            r.agg_tps,
            r.agg_tps / r.n as f64,
            r.kv_final as f64 / 1e6,
            r.kv_final as f64 / r.alloc_step.max(1) as f64 / 1e3,
            r.peak_decode as f64 / 1e6,
        );
    }
    // Raw KB/slot is not constant across widths: resident memory is a fixed
    // per-run overhead plus a per-slot cost. Fitting both separates them, and
    // the fit's R² says whether the model is trustworthy enough to convert the
    // analytic waste percentages into real megabytes below.
    let fit = fit_resident(&fit_rows);
    if let Some(f) = &fit {
        println!(
            "resident ≈ {:.1} MB per seq + {:.1} KB per allocated slot  \
             (R²={:.4}, worst point off by {:.1}%, over {} points incl. warmups)",
            f.per_seq / 1e6,
            f.per_slot / 1e3,
            f.r2,
            f.worst_rel_err * 100.0,
            f.n_points,
        );
        println!(
            "  the per-seq term is length-independent state (linear-attention conv/SSM slots on \
             this family) — paging compacts full-attention KV only, so it cannot touch that term."
        );
        if f.r2 < 0.98 || f.worst_rel_err > 0.25 {
            println!(
                "  ^ fit is not tight (R² < 0.98 or a point off by > 25%): treat the \
                 reclaimable-MB figure below as indicative only."
            );
        }
    }
    if let Some(worst) = rows.iter().map(|r| r.residual).max() {
        println!(
            "residual after release across all widths: max {:.1} MB (0 means no leak)",
            worst as f64 / 1e6
        );
    }
    for (n, stage) in &skipped {
        println!("N={n} NOT MEASURED — hit the memory budget during {stage}");
    }
    let unattempted: Vec<usize> = args
        .widths
        .iter()
        .copied()
        .filter(|w| !rows.iter().any(|r| r.n == *w) && !skipped.iter().any(|(n, _)| n == w))
        .collect();
    if !unattempted.is_empty() {
        println!("widths never attempted after the abort: {unattempted:?}");
    }

    println!();
    println!("--- what this decides (007 PLAN.md, Phase 1) ---");
    if let Some(top) = rows.last() {
        let recoverable =
            pct(top.alloc_step, top.real_tokens) - pct(top.alloc_paged, top.real_tokens);
        println!(
            "At N={}, block rounding costs {:.1}% of real KV; a {PAGED_BLOCK}-token paged block \
             would recover {:.1} of those points.",
            top.n,
            pct(top.alloc_step, top.real_tokens),
            recoverable,
        );
        match &fit {
            Some(f) => {
                let reclaim = f.per_slot * (top.alloc_step - top.alloc_paged) as f64;
                let seq_term = f.per_seq * top.n as f64;
                println!(
                    "Resident at N={} is {:.1} MB, of which {:.1} MB ({:.0}%) is per-sequence \
                     state paging cannot compact.",
                    top.n,
                    top.kv_final as f64 / 1e6,
                    seq_term / 1e6,
                    seq_term / top.kv_final.max(1) as f64 * 100.0,
                );
                println!(
                    "Moving from {}-token to {PAGED_BLOCK}-token blocks drops {} allocated slots \
                     to {}, reclaiming {:.1} MB — {:.1}% of resident, {:.2}% of the {:.1} GB the \
                     process already holds.",
                    KV_CACHE_STEP,
                    top.alloc_step,
                    top.alloc_paged,
                    reclaim / 1e6,
                    reclaim / top.kv_final.max(1) as f64 * 100.0,
                    reclaim / (baseline + top.kv_final) as f64 * 100.0,
                    gb(baseline + top.kv_final),
                );
            }
            None => println!(
                "Resident KV at N={} is {:.1} MB (need ≥2 batch widths to fit a per-slot cost).",
                top.n,
                top.kv_final as f64 / 1e6,
            ),
        }
        println!(
            "Prefill peaks at {:.1} MB over baseline against a decode peak of {:.1} MB. \
             The prefill peak is per-chunk activations — it scales with \
             LUMEN_QWEN35_PREFILL_CHUNK (currently {}), not with prompt length, and paging \
             does not address it either way.",
            top.peak_prefill as f64 / 1e6,
            top.peak_decode as f64 / 1e6,
            env_str("LUMEN_QWEN35_PREFILL_CHUNK"),
        );
        println!(
            "Decision rule: if that recoverable fraction is small at the lengths this server \
             actually serves, PagedAttention has little to win and 007 stops here — record the \
             numbers in CONTEXT.md and leave the crate parked."
        );
    }
    println!(
        "Note: prefill timings include per-length kernel specialization (cold); decode timings \
         are warmed at each batch width."
    );

    Ok(())
}
