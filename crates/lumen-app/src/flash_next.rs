//! Qwen3.8-Flash-Next (`qwen4-exp`) on this Mac: what it needs from the machine, what decode
//! speed that buys, and a read-speed probe for the volume that holds its weights.
//!
//! The server keeps ~3.6 GB of the model in memory and serves its routed experts (68 GB of the
//! 97 GB) from the SSD through a pool of memory slots (`docs/flash-next.md`). The questions that
//! decide whether it runs well are therefore not "does it fit" but "what does a prefill reserve
//! out of the GPU's working set, and how much is left for the pool", and "can the disk feed it".
//!
//! The memory plan mirrors `plan_memory` in the server's `qwen4_exp/runner.rs`, and the numbers
//! mirror `frontend/src/lib/memory-estimate.ts` (the frontend test reads this file and fails when
//! they differ). Change them together.

use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::catalog::Catalog;
use crate::config::PersistentConfig;
use crate::models;

// ── What the model needs ─────────────────────────────────────────────

/// Resident weights: embeddings, attention, DeltaNet, routers, shared experts.
pub const TRUNK_BYTES: u64 = 3_600_000_000;
/// One routed expert, as the server reads it on a miss.
pub const EXPERT_BYTES: u64 = 2_760_000;
/// The smallest expert pool the server runs with: a staging region plus a layer beside it,
/// 2 x 512 slots.
pub const POOL_FLOOR_BYTES: u64 = 2_826_240_000;
/// With room for three layers (1,536 slots) the pool lends a second staging region, and the
/// server reads the next layer while it computes the current one. Below it prefill is slower,
/// nothing fails.
pub const POOL_PREFETCH_BYTES: u64 = 4_239_360_000;

/// Tokens per prefill chunk when `LUMEN_QWEN35_PREFILL_CHUNK` is unset; a tight machine takes half.
pub const DEFAULT_CHUNK: u64 = 1024;
pub const TIGHT_CHUNK: u64 = 512;
/// The longest prompt the plan holds KV for when `LUMEN_MAX_PROMPT_TOKENS` / `LUMEN_MAX_CTX` is
/// unset; a tight machine takes half.
pub const DEFAULT_CONTEXT: u64 = 16_384;
pub const TIGHT_CONTEXT: u64 = 8_192;
/// The model's context limit.
pub const MAX_CONTEXT: u64 = 262_144;
/// A GPU that may use less than this is "tight" (a 16 GB Mac): every reserve is trimmed.
pub const TIGHT_BELOW_BYTES: u64 = 11_000_000_000;
/// Most bytes of attention scores a chunk holds at once; longer contexts attend in blocks.
const SCORES_BUDGET_BYTES: u64 = 2_000_000_000;
const TIGHT_SCORES_BUDGET_BYTES: u64 = 800_000_000;
/// Fixed prefill buffers, plus the chunk's activations per token.
const RESERVE_BASE_BYTES: u64 = 600_000_000;
const RESERVE_PER_TOKEN_BYTES: u64 = 1_100_000;
/// KV bytes of one token of context: 13 layers (12 full-attention + the draft head's) x
/// (K and V of 2 heads x 256 dims in f32 + the indexer's 128-dim key in f32).
pub const KV_BYTES_PER_TOKEN: u64 = 59_904;
const ATTENTION_HEADS: u64 = 24;
/// The GPU working set as a share of RAM (Apple reports about 2/3 up to 32 GiB and 3/4 above;
/// a 36 GB Mac is really ~0.78), and the share of it the server lets itself fill
/// (`WORKING_SET_SHARE` in lumen-mlx).
const WORKING_SET_SHARE_SMALL: f64 = 2.0 / 3.0;
const WORKING_SET_SHARE_LARGE: f64 = 0.75;
const SMALL_RAM_BYTES: u64 = 32 << 30;
const WORKING_SET_FILLABLE: f64 = 0.88;
/// The server's `LUMEN_MEMORY_LIMIT_GB` default, in GiB.
const MEMORY_LIMIT_DEFAULT_GIB: u64 = 32;

/// `experts.lpack`, as built from the shipped weights (68,048,453,632 bytes).
pub const PACK_BYTES: u64 = 68_100_000_000;
/// A pack smaller than this is an interrupted copy; the server ignores it.
pub const PACK_COMPLETE_MIN_BYTES: u64 = 60_000_000_000;

/// The part of `ram_bytes` the server may fill, lowered to `memory_limit` (what
/// `LUMEN_MEMORY_LIMIT_GB` allows) when there is one.
pub fn usable_bytes(ram_bytes: u64, memory_limit: Option<u64>) -> u64 {
    let share = if ram_bytes <= SMALL_RAM_BYTES {
        WORKING_SET_SHARE_SMALL
    } else {
        WORKING_SET_SHARE_LARGE
    };
    let usable = (ram_bytes as f64 * share * WORKING_SET_FILLABLE) as u64;
    memory_limit.map_or(usable, |limit| usable.min(limit))
}

/// What the memory is spent on, for a GPU that may use `usable` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryPlan {
    pub chunk: u64,
    /// Longest prompt in tokens the plan holds the KV cache and attention scores for.
    pub ctx_tokens: u64,
    /// Memory a prefill needs beyond the trunk and the expert pool.
    pub reserve: u64,
    /// What `auto` gives the expert pool: the rest. Below [`POOL_FLOOR_BYTES`] the model does
    /// not fit (the server raises the pool to the floor and warns).
    pub pool: u64,
}

impl MemoryPlan {
    /// The pool the server actually runs with.
    pub fn pool_bytes(&self) -> u64 {
        self.pool.max(POOL_FLOOR_BYTES)
    }

    /// The trunk, the smallest pool and the reserve fit the usable memory.
    pub fn fits(&self) -> bool {
        self.pool >= POOL_FLOOR_BYTES
    }
}

/// Memory a prefill needs beyond the trunk and the pool: fixed buffers, the chunk's activations,
/// the KV cache of the longest prompt, and half the attention scores.
fn prefill_reserve(chunk: u64, ctx_tokens: u64, scores_budget: u64) -> u64 {
    let scores = (ATTENTION_HEADS * chunk * ctx_tokens * 4).min(scores_budget);
    RESERVE_BASE_BYTES
        + chunk * RESERVE_PER_TOKEN_BYTES
        + ctx_tokens * KV_BYTES_PER_TOKEN
        + scores / 2
}

/// The prefill chunk the server takes on a GPU that may use `usable` bytes when nothing sets it.
pub fn default_chunk(usable: u64) -> u64 {
    if usable < TIGHT_BELOW_BYTES {
        TIGHT_CHUNK
    } else {
        DEFAULT_CHUNK
    }
}

/// The longest prompt the server plans for on a GPU that may use `usable` bytes when nothing
/// sets it (the same rule as its `plan_memory`): 16,384 tokens, 8,192 on a tight machine.
pub fn default_context(usable: u64) -> u64 {
    if usable < TIGHT_BELOW_BYTES {
        TIGHT_CONTEXT
    } else {
        DEFAULT_CONTEXT
    }
}

/// The server's plan for a GPU that may use `usable` bytes: `chunk` and `ctx_tokens` are what
/// `LUMEN_QWEN35_PREFILL_CHUNK` and `LUMEN_MAX_PROMPT_TOKENS` / `LUMEN_MAX_CTX` set, else the
/// machine's defaults.
pub fn plan_memory(usable: u64, chunk: Option<u64>, ctx_tokens: Option<u64>) -> MemoryPlan {
    let tight = usable < TIGHT_BELOW_BYTES;
    let chunk = chunk.unwrap_or_else(|| default_chunk(usable));
    let ctx_tokens = ctx_tokens.unwrap_or_else(|| default_context(usable));
    let budget = if tight {
        TIGHT_SCORES_BUDGET_BYTES
    } else {
        SCORES_BUDGET_BYTES
    };
    let reserve = prefill_reserve(chunk, ctx_tokens, budget);
    MemoryPlan {
        chunk,
        ctx_tokens,
        reserve,
        pool: usable.saturating_sub(TRUNK_BYTES + reserve),
    }
}

/// The longest prompt the plan can hold with the pool at its floor, for this chunk (the
/// machine's default when `None`). 0 when not even an empty prompt fits.
pub fn max_fitting_context(usable: u64, chunk: Option<u64>) -> u64 {
    if !plan_memory(usable, chunk, Some(0)).fits() {
        return 0;
    }
    let (mut lo, mut hi) = (0, MAX_CONTEXT);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if plan_memory(usable, chunk, Some(mid)).fits() {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// Whether a Mac with `ram_bytes` runs the model at its default plan.
pub fn fits(ram_bytes: u64) -> bool {
    plan_memory(usable_bytes(ram_bytes, None), None, None).fits()
}

/// RAM below which the model cannot run at the default plan, rounded up to whole GiB.
pub fn min_ram_gib() -> u64 {
    (1..=1024).find(|gib| fits(gib << 30)).unwrap_or(1024)
}

/// The longest prompt the server plans memory for, from the two things it reads (`context_override`
/// in the server's `qwen4_exp/runner.rs`): its prompt cap (`LUMEN_MAX_PROMPT_TOKENS`, else the
/// legacy `LUMEN_PREFILL_CHUNK`, in the order its prompt guard reads them), lowered to
/// `LUMEN_MAX_CTX` when that is smaller. `LUMEN_MAX_CTX` alone never raises it: the guard would
/// still reject a prompt over the cap.
pub fn planned_context(prompt_cap: Option<u64>, max_ctx: Option<u64>) -> Option<u64> {
    match (prompt_cap, max_ctx) {
        (Some(cap), Some(ctx)) => Some(cap.min(ctx)),
        (Some(cap), None) => Some(cap),
        (None, Some(ctx)) => Some(ctx.min(DEFAULT_CONTEXT)),
        (None, None) => None,
    }
}

/// What the app's launch hands the server's memory plan: the three things it reads that the
/// app can set. For Flash-Next the app sends `LUMEN_MAX_PROMPT_TOKENS` itself (the machine's
/// default, see [`default_context`]) unless the Flash-Next settings or an env override set it, so
/// the plan is not capped by the Prefill card, which is the other families' setting and which the
/// server would otherwise read as the prompt cap. The Max context card (`LUMEN_MAX_CTX`) only
/// lowers it. And, unless the memory caps are disabled, `LUMEN_MEMORY_LIMIT_GB`. An env override
/// of the same name wins over the typed setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchPlan {
    pub memory_limit: Option<u64>,
    pub chunk: Option<u64>,
    pub ctx_tokens: Option<u64>,
}

/// The Flash-Next variables a launch carries on top of the machine's own defaults: the typed
/// settings, replaced by an env override of the same name (overrides are applied last).
pub fn effective_env(cfg: &PersistentConfig) -> BTreeMap<String, String> {
    let mut env: BTreeMap<String, String> = cfg
        .flash_next
        .env()
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    env.extend(
        cfg.env_overrides
            .iter()
            .map(|(k, v)| (k.clone(), v.clone())),
    );
    env
}

fn env_value(cfg: &PersistentConfig, key: &str) -> Option<u64> {
    effective_env(cfg)
        .get(key)
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
}

/// The cap on MLX memory (bytes) a launch carries: an override, else the SERVER card's value,
/// else the server's own 32 GiB; `None` when the caps are disabled.
fn memory_limit(cfg: &PersistentConfig) -> Option<u64> {
    let gib = cfg
        .env_overrides
        .get("LUMEN_MEMORY_LIMIT_GB")
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
        .or(if cfg.server.disable_wired_limit {
            None
        } else {
            Some(
                cfg.server
                    .memory_limit_gb
                    .map_or(MEMORY_LIMIT_DEFAULT_GIB, |g| g as u64),
            )
        });
    gib.map(|g| g << 30)
}

/// The `LUMEN_MAX_PROMPT_TOKENS` the app sends a Flash-Next server on a Mac with `ram_bytes`,
/// when the user has not set one: the longest prompt the server plans for by default.
pub fn default_prompt_tokens(cfg: &PersistentConfig, ram_bytes: u64) -> u64 {
    default_context(usable_bytes(ram_bytes, memory_limit(cfg)))
}

pub fn launch_plan(cfg: &PersistentConfig, ram_bytes: u64) -> LaunchPlan {
    let memory_limit = memory_limit(cfg);
    let prompt_cap = env_value(cfg, "LUMEN_MAX_PROMPT_TOKENS")
        .unwrap_or_else(|| default_context(usable_bytes(ram_bytes, memory_limit)));
    let max_ctx =
        env_value(cfg, "LUMEN_MAX_CTX").or(Some(cfg.context.max as u64).filter(|&c| c > 0));
    LaunchPlan {
        memory_limit,
        chunk: env_value(cfg, "LUMEN_QWEN35_PREFILL_CHUNK"),
        ctx_tokens: planned_context(Some(prompt_cap), max_ctx),
    }
}

// ── What speed that buys ─────────────────────────────────────────────

/// Decode speed (tokens/s) by expert-pool size (GB), measured on a 36 GB M3 Max whose SSD reads
/// at [`REFERENCE_READ_GBPS`], reading the shipped shards. Prefill (~230 tokens/s) does not
/// depend on the pool.
pub const DECODE_TPS_SHARDS: [(f64, f64); 4] =
    [(6.7, 5.75), (10.0, 7.53), (15.0, 9.73), (19.8, 13.0)];
/// The same with the expert pack. The 4.2 and 6.85 GB points were run on that Mac with its
/// memory limited to emulate a 16 and a 24 GB one, so they show what the plan does, not how a
/// real small Mac's own memory pressure behaves.
pub const DECODE_TPS_PACK: [(f64, f64); 5] = [
    (4.2, 5.6),
    (6.7, 6.83),
    (6.85, 6.7),
    (10.0, 8.22),
    (15.0, 10.86),
];
/// Random-read speed of the SSD behind those tables: four threads, `F_NOCACHE`, reads of 1 MB
/// and up (`lumen-mlx/src/offload/io.rs`).
pub const REFERENCE_READ_GBPS: f64 = 6.5;
/// Below this the SSD, not the GPU, sets a decode speed worth warning about.
pub const SLOW_READ_GBPS: f64 = 2.0;

/// Linear interpolation over `points` (sorted by x). Above the last point the value is held
/// (nothing is measured past it); below the first the first segment's slope continues, which
/// is an extrapolation.
fn interpolate(points: &[(f64, f64)], x: f64) -> f64 {
    let last = points[points.len() - 1];
    if x >= last.0 {
        return last.1;
    }
    let (lo, hi) = points
        .windows(2)
        .find(|w| x < w[1].0)
        .map_or((points[0], points[1]), |w| (w[0], w[1]));
    let slope = (hi.1 - lo.1) / (hi.0 - lo.0);
    (lo.1 + (x - lo.0) * slope).max(0.0)
}

/// Decode speed for a pool of `pool_gb`. With `pack` and a pool inside the pack's measured range
/// (4.2 to 15 GB) it follows the pack table; otherwise the shard table.
pub fn decode_tps_for_pool(pool_gb: f64, pack: bool) -> f64 {
    let (lo, hi) = (
        DECODE_TPS_PACK[0].0,
        DECODE_TPS_PACK[DECODE_TPS_PACK.len() - 1].0,
    );
    if pack && (lo..=hi).contains(&pool_gb) {
        interpolate(&DECODE_TPS_PACK, pool_gb)
    } else {
        interpolate(&DECODE_TPS_SHARDS, pool_gb)
    }
}

/// Decode speed for a pool of `pool_gb` on a disk that serves random reads at `read_gbps`:
/// the table scaled down by how much slower the disk is than the one it was measured on. A
/// faster disk is not promised more.
pub fn decode_tps_estimate(pool_gb: f64, read_gbps: f64, pack: bool) -> f64 {
    decode_tps_for_pool(pool_gb, pack) * (read_gbps / REFERENCE_READ_GBPS).clamp(0.0, 1.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadVerdict {
    Fast,
    Slow,
}

pub fn read_verdict(random_gbps: f64) -> ReadVerdict {
    if random_gbps < SLOW_READ_GBPS {
        ReadVerdict::Slow
    } else {
        ReadVerdict::Fast
    }
}

// ── The weights on disk ──────────────────────────────────────────────

/// Same rule as `family_of` in `lumen-server/src/catalog.rs`.
pub fn is_flash_next_id(id: &str) -> bool {
    let lower = id.to_lowercase();
    lower.contains("flash-next") || lower.contains("qwen4_exp") || lower.contains("qwen4-exp")
}

/// The weights directory of an installed Flash-Next, the active model first. `None` when no
/// complete copy is on disk.
pub fn find_model_dir(cfg: &PersistentConfig) -> Option<PathBuf> {
    let entries = models::scan_local(&cfg.models_dir, &Catalog::default()).ok()?;
    let mut found: Vec<_> = entries
        .iter()
        .filter(|e| e.ready && is_flash_next_id(&e.id))
        .collect();
    found.sort_by_key(|e| cfg.active_model.as_deref() != Some(e.id.as_str()));
    let path = &found.first()?.path;
    if path.join("config.json").is_file() {
        return Some(path.clone());
    }
    models::resolve_hf_snapshot(path)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackState {
    Absent,
    /// An interrupted copy: the server ignores it and reads the shards.
    Incomplete(u64),
    Complete(u64),
}

pub fn pack_state(dir: &Path) -> PackState {
    match std::fs::metadata(dir.join("experts.lpack")) {
        Ok(m) if m.len() >= PACK_COMPLETE_MIN_BYTES => PackState::Complete(m.len()),
        Ok(m) => PackState::Incomplete(m.len()),
        Err(_) => PackState::Absent,
    }
}

pub fn has_mtp_head(dir: &Path) -> bool {
    dir.join("mtp.safetensors").is_file()
}

/// The shard the probe reads: the largest `.safetensors` file in `dir`.
pub fn probe_shard(dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .filter(|p| p.file_name().is_some_and(|n| n != "mtp.safetensors"))
        .filter_map(|p| Some((std::fs::metadata(&p).ok()?.len(), p)))
        .max_by_key(|(len, _)| *len)
        .map(|(_, p)| p)
}

// ── Read-speed probe ─────────────────────────────────────────────────

/// Alignment of the read buffers: the Apple silicon page size, as the server's reader uses.
const ALIGN: usize = 16 * 1024;
/// Size of each read of the sequential phase.
const SEQ_CHUNK: usize = 8 << 20;
/// An expert, rounded up to whole pages: what one miss reads.
const RECORD_BYTES: usize = (EXPERT_BYTES as usize).div_ceil(ALIGN) * ALIGN;

/// What the probe reads and how long it may take.
#[derive(Debug, Clone, Copy)]
pub struct ProbePlan {
    /// Bytes read front to back, in `SEQ_CHUNK` reads on one thread.
    pub seq_bytes: u64,
    /// Bytes read as random `record_bytes` reads spread over `threads` threads.
    pub rand_bytes: u64,
    pub record_bytes: usize,
    pub threads: usize,
    /// The probe stops by then with whatever it has measured. The sequential phase gets at most
    /// two fifths of it.
    pub budget: Duration,
}

impl ProbePlan {
    /// ~1 GiB: half front to back, half the way the server fetches experts (four threads, one
    /// expert per read). A fast SSD finishes in well under a second.
    pub const DEFAULT: ProbePlan = ProbePlan {
        seq_bytes: 512 << 20,
        rand_bytes: 512 << 20,
        record_bytes: RECORD_BYTES,
        threads: 4,
        budget: Duration::from_secs(7),
    };
}

/// Why a probe stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeEnd {
    Finished,
    TimedOut,
    Cancelled,
}

#[derive(Debug, Clone, Copy)]
pub struct ProbeResult {
    pub seq_bytes: u64,
    pub seq_secs: f64,
    pub rand_bytes: u64,
    pub rand_secs: f64,
    pub rand_reads: u64,
    pub end: ProbeEnd,
}

fn gbps(bytes: u64, secs: f64) -> Option<f64> {
    (bytes > 0 && secs > 0.0).then(|| bytes as f64 / secs / 1e9)
}

impl ProbeResult {
    pub fn seq_gbps(&self) -> Option<f64> {
        gbps(self.seq_bytes, self.seq_secs)
    }
    pub fn rand_gbps(&self) -> Option<f64> {
        gbps(self.rand_bytes, self.rand_secs)
    }
}

/// Read `plan.seq_bytes` front to back, then `plan.rand_bytes` at random offsets of `path`, with
/// the page cache bypassed, and report GB/s for each. Checks `cancel` and the deadline between
/// reads, so it ends within one read of either.
pub fn probe_read_speed(
    path: &Path,
    plan: &ProbePlan,
    cancel: &AtomicBool,
) -> io::Result<ProbeResult> {
    let file = File::open(path)?;
    let size = file.metadata()?.len();
    if size == 0 {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "empty file"));
    }
    bypass_page_cache(&file);

    let seed = clock_seed();
    let started = Instant::now();
    let mut res = ProbeResult {
        seq_bytes: 0,
        seq_secs: 0.0,
        rand_bytes: 0,
        rand_secs: 0.0,
        rand_reads: 0,
        end: ProbeEnd::Finished,
    };

    let seq_stop = Stop {
        cancel,
        deadline: started + plan.budget.mul_f64(0.4),
    };
    let (bytes, secs, end) = read_sequential(&file, size, plan, &seq_stop, seed)?;
    res.seq_bytes = bytes;
    res.seq_secs = secs;
    // Running out of the sequential phase's share is not the end of the probe.
    if end == Some(ProbeEnd::Cancelled) {
        res.end = ProbeEnd::Cancelled;
        return Ok(res);
    }

    let stop = Stop {
        cancel,
        deadline: started + plan.budget,
    };
    let (bytes, reads, secs, end) = read_random(&file, size, plan, &stop, seed)?;
    res.rand_bytes = bytes;
    res.rand_reads = reads;
    res.rand_secs = secs;
    if let Some(e) = end {
        res.end = e;
    }
    Ok(res)
}

struct Stop<'a> {
    cancel: &'a AtomicBool,
    deadline: Instant,
}

impl Stop<'_> {
    fn reason(&self) -> Option<ProbeEnd> {
        if self.cancel.load(Ordering::Relaxed) {
            Some(ProbeEnd::Cancelled)
        } else if Instant::now() >= self.deadline {
            Some(ProbeEnd::TimedOut)
        } else {
            None
        }
    }
}

fn read_sequential(
    file: &File,
    size: u64,
    plan: &ProbePlan,
    stop: &Stop,
    seed: u64,
) -> io::Result<(u64, f64, Option<ProbeEnd>)> {
    let want = plan.seq_bytes.min(size);
    let slack = size - want;
    let mut off = if slack == 0 {
        0
    } else {
        (seed % slack) & !(ALIGN as u64 - 1)
    };
    let mut buf = AlignedBuf::new(SEQ_CHUNK);
    let (mut done, mut end) = (0u64, None);
    let t0 = Instant::now();
    while done < want {
        end = stop.reason();
        if end.is_some() {
            break;
        }
        let take = SEQ_CHUNK.min((want - done) as usize);
        let n = read_full_at(file, &mut buf.bytes()[..take], off)?;
        done += n as u64;
        off += n as u64;
        if n < take {
            break;
        }
    }
    Ok((done, t0.elapsed().as_secs_f64(), end))
}

fn read_random(
    file: &File,
    size: u64,
    plan: &ProbePlan,
    stop: &Stop,
    seed: u64,
) -> io::Result<(u64, u64, f64, Option<ProbeEnd>)> {
    let rec = plan.record_bytes as u64;
    if size < rec {
        return Ok((0, 0, 0.0, None));
    }
    // Every 16 KiB-aligned start that leaves room for a whole record.
    let starts = (size - rec) / ALIGN as u64 + 1;
    let claimed = AtomicU64::new(0);
    let (bytes, reads) = (AtomicU64::new(0), AtomicU64::new(0));
    let t0 = Instant::now();
    let ends: Vec<io::Result<Option<ProbeEnd>>> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..plan.threads.max(1) as u64)
            .map(|i| {
                let (claimed, bytes, reads) = (&claimed, &bytes, &reads);
                s.spawn(move || -> io::Result<Option<ProbeEnd>> {
                    let mut rng = XorShift::new(seed ^ (i + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    let mut buf = AlignedBuf::new(plan.record_bytes);
                    loop {
                        if let Some(e) = stop.reason() {
                            return Ok(Some(e));
                        }
                        if claimed.fetch_add(rec, Ordering::Relaxed) >= plan.rand_bytes {
                            return Ok(None);
                        }
                        let off = (rng.next_u64() % starts) * ALIGN as u64;
                        let n = read_full_at(file, buf.bytes(), off)?;
                        bytes.fetch_add(n as u64, Ordering::Relaxed);
                        reads.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|w| {
                w.join()
                    .unwrap_or_else(|_| Err(io::Error::other("probe thread panicked")))
            })
            .collect()
    });
    let secs = t0.elapsed().as_secs_f64();

    let mut end = None;
    for e in ends {
        match e? {
            Some(ProbeEnd::Cancelled) => end = Some(ProbeEnd::Cancelled),
            Some(other) if end.is_none() => end = Some(other),
            _ => {}
        }
    }
    Ok((
        bytes.load(Ordering::Relaxed),
        reads.load(Ordering::Relaxed),
        secs,
        end,
    ))
}

/// Fill `buf` from `off`, looping over short reads. Returns fewer bytes than `buf.len()` only
/// at end of file.
fn read_full_at(file: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
    let mut done = 0;
    while done < buf.len() {
        match file.read_at(&mut buf[done..], off + done as u64) {
            Ok(0) => break,
            Ok(n) => done += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(done)
}

/// `F_NOCACHE` and no read-ahead, the way the server's direct mode reads: the bytes come from
/// the disk, not from pages an earlier read left in memory.
#[cfg(target_os = "macos")]
fn bypass_page_cache(file: &File) {
    use std::os::unix::io::AsRawFd;
    // SAFETY: plain fcntl calls on an fd we own.
    unsafe {
        libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1);
        libc::fcntl(file.as_raw_fd(), libc::F_RDAHEAD, 0);
    }
}

#[cfg(not(target_os = "macos"))]
fn bypass_page_cache(_file: &File) {}

/// A read buffer whose start is 16 KiB-aligned.
struct AlignedBuf {
    raw: Vec<u8>,
    start: usize,
    len: usize,
}

impl AlignedBuf {
    fn new(len: usize) -> Self {
        let raw = vec![0u8; len + ALIGN];
        let start = raw.as_ptr().align_offset(ALIGN);
        Self { raw, start, len }
    }

    fn bytes(&mut self) -> &mut [u8] {
        &mut self.raw[self.start..self.start + self.len]
    }
}

struct XorShift(u64);

impl XorShift {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Different offsets on every probe, so a repeat does not read what the last one cached.
fn clock_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0x5EED, |d| d.as_nanos() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    const GIB: u64 = 1 << 30;

    fn temp_file(name: &str, len: usize) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("lumen-flash-next-{name}-{}", std::process::id()));
        // A counter pattern, so a read at the wrong offset is visible.
        let data: Vec<u8> = (0..len).map(|i| (i / 4096 % 251) as u8).collect();
        File::create(&path).unwrap().write_all(&data).unwrap();
        path
    }

    fn small_plan() -> ProbePlan {
        ProbePlan {
            seq_bytes: 2 << 20,
            rand_bytes: 4 << 20,
            record_bytes: 3 * ALIGN,
            threads: 3,
            budget: Duration::from_secs(5),
        }
    }

    #[test]
    fn the_record_is_one_expert_rounded_up_to_whole_pages() {
        assert_eq!(RECORD_BYTES % ALIGN, 0);
        assert!(RECORD_BYTES as u64 >= EXPERT_BYTES);
        assert!((RECORD_BYTES as u64) < EXPERT_BYTES + ALIGN as u64);
    }

    #[test]
    fn the_buffer_start_is_page_aligned() {
        for len in [1, ALIGN, RECORD_BYTES] {
            let mut buf = AlignedBuf::new(len);
            assert_eq!(buf.bytes().len(), len);
            assert_eq!(buf.bytes().as_ptr() as usize % ALIGN, 0);
        }
    }

    #[test]
    fn a_full_read_returns_the_bytes_at_the_offset() {
        let path = temp_file("full-read", 3 * ALIGN + 100);
        let file = File::open(&path).unwrap();
        let mut buf = AlignedBuf::new(ALIGN);
        assert_eq!(
            read_full_at(&file, buf.bytes(), 2 * ALIGN as u64).unwrap(),
            ALIGN
        );
        let expected = (2 * ALIGN / 4096 % 251) as u8;
        assert_eq!(buf.bytes()[0], expected);
        // Past the end the read is short, never padded.
        assert_eq!(
            read_full_at(&file, buf.bytes(), 3 * ALIGN as u64).unwrap(),
            100
        );
        assert_eq!(
            read_full_at(&file, buf.bytes(), 4 * ALIGN as u64).unwrap(),
            0
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn the_probe_reads_what_it_was_asked_to_and_reports_both_speeds() {
        let path = temp_file("probe", 8 << 20);
        let r = probe_read_speed(&path, &small_plan(), &AtomicBool::new(false)).unwrap();
        assert_eq!(r.end, ProbeEnd::Finished);
        assert_eq!(r.seq_bytes, 2 << 20);
        // Reads are claimed a record at a time, so the total lands on a record boundary
        // at or past the target.
        let rec = (3 * ALIGN) as u64;
        assert!(r.rand_bytes >= 4 << 20, "{}", r.rand_bytes);
        assert_eq!(r.rand_bytes, r.rand_reads * rec);
        assert!(r.seq_gbps().unwrap() > 0.0);
        assert!(r.rand_gbps().unwrap() > 0.0);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn a_file_smaller_than_the_plan_is_read_to_its_end() {
        let path = temp_file("tiny", 5 * ALIGN);
        let r = probe_read_speed(&path, &small_plan(), &AtomicBool::new(false)).unwrap();
        assert_eq!(r.seq_bytes, (5 * ALIGN) as u64);
        // Three pages of record in five pages of file: room for a few start offsets.
        assert!(r.rand_reads > 0);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn a_file_smaller_than_one_record_has_no_random_phase() {
        let path = temp_file("sub-record", ALIGN);
        let r = probe_read_speed(&path, &small_plan(), &AtomicBool::new(false)).unwrap();
        assert_eq!(r.seq_bytes, ALIGN as u64);
        assert_eq!(r.rand_bytes, 0);
        assert!(r.rand_gbps().is_none());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn a_cancelled_probe_stops_without_reading() {
        let path = temp_file("cancel", 8 << 20);
        let r = probe_read_speed(&path, &small_plan(), &AtomicBool::new(true)).unwrap();
        assert_eq!(r.end, ProbeEnd::Cancelled);
        assert_eq!((r.seq_bytes, r.rand_bytes), (0, 0));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn an_expired_budget_ends_the_probe_with_what_it_has() {
        let path = temp_file("budget", 8 << 20);
        let plan = ProbePlan {
            budget: Duration::ZERO,
            ..small_plan()
        };
        let r = probe_read_speed(&path, &plan, &AtomicBool::new(false)).unwrap();
        assert_eq!(r.end, ProbeEnd::TimedOut);
        assert_eq!((r.seq_bytes, r.rand_bytes), (0, 0));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn an_empty_or_missing_file_is_an_error_not_a_speed() {
        let path = temp_file("empty", 0);
        let none = AtomicBool::new(false);
        assert!(probe_read_speed(&path, &small_plan(), &none).is_err());
        std::fs::remove_file(&path).ok();
        assert!(probe_read_speed(&path, &small_plan(), &none).is_err());
    }

    #[test]
    fn the_default_plan_stays_inside_the_ten_second_limit() {
        let p = ProbePlan::DEFAULT;
        assert!(p.budget < Duration::from_secs(10));
        assert_eq!(p.seq_bytes + p.rand_bytes, 1 << 30);
        assert_eq!(p.threads, 4);
    }

    #[test]
    fn the_tables_are_reproduced_at_their_measured_points() {
        for (pool, tps) in DECODE_TPS_SHARDS {
            assert!(
                (decode_tps_for_pool(pool, false) - tps).abs() < 1e-9,
                "{pool}"
            );
        }
        for (pool, tps) in DECODE_TPS_PACK {
            assert!(
                (decode_tps_for_pool(pool, true) - tps).abs() < 1e-9,
                "{pool}"
            );
        }
    }

    #[test]
    fn between_points_the_speed_is_a_straight_line() {
        // Halfway between 10 GB (7.53) and 15 GB (9.73), and (8.22) and (10.86) with the pack.
        assert!((decode_tps_for_pool(12.5, false) - 8.63).abs() < 1e-9);
        assert!((decode_tps_for_pool(12.5, true) - 9.54).abs() < 1e-9);
    }

    #[test]
    fn past_the_last_point_the_speed_is_held_not_extrapolated() {
        assert_eq!(decode_tps_for_pool(60.0, false), 13.0);
        // The pack table ends at 15 GB; past it the shard table answers.
        assert_eq!(decode_tps_for_pool(19.8, true), 13.0);
    }

    #[test]
    fn below_the_first_point_the_first_slope_is_continued_and_never_negative() {
        let below = decode_tps_for_pool(5.0, false);
        let slope = (7.53 - 5.75) / (10.0 - 6.7);
        assert!((below - (5.75 - 1.7 * slope)).abs() < 1e-9);
        assert_eq!(decode_tps_for_pool(-50.0, false), 0.0);
        // The pack table reaches down to 4.2 GB; under it the shard table is extrapolated.
        assert!(decode_tps_for_pool(3.0, true) < 5.6);
    }

    #[test]
    fn a_slower_disk_scales_the_table_down_and_a_faster_one_does_not_scale_it_up() {
        let at_ref = decode_tps_estimate(10.0, REFERENCE_READ_GBPS, false);
        assert!((at_ref - 7.53).abs() < 1e-9);
        let half = decode_tps_estimate(10.0, REFERENCE_READ_GBPS / 2.0, false);
        assert!((half - 7.53 / 2.0).abs() < 1e-9);
        assert_eq!(decode_tps_estimate(10.0, 25.0, false), at_ref);
        assert_eq!(decode_tps_estimate(10.0, 0.0, false), 0.0);
        assert!((decode_tps_estimate(10.0, 6.5, true) - 8.22).abs() < 1e-9);
    }

    #[test]
    fn two_gb_per_second_is_the_line_between_slow_and_fast() {
        assert_eq!(read_verdict(1.99), ReadVerdict::Slow);
        assert_eq!(read_verdict(SLOW_READ_GBPS), ReadVerdict::Fast);
        assert_eq!(read_verdict(6.5), ReadVerdict::Fast);
        assert_eq!(read_verdict(0.0), ReadVerdict::Slow);
    }

    #[test]
    fn the_pool_sizes_are_the_slot_counts_of_the_servers_pool() {
        assert_eq!(POOL_FLOOR_BYTES, 2 * 512 * EXPERT_BYTES);
        assert_eq!(POOL_PREFETCH_BYTES, 3 * 512 * EXPERT_BYTES);
        // 13 layers (12 full-attention and the draft head's) x (K,V + the indexer key), f32.
        assert_eq!(KV_BYTES_PER_TOKEN, 13 * (2 * 2 * 256 * 4 + 128 * 4));
    }

    #[test]
    fn usable_memory_is_two_thirds_up_to_32_gib_and_three_quarters_above() {
        let at = |gib: u64| usable_bytes(gib * GIB, None);
        assert_eq!(at(16), (16.0 * GIB as f64 * 2.0 / 3.0 * 0.88) as u64);
        assert_eq!(at(32), (32.0 * GIB as f64 * 2.0 / 3.0 * 0.88) as u64);
        assert_eq!(at(36), (36.0 * GIB as f64 * 0.75 * 0.88) as u64);
        // A memory limit only ever lowers it.
        assert_eq!(usable_bytes(36 * GIB, Some(9 * GIB)), 9 * GIB);
        assert_eq!(usable_bytes(16 * GIB, Some(64 * GIB)), at(16));
    }

    #[test]
    fn the_reserve_adds_buffers_chunk_kv_and_half_the_scores() {
        let usable = usable_bytes(36 * GIB, None);
        let p = plan_memory(usable, None, None);
        assert_eq!((p.chunk, p.ctx_tokens), (1024, 16_384));
        let scores = 24 * 1024 * 16_384 * 4; // 1.61 GB, under the 2 GB budget
        assert_eq!(
            p.reserve,
            600_000_000 + 1024 * 1_100_000 + 16_384 * 59_904 + scores / 2
        );
        assert_eq!(p.pool, usable - TRUNK_BYTES - p.reserve);
        assert!(p.fits());
        // Past the budget the scores stop growing.
        let long = plan_memory(usable, None, Some(65_536));
        assert_eq!(
            long.reserve,
            600_000_000 + 1024 * 1_100_000 + 65_536 * 59_904 + 2_000_000_000 / 2
        );
    }

    #[test]
    fn a_tight_machine_takes_the_smaller_chunk_context_and_scores_budget() {
        let p = plan_memory(usable_bytes(16 * GIB, None), None, None);
        assert_eq!((p.chunk, p.ctx_tokens), (512, 8192));
        assert_eq!(p.reserve, 1_855_260_160);
        assert_eq!(p.pool, 4_623_596_427);
        // 24 GB is not tight: 1,024-token chunks, 16K prompts, a ~8 GB pool.
        let p = plan_memory(usable_bytes(24 * GIB, None), None, None);
        assert_eq!((p.chunk, p.ctx_tokens), (1024, 16_384));
        assert_eq!(p.pool, 8_005_111_377);
    }

    #[test]
    fn an_override_replaces_the_default_it_names_and_nothing_else() {
        let usable = usable_bytes(16 * GIB, None);
        let p = plan_memory(usable, Some(2048), None);
        assert_eq!((p.chunk, p.ctx_tokens), (2048, 8192));
        let p = plan_memory(usable, None, Some(32_768));
        assert_eq!((p.chunk, p.ctx_tokens), (512, 32_768));
    }

    #[test]
    fn the_context_a_user_picks_comes_out_of_the_pool() {
        let usable = usable_bytes(36 * GIB, None);
        let small = plan_memory(usable, None, Some(16_384));
        let big = plan_memory(usable, None, Some(81_920));
        assert!(big.pool < small.pool);
        assert_eq!(big.pool, 14_278_370_058);
        // On a 16 GB Mac the app's default 81,920-token Max context leaves no room at all.
        let tight = plan_memory(usable_bytes(16 * GIB, None), None, Some(81_920));
        assert!(!tight.fits());
        assert_eq!(tight.pool_bytes(), POOL_FLOOR_BYTES);
    }

    #[test]
    fn the_longest_prompt_that_fits_is_found_by_search() {
        let usable = usable_bytes(16 * GIB, None);
        assert_eq!(max_fitting_context(usable, None), 34_879);
        let n = max_fitting_context(usable, None);
        assert!(plan_memory(usable, None, Some(n)).fits());
        assert!(!plan_memory(usable, None, Some(n + 1)).fits());
        assert_eq!(
            max_fitting_context(usable_bytes(36 * GIB, None), None),
            MAX_CONTEXT
        );
        assert_eq!(max_fitting_context(usable_bytes(8 * GIB, None), None), 0);
    }

    #[test]
    fn a_14_gb_mac_is_the_smallest_that_runs_the_model() {
        assert!(!fits(13 * GIB));
        assert!(fits(14 * GIB));
        assert!(fits(16 * GIB));
        assert!(!fits(8 * GIB));
        assert_eq!(min_ram_gib(), 14);
    }

    #[test]
    fn the_prompt_cap_sets_the_planned_context_and_max_ctx_only_lowers_it() {
        assert_eq!(planned_context(Some(16_384), Some(81_920)), Some(16_384));
        assert_eq!(planned_context(Some(100_000), Some(81_920)), Some(81_920));
        assert_eq!(planned_context(Some(5_000), None), Some(5_000));
        // LUMEN_MAX_CTX alone cannot raise it past the default.
        assert_eq!(planned_context(None, Some(81_920)), Some(DEFAULT_CONTEXT));
        assert_eq!(planned_context(None, Some(4_096)), Some(4_096));
        assert_eq!(planned_context(None, None), None);
    }

    fn app_config() -> PersistentConfig {
        let mut cfg = PersistentConfig::default();
        cfg.context.max = 81_920;
        cfg.context.prefill = 40_960;
        cfg.server.memory_limit_gb = Some(64);
        cfg.server.disable_wired_limit = false;
        cfg
    }

    /// The app sends a Flash-Next server `LUMEN_MAX_PROMPT_TOKENS` = the machine's default, so
    /// the Prefill card (40,960, the other families' setting) and the Max context card (81,920)
    /// do not decide what the plan is held to.
    #[test]
    fn a_default_launch_plans_the_machines_default_prompt_not_the_prefill_cards() {
        let cfg = app_config();
        assert_eq!(default_prompt_tokens(&cfg, 36 * GIB), 16_384);
        assert_eq!(default_prompt_tokens(&cfg, 16 * GIB), 8_192);
        assert_eq!(launch_plan(&cfg, 36 * GIB).ctx_tokens, Some(16_384));
        assert_eq!(launch_plan(&cfg, 16 * GIB).ctx_tokens, Some(8_192));
        assert_eq!(launch_plan(&cfg, 36 * GIB).chunk, None);
    }

    #[test]
    fn the_machines_default_follows_the_memory_limit_it_is_tight_under() {
        let mut cfg = app_config();
        // A big Mac held to 8 GiB of MLX memory is a tight machine to the server.
        cfg.server.memory_limit_gb = Some(8);
        assert_eq!(default_prompt_tokens(&cfg, 36 * GIB), 8_192);
        assert_eq!(launch_plan(&cfg, 36 * GIB).memory_limit, Some(8 << 30));
    }

    #[test]
    fn the_defaults_are_the_ones_the_plan_uses() {
        for gib in [8, 16, 24, 36, 64] {
            let usable = usable_bytes(gib * GIB, None);
            let p = plan_memory(usable, None, None);
            assert_eq!(p.chunk, default_chunk(usable), "{gib}");
            assert_eq!(p.ctx_tokens, default_context(usable), "{gib}");
        }
    }

    #[test]
    fn the_overrides_the_server_reads_win_over_the_default_and_the_cards() {
        let mut cfg = app_config();
        // An override of the clear name beats the machine's default ...
        cfg.env_overrides
            .insert("LUMEN_MAX_PROMPT_TOKENS".into(), "32768".into());
        assert_eq!(launch_plan(&cfg, 36 * GIB).ctx_tokens, Some(32_768));
        // ... a smaller Max context lowers it; a larger one does not raise it.
        cfg.env_overrides
            .insert("LUMEN_MAX_CTX".into(), "8000".into());
        assert_eq!(launch_plan(&cfg, 36 * GIB).ctx_tokens, Some(8_000));
        cfg.env_overrides
            .insert("LUMEN_MAX_CTX".into(), "200000".into());
        assert_eq!(launch_plan(&cfg, 36 * GIB).ctx_tokens, Some(32_768));
        // The Max context card alone also only lowers the machine's default.
        cfg.env_overrides.clear();
        cfg.context.max = 4_096;
        assert_eq!(launch_plan(&cfg, 36 * GIB).ctx_tokens, Some(4_096));
        cfg.context.max = 81_920;
        cfg.env_overrides
            .insert("LUMEN_QWEN35_PREFILL_CHUNK".into(), "512".into());
        cfg.env_overrides
            .insert("LUMEN_MEMORY_LIMIT_GB".into(), "9".into());
        let plan = launch_plan(&cfg, 36 * GIB);
        assert_eq!((plan.chunk, plan.memory_limit), (Some(512), Some(9 << 30)));
        // A garbled value is ignored, as the server's parser would: the 9 GiB limit makes
        // this a tight machine, so the default is 8,192.
        cfg.env_overrides
            .insert("LUMEN_MAX_PROMPT_TOKENS".into(), "lots".into());
        assert_eq!(launch_plan(&cfg, 36 * GIB).ctx_tokens, Some(8_192));
    }

    /// The settings card's typed fields plan the launch exactly as the overrides above do, and
    /// an override of the same name still wins (it is applied last, as the escape hatch).
    #[test]
    fn the_typed_settings_plan_the_launch_and_an_override_beats_them() {
        let mut cfg = app_config();
        cfg.flash_next.max_prompt_tokens = Some(32_768);
        cfg.flash_next.prefill_chunk = Some(512);
        let plan = launch_plan(&cfg, 36 * GIB);
        assert_eq!((plan.ctx_tokens, plan.chunk), (Some(32_768), Some(512)));
        // Unset again: the machine's default.
        cfg.flash_next.max_prompt_tokens = None;
        cfg.flash_next.prefill_chunk = None;
        let plan = launch_plan(&cfg, 36 * GIB);
        assert_eq!((plan.ctx_tokens, plan.chunk), (Some(16_384), None));
        // The Max context card still only lowers a typed prompt cap.
        cfg.flash_next.max_prompt_tokens = Some(65_536);
        cfg.context.max = 40_000;
        assert_eq!(launch_plan(&cfg, 36 * GIB).ctx_tokens, Some(40_000));
        cfg.context.max = 81_920;
        cfg.env_overrides
            .insert("LUMEN_MAX_PROMPT_TOKENS".into(), "8192".into());
        assert_eq!(launch_plan(&cfg, 36 * GIB).ctx_tokens, Some(8_192));
    }

    #[test]
    fn the_effective_environment_is_the_typed_settings_with_the_overrides_on_top() {
        let mut cfg = app_config();
        assert!(effective_env(&cfg).is_empty());
        cfg.flash_next.pack = crate::config::FlashNextPack::Off;
        cfg.flash_next.draft_head = crate::config::FlashNextDraftHead::On;
        cfg.env_overrides
            .insert("LUMEN_QWEN4_MTP".into(), "off".into());
        let env = effective_env(&cfg);
        assert_eq!(env.get("LUMEN_MOE_PACK").map(String::as_str), Some("off"));
        assert_eq!(env.get("LUMEN_QWEN4_MTP").map(String::as_str), Some("off"));
    }

    #[test]
    fn without_a_memory_cap_in_the_config_the_servers_default_of_32_gib_applies() {
        let mut cfg = PersistentConfig::default();
        cfg.server.memory_limit_gb = None;
        cfg.server.disable_wired_limit = false;
        assert_eq!(launch_plan(&cfg, 36 * GIB).memory_limit, Some(32 << 30));
        // "Disable caps" leaves MLX's own limit, which the plan does not lower itself to.
        cfg.server.disable_wired_limit = true;
        assert_eq!(launch_plan(&cfg, 36 * GIB).memory_limit, None);
    }

    #[test]
    fn the_family_is_recognised_by_the_servers_rule() {
        assert!(is_flash_next_id("pipenetwork/Qwen3.8-Flash-Next-MLX-4bit"));
        assert!(is_flash_next_id("pipenetwork--Qwen3.8-Flash-Next-MLX-4bit"));
        assert!(is_flash_next_id("qwen4_exp-local"));
        assert!(!is_flash_next_id("mlx-community/Qwen3.6-35B-A3B-mxfp4"));
    }

    #[test]
    fn the_pack_is_complete_only_above_sixty_gb() {
        let dir =
            std::env::temp_dir().join(format!("lumen-flash-next-pack-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(pack_state(&dir), PackState::Absent);
        // Sparse files: the size is real, the disk use is not.
        let pack = File::create(dir.join("experts.lpack")).unwrap();
        pack.set_len(PACK_COMPLETE_MIN_BYTES - 1).unwrap();
        assert_eq!(
            pack_state(&dir),
            PackState::Incomplete(PACK_COMPLETE_MIN_BYTES - 1)
        );
        pack.set_len(PACK_BYTES).unwrap();
        assert_eq!(pack_state(&dir), PackState::Complete(PACK_BYTES));
        assert!(!has_mtp_head(&dir));
        File::create(dir.join("mtp.safetensors")).unwrap();
        assert!(has_mtp_head(&dir));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn the_probe_reads_a_trunk_shard_not_the_draft_head() {
        let dir =
            std::env::temp_dir().join(format!("lumen-flash-next-shard-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model-00001.safetensors"), vec![0u8; 1000]).unwrap();
        std::fs::write(dir.join("model-00002.safetensors"), vec![0u8; 3000]).unwrap();
        std::fs::write(dir.join("mtp.safetensors"), vec![0u8; 9000]).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        assert_eq!(
            probe_shard(&dir).unwrap().file_name().unwrap(),
            "model-00002.safetensors"
        );
        std::fs::remove_dir_all(dir).ok();
    }
}
