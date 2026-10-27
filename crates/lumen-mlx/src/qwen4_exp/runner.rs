//! Qwen3.8-Flash-Next as one of the Qwen backend's runners: sequences, prefill
//! in chunks, greedy and sampled decode, snapshots and rollback points.
//!
//! Everything the chat layer does with a runner (templates, tool calls, sampling,
//! sessions, the prefix cache) is the Qwen family's own and comes with the
//! `Runner` impl; this file only has to say how a sequence advances and how a
//! state is copied. A sequence's state is a [`Qwen4ExpCache`], a bag of
//! immutable MLX array handles plus the KV buffers, so a snapshot or a rollback
//! point is a clone that shares storage until either side appends.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use mlx_rs::Array;

use super::config::Qwen4ExpConfig;
use super::model::{LoadOptions, MtpMode, NgramSource, Qwen4ExpCache, Qwen4ExpModel};
use crate::host_memory::HostMemory;
use crate::kv_disk::{DiskKvStore, KvManifest};
use crate::session_feed::PrefillGrid;
use crate::{ProbeRows, Runner};

/// What the backend needs to know after a load.
pub(crate) struct LoadInfo {
    pub eos_tokens: Vec<u32>,
    pub vocab_size: usize,
}

/// Rounds a sequence looks back over when deciding whether drafting pays.
const SPEC_WINDOW: usize = 32;

/// Calls of each kind the governor averages over.
const GOVERNOR_WINDOW: usize = 48;
/// Calls between looks at the mode that is not in use.
const PROBE_EVERY: u32 = 160;
/// Calls spent in the other mode at each look.
const PROBE_LEN: u32 = 16;
/// Speculation must beat plain steps by this factor (per token) to be switched on...
const GOVERNOR_ON_BELOW: f64 = 0.95;
/// ...and is switched off once it is slower than plain steps by this factor. The gap keeps a
/// measurement that hovers around break-even from flipping the mode at every look.
const GOVERNOR_OFF_ABOVE: f64 = 1.02;

/// Picks between speculating and plain steps from what each has actually cost per token on this
/// machine, with this pool and this text: a round's price is mostly the extra experts the second
/// row needs, which no static rule predicts. It samples the mode not in use for a few calls now
/// and then, so a change of conditions (a fuller cache, a different kind of text) is noticed.
struct Governor {
    use_spec: bool,
    spec_ms: std::collections::VecDeque<f64>,
    plain_ms: std::collections::VecDeque<f64>,
    since_probe: u32,
    probe_left: u32,
}

impl Governor {
    fn new() -> Self {
        Self {
            use_spec: true,
            spec_ms: Default::default(),
            plain_ms: Default::default(),
            since_probe: 0,
            probe_left: 0,
        }
    }

    /// Should the next call speculate?
    fn should_speculate(&mut self) -> bool {
        // Until both modes have a few samples, alternate to get them.
        if self.spec_ms.len() < 8 {
            return true;
        }
        if self.plain_ms.len() < 8 {
            return false;
        }
        if self.probe_left > 0 {
            self.probe_left -= 1;
            return !self.use_spec;
        }
        self.since_probe += 1;
        if self.since_probe >= PROBE_EVERY {
            self.since_probe = 0;
            self.probe_left = PROBE_LEN;
        }
        self.use_spec
    }

    /// Record a call that took `ms` and produced one token.
    fn note(&mut self, speculated: bool, ms: f64) {
        let window = if speculated {
            &mut self.spec_ms
        } else {
            &mut self.plain_ms
        };
        window.push_back(ms);
        if window.len() > GOVERNOR_WINDOW {
            window.pop_front();
        }
        if self.spec_ms.len() >= 8 && self.plain_ms.len() >= 8 && self.probe_left == 0 {
            let mean = |w: &std::collections::VecDeque<f64>| w.iter().sum::<f64>() / w.len() as f64;
            let ratio = mean(&self.spec_ms) / mean(&self.plain_ms);
            let next = if self.use_spec {
                ratio <= GOVERNOR_OFF_ABOVE
            } else {
                ratio < GOVERNOR_ON_BELOW
            };
            if next != self.use_spec {
                self.use_spec = next;
                eprintln!(
                    "[qwen4_exp] speculation {}: {:.0} ms/token with drafts, {:.0} ms/token plain",
                    if next { "on" } else { "off" },
                    mean(&self.spec_ms),
                    mean(&self.plain_ms)
                );
            }
        }
    }
}

/// The draft head's runtime policy and counters.
struct Speculation {
    /// A draft head is loaded and speculation is on.
    enabled: bool,
    /// Share of the last [`SPEC_WINDOW`] drafts that must be accepted for a sequence to keep
    /// speculating.
    floor: f32,
    governor: Governor,
    /// Let the governor choose between speculating and plain steps (off in tests that pin the
    /// rounds).
    adaptive: bool,
    rounds: u64,
    accepted: u64,
    draft: std::time::Duration,
    verify: std::time::Duration,
}

impl Speculation {
    fn off() -> Self {
        Self {
            enabled: false,
            floor: 0.0,
            governor: Governor::new(),
            adaptive: false,
            rounds: 0,
            accepted: 0,
            draft: Default::default(),
            verify: Default::default(),
        }
    }

    fn note_round(
        &mut self,
        accepted: bool,
        draft: std::time::Duration,
        verify: std::time::Duration,
    ) {
        self.rounds += 1;
        self.accepted += u64::from(accepted);
        self.draft += draft;
        self.verify += verify;
        if self.rounds.is_multiple_of(512) {
            eprintln!("[qwen4_exp] {}", self.summary());
        }
    }

    fn summary(&self) -> String {
        let n = self.rounds.max(1) as f64;
        format!(
            "speculation: {} rounds, {:.1}% of drafts accepted, draft {:.1} ms + verify {:.1} ms per round",
            self.rounds,
            100.0 * self.accepted as f64 / n,
            self.draft.as_secs_f64() * 1e3 / n,
            self.verify.as_secs_f64() * 1e3 / n,
        )
    }
}

/// A token a speculative round has already verified: the next call, passing `after`, gets
/// `next` and the state with `after` processed.
struct Pending {
    after: u32,
    next: u32,
    state: Qwen4ExpCache,
}

struct Seq {
    cache: Qwen4ExpCache,
    position: usize,
    /// Set by an accepted draft; see [`Pending`]. Anything that touches the sequence drops it.
    pending: Option<Pending>,
    /// Rounds drafted and whether each draft was accepted, newest last.
    recent: std::collections::VecDeque<bool>,
    /// Speculation gave up on this sequence (drafts were rejected too often to pay).
    spec_off: bool,
    rollback: Option<(Qwen4ExpCache, usize)>,
    /// The disk entry this sequence last wrote or was restored from, so a longer
    /// state replaces it instead of piling up beside it.
    persisted_key: Option<String>,
    /// The snapshot this sequence was forked from, while it still is that snapshot (anything
    /// that may change it clears this): what a joint feed computed ahead is keyed on.
    forked_from: Option<u64>,
}

/// A feed [`Qwen4ExpRunner::prefeed`] computed ahead: what extending a sequence forked from
/// `snapshot` by `tokens` (keeping the state `mark` tokens in) gives.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PrefedKey {
    snapshot: u64,
    mark: Option<usize>,
    tokens: Vec<u32>,
}

struct Prefed {
    cache: Qwen4ExpCache,
    next: u32,
    point: Option<Qwen4ExpCache>,
}

/// The most tokens one joint feed takes, all its sequences together: one prefill group, whose
/// residual streams the memory plan holds.
const JOINT_FEED_TOKENS: usize = super::model::PREFILL_GROUP_TOKENS;

pub(crate) struct Qwen4ExpRunner {
    model: Qwen4ExpModel,
    seqs: HashMap<u64, Seq>,
    snapshots: HashMap<u64, (Qwen4ExpCache, usize)>,
    next_snapshot: u64,
    chunk: usize,
    /// Longest prompt the memory is planned for (`LUMEN_MAX_PROMPT_TOKENS`).
    ctx_tokens: usize,
    /// Bytes MLX may hold (live buffers plus its buffer cache) before it stops recycling the
    /// cache: the GPU memory the plan was made for.
    memory_budget: u64,
    /// L2 tier (`LUMEN_KV_DISK`): session states at prefill-grid points, so a
    /// restarted server resumes a conversation without reading the model again.
    disk: Option<DiskKvStore>,
    fingerprint: String,
    spec: Speculation,
    /// Feeds computed ahead by [`Self::prefeed`], until the extend they belong to takes them.
    prefed: HashMap<PrefedKey, Prefed>,
    /// Tests choose the draft of each round.
    #[cfg(test)]
    forced_draft: Option<Box<dyn FnMut() -> u32>>,
}

/// Tokens per prefill chunk. The chunks of a prompt go through the model layer by layer, so
/// the SSD reads each layer's experts once for the whole prompt whatever the chunk size
/// (measured on an 18K-token prompt, 6.7 GB pool: 234 / 232-241 / 227 tok/s at chunks of 512 /
/// 2,048 / 4,096, where feeding chunk by chunk gave 73 / 153 / 145). The chunk only sets the
/// activation memory (and how close to a conversation's end a persisted state can sit), so it
/// is small: a chunk's memory is what the pool gives up. A Mac that can hardly hold the model
/// takes half of it.
const PREFILL_CHUNK: usize = 1024;
const TIGHT_PREFILL_CHUNK: usize = 512;

lumen_flags::flag! {
    /// Log MLX's memory (active, cache, peak) and what each sequence holds: after every prefill,
    /// when a sequence is dropped, and every 32 decode steps with the speed of those steps. For
    /// finding out where a memory limit is being hit; it adds one allocator query per line.
    /// After each prefill it also prints the expert bank's account of that feed alone (layer
    /// calls, reads, layer calls that took the staging path), restarting the bank's counters.
    pub(crate) mem_log {
        env: "LUMEN_QWEN4_MEM_LOG",
        default: false,
        kind: Diagnostic,
    }
}

/// Decode steps between two `mem_log` lines.
const MEM_LOG_EVERY: u64 = 32;

/// Decode steps counted since the last `mem_log` line, and when that line was written.
static MEM_LOG_DECODE: std::sync::Mutex<(u64, Option<std::time::Instant>)> =
    std::sync::Mutex::new((0, None));

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

/// Longest prompt the memory is planned for. The server rejects longer ones (it reads the same
/// `LUMEN_MAX_PROMPT_TOKENS`), and raising it takes the KV cache's share out of the pool.
const CONTEXT_TOKENS: usize = 16_384;
const TIGHT_CONTEXT_TOKENS: usize = 8_192;

/// A GPU that may use less than this is "tight" (a 16 GB Mac: ~9.4 GB): the model barely fits
/// and every reserve is trimmed (smaller chunk and context, a smaller scores budget). A 24 GB
/// Mac (~14 GB) is not.
const TIGHT_BELOW: u64 = 11_000_000_000;

/// What the resident trunk (embeddings, attention, DeltaNet, hyper-connections,
/// routers, shared experts) occupies.
const TRUNK_BYTES: u64 = 3_600_000_000;

/// Memory MLX's buffer cache needs beside the live buffers for decode to run at full speed.
/// Past the memory limit MLX gives the cache back on every allocation, and each of the thousands
/// of buffers a decode step makes becomes a new Metal buffer. Measured with a 14 GB budget and
/// two 16K-token conversations live: 0.18 to 0.64 GB of cache room decoded at 8.4 to 9.3 tok/s
/// (the speed with no limit), none (0.01 GB) at 4.3. Half a GB, then.
const CACHE_ROOM: u64 = 500_000_000;

/// A feed that leaves less than this behind (a title request, a short chat turn) is not made to
/// leave the cache its room: it cannot be what overloads the machine, and the session it would
/// drop to make that room is worth a long prefill. It still may not take the live memory past
/// the budget itself, where MLX stops caching altogether.
const SMALL_FEED: u64 = 500_000_000;

/// Bytes a feed leaves behind when it leaves `kv_tokens` tokens of KV (every copy counted: a
/// session holds its own and its rollback point's, a prefix snapshot is another) and `states`
/// sequence states.
fn feed_growth(cfg: &Qwen4ExpConfig, kv_tokens: usize, states: usize) -> u64 {
    kv_tokens as u64 * cfg.kv_bytes_per_token() + states as u64 * cfg.state_bytes()
}

/// Bytes by which `live` bytes in use plus a feed that adds `growth` would go past `budget`,
/// less the cache room decode needs unless the feed is [`SMALL_FEED`].
fn shortfall(live: u64, growth: u64, budget: u64) -> u64 {
    let room = if growth < SMALL_FEED { 0 } else { CACHE_ROOM };
    (live + growth + room).saturating_sub(budget)
}

/// The offsets, within `len` tokens that start at absolute position `start`, where chunks of
/// `chunk` end when chunk boundaries sit on multiples of `chunk` counted from position 0 (the
/// grid a cold prefill uses). The last cut is `len`.
fn chunk_cuts(start: usize, len: usize, chunk: usize) -> Vec<usize> {
    let chunk = chunk.max(1);
    let mut cuts: Vec<usize> = ((start / chunk + 1)..)
        .map(|k| k * chunk - start)
        .take_while(|&c| c < len)
        .collect();
    cuts.push(len);
    cuts
}

/// What the memory is spent on, for a GPU that may use `usable` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MemoryPlan {
    chunk: usize,
    /// Longest prompt in tokens the plan holds the KV cache and attention scores for.
    ctx_tokens: usize,
    /// Most bytes of attention scores a chunk holds at once (longer contexts attend in blocks).
    scores_budget: u64,
    /// Memory a prefill needs beyond the trunk and the expert pool.
    reserve: u64,
    /// The resident weights: [`TRUNK_BYTES`], and the image tower under `LUMEN_VISION`.
    trunk: u64,
    /// What is left for the expert pool.
    pool: u64,
}

/// Memory a prefill needs beyond the trunk and the pool: ~0.6 GB of fixed buffers, ~1.1 MB per
/// chunk token of activations (measured: a 2,460-token chunk peaked 2.5 GB above trunk + pool, a
/// 1,024-token one 1.3 GB), the KV cache of the longest prompt, and half the attention scores
/// (the measured peak at an 18K-token context was 2.56 GB all told, scores included).
fn prefill_reserve(chunk: usize, ctx_tokens: usize, kv_per_token: u64, scores: u64) -> u64 {
    600_000_000 + chunk as u64 * 1_100_000 + ctx_tokens as u64 * kv_per_token + scores / 2
}

/// The chunk, the context and the expert pool for a Mac whose GPU may use `usable` bytes: what
/// the working set leaves after the trunk and a prefill's reserve.
fn plan_memory(
    usable: u64,
    cfg: &Qwen4ExpConfig,
    chunk_override: Option<usize>,
    ctx_override: Option<usize>,
) -> MemoryPlan {
    let tight = usable < TIGHT_BELOW;
    let chunk = chunk_override.unwrap_or(if tight {
        TIGHT_PREFILL_CHUNK
    } else {
        PREFILL_CHUNK
    });
    let ctx_tokens = ctx_override.unwrap_or(if tight {
        TIGHT_CONTEXT_TOKENS
    } else {
        CONTEXT_TOKENS
    });
    let scores_budget: u64 = if tight { 800_000_000 } else { 2_000_000_000 };
    // What a chunk against the whole context would hold, capped by the budget it is blocked to.
    let scores =
        (cfg.num_attention_heads as u64 * chunk as u64 * ctx_tokens as u64 * 4).min(scores_budget);
    let reserve = prefill_reserve(chunk, ctx_tokens, cfg.kv_bytes_per_token(), scores);
    let trunk = TRUNK_BYTES
        + if crate::vision_splice::tower::vision_enabled() {
            cfg.vision_tower_bytes()
        } else {
            0
        };
    MemoryPlan {
        chunk,
        ctx_tokens,
        scores_budget,
        reserve,
        trunk,
        pool: usable.saturating_sub(trunk + reserve),
    }
}

/// The longest prompt the server will accept, which the memory is planned for: its prompt cap
/// (`LUMEN_MAX_PROMPT_TOKENS`, else the legacy `LUMEN_PREFILL_CHUNK`; the same order its prompt
/// guard reads them in), lowered to `LUMEN_MAX_CTX` when that is smaller. `LUMEN_MAX_CTX` alone
/// does not raise it: the guard would still reject a prompt over the cap, and the desktop app
/// always sets it (to 81,920), which must not turn a 16K plan into an 80K one.
fn context_override() -> Option<usize> {
    let number = |k: &str| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&c| c > 0)
    };
    let cap = number("LUMEN_MAX_PROMPT_TOKENS").or_else(|| number("LUMEN_PREFILL_CHUNK"));
    match (cap, number("LUMEN_MAX_CTX")) {
        (Some(cap), Some(ctx)) => Some(cap.min(ctx)),
        (Some(cap), None) => Some(cap),
        (None, Some(ctx)) => Some(ctx.min(CONTEXT_TOKENS)),
        (None, None) => None,
    }
}

/// The rows of `[1, N, vocab]` f32 logits on the host.
fn rows_to_host(logits: &Array) -> Result<Vec<Vec<f32>>> {
    let shape = logits.shape();
    let (n, v) = (shape[1] as usize, shape[2] as usize);
    logits.eval().context("qwen4_exp: evaluate logits")?;
    let all = logits.as_slice::<f32>();
    Ok((0..n).map(|i| all[i * v..(i + 1) * v].to_vec()).collect())
}

fn argmax_host(logits: &[f32]) -> Result<u32> {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .ok_or_else(|| anyhow!("qwen4_exp: empty logits"))
}

/// `LUMEN_QWEN4_MTP`: `off` (the default), `auto` or `on`.
///
/// Off by default: speculative decoding is not bit-identical to plain decoding, and on a 36 GB
/// M3 Max it gained at most 4% and lost up to 4.4% (a verify pass reads about twice the experts
/// of a plain step, and expert reads are what limit decode). `auto` and `on` run the same
/// governor; `on` refuses to start when the head's weights are missing.
fn mtp_mode_from_env() -> Result<MtpMode> {
    parse_mtp_mode(std::env::var("LUMEN_QWEN4_MTP").ok().as_deref())
}

fn parse_mtp_mode(raw: Option<&str>) -> Result<MtpMode> {
    match raw.map(str::trim) {
        None | Some("") | Some("off") | Some("0") | Some("false") => Ok(MtpMode::Off),
        Some("auto") => Ok(MtpMode::Auto),
        Some("on") | Some("1") | Some("true") => Ok(MtpMode::On),
        Some(other) => bail!("LUMEN_QWEN4_MTP={other:?} is not `off`, `auto` or `on`"),
    }
}

/// `LUMEN_QWEN4_MTP_FLOOR`: the share of drafts a sequence must keep getting accepted to keep
/// speculating (0 to 1; default 0.4).
fn mtp_floor_from_env() -> Result<f32> {
    match std::env::var("LUMEN_QWEN4_MTP_FLOOR") {
        Ok(v) if !v.trim().is_empty() => v
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|f| (0.0..=1.0).contains(f))
            .ok_or_else(|| anyhow!("LUMEN_QWEN4_MTP_FLOOR={v:?} is not a number from 0 to 1")),
        _ => Ok(0.4),
    }
}

fn chunk_override() -> Option<usize> {
    std::env::var("LUMEN_QWEN35_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&c| c > 0)
}

/// A pool size as an operator writes it: `auto` (or nothing) for the planned one, else a number
/// of GB with an optional unit: `8`, `8g`, `8GB`, `12.5 g`, `512m`, `512MB` (decimal units, like
/// the rest of the sizes printed here).
fn parse_pool_size(raw: &str) -> Result<Option<u64>> {
    let t = raw.trim();
    if t.is_empty() || t.eq_ignore_ascii_case("auto") {
        return Ok(None);
    }
    let lower = t.to_ascii_lowercase();
    let (number, per_unit) =
        if let Some(n) = lower.strip_suffix("gb").or_else(|| lower.strip_suffix('g')) {
            (n, 1e9)
        } else if let Some(n) = lower.strip_suffix("mb").or_else(|| lower.strip_suffix('m')) {
            (n, 1e6)
        } else {
            (lower.as_str(), 1e9)
        };
    let value: f64 = number
        .trim()
        .parse()
        .ok()
        .filter(|v: &f64| v.is_finite() && *v > 0.0)
        .ok_or_else(|| {
            anyhow!(
                "LUMEN_MOE_EXPERT_CACHE_GB={raw:?} is not a size like `8`, `8g`, `12GB` or `512m`, or `auto`"
            )
        })?;
    Ok(Some((value * per_unit) as u64))
}

/// Slack allowed over the planned pool before an explicit request is lowered: the plan's own
/// reserve is conservative, and the sizes people write are round.
const POOL_OVERSHOOT_OK: u64 = 500_000_000;

lumen_flags::flag! {
    /// Plan the expert pool within the memory that is free right now, not only within the GPU's
    /// working set: the working set's share is lowered to what this process holds plus the pages
    /// the rest of the machine gives up without being compressed or swapped (free, file-backed
    /// and purgeable ones), less a headroom of 2 GB or 5% of RAM, whichever is larger. The
    /// pool cannot change while the server runs, so a server started while other apps hold
    /// memory keeps a smaller pool (and decodes slower) until it is restarted; the log says so.
    /// `0` plans from the working set alone, as before, and the system makes room for the pool by
    /// compressing other apps and writing them to swap.
    pub(crate) plan_reclaimable {
        env: "LUMEN_MEMORY_PLAN_RECLAIMABLE",
        default: true,
        kind: Behavior,
    }
}

/// What a plan made from the memory free now leaves the rest of the machine: [`HEADROOM_MIN`],
/// or [`HEADROOM_RAM_SHARE`] of RAM on a Mac of more than 40 GB. The plan is MLX's, and the
/// process goes past it by ~1.2 GB whatever the pool (measured on Flash-Next with a 15K-token
/// prompt at pools of 9.6, 13.0 and 19.2 GB: peak footprints 1.21, 1.22 and 1.23 GB over the
/// plan): 0.55-0.75 GB it keeps beside MLX's buffers (tokenizer, I/O buffers, Metal driver) and
/// ~0.5 GB by which MLX's prefill peak passes its soft limit. The rest is for the free pages the
/// kernel keeps for itself (0.1-0.4 GB on a busy Mac, which it frees by compressing other apps)
/// and for other apps growing while the server runs. At 1.5 GB that rest would be 0.3 GB.
const HEADROOM_MIN: u64 = 2_000_000_000;
const HEADROOM_RAM_SHARE: f64 = 0.05;

fn headroom(ram: u64) -> u64 {
    HEADROOM_MIN.max((ram as f64 * HEADROOM_RAM_SHARE) as u64)
}

/// The most memory a plan made now may count on (see [`plan_reclaimable`]): what this process
/// holds plus what the rest of the machine gives up without swapping, less the headroom.
fn free_now(host: &HostMemory) -> u64 {
    host.available_to_self().saturating_sub(headroom(host.ram))
}

/// The smallest pool the expert bank runs with: a staging region and a layer beside it, two
/// layers' worth of experts (1,024 slots, 2.83 GB on Flash-Next). The bank raises any smaller
/// budget to its own count of this.
fn pool_floor_bytes(cfg: &Qwen4ExpConfig) -> u64 {
    2 * cfg.num_experts as u64 * cfg.routed_expert_bytes()
}

/// The expert pool a load builds and the memory it is planned within.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PoolChoice {
    /// Bytes of expert pool asked of the bank, never below the floor.
    pool: u64,
    /// The memory the plan is made within: the GPU's share, lowered to what is free now. MLX's
    /// memory limit and the budget the session code keeps the live memory under.
    usable: u64,
    /// Bytes that what is free now took off the GPU's share (0: it took nothing).
    clamped_by: u64,
    /// Not even the smallest pool fits beside the trunk and the reserve: the pool is the floor.
    at_floor: bool,
    /// What an explicit `LUMEN_MOE_EXPERT_CACHE_GB` asked for, when it was lowered to fit.
    lowered_from: Option<u64>,
}

/// The pool for a GPU that may use `device` bytes when `free` bytes are free now (`None`: not
/// read, or [`plan_reclaimable`] is off), with `requested` the operator's
/// `LUMEN_MOE_EXPERT_CACHE_GB` and `floor` the smallest pool the bank runs.
///
/// `plan` is made for `device`: a busy machine keeps the chunk and the context its working set
/// gets (the prompt guard reads the plan's context), only its pool shrinks. A requested size is
/// lowered to what fits, as it is against the device alone: a pool that does not fit would not
/// fail at load but halfway through a long prompt, when the GPU runs out of memory.
fn choose_pool(
    device: u64,
    free: Option<u64>,
    plan: &MemoryPlan,
    requested: Option<u64>,
    floor: u64,
) -> PoolChoice {
    let within = free.map_or(device, |f| f.min(device));
    let fits = within.saturating_sub(plan.trunk + plan.reserve);
    let (wanted, lowered_from) = match requested {
        Some(r) if r > fits + POOL_OVERSHOOT_OK => (fits, Some(r)),
        Some(r) => (r, None),
        None => (fits, None),
    };
    let at_floor = wanted < floor;
    let pool = wanted.max(floor);
    // A pool at its floor holds more than `within` leaves: the plan is for what it holds, up to
    // the device (where a tight Mac's plan stays, as before).
    let usable = if at_floor {
        within.max((plan.trunk + pool + plan.reserve).min(device))
    } else {
        within
    };
    PoolChoice {
        pool,
        usable,
        clamped_by: device - within,
        at_floor,
        lowered_from,
    }
}

/// What is free now, for the plan: `None` when [`plan_reclaimable`] is off (or the counts
/// cannot be read), which plans from the working set alone.
fn host_for_plan(device: u64) -> Option<HostMemory> {
    if !plan_reclaimable::get() {
        return None;
    }
    settled(HostMemory::read, std::thread::sleep, device)
}

/// Readings [`settled`] takes at most after the first, [`SETTLE_EVERY`] apart.
const SETTLE_READS: usize = 6;
const SETTLE_EVERY: std::time::Duration = std::time::Duration::from_millis(500);
/// Growth between two readings below which the memory is taken as settled.
const SETTLE_GROWTH: u64 = 250_000_000;

/// A reading of what is free that is not still growing. The memory of a process that has just
/// exited comes back about a second later (MLX's buffers are wired): measured when a loaded
/// server's process had exited, 2.8 GB reclaimable, then 26 GB a second later. A server restarted
/// at once would plan for the first. So while a reading would cut the plan, it is taken again,
/// half a second apart, until it stops growing (at most 3 s). Memory other apps hold does not
/// grow back, and costs the load one extra reading.
fn settled(
    mut read: impl FnMut() -> Option<HostMemory>,
    mut sleep: impl FnMut(std::time::Duration),
    device: u64,
) -> Option<HostMemory> {
    let mut host = read()?;
    for _ in 0..SETTLE_READS {
        if free_now(&host) >= device {
            break;
        }
        sleep(SETTLE_EVERY);
        let Some(next) = read() else { break };
        let growing = free_now(&next) > free_now(&host) + SETTLE_GROWTH;
        host = next;
        if !growing {
            break;
        }
    }
    Some(host)
}

/// The log line that says what was free when the plan was made, part by part.
fn free_line(host: &HostMemory) -> String {
    format!(
        "[qwen4_exp] free now: {:.1} GB (free {:.1} + file-backed {:.1} + purgeable {:.1}; other \
         apps hold {:.1} GB that only compressing or swapping them frees) + this process {:.1} - \
         headroom {:.1} = {:.1} GB for the plan (LUMEN_MEMORY_PLAN_RECLAIMABLE=0 plans without it)",
        gb(host.reclaimable()),
        gb(host.free),
        gb(host.file_backed),
        gb(host.purgeable),
        gb(host.held_elsewhere()),
        gb(host.own),
        gb(headroom(host.ram)),
        gb(free_now(host)),
    )
}

/// The plan's log line: what the GPU may use, the pool, the chunk and the context, then why the
/// pool is smaller than the working set would make it when it is (other apps hold memory, a
/// requested size does not fit, not even the smallest pool fits). With nothing of that, the line
/// is the one the plan has always printed.
fn plan_line(device: u64, choice: &PoolChoice, plan: &MemoryPlan, use_head: bool) -> String {
    let others = choice.clamped_by > 0;
    let within = device - choice.clamped_by;
    let mut line = format!("[qwen4_exp] GPU may use {:.1} GB", gb(device));
    if others {
        line += &format!(", {:.1} GB of it free now", gb(within));
    }
    line += &format!(
        ": expert pool {:.2} GB, prefill chunk {} tokens, prompts up to {} tokens \
         (LUMEN_MOE_EXPERT_CACHE_GB / LUMEN_QWEN35_PREFILL_CHUNK / LUMEN_MAX_PROMPT_TOKENS \
         override){}",
        gb(choice.pool),
        plan.chunk,
        plan.ctx_tokens,
        if use_head { ", MTP draft head on" } else { "" }
    );
    if let Some(asked) = choice.lowered_from {
        line += &format!(
            ". A {:.1} GB expert pool does not fit: the trunk takes {:.1} GB and a {}-token \
             prompt (chunk {}) {:.1} of the {:.1} GB {}, and a smaller LUMEN_MAX_PROMPT_TOKENS \
             leaves more for the pool",
            gb(asked),
            gb(plan.trunk),
            plan.ctx_tokens,
            plan.chunk,
            gb(plan.reserve),
            gb(within),
            if others {
                "other apps leave"
            } else {
                "this GPU may use"
            },
        );
    }
    if choice.at_floor {
        line += &format!(
            ". Pool {:.2} GB, the smallest the engine runs",
            gb(choice.pool)
        );
        if others {
            line += &format!(
                " (other apps hold memory: {:.1} GB below the working set, and not even this \
                 pool fits beside them, so expect swapping); close apps and restart for a \
                 bigger pool",
                gb(choice.clamped_by)
            );
        } else {
            line += ": the trunk and the prompt's reserve leave less (a smaller \
                     LUMEN_MAX_PROMPT_TOKENS leaves more)";
        }
    } else if others {
        line += &format!(
            ". Pool {:.1} GB (other apps hold memory: {:.1} GB below the working set); close apps \
             and restart for a bigger pool",
            gb(choice.pool),
            gb(choice.clamped_by)
        );
    }
    line
}

impl Qwen4ExpRunner {
    pub(crate) fn load(dir: &Path) -> Result<(Self, LoadInfo)> {
        let ws = crate::metal_memory::working_set_limit().unwrap_or(16_000_000_000) as f64;
        let mut device = (ws * crate::metal_memory::WORKING_SET_SHARE) as u64;
        // An operator who capped MLX's memory plans within it. The server applies
        // `LUMEN_MEMORY_LIMIT_GB` (GiB) after the model is loaded, so read it here as well.
        let capped = std::env::var("LUMEN_MEMORY_LIMIT_GB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map(|gib| gib << 30)
            .into_iter()
            .chain(crate::metal_memory::memory_limit().map(|l| l as u64))
            .min();
        if let Some(limit) = capped
            && limit < device
        {
            device = limit;
        }
        let cfg = Qwen4ExpConfig::load(dir)?;
        let plan = plan_memory(device, &cfg, chunk_override(), context_override());
        let (chunk, ctx_tokens) = (plan.chunk, plan.ctx_tokens);
        let host = host_for_plan(device);
        if let Some(h) = &host {
            eprintln!("{}", free_line(h));
        }
        let requested =
            parse_pool_size(&std::env::var("LUMEN_MOE_EXPERT_CACHE_GB").unwrap_or_default())?;
        let choice = choose_pool(
            device,
            host.as_ref().map(free_now),
            &plan,
            requested,
            pool_floor_bytes(&cfg),
        );
        // MLX's buffer cache would otherwise grow into the memory the plan left other apps: the
        // server lowers MLX's memory limit to this when it sets it (after the load).
        if choice.usable < device {
            crate::metal_memory::set_planned_limit(choice.usable as usize);
        }
        let budget = choice.pool;
        // The draft head's experts go through the pool, so the head costs next to no memory; the
        // governor measures whether drafts pay and stops them when they do not.
        let mode = mtp_mode_from_env()?;
        let head_there = dir.join("mtp.safetensors").is_file();
        if mode == MtpMode::On && !head_there {
            bail!(
                "LUMEN_QWEN4_MTP=on but there is no mtp.safetensors in {}; run scripts/qwen4_mtp_fetch.py",
                dir.display()
            );
        }
        let use_head = head_there && mode != MtpMode::Off;
        eprintln!("{}", plan_line(device, &choice, &plan, use_head));
        let model = Qwen4ExpModel::load_with(
            dir,
            LoadOptions {
                mtp: if use_head { MtpMode::On } else { MtpMode::Off },
                expert_budget_bytes: Some(budget),
                ngram: NgramSource::Auto,
                scores_budget_bytes: 0,
            },
        )
        .with_context(|| format!("qwen4_exp: load `{}`", dir.display()))?;
        let spec = if model.has_mtp() {
            Speculation {
                enabled: true,
                adaptive: true,
                floor: mtp_floor_from_env()?,
                ..Speculation::off()
            }
        } else {
            Speculation::off()
        };
        let info = LoadInfo {
            eos_tokens: model.config().eos_tokens.clone(),
            vocab_size: model.config().vocab_size,
        };
        // A persisted state is only valid for the same weights, MLX build and prefill grid.
        let mut settings = crate::kv_disk::identity_settings_from_env();
        settings.push(("mlx".into(), crate::kv_disk::mlx_version()));
        settings.push(("qwen4_exp_prefill_chunk".into(), chunk.to_string()));
        settings.push(("qwen4_exp_mtp".into(), model.has_mtp().to_string()));
        let fingerprint = crate::kv_disk::kv_identity(&dir.display().to_string(), dir, &settings);
        let disk = crate::kv_disk::open_store_from_env(&fingerprint, "qwen4_exp");
        Ok((
            Self {
                model,
                seqs: HashMap::new(),
                snapshots: HashMap::new(),
                next_snapshot: 1,
                chunk,
                ctx_tokens,
                memory_budget: choice.usable,
                disk,
                fingerprint,
                spec,
                prefed: HashMap::new(),
                #[cfg(test)]
                forced_draft: None,
            },
            info,
        ))
    }

    /// Bytes by which a feed that leaves `kv_tokens` tokens of KV and `states` states in memory
    /// (see [`feed_growth`]) would take MLX past the memory this runner was planned for, counting
    /// everything every sequence and snapshot holds now (and [`CACHE_ROOM`], for a feed that is
    /// not [`SMALL_FEED`]). The plan reserves room for one prompt's KV and activations, so a
    /// second long conversation beside it does not fit; the session and prefix-cache code drops
    /// what is idle until this is 0.
    pub(crate) fn memory_shortfall(&self, kv_tokens: usize, states: usize) -> u64 {
        let Ok(live) = crate::metal_memory::get_active_memory() else {
            return 0;
        };
        // A prompt past the context is refused later, not fed.
        let kv_tokens = kv_tokens.min(3 * self.ctx_tokens);
        shortfall(
            live as u64,
            feed_growth(self.model.config(), kv_tokens, states),
            self.memory_budget,
        )
    }

    /// One `mem_log` line: the allocator's counters and the bytes the sequences and snapshots
    /// reference (a rollback point shares the buffers it has not outgrown, so these can overlap).
    fn mem_report(&self, what: &str) {
        if !mem_log::get() {
            return;
        }
        let active = crate::metal_memory::get_active_memory().unwrap_or(0) as u64;
        let cache = crate::metal_memory::get_cache_memory().unwrap_or(0) as u64;
        let peak = crate::metal_memory::get_peak_memory().unwrap_or(0) as u64;
        let mut ids: Vec<&u64> = self.seqs.keys().collect();
        ids.sort();
        let seqs: Vec<String> = ids
            .into_iter()
            .map(|id| {
                let s = &self.seqs[id];
                format!(
                    "{id}@{} {:.2}+{:.2}+{:.2}",
                    s.position,
                    gb(s.cache.nbytes()),
                    gb(s.rollback.as_ref().map_or(0, |(c, _)| c.nbytes())),
                    gb(s.pending.as_ref().map_or(0, |p| p.state.nbytes())),
                )
            })
            .collect();
        let snaps: u64 = self.snapshots.values().map(|(c, _)| c.nbytes()).sum();
        eprintln!(
            "[qwen4_exp] mem {what}: active {:.2} GB, cache {:.2}, peak {:.2} | seqs [{}] (state+rollback+draft GB), {} snapshots {:.2} GB",
            gb(active),
            gb(cache),
            gb(peak),
            seqs.join(", "),
            self.snapshots.len(),
            gb(snaps),
        );
    }

    /// Count a decode step of `seq_id`; every [`MEM_LOG_EVERY`] steps write a `mem_log` line.
    fn mem_note_decode(&self, seq_id: u64) {
        if !mem_log::get() {
            return;
        }
        let now = std::time::Instant::now();
        let (steps, since) = {
            let mut g = MEM_LOG_DECODE.lock().unwrap_or_else(|e| e.into_inner());
            g.0 += 1;
            if g.0 < MEM_LOG_EVERY && g.1.is_some() {
                return;
            }
            let since = g.1.replace(now);
            let steps = std::mem::take(&mut g.0);
            (steps, since)
        };
        let rate = since.map_or(String::new(), |t| {
            format!(
                ", {steps} steps at {:.2} tok/s",
                steps as f64 / (now - t).as_secs_f64().max(1e-9)
            )
        });
        self.mem_report(&format!("decode seq {seq_id}{rate}"));
    }

    /// Start of a prefill: forget the previous peak so the next report's peak is this prompt's,
    /// and the expert bank's per-call split so the next feed report covers this feed alone.
    fn mem_begin(&self) {
        if mem_log::get() {
            crate::metal_memory::reset_peak_memory();
            self.model.offload_reset_timing();
        }
    }

    /// [`Self::mem_report`] after a feed, followed by the expert bank's account of that feed
    /// (layer calls, reads, and how many layer calls took the staging path).
    fn feed_report(&self, what: &str) {
        self.mem_report(what);
        if mem_log::get()
            && let Some(report) = self.model.offload_report()
        {
            let line = report.lines().next().unwrap_or_default();
            eprintln!("[qwen4_exp] mem {what}: {line}");
        }
    }

    /// Run `tokens` through `cache` in chunks of `self.chunk` (counted from the first token) and
    /// return the last position's logits `[1, 1, vocab]`. The chunks go through the model layer by
    /// layer, so each layer's experts are read once for the whole prompt rather than once per chunk.
    fn forward_chunked(&self, tokens: &[u32], cache: &mut Qwen4ExpCache) -> Result<Array> {
        if tokens.is_empty() {
            bail!("qwen4_exp: no tokens to feed");
        }
        let cuts = chunk_cuts(0, tokens.len(), self.chunk);
        let (logits, _) = self.model.forward_pieces(tokens, cache, &cuts, None)?;
        logits.eval().context("qwen4_exp: evaluate prefill")?;
        Ok(logits)
    }

    /// Feed `tokens` (the start of a sequence when `fresh`, else its continuation) in chunks
    /// that end on the absolute prefill grid, leaving the rollback point `mark` tokens in. That
    /// is what the session's cut-by-cut feed produces, in one layer-major pass.
    pub(crate) fn feed_with_mark(
        &mut self,
        seq_id: u64,
        fresh: bool,
        tokens: &[u32],
        mark: usize,
    ) -> Result<Option<(u32, usize)>> {
        self.mem_begin();
        let start = if fresh {
            0
        } else {
            self.seq(seq_id, "feed")?.position
        };
        self.check_context(start, tokens.len())?;
        let cuts = chunk_cuts(start, tokens.len(), self.chunk);
        if tokens.is_empty() || !cuts.contains(&mark) {
            return Ok(None);
        }
        let mut cache = if fresh {
            self.model.make_cache()
        } else {
            self.seq(seq_id, "feed")?.cache.clone()
        };
        let (logits, point) = self
            .model
            .forward_pieces(tokens, &mut cache, &cuts, Some(mark))?;
        logits.eval().context("qwen4_exp: evaluate prefill")?;
        let next = Self::argmax(&logits)?;
        let position = start + tokens.len();
        let point = point.map(|p| (p, start + mark));
        match self.seqs.get_mut(&seq_id) {
            Some(seq) if !fresh => {
                seq.cache = cache;
                seq.position = position;
                seq.rollback = point;
            }
            _ => {
                self.seqs.insert(
                    seq_id,
                    Seq {
                        cache,
                        position,
                        rollback: point,
                        persisted_key: None,
                        forked_from: None,
                        pending: None,
                        recent: Default::default(),
                        spec_off: false,
                    },
                );
            }
        }
        self.feed_report(&format!("fed {} tokens to seq {seq_id}", tokens.len()));
        Ok(Some((next, position)))
    }

    /// Refuse to take a prompt the memory was not planned for, before any of it is computed: the
    /// KV cache and attention scores would not fit, and the failure would come as a GPU abort.
    fn check_context(&self, have: usize, adding: usize) -> Result<()> {
        if have + adding > self.ctx_tokens {
            bail!(
                "qwen4_exp: a context of {} tokens is more than the {} this machine's memory is \
                 planned for; shorten the conversation, or raise LUMEN_MAX_PROMPT_TOKENS (the \
                 expert pool shrinks to make room)",
                have + adding,
                self.ctx_tokens
            );
        }
        Ok(())
    }

    /// Longest prompt, in tokens, the memory is planned for.
    pub(crate) fn max_context(&self) -> usize {
        self.ctx_tokens
    }

    fn argmax(logits: &Array) -> Result<u32> {
        let flat = mlx_rs::ops::reshape(logits, &[-1])?;
        let idx = mlx_rs::ops::indexing::argmax_axis(&flat, 0, false)?;
        idx.eval().context("qwen4_exp: argmax")?;
        Ok(idx.item::<u32>())
    }

    fn seq(&mut self, seq_id: u64, what: &str) -> Result<&mut Seq> {
        let seq = self
            .seqs
            .get_mut(&seq_id)
            .ok_or_else(|| anyhow!("qwen4_exp runner: {what} on unknown seq_id {seq_id}"))?;
        // Whoever asks may change it: it no longer stands for the snapshot it was forked from.
        seq.forked_from = None;
        Ok(seq)
    }

    /// One decode step with the pick left to the caller, over the host logits. Always a plain
    /// step: under a grammar mask the token a draft would have to match is not known until
    /// the mask is applied to the logits of the position before it.
    pub(crate) fn decode_step_pick(
        &mut self,
        seq_id: u64,
        last_token: u32,
        pick: &mut dyn FnMut(&mut [f32]) -> Result<u32>,
    ) -> Result<(u32, usize)> {
        self.mem_note_decode(seq_id);
        let model = &self.model;
        let seq = self.seqs.get_mut(&seq_id).ok_or_else(|| {
            anyhow!("qwen4_exp runner: decode_step_pick on unknown seq_id {seq_id}")
        })?;
        // A token buffered for a plain call is not valid for a masked one (it was picked without
        // the mask): go back to the state before it and take the step.
        seq.pending = None;
        seq.forked_from = None;
        let logits = model.forward(&[last_token], &mut seq.cache, true)?;
        let mut host = crate::gemma4_sampling::imp::last_logits_to_cpu_f32(&logits)
            .context("qwen4_exp: logits to host")?;
        let next = pick(&mut host).context("qwen4_exp: sampler callback")?;
        seq.position += 1;
        Ok((next, seq.position))
    }

    /// A decode step that may speculate: with the draft head loaded, a round drafts the token
    /// after next, verifies it in one two-row pass, and when the verifier agrees returns the
    /// first token now and the second on the next call (see [`Pending`]). `pick` chooses a
    /// token from one position's host logits; it is called for the first position, and for the
    /// second only when the draft matched, in the order a plain loop would call it.
    /// Without a head, or once drafts stopped paying, this is a plain step.
    pub(crate) fn decode_step_spec(
        &mut self,
        seq_id: u64,
        last_token: u32,
        pick: &mut dyn FnMut(&mut [f32]) -> Result<u32>,
    ) -> Result<(u32, usize)> {
        self.decode_step_spec_as(seq_id, last_token, pick, false)
    }

    /// [`Self::decode_step_spec`]; `greedy` says `pick` is a plain argmax, which a plain step
    /// takes on the GPU instead of copying a vocabulary of logits to the host.
    fn decode_step_spec_as(
        &mut self,
        seq_id: u64,
        last_token: u32,
        pick: &mut dyn FnMut(&mut [f32]) -> Result<u32>,
        greedy: bool,
    ) -> Result<(u32, usize)> {
        self.mem_note_decode(seq_id);
        let seq = self.seqs.get_mut(&seq_id).ok_or_else(|| {
            anyhow!("qwen4_exp runner: decode_step_spec on unknown seq_id {seq_id}")
        })?;
        seq.forked_from = None;
        let started = std::time::Instant::now();
        if let Some(p) = seq.pending.take() {
            if p.after == last_token {
                seq.cache = p.state;
                seq.position += 1;
                if self.spec.adaptive {
                    self.spec
                        .governor
                        .note(true, started.elapsed().as_secs_f64() * 1e3);
                }
                return Ok((p.next, seq.position));
            }
            // The caller went another way: the state before the buffered token is the one in
            // `seq.cache`, and `last_token` is not yet processed in it.
        }
        let model = &self.model;
        let capable = self.spec.enabled && !seq.spec_off && model.mtp_ready(&seq.cache);
        let speculate = capable && (!self.spec.adaptive || self.spec.governor.should_speculate());
        if !speculate {
            let logits = model.forward(&[last_token], &mut seq.cache, true)?;
            let next = if greedy {
                Self::argmax(&logits)?
            } else {
                let mut host = crate::gemma4_sampling::imp::last_logits_to_cpu_f32(&logits)
                    .context("qwen4_exp: logits to host")?;
                pick(&mut host).context("qwen4_exp: sampler callback")?
            };
            seq.position += 1;
            if capable && self.spec.adaptive {
                self.spec
                    .governor
                    .note(false, started.elapsed().as_secs_f64() * 1e3);
            }
            return Ok((next, seq.position));
        }

        // Draft: the head's guess for the token after `last_token`'s successor.
        let t0 = std::time::Instant::now();
        let draft = model
            .draft_for(&mut seq.cache, last_token)?
            .ok_or_else(|| anyhow!("qwen4_exp: the draft head has no state"))?;
        #[cfg(test)]
        let draft = self.forced_draft.as_mut().map_or(draft, |f| f());
        let t_draft = t0.elapsed();
        // Verify on a copy: the sequence keeps its state until the round is settled.
        let t1 = std::time::Instant::now();
        let mut full = seq.cache.clone();
        let mut v = model.forward_verify(&[last_token, draft], &mut full)?;
        let t_verify = t1.elapsed();
        let rows = rows_to_host(&v.logits).context("qwen4_exp: verify logits to host")?;
        let mut rows = rows.into_iter();
        let mut row0 = rows
            .next()
            .ok_or_else(|| anyhow!("qwen4_exp: no verify row 0"))?;
        let first = pick(&mut row0).context("qwen4_exp: sampler callback")?;
        let accepted = first == draft;
        self.spec.note_round(accepted, t_draft, t_verify);
        seq.recent.push_back(accepted);
        if seq.recent.len() > SPEC_WINDOW {
            seq.recent.pop_front();
        }
        if seq.recent.len() == SPEC_WINDOW {
            let hits = seq.recent.iter().filter(|a| **a).count();
            if (hits as f32) < self.spec.floor * SPEC_WINDOW as f32 {
                seq.spec_off = true;
                eprintln!(
                    "[qwen4_exp] seq {seq_id}: {hits} of the last {SPEC_WINDOW} drafts accepted, \
                     below the {:.0}% floor; speculation off for this sequence",
                    self.spec.floor * 100.0
                );
            }
        }
        if accepted {
            let mut row1 = rows
                .next()
                .ok_or_else(|| anyhow!("qwen4_exp: no verify row 1"))?;
            let second = pick(&mut row1).context("qwen4_exp: sampler callback")?;
            model.mtp_finish_round(&mut v.first, Some(&mut full), &v.multi, draft)?;
            seq.cache = v.first;
            seq.position += 1;
            seq.pending = Some(Pending {
                after: first,
                next: second,
                state: full,
            });
        } else {
            model.mtp_finish_round(&mut v.first, None, &v.multi, draft)?;
            seq.cache = v.first;
            seq.position += 1;
        }
        if self.spec.adaptive {
            self.spec
                .governor
                .note(true, started.elapsed().as_secs_f64() * 1e3);
        }
        Ok((first, seq.position))
    }

    pub(crate) fn decode_step_masked(
        &mut self,
        seq_id: u64,
        last_token: u32,
        mask: &mut dyn FnMut(&mut [f32]) -> Result<()>,
    ) -> Result<(u32, usize)> {
        self.decode_step_pick(seq_id, last_token, &mut |logits| {
            mask(logits)?;
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .ok_or_else(|| anyhow!("qwen4_exp: empty logits"))
        })
    }
}

impl Qwen4ExpRunner {
    /// Decode and resize one image onto the tower's patch grid.
    pub(crate) fn prepare_image(
        &self,
        encoded: &[u8],
    ) -> Result<crate::qwen36_vision::PreparedImage> {
        let tower = self.model.vision().ok_or_else(|| {
            anyhow!("image input requires LUMEN_VISION=1 and a checkpoint with a vision tower")
        })?;
        crate::qwen36_vision::prepare_image_with(
            encoded,
            tower.config(),
            self.model.config().image_norm,
        )
        .map_err(|e| anyhow!(e))
    }

    /// The id whose rows an image's features replace, when the tower is loaded.
    pub(crate) fn image_token_id(&self) -> Option<u32> {
        self.model.vision().and(self.model.config().image_token_id)
    }

    /// The loaded tower's config, for sizing a prompt's images without decoding them.
    pub(crate) fn vision_config(&self) -> Option<crate::qwen36_vision::NativeQwen36VisionConfig> {
        self.model.vision().map(|t| t.config().clone())
    }

    /// Start `seq_id` with an image prompt: `tokens` holds each image's placeholder run, `images`
    /// the images in prompt order. The sequence takes 3-D rotary positions (decode then runs at
    /// index + delta), its own fresh state (no prefix cache, session or disk tier: they are keyed
    /// on text), and decodes without a draft head or speculation. Without images it is a plain
    /// prefill.
    pub(crate) fn prefill_with_images(
        &mut self,
        seq_id: u64,
        tokens: &[u32],
        images: &[crate::qwen36_vision::PreparedImage],
    ) -> Result<(u32, usize)> {
        if images.is_empty() {
            return Runner::prefill(self, seq_id, tokens);
        }
        if self.seqs.contains_key(&seq_id) {
            bail!("qwen4_exp: seq {seq_id} already exists");
        }
        self.mem_begin();
        self.check_context(0, tokens.len())?;
        let merge = self
            .model
            .vision()
            .ok_or_else(|| anyhow!("image input requires LUMEN_VISION=1 and a vision tower"))?
            .config()
            .spatial_merge_size;
        let image_token = self
            .model
            .config()
            .image_token_id
            .ok_or_else(|| anyhow!("config.json has no image_token_id"))?;
        let grids: Vec<(usize, usize)> = images.iter().map(|p| p.merged_grid(merge)).collect();
        let positions = crate::qwen36_vision::mrope_positions(tokens, image_token, &grids)
            .map_err(|e| anyhow!("qwen4_exp: image positions: {e}"))?;
        let delta = crate::qwen36_vision::next_position(&positions) - tokens.len() as i32;
        let (runs, soft) = self.model.encode_images(tokens, images)?;
        let mut cache = self.model.make_cache();
        let logits = self.model.forward_image_prompt(
            tokens, &mut cache, self.chunk, &positions, delta, &runs, &soft,
        )?;
        let next = Self::argmax(&logits)?;
        self.seqs.insert(
            seq_id,
            Seq {
                cache,
                position: tokens.len(),
                rollback: None,
                persisted_key: None,
                forked_from: None,
                pending: None,
                recent: Default::default(),
                spec_off: true,
            },
        );
        self.feed_report(&format!(
            "prefilled an image prompt of {} tokens ({} image(s)) for seq {seq_id}",
            tokens.len(),
            images.len()
        ));
        Ok((next, tokens.len()))
    }

    /// Write `seq_id`'s rollback point (the state after `prefix`, placed on a prefill grid of
    /// `chunk`) to the disk tier. Returns whether anything was written.
    pub(crate) fn persist_point(
        &mut self,
        seq_id: u64,
        prefix: &[u32],
        chunk: usize,
    ) -> Result<bool> {
        let Some(disk) = self.disk.as_ref() else {
            return Ok(false);
        };
        let key = crate::kv_disk::point_key(chunk, prefix);
        if disk.contains(&key) {
            return Ok(false);
        }
        let Some(seq) = self.seqs.get(&seq_id) else {
            return Ok(false);
        };
        let Some((point, position)) = seq.rollback.as_ref().filter(|(_, p)| *p == prefix.len())
        else {
            return Ok(false);
        };
        let _ = position;
        let t_read = std::time::Instant::now();
        let (layers, records) = point.to_records()?;
        let read_ms = t_read.elapsed().as_secs_f64() * 1000.0;
        let manifest = KvManifest {
            model_fingerprint: self.fingerprint.clone(),
            created_at_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            position: prefix.len(),
            prefix_tokens: prefix.to_vec(),
            last_token: None,
            is_deep: true,
            layers,
        };
        let bytes: usize = records.iter().map(|r| r.data.len()).sum();
        let disk = self
            .disk
            .as_mut()
            .expect("disk tier present (checked above)");
        disk.put(&key, &manifest, &records)?;
        let superseded = self
            .seqs
            .get_mut(&seq_id)
            .and_then(|s| s.persisted_key.replace(key.clone()))
            .filter(|old| *old != key);
        // Dropped once this state is on the disk; a refused write keeps the older one.
        if let Some(old) = superseded {
            disk.supersede(&old, &key)?;
        }
        eprintln!(
            "[qwen4_exp] kv-disk: point at {} = {:.0} MB (readback {read_ms:.0} ms)",
            prefix.len(),
            bytes as f64 / 1e6
        );
        Ok(true)
    }

    /// Start `seq_id` from the longest persisted state `prompt` continues (on this
    /// prompt's own prefill grid), with a rollback point there. Returns its length.
    pub(crate) fn restore_point(&mut self, seq_id: u64, prompt: &[u32]) -> Result<Option<usize>> {
        if self.disk.is_none() || self.seqs.contains_key(&seq_id) {
            return Ok(None);
        }
        let chunk = self.chunk.max(1);
        let last = prompt.len().saturating_sub(1) / chunk * chunk;
        for at in (chunk..=last).rev().step_by(chunk) {
            let key = crate::kv_disk::point_key(chunk, &prompt[..at]);
            let disk = self
                .disk
                .as_mut()
                .expect("disk tier present (checked above)");
            if !disk.contains(&key) {
                continue;
            }
            let Some((manifest, records)) = disk.get_arrays(&key)? else {
                continue;
            };
            if manifest.position != at || manifest.prefix_tokens != prompt[..at] {
                continue;
            }
            let cache =
                match self
                    .model
                    .cache_from_records(&manifest.layers, &records, &prompt[..at])
                {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!(
                            "[qwen4_exp] kv-disk: state at {at} failed to load ({e:#}); skipping"
                        );
                        continue;
                    }
                };
            self.seqs.insert(
                seq_id,
                Seq {
                    rollback: Some((cache.clone(), at)),
                    cache,
                    position: at,
                    persisted_key: Some(key),
                    forked_from: None,
                    pending: None,
                    recent: Default::default(),
                    spec_off: false,
                },
            );
            return Ok(Some(at));
        }
        Ok(None)
    }

    /// Prefill `tokens` from position 0 in the chunks a prefill of `tokens[..at]` followed by an
    /// extend of the rest makes (chunks counted from 0 up to `at`, then from `at`), in one
    /// layer-major pass: each layer's experts are read once where the two calls read them twice.
    /// With `keep`, the state after `at` tokens is kept as a snapshot: `(next token, position,
    /// snapshot id)`. The prefix cache's cold MISS on this runner, `at` being the system/tools
    /// boundary: a head forked from there and extended is the same computation.
    pub(crate) fn prefill_split(
        &mut self,
        seq_id: u64,
        tokens: &[u32],
        at: usize,
        keep: bool,
    ) -> Result<(u32, usize, Option<u64>)> {
        let (next, position, head, _) =
            self.prefill_split_marked(seq_id, tokens, at, keep, None)?;
        Ok((next, position, head))
    }

    /// [`Self::prefill_split`] that also keeps the state at `mark` (a cut of the split feed: `at`
    /// plus a whole number of chunks) as a snapshot: `(next token, position, head snapshot, mark
    /// snapshot)`, still in one pass.
    pub(crate) fn prefill_split_marked(
        &mut self,
        seq_id: u64,
        tokens: &[u32],
        at: usize,
        keep: bool,
        mark: Option<usize>,
    ) -> Result<(u32, usize, Option<u64>, Option<u64>)> {
        if at == 0 || at >= tokens.len() {
            bail!(
                "qwen4_exp: split at {at} is not inside {} tokens",
                tokens.len()
            );
        }
        self.mem_begin();
        self.check_context(0, tokens.len())?;
        let cuts = split_cuts(tokens.len(), at, self.chunk);
        let mark = mark.filter(|&m| m != at);
        let marks: Vec<usize> = keep.then_some(at).into_iter().chain(mark).collect();
        let mut cache = self.model.make_cache();
        let (logits, mut points) = self
            .model
            .forward_pieces_marked(tokens, &mut cache, &cuts, &marks)?;
        logits.eval().context("qwen4_exp: evaluate prefill")?;
        let next = Self::argmax(&logits)?;
        let mark_point = mark.and_then(|m| points.pop().map(|p| (p, m)));
        let point = points.pop();
        self.seqs.insert(
            seq_id,
            Seq {
                cache,
                position: tokens.len(),
                rollback: None,
                persisted_key: None,
                forked_from: None,
                pending: None,
                recent: Default::default(),
                spec_off: false,
            },
        );
        let snapshot = point.map(|p| self.keep_snapshot(p, at));
        let mark_snapshot = mark_point.map(|(p, m)| self.keep_snapshot(p, m));
        self.feed_report(&format!(
            "prefilled {} tokens for seq {seq_id} (head at {at})",
            tokens.len()
        ));
        Ok((next, tokens.len(), snapshot, mark_snapshot))
    }

    /// File `state` (standing at `position`) as a snapshot and return its id.
    fn keep_snapshot(&mut self, state: Qwen4ExpCache, position: usize) -> u64 {
        let id = self.next_snapshot;
        self.next_snapshot += 1;
        self.snapshots.insert(id, (state, position));
        id
    }

    /// Compute ahead, together, the extends `items` describe: each one a sequence forked from
    /// `snapshot` and extended by `tokens`, keeping the state `mark` tokens in, exactly as
    /// [`Self::extend_marked`] would compute it (the chunks counted from the snapshot, through
    /// [`Qwen4ExpModel::forward_many`]), so that one pass over the layers reads each layer's
    /// experts once for all of them. The extend that later asks for one of them takes the result
    /// instead of computing it; one nobody asks for is dropped by [`Self::clear_prefeed`].
    ///
    /// `items` are the feeds of every request still to be served, so a feed computed earlier that
    /// is not among them (its request was served, or went another way) is dropped first. Takes
    /// the rest in order while they fit one joint feed ([`JOINT_FEED_TOKENS`]) and skips any this
    /// runner could not extend as asked (an unknown snapshot, a context past the plan, a mark
    /// between chunk ends: the extend itself reports those). Returns how many it fed; with fewer
    /// than two there is nothing to share and it feeds none.
    pub(crate) fn prefeed(&mut self, items: &[crate::prefix_cache::PrefeedItem]) -> Result<usize> {
        let before = self.prefed.len();
        self.prefed.retain(|k, _| {
            items
                .iter()
                .any(|i| i.snapshot == k.snapshot && i.mark == k.mark && i.tokens == k.tokens)
        });
        if self.prefed.len() < before {
            let _ = crate::metal_memory::clear_cache();
        }
        let mut take: Vec<(PrefedKey, Qwen4ExpCache, usize, Vec<usize>)> = Vec::new();
        let mut tokens = 0;
        for item in items {
            let key = PrefedKey {
                snapshot: item.snapshot,
                mark: item.mark,
                tokens: item.tokens.clone(),
            };
            if item.tokens.is_empty()
                || self.prefed.contains_key(&key)
                || take.iter().any(|(k, ..)| *k == key)
                || tokens + item.tokens.len() > JOINT_FEED_TOKENS
            {
                continue;
            }
            let Some((cache, position)) = self.snapshots.get(&item.snapshot).cloned() else {
                continue;
            };
            let cuts = chunk_cuts(0, item.tokens.len(), self.chunk);
            if self.check_context(position, item.tokens.len()).is_err()
                || item.mark.is_some_and(|m| !cuts.contains(&m))
            {
                continue;
            }
            tokens += item.tokens.len();
            take.push((key, cache, position, cuts));
        }
        if take.len() < 2 {
            return Ok(0);
        }
        self.mem_begin();
        let started = std::time::Instant::now();
        let mut caches: Vec<Qwen4ExpCache> = take.iter().map(|(_, c, ..)| c.clone()).collect();
        let results = {
            let mut feeds: Vec<super::model::JointFeed<'_>> = take
                .iter()
                .zip(caches.iter_mut())
                .map(|((key, _, _, cuts), cache)| super::model::JointFeed {
                    ids: &key.tokens,
                    cache,
                    cuts,
                    mark: key.mark,
                })
                .collect();
            self.model.forward_many(&mut feeds)?
        };
        let fed = take.len();
        for (((key, ..), cache), (logits, point)) in take.into_iter().zip(caches).zip(results) {
            logits.eval().context("qwen4_exp: evaluate a joint feed")?;
            let next = Self::argmax(&logits)?;
            self.prefed.insert(key, Prefed { cache, next, point });
        }
        self.feed_report(&format!(
            "fed {fed} sequences together ({tokens} tokens) in {:.0} ms",
            started.elapsed().as_secs_f64() * 1e3
        ));
        Ok(fed)
    }

    /// The feed [`Self::prefeed`] computed for extending `seq_id` by `tokens` (keeping the state
    /// `mark` tokens in), installed as that extend would leave it: `(next token, position,
    /// snapshot id of the kept state)`. `None` unless the sequence is still the snapshot the feed
    /// started from.
    fn take_prefed(
        &mut self,
        seq_id: u64,
        tokens: &[u32],
        mark: Option<usize>,
    ) -> Option<(u32, usize, Option<u64>)> {
        if self.prefed.is_empty() {
            return None;
        }
        let snapshot = self.seqs.get(&seq_id)?.forked_from?;
        let have = self.snapshots.get(&snapshot)?.1;
        if self.seqs.get(&seq_id)?.position != have {
            return None;
        }
        let fed = self.prefed.remove(&PrefedKey {
            snapshot,
            mark,
            tokens: tokens.to_vec(),
        })?;
        let seq = self.seqs.get_mut(&seq_id)?;
        seq.forked_from = None;
        seq.pending = None;
        seq.cache = fed.cache;
        seq.position = have + tokens.len();
        let position = seq.position;
        let kept = fed
            .point
            .zip(mark)
            .map(|(p, m)| self.keep_snapshot(p, have + m));
        self.feed_report(&format!(
            "extended seq {seq_id} by {} tokens (fed ahead, together with others)",
            tokens.len()
        ));
        Some((fed.next, position, kept))
    }

    /// Whether [`Self::prefeed`] holds the result of `item`.
    pub(crate) fn holds_prefed(&self, item: &crate::prefix_cache::PrefeedItem) -> bool {
        self.prefed.contains_key(&PrefedKey {
            snapshot: item.snapshot,
            mark: item.mark,
            tokens: item.tokens.clone(),
        })
    }

    /// Drop the feeds [`Self::prefeed`] computed that no extend took.
    pub(crate) fn clear_prefeed(&mut self) {
        if !self.prefed.is_empty() {
            self.prefed.clear();
            let _ = crate::metal_memory::clear_cache();
        }
    }

    /// [`Runner::extend`] (chunks counted from the sequence's position) that also keeps the state
    /// `mark` tokens in as a snapshot when `mark` is one of those chunk ends: `(next token,
    /// position, snapshot id)`. A mark between chunk ends is an error, not a guess.
    pub(crate) fn extend_marked(
        &mut self,
        seq_id: u64,
        tokens: &[u32],
        mark: Option<usize>,
    ) -> Result<(u32, usize, Option<u64>)> {
        let Some(mark) = mark else {
            let (next, position) = Runner::extend(self, seq_id, tokens)?;
            return Ok((next, position, None));
        };
        self.mem_begin();
        if let Some(fed) = self.take_prefed(seq_id, tokens, Some(mark)) {
            return Ok(fed);
        }
        let have = self.seq(seq_id, "extend")?.position;
        self.check_context(have, tokens.len())?;
        if tokens.is_empty() {
            bail!("qwen4_exp: no tokens to feed");
        }
        let cuts = chunk_cuts(0, tokens.len(), self.chunk);
        if !cuts.contains(&mark) {
            bail!(
                "qwen4_exp: mark {mark} is not a chunk end of a {}-token extend",
                tokens.len()
            );
        }
        let seq = self.seq(seq_id, "extend")?;
        seq.pending = None;
        let mut cache = seq.cache.clone();
        let (logits, point) = self
            .model
            .forward_pieces(tokens, &mut cache, &cuts, Some(mark))?;
        logits.eval().context("qwen4_exp: evaluate prefill")?;
        let next = Self::argmax(&logits)?;
        let seq = self.seq(seq_id, "extend")?;
        seq.cache = cache;
        seq.position += tokens.len();
        let position = seq.position;
        let snapshot = point.map(|p| self.keep_snapshot(p, have + mark));
        self.feed_report(&format!(
            "extended seq {seq_id} by {} tokens (point at {})",
            tokens.len(),
            have + mark
        ));
        Ok((next, position, snapshot))
    }

    /// Write prefix-cache snapshot `snapshot_id`, the state after `prefix`, to the disk tier
    /// under `key` (the prefix cache gives heads only, under a key made from their tokens). An
    /// entry the disk already holds is only marked used (its TTL counts from its last use).
    /// Called after the reply: the readback is the only part on this thread, the file write goes
    /// to the tier's writer thread.
    pub(crate) fn persist_head(
        &mut self,
        snapshot_id: u64,
        key: &str,
        prefix: &[u32],
    ) -> Result<()> {
        let Some(disk) = self.disk.as_mut() else {
            return Ok(());
        };
        if disk.touch(key)? {
            return Ok(());
        }
        let (cache, position) = self
            .snapshots
            .get(&snapshot_id)
            .ok_or_else(|| anyhow!("qwen4_exp: persist of unknown snapshot {snapshot_id}"))?;
        if *position != prefix.len() {
            bail!(
                "qwen4_exp: snapshot {snapshot_id} stands at {position}, not at the {} tokens it is filed under",
                prefix.len()
            );
        }
        let t_read = std::time::Instant::now();
        let (layers, records) = cache.to_records()?;
        let read_ms = t_read.elapsed().as_secs_f64() * 1000.0;
        let manifest = KvManifest {
            model_fingerprint: self.fingerprint.clone(),
            created_at_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            position: prefix.len(),
            prefix_tokens: prefix.to_vec(),
            last_token: None,
            is_deep: true,
            layers,
        };
        let bytes: usize = records.iter().map(|r| r.data.len()).sum();
        let t_put = std::time::Instant::now();
        self.disk
            .as_mut()
            .expect("disk tier present (checked above)")
            .put(key, &manifest, &records)?;
        eprintln!(
            "[qwen4_exp] kv-disk: head at {} = {:.0} MB (readback {read_ms:.0} ms, queued in {:.0} ms)",
            prefix.len(),
            bytes as f64 / 1e6,
            t_put.elapsed().as_secs_f64() * 1000.0
        );
        Ok(())
    }

    /// The head stored under `key`, loaded as a snapshot when `prompt` continues it with at least
    /// one token left to feed: `(snapshot id, its tokens, None)`. A missing, truncated or corrupt
    /// file (CRC), one written for other weights or settings, one whose tokens are not a prefix
    /// of `prompt`, or a state that does not fit the model is a miss, never an error.
    pub(crate) fn load_head(
        &mut self,
        key: &str,
        prompt: &[u32],
    ) -> Result<Option<(u64, Vec<u32>, Option<u32>)>> {
        let Some(disk) = self.disk.as_mut() else {
            return Ok(None);
        };
        let t = std::time::Instant::now();
        let Some((manifest, records)) = disk.get_arrays(key)? else {
            return Ok(None);
        };
        let n = manifest.prefix_tokens.len();
        if n == 0
            || manifest.position != n
            || n >= prompt.len()
            || !prompt.starts_with(&manifest.prefix_tokens)
        {
            return Ok(None);
        }
        let cache =
            match self
                .model
                .cache_from_records(&manifest.layers, &records, &manifest.prefix_tokens)
            {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("[qwen4_exp] kv-disk: head at {n} failed to load ({e:#}); not used");
                    return Ok(None);
                }
            };
        eprintln!(
            "[qwen4_exp] kv-disk: head at {n} ({:.0} MB) read back in {:.0} ms",
            cache.nbytes() as f64 / 1e6,
            t.elapsed().as_secs_f64() * 1000.0
        );
        let id = self.next_snapshot;
        self.next_snapshot += 1;
        self.snapshots.insert(id, (cache, n));
        Ok(Some((id, manifest.prefix_tokens, None)))
    }

    /// Tests: attach a disk tier rooted at `root` under `fingerprint` (inline writes), or detach
    /// it with `None`.
    #[cfg(test)]
    pub(crate) fn set_disk_root_for_test(&mut self, root: Option<&Path>, fingerprint: &str) {
        self.fingerprint = fingerprint.to_string();
        self.disk =
            root.map(|r| DiskKvStore::open(r, fingerprint, 0, 0).expect("open test disk tier"));
    }
}

/// The chunk ends of a feed of `len` tokens split at `at`: the chunks of a prefill of the first
/// `at` tokens, then those of an extend of the rest (each counted from its own start).
fn split_cuts(len: usize, at: usize, chunk: usize) -> Vec<usize> {
    let mut cuts = chunk_cuts(0, at, chunk);
    cuts.extend(chunk_cuts(0, len - at, chunk).into_iter().map(|c| c + at));
    cuts
}

/// The prefix cache's view of this runner: the `Runner` methods, a one-pass split prefill, the
/// disk tier as heads, and the memory plan.
impl crate::prefix_cache::SnapshotRunner for Qwen4ExpRunner {
    fn prefill(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
        Runner::prefill(self, seq_id, tokens)
    }
    fn extend(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
        Runner::extend(self, seq_id, tokens)
    }
    fn snapshot_state_deep(&mut self, seq_id: u64) -> Result<(u64, usize)> {
        Runner::snapshot_state_deep(self, seq_id)
    }
    fn fork_from_snapshot(&mut self, snapshot_id: u64, dst_seq_id: u64) -> Result<usize> {
        Runner::fork_from_snapshot(self, snapshot_id, dst_seq_id)
    }
    fn release_snapshot(&mut self, snapshot_id: u64) -> Result<()> {
        Runner::release_snapshot(self, snapshot_id)
    }
    fn persist_snapshot(
        &mut self,
        snapshot_id: u64,
        key: &str,
        prefix_tokens: &[u32],
        _last_token: Option<u32>,
    ) -> Result<()> {
        self.persist_head(snapshot_id, key, prefix_tokens)
    }
    fn load_persisted_snapshot(
        &mut self,
        key: &str,
        prompt_ids: &[u32],
    ) -> Result<Option<(u64, Vec<u32>, Option<u32>)>> {
        self.load_head(key, prompt_ids)
    }
    fn memory_shortfall(&self, kv_tokens: usize, states: usize) -> u64 {
        Qwen4ExpRunner::memory_shortfall(self, kv_tokens, states)
    }
    fn cut_sensitive(&self) -> bool {
        true
    }
    fn prefill_split(
        &mut self,
        seq_id: u64,
        tokens: &[u32],
        at: usize,
        keep: bool,
    ) -> Result<(u32, usize, Option<u64>)> {
        Qwen4ExpRunner::prefill_split(self, seq_id, tokens, at, keep)
    }
    fn resume_chunk(&self) -> Option<usize> {
        Some(self.chunk.max(1))
    }
    fn prefill_split_marked(
        &mut self,
        seq_id: u64,
        tokens: &[u32],
        at: usize,
        keep: bool,
        mark: Option<usize>,
    ) -> Result<(u32, usize, Option<u64>, Option<u64>)> {
        Qwen4ExpRunner::prefill_split_marked(self, seq_id, tokens, at, keep, mark)
    }
    fn extend_marked(
        &mut self,
        seq_id: u64,
        tokens: &[u32],
        mark: Option<usize>,
    ) -> Result<(u32, usize, Option<u64>)> {
        Qwen4ExpRunner::extend_marked(self, seq_id, tokens, mark)
    }
    fn prefeed(&mut self, items: &[crate::prefix_cache::PrefeedItem]) -> Result<usize> {
        Qwen4ExpRunner::prefeed(self, items)
    }
    fn clear_prefeed(&mut self) {
        Qwen4ExpRunner::clear_prefeed(self)
    }
    fn holds_prefed(&self, item: &crate::prefix_cache::PrefeedItem) -> bool {
        Qwen4ExpRunner::holds_prefed(self, item)
    }
}

impl Runner for Qwen4ExpRunner {
    fn name(&self) -> &'static str {
        "qwen4_exp"
    }

    fn prefill_with_images(
        &mut self,
        seq_id: u64,
        tokens: &[u32],
        images: &[crate::qwen36_vision::PreparedImage],
    ) -> Result<(u32, usize)> {
        Qwen4ExpRunner::prefill_with_images(self, seq_id, tokens, images)
    }

    fn prefill(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
        self.mem_begin();
        self.check_context(0, tokens.len())?;
        let mut cache = self.model.make_cache();
        let logits = self.forward_chunked(tokens, &mut cache)?;
        let next = Self::argmax(&logits)?;
        self.seqs.insert(
            seq_id,
            Seq {
                cache,
                position: tokens.len(),
                rollback: None,
                persisted_key: None,
                forked_from: None,
                pending: None,
                recent: Default::default(),
                spec_off: false,
            },
        );
        self.feed_report(&format!(
            "prefilled {} tokens for seq {seq_id}",
            tokens.len()
        ));
        Ok((next, tokens.len()))
    }

    fn decode_step(
        &mut self,
        seq_id: u64,
        last_token: u32,
        _position: usize,
    ) -> Result<(u32, usize)> {
        if self.spec.enabled {
            return self.decode_step_spec_as(seq_id, last_token, &mut |l| argmax_host(l), true);
        }
        self.mem_note_decode(seq_id);
        let model = &self.model;
        let seq = self
            .seqs
            .get_mut(&seq_id)
            .ok_or_else(|| anyhow!("qwen4_exp runner: decode_step on unknown seq_id {seq_id}"))?;
        seq.forked_from = None;
        let logits = model.forward(&[last_token], &mut seq.cache, true)?;
        let next = Self::argmax(&logits)?;
        seq.position += 1;
        Ok((next, seq.position))
    }

    fn remove_seq(&mut self, seq_id: u64) -> Result<()> {
        self.seqs.remove(&seq_id);
        let _ = crate::metal_memory::clear_cache();
        self.mem_report(&format!("dropped seq {seq_id}"));
        Ok(())
    }

    fn extend(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
        // Feed a clone: a failed forward leaves the sequence exactly as it was.
        self.mem_begin();
        if let Some((next, position, _)) = self.take_prefed(seq_id, tokens, None) {
            return Ok((next, position));
        }
        let have = self.seq(seq_id, "extend")?.position;
        self.check_context(have, tokens.len())?;
        let seq = self.seq(seq_id, "extend")?;
        seq.pending = None;
        let mut cache = seq.cache.clone();
        let logits = self.forward_chunked(tokens, &mut cache)?;
        let next = Self::argmax(&logits)?;
        let seq = self.seq(seq_id, "extend")?;
        seq.cache = cache;
        seq.position += tokens.len();
        let position = seq.position;
        self.feed_report(&format!("extended seq {seq_id} by {} tokens", tokens.len()));
        Ok((next, position))
    }

    fn forward_probe(&mut self, _seq_id: u64, _tokens: &[u32]) -> Result<ProbeRows> {
        bail!("qwen4_exp runner has no forward probe")
    }

    fn snapshot_state(&mut self, seq_id: u64) -> Result<u64> {
        Ok(self.snapshot_state_deep(seq_id)?.0)
    }

    fn restore_state(&mut self, seq_id: u64, snapshot_id: u64) -> Result<usize> {
        let (cache, position) = self
            .snapshots
            .get(&snapshot_id)
            .cloned()
            .ok_or_else(|| anyhow!("qwen4_exp runner: unknown snapshot {snapshot_id}"))?;
        let seq = self.seq(seq_id, "restore_state")?;
        seq.pending = None;
        seq.cache = cache;
        seq.position = position;
        Ok(position)
    }

    fn release_snapshot(&mut self, snapshot_id: u64) -> Result<()> {
        self.snapshots.remove(&snapshot_id);
        Ok(())
    }

    fn snapshot_state_deep(&mut self, seq_id: u64) -> Result<(u64, usize)> {
        let seq = self.seq(seq_id, "snapshot_state_deep")?;
        let copy = (seq.cache.clone(), seq.position);
        let position = copy.1;
        let id = self.next_snapshot;
        self.next_snapshot += 1;
        self.snapshots.insert(id, copy);
        Ok((id, position))
    }

    fn fork_from_snapshot(&mut self, snapshot_id: u64, dst_seq_id: u64) -> Result<usize> {
        let (cache, position) = self
            .snapshots
            .get(&snapshot_id)
            .cloned()
            .ok_or_else(|| anyhow!("qwen4_exp runner: unknown snapshot {snapshot_id}"))?;
        self.seqs.insert(
            dst_seq_id,
            Seq {
                cache,
                position,
                rollback: None,
                persisted_key: None,
                forked_from: Some(snapshot_id),
                pending: None,
                recent: Default::default(),
                spec_off: false,
            },
        );
        Ok(position)
    }

    fn prefill_grid(&self, _n: usize) -> Option<PrefillGrid> {
        Some(PrefillGrid { chunk: self.chunk })
    }

    fn mark_rollback(&mut self, seq_id: u64) -> Result<()> {
        let seq = self.seq(seq_id, "mark_rollback")?;
        seq.rollback = Some((seq.cache.clone(), seq.position));
        Ok(())
    }

    fn rollback(&mut self, seq_id: u64) -> Result<usize> {
        let seq = self.seq(seq_id, "rollback")?;
        seq.pending = None;
        let (cache, position) = seq
            .rollback
            .clone()
            .ok_or_else(|| anyhow!("qwen4_exp runner: rollback without a mark on seq {seq_id}"))?;
        seq.cache = cache;
        seq.position = position;
        Ok(position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_draft_head_is_off_unless_asked_for() {
        let mode = |raw: Option<&str>| parse_mtp_mode(raw).unwrap();
        for off in [
            None,
            Some(""),
            Some("off"),
            Some("0"),
            Some("false"),
            Some(" off "),
        ] {
            assert_eq!(mode(off), MtpMode::Off, "{off:?}");
        }
        assert_eq!(mode(Some("auto")), MtpMode::Auto);
        for on in [Some("on"), Some("1"), Some("true")] {
            assert_eq!(mode(on), MtpMode::On, "{on:?}");
        }
        assert!(parse_mtp_mode(Some("maybe")).is_err());
    }

    #[test]
    fn a_pool_size_is_a_number_with_an_optional_unit() {
        let gb = |s: &str| parse_pool_size(s).unwrap();
        assert_eq!(gb("auto"), None);
        assert_eq!(gb(""), None);
        assert_eq!(gb("  AUTO "), None);
        assert_eq!(gb("8"), Some(8_000_000_000));
        assert_eq!(gb("8g"), Some(8_000_000_000));
        assert_eq!(gb("8GB"), Some(8_000_000_000));
        assert_eq!(gb("12.5 g"), Some(12_500_000_000));
        assert_eq!(gb("512m"), Some(512_000_000));
        assert_eq!(gb("512 MB"), Some(512_000_000));
        for bad in ["0", "-4g", "eight", "4 terabytes", "g", "NaN", "inf"] {
            assert!(parse_pool_size(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn a_pool_that_does_not_fit_is_lowered_to_what_does() {
        let cfg = real_config();
        let floor = pool_floor_bytes(&cfg);
        let usable = 14_100_000_000;
        let pool = |requested: u64, plan: &MemoryPlan| {
            choose_pool(usable, None, plan, Some(requested), floor).pool
        };
        // The 24 GB-class plan: ~6.9 GB left after the trunk and the reserve.
        let plan = plan_memory(usable, &cfg, None, None);
        let fits = plan.pool;
        assert_eq!(pool(4_000_000_000, &plan), 4_000_000_000);
        assert_eq!(pool(fits, &plan), fits);
        // Round numbers a little over the plan are taken as written.
        assert_eq!(pool(fits + 400_000_000, &plan), fits + 400_000_000);
        // 12 GB on a 24 GB Mac is not.
        let lowered = choose_pool(usable, None, &plan, Some(12_000_000_000), floor);
        assert_eq!(lowered.pool, fits);
        assert_eq!(lowered.lowered_from, Some(12_000_000_000));
        assert_eq!(lowered.usable, usable, "the device's plan, as before");
        let line = plan_line(usable, &lowered, &plan, false);
        assert!(
            line.contains("A 12.0 GB expert pool does not fit"),
            "{line}"
        );
        assert!(line.contains("this GPU may use"), "{line}");
        // A shorter longest prompt leaves more room for the same request.
        let short = plan_memory(usable, &cfg, None, Some(4_096));
        assert!(pool(12_000_000_000, &short) > fits);
    }

    /// The 36 GB M3 Max: 30.2 GB working set x 0.88.
    const DEVICE_36GB: u64 = 26_600_000_000;

    /// The page counts `vm_stat` printed on that Mac (16 KB pages): an editor, a browser, chat
    /// apps open and an idle model server holding 10.7 GB in the compressor, then the same
    /// after that server was stopped.
    fn busy_36gb() -> HostMemory {
        HostMemory {
            ram: 38_654_705_664,
            free: 92_438_528,
            file_backed: 8_125_726_720,
            purgeable: 42_876_928,
            anonymous: 11_263_918_080,
            wired: 4_838_866_944,
            compressor: 13_398_736_896,
            own: 0,
        }
    }

    fn after_36gb() -> HostMemory {
        HostMemory {
            free: 10_487_791_616,
            file_backed: 6_905_397_248,
            purgeable: 313_901_056,
            anonymous: 14_393_376_768,
            wired: 3_508_453_376,
            compressor: 2_424_668_160,
            ..busy_36gb()
        }
    }

    #[test]
    fn the_headroom_is_two_gb_or_five_percent_of_ram() {
        assert_eq!(headroom(16 << 30), 2_000_000_000);
        assert_eq!(headroom(24 << 30), 2_000_000_000);
        // 5% of 36 GiB is 1.93 GB: still the minimum.
        assert_eq!(headroom(38_654_705_664), 2_000_000_000);
        // 5% of 128 GiB.
        assert_eq!(headroom(128 << 30), 6_871_947_673);
    }

    #[test]
    fn free_now_is_free_file_backed_and_purgeable_pages_plus_our_own_less_the_headroom() {
        let h = after_36gb();
        assert_eq!(h.reclaimable(), 17_707_089_920);
        assert_eq!(free_now(&h), 17_707_089_920 - 2_000_000_000);
        // What this process already holds is the plan's to reuse.
        let mine = HostMemory {
            own: 2_000_000_000,
            ..h
        };
        assert_eq!(free_now(&mine), free_now(&h) + 2_000_000_000);
        // Less free than the headroom is nothing, not an underflow.
        let full = HostMemory {
            free: 0,
            file_backed: 1_000_000_000,
            purgeable: 0,
            ..h
        };
        assert_eq!(free_now(&full), 0);
        // Each part is in the log line.
        let line = free_line(&h);
        for part in [
            "free now: 17.7 GB",
            "free 10.5",
            "file-backed 6.9",
            "purgeable 0.3",
            "other apps hold 20.9 GB",
            "this process 0.0",
            "headroom 2.0",
            "= 15.7 GB for the plan",
        ] {
            assert!(line.contains(part), "{part:?} not in {line}");
        }
    }

    #[test]
    fn an_idle_machine_plans_from_the_working_set_as_before() {
        let cfg = real_config();
        let plan = plan_memory(DEVICE_36GB, &cfg, None, None);
        let idle = HostMemory {
            free: 25_000_000_000,
            ..after_36gb()
        };
        let choice = choose_pool(
            DEVICE_36GB,
            Some(free_now(&idle)),
            &plan,
            None,
            pool_floor_bytes(&cfg),
        );
        let before = choose_pool(DEVICE_36GB, None, &plan, None, pool_floor_bytes(&cfg));
        assert_eq!(choice, before);
        assert_eq!(
            (choice.pool, choice.usable, choice.clamped_by),
            (plan.pool, DEVICE_36GB, 0)
        );
        assert_eq!(
            plan_line(DEVICE_36GB, &choice, &plan, false),
            format!(
                "[qwen4_exp] GPU may use 26.6 GB: expert pool {:.2} GB, prefill chunk 1024 \
                 tokens, prompts up to 16384 tokens (LUMEN_MOE_EXPERT_CACHE_GB / \
                 LUMEN_QWEN35_PREFILL_CHUNK / LUMEN_MAX_PROMPT_TOKENS override)",
                plan.pool as f64 / 1e9
            )
        );
    }

    #[test]
    fn what_other_apps_hold_comes_out_of_the_pool_and_the_line_says_how_much() {
        f32_kv(|| {
            let cfg = real_config();
            let plan = plan_memory(DEVICE_36GB, &cfg, None, None);
            let free = free_now(&after_36gb());
            let choice = choose_pool(DEVICE_36GB, Some(free), &plan, None, pool_floor_bytes(&cfg));
            assert_eq!(choice.usable, free);
            assert_eq!(choice.clamped_by, DEVICE_36GB - free);
            assert_eq!(choice.pool, free - TRUNK_BYTES - plan.reserve);
            assert_eq!(plan.pool - choice.pool, choice.clamped_by);
            // 19.5 GB of pool on an idle 36 GB Mac, 8.6 GB with these apps open.
            assert!(
                (8_500_000_000..8_700_000_000).contains(&choice.pool),
                "{choice:?}"
            );
            assert!(!choice.at_floor);
            // The chunk and the context are the machine's: only the pool shrinks.
            assert_eq!((plan.chunk, plan.ctx_tokens), (1024, 16_384));
            let line = plan_line(DEVICE_36GB, &choice, &plan, false);
            assert!(
                line.contains("GPU may use 26.6 GB, 15.7 GB of it free now"),
                "{line}"
            );
            assert!(
                line.ends_with(
                    "Pool 8.6 GB (other apps hold memory: 10.9 GB below the working set); close \
                 apps and restart for a bigger pool"
                ),
                "{line}"
            );
        })
    }

    #[test]
    fn an_explicit_pool_is_lowered_to_what_is_free_now() {
        let cfg = real_config();
        let plan = plan_memory(DEVICE_36GB, &cfg, None, None);
        let free = free_now(&after_36gb());
        let floor = pool_floor_bytes(&cfg);
        // 15 GB fits the working set, not what other apps leave.
        let choice = choose_pool(DEVICE_36GB, Some(free), &plan, Some(15_000_000_000), floor);
        assert_eq!(choice.pool, free - TRUNK_BYTES - plan.reserve);
        assert_eq!(choice.lowered_from, Some(15_000_000_000));
        let line = plan_line(DEVICE_36GB, &choice, &plan, false);
        assert!(
            line.contains("A 15.0 GB expert pool does not fit"),
            "{line}"
        );
        assert!(line.contains("of the 15.7 GB other apps leave"), "{line}");
        assert!(line.contains("10.9 GB below the working set"), "{line}");
        // A size that fits is taken as written; the memory limit is still what is free now.
        let small = choose_pool(DEVICE_36GB, Some(free), &plan, Some(6_700_000_000), floor);
        assert_eq!((small.pool, small.usable), (6_700_000_000, free));
        assert_eq!(small.lowered_from, None);
    }

    #[test]
    fn the_opt_out_plans_from_the_working_set_alone() {
        plan_reclaimable::with(false, || assert_eq!(host_for_plan(DEVICE_36GB), None));
        #[cfg(target_os = "macos")]
        plan_reclaimable::with(true, || assert!(host_for_plan(1).is_some()));
        // `None` is what the opt-out hands the plan: today's pool, with or without a request.
        let cfg = real_config();
        let plan = plan_memory(DEVICE_36GB, &cfg, None, None);
        let floor = pool_floor_bytes(&cfg);
        let off = choose_pool(DEVICE_36GB, None, &plan, None, floor);
        assert_eq!(
            (off.pool, off.usable, off.clamped_by),
            (plan.pool, DEVICE_36GB, 0)
        );
        let asked = choose_pool(DEVICE_36GB, None, &plan, Some(15_000_000_000), floor);
        assert_eq!(asked.pool, 15_000_000_000);
    }

    #[test]
    fn a_reading_that_is_still_growing_is_taken_again_until_it_settles() {
        let at = |reclaimable_gb: f64| HostMemory {
            free: (reclaimable_gb * 1e9) as u64,
            file_backed: 0,
            purgeable: 0,
            ..after_36gb()
        };
        let run = |readings: &[f64]| {
            let mut left = readings
                .iter()
                .map(|&g| at(g))
                .collect::<Vec<_>>()
                .into_iter();
            let mut sleeps = 0;
            let got = settled(|| left.next(), |_| sleeps += 1, DEVICE_36GB).unwrap();
            (got.free as f64 / 1e9, sleeps)
        };
        // A server that has just exited: its memory comes back a second later.
        assert_eq!(run(&[2.8, 26.0, 25.9]), (25.9, 2));
        // Memory other apps hold stays held: one more look, then the plan.
        assert_eq!(run(&[8.3, 8.4]), (8.4, 1));
        // More free than the GPU may use: nothing to wait for.
        assert_eq!(run(&[40.0]), (40.0, 0));
        // Still growing at the last look: the last reading, after at most 3 s.
        let rising: Vec<f64> = (0..10).map(|i| 2.0 + i as f64).collect();
        assert_eq!(run(&rising), (8.0, SETTLE_READS));
        // A reading that fails ends the wait with what was read.
        assert_eq!(run(&[3.0]), (3.0, 1));
        assert!(settled(|| None, |_| {}, DEVICE_36GB).is_none());
    }

    #[test]
    fn when_not_even_the_smallest_pool_fits_the_floor_is_used_and_the_line_says_so() {
        let cfg = real_config();
        let floor = pool_floor_bytes(&cfg);
        assert_eq!(floor, 2 * 512 * 2_764_800, "1,024 slots of 2.76 MB");
        let plan = plan_memory(DEVICE_36GB, &cfg, None, None);
        // The busy machine: 6.3 GB free now, less than the trunk and the reserve alone.
        let free = free_now(&busy_36gb());
        let choice = choose_pool(DEVICE_36GB, Some(free), &plan, None, floor);
        assert!(choice.at_floor);
        assert_eq!(choice.pool, floor);
        // The plan is for what the floor holds: MLX gets that, not the 6.3 GB.
        assert_eq!(choice.usable, TRUNK_BYTES + floor + plan.reserve);
        assert_eq!(choice.clamped_by, DEVICE_36GB - free);
        let line = plan_line(DEVICE_36GB, &choice, &plan, false);
        assert!(
            line.contains("Pool 2.83 GB, the smallest the engine runs"),
            "{line}"
        );
        assert!(line.contains("20.3 GB below the working set"), "{line}");
        assert!(line.contains("expect swapping"), "{line}");
        // A Mac too small for the floor even when idle: the floor, and the plan stays the
        // device's (as before).
        let tiny = plan_memory(4_000_000_000, &cfg, None, None);
        let tight = choose_pool(4_000_000_000, None, &tiny, None, floor);
        assert_eq!(
            (tight.pool, tight.usable, tight.at_floor),
            (floor, 4_000_000_000, true)
        );
        let line = plan_line(4_000_000_000, &tight, &tiny, false);
        assert!(
            line.contains("the trunk and the prompt's reserve leave less"),
            "{line}"
        );
        assert!(!line.contains("other apps"), "{line}");
    }

    fn real_config() -> Qwen4ExpConfig {
        // Flash-Next's shapes: 48 layers, every fourth full attention, 2 KV heads of 256, an
        // indexer key of 128, 24 heads.
        let mut cfg = Qwen4ExpConfig::load(
            &std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/qwen4exp_tiny"),
        )
        .unwrap();
        cfg.layer_kinds = (0..48)
            .map(|i| {
                if i % 4 == 3 {
                    super::super::config::LayerKind::FullAttention
                } else {
                    super::super::config::LayerKind::LinearAttention
                }
            })
            .collect();
        cfg.num_key_value_heads = 2;
        cfg.head_dim = 256;
        cfg.indexer_head_dim = 128;
        cfg.num_attention_heads = 24;
        cfg.linear_num_key_heads = 16;
        cfg.linear_num_value_heads = 48;
        cfg.linear_key_head_dim = 128;
        cfg.linear_value_head_dim = 128;
        cfg.linear_conv_kernel_dim = 4;
        // 512 routed experts of 2,560 x 640, 4-bit in groups of 64.
        cfg.num_experts = 512;
        cfg.hidden_size = 2_560;
        cfg.moe_intermediate_size = 640;
        cfg.quant.set(
            "language_model.model.layers.0.mlp.switch_mlp",
            super::super::config::QuantSpec {
                group_size: 64,
                bits: 4,
            },
        );
        cfg
    }

    #[test]
    fn the_pool_is_what_the_working_set_leaves_after_the_trunk_and_the_reserve() {
        f32_kv(|| {
            let cfg = real_config();
            assert_eq!(
                cfg.kv_bytes_per_token(),
                13 * (2 * 2 * 256 * 4 + 128 * 4) as u64
            );
            // 36 GB Mac: 30.2 GB working set x 0.88.
            let plan = plan_memory(26_600_000_000, &cfg, None, None);
            assert_eq!((plan.chunk, plan.ctx_tokens), (1024, 16_384));
            assert!(
                (19_000_000_000..20_100_000_000).contains(&plan.pool),
                "{plan:?}"
            );
            // 24 GB Mac: ~16 GB working set x 0.88.
            let plan = plan_memory(14_100_000_000, &cfg, None, None);
            assert_eq!(plan.chunk, 1024);
            assert!(plan.pool > 6_500_000_000, "{plan:?}");
            // A 16 GB Mac takes the tight settings and still has a pool that holds a staging region
            // and a layer (2 x 512 experts x 2.76 MB).
            let plan = plan_memory(9_400_000_000, &cfg, None, None);
            assert_eq!(
                (plan.chunk, plan.ctx_tokens, plan.scores_budget),
                (512, 8_192, 800_000_000)
            );
            assert!(plan.pool > 2_850_000_000, "{plan:?}");
            // Too little memory: the pool is zero and the slot floor takes over.
            assert_eq!(plan_memory(4_000_000_000, &cfg, None, None).pool, 0);
        })
    }

    #[test]
    fn a_longer_context_or_a_bigger_chunk_takes_its_memory_out_of_the_pool() {
        f32_kv(|| {
            let cfg = real_config();
            let base = plan_memory(26_600_000_000, &cfg, None, None);
            let longer = plan_memory(26_600_000_000, &cfg, None, Some(65_536));
            // 48K more tokens of 60 KB each: ~2.9 GB of KV, and the scores stay at their budget.
            assert!(
                base.pool - longer.pool > 2_800_000_000,
                "{base:?} {longer:?}"
            );
            assert!(base.pool - longer.pool < 4_200_000_000);
            let bigger = plan_memory(26_600_000_000, &cfg, Some(4096), None);
            assert!(bigger.pool < base.pool);
        })
    }

    /// A 15K-token session holds its KV twice (the sequence's and its rollback point's) and a new
    /// sequence its DeltaNet state twice; measured on Flash-Next: 0.95 GB of state and 0.91 GB of
    /// rollback point, and a 13-token session 0.13 GB.
    #[test]
    fn a_session_feed_leaves_its_kv_and_a_new_sequence_s_state_twice() {
        f32_kv(|| {
            let cfg = real_config();
            let (kv, states) = crate::session::feed_cost(0, 14_978);
            let fresh = feed_growth(&cfg, kv, states);
            assert!(
                (1_850_000_000..2_150_000_000).contains(&fresh),
                "{fresh} bytes for a 15K-token prompt"
            );
            // 36 layers x (48 x 128 x 128 f32 + a 3-wide window of 10,240 channels) = ~118 MB.
            assert!((115_000_000..120_000_000).contains(&cfg.state_bytes()));
            // Continuing a session adds the new tokens' KV twice, not another state.
            let (kv, states) = crate::session::feed_cost(14_978, 15_478);
            assert_eq!(
                feed_growth(&cfg, kv, states),
                2 * 500 * cfg.kv_bytes_per_token()
            );
        })
    }

    /// Measured in the soak (16 GB plan): with the cache's room asked of it, a 23-token chat turn
    /// dropped a 5.8K-token session. A short turn is not made to leave that room; one that
    /// would take the live memory past the budget itself still is.
    #[test]
    fn a_small_feed_only_has_to_stay_inside_the_budget() {
        f32_kv(|| {
            let cfg = real_config();
            let budget = 13_960_000_000;
            let (kv, states) = crate::session::feed_cost(0, 23);
            let small = feed_growth(&cfg, kv, states);
            assert!(small < SMALL_FEED, "{small}");
            assert!(13_300_000_000 + small + CACHE_ROOM > budget);
            assert_eq!(shortfall(13_300_000_000, small, budget), 0);
            assert!(shortfall(13_800_000_000, small, budget) > 0);
            // A 4.5K-token continuation is the smallest that asks for the room.
            let cont = |fed: usize| {
                let (kv, states) = crate::session::feed_cost(14_978, 14_978 + fed);
                feed_growth(&cfg, kv, states)
            };
            assert!(cont(4_600) >= SMALL_FEED);
            assert!(cont(4_000) < SMALL_FEED);
        })
    }

    #[test]
    fn a_runner_without_a_budget_never_falls_short_and_a_tiny_one_always_does() {
        let mut r = tiny_runner(16);
        assert_eq!(r.memory_shortfall(200, 2), 0);
        r.memory_budget = 1;
        assert!(r.memory_shortfall(200, 2) > 0);
        // A prompt past the context is not counted past it (it is refused, not fed).
        r.ctx_tokens = 10;
        r.memory_budget = u64::MAX;
        assert_eq!(r.memory_shortfall(1_000_000_000, 2), 0);
    }

    /// `f` with the KV cache in f32 (`LUMEN_QWEN4_KV_BF16=0`), for the plan's arithmetic, which
    /// these tests check against figures measured with an f32 cache.
    fn f32_kv<R>(f: impl FnOnce() -> R) -> R {
        super::super::config::kv_bf16::with(false, f)
    }

    fn tiny_runner(chunk: usize) -> Qwen4ExpRunner {
        tiny_runner_with(chunk, MtpMode::Off, None)
    }

    fn tiny_runner_with(chunk: usize, mtp: MtpMode, budget: Option<u64>) -> Qwen4ExpRunner {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/qwen4exp_tiny");
        let model = Qwen4ExpModel::load_with(
            &dir,
            LoadOptions {
                expert_budget_bytes: budget,
                ngram: NgramSource::Auto,
                mtp,
                scores_budget_bytes: 0,
            },
        )
        .expect("tiny replica");
        let spec = if model.has_mtp() {
            Speculation {
                enabled: true,
                floor: 0.0,
                ..Speculation::off()
            }
        } else {
            Speculation::off()
        };
        Qwen4ExpRunner {
            model,
            seqs: HashMap::new(),
            snapshots: HashMap::new(),
            next_snapshot: 0,
            chunk,
            ctx_tokens: 1_000_000,
            memory_budget: u64::MAX,
            disk: None,
            fingerprint: String::new(),
            spec,
            prefed: HashMap::new(),
            forced_draft: None,
        }
    }

    fn decode_some(r: &mut Qwen4ExpRunner, seq: u64, mut last: u32, steps: usize) -> Vec<u32> {
        let mut out = vec![last];
        for _ in 0..steps {
            let (next, _) = r.decode_step(seq, last, 0).unwrap();
            out.push(next);
            last = next;
        }
        out
    }

    /// The whole-prompt marked feed is the cut-by-cut feed the session would otherwise do:
    /// same next token and position, and the same state at the rollback point and at the end.
    #[test]
    fn a_marked_feed_in_one_call_equals_feeding_cut_by_cut() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/qwen4exp_tiny/reference/outputs.safetensors");
        let want = Array::load_safetensors(dir).expect("reference outputs");
        let ids: Vec<u32> = want["ids"]
            .as_slice::<i32>()
            .iter()
            .map(|&t| t as u32)
            .collect();
        assert!(ids.len() >= 48);
        let ids = &ids[..48];

        // Cold: chunks of 16, rollback point at 32.
        let (mut one, mut cuts) = (tiny_runner(16), tiny_runner(16));
        let a = one
            .feed_with_mark(1, true, ids, 32)
            .unwrap()
            .expect("supported");
        cuts.prefill(1, &ids[..16]).unwrap();
        cuts.extend(1, &ids[16..32]).unwrap();
        cuts.mark_rollback(1).unwrap();
        let b = cuts.extend(1, &ids[32..]).unwrap();
        assert_eq!(a, b);
        assert_eq!(
            decode_some(&mut one, 1, a.0, 5),
            decode_some(&mut cuts, 1, b.0, 5)
        );
        assert_eq!(one.rollback(1).unwrap(), 32);
        assert_eq!(cuts.rollback(1).unwrap(), 32);
        let first = ids[32];
        assert_eq!(
            decode_some(&mut one, 1, first, 5),
            decode_some(&mut cuts, 1, first, 5)
        );

        // Continuing a sequence that stands at 20 (off the grid): the pieces end on the grid.
        let (mut one, mut cuts) = (tiny_runner(16), tiny_runner(16));
        for r in [&mut one, &mut cuts] {
            r.prefill(2, &ids[..20]).unwrap();
        }
        let a = one
            .feed_with_mark(2, false, &ids[20..], 12)
            .unwrap()
            .expect("supported");
        cuts.extend(2, &ids[20..32]).unwrap();
        cuts.mark_rollback(2).unwrap();
        let b = cuts.extend(2, &ids[32..]).unwrap();
        assert_eq!(a, b);
        assert_eq!(one.rollback(2).unwrap(), 32);
        assert_eq!(cuts.rollback(2).unwrap(), 32);
        assert_eq!(
            decode_some(&mut one, 2, ids[32], 5),
            decode_some(&mut cuts, 2, ids[32], 5)
        );

        // A mark that is not on the grid is declined, not guessed at.
        let mut r = tiny_runner(16);
        assert!(r.feed_with_mark(3, true, ids, 30).unwrap().is_none());
    }

    #[test]
    fn chunks_end_on_the_absolute_grid() {
        assert_eq!(chunk_cuts(0, 40, 16), vec![16, 32, 40]);
        assert_eq!(chunk_cuts(20, 28, 16), vec![12, 28]);
        assert_eq!(chunk_cuts(16, 16, 16), vec![16]);
        assert_eq!(chunk_cuts(0, 5, 16), vec![5]);
    }

    fn fixture_prompt() -> Vec<u32> {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/qwen4exp_tiny/reference/outputs.safetensors");
        let want = Array::load_safetensors(dir).expect("reference outputs");
        want["ids"].as_slice::<i32>()[..30]
            .iter()
            .map(|&t| t as u32)
            .collect()
    }

    /// Rounds with drafts that are alternately right and wrong emit the tokens a plain loop
    /// emits, return the same positions, and leave the sequence in the state a plain loop
    /// would at every point (the one the caller can stop, snapshot or roll back at).
    #[test]
    fn speculative_rounds_emit_what_plain_decoding_emits() {
        // Experts resident, and the trunk's and the head's served from one pool.
        for budget in [None, Some(1u64)] {
            speculative_rounds_match_plain(budget);
        }
    }

    fn speculative_rounds_match_plain(budget: Option<u64>) {
        let prompt = fixture_prompt();
        let steps = 24;

        let mut plain = tiny_runner_with(16, MtpMode::Off, budget);
        let (t0, _) = plain.prefill(1, &prompt).unwrap();
        let mut expected = vec![t0];
        let mut last = t0;
        for _ in 0..steps {
            let (next, _) = plain.decode_step(1, last, 0).unwrap();
            expected.push(next);
            last = next;
        }

        let mut spec = tiny_runner_with(16, MtpMode::On, budget);
        let (s0, _) = spec.prefill(1, &prompt).unwrap();
        assert_eq!(s0, t0);
        // The draft of each round is the token the plain loop emits next, except every third
        // round, where it is a different token.
        let emitted = std::rc::Rc::new(std::cell::Cell::new(1usize));
        let rounds = std::rc::Rc::new(std::cell::Cell::new(0usize));
        let (e, r, ex) = (emitted.clone(), rounds.clone(), expected.clone());
        spec.forced_draft = Some(Box::new(move || {
            let right = ex[e.get()];
            r.set(r.get() + 1);
            if r.get() % 3 == 0 {
                (right + 1) % 128
            } else {
                right
            }
        }));
        let mut got = vec![s0];
        let mut last = s0;
        let mut n_accepted = 0;
        while got.len() <= steps {
            let (next, position) = spec.decode_step(1, last, 0).unwrap();
            let now = spec.seqs[&1].pending.is_some();
            if now {
                n_accepted += 1;
            }
            got.push(next);
            last = next;
            emitted.set(got.len());
            // A buffered token is served without a round.
            let calls_so_far = got.len() - 1;
            assert_eq!(
                position,
                prompt.len() + calls_so_far,
                "position after call {calls_so_far}"
            );
            // What the sequence holds is what a plain loop would hold after the same calls.
            let seq = &spec.seqs[&1];
            assert_eq!(
                seq.cache.len(),
                prompt.len() + calls_so_far,
                "cache after call {calls_so_far}"
            );
        }
        assert!(
            n_accepted >= 4,
            "only {n_accepted} accepted rounds exercised"
        );
        assert_eq!(got, expected, "speculative decoding changed the tokens");
    }

    /// A buffered token is dropped when the caller does anything else with the sequence, and
    /// the next call is the plain step it asked for.
    #[test]
    fn a_buffered_token_does_not_survive_a_rollback_or_a_different_token() {
        let prompt = fixture_prompt();
        let mut r = tiny_runner_with(16, MtpMode::On, None);
        let (t0, _) = r.prefill(1, &prompt).unwrap();
        // Force an accepted round: ask the plain runner what the successor is.
        let mut plain = tiny_runner(16);
        plain.prefill(1, &prompt).unwrap();
        let (t1, _) = plain.decode_step(1, t0, 0).unwrap();
        r.forced_draft = Some(Box::new(move || t1));
        r.mark_rollback(1).unwrap();
        let (a, _) = r.decode_step(1, t0, 0).unwrap();
        assert_eq!(a, t1);
        assert!(
            r.seqs[&1].pending.is_some(),
            "the draft matched: a token should be buffered"
        );
        // Rolling back drops it.
        r.rollback(1).unwrap();
        assert!(r.seqs[&1].pending.is_none());
        assert_eq!(r.seqs[&1].cache.len(), prompt.len());
        // A different `last_token` than the buffer was made for takes the ordinary path.
        r.forced_draft = Some(Box::new(move || t1));
        r.decode_step(1, t0, 0).unwrap();
        assert!(r.seqs[&1].pending.is_some());
        let other = (t1 + 5) % 128;
        let (_, position) = r.decode_step(1, other, 0).unwrap();
        assert_eq!(position, prompt.len() + 2);
        assert!(r.seqs[&1].pending.is_none() || r.seqs[&1].pending.as_ref().unwrap().after != t1);
    }

    // ───────────── the prefix cache's head: from memory, from disk, after a drop ─────────────

    fn fixture_ids() -> Vec<u32> {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/qwen4exp_tiny/reference/outputs.safetensors");
        let want = Array::load_safetensors(dir).expect("reference outputs");
        want["ids"]
            .as_slice::<i32>()
            .iter()
            .map(|&t| t as u32)
            .collect()
    }

    /// The bytes of every array of `seq`'s state.
    fn state_bytes(r: &Qwen4ExpRunner, seq: u64) -> Vec<Vec<u8>> {
        let (_, records) = r.seqs[&seq].cache.to_records().expect("readback");
        records.into_iter().map(|rec| rec.data).collect()
    }

    /// A tools prompt fed as its cold MISS was before the one-pass split: a prefill to the head,
    /// then an extend of the rest. `(next token, state bytes, six greedy steps)`.
    fn cold_in_two_calls(
        budget: Option<u64>,
        prompt: &[u32],
        at: usize,
    ) -> (u32, Vec<Vec<u8>>, Vec<u32>) {
        cold_in_two_calls_with(MtpMode::Off, budget, prompt, at)
    }

    /// [`cold_in_two_calls`] with or without the draft head (whose state the bytes include).
    fn cold_in_two_calls_with(
        mtp: MtpMode,
        budget: Option<u64>,
        prompt: &[u32],
        at: usize,
    ) -> (u32, Vec<Vec<u8>>, Vec<u32>) {
        let mut r = tiny_runner_with(16, mtp, budget);
        let (next, _, _) =
            crate::prefix_cache::prefill_split_in_two_calls(&mut r, 1, prompt, at, false).unwrap();
        let state = state_bytes(&r, 1);
        (next, state, decode_some(&mut r, 1, next, 6))
    }

    fn assert_continues_like(
        r: &mut Qwen4ExpRunner,
        seq: u64,
        next: u32,
        cold: &(u32, Vec<Vec<u8>>, Vec<u32>),
        what: &str,
    ) {
        assert_eq!(next, cold.0, "{what}: next token");
        assert!(
            state_bytes(r, seq) == cold.1,
            "{what}: the state differs from the cold one"
        );
        assert_eq!(decode_some(r, seq, next, 6), cold.2, "{what}: greedy steps");
        Runner::remove_seq(r, seq).unwrap();
    }

    /// The head sits at the system/tools boundary (21, off the 16-token grid on purpose: it is a
    /// cut the cold MISS makes, not a grid point). Forked from memory, reloaded from disk after the
    /// copy in memory is gone, or restored after a restart, it continues exactly as the prompt's
    /// cold MISS: same next token, same state byte for byte, same greedy steps. The cold MISS
    /// itself, now one pass, is the two calls it replaces. Resident experts and a pool.
    #[test]
    fn a_tools_head_from_memory_or_disk_continues_exactly_like_a_cold_prompt() {
        for budget in [None, Some(1u64)] {
            head_reuse_is_exact(budget);
        }
    }

    fn head_reuse_is_exact(budget: Option<u64>) {
        use crate::prefix_cache::PrefixCacheStore;
        let ids = fixture_ids();
        assert!(ids.len() >= 48);
        let at = 21;
        let a: Vec<u32> = ids[..40].to_vec();
        let b: Vec<u32> = ids[..at].iter().chain(&ids[30..48]).copied().collect();
        let (cold_a, cold_b) = (
            cold_in_two_calls(budget, &a, at),
            cold_in_two_calls(budget, &b, at),
        );

        let dir = tempfile::tempdir().unwrap();
        let mut r = tiny_runner_with(16, MtpMode::Off, budget);
        r.set_disk_root_for_test(Some(dir.path()), "tiny");
        let mut store = PrefixCacheStore::with_incremental(true, 4, true);
        let (next, _) = store
            .prefill_optionally_cached(&mut r, 1, &a, Some("k"), Some(at))
            .unwrap();
        assert_continues_like(&mut r, 1, next, &cold_a, "cold MISS in one pass");
        store.flush_persist(&mut r);
        assert_eq!(
            r.disk.as_ref().map(DiskKvStore::len),
            Some(1),
            "the head, not the prompt"
        );

        let (next, _) = store
            .prefill_optionally_cached(&mut r, 2, &b, Some("k"), Some(at))
            .unwrap();
        assert_eq!(
            store.branches_for_test("k")[0],
            (at, 1, true),
            "a HIT on the head"
        );
        assert_continues_like(&mut r, 2, next, &cold_b, "head from memory");

        // The copy in memory is gone (what the memory plan does to a head): reloaded from disk.
        store.clear(&mut r);
        let (next, _) = store
            .prefill_optionally_cached(&mut r, 3, &b, Some("k"), Some(at))
            .unwrap();
        assert_eq!(
            store.branches_for_test("k")[0],
            (at, 1, true),
            "a HIT on the reloaded head"
        );
        assert_continues_like(&mut r, 3, next, &cold_b, "head reloaded after a drop");

        // A restart: a new runner, an empty store, the same disk.
        let mut fresh = tiny_runner_with(16, MtpMode::Off, budget);
        fresh.set_disk_root_for_test(Some(dir.path()), "tiny");
        let mut store = PrefixCacheStore::with_incremental(true, 4, true);
        let (next, _) = store
            .prefill_optionally_cached(&mut fresh, 1, &b, Some("k"), Some(at))
            .unwrap();
        assert_eq!(
            store.branches_for_test("k")[0],
            (at, 1, true),
            "a HIT on the restored head"
        );
        assert_continues_like(
            &mut fresh,
            1,
            next,
            &cold_b,
            "head restored after a restart",
        );
    }

    /// `LUMEN_PREFIX_EXACT_REUSE`: the rounds of an agent's conversation through the prefix cache
    /// (each prompt the previous one plus more tokens; head at 21, chunks of 16) come out exactly
    /// as each round's own cold MISS: next token, state byte for byte, greedy steps. Rounds that
    /// end short of, on, and past grid points, the identical prompt retried, a round that reaches
    /// no new grid point, and a first prompt long enough to leave a point past its head. Without
    /// the flag, extending the previous round's full prompt does not (shown on the pool, where the
    /// replica's numbers depend on the cut).
    #[test]
    fn exact_reuse_rounds_continue_exactly_like_their_cold_prompts() {
        for budget in [None, Some(1u64)] {
            exact_rounds(MtpMode::Off, budget);
        }
        // With the draft head, whose cache the kept points carry too.
        exact_rounds(MtpMode::On, Some(1));
    }

    /// `LUMEN_QWEN4_KV_BF16`: the attention caches (and the indexer's keys) hold bf16, the
    /// memory plan counts half the bytes for them, the disk records keep the dtype, and an
    /// agent's rounds still continue exactly like their cold prompts (both computed in bf16,
    /// past the replica's indexer budget, so the sparse path too), with and without the expert
    /// pool and the draft head.
    #[test]
    fn bf16_kv_rounds_continue_exactly_like_their_cold_prompts() {
        use crate::kv_disk::{DtypeTag, LayerKindTag};
        let f32_bytes = f32_kv(|| tiny_runner(16).model.config().kv_bytes_per_token());
        super::super::config::kv_bf16::with(true, || {
            let mut r = tiny_runner(16);
            assert_eq!(2 * r.model.config().kv_bytes_per_token(), f32_bytes);
            Runner::prefill(&mut r, 1, &fixture_ids()[..40]).unwrap();
            let (metas, records) = r.seqs[&1].cache.to_records().unwrap();
            let full: Vec<&DtypeTag> = metas
                .iter()
                .filter(|m| m.kind == LayerKindTag::Full)
                .flat_map(|m| m.arrays.iter().flatten().map(|&i| &records[i].dtype))
                .collect();
            assert!(!full.is_empty());
            assert!(full.iter().all(|d| **d == DtypeTag::BF16), "{full:?}");

            for budget in [None, Some(1u64)] {
                exact_rounds(MtpMode::Off, budget);
            }
            exact_rounds(MtpMode::On, Some(1));
        });
    }

    /// `LUMEN_JOINT_PREFILL`: two conversations that share a head, one round in, each extend
    /// their own resume point. Feeding both rounds ahead in one pass (what the engine does when
    /// the second request is waiting) must leave each request exactly what its cold prompt
    /// gives: the next token, the state, six greedy steps, and a kept resume point the next
    /// round continues from exactly. A feed computed for a prompt nobody sends is never taken.
    #[test]
    fn rounds_fed_together_continue_exactly_like_their_cold_prompts() {
        for budget in [None, Some(1u64)] {
            joint_rounds(budget);
        }
    }

    fn joint_rounds(budget: Option<u64>) {
        use crate::prefix_cache::{PrefeedItem, PrefixCacheStore, exact_reuse};
        let ids = fixture_ids();
        let at = 21;
        // Two conversations with the same head (system prompt and tools) and their own rest.
        let convo = |shift: u32| -> Vec<u32> {
            (0..100)
                .map(|i| {
                    if i < at {
                        ids[i]
                    } else {
                        (ids[i % ids.len()] + shift + 7 * (i / ids.len()) as u32) % 128
                    }
                })
                .collect()
        };
        let (a, b) = (convo(0), convo(3));
        exact_reuse::with(true, || {
            let mut r = tiny_runner_with(16, MtpMode::Off, budget);
            let mut store = PrefixCacheStore::with_incremental(true, 4, true);
            let round = |r: &mut Qwen4ExpRunner, store: &mut PrefixCacheStore, seq, p: &[u32]| {
                store
                    .prefill_optionally_cached(r, seq, p, Some("k"), Some(at))
                    .unwrap()
                    .0
            };
            // Round 1 of each: a cold MISS (the head and A's point), then a HIT of the head.
            round(&mut r, &mut store, 1, &a[..40]);
            round(&mut r, &mut store, 2, &b[..44]);

            // Round 2 of each, fed together ahead.
            let plan_a = store.predict_feed(&r, "k", &a[..75]).expect("A extends").0;
            let plan_b = store.predict_feed(&r, "k", &b[..82]).expect("B extends").0;
            assert_ne!(plan_a.snapshot, plan_b.snapshot, "each its own point");
            assert!(
                plan_a.mark.is_some() && plan_b.mark.is_some(),
                "each keeps a new point"
            );
            let stray = PrefeedItem {
                tokens: plan_a.tokens[..plan_a.tokens.len() - 3].to_vec(),
                ..plan_a.clone()
            };
            assert_eq!(r.prefeed(&[plan_a, plan_b, stray]).unwrap(), 3);
            // What A's result holds is in memory already: its request plans no more.
            let plan = store.plan_feed(&mut r, Some("k"), &a[..75], true);
            assert_eq!((plan.kv_tokens, plan.states), (0, 0), "A's feed is held");
            for (seq, prompt) in [(3, &a[..75]), (4, &b[..82])] {
                let cold = cold_in_two_calls_with(MtpMode::Off, budget, prompt, at);
                let next = round(&mut r, &mut store, seq, prompt);
                assert_continues_like(&mut r, seq, next, &cold, &format!("{budget:?} seq {seq}"));
            }
            assert_eq!(
                r.prefed.len(),
                1,
                "both rounds took theirs, the stray one is left"
            );
            r.clear_prefeed();
            assert!(r.prefed.is_empty());

            // Round 3 continues from the points the fed rounds kept.
            for (seq, prompt) in [(5, &a[..100]), (6, &b[..100])] {
                let cold = cold_in_two_calls_with(MtpMode::Off, budget, prompt, at);
                let next = round(&mut r, &mut store, seq, prompt);
                assert_continues_like(&mut r, seq, next, &cold, &format!("{budget:?} seq {seq}"));
            }
        });
    }

    /// A fed result stands for a sequence forked from its snapshot and not touched since: one
    /// that took a decode step (or was changed any other way) computes its own extend.
    #[test]
    fn a_fed_result_is_only_taken_by_an_untouched_fork() {
        use crate::prefix_cache::PrefeedItem;
        let ids = fixture_ids();
        let mut r = tiny_runner(16);
        Runner::prefill(&mut r, 1, &ids[..20]).unwrap();
        let (snap, _) = Runner::snapshot_state_deep(&mut r, 1).unwrap();
        let item = |tokens: &[u32]| PrefeedItem {
            snapshot: snap,
            tokens: tokens.to_vec(),
            mark: None,
        };
        assert_eq!(
            r.prefeed(&[item(&ids[20..30]), item(&ids[20..34])])
                .unwrap(),
            2
        );
        // The next plan no longer has the second request (it was served, or went another way):
        // its feed goes, the other stays, and one feed alone is not computed again.
        assert_eq!(r.prefeed(&[item(&ids[20..30])]).unwrap(), 0);
        assert_eq!(r.prefed.len(), 1);

        // Touched after the fork: computes its own, the stash keeps the feed.
        Runner::fork_from_snapshot(&mut r, snap, 2).unwrap();
        let (step, _) = Runner::decode_step(&mut r, 2, ids[19], 20).unwrap();
        let _ = step;
        Runner::extend(&mut r, 2, &ids[20..30]).unwrap();
        assert_eq!(r.prefed.len(), 1);

        // Untouched: takes it, and the result is the extend's own.
        Runner::fork_from_snapshot(&mut r, snap, 3).unwrap();
        let taken = Runner::extend(&mut r, 3, &ids[20..30]).unwrap();
        assert!(r.prefed.is_empty());
        Runner::fork_from_snapshot(&mut r, snap, 4).unwrap();
        r.clear_prefeed();
        let own = Runner::extend(&mut r, 4, &ids[20..30]).unwrap();
        assert_eq!(taken, own);
        assert!(state_bytes(&r, 3) == state_bytes(&r, 4));
    }

    /// Feeds too short to take the staging path (one and two rows, a few of the replica's 16
    /// experts each) are fed together through the expert pool, each layer for one and then the
    /// other, and each still comes out as its own extend.
    #[test]
    fn short_feeds_fed_together_through_the_pool_match_their_own_extends() {
        use crate::prefix_cache::PrefeedItem;
        let ids = fixture_ids();
        let mut r = tiny_runner_with(16, MtpMode::Off, Some(1));
        Runner::prefill(&mut r, 1, &ids[..20]).unwrap();
        let (snap, _) = Runner::snapshot_state_deep(&mut r, 1).unwrap();
        let feeds = [&ids[20..21], &ids[20..22]];
        let items = feeds.map(|tokens| PrefeedItem {
            snapshot: snap,
            tokens: tokens.to_vec(),
            mark: None,
        });
        assert_eq!(r.prefeed(&items).unwrap(), 2);
        for (i, tokens) in feeds.into_iter().enumerate() {
            let (fed, own) = (10 + 2 * i as u64, 11 + 2 * i as u64);
            Runner::fork_from_snapshot(&mut r, snap, fed).unwrap();
            let taken = Runner::extend(&mut r, fed, tokens).unwrap();
            Runner::fork_from_snapshot(&mut r, snap, own).unwrap();
            let computed = Runner::extend(&mut r, own, tokens).unwrap();
            assert_eq!(taken, computed, "feed {i}");
            assert!(state_bytes(&r, fed) == state_bytes(&r, own), "feed {i}");
        }
        assert!(r.prefed.is_empty(), "both were taken");
    }

    fn exact_rounds(mtp: MtpMode, budget: Option<u64>) {
        use crate::prefix_cache::{PrefixCacheStore, exact_reuse};
        let ids = fixture_ids();
        let at = 21;
        // Longer than the fixture: its ids, then the same ids shifted.
        let convo: Vec<u32> = (0..100)
            .map(|i| (ids[i % ids.len()] + 7 * (i / ids.len()) as u32) % 128)
            .collect();
        let run = |exact: bool, ends: &[usize]| -> Vec<(Vec<(usize, u64, bool)>, bool)> {
            let mut r = tiny_runner_with(16, mtp, budget);
            let mut store = PrefixCacheStore::with_incremental(true, 4, true);
            let mut out = Vec::new();
            for (i, &n) in ends.iter().enumerate() {
                let seq = i as u64 + 1;
                let cold = cold_in_two_calls_with(mtp, budget, &convo[..n], at);
                let (next, _) = exact_reuse::with(exact, || {
                    store.prefill_optionally_cached(&mut r, seq, &convo[..n], Some("k"), Some(at))
                })
                .unwrap();
                if exact {
                    assert_continues_like(&mut r, seq, next, &cold, &format!("round {seq} ({n})"));
                } else {
                    let same = next == cold.0 && state_bytes(&r, seq) == cold.1;
                    Runner::remove_seq(&mut r, seq).unwrap();
                    out.push((store.branches_for_test("k"), same));
                    continue;
                }
                out.push((store.branches_for_test("k"), true));
            }
            out
        };
        let pinned = |v: &[(usize, u64, bool)]| -> Vec<(usize, bool)> {
            v.iter().map(|&(n, _, p)| (n, p)).collect()
        };

        // 30: under a chunk past the head (the head is the point); 45, 60: points 37, 53; 69: on
        // the grid; 69 again: the retry; 74: no new point; 100: point 85.
        let ends = [30, 45, 60, 69, 69, 74, 100];
        let exact = run(true, &ends);
        let layout: Vec<_> = exact.iter().map(|(b, _)| pinned(b)).collect();
        assert_eq!(
            layout[0],
            vec![(21, true)],
            "a MISS under a chunk keeps the head only"
        );
        assert_eq!(layout[1], vec![(21, true), (37, false)]);
        assert_eq!(
            layout[2],
            vec![(21, true), (53, false)],
            "a newer point replaces the older"
        );
        assert_eq!(layout[3], vec![(21, true), (69, false)]);
        assert_eq!(
            layout[5],
            vec![(21, true), (69, false)],
            "no grid point reached"
        );
        assert_eq!(layout[6], vec![(21, true), (85, false)]);

        // A first prompt that leaves a point past its head in the same pass.
        let exact = run(true, &[60, 75]);
        assert_eq!(pinned(&exact[0].0), vec![(21, true), (53, false)]);

        // The behaviour this replaces: on the pool some round comes out differently.
        let plain = run(false, &ends);
        assert!(plain[0].1, "a cold MISS is the cold MISS");
        if budget.is_some() {
            assert!(
                plain.iter().any(|(_, same)| !same),
                "extending full prompts matched the cold rounds: the test shows nothing"
            );
        }
    }

    /// Where the feed is cut changes the numbers even on the replica: the prompt fed in one call
    /// on the 16-token grid is not the prompt cut at the head. That is why only a head at a cut
    /// the cold MISS makes may be reused.
    #[test]
    fn a_different_cut_is_a_different_computation() {
        let ids = fixture_ids();
        let a: Vec<u32> = ids[..40].to_vec();
        let split = cold_in_two_calls(Some(1), &a, 21);
        let mut r = tiny_runner_with(16, MtpMode::Off, Some(1));
        Runner::prefill(&mut r, 1, &a).unwrap();
        let whole = state_bytes(&r, 1);
        assert!(
            whole != split.1,
            "the cut at 21 left the state bit-identical to the plain prefill"
        );
    }

    /// A head on disk that is cut short, fails its CRC, or was written under another identity is
    /// a miss: the prompt is fed cold and comes out exactly as a cold MISS. Never a wrong state.
    #[test]
    fn a_damaged_or_foreign_head_on_disk_is_a_miss_and_never_a_wrong_answer() {
        use crate::prefix_cache::PrefixCacheStore;
        let ids = fixture_ids();
        let at = 21;
        let a: Vec<u32> = ids[..40].to_vec();
        let b: Vec<u32> = ids[..at].iter().chain(&ids[30..48]).copied().collect();
        let cold_b = cold_in_two_calls(None, &b, at);
        let key = crate::kv_disk::head_key(&b[..at]);
        let dir = tempfile::tempdir().unwrap();
        let write_head = |fingerprint: &str| {
            let mut r = tiny_runner(16);
            r.set_disk_root_for_test(Some(dir.path()), fingerprint);
            let mut store = PrefixCacheStore::with_incremental(true, 4, true);
            store
                .prefill_optionally_cached(&mut r, 1, &a, Some("k"), Some(at))
                .unwrap();
            store.flush_persist(&mut r);
            dir.path().join(fingerprint).join(format!("{key}.lkv"))
        };
        let damages: [(&str, fn(&mut Vec<u8>)); 2] = [
            ("cut short", |f| f.truncate(f.len() / 2)),
            ("CRC mismatch", |f| {
                let i = f.len() / 2;
                f[i] ^= 0x40;
            }),
        ];
        for (what, damage) in damages {
            let file = write_head("tiny");
            let mut bytes = std::fs::read(&file).unwrap();
            damage(&mut bytes);
            std::fs::write(&file, &bytes).unwrap();
            let mut fresh = tiny_runner(16);
            fresh.set_disk_root_for_test(Some(dir.path()), "tiny");
            let mut store = PrefixCacheStore::with_incremental(true, 4, true);
            let (next, _) = store
                .prefill_optionally_cached(&mut fresh, 1, &b, Some("k"), Some(at))
                .unwrap();
            assert_eq!(
                store.branches_for_test("k")[0],
                (at, 0, true),
                "{what}: a cold MISS"
            );
            assert_continues_like(&mut fresh, 1, next, &cold_b, what);
        }

        // Another identity's files under this one's directory: the manifest says whose they are.
        write_head("other");
        std::fs::create_dir_all(dir.path().join("mine")).unwrap();
        for f in ["index.json", &format!("{key}.lkv")] {
            std::fs::copy(
                dir.path().join("other").join(f),
                dir.path().join("mine").join(f),
            )
            .unwrap();
        }
        let mut fresh = tiny_runner(16);
        fresh.set_disk_root_for_test(Some(dir.path()), "mine");
        assert!(
            fresh.load_head(&key, &b).unwrap().is_none(),
            "a foreign head was served"
        );
        let mut store = PrefixCacheStore::with_incremental(true, 4, true);
        let (next, _) = store
            .prefill_optionally_cached(&mut fresh, 1, &b, Some("k"), Some(at))
            .unwrap();
        assert_continues_like(&mut fresh, 1, next, &cold_b, "foreign identity");

        // The head is only served to a prompt that continues it.
        let mut fresh = tiny_runner(16);
        fresh.set_disk_root_for_test(Some(dir.path()), "other");
        assert!(
            fresh.load_head(&key, &b[..at]).unwrap().is_none(),
            "nothing left to feed"
        );
        assert!(
            fresh.load_head(&key, &ids[1..40]).unwrap().is_none(),
            "another prompt"
        );
        assert!(fresh.load_head(&key, &b).unwrap().is_some());
    }

    // ───────────── the session's rollback point: on disk, after a restart, damaged ─────────────

    /// The plain-chat path's disk tier. Turn 1 leaves its rollback point at 32 (the 16-token
    /// grid) and writes it with `persist_point`; a restart's `restore_point` starts turn 2 there,
    /// and turn 2 continues exactly as its own cold prefill: next token, state byte for byte,
    /// greedy steps. A state on disk that is cut short, fails its CRC or was written under another
    /// identity, one from another grid and one the prompt does not continue are all misses — the
    /// turn is fed cold — and a damaged file is dropped, never served. Resident experts and a pool.
    #[test]
    fn a_session_point_from_disk_continues_exactly_and_a_bad_one_is_a_cold_turn() {
        for budget in [None, Some(1u64)] {
            session_point_restores_exactly(budget);
        }
    }

    fn session_point_restores_exactly(budget: Option<u64>) {
        let ids = fixture_ids();
        assert!(ids.len() >= 46);
        let (turn1, turn2) = (&ids[..40], &ids[..46]);
        let cold = {
            let mut r = tiny_runner_with(16, MtpMode::Off, budget);
            let (next, _) = Runner::prefill(&mut r, 1, turn2).unwrap();
            (next, state_bytes(&r, 1), decode_some(&mut r, 1, next, 6))
        };
        let key = crate::kv_disk::point_key(16, &turn1[..32]);
        let dir = tempfile::tempdir().unwrap();
        // Turn 1 in a process that writes its point under `fingerprint`: the file it leaves.
        let write_point = |fingerprint: &str| {
            let mut r = tiny_runner_with(16, MtpMode::Off, budget);
            r.set_disk_root_for_test(Some(dir.path()), fingerprint);
            r.feed_with_mark(1, true, turn1, 32)
                .unwrap()
                .expect("a mark on the grid");
            let written = r.persist_point(1, &turn1[..32], 16).unwrap();
            assert!(
                !r.persist_point(1, &turn1[..32], 16).unwrap(),
                "a stored point is not written twice"
            );
            (
                dir.path().join(fingerprint).join(format!("{key}.lkv")),
                written,
            )
        };
        // A new process on the same disk: where it starts turn 2, and the store's size after.
        let restart = |fingerprint: &str, chunk: usize, prompt: &[u32]| {
            let mut r = tiny_runner_with(chunk, MtpMode::Off, budget);
            r.set_disk_root_for_test(Some(dir.path()), fingerprint);
            let at = r.restore_point(7, prompt).unwrap();
            let left = r.disk.as_ref().map_or(0, DiskKvStore::len);
            (r, at, left)
        };

        assert!(write_point("tiny").1, "turn 1's point is written");
        let (mut r, at, _) = restart("tiny", 16, turn2);
        assert_eq!(at, Some(32), "restored at the point");
        let (next, _) = Runner::extend(&mut r, 7, &turn2[32..]).unwrap();
        assert_continues_like(&mut r, 7, next, &cold, "turn 2 restored after a restart");

        // Misses that leave the entry alone: another grid, a prompt that does not continue it.
        assert_eq!(restart("tiny", 8, turn2).1, None, "another grid");
        let other: Vec<u32> = std::iter::once((turn2[0] + 1) % 128)
            .chain(turn2[1..].iter().copied())
            .collect();
        assert_eq!(restart("tiny", 16, &other).1, None, "another conversation");
        assert_eq!(
            restart("tiny", 16, &turn2[..32]).1,
            None,
            "nothing left to feed"
        );

        let damages: [(&str, fn(&mut Vec<u8>)); 2] = [
            ("cut short", |f| f.truncate(f.len() / 2)),
            ("CRC mismatch", |f| {
                let i = f.len() / 2;
                f[i] ^= 0x40;
            }),
        ];
        for (what, damage) in damages {
            let (file, _) = write_point("tiny");
            let mut bytes = std::fs::read(&file).unwrap();
            damage(&mut bytes);
            std::fs::write(&file, &bytes).unwrap();
            let (mut r, at, left) = restart("tiny", 16, turn2);
            assert_eq!(at, None, "{what}: served");
            assert_eq!(left, 0, "{what}: the damaged entry is kept");
            let (next, _) = Runner::prefill(&mut r, 7, turn2).unwrap();
            assert_continues_like(&mut r, 7, next, &cold, what);
        }

        // Another identity's files under this one's directory: the manifest says whose they are.
        write_point("other");
        let mine = dir.path().join("mine");
        std::fs::create_dir_all(&mine).unwrap();
        for f in ["index.json", &format!("{key}.lkv")] {
            std::fs::copy(dir.path().join("other").join(f), mine.join(f)).unwrap();
        }
        let (mut r, at, left) = restart("mine", 16, turn2);
        assert_eq!(at, None, "a foreign state was served");
        assert_eq!(left, 0, "the foreign entry is kept");
        let (next, _) = Runner::prefill(&mut r, 7, turn2).unwrap();
        assert_continues_like(&mut r, 7, next, &cold, "foreign identity");
    }
}
