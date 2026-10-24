//! SSD expert offload: keep a fixed number of a MoE layer's experts resident in
//! GPU-visible slots and stream the rest from disk (task 023).
//!
//! The pieces build on each other:
//!
//! * [`expert_store`] — where each routed expert's bytes live in the original
//!   safetensors shards (no repacking, no extra disk).
//! * [`io`] — a thread pool that reads those byte ranges in parallel into
//!   private aligned buffers and copies them into slot memory.
//! * [`expert_bank`] — the four pieces joined: `begin_acquire(layer, experts)` makes
//!   experts resident and returns slot numbers for the gather.
//! * [`slot_pool`] — stacked tensors whose slots the CPU can overwrite in place.
//! * [`slot_map`] — which `(layer, expert)` lives in which slot: pins, the free list and the
//!   plan → load → commit protocol.
//! * [`slot_policy`] — which resident expert goes when the pool is full: CLOCK, or the
//!   alternatives a trace replay found better (`LUMEN_MOE_EVICTION`).
//! * [`trace`] — opt-in recording of every layer call's expert ids (`LUMEN_MOE_TRACE`).
//!
//! [`MoeOffload`] is what the model holds: every MoE layer calls
//! [`MoeOffload::route`] between computing its routing and gathering.

pub(crate) mod expert_bank;
pub(crate) mod expert_store;
pub(crate) mod io;
pub(crate) mod pack;
pub(crate) mod slot_map;
pub(crate) mod slot_policy;
pub(crate) mod slot_pool;
#[cfg(test)]
mod slot_replay;
pub(crate) mod trace;

#[cfg(test)]
mod exactness_tests;

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow, bail};
use mlx_rs::Array;

pub(crate) use expert_bank::{Acquired, Acquiring, ExpertBank, RegionLoad, STAGING_REGIONS};
pub(crate) use expert_store::MTP_LAYER;
pub(crate) use expert_store::read_header;
pub(crate) use io::{IoMode, IoPool, ReadJob};
pub(crate) use pack::PackMode;

lumen_flags::flag! {
    /// Keep the GPU busy while a decode step's missing experts are read from the SSD: as soon as
    /// the routing is known, submit the routed experts the pool already holds (and the shared
    /// expert), and gather the missing ones once their reads land. Each expert's output is the
    /// same per-row `gather_qmm` kernel over the same bytes either way, and the outputs are put
    /// back in the routing's order before today's score-weighted sum, so the output is
    /// bit-identical (`offload::exactness_tests`). Only calls of up to four rows that both hit
    /// and miss, with reads expected to take at least 0.5 ms; everything else (prefill, staging,
    /// `LUMEN_QWEN4_BUILD_AHEAD`) is unchanged. Measured on the real checkpoint (M3 Max, N=5,
    /// identical text): +4.0% at a 4.2 GB pool and +4.6% at 15 GB on this 6.5 GB/s SSD, -0.8%
    /// (noise) at 6.7 GB, and only +1.3% to +2.4% with reads paced to 3 or 1.5 GB/s, where the
    /// read is 5-10 ms and the hidden GPU work 0.1-0.2 ms per layer call. Kept opt-in.
    pub(crate) overlap {
        env: "LUMEN_MOE_OVERLAP",
        default: false,
        kind: Optimization,
    }
}

/// An overlapped call runs one more GPU submission and gathers in two parts, which costs a few
/// tens of microseconds; reads expected to take less than this are not worth splitting for.
const OVERLAP_MIN_READ_US: u64 = 500;

/// [`OVERLAP_MIN_READ_US`] in ns, unless a test lowered it for its thread.
fn overlap_min_read_ns() -> u64 {
    #[cfg(test)]
    if let Some(ns) = test_overlap::MIN_READ_NS.with(std::cell::Cell::get) {
        return ns;
    }
    OVERLAP_MIN_READ_US * 1_000
}

/// The tiny replica's experts are a few KB, read in microseconds, so no call of it would reach
/// [`OVERLAP_MIN_READ_US`]: tests that want the overlapped path exercised lower the bar.
#[cfg(test)]
pub(crate) mod test_overlap {
    use std::cell::Cell;

    thread_local! {
        pub(super) static MIN_READ_NS: Cell<Option<u64>> = const { Cell::new(None) };
    }

    /// Run `f` with every read long enough to overlap (on this thread), restoring on unwind.
    pub(crate) fn with_any_read<R>(f: impl FnOnce() -> R) -> R {
        struct Reset(Option<u64>);
        impl Drop for Reset {
            fn drop(&mut self) {
                MIN_READ_NS.with(|c| c.set(self.0));
            }
        }
        let _reset = Reset(MIN_READ_NS.with(Cell::get));
        MIN_READ_NS.with(|c| c.set(Some(0)));
        f()
    }
}

/// Reader threads per bank. Four saturate the SSD for reads of 1 MB and up, but an
/// expert is 6-9 ranges of 50 KB to 900 KB and a layer call misses several experts, so
/// the queue depth decides the speed. Flash-Next at a 6.7 GB pool (about 40 reads per
/// layer call): 8 / 16 / 32 / 48 / 64 lanes decode at 4.78 / 5.24 / 5.68 / 5.73 / 5.86
/// tok/s. The 35B (about 14 reads per call) is indifferent, so 32 costs nothing there.
const DEFAULT_LANES: usize = 32;

/// Most reader threads a bank accepts: past this a typo, not a tuning choice.
const MAX_LANES: usize = 64;

/// How the expert bytes are read: the knobs that only change speed, never the
/// output.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IoTuning {
    pub lanes: usize,
    pub mode: IoMode,
    /// Whether the contiguous expert pack is used (and built when missing).
    pub pack: PackMode,
    /// Serve the draft head's routed experts (`mtp.safetensors`) from the pool too.
    pub mtp_experts: bool,
}

impl Default for IoTuning {
    fn default() -> Self {
        Self {
            lanes: DEFAULT_LANES,
            mode: IoMode::Auto,
            pack: PackMode::Auto,
            mtp_experts: false,
        }
    }
}

impl IoTuning {
    /// `LUMEN_MOE_IO_LANES` (1..=64 reader threads), `LUMEN_MOE_IO_MODE`
    /// (`direct` bypasses the page cache, `buffered` uses it, `auto` picks by the
    /// free memory and is the default) and `LUMEN_MOE_PACK` (`auto` uses an expert pack
    /// when one is there, `off` never does, `build` makes one first; see [`pack`]). Unset keeps the
    /// default; anything unparsable is an error, since a typo that silently fell
    /// back would make an A/B run measure the default twice.
    pub(crate) fn from_env() -> Result<Self> {
        Self::parse(
            std::env::var("LUMEN_MOE_IO_LANES").ok().as_deref(),
            std::env::var("LUMEN_MOE_IO_MODE").ok().as_deref(),
        )
        .and_then(|mut t| {
            t.pack = PackMode::from_env()?;
            Ok(t)
        })
    }

    fn parse(lanes: Option<&str>, mode: Option<&str>) -> Result<Self> {
        let mut tuning = Self::default();
        if let Some(raw) = lanes.map(str::trim).filter(|r| !r.is_empty()) {
            tuning.lanes = raw
                .parse::<usize>()
                .ok()
                .filter(|n| (1..=MAX_LANES).contains(n))
                .ok_or_else(|| {
                    anyhow!("LUMEN_MOE_IO_LANES={raw:?} is not a number from 1 to {MAX_LANES}")
                })?;
        }
        if let Some(raw) = mode {
            tuning.mode = match raw.trim().to_ascii_lowercase().as_str() {
                "" | "auto" => IoMode::Auto,
                "direct" => IoMode::Direct,
                "buffered" => IoMode::Buffered,
                other => bail!("LUMEN_MOE_IO_MODE={other:?} is not `auto`, `direct` or `buffered`"),
            };
        }
        Ok(tuning)
    }
}

/// How many slots a pool of `budget_bytes` has, never fewer than one staging region plus one
/// layer's worth beside it: a call that touches most of a layer's experts but not enough to take
/// the staging path must still fit, and below `MIN_SLOTS` not even decode does. A pool with room
/// for a second region (three layers' worth) gets it, and with it prefetching of the next layer.
fn pool_slot_count(budget_bytes: u64, per_expert: u64, total: usize, widest_layer: usize) -> usize {
    let floor = (2 * widest_layer).max(MIN_SLOTS).min(total);
    ((budget_bytes / per_expert.max(1)) as usize).clamp(floor, total)
}

#[cfg(test)]
mod slot_count_tests {
    use super::*;

    #[test]
    fn the_budget_decides_between_the_floor_and_the_whole_model() {
        // 100 bytes per expert, 40 layers x 256 experts.
        assert_eq!(pool_slot_count(1_000_000, 100, 10_240, 256), 10_000);
        assert_eq!(
            pool_slot_count(u64::MAX, 100, 10_240, 256),
            10_240,
            "capped at every expert"
        );
    }

    #[test]
    fn a_tiny_budget_still_holds_a_staging_region_and_a_layer() {
        assert_eq!(pool_slot_count(0, 100, 10_240, 256), 512);
        assert_eq!(pool_slot_count(5_000, 100, 10_240, 256), 512);
    }

    #[test]
    fn a_second_region_needs_three_layers_of_slots() {
        use super::expert_bank::{STAGING_REGIONS, regions_for};
        assert_eq!(regions_for(512, 256), 1);
        assert_eq!(regions_for(767, 256), 1);
        assert_eq!(regions_for(768, 256), STAGING_REGIONS);
    }

    #[test]
    fn a_tiny_model_is_not_padded_past_its_experts() {
        assert_eq!(pool_slot_count(0, 100, 8, 8), 8);
        assert_eq!(pool_slot_count(0, 100, 0, 0), 0);
    }
}

#[cfg(test)]
mod tuning_tests {
    use super::*;

    #[test]
    fn unset_or_empty_keeps_the_defaults() {
        for (lanes, mode) in [(None, None), (Some(""), Some("")), (Some("  "), Some(" "))] {
            let t = IoTuning::parse(lanes, mode).unwrap();
            assert_eq!((t.lanes, t.mode), (DEFAULT_LANES, IoMode::Auto));
        }
    }

    #[test]
    fn valid_values_are_taken_and_junk_is_an_error() {
        let t = IoTuning::parse(Some(" 16 "), Some("Buffered")).unwrap();
        assert_eq!((t.lanes, t.mode), (16, IoMode::Buffered));
        for bad in ["0", "65", "-1", "many", "8.5"] {
            assert!(IoTuning::parse(Some(bad), None).is_err(), "lanes {bad:?}");
        }
        assert!(IoTuning::parse(None, Some("mmap")).is_err());
        assert_eq!(
            IoTuning::parse(None, Some("Direct")).unwrap().mode,
            IoMode::Direct
        );
    }
}

/// Build the contiguous expert pack of the checkpoint in `dir` (68 GB for Flash-Next), or
/// report that a usable one is already there. For `lumen-server --pack-experts`.
pub fn build_expert_pack(dir: &Path) -> Result<std::path::PathBuf> {
    let source = expert_store::ExpertStore::open(dir)?;
    if pack::open_pack(dir, &source)?.is_some() {
        return Ok(pack::pack_path(dir));
    }
    let started = std::time::Instant::now();
    let path = pack::build(dir, &source, |done, total| {
        eprintln!("[offload] expert pack: layer {done} of {total}");
    })?;
    eprintln!(
        "[offload] expert pack ready in {:.0} s: {}",
        started.elapsed().as_secs_f64(),
        path.display()
    );
    Ok(path)
}

/// Fewest slots a bank is built with: enough for a decode step of a few
/// sequences (each routes 8-10 experts per layer) with room to spare.
const MIN_SLOTS: usize = 16;

/// What one routed MoE layer call gets back: the routing expressed in slots,
/// and handles to the pool arrays the gather must run over.
pub(crate) struct Routed {
    /// The routing's expert ids with every id replaced by its slot, same shape.
    pub slot_inds: Array,
    /// Pool arrays by role (`gate_proj.weight`, ...); cheap handle clones.
    arrays: Vec<(String, Array)>,
}

impl Routed {
    pub(crate) fn array(&self, role: &str) -> Option<&Array> {
        self.arrays.iter().find(|(r, _)| r == role).map(|(_, a)| a)
    }
}

/// What [`MoeOffload::route_overlapping`] returns.
pub(crate) enum Routing {
    /// As [`MoeOffload::route`]: every expert resident by now, gather over `slot_inds`.
    Whole(Routed),
    /// The call's reads overlapped with the resident experts' work.
    Split(Split),
}

/// An overlapped layer call, once its reads have landed. The routed entries (row, top-k
/// position) are in two parts: the resident ones, whose work the caller built while the reads
/// ran and which is running on the GPU, and the missing ones, to gather now.
pub(crate) struct Split {
    /// What the caller's `resident` closure built, in its order, already submitted.
    pub resident_out: Vec<Array>,
    /// The missing entries' slots (`slot_inds`, `[n]`) and the pool arrays to gather over.
    pub missing: Routed,
    /// Row of the call's input (flattened over batch and sequence) of each missing entry.
    pub missing_rows: Array,
    /// For every routed entry in the routing's order, its index among the resident entries
    /// followed by the missing ones: `take(concat(resident, missing), order)` restores the order.
    pub order: Array,
}

/// The work a caller hands [`MoeOffload::route_overlapping`]: given the resident entries (their
/// slots as a `Routed`, and their rows), build (lazily, without evaluating) the GPU work to run
/// while the reads do. The offload submits it and keeps the slots pinned until it is done.
pub(crate) type ResidentWork<'a> = dyn FnMut(&Routed, &Array) -> Result<Vec<Array>> + 'a;

/// A layer call's slots, pinned until the GPU work that reads them is known to be done.
struct Pinned {
    acquired: Acquired,
    /// Work an overlapped call submitted while its reads ran. Its result feeds the next routing,
    /// so that readback completes it too; holding it makes the release wait for it explicitly,
    /// including on the error paths where no readback follows. The plain path has none: its
    /// gather is submitted with the next routing, whose readback completes it.
    gpu: Vec<Array>,
}

impl Pinned {
    fn release(self, bank: &mut ExpertBank) {
        wait_for(&self.gpu);
        bank.release(self.acquired);
    }
}

/// Block until `arrays` are computed (or evaluate them, if they never were submitted). Errors are
/// ignored: a failed evaluation leaves nothing running, which is all a caller about to unpin needs.
fn wait_for(arrays: &[Array]) {
    for a in arrays {
        let _ = a.eval();
    }
}

/// The routed entries of a call split by whether their expert was resident: slots and rows of
/// each part, and where every entry of the routing lands in `resident ++ missing`.
#[derive(Debug, PartialEq, Eq)]
struct Parts {
    resident_slots: Vec<u32>,
    resident_rows: Vec<u32>,
    missing_slots: Vec<u32>,
    missing_rows: Vec<u32>,
    order: Vec<u32>,
}

/// Split the call's entries (`ids`, `k` per row) into resident and missing ones, each part in the
/// routing's order. `slots` and `missing` are per distinct expert (`index_of`).
fn split_entries(
    ids: &[u32],
    k: usize,
    index_of: &HashMap<u32, usize>,
    slots: &[usize],
    missing: &[bool],
) -> Parts {
    let n_resident = ids.iter().filter(|id| !missing[index_of[id]]).count() as u32;
    let mut p = Parts {
        resident_slots: Vec::with_capacity(n_resident as usize),
        resident_rows: Vec::with_capacity(n_resident as usize),
        missing_slots: Vec::with_capacity(ids.len() - n_resident as usize),
        missing_rows: Vec::with_capacity(ids.len() - n_resident as usize),
        order: Vec::with_capacity(ids.len()),
    };
    for (i, id) in ids.iter().enumerate() {
        let d = index_of[id];
        let (slot, row) = (slots[d] as u32, (i / k) as u32);
        if missing[d] {
            p.order.push(n_resident + p.missing_slots.len() as u32);
            p.missing_slots.push(slot);
            p.missing_rows.push(row);
        } else {
            p.order.push(p.resident_slots.len() as u32);
            p.resident_slots.push(slot);
            p.resident_rows.push(row);
        }
    }
    p
}

/// Whether a pool-path call overlaps its reads with resident work: a decode-sized call (the sizes
/// whose split is proven bit-identical) that both hits and misses, whose reads are expected to
/// take at least `min_read_ns` (`ns_per_expert` 0: no estimate yet, so yes).
fn overlaps(
    rows: usize,
    hits: usize,
    misses: usize,
    ns_per_expert: u64,
    min_read_ns: u64,
) -> Decision {
    if rows > trace::DECODE_ROWS as usize || hits == 0 || misses == 0 {
        Decision::Plain
    } else if ns_per_expert > 0 && misses as u64 * ns_per_expert < min_read_ns {
        Decision::Short
    } else {
        Decision::Overlap
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Decision {
    Plain,
    /// Would overlap, but the reads are expected to be too short to be worth it.
    Short,
    Overlap,
}

/// A checkpoint's routed experts, served from a slot pool instead of RAM.
///
/// Held by the model; every MoE layer calls [`Self::route`] between computing
/// its routing and gathering. See the module docs for how the pieces fit.
pub(crate) struct MoeOffload {
    bank: Mutex<ExpertBank>,
    /// The slots of the previous layer call, still pinned: that call's gather
    /// is only known to be finished once the next routing readback returns (and
    /// whatever it submitted itself, once that has been waited for; see [`Pinned`]).
    pending: Mutex<Option<Pinned>>,
    /// How long reading one missing expert takes here, in ns (a moving average over the calls
    /// that read; 0 until one has): how `LUMEN_MOE_OVERLAP` tells a long read from a short one,
    /// on this machine's SSD or under `LUMEN_MOE_IO_THROTTLE_GBPS`.
    read_ns_per_expert: AtomicU64,
    /// Which layer each staging region holds (or is being loaded with), and the load in flight.
    staging: Mutex<Staging>,
    /// Where the layer calls spend their time, to log now and then.
    timing: Mutex<Timing>,
    /// Bytes one expert occupies, to turn loaded experts into MB.
    expert_bytes: u64,
    /// Set while a prefill feeds several chunks through one layer before moving to the next:
    /// every call then takes the staging path, whatever share of the experts it routes to,
    /// because the layer is already (or about to be) in a region and the next chunk wants it.
    bulk_hint: AtomicBool,
    /// `LUMEN_MOE_TRACE`: records what every layer call asked for. `None` costs a check per call.
    trace: Option<trace::RouteTrace>,
    /// The slot-index arrays of graphs built before their routing is known, by layer and size.
    slots: Mutex<HashMap<(u16, usize), Arc<SlotIndices>>>,
}

/// Sums over the layer calls so far, split by who was working: the host
/// building the graph between two calls, the GPU (and the sync) inside the
/// routing readback, and the SSD inside `acquire`.
/// What the staging regions hold. A region is `loaded` once its reads are known to have
/// finished; the one load in flight is waited for by the call that needs its layer.
#[derive(Default)]
struct Staging {
    loaded: [Option<u16>; STAGING_REGIONS],
    inflight: Option<RegionLoad>,
}

#[derive(Default)]
struct Timing {
    calls: u64,
    /// From the end of the previous call to the start of this one: the host
    /// building lazy graphs while the GPU sits idle.
    build_ns: u64,
    /// `eval(inds)`: GPU work since the previous call plus the sync itself.
    wait_ns: u64,
    /// `acquire` on the calls that read: planning, reads, copies into slots.
    load_ns: u64,
    /// Calls that read at least one expert, and how many experts they read.
    loading_calls: u64,
    loaded_experts: u64,
    /// Layer calls served from a staging region, how many of those had to load their layer
    /// then and there (the first of a chunk), and the time spent waiting for reads that were
    /// meant to be done already (0 when the GPU outlasted the SSD).
    bulk_layers: u64,
    bulk_cold_layers: u64,
    bulk_wait_ns: u64,
    /// `LUMEN_MOE_OVERLAP`: calls whose reads overlapped with resident work, and per such call
    /// the time from the routing readback to that work's submission, how long the reads ran
    /// (submission to the last one finishing), and how long the host still waited for them
    /// after submitting. Plus the calls not overlapped because their reads looked short.
    overlap_calls: u64,
    overlap_submit_ns: u64,
    overlap_read_ns: u64,
    overlap_wait_ns: u64,
    overlap_short: u64,
    /// When the previous call returned.
    last_return: Option<std::time::Instant>,
}

impl Timing {
    fn line(&self, expert_bytes: u64, st: &slot_map::Stats) -> String {
        let per = |ns: u64| ns as f64 / self.calls.max(1) as f64 / 1e3;
        let load_us = self.load_ns as f64 / self.loading_calls.max(1) as f64 / 1e3;
        let prefetch = if self.bulk_layers == 0 {
            String::new()
        } else {
            format!(
                "; staged: {} layers ({} loaded on the spot), waited {:.0} us per layer for the reads",
                self.bulk_layers,
                self.bulk_cold_layers,
                self.bulk_wait_ns as f64 / self.bulk_layers as f64 / 1e3
            )
        };
        let overlap = if self.overlap_calls + self.overlap_short == 0 {
            String::new()
        } else {
            let per_overlap = |ns: u64| ns as f64 / self.overlap_calls.max(1) as f64 / 1e3;
            format!(
                "; overlapped {} calls ({:.0}% of those that read): resident work submitted \
                 {:.0} us after the readback, reads ran {:.0} us, then waited {:.0} us; \
                 {} calls not overlapped (reads expected under {OVERLAP_MIN_READ_US} us)",
                self.overlap_calls,
                100.0 * self.overlap_calls as f64 / self.loading_calls.max(1) as f64,
                per_overlap(self.overlap_submit_ns),
                per_overlap(self.overlap_read_ns),
                per_overlap(self.overlap_wait_ns),
                self.overlap_short,
            )
        };
        format!(
            "[offload] {} layer calls: hit rate {:.3} ({} hits, {} misses); per call: \
             host build {:.0} us, GPU wait {:.0} us, load {:.0} us; {:.0}% of calls read, \
             {:.0} us and {:.2} MB each{prefetch}{overlap}",
            self.calls,
            st.hit_rate(),
            st.hits,
            st.misses,
            per(self.build_ns),
            per(self.wait_ns),
            per(self.load_ns),
            100.0 * self.loading_calls as f64 / self.calls.max(1) as f64,
            load_us,
            self.loaded_experts as f64 * expert_bytes as f64
                / self.loading_calls.max(1) as f64
                / 1e6,
        )
    }
}

/// A layer call that needs at least this share of the layer's experts takes the staging path:
/// the whole layer is read into a region and gathered from a view of exactly that layer, and
/// the next layer is read into the other region meanwhile. A prefill chunk of a few thousand
/// tokens needs nearly all of them; a decode step, or a short prompt, a small fraction, and
/// reading a whole layer for those would be waste.
const BULK_MIN_PERCENT: usize = 60;

/// MLX's sorted `gather_qmm` (the prefill's) runs its tiled matrix-matrix kernel instead of the
/// per-row one once the call has at least this many index entries per expert **of the stacked
/// tensor it is given** (`B / E >= 4` in `GatherQMM::eval_gpu`). The two kernels accumulate and
/// round differently, so the output moves in its last bits with `E`. The resident model gathers
/// over one layer (`E` = 512), the staging path over a view of exactly one layer, and so do the
/// same bits; the pool path gathers over the whole pool (`E` = its slot count), which flips the
/// kernel at a pool-size-dependent number of rows. A call this large therefore always takes the
/// staging path, so its output is the resident model's whatever the pool size.
const TILED_GATHER_MIN_PER_EXPERT: usize = 4;

/// Whether a layer call takes the staging path instead of the pool path: a prefill feeding
/// several chunks through the layer (`hinted`), a call routing to most of the layer's experts
/// (`BULK_MIN_PERCENT`), or one big enough to change the gather kernel (see
/// `TILED_GATHER_MIN_PER_EXPERT`). `indices` is the call's `rows x top_k`.
fn takes_staging_path(hinted: bool, distinct: usize, indices: usize, experts: usize) -> bool {
    hinted
        || distinct * 100 >= experts * BULK_MIN_PERCENT
        || indices >= TILED_GATHER_MIN_PER_EXPERT * experts
}

/// A stats line every this many layer calls (about 125 decoded tokens of a
/// 40-layer model), so a long run shows how the hit rate settles.
const STATS_EVERY_CALLS: u64 = 5_000;

/// Hand one layer call to the routing trace: its distinct experts with the engine's hit flags,
/// and, for calls with more than a few rows, how many tokens chose each expert (all of them, and
/// the last `TAIL_ROWS`). Only runs when `LUMEN_MOE_TRACE` is set.
#[allow(clippy::too_many_arguments)]
fn trace_call(
    trace: &trace::RouteTrace,
    layer: u16,
    k: usize,
    ids: &[u32],
    index_of: &HashMap<u32, usize>,
    distinct: &[u16],
    hits: &[bool],
    staged: bool,
    hinted: bool,
    (entered, evaluated): (std::time::Instant, std::time::Instant),
) {
    let rows = (ids.len() / k) as u32;
    let counts = (rows > trace::DECODE_ROWS).then(|| {
        let tail_from = (rows.saturating_sub(trace::TAIL_ROWS) as usize) * k;
        let mut total = vec![0u16; distinct.len()];
        let mut tail = vec![0u16; distinct.len()];
        for (i, id) in ids.iter().enumerate() {
            let p = index_of[id];
            total[p] = total[p].saturating_add(1);
            if i >= tail_from {
                tail[p] = tail[p].saturating_add(1);
            }
        }
        (total, tail)
    });
    trace.record(&trace::Call {
        layer,
        rows,
        staged,
        hinted,
        ids: distinct,
        hits,
        counts: counts.as_ref().map(|(t, l)| (t.as_slice(), l.as_slice())),
        entered,
        evaluated,
        done: std::time::Instant::now(),
    });
}

impl MoeOffload {
    /// Build a bank over `dir` with room for about `budget_bytes` of experts.
    /// `top_k` is how many experts a token routes to per layer.
    pub(crate) fn open(dir: &Path, budget_bytes: u64, top_k: usize, io: IoTuning) -> Result<Self> {
        let store = expert_store::ExpertStore::open_with(dir, io.pack, io.mtp_experts)?;
        let per = store.expert_bytes().max(1) as u64;
        let total = store
            .layers
            .values()
            .map(|l| l.n_experts)
            .sum::<usize>()
            .max(MIN_SLOTS);
        let widest = store
            .layers
            .values()
            .map(|l| l.n_experts)
            .max()
            .unwrap_or(0);
        let n_slots = pool_slot_count(budget_bytes, per, total, widest);
        if n_slots as u64 * per > budget_bytes && n_slots < total {
            eprintln!(
                "[offload] the {:.2} GB budget is below the {:.2} GB the smallest pool needs \
                 (a staging region of {widest} experts plus one layer); using that.",
                budget_bytes as f64 / 1e9,
                n_slots as f64 * per as f64 / 1e9,
            );
        }
        // One token touches `layers x top_k` experts. A pool smaller than that
        // is scanned cyclically: every expert is evicted before the next token
        // asks for it again, so the hit rate is zero (measured: 101 slots on a
        // 40-layer top-8 model). The output is still exact, just never fast.
        let per_token = store.layers.len() * top_k;
        if n_slots < per_token {
            eprintln!(
                "[offload] WARNING: {n_slots} slots is fewer than the {per_token} experts one token \
                 touches ({} layers x top-{top_k}); nearly every read will be a miss. \
                 Raise LUMEN_MOE_EXPERT_CACHE_GB to at least {:.2}.",
                store.layers.len(),
                (per_token as f64 * per as f64 * 1.25) / 1e9,
            );
        }
        let bank = ExpertBank::open(
            dir,
            n_slots,
            io.lanes,
            io.mode,
            // Any pack this needs was built just above.
            if io.pack == PackMode::Off {
                PackMode::Off
            } else {
                PackMode::Auto
            },
            io.mtp_experts,
        )
        .with_context(|| format!("expert offload: bank over {}", dir.display()))?;
        let trace = trace::RouteTrace::from_env(|| trace::Header {
            budget_bytes,
            n_slots,
            regions: bank.regions(),
            width: widest,
            top_k,
            expert_bytes: per,
            bulk_min_percent: BULK_MIN_PERCENT,
            layers: store.layers.keys().copied().collect(),
        })?;
        eprintln!(
            "[offload] expert bank: {n_slots} slots x {:.2} MB = {:.2} GB resident of {:.2} GB of routed experts \
             ({} MoE layers, {} experts each), {} read lanes, {:?} reads",
            per as f64 / 1e6,
            n_slots as f64 * per as f64 / 1e9,
            total as f64 * per as f64 / 1e9,
            store.layers.len(),
            store.layers.values().next().map_or(0, |l| l.n_experts),
            io.lanes,
            io.mode,
        );
        Ok(Self {
            bank: Mutex::new(bank),
            pending: Mutex::new(None),
            read_ns_per_expert: AtomicU64::new(0),
            staging: Mutex::new(Staging::default()),
            timing: Mutex::new(Timing::default()),
            expert_bytes: per,
            bulk_hint: AtomicBool::new(false),
            trace,
            slots: Mutex::new(HashMap::new()),
        })
    }

    /// Route every layer call through the staging path (or stop doing so): see `bulk_hint`.
    pub(crate) fn set_bulk_hint(&self, on: bool) {
        self.bulk_hint.store(on, Ordering::Relaxed);
    }

    pub(crate) fn stats(&self) -> slot_map::Stats {
        self.bank.lock().unwrap_or_else(|p| p.into_inner()).stats()
    }

    /// Start the per-call time split over: the next [`Self::report`] covers only
    /// what happens after this (a benchmark resets it between prefill and decode).
    pub(crate) fn reset_timing(&self) {
        *self.timing.lock().unwrap_or_else(|p| p.into_inner()) = Timing::default();
    }

    /// The periodic stats line, on demand: hit rate, where a layer call's time
    /// went, and how the reads behaved.
    pub(crate) fn report(&self) -> String {
        let bank = self.bank.lock().unwrap_or_else(|p| p.into_inner());
        let t = self.timing.lock().unwrap_or_else(|p| p.into_inner());
        let mode = match bank.io_auto_state() {
            Some((direct, free)) => format!(
                "; reads {} ({:.1} GB available at the last look)",
                if direct { "direct" } else { "buffered" },
                free as f64 / 1e9
            ),
            None => String::new(),
        };
        let copied = match bank.copied_experts() {
            0 => String::new(),
            n => format!("; {n} staged experts copied from the pool instead of read"),
        };
        let paced = match bank.io_throttle_gbps() {
            Some(gbps) => format!("; reads paced to {gbps} GB/s (LUMEN_MOE_IO_THROTTLE_GBPS)"),
            None => String::new(),
        };
        format!(
            "{}\n{}{mode}{copied}{paced}",
            t.line(self.expert_bytes, &bank.stats()),
            bank.io_profile().summary()
        )
    }

    /// Back to cached decoding: finish the load in flight (it writes into the blocks about to
    /// be handed back) and return them to the slot map.
    fn leave_staging(&self, bank: &mut ExpertBank) -> Result<()> {
        let mut st = self.staging.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(load) = st.inflight.take() {
            bank.finish_region_load(load)?;
        }
        st.loaded = [None; STAGING_REGIONS];
        bank.leave_staging();
        Ok(())
    }

    /// Wait for the load in flight, if any, and record its region as loaded.
    fn settle_inflight(st: &mut Staging, bank: &mut ExpertBank) -> Result<()> {
        if let Some(load) = st.inflight.take() {
            let (layer, region) = (load.layer, load.region);
            bank.finish_region_load(load)?;
            st.loaded[region] = Some(layer);
        }
        Ok(())
    }

    /// The staging path: this layer's experts all sit in a region, expert `e` at slot `e`,
    /// so the gather runs over a view of exactly the layer and the indices stay expert ids.
    /// The next layer's reads start into the other region before this returns, and run while
    /// the GPU computes this layer (every chunk of it, in a layer-major prefill) and the first
    /// half of the next.
    fn route_bulk(
        &self,
        bank: &mut ExpertBank,
        layer: u16,
        inds: &Array,
        entered: std::time::Instant,
        evaluated: std::time::Instant,
    ) -> Result<Routed> {
        let mut st = self.staging.lock().unwrap_or_else(|p| p.into_inner());
        bank.enter_staging();
        let t_wait = std::time::Instant::now();
        // The load for this layer, started by an earlier call, may still be running.
        if st.inflight.as_ref().is_some_and(|l| l.layer == layer) {
            Self::settle_inflight(&mut st, bank)?;
        }
        let mut cold = false;
        let region = match st.loaded[..bank.regions()]
            .iter()
            .position(|l| *l == Some(layer))
        {
            Some(r) => r,
            None => {
                // Nothing staged for this layer (the first of a pass, or an earlier call
                // died mid-way): read it now, into a region not holding a layer still needed.
                cold = true;
                Self::settle_inflight(&mut st, bank)?;
                let r = st.loaded[..bank.regions()]
                    .iter()
                    .position(|l| l.is_none())
                    .unwrap_or(0);
                st.loaded[r] = None;
                let load = bank.begin_region_load(r, layer)?;
                bank.finish_region_load(load)?;
                st.loaded[r] = Some(layer);
                r
            }
        };
        let waited = t_wait.elapsed().as_nanos() as u64;
        // Read the next layer into the other region now, unless an earlier call of this layer
        // already did (a prefill feeds several chunks through one layer). That region held the
        // previous layer, whose gathers finished before this call's routing readback returned.
        // With a single region there is nothing to prefetch into: the next layer's first call
        // reads it on the spot.
        if bank.regions() > 1
            && let Some(next) = bank.layer_after(layer)
        {
            let other = 1 - region;
            let queued = st.loaded[other] == Some(next)
                || st.inflight.as_ref().is_some_and(|l| l.layer == next);
            if !queued {
                Self::settle_inflight(&mut st, bank)?;
                st.loaded[other] = None;
                st.inflight = Some(bank.begin_region_load(other, next)?);
            }
        }
        drop(st);

        {
            let now = std::time::Instant::now();
            let mut t = self.timing.lock().unwrap_or_else(|p| p.into_inner());
            t.calls += 1;
            if let Some(prev) = t.last_return {
                t.build_ns += entered.duration_since(prev).as_nanos() as u64;
            }
            t.wait_ns += evaluated.duration_since(entered).as_nanos() as u64;
            t.bulk_layers += 1;
            t.bulk_cold_layers += u64::from(cold);
            t.bulk_wait_ns += waited;
            let _ = now;
            t.last_return = Some(std::time::Instant::now());
        }
        Ok(Routed {
            slot_inds: inds.clone(),
            arrays: bank.region_arrays(region)?,
        })
    }

    /// The routed ids of an evaluated, flat routing array.
    fn read_ids(flat: &Array) -> Result<Vec<u32>> {
        Ok(match flat.dtype() {
            mlx_rs::Dtype::Uint32 => flat
                .try_as_slice::<u32>()
                .map_err(|e| anyhow!("offload route: read routing: {e:?}"))?
                .to_vec(),
            mlx_rs::Dtype::Int32 => flat
                .try_as_slice::<i32>()
                .map_err(|e| anyhow!("offload route: read routing: {e:?}"))?
                .iter()
                .map(|&v| v as u32)
                .collect(),
            other => bail!("offload route: routing ids have dtype {other:?}"),
        })
    }

    /// The pool arrays by role, as the gather takes them (cheap handle clones).
    fn pool_arrays(bank: &ExpertBank) -> Vec<(String, Array)> {
        bank.roles()
            .map(|role| {
                (
                    role.to_string(),
                    bank.pool(role).expect("role has a pool").array().clone(),
                )
            })
            .collect()
    }

    /// Make the experts `inds` routes to resident and return the routing in
    /// slot numbers plus the pool arrays to gather over.
    ///
    /// Reads `inds` back to the CPU, which waits for every GPU operation the
    /// routing depends on, that is, everything before this layer. That wait is
    /// what makes it safe to rewrite slots here: the previous layer's gather has
    /// finished, so its pins are dropped first.
    pub(crate) fn route(&self, layer: u16, inds: &Array) -> Result<Routed> {
        match self.route_with(layer, inds, None)? {
            Routing::Whole(routed) => Ok(routed),
            Routing::Split(_) => bail!("offload route: a call with no resident work was split"),
        }
    }

    /// [`Self::route`] that may overlap the call's reads with GPU work (`LUMEN_MOE_OVERLAP`): when
    /// the call qualifies (see [`overlaps`]), `resident` is called with the resident entries while
    /// the missing experts are being read, what it builds is submitted, and the reads are waited
    /// for only then. Otherwise `resident` is not called and the result is [`Routing::Whole`].
    pub(crate) fn route_overlapping(
        &self,
        layer: u16,
        inds: &Array,
        resident: &mut ResidentWork<'_>,
    ) -> Result<Routing> {
        self.route_with(layer, inds, Some(resident))
    }

    fn route_with(
        &self,
        layer: u16,
        inds: &Array,
        resident: Option<&mut ResidentWork<'_>>,
    ) -> Result<Routing> {
        let entered = std::time::Instant::now();
        // A strided view (`argpartition(..)[.., k..]`) reads back in physical order;
        // a flat reshape forces the logical order.
        let flat = mlx_rs::ops::reshape(inds, &[-1]).context("offload route: flatten routing")?;
        flat.eval().context("offload route: evaluate routing")?;
        let evaluated = std::time::Instant::now();
        let ids = Self::read_ids(&flat)?;
        let k = inds.shape().last().copied().unwrap_or(1).max(1) as usize;

        let mut bank = self.bank.lock().unwrap_or_else(|p| p.into_inner());
        let mut pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(prev) = pending.take() {
            prev.release(&mut bank);
        }
        let (distinct, index_of) = distinct_in_order(&ids)?;
        let hinted = self.bulk_hint.load(Ordering::Relaxed);
        if takes_staging_path(hinted, distinct.len(), ids.len(), bank.experts_in(layer)) {
            let routed = self.route_bulk(&mut bank, layer, inds, entered, evaluated)?;
            if let Some(trace) = &self.trace {
                // The call changed nothing in the pool map beyond borrowing the staging blocks
                // (done by now), so residency read here is what the call saw.
                let hits: Vec<bool> = distinct
                    .iter()
                    .map(|&e| bank.is_resident(layer, e))
                    .collect();
                trace_call(
                    trace,
                    layer,
                    k,
                    &ids,
                    &index_of,
                    &distinct,
                    &hits,
                    true,
                    hinted,
                    (entered, evaluated),
                );
            }
            return Ok(Routing::Whole(routed));
        }
        let call = PoolCall {
            layer,
            k,
            ids: &ids,
            index_of: &index_of,
            distinct: &distinct,
            hinted,
            entered,
            evaluated,
        };
        let placed = self.place_on_pool(&mut bank, &mut pending, &call, resident)?;
        let arrays = Self::pool_arrays(&bank);
        Ok(match placed {
            Placed::Whole(slot_ids) => Routing::Whole(Routed {
                slot_inds: Array::from_slice(&slot_ids, inds.shape()),
                arrays,
            }),
            Placed::Split {
                resident_out,
                parts,
            } => {
                let n = parts.missing_slots.len() as i32;
                Routing::Split(Split {
                    resident_out,
                    missing: Routed {
                        slot_inds: Array::from_slice(&parts.missing_slots, &[n]),
                        arrays,
                    },
                    missing_rows: Array::from_slice(&parts.missing_rows, &[n]),
                    order: Array::from_slice(&parts.order, &[parts.order.len() as i32]),
                })
            }
        })
    }

    /// The slot-index array a layer's gather reads when its graph is built before the routing
    /// is known, and the pool arrays to gather over. The graph holds the array; the CPU fills
    /// it in with [`Self::route_ahead`] once the routing has been read back, before the
    /// graph is evaluated. One array per layer and call shape, made on first use.
    pub(crate) fn routed_ahead(
        &self,
        layer: u16,
        shape: &[i32],
    ) -> Result<(Routed, Arc<SlotIndices>)> {
        let len = shape.iter().map(|&d| d.max(0) as usize).product::<usize>();
        let slots = {
            let mut table = self.slots.lock().unwrap_or_else(|p| p.into_inner());
            match table.get(&(layer, len)) {
                Some(s) => Arc::clone(s),
                None => {
                    let s = Arc::new(SlotIndices::new(shape)?);
                    table.insert((layer, len), Arc::clone(&s));
                    s
                }
            }
        };
        let bank = self.bank.lock().unwrap_or_else(|p| p.into_inner());
        let routed = Routed {
            slot_inds: slots.array().clone(),
            arrays: Self::pool_arrays(&bank),
        };
        Ok((routed, slots))
    }

    /// The second half of [`Self::route`] for a graph built ahead (see [`Self::routed_ahead`]):
    /// wait for the routing `flat` (the flattened `[.., k]` expert ids), make those experts
    /// resident and write their slots into `slots`. Pool path only: a call that would take
    /// the staging path has a different graph and is an error here.
    pub(crate) fn route_ahead(
        &self,
        layer: u16,
        flat: &Array,
        k: usize,
        slots: &SlotIndices,
    ) -> Result<()> {
        let entered = std::time::Instant::now();
        flat.eval().context("offload route: evaluate routing")?;
        let evaluated = std::time::Instant::now();
        let ids = Self::read_ids(flat)?;

        let mut bank = self.bank.lock().unwrap_or_else(|p| p.into_inner());
        let mut pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(prev) = pending.take() {
            prev.release(&mut bank);
        }
        let (distinct, index_of) = distinct_in_order(&ids)?;
        let hinted = self.bulk_hint.load(Ordering::Relaxed);
        if takes_staging_path(hinted, distinct.len(), ids.len(), bank.experts_in(layer)) {
            bail!("offload route: a call built ahead cannot take the staging path");
        }
        let call = PoolCall {
            layer,
            k,
            ids: &ids,
            index_of: &index_of,
            distinct: &distinct,
            hinted,
            entered,
            evaluated,
        };
        let Placed::Whole(slot_ids) = self.place_on_pool(&mut bank, &mut pending, &call, None)?
        else {
            bail!("offload route: a call built ahead was split");
        };
        // SAFETY: the readback above returned, so the segment that ended in this routing has
        // finished, and with it the previous gather over these slot numbers; the next graph
        // that reads them is submitted after this returns.
        unsafe { slots.write(&slot_ids) }
    }

    /// The pool path of a layer call: leave staging, make the call's distinct experts resident
    /// (reading the missing ones), hold their pins until the next call, account the time, and
    /// return the slot of every routed entry. With `resident` work and a call that qualifies
    /// ([`overlaps`]), that work is built and submitted while the reads run, and the entries
    /// come back split into resident and missing ones.
    fn place_on_pool(
        &self,
        bank: &mut ExpertBank,
        pending: &mut Option<Pinned>,
        call: &PoolCall<'_>,
        resident: Option<&mut ResidentWork<'_>>,
    ) -> Result<Placed> {
        self.leave_staging(bank)?;
        // Rows, not experts: a decode step or a short verify is a few rows, a prompt chunk many.
        let rows = call.ids.len() / call.k;
        let kind = if rows <= trace::DECODE_ROWS as usize {
            slot_map::CallKind::Decode
        } else {
            slot_map::CallKind::Prefill
        };
        let mut acquiring = bank.begin_acquire(call.layer, call.distinct, kind)?;
        let misses = acquiring.loaded_experts();
        let decision = match resident {
            Some(_) => overlaps(
                rows,
                call.distinct.len() - misses,
                misses,
                self.read_ns_per_expert.load(Ordering::Relaxed),
                overlap_min_read_ns(),
            ),
            None => Decision::Plain,
        };

        // While the reads run: build the resident entries' work and submit it. It reads only
        // slots the plan pinned as hits; the slots being written are reserved, so nothing reads
        // them until the commit below.
        let mut gpu: Vec<Array> = Vec::new();
        let mut overlapped = None;
        if let (Decision::Overlap, Some(work)) = (decision, resident) {
            let mut missing = vec![false; call.distinct.len()];
            for e in acquiring.missed() {
                missing[call.index_of[&u32::from(e)]] = true;
            }
            let parts = split_entries(call.ids, call.k, call.index_of, acquiring.slots(), &missing);
            let n = parts.resident_slots.len() as i32;
            let part = Routed {
                slot_inds: Array::from_slice(&parts.resident_slots, &[n]),
                arrays: Self::pool_arrays(bank),
            };
            let part_rows = Array::from_slice(&parts.resident_rows, &[n]);
            let submitted = work(&part, &part_rows).and_then(|out| {
                // Held before submitting, so even a failed submission is waited for.
                gpu = out.clone();
                mlx_rs::transforms::async_eval(out.iter())
                    .context("offload route: submit the resident experts")?;
                Ok(out)
            });
            match submitted {
                Ok(out) => overlapped = Some((out, parts, std::time::Instant::now())),
                Err(e) => {
                    Self::settle_failed(bank, pending, acquiring, gpu);
                    return Err(e);
                }
            }
        }
        let read_time = match acquiring.wait_reads() {
            Ok(took) => took,
            Err(e) => {
                // The resident work reads slots whose pins the abort drops.
                wait_for(&gpu);
                bank.abort(acquiring);
                return Err(e);
            }
        };
        let landed = std::time::Instant::now();
        let acquired = bank.commit(acquiring);
        if let Some(trace) = &self.trace {
            let missed: Vec<u16> = acquired.missed().collect();
            let hits: Vec<bool> = call.distinct.iter().map(|e| !missed.contains(e)).collect();
            trace_call(
                trace,
                call.layer,
                call.k,
                call.ids,
                call.index_of,
                call.distinct,
                &hits,
                false,
                call.hinted,
                (call.entered, call.evaluated),
            );
        }
        let loaded = acquired.loaded_experts();
        if loaded > 0 {
            self.note_read_time(read_time, loaded);
        }
        let submitted_at = overlapped.as_ref().map(|(_, _, at)| *at);
        let placed = match overlapped {
            None => Placed::Whole(
                call.ids
                    .iter()
                    .map(|id| acquired.slots[call.index_of[id]])
                    .collect(),
            ),
            Some((resident_out, parts, _)) => Placed::Split {
                resident_out,
                parts,
            },
        };
        *pending = Some(Pinned { acquired, gpu });
        {
            let now = std::time::Instant::now();
            let mut t = self.timing.lock().unwrap_or_else(|p| p.into_inner());
            t.calls += 1;
            if let Some(prev) = t.last_return {
                t.build_ns += call.entered.duration_since(prev).as_nanos() as u64;
            }
            t.wait_ns += call.evaluated.duration_since(call.entered).as_nanos() as u64;
            if loaded > 0 {
                t.load_ns += now.duration_since(call.evaluated).as_nanos() as u64;
                t.loading_calls += 1;
                t.loaded_experts += loaded as u64;
            }
            if let Some(submitted) = submitted_at {
                t.overlap_calls += 1;
                t.overlap_submit_ns += submitted.duration_since(call.evaluated).as_nanos() as u64;
                t.overlap_read_ns += read_time.as_nanos() as u64;
                t.overlap_wait_ns += landed.saturating_duration_since(submitted).as_nanos() as u64;
            }
            t.overlap_short += u64::from(decision == Decision::Short);
            if t.calls.is_multiple_of(STATS_EVERY_CALLS) {
                eprintln!(
                    "{}\n[offload] {}",
                    t.line(self.expert_bytes, &bank.stats()),
                    bank.io_profile().summary()
                );
            }
            t.last_return = Some(std::time::Instant::now());
        }
        Ok(placed)
    }

    /// A call that fails after its reads were submitted: wait for them and keep what landed
    /// pinned like any call's slots, or give the slots back once the GPU work reading the
    /// resident ones is done, so no pin or reservation outlives the error.
    fn settle_failed(
        bank: &mut ExpertBank,
        pending: &mut Option<Pinned>,
        mut acquiring: Acquiring,
        gpu: Vec<Array>,
    ) {
        match acquiring.wait_reads() {
            Ok(_) => {
                *pending = Some(Pinned {
                    acquired: bank.commit(acquiring),
                    gpu,
                })
            }
            Err(_) => {
                wait_for(&gpu);
                bank.abort(acquiring);
            }
        }
    }

    /// Fold one call's read time into the per-expert estimate (an average over about the last
    /// eight calls that read).
    fn note_read_time(&self, took: std::time::Duration, experts: usize) {
        let sample = (took.as_nanos() as u64 / experts.max(1) as u64).max(1);
        let old = self.read_ns_per_expert.load(Ordering::Relaxed);
        let new = if old == 0 {
            sample
        } else {
            (old * 7 + sample) / 8
        };
        self.read_ns_per_expert.store(new, Ordering::Relaxed);
    }
}

/// How a pool-path call's entries come back: every entry's slot in the routing's order, or
/// split into resident and missing parts with the resident work already submitted.
enum Placed {
    Whole(Vec<u32>),
    Split {
        resident_out: Vec<Array>,
        parts: Parts,
    },
}

/// What a pool-path layer call knows once the routing is read back.
struct PoolCall<'a> {
    layer: u16,
    /// Experts routed per row.
    k: usize,
    ids: &'a [u32],
    index_of: &'a HashMap<u32, usize>,
    distinct: &'a [u16],
    hinted: bool,
    entered: std::time::Instant,
    evaluated: std::time::Instant,
}

/// The distinct experts of `ids` in first-seen order, and each id's position among them.
fn distinct_in_order(ids: &[u32]) -> Result<(Vec<u16>, HashMap<u32, usize>)> {
    let mut distinct: Vec<u16> = Vec::with_capacity(ids.len());
    let mut index_of: HashMap<u32, usize> = HashMap::with_capacity(ids.len());
    for &id in ids {
        if let std::collections::hash_map::Entry::Vacant(v) = index_of.entry(id) {
            let e = u16::try_from(id)
                .map_err(|_| anyhow!("offload route: expert id {id} out of range"))?;
            v.insert(distinct.len());
            distinct.push(e);
        }
    }
    Ok((distinct, index_of))
}

/// A small `u32` array whose bytes the CPU rewrites between layer calls: the slot numbers one
/// layer's gather reads. A graph that holds it can be built before the routing it depends on
/// is known (see [`MoeOffload::routed_ahead`]).
///
/// Like [`slot_pool::PoolTensor`], the array is only ever an op *input* while this type holds
/// its own handle to it, so MLX never sees the buffer as donatable.
pub(crate) struct SlotIndices {
    array: Array,
    base: *mut u32,
    len: usize,
}

// SAFETY: `base` points into the buffer owned by `array`, which lives as long as the value;
// writes are serialized against GPU reads by the caller (see `write`).
unsafe impl Send for SlotIndices {}
unsafe impl Sync for SlotIndices {}

impl SlotIndices {
    fn new(shape: &[i32]) -> Result<Self> {
        let len = shape.iter().map(|&d| d.max(0) as usize).product::<usize>();
        if len == 0 {
            bail!("SlotIndices::new: empty shape {shape:?}");
        }
        let array = Array::from_slice(&vec![0u32; len], shape);
        array
            .eval()
            .map_err(|e| anyhow!("SlotIndices::new: eval: {e}"))?;
        // SAFETY: the array is evaluated, so its buffer exists.
        let base = unsafe { mlx_sys::mlx_array_data_uint8(array.as_ptr()) } as *mut u32;
        if base.is_null() {
            bail!("SlotIndices::new: evaluated array has no data pointer");
        }
        Ok(Self { array, base, len })
    }

    pub(crate) fn array(&self) -> &Array {
        &self.array
    }

    /// Overwrite the indices.
    ///
    /// # Safety
    /// No GPU work that reads this array may be running.
    unsafe fn write(&self, ids: &[u32]) -> Result<()> {
        if ids.len() != self.len {
            bail!(
                "SlotIndices::write: {} ids for an array of {}",
                ids.len(),
                self.len
            );
        }
        // SAFETY: in bounds by the length check; exclusive by the caller's promise.
        unsafe { std::ptr::copy_nonoverlapping(ids.as_ptr(), self.base, ids.len()) };
        Ok(())
    }
}

#[cfg(test)]
mod slot_indices_tests {
    use super::*;

    #[test]
    #[ignore = "needs a Metal device"]
    fn a_graph_built_before_the_write_reads_what_was_written_after() {
        let slots = SlotIndices::new(&[1, 1, 4]).unwrap();
        let table = Array::from_slice(&[10u32, 11, 12, 13, 14, 15, 16, 17], &[8]);
        // The graph exists (lazily) before the indices do.
        let picked = mlx_rs::ops::indexing::take_axis(&table, slots.array(), 0).unwrap();
        // SAFETY: nothing has evaluated a graph over the array yet.
        unsafe { slots.write(&[7, 0, 3, 3]) }.unwrap();
        picked.eval().unwrap();
        assert_eq!(picked.as_slice::<u32>(), &[17, 10, 13, 13]);
        // Rewritten for the next call; a new graph over the same array sees the new values.
        // SAFETY: the previous graph has been evaluated and dropped.
        unsafe { slots.write(&[1, 2, 5, 6]) }.unwrap();
        let again = mlx_rs::ops::indexing::take_axis(&table, slots.array(), 0).unwrap();
        again.eval().unwrap();
        assert_eq!(again.as_slice::<u32>(), &[11, 12, 15, 16]);
        assert!(unsafe { slots.write(&[1, 2, 3]) }.is_err(), "wrong length");
    }
}

#[cfg(test)]
mod overlap_tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn entries_split_by_residency_and_the_order_restores_the_routing() {
        // Two rows of three: 7 5 9 | 5 2 7, with 5 and 2 missing.
        let ids = [7u32, 5, 9, 5, 2, 7];
        let (distinct, index_of) = distinct_in_order(&ids).unwrap();
        assert_eq!(distinct, [7, 5, 9, 2]);
        let slots = [70usize, 50, 90, 20];
        let missing = [false, true, false, true];
        let p = split_entries(&ids, 3, &index_of, &slots, &missing);
        assert_eq!(p.resident_slots, [70, 90, 70]);
        assert_eq!(p.resident_rows, [0, 0, 1]);
        assert_eq!(p.missing_slots, [50, 50, 20]);
        assert_eq!(p.missing_rows, [0, 1, 1]);
        assert_eq!(p.order, [0, 3, 1, 4, 5, 2]);
        let stacked: Vec<(u32, u32)> = p
            .resident_slots
            .iter()
            .zip(&p.resident_rows)
            .chain(p.missing_slots.iter().zip(&p.missing_rows))
            .map(|(&s, &r)| (s, r))
            .collect();
        for (i, &o) in p.order.iter().enumerate() {
            let want = (slots[index_of[&ids[i]]] as u32, (i / 3) as u32);
            assert_eq!(stacked[o as usize], want, "entry {i}");
        }
    }

    #[test]
    fn only_decode_sized_calls_that_hit_and_miss_with_long_reads_overlap() {
        let min = OVERLAP_MIN_READ_US * 1_000;
        assert_eq!(
            overlaps(1, 6, 4, 0, min),
            Decision::Overlap,
            "no estimate yet"
        );
        assert_eq!(overlaps(1, 6, 4, 400_000, min), Decision::Overlap);
        assert_eq!(overlaps(1, 9, 1, 400_000, min), Decision::Short);
        assert_eq!(overlaps(4, 30, 10, 400_000, min), Decision::Overlap);
        assert_eq!(overlaps(1, 9, 1, 400_000, 0), Decision::Overlap);
        assert_eq!(
            overlaps(5, 30, 10, 400_000, min),
            Decision::Plain,
            "prefill-sized"
        );
        assert_eq!(
            overlaps(1, 10, 0, 400_000, min),
            Decision::Plain,
            "nothing to read"
        );
        assert_eq!(
            overlaps(1, 0, 10, 400_000, min),
            Decision::Plain,
            "nothing resident"
        );
    }

    fn tiny_offload() -> (MoeOffload, Vec<u16>) {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4exp_tiny");
        let io = IoTuning {
            mode: IoMode::Direct,
            ..IoTuning::default()
        };
        // A budget of one byte: the smallest pool, two layers' worth (32 slots of 16 experts).
        let off = MoeOffload::open(&dir, 1, 4, io).expect("open the replica's experts");
        let layers = {
            let bank = off.bank.lock().unwrap();
            bank.store()
                .layers
                .keys()
                .copied()
                .filter(|&l| l != MTP_LAYER)
                .collect()
        };
        (off, layers)
    }

    fn routing(ids: &[u32]) -> Array {
        Array::from_slice(ids, &[1, 1, ids.len() as i32])
    }

    fn is_available(a: &Array) -> bool {
        let mut ready = false;
        // SAFETY: a plain query on a live array.
        unsafe { mlx_sys::_mlx_array_is_available(&mut ready, a.as_ptr()) };
        ready
    }

    /// The resident work an overlapped call submits reads slots of the pool while the call has
    /// already returned. Those slots stay pinned until the next call, which waits for that work
    /// before unpinning: however many experts other calls load meanwhile, none of them is evicted.
    #[test]
    fn slots_read_by_resident_work_in_flight_stay_pinned_until_it_is_done() {
        let (off, layers) = tiny_offload();
        let (l0, l1, l2) = (layers[0], layers[1], layers[2]);
        drop(off.route(l0, &routing(&[0, 1, 2, 3])).expect("first call"));

        // Experts 0 and 1 are resident, 4 and 5 are read. The resident work gathers from the
        // pool at the resident slots and then keeps the GPU busy for a while.
        let mut seen: Vec<u32> = Vec::new();
        let mut work = |part: &Routed, _rows: &Array| -> Result<Vec<Array>> {
            part.slot_inds.eval()?;
            seen.extend_from_slice(part.slot_inds.as_slice::<u32>());
            let w = part.array("gate_proj.weight").expect("pool role");
            let picked = mlx_rs::ops::indexing::take_axis(w, &part.slot_inds, 0)?
                .as_dtype(mlx_rs::Dtype::Float32)?
                .sum(None)?;
            let m = mlx_rs::random::normal::<f32>(&[2048, 2048], None, None, None)?;
            let mut acc = m.clone();
            for _ in 0..48 {
                acc = mlx_rs::ops::matmul(&acc, &m)?;
            }
            Ok(vec![mlx_rs::ops::add(&picked, &acc.sum(None)?)?])
        };
        let routed = test_overlap::with_any_read(|| {
            off.route_overlapping(l0, &routing(&[0, 4, 1, 5]), &mut work)
        })
        .expect("overlapped call");
        let Routing::Split(split) = routed else {
            panic!("a call that hits and misses should overlap");
        };
        assert_eq!(seen.len(), 2, "two resident entries");
        let in_flight = split.resident_out[0].clone();
        eprintln!(
            "resident work still running when the call returned: {}",
            !is_available(&in_flight)
        );
        {
            let mut bank = off.bank.lock().unwrap();
            // 32 slots, 4 pinned by the overlapped call: other layers can take the other 28 ...
            let a = bank
                .acquire(l1, &(0..16).collect::<Vec<u16>>())
                .expect("16 of l1");
            let b = bank
                .acquire(l2, &(0..12).collect::<Vec<u16>>())
                .expect("12 of l2");
            // ... and nothing more, and the in-flight call's experts are all still there.
            assert!(bank.acquire(l2, &[12]).is_err(), "every slot is pinned now");
            for e in [0u16, 1, 4, 5] {
                assert!(
                    bank.is_resident(l0, e),
                    "expert {e} of the in-flight call was evicted"
                );
            }
            bank.release(a);
            bank.release(b);
            // Unpin as the next call does, but with no routing readback in between (that would
            // sync the GPU stream anyway): the release itself must wait for the work.
            let pinned = off.pending.lock().unwrap().take().expect("the call's pins");
            pinned.release(&mut bank);
            assert!(
                is_available(&in_flight),
                "the pins went before the work reading them was done"
            );
        }
        drop(off.route(l1, &routing(&[0, 1, 2, 3])).expect("next call"));
    }

    /// A call whose resident work fails to build leaves no pin and no reservation behind: once the
    /// next call has released what it kept, every slot but that call's can be taken.
    #[test]
    fn a_failed_overlapped_call_leaks_no_slot() {
        let (off, layers) = tiny_offload();
        let (l0, l1, l2, l3) = (layers[0], layers[1], layers[2], layers[3]);
        drop(off.route(l0, &routing(&[0, 1, 2, 3])).expect("first call"));
        let failed = test_overlap::with_any_read(|| {
            off.route_overlapping(l0, &routing(&[0, 6, 1, 7]), &mut |_, _| bail!("boom"))
        });
        assert!(failed.is_err());
        drop(off.route(l1, &routing(&[0, 1, 2, 3])).expect("next call"));
        let mut bank = off.bank.lock().unwrap();
        let a = bank
            .acquire(l2, &(0..16).collect::<Vec<u16>>())
            .expect("16 of l2");
        let b = bank
            .acquire(l3, &(0..12).collect::<Vec<u16>>())
            .expect("12 of l3");
        bank.release(a);
        bank.release(b);
    }
}
