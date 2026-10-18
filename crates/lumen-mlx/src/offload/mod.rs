//! SSD expert offload: keep a fixed number of a MoE layer's experts resident in
//! GPU-visible slots and stream the rest from disk (task 023).
//!
//! The pieces build on each other:
//!
//! * [`expert_store`] — where each routed expert's bytes live in the original
//!   safetensors shards (no repacking, no extra disk).
//! * [`io`] — a thread pool that reads those byte ranges in parallel into
//!   private aligned buffers and copies them into slot memory.
//! * [`expert_bank`] — the four pieces joined: `acquire(layer, experts)` makes
//!   experts resident and returns slot numbers for the gather.
//! * [`slot_pool`] — stacked tensors whose slots the CPU can overwrite in place.
//! * [`slot_map`] — which `(layer, expert)` lives in which slot: CLOCK
//!   eviction, pins, and the plan → load → commit protocol.
//!
//! [`MoeOffload`] is what the model holds: every MoE layer calls
//! [`MoeOffload::route`] between computing its routing and gathering.

pub(crate) mod expert_bank;
pub(crate) mod expert_store;
pub(crate) mod io;
pub(crate) mod pack;
pub(crate) mod slot_map;
pub(crate) mod slot_pool;

#[cfg(test)]
mod exactness_tests;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, anyhow, bail};
use mlx_rs::Array;

pub(crate) use expert_bank::{Acquired, ExpertBank, RegionLoad, STAGING_REGIONS};
pub(crate) use expert_store::MTP_LAYER;
pub(crate) use expert_store::read_header;
pub(crate) use io::{IoMode, IoPool, ReadJob};
pub(crate) use pack::PackMode;

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

/// A checkpoint's routed experts, served from a slot pool instead of RAM.
///
/// Held by the model; every MoE layer calls [`Self::route`] between computing
/// its routing and gathering. See the module docs for how the pieces fit.
pub(crate) struct MoeOffload {
    bank: Mutex<ExpertBank>,
    /// The slots of the previous layer call, still pinned: that call's gather
    /// is only known to be finished once the next routing readback returns.
    pending: Mutex<Option<Acquired>>,
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
        format!(
            "[offload] {} layer calls: hit rate {:.3} ({} hits, {} misses); per call: \
             host build {:.0} us, GPU wait {:.0} us, load {:.0} us; {:.0}% of calls read, \
             {:.0} us and {:.2} MB each{prefetch}",
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
            staging: Mutex::new(Staging::default()),
            timing: Mutex::new(Timing::default()),
            expert_bytes: per,
            bulk_hint: AtomicBool::new(false),
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
        format!(
            "{}\n{}{mode}{copied}",
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

    /// Make the experts `inds` routes to resident and return the routing in
    /// slot numbers plus the pool arrays to gather over.
    ///
    /// Reads `inds` back to the CPU, which waits for every GPU operation the
    /// routing depends on, that is, everything before this layer. That wait is
    /// what makes it safe to rewrite slots here: the previous layer's gather has
    /// finished, so its pins are dropped first.
    pub(crate) fn route(&self, layer: u16, inds: &Array) -> Result<Routed> {
        let entered = std::time::Instant::now();
        // A strided view (`argpartition(..)[.., k..]`) reads back in physical order;
        // a flat reshape forces the logical order.
        let flat = mlx_rs::ops::reshape(inds, &[-1]).context("offload route: flatten routing")?;
        flat.eval().context("offload route: evaluate routing")?;
        let evaluated = std::time::Instant::now();
        let ids: Vec<u32> = match flat.dtype() {
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
        };

        let mut bank = self.bank.lock().unwrap_or_else(|p| p.into_inner());
        let mut pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(prev) = pending.take() {
            bank.release(prev);
        }
        // Distinct experts, in first-seen order.
        let mut distinct: Vec<u16> = Vec::with_capacity(ids.len());
        let mut index_of: HashMap<u32, usize> = HashMap::with_capacity(ids.len());
        for &id in &ids {
            if let std::collections::hash_map::Entry::Vacant(v) = index_of.entry(id) {
                let e = u16::try_from(id)
                    .map_err(|_| anyhow!("offload route: expert id {id} out of range"))?;
                v.insert(distinct.len());
                distinct.push(e);
            }
        }
        let hinted = self.bulk_hint.load(Ordering::Relaxed);
        if takes_staging_path(hinted, distinct.len(), ids.len(), bank.experts_in(layer)) {
            return self.route_bulk(&mut bank, layer, inds, entered, evaluated);
        }
        self.leave_staging(&mut bank)?;
        let acquired = bank.acquire(layer, &distinct)?;
        let loaded = acquired.loaded_experts();
        let slot_ids: Vec<u32> = ids.iter().map(|id| acquired.slots[index_of[id]]).collect();
        *pending = Some(acquired);
        {
            let now = std::time::Instant::now();
            let mut t = self.timing.lock().unwrap_or_else(|p| p.into_inner());
            t.calls += 1;
            if let Some(prev) = t.last_return {
                t.build_ns += entered.duration_since(prev).as_nanos() as u64;
            }
            t.wait_ns += evaluated.duration_since(entered).as_nanos() as u64;
            if loaded > 0 {
                t.load_ns += now.duration_since(evaluated).as_nanos() as u64;
                t.loading_calls += 1;
                t.loaded_experts += loaded as u64;
            }
            if t.calls.is_multiple_of(STATS_EVERY_CALLS) {
                eprintln!(
                    "{}\n[offload] {}",
                    t.line(self.expert_bytes, &bank.stats()),
                    bank.io_profile().summary()
                );
            }
            t.last_return = Some(std::time::Instant::now());
        }

        let slot_inds = Array::from_slice(&slot_ids, inds.shape());
        let arrays = bank
            .roles()
            .map(|role| {
                (
                    role.to_string(),
                    bank.pool(role).expect("role has a pool").array().clone(),
                )
            })
            .collect();
        Ok(Routed { slot_inds, arrays })
    }
}
