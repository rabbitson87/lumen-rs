//! The expert bank: a pool of resident experts over one checkpoint.
//!
//! Joins the four pieces: [`ExpertStore`] knows where every expert's bytes are
//! in the shards, [`IoPool`] reads them, [`PoolTensor`]s (one per tensor role)
//! hold them where the GPU can read them, and [`SlotMap`] decides who lives
//! where. [`ExpertBank::begin_acquire`] and [`ExpertBank::commit`] make a layer's routed
//! experts resident and return their slot numbers; the gather then runs over the
//! pools with those slots as its indices.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use mlx_rs::Dtype;

use super::expert_store::{ExpertStore, MTP_LAYER};
use super::io::{IoMode, IoPool, IoProfile, IoTicket, ReadJob};
use super::pack::PackMode;
use super::slot_map::{CallKind, EvictionKind, ExpertKey, Plan, PolicyConfig, SlotMap, Stats};
use super::slot_pool::PoolTensor;

/// The slots of one `acquire`, pinned until [`ExpertBank::release`].
#[must_use = "pinned slots are only freed by ExpertBank::release"]
pub(crate) struct Acquired {
    plan: Plan,
    /// Slot of each requested expert, in request order.
    pub slots: Vec<u32>,
}

impl Acquired {
    /// How many of the requested experts had to be read from disk.
    pub(crate) fn loaded_experts(&self) -> usize {
        self.plan.loads.len()
    }

    /// The requested experts that were not resident: the misses (for the routing trace).
    pub(crate) fn missed(&self) -> impl Iterator<Item = u16> + '_ {
        self.plan.loads.iter().map(|&(key, _)| key.expert)
    }
}

/// An [`ExpertBank::begin_acquire`] whose reads may still be running: every requested expert
/// has its slot and is pinned, the resident ones can be read by the GPU now, the missing ones
/// only after [`Self::wait_reads`]. Hand it back with [`ExpertBank::commit`] or
/// [`ExpertBank::abort`].
#[must_use = "an acquire in flight holds pins and reserved slots until it is committed or aborted"]
pub(crate) struct Acquiring {
    plan: Plan,
    ticket: Option<IoTicket>,
    layer: u16,
    submitted: std::time::Instant,
}

impl Acquiring {
    /// Slot of each requested expert, in request order.
    pub(crate) fn slots(&self) -> &[usize] {
        &self.plan.slots
    }

    pub(crate) fn loaded_experts(&self) -> usize {
        self.plan.loads.len()
    }

    /// The requested experts being read: the misses.
    pub(crate) fn missed(&self) -> impl Iterator<Item = u16> + '_ {
        self.plan.loads.iter().map(|&(key, _)| key.expert)
    }

    /// Wait for the reads and return how long they ran (submission to the last one finishing;
    /// zero when nothing was read). Call once; an error leaves the acquire to be aborted.
    pub(crate) fn wait_reads(&mut self) -> Result<std::time::Duration> {
        let Some(ticket) = self.ticket.take() else {
            return Ok(std::time::Duration::ZERO);
        };
        let finished = ticket.wait_finished().with_context(|| {
            format!(
                "expert bank: loading layer {} experts {:?}",
                self.layer,
                self.missed().collect::<Vec<_>>()
            )
        })?;
        Ok(finished.map_or(std::time::Duration::ZERO, |t| {
            t.saturating_duration_since(self.submitted)
        }))
    }
}

/// One layer's experts being read into a staging region while the GPU does other work.
#[must_use = "a started load must be finished (dropping it waits for the reads)"]
pub(crate) struct RegionLoad {
    pub layer: u16,
    pub region: usize,
    ticket: IoTicket,
}

pub(crate) struct ExpertBank {
    store: ExpertStore,
    io: IoPool,
    /// One stacked tensor per role, all with the same slot count.
    pools: BTreeMap<String, PoolTensor>,
    map: SlotMap,
    n_slots: usize,
    /// Slots in a staging region: the widest layer's expert count. The two regions are the
    /// last `2 * width` slots of the pool.
    width: usize,
    /// How many staging regions the pool lends (1 or [`STAGING_REGIONS`]).
    regions: usize,
    /// Whether the staging blocks are borrowed from the map right now.
    staging: bool,
    /// Experts a staging load copied from the pool instead of reading from the SSD.
    copied: u64,
}

/// Staging regions. A prefill chunk routes to nearly every expert of every layer, so it needs
/// the whole layer resident, once, in order: layer L is gathered from one region while layer
/// L+1 is read into the other. They are blocks of the pool itself, lent by the slot map while a
/// prefill runs and handed back (empty) when decoding resumes, so they cost no memory of their
/// own and the experts outside them keep their place through a prefill.
pub(crate) const STAGING_REGIONS: usize = 2;

/// Regions a pool of `n_slots` can lend: two when there is room for them and a layer's worth of
/// slots beside them, else one (a layer is then read only after the previous one is done with,
/// so the reads are not hidden behind compute, but the pool can be a third smaller).
pub(crate) fn regions_for(n_slots: usize, width: usize) -> usize {
    if n_slots >= (STAGING_REGIONS + 1) * width {
        STAGING_REGIONS
    } else {
        1
    }
}

fn mlx_dtype(safetensors: &str) -> Result<Dtype> {
    Ok(match safetensors {
        "U8" => Dtype::Uint8,
        "U16" => Dtype::Uint16,
        "U32" => Dtype::Uint32,
        "I8" => Dtype::Int8,
        "I16" => Dtype::Int16,
        "I32" => Dtype::Int32,
        "F16" => Dtype::Float16,
        "BF16" => Dtype::Bfloat16,
        "F32" => Dtype::Float32,
        other => bail!("expert bank: unsupported safetensors dtype {other}"),
    })
}

impl ExpertBank {
    /// Open the shards of `dir` and allocate `n_slots` slots per role. The pool must hold the
    /// `STAGING_REGIONS` staging regions (widest layer each) and should have room beside them.
    pub(crate) fn open(
        dir: &Path,
        n_slots: usize,
        lanes: usize,
        mode: IoMode,
        pack: PackMode,
        with_mtp: bool,
    ) -> Result<Self> {
        let store = ExpertStore::open_with(dir, pack, with_mtp)?;
        let first = store
            .layers
            .values()
            .next()
            .ok_or_else(|| anyhow!("expert bank: {} has no routed experts", dir.display()))?;
        let width = store
            .layers
            .values()
            .map(|l| l.n_experts)
            .max()
            .unwrap_or(0);
        // One staging region and a layer's worth of slots beside it is the least that works.
        if n_slots < 2 * width {
            bail!(
                "expert bank: {n_slots} slots cannot hold a staging region of {width} and a layer beside it"
            );
        }
        let mut pools = BTreeMap::new();
        for (role, r) in &first.roles {
            let tail: Vec<i32> = r.tail_shape.iter().map(|&d| d as i32).collect();
            pools.insert(
                role.clone(),
                PoolTensor::new(n_slots, &tail, mlx_dtype(&r.dtype)?)
                    .with_context(|| format!("expert bank: pool for {role}"))?,
            );
        }
        let io = IoPool::new(&store.files, lanes, mode)?;
        io.throttle_from_env()?;
        let eviction = EvictionKind::from_env()?;
        eprintln!("[offload] eviction policy: {}", eviction.name());
        let policy = PolicyConfig::new(
            eviction,
            width,
            store.layers.keys().filter(|&&l| l != MTP_LAYER).count(),
        );
        Ok(Self {
            store,
            io,
            pools,
            map: SlotMap::with_policy(n_slots, &policy),
            n_slots,
            width,
            regions: regions_for(n_slots, width),
            staging: false,
            copied: 0,
        })
    }

    pub(crate) fn pool(&self, role: &str) -> Option<&PoolTensor> {
        self.pools.get(role)
    }

    pub(crate) fn roles(&self) -> impl Iterator<Item = &str> {
        self.pools.keys().map(String::as_str)
    }

    #[cfg(test)]
    pub(crate) fn store(&self) -> &ExpertStore {
        &self.store
    }

    /// `(direct, free bytes)` when the reads pick their mode by free memory.
    pub(crate) fn io_auto_state(&self) -> Option<(bool, u64)> {
        self.io.auto_state()
    }

    /// Experts staging loads took from the pool's resident ones rather than the SSD.
    pub(crate) fn copied_experts(&self) -> u64 {
        self.copied
    }

    pub(crate) fn io_profile(&self) -> &IoProfile {
        self.io.profile()
    }

    /// The rate `LUMEN_MOE_IO_THROTTLE_GBPS` paces the reads to, if set.
    pub(crate) fn io_throttle_gbps(&self) -> Option<f64> {
        self.io.throttle_gbps()
    }

    pub(crate) fn stats(&self) -> Stats {
        self.map.stats()
    }

    /// Whether `(layer, expert)` is in the pool right now. Does not pin or touch it.
    pub(crate) fn is_resident(&self, layer: u16, expert: u16) -> bool {
        self.map.peek(ExpertKey::new(layer, expert)).is_some()
    }

    /// Experts in `layer` (0 when it has none).
    pub(crate) fn experts_in(&self, layer: u16) -> usize {
        self.store.layer(layer).map_or(0, |l| l.n_experts)
    }

    /// The MoE layer after `layer`, if any.
    pub(crate) fn layer_after(&self, layer: u16) -> Option<u16> {
        self.store
            .layers
            .range(layer.saturating_add(1)..)
            .map(|(&l, _)| l)
            .find(|&l| l != MTP_LAYER)
    }

    fn region_base(&self, region: usize) -> usize {
        self.n_slots - (self.regions - region) * self.width
    }

    /// How many staging regions this pool lends: with one, a layer's reads cannot start before
    /// the layer before it is done, so nothing is prefetched.
    pub(crate) fn regions(&self) -> usize {
        self.regions
    }

    /// Borrow the staging blocks from the slot map: whoever lives there is evicted. Idempotent.
    pub(crate) fn enter_staging(&mut self) {
        if !self.staging {
            let start = self.region_base(0);
            self.map.reserve_range(start..self.n_slots);
            self.staging = true;
        }
    }

    /// Hand the staging blocks back (free and empty). The caller must have finished any load
    /// in flight into them. Idempotent.
    pub(crate) fn leave_staging(&mut self) {
        if self.staging {
            let start = self.region_base(0);
            self.map.release_range(start..self.n_slots);
            self.staging = false;
        }
    }

    /// Start reading every expert of `layer` into staging `region`, expert `e` at slot
    /// `e` of the region, and return at once.
    ///
    /// The safety condition is [`Self::begin_acquire`]'s: no queued GPU work may read the region.
    /// The routing readback that precedes every layer call completes all earlier GPU work,
    /// which includes the gather that last used this region.
    pub(crate) fn begin_region_load(&mut self, region: usize, layer: u16) -> Result<RegionLoad> {
        assert!(region < self.regions, "no staging region {region}");
        assert!(self.staging, "staging regions are not borrowed");
        let ranges = self
            .store
            .layer(layer)
            .ok_or_else(|| anyhow!("expert bank: layer {layer} has no routed experts"))?;
        let base = self.region_base(region);
        let mut jobs = Vec::with_capacity(ranges.n_experts * self.pools.len());
        // Experts the pool already holds are copied across (memory to memory, ~50 us each)
        // instead of read again from the SSD (~500 us each).
        let mut copies: Vec<(usize, usize)> = Vec::new();
        for e in 0..ranges.n_experts {
            if let Some(slot) = self.map.peek(ExpertKey::new(layer, e as u16)) {
                copies.push((slot, base + e));
                continue;
            }
            for (role, range) in &ranges.roles {
                let pool = &self.pools[role];
                debug_assert_eq!(pool.slot_bytes(), range.bytes);
                jobs.push(ReadJob::single(
                    range.file,
                    range.offset_of(e),
                    range.bytes,
                    // SAFETY (pointer only): written by the IO pool under its contract.
                    unsafe { pool.slot_ptr(base + e) },
                ));
            }
        }
        // SAFETY: every destination is its own slot of the region, which nothing else uses or
        // reads while the load is in flight, and the ticket lives in the returned `RegionLoad`
        // (dropping it waits).
        let ticket = unsafe { self.io.submit(jobs) }
            .with_context(|| format!("expert bank: starting the load of layer {layer}"))?;
        // While the reads run: the copies. A source slot is a resident expert of the main pool,
        // outside the borrowed blocks, which no queued GPU work writes (the GPU never writes
        // the pool) and the CPU writes only for a decode-path load, never during staging.
        self.copied += copies.len() as u64;
        for (from, to) in copies {
            for pool in self.pools.values() {
                // SAFETY: distinct slots of one live pool; `to` lies in the region.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        pool.slot_ptr(from),
                        pool.slot_ptr(to),
                        pool.slot_bytes(),
                    );
                }
            }
        }
        Ok(RegionLoad {
            layer,
            region,
            ticket,
        })
    }

    /// Wait for a region load; on success the region holds the layer.
    pub(crate) fn finish_region_load(&mut self, load: RegionLoad) -> Result<()> {
        let layer = load.layer;
        load.ticket
            .wait()
            .with_context(|| format!("expert bank: loading layer {layer} into staging"))
    }

    /// The pool arrays by role, restricted to staging `region`: stacked tensors of exactly
    /// `region_width` slots whose slot `e` holds expert `e` of the layer loaded there.
    pub(crate) fn region_arrays(&self, region: usize) -> Result<Vec<(String, mlx_rs::Array)>> {
        let base = self.region_base(region);
        self.pools
            .iter()
            .map(|(role, pool)| Ok((role.clone(), pool.view(base, self.width)?)))
            .collect()
    }

    fn jobs_for(&self, plan: &Plan, layer: u16) -> Vec<ReadJob> {
        let ranges = self
            .store
            .layer(layer)
            .expect("layer checked by the caller");
        let mut jobs = Vec::with_capacity(plan.loads.len() * self.pools.len());
        for &(key, slot) in &plan.loads {
            for (role, range) in &ranges.roles {
                let pool = &self.pools[role];
                debug_assert_eq!(pool.slot_bytes(), range.bytes);
                jobs.push(ReadJob::single(
                    range.file,
                    range.offset_of(key.expert as usize),
                    range.bytes,
                    // SAFETY (pointer only): written by the IO pool under its contract.
                    unsafe { pool.slot_ptr(slot) },
                ));
            }
        }
        jobs
    }

    /// Make `experts` of `layer` resident and pin them, for a decode call: [`Self::begin_acquire`],
    /// wait, then [`Self::commit`] (or [`Self::abort`] on a failed read).
    #[cfg(test)]
    pub(crate) fn acquire(&mut self, layer: u16, experts: &[u16]) -> Result<Acquired> {
        let mut acquiring = self.begin_acquire(layer, experts, CallKind::Decode)?;
        match acquiring.wait_reads() {
            Ok(_) => Ok(self.commit(acquiring)),
            Err(e) => {
                self.abort(acquiring);
                Err(e)
            }
        }
    }

    /// Start making `experts` of `layer` resident: plan the call (pin the resident experts,
    /// reserve a slot for each missing one) and submit the missing ones' reads in parallel
    /// without waiting for them. Nothing is published until [`Self::commit`], which needs every
    /// read to have succeeded. `kind` is what a learning eviction policy needs to tell the
    /// token-to-token routing of decoding from a prompt chunk's.
    ///
    /// The caller must not have any GPU work queued that reads a slot this call may rewrite: in
    /// the serving loop that holds because the per-layer routing readback has just completed
    /// everything before it. The resident experts' slots are not written, so GPU work may read
    /// them while the reads run.
    pub(crate) fn begin_acquire(
        &mut self,
        layer: u16,
        experts: &[u16],
        kind: CallKind,
    ) -> Result<Acquiring> {
        let ranges = self
            .store
            .layer(layer)
            .ok_or_else(|| anyhow!("expert bank: layer {layer} has no routed experts"))?;
        if let Some(&bad) = experts.iter().find(|&&e| e as usize >= ranges.n_experts) {
            bail!(
                "expert bank: expert {bad} out of {} in layer {layer}",
                ranges.n_experts
            );
        }
        let keys: Vec<ExpertKey> = experts.iter().map(|&e| ExpertKey::new(layer, e)).collect();
        let plan = self.map.plan_as(&keys, kind).map_err(|e| anyhow!("{e}"))?;

        let jobs = self.jobs_for(&plan, layer);
        let submitted = std::time::Instant::now();
        // SAFETY: each destination is a distinct slot of a live pool, reserved by the plan
        // (unmapped and pinned, so no lookup returns it and no GPU work is given it until the
        // commit), and the caller guarantees no queued GPU work reads it. The ticket lives in
        // the returned `Acquiring`, which waits for it before the slots are published or given
        // back (and dropping a ticket waits).
        match unsafe { self.io.submit(jobs) } {
            Ok(ticket) => Ok(Acquiring {
                plan,
                ticket: Some(ticket),
                layer,
                submitted,
            }),
            Err(e) => {
                self.map.abort(&plan);
                Err(e.context(format!(
                    "expert bank: loading layer {layer} experts {experts:?}"
                )))
            }
        }
    }

    /// Publish an acquire whose reads all succeeded ([`Acquiring::wait_reads`] returned `Ok`);
    /// its slots stay pinned until [`Self::release`].
    pub(crate) fn commit(&mut self, mut acquiring: Acquiring) -> Acquired {
        debug_assert!(
            acquiring.ticket.is_none(),
            "committed before the reads were waited for"
        );
        if let Some(ticket) = acquiring.ticket.take() {
            // Release builds: never publish bytes that may still be arriving.
            let _ = ticket.wait();
        }
        self.map.commit(&acquiring.plan);
        let slots = acquiring.plan.slots.iter().map(|&s| s as u32).collect();
        Acquired {
            plan: acquiring.plan,
            slots,
        }
    }

    /// Give an acquire's slots back without publishing anything: its reads failed, or the call
    /// is abandoned. Waits for reads still running. The resident experts' pins go too, so the
    /// caller must first make sure no GPU work it started still reads them.
    pub(crate) fn abort(&mut self, mut acquiring: Acquiring) {
        if let Some(ticket) = acquiring.ticket.take() {
            let _ = ticket.wait();
        }
        self.map.abort(&acquiring.plan);
    }

    /// Unpin an `acquire`'s slots; call once the gather that used them is done.
    pub(crate) fn release(&mut self, acquired: Acquired) {
        self.map.release(&acquired.plan);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_quant::{MODE_MXFP4, gather_qmm_with_mode};
    use mlx_rs::{Array, random};

    /// The 35B checkpoint's layout and bytes, taken through the bank, must
    /// drive `gather_qmm` to exactly what MLX's own stacked tensors give.
    #[test]
    #[ignore = "needs the Qwen3.6-35B-A3B-mxfp4 checkpoint; set LUMEN_QWEN35_MODEL_DIR"]
    fn a_layer_gathered_through_the_bank_equals_the_stacked_original_bit_for_bit() {
        let Ok(dir) = std::env::var("LUMEN_QWEN35_MODEL_DIR") else {
            return;
        };
        let dir = Path::new(&dir);
        let mut bank =
            ExpertBank::open(dir, 600, 8, IoMode::Direct, PackMode::Off, false).expect("open bank");
        assert_eq!(
            bank.roles().count(),
            6,
            "MXFP4: gate/up/down × weight/scales"
        );

        // The reference: MLX's own lazy load of the layer's stacked tensors.
        let layer = 7u16;
        let ranges = bank.store().layer(layer).expect("layer").clone();
        let mut stacked: BTreeMap<String, Array> = BTreeMap::new();
        for shard in &bank.store().files {
            for (name, arr) in Array::load_safetensors(shard).expect("load shard") {
                if let Some(rest) = name.split_once(&format!("layers.{layer}.mlp.switch_mlp.")) {
                    stacked.insert(rest.1.to_string(), arr);
                }
            }
        }
        assert_eq!(stacked.len(), ranges.roles.len());

        let hidden = 2048;
        let x = random::normal::<f32>(&[1, 1, 1, 1, hidden], None, None, None)
            .expect("x")
            .as_dtype(Dtype::Bfloat16)
            .expect("bf16");
        let wanted: [u16; 8] = [200, 5, 17, 5, 255, 0, 91, 17]; // repeats on purpose

        let first = bank.acquire(layer, &wanted).expect("acquire");
        assert_eq!(first.loaded_experts(), 6, "six distinct experts were cold");
        assert_eq!(
            first.slots[1], first.slots[3],
            "a repeated expert shares its slot"
        );
        let by_slot = Array::from_slice(&first.slots, &[1, 1, wanted.len() as i32]);
        let by_expert = Array::from_slice(&wanted.map(u32::from), &[1, 1, wanted.len() as i32]);

        let run = |w: &Array, s: &Array, idx: &Array| -> Vec<u8> {
            // gate_proj: [out=512 per expert, in packed], group 32, 4 bits, mxfp4.
            let y = gather_qmm_with_mode(
                &x,
                w,
                s,
                None,
                None,
                Some(idx),
                true,
                32,
                4,
                MODE_MXFP4,
                false,
            )
            .expect("gather_qmm");
            let v = y
                .as_dtype(Dtype::Float32)
                .expect("f32")
                .view::<u8>()
                .expect("view");
            v.eval().expect("eval");
            v.try_as_slice::<u8>().expect("slice").to_vec()
        };
        let reference = run(
            &stacked["gate_proj.weight"],
            &stacked["gate_proj.scales"],
            &by_expert,
        );
        let from_bank = run(
            bank.pool("gate_proj.weight").unwrap().array(),
            bank.pool("gate_proj.scales").unwrap().array(),
            &by_slot,
        );
        assert_eq!(
            from_bank, reference,
            "gate_proj gather over bank slots differs from the stacked original"
        );
        assert!(reference.iter().any(|&b| b != 0));
        bank.release(first);

        // A second call asking for some of the same experts plus new ones: hits
        // are free, the new ones are read, and the answer is still exact.
        let wanted2: [u16; 4] = [5, 17, 130, 77];
        let second = bank.acquire(layer, &wanted2).expect("acquire 2");
        assert_eq!(second.loaded_experts(), 2, "5 and 17 were already resident");
        let by_slot2 = Array::from_slice(&second.slots, &[1, 1, 4]);
        let by_expert2 = Array::from_slice(&wanted2.map(u32::from), &[1, 1, 4]);
        assert_eq!(
            run(
                bank.pool("gate_proj.weight").unwrap().array(),
                bank.pool("gate_proj.scales").unwrap().array(),
                &by_slot2
            ),
            run(
                &stacked["gate_proj.weight"],
                &stacked["gate_proj.scales"],
                &by_expert2
            ),
        );
        bank.release(second);
        let st = bank.stats();
        eprintln!("bank stats: {st:?} hit rate {:.2}", st.hit_rate());
        assert_eq!((st.hits, st.misses), (2, 8));
    }

    #[test]
    #[ignore = "needs the Qwen3.6-35B-A3B-mxfp4 checkpoint; set LUMEN_QWEN35_MODEL_DIR"]
    fn asking_for_a_missing_layer_or_expert_is_an_error_and_leaves_no_pins() {
        let Ok(dir) = std::env::var("LUMEN_QWEN35_MODEL_DIR") else {
            return;
        };
        // Two staging regions of 256 plus eight slots to cache in, once the regions are borrowed.
        let mut bank = ExpertBank::open(
            Path::new(&dir),
            520,
            4,
            IoMode::Direct,
            PackMode::Off,
            false,
        )
        .expect("open");
        bank.enter_staging();
        assert!(bank.acquire(999, &[0]).is_err());
        assert!(
            bank.acquire(0, &[256]).is_err(),
            "the 35B has experts 0..256"
        );
        // More distinct experts than slots.
        let too_many: Vec<u16> = (0..9).collect();
        let err = bank.acquire(0, &too_many).err().expect("pool too small");
        assert!(format!("{err}").contains("9 distinct"), "got: {err}");
        // Nothing leaked: the whole pool is still usable.
        let ok: Vec<u16> = (0..8).collect();
        let a = bank.acquire(0, &ok).expect("all eight fit");
        bank.release(a);
        // Handing the blocks back opens the whole pool again.
        bank.leave_staging();
        let wide: Vec<u16> = (0..256).collect();
        let a = bank
            .acquire(0, &wide)
            .expect("a whole layer fits once the staging is back");
        bank.release(a);
    }
}
