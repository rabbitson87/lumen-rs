//! Which routed expert lives in which slot of the pool.
//!
//! One global pool serves every MoE layer (a layer that is routed to a lot
//! gets more slots than one that is not), so the key is `(layer, expert)`.
//! This type is pure bookkeeping: it hands out slots and says what to load
//! into them. The bytes are the IO layer's business, the buffer is
//! [`super::slot_pool::PoolTensor`]'s.
//!
//! A call goes through three steps, and the order is what keeps a slot's bytes
//! from ever being read half-written:
//!
//! 1. [`SlotMap::plan`] — name the experts a layer needs. Resident ones are
//!    pinned; for each missing one a victim slot is chosen by the eviction
//!    policy, **removed from the map** and pinned. From that moment no lookup
//!    can return it.
//! 2. The caller reads the missing experts into those slots.
//! 3. [`SlotMap::commit`] publishes the new `(layer, expert) → slot` entries, or
//!    [`SlotMap::abort`] gives the slots back if a read failed. Either way
//!    [`SlotMap::release`] then drops the pins.
//!
//! Pins mean "the GPU may be reading this slot": only the experts the current
//! layer call uses are pinned, because the per-layer routing readback finishes
//! all earlier GPU work before the next call plans.
//!
//! Which resident expert goes is [`super::slot_policy`]'s choice (`LUMEN_MOE_EVICTION`): the
//! default `bias` misses 10-13% less than CLOCK on recorded Flash-Next traces at pools of 4-7 GB,
//! and ran 11% faster in a real-server A/B at 6.7 GB. CLOCK was the first choice (Slotstream
//! measured it within 2% of LRU and 4% of the best implementable policy; our traces put it 6-16%
//! of misses behind `gclock` and `bias`) and stays selectable. This type stays bookkeeping: it
//! asks the policy "who goes" and tells it what happened.

use std::collections::HashMap;

use super::slot_policy::Policy;
pub(crate) use super::slot_policy::{CallKind, EvictionKind, PolicyConfig};

/// A routed expert: which MoE layer, which expert of that layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ExpertKey {
    pub layer: u16,
    pub expert: u16,
}

impl ExpertKey {
    pub(crate) fn new(layer: u16, expert: u16) -> Self {
        Self { layer, expert }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Slot {
    /// The expert whose bytes are in the slot; `None` when it is free or being
    /// reloaded (see `loading`).
    pub(super) key: Option<ExpertKey>,
    /// Plans currently using the slot. A pinned slot is never a victim.
    pub(super) pins: u32,
    /// Reserved for an expert that has been planned but not committed.
    pub(super) loading: bool,
}

impl Slot {
    /// Pinned by a plan or reserved: not a candidate for eviction.
    #[inline]
    pub(super) fn busy(&self) -> bool {
        self.pins > 0 || self.loading
    }
}

/// What [`SlotMap::plan`] decided for one layer call.
#[derive(Debug)]
pub(crate) struct Plan {
    /// For every requested expert, in request order: its slot.
    pub slots: Vec<usize>,
    /// The experts that are not resident and must be read into their slot
    /// before the call's gather may run.
    pub loads: Vec<(ExpertKey, usize)>,
}

/// More distinct experts were requested than the pool has slots to give.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct PoolTooSmall {
    pub needed: usize,
    pub slots: usize,
}

impl std::fmt::Display for PoolTooSmall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "a layer call needs {} distinct experts resident at once but the pool has {} slots",
            self.needed, self.slots
        )
    }
}

impl std::error::Error for PoolTooSmall {}

/// Hit/miss counters since creation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Stats {
    pub hits: u64,
    pub misses: u64,
}

impl Stats {
    pub(crate) fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

pub(crate) struct SlotMap {
    slots: Vec<Slot>,
    /// Published entries only: a slot being loaded is not in here.
    index: HashMap<ExpertKey, usize>,
    /// Never-used and aborted slots, handed out before any eviction.
    free: Vec<usize>,
    policy: Policy,
    stats: Stats,
}

impl SlotMap {
    /// A map that evicts by CLOCK.
    #[cfg(test)]
    pub(crate) fn new(n_slots: usize) -> Self {
        Self::with_policy(n_slots, &PolicyConfig::default())
    }

    pub(crate) fn with_policy(n_slots: usize, cfg: &PolicyConfig) -> Self {
        assert!(n_slots > 0, "a slot map needs at least one slot");
        Self {
            slots: vec![Slot::default(); n_slots],
            index: HashMap::with_capacity(n_slots),
            // Popped from the end, so slot 0 goes first.
            free: (0..n_slots).rev().collect(),
            policy: Policy::new(n_slots, cfg),
            stats: Stats::default(),
        }
    }

    #[cfg(test)]
    pub(crate) fn resident(&self) -> usize {
        self.index.len()
    }

    /// Slots on the free list (never used, aborted, or handed back).
    #[cfg(test)]
    pub(crate) fn free_slots(&self) -> usize {
        self.free.len()
    }

    #[cfg(test)]
    pub(super) fn policy(&self) -> &Policy {
        &self.policy
    }

    pub(crate) fn stats(&self) -> Stats {
        self.stats
    }

    /// The slot holding `key`, if it is resident. Does not pin or touch it.
    pub(crate) fn peek(&self, key: ExpertKey) -> Option<usize> {
        self.index.get(&key).copied()
    }

    /// Slots currently pinned by plans (zero between layer calls).
    #[cfg(test)]
    pub(crate) fn pinned(&self) -> usize {
        self.slots.iter().filter(|s| s.pins > 0).count()
    }

    /// [`Self::plan_as`] for a decode call.
    #[cfg(test)]
    pub(crate) fn plan(&mut self, needed: &[ExpertKey]) -> Result<Plan, PoolTooSmall> {
        self.plan_as(needed, CallKind::Decode)
    }

    /// Plan one layer call for `needed` (any order; repeats are allowed and
    /// share a slot). On success every returned slot is pinned and the missing
    /// experts' slots are unmapped and reserved; on error the map is unchanged
    /// (the eviction policy's own bookkeeping, such as CLOCK's reference bits, may have moved).
    ///
    /// Resident experts are pinned first, all of them, and only then are
    /// victims chosen for the missing ones: a miss early in the list must not
    /// evict an expert that a later request in the same call is about to hit.
    ///
    /// `kind` tells a learning eviction policy whether this call's routing is the kind it learns
    /// from; CLOCK and GCLOCK ignore it.
    pub(crate) fn plan_as(
        &mut self,
        needed: &[ExpertKey],
        kind: CallKind,
    ) -> Result<Plan, PoolTooSmall> {
        const UNSET: usize = usize::MAX;
        self.policy.begin_call(kind, needed);
        let mut slots = vec![UNSET; needed.len()];
        let mut loads: Vec<(ExpertKey, usize)> = Vec::new();
        // Each slot this call pinned, once; also how a repeated key finds its slot.
        let mut pinned: HashMap<ExpertKey, usize> = HashMap::with_capacity(needed.len());
        let mut hits = 0u64;

        for (i, &key) in needed.iter().enumerate() {
            if let Some(&slot) = pinned.get(&key) {
                slots[i] = slot;
            } else if let Some(&slot) = self.index.get(&key) {
                hits += 1;
                self.policy.on_hit(slot);
                self.slots[slot].pins += 1;
                pinned.insert(key, slot);
                slots[i] = slot;
            }
        }

        for (i, &key) in needed.iter().enumerate() {
            if slots[i] != UNSET {
                continue;
            }
            if let Some(&slot) = pinned.get(&key) {
                slots[i] = slot; // the same missing expert listed twice
                continue;
            }
            let Some(slot) = self.take_victim() else {
                let pins: Vec<usize> = pinned.values().copied().collect();
                self.undo(&pins, &loads);
                return Err(PoolTooSmall {
                    needed: distinct(needed),
                    slots: self.slots.len(),
                });
            };
            let s = &mut self.slots[slot];
            s.pins += 1;
            s.loading = true;
            loads.push((key, slot));
            pinned.insert(key, slot);
            slots[i] = slot;
        }
        self.stats.hits += hits;
        self.stats.misses += loads.len() as u64;
        Ok(Plan { slots, loads })
    }

    /// Publish the loaded experts. Call after their bytes are in the slots.
    pub(crate) fn commit(&mut self, plan: &Plan) {
        for &(key, slot) in &plan.loads {
            let s = &mut self.slots[slot];
            debug_assert!(s.loading && s.key.is_none(), "slot {slot} was not reserved");
            s.key = Some(key);
            s.loading = false;
            self.policy.on_admit(slot);
            let clash = self.index.insert(key, slot);
            debug_assert!(clash.is_none(), "{key:?} was already resident");
        }
    }

    /// Give back the slots of a plan whose loads did not complete. Their
    /// previous occupants stay evicted: the bytes may be half overwritten.
    pub(crate) fn abort(&mut self, plan: &Plan) {
        for &(_, slot) in &plan.loads {
            let s = &mut self.slots[slot];
            s.loading = false;
            s.key = None;
            self.policy.on_vacate(slot);
            self.free.push(slot);
        }
        // The aborted loads' slots were pinned for this plan too.
        self.unpin(&plan.slots);
    }

    /// Drop a plan's pins. After [`Self::commit`], once the call's gather is
    /// done; after [`Self::abort`] this is already done.
    pub(crate) fn release(&mut self, plan: &Plan) {
        self.unpin(&plan.slots);
    }

    /// Take `range` out of the map's hands: whoever resident is in it is evicted, the slots
    /// leave the free list, and no plan will use or evict them until [`Self::release_range`].
    /// The prefill borrows two blocks of the pool this way for its staging regions.
    ///
    /// Returns how many resident experts were evicted. Slots a plan has pinned are a caller
    /// bug (the staging path only runs between plans), checked in debug builds.
    pub(crate) fn reserve_range(&mut self, range: std::ops::Range<usize>) -> usize {
        let mut evicted = 0;
        for slot in range.clone() {
            let s = &mut self.slots[slot];
            debug_assert_eq!(s.pins, 0, "slot {slot} is pinned by a plan");
            if let Some(key) = s.key.take() {
                self.index.remove(&key);
                evicted += 1;
            }
            *s = Slot {
                loading: true,
                ..Slot::default()
            };
            self.policy.on_vacate(slot);
        }
        self.free.retain(|f| !range.contains(f));
        evicted
    }

    /// Hand a reserved range back: its slots are free, empty, and used before any eviction.
    pub(crate) fn release_range(&mut self, range: std::ops::Range<usize>) {
        for slot in range.rev() {
            let s = &mut self.slots[slot];
            debug_assert!(s.loading && s.key.is_none(), "slot {slot} was not reserved");
            *s = Slot::default();
            self.free.push(slot);
        }
    }

    // ---- internals ----

    /// A free slot, or the policy's victim. The slot comes back unmapped.
    fn take_victim(&mut self) -> Option<usize> {
        if let Some(slot) = self.free.pop() {
            return Some(slot);
        }
        let slot = self.policy.victim(&self.slots)?;
        if let Some(key) = self.slots[slot].key.take() {
            self.index.remove(&key);
        }
        self.policy.on_vacate(slot);
        Some(slot)
    }

    fn unpin(&mut self, slots: &[usize]) {
        // A repeated expert appears twice in `slots` but was pinned once.
        let mut done: Vec<usize> = Vec::with_capacity(slots.len());
        for &slot in slots {
            if done.contains(&slot) {
                continue;
            }
            done.push(slot);
            let s = &mut self.slots[slot];
            debug_assert!(
                s.pins > 0,
                "slot {slot} released more often than it was pinned"
            );
            s.pins = s.pins.saturating_sub(1);
        }
    }

    /// Roll a half-built plan back: unpin what it pinned, free what it reserved.
    fn undo(&mut self, slots: &[usize], loads: &[(ExpertKey, usize)]) {
        for &(_, slot) in loads {
            let s = &mut self.slots[slot];
            s.loading = false;
            s.key = None;
            self.policy.on_vacate(slot);
            self.free.push(slot);
        }
        self.unpin(slots);
    }
}

fn distinct(keys: &[ExpertKey]) -> usize {
    let mut v: Vec<ExpertKey> = keys.to_vec();
    v.sort_by_key(|k| (k.layer, k.expert));
    v.dedup();
    v.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(layer: u16, expert: u16) -> ExpertKey {
        ExpertKey::new(layer, expert)
    }

    /// A map with `kind` evicting, for a model of 64 experts per layer and 6 layers per token.
    fn map_of(kind: EvictionKind, n_slots: usize) -> SlotMap {
        SlotMap::with_policy(n_slots, &PolicyConfig::new(kind, 64, 6))
    }

    /// Run one whole layer call: plan, "load", commit, release.
    fn call(map: &mut SlotMap, needed: &[ExpertKey]) -> Plan {
        let plan = map.plan(needed).expect("plan");
        map.commit(&plan);
        map.release(&plan);
        plan
    }

    #[test]
    fn a_cold_call_loads_everything_into_distinct_slots_and_a_repeat_hits() {
        let mut map = SlotMap::new(4);
        let first = call(&mut map, &[k(0, 7), k(0, 3), k(0, 9)]);
        assert_eq!(first.loads.len(), 3);
        let mut slots = first.slots.clone();
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(slots.len(), 3, "three experts, three slots");
        assert_eq!(map.stats(), Stats { hits: 0, misses: 3 });

        let again = call(&mut map, &[k(0, 7), k(0, 3), k(0, 9)]);
        assert!(again.loads.is_empty());
        assert_eq!(again.slots, first.slots);
        assert_eq!(map.stats(), Stats { hits: 3, misses: 3 });
        assert_eq!(map.pinned(), 0);
    }

    #[test]
    fn keys_are_per_layer() {
        let mut map = SlotMap::new(4);
        let p = call(&mut map, &[k(0, 5), k(1, 5)]);
        assert_ne!(
            p.slots[0], p.slots[1],
            "expert 5 of two layers are two experts"
        );
        assert_eq!(map.peek(k(0, 5)), Some(p.slots[0]));
        assert_eq!(map.peek(k(1, 5)), Some(p.slots[1]));
    }

    #[test]
    fn a_repeated_expert_in_one_call_shares_a_slot_and_loads_once() {
        for kind in EvictionKind::ALL {
            let mut map = map_of(kind, 4);
            let plan = map.plan(&[k(0, 1), k(0, 2), k(0, 1)]).expect("plan");
            assert_eq!(plan.slots[0], plan.slots[2], "{kind:?}");
            assert_eq!(plan.loads.len(), 2, "{kind:?}");
            map.commit(&plan);
            map.release(&plan);
            assert_eq!(
                map.pinned(),
                0,
                "{kind:?}: the shared slot must be unpinned exactly once"
            );
        }
    }

    #[test]
    fn the_current_calls_experts_are_never_evicted_by_that_call() {
        for kind in EvictionKind::ALL {
            let mut map = map_of(kind, 3);
            call(&mut map, &[k(0, 0), k(0, 1), k(0, 2)]);
            // Needs 0 (resident, pinned) plus two new ones: only the other two may go.
            let plan = map.plan(&[k(1, 0), k(0, 0), k(1, 1)]).expect("plan");
            let kept = map.peek(k(0, 0)).expect("expert 0 stays resident");
            assert_eq!(plan.slots[1], kept, "{kind:?}");
            assert!(plan.loads.iter().all(|&(_, s)| s != kept), "{kind:?}");
            map.commit(&plan);
            map.release(&plan);
        }
    }

    #[test]
    fn clock_gives_a_recently_hit_expert_a_second_chance() {
        // Three slots. Loading sets every reference bit, so the first miss makes
        // the hand sweep all three clear and then evict the first one (A).
        let build = || {
            let mut map = SlotMap::new(3);
            call(&mut map, &[k(0, 0)]); // A
            call(&mut map, &[k(0, 1)]); // B
            call(&mut map, &[k(0, 2)]); // C
            call(&mut map, &[k(0, 3)]); // D evicts A; B and C are now unreferenced
            assert!(map.peek(k(0, 0)).is_none(), "A was the first victim");
            map
        };

        // Without a touch, B is next in line and goes.
        let mut cold = build();
        call(&mut cold, &[k(0, 4)]);
        assert!(
            cold.peek(k(0, 1)).is_none(),
            "an untouched B is evicted next"
        );
        assert!(cold.peek(k(0, 2)).is_some());

        // A hit on B sets its bit again: the hand clears it and moves on to C.
        let mut warm = build();
        call(&mut warm, &[k(0, 1)]);
        call(&mut warm, &[k(0, 4)]);
        assert!(
            warm.peek(k(0, 1)).is_some(),
            "the hit expert got its second chance"
        );
        assert!(warm.peek(k(0, 2)).is_none(), "the untouched C went instead");
    }

    #[test]
    fn asking_for_more_distinct_experts_than_slots_fails_and_changes_nothing() {
        for kind in EvictionKind::ALL {
            let mut map = map_of(kind, 2);
            call(&mut map, &[k(0, 0), k(0, 1)]);
            let before: Vec<_> = (0..2).map(|e| map.peek(k(0, e))).collect();
            let err = map.plan(&[k(0, 0), k(0, 1), k(0, 2)]).unwrap_err();
            assert_eq!(
                err,
                PoolTooSmall {
                    needed: 3,
                    slots: 2
                },
                "{kind:?}"
            );
            assert_eq!(
                map.pinned(),
                0,
                "{kind:?}: the failed plan must not leave pins behind"
            );
            assert_eq!(
                (0..2).map(|e| map.peek(k(0, e))).collect::<Vec<_>>(),
                before,
                "{kind:?}"
            );
            // And the map is still usable.
            call(&mut map, &[k(0, 2)]);
        }
    }

    #[test]
    fn a_planned_slot_is_invisible_until_it_is_committed() {
        for kind in EvictionKind::ALL {
            let mut map = map_of(kind, 2);
            call(&mut map, &[k(0, 0), k(0, 1)]);
            let plan = map.plan(&[k(1, 5)]).expect("plan");
            let (key, slot) = plan.loads[0];
            assert_eq!(map.peek(key), None, "{kind:?}: not published before commit");
            // The evicted occupant is gone from the map too: its slot is being rewritten.
            let evicted = [k(0, 0), k(0, 1)]
                .into_iter()
                .find(|&e| map.peek(e).is_none());
            assert!(
                evicted.is_some(),
                "{kind:?}: the victim's old entry must be unmapped at plan time"
            );
            assert!(plan.slots.contains(&slot));
            map.commit(&plan);
            assert_eq!(map.peek(key), Some(slot));
            map.release(&plan);
        }
    }

    #[test]
    fn an_aborted_plan_frees_its_slots_and_leaves_the_victims_evicted() {
        for kind in EvictionKind::ALL {
            let mut map = map_of(kind, 2);
            call(&mut map, &[k(0, 0), k(0, 1)]);
            let plan = map.plan(&[k(1, 5)]).expect("plan");
            let slot = plan.loads[0].1;
            map.abort(&plan);
            assert_eq!(map.pinned(), 0, "{kind:?}");
            assert_eq!(
                map.peek(k(1, 5)),
                None,
                "{kind:?}: an aborted load is never published"
            );
            assert_eq!(
                map.resident(),
                1,
                "{kind:?}: one old expert was sacrificed to the failed load"
            );
            assert_eq!(map.free_slots(), 1, "{kind:?}");
            assert_eq!(
                map.policy().worth(slot),
                0.0,
                "{kind:?}: the policy forgot the aborted load"
            );
            // The freed slot is reused before anything else is evicted.
            let again = call(&mut map, &[k(1, 5)]);
            assert_eq!(again.loads.len(), 1, "{kind:?}");
            assert_eq!(again.slots, vec![slot], "{kind:?}");
            assert_eq!(map.resident(), 2, "{kind:?}");
        }
    }

    #[test]
    fn a_reserved_range_is_emptied_and_left_alone_until_it_is_released() {
        for kind in EvictionKind::ALL {
            let mut map = map_of(kind, 8);
            let first = call(
                &mut map,
                &[k(0, 0), k(0, 1), k(0, 2), k(0, 3), k(0, 4), k(0, 5)],
            );
            let evicted = map.reserve_range(4..8);
            assert_eq!(
                evicted,
                first.slots.iter().filter(|&&s| s >= 4).count(),
                "{kind:?}"
            );
            for e in 0..6u16 {
                let resident = map.peek(k(0, e)).is_some();
                assert_eq!(
                    resident,
                    first.slots[e as usize] < 4,
                    "{kind:?}: expert {e}"
                );
            }
            // Nothing planned meanwhile lands in the reserved range, however hard it is pushed.
            for round in 0..20u16 {
                let keys: Vec<ExpertKey> = (0..4).map(|i| k(1, round * 4 + i)).collect();
                let q = call(&mut map, &keys);
                assert!(
                    q.slots.iter().all(|&s| s < 4),
                    "{kind:?} round {round}: {:?}",
                    q.slots
                );
            }
            let five: Vec<ExpertKey> = (0..5).map(|i| k(2, i)).collect();
            assert!(
                map.plan(&five).is_err(),
                "{kind:?}: only four slots are usable"
            );
            // Released, the range is free and is handed out before anything is evicted.
            map.release_range(4..8);
            let q = call(&mut map, &five);
            assert_eq!(q.slots.len(), 5);
            assert!(q.slots.iter().filter(|&&s| s >= 4).count() >= 1, "{kind:?}");
        }
    }

    /// Whatever the request sequence, the map stays a bijection between resident keys and
    /// slots, no pins leak, the slots are all accounted for, a plan's slots hold exactly the
    /// experts that were asked for, and no call evicts what it is using.
    fn keeps_every_invariant(kind: EvictionKind, n_slots: usize, steps: usize) {
        let mut rng: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut map = map_of(kind, n_slots);
        // What each slot must hold, tracked independently of the map.
        let mut holds: Vec<Option<ExpertKey>> = vec![None; n_slots];
        // A block of slots lent out right now, as the prefill staging does.
        let block = n_slots / 6;
        let mut lent = false;

        for step in 0..steps {
            if step % 97 == 0 {
                if lent {
                    map.release_range(n_slots - block..n_slots);
                } else {
                    map.reserve_range(n_slots - block..n_slots);
                    holds[n_slots - block..].fill(None);
                }
                lent = !lent;
            }
            let layer = (next() % 6) as u16;
            let want = 1 + (next() % 10) as usize;
            // Skewed toward a hot set so the hit path is exercised too.
            let needed: Vec<ExpertKey> = (0..want)
                .map(|_| {
                    let e = if next() % 3 == 0 {
                        next() % 64
                    } else {
                        next() % 12
                    };
                    k(layer, e as u16)
                })
                .collect();
            let resident_before: Vec<ExpertKey> = needed
                .iter()
                .copied()
                .filter(|&n| map.peek(n).is_some())
                .collect();
            let call_kind = if next() % 8 == 0 {
                CallKind::Prefill
            } else {
                CallKind::Decode
            };
            let plan = map.plan_as(&needed, call_kind).expect("enough slots");
            // An expert that was resident and is asked for is a hit: no earlier
            // miss in the same call may have evicted it and forced a reload.
            for &(key, _) in &plan.loads {
                assert!(
                    !resident_before.contains(&key),
                    "{kind:?} step {step}: {key:?} was resident and requested but got reloaded"
                );
            }
            // Distinct experts own distinct slots: nothing the call uses was handed out twice.
            for (a, &sa) in needed.iter().zip(&plan.slots) {
                for (b, &sb) in needed.iter().zip(&plan.slots) {
                    assert_eq!(a == b, sa == sb, "{kind:?} step {step}: {a:?} {b:?}");
                }
            }
            for &(key, slot) in &plan.loads {
                assert!(
                    plan.slots.contains(&slot),
                    "{kind:?} step {step}: load slot not part of the plan"
                );
                assert!(
                    !lent || slot < n_slots - block,
                    "{kind:?} step {step}: a load landed in the lent block"
                );
                holds[slot] = Some(key); // the "IO" lands the bytes
            }
            if next() % 50 == 0 {
                map.abort(&plan);
                for &(_, slot) in &plan.loads {
                    holds[slot] = None;
                }
            } else {
                map.commit(&plan);
                for (key, &slot) in needed.iter().zip(&plan.slots) {
                    assert_eq!(
                        holds[slot],
                        Some(*key),
                        "{kind:?} step {step}: slot {slot} holds the wrong expert"
                    );
                    assert_eq!(map.peek(*key), Some(slot));
                }
                map.release(&plan);
            }
            assert_eq!(map.pinned(), 0, "{kind:?} step {step}: pins leaked");
            // Bijection: every resident key's slot holds that key, and no two keys share a slot.
            let mut seen = vec![false; n_slots];
            for slot in 0..n_slots {
                if let Some(key) = holds[slot] {
                    if let Some(s) = map.peek(key) {
                        assert_eq!(
                            s, slot,
                            "{kind:?} step {step}: {key:?} mapped to the wrong slot"
                        );
                        assert!(!seen[slot]);
                        seen[slot] = true;
                    }
                }
            }
            let resident = seen.iter().filter(|&&x| x).count();
            assert_eq!(map.resident(), resident, "{kind:?} step {step}");
            // Every slot is resident, free, or lent out.
            assert_eq!(
                map.resident() + map.free_slots() + if lent { block } else { 0 },
                n_slots,
                "{kind:?} step {step}: slots unaccounted for"
            );
        }
        let st = map.stats();
        assert!(
            st.hits > 100 && st.misses > 100,
            "{kind:?}: the sequence should exercise both paths: {st:?}"
        );
    }

    #[test]
    fn a_long_random_sequence_keeps_every_invariant() {
        for kind in EvictionKind::ALL {
            keeps_every_invariant(kind, 24, 20_000);
        }
    }

    /// `bias` samples 64 slots, which a 24-slot pool cannot show: a pool larger than the sample
    /// takes the other branch of the victim search.
    #[test]
    fn a_long_random_sequence_keeps_every_invariant_in_a_pool_larger_than_the_bias_sample() {
        for kind in EvictionKind::ALL {
            keeps_every_invariant(kind, 150, 6_000);
        }
    }

    /// The pre-policy implementation, kept verbatim as the reference the CLOCK policy must equal:
    /// "default stays CLOCK" is a promise about every decision, not about the tests that existed.
    mod before_policies {
        use super::super::{ExpertKey, Plan};
        use std::collections::HashMap;

        #[derive(Clone, Copy, Default)]
        struct Slot {
            key: Option<ExpertKey>,
            referenced: bool,
            pins: u32,
            loading: bool,
        }

        pub struct ClockMap {
            slots: Vec<Slot>,
            index: HashMap<ExpertKey, usize>,
            free: Vec<usize>,
            hand: usize,
        }

        impl ClockMap {
            pub fn new(n: usize) -> Self {
                Self {
                    slots: vec![Slot::default(); n],
                    index: HashMap::new(),
                    free: (0..n).rev().collect(),
                    hand: 0,
                }
            }

            pub fn peek(&self, key: ExpertKey) -> Option<usize> {
                self.index.get(&key).copied()
            }

            pub fn plan(&mut self, needed: &[ExpertKey]) -> Option<Plan> {
                const UNSET: usize = usize::MAX;
                let mut slots = vec![UNSET; needed.len()];
                let mut loads: Vec<(ExpertKey, usize)> = Vec::new();
                let mut pinned: HashMap<ExpertKey, usize> = HashMap::new();
                for (i, &key) in needed.iter().enumerate() {
                    if let Some(&slot) = pinned.get(&key) {
                        slots[i] = slot;
                    } else if let Some(&slot) = self.index.get(&key) {
                        let s = &mut self.slots[slot];
                        s.referenced = true;
                        s.pins += 1;
                        pinned.insert(key, slot);
                        slots[i] = slot;
                    }
                }
                for (i, &key) in needed.iter().enumerate() {
                    if slots[i] != UNSET {
                        continue;
                    }
                    if let Some(&slot) = pinned.get(&key) {
                        slots[i] = slot;
                        continue;
                    }
                    let slot = self.take_victim()?;
                    let s = &mut self.slots[slot];
                    s.pins += 1;
                    s.loading = true;
                    loads.push((key, slot));
                    pinned.insert(key, slot);
                    slots[i] = slot;
                }
                Some(Plan { slots, loads })
            }

            pub fn commit(&mut self, plan: &Plan) {
                for &(key, slot) in &plan.loads {
                    let s = &mut self.slots[slot];
                    s.key = Some(key);
                    s.loading = false;
                    s.referenced = true;
                    self.index.insert(key, slot);
                }
            }

            pub fn release(&mut self, plan: &Plan) {
                let mut done: Vec<usize> = Vec::new();
                for &slot in &plan.slots {
                    if done.contains(&slot) {
                        continue;
                    }
                    done.push(slot);
                    self.slots[slot].pins -= 1;
                }
            }

            pub fn reserve_range(&mut self, range: std::ops::Range<usize>) {
                for slot in range.clone() {
                    let s = &mut self.slots[slot];
                    if let Some(key) = s.key.take() {
                        self.index.remove(&key);
                    }
                    *s = Slot {
                        loading: true,
                        ..Slot::default()
                    };
                }
                self.free.retain(|f| !range.contains(f));
            }

            pub fn release_range(&mut self, range: std::ops::Range<usize>) {
                for slot in range.rev() {
                    self.slots[slot] = Slot::default();
                    self.free.push(slot);
                }
            }

            fn take_victim(&mut self) -> Option<usize> {
                if let Some(slot) = self.free.pop() {
                    return Some(slot);
                }
                let n = self.slots.len();
                for _ in 0..2 * n {
                    let i = self.hand;
                    self.hand = (self.hand + 1) % n;
                    let s = &mut self.slots[i];
                    if s.pins > 0 || s.loading {
                        continue;
                    }
                    if s.referenced {
                        s.referenced = false;
                        continue;
                    }
                    if let Some(key) = s.key.take() {
                        self.index.remove(&key);
                    }
                    return Some(i);
                }
                None
            }
        }
    }

    /// The CLOCK policy is the old CLOCK: over a long mixed sequence with staging blocks being
    /// lent and returned, every plan names the same slots and the same loads as the
    /// implementation that predates the policies.
    #[test]
    fn the_default_policy_decides_exactly_as_the_clock_it_replaced() {
        let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for n_slots in [24usize, 97, 300] {
            let mut new = SlotMap::new(n_slots);
            let mut old = before_policies::ClockMap::new(n_slots);
            let block = n_slots / 5;
            let mut lent = false;
            for step in 0..15_000 {
                if step % 131 == 0 {
                    if lent {
                        new.release_range(n_slots - block..n_slots);
                        old.release_range(n_slots - block..n_slots);
                    } else {
                        new.reserve_range(n_slots - block..n_slots);
                        old.reserve_range(n_slots - block..n_slots);
                    }
                    lent = !lent;
                }
                let layer = (next() % 5) as u16;
                let want = 1 + (next() % 14) as usize;
                let needed: Vec<ExpertKey> = (0..want)
                    .map(|_| {
                        let e = if next() % 4 == 0 {
                            next() % 90
                        } else {
                            next() % 20
                        };
                        k(layer, e as u16)
                    })
                    .collect();
                let a = new.plan(&needed).expect("plan");
                let b = old.plan(&needed).expect("old plan");
                assert_eq!(a.slots, b.slots, "{n_slots} slots, step {step}");
                assert_eq!(a.loads, b.loads, "{n_slots} slots, step {step}");
                new.commit(&a);
                old.commit(&b);
                new.release(&a);
                old.release(&b);
            }
            for layer in 0..5u16 {
                for e in 0..90u16 {
                    assert_eq!(new.peek(k(layer, e)), old.peek(k(layer, e)));
                }
            }
        }
    }
}
