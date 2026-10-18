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
//!    pinned; for each missing one a victim slot is chosen by CLOCK, **removed
//!    from the map** and pinned. From that moment no lookup can return it.
//! 2. The caller reads the missing experts into those slots.
//! 3. [`SlotMap::commit`] publishes the new `(layer, expert) → slot` entries, or
//!    [`SlotMap::abort`] gives the slots back if a read failed. Either way
//!    [`SlotMap::release`] then drops the pins.
//!
//! Pins mean "the GPU may be reading this slot": only the experts the current
//! layer call uses are pinned, because the per-layer routing readback finishes
//! all earlier GPU work before the next call plans.
//!
//! Eviction is CLOCK, not LRU or a frequency count: on Flash-Next routing
//! traces, Slotstream measured CLOCK within 2% of LRU and 4% of the best
//! implementable policy, with decayed-LFU worse than both.

use std::collections::HashMap;

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
struct Slot {
    /// The expert whose bytes are in the slot; `None` when it is free or being
    /// reloaded (see `loading`).
    key: Option<ExpertKey>,
    /// CLOCK's second-chance bit: set on every hit and on load.
    referenced: bool,
    /// Plans currently using the slot. A pinned slot is never a victim.
    pins: u32,
    /// Reserved for an expert that has been planned but not committed.
    loading: bool,
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
    hand: usize,
    stats: Stats,
}

impl SlotMap {
    pub(crate) fn new(n_slots: usize) -> Self {
        assert!(n_slots > 0, "a slot map needs at least one slot");
        Self {
            slots: vec![Slot::default(); n_slots],
            index: HashMap::with_capacity(n_slots),
            // Popped from the end, so slot 0 goes first.
            free: (0..n_slots).rev().collect(),
            hand: 0,
            stats: Stats::default(),
        }
    }

    #[cfg(test)]
    pub(crate) fn resident(&self) -> usize {
        self.index.len()
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

    /// Plan one layer call for `needed` (any order; repeats are allowed and
    /// share a slot). On success every returned slot is pinned and the missing
    /// experts' slots are unmapped and reserved; on error nothing changed.
    ///
    /// Resident experts are pinned first, all of them, and only then are
    /// victims chosen for the missing ones: a miss early in the list must not
    /// evict an expert that a later request in the same call is about to hit.
    pub(crate) fn plan(&mut self, needed: &[ExpertKey]) -> Result<Plan, PoolTooSmall> {
        const UNSET: usize = usize::MAX;
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
            s.referenced = true;
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
            s.referenced = false;
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

    /// A free slot, or the CLOCK victim. The slot comes back unmapped.
    fn take_victim(&mut self) -> Option<usize> {
        if let Some(slot) = self.free.pop() {
            return Some(slot);
        }
        let n = self.slots.len();
        // Two sweeps are enough: the first clears reference bits, the second
        // finds a clear one. A third would only find pinned slots again.
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
            s.referenced = false;
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
        let mut map = SlotMap::new(4);
        let plan = map.plan(&[k(0, 1), k(0, 2), k(0, 1)]).expect("plan");
        assert_eq!(plan.slots[0], plan.slots[2]);
        assert_eq!(plan.loads.len(), 2);
        map.commit(&plan);
        map.release(&plan);
        assert_eq!(
            map.pinned(),
            0,
            "the shared slot must be unpinned exactly once"
        );
    }

    #[test]
    fn the_current_calls_experts_are_never_evicted_by_that_call() {
        let mut map = SlotMap::new(3);
        call(&mut map, &[k(0, 0), k(0, 1), k(0, 2)]);
        // Needs 0 (resident, pinned) plus two new ones: only the other two may go.
        let plan = map.plan(&[k(1, 0), k(0, 0), k(1, 1)]).expect("plan");
        let kept = map.peek(k(0, 0)).expect("expert 0 stays resident");
        assert_eq!(plan.slots[1], kept);
        assert!(plan.loads.iter().all(|&(_, s)| s != kept));
        map.commit(&plan);
        map.release(&plan);
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
        let mut map = SlotMap::new(2);
        call(&mut map, &[k(0, 0), k(0, 1)]);
        let before: Vec<_> = (0..2).map(|e| map.peek(k(0, e))).collect();
        let err = map.plan(&[k(0, 0), k(0, 1), k(0, 2)]).unwrap_err();
        assert_eq!(
            err,
            PoolTooSmall {
                needed: 3,
                slots: 2
            }
        );
        assert_eq!(
            map.pinned(),
            0,
            "the failed plan must not leave pins behind"
        );
        assert_eq!(
            (0..2).map(|e| map.peek(k(0, e))).collect::<Vec<_>>(),
            before
        );
        // And the map is still usable.
        call(&mut map, &[k(0, 2)]);
    }

    #[test]
    fn a_planned_slot_is_invisible_until_it_is_committed() {
        let mut map = SlotMap::new(2);
        call(&mut map, &[k(0, 0), k(0, 1)]);
        let plan = map.plan(&[k(1, 5)]).expect("plan");
        let (key, slot) = plan.loads[0];
        assert_eq!(map.peek(key), None, "not published before commit");
        // The evicted occupant is gone from the map too: its slot is being rewritten.
        let evicted = [k(0, 0), k(0, 1)]
            .into_iter()
            .find(|&e| map.peek(e).is_none());
        assert!(
            evicted.is_some(),
            "the victim's old entry must be unmapped at plan time"
        );
        assert!(plan.slots.contains(&slot));
        map.commit(&plan);
        assert_eq!(map.peek(key), Some(slot));
        map.release(&plan);
    }

    #[test]
    fn an_aborted_plan_frees_its_slots_and_leaves_the_victims_evicted() {
        let mut map = SlotMap::new(2);
        call(&mut map, &[k(0, 0), k(0, 1)]);
        let plan = map.plan(&[k(1, 5)]).expect("plan");
        map.abort(&plan);
        assert_eq!(map.pinned(), 0);
        assert_eq!(
            map.peek(k(1, 5)),
            None,
            "an aborted load is never published"
        );
        assert_eq!(
            map.resident(),
            1,
            "one old expert was sacrificed to the failed load"
        );
        // The freed slot is reused before anything else is evicted.
        let again = call(&mut map, &[k(1, 5)]);
        assert_eq!(again.loads.len(), 1);
        assert_eq!(map.resident(), 2);
    }

    #[test]
    fn a_reserved_range_is_emptied_and_left_alone_until_it_is_released() {
        let mut map = SlotMap::new(8);
        let first = call(
            &mut map,
            &[k(0, 0), k(0, 1), k(0, 2), k(0, 3), k(0, 4), k(0, 5)],
        );
        let evicted = map.reserve_range(4..8);
        assert_eq!(evicted, first.slots.iter().filter(|&&s| s >= 4).count());
        for e in 0..6u16 {
            let resident = map.peek(k(0, e)).is_some();
            assert_eq!(resident, first.slots[e as usize] < 4, "expert {e}");
        }
        // Nothing planned meanwhile lands in the reserved range, however hard it is pushed.
        for round in 0..20u16 {
            let keys: Vec<ExpertKey> = (0..4).map(|i| k(1, round * 4 + i)).collect();
            let q = call(&mut map, &keys);
            assert!(
                q.slots.iter().all(|&s| s < 4),
                "round {round}: {:?}",
                q.slots
            );
        }
        let five: Vec<ExpertKey> = (0..5).map(|i| k(2, i)).collect();
        assert!(map.plan(&five).is_err(), "only four slots are usable");
        // Released, the range is free and is handed out before anything is evicted.
        map.release_range(4..8);
        let q = call(&mut map, &five);
        assert_eq!(q.slots.len(), 5);
        assert!(q.slots.iter().filter(|&&s| s >= 4).count() >= 1);
    }

    /// Whatever the request sequence, the map stays a bijection between
    /// resident keys and slots, no pins leak, and a plan's slots hold exactly
    /// the experts that were asked for.
    #[test]
    fn a_long_random_sequence_keeps_every_invariant() {
        let mut rng: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let n_slots = 24;
        let mut map = SlotMap::new(n_slots);
        // What each slot must hold, tracked independently of the map.
        let mut holds: Vec<Option<ExpertKey>> = vec![None; n_slots];

        for step in 0..20_000 {
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
            let plan = map.plan(&needed).expect("24 slots cover 10 requests");
            // An expert that was resident and is asked for is a hit: no earlier
            // miss in the same call may have evicted it and forced a reload.
            for &(key, _) in &plan.loads {
                assert!(
                    !resident_before.contains(&key),
                    "step {step}: {key:?} was resident and requested but got reloaded"
                );
            }
            for &(key, slot) in &plan.loads {
                assert!(
                    plan.slots.iter().filter(|&&s| s == slot).count() >= 1,
                    "step {step}: load slot not part of the plan"
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
                        "step {step}: slot {slot} holds the wrong expert"
                    );
                    assert_eq!(map.peek(*key), Some(slot));
                }
                map.release(&plan);
            }
            assert_eq!(map.pinned(), 0, "step {step}: pins leaked");
            // Bijection: every resident key's slot holds that key, and no two keys share a slot.
            let mut seen = vec![false; n_slots];
            for slot in 0..n_slots {
                if let Some(key) = holds[slot] {
                    if let Some(s) = map.peek(key) {
                        assert_eq!(s, slot, "step {step}: {key:?} mapped to the wrong slot");
                        assert!(!seen[slot]);
                        seen[slot] = true;
                    }
                }
            }
            assert_eq!(
                map.resident(),
                seen.iter().filter(|&&x| x).count(),
                "step {step}"
            );
        }
        let st = map.stats();
        assert!(
            st.hits > 1000 && st.misses > 1000,
            "the sequence should exercise both paths: {st:?}"
        );
    }
}
