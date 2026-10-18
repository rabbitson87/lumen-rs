//! Replay of recorded routing traces (`LUMEN_MOE_TRACE`) through the real [`SlotMap`].
//!
//! The eviction policies were chosen by a trace-driven simulator written in Python. This file is
//! the other half of that argument: it feeds the same traces to the engine's own `SlotMap`, with
//! the plan -> commit -> release sequence, the staging blocks being lent and handed back, and the
//! decode/prefill classification of every call that the engine uses, and reports what the engine
//! did. Two ignored tests use it:
//!
//! * `engine_replay_matches_the_simulator` compares the engine, call by call, with the hit flags
//!   the simulator produced for the same trace, policy and pool size.
//! * `per_call_cost` times the planning of decode-sized, mid-size and large calls per policy.
//!
//! The trace format is documented in [`super::trace`]; this reads the parts a replay needs.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use super::MTP_LAYER;
use super::expert_bank::regions_for;
use super::slot_map::{CallKind, ExpertKey, Plan, SlotMap};
use super::slot_policy::{BiasParams, EvictionKind, PolicyConfig};

/// The server's two start-up chats are not part of the workload; the simulator drops them too.
const WARMUP_REQUESTS: u32 = 2;
/// Requests with this many decode tokens or fewer are not scored (the simulator's rule).
const MIN_SCORED_TOKENS: u32 = 4;

const F_STAGED: u8 = 1;
const F_PREFILL: u8 = 1 << 2;
const F_COUNTS: u8 = 1 << 5;

pub(super) struct Call {
    flags: u8,
    layer: u16,
    request: u32,
    pass: u32,
    ids: Vec<u16>,
}

pub(super) struct Trace {
    pub width: usize,
    pub layers: Vec<u16>,
    pub calls: Vec<Call>,
}

fn le16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn le32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// Read a `LMOETR01` v2 trace: header, then call records (a record cut by a kill is ignored).
pub(super) fn read_trace(path: &Path) -> Trace {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(
        &b[..8],
        b"LMOETR01",
        "{} is not a routing trace",
        path.display()
    );
    assert_eq!(le32(&b, 8), 2, "trace version");
    let width = le32(&b, 36) as usize;
    let n_layers = le32(&b, 60) as usize;
    let layers: Vec<u16> = (0..n_layers).map(|i| le16(&b, 64 + 2 * i)).collect();
    let mut at = 64 + 2 * n_layers;
    let mut calls = Vec::new();
    while at + 18 <= b.len() {
        assert_eq!(b[at], 1, "record tag at {at}");
        let flags = b[at + 1];
        let n = usize::from(le16(&b, at + 16));
        let size = 18 + 2 * n + n.div_ceil(8) + 24 + if flags & F_COUNTS != 0 { 4 * n } else { 0 };
        if at + size > b.len() {
            break;
        }
        calls.push(Call {
            flags,
            layer: le16(&b, at + 2),
            request: le32(&b, at + 4),
            pass: le32(&b, at + 8),
            ids: (0..n).map(|i| le16(&b, at + 18 + 2 * i)).collect(),
        });
        at += size;
    }
    Trace {
        width,
        layers,
        calls,
    }
}

/// What the engine did with a trace.
pub(super) struct Run {
    /// One flag per expert of every acquire-path call, in order: was it a hit.
    pub flags: Vec<bool>,
    pub decode_hits: u64,
    pub decode_misses: u64,
}

impl Run {
    pub(super) fn decode_hit_rate(&self) -> f64 {
        let total = self.decode_hits + self.decode_misses;
        if total == 0 {
            f64::NAN
        } else {
            self.decode_hits as f64 / total as f64
        }
    }
}

/// How one replay is set up: the same pool for every call, as the simulator does.
pub(super) struct Setup {
    pub n_slots: usize,
    pub kind: EvictionKind,
    pub bias: BiasParams,
}

impl Setup {
    pub(super) fn new(n_slots: usize, kind: EvictionKind) -> Self {
        Self {
            n_slots,
            kind,
            bias: BiasParams::default(),
        }
    }
}

/// Drives a [`SlotMap`] the way `MoeOffload::route` does: a staged call borrows the last
/// `regions * width` slots (once, until the next acquire-path call hands them back), an
/// acquire-path call plans, loads (instantly: there is no IO here), commits and releases.
struct Driver {
    map: SlotMap,
    n_slots: usize,
    stage_lo: usize,
    staging: bool,
}

impl Driver {
    fn new(trace: &Trace, setup: &Setup) -> Self {
        let per_pass = trace.layers.iter().filter(|&&l| l != MTP_LAYER).count();
        let mut cfg = PolicyConfig::new(setup.kind, trace.width, per_pass);
        cfg.bias = setup.bias;
        let regions = regions_for(setup.n_slots, trace.width);
        Self {
            map: SlotMap::with_policy(setup.n_slots, &cfg),
            n_slots: setup.n_slots,
            stage_lo: setup.n_slots - regions * trace.width,
            staging: false,
        }
    }

    fn enter_staging(&mut self) {
        if !self.staging {
            self.map.reserve_range(self.stage_lo..self.n_slots);
            self.staging = true;
        }
    }

    fn leave_staging(&mut self) {
        if self.staging {
            self.map.release_range(self.stage_lo..self.n_slots);
            self.staging = false;
        }
    }

    /// One acquire-path call: the staging blocks come back, then plan, load (instantly), commit,
    /// release.
    fn call(&mut self, keys: &[ExpertKey], kind: CallKind) -> Plan {
        self.leave_staging();
        let plan = self
            .map
            .plan_as(keys, kind)
            .expect("the pool holds one call");
        self.map.commit(&plan);
        self.map.release(&plan);
        plan
    }
}

fn keys_of(layer: u16, ids: &[u16], keys: &mut Vec<ExpertKey>) {
    keys.clear();
    keys.extend(ids.iter().map(|&e| ExpertKey::new(layer, e)));
}

pub(super) fn replay(trace: &Trace, setup: &Setup) -> Run {
    let mut d = Driver::new(trace, setup);
    let mut run = Run {
        flags: Vec::new(),
        decode_hits: 0,
        decode_misses: 0,
    };
    // Per request: decode tokens (highest pass + 1) and the hits and misses of its decode calls.
    let mut per_request: HashMap<u32, (u32, u64, u64)> = HashMap::new();
    let mut keys = Vec::new();
    for c in trace.calls.iter().filter(|c| c.request >= WARMUP_REQUESTS) {
        if c.flags & F_STAGED != 0 {
            d.enter_staging();
            continue;
        }
        let prefill = c.flags & F_PREFILL != 0;
        let kind = if prefill {
            CallKind::Prefill
        } else {
            CallKind::Decode
        };
        keys_of(c.layer, &c.ids, &mut keys);
        let plan = d.call(&keys, kind);
        // The misses come back in request order, so one pass over both lists tells them apart.
        let mut next = 0;
        for k in &keys {
            let miss = plan.loads.get(next).is_some_and(|l| l.0 == *k);
            next += usize::from(miss);
            run.flags.push(!miss);
        }
        if !prefill {
            let e = per_request.entry(c.request).or_default();
            e.0 = e.0.max(c.pass + 1);
            e.1 += (c.ids.len() - plan.loads.len()) as u64;
            e.2 += plan.loads.len() as u64;
        }
    }
    for &(tokens, hits, misses) in per_request.values() {
        if tokens > MIN_SCORED_TOKENS {
            run.decode_hits += hits;
            run.decode_misses += misses;
        }
    }
    run
}

/// The simulator's policy names: the three selectable ones, plus `bias-full`, `bias` with the
/// full 64-slot sample on every call (the configuration that was first evaluated).
fn setup_named(name: &str, n_slots: usize) -> Setup {
    let mut setup = Setup::new(n_slots, EvictionKind::Bias);
    match name {
        "bias-full" => setup.bias.large_sample = setup.bias.sample,
        other => setup.kind = EvictionKind::parse(Some(other)).unwrap_or_else(|e| panic!("{e}")),
    }
    setup
}

/// One line of the simulator's `summary.json`.
#[derive(serde::Deserialize)]
struct SimRow {
    session: String,
    policy: String,
    slots: usize,
    hit_rate: f64,
    /// File with the simulator's hit flag for every expert of every acquire-path call, 1 bit each.
    bits: String,
}

fn bit(bits: &[u8], i: usize) -> bool {
    bits[i / 8] >> (i % 8) & 1 == 1
}

/// `LUMEN_POLICY_TRACES`: the directory of `<session>_12g.bin` traces. `LUMEN_POLICY_SIM`: the
/// simulator's output directory (`summary.json` and the bit files it names).
pub(super) fn parity_report(traces: &Path, sim: &Path) -> Vec<String> {
    let rows: Vec<SimRow> =
        serde_json::from_slice(&std::fs::read(sim.join("summary.json")).expect("summary.json"))
            .expect("summary.json parses");
    let mut loaded: HashMap<String, Trace> = HashMap::new();
    let mut out = vec![format!(
        "{:<20} {:>6} {:>7} {:>9} {:>9} {:>8} {:>10}",
        "session", "slots", "policy", "engine %", "sim %", "diff pp", "flag diffs"
    )];
    let mut worst: f64 = 0.0;
    let mut diffs_total = 0usize;
    let mut decisions = 0usize;
    for r in &rows {
        let trace = loaded
            .entry(r.session.clone())
            .or_insert_with(|| read_trace(&traces.join(format!("{}_12g.bin", r.session))));
        let run = replay(trace, &setup_named(&r.policy, r.slots));
        let bits = std::fs::read(sim.join(&r.bits)).expect("bit file");
        assert!(
            bits.len() * 8 >= run.flags.len() && bits.len() * 8 < run.flags.len() + 8,
            "{} {} {}: the engine made {} decisions, the simulator {}",
            r.session,
            r.policy,
            r.slots,
            run.flags.len(),
            bits.len() * 8
        );
        let diffs = run
            .flags
            .iter()
            .enumerate()
            .filter(|&(i, &f)| bit(&bits, i) != f)
            .count();
        let diff_pp = 100.0 * (run.decode_hit_rate() - r.hit_rate);
        worst = worst.max(diff_pp.abs());
        diffs_total += diffs;
        decisions += run.flags.len();
        out.push(format!(
            "{:<20} {:>6} {:>7} {:>9.3} {:>9.3} {:>+8.3} {:>10}",
            r.session,
            r.slots,
            r.policy,
            100.0 * run.decode_hit_rate(),
            100.0 * r.hit_rate,
            diff_pp,
            diffs
        ));
    }
    out.push(format!(
        "{} replays, {decisions} decisions compared, worst difference {worst:.3} pp, \
         {diffs_total} differing decisions",
        rows.len()
    ));
    assert!(
        worst <= 0.2,
        "the engine departs from the simulator by {worst:.3} pp\n{}",
        out.join("\n")
    );
    out
}

/// Mean and 99th percentile of `ns`, in microseconds.
fn mean_p99_us(ns: &mut [u64]) -> (f64, f64) {
    if ns.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    ns.sort_unstable();
    let mean = ns.iter().sum::<u64>() as f64 / ns.len() as f64;
    (
        mean / 1e3,
        ns[(ns.len() * 99 / 100).min(ns.len() - 1)] as f64 / 1e3,
    )
}

/// Time `plan -> commit -> release` for every acquire-path call of `trace`, per policy, split by
/// the number of experts a call names: decode-sized (about 10), mid-size (100-200) and large
/// (250 and up). The pool is warmed with one pass first, as a server would be. Also the cost of
/// lending and returning the staging blocks, which is what a bulk call does to the map.
pub(super) fn cost_report(trace: &Trace, n_slots: usize, kinds: &[EvictionKind]) -> Vec<String> {
    // The cost of reading the clock itself.
    let timer_ns = {
        let t = Instant::now();
        let mut sink = 0u128;
        for _ in 0..200_000 {
            sink += Instant::now().elapsed().as_nanos();
        }
        std::hint::black_box(sink);
        t.elapsed().as_nanos() as u64 / 200_000
    };
    let mut out = vec![format!(
        "pool {n_slots} slots; timer overhead {timer_ns} ns subtracted; microseconds per call \
         (plan + commit + release), mean / p99"
    )];
    out.push(format!(
        "{:<8} {:>22} {:>22} {:>22} {:>22} {:>9}",
        "policy",
        "decode (<=40 experts)",
        "mid (100-200)",
        "large (250+)",
        "staging in+out",
        "hit rate"
    ));
    for &kind in kinds {
        let mut d = Driver::new(trace, &Setup::new(n_slots, kind));
        let mut keys = Vec::new();
        let mut buckets: [Vec<u64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut staging_ns: Vec<u64> = Vec::new();
        let (mut hits, mut misses) = (0u64, 0u64);
        let mut lent_for = 0u64;
        let net = |t: Instant| (t.elapsed().as_nanos() as u64).saturating_sub(timer_ns);
        for pass in 0..2 {
            for c in trace.calls.iter().filter(|c| c.request >= WARMUP_REQUESTS) {
                if c.flags & F_STAGED != 0 {
                    let t = Instant::now();
                    let fresh = !d.staging;
                    d.enter_staging();
                    if fresh {
                        lent_for = net(t);
                    }
                    continue;
                }
                if d.staging {
                    let t = Instant::now();
                    d.leave_staging();
                    if pass == 1 {
                        staging_ns.push(lent_for + net(t));
                    }
                }
                let ck = if c.flags & F_PREFILL != 0 {
                    CallKind::Prefill
                } else {
                    CallKind::Decode
                };
                keys_of(c.layer, &c.ids, &mut keys);
                let t = Instant::now();
                let plan = d.call(&keys, ck);
                let ns = net(t);
                if pass == 1 {
                    let n = keys.len();
                    let bucket = match n {
                        0..=40 => Some(0),
                        100..=200 => Some(1),
                        250.. => Some(2),
                        _ => None,
                    };
                    if let Some(b) = bucket {
                        buckets[b].push(ns);
                    }
                    hits += (n - plan.loads.len()) as u64;
                    misses += plan.loads.len() as u64;
                }
            }
        }
        let cell = |v: &mut Vec<u64>| {
            let n = v.len();
            let (m, p) = mean_p99_us(v);
            format!("{m:8.2} / {p:8.2} (n={n})")
        };
        let [dec, mid, big] = &mut buckets;
        out.push(format!(
            "{:<8} {:>22} {:>22} {:>22} {:>22} {:>8.1}%",
            kind.name(),
            cell(dec),
            cell(mid),
            cell(big),
            cell(&mut staging_ns),
            100.0 * hits as f64 / (hits + misses).max(1) as f64
        ));
    }
    out
}

/// Calls of exactly 10, 150 and 300 experts (a decode step, a mid-size prompt chunk, the largest
/// call the pool path takes before the staging path does) over random experts of a random layer,
/// into a pool that has been filled: most of each call misses, so this is the victim search at its
/// busiest. Prefill-sized calls are `CallKind::Prefill`, as the engine classifies them.
pub(super) fn busy_cost_report(
    n_slots: usize,
    width: usize,
    layers: usize,
    kinds: &[EvictionKind],
) -> Vec<String> {
    let timer_ns = {
        let t = Instant::now();
        let mut sink = 0u128;
        for _ in 0..200_000 {
            sink += Instant::now().elapsed().as_nanos();
        }
        std::hint::black_box(sink);
        t.elapsed().as_nanos() as u64 / 200_000
    };
    let mut out = vec![
        format!(
            "pool {n_slots} slots, {layers} layers x {width} experts; random experts, so most of a call misses; \\
             microseconds per call (plan + commit + release), mean / p99, and misses per call"
        ),
        format!(
            "{:<8} {:>34} {:>34} {:>34}",
            "policy", "10 experts (decode)", "150 experts (prefill)", "300 experts (prefill)"
        ),
    ];
    for &kind in kinds {
        let cfg = PolicyConfig::new(kind, width, layers);
        let mut map = SlotMap::with_policy(n_slots, &cfg);
        let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut pool: Vec<u16> = (0..width as u16).collect();
        let mut keys = Vec::new();
        let mut run =
            |map: &mut SlotMap, n: usize, ck: CallKind, timed: bool, ns: &mut Vec<u64>| -> usize {
                let layer = (next() % layers as u64) as u16;
                for i in 0..n {
                    let j = i + (next() % (width - i) as u64) as usize;
                    pool.swap(i, j);
                }
                keys_of(layer, &pool[..n], &mut keys);
                let t = Instant::now();
                let plan = map.plan_as(&keys, ck).expect("the pool holds one call");
                map.commit(&plan);
                map.release(&plan);
                if timed {
                    ns.push((t.elapsed().as_nanos() as u64).saturating_sub(timer_ns));
                }
                plan.loads.len()
            };
        let mut scratch = Vec::new();
        // Fill the pool and let the policy settle: ten tokens' worth of decode calls per slot.
        for _ in 0..n_slots {
            run(&mut map, 10, CallKind::Decode, false, &mut scratch);
        }
        let mut cells = Vec::new();
        for (n, ck, reps) in [
            (10, CallKind::Decode, 20_000),
            (150, CallKind::Prefill, 3_000),
            (300, CallKind::Prefill, 1_000),
        ] {
            let mut ns = Vec::with_capacity(reps);
            let misses: usize = (0..reps).map(|_| run(&mut map, n, ck, true, &mut ns)).sum();
            let (mean, p99) = mean_p99_us(&mut ns);
            cells.push(format!(
                "{mean:9.2} / {p99:9.2} ({:5.1} miss)",
                misses as f64 / reps as f64
            ));
        }
        out.push(format!(
            "{:<8} {:>34} {:>34} {:>34}",
            kind.name(),
            cells[0],
            cells[1],
            cells[2]
        ));
    }
    out
}

#[test]
#[ignore = "needs recorded routing traces and the simulator's output; set LUMEN_POLICY_TRACES and LUMEN_POLICY_SIM"]
fn engine_replay_matches_the_simulator() {
    let (Ok(traces), Ok(sim)) = (
        std::env::var("LUMEN_POLICY_TRACES"),
        std::env::var("LUMEN_POLICY_SIM"),
    ) else {
        return;
    };
    for line in parity_report(Path::new(&traces), Path::new(&sim)) {
        println!("{line}");
    }
}

#[test]
#[ignore = "needs a recorded routing trace; set LUMEN_POLICY_TRACE_FILE (and run with --release)"]
fn per_call_cost() {
    let Ok(file) = std::env::var("LUMEN_POLICY_TRACE_FILE") else {
        return;
    };
    let trace = read_trace(Path::new(&file));
    for n_slots in [2423, 7161] {
        for line in cost_report(&trace, n_slots, &EvictionKind::ALL) {
            println!("{line}");
        }
        for line in busy_cost_report(n_slots, trace.width, 48, &EvictionKind::ALL) {
            println!("{line}");
        }
    }
}
