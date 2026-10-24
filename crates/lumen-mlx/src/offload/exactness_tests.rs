//! Does a layer call's output depend on where the pool keeps its experts, or on how big the pool is?
//!
//! It must not: the offload engine promises output bit-identical to the model with every expert
//! resident. These tests drive the same `switch_glu` the model runs, over synthetic quantized
//! experts of the shipped layout (512 experts, top-10, 4-bit affine, group 64, bf16 scales),
//! and compare resident, pooled-in-scrambled-slots and staged runs bit for bit, at call sizes
//! from a decode step to a 1,024-row prefill chunk.
//!
//! What they pin down: MLX's sorted `gather_qmm` picks its kernel from the number of experts in
//! the stacked tensor it is given (see `TILED_GATHER_MIN_PER_EXPERT`), so the pool path, which
//! gathers over the whole pool, is not bit-identical to the resident gather above a certain
//! number of rows unless the call is staged. The rule that decides the path
//! (`takes_staging_path`) is tested here against the real kernels.

use std::collections::BTreeMap;

use mlx_rs::{Array, Dtype, random};

use super::slot_pool::PoolTensor;
use super::{distinct_in_order, split_entries, takes_staging_path};
use crate::native_moe::{
    SwitchExpertWeights, SwitchGluWeights, expert_entries, rejoin_entries, run_experts, switch_glu,
    weighted_sum,
};
use crate::native_quant::{MODE_AFFINE, quantize_with_mode};

const EXPERTS: usize = 512;
const TOP_K: usize = 10;
const GROUP: i32 = 64;

/// The shipped layer shape, shrunk (hidden 2560, intermediate 640) so a 4,340-slot pool fits a
/// test. The kernel choice depends on the number of index entries and experts, not on these.
const HIDDEN: i32 = 512;
const INTER: i32 = 256;

/// Call sizes in rows (one row = `TOP_K` index entries): a decode step, the sizes where the
/// gather is sorted but still per-row, either side of the resident model's switch to the tiled
/// kernel at 205 rows (`10 * 205 / 512 = 4`), and the prefill chunks the engine sees.
const ROWS: [usize; 12] = [1, 6, 7, 20, 64, 204, 205, 256, 410, 512, 682, 1024];

type Roles = BTreeMap<String, Array>;

/// One layer's `EXPERTS` experts, stacked like the checkpoint stores them.
fn build_layer() -> Roles {
    random::seed(7).expect("seed");
    let mut roles = Roles::new();
    for (proj, out, inp) in [
        ("gate_proj", INTER, HIDDEN),
        ("up_proj", INTER, HIDDEN),
        ("down_proj", HIDDEN, INTER),
    ] {
        let w = random::normal::<f32>(&[EXPERTS as i32, out, inp], None, None, None)
            .expect("normal")
            .as_dtype(Dtype::Bfloat16)
            .expect("bf16");
        let (wq, sc, bi) = quantize_with_mode(&w, GROUP, 4, MODE_AFFINE).expect("quantize");
        roles.insert(format!("{proj}.weight"), wq);
        roles.insert(format!("{proj}.scales"), sc);
        roles.insert(format!("{proj}.biases"), bi.expect("affine has biases"));
    }
    for a in roles.values() {
        a.eval().expect("eval layer");
    }
    roles
}

fn glu_of(arrays: &Roles) -> SwitchGluWeights<'_> {
    let proj = |p: &str| SwitchExpertWeights {
        weight: &arrays[&format!("{p}.weight")],
        scales: &arrays[&format!("{p}.scales")],
        biases: Some(&arrays[&format!("{p}.biases")]),
    };
    SwitchGluWeights {
        gate_proj: proj("gate_proj"),
        up_proj: proj("up_proj"),
        down_proj: proj("down_proj"),
        group_size: GROUP,
        bits: 4,
        mode: MODE_AFFINE,
    }
}

/// Tiny deterministic generator (xorshift64*).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, (self.next_u64() % (i as u64 + 1)) as usize);
        }
    }
}

/// `rows` routings of `TOP_K` distinct experts each, all drawn from `subset` experts.
fn routing(rows: usize, subset: usize, seed: u64) -> Vec<u32> {
    let mut rng = Rng(seed | 1);
    let mut all: Vec<u32> = (0..EXPERTS as u32).collect();
    rng.shuffle(&mut all);
    let pool = &all[..subset];
    let mut ids = Vec::with_capacity(rows * TOP_K);
    for _ in 0..rows {
        let mut row = pool.to_vec();
        rng.shuffle(&mut row);
        ids.extend_from_slice(&row[..TOP_K]);
    }
    ids
}

fn distinct(ids: &[u32]) -> usize {
    let mut v = ids.to_vec();
    v.sort_unstable();
    v.dedup();
    v.len()
}

/// Each expert's bytes out of a stacked `[EXPERTS, ...]` array.
fn expert_bytes(stacked: &Array) -> Vec<Vec<u8>> {
    let flat = stacked.view::<u8>().expect("byte view");
    flat.eval().expect("eval");
    let all = flat.try_as_slice::<u8>().expect("slice");
    all.chunks(all.len() / EXPERTS)
        .map(<[u8]>::to_vec)
        .collect()
}

/// A pool of `n_slots` slots per role with the layer's experts written at every placement in
/// `placements` (expert `e` at slot `placement[e]`).
fn pool_with(
    layer: &Roles,
    n_slots: usize,
    placements: &[&[usize]],
) -> BTreeMap<String, PoolTensor> {
    layer
        .iter()
        .map(|(role, stacked)| {
            let pool =
                PoolTensor::new(n_slots, &stacked.shape()[1..], stacked.dtype()).expect("pool");
            for (e, bytes) in expert_bytes(stacked).iter().enumerate() {
                for placement in placements {
                    // SAFETY: nothing runs on the GPU while the pool is being filled.
                    unsafe { pool.write_slot(placement[e], bytes) };
                }
            }
            (role.clone(), pool)
        })
        .collect()
}

fn whole_pool(pools: &BTreeMap<String, PoolTensor>) -> Roles {
    pools
        .iter()
        .map(|(r, p)| (r.clone(), p.array().clone()))
        .collect()
}

fn pool_view(pools: &BTreeMap<String, PoolTensor>, base: usize, len: usize) -> Roles {
    pools
        .iter()
        .map(|(r, p)| (r.clone(), p.view(base, len).expect("view")))
        .collect()
}

/// `switch_glu` over `arrays` with `ids` as the gather indices: `[rows, TOP_K, HIDDEN]` as f32.
fn run(x: &Array, ids: &[u32], arrays: &Roles) -> Vec<f32> {
    let rows = ids.len() / TOP_K;
    let inds = Array::from_slice(ids, &[1, rows as i32, TOP_K as i32]);
    let y = switch_glu(x, &inds, &glu_of(arrays)).expect("switch_glu");
    let y = y.as_dtype(Dtype::Float32).expect("f32");
    y.eval().expect("eval");
    y.try_as_slice::<f32>().expect("slice").to_vec()
}

/// How far two outputs are apart: elements whose bits differ, and the largest gap relative to
/// the largest value.
fn gap(a: &[f32], b: &[f32]) -> (usize, f32) {
    assert_eq!(a.len(), b.len());
    let scale = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
    let differing = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let max_abs = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    (differing, max_abs / scale)
}

fn activations(rows: usize) -> Array {
    random::normal::<f32>(&[1, rows as i32, HIDDEN], None, None, None)
        .expect("x")
        .as_dtype(Dtype::Bfloat16)
        .expect("bf16")
}

/// A random placement of the layer's experts into distinct slots of `0..n_slots`.
fn scrambled_slots(n_slots: usize, seed: u64) -> Vec<usize> {
    let mut slots: Vec<usize> = (0..n_slots).collect();
    Rng(seed | 1).shuffle(&mut slots);
    slots.truncate(EXPERTS);
    slots
}

/// Which slots the experts sit in does not matter: the gather is row by row, and the sorted
/// kernel's tiling follows the sorted order of the slots without changing a row's arithmetic.
#[test]
fn which_slots_hold_the_experts_does_not_change_the_gather() {
    let layer = build_layer();
    let placements: Vec<Vec<usize>> = (0..4).map(|k| scrambled_slots(1024, 100 + k)).collect();
    let refs: Vec<&[usize]> = placements.iter().map(Vec::as_slice).collect();
    // One pool per placement: every run sees the same pool size, only the slots differ.
    let pools: Vec<_> = refs
        .iter()
        .map(|p| whole_pool(&pool_with(&layer, 1024, &[p])))
        .collect();
    for rows in ROWS {
        let x = activations(rows);
        let ids = routing(rows, 250, 5 + rows as u64);
        let outputs: Vec<Vec<f32>> = placements
            .iter()
            .zip(&pools)
            .map(|(placement, pool)| {
                let slot_ids: Vec<u32> =
                    ids.iter().map(|&e| placement[e as usize] as u32).collect();
                run(&x, &slot_ids, pool)
            })
            .collect();
        for (k, out) in outputs.iter().enumerate().skip(1) {
            let (differing, rel) = gap(&outputs[0], out);
            assert_eq!(
                differing, 0,
                "{rows} rows: placement {k} differs from placement 0 in {differing} elements \
                 (largest gap {rel:.3e} of the largest value)"
            );
        }
    }
}

/// The staging path's gather over a view of one layer is the resident gather, at every size.
/// The pool path's is, too, for every call the engine sends it, at every pool size: the rule
/// that picks the path (`takes_staging_path`) is what makes the pool size irrelevant.
#[test]
fn the_path_a_call_takes_gives_the_resident_output_at_every_pool_size() {
    let layer = build_layer();
    // Pool sizes of the 4g, 6.7g, 12g budgets and the smallest pool (a region and a layer).
    for n_slots in [1024usize, 1449, 2423, 4340] {
        let region = n_slots - EXPERTS;
        let in_pool = scrambled_slots(region, 11);
        let in_region: Vec<usize> = (0..EXPERTS).map(|e| region + e).collect();
        let pools = pool_with(&layer, n_slots, &[&in_pool, &in_region]);
        let whole = whole_pool(&pools);
        let staged_view = pool_view(&pools, region, EXPERTS);
        for rows in ROWS {
            for (subset, seed) in [(250usize, 5u64), (480, 9)] {
                let x = activations(rows);
                let ids = routing(rows, subset, seed + rows as u64);
                let want = run(&x, &ids, &layer);

                let staged = run(&x, &ids, &staged_view);
                assert_eq!(
                    gap(&want, &staged).0,
                    0,
                    "{n_slots} slots, {rows} rows: the staged gather differs from the resident one"
                );

                let stages = takes_staging_path(false, distinct(&ids), ids.len(), EXPERTS);
                let got = if stages {
                    staged
                } else {
                    let slot_ids: Vec<u32> =
                        ids.iter().map(|&e| in_pool[e as usize] as u32).collect();
                    run(&x, &slot_ids, &whole)
                };
                let (differing, rel) = gap(&want, &got);
                assert_eq!(
                    differing,
                    0,
                    "{n_slots} slots, {rows} rows, {} distinct experts ({}): {differing} elements \
                     differ from the resident gather (largest gap {rel:.3e} of the largest value)",
                    distinct(&ids),
                    if stages { "staged" } else { "pool path" }
                );
            }
        }
    }
}

/// The calls `LUMEN_MOE_OVERLAP` splits: a decode step and the verify sizes (`trace::DECODE_ROWS`).
const SPLIT_ROWS: [usize; 4] = [1, 2, 3, 4];

/// Every way of marking `n` distinct experts missing that the overlap uses (at least one resident
/// and one missing): one resident, one missing, every other, and two random shares.
fn missing_masks(n: usize, seed: u64) -> Vec<Vec<bool>> {
    let mut rng = Rng(seed | 1);
    let mut random = |share_of_8: u64| -> Vec<bool> {
        let mut m: Vec<bool> = (0..n).map(|_| rng.next_u64() % 8 < share_of_8).collect();
        m[0] = false; // keep one resident
        m[n - 1] = true; // and one missing
        m
    };
    let (third, two_thirds) = (random(3), random(5));
    vec![
        (0..n).map(|i| i != 0).collect(),
        (0..n).map(|i| i == n - 1).collect(),
        (0..n).map(|i| i % 2 == 1).collect(),
        third,
        two_thirds,
    ]
}

fn bits(a: &Array) -> Vec<u32> {
    let a = a.as_dtype(Dtype::Float32).expect("f32");
    a.eval().expect("eval");
    a.try_as_slice::<f32>()
        .expect("slice")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// An overlapped call gathers its resident entries and its missing ones separately (one row per
/// index, no sort) and puts the outputs back in the routing's order before the weighted sum. Each
/// entry must come out with the bits the whole routing gives it, for every pool size, slot
/// placement, row count and resident share, so the block's output is the plain path's. Both
/// activation dtypes: f32 activations against bf16 scales (the shipped model, cast per call) and
/// bf16 throughout.
#[test]
fn a_call_split_into_resident_and_missing_entries_gives_the_whole_calls_bits() {
    let layer = build_layer();
    for n_slots in [1024usize, 1449, 2423, 4340] {
        let placement = scrambled_slots(n_slots, 31 + n_slots as u64);
        let pool = whole_pool(&pool_with(&layer, n_slots, &[&placement]));
        let glu = glu_of(&pool);
        for rows in SPLIT_ROWS {
            let ids = routing(rows, 250, 40 + rows as u64);
            let slot_ids: Vec<u32> = ids.iter().map(|&e| placement[e as usize] as u32).collect();
            let shape = [1, rows as i32, TOP_K as i32];
            let inds = Array::from_slice(&slot_ids, &shape);
            let mut rng = Rng(rows as u64 * 977 + n_slots as u64);
            let score_vals: Vec<f32> = (0..ids.len())
                .map(|_| (rng.next_u64() % 1000) as f32 / 1000.0)
                .collect();
            let scores = Array::from_slice(&score_vals, &shape);
            let (distinct, index_of) = distinct_in_order(&ids).expect("distinct");
            let slots: Vec<usize> = distinct.iter().map(|&e| placement[e as usize]).collect();
            for (dtype, in_scale_dtype) in [(Dtype::Float32, true), (Dtype::Bfloat16, false)] {
                let x = random::normal::<f32>(&[1, rows as i32, HIDDEN], None, None, None)
                    .expect("x")
                    .as_dtype(dtype)
                    .expect("dtype");
                let whole_y = run_experts(&x, &inds, &glu, in_scale_dtype).expect("whole");
                let whole_sum = weighted_sum(&whole_y, &scores).expect("sum");
                let (want_y, want_sum) = (bits(&whole_y), bits(&whole_sum));
                assert!(want_y.iter().filter(|&&b| b != 0).count() > want_y.len() / 2);
                for (m, missing) in missing_masks(distinct.len(), rows as u64)
                    .iter()
                    .enumerate()
                {
                    let parts = split_entries(&ids, TOP_K, &index_of, &slots, missing);
                    let arr = |v: &[u32]| Array::from_slice(v, &[v.len() as i32]);
                    let part = |r: &[u32], s: &[u32]| {
                        expert_entries(&x, &arr(r), &arr(s), &glu, in_scale_dtype).expect("part")
                    };
                    let resident = part(&parts.resident_rows, &parts.resident_slots);
                    let absent = part(&parts.missing_rows, &parts.missing_slots);
                    let y = rejoin_entries(&resident, &absent, &arr(&parts.order), &shape)
                        .expect("rejoin");
                    let what = format!(
                        "{n_slots} slots, {rows} rows, {dtype:?}, mask {m} ({} of {} experts missing)",
                        missing.iter().filter(|&&b| b).count(),
                        distinct.len()
                    );
                    assert_eq!(y.shape(), whole_y.shape(), "{what}");
                    let got_y = bits(&y);
                    let differing = got_y.iter().zip(&want_y).filter(|(a, b)| a != b).count();
                    assert_eq!(differing, 0, "{what}: {differing} expert outputs differ");
                    assert!(
                        bits(&weighted_sum(&y, &scores).expect("sum")) == want_sum,
                        "{what}: the weighted sum differs"
                    );
                }
            }
        }
    }
}
