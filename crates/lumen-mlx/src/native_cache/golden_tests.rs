//! Golden digests for every KV cache kind.
//!
//! Each case drives one cache through a fixed plan (prefill chunks, decode
//! across the 256-token growth step and past the end of a ring, rewinds)
//! under both settings of `LUMEN_NATIVE_KV_STEP_PREALLOC`. After every step it
//! folds into one FNV-1a digest everything the cache returned, every backing
//! buffer (zero padding and stale slots included) and the cache's counters.
//!
//! The digests were recorded from the per-type update paths before those were
//! folded into shared ring helpers. They pin bits, not tolerances: a helper
//! that grows a buffer to the wrong size, fills it from the wrong input,
//! reorders a ring or writes one slot off changes them. Inputs come from a
//! host-side SplitMix64 rather than `mlx_rs::random`, so only the cache code
//! and, for the affine-quantized kinds, MLX's `quantize` feed the digests.
//! Re-record only for a deliberate change in behaviour; the failure message
//! prints the table to paste back.

use super::imp::{
    NativeArraysCache, NativeKvCache, NativeKvCacheQuantized, NativeLayerCache,
    NativeRotatingKvCache, NativeRotatingKvCacheQuantized, NativeRotatingKvCacheTurboQuant,
    kv_step_prealloc,
};
use mlx_rs::{Array, Dtype};

/// SplitMix64, so the inputs do not depend on MLX's generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// `[1, h, s, d]` bf16 keys or values in [-2, 2).
    fn bf16(&mut self, h: usize, s: usize, d: usize) -> Array {
        let data: Vec<f32> = (0..h * s * d)
            .map(|_| (self.next() >> 40) as f32 / (1u64 << 22) as f32 - 2.0)
            .collect();
        Array::from_slice(&data, &shape(h, s, d))
            .as_dtype(Dtype::Bfloat16)
            .unwrap()
    }

    /// `[1, h, s, d]` f32 in [0.5, 1.5): a TurboQuant σ or residual norm.
    fn positive(&mut self, h: usize, s: usize, d: usize) -> Array {
        let data: Vec<f32> = (0..h * s * d)
            .map(|_| (self.next() >> 40) as f32 / (1u64 << 24) as f32 + 0.5)
            .collect();
        Array::from_slice(&data, &shape(h, s, d))
    }

    /// `[1, h, s, d]` Lloyd-Max codes below 16.
    fn codes(&mut self, h: usize, s: usize, d: usize) -> Array {
        let data: Vec<u8> = (0..h * s * d).map(|_| (self.next() >> 60) as u8).collect();
        Array::from_slice(&data, &shape(h, s, d))
    }

    /// `[1, h, s, d]` u32 words: packed QJL signs.
    fn words(&mut self, h: usize, s: usize, d: usize) -> Array {
        let data: Vec<u32> = (0..h * s * d).map(|_| self.next() as u32).collect();
        Array::from_slice(&data, &shape(h, s, d))
    }
}

fn shape(h: usize, s: usize, d: usize) -> [i32; 4] {
    [1, h as i32, s as i32, d as i32]
}

/// FNV-1a over shapes, dtypes and element bits.
struct Digest(u64);

impl Digest {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    }

    fn num(&mut self, n: usize) {
        self.bytes(&(n as u64).to_le_bytes());
    }

    fn array(&mut self, a: &Array) {
        self.num(a.ndim());
        for &d in a.shape() {
            self.num(d as usize);
        }
        if a.size() == 0 {
            return;
        }
        // `as_slice` reads the raw buffer and ignores strides, and a fetch is
        // usually a strided view over a larger ring, so copy it out row-major
        // first. `x * 1` is exact for every dtype here, signed zeros included.
        let one = Array::from_int(1).as_dtype(a.dtype()).unwrap();
        let dense = mlx_rs::ops::multiply(a, &one).unwrap();
        match a.dtype() {
            Dtype::Uint8 => {
                self.num(1);
                self.bytes(dense.as_slice::<u8>());
            }
            Dtype::Uint32 => {
                self.num(2);
                for w in dense.as_slice::<u32>() {
                    self.bytes(&w.to_le_bytes());
                }
            }
            Dtype::Float32 => {
                self.num(3);
                self.f32_bits(&dense);
            }
            Dtype::Bfloat16 => {
                // bf16 → f32 is exact, and mlx-rs has no bf16 element type
                // without the `half` crate.
                self.num(4);
                self.f32_bits(&dense.as_dtype(Dtype::Float32).unwrap());
            }
            other => panic!("golden digest: no reader for {other:?}"),
        }
    }

    fn f32_bits(&mut self, a: &Array) {
        for x in a.as_slice::<f32>() {
            self.bytes(&x.to_bits().to_le_bytes());
        }
    }

    fn opt(&mut self, a: Option<&Array>) {
        match a {
            Some(a) => {
                self.num(1);
                self.array(a);
            }
            None => self.num(0),
        }
    }

    fn triple(&mut self, t: Option<&(Array, Array, Array)>) {
        match t {
            Some((w, s, b)) => {
                self.num(1);
                self.array(w);
                self.array(s);
                self.array(b);
            }
            None => self.num(0),
        }
    }
}

#[derive(Clone, Copy)]
enum Step {
    /// One update of this many tokens.
    Chunk(usize),
    /// This many single-token updates: the decode path.
    Decode(usize),
    /// `truncate_to(target)`; whether it was accepted goes into the digest.
    Rewind(usize),
}

use Step::{Chunk, Decode, Rewind};

/// A cache under test, with the input shapes its kind takes.
trait Case {
    /// Push `s` fresh tokens and fold what comes back.
    fn update(&mut self, s: usize, rng: &mut Rng, d: &mut Digest);
    fn rewind(&mut self, target: usize) -> bool;
    /// Fold every backing buffer and counter.
    fn state(&self, d: &mut Digest);
}

/// Grows in a 256 block from decode, takes a prefill that fits it, decodes
/// across the boundary, takes a prefill needing two blocks from an unaligned
/// offset, then rewinds into the stale tail, past the end (refused) and to
/// where it already is.
const FULL: &[Step] = &[
    Decode(3),
    Chunk(250),
    Decode(5),
    Chunk(300),
    Rewind(500),
    Decode(3),
    Rewind(600),
    Chunk(7),
    Rewind(510),
    Decode(2),
];

/// A 12-slot ring with 2 sinks: decode, a rewind before the ring has
/// wrapped, a chunk that overfills it (the next decode trims once), decode
/// that wraps, a chunk longer than the window over the wrapped ring, and a
/// rewind after wrapping, which is refused.
const SMALL_RING: &[Step] = &[
    Chunk(5),
    Decode(3),
    Rewind(6),
    Decode(4),
    Chunk(3),
    Decode(10),
    Chunk(20),
    Decode(5),
    Rewind(30),
    Chunk(4),
    Decode(3),
];

/// A 300-slot ring with 4 sinks, large enough that decode grows it by a 256
/// block and then by a partial one up to `max_size` before it wraps. A rewind
/// lands inside the first block's padding; a chunk over the wrapped ring
/// overfills it, so decode trims once more.
const LARGE_RING: &[Step] = &[
    Decode(3),
    Rewind(2),
    Chunk(260),
    Decode(40),
    Chunk(30),
    Decode(3),
    Rewind(50),
    Chunk(2),
];

/// `(name, max_size, keep, steps)`. Full caches ignore the geometry.
type Plan = (&'static str, usize, usize, &'static [Step]);

const RINGS: &[Plan] = &[("small", 12, 2, SMALL_RING), ("large", 300, 4, LARGE_RING)];

/// TurboQuant also serves full-attention layers as an append-only buffer
/// (`keep = 0`, `max_size` the context length).
const TQ_PLANS: &[Plan] = &[
    ("small", 12, 2, SMALL_RING),
    ("large", 300, 4, LARGE_RING),
    ("full", 4096, 0, FULL),
];

type Recorded = (&'static str, bool, u64);

fn run<C: Case>(mut cache: C, steps: &[Step], prealloc: bool) -> u64 {
    kv_step_prealloc::with(prealloc, || {
        let mut rng = Rng(0x0022_dedc);
        let mut d = Digest::new();
        for &step in steps {
            let (times, step) = match step {
                Decode(n) => (n, Chunk(1)),
                other => (1, other),
            };
            for _ in 0..times {
                match step {
                    Chunk(s) => cache.update(s, &mut rng, &mut d),
                    Rewind(target) => d.num(usize::from(cache.rewind(target))),
                    Decode(_) => unreachable!("expanded above"),
                }
                cache.state(&mut d);
            }
        }
        d.0
    })
}

fn digests<C: Case>(plans: &[Plan], make: impl Fn(usize, usize) -> C) -> Vec<Recorded> {
    let mut out = Vec::new();
    for &(name, max_size, keep, steps) in plans {
        for prealloc in [false, true] {
            out.push((name, prealloc, run(make(max_size, keep), steps, prealloc)));
        }
    }
    out
}

fn check(got: Vec<Recorded>, recorded: &[Recorded]) {
    if got.as_slice() != recorded {
        let table: String = got
            .iter()
            .map(|(name, prealloc, h)| format!("        (\"{name}\", {prealloc}, {h:#018x}),\n"))
            .collect();
        panic!("digests differ from the recording; this run produced:\n{table}");
    }
}

const H: usize = 2;

impl Case for NativeKvCache {
    fn update(&mut self, s: usize, rng: &mut Rng, d: &mut Digest) {
        let (k, v) = (rng.bf16(H, s, 16), rng.bf16(H, s, 24));
        let (k, v) = self.update_and_fetch(&k, &v).unwrap();
        d.array(&k);
        d.array(&v);
    }
    fn rewind(&mut self, target: usize) -> bool {
        self.truncate_to(target).is_ok()
    }
    fn state(&self, d: &mut Digest) {
        d.opt(self.keys());
        d.opt(self.values());
        d.num(self.offset());
        d.num(self.base_offset());
    }
}

impl Case for NativeKvCacheQuantized {
    fn update(&mut self, s: usize, rng: &mut Rng, d: &mut Digest) {
        let (k, v) = (rng.bf16(H, s, 64), rng.bf16(H, s, 128));
        let (k, v) = self.update_and_fetch(&k, &v).unwrap();
        d.triple(Some(&k));
        d.triple(Some(&v));
    }
    fn rewind(&mut self, target: usize) -> bool {
        self.truncate_to(target).is_ok()
    }
    fn state(&self, d: &mut Digest) {
        d.triple(self.keys());
        d.triple(self.values());
        d.num(self.offset());
    }
}

impl Case for NativeRotatingKvCache {
    fn update(&mut self, s: usize, rng: &mut Rng, d: &mut Digest) {
        let (k, v) = (rng.bf16(H, s, 16), rng.bf16(H, s, 24));
        let (k, v) = self.update_and_fetch(&k, &v).unwrap();
        d.array(&k);
        d.array(&v);
    }
    fn rewind(&mut self, target: usize) -> bool {
        self.truncate_to(target).is_ok()
    }
    fn state(&self, d: &mut Digest) {
        d.opt(self.keys());
        d.opt(self.values());
        d.num(self.offset());
        d.num(self.idx());
        d.num(self.cached_len());
    }
}

impl Case for NativeRotatingKvCacheQuantized {
    fn update(&mut self, s: usize, rng: &mut Rng, d: &mut Digest) {
        let (k, v) = (rng.bf16(H, s, 64), rng.bf16(H, s, 128));
        let (k, v) = self.update_and_fetch(&k, &v).unwrap();
        d.triple(Some(&k));
        d.triple(Some(&v));
    }
    fn rewind(&mut self, target: usize) -> bool {
        self.truncate_to(target).is_ok()
    }
    fn state(&self, d: &mut Digest) {
        d.triple(self.keys());
        d.triple(self.values());
        d.num(self.offset());
        d.num(self.idx());
        d.num(self.cached_len());
    }
}

fn turboquant_state(c: &NativeRotatingKvCacheTurboQuant, d: &mut Digest) {
    d.opt(c.keys_codes());
    d.opt(c.keys_sigma());
    d.opt(c.values_codes());
    d.opt(c.values_sigma());
    d.opt(c.keys_signs());
    d.opt(c.keys_residual_norm());
    d.num(c.offset());
    d.num(c.idx());
    d.num(c.cached_len());
}

impl Case for NativeRotatingKvCacheTurboQuant {
    fn update(&mut self, s: usize, rng: &mut Rng, d: &mut Digest) {
        let (kc, ks) = (rng.codes(H, s, 16), rng.positive(H, s, 1));
        let (vc, vs) = (rng.codes(H, s, 24), rng.positive(H, s, 1));
        let ((kc, ks), (vc, vs)) = self.update_and_fetch(&kc, &ks, &vc, &vs).unwrap();
        for a in [&kc, &ks, &vc, &vs] {
            d.array(a);
        }
    }
    fn rewind(&mut self, target: usize) -> bool {
        self.truncate_to(target).is_ok()
    }
    fn state(&self, d: &mut Digest) {
        turboquant_state(self, d);
    }
}

/// TurboQuant with its QJL Stage-2 slots, through `update_and_fetch_qjl`.
struct Qjl(NativeRotatingKvCacheTurboQuant);

impl Case for Qjl {
    fn update(&mut self, s: usize, rng: &mut Rng, d: &mut Digest) {
        let (kc, ks) = (rng.codes(H, s, 16), rng.positive(H, s, 1));
        let (ksg, krn) = (rng.words(H, s, 2), rng.positive(H, s, 1));
        let (vc, vs) = (rng.codes(H, s, 24), rng.positive(H, s, 1));
        let ((kc, ks, ksg, krn), (vc, vs)) = self
            .0
            .update_and_fetch_qjl(&kc, &ks, &ksg, &krn, &vc, &vs)
            .unwrap();
        for a in [&kc, &ks, &ksg, &krn, &vc, &vs] {
            d.array(a);
        }
    }
    fn rewind(&mut self, target: usize) -> bool {
        self.0.truncate_to(target).is_ok()
    }
    fn state(&self, d: &mut Digest) {
        turboquant_state(&self.0, d);
    }
}

#[test]
fn full_cache_matches_its_recording() {
    check(
        digests(&[("full", 0, 0, FULL)], |_, _| NativeKvCache::new()),
        &[
            ("full", false, 0x27b1ca76b702ebf0),
            ("full", true, 0xf074133e39cc9bff),
        ],
    );
}

#[test]
fn quantized_full_cache_matches_its_recording() {
    check(
        digests(&[("full", 0, 0, FULL)], |_, _| {
            NativeKvCacheQuantized::new(64, 4)
        }),
        &[
            ("full", false, 0x68f1e4fa5d294418),
            ("full", true, 0xe3d184f221cb6d4e),
        ],
    );
}

#[test]
fn rotating_cache_matches_its_recording() {
    check(
        digests(RINGS, NativeRotatingKvCache::new),
        &[
            ("small", false, 0x5834e15bbc8d76b6),
            ("small", true, 0xd1f88e7dfd07ea48),
            ("large", false, 0x2aa678acafcd12ff),
            ("large", true, 0x72aa165f2e2bd6cf),
        ],
    );
}

#[test]
fn rotating_quantized_cache_matches_its_recording() {
    check(
        digests(RINGS, |max_size, keep| {
            NativeRotatingKvCacheQuantized::new(max_size, keep, 64, 4)
        }),
        &[
            ("small", false, 0xcc42d273a7d56080),
            ("small", true, 0xf3a5dd7dbe5223ce),
            ("large", false, 0x7418140411240bfc),
            ("large", true, 0x2812feb187ccff94),
        ],
    );
}

#[test]
fn turboquant_cache_matches_its_recording() {
    check(
        digests(TQ_PLANS, |max_size, keep| {
            NativeRotatingKvCacheTurboQuant::new(max_size, keep, 4)
        }),
        &[
            ("small", false, 0x317e3bff116a9ee8),
            ("small", true, 0xecce0a8165fc6b99),
            ("large", false, 0x9212159c7532a30b),
            ("large", true, 0x4f12b91ee3f90b33),
            ("full", false, 0xf9cab87f57fb453b),
            ("full", true, 0xf2cf59207a0c8e31),
        ],
    );
}

#[test]
fn turboquant_qjl_cache_matches_its_recording() {
    check(
        digests(TQ_PLANS, |max_size, keep| {
            Qjl(NativeRotatingKvCacheTurboQuant::new(max_size, keep, 4).with_qjl(64))
        }),
        &[
            ("small", false, 0x0f372b77b7090518),
            ("small", true, 0x602909e0d05e6f27),
            ("large", false, 0xe01557c21b1fd2ce),
            ("large", true, 0xb978336b3d4715d2),
            ("full", false, 0x2931526276ecf5f7),
            ("full", true, 0xc0a764840e88a73b),
        ],
    );
}

/// Every per-layer accessor answers for its own kind and names the kind it
/// wanted otherwise; `offset` and `clear` reach every kind.
#[test]
fn layer_cache_accessors_dispatch_on_kind() {
    let mut rng = Rng(7);
    let mut full = NativeKvCache::new();
    let (k, v) = (rng.bf16(H, 3, 16), rng.bf16(H, 3, 16));
    full.update_and_fetch(&k, &v).unwrap();
    let mut quantized = NativeKvCacheQuantized::new(64, 4);
    let (k, v) = (rng.bf16(H, 3, 64), rng.bf16(H, 3, 64));
    quantized.update_and_fetch(&k, &v).unwrap();
    let mut turboquant = NativeRotatingKvCacheTurboQuant::new(4096, 0, 4);
    let (kc, ks) = (rng.codes(H, 3, 16), rng.positive(H, 3, 1));
    turboquant.update_and_fetch(&kc, &ks, &kc, &ks).unwrap();
    let mut linear = NativeArraysCache::new(2);
    linear.advance(3);

    let mut layers = [
        NativeLayerCache::Full(full),
        NativeLayerCache::Linear(linear),
        NativeLayerCache::FullTurboquant(turboquant),
        NativeLayerCache::FullQuantized(quantized),
    ];
    let wanted = ["Full", "Linear", "FullTurboquant", "FullQuantized"];
    for (layer, own) in layers.iter_mut().zip(wanted) {
        assert_eq!(layer.offset(), 3, "{own}");
        assert_eq!(layer.as_full().is_some(), own == "Full");
        assert_eq!(layer.as_linear().is_some(), own == "Linear");
        let errors = [
            ("as_full_mut", layer.as_full_mut().err()),
            ("as_linear_mut", layer.as_linear_mut().err()),
            (
                "as_full_turboquant_mut",
                layer.as_full_turboquant_mut().err(),
            ),
            ("as_full_quantized_mut", layer.as_full_quantized_mut().err()),
        ];
        for ((accessor, err), kind) in errors.into_iter().zip(wanted) {
            match err {
                None => assert_eq!(kind, own, "{accessor} accepted a {own} layer"),
                Some(e) => assert_eq!(
                    e.to_string(),
                    format!("NativeLayerCache::{accessor} called on a non-{kind} layer"),
                    "{own}"
                ),
            }
        }
        layer.clear();
        assert_eq!(layer.offset(), 0, "{own} after clear");
    }
}
