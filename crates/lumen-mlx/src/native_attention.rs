//! Native MLX scaled-dot-product attention wrapper.
//!
//! Routes through `mlx_rs::fast::scaled_dot_product_attention`, which calls
//! `mlx_fast_scaled_dot_product_attention` directly. Per MLX docs, the q_seq=1
//! decode path dispatches to an optimized Metal kernel; longer prefills fall
//! back to ops::matmul + softmax + ops::matmul. Both regimes are
//! deterministic for fixed inputs, so a correct binding produces
//! bit-identical output vs MLX's Python reference.
//!
//! GQA support is implicit: `keys`/`values` may have fewer heads than
//! `queries`; MLX handles the broadcast internally without pre-tiling.
//!
//! See `.ai/memory/active/mlx-rs-native-port/CONTEXT.md` Session 21+.

#[cfg(feature = "mlx-native")]
mod imp {
    use anyhow::{Context, Result};
    use mlx_rs::Array;
    use mlx_rs::fast::ScaledDotProductAttentionMask;
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    // lumen-rs Phase 1.6: Rust-side per-stage timing inside our sdpa()
    // wrapper. Splits the bucket into:
    //   - pre: mask construction (enum value, no alloc)
    //   - call: mlx_rs::fast::scaled_dot_product_attention(...)
    //   - post: `.context(...)` chain + Result unwrap
    // Activated via `LUMEN_NATIVE_SDPA_TIMING_DUMP=1`. Visibility is
    // `pub` (not `pub(crate)`) so the bench example (separate crate
    // boundary as an example target) can call `reset` / `dump`.
    pub mod lumen_sdpa_timing {
        use std::sync::atomic::{AtomicU64, Ordering};
        pub static CALLS: AtomicU64 = AtomicU64::new(0);
        pub static PRE_NS: AtomicU64 = AtomicU64::new(0);
        pub static CALL_NS: AtomicU64 = AtomicU64::new(0);
        pub static POST_NS: AtomicU64 = AtomicU64::new(0);
        pub static TOTAL_NS: AtomicU64 = AtomicU64::new(0);

        pub fn reset() {
            CALLS.store(0, Ordering::Relaxed);
            PRE_NS.store(0, Ordering::Relaxed);
            CALL_NS.store(0, Ordering::Relaxed);
            POST_NS.store(0, Ordering::Relaxed);
            TOTAL_NS.store(0, Ordering::Relaxed);
        }

        pub fn dump() {
            let calls = CALLS.load(Ordering::Relaxed);
            if calls == 0 {
                eprintln!("[native-sdpa-timing] no SDPA calls observed");
                return;
            }
            let fmt = |name: &str, bucket: &AtomicU64| {
                let ns = bucket.load(Ordering::Relaxed);
                let total_ms = ns as f64 / 1e6;
                let per_call_us = ns as f64 / 1000.0 / calls as f64;
                eprintln!(
                    "[native-sdpa-timing] {:<22} {:10.3} ms total   {:10.3} us/call",
                    name, total_ms, per_call_us
                );
            };
            eprintln!(
                "[native-sdpa-timing] === native_attention::sdpa breakdown (calls={}) ===",
                calls
            );
            fmt("pre", &PRE_NS);
            fmt("call (mlx-rs entry)", &CALL_NS);
            fmt("post", &POST_NS);
            fmt("TOTAL", &TOTAL_NS);
        }
    }

    /// SDPA attention path. `causal=true` applies the standard causal mask
    /// (upper-triangle = -inf in softmax space) — appropriate for prefill.
    /// `causal=false` does no masking — appropriate for single-token decode
    /// against a fully-cached KV.
    pub fn sdpa(
        queries: &Array,
        keys: &Array,
        values: &Array,
        scale: f32,
        causal: bool,
    ) -> Result<Array> {
        let t_start = Instant::now();
        let mask = if causal {
            Some(ScaledDotProductAttentionMask::Causal)
        } else {
            None
        };
        let t_after_pre = Instant::now();
        let raw = mlx_rs::fast::scaled_dot_product_attention(
            queries, keys, values, scale, mask, None, /* sinks */
        );
        let t_after_call = Instant::now();
        let final_result = raw.context("mlx-rs fast::scaled_dot_product_attention FFI call failed");
        let t_end = Instant::now();
        lumen_sdpa_timing::PRE_NS
            .fetch_add((t_after_pre - t_start).as_nanos() as u64, Ordering::Relaxed);
        lumen_sdpa_timing::CALL_NS.fetch_add(
            (t_after_call - t_after_pre).as_nanos() as u64,
            Ordering::Relaxed,
        );
        lumen_sdpa_timing::POST_NS
            .fetch_add((t_end - t_after_call).as_nanos() as u64, Ordering::Relaxed);
        lumen_sdpa_timing::TOTAL_NS
            .fetch_add((t_end - t_start).as_nanos() as u64, Ordering::Relaxed);
        lumen_sdpa_timing::CALLS.fetch_add(1, Ordering::Relaxed);
        final_result
    }

    /// Log-sum-exp of the attention scores `scale * (Q · Kᵀ)` over the key
    /// axis, returning `[B, H, Lq, 1]`. This is the normalization statistic a
    /// flash-style softmax merge needs but `scaled_dot_product_attention` does
    /// not expose. GQA is handled by broadcasting each KV head across its query
    /// head group (kv head `j` serves query heads `[j·G, (j+1)·G)`, matching
    /// MLX's internal repeat), so the scores here line up 1:1 with the scores
    /// SDPA used to produce its (per-segment normalized) output.
    pub fn attn_lse(queries: &Array, keys: &Array, scale: f32) -> Result<Array> {
        let qs = queries.shape();
        let ks = keys.shape();
        let (b, h, d) = (qs[0], qs[1], qs[3]);
        let (hkv, p) = (ks[1], ks[2]);
        // Expand KV heads to query-head count for GQA (no-op when h == hkv).
        let keys_exp = if h == hkv {
            keys.clone()
        } else {
            let g = h / hkv;
            let k5 = mlx_rs::ops::reshape(keys, &[b, hkv, 1, p, d])
                .context("attn_lse: reshape keys -> [B,Hkv,1,P,D]")?;
            let k5b = mlx_rs::ops::broadcast_to(&k5, &[b, hkv, g, p, d])
                .context("attn_lse: broadcast keys -> [B,Hkv,G,P,D]")?;
            mlx_rs::ops::reshape(&k5b, &[b, h, p, d])
                .context("attn_lse: reshape keys -> [B,H,P,D]")?
        };
        // scores = scale * Q @ Kᵀ  → [B, H, Lq, P]
        let kt = mlx_rs::ops::transpose_axes(&keys_exp, &[0, 1, 3, 2])
            .context("attn_lse: transpose keys -> [B,H,D,P]")?;
        let scores = mlx_rs::ops::matmul(queries, &kt).context("attn_lse: Q @ Kᵀ")?;
        let scale_arr = Array::from_f32(scale);
        let scores =
            mlx_rs::ops::multiply(&scores, &scale_arr).context("attn_lse: scores * scale")?;
        // logsumexp over the key axis (keepdims) → [B, H, Lq, 1]
        let m = scores
            .max_axis(-1, true)
            .context("attn_lse: max over keys")?;
        let shifted = mlx_rs::ops::subtract(&scores, &m).context("attn_lse: scores - max")?;
        let e = mlx_rs::ops::exp(&shifted).context("attn_lse: exp")?;
        let s = e.sum_axis(-1, true).context("attn_lse: sum exp")?;
        let lse = mlx_rs::ops::add(&m, &mlx_rs::ops::log(&s).context("attn_lse: log sum")?)
            .context("attn_lse: m + log(sum)")?;
        Ok(lse)
    }

    /// Flash-style split attention: attend a single query against two disjoint,
    /// independently-stored key/value segments (a SHARED prefix and a per-seq
    /// suffix) and merge the results — WITHOUT ever materializing the full
    /// `[prefix ++ suffix]` buffer. This is the mechanism behind single-copy
    /// shared-prefix KV: the prefix K/V is stored once and referenced by every
    /// sequence in the batch, while each sequence keeps only its own (small)
    /// divergent suffix.
    ///
    /// Math: with `o_p, o_s` the per-segment SDPA outputs (each softmax-
    /// normalized over its own segment) and `lse_p, lse_s` the segments'
    /// log-sum-exps, the full-attention output is the lse-weighted blend
    ///   `out = (o_p·e^{lse_p−m} + o_s·e^{lse_s−m}) / (e^{lse_p−m} + e^{lse_s−m})`,
    /// `m = max(lse_p, lse_s)`. This is algebraically identical to a single
    /// softmax over the concatenated keys; only floating-point reassociation
    /// differs (so it is NOT bit-identical to the concatenated SDPA, but
    /// matches to ~1e-4 — see `sdpa_split_matches_full`).
    ///
    /// Empty-segment fast paths: a zero-length prefix or suffix degrades to a
    /// plain SDPA over the non-empty segment (no merge, exact).
    pub fn sdpa_split(
        queries: &Array,
        k_prefix: &Array,
        v_prefix: &Array,
        k_suffix: &Array,
        v_suffix: &Array,
        scale: f32,
    ) -> Result<Array> {
        let p_len = k_prefix.shape()[2];
        let s_len = k_suffix.shape()[2];
        if p_len == 0 {
            return sdpa(queries, k_suffix, v_suffix, scale, false);
        }
        if s_len == 0 {
            return sdpa(queries, k_prefix, v_prefix, scale, false);
        }
        let o_p = sdpa(queries, k_prefix, v_prefix, scale, false)?;
        let o_s = sdpa(queries, k_suffix, v_suffix, scale, false)?;
        let lse_p = attn_lse(queries, k_prefix, scale)?;
        let lse_s = attn_lse(queries, k_suffix, scale)?;
        let m = mlx_rs::ops::maximum(&lse_p, &lse_s).context("sdpa_split: max(lse_p, lse_s)")?;
        let wp =
            mlx_rs::ops::exp(&mlx_rs::ops::subtract(&lse_p, &m).context("sdpa_split: lse_p-m")?)
                .context("sdpa_split: exp(lse_p-m)")?;
        let ws =
            mlx_rs::ops::exp(&mlx_rs::ops::subtract(&lse_s, &m).context("sdpa_split: lse_s-m")?)
                .context("sdpa_split: exp(lse_s-m)")?;
        let denom = mlx_rs::ops::add(&wp, &ws).context("sdpa_split: wp+ws")?;
        let num = mlx_rs::ops::add(
            &mlx_rs::ops::multiply(&o_p, &wp).context("sdpa_split: o_p*wp")?,
            &mlx_rs::ops::multiply(&o_s, &ws).context("sdpa_split: o_s*ws")?,
        )
        .context("sdpa_split: blend numerator")?;
        mlx_rs::ops::divide(&num, &denom).context("sdpa_split: normalize")
    }

    /// SDPA with an explicit additive mask (`0.0` for allowed positions,
    /// `-inf` for masked) — used for sliding-window attention where the
    /// vanilla causal sentinel doesn't express the window cutoff.
    pub fn sdpa_with_mask(
        queries: &Array,
        keys: &Array,
        values: &Array,
        scale: f32,
        mask: &Array,
    ) -> Result<Array> {
        mlx_rs::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale,
            Some(ScaledDotProductAttentionMask::Array(mask)),
            None, /* sinks */
        )
        .context("mlx-rs fast::scaled_dot_product_attention (explicit mask) FFI call failed")
    }

    /// Build an additive attention mask of shape `[query_len, kv_offset + query_len]`
    /// in `Float32`. Allowed positions are `0.0`; masked positions are `-inf`.
    ///
    /// * `window_size = None` → standard causal mask (lower triangle).
    /// * `window_size = Some(w)` → sliding causal: query at position
    ///   `kv_offset + i` attends to keys in `[max(0, kv_offset + i - w + 1), kv_offset + i]`.
    ///
    /// Mirrors mlx_lm's `models.base.create_causal_mask` exactly.
    /// Returns `None` when `query_len == 0` (no rows → no mask needed).
    pub fn build_causal_mask(
        query_len: usize,
        kv_offset: usize,
        window_size: Option<usize>,
    ) -> Result<Option<Array>> {
        // Compat shim: `kv_offset` was the legacy single-pass parameter where
        // every key index `j` mapped 1:1 to absolute position `j`. We now
        // route through the absolute-position-aware builder with the cache
        // assumed to start at position 0 (true for non-rotated caches and
        // for the original single-pass full-prompt forward).
        build_causal_mask_abs(
            /* query_start_abs_pos */ kv_offset,
            query_len,
            /* cache_first_held_pos */ 0,
            /* kv_actual */ kv_offset + query_len,
            window_size,
        )
    }

    /// The pre-2026-05-15 mask: a bf16 `[query_len, kv_actual]` array carrying
    /// `0.0` where a key is attendable and `-inf` where it is not.
    ///
    /// Split out of [`build_causal_mask_abs`] so both representations are
    /// directly callable. They used to be selectable only through
    /// `LUMEN_LEGACY_MASK_BUILDER`, which meant a test covering both had to
    /// mutate process-global state and could not run in parallel with anything
    /// else — so in practice neither path was covered, and the tests that
    /// existed asserted an f32 dtype that neither path had produced for months.
    pub fn build_causal_mask_legacy_bf16(
        query_start_abs_pos: usize,
        query_len: usize,
        cache_first_held_pos: usize,
        kv_actual: usize,
        window_size: Option<usize>,
    ) -> Result<Option<Array>> {
        if query_len == 0 {
            return Ok(None);
        }
        let total_keys = kv_actual;
        let total_cells = query_len * total_keys;
        let mut data = vec![f32::NEG_INFINITY; total_cells];
        let cache_end = cache_first_held_pos + total_keys;
        for i in 0..query_len {
            let qpos = query_start_abs_pos + i;
            let causal_max = qpos;
            let window_min = match window_size {
                Some(w) => qpos.saturating_sub(w.saturating_sub(1)),
                None => 0,
            };
            let valid_min_abs = std::cmp::max(window_min, cache_first_held_pos);
            let valid_max_abs_excl = std::cmp::min(causal_max + 1, cache_end);
            if valid_min_abs >= valid_max_abs_excl {
                continue;
            }
            let row_start = i * total_keys;
            let lo = row_start + (valid_min_abs - cache_first_held_pos);
            let hi = row_start + (valid_max_abs_excl - cache_first_held_pos);
            data[lo..hi].fill(0.0);
        }
        // One construction, not two. This was written twice with the first
        // binding shadowed, so every call built and dropped a whole
        // `[query_len, kv_actual]` bf16 array before building the one it
        // returned — 33 MB of pure waste at a 4K context, on the builder whose
        // own callers already complain about the size of this allocation.
        let arr = Array::from_slice(&data, &[query_len as i32, total_keys as i32])
            .as_dtype(mlx_rs::Dtype::Bfloat16)
            .context("build_causal_mask: legacy cast mask to bf16 failed")?;
        Ok(Some(arr))
    }

    /// Sliding-window-aware causal mask builder.
    ///
    /// Computes a `[query_len, kv_actual]` mask in absolute-position space:
    /// query index `i` lives at absolute position `query_start_abs_pos + i`,
    /// and key index `k` (in the K tensor returned by `update_and_fetch`) lives
    /// at absolute position `cache_first_held_pos + k`. This generalisation is
    /// what lets chunked prefill against `NativeRotatingKvCache` work — after
    /// a rotation the cache no longer covers `[0, kv_offset + query_len)`, so
    /// the old `(query_len, kv_offset + query_len)` mask shape can't broadcast
    /// against the rotated K (anti-pattern: position-relative mask vs
    /// rotated KV layout).
    pub fn build_causal_mask_abs(
        query_start_abs_pos: usize,
        query_len: usize,
        cache_first_held_pos: usize,
        kv_actual: usize,
        window_size: Option<usize>,
    ) -> Result<Option<Array>> {
        if query_len == 0 {
            return Ok(None);
        }
        // A/B gate (2026-05-15 prefill perf investigation): set
        // LUMEN_LEGACY_MASK_BUILDER=1 to revert to the legacy CPU bf16 path
        // (Vec<f32> + memset + Array::from_slice + as_dtype). Default OFF =
        // new GPU bool builder. Keep both paths until perf delta is confirmed
        // across all (ctx, model) shapes.
        if std::env::var("LUMEN_LEGACY_MASK_BUILDER")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            return build_causal_mask_legacy_bf16(
                query_start_abs_pos,
                query_len,
                cache_first_held_pos,
                kv_actual,
                window_size,
            );
        } // mlx_lm-parity GPU bool mask builder (2026-05-15).
        // Matches mlx_lm/models/base.py::create_causal_mask: builds
        //   linds[:, None] >= rinds[None]                   (causal)
        //   & (linds < rinds + window_size)                  (sliding cutoff)
        // entirely via mlx::ops::{arange, ge, lt, logical_and}, producing a
        // [L_q, L_kv] bool Array on the GPU. Two upsides vs the legacy CPU
        // path (Vec<f32> + memset + Array::from_slice + as_dtype bf16):
        //   1. No CPU allocation / GPU upload (mlx_lm parity at 8K =
        //      256 MB Vec<f32> + 128 MB bf16 upload skipped).
        //   2. bool dtype (1 byte/cell) halves the mask read bandwidth on
        //      the SDPA fallback path vs the prior bf16 (2 byte/cell).
        //      mlx::fast::sdpa supports bool masks natively (see
        //      ScaledDotProductAttention::supports_bool_mask = true),
        //      taking the `where(mask, scores, -inf)` branch instead of
        //      the `add(scores, bf16_mask)` branch. mlx_lm uses the same.
        let stream = mlx_rs::Stream::gpu();
        let linds = Array::arange_device::<_, i32>(
            Some(query_start_abs_pos as i32),
            (query_start_abs_pos + query_len) as i32,
            None,
            &stream,
        )
        .context("build_causal_mask_abs: arange linds failed")?;
        let rinds = Array::arange_device::<_, i32>(
            Some(cache_first_held_pos as i32),
            (cache_first_held_pos + kv_actual) as i32,
            None,
            &stream,
        )
        .context("build_causal_mask_abs: arange rinds failed")?;
        let linds = linds
            .expand_dims_device(1, &stream)
            .context("build_causal_mask_abs: expand linds -> [L_q,1] failed")?;
        let rinds = rinds
            .expand_dims_device(0, &stream)
            .context("build_causal_mask_abs: expand rinds -> [1,L_kv] failed")?;
        let mut mask = linds
            .ge_device(&rinds, &stream)
            .context("build_causal_mask_abs: linds >= rinds failed")?;
        if let Some(w) = window_size {
            let w_arr = Array::from_int(w as i32);
            let rinds_plus_w = mlx_rs::ops::add_device(&rinds, &w_arr, &stream)
                .context("build_causal_mask_abs: rinds + window failed")?;
            let window_mask = linds
                .lt_device(&rinds_plus_w, &stream)
                .context("build_causal_mask_abs: linds < rinds+w failed")?;
            mask = mask
                .logical_and_device(&window_mask, &stream)
                .context("build_causal_mask_abs: causal & window failed")?;
        }
        Ok(Some(mask))
    }
}

#[cfg(feature = "mlx-native")]
pub use imp::lumen_sdpa_timing;
#[cfg(feature = "mlx-native")]
#[allow(unused_imports)]
// Consumed by Phase 3b model assembly in runner_native.rs and Gemma 4 sliding attention.
pub(crate) use imp::{
    attn_lse, build_causal_mask, build_causal_mask_abs, sdpa, sdpa_split, sdpa_with_mask,
};

// SDPA bit-identical vs MLX reference.
//
// Fixtures produced by `scripts/generate_sdpa_fixture.py` (magic `TQSA`).
//
// `#[ignore]`'d — MLX FFI requires non-sandbox host with Metal device.
#[cfg(all(test, feature = "mlx-native"))]
mod parity_tests {
    use super::imp::sdpa;
    use mlx_rs::Array;
    use std::path::{Path, PathBuf};

    const MAGIC: &[u8; 4] = b"TQSA";
    const HEADER_LEN: usize = 4 /* magic */ + 8 * 4 /* 8 u32 */;

    fn fixture_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("lumen-mlx manifest dir has a parent (crates/)")
            .join("lumen-metal")
            .join("tests")
            .join("fixtures")
    }

    fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }
    fn slice_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    struct SdpaFixture {
        batch: usize,
        n_heads: usize,
        n_kv_heads: usize,
        q_seq: usize,
        kv_seq: usize,
        head_dim: usize,
        causal: bool,
        scale: f32,
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        expected_y: Vec<f32>,
    }

    fn load_fixture(name: &str) -> Option<SdpaFixture> {
        let path = fixture_dir().join(format!("embed_sdpa_{name}.bin"));
        let bytes = std::fs::read(&path).ok()?;
        if bytes.len() < HEADER_LEN || &bytes[..4] != MAGIC {
            panic!("sdpa fixture {path:?} has bad header");
        }
        let batch = read_u32_le(&bytes, 4) as usize;
        let n_heads = read_u32_le(&bytes, 8) as usize;
        let n_kv_heads = read_u32_le(&bytes, 12) as usize;
        let q_seq = read_u32_le(&bytes, 16) as usize;
        let kv_seq = read_u32_le(&bytes, 20) as usize;
        let head_dim = read_u32_le(&bytes, 24) as usize;
        let causal = read_u32_le(&bytes, 28) != 0;
        let scale = f32::from_bits(read_u32_le(&bytes, 32));

        let q_count = batch * n_heads * q_seq * head_dim;
        let kv_count = batch * n_kv_heads * kv_seq * head_dim;
        let mut cur = HEADER_LEN;
        let q = slice_f32(&bytes[cur..cur + q_count * 4]);
        cur += q_count * 4;
        let k = slice_f32(&bytes[cur..cur + kv_count * 4]);
        cur += kv_count * 4;
        let v = slice_f32(&bytes[cur..cur + kv_count * 4]);
        cur += kv_count * 4;
        let expected_y = slice_f32(&bytes[cur..cur + q_count * 4]);

        Some(SdpaFixture {
            batch,
            n_heads,
            n_kv_heads,
            q_seq,
            kv_seq,
            head_dim,
            causal,
            scale,
            q,
            k,
            v,
            expected_y,
        })
    }

    fn run_fixture(name: &str) {
        let Some(fx) = load_fixture(name) else {
            eprintln!(
                "SKIP sdpa/{name}: fixture missing under {}. Run `python scripts/generate_sdpa_fixture.py` first.",
                fixture_dir().display()
            );
            return;
        };

        let q = Array::from_slice(
            &fx.q,
            &[
                fx.batch as i32,
                fx.n_heads as i32,
                fx.q_seq as i32,
                fx.head_dim as i32,
            ],
        );
        let k = Array::from_slice(
            &fx.k,
            &[
                fx.batch as i32,
                fx.n_kv_heads as i32,
                fx.kv_seq as i32,
                fx.head_dim as i32,
            ],
        );
        let v = Array::from_slice(
            &fx.v,
            &[
                fx.batch as i32,
                fx.n_kv_heads as i32,
                fx.kv_seq as i32,
                fx.head_dim as i32,
            ],
        );

        let out = sdpa(&q, &k, &v, fx.scale, fx.causal).expect("mlx-rs SDPA FFI must succeed");
        assert_eq!(
            out.shape(),
            &[
                fx.batch as i32,
                fx.n_heads as i32,
                fx.q_seq as i32,
                fx.head_dim as i32,
            ],
            "{name}: sdpa output shape mismatch"
        );
        assert_eq!(
            out.dtype(),
            mlx_rs::Dtype::Float32,
            "{name}: sdpa output dtype unexpectedly changed"
        );
        out.eval().expect("mlx eval must succeed");

        let observed: &[f32] = out.as_slice();
        assert_eq!(
            observed.len(),
            fx.expected_y.len(),
            "{name}: sdpa output flat-size mismatch"
        );

        let mut mismatches = 0usize;
        for (i, (&got, &expected)) in observed.iter().zip(fx.expected_y.iter()).enumerate() {
            if got.to_bits() != expected.to_bits() {
                if mismatches < 5 {
                    eprintln!(
                        "sdpa/{name}[{i}]: 0x{:08x} ({got}) vs ref 0x{:08x} ({expected})",
                        got.to_bits(),
                        expected.to_bits()
                    );
                }
                mismatches += 1;
            }
        }
        assert_eq!(
            mismatches,
            0,
            "{name}: {mismatches}/{} elements diverged from MLX reference",
            observed.len()
        );
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn prefill_causal_sdpa_matches_mlx() {
        run_fixture("prefill_causal");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn decode_q1_sdpa_matches_mlx() {
        run_fixture("decode_q1");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn gqa_prefill_sdpa_matches_mlx() {
        run_fixture("gqa_prefill");
    }

    // ───────────────────── build_causal_mask (Gemma 4 sliding) ─────────────────────

    use super::imp::{build_causal_mask, build_causal_mask_legacy_bf16};

    fn assert_mask_layout(mask: &Array, expected_rows: i32, expected_cols: i32) {
        assert_eq!(mask.shape(), &[expected_rows, expected_cols]);
    }

    /// Read a mask as "may query i attend to key j".
    ///
    /// The two builders disagree on representation — the default GPU path
    /// returns a bool array, the legacy path a bf16 array of `0.0` / `-inf` —
    /// so the assertions below are written against the meaning rather than the
    /// encoding. These tests used to assert `Dtype::Float32`, which neither
    /// path had produced since the bf16 change; being `#[ignore]`d, they failed
    /// silently for months.
    fn mask_to_attendable(mask: &Array) -> Vec<bool> {
        mask.eval().expect("mlx eval must succeed");
        match mask.dtype() {
            mlx_rs::Dtype::Bool => mask.as_slice::<bool>().to_vec(),
            _ => mask
                .as_dtype(mlx_rs::Dtype::Float32)
                .expect("cast mask to f32")
                .as_slice::<f32>()
                .iter()
                .map(|v| v.is_finite())
                .collect(),
        }
    }

    /// Both builders must describe the same attendable set. This is the
    /// property that actually matters: `LUMEN_LEGACY_MASK_BUILDER=1` is a live
    /// escape hatch, so a divergence between the paths would silently change
    /// what the model attends to.
    fn assert_both_builders_agree(
        query_len: usize,
        offset: usize,
        window: Option<usize>,
        expect_attendable: &[bool],
    ) {
        let default = build_causal_mask(query_len, offset, window)
            .expect("default builder")
            .expect("non-empty mask");
        let legacy =
            build_causal_mask_legacy_bf16(offset, query_len, 0, offset + query_len, window)
                .expect("legacy builder")
                .expect("non-empty mask");
        assert_eq!(
            mask_to_attendable(&default),
            expect_attendable,
            "default (GPU bool) builder"
        );
        assert_eq!(
            mask_to_attendable(&legacy),
            expect_attendable,
            "legacy (bf16 -inf) builder"
        );
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn causal_mask_prefill_full_no_window() {
        // Query i attends to every key j <= i.
        let t = true;
        let f = false;
        assert_both_builders_agree(
            3,
            0,
            None,
            &[
                t, f, f, //
                t, t, f, //
                t, t, t, //
            ],
        );
        let mask = build_causal_mask(3, 0, None).unwrap().unwrap();
        assert_mask_layout(&mask, 3, 3);
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn causal_mask_prefill_window_truncates_past_window() {
        // Window 2: query i attends to j in [i-1, i], clamped at 0.
        let t = true;
        let f = false;
        assert_both_builders_agree(
            4,
            0,
            Some(2),
            &[
                t, f, f, f, //
                t, t, f, f, //
                f, t, t, f, //
                f, f, t, t, //
            ],
        );
        let mask = build_causal_mask(4, 0, Some(2)).unwrap().unwrap();
        assert_mask_layout(&mask, 4, 4);
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn causal_mask_decode_with_offset_and_window() {
        // Decode: one query at absolute position 3, window 2 → keys 2..=3.
        let t = true;
        let f = false;
        assert_both_builders_agree(1, 3, Some(2), &[f, f, t, t]);
        let mask = build_causal_mask(1, 3, Some(2)).unwrap().unwrap();
        assert_mask_layout(&mask, 1, 4);
    }

    #[test]
    fn causal_mask_empty_query_returns_none() {
        assert!(
            build_causal_mask(0, 5, None)
                .expect("build_causal_mask")
                .is_none()
        );
    }

    // ───────────────────── split attention (shared-prefix KV dedup) ─────────────────────
    use super::imp::sdpa_split;
    use mlx_rs::ops::concatenate_axis;

    /// Deterministic pseudo-random fill in [-1, 1) — avoids Math.random so the
    /// test is reproducible without a fixture file.
    fn det_fill(n: usize, seed: u32) -> Vec<f32> {
        let mut x = seed.wrapping_add(0x9E3779B9);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                ((x >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// `sdpa_split(prefix, suffix)` must equal a plain SDPA over the
    /// concatenated `[prefix ++ suffix]` keys/values to ~1e-4 (FP reassociation
    /// only). Exercises GQA (4 query heads, 2 KV heads) and the q_len=1 decode
    /// shape the batched scheduler uses.
    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn sdpa_split_matches_full() {
        let (b, h, hkv, d) = (1i32, 4i32, 2i32, 8i32);
        let (p_len, s_len) = (5i32, 3i32);
        let scale = 1.0f32 / (d as f32).sqrt();

        let q = Array::from_slice(&det_fill((b * h * d) as usize, 1), &[b, h, 1, d]);
        let kp = Array::from_slice(
            &det_fill((b * hkv * p_len * d) as usize, 2),
            &[b, hkv, p_len, d],
        );
        let vp = Array::from_slice(
            &det_fill((b * hkv * p_len * d) as usize, 3),
            &[b, hkv, p_len, d],
        );
        let ks = Array::from_slice(
            &det_fill((b * hkv * s_len * d) as usize, 4),
            &[b, hkv, s_len, d],
        );
        let vs = Array::from_slice(
            &det_fill((b * hkv * s_len * d) as usize, 5),
            &[b, hkv, s_len, d],
        );

        let k_full = concatenate_axis(&[&kp, &ks], 2).expect("concat k");
        let v_full = concatenate_axis(&[&vp, &vs], 2).expect("concat v");
        let full = sdpa(&q, &k_full, &v_full, scale, false).expect("full sdpa");
        let split = sdpa_split(&q, &kp, &vp, &ks, &vs, scale).expect("split sdpa");

        full.eval().expect("eval full");
        split.eval().expect("eval split");
        assert_eq!(full.shape(), split.shape(), "split/full shape mismatch");

        let fa: &[f32] = full.as_slice();
        let sa: &[f32] = split.as_slice();
        let mut max_abs = 0.0f32;
        for (&a, &c) in fa.iter().zip(sa.iter()) {
            max_abs = max_abs.max((a - c).abs());
        }
        assert!(
            max_abs < 1e-4,
            "sdpa_split diverged from concatenated SDPA: max_abs={max_abs}"
        );
    }

    /// Empty-prefix and empty-suffix fast paths degrade to a plain SDPA over the
    /// non-empty segment (exact, not merged).
    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn sdpa_split_empty_segment_paths() {
        let (b, h, hkv, d) = (1i32, 4i32, 2i32, 8i32);
        let scale = 1.0f32 / (d as f32).sqrt();
        let q = Array::from_slice(&det_fill((b * h * d) as usize, 7), &[b, h, 1, d]);
        let ks = Array::from_slice(&det_fill((b * hkv * 3 * d) as usize, 8), &[b, hkv, 3, d]);
        let vs = Array::from_slice(&det_fill((b * hkv * 3 * d) as usize, 9), &[b, hkv, 3, d]);
        let empty_k = Array::from_slice::<f32>(&[], &[b, hkv, 0, d]);
        let empty_v = Array::from_slice::<f32>(&[], &[b, hkv, 0, d]);

        // Empty prefix → suffix-only SDPA (bit-identical).
        let only_suffix =
            sdpa_split(&q, &empty_k, &empty_v, &ks, &vs, scale).expect("empty prefix");
        let ref_suffix = sdpa(&q, &ks, &vs, scale, false).expect("ref suffix");
        only_suffix.eval().unwrap();
        ref_suffix.eval().unwrap();
        assert_eq!(only_suffix.as_slice::<f32>(), ref_suffix.as_slice::<f32>());

        // Empty suffix → prefix-only SDPA (bit-identical).
        let only_prefix =
            sdpa_split(&q, &ks, &vs, &empty_k, &empty_v, scale).expect("empty suffix");
        only_prefix.eval().unwrap();
        assert_eq!(only_prefix.as_slice::<f32>(), ref_suffix.as_slice::<f32>());
    }
}
