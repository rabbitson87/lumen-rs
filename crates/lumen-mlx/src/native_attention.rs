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
        let final_result =
            raw.context("mlx-rs fast::scaled_dot_product_attention FFI call failed");
        let t_end = Instant::now();
        lumen_sdpa_timing::PRE_NS.fetch_add(
            (t_after_pre - t_start).as_nanos() as u64,
            Ordering::Relaxed,
        );
        lumen_sdpa_timing::CALL_NS.fetch_add(
            (t_after_call - t_after_pre).as_nanos() as u64,
            Ordering::Relaxed,
        );
        lumen_sdpa_timing::POST_NS.fetch_add(
            (t_end - t_after_call).as_nanos() as u64,
            Ordering::Relaxed,
        );
        lumen_sdpa_timing::TOTAL_NS.fetch_add(
            (t_end - t_start).as_nanos() as u64,
            Ordering::Relaxed,
        );
        lumen_sdpa_timing::CALLS.fetch_add(1, Ordering::Relaxed);
        final_result
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
        if std::env::var("LUMEN_LEGACY_MASK_BUILDER").map(|v| v == "1").unwrap_or(false) {
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
            let arr = Array::from_slice(&data, &[query_len as i32, total_keys as i32])
                .as_dtype(mlx_rs::Dtype::Bfloat16)
                .context("build_causal_mask: legacy cast mask to bf16 failed")?;
            return Ok(Some(arr));
        }
        // mlx_lm-parity GPU bool mask builder (2026-05-15).
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
        let linds = linds.expand_dims_device(1, &stream)
            .context("build_causal_mask_abs: expand linds -> [L_q,1] failed")?;
        let rinds = rinds.expand_dims_device(0, &stream)
            .context("build_causal_mask_abs: expand rinds -> [1,L_kv] failed")?;
        let mut mask = linds.ge_device(&rinds, &stream)
            .context("build_causal_mask_abs: linds >= rinds failed")?;
        if let Some(w) = window_size {
            let w_arr = Array::from_int(w as i32);
            let rinds_plus_w = mlx_rs::ops::add_device(&rinds, &w_arr, &stream)
                .context("build_causal_mask_abs: rinds + window failed")?;
            let window_mask = linds.lt_device(&rinds_plus_w, &stream)
                .context("build_causal_mask_abs: linds < rinds+w failed")?;
            mask = mask.logical_and_device(&window_mask, &stream)
                .context("build_causal_mask_abs: causal & window failed")?;
        }
        Ok(Some(mask))
    }
}

#[cfg(feature = "mlx-native")]
#[allow(unused_imports)] // Consumed by Phase 3b model assembly in runner_native.rs and Gemma 4 sliding attention.
pub(crate) use imp::{build_causal_mask, build_causal_mask_abs, sdpa, sdpa_with_mask};
#[cfg(feature = "mlx-native")]
pub use imp::lumen_sdpa_timing;

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

    use super::imp::build_causal_mask;

    fn assert_mask_layout(mask: &Array, expected_rows: i32, expected_cols: i32) {
        assert_eq!(mask.shape(), &[expected_rows, expected_cols]);
        assert_eq!(mask.dtype(), mlx_rs::Dtype::Float32);
    }

    /// Round-trip a mask Array back to a Vec<f32> for value assertions.
    fn mask_to_vec(mask: &Array) -> Vec<f32> {
        mask.eval().expect("mlx eval must succeed");
        mask.as_slice::<f32>().to_vec()
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn causal_mask_prefill_full_no_window() {
        let mask = build_causal_mask(3, 0, None)
            .expect("build_causal_mask")
            .expect("non-empty mask");
        assert_mask_layout(&mask, 3, 3);
        let v = mask_to_vec(&mask);
        // Row i (query i) attends to j <= i:
        // [0,0,0]  → [0,  -inf, -inf]
        // [0,0,0]  → [0,    0, -inf]
        // [0,0,0]  → [0,    0,   0]
        let inf = f32::NEG_INFINITY;
        let expected = [
            0.0, inf, inf, //
            0.0, 0.0, inf, //
            0.0, 0.0, 0.0, //
        ];
        for (i, (&got, &exp)) in v.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got.is_finite(),
                exp.is_finite(),
                "row {i}: finite mismatch (got {got}, exp {exp})"
            );
            if exp.is_finite() {
                assert_eq!(got, exp, "row {i}: value mismatch");
            }
        }
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn causal_mask_prefill_window_truncates_past_window() {
        // Window size 2, query_len 4, offset 0:
        // qpos 0: attend to j in [0,0]                 → [0, -inf, -inf, -inf]
        // qpos 1: attend to j in [0,1]                 → [0,  0,   -inf, -inf]
        // qpos 2: attend to j in [1,2]                 → [-inf, 0,  0,   -inf]
        // qpos 3: attend to j in [2,3]                 → [-inf, -inf, 0,  0]
        let mask = build_causal_mask(4, 0, Some(2))
            .expect("build_causal_mask")
            .expect("non-empty mask");
        assert_mask_layout(&mask, 4, 4);
        let v = mask_to_vec(&mask);
        let inf = f32::NEG_INFINITY;
        let expected = [
            0.0, inf, inf, inf, //
            0.0, 0.0, inf, inf, //
            inf, 0.0, 0.0, inf, //
            inf, inf, 0.0, 0.0, //
        ];
        for (i, (&got, &exp)) in v.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got.is_finite(), exp.is_finite(), "elem {i}");
            if exp.is_finite() {
                assert_eq!(got, exp, "elem {i}");
            }
        }
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn causal_mask_decode_with_offset_and_window() {
        // query_len=1, offset=3, window=2 → qpos=3, key range [2, 3]
        // mask shape [1, 4]: [-inf, -inf, 0, 0]
        let mask = build_causal_mask(1, 3, Some(2))
            .expect("build_causal_mask")
            .expect("non-empty mask");
        assert_mask_layout(&mask, 1, 4);
        let v = mask_to_vec(&mask);
        let inf = f32::NEG_INFINITY;
        let expected = [inf, inf, 0.0, 0.0];
        for (i, (&got, &exp)) in v.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got.is_finite(), exp.is_finite(), "elem {i}");
            if exp.is_finite() {
                assert_eq!(got, exp, "elem {i}");
            }
        }
    }

    #[test]
    fn causal_mask_empty_query_returns_none() {
        assert!(
            build_causal_mask(0, 5, None)
                .expect("build_causal_mask")
                .is_none()
        );
    }
}
