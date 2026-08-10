//! Native MLX prompt-cache lifecycle.
//!
//! Mirrors `mlx_lm.models.cache` for the Qwen3.5-MoE hybrid architecture:
//!
//! * **`NativeKvCache`** — full-attention layers. Append-and-fetch semantics over
//!   `(keys, values)` with shape `[B, n_kv_heads, T, head_dim]`. Block-allocates
//!   the underlying buffer in `KV_CACHE_STEP` (256) token blocks and fills via
//!   `slice_update` — mirrors `mlx_lm.cache.KVCache(step=256)`. Concat is
//!   triggered only when crossing a 256-token boundary (i.e. ~1 concat per
//!   256 decode steps) instead of every step, eliminating the ~200× per-step
//!   `concatenate_gpu + MetalAllocator::malloc` overhead seen in the prior
//!   `ConcatenateKVCache` pattern. See
//!   `notes/layer_b_b1_b2_recon_2026_05_05.md`.
//!
//! * **`NativeArraysCache`** — linear-attention (delta-net SSM) layers. Holds a
//!   fixed-size `Vec<Option<Array>>` of state slots (size 2 for Qwen3.5-MoE:
//!   `[conv_state, ssm_state]`). Tracks an `offset` advanced by the consumer
//!   plus optional `lengths`/`left_padding` per-batch metadata for variable-
//!   length prompts.
//!
//! * **`NativeLayerCache`** — enum wrapper used per layer.
//! * **`NativePromptCache`** — `Vec<NativeLayerCache>` for the whole model.
//!
//! See `.ai/memory/active/mlx-rs-native-port/CONTEXT.md` Session 24+ for
//! design rationale.

#[cfg(feature = "mlx-native")]
#[allow(dead_code)] // cache lifecycle accessors (offset/as_full/as_linear/clear/keys/etc.) reserved for future inspection / debugging consumers and lifecycle tests
mod imp {
    use anyhow::{Context, Result, anyhow};
    use mlx_rs::Array;
    use mlx_rs::ops::indexing::{TryIndexMutOp, TryIndexOp};
    use std::sync::OnceLock;

    /// Block size for KV-cache pre-allocation. Mirrors `mlx_lm.cache.KVCache.step`.
    /// Decode appends in steps of 1 token; the underlying buffer grows by this
    /// many tokens at a time so concat / fresh alloc is amortized 1:KV_CACHE_STEP.
    pub const KV_CACHE_STEP: usize = 256;

    /// Default ON since 2026-05-07 (Phase G2 LANDED): microbench long-prompt
    /// 35B-mxfp4 n=12 interleaved A/B (PROMPT_LEN=2048, STEPS=1500) showed
    /// Δ tps +4.67 tok/s (Welch t = +4.33σ), Δ p50 −1.02 ms/step (t = −4.68σ).
    /// 32-step PyO3 golden bit-identical PASS for both paths. The prior
    /// 2026-05-06 35B server-level WASH (-0.06σ) reflected dilution at the
    /// server step (107ms/step → KV concat 0.7ms = 0.65%); microbench
    /// 17ms/step exposes the same lever at 4% magnitude.
    /// Set `LUMEN_NATIVE_KV_STEP_PREALLOC=0` to force legacy per-step
    /// `concatenate_axis` for emergency revert / A/B comparison.
    fn use_step_prealloc() -> bool {
        #[cfg(test)]
        if let Some(forced) = test_support::forced_step_prealloc() {
            return forced;
        }
        static FLAG: OnceLock<bool> = OnceLock::new();
        *FLAG.get_or_init(|| {
            std::env::var("LUMEN_NATIVE_KV_STEP_PREALLOC")
                .map(|v| v != "0")
                .unwrap_or(true)
        })
    }

    /// Lets tests exercise both sides of the step-prealloc switch.
    ///
    /// The production flag is a `OnceLock` over an env var, read once per
    /// process — which means the legacy `concatenate_axis` path, kept as a live
    /// emergency revert, could not be reached from any test at all. Setting the
    /// env from a test would not work (the `OnceLock` may already be
    /// initialised) and would race every other test in the binary.
    ///
    /// The override is thread-local, so `cargo test`'s per-test threads cannot
    /// interfere with each other, and `#[cfg(test)]` keeps it out of release
    /// builds entirely.
    #[cfg(test)]
    pub(crate) mod test_support {
        use std::cell::Cell;

        thread_local! {
            static FORCED: Cell<Option<bool>> = const { Cell::new(None) };
        }

        pub(crate) fn forced_step_prealloc() -> Option<bool> {
            FORCED.with(Cell::get)
        }

        /// Run `f` with the step-prealloc switch pinned, restoring the previous
        /// setting afterwards even if `f` panics.
        pub(crate) fn with_step_prealloc<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
            struct Reset(Option<bool>);
            impl Drop for Reset {
                fn drop(&mut self) {
                    FORCED.with(|c| c.set(self.0));
                }
            }
            let _reset = Reset(FORCED.with(Cell::get));
            FORCED.with(|c| c.set(Some(enabled)));
            f()
        }
    }

    /// Slice helper for `arr[..., start:end, :]` on 4D arrays.
    fn slice_axis2(arr: &Array, start: i32, end: i32) -> Result<Array> {
        arr.try_index((.., .., start..end, ..))
            .context("NativeKvCache: slice axis=2")
    }

    /// Block-allocated full-attention KV cache (mlx_lm.KVCache step=256 semantics).
    #[derive(Clone)]
    pub struct NativeKvCache {
        /// Underlying buffer with shape `[B, n_kv_heads, n_blocks * KV_CACHE_STEP, D]`.
        /// `keys.shape()[2] >= offset` always; padded space past offset is zero-filled.
        keys: Option<Array>,
        values: Option<Array>,
        /// Logical sequence length stored in THIS buffer (the suffix length
        /// when `base_offset > 0`). Internal buffer ops (`update_and_fetch`,
        /// `keys_view`) index against this, NOT the absolute position.
        offset: usize,
        /// Absolute position of the first token THIS buffer holds. Zero for a
        /// normal cache. Non-zero only for a suffix-only cache under
        /// shared-prefix KV dedup (Phase 4): the first `base_offset` tokens
        /// live in a single shared prefix buffer, not here. RoPE / causal
        /// positioning use `offset()` = `base_offset + offset`, while the
        /// physical buffer still starts at index 0.
        base_offset: usize,
    }

    impl NativeKvCache {
        pub fn new() -> Self {
            Self {
                keys: None,
                values: None,
                offset: 0,
                base_offset: 0,
            }
        }

        /// Absolute logical position = shared-prefix length + suffix length.
        /// Equals the internal buffer length when `base_offset == 0` (the
        /// normal case), so every existing caller is byte-identical.
        pub fn offset(&self) -> usize {
            self.base_offset + self.offset
        }

        /// Number of tokens physically held in THIS buffer (suffix length under
        /// dedup; same as `offset()` otherwise).
        pub fn buffer_len(&self) -> usize {
            self.offset
        }

        /// Absolute position of the first token this buffer holds.
        pub fn base_offset(&self) -> usize {
            self.base_offset
        }

        /// Phase 4 shared-prefix dedup: drop the first `prefix_len` tokens from
        /// this buffer (they now live in a single shared prefix buffer) and
        /// record `base_offset = prefix_len` so absolute positioning is
        /// preserved. After this call the buffer holds only the divergent
        /// suffix `[prefix_len .. offset())`. No-op when `prefix_len == 0`.
        pub fn keep_suffix_from(&mut self, prefix_len: usize) -> Result<()> {
            if prefix_len == 0 {
                return Ok(());
            }
            let cur = self.base_offset + self.offset;
            if prefix_len > cur {
                return Err(anyhow!(
                    "NativeKvCache::keep_suffix_from: prefix_len={} > offset={}",
                    prefix_len,
                    cur
                ));
            }
            // Already suffix-only at this boundary (idempotent re-attach).
            if prefix_len <= self.base_offset {
                return Ok(());
            }
            let drop = prefix_len - self.base_offset; // tokens to cut from THIS buffer
            let new_len = self.offset - drop;
            if let (Some(k), Some(v)) = (self.keys.as_ref(), self.values.as_ref()) {
                let k_suffix = slice_axis2(k, drop as i32, self.offset as i32)
                    .context("keep_suffix_from: slice keys suffix")?;
                let v_suffix = slice_axis2(v, drop as i32, self.offset as i32)
                    .context("keep_suffix_from: slice values suffix")?;
                self.keys = Some(k_suffix);
                self.values = Some(v_suffix);
            }
            self.offset = new_len;
            self.base_offset = prefix_len;
            Ok(())
        }

        /// Frozen `[.., :prefix_len, :]` view of `(keys, values)` for building a
        /// [`SharedPrefixKv`] from an already-prefilled donor cache. Requires
        /// `base_offset == 0` (the donor holds the full prompt from position 0)
        /// and `prefix_len <= offset`. The returned Arrays are slices over the
        /// donor's buffer (refcount bump, not a deep copy) — the donor must
        /// outlive the shared prefix, or the slices must be eval'd/cloned first.
        pub fn prefix_kv(&self, prefix_len: usize) -> Result<(Array, Array)> {
            if self.base_offset != 0 {
                return Err(anyhow!(
                    "NativeKvCache::prefix_kv: donor base_offset={} must be 0",
                    self.base_offset
                ));
            }
            if prefix_len > self.offset {
                return Err(anyhow!(
                    "NativeKvCache::prefix_kv: prefix_len={} > offset={}",
                    prefix_len,
                    self.offset
                ));
            }
            let k = self
                .keys
                .as_ref()
                .ok_or_else(|| anyhow!("NativeKvCache::prefix_kv: empty keys"))?;
            let v = self
                .values
                .as_ref()
                .ok_or_else(|| anyhow!("NativeKvCache::prefix_kv: empty values"))?;
            let kp = slice_axis2(k, 0, prefix_len as i32).context("prefix_kv: slice keys")?;
            let vp = slice_axis2(v, 0, prefix_len as i32).context("prefix_kv: slice values")?;
            Ok((kp, vp))
        }

        pub fn empty(&self) -> bool {
            self.keys.is_none()
        }

        /// Buffer reference. **May include zero-padded space past `offset`**
        /// — for a logical-length view use the return of `update_and_fetch`
        /// or `keys_view()`.
        pub fn keys(&self) -> Option<&Array> {
            self.keys.as_ref()
        }

        pub fn values(&self) -> Option<&Array> {
            self.values.as_ref()
        }

        /// Logical-length view `keys[..., :offset, :]`. Allocates a new
        /// graph node per call — for snapshot capture and inspection only,
        /// not the hot path.
        pub fn keys_view(&self) -> Result<Option<Array>> {
            self.keys
                .as_ref()
                .map(|k| slice_axis2(k, 0, self.offset as i32))
                .transpose()
        }

        pub fn values_view(&self) -> Result<Option<Array>> {
            self.values
                .as_ref()
                .map(|v| slice_axis2(v, 0, self.offset as i32))
                .transpose()
        }

        /// Append `keys` / `values` (shape `[B, n_kv_heads, S, head_dim]`) and
        /// return the resulting full `(keys, values)` logical view covering
        /// `0..offset`.
        ///
        /// Mirrors `KVCache.update_and_fetch` (step=256) from mlx_lm — grows
        /// the underlying buffer in step-256 blocks (concat triggered only at
        /// 256-token boundaries), then `slice_update` fills the `prev:offset`
        /// slot in place. Returns logical-length slices that can feed SDPA
        /// directly (an MLX Array clone is a refcount bump on the underlying
        /// buffer, not a deep copy).
        pub fn update_and_fetch(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
            if keys.ndim() != 4 {
                return Err(anyhow!(
                    "NativeKvCache::update_and_fetch: expected 4D keys [B, n_kv_heads, S, head_dim], got {}D",
                    keys.ndim()
                ));
            }
            if values.ndim() != 4 {
                return Err(anyhow!(
                    "NativeKvCache::update_and_fetch: expected 4D values [B, n_kv_heads, S, head_dim], got {}D",
                    values.ndim()
                ));
            }
            let s = keys.shape()[2] as usize;
            if values.shape()[2] as usize != s {
                return Err(anyhow!(
                    "NativeKvCache: keys.shape[2]={} but values.shape[2]={} — must match",
                    s,
                    values.shape()[2]
                ));
            }

            let prev = self.offset;
            let new_total = prev + s;

            // Legacy path — concat per step. Default while
            // `LUMEN_NATIVE_KV_STEP_PREALLOC` is unset.
            if !use_step_prealloc() {
                let (new_keys, new_values) = match (&self.keys, &self.values) {
                    (None, None) => (keys.clone(), values.clone()),
                    (Some(k), Some(v)) => {
                        let combined_keys = mlx_rs::ops::concatenate_axis(&[k, keys], 2)
                            .context("NativeKvCache: concatenate keys along axis=2")?;
                        let combined_values = mlx_rs::ops::concatenate_axis(&[v, values], 2)
                            .context("NativeKvCache: concatenate values along axis=2")?;
                        (combined_keys, combined_values)
                    }
                    _ => unreachable!("keys/values invariant: both Some or both None"),
                };
                self.offset = new_total;
                self.keys = Some(new_keys.clone());
                self.values = Some(new_values.clone());
                return Ok((new_keys, new_values));
            }

            // STEP-PREALLOC PATH (B-1) —
            // Grow the underlying buffer by step-256 blocks if we're about
            // to overrun the current capacity. Mirrors mlx_lm KVCache:
            //   if self.keys is None or (prev + S) > self.keys.shape[2]:
            //       n_steps = ceil((step + S - 1) / step)
            //       new_k = mx.zeros([B, H, n_steps*step, D])
            //       (trim if prev not block-aligned, then concat with new_k)
            let need_grow = self
                .keys
                .as_ref()
                .map_or(true, |k| new_total > k.shape()[2] as usize);

            if need_grow {
                let b = keys.shape()[0];
                let h_k = keys.shape()[1];
                let d_k = keys.shape()[3];
                let h_v = values.shape()[1];
                let d_v = values.shape()[3];
                // Number of blocks needed for the incoming S tokens —
                // matches mlx_lm: `n_steps = (step + keys.shape[2] - 1) // step`
                let n_steps = (KV_CACHE_STEP + s - 1) / KV_CACHE_STEP;
                let new_block_len = (n_steps * KV_CACHE_STEP) as i32;

                let new_k = mlx_rs::ops::zeros_dtype(&[b, h_k, new_block_len, d_k], keys.dtype())
                    .context("NativeKvCache: zeros_dtype for k grow")?;
                let new_v = mlx_rs::ops::zeros_dtype(&[b, h_v, new_block_len, d_v], values.dtype())
                    .context("NativeKvCache: zeros_dtype for v grow")?;

                self.keys = Some(match self.keys.take() {
                    None => new_k,
                    Some(old) => {
                        // mlx_lm: trim partial last block before concat
                        let trimmed = if prev % KV_CACHE_STEP != 0 {
                            slice_axis2(&old, 0, prev as i32)?
                        } else {
                            old
                        };
                        mlx_rs::ops::concatenate_axis(&[trimmed, new_k], 2)
                            .context("NativeKvCache: concat k grow block")?
                    }
                });
                self.values = Some(match self.values.take() {
                    None => new_v,
                    Some(old) => {
                        let trimmed = if prev % KV_CACHE_STEP != 0 {
                            slice_axis2(&old, 0, prev as i32)?
                        } else {
                            old
                        };
                        mlx_rs::ops::concatenate_axis(&[trimmed, new_v], 2)
                            .context("NativeKvCache: concat v grow block")?
                    }
                });
            }

            // In-place slice update: self.keys[..., prev:new_total, :] = keys
            // mlx-rs `try_index_mut` wraps `mlx_slice_update` and reassigns
            // the new graph node back to `self.keys`. The MLX backend reuses
            // the underlying buffer when safe.
            {
                let k_buf = self.keys.as_mut().expect("keys buffer present after grow");
                k_buf
                    .try_index_mut((.., .., (prev as i32)..(new_total as i32), ..), keys)
                    .context("NativeKvCache: slice_update keys")?;
            }
            {
                let v_buf = self
                    .values
                    .as_mut()
                    .expect("values buffer present after grow");
                v_buf
                    .try_index_mut((.., .., (prev as i32)..(new_total as i32), ..), values)
                    .context("NativeKvCache: slice_update values")?;
            }

            self.offset = new_total;

            // Return logical-length view (slice on top of the buffer).
            let k_view = slice_axis2(self.keys.as_ref().unwrap(), 0, new_total as i32)?;
            let v_view = slice_axis2(self.values.as_ref().unwrap(), 0, new_total as i32)?;
            Ok((k_view, v_view))
        }

        /// Drop cache contents. Used when a sequence is removed.
        pub fn clear(&mut self) {
            self.keys = None;
            self.values = None;
            self.offset = 0;
            self.base_offset = 0;
        }

        /// Replace keys / values / offset wholesale. Used by snapshot restore
        /// (`native_snapshot::PromptCacheSnapshot::restore`) to rewind the cache
        /// to a captured state. Caller is responsible for invariants —
        /// `keys.is_some() == values.is_some()` and `keys.shape()[2] >= offset`.
        pub fn set_state(&mut self, keys: Option<Array>, values: Option<Array>, offset: usize) {
            self.keys = keys;
            self.values = values;
            self.offset = offset;
        }

        /// MTP rollback hook — truncate logical offset to `target` and
        /// **flush pending lazy writes** on the underlying buffer. Errors
        /// if `target > offset`.
        ///
        /// unlike Phase 3 which only mutated the CPU-
        /// side `offset` (leaving Step C's slice_update writes pending in
        /// the mlx lazy graph for positions past `target`), this `eval()`s
        /// the buffer to force all pending writes to materialize, then
        /// keeps the buffer at full size. The slots at `[target..buf_size)`
        /// still hold stale draft K/V data, but subsequent reads use slice
        /// `[0..new_offset)` which excludes them, and subsequent writes
        /// (`slice_update` at the new write position) overwrite cleanly.
        ///
        /// We do NOT shrink the buffer (no slice + reassign) because that
        /// causes mlx's slice op to return a strided view sharing storage
        /// with the parent — concat-during-grow on the next decode step
        /// then reads through the strided view in a way that produces
        /// different numeric results than the in-place path (observed:
        /// gemma4_mtp_generate_smoke divergence moved from idx 5 to idx 4
        /// when slicing was attempted).
        pub fn truncate_to(&mut self, target: usize) -> Result<()> {
            if target > self.offset {
                return Err(anyhow!(
                    "NativeKvCache::truncate_to: target {target} > offset {}",
                    self.offset
                ));
            }
            if target == self.offset {
                return Ok(());
            }
            if let (Some(k), Some(v)) = (self.keys.as_ref(), self.values.as_ref()) {
                k.eval().context("NativeKvCache::truncate_to: eval keys")?;
                v.eval()
                    .context("NativeKvCache::truncate_to: eval values")?;
            }
            self.offset = target;
            Ok(())
        }
    }

    impl Default for NativeKvCache {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Quantized full-attention KV cache (mlx_lm `QuantizedKVCache` mirror).
    ///
    /// Tier 1B (2026-05-16): full-attn 5 layers at D=512 dominate 8K decode
    /// KV traffic. 4-bit quantization with group_size=64 cuts per-row bytes
    /// from D*2 (bf16) to D/8*4 + 2*(D/group_size)*2 ≈ 0.28× → ~3.6× compression.
    /// Sliding layers stay bf16 (RotatingKVCache quantization NYI in mlx).
    ///
    /// Stores K and V each as a 3-tuple (packed_uint32, scales_bf16, biases_bf16)
    /// mirroring `mlx.quantize`'s output. Attention path dispatches to
    /// `quantized_matmul` for Q@K^T and scores@V instead of fused sdpa kernels.
    ///
    /// Legacy concat-per-step path (no step-prealloc) — simpler to maintain;
    /// step-prealloc with quantized triples requires zero-init + slice_update
    /// across 3 buffers per K/V, deferred until proven worth the perf delta.
    #[derive(Clone)]
    pub struct NativeKvCacheQuantized {
        /// (packed_uint32, scales, biases) — None until first update.
        keys: Option<(Array, Array, Array)>,
        values: Option<(Array, Array, Array)>,
        offset: usize,
        group_size: i32,
        bits: i32,
    }

    impl NativeKvCacheQuantized {
        pub fn new(group_size: i32, bits: i32) -> Self {
            Self {
                keys: None,
                values: None,
                offset: 0,
                group_size,
                bits,
            }
        }

        pub fn offset(&self) -> usize {
            self.offset
        }

        pub fn empty(&self) -> bool {
            self.keys.is_none()
        }

        pub fn group_size(&self) -> i32 {
            self.group_size
        }

        pub fn bits(&self) -> i32 {
            self.bits
        }

        // Quantized 3-tuple accessors `(packed_uint32, scales, biases)` + parts
        // constructor for P2 disk persistence.
        pub fn keys(&self) -> Option<&(Array, Array, Array)> {
            self.keys.as_ref()
        }
        pub fn values(&self) -> Option<&(Array, Array, Array)> {
            self.values.as_ref()
        }
        pub fn from_parts(
            keys: Option<(Array, Array, Array)>,
            values: Option<(Array, Array, Array)>,
            offset: usize,
            group_size: i32,
            bits: i32,
        ) -> Self {
            Self {
                keys,
                values,
                offset,
                group_size,
                bits,
            }
        }

        pub fn clear(&mut self) {
            self.keys = None;
            self.values = None;
            self.offset = 0;
        }

        /// Prefix-cache rollback hook. Mirrors `NativeKvCache::truncate_to`:
        /// the buffer stays put (step-prealloc grows in 256-token blocks so
        /// `keys/values` already over-allocate past `offset`); we just lower
        /// the logical offset. The next `update_and_fetch` writes at the new
        /// offset and overwrites the stale slots. Force-evaluates outstanding
        /// concat ops so the rewound write sees a settled buffer.
        ///
        /// Errors on the legacy concat path because there `keys/values`
        /// shape == offset, and lowering offset without slicing would leave
        /// the buffer pointing at unrelated data on the next concat. Callers
        /// that rely on prefix cache must keep step-prealloc enabled
        /// (`LUMEN_NATIVE_KV_STEP_PREALLOC` default).
        pub fn truncate_to(&mut self, target: usize) -> Result<()> {
            if target > self.offset {
                return Err(anyhow!(
                    "NativeKvCacheQuantized::truncate_to: target {target} > offset {}",
                    self.offset
                ));
            }
            if target == self.offset {
                return Ok(());
            }
            if !use_step_prealloc() {
                return Err(anyhow!(
                    "NativeKvCacheQuantized::truncate_to: legacy concat path \
                     does not support truncate (buffer shape == offset). \
                     Enable step-prealloc to use prefix cache + QUANT_KV."
                ));
            }
            if let (Some((kp, ks, kb)), Some((vp, vs, vb))) =
                (self.keys.as_ref(), self.values.as_ref())
            {
                for arr in [kp, ks, kb, vp, vs, vb] {
                    arr.eval()
                        .context("NativeKvCacheQuantized::truncate_to: eval triple component")?;
                }
            }
            self.offset = target;
            Ok(())
        }

        /// Append `keys` / `values` (bf16 shape `[B, n_kv_heads, S, head_dim]`)
        /// after quantizing them. Returns the concatenated quantized cache as
        /// two 3-tuples ready for `quantized_matmul`.
        ///
        /// Step-prealloc path (default): grows the underlying 3-tuple buffers
        /// in `KV_CACHE_STEP` (256) blocks, then `try_index_mut` fills the new
        /// rows in place — same pattern as `NativeKvCache` for bf16, mirrors
        /// mlx_lm `QuantizedKVCache`. Avoids per-step concat overhead at
        /// decode (6 concat ops/step/layer → 0).
        /// `LUMEN_NATIVE_KV_STEP_PREALLOC=0` reverts to legacy per-step concat.
        pub fn update_and_fetch(
            &mut self,
            keys: &Array,
            values: &Array,
        ) -> Result<((Array, Array, Array), (Array, Array, Array))> {
            if keys.ndim() != 4 {
                return Err(anyhow!(
                    "NativeKvCacheQuantized::update_and_fetch: expected 4D keys, got {}D",
                    keys.ndim()
                ));
            }
            let s = keys.shape()[2] as usize;
            let prev = self.offset;
            let new_total = prev + s;

            // Quantize incoming K, V along the last axis.
            let (k_p, k_s, k_b) = mlx_rs::ops::quantize(keys, self.group_size, self.bits)
                .context("NativeKvCacheQuantized: quantize keys")?;
            let (v_p, v_s, v_b) = mlx_rs::ops::quantize(values, self.group_size, self.bits)
                .context("NativeKvCacheQuantized: quantize values")?;

            // Legacy concat-per-step path.
            if !use_step_prealloc() {
                let (new_k, new_v) = match (self.keys.take(), self.values.take()) {
                    (None, None) => ((k_p, k_s, k_b), (v_p, v_s, v_b)),
                    (Some((kp, ks, kb)), Some((vp, vs, vb))) => {
                        let np = mlx_rs::ops::concatenate_axis(&[&kp, &k_p], 2)
                            .context("NativeKvCacheQuantized: concat k_packed")?;
                        let nks = mlx_rs::ops::concatenate_axis(&[&ks, &k_s], 2)
                            .context("NativeKvCacheQuantized: concat k_scales")?;
                        let nkb = mlx_rs::ops::concatenate_axis(&[&kb, &k_b], 2)
                            .context("NativeKvCacheQuantized: concat k_biases")?;
                        let nvp = mlx_rs::ops::concatenate_axis(&[&vp, &v_p], 2)
                            .context("NativeKvCacheQuantized: concat v_packed")?;
                        let nvs = mlx_rs::ops::concatenate_axis(&[&vs, &v_s], 2)
                            .context("NativeKvCacheQuantized: concat v_scales")?;
                        let nvb = mlx_rs::ops::concatenate_axis(&[&vb, &v_b], 2)
                            .context("NativeKvCacheQuantized: concat v_biases")?;
                        ((np, nks, nkb), (nvp, nvs, nvb))
                    }
                    _ => unreachable!("keys/values invariant: both Some or both None"),
                };
                self.offset = new_total;
                self.keys = Some(new_k.clone());
                self.values = Some(new_v.clone());
                return Ok((new_k, new_v));
            }

            // STEP-PREALLOC path — grow buffers in KV_CACHE_STEP blocks.
            let need_grow = self
                .keys
                .as_ref()
                .map_or(true, |(kp, _, _)| new_total > kp.shape()[2] as usize);

            if need_grow {
                let b = keys.shape()[0];
                let h_k = keys.shape()[1];
                let h_v = values.shape()[1];
                let kp_last = k_p.shape()[3];
                let ks_last = k_s.shape()[3];
                let kb_last = k_b.shape()[3];
                let vp_last = v_p.shape()[3];
                let vs_last = v_s.shape()[3];
                let vb_last = v_b.shape()[3];

                let n_steps = (KV_CACHE_STEP + s - 1) / KV_CACHE_STEP;
                let new_block_len = (n_steps * KV_CACHE_STEP) as i32;

                let new_kp =
                    mlx_rs::ops::zeros_dtype(&[b, h_k, new_block_len, kp_last], k_p.dtype())
                        .context("NativeKvCacheQuantized: zeros k_packed")?;
                let new_ks =
                    mlx_rs::ops::zeros_dtype(&[b, h_k, new_block_len, ks_last], k_s.dtype())
                        .context("NativeKvCacheQuantized: zeros k_scales")?;
                let new_kb =
                    mlx_rs::ops::zeros_dtype(&[b, h_k, new_block_len, kb_last], k_b.dtype())
                        .context("NativeKvCacheQuantized: zeros k_biases")?;
                let new_vp =
                    mlx_rs::ops::zeros_dtype(&[b, h_v, new_block_len, vp_last], v_p.dtype())
                        .context("NativeKvCacheQuantized: zeros v_packed")?;
                let new_vs =
                    mlx_rs::ops::zeros_dtype(&[b, h_v, new_block_len, vs_last], v_s.dtype())
                        .context("NativeKvCacheQuantized: zeros v_scales")?;
                let new_vb =
                    mlx_rs::ops::zeros_dtype(&[b, h_v, new_block_len, vb_last], v_b.dtype())
                        .context("NativeKvCacheQuantized: zeros v_biases")?;

                self.keys = Some(match self.keys.take() {
                    None => (new_kp, new_ks, new_kb),
                    Some((kp, ks, kb)) => {
                        let (kp_t, ks_t, kb_t) = if prev % KV_CACHE_STEP != 0 {
                            (
                                slice_axis2(&kp, 0, prev as i32)?,
                                slice_axis2(&ks, 0, prev as i32)?,
                                slice_axis2(&kb, 0, prev as i32)?,
                            )
                        } else {
                            (kp, ks, kb)
                        };
                        let np = mlx_rs::ops::concatenate_axis(&[&kp_t, &new_kp], 2)
                            .context("NativeKvCacheQuantized: concat k_packed grow")?;
                        let nks = mlx_rs::ops::concatenate_axis(&[&ks_t, &new_ks], 2)
                            .context("NativeKvCacheQuantized: concat k_scales grow")?;
                        let nkb = mlx_rs::ops::concatenate_axis(&[&kb_t, &new_kb], 2)
                            .context("NativeKvCacheQuantized: concat k_biases grow")?;
                        (np, nks, nkb)
                    }
                });
                self.values = Some(match self.values.take() {
                    None => (new_vp, new_vs, new_vb),
                    Some((vp, vs, vb)) => {
                        let (vp_t, vs_t, vb_t) = if prev % KV_CACHE_STEP != 0 {
                            (
                                slice_axis2(&vp, 0, prev as i32)?,
                                slice_axis2(&vs, 0, prev as i32)?,
                                slice_axis2(&vb, 0, prev as i32)?,
                            )
                        } else {
                            (vp, vs, vb)
                        };
                        let np = mlx_rs::ops::concatenate_axis(&[&vp_t, &new_vp], 2)
                            .context("NativeKvCacheQuantized: concat v_packed grow")?;
                        let nvs = mlx_rs::ops::concatenate_axis(&[&vs_t, &new_vs], 2)
                            .context("NativeKvCacheQuantized: concat v_scales grow")?;
                        let nvb = mlx_rs::ops::concatenate_axis(&[&vb_t, &new_vb], 2)
                            .context("NativeKvCacheQuantized: concat v_biases grow")?;
                        (np, nvs, nvb)
                    }
                });
            }

            // In-place slice update at [..., prev:new_total, :].
            {
                let (kp, ks, kb) = self.keys.as_mut().expect("keys present after grow");
                kp.try_index_mut((.., .., (prev as i32)..(new_total as i32), ..), &k_p)
                    .context("NativeKvCacheQuantized: slice_update k_packed")?;
                ks.try_index_mut((.., .., (prev as i32)..(new_total as i32), ..), &k_s)
                    .context("NativeKvCacheQuantized: slice_update k_scales")?;
                kb.try_index_mut((.., .., (prev as i32)..(new_total as i32), ..), &k_b)
                    .context("NativeKvCacheQuantized: slice_update k_biases")?;
            }
            {
                let (vp, vs, vb) = self.values.as_mut().expect("values present after grow");
                vp.try_index_mut((.., .., (prev as i32)..(new_total as i32), ..), &v_p)
                    .context("NativeKvCacheQuantized: slice_update v_packed")?;
                vs.try_index_mut((.., .., (prev as i32)..(new_total as i32), ..), &v_s)
                    .context("NativeKvCacheQuantized: slice_update v_scales")?;
                vb.try_index_mut((.., .., (prev as i32)..(new_total as i32), ..), &v_b)
                    .context("NativeKvCacheQuantized: slice_update v_biases")?;
            }

            self.offset = new_total;

            // Return logical-length view (slice axis-2 to [0, new_total)).
            let (kp_b, ks_b, kb_b) = self.keys.as_ref().unwrap();
            let (vp_b, vs_b, vb_b) = self.values.as_ref().unwrap();
            let k_view = (
                slice_axis2(kp_b, 0, new_total as i32)?,
                slice_axis2(ks_b, 0, new_total as i32)?,
                slice_axis2(kb_b, 0, new_total as i32)?,
            );
            let v_view = (
                slice_axis2(vp_b, 0, new_total as i32)?,
                slice_axis2(vs_b, 0, new_total as i32)?,
                slice_axis2(vb_b, 0, new_total as i32)?,
            );
            Ok((k_view, v_view))
        }
    }

    impl Default for NativeKvCacheQuantized {
        fn default() -> Self {
            Self::new(64, 4)
        }
    }

    /// Sliding-window KV cache (mlx_lm `RotatingKVCache` mirror).
    ///
    /// Used by Gemma 4 sliding-attention layers (`max_size = sliding_window`,
    /// typically 1024). Tokens past the window are evicted from the middle
    /// of the cache while the first `keep` tokens are pinned (anchor prefix).
    ///
    /// in-place decode fast path landed.
    /// Mirrors mlx_lm `RotatingKVCache._update_in_place`: for S=1 decode the
    /// underlying buffer is pre-allocated in `KV_CACHE_STEP` (256) blocks
    /// and filled via `try_index_mut` (slice_update). This eliminates the
    /// per-step `concatenate_axis` quadratic copy that was the dominant
    /// cause of the 2.2× decode gap vs mlx-lm on Gemma 4 26B-A4B (25
    /// sliding layers × per-step concat over ~30 ms / step). Rotation
    /// (buffer fully populated up to `max_size`) still routes through the
    /// legacy `_update_concat` path — rotation happens at most once per
    /// `max_size` decode steps, so it doesn't dominate the steady state.
    #[derive(Clone)]
    pub struct NativeRotatingKvCache {
        keys: Option<Array>,
        values: Option<Array>,
        /// Total tokens ever pushed (monotonically increasing; can exceed `max_size`).
        offset: usize,
        max_size: usize,
        keep: usize,
        /// Write index into the underlying buffer. After rotation, this can
        /// be less than `offset` (the buffer is filled circularly). Mirrors
        /// mlx-lm's `self._idx`. Initialized to 0; set to `offset` on grow.
        idx: usize,
    }

    impl NativeRotatingKvCache {
        pub fn new(max_size: usize, keep: usize) -> Self {
            assert!(max_size > 0, "RotatingKvCache: max_size must be > 0");
            assert!(
                keep < max_size,
                "RotatingKvCache: keep={keep} must be < max_size={max_size}"
            );
            Self {
                keys: None,
                values: None,
                offset: 0,
                max_size,
                keep,
                idx: 0,
            }
        }

        pub fn offset(&self) -> usize {
            self.offset
        }

        pub fn max_size(&self) -> usize {
            self.max_size
        }

        pub fn keep(&self) -> usize {
            self.keep
        }

        pub fn empty(&self) -> bool {
            self.keys.is_none()
        }

        /// Currently cached token count (≤ `max_size + S - 1` for the last
        /// update batch). Use this for mask construction.
        pub fn cached_len(&self) -> usize {
            match &self.keys {
                Some(k) => k.shape()[2] as usize,
                None => 0,
            }
        }

        pub fn keys(&self) -> Option<&Array> {
            self.keys.as_ref()
        }

        pub fn values(&self) -> Option<&Array> {
            self.values.as_ref()
        }

        pub fn idx(&self) -> usize {
            self.idx
        }

        /// Reconstruct a rotating cache from disk-deserialized parts (P2 disk
        /// persistence). Stores the FULL backing ring buffers verbatim (not a
        /// logical view) so the circular `idx`-based reads reproduce exactly —
        /// `offset`/`idx`/`max_size`/`keep` are restored as-is.
        pub fn from_parts(
            keys: Option<Array>,
            values: Option<Array>,
            offset: usize,
            max_size: usize,
            keep: usize,
            idx: usize,
        ) -> Self {
            Self {
                keys,
                values,
                offset,
                max_size,
                keep,
                idx,
            }
        }

        /// Update KV cache with new tokens. Dispatches to a fast in-place path
        /// for S=1 decode steps (when step-prealloc is enabled and we haven't
        /// hit the rotation cap yet), or a legacy concat path for prefill /
        /// rotation. The fast path eliminates the O(prev × head_dim) DRAM
        /// copy that was the root cause of the 2.2× gap vs mlx-lm.
        pub fn update_and_fetch(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
            if keys.ndim() != 4 {
                return Err(anyhow!(
                    "NativeRotatingKvCache::update_and_fetch: expected 4D keys [B, n_kv_heads, S, head_dim], got {}D",
                    keys.ndim()
                ));
            }
            if values.ndim() != 4 {
                return Err(anyhow!(
                    "NativeRotatingKvCache::update_and_fetch: expected 4D values [B, n_kv_heads, S, head_dim], got {}D",
                    values.ndim()
                ));
            }
            let s = keys.shape()[2] as usize;
            if values.shape()[2] as usize != s {
                return Err(anyhow!(
                    "NativeRotatingKvCache: keys.shape[2]={} but values.shape[2]={}",
                    s,
                    values.shape()[2]
                ));
            }

            // Fast path: S=1 decode. mirrors mlx_lm's
            // `RotatingKVCache._update_in_place` end-to-end (grow / one-time
            // trim if prefill overshoots / rotation via `_idx = keep` when
            // the buffer is full). Previous version bailed once `self.idx`
            // hit `max_size`, dropping back to the legacy slice+slice+concat
            // path for every decode step after rotation — the cause of
            // O(window) per-step memory traffic at long context (-50%+ tok/s
            // vs mlx-lm at 2K-4K context).
            if s == 1 && use_step_prealloc() {
                return self.update_in_place(keys, values);
            }

            // Legacy concat path (prefill / S>1 decode / rotation / disabled).
            //
            // CRITICAL — trim `old_k` to `self.offset` BEFORE concat. With
            // step-prealloc active, `old_k.shape[2]` can include zero-padded
            // slots past the logical `offset` (e.g., after a prior S=1
            // `update_in_place` grew the buffer to KV_CACHE_STEP=256). A
            // naive `concat([old_k, new_keys])` would put the new K/V at
            // physical positions `[old_k.shape[2]..]` while the SDPA mask
            // expects them at logical positions `[offset..offset+s)` — the
            // mask then reads ZEROS at logical `[offset..]` and the new
            // K/V never get attended to. Slicing first restores the
            // logical-length invariant the rest of the code assumes.
            //
            // Bug surfaced 2026-05-18 via MTP Phase 5b real-prompt smoke
            // (mtp_step's Step C uses S=K+1>1 and hit this path after
            // Step A's S=1 grew the buffer). Echo-pattern divergence (each
            // mtp_step's verify_logits[0] read zero K/V for the just-
            // written next_token slot) → committed_token never advanced.
            let (new_keys, new_values) = match (&self.keys, &self.values) {
                (None, None) => (keys.clone(), values.clone()),
                (Some(old_k), Some(old_v)) => {
                    let buf_size = old_k.shape()[2] as usize;
                    let cached = self.offset;
                    let trim_to_logical = cached < buf_size;
                    let old_k_logical = if trim_to_logical {
                        slice_axis2(old_k, 0, cached as i32)?
                    } else {
                        old_k.clone()
                    };
                    let old_v_logical = if trim_to_logical {
                        slice_axis2(old_v, 0, cached as i32)?
                    } else {
                        old_v.clone()
                    };
                    // Compute trim against the size BEFORE concatenating new keys.
                    let trim_size = (cached + 1).saturating_sub(self.max_size);
                    if trim_size > 0 {
                        let head_k = slice_axis2(&old_k_logical, 0, self.keep as i32)?;
                        let tail_k = slice_axis2(
                            &old_k_logical,
                            (trim_size + self.keep) as i32,
                            cached as i32,
                        )?;
                        let head_v = slice_axis2(&old_v_logical, 0, self.keep as i32)?;
                        let tail_v = slice_axis2(
                            &old_v_logical,
                            (trim_size + self.keep) as i32,
                            cached as i32,
                        )?;
                        let combined_keys =
                            mlx_rs::ops::concatenate_axis(&[&head_k, &tail_k, keys], 2)
                                .context("NativeRotatingKvCache: concat keys after trim")?;
                        let combined_values =
                            mlx_rs::ops::concatenate_axis(&[&head_v, &tail_v, values], 2)
                                .context("NativeRotatingKvCache: concat values after trim")?;
                        (combined_keys, combined_values)
                    } else {
                        let combined_keys =
                            mlx_rs::ops::concatenate_axis(&[&old_k_logical, keys], 2)
                                .context("NativeRotatingKvCache: concat keys (no trim)")?;
                        let combined_values =
                            mlx_rs::ops::concatenate_axis(&[&old_v_logical, values], 2)
                                .context("NativeRotatingKvCache: concat values (no trim)")?;
                        (combined_keys, combined_values)
                    }
                }
                _ => unreachable!("keys/values invariant: both Some or both None"),
            };
            self.offset += s;
            // After concat path, `idx` matches the actual cached size.
            self.idx = new_keys.shape()[2] as usize;
            self.keys = Some(new_keys.clone());
            self.values = Some(new_values.clone());
            Ok((new_keys, new_values))
        }

        /// S=1 decode in-place update — full mirror of mlx_lm's
        /// `RotatingKVCache._update_in_place`. Stages:
        ///   1. Grow underlying buffer in `KV_CACHE_STEP` (256) chunks up to
        ///      `max_size` while `offset >= buf_size`.
        ///   2. One-time trim if the buffer is larger than `max_size`
        ///      (happens when prefill stored more than `max_size` tokens).
        ///   3. Rotate `self.idx` to `keep` once it hits `max_size`.
        ///   4. `try_index_mut` (slice_update) at `self.idx`.
        ///   5. Return: full buffer once `offset >= max_size`, else the
        ///      `[0..offset]` view.
        ///
        /// Correctness: SDPA is permutation-invariant given identical K/V
        /// sets, and Gemma 4 applies RoPE BEFORE caching, so each K carries
        /// its own positional encoding. The non-temporal rotation order has
        /// no effect on the attention output for decode (q_len=1).
        ///
        /// Caller must guarantee: `s == 1`.
        fn update_in_place(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
            let s = 1usize;
            let prev = self.offset;

            // ── 1. Grow buffer if we've outrun current capacity and buffer
            //       is still under `max_size`.
            let need_grow = match &self.keys {
                None => true,
                Some(k) => {
                    let buf_size = k.shape()[2] as usize;
                    prev >= buf_size && buf_size < self.max_size
                }
            };
            if need_grow {
                let b = keys.shape()[0];
                let h_k = keys.shape()[1];
                let d_k = keys.shape()[3];
                let h_v = values.shape()[1];
                let d_v = values.shape()[3];

                let new_size = std::cmp::min(KV_CACHE_STEP, self.max_size - prev) as i32;
                let new_k = mlx_rs::ops::zeros_dtype(&[b, h_k, new_size, d_k], keys.dtype())
                    .context("NativeRotatingKvCache: zeros_dtype for k grow")?;
                let new_v = mlx_rs::ops::zeros_dtype(&[b, h_v, new_size, d_v], values.dtype())
                    .context("NativeRotatingKvCache: zeros_dtype for v grow")?;

                self.keys = Some(match self.keys.take() {
                    None => new_k,
                    Some(old) => mlx_rs::ops::concatenate_axis(&[old, new_k], 2)
                        .context("NativeRotatingKvCache: concat k grow block")?,
                });
                self.values = Some(match self.values.take() {
                    None => new_v,
                    Some(old) => mlx_rs::ops::concatenate_axis(&[old, new_v], 2)
                        .context("NativeRotatingKvCache: concat v grow block")?,
                });
                self.idx = prev;
            }

            // ── 2. One-time trim when prefill stored more than max_size
            //       tokens. Matches mlx-lm's `if trim_size > 0:` branch in
            //       `_update_in_place`. After this, buf_size == max_size.
            let buf_size = self.keys.as_ref().unwrap().shape()[2] as usize;
            if buf_size > self.max_size {
                let trim_size = buf_size - self.max_size;
                let keep = self.keep;
                let trimmed_keys = {
                    let k = self.keys.as_ref().unwrap();
                    let head_k = slice_axis2(k, 0, keep as i32)?;
                    let tail_k = slice_axis2(k, (trim_size + keep) as i32, buf_size as i32)?;
                    mlx_rs::ops::concatenate_axis(&[&head_k, &tail_k], 2)
                        .context("NativeRotatingKvCache: trim keys (one-time)")?
                };
                let trimmed_values = {
                    let v = self.values.as_ref().unwrap();
                    let head_v = slice_axis2(v, 0, keep as i32)?;
                    let tail_v = slice_axis2(v, (trim_size + keep) as i32, buf_size as i32)?;
                    mlx_rs::ops::concatenate_axis(&[&head_v, &tail_v], 2)
                        .context("NativeRotatingKvCache: trim values (one-time)")?
                };
                self.keys = Some(trimmed_keys);
                self.values = Some(trimmed_values);
                self.idx = self.max_size;
            }

            // ── 3. Rotate.
            if self.idx == self.max_size {
                self.idx = self.keep;
            }

            // ── 4. slice_update at `self.idx`.
            let write_idx = self.idx;
            let new_idx = write_idx + s;
            {
                let k_buf = self.keys.as_mut().expect("keys buffer present after grow");
                k_buf
                    .try_index_mut((.., .., (write_idx as i32)..(new_idx as i32), ..), keys)
                    .context("NativeRotatingKvCache: slice_update keys")?;
            }
            {
                let v_buf = self
                    .values
                    .as_mut()
                    .expect("values buffer present after grow");
                v_buf
                    .try_index_mut((.., .., (write_idx as i32)..(new_idx as i32), ..), values)
                    .context("NativeRotatingKvCache: slice_update values")?;
            }

            self.idx = new_idx;
            self.offset = prev + s;

            // ── 5. Return. Full buffer once we've wrapped; partial view
            //       otherwise (mirrors mlx-lm exactly).
            if self.offset >= self.max_size {
                let k = self.keys.as_ref().unwrap().clone();
                let v = self.values.as_ref().unwrap().clone();
                Ok((k, v))
            } else {
                let k_view = slice_axis2(self.keys.as_ref().unwrap(), 0, self.offset as i32)?;
                let v_view = slice_axis2(self.values.as_ref().unwrap(), 0, self.offset as i32)?;
                Ok((k_view, v_view))
            }
        }

        pub fn clear(&mut self) {
            self.keys = None;
            self.values = None;
            self.offset = 0;
            self.idx = 0;
        }

        /// MTP rollback hook — truncate logical offset to `target` and
        /// **flush pending lazy writes**. Errors if `target > offset`.
        ///
        /// `eval()` mirror of
        /// `NativeKvCache::truncate_to`. Materializes any pending
        /// `slice_update` writes (so stale draft K/V can't leak via the
        /// mlx lazy graph) while keeping the buffer at its current size.
        /// Slots at `[target..buf_size)` retain stale draft data; later
        /// reads use slice `[0..new_offset)` which excludes them, and
        /// later writes overwrite cleanly. Sets `idx = target` (valid
        /// pre-rotation only).
        ///
        /// **Post-rotation NOT SUPPORTED**: if `offset > max_size`, the
        /// rolled-back range may have overwritten the oldest tokens of
        /// the prior window — those original tokens are unrecoverable
        /// without a snapshot/restore scheme. Callers must guarantee
        /// `offset <= max_size`. Phase 4b' will add ring snapshot.
        pub fn truncate_to(&mut self, target: usize) -> Result<()> {
            if target > self.offset {
                return Err(anyhow!(
                    "NativeRotatingKvCache::truncate_to: target {target} > offset {}",
                    self.offset
                ));
            }
            if target == self.offset {
                return Ok(());
            }
            if self.offset > self.max_size {
                return Err(anyhow!(
                    "NativeRotatingKvCache::truncate_to: post-rotation rollback \
                     not supported (offset={} > max_size={}). MTP must trim \
                     n_draft so writes stay pre-rotation.",
                    self.offset,
                    self.max_size
                ));
            }
            if let (Some(k), Some(v)) = (self.keys.as_ref(), self.values.as_ref()) {
                k.eval()
                    .context("NativeRotatingKvCache::truncate_to: eval keys")?;
                v.eval()
                    .context("NativeRotatingKvCache::truncate_to: eval values")?;
            }
            self.offset = target;
            self.idx = target;
            Ok(())
        }
    }

    /// Quantized rotating-window KV cache.
    ///
    /// Tier 1B sliding (2026-05-17): Gemma 4 26B-A4B has 25 sliding-attention
    /// layers (W=1024 ring) + 5 full-attn layers. Full-attn Q4 was NEUTRAL at
    /// 8K because only 5 layers benefit; sliding has 5× more layers → expected
    /// positive memory + BW win. mlx_lm's `RotatingKVCache.to_quantized()`
    /// raises `NotImplementedError`, so this is a fresh port.
    ///
    /// Stores K and V each as a 3-tuple `(packed_uint32, scales_bf16, biases_bf16)`
    /// mirroring `mlx.quantize`'s output. SDPA dispatches to
    /// `quantized_matmul` for Q@K^T and scores@V instead of the fused sdpa
    /// kernel. Lifecycle mirrors `NativeRotatingKvCache`: S=1 decode uses
    /// `slice_update` into pre-allocated buffers with ring-rotation when
    /// `idx == max_size`; S>1 prefill takes the legacy concat-with-trim path.
    #[derive(Clone)]
    pub struct NativeRotatingKvCacheQuantized {
        /// (packed_uint32, scales, biases) — None until first update.
        keys: Option<(Array, Array, Array)>,
        values: Option<(Array, Array, Array)>,
        /// Total tokens ever pushed (monotonically increasing; can exceed `max_size`).
        offset: usize,
        max_size: usize,
        keep: usize,
        /// Write index into the underlying buffer. After rotation, can be less
        /// than `offset`. Mirrors mlx-lm's `RotatingKVCache._idx`.
        idx: usize,
        group_size: i32,
        bits: i32,
    }

    impl NativeRotatingKvCacheQuantized {
        pub fn new(max_size: usize, keep: usize, group_size: i32, bits: i32) -> Self {
            assert!(
                max_size > 0,
                "RotatingKvCacheQuantized: max_size must be > 0"
            );
            assert!(
                keep < max_size,
                "RotatingKvCacheQuantized: keep={keep} must be < max_size={max_size}"
            );
            Self {
                keys: None,
                values: None,
                offset: 0,
                max_size,
                keep,
                idx: 0,
                group_size,
                bits,
            }
        }

        pub fn offset(&self) -> usize {
            self.offset
        }

        pub fn max_size(&self) -> usize {
            self.max_size
        }

        pub fn keep(&self) -> usize {
            self.keep
        }

        pub fn group_size(&self) -> i32 {
            self.group_size
        }

        pub fn bits(&self) -> i32 {
            self.bits
        }

        pub fn idx(&self) -> usize {
            self.idx
        }

        // Quantized 3-tuple accessors + parts constructor (P2 disk persistence).
        pub fn keys(&self) -> Option<&(Array, Array, Array)> {
            self.keys.as_ref()
        }
        pub fn values(&self) -> Option<&(Array, Array, Array)> {
            self.values.as_ref()
        }
        #[allow(clippy::too_many_arguments)]
        pub fn from_parts(
            keys: Option<(Array, Array, Array)>,
            values: Option<(Array, Array, Array)>,
            offset: usize,
            max_size: usize,
            keep: usize,
            idx: usize,
            group_size: i32,
            bits: i32,
        ) -> Self {
            Self {
                keys,
                values,
                offset,
                max_size,
                keep,
                idx,
                group_size,
                bits,
            }
        }

        pub fn empty(&self) -> bool {
            self.keys.is_none()
        }

        /// Currently cached token count (≤ `max_size` after rotation).
        pub fn cached_len(&self) -> usize {
            match &self.keys {
                Some((kp, _, _)) => kp.shape()[2] as usize,
                None => 0,
            }
        }

        pub fn clear(&mut self) {
            self.keys = None;
            self.values = None;
            self.offset = 0;
            self.idx = 0;
        }

        /// Prefix-cache rollback hook. Same shape as
        /// `NativeRotatingKvCache::truncate_to` (rotation guard +
        /// offset/idx rewind), but evals the 6 triple components instead
        /// of 2 dense arrays. Pre-rotation only — post-rotation rollback
        /// is unsafe because the rolled-back range may have overwritten
        /// the oldest tokens of the prior window.
        pub fn truncate_to(&mut self, target: usize) -> Result<()> {
            if target > self.offset {
                return Err(anyhow!(
                    "NativeRotatingKvCacheQuantized::truncate_to: target {target} > offset {}",
                    self.offset
                ));
            }
            if target == self.offset {
                return Ok(());
            }
            if self.offset > self.max_size {
                return Err(anyhow!(
                    "NativeRotatingKvCacheQuantized::truncate_to: post-rotation rollback \
                     not supported (offset={} > max_size={}). Prefix cache must keep \
                     prompts ≤ sliding_window.",
                    self.offset,
                    self.max_size
                ));
            }
            if let (Some((kp, ks, kb)), Some((vp, vs, vb))) =
                (self.keys.as_ref(), self.values.as_ref())
            {
                for arr in [kp, ks, kb, vp, vs, vb] {
                    arr.eval().context(
                        "NativeRotatingKvCacheQuantized::truncate_to: eval triple component",
                    )?;
                }
            }
            self.offset = target;
            self.idx = target;
            Ok(())
        }

        /// Append `keys` / `values` (bf16, shape `[B, n_kv_heads, S, head_dim]`)
        /// after quantizing them. Returns the in-ring quantized cache as two
        /// 3-tuples ready for `quantized_matmul`.
        ///
        /// S=1 decode takes the in-place slice_update fast path; S>1 prefill
        /// uses concat-with-trim against `max_size`.
        pub fn update_and_fetch(
            &mut self,
            keys: &Array,
            values: &Array,
        ) -> Result<((Array, Array, Array), (Array, Array, Array))> {
            if keys.ndim() != 4 {
                return Err(anyhow!(
                    "NativeRotatingKvCacheQuantized::update_and_fetch: expected 4D keys, got {}D",
                    keys.ndim()
                ));
            }
            if values.ndim() != 4 {
                return Err(anyhow!(
                    "NativeRotatingKvCacheQuantized::update_and_fetch: expected 4D values, got {}D",
                    values.ndim()
                ));
            }
            let s = keys.shape()[2] as usize;
            if values.shape()[2] as usize != s {
                return Err(anyhow!(
                    "NativeRotatingKvCacheQuantized: keys.shape[2]={} but values.shape[2]={}",
                    s,
                    values.shape()[2]
                ));
            }

            // Quantize incoming K, V along the last axis.
            let (k_p, k_s, k_b) = mlx_rs::ops::quantize(keys, self.group_size, self.bits)
                .context("NativeRotatingKvCacheQuantized: quantize keys")?;
            let (v_p, v_s, v_b) = mlx_rs::ops::quantize(values, self.group_size, self.bits)
                .context("NativeRotatingKvCacheQuantized: quantize values")?;

            // S=1 decode fast path — full mirror of bf16 RotatingKvCache's
            // `_update_in_place` (grow / one-time trim / rotate / slice_update),
            // applied across 6 quantized buffers (3 each for K and V).
            if s == 1 && use_step_prealloc() {
                return self.update_in_place_q(k_p, k_s, k_b, v_p, v_s, v_b);
            }

            // Legacy concat-with-trim path — covers S>1 prefill (single-pass
            // or chunked) and the step-prealloc-disabled fallback. Trims the
            // existing cached span by `(cached + s) - max_size` before concat
            // so the post-concat length never exceeds `max_size`.
            let (new_k, new_v) = match (self.keys.take(), self.values.take()) {
                (None, None) => ((k_p, k_s, k_b), (v_p, v_s, v_b)),
                (Some((kp, ks, kb)), Some((vp, vs, vb))) => {
                    let cached = kp.shape()[2] as usize;
                    let trim_size = (cached + s).saturating_sub(self.max_size);
                    if trim_size > 0 {
                        let keep_i = self.keep as i32;
                        let trim_keep = (trim_size + self.keep) as i32;
                        let cached_i = cached as i32;
                        let head_kp = slice_axis2(&kp, 0, keep_i)?;
                        let tail_kp = slice_axis2(&kp, trim_keep, cached_i)?;
                        let head_ks = slice_axis2(&ks, 0, keep_i)?;
                        let tail_ks = slice_axis2(&ks, trim_keep, cached_i)?;
                        let head_kb = slice_axis2(&kb, 0, keep_i)?;
                        let tail_kb = slice_axis2(&kb, trim_keep, cached_i)?;
                        let head_vp = slice_axis2(&vp, 0, keep_i)?;
                        let tail_vp = slice_axis2(&vp, trim_keep, cached_i)?;
                        let head_vs = slice_axis2(&vs, 0, keep_i)?;
                        let tail_vs = slice_axis2(&vs, trim_keep, cached_i)?;
                        let head_vb = slice_axis2(&vb, 0, keep_i)?;
                        let tail_vb = slice_axis2(&vb, trim_keep, cached_i)?;
                        let new_kp = mlx_rs::ops::concatenate_axis(&[&head_kp, &tail_kp, &k_p], 2)
                            .context("RotatingQuant: concat k_packed after trim")?;
                        let new_ks = mlx_rs::ops::concatenate_axis(&[&head_ks, &tail_ks, &k_s], 2)
                            .context("RotatingQuant: concat k_scales after trim")?;
                        let new_kb = mlx_rs::ops::concatenate_axis(&[&head_kb, &tail_kb, &k_b], 2)
                            .context("RotatingQuant: concat k_biases after trim")?;
                        let new_vp = mlx_rs::ops::concatenate_axis(&[&head_vp, &tail_vp, &v_p], 2)
                            .context("RotatingQuant: concat v_packed after trim")?;
                        let new_vs = mlx_rs::ops::concatenate_axis(&[&head_vs, &tail_vs, &v_s], 2)
                            .context("RotatingQuant: concat v_scales after trim")?;
                        let new_vb = mlx_rs::ops::concatenate_axis(&[&head_vb, &tail_vb, &v_b], 2)
                            .context("RotatingQuant: concat v_biases after trim")?;
                        ((new_kp, new_ks, new_kb), (new_vp, new_vs, new_vb))
                    } else {
                        let new_kp = mlx_rs::ops::concatenate_axis(&[&kp, &k_p], 2)
                            .context("RotatingQuant: concat k_packed (no trim)")?;
                        let new_ks = mlx_rs::ops::concatenate_axis(&[&ks, &k_s], 2)
                            .context("RotatingQuant: concat k_scales (no trim)")?;
                        let new_kb = mlx_rs::ops::concatenate_axis(&[&kb, &k_b], 2)
                            .context("RotatingQuant: concat k_biases (no trim)")?;
                        let new_vp = mlx_rs::ops::concatenate_axis(&[&vp, &v_p], 2)
                            .context("RotatingQuant: concat v_packed (no trim)")?;
                        let new_vs = mlx_rs::ops::concatenate_axis(&[&vs, &v_s], 2)
                            .context("RotatingQuant: concat v_scales (no trim)")?;
                        let new_vb = mlx_rs::ops::concatenate_axis(&[&vb, &v_b], 2)
                            .context("RotatingQuant: concat v_biases (no trim)")?;
                        ((new_kp, new_ks, new_kb), (new_vp, new_vs, new_vb))
                    }
                }
                _ => unreachable!("keys/values invariant: both Some or both None"),
            };
            self.offset += s;
            // After concat path, idx tracks the actual cached size (capped at max_size).
            self.idx = new_k.0.shape()[2] as usize;
            self.keys = Some((new_k.0.clone(), new_k.1.clone(), new_k.2.clone()));
            self.values = Some((new_v.0.clone(), new_v.1.clone(), new_v.2.clone()));
            Ok((new_k, new_v))
        }

        /// S=1 decode in-place update — 6-buffer mirror of bf16
        /// `NativeRotatingKvCache::update_in_place`. Caller guarantees S=1.
        fn update_in_place_q(
            &mut self,
            k_p: Array,
            k_s: Array,
            k_b: Array,
            v_p: Array,
            v_s: Array,
            v_b: Array,
        ) -> Result<((Array, Array, Array), (Array, Array, Array))> {
            let s = 1usize;
            let prev = self.offset;

            // ── 1. Grow buffer if we've outrun current capacity AND the
            //       buffer is still smaller than `max_size`. Mirrors
            //       mlx-lm's lazy ring expansion in KV_CACHE_STEP chunks.
            let need_grow = match &self.keys {
                None => true,
                Some((kp, _, _)) => {
                    let buf_size = kp.shape()[2] as usize;
                    prev >= buf_size && buf_size < self.max_size
                }
            };
            if need_grow {
                let b = k_p.shape()[0];
                let h_k = k_p.shape()[1];
                let kp_last = k_p.shape()[3];
                let ks_last = k_s.shape()[3];
                let kb_last = k_b.shape()[3];
                let h_v = v_p.shape()[1];
                let vp_last = v_p.shape()[3];
                let vs_last = v_s.shape()[3];
                let vb_last = v_b.shape()[3];

                let new_size = std::cmp::min(KV_CACHE_STEP, self.max_size - prev) as i32;
                let new_kp = mlx_rs::ops::zeros_dtype(&[b, h_k, new_size, kp_last], k_p.dtype())
                    .context("RotatingQuant: zeros k_packed grow")?;
                let new_ks = mlx_rs::ops::zeros_dtype(&[b, h_k, new_size, ks_last], k_s.dtype())
                    .context("RotatingQuant: zeros k_scales grow")?;
                let new_kb = mlx_rs::ops::zeros_dtype(&[b, h_k, new_size, kb_last], k_b.dtype())
                    .context("RotatingQuant: zeros k_biases grow")?;
                let new_vp = mlx_rs::ops::zeros_dtype(&[b, h_v, new_size, vp_last], v_p.dtype())
                    .context("RotatingQuant: zeros v_packed grow")?;
                let new_vs = mlx_rs::ops::zeros_dtype(&[b, h_v, new_size, vs_last], v_s.dtype())
                    .context("RotatingQuant: zeros v_scales grow")?;
                let new_vb = mlx_rs::ops::zeros_dtype(&[b, h_v, new_size, vb_last], v_b.dtype())
                    .context("RotatingQuant: zeros v_biases grow")?;

                self.keys = Some(match self.keys.take() {
                    None => (new_kp, new_ks, new_kb),
                    Some((kp, ks, kb)) => (
                        mlx_rs::ops::concatenate_axis(&[&kp, &new_kp], 2)
                            .context("RotatingQuant: concat k_packed grow block")?,
                        mlx_rs::ops::concatenate_axis(&[&ks, &new_ks], 2)
                            .context("RotatingQuant: concat k_scales grow block")?,
                        mlx_rs::ops::concatenate_axis(&[&kb, &new_kb], 2)
                            .context("RotatingQuant: concat k_biases grow block")?,
                    ),
                });
                self.values = Some(match self.values.take() {
                    None => (new_vp, new_vs, new_vb),
                    Some((vp, vs, vb)) => (
                        mlx_rs::ops::concatenate_axis(&[&vp, &new_vp], 2)
                            .context("RotatingQuant: concat v_packed grow block")?,
                        mlx_rs::ops::concatenate_axis(&[&vs, &new_vs], 2)
                            .context("RotatingQuant: concat v_scales grow block")?,
                        mlx_rs::ops::concatenate_axis(&[&vb, &new_vb], 2)
                            .context("RotatingQuant: concat v_biases grow block")?,
                    ),
                });
                self.idx = prev;
            }

            // ── 2. One-time trim if prefill stored more than max_size tokens.
            //       After this, buf_size == max_size.
            let buf_size = self.keys.as_ref().unwrap().0.shape()[2] as usize;
            if buf_size > self.max_size {
                let trim_size = buf_size - self.max_size;
                let keep = self.keep;
                let keep_i = keep as i32;
                let trim_keep = (trim_size + keep) as i32;
                let buf_i = buf_size as i32;
                let (kp, ks, kb) = self.keys.take().unwrap();
                let (vp, vs, vb) = self.values.take().unwrap();
                let trimmed_k = {
                    let h_kp = slice_axis2(&kp, 0, keep_i)?;
                    let t_kp = slice_axis2(&kp, trim_keep, buf_i)?;
                    let h_ks = slice_axis2(&ks, 0, keep_i)?;
                    let t_ks = slice_axis2(&ks, trim_keep, buf_i)?;
                    let h_kb = slice_axis2(&kb, 0, keep_i)?;
                    let t_kb = slice_axis2(&kb, trim_keep, buf_i)?;
                    (
                        mlx_rs::ops::concatenate_axis(&[&h_kp, &t_kp], 2)
                            .context("RotatingQuant: trim k_packed")?,
                        mlx_rs::ops::concatenate_axis(&[&h_ks, &t_ks], 2)
                            .context("RotatingQuant: trim k_scales")?,
                        mlx_rs::ops::concatenate_axis(&[&h_kb, &t_kb], 2)
                            .context("RotatingQuant: trim k_biases")?,
                    )
                };
                let trimmed_v = {
                    let h_vp = slice_axis2(&vp, 0, keep_i)?;
                    let t_vp = slice_axis2(&vp, trim_keep, buf_i)?;
                    let h_vs = slice_axis2(&vs, 0, keep_i)?;
                    let t_vs = slice_axis2(&vs, trim_keep, buf_i)?;
                    let h_vb = slice_axis2(&vb, 0, keep_i)?;
                    let t_vb = slice_axis2(&vb, trim_keep, buf_i)?;
                    (
                        mlx_rs::ops::concatenate_axis(&[&h_vp, &t_vp], 2)
                            .context("RotatingQuant: trim v_packed")?,
                        mlx_rs::ops::concatenate_axis(&[&h_vs, &t_vs], 2)
                            .context("RotatingQuant: trim v_scales")?,
                        mlx_rs::ops::concatenate_axis(&[&h_vb, &t_vb], 2)
                            .context("RotatingQuant: trim v_biases")?,
                    )
                };
                self.keys = Some(trimmed_k);
                self.values = Some(trimmed_v);
                self.idx = self.max_size;
            }

            // ── 3. Rotate.
            if self.idx == self.max_size {
                self.idx = self.keep;
            }

            // ── 4. slice_update at idx across all 6 buffers.
            let write_idx = self.idx;
            let new_idx = write_idx + s;
            {
                let (kp, ks, kb) = self.keys.as_mut().expect("keys present after grow");
                kp.try_index_mut((.., .., (write_idx as i32)..(new_idx as i32), ..), &k_p)
                    .context("RotatingQuant: slice_update k_packed")?;
                ks.try_index_mut((.., .., (write_idx as i32)..(new_idx as i32), ..), &k_s)
                    .context("RotatingQuant: slice_update k_scales")?;
                kb.try_index_mut((.., .., (write_idx as i32)..(new_idx as i32), ..), &k_b)
                    .context("RotatingQuant: slice_update k_biases")?;
            }
            {
                let (vp, vs, vb) = self.values.as_mut().expect("values present after grow");
                vp.try_index_mut((.., .., (write_idx as i32)..(new_idx as i32), ..), &v_p)
                    .context("RotatingQuant: slice_update v_packed")?;
                vs.try_index_mut((.., .., (write_idx as i32)..(new_idx as i32), ..), &v_s)
                    .context("RotatingQuant: slice_update v_scales")?;
                vb.try_index_mut((.., .., (write_idx as i32)..(new_idx as i32), ..), &v_b)
                    .context("RotatingQuant: slice_update v_biases")?;
            }

            self.idx = new_idx;
            self.offset = prev + s;

            // ── 5. Return: full ring once wrapped; partial view otherwise.
            if self.offset >= self.max_size {
                let (kp, ks, kb) = self.keys.as_ref().unwrap();
                let (vp, vs, vb) = self.values.as_ref().unwrap();
                Ok((
                    (kp.clone(), ks.clone(), kb.clone()),
                    (vp.clone(), vs.clone(), vb.clone()),
                ))
            } else {
                let (kp, ks, kb) = self.keys.as_ref().unwrap();
                let (vp, vs, vb) = self.values.as_ref().unwrap();
                let off_i = self.offset as i32;
                let k_view = (
                    slice_axis2(kp, 0, off_i)?,
                    slice_axis2(ks, 0, off_i)?,
                    slice_axis2(kb, 0, off_i)?,
                );
                let v_view = (
                    slice_axis2(vp, 0, off_i)?,
                    slice_axis2(vs, 0, off_i)?,
                    slice_axis2(vb, 0, off_i)?,
                );
                Ok((k_view, v_view))
            }
        }
    }

    impl Default for NativeRotatingKvCacheQuantized {
        fn default() -> Self {
            Self::new(1024, 0, 64, 4)
        }
    }

    /// TurboQuant Stage-1 sliding-window KV cache.
    ///
    /// Stores K and V each as `(codes uint8, sigma f32)` instead of the
    /// quantized 3-tuple used by `NativeRotatingKvCacheQuantized`. Codes are
    /// Lloyd-Max nearest-centroid indices against a fixed N(0,1)-optimal
    /// codebook (precomputed once per-bit in `turboquant::lloyd_max_centroids`).
    ///
    /// Quantization is performed inside `update_and_fetch` after the caller
    /// has already rotated K (rotation is K-side only and lives in the SDPA
    /// inline branch). V is quantized WITHOUT rotation — it must stay in the
    /// original head_dim space so the SDPA output flows directly into o_proj.
    ///
    /// Memory at 1024-token ring (bits=4 unpacked uint8):
    ///   - codes  : [B, n_kv, 1024, D] uint8 = 1024 × D bytes per (K|V) per layer
    ///   - sigma  : [B, n_kv, 1024, 1] f32   = 4096 bytes per (K|V) per layer
    /// vs bf16 baseline of 1024 × D × 2 bytes. 2× compression at uint8;
    /// adding 4-bit packing would push to ~3.5×.
    ///
    /// QJL Stage-2 (1-bit residual correction) is deferred — Stage 1 alone
    /// recovers ≥0.99 cosine similarity per the paper.
    #[derive(Clone)]
    pub struct NativeRotatingKvCacheTurboQuant {
        keys_codes: Option<Array>,
        keys_sigma: Option<Array>,
        values_codes: Option<Array>,
        values_sigma: Option<Array>,
        // QJL Stage-2 slots — only populated when `qjl_m.is_some()`. K-side
        // only; V uses Stage 1 alone (no per-row inner-product structure).
        keys_signs: Option<Array>,         // [B, n_kv, S, m] bf16 ±1
        keys_residual_norm: Option<Array>, // [B, n_kv, S, 1] f32
        /// Total tokens ever pushed.
        offset: usize,
        max_size: usize,
        keep: usize,
        /// Write index into the ring (mirrors mlx-lm `_idx`).
        idx: usize,
        /// Cached for `cached_len()` reporting.
        bits: u32,
        /// QJL projection width (m). `None` ↔ Stage 1 only; `Some(m)` ↔
        /// Stage 2 active and the `keys_signs` / `keys_residual_norm` slots
        /// must move in lockstep with the four Stage-1 slots.
        qjl_m: Option<usize>,
    }

    impl NativeRotatingKvCacheTurboQuant {
        pub fn new(max_size: usize, keep: usize, bits: u32) -> Self {
            assert!(max_size > 0, "TurboQuant cache: max_size must be > 0");
            assert!(
                keep < max_size,
                "TurboQuant cache: keep={keep} must be < max_size={max_size}"
            );
            assert!(
                (2..=8).contains(&bits),
                "TurboQuant cache: bits must be in [2, 8], got {bits}"
            );
            Self {
                keys_codes: None,
                keys_sigma: None,
                values_codes: None,
                values_sigma: None,
                keys_signs: None,
                keys_residual_norm: None,
                offset: 0,
                max_size,
                keep,
                idx: 0,
                bits,
                qjl_m: None,
            }
        }

        /// Enable QJL Stage-2 storage with projection width `qjl_m`. Must be
        /// called before any `update_and_fetch_qjl` invocation; toggling
        /// during a session would desynchronize the K-side slots.
        pub fn with_qjl(mut self, qjl_m: usize) -> Self {
            assert!(qjl_m > 0, "TurboQuant cache: qjl_m must be > 0");
            self.qjl_m = Some(qjl_m);
            self
        }

        /// Deep-clone every backing Array into independent buffers — required
        /// for fork-to-new-seq (Track A1 prefix cache). Mirrors
        /// `native_snapshot::deep_clone_array`: `add(a, 0) + eval` forces a
        /// fresh contiguous materialization (avoids mlx-rs `deep_clone`'s
        /// stride bug). Scalar bookkeeping is copied verbatim.
        pub fn deep_clone(&self) -> Result<Self> {
            fn dc(a: &Option<Array>) -> Result<Option<Array>> {
                match a {
                    Some(arr) => {
                        let zero = Array::from_f32(0.0)
                            .as_dtype(arr.dtype())
                            .context("TurboQuant deep_clone: cast zero scalar")?;
                        let cloned = mlx_rs::ops::add(arr, &zero)
                            .context("TurboQuant deep_clone: add(a, 0)")?;
                        cloned.eval().context("TurboQuant deep_clone: eval")?;
                        Ok(Some(cloned))
                    }
                    None => Ok(None),
                }
            }
            Ok(Self {
                keys_codes: dc(&self.keys_codes)?,
                keys_sigma: dc(&self.keys_sigma)?,
                values_codes: dc(&self.values_codes)?,
                values_sigma: dc(&self.values_sigma)?,
                keys_signs: dc(&self.keys_signs)?,
                keys_residual_norm: dc(&self.keys_residual_norm)?,
                offset: self.offset,
                max_size: self.max_size,
                keep: self.keep,
                idx: self.idx,
                bits: self.bits,
                qjl_m: self.qjl_m,
            })
        }

        /// Fresh empty cache with identical geometry (max_size / keep / bits /
        /// QJL width) but no stored tokens — used by snapshot fork structure.
        pub fn structure_clone(&self) -> Self {
            let fresh = Self::new(self.max_size, self.keep, self.bits);
            match self.qjl_m {
                Some(m) => fresh.with_qjl(m),
                None => fresh,
            }
        }

        pub fn qjl_m(&self) -> Option<usize> {
            self.qjl_m
        }

        pub fn offset(&self) -> usize {
            self.offset
        }

        pub fn max_size(&self) -> usize {
            self.max_size
        }

        pub fn keep(&self) -> usize {
            self.keep
        }

        pub fn bits(&self) -> u32 {
            self.bits
        }

        pub fn idx(&self) -> usize {
            self.idx
        }

        // Per-slot Array accessors — used by disk persistence (P1c) to read the
        // compressed ring buffers out for serialization.
        pub fn keys_codes(&self) -> Option<&Array> {
            self.keys_codes.as_ref()
        }
        pub fn keys_sigma(&self) -> Option<&Array> {
            self.keys_sigma.as_ref()
        }
        pub fn values_codes(&self) -> Option<&Array> {
            self.values_codes.as_ref()
        }
        pub fn values_sigma(&self) -> Option<&Array> {
            self.values_sigma.as_ref()
        }
        pub fn keys_signs(&self) -> Option<&Array> {
            self.keys_signs.as_ref()
        }
        pub fn keys_residual_norm(&self) -> Option<&Array> {
            self.keys_residual_norm.as_ref()
        }

        /// Reconstruct a TurboQuant cache from disk-deserialized parts (P1c disk
        /// persistence) — the inverse of the per-slot accessors. Scalar geometry
        /// is restored verbatim; the caller passes arrays whose shapes already
        /// match the recorded `offset` / `idx` / `max_size` bookkeeping.
        #[allow(clippy::too_many_arguments)]
        pub fn from_parts(
            keys_codes: Option<Array>,
            keys_sigma: Option<Array>,
            values_codes: Option<Array>,
            values_sigma: Option<Array>,
            keys_signs: Option<Array>,
            keys_residual_norm: Option<Array>,
            offset: usize,
            max_size: usize,
            keep: usize,
            idx: usize,
            bits: u32,
            qjl_m: Option<usize>,
        ) -> Self {
            Self {
                keys_codes,
                keys_sigma,
                values_codes,
                values_sigma,
                keys_signs,
                keys_residual_norm,
                offset,
                max_size,
                keep,
                idx,
                bits,
                qjl_m,
            }
        }

        pub fn empty(&self) -> bool {
            self.keys_codes.is_none()
        }

        pub fn cached_len(&self) -> usize {
            match &self.keys_codes {
                Some(k) => k.shape()[2] as usize,
                None => 0,
            }
        }

        pub fn clear(&mut self) {
            self.keys_codes = None;
            self.keys_sigma = None;
            self.keys_signs = None;
            self.keys_residual_norm = None;
            self.values_codes = None;
            self.values_sigma = None;
            self.offset = 0;
            self.idx = 0;
        }

        /// Prefix-cache rollback hook for TurboQuant Stage-1 (+ optional
        /// Stage-2 QJL) caches. Mirrors `NativeRotatingKvCacheQuantized`:
        /// pre-rotation only, evals all ring slots before rewinding
        /// `offset`/`idx`. QJL slots (`keys_signs` / `keys_residual_norm`)
        /// are evaluated when present so Stage-2 cache stays in sync.
        pub fn truncate_to(&mut self, target: usize) -> Result<()> {
            if target > self.offset {
                return Err(anyhow!(
                    "NativeRotatingKvCacheTurboQuant::truncate_to: target {target} > offset {}",
                    self.offset
                ));
            }
            if target == self.offset {
                return Ok(());
            }
            if self.offset > self.max_size {
                return Err(anyhow!(
                    "NativeRotatingKvCacheTurboQuant::truncate_to: post-rotation rollback \
                     not supported (offset={} > max_size={}).",
                    self.offset,
                    self.max_size
                ));
            }
            for arr in [
                self.keys_codes.as_ref(),
                self.keys_sigma.as_ref(),
                self.values_codes.as_ref(),
                self.values_sigma.as_ref(),
                self.keys_signs.as_ref(),
                self.keys_residual_norm.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                arr.eval()
                    .context("NativeRotatingKvCacheTurboQuant::truncate_to: eval ring slot")?;
            }
            self.offset = target;
            self.idx = target;
            Ok(())
        }

        /// Insert pre-quantized `(k_codes, k_sigma, v_codes, v_sigma)` and
        /// return the in-ring views. Caller is responsible for: (a) rotating
        /// K before quantize, (b) running `lloyd_max_quantize_stage1` on
        /// rotated K and on raw V.
        ///
        /// Returns `((k_codes_view, k_sigma_view), (v_codes_view, v_sigma_view))`
        /// covering the logical token span: `[0, offset)` while filling, or
        /// the full ring `[0, max_size)` once wrapped.
        pub fn update_and_fetch(
            &mut self,
            k_codes: &Array,
            k_sigma: &Array,
            v_codes: &Array,
            v_sigma: &Array,
        ) -> Result<((Array, Array), (Array, Array))> {
            if k_codes.ndim() != 4 || v_codes.ndim() != 4 {
                return Err(anyhow!(
                    "TurboQuant cache: expected 4D codes [B, n_kv, S, D], got K={}D V={}D",
                    k_codes.ndim(),
                    v_codes.ndim()
                ));
            }
            let s = k_codes.shape()[2] as usize;
            if v_codes.shape()[2] as usize != s {
                return Err(anyhow!(
                    "TurboQuant cache: k_codes.shape[2]={} but v_codes.shape[2]={}",
                    s,
                    v_codes.shape()[2]
                ));
            }

            // S=1 decode fast path: in-place slice_update with grow / rotate.
            if s == 1 && use_step_prealloc() {
                return self.update_in_place_tq(k_codes, k_sigma, v_codes, v_sigma);
            }

            // Legacy concat-with-trim path (S>1 prefill or step-prealloc off).
            let (new_k_codes, new_k_sigma, new_v_codes, new_v_sigma) = match (
                self.keys_codes.take(),
                self.keys_sigma.take(),
                self.values_codes.take(),
                self.values_sigma.take(),
            ) {
                (None, None, None, None) => (
                    k_codes.clone(),
                    k_sigma.clone(),
                    v_codes.clone(),
                    v_sigma.clone(),
                ),
                (Some(kc), Some(ks), Some(vc), Some(vs)) => {
                    let cached = kc.shape()[2] as usize;
                    let trim_size = (cached + s).saturating_sub(self.max_size);
                    let cached_i = cached as i32;
                    let keep_i = self.keep as i32;
                    let trim_keep = (trim_size + self.keep) as i32;
                    if trim_size > 0 {
                        let head_kc = slice_axis2(&kc, 0, keep_i)?;
                        let tail_kc = slice_axis2(&kc, trim_keep, cached_i)?;
                        let head_ks = slice_axis2(&ks, 0, keep_i)?;
                        let tail_ks = slice_axis2(&ks, trim_keep, cached_i)?;
                        let head_vc = slice_axis2(&vc, 0, keep_i)?;
                        let tail_vc = slice_axis2(&vc, trim_keep, cached_i)?;
                        let head_vs = slice_axis2(&vs, 0, keep_i)?;
                        let tail_vs = slice_axis2(&vs, trim_keep, cached_i)?;
                        (
                            mlx_rs::ops::concatenate_axis(&[&head_kc, &tail_kc, k_codes], 2)
                                .context("TurboQuant: concat k_codes after trim")?,
                            mlx_rs::ops::concatenate_axis(&[&head_ks, &tail_ks, k_sigma], 2)
                                .context("TurboQuant: concat k_sigma after trim")?,
                            mlx_rs::ops::concatenate_axis(&[&head_vc, &tail_vc, v_codes], 2)
                                .context("TurboQuant: concat v_codes after trim")?,
                            mlx_rs::ops::concatenate_axis(&[&head_vs, &tail_vs, v_sigma], 2)
                                .context("TurboQuant: concat v_sigma after trim")?,
                        )
                    } else {
                        (
                            mlx_rs::ops::concatenate_axis(&[&kc, k_codes], 2)
                                .context("TurboQuant: concat k_codes (no trim)")?,
                            mlx_rs::ops::concatenate_axis(&[&ks, k_sigma], 2)
                                .context("TurboQuant: concat k_sigma (no trim)")?,
                            mlx_rs::ops::concatenate_axis(&[&vc, v_codes], 2)
                                .context("TurboQuant: concat v_codes (no trim)")?,
                            mlx_rs::ops::concatenate_axis(&[&vs, v_sigma], 2)
                                .context("TurboQuant: concat v_sigma (no trim)")?,
                        )
                    }
                }
                _ => unreachable!("TurboQuant cache invariant: all four fields move in lockstep"),
            };
            self.offset += s;
            self.idx = new_k_codes.shape()[2] as usize;
            self.keys_codes = Some(new_k_codes.clone());
            self.keys_sigma = Some(new_k_sigma.clone());
            self.values_codes = Some(new_v_codes.clone());
            self.values_sigma = Some(new_v_sigma.clone());
            Ok(((new_k_codes, new_k_sigma), (new_v_codes, new_v_sigma)))
        }

        fn update_in_place_tq(
            &mut self,
            k_codes: &Array,
            k_sigma: &Array,
            v_codes: &Array,
            v_sigma: &Array,
        ) -> Result<((Array, Array), (Array, Array))> {
            let s = 1usize;
            let prev = self.offset;

            // ── 1. Grow buffer in KV_CACHE_STEP chunks up to max_size.
            let need_grow = match &self.keys_codes {
                None => true,
                Some(kc) => {
                    let buf_size = kc.shape()[2] as usize;
                    prev >= buf_size && buf_size < self.max_size
                }
            };
            if need_grow {
                let b = k_codes.shape()[0];
                let h_k = k_codes.shape()[1];
                let d_k = k_codes.shape()[3];
                let h_v = v_codes.shape()[1];
                let d_v = v_codes.shape()[3];

                let new_size = std::cmp::min(KV_CACHE_STEP, self.max_size - prev) as i32;

                let new_kc = mlx_rs::ops::zeros_dtype(&[b, h_k, new_size, d_k], k_codes.dtype())
                    .context("TurboQuant: zeros k_codes grow")?;
                let new_ks = mlx_rs::ops::zeros_dtype(&[b, h_k, new_size, 1], k_sigma.dtype())
                    .context("TurboQuant: zeros k_sigma grow")?;
                let new_vc = mlx_rs::ops::zeros_dtype(&[b, h_v, new_size, d_v], v_codes.dtype())
                    .context("TurboQuant: zeros v_codes grow")?;
                let new_vs = mlx_rs::ops::zeros_dtype(&[b, h_v, new_size, 1], v_sigma.dtype())
                    .context("TurboQuant: zeros v_sigma grow")?;

                self.keys_codes = Some(match self.keys_codes.take() {
                    None => new_kc,
                    Some(old) => mlx_rs::ops::concatenate_axis(&[&old, &new_kc], 2)
                        .context("TurboQuant: concat k_codes grow block")?,
                });
                self.keys_sigma = Some(match self.keys_sigma.take() {
                    None => new_ks,
                    Some(old) => mlx_rs::ops::concatenate_axis(&[&old, &new_ks], 2)
                        .context("TurboQuant: concat k_sigma grow block")?,
                });
                self.values_codes = Some(match self.values_codes.take() {
                    None => new_vc,
                    Some(old) => mlx_rs::ops::concatenate_axis(&[&old, &new_vc], 2)
                        .context("TurboQuant: concat v_codes grow block")?,
                });
                self.values_sigma = Some(match self.values_sigma.take() {
                    None => new_vs,
                    Some(old) => mlx_rs::ops::concatenate_axis(&[&old, &new_vs], 2)
                        .context("TurboQuant: concat v_sigma grow block")?,
                });
                self.idx = prev;
            }

            // ── 2. One-time trim if prefill stored > max_size.
            let buf_size = self.keys_codes.as_ref().unwrap().shape()[2] as usize;
            if buf_size > self.max_size {
                let trim_size = buf_size - self.max_size;
                let keep_i = self.keep as i32;
                let trim_keep = (trim_size + self.keep) as i32;
                let buf_i = buf_size as i32;
                let kc = self.keys_codes.take().unwrap();
                let ks = self.keys_sigma.take().unwrap();
                let vc = self.values_codes.take().unwrap();
                let vs = self.values_sigma.take().unwrap();
                self.keys_codes = Some(
                    mlx_rs::ops::concatenate_axis(
                        &[
                            &slice_axis2(&kc, 0, keep_i)?,
                            &slice_axis2(&kc, trim_keep, buf_i)?,
                        ],
                        2,
                    )
                    .context("TurboQuant: trim k_codes")?,
                );
                self.keys_sigma = Some(
                    mlx_rs::ops::concatenate_axis(
                        &[
                            &slice_axis2(&ks, 0, keep_i)?,
                            &slice_axis2(&ks, trim_keep, buf_i)?,
                        ],
                        2,
                    )
                    .context("TurboQuant: trim k_sigma")?,
                );
                self.values_codes = Some(
                    mlx_rs::ops::concatenate_axis(
                        &[
                            &slice_axis2(&vc, 0, keep_i)?,
                            &slice_axis2(&vc, trim_keep, buf_i)?,
                        ],
                        2,
                    )
                    .context("TurboQuant: trim v_codes")?,
                );
                self.values_sigma = Some(
                    mlx_rs::ops::concatenate_axis(
                        &[
                            &slice_axis2(&vs, 0, keep_i)?,
                            &slice_axis2(&vs, trim_keep, buf_i)?,
                        ],
                        2,
                    )
                    .context("TurboQuant: trim v_sigma")?,
                );
                self.idx = self.max_size;
            }

            // ── 3. Rotate.
            if self.idx == self.max_size {
                self.idx = self.keep;
            }

            // ── 4. slice_update at idx across 4 buffers.
            let write_idx = self.idx;
            let new_idx = write_idx + s;
            self.keys_codes
                .as_mut()
                .expect("k_codes present after grow")
                .try_index_mut((.., .., (write_idx as i32)..(new_idx as i32), ..), k_codes)
                .context("TurboQuant: slice_update k_codes")?;
            self.keys_sigma
                .as_mut()
                .expect("k_sigma present after grow")
                .try_index_mut((.., .., (write_idx as i32)..(new_idx as i32), ..), k_sigma)
                .context("TurboQuant: slice_update k_sigma")?;
            self.values_codes
                .as_mut()
                .expect("v_codes present after grow")
                .try_index_mut((.., .., (write_idx as i32)..(new_idx as i32), ..), v_codes)
                .context("TurboQuant: slice_update v_codes")?;
            self.values_sigma
                .as_mut()
                .expect("v_sigma present after grow")
                .try_index_mut((.., .., (write_idx as i32)..(new_idx as i32), ..), v_sigma)
                .context("TurboQuant: slice_update v_sigma")?;

            self.idx = new_idx;
            self.offset = prev + s;

            // ── 5. Return: full ring if wrapped; partial view otherwise.
            if self.offset >= self.max_size {
                let kc = self.keys_codes.as_ref().unwrap().clone();
                let ks = self.keys_sigma.as_ref().unwrap().clone();
                let vc = self.values_codes.as_ref().unwrap().clone();
                let vs = self.values_sigma.as_ref().unwrap().clone();
                Ok(((kc, ks), (vc, vs)))
            } else {
                let off_i = self.offset as i32;
                let kc = slice_axis2(self.keys_codes.as_ref().unwrap(), 0, off_i)?;
                let ks = slice_axis2(self.keys_sigma.as_ref().unwrap(), 0, off_i)?;
                let vc = slice_axis2(self.values_codes.as_ref().unwrap(), 0, off_i)?;
                let vs = slice_axis2(self.values_sigma.as_ref().unwrap(), 0, off_i)?;
                Ok(((kc, ks), (vc, vs)))
            }
        }

        /// QJL Stage-2 variant of `update_and_fetch`. Mirrors the same
        /// ring-buffer logic but also writes / fetches `k_signs` (bf16
        /// `[..., m]`) and `k_residual_norm` (f32 `[..., 1]`) in lockstep
        /// with the four Stage-1 slots. `with_qjl(m)` must have been called.
        ///
        /// Returns
        ///   `((kc, ks, k_signs_view, k_rnorm_view), (vc, vs))`
        /// — Stage-1 K + V pairs identical to `update_and_fetch`, plus the
        /// in-ring views of the QJL Stage-2 K extras.
        pub fn update_and_fetch_qjl(
            &mut self,
            k_codes: &Array,
            k_sigma: &Array,
            k_signs: &Array,
            k_rnorm: &Array,
            v_codes: &Array,
            v_sigma: &Array,
        ) -> Result<((Array, Array, Array, Array), (Array, Array))> {
            if self.qjl_m.is_none() {
                return Err(anyhow!(
                    "TurboQuant cache: update_and_fetch_qjl called but qjl_m \
                     is not configured. Construct with `.with_qjl(m)`."
                ));
            }
            if k_codes.ndim() != 4 || v_codes.ndim() != 4 || k_signs.ndim() != 4 {
                return Err(anyhow!(
                    "TurboQuant cache (qjl): expected 4D inputs, got \
                     K_codes={}D, V_codes={}D, K_signs={}D",
                    k_codes.ndim(),
                    v_codes.ndim(),
                    k_signs.ndim()
                ));
            }
            let s = k_codes.shape()[2] as usize;
            if v_codes.shape()[2] as usize != s || k_signs.shape()[2] as usize != s {
                return Err(anyhow!(
                    "TurboQuant cache (qjl): seq-len mismatch — K_codes={}, \
                     V_codes={}, K_signs={}",
                    s,
                    v_codes.shape()[2],
                    k_signs.shape()[2]
                ));
            }

            if s == 1 && use_step_prealloc() {
                return self
                    .update_in_place_tq_qjl(k_codes, k_sigma, k_signs, k_rnorm, v_codes, v_sigma);
            }

            // Legacy concat-with-trim path mirroring update_and_fetch but
            // operating on six buffers.
            let (new_k_codes, new_k_sigma, new_k_signs, new_k_rnorm, new_v_codes, new_v_sigma) =
                match (
                    self.keys_codes.take(),
                    self.keys_sigma.take(),
                    self.keys_signs.take(),
                    self.keys_residual_norm.take(),
                    self.values_codes.take(),
                    self.values_sigma.take(),
                ) {
                    (None, None, None, None, None, None) => (
                        k_codes.clone(),
                        k_sigma.clone(),
                        k_signs.clone(),
                        k_rnorm.clone(),
                        v_codes.clone(),
                        v_sigma.clone(),
                    ),
                    (Some(kc), Some(ks), Some(ksg), Some(krn), Some(vc), Some(vs)) => {
                        let cached = kc.shape()[2] as usize;
                        let trim_size = (cached + s).saturating_sub(self.max_size);
                        let cached_i = cached as i32;
                        let keep_i = self.keep as i32;
                        let trim_keep = (trim_size + self.keep) as i32;
                        if trim_size > 0 {
                            let head_kc = slice_axis2(&kc, 0, keep_i)?;
                            let tail_kc = slice_axis2(&kc, trim_keep, cached_i)?;
                            let head_ks = slice_axis2(&ks, 0, keep_i)?;
                            let tail_ks = slice_axis2(&ks, trim_keep, cached_i)?;
                            let head_ksg = slice_axis2(&ksg, 0, keep_i)?;
                            let tail_ksg = slice_axis2(&ksg, trim_keep, cached_i)?;
                            let head_krn = slice_axis2(&krn, 0, keep_i)?;
                            let tail_krn = slice_axis2(&krn, trim_keep, cached_i)?;
                            let head_vc = slice_axis2(&vc, 0, keep_i)?;
                            let tail_vc = slice_axis2(&vc, trim_keep, cached_i)?;
                            let head_vs = slice_axis2(&vs, 0, keep_i)?;
                            let tail_vs = slice_axis2(&vs, trim_keep, cached_i)?;
                            (
                                mlx_rs::ops::concatenate_axis(&[&head_kc, &tail_kc, k_codes], 2)
                                    .context("TurboQuant(qjl): concat k_codes after trim")?,
                                mlx_rs::ops::concatenate_axis(&[&head_ks, &tail_ks, k_sigma], 2)
                                    .context("TurboQuant(qjl): concat k_sigma after trim")?,
                                mlx_rs::ops::concatenate_axis(&[&head_ksg, &tail_ksg, k_signs], 2)
                                    .context("TurboQuant(qjl): concat k_signs after trim")?,
                                mlx_rs::ops::concatenate_axis(&[&head_krn, &tail_krn, k_rnorm], 2)
                                    .context("TurboQuant(qjl): concat k_rnorm after trim")?,
                                mlx_rs::ops::concatenate_axis(&[&head_vc, &tail_vc, v_codes], 2)
                                    .context("TurboQuant(qjl): concat v_codes after trim")?,
                                mlx_rs::ops::concatenate_axis(&[&head_vs, &tail_vs, v_sigma], 2)
                                    .context("TurboQuant(qjl): concat v_sigma after trim")?,
                            )
                        } else {
                            (
                                mlx_rs::ops::concatenate_axis(&[&kc, k_codes], 2)
                                    .context("TurboQuant(qjl): concat k_codes (no trim)")?,
                                mlx_rs::ops::concatenate_axis(&[&ks, k_sigma], 2)
                                    .context("TurboQuant(qjl): concat k_sigma (no trim)")?,
                                mlx_rs::ops::concatenate_axis(&[&ksg, k_signs], 2)
                                    .context("TurboQuant(qjl): concat k_signs (no trim)")?,
                                mlx_rs::ops::concatenate_axis(&[&krn, k_rnorm], 2)
                                    .context("TurboQuant(qjl): concat k_rnorm (no trim)")?,
                                mlx_rs::ops::concatenate_axis(&[&vc, v_codes], 2)
                                    .context("TurboQuant(qjl): concat v_codes (no trim)")?,
                                mlx_rs::ops::concatenate_axis(&[&vs, v_sigma], 2)
                                    .context("TurboQuant(qjl): concat v_sigma (no trim)")?,
                            )
                        }
                    }
                    _ => {
                        unreachable!("TurboQuant cache (qjl): all six fields must move in lockstep")
                    }
                };
            self.offset += s;
            self.idx = new_k_codes.shape()[2] as usize;
            self.keys_codes = Some(new_k_codes.clone());
            self.keys_sigma = Some(new_k_sigma.clone());
            self.keys_signs = Some(new_k_signs.clone());
            self.keys_residual_norm = Some(new_k_rnorm.clone());
            self.values_codes = Some(new_v_codes.clone());
            self.values_sigma = Some(new_v_sigma.clone());
            Ok((
                (new_k_codes, new_k_sigma, new_k_signs, new_k_rnorm),
                (new_v_codes, new_v_sigma),
            ))
        }

        fn update_in_place_tq_qjl(
            &mut self,
            k_codes: &Array,
            k_sigma: &Array,
            k_signs: &Array,
            k_rnorm: &Array,
            v_codes: &Array,
            v_sigma: &Array,
        ) -> Result<((Array, Array, Array, Array), (Array, Array))> {
            let s = 1usize;
            let prev = self.offset;

            // ── 1. Grow buffer in KV_CACHE_STEP chunks up to max_size.
            let need_grow = match &self.keys_codes {
                None => true,
                Some(kc) => {
                    let buf_size = kc.shape()[2] as usize;
                    prev >= buf_size && buf_size < self.max_size
                }
            };
            if need_grow {
                let b = k_codes.shape()[0];
                let h_k = k_codes.shape()[1];
                let d_k = k_codes.shape()[3];
                let m_k = k_signs.shape()[3];
                let h_v = v_codes.shape()[1];
                let d_v = v_codes.shape()[3];

                let new_size = std::cmp::min(KV_CACHE_STEP, self.max_size - prev) as i32;

                let new_kc = mlx_rs::ops::zeros_dtype(&[b, h_k, new_size, d_k], k_codes.dtype())
                    .context("TurboQuant(qjl): zeros k_codes grow")?;
                let new_ks = mlx_rs::ops::zeros_dtype(&[b, h_k, new_size, 1], k_sigma.dtype())
                    .context("TurboQuant(qjl): zeros k_sigma grow")?;
                let new_ksg = mlx_rs::ops::zeros_dtype(&[b, h_k, new_size, m_k], k_signs.dtype())
                    .context("TurboQuant(qjl): zeros k_signs grow")?;
                let new_krn = mlx_rs::ops::zeros_dtype(&[b, h_k, new_size, 1], k_rnorm.dtype())
                    .context("TurboQuant(qjl): zeros k_rnorm grow")?;
                let new_vc = mlx_rs::ops::zeros_dtype(&[b, h_v, new_size, d_v], v_codes.dtype())
                    .context("TurboQuant(qjl): zeros v_codes grow")?;
                let new_vs = mlx_rs::ops::zeros_dtype(&[b, h_v, new_size, 1], v_sigma.dtype())
                    .context("TurboQuant(qjl): zeros v_sigma grow")?;

                self.keys_codes = Some(match self.keys_codes.take() {
                    None => new_kc,
                    Some(old) => mlx_rs::ops::concatenate_axis(&[&old, &new_kc], 2)
                        .context("TurboQuant(qjl): concat k_codes grow")?,
                });
                self.keys_sigma = Some(match self.keys_sigma.take() {
                    None => new_ks,
                    Some(old) => mlx_rs::ops::concatenate_axis(&[&old, &new_ks], 2)
                        .context("TurboQuant(qjl): concat k_sigma grow")?,
                });
                self.keys_signs = Some(match self.keys_signs.take() {
                    None => new_ksg,
                    Some(old) => mlx_rs::ops::concatenate_axis(&[&old, &new_ksg], 2)
                        .context("TurboQuant(qjl): concat k_signs grow")?,
                });
                self.keys_residual_norm = Some(match self.keys_residual_norm.take() {
                    None => new_krn,
                    Some(old) => mlx_rs::ops::concatenate_axis(&[&old, &new_krn], 2)
                        .context("TurboQuant(qjl): concat k_rnorm grow")?,
                });
                self.values_codes = Some(match self.values_codes.take() {
                    None => new_vc,
                    Some(old) => mlx_rs::ops::concatenate_axis(&[&old, &new_vc], 2)
                        .context("TurboQuant(qjl): concat v_codes grow")?,
                });
                self.values_sigma = Some(match self.values_sigma.take() {
                    None => new_vs,
                    Some(old) => mlx_rs::ops::concatenate_axis(&[&old, &new_vs], 2)
                        .context("TurboQuant(qjl): concat v_sigma grow")?,
                });
                self.idx = prev;
            }

            // ── 2. One-time trim if prefill stored > max_size.
            let buf_size = self.keys_codes.as_ref().unwrap().shape()[2] as usize;
            if buf_size > self.max_size {
                let trim_size = buf_size - self.max_size;
                let keep_i = self.keep as i32;
                let trim_keep = (trim_size + self.keep) as i32;
                let buf_i = buf_size as i32;

                let kc = self.keys_codes.take().unwrap();
                let ks = self.keys_sigma.take().unwrap();
                let ksg = self.keys_signs.take().unwrap();
                let krn = self.keys_residual_norm.take().unwrap();
                let vc = self.values_codes.take().unwrap();
                let vs = self.values_sigma.take().unwrap();

                self.keys_codes = Some(
                    mlx_rs::ops::concatenate_axis(
                        &[
                            &slice_axis2(&kc, 0, keep_i)?,
                            &slice_axis2(&kc, trim_keep, buf_i)?,
                        ],
                        2,
                    )
                    .context("TurboQuant(qjl): trim k_codes")?,
                );
                self.keys_sigma = Some(
                    mlx_rs::ops::concatenate_axis(
                        &[
                            &slice_axis2(&ks, 0, keep_i)?,
                            &slice_axis2(&ks, trim_keep, buf_i)?,
                        ],
                        2,
                    )
                    .context("TurboQuant(qjl): trim k_sigma")?,
                );
                self.keys_signs = Some(
                    mlx_rs::ops::concatenate_axis(
                        &[
                            &slice_axis2(&ksg, 0, keep_i)?,
                            &slice_axis2(&ksg, trim_keep, buf_i)?,
                        ],
                        2,
                    )
                    .context("TurboQuant(qjl): trim k_signs")?,
                );
                self.keys_residual_norm = Some(
                    mlx_rs::ops::concatenate_axis(
                        &[
                            &slice_axis2(&krn, 0, keep_i)?,
                            &slice_axis2(&krn, trim_keep, buf_i)?,
                        ],
                        2,
                    )
                    .context("TurboQuant(qjl): trim k_rnorm")?,
                );
                self.values_codes = Some(
                    mlx_rs::ops::concatenate_axis(
                        &[
                            &slice_axis2(&vc, 0, keep_i)?,
                            &slice_axis2(&vc, trim_keep, buf_i)?,
                        ],
                        2,
                    )
                    .context("TurboQuant(qjl): trim v_codes")?,
                );
                self.values_sigma = Some(
                    mlx_rs::ops::concatenate_axis(
                        &[
                            &slice_axis2(&vs, 0, keep_i)?,
                            &slice_axis2(&vs, trim_keep, buf_i)?,
                        ],
                        2,
                    )
                    .context("TurboQuant(qjl): trim v_sigma")?,
                );
                self.idx = self.max_size;
            }

            // ── 3. Rotate at end of ring.
            if self.idx == self.max_size {
                self.idx = self.keep;
            }

            // ── 4. slice_update at idx across 6 buffers.
            let write_idx = self.idx;
            let new_idx = write_idx + s;
            let wi = write_idx as i32;
            let ni = new_idx as i32;
            self.keys_codes
                .as_mut()
                .expect("k_codes")
                .try_index_mut((.., .., wi..ni, ..), k_codes)
                .context("TurboQuant(qjl): slice_update k_codes")?;
            self.keys_sigma
                .as_mut()
                .expect("k_sigma")
                .try_index_mut((.., .., wi..ni, ..), k_sigma)
                .context("TurboQuant(qjl): slice_update k_sigma")?;
            self.keys_signs
                .as_mut()
                .expect("k_signs")
                .try_index_mut((.., .., wi..ni, ..), k_signs)
                .context("TurboQuant(qjl): slice_update k_signs")?;
            self.keys_residual_norm
                .as_mut()
                .expect("k_rnorm")
                .try_index_mut((.., .., wi..ni, ..), k_rnorm)
                .context("TurboQuant(qjl): slice_update k_rnorm")?;
            self.values_codes
                .as_mut()
                .expect("v_codes")
                .try_index_mut((.., .., wi..ni, ..), v_codes)
                .context("TurboQuant(qjl): slice_update v_codes")?;
            self.values_sigma
                .as_mut()
                .expect("v_sigma")
                .try_index_mut((.., .., wi..ni, ..), v_sigma)
                .context("TurboQuant(qjl): slice_update v_sigma")?;

            self.idx = new_idx;
            self.offset = prev + s;

            // ── 5. Return: full ring if wrapped; partial view otherwise.
            if self.offset >= self.max_size {
                let kc = self.keys_codes.as_ref().unwrap().clone();
                let ks = self.keys_sigma.as_ref().unwrap().clone();
                let ksg = self.keys_signs.as_ref().unwrap().clone();
                let krn = self.keys_residual_norm.as_ref().unwrap().clone();
                let vc = self.values_codes.as_ref().unwrap().clone();
                let vs = self.values_sigma.as_ref().unwrap().clone();
                Ok(((kc, ks, ksg, krn), (vc, vs)))
            } else {
                let off_i = self.offset as i32;
                let kc = slice_axis2(self.keys_codes.as_ref().unwrap(), 0, off_i)?;
                let ks = slice_axis2(self.keys_sigma.as_ref().unwrap(), 0, off_i)?;
                let ksg = slice_axis2(self.keys_signs.as_ref().unwrap(), 0, off_i)?;
                let krn = slice_axis2(self.keys_residual_norm.as_ref().unwrap(), 0, off_i)?;
                let vc = slice_axis2(self.values_codes.as_ref().unwrap(), 0, off_i)?;
                let vs = slice_axis2(self.values_sigma.as_ref().unwrap(), 0, off_i)?;
                Ok(((kc, ks, ksg, krn), (vc, vs)))
            }
        }
    }

    impl Default for NativeRotatingKvCacheTurboQuant {
        fn default() -> Self {
            Self::new(1024, 0, 4)
        }
    }

    /// Generic per-layer state slots — used by linear-attn layers to hold
    /// `[conv_state, ssm_state]`.
    pub struct NativeArraysCache {
        cache: Vec<Option<Array>>,
        offset: usize,
        lengths: Option<Array>,
        left_padding: Option<Array>,
    }

    impl NativeArraysCache {
        pub fn new(size: usize) -> Self {
            Self {
                cache: (0..size).map(|_| None).collect(),
                offset: 0,
                lengths: None,
                left_padding: None,
            }
        }

        pub fn len(&self) -> usize {
            self.cache.len()
        }

        pub fn is_empty(&self) -> bool {
            self.cache.is_empty()
        }

        pub fn offset(&self) -> usize {
            self.offset
        }

        pub fn empty_slots(&self) -> bool {
            self.cache.iter().all(|c| c.is_none())
        }

        pub fn get(&self, idx: usize) -> Option<&Array> {
            self.cache.get(idx).and_then(|slot| slot.as_ref())
        }

        pub fn set(&mut self, idx: usize, value: Array) -> Result<()> {
            let len = self.cache.len();
            *self.cache.get_mut(idx).ok_or_else(|| {
                anyhow!("NativeArraysCache::set: idx {idx} out of range (len={len})")
            })? = Some(value);
            Ok(())
        }

        pub fn lengths(&self) -> Option<&Array> {
            self.lengths.as_ref()
        }

        pub fn set_lengths(&mut self, lengths: Option<Array>) {
            self.lengths = lengths;
        }

        pub fn left_padding(&self) -> Option<&Array> {
            self.left_padding.as_ref()
        }

        pub fn set_left_padding(&mut self, lp: Option<Array>) {
            self.left_padding = lp;
        }

        /// Bump the consumer-managed offset by `n` tokens. Mirrors
        /// `ArraysCache.advance(N)` in mlx_lm.
        pub fn advance(&mut self, n: usize) {
            self.offset += n;
        }

        pub fn clear(&mut self) {
            for slot in &mut self.cache {
                *slot = None;
            }
            self.offset = 0;
            self.lengths = None;
            self.left_padding = None;
        }

        /// Iterate slots (cloning Array refs — refcount bump). Used by
        /// `native_snapshot` shallow capture.
        pub fn slots(&self) -> &[Option<Array>] {
            &self.cache
        }

        /// Replace slot at `idx` with `value` (or clear via `None`). Mirrors
        /// `set` but admits `None` for the snapshot-restore path where
        /// captured state may include empty slots.
        pub fn set_slot(&mut self, idx: usize, value: Option<Array>) -> Result<()> {
            let len = self.cache.len();
            *self.cache.get_mut(idx).ok_or_else(|| {
                anyhow!("NativeArraysCache::set_slot: idx {idx} out of range (len={len})")
            })? = value;
            Ok(())
        }

        /// Reassign offset. Used by snapshot restore.
        pub fn set_offset(&mut self, offset: usize) {
            self.offset = offset;
        }
    }

    /// Per-layer cache. Discriminated by the model's static `is_linear` flag.
    pub enum NativeLayerCache {
        Full(NativeKvCache),
        Linear(NativeArraysCache),
        /// Full-attention layer whose KV is TurboQuant-compressed (Lloyd-Max
        /// Stage-1, optional QJL Stage-2). Reuses the rotating cache with
        /// `keep == 0` + `max_size == max_position_embeddings`, so rotation
        /// never fires and it behaves as a compressed append-only buffer.
        /// Opt-in via `LUMEN_QWEN35_TQ_KV` — cuts the growing full-attn KV
        /// footprint ~2-4× at the cost of dequant-on-read.
        FullTurboquant(NativeRotatingKvCacheTurboQuant),
    }

    impl NativeLayerCache {
        /// Token offset (number of tokens fed so far). For full-attn this is
        /// the KV-history length; for linear-attn it's the consumer-bumped
        /// counter.
        pub fn offset(&self) -> usize {
            match self {
                NativeLayerCache::Full(c) => c.offset(),
                NativeLayerCache::Linear(c) => c.offset(),
                NativeLayerCache::FullTurboquant(c) => c.offset(),
            }
        }

        pub fn as_full_mut(&mut self) -> Result<&mut NativeKvCache> {
            match self {
                NativeLayerCache::Full(c) => Ok(c),
                _ => Err(anyhow!(
                    "NativeLayerCache::as_full_mut called on a non-Full layer"
                )),
            }
        }

        pub fn as_linear_mut(&mut self) -> Result<&mut NativeArraysCache> {
            match self {
                NativeLayerCache::Linear(c) => Ok(c),
                _ => Err(anyhow!(
                    "NativeLayerCache::as_linear_mut called on a non-Linear layer"
                )),
            }
        }

        /// Mutable access to the TurboQuant-compressed full-attn cache.
        pub fn as_full_turboquant_mut(&mut self) -> Result<&mut NativeRotatingKvCacheTurboQuant> {
            match self {
                NativeLayerCache::FullTurboquant(c) => Ok(c),
                _ => Err(anyhow!(
                    "NativeLayerCache::as_full_turboquant_mut called on a non-FullTurboquant layer"
                )),
            }
        }

        pub fn as_full(&self) -> Option<&NativeKvCache> {
            match self {
                NativeLayerCache::Full(c) => Some(c),
                _ => None,
            }
        }

        pub fn as_linear(&self) -> Option<&NativeArraysCache> {
            match self {
                NativeLayerCache::Linear(c) => Some(c),
                _ => None,
            }
        }

        pub fn clear(&mut self) {
            match self {
                NativeLayerCache::Full(c) => c.clear(),
                NativeLayerCache::Linear(c) => c.clear(),
                NativeLayerCache::FullTurboquant(c) => c.clear(),
            }
        }
    }

    /// Single-copy shared-prefix KV (Phase 4 in-batch dedup). Holds, per model
    /// layer, the frozen prefix keys/values for FULL-ATTENTION layers — stored
    /// ONCE and referenced by every sequence in a batch that shares this prompt
    /// prefix, instead of each sequence carrying its own copy. Linear/Mamba
    /// layers carry `None`: their recurrent state is per-seq and O(1)-sized, so
    /// there is nothing to dedup (each seq keeps its own already-correct state).
    pub struct SharedPrefixKv {
        /// Number of shared prefix tokens (absolute positions `0..prefix_len`).
        pub prefix_len: usize,
        /// Per layer (indexed by `layer_idx`): `Some((k, v))` with shape
        /// `[1, n_kv_heads, prefix_len, head_dim]` for full-attn layers, `None`
        /// for linear/Mamba/TurboQuant layers.
        pub layers: Vec<Option<(Array, Array)>>,
    }

    /// One cache list per sequence — mirrors mlx-lm's `model.make_cache()`
    /// returning a `List[Cache]` of length `num_hidden_layers`.
    pub struct NativePromptCache {
        layers: Vec<NativeLayerCache>,
    }

    impl NativePromptCache {
        /// Build a fresh cache list given each layer's `is_linear` flag and
        /// the SSM state-slot count (Qwen3.5-MoE = 2: conv_state + ssm_state).
        pub fn new(is_linear_per_layer: &[bool], ssm_slots: usize) -> Self {
            let layers = is_linear_per_layer
                .iter()
                .map(|&is_linear| {
                    if is_linear {
                        NativeLayerCache::Linear(NativeArraysCache::new(ssm_slots))
                    } else {
                        NativeLayerCache::Full(NativeKvCache::new())
                    }
                })
                .collect();
            Self { layers }
        }

        /// Like [`NativePromptCache::new`] but full-attention layers are
        /// TurboQuant-compressed when `tq_enabled`. Linear-attn layers are
        /// unaffected (their SSM state is already compact). The TQ cache uses
        /// `keep = 0` + `max_size = max_position_embeddings` so it never rotates
        /// — a compressed append-only buffer. `tq_bits` is the Lloyd-Max bit
        /// width (8 = sanity, 6 = lowest-clean, 4 = aggressive/lossy).
        pub fn new_with_tq(
            is_linear_per_layer: &[bool],
            ssm_slots: usize,
            tq_enabled: bool,
            max_position_embeddings: usize,
            tq_bits: u32,
        ) -> Self {
            if !tq_enabled {
                return Self::new(is_linear_per_layer, ssm_slots);
            }
            let max_ctx = max_position_embeddings.max(1);
            let layers = is_linear_per_layer
                .iter()
                .map(|&is_linear| {
                    if is_linear {
                        NativeLayerCache::Linear(NativeArraysCache::new(ssm_slots))
                    } else {
                        NativeLayerCache::FullTurboquant(NativeRotatingKvCacheTurboQuant::new(
                            max_ctx, 0, tq_bits,
                        ))
                    }
                })
                .collect();
            Self { layers }
        }

        pub fn len(&self) -> usize {
            self.layers.len()
        }

        pub fn is_empty(&self) -> bool {
            self.layers.is_empty()
        }

        pub fn layer(&self, idx: usize) -> Option<&NativeLayerCache> {
            self.layers.get(idx)
        }

        pub fn layer_mut(&mut self, idx: usize) -> Option<&mut NativeLayerCache> {
            self.layers.get_mut(idx)
        }

        pub fn layers(&self) -> &[NativeLayerCache] {
            &self.layers
        }

        pub fn layers_mut(&mut self) -> &mut [NativeLayerCache] {
            &mut self.layers
        }

        /// Phase 4: capture the first `prefix_len` tokens of every full-attn
        /// layer as a [`SharedPrefixKv`]. Call on a fully-prefilled DONOR cache
        /// (`base_offset == 0`); the result is shared by all sequences in the
        /// batch that have this prompt prefix. The captured Arrays are eval'd so
        /// they are independent of the donor's subsequent mutation/eviction.
        pub fn extract_shared_prefix(&self, prefix_len: usize) -> Result<SharedPrefixKv> {
            let mut layers = Vec::with_capacity(self.layers.len());
            for layer in &self.layers {
                match layer {
                    NativeLayerCache::Full(kv) => {
                        let (k, v) = kv.prefix_kv(prefix_len)?;
                        // Force materialization so the shared buffer doesn't
                        // alias (and pin) the donor seq's growing cache.
                        k.eval().context("extract_shared_prefix: eval k")?;
                        v.eval().context("extract_shared_prefix: eval v")?;
                        layers.push(Some((k, v)));
                    }
                    _ => layers.push(None),
                }
            }
            Ok(SharedPrefixKv { prefix_len, layers })
        }

        /// Phase 4: drop the first `prefix_len` tokens from every full-attn
        /// layer (they now live in a single [`SharedPrefixKv`]); each layer
        /// keeps only its divergent suffix with `base_offset = prefix_len`.
        /// Linear/TurboQuant layers are untouched (their state stays per-seq
        /// and is not deduped). Idempotent.
        pub fn keep_suffix_from(&mut self, prefix_len: usize) -> Result<()> {
            for layer in &mut self.layers {
                if let NativeLayerCache::Full(kv) = layer {
                    kv.keep_suffix_from(prefix_len)?;
                }
            }
            Ok(())
        }

        /// First full-attn layer's `base_offset` (the shared-prefix length this
        /// cache is currently attached to, or 0 if unshared).
        pub fn shared_prefix_len(&self) -> usize {
            for layer in &self.layers {
                if let NativeLayerCache::Full(kv) = layer {
                    return kv.base_offset();
                }
            }
            0
        }

        /// Token offset of the first full-attn layer — convenient for the
        /// decode loop's RoPE offset (all full-attn layers share the same
        /// offset because they're updated in lockstep).
        pub fn full_attn_offset(&self) -> usize {
            self.layers
                .iter()
                .find_map(|l| l.as_full().map(|c| c.offset()))
                .unwrap_or(0)
        }

        pub fn clear(&mut self) {
            for layer in &mut self.layers {
                layer.clear();
            }
        }

        /// Create a fresh empty `NativePromptCache` mirroring this one's layer
        /// kinds (Full/Linear) and slot widths. Used by snapshot fork —
        /// `native_snapshot::PromptCacheSnapshot::install_into_fresh` writes
        /// cloned state into the result of this method.
        pub fn clone_structure_only(&self) -> Self {
            let layers = self
                .layers
                .iter()
                .map(|l| match l {
                    NativeLayerCache::Full(_) => NativeLayerCache::Full(NativeKvCache::new()),
                    NativeLayerCache::Linear(c) => {
                        NativeLayerCache::Linear(NativeArraysCache::new(c.len()))
                    }
                    NativeLayerCache::FullTurboquant(c) => {
                        NativeLayerCache::FullTurboquant(c.structure_clone())
                    }
                })
                .collect();
            Self { layers }
        }
    }
}

#[cfg(feature = "mlx-native")]
#[allow(unused_imports)] // Consumed by Phase 3d decode loop in runner_native.rs.
pub(crate) use imp::{
    NativeArraysCache, NativeKvCache, NativeKvCacheQuantized, NativeLayerCache, NativePromptCache,
    NativeRotatingKvCache, NativeRotatingKvCacheQuantized, NativeRotatingKvCacheTurboQuant,
    SharedPrefixKv,
};

// exercise concat / set-get / advance / construction
// via the real MLX FFI. Same `#[ignore]` policy as the rest of the native parity
// tests (Issue #5: sandbox lacks Metal device).
//
//   cargo test -p lumen-mlx --features mlx-native \
//       native_cache::lifecycle_tests:: -- --ignored
#[cfg(all(test, feature = "mlx-native"))]
mod lifecycle_tests {

    use super::imp::{
        NativeArraysCache, NativeKvCache, NativeLayerCache, NativePromptCache,
        NativeRotatingKvCache,
    };
    use mlx_rs::Array;

    fn make_kv_block(b: i32, h: i32, s: i32, d: i32, fill: f32) -> Array {
        let n = (b * h * s * d) as usize;
        let data: Vec<f32> = (0..n).map(|i| fill + i as f32 * 0.001).collect();
        Array::from_slice(&data, &[b, h, s, d])
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn kv_cache_starts_empty() {
        let c = NativeKvCache::new();
        assert!(c.empty());
        assert_eq!(c.offset(), 0);
        assert!(c.keys().is_none());
        assert!(c.values().is_none());
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn kv_cache_first_update_stores_input() {
        let mut c = NativeKvCache::new();
        let k = make_kv_block(1, 2, 4, 8, 0.0);
        let v = make_kv_block(1, 2, 4, 8, 1.0);
        let (kf, vf) = c
            .update_and_fetch(&k, &v)
            .expect("first update_and_fetch must succeed");
        assert_eq!(c.offset(), 4);
        assert_eq!(kf.shape(), &[1, 2, 4, 8]);
        assert_eq!(vf.shape(), &[1, 2, 4, 8]);
        assert!(!c.empty());
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn kv_cache_subsequent_updates_concatenate() {
        let mut c = NativeKvCache::new();
        let k0 = make_kv_block(1, 2, 4, 8, 0.0);
        let v0 = make_kv_block(1, 2, 4, 8, 1.0);
        c.update_and_fetch(&k0, &v0).unwrap();

        let k1 = make_kv_block(1, 2, 1, 8, 2.0);
        let v1 = make_kv_block(1, 2, 1, 8, 3.0);
        let (kf, vf) = c.update_and_fetch(&k1, &v1).unwrap();
        assert_eq!(c.offset(), 5);
        assert_eq!(kf.shape(), &[1, 2, 5, 8]);
        assert_eq!(vf.shape(), &[1, 2, 5, 8]);

        // 3rd step: another single token decode → offset 6, T=6 in the slice.
        let k2 = make_kv_block(1, 2, 1, 8, 4.0);
        let v2 = make_kv_block(1, 2, 1, 8, 5.0);
        let (kf2, _) = c.update_and_fetch(&k2, &v2).unwrap();
        assert_eq!(c.offset(), 6);
        assert_eq!(kf2.shape(), &[1, 2, 6, 8]);
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn kv_cache_clear_resets_state() {
        let mut c = NativeKvCache::new();
        let k = make_kv_block(1, 2, 4, 8, 0.0);
        let v = make_kv_block(1, 2, 4, 8, 1.0);
        c.update_and_fetch(&k, &v).unwrap();
        assert_eq!(c.offset(), 4);
        c.clear();
        assert!(c.empty());
        assert_eq!(c.offset(), 0);
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn kv_cache_rejects_mismatched_seq_length() {
        let mut c = NativeKvCache::new();
        let k = make_kv_block(1, 2, 4, 8, 0.0);
        let v = make_kv_block(1, 2, 3, 8, 1.0);
        assert!(c.update_and_fetch(&k, &v).is_err());
    }

    // ───────────────────── NativeRotatingKvCache (Gemma 4) ─────────────────────

    #[test]
    fn rotating_cache_starts_empty_no_ffi() {
        // State-only checks that don't require the MLX runtime.
        let c = NativeRotatingKvCache::new(8, 0);
        assert!(c.empty());
        assert_eq!(c.offset(), 0);
        assert_eq!(c.max_size(), 8);
        assert_eq!(c.keep(), 0);
        assert_eq!(c.cached_len(), 0);
    }

    #[test]
    #[should_panic(expected = "keep")]
    fn rotating_cache_rejects_keep_ge_max_size() {
        let _ = NativeRotatingKvCache::new(4, 4);
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn rotating_cache_first_update_stores_input() {
        let mut c = NativeRotatingKvCache::new(8, 0);
        let k = make_kv_block(1, 2, 3, 8, 0.0);
        let v = make_kv_block(1, 2, 3, 8, 1.0);
        let (kf, _vf) = c
            .update_and_fetch(&k, &v)
            .expect("first update_and_fetch must succeed");
        assert_eq!(c.offset(), 3);
        assert_eq!(c.cached_len(), 3);
        assert_eq!(kf.shape(), &[1, 2, 3, 8]);
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn rotating_cache_growth_within_max_size() {
        // Run the whole sequence under BOTH sides of the step-prealloc switch.
        // The logical contract — how many tokens the cache holds and where the
        // next one lands — is identical; only the physical buffer differs.
        // This test used to assert the physical shape, which the default path
        // stopped matching when it began growing in blocks: `kf.shape()[2]` is
        // the buffer, not the fill. Being `#[ignore]`d, it failed unseen.
        for prealloc in [false, true] {
            super::imp::test_support::with_step_prealloc(prealloc, || {
                let mut c = NativeRotatingKvCache::new(8, 0);
                // Push 1 token at a time up to (max_size - 1) → no eviction yet.
                for step in 0..7 {
                    let k = make_kv_block(1, 2, 1, 8, step as f32);
                    let v = make_kv_block(1, 2, 1, 8, step as f32 + 100.0);
                    let (kf, _vf) = c.update_and_fetch(&k, &v).unwrap();
                    let expected_len = step + 1;
                    // The *fetched* K is the logical fill, and `offset()` is
                    // the running token count. Both are identical under either
                    // path — that is the contract SDPA depends on.
                    assert_eq!(
                        kf.shape()[2] as usize,
                        expected_len,
                        "prealloc={prealloc} step={step}: fetch is the logical fill"
                    );
                    assert_eq!(c.offset(), expected_len, "prealloc={prealloc} step={step}");
                    // `cached_len()` is by definition `keys.shape()[2]`, the
                    // *allocation*: legacy grows one token at a time, prealloc
                    // takes the whole window up front. The original test
                    // asserted these were equal, which has been false for the
                    // default path since prealloc landed — and `#[ignore]`d, it
                    // never said so. Only the invariant is shared.
                    let buf = c.cached_len();
                    assert!(
                        (expected_len..=8).contains(&buf),
                        "prealloc={prealloc} step={step}: allocation {buf} outside \
                         [{expected_len}, 8]"
                    );
                    // …and the fetched rows must be the pushed blocks, in
                    // order. Neither the shape nor the counters catch a block
                    // landing in the wrong slot, or padding leaking in from an
                    // untrimmed buffer — which is the whole risk of having two
                    // append strategies.
                    kf.eval().expect("eval fetched K");
                    let got = kf.as_slice::<f32>();
                    for j in 0..=step {
                        // `make_kv_block` fills element 0 of each block with
                        // the step number; element 0 of row j sits at j*d
                        // within each (b,h) plane, and b=h=1 here.
                        let first = got[j * 8];
                        assert!(
                            (first - j as f32).abs() < 1e-3,
                            "prealloc={prealloc} step={step}: row {j} holds {first}, \
                             expected the block pushed at step {j}"
                        );
                    }
                }
                // Next push hits max_size exactly — still no trim (cached 7,
                // trim_size = 7 - 8 + 1 = 0, append → cached 8).
                let k = make_kv_block(1, 2, 1, 8, 7.0);
                let v = make_kv_block(1, 2, 1, 8, 107.0);
                let (kf, _vf) = c.update_and_fetch(&k, &v).unwrap();
                assert_eq!(c.offset(), 8, "prealloc={prealloc}");
                assert_eq!(c.cached_len(), 8, "prealloc={prealloc}: buffer full");
                assert_eq!(kf.shape()[2], 8, "prealloc={prealloc}: full buffer");
            });
        }
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn rotating_cache_evicts_past_max_size() {
        let mut c = NativeRotatingKvCache::new(8, 0);
        // Fill to max_size.
        for step in 0..8 {
            let k = make_kv_block(1, 2, 1, 8, step as f32);
            let v = make_kv_block(1, 2, 1, 8, step as f32 + 100.0);
            c.update_and_fetch(&k, &v).unwrap();
        }
        assert_eq!(c.cached_len(), 8);
        // 9th push: cached=8 → trim_size = 8 - 8 + 1 = 1 → drop oldest, append.
        let k = make_kv_block(1, 2, 1, 8, 8.0);
        let v = make_kv_block(1, 2, 1, 8, 108.0);
        let (kf, _vf) = c.update_and_fetch(&k, &v).unwrap();
        assert_eq!(kf.shape()[2], 8, "cache should stay capped at max_size");
        assert_eq!(c.cached_len(), 8);
        assert_eq!(c.offset(), 9, "offset reflects total tokens ever pushed");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn rotating_cache_keep_prefix_preserved_under_eviction() {
        let mut c = NativeRotatingKvCache::new(4, 2);
        // Seed 2 keep-anchor tokens via a single S=2 push.
        let k0 = make_kv_block(1, 1, 2, 4, 0.0);
        let v0 = make_kv_block(1, 1, 2, 4, 100.0);
        c.update_and_fetch(&k0, &v0).unwrap();
        // Now push 3 more tokens — total would be 5, exceeds max_size=4.
        // trim_size = 2 - 4 + 1 = -1 → still <= 0 → no trim, append → cached=5.
        // Hmm: max_size + S - 1 = 4 + 3 - 1 = 6, so 5 is fine; mlx_lm matches.
        let k1 = make_kv_block(1, 1, 3, 4, 2.0);
        let v1 = make_kv_block(1, 1, 3, 4, 102.0);
        let (kf, _vf) = c.update_and_fetch(&k1, &v1).unwrap();
        // After this push: cached = 5 (2 keep + 3 new), under cap of max_size + S - 1 = 6.
        assert_eq!(kf.shape()[2], 5);
        assert_eq!(c.offset(), 5);

        // Push 1 more single token — cached=5, trim_size = 5 - 4 + 1 = 2.
        // After trim: keep first 2 (anchors) + drop next 2 + keep rest (last 1)
        // + append 1 = 4 total.
        let k2 = make_kv_block(1, 1, 1, 4, 99.0);
        let v2 = make_kv_block(1, 1, 1, 4, 199.0);
        let (kf2, _vf2) = c.update_and_fetch(&k2, &v2).unwrap();
        assert_eq!(kf2.shape()[2], 4, "evicted to max_size after trim");
        assert_eq!(c.cached_len(), 4);
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn rotating_cache_clear_resets_state() {
        let mut c = NativeRotatingKvCache::new(8, 0);
        let k = make_kv_block(1, 2, 3, 8, 0.0);
        let v = make_kv_block(1, 2, 3, 8, 1.0);
        c.update_and_fetch(&k, &v).unwrap();
        assert_eq!(c.offset(), 3);
        c.clear();
        assert!(c.empty());
        assert_eq!(c.offset(), 0);
        assert_eq!(c.cached_len(), 0);
    }

    #[test]
    fn rotating_cache_rejects_mismatched_seq_length_compile_time() {
        // Construction-time invariant ruled out via assert; the runtime
        // shape mismatch check requires MLX FFI and is covered by the
        // ignored test below.
        let c = NativeRotatingKvCache::new(8, 0);
        assert_eq!(c.max_size(), 8);
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn rotating_cache_rejects_mismatched_seq_length() {
        let mut c = NativeRotatingKvCache::new(8, 0);
        let k = make_kv_block(1, 2, 4, 8, 0.0);
        let v = make_kv_block(1, 2, 3, 8, 1.0);
        assert!(c.update_and_fetch(&k, &v).is_err());
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn arrays_cache_set_get_round_trip() {
        let mut c = NativeArraysCache::new(2);
        assert_eq!(c.len(), 2);
        assert!(c.empty_slots());
        assert!(c.get(0).is_none());

        let conv_state = make_kv_block(1, 1, 4, 8, 0.0); // shape doesn't matter
        c.set(0, conv_state).unwrap();
        assert!(!c.empty_slots());
        assert!(c.get(0).is_some());
        assert!(c.get(1).is_none());

        let ssm_state = make_kv_block(1, 2, 4, 8, 1.0);
        c.set(1, ssm_state).unwrap();
        assert!(c.get(0).is_some());
        assert!(c.get(1).is_some());
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn arrays_cache_set_out_of_range_errors() {
        let mut c = NativeArraysCache::new(2);
        let arr = make_kv_block(1, 1, 1, 1, 0.0);
        assert!(c.set(2, arr).is_err());
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn arrays_cache_advance_bumps_offset() {
        let mut c = NativeArraysCache::new(2);
        assert_eq!(c.offset(), 0);
        c.advance(7);
        assert_eq!(c.offset(), 7);
        c.advance(3);
        assert_eq!(c.offset(), 10);
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn arrays_cache_lengths_and_padding_are_optional() {
        let mut c = NativeArraysCache::new(2);
        assert!(c.lengths().is_none());
        assert!(c.left_padding().is_none());

        let lengths = Array::from_slice::<i32>(&[3, 7, 5], &[3]);
        c.set_lengths(Some(lengths));
        assert!(c.lengths().is_some());

        let left_padding = Array::from_slice::<i32>(&[0, 2, 1], &[3]);
        c.set_left_padding(Some(left_padding));
        assert!(c.left_padding().is_some());

        c.set_lengths(None);
        assert!(c.lengths().is_none());
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn prompt_cache_construction_matches_layer_kinds() {
        // Mirror the Qwen3.5-MoE structure: full_attention_interval=4 ⇒
        // every 4th layer is full-attn, rest are linear.
        let kinds = [
            true, true, true, false, // layer 3 = full
            true, true, true, false, // layer 7 = full
        ];
        let pc = NativePromptCache::new(&kinds, 2);
        assert_eq!(pc.len(), 8);

        for (i, &is_linear) in kinds.iter().enumerate() {
            let layer = pc.layer(i).unwrap();
            if is_linear {
                assert!(matches!(layer, NativeLayerCache::Linear(_)));
            } else {
                assert!(matches!(layer, NativeLayerCache::Full(_)));
            }
        }

        assert_eq!(pc.full_attn_offset(), 0);
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn prompt_cache_full_attn_offset_tracks_first_full_layer() {
        // 2 linear, 1 full, 2 linear, 1 full — full layers are 2 and 5.
        let kinds = [true, true, false, true, true, false];
        let mut pc = NativePromptCache::new(&kinds, 2);

        let full = pc.layer_mut(2).unwrap().as_full_mut().unwrap();
        let k = make_kv_block(1, 2, 4, 8, 0.0);
        let v = make_kv_block(1, 2, 4, 8, 1.0);
        full.update_and_fetch(&k, &v).unwrap();
        assert_eq!(pc.full_attn_offset(), 4);

        // The second full-attn layer is updated independently; offset() reads
        // the FIRST full-attn layer (lockstep invariant).
        let full2 = pc.layer_mut(5).unwrap().as_full_mut().unwrap();
        let k2 = make_kv_block(1, 2, 4, 8, 0.0);
        let v2 = make_kv_block(1, 2, 4, 8, 1.0);
        full2.update_and_fetch(&k2, &v2).unwrap();
        assert_eq!(pc.full_attn_offset(), 4);
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn prompt_cache_clear_resets_all_layers() {
        let kinds = [true, false, true, false];
        let mut pc = NativePromptCache::new(&kinds, 2);

        // Populate one full-attn + one linear-attn layer.
        let k = make_kv_block(1, 2, 4, 8, 0.0);
        let v = make_kv_block(1, 2, 4, 8, 1.0);
        pc.layer_mut(1)
            .unwrap()
            .as_full_mut()
            .unwrap()
            .update_and_fetch(&k, &v)
            .unwrap();
        let conv_state = make_kv_block(1, 1, 1, 1, 0.0);
        pc.layer_mut(0)
            .unwrap()
            .as_linear_mut()
            .unwrap()
            .set(0, conv_state)
            .unwrap();
        pc.layer_mut(0).unwrap().as_linear_mut().unwrap().advance(4);

        assert_eq!(pc.full_attn_offset(), 4);
        assert_eq!(pc.layer(0).unwrap().offset(), 4);

        pc.clear();
        assert_eq!(pc.full_attn_offset(), 0);
        assert_eq!(pc.layer(0).unwrap().offset(), 0);
        assert!(pc.layer(0).unwrap().as_linear().unwrap().empty_slots());
        assert!(pc.layer(1).unwrap().as_full().unwrap().empty());
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn layer_cache_kind_dispatch_errors_on_mismatch() {
        let mut linear = NativeLayerCache::Linear(NativeArraysCache::new(2));
        assert!(linear.as_full_mut().is_err());
        assert!(linear.as_linear_mut().is_ok());

        let mut full = NativeLayerCache::Full(NativeKvCache::new());
        assert!(full.as_linear_mut().is_err());
        assert!(full.as_full_mut().is_ok());
    }
}
