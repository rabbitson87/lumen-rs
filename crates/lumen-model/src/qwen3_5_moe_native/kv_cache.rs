//! Native KV cache (Phase D): pre-allocated `[B, kv_heads, max_seq, head_dim]`
//! NativeTensor backing for K and V, with append-only state advance.
//!
//! Why a separate cache from Candle's `KvCache`?
//!
//! For decode steps the production hot loop currently runs a Candle-side
//! `cache.append(k_new, v_new)` that returns `(k_cat, v_cat)` covering past + new.
//! Bridging those tensors back into native land per layer per step costs ~3
//! round-trips × 40 layers × ms each — eating the dispatch savings the native
//! pipeline aims to unlock. By keeping K/V resident as `NativeTensor` from the
//! moment they're produced, attention can read them directly without crossing
//! the Candle ↔ native boundary on every decode step.
//!
//! ## Layout
//!
//! Cache buffers are BHLD: `[B, kv_heads, max_seq_len, head_dim]`, contiguous,
//! row-major. The attention kernel walks K/V with a stride of `max_seq_len`
//! between heads (the *physical* capacity), but only attends to the first
//! `current_seq_len` slots (the *logical* active range). To make this work the
//! kernel takes a `kv_layout_stride` parameter independent from `l_kv`.
//!
//! ## Append
//!
//! `append(k_new, v_new)` blits each `(b, h)` head's slice from the source
//! tensor into the cache buffer at `[..., past..past+S, :]`. For Qwen3.5-MoE
//! (B=1, kv_heads=2) this is 4 small Metal blit copies per layer per step
//! (2 for K, 2 for V), each at most `head_dim * 4 * S` bytes (256 * 4 * S =
//! 1 KB for S=1, 11 KB for prefill S=11). Cost is microseconds.
//!
//! ## Reset
//!
//! `reset()` drops the active range to 0. The underlying buffers stay
//! allocated so subsequent generations reuse them — this is how the cache
//! stays cheap across many requests.

use anyhow::{Result, anyhow};
use lumen_metal::metal::{BlitCommandEncoderRef, MTLBlitOption};

use super::context::NativeContext;
use super::tensor::{NativeDType, NativeTensor};

pub struct NativeKvCache {
    k_buf: NativeTensor,
    v_buf: NativeTensor,
    current_seq_len: usize,
    max_seq_len: usize,
    b: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl NativeKvCache {
    /// Allocate a fresh cache. Both K and V share the same shape:
    /// `[b, kv_heads, max_seq_len, head_dim]` F32, zero-initialized.
    pub fn new(
        ctx: &NativeContext,
        b: usize,
        kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        if b == 0 || kv_heads == 0 || head_dim == 0 || max_seq_len == 0 {
            return Err(anyhow!(
                "NativeKvCache::new: all dims must be > 0 (got b={b}, kv={kv_heads}, d={head_dim}, max={max_seq_len})"
            ));
        }
        let k_buf = ctx.zeros(vec![b, kv_heads, max_seq_len, head_dim], NativeDType::F32)?;
        let v_buf = ctx.zeros(vec![b, kv_heads, max_seq_len, head_dim], NativeDType::F32)?;
        Ok(Self {
            k_buf,
            v_buf,
            current_seq_len: 0,
            max_seq_len,
            b,
            kv_heads,
            head_dim,
        })
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn batch(&self) -> usize {
        self.b
    }

    pub fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Drop the active range. Underlying buffers are retained so that the next
    /// `append` writes into them without re-allocating.
    pub fn reset(&mut self) {
        self.current_seq_len = 0;
    }

    /// Roll the active range back so at most `n_keep` tokens remain. Bytes
    /// past `n_keep` stay in the buffer and will be overwritten on the next
    /// `append`. No-op when `current_seq_len ≤ n_keep`. Used by speculative
    /// decoding rollback.
    pub fn truncate(&mut self, n_keep: usize) {
        if self.current_seq_len > n_keep {
            self.current_seq_len = n_keep;
        }
    }

    /// Append BHLD K_new and V_new (shape `[B, kv_heads, S, head_dim]`) to the
    /// cache. After return, `current_seq_len` advances by `S`.
    ///
    /// Implementation: one blit copy per `(b, h)` head per K/V (so up to
    /// `2 * b * kv_heads` copies). The blits are submitted on `ctx.queue`,
    /// committed, and waited on — synchronous from the caller's perspective.
    pub fn append(
        &mut self,
        ctx: &NativeContext,
        k_new: &NativeTensor,
        v_new: &NativeTensor,
    ) -> Result<()> {
        // Validate before opening the blit encoder so an early Err doesn't
        // leak an un-`end_encoding`'d encoder (Metal asserts on dealloc).
        self.validate_append(k_new, v_new)?;
        if k_new.shape()[2] == 0 {
            return Ok(());
        }
        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let mut blit = lumen_metal::metal::blit_command_encoder_no_fence(&cmd);
        // `encode_append` re-validates but cannot fail after `validate_append`
        // because input shape didn't change between the two calls.
        let _ = self.encode_append(&mut blit, k_new, v_new);
        blit.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    /// Validation-only helper. Mirrors the checks inside `encode_append`
    /// without touching `current_seq_len` or the encoder state.
    fn validate_append(&self, k_new: &NativeTensor, v_new: &NativeTensor) -> Result<()> {
        if k_new.rank() != 4 || v_new.rank() != 4 {
            return Err(anyhow!(
                "NativeKvCache::append: K/V must be rank 4 BHLD, got K={:?} V={:?}",
                k_new.shape(),
                v_new.shape()
            ));
        }
        let s = k_new.shape()[2];
        let expect = [self.b, self.kv_heads, s, self.head_dim];
        if k_new.shape() != expect || v_new.shape() != expect {
            return Err(anyhow!(
                "NativeKvCache::append: shape mismatch. expected [{},{},{},{}] got K={:?} V={:?}",
                self.b,
                self.kv_heads,
                s,
                self.head_dim,
                k_new.shape(),
                v_new.shape()
            ));
        }
        if self.current_seq_len + s > self.max_seq_len {
            return Err(anyhow!(
                "NativeKvCache::append: overflow {} + {} > {}",
                self.current_seq_len,
                s,
                self.max_seq_len
            ));
        }
        if k_new.dtype() != NativeDType::F32 || v_new.dtype() != NativeDType::F32 {
            return Err(anyhow!(
                "NativeKvCache::append: only F32 supported, got K={:?} V={:?}",
                k_new.dtype(),
                v_new.dtype()
            ));
        }
        Ok(())
    }

    /// Encode-only append: writes the blit copies into the caller-provided blit
    /// encoder and advances `current_seq_len` if validation succeeds. Caller is
    /// responsible for the encoder + command buffer lifecycle. Use this when
    /// fusing the cache append with surrounding compute kernels into a single
    /// command buffer to amortize commit/wait overhead.
    ///
    /// On validation failure (shape/overflow/etc), returns Err and the cache
    /// state stays untouched — the encoder may still have been used by earlier
    /// calls; whether to discard the in-progress command buffer is the
    /// caller's call.
    pub fn encode_append(
        &mut self,
        blit: &mut BlitCommandEncoderRef,
        k_new: &NativeTensor,
        v_new: &NativeTensor,
    ) -> Result<()> {
        if k_new.rank() != 4 || v_new.rank() != 4 {
            return Err(anyhow!(
                "NativeKvCache::append: K/V must be rank 4 BHLD, got K={:?} V={:?}",
                k_new.shape(),
                v_new.shape()
            ));
        }
        let s = k_new.shape()[2];
        let expect = [self.b, self.kv_heads, s, self.head_dim];
        if k_new.shape() != expect || v_new.shape() != expect {
            return Err(anyhow!(
                "NativeKvCache::append: shape mismatch. expected [{},{},{},{}] got K={:?} V={:?}",
                self.b,
                self.kv_heads,
                s,
                self.head_dim,
                k_new.shape(),
                v_new.shape()
            ));
        }
        if s == 0 {
            return Ok(());
        }
        if self.current_seq_len + s > self.max_seq_len {
            return Err(anyhow!(
                "NativeKvCache::append: overflow {} + {} > {}",
                self.current_seq_len,
                s,
                self.max_seq_len
            ));
        }
        if k_new.dtype() != NativeDType::F32 || v_new.dtype() != NativeDType::F32 {
            return Err(anyhow!(
                "NativeKvCache::append: only F32 supported, got K={:?} V={:?}",
                k_new.dtype(),
                v_new.dtype()
            ));
        }

        let elem_bytes = NativeDType::F32.size_in_bytes();
        let head_dim_us = self.head_dim;
        let s_us = s;
        let max_us = self.max_seq_len;
        let bytes_per_head = s_us * head_dim_us * elem_bytes;

        for b_idx in 0..self.b {
            for h in 0..self.kv_heads {
                let head_start_src = (b_idx * self.kv_heads + h) * s_us * head_dim_us;
                let src_off_k = k_new.offset_bytes() + head_start_src * elem_bytes;
                let src_off_v = v_new.offset_bytes() + head_start_src * elem_bytes;

                let head_start_dst = (b_idx * self.kv_heads + h) * max_us * head_dim_us
                    + self.current_seq_len * head_dim_us;
                let dst_off_k = self.k_buf.offset_bytes() + head_start_dst * elem_bytes;
                let dst_off_v = self.v_buf.offset_bytes() + head_start_dst * elem_bytes;

                blit.copy_from_buffer(
                    k_new.buffer(),
                    src_off_k,
                    self.k_buf.buffer(),
                    dst_off_k,
                    bytes_per_head,
                );
                blit.copy_from_buffer(
                    v_new.buffer(),
                    src_off_v,
                    self.v_buf.buffer(),
                    dst_off_v,
                    bytes_per_head,
                );
            }
        }
        let _ = MTLBlitOption::None;

        self.current_seq_len += s;
        Ok(())
    }

    /// Read access to the K buffer. Shape is `[B, kv_heads, max_seq_len, head_dim]`
    /// — the FULL physical capacity. The attention kernel must consume only the
    /// first `current_seq_len` slots and use `kv_layout_stride = max_seq_len` so
    /// indexing skips the unfilled tail correctly.
    pub fn k_buf(&self) -> &NativeTensor {
        &self.k_buf
    }

    pub fn v_buf(&self) -> &NativeTensor {
        &self.v_buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Append a single-step K/V and verify the cache reads back identically per head.
    #[test]
    fn append_single_step_roundtrip() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let b = 1;
        let kv = 2;
        let d = 4;
        let max = 8;
        let mut cache = NativeKvCache::new(&ctx, b, kv, d, max).unwrap();
        assert_eq!(cache.current_seq_len(), 0);

        let k_data: Vec<f32> = (0..b * kv * 1 * d).map(|i| i as f32 + 1.0).collect();
        let v_data: Vec<f32> = (0..b * kv * 1 * d).map(|i| -(i as f32) - 1.0).collect();
        let k_new = ctx.from_slice_f32(&k_data, vec![b, kv, 1, d]).unwrap();
        let v_new = ctx.from_slice_f32(&v_data, vec![b, kv, 1, d]).unwrap();

        cache.append(&ctx, &k_new, &v_new).unwrap();
        assert_eq!(cache.current_seq_len(), 1);

        // Read back the K buffer. Shape [b, kv, max, d] — only slot 0 of the L axis
        // should hold our data (slots 1..max remain zero from `ctx.zeros`).
        let k_full = cache.k_buf().to_vec_f32().unwrap();
        let v_full = cache.v_buf().to_vec_f32().unwrap();
        assert_eq!(k_full.len(), b * kv * max * d);

        // For each (b_idx, h), slot 0 = k_data[(b_idx*kv+h)*d .. (b_idx*kv+h+1)*d].
        for b_idx in 0..b {
            for h in 0..kv {
                let src_off = (b_idx * kv + h) * d;
                let dst_off = ((b_idx * kv + h) * max + 0) * d;
                for j in 0..d {
                    assert_eq!(
                        k_full[dst_off + j],
                        k_data[src_off + j],
                        "K mismatch h={h} j={j}"
                    );
                    assert_eq!(
                        v_full[dst_off + j],
                        v_data[src_off + j],
                        "V mismatch h={h} j={j}"
                    );
                }
                // Slot 1 must remain zero.
                let zero_off = ((b_idx * kv + h) * max + 1) * d;
                for j in 0..d {
                    assert_eq!(k_full[zero_off + j], 0.0, "tail K not zero h={h} j={j}");
                }
            }
        }
    }

    /// Append a multi-step prefill, then a single-step decode. Verify the cache
    /// holds the concatenation in the correct order per head.
    #[test]
    fn append_prefill_then_decode_concatenates() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let b = 1;
        let kv = 2;
        let d = 4;
        let max = 16;
        let mut cache = NativeKvCache::new(&ctx, b, kv, d, max).unwrap();

        // Prefill: S = 3 tokens.
        let s1 = 3;
        let k1: Vec<f32> = (0..b * kv * s1 * d).map(|i| i as f32 * 0.5).collect();
        let v1: Vec<f32> = (0..b * kv * s1 * d).map(|i| i as f32 * 0.25).collect();
        let k1_t = ctx.from_slice_f32(&k1, vec![b, kv, s1, d]).unwrap();
        let v1_t = ctx.from_slice_f32(&v1, vec![b, kv, s1, d]).unwrap();
        cache.append(&ctx, &k1_t, &v1_t).unwrap();
        assert_eq!(cache.current_seq_len(), s1);

        // Decode: S = 1 token.
        let s2 = 1;
        let k2: Vec<f32> = (0..b * kv * s2 * d).map(|i| 100.0 + i as f32).collect();
        let v2: Vec<f32> = (0..b * kv * s2 * d).map(|i| -100.0 - i as f32).collect();
        let k2_t = ctx.from_slice_f32(&k2, vec![b, kv, s2, d]).unwrap();
        let v2_t = ctx.from_slice_f32(&v2, vec![b, kv, s2, d]).unwrap();
        cache.append(&ctx, &k2_t, &v2_t).unwrap();
        assert_eq!(cache.current_seq_len(), s1 + s2);

        // Verify per-head concatenation in the cache buffer.
        let k_full = cache.k_buf().to_vec_f32().unwrap();
        for b_idx in 0..b {
            for h in 0..kv {
                // First s1 slots come from k1.
                for k_slot in 0..s1 {
                    let src_off = ((b_idx * kv + h) * s1 + k_slot) * d;
                    let dst_off = ((b_idx * kv + h) * max + k_slot) * d;
                    for j in 0..d {
                        assert_eq!(
                            k_full[dst_off + j],
                            k1[src_off + j],
                            "prefill K mismatch h={h} k={k_slot}"
                        );
                    }
                }
                // Slot s1 comes from k2.
                let src_off = (b_idx * kv + h) * d;
                let dst_off = ((b_idx * kv + h) * max + s1) * d;
                for j in 0..d {
                    assert_eq!(
                        k_full[dst_off + j],
                        k2[src_off + j],
                        "decode K mismatch h={h}"
                    );
                }
            }
        }
    }

    #[test]
    fn overflow_rejected() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let mut cache = NativeKvCache::new(&ctx, 1, 1, 4, 4).unwrap();
        let k = ctx
            .from_slice_f32(&vec![0.0; 1 * 1 * 5 * 4], vec![1, 1, 5, 4])
            .unwrap();
        let v = ctx
            .from_slice_f32(&vec![0.0; 1 * 1 * 5 * 4], vec![1, 1, 5, 4])
            .unwrap();
        let r = cache.append(&ctx, &k, &v);
        assert!(r.is_err());
    }

    #[test]
    fn shape_mismatch_rejected() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let mut cache = NativeKvCache::new(&ctx, 1, 2, 4, 8).unwrap();
        // Wrong kv (3 instead of 2)
        let k = ctx
            .from_slice_f32(&vec![0.0; 1 * 3 * 1 * 4], vec![1, 3, 1, 4])
            .unwrap();
        let v = ctx
            .from_slice_f32(&vec![0.0; 1 * 3 * 1 * 4], vec![1, 3, 1, 4])
            .unwrap();
        let r = cache.append(&ctx, &k, &v);
        assert!(r.is_err());
    }

    #[test]
    fn reset_drops_state() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let mut cache = NativeKvCache::new(&ctx, 1, 1, 2, 4).unwrap();
        let k = ctx
            .from_slice_f32(&vec![1.0; 1 * 1 * 2 * 2], vec![1, 1, 2, 2])
            .unwrap();
        let v = ctx
            .from_slice_f32(&vec![2.0; 1 * 1 * 2 * 2], vec![1, 1, 2, 2])
            .unwrap();
        cache.append(&ctx, &k, &v).unwrap();
        assert_eq!(cache.current_seq_len(), 2);
        cache.reset();
        assert_eq!(cache.current_seq_len(), 0);
        // Re-append OK.
        cache.append(&ctx, &k, &v).unwrap();
        assert_eq!(cache.current_seq_len(), 2);
    }
}
