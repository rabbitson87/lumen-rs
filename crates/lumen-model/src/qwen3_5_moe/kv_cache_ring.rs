//! Sliding-window ring-buffer KV cache for the 10 full-attention layers.
//!
//! ## Memory model
//!
//! Without this module, `SelfAttention` uses Candle's grow-forever `KvCache` which
//! accumulates O(seq_len) tensors even though SDPA already trims computation to the last
//! `LUMEN_FULL_ATTN_WINDOW` tokens.  `SlidingWindowKvCache` closes that gap: it keeps only
//! the last `window` K/V tokens in memory, giving O(window) steady-state storage regardless
//! of sequence length.
//!
//! ## Semantics
//!
//! `append` returns the **full** concatenated K/V to the caller (so each forward pass operates
//! on correct history), but stores only the trailing `window` tokens.  The split means:
//!
//! - **Prefill** (`seq_len > 1`): SDPA sees every prompt token; after the call only the last
//!   `window` tokens remain buffered, ready for decode.
//! - **Decode** (`seq_len == 1`): the stored buffer is always ≤ `window` tokens, so the
//!   returned tensor is at most `window + 1` long; the existing window-trim in
//!   `self_attn::forward` narrows it to exactly `window` for SDPA.

use candle_core::{Result as CandleResult, Tensor};

// ─── Sliding-window buffer ────────────────────────────────────────────────────

/// Fixed-capacity KV buffer that retains only the last `window` tokens.
pub struct SlidingWindowKvCache {
    concat_dim: usize,
    window: usize,
    key_buf: Option<Tensor>,
    val_buf: Option<Tensor>,
}

impl SlidingWindowKvCache {
    pub fn new(concat_dim: usize, window: usize) -> Self {
        Self {
            concat_dim,
            window,
            key_buf: None,
            val_buf: None,
        }
    }

    /// Number of tokens currently stored (≤ `window`).
    pub fn current_seq_len(&self) -> usize {
        self.key_buf
            .as_ref()
            .and_then(|t| t.dim(self.concat_dim).ok())
            .unwrap_or(0)
    }

    /// Concatenates `k`/`v` with the stored buffer, stores only the last `window` tokens,
    /// and returns the **full** concatenation for this forward pass.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let (k_cat, v_cat) = match (&self.key_buf, &self.val_buf) {
            (Some(kb), Some(vb)) => (
                Tensor::cat(&[kb, k], self.concat_dim)?,
                Tensor::cat(&[vb, v], self.concat_dim)?,
            ),
            _ => (k.clone(), v.clone()),
        };

        let cat_len = k_cat.dim(self.concat_dim)?;
        if cat_len > self.window {
            let start = cat_len - self.window;
            self.key_buf = Some(k_cat.narrow(self.concat_dim, start, self.window)?);
            self.val_buf = Some(v_cat.narrow(self.concat_dim, start, self.window)?);
        } else {
            self.key_buf = Some(k_cat.clone());
            self.val_buf = Some(v_cat.clone());
        }

        Ok((k_cat, v_cat))
    }

    /// Clears the buffer (conversation reset / new request).
    pub fn reset(&mut self) {
        self.key_buf = None;
        self.val_buf = None;
    }

    /// Keeps only the first `n_keep` stored tokens; no-op if `n_keep ≥ current_seq_len`.
    /// Mirrors `candle_nn::kv_cache::KvCache::truncate` for speculative-decoding rollback.
    pub fn truncate(&mut self, n_keep: usize) {
        let stored = self.current_seq_len();
        if n_keep < stored {
            self.key_buf = self
                .key_buf
                .take()
                .and_then(|t| t.narrow(self.concat_dim, 0, n_keep).ok());
            self.val_buf = self
                .val_buf
                .take()
                .and_then(|t| t.narrow(self.concat_dim, 0, n_keep).ok());
        }
    }
}

// ─── Unified cache handle ─────────────────────────────────────────────────────

/// Wraps either a standard grow-forever cache or the memory-bounded sliding-window cache.
///
/// Both variants expose the same interface so `SelfAttention` requires no conditional
/// logic after construction — call sites use `AttentionKvCache` uniformly.
pub enum AttentionKvCache {
    Standard(candle_nn::kv_cache::KvCache),
    SlidingWindow(SlidingWindowKvCache),
}

impl AttentionKvCache {
    /// Standard Candle cache (no memory bound).
    pub fn new_standard(max_seq_len: usize) -> Self {
        Self::Standard(candle_nn::kv_cache::KvCache::new(2, max_seq_len))
    }

    /// Sliding-window cache bounded to `window` tokens.
    pub fn new_sliding_window(window: usize) -> Self {
        Self::SlidingWindow(SlidingWindowKvCache::new(2, window))
    }

    pub fn current_seq_len(&self) -> usize {
        match self {
            Self::Standard(c) => c.current_seq_len(),
            Self::SlidingWindow(c) => c.current_seq_len(),
        }
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        match self {
            Self::Standard(c) => c.append(k, v),
            Self::SlidingWindow(c) => c.append(k, v),
        }
    }

    pub fn reset(&mut self) {
        match self {
            Self::Standard(c) => c.reset(),
            Self::SlidingWindow(c) => c.reset(),
        }
    }

    pub fn truncate(&mut self, n_keep: usize) {
        match self {
            Self::Standard(c) => c.truncate(n_keep),
            Self::SlidingWindow(c) => c.truncate(n_keep),
        }
    }
}
