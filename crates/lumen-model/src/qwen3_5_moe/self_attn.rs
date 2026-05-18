//! Gated GQA `self_attn` sub-module — dimensions, forward pass, and fixture-ready tests.
//!
//! Applies to the 10 full-attention layers (indices 3, 7, …, 39 in the 3:1 layer pattern).
//! The forward reference is [`mlx_lm.models.qwen3_next.Qwen3NextAttention`] (Apple MLX), and all
//! numerical choices below mirror that class exactly.
//!
//! ## Key structural facts
//! - **Gated attention output** (`text_config.attn_output_gate == true`, Qwen3-Next style):
//!   `q_proj` emits `[Q ‖ Gate]` per-head-interleaved, so its output width is
//!   **`2 · num_heads · head_dim`** rather than the usual `num_heads · head_dim`. Reshape as
//!   `[B, L, num_heads, 2·head_dim]`, split along the last axis — queries and the gate branch
//!   come out head-interleaved in the flat projection. The gate is a sigmoid applied to the
//!   attention output (before `o_proj`).
//! - **GQA** with `num_attention_heads=16`, `num_key_value_heads=2`, `head_dim=256`.
//! - **QK-norm** (`q_norm`, `k_norm`) is a single RMSNorm over `head_dim`, applied per-head
//!   **before** the rotary embedding.
//! - **mRoPE with partial rotary** (`partial_rotary_factor = 0.25`): only the first
//!   `head_dim × 0.25 = 64` dimensions of each head carry rotary phase; the rest pass through
//!   unchanged. For text-only decoding, mRoPE collapses to standard non-traditional RoPE
//!   (GPT-NeoX split form) — `mrope_section = [11, 11, 10]` only matters when image/video
//!   position IDs are interleaved, which never happens on a pure text path.
//! - **Causal mask**: additive `-inf` above the diagonal, `0` on/below.
//!
//! Ground truth for the shapes below was read from the header of
//! `model-00001-of-00004.safetensors`, layer 3 `self_attn.*`, on 2026-04-23.
//!
//! ## MLX activation dtype policy (verified mlx-lm 0.31.3, qwen3_next.py:97-141)
//!
//! `Qwen3NextAttention.__call__` contains **0 `.astype()` calls**. The activation
//! flows through `q_proj` → `q_norm` → `transpose` → `rope` →
//! `scaled_dot_product_attention` → `o_proj` entirely in input dtype (bf16 for
//! the bf16 checkpoint). This is the baseline our Rust impl is measured
//! against; see `workstream_b_phase3_mlx_baseline.md` for the full divergence
//! map. The current Rust pipeline runs f32-default with bf16 opt-in (Lever F)
//! followed by cast-back — this is **inverted relative to MLX** and is the
//! root cause of the bf16 chain's -1.58% / σ=-24.67 NEGATIVE bench.
//! Phase 3+ work (B.3 self_attn / B.4 linear_attn / Workstream C SDPA / B.5
//! post_attn norm / B.6 MLP+MoE) is the path to MLX-equivalent dtype policy.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use candle_nn::{rotary_emb, Module, RmsNorm};

use super::config::TextConfig;
use super::kv_cache_ring::AttentionKvCache;
use super::proj::ProjLinear;

/// Scalar dimensions for a single `self_attn` block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelfAttnDims {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Whether `q_proj` emits an extra gate branch concatenated after Q (Qwen3-Next).
    pub attn_output_gate: bool,
    /// `partial_rotary_factor × head_dim`, snapped to an even integer. Only this many
    /// leading per-head dims carry rotary phase; the rest are unmodified.
    pub rotary_dim: usize,
}

impl SelfAttnDims {
    pub fn from_config(t: &TextConfig) -> Result<Self, SelfAttnDimsError> {
        if t.num_attention_heads % t.num_key_value_heads != 0 {
            return Err(SelfAttnDimsError::GqaGroupIndivisible {
                n_heads: t.num_attention_heads,
                n_kv: t.num_key_value_heads,
            });
        }
        let raw_rotary = (t.head_dim as f32) * t.partial_rotary_factor;
        let rotary_dim = raw_rotary.round() as usize;
        if rotary_dim % 2 != 0 {
            return Err(SelfAttnDimsError::RotaryDimOdd { rotary_dim });
        }
        if rotary_dim > t.head_dim {
            return Err(SelfAttnDimsError::RotaryExceedsHeadDim {
                rotary_dim,
                head_dim: t.head_dim,
            });
        }
        Ok(Self {
            hidden_size: t.hidden_size,
            num_heads: t.num_attention_heads,
            num_kv_heads: t.num_key_value_heads,
            head_dim: t.head_dim,
            attn_output_gate: t.attn_output_gate,
            rotary_dim,
        })
    }

    /// Width of the Q projection output, accounting for the output-gate concatenation.
    pub fn q_out_dim(self) -> usize {
        let base = self.num_heads * self.head_dim;
        if self.attn_output_gate {
            2 * base
        } else {
            base
        }
    }

    /// Width of the K or V projection output (`num_kv_heads × head_dim`).
    pub fn kv_out_dim(self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Width of the attention value stream fed into `o_proj` (pre-gate width).
    pub fn attn_value_dim(self) -> usize {
        self.num_heads * self.head_dim
    }

    /// How many queries share one KV head under GQA.
    pub fn gqa_group_size(self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Logical (post-dequant) tensor shapes for every `self_attn.*` weight in a single layer.
    pub fn shapes(self) -> SelfAttnShapes {
        SelfAttnShapes {
            q_norm: vec![self.head_dim],
            k_norm: vec![self.head_dim],
            q_proj: vec![self.q_out_dim(), self.hidden_size],
            k_proj: vec![self.kv_out_dim(), self.hidden_size],
            v_proj: vec![self.kv_out_dim(), self.hidden_size],
            o_proj: vec![self.hidden_size, self.attn_value_dim()],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfAttnShapes {
    pub q_norm: Vec<usize>,
    pub k_norm: Vec<usize>,
    pub q_proj: Vec<usize>,
    pub k_proj: Vec<usize>,
    pub v_proj: Vec<usize>,
    pub o_proj: Vec<usize>,
}

#[derive(Debug, thiserror::Error)]
pub enum SelfAttnDimsError {
    #[error("num_attention_heads ({n_heads}) not divisible by num_key_value_heads ({n_kv})")]
    GqaGroupIndivisible { n_heads: usize, n_kv: usize },

    #[error("rotary_dim ({rotary_dim}) is odd; partial_rotary_factor × head_dim must be even")]
    RotaryDimOdd { rotary_dim: usize },

    #[error("rotary_dim ({rotary_dim}) exceeds head_dim ({head_dim})")]
    RotaryExceedsHeadDim { rotary_dim: usize, head_dim: usize },
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward pass
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime configuration for a single `self_attn` block. Split from [`SelfAttnDims`] because
/// these values (`rope_theta`, `rms_norm_eps`) are model-wide constants and change which
/// sin/cos table we need — but never the tensor shapes.
#[derive(Debug, Clone, Copy)]
pub struct SelfAttnRuntime {
    pub dims: SelfAttnDims,
    pub rope_theta: f32,
    pub rms_norm_eps: f64,
}

impl SelfAttnRuntime {
    pub fn from_text_config(t: &TextConfig) -> Result<Self, SelfAttnDimsError> {
        Ok(Self {
            dims: SelfAttnDims::from_config(t)?,
            rope_theta: t.rope_parameters.rope_theta,
            rms_norm_eps: t.rms_norm_eps as f64,
        })
    }
}

/// The [`Qwen3NextAttention`] Rust port. Constructed from already-materialized Candle layers so
/// that:
///   1. MXFP4-quantized projections can be dequantized once at load time and handed in, and
///   2. unit tests can stitch together synthetic `Linear`/`RmsNorm` pieces without touching
///      the safetensors loader.
///
/// The forward path is fixture-accurate against `mlx_lm.models.qwen3_next.Qwen3NextAttention`
/// for every full-attention layer of the MXFP4 checkpoint.
pub struct SelfAttention {
    /// Option M2 (2026-04-25): q+k+v fused into one `[q_out + 2*kv_out, hidden]` projection.
    /// The forward narrows the combined matmul output back into q_raw / keys / values.
    /// Saves 2 MXFP4 dispatches per full-attention layer × 10 layers per token.
    qkv_proj: ProjLinear,
    o_proj: ProjLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    runtime: SelfAttnRuntime,
    /// Per-sequence K/V append caches. Key = seq_id (0 = legacy single-seq path).
    /// Empty when `enable_kv_cache` has never been called. Each sequence gets its
    /// own `AttentionKvCache` via `init_seq_kv_cache(seq_id)`.
    kv_caches: HashMap<u64, AttentionKvCache>,
    /// Which sequence is currently active. All cache operations in `forward_with_tq_inner`
    /// read/write `kv_caches[current_seq_id]`. Set via `set_current_seq_id`.
    current_seq_id: u64,
    /// Capacity hint stored from the last `enable_kv_cache` call. Used by
    /// `init_seq_kv_cache` to allocate new per-seq entries without needing to re-pass
    /// the max_seq_len parameter.
    kv_max_seq_len: Option<usize>,
    /// native KV cache. Only valid for `current_seq_id == 0`.
    /// Populated lazily on first `LUMEN_NATIVE_OUTPUT=1` forward after
    /// `enable_kv_cache(max_seq)` was called.
    #[cfg(feature = "turboquant-gpu")]
    native_kv_cache: Option<crate::qwen3_5_moe_native::NativeKvCache>,
    /// Capacity passed to [`Self::enable_kv_cache`] — held until lazy native cache
    /// allocation. `None` means "no cache enabled" (cache won't fire even with the env flag).
    native_kv_cache_max: Option<usize>,
    /// Global decoder layer position (0..40) — needed when a TurboQuant compressor is
    /// attached so cache slots can be addressed by a sparse full-attn index.
    layer_idx: usize,
    /// If this layer participates in TurboQuant KV compression, this is its slot index in the
    /// compressor (0..num_full_attn_layers). `None` → compression disabled for this layer.
    tq_slot: Option<usize>,
    /// Lever E Option A (2026-04-28): pre-built RoPE cos/sin table covering
    /// `[0, max_seq_len)` positions. Lazy-init on first forward when
    /// `rope_cache_max_seq` is set; subsequent forwards take a `narrow` view
    /// (zero dispatch) instead of rebuilding `build_rope_table`'s 7-dispatch
    /// chain per layer per token.
    rope_cache: Option<Arc<RopeCache>>,
    /// Capacity hint passed via `enable_kv_cache` — mirrors `native_kv_cache_max`.
    /// `None` → cache disabled, falls back to per-call `build_rope_table`.
    rope_cache_max_seq: Option<usize>,
}

/// Reads `LUMEN_FULL_ATTN_WINDOW` once and caches it. Used by both
/// `enable_kv_cache` and `init_seq_kv_cache` to avoid two OnceLocks.
fn attn_window_size() -> usize {
    static ATTN_WINDOW: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ATTN_WINDOW.get_or_init(|| {
        std::env::var("LUMEN_FULL_ATTN_WINDOW")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    })
}

/// Reads `LUMEN_ATTN_SINK_SIZE` once. StreamingLLM-style attention sink:
/// when SWA is active and sink > 0, the first `sink` tokens are kept in the
/// window even after sliding past them. Default 0 (pure SWA).
fn attn_sink_size() -> usize {
    static ATTN_SINK: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ATTN_SINK.get_or_init(|| {
        std::env::var("LUMEN_ATTN_SINK_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    })
}

/// Reads `LUMEN_FA_GQA_INKERNEL` once. When enabled, the GQA expand+contiguous
/// (`repeat_kv_heads`) is skipped for the flash_attn path: K/V are passed at
/// `[B, num_kv_heads, S, D]` and the kernel handles `h_kv = h / group`
/// internally. Saves ~50% per-step time at long-prompt 35B (103ms → 52ms,
/// +127.6% throughput, +294σ STRONG, 20/20 bit-identical — see
/// `fa_gqa_inkernel_landed.md`). Default ON; set to "0" to opt out.
fn fa_gqa_inkernel_enabled() -> bool {
    static FA_GQA: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FA_GQA.get_or_init(|| {
        // Default ON: only the explicit string "0" disables the lever.
        std::env::var("LUMEN_FA_GQA_INKERNEL").as_deref() != Ok("0")
    })
}

/// Reads `LUMEN_RESIDUAL_FUSION` / `LUMEN_DISABLE_RESIDUAL_FUSION`.
/// When enabled, the layer-level `(x + r)?` (attn-side, ful-attn) and
/// `(h + mlp_out)?` (MoE-side, all 40 layers) adds are folded into the
/// MXFP4 o_proj kernel and into MoE's tri_add kernel respectively. LANDED
/// 2026-05-07: Step 1+2 combined +3.99σ decode +0.6%, -3.08σ TTFT -8.9%,
/// 20/20 bit-identical (see `l1_residual_fusion_step1_and_step2_landed.md`).
/// Default ON. Set `LUMEN_DISABLE_RESIDUAL_FUSION=1` to opt out (rollback
/// or debugging). Legacy `LUMEN_RESIDUAL_FUSION=0` still honored as opt-out.
#[cfg(feature = "turboquant-gpu")]
pub(super) fn residual_fusion_enabled() -> bool {
    static RF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *RF.get_or_init(|| {
        // Explicit disable wins.
        if std::env::var("LUMEN_DISABLE_RESIDUAL_FUSION").as_deref() == Ok("1") {
            return false;
        }
        // Legacy env semantics (pre-LAND): "0" = opt-out, anything else = ON.
        std::env::var("LUMEN_RESIDUAL_FUSION").as_deref() != Ok("0")
    })
}

impl SelfAttention {
    pub fn new(
        runtime: SelfAttnRuntime,
        qkv_proj: ProjLinear,
        o_proj: ProjLinear,
        q_norm: RmsNorm,
        k_norm: RmsNorm,
    ) -> Self {
        Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            runtime,
            kv_caches: HashMap::new(),
            current_seq_id: 0,
            kv_max_seq_len: None,
            #[cfg(feature = "turboquant-gpu")]
            native_kv_cache: None,
            native_kv_cache_max: None,
            layer_idx: 0,
            tq_slot: None,
            rope_cache: None,
            rope_cache_max_seq: None,
        }
    }

    /// Opt-in the KV append cache (call at load time).
    ///
    /// When `LUMEN_FULL_ATTN_WINDOW > 0` a bounded `SlidingWindowKvCache` is used, keeping
    /// storage at O(window) regardless of sequence length.  Otherwise a standard Candle
    /// `KvCache` is used (grow-forever, initial capacity `max_seq_len`).
    pub fn enable_kv_cache(&mut self, max_seq_len: usize) {
        let cache = if attn_window_size() > 0 {
            AttentionKvCache::new_sliding_window(attn_window_size())
        } else {
            AttentionKvCache::new_standard(max_seq_len)
        };
        // seq_id=0 is the legacy single-seq slot; always init it here.
        self.kv_caches.insert(0, cache);
        self.kv_max_seq_len = Some(max_seq_len);
        self.native_kv_cache_max = Some(max_seq_len);
        self.rope_cache_max_seq = Some(max_seq_len);
    }

    pub fn reset_kv_cache(&mut self) {
        if let Some(c) = self.kv_caches.get_mut(&self.current_seq_id) {
            c.reset();
        }
        #[cfg(feature = "turboquant-gpu")]
        if self.current_seq_id == 0 {
            if let Some(nc) = self.native_kv_cache.as_mut() {
                nc.reset();
            }
        }
    }

    /// Roll the per-layer KV append cache back to at most `n_keep` tokens.
    /// Targets both the Candle `KvCache` and the lazy native cache (if
    /// allocated). Used by speculative decoding rollback.
    pub fn truncate_kv_cache(&mut self, n_keep: usize) {
        if let Some(c) = self.kv_caches.get_mut(&self.current_seq_id) {
            c.truncate(n_keep);
        }
        #[cfg(feature = "turboquant-gpu")]
        if self.current_seq_id == 0 {
            if let Some(nc) = self.native_kv_cache.as_mut() {
                nc.truncate(n_keep);
            }
        }
    }

    pub fn set_layer_idx(&mut self, layer_idx: usize) {
        self.layer_idx = layer_idx;
    }

    pub fn set_tq_slot(&mut self, slot: Option<usize>) {
        self.tq_slot = slot;
    }

    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    pub fn tq_slot(&self) -> Option<usize> {
        self.tq_slot
    }

    pub fn kv_len(&self) -> usize {
        self.kv_caches
            .get(&self.current_seq_id)
            .map(|c| c.current_seq_len())
            .unwrap_or(0)
    }

    /// Set which sequence's KV cache to use for subsequent forward calls.
    pub fn set_current_seq_id(&mut self, seq_id: u64) {
        self.current_seq_id = seq_id;
    }

    /// Allocate a KV cache entry for `seq_id`. No-op if already exists.
    /// Requires `enable_kv_cache` to have been called first.
    pub fn init_seq_kv_cache(&mut self, seq_id: u64) {
        if self.kv_caches.contains_key(&seq_id) {
            return;
        }
        let max_seq = match self.kv_max_seq_len {
            Some(m) => m,
            None => return,
        };
        let cache = if attn_window_size() > 0 {
            AttentionKvCache::new_sliding_window(attn_window_size())
        } else {
            AttentionKvCache::new_standard(max_seq)
        };
        self.kv_caches.insert(seq_id, cache);
    }

    /// Free the KV cache for `seq_id` (called after a sequence completes).
    pub fn remove_seq_kv_cache(&mut self, seq_id: u64) {
        self.kv_caches.remove(&seq_id);
    }

    /// Reset (keep allocated) the KV cache for `seq_id`.
    pub fn reset_seq_kv_cache(&mut self, seq_id: u64) {
        if let Some(c) = self.kv_caches.get_mut(&seq_id) {
            c.reset();
        }
        #[cfg(feature = "turboquant-gpu")]
        if seq_id == 0 {
            if let Some(nc) = self.native_kv_cache.as_mut() {
                nc.reset();
            }
        }
    }

    /// Lever E Option A (2026-04-28): cos/sin lookup wrapper. Returns a narrow
    /// view of the per-layer pre-built cache when active (zero dispatch),
    /// otherwise falls through to the legacy `build_rope_table` (the path used
    /// before caching was introduced). Cache is rebuilt only when the
    /// `(rotary_dim, theta, dtype)` tuple changes — so a steady-state decoder
    /// pays the build cost exactly once per layer for the entire run.
    ///
    /// Opt-out: `LUMEN_DISABLE_ROPE_CACHE=1`. Off-by-default would slow every
    /// production decode step, so this is opt-out.
    fn cached_rope_table_or_build(
        &mut self,
        rotary_dim: usize,
        seq_len: usize,
        pos_offset: usize,
        dtype: DType,
        device: &Device,
    ) -> CandleResult<(Tensor, Tensor)> {
        let cache_disabled = std::env::var("LUMEN_DISABLE_ROPE_CACHE")
            .map(|v| v == "1")
            .unwrap_or(false);
        let max_seq = self.rope_cache_max_seq.unwrap_or(0);
        let theta = self.runtime.rope_theta;
        if !cache_disabled && max_seq > 0 && pos_offset + seq_len <= max_seq {
            let need_rebuild = match self.rope_cache.as_ref() {
                Some(c) => !c.matches(rotary_dim, theta, dtype) || c.max_seq_len < max_seq,
                None => true,
            };
            if need_rebuild {
                let new_cache = RopeCache::new(rotary_dim, max_seq, theta, device, dtype)?;
                self.rope_cache = Some(Arc::new(new_cache));
            }
            return self
                .rope_cache
                .as_ref()
                .expect("rope_cache populated above")
                .get(seq_len, pos_offset);
        }
        build_rope_table(rotary_dim, seq_len, pos_offset, theta, device, dtype)
    }

    pub fn dims(&self) -> SelfAttnDims {
        self.runtime.dims
    }

    /// Build a pure-prefill causal mask for `seq_len` tokens in the given dtype/device. Pass
    /// the result as `mask` to [`Self::forward`] for production prefill; in block-level
    /// fixture tests pass `None` instead to match the MLX fixture's unmasked reference.
    pub fn prefill_causal_mask(
        seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> CandleResult<Tensor> {
        causal_mask(seq_len, dtype, device)
    }

    /// Forward pass. Matches `Qwen3NextAttention.__call__` in mlx-lm 0.31.3.
    ///
    /// `x`: shape `[B, L, hidden_size]` (the `input_layernorm` output).
    /// `pos_offset`: token index of `x[:, 0]` — used by RoPE. For full-context prefill this is
    ///   `0`; for a `cache.offset > 0` decode step, pass the current KV length.
    /// `mask`: additive attention mask broadcastable to `[B, H, L, S]` where `S` is the total
    ///   attended sequence length. Use `None` for unmasked (bidirectional) attention — this is
    ///   what MLX's `layer.self_attn(x)` defaults to, so block-level fixture comparisons must
    ///   use `None`. Production prefill/decode should pass a causal mask via
    ///   [`Self::prefill_causal_mask`].
    ///
    /// Returns: shape `[B, L, hidden_size]`.
    ///
    /// No KV cache wiring yet — prefill-only. The cache landing happens alongside Stage 5.
    pub fn forward(
        &mut self,
        x: &Tensor,
        pos_offset: usize,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        self.forward_with_tq(x, pos_offset, mask, &mut None)
    }

    /// Forward that additionally threads a TurboQuant compressed-KV backend. On prefill steps
    /// whose cumulative context reaches `TQ_THRESHOLD`, the pre-GQA-repeat K/V tensors for
    /// the new tokens are pushed into the compressor. On decode (seq_len=1) calls, once the
    /// compressed cache has ≥ threshold tokens, the full attention output is served from the
    /// compressor, bypassing SDPA entirely. `TQ_THRESHOLD` defaults to 2048.
    pub fn forward_with_tq(
        &mut self,
        x: &Tensor,
        pos_offset: usize,
        mask: Option<&Tensor>,
        compressed_kv: &mut Option<Box<dyn candle_transformers::models::quantized_gemma4::CompressedKVBackend + Send>>,
    ) -> CandleResult<Tensor> {
        self.forward_with_tq_inner(x, None, None, pos_offset, mask, compressed_kv)
    }

    /// Lever L1 (residual fusion): forward variant that folds the layer-level
    /// post-attention residual `x + r` into o_proj's MXFP4 matmul kernel.
    /// `residual` should be the layer's pre-norm input; the returned tensor
    /// already includes the add, so the caller (layer.rs) MUST skip its own
    /// `(x + r)?`. Falls back gracefully (legacy o_proj + broadcast_add) when
    /// the fusion is not applicable (Dense weight, bf16 path, etc.).
    pub fn forward_with_residual_fused(
        &mut self,
        x: &Tensor,
        residual: &Tensor,
        pos_offset: usize,
        mask: Option<&Tensor>,
        compressed_kv: &mut Option<Box<dyn candle_transformers::models::quantized_gemma4::CompressedKVBackend + Send>>,
    ) -> CandleResult<Tensor> {
        self.forward_with_tq_inner(x, None, Some(residual), pos_offset, mask, compressed_kv)
    }

    /// Lever H Step 3 (2026-04-28): pre-attention RmsNorm fused into the qkv
    /// matmul. Reads RAW `x_raw` (un-normalized) and the `input_layernorm`
    /// weight; the qkv kernel computes RmsNorm internally. Caller must verify
    /// `has_mxfp4_qkv() == true` — the Dense path (test fixtures only) is not
    /// supported here.
    ///
    /// Mutually exclusive with `LUMEN_NATIVE_OUTPUT=1` (native substitute
    /// bypasses the candle qkv stage entirely) and `LUMEN_BF16_OUT*=1` (uses
    /// a different qkv kernel). When either is set the caller must keep
    /// `LUMEN_DISABLE_INPUT_RMSNORM_FUSION=1`.
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_with_input_rmsnorm(
        &mut self,
        x_raw: &Tensor,
        rms_weight: &Tensor,
        rms_eps: f32,
        pos_offset: usize,
        mask: Option<&Tensor>,
        compressed_kv: &mut Option<Box<dyn candle_transformers::models::quantized_gemma4::CompressedKVBackend + Send>>,
    ) -> CandleResult<Tensor> {
        self.forward_with_tq_inner(
            x_raw,
            Some((rms_weight, rms_eps)),
            None,
            pos_offset,
            mask,
            compressed_kv,
        )
    }

    /// True iff the qkv projection holds MXFP4 packed weights — required for
    /// the Lever H Step 3 input-rmsnorm fusion path.
    pub fn has_mxfp4_qkv(&self) -> bool {
        self.qkv_proj.is_mxfp4()
    }

    /// True iff the qkv projection holds Affine4 4-bit packed weights.
    /// Sibling to `has_mxfp4_qkv` for Workstream B bf16-rmsnorm gating.
    pub fn has_affine4_qkv(&self) -> bool {
        self.qkv_proj.is_affine4()
    }

    /// True iff the qkv projection supports input-RmsNorm fusion via a fused
    /// GPU kernel (MXFP4 Lever H Step 3 or Affine4 Lever R1). Used by callers
    /// to gate the `forward_with_input_rmsnorm` path.
    pub fn supports_fused_input_rmsnorm(&self) -> bool {
        self.qkv_proj.is_mxfp4() || self.qkv_proj.is_affine4()
    }

    fn forward_with_tq_inner(
        &mut self,
        x: &Tensor,
        input_rms: Option<(&Tensor, f32)>,
        residual: Option<&Tensor>,
        pos_offset: usize,
        mask: Option<&Tensor>,
        compressed_kv: &mut Option<Box<dyn candle_transformers::models::quantized_gemma4::CompressedKVBackend + Send>>,
    ) -> CandleResult<Tensor> {
        let (batch, seq_len, hidden) = x.dims3()?;
        if hidden != self.runtime.dims.hidden_size {
            candle_core::bail!(
                "self_attn input hidden {hidden} does not match config {}",
                self.runtime.dims.hidden_size
            );
        }
        let d = self.runtime.dims;
        // Lever B L.3 (2026-04-28) — bf16 input path: when LUMEN_BF16_RMSNORM=1,
        // upstream `input_layernorm` produces a bf16 tensor.
        //
        // Workstream B Phase 3 (2026-05-08, MLX policy alignment): `dtype` now
        // tracks the input dtype. With Workstream C's bf16 flash_attn kernel,
        // the chain `qkv → q_norm → k_norm → transpose → RoPE → flash_attn`
        // can stay in bf16 throughout — matching MLX `Qwen3NextAttention.__call__`
        // which has 0 explicit `.astype()` calls. The single cast back to f32
        // is deferred to the o_proj boundary (residual stream is still f32 for
        // now; B.5+ extends bf16 to layer level).
        //
        // MLX reference (qwen3_next.py:97-141, mlx-lm 0.31.3):
        //   q_proj_output → split → q_norm/k_norm/v reshape → transpose →
        //   rope → scaled_dot_product_attention → o_proj — all in input dtype.
        // See `workstream_b_phase3_mlx_baseline.md` and
        //     `workstream_c_flash_attn_bf16_landed.md`.
        let bf16_in_path = x.dtype() == candle_core::DType::BF16;
        let dtype = x.dtype();
        let device = x.device().clone();

        // ── Phase A.7 native parity probe ──────────────────────────────────
        // When `LUMEN_QWEN35MOE_NATIVE=1` and the call is a no-cache prefill
        // (past_kv_len == 0, no TurboQuant compressor), run the native forward
        // alongside Candle and report cosine + ms. This validates the native
        // pipeline on real model weights and quantifies the per-layer ROI.
        // The native result is *not* substituted into the production output —
        // see `LUMEN_NATIVE_OUTPUT=1` below for that.
        let past_kv_len_for_probe = self.kv_caches.get(&self.current_seq_id).map(|c| c.current_seq_len()).unwrap_or(0);
        // Lever H Step 3: native parity probe expects pre-normalized x; skip
        // when the input-rmsnorm fusion is in effect (raw x).
        let native_probe = crate::qwen3_5_moe_native::native_forward_enabled()
            && past_kv_len_for_probe == 0
            && compressed_kv.is_none()
            && input_rms.is_none();
        let native_probe_out: Option<(candle_core::Tensor, f64)> = if native_probe {
            run_native_self_attn_probe(self, x, pos_offset)
        } else {
            None
        };

        // ── Phase D native output substitute ───────────────────────────────
        // When `LUMEN_NATIVE_OUTPUT=1`, route prefill AND decode through the
        // native pipeline backed by `NativeKvCache`. No bridge round-trip on K/V
        // per step — once the cache is populated it stays resident as Metal
        // buffers, so attention reads it directly.
        //
        // Falls back to Candle if a TurboQuant compressor is attached (`ckv` path
        // not yet wired into native cache) or if `enable_kv_cache` was never
        // called (no `native_kv_cache_max`).
        // Lever H Step 3: NATIVE_OUTPUT substitute bypasses the candle qkv
        // stage; incompatible with input-rmsnorm fusion (raw x). Skip when
        // fusion is in effect.
        #[cfg(feature = "turboquant-gpu")]
        if crate::qwen3_5_moe_native::native_output_enabled()
            && self.native_kv_cache_max.is_some()
            && compressed_kv.is_none()
            && input_rms.is_none()
        {
            // Sanity: the native cache derives `pos_offset` from its own
            // `current_seq_len`; the Candle-side `pos_offset` argument must
            // agree (production model.rs derives both from the same
            // `seqlen_offset`).
            let _ = pos_offset;
            if let Some(out) = run_native_self_attn_substitute_with_cache(self, x) {
                return Ok(out);
            }
            // Helper returned None → fall through to Candle.
        }

        // Optional fine-grained self_attn timing: `LUMEN_SELF_ATTN_TIMING=1`
        // breaks the forward into 6 stages (qkv_proj / qknorm / rope / kv_append
        // / sdpa / o_proj). Each marker syncs the device, so off by default.
        // Used to size the ROI of bf16 vs further self_attn kernel work; only
        // active when the Candle path runs (LUMEN_NATIVE_OUTPUT unset).
        let sa_timing = std::env::var("LUMEN_SELF_ATTN_TIMING")
            .map(|v| v == "1")
            .unwrap_or(false);
        let mut sa_marks: Vec<(&'static str, std::time::Instant)> = Vec::new();
        if sa_timing {
            sa_marks.push(("start", std::time::Instant::now()));
        }

        // ── 1. Q / K / V projections ───────────────────────────────────────
        // Option M2: one fused matmul produces [..., q_out + 2*kv_out]; narrow into
        // q_raw / k_raw / v_raw. Same trick as Option M for linear_attn — saves 2
        // MXFP4 dispatches per full-attention layer.
        let q_out = d.q_out_dim();
        let kv_out = d.kv_out_dim();
        // Lever H Step 3 (2026-04-28): when input-rmsnorm fusion is in effect,
        // dispatch the rmsnorm-fused qkv kernel (raw x + ln_w + ln_eps). This
        // path requires MXFP4 qkv_proj — caller checks `has_mxfp4_qkv()`.
        // Otherwise, Phase A.1.5: bf16-output qkv_proj when LUMEN_BF16_OUT=1,
        // then cast back to f32 so downstream narrow/RMSNorm/RoPE/attention
        // are unchanged.
        let qkv_combined = match input_rms {
            #[cfg(feature = "turboquant-gpu")]
            Some((rms_w, rms_eps)) => {
                // Both MXFP4 and 4-bit affine support input-RmsNorm fusion
                // (Lever H Step 3 / Lever R1). Direct dispatch skips ProjLinear's
                // MxFp4Context-dependent unified entry — neither quant variant
                // needs ctx for its own fused kernel.
                if let Some(l) = self.qkv_proj.as_mxfp4() {
                    l.forward_with_rmsnorm(x, rms_w, rms_eps).map_err(|e| {
                        candle_core::Error::Msg(format!(
                            "self_attn qkv mxfp4 forward_with_rmsnorm: {e}"
                        ))
                    })?
                } else if let Some(l) = self.qkv_proj.as_affine4() {
                    l.forward_with_rmsnorm(x, rms_w, rms_eps).map_err(|e| {
                        candle_core::Error::Msg(format!(
                            "self_attn qkv affine4 forward_with_rmsnorm: {e}"
                        ))
                    })?
                } else {
                    candle_core::bail!(
                        "self_attn input-rmsnorm fusion requires MXFP4 or Affine4 qkv_proj"
                    )
                }
            }
            #[cfg(not(feature = "turboquant-gpu"))]
            Some(_) => candle_core::bail!(
                "self_attn input-rmsnorm fusion requires turboquant-gpu feature"
            ),
            None => {
                if bf16_in_path {
                    // Workstream B Phase 3 (2026-05-08): bf16-in-bf16-out path.
                    // Returns bf16 directly; downstream (q_norm/k_norm/transpose/
                    // RoPE/flash_attn) is dtype-generic and stays in bf16 until
                    // the o_proj boundary. Removes the prior cast-back-to-f32
                    // detour that paired with `forward_bf16_in`.
                    //
                    // MLX reference (qwen3_next.py:99): `q_proj_output = self.q_proj(x)`
                    // — output dtype matches input (bf16 → bf16). No astype.
                    #[cfg(feature = "turboquant-gpu")]
                    {
                        self.qkv_proj.forward_bf16_in_bf16_out(x)?
                    }
                    #[cfg(not(feature = "turboquant-gpu"))]
                    {
                        candle_core::bail!(
                            "self_attn bf16-in path requires turboquant-gpu feature"
                        )
                    }
                } else if super::moe::bf16_out_enabled() {
                    let y_bf16 = self.qkv_proj.forward_bf16_out(x)?;
                    y_bf16.to_dtype(candle_core::DType::F32)?
                } else {
                    self.qkv_proj.forward(x)?
                }
            }
        };
        let last = qkv_combined.dims().len() - 1;
        let q_raw = qkv_combined.narrow(last, 0, q_out)?.contiguous()?;
        let k_raw = qkv_combined.narrow(last, q_out, kv_out)?.contiguous()?;
        let v_raw = qkv_combined
            .narrow(last, q_out + kv_out, kv_out)?
            .contiguous()?;

        let (queries, gate_flat) = if d.attn_output_gate {
            // q_raw: [B, L, 2·H·D] — interpret as [B, L, H, 2·D], split last axis.
            let q_split = q_raw.reshape((batch, seq_len, d.num_heads, 2 * d.head_dim))?;
            let q = q_split.narrow(D::Minus1, 0, d.head_dim)?;
            let g = q_split.narrow(D::Minus1, d.head_dim, d.head_dim)?;
            // gate goes back to flat per-token width: [B, L, H·D].
            let g_flat = g
                .contiguous()?
                .reshape((batch, seq_len, d.num_heads * d.head_dim))?;
            (q.contiguous()?, Some(g_flat))
        } else {
            let q = q_raw.reshape((batch, seq_len, d.num_heads, d.head_dim))?;
            (q, None)
        };

        let keys = k_raw.reshape((batch, seq_len, d.num_kv_heads, d.head_dim))?;
        let values = v_raw.reshape((batch, seq_len, d.num_kv_heads, d.head_dim))?;
        if sa_timing {
            let _ = device.synchronize();
            sa_marks.push(("qkv_proj", std::time::Instant::now()));
        }

        // ── LUMEN_DISABLE_QKNORM_ROPE: L.0 cost-model bypass ────────────────
        static QNR_SKIP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let skip_qknorm_rope = *QNR_SKIP.get_or_init(|| {
            std::env::var("LUMEN_DISABLE_QKNORM_ROPE").as_deref() == Ok("1")
        });

        // ── 2. QK-norm (per-head RMSNorm along head_dim) ───────────────────
        // Workstream B Phase 3: candle_nn::RmsNorm errors on dtype mismatch
        // ("rmsnorm is not implemented for BF16 F32"). When the bf16 chain is
        // active, queries/keys are bf16 but the weight is f32. Cast the weight
        // to match — small ([head_dim] = 256 elements) so the per-call cost is
        // negligible (sub-microsecond), and avoids casting the larger
        // queries/keys tensors. Equivalent to MLX `nn.RMSNorm` which is
        // dtype-generic via fast.rms_norm.
        let q_norm_apply = |t: &Tensor| -> CandleResult<Tensor> {
            if t.dtype() == self.q_norm.weight().dtype() {
                self.q_norm.forward(t)
            } else {
                let w = self.q_norm.weight().to_dtype(t.dtype())?;
                candle_nn::ops::rms_norm(t, &w, self.q_norm.eps() as f32)
            }
        };
        let k_norm_apply = |t: &Tensor| -> CandleResult<Tensor> {
            if t.dtype() == self.k_norm.weight().dtype() {
                self.k_norm.forward(t)
            } else {
                let w = self.k_norm.weight().to_dtype(t.dtype())?;
                candle_nn::ops::rms_norm(t, &w, self.k_norm.eps() as f32)
            }
        };
        let queries = if skip_qknorm_rope { queries } else { q_norm_apply(&queries)? };
        let keys = if skip_qknorm_rope { keys } else { k_norm_apply(&keys)? };

        // ── 3. Transpose to [B, H, L, D] layout for attention ──────────────
        let queries = queries.transpose(1, 2)?.contiguous()?;
        let keys = keys.transpose(1, 2)?.contiguous()?;
        let values = values.transpose(1, 2)?.contiguous()?;
        if sa_timing {
            let _ = device.synchronize();
            sa_marks.push(("qknorm", std::time::Instant::now()));
        }

        // ── 4. Partial RoPE on first `rotary_dim` head-dim components ──────
        // Lever E Option A (2026-04-28): use the pre-built cos/sin cache when
        // available — saves the ~7-dispatch `build_rope_table` chain per layer
        // per token (×10 full-attn = ~70 dispatches/decode step). Cache build
        // shares the same Candle ops as `build_rope_table(..., max_seq, 0,
        // ...)` and `narrow` is element-wise sub-view, so the bit-identical
        // invariant is preserved.
        let (queries, keys) = if skip_qknorm_rope {
            (queries, keys)
        } else {
            let (cos, sin) = self.cached_rope_table_or_build(
                d.rotary_dim,
                seq_len,
                pos_offset,
                dtype,
                &device,
            )?;
            let q = apply_partial_rope(&queries, &cos, &sin, d.rotary_dim)?;
            let k = apply_partial_rope(&keys, &cos, &sin, d.rotary_dim)?;
            (q, k)
        };
        if sa_timing {
            let _ = device.synchronize();
            sa_marks.push(("rope", std::time::Instant::now()));
        }

        // ── 4b. Cache the new (pre-GQA-repeat) K/V ──────────────────────
        // The vanilla Candle `KvCache` stores [B, num_kv_heads, S, D] and returns the full
        // concatenation. The TurboQuant compressor (if attached) receives only the NEW tokens
        // so its internal `store_kv` can append them to its quantized state.
        static TQ_THRESHOLD: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let tq_thresh = *TQ_THRESHOLD.get_or_init(|| {
            std::env::var("TQ_THRESHOLD")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(2048)
        });
        let past_kv_len = self.kv_caches.get(&self.current_seq_id).map(|c| c.current_seq_len()).unwrap_or(0);
        let cumulative_kv_after = past_kv_len + seq_len;

        // Push new tokens into the TurboQuant compressor (prefill-phase accumulation).
        if let (Some(ckv), Some(slot)) = (compressed_kv.as_mut(), self.tq_slot) {
            if cumulative_kv_after >= tq_thresh {
                let new_k = keys.contiguous()?;
                let new_v = values.contiguous()?;
                let _ = ckv.store_kv(slot, &new_k, &new_v);
            }
        }

        // Vanilla KV append: returns [B, num_kv_heads, past+S, D].
        let (keys, values) = if let Some(cache) = self.kv_caches.get_mut(&self.current_seq_id) {
            let (k_cat, v_cat) = cache.append(&keys.contiguous()?, &values.contiguous()?)?;
            (k_cat, v_cat)
        } else {
            (keys, values)
        };
        if sa_timing {
            let _ = device.synchronize();
            sa_marks.push(("kv_append", std::time::Instant::now()));
        }

        // ── 4c. Decode fast path via TurboQuant ─────────────────────────
        // Single-step decode whose cache has crossed threshold: serve attention from the
        // compressed backend and jump past SDPA. Shape contract matches Gemma's Metal path:
        // `compressed_attention` returns [B, n_head, 1, head_dim].
        // `LUMEN_TQ_BYPASS_ATTN=1` keeps TQ
        // storage running but routes attention through SDPA, isolating
        // storage-side from attention-side bugs during deep debug.
        let bypass_tq_attn = std::env::var("LUMEN_TQ_BYPASS_ATTN")
            .map(|v| v == "1")
            .unwrap_or(false);
        let use_tq = !bypass_tq_attn
            && seq_len == 1
            && cumulative_kv_after >= tq_thresh
            && self.tq_slot.is_some()
            && compressed_kv.is_some();
        if use_tq {
            let slot = self.tq_slot.unwrap();
            let ckv = compressed_kv.as_mut().unwrap();
            if let Some(attn_out) = ckv.compressed_attention(
                slot,
                &queries,
                d.num_heads,
                d.num_kv_heads,
                d.head_dim,
            ) {
                // compressed_attention vs SDPA
                // single-step cos. Activated by `LUMEN_TQ_DEBUG_COS=1`. Uses
                // the full pre-GQA-repeat K/V from the vanilla KvCache (already
                // populated above at line 874-879) as ground truth — completely
                // bypasses TQ's compress/decompress for the reference. Computes
                // per-Q-head cos + max|Δ| against ref SDPA output to localise
                // exactly where TQ output diverges.
                if std::env::var("LUMEN_TQ_DEBUG_COS").map(|v| v == "1").unwrap_or(false) {
                    let _ = device.synchronize();
                    let group = d.gqa_group_size();
                    let k_full = repeat_kv_heads(&keys, group)?;
                    let v_full = repeat_kv_heads(&values, group)?;
                    let scale = (d.head_dim as f64).powf(-0.5);
                    let ref_scores = (queries.matmul(&k_full.transpose(D::Minus2, D::Minus1)?)? * scale)?;
                    let ref_out = candle_nn::ops::softmax_last_dim(&ref_scores)?.matmul(&v_full)?;
                    // Both [B=1, num_heads, S=1, head_dim].
                    let tq_vals = attn_out.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    let ref_vals = ref_out.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    let hd = d.head_dim;
                    let nh = d.num_heads;
                    let mut overall_dot = 0.0f64;
                    let mut overall_na = 0.0f64;
                    let mut overall_nb = 0.0f64;
                    let mut max_diff_overall = 0.0f32;
                    let mut head_cos: Vec<f32> = Vec::with_capacity(nh);
                    for h in 0..nh {
                        let lo = h * hd;
                        let hi = lo + hd;
                        let a = &tq_vals[lo..hi];
                        let b = &ref_vals[lo..hi];
                        let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
                        let mut md = 0.0f32;
                        for i in 0..hd {
                            let av = a[i] as f64;
                            let bv = b[i] as f64;
                            dot += av * bv; na += av * av; nb += bv * bv;
                            let dd = (a[i] - b[i]).abs();
                            if dd > md { md = dd; }
                        }
                        overall_dot += dot; overall_na += na; overall_nb += nb;
                        if md > max_diff_overall { max_diff_overall = md; }
                        let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
                        head_cos.push((dot / denom) as f32);
                    }
                    let overall_cos = overall_dot / (overall_na.sqrt() * overall_nb.sqrt()).max(1e-12);
                    let cos_min = head_cos.iter().cloned().fold(f32::INFINITY, f32::min);
                    let cos_max = head_cos.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let cos_mean = head_cos.iter().sum::<f32>() / nh as f32;
                    let n_kv = keys.dim(2)?;
                    let tq_max_abs = tq_vals.iter().cloned().map(f32::abs).fold(0.0f32, f32::max);
                    let ref_max_abs = ref_vals.iter().cloned().map(f32::abs).fold(0.0f32, f32::max);
                    let tq_nans = tq_vals.iter().filter(|v| !v.is_finite()).count();
                    eprintln!(
                        "[TQ-DEBUG] L{:02} n_kv={n_kv} cos_overall={:.4} cos_mean={:.4} cos_min={:.4} cos_max={:.4} max|Δ|={:.3e} tq_max|x|={:.3e} ref_max|x|={:.3e} tq_nan={tq_nans}",
                        self.layer_idx, overall_cos, cos_mean, cos_min, cos_max, max_diff_overall, tq_max_abs, ref_max_abs
                    );
                }
                // Collapse heads → flat hidden, apply gate/out-proj, return.
                let flat = attn_out
                    .transpose(1, 2)?
                    .contiguous()?
                    .reshape((batch, seq_len, d.num_heads * d.head_dim))?;
                let gated = if let Some(gate) = gate_flat {
                    (flat * candle_nn::ops::sigmoid(&gate)?)?
                } else {
                    flat
                };
                // Workstream B Phase 3 boundary cast (TQ fast-path mirror).
                // lift this cast under `LUMEN_BF16_RESIDUAL=1` so
                // the bf16 carrier reaches o_proj directly.
                let bf16_residual = super::moe::bf16_residual_enabled();
                let gated = if gated.dtype() != candle_core::DType::F32 && !bf16_residual {
                    gated.to_dtype(candle_core::DType::F32)?
                } else {
                    gated
                };
                // bf16 o_proj + cast back when env flag set.
                return self.apply_o_proj_with_optional_residual(&gated, residual);
            }
            // Fall through to SDPA if backend declined (layer not registered, etc.)
        }

        // ── 4d. Sliding-window attention: limit SDPA to last N tokens ──────
        // Breaks O(seq_len) growth on the 10 full-attn layers during decode.
        // Set LUMEN_FULL_ATTN_WINDOW=<N> (e.g. 512). 0 = disabled.
        // Linear-attn layers are unaffected (recurrent state keeps full context).
        static FULL_ATTN_WINDOW: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let attn_window = *FULL_ATTN_WINDOW.get_or_init(|| {
            std::env::var("LUMEN_FULL_ATTN_WINDOW")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0)
        });
        let kv_len = keys.dim(2)?;
        let window_trim = seq_len == 1 && attn_window > 0 && kv_len > attn_window;
        // StreamingLLM attention sink: keep first `sink` tokens in the window.
        // Sink > 0 only meaningful with `window_trim`; sink size must be < window.
        let attn_sink = if window_trim {
            let s = attn_sink_size();
            if s == 0 || s >= attn_window { 0 } else { s }
        } else {
            0
        };
        let (keys, values) = if window_trim {
            if attn_sink == 0 {
                let start = kv_len - attn_window;
                (keys.narrow(2, start, attn_window)?, values.narrow(2, start, attn_window)?)
            } else {
                // sink (positions 0..sink) + recent (kv_len - (window - sink)..kv_len)
                let recent = attn_window - attn_sink;
                let recent_start = kv_len - recent;
                let k_sink = keys.narrow(2, 0, attn_sink)?;
                let v_sink = values.narrow(2, 0, attn_sink)?;
                let k_recent = keys.narrow(2, recent_start, recent)?;
                let v_recent = values.narrow(2, recent_start, recent)?;
                (
                    Tensor::cat(&[&k_sink, &k_recent], 2)?,
                    Tensor::cat(&[&v_sink, &v_recent], 2)?,
                )
            }
        } else {
            (keys, values)
        };
        let mask_buf: Option<Tensor> = if window_trim {
            mask.and_then(|m| {
                let m_kv = m.dim(D::Minus1).ok()?;
                if m_kv <= attn_window {
                    return None;
                }
                if attn_sink == 0 {
                    m.narrow(D::Minus1, m_kv - attn_window, attn_window).ok()
                } else {
                    let recent = attn_window - attn_sink;
                    let m_sink = m.narrow(D::Minus1, 0, attn_sink).ok()?;
                    let m_recent = m.narrow(D::Minus1, m_kv - recent, recent).ok()?;
                    Tensor::cat(&[&m_sink, &m_recent], D::Minus1).ok()
                }
            })
        } else {
            None
        };
        let mask: Option<&Tensor> = if window_trim { mask_buf.as_ref().or(mask) } else { mask };

        // ── 5. GQA broadcast + scaled dot-product attention ────────────────
        let group = d.gqa_group_size();
        let scale = (d.head_dim as f64).powf(-0.5);

        // GQA in-kernel (LUMEN_FA_GQA_INKERNEL=1): pass pre-repeat K/V at
        // [B, num_kv_heads, S, D] directly to flash_attn. Saves one
        // `expand + reshape contiguous` per K/V (40 dispatches/decode step
        // across 10 full-attn layers) and shrinks resident K/V by `group×`.
        let gqa_inkernel = fa_gqa_inkernel_enabled();

        // ── Flash Attention 2 (fused single dispatch) ─────────────────────────
        // Falls back to 3-dispatch SDPA when flash_attn returns None (wrong
        // dtype, wrong head_dim, or LUMEN_DISABLE_FLASH_ATTN=1).
        #[cfg(feature = "turboquant-gpu")]
        let out = if gqa_inkernel {
            match lumen_metal::flash_attn::flash_attn_candle(
                &queries, &keys, &values, mask, scale as f32,
            ) {
                Some(result) => result?,
                None => {
                    // Fallback path needs full-rank K/V for matmul.
                    let keys_full = repeat_kv_heads(&keys, group)?;
                    let values_full = repeat_kv_heads(&values, group)?;
                    let mut scores =
                        (queries.matmul(&keys_full.transpose(D::Minus2, D::Minus1)?)? * scale)?;
                    if let Some(m) = mask {
                        scores = scores.broadcast_add(m)?;
                    }
                    candle_nn::ops::softmax_last_dim(&scores)?.matmul(&values_full)?
                }
            }
        } else {
            let keys = repeat_kv_heads(&keys, group)?;
            let values = repeat_kv_heads(&values, group)?;
            match lumen_metal::flash_attn::flash_attn_candle(
                &queries, &keys, &values, mask, scale as f32,
            ) {
                Some(result) => result?,
                None => {
                    let mut scores =
                        (queries.matmul(&keys.transpose(D::Minus2, D::Minus1)?)? * scale)?;
                    if let Some(m) = mask {
                        scores = scores.broadcast_add(m)?;
                    }
                    candle_nn::ops::softmax_last_dim(&scores)?.matmul(&values)?
                }
            }
        };

        #[cfg(not(feature = "turboquant-gpu"))]
        let out = {
            let _ = gqa_inkernel; // gate has no effect on the SDPA-only build.
            let keys = repeat_kv_heads(&keys, group)?;
            let values = repeat_kv_heads(&values, group)?;
            let mut scores =
                (queries.matmul(&keys.transpose(D::Minus2, D::Minus1)?)? * scale)?;
            if let Some(m) = mask {
                scores = scores.broadcast_add(m)?;
            }
            candle_nn::ops::softmax_last_dim(&scores)?.matmul(&values)?
        };
        if sa_timing {
            let _ = device.synchronize();
            sa_marks.push(("sdpa", std::time::Instant::now()));
        }

        // ── 6. Collapse heads, apply sigmoid gate, project out ─────────────
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, d.num_heads * d.head_dim))?;
        let out = if let Some(gate) = gate_flat {
            (out * candle_nn::ops::sigmoid(&gate)?)?
        } else {
            out
        };
        // Workstream B Phase 3 boundary cast: bf16 chain previously ended here.
        // Workstream B Phase 9 (2026-05-09): when `LUMEN_BF16_RESIDUAL=1` is
        // active, lift this cast — `out` stays bf16 down through o_proj and
        // the layer-level residual add. The single cast back to f32 (when
        // needed) happens at layer exit in `layer.rs`. Otherwise the legacy
        // f32 residual stream fires the cast here as before.
        let bf16_residual = super::moe::bf16_residual_enabled();
        let out = if out.dtype() != candle_core::DType::F32 && !bf16_residual {
            out.to_dtype(candle_core::DType::F32)?
        } else {
            out
        };
        // bf16 o_proj + cast back when env flag set.
        let candle_out = self.apply_o_proj_with_optional_residual(&out, residual)?;
        if sa_timing {
            let _ = device.synchronize();
            sa_marks.push(("o_proj", std::time::Instant::now()));
            let mut parts = String::new();
            for w in sa_marks.windows(2) {
                let label = w[1].0;
                let ms = w[1].1.duration_since(w[0].1).as_secs_f64() * 1000.0;
                parts.push_str(&format!("{label}={ms:.2} "));
            }
            eprintln!(
                "      sa-L{:02} (past={past_kv_len}, S={seq_len}): {parts}",
                self.layer_idx
            );
        }

        // Native parity probe: emit cosine + per-path ms (one line per probed layer).
        if let Some((native_out, native_ms)) = native_probe_out {
            report_native_parity(self.layer_idx, &candle_out, &native_out, native_ms);
        }

        Ok(candle_out)
    }

    /// Common o_proj application: applies bf16 cast-back if enabled, else
    /// f32 forward; when `residual` is `Some`, the layer-level post-attn add
    /// is folded INTO the matmul kernel via `forward_with_residual` whenever
    /// possible (MXFP4 weight + f32 path). Otherwise the residual is added
    /// post-hoc with `broadcast_add` so callers can rely on the contract:
    /// "residual is Some → output already includes the add."
    #[cfg(feature = "turboquant-gpu")]
    fn apply_o_proj_with_optional_residual(
        &self,
        x: &Tensor,
        residual: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let bf16 = super::moe::bf16_out_enabled();
        // Workstream B Phase 9 — bf16 residual stream branch. When the carrier
        // dtype is bf16 (input lifted by `LUMEN_BF16_RESIDUAL=1`), route
        // through `forward_bf16_in_bf16_out` and add the bf16 residual via
        // broadcast_add. The MXFP4-residual-fusion fast path is skipped
        // (its kernel is f32-only); BW savings come from staying bf16 across
        // the layer boundary instead.
        let bf16_in = x.dtype() == candle_core::DType::BF16;
        if bf16_in {
            return match residual {
                None => self.o_proj.forward_bf16_in_bf16_out(x),
                Some(res) => {
                    if res.dtype() != candle_core::DType::BF16 {
                        candle_core::bail!(
                            "bf16 residual stream: residual dtype {:?} != BF16",
                            res.dtype()
                        );
                    }
                    // Workstream B Phase 11 — fused bf16 matmul+residual fast
                    // path. Closes B.10 σ-NEGATIVE root cause: previous code
                    // here was `forward_bf16_in_bf16_out + broadcast_add(bf16)`,
                    // which split the f32 fused-residual fast path into 2
                    // dispatches once the carrier became bf16 (~+234 ms /
                    // decode batch in profile). The new fused kernel restores
                    // single-dispatch fusion on the bf16 chain.
                    self.o_proj.forward_with_residual_bf16_in_bf16_out(x, res)
                }
            };
        }
        match residual {
            None => {
                if bf16 {
                    let y_bf16 = self.o_proj.forward_bf16_out(x)?;
                    y_bf16.to_dtype(candle_core::DType::F32)
                } else {
                    self.o_proj.forward(x)
                }
            }
            Some(res) => {
                // Fast path: MXFP4 + f32 → fold residual into matmul kernel.
                if !bf16 && self.o_proj.is_mxfp4() && x.dtype() == candle_core::DType::F32 {
                    return self.o_proj.forward_with_residual(x, res);
                }
                // Fallback: legacy o_proj + broadcast_add. Honors the contract
                // even when fusion isn't applicable (no perf win, but safe).
                let y = if bf16 {
                    let y_bf16 = self.o_proj.forward_bf16_out(x)?;
                    y_bf16.to_dtype(candle_core::DType::F32)?
                } else {
                    self.o_proj.forward(x)?
                };
                y.broadcast_add(res)
            }
        }
    }

    /// CPU-only stub: residual is added post-hoc when present, mirroring the
    /// GPU path's contract.
    #[cfg(not(feature = "turboquant-gpu"))]
    fn apply_o_proj_with_optional_residual(
        &self,
        x: &Tensor,
        residual: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let y = self.o_proj.forward(x)?;
        match residual {
            Some(res) => y.broadcast_add(res),
            None => Ok(y),
        }
    }
}

#[cfg(feature = "turboquant-gpu")]
fn run_native_self_attn_probe(
    sa: &SelfAttention,
    x: &Tensor,
    pos_offset: usize,
) -> Option<(Tensor, f64)> {
    let res = match crate::qwen3_5_moe_native::shared_native_resources_for(x.device()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("    [native] resources init failed: {e}");
            return None;
        }
    };
    let guard = match res.lock() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("    [native] mutex poisoned: {e}");
            return None;
        }
    };
    let d = sa.runtime.dims;
    let cfg = crate::qwen3_5_moe_native::self_attn::SelfAttnConfig {
        num_heads: d.num_heads,
        num_kv_heads: d.num_kv_heads,
        head_dim: d.head_dim,
        rotary_dim: d.rotary_dim,
        rms_eps: sa.runtime.rms_norm_eps as f32,
        rope_theta: sa.runtime.rope_theta as f32,
        attn_output_gate: d.attn_output_gate,
        // Production model.rs invokes self_attn with `mask=None` → bidirectional.
        // Match that here so the parity probe compares like-for-like.
        apply_causal: false,
    };

    // (LUMEN_NATIVE_STAGE_DEBUG removed — replicating Mxfp4 forward inside the probe
    //  triggered SIGSEGV from Mxfp4Linear's transmuted Metal buffer being touched twice.
    //  Stage isolation now lives inside `qwen3_5_moe_native::forward_self_attn` itself,
    //  gated by `LUMEN_NATIVE_INTERNAL_DEBUG=1`.)

    let t0 = std::time::Instant::now();
    let result = crate::qwen3_5_moe_native::forward_self_attn(
        x,
        &sa.qkv_proj,
        sa.q_norm.weight(),
        sa.k_norm.weight(),
        &sa.o_proj,
        pos_offset,
        &cfg,
        &guard.ctx,
        &guard.lib,
    );
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    match result {
        Ok(t) => Some((t, ms)),
        Err(e) => {
            eprintln!("    [native] self_attn[layer={}] failed: {e}", sa.layer_idx);
            None
        }
    }
}

// Stage-by-stage parity helper kept around (compiled out by default) so future
// debugging sessions don't need to re-derive the boilerplate. Left disabled by
// `#[cfg(any())]` rather than deleted because the patterns (manual rmsnorm,
// flat_f32, cosine helpers in this scope) are tedious to re-author.
#[cfg(any())]
fn debug_stage_by_stage(
    sa: &SelfAttention,
    x: &Tensor,
    pos_offset: usize,
    res: &crate::qwen3_5_moe_native::NativeResources,
    cfg: &crate::qwen3_5_moe_native::self_attn::SelfAttnConfig,
) {
    use crate::qwen3_5_moe_native::bridge::{from_candle_tensor, to_candle_tensor};
    use crate::qwen3_5_moe_native::tensor::NativeDType;

    let device = x.device();
    let d = sa.runtime.dims;
    let (b, l, _) = match x.dims3() { Ok(t) => t, Err(_) => return };

    fn flat_f32(t: &Tensor) -> Vec<f32> {
        t.flatten_all()
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.to_vec1::<f32>())
            .unwrap_or_default()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
        let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
    }

    fn report(stage: &str, candle: &Tensor, native: &Tensor) {
        let cv = flat_f32(candle);
        let nv = flat_f32(native);
        if cv.len() != nv.len() {
            eprintln!(
                "    [stage] {stage}: SHAPE MISMATCH candle={:?} native={:?}",
                candle.dims(),
                native.dims()
            );
            return;
        }
        let cos = cosine(&cv, &nv);
        let max_abs = cv
            .iter()
            .zip(nv.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let n = cv.len().min(4);
        eprintln!(
            "    [stage] {stage}: cos={cos:.6} max_abs={max_abs:.4} c0_4={:?} n0_4={:?}",
            &cv[..n],
            &nv[..n]
        );
    }

    eprintln!("    [stage] === Layer 3 step-by-step parity (B={b}, L={l}) ===");

    // Stage 1: qkv_proj output
    let qkv_candle = match sa.qkv_proj.forward(x) {
        Ok(t) => t,
        Err(e) => { eprintln!("    [stage] qkv_proj failed: {e}"); return }
    };
    eprintln!(
        "    [stage] qkv_proj output: shape={:?} dtype={:?}",
        qkv_candle.dims(),
        qkv_candle.dtype()
    );

    // Stage 2: q/k/v split (same logic in both paths — just verify).
    let last = qkv_candle.dims().len() - 1;
    let q_section_dim = if cfg.attn_output_gate { 2 * d.num_heads * d.head_dim } else { d.num_heads * d.head_dim };
    let kv_out = d.num_kv_heads * d.head_dim;
    let (q_raw_candle, gate_flat_candle) = if cfg.attn_output_gate {
        let q_section = qkv_candle.narrow(last, 0, q_section_dim).unwrap();
        let q_split = q_section.reshape((b, l, d.num_heads, 2 * d.head_dim)).unwrap();
        let q = q_split.narrow(D::Minus1, 0, d.head_dim).unwrap().contiguous().unwrap();
        let g = q_split.narrow(D::Minus1, d.head_dim, d.head_dim).unwrap().contiguous().unwrap();
        let g_flat = g.reshape((b, l, d.num_heads * d.head_dim)).unwrap();
        (q, Some(g_flat))
    } else {
        let q = qkv_candle.narrow(last, 0, q_section_dim).unwrap()
            .reshape((b, l, d.num_heads, d.head_dim)).unwrap()
            .contiguous().unwrap();
        (q, None)
    };
    let k_raw_candle = qkv_candle.narrow(last, q_section_dim, kv_out).unwrap()
        .reshape((b, l, d.num_kv_heads, d.head_dim)).unwrap().contiguous().unwrap();

    // Stage 3: q_norm output. We compute the candle reference manually (sqr → mean → rsqrt
    // → mul) instead of calling `sa.q_norm.forward` because that path was triggering an
    // intermittent SIGSEGV under the probe (likely a buffer-lifetime interaction between
    // Mxfp4Linear's transmuted Metal buffer and the secondary RmsNorm kernel dispatch).
    let q_normed_candle = {
        let xf = match q_raw_candle.to_dtype(DType::F32).and_then(|t| t.contiguous()) {
            Ok(t) => t,
            Err(e) => { eprintln!("    [stage] candle q to f32 failed: {e}"); return }
        };
        let res_seq = xf
            .sqr()
            .and_then(|t| t.mean_keepdim(D::Minus1))
            .and_then(|t| (t + cfg.rms_eps as f64))
            .and_then(|t| t.sqrt())
            .and_then(|t| t.recip())
            .and_then(|scale| xf.broadcast_mul(&scale))
            .and_then(|t| t.broadcast_mul(sa.q_norm.weight()));
        match res_seq {
            Ok(t) => t,
            Err(e) => { eprintln!("    [stage] candle manual rmsnorm failed: {e}"); return }
        }
    };

    let rows_qk = b * l * d.num_heads;
    let q_2d = q_raw_candle
        .reshape((rows_qk, d.head_dim))
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.contiguous());
    let q_2d = match q_2d {
        Ok(t) => t,
        Err(e) => { eprintln!("    [stage] q_2d reshape failed: {e}"); return }
    };
    let gamma_q_candle = sa.q_norm.weight();
    let q_2d_native = match from_candle_tensor(&res.ctx, &q_2d) {
        Ok(t) => t, Err(e) => { eprintln!("    [stage] bridge q failed: {e}"); return }
    };
    let gamma_q_native = match from_candle_tensor(&res.ctx, gamma_q_candle) {
        Ok(t) => t, Err(e) => { eprintln!("    [stage] bridge gamma failed: {e}"); return }
    };
    // Verify native sees non-zero input.
    let q_2d_v = q_2d_native.to_vec_f32().unwrap_or_default();
    let gamma_v_n = gamma_q_native.to_vec_f32().unwrap_or_default();
    let q_2d_candle_v = flat_f32(&q_2d);
    eprintln!(
        "    [stage] bridge check: q_2d candle_first4={:?} native_first4={:?}",
        &q_2d_candle_v[..q_2d_candle_v.len().min(4)],
        &q_2d_v[..q_2d_v.len().min(4)],
    );
    eprintln!(
        "    [stage] bridge check: gamma candle_first4={:?} native_first4={:?}",
        &flat_f32(gamma_q_candle)[..4.min(gamma_q_candle.dims().last().copied().unwrap_or(1))],
        &gamma_v_n[..gamma_v_n.len().min(4)],
    );
    let q_normed_n_2d = match res.ctx.zeros(vec![rows_qk, d.head_dim], NativeDType::F32) {
        Ok(t) => t, Err(e) => { eprintln!("    [stage] alloc failed: {e}"); return }
    };
    if let Err(e) = res.lib.rms_norm(
        &res.ctx,
        &q_2d_native,
        &gamma_q_native,
        cfg.rms_eps,
        &q_normed_n_2d,
    ) {
        eprintln!("    [stage] native rms_norm failed: {e}"); return;
    }
    let q_normed_native = match to_candle_tensor(&q_normed_n_2d, device) {
        Ok(t) => match t.reshape(q_raw_candle.dims()) {
            Ok(t) => t,
            Err(e) => { eprintln!("    [stage] reshape failed: {e}"); return }
        },
        Err(e) => { eprintln!("    [stage] back-bridge failed: {e}"); return }
    };
    report("q_norm", &q_normed_candle, &q_normed_native);

    // Bonus: dump weight stats so we can see if gamma is what we expect.
    let gamma_v = flat_f32(gamma_q_candle);
    eprintln!(
        "    [stage] q_norm.weight: shape={:?} first4={:?} last4={:?} (eps={})",
        gamma_q_candle.dims(),
        &gamma_v[..gamma_v.len().min(4)],
        &gamma_v[gamma_v.len().saturating_sub(4)..],
        cfg.rms_eps,
    );
    let _ = (gate_flat_candle, k_raw_candle); // future stages
}

#[cfg(not(feature = "turboquant-gpu"))]
fn run_native_self_attn_probe(_: &SelfAttention, _: &Tensor, _: usize) -> Option<(Tensor, f64)> {
    None
}

/// Run the native self-attention pipeline as a **substitute** for the Candle path.
///
/// On success returns the post-`o_proj` attention output and seeds the KV cache
/// (if enabled) with the post-RoPE K and V tensors so future decode steps remain
/// consistent. On any failure returns `None` and the caller falls back to Candle.
///
/// Caller must verify `past_kv_len == 0` and `compressed_kv.is_none()` before
/// invoking — those guards live in `forward_with_tq` so the substitute stays a
/// single-call helper.
#[cfg(feature = "turboquant-gpu")]
fn run_native_self_attn_substitute(
    sa: &mut SelfAttention,
    x: &Tensor,
    pos_offset: usize,
) -> Option<Tensor> {
    let res = match crate::qwen3_5_moe_native::shared_native_resources_for(x.device()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("    [native-out] resources init failed: {e}");
            return None;
        }
    };
    let guard = match res.lock() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("    [native-out] mutex poisoned: {e}");
            return None;
        }
    };
    let d = sa.runtime.dims;
    let cfg = crate::qwen3_5_moe_native::self_attn::SelfAttnConfig {
        num_heads: d.num_heads,
        num_kv_heads: d.num_kv_heads,
        head_dim: d.head_dim,
        rotary_dim: d.rotary_dim,
        rms_eps: sa.runtime.rms_norm_eps as f32,
        rope_theta: sa.runtime.rope_theta as f32,
        attn_output_gate: d.attn_output_gate,
        // Production model.rs invokes self_attn with `mask=None` → bidirectional.
        apply_causal: false,
    };

    let result = crate::qwen3_5_moe_native::forward_self_attn_full(
        x,
        &sa.qkv_proj,
        sa.q_norm.weight(),
        sa.k_norm.weight(),
        &sa.o_proj,
        pos_offset,
        &cfg,
        &guard.ctx,
        &guard.lib,
    );
    drop(guard);

    let (out, k_bhld, v_bhld) = match result {
        Ok(t) => t,
        Err(e) => {
            eprintln!("    [native-out] self_attn[layer={}] failed: {e}", sa.layer_idx);
            return None;
        }
    };

    // Seed KV cache so downstream decode steps see correct past state. Failure
    // here means the cache is now inconsistent with the returned output, so we
    // surface the error rather than silently continuing.
    if let Some(cache) = sa.kv_caches.get_mut(&sa.current_seq_id) {
        let k_c = match k_bhld.contiguous() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("    [native-out] k_bhld contiguous failed: {e}");
                return None;
            }
        };
        let v_c = match v_bhld.contiguous() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("    [native-out] v_bhld contiguous failed: {e}");
                return None;
            }
        };
        if let Err(e) = cache.append(&k_c, &v_c) {
            eprintln!("    [native-out] cache.append failed: {e}");
            return None;
        }
    }
    Some(out)
}

#[cfg(not(feature = "turboquant-gpu"))]
fn run_native_self_attn_substitute(
    _: &mut SelfAttention,
    _: &Tensor,
    _: usize,
) -> Option<Tensor> {
    None
}

/// native substitute path that uses the layer-resident
/// [`NativeKvCache`]. Lazily allocates the cache on the first call and routes
/// **all** subsequent forward calls (prefill + decode) through the native
/// pipeline. Only fires when `LUMEN_NATIVE_OUTPUT=1` and no TurboQuant
/// compressor is attached — the dispatcher in `forward_with_tq` enforces both
/// preconditions.
///
/// Returns `None` on any failure so the caller falls back to Candle. Once the
/// native cache is populated and a Candle fallback runs instead, the two
/// caches diverge — that's the trade-off of opt-in routing. Don't toggle the
/// env flag mid-generation; reset the model between flag changes.
#[cfg(feature = "turboquant-gpu")]
fn run_native_self_attn_substitute_with_cache(
    sa: &mut SelfAttention,
    x: &Tensor,
) -> Option<Tensor> {
    let res = match crate::qwen3_5_moe_native::shared_native_resources_for(x.device()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("    [native-out-cache] resources init failed: {e}");
            return None;
        }
    };
    let guard = match res.lock() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("    [native-out-cache] mutex poisoned: {e}");
            return None;
        }
    };
    let d = sa.runtime.dims;
    let cfg = crate::qwen3_5_moe_native::self_attn::SelfAttnConfig {
        num_heads: d.num_heads,
        num_kv_heads: d.num_kv_heads,
        head_dim: d.head_dim,
        rotary_dim: d.rotary_dim,
        rms_eps: sa.runtime.rms_norm_eps as f32,
        rope_theta: sa.runtime.rope_theta as f32,
        attn_output_gate: d.attn_output_gate,
        // Production passes `mask=None` from `model.rs` → bidirectional attention.
        apply_causal: false,
    };

    let (b, _l, _hidden) = match x.dims3() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("    [native-out-cache] dims3 failed: {e}");
            return None;
        }
    };

    if sa.native_kv_cache.is_none() {
        let max_seq = match sa.native_kv_cache_max {
            Some(m) => m,
            None => return None,
        };
        let cache = match crate::qwen3_5_moe_native::NativeKvCache::new(
            &guard.ctx,
            b,
            d.num_kv_heads,
            d.head_dim,
            max_seq,
        ) {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "    [native-out-cache] NativeKvCache::new failed (b={b}, kv={}, d={}, max={max_seq}): {e}",
                    d.num_kv_heads, d.head_dim
                );
                return None;
            }
        };
        sa.native_kv_cache = Some(cache);
    }

    let cache = sa.native_kv_cache.as_mut().expect("just set");
    let result = crate::qwen3_5_moe_native::forward_self_attn_with_native_cache(
        x,
        &sa.qkv_proj,
        sa.q_norm.weight(),
        sa.k_norm.weight(),
        &sa.o_proj,
        &cfg,
        &guard.ctx,
        &guard.lib,
        cache,
    );
    drop(guard);

    match result {
        Ok(t) => Some(t),
        Err(e) => {
            eprintln!(
                "    [native-out-cache] self_attn[layer={}] failed: {e}",
                sa.layer_idx
            );
            None
        }
    }
}

#[cfg(not(feature = "turboquant-gpu"))]
fn run_native_self_attn_substitute_with_cache(
    _: &mut SelfAttention,
    _: &Tensor,
) -> Option<Tensor> {
    None
}

fn report_native_parity(layer_idx: usize, candle_out: &Tensor, native_out: &Tensor, native_ms: f64) {
    if candle_out.dims() != native_out.dims() {
        eprintln!(
            "    [native] self_attn[layer={layer_idx}] shape mismatch: candle={:?} native={:?}",
            candle_out.dims(),
            native_out.dims()
        );
        return;
    }
    let candle_v = match candle_out.flatten_all().and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.to_vec1::<f32>()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("    [native] self_attn[layer={layer_idx}] candle vec failed: {e}");
            return;
        }
    };
    let native_v = match native_out.flatten_all().and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.to_vec1::<f32>()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("    [native] self_attn[layer={layer_idx}] native vec failed: {e}");
            return;
        }
    };
    let dot: f64 = candle_v
        .iter()
        .zip(native_v.iter())
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum();
    let na: f64 = candle_v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = native_v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let cos = if na > 0.0 && nb > 0.0 { dot / (na * nb) } else { 0.0 };
    let max_abs = candle_v
        .iter()
        .zip(native_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "    [native] self_attn[layer={layer_idx}] cosine={cos:.6} max_abs={max_abs:.4} native_ms={native_ms:.1}"
    );
}

/// Lever E Option A (2026-04-28): pre-built RoPE cos/sin table over
/// `[0, max_seq_len)` positions. Each layer holds its own cache (lazy-init on
/// first forward); `narrow(0, pos_offset, seq_len)` produces the per-step view
/// without rebuilding `build_rope_table`'s ~7-dispatch chain.
///
/// Bit-identical invariant: the cache is built by calling
/// `build_rope_table(rotary_dim, max_seq_len, 0, theta, ...)`, so the cos/sin
/// values at row `pos_offset + t` of the cache equal the value at row `t` of a
/// fresh `build_rope_table(rotary_dim, seq_len, pos_offset, theta, ...)` —
/// `narrow` is a sub-view, not a recomputation.
#[derive(Debug)]
pub struct RopeCache {
    cos: Tensor,
    sin: Tensor,
    rotary_dim: usize,
    theta: f32,
    dtype: DType,
    max_seq_len: usize,
}

impl RopeCache {
    pub fn new(
        rotary_dim: usize,
        max_seq_len: usize,
        theta: f32,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        let (cos, sin) = build_rope_table(rotary_dim, max_seq_len, 0, theta, device, dtype)?;
        Ok(Self {
            cos,
            sin,
            rotary_dim,
            theta,
            dtype,
            max_seq_len,
        })
    }

    pub fn get(&self, seq_len: usize, pos_offset: usize) -> CandleResult<(Tensor, Tensor)> {
        let cos = self.cos.narrow(0, pos_offset, seq_len)?;
        let sin = self.sin.narrow(0, pos_offset, seq_len)?;
        Ok((cos, sin))
    }

    pub fn matches(&self, rotary_dim: usize, theta: f32, dtype: DType) -> bool {
        self.rotary_dim == rotary_dim && (self.theta - theta).abs() < 1e-9 && self.dtype == dtype
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

/// Build the cos/sin RoPE tables for `rotary_dim` per-head rotary features over
/// `pos_offset..pos_offset+seq_len`. Returns `([L, rotary_dim/2], [L, rotary_dim/2])`.
///
/// Formula mirrors mlx-lm's `mx.fast.rope(..., traditional=False)` which, with
/// `scaling_config=None`, resolves to `freqs[i] = base^(2i / rotary_dim)` and
/// `angle(t, i) = (t + offset) / freqs[i]`.
fn build_rope_table(
    rotary_dim: usize,
    seq_len: usize,
    pos_offset: usize,
    theta: f32,
    device: &Device,
    dtype: DType,
) -> CandleResult<(Tensor, Tensor)> {
    let half = rotary_dim / 2;
    // freqs: [half], inv_freqs = 1 / freqs
    let inv_freqs: Vec<f32> = (0..half)
        .map(|i| {
            let exp = (2 * i) as f32 / rotary_dim as f32;
            1.0 / theta.powf(exp)
        })
        .collect();
    let inv_freqs = Tensor::from_vec(inv_freqs, (half,), device)?.to_dtype(DType::F32)?;

    let positions: Vec<f32> = (0..seq_len).map(|t| (pos_offset + t) as f32).collect();
    let positions = Tensor::from_vec(positions, (seq_len,), device)?;

    // angles: [L, half]
    let angles = positions.unsqueeze(1)?.broadcast_mul(&inv_freqs.unsqueeze(0)?)?;
    let cos = angles.cos()?.to_dtype(dtype)?;
    let sin = angles.sin()?.to_dtype(dtype)?;
    Ok((cos, sin))
}

/// Apply non-traditional (GPT-NeoX split-form) RoPE to the first `rotary_dim` components of
/// each head. Remaining `head_dim - rotary_dim` components pass through unchanged.
fn apply_partial_rope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rotary_dim: usize,
) -> CandleResult<Tensor> {
    let head_dim = x.dim(D::Minus1)?;
    if rotary_dim == head_dim {
        return rotary_emb::rope(&x.contiguous()?, cos, sin);
    }
    let rot = x.narrow(D::Minus1, 0, rotary_dim)?.contiguous()?;
    let pass = x.narrow(D::Minus1, rotary_dim, head_dim - rotary_dim)?;
    let rot = rotary_emb::rope(&rot, cos, sin)?;
    Tensor::cat(&[&rot, &pass], D::Minus1)?.contiguous()
}

/// Repeat each KV head `group` times along the head axis (GQA expansion).
/// Input `[B, n_kv, L, D]` → output `[B, n_kv·group, L, D]` with `head_out[g·k + r]` sourced
/// from `head_in[k]` (row-major semantics match `torch.repeat_interleave`).
fn repeat_kv_heads(xs: &Tensor, group: usize) -> CandleResult<Tensor> {
    if group == 1 {
        return Ok(xs.clone());
    }
    let (b, n_kv, l, d) = xs.dims4()?;
    // [B, n_kv, 1, L, D] → [B, n_kv, group, L, D] → [B, n_kv·group, L, D]
    xs.unsqueeze(2)?
        .expand((b, n_kv, group, l, d))?
        .reshape((b, n_kv * group, l, d))
}

/// Additive causal mask: `-inf` strictly above diagonal, `0` elsewhere.
fn causal_mask(seq_len: usize, dtype: DType, device: &Device) -> CandleResult<Tensor> {
    let mut data = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (seq_len, seq_len), device)?.to_dtype(dtype)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen3_5_moe::config::Qwen3_5MoeConfig;

    const CONFIG_JSON: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qwen3_5_moe_config.json"
    ));

    fn dims_from_fixture() -> SelfAttnDims {
        let cfg: Qwen3_5MoeConfig = serde_json::from_str(CONFIG_JSON).unwrap();
        SelfAttnDims::from_config(&cfg.text_config).unwrap()
    }

    #[test]
    fn dims_match_config_fixture() {
        let d = dims_from_fixture();
        assert_eq!(d.hidden_size, 2048);
        assert_eq!(d.num_heads, 16);
        assert_eq!(d.num_kv_heads, 2);
        assert_eq!(d.head_dim, 256);
        assert!(d.attn_output_gate, "Qwen3-Next gated attention output");
        assert_eq!(d.rotary_dim, 64, "head_dim 256 × partial_rotary 0.25");
    }

    #[test]
    fn derived_widths_are_consistent() {
        let d = dims_from_fixture();
        assert_eq!(d.q_out_dim(), 8192, "gated: 2 × 16 × 256");
        assert_eq!(d.kv_out_dim(), 512, "GQA: 2 × 256");
        assert_eq!(d.attn_value_dim(), 4096, "16 × 256");
        assert_eq!(d.gqa_group_size(), 8, "16 / 2");
    }

    /// Real layer-3 `self_attn.*` shapes from `model-00001-of-00004.safetensors` (2026-04-23).
    /// MXFP4 packing rule: `.weight` last dim × 8 = logical last dim.
    fn canonical_mlx_shapes() -> SelfAttnShapes {
        fn unpack(packed: usize) -> usize {
            packed * 8
        }
        SelfAttnShapes {
            q_norm: vec![256],
            k_norm: vec![256],
            // q_proj header [8192, 256]  → logical [8192, 2048]
            q_proj: vec![8192, unpack(256)],
            // k_proj header [512, 256]   → logical [512, 2048]
            k_proj: vec![512, unpack(256)],
            // v_proj header [512, 256]   → logical [512, 2048]
            v_proj: vec![512, unpack(256)],
            // o_proj header [2048, 512]  → logical [2048, 4096]
            o_proj: vec![2048, unpack(512)],
        }
    }

    #[test]
    fn predicted_shapes_match_real_shard_header() {
        let predicted = dims_from_fixture().shapes();
        let canonical = canonical_mlx_shapes();
        assert_eq!(predicted.q_norm, canonical.q_norm);
        assert_eq!(predicted.k_norm, canonical.k_norm);
        assert_eq!(predicted.q_proj, canonical.q_proj);
        assert_eq!(predicted.k_proj, canonical.k_proj);
        assert_eq!(predicted.v_proj, canonical.v_proj);
        assert_eq!(predicted.o_proj, canonical.o_proj);
    }

    #[test]
    fn rotary_dim_adapts_to_partial_factor() {
        let cfg: Qwen3_5MoeConfig = serde_json::from_str(CONFIG_JSON).unwrap();
        let mut t = cfg.text_config.clone();
        t.partial_rotary_factor = 1.0;
        assert_eq!(SelfAttnDims::from_config(&t).unwrap().rotary_dim, 256);
        t.partial_rotary_factor = 0.5;
        assert_eq!(SelfAttnDims::from_config(&t).unwrap().rotary_dim, 128);
    }

    #[test]
    fn rejects_indivisible_gqa_group() {
        let cfg: Qwen3_5MoeConfig = serde_json::from_str(CONFIG_JSON).unwrap();
        let mut t = cfg.text_config.clone();
        t.num_key_value_heads = 3;
        let err = SelfAttnDims::from_config(&t).unwrap_err();
        assert!(matches!(
            err,
            SelfAttnDimsError::GqaGroupIndivisible { n_heads: 16, n_kv: 3 }
        ));
    }

    #[test]
    fn non_gated_path_halves_q_out_dim() {
        let cfg: Qwen3_5MoeConfig = serde_json::from_str(CONFIG_JSON).unwrap();
        let mut t = cfg.text_config.clone();
        t.attn_output_gate = false;
        let d = SelfAttnDims::from_config(&t).unwrap();
        assert_eq!(d.q_out_dim(), 4096, "without gate: 16 × 256");
    }

    /// Every [`SelfAttnPart`] must have a corresponding shape slot. The exhaustive match
    /// ensures adding a new variant triggers a compile error here.
    #[test]
    fn shape_slots_cover_every_self_attn_part() {
        use crate::qwen3_5_moe::weights::SelfAttnPart;
        let all_parts = [
            SelfAttnPart::QNorm,
            SelfAttnPart::KNorm,
            SelfAttnPart::QProj,
            SelfAttnPart::KProj,
            SelfAttnPart::VProj,
            SelfAttnPart::OProj,
        ];
        let shapes = dims_from_fixture().shapes();
        for part in all_parts {
            let _shape: &Vec<usize> = match part {
                SelfAttnPart::QNorm => &shapes.q_norm,
                SelfAttnPart::KNorm => &shapes.k_norm,
                SelfAttnPart::QProj => &shapes.q_proj,
                SelfAttnPart::KProj => &shapes.k_proj,
                SelfAttnPart::VProj => &shapes.v_proj,
                SelfAttnPart::OProj => &shapes.o_proj,
            };
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Forward-pass tests (synthetic weights).
    //
    // Numerical parity against the MLX fixture (`layer3_self_attn.safetensors`) is enforced
    // separately by `tests/self_attn_fixture.rs`, gated on the HF cache being present. The
    // tests below only check shapes, invariants, and path routing — they run in CI.
    // ─────────────────────────────────────────────────────────────────────

    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Linear, RmsNorm};
    use rand::{rngs::StdRng, RngExt, SeedableRng};

    /// Use a tiny config so tests stay fast on CPU. Shape invariants are the goal.
    fn tiny_dims(attn_output_gate: bool) -> SelfAttnDims {
        SelfAttnDims {
            hidden_size: 16,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            attn_output_gate,
            rotary_dim: 4,
        }
    }

    fn tiny_runtime(attn_output_gate: bool) -> SelfAttnRuntime {
        SelfAttnRuntime {
            dims: tiny_dims(attn_output_gate),
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
        }
    }

    fn random_tensor(shape: &[usize], rng: &mut StdRng, device: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| rng.random_range(-0.1..0.1)).collect();
        Tensor::from_vec(data, shape, device).unwrap()
    }

    fn build_tiny(runtime: SelfAttnRuntime, seed: u64, device: &Device) -> SelfAttention {
        let d = runtime.dims;
        let mut rng = StdRng::seed_from_u64(seed);
        // Option M2: pre-fused [q_out + 2*kv_out, hidden] qkv weight.
        let combined_out = d.q_out_dim() + 2 * d.kv_out_dim();
        let qkv = Linear::new(
            random_tensor(&[combined_out, d.hidden_size], &mut rng, device),
            None,
        );
        let o = Linear::new(
            random_tensor(&[d.hidden_size, d.attn_value_dim()], &mut rng, device),
            None,
        );
        let q_norm_w =
            Tensor::from_vec(vec![1f32; d.head_dim], (d.head_dim,), device).unwrap();
        let k_norm_w =
            Tensor::from_vec(vec![1f32; d.head_dim], (d.head_dim,), device).unwrap();
        SelfAttention::new(
            runtime,
            qkv.into(),
            o.into(),
            RmsNorm::new(q_norm_w, runtime.rms_norm_eps),
            RmsNorm::new(k_norm_w, runtime.rms_norm_eps),
        )
    }

    fn is_finite(t: &Tensor) -> bool {
        let flat = t
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        flat.iter().all(|v| v.is_finite())
    }

    #[test]
    fn gated_forward_returns_hidden_shape_and_is_finite() {
        let device = Device::Cpu;
        let mut attn = build_tiny(tiny_runtime(true), 0xABCD, &device);
        let mut rng = StdRng::seed_from_u64(0xFEED);
        let x = random_tensor(&[2, 5, 16], &mut rng, &device);
        let y = attn.forward(&x, 0, None).unwrap();
        assert_eq!(y.dims(), &[2, 5, 16]);
        assert!(is_finite(&y), "output should be finite");
    }

    #[test]
    fn non_gated_path_bypasses_sigmoid_mix() {
        let device = Device::Cpu;
        let mut attn = build_tiny(tiny_runtime(false), 0xBEEF, &device);
        let d = attn.dims();
        // Non-gated q_proj emits num_heads·head_dim only.
        assert_eq!(d.q_out_dim(), d.num_heads * d.head_dim);
        let mut rng = StdRng::seed_from_u64(0x1234);
        let x = random_tensor(&[1, 3, d.hidden_size], &mut rng, &device);
        let y = attn.forward(&x, 0, None).unwrap();
        assert_eq!(y.dims(), &[1, 3, d.hidden_size]);
        assert!(is_finite(&y));
    }

    #[test]
    fn rope_table_formula_matches_mlx_fast_rope() {
        // mlx-lm's nn.RoPE(dims=rotary_dim, base=theta, traditional=False) uses
        //   freqs[i] = base^(2i/dims), angle(t, i) = (t + offset) / freqs[i].
        // This test locks that formula down numerically for the first offset/position pairs
        // so the fixture test can trust the RoPE table and blame mismatches elsewhere.
        let device = Device::Cpu;
        let rotary_dim = 4usize;
        let theta = 10_000f32;
        let seq_len = 3;
        let offset = 5;
        let (cos, sin) = build_rope_table(
            rotary_dim,
            seq_len,
            offset,
            theta,
            &device,
            DType::F32,
        )
        .unwrap();
        assert_eq!(cos.dims(), &[seq_len, rotary_dim / 2]);
        let cos_v = cos.to_vec2::<f32>().unwrap();
        let sin_v = sin.to_vec2::<f32>().unwrap();
        for t in 0..seq_len {
            for i in 0..(rotary_dim / 2) {
                let freq = theta.powf((2 * i) as f32 / rotary_dim as f32);
                let angle = (t + offset) as f32 / freq;
                let expected_cos = angle.cos();
                let expected_sin = angle.sin();
                assert!(
                    (cos_v[t][i] - expected_cos).abs() < 1e-5,
                    "cos[{t}][{i}] = {} vs expected {}",
                    cos_v[t][i],
                    expected_cos
                );
                assert!(
                    (sin_v[t][i] - expected_sin).abs() < 1e-5,
                    "sin[{t}][{i}] = {} vs expected {}",
                    sin_v[t][i],
                    expected_sin
                );
            }
        }
    }

    #[test]
    fn rope_cache_narrow_matches_build_rope_table() {
        // Lever E Option A (2026-04-28): the cache is built by calling
        // build_rope_table over [0, max_seq_len); narrow(0, pos_offset, seq_len)
        // must produce values bit-identical to a fresh
        // build_rope_table(rotary_dim, seq_len, pos_offset). This test locks the
        // invariant down across multiple offset/length combos so future cache
        // refactors cannot silently drift.
        let device = Device::Cpu;
        let rotary_dim = 64usize; // production value (head_dim=256 × partial_rotary 0.25)
        let theta = 10_000_000f32; // production rope_theta from Qwen3.5-MoE config
        let max_seq = 256usize;
        let cache = RopeCache::new(rotary_dim, max_seq, theta, &device, DType::F32).unwrap();
        for &(seq_len, pos_offset) in &[
            (1usize, 0usize),
            (1, 7),
            (1, 100),
            (1, 255),
            (5, 12),
            (16, 0),
            (16, 240), // last legal slice
        ] {
            let (cos_cached, sin_cached) = cache.get(seq_len, pos_offset).unwrap();
            let (cos_ref, sin_ref) =
                build_rope_table(rotary_dim, seq_len, pos_offset, theta, &device, DType::F32)
                    .unwrap();
            assert_eq!(cos_cached.dims(), cos_ref.dims());
            let cos_diff = (&cos_cached - &cos_ref)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let sin_diff = (&sin_cached - &sin_ref)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            // Bit-identical expectation: same Candle ops, same input data, no
            // floating-point reduction differences. Allow 1 ULP slack at f32
            // precision in case a future Candle backend introduces fused
            // expressions, but enforce a tight ceiling.
            assert!(
                cos_diff <= f32::EPSILON,
                "rope_cache cos drift at seq_len={seq_len} pos_offset={pos_offset}: max|Δ|={cos_diff:.3e}"
            );
            assert!(
                sin_diff <= f32::EPSILON,
                "rope_cache sin drift at seq_len={seq_len} pos_offset={pos_offset}: max|Δ|={sin_diff:.3e}"
            );
        }
    }

    #[test]
    fn rope_cache_matches_predicate_changes_with_dtype() {
        let device = Device::Cpu;
        let cache = RopeCache::new(4, 8, 10_000.0, &device, DType::F32).unwrap();
        assert!(cache.matches(4, 10_000.0, DType::F32));
        assert!(!cache.matches(4, 10_000.0, DType::F16)); // different dtype
        assert!(!cache.matches(8, 10_000.0, DType::F32)); // different rotary_dim
        assert!(!cache.matches(4, 1_000_000.0, DType::F32)); // different theta
    }

    #[test]
    fn rope_offset_alters_table() {
        // Two offsets must yield different tables — otherwise the forward pass cannot depend
        // on positional information. We compare the tables directly (not the forward output,
        // which can be dominated by a near-uniform softmax under tiny synthetic weights).
        let device = Device::Cpu;
        let (cos0, _) = build_rope_table(4, 3, 0, 10_000.0, &device, DType::F32).unwrap();
        let (cos5, _) = build_rope_table(4, 3, 5, 10_000.0, &device, DType::F32).unwrap();
        let diff = (&cos0 - &cos5)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff > 1e-2, "offset 5 should noticeably rotate cos table; got {diff}");
    }

    #[test]
    fn partial_rope_preserves_non_rotated_tail() {
        // Sanity: `apply_partial_rope` must leave the trailing `head_dim - rotary_dim` slab
        // bit-identical. A silent contiguity bug here would wreck attention without NaNs.
        let device = Device::Cpu;
        let dims = tiny_dims(true); // head_dim=8, rotary_dim=4
        let (cos, sin) = build_rope_table(
            dims.rotary_dim,
            3,
            0,
            10_000.0,
            &device,
            DType::F32,
        )
        .unwrap();
        // Shape [B=1, H=1, L=3, D=8]
        let x_data: Vec<f32> = (0..(1 * 1 * 3 * 8)).map(|i| i as f32 / 10.0).collect();
        let x = Tensor::from_vec(x_data, (1, 1, 3, 8), &device).unwrap();
        let y = apply_partial_rope(&x, &cos, &sin, dims.rotary_dim).unwrap();
        let tail_x = x
            .narrow(D::Minus1, dims.rotary_dim, 8 - dims.rotary_dim)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let tail_y = y
            .narrow(D::Minus1, dims.rotary_dim, 8 - dims.rotary_dim)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(tail_x, tail_y, "non-rotated tail must be bit-identical");
    }

    #[test]
    fn causal_mask_has_zero_diagonal_and_neg_inf_above() {
        let device = Device::Cpu;
        let m = causal_mask(4, DType::F32, &device).unwrap();
        let v = m.to_vec2::<f32>().unwrap();
        for i in 0..4 {
            assert_eq!(v[i][i], 0.0);
            for j in (i + 1)..4 {
                assert!(v[i][j].is_infinite() && v[i][j] < 0.0);
            }
            for j in 0..i {
                assert_eq!(v[i][j], 0.0);
            }
        }
    }

    #[test]
    fn repeat_kv_heads_matches_repeat_interleave_semantics() {
        // Verify that head k of the input lands at slots [g·k .. g·k + g) of the output,
        // which is the semantic needed for GQA (each Q head within a group sees the same KV).
        let device = Device::Cpu;
        // [B=1, n_kv=2, L=1, D=3]
        let xs = Tensor::from_vec(
            vec![
                10f32, 11., 12., // kv head 0
                20., 21., 22., // kv head 1
            ],
            (1, 2, 1, 3),
            &device,
        )
        .unwrap();
        let out = repeat_kv_heads(&xs, 3).unwrap();
        assert_eq!(out.dims(), &[1, 6, 1, 3]);
        let flat = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            flat,
            vec![
                10., 11., 12., // head 0 (kv0)
                10., 11., 12., // head 1 (kv0)
                10., 11., 12., // head 2 (kv0)
                20., 21., 22., // head 3 (kv1)
                20., 21., 22., // head 4 (kv1)
                20., 21., 22., // head 5 (kv1)
            ]
        );
    }
}
