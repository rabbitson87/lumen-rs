//! Dimension derivation and tensor-shape validation for the Mamba2 / gated-delta
//! `linear_attn` sub-module (Qwen3-Next lineage).
//!
//! This module is intentionally scoped to **shapes only**. The SSM forward kernel (causal
//! conv1d + selective scan + output projection) lands as a subsequent atomic unit; doing it
//! here would ship untested numerical code since we have no reference fixture yet.
//!
//! ## What this file does
//! 1. Derive all internal widths from [`TextConfig`] (hidden, k/v head counts, head_dim,
//!    conv kernel).
//! 2. Predict the post-dequantization shape of every `linear_attn.*` tensor exactly as the
//!    MLX checkpoint stores it (MXFP4-packed U32 weights → logical fp32 shapes, plus the plain
//!    BF16 state weights).
//! 3. Verify those predictions against the real shard-header shapes fetched from
//!    `model-00001-of-00004.safetensors` for layer 0 (the `canonical_mlx_shapes` constant).
//!
//! ## Qwen3-Next linear_attn layout (as realized in the MLX MXFP4 checkpoint)
//! - `hidden = 2048`, `num_k_heads = 16`, `num_v_heads = 32`, `head_dim = 128`, `conv_kernel = 4`
//! - `k_dim = 16 × 128 = 2048`
//! - `v_dim = 32 × 128 = 4096`
//! - `qkv_dim = 2·k_dim + v_dim = 8192` (Q, K share K-head layout; V is wider)
//! - `in_proj_qkv: [qkv_dim, hidden]`, followed by depth-wise causal `conv1d: [qkv_dim, k, 1]`
//!   that convolves over the time axis before the SSM kernel consumes Q, K, V slices.
//! - `in_proj_z: [v_dim, hidden]` is the output-side sigmoid gate (value-dim wide).
//! - `in_proj_a`, `in_proj_b`, `A_log`, `dt_bias` all have shape `[num_v_heads]` on the
//!   per-head axis (one scalar per V head — the SSM state decay + delta-t parameters).
//! - `norm: [head_dim]` is the intra-block RMSNorm applied per-head.
//! - `out_proj: [hidden, v_dim]` projects the gated SSM output back to the residual stream.
//!
//! Cross-referencing these against the shard header is the only safe way to lock the interface
//! before writing the forward pass — naming conventions in Mamba2 variants shift frequently.

use candle_core::{D, DType, Device, Result as CandleResult, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module};

use super::config::TextConfig;
use super::proj::ProjLinear;

/// Scalar dimensions that fully determine every `linear_attn.*` tensor shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearAttnDims {
    pub hidden_size: usize,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    /// Per-head dimension. Qwen3.5-VL-MoE uses the same head_dim for K and V.
    pub head_dim: usize,
    /// Causal 1D convolution kernel size (typically 4).
    pub conv_kernel: usize,
}

impl LinearAttnDims {
    /// Build from config, enforcing the `linear_key_head_dim == linear_value_head_dim`
    /// invariant that the shipped checkpoint relies on.
    pub fn from_config(t: &TextConfig) -> Result<Self, DimsError> {
        if t.linear_key_head_dim != t.linear_value_head_dim {
            return Err(DimsError::UnequalHeadDims {
                key: t.linear_key_head_dim,
                value: t.linear_value_head_dim,
            });
        }
        Ok(Self {
            hidden_size: t.hidden_size,
            num_k_heads: t.linear_num_key_heads,
            num_v_heads: t.linear_num_value_heads,
            head_dim: t.linear_key_head_dim,
            conv_kernel: t.linear_conv_kernel_dim,
        })
    }

    /// K branch width (`num_k_heads × head_dim`).
    pub fn k_dim(self) -> usize {
        self.num_k_heads * self.head_dim
    }

    /// V branch width (`num_v_heads × head_dim`).
    pub fn v_dim(self) -> usize {
        self.num_v_heads * self.head_dim
    }

    /// Packed QKV width. Q and K share K-head layout, V is wider:
    /// `qkv_dim = 2·k_dim + v_dim`.
    pub fn qkv_dim(self) -> usize {
        2 * self.k_dim() + self.v_dim()
    }

    /// Per-V-head scalar count (one A_log / dt_bias / a / b entry per value head).
    pub fn per_vhead_dim(self) -> usize {
        self.num_v_heads
    }

    /// Logical (post-dequant) tensor shapes for every `linear_attn.*` weight in a single layer.
    pub fn shapes(self) -> LinearAttnShapes {
        LinearAttnShapes {
            a_log: vec![self.per_vhead_dim()],
            dt_bias: vec![self.per_vhead_dim()],
            conv1d_weight: vec![self.qkv_dim(), self.conv_kernel, 1],
            norm: vec![self.head_dim],
            in_proj_a: vec![self.per_vhead_dim(), self.hidden_size],
            in_proj_b: vec![self.per_vhead_dim(), self.hidden_size],
            in_proj_qkv: vec![self.qkv_dim(), self.hidden_size],
            in_proj_z: vec![self.v_dim(), self.hidden_size],
            out_proj: vec![self.hidden_size, self.v_dim()],
        }
    }
}

/// Expected logical shapes of the `linear_attn.*` weights in a single decoder layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearAttnShapes {
    pub a_log: Vec<usize>,
    pub dt_bias: Vec<usize>,
    pub conv1d_weight: Vec<usize>,
    pub norm: Vec<usize>,
    pub in_proj_a: Vec<usize>,
    pub in_proj_b: Vec<usize>,
    pub in_proj_qkv: Vec<usize>,
    pub in_proj_z: Vec<usize>,
    pub out_proj: Vec<usize>,
}

#[derive(Debug, thiserror::Error)]
pub enum DimsError {
    #[error(
        "linear_key_head_dim ({key}) != linear_value_head_dim ({value}); this loader assumes \
        equal per-head dimensions as in the shipped Qwen3.5-VL-MoE MXFP4 checkpoint"
    )]
    UnequalHeadDims { key: usize, value: usize },
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward pass
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime configuration for a single `GatedDeltaNet` block. Split from [`LinearAttnDims`]
/// because `rms_norm_eps` is a model-wide scalar that affects numerics but not shapes.
#[derive(Debug, Clone, Copy)]
pub struct GatedDeltaNetRuntime {
    pub dims: LinearAttnDims,
    pub rms_norm_eps: f64,
}

impl GatedDeltaNetRuntime {
    pub fn from_text_config(t: &TextConfig) -> Result<Self, DimsError> {
        Ok(Self {
            dims: LinearAttnDims::from_config(t)?,
            rms_norm_eps: t.rms_norm_eps as f64,
        })
    }
}

/// Rust port of `mlx_lm.models.qwen3_5.GatedDeltaNet` (Qwen3-Next gated-delta SSM variant).
///
/// Pre-dequantized Candle layers are injected at construction time — the caller (Stage 2-f
/// loader, or a fixture test) is responsible for MXFP4 dequantization and the per-storage
/// reshape quirks. Compared to the `qwen3_next.Qwen3NextGatedDeltaNet` class (packed `qkvz`
/// / `ba` inputs), this variant splits the four input projections into separate `in_proj_qkv`,
/// `in_proj_z`, `in_proj_b`, `in_proj_a` layers. **The shipped MXFP4 checkpoint uses this
/// variant** — see `dump_qwen3_5_moe_layer_weights.py::dump_linear_attn` for the mapping.
///
/// The forward pass follows `gated_delta_ops` (the ops-based reference path MLX falls back
/// to when the Metal kernel is unavailable or during training). That ops path is
/// algorithmically identical to the kernel, modulo f32 rounding order inside the SIMD
/// reduction. We target numerical parity within the measured bf16↔f32 floor (~6e-3 per
/// element), not bit exactness.
pub struct GatedDeltaNet {
    /// Option M (2026-04-25): qkv + z + b + a fused into one `[qkv_dim + v_dim + 2*Hv, hidden]`
    /// projection. The forward narrows this combined output back into the four sub-tensors.
    /// Saves 3 MXFP4 dispatches per layer × 30 linear_attn layers per token.
    in_proj_combined: ProjLinear,
    conv1d: Conv1d,
    /// `[num_v_heads]` — SSM state decay log-parameter (stored as-is, exponentiated inside `g`).
    a_log: Tensor,
    /// `[num_v_heads]` — delta-t bias.
    dt_bias: Tensor,
    /// `[head_v_dim]` — RMSNorm weight used by the final `RMSNormGated`.
    norm_weight: Tensor,
    out_proj: ProjLinear,
    runtime: GatedDeltaNetRuntime,

    /// Depth-wise conv1d history. `[B, kernel-1, conv_dim]` once populated.
    /// Lazily materialized on the first forward call so we don't need the batch
    /// size at construction time. Persists across `forward` calls → recurrent
    /// conv behavior matching MLX's `cache.state` on `Qwen3NextGatedDeltaNet`.
    conv_state: Option<Tensor>,
    /// SSM recurrent state. `[B, num_v_heads, head_dim (V), head_dim (K)]` f32.
    /// Also lazily allocated.
    ssm_state: Option<Tensor>,
    /// native SSM state. Populated lazily on first
    /// `LUMEN_LINEAR_ATTN_NATIVE=1` forward; once active, this carries the
    /// SSM recurrent state across decode steps as a Metal-resident buffer
    /// (no per-step `Tensor::clone` allocation). [`Self::reset_state`] keeps
    /// this in sync with `ssm_state`.
    #[cfg(feature = "turboquant-gpu")]
    native_ssm_state: Option<crate::qwen3_5_moe_native::NativeSsmState>,
    /// lazily-cached native copies of the per-layer constants
    /// (`dt_bias` and `exp(a_log)`). Populated on the first
    /// `LUMEN_LINEAR_ATTN_NATIVE=1` call and reused thereafter. Without this,
    /// every decoded token paid for two BF16→F32 conversions, an `exp`, and
    /// a `contiguous` per layer (~60 GPU dispatches/token across 30 linear
    /// layers). The tensors are pure constants of the layer weights, so the
    /// cache never invalidates.
    #[cfg(feature = "turboquant-gpu")]
    cached_native_weights: Option<CachedNativeWeights>,

    // ── Phase 1.5 per-sequence SSM state (lazy-swap) ────────────────────────
    //
    // Hot-path fields (`conv_state`, `ssm_state`, `native_ssm_state`) always hold
    // the recurrent state for `current_seq_id`. When the engine switches to a
    // different sequence, `set_current_seq_id` stashes the current state here and
    // restores the target sequence's state into the hot fields.
    current_seq_id: u64,
    stashed_seq_states: std::collections::HashMap<u64, SsmStateEntry>,
}

/// Captured recurrent state of a [`GatedDeltaNet`] layer for speculative
/// decoding rollback.
#[derive(Clone)]
pub struct GatedDeltaNetSnapshot {
    conv_state: Option<Tensor>,
    ssm_state: Option<Tensor>,
    #[cfg(feature = "turboquant-gpu")]
    native_ssm_snapshot: Option<crate::qwen3_5_moe_native::NativeSsmSnapshot>,
}

/// Stashed recurrent state for one inactive sequence (Phase 1.5 per-seq isolation).
/// Kept in `GatedDeltaNet::stashed_seq_states` for every sequence that is not
/// currently being processed. The hot-path fields (`conv_state`, `ssm_state`,
/// `native_ssm_state`) always hold the state for `current_seq_id`.
struct SsmStateEntry {
    conv_state: Option<Tensor>,
    ssm_state: Option<Tensor>,
    #[cfg(feature = "turboquant-gpu")]
    native_ssm_state: Option<crate::qwen3_5_moe_native::NativeSsmState>,
}

#[cfg(feature = "turboquant-gpu")]
struct CachedNativeWeights {
    dt_bias: crate::qwen3_5_moe_native::NativeTensor,
    exp_a_log: crate::qwen3_5_moe_native::NativeTensor,
    /// native depthwise conv1d weight, shape `[C, kernel]`.
    /// Built once by squeezing the singleton groups axis from
    /// `Conv1d::weight()` ( `[C, 1, kernel]`).
    conv1d_weight: crate::qwen3_5_moe_native::NativeTensor,
}

impl GatedDeltaNet {
    pub fn new(
        runtime: GatedDeltaNetRuntime,
        in_proj_combined: ProjLinear,
        conv1d: Conv1d,
        a_log: Tensor,
        dt_bias: Tensor,
        norm_weight: Tensor,
        out_proj: ProjLinear,
    ) -> Self {
        Self {
            in_proj_combined,
            conv1d,
            a_log,
            dt_bias,
            norm_weight,
            out_proj,
            runtime,
            conv_state: None,
            ssm_state: None,
            #[cfg(feature = "turboquant-gpu")]
            native_ssm_state: None,
            #[cfg(feature = "turboquant-gpu")]
            cached_native_weights: None,
            current_seq_id: 0,
            stashed_seq_states: std::collections::HashMap::new(),
        }
    }

    pub fn dims(&self) -> LinearAttnDims {
        self.runtime.dims
    }

    /// Drop any accumulated recurrent state — call at the start of a new request.
    pub fn reset_state(&mut self) {
        self.conv_state = None;
        self.ssm_state = None;
        #[cfg(feature = "turboquant-gpu")]
        {
            // Keep the buffer allocated; just zero it so the next prefill sees
            // a cold state without paying for re-allocation.
            if let Some(state) = self.native_ssm_state.as_mut() {
                if let Ok(mut res) =
                    crate::qwen3_5_moe_native::shared_native_resources().map(|m| m.lock())
                {
                    if let Ok(res) = res.as_mut() {
                        let _ = state.reset(&res.ctx);
                    }
                }
            }
        }
    }

    /// Capture all recurrent state for later [`restore_state`]. Used by
    /// speculative decoding to roll back after a verify-batch forward whose
    /// drafted tokens were partially or fully rejected.
    ///
    /// The returned snapshot owns a host-side copy of the native SSM buffer
    /// (so subsequent SSM steps don't pollute it) and `Tensor::clone()` of
    /// the conv/ssm fallback states (Candle Tensors are reference counted —
    /// later assignments replace, not mutate).
    pub fn snapshot_state(&self) -> CandleResult<GatedDeltaNetSnapshot> {
        #[cfg(feature = "turboquant-gpu")]
        let native_ssm_snapshot = match self.native_ssm_state.as_ref() {
            Some(s) => Some(
                s.snapshot()
                    .map_err(|e| candle_core::Error::Msg(format!("snapshot ssm: {e}")))?,
            ),
            None => None,
        };
        Ok(GatedDeltaNetSnapshot {
            conv_state: self.conv_state.clone(),
            ssm_state: self.ssm_state.clone(),
            #[cfg(feature = "turboquant-gpu")]
            native_ssm_snapshot,
        })
    }

    /// Restore a previously captured [`GatedDeltaNetSnapshot`].
    pub fn restore_state(&mut self, snap: &GatedDeltaNetSnapshot) -> CandleResult<()> {
        self.conv_state = snap.conv_state.clone();
        self.ssm_state = snap.ssm_state.clone();
        #[cfg(feature = "turboquant-gpu")]
        {
            match (
                snap.native_ssm_snapshot.as_ref(),
                self.native_ssm_state.as_mut(),
            ) {
                (Some(s), Some(state)) => state
                    .restore(s)
                    .map_err(|e| candle_core::Error::Msg(format!("restore ssm: {e}")))?,
                (Some(_), None) => {
                    return Err(candle_core::Error::Msg(
                        "restore_state: snapshot has native ssm but layer has none allocated"
                            .into(),
                    ));
                }
                (None, Some(state)) => {
                    // Caller had no native state when snapshot was taken; zero ours so
                    // post-restore behavior matches the captured moment.
                    if let Ok(mut res) =
                        crate::qwen3_5_moe_native::shared_native_resources().map(|m| m.lock())
                    {
                        if let Ok(res) = res.as_mut() {
                            state
                                .reset(&res.ctx)
                                .map_err(|e| candle_core::Error::Msg(format!("reset ssm: {e}")))?;
                        }
                    }
                }
                (None, None) => {}
            }
        }
        Ok(())
    }

    // ── Phase 1.5: per-sequence SSM state management ─────────────────────────

    /// Switch the active sequence. Stashes the current seq's hot-path state into
    /// `stashed_seq_states` and restores the target seq's state (or clears to None
    /// if the seq is new). No-op when `seq_id == current_seq_id`.
    pub fn set_current_seq_id(&mut self, seq_id: u64) {
        if seq_id == self.current_seq_id {
            return;
        }
        // Stash current state.
        let old_entry = SsmStateEntry {
            conv_state: self.conv_state.take(),
            ssm_state: self.ssm_state.take(),
            #[cfg(feature = "turboquant-gpu")]
            native_ssm_state: self.native_ssm_state.take(),
        };
        self.stashed_seq_states
            .insert(self.current_seq_id, old_entry);
        self.current_seq_id = seq_id;
        // Restore target seq's state (new seq starts with None).
        if let Some(entry) = self.stashed_seq_states.remove(&seq_id) {
            self.conv_state = entry.conv_state;
            self.ssm_state = entry.ssm_state;
            #[cfg(feature = "turboquant-gpu")]
            {
                self.native_ssm_state = entry.native_ssm_state;
            }
        } else {
            self.conv_state = None;
            self.ssm_state = None;
            #[cfg(feature = "turboquant-gpu")]
            {
                self.native_ssm_state = None;
            }
        }
    }

    /// Pre-register a sequence slot (Phase 1.5 lifecycle). Safe to call even if
    /// `seq_id` is already registered or is the current active seq.
    pub fn init_seq(&mut self, seq_id: u64) {
        if seq_id != self.current_seq_id {
            self.stashed_seq_states
                .entry(seq_id)
                .or_insert_with(|| SsmStateEntry {
                    conv_state: None,
                    ssm_state: None,
                    #[cfg(feature = "turboquant-gpu")]
                    native_ssm_state: None,
                });
        }
    }

    /// Release all SSM state for `seq_id`. If this is the currently active
    /// sequence, the hot-path fields are cleared (same as `reset_state`).
    pub fn remove_seq(&mut self, seq_id: u64) {
        if seq_id == self.current_seq_id {
            self.reset_state();
        } else {
            self.stashed_seq_states.remove(&seq_id);
        }
    }

    /// Forward pass. Mirrors `GatedDeltaNet.__call__` in mlx-lm 0.31.3 (`models/qwen3_5.py`).
    ///
    /// `x`: `[B, S, hidden_size]` (the `input_layernorm` output).
    /// `mask`: `[B, S]` boolean — MLX zeroes `qkv` at masked positions before the conv and
    ///   restores the pre-update state at masked positions inside the SSM loop. For prefill
    ///   without caching, MLX passes `None` (default for `layer.linear_attn(x)`) — this is
    ///   what the block-level fixture expects.
    ///
    /// Returns `[B, S, hidden_size]`. The SSM state is currently **not exposed** — this is
    /// prefill-only, matching self_attn. Decoding with a persisted state comes alongside
    /// Stage 5 cache wiring.
    pub fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>) -> CandleResult<Tensor> {
        self.forward_inner(x, None, mask)
    }

    /// Lever H Step 3 (2026-04-28): pre-attention RmsNorm fused into the
    /// in_proj_combined matmul. Reads RAW `x_raw` (un-normalized) plus the
    /// `input_layernorm` weight; the kernel computes RmsNorm internally.
    /// Caller must verify `has_mxfp4_in_proj() == true` — Dense fixtures use
    /// the unfused path.
    ///
    /// Mutually exclusive with `LUMEN_BF16_OUT*=1` (different in_proj
    /// kernel). When set the caller must keep
    /// `LUMEN_DISABLE_INPUT_RMSNORM_FUSION=1`.
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_with_input_rmsnorm(
        &mut self,
        x_raw: &Tensor,
        rms_weight: &Tensor,
        rms_eps: f32,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        self.forward_inner(x_raw, Some((rms_weight, rms_eps)), mask)
    }

    /// True iff the in_proj_combined holds MXFP4 packed weights — required
    /// for the Lever H Step 3 input-rmsnorm fusion path.
    pub fn has_mxfp4_in_proj(&self) -> bool {
        self.in_proj_combined.is_mxfp4()
    }

    /// True iff the in_proj_combined holds Affine4 4-bit packed weights.
    /// Sibling to `has_mxfp4_in_proj` for Workstream B bf16-rmsnorm gating —
    /// the bf16 activation flowing out of `MpsRmsNormBf16Out` is consumed by
    /// `Affine4Linear::forward_bf16_in` which routes through qmv_fast bf16
    /// (Workstream A) when shape qualifies.
    pub fn has_affine4_in_proj(&self) -> bool {
        self.in_proj_combined.as_affine4().is_some()
    }

    fn forward_inner(
        &mut self,
        x: &Tensor,
        input_rms: Option<(&Tensor, f32)>,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let (batch, seq_len, hidden) = x.dims3()?;
        let d = self.runtime.dims;
        if hidden != d.hidden_size {
            candle_core::bail!(
                "GatedDeltaNet input hidden {hidden} does not match config {}",
                d.hidden_size
            );
        }
        // Lever B L.3 (2026-04-28): when LUMEN_BF16_RMSNORM=1, the upstream
        // input_layernorm produces a bf16 tensor that we route through the
        // bf16-in in_proj_combined kernel.
        //
        // MLX baseline (qwen3_next.py + gated_delta.py)
        // runs the linear-attn block in input dtype throughout with exactly 3
        // f32 escapes inside this region:
        //   1. `compute_g`           (gated_delta.py:9-11) — A_log widened to f32
        //   2. SSM recurrent state   (gated_delta_kernel `StT=f32`)
        //   3. `_precise_swiglu`     (qwen3_next.py:18-22) — silu(z)*y in f32
        // All 3 are explicitly preserved below. The chain dtype `dtype` follows
        // `combined`'s dtype after in_proj — which itself depends on which
        // in_proj variant the bf16/GDN branching below selects.
        let bf16_in_path = x.dtype() == candle_core::DType::BF16;
        let device = x.device().clone();

        // Optional fine-grained timing: `LUMEN_LINEAR_ATTN_TIMING=1` breaks the forward into
        // 10 sub-blocks and reports per-section ms. Each marker syncs the device first so the
        // measurements reflect actual GPU execution, not just queue submission.
        let la_timing = std::env::var("LUMEN_LINEAR_ATTN_TIMING")
            .map(|v| v == "1")
            .unwrap_or(false);
        let mut marks: Vec<(&'static str, std::time::Instant)> = Vec::new();
        let sync_mark = |marks: &mut Vec<(&'static str, std::time::Instant)>,
                         label: &'static str| {
            if la_timing {
                let _ = device.synchronize();
                marks.push((label, std::time::Instant::now()));
            }
        };
        sync_mark(&mut marks, "start");

        // ── 1. Input projections ───────────────────────────────────────────
        // Option M: one fused matmul produces [..., qkv_dim + v_dim + 2*Hv]; narrow into
        // the four sub-tensors. `.contiguous()` after each narrow ensures downstream ops
        // hit the packed-layout fast path instead of the strided fallback.
        //
        // Lever H Step 3 (2026-04-28): when input-rmsnorm fusion is in effect
        // dispatch the rmsnorm-fused matmul kernel (raw x + ln_w + ln_eps).
        // This path requires MXFP4 in_proj_combined.
        // Otherwise Phase A.1.5 bf16 in_proj_combined + cast back when
        // LUMEN_BF16_OUT=1.
        let combined = match input_rms {
            #[cfg(feature = "turboquant-gpu")]
            Some((rms_w, rms_eps)) => match self.in_proj_combined.as_mxfp4() {
                Some(l) => l.forward_with_rmsnorm(x, rms_w, rms_eps).map_err(|e| {
                    candle_core::Error::Msg(format!(
                        "linear_attn in_proj forward_with_rmsnorm: {e}"
                    ))
                })?,
                None => candle_core::bail!(
                    "linear_attn input-rmsnorm fusion requires MXFP4 in_proj_combined"
                ),
            },
            #[cfg(not(feature = "turboquant-gpu"))]
            Some(_) => candle_core::bail!(
                "linear_attn input-rmsnorm fusion requires turboquant-gpu feature"
            ),
            None => {
                if bf16_in_path {
                    // route in_proj based on whether the GDN kernel
                    // path will downstream the activation in bf16 end-to-end.
                    //
                    // - GDN kernel ENABLED (`LUMEN_USE_GDN_KERNEL=1`):
                    //   keep combined bf16 → no cast cycles in the chain
                    //   `qkv_flat / z_flat / b_flat / a_flat / conv_out / q / k / v`
                    //   all stay bf16 down to the bf16 SSM kernel.
                    //
                    // - Ops fallback or native path:
                    //   the SSM ops loop / native kernel still want f32 q/k/v
                    //   for the f32 recurrent state (Escape #3). Keep the
                    //   existing bf16-in / f32-out matmul that pairs with the
                    //   upstream MpsRmsNormBf16Out — no chain regression.
                    //
                    // Both branches preserve MLX dtype policy; the split is
                    // purely a localization of where the bf16→f32 cast lives.
                    #[cfg(feature = "turboquant-gpu")]
                    {
                        if lumen_metal::gated_delta::is_enabled() {
                            self.in_proj_combined.forward_bf16_in_bf16_out(x)?
                        } else {
                            self.in_proj_combined.forward_bf16_in(x)?
                        }
                    }
                    #[cfg(not(feature = "turboquant-gpu"))]
                    {
                        candle_core::bail!(
                            "linear_attn bf16-in path requires turboquant-gpu feature"
                        )
                    }
                } else if super::moe::bf16_out_enabled() {
                    let y_bf16 = self.in_proj_combined.forward_bf16_out(x)?;
                    y_bf16.to_dtype(candle_core::DType::F32)?
                } else {
                    self.in_proj_combined.forward(x)?
                }
            }
        };
        // chain dtype follows in_proj output. When bf16_in_path is
        // active AND the bf16 GDN kernel branch was taken, this is bf16 and
        // the entire post-conv chain (qkv_flat / z_flat / b_flat / a_flat /
        // conv_out / q / k / v / g / beta / y) stays bf16 down to the SSM
        // kernel boundary. Otherwise this matches `forward_bf16_in`'s f32
        // output and the chain runs f32 (legacy behavior).
        let dtype = combined.dtype();
        let last = combined.dims().len() - 1;
        let qkv_dim = d.qkv_dim();
        let v_dim = d.v_dim();
        let hv = d.num_v_heads;
        let qkv_flat = combined.narrow(last, 0, qkv_dim)?.contiguous()?;
        let z_flat = combined.narrow(last, qkv_dim, v_dim)?.contiguous()?;
        let b_flat = combined.narrow(last, qkv_dim + v_dim, hv)?.contiguous()?;
        let a_flat = combined
            .narrow(last, qkv_dim + v_dim + hv, hv)?
            .contiguous()?;
        sync_mark(&mut marks, "in_projs");

        // ── 2. Optional mask on qkv before conv ───────────────────────────
        let qkv_flat = match mask {
            Some(m) => {
                // m: [B, S] bool/u8/f32 → expand to [B, S, 1] for broadcast zeroing.
                let m = m.unsqueeze(D::Minus1)?.to_dtype(dtype)?;
                qkv_flat.broadcast_mul(&m)?
            }
            None => qkv_flat,
        };

        // ── 3. Depth-wise causal conv1d over time ─────────────────────────
        //
        // `self.conv_state` carries the last `kernel-1` qkv rows from the previous
        // forward call. Prepending it lets decode-step inputs of length 1 compose
        // with the prior context without redoing the whole prefill.
        // On the very first forward (cold start) the state is lazily initialized
        // to zeros — matching MLX's `cache.conv_state = zeros([B, kernel-1, C])`.
        let conv_dim = d.qkv_dim();
        let conv_pad = d.conv_kernel - 1;
        let prev_conv_state = match &self.conv_state {
            Some(s) if s.dim(0)? == batch => s.clone(),
            _ => Tensor::zeros((batch, conv_pad, conv_dim), dtype, &device)?,
        };
        let conv_input = Tensor::cat(&[&prev_conv_state, &qkv_flat], 1)?; // [B, kernel-1+S, conv_dim]

        // Depth-wise causal conv1d. We deliberately avoid Candle's `Conv1d::forward` when
        // `groups == channels` because candle-core/src/conv.rs:193 implements that branch by
        // chunking the input into `groups` single-group conv1d calls. For qkv_dim=8192 that
        // becomes 8192 × (im2col + matmul) Metal dispatches per layer — ~350ms/layer on decode.
        //
        // Equivalent formulation: materialize the 4-wide kernel window via `narrow()` + stack,
        // then broadcast_mul + sum over the kernel axis. That collapses to ~7 Candle ops
        // regardless of channel count, and measures at ~0.3ms/layer (~1000× faster). Parity
        // with the default path is checked in `conv_matmul_path_matches_default_on_cpu`.
        //
        // Set `LUMEN_LINEAR_ATTN_CONV_LEGACY=1` to fall back to Candle's groups=C path
        // (kept as a rollback lever while the new path soaks).
        let conv_legacy = std::env::var("LUMEN_LINEAR_ATTN_CONV_LEGACY")
            .map(|v| v == "1")
            .unwrap_or(false);

        // when `LUMEN_CONV1D_NATIVE=1`, try
        // the fused depthwise conv1d + SiLU Metal kernel. Off by default while
        // we A/B-measure: the native path saves 7 Candle ops but adds one
        // cross-queue `from_candle_tensor` sync per layer, and the win depends
        // on whether the Candle dispatch overhead dominates the bridge cost.
        // Falls back to the Candle paths below on any pre-condition mismatch.
        #[cfg(feature = "turboquant-gpu")]
        let native_conv_out: Option<Tensor> = if !conv_legacy
            && mask.is_none()
            && batch == 1
            && std::env::var("LUMEN_CONV1D_NATIVE")
                .map(|v| v == "1")
                .unwrap_or(false)
        {
            self.try_native_conv1d(&conv_input, &device, seq_len)?
        } else {
            None
        };
        #[cfg(not(feature = "turboquant-gpu"))]
        let native_conv_out: Option<Tensor> = None;

        let conv_out = if let Some(out) = native_conv_out {
            out
        } else if conv_legacy {
            let conv_bcl = conv_input.transpose(1, 2)?.contiguous()?; // [B, conv_dim, kernel-1+S]
            let conv_out_bcl = self.conv1d.forward(&conv_bcl)?; // [B, conv_dim, S]
            let conv_out = conv_out_bcl.transpose(1, 2)?.contiguous()?; // [B, S, conv_dim]
            candle_nn::ops::silu(&conv_out)?
        } else {
            // conv_input: [B, pad+S, C]. Build [B, S, kernel, C] by sliding window.
            let mut slices = Vec::with_capacity(d.conv_kernel);
            for k in 0..d.conv_kernel {
                slices.push(conv_input.narrow(1, k, seq_len)?); // [B, S, C]
            }
            let windowed = Tensor::stack(&slices, 2)?; // [B, S, kernel, C]
            // conv1d weight: [C, 1, kernel] (per conv1d_from_mlx_weight). Drop the singleton
            // in_channels/group axis and transpose to [kernel, C] so broadcast aligns on the
            // last two dims of `windowed`.
            let w = self
                .conv1d
                .weight()
                .squeeze(1)? // [C, kernel]
                .transpose(0, 1)? // [kernel, C]
                .contiguous()?;
            // when the chain runs bf16 (windowed.dtype != weight.dtype),
            // cast the small weight to match input. broadcast_mul requires
            // matching dtypes; the weight is `[kernel, C]` = 4×8192 = 32 KB,
            // negligible vs the bf16 BW saving on `windowed` itself.
            let w = if w.dtype() == windowed.dtype() {
                w
            } else {
                w.to_dtype(windowed.dtype())?
            };
            let prod = windowed.broadcast_mul(&w.unsqueeze(0)?.unsqueeze(0)?)?;
            let conv_out = prod.sum(2)?; // [B, S, C]
            candle_nn::ops::silu(&conv_out)?
        };

        // Save the last `kernel-1` rows of the *unprocessed* `qkv_flat` (not `conv_out`)
        // for the next call — this matches the MLX cache semantics where the state is the
        // raw conv input history, so subsequent decodes pick up exactly where this call left off.
        let total_len = conv_pad + seq_len;
        let keep_start = total_len - conv_pad;
        // Rebuild from conv_input (prev_conv_state ++ qkv_flat) and slice.
        let new_conv_state = conv_input.narrow(1, keep_start, conv_pad)?.contiguous()?;
        self.conv_state = Some(new_conv_state);
        sync_mark(&mut marks, "conv1d");

        // ── 3.5. Native fused post-conv path (Phase A.8-C.5) ──────────────
        // Routes the entire post-conv pipeline (Q/K split, RMSNorm, gating,
        // repeat_heads, SSM loop, RMSNormGated, out_proj) through
        // `forward_post_conv_fused`. Saves ≈10 commits/layer × 30 linear-attn
        // layers per token ≈ 13ms of dispatch overhead on M3 Max. Verified
        // bit-identical vs candle path, +8.86% tok/s (18.80→20.47, σ=-23.21).
        // See `la_native_landed.md`.
        //
        // Default ON. Set `LUMEN_LINEAR_ATTN_NATIVE=0` for emergency revert.
        // Falls back to Candle automatically if mask is provided (native path
        // doesn't handle mask) or if `B != 1` (native is single-stream only).
        #[cfg(feature = "turboquant-gpu")]
        if mask.is_none()
            && batch == 1
            && std::env::var("LUMEN_LINEAR_ATTN_NATIVE")
                .map(|v| v != "0")
                .unwrap_or(true)
        {
            let pre_native_len = marks.len();
            if let Some(out) = self.try_native_post_conv(
                &conv_out, &z_flat, &b_flat, &a_flat, &device, &mut marks, la_timing,
            )? {
                if la_timing && marks.len() >= 2 {
                    let mut msg = String::from("    la(native):");
                    let mut total_ms = 0.0;
                    for pair in marks.windows(2) {
                        let (_, t0) = pair[0];
                        let (lbl, t1) = pair[1];
                        let dur = t1.duration_since(t0).as_secs_f64() * 1e3;
                        total_ms += dur;
                        msg.push_str(&format!(" {lbl}={dur:.2}"));
                    }
                    msg.push_str(&format!(" total={total_ms:.2}"));
                    eprintln!("{msg}");
                }
                // when `LUMEN_BF16_RESIDUAL=1` and
                // the chain runs bf16 (combined dtype = BF16), surface the
                // output as bf16 so the layer-level residual stream stays
                // bf16. The native fast paths return F32 (Dense out_proj or
                // MXFP4 fused tail); cast once here. No-op when the chain is
                // f32 or the flag is off.
                if super::moe::bf16_residual_enabled() && bf16_in_path && out.dtype() != DType::BF16
                {
                    return Ok(out.to_dtype(DType::BF16)?);
                }
                return Ok(out);
            }
            // None → drop any partial native marks, fall through to Candle.
            marks.truncate(pre_native_len);
        }

        // ── 4. Split q / k / v, reshape to [B, S, H, D] ───────────────────
        let k_dim = d.k_dim();
        let v_dim = d.v_dim();
        let q = conv_out.narrow(D::Minus1, 0, k_dim)?.reshape((
            batch,
            seq_len,
            d.num_k_heads,
            d.head_dim,
        ))?;
        let k = conv_out.narrow(D::Minus1, k_dim, k_dim)?.reshape((
            batch,
            seq_len,
            d.num_k_heads,
            d.head_dim,
        ))?;
        let v = conv_out.narrow(D::Minus1, 2 * k_dim, v_dim)?.reshape((
            batch,
            seq_len,
            d.num_v_heads,
            d.head_dim,
        ))?;

        // ── 5. QK-norm (weightless RMS on last axis, eps=1e-6) with inv_scale ──
        //   MLX applies `q = (inv_scale**2) * rms_norm(q, None, 1e-6)` and
        //   `k = inv_scale * rms_norm(k, None, 1e-6)`. The `1e-6` is MLX's hardcoded SSM
        //   rms-norm epsilon (distinct from the model-wide `rms_norm_eps`).
        let ssm_eps = 1e-6f64;
        let inv_scale = (d.head_dim as f64).powf(-0.5);
        let q = (weightless_rms_norm(&q, ssm_eps)?.affine(inv_scale * inv_scale, 0.0))?;
        let k = (weightless_rms_norm(&k, ssm_eps)?.affine(inv_scale, 0.0))?;
        sync_mark(&mut marks, "split_qknorm");

        // ── 6. gated_delta_update (ops path) ──────────────────────────────
        // beta = sigmoid(b) ; g = exp(-exp(A_log.f32) * softplus(a + dt_bias))
        //
        // MLX reference (gated_delta.py:9-11, mlx-lm 0.31.3 — Escape #2):
        //   @partial(mx.compile, shapeless=True)
        //   def compute_g(A_log, a, dt_bias):
        //       return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))
        // MLX widens ONLY `A_log` to f32 — `a + dt_bias` and softplus stay in
        // input dtype. We widen softplus output too (line below); slightly
        // broader f32 coverage but numerically equivalent.
        let beta = candle_nn::ops::sigmoid(&b_flat)?; // [B, S, Hv]
        // dt_bias is loaded as f32 (dequantized weight); cast to
        // chain dtype on the fly when bf16. dt_bias has only `num_v_heads`
        // (≤ 32) elements — cost is negligible vs the bf16 BW saving on a_flat.
        let dt_bias_chain = if self.dt_bias.dtype() == dtype {
            self.dt_bias.clone()
        } else {
            self.dt_bias.to_dtype(dtype)?
        };
        let a_plus_dt = a_flat.broadcast_add(&dt_bias_chain.reshape((1, 1, d.num_v_heads))?)?; // [B, S, Hv]
        let softplus_a = softplus(&a_plus_dt)?;
        let a_log_f32 = self.a_log.to_dtype(DType::F32)?.exp()?; // [Hv]
        let g = softplus_a
            .to_dtype(DType::F32)?
            .broadcast_mul(&a_log_f32.reshape((1, 1, d.num_v_heads))?)?
            .neg()?
            .exp()?
            .to_dtype(dtype)?; // [B, S, Hv]
        sync_mark(&mut marks, "gated_prep");

        // ── 7-8. SSM kernel path (env opt-in) ─────────────────────────────
        // When `LUMEN_USE_GDN_KERNEL=1`, fuse the entire 8-dispatch-per-step
        // Candle ops loop into a single Metal kernel dispatch. The kernel
        // takes q/k at [B,T,Hk,Dk] (un-repeated) and handles GQA indexing
        // internally — saves the `repeat_heads` cost too.
        #[cfg(feature = "turboquant-gpu")]
        if lumen_metal::gated_delta::is_enabled() {
            // MLX baseline (gated_delta.py — `gated_delta_kernel`, verified 2026-05-08):
            //   template=[("InT", q.dtype), ("StT", state.dtype), ...]
            //   InT ∈ {bf16, f32}; StT = f32 always (Escape #3 — recurrent state).
            //   I/O boundary in MLX bf16 mode: q/k/v/g/beta bf16, state f32, y bf16.
            //
            // pass q/k/v/g/beta in their NATIVE dtype.
            // The kernel dispatcher in lumen-metal::gated_delta picks the
            // bf16 or f32 variant based on input dtype. state stays f32 either
            // way. Eliminates the 5 cast dispatches × 36 linear-attn layers =
            // 180 extra dispatches per decode token that Phase B.3 had measured.
            let q_in = q.contiguous()?;
            let k_in = k.contiguous()?;
            let v_in = v.contiguous()?;
            let g_in = g.contiguous()?;
            let beta_in = beta.contiguous()?;
            let state_in = match &self.ssm_state {
                Some(s) if s.dim(0)? == batch => s.clone(),
                _ => Tensor::zeros(
                    (batch, d.num_v_heads, d.head_dim, d.head_dim),
                    DType::F32, // Escape #3 — recurrent state always f32
                    &device,
                )?,
            };
            if let Some(res) = lumen_metal::gated_delta::gated_delta_step_candle(
                &q_in, &k_in, &v_in, &g_in, &beta_in, &state_in,
            ) {
                let (y_4d, state_out) = res?; // [B, S, Hv, Dv] (input dtype), [B, Hv, Dv, Dk] f32
                self.ssm_state = Some(state_out);
                sync_mark(&mut marks, "ssm_kernel");

                // y_4d already matches `dtype` (input dtype) — kernel returns
                // matching I/O dtype. `to_dtype` is a no-op when bf16-bf16 or
                // f32-f32; preserves a safe path on dtype mismatch (e.g.,
                // future kernel changes).
                let y = if y_4d.dtype() == dtype {
                    y_4d
                } else {
                    y_4d.to_dtype(dtype)?
                };
                let z = z_flat.reshape((batch, seq_len, d.num_v_heads, d.head_dim))?;
                // candle_nn::ops::rms_norm requires input dtype to
                // match weight dtype. norm_weight is f32 (dequantized weight);
                // cast on the fly when chain runs bf16. norm_weight has only
                // `head_dim` elements (~256) — cost is trivial.
                let y_in = y.contiguous()?;
                let y_normed = if y_in.dtype() == self.norm_weight.dtype() {
                    candle_nn::ops::rms_norm(
                        &y_in,
                        &self.norm_weight,
                        self.runtime.rms_norm_eps as f32,
                    )?
                } else {
                    let w = self.norm_weight.to_dtype(y_in.dtype())?;
                    candle_nn::ops::rms_norm(&y_in, &w, self.runtime.rms_norm_eps as f32)?
                };
                // MLX reference (qwen3_next.py:18-22 `_precise_swiglu` — Escape #1):
                //   gate = nn.silu(gate.astype(mx.float32))
                //   x = x.astype(mx.float32)
                //   return (gate * x).astype(h.dtype)
                // Dtype-preserving boundary, internal f32 compute. MATCHES MLX.
                // Do NOT remove — this is one of the 3 f32 escapes inside this
                // region that MLX itself uses.
                let gated_f32 = (candle_nn::ops::silu(&z.to_dtype(DType::F32)?)?
                    * y_normed.to_dtype(DType::F32)?)?;
                let gated = gated_f32.to_dtype(dtype)?;
                sync_mark(&mut marks, "rms_norm_gated");

                let out_flat = gated.reshape((batch, seq_len, d.v_dim()))?;
                // residual stream is f32 by default. With
                // Phase B.9 (`LUMEN_BF16_RESIDUAL=1`) the bf16 carrier rides
                // through the layer-level residual stream — keep `out_flat` in
                // its native dtype and route the out_proj through the
                // bf16-in/bf16-out kernel.
                let bf16_residual_active = super::moe::bf16_residual_enabled()
                    && out_flat.dtype() == candle_core::DType::BF16;
                let out_flat =
                    if !bf16_residual_active && out_flat.dtype() != candle_core::DType::F32 {
                        out_flat.to_dtype(candle_core::DType::F32)?
                    } else {
                        out_flat
                    };
                let out = if bf16_residual_active {
                    self.out_proj.forward_bf16_in_bf16_out(&out_flat)?
                } else if super::moe::bf16_out_enabled() {
                    let y_bf16 = self.out_proj.forward_bf16_out(&out_flat)?;
                    y_bf16.to_dtype(candle_core::DType::F32)?
                } else {
                    self.out_proj.forward(&out_flat)?
                };
                sync_mark(&mut marks, "out_proj");

                if la_timing && marks.len() >= 2 {
                    let mut msg = String::from("    la(kernel):");
                    let mut total_ms = 0.0;
                    for pair in marks.windows(2) {
                        let (_, t0) = pair[0];
                        let (label, t1) = pair[1];
                        let ms = t1.duration_since(t0).as_secs_f64() * 1000.0;
                        total_ms += ms;
                        msg.push_str(&format!(" {label}={ms:.1}"));
                    }
                    eprintln!("{msg} total={total_ms:.1}ms (S={seq_len})");
                }
                return Ok(out);
            }
            // Kernel returned None (precondition failed) — fall through to ops path.
        }

        // ── 7. Repeat q, k along head axis to match Hv ────────────────────
        let q = repeat_heads(&q, d.num_v_heads / d.num_k_heads)?; // [B, S, Hv, Dk]
        let k = repeat_heads(&k, d.num_v_heads / d.num_k_heads)?; // [B, S, Hv, Dk]
        sync_mark(&mut marks, "repeat_heads");

        // Widen q/k/v/beta/g to f32 *outside* the per-timestep loop so we do one
        // dtype conversion per tensor per forward instead of 5·seq_len conversions.
        // Each `to_dtype` triggers a Candle command buffer + sync; on a 1270-matmul
        // decode this alone saved hundreds of ms on the original bf16 path.
        let q_f32 = q.to_dtype(DType::F32)?.contiguous()?;
        let k_f32 = k.to_dtype(DType::F32)?.contiguous()?;
        let v_f32 = v.to_dtype(DType::F32)?.contiguous()?;
        let g_f32 = g.to_dtype(DType::F32)?.contiguous()?;
        let beta_f32 = beta.to_dtype(DType::F32)?.contiguous()?;
        sync_mark(&mut marks, "dtype_conv");

        // ── 8. Sequential SSM loop ────────────────────────────────────────
        // Load the previous SSM state if any (decode step); otherwise cold-start from zeros.
        let mut state = match &self.ssm_state {
            Some(s) if s.dim(0)? == batch => s.clone(),
            _ => Tensor::zeros(
                (batch, d.num_v_heads, d.head_dim, d.head_dim),
                DType::F32,
                &device,
            )?,
        }; // [B, Hv, Dv, Dk]
        let mut y_steps: Vec<Tensor> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            // Pre-widened slices — no per-step `to_dtype` in the loop body.
            let q_t = q_f32.narrow(1, t, 1)?.squeeze(1)?; // [B, Hv, Dk]
            let k_t = k_f32.narrow(1, t, 1)?.squeeze(1)?; // [B, Hv, Dk]
            let v_t = v_f32.narrow(1, t, 1)?.squeeze(1)?; // [B, Hv, Dv]
            let g_t = g_f32.narrow(1, t, 1)?.squeeze(1)?; // [B, Hv]
            let beta_t = beta_f32.narrow(1, t, 1)?.squeeze(1)?; // [B, Hv]

            // state *= g[..., None, None]
            let decay = g_t.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?; // [B, Hv, 1, 1]
            state = state.broadcast_mul(&decay)?;

            // k_bc: [B, Hv, 1, Dk]  (broadcast against state's Dv axis)
            let k_bc = k_t.unsqueeze(D::Minus2)?;
            // kv_mem = sum(state * k_bc, dim=-1) → [B, Hv, Dv]
            let kv_mem = state.broadcast_mul(&k_bc)?.sum(D::Minus1)?;
            // delta = (v - kv_mem) * beta[..., None]
            let delta = (v_t - kv_mem)?.broadcast_mul(&beta_t.unsqueeze(D::Minus1)?)?;
            // state += k_bc * delta[..., None]   (outer product along Dk)
            let outer = k_bc.broadcast_mul(&delta.unsqueeze(D::Minus1)?)?; // [B, Hv, Dv, Dk]
            state = (state + outer)?;
            // y_t = sum(state * q[..., None, :], dim=-1) → [B, Hv, Dv]
            let q_bc = q_t.unsqueeze(D::Minus2)?; // [B, Hv, 1, Dk]
            let y_t = state.broadcast_mul(&q_bc)?.sum(D::Minus1)?;
            // Keep f32 through the loop; cast back to input dtype once at the end.
            y_steps.push(y_t);
        }
        // [B, S, Hv, Dv] in f32 — single cast here instead of seq_len casts above.
        let y = Tensor::stack(&y_steps, 1)?.to_dtype(dtype)?;

        // Persist the final SSM state so the next forward picks up where this one ended.
        self.ssm_state = Some(state);
        sync_mark(&mut marks, "ssm_loop");

        // ── 9. RMSNormGated: rms_norm(y, norm_weight) combined with silu(z) in f32 ──
        //   MLX's `Qwen3NextRMSNormGated` (qwen3_next.py:18-22 `_precise_swiglu` — Escape #1):
        //     x_rms = fast.rms_norm(y, weight, eps)
        //     return (silu(z.astype(f32)) * x_rms.astype(f32)).astype(y.dtype)
        //   This is one of MLX's 4 explicit f32 escapes — MATCHES MLX. Keep in Phase B.4.
        let z = z_flat.reshape((batch, seq_len, d.num_v_heads, d.head_dim))?;
        // norm_weight is f32; cast on the fly when chain is bf16
        // (same pattern as the kernel-path block above; head_dim ≈ 256 elems).
        let y_in = y.contiguous()?;
        let y_normed = if y_in.dtype() == self.norm_weight.dtype() {
            candle_nn::ops::rms_norm(&y_in, &self.norm_weight, self.runtime.rms_norm_eps as f32)?
        } else {
            let w = self.norm_weight.to_dtype(y_in.dtype())?;
            candle_nn::ops::rms_norm(&y_in, &w, self.runtime.rms_norm_eps as f32)?
        };
        let gated_f32 =
            (candle_nn::ops::silu(&z.to_dtype(DType::F32)?)? * y_normed.to_dtype(DType::F32)?)?;
        let gated = gated_f32.to_dtype(dtype)?;
        sync_mark(&mut marks, "rms_norm_gated");

        let out_flat = gated.reshape((batch, seq_len, d.v_dim()))?;
        // residual stream is f32 by default. Phase B.9
        // (`LUMEN_BF16_RESIDUAL=1`) lifts the cast and keeps the chain in
        // bf16 down through out_proj. No-op when ops-fallback ran f32 inputs
        // (legacy path).
        let bf16_residual_active =
            super::moe::bf16_residual_enabled() && out_flat.dtype() == candle_core::DType::BF16;
        let out_flat = if !bf16_residual_active && out_flat.dtype() != candle_core::DType::F32 {
            out_flat.to_dtype(candle_core::DType::F32)?
        } else {
            out_flat
        };
        let out = if bf16_residual_active {
            self.out_proj.forward_bf16_in_bf16_out(&out_flat)?
        } else if super::moe::bf16_out_enabled() {
            let y_bf16 = self.out_proj.forward_bf16_out(&out_flat)?;
            y_bf16.to_dtype(candle_core::DType::F32)?
        } else {
            self.out_proj.forward(&out_flat)?
        };
        sync_mark(&mut marks, "out_proj");

        if la_timing && marks.len() >= 2 {
            let mut msg = String::from("    la:");
            let mut total_ms = 0.0;
            for pair in marks.windows(2) {
                let (_, t0) = pair[0];
                let (label, t1) = pair[1];
                let ms = t1.duration_since(t0).as_secs_f64() * 1000.0;
                total_ms += ms;
                msg.push_str(&format!(" {label}={ms:.1}"));
            }
            eprintln!("{msg} total={total_ms:.1}ms (S={seq_len})");
        }
        Ok(out)
    }

    /// Run the post-conv pipeline through the native fused kernel
    /// (Phase A.8-C.5). Returns `Ok(None)` on init failure → caller
    /// transparently falls back to Candle.
    ///
    /// Lazy-allocates [`Self::native_ssm_state`] on first use, sized to the
    /// runtime's `num_v_heads × head_dim`. Subsequent calls reuse the same
    /// Metal buffer so the SSM state carries across decode steps without a
    /// per-step `Tensor::clone`.
    #[cfg(feature = "turboquant-gpu")]
    fn try_native_post_conv(
        &mut self,
        conv_out: &Tensor,
        z_flat: &Tensor,
        b_flat: &Tensor,
        a_flat: &Tensor,
        device: &Device,
        marks: &mut Vec<(&'static str, std::time::Instant)>,
        la_timing: bool,
    ) -> CandleResult<Option<Tensor>> {
        use crate::qwen3_5_moe_native::{
            LinearAttnConfig, NativeSsmState, forward_post_conv_fused_with_cache,
            forward_post_conv_fused_with_cache_candle_queue, from_candle_tensor,
            shared_native_resources_for,
        };

        // Sub-stage timing inside the native post-conv path. Each marker syncs
        // the device first so we measure actual GPU completion, not submission.
        // Stages: lock acquire | weight cache prep | fused kernel dispatch.
        let mark = |marks: &mut Vec<(&'static str, std::time::Instant)>, label: &'static str| {
            if la_timing {
                let _ = device.synchronize();
                marks.push((label, std::time::Instant::now()));
            }
        };

        let res_lock = match shared_native_resources_for(device) {
            Ok(m) => m,
            Err(_) => return Ok(None), // device not Metal-bound or context init failed
        };
        let res = match res_lock.lock() {
            Ok(g) => g,
            Err(_) => return Ok(None),
        };
        mark(marks, "n_lock");

        let d = self.runtime.dims;
        let cfg = LinearAttnConfig {
            hidden_size: d.hidden_size,
            num_k_heads: d.num_k_heads,
            num_v_heads: d.num_v_heads,
            head_dim: d.head_dim,
            conv_kernel: d.conv_kernel,
            rms_norm_eps: self.runtime.rms_norm_eps as f32,
            ssm_eps: 1e-6,
        };

        if self.native_ssm_state.is_none() {
            let st = match NativeSsmState::new(&res.ctx, 1, d.num_v_heads, d.head_dim, d.head_dim) {
                Ok(s) => s,
                Err(_) => return Ok(None),
            };
            self.native_ssm_state = Some(st);
        }

        // lazily cache `dt_bias`, `exp(a_log)`, and conv1d
        // weight as `NativeTensor`s once per layer. These are layer constants,
        // so the cache lives for the lifetime of the layer.
        if !self.ensure_cached_native_weights(&res.ctx) {
            return Ok(None);
        }
        let cached = self.cached_native_weights.as_ref().unwrap();
        let state = self.native_ssm_state.as_mut().unwrap();
        mark(marks, "n_wcache");

        // Lever D — encode the post-conv pipeline on Candle's command queue
        // instead of `NativeContext.queue`. Eliminates the cross-queue bridge
        // tax that anti-pattern #15 measured. **Default ON** post L.2 35B A/B
        // (σ=+442.99 STRONG WIN, Δ=-19.97 ms/token = -48.8%, decode 24.4 →
        // 47.7 tok/s, bit-identical 20/20 ALL). Opt-out for emergency revert
        // via `LUMEN_DISABLE_SSM_CANDLE_QUEUE=1`.
        let candle_queue_enabled = std::env::var("LUMEN_DISABLE_SSM_CANDLE_QUEUE")
            .map(|v| v == "0")
            .unwrap_or(true);
        let result = if candle_queue_enabled {
            forward_post_conv_fused_with_cache_candle_queue(
                conv_out,
                z_flat,
                b_flat,
                a_flat,
                &self.a_log,
                &self.dt_bias,
                &self.norm_weight,
                &self.out_proj,
                &cfg,
                &res.ctx,
                &res.lib,
                state,
                Some(&cached.dt_bias),
                Some(&cached.exp_a_log),
            )
        } else {
            forward_post_conv_fused_with_cache(
                conv_out,
                z_flat,
                b_flat,
                a_flat,
                &self.a_log,
                &self.dt_bias,
                &self.norm_weight,
                &self.out_proj,
                &cfg,
                &res.ctx,
                &res.lib,
                state,
                Some(&cached.dt_bias),
                Some(&cached.exp_a_log),
            )
        };
        let out = match result {
            Ok(t) => t,
            Err(e) => {
                // Don't poison the model on a transient failure — log + fallback.
                eprintln!("forward_post_conv_fused failed, falling back to Candle: {e}");
                return Ok(None);
            }
        };
        mark(marks, "n_kernel");
        Ok(Some(out))
    }

    /// Lazily build the per-layer native weight cache (`dt_bias`,
    /// `exp(a_log)`, depthwise conv1d weight). Idempotent: subsequent calls
    /// observe `is_some()` and return immediately. Returns `false` if any
    /// of the conversion / bridge steps fail (caller falls back to Candle).
    #[cfg(feature = "turboquant-gpu")]
    fn ensure_cached_native_weights(
        &mut self,
        ctx: &crate::qwen3_5_moe_native::NativeContext,
    ) -> bool {
        use crate::qwen3_5_moe_native::from_candle_tensor;
        if self.cached_native_weights.is_some() {
            return true;
        }
        let dt_bias_f32 = self
            .dt_bias
            .to_dtype(DType::F32)
            .and_then(|t| t.contiguous());
        let exp_a_log = self
            .a_log
            .to_dtype(DType::F32)
            .and_then(|t| t.exp())
            .and_then(|t| t.contiguous());
        let conv1d_weight = self
            .conv1d
            .weight()
            .squeeze(1)
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.contiguous());
        let (dt_bias_f32, exp_a_log, conv1d_w) = match (dt_bias_f32, exp_a_log, conv1d_weight) {
            (Ok(a), Ok(b), Ok(c)) => (a, b, c),
            _ => return false,
        };
        let dt_bias_native = match from_candle_tensor(ctx, &dt_bias_f32) {
            Ok(t) => t,
            Err(_) => return false,
        };
        let exp_a_log_native = match from_candle_tensor(ctx, &exp_a_log) {
            Ok(t) => t,
            Err(_) => return false,
        };
        let conv1d_native = match from_candle_tensor(ctx, &conv1d_w) {
            Ok(t) => t,
            Err(_) => return false,
        };
        self.cached_native_weights = Some(CachedNativeWeights {
            dt_bias: dt_bias_native,
            exp_a_log: exp_a_log_native,
            conv1d_weight: conv1d_native,
        });
        true
    }

    /// native conv1d + SiLU dispatch. Replaces the 8-op
    /// Candle path (4 narrows + stack + broadcast_mul + sum + silu) with a
    /// single `depthwise_conv1d_silu_f32` Metal kernel.
    ///
    /// Returns `Some(conv_out)` when the native path was used, `None` to
    /// signal the caller should fall through to the Candle path.
    ///
    /// Pre-conditions for the native path:
    ///   - `LUMEN_LINEAR_ATTN_NATIVE=1` (conv1d native rides on the same
    ///     gating env var as the post-conv path; they're a coupled win).
    ///   - `batch == 1` (matches `try_native_post_conv` constraint).
    ///   - `conv_input` device is Metal.
    #[cfg(feature = "turboquant-gpu")]
    fn try_native_conv1d(
        &mut self,
        conv_input: &Tensor,
        device: &Device,
        seq_len: usize,
    ) -> CandleResult<Option<Tensor>> {
        use crate::qwen3_5_moe_native::{
            from_candle_tensor, shared_native_resources_for, to_candle_tensor,
        };
        let res_lock = match shared_native_resources_for(device) {
            Ok(m) => m,
            Err(_) => return Ok(None),
        };
        let res = match res_lock.lock() {
            Ok(g) => g,
            Err(_) => return Ok(None),
        };
        if !self.ensure_cached_native_weights(&res.ctx) {
            return Ok(None);
        }
        let cached = self.cached_native_weights.as_ref().unwrap();
        let d = self.runtime.dims;
        let conv_dim = d.qkv_dim();

        // conv_input: [B, kernel-1+S, C] (the prev_conv_state ++ qkv_flat
        // concatenation that Candle just produced via `Tensor::cat`).
        let conv_input_f32 = match conv_input.dtype() {
            DType::F32 => conv_input.contiguous(),
            _ => conv_input.to_dtype(DType::F32).and_then(|t| t.contiguous()),
        };
        let conv_input_f32 = match conv_input_f32 {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };
        let x_native = match from_candle_tensor(&res.ctx, &conv_input_f32) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };
        let y_native = match res.ctx.zeros(
            vec![1, seq_len, conv_dim],
            crate::qwen3_5_moe_native::NativeDType::F32,
        ) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };
        if let Err(e) =
            res.lib
                .depthwise_conv1d_silu(&res.ctx, &x_native, &cached.conv1d_weight, &y_native)
        {
            eprintln!("depthwise_conv1d_silu failed, falling back: {e}");
            return Ok(None);
        }
        let conv_out = match to_candle_tensor(&y_native, device) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };
        Ok(Some(conv_out))
    }
}

/// `softplus(x) = ln(1 + exp(x))`, computed stably via `max(x, 0) + ln(1 + exp(-|x|))`.
fn softplus(x: &Tensor) -> CandleResult<Tensor> {
    // Stable form: see `torch.nn.functional.softplus`. For x > 20 this collapses to x within
    // f32 precision, but we keep the exact expression to stay faithful to MLX's `nn.softplus`.
    let zero = x.zeros_like()?;
    let pos = x.maximum(&zero)?;
    let abs_x = x.abs()?;
    let log1p_exp = (abs_x.neg()?.exp()? + 1.0)?.log()?;
    pos + log1p_exp
}

/// Weightless RMS norm on the last axis: `x / sqrt(mean(x², dim=-1, keepdim) + eps)`.
/// Mirrors `mx.fast.rms_norm(x, None, eps)`, used by the SSM Q/K normalizations.
fn weightless_rms_norm(x: &Tensor, eps: f64) -> CandleResult<Tensor> {
    let x_dtype = x.dtype();
    let internal = match x_dtype {
        DType::BF16 | DType::F16 => DType::F32,
        d => d,
    };
    let last = x.dim(D::Minus1)? as f64;
    let xf = x.to_dtype(internal)?;
    let mean_sq = (xf.sqr()?.sum_keepdim(D::Minus1)? / last)?;
    let denom = (mean_sq + eps)?.sqrt()?;
    xf.broadcast_div(&denom)?.to_dtype(x_dtype)
}

/// Repeat each head `repeats` times along axis 2 of a `[B, S, H, D]` tensor, matching
/// `mx.repeat(x, repeats, axis=-2)` semantics (consecutive = repeat_interleave).
fn repeat_heads(x: &Tensor, repeats: usize) -> CandleResult<Tensor> {
    if repeats == 1 {
        return Ok(x.clone());
    }
    let (b, s, h, d) = x.dims4()?;
    x.unsqueeze(3)?
        .expand((b, s, h, repeats, d))?
        .reshape((b, s, h * repeats, d))
}

/// Build a Candle `Conv1d` from an MLX-style `conv1d.weight` of shape `[channels, kernel, 1]`.
///
/// MLX's `nn.Conv1d(groups=channels)` stores the depth-wise weight as `[channels, kernel, 1]`
/// after `sanitize` (see `qwen3_5.py::TextModel.sanitize`). Candle's `Conv1d` expects
/// `[out_channels, in_channels/groups, kernel]` = `[channels, 1, kernel]`. We squeeze the
/// trailing 1 and transpose. The result is a drop-in depth-wise conv with `groups=channels`.
pub fn conv1d_from_mlx_weight(weight: Tensor, kernel: usize) -> CandleResult<Conv1d> {
    let dims = weight.dims();
    if dims.len() != 3 || dims[1] != kernel || dims[2] != 1 {
        candle_core::bail!(
            "conv1d weight must be shaped [channels, kernel={kernel}, 1]; got {:?}",
            dims
        );
    }
    let channels = dims[0];
    // [channels, kernel, 1] → squeeze last → [channels, kernel] → unsqueeze(1) → [channels, 1, kernel]
    let reshaped = weight.squeeze(D::Minus1)?.unsqueeze(1)?.contiguous()?;
    let cfg = Conv1dConfig {
        padding: 0,
        stride: 1,
        dilation: 1,
        groups: channels,
        cudnn_fwd_algo: None,
    };
    Ok(Conv1d::new(reshaped, None, cfg))
}

// (keep softplus / rms helpers tagged so dead-code lints stay quiet under cfg(test) only)
#[allow(dead_code)]
fn _unused_device_type_marker(_: &Device) {}

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

    fn dims_from_fixture() -> LinearAttnDims {
        let cfg: Qwen3_5MoeConfig = serde_json::from_str(CONFIG_JSON).unwrap();
        LinearAttnDims::from_config(&cfg.text_config).unwrap()
    }

    #[test]
    fn dims_match_config_fixture() {
        let d = dims_from_fixture();
        assert_eq!(d.hidden_size, 2048);
        assert_eq!(d.num_k_heads, 16);
        assert_eq!(d.num_v_heads, 32);
        assert_eq!(d.head_dim, 128);
        assert_eq!(d.conv_kernel, 4);
    }

    #[test]
    fn derived_widths_are_consistent() {
        let d = dims_from_fixture();
        assert_eq!(d.k_dim(), 2048, "16 × 128");
        assert_eq!(d.v_dim(), 4096, "32 × 128");
        assert_eq!(d.qkv_dim(), 8192, "2·2048 + 4096");
        assert_eq!(d.per_vhead_dim(), 32);
    }

    /// The real layer-0 `linear_attn.*` tensor shapes as read from the header of
    /// `model-00001-of-00004.safetensors` (HTTP Range request, 2026-04-23).
    ///
    /// MLX stores MXFP4 `.weight` as `u32[rows, cols/8]` — i.e. the last dim is packed at
    /// 8 nibbles per word. To reconstruct the logical (post-dequant) shape, multiply the
    /// last dim by 8. Plain BF16 weights are stored with their true shape.
    fn canonical_mlx_shapes() -> LinearAttnShapes {
        // Helper: take a packed MXFP4 weight storage shape [rows, cols_packed] and unpack
        // the column dimension. Leaves `rows` alone.
        fn unpack(packed_last_dim: usize) -> usize {
            packed_last_dim * 8
        }
        LinearAttnShapes {
            a_log: vec![32],
            dt_bias: vec![32],
            // conv1d is plain BF16 so the header shape [8192, 4, 1] is already logical.
            conv1d_weight: vec![8192, 4, 1],
            norm: vec![128],
            // All in_proj / out_proj weights are MXFP4-packed U32:
            //   in_proj_a  header: [32,   256] → logical [32,   2048]
            //   in_proj_b  header: [32,   256] → logical [32,   2048]
            //   in_proj_qkv header: [8192, 256] → logical [8192, 2048]
            //   in_proj_z  header: [4096, 256] → logical [4096, 2048]
            //   out_proj   header: [2048, 512] → logical [2048, 4096]
            in_proj_a: vec![32, unpack(256)],
            in_proj_b: vec![32, unpack(256)],
            in_proj_qkv: vec![8192, unpack(256)],
            in_proj_z: vec![4096, unpack(256)],
            out_proj: vec![2048, unpack(512)],
        }
    }

    #[test]
    fn predicted_shapes_match_real_shard_header() {
        let predicted = dims_from_fixture().shapes();
        let canonical = canonical_mlx_shapes();
        assert_eq!(predicted.a_log, canonical.a_log);
        assert_eq!(predicted.dt_bias, canonical.dt_bias);
        assert_eq!(predicted.conv1d_weight, canonical.conv1d_weight);
        assert_eq!(predicted.norm, canonical.norm);
        assert_eq!(predicted.in_proj_a, canonical.in_proj_a);
        assert_eq!(predicted.in_proj_b, canonical.in_proj_b);
        assert_eq!(predicted.in_proj_qkv, canonical.in_proj_qkv);
        assert_eq!(predicted.in_proj_z, canonical.in_proj_z);
        assert_eq!(predicted.out_proj, canonical.out_proj);
    }

    #[test]
    fn rejects_unequal_head_dims() {
        let mut cfg: Qwen3_5MoeConfig = serde_json::from_str(CONFIG_JSON).unwrap();
        cfg.text_config.linear_value_head_dim = 256;
        let err = LinearAttnDims::from_config(&cfg.text_config).unwrap_err();
        assert!(matches!(
            err,
            DimsError::UnequalHeadDims {
                key: 128,
                value: 256
            }
        ));
    }

    /// Cross-link: the `LinearAttnPart` set expected by the weights classifier must align
    /// one-to-one with the shape slots this module predicts. If either list ever drifts, the
    /// loader will hit a runtime surprise — enforce at compile/test time.
    #[test]
    fn shape_slots_cover_every_linear_attn_part() {
        use crate::qwen3_5_moe::weights::LinearAttnPart;
        let all_parts = [
            LinearAttnPart::ALog,
            LinearAttnPart::DtBias,
            LinearAttnPart::Conv1d,
            LinearAttnPart::Norm,
            LinearAttnPart::InProjA,
            LinearAttnPart::InProjB,
            LinearAttnPart::InProjQkv,
            LinearAttnPart::InProjZ,
            LinearAttnPart::OutProj,
        ];
        let shapes = dims_from_fixture().shapes();
        // Each arm below must reference exactly one shape slot; the `_` catch-all is absent on
        // purpose so adding a new variant becomes a compile error here.
        for part in all_parts {
            let _shape: &Vec<usize> = match part {
                LinearAttnPart::ALog => &shapes.a_log,
                LinearAttnPart::DtBias => &shapes.dt_bias,
                LinearAttnPart::Conv1d => &shapes.conv1d_weight,
                LinearAttnPart::Norm => &shapes.norm,
                LinearAttnPart::InProjA => &shapes.in_proj_a,
                LinearAttnPart::InProjB => &shapes.in_proj_b,
                LinearAttnPart::InProjQkv => &shapes.in_proj_qkv,
                LinearAttnPart::InProjZ => &shapes.in_proj_z,
                LinearAttnPart::OutProj => &shapes.out_proj,
            };
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Forward-pass tests (synthetic weights).
    //
    // MLX-level numerical parity on the real checkpoint is gated in
    // `tests/linear_attn_fixture.rs`. The tests here lock in shape invariants,
    // softplus/RMS helper correctness, and head-repeat semantics.
    // ─────────────────────────────────────────────────────────────────────

    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Conv1d, Linear};
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    fn tiny_dims() -> LinearAttnDims {
        // Dk = Dv = 4; Hk = 2; Hv = 4 → repeat_factor = 2.
        LinearAttnDims {
            hidden_size: 8,
            num_k_heads: 2,
            num_v_heads: 4,
            head_dim: 4,
            conv_kernel: 4,
        }
    }

    fn tiny_runtime() -> GatedDeltaNetRuntime {
        GatedDeltaNetRuntime {
            dims: tiny_dims(),
            rms_norm_eps: 1e-6,
        }
    }

    fn rnd(shape: &[usize], rng: &mut StdRng, device: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| rng.random_range(-0.1..0.1)).collect();
        Tensor::from_vec(data, shape, device).unwrap()
    }

    fn build_tiny(seed: u64, device: &Device) -> GatedDeltaNet {
        let d = tiny_dims();
        let mut r = StdRng::seed_from_u64(seed);
        // Option M: pre-fused [qkv_dim + v_dim + 2*Hv, hidden] in_proj.
        let combined_out = d.qkv_dim() + d.v_dim() + 2 * d.num_v_heads;
        let in_proj = Linear::new(rnd(&[combined_out, d.hidden_size], &mut r, device), None);
        let conv_w = rnd(&[d.qkv_dim(), d.conv_kernel, 1], &mut r, device);
        let conv = conv1d_from_mlx_weight(conv_w, d.conv_kernel).unwrap();
        let a_log = rnd(&[d.num_v_heads], &mut r, device);
        let dt_bias = rnd(&[d.num_v_heads], &mut r, device);
        let norm_w = Tensor::from_vec(vec![1f32; d.head_dim], (d.head_dim,), device).unwrap();
        let out = Linear::new(rnd(&[d.hidden_size, d.v_dim()], &mut r, device), None);
        GatedDeltaNet::new(
            tiny_runtime(),
            in_proj.into(),
            conv,
            a_log,
            dt_bias,
            norm_w,
            out.into(),
        )
    }

    fn is_finite(t: &Tensor) -> bool {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite())
    }

    #[test]
    fn forward_returns_hidden_shape_and_is_finite() {
        let device = Device::Cpu;
        let mut net = build_tiny(0xCAFE, &device);
        let mut r = StdRng::seed_from_u64(0xBEEF);
        let x = rnd(&[1, 5, 8], &mut r, &device);
        let y = net.forward(&x, None).unwrap();
        assert_eq!(y.dims(), &[1, 5, 8]);
        assert!(is_finite(&y));
    }

    #[test]
    fn mask_zeroing_masks_qkv_positions() {
        // A zero mask should drive conv1d output to zero (since the conv kernel only sees
        // zero-padding + zeroed qkv) and — because silu(0)=0, q=k=v=0 → state stays zero
        // → y=0 → rms_norm of zeros is 0 / sqrt(eps) = 0 → out_proj(0) + silu(z)*0 = out_proj(0).
        // Since out_proj has no bias, the final result must be bit-zero.
        let device = Device::Cpu;
        let mut net = build_tiny(0x1234, &device);
        let mut r = StdRng::seed_from_u64(0x5678);
        let x = rnd(&[1, 3, 8], &mut r, &device);
        let mask = Tensor::zeros((1, 3), DType::F32, &device).unwrap();
        let y = net.forward(&x, Some(&mask)).unwrap();
        // With a zeroing mask: qkv=0 → conv(0-pad + 0)=0 (pure depthwise, no bias) → q=k=v=0.
        // z is NOT zeroed by the mask (only qkv is). But silu(z)*rms_norm(0) = silu(z)*0 = 0.
        // So the entire y_normed=0 → gated=0 → out_proj(0)=0 (Linear has no bias in this net).
        let y_abs_max = y
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            y_abs_max < 1e-5,
            "zero mask should drive output to ~0; got max_abs = {y_abs_max}"
        );
    }

    #[test]
    fn weightless_rms_norm_matches_formula() {
        // Check against the analytic formula on a hand-crafted vector.
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let eps = 1e-6f64;
        let y = weightless_rms_norm(&x, eps)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        // mean_sq = (1 + 4 + 9 + 16) / 4 = 7.5; denom = sqrt(7.5 + 1e-6) ≈ 2.7386128
        let denom = (7.5f32 + 1e-6f32).sqrt();
        for (got, x) in y[0].iter().zip([1f32, 2.0, 3.0, 4.0].iter()) {
            let expected = x / denom;
            assert!(
                (got - expected).abs() < 1e-5,
                "weightless_rms_norm: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn softplus_matches_reference_values() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![-4f32, -1.0, 0.0, 1.0, 4.0], (5,), &device).unwrap();
        let y = softplus(&x).unwrap().to_vec1::<f32>().unwrap();
        for (got, &xi) in y.iter().zip([-4f32, -1.0, 0.0, 1.0, 4.0].iter()) {
            // Reference: ln(1 + exp(x)).
            let expected = (1.0 + xi.exp()).ln();
            assert!(
                (got - expected).abs() < 1e-5,
                "softplus({xi}): got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn repeat_heads_duplicates_consecutively() {
        let device = Device::Cpu;
        // [B=1, S=1, H=2, D=3] with rows 10..12 and 20..22
        let xs =
            Tensor::from_vec(vec![10f32, 11., 12., 20., 21., 22.], (1, 1, 2, 3), &device).unwrap();
        let out = repeat_heads(&xs, 2).unwrap();
        assert_eq!(out.dims(), &[1, 1, 4, 3]);
        let flat = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            flat,
            vec![
                10., 11., 12., // h0 (src 0)
                10., 11., 12., // h1 (src 0 repeated)
                20., 21., 22., // h2 (src 1)
                20., 21., 22., // h3 (src 1 repeated)
            ]
        );
    }

    /// The default broadcast_mul+sum depth-wise conv path must produce the same output as
    /// Candle's legacy `Conv1d(groups=channels)` path (toggled via
    /// `LUMEN_LINEAR_ATTN_CONV_LEGACY=1`). We run both on the same tiny synthetic net and
    /// assert the final GatedDeltaNet output matches within an f32 epsilon. Guards against
    /// rewrite drift (off-by-one window, missing silu, wrong broadcast axis) that a full
    /// end-to-end decode diff would detect late.
    #[test]
    fn conv_matmul_path_matches_legacy_on_cpu() {
        let device = Device::Cpu;
        let mut net_a = build_tiny(0x9001, &device);
        let mut net_b = build_tiny(0x9001, &device);
        let mut r = StdRng::seed_from_u64(0x9002);
        let x = rnd(&[1, 5, 8], &mut r, &device);

        // Rust 2024: std::env::{set_var,remove_var} are now `unsafe` because they race
        // with concurrent getenv. This test does not spawn threads, so the mutation is safe.
        unsafe {
            std::env::remove_var("LUMEN_LINEAR_ATTN_CONV_LEGACY");
        }
        let y_new = net_a.forward(&x, None).unwrap();

        unsafe {
            std::env::set_var("LUMEN_LINEAR_ATTN_CONV_LEGACY", "1");
        }
        let y_legacy = net_b.forward(&x, None).unwrap();
        unsafe {
            std::env::remove_var("LUMEN_LINEAR_ATTN_CONV_LEGACY");
        }

        let a = y_new.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = y_legacy.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a.len(), b.len());
        let max_abs_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 1e-5,
            "new conv path drifted from legacy Candle Conv1d; max_abs_diff = {max_abs_diff}"
        );
    }

    #[test]
    fn conv1d_from_mlx_weight_produces_depthwise_layout() {
        let device = Device::Cpu;
        // [channels=3, kernel=2, 1] — MLX sanitize layout.
        let w = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (3, 2, 1), &device).unwrap();
        let conv: Conv1d = conv1d_from_mlx_weight(w, 2).unwrap();
        let wr = conv.weight().dims().to_vec();
        assert_eq!(wr, vec![3, 1, 2], "depth-wise Candle layout");
        assert_eq!(conv.config().groups, 3);
        assert_eq!(conv.config().padding, 0);
    }

    #[test]
    fn conv1d_from_mlx_weight_rejects_wrong_shape() {
        let device = Device::Cpu;
        // Missing trailing "1" axis.
        let w = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), &device).unwrap();
        assert!(conv1d_from_mlx_weight(w, 2).is_err());
    }

    // ─── A.8-C.5 native dispatcher parity ──────────────────────────────────

    /// Process-wide guard: the parity tests below mutate the
    /// `LUMEN_LINEAR_ATTN_NATIVE` env var to flip the dispatcher branch,
    /// and `cargo test` runs tests in parallel by default. Holding this
    /// mutex for the duration of each test serializes the env-var window
    /// without affecting the rest of the suite.
    #[cfg(feature = "turboquant-gpu")]
    static NATIVE_FLAG_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Two `GatedDeltaNet` instances built from the same RNG seed produce the
    /// same Candle tensors (in_proj, conv1d, a_log, dt_bias, norm_w, out_proj).
    /// Running one through the Candle path and the other through the
    /// `LUMEN_LINEAR_ATTN_NATIVE=1` dispatcher must yield matching outputs
    /// (within bf16/f32 numerical floor).
    #[cfg(feature = "turboquant-gpu")]
    #[test]
    fn native_dispatcher_matches_candle_prefill() {
        // Native dispatcher requires a Metal device; CPU tests fall through to
        // Candle. Skip if Metal isn't available in the test sandbox.
        let _guard = NATIVE_FLAG_GUARD.lock().unwrap_or_else(|p| p.into_inner());
        let device = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut net_candle = build_tiny(0xA8C5, &device);
        let mut net_native = build_tiny(0xA8C5, &device);
        let mut r = StdRng::seed_from_u64(0x515E);
        let x = rnd(&[1, 6, 8], &mut r, &device);

        let y_candle = net_candle.forward(&x, None).unwrap();

        // Force the dispatcher branch on for the second instance.
        unsafe {
            std::env::set_var("LUMEN_LINEAR_ATTN_NATIVE", "1");
        }
        let y_native = net_native.forward(&x, None).unwrap();
        unsafe {
            std::env::remove_var("LUMEN_LINEAR_ATTN_NATIVE");
        }

        assert_eq!(y_candle.dims(), y_native.dims());
        let yc = y_candle.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let yn = y_native.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let dot: f64 = yc
            .iter()
            .zip(yn.iter())
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        let na: f64 = yc.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = yn.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (na * nb);
        let max_abs = yc
            .iter()
            .zip(yn.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let max_mag = yc
            .iter()
            .chain(yn.iter())
            .map(|v| v.abs())
            .fold(0.0_f32, f32::max);
        let rel = max_abs / max_mag.max(1e-6);
        assert!(
            cos > 0.999 && rel < 5e-3,
            "native vs Candle: cos={cos} rel={rel} abs_max={max_abs}"
        );
        // The native state must have been allocated and marked populated.
        assert!(net_native.native_ssm_state.is_some());
        assert!(net_native.native_ssm_state.as_ref().unwrap().is_populated());
    }

    /// Decode-style: prefill a chunk, then run a single-token decode step.
    /// Both Candle and native paths persist their respective SSM state across
    /// calls, so the second call's output must match.
    #[cfg(feature = "turboquant-gpu")]
    #[test]
    fn native_dispatcher_matches_candle_prefill_then_decode() {
        let _guard = NATIVE_FLAG_GUARD.lock().unwrap_or_else(|p| p.into_inner());
        let device = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut net_candle = build_tiny(0xD3CD, &device);
        let mut net_native = build_tiny(0xD3CD, &device);
        let mut r = StdRng::seed_from_u64(0x77E1);

        let x_prefill = rnd(&[1, 4, 8], &mut r, &device);
        let x_decode = rnd(&[1, 1, 8], &mut r, &device);

        let _ = net_candle.forward(&x_prefill, None).unwrap();
        let yc_decode = net_candle.forward(&x_decode, None).unwrap();

        unsafe {
            std::env::set_var("LUMEN_LINEAR_ATTN_NATIVE", "1");
        }
        let _ = net_native.forward(&x_prefill, None).unwrap();
        let yn_decode = net_native.forward(&x_decode, None).unwrap();
        unsafe {
            std::env::remove_var("LUMEN_LINEAR_ATTN_NATIVE");
        }

        assert_eq!(yc_decode.dims(), yn_decode.dims());
        let yc = yc_decode.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let yn = yn_decode.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let dot: f64 = yc
            .iter()
            .zip(yn.iter())
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        let na: f64 = yc.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = yn.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (na * nb);
        let max_abs = yc
            .iter()
            .zip(yn.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let max_mag = yc
            .iter()
            .chain(yn.iter())
            .map(|v| v.abs())
            .fold(0.0_f32, f32::max);
        let rel = max_abs / max_mag.max(1e-6);
        assert!(
            cos > 0.999 && rel < 5e-3,
            "decode native vs Candle: cos={cos} rel={rel} abs_max={max_abs}"
        );
    }

    /// snapshot → run K decode steps that mutate state → restore → run a
    /// reference token sequence — output must bit-match the same reference
    /// run made directly after the snapshot point. This locks down the
    /// rollback contract used by speculative decoding.
    #[test]
    fn snapshot_restore_roundtrip_matches_baseline() {
        let device = Device::Cpu;
        let mut r = StdRng::seed_from_u64(0xABCDEF01);
        let x_prefill = rnd(&[1, 4, 8], &mut r, &device);
        let x_a = rnd(&[1, 1, 8], &mut r, &device);
        let x_b = rnd(&[1, 1, 8], &mut r, &device);
        let x_c = rnd(&[1, 1, 8], &mut r, &device);

        // Baseline: prefill, then decode A.
        let mut net_base = build_tiny(0xCAFE, &device);
        let _ = net_base.forward(&x_prefill, None).unwrap();
        let y_baseline = net_base.forward(&x_a, None).unwrap();
        let baseline_vec = y_baseline.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Test path: prefill, snapshot, run B+C (mutate state), restore, decode A.
        let mut net_test = build_tiny(0xCAFE, &device);
        let _ = net_test.forward(&x_prefill, None).unwrap();
        let snap = net_test.snapshot_state().unwrap();
        let _ = net_test.forward(&x_b, None).unwrap();
        let _ = net_test.forward(&x_c, None).unwrap();
        net_test.restore_state(&snap).unwrap();
        let y_test = net_test.forward(&x_a, None).unwrap();
        let test_vec = y_test.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(baseline_vec.len(), test_vec.len());
        let mut max_abs = 0.0f32;
        for (a, b) in baseline_vec.iter().zip(test_vec.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert!(
            max_abs < 1e-5,
            "snapshot/restore did not roundtrip — max |Δ| = {max_abs}"
        );
    }

    /// snapshot before any forward (cold layer) → run + advance → restore →
    /// run again. Restored output must match a fresh layer's forward.
    #[test]
    fn snapshot_restore_from_cold_state() {
        let device = Device::Cpu;
        let mut r = StdRng::seed_from_u64(0xC01D_C01D);
        let x_a = rnd(&[1, 3, 8], &mut r, &device);
        let x_b = rnd(&[1, 2, 8], &mut r, &device);

        let mut net_baseline = build_tiny(0x4242, &device);
        let y_base = net_baseline.forward(&x_a, None).unwrap();
        let base_vec = y_base.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let mut net_test = build_tiny(0x4242, &device);
        let cold_snap = net_test.snapshot_state().unwrap();
        let _ = net_test.forward(&x_b, None).unwrap(); // advance off the cold state
        net_test.restore_state(&cold_snap).unwrap();
        let y_test = net_test.forward(&x_a, None).unwrap();
        let test_vec = y_test.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let mut max_abs = 0.0f32;
        for (a, b) in base_vec.iter().zip(test_vec.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert!(
            max_abs < 1e-5,
            "cold-state restore did not roundtrip — max |Δ| = {max_abs}"
        );
    }

    /// Native (`LUMEN_LINEAR_ATTN_NATIVE=1`) path snapshot/restore must
    /// preserve the device-resident SSM state buffer. Same shape as the
    /// CPU baseline test: prefill → snapshot → mutate → restore → decode A.
    #[cfg(feature = "turboquant-gpu")]
    #[test]
    fn snapshot_restore_roundtrip_matches_baseline_native() {
        let _guard = NATIVE_FLAG_GUARD.lock().unwrap_or_else(|p| p.into_inner());
        let device = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut r = StdRng::seed_from_u64(0x5EC0_DECC);
        let x_prefill = rnd(&[1, 4, 8], &mut r, &device);
        let x_a = rnd(&[1, 1, 8], &mut r, &device);
        let x_b = rnd(&[1, 1, 8], &mut r, &device);
        let x_c = rnd(&[1, 1, 8], &mut r, &device);

        unsafe {
            std::env::set_var("LUMEN_LINEAR_ATTN_NATIVE", "1");
        }
        let baseline_vec = {
            let mut net = build_tiny(0xC0DE, &device);
            let _ = net.forward(&x_prefill, None).unwrap();
            let y = net.forward(&x_a, None).unwrap();
            y.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        };

        let test_vec = {
            let mut net = build_tiny(0xC0DE, &device);
            let _ = net.forward(&x_prefill, None).unwrap();
            let snap = net.snapshot_state().unwrap();
            let _ = net.forward(&x_b, None).unwrap();
            let _ = net.forward(&x_c, None).unwrap();
            net.restore_state(&snap).unwrap();
            let y = net.forward(&x_a, None).unwrap();
            y.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        };
        unsafe {
            std::env::remove_var("LUMEN_LINEAR_ATTN_NATIVE");
        }

        assert_eq!(baseline_vec.len(), test_vec.len());
        let mut max_abs = 0.0f32;
        for (a, b) in baseline_vec.iter().zip(test_vec.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert!(
            max_abs < 5e-4,
            "native snapshot/restore did not roundtrip — max |Δ| = {max_abs}"
        );
    }

    /// `reset_state` must drop both Candle and native SSM state.
    #[cfg(feature = "turboquant-gpu")]
    #[test]
    fn reset_state_clears_native_ssm_state() {
        let _guard = NATIVE_FLAG_GUARD.lock().unwrap_or_else(|p| p.into_inner());
        let device = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut net = build_tiny(0xCAFE, &device);
        let mut r = StdRng::seed_from_u64(0xABCD);
        let x = rnd(&[1, 3, 8], &mut r, &device);
        unsafe {
            std::env::set_var("LUMEN_LINEAR_ATTN_NATIVE", "1");
        }
        let _ = net.forward(&x, None).unwrap();
        assert!(net.native_ssm_state.is_some());
        assert!(net.native_ssm_state.as_ref().unwrap().is_populated());

        net.reset_state();
        // Buffer is kept but must be marked unpopulated post-reset.
        assert!(net.native_ssm_state.is_some());
        assert!(!net.native_ssm_state.as_ref().unwrap().is_populated());

        unsafe {
            std::env::remove_var("LUMEN_LINEAR_ATTN_NATIVE");
        }
    }

    /// per-seq SSM state isolation.
    ///
    /// Two sequences (A=1, B=2) interleaved through a single `GatedDeltaNet`
    /// via `set_current_seq_id`. Each sequence's output after interleaving
    /// must be bit-identical to the output it would produce if it were the
    /// only sequence ever run through the net.
    #[test]
    fn per_seq_ssm_state_is_isolated() {
        let device = Device::Cpu;
        let mut r = StdRng::seed_from_u64(0xF00_D_BAD);

        // Shared random inputs — different lengths so the test is asymmetric.
        let x_a_pre = rnd(&[1, 4, 8], &mut r, &device); // A prefill
        let x_a_dec1 = rnd(&[1, 1, 8], &mut r, &device); // A decode step 1
        let x_a_dec2 = rnd(&[1, 1, 8], &mut r, &device); // A decode step 2
        let x_b_pre = rnd(&[1, 3, 8], &mut r, &device); // B prefill (different length)
        let x_b_dec1 = rnd(&[1, 1, 8], &mut r, &device); // B decode step 1
        let x_b_dec2 = rnd(&[1, 1, 8], &mut r, &device); // B decode step 2

        // ── Solo A: seq runs alone, no other sequence touches the net ──────────
        let (solo_a_d1, solo_a_d2) = {
            let mut net = build_tiny(0x5010, &device);
            net.set_current_seq_id(1);
            let _ = net.forward(&x_a_pre, None).unwrap();
            let d1 = net
                .forward(&x_a_dec1, None)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let d2 = net
                .forward(&x_a_dec2, None)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            (d1, d2)
        };

        // ── Solo B: same ───────────────────────────────────────────────────────
        let (solo_b_d1, solo_b_d2) = {
            let mut net = build_tiny(0x5010, &device);
            net.set_current_seq_id(2);
            let _ = net.forward(&x_b_pre, None).unwrap();
            let d1 = net
                .forward(&x_b_dec1, None)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let d2 = net
                .forward(&x_b_dec2, None)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            (d1, d2)
        };

        // ── Interleaved: A and B alternate through the same net instance ───────
        let (inter_a_d1, inter_a_d2, inter_b_d1, inter_b_d2) = {
            let mut net = build_tiny(0x5010, &device);

            // Prefill A then B.
            net.set_current_seq_id(1);
            let _ = net.forward(&x_a_pre, None).unwrap();

            net.set_current_seq_id(2);
            let _ = net.forward(&x_b_pre, None).unwrap();

            // Interleaved decode: A → B → A → B.
            net.set_current_seq_id(1);
            let a_d1 = net
                .forward(&x_a_dec1, None)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            net.set_current_seq_id(2);
            let b_d1 = net
                .forward(&x_b_dec1, None)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            net.set_current_seq_id(1);
            let a_d2 = net
                .forward(&x_a_dec2, None)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            net.set_current_seq_id(2);
            let b_d2 = net
                .forward(&x_b_dec2, None)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            (a_d1, a_d2, b_d1, b_d2)
        };

        let max_diff = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };

        let diff_a1 = max_diff(&solo_a_d1, &inter_a_d1);
        let diff_a2 = max_diff(&solo_a_d2, &inter_a_d2);
        let diff_b1 = max_diff(&solo_b_d1, &inter_b_d1);
        let diff_b2 = max_diff(&solo_b_d2, &inter_b_d2);

        assert!(diff_a1 < 1e-5, "A decode-1 contaminated: max|Δ|={diff_a1}");
        assert!(diff_a2 < 1e-5, "A decode-2 contaminated: max|Δ|={diff_a2}");
        assert!(diff_b1 < 1e-5, "B decode-1 contaminated: max|Δ|={diff_b1}");
        assert!(diff_b2 < 1e-5, "B decode-2 contaminated: max|Δ|={diff_b2}");
    }

    /// Prefill-failure cleanup path (mirrors the engine.rs fix).
    ///
    /// Seq A runs several steps, then seq B is admitted (init_seq + set_current_seq_id)
    /// but "fails" before any forward — simulated by immediately calling remove_seq.
    /// After the cleanup, seq A must resume from exactly where it left off
    /// (state unchanged, output bit-identical to an uninterrupted reference).
    #[test]
    fn remove_seq_after_failed_admit_leaves_active_seq_intact() {
        let device = Device::Cpu;
        let mut r = StdRng::seed_from_u64(0xDEAD_BEEF);

        let x_pre = rnd(&[1, 4, 8], &mut r, &device); // A prefill
        let x_dec1 = rnd(&[1, 1, 8], &mut r, &device); // A decode step 1
        let x_dec2 = rnd(&[1, 1, 8], &mut r, &device); // A decode step 2

        // Reference: A runs completely alone.
        let (ref_d1, ref_d2) = {
            let mut net = build_tiny(0xABBA, &device);
            net.set_current_seq_id(1);
            let _ = net.forward(&x_pre, None).unwrap();
            let d1 = net
                .forward(&x_dec1, None)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let d2 = net
                .forward(&x_dec2, None)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            (d1, d2)
        };

        // Test: A prefills, B is admitted but immediately removed (prefill failure),
        // then A continues.
        let (test_d1, test_d2) = {
            let mut net = build_tiny(0xABBA, &device);
            net.set_current_seq_id(1);
            let _ = net.forward(&x_pre, None).unwrap();

            // Seq B admitted: init + switch active — then "failure" → remove.
            net.init_seq(2);
            net.set_current_seq_id(2);
            // (no forward — simulate prefill crash)
            net.remove_seq(2);

            // Resume seq A.
            net.set_current_seq_id(1);
            let d1 = net
                .forward(&x_dec1, None)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let d2 = net
                .forward(&x_dec2, None)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            (d1, d2)
        };

        let max_diff = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        let d1 = max_diff(&ref_d1, &test_d1);
        let d2 = max_diff(&ref_d2, &test_d2);
        assert!(
            d1 < 1e-5,
            "A decode-1 corrupted after failed B admit: max|Δ|={d1}"
        );
        assert!(
            d2 < 1e-5,
            "A decode-2 corrupted after failed B admit: max|Δ|={d2}"
        );
    }
}
