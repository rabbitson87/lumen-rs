//! Composition of a single Qwen3.5-VL-MoE decoder layer.
//!
//! Each of the 40 layers has one of two attention variants — [`GatedDeltaNet`] (linear) or
//! [`SelfAttention`] (full GQA) — wrapped in the standard pre-norm + residual + post-norm +
//! MLP-residual scaffold:
//!
//! ```text
//!   r = attention(input_layernorm(x), masks)
//!   h = x + r
//!   out = h + mlp(post_attention_layernorm(h))
//! ```
//!
//! This module is **plumbing only**: the expensive numerics live in the block modules
//! ([`self_attn`], [`linear_attn`], [`moe`]) and their fixture tests. Here we lock down the
//! dispatch pattern and the residual wiring so the 40-layer loop in [`super::model`] has
//! nothing left to decide.
//!
//! No shard I/O yet — Stage 2-f-b owns that. Tests exercise a synthetic tiny layer to catch
//! regressions in the composition (residual add order, norm placement, mask routing).
//!
//! Reference: `mlx_lm.models.qwen3_5.DecoderLayer` (mlx-lm 0.31.3).

use candle_core::{Result as CandleResult, Tensor};
use candle_nn::{Module, RmsNorm};
use candle_transformers::models::quantized_gemma4::CompressedKVBackend;

/// Alias for the `Option<Box<dyn ...>>` threaded through forward calls.
pub type CompressedKvHandle = Option<Box<dyn CompressedKVBackend + Send>>;

use super::config::LayerType;
use super::linear_attn::GatedDeltaNet;
use super::moe::{MlpBlock, SparseMoeBlock};
use super::self_attn::SelfAttention;

/// Lever B L.3 (2026-04-28): `LUMEN_BF16_RMSNORM=1` enables a bf16 chain
/// across `input_layernorm` → qkv / in_proj_combined. Default OFF — the f32
/// chain is the production path until 35B A/B clears σ ≥ +5.
#[inline]
fn bf16_rmsnorm_enabled() -> bool {
    std::env::var("LUMEN_BF16_RMSNORM")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Generic helper: run a Candle `RmsNorm` site through the native Metal
/// `RmsNormBf16Out` shader (Workstream B Phase 2 replacement for the
/// MPSGraph version). Output dtype is `BF16` so a downstream `forward_bf16_in`
/// matmul kernel (Workstream A's qmv_fast bf16) can consume it natively
/// without the cast-back trap.
///
/// **Why native over MPSGraph**: the prior `MpsRmsNormBf16Out` produced
/// non-bit-deterministic output across calls (5119/5120 bits flipped per
/// invocation — Apple-internal reduction-order optimization). That
/// regressed the bf16 chain's `R1↔R2` token determinism (12/12 → 0/12).
/// The native Metal shader at `crates/lumen-metal/src/shaders/
/// rms_norm.metal` pins reduction order via `simd_sum` + threadgroup
/// memory and is bit-stable across invocations (verified by
/// `tests/native_rms_norm_bf16_out.rs::native_rms_norm_determinism_repeat_call_bit_identical`).
///
/// Cached by `eps` bits — single config value across all RmsNorm call
/// sites in Qwen3.5-MoE / Qwen3.6 Dense (input_layernorm + post_attn_layernorm
/// share the same `rms_norm_eps` in the production checkpoints), so one
/// cache slot is enough.
///
/// Used by:
///   - `input_layernorm` site (Lever B L.3 / Workstream B Phase 2)
///   - `post_attention_layernorm` site (Workstream B Phase 5 — paired with
///     `MlpBlock::forward_bf16_in` to keep the post-attn bf16 chain alive
///     through the MLP gate_up matmul on the Affine4 fast path).
#[cfg(feature = "turboquant-gpu")]
fn apply_rms_norm_bf16_out(
    x: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> CandleResult<Tensor> {
    use std::sync::Mutex;
    use std::sync::OnceLock;
    use lumen_metal::rms_norm::RmsNormBf16Out;

    static CACHE: OnceLock<Mutex<Option<(u32, RmsNormBf16Out)>>> = OnceLock::new();
    let lock = CACHE.get_or_init(|| Mutex::new(None));
    let mut guard = lock.lock().expect("RmsNormBf16Out cache poisoned");
    let eps_bits = eps.to_bits();
    if guard.as_ref().map(|(b, _)| *b) != Some(eps_bits) {
        let runtime = RmsNormBf16Out::new(eps).map_err(|e| {
            candle_core::Error::Msg(format!("RmsNormBf16Out init: {e}"))
        })?;
        *guard = Some((eps_bits, runtime));
    }
    let runtime = &guard.as_ref().expect("just set above").1;
    runtime.forward(x, weight)
}

/// Workstream B Phase 10 — bf16-in / bf16-out RmsNorm. Used by the
/// model-wide bf16 carrier stream (`LUMEN_BF16_RESIDUAL=1`) so the
/// layer-level `h` carrier never has to be cast back to f32 at the
/// layernorm boundary. Same eps-keyed cache pattern as `apply_rms_norm_bf16_out`.
#[cfg(feature = "turboquant-gpu")]
fn apply_rms_norm_bf16_in_bf16_out(
    x: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> CandleResult<Tensor> {
    use std::sync::Mutex;
    use std::sync::OnceLock;
    use lumen_metal::rms_norm::RmsNormBf16InBf16Out;

    static CACHE: OnceLock<Mutex<Option<(u32, RmsNormBf16InBf16Out)>>> = OnceLock::new();
    let lock = CACHE.get_or_init(|| Mutex::new(None));
    let mut guard = lock.lock().expect("RmsNormBf16InBf16Out cache poisoned");
    let eps_bits = eps.to_bits();
    if guard.as_ref().map(|(b, _)| *b) != Some(eps_bits) {
        let runtime = RmsNormBf16InBf16Out::new(eps).map_err(|e| {
            candle_core::Error::Msg(format!("RmsNormBf16InBf16Out init: {e}"))
        })?;
        *guard = Some((eps_bits, runtime));
    }
    let runtime = &guard.as_ref().expect("just set above").1;
    runtime.forward(x, weight)
}

/// The two attention variants that can populate a decoder layer's `attention` slot.
///
/// Qwen3.5-VL-MoE uses the Qwen3-Next 3:1 pattern: layers `0..4k+3` get [`Linear`] (Gated
/// Delta Net SSM) and every fourth layer (`4k+3`) gets [`Full`] (gated GQA). The mapping is
/// driven by `TextConfig::layer_types`, already validated by [`super::config`].
///
/// [`Linear`]: AttentionBlock::Linear
/// [`Full`]: AttentionBlock::Full
pub enum AttentionBlock {
    /// Gated Delta Net (SSM). Consumes the `ssm_mask` branch, ignores `pos_offset`.
    Linear(GatedDeltaNet),
    /// Full gated GQA. Consumes the `fa_mask` + `pos_offset` branch.
    Full(SelfAttention),
}

impl AttentionBlock {
    /// Forward one attention block. `fa_mask` and `ssm_mask` are routed by variant so callers
    /// don't need to match before each call — they only need to know that self-attn wants a
    /// causal-shaped additive mask and gated-delta wants a per-token bool mask (both nullable
    /// for untouched-default behavior).
    pub fn forward(
        &mut self,
        x: &Tensor,
        pos_offset: usize,
        fa_mask: Option<&Tensor>,
        ssm_mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        self.forward_with_tq(x, pos_offset, fa_mask, ssm_mask, &mut None)
    }

    pub fn forward_with_tq(
        &mut self,
        x: &Tensor,
        pos_offset: usize,
        fa_mask: Option<&Tensor>,
        ssm_mask: Option<&Tensor>,
        compressed_kv: &mut CompressedKvHandle,
    ) -> CandleResult<Tensor> {
        match self {
            Self::Linear(la) => la.forward(x, ssm_mask),
            Self::Full(sa) => sa.forward_with_tq(x, pos_offset, fa_mask, compressed_kv),
        }
    }

    /// Lever L1: variant-routed forward that folds the post-attention residual
    /// add `(x + r)?` into o_proj's MXFP4 matmul kernel for the `Full` variant
    /// (saves one element-wise add dispatch / decode step). The `Linear`
    /// variant doesn't have a fusion path yet, so it falls back to the
    /// legacy chain + post-hoc `broadcast_add` (correct, but no perf gain).
    /// Both paths return a tensor that already includes the residual — the
    /// caller MUST skip its own `(x + r)?`.
    pub fn forward_with_residual_fused(
        &mut self,
        x: &Tensor,
        residual: &Tensor,
        pos_offset: usize,
        fa_mask: Option<&Tensor>,
        ssm_mask: Option<&Tensor>,
        compressed_kv: &mut CompressedKvHandle,
    ) -> CandleResult<Tensor> {
        match self {
            Self::Linear(la) => {
                let r = la.forward(x, ssm_mask)?;
                r.broadcast_add(residual)
            }
            Self::Full(sa) => {
                sa.forward_with_residual_fused(x, residual, pos_offset, fa_mask, compressed_kv)
            }
        }
    }

    /// Lever H Step 3: variant-routed forward that fuses pre-attention RmsNorm
    /// into the first projection. Reads RAW `x_raw`; the underlying kernel
    /// applies RmsNorm internally. Caller must verify
    /// `has_mxfp4_input_proj() == true`.
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_with_input_rmsnorm_with_tq(
        &mut self,
        x_raw: &Tensor,
        rms_weight: &Tensor,
        rms_eps: f32,
        pos_offset: usize,
        fa_mask: Option<&Tensor>,
        ssm_mask: Option<&Tensor>,
        compressed_kv: &mut CompressedKvHandle,
    ) -> CandleResult<Tensor> {
        match self {
            Self::Linear(la) => {
                la.forward_with_input_rmsnorm(x_raw, rms_weight, rms_eps, ssm_mask)
            }
            Self::Full(sa) => sa.forward_with_input_rmsnorm(
                x_raw,
                rms_weight,
                rms_eps,
                pos_offset,
                fa_mask,
                compressed_kv,
            ),
        }
    }

    /// True iff the input projection of this attention variant holds MXFP4
    /// packed weights — required for the Lever H Step 3 input-rmsnorm fusion
    /// path. Dense fixtures (CPU tests) return `false`, falling back to the
    /// unfused external RmsNorm dispatch.
    pub fn has_mxfp4_input_proj(&self) -> bool {
        match self {
            Self::Linear(la) => la.has_mxfp4_in_proj(),
            Self::Full(sa) => sa.has_mxfp4_qkv(),
        }
    }

    /// True iff the input projection holds quantized weights (MXFP4 OR
    /// Affine4). Used by Workstream B bf16-rmsnorm gating: bf16 activation
    /// from `MpsRmsNormBf16Out` is consumable by either quant variant's
    /// `forward_bf16_in` fast-path. Broader than `has_mxfp4_input_proj`,
    /// which limits to the MXFP4-only Lever B L.3 path.
    pub fn has_quant_input_proj(&self) -> bool {
        match self {
            Self::Linear(la) => la.has_mxfp4_in_proj() || la.has_affine4_in_proj(),
            Self::Full(sa) => sa.has_mxfp4_qkv() || sa.has_affine4_qkv(),
        }
    }

    /// True iff the input projection supports input-RmsNorm fusion (MXFP4 Lever
    /// H Step 3 or Affine4 Lever R1). Use this rather than `has_mxfp4_input_proj`
    /// when the caller wants the fusion to fire on either quant variant.
    pub fn supports_fused_input_rmsnorm(&self) -> bool {
        match self {
            Self::Linear(la) => la.has_mxfp4_in_proj(),
            Self::Full(sa) => sa.supports_fused_input_rmsnorm(),
        }
    }

    /// Reset any stateful caches (KV append cache for full-attn, SSM + conv state for
    /// gated-delta linear-attn). Safe to call between generation requests.
    pub fn reset_cache(&mut self) {
        match self {
            Self::Full(sa) => sa.reset_kv_cache(),
            Self::Linear(la) => la.reset_state(),
        }
    }

    pub fn enable_kv_cache(&mut self, max_seq_len: usize) {
        if let Self::Full(sa) = self {
            sa.enable_kv_cache(max_seq_len);
        }
    }

    pub fn set_current_seq_id(&mut self, seq_id: u64) {
        match self {
            Self::Full(sa) => sa.set_current_seq_id(seq_id),
            Self::Linear(la) => la.set_current_seq_id(seq_id),
        }
    }

    pub fn init_seq_kv_cache(&mut self, seq_id: u64) {
        match self {
            Self::Full(sa) => sa.init_seq_kv_cache(seq_id),
            Self::Linear(la) => la.init_seq(seq_id),
        }
    }

    pub fn remove_seq_kv_cache(&mut self, seq_id: u64) {
        match self {
            Self::Full(sa) => sa.remove_seq_kv_cache(seq_id),
            Self::Linear(la) => la.remove_seq(seq_id),
        }
    }

    /// Observed kind — useful when the loader needs to know which sub-module to fill without
    /// re-reading the config. Matches [`LayerType`] one-to-one.
    pub fn kind(&self) -> LayerType {
        match self {
            Self::Linear(_) => LayerType::LinearAttention,
            Self::Full(_) => LayerType::FullAttention,
        }
    }

    /// True iff this layer uses the SSM variant. Exposed so tests and the final 40-layer loop
    /// can verify the 3:1 dispatch pattern without unpacking the enum.
    pub fn is_linear(&self) -> bool {
        matches!(self, Self::Linear(_))
    }

    /// Capture all recurrent / append-cache state for spec-decoding rollback.
    pub fn snapshot_state(&self) -> CandleResult<AttentionBlockSnapshot> {
        match self {
            Self::Linear(la) => Ok(AttentionBlockSnapshot::Linear(la.snapshot_state()?)),
            Self::Full(sa) => {
                let kv_len = sa.kv_len();
                Ok(AttentionBlockSnapshot::Full { kv_len })
            }
        }
    }

    /// Restore state captured by [`snapshot_state`]. The `tq_layer` argument
    /// is the TurboQuant compressor slot index for full-attn layers (so a
    /// shared compressed-KV backend can be truncated alongside this layer's
    /// append cache); pass `None` for linear-attn layers.
    pub fn restore_state(
        &mut self,
        snap: &AttentionBlockSnapshot,
        compressed_kv: &mut CompressedKvHandle,
    ) -> CandleResult<()> {
        match (self, snap) {
            (Self::Linear(la), AttentionBlockSnapshot::Linear(s)) => la.restore_state(s),
            (Self::Full(sa), AttentionBlockSnapshot::Full { kv_len }) => {
                sa.truncate_kv_cache(*kv_len);
                if let (Some(slot), Some(ckv)) = (sa.tq_slot(), compressed_kv.as_mut()) {
                    ckv.truncate(slot, *kv_len);
                }
                Ok(())
            }
            (Self::Linear(_), AttentionBlockSnapshot::Full { .. })
            | (Self::Full(_), AttentionBlockSnapshot::Linear(_)) => Err(candle_core::Error::Msg(
                "AttentionBlock::restore_state: snapshot variant mismatch".into(),
            )),
        }
    }
}

/// Captured per-layer attention state. `Linear` carries the full
/// [`GatedDeltaNetSnapshot`]; `Full` only stores the append-cache length so
/// the rollback can be a pure index reset (the underlying buffer is left
/// in place — subsequent appends overwrite the dropped tail).
#[derive(Clone)]
pub enum AttentionBlockSnapshot {
    Linear(super::linear_attn::GatedDeltaNetSnapshot),
    Full { kv_len: usize },
}

/// A fully composed Qwen3.5-family decoder layer. Construct via [`DecoderLayer::new`]; the
/// loader or a test harness is responsible for assembling the [`RmsNorm`], [`AttentionBlock`],
/// and [`MlpBlock`] parts. The `mlp` field is the Moe-or-Dense enum so the same layer
/// shape covers Qwen3.6-35B-A3B-mxfp4 (MoE) and Qwen3.6-27B (Dense SwiGLU).
pub struct DecoderLayer {
    pub(super) input_layernorm: RmsNorm,
    attention: AttentionBlock,
    post_attention_layernorm: RmsNorm,
    mlp: MlpBlock,
    layer_idx: usize,
}

impl DecoderLayer {
    /// Construct a layer. `mlp` accepts either [`SparseMoeBlock`] or [`super::moe::DenseMlp`]
    /// directly via the `From` impls on [`MlpBlock`], so existing MoE call sites and tests
    /// (passing `SparseMoeBlock`) compile unchanged.
    pub fn new(
        input_layernorm: RmsNorm,
        attention: AttentionBlock,
        post_attention_layernorm: RmsNorm,
        mlp: impl Into<MlpBlock>,
    ) -> Self {
        Self {
            input_layernorm,
            attention,
            post_attention_layernorm,
            mlp: mlp.into(),
            layer_idx: 0,
        }
    }

    pub fn set_layer_idx(&mut self, idx: usize) {
        self.layer_idx = idx;
    }

    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    pub fn is_linear(&self) -> bool {
        self.attention.is_linear()
    }

    pub fn attention(&self) -> &AttentionBlock {
        &self.attention
    }

    /// Prefill/decode forward. The residual + norm pattern mirrors MLX exactly — any drift
    /// here (e.g. normalizing after the residual add) silently invalidates multi-layer parity.
    ///
    /// `pos_offset` is forwarded to self-attn for RoPE; the gated-delta branch ignores it.
    pub fn forward(
        &mut self,
        x: &Tensor,
        pos_offset: usize,
        fa_mask: Option<&Tensor>,
        ssm_mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let (out, _) = self.forward_with_tq(
            x, pos_offset, fa_mask, ssm_mask, &mut None, None, None,
        )?;
        Ok(out)
    }

    /// Lever L4: forward variant that participates in cross-layer norm fusion.
    /// `prev_attn_in`: when `Some`, the input_layernorm dispatch is skipped
    /// and the value is used as the attention input (pre-normalized from
    /// the previous layer's fused mlp_final + next_input_rmsnorm dispatch).
    /// `next_input_rmsnorm`: when `Some(weight, eps)`, the MoE final-combine
    /// kernel ALSO produces `attn_in` for the NEXT layer using these
    /// rms_weight/eps, saving the next layer's input_layernorm dispatch.
    /// Returns `(out, attn_in_for_next_layer_opt)`.
    pub fn forward_with_tq(
        &mut self,
        x: &Tensor,
        pos_offset: usize,
        fa_mask: Option<&Tensor>,
        ssm_mask: Option<&Tensor>,
        compressed_kv: &mut CompressedKvHandle,
        prev_attn_in: Option<&Tensor>,
        next_input_rmsnorm: Option<(&Tensor, f32)>,
    ) -> CandleResult<(Tensor, Option<Tensor>)> {
        // Optional per-layer intermediate dump: set LUMEN_DUMP_LAYER_INTERNALS to a
        // directory AND LUMEN_DUMP_LAYER_IDX to the integer layer index to trace.
        let dump_dir = std::env::var("LUMEN_DUMP_LAYER_INTERNALS").ok();
        let target_idx: Option<usize> = std::env::var("LUMEN_DUMP_LAYER_IDX")
            .ok()
            .and_then(|s| s.parse().ok());
        let do_dump = matches!((dump_dir.as_ref(), target_idx),
            (Some(_), Some(t)) if t == self.layer_idx);

        // Optional per-layer timing: `LUMEN_LAYER_TIMING=1` prints sub-block ms per
        // forward call and aggregates the per-forward totals so callers can assess where
        // decode-step wall-time actually goes (self_attn vs linear_attn vs MoE).
        let timing = std::env::var("LUMEN_LAYER_TIMING")
            .map(|v| v == "1")
            .unwrap_or(false);

        // Each marker syncs the device first so reported ms reflect GPU completion of the
        // immediately preceding stage (not queue submission). Without sync, attn-stage ms
        // absorbs the prior ln's GPU work and ln_ms collapses to ~0.
        let device = x.device().clone();
        let mark = |timing: bool| -> Option<std::time::Instant> {
            if timing {
                let _ = device.synchronize();
                Some(std::time::Instant::now())
            } else {
                None
            }
        };
        let t0 = mark(timing);
        // wire-in NEGATIVE (twice).
        //
        // 1. First attempt (no compile descriptor): per-layer MPSGraph
        //    RmsNorm caused cumulative numerical drift across 80 callsites;
        //    token sequence diverged from baseline starting on the warmup
        //    prompt. See `mpsgraph_phase3_per_layer_negative.md`.
        // 2. Option-1 retry under strict-math compile descriptor
        //    (`optimizationLevel=Level0` + `reducedPrecisionFastMath=None`,
        //    set in `mpsgraph::executable::compile`): drift unchanged
        //    (warmup tok=35429 vs baseline 9419), perf identical
        //    (-0.55%, σ=-1.59). Apple's public knobs can't reach the
        //    actual drift cause (parallel reduction order +
        //    `reciprocalSquareRoot` polynomial). See
        //    `mpsgraph_phase3_strict_math_negative.md`.
        //
        // Singleton stays in `mpsgraph_norm` for `final_norm` only (Phase
        // 3.1 LANDED, bit-identical there).
        // Lever H Step 3 (2026-04-28) NEGATIVE — pre-attention RmsNorm fusion
        // into qkv_proj (full attention) / in_proj_combined (linear attention).
        // Wired but **default OFF** because A/B measurement (35B, n=720
        // pooled per variant) showed σ=-12.32 STRONG regression: +0.758 ms
        // per decode step (+1.84%, 24.28 → 23.84 tok/s).
        //
        // Hypothesis: the v3 RmsNorm kernel topology is tuned for moe gate_up
        // dimensions (out=4096). For the larger qkv (out=9216) and in_proj
        // (out=12352) outputs, the cooperative RmsNorm reduction overhead
        // exceeds the dispatch savings. The unfused path's stand-alone matmul
        // kernel keeps a more efficient topology for these sizes.
        //
        // Bit-identical correctness confirmed: 20/20 token matches across all
        // cross-comparisons — only perf regresses.
        //
        // Opt-in via `LUMEN_ENABLE_INPUT_RMSNORM_FUSION=1` for future
        // kernel-tuning experiments (test the impact of a per-out-size kernel
        // variant). Auto-falls-back when the attention variant's input
        // projection is Dense (CPU/fixture tests) or the GPU feature is off.
        let input_rmsnorm_enabled = std::env::var("LUMEN_ENABLE_INPUT_RMSNORM_FUSION")
            .map(|v| v == "1")
            .unwrap_or(false);
        #[cfg(feature = "turboquant-gpu")]
        let input_fusion_active =
            input_rmsnorm_enabled && self.attention.supports_fused_input_rmsnorm();
        #[cfg(not(feature = "turboquant-gpu"))]
        let input_fusion_active = {
            let _ = input_rmsnorm_enabled;
            false
        };

        // Lever B L.3 (2026-04-28): when LUMEN_BF16_RMSNORM=1 and the
        // attention variant uses an MXFP4 input projection, route
        // input_layernorm through `MpsRmsNormBf16Out` (L.1) so the activation
        // is emitted in bf16, then consumed by `forward_bf16_in` on the qkv /
        // in_proj_combined kernel (L.2). Mutually exclusive with
        // input-rmsnorm fusion (which bypasses input_layernorm entirely).
        // Default OFF — the f32 chain is the production path until 35B A/B
        // proves σ ≥ +5.
        // Workstream B Phase 2: native `RmsNormBf16Out` replaced the
        // MPSGraph version, so the gate no longer needs the `mpsgraph`
        // feature — only `turboquant-gpu`.
        #[cfg(feature = "turboquant-gpu")]
        let bf16_rmsnorm_active = bf16_rmsnorm_enabled()
            && !input_fusion_active
            && self.attention.has_quant_input_proj()
            && x.device().is_metal();
        #[cfg(not(feature = "turboquant-gpu"))]
        let bf16_rmsnorm_active = {
            // Touch the helper's env so unset-builds don't warn-as-dead-code.
            let _ = bf16_rmsnorm_enabled();
            false
        };

        // Lever L4: when prev_attn_in is Some, the previous layer's fused
        // mlp_final + next_input_rmsnorm dispatch already produced the
        // pre-normalized attn_in for THIS layer — skip input_layernorm.
        // Mutually exclusive with input_fusion_active and bf16_rmsnorm_active
        // (caller / model.rs guarantees this).
        let attn_in_opt = if let Some(prev) = prev_attn_in {
            Some(prev.clone())
        } else if input_fusion_active {
            None
        } else if bf16_rmsnorm_active {
            // Workstream B Phase 5 (2026-05-08): drop the leftover
            // `feature = "mpsgraph"` requirement here — Phase 2 replaced
            // MpsRmsNormBf16Out with the native shader.
            //
            // when the model-wide bf16 carrier is
            // active, `x` arrives as bf16 → use the bf16-in/bf16-out shader.
            // Otherwise (legacy f32 entry path) the f32-in shader fires.
            #[cfg(feature = "turboquant-gpu")]
            {
                if x.dtype() == candle_core::DType::BF16 {
                    Some(apply_rms_norm_bf16_in_bf16_out(
                        x,
                        self.input_layernorm.weight(),
                        self.input_layernorm.eps() as f32,
                    )?)
                } else {
                    Some(apply_rms_norm_bf16_out(
                        x,
                        self.input_layernorm.weight(),
                        self.input_layernorm.eps() as f32,
                    )?)
                }
            }
            #[cfg(not(feature = "turboquant-gpu"))]
            {
                Some(self.input_layernorm.forward(x)?)
            }
        } else {
            Some(self.input_layernorm.forward(x)?)
        };
        if do_dump {
            if let Some(attn_in) = attn_in_opt.as_ref() {
                if let Some(d) = &dump_dir {
                    let _ = super::model::dump_tensor_f32_public(
                        attn_in,
                        &format!("{d}/attn_in.bin"),
                    );
                }
            }
        }
        let t1 = mark(timing);
        // Lever L1 (residual fusion): when LUMEN_RESIDUAL_FUSION=1 and the
        // input-fusion / bf16-rmsnorm paths are inactive, fold the post-attn
        // `(x + r)?` add into o_proj's MXFP4 matmul kernel via
        // `forward_with_residual_fused`. The returned `r` already includes
        // the residual; we skip the layer-level add. Default OFF until A/B
        // confirms σ ≥ +2.
        #[cfg(feature = "turboquant-gpu")]
        let residual_fusion_active = !input_fusion_active
            && !bf16_rmsnorm_active
            && super::self_attn::residual_fusion_enabled();
        #[cfg(not(feature = "turboquant-gpu"))]
        let residual_fusion_active = false;

        // Workstream B Phase 9 (2026-05-09): opt-in bf16 residual stream.
        // When active, `h` (the layer-level residual carrier) stays in bf16
        // throughout this layer; the o_proj / out_proj boundary casts that
        // previously demoted to f32 are lifted (see self_attn.rs +
        // qwen3_5_moe::linear_attn + qwen3_5_moe_native::linear_attn). Cast
        // back to f32 happens once at layer exit.
        //
        // Prerequisites:
        //   - `LUMEN_BF16_RESIDUAL=1` (env opt-in)
        //   - `bf16_rmsnorm_active`            : input_layernorm produces bf16
        //   - `supports_bf16_in_bf16_out_fast_path()` : Dense + Affine4 only
        //   - `prev_attn_in.is_none()`         : L4 cross-layer fused norm
        //                                        threads pre-normed f32 attn_in
        //   - `!residual_fusion_active`        : MXFP4 residual-fusion kernel
        //                                        is f32-only (already gated
        //                                        on `!bf16_rmsnorm_active`,
        //                                        kept explicit here)
        #[cfg(feature = "turboquant-gpu")]
        let bf16_residual_active = super::moe::bf16_residual_enabled()
            && bf16_rmsnorm_active
            && self.mlp.supports_bf16_in_bf16_out_fast_path()
            && prev_attn_in.is_none()
            && !residual_fusion_active
            && !input_fusion_active
            // Mutual exclusion with the Dense fully-fused post-attn kernel
            // (LUMEN_ENABLE_DENSE_POST_ATTN_FUSION=1): that path consumes
            // raw f32 `h` and would clobber the bf16 carrier. Default OFF
            // for that flag so this is rarely a real conflict, but keeps
            // the gate explicit.
            && !std::env::var("LUMEN_ENABLE_DENSE_POST_ATTN_FUSION")
                .map(|v| v == "1")
                .unwrap_or(false);
        #[cfg(not(feature = "turboquant-gpu"))]
        let bf16_residual_active = false;

        // when bf16-residual is active, the model-wide carrier
        // (cast once after embedding in model.rs) already arrives as bf16 —
        // no per-layer cast. `x_carrier` aliases `x`. The legacy off-path
        // also aliases `x` (no behavior change).
        let x_carrier: &Tensor = x;

        #[cfg(feature = "turboquant-gpu")]
        let r = if residual_fusion_active {
            self.attention.forward_with_residual_fused(
                attn_in_opt
                    .as_ref()
                    .expect("attn_in present when residual fusion is active"),
                x,
                pos_offset,
                fa_mask,
                ssm_mask,
                compressed_kv,
            )?
        } else if input_fusion_active {
            self.attention.forward_with_input_rmsnorm_with_tq(
                x,
                self.input_layernorm.weight(),
                self.input_layernorm.eps() as f32,
                pos_offset,
                fa_mask,
                ssm_mask,
                compressed_kv,
            )?
        } else {
            self.attention.forward_with_tq(
                attn_in_opt
                    .as_ref()
                    .expect("attn_in present when input fusion off"),
                pos_offset,
                fa_mask,
                ssm_mask,
                compressed_kv,
            )?
        };
        #[cfg(not(feature = "turboquant-gpu"))]
        let r = self.attention.forward_with_tq(
            attn_in_opt
                .as_ref()
                .expect("attn_in present when input fusion off"),
            pos_offset,
            fa_mask,
            ssm_mask,
            compressed_kv,
        )?;
        if do_dump {
            if let Some(d) = &dump_dir {
                let _ = super::model::dump_tensor_f32_public(&r, &format!("{d}/attn_out.bin"));
            }
        }
        let t2 = mark(timing);
        // When residual fusion is active, `r` already has `x` added inside.
        // when bf16-residual is active, the `r` from attention is
        // bf16 (boundary cast lifted in self_attn / linear_attn) and
        // `x_carrier` is bf16 → the add stays bf16. Otherwise behaviour
        // matches the legacy f32 stream.
        let h = if residual_fusion_active { r } else { (x_carrier + r)? };
        if do_dump {
            if let Some(d) = &dump_dir {
                let _ = super::model::dump_tensor_f32_public(
                    &h,
                    &format!("{d}/post_attn_residual.bin"),
                );
            }
        }
        let t_res1 = mark(timing);
        // Lever H Step 2 (2026-04-28) LANDED, default ON — skip
        // `post_attention_layernorm.forward(&h)` and route raw `h` + the
        // layernorm weight/eps through `mlp.forward_with_rmsnorm`. All 4
        // MoE input consumers (routing gate, routed gate_up, shared expert
        // gate_up, shared_expert_gate) compute RmsNorm internally inside
        // their Metal kernels.
        //
        // 35B A/B (Run 2 + Run 3 confirmation, n=720 pooled per variant):
        //   KH=0 ~56.0 ms vs KH=1 ~48.5 ms = Δ -7.5 ms (-13.3%) σ +44~49
        //   STRONG signal, bit-identical 20/20 across all cross-comparisons.
        // Decode throughput 18.2 → 20.9 tok/s (+15%).
        //
        // Set `LUMEN_DISABLE_RMSNORM_FUSION=1` to opt-out (rollback /
        // debugging / required when any of these conflicting opt-in flags
        // are also set: LUMEN_ENABLE_SMALL_OUT_GATE, LUMEN_BF16_OUT*,
        // LUMEN_ENABLE_MOE_BF16_CHAIN, LUMEN_ENABLE_MOE_MATMUL_WSUM_*,
        // LUMEN_ENABLE_ROUTING_TOPK_FUSION, LUMEN_ENABLE_GATE_UP_SILU_MUL_FUSION).
        //
        // The legacy `LUMEN_ENABLE_RMSNORM_FUSION=1` is still honored
        // (no-op now since fusion is on by default) for back-compat.
        //
        // same drift cause as `input_layernorm` above
        // (twice, including under strict-math compile descriptor).
        let rmsnorm_fusion_disabled = std::env::var("LUMEN_DISABLE_RMSNORM_FUSION")
            .map(|v| v == "1")
            .unwrap_or(false);
        #[cfg(feature = "turboquant-gpu")]
        // Falls back when the MoE backend isn't MXFP4 (Dense / CPU fixture
        // tests don't have the in-kernel rmsnorm path).
        let fusion_active = !rmsnorm_fusion_disabled && self.mlp.has_mxfp4_backend();
        #[cfg(not(feature = "turboquant-gpu"))]
        let fusion_active = {
            let _ = rmsnorm_fusion_disabled;
            false
        };

        // Standard path: external RmsNorm dispatch then MLP. When fusion is
        // active (MoE) or the Dense fully-fused post-attn path qualifies,
        // skip the external dispatch — the MLP block does the RmsNorm inside.
        #[cfg(feature = "turboquant-gpu")]
        let dense_post_attn_will_fuse = !fusion_active
            && std::env::var("LUMEN_ENABLE_DENSE_POST_ATTN_FUSION")
                .map(|v| v == "1")
                .unwrap_or(false)
            && self.mlp.supports_dense_post_attn_fusion();
        #[cfg(not(feature = "turboquant-gpu"))]
        let dense_post_attn_will_fuse = false;

        // Workstream B Phase 5 (2026-05-08): when `bf16_rmsnorm_active` and
        // the standard non-fused / dense-residual-fused Dense path will run,
        // emit `mlp_in` as bf16 (via `apply_rms_norm_bf16_out`) so the MLP's
        // gate_up matmul can read it natively (`forward_bf16_in` Affine4 fast
        // path = `affine4_qmv_fast_bf16in_bf16out`). Saves load+store BW on
        // the largest matmul inside SharedExpert. Output stays f32 — the
        // residual contract is preserved by the bf16-in MLP variants.
        //
        // Skipped (falls back to f32 norm) when:
        //   - `fusion_active`           : MoE forward_with_rmsnorm consumes
        //                                 raw `h` with internal f32 norm
        //   - `dense_post_attn_will_fuse`: dense fully-fused kernel does
        //                                 the same on raw `h`
        //   - MLP doesn't support a bf16-in fast path (e.g. MoE arm or
        //     non-Affine4 Dense gate_up_proj — those would just defer
        //     through a cast with no BW win, so we skip the bf16-norm).
        #[cfg(feature = "turboquant-gpu")]
        let bf16_mlp_in_active = bf16_rmsnorm_active
            && !fusion_active
            && !dense_post_attn_will_fuse
            && self.mlp.supports_bf16_in_fast_path();
        #[cfg(not(feature = "turboquant-gpu"))]
        let bf16_mlp_in_active = false;

        // dispatch by `h` dtype directly. When bf16-residual is
        // active (model-wide carrier), `h` is bf16 → bf16-in shader. When
        // legacy f32 stream, `h` is f32 → f32-in shader (or plain candle
        // layernorm). The earlier B.9 `h.to_dtype(F32)` cast is removed —
        // the bf16-in shader consumes bf16 natively.
        #[cfg(feature = "turboquant-gpu")]
        let mlp_in_opt = if fusion_active || dense_post_attn_will_fuse {
            None
        } else if bf16_mlp_in_active {
            if h.dtype() == candle_core::DType::BF16 {
                Some(apply_rms_norm_bf16_in_bf16_out(
                    &h,
                    self.post_attention_layernorm.weight(),
                    self.post_attention_layernorm.eps() as f32,
                )?)
            } else {
                Some(apply_rms_norm_bf16_out(
                    &h,
                    self.post_attention_layernorm.weight(),
                    self.post_attention_layernorm.eps() as f32,
                )?)
            }
        } else {
            Some(self.post_attention_layernorm.forward(&h)?)
        };
        #[cfg(not(feature = "turboquant-gpu"))]
        let mlp_in_opt = Some(self.post_attention_layernorm.forward(&h)?);
        let t3 = mark(timing);
        // Lever L1 Step 2: when residual fusion is active AND the MoE rmsnorm-
        // fused path is in use, fold the layer-level `(h + mlp_out)?` into the
        // MoE's final tri_add by passing `Some(&h)` as residual to
        // `forward_with_rmsnorm`. mlp_out then ALREADY contains the residual.
        #[cfg(feature = "turboquant-gpu")]
        let moe_residual_fused = residual_fusion_active && fusion_active;
        #[cfg(not(feature = "turboquant-gpu"))]
        let moe_residual_fused = false;

        // Lever L4: only fold next layer's input_layernorm when MoE residual
        // fusion is also active (we need the fused kernel that produces both
        // out and attn_in). Otherwise fall back to the Step 1+2+3.5 chain.
        let l4_active = moe_residual_fused && next_input_rmsnorm.is_some();
        // The fused `forward_with_rmsnorm` path lives only on `SparseMoeBlock`;
        // `Dense` always falls through to standard `MlpBlock::forward` consuming
        // the externally-normalised `mlp_in_opt` (since `has_mxfp4_backend = false`
        // → `fusion_active = false` for the Dense arm).
        // For Dense MLP the down_proj kernel can fold the layer-tail residual
        // add directly. Default ON; `LUMEN_DISABLE_DENSE_MLP_RESIDUAL_FUSION=1`
        // opts out for A/B regression checks.
        #[cfg(feature = "turboquant-gpu")]
        let dense_res_fusion_enabled = !fusion_active
            && !std::env::var("LUMEN_DISABLE_DENSE_MLP_RESIDUAL_FUSION")
                .map(|v| v == "1")
                .unwrap_or(false);
        #[cfg(not(feature = "turboquant-gpu"))]
        let dense_res_fusion_enabled = false;

        // P5b: Dense fully-fused post-attn path (RmsNorm + gate_up + silu*up
        // + down + residual). Empirically NEUTRAL on 27B Dense decode — the
        // qmv_fast_rmsnorm kernel's 2-pass input read cancels the saved Candle
        // RmsNorm dispatch chain. Default OFF; opt in via
        // `LUMEN_ENABLE_DENSE_POST_ATTN_FUSION=1` for hardware where dispatch
        // overhead dominates more.
        #[cfg(feature = "turboquant-gpu")]
        let dense_post_attn_fusion_enabled = !fusion_active
            && std::env::var("LUMEN_ENABLE_DENSE_POST_ATTN_FUSION")
                .map(|v| v == "1")
                .unwrap_or(false);
        #[cfg(not(feature = "turboquant-gpu"))]
        let dense_post_attn_fusion_enabled = false;

        // when bf16-residual is active, route the Dense
        // MLP through the bf16-in/bf16-out kernel pair so the residual stream
        // never demotes to f32 inside this layer. The gate already pinned us
        // to Dense + Affine4 (`supports_bf16_in_bf16_out_fast_path`). On
        // success, `mlp_out` is bf16 and the layer-level add stays bf16.
        #[cfg(feature = "turboquant-gpu")]
        let (mlp_out, attn_in_for_next, dense_res_fused) = if bf16_residual_active {
            let mlp_in = mlp_in_opt
                .as_ref()
                .expect("mlp_in present when bf16_residual_active");
            if dense_res_fusion_enabled {
                match self.mlp.forward_with_residual_bf16_in_bf16_out(mlp_in, &h) {
                    Some(res) => (res?, None, true),
                    None => (
                        self.mlp
                            .forward_bf16_in_bf16_out(mlp_in)
                            .expect("Dense+Affine4 bf16-in-bf16-out")?,
                        None,
                        false,
                    ),
                }
            } else {
                (
                    self.mlp
                        .forward_bf16_in_bf16_out(mlp_in)
                        .expect("Dense+Affine4 bf16-in-bf16-out")?,
                    None,
                    false,
                )
            }
        } else {
            match (&self.mlp, fusion_active) {
            (MlpBlock::Moe(moe), true) => {
                let (out, attn_in_opt_next) = moe.forward_with_rmsnorm(
                    &h,
                    self.post_attention_layernorm.weight(),
                    self.post_attention_layernorm.eps() as f32,
                    if moe_residual_fused { Some(&h) } else { None },
                    if l4_active { next_input_rmsnorm } else { None },
                )?;
                (out, attn_in_opt_next, false)
            }
            _ => {
                // First try the Dense fully-fused post-attn path.
                if dense_post_attn_fusion_enabled {
                    if let Some(res) = self.mlp.forward_post_attn_fused(
                        &h,
                        self.post_attention_layernorm.weight(),
                        self.post_attention_layernorm.eps() as f32,
                        &h,
                    ) {
                        // Fully fused — skip the external mlp_in_opt path entirely.
                        (res?, None, true)
                    } else if dense_res_fusion_enabled {
                        let mlp_in = mlp_in_opt.as_ref().expect("mlp_in present when fusion off");
                        // Workstream B Phase 5: bf16-in residual-fused path
                        // when mlp_in is bf16 (set above).
                        if bf16_mlp_in_active {
                            match self.mlp.forward_with_residual_bf16_in(mlp_in, &h) {
                                Some(res) => (res?, None, true),
                                None => (self.mlp.forward_bf16_in(mlp_in).expect("Dense arm")?, None, false),
                            }
                        } else if let Some(res) = self.mlp.forward_with_residual(mlp_in, &h) {
                            (res?, None, true)
                        } else {
                            (self.mlp.forward(mlp_in)?, None, false)
                        }
                    } else {
                        let mlp_in = mlp_in_opt.as_ref().expect("mlp_in present when fusion off");
                        if bf16_mlp_in_active {
                            (
                                self.mlp.forward_bf16_in(mlp_in).expect("Dense arm")?,
                                None,
                                false,
                            )
                        } else {
                            (self.mlp.forward(mlp_in)?, None, false)
                        }
                    }
                } else if dense_res_fusion_enabled {
                    let mlp_in = mlp_in_opt.as_ref().expect("mlp_in present when fusion off");
                    if bf16_mlp_in_active {
                        match self.mlp.forward_with_residual_bf16_in(mlp_in, &h) {
                            Some(res) => (res?, None, true),
                            None => (self.mlp.forward_bf16_in(mlp_in).expect("Dense arm")?, None, false),
                        }
                    } else if let Some(res) = self.mlp.forward_with_residual(mlp_in, &h) {
                        (res?, None, true)
                    } else {
                        (self.mlp.forward(mlp_in)?, None, false)
                    }
                } else {
                    let mlp_in = mlp_in_opt.as_ref().expect("mlp_in present when fusion off");
                    if bf16_mlp_in_active {
                        (
                            self.mlp.forward_bf16_in(mlp_in).expect("Dense arm")?,
                            None,
                            false,
                        )
                    } else {
                        (self.mlp.forward(mlp_in)?, None, false)
                    }
                }
            }
            }
        };
        #[cfg(not(feature = "turboquant-gpu"))]
        let (mlp_out, attn_in_for_next, dense_res_fused): (Tensor, Option<Tensor>, bool) = (
            self.mlp.forward(mlp_in_opt.as_ref().expect("mlp_in present"))?,
            None,
            false,
        );
        if do_dump {
            if let Some(d) = &dump_dir {
                let _ = super::model::dump_tensor_f32_public(&mlp_out, &format!("{d}/mlp_out.bin"));
            }
        }
        let t4 = mark(timing);
        let out = if moe_residual_fused || dense_res_fused { mlp_out } else { (&h + mlp_out)? };
        // no layer-exit cast. The model-wide bf16 carrier rides
        // through every layer in bf16; the single f32 cast happens at the
        // lm_head boundary in `model.rs`.
        let t_res2 = mark(timing);
        if timing {
            let ln_ms = t1.unwrap().duration_since(t0.unwrap()).as_secs_f64() * 1000.0;
            let attn_ms = t2.unwrap().duration_since(t1.unwrap()).as_secs_f64() * 1000.0;
            let res1_ms = t_res1.unwrap().duration_since(t2.unwrap()).as_secs_f64() * 1000.0;
            let ln2_ms = t3.unwrap().duration_since(t_res1.unwrap()).as_secs_f64() * 1000.0;
            let moe_ms = t4.unwrap().duration_since(t3.unwrap()).as_secs_f64() * 1000.0;
            let res2_ms = t_res2.unwrap().duration_since(t4.unwrap()).as_secs_f64() * 1000.0;
            let kind = if self.attention.is_linear() {
                "lin"
            } else {
                "ful"
            };
            eprintln!(
                "    L{:02} ({kind}): ln={ln_ms:.2} attn={attn_ms:.2} res1={res1_ms:.2} ln2={ln2_ms:.2} moe={moe_ms:.2} res2={res2_ms:.2}",
                self.layer_idx
            );
        }
        Ok((out, attn_in_for_next))
    }

    /// CB Phase 2: attention-only split.
    ///
    /// Runs `input_layernorm → attention → residual 1` and returns
    /// `h = x + attn_out`. Pair with [`Self::forward_moe_part`] to amortize
    /// the MoE dispatch across a batch of sequences.
    pub fn forward_attn_part(
        &mut self,
        x: &Tensor,
        pos_offset: usize,
        fa_mask: Option<&Tensor>,
        ssm_mask: Option<&Tensor>,
        compressed_kv: &mut CompressedKvHandle,
    ) -> CandleResult<Tensor> {
        let input_rmsnorm_enabled = std::env::var("LUMEN_ENABLE_INPUT_RMSNORM_FUSION")
            .map(|v| v == "1")
            .unwrap_or(false);
        #[cfg(feature = "turboquant-gpu")]
        let input_fusion_active =
            input_rmsnorm_enabled && self.attention.supports_fused_input_rmsnorm();
        #[cfg(not(feature = "turboquant-gpu"))]
        let input_fusion_active = {
            let _ = input_rmsnorm_enabled;
            false
        };

        // Workstream B Phase 5 (2026-05-08): drop the leftover
        // `feature = "mpsgraph"` requirement here — same fix as in
        // `forward` above. `apply_rms_norm_bf16_out` only needs
        // `turboquant-gpu` since Phase 2 replaced MpsRmsNormBf16Out
        // with the native shader.
        #[cfg(feature = "turboquant-gpu")]
        let bf16_rmsnorm_active = bf16_rmsnorm_enabled()
            && !input_fusion_active
            && self.attention.has_quant_input_proj()
            && x.device().is_metal();
        #[cfg(not(feature = "turboquant-gpu"))]
        let bf16_rmsnorm_active = {
            let _ = bf16_rmsnorm_enabled();
            false
        };

        let attn_in_opt = if input_fusion_active {
            None
        } else if bf16_rmsnorm_active {
            #[cfg(feature = "turboquant-gpu")]
            {
                Some(apply_rms_norm_bf16_out(
                    x,
                    self.input_layernorm.weight(),
                    self.input_layernorm.eps() as f32,
                )?)
            }
            #[cfg(not(feature = "turboquant-gpu"))]
            {
                Some(self.input_layernorm.forward(x)?)
            }
        } else {
            Some(self.input_layernorm.forward(x)?)
        };

        // Lever L1 (residual fusion): same logic as `Self::forward`. Skip the
        // layer-level `(x + r)?` add when active and let the o_proj kernel
        // fold it in.
        #[cfg(feature = "turboquant-gpu")]
        let residual_fusion_active = !input_fusion_active
            && !bf16_rmsnorm_active
            && super::self_attn::residual_fusion_enabled();
        #[cfg(not(feature = "turboquant-gpu"))]
        let residual_fusion_active = false;

        #[cfg(feature = "turboquant-gpu")]
        let r = if residual_fusion_active {
            self.attention.forward_with_residual_fused(
                attn_in_opt
                    .as_ref()
                    .expect("attn_in present when residual fusion is active"),
                x,
                pos_offset,
                fa_mask,
                ssm_mask,
                compressed_kv,
            )?
        } else if input_fusion_active {
            self.attention.forward_with_input_rmsnorm_with_tq(
                x,
                self.input_layernorm.weight(),
                self.input_layernorm.eps() as f32,
                pos_offset,
                fa_mask,
                ssm_mask,
                compressed_kv,
            )?
        } else {
            self.attention.forward_with_tq(
                attn_in_opt
                    .as_ref()
                    .expect("attn_in present when input fusion off"),
                pos_offset,
                fa_mask,
                ssm_mask,
                compressed_kv,
            )?
        };
        #[cfg(not(feature = "turboquant-gpu"))]
        let r = self.attention.forward_with_tq(
            attn_in_opt
                .as_ref()
                .expect("attn_in present when input fusion off"),
            pos_offset,
            fa_mask,
            ssm_mask,
            compressed_kv,
        )?;

        if residual_fusion_active {
            Ok(r)
        } else {
            Ok((x + r)?)
        }
    }

    /// CB Phase 2: MoE-only split.
    ///
    /// Runs `post_attention_layernorm → MoE → residual 2` on `h` (the
    /// post-attention residual from [`Self::forward_attn_part`]) and returns
    /// `out = h + mlp_out`. Safe to call with a batched `[B, 1, hidden]`
    /// tensor — MoE routing and expert matmuls are independent across tokens.
    pub fn forward_moe_part(&mut self, h: &Tensor) -> CandleResult<Tensor> {
        let rmsnorm_fusion_disabled = std::env::var("LUMEN_DISABLE_RMSNORM_FUSION")
            .map(|v| v == "1")
            .unwrap_or(false);
        #[cfg(feature = "turboquant-gpu")]
        let fusion_active = !rmsnorm_fusion_disabled && self.mlp.has_mxfp4_backend();
        #[cfg(not(feature = "turboquant-gpu"))]
        let fusion_active = {
            let _ = rmsnorm_fusion_disabled;
            false
        };

        #[cfg(feature = "turboquant-gpu")]
        let mlp_in_opt = if fusion_active {
            None
        } else {
            Some(self.post_attention_layernorm.forward(h)?)
        };
        #[cfg(not(feature = "turboquant-gpu"))]
        let mlp_in_opt = Some(self.post_attention_layernorm.forward(h)?);

        // Lever L1 Step 2: same MoE-side residual fusion as in `Self::forward`.
        // Pass `Some(h)` as residual when active so MoE folds `(h + mlp_out)?`
        // into its final tri_add — and skip the layer-level add here.
        #[cfg(feature = "turboquant-gpu")]
        let moe_residual_fused =
            super::self_attn::residual_fusion_enabled() && fusion_active;
        #[cfg(not(feature = "turboquant-gpu"))]
        let moe_residual_fused = false;

        // Same MlpBlock dispatch as `forward_with_tq`: the fused MoE rmsnorm
        // path is only available when the inner block is `Moe` and fusion is
        // enabled. Dense (Qwen3.6-27B) and CPU fixture tests always fall through
        // to the standard external-RmsNorm + `MlpBlock::forward` branch.
        #[cfg(feature = "turboquant-gpu")]
        let mlp_out = match (&self.mlp, fusion_active) {
            (MlpBlock::Moe(moe), true) => {
                let (out, _) = moe.forward_with_rmsnorm(
                    h,
                    self.post_attention_layernorm.weight(),
                    self.post_attention_layernorm.eps() as f32,
                    if moe_residual_fused { Some(h) } else { None },
                    None,
                )?;
                out
            }
            _ => self
                .mlp
                .forward(mlp_in_opt.as_ref().expect("mlp_in present when fusion off"))?,
        };
        #[cfg(not(feature = "turboquant-gpu"))]
        let mlp_out = self
            .mlp
            .forward(mlp_in_opt.as_ref().expect("mlp_in present"))?;

        if moe_residual_fused {
            Ok(mlp_out)
        } else {
            Ok((h + mlp_out)?)
        }
    }

    pub fn reset_cache(&mut self) {
        self.attention.reset_cache();
    }

    pub fn enable_kv_cache(&mut self, max_seq_len: usize) {
        self.attention.enable_kv_cache(max_seq_len);
    }

    pub fn set_current_seq_id(&mut self, seq_id: u64) {
        self.attention.set_current_seq_id(seq_id);
    }

    pub fn init_seq_kv_cache(&mut self, seq_id: u64) {
        self.attention.init_seq_kv_cache(seq_id);
    }

    pub fn remove_seq_kv_cache(&mut self, seq_id: u64) {
        self.attention.remove_seq_kv_cache(seq_id);
    }

    pub fn attention_mut(&mut self) -> &mut AttentionBlock {
        &mut self.attention
    }

    pub fn snapshot_state(&self) -> CandleResult<AttentionBlockSnapshot> {
        self.attention.snapshot_state()
    }

    pub fn restore_state(
        &mut self,
        snap: &AttentionBlockSnapshot,
        compressed_kv: &mut CompressedKvHandle,
    ) -> CandleResult<()> {
        self.attention.restore_state(snap, compressed_kv)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen3_5_moe::linear_attn::{
        GatedDeltaNet, GatedDeltaNetRuntime, LinearAttnDims, conv1d_from_mlx_weight,
    };
    use crate::qwen3_5_moe::moe::{
        MoeDims, SharedExpert, SparseMoeBlock, SparseMoeRuntime, SwitchMlp,
    };
    use crate::qwen3_5_moe::self_attn::{SelfAttention, SelfAttnDims, SelfAttnRuntime};

    use candle_core::{Device, Tensor};
    use candle_nn::{Linear, RmsNorm};
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    fn rnd(shape: &[usize], rng: &mut StdRng, device: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| rng.random_range(-0.1..0.1)).collect();
        Tensor::from_vec(data, shape, device).unwrap()
    }

    /// Build a consistent set of dims where every block's `hidden_size` agrees so the layer
    /// composition can chain. Keeping these tiny keeps CPU CI under a second.
    const HIDDEN: usize = 16;

    fn self_attn_dims() -> SelfAttnDims {
        SelfAttnDims {
            hidden_size: HIDDEN,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            attn_output_gate: true,
            rotary_dim: 4,
        }
    }

    fn linear_attn_dims() -> LinearAttnDims {
        LinearAttnDims {
            hidden_size: HIDDEN,
            num_k_heads: 2,
            num_v_heads: 4,
            head_dim: 4,
            conv_kernel: 4,
        }
    }

    fn moe_dims() -> MoeDims {
        MoeDims {
            hidden_size: HIDDEN,
            num_experts: 6,
            moe_intermediate_size: 12,
            shared_expert_intermediate_size: 10,
        }
    }

    fn build_self_attn(seed: u64, device: &Device) -> SelfAttention {
        let d = self_attn_dims();
        let mut r = StdRng::seed_from_u64(seed);
        // Option M2: pre-fused [q_out + 2*kv_out, hidden] qkv weight.
        let combined_out = d.q_out_dim() + 2 * d.kv_out_dim();
        let qkv = Linear::new(rnd(&[combined_out, d.hidden_size], &mut r, device), None);
        let o = Linear::new(
            rnd(&[d.hidden_size, d.attn_value_dim()], &mut r, device),
            None,
        );
        let ones = Tensor::from_vec(vec![1f32; d.head_dim], (d.head_dim,), device).unwrap();
        SelfAttention::new(
            SelfAttnRuntime {
                dims: d,
                rope_theta: 10_000.0,
                rms_norm_eps: 1e-6,
            },
            qkv.into(),
            o.into(),
            RmsNorm::new(ones.clone(), 1e-6),
            RmsNorm::new(ones, 1e-6),
        )
    }

    fn build_linear_attn(seed: u64, device: &Device) -> GatedDeltaNet {
        let d = linear_attn_dims();
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
            GatedDeltaNetRuntime {
                dims: d,
                rms_norm_eps: 1e-6,
            },
            in_proj.into(),
            conv,
            a_log,
            dt_bias,
            norm_w,
            out.into(),
        )
    }

    fn build_moe(seed: u64, device: &Device) -> SparseMoeBlock {
        let d = moe_dims();
        let mut r = StdRng::seed_from_u64(seed);
        let gate = Linear::new(rnd(&[d.num_experts, d.hidden_size], &mut r, device), None);
        let seg = Linear::new(rnd(&[1, d.hidden_size], &mut r, device), None);
        // Option J: pre-fused [2*inter, hidden] gate+up.
        let shared = SharedExpert::new(
            Linear::new(
                rnd(
                    &[2 * d.shared_expert_intermediate_size, d.hidden_size],
                    &mut r,
                    device,
                ),
                None,
            )
            .into(),
            Linear::new(
                rnd(
                    &[d.hidden_size, d.shared_expert_intermediate_size],
                    &mut r,
                    device,
                ),
                None,
            )
            .into(),
            d.shared_expert_intermediate_size,
        );
        let switch = SwitchMlp::new(
            rnd(
                &[d.num_experts, d.moe_intermediate_size, d.hidden_size],
                &mut r,
                device,
            ),
            rnd(
                &[d.num_experts, d.moe_intermediate_size, d.hidden_size],
                &mut r,
                device,
            ),
            rnd(
                &[d.num_experts, d.hidden_size, d.moe_intermediate_size],
                &mut r,
                device,
            ),
            d,
        )
        .unwrap();
        SparseMoeBlock::new(
            SparseMoeRuntime {
                dims: d,
                top_k: 3,
                norm_topk_prob: true,
            },
            gate.into(),
            seg.into(),
            shared,
            switch.into(),
        )
    }

    fn build_norm(device: &Device) -> RmsNorm {
        let w = Tensor::from_vec(vec![1f32; HIDDEN], (HIDDEN,), device).unwrap();
        RmsNorm::new(w, 1e-6)
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
    fn full_attention_layer_forward_returns_hidden_shape() {
        let device = Device::Cpu;
        let mut layer = DecoderLayer::new(
            build_norm(&device),
            AttentionBlock::Full(build_self_attn(0xAA, &device)),
            build_norm(&device),
            build_moe(0xBB, &device),
        );
        assert_eq!(layer.attention().kind(), LayerType::FullAttention);
        assert!(!layer.is_linear());

        let mut r = StdRng::seed_from_u64(0xCC);
        let x = rnd(&[1, 3, HIDDEN], &mut r, &device);
        let y = layer.forward(&x, 0, None, None).unwrap();
        assert_eq!(y.dims(), &[1, 3, HIDDEN]);
        assert!(is_finite(&y));
    }

    #[test]
    fn linear_attention_layer_forward_returns_hidden_shape() {
        let device = Device::Cpu;
        let mut layer = DecoderLayer::new(
            build_norm(&device),
            AttentionBlock::Linear(build_linear_attn(0x11, &device)),
            build_norm(&device),
            build_moe(0x22, &device),
        );
        assert_eq!(layer.attention().kind(), LayerType::LinearAttention);
        assert!(layer.is_linear());

        let mut r = StdRng::seed_from_u64(0x33);
        let x = rnd(&[1, 4, HIDDEN], &mut r, &device);
        let y = layer.forward(&x, 0, None, None).unwrap();
        assert_eq!(y.dims(), &[1, 4, HIDDEN]);
        assert!(is_finite(&y));
    }

    /// Invariant: a layer configured with **identity norms** + **zero attention output** +
    /// **zero MLP output** must preserve its input bit-exactly through the residual adds.
    /// We can't easily zero attention/MLP in-place, but we CAN check the inverse — that when
    /// the input is a known vector and norms are identity, the output equals the input plus
    /// real (non-zero) contributions from the sub-blocks. A buggy "norm-after-residual" wiring
    /// would destroy this addititve structure.
    ///
    /// Concrete check: `y - x` must equal the sum of the two residual branches' outputs.
    #[test]
    fn residual_structure_matches_pre_norm_formula() {
        let device = Device::Cpu;
        let mut layer = DecoderLayer::new(
            build_norm(&device),
            AttentionBlock::Full(build_self_attn(0x5A, &device)),
            build_norm(&device),
            build_moe(0x6B, &device),
        );
        let mut r = StdRng::seed_from_u64(0x7C);
        let x = rnd(&[1, 2, HIDDEN], &mut r, &device);
        let y = layer.forward(&x, 0, None, None).unwrap();

        // Recompute the two residual branches independently and verify they sum to (y - x).
        let attn_branch = layer
            .attention
            .forward(&layer.input_layernorm.forward(&x).unwrap(), 0, None, None)
            .unwrap();
        let h = (&x + &attn_branch).unwrap();
        let mlp_branch = layer
            .mlp
            .forward(&layer.post_attention_layernorm.forward(&h).unwrap())
            .unwrap();
        let expected = (&attn_branch + &mlp_branch).unwrap();
        let diff = ((&y - &x).unwrap() - expected).unwrap();
        let max_abs = diff
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            max_abs < 1e-5,
            "residual structure broken: max_abs = {max_abs}"
        );
    }
}
