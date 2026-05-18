//! Dimension derivation and tensor-shape validation for the Qwen3.5-VL-MoE MLP block.
//!
//! Every decoder layer owns an identical MoE MLP with this anatomy:
//! - `gate`               — router logits (`num_experts` outputs, int8-affine per override)
//! - `shared_expert_gate` — scalar sigmoid mixing shared-expert vs routed-expert contributions
//!                          (single output, int8-affine per override)
//! - `shared_expert.{gate,up,down}_proj` — a dense SwiGLU FFN always evaluated every token
//! - `switch_mlp.{gate,up,down}_proj`    — a **single grouped tensor** containing all 256 experts
//!                                          along the leading axis
//!                                          (`[num_experts, inter, hidden]` etc.)
//!
//! ## Critical loader invariants observed from the real shard headers
//! - `switch_mlp.*` is NOT split per expert (`.experts.0.*`, `.experts.1.*`, …). It's one
//!   3-D tensor whose leading dim is the expert axis. The Rust loader must NOT iterate
//!   `experts.{i}` — see the `mxfp4_mlx_storage_gotchas` memory note.
//! - `gate` and `shared_expert_gate` use **int8-affine** quantization (`bits=8, group=64`),
//!   so their `.scales` and `.biases` are **BF16** rather than the U8/E8M0 layout used by
//!   MXFP4 weights.
//! - `gate` output is `[num_experts]` per token (router logits, pre-softmax top-k).
//! - `shared_expert_gate` output is `[1]` per token (sigmoid mixing coefficient).
//!
//! ## MXFP4 vs int8-affine packing rules (packed storage last-dim → logical last-dim)
//! - MXFP4     (bits=4, group=32, `.weight` stored as U32): logical = packed × 8
//! - Int8-aff. (bits=8, group=64, `.weight` stored as U32): logical = packed × 4
//! Both reduce `packed_last_dim` by a factor of `32 / bits`.
//!
//! Ground truth shapes below were read from `model-00001-of-00004.safetensors` layer 0 on
//! 2026-04-23.

use candle_core::{Result as CandleResult, Tensor, D};
use candle_nn::Module;
use std::sync::Mutex;

#[cfg(feature = "turboquant-gpu")]
use std::sync::Arc;
#[cfg(feature = "turboquant-gpu")]
use lumen_metal::affine4_gpu::{Affine4Context, Affine4Weight};
use lumen_metal::metal::{BatchedEncoderExt, Buffer, ComputeEncoderCompat, IndirectCommandBuffer};
use lumen_metal::mxfp4_gpu::MxFp4Context;
use lumen_metal::silu_mul::SiluMulBf16InBf16Out;
#[cfg(feature = "turboquant-gpu")]
use lumen_metal::mxfp4_linear::{ExpertProj, Mxfp4SwitchMlp};

use super::config::TextConfig;
use super::proj::ProjLinear;

/// Scalar dimensions that fully determine every MoE weight shape in a single layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeDims {
    pub hidden_size: usize,
    pub num_experts: usize,
    /// Per-expert SwiGLU intermediate size (`moe_intermediate_size`).
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
}

impl MoeDims {
    pub fn from_config(t: &TextConfig) -> Self {
        Self {
            hidden_size: t.hidden_size,
            num_experts: t.num_experts,
            moe_intermediate_size: t.moe_intermediate_size,
            shared_expert_intermediate_size: t.shared_expert_intermediate_size,
        }
    }

    /// Logical (post-dequant) tensor shapes for every MoE weight in a single layer.
    pub fn shapes(self) -> MoeShapes {
        MoeShapes {
            gate: vec![self.num_experts, self.hidden_size],
            shared_expert_gate: vec![1, self.hidden_size],
            shared_expert_gate_proj: vec![self.shared_expert_intermediate_size, self.hidden_size],
            shared_expert_up_proj: vec![self.shared_expert_intermediate_size, self.hidden_size],
            shared_expert_down_proj: vec![self.hidden_size, self.shared_expert_intermediate_size],
            switch_mlp_gate_proj: vec![
                self.num_experts,
                self.moe_intermediate_size,
                self.hidden_size,
            ],
            switch_mlp_up_proj: vec![
                self.num_experts,
                self.moe_intermediate_size,
                self.hidden_size,
            ],
            switch_mlp_down_proj: vec![
                self.num_experts,
                self.hidden_size,
                self.moe_intermediate_size,
            ],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeShapes {
    pub gate: Vec<usize>,
    pub shared_expert_gate: Vec<usize>,
    pub shared_expert_gate_proj: Vec<usize>,
    pub shared_expert_up_proj: Vec<usize>,
    pub shared_expert_down_proj: Vec<usize>,
    pub switch_mlp_gate_proj: Vec<usize>,
    pub switch_mlp_up_proj: Vec<usize>,
    pub switch_mlp_down_proj: Vec<usize>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward pass
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime configuration for a single `SparseMoeBlock`. Split from [`MoeDims`] because the
/// Qwen3-Next routing knobs (`top_k`, `norm_topk_prob`) are policy, not shape.
#[derive(Debug, Clone, Copy)]
pub struct SparseMoeRuntime {
    pub dims: MoeDims,
    /// Number of experts selected per token (`num_experts_per_tok`). MLX default 8.
    pub top_k: usize,
    /// If true, re-normalize the `top_k` gate probabilities so they sum to 1 per token. The
    /// MLX `TextModelArgs` default is `True` and the Qwen3.5-MoE checkpoint inherits it.
    pub norm_topk_prob: bool,
}

impl SparseMoeRuntime {
    pub fn from_text_config(t: &TextConfig) -> Self {
        // The Qwen3.5-MoE HF config doesn't serialize `norm_topk_prob`; MLX hardcodes the
        // default `True`. We mirror that so the reference path stays bit-faithful.
        Self {
            dims: MoeDims::from_config(t),
            top_k: t.num_experts_per_tok,
            norm_topk_prob: true,
        }
    }
}

/// Dense SwiGLU feed-forward block used as the shared expert (always-on, complementary to the
/// routed sparse experts).
///
/// **Gate+Up load-time fusion (Option J, 2026-04-25):** `gate_proj` and `up_proj` are
/// concatenated on axis 0 at load time into one `[2 * intermediate, hidden]` projection.
/// A single matmul produces both halves, which the forward path narrows back into `gate`
/// and `up`. Saves one MXFP4 dispatch per layer (~0.5 ms × 40 layers = ~20 ms / token)
/// because shared_expert runs unconditionally for every token, regardless of routing.
pub struct SharedExpert {
    pub gate_up_proj: ProjLinear,
    pub down_proj: ProjLinear,
    intermediate_size: usize,
    /// per-MLP-block ICB chain cache. Lazy-init on first
    /// `forward_with_residual_bf16_in_bf16_out_mlp_icb` call when env-gate
    /// `LUMEN_MLP_ICB=1` AND both projections are Affine4 + qmv_fast.
    /// Disambiguation A/B/C measurements (mlp_block_icb_poc): Δ med
    /// -3.84% σ=+2.86 vs 8-dispatch standard, Δ med -4.37% σ=+3.47 vs
    /// fused-no-ICB 3-dispatch (i.e., ICB itself contributes the gain,
    /// not the dispatch reduction).
    #[cfg(feature = "turboquant-gpu")]
    mlp_icb_cache: Mutex<Option<MlpIcbCache>>,
}

#[cfg(feature = "turboquant-gpu")]
struct MlpIcbCache {
    /// 3 slots: gate_up_proj | silu*mul | down_proj_residual.
    icb: IndirectCommandBuffer,
    /// Cached silu*mul kernel (pipeline + ICB pipeline).
    silu_kernel: SiluMulBf16InBf16Out,
    /// Persistent intermediate buffers — pre-zeroed via synchronous
    /// `MetalContext::buffer_zeroed` to avoid the Candle `Tensor::zeros`
    /// async-fill race observed in 17.D-1c.
    combined_buf: Buffer,           // [m, 2*inter] — gate_up output
    hidden_buf: Buffer,             // [m, inter]   — silu*mul output
    gate_up_dims_buf: Buffer,       // Affine4Dims for slot 0
    down_dims_buf: Buffer,          // Affine4Dims for slot 2
    silu_dims_buf: Buffer,          // SiluMulDims for slot 1
    batch_buf: Buffer,              // u32 for slots 0+2
    /// Bound buffer addresses (gpuAddress) — fast-path validation.
    bound_x_id: usize,
    bound_x_off: u64,
    bound_r_id: usize,
    bound_r_off: u64,
    bound_y_id: usize,
    bound_y_off: u64,
    bound_batch: usize,
    recorded: bool,
}

#[cfg(feature = "turboquant-gpu")]
fn buffer_gpu_id(b: &Buffer) -> usize {
    use objc2_metal::MTLBuffer as _;
    b.as_ref().gpuAddress() as usize
}

#[cfg(feature = "turboquant-gpu")]
#[repr(C)]
#[derive(Clone, Copy)]
struct Affine4Dims {
    out_features: u32,
    in_features: u32,
}

impl SharedExpert {
    pub fn new(
        gate_up_proj: ProjLinear,
        down_proj: ProjLinear,
        intermediate_size: usize,
    ) -> Self {
        Self {
            gate_up_proj,
            down_proj,
            intermediate_size,
            #[cfg(feature = "turboquant-gpu")]
            mlp_icb_cache: Mutex::new(None),
        }
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        self.forward_with_marks(x, None)
    }

    /// Dense MLP fully fused: reads RAW post-attention `x_raw` (before
    /// `post_attention_layernorm`) + the layernorm weight, fuses RmsNorm into
    /// `gate_up_proj` (qmv_fast_rmsnorm), splits + silu*up, then fuses the
    /// final `down_proj + residual` via qmv_fast_residual. Saves up to ~6
    /// dispatches per layer × 64 layers = ~384 dispatches/token.
    /// Only available when both gate_up_proj and down_proj are Affine4 + qmv_fast
    /// supports the shape — caller verifies via [`forward_with_post_attn_fusion_supported`].
    #[cfg(feature = "turboquant-gpu")]
    fn forward_post_attn_fused(
        &self,
        x_raw: &Tensor,
        post_attn_ln_weight: &Tensor,
        rms_eps: f32,
        residual: &Tensor,
    ) -> CandleResult<Tensor> {
        // gate_up matmul with in-kernel RmsNorm.
        let gate_up_lin = self.gate_up_proj.as_affine4().expect(
            "forward_post_attn_fused requires Affine4 gate_up_proj",
        );
        let combined = gate_up_lin
            .forward_with_rmsnorm(x_raw, post_attn_ln_weight, rms_eps)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
        let last = combined.dims().len() - 1;
        let gate = combined
            .narrow(last, 0, self.intermediate_size)?
            .contiguous()?;
        let up = combined
            .narrow(last, self.intermediate_size, self.intermediate_size)?
            .contiguous()?;
        let hidden = (candle_nn::ops::silu(&gate)? * up)?;
        // Fused down_proj + residual.
        self.down_proj.forward_with_residual(&hidden, residual)
    }

    /// Detect whether the fully-fused post-attention path applies for the
    /// current shapes. Required: both projections Affine4 with qmv_fast support.
    #[cfg(feature = "turboquant-gpu")]
    fn forward_with_post_attn_fusion_supported(&self) -> bool {
        let gate_ok = self.gate_up_proj.as_affine4().map_or(false, |l| {
            lumen_metal::affine4_gpu::Affine4Context::qmv_fast_supports(
                l.in_features(),
                l.out_features(),
            )
        });
        let down_ok = self.down_proj.as_affine4().map_or(false, |l| {
            lumen_metal::affine4_gpu::Affine4Context::qmv_fast_supports(
                l.in_features(),
                l.out_features(),
            )
        });
        gate_ok && down_ok
    }

    /// Workstream B Phase 5 (2026-05-08) — bf16-input forward.
    ///
    /// Companion to `forward_bf16_in_bf16_out` on `gate_up_proj`: when the
    /// post-attention chain is bf16 (via `apply_rms_norm_bf16_out` on
    /// `post_attention_layernorm`), the gate_up matmul reads bf16 input AND
    /// writes bf16 output natively (Affine4 fast path = `affine4_qmv_fast_bf16in_bf16out`).
    /// We then cast bf16→f32 once for `silu(gate)*up` (Dense f32 ops) and
    /// route the resulting `hidden` through the f32 `down_proj`.
    ///
    /// Output dtype: f32 (preserves the residual contract — until B.6+
    /// extends bf16 to the residual itself).
    ///
    /// Affine4 path: native bf16-in-bf16-out kernel — saves 50% load + 50%
    /// store BW on the gate_up matmul, ~halves the largest single op cost
    /// inside SharedExpert.
    /// Mxfp4 / Dense paths: cast input to f32 inside, defer to the existing
    /// `forward_with_marks` path. Equivalent numerics, no BW win — Mxfp4
    /// gate_up bf16-in-bf16-out kernel is a future opt.
    #[cfg(feature = "turboquant-gpu")]
    fn forward_bf16_in(&self, x_bf16: &Tensor) -> CandleResult<Tensor> {
        if !self.gate_up_proj.is_affine4() {
            // Non-Affine4 → no native bf16-in-bf16-out kernel. Cast to f32
            // and run the f32 path. Avoids double-narrow / cast-cycle traps.
            let x_f32 = x_bf16.to_dtype(candle_core::DType::F32)?;
            return self.forward(&x_f32);
        }
        // Affine4 fast path. The gate_up_proj kernel produces bf16 output
        // directly; we cast to f32 once for the silu/up Dense ops.
        let combined_bf16 = self.gate_up_proj.forward_bf16_in_bf16_out(x_bf16)?;
        let combined = combined_bf16.to_dtype(candle_core::DType::F32)?;
        let last = combined.dims().len() - 1;
        let gate = combined
            .narrow(last, 0, self.intermediate_size)?
            .contiguous()?;
        let up = combined
            .narrow(last, self.intermediate_size, self.intermediate_size)?
            .contiguous()?;
        let hidden = (candle_nn::ops::silu(&gate)? * up)?;
        // down_proj stays f32 — output flows into the f32 residual stream.
        self.down_proj.forward(&hidden)
    }

    /// Workstream B Phase 5 — bf16-input + residual-fused down_proj.
    ///
    /// Mirrors `forward_with_residual` but takes bf16 `x_bf16` for the
    /// gate_up matmul, then runs `forward_with_residual` on the f32-output
    /// down_proj so the layer-tail `(h + mlp_out)` add is folded into the
    /// final matmul. Output is f32 (residual contract).
    #[cfg(feature = "turboquant-gpu")]
    fn forward_with_residual_bf16_in(
        &self,
        x_bf16: &Tensor,
        residual: &Tensor,
    ) -> CandleResult<Tensor> {
        if !self.gate_up_proj.is_affine4() {
            let x_f32 = x_bf16.to_dtype(candle_core::DType::F32)?;
            return self.forward_with_residual(&x_f32, residual);
        }
        let combined_bf16 = self.gate_up_proj.forward_bf16_in_bf16_out(x_bf16)?;
        let combined = combined_bf16.to_dtype(candle_core::DType::F32)?;
        let last = combined.dims().len() - 1;
        let gate = combined
            .narrow(last, 0, self.intermediate_size)?
            .contiguous()?;
        let up = combined
            .narrow(last, self.intermediate_size, self.intermediate_size)?
            .contiguous()?;
        let hidden = (candle_nn::ops::silu(&gate)? * up)?;
        self.down_proj.forward_with_residual(&hidden, residual)
    }

    /// Workstream B Phase 8 — bf16-in + bf16-out variant of [`Self::forward_bf16_in`].
    ///
    /// Same Affine4 fast path as B.5's `forward_bf16_in`, but `down_proj`
    /// returns bf16 instead of f32. Used by the bf16 residual stream path
    /// (B.8): the layer keeps `h` in bf16 throughout, so the MLP's output
    /// must also be bf16 to feed the next residual add without a cast cycle.
    ///
    /// Output dtype contract: BF16. Caller (layer.rs) is responsible for
    /// matching residual dtype before the `(h + mlp_out)` add.
    #[cfg(feature = "turboquant-gpu")]
    fn forward_bf16_in_bf16_out(&self, x_bf16: &Tensor) -> CandleResult<Tensor> {
        if !self.gate_up_proj.is_affine4() {
            // Non-Affine4 → cast to f32, run standard f32 path, cast result
            // back to bf16. No native fast path; semantically equivalent to
            // `forward_bf16_in` plus a final `to_dtype(BF16)`.
            let x_f32 = x_bf16.to_dtype(candle_core::DType::F32)?;
            let y_f32 = self.forward(&x_f32)?;
            return y_f32.to_dtype(candle_core::DType::BF16);
        }
        let combined_bf16 = self.gate_up_proj.forward_bf16_in_bf16_out(x_bf16)?;
        let combined = combined_bf16.to_dtype(candle_core::DType::F32)?;
        let last = combined.dims().len() - 1;
        let gate = combined
            .narrow(last, 0, self.intermediate_size)?
            .contiguous()?;
        let up = combined
            .narrow(last, self.intermediate_size, self.intermediate_size)?
            .contiguous()?;
        let hidden = (candle_nn::ops::silu(&gate)? * up)?;
        // bf16 down_proj output (`forward_bf16_out` casts the f32 matmul
        // result down to bf16 for Affine4/Mxfp4 — Dense composes via
        // `to_dtype`). The bf16 residual add downstream stays in bf16.
        self.down_proj.forward_bf16_out(&hidden)
    }

    /// Workstream B Phase 11 — bf16-in + bf16-residual + bf16-out with FUSED
    /// down_proj residual add.
    ///
    /// Closes the B.10 σ-NEGATIVE root cause: previously this method ran
    /// `forward_bf16_in_bf16_out` (which uses `down_proj.forward_bf16_out`,
    /// f32 input qmv_fast f32 path) + a separate `broadcast_add(bf16)`. Per
    /// per-shader profile, that split caused +650 ms of `affine4_qmv_fast`
    /// (f32 path) and +8 ms of `badd_bf16` per decode batch. New routing:
    /// gate_up bf16-in/bf16-out + f32 silu_mul (precise_swiglu MLX Escape #1
    /// preserved) → cast hidden bf16 → down_proj single-shot fused
    /// `forward_with_residual_bf16_in_bf16_out` (matmul + broadcast_add fused
    /// in one Metal dispatch, full bf16 chain).
    #[cfg(feature = "turboquant-gpu")]
    fn forward_with_residual_bf16_in_bf16_out(
        &self,
        x_bf16: &Tensor,
        residual_bf16: &Tensor,
    ) -> CandleResult<Tensor> {
        // Validate the bf16 contract on `residual` — caller (layer.rs) must
        // upstream-cast before invoking us. Forward residual dtype mismatch
        // is a contract bug, not a recoverable runtime case, so we surface
        // it loudly rather than silently casting.
        if residual_bf16.dtype() != candle_core::DType::BF16 {
            candle_core::bail!(
                "forward_with_residual_bf16_in_bf16_out: residual dtype {:?} != BF16",
                residual_bf16.dtype()
            );
        }

        // per-MLP-block ICB chain (gate_up + silu*mul + down).
        // Disambiguation A/B/C: ICB contributes σ=+3.47 vs fused-no-ICB
        // (i.e., the gain is from ICB itself, not just dispatch reduction).
        // Env-gated to allow a clean A/B until production rollout.
        #[cfg(feature = "turboquant-gpu")]
        {
            let mlp_icb_on = std::env::var("LUMEN_MLP_ICB")
                .map(|v| v == "1")
                .unwrap_or(false);
            if mlp_icb_on
                && self.gate_up_proj.is_affine4()
                && self.down_proj.is_affine4()
            {
                if let Some(out) = self.try_forward_mlp_icb(x_bf16, residual_bf16)? {
                    return Ok(out);
                }
            }
        }

        // Non-Affine4 backends fall back to the prior compose path.
        // 27B Dense production target is Affine4 — this branch only fires
        // for fixture/Mxfp4 paths.
        if !self.gate_up_proj.is_affine4() || !self.down_proj.is_affine4() {
            let y_bf16 = self.forward_bf16_in_bf16_out(x_bf16)?;
            return y_bf16.broadcast_add(residual_bf16);
        }

        // gate_up bf16-in/bf16-out (Affine4 qmv_fast fast path) + f32 silu_mul.
        //
        // NOTE on dispatch chain choice (Phase 12+13 NEGATIVE benchmarks,
        // both reverted): two attempts to replace this 5-dispatch chain with
        // a single fused gate_up_silu_mul_bf16_out kernel both measured σ
        // STRONG NEGATIVE:
        //   - Phase 12 used the v3 family kernel (256 threads/TG, 2-pass
        //     input read via threadgroup memory): σ=-8.73 (Δ=+8.97% slower).
        //   - Phase 13 used a custom qmv_fast-architecture fused kernel
        //     (64 threads/TG, single-pass, gate+up shared x_thread): σ=-15.87
        //     (Δ=+1.02% slower). Despite saving ~4 dispatches/layer, the
        //     2x register pressure (8 result floats per simdgroup vs 4) and
        //     enlarged inner loop offset the dispatch savings.
        //
        // Confirms `qmv_fast_port_landed` P5b: qmv_fast (64 thread / single-
        // pass) + separate Candle silu+mul beats any fused gate_up_silu_mul
        // variant on Apple Silicon for 27B Dense decode shape. The qmv_fast
        // 4-bit decode arithmetic per-output is already at the kernel's
        // BW/register pareto front.
        let combined_bf16 = self.gate_up_proj.forward_bf16_in_bf16_out(x_bf16)?;
        let combined = combined_bf16.to_dtype(candle_core::DType::F32)?;
        let last = combined.dims().len() - 1;
        let gate = combined
            .narrow(last, 0, self.intermediate_size)?
            .contiguous()?;
        let up = combined
            .narrow(last, self.intermediate_size, self.intermediate_size)?
            .contiguous()?;
        // f32 silu_mul (precise_swiglu — MLX Escape #1, kept f32 for accuracy).
        let hidden_f32 = (candle_nn::ops::silu(&gate)? * up)?;
        // Single-shot fused down_proj matmul + bf16 residual add. Internal cast
        // f32→bf16 on `hidden_f32` happens inside the dispatcher (single small
        // dispatch on `[B,L,intermediate]`). The fused kernel replaces the prior
        // `forward_bf16_out + broadcast_add(bf16)` 2-dispatch path.
        self.down_proj
            .forward_with_residual_bf16_in_bf16_out(&hidden_f32, residual_bf16)
    }

    /// Same as `forward`, but the final `down_proj` matmul also adds `residual`
    /// to its output in a single fused dispatch — saves one downstream
    /// `broadcast_add` per call. Used by the Dense MLP layer-tail residual fold.
    /// Only safe when bf16-out / shared-down bf16 paths are NOT active (those
    /// produce bf16 outputs that need a cast before residual add anyway).
    #[cfg(feature = "turboquant-gpu")]
    fn forward_with_residual(&self, x: &Tensor, residual: &Tensor) -> CandleResult<Tensor> {
        // Compute `hidden = silu(gate)*up` exactly like `forward_with_marks` does
        // up to the down-projection, then call the residual-fused down_proj.
        let enable_fusion = std::env::var("LUMEN_ENABLE_GATE_UP_SILU_MUL_FUSION")
            .map(|v| v == "1")
            .unwrap_or(false);
        let bf16_out = bf16_out_enabled();
        // bf16 paths produce non-f32 down inputs — defer to the unfused path.
        let bf16_shared_down = std::env::var("LUMEN_BF16_OUT_SHARED_DOWN")
            .map(|v| v == "1")
            .unwrap_or(false);
        if bf16_out || bf16_shared_down {
            let y = self.forward_with_marks(x, None)?;
            return y.broadcast_add(residual);
        }
        let hidden = if enable_fusion {
            self.gate_up_proj
                .forward_gate_up_silu_mul(x, self.intermediate_size)?
        } else {
            let combined = self.gate_up_proj.forward(x)?;
            let last = combined.dims().len() - 1;
            let gate = combined
                .narrow(last, 0, self.intermediate_size)?
                .contiguous()?;
            let up = combined
                .narrow(last, self.intermediate_size, self.intermediate_size)?
                .contiguous()?;
            (candle_nn::ops::silu(&gate)? * up)?
        };
        // Fused down_proj + residual add (Affine4 → tiled-residual; Mxfp4 → v3-residual;
        // Dense fallback → matmul + broadcast_add).
        self.down_proj.forward_with_residual(&hidden, residual)
    }

    /// Lever H Step 2: RmsNorm-fused SharedExpert forward. Reads RAW x (the
    /// post-attention residual, BEFORE `post_attention_layernorm`) plus the
    /// layernorm weight and runs the in-kernel cooperative RmsNorm fused into
    /// the gate_up matmul. The down projection is unaffected — it consumes
    /// `silu(gate)*up`, not `x`.
    ///
    /// Production-path-only: skips the M.1 fused-kernel branch (default OFF)
    /// and the bf16-out shared-down branch (default OFF, controlled separately).
    /// Falls back to f32 unfused gate_up matmul → narrow → silu*up → down.
    #[cfg(feature = "turboquant-gpu")]
    fn forward_with_rmsnorm(
        &self,
        x_raw: &Tensor,
        rms_weight: &Tensor,
        rms_eps: f32,
        ctx: &Arc<MxFp4Context>,
    ) -> CandleResult<Tensor> {
        // Combined matmul on raw x with internal RmsNorm: [..., 2*inter].
        let combined = self.gate_up_proj.forward_with_rmsnorm(
            x_raw,
            rms_weight,
            rms_eps,
            ctx,
        )?;
        let last = combined.dims().len() - 1;
        let gate = combined
            .narrow(last, 0, self.intermediate_size)?
            .contiguous()?;
        let up = combined
            .narrow(last, self.intermediate_size, self.intermediate_size)?
            .contiguous()?;
        let hidden = (candle_nn::ops::silu(&gate)? * up)?;
        // down_proj consumes the silu*up tensor (no RmsNorm needed): unchanged.
        self.down_proj.forward(&hidden)
    }

    /// Same as `forward`, but pushes per-sub-op markers into `marks` when provided so the
    /// caller can split the dense gate_up matmul / narrow+contiguous split / silu*up /
    /// down matmul costs. Each marker syncs the device first.
    fn forward_with_marks(
        &self,
        x: &Tensor,
        mut marks: Option<&mut Vec<(&'static str, std::time::Instant)>>,
    ) -> CandleResult<Tensor> {
        let device = x.device().clone();
        let push = |label: &'static str,
                    marks: &mut Option<&mut Vec<(&'static str, std::time::Instant)>>| {
            if let Some(m) = marks.as_deref_mut() {
                let _ = device.synchronize();
                m.push((label, std::time::Instant::now()));
            }
        };

        // fused gate+up+silu*up kernel exists but is **opt-in only**.
        // Empirical 35B measurement (2026-04-26) shows fusion regressed decode
        // latency by ~3ms (5.77 vs 5.87 tok/s) because halving `n_groups_x`
        // from `2*inter/8`=128 TGs to `inter/8`=64 TGs cuts GPU simdgroup
        // occupancy below the silu/mul kernel-saving threshold. Token sequence
        // parity remained bit-identical (cosine ≥ 0.9999 + greedy 60-token
        // match), so the kernel itself is correct — the design needs more TGs
        // to be worth landing.
        //
        // Set `LUMEN_ENABLE_GATE_UP_SILU_MUL_FUSION=1` to opt into the
        // fused path (kept for future kernel redesign / smaller-shape testing).
        let enable_fusion = std::env::var("LUMEN_ENABLE_GATE_UP_SILU_MUL_FUSION")
            .map(|v| v == "1")
            .unwrap_or(false);
        // when bf16_out_enabled(), route the unfused gate_up
        // matmul through the bf16-output kernel and cast back to f32 before
        // narrow/silu/mul. Fused path with bf16 also available.
        let bf16_out = bf16_out_enabled();
        let hidden = if enable_fusion {
            let h = if bf16_out {
                let y_bf16 = self
                    .gate_up_proj
                    .forward_gate_up_silu_mul_bf16_out(x, self.intermediate_size)?;
                y_bf16.to_dtype(candle_core::DType::F32)?
            } else {
                self.gate_up_proj
                    .forward_gate_up_silu_mul(x, self.intermediate_size)?
            };
            push("sh_gate_up_silu_mul", &mut marks);
            h
        } else {
            let combined = if bf16_out {
                let y_bf16 = self.gate_up_proj.forward_bf16_out(x)?;
                y_bf16.to_dtype(candle_core::DType::F32)?
            } else {
                self.gate_up_proj.forward(x)?
            };
            push("sh_gate_up", &mut marks);
            let last = combined.dims().len() - 1;
            let gate = combined
                .narrow(last, 0, self.intermediate_size)?
                .contiguous()?;
            let up = combined
                .narrow(last, self.intermediate_size, self.intermediate_size)?
                .contiguous()?;
            push("sh_split", &mut marks);
            let h = (candle_nn::ops::silu(&gate)? * up)?;
            push("sh_silu_mul", &mut marks);
            h
        };
        // when `LUMEN_BF16_OUT=1` (or the
        // legacy per-callsite `LUMEN_BF16_OUT_SHARED_DOWN=1`), route
        // shared_expert.down_proj through the bf16-output kernel and cast
        // back to f32 at the boundary. See `phase_a_step_a0_landed.md`.
        let bf16_out = bf16_out_enabled();
        let out = if bf16_out
            || std::env::var("LUMEN_BF16_OUT_SHARED_DOWN")
                .map(|v| v == "1")
                .unwrap_or(false)
        {
            let y_bf16 = self.down_proj.forward_bf16_out(&hidden)?;
            y_bf16.to_dtype(candle_core::DType::F32)?
        } else {
            self.down_proj.forward(&hidden)?
        };
        push("sh_down", &mut marks);
        Ok(out)
    }

    /// production per-MLP-block ICB chain.
    ///
    /// Returns `Ok(Some(out))` on the ICB fast path, `Ok(None)` if the call
    /// should fall back to the standard 8-dispatch path (shape doesn't
    /// qualify, batch != 1, etc.). Returns `Err` only on actual failure.
    ///
    /// Behaviour gated by `LUMEN_MLP_ICB=1` (checked by the caller) AND
    /// requires both projections to be Affine4 + qmv_fast_supports.
    #[cfg(feature = "turboquant-gpu")]
    fn try_forward_mlp_icb(
        &self,
        x_bf16: &Tensor,
        residual_bf16: &Tensor,
    ) -> CandleResult<Option<Tensor>> {
        use candle_core::{DType, Device, Storage};
        use lumen_metal::metal::MTLResourceUsage;

        let gate_up_lin = match self.gate_up_proj.as_affine4() {
            Some(l) => l,
            None => return Ok(None),
        };
        let down_lin = match self.down_proj.as_affine4() {
            Some(l) => l,
            None => return Ok(None),
        };
        let gate_up_w = gate_up_lin.weight();
        let down_w = down_lin.weight();

        // Shape guards — qmv_fast support + intermediate match.
        if !Affine4Context::qmv_fast_supports(gate_up_w.in_features, gate_up_w.out_features)
            || !Affine4Context::qmv_fast_supports(down_w.in_features, down_w.out_features)
        {
            return Ok(None);
        }
        if gate_up_w.out_features != 2 * self.intermediate_size
            || down_w.in_features != self.intermediate_size
        {
            return Ok(None);
        }

        let dims = x_bf16.dims();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        // Decode-shape only — prefill (batch != 1) keeps the standard path.
        if batch != 1 {
            return Ok(None);
        }

        let metal_dev = match x_bf16.device() {
            Device::Metal(m) => m,
            _ => return Ok(None),
        };

        // Materialise contiguous bf16 inputs (no-op when already bf16+contig).
        let x_b = x_bf16.contiguous()?;
        let r_b = residual_bf16.contiguous()?;

        let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
        out_shape.push(down_w.out_features);
        let y = Tensor::zeros(out_shape, DType::BF16, x_bf16.device())?;

        // Extract MTLBuffers + offsets.
        let extract = |t: &Tensor| -> CandleResult<(Buffer, u64)> {
            let (storage, layout) = t.storage_and_layout();
            match &*storage {
                Storage::Metal(ms) => {
                    let off = (layout.start_offset() * t.dtype().size_in_bytes()) as u64;
                    Ok((ms.buffer().clone(), off))
                }
                _ => Err(candle_core::Error::Msg("not metal".into())),
            }
        };
        let (x_buf, x_off) = extract(&x_b)?;
        let (r_buf, r_off) = extract(&r_b)?;
        let (y_buf, y_off) = extract(&y)?;

        let ctx = gate_up_lin.ctx().clone();

        let mut cache_guard = self
            .mlp_icb_cache
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("mlp_icb_cache lock: {e}")))?;

        if cache_guard.is_none() {
            let icb = ctx.ctx.new_indirect_command_buffer(3, 8)
                .map_err(|e| candle_core::Error::Msg(format!("ICB alloc: {e}")))?;
            let silu_kernel = SiluMulBf16InBf16Out::new()
                .map_err(|e| candle_core::Error::Msg(format!("silu kernel init: {e}")))?;

            // Synchronous-zeroed persistent intermediates (no async fill — see
            // 17.D-1c lesson on Tensor::zeros race).
            let combined_buf = ctx.ctx.buffer_zeroed((2 * self.intermediate_size * 2) as u64);
            let hidden_buf = ctx.ctx.buffer_zeroed((self.intermediate_size * 2) as u64);

            let gate_up_dims_buf = ctx.ctx.buffer_with_data(&[Affine4Dims {
                out_features: gate_up_w.out_features as u32,
                in_features: gate_up_w.in_features as u32,
            }]);
            let down_dims_buf = ctx.ctx.buffer_with_data(&[Affine4Dims {
                out_features: down_w.out_features as u32,
                in_features: down_w.in_features as u32,
            }]);
            let silu_dims_buf = silu_kernel.make_dims_buf(self.intermediate_size);
            let batch_buf = ctx.ctx.buffer_with_data(&[batch as u32]);

            *cache_guard = Some(MlpIcbCache {
                icb,
                silu_kernel,
                combined_buf,
                hidden_buf,
                gate_up_dims_buf,
                down_dims_buf,
                silu_dims_buf,
                batch_buf,
                bound_x_id: 0,
                bound_x_off: 0,
                bound_r_id: 0,
                bound_r_off: 0,
                bound_y_id: 0,
                bound_y_off: 0,
                bound_batch: batch,
                recorded: false,
            });
        }
        let cache = cache_guard.as_mut().unwrap();

        if cache.bound_batch != batch {
            cache.batch_buf = ctx.ctx.buffer_with_data(&[batch as u32]);
            cache.bound_batch = batch;
            cache.recorded = false;
        }

        let x_id = buffer_gpu_id(&x_buf);
        let r_id = buffer_gpu_id(&r_buf);
        let y_id = buffer_gpu_id(&y_buf);
        let needs_record = !cache.recorded
            || cache.bound_x_id != x_id || cache.bound_x_off != x_off
            || cache.bound_r_id != r_id || cache.bound_r_off != r_off
            || cache.bound_y_id != y_id || cache.bound_y_off != y_off;

        if needs_record {
            // Slot 0: gate_up_proj qmv_fast.
            ctx.record_qmv_fast_bf16in_bf16out_icb(
                &cache.icb, 0, gate_up_w,
                &x_buf, x_off,
                &cache.combined_buf, 0,
                &cache.gate_up_dims_buf, &cache.batch_buf, batch,
            );
            // Slot 1: silu*mul.
            cache.silu_kernel.record_icb(
                &cache.icb, 1,
                &cache.combined_buf, 0,
                &cache.hidden_buf, 0,
                &cache.silu_dims_buf, batch, self.intermediate_size,
            );
            // Slot 2: down_proj qmv_fast bf16-residual.
            ctx.record_qmv_fast_bf16in_bf16out_residual_icb(
                &cache.icb, 2, down_w,
                &cache.hidden_buf, 0,
                &r_buf, r_off,
                &y_buf, y_off,
                &cache.down_dims_buf, &cache.batch_buf, batch,
            );
            cache.bound_x_id = x_id;
            cache.bound_x_off = x_off;
            cache.bound_r_id = r_id;
            cache.bound_r_off = r_off;
            cache.bound_y_id = y_id;
            cache.bound_y_off = y_off;
            cache.recorded = true;
        }

        // Encode through Candle's shared encoder so this CB is ordered with
        // any pending Tensor allocations / zero-fills (no `synchronize`
        // needed in production — Candle's hazard tracking handles it).
        let encoder = metal_dev
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("mlp_icb encoder: {e}")))?;
        encoder.set_label("lumen:mlp_block_icb_chain");
        let usage = MTLResourceUsage(MTLResourceUsage::Read.0 | MTLResourceUsage::Write.0);
        let (gp, gs, gb) = gate_up_w.buffers();
        let (dp, ds, db) = down_w.buffers();
        encoder.use_buffers_for_icb(
            &[
                gp, gs, gb,
                &x_buf,
                &cache.combined_buf,
                &cache.silu_dims_buf,
                &cache.hidden_buf,
                dp, ds, db,
                &r_buf,
                &y_buf,
                &cache.gate_up_dims_buf,
                &cache.down_dims_buf,
                &cache.batch_buf,
            ],
            usage,
        );
        // Three serialized executes — each call is a barrier (17.D-1e finding).
        encoder.execute_commands_in_buffer_range(&cache.icb, 0, 1);
        encoder.execute_commands_in_buffer_range(&cache.icb, 1, 1);
        encoder.execute_commands_in_buffer_range(&cache.icb, 2, 1);
        drop(encoder);
        drop(cache_guard);

        Ok(Some(y))
    }
}

/// unified env-flag check for bf16-output wire-ins
/// across self_attn / linear_attn / moe ProjLinear callsites. Default OFF.
/// Set `LUMEN_BF16_OUT=1` to opt all bf16-capable callsites in at once;
/// per-callsite legacy flags still honored for narrower A/B testing.
#[inline]
pub(super) fn bf16_out_enabled() -> bool {
    std::env::var("LUMEN_BF16_OUT")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Workstream B Phase 9 (2026-05-09): opt-in for the bf16 residual stream.
/// When set, `layer.rs` keeps `h` (the layer-level residual carrier) in
/// bf16 throughout one decoder layer; the o_proj / out_proj / mlp boundary
/// casts that previously demoted to f32 are lifted, and a single cast back
/// to f32 happens at layer exit.
///
/// Default OFF — chain only kicks in when the rest of the bf16 pipeline is
/// already active (`LUMEN_BF16_RMSNORM=1`) AND the MLP arm supports a
/// bf16-in-bf16-out fast path (Dense + Affine4 gate_up_proj). Otherwise
/// the layer transparently falls back to the f32 residual stream.
#[inline]
pub(super) fn bf16_residual_enabled() -> bool {
    std::env::var("LUMEN_BF16_RESIDUAL")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Standard SwiGLU MLP block for non-MoE Qwen3.5 family checkpoints (Qwen3.6-27B).
///
/// Structurally identical to a single MoE shared-expert: gate+up fused into a
/// `[2 * intermediate_size, hidden_size]` projection at load time, the kernel
/// output split + `silu(gate) * up` mixed, then projected through
/// `[hidden_size, intermediate_size]` down. By wrapping [`SharedExpert`] we
/// inherit every optimisation already proven there — bf16-out wire-ins, the
/// gate+up+silu*mul fusion env flag, the M.1 fused-kernel branch, and the M.5
/// alloc-reuse caches — so future Lever H / bf16 / Phase A work that lands on
/// `SharedExpert` automatically applies to the dense MLP path as well.
///
/// Unlike a MoE block this carries no router, no shared-expert-gate, and no
/// per-expert switch_mlp; the per-layer activation cost is the SwiGLU pair
/// rather than `top_k` routed experts plus the shared expert.
pub struct DenseMlp {
    inner: SharedExpert,
}

impl DenseMlp {
    pub fn new(
        gate_up_proj: ProjLinear,
        down_proj: ProjLinear,
        intermediate_size: usize,
    ) -> Self {
        Self {
            inner: SharedExpert::new(gate_up_proj, down_proj, intermediate_size),
        }
    }

    /// Standard non-fused forward: external `post_attention_layernorm` already
    /// applied by the caller, this consumes the normed hidden tensor.
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Same module → access SharedExpert's private `forward` (the marks-aware
        // helper without timing). This is the SwiGLU pair `silu(gate)*up` →
        // `down_proj`, dispatched through the same fused / bf16 paths the MoE
        // shared expert uses.
        self.inner.forward(x)
    }

    /// `forward(x) + residual` in fused form: the final `down_proj` matmul folds
    /// the residual addition into its tail, saving one dispatch per layer.
    /// Used by the layer-tail residual fold for Dense MLP.
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_with_residual(&self, x: &Tensor, residual: &Tensor) -> CandleResult<Tensor> {
        self.inner.forward_with_residual(x, residual)
    }

    /// Workstream B Phase 5 — bf16-input forward (output f32). Wraps
    /// `SharedExpert::forward_bf16_in`. Used when the post-attention norm
    /// emits bf16 (via `apply_rms_norm_bf16_out`) so the gate_up matmul
    /// reads bf16 natively on the Affine4 fast path. Output stays f32 for
    /// the residual contract.
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_bf16_in(&self, x_bf16: &Tensor) -> CandleResult<Tensor> {
        self.inner.forward_bf16_in(x_bf16)
    }

    /// Workstream B Phase 5 — bf16-input + residual-fused down_proj. Mirror
    /// of `forward_with_residual` for the bf16 chain.
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_with_residual_bf16_in(
        &self,
        x_bf16: &Tensor,
        residual: &Tensor,
    ) -> CandleResult<Tensor> {
        self.inner.forward_with_residual_bf16_in(x_bf16, residual)
    }

    /// Workstream B Phase 8 — bf16-in + bf16-out forward. Output dtype is BF16
    /// for the bf16 residual stream. Wraps `SharedExpert::forward_bf16_in_bf16_out`.
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_bf16_in_bf16_out(&self, x_bf16: &Tensor) -> CandleResult<Tensor> {
        self.inner.forward_bf16_in_bf16_out(x_bf16)
    }

    /// Workstream B Phase 8 — bf16-in + bf16-residual + bf16-out. Caller must
    /// supply a bf16 residual; dtype mismatch is a contract bug surfaced loudly.
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_with_residual_bf16_in_bf16_out(
        &self,
        x_bf16: &Tensor,
        residual_bf16: &Tensor,
    ) -> CandleResult<Tensor> {
        self.inner
            .forward_with_residual_bf16_in_bf16_out(x_bf16, residual_bf16)
    }

    /// Returns `true` iff the gate_up_proj is Affine4 — the only MLP variant
    /// that has a native bf16-in-bf16-out gate_up kernel. Other variants fall
    /// back to a cast-and-defer path (still correct, no BW win).
    #[cfg(feature = "turboquant-gpu")]
    pub fn has_affine4_gate_up(&self) -> bool {
        self.inner.gate_up_proj.is_affine4()
    }

    /// Fully fused post-attention path: RmsNorm + gate_up + silu*up + down + residual.
    /// Returns `None` (caller falls back) when the qmv_fast shape constraint is
    /// not met (in % 512 == 0 + out % 8 == 0 on both projections).
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_post_attn_fused(
        &self,
        x_raw: &Tensor,
        post_attn_ln_weight: &Tensor,
        rms_eps: f32,
        residual: &Tensor,
    ) -> Option<CandleResult<Tensor>> {
        if !self.inner.forward_with_post_attn_fusion_supported() {
            return None;
        }
        Some(self
            .inner
            .forward_post_attn_fused(x_raw, post_attn_ln_weight, rms_eps, residual))
    }

    #[cfg(feature = "turboquant-gpu")]
    pub fn supports_post_attn_fusion(&self) -> bool {
        self.inner.forward_with_post_attn_fusion_supported()
    }
}

/// Per-layer MLP variant — Qwen3.5 family checkpoints share the hybrid
/// linear+full-attention backbone but split here on whether the per-layer MLP
/// is a sparse MoE (35B-A3B-mxfp4) or a standard dense SwiGLU (Qwen3.6-27B).
///
/// `DecoderLayer::mlp` is typed as `MlpBlock`; per-token forward dispatches
/// into the `Moe` arm (existing fusion + L1/L4 paths) or the `Dense` arm
/// (single SwiGLU dispatch, no fusion paths yet).
pub enum MlpBlock {
    Moe(SparseMoeBlock),
    Dense(DenseMlp),
}

impl MlpBlock {
    /// Standard non-fused forward — both arms consume an externally-normed
    /// hidden tensor (i.e. `post_attention_layernorm` already applied by
    /// the caller).
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self {
            MlpBlock::Moe(moe) => moe.forward(x),
            MlpBlock::Dense(dense) => dense.forward(x),
        }
    }

    /// `forward(x) + residual` in fused form. Available for the Dense arm
    /// when down_proj supports residual fusion (Affine4 with tile, MXFP4 v3).
    /// Returns `None` if this arm doesn't support fused-residual; caller
    /// should fall back to `forward(x) + broadcast_add(residual)`.
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_with_residual(
        &self,
        x: &Tensor,
        residual: &Tensor,
    ) -> Option<CandleResult<Tensor>> {
        match self {
            // MoE arm: residual fusion handled by `forward_with_rmsnorm`'s
            // existing `Some(&h)` parameter, not exposed here.
            MlpBlock::Moe(_) => None,
            MlpBlock::Dense(dense) => Some(dense.forward_with_residual(x, residual)),
        }
    }

    /// Workstream B Phase 5 — bf16-input forward (output f32).
    ///
    /// Available on the Dense arm only — MoE's bf16 wiring lands in B.7+
    /// where `forward_with_rmsnorm` gets a bf16-aware kernel variant. For
    /// MoE we return `None` so the caller falls back to its f32 path.
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_bf16_in(&self, x_bf16: &Tensor) -> Option<CandleResult<Tensor>> {
        match self {
            MlpBlock::Moe(_) => None,
            MlpBlock::Dense(dense) => Some(dense.forward_bf16_in(x_bf16)),
        }
    }

    /// Workstream B Phase 5 — bf16-input + residual-fused down_proj. Dense arm only.
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_with_residual_bf16_in(
        &self,
        x_bf16: &Tensor,
        residual: &Tensor,
    ) -> Option<CandleResult<Tensor>> {
        match self {
            MlpBlock::Moe(_) => None,
            MlpBlock::Dense(dense) => {
                Some(dense.forward_with_residual_bf16_in(x_bf16, residual))
            }
        }
    }

    /// Workstream B Phase 8 — bf16-in + bf16-out forward. Available on the
    /// Dense arm; MoE returns `None` (its production rmsnorm-fused path is
    /// f32-only and the bf16 wiring lands in B.7+).
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_bf16_in_bf16_out(
        &self,
        x_bf16: &Tensor,
    ) -> Option<CandleResult<Tensor>> {
        match self {
            MlpBlock::Moe(_) => None,
            MlpBlock::Dense(dense) => Some(dense.forward_bf16_in_bf16_out(x_bf16)),
        }
    }

    /// Workstream B Phase 8 — bf16-in + bf16-residual + bf16-out. Dense arm only.
    /// Caller must supply a bf16 residual.
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_with_residual_bf16_in_bf16_out(
        &self,
        x_bf16: &Tensor,
        residual_bf16: &Tensor,
    ) -> Option<CandleResult<Tensor>> {
        match self {
            MlpBlock::Moe(_) => None,
            MlpBlock::Dense(dense) => Some(
                dense.forward_with_residual_bf16_in_bf16_out(x_bf16, residual_bf16),
            ),
        }
    }

    /// Returns `true` iff the bf16-in MLP fast path can carry actual BW
    /// savings on this arm — i.e. Dense + Affine4 gate_up_proj. False for
    /// MoE and for non-Affine4 Dense variants (those still accept the call,
    /// they just defer through a f32 cast with no BW win).
    #[cfg(feature = "turboquant-gpu")]
    pub fn supports_bf16_in_fast_path(&self) -> bool {
        match self {
            MlpBlock::Moe(_) => false,
            MlpBlock::Dense(dense) => dense.has_affine4_gate_up(),
        }
    }

    /// Workstream B Phase 8 — same Dense+Affine4 condition as
    /// [`Self::supports_bf16_in_fast_path`]. The bf16-in-bf16-out fast path
    /// requires the gate_up matmul to be Affine4 (only variant with a native
    /// bf16-in-bf16-out kernel). MoE returns false (no bf16 path yet).
    #[cfg(feature = "turboquant-gpu")]
    pub fn supports_bf16_in_bf16_out_fast_path(&self) -> bool {
        match self {
            MlpBlock::Moe(_) => false,
            MlpBlock::Dense(dense) => dense.has_affine4_gate_up(),
        }
    }

    /// Dense arm: fully fused post-attention path
    /// (RmsNorm + gate_up + silu*up + down + residual all in 2 dispatches).
    /// Returns `None` for MoE or when shape constraints aren't met.
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_post_attn_fused(
        &self,
        x_raw: &Tensor,
        post_attn_ln_weight: &Tensor,
        rms_eps: f32,
        residual: &Tensor,
    ) -> Option<CandleResult<Tensor>> {
        match self {
            MlpBlock::Moe(_) => None,
            MlpBlock::Dense(dense) => {
                dense.forward_post_attn_fused(x_raw, post_attn_ln_weight, rms_eps, residual)
            }
        }
    }

    /// Cheap precondition check used by the layer to decide whether to skip
    /// the external `post_attention_layernorm.forward(h)` dispatch (Dense path).
    /// Mirrors the conditions in `forward_post_attn_fused`.
    #[cfg(feature = "turboquant-gpu")]
    pub fn supports_dense_post_attn_fusion(&self) -> bool {
        match self {
            MlpBlock::Moe(_) => false,
            MlpBlock::Dense(dense) => dense.supports_post_attn_fusion(),
        }
    }

    /// True iff the MLP block exposes the in-kernel `forward_with_rmsnorm`
    /// fusion path. Currently MXFP4 MoE only — Dense always returns `false`,
    /// causing the layer-level `fusion_active` gate to fall back to external
    /// `post_attention_layernorm` + standard forward.
    pub fn has_mxfp4_backend(&self) -> bool {
        match self {
            MlpBlock::Moe(moe) => moe.has_mxfp4_backend(),
            MlpBlock::Dense(_) => false,
        }
    }

    /// Borrow the underlying `SparseMoeBlock` if this is a MoE variant.
    /// Returns `None` for `Dense`, allowing callers that depend on MoE-only
    /// state (router caches, switch_mlp backend introspection) to skip the
    /// branch cleanly without a panic.
    pub fn as_moe(&self) -> Option<&SparseMoeBlock> {
        match self {
            MlpBlock::Moe(moe) => Some(moe),
            MlpBlock::Dense(_) => None,
        }
    }
}

impl From<SparseMoeBlock> for MlpBlock {
    fn from(moe: SparseMoeBlock) -> Self {
        MlpBlock::Moe(moe)
    }
}

impl From<DenseMlp> for MlpBlock {
    fn from(dense: DenseMlp) -> Self {
        MlpBlock::Dense(dense)
    }
}

/// alloc-reuse experiment (2026-05-03). Lever G concluded WASH due
/// to `Tensor::zeros × 2` per-layer overhead matching the savings from fusing
/// `arg_sort + narrow + gather`. This gate amortises those allocations across
/// decode steps by caching the `[bl, k]` U32/F32 buffers on the `SparseMoeBlock`.
/// Only meaningful in combination with `LUMEN_ENABLE_ROUTING_TOPK_FUSION=1`.
/// Default OFF.
#[inline]
pub(super) fn router_alloc_reuse_enabled() -> bool {
    std::env::var("LUMEN_ROUTER_ALLOC_REUSE")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// The three grouped `[num_experts, out_dim, in_dim]` tensors for the routed SwiGLU experts
/// in the **dense** backend. Kept for CPU fixture testing and non-GPU builds.
pub struct SwitchMlp {
    /// `[num_experts, moe_intermediate_size, hidden_size]`
    pub gate_proj: Tensor,
    /// `[num_experts, moe_intermediate_size, hidden_size]`
    pub up_proj: Tensor,
    /// `[num_experts, hidden_size, moe_intermediate_size]`
    pub down_proj: Tensor,
}

impl SwitchMlp {
    pub fn new(
        gate_proj: Tensor,
        up_proj: Tensor,
        down_proj: Tensor,
        dims: MoeDims,
    ) -> Result<Self, SparseMoeError> {
        let expected_gate = [
            dims.num_experts,
            dims.moe_intermediate_size,
            dims.hidden_size,
        ];
        let expected_down = [
            dims.num_experts,
            dims.hidden_size,
            dims.moe_intermediate_size,
        ];
        check_shape("switch_mlp.gate_proj", &gate_proj, &expected_gate)?;
        check_shape("switch_mlp.up_proj", &up_proj, &expected_gate)?;
        check_shape("switch_mlp.down_proj", &down_proj, &expected_down)?;
        Ok(Self { gate_proj, up_proj, down_proj })
    }
}

/// Dispatch between CPU-resident dense experts (for fixtures / legacy path) and
/// GPU-resident MXFP4 experts (the shipped inference path).
pub enum SwitchMlpBackend {
    Dense(SwitchMlp),
    #[cfg(feature = "turboquant-gpu")]
    Mxfp4(Mxfp4SwitchMlp),
}

impl SwitchMlpBackend {
    /// Compute one expert's full SwiGLU FFN for a single token: `x_t @ W_gate^T` → silu →
    /// elementwise × `x_t @ W_up^T` → `... @ W_down^T`. `x_t` must be shaped `[1, hidden]`.
    fn expert_forward(&self, x_t: &Tensor, expert: usize) -> CandleResult<Tensor> {
        match self {
            Self::Dense(s) => dense_expert_forward(s, x_t, expert),
            #[cfg(feature = "turboquant-gpu")]
            Self::Mxfp4(s) => {
                let gate_out = s.expert_matmul(x_t, expert, ExpertProj::Gate)?;
                let up_out = s.expert_matmul(x_t, expert, ExpertProj::Up)?;
                let hidden_out = (candle_nn::ops::silu(&gate_out)? * up_out)?;
                s.expert_matmul(&hidden_out, expert, ExpertProj::Down)
            }
        }
    }
}

impl From<SwitchMlp> for SwitchMlpBackend {
    fn from(s: SwitchMlp) -> Self {
        Self::Dense(s)
    }
}

#[cfg(feature = "turboquant-gpu")]
impl From<Mxfp4SwitchMlp> for SwitchMlpBackend {
    fn from(s: Mxfp4SwitchMlp) -> Self {
        Self::Mxfp4(s)
    }
}

fn dense_expert_forward(s: &SwitchMlp, x_t: &Tensor, expert: usize) -> CandleResult<Tensor> {
    let gate_w = s
        .gate_proj
        .narrow(0, expert, 1)?
        .squeeze(0)?
        .t()?
        .contiguous()?; // [hidden, inter]
    let up_w = s
        .up_proj
        .narrow(0, expert, 1)?
        .squeeze(0)?
        .t()?
        .contiguous()?; // [hidden, inter]
    let down_w = s
        .down_proj
        .narrow(0, expert, 1)?
        .squeeze(0)?
        .t()?
        .contiguous()?; // [inter, hidden]

    let gate_out = x_t.matmul(&gate_w)?; // [1, inter]
    let up_out = x_t.matmul(&up_w)?; // [1, inter]
    let hidden_out = (candle_nn::ops::silu(&gate_out)? * up_out)?;
    hidden_out.matmul(&down_w)
}

fn check_shape(name: &'static str, t: &Tensor, expected: &[usize]) -> Result<(), SparseMoeError> {
    if t.dims() != expected {
        return Err(SparseMoeError::WeightShape {
            name,
            expected: expected.to_vec(),
            found: t.dims().to_vec(),
        });
    }
    Ok(())
}

/// Rust port of `mlx_lm.models.qwen3_next.Qwen3NextSparseMoeBlock`. Constructed from already
/// materialized Candle layers so the loader (Stage 2-f) can dequantize once and hand every
/// sub-weight in. Tests stitch synthetic `Linear`/tensor pieces in the same way.
pub struct SparseMoeBlock {
    gate: ProjLinear,
    shared_expert_gate: ProjLinear,
    shared_expert: SharedExpert,
    switch_mlp: SwitchMlpBackend,
    runtime: SparseMoeRuntime,
    /// Cached `[bl, k]` U32/F32 buffers for the fused topk_partial_select path
    /// when `LUMEN_ROUTER_ALLOC_REUSE=1`. Stores `(bl, inds_buf, vals_buf)`;
    /// re-allocates when `bl` changes. Lever G's `Tensor::zeros × 2` per-layer
    /// allocator pressure was the WASH ceiling (~0.6-1.8 ms/step instrumented);
    /// caching amortises that across decode steps. See lever_g_routing_topk_concluded.md.
    cached_router_buffers: Mutex<Option<(usize, Tensor, Tensor)>>,
}

impl SparseMoeBlock {
    pub fn new(
        runtime: SparseMoeRuntime,
        gate: ProjLinear,
        shared_expert_gate: ProjLinear,
        shared_expert: SharedExpert,
        switch_mlp: SwitchMlpBackend,
    ) -> Self {
        Self {
            gate,
            shared_expert_gate,
            shared_expert,
            switch_mlp,
            runtime,
            cached_router_buffers: Mutex::new(None),
        }
    }

    pub fn dims(&self) -> MoeDims {
        self.runtime.dims
    }

    pub fn top_k(&self) -> usize {
        self.runtime.top_k
    }

    /// True iff this MoE block uses the GPU MXFP4 switch_mlp backend (i.e. the
    /// production path). False for Dense / CPU fixture tests. Used by
    /// `DecoderLayer.forward_with_tq` to gate `forward_with_rmsnorm` —
    /// the rmsnorm-fused path requires the MXFP4 backend.
    pub fn has_mxfp4_backend(&self) -> bool {
        #[cfg(feature = "turboquant-gpu")]
        {
            matches!(self.switch_mlp, SwitchMlpBackend::Mxfp4(_))
        }
        #[cfg(not(feature = "turboquant-gpu"))]
        {
            false
        }
    }

    /// Forward pass. Matches `Qwen3NextSparseMoeBlock.__call__` (mlx-lm 0.31.3 / qwen3_5).
    ///
    /// `x`: shape `[B, L, hidden_size]` — the `post_attention_layernorm` output.
    /// Returns: `[B, L, hidden_size]`.
    ///
    /// Algorithm (mirrors MLX step-for-step):
    ///   1. `probs  = softmax(gate(x), axis=-1)`
    ///   2. `inds   = argpartition(probs, -k, axis=-1)[..., -k:]`  (MLX non-sorted top-k)
    ///   3. `scores = gather_along(probs, inds)`; if `norm_topk_prob`, divide by row sum
    ///   4. For each token, sum each selected expert's SwiGLU output weighted by its score
    ///   5. Add a scalar-gated shared-expert SwiGLU output
    ///
    /// Candle lacks `argpartition`, so step 2 uses a full descending arg-sort. The set of
    /// top-k indices is identical; only their ordering (and therefore partial-sum rounding)
    /// differs. Empirically this lands inside the fixture's 1e-2 bf16↔f32 bound.
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (batch, seq_len, hidden) = x.dims3()?;
        if hidden != self.runtime.dims.hidden_size {
            candle_core::bail!(
                "SparseMoeBlock input hidden {hidden} does not match config {}",
                self.runtime.dims.hidden_size
            );
        }
        let bl = batch * seq_len;
        let k = self.runtime.top_k;
        let x_flat = x.reshape((bl, hidden))?;

        // Two-tier timing flags so callers can isolate sync overhead:
        //   - `LUMEN_MOE_TIMING=1`     → 5 top-level marks (routing/host_xfer/routed_loop/
        //                                  shared_expert/combine). Cheap; ~5 syncs/layer.
        //   - `LUMEN_MOE_SUB_TIMING=1` → adds per-sub-op marks inside routing,
        //                                  routed_loop, and shared_expert. Heavy;
        //                                  ~25 syncs/layer.
        let moe_timing = std::env::var("LUMEN_MOE_TIMING")
            .map(|v| v == "1")
            .unwrap_or(false);
        let moe_sub_timing = std::env::var("LUMEN_MOE_SUB_TIMING")
            .map(|v| v == "1")
            .unwrap_or(false);
        let device = x.device().clone();
        let mut marks: Vec<(&'static str, std::time::Instant)> = Vec::new();
        let mut sub_marks: Vec<(&'static str, std::time::Instant)> = Vec::new();
        let sync_mark = |marks: &mut Vec<(&'static str, std::time::Instant)>, label: &'static str| {
            if moe_timing {
                let _ = device.synchronize();
                marks.push((label, std::time::Instant::now()));
            }
        };
        let sync_sub = |marks: &mut Vec<(&'static str, std::time::Instant)>, label: &'static str| {
            if moe_sub_timing {
                let _ = device.synchronize();
                marks.push((label, std::time::Instant::now()));
            }
        };
        sync_mark(&mut marks, "start");
        sync_sub(&mut sub_marks, "r_start");

        // ── Routing ────────────────────────────────────────────────────────
        // opt-in small-out kernel for the routing gate. Microbench
        // shows ~3× speedup on the r_gate decode shape (out=256, in=2048, b=1)
        // because v3's `n_groups_x = ceil(256/8) = 32 TGs` under-occupies the
        // M3 Max GPU. Off by default until 35B A/B confirms the win — same
        // protocol M.1 should have used.
        let enable_small_out_gate = std::env::var("LUMEN_ENABLE_SMALL_OUT_GATE")
            .map(|v| v == "1")
            .unwrap_or(false);
        // bf16 routing gate when LUMEN_BF16_OUT=1, cast back so
        // softmax_last_dim sees f32 (its kernel expects f32 in this build).
        let bf16_out = bf16_out_enabled();
        let logits = if enable_small_out_gate {
            if bf16_out {
                let y_bf16 = self.gate.forward_small_out_bf16_out(&x_flat)?;
                y_bf16.to_dtype(candle_core::DType::F32)? // [BL, E]
            } else {
                self.gate.forward_small_out(&x_flat)? // [BL, E]
            }
        } else if bf16_out {
            let y_bf16 = self.gate.forward_bf16_out(&x_flat)?;
            y_bf16.to_dtype(candle_core::DType::F32)? // [BL, E]
        } else {
            self.gate.forward(&x_flat)? // [BL, E]
        };
        sync_sub(&mut sub_marks, "r_gate");

        // LUMEN_ROUTER_FUSED=1 collapses the
        // entire 6-dispatch routing chain (`softmax → arg_sort → narrow →
        // gather → sum_keepdim → broadcast_div`) into a single Metal dispatch
        // via `router_softmax_topk_renorm_f32`. Saves 5 dispatches/layer × 40
        // MoE layers = 200 dispatches/decode step. Default OFF until 35B A/B
        // confirms σ ≥ +2 + bit-identical text. Mutually exclusive with the
        // partial Lever G path below (full fusion takes precedence).
        //
        // Drift hazard: cooperative SG-reduce ordering may differ ≤1 ULP from
        // Candle softmax kernel — top-K argpartition stable when logit gaps
        // are non-trivial; weights cos ≥ 0.9999 expected; end-to-end bit-id-
        // entical decode is the gold gate.
        let router_fused_active = std::env::var("LUMEN_ROUTER_FUSED")
            .map(|v| v == "1")
            .unwrap_or(false);
        let logits_dims_for_fusion = logits.dims();
        let router_fused_path = router_fused_active
            && matches!(self.switch_mlp, SwitchMlpBackend::Mxfp4(_))
            && logits.dtype() == candle_core::DType::F32
            && matches!(logits.device(), candle_core::Device::Metal(_))
            && logits_dims_for_fusion.len() == 2
            && logits_dims_for_fusion[1] <= 256
            && k <= 32
            && self.runtime.norm_topk_prob;

        let (inds, scores) = if router_fused_path
            && let SwitchMlpBackend::Mxfp4(mxfp4_ref) = &self.switch_mlp
        {
            // Full fusion: softmax + topk + renorm in one dispatch.
            let (inds_out, vals_out) = if router_alloc_reuse_enabled() {
                let mut guard = self
                    .cached_router_buffers
                    .lock()
                    .expect("router cache poisoned");
                match guard.as_ref() {
                    Some((cached_bl, inds, vals)) if *cached_bl == bl => {
                        (inds.clone(), vals.clone())
                    }
                    _ => {
                        let inds_out = Tensor::zeros(
                            vec![bl, k],
                            candle_core::DType::U32,
                            logits.device(),
                        )?;
                        let vals_out = Tensor::zeros(
                            vec![bl, k],
                            candle_core::DType::F32,
                            logits.device(),
                        )?;
                        *guard = Some((bl, inds_out.clone(), vals_out.clone()));
                        (inds_out, vals_out)
                    }
                }
            } else {
                let inds_out = Tensor::zeros(
                    vec![bl, k],
                    candle_core::DType::U32,
                    logits.device(),
                )?;
                let vals_out = Tensor::zeros(
                    vec![bl, k],
                    candle_core::DType::F32,
                    logits.device(),
                )?;
                (inds_out, vals_out)
            };
            mxfp4_ref.router_softmax_topk_renorm_f32_candle_queue_into(
                &logits, &inds_out, &vals_out,
            )?;
            sync_sub(&mut sub_marks, "r_router_fused");
            sync_mark(&mut marks, "routing");
            (inds_out, vals_out) // already renormalized
        } else {
            let probs = candle_nn::ops::softmax_last_dim(&logits)?;
            sync_sub(&mut sub_marks, "r_softmax");

            // Lever G (2026-04-27): when the env opt-in is set AND the Mxfp4
            // backend is active AND probs is Metal F32 AND num_experts ≤ 256,
            // replace the chain `arg_sort_last_dim → narrow → contiguous → gather`
            // with one fused dispatch that produces top-k inds + vals directly.
            // Default OFF; flip ON after 35B A/B (σ ≥ 2). Falls back to the
            // existing chain otherwise. Bit-exact equivalent to the chain (stable
            // descending sort + lowest-index tie-break).
            let enable_routing_topk_fusion =
                std::env::var("LUMEN_ENABLE_ROUTING_TOPK_FUSION")
                    .map(|v| v == "1")
                    .unwrap_or(false);
            let probs_dims = probs.dims();
            let topk_fused_path = enable_routing_topk_fusion
                && matches!(self.switch_mlp, SwitchMlpBackend::Mxfp4(_))
                && probs.dtype() == candle_core::DType::F32
                && matches!(probs.device(), candle_core::Device::Metal(_))
                && probs_dims.len() == 2
                && probs_dims[1] <= 256;

            let (inds, scores) = if topk_fused_path
                && let SwitchMlpBackend::Mxfp4(mxfp4_ref) = &self.switch_mlp
            {
                // when `LUMEN_ROUTER_ALLOC_REUSE=1`, reuse a
                // cached `[bl, k]` buffer pair across decode steps. Lever G's WASH
                // (σ −1.49) was attributed to the per-call `Tensor::zeros × 2` overhead
                // matching the savings from fusing arg_sort + narrow + gather; this
                // gate tests whether amortising the alloc recovers the missing win.
                let (inds_out, vals_out) = if router_alloc_reuse_enabled() {
                    let mut guard = self
                        .cached_router_buffers
                        .lock()
                        .expect("router cache poisoned");
                    match guard.as_ref() {
                        Some((cached_bl, inds, vals)) if *cached_bl == bl => {
                            (inds.clone(), vals.clone())
                        }
                        _ => {
                            let inds_out = Tensor::zeros(
                                vec![bl, k],
                                candle_core::DType::U32,
                                probs.device(),
                            )?;
                            let vals_out = Tensor::zeros(
                                vec![bl, k],
                                candle_core::DType::F32,
                                probs.device(),
                            )?;
                            *guard = Some((bl, inds_out.clone(), vals_out.clone()));
                            (inds_out, vals_out)
                        }
                    }
                } else {
                    let inds_out = Tensor::zeros(
                        vec![bl, k],
                        candle_core::DType::U32,
                        probs.device(),
                    )?;
                    let vals_out = Tensor::zeros(
                        vec![bl, k],
                        candle_core::DType::F32,
                        probs.device(),
                    )?;
                    (inds_out, vals_out)
                };
                mxfp4_ref.topk_partial_select_candle_queue_into(
                    &probs, &inds_out, &vals_out,
                )?;
                sync_sub(&mut sub_marks, "r_topk_fused");
                (inds_out, vals_out)
            } else {
                // Descending arg-sort over experts, keep the first `k` columns = top-k by score.
                let sorted_idx = probs.arg_sort_last_dim(false)?; // [BL, E], u32
                sync_sub(&mut sub_marks, "r_argsort");
                let inds = sorted_idx.narrow(D::Minus1, 0, k)?.contiguous()?; // [BL, k]
                let scores = probs.gather(&inds, D::Minus1)?; // [BL, k]
                sync_sub(&mut sub_marks, "r_narrow_gather");
                (inds, scores)
            };
            let scores = if self.runtime.norm_topk_prob {
                let denom = scores.sum_keepdim(D::Minus1)?; // [BL, 1]
                scores.broadcast_div(&denom)?
            } else {
                scores
            };
            sync_sub(&mut sub_marks, "r_norm");
            sync_mark(&mut marks, "routing");
            (inds, scores)
        };

        // ── Routed experts ─────────────────────────────────────────────────
        // Pull the per-(token, slot) picks into host memory. `bl * k` is small (e.g. 32 for
        // the fixture, at most ~thousands in normal prefill), so the cost of leaving the
        // accelerator is negligible next to the per-expert GEMMs that follow.
        //
        // the previous code also pulled `scores` to host
        // and then re-uploaded a per-token slice via `Tensor::from_vec` for the weighted
        // sum. That round-trip is one Candle-queue sync per layer (40 syncs/forward at
        // ~218 μs each per M.2-A finding). Now we keep `scores` on GPU and slice it
        // directly inside the t loop. Indices are still pulled (Step 2 lands separately).
        //
        // Slow / diagnostic / legacy paths still need a host-side `scores_host` slice;
        // we materialize it on demand below to keep the GPU-resident path's host_xfer
        // cost at zero on the production hot path.
        let enable_gpu_scores = std::env::var("LUMEN_DISABLE_GPU_SCORES")
            .map(|v| v != "1")
            .unwrap_or(true);
        // keep `inds` on the GPU as well. The grouped
        // hot path now binds `inds.narrow(0, t, 1)?` directly into the kernel via
        // `moe_*_with_indices_buffer`, skipping the `inds.flatten_all().to_vec1::<u32>()`
        // sync. Slow / fallback paths still need the host slice (lazy materialized).
        let enable_gpu_inds = std::env::var("LUMEN_DISABLE_GPU_INDS")
            .map(|v| v != "1")
            .unwrap_or(true);
        // submit MoE kernels through Candle's command
        // queue so they share command buffers with surrounding Candle ops. Same-queue
        // ordering eliminates the `wait_until_completed()` round-trip per kernel call
        // (40 layers × 2 = 80 syncs/forward in production decode). Default ON, opt-out
        // via `LUMEN_DISABLE_CANDLE_QUEUE=1`.
        //
        // **bl==1 gate**: only enable for decode (single-token forward). For prefill
        // (bl > 1), the t loop's repeated `Tensor::zeros` + no-wait pattern thrashes
        // Candle's buffer allocator (buffers stay in_flight, allocator can't reuse) and
        // exhausts the command pool (`COMPUTE_PER_BUFFER=50` × `POOL_SIZE=5`). Empirically,
        // bl=23 prefill regressed 367ms→4023ms (-11x). bl=1 decode wins 157→138ms (+12%).
        // `LUMEN_FORCE_CANDLE_QUEUE_ALL_BL=1` disables the bl==1 gate (Option D probe:
        // measure ungated behavior under tuned `CANDLE_METAL_COMPUTE_PER_BUFFER` /
        // `COMMAND_POOL_SIZE` to see if prefill regression dissolves).
        let force_all_bl = std::env::var("LUMEN_FORCE_CANDLE_QUEUE_ALL_BL")
            .map(|v| v == "1")
            .unwrap_or(false);
        let enable_candle_queue = std::env::var("LUMEN_DISABLE_CANDLE_QUEUE")
            .map(|v| v != "1")
            .unwrap_or(true)
            && (force_all_bl || bl == 1);

        // Grouped expert dispatch is the default now (2026-04-25). With
        // `LUMEN_MOE_BATCH_SINGLE=1` semantics baked in (safe fanout), decode gains ~25%
        // over the sequential per-expert loop, matching post-conv1d-fix profiling.
        // `LUMEN_MOE_LEGACY=1` restores the prior sequential path (rollback lever).
        // `LUMEN_MOE_BATCHED=0` is also accepted as an alias.
        let legacy_sequential = std::env::var("LUMEN_MOE_LEGACY")
            .map(|v| v == "1")
            .unwrap_or(false)
            || std::env::var("LUMEN_MOE_BATCHED")
                .map(|v| v == "0")
                .unwrap_or(false);
        let batched_enabled = !legacy_sequential;
        let mxfp4_backend = matches!(self.switch_mlp, SwitchMlpBackend::Mxfp4(_));
        // Option I grouped-kernel MoE path: one Metal dispatch handles all k experts via
        // indirect indexing into unified per-proj weight buffers. Default on; rollback
        // with `LUMEN_MOE_GROUPED=0` falls back to the multi_cmdbuf fanout from Option H.
        let grouped_kernel = std::env::var("LUMEN_MOE_GROUPED")
            .map(|v| v != "0")
            .unwrap_or(true);
        // Hot path is `batched_enabled && mxfp4_backend`; everything else (CPU
        // fallback, legacy sequential, or moe_timing diagnostic) still needs the
        // host-side scores slice for per-expert weighting.
        let production_grouped = batched_enabled && mxfp4_backend && !moe_timing;
        let need_scores_host = !enable_gpu_scores || !production_grouped;
        // Indices host slice is required for: grouped_kernel=false fallback (uses
        // `gate_and_up_group_big`/`expert_matmul_group_multi_x_big` with `&[usize]`),
        // moe_timing diagnostic path, legacy sequential path, and CPU backend.
        let need_inds_host = !enable_gpu_inds || !production_grouped || !grouped_kernel;

        let inds_host: Option<Vec<usize>> = if need_inds_host {
            Some(
                inds.flatten_all()?
                    .to_vec1::<u32>()?
                    .into_iter()
                    .map(|v| v as usize)
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
        let scores_host: Option<Vec<f32>> = if need_scores_host {
            Some(scores.flatten_all()?.to_vec1::<f32>()?)
        } else {
            None
        };
        sync_mark(&mut marks, "host_xfer");
        let y_routed: Tensor = if batched_enabled
            && let SwitchMlpBackend::Mxfp4(mxfp4) = &self.switch_mlp
        {
            let mut y_rows: Vec<Tensor> = Vec::with_capacity(bl);
            let device_scores = x_flat.device();
            // Sub-stage accumulators for the grouped-kernel hot path. Only populated when
            // LUMEN_MOE_TIMING=1; each marker syncs first so reported ms reflect GPU completion.
            let mut gate_up_ms = 0.0f64;
            let mut silu_mul_ms = 0.0f64;
            let mut down_ms = 0.0f64;
            let mut wsum_ms = 0.0f64;
            // when `enable_gpu_inds`, the grouped path uses
            // `inds.narrow(0, t, 1)?` (metadata-only) → `_with_indices_buffer` variants
            // and never builds the per-token `Vec<usize>`. Fallback path still needs the
            // host slice (`inds_host` is `Some` when `need_inds_host` was true).
            let use_gpu_inds = grouped_kernel && enable_gpu_inds && inds_host.is_none();

            // pre-allocate the per-step output pools
            // outside the t loop so multi-batch (prefill) iterations don't churn the
            // Candle buffer allocator. For decode (bl == 1) this is identical to the
            // previous per-iter Tensor::zeros (single allocation either way). For
            // prefill (bl > 1) we collapse `2 * bl` allocations into 2.
            //
            // Only used on the candle-queue path: own-queue variants still allocate
            // internally because they wait per call (their commit/wait gives buffer reuse).
            // Default ON, opt-out via `LUMEN_DISABLE_MOE_Y_POOL=1`.
            let enable_y_pool = std::env::var("LUMEN_DISABLE_MOE_Y_POOL")
                .map(|v| v != "1")
                .unwrap_or(true);
            // Lever A (2026-04-27): routed-grouped fused gate+up+silu*up. LANDED.
            // 35B A/B (n=421 decode steps each): 48.603ms → 47.461ms (σ=+5.91),
            // 15/15 token sequences bit-identical.
            // Default ON, opt-out via `LUMEN_DISABLE_MOE_GATE_UP_SILU_MUL_FUSION=1`.
            // When ON, the gate_up pool collapses to `[bl*k, moe_inter]`
            // (half the size) since the kernel writes silu(gate)*up directly.
            let enable_moe_swiglu_fusion =
                std::env::var("LUMEN_DISABLE_MOE_GATE_UP_SILU_MUL_FUSION")
                    .map(|v| v != "1")
                    .unwrap_or(true);
            // Lever D (2026-04-27): bf16 chain — gate_up writes bf16, down
            // reads bf16, no cast back. Half the device-memory bytes for the
            // gate_up_pool. Only active on the candle-queue + GPU-inds + pool
            // + Lever-A fused path; falls back to F32 chain otherwise.
            // Default OFF; flip after 35B A/B (σ ≥ 2).
            let enable_bf16_chain =
                std::env::var("LUMEN_ENABLE_MOE_BF16_CHAIN")
                    .map(|v| v == "1")
                    .unwrap_or(false);
            let bf16_chain_active = enable_bf16_chain
                && enable_moe_swiglu_fusion
                && use_gpu_inds
                && enable_candle_queue
                && enable_y_pool
                && matches!(self.switch_mlp, SwitchMlpBackend::Mxfp4(_));
            let gate_up_pool_dtype = if bf16_chain_active {
                candle_core::DType::BF16
            } else {
                candle_core::DType::F32
            };
            let (gate_up_pool, down_pool) = if use_gpu_inds && enable_candle_queue && enable_y_pool
                && let SwitchMlpBackend::Mxfp4(mxfp4) = &self.switch_mlp
            {
                let g_out = if enable_moe_swiglu_fusion {
                    mxfp4.moe_inter
                } else {
                    2 * mxfp4.moe_inter
                };
                let g_pool = Tensor::zeros(
                    vec![bl * k, g_out],
                    gate_up_pool_dtype,
                    x_flat.device(),
                )?;
                let d_pool = Tensor::zeros(
                    vec![bl * k, mxfp4.hidden],
                    candle_core::DType::F32,
                    x_flat.device(),
                )?;
                (Some(g_pool), Some(d_pool))
            } else {
                (None, None)
            };

            for t in 0..bl {
                let x_t = x_flat.narrow(0, t, 1)?; // [1, hidden]
                // Build `experts: Vec<usize>` only on the host-fallback paths.
                let experts: Vec<usize> = if let Some(host) = inds_host.as_ref() {
                    (0..k).map(|j| host[t * k + j]).collect()
                } else {
                    Vec::new()
                };
                let inds_slice = if use_gpu_inds {
                    Some(inds.narrow(0, t, 1)?.reshape((k,))?)
                } else {
                    None
                };

                if moe_sub_timing {
                    let _ = device.synchronize();
                }
                let t0 = std::time::Instant::now();

                // Gate+Up: grouped kernel dispatches all k experts in one launch (Option I)
                // or falls back to multi_cmdbuf fanout (Option H) when LUMEN_MOE_GROUPED=0.
                //
                // Lever A (2026-04-27): when `enable_moe_swiglu_fusion` is set AND we're on
                // the candle-queue + pool hot path, the fused kernel writes `silu(gate)*up`
                // directly into `y_slice` (shape `[k, moe_inter]`, half the non-fused size).
                // `gate_up_pair` is `None` in that case and we skip the host-side silu*mul.
                let mut hiddens_big_fused: Option<Tensor> = None;
                let gate_up_pair: Option<(Tensor, Tensor)> = if let Some(inds_t) =
                    inds_slice.as_ref()
                {
                    if enable_candle_queue {
                        if let Some(pool) = gate_up_pool.as_ref() {
                            // Option E: caller-provided y, no per-iter alloc.
                            let y_slice = pool.narrow(0, t * k, k)?;
                            if enable_moe_swiglu_fusion {
                                if bf16_chain_active {
                                    // Lever D: bf16-output variant of fused gate_up.
                                    mxfp4
                                        .moe_gate_up_silu_mul_bf16out_with_indices_buffer_candle_queue_into(
                                            &x_t, inds_t, k, &y_slice,
                                        )?;
                                } else {
                                    mxfp4
                                        .moe_gate_up_silu_mul_with_indices_buffer_candle_queue_into(
                                            &x_t, inds_t, k, &y_slice,
                                        )?;
                                }
                                hiddens_big_fused = Some(y_slice.contiguous()?);
                                None
                            } else {
                                mxfp4.moe_gate_up_with_indices_buffer_candle_queue_into(
                                    &x_t, inds_t, k, &y_slice,
                                )?;
                                let moe_inter = mxfp4.moe_inter;
                                let gate_big =
                                    y_slice.narrow(1, 0, moe_inter)?.contiguous()?;
                                let up_big = y_slice
                                    .narrow(1, moe_inter, moe_inter)?
                                    .contiguous()?;
                                Some((gate_big, up_big))
                            }
                        } else {
                            Some(
                                mxfp4
                                    .moe_gate_up_with_indices_buffer_candle_queue(&x_t, inds_t, k)?,
                            )
                        }
                    } else {
                        Some(mxfp4.moe_gate_up_with_indices_buffer(&x_t, inds_t, k)?)
                    }
                } else if grouped_kernel {
                    Some(mxfp4.moe_gate_up(&x_t, &experts)?)
                } else {
                    Some(mxfp4.gate_and_up_group_big(&x_t, &experts)?)
                };
                if moe_sub_timing {
                    let _ = device.synchronize();
                }
                let t1 = std::time::Instant::now();

                let hiddens_big = if let Some(h) = hiddens_big_fused {
                    h
                } else {
                    let (gate_big, up_big) = gate_up_pair.expect(
                        "gate_up_pair is Some when fusion is OFF",
                    );
                    let f32_h = (candle_nn::ops::silu(&gate_big)? * up_big)?;
                    if bf16_chain_active {
                        // Defensive: shouldn't actually hit this branch since
                        // bf16_chain_active requires fusion ON, but cast just
                        // in case the contract is widened in the future.
                        f32_h.to_dtype(candle_core::DType::BF16)?
                    } else {
                        f32_h
                    }
                };
                if moe_sub_timing {
                    let _ = device.synchronize();
                }
                let t2 = std::time::Instant::now();

                // Lever C (2026-04-27): fused down + wsum dispatch. When an
                // env opt-in is set AND the candle-queue + GPU-resident inds
                // path is active AND hiddens is Metal F32, skip the separate
                // `downs_big` allocation + `moe_wsum_*` step and go directly
                // from `hiddens_big [k, moe_inter]` to `row [1, hidden]`.
                //
                // Two variants gated by separate env vars (mutually exclusive
                // — if both set, atomic wins):
                //   - `LUMEN_ENABLE_MOE_MATMUL_WSUM_ATOMIC=1`: grid-parallel
                //     `mxfp4_matmul_moe_wsum_atomic_f32_v3` (k slots in grid.z,
                //     atomic_fetch_add per output element). Caller pre-zeroes.
                //   - `LUMEN_ENABLE_MOE_MATMUL_WSUM_FUSION=1`: serial-fold
                //     `mxfp4_matmul_moe_wsum_f32_v3` (k folded into TG-internal
                //     loop; NEGATIVE σ=−10.70, see
                //     `lever_c_moe_matmul_wsum_concluded.md`).
                //
                // Both default OFF; A/B benchmarks decide which (if any) flips
                // ON. Falls back to the existing 2-stage chain when off.
                let enable_matmul_wsum_atomic =
                    std::env::var("LUMEN_ENABLE_MOE_MATMUL_WSUM_ATOMIC")
                        .map(|v| v == "1")
                        .unwrap_or(false);
                let enable_matmul_wsum_fusion =
                    std::env::var("LUMEN_ENABLE_MOE_MATMUL_WSUM_FUSION")
                        .map(|v| v == "1")
                        .unwrap_or(false);
                let lever_c_path = (enable_matmul_wsum_atomic || enable_matmul_wsum_fusion)
                    && inds_slice.is_some()
                    && enable_candle_queue
                    && hiddens_big.dtype() == candle_core::DType::F32
                    && matches!(hiddens_big.device(), candle_core::Device::Metal(_));

                let row = if lever_c_path {
                    let inds_t = inds_slice.as_ref().expect("lever_c_path implies inds_slice");
                    let hidden_dim = mxfp4.hidden;
                    let w_kx1 = if let Some(host) = scores_host.as_ref() {
                        let weights_vec: Vec<f32> = host[t * k..(t + 1) * k].to_vec();
                        let w = Tensor::from_vec(weights_vec, (k,), device_scores)?;
                        w.reshape((k, 1))?
                    } else {
                        scores.narrow(0, t, 1)?.reshape((k, 1))?
                    };
                    let w_flat = w_kx1.flatten_all()?;
                    let out = Tensor::zeros(
                        vec![1, hidden_dim],
                        candle_core::DType::F32,
                        hiddens_big.device(),
                    )?;
                    if enable_matmul_wsum_atomic {
                        mxfp4.moe_matmul_wsum_atomic_with_indices_buffer_candle_queue_into(
                            &hiddens_big,
                            inds_t,
                            k,
                            &w_flat,
                            &out,
                        )?;
                    } else {
                        mxfp4.moe_matmul_wsum_with_indices_buffer_candle_queue_into(
                            &hiddens_big,
                            inds_t,
                            k,
                            &w_flat,
                            &out,
                        )?;
                    }
                    if moe_sub_timing {
                        let _ = device.synchronize();
                        let t4 = std::time::Instant::now();
                        gate_up_ms += t1.duration_since(t0).as_secs_f64() * 1000.0;
                        silu_mul_ms += t2.duration_since(t1).as_secs_f64() * 1000.0;
                        // Lever C fuses down + wsum into one dispatch — attribute
                        // the combined kernel time to `down_ms` and leave
                        // `wsum_ms = 0` so the breakdown reflects the new shape.
                        down_ms += t4.duration_since(t2).as_secs_f64() * 1000.0;
                    }
                    out
                } else {
                    // Existing 2-stage path: down → wsum.
                    //
                    // Down: grouped kernel consumes the [k, moe_inter] hiddens tensor directly
                    // (no per-expert narrow/reshape/Vec construction). Fallback path builds
                    // k views and dispatches via multi_x multi_cmdbuf.
                    let downs_big = if let Some(inds_t) = inds_slice.as_ref() {
                        if enable_candle_queue {
                            if let Some(pool) = down_pool.as_ref() {
                                let y_slice = pool.narrow(0, t * k, k)?;
                                if bf16_chain_active
                                    && hiddens_big.dtype() == candle_core::DType::BF16
                                {
                                    // Lever D: bf16-input variant of MoE down.
                                    mxfp4.moe_down_bf16in_with_indices_buffer_candle_queue_into(
                                        &hiddens_big,
                                        inds_t,
                                        k,
                                        &y_slice,
                                    )?;
                                } else {
                                    mxfp4.moe_down_with_indices_buffer_candle_queue_into(
                                        &hiddens_big,
                                        inds_t,
                                        k,
                                        &y_slice,
                                    )?;
                                }
                                y_slice.contiguous()?
                            } else {
                                mxfp4.moe_down_with_indices_buffer_candle_queue(
                                    &hiddens_big,
                                    inds_t,
                                    k,
                                )?
                            }
                        } else {
                            mxfp4.moe_down_with_indices_buffer(&hiddens_big, inds_t, k)?
                        }
                    } else if grouped_kernel {
                        mxfp4.moe_down(&hiddens_big, &experts)?
                    } else {
                        let moe_inter = hiddens_big.dims()[1];
                        let mut hidden_views: Vec<Tensor> = Vec::with_capacity(k);
                        for j in 0..k {
                            hidden_views.push(hiddens_big.narrow(0, j, 1)?.reshape((1, moe_inter))?);
                        }
                        mxfp4.expert_matmul_group_multi_x_big(
                            &hidden_views,
                            &experts,
                            lumen_metal::mxfp4_linear::ExpertProj::Down,
                        )?
                    };
                    if moe_sub_timing {
                        let _ = device.synchronize();
                    }
                    let t3 = std::time::Instant::now();

                    // Weighted sum: downs_big: [k, hidden] × scores: [k] → [1, hidden].
                    //
                    // keep the per-token scores slice on GPU
                    // when `enable_gpu_scores` (default). Slicing `scores` (already a GPU
                    // tensor of shape `[BL, k]`) is metadata-only; the legacy host →
                    // `Tensor::from_vec` round-trip is bypassed entirely. Falls back to
                    // the host vec on opt-out, retaining a known-good rollback path.
                    let hidden_dim = downs_big.dims()[1];
                    let w_kx1 = if let Some(host) = scores_host.as_ref() {
                        let weights_vec: Vec<f32> = host[t * k..(t + 1) * k].to_vec();
                        let w = Tensor::from_vec(weights_vec, (k,), device_scores)?;
                        w.reshape((k, 1))?
                    } else {
                        // GPU-resident: scores is `[BL, k]`. Take row t and reshape to `[k, 1]`.
                        scores.narrow(0, t, 1)?.reshape((k, 1))?
                    };
                    // Lever B (2026-04-27): fused wsum kernel writes
                    // `out[r] = sum_e w[e] * downs[e, r]` directly, replacing
                    // the broadcast_mul + sum_keepdim chain (2 Candle kernels +
                    // intermediate [k, hidden] device tensor). LANDED.
                    // 35B A/B (n=842 pooled across orderings): 47.768→47.414ms
                    // (σ=+4.09), 60/60 token sequences bit-identical.
                    // Default ON, opt-out via `LUMEN_DISABLE_MOE_WSUM_FUSION=1`.
                    // Falls back to the Candle chain when off, or when downs_big
                    // is not Metal F32 (CPU fallback paths).
                    let enable_wsum_fusion =
                        std::env::var("LUMEN_DISABLE_MOE_WSUM_FUSION")
                            .map(|v| v != "1")
                            .unwrap_or(true);
                    let row = if enable_wsum_fusion
                        && downs_big.dtype() == candle_core::DType::F32
                        && matches!(downs_big.device(), candle_core::Device::Metal(_))
                    {
                        let out = Tensor::zeros(
                            vec![1, hidden_dim],
                            candle_core::DType::F32,
                            downs_big.device(),
                        )?;
                        let w_flat = w_kx1.flatten_all()?;
                        mxfp4.moe_wsum_candle_queue_into(&downs_big, &w_flat, &out)?;
                        out
                    } else {
                        let weighted = downs_big.broadcast_mul(&w_kx1)?;
                        weighted.sum_keepdim(0)?.reshape((1, hidden_dim))?
                    };
                    if moe_sub_timing {
                        let _ = device.synchronize();
                        let t4 = std::time::Instant::now();
                        gate_up_ms += t1.duration_since(t0).as_secs_f64() * 1000.0;
                        silu_mul_ms += t2.duration_since(t1).as_secs_f64() * 1000.0;
                        down_ms += t3.duration_since(t2).as_secs_f64() * 1000.0;
                        wsum_ms += t4.duration_since(t3).as_secs_f64() * 1000.0;
                    }
                    row
                };
                y_rows.push(row);
            }
            if moe_sub_timing {
                eprintln!(
                    "      moe-grouped: gate_up={gate_up_ms:.2} silu_mul={silu_mul_ms:.2} down={down_ms:.2} wsum={wsum_ms:.2} ({} tokens, k={})",
                    bl, k
                );
            }
            Tensor::cat(&y_rows, 0)? // [BL, hidden]
        } else if moe_timing {
            // Diagnostic path: inline expert_forward so we can aggregate per-sub-op ms across
            // the 8 × BL iterations. Only used when LUMEN_MOE_TIMING=1; costs extra syncs.
            // `scores_host` is guaranteed populated when `moe_timing` is set (see
            // `need_scores_host` above).
            let scores_host_ref = scores_host
                .as_ref()
                .expect("scores_host populated when moe_timing");
            let inds_host_ref = inds_host
                .as_ref()
                .expect("inds_host populated when moe_timing");
            #[cfg(feature = "turboquant-gpu")]
            use lumen_metal::mxfp4_linear::ExpertProj;
            let mut gate_ms = 0.0f64;
            let mut up_ms = 0.0f64;
            let mut silu_mul_ms = 0.0f64;
            let mut down_ms = 0.0f64;
            let mut acc_ms = 0.0f64;
            let mut y_rows: Vec<Tensor> = Vec::with_capacity(bl);
            for t in 0..bl {
                let x_t = x_flat.narrow(0, t, 1)?;
                let mut acc: Option<Tensor> = None;
                for j in 0..k {
                    let expert = inds_host_ref[t * k + j];
                    let w = scores_host_ref[t * k + j] as f64;
                    match &self.switch_mlp {
                        #[cfg(feature = "turboquant-gpu")]
                        SwitchMlpBackend::Mxfp4(mx) => {
                            let _ = device.synchronize();
                            let t0 = std::time::Instant::now();
                            let gate_out = mx.expert_matmul(&x_t, expert, ExpertProj::Gate)?;
                            let _ = device.synchronize();
                            let t1 = std::time::Instant::now();
                            let up_out = mx.expert_matmul(&x_t, expert, ExpertProj::Up)?;
                            let _ = device.synchronize();
                            let t2 = std::time::Instant::now();
                            let hidden_out = (candle_nn::ops::silu(&gate_out)? * up_out)?;
                            let _ = device.synchronize();
                            let t3 = std::time::Instant::now();
                            let down_out = mx.expert_matmul(&hidden_out, expert, ExpertProj::Down)?;
                            let _ = device.synchronize();
                            let t4 = std::time::Instant::now();
                            let contrib = down_out.affine(w, 0.0)?;
                            acc = Some(match acc {
                                None => contrib,
                                Some(prev) => (prev + contrib)?,
                            });
                            let _ = device.synchronize();
                            let t5 = std::time::Instant::now();
                            gate_ms += t1.duration_since(t0).as_secs_f64() * 1000.0;
                            up_ms += t2.duration_since(t1).as_secs_f64() * 1000.0;
                            silu_mul_ms += t3.duration_since(t2).as_secs_f64() * 1000.0;
                            down_ms += t4.duration_since(t3).as_secs_f64() * 1000.0;
                            acc_ms += t5.duration_since(t4).as_secs_f64() * 1000.0;
                        }
                        _ => {
                            let contrib = self.expert_forward(&x_t, expert)?.affine(w, 0.0)?;
                            acc = Some(match acc {
                                None => contrib,
                                Some(prev) => (prev + contrib)?,
                            });
                        }
                    }
                }
                y_rows.push(acc.expect("top_k >= 1"));
            }
            eprintln!(
                "      moe-loop: gate={gate_ms:.1} up={up_ms:.1} silu_mul={silu_mul_ms:.1} down={down_ms:.1} acc={acc_ms:.1} ({} iters)",
                bl * k
            );
            Tensor::cat(&y_rows, 0)?
        } else {
            // Legacy / non-Mxfp4 backend fallback. `scores_host` is guaranteed Some
            // here because `production_grouped` was false → `need_scores_host` true.
            let scores_host_ref = scores_host
                .as_ref()
                .expect("scores_host populated for non-grouped fallback");
            let inds_host_ref = inds_host
                .as_ref()
                .expect("inds_host populated for non-grouped fallback");
            let mut y_rows: Vec<Tensor> = Vec::with_capacity(bl);
            for t in 0..bl {
                let x_t = x_flat.narrow(0, t, 1)?;
                let mut acc: Option<Tensor> = None;
                for j in 0..k {
                    let expert = inds_host_ref[t * k + j];
                    let w = scores_host_ref[t * k + j] as f64;
                    let contrib = self.expert_forward(&x_t, expert)?.affine(w, 0.0)?;
                    acc = Some(match acc {
                        None => contrib,
                        Some(prev) => (prev + contrib)?,
                    });
                }
                y_rows.push(acc.expect("top_k >= 1"));
            }
            Tensor::cat(&y_rows, 0)?
        };
        sync_mark(&mut marks, "routed_loop");

        // ── Shared expert + sigmoid scalar gate ────────────────────────────
        sync_sub(&mut sub_marks, "sh_start");
        let shared_out = if moe_sub_timing {
            self.shared_expert
                .forward_with_marks(&x_flat, Some(&mut sub_marks))?
        } else {
            self.shared_expert.forward(&x_flat)?
        };
        let gate_logit = self.shared_expert_gate.forward(&x_flat)?;
        sync_sub(&mut sub_marks, "sh_gate_proj");
        let shared_coef = candle_nn::ops::sigmoid(&gate_logit)?;
        let shared_y = shared_out.broadcast_mul(&shared_coef)?; // [BL, hidden]
        sync_sub(&mut sub_marks, "sh_sigmoid_mul");
        sync_mark(&mut marks, "shared_expert");

        let summed = (y_routed + shared_y)?;
        sync_sub(&mut sub_marks, "combine_add");
        let out = summed.reshape((batch, seq_len, hidden))?;
        sync_sub(&mut sub_marks, "combine_reshape");
        sync_mark(&mut marks, "combine");

        if moe_timing && marks.len() >= 2 {
            let mut msg = String::from("    moe:");
            let mut total_ms = 0.0;
            for pair in marks.windows(2) {
                let (_, t0) = pair[0];
                let (label, t1) = pair[1];
                let ms = t1.duration_since(t0).as_secs_f64() * 1000.0;
                total_ms += ms;
                msg.push_str(&format!(" {label}={ms:.2}"));
            }
            eprintln!("{msg} total={total_ms:.2}ms (BL={bl} k={k})");
        }
        if moe_sub_timing && sub_marks.len() >= 2 {
            let mut msg = String::from("      moe-sub:");
            for pair in sub_marks.windows(2) {
                let (_, t0) = pair[0];
                let (label, t1) = pair[1];
                let ms = t1.duration_since(t0).as_secs_f64() * 1000.0;
                msg.push_str(&format!(" {label}={ms:.2}"));
            }
            eprintln!("{msg}");
        }
        Ok(out)
    }

    /// Compute `down_proj[e] @ (silu(gate_proj[e] @ x_t) * (up_proj[e] @ x_t))` for a single
    /// token `x_t` and expert index `e`. `x_t` shape `[1, hidden]` → output `[1, hidden]`.
    /// Dispatches to the currently installed [`SwitchMlpBackend`].
    fn expert_forward(&self, x_t: &Tensor, expert: usize) -> CandleResult<Tensor> {
        self.switch_mlp.expert_forward(x_t, expert)
    }

    /// Lever H Step 2: RmsNorm-fused MoE forward. Replaces
    /// `forward(post_rmsnorm_output)` with `forward_with_rmsnorm(raw_h, rms_weight, eps)` —
    /// each of the 4 MoE input consumers (routing gate, routed gate_up,
    /// shared expert gate_up, shared_expert_gate) computes RmsNorm internally
    /// inside its own Metal kernel, eliminating the separate
    /// `post_attention_layernorm.forward` device dispatch.
    ///
    /// **Production-path-only**: requires the Mxfp4 switch_mlp backend; skips
    /// every experimental opt-in branch (small_out_gate, bf16_out, bf16_chain,
    /// Lever C/G fusions, moe_timing). Caller must ensure these flags are
    /// off when `LUMEN_ENABLE_RMSNORM_FUSION=1`. The wsum/down chain is
    /// reused unchanged (those consumers operate on intermediate tensors,
    /// not on `x`).
    ///
    /// `h_raw`: shape `[B, L, hidden]` — the post-attention residual BEFORE
    /// `post_attention_layernorm`.
    /// `rms_weight`: shape `[hidden]` — `post_attention_layernorm.weight()`.
    /// `rms_eps`: f32 — typically `1e-6`.
    /// `residual`: optional pre-norm layer input (shape `[B, L, hidden]` or
    /// flat `[BL, hidden]`). When `Some`, the final `(y_routed + shared_y)`
    /// add at the end of MoE is replaced with a single `tri_add` dispatch
    /// that also folds in the residual — caller (layer.rs) MUST skip its
    /// own `(h + mlp_out)?` add. Lever L1 Step 2.
    /// `next_input_rmsnorm`: optional `(rms_weight, rms_eps)` for the NEXT
    /// layer's `input_layernorm`. Lever L4 — when `Some`, the final combine
    /// is `scalar_mul_tri_add_rmsnorm` which produces BOTH the residual
    /// stream `out` AND a pre-normalized `attn_in` for the next layer.
    /// Returns `(out, attn_in_for_next_layer_opt)` — `attn_in_for_next_layer_opt`
    /// is `Some` iff `next_input_rmsnorm` was `Some` AND `residual` was `Some`.
    /// Caller (model.rs) carries the `attn_in` to skip the next layer's
    /// `input_layernorm` dispatch.
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_with_rmsnorm(
        &self,
        h_raw: &Tensor,
        rms_weight: &Tensor,
        rms_eps: f32,
        residual: Option<&Tensor>,
        next_input_rmsnorm: Option<(&Tensor, f32)>,
    ) -> CandleResult<(Tensor, Option<Tensor>)> {
        let (batch, seq_len, hidden) = h_raw.dims3()?;
        if hidden != self.runtime.dims.hidden_size {
            candle_core::bail!(
                "forward_with_rmsnorm: input hidden {hidden} != config {}",
                self.runtime.dims.hidden_size
            );
        }
        let bl = batch * seq_len;
        let k = self.runtime.top_k;
        let h_flat = h_raw.reshape((bl, hidden))?;

        // The rmsnorm path requires the Mxfp4 switch_mlp backend (production
        // hot path). Get the shared MxFp4Context for routing the dense f32
        // RmsNorm-fused dispatches (gate, shared_expert_gate) — both are
        // int8-affine-dequantized to dense Candle Linear at load time.
        let mxfp4 = match &self.switch_mlp {
            SwitchMlpBackend::Mxfp4(mx) => mx,
            SwitchMlpBackend::Dense(_) => candle_core::bail!(
                "forward_with_rmsnorm: requires Mxfp4 switch_mlp backend"
            ),
        };
        let ctx = mxfp4.ctx();

        // ── Routing ────────────────────────────────────────────────────────
        // Routing gate is int8-affine → Dense; dispatch the dense f32 RmsNorm
        // kernel. Output: [BL, num_experts] f32.
        let logits = self.gate.forward_with_rmsnorm(&h_flat, rms_weight, rms_eps, ctx)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits)?;

        // Top-k via arg_sort + narrow + gather (Lever G fused topk path is
        // skipped — caller must keep LUMEN_ENABLE_ROUTING_TOPK_FUSION=0).
        let sorted_idx = probs.arg_sort_last_dim(false)?;
        let inds = sorted_idx.narrow(D::Minus1, 0, k)?.contiguous()?;
        let scores = probs.gather(&inds, D::Minus1)?;
        let scores = if self.runtime.norm_topk_prob {
            let denom = scores.sum_keepdim(D::Minus1)?;
            scores.broadcast_div(&denom)?
        } else {
            scores
        };

        // ── Routed experts ─────────────────────────────────────────────────
        // CB Phase 2 (2026-04-29): when `bl > 1` and `LUMEN_MOE_MULTI_TOKEN=1`,
        // dispatch the 3 multi-token kernels once instead of looping per token.
        // Each kernel reads `expert_indices[b * k + slot]` for per-token expert
        // routing. Pools have shape `[k*B, *]` interpreted as `[k, B, *]` by the
        // kernels (vs `[B, k, *]` for the legacy per-token loop).
        let multi_token_enabled = std::env::var("LUMEN_MOE_MULTI_TOKEN")
            .map(|v| v == "1")
            .unwrap_or(false)
            && bl > 1;

        let y_routed = if multi_token_enabled {
            let g_pool = Tensor::zeros(
                vec![k * bl, mxfp4.moe_inter],
                candle_core::DType::F32,
                h_flat.device(),
            )?;
            let d_pool = Tensor::zeros(
                vec![k * bl, mxfp4.hidden],
                candle_core::DType::F32,
                h_flat.device(),
            )?;

            // inds is [bl, k] u32; flatten to [bl*k] for the kernel (b-major).
            let inds_flat = inds.flatten_all()?;
            // scores is [bl, k] f32; flatten to [bl*k] (b-major).
            let scores_flat = scores.flatten_all()?;

            // Stage 1: routed gate_up + silu*up + RmsNorm — one dispatch.
            mxfp4.moe_gate_up_silu_mul_rmsnorm_multi_candle_queue_into(
                &h_flat,
                rms_weight,
                rms_eps,
                &inds_flat,
                k,
                bl,
                &g_pool,
            )?;
            let hiddens_big = g_pool.contiguous()?;

            // Stage 2: down — one dispatch.
            mxfp4.moe_down_multi_candle_queue_into(
                &hiddens_big,
                &inds_flat,
                k,
                bl,
                &d_pool,
            )?;
            let downs_big = d_pool.contiguous()?;

            // Stage 3: weighted sum — one dispatch, output [B, hidden].
            let y_out = Tensor::zeros(
                vec![bl, mxfp4.hidden],
                candle_core::DType::F32,
                downs_big.device(),
            )?;
            mxfp4.moe_wsum_multi_candle_queue_into(&downs_big, &scores_flat, k, bl, &y_out)?;
            y_out
        } else {
            // Legacy per-token loop (single-token kernel path; production for bl == 1).
            let g_pool = Tensor::zeros(
                vec![bl * k, mxfp4.moe_inter],
                candle_core::DType::F32,
                h_flat.device(),
            )?;
            let d_pool = Tensor::zeros(
                vec![bl * k, mxfp4.hidden],
                candle_core::DType::F32,
                h_flat.device(),
            )?;

            let mut y_rows: Vec<Tensor> = Vec::with_capacity(bl);
            for t in 0..bl {
                let h_t = h_flat.narrow(0, t, 1)?; // [1, hidden] — RAW
                let inds_t = inds.narrow(0, t, 1)?.reshape((k,))?;

                let g_slice = g_pool.narrow(0, t * k, k)?;
                mxfp4.moe_gate_up_silu_mul_rmsnorm_with_indices_buffer_candle_queue_into(
                    &h_t,
                    rms_weight,
                    rms_eps,
                    &inds_t,
                    k,
                    &g_slice,
                )?;
                let hiddens_big = g_slice.contiguous()?;

                let d_slice = d_pool.narrow(0, t * k, k)?;
                mxfp4.moe_down_with_indices_buffer_candle_queue_into(
                    &hiddens_big,
                    &inds_t,
                    k,
                    &d_slice,
                )?;
                let downs_big = d_slice.contiguous()?;

                let w_kx1 = scores.narrow(0, t, 1)?.reshape((k, 1))?;
                let w_flat = w_kx1.flatten_all()?;
                let row = Tensor::zeros(
                    vec![1, mxfp4.hidden],
                    candle_core::DType::F32,
                    downs_big.device(),
                )?;
                mxfp4.moe_wsum_candle_queue_into(&downs_big, &w_flat, &row)?;
                y_rows.push(row);
            }
            Tensor::cat(&y_rows, 0)?
        };

        // ── Shared expert + sigmoid scalar gate ────────────────────────────
        let shared_out = self
            .shared_expert
            .forward_with_rmsnorm(&h_flat, rms_weight, rms_eps, ctx)?;
        let gate_logit = self
            .shared_expert_gate
            .forward_with_rmsnorm(&h_flat, rms_weight, rms_eps, ctx)?;
        // Sigmoid stays as a Candle dispatch — Step 3 NEGATIVE proved that
        // shader-side `metal::exp` drifts ≤1 ULP vs Candle's f32::exp,
        // breaking bit-identical decode. Keep the transcendental host-side.
        let shared_coef = candle_nn::ops::sigmoid(&gate_logit)?;

        // Lever L1 Step 2/3.5: when residual is provided, fuse the
        // `broadcast_mul + tri_add` 2-op chain into 1 dispatch via
        // `scalar_mul_tri_add_f32`. shared_coef is the per-token scalar
        // (already sigmoided). No transcendental in the kernel → bit-
        // identical preserved. Caller (layer.rs) must skip its own
        // `(h + mlp_out)?` add.
        //
        // Lever L4: when BOTH residual AND next_input_rmsnorm are Some,
        // additionally fold the next layer's input_layernorm into the
        // same kernel — produces (out, attn_in) where attn_in is already
        // pre-normalized using next_layer_rms_weight. Caller (model.rs)
        // carries attn_in across layers to skip next layer's input_layernorm.
        match (residual, next_input_rmsnorm) {
            (Some(res), Some((next_rms_w, next_rms_eps))) => {
                let res_flat = res.reshape((bl, hidden))?;
                let coef_flat = shared_coef.reshape((bl,))?;
                let out_flat = Tensor::zeros(
                    vec![bl, hidden],
                    candle_core::DType::F32,
                    h_flat.device(),
                )?;
                let attn_in_flat = Tensor::zeros(
                    vec![bl, hidden],
                    candle_core::DType::F32,
                    h_flat.device(),
                )?;
                lumen_metal::mxfp4_linear::scalar_mul_tri_add_rmsnorm_f32_candle_queue_into(
                    ctx,
                    &y_routed,
                    &shared_out,
                    &coef_flat,
                    &res_flat,
                    next_rms_w,
                    &out_flat,
                    &attn_in_flat,
                    bl,
                    hidden,
                    next_rms_eps,
                )?;
                let out = out_flat.reshape((batch, seq_len, hidden))?;
                let attn_in = attn_in_flat.reshape((batch, seq_len, hidden))?;
                Ok((out, Some(attn_in)))
            }
            (Some(res), None) => {
                let res_flat = res.reshape((bl, hidden))?;
                let coef_flat = shared_coef.reshape((bl,))?;
                let out_flat = Tensor::zeros(
                    vec![bl, hidden],
                    candle_core::DType::F32,
                    h_flat.device(),
                )?;
                lumen_metal::mxfp4_linear::scalar_mul_tri_add_f32_candle_queue_into(
                    ctx,
                    &y_routed,
                    &shared_out,
                    &coef_flat,
                    &res_flat,
                    &out_flat,
                    bl,
                    hidden,
                )?;
                Ok((out_flat.reshape((batch, seq_len, hidden))?, None))
            }
            (None, _) => {
                let shared_y = shared_out.broadcast_mul(&shared_coef)?;
                let summed = (y_routed + shared_y)?;
                Ok((summed.reshape((batch, seq_len, hidden))?, None))
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SparseMoeError {
    #[error(
        "weight `{name}` has shape {found:?}, expected {expected:?}"
    )]
    WeightShape {
        name: &'static str,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
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

    fn dims_from_fixture() -> MoeDims {
        let cfg: Qwen3_5MoeConfig = serde_json::from_str(CONFIG_JSON).unwrap();
        MoeDims::from_config(&cfg.text_config)
    }

    #[test]
    fn dims_match_config_fixture() {
        let d = dims_from_fixture();
        assert_eq!(d.hidden_size, 2048);
        assert_eq!(d.num_experts, 256);
        assert_eq!(d.moe_intermediate_size, 512);
        assert_eq!(d.shared_expert_intermediate_size, 512);
    }

    /// Real layer-0 MoE shapes from `model-00001-of-00004.safetensors` (2026-04-23).
    ///
    /// Packing rules applied:
    ///   - MXFP4 (`bits=4, group=32`, U32): logical last_dim = packed × 8
    ///   - int8  (`bits=8, group=64`, U32): logical last_dim = packed × 4
    fn canonical_mlx_shapes() -> MoeShapes {
        fn unpack_mxfp4(packed: usize) -> usize {
            packed * 8
        }
        fn unpack_int8(packed: usize) -> usize {
            packed * 4
        }
        MoeShapes {
            // gate.weight header [256, 512] U32 (int8-affine) → logical [256, 2048]
            gate: vec![256, unpack_int8(512)],
            // shared_expert_gate.weight header [1, 512] U32 (int8-affine) → logical [1, 2048]
            shared_expert_gate: vec![1, unpack_int8(512)],
            // shared_expert.{gate,up}_proj.weight [512, 256] U32 (mxfp4) → [512, 2048]
            shared_expert_gate_proj: vec![512, unpack_mxfp4(256)],
            shared_expert_up_proj: vec![512, unpack_mxfp4(256)],
            // shared_expert.down_proj.weight [2048, 64] U32 (mxfp4) → [2048, 512]
            shared_expert_down_proj: vec![2048, unpack_mxfp4(64)],
            // switch_mlp.{gate,up}_proj.weight [256, 512, 256] U32 (mxfp4) → [256, 512, 2048]
            switch_mlp_gate_proj: vec![256, 512, unpack_mxfp4(256)],
            switch_mlp_up_proj: vec![256, 512, unpack_mxfp4(256)],
            // switch_mlp.down_proj.weight [256, 2048, 64] U32 (mxfp4) → [256, 2048, 512]
            switch_mlp_down_proj: vec![256, 2048, unpack_mxfp4(64)],
        }
    }

    #[test]
    fn predicted_shapes_match_real_shard_header() {
        let predicted = dims_from_fixture().shapes();
        let canonical = canonical_mlx_shapes();
        assert_eq!(predicted.gate, canonical.gate);
        assert_eq!(predicted.shared_expert_gate, canonical.shared_expert_gate);
        assert_eq!(predicted.shared_expert_gate_proj, canonical.shared_expert_gate_proj);
        assert_eq!(predicted.shared_expert_up_proj, canonical.shared_expert_up_proj);
        assert_eq!(predicted.shared_expert_down_proj, canonical.shared_expert_down_proj);
        assert_eq!(predicted.switch_mlp_gate_proj, canonical.switch_mlp_gate_proj);
        assert_eq!(predicted.switch_mlp_up_proj, canonical.switch_mlp_up_proj);
        assert_eq!(predicted.switch_mlp_down_proj, canonical.switch_mlp_down_proj);
    }

    /// Sanity: switch_mlp is a single grouped tensor, NOT a per-expert split. The leading
    /// dimension must equal `num_experts`, and the loader must not attempt to iterate
    /// `experts.{i}.*` keys (they don't exist in the checkpoint).
    #[test]
    fn switch_mlp_is_grouped_not_split() {
        let d = dims_from_fixture();
        let s = d.shapes();
        assert_eq!(s.switch_mlp_gate_proj[0], d.num_experts);
        assert_eq!(s.switch_mlp_up_proj[0], d.num_experts);
        assert_eq!(s.switch_mlp_down_proj[0], d.num_experts);
        // Per-expert inner layout: gate/up produce [inter, hidden]; down reverses to [hidden, inter].
        assert_eq!(&s.switch_mlp_gate_proj[1..], &[d.moe_intermediate_size, d.hidden_size]);
        assert_eq!(&s.switch_mlp_down_proj[1..], &[d.hidden_size, d.moe_intermediate_size]);
    }

    #[test]
    fn gate_outputs_one_logit_per_expert() {
        let d = dims_from_fixture();
        let s = d.shapes();
        assert_eq!(s.gate, vec![d.num_experts, d.hidden_size]);
    }

    #[test]
    fn shared_expert_gate_is_single_output() {
        let s = dims_from_fixture().shapes();
        assert_eq!(s.shared_expert_gate[0], 1);
    }

    /// Every [`MlpPart`] / [`ProjKind`] pair must resolve to exactly one shape slot. Keeping
    /// this match exhaustive (no `_` arm) means a future variant addition breaks compilation
    /// here and forces a loader update.
    #[test]
    fn shape_slots_cover_every_mlp_part() {
        use crate::qwen3_5_moe::weights::{MlpPart, ProjKind};
        let all_parts = [
            MlpPart::Gate,
            MlpPart::SharedExpertGate,
            MlpPart::SharedExpert(ProjKind::Gate),
            MlpPart::SharedExpert(ProjKind::Up),
            MlpPart::SharedExpert(ProjKind::Down),
            MlpPart::SwitchMlp(ProjKind::Gate),
            MlpPart::SwitchMlp(ProjKind::Up),
            MlpPart::SwitchMlp(ProjKind::Down),
        ];
        let shapes = dims_from_fixture().shapes();
        for part in all_parts {
            let _shape: &Vec<usize> = match part {
                MlpPart::Gate => &shapes.gate,
                MlpPart::SharedExpertGate => &shapes.shared_expert_gate,
                MlpPart::SharedExpert(ProjKind::Gate) => &shapes.shared_expert_gate_proj,
                MlpPart::SharedExpert(ProjKind::Up) => &shapes.shared_expert_up_proj,
                MlpPart::SharedExpert(ProjKind::Down) => &shapes.shared_expert_down_proj,
                MlpPart::SwitchMlp(ProjKind::Gate) => &shapes.switch_mlp_gate_proj,
                MlpPart::SwitchMlp(ProjKind::Up) => &shapes.switch_mlp_up_proj,
                MlpPart::SwitchMlp(ProjKind::Down) => &shapes.switch_mlp_down_proj,
                // The MoE shape table doesn't carry Dense entries — this test only
                // covers the MoE fixture, which never produces `MlpPart::Dense`. The
                // unreachable arm exists solely to satisfy the exhaustiveness check.
                MlpPart::Dense(_) => unreachable!(
                    "MoE shape coverage test does not include Dense MLP parts"
                ),
            };
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Forward-pass tests (synthetic weights).
    //
    // Fixture-level numerical parity against MLX lives in
    // `tests/moe_fixture.rs`, gated on the HF-cache weight dump. The tests below
    // run in CI and lock in routing semantics, shape invariants, and shared-expert
    // mixing behavior.
    // ─────────────────────────────────────────────────────────────────────

    use candle_core::{Device, Tensor};
    use candle_nn::Linear;
    use rand::{rngs::StdRng, RngExt, SeedableRng};

    /// Tiny dims so CPU tests stay well under a second.
    fn tiny_dims() -> MoeDims {
        MoeDims {
            hidden_size: 8,
            num_experts: 6,
            moe_intermediate_size: 12,
            shared_expert_intermediate_size: 10,
        }
    }

    fn tiny_runtime(top_k: usize, norm: bool) -> SparseMoeRuntime {
        SparseMoeRuntime {
            dims: tiny_dims(),
            top_k,
            norm_topk_prob: norm,
        }
    }

    fn random_tensor(shape: &[usize], rng: &mut StdRng, device: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| rng.random_range(-0.1..0.1)).collect();
        Tensor::from_vec(data, shape, device).unwrap()
    }

    fn build_tiny(rt: SparseMoeRuntime, seed: u64, device: &Device) -> SparseMoeBlock {
        let d = rt.dims;
        let mut rng = StdRng::seed_from_u64(seed);
        let gate = Linear::new(
            random_tensor(&[d.num_experts, d.hidden_size], &mut rng, device),
            None,
        );
        let shared_expert_gate =
            Linear::new(random_tensor(&[1, d.hidden_size], &mut rng, device), None);
        // Option J: pre-fused [2*inter, hidden] gate+up weight (test fixture).
        let shared = SharedExpert::new(
            Linear::new(
                random_tensor(
                    &[2 * d.shared_expert_intermediate_size, d.hidden_size],
                    &mut rng,
                    device,
                ),
                None,
            )
            .into(),
            Linear::new(
                random_tensor(
                    &[d.hidden_size, d.shared_expert_intermediate_size],
                    &mut rng,
                    device,
                ),
                None,
            )
            .into(),
            d.shared_expert_intermediate_size,
        );
        let switch = SwitchMlp::new(
            random_tensor(
                &[d.num_experts, d.moe_intermediate_size, d.hidden_size],
                &mut rng,
                device,
            ),
            random_tensor(
                &[d.num_experts, d.moe_intermediate_size, d.hidden_size],
                &mut rng,
                device,
            ),
            random_tensor(
                &[d.num_experts, d.hidden_size, d.moe_intermediate_size],
                &mut rng,
                device,
            ),
            d,
        )
        .unwrap();
        SparseMoeBlock::new(rt, gate.into(), shared_expert_gate.into(), shared, switch.into())
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
        let block = build_tiny(tiny_runtime(3, true), 0xBEEF, &device);
        let mut rng = StdRng::seed_from_u64(0xFEED);
        let x = random_tensor(&[2, 4, 8], &mut rng, &device);
        let y = block.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 4, 8]);
        assert!(is_finite(&y));
    }

    #[test]
    fn switch_mlp_rejects_wrong_shape() {
        let device = Device::Cpu;
        let d = tiny_dims();
        let mut rng = StdRng::seed_from_u64(0);
        let ok_gate = random_tensor(
            &[d.num_experts, d.moe_intermediate_size, d.hidden_size],
            &mut rng,
            &device,
        );
        let bad_down = random_tensor(
            &[d.num_experts, d.moe_intermediate_size, d.hidden_size], // wrong: should be [E, hidden, inter]
            &mut rng,
            &device,
        );
        let result = SwitchMlp::new(ok_gate.clone(), ok_gate.clone(), bad_down, d);
        match result {
            Ok(_) => panic!("expected shape error for down_proj"),
            Err(SparseMoeError::WeightShape { name, .. }) => {
                assert_eq!(name, "switch_mlp.down_proj");
            }
        }
    }

    /// Reproduce the MLX routing pipeline on a hand-crafted gate output to confirm that
    /// (1) top-k selects the right expert indices, (2) scores are renormalized to sum=1
    /// when `norm_topk_prob`, and (3) the final output equals the weighted sum of the
    /// selected experts' SwiGLU outputs plus the scalar-gated shared expert.
    #[test]
    fn routing_weights_selected_experts_and_adds_shared_expert() {
        let device = Device::Cpu;
        let rt = tiny_runtime(2, true);
        let d = rt.dims;

        // Build weights that make routing predictable.
        // Gate: identity-like projection that picks expert e for token with x[e] dominant.
        // Simpler: set gate so its output per token is just the first num_experts
        // components of x. Use a selector matrix [E, hidden] with 1.0 on the diagonal in
        // the leading E columns, zero elsewhere.
        let mut gate_data = vec![0f32; d.num_experts * d.hidden_size];
        for e in 0..d.num_experts {
            gate_data[e * d.hidden_size + e] = 1.0;
        }
        let gate = Linear::new(
            Tensor::from_vec(gate_data, (d.num_experts, d.hidden_size), &device).unwrap(),
            None,
        );

        // shared_expert_gate = -inf so sigmoid(.) ≈ 0 and the shared path contributes
        // nothing; simplifies the algebraic check below.
        let neg_large = vec![-1e9f32; d.hidden_size];
        let shared_expert_gate = Linear::new(
            Tensor::from_vec(neg_large, (1, d.hidden_size), &device).unwrap(),
            None,
        );

        // Random shared expert; it gets multiplied by ~0 so values don't matter.
        // Option J: pre-fused [2*inter, hidden] gate+up.
        let mut rng = StdRng::seed_from_u64(0);
        let shared = SharedExpert::new(
            Linear::new(
                random_tensor(
                    &[2 * d.shared_expert_intermediate_size, d.hidden_size],
                    &mut rng,
                    &device,
                ),
                None,
            )
            .into(),
            Linear::new(
                random_tensor(
                    &[d.hidden_size, d.shared_expert_intermediate_size],
                    &mut rng,
                    &device,
                ),
                None,
            )
            .into(),
            d.shared_expert_intermediate_size,
        );

        // switch_mlp experts: expert e emits the constant vector e*ones(hidden) for any
        // input. Achieved by zero gate_proj (silu(0)=0) → down_proj output is also zero,
        // EXCEPT we want a non-zero response to test weighting. Simpler trick: set
        // gate_proj so silu(gate_out) = 1, up_proj = 1, down_proj produces the constant.
        // Cleanest: gate_proj outputs a constant large positive (silu≈x for large x);
        // up_proj = ones; down_proj = e on the diagonal.
        //
        // To stay tractable we hand-craft outputs for a single-token test and check the
        // weighted sum numerically rather than with an explicit constant-expert pattern.
        // That isolates the routing+weighting logic.
        let switch = SwitchMlp::new(
            random_tensor(
                &[d.num_experts, d.moe_intermediate_size, d.hidden_size],
                &mut rng,
                &device,
            ),
            random_tensor(
                &[d.num_experts, d.moe_intermediate_size, d.hidden_size],
                &mut rng,
                &device,
            ),
            random_tensor(
                &[d.num_experts, d.hidden_size, d.moe_intermediate_size],
                &mut rng,
                &device,
            ),
            d,
        )
        .unwrap();
        let block = SparseMoeBlock::new(
            rt,
            gate.into(),
            shared_expert_gate.into(),
            shared,
            switch.into(),
        );

        // Token whose gate logits pick experts 5 and 4 (the two largest components of x).
        // x[5]=3.0, x[4]=2.0, rest=0 → softmax concentrates on those two.
        let mut x_row = vec![0f32; d.hidden_size];
        x_row[5] = 3.0;
        x_row[4] = 2.0;
        let x = Tensor::from_vec(x_row.clone(), (1, 1, d.hidden_size), &device).unwrap();

        let y = block.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 1, d.hidden_size]);

        // Independently recompute what the routed sum should be: for each of the top-2
        // selected experts, run the SwiGLU FFN and weight by normalized softmax probs.
        let logits = block.gate.forward(&x.reshape((1, d.hidden_size)).unwrap()).unwrap();
        let probs = candle_nn::ops::softmax_last_dim(&logits).unwrap().to_vec2::<f32>().unwrap();
        let probs_row = &probs[0];
        // Find top-2 indices.
        let mut ranked: Vec<(usize, f32)> = probs_row.iter().cloned().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let (e1, e2) = (ranked[0].0, ranked[1].0);
        let p1 = ranked[0].1;
        let p2 = ranked[1].1;
        let sum_p = p1 + p2;
        let w1 = (p1 / sum_p) as f64;
        let w2 = (p2 / sum_p) as f64;

        let x_t = x.reshape((1, d.hidden_size)).unwrap();
        let contrib1 = block.expert_forward(&x_t, e1).unwrap().affine(w1, 0.0).unwrap();
        let contrib2 = block.expert_forward(&x_t, e2).unwrap().affine(w2, 0.0).unwrap();
        let expected = (contrib1 + contrib2).unwrap();
        // shared_y ≈ 0 because sigmoid(shared_expert_gate(x)) ≈ 0; the rtol is set generously
        // because sigmoid(-1e9)·shared_out is a tiny but nonzero drift.
        let y_flat = y.reshape((1, d.hidden_size)).unwrap();
        let diff = (&y_flat - &expected)
            .unwrap()
            .abs()
            .unwrap()
            .max(0)
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let ref_norm = expected
            .abs()
            .unwrap()
            .max(0)
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .max(1e-6);
        assert!(
            diff / ref_norm < 1e-5,
            "routing result disagreement: max_diff={diff}, ref_max={ref_norm}"
        );
        assert!(
            (w1 + w2 - 1.0).abs() < 1e-5,
            "norm_topk_prob should renormalize to sum=1; got {}",
            w1 + w2
        );
    }

    /// Without `norm_topk_prob`, the raw softmax probabilities are used as weights directly
    /// (no renormalization). Verify that flipping the flag changes the output scale.
    #[test]
    fn norm_topk_prob_flag_controls_renormalization() {
        let device = Device::Cpu;
        let with_norm = build_tiny(tiny_runtime(3, true), 0xA11CE, &device);
        let without_norm = build_tiny(tiny_runtime(3, false), 0xA11CE, &device);
        let mut rng = StdRng::seed_from_u64(0xFACE);
        let x = random_tensor(&[1, 2, 8], &mut rng, &device);
        let y1 = with_norm.forward(&x).unwrap();
        let y2 = without_norm.forward(&x).unwrap();
        // Outputs must differ — they cannot both be correct for both flag states.
        let diff = (&y1 - &y2)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff > 1e-6, "norm flag should influence output; got diff={diff}");
    }

    fn build_tiny_dense(seed: u64, hidden: usize, intermediate: usize, device: &Device) -> DenseMlp {
        let mut rng = StdRng::seed_from_u64(seed);
        // gate+up fused along axis 0 → [2 * intermediate, hidden] (matches loader output).
        let gate_up = Linear::new(
            random_tensor(&[2 * intermediate, hidden], &mut rng, device),
            None,
        );
        let down = Linear::new(
            random_tensor(&[hidden, intermediate], &mut rng, device),
            None,
        );
        DenseMlp::new(gate_up.into(), down.into(), intermediate)
    }

    #[test]
    fn dense_mlp_forward_returns_hidden_shape_and_is_finite() {
        let device = Device::Cpu;
        let hidden = 8;
        let intermediate = 16;
        let dense = build_tiny_dense(0xD415E, hidden, intermediate, &device);
        let mut rng = StdRng::seed_from_u64(0xFACE);
        let x = random_tensor(&[2, 4, hidden], &mut rng, &device);
        let y = dense.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 4, hidden]);
        assert!(is_finite(&y));
    }

    #[test]
    fn dense_mlp_responds_to_input_changes() {
        // Different inputs must produce different outputs — guards against the SwiGLU pair
        // collapsing to a constant (e.g. if narrow indices were swapped or gate/up swapped).
        let device = Device::Cpu;
        let dense = build_tiny_dense(0xD416E, 8, 16, &device);
        let mut rng_a = StdRng::seed_from_u64(0xA11);
        let mut rng_b = StdRng::seed_from_u64(0xB22);
        let xa = random_tensor(&[1, 2, 8], &mut rng_a, &device);
        let xb = random_tensor(&[1, 2, 8], &mut rng_b, &device);
        let ya = dense.forward(&xa).unwrap();
        let yb = dense.forward(&xb).unwrap();
        let diff = (&ya - &yb)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        // Threshold is well above the f32 noise floor (~1e-7) but tolerant of the
        // tiny weight scale `random_tensor` uses (`±0.1` range).
        assert!(diff > 1e-6, "dense MLP outputs should differ across inputs; got {diff}");
    }

    #[test]
    fn mlpblock_dispatches_dense_and_moe_via_forward() {
        // Dispatch sanity: MlpBlock::forward must route to the correct inner block,
        // producing the same output as calling the inner directly.
        let device = Device::Cpu;
        let dense = build_tiny_dense(0x12, 8, 16, &device);
        let moe = build_tiny(tiny_runtime(3, true), 0x34, &device);

        let mut rng = StdRng::seed_from_u64(0xABCD);
        let x = random_tensor(&[1, 3, 8], &mut rng, &device);

        let dense_direct = dense.forward(&x).unwrap();
        let moe_direct = moe.forward(&x).unwrap();

        let dense_block: MlpBlock = dense.into();
        let moe_block: MlpBlock = moe.into();

        let dense_via_enum = dense_block.forward(&x).unwrap();
        let moe_via_enum = moe_block.forward(&x).unwrap();

        let dd = (&dense_direct - &dense_via_enum)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let dm = (&moe_direct - &moe_via_enum)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(dd < 1e-6, "MlpBlock::Dense should match DenseMlp::forward exactly; got {dd}");
        assert!(dm < 1e-6, "MlpBlock::Moe should match SparseMoeBlock::forward exactly; got {dm}");
    }

    #[test]
    fn mlpblock_has_mxfp4_backend_reflects_variant() {
        let device = Device::Cpu;
        let dense = build_tiny_dense(0x55, 8, 16, &device);
        let moe = build_tiny(tiny_runtime(3, true), 0x66, &device);

        let dense_block: MlpBlock = dense.into();
        let moe_block: MlpBlock = moe.into();

        // CPU fixture MoE uses the `Dense(SwitchMlp)` backend (not Mxfp4) → false.
        // The Dense MLP variant always returns false (no fused mxfp4 path yet).
        // What this test guards: Dense must NEVER claim an mxfp4 backend, otherwise
        // layer.rs would route through `forward_with_rmsnorm` which does not exist
        // on DenseMlp and would panic.
        assert!(
            !dense_block.has_mxfp4_backend(),
            "Dense MLP must report no mxfp4 backend"
        );
        // MoE fixture is CPU-only → also false here, but the assertion targets the
        // Dense invariant; the MoE GPU path is exercised in the production-path tests.
        let _ = moe_block.has_mxfp4_backend();

        assert!(matches!(dense_block, MlpBlock::Dense(_)));
        assert!(matches!(moe_block, MlpBlock::Moe(_)));
    }
}
