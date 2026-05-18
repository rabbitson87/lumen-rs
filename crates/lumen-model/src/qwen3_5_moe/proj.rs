//! Dispatch wrapper around Candle `Linear` and the GPU-resident `Mxfp4Linear`.
//!
//! Every weight slot in the Qwen3.5-VL-MoE decoder that was historically a
//! `candle_nn::Linear` now accepts one of:
//!   - `ProjLinear::Dense(Linear)` — plain/int8-affine weights dequantized at load time.
//!   - `ProjLinear::Mxfp4(Mxfp4Linear)` — MXFP4 weights held packed on GPU; dequant +
//!     matmul happens inside the Metal kernel at forward time.
//!
//! The `Mxfp4` variant is only compiled when the `turboquant-gpu` feature is enabled.
//! Dense-only builds still type-check — they just never observe an `Mxfp4` variant.

use candle_core::{Result as CandleResult, Tensor};
use candle_nn::{Linear, Module};

#[cfg(feature = "turboquant-gpu")]
use std::sync::Arc;

#[cfg(feature = "turboquant-gpu")]
use lumen_metal::affine4_linear::Affine4Linear;
#[cfg(feature = "turboquant-gpu")]
use lumen_metal::mxfp4_gpu::MxFp4Context;
#[cfg(feature = "turboquant-gpu")]
use lumen_metal::mxfp4_linear::{
    dense_f32_matmul_rmsnorm_candle_queue_into, Mxfp4Linear,
};

pub enum ProjLinear {
    Dense(Linear),
    #[cfg(feature = "turboquant-gpu")]
    Mxfp4(Mxfp4Linear),
    /// 4-bit affine quantized, GPU-resident. Used by `affine`-mode MLX checkpoints
    /// (e.g. Qwen3.6-27B-MLX-4bit). bf16/fusion variants fall back to f32 + cast
    /// in this MVP — no bespoke fused kernels yet.
    #[cfg(feature = "turboquant-gpu")]
    Affine4(Affine4Linear),
}

impl ProjLinear {
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Dense(l) => l.forward(x),
            #[cfg(feature = "turboquant-gpu")]
            Self::Mxfp4(l) => l
                .forward(x)
                .map_err(|e| candle_core::Error::Msg(format!("mxfp4 proj: {e}"))),
            #[cfg(feature = "turboquant-gpu")]
            Self::Affine4(l) => l
                .forward(x)
                .map_err(|e| candle_core::Error::Msg(format!("affine4 proj: {e}"))),
        }
    }

    /// bf16-output variant of [`Self::forward`]. Returns
    /// a `DType::BF16` tensor when this projection is MXFP4-quantized; for the
    /// dense fallback path computes f32 then casts to bf16 (slow path — only
    /// hits for non-quantized weights, which is a debug/fixture build mode).
    ///
    /// Caller decides downstream dtype handling:
    ///   - propagate bf16 (downstream Candle ops handle it via auto-promotion)
    ///   - explicit `to_dtype(F32)` for a clean A/B baseline against
    ///     [`Self::forward`]
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_bf16_out(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Dense(l) => {
                let y = l.forward(x)?;
                y.to_dtype(candle_core::DType::BF16)
            }
            Self::Mxfp4(l) => l
                .forward_bf16_out(x)
                .map_err(|e| candle_core::Error::Msg(format!("mxfp4 proj bf16: {e}"))),
            Self::Affine4(l) => l
                .forward_bf16_out(x)
                .map_err(|e| candle_core::Error::Msg(format!("affine4 proj bf16-out: {e}"))),
        }
    }

    /// Lever B L.3 (2026-04-28) bf16-input variant of [`Self::forward`].
    /// Activation is bf16 (typically produced by `MpsRmsNormBf16Out`); output
    /// is f32 — the matmul accumulator stays f32 inside the MXFP4 kernel,
    /// only the activation read narrows.
    ///
    /// Dense fallback: widens bf16 → f32 and routes through the normal Linear
    /// (no native bf16-input kernel for plain Candle Linear). Production hits
    /// only the MXFP4 path; the Dense branch exists for fixture parity.
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_bf16_in(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Dense(l) => {
                let x_f32 = if x.dtype() == candle_core::DType::F32 {
                    x.clone()
                } else {
                    x.to_dtype(candle_core::DType::F32)?
                };
                l.forward(&x_f32)
            }
            Self::Mxfp4(l) => l
                .forward_bf16_in(x)
                .map_err(|e| candle_core::Error::Msg(format!("mxfp4 proj bf16-in: {e}"))),
            Self::Affine4(l) => l
                .forward_bf16_in(x)
                .map_err(|e| candle_core::Error::Msg(format!("affine4 proj bf16-in: {e}"))),
        }
    }

    /// Workstream B Phase 3 (2026-05-08) bf16-in-bf16-out variant. Both
    /// activation read and result store narrow to bf16; the matmul accumulator
    /// inside the kernel stays f32. Used by the self_attn / linear_attn bf16
    /// chain to skip the cast-back-to-f32 at the projection boundary that the
    /// older `forward_bf16_in` pattern still pays.
    ///
    /// Affine4: dispatches the native `affine4_qmv_fast_bf16in_bf16out` kernel
    /// (Workstream A) when shape qualifies (`in % 512 == 0 && out % 8 == 0`);
    /// otherwise falls back to `forward_bf16_in` + `to_dtype(BF16)` cast.
    ///
    /// Mxfp4: no native kernel variant yet — uses `forward_bf16_in` + cast.
    /// Production target (Qwen3.6-27B-MLX-4bit) is Affine4-only, so the Mxfp4
    /// branch is fixture-parity only.
    ///
    /// Dense: same composition fallback.
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_bf16_in_bf16_out(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Dense(l) => {
                let x_f32 = if x.dtype() == candle_core::DType::F32 {
                    x.clone()
                } else {
                    x.to_dtype(candle_core::DType::F32)?
                };
                let y = l.forward(&x_f32)?;
                y.to_dtype(candle_core::DType::BF16)
            }
            Self::Mxfp4(l) => {
                // No native bf16-in-bf16-out kernel for Mxfp4. Compose the
                // bf16-in path with a cast — equivalent to the bf16 chain's
                // existing trap, but kept here so Mxfp4 fixtures still build.
                let y_f32 = l
                    .forward_bf16_in(x)
                    .map_err(|e| candle_core::Error::Msg(format!("mxfp4 proj bf16-in: {e}")))?;
                y_f32.to_dtype(candle_core::DType::BF16)
            }
            Self::Affine4(l) => l
                .forward_bf16_in_bf16_out_icb(x)
                .map_err(|e| candle_core::Error::Msg(format!("affine4 proj bf16-in-bf16-out: {e}"))),
        }
    }

    /// bf16-in / bf16-out / bf16 residual fused fast-path. Closes the B.10
    /// σ-NEGATIVE root cause: under bf16 carrier, the f32 fused-residual
    /// path can't fire (caller carries bf16 residual), so the previous
    /// fallback split into matmul + separate broadcast_add — measured
    /// +234 ms / decode batch in profile data. This routes through the
    /// `qmv_fast_bf16in_bf16out_residual` Metal kernel when shape qualifies.
    ///
    /// Caller contract: `x` and `residual` MUST be bf16 (or convertible).
    /// Output is bf16. Non-Affine4 backends fall back to compose
    /// (`forward_bf16_in_bf16_out` + `broadcast_add(bf16)`).
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_with_residual_bf16_in_bf16_out(
        &self,
        x: &Tensor,
        residual: &Tensor,
    ) -> CandleResult<Tensor> {
        match self {
            Self::Affine4(l) => l
                .forward_with_residual_bf16_in_bf16_out_icb(x, residual)
                .map_err(|e| {
                    candle_core::Error::Msg(format!(
                        "affine4 forward_with_residual_bf16_in_bf16_out: {e}"
                    ))
                }),
            // Dense / Mxfp4 fallback: same composition the call sites used
            // before this fast path existed. Mxfp4 production target
            // (35B-A3B MoE) doesn't drive the 27B Dense bf16 chain, so the
            // compose path is acceptable and identical to prior behavior.
            _ => {
                let y_bf16 = self.forward_bf16_in_bf16_out(x)?;
                y_bf16.broadcast_add(residual)
            }
        }
    }

    /// bf16-output variant of
    /// [`Self::forward_gate_up_silu_mul`].
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_gate_up_silu_mul_bf16_out(
        &self,
        x: &Tensor,
        inter: usize,
    ) -> CandleResult<Tensor> {
        match self {
            Self::Dense(l) => {
                let combined = l.forward(x)?;
                let last = combined.dims().len() - 1;
                let gate = combined.narrow(last, 0, inter)?.contiguous()?;
                let up = combined.narrow(last, inter, inter)?.contiguous()?;
                let h = (candle_nn::ops::silu(&gate)? * up)?;
                h.to_dtype(candle_core::DType::BF16)
            }
            Self::Mxfp4(l) => l.forward_gate_up_silu_mul_bf16_out(x).map_err(|e| {
                candle_core::Error::Msg(format!("mxfp4 gate_up_silu_mul bf16: {e}"))
            }),
            Self::Affine4(l) => {
                // Sanity: caller's `inter` must match the kernel's view of inter
                // (= weight.out_features / 2). For Dense MLP this is always true.
                let _ = inter;
                l.forward_gate_up_silu_mul_bf16_out(x)
                    .map_err(|e| candle_core::Error::Msg(format!("affine4 gate_up bf16: {e}")))
            }
        }
    }

    /// bf16-output variant of [`Self::forward_small_out`].
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_small_out_bf16_out(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Dense(l) => {
                let y = l.forward(x)?;
                y.to_dtype(candle_core::DType::BF16)
            }
            Self::Mxfp4(l) => l
                .forward_small_out_bf16_out(x)
                .map_err(|e| candle_core::Error::Msg(format!("mxfp4 small_out bf16: {e}"))),
            Self::Affine4(l) => {
                let y = l.forward(x).map_err(|e| {
                    candle_core::Error::Msg(format!("affine4 small_out bf16: {e}"))
                })?;
                y.to_dtype(candle_core::DType::BF16)
            }
        }
    }

    /// fused `silu(x @ W_gate^T) * (x @ W_up^T)` for SharedExpert.
    ///
    /// Used by [`super::moe::SharedExpert::forward_with_marks`] to collapse the
    /// 3-step gate_up matmul → silu → mul pipeline into a single MXFP4 kernel.
    /// Dense backend transparently falls back to the explicit pipeline so
    /// CPU/non-quantized fixtures are unaffected.
    ///
    /// `inter` is the intermediate width (half of the gate_up weight rows);
    /// the output's last dimension equals `inter`.
    pub fn forward_gate_up_silu_mul(
        &self,
        x: &Tensor,
        inter: usize,
    ) -> CandleResult<Tensor> {
        match self {
            Self::Dense(l) => {
                let combined = l.forward(x)?;
                let last = combined.dims().len() - 1;
                let gate = combined.narrow(last, 0, inter)?.contiguous()?;
                let up = combined.narrow(last, inter, inter)?.contiguous()?;
                candle_nn::ops::silu(&gate)? * up
            }
            #[cfg(feature = "turboquant-gpu")]
            Self::Mxfp4(l) => l.forward_gate_up_silu_mul(x).map_err(|e| {
                candle_core::Error::Msg(format!("mxfp4 gate_up_silu_mul: {e}"))
            }),
            #[cfg(feature = "turboquant-gpu")]
            Self::Affine4(l) => {
                let _ = inter;
                l.forward_gate_up_silu_mul(x).map_err(|e| {
                    candle_core::Error::Msg(format!("affine4 gate_up_silu_mul: {e}"))
                })
            }
        }
    }

    /// route through the small-out matmul kernel when this is
    /// `Mxfp4`. Dense path is identical to [`Self::forward`] (no kernel
    /// alternative), so the caller simply pays a method-dispatch noop.
    pub fn forward_small_out(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Dense(l) => l.forward(x),
            #[cfg(feature = "turboquant-gpu")]
            Self::Mxfp4(l) => l
                .forward_small_out(x)
                .map_err(|e| candle_core::Error::Msg(format!("mxfp4 small_out: {e}"))),
            #[cfg(feature = "turboquant-gpu")]
            Self::Affine4(l) => l
                .forward(x)
                .map_err(|e| candle_core::Error::Msg(format!("affine4 small_out: {e}"))),
        }
    }

    /// Lever L1 (residual fusion): `y = x @ W^T + residual` in a single
    /// MXFP4 dispatch. Folds the downstream Tensor `+` element-wise add
    /// into the matmul kernel's tail. f32 input, f32 output. Dense fallback
    /// computes f32 matmul + broadcast_add in two ops (still parity).
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_with_residual(&self, x: &Tensor, residual: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Dense(l) => {
                let y = l.forward(x)?;
                y.broadcast_add(residual)
            }
            Self::Mxfp4(l) => l.forward_with_residual_f32(x, residual).map_err(|e| {
                candle_core::Error::Msg(format!("mxfp4 forward_with_residual: {e}"))
            }),
            Self::Affine4(l) => l.forward_with_residual_f32(x, residual).map_err(|e| {
                candle_core::Error::Msg(format!("affine4 forward_with_residual: {e}"))
            }),
        }
    }

    /// Lever H Step 2: RmsNorm-fused forward. Reads RAW x (un-normalized) and
    /// the `post_attention_layernorm` weight; the underlying Metal kernel
    /// computes RmsNorm cooperatively before the matmul.
    ///
    /// `ctx` is required for the Dense (int8-affine-dequantized) path because
    /// `ProjLinear::Dense` doesn't carry a Metal context internally; the caller
    /// (e.g. `SparseMoeBlock::forward_with_rmsnorm`) supplies it from any
    /// MXFP4 backend it has access to. For the MXFP4 path `ctx` is ignored
    /// (the inner [`Mxfp4Linear`] already owns its context).
    #[cfg(feature = "turboquant-gpu")]
    pub fn forward_with_rmsnorm(
        &self,
        x_raw: &Tensor,
        rms_weight: &Tensor,
        rms_eps: f32,
        ctx: &Arc<MxFp4Context>,
    ) -> CandleResult<Tensor> {
        match self {
            Self::Dense(l) => dense_f32_matmul_rmsnorm_candle_queue_into(
                ctx,
                l.weight(),
                x_raw,
                rms_weight,
                rms_eps,
            ),
            Self::Mxfp4(l) => l
                .forward_with_rmsnorm(x_raw, rms_weight, rms_eps)
                .map_err(|e| {
                    candle_core::Error::Msg(format!("mxfp4 forward_with_rmsnorm: {e}"))
                }),
            Self::Affine4(l) => l
                .forward_with_rmsnorm(x_raw, rms_weight, rms_eps)
                .map_err(|e| {
                    candle_core::Error::Msg(format!("affine4 forward_with_rmsnorm: {e}"))
                }),
        }
    }

    pub fn in_features(&self) -> usize {
        match self {
            Self::Dense(l) => l.weight().dims()[1],
            #[cfg(feature = "turboquant-gpu")]
            Self::Mxfp4(l) => l.in_features(),
            #[cfg(feature = "turboquant-gpu")]
            Self::Affine4(l) => l.in_features(),
        }
    }

    pub fn out_features(&self) -> usize {
        match self {
            Self::Dense(l) => l.weight().dims()[0],
            #[cfg(feature = "turboquant-gpu")]
            Self::Mxfp4(l) => l.out_features(),
            #[cfg(feature = "turboquant-gpu")]
            Self::Affine4(l) => l.out_features(),
        }
    }

    /// True iff this projection holds MXFP4 packed weights on GPU.
    pub fn is_mxfp4(&self) -> bool {
        #[cfg(feature = "turboquant-gpu")]
        {
            matches!(self, Self::Mxfp4(_))
        }
        #[cfg(not(feature = "turboquant-gpu"))]
        {
            false
        }
    }

    /// Borrow the MXFP4 inner if this projection is MXFP4-quantized; `None`
    /// otherwise. Used by the native fused forward path to skip the Candle
    /// Tensor round-trip and dispatch directly into a shared command buffer.
    #[cfg(feature = "turboquant-gpu")]
    pub fn as_mxfp4(&self) -> Option<&Mxfp4Linear> {
        match self {
            Self::Mxfp4(l) => Some(l),
            Self::Dense(_) | Self::Affine4(_) => None,
        }
    }

    /// True iff this projection is 4-bit affine quantized (GPU-resident).
    pub fn is_affine4(&self) -> bool {
        #[cfg(feature = "turboquant-gpu")]
        {
            matches!(self, Self::Affine4(_))
        }
        #[cfg(not(feature = "turboquant-gpu"))]
        {
            false
        }
    }

    /// Borrow the inner `Affine4Linear` if this projection is 4-bit affine
    /// quantized; `None` otherwise. Used by callsites that need direct access
    /// to the GPU-resident wrapper without going through ProjLinear's match.
    #[cfg(feature = "turboquant-gpu")]
    pub fn as_affine4(&self) -> Option<&Affine4Linear> {
        match self {
            Self::Affine4(l) => Some(l),
            Self::Dense(_) | Self::Mxfp4(_) => None,
        }
    }
}

impl From<Linear> for ProjLinear {
    fn from(l: Linear) -> Self {
        Self::Dense(l)
    }
}

#[cfg(feature = "turboquant-gpu")]
impl From<Affine4Linear> for ProjLinear {
    fn from(l: Affine4Linear) -> Self {
        Self::Affine4(l)
    }
}

#[cfg(feature = "turboquant-gpu")]
impl From<Mxfp4Linear> for ProjLinear {
    fn from(l: Mxfp4Linear) -> Self {
        Self::Mxfp4(l)
    }
}
