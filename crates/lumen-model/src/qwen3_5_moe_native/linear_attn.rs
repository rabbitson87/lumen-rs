//! Native linear-attention (GatedDeltaNet) helpers (Phase A.4).
//!
//! Two layers:
//!
//! 1. [`forward_ssm_loop`] — host-driven SSM recurrence wrapper around
//!    [`super::kernels::KernelLib::ssm_step`]. `B == 1`, no cache.
//!
//! 2. [`forward_linear_attn`] — full GatedDeltaNet prefill forward. Mirrors
//!    `qwen3_5_moe::linear_attn::GatedDeltaNet::forward` for the no-cache
//!    case (cold conv state, cold SSM state). Pushes the QK weightless
//!    RMSNorm and the SSM loop to native; keeps `in_proj_combined`, the
//!    depthwise conv1d, and `out_proj` on Candle (those are either tiny or
//!    benefit from Candle's existing fused MXFP4 path).
//!
//! Layout assumption: `B == 1`. Every per-timestep slice into the SSM inputs
//! is a contiguous byte range; supporting `B > 1` would require either a
//! stride argument on `ssm_step` or a `[S, B, …]` layout repack. Production
//! decoder traffic is `B == 1` (single-stream autoregressive), so we defer
//! the generalization until a multi-batch use case appears.

use lumen_metal::metal::CommandBufferExt;
use anyhow::{anyhow, Result};
use candle_core::{DType, Result as CandleResult, Tensor, D};
use candle_nn::Conv1d;

use crate::qwen3_5_moe::proj::ProjLinear;

use super::bridge::{from_candle_tensor, from_candle_tensor_no_sync, to_candle_tensor};
use super::context::NativeContext;
use super::kernels::KernelLib;
use super::linear_state::NativeSsmState;
use super::tensor::{NativeDType, NativeTensor};

/// Run the full sequential SSM loop, writing per-step outputs into `y`.
///
/// Shapes (`B == 1`):
///   - `q`, `k`: `[1, S, Hv, Dk]` F32
///   - `v`:      `[1, S, Hv, Dv]` F32
///   - `beta`, `g`: `[1, S, Hv]` F32
///   - `y` (out): `[1, S, Hv, Dv]` F32 (pre-allocated)
///
/// Allocates a fresh state buffer `[1, Hv, Dv, Dk]` initialized to zero. The
/// state is consumed inside the function — callers that want to persist it
/// across calls (KV-cache style) should pass an explicit state buffer; that
/// API lands once we wire the loop into the actual decoder forward.
pub fn forward_ssm_loop(
    ctx: &NativeContext,
    lib: &KernelLib,
    q: &NativeTensor,
    k: &NativeTensor,
    v: &NativeTensor,
    beta: &NativeTensor,
    g: &NativeTensor,
    y: &NativeTensor,
) -> Result<()> {
    for (name, t) in [
        ("q", q),
        ("k", k),
        ("v", v),
        ("beta", beta),
        ("g", g),
        ("y", y),
    ] {
        if t.dtype() != NativeDType::F32 {
            return Err(anyhow!(
                "forward_ssm_loop: {name} dtype {:?} != F32",
                t.dtype()
            ));
        }
    }
    if q.rank() != 4 {
        return Err(anyhow!(
            "forward_ssm_loop: q must be rank 4 [1,S,Hv,Dk], got {:?}",
            q.shape()
        ));
    }
    let (b, s, hv, dk) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    if b != 1 {
        return Err(anyhow!(
            "forward_ssm_loop currently requires B=1 (got {b})"
        ));
    }
    if k.shape() != [1, s, hv, dk] {
        return Err(anyhow!(
            "k shape {:?} != [1, {s}, {hv}, {dk}]",
            k.shape()
        ));
    }
    let dv = v.shape()[3];
    if v.shape() != [1, s, hv, dv] {
        return Err(anyhow!(
            "v shape {:?} != [1, {s}, {hv}, {dv}]",
            v.shape()
        ));
    }
    if beta.shape() != [1, s, hv] || g.shape() != [1, s, hv] {
        return Err(anyhow!(
            "beta/g shapes must be [1,{s},{hv}], got beta={:?} g={:?}",
            beta.shape(),
            g.shape()
        ));
    }
    if y.shape() != [1, s, hv, dv] {
        return Err(anyhow!(
            "y shape {:?} != [1, {s}, {hv}, {dv}]",
            y.shape()
        ));
    }
    if s == 0 || hv == 0 || dv == 0 || dk == 0 {
        return Ok(());
    }

    let state = ctx.zeros(vec![1, hv, dv, dk], NativeDType::F32)?;

    let qk_step = hv * dk * 4;
    let v_step = hv * dv * 4;
    let scalar_step = hv * 4;

    for t in 0..s {
        let q_t = q.slice(q.offset_bytes() + t * qk_step, vec![1, hv, dk])?;
        let k_t = k.slice(k.offset_bytes() + t * qk_step, vec![1, hv, dk])?;
        let v_t = v.slice(v.offset_bytes() + t * v_step, vec![1, hv, dv])?;
        let beta_t = beta.slice(beta.offset_bytes() + t * scalar_step, vec![1, hv])?;
        let g_t = g.slice(g.offset_bytes() + t * scalar_step, vec![1, hv])?;
        let y_t = y.slice(y.offset_bytes() + t * v_step, vec![1, hv, dv])?;
        lib.ssm_step(ctx, &state, &q_t, &k_t, &v_t, &beta_t, &g_t, &y_t)?;
    }

    Ok(())
}

/// Dimensions + numerics for a single GatedDeltaNet block.
///
/// Mirrors the runtime portion of `qwen3_5_moe::linear_attn::LinearAttnDims`
/// + `GatedDeltaNetRuntime` so the native forward stays decoupled from those
/// types (the loader already owns them; we only need the scalars here).
#[derive(Debug, Clone, Copy)]
pub struct LinearAttnConfig {
    pub hidden_size: usize,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_dim: usize,
    pub conv_kernel: usize,
    /// Eps used by the final `RMSNormGated` (model-wide `rms_norm_eps`).
    pub rms_norm_eps: f32,
    /// Eps used by the SSM Q/K weightless RMSNorm. MLX hardcodes this to 1e-6
    /// (distinct from `rms_norm_eps`).
    pub ssm_eps: f32,
}

impl LinearAttnConfig {
    pub fn k_dim(&self) -> usize {
        self.num_k_heads * self.head_dim
    }
    pub fn v_dim(&self) -> usize {
        self.num_v_heads * self.head_dim
    }
    pub fn qkv_dim(&self) -> usize {
        2 * self.k_dim() + self.v_dim()
    }
    pub fn validate(&self) -> Result<()> {
        if self.num_k_heads == 0 || self.num_v_heads == 0 {
            return Err(anyhow!("num_k_heads / num_v_heads must be > 0"));
        }
        if self.num_v_heads % self.num_k_heads != 0 {
            return Err(anyhow!(
                "num_v_heads ({}) must be a multiple of num_k_heads ({})",
                self.num_v_heads,
                self.num_k_heads
            ));
        }
        if self.head_dim == 0 || self.conv_kernel == 0 {
            return Err(anyhow!("head_dim and conv_kernel must be > 0"));
        }
        Ok(())
    }
}

/// Full GatedDeltaNet prefill forward (no cache). Mirrors
/// `qwen3_5_moe::linear_attn::GatedDeltaNet::forward` algorithmically:
///
/// 1. `in_proj_combined(x)` → narrow into `qkv / z / b / a`.
/// 2. Depthwise causal conv1d + SiLU (cold zero state).
/// 3. Split into Q, K, V; reshape to `[B, S, H, D]`.
/// 4. Weightless RMSNorm on Q/K (native), scale by `head_dim^-0.5`.
/// 5. `beta = sigmoid(b)`, `g = exp(-exp(a_log) * softplus(a + dt_bias))`.
/// 6. Repeat Q/K heads to match V head count.
/// 7. Native `forward_ssm_loop` on F32 inputs.
/// 8. RMSNormGated: `rms_norm(y, norm_weight) * silu(z)`.
/// 9. `out_proj(gated.reshape([B, S, v_dim]))`.
///
/// Returns `[B, S, hidden_size]`.
#[allow(clippy::too_many_arguments)]
pub fn forward_linear_attn(
    hidden: &Tensor,
    in_proj_combined: &ProjLinear,
    conv1d: &Conv1d,
    a_log: &Tensor,
    dt_bias: &Tensor,
    norm_weight: &Tensor,
    out_proj: &ProjLinear,
    cfg: &LinearAttnConfig,
    ctx: &NativeContext,
    lib: &KernelLib,
) -> Result<Tensor> {
    cfg.validate()?;
    let (batch, seq_len, hidden_dim) = hidden.dims3().map_err(|e| anyhow!("{e}"))?;
    if hidden_dim != cfg.hidden_size {
        return Err(anyhow!(
            "hidden dim {hidden_dim} != cfg.hidden_size {}",
            cfg.hidden_size
        ));
    }
    if batch != 1 {
        return Err(anyhow!("forward_linear_attn currently requires B=1 (got {batch})"));
    }
    let device = hidden.device().clone();
    let dtype = hidden.dtype();

    // ── 1. fused projections ─────────────────────────────────────────
    let combined = in_proj_combined
        .forward(hidden)
        .map_err(|e| anyhow!("in_proj_combined: {e}"))?;
    let last = combined.dims().len() - 1;
    let qkv_dim = cfg.qkv_dim();
    let v_dim = cfg.v_dim();
    let k_dim = cfg.k_dim();
    let hv = cfg.num_v_heads;
    let hk = cfg.num_k_heads;
    let dh = cfg.head_dim;

    let qkv_flat = combined
        .narrow(last, 0, qkv_dim)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("{e}"))?;
    let z_flat = combined
        .narrow(last, qkv_dim, v_dim)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("{e}"))?;
    let b_flat = combined
        .narrow(last, qkv_dim + v_dim, hv)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("{e}"))?;
    let a_flat = combined
        .narrow(last, qkv_dim + v_dim + hv, hv)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("{e}"))?;

    // ── 2. depthwise causal conv1d + SiLU (no cache, cold zero state) ─
    // Use the broadcast_mul + sum window path used by the Candle reference;
    // this avoids Candle's groups=channels chunked dispatch (one matmul per
    // channel) and matches the GatedDeltaNet::forward optimized branch.
    let conv_pad = cfg.conv_kernel - 1;
    let prev_conv_state =
        Tensor::zeros((batch, conv_pad, qkv_dim), dtype, &device).map_err(|e| anyhow!("{e}"))?;
    let conv_input =
        Tensor::cat(&[&prev_conv_state, &qkv_flat], 1).map_err(|e| anyhow!("{e}"))?;
    let mut window_slices = Vec::with_capacity(cfg.conv_kernel);
    for k in 0..cfg.conv_kernel {
        window_slices.push(
            conv_input
                .narrow(1, k, seq_len)
                .map_err(|e| anyhow!("{e}"))?,
        );
    }
    let windowed = Tensor::stack(&window_slices, 2).map_err(|e| anyhow!("{e}"))?;
    let conv_w = conv1d
        .weight()
        .squeeze(1)
        .and_then(|t| t.transpose(0, 1))
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("conv1d weight reshape: {e}"))?;
    let conv_out = windowed
        .broadcast_mul(
            &conv_w
                .unsqueeze(0)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| anyhow!("{e}"))?,
        )
        .and_then(|t| t.sum(2))
        .map_err(|e| anyhow!("{e}"))?;
    let conv_out = candle_nn::ops::silu(&conv_out).map_err(|e| anyhow!("{e}"))?;

    // ── 3. split q / k / v ───────────────────────────────────────────
    let q = conv_out
        .narrow(D::Minus1, 0, k_dim)
        .and_then(|t| t.reshape((batch, seq_len, hk, dh)))
        .map_err(|e| anyhow!("{e}"))?;
    let k = conv_out
        .narrow(D::Minus1, k_dim, k_dim)
        .and_then(|t| t.reshape((batch, seq_len, hk, dh)))
        .map_err(|e| anyhow!("{e}"))?;
    let v = conv_out
        .narrow(D::Minus1, 2 * k_dim, v_dim)
        .and_then(|t| t.reshape((batch, seq_len, hv, dh)))
        .map_err(|e| anyhow!("{e}"))?;

    // ── 4. native weightless RMSNorm on Q/K + inv_scale ──────────────
    let rows_qk = batch * seq_len * hk;
    let q_2d_f32 = q
        .reshape((rows_qk, dh))
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("{e}"))?;
    let k_2d_f32 = k
        .reshape((rows_qk, dh))
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("{e}"))?;
    let q_2d_n = from_candle_tensor(ctx, &q_2d_f32)?;
    let k_2d_n = from_candle_tensor(ctx, &k_2d_f32)?;
    let q_normed_n = ctx.zeros(vec![rows_qk, dh], NativeDType::F32)?;
    let k_normed_n = ctx.zeros(vec![rows_qk, dh], NativeDType::F32)?;
    lib.rms_norm_weightless(ctx, &q_2d_n, cfg.ssm_eps, &q_normed_n)?;
    lib.rms_norm_weightless(ctx, &k_2d_n, cfg.ssm_eps, &k_normed_n)?;
    let q_normed = to_candle_tensor(&q_normed_n, &device)?
        .reshape((batch, seq_len, hk, dh))
        .map_err(|e| anyhow!("{e}"))?;
    let k_normed = to_candle_tensor(&k_normed_n, &device)?
        .reshape((batch, seq_len, hk, dh))
        .map_err(|e| anyhow!("{e}"))?;

    let inv_scale = (dh as f64).powf(-0.5);
    let q_scaled = q_normed
        .affine(inv_scale * inv_scale, 0.0)
        .and_then(|t| t.to_dtype(dtype))
        .map_err(|e| anyhow!("{e}"))?;
    let k_scaled = k_normed
        .affine(inv_scale, 0.0)
        .and_then(|t| t.to_dtype(dtype))
        .map_err(|e| anyhow!("{e}"))?;

    // ── 5. beta + g (gated_delta_update parameters) ──────────────────
    let beta = candle_nn::ops::sigmoid(&b_flat).map_err(|e| anyhow!("{e}"))?;
    let a_plus_dt = a_flat
        .broadcast_add(&dt_bias.reshape((1, 1, hv)).map_err(|e| anyhow!("{e}"))?)
        .map_err(|e| anyhow!("{e}"))?;
    let softplus_a = stable_softplus(&a_plus_dt).map_err(|e| anyhow!("{e}"))?;
    let a_log_f32 = a_log
        .to_dtype(DType::F32)
        .and_then(|t| t.exp())
        .map_err(|e| anyhow!("{e}"))?;
    let g = softplus_a
        .to_dtype(DType::F32)
        .and_then(|t| t.broadcast_mul(&a_log_f32.reshape((1, 1, hv))?))
        .and_then(|t| t.neg())
        .and_then(|t| t.exp())
        .and_then(|t| t.to_dtype(dtype))
        .map_err(|e| anyhow!("{e}"))?;

    // ── 6. repeat K-head dim up to V-head count ──────────────────────
    let repeats = hv / hk;
    let q_rep = repeat_heads(&q_scaled, repeats).map_err(|e| anyhow!("{e}"))?;
    let k_rep = repeat_heads(&k_scaled, repeats).map_err(|e| anyhow!("{e}"))?;

    // ── 7. dtype to f32 once for SSM loop ────────────────────────────
    let q_f32 = q_rep
        .to_dtype(DType::F32)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("{e}"))?;
    let k_f32 = k_rep
        .to_dtype(DType::F32)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("{e}"))?;
    let v_f32 = v
        .to_dtype(DType::F32)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("{e}"))?;
    let beta_f32 = beta
        .to_dtype(DType::F32)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("{e}"))?;
    let g_f32 = g
        .to_dtype(DType::F32)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("{e}"))?;

    // ── 8. native SSM loop ───────────────────────────────────────────
    let q_n = from_candle_tensor(ctx, &q_f32)?;
    let k_n = from_candle_tensor(ctx, &k_f32)?;
    let v_n = from_candle_tensor(ctx, &v_f32)?;
    let beta_n = from_candle_tensor(ctx, &beta_f32)?;
    let g_n = from_candle_tensor(ctx, &g_f32)?;
    let y_n = ctx.zeros(vec![batch, seq_len, hv, dh], NativeDType::F32)?;
    forward_ssm_loop(ctx, lib, &q_n, &k_n, &v_n, &beta_n, &g_n, &y_n)?;
    let y = to_candle_tensor(&y_n, &device)?
        .to_dtype(dtype)
        .map_err(|e| anyhow!("{e}"))?;

    // ── 9. RMSNormGated: rms_norm(y, norm_weight) * silu(z) ──────────
    let z = z_flat
        .reshape((batch, seq_len, hv, dh))
        .map_err(|e| anyhow!("{e}"))?;
    let y_normed = candle_nn::ops::rms_norm(
        &y.contiguous().map_err(|e| anyhow!("{e}"))?,
        norm_weight,
        cfg.rms_norm_eps,
    )
    .map_err(|e| anyhow!("{e}"))?;
    let gated_f32 = (candle_nn::ops::silu(&z.to_dtype(DType::F32).map_err(|e| anyhow!("{e}"))?)
        .map_err(|e| anyhow!("{e}"))?
        * y_normed
            .to_dtype(DType::F32)
            .map_err(|e| anyhow!("{e}"))?)
    .map_err(|e| anyhow!("{e}"))?;
    let gated = gated_f32.to_dtype(dtype).map_err(|e| anyhow!("{e}"))?;

    let out_flat = gated
        .reshape((batch, seq_len, v_dim))
        .map_err(|e| anyhow!("{e}"))?;
    out_proj
        .forward(&out_flat)
        .map_err(|e| anyhow!("out_proj: {e}"))
}

/// Fused single-cmd-buffer GatedDeltaNet forward (Phase A.8-C.4).
///
/// Same algorithm as [`forward_linear_attn`] but:
///   - Every native op (RMSNorm, scale, sigmoid, broadcast-add, softplus,
///     mul-broadcast, neg-exp, head repeat, SSM loop) lands in a single
///     `MTLCommandBuffer` to amortize ~50µs per-commit overhead. With 30
///     linear-attn layers and a default decode loop ≈ 60+ commits saved per
///     token.
///   - The SSM state buffer is **caller-owned** (`ssm_state: &mut`). Decode
///     callers reuse the same buffer across tokens to preserve recurrent
///     state — which the old [`forward_linear_attn`] could not do because
///     it allocated a fresh zero state inside.
///
/// Conv1d, the QKV split projections, and the final `out_proj` stay on
/// Candle: depthwise conv1d already uses Candle's fused MXFP4 path, and
/// switching it adds no fusion benefit (it sits between the projection
/// and the native pipeline, requiring its own commit either way).
///
/// Pre-conditions:
///   - `ssm_state.shape() == [batch, num_v_heads, head_dim, head_dim]`.
///   - For prefill, the caller must `ssm_state.reset(ctx)` so the buffer is
///     all-zero; for decode-continuation, leave the populated buffer alone.
#[allow(clippy::too_many_arguments)]
pub fn forward_linear_attn_fused(
    hidden: &Tensor,
    in_proj_combined: &ProjLinear,
    conv1d: &Conv1d,
    a_log: &Tensor,
    dt_bias: &Tensor,
    norm_weight: &Tensor,
    out_proj: &ProjLinear,
    cfg: &LinearAttnConfig,
    ctx: &NativeContext,
    lib: &KernelLib,
    ssm_state: &mut NativeSsmState,
) -> Result<Tensor> {
    cfg.validate()?;
    let (batch, seq_len, hidden_dim) = hidden.dims3().map_err(|e| anyhow!("{e}"))?;
    if hidden_dim != cfg.hidden_size {
        return Err(anyhow!(
            "hidden dim {hidden_dim} != cfg.hidden_size {}",
            cfg.hidden_size
        ));
    }
    if batch != 1 {
        return Err(anyhow!(
            "forward_linear_attn_fused currently requires B=1 (got {batch})"
        ));
    }
    let device = hidden.device().clone();
    let dtype = hidden.dtype();
    let qkv_dim = cfg.qkv_dim();
    let v_dim = cfg.v_dim();
    let hv = cfg.num_v_heads;
    let dh = cfg.head_dim;

    if ssm_state.batch() != batch
        || ssm_state.num_v_heads() != hv
        || ssm_state.head_dim_v() != dh
        || ssm_state.head_dim_k() != dh
    {
        return Err(anyhow!(
            "ssm_state shape mismatch: expected [{batch},{hv},{dh},{dh}], got [{},{},{},{}]",
            ssm_state.batch(),
            ssm_state.num_v_heads(),
            ssm_state.head_dim_v(),
            ssm_state.head_dim_k(),
        ));
    }

    // ── 1. fused projections (Candle) ───────────────────────────────────
    let combined = in_proj_combined
        .forward(hidden)
        .map_err(|e| anyhow!("in_proj_combined: {e}"))?;
    let last = combined.dims().len() - 1;
    let qkv_flat = combined
        .narrow(last, 0, qkv_dim)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("{e}"))?;
    let z_flat = combined
        .narrow(last, qkv_dim, v_dim)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("{e}"))?;
    let b_flat = combined
        .narrow(last, qkv_dim + v_dim, hv)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("{e}"))?;
    let a_flat = combined
        .narrow(last, qkv_dim + v_dim + hv, hv)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("{e}"))?;

    // ── 2. depthwise causal conv1d + SiLU (Candle, cold state) ──────────
    let conv_pad = cfg.conv_kernel - 1;
    let prev_conv_state =
        Tensor::zeros((batch, conv_pad, qkv_dim), dtype, &device).map_err(|e| anyhow!("{e}"))?;
    let conv_input =
        Tensor::cat(&[&prev_conv_state, &qkv_flat], 1).map_err(|e| anyhow!("{e}"))?;
    let mut window_slices = Vec::with_capacity(cfg.conv_kernel);
    for k in 0..cfg.conv_kernel {
        window_slices.push(
            conv_input
                .narrow(1, k, seq_len)
                .map_err(|e| anyhow!("{e}"))?,
        );
    }
    let windowed = Tensor::stack(&window_slices, 2).map_err(|e| anyhow!("{e}"))?;
    let conv_w = conv1d
        .weight()
        .squeeze(1)
        .and_then(|t| t.transpose(0, 1))
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("conv1d weight reshape: {e}"))?;
    let conv_out = windowed
        .broadcast_mul(
            &conv_w
                .unsqueeze(0)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| anyhow!("{e}"))?,
        )
        .and_then(|t| t.sum(2))
        .map_err(|e| anyhow!("{e}"))?;
    let conv_out = candle_nn::ops::silu(&conv_out).map_err(|e| anyhow!("{e}"))?;

    run_post_conv_fused(
        &conv_out,
        &z_flat,
        &b_flat,
        &a_flat,
        a_log,
        dt_bias,
        norm_weight,
        out_proj,
        cfg,
        ctx,
        lib,
        ssm_state,
        None,
        None,
    )
}

/// Production wire-in entry point (Phase A.8-C.5). Same fused single-cmd-buffer
/// pipeline as [`forward_linear_attn_fused`] but skips conv1d — the caller has
/// already produced `conv_out` (post-SiLU, post-conv-state-handling) on Candle.
///
/// This is what `qwen3_5_moe::linear_attn::GatedDeltaNet::forward` calls when
/// `LUMEN_LINEAR_ATTN_NATIVE=1`: production keeps conv1d on Candle (it owns
/// the conv-state lifecycle), then hands the post-conv pipeline to native.
///
/// Inputs (Candle):
///   - `conv_out`: `[B, S, qkv_dim]` (post-SiLU)
///   - `z_flat`:   `[B, S, v_dim]`
///   - `b_flat`:   `[B, S, num_v_heads]`
///   - `a_flat`:   `[B, S, num_v_heads]`
///   - `a_log`, `dt_bias`, `norm_weight`: per-layer weights
///
/// Pre-conditions (same as [`forward_linear_attn_fused`]):
///   - B = 1; ssm_state shape matches `[1, num_v_heads, head_dim, head_dim]`.
///   - Caller sized `ssm_state` to match `conv_out`'s leading batch dim.
#[allow(clippy::too_many_arguments)]
pub fn forward_post_conv_fused(
    conv_out: &Tensor,
    z_flat: &Tensor,
    b_flat: &Tensor,
    a_flat: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    norm_weight: &Tensor,
    out_proj: &ProjLinear,
    cfg: &LinearAttnConfig,
    ctx: &NativeContext,
    lib: &KernelLib,
    ssm_state: &mut NativeSsmState,
) -> Result<Tensor> {
    forward_post_conv_fused_with_cache(
        conv_out, z_flat, b_flat, a_flat, a_log, dt_bias, norm_weight, out_proj, cfg, ctx, lib,
        ssm_state, /* cached_dt_bias */ None, /* cached_exp_a_log */ None,
    )
}

/// Same as [`forward_post_conv_fused`] but accepts pre-converted native tensors
/// for `dt_bias` and `exp(a_log)`. Used by production to avoid the per-call
/// BF16→F32 conversion + `exp` GPU dispatch on these constant per-layer weights.
///
/// When `cached_dt_bias` / `cached_exp_a_log` are `Some`, the supplied
/// `dt_bias` / `a_log` Candle tensors are ignored. They must be pre-validated to
/// match the layer's expected shape; this function does not re-check.
#[allow(clippy::too_many_arguments)]
pub fn forward_post_conv_fused_with_cache(
    conv_out: &Tensor,
    z_flat: &Tensor,
    b_flat: &Tensor,
    a_flat: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    norm_weight: &Tensor,
    out_proj: &ProjLinear,
    cfg: &LinearAttnConfig,
    ctx: &NativeContext,
    lib: &KernelLib,
    ssm_state: &mut NativeSsmState,
    cached_dt_bias: Option<&NativeTensor>,
    cached_exp_a_log: Option<&NativeTensor>,
) -> Result<Tensor> {
    cfg.validate()?;
    let (batch, seq_len, qkv_actual) = conv_out.dims3().map_err(|e| anyhow!("{e}"))?;
    if qkv_actual != cfg.qkv_dim() {
        return Err(anyhow!(
            "conv_out trailing dim {qkv_actual} != cfg.qkv_dim() {}",
            cfg.qkv_dim()
        ));
    }
    if batch != 1 {
        return Err(anyhow!(
            "forward_post_conv_fused currently requires B=1 (got {batch})"
        ));
    }
    let hv = cfg.num_v_heads;
    let dh = cfg.head_dim;
    if ssm_state.batch() != batch
        || ssm_state.num_v_heads() != hv
        || ssm_state.head_dim_v() != dh
        || ssm_state.head_dim_k() != dh
    {
        return Err(anyhow!(
            "ssm_state shape mismatch: expected [{batch},{hv},{dh},{dh}], got [{},{},{},{}]",
            ssm_state.batch(),
            ssm_state.num_v_heads(),
            ssm_state.head_dim_v(),
            ssm_state.head_dim_k(),
        ));
    }
    if z_flat.dims() != [batch, seq_len, cfg.v_dim()] {
        return Err(anyhow!(
            "z_flat dims {:?} != [{batch}, {seq_len}, {}]",
            z_flat.dims(),
            cfg.v_dim()
        ));
    }
    if b_flat.dims() != [batch, seq_len, hv] || a_flat.dims() != [batch, seq_len, hv] {
        return Err(anyhow!(
            "b_flat/a_flat must be [{batch}, {seq_len}, {hv}], got b={:?} a={:?}",
            b_flat.dims(),
            a_flat.dims()
        ));
    }
    run_post_conv_fused(
        conv_out,
        z_flat,
        b_flat,
        a_flat,
        a_log,
        dt_bias,
        norm_weight,
        out_proj,
        cfg,
        ctx,
        lib,
        ssm_state,
        cached_dt_bias,
        cached_exp_a_log,
    )
}

/// Lever D L.1.a — same shape as [`forward_post_conv_fused_with_cache`], but
/// encodes on **Candle's command queue** instead of `NativeContext.queue`.
///
/// Eliminates the cross-queue bridge tax that anti-pattern #15 (Lever C
/// L.0.b) measured at ~0.143 ms / cross-queue sync. The legacy path pays
/// 2 syncs per layer (1 to drain Candle before adopting buffers, 1 to drain
/// the native command buffer after). Across 30 linear-attn layers that's
/// 60 syncs / token. This variant keeps every encode on the same queue
/// lineage as the surrounding Candle ops (Phase M.5 Option F + Lever H
/// Step 2 same-queue helper-function pattern), so neither sync is needed:
/// the GPU serializes work in queue submission order.
///
/// Pre-allocates the MXFP4-fast-path output as a Candle tensor and writes
/// to it via a zero-copy `NativeTensor` view, sidestepping the
/// `to_candle_tensor` host round-trip in the legacy path.
#[cfg(feature = "turboquant-gpu")]
#[allow(clippy::too_many_arguments)]
pub fn forward_post_conv_fused_with_cache_candle_queue(
    conv_out: &Tensor,
    z_flat: &Tensor,
    b_flat: &Tensor,
    a_flat: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    norm_weight: &Tensor,
    out_proj: &ProjLinear,
    cfg: &LinearAttnConfig,
    ctx: &NativeContext,
    lib: &KernelLib,
    ssm_state: &mut NativeSsmState,
    cached_dt_bias: Option<&NativeTensor>,
    cached_exp_a_log: Option<&NativeTensor>,
) -> Result<Tensor> {
    cfg.validate()?;
    let (batch, seq_len, qkv_actual) = conv_out.dims3().map_err(|e| anyhow!("{e}"))?;
    if qkv_actual != cfg.qkv_dim() {
        return Err(anyhow!(
            "conv_out trailing dim {qkv_actual} != cfg.qkv_dim() {}",
            cfg.qkv_dim()
        ));
    }
    if batch != 1 {
        return Err(anyhow!(
            "forward_post_conv_fused_candle_queue currently requires B=1 (got {batch})"
        ));
    }
    let hv = cfg.num_v_heads;
    let dh = cfg.head_dim;
    if ssm_state.batch() != batch
        || ssm_state.num_v_heads() != hv
        || ssm_state.head_dim_v() != dh
        || ssm_state.head_dim_k() != dh
    {
        return Err(anyhow!(
            "ssm_state shape mismatch: expected [{batch},{hv},{dh},{dh}], got [{},{},{},{}]",
            ssm_state.batch(),
            ssm_state.num_v_heads(),
            ssm_state.head_dim_v(),
            ssm_state.head_dim_k(),
        ));
    }
    if z_flat.dims() != [batch, seq_len, cfg.v_dim()] {
        return Err(anyhow!(
            "z_flat dims {:?} != [{batch}, {seq_len}, {}]",
            z_flat.dims(),
            cfg.v_dim()
        ));
    }
    if b_flat.dims() != [batch, seq_len, hv] || a_flat.dims() != [batch, seq_len, hv] {
        return Err(anyhow!(
            "b_flat/a_flat must be [{batch}, {seq_len}, {hv}], got b={:?} a={:?}",
            b_flat.dims(),
            a_flat.dims()
        ));
    }
    run_post_conv_fused_candle_queue(
        conv_out,
        z_flat,
        b_flat,
        a_flat,
        a_log,
        dt_bias,
        norm_weight,
        out_proj,
        cfg,
        ctx,
        lib,
        ssm_state,
        cached_dt_bias,
        cached_exp_a_log,
    )
}

/// Shared body of `forward_linear_attn_fused` and `forward_post_conv_fused`:
/// the post-conv pipeline (Q/K split, RMSNorm, gating, repeat, SSM loop,
/// RMSNormGated, out_proj) collapsed into one Metal command buffer.
///
/// `cached_dt_bias` / `cached_exp_a_log`, when `Some`, replace the per-call
/// Candle dtype-conversion + `exp` for the constant per-layer SSM weights.
#[allow(clippy::too_many_arguments)]
fn run_post_conv_fused(
    conv_out: &Tensor,
    z_flat: &Tensor,
    b_flat: &Tensor,
    a_flat: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    norm_weight: &Tensor,
    out_proj: &ProjLinear,
    cfg: &LinearAttnConfig,
    ctx: &NativeContext,
    lib: &KernelLib,
    ssm_state: &mut NativeSsmState,
    cached_dt_bias: Option<&NativeTensor>,
    cached_exp_a_log: Option<&NativeTensor>,
) -> Result<Tensor> {
    let (batch, seq_len, _) = conv_out.dims3().map_err(|e| anyhow!("{e}"))?;
    let device = conv_out.device().clone();
    let dtype = conv_out.dtype();
    let v_dim = cfg.v_dim();
    let k_dim = cfg.k_dim();
    let hv = cfg.num_v_heads;
    let hk = cfg.num_k_heads;
    let dh = cfg.head_dim;

    // Workstream B Phase 6 — bf16-in path. When `conv_out` arrives as bf16
    // (production bf16 chain via `forward_bf16_in_bf16_out` on `in_proj`),
    // skip the entry `to_dtype(F32)` on q/k/v and run the SSM subgraph
    // (rms_norm_weightless / affine_scalar / repeat_heads / ssm_step) in
    // bf16. Tail kernels (rms_norm with weight, silu_mul, MXFP4 out_proj)
    // stay f32 — y_n is cast bf16→f32 once after ssm_step. State stays f32
    // (Escape #3, recurrent SSM state).
    let bf16_in = dtype == DType::BF16;
    let qkv_native_dtype = if bf16_in {
        NativeDType::BF16
    } else {
        NativeDType::F32
    };

    // ── 3. split q / k / v + bridge to native ───────────────────────────
    // bf16_in skips the entry `to_dtype(F32)` on q/k/v (the largest tensors
    // in the bridge). z/b/a/norm_weight stay f32-cast — their consumer
    // kernels (sigmoid/softplus/silu_mul/rms_norm) remain f32-only.
    let q_blhd = if bf16_in {
        conv_out
            .narrow(D::Minus1, 0, k_dim)
            .and_then(|t| t.reshape((batch, seq_len, hk, dh)))
            .and_then(|t| t.contiguous())
            .map_err(|e| anyhow!("{e}"))?
    } else {
        conv_out
            .narrow(D::Minus1, 0, k_dim)
            .and_then(|t| t.reshape((batch, seq_len, hk, dh)))
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.contiguous())
            .map_err(|e| anyhow!("{e}"))?
    };
    let k_blhd = if bf16_in {
        conv_out
            .narrow(D::Minus1, k_dim, k_dim)
            .and_then(|t| t.reshape((batch, seq_len, hk, dh)))
            .and_then(|t| t.contiguous())
            .map_err(|e| anyhow!("{e}"))?
    } else {
        conv_out
            .narrow(D::Minus1, k_dim, k_dim)
            .and_then(|t| t.reshape((batch, seq_len, hk, dh)))
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.contiguous())
            .map_err(|e| anyhow!("{e}"))?
    };
    let v_blhd = if bf16_in {
        conv_out
            .narrow(D::Minus1, 2 * k_dim, v_dim)
            .and_then(|t| t.reshape((batch, seq_len, hv, dh)))
            .and_then(|t| t.contiguous())
            .map_err(|e| anyhow!("{e}"))?
    } else {
        conv_out
            .narrow(D::Minus1, 2 * k_dim, v_dim)
            .and_then(|t| t.reshape((batch, seq_len, hv, dh)))
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.contiguous())
            .map_err(|e| anyhow!("{e}"))?
    };
    let b_f32 = b_flat
        .to_dtype(DType::F32)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("{e}"))?;
    let a_f32 = a_flat
        .to_dtype(DType::F32)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("{e}"))?;
    // drag the RMSNormGated tail (and out_proj when MXFP4) into
    // the same fused command buffer. We need `z_flat` and `norm_weight` as
    // F32 native tensors; cast on the Candle queue here so the upcoming
    // single sync drains them along with q/k/v/b/a.
    let z_f32 = z_flat
        .to_dtype(DType::F32)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("z_flat to f32: {e}"))?;
    let norm_weight_f32 = norm_weight
        .to_dtype(DType::F32)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("norm_weight to f32: {e}"))?;
    // Constant per-layer weights: `dt_bias` and `exp(a_log)` never change after
    // weight load. When the caller supplies pre-converted native copies, skip
    // the per-call BF16→F32 conversion and `exp` GPU dispatch. With 30
    // linear-attn layers per token, this drops 60+ small dispatches/token.
    let (exp_a_log_owned, dt_bias_f32_owned) = if cached_exp_a_log.is_some() && cached_dt_bias.is_some() {
        (None, None)
    } else {
        let e = a_log
            .to_dtype(DType::F32)
            .and_then(|t| t.exp())
            .and_then(|t| t.contiguous())
            .map_err(|e| anyhow!("{e}"))?;
        let d = dt_bias
            .to_dtype(DType::F32)
            .and_then(|t| t.contiguous())
            .map_err(|e| anyhow!("{e}"))?;
        (Some(e), Some(d))
    };

    // Sync Candle's queue exactly once before adopting any of the dynamic buffers.
    // The remaining bridges reuse the same queue lineage, so a per-bridge
    // `wait_until_completed` would be redundant round trips per layer
    // (~180 per token across 30 linear-attn layers). The first call carries
    // the sync; the others use `_no_sync` and rely on the caller-side guarantee.
    let q_in = from_candle_tensor(ctx, &q_blhd)?;
    let k_in = from_candle_tensor_no_sync(ctx, &k_blhd)?;
    let v_in = from_candle_tensor_no_sync(ctx, &v_blhd)?;
    let b_in = from_candle_tensor_no_sync(ctx, &b_f32)?;
    let a_in = from_candle_tensor_no_sync(ctx, &a_f32)?;

    // dt_bias / exp(a_log): cached path takes the prebuilt NativeTensor; cold
    // path bridges the freshly-converted Candle tensor. We hold owned variants
    // alive for the duration of the command buffer in either case.
    let dt_bias_owned_native;
    let exp_a_log_owned_native;
    let dt_bias_in: &NativeTensor = if let Some(c) = cached_dt_bias {
        c
    } else {
        dt_bias_owned_native = from_candle_tensor_no_sync(
            ctx,
            dt_bias_f32_owned.as_ref().expect("uncached dt_bias"),
        )?;
        // SAFETY: tied to function lifetime; not actually unsafe — just shadowing.
        // Borrow-checker dance: we need a `&NativeTensor` whose backing storage
        // outlives the kernel dispatch below. The owned version above does.
        &dt_bias_owned_native
    };
    let exp_a_log_in: &NativeTensor = if let Some(c) = cached_exp_a_log {
        c
    } else {
        exp_a_log_owned_native = from_candle_tensor_no_sync(
            ctx,
            exp_a_log_owned.as_ref().expect("uncached exp_a_log"),
        )?;
        &exp_a_log_owned_native
    };

    // bridges (RMSNormGated tail). The first `from_candle_tensor`
    // above already drained Candle's queue, so these adopt with no extra sync.
    let z_in = from_candle_tensor_no_sync(ctx, &z_f32)?;
    let norm_weight_in = from_candle_tensor_no_sync(ctx, &norm_weight_f32)?;

    // ── 4. native scratch buffers (kept alive until cmd buffer commits) ─
    // qkv subgraph scratch: bf16 in bf16-in path, f32 otherwise. Tail
    // intermediates (y_normed_n, gated_n, out_y_n) stay f32 across both.
    let q_scaled = ctx.zeros(vec![batch, seq_len, hk, dh], qkv_native_dtype)?;
    let k_scaled = ctx.zeros(vec![batch, seq_len, hk, dh], qkv_native_dtype)?;
    let q_rep = ctx.zeros(vec![batch, seq_len, hv, dh], qkv_native_dtype)?;
    let k_rep = ctx.zeros(vec![batch, seq_len, hv, dh], qkv_native_dtype)?;
    let beta = ctx.zeros(vec![batch, seq_len, hv], NativeDType::F32)?;
    let a_plus_dt = ctx.zeros(vec![batch, seq_len, hv], NativeDType::F32)?;
    let softplus_a = ctx.zeros(vec![batch, seq_len, hv], NativeDType::F32)?;
    let g_pre = ctx.zeros(vec![batch, seq_len, hv], NativeDType::F32)?;
    let g = ctx.zeros(vec![batch, seq_len, hv], NativeDType::F32)?;
    let y_n = ctx.zeros(vec![batch, seq_len, hv, dh], qkv_native_dtype)?;
    // bf16-in only: ssm_step_bf16 expects beta/g in bf16. The sigmoid /
    // softplus chain runs in f32; cast just before the SSM loop.
    let beta_bf = if bf16_in {
        Some(ctx.zeros(vec![batch, seq_len, hv], NativeDType::BF16)?)
    } else {
        None
    };
    let g_bf = if bf16_in {
        Some(ctx.zeros(vec![batch, seq_len, hv], NativeDType::BF16)?)
    } else {
        None
    };
    // bf16-in only: the post-SSM RMSNormGated tail consumes y_n through the
    // f32 `rms_norm` kernel. Cast bf16 y_n → y_n_f32 once after the SSM
    // loop, then the tail (and the Dense fallback's `to_candle_tensor`)
    // reads the f32 copy.
    let y_n_f32 = if bf16_in {
        Some(ctx.zeros(vec![batch, seq_len, hv, dh], NativeDType::F32)?)
    } else {
        None
    };

    // fused-tail scratch. RMSNormGated outputs (always allocated
    // since the native tail uses them on the MXFP4 fast path; the legacy
    // Candle fallback simply ignores the buffers).
    let y_normed_n = ctx.zeros(vec![batch, seq_len, hv, dh], NativeDType::F32)?;
    let gated_n = ctx.zeros(vec![batch, seq_len, v_dim], NativeDType::F32)?;
    // `out_proj` matmul output: only allocated when `out_proj` is MXFP4.
    // For dense projections the legacy Candle tail runs after the commit.
    let out_mxfp4 = out_proj.as_mxfp4();
    let out_y_n = if let Some(linear) = out_mxfp4 {
        Some(ctx.zeros(
            vec![batch * seq_len, linear.out_features()],
            NativeDType::F32,
        )?)
    } else {
        None
    };

    // ── 5. single fused command buffer ──────────────────────────────────
    let inv_scale = (dh as f64).powf(-0.5) as f32;
    let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
    let enc = cmd.auto_compute_encoder();
    enc.set_label("lumen:linear_attn_fused");

    if bf16_in {
        lib.encode_rms_norm_weightless_bf16(&enc, &q_in, cfg.ssm_eps, &q_scaled)?;
        lib.encode_rms_norm_weightless_bf16(&enc, &k_in, cfg.ssm_eps, &k_scaled)?;
        lib.encode_affine_scalar_bf16(&enc, &q_scaled, inv_scale * inv_scale, 0.0, &q_scaled)?;
        lib.encode_affine_scalar_bf16(&enc, &k_scaled, inv_scale, 0.0, &k_scaled)?;
    } else {
        lib.encode_rms_norm_weightless(&enc, &q_in, cfg.ssm_eps, &q_scaled)?;
        lib.encode_rms_norm_weightless(&enc, &k_in, cfg.ssm_eps, &k_scaled)?;
        lib.encode_affine_scalar(&enc, &q_scaled, inv_scale * inv_scale, 0.0, &q_scaled)?;
        lib.encode_affine_scalar(&enc, &k_scaled, inv_scale, 0.0, &k_scaled)?;
    }
    // beta/g chain stays f32 regardless — kernels here are f32-only and the
    // tensors are tiny ([B, S, Hv]). bf16-in path casts the outputs to bf16
    // below, just before the SSM loop.
    //
    // fused `compute_g_full` (5→1 dispatch). Mirrors MLX's
    // `@partial(mx.compile) compute_g`. **Default OFF — production A/B
    // NEGATIVE (+3 % regression on Qwen3.6-27B-4bit decode despite 0.70 %
    // microbench saving)** because the 5 small dispatches wave-overlap with
    // surrounding rms_norm/transpose/ssm ops in production's single
    // command buffer, whereas the fused ALU-heavy dispatch breaks that
    // overlap. Canonical case for anti-pattern #27 (microbench≠production).
    // Toggle with `LUMEN_FUSED_COMPUTE_G=1` to re-evaluate when M5/M6 or
    // GPU scheduling changes. See `phase19a4_inner_fusion_negative.md`.
    let fused_compute_g = std::env::var("LUMEN_FUSED_COMPUTE_G")
        .map(|v| v == "1")
        .unwrap_or(false);
    if fused_compute_g {
        lib.encode_compute_g_full(
            &enc,
            &b_in,
            &a_in,
            dt_bias_in,
            exp_a_log_in,
            &beta,
            &g,
        )?;
    } else {
        lib.encode_sigmoid(&enc, &b_in, &beta)?;
        lib.encode_broadcast_add_per_head(&enc, &a_in, dt_bias_in, &a_plus_dt)?;
        lib.encode_softplus(&enc, &a_plus_dt, &softplus_a)?;
        lib.encode_mul_broadcast_per_head(&enc, &softplus_a, exp_a_log_in, &g_pre)?;
        lib.encode_neg_exp(&enc, &g_pre, &g)?;
    }

    let repeats = hv / hk;
    if repeats == 1 {
        if bf16_in {
            lib.encode_affine_scalar_bf16(&enc, &q_scaled, 1.0, 0.0, &q_rep)?;
            lib.encode_affine_scalar_bf16(&enc, &k_scaled, 1.0, 0.0, &k_rep)?;
        } else {
            lib.encode_affine_scalar(&enc, &q_scaled, 1.0, 0.0, &q_rep)?;
            lib.encode_affine_scalar(&enc, &k_scaled, 1.0, 0.0, &k_rep)?;
        }
    } else if bf16_in {
        lib.encode_repeat_heads_blhd_bf16(&enc, &q_scaled, &q_rep, repeats)?;
        lib.encode_repeat_heads_blhd_bf16(&enc, &k_scaled, &k_rep, repeats)?;
    } else {
        lib.encode_repeat_heads_blhd(&enc, &q_scaled, &q_rep, repeats)?;
        lib.encode_repeat_heads_blhd(&enc, &k_scaled, &k_rep, repeats)?;
    }

    if bf16_in {
        // Cast the f32 sigmoid/exp-chain outputs into the bf16 buffers
        // ssm_step_bf16 expects. Two trivial element-wise dispatches over
        // [B, S, Hv].
        lib.encode_cast_f32_to_bf16(&enc, &beta, beta_bf.as_ref().unwrap())?;
        lib.encode_cast_f32_to_bf16(&enc, &g, g_bf.as_ref().unwrap())?;
    }

    let elem = qkv_native_dtype.size_in_bytes();
    let qk_step = hv * dh * elem;
    let v_step = hv * dh * elem;
    let scalar_step_f32 = hv * 4;
    let scalar_step_bf16 = hv * 2;
    for t in 0..seq_len {
        let q_t = q_rep.slice(q_rep.offset_bytes() + t * qk_step, vec![batch, hv, dh])?;
        let k_t = k_rep.slice(k_rep.offset_bytes() + t * qk_step, vec![batch, hv, dh])?;
        let v_t = v_in.slice(v_in.offset_bytes() + t * v_step, vec![batch, hv, dh])?;
        let y_t = y_n.slice(y_n.offset_bytes() + t * v_step, vec![batch, hv, dh])?;
        if bf16_in {
            let beta_b = beta_bf.as_ref().unwrap();
            let g_b = g_bf.as_ref().unwrap();
            let beta_t = beta_b
                .slice(beta_b.offset_bytes() + t * scalar_step_bf16, vec![batch, hv])?;
            let g_t = g_b.slice(g_b.offset_bytes() + t * scalar_step_bf16, vec![batch, hv])?;
            lib.encode_ssm_step_bf16(
                &enc,
                ssm_state.buf(),
                &q_t,
                &k_t,
                &v_t,
                &beta_t,
                &g_t,
                &y_t,
            )?;
        } else {
            let beta_t = beta.slice(beta.offset_bytes() + t * scalar_step_f32, vec![batch, hv])?;
            let g_t = g.slice(g.offset_bytes() + t * scalar_step_f32, vec![batch, hv])?;
            lib.encode_ssm_step(&enc, ssm_state.buf(), &q_t, &k_t, &v_t, &beta_t, &g_t, &y_t)?;
        }
    }

    // bf16-in: cast y_n bf16 → y_n_f32 so the f32 RMSNormGated tail (and the
    // Dense fallback's host bridge) reads a familiar dtype. f32 path: no-op.
    let y_n_for_tail: &NativeTensor = if bf16_in {
        let y_f32 = y_n_f32.as_ref().unwrap();
        lib.encode_cast_bf16_to_f32(&enc, &y_n, y_f32)?;
        y_f32
    } else {
        &y_n
    };

    // RMSNormGated + out_proj fused into the same command buffer
    // when out_proj is MXFP4. Drops 1 commit/layer (~50µs × 30 layers/token =
    // ~1.5ms on M3 Max). For Dense/int8 out_proj, the legacy Candle tail below
    // runs after the commit instead.
    if let Some(out_mxfp4_linear) = out_mxfp4 {
        // RMSNorm over the trailing `dh` axis: y_n [B, S, Hv, Dh] viewed as
        // [B*S*Hv, Dh] rows.
        let y_n_2d = y_n_for_tail.reshape(vec![batch * seq_len * hv, dh])?;
        let y_normed_2d = y_normed_n.reshape(vec![batch * seq_len * hv, dh])?;
        lib.encode_rms_norm(
            &enc,
            &y_n_2d,
            &norm_weight_in,
            cfg.rms_norm_eps,
            &y_normed_2d,
        )?;
        // silu(z) * y_normed; both [B, S, V_dim].
        let y_normed_flat = y_normed_n.reshape(vec![batch, seq_len, v_dim])?;
        lib.encode_silu_mul(&enc, &z_in, &y_normed_flat, &gated_n)?;
        // out_proj matmul: gated [B*S, V_dim] → out_y [B*S, hidden_size] F32.
        let gated_2d = gated_n.reshape(vec![batch * seq_len, v_dim])?;
        let out_y = out_y_n
            .as_ref()
            .expect("out_y_n is allocated whenever out_mxfp4 is Some");
        out_mxfp4_linear.encode_native(
            &enc,
            gated_2d.buffer(),
            gated_2d.offset_bytes() as u64,
            out_y.buffer(),
            out_y.offset_bytes() as u64,
            batch * seq_len,
        );
    }

    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    if seq_len > 0 {
        ssm_state.mark_populated();
    }

    // RMSNormGated + out_proj already ran on the GPU
    // inside the fused command buffer. Bridge the result back to Candle and
    // return — skipping the legacy tail entirely.
    if let Some(linear) = out_mxfp4 {
        let out_y = out_y_n.expect("out_y_n is allocated whenever out_mxfp4 is Some");
        let hidden_out = linear.out_features();
        return to_candle_tensor(&out_y, &device)?
            .reshape((batch, seq_len, hidden_out))
            .map_err(|e| anyhow!("{e}"));
    }

    // ── 6. RMSNormGated + out_proj on Candle (Dense / int8 fallback) ───
    // bf16-in path bridges via the f32 copy because `to_candle_tensor` does
    // not support BF16 yet (host roundtrip path is f32/u32-only).
    let y = to_candle_tensor(y_n_for_tail, &device)?
        .to_dtype(dtype)
        .map_err(|e| anyhow!("{e}"))?;
    let z = z_flat
        .reshape((batch, seq_len, hv, dh))
        .map_err(|e| anyhow!("{e}"))?;
    // bf16-in: candle_nn::ops::rms_norm requires input/weight dtype match.
    // Cast norm_weight on the fly (head_dim-sized — trivial). Same pattern as
    // B.3/B.4 in qwen3_5_moe::linear_attn.
    let norm_weight_chain = if norm_weight.dtype() == dtype {
        norm_weight.clone()
    } else {
        norm_weight.to_dtype(dtype).map_err(|e| anyhow!("{e}"))?
    };
    let y_normed = candle_nn::ops::rms_norm(
        &y.contiguous().map_err(|e| anyhow!("{e}"))?,
        &norm_weight_chain,
        cfg.rms_norm_eps,
    )
    .map_err(|e| anyhow!("{e}"))?;
    let gated_f32 = (candle_nn::ops::silu(&z.to_dtype(DType::F32).map_err(|e| anyhow!("{e}"))?)
        .map_err(|e| anyhow!("{e}"))?
        * y_normed
            .to_dtype(DType::F32)
            .map_err(|e| anyhow!("{e}"))?)
    .map_err(|e| anyhow!("{e}"))?;
    let gated = gated_f32.to_dtype(dtype).map_err(|e| anyhow!("{e}"))?;
    let out_flat = gated
        .reshape((batch, seq_len, v_dim))
        .map_err(|e| anyhow!("{e}"))?;
    // bf16-in fallback: Dense out_proj weights are f32 (dequantized). MXFP4 /
    // Affine4 paths handle their own dtype routing internally. Cast back to
    // f32 here so Dense matmul doesn't trip on the bf16 lhs. Same boundary
    // pattern as B.4 in qwen3_5_moe::linear_attn.
    let out_flat = if out_flat.dtype() != DType::F32 {
        out_flat.to_dtype(DType::F32).map_err(|e| anyhow!("{e}"))?
    } else {
        out_flat
    };
    out_proj
        .forward(&out_flat)
        .map_err(|e| anyhow!("out_proj: {e}"))
}

/// Lever D L.1.a body — encode the post-conv pipeline on Candle's command
/// queue rather than `NativeContext.queue`. Pre-allocates the MXFP4-fast-path
/// output as a Candle tensor for zero-copy return (no host roundtrip).
///
/// All `lib.encode_*` calls share an encoder borrowed from Candle's queue, so
/// prior Candle work and our writes are serialized by the GPU's queue order.
/// No explicit `wait_until_completed` is issued: subsequent Candle reads pick
/// up the result through Candle's own command-buffer cycle.
#[cfg(feature = "turboquant-gpu")]
#[allow(clippy::too_many_arguments)]
fn run_post_conv_fused_candle_queue(
    conv_out: &Tensor,
    z_flat: &Tensor,
    b_flat: &Tensor,
    a_flat: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    norm_weight: &Tensor,
    out_proj: &ProjLinear,
    cfg: &LinearAttnConfig,
    ctx: &NativeContext,
    lib: &KernelLib,
    ssm_state: &mut NativeSsmState,
    cached_dt_bias: Option<&NativeTensor>,
    cached_exp_a_log: Option<&NativeTensor>,
) -> Result<Tensor> {
    let (batch, seq_len, _) = conv_out.dims3().map_err(|e| anyhow!("{e}"))?;
    let device = conv_out.device().clone();
    let candle_md = match &device {
        candle_core::Device::Metal(md) => md,
        _ => {
            return Err(anyhow!(
                "forward_post_conv_fused_candle_queue: Metal device required"
            ))
        }
    };
    let dtype = conv_out.dtype();
    let v_dim = cfg.v_dim();
    let k_dim = cfg.k_dim();
    let hv = cfg.num_v_heads;
    let hk = cfg.num_k_heads;
    let dh = cfg.head_dim;
    // Workstream B Phase 6 — bf16-in mirror of `run_post_conv_fused`. See
    // that function's bf16 docs for the dtype contract.
    let bf16_in = dtype == DType::BF16;
    let qkv_native_dtype = if bf16_in {
        NativeDType::BF16
    } else {
        NativeDType::F32
    };

    // ── 3. split q / k / v + bridge to native (skip syncs — same queue) ────
    let q_blhd = if bf16_in {
        conv_out
            .narrow(D::Minus1, 0, k_dim)
            .and_then(|t| t.reshape((batch, seq_len, hk, dh)))
            .and_then(|t| t.contiguous())
            .map_err(|e| anyhow!("{e}"))?
    } else {
        conv_out
            .narrow(D::Minus1, 0, k_dim)
            .and_then(|t| t.reshape((batch, seq_len, hk, dh)))
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.contiguous())
            .map_err(|e| anyhow!("{e}"))?
    };
    let k_blhd = if bf16_in {
        conv_out
            .narrow(D::Minus1, k_dim, k_dim)
            .and_then(|t| t.reshape((batch, seq_len, hk, dh)))
            .and_then(|t| t.contiguous())
            .map_err(|e| anyhow!("{e}"))?
    } else {
        conv_out
            .narrow(D::Minus1, k_dim, k_dim)
            .and_then(|t| t.reshape((batch, seq_len, hk, dh)))
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.contiguous())
            .map_err(|e| anyhow!("{e}"))?
    };
    let v_blhd = if bf16_in {
        conv_out
            .narrow(D::Minus1, 2 * k_dim, v_dim)
            .and_then(|t| t.reshape((batch, seq_len, hv, dh)))
            .and_then(|t| t.contiguous())
            .map_err(|e| anyhow!("{e}"))?
    } else {
        conv_out
            .narrow(D::Minus1, 2 * k_dim, v_dim)
            .and_then(|t| t.reshape((batch, seq_len, hv, dh)))
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.contiguous())
            .map_err(|e| anyhow!("{e}"))?
    };
    let b_f32 = b_flat
        .to_dtype(DType::F32)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("{e}"))?;
    let a_f32 = a_flat
        .to_dtype(DType::F32)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("{e}"))?;
    let z_f32 = z_flat
        .to_dtype(DType::F32)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("z_flat to f32: {e}"))?;
    let norm_weight_f32 = norm_weight
        .to_dtype(DType::F32)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("norm_weight to f32: {e}"))?;
    let (exp_a_log_owned, dt_bias_f32_owned) =
        if cached_exp_a_log.is_some() && cached_dt_bias.is_some() {
            (None, None)
        } else {
            let e = a_log
                .to_dtype(DType::F32)
                .and_then(|t| t.exp())
                .and_then(|t| t.contiguous())
                .map_err(|e| anyhow!("{e}"))?;
            let d = dt_bias
                .to_dtype(DType::F32)
                .and_then(|t| t.contiguous())
                .map_err(|e| anyhow!("{e}"))?;
            (Some(e), Some(d))
        };

    // Same-queue lineage: every adoption uses `_no_sync` because we encode on
    // Candle's queue immediately after — no cross-queue boundary to drain.
    let q_in = from_candle_tensor_no_sync(ctx, &q_blhd)?;
    let k_in = from_candle_tensor_no_sync(ctx, &k_blhd)?;
    let v_in = from_candle_tensor_no_sync(ctx, &v_blhd)?;
    let b_in = from_candle_tensor_no_sync(ctx, &b_f32)?;
    let a_in = from_candle_tensor_no_sync(ctx, &a_f32)?;

    let dt_bias_owned_native;
    let exp_a_log_owned_native;
    let dt_bias_in: &NativeTensor = if let Some(c) = cached_dt_bias {
        c
    } else {
        dt_bias_owned_native = from_candle_tensor_no_sync(
            ctx,
            dt_bias_f32_owned.as_ref().expect("uncached dt_bias"),
        )?;
        &dt_bias_owned_native
    };
    let exp_a_log_in: &NativeTensor = if let Some(c) = cached_exp_a_log {
        c
    } else {
        exp_a_log_owned_native = from_candle_tensor_no_sync(
            ctx,
            exp_a_log_owned.as_ref().expect("uncached exp_a_log"),
        )?;
        &exp_a_log_owned_native
    };

    let z_in = from_candle_tensor_no_sync(ctx, &z_f32)?;
    let norm_weight_in = from_candle_tensor_no_sync(ctx, &norm_weight_f32)?;

    // ── 4. native scratch buffers (kept alive until encoder drops) ─────────
    let q_scaled = ctx.zeros(vec![batch, seq_len, hk, dh], qkv_native_dtype)?;
    let k_scaled = ctx.zeros(vec![batch, seq_len, hk, dh], qkv_native_dtype)?;
    let q_rep = ctx.zeros(vec![batch, seq_len, hv, dh], qkv_native_dtype)?;
    let k_rep = ctx.zeros(vec![batch, seq_len, hv, dh], qkv_native_dtype)?;
    let beta = ctx.zeros(vec![batch, seq_len, hv], NativeDType::F32)?;
    let a_plus_dt = ctx.zeros(vec![batch, seq_len, hv], NativeDType::F32)?;
    let softplus_a = ctx.zeros(vec![batch, seq_len, hv], NativeDType::F32)?;
    let g_pre = ctx.zeros(vec![batch, seq_len, hv], NativeDType::F32)?;
    let g = ctx.zeros(vec![batch, seq_len, hv], NativeDType::F32)?;
    let y_n = ctx.zeros(vec![batch, seq_len, hv, dh], qkv_native_dtype)?;
    let beta_bf = if bf16_in {
        Some(ctx.zeros(vec![batch, seq_len, hv], NativeDType::BF16)?)
    } else {
        None
    };
    let g_bf = if bf16_in {
        Some(ctx.zeros(vec![batch, seq_len, hv], NativeDType::BF16)?)
    } else {
        None
    };
    let y_n_f32 = if bf16_in {
        Some(ctx.zeros(vec![batch, seq_len, hv, dh], NativeDType::F32)?)
    } else {
        None
    };
    let y_normed_n = ctx.zeros(vec![batch, seq_len, hv, dh], NativeDType::F32)?;
    let gated_n = ctx.zeros(vec![batch, seq_len, v_dim], NativeDType::F32)?;

    // ── 4b. fast-path output: pre-allocate as Candle tensor ───────────────
    // Lever D delta vs `run_post_conv_fused`: instead of writing to a native
    // scratch buffer + `to_candle_tensor` host roundtrip, write directly into
    // a Candle-allocated buffer. After the encoder drops, the Candle tensor
    // already holds the result.
    //
    // also wire the fast-path for Affine4
    // out_proj on the same encoder. Saves the per-layer `wait_until_completed`
    // + 4-5 Candle dispatches (rms_norm + silu*x + matmul) for 27B Dense.
    // Default ON; opt-out via `LUMEN_AFFINE4_POST_CONV_FUSION=0`.
    let out_mxfp4 = out_proj.as_mxfp4();
    let affine4_fusion_enabled = std::env::var("LUMEN_AFFINE4_POST_CONV_FUSION")
        .map(|v| v != "0")
        .unwrap_or(true);
    let out_affine4 = if out_mxfp4.is_none() && affine4_fusion_enabled {
        out_proj.as_affine4()
    } else {
        None
    };
    let fused_out_features = out_mxfp4
        .map(|l| l.out_features())
        .or_else(|| out_affine4.map(|l| l.out_features()));
    let (out_y_candle, out_y_native) = if let Some(hidden_out) = fused_out_features {
        let t = Tensor::zeros((batch * seq_len, hidden_out), DType::F32, &device)
            .map_err(|e| anyhow!("out_y candle alloc: {e}"))?;
        let nt = from_candle_tensor_no_sync(ctx, &t)?;
        (Some(t), Some(nt))
    } else {
        (None, None)
    };

    // ── 5. encode on Candle's queue ───────────────────────────────────────
    let inv_scale = (dh as f64).powf(-0.5) as f32;
    let encoder = candle_md
        .command_encoder()
        .map_err(|e| anyhow!("candle command_encoder: {e}"))?;
    encoder.set_label("lumen:linear_attn_fused_candle_queue");

    if bf16_in {
        lib.encode_rms_norm_weightless_bf16(encoder.as_ref(), &q_in, cfg.ssm_eps, &q_scaled)?;
        lib.encode_rms_norm_weightless_bf16(encoder.as_ref(), &k_in, cfg.ssm_eps, &k_scaled)?;
        lib.encode_affine_scalar_bf16(encoder.as_ref(), &q_scaled, inv_scale * inv_scale, 0.0, &q_scaled)?;
        lib.encode_affine_scalar_bf16(encoder.as_ref(), &k_scaled, inv_scale, 0.0, &k_scaled)?;
    } else {
        lib.encode_rms_norm_weightless(encoder.as_ref(), &q_in, cfg.ssm_eps, &q_scaled)?;
        lib.encode_rms_norm_weightless(encoder.as_ref(), &k_in, cfg.ssm_eps, &k_scaled)?;
        lib.encode_affine_scalar(encoder.as_ref(), &q_scaled, inv_scale * inv_scale, 0.0, &q_scaled)?;
        lib.encode_affine_scalar(encoder.as_ref(), &k_scaled, inv_scale, 0.0, &k_scaled)?;
    }
    lib.encode_sigmoid(encoder.as_ref(), &b_in, &beta)?;
    lib.encode_broadcast_add_per_head(encoder.as_ref(), &a_in, dt_bias_in, &a_plus_dt)?;
    lib.encode_softplus(encoder.as_ref(), &a_plus_dt, &softplus_a)?;
    lib.encode_mul_broadcast_per_head(encoder.as_ref(), &softplus_a, exp_a_log_in, &g_pre)?;
    lib.encode_neg_exp(encoder.as_ref(), &g_pre, &g)?;

    let repeats = hv / hk;
    if repeats == 1 {
        if bf16_in {
            lib.encode_affine_scalar_bf16(encoder.as_ref(), &q_scaled, 1.0, 0.0, &q_rep)?;
            lib.encode_affine_scalar_bf16(encoder.as_ref(), &k_scaled, 1.0, 0.0, &k_rep)?;
        } else {
            lib.encode_affine_scalar(encoder.as_ref(), &q_scaled, 1.0, 0.0, &q_rep)?;
            lib.encode_affine_scalar(encoder.as_ref(), &k_scaled, 1.0, 0.0, &k_rep)?;
        }
    } else if bf16_in {
        lib.encode_repeat_heads_blhd_bf16(encoder.as_ref(), &q_scaled, &q_rep, repeats)?;
        lib.encode_repeat_heads_blhd_bf16(encoder.as_ref(), &k_scaled, &k_rep, repeats)?;
    } else {
        lib.encode_repeat_heads_blhd(encoder.as_ref(), &q_scaled, &q_rep, repeats)?;
        lib.encode_repeat_heads_blhd(encoder.as_ref(), &k_scaled, &k_rep, repeats)?;
    }

    if bf16_in {
        lib.encode_cast_f32_to_bf16(encoder.as_ref(), &beta, beta_bf.as_ref().unwrap())?;
        lib.encode_cast_f32_to_bf16(encoder.as_ref(), &g, g_bf.as_ref().unwrap())?;
    }

    let elem = qkv_native_dtype.size_in_bytes();
    let qk_step = hv * dh * elem;
    let v_step = hv * dh * elem;
    let scalar_step_f32 = hv * 4;
    let scalar_step_bf16 = hv * 2;
    for t in 0..seq_len {
        let q_t = q_rep.slice(q_rep.offset_bytes() + t * qk_step, vec![batch, hv, dh])?;
        let k_t = k_rep.slice(k_rep.offset_bytes() + t * qk_step, vec![batch, hv, dh])?;
        let v_t = v_in.slice(v_in.offset_bytes() + t * v_step, vec![batch, hv, dh])?;
        let y_t = y_n.slice(y_n.offset_bytes() + t * v_step, vec![batch, hv, dh])?;
        if bf16_in {
            let beta_b = beta_bf.as_ref().unwrap();
            let g_b = g_bf.as_ref().unwrap();
            let beta_t =
                beta_b.slice(beta_b.offset_bytes() + t * scalar_step_bf16, vec![batch, hv])?;
            let g_t = g_b.slice(g_b.offset_bytes() + t * scalar_step_bf16, vec![batch, hv])?;
            lib.encode_ssm_step_bf16(
                encoder.as_ref(),
                ssm_state.buf(),
                &q_t,
                &k_t,
                &v_t,
                &beta_t,
                &g_t,
                &y_t,
            )?;
        } else {
            let beta_t = beta.slice(beta.offset_bytes() + t * scalar_step_f32, vec![batch, hv])?;
            let g_t = g.slice(g.offset_bytes() + t * scalar_step_f32, vec![batch, hv])?;
            lib.encode_ssm_step(
                encoder.as_ref(),
                ssm_state.buf(),
                &q_t,
                &k_t,
                &v_t,
                &beta_t,
                &g_t,
                &y_t,
            )?;
        }
    }

    // bf16-in: cast y_n bf16 → y_n_f32 so the f32 RMSNormGated tail (and the
    // Dense fallback host bridge) reads a familiar dtype. f32 path: no-op.
    let y_n_for_tail: &NativeTensor = if bf16_in {
        let y_f32 = y_n_f32.as_ref().unwrap();
        lib.encode_cast_bf16_to_f32(encoder.as_ref(), &y_n, y_f32)?;
        y_f32
    } else {
        &y_n
    };

    // Fused tail: rms_norm + silu*x + out_proj matmul, encoded onto the same
    // encoder when out_proj is MXFP4 OR Affine4. For Dense Affine4 (27B Dense)
    // this skips the per-layer `wait_until_completed` + 4-5 Candle dispatch
    // chain that the legacy fallback below pays.
    if out_mxfp4.is_some() || out_affine4.is_some() {
        let y_n_2d = y_n_for_tail.reshape(vec![batch * seq_len * hv, dh])?;
        let y_normed_2d = y_normed_n.reshape(vec![batch * seq_len * hv, dh])?;
        lib.encode_rms_norm(
            encoder.as_ref(),
            &y_n_2d,
            &norm_weight_in,
            cfg.rms_norm_eps,
            &y_normed_2d,
        )?;
        let y_normed_flat = y_normed_n.reshape(vec![batch, seq_len, v_dim])?;
        lib.encode_silu_mul(encoder.as_ref(), &z_in, &y_normed_flat, &gated_n)?;
        let gated_2d = gated_n.reshape(vec![batch * seq_len, v_dim])?;
        let out_y = out_y_native
            .as_ref()
            .expect("out_y_native is allocated for both mxfp4 and affine4 fused tails");
        if let Some(out_mxfp4_linear) = out_mxfp4 {
            out_mxfp4_linear.encode_native(
                encoder.as_ref(),
                gated_2d.buffer(),
                gated_2d.offset_bytes() as u64,
                out_y.buffer(),
                out_y.offset_bytes() as u64,
                batch * seq_len,
            );
        } else if let Some(out_affine4_linear) = out_affine4 {
            out_affine4_linear.encode_native(
                encoder.as_ref(),
                gated_2d.buffer(),
                gated_2d.offset_bytes() as u64,
                out_y.buffer(),
                out_y.offset_bytes() as u64,
                batch * seq_len,
            );
        }
    }

    // Drop the encoder to end encoding. Candle's command buffer commits
    // through its normal cycle; downstream Candle ops read the output via
    // implicit queue ordering.
    drop(encoder);

    if seq_len > 0 {
        ssm_state.mark_populated();
    }

    // Fast path (MXFP4 or Affine4 fused): return the pre-allocated Candle
    // tensor directly. Skips the per-layer `wait_until_completed` below.
    if let Some(hidden_out) = fused_out_features {
        let out_y_candle =
            out_y_candle.expect("out_y_candle is allocated when fused_out_features is Some");
        return out_y_candle
            .reshape((batch, seq_len, hidden_out))
            .map_err(|e| anyhow!("{e}"));
    }

    // Dense / int8 fallback: y_n still requires host roundtrip. This is the
    // legacy slow path (rarely taken in production, where out_proj is MXFP4).
    // Drain Candle's command queue first so the encoder's writes to `y_n` are
    // visible to host. The MXFP4 fast path above doesn't need this because
    // the output stays on-device — Candle queue ordering handles it.
    candle_md
        .wait_until_completed()
        .map_err(|e| anyhow!("candle wait: {e}"))?;
    // bf16-in path bridges via `y_n_for_tail` (the f32 copy) since
    // `to_candle_tensor` doesn't support BF16.
    let y = to_candle_tensor(y_n_for_tail, &device)?
        .to_dtype(dtype)
        .map_err(|e| anyhow!("{e}"))?;
    let z = z_flat
        .reshape((batch, seq_len, hv, dh))
        .map_err(|e| anyhow!("{e}"))?;
    let norm_weight_chain = if norm_weight.dtype() == dtype {
        norm_weight.clone()
    } else {
        norm_weight.to_dtype(dtype).map_err(|e| anyhow!("{e}"))?
    };
    let y_normed = candle_nn::ops::rms_norm(
        &y.contiguous().map_err(|e| anyhow!("{e}"))?,
        &norm_weight_chain,
        cfg.rms_norm_eps,
    )
    .map_err(|e| anyhow!("{e}"))?;
    let gated_f32 = (candle_nn::ops::silu(&z.to_dtype(DType::F32).map_err(|e| anyhow!("{e}"))?)
        .map_err(|e| anyhow!("{e}"))?
        * y_normed
            .to_dtype(DType::F32)
            .map_err(|e| anyhow!("{e}"))?)
    .map_err(|e| anyhow!("{e}"))?;
    let gated = gated_f32.to_dtype(dtype).map_err(|e| anyhow!("{e}"))?;
    let out_flat = gated
        .reshape((batch, seq_len, v_dim))
        .map_err(|e| anyhow!("{e}"))?;
    let out_flat = if out_flat.dtype() != DType::F32 {
        out_flat.to_dtype(DType::F32).map_err(|e| anyhow!("{e}"))?
    } else {
        out_flat
    };
    out_proj
        .forward(&out_flat)
        .map_err(|e| anyhow!("out_proj: {e}"))
}

/// `softplus(x) = ln(1 + exp(x))`, stable: `max(x, 0) + ln(1 + exp(-|x|))`.
fn stable_softplus(x: &Tensor) -> CandleResult<Tensor> {
    let zero = x.zeros_like()?;
    let pos = x.maximum(&zero)?;
    let abs_x = x.abs()?;
    let log1p_exp = (abs_x.neg()?.exp()? + 1.0)?.log()?;
    pos + log1p_exp
}

/// Repeat each head `repeats` times along axis 2 of a `[B, S, H, D]` tensor
/// (consecutive duplication, matching `mx.repeat(x, repeats, axis=-2)` /
/// `repeat_interleave`).
fn repeat_heads(x: &Tensor, repeats: usize) -> CandleResult<Tensor> {
    if repeats == 1 {
        return Ok(x.clone());
    }
    let (b, s, h, d) = x.dims4()?;
    x.unsqueeze(3)?
        .expand((b, s, h, repeats, d))?
        .reshape((b, s, h * repeats, d))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bit-for-bit reference for `forward_ssm_loop`. Mirrors the loop body in
    /// `qwen3_5_moe::linear_attn::GatedDeltaNet::forward`.
    #[allow(clippy::too_many_arguments)]
    fn cpu_ssm_loop(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        beta: &[f32],
        g: &[f32],
        s: usize,
        hv: usize,
        dv: usize,
        dk: usize,
    ) -> Vec<f32> {
        let mut state = vec![0.0_f32; hv * dv * dk];
        let mut y_all = Vec::with_capacity(s * hv * dv);
        for t in 0..s {
            let q_t = &q[t * hv * dk..(t + 1) * hv * dk];
            let k_t = &k[t * hv * dk..(t + 1) * hv * dk];
            let v_t = &v[t * hv * dv..(t + 1) * hv * dv];
            let beta_t = &beta[t * hv..(t + 1) * hv];
            let g_t = &g[t * hv..(t + 1) * hv];

            let mut y_t = vec![0.0_f32; hv * dv];
            for hi in 0..hv {
                let beta_v = beta_t[hi];
                let g_v = g_t[hi];
                for di in 0..dv {
                    let st_off = (hi * dv + di) * dk;

                    let mut kv_mem = 0.0_f32;
                    for j in 0..dk {
                        let st = state[st_off + j] * g_v;
                        state[st_off + j] = st;
                        kv_mem += st * k_t[hi * dk + j];
                    }
                    let delta = (v_t[hi * dv + di] - kv_mem) * beta_v;

                    let mut y_val = 0.0_f32;
                    for j in 0..dk {
                        let st = state[st_off + j] + k_t[hi * dk + j] * delta;
                        state[st_off + j] = st;
                        y_val += st * q_t[hi * dk + j];
                    }
                    y_t[hi * dv + di] = y_val;
                }
            }
            y_all.extend(y_t);
        }
        y_all
    }

    #[test]
    fn forward_ssm_loop_matches_cpu_reference() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();

        let s = 6;
        let hv = 4;
        let dv = 8;
        let dk = 16;

        let make = |seed: u32, n: usize| -> Vec<f32> {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    (((s >> 8) & 0xFFFF) as f32 / 65535.0 - 0.5) * 0.8
                })
                .collect()
        };
        let q_v = make(0x01, s * hv * dk);
        let k_v = make(0x02, s * hv * dk);
        let v_v = make(0x03, s * hv * dv);
        let beta_v: Vec<f32> = (0..s * hv).map(|i| 0.4 + ((i % 7) as f32) * 0.04).collect();
        let g_v: Vec<f32> = (0..s * hv)
            .map(|i| 0.92 - ((i % 5) as f32) * 0.01)
            .collect();

        let q_t = ctx.from_slice_f32(&q_v, vec![1, s, hv, dk]).unwrap();
        let k_t = ctx.from_slice_f32(&k_v, vec![1, s, hv, dk]).unwrap();
        let v_t = ctx.from_slice_f32(&v_v, vec![1, s, hv, dv]).unwrap();
        let beta_t = ctx.from_slice_f32(&beta_v, vec![1, s, hv]).unwrap();
        let g_t = ctx.from_slice_f32(&g_v, vec![1, s, hv]).unwrap();
        let y_t = ctx.zeros(vec![1, s, hv, dv], NativeDType::F32).unwrap();

        forward_ssm_loop(&ctx, &lib, &q_t, &k_t, &v_t, &beta_t, &g_t, &y_t).unwrap();
        let got = y_t.to_vec_f32().unwrap();
        let expected = cpu_ssm_loop(&q_v, &k_v, &v_v, &beta_v, &g_v, s, hv, dv, dk);

        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            let err = (g - e).abs();
            let rel = err / e.abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "ssm_loop idx={i} got={g} expected={e} rel_err={rel}"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // forward_linear_attn parity vs Candle-only reference
    // ─────────────────────────────────────────────────────────────────

    use candle_core::Device;
    use candle_nn::{Conv1d, Conv1dConfig, Linear};

    fn lcg_vec(seed: u32, n: usize, scale: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (((s >> 8) & 0xFFFF) as f32 / 65535.0 - 0.5) * scale
            })
            .collect()
    }

    fn dense(out_f: usize, in_f: usize, seed: u32, device: &Device) -> ProjLinear {
        let w = lcg_vec(seed, out_f * in_f, 0.4);
        let weight = Tensor::from_vec(w, (out_f, in_f), device).unwrap();
        ProjLinear::Dense(Linear::new(weight, None))
    }

    /// Stateless Candle-only reference. Mirrors the cold-state prefill path of
    /// `qwen3_5_moe::linear_attn::GatedDeltaNet::forward` exactly, expressed as
    /// a free function so we can compare without holding `&mut self` state.
    #[allow(clippy::too_many_arguments)]
    fn candle_linear_attn_reference(
        hidden: &Tensor,
        in_proj_combined: &ProjLinear,
        conv1d: &Conv1d,
        a_log: &Tensor,
        dt_bias: &Tensor,
        norm_weight: &Tensor,
        out_proj: &ProjLinear,
        cfg: &LinearAttnConfig,
    ) -> Tensor {
        let (batch, seq_len, _) = hidden.dims3().unwrap();
        let device = hidden.device().clone();
        let dtype = hidden.dtype();
        let qkv_dim = cfg.qkv_dim();
        let v_dim = cfg.v_dim();
        let k_dim = cfg.k_dim();
        let hv = cfg.num_v_heads;
        let hk = cfg.num_k_heads;
        let dh = cfg.head_dim;

        let combined = in_proj_combined.forward(hidden).unwrap();
        let last = combined.dims().len() - 1;
        let qkv_flat = combined.narrow(last, 0, qkv_dim).unwrap().contiguous().unwrap();
        let z_flat = combined
            .narrow(last, qkv_dim, v_dim)
            .unwrap()
            .contiguous()
            .unwrap();
        let b_flat = combined
            .narrow(last, qkv_dim + v_dim, hv)
            .unwrap()
            .contiguous()
            .unwrap();
        let a_flat = combined
            .narrow(last, qkv_dim + v_dim + hv, hv)
            .unwrap()
            .contiguous()
            .unwrap();

        let conv_pad = cfg.conv_kernel - 1;
        let prev = Tensor::zeros((batch, conv_pad, qkv_dim), dtype, &device).unwrap();
        let conv_input = Tensor::cat(&[&prev, &qkv_flat], 1).unwrap();
        let mut slices = Vec::with_capacity(cfg.conv_kernel);
        for k in 0..cfg.conv_kernel {
            slices.push(conv_input.narrow(1, k, seq_len).unwrap());
        }
        let windowed = Tensor::stack(&slices, 2).unwrap();
        let w = conv1d
            .weight()
            .squeeze(1)
            .unwrap()
            .transpose(0, 1)
            .unwrap()
            .contiguous()
            .unwrap();
        let conv_out = windowed
            .broadcast_mul(&w.unsqueeze(0).unwrap().unsqueeze(0).unwrap())
            .unwrap()
            .sum(2)
            .unwrap();
        let conv_out = candle_nn::ops::silu(&conv_out).unwrap();

        let q = conv_out
            .narrow(D::Minus1, 0, k_dim)
            .unwrap()
            .reshape((batch, seq_len, hk, dh))
            .unwrap();
        let k = conv_out
            .narrow(D::Minus1, k_dim, k_dim)
            .unwrap()
            .reshape((batch, seq_len, hk, dh))
            .unwrap();
        let v = conv_out
            .narrow(D::Minus1, 2 * k_dim, v_dim)
            .unwrap()
            .reshape((batch, seq_len, hv, dh))
            .unwrap();

        let inv_scale = (dh as f64).powf(-0.5);
        let manual_rms = |x: &Tensor| -> Tensor {
            let xf = x.to_dtype(DType::F32).unwrap();
            let ms = xf.sqr().unwrap().mean_keepdim(D::Minus1).unwrap();
            let scale = (ms + cfg.ssm_eps as f64)
                .unwrap()
                .sqrt()
                .unwrap()
                .recip()
                .unwrap();
            xf.broadcast_mul(&scale).unwrap().to_dtype(dtype).unwrap()
        };
        let q = manual_rms(&q).affine(inv_scale * inv_scale, 0.0).unwrap();
        let k = manual_rms(&k).affine(inv_scale, 0.0).unwrap();

        let beta = candle_nn::ops::sigmoid(&b_flat).unwrap();
        let a_plus_dt = a_flat
            .broadcast_add(&dt_bias.reshape((1, 1, hv)).unwrap())
            .unwrap();
        let zero = a_plus_dt.zeros_like().unwrap();
        let pos = a_plus_dt.maximum(&zero).unwrap();
        let abs_x = a_plus_dt.abs().unwrap();
        let log1p_exp = ((abs_x.neg().unwrap().exp().unwrap() + 1.0).unwrap())
            .log()
            .unwrap();
        let softplus_a = (pos + log1p_exp).unwrap();
        let a_log_f32 = a_log.to_dtype(DType::F32).unwrap().exp().unwrap();
        let g = softplus_a
            .to_dtype(DType::F32)
            .unwrap()
            .broadcast_mul(&a_log_f32.reshape((1, 1, hv)).unwrap())
            .unwrap()
            .neg()
            .unwrap()
            .exp()
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        let repeats = hv / hk;
        let q_rep = if repeats == 1 {
            q
        } else {
            q.unsqueeze(3)
                .unwrap()
                .expand((batch, seq_len, hk, repeats, dh))
                .unwrap()
                .reshape((batch, seq_len, hk * repeats, dh))
                .unwrap()
        };
        let k_rep = if repeats == 1 {
            k
        } else {
            k.unsqueeze(3)
                .unwrap()
                .expand((batch, seq_len, hk, repeats, dh))
                .unwrap()
                .reshape((batch, seq_len, hk * repeats, dh))
                .unwrap()
        };

        let q_f32 = q_rep.to_dtype(DType::F32).unwrap().contiguous().unwrap();
        let k_f32 = k_rep.to_dtype(DType::F32).unwrap().contiguous().unwrap();
        let v_f32 = v.to_dtype(DType::F32).unwrap().contiguous().unwrap();
        let g_f32 = g.to_dtype(DType::F32).unwrap().contiguous().unwrap();
        let beta_f32 = beta.to_dtype(DType::F32).unwrap().contiguous().unwrap();

        let mut state = Tensor::zeros((batch, hv, dh, dh), DType::F32, &device).unwrap();
        let mut y_steps: Vec<Tensor> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let q_t = q_f32.narrow(1, t, 1).unwrap().squeeze(1).unwrap();
            let k_t = k_f32.narrow(1, t, 1).unwrap().squeeze(1).unwrap();
            let v_t = v_f32.narrow(1, t, 1).unwrap().squeeze(1).unwrap();
            let g_t = g_f32.narrow(1, t, 1).unwrap().squeeze(1).unwrap();
            let beta_t = beta_f32.narrow(1, t, 1).unwrap().squeeze(1).unwrap();
            let decay = g_t
                .unsqueeze(D::Minus1)
                .unwrap()
                .unsqueeze(D::Minus1)
                .unwrap();
            state = state.broadcast_mul(&decay).unwrap();
            let k_bc = k_t.unsqueeze(D::Minus2).unwrap();
            let kv_mem = state.broadcast_mul(&k_bc).unwrap().sum(D::Minus1).unwrap();
            let delta = (v_t - kv_mem)
                .unwrap()
                .broadcast_mul(&beta_t.unsqueeze(D::Minus1).unwrap())
                .unwrap();
            let outer = k_bc.broadcast_mul(&delta.unsqueeze(D::Minus1).unwrap()).unwrap();
            state = (state + outer).unwrap();
            let q_bc = q_t.unsqueeze(D::Minus2).unwrap();
            let y_t = state.broadcast_mul(&q_bc).unwrap().sum(D::Minus1).unwrap();
            y_steps.push(y_t);
        }
        let y = Tensor::stack(&y_steps, 1).unwrap().to_dtype(dtype).unwrap();

        let z = z_flat.reshape((batch, seq_len, hv, dh)).unwrap();
        let y_normed = candle_nn::ops::rms_norm(
            &y.contiguous().unwrap(),
            norm_weight,
            cfg.rms_norm_eps,
        )
        .unwrap();
        let gated_f32 = (candle_nn::ops::silu(&z.to_dtype(DType::F32).unwrap()).unwrap()
            * y_normed.to_dtype(DType::F32).unwrap())
        .unwrap();
        let gated = gated_f32.to_dtype(dtype).unwrap();
        let out_flat = gated.reshape((batch, seq_len, v_dim)).unwrap();
        out_proj.forward(&out_flat).unwrap()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (*x as f64) * (*y as f64))
            .sum();
        let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        (dot / (na * nb)) as f32
    }

    #[test]
    fn forward_linear_attn_matches_candle_reference() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();
        let device = Device::Cpu;

        let cfg = LinearAttnConfig {
            hidden_size: 16,
            num_k_heads: 2,
            num_v_heads: 4,
            head_dim: 8,
            conv_kernel: 4,
            rms_norm_eps: 1e-6,
            ssm_eps: 1e-6,
        };
        let batch = 1;
        let seq_len = 5;

        let hidden_v = lcg_vec(0xA1, batch * seq_len * cfg.hidden_size, 1.5);
        let hidden = Tensor::from_vec(hidden_v, (batch, seq_len, cfg.hidden_size), &device).unwrap();

        let combined_out = cfg.qkv_dim() + cfg.v_dim() + 2 * cfg.num_v_heads;
        let in_proj = dense(combined_out, cfg.hidden_size, 0xB2, &device);

        // depthwise conv1d: weight [channels, 1, kernel]
        let conv_w = lcg_vec(0xC3, cfg.qkv_dim() * cfg.conv_kernel, 0.5);
        let conv_w_t =
            Tensor::from_vec(conv_w, (cfg.qkv_dim(), 1, cfg.conv_kernel), &device).unwrap();
        let conv_cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: cfg.qkv_dim(),
            cudnn_fwd_algo: None,
        };
        let conv1d = Conv1d::new(conv_w_t, None, conv_cfg);

        let a_log_v = lcg_vec(0xD4, cfg.num_v_heads, 0.3);
        let a_log = Tensor::from_vec(a_log_v, (cfg.num_v_heads,), &device).unwrap();
        let dt_bias_v = lcg_vec(0xE5, cfg.num_v_heads, 0.2);
        let dt_bias = Tensor::from_vec(dt_bias_v, (cfg.num_v_heads,), &device).unwrap();
        let norm_w_v: Vec<f32> = (0..cfg.head_dim)
            .map(|i| 1.0 + (i as f32) * 0.02)
            .collect();
        let norm_weight = Tensor::from_vec(norm_w_v, (cfg.head_dim,), &device).unwrap();
        let out_proj = dense(cfg.hidden_size, cfg.v_dim(), 0xF6, &device);

        let got_t = forward_linear_attn(
            &hidden,
            &in_proj,
            &conv1d,
            &a_log,
            &dt_bias,
            &norm_weight,
            &out_proj,
            &cfg,
            &ctx,
            &lib,
        )
        .unwrap();
        let exp_t = candle_linear_attn_reference(
            &hidden,
            &in_proj,
            &conv1d,
            &a_log,
            &dt_bias,
            &norm_weight,
            &out_proj,
            &cfg,
        );

        assert_eq!(got_t.dims(), exp_t.dims());
        let got = got_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = exp_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let cos = cosine(&got, &expected);
        let max_mag = got
            .iter()
            .chain(expected.iter())
            .map(|v| v.abs())
            .fold(0.0_f32, f32::max);
        let max_abs = got
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let rel = max_abs / max_mag.max(1e-6);
        assert!(
            cos > 0.999,
            "linear_attn cosine {cos} (rel_max={rel}, abs_max={max_abs})"
        );
    }

    #[test]
    fn forward_linear_attn_rejects_b_gt_1() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();
        let device = Device::Cpu;

        let cfg = LinearAttnConfig {
            hidden_size: 8,
            num_k_heads: 2,
            num_v_heads: 4,
            head_dim: 4,
            conv_kernel: 2,
            rms_norm_eps: 1e-6,
            ssm_eps: 1e-6,
        };

        let hidden = Tensor::zeros((2, 3, cfg.hidden_size), DType::F32, &device).unwrap();
        let combined_out = cfg.qkv_dim() + cfg.v_dim() + 2 * cfg.num_v_heads;
        let in_proj = dense(combined_out, cfg.hidden_size, 1, &device);
        let conv_w = Tensor::zeros((cfg.qkv_dim(), 1, cfg.conv_kernel), DType::F32, &device).unwrap();
        let conv_cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: cfg.qkv_dim(),
            cudnn_fwd_algo: None,
        };
        let conv1d = Conv1d::new(conv_w, None, conv_cfg);
        let a_log = Tensor::zeros((cfg.num_v_heads,), DType::F32, &device).unwrap();
        let dt_bias = Tensor::zeros((cfg.num_v_heads,), DType::F32, &device).unwrap();
        let norm_weight = Tensor::ones((cfg.head_dim,), DType::F32, &device).unwrap();
        let out_proj = dense(cfg.hidden_size, cfg.v_dim(), 2, &device);

        let r = forward_linear_attn(
            &hidden,
            &in_proj,
            &conv1d,
            &a_log,
            &dt_bias,
            &norm_weight,
            &out_proj,
            &cfg,
            &ctx,
            &lib,
        );
        assert!(r.is_err());
        assert!(format!("{}", r.unwrap_err()).contains("B=1"));
    }

    #[test]
    fn forward_ssm_loop_rejects_b_gt_1() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();

        let s = 2;
        let hv = 2;
        let dv = 2;
        let dk = 2;
        let n_qk = 2 * s * hv * dk;
        let n_v = 2 * s * hv * dv;
        let n_scalar = 2 * s * hv;

        let q = ctx
            .from_slice_f32(&vec![0.0; n_qk], vec![2, s, hv, dk])
            .unwrap();
        let k = ctx
            .from_slice_f32(&vec![0.0; n_qk], vec![2, s, hv, dk])
            .unwrap();
        let v = ctx
            .from_slice_f32(&vec![0.0; n_v], vec![2, s, hv, dv])
            .unwrap();
        let beta = ctx
            .from_slice_f32(&vec![0.0; n_scalar], vec![2, s, hv])
            .unwrap();
        let g = ctx
            .from_slice_f32(&vec![0.0; n_scalar], vec![2, s, hv])
            .unwrap();
        let y = ctx.zeros(vec![2, s, hv, dv], NativeDType::F32).unwrap();
        let r = forward_ssm_loop(&ctx, &lib, &q, &k, &v, &beta, &g, &y);
        assert!(r.is_err());
        assert!(format!("{}", r.unwrap_err()).contains("B=1"));
    }

    // ─── forward_linear_attn_fused (A.8-C.4) parity ────────────────────────

    fn build_linear_attn_fixture(
        device: &Device,
    ) -> (
        LinearAttnConfig,
        Tensor,
        ProjLinear,
        Conv1d,
        Tensor,
        Tensor,
        Tensor,
        ProjLinear,
    ) {
        let cfg = LinearAttnConfig {
            hidden_size: 16,
            num_k_heads: 2,
            num_v_heads: 4,
            head_dim: 8,
            conv_kernel: 4,
            rms_norm_eps: 1e-6,
            ssm_eps: 1e-6,
        };
        let batch = 1;
        let seq_len = 5;

        let hidden_v = lcg_vec(0xA1, batch * seq_len * cfg.hidden_size, 1.5);
        let hidden = Tensor::from_vec(hidden_v, (batch, seq_len, cfg.hidden_size), device).unwrap();
        let combined_out = cfg.qkv_dim() + cfg.v_dim() + 2 * cfg.num_v_heads;
        let in_proj = dense(combined_out, cfg.hidden_size, 0xB2, device);
        let conv_w = lcg_vec(0xC3, cfg.qkv_dim() * cfg.conv_kernel, 0.5);
        let conv_w_t =
            Tensor::from_vec(conv_w, (cfg.qkv_dim(), 1, cfg.conv_kernel), device).unwrap();
        let conv_cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: cfg.qkv_dim(),
            cudnn_fwd_algo: None,
        };
        let conv1d = Conv1d::new(conv_w_t, None, conv_cfg);
        let a_log = Tensor::from_vec(lcg_vec(0xD4, cfg.num_v_heads, 0.3), (cfg.num_v_heads,), device)
            .unwrap();
        let dt_bias = Tensor::from_vec(
            lcg_vec(0xE5, cfg.num_v_heads, 0.2),
            (cfg.num_v_heads,),
            device,
        )
        .unwrap();
        let norm_w_v: Vec<f32> = (0..cfg.head_dim).map(|i| 1.0 + (i as f32) * 0.02).collect();
        let norm_weight = Tensor::from_vec(norm_w_v, (cfg.head_dim,), device).unwrap();
        let out_proj = dense(cfg.hidden_size, cfg.v_dim(), 0xF6, device);
        (
            cfg,
            hidden,
            in_proj,
            conv1d,
            a_log,
            dt_bias,
            norm_weight,
            out_proj,
        )
    }

    #[test]
    fn forward_linear_attn_fused_matches_unfused_cold_state() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();
        let device = Device::Cpu;

        let (cfg, hidden, in_proj, conv1d, a_log, dt_bias, norm_weight, out_proj) =
            build_linear_attn_fixture(&device);

        let unfused = forward_linear_attn(
            &hidden,
            &in_proj,
            &conv1d,
            &a_log,
            &dt_bias,
            &norm_weight,
            &out_proj,
            &cfg,
            &ctx,
            &lib,
        )
        .unwrap();

        let mut state =
            NativeSsmState::new(&ctx, 1, cfg.num_v_heads, cfg.head_dim, cfg.head_dim).unwrap();
        let fused = forward_linear_attn_fused(
            &hidden,
            &in_proj,
            &conv1d,
            &a_log,
            &dt_bias,
            &norm_weight,
            &out_proj,
            &cfg,
            &ctx,
            &lib,
            &mut state,
        )
        .unwrap();

        assert_eq!(unfused.dims(), fused.dims());
        let exp = unfused.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let got = fused.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let cos = cosine(&got, &exp);
        let max_abs = got
            .iter()
            .zip(exp.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let max_mag = got
            .iter()
            .chain(exp.iter())
            .map(|v| v.abs())
            .fold(0.0_f32, f32::max);
        let rel = max_abs / max_mag.max(1e-6);
        // Operations differ in order (single fused buffer vs many committed
        // ones), so we accept a small numerical drift.
        assert!(
            cos > 0.9999 && rel < 1e-3,
            "fused vs unfused cos={cos} rel_max={rel} abs_max={max_abs}"
        );
        assert!(state.is_populated());
    }

    /// Lever D L.1.b — parity check: candle-queue variant must produce
    /// numerically equivalent output to the native-queue legacy variant for the
    /// same inputs. Both variants encode the same kernels in the same order, so
    /// drift should be at most a few ULPs. Skips when no Metal device is
    /// available (CI / non-macOS).
    #[cfg(feature = "turboquant-gpu")]
    #[test]
    fn forward_post_conv_fused_candle_queue_matches_legacy() {
        let device = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let ctx = match NativeContext::from_candle_device(&device) {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();

        let cfg = LinearAttnConfig {
            hidden_size: 16,
            num_k_heads: 2,
            num_v_heads: 4,
            head_dim: 8,
            conv_kernel: 4,
            rms_norm_eps: 1e-6,
            ssm_eps: 1e-6,
        };
        let batch = 1;
        let seq_len = 5;
        let qkv_dim = cfg.qkv_dim();
        let v_dim = cfg.v_dim();
        let hv = cfg.num_v_heads;

        let conv_out = Tensor::from_vec(
            lcg_vec(0xA1, batch * seq_len * qkv_dim, 0.5),
            (batch, seq_len, qkv_dim),
            &device,
        )
        .unwrap();
        let z_flat = Tensor::from_vec(
            lcg_vec(0xB2, batch * seq_len * v_dim, 0.5),
            (batch, seq_len, v_dim),
            &device,
        )
        .unwrap();
        let b_flat = Tensor::from_vec(
            lcg_vec(0xC3, batch * seq_len * hv, 0.5),
            (batch, seq_len, hv),
            &device,
        )
        .unwrap();
        let a_flat = Tensor::from_vec(
            lcg_vec(0xD4, batch * seq_len * hv, 0.5),
            (batch, seq_len, hv),
            &device,
        )
        .unwrap();
        let a_log =
            Tensor::from_vec(lcg_vec(0xE5, hv, 0.3), (hv,), &device).unwrap();
        let dt_bias =
            Tensor::from_vec(lcg_vec(0xF6, hv, 0.2), (hv,), &device).unwrap();
        let norm_w_v: Vec<f32> = (0..cfg.head_dim).map(|i| 1.0 + (i as f32) * 0.02).collect();
        let norm_weight = Tensor::from_vec(norm_w_v, (cfg.head_dim,), &device).unwrap();
        let out_proj = dense(cfg.hidden_size, v_dim, 0xA7, &device);

        let mut state_legacy =
            NativeSsmState::new(&ctx, batch, hv, cfg.head_dim, cfg.head_dim).unwrap();
        let legacy = forward_post_conv_fused_with_cache(
            &conv_out,
            &z_flat,
            &b_flat,
            &a_flat,
            &a_log,
            &dt_bias,
            &norm_weight,
            &out_proj,
            &cfg,
            &ctx,
            &lib,
            &mut state_legacy,
            None,
            None,
        )
        .unwrap();

        let mut state_cq =
            NativeSsmState::new(&ctx, batch, hv, cfg.head_dim, cfg.head_dim).unwrap();
        let candle_queue = forward_post_conv_fused_with_cache_candle_queue(
            &conv_out,
            &z_flat,
            &b_flat,
            &a_flat,
            &a_log,
            &dt_bias,
            &norm_weight,
            &out_proj,
            &cfg,
            &ctx,
            &lib,
            &mut state_cq,
            None,
            None,
        )
        .unwrap();

        assert_eq!(legacy.dims(), candle_queue.dims());
        let exp = legacy.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let got = candle_queue.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let cos = cosine(&got, &exp);
        let max_abs = got
            .iter()
            .zip(exp.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let max_mag = got
            .iter()
            .chain(exp.iter())
            .map(|v| v.abs())
            .fold(0.0_f32, f32::max);
        let rel = max_abs / max_mag.max(1e-6);
        assert!(
            cos > 0.9999 && rel < 1e-3,
            "candle_queue vs legacy cos={cos} rel_max={rel} abs_max={max_abs}"
        );
        assert!(state_legacy.is_populated());
        assert!(state_cq.is_populated());
    }

    /// Workstream B Phase 6 — bf16-in parity. Same inputs as the f32 path,
    /// but `conv_out` / `z_flat` / `b_flat` / `a_flat` are cast to bf16
    /// before the call so the post-conv helper takes its bf16-in branch
    /// (skip-cast on q/k/v, bf16 SSM subgraph kernels). Output should match
    /// the f32 reference within bf16 rounding tolerance.
    ///
    /// Two passes (legacy queue + candle queue) confirm bf16-in works on
    /// both transport variants. dt_bias / a_log / norm_weight stay f32 to
    /// mirror production (those are dequantized constants, never cast for
    /// the chain).
    #[cfg(feature = "turboquant-gpu")]
    #[test]
    fn forward_post_conv_fused_bf16_in_matches_f32() {
        let device = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let ctx = match NativeContext::from_candle_device(&device) {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();

        let cfg = LinearAttnConfig {
            hidden_size: 16,
            num_k_heads: 2,
            num_v_heads: 4,
            head_dim: 8,
            conv_kernel: 4,
            rms_norm_eps: 1e-6,
            ssm_eps: 1e-6,
        };
        let batch = 1;
        let seq_len = 5;
        let qkv_dim = cfg.qkv_dim();
        let v_dim = cfg.v_dim();
        let hv = cfg.num_v_heads;

        let conv_out_f32 = Tensor::from_vec(
            lcg_vec(0xA1, batch * seq_len * qkv_dim, 0.5),
            (batch, seq_len, qkv_dim),
            &device,
        )
        .unwrap();
        let z_flat_f32 = Tensor::from_vec(
            lcg_vec(0xB2, batch * seq_len * v_dim, 0.5),
            (batch, seq_len, v_dim),
            &device,
        )
        .unwrap();
        let b_flat_f32 = Tensor::from_vec(
            lcg_vec(0xC3, batch * seq_len * hv, 0.5),
            (batch, seq_len, hv),
            &device,
        )
        .unwrap();
        let a_flat_f32 = Tensor::from_vec(
            lcg_vec(0xD4, batch * seq_len * hv, 0.5),
            (batch, seq_len, hv),
            &device,
        )
        .unwrap();
        let a_log = Tensor::from_vec(lcg_vec(0xE5, hv, 0.3), (hv,), &device).unwrap();
        let dt_bias = Tensor::from_vec(lcg_vec(0xF6, hv, 0.2), (hv,), &device).unwrap();
        let norm_w_v: Vec<f32> = (0..cfg.head_dim).map(|i| 1.0 + (i as f32) * 0.02).collect();
        let norm_weight = Tensor::from_vec(norm_w_v, (cfg.head_dim,), &device).unwrap();
        let out_proj = dense(cfg.hidden_size, v_dim, 0xA7, &device);

        // bf16 copies of the chain-active tensors (conv_out / z / b / a). The
        // f32 → bf16 → f32 round-trip on the inputs models what production
        // sees when `LUMEN_BF16_RMSNORM=1` flips the chain.
        let conv_out_bf = conv_out_f32.to_dtype(DType::BF16).unwrap().contiguous().unwrap();
        let z_flat_bf = z_flat_f32.to_dtype(DType::BF16).unwrap().contiguous().unwrap();
        let b_flat_bf = b_flat_f32.to_dtype(DType::BF16).unwrap().contiguous().unwrap();
        let a_flat_bf = a_flat_f32.to_dtype(DType::BF16).unwrap().contiguous().unwrap();

        // Reference: f32 path (legacy queue).
        let mut state_ref =
            NativeSsmState::new(&ctx, batch, hv, cfg.head_dim, cfg.head_dim).unwrap();
        let y_ref = forward_post_conv_fused_with_cache(
            &conv_out_f32,
            &z_flat_f32,
            &b_flat_f32,
            &a_flat_f32,
            &a_log,
            &dt_bias,
            &norm_weight,
            &out_proj,
            &cfg,
            &ctx,
            &lib,
            &mut state_ref,
            None,
            None,
        )
        .unwrap();

        // bf16-in path (legacy queue).
        let mut state_bf16 =
            NativeSsmState::new(&ctx, batch, hv, cfg.head_dim, cfg.head_dim).unwrap();
        let y_bf16 = forward_post_conv_fused_with_cache(
            &conv_out_bf,
            &z_flat_bf,
            &b_flat_bf,
            &a_flat_bf,
            &a_log,
            &dt_bias,
            &norm_weight,
            &out_proj,
            &cfg,
            &ctx,
            &lib,
            &mut state_bf16,
            None,
            None,
        )
        .unwrap();

        // Output dtype is dictated by `out_proj`. Dense (test build) and
        // MXFP4/Affine4 (production) all materialize f32 — the bf16 chain
        // collapses at the matmul boundary regardless of the bf16-in input.
        assert_eq!(y_ref.dims(), y_bf16.dims());

        let exp = y_ref.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let got = y_bf16
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let cos = cosine(&got, &exp);
        let max_abs = got
            .iter()
            .zip(exp.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let max_mag = got
            .iter()
            .chain(exp.iter())
            .map(|v| v.abs())
            .fold(0.0_f32, f32::max);
        let rel = max_abs / max_mag.max(1e-6);
        eprintln!(
            "B.6 bf16-in (legacy queue): cos={cos:.6} rel={rel:.4e} max_abs={max_abs:.4e}"
        );
        // bf16 rounding ≈ 1/256. The chain widens this through the SSM loop,
        // so allow ~5x headroom on relative L_inf — same envelope as B.4.
        assert!(
            cos > 0.998,
            "bf16-in cos={cos} below floor 0.998 (legacy queue)"
        );
        assert!(state_bf16.is_populated());

        // bf16-in path (candle queue) — same parity envelope.
        let mut state_cq_bf =
            NativeSsmState::new(&ctx, batch, hv, cfg.head_dim, cfg.head_dim).unwrap();
        let y_cq_bf = forward_post_conv_fused_with_cache_candle_queue(
            &conv_out_bf,
            &z_flat_bf,
            &b_flat_bf,
            &a_flat_bf,
            &a_log,
            &dt_bias,
            &norm_weight,
            &out_proj,
            &cfg,
            &ctx,
            &lib,
            &mut state_cq_bf,
            None,
            None,
        )
        .unwrap();
        let got_cq = y_cq_bf
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let cos_cq = cosine(&got_cq, &exp);
        eprintln!("B.6 bf16-in (candle queue): cos={cos_cq:.6}");
        assert!(
            cos_cq > 0.998,
            "bf16-in cos={cos_cq} below floor 0.998 (candle queue)"
        );
    }

    #[test]
    fn forward_linear_attn_fused_state_persists_between_calls() {
        // Decode-style: run prefill on the first half of the sequence with a
        // cold state, then continue token-by-token on the second half. The
        // concatenated output should match a single all-at-once prefill of
        // the full sequence.
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();
        let device = Device::Cpu;

        let cfg = LinearAttnConfig {
            hidden_size: 16,
            num_k_heads: 2,
            num_v_heads: 4,
            head_dim: 8,
            conv_kernel: 4,
            rms_norm_eps: 1e-6,
            ssm_eps: 1e-6,
        };
        let total_len = 6;
        let split = 3;

        // Build full sequence + projections.
        let hidden_v = lcg_vec(0x77AA, total_len * cfg.hidden_size, 1.2);
        let hidden_full = Tensor::from_vec(hidden_v, (1, total_len, cfg.hidden_size), &device)
            .unwrap();
        let combined_out = cfg.qkv_dim() + cfg.v_dim() + 2 * cfg.num_v_heads;
        let in_proj = dense(combined_out, cfg.hidden_size, 0x77BB, &device);
        let conv_w = lcg_vec(0x77CC, cfg.qkv_dim() * cfg.conv_kernel, 0.5);
        let conv_w_t = Tensor::from_vec(
            conv_w,
            (cfg.qkv_dim(), 1, cfg.conv_kernel),
            &device,
        )
        .unwrap();
        let conv_cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: cfg.qkv_dim(),
            cudnn_fwd_algo: None,
        };
        let conv1d = Conv1d::new(conv_w_t, None, conv_cfg);
        let a_log = Tensor::from_vec(
            lcg_vec(0x77DD, cfg.num_v_heads, 0.25),
            (cfg.num_v_heads,),
            &device,
        )
        .unwrap();
        let dt_bias = Tensor::from_vec(
            lcg_vec(0x77EE, cfg.num_v_heads, 0.15),
            (cfg.num_v_heads,),
            &device,
        )
        .unwrap();
        let norm_weight = Tensor::from_vec(
            (0..cfg.head_dim)
                .map(|i| 1.0 + (i as f32) * 0.01)
                .collect::<Vec<_>>(),
            (cfg.head_dim,),
            &device,
        )
        .unwrap();
        let out_proj = dense(cfg.hidden_size, cfg.v_dim(), 0x77F6, &device);

        // Cross-check: persistence test currently exercises the SSM state
        // hand-off but conv1d still runs cold each call (Phase A.8-C.4 keeps
        // conv1d on Candle). For deterministic comparison we therefore run
        // full prefill + fused chunk prefill against the *same* unfused
        // baseline that re-runs conv1d cold each call — mirroring the planned
        // wire-in path. State persistence is validated by checking the
        // second-call output is *non-zero* and *different* from a fresh
        // cold-state second-call output (i.e. the state actually carried).

        let mut state_full =
            NativeSsmState::new(&ctx, 1, cfg.num_v_heads, cfg.head_dim, cfg.head_dim).unwrap();
        let _y_full = forward_linear_attn_fused(
            &hidden_full,
            &in_proj,
            &conv1d,
            &a_log,
            &dt_bias,
            &norm_weight,
            &out_proj,
            &cfg,
            &ctx,
            &lib,
            &mut state_full,
        )
        .unwrap();

        // Chunked: chunk-1 produces a populated state; capture it, then run
        // chunk-2 with that state vs chunk-2 with a fresh cold state and show
        // the outputs differ.
        let chunk1 = hidden_full.narrow(1, 0, split).unwrap();
        let chunk2 = hidden_full.narrow(1, split, total_len - split).unwrap();
        let mut state_warm =
            NativeSsmState::new(&ctx, 1, cfg.num_v_heads, cfg.head_dim, cfg.head_dim).unwrap();
        let _ = forward_linear_attn_fused(
            &chunk1,
            &in_proj,
            &conv1d,
            &a_log,
            &dt_bias,
            &norm_weight,
            &out_proj,
            &cfg,
            &ctx,
            &lib,
            &mut state_warm,
        )
        .unwrap();
        assert!(state_warm.is_populated());
        let warm_state_snapshot = state_warm.buf().to_vec_f32().unwrap();
        // State must be non-trivial after a real prefill chunk.
        let max_abs = warm_state_snapshot
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs > 1e-4,
            "warm SSM state should be non-trivial after chunk1, max_abs={max_abs}"
        );

        let y_warm = forward_linear_attn_fused(
            &chunk2,
            &in_proj,
            &conv1d,
            &a_log,
            &dt_bias,
            &norm_weight,
            &out_proj,
            &cfg,
            &ctx,
            &lib,
            &mut state_warm,
        )
        .unwrap();

        let mut state_cold =
            NativeSsmState::new(&ctx, 1, cfg.num_v_heads, cfg.head_dim, cfg.head_dim).unwrap();
        let y_cold = forward_linear_attn_fused(
            &chunk2,
            &in_proj,
            &conv1d,
            &a_log,
            &dt_bias,
            &norm_weight,
            &out_proj,
            &cfg,
            &ctx,
            &lib,
            &mut state_cold,
        )
        .unwrap();

        let warm_v = y_warm.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let cold_v = y_cold.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let diff = warm_v
            .iter()
            .zip(cold_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            diff > 1e-4,
            "warm-state and cold-state outputs should differ when SSM state carries (diff={diff})"
        );
    }

    #[test]
    fn forward_linear_attn_fused_rejects_state_shape_mismatch() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();
        let device = Device::Cpu;
        let (cfg, hidden, in_proj, conv1d, a_log, dt_bias, norm_weight, out_proj) =
            build_linear_attn_fixture(&device);
        let mut bad_state =
            NativeSsmState::new(&ctx, 1, cfg.num_v_heads, cfg.head_dim, cfg.head_dim + 1).unwrap();
        let r = forward_linear_attn_fused(
            &hidden,
            &in_proj,
            &conv1d,
            &a_log,
            &dt_bias,
            &norm_weight,
            &out_proj,
            &cfg,
            &ctx,
            &lib,
            &mut bad_state,
        );
        assert!(r.is_err(), "expected shape mismatch error");
        assert!(format!("{}", r.unwrap_err()).contains("ssm_state shape"));
    }
}
