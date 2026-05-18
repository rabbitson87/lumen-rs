//! Native self-attention block (Phase A.3).
//!
//! Two layers of API:
//!
//! 1. [`forward_attn_block`] — pure native pipeline from Q/K/V (post-projection)
//!    to attention output, all `NativeTensor`. Composes:
//!    rms_norm → rope → transpose BLHD↔BHLD → GQA causal attention.
//!
//! 2. [`forward_self_attn`] — full self-attention forward including the
//!    `qkv_proj` and `o_proj` projections (Candle `ProjLinear`) and the
//!    optional `attn_output_gate` (q split + sigmoid * out). Bridges Candle
//!    Tensor ↔ NativeTensor at the projection boundaries via [`super::bridge`].

use lumen_metal::metal::CommandBufferExt;
use anyhow::{anyhow, Result};
use candle_core::{DType, Tensor, D};

use crate::qwen3_5_moe::proj::ProjLinear;

use super::bridge::{from_candle_tensor, from_candle_tensor_no_sync, to_candle_tensor};
use super::context::NativeContext;
use super::kernels::{build_rope_tables, KernelLib};
use super::kv_cache::NativeKvCache;
use super::tensor::{NativeDType, NativeTensor};

#[derive(Debug, Clone, Copy)]
pub struct AttnBlockConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// First `2 * (rotary_dim / 2)` head-dim components carry rotary phase;
    /// the remaining `head_dim - rotary_dim` pass through.
    pub rotary_dim: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    /// When `false`, attention is bidirectional (every query attends to every key).
    /// Production Candle prefill uses bidirectional (`mask=None` in `model.rs`); set this
    /// to `false` to match. Earlier parity tests left this `true` because their reference
    /// applied a causal mask manually.
    pub apply_causal: bool,
}

impl AttnBlockConfig {
    pub fn group_size(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    pub fn validate(&self) -> Result<()> {
        if self.num_kv_heads == 0 || self.num_heads % self.num_kv_heads != 0 {
            return Err(anyhow!(
                "num_heads ({}) must be a positive multiple of num_kv_heads ({})",
                self.num_heads,
                self.num_kv_heads
            ));
        }
        if self.head_dim == 0 {
            return Err(anyhow!("head_dim must be > 0"));
        }
        if self.rotary_dim % 2 != 0 {
            return Err(anyhow!(
                "rotary_dim ({}) must be even",
                self.rotary_dim
            ));
        }
        if self.rotary_dim > self.head_dim {
            return Err(anyhow!(
                "rotary_dim ({}) > head_dim ({})",
                self.rotary_dim,
                self.head_dim
            ));
        }
        Ok(())
    }
}

/// Run the post-projection self-attention block.
///
/// Inputs (all F32, BLHD layout):
///   - `q_raw`: `[B, L, num_heads, head_dim]`
///   - `k_raw`, `v_raw`: `[B, L, num_kv_heads, head_dim]`
///   - `gamma_q`, `gamma_k`: `[head_dim]`
///   - `pos_offset`: position of `q[..., 0, :]` in the absolute sequence.
///
/// Output: pre-allocated `[B, L, num_heads, head_dim]` F32 tensor.
///
/// Side effects: dispatches multiple kernels through `ctx.queue`. Each kernel
/// `wait_until_completed`s internally — fine for parity testing, optimized out
/// once we move to the FlashAttention/tiled path.
#[allow(clippy::too_many_arguments)]
pub fn forward_attn_block(
    ctx: &NativeContext,
    lib: &KernelLib,
    q_raw: &NativeTensor,
    k_raw: &NativeTensor,
    v_raw: &NativeTensor,
    gamma_q: &NativeTensor,
    gamma_k: &NativeTensor,
    pos_offset: usize,
    out: &NativeTensor,
    cfg: &AttnBlockConfig,
) -> Result<()> {
    let (b, l) = (q_raw.shape()[0], q_raw.shape()[1]);
    let kv = cfg.num_kv_heads;
    let d = cfg.head_dim;
    let k_bhld = ctx.zeros(vec![b, kv, l, d], NativeDType::F32)?;
    let v_bhld = ctx.zeros(vec![b, kv, l, d], NativeDType::F32)?;
    forward_attn_block_full(
        ctx, lib, q_raw, k_raw, v_raw, gamma_q, gamma_k, pos_offset, out, &k_bhld, &v_bhld, cfg,
    )
}

/// Same as [`forward_attn_block`] but writes the post-RoPE BHLD K and BHLD V into
/// caller-provided buffers. These are exactly the tensors Candle's `KvCache::append`
/// expects (`[B, num_kv_heads, L, head_dim]`), so production wiring can append them
/// to the live KV cache without re-running the QKV projection or RoPE.
///
/// Use this variant when integrating native attention into the production hot loop.
/// The plain [`forward_attn_block`] delegates here with internally-allocated K/V
/// buffers that get dropped immediately — fine for parity tests, wasteful in production.
#[allow(clippy::too_many_arguments)]
pub fn forward_attn_block_full(
    ctx: &NativeContext,
    lib: &KernelLib,
    q_raw: &NativeTensor,
    k_raw: &NativeTensor,
    v_raw: &NativeTensor,
    gamma_q: &NativeTensor,
    gamma_k: &NativeTensor,
    pos_offset: usize,
    out: &NativeTensor,
    k_bhld_out: &NativeTensor,
    v_bhld_out: &NativeTensor,
    cfg: &AttnBlockConfig,
) -> Result<()> {
    cfg.validate()?;
    let h = cfg.num_heads;
    let kv = cfg.num_kv_heads;
    let d = cfg.head_dim;
    let half = cfg.rotary_dim / 2;

    if q_raw.rank() != 4 {
        return Err(anyhow!(
            "q_raw must be rank 4 [B,L,H,D], got {:?}",
            q_raw.shape()
        ));
    }
    let (b, l) = (q_raw.shape()[0], q_raw.shape()[1]);
    if q_raw.shape() != [b, l, h, d] {
        return Err(anyhow!(
            "q_raw shape {:?} != [{b}, {l}, {h}, {d}]",
            q_raw.shape()
        ));
    }
    if k_raw.shape() != [b, l, kv, d] || v_raw.shape() != [b, l, kv, d] {
        return Err(anyhow!(
            "k/v shape mismatch: K={:?} V={:?} expected [{b},{l},{kv},{d}]",
            k_raw.shape(),
            v_raw.shape()
        ));
    }
    if gamma_q.shape() != [d] || gamma_k.shape() != [d] {
        return Err(anyhow!(
            "gamma_q/gamma_k must be [head_dim={d}], got {:?} / {:?}",
            gamma_q.shape(),
            gamma_k.shape()
        ));
    }
    if out.shape() != [b, l, h, d] {
        return Err(anyhow!(
            "out shape {:?} != [{b},{l},{h},{d}]",
            out.shape()
        ));
    }
    if k_bhld_out.shape() != [b, kv, l, d] || v_bhld_out.shape() != [b, kv, l, d] {
        return Err(anyhow!(
            "k/v_bhld_out shape mismatch: K={:?} V={:?} expected [{b},{kv},{l},{d}]",
            k_bhld_out.shape(),
            v_bhld_out.shape()
        ));
    }

    // RMSNorm operates on [rows, hidden] = [B*L*H, head_dim] views.
    let q_2d = q_raw.reshape(vec![b * l * h, d])?;
    let k_2d = k_raw.reshape(vec![b * l * kv, d])?;
    let q_normed_2d = ctx.zeros(vec![b * l * h, d], NativeDType::F32)?;
    let k_normed_2d = ctx.zeros(vec![b * l * kv, d], NativeDType::F32)?;
    lib.rms_norm(ctx, &q_2d, gamma_q, cfg.rms_eps, &q_normed_2d)?;
    lib.rms_norm(ctx, &k_2d, gamma_k, cfg.rms_eps, &k_normed_2d)?;

    // Reinterpret normalized outputs as BLHD for RoPE (which keys cos/sin on
    // the second axis = sequence).
    let q_blhd_normed = q_normed_2d.reshape(vec![b, l, h, d])?;
    let k_blhd_normed = k_normed_2d.reshape(vec![b, l, kv, d])?;

    // Build RoPE tables once for this call.
    let (cos_t, sin_t) = build_rope_tables(ctx, cfg.rotary_dim, l, pos_offset, cfg.rope_theta)?;

    // Apply RoPE on BLHD (kernel forbids in-place).
    let q_blhd_roped = ctx.zeros(vec![b, l, h, d], NativeDType::F32)?;
    let k_blhd_roped = ctx.zeros(vec![b, l, kv, d], NativeDType::F32)?;
    lib.rope_partial(ctx, &q_blhd_normed, &cos_t, &sin_t, &q_blhd_roped, half)?;
    lib.rope_partial(ctx, &k_blhd_normed, &cos_t, &sin_t, &k_blhd_roped, half)?;

    // Transpose to BHLD. K and V land in caller-provided buffers so production can
    // forward them to Candle's KvCache without re-reading the data.
    let q_bhld = ctx.zeros(vec![b, h, l, d], NativeDType::F32)?;
    lib.transpose_blhd(ctx, &q_blhd_roped, &q_bhld, 0)?;
    lib.transpose_blhd(ctx, &k_blhd_roped, k_bhld_out, 0)?;
    lib.transpose_blhd(ctx, v_raw, v_bhld_out, 0)?;

    // GQA-aware attention (causal flag from cfg).
    let attn_bhld = ctx.zeros(vec![b, h, l, d], NativeDType::F32)?;
    lib.attention(
        ctx,
        &q_bhld,
        k_bhld_out,
        v_bhld_out,
        &attn_bhld,
        pos_offset,
        cfg.apply_causal,
    )?;

    // Transpose result back to BLHD and write into the caller's output tensor.
    lib.transpose_blhd(ctx, &attn_bhld, out, 1)?;
    Ok(())
}

/// Full self-attention forward configuration.
///
/// Mirrors the relevant subset of `qwen3_5_moe::self_attn::SelfAttnDims`:
/// dimensions plus the `attn_output_gate` flag that determines whether the
/// `qkv_proj` output begins with a `2 · num_heads · head_dim`-wide section
/// (q + gate split) instead of the usual `num_heads · head_dim`.
#[derive(Debug, Clone, Copy)]
pub struct SelfAttnConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub attn_output_gate: bool,
    /// Forwarded to [`AttnBlockConfig::apply_causal`].
    pub apply_causal: bool,
}

impl SelfAttnConfig {
    fn block_config(&self) -> AttnBlockConfig {
        AttnBlockConfig {
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            rotary_dim: self.rotary_dim,
            rms_eps: self.rms_eps,
            rope_theta: self.rope_theta,
            apply_causal: self.apply_causal,
        }
    }

    fn q_section_dim(&self) -> usize {
        if self.attn_output_gate {
            2 * self.num_heads * self.head_dim
        } else {
            self.num_heads * self.head_dim
        }
    }

    fn kv_out_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }
}

/// Full prefill self-attention forward (no KV cache, no TurboQuant compressor).
///
/// Pipeline:
///   1. `qkv_proj(hidden)` → narrow into `q_raw / k_raw / v_raw` (with optional
///      `q + gate` split when `attn_output_gate` is set).
///   2. Bridge Candle → NativeTensor (zero-copy when contiguous Metal-resident).
///   3. [`forward_attn_block`] — rms_norm → rope → transpose → GQA causal attn.
///   4. Bridge back to Candle, apply sigmoid gate (if any), `o_proj`.
///
/// Caller is responsible for supplying the gamma weights as Candle Tensors that
/// match the model's loaded `q_norm` / `k_norm` (1-D `[head_dim]`).
///
/// KV cache & decode-mode reuse land in a follow-up step alongside Candle's
/// `KvCache` integration; the current shape is the prefill / stateless case.
#[allow(clippy::too_many_arguments)]
pub fn forward_self_attn(
    hidden: &Tensor,
    qkv_proj: &ProjLinear,
    q_norm_gamma: &Tensor,
    k_norm_gamma: &Tensor,
    o_proj: &ProjLinear,
    pos_offset: usize,
    cfg: &SelfAttnConfig,
    ctx: &NativeContext,
    lib: &KernelLib,
) -> Result<Tensor> {
    let (out, _k, _v) = forward_self_attn_full(
        hidden,
        qkv_proj,
        q_norm_gamma,
        k_norm_gamma,
        o_proj,
        pos_offset,
        cfg,
        ctx,
        lib,
    )?;
    Ok(out)
}

/// Same as [`forward_self_attn`] but additionally returns the post-RoPE K (BHLD) and
/// V (BHLD) tensors as Candle `Tensor`s. Production wiring uses these to feed
/// Candle's `KvCache::append`, allowing the native attention path to coexist with
/// the existing cache machinery without duplicating QKV projection / RoPE work.
///
/// Returns `(attn_out, k_bhld_post_rope, v_bhld)`.
#[allow(clippy::too_many_arguments)]
pub fn forward_self_attn_full(
    hidden: &Tensor,
    qkv_proj: &ProjLinear,
    q_norm_gamma: &Tensor,
    k_norm_gamma: &Tensor,
    o_proj: &ProjLinear,
    pos_offset: usize,
    cfg: &SelfAttnConfig,
    ctx: &NativeContext,
    lib: &KernelLib,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (b, l, hidden_dim) = hidden.dims3().map_err(|e| anyhow!("{e}"))?;
    let h = cfg.num_heads;
    let kv = cfg.num_kv_heads;
    let d = cfg.head_dim;
    let _ = hidden_dim;

    // 1. qkv projection (Candle path).
    let qkv = qkv_proj
        .forward(hidden)
        .map_err(|e| anyhow!("qkv_proj: {e}"))?;
    let last = qkv.dims().len() - 1;
    let q_section_dim = cfg.q_section_dim();
    let kv_out = cfg.kv_out_dim();

    // 2. Split into Q (+ optional gate), K, V.
    //
    // Critical: `.contiguous()` after every `narrow` on a non-last axis (or last-axis narrow
    // that doesn't tile the full row). Reshape on a non-contiguous tensor either panics or
    // silently produces garbage strides, which then bridge into native as misindexed data —
    // observed as cos≈0.003 between native and candle paths at production size before this
    // fix. The earlier synthetic parity tests didn't catch it because their Dense Linear
    // outputs happened to be tight enough that the missing `contiguous` was a no-op.
    let (q_raw_candle, gate_flat) = if cfg.attn_output_gate {
        let q_section = qkv
            .narrow(last, 0, q_section_dim)
            .and_then(|t| t.contiguous())
            .map_err(|e| anyhow!("{e}"))?;
        let q_split = q_section
            .reshape((b, l, h, 2 * d))
            .map_err(|e| anyhow!("{e}"))?;
        let q = q_split
            .narrow(D::Minus1, 0, d)
            .map_err(|e| anyhow!("{e}"))?
            .contiguous()
            .map_err(|e| anyhow!("{e}"))?;
        let g = q_split
            .narrow(D::Minus1, d, d)
            .map_err(|e| anyhow!("{e}"))?
            .contiguous()
            .map_err(|e| anyhow!("{e}"))?
            .reshape((b, l, h * d))
            .map_err(|e| anyhow!("{e}"))?;
        (q, Some(g))
    } else {
        let q = qkv
            .narrow(last, 0, q_section_dim)
            .and_then(|t| t.contiguous())
            .map_err(|e| anyhow!("{e}"))?
            .reshape((b, l, h, d))
            .map_err(|e| anyhow!("{e}"))?;
        (q, None)
    };
    let k_raw_candle = qkv
        .narrow(last, q_section_dim, kv_out)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("{e}"))?
        .reshape((b, l, kv, d))
        .map_err(|e| anyhow!("{e}"))?
        .contiguous()
        .map_err(|e| anyhow!("{e}"))?;
    let v_raw_candle = qkv
        .narrow(last, q_section_dim + kv_out, kv_out)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("{e}"))?
        .reshape((b, l, kv, d))
        .map_err(|e| anyhow!("{e}"))?
        .contiguous()
        .map_err(|e| anyhow!("{e}"))?;

    // 3. Bridge Candle → NativeTensor.
    let q_raw = from_candle_tensor(ctx, &q_raw_candle)?;
    let k_raw = from_candle_tensor(ctx, &k_raw_candle)?;
    let v_raw = from_candle_tensor(ctx, &v_raw_candle)?;
    let gamma_q = from_candle_tensor(ctx, q_norm_gamma)?;
    let gamma_k = from_candle_tensor(ctx, k_norm_gamma)?;

    // Optional internal stage debug. Off by default. When `LUMEN_NATIVE_INTERNAL_DEBUG=1`
    // dumps first-4-element snapshots of every intermediate so the divergence vs the candle
    // path can be located precisely.
    let internal_debug = std::env::var("LUMEN_NATIVE_INTERNAL_DEBUG")
        .map(|v| v == "1")
        .unwrap_or(false);
    if internal_debug {
        let q_raw_v = q_raw.to_vec_f32().unwrap_or_default();
        let k_raw_v = k_raw.to_vec_f32().unwrap_or_default();
        let v_raw_v = v_raw.to_vec_f32().unwrap_or_default();
        let gamma_q_v = gamma_q.to_vec_f32().unwrap_or_default();
        let gamma_k_v = gamma_k.to_vec_f32().unwrap_or_default();
        eprintln!(
            "    [native-internal] q_raw shape={:?} dtype={:?} first4={:?} last4={:?}",
            q_raw.shape(),
            q_raw.dtype(),
            &q_raw_v[..q_raw_v.len().min(4)],
            &q_raw_v[q_raw_v.len().saturating_sub(4)..],
        );
        eprintln!(
            "    [native-internal] k_raw first4={:?} v_raw first4={:?}",
            &k_raw_v[..k_raw_v.len().min(4)],
            &v_raw_v[..v_raw_v.len().min(4)],
        );
        eprintln!(
            "    [native-internal] gamma_q first4={:?} gamma_k first4={:?}",
            &gamma_q_v[..gamma_q_v.len().min(4)],
            &gamma_k_v[..gamma_k_v.len().min(4)],
        );
    }

    // 4. Native attention block. Capture post-RoPE K (BHLD) and V (BHLD) so the
    //    caller can feed them into Candle's KvCache::append on prefill steps.
    let block_cfg = cfg.block_config();
    let attn_out_native = ctx.zeros(vec![b, l, h, d], NativeDType::F32)?;
    let k_bhld_native = ctx.zeros(vec![b, kv, l, d], NativeDType::F32)?;
    let v_bhld_native = ctx.zeros(vec![b, kv, l, d], NativeDType::F32)?;
    forward_attn_block_full(
        ctx,
        lib,
        &q_raw,
        &k_raw,
        &v_raw,
        &gamma_q,
        &gamma_k,
        pos_offset,
        &attn_out_native,
        &k_bhld_native,
        &v_bhld_native,
        &block_cfg,
    )?;

    // 5. Native → Candle, reshape attn output to flat hidden width.
    let attn_out = to_candle_tensor(&attn_out_native, hidden.device())?
        .reshape((b, l, h * d))
        .map_err(|e| anyhow!("{e}"))?;
    let k_bhld = to_candle_tensor(&k_bhld_native, hidden.device())?;
    let v_bhld = to_candle_tensor(&v_bhld_native, hidden.device())?;

    // 6. Apply sigmoid gate if `attn_output_gate` was set.
    let gated = if let Some(gate) = gate_flat {
        let sig = candle_nn::ops::sigmoid(&gate).map_err(|e| anyhow!("{e}"))?;
        (attn_out * sig).map_err(|e| anyhow!("{e}"))?
    } else {
        attn_out
    };

    // 7. Output projection.
    let out = o_proj
        .forward(&gated)
        .map_err(|e| anyhow!("o_proj: {e}"))?;
    Ok((out, k_bhld, v_bhld))
}

/// Decode/prefill forward that keeps the KV state native end-to-end.
///
/// Unlike [`forward_self_attn_full`], which returns post-RoPE K/V back to Candle
/// land for `KvCache::append`, this entry point appends K/V directly into a
/// caller-owned [`NativeKvCache`] buffer and runs attention against the
/// cumulative cache range. No bridge round-trip is needed for K/V — the only
/// Candle ↔ native bridges are on the input `hidden`, the per-call gamma
/// weights, and the final attention output.
///
/// Handles both prefill (`S > 1`, `cache.current_seq_len() == 0`) and decode
/// (`S = 1`, `cache.current_seq_len() > 0`) uniformly. Position offsets for RoPE
/// derive from `cache.current_seq_len()` *before* the append.
///
/// `cache.batch()`, `cache.kv_heads()`, and `cache.head_dim()` must match
/// `cfg.num_kv_heads` and `cfg.head_dim` for the input batch size — caller
/// owns the cache lifecycle (allocate once at model load, reset between
/// independent generation requests).
#[allow(clippy::too_many_arguments)]
pub fn forward_self_attn_with_native_cache(
    hidden: &Tensor,
    qkv_proj: &ProjLinear,
    q_norm_gamma: &Tensor,
    k_norm_gamma: &Tensor,
    o_proj: &ProjLinear,
    cfg: &SelfAttnConfig,
    ctx: &NativeContext,
    lib: &KernelLib,
    cache: &mut NativeKvCache,
) -> Result<Tensor> {
    let (b, l, _hidden_dim) = hidden.dims3().map_err(|e| anyhow!("{e}"))?;
    let h = cfg.num_heads;
    let kv = cfg.num_kv_heads;
    let d = cfg.head_dim;
    let half = cfg.rotary_dim / 2;

    if cache.batch() != b {
        return Err(anyhow!(
            "cache batch ({}) != hidden batch ({b})",
            cache.batch()
        ));
    }
    if cache.kv_heads() != kv || cache.head_dim() != d {
        return Err(anyhow!(
            "cache (kv={}, d={}) != cfg (kv={kv}, d={d})",
            cache.kv_heads(),
            cache.head_dim()
        ));
    }
    let pos_offset = cache.current_seq_len();
    if pos_offset + l > cache.max_seq_len() {
        return Err(anyhow!(
            "cache overflow: past {} + S {l} > max {}",
            pos_offset,
            cache.max_seq_len()
        ));
    }

    // 1. QKV projection (Candle path).
    let qkv = qkv_proj
        .forward(hidden)
        .map_err(|e| anyhow!("qkv_proj: {e}"))?;
    let last = qkv.dims().len() - 1;
    let q_section_dim = cfg.q_section_dim();
    let kv_out = cfg.kv_out_dim();

    // 2. Split Q (+ optional gate), K, V. Same `.contiguous()` discipline as
    //    `forward_self_attn_full`.
    let (q_raw_candle, gate_flat) = if cfg.attn_output_gate {
        let q_section = qkv
            .narrow(last, 0, q_section_dim)
            .and_then(|t| t.contiguous())
            .map_err(|e| anyhow!("{e}"))?;
        let q_split = q_section
            .reshape((b, l, h, 2 * d))
            .map_err(|e| anyhow!("{e}"))?;
        let q = q_split
            .narrow(D::Minus1, 0, d)
            .map_err(|e| anyhow!("{e}"))?
            .contiguous()
            .map_err(|e| anyhow!("{e}"))?;
        let g = q_split
            .narrow(D::Minus1, d, d)
            .map_err(|e| anyhow!("{e}"))?
            .contiguous()
            .map_err(|e| anyhow!("{e}"))?
            .reshape((b, l, h * d))
            .map_err(|e| anyhow!("{e}"))?;
        (q, Some(g))
    } else {
        let q = qkv
            .narrow(last, 0, q_section_dim)
            .and_then(|t| t.contiguous())
            .map_err(|e| anyhow!("{e}"))?
            .reshape((b, l, h, d))
            .map_err(|e| anyhow!("{e}"))?;
        (q, None)
    };
    let k_raw_candle = qkv
        .narrow(last, q_section_dim, kv_out)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("{e}"))?
        .reshape((b, l, kv, d))
        .map_err(|e| anyhow!("{e}"))?
        .contiguous()
        .map_err(|e| anyhow!("{e}"))?;
    let v_raw_candle = qkv
        .narrow(last, q_section_dim + kv_out, kv_out)
        .and_then(|t| t.contiguous())
        .map_err(|e| anyhow!("{e}"))?
        .reshape((b, l, kv, d))
        .map_err(|e| anyhow!("{e}"))?
        .contiguous()
        .map_err(|e| anyhow!("{e}"))?;

    // pre-cast `gate_flat` to F32 on Candle's queue when we'll
    // bridge it natively. The next `from_candle_tensor` call below carries
    // the single Candle-queue sync, so this sit on the same FIFO.
    let out_mxfp4 = o_proj.as_mxfp4();
    let gate_f32_for_native = if out_mxfp4.is_some() {
        match gate_flat.as_ref() {
            Some(g) => Some(
                g.to_dtype(DType::F32)
                    .and_then(|t| t.contiguous())
                    .map_err(|e| anyhow!("gate to f32: {e}"))?,
            ),
            None => None,
        }
    } else {
        None
    };

    // 3. Bridge to native.
    let q_raw = from_candle_tensor(ctx, &q_raw_candle)?;
    let k_raw = from_candle_tensor(ctx, &k_raw_candle)?;
    let v_raw = from_candle_tensor(ctx, &v_raw_candle)?;
    let gamma_q = from_candle_tensor(ctx, q_norm_gamma)?;
    let gamma_k = from_candle_tensor(ctx, k_norm_gamma)?;

    // 4. Pre-allocate every native intermediate so the fused command buffer
    //    can encode all dispatches without going back to the allocator.
    let q_2d = q_raw.reshape(vec![b * l * h, d])?;
    let k_2d = k_raw.reshape(vec![b * l * kv, d])?;
    let q_normed_2d = ctx.zeros(vec![b * l * h, d], NativeDType::F32)?;
    let k_normed_2d = ctx.zeros(vec![b * l * kv, d], NativeDType::F32)?;
    let q_blhd_normed = q_normed_2d.reshape(vec![b, l, h, d])?;
    let k_blhd_normed = k_normed_2d.reshape(vec![b, l, kv, d])?;
    let (cos_t, sin_t) = build_rope_tables(ctx, cfg.rotary_dim, l, pos_offset, cfg.rope_theta)?;
    let q_blhd_roped = ctx.zeros(vec![b, l, h, d], NativeDType::F32)?;
    let k_blhd_roped = ctx.zeros(vec![b, l, kv, d], NativeDType::F32)?;
    let q_bhld = ctx.zeros(vec![b, h, l, d], NativeDType::F32)?;
    let k_new_bhld = ctx.zeros(vec![b, kv, l, d], NativeDType::F32)?;
    let v_new_bhld = ctx.zeros(vec![b, kv, l, d], NativeDType::F32)?;
    let attn_bhld = ctx.zeros(vec![b, h, l, d], NativeDType::F32)?;
    let attn_blhd = ctx.zeros(vec![b, l, h, d], NativeDType::F32)?;

    // fused-tail scratch (only when MXFP4 o_proj is available).
    // Bridge gate after the post-allocation single Candle sync; keep both the
    // F32 owner Tensor and its NativeTensor view alive for the cmd buffer.
    let gate_native = match gate_f32_for_native.as_ref() {
        Some(g_f32) => Some(from_candle_tensor_no_sync(ctx, g_f32)?),
        None => None,
    };
    let gated_n = if out_mxfp4.is_some() && gate_native.is_some() {
        Some(ctx.zeros(vec![b, l, h * d], NativeDType::F32)?)
    } else {
        None
    };
    let out_y_n = if let Some(linear) = out_mxfp4 {
        Some(ctx.zeros(
            vec![b * l, linear.out_features()],
            NativeDType::F32,
        )?)
    } else {
        None
    };

    // 5. **Single fused command buffer** for every native op below.
    //    Splits into three encoders because Metal types differ:
    //      a) compute encoder for the pre-cache pipeline (rms_norm × 2,
    //         rope × 2, transpose × 3),
    //      b) blit encoder for `cache.encode_append` (memcpy into cache buffer),
    //      c) compute encoder for attention + final transpose-back.
    //    Metal's default tracked mode inserts implicit fences between encoders
    //    in the same command buffer that access the same resource — so the
    //    write-after-write/read-after-write hazards across phases are handled
    //    automatically. Single commit + single wait_until_completed amortizes
    //    the per-commit overhead (~50µs each) across 10 dispatches per layer.
    let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);

    // pre-cache compute.
    let enc_a = cmd.auto_compute_encoder();
    enc_a.set_label("lumen:self_attn_phase_a_pre_cache");
    lib.encode_rms_norm(&enc_a, &q_2d, &gamma_q, cfg.rms_eps, &q_normed_2d)?;
    lib.encode_rms_norm(&enc_a, &k_2d, &gamma_k, cfg.rms_eps, &k_normed_2d)?;
    lib.encode_rope_partial(&enc_a, &q_blhd_normed, &cos_t, &sin_t, &q_blhd_roped, half)?;
    lib.encode_rope_partial(&enc_a, &k_blhd_normed, &cos_t, &sin_t, &k_blhd_roped, half)?;
    lib.encode_transpose_blhd(&enc_a, &q_blhd_roped, &q_bhld, 0)?;
    lib.encode_transpose_blhd(&enc_a, &k_blhd_roped, &k_new_bhld, 0)?;
    lib.encode_transpose_blhd(&enc_a, &v_raw, &v_new_bhld, 0)?;
    enc_a.end_encoding();

    // blit append into the persistent cache buffer.
    let mut blit = lumen_metal::metal::blit_command_encoder_no_fence(&cmd);
    cache.encode_append(&mut blit, &k_new_bhld, &v_new_bhld)?;
    blit.end_encoding();
    let l_kv_active = cache.current_seq_len();
    let kv_stride = cache.max_seq_len();

    // attention + transpose-back.
    let enc_c = cmd.auto_compute_encoder();
    enc_c.set_label("lumen:self_attn_phase_c_attention");
    lib.encode_attention_with_kv_stride(
        &enc_c,
        &q_bhld,
        cache.k_buf(),
        cache.v_buf(),
        &attn_bhld,
        pos_offset,
        cfg.apply_causal,
        l_kv_active,
        kv_stride,
    )?;
    lib.encode_transpose_blhd(&enc_c, &attn_bhld, &attn_blhd, 1)?;

    // fold the output gate (if any) and o_proj into the same
    // command buffer when o_proj is MXFP4. Drops 1 commit/layer × 10 full
    // attention layers per token (~0.5ms on M3 Max).
    if let Some(out_mxfp4_linear) = out_mxfp4 {
        let attn_flat = attn_blhd.reshape(vec![b, l, h * d])?;
        let final_input: &NativeTensor = match gate_native.as_ref() {
            Some(gate_n) => {
                let gated = gated_n
                    .as_ref()
                    .expect("gated_n is allocated whenever gate_native is Some on the mxfp4 path");
                lib.encode_sigmoid_mul(&enc_c, gate_n, &attn_flat, gated)?;
                gated
            }
            None => &attn_flat,
        };
        let final_input_2d = final_input.reshape(vec![b * l, h * d])?;
        let out_y = out_y_n
            .as_ref()
            .expect("out_y_n is allocated whenever out_mxfp4 is Some");
        out_mxfp4_linear.encode_native(
            &enc_c,
            final_input_2d.buffer(),
            final_input_2d.offset_bytes() as u64,
            out_y.buffer(),
            out_y.offset_bytes() as u64,
            b * l,
        );
    }
    enc_c.end_encoding();

    cmd.commit();
    cmd.wait_until_completed();

    // gate + o_proj already happened on the GPU inside
    // the fused buffer. Bridge the result back to Candle and return.
    if let Some(linear) = out_mxfp4 {
        let out_y = out_y_n.expect("out_y_n is allocated whenever out_mxfp4 is Some");
        let hidden_out = linear.out_features();
        return to_candle_tensor(&out_y, hidden.device())?
            .reshape((b, l, hidden_out))
            .map_err(|e| anyhow!("{e}"));
    }

    // 6. Bridge attention output back to Candle land for the gate + o_proj.
    let attn_out = to_candle_tensor(&attn_blhd, hidden.device())?
        .reshape((b, l, h * d))
        .map_err(|e| anyhow!("{e}"))?;

    // 10. Apply sigmoid gate if `attn_output_gate` was set, then `o_proj`.
    let gated = if let Some(gate) = gate_flat {
        let sig = candle_nn::ops::sigmoid(&gate).map_err(|e| anyhow!("{e}"))?;
        (attn_out * sig).map_err(|e| anyhow!("{e}"))?
    } else {
        attn_out
    };
    o_proj
        .forward(&gated)
        .map_err(|e| anyhow!("o_proj: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor, D};
    use candle_nn::{rms_norm, rotary_emb, VarBuilder, VarMap};

    /// Candle reference: same algorithm as `forward_attn_block`, expressed in
    /// Candle ops. Matches the post-projection forward portion of
    /// `qwen3_5_moe::self_attn::SelfAttention::forward_with_tq` (no KV cache,
    /// no TurboQuant compressor — pure prefill on a fresh sequence).
    #[allow(clippy::too_many_arguments)]
    fn candle_attn_block(
        q_raw: &Tensor,
        k_raw: &Tensor,
        v_raw: &Tensor,
        gamma_q: &Tensor,
        gamma_k: &Tensor,
        cfg: &AttnBlockConfig,
        pos_offset: usize,
    ) -> Tensor {
        let device = q_raw.device().clone();

        // Per-head RMSNorm. Build a Candle RmsNorm using the gamma weights.
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        // RmsNorm needs to grab "weight" from VarBuilder; use a custom path.
        let _ = vb; // varmap path is awkward; do it manually.

        let manual_rms = |x: &Tensor, gamma: &Tensor| -> Tensor {
            let _ = rms_norm; // suppress unused
            let ms = x.sqr().unwrap().mean_keepdim(D::Minus1).unwrap();
            let scale = (ms + cfg.rms_eps as f64).unwrap().sqrt().unwrap().recip().unwrap();
            x.broadcast_mul(&scale).unwrap().broadcast_mul(gamma).unwrap()
        };

        let q_n = manual_rms(q_raw, gamma_q); // [B,L,H,D]
        let k_n = manual_rms(k_raw, gamma_k); // [B,L,kv,D]

        // Transpose to BHLD.
        let q_t = q_n.transpose(1, 2).unwrap().contiguous().unwrap();
        let k_t = k_n.transpose(1, 2).unwrap().contiguous().unwrap();
        let v_t = v_raw.transpose(1, 2).unwrap().contiguous().unwrap();

        // RoPE: build cos/sin and apply to first rotary_dim of last axis.
        let half = cfg.rotary_dim / 2;
        let l = q_raw.dim(1).unwrap();
        let inv_freqs: Vec<f32> = (0..half)
            .map(|i| {
                let exp = (2 * i) as f32 / cfg.rotary_dim as f32;
                1.0 / cfg.rope_theta.powf(exp)
            })
            .collect();
        let inv_t = Tensor::from_vec(inv_freqs, (half,), &device).unwrap();
        let positions: Vec<f32> = (0..l).map(|t| (pos_offset + t) as f32).collect();
        let pos_t = Tensor::from_vec(positions, (l,), &device).unwrap();
        let angles = pos_t
            .unsqueeze(1)
            .unwrap()
            .broadcast_mul(&inv_t.unsqueeze(0).unwrap())
            .unwrap();
        let cos = angles.cos().unwrap();
        let sin = angles.sin().unwrap();

        let apply_rope = |x: &Tensor, rotary_dim: usize| -> Tensor {
            let head_dim = x.dim(D::Minus1).unwrap();
            if rotary_dim == head_dim {
                rotary_emb::rope(&x.contiguous().unwrap(), &cos, &sin).unwrap()
            } else {
                let rot = x.narrow(D::Minus1, 0, rotary_dim).unwrap().contiguous().unwrap();
                let pass = x.narrow(D::Minus1, rotary_dim, head_dim - rotary_dim).unwrap();
                let rot = rotary_emb::rope(&rot, &cos, &sin).unwrap();
                Tensor::cat(&[&rot, &pass], D::Minus1).unwrap().contiguous().unwrap()
            }
        };

        let q_r = apply_rope(&q_t, cfg.rotary_dim);
        let k_r = apply_rope(&k_t, cfg.rotary_dim);

        // GQA: repeat KV heads.
        let group = cfg.group_size();
        let (b, n_kv, l_kv, dh) = k_r.dims4().unwrap();
        let k_x = if group == 1 {
            k_r
        } else {
            k_r.unsqueeze(2)
                .unwrap()
                .expand((b, n_kv, group, l_kv, dh))
                .unwrap()
                .reshape((b, n_kv * group, l_kv, dh))
                .unwrap()
        };
        let (b, n_kv, l_kv, dh) = v_t.dims4().unwrap();
        let v_x = if group == 1 {
            v_t
        } else {
            v_t.unsqueeze(2)
                .unwrap()
                .expand((b, n_kv, group, l_kv, dh))
                .unwrap()
                .reshape((b, n_kv * group, l_kv, dh))
                .unwrap()
        };

        // Causal attention (mask via lower-triangular, additive -inf above diagonal).
        let scale = (cfg.head_dim as f64).powf(-0.5);
        let scores = (q_r.matmul(&k_x.transpose(D::Minus2, D::Minus1).unwrap()).unwrap()
            * scale)
            .unwrap();
        // Build mask [L, L]: 0 on/below diagonal, -INF above.
        let l_q = q_r.dim(D::Minus2).unwrap();
        let mut mask_v = vec![0.0_f32; l_q * l_kv];
        for qi in 0..l_q {
            let q_pos = pos_offset + qi;
            for ki in 0..l_kv {
                if ki > q_pos {
                    mask_v[qi * l_kv + ki] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec(mask_v, (l_q, l_kv), &device).unwrap();
        let scores = scores.broadcast_add(&mask).unwrap();
        let attn = candle_nn::ops::softmax_last_dim(&scores).unwrap();
        let out = attn.matmul(&v_x).unwrap();

        // Back to BLHD.
        out.transpose(1, 2).unwrap().contiguous().unwrap()
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

    fn dense_linear(out_f: usize, in_f: usize, seed: u32, device: &Device) -> ProjLinear {
        let mut s = seed;
        let w: Vec<f32> = (0..out_f * in_f)
            .map(|_| {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((s >> 8 & 0xFFFF) as f32 / 65535.0 - 0.5) * 0.5
            })
            .collect();
        let weight = Tensor::from_vec(w, (out_f, in_f), device).unwrap();
        ProjLinear::Dense(candle_nn::Linear::new(weight, None))
    }

    /// Candle reference for `forward_self_attn`. Mirrors the algorithm in
    /// `qwen3_5_moe::self_attn::SelfAttention::forward_with_tq` for the no-cache,
    /// no-TurboQuant prefill case.
    #[allow(clippy::too_many_arguments)]
    fn candle_self_attn(
        hidden: &Tensor,
        qkv_proj: &ProjLinear,
        q_norm_gamma: &Tensor,
        k_norm_gamma: &Tensor,
        o_proj: &ProjLinear,
        cfg: &SelfAttnConfig,
        pos_offset: usize,
    ) -> Tensor {
        use candle_core::DType;
        let (b, l, _) = hidden.dims3().unwrap();
        let h = cfg.num_heads;
        let kv = cfg.num_kv_heads;
        let d = cfg.head_dim;
        let q_section_dim = cfg.q_section_dim();
        let kv_out = cfg.kv_out_dim();

        let qkv = qkv_proj.forward(hidden).unwrap();
        let last = qkv.dims().len() - 1;
        let (q_raw, gate_flat) = if cfg.attn_output_gate {
            let q_section = qkv.narrow(last, 0, q_section_dim).unwrap();
            let q_split = q_section.reshape((b, l, h, 2 * d)).unwrap();
            let q = q_split.narrow(D::Minus1, 0, d).unwrap().contiguous().unwrap();
            let g = q_split
                .narrow(D::Minus1, d, d)
                .unwrap()
                .contiguous()
                .unwrap()
                .reshape((b, l, h * d))
                .unwrap();
            (q, Some(g))
        } else {
            let q = qkv
                .narrow(last, 0, q_section_dim)
                .unwrap()
                .reshape((b, l, h, d))
                .unwrap();
            (q, None)
        };
        let k_raw = qkv
            .narrow(last, q_section_dim, kv_out)
            .unwrap()
            .reshape((b, l, kv, d))
            .unwrap();
        let v_raw = qkv
            .narrow(last, q_section_dim + kv_out, kv_out)
            .unwrap()
            .reshape((b, l, kv, d))
            .unwrap();

        let block_cfg = cfg.block_config();
        let attn_out = candle_attn_block(
            &q_raw, &k_raw, &v_raw, q_norm_gamma, k_norm_gamma, &block_cfg, pos_offset,
        );
        let attn_flat = attn_out.reshape((b, l, h * d)).unwrap();
        let _ = DType::F32; // ensure import use
        let gated = if let Some(gate) = gate_flat {
            let sig = candle_nn::ops::sigmoid(&gate).unwrap();
            (attn_flat * sig).unwrap()
        } else {
            attn_flat
        };
        o_proj.forward(&gated).unwrap()
    }

    #[test]
    fn forward_self_attn_matches_candle_no_gate() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();
        let device = Device::Cpu;

        let cfg = SelfAttnConfig {
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 16,
            rotary_dim: 8,
            rms_eps: 1e-6,
            rope_theta: 10_000.0,
            attn_output_gate: false,
            apply_causal: true,
        };
        let b = 1;
        let l = 4;
        let pos_offset = 0;
        let hidden_dim = 32;

        let hidden_v: Vec<f32> = (0..b * l * hidden_dim)
            .map(|i| ((i as f32) * 0.011).sin() * 1.2)
            .collect();
        let hidden = Tensor::from_vec(hidden_v, (b, l, hidden_dim), &device).unwrap();

        let q_section = cfg.q_section_dim();
        let kv_out = cfg.kv_out_dim();
        let qkv_out = q_section + 2 * kv_out;
        let qkv_proj = dense_linear(qkv_out, hidden_dim, 0xA1, &device);
        let o_proj = dense_linear(hidden_dim, cfg.num_heads * cfg.head_dim, 0xB2, &device);
        let gamma_q = Tensor::from_vec(
            (0..cfg.head_dim)
                .map(|i| 1.0 + (i as f32) * 0.01)
                .collect::<Vec<_>>(),
            (cfg.head_dim,),
            &device,
        )
        .unwrap();
        let gamma_k = Tensor::from_vec(
            (0..cfg.head_dim)
                .map(|i| 0.95 + (i as f32) * 0.011)
                .collect::<Vec<_>>(),
            (cfg.head_dim,),
            &device,
        )
        .unwrap();

        let got_t = forward_self_attn(
            &hidden, &qkv_proj, &gamma_q, &gamma_k, &o_proj, pos_offset, &cfg, &ctx, &lib,
        )
        .unwrap();
        let exp_t = candle_self_attn(
            &hidden, &qkv_proj, &gamma_q, &gamma_k, &o_proj, &cfg, pos_offset,
        );

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
            "self_attn no-gate cosine {cos} (rel={rel}, abs={max_abs})"
        );
    }

    #[test]
    fn forward_self_attn_matches_candle_with_gate() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();
        let device = Device::Cpu;

        let cfg = SelfAttnConfig {
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 16,
            rotary_dim: 8,
            rms_eps: 1e-6,
            rope_theta: 10_000.0,
            attn_output_gate: true,
            apply_causal: true,
        };
        let b = 1;
        let l = 5;
        let pos_offset = 0;
        let hidden_dim = 32;

        let hidden_v: Vec<f32> = (0..b * l * hidden_dim)
            .map(|i| ((i as f32) * 0.013).cos() * 0.9)
            .collect();
        let hidden = Tensor::from_vec(hidden_v, (b, l, hidden_dim), &device).unwrap();

        let q_section = cfg.q_section_dim();
        let kv_out = cfg.kv_out_dim();
        let qkv_out = q_section + 2 * kv_out;
        let qkv_proj = dense_linear(qkv_out, hidden_dim, 0xC3, &device);
        let o_proj = dense_linear(hidden_dim, cfg.num_heads * cfg.head_dim, 0xD4, &device);
        let gamma_q = Tensor::from_vec(
            (0..cfg.head_dim)
                .map(|i| 1.0 + (i as f32) * 0.005)
                .collect::<Vec<_>>(),
            (cfg.head_dim,),
            &device,
        )
        .unwrap();
        let gamma_k = Tensor::from_vec(
            (0..cfg.head_dim)
                .map(|i| 0.97 + (i as f32) * 0.007)
                .collect::<Vec<_>>(),
            (cfg.head_dim,),
            &device,
        )
        .unwrap();

        let got_t = forward_self_attn(
            &hidden, &qkv_proj, &gamma_q, &gamma_k, &o_proj, pos_offset, &cfg, &ctx, &lib,
        )
        .unwrap();
        let exp_t = candle_self_attn(
            &hidden, &qkv_proj, &gamma_q, &gamma_k, &o_proj, &cfg, pos_offset,
        );
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
            "self_attn with-gate cosine {cos} (rel={rel}, abs={max_abs})"
        );
    }

    #[test]
    fn forward_attn_block_matches_candle() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();

        let cfg = AttnBlockConfig {
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 16,
            rotary_dim: 8,
            rms_eps: 1e-6,
            rope_theta: 10_000.0,
            apply_causal: true,
        };
        let b = 1;
        let l = 6;
        let pos_offset = 0;
        cfg.validate().unwrap();

        // Synthetic inputs (BLHD).
        let q_raw_v: Vec<f32> = (0..b * l * cfg.num_heads * cfg.head_dim)
            .map(|i| ((i as f32) * 0.013).sin() * 1.4)
            .collect();
        let k_raw_v: Vec<f32> = (0..b * l * cfg.num_kv_heads * cfg.head_dim)
            .map(|i| (((i as f32) * 0.017) + 0.5).cos() * 1.1)
            .collect();
        let v_raw_v: Vec<f32> = (0..b * l * cfg.num_kv_heads * cfg.head_dim)
            .map(|i| (i as f32) * 0.005 - 0.3)
            .collect();
        let gamma_q_v: Vec<f32> = (0..cfg.head_dim).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let gamma_k_v: Vec<f32> = (0..cfg.head_dim).map(|i| 0.95 + (i as f32) * 0.011).collect();

        // ─── Native ─────────────────────────────────────────────────────────
        let q_raw = ctx
            .from_slice_f32(&q_raw_v, vec![b, l, cfg.num_heads, cfg.head_dim])
            .unwrap();
        let k_raw = ctx
            .from_slice_f32(&k_raw_v, vec![b, l, cfg.num_kv_heads, cfg.head_dim])
            .unwrap();
        let v_raw = ctx
            .from_slice_f32(&v_raw_v, vec![b, l, cfg.num_kv_heads, cfg.head_dim])
            .unwrap();
        let gamma_q = ctx.from_slice_f32(&gamma_q_v, vec![cfg.head_dim]).unwrap();
        let gamma_k = ctx.from_slice_f32(&gamma_k_v, vec![cfg.head_dim]).unwrap();
        let out = ctx
            .zeros(vec![b, l, cfg.num_heads, cfg.head_dim], NativeDType::F32)
            .unwrap();

        forward_attn_block(
            &ctx, &lib, &q_raw, &k_raw, &v_raw, &gamma_q, &gamma_k, pos_offset, &out, &cfg,
        )
        .unwrap();
        let got = out.to_vec_f32().unwrap();

        // ─── Candle reference ───────────────────────────────────────────────
        let device = Device::Cpu;
        let q_raw_c = Tensor::from_vec(
            q_raw_v.clone(),
            (b, l, cfg.num_heads, cfg.head_dim),
            &device,
        )
        .unwrap();
        let k_raw_c = Tensor::from_vec(
            k_raw_v.clone(),
            (b, l, cfg.num_kv_heads, cfg.head_dim),
            &device,
        )
        .unwrap();
        let v_raw_c = Tensor::from_vec(
            v_raw_v.clone(),
            (b, l, cfg.num_kv_heads, cfg.head_dim),
            &device,
        )
        .unwrap();
        let gamma_q_c =
            Tensor::from_vec(gamma_q_v.clone(), (cfg.head_dim,), &device).unwrap();
        let gamma_k_c =
            Tensor::from_vec(gamma_k_v.clone(), (cfg.head_dim,), &device).unwrap();

        let cand_out = candle_attn_block(
            &q_raw_c, &k_raw_c, &v_raw_c, &gamma_q_c, &gamma_k_c, &cfg, pos_offset,
        );
        let expected = cand_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

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
            "block vs Candle cosine {cos} (rel_max={rel}, abs_max={max_abs})"
        );
    }

    /// Single-call prefill through `forward_self_attn_with_native_cache` must
    /// match the Candle reference for all L positions. Cache starts empty;
    /// `pos_offset` derives from cache.current_seq_len() = 0.
    #[test]
    fn forward_self_attn_with_native_cache_prefill_matches_candle() {
        use crate::qwen3_5_moe_native::kv_cache::NativeKvCache;
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();
        let device = Device::Cpu;

        let cfg = SelfAttnConfig {
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 16,
            rotary_dim: 8,
            rms_eps: 1e-6,
            rope_theta: 10_000.0,
            attn_output_gate: true,
            apply_causal: true,
        };
        let b = 1;
        let l = 5;
        let hidden_dim = 32;

        let hidden_v: Vec<f32> = (0..b * l * hidden_dim)
            .map(|i| ((i as f32) * 0.013).cos() * 0.9)
            .collect();
        let hidden = Tensor::from_vec(hidden_v, (b, l, hidden_dim), &device).unwrap();

        let q_section = cfg.q_section_dim();
        let kv_out = cfg.kv_out_dim();
        let qkv_out = q_section + 2 * kv_out;
        let qkv_proj = dense_linear(qkv_out, hidden_dim, 0xE5, &device);
        let o_proj = dense_linear(hidden_dim, cfg.num_heads * cfg.head_dim, 0xF6, &device);
        let gamma_q = Tensor::from_vec(
            (0..cfg.head_dim).map(|i| 1.0 + (i as f32) * 0.005).collect::<Vec<_>>(),
            (cfg.head_dim,),
            &device,
        )
        .unwrap();
        let gamma_k = Tensor::from_vec(
            (0..cfg.head_dim).map(|i| 0.97 + (i as f32) * 0.007).collect::<Vec<_>>(),
            (cfg.head_dim,),
            &device,
        )
        .unwrap();

        let mut cache = NativeKvCache::new(&ctx, b, cfg.num_kv_heads, cfg.head_dim, l + 4).unwrap();
        let got_t = forward_self_attn_with_native_cache(
            &hidden, &qkv_proj, &gamma_q, &gamma_k, &o_proj, &cfg, &ctx, &lib, &mut cache,
        )
        .unwrap();
        assert_eq!(cache.current_seq_len(), l);

        let exp_t = candle_self_attn(
            &hidden, &qkv_proj, &gamma_q, &gamma_k, &o_proj, &cfg, 0,
        );
        let got = got_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = exp_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let cos = cosine(&got, &expected);
        let max_abs = got
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            cos > 0.999,
            "prefill via native cache cosine {cos} (max_abs={max_abs})"
        );
    }

    /// Multi-step decode: feed tokens one at a time through the native cache and
    /// verify the cumulative output matches Candle's full-prefill reference for
    /// the same sequence. This is the decode hot-path correctness gate.
    #[test]
    fn forward_self_attn_with_native_cache_token_by_token_matches_candle() {
        use crate::qwen3_5_moe_native::kv_cache::NativeKvCache;
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let lib = KernelLib::new(&ctx).unwrap();
        let device = Device::Cpu;

        let cfg = SelfAttnConfig {
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 16,
            rotary_dim: 8,
            rms_eps: 1e-6,
            rope_theta: 10_000.0,
            attn_output_gate: true,
            apply_causal: true,
        };
        let b = 1;
        let total_l = 4;
        let hidden_dim = 32;

        let hidden_v: Vec<f32> = (0..b * total_l * hidden_dim)
            .map(|i| ((i as f32) * 0.017).sin() * 1.1)
            .collect();
        let hidden_full = Tensor::from_vec(
            hidden_v.clone(),
            (b, total_l, hidden_dim),
            &device,
        )
        .unwrap();

        let q_section = cfg.q_section_dim();
        let kv_out = cfg.kv_out_dim();
        let qkv_out = q_section + 2 * kv_out;
        let qkv_proj = dense_linear(qkv_out, hidden_dim, 0x07, &device);
        let o_proj = dense_linear(hidden_dim, cfg.num_heads * cfg.head_dim, 0x18, &device);
        let gamma_q = Tensor::from_vec(
            (0..cfg.head_dim).map(|i| 1.0 + (i as f32) * 0.003).collect::<Vec<_>>(),
            (cfg.head_dim,),
            &device,
        )
        .unwrap();
        let gamma_k = Tensor::from_vec(
            (0..cfg.head_dim).map(|i| 0.95 + (i as f32) * 0.004).collect::<Vec<_>>(),
            (cfg.head_dim,),
            &device,
        )
        .unwrap();

        // Native: emit one token at a time into the same cache.
        let mut cache =
            NativeKvCache::new(&ctx, b, cfg.num_kv_heads, cfg.head_dim, total_l + 2).unwrap();
        let mut step_outs: Vec<Vec<f32>> = Vec::with_capacity(total_l);
        for t in 0..total_l {
            // Slice [B, t..t+1, hidden] off the full hidden tensor.
            let h_t = hidden_full
                .narrow(1, t, 1)
                .unwrap()
                .contiguous()
                .unwrap();
            let out_t = forward_self_attn_with_native_cache(
                &h_t, &qkv_proj, &gamma_q, &gamma_k, &o_proj, &cfg, &ctx, &lib, &mut cache,
            )
            .unwrap();
            assert_eq!(out_t.dims(), &[b, 1, hidden_dim]);
            step_outs.push(out_t.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        }
        assert_eq!(cache.current_seq_len(), total_l);

        // Candle reference: full prefill of the same sequence.
        let exp_t = candle_self_attn(
            &hidden_full, &qkv_proj, &gamma_q, &gamma_k, &o_proj, &cfg, 0,
        );
        // exp_t has shape [B, total_l, hidden]; compare per-step.
        for t in 0..total_l {
            let exp_step = exp_t
                .narrow(1, t, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let cos = cosine(&step_outs[t], &exp_step);
            let max_abs = step_outs[t]
                .iter()
                .zip(exp_step.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                cos > 0.999,
                "decode step {t} cosine {cos} (max_abs={max_abs})"
            );
        }
    }
}
