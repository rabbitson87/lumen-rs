//! Candle `Linear`-shaped wrapper around [`Mxfp4Weight`].
//!
//! Keeps MXFP4 weights resident on the GPU and exposes `forward(&Tensor) -> Tensor`
//! so transformer layers can call it interchangeably with `candle_nn::Linear`.
//!
//! v1: CPU roundtrip (flatten → host → GPU matmul → host → Tensor). The existing
//! zero-copy Metal-buffer path in [`crate::candle_integration`] can be layered on
//! later once the end-to-end loader path is proven.

use std::sync::Arc;

use anyhow::Result;
use candle_core::{DType, Device, Storage, Tensor};
use candle_nn::{Linear, Module};

use crate::metal::Buffer;

use crate::mxfp4_gpu::{MxFp4Context, Mxfp4Weight};

/// Extract the Metal buffer + byte offset backing a Candle tensor, transmuting between the
/// two `metal` crate versions (Candle's vs ours — same underlying `objc::runtime::Object`,
/// ABI-identical `crate::metal::Buffer` wrapper).
///
/// Returns `None` if the tensor is not Metal-resident (e.g. CPU device). Safe because
/// `Mmap`-equivalent objc objects live longer than the borrow; the caller must not hold the
/// returned reference past the tensor's lifetime.
fn metal_buffer_of(t: &Tensor) -> Option<(&crate::metal::Buffer, u64)> {
    let (storage_guard, layout) = t.storage_and_layout();
    match &*storage_guard {
        Storage::Metal(ms) => {
            let offset_bytes = (layout.start_offset() * t.dtype().size_in_bytes()) as u64;
            let candle_buf = ms.buffer();
            let buf_ptr = candle_buf as *const _ as *const crate::metal::Buffer;
            let buf_ref: &crate::metal::Buffer = unsafe { &*buf_ptr };
            Some((buf_ref, offset_bytes))
        }
        _ => None,
    }
}

/// Diagnostic counters: set `LUMEN_MXFP4_TRACE=1` to get a per-token summary of how many
/// matmuls took the zero-copy Metal path vs the CPU fallback. The hot path logs nothing.
static ZERO_COPY_HITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static CPU_FALLBACK_HITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Lever H Step 2: dispatch the dense f32-weight RmsNorm-fused matmul kernel
/// against Candle tensors, sharing Candle's Metal command queue.
///
/// Computes `y = (x * rms_weight * inv_rms) @ weight^T` in one kernel — used
/// for the int8-affine-dequantized routing gate (`gate`) and
/// `shared_expert_gate` projections. Pre-allocates the output internally and
/// returns it; eliminates one external `post_attention_layernorm.forward`
/// dispatch when paired with the matching MXFP4-side rmsnorm variants.
///
/// Inputs:
///   - `weight`:    `[out, in]` f32 Candle Tensor (Linear's `.weight()`)
///   - `x_raw`:     `[..., in]` f32 — raw post-attn residual (un-normalized)
///   - `rms_weight`:`[in]` f32 — `post_attention_layernorm.weight()`
///   - `rms_eps`:   f32 (typically `1e-6`)
///
/// Returns: `[..., out]` f32 Tensor on the same Metal device as `x_raw`.
///
/// Restrictions: all tensors must be Metal-resident f32; `in` must be a
/// multiple of 4 (kernel uses float4 vector loads).
pub fn dense_f32_matmul_rmsnorm_candle_queue_into(
    ctx: &MxFp4Context,
    weight: &Tensor,
    x_raw: &Tensor,
    rms_weight: &Tensor,
    rms_eps: f32,
) -> candle_core::Result<Tensor> {
    use candle_core::Device;

    if weight.dtype() != DType::F32 {
        return Err(candle_core::Error::Msg(format!(
            "dense_f32_matmul_rmsnorm: weight dtype {:?} != F32",
            weight.dtype()
        )));
    }
    let w_dims = weight.dims();
    if w_dims.len() != 2 {
        return Err(candle_core::Error::Msg(format!(
            "dense_f32_matmul_rmsnorm: weight must be 2-D, got {:?}",
            w_dims
        )));
    }
    let (out_features, in_features) = (w_dims[0], w_dims[1]);
    if !in_features.is_multiple_of(4) {
        return Err(candle_core::Error::Msg(format!(
            "dense_f32_matmul_rmsnorm: in_features {in_features} must be multiple of 4"
        )));
    }

    let x_dims = x_raw.dims();
    if x_dims.is_empty() || x_dims[x_dims.len() - 1] != in_features {
        return Err(candle_core::Error::Msg(format!(
            "dense_f32_matmul_rmsnorm: x last dim != in_features ({:?} vs {})",
            x_dims, in_features
        )));
    }
    let batch: usize = x_dims[..x_dims.len() - 1].iter().product();
    let mut out_shape: Vec<usize> = x_dims[..x_dims.len() - 1].to_vec();
    out_shape.push(out_features);

    if batch == 0 {
        return Tensor::zeros(out_shape, DType::F32, x_raw.device());
    }

    let rms_dims = rms_weight.dims();
    if rms_dims.len() != 1 || rms_dims[0] != in_features {
        return Err(candle_core::Error::Msg(format!(
            "dense_f32_matmul_rmsnorm: rms_weight shape {:?} != [{in_features}]",
            rms_dims
        )));
    }

    let device = x_raw.device().clone();
    let md = match &device {
        Device::Metal(md) => md,
        _ => {
            return Err(candle_core::Error::Msg(
                "dense_f32_matmul_rmsnorm: Metal device required".into(),
            ));
        }
    };

    let weight_contig = if weight.is_contiguous() {
        weight.clone()
    } else {
        weight.contiguous()?
    };
    let x_f32 = if x_raw.dtype() == DType::F32 {
        x_raw.contiguous()?
    } else {
        x_raw.to_dtype(DType::F32)?.contiguous()?
    };
    let rms_w_f32 = if rms_weight.dtype() == DType::F32 {
        rms_weight.contiguous()?
    } else {
        rms_weight.to_dtype(DType::F32)?.contiguous()?
    };

    let y_out = Tensor::zeros(out_shape.clone(), DType::F32, &device)?;

    let (weight_buf, weight_offset) = metal_buffer_of(&weight_contig)
        .ok_or_else(|| candle_core::Error::Msg("no metal buffer for weight".into()))?;
    let (x_buf, x_offset) = metal_buffer_of(&x_f32)
        .ok_or_else(|| candle_core::Error::Msg("no metal buffer for x".into()))?;
    let (rms_buf, rms_offset) = metal_buffer_of(&rms_w_f32)
        .ok_or_else(|| candle_core::Error::Msg("no metal buffer for rms_weight".into()))?;
    let (y_buf, y_offset) = metal_buffer_of(&y_out)
        .ok_or_else(|| candle_core::Error::Msg("no metal buffer for y".into()))?;

    let encoder = md
        .command_encoder()
        .map_err(|e| candle_core::Error::Msg(format!("candle command_encoder: {e}")))?;
    ctx.dense_f32_matmul_rmsnorm_zero_copy_inline(
        encoder.as_ref(),
        weight_buf,
        weight_offset,
        x_buf,
        x_offset,
        rms_buf,
        rms_offset,
        y_buf,
        y_offset,
        out_features,
        in_features,
        batch,
        rms_eps,
    )
    .map_err(|e| candle_core::Error::Msg(format!("dense_f32_matmul_rmsnorm: {e}")))?;
    drop(encoder);
    Ok(y_out)
}

/// Lever L1 Step 2 (MoE-side residual fusion): `y = a + b + c` element-wise
/// over flat f32 Metal tensors. Replaces a 2-add chain (Candle binary_add
/// twice) with one Metal kernel dispatch.
///
/// All three inputs and `y` must share the same flat element count and live
/// on the same Metal device with `DType::F32`. The function does NOT
/// allocate `y` — caller passes a pre-allocated f32 Tensor of the right
/// shape. (Mirrors the in-place `*_into` convention used by
/// `moe_wsum_candle_queue_into`.)
///
/// Caller is responsible for `.contiguous()` on inputs that may have
/// strided layouts; the function checks but does not auto-contiguous-ify
/// the output `y`.
pub fn tri_add_f32_candle_queue_into(
    ctx: &MxFp4Context,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    y: &Tensor,
) -> candle_core::Result<()> {
    use candle_core::Device;

    for (name, t) in [("a", a), ("b", b), ("c", c), ("y", y)] {
        if t.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(format!(
                "tri_add_f32: {name} dtype {:?} != F32",
                t.dtype()
            )));
        }
    }
    let n: usize = a.dims().iter().product();
    for (name, t) in [("b", b), ("c", c), ("y", y)] {
        let m: usize = t.dims().iter().product();
        if m != n {
            return Err(candle_core::Error::Msg(format!(
                "tri_add_f32: {name} flat-elements {m} != a {n}"
            )));
        }
    }
    if n == 0 {
        return Ok(());
    }

    let device = a.device().clone();
    let md = match &device {
        Device::Metal(md) => md,
        _ => {
            return Err(candle_core::Error::Msg(
                "tri_add_f32: Metal device required".into(),
            ));
        }
    };

    let a_c = if a.is_contiguous() {
        a.clone()
    } else {
        a.contiguous()?
    };
    let b_c = if b.is_contiguous() {
        b.clone()
    } else {
        b.contiguous()?
    };
    let c_c = if c.is_contiguous() {
        c.clone()
    } else {
        c.contiguous()?
    };

    let (a_buf, a_off) = metal_buffer_of(&a_c)
        .ok_or_else(|| candle_core::Error::Msg("tri_add_f32: no metal buffer for a".into()))?;
    let (b_buf, b_off) = metal_buffer_of(&b_c)
        .ok_or_else(|| candle_core::Error::Msg("tri_add_f32: no metal buffer for b".into()))?;
    let (c_buf, c_off) = metal_buffer_of(&c_c)
        .ok_or_else(|| candle_core::Error::Msg("tri_add_f32: no metal buffer for c".into()))?;
    let (y_buf, y_off) = metal_buffer_of(y)
        .ok_or_else(|| candle_core::Error::Msg("tri_add_f32: no metal buffer for y".into()))?;

    let encoder = md
        .command_encoder()
        .map_err(|e| candle_core::Error::Msg(format!("candle command_encoder: {e}")))?;
    ctx.tri_add_f32_zero_copy_inline(
        encoder.as_ref(),
        a_buf,
        a_off,
        b_buf,
        b_off,
        c_buf,
        c_off,
        y_buf,
        y_off,
        n,
    )
    .map_err(|e| candle_core::Error::Msg(format!("tri_add_f32: {e}")))?;
    drop(encoder);
    Ok(())
}

/// Lever L1 Step 3.5 (drift-safe Step 3 partial): in-place
/// `y[t, h] = a[t, h] + b[t, h] * coef[t] + d[t, h]`. Caller computes
/// `coef = sigmoid(gate_logit)` via Candle (no transcendental in shader →
/// bit-identical preserved). Replaces `broadcast_mul + tri_add` chain
/// with one Metal kernel dispatch.
pub fn scalar_mul_tri_add_f32_candle_queue_into(
    ctx: &MxFp4Context,
    a: &Tensor,
    b: &Tensor,
    coef: &Tensor,
    d: &Tensor,
    y: &Tensor,
    bl: usize,
    hidden: usize,
) -> candle_core::Result<()> {
    use candle_core::Device;

    for (name, t) in [("a", a), ("b", b), ("coef", coef), ("d", d), ("y", y)] {
        if t.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(format!(
                "scalar_mul_tri_add_f32: {name} dtype {:?} != F32",
                t.dtype()
            )));
        }
    }
    let n = bl * hidden;
    for (name, t) in [("a", a), ("b", b), ("d", d), ("y", y)] {
        let m: usize = t.dims().iter().product();
        if m != n {
            return Err(candle_core::Error::Msg(format!(
                "scalar_mul_tri_add_f32: {name} flat {m} != bl*hidden {n}"
            )));
        }
    }
    let coef_n: usize = coef.dims().iter().product();
    if coef_n != bl {
        return Err(candle_core::Error::Msg(format!(
            "scalar_mul_tri_add_f32: coef flat {coef_n} != bl {bl}"
        )));
    }
    if n == 0 {
        return Ok(());
    }

    let device = a.device().clone();
    let md = match &device {
        Device::Metal(md) => md,
        _ => {
            return Err(candle_core::Error::Msg(
                "scalar_mul_tri_add_f32: Metal device required".into(),
            ));
        }
    };

    let a_c = if a.is_contiguous() {
        a.clone()
    } else {
        a.contiguous()?
    };
    let b_c = if b.is_contiguous() {
        b.clone()
    } else {
        b.contiguous()?
    };
    let coef_c = if coef.is_contiguous() {
        coef.clone()
    } else {
        coef.contiguous()?
    };
    let d_c = if d.is_contiguous() {
        d.clone()
    } else {
        d.contiguous()?
    };

    let (a_buf, a_off) = metal_buffer_of(&a_c).ok_or_else(|| {
        candle_core::Error::Msg("scalar_mul_tri_add_f32: no metal buffer for a".into())
    })?;
    let (b_buf, b_off) = metal_buffer_of(&b_c).ok_or_else(|| {
        candle_core::Error::Msg("scalar_mul_tri_add_f32: no metal buffer for b".into())
    })?;
    let (coef_buf, coef_off) = metal_buffer_of(&coef_c).ok_or_else(|| {
        candle_core::Error::Msg("scalar_mul_tri_add_f32: no metal buffer for coef".into())
    })?;
    let (d_buf, d_off) = metal_buffer_of(&d_c).ok_or_else(|| {
        candle_core::Error::Msg("scalar_mul_tri_add_f32: no metal buffer for d".into())
    })?;
    let (y_buf, y_off) = metal_buffer_of(y).ok_or_else(|| {
        candle_core::Error::Msg("scalar_mul_tri_add_f32: no metal buffer for y".into())
    })?;

    let encoder = md
        .command_encoder()
        .map_err(|e| candle_core::Error::Msg(format!("candle command_encoder: {e}")))?;
    ctx.scalar_mul_tri_add_f32_zero_copy_inline(
        encoder.as_ref(),
        a_buf,
        a_off,
        b_buf,
        b_off,
        coef_buf,
        coef_off,
        d_buf,
        d_off,
        y_buf,
        y_off,
        bl,
        hidden,
    )
    .map_err(|e| candle_core::Error::Msg(format!("scalar_mul_tri_add_f32: {e}")))?;
    drop(encoder);
    Ok(())
}

/// Lever L4 (cross-layer megafusion): in-place fused
/// `out = a + b * coef + d; attn_in = out * rms_weight * inv_rms` with
/// per-token RmsNorm reduction. Caller must pre-allocate both `out` and
/// `attn_in` (`[bl, hidden]` f32 each).
#[allow(clippy::too_many_arguments)]
pub fn scalar_mul_tri_add_rmsnorm_f32_candle_queue_into(
    ctx: &MxFp4Context,
    a: &Tensor,
    b: &Tensor,
    coef: &Tensor,
    d: &Tensor,
    rms_weight: &Tensor,
    out: &Tensor,
    attn_in: &Tensor,
    bl: usize,
    hidden: usize,
    rms_eps: f32,
) -> candle_core::Result<()> {
    use candle_core::Device;

    for (name, t) in [
        ("a", a),
        ("b", b),
        ("coef", coef),
        ("d", d),
        ("rms_weight", rms_weight),
        ("out", out),
        ("attn_in", attn_in),
    ] {
        if t.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(format!(
                "scalar_mul_tri_add_rmsnorm_f32: {name} dtype {:?} != F32",
                t.dtype()
            )));
        }
    }
    let n = bl * hidden;
    for (name, t) in [
        ("a", a),
        ("b", b),
        ("d", d),
        ("out", out),
        ("attn_in", attn_in),
    ] {
        let m: usize = t.dims().iter().product();
        if m != n {
            return Err(candle_core::Error::Msg(format!(
                "scalar_mul_tri_add_rmsnorm_f32: {name} flat {m} != bl*hidden {n}"
            )));
        }
    }
    let coef_n: usize = coef.dims().iter().product();
    if coef_n != bl {
        return Err(candle_core::Error::Msg(format!(
            "scalar_mul_tri_add_rmsnorm_f32: coef flat {coef_n} != bl {bl}"
        )));
    }
    let rms_w_n: usize = rms_weight.dims().iter().product();
    if rms_w_n != hidden {
        return Err(candle_core::Error::Msg(format!(
            "scalar_mul_tri_add_rmsnorm_f32: rms_weight flat {rms_w_n} != hidden {hidden}"
        )));
    }
    if n == 0 {
        return Ok(());
    }

    let device = a.device().clone();
    let md = match &device {
        Device::Metal(md) => md,
        _ => {
            return Err(candle_core::Error::Msg(
                "scalar_mul_tri_add_rmsnorm_f32: Metal device required".into(),
            ));
        }
    };

    let a_c = if a.is_contiguous() {
        a.clone()
    } else {
        a.contiguous()?
    };
    let b_c = if b.is_contiguous() {
        b.clone()
    } else {
        b.contiguous()?
    };
    let coef_c = if coef.is_contiguous() {
        coef.clone()
    } else {
        coef.contiguous()?
    };
    let d_c = if d.is_contiguous() {
        d.clone()
    } else {
        d.contiguous()?
    };
    let rms_w_c = if rms_weight.is_contiguous() {
        rms_weight.clone()
    } else {
        rms_weight.contiguous()?
    };

    let (a_buf, a_off) = metal_buffer_of(&a_c).ok_or_else(|| {
        candle_core::Error::Msg("scalar_mul_tri_add_rmsnorm_f32: no metal buffer for a".into())
    })?;
    let (b_buf, b_off) = metal_buffer_of(&b_c).ok_or_else(|| {
        candle_core::Error::Msg("scalar_mul_tri_add_rmsnorm_f32: no metal buffer for b".into())
    })?;
    let (coef_buf, coef_off) = metal_buffer_of(&coef_c).ok_or_else(|| {
        candle_core::Error::Msg("scalar_mul_tri_add_rmsnorm_f32: no metal buffer for coef".into())
    })?;
    let (d_buf, d_off) = metal_buffer_of(&d_c).ok_or_else(|| {
        candle_core::Error::Msg("scalar_mul_tri_add_rmsnorm_f32: no metal buffer for d".into())
    })?;
    let (rms_buf, rms_off) = metal_buffer_of(&rms_w_c).ok_or_else(|| {
        candle_core::Error::Msg(
            "scalar_mul_tri_add_rmsnorm_f32: no metal buffer for rms_weight".into(),
        )
    })?;
    let (out_buf, out_off) = metal_buffer_of(out).ok_or_else(|| {
        candle_core::Error::Msg("scalar_mul_tri_add_rmsnorm_f32: no metal buffer for out".into())
    })?;
    let (attn_in_buf, attn_in_off) = metal_buffer_of(attn_in).ok_or_else(|| {
        candle_core::Error::Msg(
            "scalar_mul_tri_add_rmsnorm_f32: no metal buffer for attn_in".into(),
        )
    })?;

    let encoder = md
        .command_encoder()
        .map_err(|e| candle_core::Error::Msg(format!("candle command_encoder: {e}")))?;
    ctx.scalar_mul_tri_add_rmsnorm_f32_zero_copy_inline(
        encoder.as_ref(),
        a_buf,
        a_off,
        b_buf,
        b_off,
        coef_buf,
        coef_off,
        d_buf,
        d_off,
        rms_buf,
        rms_off,
        out_buf,
        out_off,
        attn_in_buf,
        attn_in_off,
        bl,
        hidden,
        rms_eps,
    )
    .map_err(|e| candle_core::Error::Msg(format!("scalar_mul_tri_add_rmsnorm_f32: {e}")))?;
    drop(encoder);
    Ok(())
}

/// Snapshot + reset the trace counters. Callers (e.g. `backend::generate`) can print the
/// delta after each forward pass to locate a regressed path.
pub fn take_trace_counts() -> (usize, usize) {
    use std::sync::atomic::Ordering::Relaxed;
    let z = ZERO_COPY_HITS.swap(0, Relaxed);
    let c = CPU_FALLBACK_HITS.swap(0, Relaxed);
    (z, c)
}

/// Try the zero-copy Metal path for `y = x @ W^T`; fall back to CPU roundtrip if `x` is not
/// on Metal or not f32 contiguous. Output tensor is always f32 on `x`'s device.
fn mxfp4_matmul_tensor(
    ctx: &MxFp4Context,
    weight: &Mxfp4Weight,
    x: &Tensor,
) -> candle_core::Result<Tensor> {
    let dims = x.dims();
    if dims.is_empty() {
        return Err(candle_core::Error::Msg(
            "mxfp4 matmul requires a non-scalar input".into(),
        ));
    }
    let last = dims[dims.len() - 1];
    if last != weight.in_features {
        return Err(candle_core::Error::Msg(format!(
            "input last dim {} != in_features {}",
            last, weight.in_features
        )));
    }
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
    out_shape.push(weight.out_features);

    if batch == 0 {
        return Tensor::zeros(out_shape, DType::F32, x.device());
    }

    use std::sync::atomic::Ordering::Relaxed;

    // Fast path: input already on Metal. Run kernel in-place on device buffers.
    //
    // `LUMEN_MXFP4_FORCE_CPU=1` short-circuits this for A/B debugging.
    //
    // Cross-queue hazard: Candle's metal backend and our `MxFp4Context` hold independent
    // command queues, so writes to `x_f32` / zeros for `y` queued on Candle's side aren't
    // visible to our kernel until Candle's pending command buffer commits. We call
    // `MetalDevice::wait_until_completed()` to force Candle to flush+sync before we
    // launch our matmul. Without this the kernel reads uninitialized memory and produces
    // a zero-valued output.
    let force_cpu = std::env::var("LUMEN_MXFP4_FORCE_CPU")
        .map(|v| v == "1")
        .unwrap_or(false);
    if !force_cpu && x.device().is_metal() {
        let x_f32 = if x.dtype() == DType::F32 {
            x.contiguous()?
        } else {
            x.to_dtype(DType::F32)?.contiguous()?
        };
        if let Some((x_buf, x_offset)) = metal_buffer_of(&x_f32) {
            let y = Tensor::zeros(out_shape.clone(), DType::F32, x.device())?;
            if let Some((y_buf, y_offset)) = metal_buffer_of(&y) {
                // for decode (batch == 1) submit the
                // matmul through Candle's command encoder. Same-queue ordering removes
                // the cross-queue wait + own-queue commit/wait per call (~80 syncs/forward
                // across qkv_proj, o_proj, shared_expert.down_proj, linear_attn.{in,out}_proj
                // at decode). Default ON, opt-out via `LUMEN_DISABLE_PROJ_CANDLE_QUEUE=1`.
                // Gated on batch == 1 to mirror the MoE bl==1 lesson — prefill's
                // multi-batch t-loop pattern would thrash Candle's buffer allocator.
                let proj_candle_queue = std::env::var("LUMEN_DISABLE_PROJ_CANDLE_QUEUE")
                    .map(|v| v != "1")
                    .unwrap_or(true)
                    && batch == 1;
                if proj_candle_queue {
                    if let Device::Metal(metal_dev) = x.device() {
                        let encoder = metal_dev.command_encoder().map_err(|e| {
                            candle_core::Error::Msg(format!("candle command_encoder: {e}"))
                        })?;
                        ctx.encode_matmul_dispatch(
                            encoder.as_ref(),
                            weight,
                            x_buf,
                            x_offset,
                            y_buf,
                            y_offset,
                            batch,
                        );
                        drop(encoder);
                        ZERO_COPY_HITS.fetch_add(1, Relaxed);
                        return Ok(y);
                    }
                }
                // `LUMEN_MXFP4_SKIP_CANDLE_SYNC=1` skips the Candle-queue flush before
                // our kernel reads `x_f32` / writes `y`. Cross-queue hazard returns — only
                // set when measuring theoretical upper bound of our kernel throughput.
                let skip_sync = std::env::var("LUMEN_MXFP4_SKIP_CANDLE_SYNC")
                    .map(|v| v == "1")
                    .unwrap_or(false);
                if !skip_sync {
                    if let Device::Metal(metal_dev) = x.device() {
                        metal_dev.wait_until_completed()?;
                    }
                }
                ctx.matmul_zero_copy(weight, x_buf, x_offset, y_buf, y_offset, batch)
                    .map_err(|e| candle_core::Error::Msg(format!("mxfp4 zero-copy: {e}")))?;
                ZERO_COPY_HITS.fetch_add(1, Relaxed);
                return Ok(y);
            }
        }
        // Fall through to CPU path if buffer extraction somehow failed.
    }

    // CPU fallback: flatten → host → kernel → host → tensor.
    CPU_FALLBACK_HITS.fetch_add(1, Relaxed);
    let x_flat = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let y_vec = ctx
        .matmul_with_weight(weight, &x_flat, batch)
        .map_err(|e| candle_core::Error::Msg(format!("mxfp4 matmul: {e}")))?;
    Tensor::from_vec(y_vec, out_shape, x.device())
}

/// of [`mxfp4_matmul_tensor`].
/// Same lifecycle; produces a `DType::BF16` tensor instead of `DType::F32`.
/// The kernel still accumulates in f32 and only narrows on the final store,
/// so cosine drift vs the f32 path is bounded by bf16's 7-bit mantissa
/// (cosine ≥ 0.999, rel_max_err ≤ 1e-2 on production shapes).
///
/// Input: any shape `[..., in_features]`. The activation is force-cast to
/// f32 before dispatch (the kernel still expects an f32 `x`); a bf16-input
/// variant is Phase A.1 work once propagation through the model is wired.
fn mxfp4_matmul_tensor_bf16_out(
    ctx: &MxFp4Context,
    weight: &Mxfp4Weight,
    x: &Tensor,
) -> candle_core::Result<Tensor> {
    let dims = x.dims();
    if dims.is_empty() {
        return Err(candle_core::Error::Msg(
            "mxfp4 matmul (bf16-out) requires a non-scalar input".into(),
        ));
    }
    let last = dims[dims.len() - 1];
    if last != weight.in_features {
        return Err(candle_core::Error::Msg(format!(
            "input last dim {} != in_features {}",
            last, weight.in_features
        )));
    }
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
    out_shape.push(weight.out_features);

    if batch == 0 {
        return Tensor::zeros(out_shape, DType::BF16, x.device());
    }

    use std::sync::atomic::Ordering::Relaxed;

    let force_cpu = std::env::var("LUMEN_MXFP4_FORCE_CPU")
        .map(|v| v == "1")
        .unwrap_or(false);
    if !force_cpu && x.device().is_metal() {
        let x_f32 = if x.dtype() == DType::F32 {
            x.contiguous()?
        } else {
            x.to_dtype(DType::F32)?.contiguous()?
        };
        if let Some((x_buf, x_offset)) = metal_buffer_of(&x_f32) {
            let y = Tensor::zeros(out_shape.clone(), DType::BF16, x.device())?;
            if let Some((y_buf, y_offset)) = metal_buffer_of(&y) {
                // Mirrors `mxfp4_matmul_tensor`'s candle-queue fast path: at
                // decode (batch == 1) submit through Candle's encoder so the
                // kernel runs on the same queue as upstream Candle ops, no
                // cross-queue wait. Default ON, opt-out via the same env flag
                // as the f32 path.
                let proj_candle_queue = std::env::var("LUMEN_DISABLE_PROJ_CANDLE_QUEUE")
                    .map(|v| v != "1")
                    .unwrap_or(true)
                    && batch == 1;
                if proj_candle_queue {
                    if let Device::Metal(metal_dev) = x.device() {
                        let encoder = metal_dev.command_encoder().map_err(|e| {
                            candle_core::Error::Msg(format!("candle command_encoder: {e}"))
                        })?;
                        ctx.encode_matmul_dispatch_bf16_out(
                            encoder.as_ref(),
                            weight,
                            x_buf,
                            x_offset,
                            y_buf,
                            y_offset,
                            batch,
                        );
                        drop(encoder);
                        ZERO_COPY_HITS.fetch_add(1, Relaxed);
                        return Ok(y);
                    }
                }
                let skip_sync = std::env::var("LUMEN_MXFP4_SKIP_CANDLE_SYNC")
                    .map(|v| v == "1")
                    .unwrap_or(false);
                if !skip_sync {
                    if let Device::Metal(metal_dev) = x.device() {
                        metal_dev.wait_until_completed()?;
                    }
                }
                ctx.matmul_zero_copy_bf16_out(weight, x_buf, x_offset, y_buf, y_offset, batch)
                    .map_err(|e| candle_core::Error::Msg(format!("mxfp4 zero-copy bf16: {e}")))?;
                ZERO_COPY_HITS.fetch_add(1, Relaxed);
                return Ok(y);
            }
        }
    }

    // CPU fallback: compute in f32 (the canonical reference path) then cast
    // down to bf16. Same numerical contract as the GPU path — both narrow
    // f32 accumulators to bf16 via round-to-nearest-even.
    CPU_FALLBACK_HITS.fetch_add(1, Relaxed);
    let x_flat = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let y_vec = ctx
        .matmul_with_weight(weight, &x_flat, batch)
        .map_err(|e| candle_core::Error::Msg(format!("mxfp4 matmul (bf16 fallback): {e}")))?;
    let y_f32 = Tensor::from_vec(y_vec, out_shape, x.device())?;
    y_f32.to_dtype(DType::BF16)
}

/// Lever B L.2 (2026-04-28) bf16-input sister of [`mxfp4_matmul_tensor`].
/// Same lifecycle; consumes a `DType::BF16` activation tensor, produces a
/// `DType::F32` output. The kernel widens bf16 → f32 once during threadgroup
/// staging and accumulates in f32, so cosine vs the f32-in path is bounded
/// by the input-side bf16 mantissa truncation already absorbed by the
/// upstream RmsNorm output (≤ 6e-3 abs, validated by L.1).
///
/// Input: any shape `[..., in_features]` with dtype `BF16` or `F32`. Non-bf16
/// inputs are cast to bf16 first to mirror the production wiring (caller
/// is expected to pass a bf16 tensor produced by `MpsRmsNormBf16Out`).
fn mxfp4_matmul_tensor_bf16_in(
    ctx: &MxFp4Context,
    weight: &Mxfp4Weight,
    x: &Tensor,
) -> candle_core::Result<Tensor> {
    let dims = x.dims();
    if dims.is_empty() {
        return Err(candle_core::Error::Msg(
            "mxfp4 matmul (bf16-in) requires a non-scalar input".into(),
        ));
    }
    let last = dims[dims.len() - 1];
    if last != weight.in_features {
        return Err(candle_core::Error::Msg(format!(
            "input last dim {} != in_features {}",
            last, weight.in_features
        )));
    }
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
    out_shape.push(weight.out_features);

    if batch == 0 {
        return Tensor::zeros(out_shape, DType::F32, x.device());
    }

    use std::sync::atomic::Ordering::Relaxed;

    let force_cpu = std::env::var("LUMEN_MXFP4_FORCE_CPU")
        .map(|v| v == "1")
        .unwrap_or(false);
    if !force_cpu && x.device().is_metal() {
        let x_bf16 = if x.dtype() == DType::BF16 {
            x.contiguous()?
        } else {
            x.to_dtype(DType::BF16)?.contiguous()?
        };
        if let Some((x_buf, x_offset)) = metal_buffer_of(&x_bf16) {
            let y = Tensor::zeros(out_shape.clone(), DType::F32, x.device())?;
            if let Some((y_buf, y_offset)) = metal_buffer_of(&y) {
                // Mirrors the f32-in candle-queue fast path: at decode (batch == 1)
                // submit through Candle's encoder so the kernel runs same-queue
                // as upstream Candle ops, no cross-queue wait.
                let proj_candle_queue = std::env::var("LUMEN_DISABLE_PROJ_CANDLE_QUEUE")
                    .map(|v| v != "1")
                    .unwrap_or(true)
                    && batch == 1;
                if proj_candle_queue {
                    if let Device::Metal(metal_dev) = x.device() {
                        let encoder = metal_dev.command_encoder().map_err(|e| {
                            candle_core::Error::Msg(format!("candle command_encoder: {e}"))
                        })?;
                        ctx.encode_matmul_dispatch_bf16_in(
                            encoder.as_ref(),
                            weight,
                            x_buf,
                            x_offset,
                            y_buf,
                            y_offset,
                            batch,
                        );
                        drop(encoder);
                        ZERO_COPY_HITS.fetch_add(1, Relaxed);
                        return Ok(y);
                    }
                }
                let skip_sync = std::env::var("LUMEN_MXFP4_SKIP_CANDLE_SYNC")
                    .map(|v| v == "1")
                    .unwrap_or(false);
                if !skip_sync {
                    if let Device::Metal(metal_dev) = x.device() {
                        metal_dev.wait_until_completed()?;
                    }
                }
                ctx.matmul_zero_copy_bf16_in(weight, x_buf, x_offset, y_buf, y_offset, batch)
                    .map_err(|e| {
                        candle_core::Error::Msg(format!("mxfp4 zero-copy bf16-in: {e}"))
                    })?;
                ZERO_COPY_HITS.fetch_add(1, Relaxed);
                return Ok(y);
            }
        }
    }

    // CPU fallback: widen bf16 → f32 host-side (the matmul_with_weight kernel
    // is the canonical f32 reference). Same numerical contract — bf16 → f32
    // widening is exact, so the reference is bit-identical to feeding the
    // already-f32 widened activation through the f32-in CPU path.
    CPU_FALLBACK_HITS.fetch_add(1, Relaxed);
    let x_flat = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let y_vec = ctx
        .matmul_with_weight(weight, &x_flat, batch)
        .map_err(|e| candle_core::Error::Msg(format!("mxfp4 matmul (bf16-in fallback): {e}")))?;
    Tensor::from_vec(y_vec, out_shape, x.device())
}

/// small-out variant of [`mxfp4_matmul_tensor`]. Same input/output
/// contract, only the dispatch topology differs (1 TG = 1 row, 256 cooperating
/// threads). Caller decides when to use this — typical use is the routing gate
/// at decode time (out=256, batch=1) where v3's 32 TGs leave the GPU
/// under-occupied and microbench shows ~3× speedup.
fn mxfp4_matmul_small_out_tensor(
    ctx: &MxFp4Context,
    weight: &Mxfp4Weight,
    x: &Tensor,
) -> candle_core::Result<Tensor> {
    let dims = x.dims();
    if dims.is_empty() {
        return Err(candle_core::Error::Msg(
            "mxfp4 small_out matmul requires a non-scalar input".into(),
        ));
    }
    let last = dims[dims.len() - 1];
    if last != weight.in_features {
        return Err(candle_core::Error::Msg(format!(
            "input last dim {} != in_features {}",
            last, weight.in_features
        )));
    }
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
    out_shape.push(weight.out_features);

    if batch == 0 {
        return Tensor::zeros(out_shape, DType::F32, x.device());
    }

    use std::sync::atomic::Ordering::Relaxed;

    let force_cpu = std::env::var("LUMEN_MXFP4_FORCE_CPU")
        .map(|v| v == "1")
        .unwrap_or(false);
    if !force_cpu && x.device().is_metal() {
        let x_f32 = if x.dtype() == DType::F32 {
            x.contiguous()?
        } else {
            x.to_dtype(DType::F32)?.contiguous()?
        };
        if let Some((x_buf, x_offset)) = metal_buffer_of(&x_f32) {
            let y = Tensor::zeros(out_shape.clone(), DType::F32, x.device())?;
            if let Some((y_buf, y_offset)) = metal_buffer_of(&y) {
                // routing gate at decode is the canonical small_out
                // caller (out=256, batch=1). Submit via Candle's encoder for same-queue
                // ordering, removing the 2 syncs/call.
                let proj_candle_queue = std::env::var("LUMEN_DISABLE_PROJ_CANDLE_QUEUE")
                    .map(|v| v != "1")
                    .unwrap_or(true)
                    && batch == 1;
                if proj_candle_queue {
                    if let Device::Metal(metal_dev) = x.device() {
                        let encoder = metal_dev.command_encoder().map_err(|e| {
                            candle_core::Error::Msg(format!("candle command_encoder: {e}"))
                        })?;
                        ctx.encode_matmul_small_out_dispatch(
                            encoder.as_ref(),
                            weight,
                            x_buf,
                            x_offset,
                            y_buf,
                            y_offset,
                            batch,
                        );
                        drop(encoder);
                        ZERO_COPY_HITS.fetch_add(1, Relaxed);
                        return Ok(y);
                    }
                }
                let skip_sync = std::env::var("LUMEN_MXFP4_SKIP_CANDLE_SYNC")
                    .map(|v| v == "1")
                    .unwrap_or(false);
                if !skip_sync {
                    if let Device::Metal(metal_dev) = x.device() {
                        metal_dev.wait_until_completed()?;
                    }
                }
                ctx.matmul_small_out_zero_copy(weight, x_buf, x_offset, y_buf, y_offset, batch)
                    .map_err(|e| {
                        candle_core::Error::Msg(format!("mxfp4 small_out zero-copy: {e}"))
                    })?;
                ZERO_COPY_HITS.fetch_add(1, Relaxed);
                return Ok(y);
            }
        }
    }

    CPU_FALLBACK_HITS.fetch_add(1, Relaxed);
    let x_flat = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let y_vec = ctx
        .matmul_small_out_with_weight(weight, &x_flat, batch)
        .map_err(|e| candle_core::Error::Msg(format!("mxfp4 small_out matmul: {e}")))?;
    Tensor::from_vec(y_vec, out_shape, x.device())
}

/// `y = silu(gate) * up` where `gate = x @ W_gate^T`, `up = x @ W_up^T`.
///
/// `weight` must be the SharedExpert `gate_up_proj` shape `[2*inter, hidden]`. Output last
/// dim is `inter` (half of `weight.out_features`). Mirrors [`mxfp4_matmul_tensor`]'s
/// zero-copy fast path + CPU fallback structure.
fn mxfp4_gate_up_silu_mul_tensor(
    ctx: &MxFp4Context,
    weight: &Mxfp4Weight,
    x: &Tensor,
) -> candle_core::Result<Tensor> {
    let dims = x.dims();
    if dims.is_empty() {
        return Err(candle_core::Error::Msg(
            "mxfp4 gate_up_silu_mul requires a non-scalar input".into(),
        ));
    }
    let last = dims[dims.len() - 1];
    if last != weight.in_features {
        return Err(candle_core::Error::Msg(format!(
            "input last dim {} != in_features {}",
            last, weight.in_features
        )));
    }
    if weight.out_features % 2 != 0 {
        return Err(candle_core::Error::Msg(format!(
            "gate_up_proj weight rows must be 2*inter (even); got {}",
            weight.out_features
        )));
    }
    let inter = weight.out_features / 2;
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
    out_shape.push(inter);

    if batch == 0 {
        return Tensor::zeros(out_shape, DType::F32, x.device());
    }

    use std::sync::atomic::Ordering::Relaxed;

    let force_cpu = std::env::var("LUMEN_MXFP4_FORCE_CPU")
        .map(|v| v == "1")
        .unwrap_or(false);
    if !force_cpu && x.device().is_metal() {
        let x_f32 = if x.dtype() == DType::F32 {
            x.contiguous()?
        } else {
            x.to_dtype(DType::F32)?.contiguous()?
        };
        if let Some((x_buf, x_offset)) = metal_buffer_of(&x_f32) {
            let y = Tensor::zeros(out_shape.clone(), DType::F32, x.device())?;
            if let Some((y_buf, y_offset)) = metal_buffer_of(&y) {
                // same-queue dispatch for shared_expert
                // gate+up fused matmul at decode (batch == 1). Saves the cross-queue +
                // own-queue sync per layer (40 layers × 1 fused call = 40 saved syncs).
                let proj_candle_queue = std::env::var("LUMEN_DISABLE_PROJ_CANDLE_QUEUE")
                    .map(|v| v != "1")
                    .unwrap_or(true)
                    && batch == 1;
                if proj_candle_queue {
                    if let Device::Metal(metal_dev) = x.device() {
                        let encoder = metal_dev.command_encoder().map_err(|e| {
                            candle_core::Error::Msg(format!("candle command_encoder: {e}"))
                        })?;
                        ctx.encode_gate_up_silu_mul_dispatch(
                            encoder.as_ref(),
                            weight,
                            x_buf,
                            x_offset,
                            y_buf,
                            y_offset,
                            batch,
                        );
                        drop(encoder);
                        ZERO_COPY_HITS.fetch_add(1, Relaxed);
                        return Ok(y);
                    }
                }
                let skip_sync = std::env::var("LUMEN_MXFP4_SKIP_CANDLE_SYNC")
                    .map(|v| v == "1")
                    .unwrap_or(false);
                if !skip_sync {
                    if let Device::Metal(metal_dev) = x.device() {
                        metal_dev.wait_until_completed()?;
                    }
                }
                ctx.gate_up_silu_mul_zero_copy(weight, x_buf, x_offset, y_buf, y_offset, batch)
                    .map_err(|e| {
                        candle_core::Error::Msg(format!("mxfp4 gate_up_silu_mul zero-copy: {e}"))
                    })?;
                ZERO_COPY_HITS.fetch_add(1, Relaxed);
                return Ok(y);
            }
        }
    }

    // CPU fallback: full kernel via host bounce.
    CPU_FALLBACK_HITS.fetch_add(1, Relaxed);
    let x_flat = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let y_vec = ctx
        .gate_up_silu_mul_with_weight(weight, &x_flat, batch)
        .map_err(|e| candle_core::Error::Msg(format!("mxfp4 gate_up_silu_mul: {e}")))?;
    Tensor::from_vec(y_vec, out_shape, x.device())
}

/// of `mxfp4_matmul_small_out_tensor`.
fn mxfp4_matmul_small_out_tensor_bf16_out(
    ctx: &MxFp4Context,
    weight: &Mxfp4Weight,
    x: &Tensor,
) -> candle_core::Result<Tensor> {
    let dims = x.dims();
    if dims.is_empty() {
        return Err(candle_core::Error::Msg(
            "mxfp4 small_out matmul (bf16-out) requires a non-scalar input".into(),
        ));
    }
    let last = dims[dims.len() - 1];
    if last != weight.in_features {
        return Err(candle_core::Error::Msg(format!(
            "input last dim {} != in_features {}",
            last, weight.in_features
        )));
    }
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
    out_shape.push(weight.out_features);

    if batch == 0 {
        return Tensor::zeros(out_shape, DType::BF16, x.device());
    }

    use std::sync::atomic::Ordering::Relaxed;

    let force_cpu = std::env::var("LUMEN_MXFP4_FORCE_CPU")
        .map(|v| v == "1")
        .unwrap_or(false);
    if !force_cpu && x.device().is_metal() {
        let x_f32 = if x.dtype() == DType::F32 {
            x.contiguous()?
        } else {
            x.to_dtype(DType::F32)?.contiguous()?
        };
        if let Some((x_buf, x_offset)) = metal_buffer_of(&x_f32) {
            let y = Tensor::zeros(out_shape.clone(), DType::BF16, x.device())?;
            if let Some((y_buf, y_offset)) = metal_buffer_of(&y) {
                let proj_candle_queue = std::env::var("LUMEN_DISABLE_PROJ_CANDLE_QUEUE")
                    .map(|v| v != "1")
                    .unwrap_or(true)
                    && batch == 1;
                if proj_candle_queue {
                    if let Device::Metal(metal_dev) = x.device() {
                        let encoder = metal_dev.command_encoder().map_err(|e| {
                            candle_core::Error::Msg(format!("candle command_encoder: {e}"))
                        })?;
                        ctx.encode_matmul_small_out_dispatch_bf16_out(
                            encoder.as_ref(),
                            weight,
                            x_buf,
                            x_offset,
                            y_buf,
                            y_offset,
                            batch,
                        );
                        drop(encoder);
                        ZERO_COPY_HITS.fetch_add(1, Relaxed);
                        return Ok(y);
                    }
                }
                let skip_sync = std::env::var("LUMEN_MXFP4_SKIP_CANDLE_SYNC")
                    .map(|v| v == "1")
                    .unwrap_or(false);
                if !skip_sync {
                    if let Device::Metal(metal_dev) = x.device() {
                        metal_dev.wait_until_completed()?;
                    }
                }
                ctx.matmul_small_out_zero_copy_bf16_out(
                    weight, x_buf, x_offset, y_buf, y_offset, batch,
                )
                .map_err(|e| {
                    candle_core::Error::Msg(format!("mxfp4 small_out zero-copy bf16: {e}"))
                })?;
                ZERO_COPY_HITS.fetch_add(1, Relaxed);
                return Ok(y);
            }
        }
    }

    CPU_FALLBACK_HITS.fetch_add(1, Relaxed);
    let x_flat = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let y_vec = ctx
        .matmul_small_out_with_weight(weight, &x_flat, batch)
        .map_err(|e| {
            candle_core::Error::Msg(format!("mxfp4 small_out matmul (bf16 fallback): {e}"))
        })?;
    let y_f32 = Tensor::from_vec(y_vec, out_shape, x.device())?;
    y_f32.to_dtype(DType::BF16)
}

/// of `mxfp4_gate_up_silu_mul_tensor`.
fn mxfp4_gate_up_silu_mul_tensor_bf16_out(
    ctx: &MxFp4Context,
    weight: &Mxfp4Weight,
    x: &Tensor,
) -> candle_core::Result<Tensor> {
    let dims = x.dims();
    if dims.is_empty() {
        return Err(candle_core::Error::Msg(
            "mxfp4 gate_up_silu_mul (bf16-out) requires a non-scalar input".into(),
        ));
    }
    let last = dims[dims.len() - 1];
    if last != weight.in_features {
        return Err(candle_core::Error::Msg(format!(
            "input last dim {} != in_features {}",
            last, weight.in_features
        )));
    }
    if weight.out_features % 2 != 0 {
        return Err(candle_core::Error::Msg(format!(
            "gate_up_proj weight rows must be 2*inter (even); got {}",
            weight.out_features
        )));
    }
    let inter = weight.out_features / 2;
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
    out_shape.push(inter);

    if batch == 0 {
        return Tensor::zeros(out_shape, DType::BF16, x.device());
    }

    use std::sync::atomic::Ordering::Relaxed;

    let force_cpu = std::env::var("LUMEN_MXFP4_FORCE_CPU")
        .map(|v| v == "1")
        .unwrap_or(false);
    if !force_cpu && x.device().is_metal() {
        let x_f32 = if x.dtype() == DType::F32 {
            x.contiguous()?
        } else {
            x.to_dtype(DType::F32)?.contiguous()?
        };
        if let Some((x_buf, x_offset)) = metal_buffer_of(&x_f32) {
            let y = Tensor::zeros(out_shape.clone(), DType::BF16, x.device())?;
            if let Some((y_buf, y_offset)) = metal_buffer_of(&y) {
                let proj_candle_queue = std::env::var("LUMEN_DISABLE_PROJ_CANDLE_QUEUE")
                    .map(|v| v != "1")
                    .unwrap_or(true)
                    && batch == 1;
                if proj_candle_queue {
                    if let Device::Metal(metal_dev) = x.device() {
                        let encoder = metal_dev.command_encoder().map_err(|e| {
                            candle_core::Error::Msg(format!("candle command_encoder: {e}"))
                        })?;
                        ctx.encode_gate_up_silu_mul_dispatch_bf16_out(
                            encoder.as_ref(),
                            weight,
                            x_buf,
                            x_offset,
                            y_buf,
                            y_offset,
                            batch,
                        );
                        drop(encoder);
                        ZERO_COPY_HITS.fetch_add(1, Relaxed);
                        return Ok(y);
                    }
                }
                let skip_sync = std::env::var("LUMEN_MXFP4_SKIP_CANDLE_SYNC")
                    .map(|v| v == "1")
                    .unwrap_or(false);
                if !skip_sync {
                    if let Device::Metal(metal_dev) = x.device() {
                        metal_dev.wait_until_completed()?;
                    }
                }
                ctx.gate_up_silu_mul_zero_copy_bf16_out(
                    weight, x_buf, x_offset, y_buf, y_offset, batch,
                )
                .map_err(|e| {
                    candle_core::Error::Msg(format!("mxfp4 gate_up_silu_mul zero-copy bf16: {e}"))
                })?;
                ZERO_COPY_HITS.fetch_add(1, Relaxed);
                return Ok(y);
            }
        }
    }

    CPU_FALLBACK_HITS.fetch_add(1, Relaxed);
    let x_flat = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let y_vec = ctx
        .gate_up_silu_mul_with_weight(weight, &x_flat, batch)
        .map_err(|e| {
            candle_core::Error::Msg(format!("mxfp4 gate_up_silu_mul (bf16 fallback): {e}"))
        })?;
    let y_f32 = Tensor::from_vec(y_vec, out_shape, x.device())?;
    y_f32.to_dtype(DType::BF16)
}

/// Drop-in replacement for `candle_nn::Linear` whose weight is MXFP4-quantized on GPU.
pub struct Mxfp4Linear {
    weight: Mxfp4Weight,
    bias: Option<Tensor>,
    ctx: Arc<MxFp4Context>,
}

impl Mxfp4Linear {
    pub fn new(weight: Mxfp4Weight, bias: Option<Tensor>, ctx: Arc<MxFp4Context>) -> Self {
        Self { weight, bias, ctx }
    }

    pub fn out_features(&self) -> usize {
        self.weight.out_features
    }

    pub fn in_features(&self) -> usize {
        self.weight.in_features
    }

    pub fn device_bytes(&self) -> usize {
        self.weight.approx_bytes()
    }

    /// `y = x @ W^T (+ bias)` with W held on GPU in MXFP4 format.
    ///
    /// Input: any shape `[..., in_features]`. Output: `[..., out_features]`.
    ///
    /// When `x` lives on Metal (the production path), dispatches a fused dequant+matmul kernel
    /// directly against the device buffers — no CPU roundtrip. Falls back to a staged
    /// flatten→host→kernel→host→tensor path for non-Metal inputs (unit tests).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y =
            mxfp4_matmul_tensor(&self.ctx, &self.weight, x).map_err(|e| anyhow::anyhow!("{e}"))?;
        let y = match &self.bias {
            Some(b) => y.broadcast_add(b).map_err(|e| anyhow::anyhow!("{e}"))?,
            None => y,
        };
        Ok(y)
    }

    /// bf16-output variant of [`Self::forward`].
    /// Same compute path, but the returned tensor has `DType::BF16` so
    /// downstream Candle ops process activations in 16-bit. Bias (if any)
    /// is cast to bf16 to match the output dtype before broadcast-add.
    ///
    /// Numerical contract: cosine ≥ 0.999 vs `forward` on production shapes
    /// (validated by `tests/mxfp4_bf16_out_parity.rs`). The accumulator
    /// remains f32 inside the kernel; only the device-memory store narrows.
    pub fn forward_bf16_out(&self, x: &Tensor) -> Result<Tensor> {
        let y = mxfp4_matmul_tensor_bf16_out(&self.ctx, &self.weight, x)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let y = match &self.bias {
            Some(b) => {
                let b_bf16 = b
                    .to_dtype(DType::BF16)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                y.broadcast_add(&b_bf16)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
            }
            None => y,
        };
        Ok(y)
    }

    /// Lever B L.2 (2026-04-28): bf16-input variant of [`Self::forward`].
    /// Activation is bf16 (typically produced by `MpsRmsNormBf16Out`); output
    /// is f32 — the matmul accumulator stays f32 inside the kernel, only the
    /// activation read narrows. Bias (if any) stays in its native dtype and
    /// broadcasts onto the f32 result.
    ///
    /// Numerical contract: cosine ≥ 0.9999 vs `forward(x.to_dtype(F32))` on
    /// production shapes (validated by `tests/mxfp4_bf16_in_parity.rs`).
    /// The only difference vs the f32-in path is the input-side bf16 mantissa
    /// truncation (≤ 6e-3 abs from upstream RmsNorm), which the matmul
    /// reduction folds into a single output value.
    pub fn forward_bf16_in(&self, x: &Tensor) -> Result<Tensor> {
        let y = mxfp4_matmul_tensor_bf16_in(&self.ctx, &self.weight, x)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let y = match &self.bias {
            Some(b) => y.broadcast_add(b).map_err(|e| anyhow::anyhow!("{e}"))?,
            None => y,
        };
        Ok(y)
    }

    /// for SharedExpert's gate+up projection. Returns
    /// `silu(x @ W_gate^T) * (x @ W_up^T)` as `[..., inter]`. The weight must
    /// have row count `2*inter` (gate concatenated above up); bias is silently
    /// ignored (SharedExpert's gate_up_proj has none).
    pub fn forward_gate_up_silu_mul(&self, x: &Tensor) -> Result<Tensor> {
        mxfp4_gate_up_silu_mul_tensor(&self.ctx, &self.weight, x)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// of
    /// [`Self::forward_gate_up_silu_mul`]. Output dtype is `BF16`.
    pub fn forward_gate_up_silu_mul_bf16_out(&self, x: &Tensor) -> Result<Tensor> {
        mxfp4_gate_up_silu_mul_tensor_bf16_out(&self.ctx, &self.weight, x)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// small-out variant of [`Self::forward`]. Routes through the
    /// 1-TG-per-row Metal kernel that targets shapes like the routing gate
    /// (out=256). Bias is applied identically to `forward`. Caller selects
    /// this when the shape's `out_features` is small enough that v3's
    /// `n_groups_x = ceil(out/8)` under-occupies the GPU.
    pub fn forward_small_out(&self, x: &Tensor) -> Result<Tensor> {
        let y = mxfp4_matmul_small_out_tensor(&self.ctx, &self.weight, x)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let y = match &self.bias {
            Some(b) => y.broadcast_add(b).map_err(|e| anyhow::anyhow!("{e}"))?,
            None => y,
        };
        Ok(y)
    }

    /// of [`Self::forward_small_out`].
    pub fn forward_small_out_bf16_out(&self, x: &Tensor) -> Result<Tensor> {
        let y = mxfp4_matmul_small_out_tensor_bf16_out(&self.ctx, &self.weight, x)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let y = match &self.bias {
            Some(b) => {
                let b_bf16 = b
                    .to_dtype(DType::BF16)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                y.broadcast_add(&b_bf16)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
            }
            None => y,
        };
        Ok(y)
    }

    /// Lever H Step 2: RmsNorm-fused matmul. Reads RAW x and the
    /// `post_attention_layernorm` weight; the kernel computes `inv_rms` and
    /// applies `x * rms_weight * inv_rms` before the MXFP4 v3 dot product.
    /// Used by SharedExpert.gate_up_proj when this projection is MXFP4
    /// (the load-time-fused `[2*shared_inter, hidden]` weight) and the
    /// `LUMEN_ENABLE_RMSNORM_FUSION` flag is on.
    ///
    /// Bias (if any) is broadcast-added to the output post-matmul, mirroring
    /// `Self::forward`'s contract.
    pub fn forward_with_rmsnorm(
        &self,
        x_raw: &Tensor,
        rms_weight: &Tensor,
        rms_eps: f32,
    ) -> Result<Tensor> {
        use candle_core::Device;

        let x_dims = x_raw.dims();
        if x_dims.is_empty() || x_dims[x_dims.len() - 1] != self.weight.in_features {
            return Err(anyhow::anyhow!(
                "forward_with_rmsnorm: x last dim != in_features ({:?} vs {})",
                x_dims,
                self.weight.in_features
            ));
        }
        let batch: usize = x_dims[..x_dims.len() - 1].iter().product();
        let mut out_shape: Vec<usize> = x_dims[..x_dims.len() - 1].to_vec();
        out_shape.push(self.weight.out_features);
        if batch == 0 {
            let z = Tensor::zeros(out_shape, DType::F32, x_raw.device())
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            return Ok(z);
        }

        let device = x_raw.device().clone();
        let md = match &device {
            Device::Metal(md) => md,
            _ => {
                return Err(anyhow::anyhow!(
                    "forward_with_rmsnorm: Metal device required",
                ));
            }
        };

        let x_f32 = if x_raw.dtype() == DType::F32 {
            x_raw.contiguous().map_err(|e| anyhow::anyhow!("{e}"))?
        } else {
            x_raw
                .to_dtype(DType::F32)
                .and_then(|t| t.contiguous())
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let rms_w_f32 = if rms_weight.dtype() == DType::F32 {
            rms_weight
                .contiguous()
                .map_err(|e| anyhow::anyhow!("{e}"))?
        } else {
            rms_weight
                .to_dtype(DType::F32)
                .and_then(|t| t.contiguous())
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let y_out = Tensor::zeros(out_shape.clone(), DType::F32, &device)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let (x_buf, x_offset) = metal_buffer_of(&x_f32)
            .ok_or_else(|| anyhow::anyhow!("forward_with_rmsnorm: no metal buffer for x"))?;
        let (rms_buf, rms_offset) = metal_buffer_of(&rms_w_f32).ok_or_else(|| {
            anyhow::anyhow!("forward_with_rmsnorm: no metal buffer for rms_weight")
        })?;
        let (y_buf, y_offset) = metal_buffer_of(&y_out)
            .ok_or_else(|| anyhow::anyhow!("forward_with_rmsnorm: no metal buffer for y"))?;

        let encoder = md
            .command_encoder()
            .map_err(|e| anyhow::anyhow!("candle command_encoder: {e}"))?;

        // Lever H Step 3 retry (2026-04-28): auto-pick topology variant
        // based on `out_features`. Three tiers:
        //   - out < 8192   → small  (8 rows/TG, 256 threads)
        //                    Tuned for moe gate_up (out=4096). Step 2 sweet
        //                    spot, σ +44~49 STRONG WIN.
        //   - 8192 ≤ out < 12288 → large (16 rows/TG, 512 threads)
        //                    Halves TG count for qkv (out=9216, 576 TGs vs
        //                    1152). Tier-1 retry: σ -3.34 mild regression
        //                    (74% improvement vs small at this size).
        //   - out ≥ 12288  → xlarge (32 rows/TG, 1024 threads)
        //                    Quarters TG count for in_proj (out=12352, 386
        //                    TGs vs 1544). Apple Silicon max threadgroup.
        //
        // The variant pick is internal — the Step 3 lever in layer.rs is
        // still default OFF (`LUMEN_ENABLE_INPUT_RMSNORM_FUSION=1` to opt
        // in). The kernels themselves are bit-identical across tiers (parity
        // tests pass for all 3); only dispatch grid + reduce_buf size differ.
        const LARGE_OUT_THRESHOLD: usize = 8192;
        const XLARGE_OUT_THRESHOLD: usize = 12288;
        if self.weight.out_features >= XLARGE_OUT_THRESHOLD {
            self.ctx.matmul_f32_v3_rmsnorm_xlarge_zero_copy_inline(
                encoder.as_ref(),
                self.weight.packed_buffer(),
                self.weight.scales_buffer(),
                x_buf,
                x_offset,
                rms_buf,
                rms_offset,
                y_buf,
                y_offset,
                self.weight.out_features,
                self.weight.in_features,
                batch,
                rms_eps,
            )?;
        } else if self.weight.out_features >= LARGE_OUT_THRESHOLD {
            self.ctx.matmul_f32_v3_rmsnorm_large_zero_copy_inline(
                encoder.as_ref(),
                self.weight.packed_buffer(),
                self.weight.scales_buffer(),
                x_buf,
                x_offset,
                rms_buf,
                rms_offset,
                y_buf,
                y_offset,
                self.weight.out_features,
                self.weight.in_features,
                batch,
                rms_eps,
            )?;
        } else {
            self.ctx.matmul_f32_v3_rmsnorm_zero_copy_inline(
                encoder.as_ref(),
                self.weight.packed_buffer(),
                self.weight.scales_buffer(),
                x_buf,
                x_offset,
                rms_buf,
                rms_offset,
                y_buf,
                y_offset,
                self.weight.out_features,
                self.weight.in_features,
                batch,
                rms_eps,
            )?;
        }
        drop(encoder);

        let y = match &self.bias {
            Some(b) => y_out.broadcast_add(b).map_err(|e| anyhow::anyhow!("{e}"))?,
            None => y_out,
        };
        Ok(y)
    }

    /// Lever L1 (residual fusion): `y = x @ W^T + residual`. Folds the
    /// downstream Tensor `+` add into the matmul kernel's tail. `residual`
    /// must broadcast cleanly to the output shape (typically same shape).
    /// f32-input / f32-output only. Bias (if any) is broadcast-added after.
    pub fn forward_with_residual_f32(&self, x: &Tensor, residual: &Tensor) -> Result<Tensor> {
        use candle_core::Device;

        let x_dims = x.dims();
        if x_dims.is_empty() || x_dims[x_dims.len() - 1] != self.weight.in_features {
            return Err(anyhow::anyhow!(
                "forward_with_residual_f32: x last dim != in_features ({:?} vs {})",
                x_dims,
                self.weight.in_features
            ));
        }
        let batch: usize = x_dims[..x_dims.len() - 1].iter().product();
        let mut out_shape: Vec<usize> = x_dims[..x_dims.len() - 1].to_vec();
        out_shape.push(self.weight.out_features);
        if batch == 0 {
            let z = Tensor::zeros(out_shape, DType::F32, x.device())
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            return Ok(z);
        }

        // Residual must have the same flat layout as the output: [batch, out_features].
        let residual_dims = residual.dims();
        let residual_last = residual_dims
            .last()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("forward_with_residual_f32: residual is rank 0"))?;
        if residual_last != self.weight.out_features {
            return Err(anyhow::anyhow!(
                "forward_with_residual_f32: residual last dim {} != out_features {}",
                residual_last,
                self.weight.out_features
            ));
        }
        let residual_batch: usize = residual_dims[..residual_dims.len() - 1].iter().product();
        if residual_batch != batch {
            return Err(anyhow::anyhow!(
                "forward_with_residual_f32: residual flat batch {} != x flat batch {}",
                residual_batch,
                batch
            ));
        }

        let device = x.device().clone();
        let md = match &device {
            Device::Metal(md) => md,
            _ => {
                return Err(anyhow::anyhow!(
                    "forward_with_residual_f32: Metal device required",
                ));
            }
        };

        let x_f32 = if x.dtype() == DType::F32 {
            x.contiguous().map_err(|e| anyhow::anyhow!("{e}"))?
        } else {
            x.to_dtype(DType::F32)
                .and_then(|t| t.contiguous())
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let residual_f32 = if residual.dtype() == DType::F32 {
            residual.contiguous().map_err(|e| anyhow::anyhow!("{e}"))?
        } else {
            residual
                .to_dtype(DType::F32)
                .and_then(|t| t.contiguous())
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let y_out = Tensor::zeros(out_shape.clone(), DType::F32, &device)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let (x_buf, x_offset) = metal_buffer_of(&x_f32)
            .ok_or_else(|| anyhow::anyhow!("forward_with_residual_f32: no metal buffer for x"))?;
        let (res_buf, res_offset) = metal_buffer_of(&residual_f32).ok_or_else(|| {
            anyhow::anyhow!("forward_with_residual_f32: no metal buffer for residual")
        })?;
        let (y_buf, y_offset) = metal_buffer_of(&y_out)
            .ok_or_else(|| anyhow::anyhow!("forward_with_residual_f32: no metal buffer for y"))?;

        let encoder = md
            .command_encoder()
            .map_err(|e| anyhow::anyhow!("candle command_encoder: {e}"))?;

        self.ctx.matmul_f32_v3_residual_zero_copy_inline(
            encoder.as_ref(),
            self.weight.packed_buffer(),
            self.weight.scales_buffer(),
            x_buf,
            x_offset,
            y_buf,
            y_offset,
            res_buf,
            res_offset,
            self.weight.out_features,
            self.weight.in_features,
            batch,
        )?;
        drop(encoder);

        let y = match &self.bias {
            Some(b) => y_out.broadcast_add(b).map_err(|e| anyhow::anyhow!("{e}"))?,
            None => y_out,
        };
        Ok(y)
    }

    /// Device the bias/output tensors live on (falls back to CPU if no bias set).
    pub fn bias_device(&self) -> Option<&Device> {
        self.bias.as_ref().map(|t| t.device())
    }

    /// Native fast path: encode `y = x @ W^T` into the caller's compute
    /// encoder. No commit, no Candle-queue sync, no Tensor allocation. Used
    /// by [`crate::qwen3_5_moe_native`]'s fused forward path to avoid the
    /// per-matmul cmd-buffer + cross-queue handshake that `forward` carries.
    ///
    /// `x_buf`/`y_buf` must outlive the encoded dispatch (they're set as
    /// argument buffers on the encoder, and the kernel reads/writes them when
    /// the cmd buffer commits). The bias is silently dropped — callers that
    /// need bias must apply it as a separate elementwise kernel.
    pub fn encode_native(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) {
        self.ctx.encode_matmul_dispatch(
            encoder,
            &self.weight,
            x_buf,
            x_offset,
            y_buf,
            y_offset,
            batch,
        );
    }

    /// Borrow the GPU-resident MXFP4 weight (for callers that want to
    /// dispatch directly through `MxFp4Context`).
    pub fn weight(&self) -> &Mxfp4Weight {
        &self.weight
    }

    /// Borrow the shared MXFP4 dispatch context.
    pub fn ctx(&self) -> &Arc<MxFp4Context> {
        &self.ctx
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProjLinear — enum dispatch for plain vs MXFP4 projections
// ─────────────────────────────────────────────────────────────────────────────

/// Dispatch wrapper so transformer layers can hold either a dense `candle_nn::Linear`
/// or a GPU-resident `Mxfp4Linear` without generics. Used for every weight slot whose
/// storage kind is one of `Plain`, `Int8Affine` (dequantized on load → dense), or
/// `Mxfp4` (kept packed on GPU).
pub enum ProjLinear {
    Dense(Linear),
    Mxfp4(Mxfp4Linear),
}

impl ProjLinear {
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(l) => l.forward(x),
            Self::Mxfp4(l) => l
                .forward(x)
                .map_err(|e| candle_core::Error::Msg(format!("mxfp4 proj: {e}"))),
        }
    }

    /// fused `silu(x @ W_gate^T) * (x @ W_up^T)` for SharedExpert.
    ///
    /// MXFP4 path dispatches a single Metal kernel that combines the matmul +
    /// silu*up. Dense path falls back to the explicit 4-step pipeline so CPU
    /// fixtures and non-quantized backends keep working unchanged.
    ///
    /// `inter` is the size of each half of the gate_up weight rows; the output's
    /// last dimension equals `inter`.
    pub fn forward_gate_up_silu_mul(
        &self,
        x: &Tensor,
        inter: usize,
    ) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(l) => {
                let combined = l.forward(x)?;
                let last = combined.dims().len() - 1;
                let gate = combined.narrow(last, 0, inter)?.contiguous()?;
                let up = combined.narrow(last, inter, inter)?.contiguous()?;
                (candle_nn::ops::silu(&gate)? * up)
            }
            Self::Mxfp4(l) => l
                .forward_gate_up_silu_mul(x)
                .map_err(|e| candle_core::Error::Msg(format!("mxfp4 gate_up_silu_mul: {e}"))),
        }
    }

    /// route through the small-out matmul kernel when this is
    /// `Mxfp4`; falls back to the dense path otherwise. Caller decides at the
    /// call site (e.g. routing gate forward).
    pub fn forward_small_out(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(l) => l.forward(x),
            Self::Mxfp4(l) => l
                .forward_small_out(x)
                .map_err(|e| candle_core::Error::Msg(format!("mxfp4 small_out: {e}"))),
        }
    }

    pub fn out_features(&self) -> usize {
        match self {
            Self::Dense(l) => l.weight().dims()[0],
            Self::Mxfp4(l) => l.out_features(),
        }
    }

    pub fn in_features(&self) -> usize {
        match self {
            Self::Dense(l) => l.weight().dims()[1],
            Self::Mxfp4(l) => l.in_features(),
        }
    }
}

impl From<Linear> for ProjLinear {
    fn from(l: Linear) -> Self {
        Self::Dense(l)
    }
}

impl From<Mxfp4Linear> for ProjLinear {
    fn from(l: Mxfp4Linear) -> Self {
        Self::Mxfp4(l)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mxfp4SwitchMlp — 256 experts held as GPU-resident MXFP4 weights
// ─────────────────────────────────────────────────────────────────────────────

/// Identifier for one of the three SwiGLU projections inside a routed expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertProj {
    /// `[moe_inter, hidden]` — input → intermediate (goes through SiLU).
    Gate,
    /// `[moe_inter, hidden]` — input → intermediate (element-wise multiplied with gate).
    Up,
    /// `[hidden, moe_inter]` — intermediate → output.
    Down,
}

/// GPU-resident `switch_mlp` weights for all routed experts.
///
/// **Unified storage (Option I, 2026-04-25):** Every expert's Gate+Up (combined on axis 0)
/// and Down weights live in **four** contiguous Metal buffers — one pair per projection —
/// instead of one `Mxfp4Weight` per expert per proj. This enables the
/// `mxfp4_matmul_moe_f32` kernel to run all k selected experts in a single dispatch by
/// indexing via `expert_indices[gid.z]` instead of serializing as k separate command
/// buffers.
///
/// Per-expert `Mxfp4Weight` views (`gate_up` / `down`) are kept alongside the unified
/// buffers for the fanout / multi_cmdbuf fallback paths. They share the underlying
/// Metal buffer via `Mxfp4Weight::from_buffers`, so this is ref-count cheap (no
/// memory duplication).
///
/// **Gate+Up fusion (Option K, retained):** Gate and Up share `[moe_inter, hidden]`
/// shape and the same input `x`. We concatenate them at load time into one combined
/// `[2*moe_inter, hidden]` weight per expert, so a single matmul dispatch produces
/// both Gate and Up outputs.
pub struct Mxfp4SwitchMlp {
    /// Unified packed nibbles, shape `[num_experts, 2*moe_inter, hidden/8]` u32 flat.
    gate_up_packed_all: Buffer,
    /// Unified E8M0 scales, shape `[num_experts, 2*moe_inter, hidden/32]` u8 flat.
    gate_up_scales_all: Buffer,
    /// Unified down packed nibbles, shape `[num_experts, hidden, moe_inter/8]` u32.
    down_packed_all: Buffer,
    /// Unified down scales, shape `[num_experts, hidden, moe_inter/32]` u8.
    down_scales_all: Buffer,
    /// Per-expert view into `gate_up_*_all` (combined `[2*moe_inter, hidden]` each).
    gate_up: Vec<Mxfp4Weight>,
    /// Per-expert view into `down_*_all`.
    down: Vec<Mxfp4Weight>,
    pub num_experts: usize,
    pub hidden: usize,
    pub moe_inter: usize,
    ctx: Arc<MxFp4Context>,
}

impl Mxfp4SwitchMlp {
    /// Borrow the shared MXFP4 dispatch context. Used by callers (e.g.
    /// `SparseMoeBlock::forward_with_rmsnorm` in moe.rs) that need to
    /// dispatch dense f32 RmsNorm-fused matmuls for non-MXFP4 callsites
    /// (routing gate, shared_expert_gate) using the same Metal context.
    pub fn ctx(&self) -> &Arc<MxFp4Context> {
        &self.ctx
    }

    /// Build from 3 dequant-ready 3D grouped tensors. Each call consumes per-expert slices
    /// of the already-uploaded packed/scale buffers and produces 3 `Mxfp4Weight`s.
    ///
    /// `gate_packed`, `up_packed`: flat `[num_experts * moe_inter * hidden / 8]` u32.
    /// `gate_scales`, `up_scales`: flat `[num_experts * moe_inter * hidden / 32]` u8.
    /// `down_*`: analogous with shape `[num_experts * hidden * moe_inter / {8,32}]`.
    ///
    /// All 3 × num_experts buffers are allocated on the GPU at construction time.
    #[allow(clippy::too_many_arguments)]
    pub fn from_host(
        ctx: Arc<MxFp4Context>,
        num_experts: usize,
        hidden: usize,
        moe_inter: usize,
        gate_packed: &[u32],
        gate_scales: &[u8],
        up_packed: &[u32],
        up_scales: &[u8],
        down_packed: &[u32],
        down_scales: &[u8],
    ) -> Result<Self> {
        let gate_row_packed = moe_inter * hidden / 8;
        let gate_row_scales = moe_inter * hidden / 32;
        let down_row_packed = hidden * moe_inter / 8;
        let down_row_scales = hidden * moe_inter / 32;

        anyhow::ensure!(
            gate_packed.len() == num_experts * gate_row_packed,
            "gate_packed length {} != {} × {}",
            gate_packed.len(),
            num_experts,
            gate_row_packed
        );
        anyhow::ensure!(
            gate_scales.len() == num_experts * gate_row_scales,
            "gate_scales length mismatch"
        );
        anyhow::ensure!(
            up_packed.len() == num_experts * gate_row_packed,
            "up_packed length mismatch"
        );
        anyhow::ensure!(
            up_scales.len() == num_experts * gate_row_scales,
            "up_scales length mismatch"
        );
        anyhow::ensure!(
            down_packed.len() == num_experts * down_row_packed,
            "down_packed length mismatch"
        );
        anyhow::ensure!(
            down_scales.len() == num_experts * down_row_scales,
            "down_scales length mismatch"
        );

        // Build unified host-side buffers by concatenating per-expert slabs, then upload
        // once per proj. The gate+up slab is `[2*moe_inter, hidden]` per expert with gate
        // rows followed by up rows (axis-0 concat). Down is already in the final layout.
        //
        // Host-side concat costs one extra copy of the weights during load; the upload
        // itself is identical in total bytes. This trades a few hundred ms of load time
        // for the ability to address all 256 experts with a single bind in the MoE
        // grouped kernel later.
        let combined_row_packed = 2 * gate_row_packed;
        let combined_row_scales = 2 * gate_row_scales;
        let mut all_gu_packed = Vec::with_capacity(num_experts * combined_row_packed);
        let mut all_gu_scales = Vec::with_capacity(num_experts * combined_row_scales);
        for e in 0..num_experts {
            let gp = &gate_packed[e * gate_row_packed..(e + 1) * gate_row_packed];
            let up = &up_packed[e * gate_row_packed..(e + 1) * gate_row_packed];
            let gs = &gate_scales[e * gate_row_scales..(e + 1) * gate_row_scales];
            let us = &up_scales[e * gate_row_scales..(e + 1) * gate_row_scales];
            all_gu_packed.extend_from_slice(gp);
            all_gu_packed.extend_from_slice(up);
            all_gu_scales.extend_from_slice(gs);
            all_gu_scales.extend_from_slice(us);
        }

        let gate_up_packed_all = ctx.ctx.buffer_with_data(&all_gu_packed);
        let gate_up_scales_all = ctx.ctx.buffer_with_data(&all_gu_scales);
        let down_packed_all = ctx.ctx.buffer_with_data(down_packed);
        let down_scales_all = ctx.ctx.buffer_with_data(down_scales);

        // Per-expert views for the fanout / multi_cmdbuf fallback paths. These share the
        // underlying Metal buffer (ref-count clone) with `gate_up_*_all` / `down_*_all` —
        // no additional GPU memory.
        let mut gate_up = Vec::with_capacity(num_experts);
        let mut down = Vec::with_capacity(num_experts);
        let gu_stride_packed_bytes = (combined_row_packed * 4) as u64;
        let gu_stride_scales_bytes = combined_row_scales as u64;
        let down_stride_packed_bytes = (down_row_packed * 4) as u64;
        let down_stride_scales_bytes = down_row_scales as u64;
        for e in 0..num_experts {
            let gu = Mxfp4Weight::from_buffers(
                gate_up_packed_all.clone(),
                (e as u64) * gu_stride_packed_bytes,
                gate_up_scales_all.clone(),
                (e as u64) * gu_stride_scales_bytes,
                2 * moe_inter,
                hidden,
            )?;
            let dw = Mxfp4Weight::from_buffers(
                down_packed_all.clone(),
                (e as u64) * down_stride_packed_bytes,
                down_scales_all.clone(),
                (e as u64) * down_stride_scales_bytes,
                hidden,
                moe_inter,
            )?;
            gate_up.push(gu);
            down.push(dw);
        }

        Ok(Self {
            gate_up_packed_all,
            gate_up_scales_all,
            down_packed_all,
            down_scales_all,
            gate_up,
            down,
            num_experts,
            hidden,
            moe_inter,
            ctx,
        })
    }

    /// Total on-device byte footprint of all expert weights.
    pub fn device_bytes(&self) -> usize {
        self.gate_up.iter().map(|w| w.approx_bytes()).sum::<usize>()
            + self.down.iter().map(|w| w.approx_bytes()).sum::<usize>()
    }

    /// `y = x @ W_{expert, proj}^T`. `x` is any shape ending in the matching in_features.
    ///
    /// Uses the zero-copy Metal path when `x` is device-resident; falls back to CPU
    /// roundtrip otherwise (tests).
    ///
    /// With gate+up fused into a combined weight, the `Gate` and `Up` projections each
    /// run a `2*moe_inter`-wide matmul and narrow the output to the first/second half.
    /// This doubles the work for a *solo* Gate or Up call (legacy sequential path) but
    /// is free when the caller uses `gate_and_up_group_big`.
    pub fn expert_matmul(
        &self,
        x: &Tensor,
        expert_idx: usize,
        proj: ExpertProj,
    ) -> candle_core::Result<Tensor> {
        if expert_idx >= self.num_experts {
            return Err(candle_core::Error::Msg(format!(
                "expert_idx {} out of range (num_experts {})",
                expert_idx, self.num_experts
            )));
        }
        match proj {
            ExpertProj::Gate => {
                let y = mxfp4_matmul_tensor(&self.ctx, &self.gate_up[expert_idx], x)?;
                y.narrow(y.dims().len() - 1, 0, self.moe_inter)
            }
            ExpertProj::Up => {
                let y = mxfp4_matmul_tensor(&self.ctx, &self.gate_up[expert_idx], x)?;
                y.narrow(y.dims().len() - 1, self.moe_inter, self.moe_inter)
            }
            ExpertProj::Down => mxfp4_matmul_tensor(&self.ctx, &self.down[expert_idx], x),
        }
    }

    /// Batched matmul — `k` expert projections sharing a single input tensor, fused into
    /// one Metal command buffer (1 commit + 1 wait for all `k` dispatches). Used by the
    /// MoE decode fast path where gate/up phases have all experts reading the same `x`.
    ///
    /// Input `x` must be Metal-resident. Returns a vector of k f32 output tensors, one per
    /// selected expert. Falls back to sequential `expert_matmul` if the input isn't on
    /// Metal (CPU tests) or buffer extraction fails.
    pub fn expert_matmul_group_same_x(
        &self,
        x: &Tensor,
        experts: &[usize],
        proj: ExpertProj,
    ) -> candle_core::Result<Vec<Tensor>> {
        self.expert_matmul_group_impl(&[x].repeat(experts.len()), experts, proj)
    }

    /// Variant of `expert_matmul_group_same_x` that returns the contiguous `[k*batch, out]`
    /// output tensor instead of `k` pre-sliced views. Lets callers fold subsequent
    /// elementwise ops (silu, mul, weighted-sum) into a single big Candle op instead of
    /// `k` per-expert ops — important on the MoE hot path where the constant-factor
    /// overhead of dispatching small Candle ops dominates.
    pub fn expert_matmul_group_same_x_big(
        &self,
        x: &Tensor,
        experts: &[usize],
        proj: ExpertProj,
    ) -> candle_core::Result<Tensor> {
        let refs = [x].repeat(experts.len());
        self.expert_matmul_group_big_impl(&refs, experts, proj)
    }

    /// Like `expert_matmul_group_same_x` but each expert has its own input tensor
    /// (used for the `Down` projection: each hidden[i] = silu(gate[i]) * up[i]).
    /// All tensors must live on the same device.
    pub fn expert_matmul_group_multi_x(
        &self,
        inputs: &[Tensor],
        experts: &[usize],
        proj: ExpertProj,
    ) -> candle_core::Result<Vec<Tensor>> {
        let refs: Vec<&Tensor> = inputs.iter().collect();
        self.expert_matmul_group_impl(&refs, experts, proj)
    }

    /// Big-tensor variant of `expert_matmul_group_multi_x`. Returns `[k*batch, out]`.
    pub fn expert_matmul_group_multi_x_big(
        &self,
        inputs: &[Tensor],
        experts: &[usize],
        proj: ExpertProj,
    ) -> candle_core::Result<Tensor> {
        let refs: Vec<&Tensor> = inputs.iter().collect();
        self.expert_matmul_group_big_impl(&refs, experts, proj)
    }

    /// Option I grouped-kernel Gate+Up: single Metal dispatch handles all k experts in
    /// parallel by looking up `gate_up_packed_all` via `expert_indices[gid.z]`.
    ///
    /// Returns `(gate_big [k*batch, moe_inter], up_big [k*batch, moe_inter])` — same
    /// shape contract as `gate_and_up_group_big` so callers are interchangeable.
    ///
    /// Compared to `gate_and_up_group_big` (multi_cmdbuf fan-out), this collapses the k
    /// independent command buffers into one kernel launch where the GPU schedules all
    /// slots concurrently on its available threadgroups. The tradeoff is a single
    /// larger grid instead of k sequentialized smaller ones.
    pub fn moe_gate_up(
        &self,
        x: &Tensor,
        experts: &[usize],
    ) -> candle_core::Result<(Tensor, Tensor)> {
        use candle_core::Device;
        use std::sync::atomic::Ordering::Relaxed;

        if experts.is_empty() {
            return Err(candle_core::Error::Msg(
                "moe_gate_up: empty expert list".into(),
            ));
        }
        for e in experts {
            if *e >= self.num_experts {
                return Err(candle_core::Error::Msg(format!(
                    "expert_idx {} out of range (num_experts {})",
                    e, self.num_experts
                )));
            }
        }

        let device = x.device().clone();
        if !device.is_metal() {
            // CPU fallback via the proven per-expert path.
            return self.gate_and_up_group_big(x, experts);
        }

        let x_f32 = if x.dtype() == DType::F32 {
            x.contiguous()?
        } else {
            x.to_dtype(DType::F32)?.contiguous()?
        };
        let in_dims = x_f32.dims();
        let batch: usize = in_dims[..in_dims.len() - 1].iter().product();
        let combined_out = 2 * self.moe_inter;
        let k = experts.len();
        let y_big = Tensor::zeros(vec![k * batch, combined_out], DType::F32, &device)?;

        if let Device::Metal(md) = &device {
            md.wait_until_completed()?;
        }

        let (y_buf, y_offset) = metal_buffer_of(&y_big).ok_or_else(|| {
            candle_core::Error::Msg("no metal buffer for moe_gate_up y_big".into())
        })?;
        let (x_buf, x_offset) = metal_buffer_of(&x_f32)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for x".into()))?;

        let indices: Vec<u32> = experts.iter().map(|&e| e as u32).collect();
        self.ctx
            .matmul_moe_zero_copy(
                &self.gate_up_packed_all,
                &self.gate_up_scales_all,
                &indices,
                x_buf,
                x_offset,
                y_buf,
                y_offset,
                combined_out,
                self.hidden,
                batch,
                true, // broadcast_x
            )
            .map_err(|e| candle_core::Error::Msg(format!("moe_gate_up: {e}")))?;
        ZERO_COPY_HITS.fetch_add(k, Relaxed);

        let gate_big = y_big.narrow(1, 0, self.moe_inter)?.contiguous()?;
        let up_big = y_big
            .narrow(1, self.moe_inter, self.moe_inter)?
            .contiguous()?;
        Ok((gate_big, up_big))
    }

    /// Option I grouped-kernel Down: `y[slot] = W_down[experts[slot]] @ hiddens[slot]`
    /// for all k experts in one dispatch.
    ///
    /// `hiddens_big` must be `[k*batch, moe_inter]` contiguous (typically the output of
    /// `silu(gate_big) * up_big`). Returns `[k*batch, hidden]`.
    pub fn moe_down(&self, hiddens_big: &Tensor, experts: &[usize]) -> candle_core::Result<Tensor> {
        use candle_core::Device;
        use std::sync::atomic::Ordering::Relaxed;

        if experts.is_empty() {
            return Err(candle_core::Error::Msg(
                "moe_down: empty expert list".into(),
            ));
        }
        for e in experts {
            if *e >= self.num_experts {
                return Err(candle_core::Error::Msg(format!(
                    "expert_idx {} out of range (num_experts {})",
                    e, self.num_experts
                )));
            }
        }

        let k = experts.len();
        let in_dims = hiddens_big.dims();
        if in_dims.len() != 2 || in_dims[1] != self.moe_inter {
            return Err(candle_core::Error::Msg(format!(
                "moe_down: hiddens_big shape {:?} != [k*batch, moe_inter={}]",
                in_dims, self.moe_inter
            )));
        }
        if in_dims[0] % k != 0 {
            return Err(candle_core::Error::Msg(format!(
                "moe_down: rows {} not divisible by k {}",
                in_dims[0], k
            )));
        }
        let batch = in_dims[0] / k;

        let device = hiddens_big.device().clone();
        if !device.is_metal() {
            // CPU fallback: per-expert matmul + cat.
            let mut per: Vec<Tensor> = Vec::with_capacity(k);
            for (slot, &e) in experts.iter().enumerate() {
                let x_slot = hiddens_big.narrow(0, slot * batch, batch)?.contiguous()?;
                per.push(self.expert_matmul(&x_slot, e, ExpertProj::Down)?);
            }
            let refs: Vec<&Tensor> = per.iter().collect();
            return Tensor::cat(&refs, 0);
        }

        let x_f32 = if hiddens_big.dtype() == DType::F32 {
            hiddens_big.contiguous()?
        } else {
            hiddens_big.to_dtype(DType::F32)?.contiguous()?
        };
        let y_big = Tensor::zeros(vec![k * batch, self.hidden], DType::F32, &device)?;

        if let Device::Metal(md) = &device {
            md.wait_until_completed()?;
        }

        let (y_buf, y_offset) = metal_buffer_of(&y_big)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for moe_down y_big".into()))?;
        let (x_buf, x_offset) = metal_buffer_of(&x_f32)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for hiddens_big".into()))?;

        let indices: Vec<u32> = experts.iter().map(|&e| e as u32).collect();
        self.ctx
            .matmul_moe_zero_copy(
                &self.down_packed_all,
                &self.down_scales_all,
                &indices,
                x_buf,
                x_offset,
                y_buf,
                y_offset,
                self.hidden,
                self.moe_inter,
                batch,
                false, // per-slot x band
            )
            .map_err(|e| candle_core::Error::Msg(format!("moe_down: {e}")))?;
        ZERO_COPY_HITS.fetch_add(k, Relaxed);

        Ok(y_big)
    }

    /// GPU-resident variant of `moe_gate_up`. The expert
    /// IDs are taken directly from `inds_slice` (a u32 GPU tensor of shape `[k]` or
    /// `[1, k]`), bypassing the host transfer that `moe_gate_up` requires from the
    /// caller. Caller still flushes Candle's queue before invoking this method.
    pub fn moe_gate_up_with_indices_buffer(
        &self,
        x: &Tensor,
        inds_slice: &Tensor,
        k: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        use candle_core::Device;
        use std::sync::atomic::Ordering::Relaxed;

        if k == 0 {
            return Err(candle_core::Error::Msg(
                "moe_gate_up_with_indices_buffer: k==0".into(),
            ));
        }
        if inds_slice.dtype() != DType::U32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_with_indices_buffer: inds dtype {:?} != U32",
                inds_slice.dtype()
            )));
        }
        let inds_elems: usize = inds_slice.dims().iter().product();
        if inds_elems != k {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_with_indices_buffer: inds_slice has {} elems, expected k={}",
                inds_elems, k
            )));
        }

        let device = x.device().clone();
        if !device.is_metal() {
            return Err(candle_core::Error::Msg(
                "moe_gate_up_with_indices_buffer: CPU device not supported".into(),
            ));
        }

        let x_f32 = if x.dtype() == DType::F32 {
            x.contiguous()?
        } else {
            x.to_dtype(DType::F32)?.contiguous()?
        };
        let in_dims = x_f32.dims();
        let batch: usize = in_dims[..in_dims.len() - 1].iter().product();
        let combined_out = 2 * self.moe_inter;
        let y_big = Tensor::zeros(vec![k * batch, combined_out], DType::F32, &device)?;

        let inds_contig = if inds_slice.is_contiguous() {
            inds_slice.clone()
        } else {
            inds_slice.contiguous()?
        };

        if let Device::Metal(md) = &device {
            md.wait_until_completed()?;
        }

        let (y_buf, y_offset) = metal_buffer_of(&y_big).ok_or_else(|| {
            candle_core::Error::Msg(
                "no metal buffer for moe_gate_up_with_indices_buffer y_big".into(),
            )
        })?;
        let (x_buf, x_offset) = metal_buffer_of(&x_f32)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for x".into()))?;
        let (inds_buf, inds_offset) = metal_buffer_of(&inds_contig)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for inds_slice".into()))?;

        self.ctx
            .matmul_moe_zero_copy_with_indices_buffer(
                &self.gate_up_packed_all,
                &self.gate_up_scales_all,
                inds_buf,
                inds_offset,
                k,
                x_buf,
                x_offset,
                y_buf,
                y_offset,
                combined_out,
                self.hidden,
                batch,
                true, // broadcast_x
            )
            .map_err(|e| {
                candle_core::Error::Msg(format!("moe_gate_up_with_indices_buffer: {e}"))
            })?;
        ZERO_COPY_HITS.fetch_add(k, Relaxed);

        let gate_big = y_big.narrow(1, 0, self.moe_inter)?.contiguous()?;
        let up_big = y_big
            .narrow(1, self.moe_inter, self.moe_inter)?
            .contiguous()?;
        Ok((gate_big, up_big))
    }

    /// Candle-queue variant of `moe_gate_up_with_indices_buffer`.
    /// Submits the dispatch through Candle's command queue (`metal_device.command_encoder()`),
    /// so the work joins Candle's command buffer pool. No `commit()` and no
    /// `wait_until_completed()` are issued by us; Candle commits when its compute pool
    /// rolls over (or when CPU-bound work demands it).
    ///
    /// Same-queue ordering means subsequent Candle reads of `y_big` are serialized after
    /// our dispatch by the driver, **eliminating the previous cross-queue round-trip**.
    /// In production decode this removes a `wait_until_completed()` per layer × 2
    /// (gate_up + down).
    pub fn moe_gate_up_with_indices_buffer_candle_queue(
        &self,
        x: &Tensor,
        inds_slice: &Tensor,
        k: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        use candle_core::Device;
        use std::sync::atomic::Ordering::Relaxed;

        if k == 0 {
            return Err(candle_core::Error::Msg(
                "moe_gate_up_with_indices_buffer_candle_queue: k==0".into(),
            ));
        }
        if inds_slice.dtype() != DType::U32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_with_indices_buffer_candle_queue: inds dtype {:?} != U32",
                inds_slice.dtype()
            )));
        }
        let inds_elems: usize = inds_slice.dims().iter().product();
        if inds_elems != k {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_with_indices_buffer_candle_queue: inds_slice {} != k {}",
                inds_elems, k
            )));
        }

        let device = x.device().clone();
        let md = match &device {
            Device::Metal(md) => md,
            _ => {
                return Err(candle_core::Error::Msg(
                    "moe_gate_up_with_indices_buffer_candle_queue: Metal device required".into(),
                ));
            }
        };

        let x_f32 = if x.dtype() == DType::F32 {
            x.contiguous()?
        } else {
            x.to_dtype(DType::F32)?.contiguous()?
        };
        let in_dims = x_f32.dims();
        let batch: usize = in_dims[..in_dims.len() - 1].iter().product();
        let combined_out = 2 * self.moe_inter;
        let y_big = Tensor::zeros(vec![k * batch, combined_out], DType::F32, &device)?;

        let inds_contig = if inds_slice.is_contiguous() {
            inds_slice.clone()
        } else {
            inds_slice.contiguous()?
        };

        let (y_buf, y_offset) = metal_buffer_of(&y_big)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for y_big".into()))?;
        let (x_buf, x_offset) = metal_buffer_of(&x_f32)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for x".into()))?;
        let (inds_buf, inds_offset) = metal_buffer_of(&inds_contig)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for inds_slice".into()))?;

        let encoder = md
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("candle command_encoder: {e}")))?;
        self.ctx
            .matmul_moe_zero_copy_with_indices_buffer_inline(
                encoder.as_ref(),
                &self.gate_up_packed_all,
                &self.gate_up_scales_all,
                inds_buf,
                inds_offset,
                k,
                x_buf,
                x_offset,
                y_buf,
                y_offset,
                combined_out,
                self.hidden,
                batch,
                true,
            )
            .map_err(|e| {
                candle_core::Error::Msg(format!(
                    "moe_gate_up_with_indices_buffer_candle_queue: {e}"
                ))
            })?;
        drop(encoder); // ends encoding, releases the encoding semaphore.
        ZERO_COPY_HITS.fetch_add(k, Relaxed);

        let gate_big = y_big.narrow(1, 0, self.moe_inter)?.contiguous()?;
        let up_big = y_big
            .narrow(1, self.moe_inter, self.moe_inter)?
            .contiguous()?;
        Ok((gate_big, up_big))
    }

    /// Candle-queue variant of `moe_down_with_indices_buffer`.
    /// See `moe_gate_up_with_indices_buffer_candle_queue` for contract details.
    pub fn moe_down_with_indices_buffer_candle_queue(
        &self,
        hiddens_big: &Tensor,
        inds_slice: &Tensor,
        k: usize,
    ) -> candle_core::Result<Tensor> {
        use candle_core::Device;
        use std::sync::atomic::Ordering::Relaxed;

        if k == 0 {
            return Err(candle_core::Error::Msg(
                "moe_down_with_indices_buffer_candle_queue: k==0".into(),
            ));
        }
        if inds_slice.dtype() != DType::U32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_with_indices_buffer_candle_queue: inds dtype {:?} != U32",
                inds_slice.dtype()
            )));
        }
        let inds_elems: usize = inds_slice.dims().iter().product();
        if inds_elems != k {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_with_indices_buffer_candle_queue: inds_slice {} != k {}",
                inds_elems, k
            )));
        }
        let in_dims = hiddens_big.dims();
        if in_dims.len() != 2 || in_dims[1] != self.moe_inter {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_with_indices_buffer_candle_queue: hiddens_big {:?} != [k*batch, {}]",
                in_dims, self.moe_inter
            )));
        }
        if in_dims[0] % k != 0 {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_with_indices_buffer_candle_queue: rows {} not divisible by k {}",
                in_dims[0], k
            )));
        }
        let batch = in_dims[0] / k;

        let device = hiddens_big.device().clone();
        let md = match &device {
            Device::Metal(md) => md,
            _ => {
                return Err(candle_core::Error::Msg(
                    "moe_down_with_indices_buffer_candle_queue: Metal device required".into(),
                ));
            }
        };

        let x_f32 = if hiddens_big.dtype() == DType::F32 {
            hiddens_big.contiguous()?
        } else {
            hiddens_big.to_dtype(DType::F32)?.contiguous()?
        };
        let y_big = Tensor::zeros(vec![k * batch, self.hidden], DType::F32, &device)?;

        let inds_contig = if inds_slice.is_contiguous() {
            inds_slice.clone()
        } else {
            inds_slice.contiguous()?
        };

        let (y_buf, y_offset) = metal_buffer_of(&y_big)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for y_big".into()))?;
        let (x_buf, x_offset) = metal_buffer_of(&x_f32)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for hiddens_big".into()))?;
        let (inds_buf, inds_offset) = metal_buffer_of(&inds_contig)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for inds_slice".into()))?;

        let encoder = md
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("candle command_encoder: {e}")))?;
        self.ctx
            .matmul_moe_zero_copy_with_indices_buffer_inline(
                encoder.as_ref(),
                &self.down_packed_all,
                &self.down_scales_all,
                inds_buf,
                inds_offset,
                k,
                x_buf,
                x_offset,
                y_buf,
                y_offset,
                self.hidden,
                self.moe_inter,
                batch,
                false,
            )
            .map_err(|e| {
                candle_core::Error::Msg(format!("moe_down_with_indices_buffer_candle_queue: {e}"))
            })?;
        drop(encoder);
        ZERO_COPY_HITS.fetch_add(k, Relaxed);

        Ok(y_big)
    }

    /// caller-provided y variant of
    /// `moe_gate_up_with_indices_buffer_candle_queue`. `y_slice` must have shape
    /// `[k * batch, 2 * moe_inter]` and be Metal-resident; the kernel writes
    /// directly into its buffer with no internal allocation. Lets the caller
    /// pre-allocate one big tensor outside a t-loop and pass per-step slices,
    /// removing per-iteration Tensor::zeros pressure on the buffer allocator.
    pub fn moe_gate_up_with_indices_buffer_candle_queue_into(
        &self,
        x: &Tensor,
        inds_slice: &Tensor,
        k: usize,
        y_slice: &Tensor,
    ) -> candle_core::Result<()> {
        use candle_core::Device;
        use std::sync::atomic::Ordering::Relaxed;

        if k == 0 {
            return Err(candle_core::Error::Msg(
                "moe_gate_up_with_indices_buffer_candle_queue_into: k==0".into(),
            ));
        }
        if inds_slice.dtype() != DType::U32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_with_indices_buffer_candle_queue_into: inds dtype {:?} != U32",
                inds_slice.dtype()
            )));
        }
        let inds_elems: usize = inds_slice.dims().iter().product();
        if inds_elems != k {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_with_indices_buffer_candle_queue_into: inds_slice {} != k {}",
                inds_elems, k
            )));
        }
        let combined_out = 2 * self.moe_inter;
        let y_dims = y_slice.dims();
        if y_dims.len() != 2 || y_dims[1] != combined_out {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_with_indices_buffer_candle_queue_into: y_slice {:?} != [.., {}]",
                y_dims, combined_out
            )));
        }
        if y_dims[0] % k != 0 {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_with_indices_buffer_candle_queue_into: y rows {} not divisible by k {}",
                y_dims[0], k
            )));
        }
        if y_slice.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_with_indices_buffer_candle_queue_into: y dtype {:?} != F32",
                y_slice.dtype()
            )));
        }

        let device = x.device().clone();
        let md = match &device {
            Device::Metal(md) => md,
            _ => {
                return Err(candle_core::Error::Msg(
                    "moe_gate_up_with_indices_buffer_candle_queue_into: Metal device required"
                        .into(),
                ));
            }
        };

        let x_f32 = if x.dtype() == DType::F32 {
            x.contiguous()?
        } else {
            x.to_dtype(DType::F32)?.contiguous()?
        };
        let in_dims = x_f32.dims();
        let batch: usize = in_dims[..in_dims.len() - 1].iter().product();
        if y_dims[0] != k * batch {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_with_indices_buffer_candle_queue_into: y rows {} != k*batch {}",
                y_dims[0],
                k * batch
            )));
        }

        let inds_contig = if inds_slice.is_contiguous() {
            inds_slice.clone()
        } else {
            inds_slice.contiguous()?
        };

        let (y_buf, y_offset) = metal_buffer_of(y_slice)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for y_slice".into()))?;
        let (x_buf, x_offset) = metal_buffer_of(&x_f32)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for x".into()))?;
        let (inds_buf, inds_offset) = metal_buffer_of(&inds_contig)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for inds_slice".into()))?;

        let encoder = md
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("candle command_encoder: {e}")))?;
        self.ctx
            .matmul_moe_zero_copy_with_indices_buffer_inline(
                encoder.as_ref(),
                &self.gate_up_packed_all,
                &self.gate_up_scales_all,
                inds_buf,
                inds_offset,
                k,
                x_buf,
                x_offset,
                y_buf,
                y_offset,
                combined_out,
                self.hidden,
                batch,
                true,
            )
            .map_err(|e| {
                candle_core::Error::Msg(format!(
                    "moe_gate_up_with_indices_buffer_candle_queue_into: {e}"
                ))
            })?;
        drop(encoder);
        ZERO_COPY_HITS.fetch_add(k, Relaxed);

        Ok(())
    }

    /// Lever A (2026-04-27): routed-grouped fused gate+up+silu*up MoE dispatch.
    /// Replaces the `moe_gate_up_with_indices_buffer_candle_queue_into` →
    /// narrow → silu → mul chain with a single kernel call. `y_slice` shape is
    /// `[k * batch, moe_inter]` (half of the non-fused
    /// `[k * batch, 2 * moe_inter]` output). Output rows hold
    /// `silu(gate) * up` directly.
    ///
    /// Reuses `gate_up_packed_all` / `gate_up_scales_all` (the gate+up combined
    /// expert weight slabs) — same buffers as the non-fused path. The kernel
    /// folds the gate/up split into per-thread offsets (gate at [0..inter), up
    /// at [inter..2*inter) per expert).
    pub fn moe_gate_up_silu_mul_with_indices_buffer_candle_queue_into(
        &self,
        x: &Tensor,
        inds_slice: &Tensor,
        k: usize,
        y_slice: &Tensor,
    ) -> candle_core::Result<()> {
        use candle_core::Device;
        use std::sync::atomic::Ordering::Relaxed;

        if k == 0 {
            return Err(candle_core::Error::Msg(
                "moe_gate_up_silu_mul_with_indices_buffer_candle_queue_into: k==0".into(),
            ));
        }
        if inds_slice.dtype() != DType::U32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_silu_mul_with_indices_buffer_candle_queue_into: inds dtype {:?} != U32",
                inds_slice.dtype()
            )));
        }
        let inds_elems: usize = inds_slice.dims().iter().product();
        if inds_elems != k {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_silu_mul_with_indices_buffer_candle_queue_into: inds_slice {} != k {}",
                inds_elems, k
            )));
        }
        let y_dims = y_slice.dims();
        if y_dims.len() != 2 || y_dims[1] != self.moe_inter {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_silu_mul_with_indices_buffer_candle_queue_into: y_slice {:?} != [.., {}]",
                y_dims, self.moe_inter
            )));
        }
        if y_dims[0] % k != 0 {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_silu_mul_with_indices_buffer_candle_queue_into: y rows {} not divisible by k {}",
                y_dims[0], k
            )));
        }
        if y_slice.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_silu_mul_with_indices_buffer_candle_queue_into: y dtype {:?} != F32",
                y_slice.dtype()
            )));
        }

        let device = x.device().clone();
        let md = match &device {
            Device::Metal(md) => md,
            _ => {
                return Err(candle_core::Error::Msg(
                    "moe_gate_up_silu_mul_with_indices_buffer_candle_queue_into: Metal device required"
                        .into(),
                ));
            }
        };

        let x_f32 = if x.dtype() == DType::F32 {
            x.contiguous()?
        } else {
            x.to_dtype(DType::F32)?.contiguous()?
        };
        let in_dims = x_f32.dims();
        let batch: usize = in_dims[..in_dims.len() - 1].iter().product();
        if y_dims[0] != k * batch {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_silu_mul_with_indices_buffer_candle_queue_into: y rows {} != k*batch {}",
                y_dims[0],
                k * batch
            )));
        }

        let inds_contig = if inds_slice.is_contiguous() {
            inds_slice.clone()
        } else {
            inds_slice.contiguous()?
        };

        let (y_buf, y_offset) = metal_buffer_of(y_slice)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for y_slice".into()))?;
        let (x_buf, x_offset) = metal_buffer_of(&x_f32)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for x".into()))?;
        let (inds_buf, inds_offset) = metal_buffer_of(&inds_contig)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for inds_slice".into()))?;

        let encoder = md
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("candle command_encoder: {e}")))?;
        self.ctx
            .matmul_moe_gate_up_silu_mul_zero_copy_with_indices_buffer_inline(
                encoder.as_ref(),
                &self.gate_up_packed_all,
                &self.gate_up_scales_all,
                inds_buf,
                inds_offset,
                k,
                x_buf,
                x_offset,
                y_buf,
                y_offset,
                self.moe_inter,
                self.hidden,
                batch,
            )
            .map_err(|e| {
                candle_core::Error::Msg(format!(
                    "moe_gate_up_silu_mul_with_indices_buffer_candle_queue_into: {e}"
                ))
            })?;
        drop(encoder);
        ZERO_COPY_HITS.fetch_add(k, Relaxed);

        Ok(())
    }

    /// Lever H Step 2: RmsNorm-fused sister of
    /// `moe_gate_up_silu_mul_with_indices_buffer_candle_queue_into`. Reads RAW
    /// x + the `post_attention_layernorm` weight; the kernel internally
    /// computes `inv_rms` cooperatively and applies
    /// `x * rms_weight * inv_rms` before the routed gate+up+silu*up matmul.
    /// Used by the routed-experts `t` loop in moe.rs when the
    /// `LUMEN_ENABLE_RMSNORM_FUSION` flag is on.
    pub fn moe_gate_up_silu_mul_rmsnorm_with_indices_buffer_candle_queue_into(
        &self,
        x_raw: &Tensor,
        rms_weight: &Tensor,
        rms_eps: f32,
        inds_slice: &Tensor,
        k: usize,
        y_slice: &Tensor,
    ) -> candle_core::Result<()> {
        use candle_core::Device;
        use std::sync::atomic::Ordering::Relaxed;

        if k == 0 {
            return Err(candle_core::Error::Msg(
                "moe_gate_up_silu_mul_rmsnorm_with_indices_buffer_candle_queue_into: k==0".into(),
            ));
        }
        if inds_slice.dtype() != DType::U32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_silu_mul_rmsnorm_…: inds dtype {:?} != U32",
                inds_slice.dtype()
            )));
        }
        let inds_elems: usize = inds_slice.dims().iter().product();
        if inds_elems != k {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_silu_mul_rmsnorm_…: inds_slice {} != k {}",
                inds_elems, k
            )));
        }
        let y_dims = y_slice.dims();
        if y_dims.len() != 2 || y_dims[1] != self.moe_inter {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_silu_mul_rmsnorm_…: y_slice {:?} != [.., {}]",
                y_dims, self.moe_inter
            )));
        }
        if y_dims[0] % k != 0 {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_silu_mul_rmsnorm_…: y rows {} not divisible by k {}",
                y_dims[0], k
            )));
        }
        if y_slice.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_silu_mul_rmsnorm_…: y dtype {:?} != F32",
                y_slice.dtype()
            )));
        }

        let device = x_raw.device().clone();
        let md = match &device {
            Device::Metal(md) => md,
            _ => {
                return Err(candle_core::Error::Msg(
                    "moe_gate_up_silu_mul_rmsnorm_…: Metal device required".into(),
                ));
            }
        };

        let x_f32 = if x_raw.dtype() == DType::F32 {
            x_raw.contiguous()?
        } else {
            x_raw.to_dtype(DType::F32)?.contiguous()?
        };
        let in_dims = x_f32.dims();
        let batch: usize = in_dims[..in_dims.len() - 1].iter().product();
        if y_dims[0] != k * batch {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_silu_mul_rmsnorm_…: y rows {} != k*batch {}",
                y_dims[0],
                k * batch
            )));
        }
        let rms_w_f32 = if rms_weight.dtype() == DType::F32 {
            rms_weight.contiguous()?
        } else {
            rms_weight.to_dtype(DType::F32)?.contiguous()?
        };

        let inds_contig = if inds_slice.is_contiguous() {
            inds_slice.clone()
        } else {
            inds_slice.contiguous()?
        };

        let (y_buf, y_offset) = metal_buffer_of(y_slice)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for y_slice".into()))?;
        let (x_buf, x_offset) = metal_buffer_of(&x_f32)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for x".into()))?;
        let (rms_buf, rms_offset) = metal_buffer_of(&rms_w_f32)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for rms_weight".into()))?;
        let (inds_buf, inds_offset) = metal_buffer_of(&inds_contig)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for inds_slice".into()))?;

        let encoder = md
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("candle command_encoder: {e}")))?;
        self.ctx
            .matmul_moe_gate_up_silu_mul_rmsnorm_zero_copy_with_indices_buffer_inline(
                encoder.as_ref(),
                &self.gate_up_packed_all,
                &self.gate_up_scales_all,
                inds_buf,
                inds_offset,
                k,
                x_buf,
                x_offset,
                rms_buf,
                rms_offset,
                y_buf,
                y_offset,
                self.moe_inter,
                self.hidden,
                batch,
                rms_eps,
            )
            .map_err(|e| {
                candle_core::Error::Msg(format!(
                    "moe_gate_up_silu_mul_rmsnorm_with_indices_buffer_candle_queue_into: {e}"
                ))
            })?;
        drop(encoder);
        ZERO_COPY_HITS.fetch_add(k, Relaxed);

        Ok(())
    }

    /// Lever D (2026-04-27): bf16-output sister of
    /// `moe_gate_up_silu_mul_with_indices_buffer_candle_queue_into`. Same
    /// contract — only `y_slice` dtype differs (BF16 instead of F32) and
    /// y_slice's underlying buffer is half the byte size.
    pub fn moe_gate_up_silu_mul_bf16out_with_indices_buffer_candle_queue_into(
        &self,
        x: &Tensor,
        inds_slice: &Tensor,
        k: usize,
        y_slice: &Tensor,
    ) -> candle_core::Result<()> {
        use candle_core::Device;
        use std::sync::atomic::Ordering::Relaxed;

        if k == 0 {
            return Err(candle_core::Error::Msg(
                "moe_gate_up_silu_mul_bf16out_with_indices_buffer_candle_queue_into: k==0".into(),
            ));
        }
        if inds_slice.dtype() != DType::U32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_silu_mul_bf16out_with_indices_buffer_candle_queue_into: inds dtype {:?} != U32",
                inds_slice.dtype()
            )));
        }
        let inds_elems: usize = inds_slice.dims().iter().product();
        if inds_elems != k {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_silu_mul_bf16out_with_indices_buffer_candle_queue_into: inds_slice {} != k {}",
                inds_elems, k
            )));
        }
        let y_dims = y_slice.dims();
        if y_dims.len() != 2 || y_dims[1] != self.moe_inter {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_silu_mul_bf16out_with_indices_buffer_candle_queue_into: y_slice {:?} != [.., {}]",
                y_dims, self.moe_inter
            )));
        }
        if y_dims[0] % k != 0 {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_silu_mul_bf16out_with_indices_buffer_candle_queue_into: y rows {} not divisible by k {}",
                y_dims[0], k
            )));
        }
        if y_slice.dtype() != DType::BF16 {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_silu_mul_bf16out_with_indices_buffer_candle_queue_into: y dtype {:?} != BF16",
                y_slice.dtype()
            )));
        }

        let device = x.device().clone();
        let md = match &device {
            Device::Metal(md) => md,
            _ => {
                return Err(candle_core::Error::Msg(
                    "moe_gate_up_silu_mul_bf16out_with_indices_buffer_candle_queue_into: Metal device required".into(),
                ));
            }
        };

        let x_f32 = if x.dtype() == DType::F32 {
            x.contiguous()?
        } else {
            x.to_dtype(DType::F32)?.contiguous()?
        };
        let in_dims = x_f32.dims();
        let batch: usize = in_dims[..in_dims.len() - 1].iter().product();
        if y_dims[0] != k * batch {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_silu_mul_bf16out_with_indices_buffer_candle_queue_into: y rows {} != k*batch {}",
                y_dims[0],
                k * batch
            )));
        }

        let inds_contig = if inds_slice.is_contiguous() {
            inds_slice.clone()
        } else {
            inds_slice.contiguous()?
        };

        let (y_buf, y_offset) = metal_buffer_of(y_slice)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for y_slice".into()))?;
        let (x_buf, x_offset) = metal_buffer_of(&x_f32)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for x".into()))?;
        let (inds_buf, inds_offset) = metal_buffer_of(&inds_contig)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for inds_slice".into()))?;

        let encoder = md
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("candle command_encoder: {e}")))?;
        self.ctx
            .matmul_moe_gate_up_silu_mul_bf16out_zero_copy_with_indices_buffer_inline(
                encoder.as_ref(),
                &self.gate_up_packed_all,
                &self.gate_up_scales_all,
                inds_buf,
                inds_offset,
                k,
                x_buf,
                x_offset,
                y_buf,
                y_offset,
                self.moe_inter,
                self.hidden,
                batch,
            )
            .map_err(|e| {
                candle_core::Error::Msg(format!(
                    "moe_gate_up_silu_mul_bf16out_with_indices_buffer_candle_queue_into: {e}"
                ))
            })?;
        drop(encoder);
        ZERO_COPY_HITS.fetch_add(k, Relaxed);

        Ok(())
    }

    /// Lever D (2026-04-27): bf16-input sister of
    /// `moe_down_with_indices_buffer_candle_queue_into`. Reads `hiddens_big`
    /// as BF16 (half the bytes of the f32 path); inner FMA + output stay F32.
    pub fn moe_down_bf16in_with_indices_buffer_candle_queue_into(
        &self,
        hiddens_big: &Tensor,
        inds_slice: &Tensor,
        k: usize,
        y_slice: &Tensor,
    ) -> candle_core::Result<()> {
        use candle_core::Device;
        use std::sync::atomic::Ordering::Relaxed;

        if k == 0 {
            return Err(candle_core::Error::Msg(
                "moe_down_bf16in_with_indices_buffer_candle_queue_into: k==0".into(),
            ));
        }
        if inds_slice.dtype() != DType::U32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_bf16in_with_indices_buffer_candle_queue_into: inds dtype {:?} != U32",
                inds_slice.dtype()
            )));
        }
        let inds_elems: usize = inds_slice.dims().iter().product();
        if inds_elems != k {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_bf16in_with_indices_buffer_candle_queue_into: inds_slice {} != k {}",
                inds_elems, k
            )));
        }
        if hiddens_big.dtype() != DType::BF16 {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_bf16in_with_indices_buffer_candle_queue_into: hiddens dtype {:?} != BF16",
                hiddens_big.dtype()
            )));
        }
        let in_dims = hiddens_big.dims();
        if in_dims.len() != 2 || in_dims[1] != self.moe_inter {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_bf16in_with_indices_buffer_candle_queue_into: hiddens_big {:?} != [k*batch, {}]",
                in_dims, self.moe_inter
            )));
        }
        if in_dims[0] % k != 0 {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_bf16in_with_indices_buffer_candle_queue_into: rows {} not divisible by k {}",
                in_dims[0], k
            )));
        }
        let batch = in_dims[0] / k;

        let y_dims = y_slice.dims();
        if y_dims.len() != 2 || y_dims[1] != self.hidden || y_dims[0] != k * batch {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_bf16in_with_indices_buffer_candle_queue_into: y_slice {:?} != [{}, {}]",
                y_dims,
                k * batch,
                self.hidden
            )));
        }
        if y_slice.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_bf16in_with_indices_buffer_candle_queue_into: y dtype {:?} != F32",
                y_slice.dtype()
            )));
        }

        let device = hiddens_big.device().clone();
        let md = match &device {
            Device::Metal(md) => md,
            _ => {
                return Err(candle_core::Error::Msg(
                    "moe_down_bf16in_with_indices_buffer_candle_queue_into: Metal device required"
                        .into(),
                ));
            }
        };

        let x_c = if hiddens_big.is_contiguous() {
            hiddens_big.clone()
        } else {
            hiddens_big.contiguous()?
        };
        let inds_c = if inds_slice.is_contiguous() {
            inds_slice.clone()
        } else {
            inds_slice.contiguous()?
        };

        let (y_buf, y_offset) = metal_buffer_of(y_slice)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for y_slice".into()))?;
        let (x_buf, x_offset) = metal_buffer_of(&x_c)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for hiddens_big".into()))?;
        let (inds_buf, inds_offset) = metal_buffer_of(&inds_c)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for inds_slice".into()))?;

        let encoder = md
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("candle command_encoder: {e}")))?;
        self.ctx
            .matmul_moe_bf16in_zero_copy_with_indices_buffer_inline(
                encoder.as_ref(),
                &self.down_packed_all,
                &self.down_scales_all,
                inds_buf,
                inds_offset,
                k,
                x_buf,
                x_offset,
                y_buf,
                y_offset,
                self.hidden,
                self.moe_inter,
                batch,
                false,
            )
            .map_err(|e| {
                candle_core::Error::Msg(format!(
                    "moe_down_bf16in_with_indices_buffer_candle_queue_into: {e}"
                ))
            })?;
        drop(encoder);
        ZERO_COPY_HITS.fetch_add(k, Relaxed);

        Ok(())
    }

    /// Lever B (2026-04-27): MoE weighted-sum dispatch on candle queue.
    /// Replaces `downs.broadcast_mul(weights).sum_keepdim(0)` with one
    /// kernel call that writes `out[r] = sum_e w[e] * downs[e, r]` directly.
    ///
    /// Inputs:
    ///   - `downs`:   F32 tensor of shape `[k, hidden]` Metal-resident.
    ///   - `weights`: F32 tensor with `k` elements (any shape that flattens to k).
    ///   - `out`:     F32 tensor of shape `[hidden]` or `[1, hidden]` Metal-resident.
    pub fn moe_wsum_candle_queue_into(
        &self,
        downs: &Tensor,
        weights: &Tensor,
        out: &Tensor,
    ) -> candle_core::Result<()> {
        use candle_core::Device;

        if downs.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_wsum_candle_queue_into: downs dtype {:?} != F32",
                downs.dtype()
            )));
        }
        if weights.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_wsum_candle_queue_into: weights dtype {:?} != F32",
                weights.dtype()
            )));
        }
        if out.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_wsum_candle_queue_into: out dtype {:?} != F32",
                out.dtype()
            )));
        }

        let downs_dims = downs.dims();
        if downs_dims.len() != 2 {
            return Err(candle_core::Error::Msg(format!(
                "moe_wsum_candle_queue_into: downs shape {:?} must be 2-D [k, hidden]",
                downs_dims
            )));
        }
        let k = downs_dims[0];
        let hidden = downs_dims[1];

        let weights_elems: usize = weights.dims().iter().product();
        if weights_elems != k {
            return Err(candle_core::Error::Msg(format!(
                "moe_wsum_candle_queue_into: weights {} != k {}",
                weights_elems, k
            )));
        }
        let out_elems: usize = out.dims().iter().product();
        if out_elems != hidden {
            return Err(candle_core::Error::Msg(format!(
                "moe_wsum_candle_queue_into: out {} != hidden {}",
                out_elems, hidden
            )));
        }

        let device = downs.device().clone();
        let md = match &device {
            Device::Metal(md) => md,
            _ => {
                return Err(candle_core::Error::Msg(
                    "moe_wsum_candle_queue_into: Metal device required".into(),
                ));
            }
        };

        let downs_c = if downs.is_contiguous() {
            downs.clone()
        } else {
            downs.contiguous()?
        };
        let weights_c = if weights.is_contiguous() {
            weights.clone()
        } else {
            weights.contiguous()?
        };

        let (downs_buf, downs_off) = metal_buffer_of(&downs_c)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for downs".into()))?;
        let (weights_buf, weights_off) = metal_buffer_of(&weights_c)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for weights".into()))?;
        let (out_buf, out_off) = metal_buffer_of(out)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for out".into()))?;

        let encoder = md
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("candle command_encoder: {e}")))?;
        self.ctx
            .moe_wsum_zero_copy_inline(
                encoder.as_ref(),
                downs_buf,
                downs_off,
                weights_buf,
                weights_off,
                out_buf,
                out_off,
                k,
                hidden,
            )
            .map_err(|e| candle_core::Error::Msg(format!("moe_wsum_candle_queue_into: {e}")))?;
        drop(encoder);

        Ok(())
    }

    /// Lever C (2026-04-27): fused MoE down matmul + weighted sum on the
    /// Candle queue. Replaces the chain
    ///   `moe_down_with_indices_buffer_candle_queue_into` (writing
    ///     `downs_big [k*batch, hidden]`) +
    ///   `moe_wsum_candle_queue_into` (reducing to `[hidden]`)
    /// with a single dispatch that folds the slot axis into an inner serial
    /// loop, eliminating `downs_big` and one kernel-launch+sync boundary.
    ///
    /// Inputs:
    ///   - `hiddens_big`: F32 Metal-resident `[k * batch, moe_inter]` (the
    ///     gate_up_silu_mul output). Same buffer the down kernel currently
    ///     consumes.
    ///   - `inds_slice`:  U32 Metal-resident `[k]` (expert indices, GPU-resident).
    ///   - `weights`:     F32 Metal-resident, `k` elements (any shape that
    ///     flattens to k). Routing weights.
    ///   - `out`:         F32 Metal-resident `[batch, hidden]` or
    ///     `[1, hidden]`. Caller-allocated.
    pub fn moe_matmul_wsum_with_indices_buffer_candle_queue_into(
        &self,
        hiddens_big: &Tensor,
        inds_slice: &Tensor,
        k: usize,
        weights: &Tensor,
        out: &Tensor,
    ) -> candle_core::Result<()> {
        use candle_core::Device;
        use std::sync::atomic::Ordering::Relaxed;

        if k == 0 {
            return Err(candle_core::Error::Msg(
                "moe_matmul_wsum_with_indices_buffer_candle_queue_into: k==0".into(),
            ));
        }
        if inds_slice.dtype() != DType::U32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_matmul_wsum_with_indices_buffer_candle_queue_into: inds dtype {:?} != U32",
                inds_slice.dtype()
            )));
        }
        let inds_elems: usize = inds_slice.dims().iter().product();
        if inds_elems != k {
            return Err(candle_core::Error::Msg(format!(
                "moe_matmul_wsum_with_indices_buffer_candle_queue_into: inds_slice {} != k {}",
                inds_elems, k
            )));
        }
        let in_dims = hiddens_big.dims();
        if in_dims.len() != 2 || in_dims[1] != self.moe_inter {
            return Err(candle_core::Error::Msg(format!(
                "moe_matmul_wsum_with_indices_buffer_candle_queue_into: hiddens_big {:?} != [k*batch, {}]",
                in_dims, self.moe_inter
            )));
        }
        if in_dims[0] % k != 0 {
            return Err(candle_core::Error::Msg(format!(
                "moe_matmul_wsum_with_indices_buffer_candle_queue_into: rows {} not divisible by k {}",
                in_dims[0], k
            )));
        }
        let batch = in_dims[0] / k;

        if hiddens_big.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_matmul_wsum_with_indices_buffer_candle_queue_into: hiddens dtype {:?} != F32",
                hiddens_big.dtype()
            )));
        }
        if weights.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_matmul_wsum_with_indices_buffer_candle_queue_into: weights dtype {:?} != F32",
                weights.dtype()
            )));
        }
        if out.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_matmul_wsum_with_indices_buffer_candle_queue_into: out dtype {:?} != F32",
                out.dtype()
            )));
        }
        let weights_elems: usize = weights.dims().iter().product();
        if weights_elems != k {
            return Err(candle_core::Error::Msg(format!(
                "moe_matmul_wsum_with_indices_buffer_candle_queue_into: weights {} != k {}",
                weights_elems, k
            )));
        }
        let out_elems: usize = out.dims().iter().product();
        if out_elems != batch * self.hidden {
            return Err(candle_core::Error::Msg(format!(
                "moe_matmul_wsum_with_indices_buffer_candle_queue_into: out {} != batch*hidden {}",
                out_elems,
                batch * self.hidden
            )));
        }

        let device = hiddens_big.device().clone();
        let md = match &device {
            Device::Metal(md) => md,
            _ => {
                return Err(candle_core::Error::Msg(
                    "moe_matmul_wsum_with_indices_buffer_candle_queue_into: Metal device required"
                        .into(),
                ));
            }
        };

        let x_c = if hiddens_big.is_contiguous() {
            hiddens_big.clone()
        } else {
            hiddens_big.contiguous()?
        };
        let inds_c = if inds_slice.is_contiguous() {
            inds_slice.clone()
        } else {
            inds_slice.contiguous()?
        };
        let weights_c = if weights.is_contiguous() {
            weights.clone()
        } else {
            weights.contiguous()?
        };

        let (x_buf, x_offset) = metal_buffer_of(&x_c)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for hiddens_big".into()))?;
        let (inds_buf, inds_offset) = metal_buffer_of(&inds_c)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for inds_slice".into()))?;
        let (weights_buf, weights_offset) = metal_buffer_of(&weights_c)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for weights".into()))?;
        let (out_buf, out_offset) = metal_buffer_of(out)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for out".into()))?;

        let encoder = md
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("candle command_encoder: {e}")))?;
        self.ctx
            .matmul_moe_wsum_zero_copy_with_indices_buffer_inline(
                encoder.as_ref(),
                &self.down_packed_all,
                &self.down_scales_all,
                inds_buf,
                inds_offset,
                k,
                weights_buf,
                weights_offset,
                x_buf,
                x_offset,
                out_buf,
                out_offset,
                self.hidden,
                self.moe_inter,
                batch,
            )
            .map_err(|e| {
                candle_core::Error::Msg(format!(
                    "moe_matmul_wsum_with_indices_buffer_candle_queue_into: {e}"
                ))
            })?;
        drop(encoder);
        ZERO_COPY_HITS.fetch_add(k, Relaxed);

        Ok(())
    }

    /// Lever G (2026-04-27): routing top-k partial select on the candle queue.
    /// Replaces the chain
    ///   `probs.arg_sort_last_dim(false)? .narrow(D::Minus1, 0, k)? .contiguous()?`
    ///   + `probs.gather(&inds, D::Minus1)?`
    /// with a single dispatch that produces caller-allocated `inds_out [BL, k]`
    /// (U32) and `vals_out [BL, k]` (F32) directly.
    ///
    /// Inputs:
    ///   - `probs`:    F32 Metal-resident `[BL, num_experts]`. Read-only.
    ///   - `inds_out`: U32 Metal-resident `[BL, k]`. Caller-allocated.
    ///   - `vals_out`: F32 Metal-resident `[BL, k]`. Caller-allocated.
    ///
    /// Constraints: `num_experts ≤ 256` (TG-size limit). Production
    /// Qwen3.5-MoE has E=256 (exact fit).
    pub fn topk_partial_select_candle_queue_into(
        &self,
        probs: &Tensor,
        inds_out: &Tensor,
        vals_out: &Tensor,
    ) -> candle_core::Result<()> {
        use candle_core::Device;

        if probs.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(format!(
                "topk_partial_select_candle_queue_into: probs dtype {:?} != F32",
                probs.dtype()
            )));
        }
        if inds_out.dtype() != DType::U32 {
            return Err(candle_core::Error::Msg(format!(
                "topk_partial_select_candle_queue_into: inds_out dtype {:?} != U32",
                inds_out.dtype()
            )));
        }
        if vals_out.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(format!(
                "topk_partial_select_candle_queue_into: vals_out dtype {:?} != F32",
                vals_out.dtype()
            )));
        }

        let probs_dims = probs.dims();
        if probs_dims.len() != 2 {
            return Err(candle_core::Error::Msg(format!(
                "topk_partial_select_candle_queue_into: probs shape {:?} must be 2-D",
                probs_dims
            )));
        }
        let bl = probs_dims[0];
        let num_experts = probs_dims[1];

        let inds_dims = inds_out.dims();
        let vals_dims = vals_out.dims();
        if inds_dims.len() != 2 || inds_dims[0] != bl {
            return Err(candle_core::Error::Msg(format!(
                "topk_partial_select_candle_queue_into: inds_out {:?} != [{}, k]",
                inds_dims, bl
            )));
        }
        if vals_dims != inds_dims {
            return Err(candle_core::Error::Msg(format!(
                "topk_partial_select_candle_queue_into: vals_out {:?} != inds_out {:?}",
                vals_dims, inds_dims
            )));
        }
        let k = inds_dims[1];

        let device = probs.device().clone();
        let md = match &device {
            Device::Metal(md) => md,
            _ => {
                return Err(candle_core::Error::Msg(
                    "topk_partial_select_candle_queue_into: Metal device required".into(),
                ));
            }
        };

        let probs_c = if probs.is_contiguous() {
            probs.clone()
        } else {
            probs.contiguous()?
        };
        let (probs_buf, probs_off) = metal_buffer_of(&probs_c)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for probs".into()))?;
        let (inds_buf, inds_off) = metal_buffer_of(inds_out)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for inds_out".into()))?;
        let (vals_buf, vals_off) = metal_buffer_of(vals_out)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for vals_out".into()))?;

        let encoder = md
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("candle command_encoder: {e}")))?;
        self.ctx
            .topk_partial_select_zero_copy_inline(
                encoder.as_ref(),
                probs_buf,
                probs_off,
                inds_buf,
                inds_off,
                vals_buf,
                vals_off,
                bl,
                num_experts,
                k,
            )
            .map_err(|e| {
                candle_core::Error::Msg(format!("topk_partial_select_candle_queue_into: {e}"))
            })?;
        drop(encoder);
        Ok(())
    }

    /// fusion (Candle-queue
    /// integrated). Replaces the entire 6-dispatch routing chain `softmax →
    /// arg_sort → narrow → gather → sum_keepdim → broadcast_div` with a
    /// single dispatch on Candle's command queue (zero-copy, encoder-injected,
    /// no extra sync).
    ///
    /// **I/O:**
    ///   - `logits`:   F32 Metal-resident `[BL, num_experts]`. **Raw logits**
    ///                 (post-routing-gate matmul, **before** softmax). Read-only.
    ///   - `inds_out`: U32 Metal-resident `[BL, k]`. Caller-allocated.
    ///   - `vals_out`: F32 Metal-resident `[BL, k]`. Caller-allocated. Holds
    ///                 the **renormalized** top-k weights on return.
    ///
    /// **Constraints:** `num_experts ≤ 256`, `k ≤ 32`. Production
    /// Qwen3.5-MoE: E=256, k=8.
    pub fn router_softmax_topk_renorm_f32_candle_queue_into(
        &self,
        logits: &Tensor,
        inds_out: &Tensor,
        vals_out: &Tensor,
    ) -> candle_core::Result<()> {
        use candle_core::Device;

        if logits.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(format!(
                "router_softmax_topk_renorm_f32_candle_queue_into: logits dtype {:?} != F32",
                logits.dtype()
            )));
        }
        if inds_out.dtype() != DType::U32 {
            return Err(candle_core::Error::Msg(format!(
                "router_softmax_topk_renorm_f32_candle_queue_into: inds_out dtype {:?} != U32",
                inds_out.dtype()
            )));
        }
        if vals_out.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(format!(
                "router_softmax_topk_renorm_f32_candle_queue_into: vals_out dtype {:?} != F32",
                vals_out.dtype()
            )));
        }

        let logits_dims = logits.dims();
        if logits_dims.len() != 2 {
            return Err(candle_core::Error::Msg(format!(
                "router_softmax_topk_renorm_f32_candle_queue_into: logits shape {:?} must be 2-D",
                logits_dims
            )));
        }
        let bl = logits_dims[0];
        let num_experts = logits_dims[1];

        let inds_dims = inds_out.dims();
        let vals_dims = vals_out.dims();
        if inds_dims.len() != 2 || inds_dims[0] != bl {
            return Err(candle_core::Error::Msg(format!(
                "router_softmax_topk_renorm_f32_candle_queue_into: inds_out {:?} != [{}, k]",
                inds_dims, bl
            )));
        }
        if vals_dims != inds_dims {
            return Err(candle_core::Error::Msg(format!(
                "router_softmax_topk_renorm_f32_candle_queue_into: vals_out {:?} != inds_out {:?}",
                vals_dims, inds_dims
            )));
        }
        let k = inds_dims[1];

        let device = logits.device().clone();
        let md = match &device {
            Device::Metal(md) => md,
            _ => {
                return Err(candle_core::Error::Msg(
                    "router_softmax_topk_renorm_f32_candle_queue_into: Metal device required"
                        .into(),
                ));
            }
        };

        let logits_c = if logits.is_contiguous() {
            logits.clone()
        } else {
            logits.contiguous()?
        };
        let (logits_buf, logits_off) = metal_buffer_of(&logits_c)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for logits".into()))?;
        let (inds_buf, inds_off) = metal_buffer_of(inds_out)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for inds_out".into()))?;
        let (vals_buf, vals_off) = metal_buffer_of(vals_out)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for vals_out".into()))?;

        let encoder = md
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("candle command_encoder: {e}")))?;
        self.ctx
            .router_softmax_topk_renorm_zero_copy_inline(
                encoder.as_ref(),
                logits_buf,
                logits_off,
                inds_buf,
                inds_off,
                vals_buf,
                vals_off,
                bl,
                num_experts,
                k,
            )
            .map_err(|e| {
                candle_core::Error::Msg(format!(
                    "router_softmax_topk_renorm_f32_candle_queue_into: {e}"
                ))
            })?;
        drop(encoder);
        Ok(())
    }

    /// Lever C-atomic (2026-04-27): grid-parallel variant of the fused MoE
    /// down+wsum dispatch. Same signature as
    /// `moe_matmul_wsum_with_indices_buffer_candle_queue_into` but uses the
    /// atomic-add kernel that keeps `grid.z = k` (2048 TGs production) and
    /// trades inter-TG parallelism for k-way atomic contention per output
    /// element.
    ///
    /// Caller must pre-zero `out` before calling. `Tensor::zeros(.., F32, ..)`
    /// in the call site provides this.
    pub fn moe_matmul_wsum_atomic_with_indices_buffer_candle_queue_into(
        &self,
        hiddens_big: &Tensor,
        inds_slice: &Tensor,
        k: usize,
        weights: &Tensor,
        out: &Tensor,
    ) -> candle_core::Result<()> {
        use candle_core::Device;
        use std::sync::atomic::Ordering::Relaxed;

        if k == 0 {
            return Err(candle_core::Error::Msg(
                "moe_matmul_wsum_atomic_with_indices_buffer_candle_queue_into: k==0".into(),
            ));
        }
        if inds_slice.dtype() != DType::U32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_matmul_wsum_atomic_with_indices_buffer_candle_queue_into: inds dtype {:?} != U32",
                inds_slice.dtype()
            )));
        }
        let inds_elems: usize = inds_slice.dims().iter().product();
        if inds_elems != k {
            return Err(candle_core::Error::Msg(format!(
                "moe_matmul_wsum_atomic_with_indices_buffer_candle_queue_into: inds_slice {} != k {}",
                inds_elems, k
            )));
        }
        let in_dims = hiddens_big.dims();
        if in_dims.len() != 2 || in_dims[1] != self.moe_inter {
            return Err(candle_core::Error::Msg(format!(
                "moe_matmul_wsum_atomic_with_indices_buffer_candle_queue_into: hiddens_big {:?} != [k*batch, {}]",
                in_dims, self.moe_inter
            )));
        }
        if in_dims[0] % k != 0 {
            return Err(candle_core::Error::Msg(format!(
                "moe_matmul_wsum_atomic_with_indices_buffer_candle_queue_into: rows {} not divisible by k {}",
                in_dims[0], k
            )));
        }
        let batch = in_dims[0] / k;

        if hiddens_big.dtype() != DType::F32
            || weights.dtype() != DType::F32
            || out.dtype() != DType::F32
        {
            return Err(candle_core::Error::Msg(
                "moe_matmul_wsum_atomic_with_indices_buffer_candle_queue_into: F32 required for hiddens/weights/out".into(),
            ));
        }
        let weights_elems: usize = weights.dims().iter().product();
        if weights_elems != k {
            return Err(candle_core::Error::Msg(format!(
                "moe_matmul_wsum_atomic_with_indices_buffer_candle_queue_into: weights {} != k {}",
                weights_elems, k
            )));
        }
        let out_elems: usize = out.dims().iter().product();
        if out_elems != batch * self.hidden {
            return Err(candle_core::Error::Msg(format!(
                "moe_matmul_wsum_atomic_with_indices_buffer_candle_queue_into: out {} != batch*hidden {}",
                out_elems,
                batch * self.hidden
            )));
        }

        let device = hiddens_big.device().clone();
        let md = match &device {
            Device::Metal(md) => md,
            _ => {
                return Err(candle_core::Error::Msg(
                    "moe_matmul_wsum_atomic_with_indices_buffer_candle_queue_into: Metal device required"
                        .into(),
                ));
            }
        };

        let x_c = if hiddens_big.is_contiguous() {
            hiddens_big.clone()
        } else {
            hiddens_big.contiguous()?
        };
        let inds_c = if inds_slice.is_contiguous() {
            inds_slice.clone()
        } else {
            inds_slice.contiguous()?
        };
        let weights_c = if weights.is_contiguous() {
            weights.clone()
        } else {
            weights.contiguous()?
        };

        let (x_buf, x_offset) = metal_buffer_of(&x_c)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for hiddens_big".into()))?;
        let (inds_buf, inds_offset) = metal_buffer_of(&inds_c)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for inds_slice".into()))?;
        let (weights_buf, weights_offset) = metal_buffer_of(&weights_c)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for weights".into()))?;
        let (out_buf, out_offset) = metal_buffer_of(out)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for out".into()))?;

        let encoder = md
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("candle command_encoder: {e}")))?;
        self.ctx
            .matmul_moe_wsum_atomic_zero_copy_with_indices_buffer_inline(
                encoder.as_ref(),
                &self.down_packed_all,
                &self.down_scales_all,
                inds_buf,
                inds_offset,
                k,
                weights_buf,
                weights_offset,
                x_buf,
                x_offset,
                out_buf,
                out_offset,
                self.hidden,
                self.moe_inter,
                batch,
            )
            .map_err(|e| {
                candle_core::Error::Msg(format!(
                    "moe_matmul_wsum_atomic_with_indices_buffer_candle_queue_into: {e}"
                ))
            })?;
        drop(encoder);
        ZERO_COPY_HITS.fetch_add(k, Relaxed);

        Ok(())
    }

    /// caller-provided y variant of
    /// `moe_down_with_indices_buffer_candle_queue`. `y_slice` must be
    /// `[k * batch, hidden]` Metal F32 and live long enough for the kernel
    /// to commit.
    pub fn moe_down_with_indices_buffer_candle_queue_into(
        &self,
        hiddens_big: &Tensor,
        inds_slice: &Tensor,
        k: usize,
        y_slice: &Tensor,
    ) -> candle_core::Result<()> {
        use candle_core::Device;
        use std::sync::atomic::Ordering::Relaxed;

        if k == 0 {
            return Err(candle_core::Error::Msg(
                "moe_down_with_indices_buffer_candle_queue_into: k==0".into(),
            ));
        }
        if inds_slice.dtype() != DType::U32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_with_indices_buffer_candle_queue_into: inds dtype {:?} != U32",
                inds_slice.dtype()
            )));
        }
        let inds_elems: usize = inds_slice.dims().iter().product();
        if inds_elems != k {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_with_indices_buffer_candle_queue_into: inds_slice {} != k {}",
                inds_elems, k
            )));
        }
        let in_dims = hiddens_big.dims();
        if in_dims.len() != 2 || in_dims[1] != self.moe_inter {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_with_indices_buffer_candle_queue_into: hiddens_big {:?} != [k*batch, {}]",
                in_dims, self.moe_inter
            )));
        }
        if in_dims[0] % k != 0 {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_with_indices_buffer_candle_queue_into: rows {} not divisible by k {}",
                in_dims[0], k
            )));
        }
        let batch = in_dims[0] / k;

        let y_dims = y_slice.dims();
        if y_dims.len() != 2 || y_dims[1] != self.hidden {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_with_indices_buffer_candle_queue_into: y_slice {:?} != [.., {}]",
                y_dims, self.hidden
            )));
        }
        if y_dims[0] != k * batch {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_with_indices_buffer_candle_queue_into: y rows {} != k*batch {}",
                y_dims[0],
                k * batch
            )));
        }
        if y_slice.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_with_indices_buffer_candle_queue_into: y dtype {:?} != F32",
                y_slice.dtype()
            )));
        }

        let device = hiddens_big.device().clone();
        let md = match &device {
            Device::Metal(md) => md,
            _ => {
                return Err(candle_core::Error::Msg(
                    "moe_down_with_indices_buffer_candle_queue_into: Metal device required".into(),
                ));
            }
        };

        let x_f32 = if hiddens_big.dtype() == DType::F32 {
            hiddens_big.contiguous()?
        } else {
            hiddens_big.to_dtype(DType::F32)?.contiguous()?
        };

        let inds_contig = if inds_slice.is_contiguous() {
            inds_slice.clone()
        } else {
            inds_slice.contiguous()?
        };

        let (y_buf, y_offset) = metal_buffer_of(y_slice)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for y_slice".into()))?;
        let (x_buf, x_offset) = metal_buffer_of(&x_f32)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for hiddens_big".into()))?;
        let (inds_buf, inds_offset) = metal_buffer_of(&inds_contig)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for inds_slice".into()))?;

        let encoder = md
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("candle command_encoder: {e}")))?;
        self.ctx
            .matmul_moe_zero_copy_with_indices_buffer_inline(
                encoder.as_ref(),
                &self.down_packed_all,
                &self.down_scales_all,
                inds_buf,
                inds_offset,
                k,
                x_buf,
                x_offset,
                y_buf,
                y_offset,
                self.hidden,
                self.moe_inter,
                batch,
                false,
            )
            .map_err(|e| {
                candle_core::Error::Msg(format!(
                    "moe_down_with_indices_buffer_candle_queue_into: {e}"
                ))
            })?;
        drop(encoder);
        ZERO_COPY_HITS.fetch_add(k, Relaxed);

        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────────
    // CB Phase 2 (2026-04-29) — multi-token MoE wrappers.
    //
    // These collapse the host `for t in 0..bl` loop in
    // `qwen3_5_moe::moe::SparseMoeBlock::forward_with_rmsnorm` into a single
    // dispatch per stage when `bl > 1`. All three accept a per-token
    // expert-indices tensor `inds: [B, k]` (contiguous, length `B*k`) and
    // dispatch through the candle command encoder so they share a buffer
    // with the surrounding ops.
    // ──────────────────────────────────────────────────────────────────────

    /// Multi-token version of
    /// `moe_gate_up_silu_mul_rmsnorm_with_indices_buffer_candle_queue_into`.
    ///
    /// `x_raw`: `[B, hidden]` (or higher-rank with batch product `B`); RAW
    /// (pre-RmsNorm). `inds`: `[B, k]` per-token expert IDs.
    /// `y_slice`: `[k, B, moe_inter]` flat-row pool.
    pub fn moe_gate_up_silu_mul_rmsnorm_multi_candle_queue_into(
        &self,
        x_raw: &Tensor,
        rms_weight: &Tensor,
        rms_eps: f32,
        inds: &Tensor,
        k: usize,
        batch: usize,
        y_slice: &Tensor,
    ) -> candle_core::Result<()> {
        use candle_core::Device;
        use std::sync::atomic::Ordering::Relaxed;

        if k == 0 || batch == 0 {
            return Ok(());
        }
        if inds.dtype() != DType::U32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_silu_mul_rmsnorm_multi_…: inds dtype {:?} != U32",
                inds.dtype()
            )));
        }
        let inds_elems: usize = inds.dims().iter().product();
        if inds_elems != batch * k {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_silu_mul_rmsnorm_multi_…: inds {} != B*k = {}",
                inds_elems,
                batch * k
            )));
        }
        let y_dims = y_slice.dims();
        if y_dims != [k * batch, self.moe_inter] {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_silu_mul_rmsnorm_multi_…: y {:?} != [k*B={}, {}]",
                y_dims,
                k * batch,
                self.moe_inter
            )));
        }
        if y_slice.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_silu_mul_rmsnorm_multi_…: y dtype {:?} != F32",
                y_slice.dtype()
            )));
        }

        let device = x_raw.device().clone();
        let md = match &device {
            Device::Metal(md) => md,
            _ => {
                return Err(candle_core::Error::Msg(
                    "moe_gate_up_silu_mul_rmsnorm_multi_…: Metal device required".into(),
                ));
            }
        };

        let x_f32 = if x_raw.dtype() == DType::F32 {
            x_raw.contiguous()?
        } else {
            x_raw.to_dtype(DType::F32)?.contiguous()?
        };
        let in_dims = x_f32.dims();
        let in_batch: usize = in_dims[..in_dims.len() - 1].iter().product();
        if in_batch != batch {
            return Err(candle_core::Error::Msg(format!(
                "moe_gate_up_silu_mul_rmsnorm_multi_…: x batch {} != batch arg {}",
                in_batch, batch
            )));
        }

        let rms_w_f32 = if rms_weight.dtype() == DType::F32 {
            rms_weight.contiguous()?
        } else {
            rms_weight.to_dtype(DType::F32)?.contiguous()?
        };
        let inds_contig = if inds.is_contiguous() {
            inds.clone()
        } else {
            inds.contiguous()?
        };

        let (y_buf, y_offset) = metal_buffer_of(y_slice)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for y_slice".into()))?;
        let (x_buf, x_offset) = metal_buffer_of(&x_f32)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for x".into()))?;
        let (rms_buf, rms_offset) = metal_buffer_of(&rms_w_f32)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for rms_weight".into()))?;
        let (inds_buf, inds_offset) = metal_buffer_of(&inds_contig)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for inds".into()))?;

        let encoder = md
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("candle command_encoder: {e}")))?;
        self.ctx
            .matmul_moe_gate_up_silu_mul_rmsnorm_multi_zero_copy_inline(
                encoder.as_ref(),
                &self.gate_up_packed_all,
                &self.gate_up_scales_all,
                inds_buf,
                inds_offset,
                k,
                x_buf,
                x_offset,
                rms_buf,
                rms_offset,
                y_buf,
                y_offset,
                self.moe_inter,
                self.hidden,
                batch,
                rms_eps,
            )
            .map_err(|e| {
                candle_core::Error::Msg(format!(
                    "moe_gate_up_silu_mul_rmsnorm_multi_candle_queue_into: {e}"
                ))
            })?;
        drop(encoder);
        ZERO_COPY_HITS.fetch_add(k * batch, Relaxed);
        Ok(())
    }

    /// Multi-token version of `moe_down_with_indices_buffer_candle_queue_into`.
    ///
    /// `hiddens_big`: `[k*B, moe_inter]` (gate_up_silu_mul output).
    /// `inds`: `[B, k]`. `y_slice`: `[k*B, hidden]`.
    pub fn moe_down_multi_candle_queue_into(
        &self,
        hiddens_big: &Tensor,
        inds: &Tensor,
        k: usize,
        batch: usize,
        y_slice: &Tensor,
    ) -> candle_core::Result<()> {
        use candle_core::Device;
        use std::sync::atomic::Ordering::Relaxed;

        if k == 0 || batch == 0 {
            return Ok(());
        }
        if inds.dtype() != DType::U32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_multi_…: inds dtype {:?} != U32",
                inds.dtype()
            )));
        }
        let inds_elems: usize = inds.dims().iter().product();
        if inds_elems != batch * k {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_multi_…: inds {} != B*k = {}",
                inds_elems,
                batch * k
            )));
        }
        let in_dims = hiddens_big.dims();
        if in_dims != [k * batch, self.moe_inter] {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_multi_…: hiddens_big {:?} != [k*B={}, {}]",
                in_dims,
                k * batch,
                self.moe_inter
            )));
        }
        let y_dims = y_slice.dims();
        if y_dims != [k * batch, self.hidden] {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_multi_…: y {:?} != [k*B={}, {}]",
                y_dims,
                k * batch,
                self.hidden
            )));
        }
        if y_slice.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_multi_…: y dtype {:?} != F32",
                y_slice.dtype()
            )));
        }

        let device = hiddens_big.device().clone();
        let md = match &device {
            Device::Metal(md) => md,
            _ => {
                return Err(candle_core::Error::Msg(
                    "moe_down_multi_…: Metal device required".into(),
                ));
            }
        };

        let x_f32 = if hiddens_big.dtype() == DType::F32 {
            hiddens_big.contiguous()?
        } else {
            hiddens_big.to_dtype(DType::F32)?.contiguous()?
        };
        let inds_contig = if inds.is_contiguous() {
            inds.clone()
        } else {
            inds.contiguous()?
        };

        let (y_buf, y_offset) = metal_buffer_of(y_slice)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for y_slice".into()))?;
        let (x_buf, x_offset) = metal_buffer_of(&x_f32)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for hiddens_big".into()))?;
        let (inds_buf, inds_offset) = metal_buffer_of(&inds_contig)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for inds".into()))?;

        let encoder = md
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("candle command_encoder: {e}")))?;
        self.ctx
            .matmul_moe_multi_zero_copy_inline(
                encoder.as_ref(),
                &self.down_packed_all,
                &self.down_scales_all,
                inds_buf,
                inds_offset,
                k,
                x_buf,
                x_offset,
                y_buf,
                y_offset,
                self.hidden,
                self.moe_inter,
                batch,
            )
            .map_err(|e| {
                candle_core::Error::Msg(format!("moe_down_multi_candle_queue_into: {e}"))
            })?;
        drop(encoder);
        ZERO_COPY_HITS.fetch_add(k * batch, Relaxed);
        Ok(())
    }

    /// Multi-token version of `moe_wsum_candle_queue_into`.
    ///
    /// `downs`: `[k*B, hidden]` (down output).
    /// `weights`: `[B, k]` per-token routing weights.
    /// `out`: `[B, hidden]`.
    pub fn moe_wsum_multi_candle_queue_into(
        &self,
        downs: &Tensor,
        weights: &Tensor,
        k: usize,
        batch: usize,
        out: &Tensor,
    ) -> candle_core::Result<()> {
        use candle_core::Device;

        if k == 0 || batch == 0 {
            return Ok(());
        }
        if downs.dtype() != DType::F32 || weights.dtype() != DType::F32 || out.dtype() != DType::F32
        {
            return Err(candle_core::Error::Msg(
                "moe_wsum_multi_…: all tensors must be F32".into(),
            ));
        }
        let downs_dims = downs.dims();
        if downs_dims != [k * batch, self.hidden] {
            return Err(candle_core::Error::Msg(format!(
                "moe_wsum_multi_…: downs {:?} != [k*B={}, {}]",
                downs_dims,
                k * batch,
                self.hidden
            )));
        }
        let weights_elems: usize = weights.dims().iter().product();
        if weights_elems != batch * k {
            return Err(candle_core::Error::Msg(format!(
                "moe_wsum_multi_…: weights {} != B*k = {}",
                weights_elems,
                batch * k
            )));
        }
        let out_elems: usize = out.dims().iter().product();
        if out_elems != batch * self.hidden {
            return Err(candle_core::Error::Msg(format!(
                "moe_wsum_multi_…: out {} != B*hidden = {}",
                out_elems,
                batch * self.hidden
            )));
        }

        let device = downs.device().clone();
        let md = match &device {
            Device::Metal(md) => md,
            _ => {
                return Err(candle_core::Error::Msg(
                    "moe_wsum_multi_…: Metal device required".into(),
                ));
            }
        };

        let downs_c = if downs.is_contiguous() {
            downs.clone()
        } else {
            downs.contiguous()?
        };
        let weights_c = if weights.is_contiguous() {
            weights.clone()
        } else {
            weights.contiguous()?
        };

        let (downs_buf, downs_off) = metal_buffer_of(&downs_c)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for downs".into()))?;
        let (weights_buf, weights_off) = metal_buffer_of(&weights_c)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for weights".into()))?;
        let (out_buf, out_off) = metal_buffer_of(out)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for out".into()))?;

        let encoder = md
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("candle command_encoder: {e}")))?;
        self.ctx
            .moe_wsum_multi_zero_copy_inline(
                encoder.as_ref(),
                downs_buf,
                downs_off,
                weights_buf,
                weights_off,
                out_buf,
                out_off,
                k,
                batch,
                self.hidden,
            )
            .map_err(|e| {
                candle_core::Error::Msg(format!("moe_wsum_multi_candle_queue_into: {e}"))
            })?;
        drop(encoder);
        Ok(())
    }

    /// GPU-resident variant of `moe_down`. See
    /// `moe_gate_up_with_indices_buffer` for contract details.
    pub fn moe_down_with_indices_buffer(
        &self,
        hiddens_big: &Tensor,
        inds_slice: &Tensor,
        k: usize,
    ) -> candle_core::Result<Tensor> {
        use candle_core::Device;
        use std::sync::atomic::Ordering::Relaxed;

        if k == 0 {
            return Err(candle_core::Error::Msg(
                "moe_down_with_indices_buffer: k==0".into(),
            ));
        }
        if inds_slice.dtype() != DType::U32 {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_with_indices_buffer: inds dtype {:?} != U32",
                inds_slice.dtype()
            )));
        }
        let inds_elems: usize = inds_slice.dims().iter().product();
        if inds_elems != k {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_with_indices_buffer: inds_slice has {} elems, expected k={}",
                inds_elems, k
            )));
        }
        let in_dims = hiddens_big.dims();
        if in_dims.len() != 2 || in_dims[1] != self.moe_inter {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_with_indices_buffer: hiddens_big {:?} != [k*batch, {}]",
                in_dims, self.moe_inter
            )));
        }
        if in_dims[0] % k != 0 {
            return Err(candle_core::Error::Msg(format!(
                "moe_down_with_indices_buffer: rows {} not divisible by k {}",
                in_dims[0], k
            )));
        }
        let batch = in_dims[0] / k;

        let device = hiddens_big.device().clone();
        if !device.is_metal() {
            return Err(candle_core::Error::Msg(
                "moe_down_with_indices_buffer: CPU device not supported".into(),
            ));
        }

        let x_f32 = if hiddens_big.dtype() == DType::F32 {
            hiddens_big.contiguous()?
        } else {
            hiddens_big.to_dtype(DType::F32)?.contiguous()?
        };
        let y_big = Tensor::zeros(vec![k * batch, self.hidden], DType::F32, &device)?;

        let inds_contig = if inds_slice.is_contiguous() {
            inds_slice.clone()
        } else {
            inds_slice.contiguous()?
        };

        if let Device::Metal(md) = &device {
            md.wait_until_completed()?;
        }

        let (y_buf, y_offset) = metal_buffer_of(&y_big).ok_or_else(|| {
            candle_core::Error::Msg("no metal buffer for moe_down_with_indices_buffer y_big".into())
        })?;
        let (x_buf, x_offset) = metal_buffer_of(&x_f32)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for hiddens_big".into()))?;
        let (inds_buf, inds_offset) = metal_buffer_of(&inds_contig)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for inds_slice".into()))?;

        self.ctx
            .matmul_moe_zero_copy_with_indices_buffer(
                &self.down_packed_all,
                &self.down_scales_all,
                inds_buf,
                inds_offset,
                k,
                x_buf,
                x_offset,
                y_buf,
                y_offset,
                self.hidden,
                self.moe_inter,
                batch,
                false, // per-slot x band
            )
            .map_err(|e| candle_core::Error::Msg(format!("moe_down_with_indices_buffer: {e}")))?;
        ZERO_COPY_HITS.fetch_add(k, Relaxed);

        Ok(y_big)
    }

    /// Fused Gate+Up group matmul: one multi_cmdbuf submission handles all 2k jobs (k for
    /// Gate, k for Up) with a single Candle sync + single device wait. Returns
    /// `(gate_big [k*batch, moe_inter], up_big [k*batch, moe_inter])`.
    ///
    /// Saves one sync + one wait per MoE layer compared to calling `Gate` and `Up`
    /// groups sequentially. Gate and Up share the same input `x` by construction.
    pub fn gate_and_up_group_big(
        &self,
        x: &Tensor,
        experts: &[usize],
    ) -> candle_core::Result<(Tensor, Tensor)> {
        use crate::mxfp4_gpu::Mxfp4Job;
        use candle_core::Device;
        use std::sync::atomic::Ordering::Relaxed;

        if experts.is_empty() {
            return Err(candle_core::Error::Msg(
                "gate_and_up_group_big: empty expert list".into(),
            ));
        }
        for e in experts {
            if *e >= self.num_experts {
                return Err(candle_core::Error::Msg(format!(
                    "expert_idx {} out of range (num_experts {})",
                    e, self.num_experts
                )));
            }
        }

        let device = x.device().clone();
        if !device.is_metal() {
            // CPU fallback: do them separately and stack. Slow path used by unit tests.
            let x_refs = [x].repeat(experts.len());
            let refs: Vec<&Tensor> = x_refs.iter().copied().collect();
            let gate_big = self.expert_matmul_group_big_impl(&refs, experts, ExpertProj::Gate)?;
            let up_big = self.expert_matmul_group_big_impl(&refs, experts, ExpertProj::Up)?;
            return Ok((gate_big, up_big));
        }

        // Cast x once; all k jobs share it.
        let x_f32 = if x.dtype() == DType::F32 {
            x.contiguous()?
        } else {
            x.to_dtype(DType::F32)?.contiguous()?
        };

        // With combined gate+up weights, each expert matmul produces `2*moe_inter` outputs
        // at once. k jobs total (was 2k before load-time fusion).
        let in_dims = x_f32.dims();
        let batch: usize = in_dims[..in_dims.len() - 1].iter().product();
        let combined_out = 2 * self.moe_inter;
        let k = experts.len();
        // Big output shape: [k*batch, 2*moe_inter]. Each expert i writes to row band
        // [i*batch, (i+1)*batch).
        let y_big = Tensor::zeros(vec![k * batch, combined_out], DType::F32, &device)?;

        // Single Candle sync drains the x cast + y_big zero-init.
        if let Device::Metal(md) = &device {
            md.wait_until_completed()?;
        }

        let (y_base_buf, y_base_offset) = metal_buffer_of(&y_big)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for gate+up y_big".into()))?;
        let (x_buf, x_offset) = metal_buffer_of(&x_f32)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for x".into()))?;
        let slab_stride_bytes = (batch * combined_out * 4) as u64;

        // One job per expert using the combined gate_up weight (out = 2*moe_inter).
        let mut jobs: Vec<Mxfp4Job<'_>> = Vec::with_capacity(k);
        for (i, &e) in experts.iter().enumerate() {
            jobs.push(Mxfp4Job {
                weight: &self.gate_up[e],
                x_buf,
                x_offset,
                y_buf: y_base_buf,
                y_offset: y_base_offset + (i as u64) * slab_stride_bytes,
                batch,
            });
        }

        self.ctx
            .matmul_zero_copy_multi_cmdbuf(&jobs)
            .map_err(|e| candle_core::Error::Msg(format!("mxfp4 gate_up fused: {e}")))?;
        ZERO_COPY_HITS.fetch_add(k, Relaxed);

        // Split y_big on the last axis: `[k*batch, 2*moe_inter]` → gate `[k*batch, moe_inter]`,
        // up `[k*batch, moe_inter]`. `.contiguous()` forces a memcpy-style gather so downstream
        // elementwise ops (silu/mul) hit the fast packed-layout Metal kernel instead of the
        // generic strided one.
        let gate_big = y_big.narrow(1, 0, self.moe_inter)?.contiguous()?;
        let up_big = y_big
            .narrow(1, self.moe_inter, self.moe_inter)?
            .contiguous()?;
        Ok((gate_big, up_big))
    }

    fn expert_matmul_group_impl(
        &self,
        inputs: &[&Tensor],
        experts: &[usize],
        proj: ExpertProj,
    ) -> candle_core::Result<Vec<Tensor>> {
        // Reuse the big-tensor path and split. Zero-cost `narrow` views.
        let y_big = self.expert_matmul_group_big_impl(inputs, experts, proj)?;
        if experts.is_empty() {
            return Ok(Vec::new());
        }
        let in_dims = inputs[0].dims();
        let batch: usize = in_dims[..in_dims.len() - 1].iter().product();
        let y_big_shape = y_big.dims().to_vec();
        let out_features = *y_big_shape.last().unwrap();
        let mut ys: Vec<Tensor> = Vec::with_capacity(experts.len());
        for i in 0..experts.len() {
            let slice = y_big.narrow(0, i * batch, batch)?;
            let mut out_shape: Vec<usize> = in_dims[..in_dims.len() - 1].to_vec();
            out_shape.push(out_features);
            ys.push(slice.reshape(out_shape)?);
        }
        Ok(ys)
    }

    fn expert_matmul_group_big_impl(
        &self,
        inputs: &[&Tensor],
        experts: &[usize],
        proj: ExpertProj,
    ) -> candle_core::Result<Tensor> {
        use crate::mxfp4_gpu::Mxfp4Job;
        use candle_core::Device;
        use std::sync::atomic::Ordering::Relaxed;

        assert_eq!(inputs.len(), experts.len());
        if experts.is_empty() {
            return Err(candle_core::Error::Msg(
                "expert_matmul_group_big: empty expert list".into(),
            ));
        }

        for e in experts {
            if *e >= self.num_experts {
                return Err(candle_core::Error::Msg(format!(
                    "expert_idx {} out of range (num_experts {})",
                    e, self.num_experts
                )));
            }
        }

        // Sanity + fall back to sequential CPU path if any input isn't on Metal.
        let device = inputs[0].device().clone();
        if !device.is_metal() {
            let per_expert: Vec<Tensor> = inputs
                .iter()
                .zip(experts.iter())
                .map(|(x, &e)| self.expert_matmul(x, e, proj))
                .collect::<candle_core::Result<_>>()?;
            // Flatten leading dims to [k*batch, out] and cat on axis 0.
            let flats: Vec<Tensor> = per_expert
                .iter()
                .map(|t| {
                    let d = t.dims();
                    let batch: usize = d[..d.len() - 1].iter().product();
                    t.reshape((batch, d[d.len() - 1]))
                })
                .collect::<candle_core::Result<_>>()?;
            let refs: Vec<&Tensor> = flats.iter().collect();
            return Tensor::cat(&refs, 0);
        }
        for x in inputs.iter() {
            if !x.device().is_metal() {
                return Err(candle_core::Error::Msg(
                    "expert_matmul_group: mixed devices".into(),
                ));
            }
        }

        // Cast each input to f32 contiguous; cache buffers so Metal sees valid storage.
        let xs_f32: Vec<Tensor> = inputs
            .iter()
            .map(|x| {
                if x.dtype() == DType::F32 {
                    x.contiguous()
                } else {
                    x.to_dtype(DType::F32).and_then(|t| t.contiguous())
                }
            })
            .collect::<candle_core::Result<_>>()?;

        // Pre-allocate the big output tensor BEFORE the Candle sync so its zero-init and
        // the input dtype casts are both in-flight on Candle's queue and can be drained
        // by a single `wait_until_completed()` below.
        //
        // After Gate+Up fusion (Option K), a single combined weight `[2*moe_inter, hidden]`
        // serves both Gate and Up projections. For legacy/test callers that request Gate
        // or Up here, we dispatch the combined matmul and narrow the output at the end.
        let w0 = match proj {
            ExpertProj::Gate | ExpertProj::Up => &self.gate_up[experts[0]],
            ExpertProj::Down => &self.down[experts[0]],
        };
        let out_features = w0.out_features;
        let in_dims = xs_f32[0].dims();
        let batch: usize = in_dims[..in_dims.len() - 1].iter().product();
        for x in &xs_f32 {
            let d = x.dims();
            let b: usize = d[..d.len() - 1].iter().product();
            if b != batch {
                return Err(candle_core::Error::Msg(format!(
                    "expert_matmul_group: mixed batch sizes {batch} vs {b}"
                )));
            }
        }
        // Shape of the big output: [k, batch, out_features] flattened.
        let big_shape = vec![experts.len() * batch, out_features];
        let y_big = Tensor::zeros(big_shape, DType::F32, &device)?;

        // Single Candle sync drains both the to_dtype/contiguous writes and the y_big
        // zero-init. Order matters — Candle's queue is FIFO, so by the time we observe
        // a completed flush, every op queued above has completed.
        if let Device::Metal(md) = &device {
            md.wait_until_completed()?;
        }

        // Extract buffer pointers. `y_big` is contiguous, so job[i] writes to
        //     y_base_offset + i * (batch * out_features) * sizeof(f32)
        let (y_base_buf, y_base_offset) = metal_buffer_of(&y_big)
            .ok_or_else(|| candle_core::Error::Msg("no metal buffer for y_big".into()))?;
        let slab_stride_bytes = (batch * out_features * 4) as u64;

        let mut jobs: Vec<Mxfp4Job<'_>> = Vec::with_capacity(experts.len());
        let mut x_bufs: Vec<(&crate::metal::Buffer, u64)> = Vec::with_capacity(experts.len());
        for x in xs_f32.iter() {
            let xb = metal_buffer_of(x)
                .ok_or_else(|| candle_core::Error::Msg("no metal buffer for x".into()))?;
            x_bufs.push(xb);
        }
        for (i, &e) in experts.iter().enumerate() {
            let w = match proj {
                ExpertProj::Gate | ExpertProj::Up => &self.gate_up[e],
                ExpertProj::Down => &self.down[e],
            };
            jobs.push(Mxfp4Job {
                weight: w,
                x_buf: x_bufs[i].0,
                x_offset: x_bufs[i].1,
                y_buf: y_base_buf,
                y_offset: y_base_offset + (i as u64) * slab_stride_bytes,
                batch,
            });
        }

        // Dispatch strategy (selectable via LUMEN_MOE_DISPATCH). Default is `multi_cmdbuf`.
        //
        // - `multi_cmdbuf` (default, post-G): k independent command buffers (1 encoder +
        //   1 dispatch each). Commit all, wait on last only. Candle sync 1×, device wait 1×.
        //   Each buffer is the exact topology that the proven `matmul_zero_copy` uses, so no
        //   new hazard surface.
        // - `fanout`: k sequential `matmul_zero_copy` calls, each with its own Candle sync
        //   and wait. Correct but slower (k× sync + k× wait).
        // - `multi_encoder`: one cmd buffer with k encoders (via `matmul_zero_copy_batch`).
        //   Known broken on M3 Max — garbled outputs. Debug only.
        //
        // Legacy flags still work:
        //   `LUMEN_MOE_BATCH_SINGLE=1` → fanout
        //   `LUMEN_MOE_BATCH_SINGLE=0` or `LUMEN_MOE_MULTI_ENCODER=1` → multi_encoder
        let strategy_raw = std::env::var("LUMEN_MOE_DISPATCH").unwrap_or_default();
        let strategy: &str = if !strategy_raw.is_empty() {
            strategy_raw.as_str()
        } else if std::env::var("LUMEN_MOE_MULTI_ENCODER")
            .map(|v| v == "1")
            .unwrap_or(false)
            || std::env::var("LUMEN_MOE_BATCH_SINGLE")
                .map(|v| v == "0")
                .unwrap_or(false)
        {
            "multi_encoder"
        } else if std::env::var("LUMEN_MOE_BATCH_SINGLE")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            "fanout"
        } else {
            "multi_cmdbuf"
        };
        match strategy {
            "fanout" => {
                for job in &jobs {
                    if let Device::Metal(md) = &device {
                        md.wait_until_completed()?;
                    }
                    self.ctx
                        .matmul_zero_copy(
                            job.weight,
                            job.x_buf,
                            job.x_offset,
                            job.y_buf,
                            job.y_offset,
                            job.batch,
                        )
                        .map_err(|e| candle_core::Error::Msg(format!("mxfp4 fanout: {e}")))?;
                }
            }
            "multi_encoder" => {
                self.ctx
                    .matmul_zero_copy_batch(&jobs)
                    .map_err(|e| candle_core::Error::Msg(format!("mxfp4 multi_encoder: {e}")))?;
            }
            _ => {
                self.ctx
                    .matmul_zero_copy_multi_cmdbuf(&jobs)
                    .map_err(|e| candle_core::Error::Msg(format!("mxfp4 multi_cmdbuf: {e}")))?;
            }
        }
        ZERO_COPY_HITS.fetch_add(experts.len(), Relaxed);

        // Return the `[k*batch, out_features]` tensor. For Gate/Up projs we dispatched the
        // combined `[2*moe_inter]`-wide matmul, so the caller's expected `moe_inter` slice
        // is the first half (Gate) or second half (Up) of the last axis.
        let _ = in_dims; // silence unused
        let _ = out_features;
        match proj {
            ExpertProj::Gate => y_big.narrow(1, 0, self.moe_inter),
            ExpertProj::Up => y_big.narrow(1, self.moe_inter, self.moe_inter),
            ExpertProj::Down => Ok(y_big),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mxfp4::dequantize_f32;
    use candle_core::Device;

    fn synth(out: usize, ins: usize, seed: u32) -> (Vec<u32>, Vec<u8>, Vec<f32>) {
        let n_groups = out * ins / 32;
        let n_words = out * ins / 8;
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            s
        };
        let packed: Vec<u32> = (0..n_words).map(|_| next()).collect();
        let scales: Vec<u8> = (0..n_groups).map(|_| 120 + (next() & 0x0F) as u8).collect();
        let mut dense = vec![0.0_f32; out * ins];
        dequantize_f32(&packed, &scales, &mut dense).unwrap();
        (packed, scales, dense)
    }

    #[test]
    fn forward_2d_matches_dense_candle_matmul() {
        let gpu_ctx = match MxFp4Context::new() {
            Ok(c) => Arc::new(c),
            Err(_) => return,
        };
        let device = Device::Cpu;
        let (out, ins) = (16, 64);
        let (packed, scales, dense) = synth(out, ins, 0xBADC0DE);
        let weight = Mxfp4Weight::from_host(&gpu_ctx.ctx, &packed, &scales, out, ins).unwrap();
        let linear = Mxfp4Linear::new(weight, None, Arc::clone(&gpu_ctx));

        let batch = 3;
        let x_vec: Vec<f32> = (0..batch * ins).map(|i| (i as f32 * 0.013).sin()).collect();
        let x = Tensor::from_vec(x_vec.clone(), (batch, ins), &device).unwrap();

        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[batch, out]);

        // Reference: dense Candle matmul (x @ W^T)
        let w_ref = Tensor::from_vec(dense.clone(), (out, ins), &device).unwrap();
        let y_ref = x.matmul(&w_ref.t().unwrap()).unwrap();
        let y_vec = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let ref_vec = y_ref.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let mut max_err = 0.0_f32;
        for (a, b) in y_vec.iter().zip(ref_vec.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 1e-2, "max abs err {max_err} exceeds tolerance");
    }

    #[test]
    fn forward_3d_preserves_leading_dims() {
        let gpu_ctx = match MxFp4Context::new() {
            Ok(c) => Arc::new(c),
            Err(_) => return,
        };
        let device = Device::Cpu;
        let (out, ins) = (8, 32);
        let (packed, scales, _dense) = synth(out, ins, 0x1234);
        let weight = Mxfp4Weight::from_host(&gpu_ctx.ctx, &packed, &scales, out, ins).unwrap();
        let linear = Mxfp4Linear::new(weight, None, Arc::clone(&gpu_ctx));

        let x_vec: Vec<f32> = (0..2 * 3 * ins).map(|i| i as f32 * 0.01).collect();
        let x = Tensor::from_vec(x_vec, (2, 3, ins), &device).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 3, out]);
    }

    #[test]
    fn forward_applies_bias() {
        let gpu_ctx = match MxFp4Context::new() {
            Ok(c) => Arc::new(c),
            Err(_) => return,
        };
        let device = Device::Cpu;
        let (out, ins) = (4, 32);
        let (packed, scales, _dense) = synth(out, ins, 0xABCD);
        let weight = Mxfp4Weight::from_host(&gpu_ctx.ctx, &packed, &scales, out, ins).unwrap();
        let bias = Tensor::from_vec(vec![10.0_f32; out], (out,), &device).unwrap();
        let linear = Mxfp4Linear::new(weight, Some(bias), Arc::clone(&gpu_ctx));

        let x = Tensor::zeros((1, ins), DType::F32, &device).unwrap();
        let y = linear.forward(&x).unwrap();
        // x is all zero → y == bias
        let y_vec = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in &y_vec {
            assert!((v - 10.0).abs() < 1e-5);
        }
    }

    #[test]
    fn rejects_wrong_in_features() {
        let gpu_ctx = match MxFp4Context::new() {
            Ok(c) => Arc::new(c),
            Err(_) => return,
        };
        let device = Device::Cpu;
        let (packed, scales, _) = synth(4, 32, 1);
        let weight = Mxfp4Weight::from_host(&gpu_ctx.ctx, &packed, &scales, 4, 32).unwrap();
        let linear = Mxfp4Linear::new(weight, None, Arc::clone(&gpu_ctx));
        let x = Tensor::zeros((1, 16), DType::F32, &device).unwrap(); // wrong in_features
        assert!(linear.forward(&x).is_err());
    }

    #[test]
    fn proj_linear_dense_dispatch() {
        let device = Device::Cpu;
        let w = Tensor::from_vec(
            (0..4 * 8).map(|i| i as f32 * 0.01).collect::<Vec<_>>(),
            (4, 8),
            &device,
        )
        .unwrap();
        let proj = ProjLinear::from(Linear::new(w, None));
        let x = Tensor::zeros((2, 8), DType::F32, &device).unwrap();
        let y = proj.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 4]);
        assert_eq!(proj.in_features(), 8);
        assert_eq!(proj.out_features(), 4);
    }

    #[test]
    fn proj_linear_mxfp4_dispatch() {
        let gpu_ctx = match MxFp4Context::new() {
            Ok(c) => Arc::new(c),
            Err(_) => return,
        };
        let device = Device::Cpu;
        let (out, ins) = (6, 32);
        let (packed, scales, _dense) = synth(out, ins, 0xCAFE);
        let weight = Mxfp4Weight::from_host(&gpu_ctx.ctx, &packed, &scales, out, ins).unwrap();
        let proj = ProjLinear::from(Mxfp4Linear::new(weight, None, Arc::clone(&gpu_ctx)));
        let x = Tensor::zeros((3, ins), DType::F32, &device).unwrap();
        let y = proj.forward(&x).unwrap();
        assert_eq!(y.dims(), &[3, out]);
        assert_eq!(proj.in_features(), ins);
        assert_eq!(proj.out_features(), out);
    }

    #[test]
    fn switch_mlp_expert_matmul_shapes() {
        let gpu_ctx = match MxFp4Context::new() {
            Ok(c) => Arc::new(c),
            Err(_) => return,
        };
        let device = Device::Cpu;
        let (num_experts, hidden, inter) = (3, 64, 32);

        // gate/up: [num_experts, inter, hidden] packed as [N * inter * hidden / 8] u32
        let (gp, gs, _) = synth(num_experts * inter, hidden, 0x1);
        let (up, us, _) = synth(num_experts * inter, hidden, 0x2);
        // down: [num_experts, hidden, inter]
        let (dp, ds, _) = synth(num_experts * hidden, inter, 0x3);

        let smlp = Mxfp4SwitchMlp::from_host(
            Arc::clone(&gpu_ctx),
            num_experts,
            hidden,
            inter,
            &gp,
            &gs,
            &up,
            &us,
            &dp,
            &ds,
        )
        .unwrap();

        // Per-token: x_t = [1, hidden] → gate([1, inter]) → down([1, hidden])
        let x_t = Tensor::zeros((1, hidden), DType::F32, &device).unwrap();
        let gate_out = smlp.expert_matmul(&x_t, 2, ExpertProj::Gate).unwrap();
        assert_eq!(gate_out.dims(), &[1, inter]);
        let up_out = smlp.expert_matmul(&x_t, 2, ExpertProj::Up).unwrap();
        assert_eq!(up_out.dims(), &[1, inter]);

        let h = Tensor::zeros((1, inter), DType::F32, &device).unwrap();
        let down_out = smlp.expert_matmul(&h, 2, ExpertProj::Down).unwrap();
        assert_eq!(down_out.dims(), &[1, hidden]);
    }

    #[test]
    fn switch_mlp_rejects_oob_expert_index() {
        let gpu_ctx = match MxFp4Context::new() {
            Ok(c) => Arc::new(c),
            Err(_) => return,
        };
        let device = Device::Cpu;
        let (num_experts, hidden, inter) = (2, 64, 32);
        let (gp, gs, _) = synth(num_experts * inter, hidden, 0x10);
        let (up, us, _) = synth(num_experts * inter, hidden, 0x20);
        let (dp, ds, _) = synth(num_experts * hidden, inter, 0x30);
        let smlp = Mxfp4SwitchMlp::from_host(
            Arc::clone(&gpu_ctx),
            num_experts,
            hidden,
            inter,
            &gp,
            &gs,
            &up,
            &us,
            &dp,
            &ds,
        )
        .unwrap();
        let x = Tensor::zeros((1, hidden), DType::F32, &device).unwrap();
        assert!(smlp.expert_matmul(&x, 5, ExpertProj::Gate).is_err());
    }
}
