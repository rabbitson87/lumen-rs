//! Candle `Linear`-shaped wrapper around [`Affine4Weight`].
//!
//! Sister to [`crate::mxfp4_linear`] for 4-bit affine quantization. Holds the weight
//! resident on GPU and exposes a familiar `forward(&Tensor) -> Tensor` so transformer
//! layers can call it interchangeably with `candle_nn::Linear` and `Mxfp4Linear`.
//!
//! MVP scope: only the f32-in / f32-out forward path. bf16 input/output, RmsNorm
//! fusion, and gate+up fusion are routed via f32 + dtype cast at the dispatcher
//! layer (`ProjLinear`) — same compute path, slightly more memory traffic. Adding
//! bespoke bf16 kernels is a follow-up optimization once correctness is proven.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use candle_core::{DType, Device, Storage, Tensor};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLResourceUsage};

use crate::affine4_gpu::{Affine4Context, Affine4Weight, pick_tile_for_in};
use crate::metal::{BatchedEncoderExt, Buffer, IndirectCommandBuffer};

fn icb_enabled() -> bool {
    std::env::var("LUMEN_ICB")
        .map(|v| v == "1")
        .unwrap_or(false)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Affine4Dims {
    out_features: u32,
    in_features: u32,
}

/// owns one IndirectCommandBuffer per dispatch variant
/// plus a constant `dims_buf` and a per-batch `batch_buf`. Tracks the
/// buffer addresses + offsets last bound at recording time so the next
/// forward can fast-path replay if they match.
pub struct Affine4LinearIcbCache {
    /// One ICB slot for the no-residual `qmv_fast_bf16in_bf16out` variant.
    icb_no_residual: IndirectCommandBuffer,
    /// One ICB slot for the residual-fused variant.
    icb_residual: IndirectCommandBuffer,
    /// `[out_features, in_features]` u32 — fixed for this Linear's shape.
    dims_buf: Buffer,
    /// `[batch]` u32 — re-allocated lazily if the batch size changes.
    batch_buf: Buffer,
    bound_batch: usize,

    // No-residual binding state.
    nr_x_id: usize,
    nr_x_off: u64,
    nr_y_id: usize,
    nr_y_off: u64,
    nr_recorded: bool,

    // Residual binding state.
    rs_x_id: usize,
    rs_x_off: u64,
    rs_r_id: usize,
    rs_r_off: u64,
    rs_y_id: usize,
    rs_y_off: u64,
    rs_recorded: bool,
}

impl Affine4LinearIcbCache {
    fn new(ctx: &Affine4Context, weight: &Affine4Weight, batch: usize) -> Result<Self> {
        let dims = Affine4Dims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
        };
        let dims_buf = ctx.ctx.buffer_with_data(&[dims]);
        let batch_buf = ctx.ctx.buffer_with_data(&[batch as u32]);

        let raw_device: Retained<ProtocolObject<dyn MTLDevice>> =
            Retained::from(ctx.ctx.device.as_ref());
        // Single command slot for each variant; 8 buffers covers the residual
        // variant (packed/scales/biases/x/residual/y/dims/batch) which is the
        // wider of the two — Metal allows the no-residual ICB to leave the
        // last slot unused.
        let icb_no_residual = IndirectCommandBuffer::new(&raw_device, 1, 7)
            .map_err(|e| anyhow::anyhow!("ICB alloc (no-residual): {e:?}"))?;
        let icb_residual = IndirectCommandBuffer::new(&raw_device, 1, 8)
            .map_err(|e| anyhow::anyhow!("ICB alloc (residual): {e:?}"))?;

        Ok(Self {
            icb_no_residual,
            icb_residual,
            dims_buf,
            batch_buf,
            bound_batch: batch,
            nr_x_id: 0,
            nr_x_off: 0,
            nr_y_id: 0,
            nr_y_off: 0,
            nr_recorded: false,
            rs_x_id: 0,
            rs_x_off: 0,
            rs_r_id: 0,
            rs_r_off: 0,
            rs_y_id: 0,
            rs_y_off: 0,
            rs_recorded: false,
        })
    }
}

/// Stable identity for the GPU resource backing this `Buffer`. Uses
/// `MTLBuffer::gpuAddress()` (the GPU virtual address), which is unique
/// across simultaneously-live MTLBuffers — robust against the pool
/// ejection scenario where a freed Arc<Buffer>'s heap slot is reused by
/// a freshly-allocated Arc wrapping a *different* MTLBuffer (which would
/// give the same `&Buffer` pointer but a different GPU resource).
#[inline]
fn buffer_id(b: &Buffer) -> usize {
    use objc2_metal::MTLBuffer as _;
    b.as_ref().gpuAddress() as usize
}

fn qmv_fast_enabled() -> bool {
    !std::env::var("LUMEN_AFFINE4_DISABLE_QMV_FAST")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Which arm [`Affine4Linear::forward_bf16_out`] dispatched to, counted.
///
/// The two arms are numerically indistinguishable — a bf16 activation widens
/// to f32 exactly, and both kernels round back to the same bf16 — so the
/// choice leaves no trace in the output. Without a counter the only evidence
/// is wall-clock, which is not a guard: it fails on a busy machine and passes
/// on a fast one regardless of whether the dispatch is correct. A relaxed
/// increment on a path that is about to issue a multi-megabyte matmul costs
/// nothing measurable.
pub mod dispatch_counters {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub(super) static BF16_IN_FAST_PATH: AtomicU64 = AtomicU64::new(0);
    pub(super) static F32_WIDEN_DETOUR: AtomicU64 = AtomicU64::new(0);

    /// `(bf16-in fast path, f32-widen detour)` since the last [`reset`].
    pub fn snapshot() -> (u64, u64) {
        (
            BF16_IN_FAST_PATH.load(Ordering::Relaxed),
            F32_WIDEN_DETOUR.load(Ordering::Relaxed),
        )
    }

    pub fn reset() {
        BF16_IN_FAST_PATH.store(0, Ordering::Relaxed);
        F32_WIDEN_DETOUR.store(0, Ordering::Relaxed);
    }
}

/// Extract the Metal buffer + byte offset backing a Candle tensor. Returns `None`
/// when the tensor is not Metal-resident. Same pattern as `mxfp4_linear::metal_buffer_of`.
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

fn affine4_matmul_tensor(
    ctx: &Affine4Context,
    weight: &Affine4Weight,
    x: &Tensor,
) -> candle_core::Result<Tensor> {
    let dims = x.dims();
    if dims.is_empty() {
        return Err(candle_core::Error::Msg(
            "affine4 matmul requires a non-scalar input".into(),
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

    let force_cpu = std::env::var("LUMEN_AFFINE4_FORCE_CPU")
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
                // P0: MLX-pattern qmv_fast kernel — primary decode path when
                // shape constraints (in % 512 == 0, out % 8 == 0) are met.
                // 64 threads/TG (vs v3's 256) + no TG memory + reformulated
                // dot product (`s*Σ(nib*x') + b*Σ(x)`) closes most of the gap
                // to MLX reference (10.10 → ~13-15 tok/s expected on 27B Dense).
                let qmv_fast_ok = qmv_fast_enabled()
                    && Affine4Context::qmv_fast_supports(weight.in_features, weight.out_features);

                // R4: when qmv_fast doesn't apply but in_features exceeds the
                // v3 single-shot TG memory budget, use the v3-tiled path.
                let disable_tiled = std::env::var("LUMEN_AFFINE4_DISABLE_TILED")
                    .map(|v| v == "1")
                    .unwrap_or(false);
                let tile_info = if disable_tiled || qmv_fast_ok {
                    None
                } else {
                    pick_tile_for_in(weight.in_features)
                };

                // Decode (batch=1): submit through Candle's own command encoder so
                // we share its queue and skip the cross-queue `wait_until_completed`
                // (~2 sync pts / matmul → 0). Mirror of MXFP4 Phase M.5 Option F:
                // saves ~50% of decode latency for short-output workloads.
                let proj_candle_queue = std::env::var("LUMEN_DISABLE_PROJ_CANDLE_QUEUE")
                    .map(|v| v != "1")
                    .unwrap_or(true)
                    && batch == 1;

                if qmv_fast_ok {
                    if proj_candle_queue {
                        if let Device::Metal(metal_dev) = x.device() {
                            let encoder = metal_dev.command_encoder().map_err(|e| {
                                candle_core::Error::Msg(format!("candle command_encoder: {e}"))
                            })?;
                            encoder.set_label("lumen:affine4_qmv_fast");
                            ctx.encode_qmv_fast_dispatch(
                                encoder.as_ref(),
                                weight,
                                x_buf,
                                x_offset,
                                y_buf,
                                y_offset,
                                batch,
                            );
                            drop(encoder);
                            return Ok(y);
                        }
                    }
                    // Own-queue fallback for batch>1 / non-Candle path.
                    if let Device::Metal(metal_dev) = x.device() {
                        metal_dev.wait_until_completed()?;
                    }
                    let encoder = crate::metal::process_commands()
                        .command_encoder()
                        .expect("ce");
                    encoder.set_label("lumen:affine4_qmv_fast");
                    ctx.encode_qmv_fast_dispatch(
                        encoder.as_ref(),
                        weight,
                        x_buf,
                        x_offset,
                        y_buf,
                        y_offset,
                        batch,
                    );
                    drop(encoder);
                    crate::metal::process_commands()
                        .flush_and_wait()
                        .expect("flush");
                    return Ok(y);
                }

                if let Some((tile_in, n_chunks)) = tile_info {
                    let scratch_bytes =
                        Affine4Context::tiled_scratch_bytes(weight.out_features, batch, n_chunks)
                            as u64;
                    let scratch = ctx.ctx.buffer_zeroed(scratch_bytes);

                    if proj_candle_queue {
                        if let Device::Metal(metal_dev) = x.device() {
                            let encoder = metal_dev.command_encoder().map_err(|e| {
                                candle_core::Error::Msg(format!("candle command_encoder: {e}"))
                            })?;
                            encoder.set_label("lumen:affine4_matmul_tiled");
                            ctx.encode_matmul_tiled_dispatch(
                                encoder.as_ref(),
                                weight,
                                x_buf,
                                x_offset,
                                &scratch,
                                y_buf,
                                y_offset,
                                tile_in,
                                n_chunks,
                                batch,
                            );
                            drop(encoder);
                            return Ok(y);
                        }
                    }
                    if let Device::Metal(metal_dev) = x.device() {
                        metal_dev.wait_until_completed()?;
                    }
                    ctx.matmul_tiled_zero_copy(
                        weight, x_buf, x_offset, y_buf, y_offset, tile_in, n_chunks, batch,
                    )
                    .map_err(|e| candle_core::Error::Msg(format!("affine4 tiled: {e}")))?;
                    return Ok(y);
                }

                if proj_candle_queue {
                    if let Device::Metal(metal_dev) = x.device() {
                        let encoder = metal_dev.command_encoder().map_err(|e| {
                            candle_core::Error::Msg(format!("candle command_encoder: {e}"))
                        })?;
                        encoder.set_label("lumen:affine4_matmul");
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
                        return Ok(y);
                    }
                }
                if let Device::Metal(metal_dev) = x.device() {
                    metal_dev.wait_until_completed()?;
                }
                ctx.matmul_zero_copy(weight, x_buf, x_offset, y_buf, y_offset, batch)
                    .map_err(|e| candle_core::Error::Msg(format!("affine4 zero-copy: {e}")))?;
                return Ok(y);
            }
        }
    }

    // CPU fallback: flatten → host → kernel → host → tensor.
    let x_flat = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let y_vec = ctx
        .matmul_with_weight(weight, &x_flat, batch)
        .map_err(|e| candle_core::Error::Msg(format!("affine4 matmul: {e}")))?;
    Tensor::from_vec(y_vec, out_shape, x.device())
}

/// Drop-in replacement for `candle_nn::Linear` whose weight is 4-bit affine quantized
/// and held on GPU (`Affine4Weight`). f32 in, f32 out.
pub struct Affine4Linear {
    weight: Affine4Weight,
    bias: Option<Tensor>,
    ctx: Arc<Affine4Context>,
    /// lazy-init ICB cache for the bf16-in/bf16-out fast path.
    /// `None` until the first `forward_*_icb` call.
    icb_cache: Mutex<Option<Affine4LinearIcbCache>>,
}

impl Affine4Linear {
    pub fn new(weight: Affine4Weight, bias: Option<Tensor>, ctx: Arc<Affine4Context>) -> Self {
        Self {
            weight,
            bias,
            ctx,
            icb_cache: Mutex::new(None),
        }
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

    pub fn weight(&self) -> &Affine4Weight {
        &self.weight
    }

    pub fn ctx(&self) -> &Arc<Affine4Context> {
        &self.ctx
    }

    /// Encode the matmul into an existing command encoder using raw Metal
    /// buffers + offsets (no Tensor wrapping). Matches `Mxfp4Linear::encode_native`
    /// so callers can fuse the projection into a wider command buffer.
    ///
    /// Picks `qmv_fast` when the shape qualifies (`in % 512 == 0` and
    /// `out % 8 == 0`), otherwise falls back to `encode_matmul_dispatch`.
    /// The bias path is **not** handled here — callers that need bias should
    /// add it as a separate dispatch (or use `forward`).
    pub fn encode_native(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) {
        let qmv_fast_ok = qmv_fast_enabled()
            && Affine4Context::qmv_fast_supports(self.weight.in_features, self.weight.out_features);
        if qmv_fast_ok {
            self.ctx.encode_qmv_fast_dispatch(
                encoder,
                &self.weight,
                x_buf,
                x_offset,
                y_buf,
                y_offset,
                batch,
            );
        } else {
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
    }

    /// `y = x @ W^T (+ bias)` with W held on GPU in 4-bit affine format.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = affine4_matmul_tensor(&self.ctx, &self.weight, x)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let y = match &self.bias {
            Some(b) => y.broadcast_add(b).map_err(|e| anyhow::anyhow!("{e}"))?,
            None => y,
        };
        Ok(y)
    }

    /// Fused `y = (x @ W^T) + residual` in a single GPU dispatch. Matches
    /// `Mxfp4Linear::forward_with_residual_f32`. f32 in, f32 out. Falls back to
    /// 2-step path when the input isn't Metal-resident.
    ///
    /// Routes by `in_features`:
    ///   - `in ≤ 8192`: v3-residual single-shot kernel
    ///   - `in > 8192` with clean tile: v3-tiled + residual-reduce (R4 + residual fusion)
    ///   - else: manual matmul + broadcast_add
    pub fn forward_with_residual_f32(&self, x: &Tensor, residual: &Tensor) -> Result<Tensor> {
        if !x.device().is_metal() {
            let y = self.forward(x)?;
            return y
                .broadcast_add(residual)
                .map_err(|e| anyhow::anyhow!("{e}"));
        }
        let dims = x.dims();
        let last = dims[dims.len() - 1];
        anyhow::ensure!(last == self.weight.in_features, "in dim mismatch");
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
        out_shape.push(self.weight.out_features);

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
        let y = Tensor::zeros(out_shape.clone(), DType::F32, x.device())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let (x_buf, x_offset) =
            metal_buffer_of(&x_f32).ok_or_else(|| anyhow::anyhow!("x not Metal"))?;
        let (r_buf, r_offset) =
            metal_buffer_of(&residual_f32).ok_or_else(|| anyhow::anyhow!("residual not Metal"))?;
        let (y_buf, y_offset) =
            metal_buffer_of(&y).ok_or_else(|| anyhow::anyhow!("y not Metal"))?;

        // R4 + residual fusion: when in > 8192 but a clean tile exists, use
        // tiled matmul + residual-reduction (saves one downstream broadcast_add
        // dispatch per call vs manual fallback).
        let disable_tiled = std::env::var("LUMEN_AFFINE4_DISABLE_TILED")
            .map(|v| v == "1")
            .unwrap_or(false);
        let tile_info = if disable_tiled {
            None
        } else {
            pick_tile_for_in(self.weight.in_features)
        };

        let qmv_fast_ok = qmv_fast_enabled()
            && Affine4Context::qmv_fast_supports(self.weight.in_features, self.weight.out_features);

        if qmv_fast_ok {
            // P3: MLX-pattern qmv_fast + residual fused.
            if let Device::Metal(metal_dev) = x.device() {
                let encoder = metal_dev
                    .command_encoder()
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                encoder.set_label("lumen:affine4_qmv_fast_residual");
                self.ctx.encode_qmv_fast_residual_dispatch(
                    encoder.as_ref(),
                    &self.weight,
                    x_buf,
                    x_offset,
                    r_buf,
                    r_offset,
                    y_buf,
                    y_offset,
                    batch,
                );
                drop(encoder);
            }
        } else if self.weight.in_features <= crate::affine4_gpu::AFFINE4_V3_MAX_IN_FEATURES {
            // v3 single-shot fused-residual kernel.
            if let Device::Metal(metal_dev) = x.device() {
                let encoder = metal_dev
                    .command_encoder()
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                encoder.set_label("lumen:affine4_matmul_residual");
                self.ctx.encode_matmul_residual_dispatch(
                    encoder.as_ref(),
                    &self.weight,
                    x_buf,
                    x_offset,
                    r_buf,
                    r_offset,
                    y_buf,
                    y_offset,
                    batch,
                );
                drop(encoder);
            }
        } else if let Some((tile_in, n_chunks)) = tile_info {
            let scratch_bytes =
                Affine4Context::tiled_scratch_bytes(self.weight.out_features, batch, n_chunks)
                    as u64;
            let scratch = self.ctx.ctx.buffer_zeroed(scratch_bytes);
            if let Device::Metal(metal_dev) = x.device() {
                let encoder = metal_dev
                    .command_encoder()
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                encoder.set_label("lumen:affine4_matmul_tiled_residual");
                self.ctx.encode_matmul_tiled_residual_dispatch(
                    encoder.as_ref(),
                    &self.weight,
                    x_buf,
                    x_offset,
                    &scratch,
                    r_buf,
                    r_offset,
                    y_buf,
                    y_offset,
                    tile_in,
                    n_chunks,
                    batch,
                );
                drop(encoder);
            }
        } else {
            // Last-resort fallback: 2-step path.
            let y = self.forward(x)?;
            return y
                .broadcast_add(residual)
                .map_err(|e| anyhow::anyhow!("{e}"));
        }
        Ok(y)
    }

    /// f32 in, bf16 out. Halves the output write traffic vs `forward`.
    ///
    /// Workstream B fast-path: when `x` is already bf16 (upstream produced
    /// bf16 via RmsNormBf16Out / a prior bf16-out matmul) AND the shape
    /// qualifies for qmv_fast, dispatch through the bf16-in/bf16-out kernel
    /// directly. This skips the prior f32-widening detour that caused
    /// `bf16_chain` to land σ=-268 NEGATIVE — every call site that takes
    /// `forward_bf16_out` benefits without code changes.
    pub fn forward_bf16_out(&self, x: &Tensor) -> Result<Tensor> {
        let qmv_fast_ok = qmv_fast_enabled()
            && x.device().is_metal()
            && Affine4Context::qmv_fast_supports(self.weight.in_features, self.weight.out_features);
        // bf16-in fast-path: dispatch direct through bf16-in/bf16-out qmv_fast
        // (Workstream A), no f32 widen detour.
        if qmv_fast_ok && x.dtype() == DType::BF16 {
            dispatch_counters::BF16_IN_FAST_PATH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return self.forward_bf16_in_bf16_out(x);
        }
        // f32-in fast-path: qmv_fast f32 (the prior LANDED Lever P0 — fastest
        // batch=1 dispatch on Affine4) followed by a single output cast. The
        // generic `matmul_f32in_bf16out_v3` kernel below is much slower than
        // qmv_fast f32 for batch=1 decode — the original bf16_chain σ=-262
        // NEGATIVE was caused by f32-input call sites (post-Lever H input-
        // rmsnorm fusion off, BF16_RMSNORM only fires on MXFP4 input proj)
        // hitting v3 here. One extra cast dispatch ≪ v3 vs qmv_fast gap.
        if qmv_fast_ok {
            dispatch_counters::F32_WIDEN_DETOUR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let y_f32 = self.forward(x)?;
            return y_f32
                .to_dtype(DType::BF16)
                .map_err(|e| anyhow::anyhow!("{e}"));
        }
        if !x.device().is_metal()
            || self.weight.in_features > crate::affine4_gpu::AFFINE4_V3_MAX_IN_FEATURES
        {
            // Fallback: f32 forward + cast.
            let y = self.forward(x)?;
            return y.to_dtype(DType::BF16).map_err(|e| anyhow::anyhow!("{e}"));
        }
        let dims = x.dims();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
        out_shape.push(self.weight.out_features);

        let x_f32 = if x.dtype() == DType::F32 {
            x.contiguous().map_err(|e| anyhow::anyhow!("{e}"))?
        } else {
            x.to_dtype(DType::F32)
                .and_then(|t| t.contiguous())
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let y = Tensor::zeros(out_shape.clone(), DType::BF16, x.device())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let (x_buf, x_offset) =
            metal_buffer_of(&x_f32).ok_or_else(|| anyhow::anyhow!("x not Metal"))?;
        let (y_buf, y_offset) =
            metal_buffer_of(&y).ok_or_else(|| anyhow::anyhow!("y not Metal"))?;

        if let Device::Metal(metal_dev) = x.device() {
            let encoder = metal_dev
                .command_encoder()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            encoder.set_label("lumen:affine4_matmul_bf16out");
            self.ctx.encode_matmul_bf16out_dispatch(
                encoder.as_ref(),
                &self.weight,
                x_buf,
                x_offset,
                y_buf,
                y_offset,
                batch,
            );
            drop(encoder);
        }
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

    /// bf16 in, bf16 out. Workstream B chain primitive — the building block
    /// for threading bf16 activations through projection sites without
    /// inserting an f32 detour. Routes to `qmv_fast` bf16 kernel (Workstream
    /// A) when shape qualifies; otherwise falls back to `forward_bf16_in`
    /// (bf16-in/f32-out) followed by an output cast (preserves BW saving on
    /// input read but loses it on output write).
    ///
    /// Caller contract: `x` must already be bf16 (or convertible). Use this
    /// over `forward_bf16_out` when the producer is bf16, otherwise the f32
    /// detour cancels out the input-side BW saving.
    pub fn forward_bf16_in_bf16_out(&self, x: &Tensor) -> Result<Tensor> {
        let qmv_fast_ok = qmv_fast_enabled()
            && x.device().is_metal()
            && Affine4Context::qmv_fast_supports(self.weight.in_features, self.weight.out_features);
        if !qmv_fast_ok {
            // Fallback: bf16-in path produces f32, then narrow to bf16. Loses
            // output-side BW saving for these shapes; vocab%8≠0 and other
            // misaligned shapes land here. Add a paired bf16-in/bf16-out
            // generic kernel later if profiling flags this fallback as hot.
            let y_f32 = self.forward_bf16_in(x)?;
            return y_f32
                .to_dtype(DType::BF16)
                .map_err(|e| anyhow::anyhow!("{e}"));
        }

        let dims = x.dims();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
        out_shape.push(self.weight.out_features);

        let x_bf16 = if x.dtype() == DType::BF16 {
            x.contiguous().map_err(|e| anyhow::anyhow!("{e}"))?
        } else {
            x.to_dtype(DType::BF16)
                .and_then(|t| t.contiguous())
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let y = Tensor::zeros(out_shape.clone(), DType::BF16, x.device())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let (x_buf, x_offset) =
            metal_buffer_of(&x_bf16).ok_or_else(|| anyhow::anyhow!("x not Metal"))?;
        let (y_buf, y_offset) =
            metal_buffer_of(&y).ok_or_else(|| anyhow::anyhow!("y not Metal"))?;

        if let Device::Metal(metal_dev) = x.device() {
            let encoder = metal_dev
                .command_encoder()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            encoder.set_label("lumen:affine4_qmv_fast_bf16in_bf16out");
            self.ctx.encode_qmv_fast_bf16in_bf16out_dispatch(
                encoder.as_ref(),
                &self.weight,
                x_buf,
                x_offset,
                y_buf,
                y_offset,
                batch,
            );
            drop(encoder);
        }
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

    /// bf16-in / bf16-out / bf16 residual fused fast-path. Closes the B.10
    /// σ-NEGATIVE root cause: under bf16 carrier, the f32 fused-residual
    /// `forward_with_residual_f32` cannot fire (caller carries bf16 residual),
    /// and the fallback splits into non-fused matmul + separate broadcast_add,
    /// adding ~+234 ms / decode batch in profile data.
    ///
    /// This method routes through the `qmv_fast_bf16in_bf16out_residual` Metal
    /// kernel — same fusion as `forward_with_residual_f32` but with bf16 I/O
    /// + bf16 residual + bf16 output. Internal accumulation stays in f32.
    ///
    /// Caller contract: both `x` and `residual` MUST be bf16 (or convertible).
    /// Output is bf16.
    pub fn forward_with_residual_bf16_in_bf16_out(
        &self,
        x: &Tensor,
        residual: &Tensor,
    ) -> Result<Tensor> {
        let qmv_fast_ok = qmv_fast_enabled()
            && x.device().is_metal()
            && Affine4Context::qmv_fast_supports(self.weight.in_features, self.weight.out_features);
        if !qmv_fast_ok {
            // Fallback: matmul bf16-in/bf16-out + separate broadcast_add(bf16).
            // Same path as before introducing the fused kernel.
            let y_bf16 = self.forward_bf16_in_bf16_out(x)?;
            return y_bf16
                .broadcast_add(residual)
                .map_err(|e| anyhow::anyhow!("{e}"));
        }

        let dims = x.dims();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
        out_shape.push(self.weight.out_features);

        let x_bf16 = if x.dtype() == DType::BF16 {
            x.contiguous().map_err(|e| anyhow::anyhow!("{e}"))?
        } else {
            x.to_dtype(DType::BF16)
                .and_then(|t| t.contiguous())
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let residual_bf16 = if residual.dtype() == DType::BF16 {
            residual.contiguous().map_err(|e| anyhow::anyhow!("{e}"))?
        } else {
            residual
                .to_dtype(DType::BF16)
                .and_then(|t| t.contiguous())
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let y = Tensor::zeros(out_shape.clone(), DType::BF16, x.device())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let (x_buf, x_offset) =
            metal_buffer_of(&x_bf16).ok_or_else(|| anyhow::anyhow!("x not Metal"))?;
        let (r_buf, r_offset) =
            metal_buffer_of(&residual_bf16).ok_or_else(|| anyhow::anyhow!("residual not Metal"))?;
        let (y_buf, y_offset) =
            metal_buffer_of(&y).ok_or_else(|| anyhow::anyhow!("y not Metal"))?;

        if let Device::Metal(metal_dev) = x.device() {
            let encoder = metal_dev
                .command_encoder()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            encoder.set_label("lumen:affine4_qmv_fast_bf16in_bf16out_residual");
            self.ctx.encode_qmv_fast_bf16in_bf16out_residual_dispatch(
                encoder.as_ref(),
                &self.weight,
                x_buf,
                x_offset,
                r_buf,
                r_offset,
                y_buf,
                y_offset,
                batch,
            );
            drop(encoder);
        }

        // Bias still applies post-fusion (matches `forward_bf16_in_bf16_out` pattern).
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

    /// bf16 in, f32 out. Halves the input read traffic vs `forward`.
    ///
    /// Workstream B fast-path: when shape qualifies for qmv_fast, dispatch
    /// through bf16-in/bf16-out qmv_fast (Workstream A) + cast bf16→f32 to
    /// preserve caller's f32-out contract. The v3 generic `bf16in_f32out_v3`
    /// kernel that was used here previously is **much slower than qmv_fast**
    /// for batch=1 decode — the prior bf16_chain σ=-262 NEGATIVE was caused
    /// by this v3 fallback (not the f32 widen detour we patched in
    /// `forward_bf16_out`). Bench hit `bf16_in_path` first (self_attn.rs:701)
    /// and this function reached for v3 instead of qmv_fast.
    pub fn forward_bf16_in(&self, x: &Tensor) -> Result<Tensor> {
        if x.dtype() == DType::BF16
            && qmv_fast_enabled()
            && x.device().is_metal()
            && Affine4Context::qmv_fast_supports(self.weight.in_features, self.weight.out_features)
        {
            // qmv_fast bf16-in/bf16-out → cast back to f32 to preserve
            // caller contract. Single cast cost ≪ v3 vs qmv_fast gap.
            let y_bf16 = self.forward_bf16_in_bf16_out(x)?;
            return y_bf16
                .to_dtype(DType::F32)
                .map_err(|e| anyhow::anyhow!("{e}"));
        }
        if !x.device().is_metal()
            || self.weight.in_features > crate::affine4_gpu::AFFINE4_V3_MAX_IN_FEATURES
        {
            // Fallback: widen to f32 + standard forward.
            let x_f32 = if x.dtype() == DType::F32 {
                x.clone()
            } else {
                x.to_dtype(DType::F32).map_err(|e| anyhow::anyhow!("{e}"))?
            };
            return self.forward(&x_f32);
        }
        let dims = x.dims();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
        out_shape.push(self.weight.out_features);

        let x_bf16 = if x.dtype() == DType::BF16 {
            x.contiguous().map_err(|e| anyhow::anyhow!("{e}"))?
        } else {
            x.to_dtype(DType::BF16)
                .and_then(|t| t.contiguous())
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let y = Tensor::zeros(out_shape.clone(), DType::F32, x.device())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let (x_buf, x_offset) =
            metal_buffer_of(&x_bf16).ok_or_else(|| anyhow::anyhow!("x not Metal"))?;
        let (y_buf, y_offset) =
            metal_buffer_of(&y).ok_or_else(|| anyhow::anyhow!("y not Metal"))?;

        if let Device::Metal(metal_dev) = x.device() {
            let encoder = metal_dev
                .command_encoder()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            encoder.set_label("lumen:affine4_matmul_bf16in");
            self.ctx.encode_matmul_bf16in_dispatch(
                encoder.as_ref(),
                &self.weight,
                x_buf,
                x_offset,
                y_buf,
                y_offset,
                batch,
            );
            drop(encoder);
        }
        let y = match &self.bias {
            Some(b) => y.broadcast_add(b).map_err(|e| anyhow::anyhow!("{e}"))?,
            None => y,
        };
        Ok(y)
    }

    /// Fused `y = ((x_raw * inv_rms) * rms_weight) @ W^T` in one GPU dispatch.
    /// Saves one Candle RmsNorm dispatch per qkv/gate_up call. Falls back to a
    /// 2-step (manual rmsnorm + standard forward) path when v3 doesn't fit.
    pub fn forward_with_rmsnorm(
        &self,
        x_raw: &Tensor,
        rms_weight: &Tensor,
        rms_eps: f32,
    ) -> Result<Tensor> {
        // Fall back when input not Metal-resident or shape exceeds v3 budget.
        if !x_raw.device().is_metal()
            || self.weight.in_features > crate::affine4_gpu::AFFINE4_V3_MAX_IN_FEATURES
        {
            // Manual rmsnorm: y = forward(x * rms_weight * inv_rms).
            let last = x_raw.dims().len() - 1;
            let sq = x_raw.sqr().map_err(|e| anyhow::anyhow!("{e}"))?;
            let mean_sq = sq.mean_keepdim(last).map_err(|e| anyhow::anyhow!("{e}"))?;
            let inv_rms = (mean_sq + rms_eps as f64)
                .and_then(|t| t.sqrt())
                .and_then(|t| t.recip())
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let normed = x_raw
                .broadcast_mul(&inv_rms)
                .and_then(|t| t.broadcast_mul(rms_weight))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            return self.forward(&normed);
        }
        let dims = x_raw.dims();
        let last = dims[dims.len() - 1];
        anyhow::ensure!(last == self.weight.in_features, "in dim mismatch");
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
        out_shape.push(self.weight.out_features);

        let x_f32 = if x_raw.dtype() == DType::F32 {
            x_raw.contiguous().map_err(|e| anyhow::anyhow!("{e}"))?
        } else {
            x_raw
                .to_dtype(DType::F32)
                .and_then(|t| t.contiguous())
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let rms_f32 = if rms_weight.dtype() == DType::F32 {
            rms_weight
                .contiguous()
                .map_err(|e| anyhow::anyhow!("{e}"))?
        } else {
            rms_weight
                .to_dtype(DType::F32)
                .and_then(|t| t.contiguous())
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let y = Tensor::zeros(out_shape.clone(), DType::F32, x_raw.device())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let (x_buf, x_offset) =
            metal_buffer_of(&x_f32).ok_or_else(|| anyhow::anyhow!("x not Metal"))?;
        let (rms_buf, rms_offset) =
            metal_buffer_of(&rms_f32).ok_or_else(|| anyhow::anyhow!("rms not Metal"))?;
        let (y_buf, y_offset) =
            metal_buffer_of(&y).ok_or_else(|| anyhow::anyhow!("y not Metal"))?;

        // P5: prefer MLX-pattern qmv_fast_rmsnorm when shape qualifies. This
        // closes the gap to MLX's hand-tuned RmsNorm-fused matmul: cooperative
        // inv_rms reduction, then qmv_fast inner loop with x folded by
        // `inv_rms * rms_weight`. 64 threads/TG, no big TG memory cache.
        let qmv_fast_rms_ok = qmv_fast_enabled()
            && Affine4Context::qmv_fast_supports(self.weight.in_features, self.weight.out_features);
        if qmv_fast_rms_ok {
            if let Device::Metal(metal_dev) = x_raw.device() {
                let encoder = metal_dev
                    .command_encoder()
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                encoder.set_label("lumen:affine4_qmv_fast_rmsnorm");
                self.ctx.encode_qmv_fast_rmsnorm_dispatch(
                    encoder.as_ref(),
                    &self.weight,
                    x_buf,
                    x_offset,
                    rms_buf,
                    rms_offset,
                    y_buf,
                    y_offset,
                    rms_eps,
                    batch,
                );
                drop(encoder);
            }
            let y = match &self.bias {
                Some(b) => y.broadcast_add(b).map_err(|e| anyhow::anyhow!("{e}"))?,
                None => y,
            };
            return Ok(y);
        }

        // R1 debug: temporarily route through own-queue (sync wait per call)
        // instead of Candle's command encoder. Adds cross-queue wait_until_completed
        // overhead but isolates from any Candle encoder lifecycle interaction.
        // If this restores reasonable throughput, the bug is in encoder reuse.
        // Toggle via `LUMEN_AFFINE4_RMSNORM_OWN_QUEUE=1` (default ON for debug).
        let own_queue = std::env::var("LUMEN_AFFINE4_RMSNORM_OWN_QUEUE")
            .map(|v| v != "0")
            .unwrap_or(true);
        if own_queue {
            if let Device::Metal(metal_dev) = x_raw.device() {
                metal_dev
                    .wait_until_completed()
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            self.ctx
                .matmul_rmsnorm_zero_copy(
                    &self.weight,
                    x_buf,
                    x_offset,
                    rms_buf,
                    rms_offset,
                    y_buf,
                    y_offset,
                    rms_eps,
                    batch,
                )
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        } else if let Device::Metal(metal_dev) = x_raw.device() {
            let encoder = metal_dev
                .command_encoder()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            encoder.set_label("lumen:affine4_matmul_rmsnorm");
            self.ctx.encode_matmul_rmsnorm_dispatch(
                encoder.as_ref(),
                &self.weight,
                x_buf,
                x_offset,
                rms_buf,
                rms_offset,
                y_buf,
                y_offset,
                rms_eps,
                batch,
            );
            drop(encoder);
        }
        let y = match &self.bias {
            Some(b) => y.broadcast_add(b).map_err(|e| anyhow::anyhow!("{e}"))?,
            None => y,
        };
        Ok(y)
    }

    /// Fused `silu(x @ W_gate^T) * (x @ W_up^T)` in a single GPU dispatch.
    /// Replaces 4 dispatches (2 matmul + silu + mul) with 1. `weight.out_features`
    /// must equal `2 * inter` (gate rows concatenated above up rows).
    pub fn forward_gate_up_silu_mul(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_gate_up_silu_mul_inner(x, false)
    }

    /// Same fused gate+up SiLU mul but stores output as bf16.
    pub fn forward_gate_up_silu_mul_bf16_out(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_gate_up_silu_mul_inner(x, true)
    }

    fn forward_gate_up_silu_mul_inner(&self, x: &Tensor, bf16_out: bool) -> Result<Tensor> {
        let inter = self.weight.out_features / 2;
        if !x.device().is_metal()
            || self.weight.in_features > crate::affine4_gpu::AFFINE4_V3_MAX_IN_FEATURES
            || self.weight.out_features % 2 != 0
        {
            // Fallback: standard matmul + manual split + silu*mul.
            let combined = self.forward(x)?;
            let last = combined.dims().len() - 1;
            let gate = combined
                .narrow(last, 0, inter)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .contiguous()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let up = combined
                .narrow(last, inter, inter)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .contiguous()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let h = (candle_nn::ops::silu(&gate).map_err(|e| anyhow::anyhow!("{e}"))? * up)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            return if bf16_out {
                h.to_dtype(DType::BF16).map_err(|e| anyhow::anyhow!("{e}"))
            } else {
                Ok(h)
            };
        }
        let dims = x.dims();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
        out_shape.push(inter);

        let x_f32 = if x.dtype() == DType::F32 {
            x.contiguous().map_err(|e| anyhow::anyhow!("{e}"))?
        } else {
            x.to_dtype(DType::F32)
                .and_then(|t| t.contiguous())
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let y_dtype = if bf16_out { DType::BF16 } else { DType::F32 };
        let y = Tensor::zeros(out_shape.clone(), y_dtype, x.device())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let (x_buf, x_offset) =
            metal_buffer_of(&x_f32).ok_or_else(|| anyhow::anyhow!("x not Metal"))?;
        let (y_buf, y_offset) =
            metal_buffer_of(&y).ok_or_else(|| anyhow::anyhow!("y not Metal"))?;

        if let Device::Metal(metal_dev) = x.device() {
            let encoder = metal_dev
                .command_encoder()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            encoder.set_label("lumen:affine4_gate_up_silu_mul");
            self.ctx.encode_gate_up_silu_mul_dispatch(
                encoder.as_ref(),
                &self.weight,
                x_buf,
                x_offset,
                y_buf,
                y_offset,
                batch,
                bf16_out,
            );
            drop(encoder);
        }
        Ok(y)
    }

    /// bf16-in / bf16-out via ICB record-once-replay-many.
    ///
    /// Mirrors [`Self::forward_bf16_in_bf16_out`] semantically but routes the
    /// dispatch through a lazily-recorded `IndirectCommandBuffer`. Falls back
    /// to the standard path when (a) `LUMEN_ICB=1` env-gate is off, (b) the
    /// shape doesn't qualify for `qmv_fast`, or (c) the producer's buffer
    /// addresses change between calls (rare; a re-record fires).
    pub fn forward_bf16_in_bf16_out_icb(&self, x: &Tensor) -> Result<Tensor> {
        if !icb_enabled() {
            return self.forward_bf16_in_bf16_out(x);
        }
        let qmv_fast_ok = qmv_fast_enabled()
            && x.device().is_metal()
            && Affine4Context::qmv_fast_supports(self.weight.in_features, self.weight.out_features);
        if !qmv_fast_ok {
            return self.forward_bf16_in_bf16_out(x);
        }

        let dims = x.dims();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
        out_shape.push(self.weight.out_features);

        let x_bf16 = if x.dtype() == DType::BF16 {
            x.contiguous().map_err(|e| anyhow::anyhow!("{e}"))?
        } else {
            x.to_dtype(DType::BF16)
                .and_then(|t| t.contiguous())
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let y = Tensor::zeros(out_shape.clone(), DType::BF16, x.device())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let (x_buf, x_offset) =
            metal_buffer_of(&x_bf16).ok_or_else(|| anyhow::anyhow!("x not Metal"))?;
        let (y_buf, y_offset) =
            metal_buffer_of(&y).ok_or_else(|| anyhow::anyhow!("y not Metal"))?;

        let metal_dev = match x.device() {
            Device::Metal(m) => m,
            _ => unreachable!("guarded by qmv_fast_ok"),
        };

        let mut cache_guard = self
            .icb_cache
            .lock()
            .map_err(|e| anyhow::anyhow!("icb cache lock: {e}"))?;
        if cache_guard.is_none() {
            *cache_guard = Some(Affine4LinearIcbCache::new(&self.ctx, &self.weight, batch)?);
        }
        let cache = cache_guard.as_mut().unwrap();

        // Re-allocate `batch_buf` if batch size changed (cold path; decode
        // typically holds batch=1).
        if cache.bound_batch != batch {
            cache.batch_buf = self.ctx.ctx.buffer_with_data(&[batch as u32]);
            cache.bound_batch = batch;
            cache.nr_recorded = false;
            cache.rs_recorded = false;
        }

        let x_id = buffer_id(x_buf);
        let y_id = buffer_id(y_buf);
        let needs_record = !cache.nr_recorded
            || cache.nr_x_id != x_id
            || cache.nr_x_off != x_offset
            || cache.nr_y_id != y_id
            || cache.nr_y_off != y_offset;

        if needs_record {
            self.ctx.record_qmv_fast_bf16in_bf16out_icb(
                &cache.icb_no_residual,
                0,
                &self.weight,
                x_buf,
                x_offset,
                y_buf,
                y_offset,
                &cache.dims_buf,
                &cache.batch_buf,
                batch,
            );
            cache.nr_x_id = x_id;
            cache.nr_x_off = x_offset;
            cache.nr_y_id = y_id;
            cache.nr_y_off = y_offset;
            cache.nr_recorded = true;
        }

        let encoder = metal_dev
            .command_encoder()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        encoder.set_label("lumen:affine4_qmv_fast_bf16in_bf16out_icb");
        let usage = MTLResourceUsage(MTLResourceUsage::Read.0 | MTLResourceUsage::Write.0);
        let (wp, ws, wb) = self.weight.buffers();
        // single FFI call for all 7 buffers (was 7 separate
        // useResource calls; PoC #1 N=1 -2.87% trace pinpointed this).
        encoder.use_buffers_for_icb(
            &[wp, ws, wb, x_buf, y_buf, &cache.dims_buf, &cache.batch_buf],
            usage,
        );
        // Order the ICB dispatch against the rest of this concurrent encoder:
        // before, so `x`'s producer and `y`'s zero-fill have landed; after, so
        // the consumer that reads `y` back sees the finished result. Candle's
        // hazard tracker cannot see ICB-bound buffers, so it emits neither.
        encoder.insert_memory_barrier();
        encoder.execute_commands_in_buffer(&cache.icb_no_residual, 1);
        encoder.insert_memory_barrier();
        drop(encoder);
        drop(cache_guard);

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

    /// bf16-in / bf16-residual / bf16-out via ICB. Mirrors
    /// [`Self::forward_with_residual_bf16_in_bf16_out`].
    pub fn forward_with_residual_bf16_in_bf16_out_icb(
        &self,
        x: &Tensor,
        residual: &Tensor,
    ) -> Result<Tensor> {
        if !icb_enabled() {
            return self.forward_with_residual_bf16_in_bf16_out(x, residual);
        }
        let qmv_fast_ok = qmv_fast_enabled()
            && x.device().is_metal()
            && Affine4Context::qmv_fast_supports(self.weight.in_features, self.weight.out_features);
        if !qmv_fast_ok {
            return self.forward_with_residual_bf16_in_bf16_out(x, residual);
        }

        let dims = x.dims();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
        out_shape.push(self.weight.out_features);

        let x_bf16 = if x.dtype() == DType::BF16 {
            x.contiguous().map_err(|e| anyhow::anyhow!("{e}"))?
        } else {
            x.to_dtype(DType::BF16)
                .and_then(|t| t.contiguous())
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let residual_bf16 = if residual.dtype() == DType::BF16 {
            residual.contiguous().map_err(|e| anyhow::anyhow!("{e}"))?
        } else {
            residual
                .to_dtype(DType::BF16)
                .and_then(|t| t.contiguous())
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let y = Tensor::zeros(out_shape.clone(), DType::BF16, x.device())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let (x_buf, x_offset) =
            metal_buffer_of(&x_bf16).ok_or_else(|| anyhow::anyhow!("x not Metal"))?;
        let (r_buf, r_offset) =
            metal_buffer_of(&residual_bf16).ok_or_else(|| anyhow::anyhow!("residual not Metal"))?;
        let (y_buf, y_offset) =
            metal_buffer_of(&y).ok_or_else(|| anyhow::anyhow!("y not Metal"))?;

        let metal_dev = match x.device() {
            Device::Metal(m) => m,
            _ => unreachable!("guarded by qmv_fast_ok"),
        };

        let mut cache_guard = self
            .icb_cache
            .lock()
            .map_err(|e| anyhow::anyhow!("icb cache lock: {e}"))?;
        if cache_guard.is_none() {
            *cache_guard = Some(Affine4LinearIcbCache::new(&self.ctx, &self.weight, batch)?);
        }
        let cache = cache_guard.as_mut().unwrap();

        if cache.bound_batch != batch {
            cache.batch_buf = self.ctx.ctx.buffer_with_data(&[batch as u32]);
            cache.bound_batch = batch;
            cache.nr_recorded = false;
            cache.rs_recorded = false;
        }

        let x_id = buffer_id(x_buf);
        let r_id = buffer_id(r_buf);
        let y_id = buffer_id(y_buf);
        let needs_record = !cache.rs_recorded
            || cache.rs_x_id != x_id
            || cache.rs_x_off != x_offset
            || cache.rs_r_id != r_id
            || cache.rs_r_off != r_offset
            || cache.rs_y_id != y_id
            || cache.rs_y_off != y_offset;

        if needs_record {
            self.ctx.record_qmv_fast_bf16in_bf16out_residual_icb(
                &cache.icb_residual,
                0,
                &self.weight,
                x_buf,
                x_offset,
                r_buf,
                r_offset,
                y_buf,
                y_offset,
                &cache.dims_buf,
                &cache.batch_buf,
                batch,
            );
            cache.rs_x_id = x_id;
            cache.rs_x_off = x_offset;
            cache.rs_r_id = r_id;
            cache.rs_r_off = r_offset;
            cache.rs_y_id = y_id;
            cache.rs_y_off = y_offset;
            cache.rs_recorded = true;
        }

        let encoder = metal_dev
            .command_encoder()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        encoder.set_label("lumen:affine4_qmv_fast_bf16in_bf16out_residual_icb");
        let usage = MTLResourceUsage(MTLResourceUsage::Read.0 | MTLResourceUsage::Write.0);
        let (wp, ws, wb) = self.weight.buffers();
        // single FFI call for all 8 buffers (was 8 separate).
        encoder.use_buffers_for_icb(
            &[
                wp,
                ws,
                wb,
                x_buf,
                r_buf,
                y_buf,
                &cache.dims_buf,
                &cache.batch_buf,
            ],
            usage,
        );
        // See the no-residual variant: ICB-bound buffers bypass candle's
        // hazard tracker, so the ordering has to be stated explicitly.
        encoder.insert_memory_barrier();
        encoder.execute_commands_in_buffer(&cache.icb_residual, 1);
        encoder.insert_memory_barrier();
        drop(encoder);
        drop(cache_guard);

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
}

#[allow(dead_code)]
fn _buf_arg_ref(b: &Buffer) -> &Buffer {
    b
}
