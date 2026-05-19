//! Native Metal `RmsNorm` kernels — bit-deterministic replacement for the
//! MPSGraph-based `MpsRmsNormBf16Out`. The MPSGraph version passed scalar
//! parity (≤6e-3 vs CPU) but produced non-bit-equal output across calls on
//! identical inputs (5119/5120 bits flipped — Apple-internal reduction-order
//! optimization), breaking the bf16 chain's `R1↔R2` token determinism.
//!
//! The kernel here pins reduction order via `simd_sum` + threadgroup memory
//! and uses fp32 accumulation throughout, so output is bit-stable across
//! invocations.

#[cfg(feature = "model-integration")]
use crate::metal::{BatchedEncoderExt, CommandBufferExt, ComputeEncoderCompat};
use anyhow::Result;
#[cfg(feature = "model-integration")]
use candle_core::{DType, Device, Storage, Tensor};

#[cfg(feature = "model-integration")]
use crate::metal::{Buffer, ComputePipelineState, IndirectCommandBuffer, MTLSize};

#[cfg(feature = "model-integration")]
const SHADER_SRC: &str = include_str!("shaders/rms_norm.metal");

#[cfg(feature = "model-integration")]
#[repr(C)]
#[derive(Clone, Copy)]
struct RmsNormBf16OutDims {
    hidden: u32,
    eps: f32,
}

/// Native Metal `RmsNormBf16Out`. Drop-in replacement for the MPSGraph
/// version with a bit-identical-across-calls determinism guarantee.
///
/// Caller contract matches `MpsRmsNormBf16Out`:
///   - `x`     : `f32` tensor on Metal device, last dim = `hidden`
///   - `weight`: `f32` tensor `[hidden]` on Metal device
///   - returns : `bf16` tensor with `x`'s shape
///
/// Pipeline state is compiled once per process; instantiate one per `eps`.
#[cfg(feature = "model-integration")]
pub struct RmsNormBf16Out {
    pipeline: ComputePipelineState,
    eps: f32,
}

#[cfg(feature = "model-integration")]
impl RmsNormBf16Out {
    pub fn new(eps: f32) -> Result<Self> {
        let ctx = crate::device::MetalContext::new()?;
        let opts = crate::metal::new_compile_options();
        // Version 3.1 required for the `bfloat` type used at the bf16 store.
        opts.set_language_version(crate::metal::MTLLanguageVersion::Version3_1);
        let lib = ctx
            .device
            .new_library_with_source(SHADER_SRC, Some(opts.as_ref()))
            .map_err(|e| anyhow::anyhow!("rms_norm shader compile: {e}"))?;
        let func = lib
            .get_function("rms_norm_f32in_bf16out", None)
            .map_err(|e| anyhow::anyhow!("rms_norm: get_function: {e}"))?;
        let pipeline = ctx
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow::anyhow!("rms_norm: pipeline: {e}"))?;
        Ok(Self { pipeline, eps })
    }

    /// `y = bf16(x · rsqrt(mean(x², axis=-1) + eps) · weight)`. Uses the
    /// caller's Metal command queue (Candle device queue) so the dispatch
    /// fuses with surrounding kernels' submission stream.
    pub fn forward(&self, x: &Tensor, weight: &Tensor) -> candle_core::Result<Tensor> {
        if x.dtype() != DType::F32 || weight.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(
                "RmsNormBf16Out: inputs must be f32 (only the output is bf16)".into(),
            ));
        }
        let dims = x.dims();
        if dims.len() < 2 {
            return Err(candle_core::Error::Msg("RmsNormBf16Out: rank < 2".into()));
        }
        let hidden = *dims.last().unwrap();
        let m: usize = dims[..dims.len() - 1].iter().product();

        if weight.dims() != [hidden] {
            return Err(candle_core::Error::Msg(format!(
                "RmsNormBf16Out: weight shape {:?} mismatches hidden {hidden}",
                weight.dims()
            )));
        }

        let x = x.contiguous()?;
        let weight = weight.contiguous()?;

        // Bake bf16 zeros at allocation — same pattern as MpsRmsNormBf16Out
        // to avoid the Candle pool's `fill_buffer(0)` blit landing after our
        // kernel encode in the queue's command-buffer ordering.
        let n: usize = x.dims().iter().product();
        let zeros: Vec<half::bf16> = vec![half::bf16::ZERO; n];
        let y = Tensor::from_vec(zeros, x.shape(), x.device())?.contiguous()?;

        let metal_dev = match x.device() {
            Device::Metal(d) => d,
            _ => {
                return Err(candle_core::Error::Msg(
                    "RmsNormBf16Out: tensor not on Metal device".into(),
                ));
            }
        };

        let (x_buf, x_off) = buffer_from_tensor(&x)?;
        let (w_buf, w_off) = buffer_from_tensor(&weight)?;
        let (y_buf, y_off) = buffer_from_tensor(&y)?;

        let encoder = metal_dev
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("rms_norm: encoder: {e}")))?;
        encoder.set_label("lumen:rms_norm_bf16_out");
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(x_buf), x_off as usize);
        encoder.set_buffer(1, Some(w_buf), w_off as usize);
        encoder.set_buffer(2, Some(y_buf), y_off as usize);
        let dims_struct = RmsNormBf16OutDims {
            hidden: hidden as u32,
            eps: self.eps,
        };
        encoder.set_bytes_directly(
            3,
            std::mem::size_of::<RmsNormBf16OutDims>(),
            &dims_struct as *const _ as *const _,
        );
        // Threadgroup memory: NSG=8 floats for the partial-sum reduction tier.
        encoder.set_threadgroup_memory_length(0, 8 * std::mem::size_of::<f32>());

        const THREADS_PER_TG: usize = 256;
        let grid = MTLSize {
            width: m,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid, tg);
        drop(encoder);
        Ok(y)
    }
}

/// Workstream B Phase 10 — bf16-in / bf16-out RmsNorm. Companion to
/// [`RmsNormBf16Out`] for the model-wide bf16 carrier stream: lifts the
/// f32-input requirement so the bf16 residual carrier never has to be
/// cast back to f32 at the layernorm boundary.
///
/// Caller contract:
///   - `x`     : `bf16` tensor on Metal device, last dim = `hidden`
///   - `weight`: `f32`  tensor `[hidden]` on Metal device
///   - returns : `bf16` tensor with `x`'s shape
///
/// Internal accumulation widened to f32 (Escape pattern — bf16 mantissa
/// too narrow for hidden∈[2K..16K] reductions). Output store narrows
/// once at the end. Determinism guarantee identical to `RmsNormBf16Out`.
#[cfg(feature = "model-integration")]
pub struct RmsNormBf16InBf16Out {
    pipeline: ComputePipelineState,
    /// ICB-compatible variant of `pipeline`. Compiled with
    /// `setSupportIndirectCommandBuffers=true` so it can be assigned to
    /// `MTLIndirectComputeCommand`. Output bit-identical to `pipeline`.
    pipeline_icb: ComputePipelineState,
    /// Pre-allocated `[hidden, eps]` buffer for ICB record paths (which
    /// can't use `setBytes` like the standard path). Constructed once per
    /// `RmsNormBf16InBf16Out` instance — `eps` and `hidden` are both
    /// instance-scoped invariants.
    /// Note: `hidden` is NOT known at construction time; we lazy-allocate
    /// `dims_bufs_by_hidden` keyed by hidden when ICB is first recorded.
    /// For now we expose constructors that bake `hidden` upfront via
    /// `record_icb`'s caller-supplied `dims_buf`.
    eps: f32,
    /// Cached `MetalContext` so callers can request a `dims_buf` of the
    /// right shape without re-grabbing the device handle.
    ctx: crate::device::MetalContext,
}

#[cfg(feature = "model-integration")]
impl RmsNormBf16InBf16Out {
    pub fn new(eps: f32) -> Result<Self> {
        let ctx = crate::device::MetalContext::new()?;
        let opts = crate::metal::new_compile_options();
        opts.set_language_version(crate::metal::MTLLanguageVersion::Version3_1);
        let lib = ctx
            .device
            .new_library_with_source(SHADER_SRC, Some(opts.as_ref()))
            .map_err(|e| anyhow::anyhow!("rms_norm shader compile: {e}"))?;
        let func = lib
            .get_function("rms_norm_bf16in_bf16out", None)
            .map_err(|e| anyhow::anyhow!("rms_norm: get_function: {e}"))?;
        let pipeline = ctx
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow::anyhow!("rms_norm: pipeline: {e}"))?;
        let pipeline_icb = ctx
            .device
            .new_compute_pipeline_state_with_function_for_icb(&func)
            .map_err(|e| anyhow::anyhow!("rms_norm: pipeline_icb: {e:?}"))?;
        Ok(Self {
            pipeline,
            pipeline_icb,
            eps,
            ctx,
        })
    }

    /// Allocate a `dims_buf` containing `[hidden, eps]` for use with
    /// [`Self::record_icb`]. The buffer is small (8 bytes) and ties to
    /// this instance's `eps`. Callers should hold it for as long as the
    /// ICB references it.
    pub fn make_dims_buf(&self, hidden: usize) -> Buffer {
        let dims = RmsNormBf16OutDims {
            hidden: hidden as u32,
            eps: self.eps,
        };
        self.ctx.buffer_with_data(&[dims])
    }

    /// record a single `rms_norm_bf16in_bf16out` dispatch
    /// into `icb` at `slot`. Replay via
    /// `encoder.use_buffers_for_icb([x, weight, y, dims_buf]) +
    ///  execute_commands_in_buffer(&icb, count)`.
    ///
    /// `m` is the row count (typically batch × seq). Threadgroup memory
    /// (8 floats for the simdgroup partial-sum reduction) is set on the
    /// recorded command, matching the non-ICB encode path.
    pub fn record_icb(
        &self,
        icb: &IndirectCommandBuffer,
        slot: usize,
        x_buf: &Buffer,
        x_off: u64,
        w_buf: &Buffer,
        w_off: u64,
        y_buf: &Buffer,
        y_off: u64,
        dims_buf: &Buffer,
        m: usize,
    ) {
        const THREADS_PER_TG: usize = 256;
        const SIMD_PARTIALS_BYTES: usize = 8 * std::mem::size_of::<f32>();
        let grid = MTLSize {
            width: m,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        icb.record_compute_full(
            slot,
            &self.pipeline_icb,
            &[
                (x_buf, x_off as usize, 0),
                (w_buf, w_off as usize, 1),
                (y_buf, y_off as usize, 2),
                (dims_buf, 0, 3),
            ],
            &[(SIMD_PARTIALS_BYTES, 0)],
            grid,
            tg,
        );
    }

    pub fn forward(&self, x: &Tensor, weight: &Tensor) -> candle_core::Result<Tensor> {
        if x.dtype() != DType::BF16 {
            return Err(candle_core::Error::Msg(
                "RmsNormBf16InBf16Out: x must be bf16".into(),
            ));
        }
        if weight.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(
                "RmsNormBf16InBf16Out: weight must be f32".into(),
            ));
        }
        let dims = x.dims();
        if dims.len() < 2 {
            return Err(candle_core::Error::Msg(
                "RmsNormBf16InBf16Out: rank < 2".into(),
            ));
        }
        let hidden = *dims.last().unwrap();
        let m: usize = dims[..dims.len() - 1].iter().product();

        if weight.dims() != [hidden] {
            return Err(candle_core::Error::Msg(format!(
                "RmsNormBf16InBf16Out: weight shape {:?} mismatches hidden {hidden}",
                weight.dims()
            )));
        }

        let x = x.contiguous()?;
        let weight = weight.contiguous()?;

        let n: usize = x.dims().iter().product();
        let zeros: Vec<half::bf16> = vec![half::bf16::ZERO; n];
        let y = Tensor::from_vec(zeros, x.shape(), x.device())?.contiguous()?;

        let metal_dev = match x.device() {
            Device::Metal(d) => d,
            _ => {
                return Err(candle_core::Error::Msg(
                    "RmsNormBf16InBf16Out: tensor not on Metal device".into(),
                ));
            }
        };

        let (x_buf, x_off) = buffer_from_tensor(&x)?;
        let (w_buf, w_off) = buffer_from_tensor(&weight)?;
        let (y_buf, y_off) = buffer_from_tensor(&y)?;

        let encoder = metal_dev
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("rms_norm: encoder: {e}")))?;
        encoder.set_label("lumen:rms_norm_bf16in_bf16out");
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(x_buf), x_off as usize);
        encoder.set_buffer(1, Some(w_buf), w_off as usize);
        encoder.set_buffer(2, Some(y_buf), y_off as usize);
        let dims_struct = RmsNormBf16OutDims {
            hidden: hidden as u32,
            eps: self.eps,
        };
        encoder.set_bytes_directly(
            3,
            std::mem::size_of::<RmsNormBf16OutDims>(),
            &dims_struct as *const _ as *const _,
        );
        encoder.set_threadgroup_memory_length(0, 8 * std::mem::size_of::<f32>());

        const THREADS_PER_TG: usize = 256;
        let grid = MTLSize {
            width: m,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid, tg);
        drop(encoder);
        Ok(y)
    }
}

#[cfg(feature = "model-integration")]
fn buffer_from_tensor(t: &Tensor) -> candle_core::Result<(&Buffer, u64)> {
    let (storage, layout) = t.storage_and_layout();
    match &*storage {
        Storage::Metal(ms) => {
            let off = (layout.start_offset() * t.dtype().size_in_bytes()) as u64;
            let candle_buf = ms.buffer();
            // Same transmute trick as crate::affine4_linear::metal_buffer_of —
            // candle's Metal buffer wraps the same objc2 type as ours.
            let buf_ptr = candle_buf as *const _ as *const Buffer;
            Ok((unsafe { &*buf_ptr }, off))
        }
        _ => Err(candle_core::Error::Msg(
            "rms_norm: tensor storage is not Metal".into(),
        )),
    }
}
