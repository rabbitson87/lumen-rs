//! standalone `silu(gate) * up` kernel + ICB-compat support.
//!
//! Production 27B Dense MLP currently runs the silu*mul step as a 5-dispatch
//! chain in `DenseMlp::forward_with_residual_bf16_in_bf16_out`:
//!
//!     combined_bf16 = gate_up_proj(x_bf16)        // 1 dispatch (Affine4)
//!     combined_f32  = combined_bf16.to_dtype(F32) // 1 dispatch
//!     gate          = narrow(combined_f32, 0)     // 0-1 dispatch (contiguous)
//!     up            = narrow(combined_f32, inter) // 0-1 dispatch
//!     hidden_f32    = silu(gate) * up             // 1-2 dispatch (silu + mul)
//!     down_proj(hidden_f32, residual)             // 1 dispatch (Affine4)
//!
//! This module collapses the 5 middle dispatches (cast + narrow×2 + silu*mul)
//! into a single bf16-in / bf16-out elementwise kernel. Output goes directly
//! into `down_proj.forward_with_residual_bf16_in_bf16_out` (which already
//! accepts bf16 input). Net per-MLP saving: 4 dispatches.
//!
//! Distinction from Phase 12/13's NEGATIVE `affine4_qmv_fast_gate_up_silu_mul_bf16_out`:
//! that kernel folded the matmul + silu + mul into one dispatch; the
//! resulting register pressure (8 result floats per simdgroup vs 4) made
//! the inner loop slower than the current Pareto-optimal "qmv_fast +
//! separate silu*mul". This module preserves that Pareto choice — it ports
//! ONLY the elementwise step, leaving gate_up_proj as its own qmv_fast
//! dispatch.

#[cfg(feature = "model-integration")]
use crate::metal::{BatchedEncoderExt, CommandBufferExt, ComputeEncoderCompat};
use anyhow::Result;
#[cfg(feature = "model-integration")]
use candle_core::{DType, Device, Storage, Tensor};

#[cfg(feature = "model-integration")]
use crate::metal::{Buffer, ComputePipelineState, IndirectCommandBuffer, MTLSize};

#[cfg(feature = "model-integration")]
const SHADER_SRC: &str = include_str!("shaders/silu_mul.metal");

#[cfg(feature = "model-integration")]
#[repr(C)]
#[derive(Clone, Copy)]
struct SiluMulDims {
    inter: u32,
}

/// `silu(gate) * up` elementwise kernel (bf16 in / bf16 out).
///
/// Caller contract:
///   - `combined`: bf16 tensor `[..., 2 * inter]` — gate concatenated with up
///   - returns: bf16 tensor `[..., inter]`
///
/// Bit-deterministic: kernel is purely elementwise (no reductions), so for
/// identical inputs the output is bit-equal across calls.
#[cfg(feature = "model-integration")]
pub struct SiluMulBf16InBf16Out {
    pipeline: ComputePipelineState,
    /// ICB-compat variant of `pipeline`.
    pipeline_icb: ComputePipelineState,
    ctx: crate::device::MetalContext,
}

#[cfg(feature = "model-integration")]
impl SiluMulBf16InBf16Out {
    pub fn new() -> Result<Self> {
        let ctx = crate::device::MetalContext::new()?;
        let opts = crate::metal::new_compile_options();
        opts.set_language_version(crate::metal::MTLLanguageVersion::Version3_1);
        let lib = ctx
            .device
            .new_library_with_source(SHADER_SRC, Some(opts.as_ref()))
            .map_err(|e| anyhow::anyhow!("silu_mul shader compile: {e}"))?;
        let func = lib
            .get_function("silu_mul_bf16in_bf16out", None)
            .map_err(|e| anyhow::anyhow!("silu_mul: get_function: {e}"))?;
        let pipeline = ctx
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow::anyhow!("silu_mul: pipeline: {e}"))?;
        let pipeline_icb = ctx
            .device
            .new_compute_pipeline_state_with_function_for_icb(&func)
            .map_err(|e| anyhow::anyhow!("silu_mul: pipeline_icb: {e:?}"))?;
        Ok(Self { pipeline, pipeline_icb, ctx })
    }

    /// Pre-allocate the small `[inter]` constant buffer for ICB use.
    pub fn make_dims_buf(&self, inter: usize) -> Buffer {
        self.ctx.buffer_with_data(&[SiluMulDims { inter: inter as u32 }])
    }

    /// Standalone (non-ICB) forward path. Allocates a fresh output Tensor.
    ///
    /// `combined.shape().last() == 2 * inter` is required.
    pub fn forward(&self, combined: &Tensor) -> candle_core::Result<Tensor> {
        if combined.dtype() != DType::BF16 {
            return Err(candle_core::Error::Msg(
                "SiluMulBf16InBf16Out: combined must be bf16".into(),
            ));
        }
        let dims = combined.dims();
        if dims.is_empty() {
            return Err(candle_core::Error::Msg(
                "SiluMulBf16InBf16Out: rank 0".into(),
            ));
        }
        let last = *dims.last().unwrap();
        if last % 2 != 0 {
            return Err(candle_core::Error::Msg(format!(
                "SiluMulBf16InBf16Out: last dim {last} not even (expected 2*inter)"
            )));
        }
        let inter = last / 2;
        let m: usize = dims[..dims.len() - 1].iter().product();

        let combined = combined.contiguous()?;

        let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
        out_shape.push(inter);

        let n_out: usize = out_shape.iter().product();
        let zeros: Vec<half::bf16> = vec![half::bf16::ZERO; n_out];
        let y = Tensor::from_vec(zeros, out_shape.as_slice(), combined.device())?
            .contiguous()?;

        let metal_dev = match combined.device() {
            Device::Metal(d) => d,
            _ => {
                return Err(candle_core::Error::Msg(
                    "SiluMulBf16InBf16Out: tensor not on Metal device".into(),
                ));
            }
        };

        let (c_buf, c_off) = buffer_from_tensor(&combined)?;
        let (y_buf, y_off) = buffer_from_tensor(&y)?;

        let encoder = metal_dev
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("silu_mul: encoder: {e}")))?;
        encoder.set_label("lumen:silu_mul_bf16in_bf16out");
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(c_buf), c_off as usize);
        encoder.set_buffer(1, Some(y_buf), y_off as usize);
        let dims_struct = SiluMulDims { inter: inter as u32 };
        encoder.set_bytes_directly(
            2,
            std::mem::size_of::<SiluMulDims>(),
            &dims_struct as *const _ as *const _,
        );
        // 2D dispatch: x = inter cols, y = m rows. Threads-per-tg pinned to
        // 64 lanes for warp-aligned coalesced writes (one cache line per
        // simdgroup at bf16 = 32 elements; 64 over-covers safely).
        const THREADS_X: usize = 64;
        let grid = MTLSize {
            width: inter.div_ceil(THREADS_X) * THREADS_X,
            height: m,
            depth: 1,
        };
        let tg = MTLSize { width: THREADS_X, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(
            MTLSize { width: grid.width / THREADS_X, height: grid.height, depth: 1 },
            tg,
        );
        drop(encoder);
        Ok(y)
    }

    /// record one silu*mul dispatch into `icb` at `slot`.
    pub fn record_icb(
        &self,
        icb: &IndirectCommandBuffer,
        slot: usize,
        combined_buf: &Buffer,
        combined_off: u64,
        y_buf: &Buffer,
        y_off: u64,
        dims_buf: &Buffer,
        m: usize,
        inter: usize,
    ) {
        const THREADS_X: usize = 64;
        let grid = MTLSize {
            width: inter.div_ceil(THREADS_X),
            height: m,
            depth: 1,
        };
        let tg = MTLSize { width: THREADS_X, height: 1, depth: 1 };
        icb.record_compute(
            slot,
            &self.pipeline_icb,
            &[
                (combined_buf, combined_off as usize, 0),
                (y_buf, y_off as usize, 1),
                (dims_buf, 0, 2),
            ],
            grid,
            tg,
        );
    }
}

#[cfg(feature = "model-integration")]
fn buffer_from_tensor(t: &Tensor) -> candle_core::Result<(&Buffer, u64)> {
    let (storage, layout) = t.storage_and_layout();
    match &*storage {
        Storage::Metal(ms) => {
            let off = (layout.start_offset() * t.dtype().size_in_bytes()) as u64;
            let buf_ptr = ms.buffer() as *const _ as *const Buffer;
            let buf_ref: &Buffer = unsafe { &*buf_ptr };
            Ok((buf_ref, off))
        }
        _ => Err(candle_core::Error::Msg(
            "SiluMulBf16InBf16Out: tensor not on Metal device".into(),
        )),
    }
}
