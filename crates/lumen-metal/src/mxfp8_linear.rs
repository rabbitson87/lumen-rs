//! Candle `Linear`-shaped wrapper around [`Mxfp8Weight`].
//!
//! Sister to [`crate::affine8_linear`] for MLX MXFP8 (OCP) checkpoints
//! (e.g. `mlx-community/Qwen3-Embedding-4B-mxfp8`). Holds the weight
//! resident on the GPU as packed E4M3 bytes (4 per uint32) + per-group
//! E8M0 scales, and exposes `forward(&Tensor) -> Tensor` taking bf16
//! input → bf16 output.
//!
//! The wrapper dispatches automatically between the cooperative
//! `mxfp8_qmv_fast_bf16` simdgroup kernel (when
//! `in % 512 == 0 && out % 8 == 0`) and the naive
//! `mxfp8_matmul_bf16` fallback. Both hold across every projection in
//! Qwen3-Embedding-4B-mxfp8 (in ∈ {1024, 2560, 9728}, out ∈ {1024, 2560,
//! 4096, 9728, vocab}).

use std::sync::Arc;

use anyhow::Result;
use candle_core::{DType, Device, Storage, Tensor};

use crate::metal::Buffer;
use crate::mxfp8_gpu::{Mxfp8Context, Mxfp8Weight};

/// Extract the Metal buffer + byte offset backing a Candle tensor.
/// Mirrors [`crate::affine8_linear::metal_buffer_of`].
fn metal_buffer_of(t: &Tensor) -> Option<(&Buffer, u64)> {
    let (storage_guard, layout) = t.storage_and_layout();
    match &*storage_guard {
        Storage::Metal(ms) => {
            let offset_bytes = (layout.start_offset() * t.dtype().size_in_bytes()) as u64;
            let candle_buf = ms.buffer();
            let buf_ptr = candle_buf as *const _ as *const Buffer;
            let buf_ref: &Buffer = unsafe { &*buf_ptr };
            Some((buf_ref, offset_bytes))
        }
        _ => None,
    }
}

/// `y = x @ W^T` where W is an MLX-format MXFP8 quantized weight.
///
/// Input/output dtype: `bf16`. Input shape: `(..., in_features)`.
/// Output shape: `(..., out_features)`. Leading dims are flattened into a
/// batch dimension for the kernel and unflattened on return.
pub fn mxfp8_matmul_bf16_tensor(
    ctx: &Mxfp8Context,
    weight: &Mxfp8Weight,
    x: &Tensor,
) -> candle_core::Result<Tensor> {
    let dims = x.dims();
    if dims.is_empty() {
        return Err(candle_core::Error::Msg(
            "mxfp8 matmul requires a non-scalar input".into(),
        ));
    }
    let last = dims[dims.len() - 1];
    if last != weight.in_features {
        return Err(candle_core::Error::Msg(format!(
            "mxfp8 input last dim {} != in_features {}",
            last, weight.in_features
        )));
    }
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
    out_shape.push(weight.out_features);

    if batch == 0 {
        return Tensor::zeros(out_shape, DType::BF16, x.device());
    }
    if x.dtype() != DType::BF16 {
        return Err(candle_core::Error::Msg(format!(
            "mxfp8 requires bf16 input, got {:?}",
            x.dtype()
        )));
    }
    if !x.device().is_metal() {
        return Err(candle_core::Error::Msg(
            "mxfp8 requires Metal-resident input".into(),
        ));
    }

    let x_c = x.contiguous()?;
    let y = Tensor::zeros(out_shape, DType::BF16, x.device())?;

    let (x_buf, x_offset) =
        metal_buffer_of(&x_c).ok_or_else(|| candle_core::Error::Msg("x not Metal".into()))?;
    let (y_buf, y_offset) =
        metal_buffer_of(&y).ok_or_else(|| candle_core::Error::Msg("y not Metal".into()))?;

    if let Device::Metal(metal_dev) = x.device() {
        let encoder = metal_dev
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("candle command_encoder: {e}")))?;
        encoder.set_label("lumen:mxfp8_matmul_bf16");
        ctx.encode_matmul_bf16_dispatch(
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
    Err(candle_core::Error::Msg("unsupported device".into()))
}

/// Linear layer backed by an MLX MXFP8 quantized weight. Drop-in shape
/// replacement for `candle_nn::Linear` when the weight comes from a
/// `mlx.quantize(bits=8, group_size=32, mode="mxfp8")` checkpoint.
pub struct Mxfp8Linear {
    weight: Mxfp8Weight,
    ctx: Arc<Mxfp8Context>,
}

impl Mxfp8Linear {
    pub fn new(weight: Mxfp8Weight, ctx: Arc<Mxfp8Context>) -> Self {
        Self { weight, ctx }
    }

    pub fn in_features(&self) -> usize {
        self.weight.in_features
    }

    pub fn out_features(&self) -> usize {
        self.weight.out_features
    }

    pub fn device_bytes(&self) -> usize {
        self.weight.approx_bytes()
    }

    /// bf16 → bf16. Mirrors `candle_nn::Linear::forward` shape semantics.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(mxfp8_matmul_bf16_tensor(&self.ctx, &self.weight, x)?)
    }
}
