//! Candle `Linear`-shaped wrapper around [`Affine8Weight`].
//!
//! Sister to [`crate::affine4_linear`] for MLX 8-bit. Holds the weight
//! resident on the GPU as packed `u32` + bf16 scales/biases and exposes
//! `forward(&Tensor) -> Tensor` taking bf16 input → bf16 output.
//!
//! single `forward_bf16_in_bf16_out` path using the
//! basic `affine8_matmul_bf16` kernel. v3/qmv_fast / RMSNorm-fusion /
//! gate+up fusion variants can land later once embedding inference is
//! validated end-to-end.

use std::sync::Arc;

use anyhow::Result;
use candle_core::{DType, Device, Storage, Tensor};

use crate::affine8_gpu::{Affine8Context, Affine8Weight};
use crate::metal::Buffer;

/// Extract the Metal buffer + byte offset backing a Candle tensor. Returns
/// `None` when the tensor is not Metal-resident. Same pattern as in
/// `affine4_linear::metal_buffer_of` / `mxfp4_linear::metal_buffer_of`.
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

/// `y = x @ W^T` where W is an MLX-format 8-bit quantized weight.
///
/// Input/output dtype: `bf16`. Input shape: `(..., in_features)`.
/// Output shape: `(..., out_features)`. The leading dims are flattened
/// into a batch dimension for the kernel and unflattened on return.
pub fn affine8_matmul_bf16_tensor(
    ctx: &Affine8Context,
    weight: &Affine8Weight,
    x: &Tensor,
) -> candle_core::Result<Tensor> {
    let dims = x.dims();
    if dims.is_empty() {
        return Err(candle_core::Error::Msg(
            "affine8 matmul requires a non-scalar input".into(),
        ));
    }
    let last = dims[dims.len() - 1];
    if last != weight.in_features {
        return Err(candle_core::Error::Msg(format!(
            "affine8 input last dim {} != in_features {}",
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
            "affine8 requires bf16 input, got {:?}",
            x.dtype()
        )));
    }
    if !x.device().is_metal() {
        return Err(candle_core::Error::Msg(
            "affine8 requires Metal-resident input".into(),
        ));
    }

    let x_c = x.contiguous()?;
    let y = Tensor::zeros(out_shape, DType::BF16, x.device())?;

    let (x_buf, x_offset) =
        metal_buffer_of(&x_c).ok_or_else(|| candle_core::Error::Msg("x not Metal".into()))?;
    let (y_buf, y_offset) =
        metal_buffer_of(&y).ok_or_else(|| candle_core::Error::Msg("y not Metal".into()))?;

    // Submit our kernel onto candle's own command queue. This keeps all
    // ops in the same queue so Metal's per-buffer hazard tracking handles
    // ordering automatically — we don't need cross-queue sync. Validated
    // for batch > 1 by the `affine8_linear_candle_*` parity tests.
    if let Device::Metal(metal_dev) = x.device() {
        let encoder = metal_dev
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("candle command_encoder: {e}")))?;
        encoder.set_label("lumen:affine8_matmul_bf16");
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

/// Linear layer backed by an MLX 8-bit quantized weight. Drop-in shape
/// replacement for `candle_nn::Linear` when the weight comes from a
/// `mlx.quantize(bits=8, group_size=64)` checkpoint.
pub struct Affine8Linear {
    weight: Affine8Weight,
    ctx: Arc<Affine8Context>,
}

impl Affine8Linear {
    pub fn new(weight: Affine8Weight, ctx: Arc<Affine8Context>) -> Self {
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
        Ok(affine8_matmul_bf16_tensor(&self.ctx, &self.weight, x)?)
    }
}
