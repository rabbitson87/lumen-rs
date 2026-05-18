//! Conversion helpers between Candle `Tensor` and `NativeTensor`.
//!
//! incrementally migrates the forward path off Candle, but for the
//! handoff at module boundaries we need to round-trip activations cheaply.
//! When the input lives on Metal the conversion is zero-copy: we adopt the
//! same `metal::Buffer` (ref-count clone) and remember the start offset.
//!
//! When the input is on CPU we copy through a host-side staging buffer. This
//! is the slow path used only by tests; the production forward keeps every
//! activation on Metal.

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Storage, Tensor};

use lumen_metal::metal::Buffer;

use super::context::NativeContext;
use super::tensor::{NativeDType, NativeTensor};

fn candle_dtype_to_native(dtype: DType) -> Result<NativeDType> {
    match dtype {
        DType::F32 => Ok(NativeDType::F32),
        DType::BF16 => Ok(NativeDType::BF16),
        DType::U32 => Ok(NativeDType::U32),
        DType::U8 => Ok(NativeDType::U8),
        other => Err(anyhow!(
            "candle dtype {:?} has no NativeDType counterpart yet",
            other
        )),
    }
}

fn native_dtype_to_candle(dtype: NativeDType) -> DType {
    match dtype {
        NativeDType::F32 => DType::F32,
        NativeDType::BF16 => DType::BF16,
        NativeDType::U32 => DType::U32,
        NativeDType::U8 => DType::U8,
    }
}

/// Extract `(&metal::Buffer, offset_bytes)` from a Metal-resident Candle tensor
/// using the same ABI-transmute pattern as `mxfp4_linear::metal_buffer_of`.
/// Returns `None` if the tensor is not on Metal.
fn metal_buffer_of(t: &Tensor) -> Option<(Buffer, usize)> {
    let (storage_guard, layout) = t.storage_and_layout();
    match &*storage_guard {
        Storage::Metal(ms) => {
            let offset_bytes = layout.start_offset() * t.dtype().size_in_bytes();
            let candle_buf = ms.buffer();
            // Candle's `metal` crate version and ours are ABI-identical (objc
            // ref-counted handle to the same MTLBuffer). Transmute the pointer
            // to obtain a borrow against the same underlying Metal object,
            // then `.clone()` to get an owned Buffer with bumped refcount.
            let buf_ptr = candle_buf as *const _ as *const Buffer;
            let buf_ref: &Buffer = unsafe { &*buf_ptr };
            Some((buf_ref.clone(), offset_bytes))
        }
        _ => None,
    }
}

/// Adopt a Candle `Tensor` as a `NativeTensor` view. Zero-copy when `t` is
/// Metal-resident and contiguous; otherwise falls back to a CPU staging copy
/// onto `ctx`.
///
/// The returned `NativeTensor` shares storage with `t` when zero-copy succeeds,
/// so any downstream Metal kernel write will be observed by Candle if and only
/// if the caller has flushed Candle's command queue beforehand. Cross-queue
/// hazards apply — see the warnings in `mxfp4_linear`.
pub fn from_candle_tensor(ctx: &NativeContext, t: &Tensor) -> Result<NativeTensor> {
    from_candle_tensor_inner(ctx, t, /* sync */ true)
}

/// Same as [`from_candle_tensor`] but skips the Candle-queue `wait_until_completed`
/// synchronization. Use this when the caller has already synced (e.g. when
/// adopting multiple tensors from the same Candle queue lineage in a tight loop).
///
/// Caller MUST guarantee that all Candle work producing `t` has committed +
/// completed before this is invoked, otherwise the kernel will read uninitialized
/// memory.
pub fn from_candle_tensor_no_sync(ctx: &NativeContext, t: &Tensor) -> Result<NativeTensor> {
    from_candle_tensor_inner(ctx, t, /* sync */ false)
}

fn from_candle_tensor_inner(ctx: &NativeContext, t: &Tensor, sync: bool) -> Result<NativeTensor> {
    let dtype = candle_dtype_to_native(t.dtype())?;
    let shape = t.dims().to_vec();

    // Zero-copy fast path is now **default ON** post ABI-M migration (2026-04-26):
    // both Candle and our crate link the same `objc2-metal v0.3.2` Buffer wrapper via
    // `candle-metal-kernels`, so the buffer transmute lands on a structurally-identical
    // wrapper around the same `MTLBuffer` object. Set `LUMEN_BRIDGE_ZEROCOPY=0` to
    // force the host-staging fallback (debugging only).
    //
    // Cross-queue hazard: Candle and our NativeContext hold independent command queues.
    // Anything pending on Candle's queue isn't visible until Candle commits + waits.
    // mxfp4_linear handles this with `MetalDevice::wait_until_completed()` before
    // adopting buffers; we mirror that pattern here. Set `LUMEN_BRIDGE_SKIP_SYNC=1`
    // to bypass globally (only when the caller already synchronized).
    let zero_copy_enabled = std::env::var("LUMEN_BRIDGE_ZEROCOPY")
        .map(|v| v != "0")
        .unwrap_or(true);
    if zero_copy_enabled && t.device().is_metal() && t.is_contiguous() {
        let skip_sync_env = std::env::var("LUMEN_BRIDGE_SKIP_SYNC")
            .map(|v| v == "1")
            .unwrap_or(false);
        if sync && !skip_sync_env {
            if let Device::Metal(md) = t.device() {
                md.wait_until_completed()
                    .map_err(|e| anyhow!("bridge: candle queue sync failed: {e}"))?;
            }
        }
        if let Some((buf, offset)) = metal_buffer_of(t) {
            // Anchor the source Candle tensor for the lifetime of this NativeTensor —
            // see the doc comment on `NativeTensor::_anchor`.
            return NativeTensor::from_buffer_with_anchor(
                buf,
                offset,
                shape,
                dtype,
                Some(t.clone()),
            );
        }
    }

    // Fallback: copy through host. Phase A.0 uses this for tests where Candle
    // tensors live on CPU.
    match dtype {
        NativeDType::F32 => {
            let v = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            ctx.from_slice_f32(&v, shape)
        }
        NativeDType::U32 => {
            let v = t.to_dtype(DType::U32)?.flatten_all()?.to_vec1::<u32>()?;
            // No specialised U32 constructor yet — go through a fresh buffer.
            let bytes = v.len() * 4;
            let buffer = ctx
                .device
                .new_buffer_with_data(
                    v.as_ptr() as *const _,
                    bytes,
                    lumen_metal::metal::MTLResourceOptions::StorageModeShared,
                )
                .map_err(|e| anyhow!("new_buffer_with_data: {e}"))?;
            NativeTensor::from_buffer(buffer, 0, shape, NativeDType::U32)
        }
        NativeDType::BF16 | NativeDType::U8 => Err(anyhow!(
            "from_candle_tensor: CPU fallback for {:?} not implemented yet",
            dtype
        )),
    }
}

/// Materialize a `NativeTensor` as a Candle `Tensor` on the requested device.
/// Currently always copies through the host because Candle's metal storage
/// constructor is private; a true zero-copy path needs upstream changes or a
/// shared device handle.
pub fn to_candle_tensor(t: &NativeTensor, device: &Device) -> Result<Tensor> {
    let dtype = native_dtype_to_candle(t.dtype());
    match t.dtype() {
        NativeDType::F32 => {
            let v = t.to_vec_f32()?;
            Tensor::from_vec(v, t.shape().to_vec(), device)
                .and_then(|x| x.to_dtype(dtype))
                .map_err(|e| anyhow!("{e}"))
        }
        NativeDType::U32 => {
            let v = t.to_vec_u32()?;
            Tensor::from_vec(v, t.shape().to_vec(), device).map_err(|e| anyhow!("{e}"))
        }
        NativeDType::BF16 | NativeDType::U8 => Err(anyhow!(
            "to_candle_tensor: {:?} not implemented yet",
            t.dtype()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn cpu_f32_roundtrip() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let device = Device::Cpu;
        let data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.25).collect();
        let t = Tensor::from_vec(data.clone(), (3, 4), &device).unwrap();
        let nt = from_candle_tensor(&ctx, &t).unwrap();
        assert_eq!(nt.shape(), &[3, 4]);
        assert_eq!(nt.to_vec_f32().unwrap(), data);

        let back = to_candle_tensor(&nt, &device).unwrap();
        assert_eq!(back.dims(), &[3, 4]);
        let back_v = back.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(back_v, data);
    }

    #[test]
    fn cpu_u32_roundtrip() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let device = Device::Cpu;
        let data: Vec<u32> = (0..6).map(|i| 100 + i as u32).collect();
        let t = Tensor::from_vec(data.clone(), (2, 3), &device).unwrap();
        let nt = from_candle_tensor(&ctx, &t).unwrap();
        assert_eq!(nt.dtype(), NativeDType::U32);
        assert_eq!(nt.to_vec_u32().unwrap(), data);

        let back = to_candle_tensor(&nt, &device).unwrap();
        let back_v = back.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        assert_eq!(back_v, data);
    }

    #[test]
    fn metal_zero_copy_f32() {
        let device = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let ctx = NativeContext::new().unwrap();
        let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let t = Tensor::from_vec(data.clone(), (4, 4), &device).unwrap();
        let nt = from_candle_tensor(&ctx, &t).unwrap();
        assert_eq!(nt.shape(), &[4, 4]);
        assert_eq!(nt.to_vec_f32().unwrap(), data);
    }
}
