//! `NativeContext`: single Metal command queue + buffer factory for the native
//! forward path.
//!
//! Why not reuse `MxFp4Context` from `lumen-metal`? That context already
//! owns a queue and a set of pipelines bound to MXFP4 matmul. The native path
//! will register additional kernels (RMSNorm, RoPE, soft cap, sampling, etc.)
//! and wants its own pipeline cache. Both contexts can target the *same* device
//! and share buffer ownership across queues — Apple Silicon Metal allows
//! cross-queue resource access with explicit `wait_until_completed` or
//! shared events.
//!
//! bare-bones context that allocates buffers and runs
//! a no-op command. Subsequent phases register pipelines (mxfp4 v2, rmsnorm,
//! etc.) and add fence helpers.

use anyhow::{anyhow, Result};
use lumen_metal::metal::{Buffer, CommandQueue, Device, MTLResourceOptions};

use super::tensor::{NativeDType, NativeTensor};

pub struct NativeContext {
    pub device: Device,
    pub queue: CommandQueue,
}

impl NativeContext {
    pub fn new() -> Result<Self> {
        let device = Device::system_default()
            .ok_or_else(|| anyhow!("NativeContext: no Metal-capable device"))?;
        let queue = device
            .new_command_queue()
            .map_err(|e| anyhow!("new_command_queue: {e}"))?;
        Ok(Self { device, queue })
    }

    /// Construct a context that shares a `Device` with another component
    /// (e.g. `MxFp4Context`). Only the queue is new; buffers allocated here
    /// remain interoperable with the shared device's allocations.
    pub fn from_device(device: Device) -> Self {
        let queue = device
            .new_command_queue()
            .expect("new_command_queue failed");
        Self { device, queue }
    }

    /// Construct from a Candle Metal device handle. This is the path the production
    /// forward uses: it shares the **same** underlying `MTLDevice` as Candle so
    /// zero-copy buffer adoption (`from_candle_tensor`) hands us a buffer the device
    /// recognizes. Returns `None` when `candle_device` is not Metal.
    ///
    /// Cross-version transmute: Candle's `metal::Device` and ours are ABI-identical
    /// (same `objc::runtime::Object` retain/release). Same trick `mxfp4_linear` uses
    /// for buffers, applied to the device handle here.
    pub fn from_candle_device(candle_device: &candle_core::Device) -> Result<Self> {
        let candle_md = match candle_device {
            candle_core::Device::Metal(md) => md,
            _ => return Err(anyhow!("from_candle_device: not a Metal device")),
        };
        // Candle's `metal_device()` returns `&candle_metal::Device`. Transmute pointer
        // to our `metal::Device` (ABI identical), then `.clone()` to bump objc refcount.
        let candle_dev_ref: &lumen_metal::metal::Device = unsafe {
            &*(candle_md.metal_device() as *const _ as *const lumen_metal::metal::Device)
        };
        let device: Device = candle_dev_ref.clone();
        let queue = device
            .new_command_queue()
            .map_err(|e| anyhow!("new_command_queue: {e}"))?;
        Ok(Self { device, queue })
    }

    /// Allocate a zero-filled `NativeTensor` of the requested shape/dtype.
    pub fn zeros(&self, shape: Vec<usize>, dtype: NativeDType) -> Result<NativeTensor> {
        let numel: usize = shape.iter().product();
        let bytes = numel * dtype.size_in_bytes();
        let buffer = self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared)
            .map_err(|e| anyhow!("new_buffer: {e}"))?;
        // StorageModeShared zeros are not guaranteed by Metal; explicitly clear
        // via host pointer (single linear write, microsecond scale even at
        // hidden=2048 × seq=4096 = 32 MB).
        unsafe {
            std::ptr::write_bytes(buffer.contents() as *mut u8, 0, bytes);
        }
        NativeTensor::from_buffer(buffer, 0, shape, dtype)
    }

    /// Allocate an uninitialized `NativeTensor`. Use when the caller is about
    /// to write the entire buffer (skips the zero-fill).
    pub fn uninit(&self, shape: Vec<usize>, dtype: NativeDType) -> Result<NativeTensor> {
        let numel: usize = shape.iter().product();
        let bytes = numel * dtype.size_in_bytes();
        let buffer = self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared)
            .map_err(|e| anyhow!("new_buffer: {e}"))?;
        NativeTensor::from_buffer(buffer, 0, shape, dtype)
    }

    /// Upload F32 host data to a fresh `NativeTensor`.
    pub fn from_slice_f32(&self, data: &[f32], shape: Vec<usize>) -> Result<NativeTensor> {
        let numel: usize = shape.iter().product();
        if numel != data.len() {
            return Err(anyhow!(
                "from_slice_f32 length mismatch: shape numel {} vs data {}",
                numel,
                data.len()
            ));
        }
        let bytes = data.len() * 4;
        let buffer = self
            .device
            .new_buffer_with_data(
                data.as_ptr() as *const _,
                bytes,
                MTLResourceOptions::StorageModeShared,
            )
            .map_err(|e| anyhow!("new_buffer_with_data: {e}"))?;
        NativeTensor::from_buffer(buffer, 0, shape, NativeDType::F32)
    }

    /// Adopt an externally-allocated buffer (e.g. one already loaded by Candle's
    /// safetensors path) into a `NativeTensor`. The buffer is ref-counted via
    /// objc; cloning is cheap.
    pub fn adopt_buffer(
        &self,
        buffer: Buffer,
        offset_bytes: usize,
        shape: Vec<usize>,
        dtype: NativeDType,
    ) -> Result<NativeTensor> {
        NativeTensor::from_buffer(buffer, offset_bytes, shape, dtype)
    }

    /// Drain all pending work on this queue. Phase A.0 placeholder: identical
    /// to `cmd.wait_until_completed()` on a no-op buffer. Used as a coarse
    /// sync point until we build out fence-based barriers.
    pub fn drain(&self) {
        let cmd = lumen_metal::metal::new_command_buffer(&self.queue);
        cmd.commit();
        cmd.wait_until_completed();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_allocates_zeros() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let t = ctx.zeros(vec![4, 4], NativeDType::F32).unwrap();
        let v = t.to_vec_f32().unwrap();
        assert_eq!(v.len(), 16);
        for x in &v {
            assert_eq!(*x, 0.0);
        }
    }

    #[test]
    fn from_slice_roundtrip() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
        let t = ctx.from_slice_f32(&data, vec![3, 4]).unwrap();
        let v = t.to_vec_f32().unwrap();
        assert_eq!(v, data);
    }

    #[test]
    fn from_slice_length_mismatch_rejected() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let data = vec![0.0_f32; 10];
        assert!(ctx.from_slice_f32(&data, vec![3, 4]).is_err());
    }

    #[test]
    fn drain_no_op() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        ctx.drain();
    }
}
