//! Native forward path entry points.
//!
//! ships a single placeholder that round-trips an input through
//! `NativeTensor` so we can validate routing and the bridge helper without
//! yet having any real kernels. Subsequent phases (A.2…A.7) replace this
//! with actual decoder layers.

use anyhow::Result;
use candle_core::Tensor;

use super::bridge::{from_candle_tensor, to_candle_tensor};
use super::context::NativeContext;

/// Identity forward: Candle Tensor → NativeTensor → Candle Tensor.
///
/// Used by Phase A.0 wiring to prove the bridge end-to-end. Returns a tensor
/// equal to `input` (modulo possible dtype roundtrip through F32 staging).
pub fn placeholder_forward(ctx: &NativeContext, input: &Tensor) -> Result<Tensor> {
    let nt = from_candle_tensor(ctx, input)?;
    to_candle_tensor(&nt, input.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn placeholder_identity_cpu_f32() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let device = Device::Cpu;
        let data: Vec<f32> = (0..8).map(|i| (i as f32) * 0.125).collect();
        let t = Tensor::from_vec(data.clone(), (2, 4), &device).unwrap();
        let out = placeholder_forward(&ctx, &t).unwrap();
        assert_eq!(out.dims(), &[2, 4]);
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v, data);
    }

    #[test]
    fn placeholder_identity_cpu_u32() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let device = Device::Cpu;
        let data: Vec<u32> = vec![5, 11, 17, 23];
        let t = Tensor::from_vec(data.clone(), (4,), &device).unwrap();
        let out = placeholder_forward(&ctx, &t).unwrap();
        let v = out.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        assert_eq!(v, data);
    }
}
