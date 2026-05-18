//! Candle ↔ MPSGraph RmsNorm wrapper.
//!
//! Bridges a Candle [`Tensor`] (with Metal storage) into the MPSGraph
//! pipeline for fused RmsNorm execution. The compiled graph is cached
//! per `(m, hidden)` so repeated calls avoid the JIT compile cost.

#![cfg(feature = "model-integration")]

use std::collections::HashMap;
use std::sync::Mutex;

use candle_core::{DType, Device, Storage, Tensor};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use objc2_metal_performance_shaders::MPSDataType;
use objc2_metal_performance_shaders_graph::MPSGraphExecutable;

use super::{
    compile, encode_and_commit, shape_from_dims, tensor_data_from_buffer, MpsGraphContext,
    RmsNormBf16OutGraph, RmsNormGraph,
};

/// Cached compiled RmsNorm graph for a given `(m, hidden)` pair.
struct CompiledRmsNorm {
    exe: Retained<MPSGraphExecutable>,
}

/// Per-process MPSGraph RmsNorm runtime.
///
/// Holds a single `MpsGraphContext` and a JIT-cache of compiled executables
/// keyed by tensor shape. Thread-safe via a `Mutex` around the cache.
///
/// encodes onto Candle's own `MTLCommandQueue`, so MPSGraph
/// work is serialized in-order with Candle's prior kernels. No
/// `device.synchronize()` needed — cross-queue race avoided by single-queue
/// ordering. `commit()` only; `wait_until_completed()` is left to Candle's
/// downstream ops (which themselves issue their own commit/wait).
pub struct MpsRmsNorm {
    ctx: MpsGraphContext,
    cache: Mutex<HashMap<(usize, usize), CompiledRmsNorm>>,
    eps: f32,
}

// SAFETY: Apple's Metal / MPSGraph ObjC objects use atomic retain/release and
// are documented as thread-safe at runtime. The `Mutex` on `cache` guards the
// only mutable shared state. `objc2` does not derive Send/Sync for `Retained`
// to stay conservative, but for the MTL stack on darwin this is sound.
unsafe impl Send for MpsRmsNorm {}
unsafe impl Sync for MpsRmsNorm {}

impl MpsRmsNorm {
    pub fn new(eps: f32) -> Result<Self, super::MpsGraphError> {
        let ctx = MpsGraphContext::new()?;
        Ok(Self {
            ctx,
            cache: Mutex::new(HashMap::new()),
            eps,
        })
    }

    /// Compute `y = x · rsqrt(mean(x², axis=-1) + eps) · weight` via MPSGraph.
    ///
    /// Inputs must be on the Metal device (any non-Metal storage is rejected).
    /// Shapes:
    /// - `x` is `[..., hidden]` (rank ≥ 2); leading dims flatten into `m`.
    /// - `weight` is `[hidden]`.
    /// Output mirrors `x`'s shape and dtype must be `f32`.
    pub fn forward(&self, x: &Tensor, weight: &Tensor) -> candle_core::Result<Tensor> {
        let x = x.contiguous()?;
        let weight = weight.contiguous()?;

        if x.dtype() != DType::F32 || weight.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(
                "MpsRmsNorm: only f32 supported in PoC".into(),
            ));
        }

        let dims = x.dims();
        if dims.len() < 2 {
            return Err(candle_core::Error::Msg("MpsRmsNorm: rank < 2".into()));
        }
        let hidden = *dims.last().unwrap();
        let m: usize = dims[..dims.len() - 1].iter().product();

        if weight.dims() != &[hidden] {
            return Err(candle_core::Error::Msg(format!(
                "MpsRmsNorm: weight shape {:?} mismatches hidden {hidden}",
                weight.dims()
            )));
        }

        // Compile (or fetch cached) executable for this (m, hidden) shape.
        let key = (m, hidden);
        {
            let mut cache = self.cache.lock().unwrap();
            if !cache.contains_key(&key) {
                let rn = RmsNormGraph::build(self.ctx.new_graph(), m, hidden, self.eps);
                let exe = compile(&rn.graph, self.ctx.device(), &rn.feeds, &[&*rn.y]);
                cache.insert(key, CompiledRmsNorm { exe });
            }
        }

        // Allocate output Tensor with `from_vec` (data baked in at alloc) so
        // no `fill_buffer(0)` command is queued — that pending blit on the
        // same Candle queue would otherwise commit *after* our encode and
        // overwrite the RmsNorm result. Verified in the diagnostic memo.
        let n: usize = x.dims().iter().product();
        let y = Tensor::from_vec(vec![0.0_f32; n], x.shape(), x.device())?.contiguous()?;

        // Borrow Candle's own command queue so our encode is serialized
        // in-order with Candle's pending kernels. No cross-queue race; no
        // pre-flush sync.
        let queue = match x.device() {
            Device::Metal(metal_dev) => metal_dev.command_queue()?,
            _ => {
                return Err(candle_core::Error::Msg(
                    "MpsRmsNorm: tensor not on Metal device".into(),
                ));
            }
        };

        // Bridge buffers.
        let (x_buf, x_off) = buffer_from_tensor(&x)?;
        let (w_buf, w_off) = buffer_from_tensor(&weight)?;
        let (y_buf, y_off) = buffer_from_tensor(&y)?;

        // PoC requirement: contiguous fresh allocations land at offset 0.
        // (Future: support nonzero via sub-buffer or copy.)
        if x_off != 0 || w_off != 0 || y_off != 0 {
            return Err(candle_core::Error::Msg(format!(
                "MpsRmsNorm PoC: nonzero buffer offsets unsupported (x={x_off} w={w_off} y={y_off})"
            )));
        }

        let x_shape = shape_from_dims(&[m, hidden]);
        let w_shape = shape_from_dims(&[hidden]);
        let x_td = tensor_data_from_buffer(x_buf, &x_shape, MPSDataType::Float32);
        let w_td = tensor_data_from_buffer(w_buf, &w_shape, MPSDataType::Float32);
        let y_td = tensor_data_from_buffer(y_buf, &x_shape, MPSDataType::Float32);

        let cache = self.cache.lock().unwrap();
        let exe = &cache.get(&key).unwrap().exe;
        // commit-only, no wait. y was allocated with `from_vec` so no
        // pending zero-fill races on the same queue. Downstream Candle ops
        // share the queue and their own commit/wait covers ours via Apple's
        // in-queue ordering guarantee (committed buffers execute in the
        // order they were committed).
        let _cb = encode_and_commit(exe, &queue, &[&*x_td, &*w_td], &[&*y_td]);

        // y's storage now holds the result on the GPU (after queue dispatch).
        Ok(y.reshape(x.shape())?)
    }
}

/// Per-process MPSGraph RmsNorm runtime that produces **bf16** output.
///
/// Lever B Stage 2 staging vehicle. Same f32 reduction as [`MpsRmsNorm`]; the
/// only difference is the final cast f32 → bf16 inside the fused graph plus
/// the narrower output store. Cache key is `(m, hidden)` and is independent
/// of the f32 cache.
pub struct MpsRmsNormBf16Out {
    ctx: MpsGraphContext,
    cache: Mutex<HashMap<(usize, usize), CompiledRmsNorm>>,
    eps: f32,
}

unsafe impl Send for MpsRmsNormBf16Out {}
unsafe impl Sync for MpsRmsNormBf16Out {}

impl MpsRmsNormBf16Out {
    pub fn new(eps: f32) -> Result<Self, super::MpsGraphError> {
        let ctx = MpsGraphContext::new()?;
        Ok(Self {
            ctx,
            cache: Mutex::new(HashMap::new()),
            eps,
        })
    }

    /// Compute `y_bf16 = bf16(x · rsqrt(mean(x², axis=-1) + eps) · weight)`.
    ///
    /// Inputs (`x`, `weight`) must be `DType::F32` on the Metal device.
    /// Output mirrors `x`'s shape with `DType::BF16`.
    pub fn forward(&self, x: &Tensor, weight: &Tensor) -> candle_core::Result<Tensor> {
        let x = x.contiguous()?;
        let weight = weight.contiguous()?;

        if x.dtype() != DType::F32 || weight.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(
                "MpsRmsNormBf16Out: inputs must be f32 (only the output is bf16)".into(),
            ));
        }

        let dims = x.dims();
        if dims.len() < 2 {
            return Err(candle_core::Error::Msg(
                "MpsRmsNormBf16Out: rank < 2".into(),
            ));
        }
        let hidden = *dims.last().unwrap();
        let m: usize = dims[..dims.len() - 1].iter().product();

        if weight.dims() != &[hidden] {
            return Err(candle_core::Error::Msg(format!(
                "MpsRmsNormBf16Out: weight shape {:?} mismatches hidden {hidden}",
                weight.dims()
            )));
        }

        let key = (m, hidden);
        {
            let mut cache = self.cache.lock().unwrap();
            if !cache.contains_key(&key) {
                let rn =
                    RmsNormBf16OutGraph::build(self.ctx.new_graph(), m, hidden, self.eps);
                let exe = compile(&rn.graph, self.ctx.device(), &rn.feeds, &[&*rn.y]);
                cache.insert(key, CompiledRmsNorm { exe });
            }
        }

        // Bake bf16 zeros at allocation (same pattern as MpsRmsNorm) to avoid
        // the Candle pool's `fill_buffer(0)` blit ordering after our encode.
        let n: usize = x.dims().iter().product();
        let zeros: Vec<half::bf16> = vec![half::bf16::ZERO; n];
        let y = Tensor::from_vec(zeros, x.shape(), x.device())?.contiguous()?;

        let queue = match x.device() {
            Device::Metal(metal_dev) => metal_dev.command_queue()?,
            _ => {
                return Err(candle_core::Error::Msg(
                    "MpsRmsNormBf16Out: tensor not on Metal device".into(),
                ));
            }
        };

        let (x_buf, x_off) = buffer_from_tensor(&x)?;
        let (w_buf, w_off) = buffer_from_tensor(&weight)?;
        let (y_buf, y_off) = buffer_from_tensor(&y)?;

        if x_off != 0 || w_off != 0 || y_off != 0 {
            return Err(candle_core::Error::Msg(format!(
                "MpsRmsNormBf16Out PoC: nonzero buffer offsets unsupported (x={x_off} w={w_off} y={y_off})"
            )));
        }

        let x_shape = shape_from_dims(&[m, hidden]);
        let w_shape = shape_from_dims(&[hidden]);
        let x_td = tensor_data_from_buffer(x_buf, &x_shape, MPSDataType::Float32);
        let w_td = tensor_data_from_buffer(w_buf, &w_shape, MPSDataType::Float32);
        let y_td = tensor_data_from_buffer(y_buf, &x_shape, MPSDataType::BFloat16);

        let cache = self.cache.lock().unwrap();
        let exe = &cache.get(&key).unwrap().exe;
        let _cb = encode_and_commit(exe, &queue, &[&*x_td, &*w_td], &[&*y_td]);

        Ok(y.reshape(x.shape())?)
    }
}

/// Extract the MTLBuffer + byte offset from a Candle Tensor with Metal storage.
fn buffer_from_tensor(
    t: &Tensor,
) -> candle_core::Result<(&ProtocolObject<dyn MTLBuffer>, usize)> {
    let (storage_guard, layout) = t.storage_and_layout();
    match &*storage_guard {
        Storage::Metal(ms) => {
            let offset_bytes = layout.start_offset() * t.dtype().size_in_bytes();
            let candle_buf: &candle_metal_kernels::metal::Buffer = ms.buffer();
            // SAFETY: AsRef<ProtocolObject<dyn MTLBuffer>> is implemented by candle.
            let buf_ref: &ProtocolObject<dyn MTLBuffer> = candle_buf.as_ref();
            // Borrow lives as long as the storage_guard / Tensor.
            // The signature exposes a borrow tied to `t`, which is correct.
            let buf_ref: &ProtocolObject<dyn MTLBuffer> =
                unsafe { std::mem::transmute(buf_ref) };
            Ok((buf_ref, offset_bytes))
        }
        _ => Err(candle_core::Error::Msg(
            "MpsRmsNorm: tensor is not on Metal device".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    /// Diagnostic — does `Tensor::zeros` alias x's buffer (pool reuse)?
    /// Logs the raw `MTLBuffer*` pointer for both tensors; if equal, the
    /// "bug" was buffer aliasing, not a cross-queue race.
    #[test]
    fn diagnose_buffer_aliasing() {
        let device = Device::new_metal(0).expect("Metal device");
        let m = 2usize;
        let hidden = 4usize;
        let x =
            Tensor::from_vec((0..(m * hidden) as i32).map(|i| i as f32).collect::<Vec<_>>(),
                (m, hidden), &device).unwrap();
        let y = Tensor::zeros((m, hidden), candle_core::DType::F32, &device).unwrap();

        let (x_buf, _) = buffer_from_tensor(&x).unwrap();
        let (y_buf, _) = buffer_from_tensor(&y).unwrap();
        let x_ptr = x_buf as *const _ as usize;
        let y_ptr = y_buf as *const _ as usize;
        eprintln!("x_buf ptr = 0x{x_ptr:x}");
        eprintln!("y_buf ptr = 0x{y_ptr:x}");
        eprintln!("aliased = {}", x_ptr == y_ptr);
        // Assert non-alias for now; if this fails the diagnosis pivots.
        assert_ne!(
            x_ptr, y_ptr,
            "x and y resolved to the same MTLBuffer — pool aliasing"
        );
    }

    /// PoC 1c — Candle Tensor wrapper matches CPU reference within tolerance.
    #[test]
    fn mps_rms_norm_candle_tensor_path() {
        let device = Device::new_metal(0).expect("Metal device");
        let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
        let w_data: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0];
        let m = 2usize;
        let hidden = 4usize;
        let eps = 1e-6_f32;

        let x = Tensor::from_vec(x_data.clone(), (m, hidden), &device).unwrap();
        let weight = Tensor::from_vec(w_data.clone(), hidden, &device).unwrap();

        let runtime = MpsRmsNorm::new(eps).expect("MpsRmsNorm init");
        let y = runtime.forward(&x, &weight).expect("forward");

        let got = y.to_vec2::<f32>().unwrap();

        // CPU reference.
        let mut expected = vec![vec![0.0_f32; hidden]; m];
        for row in 0..m {
            let off = row * hidden;
            let mean_sq: f32 =
                x_data[off..off + hidden].iter().map(|v| v * v).sum::<f32>() / hidden as f32;
            let inv = (mean_sq + eps).sqrt().recip();
            for c in 0..hidden {
                expected[row][c] = x_data[off + c] * inv * w_data[c];
            }
        }

        for (r, row) in got.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                let d = (v - expected[r][c]).abs();
                assert!(
                    d < 1e-4,
                    "mismatch at ({r},{c}): got {v}, expected {}",
                    expected[r][c]
                );
            }
        }

        // Sanity that the test isn't comparing zero vs zero.
        assert!(got.iter().flatten().any(|&v| v.abs() > 0.1));
    }
}
