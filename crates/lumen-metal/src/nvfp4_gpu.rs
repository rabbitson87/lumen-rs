//! Metal dispatcher for NVFP4 fused dequant + matvec (Phase 2 first unit).
//!
//! Mirrors the MXFP4 matvec path ([`crate::mxfp4_gpu`]) for the NVFP4 storage layout
//! (group_size=16, E4M3 per-group scales). Shader: [`shaders/nvfp4.metal`], layout-compatible
//! with [`crate::nvfp4`]. Only the standalone one-shot `matvec_f32` path is wired so far; it is
//! parity-tested against the CPU reference [`crate::nvfp4::dequantize_f32`]. Fused matmul / MoE
//! variants (the bulk of `mxfp4_gpu.rs`) are pending — see the nvfp4 decision memory note.

use crate::device::MetalContext;
use crate::metal::{BatchedEncoderExt, ComputeEncoderCompat};
use crate::metal::{ComputePipelineState, MTLSize};
use anyhow::Result;

const SHADER_SRC: &str = include_str!("shaders/nvfp4.metal");

#[repr(C)]
#[derive(Clone, Copy)]
struct NvFp4Dims {
    out_features: u32,
    in_features: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NvFp4MoeDims {
    out_features: u32,
    in_features: u32,
    batch: u32,
    /// 1 if all slots share `x[0]`, 0 if each slot reads its own `[batch, in]` band.
    broadcast_x: u32,
}

/// Owns the NVFP4 shader library and pipeline states (matvec + batched matmul + MoE grouped).
pub struct Nvfp4Context {
    ctx: MetalContext,
    matvec_f32: ComputePipelineState,
    matmul_f32: ComputePipelineState,
    matmul_moe_f32: ComputePipelineState,
}

impl Nvfp4Context {
    pub fn new() -> Result<Self> {
        let ctx = MetalContext::new()?;
        let options = crate::metal::new_compile_options();
        options.set_language_version(crate::metal::MTLLanguageVersion::Version3_1);
        options.set_fast_math_enabled(true);
        let library = ctx
            .device
            .new_library_with_source(SHADER_SRC, Some(options.as_ref()))
            .map_err(|e| anyhow::anyhow!("NVFP4 shader compile error: {e}"))?;
        let pipeline = |name: &str| -> Result<ComputePipelineState> {
            let func = library
                .get_function(name, None)
                .map_err(|e| anyhow::anyhow!("kernel `{name}` not found: {e}"))?;
            ctx.device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| anyhow::anyhow!("pipeline `{name}` failed: {e}"))
        };
        let matvec_f32 = pipeline("nvfp4_matvec_f32")?;
        let matmul_f32 = pipeline("nvfp4_matmul_f32")?;
        let matmul_moe_f32 = pipeline("nvfp4_matmul_moe_f32")?;
        Ok(Self {
            ctx,
            matvec_f32,
            matmul_f32,
            matmul_moe_f32,
        })
    }

    /// One-shot fused dequant + matvec: `y[r] = sum_k dequant(W)[r, k] * x[k]`.
    /// Allocates buffers per call (parity / smoke path, not the hot loop).
    pub fn matvec_f32(
        &self,
        packed: &[u32],
        scales: &[u8],
        x: &[f32],
        out_features: usize,
        in_features: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(
            in_features.is_multiple_of(16),
            "in_features {in_features} must be a multiple of 16 (NVFP4 group size)"
        );
        let expected_packed = out_features * in_features / 8;
        anyhow::ensure!(
            packed.len() == expected_packed,
            "packed length {} != {expected_packed}",
            packed.len()
        );
        let expected_scales = out_features * in_features / 16;
        anyhow::ensure!(
            scales.len() == expected_scales,
            "scales length {} != {expected_scales}",
            scales.len()
        );
        anyhow::ensure!(
            x.len() == in_features,
            "x length {} != in_features {in_features}",
            x.len()
        );

        let packed_buf = self.ctx.buffer_with_data(packed);
        let scales_buf = self.ctx.buffer_with_data(scales);
        let x_buf = self.ctx.buffer_with_data(x);
        let y_buf = self.ctx.buffer_for::<f32>(out_features);

        let dims = NvFp4Dims {
            out_features: out_features as u32,
            in_features: in_features as u32,
        };
        let dims_buf = self.ctx.buffer_with_data(std::slice::from_ref(&dims));

        let encoder = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        encoder.set_label("lumen:nvfp4_matvec");
        encoder.set_compute_pipeline_state(&self.matvec_f32);
        encoder.set_buffer(0, Some(&packed_buf), 0usize);
        encoder.set_buffer(1, Some(&scales_buf), 0usize);
        encoder.set_buffer(2, Some(&x_buf), 0usize);
        encoder.set_buffer(3, Some(&y_buf), 0usize);
        encoder.set_buffer(4, Some(&dims_buf), 0usize);

        let max_threads = self.matvec_f32.max_total_threads_per_threadgroup();
        let threads_per_tg = max_threads.min(256);
        let grid = MTLSize {
            width: out_features,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: threads_per_tg as usize,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(grid, tg);
        drop(encoder);
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");

        Ok(self.ctx.read_buffer::<f32>(&y_buf, out_features))
    }

    /// One-shot batched matmul: `y[b, r] = sum_k dequant(W)[r, k] * x[b, k]`.
    /// `x` is `[batch, in_features]`, output `[batch, out_features]` (row-major).
    pub fn matmul_f32(
        &self,
        packed: &[u32],
        scales: &[u8],
        x: &[f32],
        out_features: usize,
        in_features: usize,
        batch: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(
            in_features.is_multiple_of(16),
            "in_features {in_features} must be a multiple of 16 (NVFP4 group size)"
        );
        anyhow::ensure!(
            packed.len() == out_features * in_features / 8,
            "packed length {} != {}",
            packed.len(),
            out_features * in_features / 8
        );
        anyhow::ensure!(
            scales.len() == out_features * in_features / 16,
            "scales length {} != {}",
            scales.len(),
            out_features * in_features / 16
        );
        anyhow::ensure!(
            x.len() == batch * in_features,
            "x length {} != batch*in {}",
            x.len(),
            batch * in_features
        );

        let packed_buf = self.ctx.buffer_with_data(packed);
        let scales_buf = self.ctx.buffer_with_data(scales);
        let x_buf = self.ctx.buffer_with_data(x);
        let y_buf = self.ctx.buffer_for::<f32>(batch * out_features);
        let dims = NvFp4Dims {
            out_features: out_features as u32,
            in_features: in_features as u32,
        };
        let dims_buf = self.ctx.buffer_with_data(std::slice::from_ref(&dims));
        let batch_buf = self
            .ctx
            .buffer_with_data(std::slice::from_ref(&(batch as u32)));

        let encoder = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        encoder.set_label("lumen:nvfp4_matmul");
        encoder.set_compute_pipeline_state(&self.matmul_f32);
        encoder.set_buffer(0, Some(&packed_buf), 0usize);
        encoder.set_buffer(1, Some(&scales_buf), 0usize);
        encoder.set_buffer(2, Some(&x_buf), 0usize);
        encoder.set_buffer(3, Some(&y_buf), 0usize);
        encoder.set_buffer(4, Some(&dims_buf), 0usize);
        encoder.set_buffer(5, Some(&batch_buf), 0usize);

        let threads_per_tg = self.matmul_f32.max_total_threads_per_threadgroup().min(256);
        let grid = MTLSize {
            width: out_features,
            height: batch,
            depth: 1,
        };
        let tg = MTLSize {
            width: threads_per_tg as usize,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(grid, tg);
        drop(encoder);
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");
        Ok(self.ctx.read_buffer::<f32>(&y_buf, batch * out_features))
    }

    /// One-shot MoE grouped matmul. `packed`/`scales` hold `num_experts` weight slabs;
    /// `expert_indices[slot]` selects the expert per z-slot. `x` is broadcast `[batch, in]`
    /// (broadcast_x=1). Output is `[k, batch, out_features]` row-major.
    pub fn matmul_moe_f32(
        &self,
        packed_all: &[u32],
        scales_all: &[u8],
        expert_indices: &[u32],
        x: &[f32],
        out_features: usize,
        in_features: usize,
        batch: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(
            in_features.is_multiple_of(16),
            "in_features {in_features} must be a multiple of 16 (NVFP4 group size)"
        );
        anyhow::ensure!(
            x.len() == batch * in_features,
            "x length {} != batch*in {}",
            x.len(),
            batch * in_features
        );
        let k = expert_indices.len();

        let packed_buf = self.ctx.buffer_with_data(packed_all);
        let scales_buf = self.ctx.buffer_with_data(scales_all);
        let idx_buf = self.ctx.buffer_with_data(expert_indices);
        let x_buf = self.ctx.buffer_with_data(x);
        let y_buf = self.ctx.buffer_for::<f32>(k * batch * out_features);
        let dims = NvFp4MoeDims {
            out_features: out_features as u32,
            in_features: in_features as u32,
            batch: batch as u32,
            broadcast_x: 1,
        };
        let dims_buf = self.ctx.buffer_with_data(std::slice::from_ref(&dims));

        let encoder = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        encoder.set_label("lumen:nvfp4_matmul_moe");
        encoder.set_compute_pipeline_state(&self.matmul_moe_f32);
        encoder.set_buffer(0, Some(&packed_buf), 0usize);
        encoder.set_buffer(1, Some(&scales_buf), 0usize);
        encoder.set_buffer(2, Some(&idx_buf), 0usize);
        encoder.set_buffer(3, Some(&x_buf), 0usize);
        encoder.set_buffer(4, Some(&y_buf), 0usize);
        encoder.set_buffer(5, Some(&dims_buf), 0usize);

        let threads_per_tg = self
            .matmul_moe_f32
            .max_total_threads_per_threadgroup()
            .min(256);
        let grid = MTLSize {
            width: out_features,
            height: batch,
            depth: k,
        };
        let tg = MTLSize {
            width: threads_per_tg as usize,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(grid, tg);
        drop(encoder);
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");
        Ok(self
            .ctx
            .read_buffer::<f32>(&y_buf, k * batch * out_features))
    }

    /// Benchmark the matvec kernel with **device-resident weights** (upload `packed`/`scales`/`x`
    /// once, then time only encode+dispatch+sync across `iters` runs after `warmup`). This is the
    /// fair counterpart to mlx FFI `quantized_matmul`, whose quantized weight Array is also
    /// device-resident after the first eval — so neither side pays per-call host upload.
    /// Returns `(avg_ms_per_call, last_output)`.
    pub fn bench_matvec(
        &self,
        packed: &[u32],
        scales: &[u8],
        x: &[f32],
        out_features: usize,
        in_features: usize,
        warmup: usize,
        iters: usize,
    ) -> Result<(f64, Vec<f32>)> {
        anyhow::ensure!(
            in_features.is_multiple_of(16),
            "in_features {in_features} must be a multiple of 16 (NVFP4 group size)"
        );
        anyhow::ensure!(
            packed.len() == out_features * in_features / 8,
            "packed length {} != {}",
            packed.len(),
            out_features * in_features / 8
        );
        anyhow::ensure!(
            scales.len() == out_features * in_features / 16,
            "scales length {} != {}",
            scales.len(),
            out_features * in_features / 16
        );
        anyhow::ensure!(
            x.len() == in_features,
            "x length {} != {in_features}",
            x.len()
        );

        let packed_buf = self.ctx.buffer_with_data(packed);
        let scales_buf = self.ctx.buffer_with_data(scales);
        let x_buf = self.ctx.buffer_with_data(x);
        let y_buf = self.ctx.buffer_for::<f32>(out_features);
        let dims = NvFp4Dims {
            out_features: out_features as u32,
            in_features: in_features as u32,
        };
        let dims_buf = self.ctx.buffer_with_data(std::slice::from_ref(&dims));

        let threads_per_tg = self.matvec_f32.max_total_threads_per_threadgroup().min(256) as usize;
        let grid = MTLSize {
            width: out_features,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: threads_per_tg,
            height: 1,
            depth: 1,
        };

        let dispatch = || {
            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            encoder.set_compute_pipeline_state(&self.matvec_f32);
            encoder.set_buffer(0, Some(&packed_buf), 0usize);
            encoder.set_buffer(1, Some(&scales_buf), 0usize);
            encoder.set_buffer(2, Some(&x_buf), 0usize);
            encoder.set_buffer(3, Some(&y_buf), 0usize);
            encoder.set_buffer(4, Some(&dims_buf), 0usize);
            encoder.dispatch_threads(grid, tg);
            drop(encoder);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
        };

        for _ in 0..warmup {
            dispatch();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            dispatch();
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters.max(1) as f64;
        let y = self.ctx.read_buffer::<f32>(&y_buf, out_features);
        Ok((ms, y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nvfp4::dequantize_f32;

    const OUT: usize = 2;
    const IN: usize = 32;

    /// mlx-verified golden NVFP4 weight (2×32, group 16), from `crate::nvfp4` tests.
    fn golden() -> ([u32; 8], [u8; 4]) {
        (
            [
                0x4e75795b, 0xe6ec6617, 0xe453e356, 0x5fa5a1ca, 0xc300d25c, 0x0ecaa168, 0x8419dc74,
                0x5bbb1b88,
            ],
            [38, 44, 47, 47],
        )
    }

    /// CPU dequant of the golden weight, row-major `[OUT, IN]`.
    fn golden_w() -> Vec<f32> {
        let (packed, scales) = golden();
        let mut w = vec![0.0f32; OUT * IN];
        dequantize_f32(&packed, &scales, &mut w).unwrap();
        w
    }

    fn try_ctx() -> Option<Nvfp4Context> {
        match Nvfp4Context::new() {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("skipping nvfp4 GPU parity (no Metal device): {e}");
                None
            }
        }
    }

    fn assert_close(got: &[f32], want: &[f32]) {
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            let rel = (g - w).abs() / w.abs().max(1e-6);
            assert!(rel < 1e-5, "idx {i}: gpu={g} cpu={w} rel={rel}");
        }
    }

    /// GPU matvec must match a CPU dequant-then-dot reference (same nibble/scale decode,
    /// same accumulation order → expect exact f32 agreement; assert rel < 1e-5 for safety).
    #[test]
    fn matvec_parity_vs_cpu_dequant() {
        let (packed, scales) = golden();
        let w = golden_w();
        let x: Vec<f32> = (0..IN).map(|i| 0.1 * (i as f32) - 1.0).collect();
        let want: Vec<f32> = (0..OUT)
            .map(|r| (0..IN).map(|k| w[r * IN + k] * x[k]).sum())
            .collect();

        let Some(ctx) = try_ctx() else { return };
        let got = ctx.matvec_f32(&packed, &scales, &x, OUT, IN).unwrap();
        assert_close(&got, &want);
    }

    /// GPU batched matmul (batch=3) vs CPU reference.
    #[test]
    fn matmul_parity_vs_cpu_dequant() {
        let (packed, scales) = golden();
        let w = golden_w();
        let batch = 3usize;
        let x: Vec<f32> = (0..batch * IN).map(|i| 0.05 * (i as f32) - 0.7).collect();
        let mut want = vec![0.0f32; batch * OUT];
        for b in 0..batch {
            for r in 0..OUT {
                want[b * OUT + r] = (0..IN).map(|k| w[r * IN + k] * x[b * IN + k]).sum();
            }
        }

        let Some(ctx) = try_ctx() else { return };
        let got = ctx
            .matmul_f32(&packed, &scales, &x, OUT, IN, batch)
            .unwrap();
        assert_close(&got, &want);
    }

    /// GPU MoE grouped matmul: 2-expert slab (expert 1 = expert 0 weights), expert_indices
    /// = [1, 0] so slot 0 reads expert 1 and slot 1 reads expert 0. Output `[k, batch, OUT]`.
    #[test]
    fn matmul_moe_parity_vs_cpu_dequant() {
        let (packed, scales) = golden();
        let w = golden_w();
        // Two identical experts back-to-back (per-expert slab layout).
        let mut packed_all = packed.to_vec();
        packed_all.extend_from_slice(&packed);
        let mut scales_all = scales.to_vec();
        scales_all.extend_from_slice(&scales);
        let expert_indices = [1u32, 0u32];
        let batch = 1usize;
        let x: Vec<f32> = (0..IN).map(|i| 0.1 * (i as f32) - 1.0).collect();

        // CPU reference: both experts equal w → every slot gives the same matvec.
        let mv: Vec<f32> = (0..OUT)
            .map(|r| (0..IN).map(|k| w[r * IN + k] * x[k]).sum())
            .collect();
        let mut want = Vec::new();
        for _slot in 0..expert_indices.len() {
            want.extend_from_slice(&mv);
        }

        let Some(ctx) = try_ctx() else { return };
        let got = ctx
            .matmul_moe_f32(
                &packed_all,
                &scales_all,
                &expert_indices,
                &x,
                OUT,
                IN,
                batch,
            )
            .unwrap();
        assert_close(&got, &want);
    }
}
