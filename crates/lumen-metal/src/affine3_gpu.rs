//! Affine 3-bit dequant + matvec (Apple Silicon Metal).
//!
//! Bit-plane packing layout (32 elements per 3 u32 = 96 bits):
//!   chunk[0] = u32_lo  (LSBs across 32 elements)
//!   chunk[1] = u32_mid (mid bits)
//!   chunk[2] = u32_hi  (MSBs)
//!
//! Group size = 64 elements (matches Affine4); 2 chunks per group.
//! Per-group: 1 bf16 scale + 1 bf16 bias (unchanged from Affine4).
//!
//! POC scope: ONLY `affine3_matvec_bf16in_bf16out` (1 thread per output row).
//! Intent: validate that 25% packed-byte saving translates to actual µs/call
//! reduction despite ~2.25× dequant compute overhead vs Affine4.
//!
//! See `phase18a_affine3_design.md` for risk analysis + roadmap.

use crate::metal::{BatchedEncoderExt, CommandBufferExt, ComputeEncoderCompat};
use anyhow::{Result, anyhow, ensure};

use crate::device::MetalContext;
use crate::metal::{Buffer, ComputePipelineState, Library, MTLSize};

const SHADER_SRC: &str = include_str!("shaders/affine3.metal");

pub const AFFINE3_GROUP_SIZE: usize = 64;
pub const AFFINE3_CHUNK_SIZE: usize = 32;

#[repr(C)]
#[derive(Clone, Copy)]
struct Affine3Dims {
    out_features: u32,
    in_features: u32,
}

pub struct Affine3Weight {
    packed: Buffer,
    scales: Buffer,
    biases: Buffer,
    pub out_features: usize,
    pub in_features: usize,
}

impl Affine3Weight {
    /// Pack raw 3-bit codes (each in [0..7]) into bit-plane layout and upload.
    ///
    /// Shape contract:
    ///   - `codes.len() == out * in`              (one byte per element)
    ///   - `scales_bf16.len() == out * in / 64`   (bf16 stored as u16)
    ///   - `biases_bf16.len() == out * in / 64`   (bf16 stored as u16)
    ///   - `in_features` is a multiple of 64
    pub fn from_codes_3bit(
        ctx: &MetalContext,
        codes: &[u8],
        scales_bf16: &[u16],
        biases_bf16: &[u16],
        out_features: usize,
        in_features: usize,
    ) -> Result<Self> {
        ensure!(
            in_features.is_multiple_of(AFFINE3_GROUP_SIZE),
            "in_features {in_features} must be multiple of 64 (group size)"
        );
        let expected_codes = out_features * in_features;
        ensure!(
            codes.len() == expected_codes,
            "codes length {} != {expected_codes}",
            codes.len()
        );
        let groups = out_features * in_features / AFFINE3_GROUP_SIZE;
        ensure!(
            scales_bf16.len() == groups,
            "scales length {} != {groups}",
            scales_bf16.len()
        );
        ensure!(
            biases_bf16.len() == groups,
            "biases length {} != {groups}",
            biases_bf16.len()
        );

        // Pack 32 elements at a time into bit-plane format.
        let n_chunks = out_features * in_features / AFFINE3_CHUNK_SIZE;
        let mut packed = vec![0u32; n_chunks * 3];
        for chunk_idx in 0..n_chunks {
            let mut u_lo: u32 = 0;
            let mut u_mid: u32 = 0;
            let mut u_hi: u32 = 0;
            for i in 0..32usize {
                let code = codes[chunk_idx * 32 + i];
                debug_assert!(code < 8, "code {code} out of 3-bit range [0..7]");
                u_lo |= ((code & 1) as u32) << i;
                u_mid |= (((code >> 1) & 1) as u32) << i;
                u_hi |= (((code >> 2) & 1) as u32) << i;
            }
            packed[chunk_idx * 3] = u_lo;
            packed[chunk_idx * 3 + 1] = u_mid;
            packed[chunk_idx * 3 + 2] = u_hi;
        }

        Ok(Self {
            packed: ctx.buffer_with_data(&packed),
            scales: ctx.buffer_with_data(scales_bf16),
            biases: ctx.buffer_with_data(biases_bf16),
            out_features,
            in_features,
        })
    }

    /// Bytes-on-device estimate (packed 3-bit + 2× bf16 group meta).
    pub fn approx_bytes(&self) -> usize {
        let packed_bytes = self.out_features * self.in_features * 3 / 8; // 0.375 B/elem
        let meta_bytes = self.out_features * self.in_features / AFFINE3_GROUP_SIZE * 2 * 2;
        packed_bytes + meta_bytes
    }

    pub fn buffers(&self) -> (&Buffer, &Buffer, &Buffer) {
        (&self.packed, &self.scales, &self.biases)
    }
}

pub struct Affine3Context {
    pub ctx: MetalContext,
    matvec_bf16in_bf16out: ComputePipelineState,
    qmv_fast_bf16in_bf16out: ComputePipelineState,
    #[allow(dead_code)]
    library: Library,
}

impl Affine3Context {
    /// `in_features` constraint for the qmv_fast topology.
    pub const QMV_FAST_BLK: usize = 512;

    pub fn qmv_fast_supports(in_features: usize, out_features: usize) -> bool {
        in_features.is_multiple_of(Self::QMV_FAST_BLK) && out_features.is_multiple_of(8)
    }

    pub fn new() -> Result<Self> {
        let ctx = MetalContext::new()?;
        let options = crate::metal::new_compile_options();
        options.set_language_version(crate::metal::MTLLanguageVersion::Version3_1);
        options.set_fast_math_enabled(true);
        let library = ctx
            .device
            .new_library_with_source(SHADER_SRC, Some(options.as_ref()))
            .map_err(|e| anyhow!("Affine3 shader compile error: {e}"))?;

        let func = library
            .get_function("affine3_matvec_bf16in_bf16out", None)
            .map_err(|e| anyhow!("kernel `affine3_matvec_bf16in_bf16out` not found: {e}"))?;
        let matvec_bf16in_bf16out = ctx
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow!("pipeline `affine3_matvec_bf16in_bf16out` failed: {e}"))?;

        let qmv_fast_func = library
            .get_function("affine3_qmv_fast_bf16in_bf16out", None)
            .map_err(|e| anyhow!("kernel `affine3_qmv_fast_bf16in_bf16out` not found: {e}"))?;
        let qmv_fast_bf16in_bf16out = ctx
            .device
            .new_compute_pipeline_state_with_function(&qmv_fast_func)
            .map_err(|e| anyhow!("pipeline `affine3_qmv_fast_bf16in_bf16out` failed: {e}"))?;

        Ok(Self {
            ctx,
            matvec_bf16in_bf16out,
            qmv_fast_bf16in_bf16out,
            library,
        })
    }

    /// Encode + commit a single matvec dispatch. Caller is responsible for
    /// `wait_until_completed` if synchronous behavior is needed; without
    /// wait, callers can pipeline N forwards and sync once at the end.
    pub fn matvec_bf16in_bf16out_pipelined(
        &self,
        weight: &Affine3Weight,
        x_buf: &Buffer,
        y_buf: &Buffer,
    ) -> Result<()> {
        let dims = Affine3Dims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
        };

        let encoder = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        encoder.set_label("lumen:affine3_matvec");
        encoder.set_compute_pipeline_state(&self.matvec_bf16in_bf16out);
        encoder.set_buffer(0, Some(&weight.packed), 0);
        encoder.set_buffer(1, Some(&weight.scales), 0);
        encoder.set_buffer(2, Some(&weight.biases), 0);
        encoder.set_buffer(3, Some(x_buf), 0);
        encoder.set_buffer(4, Some(y_buf), 0);
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<Affine3Dims>(),
            &dims as *const _ as *const _,
        );

        let max_threads = self
            .matvec_bf16in_bf16out
            .max_total_threads_per_threadgroup();
        let threads_per_tg = max_threads.min(256);
        let grid = MTLSize {
            width: weight.out_features,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: threads_per_tg,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(grid, tg);
        // No wait — caller pipelines.
        Ok(())
    }

    /// qmv_fast topology variant. Production-shaped
    /// kernel (NSG=2, RPS=4, VPT=16, BLK=512). Mirrors Affine4's qmv_fast
    /// signature but with bit-plane decode.
    ///
    /// Caller pipelines (no wait); call `drain` separately.
    pub fn qmv_fast_bf16in_bf16out_pipelined(
        &self,
        weight: &Affine3Weight,
        x_buf: &Buffer,
        y_buf: &Buffer,
        batch: usize,
    ) -> Result<()> {
        ensure!(
            Self::qmv_fast_supports(weight.in_features, weight.out_features),
            "qmv_fast requires in % {} == 0 and out % 8 == 0 (got in={}, out={})",
            Self::QMV_FAST_BLK,
            weight.in_features,
            weight.out_features
        );
        let dims = Affine3Dims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
        };
        let batch_u32 = batch as u32;

        let encoder = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        encoder.set_label("lumen:affine3_qmv_fast");
        encoder.set_compute_pipeline_state(&self.qmv_fast_bf16in_bf16out);
        encoder.set_buffer(0, Some(&weight.packed), 0);
        encoder.set_buffer(1, Some(&weight.scales), 0);
        encoder.set_buffer(2, Some(&weight.biases), 0);
        encoder.set_buffer(3, Some(x_buf), 0);
        encoder.set_buffer(4, Some(y_buf), 0);
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<Affine3Dims>(),
            &dims as *const _ as *const _,
        );
        encoder.set_bytes_directly(
            6,
            std::mem::size_of::<u32>(),
            &batch_u32 as *const _ as *const _,
        );

        // NSG=2 simdgroups per TG, RPS=4 rows per simdgroup → 8 rows per TG.
        // Grid: (batch, out / 8, 1). TG: (1, 64, 1).
        let tg_rows = weight.out_features / 8;
        let grid = MTLSize {
            width: batch,
            height: tg_rows,
            depth: 1,
        };
        let tg = MTLSize {
            width: 1,
            height: 64,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }

    /// Synchronous variant for parity testing — submits, waits, returns.
    pub fn matvec_bf16in_bf16out_sync(
        &self,
        weight: &Affine3Weight,
        x_buf: &Buffer,
        y_buf: &Buffer,
    ) -> Result<()> {
        self.matvec_bf16in_bf16out_pipelined(weight, x_buf, y_buf)?;
        // Drain through the shared cmk Commands scheduler so the caller sees
        // a fully-realized GPU result, including any prior pipelined work.
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_from_f32(x: f32) -> u16 {
        (x.to_bits() >> 16) as u16
    }

    fn bf16_to_f32(b: u16) -> f32 {
        f32::from_bits((b as u32) << 16)
    }

    /// Identity dequant: scale=1.0, bias=0.0 → reconstructed value == raw 3-bit code.
    /// Verifies the bit-plane packing/unpacking + kernel reconstruction.
    #[test]
    fn identity_dequant_matvec_matches_cpu() -> Result<()> {
        let ctx = match Affine3Context::new() {
            Ok(c) => c,
            Err(_) => return Ok(()), // No Metal device → skip on CI without GPU.
        };

        let out: usize = 8;
        let in_f: usize = 64;

        // Codes: 0..7 cycling.
        let codes: Vec<u8> = (0..out * in_f).map(|i| (i % 8) as u8).collect();
        let groups = out * in_f / AFFINE3_GROUP_SIZE;
        let scales: Vec<u16> = vec![bf16_from_f32(1.0); groups];
        let biases: Vec<u16> = vec![bf16_from_f32(0.0); groups];

        let weight = Affine3Weight::from_codes_3bit(&ctx.ctx, &codes, &scales, &biases, out, in_f)?;

        // Activation: all 1.0 in bf16 → output should be sum of decoded codes.
        let x_data: Vec<u16> = vec![bf16_from_f32(1.0); in_f];
        let x_buf = ctx.ctx.buffer_with_data(&x_data);
        let y_buf = ctx.ctx.buffer_for::<u16>(out);

        ctx.matvec_bf16in_bf16out_sync(&weight, &x_buf, &y_buf)?;
        let y_bf16 = ctx.ctx.read_buffer::<u16>(&y_buf, out);
        let y_f32: Vec<f32> = y_bf16.iter().map(|&b| bf16_to_f32(b)).collect();

        // CPU reference: per row, sum codes for that row.
        for row in 0..out {
            let cpu_sum: f32 = (0..in_f).map(|k| codes[row * in_f + k] as f32).sum();
            let gpu = y_f32[row];
            // bf16 has ~7-bit mantissa → relative tol ~ 1/128 = 0.78%.
            let rel_err = (gpu - cpu_sum).abs() / cpu_sum.abs().max(1e-6);
            assert!(
                rel_err < 0.01,
                "row {row}: cpu={cpu_sum} gpu={gpu} rel_err={rel_err}"
            );
        }
        Ok(())
    }
}
