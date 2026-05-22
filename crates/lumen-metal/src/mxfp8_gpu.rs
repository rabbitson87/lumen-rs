//! Metal dispatcher for MXFP8 (OCP) fused dequant + matmul.
//!
//! Sister to [`crate::affine8_gpu`] but for
//! `mlx.quantize(bits=8, group_size=32, mode="mxfp8")` checkpoints
//! (e.g. `mlx-community/Qwen3-Embedding-4B-mxfp8`).
//!
//! Memory layout (matches on-disk MLX format):
//!   - `packed`  : `[out, in/4]` `u32`     (4 E4M3 bytes per word, LSB-first)
//!   - `scales`  : `[out, in/32]` `u8`     (one E8M0 byte per 32-element group)
//!   - **no biases** — MXFP8 is symmetric (no zero-point); E4M3 sign bit
//!     carries the sign.
//!
//! Dequant:
//!   `w[o,i] = e4m3_to_f32(packed_byte[o,i]) * 2^(e8m0[o, i/32] - 127)`.

use anyhow::Result;

use crate::device::MetalContext;
use crate::metal::ComputeEncoderCompat;
use crate::metal::{Buffer, ComputePipelineState, Library, MTLSize};

const SHADER_SRC: &str = include_str!("shaders/mxfp8.metal");

/// MXFP8 (OCP) fixed group size.
pub const MXFP8_GROUP_SIZE: usize = 32;

#[repr(C)]
#[derive(Clone, Copy)]
struct Mxfp8Dims {
    out_features: u32,
    in_features: u32,
}

/// GPU-resident MXFP8 weight. Allocated once at model load.
///
/// Memory footprint per logical element:
///   1 byte (E4M3) + 1 byte (E8M0) / 32 = ~1.031 bytes/elem
/// vs AFFINE 8-bit: 1 byte (uint8) + 4 bytes (f16 scale+bias) / 64 = 1.0625
/// → MXFP8 is ~3% smaller at iso-bit (the scale tax is half as expensive
/// but the group is half as wide, so it nearly cancels out).
pub struct Mxfp8Weight {
    packed: Buffer,
    scales: Buffer,
    pub out_features: usize,
    pub in_features: usize,
}

impl Mxfp8Weight {
    /// Upload packed u32 (4 E4M3 bytes/word) + u8 E8M0 scales to device.
    ///
    /// Shape contract:
    ///   - `packed.len() == out * in / 4`     (u32 words)
    ///   - `scales.len() == out * in / 32`    (one byte per group)
    ///   - `in_features` is a multiple of 32.
    pub fn from_host(
        ctx: &MetalContext,
        packed: &[u32],
        scales_e8m0: &[u8],
        out_features: usize,
        in_features: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            in_features.is_multiple_of(MXFP8_GROUP_SIZE),
            "in_features {in_features} must be a multiple of {MXFP8_GROUP_SIZE}"
        );
        let expected_packed = out_features * in_features / 4;
        anyhow::ensure!(
            packed.len() == expected_packed,
            "packed length {} != {}",
            packed.len(),
            expected_packed
        );
        let expected_groups = out_features * in_features / MXFP8_GROUP_SIZE;
        anyhow::ensure!(
            scales_e8m0.len() == expected_groups,
            "scales length {} != {}",
            scales_e8m0.len(),
            expected_groups
        );
        Ok(Self {
            packed: ctx.buffer_with_data(packed),
            scales: ctx.buffer_with_data(scales_e8m0),
            out_features,
            in_features,
        })
    }

    /// Device byte-footprint estimate (packed u32 + u8 E8M0 scales).
    pub fn approx_bytes(&self) -> usize {
        let packed = self.out_features * self.in_features; // 1 byte/elem
        let scales = self.out_features * self.in_features / MXFP8_GROUP_SIZE;
        packed + scales
    }

    pub fn packed_buffer(&self) -> &Buffer {
        &self.packed
    }
}

/// Compiled Metal pipelines for the MXFP8 format.
///
///   - `matmul_bf16`: naive 1 thread/output kernel; always usable.
///   - `qmv_fast_bf16`: cooperative 32-lane simdgroup kernel (NSG=2 × RPS=4
///     = 8 outputs per TG); requires `in_features % 512 == 0`
///     AND `out_features % 8 == 0`.
pub struct Mxfp8Context {
    pub ctx: MetalContext,
    matmul_bf16: ComputePipelineState,
    qmv_fast_bf16: ComputePipelineState,
    #[allow(dead_code)]
    library: Library,
}

impl Mxfp8Context {
    pub fn new() -> Result<Self> {
        let ctx = MetalContext::new()?;
        let options = crate::metal::new_compile_options();
        options.set_language_version(crate::metal::MTLLanguageVersion::Version3_1);
        options.set_fast_math_enabled(true);
        let library = ctx
            .device
            .new_library_with_source(SHADER_SRC, Some(options.as_ref()))
            .map_err(|e| anyhow::anyhow!("MXFP8 shader compile: {e}"))?;
        let func = library
            .get_function("mxfp8_matmul_bf16", None)
            .map_err(|e| anyhow::anyhow!("kernel `mxfp8_matmul_bf16` missing: {e}"))?;
        let matmul_bf16 = ctx
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow::anyhow!("pipeline `mxfp8_matmul_bf16` failed: {e}"))?;
        let qmv_func = library
            .get_function("mxfp8_qmv_fast_bf16", None)
            .map_err(|e| anyhow::anyhow!("kernel `mxfp8_qmv_fast_bf16` missing: {e}"))?;
        let qmv_fast_bf16 = ctx
            .device
            .new_compute_pipeline_state_with_function(&qmv_func)
            .map_err(|e| anyhow::anyhow!("pipeline `mxfp8_qmv_fast_bf16` failed: {e}"))?;
        Ok(Self {
            ctx,
            matmul_bf16,
            qmv_fast_bf16,
            library,
        })
    }

    /// Whether the cooperative qmv_fast kernel can dispatch for these dims.
    /// Same constraint as AFFINE 8: BLK=512 on in, NSG*RPS=8 on out.
    pub fn qmv_fast_supports(in_features: usize, out_features: usize) -> bool {
        in_features.is_multiple_of(512) && out_features.is_multiple_of(8)
    }

    /// Encode `y = x @ W^T` into the caller's compute encoder.
    /// Selects qmv_fast when alignment allows; falls back to naive otherwise.
    /// `LUMEN_MXFP8_NAIVE=1` forces the naive kernel (A/B testing).
    pub fn encode_matmul_bf16_dispatch(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Mxfp8Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) {
        let force_naive = std::env::var("LUMEN_MXFP8_NAIVE")
            .map(|v| v == "1")
            .unwrap_or(false);
        let can_qmv_fast =
            !force_naive && Self::qmv_fast_supports(weight.in_features, weight.out_features);

        if can_qmv_fast {
            self.encode_qmv_fast_bf16_dispatch(
                encoder, weight, x_buf, x_offset, y_buf, y_offset, batch,
            );
        } else {
            self.encode_naive_bf16_dispatch(
                encoder, weight, x_buf, x_offset, y_buf, y_offset, batch,
            );
        }
    }

    /// Naive 1-thread-per-output dispatch. Always works.
    pub fn encode_naive_bf16_dispatch(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Mxfp8Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.matmul_bf16);
        encoder.set_buffer(0, Some(&weight.packed), 0);
        encoder.set_buffer(1, Some(&weight.scales), 0);
        encoder.set_buffer(2, Some(x_buf), x_offset as usize);
        encoder.set_buffer(3, Some(y_buf), y_offset as usize);
        let dims = Mxfp8Dims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
        };
        encoder.set_bytes_directly(
            4,
            std::mem::size_of::<Mxfp8Dims>(),
            &dims as *const _ as *const _,
        );
        let batch_u32 = batch as u32;
        encoder.set_bytes_directly(5, 4, &batch_u32 as *const _ as *const _);

        let grid = MTLSize {
            width: weight.out_features,
            height: batch,
            depth: 1,
        };
        let tg = MTLSize {
            width: 32.min(weight.out_features),
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(grid, tg);
    }

    /// Cooperative simdgroup dispatch (qmv_fast).
    pub fn encode_qmv_fast_bf16_dispatch(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Mxfp8Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) {
        debug_assert!(Self::qmv_fast_supports(
            weight.in_features,
            weight.out_features
        ));
        encoder.set_compute_pipeline_state(&self.qmv_fast_bf16);
        encoder.set_buffer(0, Some(&weight.packed), 0);
        encoder.set_buffer(1, Some(&weight.scales), 0);
        encoder.set_buffer(2, Some(x_buf), x_offset as usize);
        encoder.set_buffer(3, Some(y_buf), y_offset as usize);
        let dims = Mxfp8Dims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
        };
        encoder.set_bytes_directly(
            4,
            std::mem::size_of::<Mxfp8Dims>(),
            &dims as *const _ as *const _,
        );
        let batch_u32 = batch as u32;
        encoder.set_bytes_directly(5, 4, &batch_u32 as *const _ as *const _);

        // Grid: (batch, out / (NSG*RPS), 1). TG: 64 threads (NSG=2 × SIMD=32).
        const NSG: usize = 2;
        const RPS: usize = 4;
        const THREADS_PER_TG: usize = NSG * 32;
        let row_groups = weight.out_features.div_ceil(NSG * RPS);
        let grid = MTLSize {
            width: batch,
            height: row_groups,
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid, tg);
    }

    /// One-shot synchronous matmul. Used by unit tests.
    pub fn matmul_bf16_with_weight(
        &self,
        weight: &Mxfp8Weight,
        x_bf16: &[u16],
        batch: usize,
    ) -> Result<Vec<u16>> {
        anyhow::ensure!(
            x_bf16.len() == batch * weight.in_features,
            "x length {} != batch({}) * in_features({})",
            x_bf16.len(),
            batch,
            weight.in_features
        );
        if batch == 0 {
            return Ok(Vec::new());
        }
        let x_buf = self.ctx.buffer_with_data(x_bf16);
        let y_buf = self.ctx.buffer_for::<u16>(batch * weight.out_features);
        {
            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("encoder");
            encoder.set_label("lumen:mxfp8_matmul_bf16");
            self.encode_matmul_bf16_dispatch(encoder.as_ref(), weight, &x_buf, 0, &y_buf, 0, batch);
            drop(encoder);
        }
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");
        Ok(self
            .ctx
            .read_buffer::<u16>(&y_buf, batch * weight.out_features))
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Host-side reference dequant — also used to validate Metal kernel.
// ─────────────────────────────────────────────────────────────────────────

/// Decode one OCP E4M3 byte to f32. NaN encoding (S | 1111 | 111) → 0.
/// See `shaders/mxfp8.metal` for the matching device-side decode.
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = (b >> 7) & 0x1;
    let exp = (b >> 3) & 0xF;
    let mant = b & 0x7;
    let f = if exp == 0 {
        // Subnormal: value = mant * 2^-9
        (mant as f32) * (1.0_f32 / 512.0_f32)
    } else if exp == 15 && mant == 7 {
        // E4M3 NaN — caller-defined sentinel; we collapse to 0 to match the
        // kernel + MLX's host-side dequant.
        0.0
    } else {
        let e = (exp as i32) - 7;
        let scale = 2.0_f32.powi(e);
        scale * (1.0 + (mant as f32) / 8.0)
    };
    if sign != 0 { -f } else { f }
}

/// Decode one OCP E8M0 byte to f32. NaN encoding (0xFF) → 0.
pub fn e8m0_to_f32(b: u8) -> f32 {
    if b == 0xFF {
        0.0
    } else {
        2.0_f32.powi((b as i32) - 127)
    }
}

/// CPU reference: dequant MXFP8 weight and compute matmul.
/// Public for use by integration parity tests.
pub fn cpu_reference_matmul_bf16(
    packed: &[u32],
    scales: &[u8],
    x_bf16: &[u16],
    out: usize,
    in_f: usize,
    batch: usize,
) -> Vec<u16> {
    use half::bf16;
    let groups = in_f / MXFP8_GROUP_SIZE;
    let mut y = vec![0u16; batch * out];
    for b in 0..batch {
        for o in 0..out {
            let mut acc = 0f32;
            for g in 0..groups {
                let s = e8m0_to_f32(scales[o * groups + g]);
                for w in 0..(MXFP8_GROUP_SIZE / 4) {
                    let word = packed[o * (in_f / 4) + g * (MXFP8_GROUP_SIZE / 4) + w];
                    for byte_idx in 0..4 {
                        let bv = ((word >> (8 * byte_idx)) & 0xFF) as u8;
                        let q = e4m3_to_f32(bv);
                        let i = g * MXFP8_GROUP_SIZE + w * 4 + byte_idx;
                        let xv = bf16::from_bits(x_bf16[b * in_f + i]).to_f32();
                        acc += (q * s) * xv;
                    }
                }
            }
            y[b * out + o] = bf16::from_f32(acc).to_bits();
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_known_values() {
        // Zero
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x80), -0.0);
        // 1.0 = exp=7, mant=0 → 0b0_0111_000 = 0x38
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        // -1.0 = 0xB8
        assert_eq!(e4m3_to_f32(0xB8), -1.0);
        // 2.0 = exp=8, mant=0 → 0b0_1000_000 = 0x40
        assert_eq!(e4m3_to_f32(0x40), 2.0);
        // 0.5 = exp=6, mant=0 → 0x30
        assert_eq!(e4m3_to_f32(0x30), 0.5);
        // Max normal: exp=15, mant=6 → 0b0_1111_110 = 0x7E → 1.75 * 2^8 = 448
        assert_eq!(e4m3_to_f32(0x7E), 448.0);
        // NaN sentinel collapses to 0
        assert_eq!(e4m3_to_f32(0x7F), 0.0);
        assert_eq!(e4m3_to_f32(0xFF), 0.0);
        // Subnormal: exp=0, mant=1 → 1 / 512
        assert!((e4m3_to_f32(0x01) - (1.0 / 512.0)).abs() < 1e-9);
        // Subnormal: exp=0, mant=7 → 7 / 512
        assert!((e4m3_to_f32(0x07) - (7.0 / 512.0)).abs() < 1e-9);
    }

    #[test]
    fn e8m0_known_values() {
        // 127 → 2^0 = 1.0
        assert_eq!(e8m0_to_f32(127), 1.0);
        // 128 → 2^1 = 2.0
        assert_eq!(e8m0_to_f32(128), 2.0);
        // 126 → 2^-1 = 0.5
        assert_eq!(e8m0_to_f32(126), 0.5);
        // 0 → 2^-127 (very small)
        let tiny = e8m0_to_f32(0);
        assert!(tiny > 0.0 && tiny < 1e-37);
        // NaN sentinel
        assert_eq!(e8m0_to_f32(0xFF), 0.0);
    }

    #[test]
    fn cpu_reference_identity_matmul() {
        // 32 in_features, 1 out_features, scale=1.0 (e8m0=127), all weights = 1.0 (e4m3=0x38)
        // x = [1.0; 32], expected y = 32.0
        use half::bf16;
        let in_f = 32usize;
        let out = 1usize;
        let words_per_row = in_f / 4;
        // All 32 E4M3 bytes = 0x38 (=1.0) packed LSB-first: 0x38_38_38_38
        let mut packed = vec![0u32; out * words_per_row];
        for w in 0..words_per_row {
            packed[w] = 0x3838_3838;
        }
        let scales = vec![127u8; out * (in_f / MXFP8_GROUP_SIZE)];
        let x: Vec<u16> = (0..in_f).map(|_| bf16::from_f32(1.0).to_bits()).collect();
        let y = cpu_reference_matmul_bf16(&packed, &scales, &x, out, in_f, 1);
        assert_eq!(y.len(), 1);
        let yv = bf16::from_bits(y[0]).to_f32();
        assert!((yv - 32.0).abs() < 0.5, "expected ~32.0, got {yv}");
    }
}
