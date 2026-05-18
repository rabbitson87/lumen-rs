//! Metal dispatcher for 8-bit MLX-format affine fused dequant + matmul.
//!
//! Sister to [`crate::affine4_gpu`] for `mlx.quantize(bits=8, group_size=64)`
//! checkpoints (e.g. `Qwen3-Embedding-0.6B-8bit`).
//!
//! Memory layout (matches on-disk MLX format):
//!   - `packed`  : `[out, in/4]` `u32`     (4 bytes per word, LSB-first byte 0..3)
//!   - `scales`  : `[out, in/64]` `bf16`   (per-group scale)
//!   - `biases`  : `[out, in/64]` `bf16`   (per-group bias)
//!
//! Dequant: `w[o,i] = byte_in(packed[o, i/4], i%4) * scales[o, i/64] + biases[o, i/64]`.
//!
//! Single batched kernel (`affine8_matmul_bf16`, 1 thread per output
//! element). Sufficient for the Qwen3-Embedding-0.6B shapes
//! (in ≤ 3072, out ≤ 3072). qmv_fast cooperative variants follow when
//! `in_features % 512 == 0` and `out_features % 8 == 0`.

use anyhow::Result;

use crate::device::MetalContext;
use crate::metal::{Buffer, ComputePipelineState, Library, MTLSize};
use crate::metal::ComputeEncoderCompat;

const SHADER_SRC: &str = include_str!("shaders/affine8.metal");

/// 8-bit MLX-format fixed group size.
pub const AFFINE8_GROUP_SIZE: usize = 64;

#[repr(C)]
#[derive(Clone, Copy)]
struct Affine8Dims {
    out_features: u32,
    in_features: u32,
}

/// GPU-resident 8-bit MLX-format weight. Allocated once at model load.
pub struct Affine8Weight {
    packed: Buffer,
    scales: Buffer,
    biases: Buffer,
    pub out_features: usize,
    pub in_features: usize,
}

impl Affine8Weight {
    /// Upload packed u32 + bf16 scales + bf16 biases to device.
    ///
    /// Shape contract:
    ///   - `packed.len() == out * in / 4`     (u32 words)
    ///   - `scales.len() == out * in / 64`    (bf16 as u16)
    ///   - `biases.len() == out * in / 64`    (bf16 as u16)
    ///   - `in_features` is a multiple of 64.
    pub fn from_host(
        ctx: &MetalContext,
        packed: &[u32],
        scales_bf16: &[u16],
        biases_bf16: &[u16],
        out_features: usize,
        in_features: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            in_features.is_multiple_of(AFFINE8_GROUP_SIZE),
            "in_features {in_features} must be a multiple of 64"
        );
        let expected_packed = out_features * in_features / 4;
        anyhow::ensure!(
            packed.len() == expected_packed,
            "packed length {} != {}",
            packed.len(),
            expected_packed
        );
        let expected_groups = out_features * in_features / AFFINE8_GROUP_SIZE;
        anyhow::ensure!(
            scales_bf16.len() == expected_groups,
            "scales length {} != {}",
            scales_bf16.len(),
            expected_groups
        );
        anyhow::ensure!(
            biases_bf16.len() == expected_groups,
            "biases length {} != {}",
            biases_bf16.len(),
            expected_groups
        );
        Ok(Self {
            packed: ctx.buffer_with_data(packed),
            scales: ctx.buffer_with_data(scales_bf16),
            biases: ctx.buffer_with_data(biases_bf16),
            out_features,
            in_features,
        })
    }

    /// Device byte-footprint estimate (packed u32 + bf16 scales + bf16 biases).
    pub fn approx_bytes(&self) -> usize {
        let packed = self.out_features * self.in_features / 4 * 4;
        let sb = self.out_features * self.in_features / AFFINE8_GROUP_SIZE * 2 * 2;
        packed + sb
    }

    pub fn packed_buffer(&self) -> &Buffer {
        &self.packed
    }
}

/// Compiled Metal pipelines for the 8-bit MLX format.
///
/// Holds two variants:
///   - `matmul_bf16`: naive 1 thread/output kernel (BW-bound for in_features
///     small / out_features small; always usable).
///   - `qmv_fast_bf16`: cooperative 32-lane simdgroup kernel (NSG=2 × RPS=4
///     = 8 outputs per TG); requires `in_features % 512 == 0`
///     AND `out_features % 8 == 0`. Mirrors `affine4_qmv_fast` layout.
pub struct Affine8Context {
    pub ctx: MetalContext,
    matmul_bf16: ComputePipelineState,
    qmv_fast_bf16: ComputePipelineState,
    #[allow(dead_code)]
    library: Library,
}

impl Affine8Context {
    pub fn new() -> Result<Self> {
        let ctx = MetalContext::new()?;
        let options = crate::metal::new_compile_options();
        options.set_language_version(crate::metal::MTLLanguageVersion::Version3_1);
        options.set_fast_math_enabled(true);
        let library = ctx
            .device
            .new_library_with_source(SHADER_SRC, Some(options.as_ref()))
            .map_err(|e| anyhow::anyhow!("Affine8 shader compile: {e}"))?;
        let func = library
            .get_function("affine8_matmul_bf16", None)
            .map_err(|e| anyhow::anyhow!("kernel `affine8_matmul_bf16` missing: {e}"))?;
        let matmul_bf16 = ctx
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow::anyhow!("pipeline `affine8_matmul_bf16` failed: {e}"))?;
        let qmv_func = library
            .get_function("affine8_qmv_fast_bf16", None)
            .map_err(|e| anyhow::anyhow!("kernel `affine8_qmv_fast_bf16` missing: {e}"))?;
        let qmv_fast_bf16 = ctx
            .device
            .new_compute_pipeline_state_with_function(&qmv_func)
            .map_err(|e| anyhow::anyhow!("pipeline `affine8_qmv_fast_bf16` failed: {e}"))?;
        Ok(Self {
            ctx,
            matmul_bf16,
            qmv_fast_bf16,
            library,
        })
    }

    /// Whether the cooperative qmv_fast kernel can dispatch for these dims.
    /// Constraint mirrors affine4: BLK=512 alignment on in, NSG*RPS=8 on out.
    pub fn qmv_fast_supports(in_features: usize, out_features: usize) -> bool {
        in_features % 512 == 0 && out_features % 8 == 0
    }

    /// Encode `y = x @ W^T` into the caller's compute encoder.
    /// x: bf16 (batch, in_features), y: bf16 (batch, out_features).
    ///
    /// Selects the qmv_fast cooperative kernel when alignment allows
    /// (in % 512 == 0 AND out % 8 == 0); falls back to the naive
    /// 1-thread/output kernel otherwise. The env var
    /// `LUMEN_AFFINE8_NAIVE=1` forces the naive path (for A/B testing).
    pub fn encode_matmul_bf16_dispatch(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Affine8Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) {
        let force_naive = std::env::var("LUMEN_AFFINE8_NAIVE")
            .map(|v| v == "1")
            .unwrap_or(false);
        let can_qmv_fast =
            !force_naive && Self::qmv_fast_supports(weight.in_features, weight.out_features);

        if can_qmv_fast {
            self.encode_qmv_fast_bf16_dispatch(encoder, weight, x_buf, x_offset, y_buf, y_offset, batch);
        } else {
            self.encode_naive_bf16_dispatch(encoder, weight, x_buf, x_offset, y_buf, y_offset, batch);
        }
    }

    /// Naive 1-thread-per-output dispatch. Always works. Used as a fallback
    /// when qmv_fast alignment isn't met, and for A/B perf testing.
    pub fn encode_naive_bf16_dispatch(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Affine8Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.matmul_bf16);
        encoder.set_buffer(0, Some(&weight.packed), 0);
        encoder.set_buffer(1, Some(&weight.scales), 0);
        encoder.set_buffer(2, Some(&weight.biases), 0);
        encoder.set_buffer(3, Some(x_buf), x_offset as usize);
        encoder.set_buffer(4, Some(y_buf), y_offset as usize);
        let dims = Affine8Dims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
        };
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<Affine8Dims>(),
            &dims as *const _ as *const _,
        );
        let batch_u32 = batch as u32;
        encoder.set_bytes_directly(6, 4, &batch_u32 as *const _ as *const _);

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

    /// Cooperative simdgroup dispatch (qmv_fast). 32 lanes split the K
    /// dimension; one simdgroup produces 4 contiguous output rows; 2
    /// simdgroups per TG (64 threads / 8 outputs per TG).
    ///
    /// Caller must ensure `qmv_fast_supports(in, out) == true`. Mirrors
    /// `Affine4Context::encode_qmv_fast_bf16in_bf16out_dispatch`.
    pub fn encode_qmv_fast_bf16_dispatch(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Affine8Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) {
        debug_assert!(Self::qmv_fast_supports(weight.in_features, weight.out_features));
        encoder.set_compute_pipeline_state(&self.qmv_fast_bf16);
        encoder.set_buffer(0, Some(&weight.packed), 0);
        encoder.set_buffer(1, Some(&weight.scales), 0);
        encoder.set_buffer(2, Some(&weight.biases), 0);
        encoder.set_buffer(3, Some(x_buf), x_offset as usize);
        encoder.set_buffer(4, Some(y_buf), y_offset as usize);
        let dims = Affine8Dims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
        };
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<Affine8Dims>(),
            &dims as *const _ as *const _,
        );
        let batch_u32 = batch as u32;
        encoder.set_bytes_directly(6, 4, &batch_u32 as *const _ as *const _);

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

    /// One-shot synchronous matmul. Allocates device buffers from host
    /// `x_bf16` (u16-coded bf16), runs the kernel, reads back `y_bf16`.
    /// Used by unit tests + microbench.
    pub fn matmul_bf16_with_weight(
        &self,
        weight: &Affine8Weight,
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
            encoder.set_label("lumen:affine8_matmul_bf16");
            self.encode_matmul_bf16_dispatch(encoder.as_ref(), weight, &x_buf, 0, &y_buf, 0, batch);
            drop(encoder);
        }
        crate::metal::process_commands().flush_and_wait().expect("flush");
        Ok(self
            .ctx
            .read_buffer::<u16>(&y_buf, batch * weight.out_features))
    }
}

/// CPU reference: dequant 8-bit MLX-format weight and compute matmul.
/// Public for use by the integration parity test (`tests/affine8_parity.rs`).
pub fn cpu_reference_matmul_bf16(
    packed: &[u32],
    scales: &[u16],
    biases: &[u16],
    x_bf16: &[u16],
    out: usize,
    in_f: usize,
    batch: usize,
) -> Vec<u16> {
    cpu_reference_impl(packed, scales, biases, x_bf16, out, in_f, batch)
}

fn cpu_reference_impl(
    packed: &[u32],
    scales: &[u16],
    biases: &[u16],
    x_bf16: &[u16],
    out: usize,
    in_f: usize,
    batch: usize,
) -> Vec<u16> {
    use half::bf16;
    let groups = in_f / AFFINE8_GROUP_SIZE;
    let mut y = vec![0u16; batch * out];
    for b in 0..batch {
        for o in 0..out {
            let mut acc = 0f32;
            for g in 0..groups {
                let s = bf16::from_bits(scales[o * groups + g]).to_f32();
                let bi = bf16::from_bits(biases[o * groups + g]).to_f32();
                for w in 0..(AFFINE8_GROUP_SIZE / 4) {
                    let word = packed[o * (in_f / 4) + g * (AFFINE8_GROUP_SIZE / 4) + w];
                    for byte_idx in 0..4 {
                        let q = ((word >> (8 * byte_idx)) & 0xFF) as f32;
                        let i = g * AFFINE8_GROUP_SIZE + w * 4 + byte_idx;
                        let xv = bf16::from_bits(x_bf16[b * in_f + i]).to_f32();
                        acc += (q * s + bi) * xv;
                    }
                }
            }
            y[b * out + o] = bf16::from_f32(acc).to_bits();
        }
    }
    y
}
