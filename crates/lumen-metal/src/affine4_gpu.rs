//! Metal dispatcher for 4-bit affine fused dequant + matvec/matmul.
//!
//! Sister to [`crate::mxfp4_gpu`] for the 4-bit affine quantization scheme used by
//! MLX checkpoints whose `quantization_config.mode == "affine"` and `bits == 4`
//! (e.g. `Qwen3.6-27B-MLX-4bit`).
//!
//! Differences vs MXFP4:
//!   - Group size: 64 (vs 32)
//!   - Per-group metadata: bf16 scale + bf16 bias (vs single E8M0 byte)
//!   - Nibble interpretation: raw unsigned u4 value 0..15 (vs E2M1 LUT lookup)
//!   - Dequant formula: `w_real = u4_value * scale + bias` (vs `LUT[nib] * scale`)
//!
//! Owns its own Metal library + pipeline states. The kernel source is
//! [`shaders/affine4.metal`].

use crate::metal::{BatchedEncoderExt, CommandBufferExt, ComputeEncoderCompat};
use anyhow::Result;

use crate::device::MetalContext;
use crate::metal::{Buffer, ComputePipelineState, Library, MTLSize};

const SHADER_SRC: &str = include_str!("shaders/affine4.metal");

#[repr(C)]
#[derive(Clone, Copy)]
struct Affine4Dims {
    out_features: u32,
    in_features: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Affine4GateUpSiluMulDims {
    inter: u32,
    in_features: u32,
    batch: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Affine4MatmulRmsnormDims {
    out_features: u32,
    in_features: u32,
    rms_eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Affine4TiledDims {
    out_features: u32,
    in_features: u32,
    tile_in: u32,
    n_chunks: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Affine4ReduceDims {
    out_features: u32,
    n_chunks: u32,
}

/// Pick (tile_in, n_chunks) for the v3-tiled kernel.
///
/// Returns `Some` only when tiling is actually needed (`in_features` exceeds the
/// single-shot v3 budget). `tile_in` is the largest divisor of `in_features` that
/// is `<= AFFINE4_V3_MAX_IN_FEATURES`, a multiple of `AFFINE4_GROUP_SIZE`, and
/// yields equal-sized chunks. Returns `None` to signal the caller should use the
/// non-tiled path (v3 if in fits; v2 otherwise).
pub fn pick_tile_for_in(in_features: usize) -> Option<(usize, usize)> {
    if in_features <= AFFINE4_V3_MAX_IN_FEATURES {
        return None;
    }
    if !in_features.is_multiple_of(AFFINE4_GROUP_SIZE) {
        return None;
    }
    let min_chunks = in_features.div_ceil(AFFINE4_V3_MAX_IN_FEATURES);
    for n_chunks in min_chunks..=64 {
        if !in_features.is_multiple_of(n_chunks) {
            continue;
        }
        let tile = in_features / n_chunks;
        if tile.is_multiple_of(AFFINE4_GROUP_SIZE) && tile <= AFFINE4_V3_MAX_IN_FEATURES {
            return Some((tile, n_chunks));
        }
    }
    None
}

/// 4-bit affine fixed group size. MLX writes scales+biases at this granularity.
pub const AFFINE4_GROUP_SIZE: usize = 64;

/// Long-lived GPU-resident affine 4-bit weight.
///
/// Holds packed nibbles (u32 words, 8 nibbles each) + bf16 scales + bf16 biases as
/// Metal buffers allocated once at load time. Forward passes reuse these buffers
/// instead of re-uploading every call. Layout matches the on-disk MLX format.
pub struct Affine4Weight {
    packed: Buffer,
    scales: Buffer,
    biases: Buffer,
    pub packed_offset: u64,
    pub scales_offset: u64,
    pub biases_offset: u64,
    pub out_features: usize,
    pub in_features: usize,
}

impl Affine4Weight {
    /// Upload packed nibbles + bf16 scales + bf16 biases to device.
    ///
    /// Shape contract:
    ///   - `packed.len() == out_features * in_features / 8`            (u32 words)
    ///   - `scales.len() == out_features * in_features / 64`           (bf16 as u16)
    ///   - `biases.len() == out_features * in_features / 64`           (bf16 as u16)
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
            in_features.is_multiple_of(AFFINE4_GROUP_SIZE),
            "in_features {in_features} must be a multiple of 64 (affine4 group size)"
        );
        let expected_packed = out_features * in_features / 8;
        anyhow::ensure!(
            packed.len() == expected_packed,
            "packed length {} != {}",
            packed.len(),
            expected_packed
        );
        let expected_groups = out_features * in_features / AFFINE4_GROUP_SIZE;
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
            packed_offset: 0,
            scales_offset: 0,
            biases_offset: 0,
            out_features,
            in_features,
        })
    }

    pub fn approx_bytes(&self) -> usize {
        let packed_bytes = self.out_features * self.in_features / 8 * 4;
        let scale_bias_bytes = self.out_features * self.in_features / AFFINE4_GROUP_SIZE * 2 * 2;
        packed_bytes + scale_bias_bytes
    }

    /// borrow the weight's GPU buffers for ICB hazard tracking
    /// or other low-level Metal operations. Returns (packed, scales, biases)
    /// in the order expected by `qmv_fast_*` kernel buffer indices 0..2.
    pub fn buffers(&self) -> (&Buffer, &Buffer, &Buffer) {
        (&self.packed, &self.scales, &self.biases)
    }

    pub fn packed_buffer_ref(&self) -> &Buffer {
        &self.packed
    }
}

/// Maximum `in_features` (in floats) that fits in 32 KB threadgroup memory for
/// the v3 cooperative kernel. `32 KB / 4 bytes = 8192`. Shapes with `in_features`
/// above this fall back to v1 dispatch.
pub const AFFINE4_V3_MAX_IN_FEATURES: usize = 8192;

/// Metal pipeline holder for affine 4-bit kernels.
pub struct Affine4Context {
    pub ctx: MetalContext,
    matvec_f32: ComputePipelineState,
    matmul_f32: ComputePipelineState,
    matmul_f32_v2: ComputePipelineState,
    matmul_f32_v3: ComputePipelineState,
    matmul_f32_v3_residual: ComputePipelineState,
    matmul_f32in_bf16out_v3: ComputePipelineState,
    matmul_bf16in_f32out_v3: ComputePipelineState,
    gate_up_silu_mul_v3: ComputePipelineState,
    gate_up_silu_mul_bf16out_v3: ComputePipelineState,
    matmul_f32_v3_rmsnorm: ComputePipelineState,
    matmul_f32_v3_tiled: ComputePipelineState,
    reduce_chunks_f32: ComputePipelineState,
    reduce_chunks_f32_residual: ComputePipelineState,
    qmv_fast: ComputePipelineState,
    qmv_fast_residual: ComputePipelineState,
    qmv_fast_rmsnorm: ComputePipelineState,
    qmv_fast_bf16in_bf16out: ComputePipelineState,
    qmv_fast_bf16in_bf16out_residual: ComputePipelineState,
    /// ICB-supporting pipelines (`supportIndirectCommandBuffers
    /// = true`). Identical kernel binaries to the non-ICB variants above
    /// but referenceable from `MTLIndirectComputeCommand::setComputePipelineState`
    /// without crashing. Used by ICB record/replay code path; non-ICB code
    /// continues using the non-ICB pipelines (keeps Phase 17 changes
    /// non-disruptive).
    qmv_fast_bf16in_bf16out_icb: ComputePipelineState,
    qmv_fast_bf16in_bf16out_residual_icb: ComputePipelineState,
    #[allow(dead_code)]
    library: Library,
}

impl Affine4Context {
    pub fn new() -> Result<Self> {
        let ctx = MetalContext::new()?;
        let options = crate::metal::new_compile_options();
        options.set_language_version(crate::metal::MTLLanguageVersion::Version3_1);
        options.set_fast_math_enabled(true);
        let library = ctx
            .device
            .new_library_with_source(SHADER_SRC, Some(options.as_ref()))
            .map_err(|e| anyhow::anyhow!("Affine4 shader compile error: {e}"))?;

        let matvec_func = library
            .get_function("affine4_matvec_f32", None)
            .map_err(|e| anyhow::anyhow!("kernel `affine4_matvec_f32` not found: {e}"))?;
        let matvec_f32 = ctx
            .device
            .new_compute_pipeline_state_with_function(&matvec_func)
            .map_err(|e| anyhow::anyhow!("pipeline `affine4_matvec_f32` failed: {e}"))?;

        let matmul_func = library
            .get_function("affine4_matmul_f32", None)
            .map_err(|e| anyhow::anyhow!("kernel `affine4_matmul_f32` not found: {e}"))?;
        let matmul_f32 = ctx
            .device
            .new_compute_pipeline_state_with_function(&matmul_func)
            .map_err(|e| anyhow::anyhow!("pipeline `affine4_matmul_f32` failed: {e}"))?;

        let matmul_v2_func = library
            .get_function("affine4_matmul_f32_v2", None)
            .map_err(|e| anyhow::anyhow!("kernel `affine4_matmul_f32_v2` not found: {e}"))?;
        let matmul_f32_v2 = ctx
            .device
            .new_compute_pipeline_state_with_function(&matmul_v2_func)
            .map_err(|e| anyhow::anyhow!("pipeline `affine4_matmul_f32_v2` failed: {e}"))?;

        let matmul_v3_func = library
            .get_function("affine4_matmul_f32_v3", None)
            .map_err(|e| anyhow::anyhow!("kernel `affine4_matmul_f32_v3` not found: {e}"))?;
        let matmul_f32_v3 = ctx
            .device
            .new_compute_pipeline_state_with_function(&matmul_v3_func)
            .map_err(|e| anyhow::anyhow!("pipeline `affine4_matmul_f32_v3` failed: {e}"))?;

        let residual_func = library
            .get_function("affine4_matmul_f32_v3_residual", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `affine4_matmul_f32_v3_residual` not found: {e}")
            })?;
        let matmul_f32_v3_residual = ctx
            .device
            .new_compute_pipeline_state_with_function(&residual_func)
            .map_err(|e| {
                anyhow::anyhow!("pipeline `affine4_matmul_f32_v3_residual` failed: {e}")
            })?;

        let bf16out_func = library
            .get_function("affine4_matmul_f32in_bf16out_v3", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `affine4_matmul_f32in_bf16out_v3` not found: {e}")
            })?;
        let matmul_f32in_bf16out_v3 = ctx
            .device
            .new_compute_pipeline_state_with_function(&bf16out_func)
            .map_err(|e| {
                anyhow::anyhow!("pipeline `affine4_matmul_f32in_bf16out_v3` failed: {e}")
            })?;

        let bf16in_func = library
            .get_function("affine4_matmul_bf16in_f32out_v3", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `affine4_matmul_bf16in_f32out_v3` not found: {e}")
            })?;
        let matmul_bf16in_f32out_v3 = ctx
            .device
            .new_compute_pipeline_state_with_function(&bf16in_func)
            .map_err(|e| {
                anyhow::anyhow!("pipeline `affine4_matmul_bf16in_f32out_v3` failed: {e}")
            })?;

        let gate_up_func = library
            .get_function("affine4_gate_up_silu_mul_f32_v3", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `affine4_gate_up_silu_mul_f32_v3` not found: {e}")
            })?;
        let gate_up_silu_mul_v3 = ctx
            .device
            .new_compute_pipeline_state_with_function(&gate_up_func)
            .map_err(|e| {
                anyhow::anyhow!("pipeline `affine4_gate_up_silu_mul_f32_v3` failed: {e}")
            })?;

        let gate_up_bf16out_func = library
            .get_function("affine4_gate_up_silu_mul_f32in_bf16out_v3", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `affine4_gate_up_silu_mul_f32in_bf16out_v3` not found: {e}")
            })?;
        let gate_up_silu_mul_bf16out_v3 = ctx
            .device
            .new_compute_pipeline_state_with_function(&gate_up_bf16out_func)
            .map_err(|e| {
                anyhow::anyhow!("pipeline `affine4_gate_up_silu_mul_f32in_bf16out_v3` failed: {e}")
            })?;

        let rmsnorm_func = library
            .get_function("affine4_matmul_f32_v3_rmsnorm", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `affine4_matmul_f32_v3_rmsnorm` not found: {e}")
            })?;
        let matmul_f32_v3_rmsnorm = ctx
            .device
            .new_compute_pipeline_state_with_function(&rmsnorm_func)
            .map_err(|e| anyhow::anyhow!("pipeline `affine4_matmul_f32_v3_rmsnorm` failed: {e}"))?;

        let tiled_func = library
            .get_function("affine4_matmul_f32_v3_tiled", None)
            .map_err(|e| anyhow::anyhow!("kernel `affine4_matmul_f32_v3_tiled` not found: {e}"))?;
        let matmul_f32_v3_tiled = ctx
            .device
            .new_compute_pipeline_state_with_function(&tiled_func)
            .map_err(|e| anyhow::anyhow!("pipeline `affine4_matmul_f32_v3_tiled` failed: {e}"))?;

        let reduce_func = library
            .get_function("affine4_reduce_chunks_f32", None)
            .map_err(|e| anyhow::anyhow!("kernel `affine4_reduce_chunks_f32` not found: {e}"))?;
        let reduce_chunks_f32 = ctx
            .device
            .new_compute_pipeline_state_with_function(&reduce_func)
            .map_err(|e| anyhow::anyhow!("pipeline `affine4_reduce_chunks_f32` failed: {e}"))?;

        let reduce_res_func = library
            .get_function("affine4_reduce_chunks_f32_residual", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `affine4_reduce_chunks_f32_residual` not found: {e}")
            })?;
        let reduce_chunks_f32_residual = ctx
            .device
            .new_compute_pipeline_state_with_function(&reduce_res_func)
            .map_err(|e| {
                anyhow::anyhow!("pipeline `affine4_reduce_chunks_f32_residual` failed: {e}")
            })?;

        let qmv_fast_func = library
            .get_function("affine4_qmv_fast", None)
            .map_err(|e| anyhow::anyhow!("kernel `affine4_qmv_fast` not found: {e}"))?;
        let qmv_fast = ctx
            .device
            .new_compute_pipeline_state_with_function(&qmv_fast_func)
            .map_err(|e| anyhow::anyhow!("pipeline `affine4_qmv_fast` failed: {e}"))?;

        let qmv_fast_res_func = library
            .get_function("affine4_qmv_fast_residual", None)
            .map_err(|e| anyhow::anyhow!("kernel `affine4_qmv_fast_residual` not found: {e}"))?;
        let qmv_fast_residual = ctx
            .device
            .new_compute_pipeline_state_with_function(&qmv_fast_res_func)
            .map_err(|e| anyhow::anyhow!("pipeline `affine4_qmv_fast_residual` failed: {e}"))?;

        let qmv_fast_rms_func = library
            .get_function("affine4_qmv_fast_rmsnorm", None)
            .map_err(|e| anyhow::anyhow!("kernel `affine4_qmv_fast_rmsnorm` not found: {e}"))?;
        let qmv_fast_rmsnorm = ctx
            .device
            .new_compute_pipeline_state_with_function(&qmv_fast_rms_func)
            .map_err(|e| anyhow::anyhow!("pipeline `affine4_qmv_fast_rmsnorm` failed: {e}"))?;

        let qmv_fast_bf16_func = library
            .get_function("affine4_qmv_fast_bf16in_bf16out", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `affine4_qmv_fast_bf16in_bf16out` not found: {e}")
            })?;
        let qmv_fast_bf16in_bf16out = ctx
            .device
            .new_compute_pipeline_state_with_function(&qmv_fast_bf16_func)
            .map_err(|e| {
                anyhow::anyhow!("pipeline `affine4_qmv_fast_bf16in_bf16out` failed: {e}")
            })?;

        let qmv_fast_bf16_residual_func = library
            .get_function("affine4_qmv_fast_bf16in_bf16out_residual", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `affine4_qmv_fast_bf16in_bf16out_residual` not found: {e}")
            })?;
        let qmv_fast_bf16in_bf16out_residual = ctx
            .device
            .new_compute_pipeline_state_with_function(&qmv_fast_bf16_residual_func)
            .map_err(|e| {
                anyhow::anyhow!("pipeline `affine4_qmv_fast_bf16in_bf16out_residual` failed: {e}")
            })?;

        // ICB-supporting variants of the two bf16 fast paths
        // (qmv_fast_bf16in_bf16out + ..._residual). Built via the new
        // `_for_icb` Device API so they can be assigned to
        // MTLIndirectComputeCommand without crash. Same kernel binary, only
        // the supportIndirectCommandBuffers flag differs.
        let qmv_fast_bf16in_bf16out_icb = ctx
            .device
            .new_compute_pipeline_state_with_function_for_icb(&qmv_fast_bf16_func)
            .map_err(|e| {
                anyhow::anyhow!("pipeline `affine4_qmv_fast_bf16in_bf16out` (ICB) failed: {e}")
            })?;
        let qmv_fast_bf16in_bf16out_residual_icb = ctx
            .device
            .new_compute_pipeline_state_with_function_for_icb(&qmv_fast_bf16_residual_func)
            .map_err(|e| {
                anyhow::anyhow!(
                    "pipeline `affine4_qmv_fast_bf16in_bf16out_residual` (ICB) failed: {e}"
                )
            })?;

        Ok(Self {
            ctx,
            matvec_f32,
            matmul_f32,
            matmul_f32_v2,
            matmul_f32_v3,
            matmul_f32_v3_residual,
            matmul_f32in_bf16out_v3,
            matmul_bf16in_f32out_v3,
            gate_up_silu_mul_v3,
            gate_up_silu_mul_bf16out_v3,
            matmul_f32_v3_rmsnorm,
            matmul_f32_v3_tiled,
            reduce_chunks_f32,
            reduce_chunks_f32_residual,
            qmv_fast,
            qmv_fast_residual,
            qmv_fast_rmsnorm,
            qmv_fast_bf16in_bf16out,
            qmv_fast_bf16in_bf16out_residual,
            qmv_fast_bf16in_bf16out_icb,
            qmv_fast_bf16in_bf16out_residual_icb,
            library,
        })
    }

    /// Encode `y = x @ W^T` into the supplied compute encoder. Both `x_buf` and
    /// `y_buf` are caller-owned; this method only encodes — no commit, no wait.
    pub fn encode_matmul_dispatch(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Affine4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) {
        // Kernel selection (decode hot path):
        //   in ≤ 8192   → v3 cooperative simdgroup + TG memory cache (fastest)
        //   in > 8192   → v2 vectorized uint4/float4 loads (handles 27B-Dense down_proj
        //                 with in=22528 where v3's 32 KB TG mem doesn't fit)
        //   v1 (scalar) → only on `LUMEN_AFFINE4_V1_ONLY=1` opt-out
        let force_v1 = std::env::var("LUMEN_AFFINE4_V1_ONLY")
            .map(|v| v == "1")
            .unwrap_or(false);
        let use_v3 = !force_v1 && weight.in_features <= AFFINE4_V3_MAX_IN_FEATURES;
        let use_v2 = !force_v1 && !use_v3;

        if use_v3 {
            encoder.set_compute_pipeline_state(&self.matmul_f32_v3);
        } else if use_v2 {
            encoder.set_compute_pipeline_state(&self.matmul_f32_v2);
        } else {
            encoder.set_compute_pipeline_state(&self.matmul_f32);
        }
        encoder.set_buffer(0, Some(&weight.packed), weight.packed_offset as usize);
        encoder.set_buffer(1, Some(&weight.scales), weight.scales_offset as usize);
        encoder.set_buffer(2, Some(&weight.biases), weight.biases_offset as usize);
        encoder.set_buffer(3, Some(x_buf), x_offset as usize);
        encoder.set_buffer(4, Some(y_buf), y_offset as usize);
        let dims = Affine4Dims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
        };
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<Affine4Dims>(),
            &dims as *const _ as *const _,
        );
        let batch_u32 = batch as u32;
        encoder.set_bytes_directly(6, 4, &batch_u32 as *const _ as *const _);

        if use_v3 {
            // v3: 256 threads/threadgroup, 8 rows per threadgroup, threadgroup
            // memory caches the staged x[b, :] activation.
            const ROWS_PER_TG: usize = 8;
            const THREADS_PER_TG: usize = 256;
            let tg_mem_bytes = weight.in_features * 4;
            encoder.set_threadgroup_memory_length(0, tg_mem_bytes);
            let n_groups_x = weight.out_features.div_ceil(ROWS_PER_TG);
            let grid = MTLSize {
                width: n_groups_x,
                height: batch,
                depth: 1,
            };
            let tg = MTLSize {
                width: THREADS_PER_TG,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(grid, tg);
        } else {
            // v1: 1 thread per (row, batch_elem), flat dispatch.
            let max_threads = self.matmul_f32.max_total_threads_per_threadgroup();
            let threads_per_tg = max_threads.min(256);
            let grid = MTLSize {
                width: weight.out_features,
                height: batch,
                depth: 1,
            };
            let tg = MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_threads(grid, tg);
        }
    }

    /// True if the MLX-pattern `qmv_fast` decode kernel can handle this shape.
    /// Constraints: `in_features % 512 == 0` (block_size) AND
    /// `out_features % 8 == 0` (NSG=2 × RPS=4 = rows per TG).
    /// All 27B-Dense projection shapes (5120, 22528 in × 5120/22528 out)
    /// satisfy both. Other shapes fall back to v3 / v3_tiled / v2.
    pub fn qmv_fast_supports(in_features: usize, out_features: usize) -> bool {
        in_features.is_multiple_of(512) && out_features.is_multiple_of(8)
    }

    /// Encode a `qmv_fast` dispatch into the supplied encoder.
    /// Caller must verify `qmv_fast_supports(in, out)` before calling.
    /// Hot path is batch=1 decode; the kernel itself supports any batch.
    pub fn encode_qmv_fast_dispatch(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Affine4Weight,
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
        encoder.set_compute_pipeline_state(&self.qmv_fast);
        encoder.set_buffer(0, Some(&weight.packed), weight.packed_offset as usize);
        encoder.set_buffer(1, Some(&weight.scales), weight.scales_offset as usize);
        encoder.set_buffer(2, Some(&weight.biases), weight.biases_offset as usize);
        encoder.set_buffer(3, Some(x_buf), x_offset as usize);
        encoder.set_buffer(4, Some(y_buf), y_offset as usize);
        let dims = Affine4Dims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
        };
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<Affine4Dims>(),
            &dims as *const _ as *const _,
        );
        let batch_u32 = batch as u32;
        encoder.set_bytes_directly(6, 4, &batch_u32 as *const _ as *const _);

        // Grid: (batch, out / (NSG*RPS), 1). TG: 64 threads (NSG=2 simdgroups × 32 lanes).
        const NSG: usize = 2;
        const RPS: usize = 4;
        const THREADS_PER_TG: usize = 64;
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

    /// bf16-in/bf16-out variant of `encode_qmv_fast_dispatch`. Activation read
    /// and result write narrow to bf16; internal accumulation stays in f32.
    /// Caller owns dtype contract — `x_buf` must contain bf16 values, `y_buf`
    /// receives bf16 values (each element is 2 bytes).
    pub fn encode_qmv_fast_bf16in_bf16out_dispatch(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Affine4Weight,
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
        encoder.set_compute_pipeline_state(&self.qmv_fast_bf16in_bf16out);
        encoder.set_buffer(0, Some(&weight.packed), weight.packed_offset as usize);
        encoder.set_buffer(1, Some(&weight.scales), weight.scales_offset as usize);
        encoder.set_buffer(2, Some(&weight.biases), weight.biases_offset as usize);
        encoder.set_buffer(3, Some(x_buf), x_offset as usize);
        encoder.set_buffer(4, Some(y_buf), y_offset as usize);
        let dims = Affine4Dims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
        };
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<Affine4Dims>(),
            &dims as *const _ as *const _,
        );
        let batch_u32 = batch as u32;
        encoder.set_bytes_directly(6, 4, &batch_u32 as *const _ as *const _);

        const NSG: usize = 2;
        const RPS: usize = 4;
        const THREADS_PER_TG: usize = 64;
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

    /// Same as `encode_qmv_fast_dispatch` but the kernel also adds `residual`
    /// to each output element (saves one downstream broadcast_add per call).
    pub fn encode_qmv_fast_residual_dispatch(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Affine4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        residual_buf: &Buffer,
        residual_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) {
        debug_assert!(Self::qmv_fast_supports(
            weight.in_features,
            weight.out_features
        ));
        encoder.set_compute_pipeline_state(&self.qmv_fast_residual);
        encoder.set_buffer(0, Some(&weight.packed), weight.packed_offset as usize);
        encoder.set_buffer(1, Some(&weight.scales), weight.scales_offset as usize);
        encoder.set_buffer(2, Some(&weight.biases), weight.biases_offset as usize);
        encoder.set_buffer(3, Some(x_buf), x_offset as usize);
        encoder.set_buffer(4, Some(residual_buf), residual_offset as usize);
        encoder.set_buffer(5, Some(y_buf), y_offset as usize);
        let dims = Affine4Dims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
        };
        encoder.set_bytes_directly(
            6,
            std::mem::size_of::<Affine4Dims>(),
            &dims as *const _ as *const _,
        );
        let batch_u32 = batch as u32;
        encoder.set_bytes_directly(7, 4, &batch_u32 as *const _ as *const _);

        const NSG: usize = 2;
        const RPS: usize = 4;
        const THREADS_PER_TG: usize = 64;
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

    /// bf16-in/bf16-out variant of `encode_qmv_fast_residual_dispatch`.
    /// Mirrors the f32 fused-residual kernel on the bf16 chain — closes the
    /// B.10 σ-NEGATIVE root cause where the f32 fused fast-path could not
    /// fire under bf16 carrier. Activation, residual, and output are bf16;
    /// internal accumulation stays in f32.
    pub fn encode_qmv_fast_bf16in_bf16out_residual_dispatch(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Affine4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        residual_buf: &Buffer,
        residual_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) {
        debug_assert!(Self::qmv_fast_supports(
            weight.in_features,
            weight.out_features
        ));
        encoder.set_compute_pipeline_state(&self.qmv_fast_bf16in_bf16out_residual);
        encoder.set_buffer(0, Some(&weight.packed), weight.packed_offset as usize);
        encoder.set_buffer(1, Some(&weight.scales), weight.scales_offset as usize);
        encoder.set_buffer(2, Some(&weight.biases), weight.biases_offset as usize);
        encoder.set_buffer(3, Some(x_buf), x_offset as usize);
        encoder.set_buffer(4, Some(residual_buf), residual_offset as usize);
        encoder.set_buffer(5, Some(y_buf), y_offset as usize);
        let dims = Affine4Dims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
        };
        encoder.set_bytes_directly(
            6,
            std::mem::size_of::<Affine4Dims>(),
            &dims as *const _ as *const _,
        );
        let batch_u32 = batch as u32;
        encoder.set_bytes_directly(7, 4, &batch_u32 as *const _ as *const _);

        const NSG: usize = 2;
        const RPS: usize = 4;
        const THREADS_PER_TG: usize = 64;
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

    /// record a single
    /// `affine4_qmv_fast_bf16in_bf16out_residual` dispatch into the supplied
    /// `IndirectCommandBuffer` at `slot`.
    ///
    /// `dims_buf` and `batch_buf` are pre-allocated GPU buffers carrying
    /// `Affine4Dims` and `batch` (u32) respectively. ICB record cannot use
    /// `set_bytes_directly`, so the caller must materialize these constants
    /// in real buffers (one per shape — typically all decoder layers share
    /// the same shape, so a single `dims_buf` per shape suffices).
    ///
    /// To replay: encoder.use_buffer_for_icb(...) all 7 buffers, then
    /// encoder.execute_commands_in_buffer(icb, count).
    pub fn record_qmv_fast_bf16in_bf16out_residual_icb(
        &self,
        icb: &crate::metal::IndirectCommandBuffer,
        slot: usize,
        weight: &Affine4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        residual_buf: &Buffer,
        residual_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        dims_buf: &Buffer,
        batch_buf: &Buffer,
        batch: usize,
    ) {
        debug_assert!(Self::qmv_fast_supports(
            weight.in_features,
            weight.out_features
        ));

        const NSG: usize = 2;
        const RPS: usize = 4;
        const THREADS_PER_TG: usize = 64;
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

        icb.record_compute(
            slot,
            &self.qmv_fast_bf16in_bf16out_residual_icb,
            &[
                (&weight.packed, weight.packed_offset as usize, 0),
                (&weight.scales, weight.scales_offset as usize, 1),
                (&weight.biases, weight.biases_offset as usize, 2),
                (x_buf, x_offset as usize, 3),
                (residual_buf, residual_offset as usize, 4),
                (y_buf, y_offset as usize, 5),
                (dims_buf, 0, 6),
                (batch_buf, 0, 7),
            ],
            grid,
            tg,
        );
    }

    /// record a single `affine4_qmv_fast_bf16in_bf16out`
    /// (no-residual) dispatch into the supplied ICB at `slot`. Same
    /// requirements as the residual variant; one fewer buffer (no residual).
    pub fn record_qmv_fast_bf16in_bf16out_icb(
        &self,
        icb: &crate::metal::IndirectCommandBuffer,
        slot: usize,
        weight: &Affine4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        dims_buf: &Buffer,
        batch_buf: &Buffer,
        batch: usize,
    ) {
        debug_assert!(Self::qmv_fast_supports(
            weight.in_features,
            weight.out_features
        ));

        const NSG: usize = 2;
        const RPS: usize = 4;
        const THREADS_PER_TG: usize = 64;
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

        icb.record_compute(
            slot,
            &self.qmv_fast_bf16in_bf16out_icb,
            &[
                (&weight.packed, weight.packed_offset as usize, 0),
                (&weight.scales, weight.scales_offset as usize, 1),
                (&weight.biases, weight.biases_offset as usize, 2),
                (x_buf, x_offset as usize, 3),
                (y_buf, y_offset as usize, 4),
                (dims_buf, 0, 5),
                (batch_buf, 0, 6),
            ],
            grid,
            tg,
        );
    }

    /// Encode `qmv_fast` with fused input RmsNorm. Reads RAW (un-normalized)
    /// `x` and the layernorm weight; the kernel computes `inv_rms` cooperatively
    /// then folds the normalization into the matmul. Saves the entire Candle
    /// RmsNorm dispatch chain per call (5+ dispatches → 1).
    pub fn encode_qmv_fast_rmsnorm_dispatch(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Affine4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        rms_weight_buf: &Buffer,
        rms_weight_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        rms_eps: f32,
        batch: usize,
    ) {
        debug_assert!(Self::qmv_fast_supports(
            weight.in_features,
            weight.out_features
        ));
        encoder.set_compute_pipeline_state(&self.qmv_fast_rmsnorm);
        encoder.set_buffer(0, Some(&weight.packed), weight.packed_offset as usize);
        encoder.set_buffer(1, Some(&weight.scales), weight.scales_offset as usize);
        encoder.set_buffer(2, Some(&weight.biases), weight.biases_offset as usize);
        encoder.set_buffer(3, Some(x_buf), x_offset as usize);
        encoder.set_buffer(4, Some(rms_weight_buf), rms_weight_offset as usize);
        encoder.set_buffer(5, Some(y_buf), y_offset as usize);
        let dims = Affine4MatmulRmsnormDims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
            rms_eps,
        };
        encoder.set_bytes_directly(
            6,
            std::mem::size_of::<Affine4MatmulRmsnormDims>(),
            &dims as *const _ as *const _,
        );
        let batch_u32 = batch as u32;
        encoder.set_bytes_directly(7, 4, &batch_u32 as *const _ as *const _);

        // TG memory: 2 floats (one per simdgroup) for cross-simdgroup partial sum.
        const NSG: usize = 2;
        const RPS: usize = 4;
        const THREADS_PER_TG: usize = 64;
        encoder.set_threadgroup_memory_length(0, NSG * 4);
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

    /// Bytes needed for the v3-tiled scratch buffer.
    /// Layout: `[n_chunks, batch, out_features]` f32.
    pub fn tiled_scratch_bytes(out_features: usize, batch: usize, n_chunks: usize) -> usize {
        n_chunks * batch * out_features * std::mem::size_of::<f32>()
    }

    /// Encode the v3-tiled matmul + reduction into a single encoder.
    ///
    /// Used when `in_features > AFFINE4_V3_MAX_IN_FEATURES` (e.g. 27B-Dense down_proj
    /// `in=22528`). Splits the in-axis into `n_chunks` equal tiles of `tile_in`
    /// floats each so each threadgroup can stage one tile in TG memory.
    /// Per-chunk partial sums go to `scratch` (caller-allocated, sized by
    /// `tiled_scratch_bytes`); the reduction kernel sums over the chunk axis to `y`.
    ///
    /// Caller invariants:
    /// - `tile_in * n_chunks == weight.in_features`
    /// - `tile_in <= AFFINE4_V3_MAX_IN_FEATURES`
    /// - `tile_in % AFFINE4_GROUP_SIZE == 0`
    /// - `scratch` length ≥ `tiled_scratch_bytes(weight.out_features, batch, n_chunks)` bytes
    pub fn encode_matmul_tiled_dispatch(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Affine4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        scratch_buf: &Buffer,
        y_buf: &Buffer,
        y_offset: u64,
        tile_in: usize,
        n_chunks: usize,
        batch: usize,
    ) {
        debug_assert_eq!(tile_in * n_chunks, weight.in_features);
        debug_assert!(tile_in <= AFFINE4_V3_MAX_IN_FEATURES);
        debug_assert!(tile_in.is_multiple_of(AFFINE4_GROUP_SIZE));

        // tiled matmul → scratch[n_chunks, batch, out].
        encoder.set_compute_pipeline_state(&self.matmul_f32_v3_tiled);
        encoder.set_buffer(0, Some(&weight.packed), weight.packed_offset as usize);
        encoder.set_buffer(1, Some(&weight.scales), weight.scales_offset as usize);
        encoder.set_buffer(2, Some(&weight.biases), weight.biases_offset as usize);
        encoder.set_buffer(3, Some(x_buf), x_offset as usize);
        encoder.set_buffer(4, Some(scratch_buf), 0);
        let tdims = Affine4TiledDims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
            tile_in: tile_in as u32,
            n_chunks: n_chunks as u32,
        };
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<Affine4TiledDims>(),
            &tdims as *const _ as *const _,
        );
        let batch_u32 = batch as u32;
        encoder.set_bytes_directly(6, 4, &batch_u32 as *const _ as *const _);

        const ROWS_PER_TG: usize = 8;
        const THREADS_PER_TG: usize = 256;
        let tg_mem_bytes = tile_in * 4;
        encoder.set_threadgroup_memory_length(0, tg_mem_bytes);
        let n_groups_x = weight.out_features.div_ceil(ROWS_PER_TG);
        let grid = MTLSize {
            width: n_groups_x,
            height: batch,
            depth: n_chunks,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid, tg);

        // reduce scratch over chunk axis → y[batch, out].
        encoder.set_compute_pipeline_state(&self.reduce_chunks_f32);
        encoder.set_buffer(0, Some(scratch_buf), 0);
        encoder.set_buffer(1, Some(y_buf), y_offset as usize);
        let rdims = Affine4ReduceDims {
            out_features: weight.out_features as u32,
            n_chunks: n_chunks as u32,
        };
        encoder.set_bytes_directly(
            2,
            std::mem::size_of::<Affine4ReduceDims>(),
            &rdims as *const _ as *const _,
        );
        encoder.set_bytes_directly(3, 4, &batch_u32 as *const _ as *const _);
        let max_threads = self.reduce_chunks_f32.max_total_threads_per_threadgroup();
        let threads_per_tg = max_threads.min(256);
        let red_grid = MTLSize {
            width: weight.out_features,
            height: batch,
            depth: 1,
        };
        let red_tg = MTLSize {
            width: threads_per_tg,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(red_grid, red_tg);
    }

    /// Same as `encode_matmul_tiled_dispatch` but the final reduction also adds
    /// `residual[batch, out]` to each output element — saving one downstream
    /// `broadcast_add` dispatch when the matmul output feeds a residual addition.
    /// Identical scratch buffer / tile contract.
    pub fn encode_matmul_tiled_residual_dispatch(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Affine4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        scratch_buf: &Buffer,
        residual_buf: &Buffer,
        residual_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        tile_in: usize,
        n_chunks: usize,
        batch: usize,
    ) {
        debug_assert_eq!(tile_in * n_chunks, weight.in_features);
        debug_assert!(tile_in <= AFFINE4_V3_MAX_IN_FEATURES);
        debug_assert!(tile_in.is_multiple_of(AFFINE4_GROUP_SIZE));

        // identical to non-residual path — partial sums
        // accumulate into scratch[n_chunks, batch, out].
        encoder.set_compute_pipeline_state(&self.matmul_f32_v3_tiled);
        encoder.set_buffer(0, Some(&weight.packed), weight.packed_offset as usize);
        encoder.set_buffer(1, Some(&weight.scales), weight.scales_offset as usize);
        encoder.set_buffer(2, Some(&weight.biases), weight.biases_offset as usize);
        encoder.set_buffer(3, Some(x_buf), x_offset as usize);
        encoder.set_buffer(4, Some(scratch_buf), 0);
        let tdims = Affine4TiledDims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
            tile_in: tile_in as u32,
            n_chunks: n_chunks as u32,
        };
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<Affine4TiledDims>(),
            &tdims as *const _ as *const _,
        );
        let batch_u32 = batch as u32;
        encoder.set_bytes_directly(6, 4, &batch_u32 as *const _ as *const _);

        const ROWS_PER_TG: usize = 8;
        const THREADS_PER_TG: usize = 256;
        let tg_mem_bytes = tile_in * 4;
        encoder.set_threadgroup_memory_length(0, tg_mem_bytes);
        let n_groups_x = weight.out_features.div_ceil(ROWS_PER_TG);
        let grid = MTLSize {
            width: n_groups_x,
            height: batch,
            depth: n_chunks,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid, tg);

        // one fused dispatch.
        encoder.set_compute_pipeline_state(&self.reduce_chunks_f32_residual);
        encoder.set_buffer(0, Some(scratch_buf), 0);
        encoder.set_buffer(1, Some(residual_buf), residual_offset as usize);
        encoder.set_buffer(2, Some(y_buf), y_offset as usize);
        let rdims = Affine4ReduceDims {
            out_features: weight.out_features as u32,
            n_chunks: n_chunks as u32,
        };
        encoder.set_bytes_directly(
            3,
            std::mem::size_of::<Affine4ReduceDims>(),
            &rdims as *const _ as *const _,
        );
        encoder.set_bytes_directly(4, 4, &batch_u32 as *const _ as *const _);
        let max_threads = self
            .reduce_chunks_f32_residual
            .max_total_threads_per_threadgroup();
        let threads_per_tg = max_threads.min(256);
        let red_grid = MTLSize {
            width: weight.out_features,
            height: batch,
            depth: 1,
        };
        let red_tg = MTLSize {
            width: threads_per_tg,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(red_grid, red_tg);
    }

    /// Own-queue equivalent of `encode_matmul_tiled_dispatch`. Allocates scratch,
    /// submits on the context's own queue, blocks for completion. Used as the
    /// non-Candle-queue / fallback path.
    pub fn matmul_tiled_zero_copy(
        &self,
        weight: &Affine4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        tile_in: usize,
        n_chunks: usize,
        batch: usize,
    ) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let scratch_bytes = Self::tiled_scratch_bytes(weight.out_features, batch, n_chunks) as u64;
        let scratch = self.ctx.buffer_zeroed(scratch_bytes);

        let encoder = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        encoder.set_label("lumen:affine4_matmul_tiled");
        self.encode_matmul_tiled_dispatch(
            encoder.as_ref(),
            weight,
            x_buf,
            x_offset,
            &scratch,
            y_buf,
            y_offset,
            tile_in,
            n_chunks,
            batch,
        );
        drop(encoder);
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");
        Ok(())
    }

    /// Common 6-binding setup for v3-family kernels: packed/scales/biases/x/y/dims.
    /// `extra_buf_idx` is the next free buffer slot for variant-specific bindings.
    fn encode_v3_buffers(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Affine4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        x_buf_idx: usize,
        y_buf_idx: usize,
        dims_buf_idx: usize,
    ) {
        encoder.set_buffer(0, Some(&weight.packed), weight.packed_offset as usize);
        encoder.set_buffer(1, Some(&weight.scales), weight.scales_offset as usize);
        encoder.set_buffer(2, Some(&weight.biases), weight.biases_offset as usize);
        encoder.set_buffer(x_buf_idx, Some(x_buf), x_offset as usize);
        encoder.set_buffer(y_buf_idx, Some(y_buf), y_offset as usize);
        let dims = Affine4Dims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
        };
        encoder.set_bytes_directly(
            dims_buf_idx,
            std::mem::size_of::<Affine4Dims>(),
            &dims as *const _ as *const _,
        );
    }

    fn dispatch_v3_grid(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Affine4Weight,
        batch: usize,
    ) {
        const ROWS_PER_TG: usize = 8;
        const THREADS_PER_TG: usize = 256;
        let tg_mem_bytes = weight.in_features * 4;
        encoder.set_threadgroup_memory_length(0, tg_mem_bytes);
        let n_groups_x = weight.out_features.div_ceil(ROWS_PER_TG);
        let grid = MTLSize {
            width: n_groups_x,
            height: batch,
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid, tg);
    }

    /// Encode v3 fused-residual dispatch: `y = (x @ W^T) + residual`.
    /// Slots: 0..2 weight, 3 x, 4 residual, 5 y, 6 dims, 7 batch.
    pub fn encode_matmul_residual_dispatch(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Affine4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        residual_buf: &Buffer,
        residual_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.matmul_f32_v3_residual);
        encoder.set_buffer(0, Some(&weight.packed), weight.packed_offset as usize);
        encoder.set_buffer(1, Some(&weight.scales), weight.scales_offset as usize);
        encoder.set_buffer(2, Some(&weight.biases), weight.biases_offset as usize);
        encoder.set_buffer(3, Some(x_buf), x_offset as usize);
        encoder.set_buffer(4, Some(residual_buf), residual_offset as usize);
        encoder.set_buffer(5, Some(y_buf), y_offset as usize);
        let dims = Affine4Dims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
        };
        encoder.set_bytes_directly(
            6,
            std::mem::size_of::<Affine4Dims>(),
            &dims as *const _ as *const _,
        );
        let batch_u32 = batch as u32;
        encoder.set_bytes_directly(7, 4, &batch_u32 as *const _ as *const _);
        self.dispatch_v3_grid(encoder, weight, batch);
    }

    /// Encode v3 f32-in / bf16-out dispatch.
    pub fn encode_matmul_bf16out_dispatch(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Affine4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.matmul_f32in_bf16out_v3);
        self.encode_v3_buffers(encoder, weight, x_buf, x_offset, y_buf, y_offset, 3, 4, 5);
        let batch_u32 = batch as u32;
        encoder.set_bytes_directly(6, 4, &batch_u32 as *const _ as *const _);
        self.dispatch_v3_grid(encoder, weight, batch);
    }

    /// Encode v3 bf16-in / f32-out dispatch.
    pub fn encode_matmul_bf16in_dispatch(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Affine4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.matmul_bf16in_f32out_v3);
        self.encode_v3_buffers(encoder, weight, x_buf, x_offset, y_buf, y_offset, 3, 4, 5);
        let batch_u32 = batch as u32;
        encoder.set_bytes_directly(6, 4, &batch_u32 as *const _ as *const _);
        self.dispatch_v3_grid(encoder, weight, batch);
    }

    /// Encode the fused RmsNorm + matmul kernel: `y = (x_raw * inv_rms * rms_w) @ W^T`.
    /// Folds an entire layer's input RmsNorm into the v3 matmul, saving 1 dispatch
    /// per call. Caller is responsible for `in_features ≤ AFFINE4_V3_MAX_IN_FEATURES`
    /// (32 KB TG memory budget).
    pub fn encode_matmul_rmsnorm_dispatch(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Affine4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        rms_weight_buf: &Buffer,
        rms_weight_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        rms_eps: f32,
        batch: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.matmul_f32_v3_rmsnorm);
        encoder.set_buffer(0, Some(&weight.packed), weight.packed_offset as usize);
        encoder.set_buffer(1, Some(&weight.scales), weight.scales_offset as usize);
        encoder.set_buffer(2, Some(&weight.biases), weight.biases_offset as usize);
        encoder.set_buffer(3, Some(x_buf), x_offset as usize);
        encoder.set_buffer(4, Some(rms_weight_buf), rms_weight_offset as usize);
        encoder.set_buffer(5, Some(y_buf), y_offset as usize);
        let dims = Affine4MatmulRmsnormDims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
            rms_eps,
        };
        encoder.set_bytes_directly(
            6,
            std::mem::size_of::<Affine4MatmulRmsnormDims>(),
            &dims as *const _ as *const _,
        );
        let batch_u32 = batch as u32;
        encoder.set_bytes_directly(7, 4, &batch_u32 as *const _ as *const _);

        const ROWS_PER_TG: usize = 8;
        const THREADS_PER_TG: usize = 256;
        let x_shared_bytes = weight.in_features * 4;
        // reduce_buf: 8 partial sums (one per simdgroup), 4 bytes each = 32 bytes.
        let reduce_buf_bytes = 8 * 4;
        encoder.set_threadgroup_memory_length(0, x_shared_bytes);
        encoder.set_threadgroup_memory_length(1, reduce_buf_bytes);
        let n_groups_x = weight.out_features.div_ceil(ROWS_PER_TG);
        let grid = MTLSize {
            width: n_groups_x,
            height: batch,
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid, tg);
    }

    /// Own-queue equivalent of `encode_matmul_rmsnorm_dispatch`. Submits on the
    /// context's own queue + waits for completion. Used as a debug fallback when
    /// Candle-queue dispatch interacts badly with fused-kernel TG memory.
    pub fn matmul_rmsnorm_zero_copy(
        &self,
        weight: &Affine4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        rms_weight_buf: &Buffer,
        rms_weight_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        rms_eps: f32,
        batch: usize,
    ) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let encoder = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        encoder.set_label("lumen:affine4_matmul_rmsnorm");
        self.encode_matmul_rmsnorm_dispatch(
            encoder.as_ref(),
            weight,
            x_buf,
            x_offset,
            rms_weight_buf,
            rms_weight_offset,
            y_buf,
            y_offset,
            rms_eps,
            batch,
        );
        drop(encoder);
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");
        Ok(())
    }

    /// Encode the fused gate+up SiLU mul kernel.
    /// `weight.out_features` must equal `2 * inter` (gate concatenated above up).
    /// Output shape: `[batch, inter]`.
    pub fn encode_gate_up_silu_mul_dispatch(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Affine4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
        bf16_out: bool,
    ) {
        if bf16_out {
            encoder.set_compute_pipeline_state(&self.gate_up_silu_mul_bf16out_v3);
        } else {
            encoder.set_compute_pipeline_state(&self.gate_up_silu_mul_v3);
        }
        encoder.set_buffer(0, Some(&weight.packed), weight.packed_offset as usize);
        encoder.set_buffer(1, Some(&weight.scales), weight.scales_offset as usize);
        encoder.set_buffer(2, Some(&weight.biases), weight.biases_offset as usize);
        encoder.set_buffer(3, Some(x_buf), x_offset as usize);
        encoder.set_buffer(4, Some(y_buf), y_offset as usize);
        let inter = weight.out_features / 2;
        let dims = Affine4GateUpSiluMulDims {
            inter: inter as u32,
            in_features: weight.in_features as u32,
            batch: batch as u32,
        };
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<Affine4GateUpSiluMulDims>(),
            &dims as *const _ as *const _,
        );
        const ROWS_PER_TG: usize = 8;
        const THREADS_PER_TG: usize = 256;
        let tg_mem_bytes = weight.in_features * 4;
        encoder.set_threadgroup_memory_length(0, tg_mem_bytes);
        let n_groups_x = inter.div_ceil(ROWS_PER_TG);
        let grid = MTLSize {
            width: n_groups_x,
            height: batch,
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid, tg);
    }

    /// One-shot CPU-staged matmul: `y = x @ W^T` where `x` is a host slice.
    ///
    /// Allocates fresh device buffers, submits a command, blocks for completion,
    /// reads the output back. Use `encode_matmul_dispatch` + caller-managed
    /// command buffers for the production hot path.
    pub fn matmul_with_weight(
        &self,
        weight: &Affine4Weight,
        x: &[f32],
        batch: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(
            x.len() == batch * weight.in_features,
            "x length {} != batch({}) * in_features({})",
            x.len(),
            batch,
            weight.in_features
        );
        if batch == 0 {
            return Ok(Vec::new());
        }

        let x_buf = self.ctx.buffer_with_data(x);
        let y_buf = self.ctx.buffer_for::<f32>(batch * weight.out_features);

        let encoder = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        encoder.set_label("lumen:affine4_matmul");
        self.encode_matmul_dispatch(encoder.as_ref(), weight, &x_buf, 0, &y_buf, 0, batch);
        drop(encoder);
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");

        Ok(self
            .ctx
            .read_buffer::<f32>(&y_buf, batch * weight.out_features))
    }

    /// Zero-copy variant: `x` and `y` already live as Metal buffers.
    /// Submits on the context's own queue and waits — same hazard handling
    /// rationale as `MxFp4Context::matmul_zero_copy` (cross-queue sync via
    /// caller's `wait_until_completed` before invocation).
    pub fn matmul_zero_copy(
        &self,
        weight: &Affine4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let encoder = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        encoder.set_label("lumen:affine4_matmul");
        self.encode_matmul_dispatch(
            encoder.as_ref(),
            weight,
            x_buf,
            x_offset,
            y_buf,
            y_offset,
            batch,
        );
        drop(encoder);
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");
        Ok(())
    }

    /// Convenience wrapper for batch=1.
    pub fn matvec_with_weight(&self, weight: &Affine4Weight, x: &[f32]) -> Result<Vec<f32>> {
        anyhow::ensure!(
            x.len() == weight.in_features,
            "x length {} != in_features {}",
            x.len(),
            weight.in_features
        );
        let x_buf = self.ctx.buffer_with_data(x);
        let y_buf = self.ctx.buffer_for::<f32>(weight.out_features);

        let encoder = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        encoder.set_label("lumen:affine4_matvec");
        encoder.set_compute_pipeline_state(&self.matvec_f32);
        encoder.set_buffer(0, Some(&weight.packed), weight.packed_offset as usize);
        encoder.set_buffer(1, Some(&weight.scales), weight.scales_offset as usize);
        encoder.set_buffer(2, Some(&weight.biases), weight.biases_offset as usize);
        encoder.set_buffer(3, Some(&x_buf), 0);
        encoder.set_buffer(4, Some(&y_buf), 0);
        let dims = Affine4Dims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
        };
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<Affine4Dims>(),
            &dims as *const _ as *const _,
        );

        let max_threads = self.matvec_f32.max_total_threads_per_threadgroup();
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
        drop(encoder);
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");

        Ok(self.ctx.read_buffer::<f32>(&y_buf, weight.out_features))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_from_f32(x: f32) -> u16 {
        (x.to_bits() >> 16) as u16
    }

    /// Identity dequant: scale=1.0, bias=0.0 → `w_real == nibble_value`. Verifies
    /// the basic kernel arithmetic against a hand-checked reference.
    #[test]
    fn identity_dequant_matvec_matches_cpu() -> Result<()> {
        let ctx = match Affine4Context::new() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("affine4: skipping (no Metal device)");
                return Ok(());
            }
        };
        // out=2, in=64 — exactly one group.
        let out_features = 2;
        let in_features = 64;
        // Row 0: nibbles 0..15 repeated 4×; Row 1: all 1s.
        let mut packed = vec![0u32; out_features * in_features / 8];
        for i in 0..(in_features / 8) {
            // Row 0: 0,1,2,3,4,5,6,7
            let w0 = (0..8).fold(0u32, |acc, k| acc | ((k as u32) << (k * 4)));
            packed[i] = w0;
        }
        for i in 0..(in_features / 8) {
            let w1 = (0..8).fold(0u32, |acc, k| acc | (1u32 << (k * 4)));
            packed[(in_features / 8) + i] = w1;
        }
        let scales = vec![bf16_from_f32(1.0); out_features];
        let biases = vec![bf16_from_f32(0.0); out_features];

        let weight = Affine4Weight::from_host(
            &ctx.ctx,
            &packed,
            &scales,
            &biases,
            out_features,
            in_features,
        )?;

        // x = all 1s. y[r] = sum_k w[r,k] * 1 = sum of dequantized nibbles in row r.
        let x = vec![1.0f32; in_features];
        let y = ctx.matvec_with_weight(&weight, &x)?;

        // Row 0: 8 groups × (0+1+2+3+4+5+6+7) = 8 × 28 = 224. Wait — only 1 group
        // here since in=64. Actually packed[0..8] = 8 words = 1 row. Each word
        // has nibbles 0..7. Row 0 sum = 8 words × (0+1+2+3+4+5+6+7) = 8 × 28 = 224.
        assert!(
            (y[0] - 224.0).abs() < 1e-4,
            "row 0: expected 224.0, got {}",
            y[0]
        );
        // Row 1: 64 nibbles all = 1, scale 1, bias 0 → sum = 64.
        assert!(
            (y[1] - 64.0).abs() < 1e-4,
            "row 1: expected 64.0, got {}",
            y[1]
        );
        Ok(())
    }

    /// Bias-only dequant: scale=0, bias=B → every dequantized weight equals B.
    /// y[r] = B * sum(x_k). Validates bias path independently of nibble values.
    #[test]
    fn bias_only_dequant() -> Result<()> {
        let ctx = match Affine4Context::new() {
            Ok(c) => c,
            Err(_) => return Ok(()),
        };
        let out_features = 1;
        let in_features = 64;
        let packed = vec![0xDEAD_BEEFu32; out_features * in_features / 8];
        let scales = vec![bf16_from_f32(0.0)];
        let biases = vec![bf16_from_f32(2.5)];
        let weight = Affine4Weight::from_host(
            &ctx.ctx,
            &packed,
            &scales,
            &biases,
            out_features,
            in_features,
        )?;
        let x = vec![1.0f32; in_features];
        let y = ctx.matvec_with_weight(&weight, &x)?;
        // Each weight = 0 * nibble + 2.5 = 2.5; 64 weights × 1.0 input = 160.
        assert!(
            (y[0] - 160.0).abs() < 1e-4,
            "bias-only: expected 160.0, got {}",
            y[0]
        );
        Ok(())
    }
}
