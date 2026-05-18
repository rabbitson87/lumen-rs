use anyhow::Result;

use crate::device::MetalContext;
use crate::metal::{BatchedEncoderExt, CommandBufferExt, ComputeEncoderCompat};
use crate::pipeline::ShaderPipelines;

/// Dispatch the 4 compression kernels to compress KV vectors on GPU.
///
/// Pipeline: rotate_and_normalize -> lloyd_max_quantize -> bitpack_and_residual -> qjl_project_signs
///
/// All 4 kernels run sequentially within one command buffer (GPU-side ordering).
/// No CPU round-trip between kernels.
///
/// Uses `ctx` to allocate temporary intermediate buffers. Creates its own
/// command buffer, commits, and waits (synchronous).
pub fn compress_vectors(
    ctx: &MetalContext,
    pipelines: &ShaderPipelines,
    // Input
    kv_vectors: &crate::metal::Buffer, // [n_vecs x dim] f32
    // Pre-loaded constant buffers
    rotation: &crate::metal::Buffer,   // [dim x dim] f32
    boundaries: &crate::metal::Buffer, // [n_levels + 1] f32
    centroids: &crate::metal::Buffer,  // [n_levels] f32
    qjl_matrix: &crate::metal::Buffer, // [qjl_m x dim] f32
    // Output (pre-allocated from pool)
    packed_out: &crate::metal::Buffer,    // [n_vecs x n_packed] u64
    scales_out: &crate::metal::Buffer,    // [n_vecs] f32
    res_norms_out: &crate::metal::Buffer, // [n_vecs] f32
    qjl_packed_out: &crate::metal::Buffer, // [n_vecs x n_qjl_packed] u64
    // Parameters
    dim: u32,
    n_vecs: u32,
    bits: u32,
    n_levels: u32,
    n_packed: u32,
    qjl_m: u32,
    n_qjl_packed: u32,
) -> Result<()> {

    // Intermediate buffers (temporary, within this command buffer)
    let rotated_buf = ctx.buffer_for::<f32>(n_vecs as usize * dim as usize);
    let codes_buf = ctx.buffer_for::<u8>(n_vecs as usize * dim as usize);
    let residuals_buf = ctx.buffer_for::<f32>(n_vecs as usize * dim as usize);

    // Note: compute encoders auto-end via Drop; no explicit end_encoding().

    // --- Kernel 1: Rotate and Normalize ---
    {
        let encoder = crate::metal::process_commands().command_encoder().expect("ce");
        encoder.set_label("lumen:tq_rotate_and_normalize");
        let pipeline = pipelines.get("tq_rotate_and_normalize")?;
        encoder.set_compute_pipeline_state(pipeline);

        encoder.set_buffer(0, Some(kv_vectors), (0) as usize);
        encoder.set_buffer(1, Some(rotation), (0) as usize);
        encoder.set_buffer(2, Some(&rotated_buf), (0) as usize);
        encoder.set_buffer(3, Some(scales_out), (0) as usize);
        set_u32(encoder.as_ref(), 4, dim);
        set_u32(encoder.as_ref(), 5, n_vecs);

        // Grid: [dim, n_vecs] — one thread per element
        let grid = crate::mtl_size!(dim as usize, n_vecs as usize, 1);
        let threadgroup = crate::mtl_size!(dim.min(256) as u64, 1, 1);
        encoder.dispatch_threads(grid, threadgroup);
    }

    // --- Kernel 2: Lloyd-Max Quantize ---
    {
        let encoder = crate::metal::process_commands().command_encoder().expect("ce");
        encoder.set_label("lumen:tq_lloyd_max_quantize");
        let pipeline = pipelines.get("tq_lloyd_max_quantize")?;
        encoder.set_compute_pipeline_state(pipeline);

        encoder.set_buffer(0, Some(&rotated_buf), (0) as usize);
        encoder.set_buffer(1, Some(boundaries), (0) as usize);
        encoder.set_buffer(2, Some(&codes_buf), (0) as usize);
        set_u32(encoder.as_ref(), 3, dim);
        set_u32(encoder.as_ref(), 4, n_vecs);
        set_u32(encoder.as_ref(), 5, n_levels);

        let grid = crate::mtl_size!(dim as usize, n_vecs as usize, 1);
        let threadgroup = crate::mtl_size!(dim.min(256) as u64, 1, 1);
        encoder.dispatch_threads(grid, threadgroup);
    }

    // --- Kernel 3: Bitpack and Residual ---
    {
        let encoder = crate::metal::process_commands().command_encoder().expect("ce");
        encoder.set_label("lumen:tq_bitpack_and_residual");
        let pipeline = pipelines.get("tq_bitpack_and_residual")?;
        encoder.set_compute_pipeline_state(pipeline);

        encoder.set_buffer(0, Some(&codes_buf), (0) as usize);
        encoder.set_buffer(1, Some(centroids), (0) as usize);
        encoder.set_buffer(2, Some(scales_out), (0) as usize);
        encoder.set_buffer(3, Some(rotation), (0) as usize);
        encoder.set_buffer(4, Some(kv_vectors), (0) as usize);
        encoder.set_buffer(5, Some(packed_out), (0) as usize);
        encoder.set_buffer(6, Some(&residuals_buf), (0) as usize);
        encoder.set_buffer(7, Some(res_norms_out), (0) as usize);
        set_u32(encoder.as_ref(), 8, dim);
        set_u32(encoder.as_ref(), 9, n_vecs);
        set_u32(encoder.as_ref(), 10, bits);
        set_u32(encoder.as_ref(), 11, n_packed);

        let grid = crate::mtl_size!(dim as usize, n_vecs as usize, 1);
        let threadgroup = crate::mtl_size!(dim.min(256) as u64, 1, 1);
        encoder.dispatch_threads(grid, threadgroup);
    }

    // --- Kernel 4: QJL Project Signs (1 thread per vector) ---
    {
        let encoder = crate::metal::process_commands().command_encoder().expect("ce");
        encoder.set_label("lumen:tq_qjl_project_signs");
        let pipeline = pipelines.get("tq_qjl_project_signs")?;
        encoder.set_compute_pipeline_state(pipeline);

        encoder.set_buffer(0, Some(&residuals_buf), (0) as usize);
        encoder.set_buffer(1, Some(qjl_matrix), (0) as usize);
        encoder.set_buffer(2, Some(qjl_packed_out), (0) as usize);
        set_u32(encoder.as_ref(), 3, dim);
        set_u32(encoder.as_ref(), 4, n_vecs);
        set_u32(encoder.as_ref(), 5, qjl_m);
        set_u32(encoder.as_ref(), 6, n_qjl_packed);

        let grid = crate::mtl_size!(n_vecs as usize, 1, 1);
        let max_tg = pipeline.max_total_threads_per_threadgroup();
        let threadgroup = crate::mtl_size!((n_vecs as u64).min(max_tg as u64), 1, 1);
        encoder.dispatch_threads(grid, threadgroup);
    }

    crate::metal::process_commands().flush_and_wait().expect("flush");

    Ok(())
}

/// Encode compression kernels into an existing command buffer (async, no wait).
///
/// Same as `compress_vectors` but does NOT create/commit/wait on a command buffer.
/// Caller is responsible for committing and waiting.
///
/// `scratch_rotated`, `scratch_codes`, `scratch_residuals` are pre-allocated scratch
/// buffers reused across calls to avoid per-call Metal buffer allocations.
pub fn encode_compress(
    _ctx: &MetalContext,
    pipelines: &ShaderPipelines,
    kv_vectors: &crate::metal::Buffer,
    kv_offset: usize, // byte offset into kv_vectors
    rotation: &crate::metal::Buffer,
    boundaries: &crate::metal::Buffer,
    centroids: &crate::metal::Buffer,
    qjl_matrix: &crate::metal::Buffer,
    packed_out: &crate::metal::Buffer,
    packed_offset: usize,
    scales_out: &crate::metal::Buffer,
    scales_offset: usize,
    res_norms_out: &crate::metal::Buffer,
    rn_offset: usize,
    qjl_packed_out: &crate::metal::Buffer,
    qjl_offset: usize,
    // Pre-allocated scratch buffers (eliminates per-call allocation)
    scratch_rotated: &crate::metal::Buffer, // [n_vecs * dim] f32
    scratch_codes: &crate::metal::Buffer,   // [n_vecs * dim] u8
    scratch_residuals: &crate::metal::Buffer, // [n_vecs * dim] f32
    dim: u32,
    n_vecs: u32,
    bits: u32,
    n_levels: u32,
    n_packed: u32,
    qjl_m: u32,
    n_qjl_packed: u32,
) -> anyhow::Result<()> {
    let rotated_buf = scratch_rotated;
    let codes_buf = scratch_codes;
    let residuals_buf = scratch_residuals;

    // Fused Kernel 1+2: Rotate, Normalize, and Quantize (single dispatch)
    //
    // Note: ComputeCommandEncoder has a Drop impl that auto-calls end_encoding().
    // We rely on scope-end Drop instead of explicit end_encoding() to avoid
    // double-end, which fires `endEncoding has already been called` Metal
    // assertions on long-prompt prefill (BL ≥ 2048) when many encoders queue
    // up in one command buffer.
    {
        let enc = crate::metal::process_commands().command_encoder().expect("ce");
        enc.set_label("lumen:tq_rotate_normalize_quantize");
        let p = pipelines.get("tq_rotate_normalize_quantize")?;
        enc.set_compute_pipeline_state(p);
        enc.set_buffer(0, Some(kv_vectors), (kv_offset) as usize);
        enc.set_buffer(1, Some(rotation), (0) as usize);
        enc.set_buffer(2, Some(boundaries), (0) as usize);
        enc.set_buffer(3, Some(scales_out), (scales_offset) as usize);
        enc.set_buffer(4, Some(codes_buf), (0) as usize);
        enc.set_buffer(5, Some(rotated_buf), (0) as usize);
        set_u32(enc.as_ref(), 6, dim);
        set_u32(enc.as_ref(), 7, n_vecs);
        set_u32(enc.as_ref(), 8, n_levels);
        let grid = crate::mtl_size!(dim as usize, n_vecs as usize, 1);
        let tg = crate::mtl_size!(dim.min(256) as u64, 1, 1);
        enc.dispatch_threads(grid, tg);
    }
    // Kernel 3: Bitpack and Residual
    {
        let enc = crate::metal::process_commands().command_encoder().expect("ce");
        enc.set_label("lumen:tq_bitpack_and_residual");
        let p = pipelines.get("tq_bitpack_and_residual")?;
        enc.set_compute_pipeline_state(p);
        enc.set_buffer(0, Some(&codes_buf), (0) as usize);
        enc.set_buffer(1, Some(centroids), (0) as usize);
        enc.set_buffer(2, Some(scales_out), (scales_offset) as usize);
        enc.set_buffer(3, Some(rotation), (0) as usize);
        enc.set_buffer(4, Some(kv_vectors), (kv_offset) as usize);
        enc.set_buffer(5, Some(packed_out), (packed_offset) as usize);
        enc.set_buffer(6, Some(&residuals_buf), (0) as usize);
        enc.set_buffer(7, Some(res_norms_out), (rn_offset) as usize);
        set_u32(enc.as_ref(), 8, dim);
        set_u32(enc.as_ref(), 9, n_vecs);
        set_u32(enc.as_ref(), 10, bits);
        set_u32(enc.as_ref(), 11, n_packed);
        let grid = crate::mtl_size!(dim as usize, n_vecs as usize, 1);
        let tg = crate::mtl_size!(dim.min(256) as u64, 1, 1);
        enc.dispatch_threads(grid, tg);
    }
    // Kernel 4: QJL Project Signs
    {
        let enc = crate::metal::process_commands().command_encoder().expect("ce");
        enc.set_label("lumen:tq_qjl_project_signs");
        let p = pipelines.get("tq_qjl_project_signs")?;
        enc.set_compute_pipeline_state(p);
        enc.set_buffer(0, Some(&residuals_buf), (0) as usize);
        enc.set_buffer(1, Some(qjl_matrix), (0) as usize);
        enc.set_buffer(2, Some(qjl_packed_out), (qjl_offset) as usize);
        set_u32(enc.as_ref(), 3, dim);
        set_u32(enc.as_ref(), 4, n_vecs);
        set_u32(enc.as_ref(), 5, qjl_m);
        set_u32(enc.as_ref(), 6, n_qjl_packed);
        let grid = crate::mtl_size!(n_vecs as usize, 1, 1);
        let max_tg = p.max_total_threads_per_threadgroup();
        let tg = crate::mtl_size!((n_vecs as u64).min(max_tg as u64), 1, 1);
        enc.dispatch_threads(grid, tg);
    }
    Ok(())
}

/// Helper: set a u32 constant at buffer index.
fn set_u32(encoder: &crate::metal::ComputeCommandEncoderRef, index: u64, value: u32) {
    let bytes = value.to_ne_bytes();
    encoder.set_bytes_directly(
        index as usize,
        std::mem::size_of::<u32>(),
        bytes.as_ptr() as *const _,
    );
}
