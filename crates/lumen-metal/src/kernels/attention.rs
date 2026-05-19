use anyhow::Result;

use crate::device::MetalContext;
use crate::metal::{BatchedEncoderExt, CommandBufferExt, ComputeEncoderCompat};
use crate::pipeline::ShaderPipelines;

/// Compute attention scores from compressed key cache on GPU.
///
/// score[i] = stage1(rotated_query, packed_keys[i]) + stage2(query, qjl_bits[i])
///
/// Returns scores in `scores_out` buffer [n_kv] f32.
pub fn compressed_attention_scores(
    ctx: &MetalContext,
    pipelines: &ShaderPipelines,
    // Query (pre-rotated and original)
    rotated_query: &crate::metal::Buffer, // [dim] f32
    query_orig: &crate::metal::Buffer,    // [dim] f32
    // Compressed key cache
    packed_codes: &crate::metal::Buffer, // [n_kv x n_packed] u64
    scales: &crate::metal::Buffer,       // [n_kv] f32
    centroids: &crate::metal::Buffer,    // [n_levels] f32
    qjl_packed: &crate::metal::Buffer,   // [n_kv x n_qjl_packed] u64
    qjl_matrix: &crate::metal::Buffer,   // [qjl_m x dim] f32
    res_norms: &crate::metal::Buffer,    // [n_kv] f32
    // Output
    scores_out: &crate::metal::Buffer, // [n_kv] f32
    // Parameters
    dim: u32,
    n_kv: u32,
    bits: u32,
    n_packed: u32,
    qjl_m: u32,
    n_qjl_packed: u32,
    n_levels: u32,
) -> Result<()> {
    let encoder = crate::metal::process_commands()
        .command_encoder()
        .expect("ce");
    encoder.set_label("lumen:tq_compressed_attention_scores");
    let pipeline = pipelines.get("tq_compressed_attention_scores")?;
    encoder.set_compute_pipeline_state(pipeline);

    encoder.set_buffer(0, Some(rotated_query), (0) as usize);
    encoder.set_buffer(1, Some(query_orig), (0) as usize);
    encoder.set_buffer(2, Some(packed_codes), (0) as usize);
    encoder.set_buffer(3, Some(scales), (0) as usize);
    encoder.set_buffer(4, Some(centroids), (0) as usize);
    encoder.set_buffer(5, Some(qjl_packed), (0) as usize);
    encoder.set_buffer(6, Some(qjl_matrix), (0) as usize);
    encoder.set_buffer(7, Some(res_norms), (0) as usize);
    encoder.set_buffer(8, Some(scores_out), (0) as usize);
    set_u32(encoder.as_ref(), 9, dim);
    set_u32(encoder.as_ref(), 10, n_kv);
    set_u32(encoder.as_ref(), 11, bits);
    set_u32(encoder.as_ref(), 12, n_packed);
    set_u32(encoder.as_ref(), 13, qjl_m);
    set_u32(encoder.as_ref(), 14, n_qjl_packed);
    set_u32(encoder.as_ref(), 15, n_levels);

    // One thread per KV position
    let grid = crate::mtl_size!(n_kv as usize, 1, 1);
    let max_threads = pipeline.max_total_threads_per_threadgroup();
    let threadgroup = crate::mtl_size!(n_kv.min(max_threads as u32) as u64, 1, 1);
    encoder.dispatch_threads(grid, threadgroup);

    drop(encoder);
    crate::metal::process_commands()
        .flush_and_wait()
        .expect("flush");

    Ok(())
}

/// Gather weighted sum of compressed values using attention weights.
///
/// output[d] = sum_i weights[i] * reconstruct(packed_values[i])[d]
///
/// Returns the weighted sum in `output` buffer [dim] f32.
pub fn compressed_value_gather(
    ctx: &MetalContext,
    pipelines: &ShaderPipelines,
    // Attention weights (post-softmax)
    weights: &crate::metal::Buffer, // [n_kv] f32
    // Compressed value cache
    packed_codes: &crate::metal::Buffer, // [n_kv x n_packed] u64
    scales: &crate::metal::Buffer,       // [n_kv] f32
    centroids: &crate::metal::Buffer,    // [n_levels] f32
    rotation: &crate::metal::Buffer,     // [dim x dim] f32
    // Output
    output: &crate::metal::Buffer, // [dim] f32
    // Parameters
    dim: u32,
    n_kv: u32,
    bits: u32,
    n_packed: u32,
    n_levels: u32,
) -> Result<()> {
    let encoder = crate::metal::process_commands()
        .command_encoder()
        .expect("ce");
    encoder.set_label("lumen:tq_compressed_value_gather");
    let pipeline = pipelines.get("tq_compressed_value_gather")?;
    encoder.set_compute_pipeline_state(pipeline);

    encoder.set_buffer(0, Some(weights), (0) as usize);
    encoder.set_buffer(1, Some(packed_codes), (0) as usize);
    encoder.set_buffer(2, Some(scales), (0) as usize);
    encoder.set_buffer(3, Some(centroids), (0) as usize);
    encoder.set_buffer(4, Some(rotation), (0) as usize);
    encoder.set_buffer(5, Some(output), (0) as usize);
    set_u32(encoder.as_ref(), 6, dim);
    set_u32(encoder.as_ref(), 7, n_kv);
    set_u32(encoder.as_ref(), 8, bits);
    set_u32(encoder.as_ref(), 9, n_packed);
    set_u32(encoder.as_ref(), 10, n_levels);

    // Two-pass kernel: all threads must be in ONE threadgroup for barrier sync
    encoder.dispatch_thread_groups(
        crate::mtl_size!(1, 1, 1),
        crate::mtl_size!(dim as usize, 1, 1),
    );

    drop(encoder);
    crate::metal::process_commands()
        .flush_and_wait()
        .expect("flush");

    Ok(())
}

/// Project query through QJL matrix once: qjl_proj[j] = sum_d qjl_matrix[j*dim+d] * query[d].
///
/// Output is reused by `compressed_attention_scores_v2` for every kv_idx, eliminating
/// the O(qjl_m * dim * n_kv) redundancy of the legacy kernel.
pub fn qjl_project_query(
    ctx: &MetalContext,
    pipelines: &ShaderPipelines,
    query: &crate::metal::Buffer,        // [dim] f32
    qjl_matrix: &crate::metal::Buffer,   // [qjl_m x dim] f32
    qjl_proj_out: &crate::metal::Buffer, // [qjl_m] f32
    dim: u32,
    qjl_m: u32,
) -> Result<()> {
    let encoder = crate::metal::process_commands()
        .command_encoder()
        .expect("ce");
    encoder.set_label("lumen:tq_qjl_project_query");
    let pipeline = pipelines.get("tq_qjl_project_query")?;
    encoder.set_compute_pipeline_state(pipeline);

    encoder.set_buffer(0, Some(query), 0);
    encoder.set_buffer(1, Some(qjl_matrix), 0);
    encoder.set_buffer(2, Some(qjl_proj_out), 0);
    set_u32(encoder.as_ref(), 3, dim);
    set_u32(encoder.as_ref(), 4, qjl_m);

    let grid = crate::mtl_size!(qjl_m as usize, 1, 1);
    let max_threads = pipeline.max_total_threads_per_threadgroup();
    let threadgroup = crate::mtl_size!(qjl_m.min(max_threads as u32) as u64, 1, 1);
    encoder.dispatch_threads(grid, threadgroup);

    drop(encoder);
    crate::metal::process_commands()
        .flush_and_wait()
        .expect("flush");
    Ok(())
}

/// V2 of compressed_attention_scores: consumes precomputed QJL projection.
///
/// Caller MUST first dispatch `qjl_project_query` to populate `qjl_proj`.
/// Eliminates per-kv_idx recomputation of qjl_matrix . query.
pub fn compressed_attention_scores_v2(
    ctx: &MetalContext,
    pipelines: &ShaderPipelines,
    rotated_query: &crate::metal::Buffer, // [dim] f32
    qjl_proj: &crate::metal::Buffer,      // [qjl_m] f32, precomputed
    packed_codes: &crate::metal::Buffer,
    scales: &crate::metal::Buffer,
    centroids: &crate::metal::Buffer,
    qjl_packed: &crate::metal::Buffer,
    res_norms: &crate::metal::Buffer,
    scores_out: &crate::metal::Buffer,
    dim: u32,
    n_kv: u32,
    bits: u32,
    n_packed: u32,
    qjl_m: u32,
    n_qjl_packed: u32,
    n_levels: u32,
) -> Result<()> {
    let encoder = crate::metal::process_commands()
        .command_encoder()
        .expect("ce");
    encoder.set_label("lumen:tq_compressed_attention_scores_v2");
    let pipeline = pipelines.get("tq_compressed_attention_scores_v2")?;
    encoder.set_compute_pipeline_state(pipeline);

    encoder.set_buffer(0, Some(rotated_query), 0);
    encoder.set_buffer(1, Some(qjl_proj), 0);
    encoder.set_buffer(2, Some(packed_codes), 0);
    encoder.set_buffer(3, Some(scales), 0);
    encoder.set_buffer(4, Some(centroids), 0);
    encoder.set_buffer(5, Some(qjl_packed), 0);
    encoder.set_buffer(6, Some(res_norms), 0);
    encoder.set_buffer(7, Some(scores_out), 0);
    set_u32(encoder.as_ref(), 8, dim);
    set_u32(encoder.as_ref(), 9, n_kv);
    set_u32(encoder.as_ref(), 10, bits);
    set_u32(encoder.as_ref(), 11, n_packed);
    set_u32(encoder.as_ref(), 12, qjl_m);
    set_u32(encoder.as_ref(), 13, n_qjl_packed);
    set_u32(encoder.as_ref(), 14, n_levels);

    let grid = crate::mtl_size!(n_kv as usize, 1, 1);
    let max_threads = pipeline.max_total_threads_per_threadgroup();
    let threadgroup = crate::mtl_size!(n_kv.min(max_threads as u32) as u64, 1, 1);
    encoder.dispatch_threads(grid, threadgroup);

    drop(encoder);
    crate::metal::process_commands()
        .flush_and_wait()
        .expect("flush");
    Ok(())
}

/// V3: same input/output contract as v2 but Stage 1 inner loop is float4-vectorized.
/// Requires `dim % 4 == 0` (caller asserts).
pub fn compressed_attention_scores_v3(
    ctx: &MetalContext,
    pipelines: &ShaderPipelines,
    rotated_query: &crate::metal::Buffer,
    qjl_proj: &crate::metal::Buffer,
    packed_codes: &crate::metal::Buffer,
    scales: &crate::metal::Buffer,
    centroids: &crate::metal::Buffer,
    qjl_packed: &crate::metal::Buffer,
    res_norms: &crate::metal::Buffer,
    scores_out: &crate::metal::Buffer,
    dim: u32,
    n_kv: u32,
    bits: u32,
    n_packed: u32,
    qjl_m: u32,
    n_qjl_packed: u32,
    n_levels: u32,
) -> Result<()> {
    assert!(dim % 4 == 0, "v3 requires dim % 4 == 0 (got dim={dim})");

    let encoder = crate::metal::process_commands()
        .command_encoder()
        .expect("ce");
    encoder.set_label("lumen:tq_compressed_attention_scores_v3");
    let pipeline = pipelines.get("tq_compressed_attention_scores_v3")?;
    encoder.set_compute_pipeline_state(pipeline);

    encoder.set_buffer(0, Some(rotated_query), 0);
    encoder.set_buffer(1, Some(qjl_proj), 0);
    encoder.set_buffer(2, Some(packed_codes), 0);
    encoder.set_buffer(3, Some(scales), 0);
    encoder.set_buffer(4, Some(centroids), 0);
    encoder.set_buffer(5, Some(qjl_packed), 0);
    encoder.set_buffer(6, Some(res_norms), 0);
    encoder.set_buffer(7, Some(scores_out), 0);
    set_u32(encoder.as_ref(), 8, dim);
    set_u32(encoder.as_ref(), 9, n_kv);
    set_u32(encoder.as_ref(), 10, bits);
    set_u32(encoder.as_ref(), 11, n_packed);
    set_u32(encoder.as_ref(), 12, qjl_m);
    set_u32(encoder.as_ref(), 13, n_qjl_packed);
    set_u32(encoder.as_ref(), 14, n_levels);

    let grid = crate::mtl_size!(n_kv as usize, 1, 1);
    let max_threads = pipeline.max_total_threads_per_threadgroup();
    let threadgroup = crate::mtl_size!(n_kv.min(max_threads as u32) as u64, 1, 1);
    encoder.dispatch_threads(grid, threadgroup);

    drop(encoder);
    crate::metal::process_commands()
        .flush_and_wait()
        .expect("flush");
    Ok(())
}

/// V4: V3 + centroids cached in threadgroup memory.
/// Single-threadgroup dispatch (one threadgroup, n_kv threads).
pub fn compressed_attention_scores_v4(
    ctx: &MetalContext,
    pipelines: &ShaderPipelines,
    rotated_query: &crate::metal::Buffer,
    qjl_proj: &crate::metal::Buffer,
    packed_codes: &crate::metal::Buffer,
    scales: &crate::metal::Buffer,
    centroids: &crate::metal::Buffer,
    qjl_packed: &crate::metal::Buffer,
    res_norms: &crate::metal::Buffer,
    scores_out: &crate::metal::Buffer,
    dim: u32,
    n_kv: u32,
    bits: u32,
    n_packed: u32,
    qjl_m: u32,
    n_qjl_packed: u32,
    n_levels: u32,
) -> Result<()> {
    assert!(dim % 4 == 0, "v4 requires dim % 4 == 0 (got dim={dim})");

    let encoder = crate::metal::process_commands()
        .command_encoder()
        .expect("ce");
    encoder.set_label("lumen:tq_compressed_attention_scores_v4");
    let pipeline = pipelines.get("tq_compressed_attention_scores_v4")?;
    encoder.set_compute_pipeline_state(pipeline);

    encoder.set_buffer(0, Some(rotated_query), 0);
    encoder.set_buffer(1, Some(qjl_proj), 0);
    encoder.set_buffer(2, Some(packed_codes), 0);
    encoder.set_buffer(3, Some(scales), 0);
    encoder.set_buffer(4, Some(centroids), 0);
    encoder.set_buffer(5, Some(qjl_packed), 0);
    encoder.set_buffer(6, Some(res_norms), 0);
    encoder.set_buffer(7, Some(scores_out), 0);
    set_u32(encoder.as_ref(), 8, dim);
    set_u32(encoder.as_ref(), 9, n_kv);
    set_u32(encoder.as_ref(), 10, bits);
    set_u32(encoder.as_ref(), 11, n_packed);
    set_u32(encoder.as_ref(), 12, qjl_m);
    set_u32(encoder.as_ref(), 13, n_qjl_packed);
    set_u32(encoder.as_ref(), 14, n_levels);

    // Multi-threadgroup: each threadgroup loads centroids cache once, then
    // every thread in that group runs scores for one kv_idx. Threadgroup
    // size capped at kernel max so n_kv > 1024 still works correctly.
    let max_threads = pipeline.max_total_threads_per_threadgroup() as u64;
    let tg = (n_kv as u64).min(max_threads).max(n_levels as u64);
    let grid = crate::mtl_size!(n_kv as usize, 1, 1);
    let threadgroup = crate::mtl_size!(tg, 1, 1);
    encoder.dispatch_threads(grid, threadgroup);

    drop(encoder);
    crate::metal::process_commands()
        .flush_and_wait()
        .expect("flush");
    Ok(())
}

/// V5: V4 + fp16 Stage 1 (half precision FMA, f32 accumulation, threadgroup-cached half centroids).
/// Multi-threadgroup dispatch (correctly handles n_kv > max_threadgroup).
pub fn compressed_attention_scores_v5(
    ctx: &MetalContext,
    pipelines: &ShaderPipelines,
    rotated_query: &crate::metal::Buffer,
    qjl_proj: &crate::metal::Buffer,
    packed_codes: &crate::metal::Buffer,
    scales: &crate::metal::Buffer,
    centroids: &crate::metal::Buffer,
    qjl_packed: &crate::metal::Buffer,
    res_norms: &crate::metal::Buffer,
    scores_out: &crate::metal::Buffer,
    dim: u32,
    n_kv: u32,
    bits: u32,
    n_packed: u32,
    qjl_m: u32,
    n_qjl_packed: u32,
    n_levels: u32,
) -> Result<()> {
    assert!(dim % 4 == 0, "v5 requires dim % 4 == 0 (got dim={dim})");

    let encoder = crate::metal::process_commands()
        .command_encoder()
        .expect("ce");
    encoder.set_label("lumen:tq_compressed_attention_scores_v5");
    let pipeline = pipelines.get("tq_compressed_attention_scores_v5")?;
    encoder.set_compute_pipeline_state(pipeline);

    encoder.set_buffer(0, Some(rotated_query), 0);
    encoder.set_buffer(1, Some(qjl_proj), 0);
    encoder.set_buffer(2, Some(packed_codes), 0);
    encoder.set_buffer(3, Some(scales), 0);
    encoder.set_buffer(4, Some(centroids), 0);
    encoder.set_buffer(5, Some(qjl_packed), 0);
    encoder.set_buffer(6, Some(res_norms), 0);
    encoder.set_buffer(7, Some(scores_out), 0);
    set_u32(encoder.as_ref(), 8, dim);
    set_u32(encoder.as_ref(), 9, n_kv);
    set_u32(encoder.as_ref(), 10, bits);
    set_u32(encoder.as_ref(), 11, n_packed);
    set_u32(encoder.as_ref(), 12, qjl_m);
    set_u32(encoder.as_ref(), 13, n_qjl_packed);
    set_u32(encoder.as_ref(), 14, n_levels);

    let max_threads = pipeline.max_total_threads_per_threadgroup() as u64;
    let tg = (n_kv as u64).min(max_threads).max(n_levels as u64);
    let grid = crate::mtl_size!(n_kv as usize, 1, 1);
    let threadgroup = crate::mtl_size!(tg, 1, 1);
    encoder.dispatch_threads(grid, threadgroup);

    drop(encoder);
    crate::metal::process_commands()
        .flush_and_wait()
        .expect("flush");
    Ok(())
}

/// V6: SIMD-group cooperative reduction over dim. 1 simd-group (32 threads) per kv_idx.
/// Each lane processes dim/32 chunks; simd_sum reduces. 8 kv_idx per TG (256 threads).
pub fn compressed_attention_scores_v6(
    ctx: &MetalContext,
    pipelines: &ShaderPipelines,
    rotated_query: &crate::metal::Buffer,
    qjl_proj: &crate::metal::Buffer,
    packed_codes: &crate::metal::Buffer,
    scales: &crate::metal::Buffer,
    centroids: &crate::metal::Buffer,
    qjl_packed: &crate::metal::Buffer,
    res_norms: &crate::metal::Buffer,
    scores_out: &crate::metal::Buffer,
    dim: u32,
    n_kv: u32,
    bits: u32,
    n_packed: u32,
    qjl_m: u32,
    n_qjl_packed: u32,
    n_levels: u32,
) -> Result<()> {
    let encoder = crate::metal::process_commands()
        .command_encoder()
        .expect("ce");
    encoder.set_label("lumen:tq_compressed_attention_scores_v6");
    let pipeline = pipelines.get("tq_compressed_attention_scores_v6")?;
    encoder.set_compute_pipeline_state(pipeline);

    encoder.set_buffer(0, Some(rotated_query), 0);
    encoder.set_buffer(1, Some(qjl_proj), 0);
    encoder.set_buffer(2, Some(packed_codes), 0);
    encoder.set_buffer(3, Some(scales), 0);
    encoder.set_buffer(4, Some(centroids), 0);
    encoder.set_buffer(5, Some(qjl_packed), 0);
    encoder.set_buffer(6, Some(res_norms), 0);
    encoder.set_buffer(7, Some(scores_out), 0);
    set_u32(encoder.as_ref(), 8, dim);
    set_u32(encoder.as_ref(), 9, n_kv);
    set_u32(encoder.as_ref(), 10, bits);
    set_u32(encoder.as_ref(), 11, n_packed);
    set_u32(encoder.as_ref(), 12, qjl_m);
    set_u32(encoder.as_ref(), 13, n_qjl_packed);
    set_u32(encoder.as_ref(), 14, n_levels);

    const SIMD_SIZE: u32 = 32;
    const KV_PER_TG: u32 = 8;
    let tg_size = SIMD_SIZE * KV_PER_TG; // 256 threads
    let num_tgs = (n_kv + KV_PER_TG - 1) / KV_PER_TG;
    encoder.dispatch_thread_groups(
        crate::mtl_size!(num_tgs as usize, 1, 1),
        crate::mtl_size!(tg_size as u64, 1, 1),
    );

    drop(encoder);
    crate::metal::process_commands()
        .flush_and_wait()
        .expect("flush");
    Ok(())
}

/// GQA fan-out value gather: same compute as `compressed_value_gather`, but writes
/// the result to `gqa_ratio` consecutive output slices. `output_base` should point
/// to the FIRST Q head's output slot; the kernel writes to `output_base[q*dim + d]`
/// for q in [0, gqa_ratio).
pub fn compressed_value_gather_multi(
    ctx: &MetalContext,
    pipelines: &ShaderPipelines,
    weights: &crate::metal::Buffer,
    packed_codes: &crate::metal::Buffer,
    scales: &crate::metal::Buffer,
    centroids: &crate::metal::Buffer,
    rotation: &crate::metal::Buffer,
    output_base: &crate::metal::Buffer,
    dim: u32,
    n_kv: u32,
    bits: u32,
    n_packed: u32,
    n_levels: u32,
    gqa_ratio: u32,
) -> Result<()> {
    let encoder = crate::metal::process_commands()
        .command_encoder()
        .expect("ce");
    encoder.set_label("lumen:tq_compressed_value_gather_multi");
    let pipeline = pipelines.get("tq_compressed_value_gather_multi")?;
    encoder.set_compute_pipeline_state(pipeline);

    encoder.set_buffer(0, Some(weights), 0);
    encoder.set_buffer(1, Some(packed_codes), 0);
    encoder.set_buffer(2, Some(scales), 0);
    encoder.set_buffer(3, Some(centroids), 0);
    encoder.set_buffer(4, Some(rotation), 0);
    encoder.set_buffer(5, Some(output_base), 0);
    set_u32(encoder.as_ref(), 6, dim);
    set_u32(encoder.as_ref(), 7, n_kv);
    set_u32(encoder.as_ref(), 8, bits);
    set_u32(encoder.as_ref(), 9, n_packed);
    set_u32(encoder.as_ref(), 10, n_levels);
    set_u32(encoder.as_ref(), 11, gqa_ratio);

    encoder.dispatch_thread_groups(
        crate::mtl_size!(1, 1, 1),
        crate::mtl_size!(dim as usize, 1, 1),
    );

    drop(encoder);
    crate::metal::process_commands()
        .flush_and_wait()
        .expect("flush");
    Ok(())
}

/// Parallel softmax (single threadgroup, three-pass max → exp+sum → normalize).
/// Correctly handles arbitrary n via threadgroup-strided iteration.
/// Threadgroup size must be a power of 2 (caller enforces).
pub fn softmax_parallel(
    ctx: &MetalContext,
    pipelines: &ShaderPipelines,
    scores: &crate::metal::Buffer, // [n] f32 in/out
    n: u32,
) -> Result<()> {
    let encoder = crate::metal::process_commands()
        .command_encoder()
        .expect("ce");
    encoder.set_label("lumen:tq_softmax_parallel");
    let pipeline = pipelines.get("tq_softmax_parallel")?;
    encoder.set_compute_pipeline_state(pipeline);

    encoder.set_buffer(0, Some(scores), 0);
    set_u32(encoder.as_ref(), 1, n);

    let max_tg = pipeline.max_total_threads_per_threadgroup() as u64;
    let tg = (n as u64).next_power_of_two().min(max_tg);
    encoder.dispatch_thread_groups(crate::mtl_size!(1, 1, 1), crate::mtl_size!(tg, 1, 1));

    drop(encoder);
    crate::metal::process_commands()
        .flush_and_wait()
        .expect("flush");
    Ok(())
}

fn set_u32(encoder: &crate::metal::ComputeCommandEncoderRef, index: u64, value: u32) {
    let bytes = value.to_ne_bytes();
    encoder.set_bytes_directly(
        index as usize,
        std::mem::size_of::<u32>(),
        bytes.as_ptr() as *const _,
    );
}
