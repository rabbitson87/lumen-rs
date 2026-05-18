//! Rust ↔ Metal kernel dispatch bindings for PagedAttention.
//!
//! Built on top of Candle's Metal stack (`candle-metal-kernels` / `objc2-metal`)
//! so that kernel dispatches share the same `MTLDevice` and command buffer pool
//! as Candle's own tensor ops. This eliminates cross-queue synchronization and
//! lets us bind Candle-owned `Buffer`s directly to our kernels (zero-copy).

use std::collections::HashMap;

use anyhow::{Context, Result};
use candle_metal_kernels::metal::{Buffer, ComputePipeline, Device, Library};
use candle_metal_kernels::utils::EncoderProvider;
use objc2_metal::{MTLCompileOptions, MTLLanguageVersion, MTLSize};

use crate::block_allocator::BlockAllocator;

/// Names of all PagedAttention Metal kernels.
pub const KERNEL_NAMES: &[&str] = &[
    "paged_attention_decode",
    "paged_attention_decode_v2",
    "write_kv_to_blocks",
    "write_kv_to_blocks_f32_candle",
    "copy_blocks",
];

/// Compiled Metal pipeline states for paged attention kernels.
pub struct PagedAttentionPipelines {
    #[allow(dead_code)]
    library: Library,
    pipelines: HashMap<&'static str, ComputePipeline>,
}

impl PagedAttentionPipelines {
    /// Compile paged attention Metal shaders and build pipeline states.
    pub fn new(device: &Device) -> Result<Self> {
        let source = include_str!("../shaders/paged_attention.metal");

        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version3_0);

        let library = device
            .new_library_with_source(source, Some(&options))
            .map_err(|e| anyhow::anyhow!("PagedAttention Metal compile error: {e}"))?;

        let mut pipelines = HashMap::new();
        for &name in KERNEL_NAMES {
            let func = library
                .get_function(name, None)
                .map_err(|e| anyhow::anyhow!("kernel '{name}' not found: {e}"))?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| anyhow::anyhow!("pipeline '{name}' failed: {e}"))?;
            pipelines.insert(name, pipeline);
        }

        eprintln!("PagedAttention Metal: compiled {} kernels", pipelines.len());
        Ok(Self { library, pipelines })
    }

    /// Get a cached pipeline state by kernel name.
    pub fn get(&self, name: &str) -> Result<&ComputePipeline> {
        self.pipelines
            .get(name)
            .with_context(|| format!("unknown kernel: {name}"))
    }
}

/// Dispatch paged attention decode for one layer (v1, 3-pass).
///
/// Use [`dispatch_paged_attention_decode_v2`] for production — v1 is kept for
/// reference and debugging.
///
/// # Arguments
/// * `pipelines` — Compiled Metal pipelines
/// * `ep` — Encoder provider (pass `&command_buffer` from Candle for zero-sync)
/// * `q` — Query buffer `[batch, n_q_heads, head_dim]` as f32
/// * `q_offset` — Byte offset into q buffer
/// * `allocator` — Block allocator (to get layer buffer)
/// * `layer` — Which layer's KV buffer to read from
/// * `block_table_buf` — GPU buffer with `[batch, max_num_blocks]` block IDs (int32)
/// * `context_lens_buf` — GPU buffer with `[batch]` context lengths (int32)
/// * `output` — Output buffer `[batch, n_q_heads, head_dim]` as f32
/// * `out_offset` — Byte offset into output buffer
/// * `batch_size` — Number of sequences in the batch
/// * `n_q_heads` — Number of query heads
/// * `n_kv_heads` — Number of KV heads for this layer
/// * `head_dim` — Head dimension for this layer
/// * `block_size` — Tokens per block
/// * `max_num_blocks` — Max blocks per sequence (for block_table stride)
/// * `scale` — Attention scale factor
#[allow(clippy::too_many_arguments)]
pub fn dispatch_paged_attention_decode(
    pipelines: &PagedAttentionPipelines,
    ep: impl EncoderProvider,
    q: &Buffer,
    q_offset: usize,
    allocator: &BlockAllocator,
    layer: usize,
    block_table_buf: &Buffer,
    context_lens_buf: &Buffer,
    output: &Buffer,
    out_offset: usize,
    batch_size: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    max_num_blocks: u32,
    scale: f32,
) -> Result<()> {
    let pipeline = pipelines.get("paged_attention_decode")?;
    let kv_buf = allocator.layer_buffer(layer);

    let encoder = ep.encoder();
    let enc = encoder.as_ref();
    enc.set_compute_pipeline_state(pipeline);

    enc.set_input_buffer(0, Some(q), q_offset);
    enc.set_input_buffer(1, Some(kv_buf), 0);
    enc.set_input_buffer(2, Some(block_table_buf), 0);
    enc.set_input_buffer(3, Some(context_lens_buf), 0);
    enc.set_output_buffer(4, Some(output), out_offset);
    enc.set_bytes(5, &head_dim);
    enc.set_bytes(6, &block_size);
    enc.set_bytes(7, &n_q_heads);
    enc.set_bytes(8, &n_kv_heads);
    enc.set_bytes(9, &scale);
    enc.set_bytes(10, &max_num_blocks);

    let tg_size = 256u32;
    enc.dispatch_thread_groups(
        MTLSize {
            width: batch_size as usize,
            height: n_q_heads as usize,
            depth: 1,
        },
        MTLSize {
            width: tg_size as usize,
            height: 1,
            depth: 1,
        },
    );

    Ok(())
}

/// Dispatch paged attention decode v2 (FlashAttention-style, 1-pass).
///
/// Faster than v1: single pass over KV blocks using online softmax.
/// Requires head_dim ≤ 512 and block_size ≤ 64 (threadgroup memory limits).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_paged_attention_decode_v2(
    pipelines: &PagedAttentionPipelines,
    ep: impl EncoderProvider,
    q: &Buffer,
    q_offset: usize,
    allocator: &BlockAllocator,
    layer: usize,
    block_table_buf: &Buffer,
    context_lens_buf: &Buffer,
    output: &Buffer,
    out_offset: usize,
    batch_size: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    max_num_blocks: u32,
    scale: f32,
) -> Result<()> {
    let pipeline = pipelines.get("paged_attention_decode_v2")?;
    let kv_buf = allocator.layer_buffer(layer);

    let encoder = ep.encoder();
    let enc = encoder.as_ref();
    enc.set_compute_pipeline_state(pipeline);

    enc.set_input_buffer(0, Some(q), q_offset);
    enc.set_input_buffer(1, Some(kv_buf), 0);
    enc.set_input_buffer(2, Some(block_table_buf), 0);
    enc.set_input_buffer(3, Some(context_lens_buf), 0);
    enc.set_output_buffer(4, Some(output), out_offset);
    enc.set_bytes(5, &head_dim);
    enc.set_bytes(6, &block_size);
    enc.set_bytes(7, &n_q_heads);
    enc.set_bytes(8, &n_kv_heads);
    enc.set_bytes(9, &scale);
    enc.set_bytes(10, &max_num_blocks);

    // Threadgroup size: match head_dim for parallel output writes
    // Cap at 256 (threadgroup mem scratch size)
    let tg_size = head_dim.min(256);

    enc.dispatch_thread_groups(
        MTLSize {
            width: batch_size as usize,
            height: n_q_heads as usize,
            depth: 1,
        },
        MTLSize {
            width: tg_size as usize,
            height: 1,
            depth: 1,
        },
    );

    Ok(())
}

/// Dispatch write_kv_to_blocks for one layer.
///
/// Scatters K and V tensors into the correct paged block locations.
///
/// # Arguments
/// * `k_src` — K tensor `[seq_len, n_kv_heads, head_dim]` as f16
/// * `v_src` — V tensor (same layout as K)
#[allow(clippy::too_many_arguments)]
pub fn dispatch_write_kv_to_blocks(
    pipelines: &PagedAttentionPipelines,
    ep: impl EncoderProvider,
    k_src: &Buffer,
    k_offset: usize,
    v_src: &Buffer,
    v_offset: usize,
    allocator: &BlockAllocator,
    layer: usize,
    block_table_buf: &Buffer,
    seq_len: u32,
    start_pos: u32,
    n_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
) -> Result<()> {
    let pipeline = pipelines.get("write_kv_to_blocks")?;
    let kv_buf = allocator.layer_buffer(layer);

    let encoder = ep.encoder();
    let enc = encoder.as_ref();
    enc.set_compute_pipeline_state(pipeline);

    enc.set_input_buffer(0, Some(k_src), k_offset);
    enc.set_input_buffer(1, Some(v_src), v_offset);
    enc.set_output_buffer(2, Some(kv_buf), 0);
    enc.set_input_buffer(3, Some(block_table_buf), 0);
    enc.set_bytes(4, &seq_len);
    enc.set_bytes(5, &start_pos);
    enc.set_bytes(6, &block_size);
    enc.set_bytes(7, &n_kv_heads);
    enc.set_bytes(8, &head_dim);

    let tg_x = head_dim.min(256);
    enc.dispatch_threads(
        MTLSize {
            width: head_dim as usize,
            height: seq_len as usize,
            depth: n_kv_heads as usize,
        },
        MTLSize {
            width: tg_x as usize,
            height: 1,
            depth: 1,
        },
    );

    Ok(())
}

/// Dispatch write_kv_to_blocks_f32_candle for one layer.
///
/// Zero-copy variant: takes F32 K/V in Candle's `[1, n_kv, seq, dim]` layout,
/// converts to F16 inside the kernel before writing to paged blocks.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_write_kv_f32_candle(
    pipelines: &PagedAttentionPipelines,
    ep: impl EncoderProvider,
    k_src: &Buffer,
    k_offset: usize,
    v_src: &Buffer,
    v_offset: usize,
    allocator: &BlockAllocator,
    layer: usize,
    block_table_buf: &Buffer,
    seq_len: u32,
    start_pos: u32,
    n_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
) -> Result<()> {
    let pipeline = pipelines.get("write_kv_to_blocks_f32_candle")?;
    let kv_buf = allocator.layer_buffer(layer);

    let encoder = ep.encoder();
    let enc = encoder.as_ref();
    enc.set_compute_pipeline_state(pipeline);

    enc.set_input_buffer(0, Some(k_src), k_offset);
    enc.set_input_buffer(1, Some(v_src), v_offset);
    enc.set_output_buffer(2, Some(kv_buf), 0);
    enc.set_input_buffer(3, Some(block_table_buf), 0);
    enc.set_bytes(4, &seq_len);
    enc.set_bytes(5, &start_pos);
    enc.set_bytes(6, &block_size);
    enc.set_bytes(7, &n_kv_heads);
    enc.set_bytes(8, &head_dim);

    let tg_x = head_dim.min(256);
    enc.dispatch_threads(
        MTLSize {
            width: head_dim as usize,
            height: seq_len as usize,
            depth: n_kv_heads as usize,
        },
        MTLSize {
            width: tg_x as usize,
            height: 1,
            depth: 1,
        },
    );

    Ok(())
}

/// Dispatch copy_blocks for one layer.
///
/// Copies physical blocks for beam search forking or COW.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_copy_blocks(
    pipelines: &PagedAttentionPipelines,
    ep: impl EncoderProvider,
    allocator: &BlockAllocator,
    layer: usize,
    block_mapping_buf: &Buffer,
    num_pairs: u32,
    n_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
) -> Result<()> {
    let pipeline = pipelines.get("copy_blocks")?;
    let kv_buf = allocator.layer_buffer(layer);

    // Total elements per block: 2(K+V) * block_size * n_kv_heads * head_dim
    let block_elems = 2 * block_size * n_kv_heads * head_dim;

    let encoder = ep.encoder();
    let enc = encoder.as_ref();
    enc.set_compute_pipeline_state(pipeline);

    // copy_blocks reads from source pages and writes to dest pages within the same
    // `kv_buf`. Mark it output so the auto-barrier inserts a fence vs any prior write
    // to this layer's pages.
    enc.set_output_buffer(0, Some(kv_buf), 0);
    enc.set_input_buffer(1, Some(block_mapping_buf), 0);
    enc.set_bytes(2, &num_pairs);
    enc.set_bytes(3, &block_elems);

    let tg_x = block_elems.min(256);
    enc.dispatch_threads(
        MTLSize {
            width: block_elems as usize,
            height: num_pairs as usize,
            depth: 1,
        },
        MTLSize {
            width: tg_x as usize,
            height: 1,
            depth: 1,
        },
    );

    Ok(())
}
