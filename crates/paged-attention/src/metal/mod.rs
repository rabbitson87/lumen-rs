//! Metal GPU kernels for PagedAttention.
//!
//! Three core kernels:
//! - `paged_attention_decode`: Single-query attention over paged KV blocks
//! - `write_kv_to_blocks`: Write K/V tensors into paged blocks after projection
//! - `copy_blocks`: Block-level copy for beam search / COW

pub mod dispatch;

pub use dispatch::{
    PagedAttentionPipelines, dispatch_copy_blocks, dispatch_paged_attention_decode,
    dispatch_paged_attention_decode_v2, dispatch_write_kv_f32_candle, dispatch_write_kv_to_blocks,
};
