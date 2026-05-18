//! PagedAttention — vLLM-style block-based KV cache management for Metal GPU.
//!
//! This crate implements the core infrastructure for continuous batching:
//!
//! - **Block Allocator**: Pre-allocated Metal buffer pool with free-list management
//! - **Page Table**: Per-sequence logical-to-physical block mapping
//! - **Sequence State**: Per-request lifecycle, tokens, and sampling params
//! - **Scheduler**: FCFS continuous batching with preemption support
//! - **Metal Kernels**: Paged attention decode/prefill/copy (Phase 3)
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────┐
//! │          Scheduler               │
//! │  waiting queue → running batch   │
//! ├──────────────────────────────────┤
//! │       Sequence Manager           │
//! │  SeqState { page_table, pos }    │
//! ├──────────────────────────────────┤
//! │       Block Allocator            │
//! │  Metal buffer pool + free list   │
//! ├──────────────────────────────────┤
//! │    Metal Paged Attention         │
//! │  decode / write_kv / copy        │
//! └──────────────────────────────────┘
//! ```

pub mod block_allocator;
pub mod block_table;
pub mod metal;
pub mod scheduler;
pub mod sequence;

// Re-export key types for convenience.
pub use block_allocator::{BlockAllocator, BlockConfig};
pub use block_table::PageTable;
pub use scheduler::{Scheduler, SchedulerConfig, SchedulerOutput, SchedulerStats};
pub use sequence::{SamplingParams, SeqEvent, SeqId, SeqState, SeqStatus};
