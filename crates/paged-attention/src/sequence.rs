//! Sequence State — Per-request state for continuous batching.
//!
//! Each incoming request becomes a `SeqState` that tracks:
//! - Its page table (KV block mapping)
//! - Current generation position
//! - Input/output tokens
//! - Sampling parameters
//! - An optional channel for streaming deltas back to the caller

use tokio::sync::mpsc;

use crate::block_allocator::BlockAllocator;
use crate::block_table::PageTable;

/// Unique identifier for a sequence (request).
pub type SeqId = u64;

/// Lifecycle status of a sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeqStatus {
    /// Waiting in queue, not yet scheduled.
    Waiting,
    /// Currently running in the active batch (prefill or decode).
    Running,
    /// Generation complete (EOS or max_tokens reached).
    Finished,
    /// Preempted: KV blocks swapped to CPU (future use).
    Swapped,
}

/// Sampling parameters for a sequence.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
        }
    }
}

/// Event emitted by the paged engine to the caller.
#[derive(Debug, Clone)]
pub enum SeqEvent {
    /// A new text delta.
    Delta(String),
    /// Generation complete.
    Done {
        prompt_tokens: u32,
        completion_tokens: u32,
    },
    /// An error occurred.
    Error(String),
}

/// Per-request sequence state.
pub struct SeqState {
    /// Unique sequence ID.
    pub id: SeqId,

    /// Current lifecycle status.
    pub status: SeqStatus,

    /// Page table: logical→physical block mapping for this sequence.
    pub page_table: PageTable,

    /// Number of tokens whose KV are stored in the cache (= position for next token).
    pub context_len: usize,

    /// Original input token IDs (for prefill).
    pub input_tokens: Vec<u32>,

    /// Generated output token IDs so far.
    pub output_tokens: Vec<u32>,

    /// Tokens to process in the next forward pass.
    /// - Prefill: all input tokens
    /// - Decode: single token (last generated)
    pub pending_tokens: Vec<u32>,

    /// Sampling parameters.
    pub sampling: SamplingParams,

    /// Optional streaming channel. If `Some`, deltas are sent as they're generated.
    pub event_tx: Option<mpsc::Sender<SeqEvent>>,
}

impl SeqState {
    /// Create a new sequence from input tokens.
    pub fn new(
        id: SeqId,
        input_tokens: Vec<u32>,
        sampling: SamplingParams,
        event_tx: Option<mpsc::Sender<SeqEvent>>,
    ) -> Self {
        let pending_tokens = input_tokens.clone();
        Self {
            id,
            status: SeqStatus::Waiting,
            page_table: PageTable::new(),
            context_len: 0,
            input_tokens,
            output_tokens: Vec::new(),
            pending_tokens,
            sampling,
            event_tx,
        }
    }

    /// Ensure enough blocks are allocated for the next step.
    ///
    /// After this call, the page table has enough blocks to store
    /// `context_len + pending_tokens.len()` tokens.
    pub fn ensure_blocks(
        &mut self,
        allocator: &mut BlockAllocator,
        block_size: u32,
    ) -> anyhow::Result<()> {
        let total_tokens = self.context_len + self.pending_tokens.len();
        let needed_blocks = total_tokens.div_ceil(block_size as usize);
        while self.page_table.num_blocks() < needed_blocks {
            self.page_table.append_block(allocator)?;
        }
        Ok(())
    }

    /// Whether this sequence is in prefill phase (first forward pass).
    pub fn is_prefill(&self) -> bool {
        self.context_len == 0 && !self.pending_tokens.is_empty()
    }

    /// Whether generation is complete.
    pub fn is_finished(&self) -> bool {
        self.status == SeqStatus::Finished
    }

    /// Number of tokens generated so far.
    pub fn num_generated(&self) -> usize {
        self.output_tokens.len()
    }

    /// Check if generation should stop (max tokens or EOS).
    pub fn should_stop(&self, eos_token_id: u32) -> bool {
        if self.output_tokens.len() >= self.sampling.max_tokens {
            return true;
        }
        self.output_tokens.last().copied() == Some(eos_token_id)
    }

    /// Advance state after one decode step: push new token, update context.
    pub fn advance(&mut self, new_token: u32) {
        self.output_tokens.push(new_token);
        self.context_len += self.pending_tokens.len();
        self.pending_tokens = vec![new_token];
    }

    /// Mark this sequence as finished and free its blocks.
    pub fn finish(&mut self, allocator: &mut BlockAllocator) {
        self.status = SeqStatus::Finished;
        self.page_table.free_all(allocator);
        self.pending_tokens.clear();
    }

    /// Send a streaming event (non-blocking). Drops silently if receiver is gone.
    pub fn send_event(&self, event: SeqEvent) {
        if let Some(tx) = &self.event_tx {
            let _ = tx.try_send(event);
        }
    }
}
