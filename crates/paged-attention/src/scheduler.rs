//! Scheduler — FCFS continuous batching with preemption support.
//!
//! Decides which sequences run in each iteration:
//! 1. Admit new sequences from `waiting` queue if blocks are available
//! 2. Classify running sequences as prefill or decode
//! 3. Preempt (swap out) lowest-priority sequences when OOM
//! 4. Clean up finished sequences

use std::collections::VecDeque;

use crate::block_allocator::BlockAllocator;
use crate::sequence::{SeqId, SeqState, SeqStatus};

/// Output of a scheduling step — tells the engine what to do.
pub struct SchedulerOutput {
    /// Sequences to run in this iteration (IDs index into `running`).
    pub scheduled_seq_ids: Vec<SeqId>,

    /// Which sequences are in prefill phase.
    pub prefill_seq_ids: Vec<SeqId>,

    /// Which sequences are in decode phase.
    pub decode_seq_ids: Vec<SeqId>,

    /// Block copy operations for beam search / COW: `(src_block, dst_block)`.
    pub blocks_to_copy: Vec<(u32, u32)>,
}

/// Configuration for the scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum number of sequences in a single batch.
    pub max_batch_size: usize,
    /// Maximum total tokens across all sequences in one forward pass.
    pub max_num_batched_tokens: usize,
    /// Tokens per block (must match BlockAllocator).
    pub block_size: u32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_num_batched_tokens: 2048,
            block_size: 16,
        }
    }
}

/// FCFS (First-Come, First-Served) continuous batching scheduler.
pub struct Scheduler {
    /// Sequences waiting to be admitted.
    waiting: VecDeque<SeqState>,

    /// Currently running sequences (in the active batch).
    running: Vec<SeqState>,

    /// Block allocator (shared with the engine).
    pub allocator: BlockAllocator,

    /// Scheduler configuration.
    config: SchedulerConfig,

    /// Monotonically increasing sequence ID counter.
    next_seq_id: SeqId,
}

impl Scheduler {
    /// Create a new scheduler with the given allocator and config.
    pub fn new(allocator: BlockAllocator, config: SchedulerConfig) -> Self {
        Self {
            waiting: VecDeque::new(),
            running: Vec::new(),
            allocator,
            config,
            next_seq_id: 0,
        }
    }

    /// Assign a new unique sequence ID.
    pub fn next_id(&mut self) -> SeqId {
        let id = self.next_seq_id;
        self.next_seq_id += 1;
        id
    }

    /// Add a new sequence to the waiting queue.
    pub fn add_sequence(&mut self, seq: SeqState) {
        self.waiting.push_back(seq);
    }

    /// Whether there are any sequences to process (waiting or running).
    pub fn is_empty(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    /// Total number of sequences (waiting + running).
    pub fn total_sequences(&self) -> usize {
        self.waiting.len() + self.running.len()
    }

    /// Run one scheduling step. Returns what the engine should execute.
    pub fn schedule(&mut self) -> SchedulerOutput {
        // 1. Clean up finished sequences, free their blocks
        self.cleanup_finished();

        // 2. Try to admit waiting sequences
        self.admit_waiting();

        // 3. Ensure all running sequences have enough blocks
        //    If OOM, preempt the most recently added sequence
        self.ensure_blocks_or_preempt();

        // 4. Classify into prefill vs decode
        let mut prefill_ids = Vec::new();
        let mut decode_ids = Vec::new();
        let mut all_ids = Vec::new();

        for seq in &self.running {
            all_ids.push(seq.id);
            if seq.is_prefill() {
                prefill_ids.push(seq.id);
            } else {
                decode_ids.push(seq.id);
            }
        }

        SchedulerOutput {
            scheduled_seq_ids: all_ids,
            prefill_seq_ids: prefill_ids,
            decode_seq_ids: decode_ids,
            blocks_to_copy: Vec::new(), // no COW yet
        }
    }

    /// Remove finished sequences from `running`.
    fn cleanup_finished(&mut self) {
        self.running.retain(|seq| !seq.is_finished());
    }

    /// Try to admit sequences from `waiting` into `running`.
    fn admit_waiting(&mut self) {
        while self.running.len() < self.config.max_batch_size {
            let Some(mut seq) = self.waiting.pop_front() else {
                break;
            };

            // Check if we have enough blocks for this sequence's prefill
            let needed_tokens = seq.pending_tokens.len();
            let needed_blocks = needed_tokens.div_ceil(self.config.block_size as usize);

            if self.allocator.num_free_blocks() < needed_blocks {
                // Not enough blocks — put it back and stop admitting
                self.waiting.push_front(seq);
                break;
            }

            seq.status = SeqStatus::Running;
            self.running.push(seq);
        }
    }

    /// Ensure all running sequences have allocated blocks.
    /// Preempt (requeue) the most recent sequence if OOM.
    fn ensure_blocks_or_preempt(&mut self) {
        // Try to allocate blocks for all running sequences
        let mut failed = Vec::new();
        for seq in &mut self.running {
            if let Err(_) = seq.ensure_blocks(&mut self.allocator, self.config.block_size) {
                failed.push(seq.id);
            }
        }

        // Preempt failed sequences: move back to waiting
        if !failed.is_empty() {
            let mut requeued = Vec::new();
            self.running.retain_mut(|seq| {
                if failed.contains(&seq.id) {
                    seq.status = SeqStatus::Waiting;
                    // Free any blocks we managed to allocate
                    seq.page_table.free_all(&mut self.allocator);
                    seq.context_len = 0;
                    seq.pending_tokens = seq.input_tokens.clone();
                    seq.output_tokens.clear();
                    requeued.push(seq.id);
                    false
                } else {
                    true
                }
            });

            // Move preempted sequences to front of waiting queue
            // (they were already partially processed)
            // Note: we can't easily move SeqState out of a retain_mut,
            // so for now preempted sequences are simply dropped and re-added.
            // TODO: implement proper swap-out in Phase 5+
        }
    }

    /// Get a reference to a running sequence by ID.
    pub fn get_seq(&self, id: SeqId) -> Option<&SeqState> {
        self.running.iter().find(|s| s.id == id)
    }

    /// Get a mutable reference to a running sequence by ID.
    pub fn get_seq_mut(&mut self, id: SeqId) -> Option<&mut SeqState> {
        self.running.iter_mut().find(|s| s.id == id)
    }

    /// Get all running sequences.
    pub fn running_seqs(&self) -> &[SeqState] {
        &self.running
    }

    /// Get mutable access to all running sequences.
    pub fn running_seqs_mut(&mut self) -> &mut Vec<SeqState> {
        &mut self.running
    }

    /// Number of waiting sequences.
    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    /// Number of running sequences.
    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    /// Report current utilization stats.
    pub fn stats(&self) -> SchedulerStats {
        let total_blocks = self.allocator.num_total_blocks() as usize;
        let free_blocks = self.allocator.num_free_blocks();
        SchedulerStats {
            num_waiting: self.waiting.len(),
            num_running: self.running.len(),
            blocks_used: total_blocks - free_blocks,
            blocks_total: total_blocks,
            blocks_free: free_blocks,
        }
    }
}

/// Scheduler utilization statistics.
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub num_waiting: usize,
    pub num_running: usize,
    pub blocks_used: usize,
    pub blocks_total: usize,
    pub blocks_free: usize,
}

impl std::fmt::Display for SchedulerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "waiting={} running={} blocks={}/{} ({:.1}% used)",
            self.num_waiting,
            self.num_running,
            self.blocks_used,
            self.blocks_total,
            if self.blocks_total > 0 {
                self.blocks_used as f64 / self.blocks_total as f64 * 100.0
            } else {
                0.0
            }
        )
    }
}
