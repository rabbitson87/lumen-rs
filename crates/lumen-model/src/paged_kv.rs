//! Paged KV cache backend for Gemma 4 GGUF models — zero-copy on Metal.
//!
//! multi-sequence state container. Internal state is a
//! `HashMap<SeqId, PerSeqState>` with a `current_seq_id` selector; the
//! `CompressedKVBackend` trait impl always operates on the current seq.
//! Explicit `new_seq` / `set_current_seq` / `remove_seq` pub methods let a
//! future engine-level scheduler (Phase 5c) drive seq switching before each
//! forward. Until then, the backend auto-rotates to a fresh SeqId when it
//! detects a cache reset (`cache_seq_len < current.seq_len`), preserving
//! single-seq behavior bit-for-bit.

use std::collections::HashMap;

use anyhow::Result;
use candle_core::{DType, Device, Storage, Tensor};
use candle_metal_kernels::metal::{Buffer, MTLResourceOptions};
use candle_transformers::models::quantized_gemma4::CompressedKVBackend;
use paged_attention::block_allocator::{BlockAllocator, BlockConfig, LayerAttentionConfig};
use paged_attention::block_table::PageTable;
use paged_attention::metal::dispatch::PagedAttentionPipelines;
use paged_attention::sequence::SeqId;

/// Per-sequence paged-KV state.
struct PerSeqState {
    /// Logical→physical block mapping for this sequence.
    page_table: PageTable,
    /// Committed sequence length (tokens whose KV are in paged blocks).
    seq_len: usize,
    /// Tokens written in the current sweep, not yet committed.
    pending_tokens: usize,
    /// Last layer seen (sweep detection).
    last_op_layer: Option<usize>,
}

impl PerSeqState {
    fn new() -> Self {
        Self {
            page_table: PageTable::new(),
            seq_len: 0,
            pending_tokens: 0,
            last_op_layer: None,
        }
    }

    fn effective_seq_len(&self) -> usize {
        self.seq_len + self.pending_tokens
    }
}

/// Paged KV backend with per-sequence state.
///
/// The active sequence is selected by `current_seq_id`. The `CompressedKVBackend`
/// trait methods (`store_kv`, `compressed_attention`, `seq_len`, `clear`) all
/// operate on the current seq. Phase 5c will call `set_current_seq` between
/// scheduled steps to switch context; until then, a cache reset auto-rotates.
pub struct PagedKVBackend {
    device: Device,
    pipelines: PagedAttentionPipelines,
    allocator: BlockAllocator,

    /// Per-sequence state.
    seqs: HashMap<SeqId, PerSeqState>,
    /// Currently active sequence. Auto-created on first `store_kv` if unset.
    current_seq_id: SeqId,
    /// Monotonic SeqId generator for auto-allocated sequences.
    next_auto_id: u64,

    /// Pre-allocated padded block-table buffer (re-uploaded per current seq).
    /// Reserved for the PagedAttention scheduler path (Phase 3); not wired
    /// into the current single-seq dispatch yet.
    #[allow(dead_code)]
    block_table_buf: Buffer,
    /// Pre-allocated context_lens buffer (1 element for single active seq).
    /// Same Phase-3 scaffolding as `block_table_buf`.
    #[allow(dead_code)]
    context_lens_buf: Buffer,
    /// Max blocks per sequence (GPU block_table stride).
    max_num_blocks: u32,
    /// Max sequences in a single batched dispatch (block_table/context_lens rows).
    max_batch: u32,
}

impl PagedKVBackend {
    pub fn new(
        device: Device,
        gpu_memory_budget_mb: usize,
        block_size: u32,
        n_layers: u32,
        n_kv_heads: u32,
        head_dim_sliding: u32,
        head_dim_global: u32,
        global_every: u32,
    ) -> Result<Self> {
        let layers: Vec<LayerAttentionConfig> = (0..n_layers)
            .map(|i| {
                let is_sliding = (i + 1) % global_every != 0;
                LayerAttentionConfig {
                    n_kv_heads,
                    head_dim: if is_sliding {
                        head_dim_sliding
                    } else {
                        head_dim_global
                    },
                    is_sliding,
                }
            })
            .collect();
        let config = BlockConfig::from_layers(block_size, layers);

        let md = match &device {
            Device::Metal(md) => md.clone(),
            _ => anyhow::bail!("PagedKVBackend requires a Metal device"),
        };
        let metal_dev = md.metal_device();
        let pipelines = PagedAttentionPipelines::new(metal_dev)?;

        let budget_bytes = gpu_memory_budget_mb * 1024 * 1024;
        let allocator = BlockAllocator::new(metal_dev, budget_bytes, config)?;
        let max_num_blocks = allocator.num_total_blocks();

        // Pre-allocate metadata buffers sized for the max batch. Single-seq
        // dispatches read only row 0 of the stacked block table, so oversizing
        // is harmless.
        let max_batch: u32 = std::env::var("PAGED_MAX_BATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let block_table_buf = metal_dev
            .new_buffer(
                (max_batch as usize) * (max_num_blocks as usize) * 4,
                MTLResourceOptions::StorageModeShared,
            )
            .map_err(|e| anyhow::anyhow!("block_table buffer alloc: {e}"))?;
        let context_lens_buf = metal_dev
            .new_buffer(
                (max_batch as usize) * 4,
                MTLResourceOptions::StorageModeShared,
            )
            .map_err(|e| anyhow::anyhow!("context_lens buffer alloc: {e}"))?;

        // Start with one default sequence (id=0). Engine can replace via
        // set_current_seq before issuing a forward; auto-rotation on cache
        // reset preserves the old single-seq behavior when engine isn't wired.
        let mut seqs = HashMap::new();
        seqs.insert(0, PerSeqState::new());

        Ok(Self {
            device,
            pipelines,
            allocator,
            seqs,
            current_seq_id: 0,
            next_auto_id: 1,
            block_table_buf,
            context_lens_buf,
            max_num_blocks,
            max_batch,
        })
    }

    // ─── Phase 5a public seq-lifecycle API ────────────────────────────────

    /// Allocate an empty state slot for `seq_id` (does not activate it).
    ///
    /// No-op if the seq already exists. Use `set_current_seq` to activate.
    pub fn new_seq(&mut self, seq_id: SeqId) {
        self.seqs.entry(seq_id).or_insert_with(PerSeqState::new);
    }

    /// Switch the active sequence. Creates the slot if it doesn't exist.
    ///
    /// Call this *before* the forward pass of seq `seq_id`. `store_kv` and
    /// `compressed_attention` operate on this seq's page table and lengths.
    pub fn set_current_seq(&mut self, seq_id: SeqId) {
        self.new_seq(seq_id);
        if self.current_seq_id != seq_id {
            self.current_seq_id = seq_id;
            // Upload new seq's block table lazily on next sweep start.
        }
    }

    /// Free a sequence's blocks and remove it from the state map.
    pub fn remove_seq(&mut self, seq_id: SeqId) {
        if let Some(mut st) = self.seqs.remove(&seq_id) {
            st.page_table.free_all(&mut self.allocator);
        }
        if self.current_seq_id == seq_id {
            // Fall back to a fresh default seq so the next forward is valid.
            let new_id = self.alloc_seq_id();
            self.seqs.insert(new_id, PerSeqState::new());
            self.current_seq_id = new_id;
        }
    }

    /// Current active SeqId.
    pub fn current_seq_id(&self) -> SeqId {
        self.current_seq_id
    }

    /// Number of live sequences (including empty ones).
    pub fn num_seqs(&self) -> usize {
        self.seqs.len()
    }

    fn alloc_seq_id(&mut self) -> SeqId {
        let id = self.next_auto_id;
        self.next_auto_id = self.next_auto_id.wrapping_add(1);
        id
    }

    // ─── Internal helpers (operate on current seq) ────────────────────────

    fn current(&self) -> &PerSeqState {
        self.seqs
            .get(&self.current_seq_id)
            .expect("current seq must exist")
    }

    fn current_mut(&mut self) -> &mut PerSeqState {
        self.seqs
            .get_mut(&self.current_seq_id)
            .expect("current seq must exist")
    }

    fn maybe_start_new_sweep(&mut self, layer: usize) {
        let st = self.current_mut();
        let is_new_sweep = match st.last_op_layer {
            Some(prev) => layer < prev,
            None => false,
        };
        if is_new_sweep && st.pending_tokens > 0 {
            st.seq_len += st.pending_tokens;
            st.pending_tokens = 0;
        }
    }

    fn ensure_blocks_for(&mut self, n_new_tokens: usize) -> Result<()> {
        let block_size = self.allocator.config.block_size as usize;
        let total = self.current().seq_len + n_new_tokens;
        let needed_blocks = total.div_ceil(block_size);
        let allocator = &mut self.allocator;
        let pt = &mut self
            .seqs
            .get_mut(&self.current_seq_id)
            .expect("current seq")
            .page_table;
        while pt.num_blocks() < needed_blocks {
            pt.append_block(allocator)?;
        }
        Ok(())
    }

    /// Allocate a fresh Metal buffer containing the current seq's page table.
    ///
    /// **Why fresh per dispatch**: Candle's command-encoder pool batches up to
    /// 50 encoders into a single command buffer, so a *shared* block-table
    /// buffer that we host-write before each dispatch gets read by all queued
    /// kernels at commit time — every kernel sees the FINAL state, not its
    /// own. Allocating a per-dispatch Buffer keeps each kernel's metadata
    /// alive independently (Metal ref-counts bound buffers until their cmd
    /// buffer completes).
    fn fresh_block_table_buf(&self) -> Option<Buffer> {
        let md = match &self.device {
            Device::Metal(md) => md,
            _ => return None,
        };
        let mdev = md.metal_device();
        let stride = self.max_num_blocks as usize;
        let buf = mdev
            .new_buffer(stride * 4, MTLResourceOptions::StorageModeShared)
            .ok()?;
        let pt = &self.current().page_table;
        let ptr = buf.contents() as *mut i32;
        unsafe {
            let slice = std::slice::from_raw_parts_mut(ptr, stride);
            for (i, slot) in slice.iter_mut().enumerate() {
                *slot = if i < pt.num_blocks() {
                    pt.get(i) as i32
                } else {
                    0
                };
            }
        }
        Some(buf)
    }

    /// Fresh per-dispatch context-lens buffer for single-seq kernels (4 bytes).
    fn fresh_context_len_buf(&self, ctx_len: i32) -> Option<Buffer> {
        let md = match &self.device {
            Device::Metal(md) => md,
            _ => return None,
        };
        let buf = md
            .metal_device()
            .new_buffer(4, MTLResourceOptions::StorageModeShared)
            .ok()?;
        unsafe {
            *(buf.contents() as *mut i32) = ctx_len;
        }
        Some(buf)
    }

    /// Fresh batched block-table buffer `[N_seqs × max_num_blocks] i32`.
    fn fresh_block_table_buf_for_seqs(&self, seq_ids: &[SeqId]) -> Option<Buffer> {
        let md = match &self.device {
            Device::Metal(md) => md,
            _ => return None,
        };
        let stride = self.max_num_blocks as usize;
        let n = seq_ids.len();
        let buf = md
            .metal_device()
            .new_buffer(n * stride * 4, MTLResourceOptions::StorageModeShared)
            .ok()?;
        let ptr = buf.contents() as *mut i32;
        unsafe {
            let slice = std::slice::from_raw_parts_mut(ptr, n * stride);
            for (row, seq_id) in seq_ids.iter().enumerate() {
                let base = row * stride;
                let pt = self.seqs.get(seq_id).map(|s| &s.page_table);
                for i in 0..stride {
                    slice[base + i] = match pt {
                        Some(pt) if i < pt.num_blocks() => pt.get(i) as i32,
                        _ => 0,
                    };
                }
            }
        }
        Some(buf)
    }

    /// Fresh batched context-lens buffer `[N_seqs] i32`.
    fn fresh_context_lens_buf_for_seqs(&self, seq_ids: &[SeqId]) -> Option<Buffer> {
        let md = match &self.device {
            Device::Metal(md) => md,
            _ => return None,
        };
        let n = seq_ids.len();
        let buf = md
            .metal_device()
            .new_buffer(n * 4, MTLResourceOptions::StorageModeShared)
            .ok()?;
        let ptr = buf.contents() as *mut i32;
        unsafe {
            let slice = std::slice::from_raw_parts_mut(ptr, n);
            for (row, seq_id) in seq_ids.iter().enumerate() {
                slice[row] = self
                    .seqs
                    .get(seq_id)
                    .map(|s| s.effective_seq_len() as i32)
                    .unwrap_or(0);
            }
        }
        Some(buf)
    }

    // ─── Batched primitive (Phase 5b) ─────────────────────────────────────

    /// Batched decode attention across multiple sequences for one layer.
    ///
    /// - `q_batched`: shape `[N_seqs, n_head, 1, head_dim]` F32 (stacked along
    ///   dim 0; row `i` is seq `seq_ids[i]`'s single decode query).
    /// - `seq_ids`: sequences to attend to, each pre-populated via
    ///   `store_kv_for_seq` at this layer in the current sweep.
    ///
    /// Returns an output tensor shaped `[N_seqs, n_head, 1, head_dim]` F32, or
    /// `None` if the batch violates `max_batch` or any seq is missing.
    ///
    /// NOTE: This primitive dispatches exactly one kernel call with
    /// `batch_size = seq_ids.len()`. The host stacks block tables + context
    /// lengths; the kernel (already batch-aware) picks the right seq per
    /// `blockIdx.x`.
    pub fn compressed_attention_batched(
        &mut self,
        layer: usize,
        q_batched: &Tensor,
        seq_ids: &[SeqId],
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
    ) -> Option<Tensor> {
        let n_seqs = seq_ids.len();
        if n_seqs == 0 || n_seqs as u32 > self.max_batch {
            return None;
        }
        let dims = q_batched.dims();
        if dims.len() != 4 || dims[0] != n_seqs || dims[2] != 1 {
            return None;
        }

        // Validate layer geometry once.
        let (lc_n_kv, lc_dim) = {
            let lc = &self.allocator.config.layer_configs[layer];
            (lc.n_kv_heads, lc.head_dim)
        };
        if head_dim as u32 != lc_dim || n_kv_head as u32 != lc_n_kv {
            return None;
        }

        // All seqs must exist and have non-empty KV.
        for id in seq_ids {
            let st = self.seqs.get(id)?;
            if st.effective_seq_len() == 0 {
                return None;
            }
        }

        let bt_buf = self.fresh_block_table_buf_for_seqs(seq_ids)?;
        let cl_buf = self.fresh_context_lens_buf_for_seqs(seq_ids)?;

        let q_f32 = q_batched.to_dtype(DType::F32).ok()?.contiguous().ok()?;
        let (q_buf, q_off) = tensor_metal_buffer(&q_f32)?;

        let out_tensor =
            Tensor::zeros((n_seqs, n_head, 1, head_dim), DType::F32, &self.device).ok()?;
        let (out_buf, out_off) = tensor_metal_buffer(&out_tensor)?;

        let scale = 1.0f32;

        let md = match &self.device {
            Device::Metal(md) => md,
            _ => return None,
        };
        let encoder = md.command_encoder().ok()?;

        paged_attention::metal::dispatch_paged_attention_decode_v2(
            &self.pipelines,
            encoder.as_ref(),
            &q_buf,
            q_off,
            &self.allocator,
            layer,
            &bt_buf,
            &cl_buf,
            &out_buf,
            out_off,
            n_seqs as u32,
            n_head as u32,
            n_kv_head as u32,
            head_dim as u32,
            self.allocator.config.block_size,
            self.max_num_blocks,
            scale,
        )
        .ok()?;
        drop(encoder);
        if std::env::var("PAGED_KV_SYNC").is_ok() {
            md.wait_until_completed().ok()?;
        }

        // Advance last_op_layer bookkeeping for each seq (sweep tracking).
        for id in seq_ids {
            if let Some(st) = self.seqs.get_mut(id) {
                st.last_op_layer = Some(layer);
            }
        }

        Some(out_tensor)
    }

    /// Seq-addressed K/V store (Phase 5b engine-driven path).
    ///
    /// Caller sets the target seq explicitly; `self.current_seq_id` is
    /// temporarily swapped during the write, then restored. Mirrors the
    /// logic of the trait-impl `store_kv` but without auto-rotation.
    pub fn store_kv_for_seq(
        &mut self,
        seq_id: SeqId,
        layer: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> bool {
        self.new_seq(seq_id);
        let prev = self.current_seq_id;
        self.current_seq_id = seq_id;
        let ok = CompressedKVBackend::store_kv(self, layer, k, v);
        // For engine-driven batching, the engine decides who's "current"
        // between layers; leave current_seq_id set to this seq so the
        // subsequent compressed_attention_batched call doesn't surprise
        // single-seq callers that also use the trait interface.
        let _ = prev;
        ok
    }

    /// Detect a cache reset (new request under single-seq auto mode) and
    /// rotate to a fresh SeqId. Called from `store_kv` when `cache_seq_len`
    /// drops below the current seq's committed length.
    fn auto_rotate_on_reset(&mut self) {
        // Free the old seq to reclaim blocks — it's been replaced by a new
        // request with an unrelated KV history.
        let old_id = self.current_seq_id;
        if let Some(mut old) = self.seqs.remove(&old_id) {
            old.page_table.free_all(&mut self.allocator);
        }
        let new_id = self.alloc_seq_id();
        self.seqs.insert(new_id, PerSeqState::new());
        self.current_seq_id = new_id;
    }
}

/// Extract `(Buffer, byte_offset)` from a Candle tensor on Metal (zero-copy).
fn tensor_metal_buffer(t: &Tensor) -> Option<(Buffer, usize)> {
    let (storage_guard, layout) = t.storage_and_layout();
    match &*storage_guard {
        Storage::Metal(ms) => {
            let offset_bytes = layout.start_offset() * t.dtype().size_in_bytes();
            Some((ms.buffer().clone(), offset_bytes))
        }
        _ => None,
    }
}

impl CompressedKVBackend for PagedKVBackend {
    fn store_kv(&mut self, layer: usize, k: &candle_core::Tensor, v: &candle_core::Tensor) -> bool {
        let dims = k.dims();
        if dims.len() != 4 {
            return false;
        }
        let (n_kv_head, cache_seq_len, head_dim) = (dims[1], dims[2], dims[3]);

        self.maybe_start_new_sweep(layer);

        // Cache reset → rotate to a fresh SeqId (preserves old single-seq
        // behavior when engine isn't wiring explicit new_seq/remove_seq).
        if cache_seq_len < self.current().seq_len {
            self.auto_rotate_on_reset();
        }

        let (lc_n_kv, lc_dim, lc_block_size) = {
            let lc = &self.allocator.config.layer_configs[layer];
            (lc.n_kv_heads, lc.head_dim, self.allocator.config.block_size)
        };
        if head_dim as u32 != lc_dim || n_kv_head as u32 != lc_n_kv {
            return false;
        }

        let seq_len_before = self.current().seq_len;
        let n_new = match cache_seq_len.checked_sub(seq_len_before) {
            Some(n) if n > 0 => n,
            _ => {
                self.current_mut().last_op_layer = Some(layer);
                return true;
            }
        };

        // First layer of a sweep → allocate blocks for the new tokens.
        if self.current().pending_tokens == 0 {
            if let Err(e) = self.ensure_blocks_for(n_new) {
                eprintln!("PagedKV: block allocation failed: {e}");
                return false;
            }
        }
        // Fresh per-dispatch block-table buffer (see `fresh_block_table_buf`
        // for race rationale).
        let bt_buf = match self.fresh_block_table_buf() {
            Some(b) => b,
            None => {
                eprintln!("PagedKV: fresh_block_table_buf failed");
                return false;
            }
        };

        let k_new = match k
            .narrow(2, seq_len_before, n_new)
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.contiguous())
        {
            Ok(t) => t,
            Err(e) => {
                eprintln!("PagedKV: k narrow/normalize failed: {e}");
                return false;
            }
        };
        let v_new = match v
            .narrow(2, seq_len_before, n_new)
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.contiguous())
        {
            Ok(t) => t,
            Err(e) => {
                eprintln!("PagedKV: v narrow/normalize failed: {e}");
                return false;
            }
        };
        let (k_buf, k_off) = match tensor_metal_buffer(&k_new) {
            Some(x) => x,
            None => return false,
        };
        let (v_buf, v_off) = match tensor_metal_buffer(&v_new) {
            Some(x) => x,
            None => return false,
        };

        let start_pos = seq_len_before as u32;

        let md = match &self.device {
            Device::Metal(md) => md,
            _ => return false,
        };
        let encoder = match md.command_encoder() {
            Ok(e) => e,
            Err(e) => {
                eprintln!("PagedKV: command_encoder failed: {e}");
                return false;
            }
        };

        if let Err(e) = paged_attention::metal::dispatch_write_kv_f32_candle(
            &self.pipelines,
            encoder.as_ref(),
            &k_buf,
            k_off,
            &v_buf,
            v_off,
            &self.allocator,
            layer,
            &bt_buf,
            n_new as u32,
            start_pos,
            lc_n_kv,
            lc_dim,
            lc_block_size,
        ) {
            eprintln!("PagedKV: write_kv_f32_candle dispatch failed: {e}");
            return false;
        }
        drop(encoder);
        if std::env::var("PAGED_KV_SYNC").is_ok() {
            let _ = md.wait_until_completed();
        }

        let st = self.current_mut();
        st.pending_tokens = n_new;
        st.last_op_layer = Some(layer);
        true
    }

    fn compressed_attention(
        &mut self,
        layer: usize,
        q: &candle_core::Tensor,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
    ) -> Option<candle_core::Tensor> {
        let dims = q.dims();
        if dims.len() != 4 || dims[2] != 1 {
            return None;
        }

        self.maybe_start_new_sweep(layer);
        self.current_mut().last_op_layer = Some(layer);

        let (lc_n_kv, lc_dim) = {
            let lc = &self.allocator.config.layer_configs[layer];
            (lc.n_kv_heads, lc.head_dim)
        };
        if head_dim as u32 != lc_dim || n_kv_head as u32 != lc_n_kv {
            return None;
        }
        let eff_seq_len = self.current().effective_seq_len();
        if eff_seq_len == 0 {
            return None;
        }

        // Fresh per-dispatch metadata (see `fresh_block_table_buf` rationale).
        let bt_buf = self.fresh_block_table_buf()?;
        let cl_buf = self.fresh_context_len_buf(eff_seq_len as i32)?;

        let q_f32 = q.to_dtype(DType::F32).ok()?.contiguous().ok()?;
        let (q_buf, q_off) = tensor_metal_buffer(&q_f32)?;

        let out_tensor = Tensor::zeros((1, n_head, 1, head_dim), DType::F32, &self.device).ok()?;
        let (out_buf, out_off) = tensor_metal_buffer(&out_tensor)?;

        let scale = 1.0f32;

        let md = match &self.device {
            Device::Metal(md) => md,
            _ => return None,
        };
        let encoder = md.command_encoder().ok()?;

        paged_attention::metal::dispatch_paged_attention_decode_v2(
            &self.pipelines,
            encoder.as_ref(),
            &q_buf,
            q_off,
            &self.allocator,
            layer,
            &bt_buf,
            &cl_buf,
            &out_buf,
            out_off,
            1, // batch_size — multi-seq batching is Phase 5b
            n_head as u32,
            n_kv_head as u32,
            head_dim as u32,
            self.allocator.config.block_size,
            self.max_num_blocks,
            scale,
        )
        .ok()?;
        drop(encoder);
        if std::env::var("PAGED_KV_SYNC").is_ok() {
            md.wait_until_completed().ok()?;
        }

        if std::env::var("PAGED_KV_DEBUG").is_ok() {
            md.wait_until_completed().ok()?;
            let ptr = out_buf.contents() as *const f32;
            let n = (n_head * head_dim).min(8);
            let sample: Vec<f32> = (0..n).map(|i| unsafe { *ptr.add(i) }).collect();
            let all_n = n_head * head_dim;
            let full: Vec<f32> = (0..all_n).map(|i| unsafe { *ptr.add(i) }).collect();
            let max_abs = full.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let any_nan = full.iter().any(|x| x.is_nan());
            eprintln!(
                "PAGED_KV seq={} L{} eff_seq={} max_abs={:.4} nan={} first8={:?}",
                self.current_seq_id, layer, eff_seq_len, max_abs, any_nan, sample,
            );
        }

        Some(out_tensor)
    }

    fn seq_len(&self, _layer: usize) -> usize {
        self.current().effective_seq_len()
    }

    /// Clears the *current* sequence only. Other seqs' state is preserved.
    /// For full-backend wipe, iterate seqs + call `remove_seq` (or drop).
    fn clear(&mut self) {
        let allocator = &mut self.allocator;
        let st = self
            .seqs
            .get_mut(&self.current_seq_id)
            .expect("current seq");
        st.page_table.free_all(allocator);
        st.seq_len = 0;
        st.pending_tokens = 0;
        st.last_op_layer = None;
    }

    // ── Multi-sequence trait overrides (Phase 5) ─────────────────────────

    fn new_seq(&mut self, seq_id: u64) {
        PagedKVBackend::new_seq(self, seq_id);
    }

    fn set_current_seq(&mut self, seq_id: u64) {
        PagedKVBackend::set_current_seq(self, seq_id);
    }

    fn remove_seq(&mut self, seq_id: u64) {
        PagedKVBackend::remove_seq(self, seq_id);
    }

    fn seq_len_for(&self, seq_id: u64, _layer: usize) -> usize {
        self.seqs
            .get(&seq_id)
            .map(|s| s.effective_seq_len())
            .unwrap_or(0)
    }

    fn store_kv_for_seq(&mut self, seq_id: u64, layer: usize, k: &Tensor, v: &Tensor) -> bool {
        PagedKVBackend::store_kv_for_seq(self, seq_id, layer, k, v)
    }

    fn compressed_attention_batched(
        &mut self,
        layer: usize,
        q: &Tensor,
        seq_ids: &[u64],
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
    ) -> Option<Tensor> {
        PagedKVBackend::compressed_attention_batched(
            self, layer, q, seq_ids, n_head, n_kv_head, head_dim,
        )
    }
}
