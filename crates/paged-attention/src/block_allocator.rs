//! Block Allocator — Pre-allocated Metal buffer pool with free list.
//!
//! Manages **per-layer** Metal buffers for paged KV cache, supporting
//! heterogeneous head dimensions (Gemma 4 31B: sliding=256, global=512).
//!
//! Each layer gets its own Metal buffer sized exactly for its actual
//! `n_kv_heads × head_dim`, eliminating memory waste from uniform max-dim
//! allocation. Block IDs are shared across layers — one logical block
//! corresponds to one slot in every layer's buffer.
//!
//! Memory layout:
//! ```text
//! Layer 0 buffer (sliding, head_dim=256):
//! ┌───────────┬───────────┬─────┐
//! │  Block 0  │  Block 1  │ ... │  each: K[16,8,256]f16 + V[16,8,256]f16
//! └───────────┴───────────┴─────┘
//!
//! Layer 1 buffer (global, head_dim=512):
//! ┌───────────┬───────────┬─────┐
//! │  Block 0  │  Block 1  │ ... │  each: K[16,8,512]f16 + V[16,8,512]f16
//! └───────────┴───────────┴─────┘
//! ```

use anyhow::{Result, bail};
use candle_metal_kernels::metal::{Buffer, Device, MTLResourceOptions};

/// Per-layer attention configuration (supports heterogeneous Gemma 4).
#[derive(Debug, Clone)]
pub struct LayerAttentionConfig {
    /// Number of KV attention heads for this layer.
    pub n_kv_heads: u32,
    /// Head dimension for this layer (256 for sliding, 512 for global in 31B).
    pub head_dim: u32,
    /// Whether this layer uses sliding window attention.
    pub is_sliding: bool,
}

impl LayerAttentionConfig {
    /// Bytes per block for this layer: block_size × n_kv_heads × head_dim × 2(f16) × 2(K+V).
    pub fn block_bytes(&self, block_size: u32) -> usize {
        block_size as usize
            * self.n_kv_heads as usize
            * self.head_dim as usize
            * 2 // f16
            * 2 // K + V
    }

    /// Bytes for K (or V) of one token for this layer.
    pub fn token_kv_bytes(&self) -> usize {
        self.n_kv_heads as usize * self.head_dim as usize * 2 // f16
    }
}

/// Configuration for the block allocator.
///
/// Gemma 4 31B (48 layers, heterogeneous):
/// ```text
/// sliding: 16 × 8 × 256 × 2 × 2 =  128KB/block/layer
/// global:  16 × 8 × 512 × 2 × 2 =  256KB/block/layer
/// total per block ≈ 8.25MB (vs 12MB uniform — 31% savings)
/// 2GB budget → ~248 blocks → 3,968 tokens
/// ```
#[derive(Debug, Clone)]
pub struct BlockConfig {
    /// Tokens per block (vLLM default: 16).
    pub block_size: u32,
    /// Per-layer attention configs.
    pub layer_configs: Vec<LayerAttentionConfig>,
}

impl BlockConfig {
    /// Create config from per-layer specs.
    pub fn from_layers(block_size: u32, layer_configs: Vec<LayerAttentionConfig>) -> Self {
        Self {
            block_size,
            layer_configs,
        }
    }

    /// Create uniform config (all layers identical — e.g., E4B).
    pub fn uniform(block_size: u32, n_layers: u32, n_kv_heads: u32, head_dim: u32) -> Self {
        let layer_configs = (0..n_layers)
            .map(|_| LayerAttentionConfig {
                n_kv_heads,
                head_dim,
                is_sliding: false,
            })
            .collect();
        Self {
            block_size,
            layer_configs,
        }
    }

    /// Number of layers.
    pub fn n_layers(&self) -> usize {
        self.layer_configs.len()
    }

    /// Total bytes per physical block across ALL layers (sum of per-layer sizes).
    pub fn block_bytes_total(&self) -> usize {
        self.layer_configs
            .iter()
            .map(|lc| lc.block_bytes(self.block_size))
            .sum()
    }

    /// Whether the model has heterogeneous head dimensions across layers.
    pub fn is_heterogeneous(&self) -> bool {
        if self.layer_configs.is_empty() {
            return false;
        }
        let first = &self.layer_configs[0];
        self.layer_configs
            .iter()
            .any(|l| l.head_dim != first.head_dim || l.n_kv_heads != first.n_kv_heads)
    }
}

/// Pre-allocated Metal buffer pool for paged KV cache.
///
/// Each layer has its own Metal buffer, sized for that layer's actual
/// `n_kv_heads × head_dim`. Block IDs index uniformly across all layers.
pub struct BlockAllocator {
    /// Per-layer GPU buffers. `kv_pools[layer]` holds all blocks for that layer.
    kv_pools: Vec<Buffer>,

    /// Per-layer block size in bytes.
    layer_block_bytes: Vec<usize>,

    /// Stack of free physical block indices (shared across layers).
    free_blocks: Vec<u32>,

    /// Total number of physical blocks.
    num_blocks: u32,

    /// Block configuration.
    pub config: BlockConfig,
}

impl BlockAllocator {
    /// Create a new block allocator with a fixed GPU memory budget.
    ///
    /// Allocates per-layer Metal buffers with exact dimensions.
    /// The block count is determined by `budget / sum(per_layer_block_bytes)`.
    pub fn new(
        device: &Device,
        gpu_memory_budget_bytes: usize,
        config: BlockConfig,
    ) -> Result<Self> {
        let block_total_bytes = config.block_bytes_total();
        if block_total_bytes == 0 {
            bail!("block size computes to 0 bytes");
        }

        let num_blocks = (gpu_memory_budget_bytes / block_total_bytes) as u32;
        if num_blocks == 0 {
            bail!(
                "GPU memory budget ({} MB) too small for even 1 block ({} KB across {} layers)",
                gpu_memory_budget_bytes / (1024 * 1024),
                block_total_bytes / 1024,
                config.n_layers(),
            );
        }

        // Allocate per-layer buffers
        let mut kv_pools = Vec::with_capacity(config.n_layers());
        let mut layer_block_bytes = Vec::with_capacity(config.n_layers());
        let mut total_allocated: u64 = 0;

        for (layer_idx, lc) in config.layer_configs.iter().enumerate() {
            let lb = lc.block_bytes(config.block_size);
            let pool_bytes = num_blocks as usize * lb;
            let buf = device
                .new_buffer(pool_bytes, MTLResourceOptions::StorageModeShared)
                .map_err(|e| {
                    anyhow::anyhow!("failed to allocate KV pool for layer {layer_idx}: {e}")
                })?;
            kv_pools.push(buf);
            layer_block_bytes.push(lb);
            total_allocated += pool_bytes as u64;
        }

        let free_blocks: Vec<u32> = (0..num_blocks).rev().collect();

        let pool_mb = total_allocated as f64 / (1024.0 * 1024.0);
        let max_tokens = num_blocks as usize * config.block_size as usize;
        eprintln!(
            "PagedAttention: {} blocks ({:.1} MB), block_size={}, max_tokens={}, {} layers{}",
            num_blocks,
            pool_mb,
            config.block_size,
            max_tokens,
            config.n_layers(),
            if config.is_heterogeneous() {
                " (heterogeneous per-layer buffers)"
            } else {
                ""
            },
        );

        Ok(Self {
            kv_pools,
            layer_block_bytes,
            free_blocks,
            num_blocks,
            config,
        })
    }

    /// Allocate a single physical block. Returns `None` if no blocks available.
    pub fn allocate(&mut self) -> Option<u32> {
        self.free_blocks.pop()
    }

    /// Allocate `n` block IDs. Returns `None` if insufficient blocks.
    pub fn allocate_n(&mut self, n: usize) -> Option<Vec<u32>> {
        if self.free_blocks.len() < n {
            return None;
        }
        let blocks: Vec<u32> = (0..n).map(|_| self.free_blocks.pop().unwrap()).collect();
        Some(blocks)
    }

    /// Free a single physical block back to the pool.
    pub fn free(&mut self, block_id: u32) {
        debug_assert!(block_id < self.num_blocks, "invalid block_id {block_id}");
        self.free_blocks.push(block_id);
    }

    /// Free multiple blocks back to the pool.
    pub fn free_many(&mut self, block_ids: &[u32]) {
        for &id in block_ids {
            self.free(id);
        }
    }

    /// Number of currently free blocks.
    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Total number of physical blocks.
    pub fn num_total_blocks(&self) -> u32 {
        self.num_blocks
    }

    /// Maximum tokens the pool can hold.
    pub fn max_tokens(&self) -> usize {
        self.num_blocks as usize * self.config.block_size as usize
    }

    /// Get the Metal buffer for a specific layer.
    pub fn layer_buffer(&self, layer: usize) -> &Buffer {
        &self.kv_pools[layer]
    }

    /// Block size in bytes for a specific layer.
    pub fn layer_block_bytes(&self, layer: usize) -> usize {
        self.layer_block_bytes[layer]
    }

    /// Compute byte offset within a layer's buffer for a specific block location.
    ///
    /// # Arguments
    /// * `layer` — Layer index (selects which buffer)
    /// * `block_id` — Physical block index within the buffer
    /// * `is_value` — `false` for K, `true` for V
    /// * `token_offset` — Token position within the block (0..block_size)
    pub fn byte_offset(
        &self,
        layer: usize,
        block_id: u32,
        is_value: bool,
        token_offset: u32,
    ) -> usize {
        let lb = self.layer_block_bytes[layer];
        let kv_half = lb / 2;

        // Block start within this layer's buffer
        let block_start = block_id as usize * lb;
        // K or V offset
        let kv_offset = if is_value { kv_half } else { 0 };
        // Token offset within K/V
        let token_bytes = token_offset as usize * self.config.layer_configs[layer].token_kv_bytes();

        block_start + kv_offset + token_bytes
    }

    /// Check if enough blocks are available for `n` tokens.
    pub fn can_allocate_tokens(&self, n_tokens: usize) -> bool {
        let blocks_needed = n_tokens.div_ceil(self.config.block_size as usize);
        self.free_blocks.len() >= blocks_needed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config_uniform() -> BlockConfig {
        BlockConfig::uniform(16, 2, 2, 64)
    }

    fn test_config_31b() -> BlockConfig {
        // Simplified 31B-like: 4 layers, mixed sliding(256)/global(512)
        let layers = vec![
            LayerAttentionConfig {
                n_kv_heads: 8,
                head_dim: 256,
                is_sliding: true,
            },
            LayerAttentionConfig {
                n_kv_heads: 8,
                head_dim: 256,
                is_sliding: true,
            },
            LayerAttentionConfig {
                n_kv_heads: 8,
                head_dim: 512,
                is_sliding: false,
            },
            LayerAttentionConfig {
                n_kv_heads: 8,
                head_dim: 256,
                is_sliding: true,
            },
        ];
        BlockConfig::from_layers(16, layers)
    }

    #[test]
    fn test_block_bytes_uniform() {
        let cfg = test_config_uniform();
        // Per layer: 16 × 2 × 64 × 2 × 2 = 8192
        assert_eq!(cfg.layer_configs[0].block_bytes(16), 8192);
        // Total: 8192 × 2 = 16384
        assert_eq!(cfg.block_bytes_total(), 16384);
        assert!(!cfg.is_heterogeneous());
    }

    #[test]
    fn test_block_bytes_31b_no_waste() {
        let cfg = test_config_31b();
        assert!(cfg.is_heterogeneous());

        // Sliding layer: 16 × 8 × 256 × 2 × 2 = 131072 = 128KB
        assert_eq!(cfg.layer_configs[0].block_bytes(16), 131072);
        // Global layer: 16 × 8 × 512 × 2 × 2 = 262144 = 256KB
        assert_eq!(cfg.layer_configs[2].block_bytes(16), 262144);

        // Total: 128KB × 3 (sliding) + 256KB × 1 (global) = 640KB
        let total = cfg.block_bytes_total();
        assert_eq!(total, 131072 * 3 + 262144 * 1);

        // Compare with uniform max: 256KB × 4 = 1024KB → 37.5% savings
        let uniform_total = 262144 * 4;
        let savings = 1.0 - (total as f64 / uniform_total as f64);
        assert!(
            savings > 0.35,
            "should save >35%, got {:.1}%",
            savings * 100.0
        );
    }

    #[test]
    fn test_allocate_free() {
        let device = match Device::system_default() {
            Some(d) => d,
            None => return,
        };
        let cfg = test_config_uniform();
        let budget = cfg.block_bytes_total() * 10;
        let mut alloc = BlockAllocator::new(&device, budget, cfg).unwrap();

        assert_eq!(alloc.num_total_blocks(), 10);
        assert_eq!(alloc.num_free_blocks(), 10);

        let b0 = alloc.allocate().unwrap();
        let b1 = alloc.allocate().unwrap();
        let b2 = alloc.allocate().unwrap();
        assert_eq!(alloc.num_free_blocks(), 7);

        alloc.free(b1);
        assert_eq!(alloc.num_free_blocks(), 8);

        let batch = alloc.allocate_n(8).unwrap();
        assert_eq!(batch.len(), 8);
        assert_eq!(alloc.num_free_blocks(), 0);
        assert!(alloc.allocate().is_none());

        alloc.free(b0);
        alloc.free(b2);
        alloc.free_many(&batch);
        assert_eq!(alloc.num_free_blocks(), 10);
    }

    #[test]
    fn test_allocate_31b_savings() {
        let device = match Device::system_default() {
            Some(d) => d,
            None => return,
        };

        let cfg_exact = test_config_31b();
        let budget = 10 * 1024 * 1024; // 10MB

        let blocks_exact = (budget / cfg_exact.block_bytes_total()) as u32;
        let alloc = BlockAllocator::new(&device, budget, cfg_exact).unwrap();
        assert_eq!(alloc.num_total_blocks(), blocks_exact);

        // Uniform max would give fewer blocks for same budget
        let cfg_uniform = BlockConfig::uniform(16, 4, 8, 512);
        let blocks_uniform = (budget / cfg_uniform.block_bytes_total()) as u32;

        // Per-layer sizing fits more blocks
        assert!(
            blocks_exact > blocks_uniform,
            "exact={} should > uniform={}",
            blocks_exact,
            blocks_uniform,
        );
    }

    #[test]
    fn test_byte_offset_per_layer() {
        let device = match Device::system_default() {
            Some(d) => d,
            None => return,
        };
        let cfg = test_config_31b();
        let budget = cfg.block_bytes_total() * 5;
        let alloc = BlockAllocator::new(&device, budget, cfg).unwrap();

        // Layer 0 (sliding, head_dim=256): block_bytes = 131072
        let off_k = alloc.byte_offset(0, 0, false, 0);
        assert_eq!(off_k, 0);
        let off_v = alloc.byte_offset(0, 0, true, 0);
        assert_eq!(off_v, 131072 / 2); // V starts at half

        // Layer 2 (global, head_dim=512): block_bytes = 262144
        let off_k2 = alloc.byte_offset(2, 0, false, 0);
        assert_eq!(off_k2, 0); // different buffer, starts at 0
        let off_v2 = alloc.byte_offset(2, 0, true, 0);
        assert_eq!(off_v2, 262144 / 2);

        // Block 1 in layer 0: starts at 131072
        let off_b1 = alloc.byte_offset(0, 1, false, 0);
        assert_eq!(off_b1, 131072);
    }
}
