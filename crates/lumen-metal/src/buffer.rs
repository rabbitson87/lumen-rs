use crate::metal::Buffer;

use crate::device::MetalContext;

/// Pre-allocated GPU buffer pool for compressed KV cache.
///
/// Replaces Tensor::cat (which reallocates every step) with fixed-size
/// circular buffers. For 3-bit, 8K context:
///   - packed_codes: ~157MB (vs ~1.3GB uncompressed f16, ~8x savings)
///   - scales/res_norms: negligible
///   - qjl_bits: ~16MB
///
/// Layout per layer per head:
///   packed_codes [max_seq x n_packed] u64
///   scales       [max_seq] f32
///   res_norms    [max_seq] f32
///   qjl_bits     [max_seq x n_qjl_packed] u64
pub struct CompressedKVPool {
    /// One pool per layer, per head.
    pub layers: Vec<LayerKVPool>,
    pub config: KVPoolConfig,
}

/// Configuration for the KV pool.
#[derive(Debug, Clone)]
pub struct KVPoolConfig {
    /// Max sequence length (context window).
    pub max_seq_len: usize,
    /// Attention head dimension.
    pub head_dim: usize,
    /// Number of KV heads per layer.
    pub n_kv_heads: usize,
    /// Number of layers.
    pub n_layers: usize,
    /// Lloyd-Max quantization bits (2, 3, or 4).
    pub bits: u32,
    /// QJL projection dimension.
    pub qjl_m: usize,
}

impl KVPoolConfig {
    /// Number of u64 words per vector for packed codes.
    /// Matches CPU bitpack::packed_words — no cross-word-boundary packing.
    pub fn n_packed(&self) -> usize {
        let codes_per_word = 64 / self.bits as usize;
        (self.head_dim + codes_per_word - 1) / codes_per_word
    }

    /// Number of u64 words per vector for QJL bits.
    pub fn n_qjl_packed(&self) -> usize {
        (self.qjl_m + 63) / 64
    }

    /// Number of Lloyd-Max quantization levels.
    pub fn n_levels(&self) -> u32 {
        1 << self.bits
    }

    /// Estimated total GPU memory for the pool (bytes).
    pub fn estimated_memory_bytes(&self) -> usize {
        let per_vec = self.n_packed() * 8   // packed codes
            + 4                              // scale
            + 4                              // res_norm
            + self.n_qjl_packed() * 8; // qjl bits
        per_vec * self.max_seq_len * self.n_kv_heads * self.n_layers * 2 // K + V
    }
}

/// KV buffers for all heads in one layer.
pub struct LayerKVPool {
    pub heads: Vec<HeadKVBuffers>,
}

/// GPU buffers for one attention head's compressed KV cache.
pub struct HeadKVBuffers {
    /// Key cache
    pub key: CompressedBuffers,
    /// Value cache
    pub value: CompressedBuffers,
    /// Current key sequence length.
    pub key_seq_len: usize,
    /// Current value sequence length.
    pub val_seq_len: usize,
}

impl HeadKVBuffers {
    /// Sequence length (min of key and value, should always match).
    pub fn seq_len(&self) -> usize {
        self.key_seq_len.min(self.val_seq_len)
    }
}

/// GPU buffers holding one type (K or V) of compressed vectors.
pub struct CompressedBuffers {
    /// Bit-packed Lloyd-Max codes [max_seq x n_packed] u64.
    pub packed_codes: Buffer,
    /// Per-vector scale factors [max_seq] f32.
    pub scales: Buffer,
    /// Per-vector residual norms [max_seq] f32.
    pub res_norms: Buffer,
    /// QJL 1-bit signs packed [max_seq x n_qjl_packed] u64.
    pub qjl_bits: Buffer,
}

impl CompressedKVPool {
    /// Allocate all GPU buffers for the compressed KV cache.
    pub fn new(ctx: &MetalContext, config: KVPoolConfig) -> Self {
        let n_packed = config.n_packed();
        let n_qjl_packed = config.n_qjl_packed();
        let max_seq = config.max_seq_len;

        let layers = (0..config.n_layers)
            .map(|_| {
                let heads = (0..config.n_kv_heads)
                    .map(|_| {
                        let key = Self::alloc_buffers(ctx, max_seq, n_packed, n_qjl_packed);
                        let value = Self::alloc_buffers(ctx, max_seq, n_packed, n_qjl_packed);
                        HeadKVBuffers {
                            key,
                            value,
                            key_seq_len: 0,
                            val_seq_len: 0,
                        }
                    })
                    .collect();
                LayerKVPool { heads }
            })
            .collect();

        let mem_mb = config.estimated_memory_bytes() as f64 / (1024.0 * 1024.0);
        eprintln!(
            "TurboQuant Metal: allocated KV pool ({} layers x {} heads, {}-bit, max_seq={}, {:.1}MB)",
            config.n_layers, config.n_kv_heads, config.bits, max_seq, mem_mb
        );

        Self { layers, config }
    }

    fn alloc_buffers(
        ctx: &MetalContext,
        max_seq: usize,
        n_packed: usize,
        n_qjl_packed: usize,
    ) -> CompressedBuffers {
        CompressedBuffers {
            packed_codes: ctx.buffer_for::<u64>(max_seq * n_packed),
            scales: ctx.buffer_for::<f32>(max_seq),
            res_norms: ctx.buffer_for::<f32>(max_seq),
            qjl_bits: ctx.buffer_for::<u64>(max_seq * n_qjl_packed),
        }
    }

    /// Reset all sequence lengths to 0 (new request).
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            for head in &mut layer.heads {
                head.key_seq_len = 0;
                head.val_seq_len = 0;
            }
        }
    }

    /// Drop the tail of every head in `layer` so that both key and value
    /// sequences are at most `n_keep` long. Buffer storage past `n_keep` is
    /// left untouched; subsequent stores will overwrite it. No-op when the
    /// current length is already ≤ `n_keep`.
    pub fn truncate(&mut self, layer: usize, n_keep: usize) {
        if layer >= self.layers.len() {
            return;
        }
        for head in &mut self.layers[layer].heads {
            if head.key_seq_len > n_keep {
                head.key_seq_len = n_keep;
            }
            if head.val_seq_len > n_keep {
                head.val_seq_len = n_keep;
            }
        }
    }

    /// Get mutable reference to a specific head's KV buffers.
    pub fn head_mut(&mut self, layer: usize, head: usize) -> &mut HeadKVBuffers {
        &mut self.layers[layer].heads[head]
    }

    /// Get reference to a specific head's KV buffers.
    pub fn head(&self, layer: usize, head: usize) -> &HeadKVBuffers {
        &self.layers[layer].heads[head]
    }
}
