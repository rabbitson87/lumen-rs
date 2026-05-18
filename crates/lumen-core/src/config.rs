use serde::{Deserialize, Serialize};

/// Configuration for the TurboQuant compression pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboQuantConfig {
    /// Lloyd-Max quantization bit width (2, 3, or 4).
    pub bits: u32,
    /// QJL projection dimension. Recommended: `head_dim / 2`.
    pub qjl_m: usize,
    /// Seed for random orthogonal matrix and Gaussian projection.
    pub seed: u64,
    /// Number of Lloyd-Max EM iterations.
    pub lloyd_max_iter: usize,
    /// Attention head dimension (e.g., 128 for Qwen2.5).
    pub head_dim: usize,
}

impl Default for TurboQuantConfig {
    fn default() -> Self {
        Self {
            bits: 3,
            qjl_m: 64,
            seed: 42,
            lloyd_max_iter: 1000,
            head_dim: 128,
        }
    }
}

/// A single KV vector after TurboQuant two-stage compression.
#[derive(Debug, Clone)]
pub struct CompressedVector {
    /// Stage 1: Lloyd-Max codes bit-packed into u64 words.
    /// For 3-bit, 256 dims: 13 u64 words (104 bytes vs 256 bytes unpacked).
    pub stage1_packed: Vec<u64>,
    /// Stage 2: QJL 1-bit projection packed into u64 words.
    pub stage2_bits: Vec<u64>,
    /// L2 norm of the residual (original − stage1 reconstruction).
    pub residual_norm: f32,
    /// Per-vector scale factor: σ = ||rotated|| / √dim.
    pub scale: f32,
    /// Number of dimensions (needed for unpacking).
    pub dim: usize,
    /// Quantization bit width (needed for unpacking).
    pub bits: u32,
}

impl CompressedVector {
    /// Estimated memory usage in bytes.
    pub fn size_bytes(&self) -> usize {
        self.stage1_packed.len() * 8   // packed codes
        + self.stage2_bits.len() * 8   // QJL bits
        + 4 + 4                        // residual_norm + scale
        + 8 + 4 // dim + bits
    }
}
