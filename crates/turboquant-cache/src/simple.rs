use std::sync::Arc;

use lumen_core::compressor::TurboQuantCompressor;
use lumen_core::config::CompressedVector;

use crate::traits::KVCache;

/// Contiguous-memory KV cache with TurboQuant compression.
///
/// Both keys and values are compressed using TurboQuant.
/// - Keys: compressed for fast inner-product estimation (attend_keys)
/// - Values: compressed and dequantized on-the-fly for weighted sum (attend_values)
pub struct SimpleCache {
    compressor: Arc<TurboQuantCompressor>,
    /// `key_cache[layer][head]` = Vec of compressed key vectors.
    key_cache: Vec<Vec<Vec<CompressedVector>>>,
    /// `value_cache[layer][head]` = Vec of compressed value vectors.
    value_cache: Vec<Vec<Vec<CompressedVector>>>,
    head_dim: usize,
}

impl SimpleCache {
    pub fn new(
        compressor: Arc<TurboQuantCompressor>,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let key_cache = vec![vec![Vec::new(); n_kv_heads]; n_layers];
        let value_cache = vec![vec![Vec::new(); n_kv_heads]; n_layers];
        Self {
            compressor,
            key_cache,
            value_cache,
            head_dim,
        }
    }

    /// Accumulate `weight * dequantize(cv)` into `output`.
    fn accumulate_weighted_value(&self, cv: &CompressedVector, weight: f32, output: &mut [f32]) {
        debug_assert_eq!(cv.dim, self.head_dim);
        debug_assert_eq!(output.len(), self.head_dim);

        let bits = cv.bits;
        let codes_per_word = 64 / bits as usize;
        let mask = (1u64 << bits) - 1;
        let scale = cv.scale * weight;
        let centroids = &self.compressor.codebook().centroids;
        let rotation = self.compressor.rotation();

        for j in 0..cv.dim {
            let word_idx = j / codes_per_word;
            let bit_offset = (j % codes_per_word) as u32 * bits;
            let code = ((cv.stage1_packed[word_idx] >> bit_offset) & mask) as usize;
            let yj = centroids[code] as f32 * scale;
            let row = &rotation.matrix[j * rotation.dim..(j + 1) * rotation.dim];

            for (out, &r) in output.iter_mut().zip(row.iter()) {
                *out += r * yj;
            }
        }
    }

    /// Estimated total memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let mut total = 0;
        for layer in &self.key_cache {
            for head in layer {
                for cv in head {
                    total += cv.size_bytes();
                }
            }
        }
        for layer in &self.value_cache {
            for head in layer {
                for cv in head {
                    total += cv.size_bytes();
                }
            }
        }
        total
    }
}

impl KVCache for SimpleCache {
    fn store(&mut self, layer: usize, head: usize, keys: &[f32], values: &[f32], n_tokens: usize) {
        debug_assert_eq!(keys.len(), n_tokens * self.head_dim);
        debug_assert_eq!(values.len(), n_tokens * self.head_dim);

        // Compress both keys and values
        self.key_cache[layer][head].reserve(n_tokens);
        let compressed_keys = self.compressor.compress(keys, n_tokens);
        self.key_cache[layer][head].extend(compressed_keys);

        self.value_cache[layer][head].reserve(n_tokens);
        let compressed_values = self.compressor.compress(values, n_tokens);
        self.value_cache[layer][head].extend(compressed_values);
    }

    fn attend_keys(&self, layer: usize, head: usize, query: &[f32]) -> Vec<f32> {
        self.compressor
            .estimate_inner_products_stage1(query, &self.key_cache[layer][head])
    }

    fn attend_values(&self, layer: usize, head: usize, weights: &[f32]) -> Vec<f32> {
        let values = &self.value_cache[layer][head];
        debug_assert_eq!(weights.len(), values.len());

        // Weighted sum: Σ w_i * dequantize(v_i)
        let mut output = vec![0.0f32; self.head_dim];
        for (&w, cv) in weights.iter().zip(values.iter()) {
            if w.abs() < 1e-10 {
                continue; // skip negligible weights for speed
            }
            self.accumulate_weighted_value(cv, w, &mut output);
        }
        output
    }

    fn seq_len(&self, layer: usize, head: usize) -> usize {
        self.key_cache[layer][head].len()
    }

    fn clear(&mut self) {
        for layer in &mut self.key_cache {
            for head in layer.iter_mut() {
                head.clear();
            }
        }
        for layer in &mut self.value_cache {
            for head in layer.iter_mut() {
                head.clear();
            }
        }
    }

    fn truncate(&mut self, layer: usize, n_keep: usize) {
        for head in self.key_cache[layer].iter_mut() {
            if head.len() > n_keep {
                head.truncate(n_keep);
            }
        }
        for head in self.value_cache[layer].iter_mut() {
            if head.len() > n_keep {
                head.truncate(n_keep);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::config::TurboQuantConfig;
    use rand::prelude::*;
    use rand_distr::StandardNormal;

    #[test]
    fn store_and_attend() {
        let config = TurboQuantConfig {
            bits: 3,
            qjl_m: 64,
            seed: 42,
            lloyd_max_iter: 1000,
            head_dim: 64,
        };
        let compressor = Arc::new(TurboQuantCompressor::new(config));
        let mut cache = SimpleCache::new(Arc::clone(&compressor), 2, 4, 64);

        let mut rng = StdRng::seed_from_u64(42);
        let n_tokens = 10;
        let dim = 64;

        let keys: Vec<f32> = (0..n_tokens * dim)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();
        let values: Vec<f32> = (0..n_tokens * dim)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();
        cache.store(0, 0, &keys, &values, n_tokens);

        assert_eq!(cache.seq_len(0, 0), n_tokens);
        assert_eq!(cache.seq_len(0, 1), 0);

        // Query and attend
        let query: Vec<f32> = (0..dim)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();
        let scores = cache.attend_keys(0, 0, &query);
        assert_eq!(scores.len(), n_tokens);

        // Softmax weights
        let sum: f32 = scores.iter().map(|s| s.exp()).sum();
        let weights: Vec<f32> = scores.iter().map(|s| s.exp() / sum).collect();

        let output = cache.attend_values(0, 0, &weights);
        assert_eq!(output.len(), dim);

        // Compare with uncompressed weighted sum
        let mut expected = vec![0.0f32; dim];
        for (i, &w) in weights.iter().enumerate() {
            for (o, &val) in expected
                .iter_mut()
                .zip(values[i * dim..(i + 1) * dim].iter())
            {
                *o += w * val;
            }
        }

        // Cosine similarity should be high
        let dot: f32 = output
            .iter()
            .zip(expected.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        let norm_o: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_e: f32 = expected.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cosine = dot / (norm_o * norm_e);
        eprintln!("value reconstruction cosine: {cosine:.4}");
        assert!(cosine > 0.95, "value cosine too low: {cosine}");
    }

    #[test]
    fn memory_usage() {
        let config = TurboQuantConfig {
            bits: 3,
            qjl_m: 64,
            seed: 42,
            lloyd_max_iter: 1000,
            head_dim: 64,
        };
        let compressor = Arc::new(TurboQuantCompressor::new(config));
        let mut cache = SimpleCache::new(Arc::clone(&compressor), 1, 1, 64);

        let mut rng = StdRng::seed_from_u64(42);
        let dim = 64;
        let n = 100;
        let keys: Vec<f32> = (0..n * dim)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();
        let vals: Vec<f32> = (0..n * dim)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();
        cache.store(0, 0, &keys, &vals, n);

        let used = cache.memory_bytes();
        let uncompressed = n * dim * 4 * 2; // K+V, f32
        let ratio = uncompressed as f64 / used as f64;
        eprintln!("memory: {used} bytes compressed vs {uncompressed} bytes raw, ratio={ratio:.2}x");
        assert!(ratio > 2.0, "compression ratio too low: {ratio:.2}x");
    }

    #[test]
    fn truncate_drops_tail() {
        let config = TurboQuantConfig::default();
        let compressor = Arc::new(TurboQuantCompressor::new(config));
        let mut cache = SimpleCache::new(Arc::clone(&compressor), 1, 2, 128);

        let mut rng = StdRng::seed_from_u64(7);
        let dim = 128;
        for _ in 0..6 {
            let keys: Vec<f32> = (0..dim)
                .map(|_| rng.sample::<f32, _>(StandardNormal))
                .collect();
            let values: Vec<f32> = (0..dim)
                .map(|_| rng.sample::<f32, _>(StandardNormal))
                .collect();
            cache.store(0, 0, &keys, &values, 1);
            cache.store(0, 1, &keys, &values, 1);
        }
        assert_eq!(cache.seq_len(0, 0), 6);
        assert_eq!(cache.seq_len(0, 1), 6);

        cache.truncate(0, 4);
        assert_eq!(cache.seq_len(0, 0), 4);
        assert_eq!(cache.seq_len(0, 1), 4);

        // No-op when already ≤ n_keep.
        cache.truncate(0, 10);
        assert_eq!(cache.seq_len(0, 0), 4);

        cache.truncate(0, 0);
        assert_eq!(cache.seq_len(0, 0), 0);
        assert_eq!(cache.seq_len(0, 1), 0);
    }

    #[test]
    fn incremental_store() {
        let config = TurboQuantConfig::default();
        let compressor = Arc::new(TurboQuantCompressor::new(config));
        let mut cache = SimpleCache::new(Arc::clone(&compressor), 1, 1, 128);

        let mut rng = StdRng::seed_from_u64(42);
        let dim = 128;

        for _ in 0..5 {
            let keys: Vec<f32> = (0..dim)
                .map(|_| rng.sample::<f32, _>(StandardNormal))
                .collect();
            let values: Vec<f32> = (0..dim)
                .map(|_| rng.sample::<f32, _>(StandardNormal))
                .collect();
            cache.store(0, 0, &keys, &values, 1);
        }

        assert_eq!(cache.seq_len(0, 0), 5);
        cache.clear();
        assert_eq!(cache.seq_len(0, 0), 0);
    }
}
