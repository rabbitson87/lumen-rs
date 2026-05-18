use crate::bitpack;
use crate::config::{CompressedVector, TurboQuantConfig};
use crate::lloyd_max::LloydMaxCodebook;
use crate::qjl::QJLProjector;
use crate::rotation::RotationMatrix;

/// Two-stage TurboQuant compressor.
///
/// Stage 1: Random rotation + Lloyd-Max scalar quantization.
/// Stage 2: QJL 1-bit residual projection for unbiased inner product correction.
pub struct TurboQuantCompressor {
    rotation: RotationMatrix,
    codebook: LloydMaxCodebook,
    qjl: QJLProjector,
    pub config: TurboQuantConfig,
}

impl TurboQuantCompressor {
    /// Create a new compressor, computing all components from config.
    pub fn new(config: TurboQuantConfig) -> Self {
        let rotation = RotationMatrix::random(config.head_dim, config.seed);
        let codebook = LloydMaxCodebook::compute(config.bits, config.lloyd_max_iter)
            .expect("valid bits in config");
        let qjl = QJLProjector::new(config.head_dim, config.qjl_m, config.seed.wrapping_add(1));

        Self {
            rotation,
            codebook,
            qjl,
            config,
        }
    }

    pub fn codebook(&self) -> &LloydMaxCodebook {
        &self.codebook
    }

    pub fn rotation(&self) -> &RotationMatrix {
        &self.rotation
    }

    /// Load pre-computed components from binary files.
    pub fn from_files(
        codebook_path: &str,
        rotation_path: &str,
        qjl_path: &str,
        config: TurboQuantConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let codebook = LloydMaxCodebook::load(codebook_path)?;
        let rotation = RotationMatrix::load(rotation_path)?;
        let qjl = QJLProjector::load(qjl_path)?;
        Ok(Self {
            rotation,
            codebook,
            qjl,
            config,
        })
    }

    /// Compress a single vector of dimension `head_dim`.
    pub fn compress_one(&self, vector: &[f32]) -> CompressedVector {
        let dim = self.config.head_dim;
        debug_assert_eq!(vector.len(), dim);

        // Stage 1: Rotate → Normalize → Quantize → Denormalize → Inverse-rotate
        let rotated = self.rotation.apply(vector);

        // Per-vector scale: σ = ||rotated|| / √dim
        // After dividing by σ, coordinates ~ N(0, 1) matching the codebook.
        let norm_sq: f32 = rotated.iter().map(|x| x * x).sum();
        let scale = (norm_sq / dim as f32).sqrt();

        let codes = if scale < 1e-10 {
            vec![0u8; dim] // degenerate zero vector
        } else {
            let inv_scale = 1.0 / scale;
            let normalized: Vec<f32> = rotated.iter().map(|&x| x * inv_scale).collect();
            self.codebook.quantize_vec(&normalized)
        };

        // Reconstruct: dequantize → denormalize → inverse-rotate
        let dequantized = self.codebook.dequantize_vec(&codes);
        let denormalized: Vec<f32> = dequantized.iter().map(|&x| x * scale).collect();
        let reconstructed = self.rotation.apply_inverse(&denormalized);

        // Residual = original - stage1_reconstruction
        let mut residual = vec![0.0f32; dim];
        let mut res_norm_sq = 0.0f32;
        for i in 0..dim {
            residual[i] = vector[i] - reconstructed[i];
            res_norm_sq += residual[i] * residual[i];
        }
        let residual_norm = res_norm_sq.sqrt();

        // Stage 2: QJL 1-bit projection of residual
        let stage2_bits = self.qjl.project_and_pack(&residual);

        let bits = self.config.bits;
        CompressedVector {
            stage1_packed: bitpack::pack_codes(&codes, bits),
            stage2_bits,
            residual_norm,
            scale,
            dim,
            bits,
        }
    }

    /// Compress n vectors. `vectors` is `[n, dim]` row-major.
    pub fn compress(&self, vectors: &[f32], n: usize) -> Vec<CompressedVector> {
        let dim = self.config.head_dim;
        debug_assert_eq!(vectors.len(), n * dim);
        (0..n)
            .map(|i| self.compress_one(&vectors[i * dim..(i + 1) * dim]))
            .collect()
    }

    /// Estimate inner products between a query and compressed vectors.
    ///
    /// Returns a score vector of length `compressed.len()`.
    /// Each score ≈ <query, original_vector>.
    pub fn estimate_inner_products(
        &self,
        query: &[f32],
        compressed: &[CompressedVector],
    ) -> Vec<f32> {
        let dim = self.config.head_dim;
        debug_assert_eq!(query.len(), dim);

        // Pre-rotate query (for Stage 1 dot products in rotated space)
        let rotated_query = self.rotation.apply(query);
        let projected_query = self.qjl.project_query(query);

        compressed
            .iter()
            .map(|cv| {
                // Stage 1: dot product in rotated + scaled space
                // Stored codes are quantized(rotated / scale), so
                // reconstruction in rotated space = dequantize(codes) * scale
                let codes = bitpack::unpack_codes(&cv.stage1_packed, cv.dim, cv.bits);
                let dequantized = self.codebook.dequantize_vec(&codes);
                let stage1: f32 = rotated_query
                    .iter()
                    .zip(dequantized.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>()
                    * cv.scale;

                // Stage 2: QJL correction in original space
                let stage2 = self.qjl.estimate_correction_from_projection(
                    &projected_query,
                    &cv.stage2_bits,
                    cv.residual_norm,
                );

                stage1 + stage2
            })
            .collect()
    }

    /// Estimate inner products using Stage 1 only (no QJL correction).
    ///
    /// Lower variance at the cost of slight bias from quantization error.
    /// Recommended when qjl_m is small relative to head_dim.
    pub fn estimate_inner_products_stage1(
        &self,
        query: &[f32],
        compressed: &[CompressedVector],
    ) -> Vec<f32> {
        let rotated_query = self.rotation.apply(query);
        compressed
            .iter()
            .map(|cv| {
                let codes = bitpack::unpack_codes(&cv.stage1_packed, cv.dim, cv.bits);
                let dequantized = self.codebook.dequantize_vec(&codes);
                rotated_query
                    .iter()
                    .zip(dequantized.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>()
                    * cv.scale
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand_distr::StandardNormal;

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    /// Helper: generate random Gaussian vectors (not unit-normalized)
    /// matching real KV cache distributions.
    fn generate_kv_vectors(rng: &mut StdRng, n: usize, dim: usize) -> Vec<f32> {
        let mut vectors = vec![0.0f32; n * dim];
        for x in vectors.iter_mut() {
            *x = rng.sample::<f32, _>(StandardNormal);
        }
        vectors
    }

    /// Helper: run e2e cosine similarity test with given config.
    fn e2e_cosine_test(config: TurboQuantConfig, n_vectors: usize, n_queries: usize) -> (f64, f32) {
        let dim = config.head_dim;
        let compressor = TurboQuantCompressor::new(config);
        let mut rng = StdRng::seed_from_u64(99);

        let vectors = generate_kv_vectors(&mut rng, n_vectors, dim);
        let compressed = compressor.compress(&vectors, n_vectors);

        let mut min_cosine = f32::MAX;
        let mut sum_cosine = 0.0f64;

        for _ in 0..n_queries {
            let query: Vec<f32> = (0..dim)
                .map(|_| rng.sample::<f32, _>(StandardNormal))
                .collect();

            let true_scores: Vec<f32> = (0..n_vectors)
                .map(|i| {
                    let v = &vectors[i * dim..(i + 1) * dim];
                    query.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum()
                })
                .collect();

            let est_scores = compressor.estimate_inner_products(&query, &compressed);
            let cos = cosine_similarity(&true_scores, &est_scores);
            min_cosine = min_cosine.min(cos);
            sum_cosine += cos as f64;
        }

        (sum_cosine / n_queries as f64, min_cosine)
    }

    #[test]
    fn end_to_end_stage1_only_3bit() {
        // Stage 1 alone (rotation + Lloyd-Max) should achieve cosine >= 0.98
        let config = TurboQuantConfig {
            bits: 3,
            qjl_m: 64,
            seed: 42,
            lloyd_max_iter: 1000,
            head_dim: 128,
        };
        let dim = config.head_dim;
        let compressor = TurboQuantCompressor::new(config);
        let mut rng = StdRng::seed_from_u64(99);

        let vectors = generate_kv_vectors(&mut rng, 200, dim);
        let compressed = compressor.compress(&vectors, 200);

        let mut sum_cosine = 0.0f64;
        let n_queries = 50;
        for _ in 0..n_queries {
            let query: Vec<f32> = (0..dim)
                .map(|_| rng.sample::<f32, _>(StandardNormal))
                .collect();
            let true_scores: Vec<f32> = (0..200)
                .map(|i| {
                    let v = &vectors[i * dim..(i + 1) * dim];
                    query.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum()
                })
                .collect();
            let est_scores = compressor.estimate_inner_products_stage1(&query, &compressed);
            sum_cosine += cosine_similarity(&true_scores, &est_scores) as f64;
        }
        let avg = sum_cosine / n_queries as f64;
        eprintln!("3-bit stage1-only: avg cosine={avg:.4}");
        assert!(avg >= 0.98, "stage1-only avg cosine {avg} < 0.98");
    }

    #[test]
    fn end_to_end_cosine_similarity_3bit() {
        // Full two-stage with sufficient projections (m=512)
        let config = TurboQuantConfig {
            bits: 3,
            qjl_m: 512,
            seed: 42,
            lloyd_max_iter: 1000,
            head_dim: 128,
        };
        let (avg, min) = e2e_cosine_test(config, 200, 50);
        eprintln!("3-bit (m=512): avg cosine={avg:.4}, min cosine={min:.4}");
        assert!(avg >= 0.99, "avg cosine {avg} < 0.99");
        assert!(min >= 0.95, "min cosine {min} < 0.95");
    }

    #[test]
    fn stage1_only_quality() {
        // Test that Stage 1 alone provides reasonable quality
        let config = TurboQuantConfig {
            bits: 3,
            qjl_m: 64,
            seed: 42,
            lloyd_max_iter: 1000,
            head_dim: 128,
        };
        let compressor = TurboQuantCompressor::new(config);
        let mut rng = StdRng::seed_from_u64(77);
        let dim = 128;

        let mut vector = vec![0.0f32; dim];
        for x in vector.iter_mut() {
            *x = rng.sample::<f32, _>(StandardNormal);
        }

        let cv = compressor.compress_one(&vector);

        // Stage 1 reconstruction: unpack → dequantize → inverse-rotate
        let codes = bitpack::unpack_codes(&cv.stage1_packed, cv.dim, cv.bits);
        let recon_rotated = compressor.codebook.dequantize_vec(&codes);
        let recon = compressor.rotation.apply_inverse(&recon_rotated);

        let cos = cosine_similarity(&vector, &recon);
        eprintln!("stage1-only vector cosine: {cos:.4}");
        assert!(cos >= 0.98, "stage1 reconstruction cosine {cos} < 0.98");
    }

    #[test]
    fn stage1_only_scores() {
        let config = TurboQuantConfig {
            bits: 3,
            qjl_m: 64,
            seed: 42,
            lloyd_max_iter: 1000,
            head_dim: 128,
        };
        let compressor = TurboQuantCompressor::new(config);
        let mut rng = StdRng::seed_from_u64(99);
        let dim = 128;
        let n_vectors = 100;

        let mut vectors = vec![0.0f32; n_vectors * dim];
        for v in 0..n_vectors {
            let slice = &mut vectors[v * dim..(v + 1) * dim];
            for x in slice.iter_mut() {
                *x = rng.sample::<f32, _>(StandardNormal);
            }
            let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in slice.iter_mut() {
                *x /= norm;
            }
        }

        let compressed = compressor.compress(&vectors, n_vectors);

        let mut query = vec![0.0f32; dim];
        for x in query.iter_mut() {
            *x = rng.sample::<f32, _>(StandardNormal);
        }
        let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in query.iter_mut() {
            *x /= norm;
        }

        // True scores
        let true_scores: Vec<f32> = (0..n_vectors)
            .map(|i| {
                let v = &vectors[i * dim..(i + 1) * dim];
                query.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum()
            })
            .collect();

        // Stage1-only scores
        let rotated_query = compressor.rotation.apply(&query);
        let stage1_scores: Vec<f32> = compressed
            .iter()
            .map(|cv| {
                let codes = bitpack::unpack_codes(&cv.stage1_packed, cv.dim, cv.bits);
                let recon = compressor.codebook.dequantize_vec(&codes);
                let dot: f32 = rotated_query
                    .iter()
                    .zip(recon.iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                dot * cv.scale
            })
            .collect();

        // Stage2 corrections
        let stage2_corrections: Vec<f32> = compressed
            .iter()
            .map(|cv| {
                compressor
                    .qjl
                    .estimate_correction(&query, &cv.stage2_bits, cv.residual_norm)
            })
            .collect();

        let cos_s1 = cosine_similarity(&true_scores, &stage1_scores);
        let combined: Vec<f32> = stage1_scores
            .iter()
            .zip(stage2_corrections.iter())
            .map(|(a, b)| a + b)
            .collect();
        let cos_combined = cosine_similarity(&true_scores, &combined);

        let s1_mag: f32 = stage1_scores.iter().map(|x| x.abs()).sum::<f32>() / n_vectors as f32;
        let s2_mag: f32 =
            stage2_corrections.iter().map(|x| x.abs()).sum::<f32>() / n_vectors as f32;
        let res_norms: f32 =
            compressed.iter().map(|cv| cv.residual_norm).sum::<f32>() / n_vectors as f32;

        eprintln!("stage1-only score cosine: {cos_s1:.4}");
        eprintln!("combined score cosine:    {cos_combined:.4}");
        eprintln!("avg |stage1 score|: {s1_mag:.4}");
        eprintln!("avg |stage2 correction|: {s2_mag:.4}");
        eprintln!("avg residual norm: {res_norms:.4}");
    }

    #[test]
    fn different_bit_widths() {
        let dim = 128;
        let mut rng = StdRng::seed_from_u64(42);
        let n_vectors = 100;

        let mut vectors = vec![0.0f32; n_vectors * dim];
        for v in 0..n_vectors {
            let slice = &mut vectors[v * dim..(v + 1) * dim];
            for x in slice.iter_mut() {
                *x = rng.sample::<f32, _>(StandardNormal);
            }
            let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in slice.iter_mut() {
                *x /= norm;
            }
        }

        let mut query = vec![0.0f32; dim];
        for x in query.iter_mut() {
            *x = rng.sample::<f32, _>(StandardNormal);
        }
        let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in query.iter_mut() {
            *x /= norm;
        }

        let true_scores: Vec<f32> = (0..n_vectors)
            .map(|i| {
                let v = &vectors[i * dim..(i + 1) * dim];
                query.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum()
            })
            .collect();

        for &bits in &[2u32, 3, 4] {
            let config = TurboQuantConfig {
                bits,
                qjl_m: 64,
                seed: 42,
                lloyd_max_iter: 1000,
                head_dim: dim,
            };
            let compressor = TurboQuantCompressor::new(config);
            let compressed = compressor.compress(&vectors, n_vectors);
            let est_scores = compressor.estimate_inner_products(&query, &compressed);
            let cos = cosine_similarity(&true_scores, &est_scores);
            eprintln!("{bits}-bit cosine similarity: {cos:.4}");
        }
    }
}
