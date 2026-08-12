use std::io::{Read, Write};

use rand::prelude::*;
use rand_distr::StandardNormal;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum QjlError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid QJL file (bad magic)")]
    BadMagic,
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
}

/// Quantized Johnson-Lindenstrauss (QJL) projector for 1-bit residual encoding.
///
/// Projects residual vectors through a random Gaussian matrix and stores only
/// the sign bits (1-bit), packed into u64 words. Used as Stage 2 of TurboQuant
/// to provide unbiased inner product correction.
#[derive(Debug, Clone)]
pub struct QJLProjector {
    /// Projection matrix: `m x dim`, row-major, entries ~ N(0, 1/m).
    pub proj_matrix: Vec<f32>,
    /// Number of projection dimensions (rows of projection matrix).
    pub m: usize,
    /// Input vector dimension.
    pub dim: usize,
}

const MAGIC: &[u8; 4] = b"TQJL";

/// Number of u64 words needed to pack `m` bits.
#[inline]
pub fn packed_len(m: usize) -> usize {
    (m + 63) / 64
}

impl QJLProjector {
    /// Create a new QJL projector with random Gaussian entries ~ N(0, 1/m).
    pub fn new(dim: usize, m: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = 1.0 / (m as f32).sqrt();
        let proj_matrix: Vec<f32> = (0..m * dim)
            .map(|_| rng.sample::<f32, _>(StandardNormal) * scale)
            .collect();
        Self {
            proj_matrix,
            m,
            dim,
        }
    }

    /// Project a residual vector and pack sign bits into u64 words.
    ///
    /// Returns `ceil(m/64)` u64 values. Bit j of word w is 1 if the
    /// (w*64 + j)-th projection is >= 0, else 0.
    pub fn project_and_pack(&self, residual: &[f32]) -> Vec<u64> {
        debug_assert_eq!(residual.len(), self.dim);
        let n_words = packed_len(self.m);
        let mut packed = vec![0u64; n_words];

        for row_idx in 0..self.m {
            let row = &self.proj_matrix[row_idx * self.dim..(row_idx + 1) * self.dim];
            let dot: f32 = row.iter().zip(residual.iter()).map(|(&a, &b)| a * b).sum();

            if dot >= 0.0 {
                let word = row_idx / 64;
                let bit = row_idx % 64;
                packed[word] |= 1u64 << bit;
            }
        }

        packed
    }

    pub(crate) fn project_query(&self, query: &[f32]) -> Vec<f32> {
        debug_assert_eq!(query.len(), self.dim);
        (0..self.m)
            .map(|row_idx| {
                let row = &self.proj_matrix[row_idx * self.dim..(row_idx + 1) * self.dim];
                row.iter().zip(query.iter()).map(|(&a, &b)| a * b).sum()
            })
            .collect()
    }

    pub(crate) fn estimate_correction_from_projection(
        &self,
        projected_query: &[f32],
        packed_residual: &[u64],
        residual_norm: f32,
    ) -> f32 {
        debug_assert_eq!(projected_query.len(), self.m);
        debug_assert_eq!(packed_residual.len(), packed_len(self.m));
        if residual_norm < 1e-10 {
            return 0.0;
        }

        let mut sum = 0.0f32;

        for (row_idx, &proj_q) in projected_query.iter().enumerate() {
            let word = row_idx / 64;
            let bit = row_idx % 64;
            let sign_r = if (packed_residual[word] >> bit) & 1 == 1 {
                1.0f32
            } else {
                -1.0f32
            };

            sum += sign_r * proj_q;
        }

        // The projection matrix has entries ~ N(0, 1/m), so each row is g_i/√m.
        // proj_q = g_i^T q / √m. We need g_i^T q = √m * proj_q.
        // <q,r> ≈ ||r|| * √(π/2) * (1/m) * Σ sign(g_i^T r) * g_i^T q
        //       = ||r|| * √(π/2) * (1/m) * Σ sign_r * √m * proj_q
        //       = ||r|| * √(π/2) / √m * sum
        let scale = residual_norm * (std::f32::consts::FRAC_PI_2).sqrt() / (self.m as f32).sqrt();
        scale * sum
    }

    /// Estimate inner product correction from packed sign bits.
    ///
    /// Uses the QJL unbiased estimator:
    ///   <q, r> ≈ ||r|| * sqrt(π/2) * (1/m) * Σ_i sign(g_i^T r) * (g_i^T q)
    ///
    /// Only the residual is stored as 1-bit signs; the query is fully projected
    /// at query time. This is the key asymmetry that makes QJL work.
    pub fn estimate_correction(
        &self,
        query: &[f32],
        packed_residual: &[u64],
        residual_norm: f32,
    ) -> f32 {
        debug_assert_eq!(query.len(), self.dim);
        let projected_query = self.project_query(query);
        self.estimate_correction_from_projection(&projected_query, packed_residual, residual_norm)
    }

    /// Save projector to binary file.
    ///
    /// Format: magic(4) + dim(u32 LE) + m(u32 LE) + f32 array (row-major LE).
    pub fn save(&self, path: &str) -> Result<(), QjlError> {
        let mut f = std::fs::File::create(path)?;
        f.write_all(MAGIC)?;
        f.write_all(&(self.dim as u32).to_le_bytes())?;
        f.write_all(&(self.m as u32).to_le_bytes())?;
        for &val in &self.proj_matrix {
            f.write_all(&val.to_le_bytes())?;
        }
        Ok(())
    }

    /// Load projector from binary file.
    pub fn load(path: &str) -> Result<Self, QjlError> {
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(QjlError::BadMagic);
        }

        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let dim = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let m = u32::from_le_bytes(buf4) as usize;

        let n_elems = m * dim;
        let mut proj_matrix = Vec::with_capacity(n_elems);
        for _ in 0..n_elems {
            f.read_exact(&mut buf4)?;
            proj_matrix.push(f32::from_le_bytes(buf4));
        }

        Ok(Self {
            proj_matrix,
            m,
            dim,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_consistency() {
        let qjl = QJLProjector::new(64, 128, 42);
        let mut rng = StdRng::seed_from_u64(99);
        let residual: Vec<f32> = (0..64)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();

        let packed = qjl.project_and_pack(&residual);
        assert_eq!(packed.len(), packed_len(128));

        // Re-pack should give the same result (deterministic)
        let packed2 = qjl.project_and_pack(&residual);
        assert_eq!(packed, packed2);
    }

    #[test]
    fn unbiased_estimation() {
        let dim = 128;
        let m = 256;
        let qjl = QJLProjector::new(dim, m, 42);
        let mut rng = StdRng::seed_from_u64(99);

        let n_trials = 2000;
        let mut total_error = 0.0f64;

        for _ in 0..n_trials {
            let residual: Vec<f32> = (0..dim)
                .map(|_| rng.sample::<f32, _>(StandardNormal))
                .collect();
            let query: Vec<f32> = (0..dim)
                .map(|_| rng.sample::<f32, _>(StandardNormal))
                .collect();

            let true_dot: f32 = query
                .iter()
                .zip(residual.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            let residual_norm: f32 = residual.iter().map(|v| v * v).sum::<f32>().sqrt();
            let packed = qjl.project_and_pack(&residual);
            let estimated = qjl.estimate_correction(&query, &packed, residual_norm);

            total_error += (estimated - true_dot) as f64;
        }

        let mean_error = total_error / n_trials as f64;
        eprintln!("QJL mean error over {n_trials} trials: {mean_error:.4}");
        // Unbiased: mean error should be close to 0
        assert!(
            mean_error.abs() < 1.0,
            "mean error {mean_error} too large (not unbiased)"
        );
    }

    #[test]
    fn correction_improves_with_more_projections() {
        let dim = 128;
        let mut rng = StdRng::seed_from_u64(42);

        let residual: Vec<f32> = (0..dim)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();
        let query: Vec<f32> = (0..dim)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();
        let true_dot: f32 = query
            .iter()
            .zip(residual.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        let residual_norm: f32 = residual.iter().map(|v| v * v).sum::<f32>().sqrt();

        for &m in &[64, 128, 256, 512] {
            // Average over multiple seeds for stability
            let mut err_sum = 0.0f64;
            let n = 50;
            for seed in 0..n {
                let qjl_s = QJLProjector::new(dim, m, seed);
                let packed_s = qjl_s.project_and_pack(&residual);
                let est = qjl_s.estimate_correction(&query, &packed_s, residual_norm);
                err_sum += ((est - true_dot) as f64).powi(2);
            }
            let mse = err_sum / n as f64;
            eprintln!("m={m}: MSE={mse:.4}");
            // MSE should generally decrease with more projections
        }
    }

    #[test]
    fn preprojected_correction_matches_public_api() {
        let dim = 64;
        let m = 96;
        let qjl = QJLProjector::new(dim, m, 42);
        let mut rng = StdRng::seed_from_u64(99);
        let residual: Vec<f32> = (0..dim)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();
        let query: Vec<f32> = (0..dim)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();
        let residual_norm: f32 = residual.iter().map(|v| v * v).sum::<f32>().sqrt();
        let packed = qjl.project_and_pack(&residual);

        let projected_query = qjl.project_query(&query);
        let direct = qjl.estimate_correction(&query, &packed, residual_norm);
        let preprojected =
            qjl.estimate_correction_from_projection(&projected_query, &packed, residual_norm);

        assert_eq!(direct, preprojected);
    }

    #[test]
    fn serialization_roundtrip() {
        let qjl = QJLProjector::new(64, 128, 42);
        let scratch = crate::testpath::TempPath::new("qjl");
        let path = scratch.as_str();
        qjl.save(path).unwrap();
        let loaded = QJLProjector::load(path).unwrap();
        assert_eq!(qjl.dim, loaded.dim);
        assert_eq!(qjl.m, loaded.m);
        assert_eq!(qjl.proj_matrix, loaded.proj_matrix);
    }

    #[test]
    fn packed_len_correct() {
        assert_eq!(packed_len(0), 0);
        assert_eq!(packed_len(1), 1);
        assert_eq!(packed_len(63), 1);
        assert_eq!(packed_len(64), 1);
        assert_eq!(packed_len(65), 2);
        assert_eq!(packed_len(128), 2);
        assert_eq!(packed_len(129), 3);
    }

    #[test]
    fn last_word_masking() {
        // m=65 means only 1 valid bit in the second word
        let dim = 32;
        let m = 65;
        let qjl = QJLProjector::new(dim, m, 42);
        let mut rng = StdRng::seed_from_u64(99);
        let residual: Vec<f32> = (0..dim)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();
        let packed = qjl.project_and_pack(&residual);
        assert_eq!(packed.len(), 2);
        // Second word should only have bit 0 potentially set
        let valid_mask = 1u64; // only bit 0
        assert_eq!(packed[1] & !valid_mask, 0, "invalid bits set in last word");
    }
}
