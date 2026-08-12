use std::io::{Read, Write};

use rand::prelude::*;
use rand_distr::StandardNormal;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RotationError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid rotation file (bad magic)")]
    BadMagic,
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
}

/// Random orthogonal matrix for rotating KV vectors before quantization.
///
/// Generated via QR decomposition of a random Gaussian matrix (Haar-distributed).
/// Stored as a flat `dim x dim` row-major f32 array.
#[derive(Debug, Clone)]
pub struct RotationMatrix {
    /// Row-major `dim x dim` matrix.
    pub matrix: Vec<f32>,
    pub dim: usize,
}

const MAGIC: &[u8; 4] = b"TQRM";

impl RotationMatrix {
    /// Generate a Haar-distributed random orthogonal matrix via QR decomposition.
    ///
    /// Uses the Stewart (1980) / Mezzadri (2007) method:
    /// 1. Fill dim x dim matrix with iid N(0,1) entries
    /// 2. Compute QR decomposition via modified Gram-Schmidt
    /// 3. Fix signs: Q[:,i] *= sign(R[i,i]) for Haar distribution
    pub fn random(dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        // Generate random Gaussian matrix (column-major for Gram-Schmidt)
        let mut cols: Vec<Vec<f64>> = (0..dim)
            .map(|_| {
                (0..dim)
                    .map(|_| rng.sample::<f64, _>(StandardNormal))
                    .collect()
            })
            .collect();

        // Modified Gram-Schmidt QR with sign correction
        let mut r_diag = vec![0.0f64; dim];

        for j in 0..dim {
            // Orthogonalize column j against all previous columns
            for k in 0..j {
                let dot: f64 = (0..dim).map(|i| cols[k][i] * cols[j][i]).sum();
                for i in 0..dim {
                    cols[j][i] -= dot * cols[k][i];
                }
            }

            // Normalize
            let norm: f64 = (0..dim)
                .map(|i| cols[j][i] * cols[j][i])
                .sum::<f64>()
                .sqrt();
            r_diag[j] = norm;
            let inv_norm = 1.0 / norm;
            for i in 0..dim {
                cols[j][i] *= inv_norm;
            }
        }

        // Mezzadri sign correction: multiply column j by sign(R[j,j])
        for j in 0..dim {
            let sign = r_diag[j].signum();
            for i in 0..dim {
                cols[j][i] *= sign;
            }
        }

        // Convert to row-major f32
        let mut matrix = vec![0.0f32; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                matrix[i * dim + j] = cols[j][i] as f32;
            }
        }

        Self { matrix, dim }
    }

    /// Apply rotation: y = R @ x.
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.dim);
        let mut y = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            let row = &self.matrix[i * self.dim..(i + 1) * self.dim];
            y[i] = row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
        }
        y
    }

    /// Apply inverse rotation: x = R^T @ y (orthogonal => inverse = transpose).
    pub fn apply_inverse(&self, y: &[f32]) -> Vec<f32> {
        debug_assert_eq!(y.len(), self.dim);
        let mut x = vec![0.0f32; self.dim];
        for j in 0..self.dim {
            let yj = y[j];
            let row = &self.matrix[j * self.dim..(j + 1) * self.dim];
            for i in 0..self.dim {
                x[i] += row[i] * yj;
            }
        }
        x
    }

    /// Apply rotation to a batch of n vectors: xs is [n, dim] row-major.
    pub fn apply_batch(&self, xs: &[f32], n: usize) -> Vec<f32> {
        debug_assert_eq!(xs.len(), n * self.dim);
        let mut out = vec![0.0f32; n * self.dim];
        for v in 0..n {
            let x = &xs[v * self.dim..(v + 1) * self.dim];
            let y = &mut out[v * self.dim..(v + 1) * self.dim];
            for i in 0..self.dim {
                let row = &self.matrix[i * self.dim..(i + 1) * self.dim];
                y[i] = row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            }
        }
        out
    }

    /// Save rotation matrix to binary file.
    ///
    /// Format: magic(4) + dim(u32 LE) + seed(u64 LE) + f32 array (row-major LE).
    pub fn save(&self, path: &str) -> Result<(), RotationError> {
        let mut f = std::fs::File::create(path)?;
        f.write_all(MAGIC)?;
        f.write_all(&(self.dim as u32).to_le_bytes())?;
        for &val in &self.matrix {
            f.write_all(&val.to_le_bytes())?;
        }
        Ok(())
    }

    /// Load rotation matrix from binary file.
    pub fn load(path: &str) -> Result<Self, RotationError> {
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(RotationError::BadMagic);
        }

        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let dim = u32::from_le_bytes(buf4) as usize;

        let n_elems = dim * dim;
        let mut matrix = Vec::with_capacity(n_elems);
        for _ in 0..n_elems {
            f.read_exact(&mut buf4)?;
            matrix.push(f32::from_le_bytes(buf4));
        }

        Ok(Self { matrix, dim })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthogonality() {
        let r = RotationMatrix::random(128, 42);
        // R * R^T should be identity
        let mut max_err = 0.0f32;
        for i in 0..r.dim {
            for j in 0..r.dim {
                let dot: f32 = (0..r.dim)
                    .map(|k| r.matrix[i * r.dim + k] * r.matrix[j * r.dim + k])
                    .sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                max_err = max_err.max((dot - expected).abs());
            }
        }
        eprintln!("max orthogonality error: {max_err:.2e}");
        assert!(max_err < 1e-5, "orthogonality error {max_err} too large");
    }

    #[test]
    fn apply_inverse_roundtrip() {
        let r = RotationMatrix::random(128, 42);
        let mut rng = StdRng::seed_from_u64(99);
        let x: Vec<f32> = (0..128)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();

        let y = r.apply(&x);
        let x_recovered = r.apply_inverse(&y);

        let max_err: f32 = x
            .iter()
            .zip(x_recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        eprintln!("apply/inverse max error: {max_err:.2e}");
        assert!(max_err < 1e-4, "roundtrip error {max_err} too large");
    }

    #[test]
    fn norm_preservation() {
        let r = RotationMatrix::random(128, 42);
        let mut rng = StdRng::seed_from_u64(77);
        let x: Vec<f32> = (0..128)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();

        let norm_x: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let y = r.apply(&x);
        let norm_y: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();

        let rel_err = (norm_x - norm_y).abs() / norm_x;
        eprintln!("norm preservation relative error: {rel_err:.2e}");
        assert!(rel_err < 1e-5, "norm not preserved: {norm_x} vs {norm_y}");
    }

    #[test]
    fn deterministic_seed() {
        let r1 = RotationMatrix::random(64, 42);
        let r2 = RotationMatrix::random(64, 42);
        assert_eq!(r1.matrix, r2.matrix, "same seed should produce same matrix");
    }

    #[test]
    fn batch_matches_single() {
        let r = RotationMatrix::random(64, 42);
        let mut rng = StdRng::seed_from_u64(55);
        let n = 10;
        let xs: Vec<f32> = (0..n * 64)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();

        let batch_out = r.apply_batch(&xs, n);

        for v in 0..n {
            let single = r.apply(&xs[v * 64..(v + 1) * 64]);
            let batch_slice = &batch_out[v * 64..(v + 1) * 64];
            for (a, b) in single.iter().zip(batch_slice.iter()) {
                assert!((a - b).abs() < 1e-10, "batch/single mismatch");
            }
        }
    }

    #[test]
    fn serialization_roundtrip() {
        let r = RotationMatrix::random(64, 42);
        let scratch = crate::testpath::TempPath::new("rotation");
        let path = scratch.as_str();
        r.save(path).unwrap();
        let loaded = RotationMatrix::load(path).unwrap();
        assert_eq!(r.dim, loaded.dim);
        assert_eq!(r.matrix, loaded.matrix);
    }
}
