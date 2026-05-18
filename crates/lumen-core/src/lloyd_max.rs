use std::f64::consts::{FRAC_1_SQRT_2, PI};
use std::io::{Read, Write};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CodebookError {
    #[error("invalid bit width {0}: must be 1..=8")]
    InvalidBits(u32),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid codebook file (bad magic)")]
    BadMagic,
}

/// Lloyd-Max optimal scalar quantizer for the standard normal distribution N(0,1).
///
/// Boundaries partition the real line into `n_levels` intervals.
/// Centroids are the conditional expectations E[X | boundary[i] <= X < boundary[i+1]].
#[derive(Debug, Clone)]
pub struct LloydMaxCodebook {
    /// Decision boundaries: `n_levels + 1` values, first = -INF, last = +INF.
    pub boundaries: Vec<f64>,
    /// Reconstruction centroids: `n_levels` values.
    pub centroids: Vec<f64>,
    /// Quantization bit width.
    pub bits: u32,
}

const MAGIC: &[u8; 4] = b"TQCB";

/// Standard normal PDF: phi(x) = exp(-x^2/2) / sqrt(2*pi).
#[inline]
fn phi(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Standard normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2))).
#[inline]
fn capital_phi(x: f64) -> f64 {
    0.5 * (1.0 + erf(x * FRAC_1_SQRT_2))
}

/// Error function approximation (Abramowitz & Stegun 7.1.26, max error < 1.5e-7).
fn erf(x: f64) -> f64 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Conditional expectation E[X | a <= X < b] for X ~ N(0,1).
///
/// Closed form: (phi(a) - phi(b)) / (Phi(b) - Phi(a)).
/// Returns 0.0 if the interval has negligible probability.
#[inline]
fn conditional_expectation(a: f64, b: f64) -> f64 {
    let prob = capital_phi(b) - capital_phi(a);
    if prob < 1e-15 {
        return 0.5 * (a + b); // fallback: midpoint
    }
    (phi(a) - phi(b)) / prob
}

impl LloydMaxCodebook {
    /// Compute the Lloyd-Max codebook for N(0,1) using the EM algorithm.
    ///
    /// `bits`: quantization width (1..=8), yielding 2^bits levels.
    /// `n_iter`: number of EM iterations (1000 is sufficient for convergence).
    pub fn compute(bits: u32, n_iter: usize) -> Result<Self, CodebookError> {
        if bits == 0 || bits > 8 {
            return Err(CodebookError::InvalidBits(bits));
        }
        let n_levels = 1usize << bits;

        // Initialize centroids uniformly in [-3, 3] (covers 99.7% of N(0,1))
        let mut centroids: Vec<f64> = (0..n_levels)
            .map(|i| -3.0 + 6.0 * (i as f64 + 0.5) / n_levels as f64)
            .collect();

        let mut boundaries = vec![0.0f64; n_levels + 1];

        for _ in 0..n_iter {
            // Step 1: Update boundaries as midpoints between adjacent centroids
            boundaries[0] = f64::NEG_INFINITY;
            boundaries[n_levels] = f64::INFINITY;
            for i in 1..n_levels {
                boundaries[i] = 0.5 * (centroids[i - 1] + centroids[i]);
            }

            // Step 2: Update centroids as conditional expectations
            for i in 0..n_levels {
                let a = boundaries[i];
                let b = boundaries[i + 1];
                centroids[i] = conditional_expectation(a, b);
            }
        }

        // Enforce symmetry: N(0,1) is symmetric, so c[i] = -c[n-1-i]
        let half = n_levels / 2;
        for i in 0..half {
            let avg = 0.5 * (centroids[i].abs() + centroids[n_levels - 1 - i].abs());
            centroids[i] = -avg;
            centroids[n_levels - 1 - i] = avg;
        }
        // Recompute boundaries from symmetrized centroids
        boundaries[0] = f64::NEG_INFINITY;
        boundaries[n_levels] = f64::INFINITY;
        for i in 1..n_levels {
            boundaries[i] = 0.5 * (centroids[i - 1] + centroids[i]);
        }

        Ok(Self {
            boundaries,
            centroids,
            bits,
        })
    }

    /// Number of quantization levels (2^bits).
    #[inline]
    pub fn n_levels(&self) -> usize {
        1 << self.bits
    }

    /// Quantize a scalar to a code index via binary search on boundaries.
    #[inline]
    pub fn quantize(&self, x: f64) -> u8 {
        // boundaries[0] = -inf, boundaries[n] = +inf
        // Find i such that boundaries[i] <= x < boundaries[i+1]
        let n = self.n_levels();
        // Linear scan is faster than binary search for small n (2..=256)
        for i in 1..n {
            if x < self.boundaries[i] {
                return (i - 1) as u8;
            }
        }
        (n - 1) as u8
    }

    /// Dequantize a code index to the centroid value.
    #[inline]
    pub fn dequantize(&self, code: u8) -> f64 {
        self.centroids[code as usize]
    }

    /// Quantize a vector of f32 scalars.
    pub fn quantize_vec(&self, xs: &[f32]) -> Vec<u8> {
        xs.iter().map(|&x| self.quantize(x as f64)).collect()
    }

    /// Dequantize a vector of codes to f32.
    pub fn dequantize_vec(&self, codes: &[u8]) -> Vec<f32> {
        codes.iter().map(|&c| self.dequantize(c) as f32).collect()
    }

    /// Compute MSE of this codebook on N(0,1).
    ///
    /// MSE = sum_i integral_{b_i}^{b_{i+1}} (x - c_i)^2 * phi(x) dx
    ///     = sum_i [ E[X^2|i]*P(i) - 2*c_i*E[X|i]*P(i) + c_i^2*P(i) ]
    pub fn mse(&self) -> f64 {
        let n = self.n_levels();
        let mut total = 0.0;
        for i in 0..n {
            let a = self.boundaries[i];
            let b = self.boundaries[i + 1];
            let prob = capital_phi(b) - capital_phi(a);
            if prob < 1e-15 {
                continue;
            }
            let c = self.centroids[i];
            // E[X|i] = (phi(a) - phi(b)) / prob
            let ex = (phi(a) - phi(b)) / prob;
            // E[X^2|i] = 1 + (a*phi(a) - b*phi(b)) / prob
            let a_phi = if a.is_finite() { a * phi(a) } else { 0.0 };
            let b_phi = if b.is_finite() { b * phi(b) } else { 0.0 };
            let ex2 = 1.0 + (a_phi - b_phi) / prob;
            total += prob * (ex2 - 2.0 * c * ex + c * c);
        }
        total
    }

    /// Save codebook to binary file.
    ///
    /// Format: magic(4) + bits(u32 LE) + n_levels(u32 LE) + boundaries(f64 LE) + centroids(f64 LE).
    pub fn save(&self, path: &str) -> Result<(), CodebookError> {
        let mut f = std::fs::File::create(path)?;
        f.write_all(MAGIC)?;
        f.write_all(&self.bits.to_le_bytes())?;
        let n = self.n_levels() as u32;
        f.write_all(&n.to_le_bytes())?;
        for &b in &self.boundaries {
            f.write_all(&b.to_le_bytes())?;
        }
        for &c in &self.centroids {
            f.write_all(&c.to_le_bytes())?;
        }
        Ok(())
    }

    /// Load codebook from binary file.
    pub fn load(path: &str) -> Result<Self, CodebookError> {
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(CodebookError::BadMagic);
        }

        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let bits = u32::from_le_bytes(buf4);

        f.read_exact(&mut buf4)?;
        let n_levels = u32::from_le_bytes(buf4) as usize;

        let mut buf8 = [0u8; 8];
        let mut boundaries = Vec::with_capacity(n_levels + 1);
        for _ in 0..=n_levels {
            f.read_exact(&mut buf8)?;
            boundaries.push(f64::from_le_bytes(buf8));
        }

        let mut centroids = Vec::with_capacity(n_levels);
        for _ in 0..n_levels {
            f.read_exact(&mut buf8)?;
            centroids.push(f64::from_le_bytes(buf8));
        }

        Ok(Self {
            boundaries,
            centroids,
            bits,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_3bit_mse() {
        let cb = LloydMaxCodebook::compute(3, 1000).unwrap();
        let mse = cb.mse();
        eprintln!("3-bit Lloyd-Max MSE: {mse:.6}");
        // Optimal 3-bit Lloyd-Max MSE for N(0,1) is ~0.0345; paper bound is 0.043
        assert!(mse <= 0.035, "MSE {mse} exceeds threshold 0.035");
    }

    #[test]
    fn codebook_symmetry() {
        let cb = LloydMaxCodebook::compute(3, 1000).unwrap();
        let n = cb.n_levels();
        // Centroids should be symmetric around 0 for N(0,1)
        for i in 0..n / 2 {
            let c_lo = cb.centroids[i];
            let c_hi = cb.centroids[n - 1 - i];
            assert!(
                (c_lo + c_hi).abs() < 1e-10,
                "symmetry violation: c[{i}]={c_lo}, c[{}]={c_hi}",
                n - 1 - i
            );
        }
    }

    #[test]
    fn quantize_dequantize_roundtrip() {
        let cb = LloydMaxCodebook::compute(3, 1000).unwrap();
        // Each centroid should quantize to its own index
        for (i, &c) in cb.centroids.iter().enumerate() {
            let code = cb.quantize(c);
            assert_eq!(code, i as u8, "centroid {i} maps to wrong code {code}");
        }
    }

    #[test]
    fn vec_quantize_roundtrip() {
        let cb = LloydMaxCodebook::compute(3, 1000).unwrap();
        let xs: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let codes = cb.quantize_vec(&xs);
        let recon = cb.dequantize_vec(&codes);
        // Reconstruction should be close to input for values near centroids
        for (x, r) in xs.iter().zip(recon.iter()) {
            assert!((x - r).abs() < 1.0, "large error: x={x}, recon={r}");
        }
    }

    #[test]
    fn empirical_mse_matches_analytical() {
        use rand::prelude::*;
        use rand_distr::StandardNormal;

        let cb = LloydMaxCodebook::compute(3, 1000).unwrap();
        let mut rng = StdRng::seed_from_u64(123);
        let n = 100_000;
        let mut mse_sum = 0.0f64;
        for _ in 0..n {
            let x: f64 = rng.sample(StandardNormal);
            let code = cb.quantize(x);
            let recon = cb.dequantize(code);
            mse_sum += (x - recon) * (x - recon);
        }
        let empirical_mse = mse_sum / n as f64;
        let analytical_mse = cb.mse();
        eprintln!("empirical MSE: {empirical_mse:.6}, analytical: {analytical_mse:.6}");
        assert!(
            (empirical_mse - analytical_mse).abs() < 0.002,
            "empirical {empirical_mse} != analytical {analytical_mse}"
        );
        assert!(
            empirical_mse <= 0.036,
            "empirical MSE {empirical_mse} too high"
        );
    }

    #[test]
    fn serialization_roundtrip() {
        let cb = LloydMaxCodebook::compute(3, 1000).unwrap();
        let path = "/tmp/tq_codebook_test.bin";
        cb.save(path).unwrap();
        let loaded = LloydMaxCodebook::load(path).unwrap();
        assert_eq!(cb.bits, loaded.bits);
        assert_eq!(cb.boundaries.len(), loaded.boundaries.len());
        for (a, b) in cb.centroids.iter().zip(loaded.centroids.iter()) {
            assert!((a - b).abs() < 1e-15, "centroid mismatch: {a} != {b}");
        }
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn codebook_2bit_and_4bit() {
        for bits in [2, 4] {
            let cb = LloydMaxCodebook::compute(bits, 1000).unwrap();
            let mse = cb.mse();
            eprintln!("{bits}-bit Lloyd-Max MSE: {mse:.6}");
            assert_eq!(cb.n_levels(), 1 << bits);
            // 4-bit should have lower MSE than 3-bit
            if bits == 4 {
                assert!(mse < 0.01, "{bits}-bit MSE {mse} unexpectedly high");
            }
        }
    }
}
