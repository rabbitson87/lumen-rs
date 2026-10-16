//! Helpers shared by the parity tests: reference-dump readers, the Klein
//! weight paths, and the element-wise comparison the op tests assert with.
//!
//! Callers keep the `/tmp/...` dump paths at the call site rather than in here,
//! so `cargo xtask tiers` still sees which test reads which dump.

/// The bf16 Klein DiT, resolved from the local HF cache.
pub(crate) fn klein_dit_path() -> String {
    crate::hf_cache::snapshot_path_or_rel(
        crate::repos::VAE,
        "transformer/diffusion_pytorch_model.safetensors",
    )
    .to_string_lossy()
    .into_owned()
}

/// The FLUX.2 VAE (shared by klein and dev), resolved from the local HF cache.
pub(crate) fn klein_vae_path() -> String {
    crate::hf_cache::snapshot_path_or_rel(
        crate::repos::VAE,
        "vae/diffusion_pytorch_model.safetensors",
    )
    .to_string_lossy()
    .into_owned()
}

/// A raw little-endian `f32` dump.
pub(crate) fn read_f32_bin(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert_eq!(bytes.len() % 4, 0, "{path} not f32-aligned");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A raw little-endian `i32` dump.
pub(crate) fn read_i32_bin(path: &str) -> Vec<i32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert_eq!(bytes.len() % 4, 0, "{path} not i32-aligned");
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Reorder a channels-first `[B,C,H,W]` flat buffer into channels-last
/// `[B,H,W,C]` flat order — pure index math, so we never have to materialize
/// a transposed mlx array (mlx `as_slice` reads the physical buffer and
/// ignores a lazy transpose's strides). The native decoder output is already
/// contiguous channels-last, so comparing in channels-last space is exact.
pub(crate) fn cf_to_cl(cf: &[f32], b: usize, c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; cf.len()];
    for bi in 0..b {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let src = ((bi * c + ci) * h + hi) * w + wi;
                    let dst = ((bi * h + hi) * w + wi) * c + ci;
                    out[dst] = cf[src];
                }
            }
        }
    }
    out
}

/// Assert `|got[i] - expected[i]| < tol` for every element, naming the first
/// one that is not as `label[i]`.
pub(crate) fn assert_all_close(label: &str, got: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        assert!((g - e).abs() < tol, "{label}[{i}]: got {g}, expected {e}");
    }
}
