//! Native MLX normalization wrappers.
//!
//! Currently exposes `rms_norm` (the only normalization the Qwen3.5-MoE
//! architecture uses for both pre-attention and pre-MLP norms). Routed
//! through `mlx_rs::fast::rms_norm`, which is itself a thin wrapper over
//! `mlx_fast_rms_norm`. The op is deterministic for fixed inputs, so a
//! correct binding produces bit-identical output vs MLX's Python reference.
//!
//! See `.ai/memory/active/mlx-rs-native-port/CONTEXT.md` Session 21+.

#[cfg(feature = "mlx-native")]
mod imp {
    use anyhow::{Context, Result};
    use mlx_rs::Array;

    /// RMSNorm along the last axis: `y = x * rsqrt(mean(x*x, axis=-1) + eps) * weight`.
    pub fn rms_norm(x: &Array, weight: &Array, eps: f32) -> Result<Array> {
        mlx_rs::fast::rms_norm(x, weight, eps).context("mlx-rs fast::rms_norm FFI call failed")
    }
}

#[cfg(feature = "mlx-native")]
#[allow(unused_imports)] // Consumed by Phase 3b model assembly in runner_native.rs.
pub(crate) use imp::rms_norm;

// RMSNorm bit-identical vs MLX reference.
//
// Fixtures produced by `scripts/generate_rmsnorm_fixture.py`
// (magic `TQRN`, header: batch/seq/hidden/eps_bits + x/weight/ref payloads).
//
// `#[ignore]`'d — MLX FFI requires a real Metal device per Issue #5. Run
// outside the sandbox:
//
//   cargo test -p lumen-mlx --features mlx-native \
//       native_norm::parity_tests:: -- --ignored
#[cfg(all(test, feature = "mlx-native"))]
mod parity_tests {
    use super::imp::rms_norm;
    use mlx_rs::Array;
    use std::path::{Path, PathBuf};

    const MAGIC: &[u8; 4] = b"TQRN";
    const HEADER_LEN: usize = 4 /* magic */ + 4 * 4 /* 4 u32 */;

    fn fixture_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("lumen-mlx manifest dir has a parent (crates/)")
            .join("lumen-metal")
            .join("tests")
            .join("fixtures")
    }

    fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }

    fn slice_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    struct RmsNormFixture {
        batch: usize,
        seq: usize,
        hidden: usize,
        eps: f32,
        x: Vec<f32>,
        weight: Vec<f32>,
        expected_y: Vec<f32>,
    }

    fn load_fixture(name: &str) -> Option<RmsNormFixture> {
        let path = fixture_dir().join(format!("embed_rmsnorm_{name}.bin"));
        let bytes = std::fs::read(&path).ok()?;
        if bytes.len() < HEADER_LEN || &bytes[..4] != MAGIC {
            panic!("rmsnorm fixture {path:?} has bad header");
        }
        let batch = read_u32_le(&bytes, 4) as usize;
        let seq = read_u32_le(&bytes, 8) as usize;
        let hidden = read_u32_le(&bytes, 12) as usize;
        let eps_bits = read_u32_le(&bytes, 16);
        let eps = f32::from_bits(eps_bits);

        let element_count = batch * seq * hidden;
        let mut cur = HEADER_LEN;
        let x = slice_f32(&bytes[cur..cur + element_count * 4]);
        cur += element_count * 4;
        let weight = slice_f32(&bytes[cur..cur + hidden * 4]);
        cur += hidden * 4;
        let expected_y = slice_f32(&bytes[cur..cur + element_count * 4]);

        Some(RmsNormFixture {
            batch,
            seq,
            hidden,
            eps,
            x,
            weight,
            expected_y,
        })
    }

    fn run_fixture(name: &str) {
        let Some(fx) = load_fixture(name) else {
            eprintln!(
                "SKIP rmsnorm/{name}: fixture missing under {}. Run `python scripts/generate_rmsnorm_fixture.py` first.",
                fixture_dir().display()
            );
            return;
        };

        let x = Array::from_slice(&fx.x, &[fx.batch as i32, fx.seq as i32, fx.hidden as i32]);
        let weight = Array::from_slice(&fx.weight, &[fx.hidden as i32]);

        let out =
            rms_norm(&x, &weight, fx.eps).expect("mlx-rs fast::rms_norm FFI call must succeed");
        assert_eq!(
            out.shape(),
            &[fx.batch as i32, fx.seq as i32, fx.hidden as i32],
            "{name}: rms_norm output shape mismatch"
        );
        assert_eq!(
            out.dtype(),
            mlx_rs::Dtype::Float32,
            "{name}: rms_norm output dtype unexpectedly changed"
        );
        out.eval().expect("mlx eval must succeed");

        let observed: &[f32] = out.as_slice();
        assert_eq!(
            observed.len(),
            fx.expected_y.len(),
            "{name}: rms_norm output flat-size mismatch"
        );

        let mut mismatches = 0usize;
        for (i, (&got, &expected)) in observed.iter().zip(fx.expected_y.iter()).enumerate() {
            if got.to_bits() != expected.to_bits() {
                if mismatches < 5 {
                    eprintln!(
                        "rmsnorm/{name}[{i}]: 0x{:08x} ({got}) vs ref 0x{:08x} ({expected})",
                        got.to_bits(),
                        expected.to_bits()
                    );
                }
                mismatches += 1;
            }
        }
        assert_eq!(
            mismatches,
            0,
            "{name}: {mismatches}/{} elements diverged from MLX reference",
            observed.len()
        );
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn tiny_rmsnorm_matches_mlx() {
        run_fixture("tiny");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn multi_rmsnorm_matches_mlx() {
        run_fixture("multi");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn prod_hidden_rmsnorm_matches_mlx() {
        run_fixture("prod_hidden");
    }
}
