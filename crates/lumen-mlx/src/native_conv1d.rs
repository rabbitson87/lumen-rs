//! Native MLX Conv1d wrapper (depthwise + dense).
//!
//! The Qwen3.5-MoE linear-attention block uses a depthwise Conv1d
//! (groups == in_channels == out_channels) with kernel_size=4, no padding,
//! followed by SiLU activation. This module wraps `mlx_rs::ops::conv1d`
//! and validates parity for both the dense and depthwise regimes.
//!
//! MLX layout matches `nn.Conv1d`:
//!     array:   [batch, length, in_channels]
//!     weight:  [out_channels, kernel_size, in_channels / groups]
//!     output:  [batch, out_length, out_channels]
//!
//! See `.ai/memory/active/mlx-rs-native-port/CONTEXT.md` Session 22+.

#[cfg(feature = "mlx-native")]
mod imp {
    use anyhow::{Context, Result};
    use mlx_rs::Array;

    /// 1D convolution. `groups == in_channels` selects the depthwise regime.
    pub fn conv1d(
        x: &Array,
        weight: &Array,
        stride: i32,
        padding: i32,
        groups: i32,
    ) -> Result<Array> {
        mlx_rs::ops::conv1d(x, weight, stride, padding, /* dilation */ 1, groups)
            .context("mlx-rs ops::conv1d FFI call failed")
    }
}

#[cfg(feature = "mlx-native")]
#[allow(unused_imports)] // Consumed by Phase 3b.5 SSM block assembly.
pub(crate) use imp::conv1d;

// Conv1d bit-identical vs MLX reference.
//
// Fixtures produced by `scripts/generate_conv1d_fixture.py` (magic `TQCV`).
#[cfg(all(test, feature = "mlx-native"))]
mod parity_tests {
    use super::imp::conv1d;
    use mlx_rs::Array;
    use std::path::{Path, PathBuf};

    const MAGIC: &[u8; 4] = b"TQCV";
    const HEADER_LEN: usize = 4 /* magic */ + 8 * 4 /* 8 u32 */;

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

    struct Conv1dFixture {
        batch: usize,
        length: usize,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: i32,
        padding: i32,
        groups: i32,
        x: Vec<f32>,
        weight: Vec<f32>,
        expected_y: Vec<f32>,
    }

    fn load_fixture(name: &str) -> Option<Conv1dFixture> {
        let path = fixture_dir().join(format!("embed_conv1d_{name}.bin"));
        let bytes = std::fs::read(&path).ok()?;
        if bytes.len() < HEADER_LEN || &bytes[..4] != MAGIC {
            panic!("conv1d fixture {path:?} has bad header");
        }
        let batch = read_u32_le(&bytes, 4) as usize;
        let length = read_u32_le(&bytes, 8) as usize;
        let in_channels = read_u32_le(&bytes, 12) as usize;
        let out_channels = read_u32_le(&bytes, 16) as usize;
        let kernel_size = read_u32_le(&bytes, 20) as usize;
        let stride = read_u32_le(&bytes, 24) as i32;
        let padding = read_u32_le(&bytes, 28) as i32;
        let groups = read_u32_le(&bytes, 32) as i32;

        let groups_usize = groups as usize;
        let weight_per_group = in_channels / groups_usize;
        let x_count = batch * length * in_channels;
        let weight_count = out_channels * kernel_size * weight_per_group;
        let out_length = (length + 2 * padding as usize - kernel_size) / stride as usize + 1;
        let y_count = batch * out_length * out_channels;

        let mut cur = HEADER_LEN;
        let x = slice_f32(&bytes[cur..cur + x_count * 4]);
        cur += x_count * 4;
        let weight = slice_f32(&bytes[cur..cur + weight_count * 4]);
        cur += weight_count * 4;
        let expected_y = slice_f32(&bytes[cur..cur + y_count * 4]);

        Some(Conv1dFixture {
            batch,
            length,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            x,
            weight,
            expected_y,
        })
    }

    fn run_fixture(name: &str) {
        let Some(fx) = load_fixture(name) else {
            eprintln!(
                "SKIP conv1d/{name}: fixture missing under {}. Run `python scripts/generate_conv1d_fixture.py` first.",
                fixture_dir().display()
            );
            return;
        };

        let weight_per_group = fx.in_channels / fx.groups as usize;
        let x = Array::from_slice(
            &fx.x,
            &[fx.batch as i32, fx.length as i32, fx.in_channels as i32],
        );
        let w = Array::from_slice(
            &fx.weight,
            &[
                fx.out_channels as i32,
                fx.kernel_size as i32,
                weight_per_group as i32,
            ],
        );

        let out = conv1d(&x, &w, fx.stride, fx.padding, fx.groups)
            .expect("mlx-rs ops::conv1d FFI call must succeed");
        let out_length =
            (fx.length + 2 * fx.padding as usize - fx.kernel_size) / fx.stride as usize + 1;
        assert_eq!(
            out.shape(),
            &[fx.batch as i32, out_length as i32, fx.out_channels as i32],
            "{name}: conv1d output shape mismatch"
        );
        assert_eq!(
            out.dtype(),
            mlx_rs::Dtype::Float32,
            "{name}: conv1d output dtype unexpectedly changed"
        );
        out.eval().expect("mlx eval must succeed");

        let observed: &[f32] = out.as_slice();
        assert_eq!(
            observed.len(),
            fx.expected_y.len(),
            "{name}: conv1d output flat-size mismatch"
        );

        let mut mismatches = 0usize;
        for (i, (&got, &expected)) in observed.iter().zip(fx.expected_y.iter()).enumerate() {
            if got.to_bits() != expected.to_bits() {
                if mismatches < 5 {
                    eprintln!(
                        "conv1d/{name}[{i}]: 0x{:08x} ({got}) vs ref 0x{:08x} ({expected})",
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
    fn tiny_dense_conv1d_matches_mlx() {
        run_fixture("tiny_dense");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn depthwise_small_conv1d_matches_mlx() {
        run_fixture("depthwise_small");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn depthwise_wide_conv1d_matches_mlx() {
        run_fixture("depthwise_wide");
    }
}
