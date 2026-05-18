//! Native MLX `lm_head` projection.
//!
//! `lm_head` projects the final hidden state to vocab logits via an MXFP4
//! quantized matmul:
//!
//! ```text
//! x: [batch, seq, hidden]   @  W.T   →   [batch, seq, vocab]
//! ```
//!
//! The op itself is `quantized_matmul_with_mode` (validated in Phase 3a.3 for
//! 1D `x`). This module exists to (a) document the lm_head call-site, and
//! (b) verify the wrapper handles the higher-rank activation `x` shape MLX
//! uses for batched prefill / decode.
//!
//! Activation `x` dtype determines output dtype (per Phase 3a.3 dtype rule
//! table — see CONTEXT.md). Production paths feed f32 hidden states for
//! deterministic logits.

#[cfg(feature = "mlx-native")]
#[allow(dead_code)] // `lm_head_mxfp4` is a thin wrapper retained for fixture parity tests + future direct callsites; live `forward()` path inlines the equivalent matmul
mod imp {
    use anyhow::Result;
    use mlx_rs::Array;

    use crate::native_quant::{MODE_MXFP4, quantized_matmul_with_mode};

    /// MXFP4 lm_head projection. Returns logits with shape
    /// `[..., vocab]` matching `x`'s leading dims.
    pub fn lm_head_mxfp4(
        x: &Array,
        packed_weight: &Array,
        scales: &Array,
        group_size: i32,
        bits: i32,
    ) -> Result<Array> {
        quantized_matmul_with_mode(
            x,
            packed_weight,
            scales,
            None,
            true, /* transpose: y = x @ deq(W).T */
            group_size,
            bits,
            MODE_MXFP4,
        )
    }
}

#[cfg(feature = "mlx-native")]
#[allow(unused_imports)] // Consumed by Phase 3b model assembly in runner_native.rs.
pub(crate) use imp::lm_head_mxfp4;

// lm_head bit-identical vs MLX reference for batched x.
#[cfg(all(test, feature = "mlx-native"))]
mod parity_tests {
    use super::imp::lm_head_mxfp4;
    use mlx_rs::Array;
    use std::path::{Path, PathBuf};

    const MAGIC: &[u8; 4] = b"TQLH";
    const HEADER_LEN: usize = 4 /* magic */ + 8 * 4 /* 8 u32 */;
    const GROUP_SIZE: i32 = 32;
    const BITS: i32 = 4;

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
    fn slice_u32(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    struct LmHeadFixture {
        batch: usize,
        seq: usize,
        hidden: usize,
        vocab: usize,
        group_size: i32,
        bits: i32,
        x: Vec<f32>,
        packed: Vec<u32>,
        scales: Vec<u8>,
        expected_y: Vec<f32>,
    }

    fn load_fixture(name: &str) -> Option<LmHeadFixture> {
        let path = fixture_dir().join(format!("embed_lm_head_{name}.bin"));
        let bytes = std::fs::read(&path).ok()?;
        if bytes.len() < HEADER_LEN || &bytes[..4] != MAGIC {
            panic!("lm_head fixture {path:?} has bad header");
        }
        let batch = read_u32_le(&bytes, 4) as usize;
        let seq = read_u32_le(&bytes, 8) as usize;
        let hidden = read_u32_le(&bytes, 12) as usize;
        let vocab = read_u32_le(&bytes, 16) as usize;
        let group_size = read_u32_le(&bytes, 20) as i32;
        let bits = read_u32_le(&bytes, 24) as i32;
        let packed_len = read_u32_le(&bytes, 28) as usize;
        let scales_len = read_u32_le(&bytes, 32) as usize;
        assert_eq!(
            packed_len,
            vocab * hidden / 8,
            "{name}: packed_len mismatch"
        );
        assert_eq!(
            scales_len,
            vocab * hidden / 32,
            "{name}: scales_len mismatch"
        );

        let x_count = batch * seq * hidden;
        let mut cur = HEADER_LEN;
        let x = slice_f32(&bytes[cur..cur + x_count * 4]);
        cur += x_count * 4;
        let packed = slice_u32(&bytes[cur..cur + packed_len * 4]);
        cur += packed_len * 4;
        let scales = bytes[cur..cur + scales_len].to_vec();
        cur += scales_len;
        let y_count = batch * seq * vocab;
        let expected_y = slice_f32(&bytes[cur..cur + y_count * 4]);

        Some(LmHeadFixture {
            batch,
            seq,
            hidden,
            vocab,
            group_size,
            bits,
            x,
            packed,
            scales,
            expected_y,
        })
    }

    fn run_fixture(name: &str) {
        let Some(fx) = load_fixture(name) else {
            eprintln!(
                "SKIP lm_head/{name}: fixture missing under {}. Run `python scripts/generate_lm_head_fixture.py` first.",
                fixture_dir().display()
            );
            return;
        };
        assert_eq!(
            fx.group_size, GROUP_SIZE,
            "{name}: fixture group_size unexpected"
        );
        assert_eq!(fx.bits, BITS, "{name}: fixture bits unexpected");

        let x = Array::from_slice(&fx.x, &[fx.batch as i32, fx.seq as i32, fx.hidden as i32]);
        let w = Array::from_slice(&fx.packed, &[fx.vocab as i32, (fx.hidden / 8) as i32]);
        let s = Array::from_slice(&fx.scales, &[fx.vocab as i32, (fx.hidden / 32) as i32]);

        let y_raw =
            lm_head_mxfp4(&x, &w, &s, GROUP_SIZE, BITS).expect("lm_head mxfp4 matmul must succeed");
        assert_eq!(
            y_raw.shape(),
            &[fx.batch as i32, fx.seq as i32, fx.vocab as i32],
            "{name}: lm_head output shape mismatch"
        );

        // Output dtype follows x dtype (f32 here) per Phase 3a.3 finding.
        let y = y_raw
            .as_dtype(mlx_rs::Dtype::Float32)
            .expect("output → f32 cast must succeed");
        y.eval().expect("mlx eval must succeed");

        let observed: &[f32] = y.as_slice();
        assert_eq!(
            observed.len(),
            fx.expected_y.len(),
            "{name}: lm_head output flat-size mismatch"
        );

        let mut mismatches = 0usize;
        for (i, (&got, &expected)) in observed.iter().zip(fx.expected_y.iter()).enumerate() {
            if got.to_bits() != expected.to_bits() {
                if mismatches < 5 {
                    eprintln!(
                        "lm_head/{name}[{i}]: 0x{:08x} ({got}) vs ref 0x{:08x} ({expected})",
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
    fn tiny_lm_head_matches_mlx() {
        run_fixture("tiny");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn prefill_lm_head_matches_mlx() {
        run_fixture("prefill");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn wide_hidden_lm_head_matches_mlx() {
        run_fixture("wide_hidden");
    }
}
