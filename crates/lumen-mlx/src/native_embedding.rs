//! Native MLX embedding lookup.
//!
//! Two paths exist in production:
//! * **Plain f32 / bf16 / f16 weight** — plain `mx.take(weight, ids, axis=0)`,
//!   used when the embedding table is stored unquantized.
//! * **MXFP4-quantized weight** — the production Qwen3.6-35B-A3B-mxfp4
//!   `language_model.model.embed_tokens` is MXFP4. mlx-rs's stock
//!   `nn::QuantizedEmbedding::forward` calls `dequantize` *without* a mode
//!   parameter (hardcodes `affine`), so it would silently corrupt MXFP4
//!   weights. We replicate the upstream forward path manually but route
//!   through `dequantize_with_mode(..., MODE_MXFP4)` instead.
//!
//! See `.ai/memory/active/mlx-rs-native-port/CONTEXT.md` Session 21 for
//! Phase 3b's overall plan.

#[cfg(feature = "mlx-native")]
#[allow(dead_code)] // `embedding_lookup` (plain f32 path) is held alongside `quantized_embedding_lookup_with_mode` for future plain-embedding callsites
mod imp {
    use anyhow::{Context, Result};
    use mlx_rs::Array;
    use std::ffi::CStr;

    use crate::native_quant::dequantize_with_mode;

    /// Plain embedding lookup: `weight[token_ids]` along axis 0.
    ///
    /// `weight`: `[vocab, hidden]` of any float dtype mlx-rs supports.
    /// `token_ids`: `[..., n_tokens]` of int (i32 expected for MLX `take`).
    /// Returns: `[..., n_tokens, hidden]`.
    pub fn embedding_lookup(weight: &Array, token_ids: &Array) -> Result<Array> {
        weight
            .take_axis(token_ids, 0)
            .context("mlx-rs take_axis (plain embedding lookup) failed")
    }

    /// MXFP4-quantized embedding lookup. Mirrors `nn::QuantizedEmbedding::forward`
    /// row-selection-then-dequantize pattern but with explicit `mode="mxfp4"`.
    ///
    /// `packed`: `[vocab, hidden / 8]` u32 (MXFP4 packed nibbles).
    /// `scales`: `[vocab, hidden / 32]` u8 (E8M0 shared exponents).
    /// `token_ids`: `[..., n_tokens]` of int.
    /// Returns: `[..., n_tokens, hidden]` in `bf16` (MLX's mxfp4 dequant
    /// default — caller can `.as_dtype(...)` if needed).
    ///
    /// Indexing the *small* selected slice (rather than dequantizing the full
    /// table) keeps memory and bandwidth proportional to the active token count.
    pub fn quantized_embedding_lookup_with_mode(
        packed: &Array,
        scales: &Array,
        token_ids: &Array,
        group_size: i32,
        bits: i32,
        mode: &CStr,
    ) -> Result<Array> {
        let selected_packed = packed
            .take_axis(token_ids, 0)
            .context("mlx-rs take_axis (mxfp4 packed rows) failed")?;
        let selected_scales = scales
            .take_axis(token_ids, 0)
            .context("mlx-rs take_axis (mxfp4 scales rows) failed")?;
        dequantize_with_mode(
            &selected_packed,
            &selected_scales,
            None,
            group_size,
            bits,
            mode,
        )
    }
}

#[cfg(feature = "mlx-native")]
#[allow(unused_imports)] // Consumed by Phase 3b model assembly in runner_native.rs.
pub(crate) use imp::{embedding_lookup, quantized_embedding_lookup_with_mode};

// bit-identical embedding lookup vs MLX reference.
//
// Fixtures are produced by `scripts/generate_embedding_fixture.py`:
//   embed_plain_<name>.bin   (magic TQEP)
//   embed_mxfp4_<name>.bin   (magic TQEM)
//
// `#[ignore]` because MLX FFI requires a real Metal device (Issue #5 in
// active/mlx-rs-native-port/CHECKLIST.md). Run outside the sandbox:
//
//   cargo test -p lumen-mlx --features mlx-native \
//       native_embedding::parity_tests:: -- --ignored
#[cfg(all(test, feature = "mlx-native"))]
mod parity_tests {
    use super::imp::{embedding_lookup, quantized_embedding_lookup_with_mode};
    use crate::native_quant::MODE_MXFP4;
    use mlx_rs::Array;
    use std::path::{Path, PathBuf};

    const PLAIN_MAGIC: &[u8; 4] = b"TQEP";
    const PLAIN_HEADER_LEN: usize = 4 /* magic */ + 4 * 4 /* 4 u32 */;
    const MXFP4_MAGIC: &[u8; 4] = b"TQEM";
    const MXFP4_HEADER_LEN: usize = 4 /* magic */ + 7 * 4 /* 7 u32 */;
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

    fn ids_to_i32(ids_u32: &[u32]) -> Vec<i32> {
        // MLX's `take` axis indices want signed int32. Fixture stores u32
        // for compactness; vocab fits well within i32 range for our shapes.
        ids_u32.iter().map(|&v| v as i32).collect()
    }

    struct PlainFixture {
        vocab: usize,
        hidden: usize,
        n_tokens: usize,
        weight: Vec<f32>,
        ids: Vec<u32>,
        expected_y: Vec<f32>,
    }

    fn load_plain_fixture(name: &str) -> Option<PlainFixture> {
        let path = fixture_dir().join(format!("embed_plain_{name}.bin"));
        let bytes = std::fs::read(&path).ok()?;
        if bytes.len() < PLAIN_HEADER_LEN || &bytes[..4] != PLAIN_MAGIC {
            panic!("plain fixture {path:?} has bad header");
        }
        let vocab = read_u32_le(&bytes, 4) as usize;
        let hidden = read_u32_le(&bytes, 8) as usize;
        let n_tokens = read_u32_le(&bytes, 12) as usize;
        let dtype_tag = read_u32_le(&bytes, 16);
        assert_eq!(dtype_tag, 0, "{name}: only f32 weight dtype tag supported");

        let weight_len = vocab * hidden;
        let mut cur = PLAIN_HEADER_LEN;
        let weight = slice_f32(&bytes[cur..cur + weight_len * 4]);
        cur += weight_len * 4;
        let ids = slice_u32(&bytes[cur..cur + n_tokens * 4]);
        cur += n_tokens * 4;
        let ref_len = n_tokens * hidden;
        let expected_y = slice_f32(&bytes[cur..cur + ref_len * 4]);

        Some(PlainFixture {
            vocab,
            hidden,
            n_tokens,
            weight,
            ids,
            expected_y,
        })
    }

    fn run_plain_fixture(name: &str) {
        let Some(fx) = load_plain_fixture(name) else {
            eprintln!(
                "SKIP plain/{name}: fixture missing under {}. Run `python scripts/generate_embedding_fixture.py` first.",
                fixture_dir().display()
            );
            return;
        };

        let weight = Array::from_slice(&fx.weight, &[fx.vocab as i32, fx.hidden as i32]);
        let ids_i32 = ids_to_i32(&fx.ids);
        let ids = Array::from_slice(&ids_i32, &[fx.n_tokens as i32]);

        let out = embedding_lookup(&weight, &ids).expect("plain embedding lookup must succeed");
        assert_eq!(
            out.shape(),
            &[fx.n_tokens as i32, fx.hidden as i32],
            "{name}: plain output shape mismatch"
        );
        assert_eq!(
            out.dtype(),
            mlx_rs::Dtype::Float32,
            "{name}: plain output dtype unexpectedly changed"
        );
        out.eval().expect("mlx eval must succeed");

        let observed: &[f32] = out.as_slice();
        assert_eq!(
            observed.len(),
            fx.expected_y.len(),
            "{name}: plain output flat-size mismatch"
        );

        let mut mismatches = 0usize;
        for (i, (&got, &expected)) in observed.iter().zip(fx.expected_y.iter()).enumerate() {
            if got.to_bits() != expected.to_bits() {
                if mismatches < 5 {
                    eprintln!(
                        "plain/{name}[{i}]: 0x{:08x} ({got}) vs ref 0x{:08x} ({expected})",
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

    struct MxfpFixture {
        vocab: usize,
        hidden: usize,
        n_tokens: usize,
        group_size: i32,
        bits: i32,
        packed: Vec<u32>,
        scales: Vec<u8>,
        ids: Vec<u32>,
        expected_y: Vec<f32>,
    }

    fn load_mxfp4_fixture(name: &str) -> Option<MxfpFixture> {
        let path = fixture_dir().join(format!("embed_mxfp4_{name}.bin"));
        let bytes = std::fs::read(&path).ok()?;
        if bytes.len() < MXFP4_HEADER_LEN || &bytes[..4] != MXFP4_MAGIC {
            panic!("mxfp4 fixture {path:?} has bad header");
        }
        let vocab = read_u32_le(&bytes, 4) as usize;
        let hidden = read_u32_le(&bytes, 8) as usize;
        let n_tokens = read_u32_le(&bytes, 12) as usize;
        let group_size = read_u32_le(&bytes, 16) as i32;
        let bits = read_u32_le(&bytes, 20) as i32;
        let packed_len = read_u32_le(&bytes, 24) as usize;
        let scales_len = read_u32_le(&bytes, 28) as usize;
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

        let mut cur = MXFP4_HEADER_LEN;
        let packed = slice_u32(&bytes[cur..cur + packed_len * 4]);
        cur += packed_len * 4;
        let scales = bytes[cur..cur + scales_len].to_vec();
        cur += scales_len;
        let ids = slice_u32(&bytes[cur..cur + n_tokens * 4]);
        cur += n_tokens * 4;
        let ref_len = n_tokens * hidden;
        let expected_y = slice_f32(&bytes[cur..cur + ref_len * 4]);

        Some(MxfpFixture {
            vocab,
            hidden,
            n_tokens,
            group_size,
            bits,
            packed,
            scales,
            ids,
            expected_y,
        })
    }

    fn run_mxfp4_fixture(name: &str) {
        let Some(fx) = load_mxfp4_fixture(name) else {
            eprintln!(
                "SKIP mxfp4/{name}: fixture missing under {}. Run `python scripts/generate_embedding_fixture.py` first.",
                fixture_dir().display()
            );
            return;
        };

        assert_eq!(
            fx.group_size, GROUP_SIZE,
            "{name}: fixture group_size unexpected"
        );
        assert_eq!(fx.bits, BITS, "{name}: fixture bits unexpected");

        let packed = Array::from_slice(&fx.packed, &[fx.vocab as i32, (fx.hidden / 8) as i32]);
        let scales = Array::from_slice(&fx.scales, &[fx.vocab as i32, (fx.hidden / 32) as i32]);
        let ids_i32 = ids_to_i32(&fx.ids);
        let ids = Array::from_slice(&ids_i32, &[fx.n_tokens as i32]);

        let bf16 = quantized_embedding_lookup_with_mode(
            &packed, &scales, &ids, GROUP_SIZE, BITS, MODE_MXFP4,
        )
        .expect("mxfp4 quantized embedding lookup must succeed");
        assert_eq!(
            bf16.shape(),
            &[fx.n_tokens as i32, fx.hidden as i32],
            "{name}: mxfp4 output shape mismatch"
        );
        assert_eq!(
            bf16.dtype(),
            mlx_rs::Dtype::Bfloat16,
            "{name}: mxfp4 dequant default dtype unexpectedly changed"
        );

        let out = bf16
            .as_dtype(mlx_rs::Dtype::Float32)
            .expect("bf16 → f32 cast must succeed");
        out.eval().expect("mlx eval must succeed");

        let observed: &[f32] = out.as_slice();
        assert_eq!(
            observed.len(),
            fx.expected_y.len(),
            "{name}: mxfp4 output flat-size mismatch"
        );

        let mut mismatches = 0usize;
        for (i, (&got, &expected)) in observed.iter().zip(fx.expected_y.iter()).enumerate() {
            if got.to_bits() != expected.to_bits() {
                if mismatches < 5 {
                    eprintln!(
                        "mxfp4/{name}[{i}]: 0x{:08x} ({got}) vs ref 0x{:08x} ({expected})",
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
    #[ignore = "MLX FFI requires non-sandbox host with Metal device; run with `cargo test -p lumen-mlx --features mlx-native -- --ignored`"]
    fn plain_tiny_embedding_matches_mlx() {
        run_plain_fixture("tiny");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn plain_small_embedding_matches_mlx() {
        run_plain_fixture("small");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn mxfp4_tiny_embedding_matches_mlx() {
        run_mxfp4_fixture("tiny");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn mxfp4_small_embedding_matches_mlx() {
        run_mxfp4_fixture("small");
    }
}
