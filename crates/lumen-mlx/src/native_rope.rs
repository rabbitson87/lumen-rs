//! Native MLX RoPE (Rotary Position Embedding) wrapper.
//!
//! Routes through `mlx_rs::fast::rope`, which calls `mlx_fast_rope` directly.
//! For Qwen3.5-MoE text-only decode, mRoPE collapses to standard
//! non-traditional RoPE (GPT-NeoX split form): only the first
//! `partial_rotary_factor * head_dim` features are rotated, the rest are
//! pass-through. `mx.fast.rope`'s `dimensions` parameter handles this
//! natively.
//!
//! See `.ai/memory/active/mlx-rs-native-port/CONTEXT.md` Session 21+.

#[cfg(feature = "mlx-native")]
mod imp {
    use anyhow::{Context, Result};
    use mlx_rs::Array;

    /// RoPE rotation along the last axis. The first `dimensions` features
    /// per position are rotated; remaining features (if `dimensions <
    /// last_axis_size`) pass through unchanged.
    ///
    /// `traditional=false` selects GPT-NeoX split form (modern); `true`
    /// selects the original interleaved form. Qwen3.5-MoE uses `false`.
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        x: &Array,
        dimensions: i32,
        traditional: bool,
        base: f32,
        scale: f32,
        offset: i32,
    ) -> Result<Array> {
        mlx_rs::fast::rope(
            x,
            dimensions,
            traditional,
            Some(base),
            scale,
            offset,
            None, /* freqs */
        )
        .context("mlx-rs fast::rope FFI call failed")
    }

    /// Same as [`rope`] but supplies a precomputed `freqs` array (shape
    /// `[dimensions / 2]`, dtype f32). MLX skips the per-call internal
    /// `arange + pow(base)` computation, dispatching the
    /// `rope_freqs_<dtype>_*` kernel instead of the default `rope_*`.
    /// Mirrors mlx-lm's pattern: freqs are built once at model load and
    /// reused for every decode step.
    pub fn rope_with_freqs(
        x: &Array,
        dimensions: i32,
        traditional: bool,
        scale: f32,
        offset: i32,
        freqs: &Array,
    ) -> Result<Array> {
        mlx_rs::fast::rope(
            x,
            dimensions,
            traditional,
            None, /* base — replaced by freqs */
            scale,
            offset,
            Some(freqs),
        )
        .context("mlx-rs fast::rope (with freqs) FFI call failed")
    }

    /// Precompute the RoPE per-pair frequency table MLX expects as the
    /// `freqs` argument to `mlx::fast::rope`. MLX applies `reciprocal()`
    /// internally to obtain its `inv_freqs`, so the user-supplied array
    /// must store the **forward** form: `freqs[i] = base^(2i / dimensions)`
    /// for `i in 0..dimensions/2`. This matches mlx-lm's convention
    /// (`base ** (mx.arange(0, dim, 2) / dim)`). Shape `[dimensions / 2]`,
    /// dtype f32 (MLX casts internally to match the input array dtype).
    pub fn precompute_rope_freqs(dimensions: i32, base: f32) -> Result<Array> {
        if dimensions <= 0 || dimensions % 2 != 0 {
            return Err(anyhow::anyhow!(
                "precompute_rope_freqs: dimensions ({dimensions}) must be a positive even integer"
            ));
        }
        let half = (dimensions / 2) as usize;
        let mut vals: Vec<f32> = Vec::with_capacity(half);
        for i in 0..half {
            // freqs[i] = base^(2i / dimensions). MLX inverts internally via
            // `reciprocal(freqs)` so the final per-position rotation angle
            // is `position * base^(-2i / dimensions)` — equivalent to the
            // default (no-freqs) path.
            let exp = (2 * i) as f32 / dimensions as f32;
            vals.push(base.powf(exp));
        }
        Ok(Array::from_slice(&vals, &[half as i32]))
    }
}

#[cfg(feature = "mlx-native")]
#[allow(unused_imports)] // Consumed by Phase 3b model assembly in runner_native.rs.
pub(crate) use imp::{precompute_rope_freqs, rope, rope_with_freqs};

// RoPE bit-identical vs MLX reference.
//
// Fixtures produced by `scripts/generate_rope_fixture.py` (magic `TQRP`).
//
// `#[ignore]`'d — MLX FFI requires non-sandbox host with Metal device.
#[cfg(all(test, feature = "mlx-native"))]
mod parity_tests {
    use super::imp::rope;
    use mlx_rs::Array;
    use std::path::{Path, PathBuf};

    const MAGIC: &[u8; 4] = b"TQRP";
    // 4 bytes magic + 6 u32 + 1 i32 + 2 u32 = 4 + 9*4 = 40
    const HEADER_LEN: usize = 4 + 9 * 4;

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
    fn read_i32_le(bytes: &[u8], offset: usize) -> i32 {
        i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }
    fn slice_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    struct RopeFixture {
        batch: usize,
        heads: usize,
        seq: usize,
        head_dim: usize,
        rope_dim: i32,
        traditional: bool,
        offset: i32,
        scale: f32,
        base: f32,
        x: Vec<f32>,
        expected_y: Vec<f32>,
    }

    fn load_fixture(name: &str) -> Option<RopeFixture> {
        let path = fixture_dir().join(format!("embed_rope_{name}.bin"));
        let bytes = std::fs::read(&path).ok()?;
        if bytes.len() < HEADER_LEN || &bytes[..4] != MAGIC {
            panic!("rope fixture {path:?} has bad header");
        }
        let batch = read_u32_le(&bytes, 4) as usize;
        let heads = read_u32_le(&bytes, 8) as usize;
        let seq = read_u32_le(&bytes, 12) as usize;
        let head_dim = read_u32_le(&bytes, 16) as usize;
        let rope_dim = read_u32_le(&bytes, 20) as i32;
        let traditional = read_u32_le(&bytes, 24) != 0;
        let offset = read_i32_le(&bytes, 28);
        let scale = f32::from_bits(read_u32_le(&bytes, 32));
        let base = f32::from_bits(read_u32_le(&bytes, 36));

        let element_count = batch * heads * seq * head_dim;
        let mut cur = HEADER_LEN;
        let x = slice_f32(&bytes[cur..cur + element_count * 4]);
        cur += element_count * 4;
        let expected_y = slice_f32(&bytes[cur..cur + element_count * 4]);

        Some(RopeFixture {
            batch,
            heads,
            seq,
            head_dim,
            rope_dim,
            traditional,
            offset,
            scale,
            base,
            x,
            expected_y,
        })
    }

    fn run_fixture(name: &str) {
        let Some(fx) = load_fixture(name) else {
            eprintln!(
                "SKIP rope/{name}: fixture missing under {}. Run `python scripts/generate_rope_fixture.py` first.",
                fixture_dir().display()
            );
            return;
        };

        let x = Array::from_slice(
            &fx.x,
            &[
                fx.batch as i32,
                fx.heads as i32,
                fx.seq as i32,
                fx.head_dim as i32,
            ],
        );

        let out = rope(
            &x,
            fx.rope_dim,
            fx.traditional,
            fx.base,
            fx.scale,
            fx.offset,
        )
        .expect("mlx-rs fast::rope FFI call must succeed");
        assert_eq!(
            out.shape(),
            &[
                fx.batch as i32,
                fx.heads as i32,
                fx.seq as i32,
                fx.head_dim as i32,
            ],
            "{name}: rope output shape mismatch"
        );
        assert_eq!(
            out.dtype(),
            mlx_rs::Dtype::Float32,
            "{name}: rope output dtype unexpectedly changed"
        );
        out.eval().expect("mlx eval must succeed");

        let observed: &[f32] = out.as_slice();
        assert_eq!(
            observed.len(),
            fx.expected_y.len(),
            "{name}: rope output flat-size mismatch"
        );

        let mut mismatches = 0usize;
        for (i, (&got, &expected)) in observed.iter().zip(fx.expected_y.iter()).enumerate() {
            if got.to_bits() != expected.to_bits() {
                if mismatches < 5 {
                    eprintln!(
                        "rope/{name}[{i}]: 0x{:08x} ({got}) vs ref 0x{:08x} ({expected})",
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
    fn tiny_rope_matches_mlx() {
        run_fixture("tiny");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn qwen_partial_rope_matches_mlx() {
        run_fixture("qwen_partial");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn decode_offset_rope_matches_mlx() {
        run_fixture("decode_offset");
    }
}
