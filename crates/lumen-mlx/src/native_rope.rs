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

    /// Apply MRoPE with explicit per-token 3-axis positions.
    ///
    /// `mlx::fast::rope` only takes a scalar `offset`, i.e. it assumes token
    /// `i` sits at position `offset + i` on a single axis. That holds for text
    /// — and for MRoPE text is the degenerate case where `t == h == w` — but an
    /// image block gives its tokens a constant `t` and a grid of `h`/`w`, which
    /// no scalar offset can express. This is the unfused path for those
    /// prompts: build cos/sin from the positions directly, then apply the same
    /// GPT-NeoX split rotation the fused kernel does.
    ///
    /// Prefill only. Decode stays on `fast::rope` because positions realign
    /// (`t == h == w`) once the image block is behind us — only the starting
    /// value shifts.
    ///
    /// * `x` — `[B, H, L, head_dim]`; the first `dimensions` features rotate,
    ///   the rest pass through (partial rotary).
    /// * `positions` — `[(t, h, w); L]`, one triple per token in `x`.
    /// * `sections` — `mrope_section`, summing to `dimensions / 2`.
    pub fn mrope(
        x: &Array,
        dimensions: i32,
        positions: &[[i32; 3]],
        sections: [usize; 3],
        interleaved: bool,
        base: f32,
    ) -> Result<Array> {
        let shape = x.shape().to_vec();
        if shape.len() != 4 {
            return Err(anyhow::anyhow!(
                "mrope: expected x of rank 4 [B, H, L, D], got {shape:?}"
            ));
        }
        let (l, head_dim) = (shape[2], shape[3]);
        if positions.len() != l as usize {
            return Err(anyhow::anyhow!(
                "mrope: {} positions for {l} tokens",
                positions.len()
            ));
        }
        if dimensions <= 0 || dimensions % 2 != 0 || dimensions > head_dim {
            return Err(anyhow::anyhow!(
                "mrope: dimensions ({dimensions}) must be a positive even number ≤ head_dim ({head_dim})"
            ));
        }
        let half = (dimensions / 2) as usize;
        if sections.iter().sum::<usize>() != half {
            return Err(anyhow::anyhow!(
                "mrope: sections {sections:?} sum to {} but dimensions/2 is {half}",
                sections.iter().sum::<usize>()
            ));
        }

        // `inv[i] = 1 / base^(2i/dimensions)`, matching what MLX derives
        // internally (it stores the forward form and takes its reciprocal).
        let mut angles: Vec<f32> = Vec::with_capacity(l as usize * half);
        let mut inv: Vec<f32> = Vec::with_capacity(half);
        for i in 0..half {
            inv.push(1.0 / base.powf((2 * i) as f32 / dimensions as f32));
        }
        // Each frequency channel is driven by one spatial axis; for text-only
        // input all three axes carry the same value, so this collapses to
        // ordinary 1-D RoPE.
        let axes: Vec<usize> = (0..half)
            .map(|c| mrope_axis_of_channel(c, sections, interleaved))
            .collect();
        for pos in positions {
            for c in 0..half {
                angles.push(pos[axes[c]] as f32 * inv[c]);
            }
        }
        let angles = Array::from_slice(&angles, &[1, 1, l, half as i32]);
        let cos = mlx_rs::ops::cos(&angles).context("mrope: cos")?;
        let sin = mlx_rs::ops::sin(&angles).context("mrope: sin")?;
        let cos = cos.as_dtype(x.dtype()).context("mrope: cast cos")?;
        let sin = sin.as_dtype(x.dtype()).context("mrope: cast sin")?;

        // GPT-NeoX split form over the rotated span: the two halves of
        // `x[..., ..dimensions]` rotate into each other, and anything past
        // `dimensions` is copied through untouched.
        use mlx_rs::ops::indexing::{Ellipsis, IndexOp};
        let h = dimensions / 2;
        let x1 = x.index((Ellipsis, 0..h));
        let x2 = x.index((Ellipsis, h..dimensions));
        let out1 = mlx_rs::ops::subtract(
            &mlx_rs::ops::multiply(&x1, &cos).context("mrope: x1·cos")?,
            &mlx_rs::ops::multiply(&x2, &sin).context("mrope: x2·sin")?,
        )
        .context("mrope: x1·cos - x2·sin")?;
        let out2 = mlx_rs::ops::add(
            &mlx_rs::ops::multiply(&x2, &cos).context("mrope: x2·cos")?,
            &mlx_rs::ops::multiply(&x1, &sin).context("mrope: x1·sin")?,
        )
        .context("mrope: x2·cos + x1·sin")?;

        if dimensions == head_dim {
            mlx_rs::ops::concatenate_axis(&[&out1, &out2], -1).context("mrope: concat halves")
        } else {
            let tail = x.index((Ellipsis, dimensions..head_dim));
            mlx_rs::ops::concatenate_axis(&[&out1, &out2, &tail], -1)
                .context("mrope: concat halves + pass-through tail")
        }
    }

    /// Which spatial axis (0 = t, 1 = h, 2 = w) drives frequency channel `c`.
    ///
    /// Two layouts exist. The original Qwen2-VL one assigns contiguous chunks
    /// (`[0..s0)` → t, then h, then w). Qwen3.6 sets `mrope_interleaved`, which
    /// spreads the axes across the spectrum instead: channel `c` goes to h when
    /// `c % 3 == 1`, to w when `c % 3 == 2`, and to t otherwise — each only
    /// while that axis still has section budget left. Interleaving matters
    /// because it gives every axis both high and low frequencies rather than
    /// confining w to the flattest end.
    pub fn mrope_axis_of_channel(c: usize, sections: [usize; 3], interleaved: bool) -> usize {
        if interleaved {
            match c % 3 {
                1 if c < sections[1] * 3 => 1,
                2 if c < sections[2] * 3 => 2,
                _ => 0,
            }
        } else if c < sections[0] {
            0
        } else if c < sections[0] + sections[1] {
            1
        } else {
            2
        }
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
pub(crate) use imp::{mrope, mrope_axis_of_channel, precompute_rope_freqs, rope, rope_with_freqs};

/// Channel→axis assignment, which is pure arithmetic and the part most likely
/// to be silently wrong (a mis-assigned channel still produces plausible
/// activations).
#[cfg(test)]
mod mrope_layout_tests {
    #[cfg(feature = "mlx-native")]
    use super::imp::mrope_axis_of_channel;

    /// Qwen3.6's shipped layout: `mrope_section [11, 11, 10]` over the 32
    /// frequency channels of a 64-wide rotary span.
    #[cfg(feature = "mlx-native")]
    #[test]
    fn interleaved_layout_covers_every_channel_with_the_configured_budget() {
        let sections = [11usize, 11, 10];
        let axes: Vec<usize> = (0..32)
            .map(|c| mrope_axis_of_channel(c, sections, true))
            .collect();
        // Each axis gets exactly the count its section asked for.
        for (axis, want) in sections.iter().enumerate() {
            let got = axes.iter().filter(|&&a| a == axis).count();
            assert_eq!(
                got, *want,
                "axis {axis} claimed {got} channels, want {want}"
            );
        }
        // …and they alternate t, h, w rather than sitting in blocks — that is
        // the whole point of the interleaved variant.
        assert_eq!(&axes[..6], &[0, 1, 2, 0, 1, 2]);
    }

    #[cfg(feature = "mlx-native")]
    #[test]
    fn chunked_layout_assigns_contiguous_blocks() {
        let sections = [11usize, 11, 10];
        let axes: Vec<usize> = (0..32)
            .map(|c| mrope_axis_of_channel(c, sections, false))
            .collect();
        assert!(axes[..11].iter().all(|&a| a == 0));
        assert!(axes[11..22].iter().all(|&a| a == 1));
        assert!(axes[22..].iter().all(|&a| a == 2));
    }

    /// When an axis runs out of budget its interleave slots fall back to `t`,
    /// so every channel still gets driven by something.
    #[cfg(feature = "mlx-native")]
    #[test]
    fn exhausted_sections_fall_back_to_the_time_axis() {
        let sections = [14usize, 1, 1];
        let axes: Vec<usize> = (0..16)
            .map(|c| mrope_axis_of_channel(c, sections, true))
            .collect();
        assert_eq!(axes.iter().filter(|&&a| a == 1).count(), 1);
        assert_eq!(axes.iter().filter(|&&a| a == 2).count(), 1);
        assert_eq!(axes.iter().filter(|&&a| a == 0).count(), 14);
    }
}

/// The unfused MRoPE path has to agree with `mlx::fast::rope` whenever the
/// three axes carry the same position — that is the entire safety argument for
/// introducing it, since every text-only prompt takes that branch.
#[cfg(all(test, feature = "mlx-native"))]
mod mrope_identity_tests {
    use super::imp::{mrope, rope};
    use mlx_rs::Array;

    #[test]
    #[ignore = "MLX FFI needs a real Metal device; run outside the sandbox"]
    fn matches_fused_rope_when_all_axes_agree() {
        const B: i32 = 1;
        const H: i32 = 4;
        const L: i32 = 12;
        const HEAD_DIM: i32 = 256;
        const ROPE_DIM: i32 = 64; // partial_rotary_factor 0.25
        const BASE: f32 = 10_000_000.0;
        const OFFSET: i32 = 5;

        let n = (B * H * L * HEAD_DIM) as usize;
        // Deterministic, varied, and not symmetric — a wrong half-split or a
        // swapped sign would survive a constant input.
        let vals: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 - 48.0) / 32.0).collect();
        let x = Array::from_slice(&vals, &[B, H, L, HEAD_DIM]);

        let fused = rope(&x, ROPE_DIM, false, BASE, 1.0, OFFSET).expect("fused rope");

        // Text-only MRoPE: t == h == w == offset + i.
        let positions: Vec<[i32; 3]> = (0..L)
            .map(|i| [OFFSET + i, OFFSET + i, OFFSET + i])
            .collect();
        let manual = mrope(&x, ROPE_DIM, &positions, [11, 11, 10], true, BASE).expect("mrope");

        let a = fused
            .as_dtype(mlx_rs::Dtype::Float32)
            .expect("cast")
            .as_slice::<f32>()
            .to_vec();
        let b = manual
            .as_dtype(mlx_rs::Dtype::Float32)
            .expect("cast")
            .as_slice::<f32>()
            .to_vec();
        assert_eq!(a.len(), b.len());
        let max_abs = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        // Not bit-identical: the fused kernel and this path build the angles
        // with different op orders. Agreement to ~1e-5 is what matters — the
        // token-level gate lives in tests/qwen36_mrope_identity.rs.
        assert!(max_abs < 1e-4, "max|Δ| vs fused rope = {max_abs}");

        // Sanity: the pass-through tail past `dimensions` is untouched.
        for head in 0..(B * H) as usize {
            for t in 0..L as usize {
                let base = (head * L as usize + t) * HEAD_DIM as usize;
                for d in ROPE_DIM as usize..HEAD_DIM as usize {
                    assert_eq!(b[base + d], vals[base + d], "tail feature {d} was modified");
                }
            }
        }
    }
}

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
