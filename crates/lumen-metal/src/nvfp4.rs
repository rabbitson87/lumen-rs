//! NVFP4 (NVIDIA FP4) quantization format decoder.
//!
//! Storage (MLX convention, `mx.quantize(w, bits=4, group_size=16, mode="nvfp4")`):
//!   - `weight`: `[..., N/8] u32` — 8 nibbles per u32, least-significant nibble first
//!     (identical packing to MXFP4).
//!   - `scales`: `[..., N/16] u8` — one **E4M3** FP8 scale per 16-element group.
//!
//! Single-level scaling: MLX's NVFP4 emits exactly two tensors (no global per-tensor FP32
//! scale). Dequant is therefore `w_real = E2M1_LUT[nibble] * e4m3_to_f32(scale)`.
//!
//! Element format E2M1 (`s ee m`) and nibble packing are shared with MXFP4 — only the group
//! size (16 vs 32) and the per-group scale encoding (E4M3 vs E8M0) differ. Verified
//! bit-identical against `mx.dequantize(..., mode="nvfp4")` (mlx 0.31.2).
//!
//! This module is device-agnostic (pure Rust) and is the CPU parity reference for the fused
//! Metal dequant+matmul kernel.

use crate::mxfp4::{E2M1_LUT, NIBBLES_PER_WORD};

/// Group size for NVFP4 (fixed: one E4M3 scale per 16 elements).
pub const NVFP4_GROUP_SIZE: usize = 16;

/// Number of u32 words per 16-element group (`16 / 8`).
pub const WORDS_PER_GROUP: usize = NVFP4_GROUP_SIZE / NIBBLES_PER_WORD;

/// Decode an E4M3 FP8 byte (`s eeee mmm`, exponent bias 7) into an f32 scale.
///
/// Encoding (OCP / NVIDIA E4M3):
///   - sign  = bit 7
///   - exp   = bits 6..3 (4 bits, bias 7)
///   - mant  = bits 2..0 (3 bits)
///   - normal   (exp != 0): `(-1)^s * (1 + mant/8) * 2^(exp - 7)`
///   - subnormal (exp == 0): `(-1)^s * (mant/8) * 2^(1 - 7)` = `(-1)^s * (mant/8) * 2^-6`
///
/// Boundary behavior:
///   - `byte & 0x7F == 0x7F` → NaN per spec (E4M3 has no infinities). This encoding should
///     never appear in valid MLX-quantized scales; mapped to `0.0` here so a malformed group
///     dequantizes to zero rather than poisoning a matmul. Loaders should reject it earlier.
///   - `byte == 0x00` → `+0.0` (zero scale; whole group is zero).
///
/// MLX-produced scales are non-negative, but the sign bit is honored for generality.
#[inline]
pub fn e4m3_to_f32(byte: u8) -> f32 {
    if byte & 0x7F == 0x7F {
        return 0.0; // NaN encoding — treat as zero scale
    }
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as f32;
    let mag = if exp == 0 {
        // subnormal: 2^(1 - bias) = 2^-6
        (mant / 8.0) * 0.015625
    } else {
        (1.0 + mant / 8.0) * 2.0_f32.powi(exp - 7)
    };
    sign * mag
}

/// Extract the `i`-th nibble (0..8) from a packed u32, LSB-first ordering.
#[inline]
fn nibble_at(word: u32, i: usize) -> u8 {
    debug_assert!(i < NIBBLES_PER_WORD);
    ((word >> (i * 4)) & 0xF) as u8
}

/// Dequantize an NVFP4 buffer into `f32` output.
///
/// # Layout contract
///   - `packed.len() * NIBBLES_PER_WORD == scales.len() * NVFP4_GROUP_SIZE == out.len()`
///   - Each contiguous 16-element group shares one E4M3 scale byte.
///   - Element at output index `i` comes from nibble `i % NIBBLES_PER_WORD` of word
///     `i / NIBBLES_PER_WORD`, scaled by `scales[i / NVFP4_GROUP_SIZE]`.
///
/// Returns `Err` on length mismatch or misalignment.
pub fn dequantize_f32(packed: &[u32], scales: &[u8], out: &mut [f32]) -> Result<(), NvFp4Error> {
    let n = out.len();
    if packed.len() * NIBBLES_PER_WORD != n {
        return Err(NvFp4Error::WeightLength {
            expected_nibbles: n,
            actual_words: packed.len(),
        });
    }
    if scales.len() * NVFP4_GROUP_SIZE != n {
        return Err(NvFp4Error::ScaleLength {
            expected_groups: n / NVFP4_GROUP_SIZE,
            actual_groups: scales.len(),
        });
    }
    if !n.is_multiple_of(NVFP4_GROUP_SIZE) {
        return Err(NvFp4Error::NotAligned { n });
    }

    for (group_idx, &scale_byte) in scales.iter().enumerate() {
        let scale = e4m3_to_f32(scale_byte);
        let word_base = group_idx * WORDS_PER_GROUP;
        let out_base = group_idx * NVFP4_GROUP_SIZE;
        for w_off in 0..WORDS_PER_GROUP {
            let word = packed[word_base + w_off];
            for i in 0..NIBBLES_PER_WORD {
                let nib = nibble_at(word, i);
                out[out_base + w_off * NIBBLES_PER_WORD + i] = E2M1_LUT[nib as usize] * scale;
            }
        }
    }
    Ok(())
}

/// Errors from NVFP4 decoding.
#[derive(Debug, thiserror::Error)]
pub enum NvFp4Error {
    #[error(
        "weight buffer length mismatch: expected {expected_nibbles} nibbles but got {actual_words} u32 words"
    )]
    WeightLength {
        expected_nibbles: usize,
        actual_words: usize,
    },
    #[error(
        "scale buffer length mismatch: expected {expected_groups} groups but got {actual_groups}"
    )]
    ScaleLength {
        expected_groups: usize,
        actual_groups: usize,
    },
    #[error(
        "element count {n} is not a multiple of NVFP4 group size {}",
        NVFP4_GROUP_SIZE
    )]
    NotAligned { n: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_known_bytes() {
        // Verified against mlx-produced scale bytes for normal(0,1) f16 weights.
        assert_eq!(e4m3_to_f32(43), 0.34375); // 0b0_0101_011 → exp5 mant3 → 1.375 * 2^-2
        assert_eq!(e4m3_to_f32(45), 0.40625); // exp5 mant5 → 1.625 * 2^-2
        assert_eq!(e4m3_to_f32(47), 0.46875); // exp5 mant7 → 1.875 * 2^-2
        assert_eq!(e4m3_to_f32(48), 0.5); // 0b0_0110_000 → exp6 mant0 → 1.0 * 2^-1
        assert_eq!(e4m3_to_f32(0), 0.0); // zero scale
        assert_eq!(e4m3_to_f32(0x38), 1.0); // 0b0_0111_000 → exp7 mant0 → 2^0
        assert_eq!(e4m3_to_f32(0x7F), 0.0); // NaN encoding → 0
    }

    /// Golden fixture: `mx.quantize(mx.random.normal((2,32),seed=7).astype(f16), bits=4,
    /// group_size=16, mode="nvfp4")` and its `mx.dequantize`, mlx 0.31.2. Bit-identical.
    #[test]
    fn dequant_matches_mlx_golden() {
        #[rustfmt::skip]
        let packed: [u32; 8] = [
            0x4e75795b, 0xe6ec6617, 0xe453e356, 0x5fa5a1ca,
            0xc300d25c, 0x0ecaa168, 0x8419dc74, 0x5bbb1b88,
        ];
        let scales: [u8; 4] = [38, 44, 47, 47];
        #[rustfmt::skip]
        let expected: [f32; 64] = [
            -0.328125, 0.65625, -0.109375, 1.3125, 0.65625, 1.3125, -0.875, 0.4375,
            1.3125, 0.109375, 0.875, 0.875, -0.4375, -0.875, 0.875, -0.875,
            1.5, 1.125, 0.5625, -1.5, 0.5625, 1.125, 0.75, -1.5,
            -0.375, -0.75, 0.1875, -0.375, 1.125, -0.375, -2.25, 1.125,
            -0.9375, 1.40625, 0.46875, -1.40625, 0.0, 0.0, 0.703125, -0.9375,
            -0.0, 1.875, 0.234375, -0.46875, -0.46875, -0.9375, -1.875, 0.0,
            0.9375, 2.8125, -0.9375, -1.40625, -0.234375, 0.234375, 0.9375, -0.0,
            -0.0, -0.0, -0.703125, 0.234375, -0.703125, -0.703125, -0.703125, 1.40625,
        ];
        let mut out = [0.0f32; 64];
        dequantize_f32(&packed, &scales, &mut out).unwrap();
        for (i, (&got, &want)) in out.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, want, "mismatch at index {i}");
        }
    }

    #[test]
    fn length_mismatch_errors() {
        let packed = [0u32; 2]; // 16 nibbles → needs 1 group (16 elems)
        let scales = [0u8; 2]; // 2 groups → 32 elems, mismatch
        let mut out = [0.0f32; 16];
        assert!(matches!(
            dequantize_f32(&packed, &scales, &mut out),
            Err(NvFp4Error::ScaleLength { .. })
        ));
    }
}
