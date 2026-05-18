//! MXFP4 (OCP Microscaling FP4) quantization format decoder.
//!
//! Storage (MLX convention):
//!   - `weight`: `[..., N/8] u32` — 8 nibbles per u32, least-significant nibble first.
//!   - `scales`: `[..., N/32] u8` — one E8M0 exponent per 32-element group.
//!
//! Element format E2M1 (`s ee m`, mantissa is implicit-0 for normals, explicit-0 for subnormals):
//!   Lookup: { ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6 }
//!
//! Scale format E8M0: biased exponent, `scale = 2^(e - 127)`. `e = 255` is reserved (NaN) and
//! `e = 0` denotes subnormal scale (2^-127, effectively zero for quantized weights).
//!
//! Dequant: `w_real = E2M1_LUT[nibble] * scale`.
//!
//! This module is device-agnostic (pure Rust) and feeds the Metal fused dequant+matmul kernel
//! registered in [`crate::kernels`].

/// Group size for MXFP4 (fixed by OCP spec).
pub const MXFP4_GROUP_SIZE: usize = 32;

/// Number of nibbles packed per u32 word in MLX storage.
pub const NIBBLES_PER_WORD: usize = 8;

/// Number of u32 words per 32-element group.
pub const WORDS_PER_GROUP: usize = MXFP4_GROUP_SIZE / NIBBLES_PER_WORD;

/// E2M1 value table indexed by raw nibble `s ee m` (sign bit at MSB).
///
/// Derivation:
///   bits 000x → subnormal: sign * mantissa * 2^(1-bias) with bias=1 → value ∈ {0, 0.5}
///   bits 0ee m → normal:   sign * (1.m) * 2^(ee - 1)                   → {1, 1.5, 2, 3, 4, 6}
#[rustfmt::skip]
pub const E2M1_LUT: [f32; 16] = [
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Decode an E8M0 byte into an f32 scale.
///
/// Reconstructs `2^(byte - 127)` via direct f32 bit manipulation (bias=127 aligns exactly with
/// IEEE 754 single-precision).
///
/// Boundary behavior:
///   - `byte == 0xFF` → NaN per OCP spec, mapped to `0.0` here. This encoding should never
///     appear in valid MLX-quantized tensors; callers loading fixtures should treat it as a
///     load error before reaching this function.
///   - `byte == 0x00` → produces `+0.0` (zero scale). The OCP spec nominally assigns `2^-127`
///     to this encoding, but f32 cannot represent it without subnormal tricks that buy nothing
///     downstream since an entire 32-element group dequantizes to zero anyway. Treat byte=0 as
///     "group is zero".
#[inline]
pub fn e8m0_to_f32(byte: u8) -> f32 {
    if byte == 0xFF {
        return 0.0;
    }
    let biased_exp = byte as u32; // matches f32 bias=127
    f32::from_bits(biased_exp << 23)
}

/// Extract the `i`-th nibble (0..8) from a packed u32, LSB-first ordering.
#[inline]
fn nibble_at(word: u32, i: usize) -> u8 {
    debug_assert!(i < NIBBLES_PER_WORD);
    ((word >> (i * 4)) & 0xF) as u8
}

/// Dequantize an MXFP4 buffer into `f32` output.
///
/// # Layout contract
///   - `packed.len() * NIBBLES_PER_WORD == scales.len() * MXFP4_GROUP_SIZE == out.len()`
///   - Each contiguous 32-element group shares one scale byte.
///   - Element at output index `i` comes from nibble `i % NIBBLES_PER_WORD` of word
///     `i / NIBBLES_PER_WORD`, scaled by `scales[i / MXFP4_GROUP_SIZE]`.
///
/// Returns `Err` on length mismatch.
pub fn dequantize_f32(packed: &[u32], scales: &[u8], out: &mut [f32]) -> Result<(), MxFp4Error> {
    let n = out.len();
    if packed.len() * NIBBLES_PER_WORD != n {
        return Err(MxFp4Error::WeightLength {
            expected_nibbles: n,
            actual_words: packed.len(),
        });
    }
    if scales.len() * MXFP4_GROUP_SIZE != n {
        return Err(MxFp4Error::ScaleLength {
            expected_groups: n / MXFP4_GROUP_SIZE,
            actual_groups: scales.len(),
        });
    }
    if !n.is_multiple_of(MXFP4_GROUP_SIZE) {
        return Err(MxFp4Error::NotAligned { n });
    }

    for (group_idx, &scale_byte) in scales.iter().enumerate() {
        let scale = e8m0_to_f32(scale_byte);
        let word_base = group_idx * WORDS_PER_GROUP;
        let out_base = group_idx * MXFP4_GROUP_SIZE;
        for w_off in 0..WORDS_PER_GROUP {
            let word = packed[word_base + w_off];
            for i in 0..NIBBLES_PER_WORD {
                let nib = nibble_at(word, i);
                let v = E2M1_LUT[nib as usize] * scale;
                out[out_base + w_off * NIBBLES_PER_WORD + i] = v;
            }
        }
    }
    Ok(())
}

/// Errors from MXFP4 decoding.
#[derive(Debug, thiserror::Error)]
pub enum MxFp4Error {
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
        "element count {n} is not a multiple of MXFP4 group size {}",
        MXFP4_GROUP_SIZE
    )]
    NotAligned { n: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_lut_matches_ocp_spec() {
        // Positive values from OCP MXFP4 spec table
        assert_eq!(E2M1_LUT[0b0000], 0.0);
        assert_eq!(E2M1_LUT[0b0001], 0.5);
        assert_eq!(E2M1_LUT[0b0010], 1.0);
        assert_eq!(E2M1_LUT[0b0011], 1.5);
        assert_eq!(E2M1_LUT[0b0100], 2.0);
        assert_eq!(E2M1_LUT[0b0101], 3.0);
        assert_eq!(E2M1_LUT[0b0110], 4.0);
        assert_eq!(E2M1_LUT[0b0111], 6.0);
        // Negatives are sign-flipped
        for i in 0..8 {
            assert_eq!(E2M1_LUT[i + 8], -E2M1_LUT[i]);
        }
    }

    #[test]
    fn e8m0_unity_scale_at_127() {
        assert_eq!(e8m0_to_f32(127), 1.0);
    }

    #[test]
    fn e8m0_power_of_two_scales() {
        assert_eq!(e8m0_to_f32(128), 2.0);
        assert_eq!(e8m0_to_f32(129), 4.0);
        assert_eq!(e8m0_to_f32(126), 0.5);
        assert_eq!(e8m0_to_f32(125), 0.25);
    }

    #[test]
    fn e8m0_extremes_handled() {
        assert_eq!(e8m0_to_f32(0xFF), 0.0); // spec NaN → mapped to 0
        assert_eq!(e8m0_to_f32(0x00), 0.0); // zero scale — entire group dequants to 0
    }

    #[test]
    fn dequantize_rejects_length_mismatch() {
        let packed = vec![0u32; 4]; // 32 nibbles → one group
        let scales = vec![127u8; 2]; // two groups — wrong
        let mut out = vec![0.0f32; 32];
        let err = dequantize_f32(&packed, &scales, &mut out).unwrap_err();
        assert!(matches!(err, MxFp4Error::ScaleLength { .. }));
    }

    #[test]
    fn dequantize_all_zeros_gives_zero() {
        let packed = vec![0u32; 4];
        let scales = vec![127u8; 1]; // scale = 1.0
        let mut out = vec![99.0f32; 32];
        dequantize_f32(&packed, &scales, &mut out).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn dequantize_single_group_known_pattern() {
        // One group of 32 elements. Build a packed pattern where nibble i == (i mod 16).
        let mut packed = vec![0u32; WORDS_PER_GROUP];
        for i in 0..MXFP4_GROUP_SIZE {
            let word_idx = i / NIBBLES_PER_WORD;
            let nib_idx = i % NIBBLES_PER_WORD;
            let nib = (i % 16) as u32;
            packed[word_idx] |= nib << (nib_idx * 4);
        }
        let scales = vec![127u8; 1]; // scale = 1.0, dequant value == LUT[nib]
        let mut out = vec![0.0f32; MXFP4_GROUP_SIZE];
        dequantize_f32(&packed, &scales, &mut out).unwrap();
        for i in 0..MXFP4_GROUP_SIZE {
            assert_eq!(
                out[i],
                E2M1_LUT[i % 16],
                "mismatch at index {i} (nibble {})",
                i % 16
            );
        }
    }

    #[test]
    fn dequantize_scale_applies_per_group() {
        // Two groups, all nibbles = 0b0010 (LUT value = 1.0).
        // Group 0 scale = 2^0 = 1.0, group 1 scale = 2^2 = 4.0.
        let word = 0x2222_2222u32; // eight copies of nibble 0b0010
        let packed = vec![word, word, word, word, word, word, word, word];
        let scales = vec![127u8, 129u8];
        let mut out = vec![0.0f32; 64];
        dequantize_f32(&packed, &scales, &mut out).unwrap();
        for i in 0..32 {
            assert_eq!(out[i], 1.0);
        }
        for i in 32..64 {
            assert_eq!(out[i], 4.0);
        }
    }

    #[test]
    fn dequantize_negative_nibbles() {
        // Nibble 0b1010 → LUT = -1.0. scale = 1.0. All elements = -1.0.
        let word = 0xAAAA_AAAAu32;
        let packed = vec![word; WORDS_PER_GROUP];
        let scales = vec![127u8; 1];
        let mut out = vec![0.0f32; MXFP4_GROUP_SIZE];
        dequantize_f32(&packed, &scales, &mut out).unwrap();
        assert!(out.iter().all(|&v| v == -1.0));
    }
}
