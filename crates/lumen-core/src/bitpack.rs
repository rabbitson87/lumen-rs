/// Bit-packing utilities for N-bit quantization codes.
///
/// Packs codes of `bits` width into u64 words.
/// For 3-bit: 21 codes per u64 (21 × 3 = 63 bits, 1 bit unused).

/// Number of u64 words needed to pack `n` codes of `bits` width.
#[inline]
pub fn packed_words(n: usize, bits: u32) -> usize {
    let codes_per_word = 64 / bits as usize;
    (n + codes_per_word - 1) / codes_per_word
}

/// Pack a slice of u8 codes (each using `bits` bits) into u64 words.
pub fn pack_codes(codes: &[u8], bits: u32) -> Vec<u64> {
    let n = codes.len();
    let codes_per_word = 64 / bits as usize;
    let n_words = packed_words(n, bits);
    let mut packed = vec![0u64; n_words];

    for (i, &code) in codes.iter().enumerate() {
        let word_idx = i / codes_per_word;
        let bit_offset = (i % codes_per_word) as u32 * bits;
        packed[word_idx] |= (code as u64) << bit_offset;
    }

    packed
}

/// Unpack u64 words into u8 codes.
pub fn unpack_codes(packed: &[u64], n: usize, bits: u32) -> Vec<u8> {
    let codes_per_word = 64 / bits as usize;
    let mask = (1u64 << bits) - 1;
    let mut codes = Vec::with_capacity(n);

    for i in 0..n {
        let word_idx = i / codes_per_word;
        let bit_offset = (i % codes_per_word) as u32 * bits;
        let code = ((packed[word_idx] >> bit_offset) & mask) as u8;
        codes.push(code);
    }

    codes
}

/// Size in bytes of packed codes for `n` dimensions at `bits` width.
#[inline]
pub fn packed_size_bytes(n: usize, bits: u32) -> usize {
    packed_words(n, bits) * 8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip_3bit() {
        let codes: Vec<u8> = (0..256).map(|i| (i % 8) as u8).collect();
        let packed = pack_codes(&codes, 3);
        let unpacked = unpack_codes(&packed, 256, 3);
        assert_eq!(codes, unpacked);
        // 256 codes × 3 bits = 768 bits = 12 u64 words (768/64 = 12)
        assert_eq!(packed.len(), 13); // ceil(256 / 21) = 13 (21 codes per u64)
    }

    #[test]
    fn pack_unpack_roundtrip_2bit() {
        let codes: Vec<u8> = (0..128).map(|i| (i % 4) as u8).collect();
        let packed = pack_codes(&codes, 2);
        let unpacked = unpack_codes(&packed, 128, 2);
        assert_eq!(codes, unpacked);
        assert_eq!(packed.len(), 4); // ceil(128 / 32) = 4
    }

    #[test]
    fn pack_unpack_roundtrip_4bit() {
        let codes: Vec<u8> = (0..64).map(|i| (i % 16) as u8).collect();
        let packed = pack_codes(&codes, 4);
        let unpacked = unpack_codes(&packed, 64, 4);
        assert_eq!(codes, unpacked);
        assert_eq!(packed.len(), 4); // ceil(64 / 16) = 4
    }

    #[test]
    fn size_bytes() {
        // 256 codes × 3-bit = 96 bytes (vs 256 bytes unpacked)
        assert_eq!(packed_size_bytes(256, 3), 13 * 8); // 104 bytes
        // 256 codes × 2-bit = 64 bytes
        assert_eq!(packed_size_bytes(256, 2), 8 * 8); // 64 bytes
        // 512 codes × 3-bit
        assert_eq!(packed_size_bytes(512, 3), 25 * 8); // 200 bytes
    }

    #[test]
    fn compression_ratio() {
        let unpacked_bytes = 256; // 256 u8 codes
        let packed_bytes = packed_size_bytes(256, 3);
        let ratio = unpacked_bytes as f64 / packed_bytes as f64;
        assert!(
            ratio > 2.0,
            "3-bit packing should be at least 2x, got {ratio:.2}x"
        );
        eprintln!(
            "3-bit packing ratio (256 dims): {ratio:.2}x ({unpacked_bytes} → {packed_bytes} bytes)"
        );
    }
}
