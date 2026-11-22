//! Inkling 2's image preprocessing: an image becomes a grid of 40-pixel blocks, one block per
//! image token.
//!
//! A port of `Inkling2ImageProcessor` from oMLX 0.8.2 (`omlx/patches/inkling2/processing.py`,
//! Apache-2.0). Nothing is resized: the image is padded right and bottom to a whole number of
//! blocks (with `-1` on the 0-255 scale, before normalization), normalized to about [-1, 1] and
//! cut into `[40, 40, 3]` blocks in raster order. A transparent image is composited over a
//! checkerboard first.

use anyhow::{Result, anyhow, ensure};

use crate::vision_image::{decode_bounded, dimensions_bounded};

/// Side of one block, in pixels: the processor's `size`.
pub const BLOCK: usize = 40;
/// Values per block: `40 * 40 * 3`.
pub const BLOCK_VALUES: usize = BLOCK * BLOCK * 3;
/// The most image tokens one image may take. A block is one token and the model's context holds
/// 192 KiB per token, so an image of 4,096 blocks (about 2,560 x 2,560 pixels) already costs
/// 768 MiB of cache; larger ones are refused, not resized (the reference does not resize).
pub const MAX_IMAGE_TOKENS: usize = 4096;

/// The processor's constants (`processor_config.json`): mean and std 0.5, rescale 1/255.
const MEAN: f32 = 0.5;
const STD: f32 = 0.5;
const RESCALE: f32 = (1.0f64 / 255.0) as f32;

/// An image cut into blocks.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedImage {
    pub rows: usize,
    pub cols: usize,
    /// `[rows * cols, 40, 40, 3]`, float32, blocks in raster order.
    pub blocks: Vec<f32>,
}

impl PreparedImage {
    /// Image tokens this image takes in the prompt: one per block.
    pub fn tokens(&self) -> usize {
        self.rows * self.cols
    }
}

/// Blocks per side for a dimension of `pixels`.
fn blocks_for(pixels: usize) -> usize {
    pixels.div_ceil(BLOCK)
}

/// The image tokens an encoded image takes, from its header alone (no pixel decode).
pub fn token_count(encoded: &[u8]) -> Result<usize> {
    let (w, h) = dimensions_bounded(encoded).map_err(|e| anyhow!(e))?;
    let tokens = blocks_for(w as usize) * blocks_for(h as usize);
    ensure!(
        tokens <= MAX_IMAGE_TOKENS,
        "image is {w}x{h}, {tokens} image tokens, over the {MAX_IMAGE_TOKENS}-token limit \
         (about 2,560 x 2,560 pixels)"
    );
    Ok(tokens)
}

/// Decodes `encoded` (PNG, JPEG or WebP) and cuts it into blocks.
pub fn prepare(encoded: &[u8]) -> Result<PreparedImage> {
    token_count(encoded)?;
    let decoded = decode_bounded(encoded).map_err(|e| anyhow!(e))?;
    let (w, h) = (decoded.width() as usize, decoded.height() as usize);
    let rgb: Vec<u8> = if decoded.color().has_alpha() {
        composite_over_checkerboard(&decoded.to_rgba8(), w, h)
    } else {
        decoded.to_rgb8().into_raw()
    };
    Ok(blocks_from_rgb(&rgb, w, h))
}

/// `rgba` (row-major RGBA, 8 bits) over the checkerboard the processor uses: 8-pixel squares,
/// 204 where `row / 8 + col / 8` is even and 153 where it is odd, in integers:
/// `(c * a + background * (255 - a) + 127) / 255`.
fn composite_over_checkerboard(rgba: &image::RgbaImage, w: usize, h: usize) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(w * h * 3);
    for (i, pixel) in rgba.pixels().enumerate() {
        let (row, col) = (i / w, i % w);
        let background: u32 = if (row / 8 + col / 8) % 2 == 0 {
            204
        } else {
            153
        };
        let alpha = u32::from(pixel[3]);
        for &c in &pixel.0[..3] {
            rgb.push(((u32::from(c) * alpha + background * (255 - alpha) + 127) / 255) as u8);
        }
    }
    debug_assert_eq!(rgb.len(), w * h * 3);
    rgb
}

/// Cuts an `h x w` RGB image (8 bits, row-major, channels last) into normalized blocks.
pub(crate) fn blocks_from_rgb(rgb: &[u8], w: usize, h: usize) -> PreparedImage {
    let (rows, cols) = (blocks_for(h), blocks_for(w));
    // The value a pad pixel has after normalization: -1 on the 0-255 scale.
    let pad = (-1.0f32 * RESCALE - MEAN) / STD;
    let mut blocks = vec![pad; rows * cols * BLOCK_VALUES];
    for r in 0..rows {
        for c in 0..cols {
            let block = &mut blocks[(r * cols + c) * BLOCK_VALUES..][..BLOCK_VALUES];
            for y in 0..BLOCK {
                let src_y = r * BLOCK + y;
                if src_y >= h {
                    break;
                }
                for x in 0..BLOCK {
                    let src_x = c * BLOCK + x;
                    if src_x >= w {
                        break;
                    }
                    let src = (src_y * w + src_x) * 3;
                    let dst = (y * BLOCK + x) * 3;
                    for ch in 0..3 {
                        block[dst + ch] = (f32::from(rgb[src + ch]) * RESCALE - MEAN) / STD;
                    }
                }
            }
        }
    }
    PreparedImage { rows, cols, blocks }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN: &str = include_str!("../../tests/fixtures/inkling2_vision/pixels.json");

    fn fixture_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling2_vision")
    }

    fn read_f32(path: &std::path::Path) -> Vec<f32> {
        std::fs::read(path)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    /// The reference processor's blocks for PNGs of every colour type the processor treats
    /// differently: bit for bit.
    #[test]
    fn the_blocks_are_the_reference_processors_bit_for_bit() {
        let golden: serde_json::Value = serde_json::from_str(GOLDEN).expect("pixels.json");
        let cases = golden["images"].as_array().expect("images");
        assert!(cases.len() >= 5, "{} cases", cases.len());
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let png = std::fs::read(fixture_dir().join(case["png"].as_str().unwrap())).unwrap();
            let want = read_f32(&fixture_dir().join(case["blocks"].as_str().unwrap()));
            let got = prepare(&png).unwrap_or_else(|e| panic!("{name}: {e:#}"));
            assert_eq!(
                (got.rows as u64, got.cols as u64),
                (
                    case["rows"].as_u64().unwrap(),
                    case["cols"].as_u64().unwrap()
                ),
                "{name}: grid"
            );
            assert_eq!(got.blocks.len(), want.len(), "{name}: length");
            let first_bad = got
                .blocks
                .iter()
                .zip(&want)
                .position(|(a, b)| a.to_bits() != b.to_bits());
            assert!(
                first_bad.is_none(),
                "{name}: first difference at value {first_bad:?}: {} against {}",
                got.blocks[first_bad.unwrap_or(0)],
                want[first_bad.unwrap_or(0)]
            );
            assert_eq!(
                token_count(&png).unwrap(),
                got.tokens(),
                "{name}: token count"
            );
        }
    }

    #[test]
    fn an_image_is_padded_not_resized() {
        // 41 x 1 pixels: two blocks across, one down; everything but the real pixels is pad.
        let rgb = vec![255u8; 41 * 3];
        let img = blocks_from_rgb(&rgb, 41, 1);
        assert_eq!((img.rows, img.cols, img.tokens()), (1, 2, 2));
        let white = (255.0f32 * RESCALE - MEAN) / STD;
        let pad = (-RESCALE - MEAN) / STD;
        assert!((white - 1.0).abs() < 1e-6);
        assert!(pad < -1.0 && pad > -1.01);
        assert_eq!(img.blocks[0], white); // first pixel of block 0
        assert_eq!(img.blocks[BLOCK * 3], pad); // block 0, y = 1 is past the image
        let second = &img.blocks[BLOCK_VALUES..];
        assert_eq!(second[0], white); // x = 40 is the first pixel of block 1
        assert_eq!(second[3], pad); // x = 41 is past the image
        assert_eq!(second[BLOCK * 3], pad); // y = 1 is past the image
    }

    /// CRC-32 (IEEE), bit by bit: enough to fix up one PNG chunk in a test.
    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = 0xffff_ffffu32;
        for &b in bytes {
            crc ^= u32::from(b);
            for _ in 0..8 {
                crc = if crc & 1 == 1 {
                    (crc >> 1) ^ 0xedb8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    #[test]
    fn a_header_of_a_huge_image_is_refused_before_any_pixel_is_decoded() {
        // A 1 x 1 PNG's header with its dimensions rewritten to 6000 x 6000 is not decodable, but
        // the header read alone must refuse it (a 900 x 900 block grid = 22,500 tokens).
        let mut png = image::RgbImage::new(1, 1);
        png.put_pixel(0, 0, image::Rgb([1, 2, 3]));
        let mut bytes = Vec::new();
        png.write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .unwrap();
        // IHDR: 8-byte signature, 4 length, 4 "IHDR", then width and height as u32 big-endian.
        bytes[16..20].copy_from_slice(&6000u32.to_be_bytes());
        bytes[20..24].copy_from_slice(&6000u32.to_be_bytes());
        // The decoder checks the chunk's CRC (over its type and its 13 bytes of data).
        let crc = crc32(&bytes[12..29]);
        bytes[29..33].copy_from_slice(&crc.to_be_bytes());
        let err = token_count(&bytes).unwrap_err().to_string();
        assert!(err.contains("over the 4096-token limit"), "{err}");
    }
}
