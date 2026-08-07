//! Bounded image decoding, shared by both image towers.
//!
//! `image::load_from_memory` applies no resource limits: a few hundred KB of
//! crafted PNG can expand into a multi-GB RGB buffer, which on a box already
//! holding tens of GB of weights takes the process down. Everything here exists
//! to turn that into a rejected request instead.
//!
//! Limits are sized off the **decoded** allocation rather than the file,
//! because a file size tells you nothing about its expansion factor.

/// Largest encoded payload we will even attempt to decode.
/// Override with `LUMEN_VISION_MAX_IMAGE_BYTES`.
const DEFAULT_MAX_ENCODED_BYTES: usize = 32 * 1024 * 1024;

/// Largest decoded pixel count. Override with `LUMEN_VISION_MAX_IMAGE_PIXELS`.
const DEFAULT_MAX_PIXELS: u64 = 50_000_000;

/// Hard cap on either decoded dimension, independent of the pixel budget — a
/// 1×2_000_000_000 strip is cheap in pixels but still nonsense.
const MAX_IMAGE_SIDE: u32 = 32_768;

/// Positive `usize` from the environment, or `default`.
pub(crate) fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| n > 0)
        .unwrap_or(default)
}

/// Reader over `encoded` carrying the configured decode limits.
///
/// Returned rather than reused because `into_dimensions` consumes the reader.
fn bounded_reader(encoded: &[u8]) -> Result<image::ImageReader<std::io::Cursor<&[u8]>>, String> {
    let max_bytes = env_usize("LUMEN_VISION_MAX_IMAGE_BYTES", DEFAULT_MAX_ENCODED_BYTES);
    if encoded.len() > max_bytes {
        return Err(format!(
            "image is {} bytes, over the {max_bytes}-byte limit \
             (raise LUMEN_VISION_MAX_IMAGE_BYTES to allow it)",
            encoded.len()
        ));
    }
    let max_pixels = env_usize("LUMEN_VISION_MAX_IMAGE_PIXELS", DEFAULT_MAX_PIXELS as usize) as u64;

    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_SIDE);
    limits.max_image_height = Some(MAX_IMAGE_SIDE);
    // 4 bytes/pixel covers the widest intermediate buffer (RGBA8) the decoder
    // may materialize before `to_rgb8`.
    limits.max_alloc = Some(max_pixels.saturating_mul(4));

    let mut r = image::ImageReader::new(std::io::Cursor::new(encoded))
        .with_guessed_format()
        .map_err(|e| format!("read image header: {e}"))?;
    r.limits(limits);
    Ok(r)
}

/// `(width, height)` from the header alone — no pixel decode.
pub(crate) fn dimensions_bounded(encoded: &[u8]) -> Result<(u32, u32), String> {
    let max_pixels = env_usize("LUMEN_VISION_MAX_IMAGE_PIXELS", DEFAULT_MAX_PIXELS as usize) as u64;
    let (w, h) = bounded_reader(encoded)?
        .into_dimensions()
        .map_err(|e| format!("read image dimensions: {e}"))?;
    let pixels = u64::from(w) * u64::from(h);
    if pixels > max_pixels {
        return Err(format!(
            "image is {w}×{h} = {pixels} pixels, over the {max_pixels}-pixel limit \
             (raise LUMEN_VISION_MAX_IMAGE_PIXELS to allow it)"
        ));
    }
    Ok((w, h))
}

/// Decode with explicit resource limits.
pub(crate) fn decode_bounded(encoded: &[u8]) -> Result<image::DynamicImage, String> {
    // Dimensions come from the header, so an oversized image is rejected
    // before anything is allocated for its pixels.
    dimensions_bounded(encoded)?;
    bounded_reader(encoded)?
        .decode()
        .map_err(|e| format!("decode image: {e}"))
}
