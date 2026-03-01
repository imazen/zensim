//! Checksum hashing for pixel data and image files.
//!
//! Pluggable via the [`ChecksumHasher`] trait. The default [`SeaHasher`]
//! is fast and non-cryptographic — fine for regression checksum comparison.

use std::path::Path;

use crate::error::RegressError;

/// Trait for computing checksum IDs from pixel data or image files.
///
/// Implementations produce prefixed string IDs like `"sea:a1b2c3d4e5f6789a"`
/// to distinguish hash algorithms. Projects can also use opaque string IDs
/// directly by skipping hashing entirely.
pub trait ChecksumHasher: Send + Sync {
    /// Hash raw RGBA pixel bytes into a checksum string.
    ///
    /// `rgba_bytes` is width × height × 4 bytes (RGBA order, row-major).
    /// The hash includes the dimensions to prevent collisions between
    /// differently-shaped images with the same pixel data.
    fn hash_pixels(&self, rgba_bytes: &[u8], width: u32, height: u32) -> String;

    /// Load an image file, decode to RGBA, and hash the pixels.
    fn hash_file(&self, path: &Path) -> Result<String, RegressError>;
}

/// SeaHash-based checksum hasher (default).
///
/// Produces IDs like `"sea:a1b2c3d4e5f6789a"` (64-bit, 16 hex chars).
/// Fast, non-cryptographic. Suitable for regression testing where
/// adversarial collision resistance is not needed.
pub struct SeaHasher;

impl SeaHasher {
    /// Hash raw bytes with SeaHash, returning the formatted ID.
    fn hash_bytes_with_dims(data: &[u8], width: u32, height: u32) -> String {
        use seahash::hash;

        // Include dimensions in the hash to avoid collisions between
        // images with the same pixel data but different layouts
        let mut buf = Vec::with_capacity(8 + data.len());
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(data);

        let h = hash(&buf);
        format!("sea:{h:016x}")
    }
}

impl ChecksumHasher for SeaHasher {
    fn hash_pixels(&self, rgba_bytes: &[u8], width: u32, height: u32) -> String {
        Self::hash_bytes_with_dims(rgba_bytes, width, height)
    }

    fn hash_file(&self, path: &Path) -> Result<String, RegressError> {
        let img = image::open(path)
            .map_err(|e| RegressError::image(path, e))?
            .to_rgba8();
        let (w, h) = img.dimensions();
        Ok(self.hash_pixels(img.as_raw(), w, h))
    }
}

/// Hash pixel data described by a [`zenpixels::PixelDescriptor`], converting
/// to RGBA8 sRGB first for format-independent hashing.
///
/// Same visual content in different pixel formats produces the same hash.
#[cfg(feature = "zenpixels")]
pub fn hash_pixels_described(
    hasher: &dyn ChecksumHasher,
    data: &[u8],
    descriptor: zenpixels::PixelDescriptor,
    width: u32,
    height: u32,
) -> String {
    let target = zenpixels::PixelDescriptor::RGBA8_SRGB;
    let converter = zenpixels::RowConverter::new(descriptor, target).unwrap_or_else(|e| {
        panic!("hash_pixels_described: cannot convert {descriptor:?} → RGBA8: {e}")
    });

    if converter.is_identity() {
        return hasher.hash_pixels(data, width, height);
    }

    let src_bpp = descriptor.bytes_per_pixel();
    let dst_bpp = target.bytes_per_pixel();
    let src_stride = width as usize * src_bpp;
    let dst_stride = width as usize * dst_bpp;
    let mut rgba8 = vec![0u8; height as usize * dst_stride];

    for y in 0..height as usize {
        let src_row = &data[y * src_stride..y * src_stride + src_stride];
        let dst_row = &mut rgba8[y * dst_stride..y * dst_stride + dst_stride];
        converter.convert_row(src_row, dst_row, width);
    }

    hasher.hash_pixels(&rgba8, width, height)
}

/// Default hasher instance (SeaHash).
pub fn default_hasher() -> SeaHasher {
    SeaHasher
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seahash_deterministic() {
        let pixels = vec![0u8; 4 * 8 * 8]; // 8x8 black RGBA
        let h1 = SeaHasher.hash_pixels(&pixels, 8, 8);
        let h2 = SeaHasher.hash_pixels(&pixels, 8, 8);
        assert_eq!(h1, h2, "SeaHash must be deterministic");
    }

    #[test]
    fn seahash_prefix() {
        let pixels = vec![0u8; 4 * 4 * 4];
        let hash = SeaHasher.hash_pixels(&pixels, 4, 4);
        assert!(hash.starts_with("sea:"), "hash should be prefixed: {hash}");
        assert_eq!(hash.len(), 4 + 16, "sea: + 16 hex chars");
    }

    #[test]
    fn seahash_dimension_matters() {
        // Same pixel data, different dimensions → different hash
        let pixels = vec![0u8; 4 * 16]; // 16 pixels worth of data
        let h1 = SeaHasher.hash_pixels(&pixels, 4, 4);
        let h2 = SeaHasher.hash_pixels(&pixels, 8, 2);
        assert_ne!(
            h1, h2,
            "different dimensions should produce different hashes"
        );
    }

    #[test]
    fn seahash_content_matters() {
        let mut p1 = vec![0u8; 4 * 4 * 4];
        let p2 = vec![255u8; 4 * 4 * 4];
        p1[0] = 1;
        let h1 = SeaHasher.hash_pixels(&p1, 4, 4);
        let h2 = SeaHasher.hash_pixels(&p2, 4, 4);
        assert_ne!(h1, h2, "different content should produce different hashes");
    }
}
