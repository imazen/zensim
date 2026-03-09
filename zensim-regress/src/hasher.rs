//! Checksum hashing for pixel data and image files.
//!
//! Pluggable via the `ChecksumHasher` trait. The default `SeaHasher`
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

    /// Hash raw file bytes directly (opaque/format-dependent hashing).
    ///
    /// Unlike [`hash_file`](Self::hash_file) which decodes to RGBA first, this
    /// hashes the encoded bytes as-is. Different encoders producing identical
    /// pixels will have different checksums.
    fn hash_file_bytes(&self, path: &Path) -> Result<String, RegressError>;
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

    fn hash_file_bytes(&self, path: &Path) -> Result<String, RegressError> {
        let data = std::fs::read(path).map_err(|e| RegressError::io(path, e))?;
        let h = seahash::hash(&data);
        Ok(format!("sea:{h:016x}"))
    }
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

    #[test]
    fn seahash_file_bytes_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");
        std::fs::write(&path, b"hello world").unwrap();

        let h1 = SeaHasher.hash_file_bytes(&path).unwrap();
        let h2 = SeaHasher.hash_file_bytes(&path).unwrap();
        assert_eq!(h1, h2);
        assert!(h1.starts_with("sea:"));
    }

    #[test]
    fn seahash_file_bytes_differs_from_pixel_hash() {
        let dir = tempfile::tempdir().unwrap();
        let pixels = vec![128u8; 4 * 4 * 4]; // 4x4 gray RGBA
        let path = dir.path().join("test.png");
        let img = image::RgbaImage::from_raw(4, 4, pixels.clone()).unwrap();
        img.save(&path).unwrap();

        let pixel_hash = SeaHasher.hash_file(&path).unwrap();
        let bytes_hash = SeaHasher.hash_file_bytes(&path).unwrap();
        assert_ne!(
            pixel_hash, bytes_hash,
            "file-bytes hash should differ from decoded-pixel hash"
        );
    }
}
