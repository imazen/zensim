//! Controlled pixel distortions for testing error classification.
//!
//! These functions produce deterministic, well-characterized pixel mutations
//! that exercise specific error categories in zensim's classification system.
//! Unlike image-processing operations (blur, sharpen), these are pure
//! bit-manipulation distortions with precisely known error profiles.
//!
//! All functions operate on flat RGBA byte slices (`&[u8]` with length
//! divisible by 4), matching the format used by
//! [`ChecksumManager::check_pixels()`](crate::manager::ChecksumManager::check_pixels).
//!
//! # Categories
//!
//! | Function | Max delta | Affected pixels | Error type |
//! |----------|-----------|-----------------|------------|
//! | [`truncate_lsb`] | 1 | ~50% | Rounding/truncation |
//! | [`expand_256`] | ~1 | varies | Bit-depth conversion |
//! | [`round_half_up`] | 1 | ~50% odd values | Rounding mode |
//! | [`premul_as_straight`] | proportional to alpha | semitransparent | Alpha handling |
//! | [`straight_as_premul`] | proportional to alpha | semitransparent | Alpha handling |
//! | [`channel_swap`] | varies | all | Channel order |
//! | [`invert`] | varies | all | Total inversion |

/// Truncate least significant bit: `floor(v / 2) * 2`.
///
/// ~50% of pixels change by 1. Produces `RoundingError` classification.
pub fn truncate_lsb(rgba: &[u8]) -> Vec<u8> {
    assert!(rgba.len() % 4 == 0, "RGBA byte length must be a multiple of 4");
    rgba.chunks_exact(4)
        .flat_map(|px| [px[0] & 0xFE, px[1] & 0xFE, px[2] & 0xFE, px[3]])
        .collect()
}

/// Wrong 8→16→8 bit-depth expansion: `val * 256 / 257` instead of `val * 257 / 257`.
///
/// Delta is proportional to value (max ~0.5 at value 128).
/// Common in incorrect bit-depth conversion code.
pub fn expand_256(rgba: &[u8]) -> Vec<u8> {
    assert!(rgba.len() % 4 == 0, "RGBA byte length must be a multiple of 4");
    rgba.chunks_exact(4)
        .flat_map(|px| {
            [
                ((px[0] as u16 * 256) / 257) as u8,
                ((px[1] as u16 * 256) / 257) as u8,
                ((px[2] as u16 * 256) / 257) as u8,
                px[3],
            ]
        })
        .collect()
}

/// Round-half-up instead of round-half-even (banker's rounding).
///
/// Adds 1 to odd values below 255. Only differs at exact .5 boundaries,
/// producing subtle systematic bias.
pub fn round_half_up(rgba: &[u8]) -> Vec<u8> {
    assert!(rgba.len() % 4 == 0, "RGBA byte length must be a multiple of 4");
    let adjust = |v: u8| -> u8 {
        if v % 2 == 1 && v < 255 {
            v.wrapping_add(1)
        } else {
            v
        }
    };
    rgba.chunks_exact(4)
        .flat_map(|px| [adjust(px[0]), adjust(px[1]), adjust(px[2]), px[3]])
        .collect()
}

/// Premultiply RGBA then interpret as straight alpha.
///
/// Darkens semitransparent pixels: `R_out = R * A / 255`.
/// Opaque and transparent pixels are unchanged.
pub fn premul_as_straight(rgba: &[u8]) -> Vec<u8> {
    assert!(rgba.len() % 4 == 0, "RGBA byte length must be a multiple of 4");
    rgba.chunks_exact(4)
        .flat_map(|px| {
            let a = px[3] as u16;
            [
                ((px[0] as u16 * a) / 255) as u8,
                ((px[1] as u16 * a) / 255) as u8,
                ((px[2] as u16 * a) / 255) as u8,
                px[3],
            ]
        })
        .collect()
}

/// Un-premultiply straight values (treat straight as premultiplied).
///
/// Brightens semitransparent pixels: `R_out = R * 255 / A`.
/// Transparent pixels (A=0) are unchanged. Can produce dramatic differences.
pub fn straight_as_premul(rgba: &[u8]) -> Vec<u8> {
    assert!(rgba.len() % 4 == 0, "RGBA byte length must be a multiple of 4");
    rgba.chunks_exact(4)
        .flat_map(|px| {
            if px[3] == 0 {
                [px[0], px[1], px[2], px[3]]
            } else {
                let a = px[3] as u16;
                [
                    ((px[0] as u16 * 255) / a).min(255) as u8,
                    ((px[1] as u16 * 255) / a).min(255) as u8,
                    ((px[2] as u16 * 255) / a).min(255) as u8,
                    px[3],
                ]
            }
        })
        .collect()
}

/// Swap R and B channels (RGB → BGR).
///
/// Produces `ChannelSwap` classification for most images.
pub fn channel_swap_rb(rgba: &[u8]) -> Vec<u8> {
    assert!(rgba.len() % 4 == 0, "RGBA byte length must be a multiple of 4");
    rgba.chunks_exact(4)
        .flat_map(|px| [px[2], px[1], px[0], px[3]])
        .collect()
}

/// Invert all color channels (not alpha).
///
/// `R_out = 255 - R`. Produces large deltas for non-midtone images.
pub fn invert(rgba: &[u8]) -> Vec<u8> {
    assert!(rgba.len() % 4 == 0, "RGBA byte length must be a multiple of 4");
    rgba.chunks_exact(4)
        .flat_map(|px| [255 - px[0], 255 - px[1], 255 - px[2], px[3]])
        .collect()
}

/// Set all pixels to a uniform delta from their original values.
///
/// `R_out = R.saturating_add(delta)` for all channels.
/// Useful for testing score sensitivity to uniform shifts.
pub fn uniform_shift(rgba: &[u8], delta: i16) -> Vec<u8> {
    assert!(rgba.len() % 4 == 0, "RGBA byte length must be a multiple of 4");
    let apply = |v: u8| -> u8 {
        (v as i16 + delta).clamp(0, 255) as u8
    };
    rgba.chunks_exact(4)
        .flat_map(|px| [apply(px[0]), apply(px[1]), apply(px[2]), px[3]])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_image() -> Vec<u8> {
        // 4 pixels with varied values
        vec![
            100, 150, 200, 255, // opaque
            50, 100, 150, 128, // semitransparent
            0, 0, 0, 255,     // black opaque
            255, 255, 255, 0,  // white transparent
        ]
    }

    #[test]
    fn truncate_lsb_max_delta_1() {
        let src = test_image();
        let dst = truncate_lsb(&src);
        for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact(4)) {
            for c in 0..3 {
                assert!(s[c].abs_diff(d[c]) <= 1);
            }
            assert_eq!(s[3], d[3], "alpha unchanged");
        }
    }

    #[test]
    fn truncate_lsb_clears_lsb() {
        let dst = truncate_lsb(&[101, 102, 103, 255]);
        assert_eq!(dst[0], 100); // 101 & 0xFE = 100
        assert_eq!(dst[1], 102); // 102 & 0xFE = 102
        assert_eq!(dst[2], 102); // 103 & 0xFE = 102
    }

    #[test]
    fn expand_256_preserves_zero_and_255() {
        let dst = expand_256(&[0, 255, 128, 255]);
        assert_eq!(dst[0], 0);
        assert_eq!(dst[1], 254); // 255 * 256 / 257 = 254
    }

    #[test]
    fn round_half_up_adds_1_to_odd() {
        let dst = round_half_up(&[101, 102, 103, 255]);
        assert_eq!(dst[0], 102); // odd → +1
        assert_eq!(dst[1], 102); // even → unchanged
        assert_eq!(dst[2], 104); // odd → +1
    }

    #[test]
    fn round_half_up_preserves_255() {
        let dst = round_half_up(&[255, 255, 255, 255]);
        assert_eq!(dst[0], 255);
    }

    #[test]
    fn premul_darkens_semitransparent() {
        let src = vec![200, 200, 200, 128]; // 50% alpha
        let dst = premul_as_straight(&src);
        assert!(dst[0] < 200, "should darken: got {}", dst[0]);
        assert_eq!(dst[3], 128, "alpha unchanged");
    }

    #[test]
    fn premul_preserves_opaque() {
        let src = vec![200, 150, 100, 255];
        let dst = premul_as_straight(&src);
        assert_eq!(dst[0], 200);
        assert_eq!(dst[1], 150);
        assert_eq!(dst[2], 100);
    }

    #[test]
    fn straight_as_premul_brightens_semitransparent() {
        let src = vec![100, 100, 100, 128]; // 50% alpha
        let dst = straight_as_premul(&src);
        assert!(dst[0] > 100, "should brighten: got {}", dst[0]);
    }

    #[test]
    fn straight_as_premul_preserves_transparent() {
        let src = vec![100, 100, 100, 0];
        let dst = straight_as_premul(&src);
        assert_eq!(dst, src);
    }

    #[test]
    fn channel_swap_rb_swaps_r_and_b() {
        let dst = channel_swap_rb(&[10, 20, 30, 255]);
        assert_eq!(dst, [30, 20, 10, 255]);
    }

    #[test]
    fn invert_complements_channels() {
        let dst = invert(&[0, 128, 255, 200]);
        assert_eq!(dst, [255, 127, 0, 200]);
    }

    #[test]
    fn uniform_shift_positive() {
        let dst = uniform_shift(&[100, 200, 250, 255], 10);
        assert_eq!(dst[0], 110);
        assert_eq!(dst[1], 210);
        assert_eq!(dst[2], 255); // saturated
        assert_eq!(dst[3], 255); // alpha unchanged
    }

    #[test]
    fn uniform_shift_negative() {
        let dst = uniform_shift(&[100, 50, 5, 255], -10);
        assert_eq!(dst[0], 90);
        assert_eq!(dst[1], 40);
        assert_eq!(dst[2], 0); // clamped
    }
}
