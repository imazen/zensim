//! Diff image generation and visual montage composition.
//!
//! Creates amplified difference images and side-by-side comparison montages
//! for visual regression analysis.

use image::{RgbaImage, Rgba, imageops};

/// Generate an amplified diff image showing per-channel absolute differences.
///
/// Unchanged pixels are dark gray. Differences glow in the color of the
/// channels that differ, amplified by the given factor for visibility.
///
/// # Panics
///
/// Panics if expected and actual have different dimensions.
pub fn generate_diff_image(
    expected: &RgbaImage,
    actual: &RgbaImage,
    amplification: u8,
) -> RgbaImage {
    let (w, h) = expected.dimensions();
    assert_eq!(
        (w, h),
        actual.dimensions(),
        "expected {}x{} but actual is {}x{}",
        w,
        h,
        actual.width(),
        actual.height(),
    );

    let amp = amplification.max(1) as i16;
    let mut diff = RgbaImage::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let e = expected.get_pixel(x, y);
            let a = actual.get_pixel(x, y);

            let dr = ((e[0] as i16 - a[0] as i16).abs() * amp).min(255) as u8;
            let dg = ((e[1] as i16 - a[1] as i16).abs() * amp).min(255) as u8;
            let db = ((e[2] as i16 - a[2] as i16).abs() * amp).min(255) as u8;

            if dr == 0 && dg == 0 && db == 0 {
                diff.put_pixel(x, y, Rgba([24, 24, 24, 255]));
            } else {
                diff.put_pixel(x, y, Rgba([dr, dg, db, 255]));
            }
        }
    }

    diff
}

/// Create a horizontal montage from multiple image panels.
///
/// Panels are placed left-to-right with a gap between them.
/// The montage height is the maximum panel height; shorter panels
/// are top-aligned against a dark background.
pub fn create_montage(panels: &[&RgbaImage], gap: u32) -> RgbaImage {
    if panels.is_empty() {
        return RgbaImage::new(1, 1);
    }

    let max_h = panels.iter().map(|p| p.height()).max().unwrap_or(1);
    let total_w: u32 = panels.iter().map(|p| p.width()).sum::<u32>()
        + gap * (panels.len() as u32).saturating_sub(1);

    let mut montage = RgbaImage::from_pixel(total_w, max_h, Rgba([32, 32, 32, 255]));

    let mut x_offset: i64 = 0;
    for panel in panels {
        imageops::overlay(&mut montage, *panel, x_offset, 0);
        x_offset += panel.width() as i64 + gap as i64;
    }

    montage
}

/// Create a 3-panel comparison montage: Expected | Actual | Diff.
///
/// The diff panel uses amplified absolute differences (default 10x).
/// Returns the composite image ready for display.
///
/// # Panics
///
/// Panics if expected and actual have different dimensions.
pub fn create_comparison_montage(
    expected: &RgbaImage,
    actual: &RgbaImage,
    amplification: u8,
    gap: u32,
) -> RgbaImage {
    let diff = generate_diff_image(expected, actual, amplification);
    create_montage(&[expected, actual, &diff], gap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_identical_images() {
        let img = RgbaImage::from_fn(8, 8, |x, y| {
            Rgba([(x * 32) as u8, (y * 32) as u8, 128, 255])
        });
        let diff = generate_diff_image(&img, &img, 10);
        // All pixels should be dark gray (no difference)
        for pixel in diff.pixels() {
            assert_eq!(*pixel, Rgba([24, 24, 24, 255]));
        }
    }

    #[test]
    fn diff_amplification() {
        let a = RgbaImage::from_pixel(4, 4, Rgba([100, 100, 100, 255]));
        let b = RgbaImage::from_pixel(4, 4, Rgba([101, 100, 98, 255]));

        let diff_1x = generate_diff_image(&a, &b, 1);
        let diff_10x = generate_diff_image(&a, &b, 10);

        let p1 = diff_1x.get_pixel(0, 0);
        let p10 = diff_10x.get_pixel(0, 0);

        assert_eq!(p1[0], 1); // |100-101| * 1
        assert_eq!(p1[2], 2); // |100-98| * 1
        assert_eq!(p10[0], 10); // |100-101| * 10
        assert_eq!(p10[2], 20); // |100-98| * 10
    }

    #[test]
    fn diff_clamps_to_255() {
        let a = RgbaImage::from_pixel(2, 2, Rgba([0, 0, 0, 255]));
        let b = RgbaImage::from_pixel(2, 2, Rgba([200, 200, 200, 255]));
        let diff = generate_diff_image(&a, &b, 10);
        let p = diff.get_pixel(0, 0);
        assert_eq!(p[0], 255); // 200*10 clamped to 255
    }

    #[test]
    fn montage_dimensions() {
        let a = RgbaImage::new(10, 20);
        let b = RgbaImage::new(10, 20);
        let c = RgbaImage::new(10, 20);
        let montage = create_montage(&[&a, &b, &c], 2);
        assert_eq!(montage.width(), 34); // 10+2+10+2+10
        assert_eq!(montage.height(), 20);
    }

    #[test]
    fn montage_different_heights() {
        let a = RgbaImage::new(10, 30);
        let b = RgbaImage::new(10, 10);
        let montage = create_montage(&[&a, &b], 4);
        assert_eq!(montage.width(), 24); // 10+4+10
        assert_eq!(montage.height(), 30); // max height
    }

    #[test]
    fn comparison_montage_works() {
        let expected = RgbaImage::from_pixel(8, 8, Rgba([100, 100, 100, 255]));
        let actual = RgbaImage::from_pixel(8, 8, Rgba([101, 99, 100, 255]));
        let montage = create_comparison_montage(&expected, &actual, 10, 2);
        assert_eq!(montage.width(), 28); // 8+2+8+2+8
        assert_eq!(montage.height(), 8);
    }
}
