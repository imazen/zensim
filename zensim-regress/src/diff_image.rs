//! Diff image generation and visual montage composition.
//!
//! Creates amplified difference images and side-by-side comparison montages
//! for visual regression analysis.

use image::{Rgba, RgbaImage, imageops};

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

/// Generate an amplified diff image from raw RGBA byte slices.
///
/// Convenience wrapper around [`generate_diff_image`] for callers working
/// with `&[u8]` pixel buffers instead of `RgbaImage`.
///
/// # Panics
///
/// Panics if either slice length doesn't match `width * height * 4`,
/// or if expected and actual have different dimensions.
pub fn generate_diff_image_raw(
    expected: &[u8],
    actual: &[u8],
    width: u32,
    height: u32,
    amplification: u8,
) -> RgbaImage {
    let exp_img = RgbaImage::from_raw(width, height, expected.to_vec())
        .expect("expected: invalid dimensions for pixel data");
    let act_img = RgbaImage::from_raw(width, height, actual.to_vec())
        .expect("actual: invalid dimensions for pixel data");
    generate_diff_image(&exp_img, &act_img, amplification)
}

/// Create a 3-panel comparison montage from raw RGBA byte slices.
///
/// Convenience wrapper around [`create_comparison_montage`] for callers
/// working with `&[u8]` pixel buffers instead of `RgbaImage`.
///
/// # Panics
///
/// Panics if either slice length doesn't match `width * height * 4`.
pub fn create_comparison_montage_raw(
    expected: &[u8],
    actual: &[u8],
    width: u32,
    height: u32,
    amplification: u8,
    gap: u32,
) -> RgbaImage {
    let exp_img = RgbaImage::from_raw(width, height, expected.to_vec())
        .expect("expected: invalid dimensions for pixel data");
    let act_img = RgbaImage::from_raw(width, height, actual.to_vec())
        .expect("actual: invalid dimensions for pixel data");
    create_comparison_montage(&exp_img, &act_img, amplification, gap)
}

/// Stats for one region of the diff (used by spatial analysis).
#[derive(Debug, Clone)]
pub struct RegionStats {
    /// Grid column (0-based, left to right).
    pub col: u32,
    /// Grid row (0-based, top to bottom).
    pub row: u32,
    /// Fraction of pixels in this region that differ (any channel).
    pub pixels_differing: f64,
    /// Average per-channel delta across differing pixels.
    pub avg_delta: f64,
    /// Maximum per-channel delta in this region.
    pub max_delta: u8,
    /// Variance of the expected image in this region (content indicator).
    pub expected_variance: f64,
    /// Variance of the actual image in this region.
    pub actual_variance: f64,
}

/// Spatial breakdown of differences across a grid of regions.
///
/// Divides the image into `cols × rows` regions and reports per-region stats.
/// This tells you where differences are concentrated, whether content is
/// present in both images, and whether a feature is missing vs. rendered
/// differently.
#[derive(Debug, Clone)]
pub struct SpatialAnalysis {
    /// Number of columns in the grid.
    pub cols: u32,
    /// Number of rows in the grid.
    pub rows: u32,
    /// Per-region stats, in row-major order.
    pub regions: Vec<RegionStats>,
}

impl std::fmt::Display for SpatialAnalysis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let labels = ["TL", "TR", "BL", "BR", "L", "R", "T", "B", "C"];
        let max_delta_idx = self
            .regions
            .iter()
            .enumerate()
            .max_by_key(|(_, r)| r.max_delta)
            .map(|(i, _)| i);

        writeln!(f, "Spatial diff ({}x{} grid):", self.cols, self.rows)?;

        for (i, r) in self.regions.iter().enumerate() {
            let label = if self.cols == 2 && self.rows == 2 {
                labels[i].to_string()
            } else {
                format!("({},{})", r.col, r.row)
            };

            let arrow = if max_delta_idx == Some(i) && r.max_delta > 0 {
                " <-- concentrated"
            } else {
                ""
            };

            if r.pixels_differing > 0.001 {
                writeln!(
                    f,
                    "  {label}: {:.1}% differ, avg delta {:.0}, max {}{arrow}",
                    r.pixels_differing * 100.0,
                    r.avg_delta,
                    r.max_delta,
                )?;

                // Structural presence check
                let exp_has_content = r.expected_variance > 10.0;
                let act_has_content = r.actual_variance > 10.0;
                if !exp_has_content && act_has_content {
                    writeln!(f, "         expected: uniform, actual: has content (added feature?)")?;
                } else if exp_has_content && !act_has_content {
                    writeln!(f, "         expected: has content, actual: uniform (missing feature?)")?;
                } else if exp_has_content && act_has_content && r.avg_delta > 10.0 {
                    writeln!(f, "         both have content -- different rendering")?;
                }
            } else {
                writeln!(f, "  {label}: identical")?;
            }
        }
        Ok(())
    }
}

/// Compute spatial diff analysis over a grid of regions.
///
/// Divides both images into `cols × rows` regions and computes per-region
/// difference statistics. Use 2×2 for quadrant analysis, 3×3 or 4×4 for
/// finer spatial resolution.
///
/// # Panics
///
/// Panics if images have different dimensions or if cols/rows is 0.
pub fn spatial_analysis(
    expected: &[u8],
    actual: &[u8],
    width: u32,
    height: u32,
    cols: u32,
    rows: u32,
) -> SpatialAnalysis {
    assert!(cols > 0 && rows > 0, "cols and rows must be > 0");
    let npx = (width * height * 4) as usize;
    assert!(expected.len() >= npx && actual.len() >= npx);

    let mut regions = Vec::with_capacity((cols * rows) as usize);

    for gy in 0..rows {
        for gx in 0..cols {
            let x0 = (gx * width) / cols;
            let x1 = ((gx + 1) * width) / cols;
            let y0 = (gy * height) / rows;
            let y1 = ((gy + 1) * height) / rows;

            let mut diff_count = 0u64;
            let mut delta_sum = 0u64;
            let mut max_d: u8 = 0;
            let mut pixel_count = 0u64;
            let mut exp_sum = 0u64;
            let mut exp_sq_sum = 0u64;
            let mut act_sum = 0u64;
            let mut act_sq_sum = 0u64;

            for y in y0..y1 {
                for x in x0..x1 {
                    let off = ((y * width + x) * 4) as usize;
                    pixel_count += 1;

                    let mut any_diff = false;
                    for c in 0..3 {
                        let e = expected[off + c] as i16;
                        let a = actual[off + c] as i16;
                        let d = (e - a).unsigned_abs() as u8;
                        if d > 0 {
                            any_diff = true;
                            delta_sum += d as u64;
                            max_d = max_d.max(d);
                        }
                        exp_sum += e as u64;
                        exp_sq_sum += (e as u64) * (e as u64);
                        act_sum += a as u64;
                        act_sq_sum += (a as u64) * (a as u64);
                    }
                    if any_diff {
                        diff_count += 1;
                    }
                }
            }

            let n = pixel_count.max(1) as f64;
            let n3 = (pixel_count * 3).max(1) as f64;
            let exp_mean = exp_sum as f64 / n3;
            let act_mean = act_sum as f64 / n3;
            let exp_var = (exp_sq_sum as f64 / n3) - exp_mean * exp_mean;
            let act_var = (act_sq_sum as f64 / n3) - act_mean * act_mean;

            let avg_delta = if diff_count > 0 {
                delta_sum as f64 / (diff_count as f64 * 3.0)
            } else {
                0.0
            };

            regions.push(RegionStats {
                col: gx,
                row: gy,
                pixels_differing: diff_count as f64 / n,
                avg_delta,
                max_delta: max_d,
                expected_variance: exp_var.max(0.0),
                actual_variance: act_var.max(0.0),
            });
        }
    }

    SpatialAnalysis {
        cols,
        rows,
        regions,
    }
}

/// Create a comparison montage with stats text burned into a canvas below.
///
/// Returns a 3-panel montage (expected | actual | amplified diff) with
/// a text strip below showing the provided annotation lines.
///
/// # Panics
///
/// Panics if expected and actual have different dimensions.
pub fn create_annotated_montage(
    expected: &RgbaImage,
    actual: &RgbaImage,
    amplification: u8,
    gap: u32,
    annotation: &str,
) -> RgbaImage {
    use crate::font;

    let montage = create_comparison_montage(expected, actual, amplification, gap);

    if annotation.is_empty() {
        return montage;
    }

    let fg = [220, 220, 220, 255]; // light gray text
    let bg = [24, 24, 24, 255]; // dark background
    let (text_buf, text_w, text_h) = font::render_text(annotation, fg, bg);

    if text_w == 0 || text_h == 0 {
        return montage;
    }

    let padding = 4u32;
    let canvas_h = text_h + padding * 2;
    let total_w = montage.width().max(text_w + padding * 2);
    let total_h = montage.height() + canvas_h;

    let mut output = RgbaImage::from_pixel(total_w, total_h, Rgba([24, 24, 24, 255]));

    // Copy montage at top
    imageops::overlay(&mut output, &montage, 0, 0);

    // Stamp text into canvas area below
    let text_img = RgbaImage::from_raw(text_w, text_h, text_buf)
        .expect("text render dimensions mismatch");
    imageops::overlay(
        &mut output,
        &text_img,
        padding as i64,
        (montage.height() + padding) as i64,
    );

    output
}

/// Create an annotated montage from raw RGBA byte slices.
///
/// Convenience wrapper for callers working with `&[u8]` pixel buffers.
pub fn create_annotated_montage_raw(
    expected: &[u8],
    actual: &[u8],
    width: u32,
    height: u32,
    amplification: u8,
    gap: u32,
    annotation: &str,
) -> RgbaImage {
    let exp_img = RgbaImage::from_raw(width, height, expected.to_vec())
        .expect("expected: invalid dimensions for pixel data");
    let act_img = RgbaImage::from_raw(width, height, actual.to_vec())
        .expect("actual: invalid dimensions for pixel data");
    create_annotated_montage(&exp_img, &act_img, amplification, gap, annotation)
}

/// Format a [`RegressionReport`](crate::testing::RegressionReport) as annotation text.
///
/// Produces 2-3 lines suitable for [`create_annotated_montage`],
/// e.g. `score:87.2  delta:[12,8,3]  34.2% differ  category:perceptual`.
pub fn format_annotation(report: &crate::testing::RegressionReport) -> String {
    let [dr, dg, db] = report.max_channel_delta();
    let pct = if report.pixel_count() > 0 {
        report.pixels_differing() as f64 / report.pixel_count() as f64 * 100.0
    } else {
        0.0
    };
    format!(
        "score:{:.1}  delta:[{},{},{}]  {:.1}% differ  {:?}",
        report.score(),
        dr,
        dg,
        db,
        pct,
        report.category(),
    )
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

    #[test]
    fn diff_raw_matches_typed() {
        let exp = RgbaImage::from_pixel(4, 4, Rgba([100, 100, 100, 255]));
        let act = RgbaImage::from_pixel(4, 4, Rgba([105, 100, 95, 255]));

        let typed = generate_diff_image(&exp, &act, 10);
        let raw = generate_diff_image_raw(exp.as_raw(), act.as_raw(), 4, 4, 10);
        assert_eq!(typed, raw);
    }

    #[test]
    fn comparison_montage_raw_matches_typed() {
        let exp = RgbaImage::from_pixel(4, 4, Rgba([100, 100, 100, 255]));
        let act = RgbaImage::from_pixel(4, 4, Rgba([105, 100, 95, 255]));

        let typed = create_comparison_montage(&exp, &act, 10, 2);
        let raw = create_comparison_montage_raw(exp.as_raw(), act.as_raw(), 4, 4, 10, 2);
        assert_eq!(typed, raw);
    }

    #[test]
    fn annotated_montage_adds_text_strip() {
        let exp = RgbaImage::from_pixel(32, 32, Rgba([100, 100, 100, 255]));
        let act = RgbaImage::from_pixel(32, 32, Rgba([110, 100, 90, 255]));
        let montage = create_annotated_montage(&exp, &act, 10, 2, "score:87.2 delta:[10,0,10]");

        // Width should be at least the 3-panel montage width
        assert!(montage.width() >= 32 * 3 + 4);
        // Height should be montage + text strip
        assert!(montage.height() > 32);
    }

    #[test]
    fn annotated_montage_empty_text_no_strip() {
        let exp = RgbaImage::from_pixel(16, 16, Rgba([100; 4]));
        let act = RgbaImage::from_pixel(16, 16, Rgba([100; 4]));
        let montage = create_annotated_montage(&exp, &act, 10, 2, "");
        // Should be same as non-annotated
        let plain = create_comparison_montage(&exp, &act, 10, 2);
        assert_eq!(montage.dimensions(), plain.dimensions());
    }

    #[test]
    fn spatial_identical_images() {
        let w = 64u32;
        let h = 64;
        let px: Vec<u8> = (0..w * h).flat_map(|i| [(i % 255) as u8, 128, 64, 255]).collect();
        let analysis = spatial_analysis(&px, &px, w, h, 2, 2);
        assert_eq!(analysis.regions.len(), 4);
        for r in &analysis.regions {
            assert_eq!(r.pixels_differing, 0.0);
            assert_eq!(r.max_delta, 0);
        }
    }

    #[test]
    fn spatial_concentrated_diff() {
        let w = 64u32;
        let h = 64;
        let exp: Vec<u8> = vec![128; (w * h * 4) as usize];
        let mut act = exp.clone();
        // Modify only bottom-right quadrant (x>=32, y>=32)
        for y in 32..64 {
            for x in 32..64 {
                let off = ((y * w + x) * 4) as usize;
                act[off] = 0; // R channel to 0
            }
        }
        let analysis = spatial_analysis(&exp, &act, w, h, 2, 2);
        // TL, TR, BL should be 0% different
        assert_eq!(analysis.regions[0].max_delta, 0); // TL
        assert_eq!(analysis.regions[1].max_delta, 0); // TR
        assert_eq!(analysis.regions[2].max_delta, 0); // BL
        // BR should have 100% different with delta 128
        assert!(analysis.regions[3].pixels_differing > 0.99);
        assert_eq!(analysis.regions[3].max_delta, 128);
    }

    #[test]
    fn spatial_display_format() {
        let w = 64u32;
        let h = 64;
        let exp: Vec<u8> = vec![128; (w * h * 4) as usize];
        let mut act = exp.clone();
        // Small diff in top-left only
        act[0] = 138;
        let analysis = spatial_analysis(&exp, &act, w, h, 2, 2);
        let text = format!("{analysis}");
        assert!(text.contains("Spatial diff"));
        assert!(text.contains("TL"));
    }
}
