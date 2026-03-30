//! Diff image generation and visual montage composition.
//!
//! Creates amplified difference images and side-by-side comparison montages
//! for visual regression analysis.

use image::{Rgba, RgbaImage, imageops};

/// Default minimum panel dimension (in pixels) for montage upscaling.
///
/// When either dimension of the input images is smaller than this, each pixel
/// is replicated N×N (nearest-neighbor integer scaling) so the montage panels
/// are large enough to inspect visually.
pub const DEFAULT_MIN_PANEL_SIZE: u32 = 256;

/// Upscale a small image using pixelated (nearest-neighbor) integer scaling.
///
/// Finds the smallest integer `N ≥ 1` such that both `width * N >= min_dim`
/// and `height * N >= min_dim`, then replicates each pixel into an N×N block.
/// If both dimensions already meet the threshold, the image is returned as-is.
///
/// This preserves sharp pixel boundaries — no interpolation, no blurring.
pub fn pixelate_upscale(img: &RgbaImage, min_dim: u32) -> RgbaImage {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 || (w >= min_dim && h >= min_dim) {
        return img.clone();
    }

    // Smallest integer N such that w*N >= min_dim AND h*N >= min_dim
    let n_w = min_dim.div_ceil(w);
    let n_h = min_dim.div_ceil(h);
    let n = n_w.max(n_h);
    if n <= 1 {
        return img.clone();
    }

    let new_w = w * n;
    let new_h = h * n;
    let mut out = RgbaImage::new(new_w, new_h);

    for y in 0..h {
        for x in 0..w {
            let px = *img.get_pixel(x, y);
            let bx = x * n;
            let by = y * n;
            for dy in 0..n {
                for dx in 0..n {
                    out.put_pixel(bx + dx, by + dy, px);
                }
            }
        }
    }

    out
}

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

            // Compare premultiplied values so transparent pixels diff to zero
            // regardless of their RGB garbage values.
            let ea = e[3] as i32;
            let aa = a[3] as i32;
            let amp32 = amp as i32;
            let dr =
                ((e[0] as i32 * ea / 255 - a[0] as i32 * aa / 255).abs() * amp32).min(255) as u8;
            let dg =
                ((e[1] as i32 * ea / 255 - a[1] as i32 * aa / 255).abs() * amp32).min(255) as u8;
            let db =
                ((e[2] as i32 * ea / 255 - a[2] as i32 * aa / 255).abs() * amp32).min(255) as u8;

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
/// **Prefer [`create_annotated_montage`]** — it produces a labeled 2×2 grid
/// with structural diff, colored constraint text, and a 9-cell spatial
/// heatmap. This function produces a bare strip with no labels or stats.
///
/// # Panics
///
/// Panics if expected and actual have different dimensions.
#[deprecated(
    since = "0.2.2",
    note = "use create_annotated_montage for labeled panels, \
    structural diff, and spatial heatmap"
)]
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
/// Convenience wrapper around [`create_comparison_montage`].
///
/// **Prefer [`create_annotated_montage_raw`]** for labeled output with
/// structural diff and spatial heatmap.
#[deprecated(
    since = "0.2.2",
    note = "use create_annotated_montage_raw for labeled panels, \
    structural diff, and spatial heatmap"
)]
#[allow(deprecated)]
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

// ─── Structural diff (high-pass residual) ───────────────────────────────

/// Generate a structural diff image showing where structure exists in one
/// image but not the other.
///
/// Uses high-pass residuals (pixel minus box-blurred local mean) to extract
/// structural features, then diffs the residual magnitudes. This is the same
/// approach zensim uses internally for edge artifact/detail features, but
/// applied here as a standalone visualization with no zensim dependency.
///
/// Output coloring:
/// - **Cyan**: structure present in expected but absent in actual (missing feature)
/// - **Orange**: structure present in actual but absent in expected (added feature)
/// - **Dark gray**: no structural difference
///
/// `blur_radius` controls the high-pass cutoff (2–3 is good for watermarks
/// and overlays, 1 for fine texture, 5+ for large shapes).
///
/// # Panics
///
/// Panics if expected and actual have different dimensions.
pub fn generate_structural_diff(
    expected: &RgbaImage,
    actual: &RgbaImage,
    blur_radius: u32,
    amplification: u8,
) -> RgbaImage {
    let (w, h) = expected.dimensions();
    assert_eq!(
        (w, h),
        actual.dimensions(),
        "dimension mismatch: expected {}x{}, actual {}x{}",
        w,
        h,
        actual.width(),
        actual.height(),
    );

    let amp = amplification.max(1) as f32;

    // Convert to grayscale f32, premultiplied by alpha so transparent
    // pixels contribute zero regardless of their RGB values.
    let to_gray = |img: &RgbaImage| -> Vec<f32> {
        img.pixels()
            .map(|px| {
                let a = px[3] as f32 / 255.0;
                (0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32) * a
            })
            .collect()
    };

    let gray_exp = to_gray(expected);
    let gray_act = to_gray(actual);

    // Box blur for local mean (running-sum, O(1) per pixel regardless of radius)
    let blurred_exp = box_blur_gray(&gray_exp, w, h, blur_radius);
    let blurred_act = box_blur_gray(&gray_act, w, h, blur_radius);

    // High-pass residuals: |pixel - local_mean|
    let mut diff = RgbaImage::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let res_exp = (gray_exp[i] - blurred_exp[i]).abs();
            let res_act = (gray_act[i] - blurred_act[i]).abs();

            // Structural diff: positive = expected has more structure
            let delta = (res_exp - res_act) * amp;

            if delta.abs() < 1.0 {
                diff.put_pixel(x, y, Rgba([24, 24, 24, 255]));
            } else if delta > 0.0 {
                // Expected has structure, actual doesn't → cyan (missing)
                let v = delta.min(255.0) as u8;
                diff.put_pixel(x, y, Rgba([0, v, v, 255]));
            } else {
                // Actual has structure, expected doesn't → orange (added)
                let v = (-delta).min(255.0) as u8;
                diff.put_pixel(x, y, Rgba([v, (v as f32 * 0.6) as u8, 0, 255]));
            }
        }
    }

    diff
}

/// Generate structural diff from raw RGBA byte slices.
pub fn generate_structural_diff_raw(
    expected: &[u8],
    actual: &[u8],
    width: u32,
    height: u32,
    blur_radius: u32,
    amplification: u8,
) -> RgbaImage {
    let exp_img = RgbaImage::from_raw(width, height, expected.to_vec())
        .expect("expected: invalid dimensions");
    let act_img =
        RgbaImage::from_raw(width, height, actual.to_vec()).expect("actual: invalid dimensions");
    generate_structural_diff(&exp_img, &act_img, blur_radius, amplification)
}

/// Box blur on a grayscale f32 buffer. O(1) per pixel via running sums.
fn box_blur_gray(src: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    if radius == 0 {
        return src.to_vec();
    }

    let w = w as usize;
    let h = h as usize;
    let r = radius as usize;
    let mut tmp = vec![0.0f32; w * h];
    let mut out = vec![0.0f32; w * h];

    // Horizontal pass
    for y in 0..h {
        let row = y * w;
        let mut sum = 0.0f32;
        let mut count = 0u32;

        // Initialize window
        for x in 0..=r.min(w - 1) {
            sum += src[row + x];
            count += 1;
        }
        tmp[row] = sum / count as f32;

        for x in 1..w {
            // Add right edge
            let right = x + r;
            if right < w {
                sum += src[row + right];
                count += 1;
            }
            // Remove left edge
            if x > r + 1 {
                // This shouldn't happen since left = x - r - 1
            }
            let left_remove = x.wrapping_sub(r + 1);
            if x > r {
                sum -= src[row + left_remove];
                count -= 1;
            }
            tmp[row + x] = sum / count as f32;
        }
    }

    // Vertical pass
    for x in 0..w {
        let mut sum = 0.0f32;
        let mut count = 0u32;

        for y in 0..=r.min(h - 1) {
            sum += tmp[y * w + x];
            count += 1;
        }
        out[x] = sum / count as f32;

        for y in 1..h {
            let bottom = y + r;
            if bottom < h {
                sum += tmp[bottom * w + x];
                count += 1;
            }
            if y > r {
                sum -= tmp[(y - r - 1) * w + x];
                count -= 1;
            }
            out[y * w + x] = sum / count as f32;
        }
    }

    out
}

/// Create a 4-panel comparison montage: Expected | Actual | Pixel Diff | Structural Diff.
///
/// The pixel diff panel shows amplified per-channel absolute differences.
/// The structural diff panel shows high-pass residual differences —
/// cyan for structure missing in actual, orange for structure added.
///
/// # Panics
///
/// Panics if expected and actual have different dimensions.
pub fn create_structural_montage(
    expected: &RgbaImage,
    actual: &RgbaImage,
    amplification: u8,
    gap: u32,
    blur_radius: u32,
) -> RgbaImage {
    let pixel_diff = generate_diff_image(expected, actual, amplification);
    let struct_diff = generate_structural_diff(expected, actual, blur_radius, amplification);
    create_montage(&[expected, actual, &pixel_diff, &struct_diff], gap)
}

/// Create a 4-panel structural montage from raw RGBA byte slices.
pub fn create_structural_montage_raw(
    expected: &[u8],
    actual: &[u8],
    width: u32,
    height: u32,
    amplification: u8,
    gap: u32,
    blur_radius: u32,
) -> RgbaImage {
    let exp_img = RgbaImage::from_raw(width, height, expected.to_vec())
        .expect("expected: invalid dimensions");
    let act_img =
        RgbaImage::from_raw(width, height, actual.to_vec()).expect("actual: invalid dimensions");
    create_structural_montage(&exp_img, &act_img, amplification, gap, blur_radius)
}

// ─── Spatial analysis ───────────────────────────────────────────────────

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
        let labels_2x2 = ["top-left", "top-right", "bot-left", "bot-right"];
        let labels_3x3 = [
            "top-left",
            "top-center",
            "top-right",
            "mid-left",
            "center",
            "mid-right",
            "bot-left",
            "bot-center",
            "bot-right",
        ];
        let max_delta_idx = self
            .regions
            .iter()
            .enumerate()
            .max_by_key(|(_, r)| r.max_delta)
            .map(|(i, _)| i);

        writeln!(f, "Spatial diff ({}x{} grid):", self.cols, self.rows)?;

        for (i, r) in self.regions.iter().enumerate() {
            let label = if self.cols == 3 && self.rows == 3 {
                labels_3x3.get(i).unwrap_or(&"??").to_string()
            } else if self.cols == 2 && self.rows == 2 {
                labels_2x2.get(i).unwrap_or(&"??").to_string()
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
                    writeln!(
                        f,
                        "         expected: uniform, actual: has content (added feature?)"
                    )?;
                } else if exp_has_content && !act_has_content {
                    writeln!(
                        f,
                        "         expected: has content, actual: uniform (missing feature?)"
                    )?;
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

                    // Premultiply by alpha so transparent pixels contribute zero.
                    let ea = expected[off + 3] as i32;
                    let aa = actual[off + 3] as i32;

                    let mut any_diff = false;
                    for c in 0..3 {
                        let e = (expected[off + c] as i32 * ea / 255) as i16;
                        let a = (actual[off + c] as i32 * aa / 255) as i16;
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

/// Create a labeled 2×2 grid montage.
///
/// **Deprecated** — use [`MontageOptions::render`] instead:
/// ```ignore
/// MontageOptions::default().render(&expected, &actual, &AnnotationText::empty())
/// ```
#[deprecated(since = "0.2.3", note = "use MontageOptions::render() instead")]
pub fn create_annotated_montage(
    expected: &RgbaImage,
    actual: &RgbaImage,
    amplification: u8,
    gap: u32,
    annotation: &AnnotationText,
) -> RgbaImage {
    MontageOptions {
        amplification,
        gap,
        ..Default::default()
    }
    .render(expected, actual, annotation)
}

/// Create a labeled 2×2 grid montage from raw RGBA byte slices.
///
/// **Deprecated** — use [`MontageOptions::render`] instead.
#[deprecated(since = "0.2.3", note = "use MontageOptions::render() instead")]
pub fn create_annotated_montage_raw(
    expected: &[u8],
    actual: &[u8],
    width: u32,
    height: u32,
    amplification: u8,
    gap: u32,
    annotation: &AnnotationText,
) -> RgbaImage {
    let exp_img = RgbaImage::from_raw(width, height, expected.to_vec())
        .expect("expected: invalid dimensions for pixel data");
    let act_img = RgbaImage::from_raw(width, height, actual.to_vec())
        .expect("actual: invalid dimensions for pixel data");
    #[allow(deprecated)]
    create_annotated_montage(&exp_img, &act_img, amplification, gap, annotation)
}

/// Format a regression report as annotation text.
///
/// **Deprecated** — use [`AnnotationText::from_report`] instead.
#[deprecated(since = "0.2.3", note = "use AnnotationText::from_report() instead")]
pub fn format_annotation(
    report: &crate::testing::RegressionReport,
    tolerance: &crate::testing::RegressionTolerance,
) -> AnnotationText {
    AnnotationText::from_report(report, tolerance)
}

/// Format annotation with spatial analysis included.
///
/// **Deprecated** — use [`AnnotationText::from_report`] instead.
/// Spatial heatmap is now computed automatically by [`MontageOptions::render`].
#[deprecated(
    since = "0.2.3",
    note = "use AnnotationText::from_report(); spatial is now computed by MontageOptions::render()"
)]
pub fn format_annotation_spatial(
    report: &crate::testing::RegressionReport,
    tolerance: &crate::testing::RegressionTolerance,
    _spatial: Option<&SpatialAnalysis>,
) -> AnnotationText {
    AnnotationText::from_report(report, tolerance)
}

fn render_montage_impl(
    options: &MontageOptions,
    expected: &RgbaImage,
    actual: &RgbaImage,
    annotation: &AnnotationText,
) -> RgbaImage {
    use crate::font;

    let amplification = options.amplification;

    // Generate diffs at original resolution (before upscale) so pixel-level
    // differences are computed on real pixels, not replicated blocks.
    let pixel_diff = generate_diff_image(expected, actual, amplification);
    let struct_diff = generate_structural_diff(expected, actual, 3, amplification);

    // Compute spatial analysis from original pixels (before upscale).
    let (w, h) = expected.dimensions();
    let (sg_cols, sg_rows) = options.spatial_grid;
    let spatial = spatial_analysis(expected.as_raw(), actual.as_raw(), w, h, sg_cols, sg_rows);
    let has_differences = spatial.regions.iter().any(|r| r.pixels_differing > 0.0);

    // Pixelate-upscale all four panels if the source is tiny.
    let min_panel_size = options.min_panel_size;
    let up_expected;
    let up_actual;
    let up_pixel_diff;
    let up_struct_diff;
    let (expected, actual, pixel_diff, struct_diff) =
        if min_panel_size > 0 && (w < min_panel_size || h < min_panel_size) {
            up_expected = pixelate_upscale(expected, min_panel_size);
            up_actual = pixelate_upscale(actual, min_panel_size);
            up_pixel_diff = pixelate_upscale(&pixel_diff, min_panel_size);
            up_struct_diff = pixelate_upscale(&struct_diff, min_panel_size);
            (&up_expected, &up_actual, &up_pixel_diff, &up_struct_diff)
        } else {
            (expected, actual, &pixel_diff, &struct_diff)
        };

    // Minimum padding: 30% of a base char width, at least 8px
    let min_pad = (font::GLYPH_W * 3 / 10).max(8);
    let pad = options.gap.max(min_pad);
    let bg = Rgba([18, 18, 18, 255]);
    let label_fg = [220, 220, 220, 255];
    let label_bg = [40, 40, 40, 255];

    let panel_images: [&RgbaImage; 4] = [expected, actual, pixel_diff, struct_diff];

    let panel_w = expected.width();
    let panel_h = expected.height();

    // Label sizing: fit "PIXEL DIFF" (10 chars) within cell_w minus padding
    let label_inset = pad;
    let label_area_w = panel_w.saturating_sub(label_inset * 2);
    let longest_label = 10u32; // "PIXEL DIFF"
    let label_char_h = ((label_area_w as f32 / longest_label as f32)
        * (font::GLYPH_H as f32 / font::GLYPH_W as f32))
        .floor() as u32;
    let label_char_h = label_char_h.clamp(font::GLYPH_H, font::GLYPH_H * 3);

    // Plain labels: rendered centered within panel_w via render_lines_fitted
    let plain_labels = ["EXPECTED", "ACTUAL", "PIXEL DIFF"];
    let plain_images: Vec<(Vec<u8>, u32, u32)> = plain_labels
        .iter()
        .map(|label| {
            let lines: Vec<(&str, [u8; 4])> = vec![(label, label_fg)];
            font::render_lines_fitted(&lines, label_bg, label_area_w)
        })
        .collect();

    // ADD  REMOVE: left-aligned and right-aligned within the cell
    let add_img = font::render_text_height("ADD", [255, 180, 80, 255], label_bg, label_char_h);
    let rem_img = font::render_text_height("REMOVE", [80, 220, 220, 255], label_bg, label_char_h);
    let ar_h = label_char_h;
    let mut ar_rgba = RgbaImage::from_pixel(panel_w, ar_h, Rgba(label_bg));
    // ADD left-aligned with inset
    if add_img.1 > 0
        && add_img.2 > 0
        && let Some(img) = RgbaImage::from_raw(add_img.1, add_img.2, add_img.0.clone())
    {
        imageops::overlay(&mut ar_rgba, &img, label_inset as i64, 0);
    }
    // REMOVE right-aligned with inset
    if rem_img.1 > 0
        && rem_img.2 > 0
        && let Some(img) = RgbaImage::from_raw(rem_img.1, rem_img.2, rem_img.0.clone())
    {
        let rx = panel_w.saturating_sub(rem_img.1 + label_inset);
        imageops::overlay(&mut ar_rgba, &img, rx as i64, 0);
    }
    let ar_raw = ar_rgba.into_raw();

    let label_images: Vec<(Vec<u8>, u32, u32)> = vec![
        plain_images[0].clone(),
        plain_images[1].clone(),
        plain_images[2].clone(),
        (ar_raw, panel_w, ar_h),
    ];

    let label_h = label_images.iter().map(|(_, _, h)| *h).max().unwrap_or(0) + 4;

    // 2x2 grid dimensions
    let cell_w = panel_w;
    let cell_h = label_h + panel_h;
    let grid_w = pad + cell_w + pad + cell_w + pad;
    let grid_h = pad + cell_h + pad + cell_h + pad;

    // Title text — white, fitted to grid width
    let text_avail = grid_w.saturating_sub(pad * 2);
    let title_rendered = annotation.title.as_ref().and_then(|t| {
        if t.is_empty() {
            return None;
        }
        let lines: Vec<(&str, [u8; 4])> = vec![(t.as_str(), [255, 255, 255, 255])];
        Some(font::render_lines_fitted(
            &lines,
            [25, 25, 25, 255],
            text_avail,
        ))
    });

    // Primary text — colored per-line, fitted to grid width (no wrapping)
    let primary_rendered = if !annotation.primary_lines.is_empty() {
        let line_refs: Vec<(&str, [u8; 4])> = annotation
            .primary_lines
            .iter()
            .map(|(s, c)| (s.as_str(), *c))
            .collect();
        let r = font::render_lines_fitted(&line_refs, [30, 30, 30, 255], text_avail);
        Some(r)
    } else {
        None
    };

    // Heatmap grid — only shown when there are actual differences
    let heatmap = if has_differences {
        Some(render_heatmap_grid(&spatial, grid_w, pad))
    } else {
        None
    };

    // Extra text (alpha info etc) — small, word-wrapped
    let primary_line_h = primary_rendered
        .as_ref()
        .map_or(font::GLYPH_H, |(_, _, h)| {
            let n = annotation.primary_lines.len().max(1) as u32;
            *h / n
        });
    let extra_char_h = (primary_line_h * 7 / 10).max(font::GLYPH_H / 4);
    let extra_rendered = if !annotation.extra.is_empty() {
        let r = font::render_text_wrapped(
            &annotation.extra,
            COLOR_DETAIL,
            [25, 25, 25, 255],
            extra_char_h,
            text_avail,
        );
        Some(r)
    } else {
        None
    };

    let title_h = title_rendered.as_ref().map_or(0, |(_, _, h)| *h + pad * 2);
    let primary_h = primary_rendered
        .as_ref()
        .map_or(0, |(_, _, h)| *h + pad * 2);
    let heatmap_h = heatmap.as_ref().map_or(0, |img| img.height());
    let extra_h = extra_rendered.as_ref().map_or(0, |(_, _, h)| *h + pad * 2);

    let total_w = grid_w;
    let total_h = grid_h + title_h + primary_h + heatmap_h + extra_h;

    let mut output = RgbaImage::from_pixel(total_w, total_h, bg);

    // Place 4 panels in 2x2 grid — clip panels to cell bounds
    for (i, panel) in panel_images.iter().enumerate() {
        let col = (i % 2) as u32;
        let row = (i / 2) as u32;
        let x0 = pad + col * (cell_w + pad);
        let y0 = pad + row * (cell_h + pad);

        // Label bar background
        fill_rect(&mut output, x0, y0, cell_w, label_h, label_bg);

        // Label text centered in bar
        let (ref lbuf, lw, lh) = label_images[i];
        if lw > 0 && lh > 0 {
            let lx_off = (cell_w.saturating_sub(lw)) / 2;
            let ly_off = (label_h.saturating_sub(lh)) / 2;
            if let Some(label_img) = RgbaImage::from_raw(lw, lh, lbuf.clone()) {
                imageops::overlay(
                    &mut output,
                    &label_img,
                    (x0 + lx_off) as i64,
                    (y0 + ly_off) as i64,
                );
            }
        }

        // Panel image — crop to cell_w × panel_h if it overflows
        let pw = panel.width().min(cell_w);
        let ph = panel.height().min(panel_h);
        let cropped = imageops::crop_imm(*panel, 0, 0, pw, ph).to_image();
        imageops::overlay(&mut output, &cropped, x0 as i64, (y0 + label_h) as i64);
    }

    // Title strip
    let mut y_cursor = grid_h;
    if let Some((tbuf, tw, th)) = title_rendered
        && tw > 0
        && th > 0
    {
        fill_rect(
            &mut output,
            0,
            y_cursor,
            total_w,
            title_h,
            [25, 25, 25, 255],
        );
        if let Some(text_img) = RgbaImage::from_raw(tw, th, tbuf) {
            let tx = (total_w.saturating_sub(tw)) / 2;
            imageops::overlay(&mut output, &text_img, tx as i64, (y_cursor + pad) as i64);
        }
        y_cursor += title_h;
        // Horizontal separator line
        fill_rect(
            &mut output,
            pad,
            y_cursor.saturating_sub(1),
            total_w - pad * 2,
            1,
            [60, 60, 60, 255],
        );
    }

    // Primary text strip — center the first line (PASS/FAIL)
    if let Some((tbuf, tw, th)) = primary_rendered
        && tw > 0
        && th > 0
    {
        fill_rect(
            &mut output,
            0,
            y_cursor,
            total_w,
            primary_h,
            [30, 30, 30, 255],
        );
        if let Some(text_img) = RgbaImage::from_raw(tw, th, tbuf) {
            // Center horizontally
            let tx = (total_w.saturating_sub(tw)) / 2;
            imageops::overlay(&mut output, &text_img, tx as i64, (y_cursor + pad) as i64);
        }
        y_cursor += primary_h;
    }

    // Heatmap grid
    if let Some(ref heatmap_img) = heatmap {
        imageops::overlay(&mut output, heatmap_img, 0, y_cursor as i64);
        y_cursor += heatmap_h;
    }

    // Extra text strip
    if let Some((tbuf, tw, th)) = extra_rendered
        && tw > 0
        && th > 0
    {
        fill_rect(
            &mut output,
            0,
            y_cursor,
            total_w,
            extra_h,
            [25, 25, 25, 255],
        );
        if let Some(text_img) = RgbaImage::from_raw(tw, th, tbuf) {
            imageops::overlay(&mut output, &text_img, pad as i64, (y_cursor + pad) as i64);
        }
    }

    output
}

/// Render a 3x3 spatial heatmap grid image.
///
/// Each cell's background is tinted red proportional to its severity
/// relative to the worst cell. Cell text shows `d:N` (max delta) and
/// `P%` (percent differing). ADDED/MISSING tags shown when applicable.
fn render_heatmap_grid(spatial: &SpatialAnalysis, total_w: u32, pad: u32) -> RgbaImage {
    use crate::font;

    let cols = spatial.cols;
    let rows = spatial.rows;
    let cell_gap = 3u32;

    let inner_w = total_w.saturating_sub(pad * 2);
    let cell_w = (inner_w.saturating_sub(cell_gap * (cols - 1))) / cols;
    let cell_h = cell_w * 3 / 4; // slightly taller to fit 3 lines
    let grid_px_w = cell_w * cols + cell_gap * (cols - 1);
    let grid_px_h = cell_h * rows + cell_gap * (rows - 1);
    let img_w = grid_px_w + pad * 2;
    let img_h = grid_px_h + pad * 2;

    let mut img = RgbaImage::from_pixel(img_w, img_h, Rgba([18, 18, 18, 255]));

    // Find worst cell for relative coloring
    let max_severity = spatial
        .regions
        .iter()
        .map(|r| r.max_delta as f32 * r.pixels_differing as f32)
        .fold(0.0f32, f32::max)
        .max(0.001); // avoid div by zero

    for r in &spatial.regions {
        let cx = pad + r.col * (cell_w + cell_gap);
        let cy = pad + r.row * (cell_h + cell_gap);

        // Background: red intensity by relative severity
        let severity = r.max_delta as f32 * r.pixels_differing as f32;
        let ratio = (severity / max_severity).clamp(0.0, 1.0);
        let bg_r = (30.0 + ratio * 120.0) as u8; // 30..150
        let bg_g = (30.0 - ratio * 20.0).max(10.0) as u8; // 30..10
        let bg_b = bg_g;
        let cell_bg = [bg_r, bg_g, bg_b, 255];
        fill_rect(&mut img, cx, cy, cell_w, cell_h, cell_bg);

        // Cell text — 3 lines for hot cells, "ok" for clean
        // Padding: 30% of one character width on each side
        let pct = r.pixels_differing * 100.0;
        let one_char_w = font::GLYPH_W; // base char width before scaling
        let cell_inset = one_char_w * 3 / 10; // 30% of one char
        let cell_text_w = cell_w.saturating_sub(cell_inset * 2);

        if pct < 0.05 {
            // Clean cell: small green "ok"
            let ok_lines: Vec<(&str, [u8; 4])> = vec![("ok", COLOR_OK)];
            if cell_text_w > 0 {
                let (tbuf, tw, th) = font::render_lines_fitted(&ok_lines, cell_bg, cell_text_w);
                if tw > 0 && th > 0 {
                    let tx = cx + (cell_w.saturating_sub(tw)) / 2;
                    let ty = cy + (cell_h.saturating_sub(th)) / 2;
                    if let Some(text_img) = RgbaImage::from_raw(tw, th, tbuf) {
                        imageops::overlay(&mut img, &text_img, tx as i64, ty as i64);
                    }
                }
            }
        } else {
            // Hot cell: 3 lines
            // Line 1: "25% \u{0394}255"
            let line1 = format!("{:.0}% \u{0394}{}", pct, r.max_delta);
            // Line 2: zdsim value for this cell (per-pixel severity)
            let cell_zdsim = r.avg_delta / 255.0; // normalized
            let line2 = format!("zdsim:{:.3}", cell_zdsim);
            // Line 3: structural tag
            let exp_has = r.expected_variance > 10.0;
            let act_has = r.actual_variance > 10.0;
            let line3 = if !exp_has && act_has {
                "ADDED"
            } else if exp_has && !act_has {
                "MISSING"
            } else if exp_has && act_has && r.avg_delta > 10.0 {
                "changed"
            } else {
                ""
            };

            let text_fg = if ratio > 0.5 {
                [255, 255, 255, 255]
            } else {
                [220, 220, 220, 255]
            };
            let mut cell_lines: Vec<(&str, [u8; 4])> =
                vec![(&line1, text_fg), (&line2, [170, 170, 170, 255])];
            if !line3.is_empty() {
                let tag_color = if line3 == "MISSING" {
                    [255, 120, 120, 255] // red-ish
                } else if line3 == "ADDED" {
                    [255, 180, 80, 255] // orange
                } else {
                    [200, 200, 120, 255] // yellow
                };
                cell_lines.push((line3, tag_color));
            }

            if cell_text_w > 0 {
                let (tbuf, tw, th) = font::render_lines_fitted(&cell_lines, cell_bg, cell_text_w);
                if tw > 0 && th > 0 {
                    let tx = cx + (cell_w.saturating_sub(tw)) / 2;
                    let ty = cy + (cell_h.saturating_sub(th)) / 2;
                    if let Some(text_img) = RgbaImage::from_raw(tw, th, tbuf) {
                        imageops::overlay(&mut img, &text_img, tx as i64, ty as i64);
                    }
                }
            }
        }
    }

    img
}

/// Fill a rectangle with a solid color.
fn fill_rect(img: &mut RgbaImage, x0: u32, y0: u32, w: u32, h: u32, color: [u8; 4]) {
    let px = Rgba(color);
    let img_w = img.width();
    let img_h = img.height();
    for y in y0..y0.saturating_add(h).min(img_h) {
        for x in x0..x0.saturating_add(w).min(img_w) {
            img.put_pixel(x, y, px);
        }
    }
}

/// Rendering options for annotated montages.
///
/// Use `Default` for the common case — all fields have sensible defaults.
///
/// ```
/// # use zensim_regress::diff_image::MontageOptions;
/// let opts = MontageOptions::default();
/// let custom = MontageOptions { amplification: 50, ..Default::default() };
/// ```
pub struct MontageOptions {
    /// Diff amplification factor (default: 10).
    pub amplification: u8,
    /// Gap between panels in pixels (default: 2, clamped up to min font padding).
    pub gap: u32,
    /// Minimum panel dimension — pixelate-upscale tiny images to at least this
    /// size. Set to 0 to disable. Default: 256.
    pub min_panel_size: u32,
    /// Spatial heatmap grid dimensions (cols, rows). Default: (3, 3).
    pub spatial_grid: (u32, u32),
}

impl Default for MontageOptions {
    fn default() -> Self {
        Self {
            amplification: 10,
            gap: 2,
            min_panel_size: DEFAULT_MIN_PANEL_SIZE,
            spatial_grid: (3, 3),
        }
    }
}

impl MontageOptions {
    /// Render a labeled 2×2 grid montage with annotation text and spatial heatmap.
    ///
    /// Layout (portrait-friendly, survives LLM image downscaling):
    /// ```text
    /// ┌──────────────┬──────────────┐
    /// │  EXPECTED    │  ACTUAL      │
    /// │  (image)     │  (image)     │
    /// ├──────────────┼──────────────┤
    /// │  PIXEL DIFF  │  STRUCTURE   │
    /// │  (image)     │  (image)     │
    /// ├──────────────┴──────────────┤
    /// │  FAIL                       │  ← primary (large font)
    /// │  zdsim: 0.13 > 0.01 FAIL   │
    /// │  delta: [12,8,3] > 1 FAIL  │
    /// ├─────────────────────────────┤
    /// │  [3×3 spatial heatmap]      │  ← auto-computed from pixels
    /// ├─────────────────────────────┤
    /// │  alpha: max delta 2 ...     │  ← extra text
    /// └─────────────────────────────┘
    /// ```
    ///
    /// Spatial heatmap is computed automatically from the pixel data when there
    /// are differences. Tiny images are pixelate-upscaled per `min_panel_size`.
    ///
    /// # Panics
    ///
    /// Panics if expected and actual have different dimensions.
    pub fn render(
        &self,
        expected: &RgbaImage,
        actual: &RgbaImage,
        annotation: &AnnotationText,
    ) -> RgbaImage {
        render_montage_impl(self, expected, actual, annotation)
    }
}

/// Annotation data for the montage: verdict lines and optional extra text.
///
/// Use [`from_report`](Self::from_report) when you have a regression report,
/// or [`empty`](Self::empty) for a bare montage with no annotations.
pub struct AnnotationText {
    /// Title shown above the constraint lines (e.g., test name + detail).
    /// Rendered in white on the dark background strip. None = no title.
    pub title: Option<String>,
    /// Colored lines for the primary text block.
    /// Red for failing constraints, green for passing, gray for info.
    pub primary_lines: Vec<(String, [u8; 4])>,
    /// Spatial analysis — ignored by [`MontageOptions::render`] which computes
    /// its own from the pixel data. Retained for backward compatibility.
    #[deprecated(
        since = "0.2.3",
        note = "spatial is now computed automatically by MontageOptions::render()"
    )]
    pub spatial: Option<SpatialAnalysis>,
    /// Extra text lines (alpha info, etc). Shown below heatmap if present.
    pub extra: String,
}

const COLOR_FAIL: [u8; 4] = [255, 80, 80, 255]; // red
const COLOR_OK: [u8; 4] = [80, 220, 80, 255]; // green
const COLOR_DETAIL: [u8; 4] = [170, 170, 170, 255]; // dim gray

impl AnnotationText {
    /// No annotation — produces a bare montage with no text or heatmap.
    #[allow(deprecated)]
    pub fn empty() -> Self {
        Self {
            title: None,
            primary_lines: vec![],
            spatial: None,
            extra: String::new(),
        }
    }

    /// Build annotation from a regression report.
    ///
    /// Shows each constraint as `actual > limit FAIL` (red) or
    /// `actual <= limit ok` (green). Spatial heatmap is computed
    /// by the montage function from the actual pixels.
    pub fn from_report(
        report: &crate::testing::RegressionReport,
        tolerance: &crate::testing::RegressionTolerance,
    ) -> Self {
        use zensim::score_to_dissimilarity;

        let mut lines: Vec<(String, [u8; 4])> = Vec::new();

        // zdsim
        let zdsim = score_to_dissimilarity(report.score());
        let zdsim_limit = score_to_dissimilarity(tolerance.min_similarity());
        let zdsim_ok = zdsim <= zdsim_limit;
        lines.push((
            if zdsim_ok {
                format!("ok: zdsim {:.4} <= {:.4}", zdsim, zdsim_limit)
            } else {
                format!("FAIL: zdsim {:.4} > {:.4}", zdsim, zdsim_limit)
            },
            if zdsim_ok { COLOR_OK } else { COLOR_FAIL },
        ));

        // delta
        let [dr, dg, db] = report.max_channel_delta();
        let max_d = dr.max(dg).max(db);
        let delta_ok = max_d <= tolerance.max_delta();
        lines.push((
            if delta_ok {
                format!(
                    "ok: \u{0394}[{},{},{}] <= {}",
                    dr,
                    dg,
                    db,
                    tolerance.max_delta()
                )
            } else {
                format!(
                    "FAIL: \u{0394}[{},{},{}] > {}",
                    dr,
                    dg,
                    db,
                    tolerance.max_delta()
                )
            },
            if delta_ok { COLOR_OK } else { COLOR_FAIL },
        ));

        // pixels differing
        let pct = if report.pixel_count() > 0 {
            report.pixels_differing() as f64 / report.pixel_count() as f64
        } else {
            0.0
        };
        let pct_limit = tolerance.max_pixels_different();
        let pct_ok = pct <= pct_limit;
        lines.push((
            if pct_ok {
                format!(
                    "ok: {:.1}% differ <= {:.1}%",
                    pct * 100.0,
                    pct_limit * 100.0
                )
            } else {
                format!(
                    "FAIL: {:.1}% differ > {:.1}%",
                    pct * 100.0,
                    pct_limit * 100.0
                )
            },
            if pct_ok { COLOR_OK } else { COLOR_FAIL },
        ));

        // Error type
        let category = format!("{:?}", report.category()).to_lowercase();
        let mut cat_text = format!("error type: {}", category);
        if let Some(bias) = report.rounding_bias() {
            if bias.balanced {
                cat_text.push_str(" (balanced)");
            } else {
                let all_pos = bias.positive_fraction.iter().all(|&f| f > 0.8);
                let all_neg = bias.positive_fraction.iter().all(|&f| f < 0.2);
                if all_pos {
                    cat_text.push_str(" (truncation)");
                } else if all_neg {
                    cat_text.push_str(" (ceiling)");
                }
            }
        }
        lines.push((cat_text, COLOR_DETAIL));

        // Extra: alpha info
        let mut extra = String::new();
        if report.alpha_max_delta() > 0 {
            extra = format!(
                "alpha: max delta {} ({} pixels differ)",
                report.alpha_max_delta(),
                report.alpha_pixels_differing(),
            );
        }

        #[allow(deprecated)]
        Self {
            title: None,
            primary_lines: lines,
            spatial: None,
            extra,
        }
    }

    /// Build annotation from a report computed on dimension-mismatched images.
    ///
    /// Shows the same constraint lines as [`from_report`](Self::from_report),
    /// plus prominent amber-colored dimension-mismatch warning lines showing
    /// the category and comparison method.
    pub fn from_resized_report(
        report: &crate::testing::RegressionReport,
        tolerance: &crate::testing::RegressionTolerance,
    ) -> Self {
        let mut ann = Self::from_report(report, tolerance);

        if let Some(dim_info) = report.dimension_info() {
            const COLOR_WARN: [u8; 4] = [255, 200, 60, 255];

            let dim_line = format!("DIMENSIONS: {}", dim_info.description());
            let method_line = format!(
                "Actual {} for comparison (approximate)",
                dim_info.method(),
            );
            ann.primary_lines.insert(0, (method_line, COLOR_WARN));
            ann.primary_lines.insert(0, (dim_line, COLOR_WARN));
        }

        ann
    }

    /// Set the title (builder pattern).
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }
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
    fn diff_transparent_pixels_are_zero() {
        // Different RGB but both fully transparent — should show no diff
        let a = RgbaImage::from_pixel(4, 4, Rgba([255, 0, 0, 0]));
        let b = RgbaImage::from_pixel(4, 4, Rgba([0, 255, 0, 0]));
        let diff = generate_diff_image(&a, &b, 10);
        for pixel in diff.pixels() {
            assert_eq!(
                *pixel,
                Rgba([24, 24, 24, 255]),
                "transparent pixels should diff to zero"
            );
        }
    }

    #[test]
    fn diff_semitransparent_scales_by_alpha() {
        // Same RGB, different alpha — diff should reflect the visual difference
        let a = RgbaImage::from_pixel(4, 4, Rgba([200, 200, 200, 128]));
        let b = RgbaImage::from_pixel(4, 4, Rgba([200, 200, 200, 0]));
        let diff = generate_diff_image(&a, &b, 1);
        let p = diff.get_pixel(0, 0);
        // a premul: 200*128/255 ≈ 100, b premul: 0. delta ≈ 100 per channel.
        assert!(p[0] > 90 && p[0] < 110, "expected ~100, got {}", p[0]);
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
    fn pixelate_noop_when_large_enough() {
        let img = RgbaImage::from_pixel(300, 300, Rgba([100, 100, 100, 255]));
        let up = pixelate_upscale(&img, 256);
        assert_eq!(up.dimensions(), (300, 300));
    }

    #[test]
    fn pixelate_scales_8x8_to_at_least_256() {
        let img = RgbaImage::from_fn(8, 8, |x, y| {
            Rgba([(x * 32) as u8, (y * 32) as u8, 128, 255])
        });
        let up = pixelate_upscale(&img, 256);
        // 256/8 = 32, so N=32, output = 256×256
        assert_eq!(up.dimensions(), (256, 256));
        // Each original pixel should be a 32×32 block
        let orig = *img.get_pixel(1, 2);
        for dy in 0..32 {
            for dx in 0..32 {
                assert_eq!(*up.get_pixel(32 + dx, 2 * 32 + dy), orig);
            }
        }
    }

    #[test]
    fn pixelate_rectangular() {
        let img = RgbaImage::from_pixel(4, 16, Rgba([50, 100, 150, 255]));
        let up = pixelate_upscale(&img, 256);
        // N = max(ceil(256/4), ceil(256/16)) = max(64, 16) = 64
        assert_eq!(up.dimensions(), (256, 1024));
    }

    #[test]
    fn pixelate_disabled_with_zero() {
        let img = RgbaImage::from_pixel(4, 4, Rgba([100; 4]));
        let up = pixelate_upscale(&img, 0);
        assert_eq!(up.dimensions(), (4, 4));
    }

    #[test]
    #[allow(deprecated)]
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
    #[allow(deprecated)]
    fn comparison_montage_raw_matches_typed() {
        let exp = RgbaImage::from_pixel(4, 4, Rgba([100, 100, 100, 255]));
        let act = RgbaImage::from_pixel(4, 4, Rgba([105, 100, 95, 255]));

        let typed = create_comparison_montage(&exp, &act, 10, 2);
        let raw = create_comparison_montage_raw(exp.as_raw(), act.as_raw(), 4, 4, 10, 2);
        assert_eq!(typed, raw);
    }

    #[test]
    fn annotated_montage_adds_text_strips() {
        let exp = RgbaImage::from_pixel(32, 32, Rgba([100, 100, 100, 255]));
        let act = RgbaImage::from_pixel(32, 32, Rgba([110, 100, 90, 255]));
        let ann = AnnotationText {
            primary_lines: vec![
                ("FAIL".into(), COLOR_FAIL),
                ("zdsim: 0.13 > 0.01 FAIL".into(), COLOR_FAIL),
            ],
            ..AnnotationText::empty()
        };
        let opts = MontageOptions {
            gap: 6,
            min_panel_size: 0, // disable upscale to test at native 32×32
            ..Default::default()
        };
        let montage = opts.render(&exp, &act, &ann);

        assert!(montage.width() >= 32 * 2 + 6 * 3);
        assert!(montage.height() > 32 * 2);
    }

    #[test]
    fn annotated_montage_empty_text_no_text_strip() {
        let exp = RgbaImage::from_pixel(16, 16, Rgba([100; 4]));
        let act = RgbaImage::from_pixel(16, 16, Rgba([100; 4]));
        let with_text = AnnotationText {
            primary_lines: vec![("hello".into(), [255; 4])],
            ..AnnotationText::empty()
        };
        let no_text = AnnotationText::empty();
        let opts = MontageOptions {
            gap: 6,
            min_panel_size: 0, // disable upscale
            ..Default::default()
        };
        let with = opts.render(&exp, &act, &with_text);
        let without = opts.render(&exp, &act, &no_text);
        assert!(without.height() < with.height());
    }

    #[test]
    fn annotated_montage_upscales_tiny_images() {
        let exp = RgbaImage::from_pixel(8, 8, Rgba([100, 100, 100, 255]));
        let act = RgbaImage::from_pixel(8, 8, Rgba([110, 100, 90, 255]));
        // Default options — upscale to 256
        let montage = MontageOptions::default().render(&exp, &act, &AnnotationText::empty());
        // Panels should be 256×256 (8×32), so montage width >= 2*256
        assert!(
            montage.width() >= 512,
            "montage width {} should be >= 512 after upscaling 8×8 panels",
            montage.width()
        );
    }

    #[test]
    fn spatial_identical_images() {
        let w = 64u32;
        let h = 64;
        let px: Vec<u8> = (0..w * h)
            .flat_map(|i| [(i % 255) as u8, 128, 64, 255])
            .collect();
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
        // Opaque pixels: R=128, G=128, B=128, A=255
        let exp: Vec<u8> = (0..w * h).flat_map(|_| [128u8, 128, 128, 255]).collect();
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
        assert!(text.contains("top-left"));
    }

    #[test]
    fn structural_diff_identical_is_dark() {
        let img = RgbaImage::from_fn(32, 32, |x, y| {
            Rgba([(x * 8) as u8, (y * 8) as u8, 128, 255])
        });
        let diff = generate_structural_diff(&img, &img, 3, 10);
        // All pixels should be dark gray (no structural difference)
        for px in diff.pixels() {
            assert_eq!(*px, Rgba([24, 24, 24, 255]));
        }
    }

    #[test]
    fn structural_diff_detects_added_edge() {
        // Uniform expected, expected+edge actual
        let exp = RgbaImage::from_pixel(64, 64, Rgba([128, 128, 128, 255]));
        let mut act = exp.clone();
        // Add a bright horizontal line (simulates a watermark edge)
        for x in 10..54 {
            act.put_pixel(x, 32, Rgba([255, 255, 255, 255]));
        }
        let diff = generate_structural_diff(&exp, &act, 2, 10);
        // Should have non-dark pixels around y=32 (orange = added structure)
        let non_dark: usize = diff
            .pixels()
            .filter(|px| px[0] > 24 || px[1] > 24 || px[2] > 24)
            .count();
        assert!(
            non_dark > 10,
            "should detect added edge, got {non_dark} non-dark pixels"
        );
    }

    #[test]
    fn structural_diff_detects_missing_edge() {
        // Expected has edge, actual is uniform
        let mut exp = RgbaImage::from_pixel(64, 64, Rgba([128, 128, 128, 255]));
        for x in 10..54 {
            exp.put_pixel(x, 32, Rgba([255, 255, 255, 255]));
        }
        let act = RgbaImage::from_pixel(64, 64, Rgba([128, 128, 128, 255]));
        let diff = generate_structural_diff(&exp, &act, 2, 10);
        // Should have cyan pixels around y=32 (missing structure)
        let cyan_pixels: usize = diff
            .pixels()
            .filter(|px| px[1] > 50 && px[2] > 50 && px[0] < 10)
            .count();
        assert!(
            cyan_pixels > 5,
            "should show missing structure as cyan, got {cyan_pixels}"
        );
    }

    #[test]
    fn structural_montage_has_4_panels() {
        let exp = RgbaImage::from_pixel(32, 32, Rgba([100, 100, 100, 255]));
        let act = RgbaImage::from_pixel(32, 32, Rgba([110, 100, 90, 255]));
        let montage = create_structural_montage(&exp, &act, 10, 2, 3);
        // 4 panels of 32px + 3 gaps of 2px = 134
        assert_eq!(montage.width(), 32 * 4 + 2 * 3);
        assert_eq!(montage.height(), 32);
    }

    #[test]
    fn box_blur_identity_radius_zero() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = box_blur_gray(&data, 2, 2, 0);
        assert_eq!(result, data);
    }
}
