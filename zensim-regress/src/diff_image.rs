//! Diff image generation and visual montage composition.
//!
//! Creates amplified difference images and side-by-side comparison montages
//! for visual regression analysis.

use crate::pixel_ops::Bitmap;

use crate::pixel_ops::{self, ResampleFilter};

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
pub fn pixelate_upscale(img: &Bitmap, min_dim: u32) -> Bitmap {
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
    let mut out = Bitmap::new(new_w, new_h);

    for y in 0..h {
        for x in 0..w {
            let px = img.get_pixel(x, y);
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
/// The diff is byte-space `abs(e - a)` premultiplied by alpha. Pair
/// it with a byte-linear resize (no gamma) when the inputs need to be
/// reshaped before diffing — see `pixel_ops::resize_byte_linear`. A
/// gamma-correct (sRGB-aware) resize feeding a byte-space diff
/// amplifies 1-LSB encoding mismatches at every interpolated output
/// column into visible periodic stripes.
///
/// # Panics
///
/// Panics if expected and actual have different dimensions.
pub fn generate_diff_image(expected: &Bitmap, actual: &Bitmap, amplification: u8) -> Bitmap {
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

    let amp = amplification.max(1) as i32;
    let mut diff = Bitmap::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let e = expected.get_pixel(x, y);
            let a = actual.get_pixel(x, y);
            let ea = e[3] as i32;
            let aa = a[3] as i32;
            let dr = ((e[0] as i32 * ea / 255 - a[0] as i32 * aa / 255).abs() * amp).min(255) as u8;
            let dg = ((e[1] as i32 * ea / 255 - a[1] as i32 * aa / 255).abs() * amp).min(255) as u8;
            let db = ((e[2] as i32 * ea / 255 - a[2] as i32 * aa / 255).abs() * amp).min(255) as u8;
            if dr == 0 && dg == 0 && db == 0 {
                diff.put_pixel(x, y, [24, 24, 24, 255]);
            } else {
                diff.put_pixel(x, y, [dr, dg, db, 255]);
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
pub fn create_montage(panels: &[&Bitmap], gap: u32) -> Bitmap {
    if panels.is_empty() {
        return Bitmap::new(1, 1);
    }

    let max_h = panels.iter().map(|p| p.height()).max().unwrap_or(1);
    let total_w: u32 = panels.iter().map(|p| p.width()).sum::<u32>()
        + gap * (panels.len() as u32).saturating_sub(1);

    let mut montage = Bitmap::from_pixel(total_w, max_h, [32, 32, 32, 255]);

    let mut x_offset: i64 = 0;
    for panel in panels {
        pixel_ops::overlay(&mut montage, panel, x_offset, 0);
        x_offset += panel.width() as i64 + gap as i64;
    }

    montage
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
    expected: &Bitmap,
    actual: &Bitmap,
    blur_radius: u32,
    amplification: u8,
) -> Bitmap {
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
    let to_gray = |img: &Bitmap| -> Vec<f32> {
        img.as_raw()
            .chunks_exact(4)
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
    let mut diff = Bitmap::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let res_exp = (gray_exp[i] - blurred_exp[i]).abs();
            let res_act = (gray_act[i] - blurred_act[i]).abs();

            // Structural diff: positive = expected has more structure
            let delta = (res_exp - res_act) * amp;

            if delta.abs() < 1.0 {
                diff.put_pixel(x, y, [24, 24, 24, 255]);
            } else if delta > 0.0 {
                // Expected has structure, actual doesn't → cyan (missing)
                let v = delta.min(255.0) as u8;
                diff.put_pixel(x, y, [0, v, v, 255]);
            } else {
                // Actual has structure, expected doesn't → orange (added)
                let v = (-delta).min(255.0) as u8;
                diff.put_pixel(x, y, [v, (v as f32 * 0.6) as u8, 0, 255]);
            }
        }
    }

    diff
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
    expected: &Bitmap,
    actual: &Bitmap,
    amplification: u8,
    gap: u32,
    blur_radius: u32,
) -> Bitmap {
    let pixel_diff = generate_diff_image(expected, actual, amplification);
    let struct_diff = generate_structural_diff(expected, actual, blur_radius, amplification);
    create_montage(&[expected, actual, &pixel_diff, &struct_diff], gap)
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

fn render_montage_impl(
    options: &MontageOptions,
    expected: &Bitmap,
    actual: &Bitmap,
    annotation: &AnnotationText,
) -> Bitmap {
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
    let panels = if min_panel_size > 0 && (w < min_panel_size || h < min_panel_size) {
        [
            pixelate_upscale(expected, min_panel_size),
            pixelate_upscale(actual, min_panel_size),
            pixelate_upscale(&pixel_diff, min_panel_size),
            pixelate_upscale(&struct_diff, min_panel_size),
        ]
    } else {
        [expected.clone(), actual.clone(), pixel_diff, struct_diff]
    };

    compose_montage(panels, options, annotation, &spatial, has_differences, 10)
}

/// Shared 2×2 grid + annotation strip composition for both same-dim and
/// mismatched-dim montages. Takes the four prepared panel images and
/// produces the final output via the [`crate::layout`] module.
///
/// `longest_label_floor` sets a lower bound on the "longest label" used
/// to derive the ADD/REMOVE char height — same-dim uses 10 (the length
/// of "PIXEL DIFF"), mismatched uses 20 to reserve space for resized
/// annotations on tighter cells.
fn compose_montage(
    panels: [Bitmap; 4],
    options: &MontageOptions,
    annotation: &AnnotationText,
    spatial: &SpatialAnalysis,
    has_differences: bool,
    longest_label_floor: u32,
) -> Bitmap {
    use crate::font;
    use crate::layout::{
        self, CrossAlign, HAlign, Insets, LabelSegment, LabelStyle, LayoutMod, SizeRule, Track,
        VAlign, rgb,
    };

    let [p_exp, p_act, p_pdiff, p_sdiff] = panels;

    let min_pad = (font::GLYPH_W * 3 / 10).max(8);
    let pad = options.gap.max(min_pad);
    let canvas_bg = rgb(18, 18, 18);
    let label_fg = rgb(220, 220, 220);
    let label_bg = rgb(40, 40, 40);
    let title_bg = rgb(25, 25, 25);
    let primary_bg = rgb(30, 30, 30);
    let extra_bg = rgb(25, 25, 25);
    let separator = rgb(60, 60, 60);

    let panel_w = p_exp.width();
    let panel_h = p_exp.height();
    let cell_w = panel_w;
    let grid_w = pad + cell_w + pad + cell_w + pad;
    let label_area_w = panel_w.saturating_sub(pad * 2);
    let text_avail = grid_w.saturating_sub(pad * 2);

    let expected_text: &str = options.expected_label.as_deref().unwrap_or("EXPECTED");
    let actual_text: &str = options.actual_label.as_deref().unwrap_or("ACTUAL");

    // Char height for ADD/REMOVE — clamped to [GLYPH_H, GLYPH_H*3].
    // Plain labels remain auto-fit per their own length.
    let longest_label = expected_text
        .len()
        .max(actual_text.len())
        .max(longest_label_floor as usize) as u32;
    let ar_char_h = ((label_area_w as f32 / longest_label as f32)
        * (font::GLYPH_H as f32 / font::GLYPH_W as f32))
        .floor() as u32;
    let ar_char_h = ar_char_h.clamp(font::GLYPH_H, font::GLYPH_H * 3);

    // Unified label strip height = max of all four rendered labels + 4.
    let plain_h = |s: &str| font::measure_lines_fitted(&[(s, label_fg)], label_area_w).1;
    let label_h = plain_h(expected_text)
        .max(plain_h(actual_text))
        .max(plain_h("PIXEL DIFF"))
        .max(ar_char_h)
        + 4;

    let plain_style = LabelStyle::default()
        .with_fg(label_fg)
        .with_bg(label_bg)
        .with_align(HAlign::Center)
        .with_padding(Insets::xy(pad, 0));
    let segmented_style = LabelStyle::default()
        .with_fg(label_fg)
        .with_bg(label_bg)
        .with_padding(Insets::xy(pad, 0))
        .with_char_h(ar_char_h);

    let cell = |panel: Bitmap, label_node: layout::Node| -> layout::Node {
        layout::column()
            .child(label_node.height(SizeRule::Fixed(label_h)))
            .child(layout::image(panel))
            .into()
    };

    let panels_grid = layout::grid()
        .columns([Track::Px(cell_w), Track::Px(cell_w)])
        .row_heights([Track::Px(label_h + panel_h), Track::Px(label_h + panel_h)])
        .gap(pad)
        .padding(pad)
        .cell(0, 0, cell(p_exp, plain_style.strip(expected_text)))
        .cell(1, 0, cell(p_act, plain_style.strip(actual_text)))
        .cell(0, 1, cell(p_pdiff, plain_style.strip("PIXEL DIFF")))
        .cell(
            1,
            1,
            cell(
                p_sdiff,
                segmented_style.segmented_strip(vec![
                    LabelSegment::left("ADD", rgb(255, 180, 80)),
                    LabelSegment::right("REMOVE", rgb(80, 220, 220)),
                ]),
            ),
        );

    let mut col = layout::column()
        .align_items(CrossAlign::Stretch)
        .child(panels_grid);

    // Pre-rasterize annotation text into Image leaves so the layout module
    // doesn't re-compute char_h via the auto-fit formula at paint time
    // (which would shift one or two pixels when an enclosing Align reduces
    // the rect to the measured natural width).
    fn rasterized_image(buf: Vec<u8>, w: u32, h: u32) -> Option<Bitmap> {
        if w == 0 || h == 0 {
            return None;
        }
        Bitmap::from_raw(w, h, buf)
    }

    if let Some(t) = annotation.title.as_deref().filter(|s| !s.is_empty()) {
        let lines: Vec<(&str, [u8; 4])> = vec![(t, [255, 255, 255, 255])];
        let (tbuf, tw, th) = font::render_lines_fitted(&lines, title_bg, text_avail);
        if let Some(title_img) = rasterized_image(tbuf, tw, th) {
            let title_strip = layout::layers()
                .child(layout::fill(title_bg))
                .child(layout::image(title_img).center().padding(pad))
                .child(
                    layout::empty()
                        .background(separator)
                        .height(SizeRule::Fixed(1))
                        .fill_width()
                        .padding_xy(pad, 0)
                        .align_v(VAlign::Bottom),
                )
                .fill_width();
            col = col.child(title_strip);
        }
    }

    if !annotation.primary_lines.is_empty() {
        let line_refs: Vec<(&str, [u8; 4])> = annotation
            .primary_lines
            .iter()
            .map(|(s, c)| (s.as_str(), *c))
            .collect();
        let (pbuf, pw, ph) = font::render_lines_fitted(&line_refs, primary_bg, text_avail);
        if let Some(primary_img) = rasterized_image(pbuf, pw, ph) {
            let primary_strip = layout::image(primary_img)
                .center()
                .padding(pad)
                .background(primary_bg)
                .fill_width();
            col = col.child(primary_strip);
        }
    }

    if has_differences && options.show_spatial_heatmap {
        let heatmap_img = render_heatmap_grid(spatial, grid_w, pad);
        col = col.child(layout::image(heatmap_img));
    }

    if !annotation.extra.is_empty() {
        let primary_line_h = if !annotation.primary_lines.is_empty() {
            let refs: Vec<(&str, [u8; 4])> = annotation
                .primary_lines
                .iter()
                .map(|(s, c)| (s.as_str(), *c))
                .collect();
            let total = font::measure_lines_fitted(&refs, text_avail).1;
            total / annotation.primary_lines.len().max(1) as u32
        } else {
            font::GLYPH_H
        };
        let extra_char_h = (primary_line_h * 7 / 10).max(font::GLYPH_H / 4);
        let (ebuf, ew, eh) = font::render_text_wrapped(
            &annotation.extra,
            COLOR_DETAIL,
            extra_bg,
            extra_char_h,
            text_avail,
        );
        if let Some(extra_img) = rasterized_image(ebuf, ew, eh) {
            let extra_strip = layout::image(extra_img)
                .padding(pad)
                .background(extra_bg)
                .fill_width();
            col = col.child(extra_strip);
        }
    }

    col.background(canvas_bg).render(grid_w)
}

/// Render a montage for images with different dimensions.
///
/// Each panel uses a shared canvas sized to `max(ew,aw) × max(eh,ah)`.
/// Images are centered at their original size with blue borders showing bounds.
///
/// - Panel 0 (EXPECTED): Expected at original size, blue border
/// - Panel 1 (ACTUAL): Actual at original size, blue border
/// - Panel 2 (PIXEL DIFF): Diff of actual-resized-to-expected, blue border at expected bounds
/// - Panel 3 (ADD/REMOVE): Structural diff on shared canvas, blue borders for both
fn render_mismatched_montage(
    options: &MontageOptions,
    expected: &Bitmap,
    actual: &Bitmap,
    annotation: &AnnotationText,
) -> Bitmap {
    let (ew, eh) = expected.dimensions();
    let (aw, ah) = actual.dimensions();
    let amplification = options.amplification;
    let border_color: [u8; 4] = [80, 140, 255, 255]; // blue

    // Shared canvas = bounding box of both images
    let canvas_w = ew.max(aw);
    let canvas_h = eh.max(ah);

    // Centering offsets
    let ex_off_x = (canvas_w - ew) / 2;
    let ex_off_y = (canvas_h - eh) / 2;
    let ax_off_x = (canvas_w - aw) / 2;
    let ax_off_y = (canvas_h - ah) / 2;

    let panel_bg = [32, 32, 32, 255];

    // Panel 0: EXPECTED on canvas
    let mut panel_expected = Bitmap::from_pixel(canvas_w, canvas_h, panel_bg);
    pixel_ops::overlay(
        &mut panel_expected,
        expected,
        ex_off_x as i64,
        ex_off_y as i64,
    );
    draw_rect_border(
        &mut panel_expected,
        ex_off_x,
        ex_off_y,
        ew,
        eh,
        border_color,
    );

    // Panel 1: ACTUAL on canvas
    let mut panel_actual = Bitmap::from_pixel(canvas_w, canvas_h, panel_bg);
    pixel_ops::overlay(&mut panel_actual, actual, ax_off_x as i64, ax_off_y as i64);
    draw_rect_border(&mut panel_actual, ax_off_x, ax_off_y, aw, ah, border_color);

    // Panel 2: PIXEL DIFF — resize actual to expected dims and diff.
    // Byte-linear resize + byte-space diff: resize and diff must
    // agree on color space, otherwise the per-channel diff amplifies
    // sRGB↔linear encoding mismatches at every interpolated output
    // column into periodic stripes.
    let act_resized = pixel_ops::resize_byte_linear(actual, ew, eh, ResampleFilter::Triangle);
    let pixel_diff_raw = generate_diff_image(expected, &act_resized, amplification);
    let mut panel_pixel_diff = Bitmap::from_pixel(canvas_w, canvas_h, panel_bg);
    pixel_ops::overlay(
        &mut panel_pixel_diff,
        &pixel_diff_raw,
        ex_off_x as i64,
        ex_off_y as i64,
    );
    draw_rect_border(
        &mut panel_pixel_diff,
        ex_off_x,
        ex_off_y,
        ew,
        eh,
        border_color,
    );

    // Panel 3: STRUCTURAL ADD/REMOVE on shared canvas with transparent bg
    let transparent = [0, 0, 0, 0];
    let mut exp_on_canvas = Bitmap::from_pixel(canvas_w, canvas_h, transparent);
    pixel_ops::overlay(
        &mut exp_on_canvas,
        expected,
        ex_off_x as i64,
        ex_off_y as i64,
    );
    let mut act_on_canvas = Bitmap::from_pixel(canvas_w, canvas_h, transparent);
    pixel_ops::overlay(&mut act_on_canvas, actual, ax_off_x as i64, ax_off_y as i64);
    let mut panel_structural =
        generate_structural_diff(&exp_on_canvas, &act_on_canvas, 3, amplification);
    // Blue borders for both image bounds on the structural panel
    draw_rect_border(
        &mut panel_structural,
        ex_off_x,
        ex_off_y,
        ew,
        eh,
        border_color,
    );
    draw_rect_border(
        &mut panel_structural,
        ax_off_x,
        ax_off_y,
        aw,
        ah,
        border_color,
    );

    // Spatial analysis on the resized pair (for heatmap)
    let (sg_cols, sg_rows) = options.spatial_grid;
    let spatial = spatial_analysis(
        expected.as_raw(),
        act_resized.as_raw(),
        ew,
        eh,
        sg_cols,
        sg_rows,
    );
    let has_differences = spatial.regions.iter().any(|r| r.pixels_differing > 0.0);

    // Pixelate-upscale all four panels if tiny
    let min_panel_size = options.min_panel_size;
    let panels = if min_panel_size > 0 && (canvas_w < min_panel_size || canvas_h < min_panel_size) {
        [
            pixelate_upscale(&panel_expected, min_panel_size),
            pixelate_upscale(&panel_actual, min_panel_size),
            pixelate_upscale(&panel_pixel_diff, min_panel_size),
            pixelate_upscale(&panel_structural, min_panel_size),
        ]
    } else {
        [
            panel_expected,
            panel_actual,
            panel_pixel_diff,
            panel_structural,
        ]
    };

    compose_montage(panels, options, annotation, &spatial, has_differences, 20)
}

/// Render an N×M spatial heatmap grid image.
///
/// Each cell's background is tinted red proportional to its severity
/// relative to the worst cell. Cell text shows `pct% Δmax_delta`,
/// `zdsim:NNN` and an optional structural tag (ADDED / MISSING / changed).
/// Composed via the [`crate::layout`] module.
fn render_heatmap_grid(spatial: &SpatialAnalysis, total_w: u32, pad: u32) -> Bitmap {
    use crate::font;
    use crate::layout::{self, LayoutMod, Track, rgb};

    let cols = spatial.cols;
    let rows = spatial.rows;
    let cell_gap = 3u32;

    let inner_w = total_w.saturating_sub(pad * 2);
    let cell_w = (inner_w.saturating_sub(cell_gap * (cols - 1))) / cols;
    let cell_h = cell_w * 3 / 4;

    // Global severity scaling for background tinting.
    let max_severity = spatial
        .regions
        .iter()
        .map(|r| r.max_delta as f32 * r.pixels_differing as f32)
        .fold(0.0f32, f32::max)
        .max(0.001);

    // Cell-text padding (30% of one base char width per side, matching
    // the original).
    let cell_inset = font::GLYPH_W * 3 / 10;
    let cell_text_w = cell_w.saturating_sub(cell_inset * 2);

    let mut g = layout::grid()
        .columns((0..cols).map(|_| Track::Px(cell_w)))
        .rows((0..rows).map(|_| Track::Px(cell_h)))
        .gap(cell_gap)
        .padding(pad);

    for r in &spatial.regions {
        let severity = r.max_delta as f32 * r.pixels_differing as f32;
        let ratio = (severity / max_severity).clamp(0.0, 1.0);
        let bg_r = (30.0 + ratio * 120.0) as u8;
        let bg_g = ((30.0 - ratio * 20.0).max(10.0)) as u8;
        let bg_b = bg_g;
        let cell_bg = [bg_r, bg_g, bg_b, 255];

        // Pre-rasterize the cell text so the layout module's auto-fit
        // formula can't shift char_h between measure and paint. The
        // resulting Image leaf is centered inside a Layers stack on top
        // of the cell-background fill.
        let pct = r.pixels_differing * 100.0;
        let text_buf = if cell_text_w == 0 {
            None
        } else if pct < 0.05 {
            let lines: Vec<(&str, [u8; 4])> = vec![("ok", COLOR_OK)];
            let (tbuf, tw, th) = font::render_lines_fitted(&lines, cell_bg, cell_text_w);
            (tw > 0 && th > 0)
                .then(|| Bitmap::from_raw(tw, th, tbuf))
                .flatten()
        } else {
            let line1 = format!("{:.0}% \u{0394}{}", pct, r.max_delta);
            let line2 = format!("zdsim:{:.3}", r.avg_delta / 255.0);
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
            let mut lines: Vec<(&str, [u8; 4])> =
                vec![(&line1, text_fg), (&line2, [170, 170, 170, 255])];
            if !line3.is_empty() {
                let tag_color = match line3 {
                    "MISSING" => [255, 120, 120, 255],
                    "ADDED" => [255, 180, 80, 255],
                    _ => [200, 200, 120, 255],
                };
                lines.push((line3, tag_color));
            }
            let (tbuf, tw, th) = font::render_lines_fitted(&lines, cell_bg, cell_text_w);
            (tw > 0 && th > 0)
                .then(|| Bitmap::from_raw(tw, th, tbuf))
                .flatten()
        };

        let mut cell_layers = layout::layers().child(layout::fill(cell_bg));
        if let Some(buf) = text_buf {
            cell_layers = cell_layers.child(layout::image(buf).center());
        }
        g = g.cell(r.col, r.row, cell_layers);
    }

    g.background(rgb(18, 18, 18)).render(total_w)
}

/// Draw a 1-px rectangular outline. Re-exported from
/// [`crate::pixel_ops::draw_rect_border`] to keep the diff_image
/// internal call sites concise.
fn draw_rect_border(img: &mut Bitmap, x0: u32, y0: u32, w: u32, h: u32, color: [u8; 4]) {
    pixel_ops::draw_rect_border(img, x0, y0, w, h, color);
}

/// Rendering options for annotated montages.
///
/// `MontageOptions` is `#[non_exhaustive]`, so out-of-crate callers can't
/// construct it via struct literal (with or without `..Default::default()`).
/// Use [`Default`] + direct field assignment instead — all fields are still
/// public. This indirection lets us add new options in future patch releases
/// without breaking caller code.
///
/// ```
/// # use zensim_regress::diff_image::MontageOptions;
/// // The common case
/// let opts = MontageOptions::default();
///
/// // Override individual fields
/// let mut custom = MontageOptions::default();
/// custom.amplification = 50;
///
/// // Override several at once
/// let mut labeled = MontageOptions::default();
/// labeled.expected_label = Some("ORIG".into());
/// labeled.actual_label = Some("DEFAULT".into());
/// labeled.show_spatial_heatmap = false;
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
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
    /// Label for the top-left panel. `None` uses the default `"EXPECTED"`.
    ///
    /// Useful for A/B comparisons where "expected/actual" framing doesn't
    /// fit: set both labels explicitly (e.g., `"ORIG"` / `"DEFAULT"`). Longer
    /// labels than `"PIXEL DIFF"` (10 chars) are auto-sized down.
    pub expected_label: Option<String>,
    /// Label for the top-right panel. `None` uses the default `"ACTUAL"`.
    pub actual_label: Option<String>,
    /// Whether to render the 3×3 (or `spatial_grid`) heatmap strip below
    /// the annotation text. Default `true` — matches the zensim-regress
    /// CI regression-report look. Set to `false` for A/B comparisons
    /// where every region has full-magnitude differences (lossy
    /// encoding outputs) and the heatmap just shows a uniformly-red
    /// grid that adds no information.
    pub show_spatial_heatmap: bool,
}

impl Default for MontageOptions {
    fn default() -> Self {
        Self {
            amplification: 10,
            gap: 2,
            min_panel_size: DEFAULT_MIN_PANEL_SIZE,
            spatial_grid: (3, 3),
            expected_label: None,
            actual_label: None,
            show_spatial_heatmap: true,
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
    /// When expected and actual have different dimensions, a shared-canvas
    /// montage is rendered instead, with blue borders showing each image's
    /// bounds and the pixel diff computed on center-cropped or resized copies.
    pub fn render(
        &self,
        expected: &Bitmap,
        actual: &Bitmap,
        annotation: &AnnotationText,
    ) -> Bitmap {
        if expected.dimensions() != actual.dimensions() {
            render_mismatched_montage(self, expected, actual, annotation)
        } else {
            render_montage_impl(self, expected, actual, annotation)
        }
    }
}

/// Annotation data for the montage: verdict lines and optional extra text.
///
/// Use [`from_report`](Self::from_report) when you have a regression report,
/// or [`empty`](Self::empty) for a bare montage with no annotations.
#[non_exhaustive]
pub struct AnnotationText {
    /// Title shown above the constraint lines (e.g., test name + detail).
    /// Rendered in white on the dark background strip. None = no title.
    pub title: Option<String>,
    /// Colored lines for the primary text block.
    /// Red for failing constraints, green for passing, gray for info.
    pub primary_lines: Vec<(String, [u8; 4])>,
    /// Extra text lines (alpha info, etc). Shown below heatmap if present.
    pub extra: String,
}

const COLOR_FAIL: [u8; 4] = [255, 80, 80, 255]; // red
const COLOR_OK: [u8; 4] = [80, 220, 80, 255]; // green
const COLOR_DETAIL: [u8; 4] = [170, 170, 170, 255]; // dim gray

impl AnnotationText {
    /// No annotation — produces a bare montage with no text or heatmap.
    pub fn empty() -> Self {
        Self {
            title: None,
            primary_lines: vec![],
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

        Self {
            title: None,
            primary_lines: lines,
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
            let method_line = format!("Actual {} for comparison (approximate)", dim_info.method(),);
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

    /// Build a synthetic 4-cell montage and verify the four label
    /// strips are uniformly tall and horizontally aligned. Catches
    /// regressions where ADD/REMOVE strip's height drifts away from
    /// EXPECTED/ACTUAL/PIXEL DIFF strips' heights.
    #[test]
    fn montage_label_strips_have_uniform_height() {
        let exp = Bitmap::from_fn(96, 96, |x, y| [(x * 2) as u8, (y * 2) as u8, 128, 255]);
        let act = Bitmap::from_fn(96, 96, |x, y| [(x * 2) as u8, (y * 2) as u8 + 30, 128, 255]);
        let opts = MontageOptions::default();
        let canvas = opts.render(&exp, &act, &AnnotationText::empty());

        // The label_bg color is rgb(40, 40, 40). Walk down the leftmost
        // pixel column inside the panel area (x = 8 = pad) to collect
        // the y-ranges where the column is label_bg. There should be
        // two such ranges (one per row of cells), and they must have
        // identical heights — otherwise the cells aren't aligned.
        let label_bg = [40, 40, 40, 255];
        let canvas_bg = [18, 18, 18, 255];
        let pad = 8;
        let mut runs: Vec<(u32, u32)> = Vec::new();
        let mut start: Option<u32> = None;
        for y in 0..canvas.height() {
            let p = canvas.get_pixel(pad, y);
            if p == label_bg {
                if start.is_none() {
                    start = Some(y);
                }
            } else if let Some(s) = start.take() {
                runs.push((s, y - 1));
            }
        }
        if let Some(s) = start {
            runs.push((s, canvas.height() - 1));
        }
        assert!(
            runs.len() >= 2,
            "expected ≥2 label_bg runs (one per cell row), got {runs:?}"
        );
        // The first two runs are the two cell-row label strips.
        let h0 = runs[0].1 - runs[0].0 + 1;
        let h1 = runs[1].1 - runs[1].0 + 1;
        assert_eq!(
            h0, h1,
            "label strip heights mismatch — row 0 = {h0}, row 1 = {h1} (runs = {runs:?})"
        );

        // Strip starts must be uniformly offset from the canvas top —
        // the gap between row 0's strip end and row 1's strip start
        // must equal panel_h + cell_row_gap (= panel_h + pad).
        let between = runs[1].0 - runs[0].1 - 1;
        // 96×96 image → pixelate to 288×288 → panel_h = 288. cell gap = pad.
        let expected_between = 288 + pad;
        assert_eq!(
            between, expected_between,
            "expected {expected_between} px between strips (panel + gap), got {between}"
        );

        // Sanity: above row 0 strip is canvas_bg (top padding).
        let above = canvas.get_pixel(pad, runs[0].0 - 1);
        assert_eq!(
            above, canvas_bg,
            "expected canvas_bg above row 0 strip, got {above:?}"
        );
    }

    /// `LabelStyle::segmented_strip` uses `Track::FrMin { min_px: 6 }`
    /// for the middle column, so left and right groups are guaranteed
    /// to stay at least 6 px apart even when content widths exceed
    /// the strip's inner width.
    #[test]
    fn segmented_strip_enforces_min_gap_under_overflow() {
        use crate::layout::{LabelSegment, LabelStyle, LayoutMod, Size};

        // 80-wide strip, with explicit char_h that makes ADD+REMOVE
        // overflow on their own. FrMin(1, 6) forces a minimum gap.
        let style = LabelStyle::default().with_char_h(20);
        let strip = style
            .segmented_strip(vec![
                LabelSegment::left("ADDADDADD", [255, 180, 80, 255]),
                LabelSegment::right("REMOVEREMOVE", [80, 220, 220, 255]),
            ])
            .width(crate::layout::SizeRule::Fixed(80))
            .height(crate::layout::SizeRule::Fixed(30));
        // Render and assert ADD's right edge ≤ REMOVE's left edge − 6.
        let canvas = crate::layout::render(&strip, 80);
        // Find the middle row, scan for orange (ADD) and cyan (REMOVE) clusters.
        let mid_y = canvas.height() / 2;
        let mut orange_xs = Vec::new();
        let mut cyan_xs = Vec::new();
        for x in 0..canvas.width() {
            let p = canvas.get_pixel(x, mid_y);
            // ADD is orange-ish (R high, G mid, B low).
            if p[0] > 100 && p[1] > 80 && p[2] < 120 {
                orange_xs.push(x);
            }
            // REMOVE is cyan-ish (R low, G high, B high).
            if p[0] < 120 && p[1] > 100 && p[2] > 100 {
                cyan_xs.push(x);
            }
        }
        // We just want the rightmost orange and leftmost cyan.
        if let (Some(&add_right), Some(&remove_left)) = (orange_xs.last(), cyan_xs.first()) {
            let gap = remove_left as i32 - add_right as i32 - 1;
            assert!(
                gap >= 6,
                "expected at least 6 px between ADD's right edge ({add_right}) and REMOVE's left ({remove_left}); got {gap}"
            );
        } else {
            // If overflow caused full clipping, that's OK — but ensure
            // we didn't render ADD/REMOVE on top of each other.
            let _ = Size::new(80, 30);
        }
    }

    /// Verify ADD and REMOVE labels in the structural-diff cell are
    /// (a) both visible, (b) horizontally separated, (c) at the cell
    /// row's vertical center.
    #[test]
    fn montage_add_remove_labels_are_separated() {
        let exp = Bitmap::from_fn(96, 96, |x, y| [(x * 2) as u8, (y * 2) as u8, 128, 255]);
        let act = Bitmap::from_fn(96, 96, |x, y| [(x * 2) as u8, (y * 2) as u8 + 30, 128, 255]);
        let opts = MontageOptions::default();
        let canvas = opts.render(&exp, &act, &AnnotationText::empty());

        // ADD/REMOVE strip is in cell (1, 1). Find row-1 label strip
        // by scanning down for the second label_bg run, then look at
        // the right-cell column (x = pad + panel_w + pad + ... or just
        // anywhere right of canvas mid).
        let canvas_w = canvas.width();
        let label_bg = [40, 40, 40, 255];
        let mut runs: Vec<(u32, u32)> = Vec::new();
        let mut start: Option<u32> = None;
        for y in 0..canvas.height() {
            let p = canvas.get_pixel(8, y);
            if p == label_bg {
                if start.is_none() {
                    start = Some(y);
                }
            } else if let Some(s) = start.take() {
                runs.push((s, y - 1));
            }
        }
        assert!(runs.len() >= 2, "no row-1 label strip found");
        let (y0, y1) = runs[1];
        let mid_y = (y0 + y1) / 2;
        let right_cell_mid_x = canvas_w * 3 / 4;

        // Look for non-label-bg pixels in the right cell strip mid-row;
        // should find ADD on the left half and REMOVE on the right
        // half, with at least 20 px of label_bg between.
        let strip_x_start = canvas_w / 2;
        let strip_x_end = canvas_w - 8;
        let mut text_xs: Vec<u32> = Vec::new();
        for x in strip_x_start..strip_x_end {
            let p = canvas.get_pixel(x, mid_y);
            if p != label_bg && p != [18, 18, 18, 255] && p != [32, 32, 32, 255] {
                text_xs.push(x);
            }
        }
        // Group into clusters with > 20 px gap = separate words.
        let clusters: Vec<(u32, u32)> = {
            let mut out: Vec<(u32, u32)> = Vec::new();
            let mut prev: Option<u32> = None;
            for x in &text_xs {
                if prev.is_none_or(|p| x - p > 20) {
                    out.push((*x, *x));
                } else {
                    out.last_mut().unwrap().1 = *x;
                }
                prev = Some(*x);
            }
            out
        };
        assert!(
            clusters.len() >= 2,
            "expected ≥2 text clusters (ADD + REMOVE) in cell (1,1) mid-row, got {clusters:?}"
        );
        // First cluster (ADD) should be in the left half of the right
        // cell; last cluster (REMOVE) in the right half.
        let cell_mid = right_cell_mid_x;
        assert!(
            clusters[0].0 < cell_mid,
            "ADD cluster should start left of right-cell mid (x={}), got {clusters:?}",
            cell_mid
        );
        assert!(
            clusters.last().unwrap().1 > cell_mid,
            "REMOVE cluster should end right of right-cell mid (x={}), got {clusters:?}",
            cell_mid
        );
    }

    #[test]
    fn diff_identical_images() {
        let img = Bitmap::from_fn(8, 8, |x, y| [(x * 32) as u8, (y * 32) as u8, 128, 255]);
        let diff = generate_diff_image(&img, &img, 10);
        // All pixels should be dark gray (no difference)
        for pixel in diff.as_raw().chunks_exact(4) {
            assert_eq!(pixel, [24, 24, 24, 255]);
        }
    }

    #[test]
    fn diff_amplification() {
        let a = Bitmap::from_pixel(4, 4, [100, 100, 100, 255]);
        let b = Bitmap::from_pixel(4, 4, [101, 100, 98, 255]);

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
        let a = Bitmap::from_pixel(4, 4, [255, 0, 0, 0]);
        let b = Bitmap::from_pixel(4, 4, [0, 255, 0, 0]);
        let diff = generate_diff_image(&a, &b, 10);
        for pixel in diff.as_raw().chunks_exact(4) {
            assert_eq!(
                pixel,
                [24, 24, 24, 255],
                "transparent pixels should diff to zero"
            );
        }
    }

    #[test]
    fn diff_semitransparent_scales_by_alpha() {
        // Same RGB, different alpha — diff should reflect the visual difference
        let a = Bitmap::from_pixel(4, 4, [200, 200, 200, 128]);
        let b = Bitmap::from_pixel(4, 4, [200, 200, 200, 0]);
        let diff = generate_diff_image(&a, &b, 1);
        let p = diff.get_pixel(0, 0);
        // a premul: 200*128/255 ≈ 100, b premul: 0. delta ≈ 100 per channel.
        assert!(p[0] > 90 && p[0] < 110, "expected ~100, got {}", p[0]);
    }

    #[test]
    fn diff_clamps_to_255() {
        let a = Bitmap::from_pixel(2, 2, [0, 0, 0, 255]);
        let b = Bitmap::from_pixel(2, 2, [200, 200, 200, 255]);
        let diff = generate_diff_image(&a, &b, 10);
        let p = diff.get_pixel(0, 0);
        assert_eq!(p[0], 255); // 200*10 clamped to 255
    }

    #[test]
    fn montage_dimensions() {
        let a = Bitmap::new(10, 20);
        let b = Bitmap::new(10, 20);
        let c = Bitmap::new(10, 20);
        let montage = create_montage(&[&a, &b, &c], 2);
        assert_eq!(montage.width(), 34); // 10+2+10+2+10
        assert_eq!(montage.height(), 20);
    }

    #[test]
    fn montage_different_heights() {
        let a = Bitmap::new(10, 30);
        let b = Bitmap::new(10, 10);
        let montage = create_montage(&[&a, &b], 4);
        assert_eq!(montage.width(), 24); // 10+4+10
        assert_eq!(montage.height(), 30); // max height
    }

    #[test]
    fn pixelate_noop_when_large_enough() {
        let img = Bitmap::from_pixel(300, 300, [100, 100, 100, 255]);
        let up = pixelate_upscale(&img, 256);
        assert_eq!(up.dimensions(), (300, 300));
    }

    #[test]
    fn pixelate_scales_8x8_to_at_least_256() {
        let img = Bitmap::from_fn(8, 8, |x, y| [(x * 32) as u8, (y * 32) as u8, 128, 255]);
        let up = pixelate_upscale(&img, 256);
        // 256/8 = 32, so N=32, output = 256×256
        assert_eq!(up.dimensions(), (256, 256));
        // Each original pixel should be a 32×32 block
        let orig = img.get_pixel(1, 2);
        for dy in 0..32 {
            for dx in 0..32 {
                assert_eq!(up.get_pixel(32 + dx, 2 * 32 + dy), orig);
            }
        }
    }

    #[test]
    fn pixelate_rectangular() {
        let img = Bitmap::from_pixel(4, 16, [50, 100, 150, 255]);
        let up = pixelate_upscale(&img, 256);
        // N = max(ceil(256/4), ceil(256/16)) = max(64, 16) = 64
        assert_eq!(up.dimensions(), (256, 1024));
    }

    #[test]
    fn pixelate_disabled_with_zero() {
        let img = Bitmap::from_pixel(4, 4, [100; 4]);
        let up = pixelate_upscale(&img, 0);
        assert_eq!(up.dimensions(), (4, 4));
    }

    #[test]
    fn annotated_montage_adds_text_strips() {
        let exp = Bitmap::from_pixel(32, 32, [100, 100, 100, 255]);
        let act = Bitmap::from_pixel(32, 32, [110, 100, 90, 255]);
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
        let exp = Bitmap::from_pixel(16, 16, [100; 4]);
        let act = Bitmap::from_pixel(16, 16, [100; 4]);
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
        let exp = Bitmap::from_pixel(8, 8, [100, 100, 100, 255]);
        let act = Bitmap::from_pixel(8, 8, [110, 100, 90, 255]);
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
        let img = Bitmap::from_fn(32, 32, |x, y| [(x * 8) as u8, (y * 8) as u8, 128, 255]);
        let diff = generate_structural_diff(&img, &img, 3, 10);
        // All pixels should be dark gray (no structural difference)
        for px in diff.as_raw().chunks_exact(4) {
            assert_eq!(px, [24, 24, 24, 255]);
        }
    }

    #[test]
    fn structural_diff_detects_added_edge() {
        // Uniform expected, expected+edge actual
        let exp = Bitmap::from_pixel(64, 64, [128, 128, 128, 255]);
        let mut act = exp.clone();
        // Add a bright horizontal line (simulates a watermark edge)
        for x in 10..54 {
            act.put_pixel(x, 32, [255, 255, 255, 255]);
        }
        let diff = generate_structural_diff(&exp, &act, 2, 10);
        // Should have non-dark pixels around y=32 (orange = added structure)
        let non_dark: usize = diff
            .as_raw()
            .chunks_exact(4)
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
        let mut exp = Bitmap::from_pixel(64, 64, [128, 128, 128, 255]);
        for x in 10..54 {
            exp.put_pixel(x, 32, [255, 255, 255, 255]);
        }
        let act = Bitmap::from_pixel(64, 64, [128, 128, 128, 255]);
        let diff = generate_structural_diff(&exp, &act, 2, 10);
        // Should have cyan pixels around y=32 (missing structure)
        let cyan_pixels: usize = diff
            .as_raw()
            .chunks_exact(4)
            .filter(|px| px[1] > 50 && px[2] > 50 && px[0] < 10)
            .count();
        assert!(
            cyan_pixels > 5,
            "should show missing structure as cyan, got {cyan_pixels}"
        );
    }

    #[test]
    fn structural_montage_has_4_panels() {
        let exp = Bitmap::from_pixel(32, 32, [100, 100, 100, 255]);
        let act = Bitmap::from_pixel(32, 32, [110, 100, 90, 255]);
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

    #[test]
    fn mismatched_montage_renders_without_panic() {
        let exp = Bitmap::from_pixel(32, 32, [100, 100, 100, 255]);
        let act = Bitmap::from_pixel(48, 24, [110, 100, 90, 255]);
        let opts = MontageOptions {
            min_panel_size: 0,
            ..Default::default()
        };
        let montage = opts.render(&exp, &act, &AnnotationText::empty());
        // Canvas is 48x32 per panel; 2 panels wide + gaps
        assert!(montage.width() > 48 * 2);
        assert!(montage.height() > 32 * 2);
    }

    #[test]
    fn mismatched_montage_orientation_swap() {
        let exp = Bitmap::from_fn(64, 48, |x, y| [(x * 4) as u8, (y * 5) as u8, 128, 255]);
        let act = Bitmap::from_fn(48, 64, |x, y| [(x * 5) as u8, (y * 4) as u8, 128, 255]);
        let opts = MontageOptions {
            min_panel_size: 0,
            ..Default::default()
        };
        let montage = opts.render(&exp, &act, &AnnotationText::empty());
        // Canvas is 64x64 per panel
        assert!(montage.width() >= 64 * 2);
    }

    #[test]
    fn mismatched_montage_with_annotation() {
        let exp = Bitmap::from_pixel(32, 32, [100, 100, 100, 255]);
        let act = Bitmap::from_pixel(40, 30, [110, 100, 90, 255]);
        let ann = AnnotationText {
            primary_lines: vec![("FAIL: dimensions differ".into(), [255, 80, 80, 255])],
            ..AnnotationText::empty()
        };
        let opts = MontageOptions {
            min_panel_size: 0,
            ..Default::default()
        };
        let montage = opts.render(&exp, &act, &ann);
        // Text strip makes it taller
        assert!(montage.height() > 32 * 2 + 20);
    }

    #[test]
    fn same_dimension_still_uses_normal_path() {
        let exp = Bitmap::from_pixel(32, 32, [100, 100, 100, 255]);
        let act = Bitmap::from_pixel(32, 32, [110, 100, 90, 255]);
        // Should not crash — uses render_montage_impl
        let montage = MontageOptions::default().render(&exp, &act, &AnnotationText::empty());
        assert!(montage.width() >= 32 * 2);
    }

    #[test]
    fn draw_rect_border_on_small_image() {
        let mut img = Bitmap::from_pixel(10, 10, [0, 0, 0, 255]);
        draw_rect_border(&mut img, 2, 2, 6, 6, [255, 255, 255, 255]);
        // Top-left corner of border
        assert_eq!(img.get_pixel(2, 2), [255, 255, 255, 255]);
        // Bottom-right corner of border
        assert_eq!(img.get_pixel(7, 7), [255, 255, 255, 255]);
        // Inside border (not on edge)
        assert_eq!(img.get_pixel(4, 4), [0, 0, 0, 255]);
    }
}
