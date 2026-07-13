//! SDF glyph-strip builder — the `sdf-font` feature's replacement for
//! the per-cell zenresize scaling path in [`crate::font`].
//!
//! When the feature is enabled, `font::cached_scaled_strip` calls
//! [`build_scaled_strip_sdf`] instead of Mitchell-resampling the
//! embedded 54px coverage strip. Everything downstream — composition,
//! wrapping, fitting, the gamma-correct blend — is unchanged: this
//! module only produces the same `(char_w * CHAR_COUNT) × char_h` RGBA
//! strip (RGB = 255, alpha = linear coverage) from a signed-distance
//! field instead of a resampled bitmap.
//!
//! The embedded atlas (`sdf_atlas.bin`, 16.9 KB) stores 96 glyph cells
//! of 13×27 texels at 4 bits each: exact-EDT signed distance, spread
//! ±4 texels, baked from `font_strip.png` by
//! `benchmarks/bake_sdf_atlas_2026-07-13.py`. Method, measurements, and
//! the c=0.2 small-size weight compensation are documented in
//! `benchmarks/sdf_font_atlas_exploration_2026-07-13.md`.
//!
//! Prototype status: glyph coverage and metrics identical to the
//! bitmap strip (ASCII 32–126 + Δ, monospace). The production bake
//! regenerates from an OFL face's vector outlines and extends coverage
//! per the measured charset tiers.

use std::sync::OnceLock;

use crate::pixel_ops::Bitmap;

/// Small-size weight compensation, in texels at `k = 0` — dilates the
/// coverage threshold by `WEIGHT_COMP * (1 - k)` when minifying so
/// thin strokes keep engine-equivalent ink (measured: ±3% of the
/// Mitchell path's linear ink at 12–18px; without it, −8%).
const WEIGHT_COMP: f32 = 0.2;

/// `<u32 cell_w><u32 cell_h><u32 n_glyphs><f32 spread><u32 per_glyph>`.
const HEADER_LEN: usize = 20;

static SDF_ATLAS_BIN: &[u8] = include_bytes!("sdf_atlas.bin");

struct Atlas {
    cell_w: u32,
    cell_h: u32,
    n_glyphs: u32,
    spread: f32,
    /// Unpacked to 8-bit (nibble × 17), glyph-major, row-major.
    data: Vec<u8>,
}

fn atlas() -> &'static Atlas {
    static ATLAS: OnceLock<Atlas> = OnceLock::new();
    ATLAS.get_or_init(|| {
        let b = SDF_ATLAS_BIN;
        assert!(b.len() > HEADER_LEN, "sdf_atlas.bin truncated");
        let u = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        let (cell_w, cell_h, n_glyphs) = (u(0), u(4), u(8));
        let spread = f32::from_le_bytes(b[12..16].try_into().unwrap());
        let per_glyph = u(16) as usize;
        let texels = (cell_w * cell_h) as usize;
        assert_eq!(per_glyph, texels.div_ceil(2), "per-glyph stride mismatch");
        assert_eq!(
            b.len() - HEADER_LEN,
            per_glyph * n_glyphs as usize,
            "sdf_atlas.bin payload size mismatch"
        );
        assert!(spread > 0.0 && spread.is_finite(), "bad spread");

        let mut data = Vec::with_capacity(texels * n_glyphs as usize);
        for g in 0..n_glyphs as usize {
            let packed = &b[HEADER_LEN + g * per_glyph..HEADER_LEN + (g + 1) * per_glyph];
            for (i, byte) in packed.iter().enumerate() {
                data.push((byte >> 4) * 17);
                if i * 2 + 1 < texels {
                    data.push((byte & 0x0F) * 17);
                }
            }
        }
        Atlas {
            cell_w,
            cell_h,
            n_glyphs,
            spread,
            data,
        }
    })
}

/// Precomputed 1-D bilinear taps for one output axis.
struct AxisTaps {
    i0: Vec<u32>,
    i1: Vec<u32>,
    frac: Vec<f32>,
}

fn axis_taps(out_len: u32, in_len: u32) -> AxisTaps {
    let inv = in_len as f32 / out_len as f32;
    let mut i0 = Vec::with_capacity(out_len as usize);
    let mut i1 = Vec::with_capacity(out_len as usize);
    let mut frac = Vec::with_capacity(out_len as usize);
    for o in 0..out_len {
        let s = ((o as f32 + 0.5) * inv - 0.5).clamp(0.0, (in_len - 1) as f32);
        let a = s as u32;
        i0.push(a);
        i1.push((a + 1).min(in_len - 1));
        frac.push(s - a as f32);
    }
    AxisTaps { i0, i1, frac }
}

/// Full-strip builder retained for the in-module tests.
#[cfg(test)]
pub(crate) fn build_scaled_strip_sdf(scaled_char_w: u32, scaled_char_h: u32) -> Bitmap {
    build_scaled_run_sdf(0, atlas().n_glyphs, scaled_char_w, scaled_char_h)
}

/// Build the scaled cells for glyph indices `[start, start+count)`
/// from the SDF atlas: dimensions and pixel layout identical to
/// `font::build_scaled_run_per_cell`'s output — `(char_w * count) ×
/// char_h` RGBA, RGB = 255, alpha = linear coverage.
pub(crate) fn build_scaled_run_sdf(
    start: u32,
    count: u32,
    scaled_char_w: u32,
    scaled_char_h: u32,
) -> Bitmap {
    let at = atlas();
    let strip_w = scaled_char_w * count.max(1);
    if scaled_char_w == 0 || scaled_char_h == 0 || count == 0 {
        return Bitmap::new(strip_w.max(1), scaled_char_h.max(1));
    }

    // Vertical scale defines the coverage band; the horizontal scale
    // matches it to within the caller's aspect rounding.
    let k = scaled_char_h as f32 / at.cell_h as f32;
    let shift = WEIGHT_COMP * (1.0 - k).max(0.0);
    let d_scale = 2.0 * at.spread / 255.0;

    let xt = axis_taps(scaled_char_w, at.cell_w);
    let yt = axis_taps(scaled_char_h, at.cell_h);

    let cell = (at.cell_w * at.cell_h) as usize;
    let mut out = vec![0u8; (strip_w as usize) * (scaled_char_h as usize) * 4];
    let row_stride = (strip_w as usize) * 4;

    for i in 0..count as usize {
        let glyph = (start as usize + i).min(at.n_glyphs as usize - 1);
        let base = glyph * cell;
        let x_off = i * scaled_char_w as usize * 4;
        for y in 0..scaled_char_h as usize {
            let r0 = base + (yt.i0[y] * at.cell_w) as usize;
            let r1 = base + (yt.i1[y] * at.cell_w) as usize;
            let fy = yt.frac[y];
            let row = &mut out
                [y * row_stride + x_off..y * row_stride + x_off + scaled_char_w as usize * 4];
            for (x, px) in row.chunks_exact_mut(4).enumerate() {
                let (x0, x1, fx) = (xt.i0[x] as usize, xt.i1[x] as usize, xt.frac[x]);
                let top = {
                    let a = at.data[r0 + x0] as f32;
                    let b = at.data[r0 + x1] as f32;
                    a + (b - a) * fx
                };
                let bot = {
                    let a = at.data[r1 + x0] as f32;
                    let b = at.data[r1 + x1] as f32;
                    a + (b - a) * fx
                };
                let q = top + (bot - top) * fy;
                let d = q * d_scale - at.spread; // texels, positive inside
                let cov = (0.5 + (d + shift) * k).clamp(0.0, 1.0);
                px.copy_from_slice(&[255, 255, 255, (cov * 255.0 + 0.5) as u8]);
            }
        }
    }

    Bitmap::from_raw(strip_w, scaled_char_h, out).expect("sdf strip dims match buffer")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_parses_and_matches_font_geometry() {
        let a = atlas();
        assert_eq!((a.cell_w, a.cell_h, a.n_glyphs), (13, 27, 96));
        assert_eq!(a.spread, 4.0);
        assert_eq!(a.data.len(), 13 * 27 * 96);
    }

    fn glyph_ink_fraction(strip: &Bitmap, glyph: u32, char_w: u32) -> f32 {
        let mut ink = 0u64;
        for y in 0..strip.height() {
            for x in 0..char_w {
                ink += strip.get_pixel(glyph * char_w + x, y)[3] as u64;
            }
        }
        ink as f32 / (char_w * strip.height() * 255) as f32
    }

    #[test]
    fn strip_dimensions_match_bitmap_contract() {
        let s = build_scaled_strip_sdf(6, 12);
        assert_eq!((s.width(), s.height()), (6 * 96, 12));
    }

    #[test]
    fn r_has_ink_and_space_is_empty_across_sizes() {
        for char_h in [8u32, 12, 27, 54, 96] {
            let char_w = (26 * char_h + 27) / 54;
            let s = build_scaled_strip_sdf(char_w, char_h);
            let r = glyph_ink_fraction(&s, ('R' as u32) - 0x20, char_w);
            let sp = glyph_ink_fraction(&s, 0, char_w);
            assert!(r > 0.10 && r < 0.60, "R ink {r} out of range at {char_h}px");
            assert!(sp < 0.005, "space ink {sp} at {char_h}px");
        }
    }

    #[test]
    fn ink_is_size_stable_with_weight_compensation() {
        // Mean coverage of 'R' should not collapse at small sizes
        // (the c=0.2 compensation's whole job). Allow generous slack —
        // this guards against regressions, not exact calibration.
        let ink_at = |char_h: u32| {
            let char_w = (26 * char_h + 27) / 54;
            let s = build_scaled_strip_sdf(char_w, char_h);
            glyph_ink_fraction(&s, ('R' as u32) - 0x20, char_w)
        };
        let (i12, i27) = (ink_at(12), ink_at(27));
        assert!(
            (i12 - i27).abs() < 0.08,
            "12px ink {i12} deviates from 27px ink {i27} beyond compensation tolerance"
        );
    }

    #[test]
    fn end_to_end_render_through_font_module() {
        // With the feature on, the production entry point routes
        // through this module; prove the full path produces output.
        let (buf, w, h) = crate::font::render_text_height("Rag7", [255; 4], [0, 0, 0, 255], 40);
        assert!(w > 0 && h == 40 && buf.len() == (w * h * 4) as usize);
        let bright = buf.chunks_exact(4).filter(|p| p[0] > 200).count();
        assert!(bright > 50, "expected bright glyph pixels, got {bright}");
    }

    #[test]
    fn upscale_beyond_strip_ceiling_renders() {
        // 96px is above the bitmap strip's 54px base — the SDF path
        // must produce clean non-empty output there.
        let (buf, w, h) = crate::font::render_text_height("R", [255; 4], [0, 0, 0, 255], 96);
        assert!(w > 0 && h == 96);
        let bright = buf.chunks_exact(4).filter(|p| p[0] > 200).count();
        assert!(
            bright > 300,
            "expected substantial ink at 96px, got {bright}"
        );
    }
}
