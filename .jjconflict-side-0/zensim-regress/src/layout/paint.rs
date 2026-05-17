//! Low-level pixel rendering — image and text leaves, plus the rect
//! drawing primitives. Called by both the per-modifier paint paths
//! ([`super::node::Node::paint`]) and the public render entry points.

use crate::pixel_ops::Bitmap;

use super::color::Color;
use super::geom::Rect;
use super::sizing::Fit;
use super::text::TextSpec;
use crate::pixel_ops::{self, ResampleFilter};

pub fn fill_rect(img: &mut Bitmap, rect: Rect, color: Color) {
    pixel_ops::fill_rect(img, rect.x, rect.y, rect.w, rect.h, color);
}

pub fn draw_rect_border(img: &mut Bitmap, rect: Rect, color: Color) {
    pixel_ops::draw_rect_border(img, rect.x, rect.y, rect.w, rect.h, color);
}

/// Paint an image leaf into `rect` with the given [`Fit`] mode.
///
/// `Fit::None` and `Fit::Stretch` (no resize needed) place the image at
/// the rect's top-left; `Fit::Contain` and `Fit::Cover` center within
/// the rect.
pub(super) fn render_image(img: &Bitmap, mode: Fit, rect: Rect, canvas: &mut Bitmap) {
    if rect.w == 0 || rect.h == 0 {
        return;
    }
    let (iw, ih) = (img.width(), img.height());
    if iw == 0 || ih == 0 {
        return;
    }
    let scaled: Option<Bitmap> = match mode {
        Fit::None => None,
        Fit::Stretch if (iw, ih) != (rect.w, rect.h) => Some(pixel_ops::resize(
            img,
            rect.w,
            rect.h,
            ResampleFilter::Triangle,
        )),
        Fit::Stretch => None,
        Fit::Contain | Fit::Cover => {
            let sx = rect.w as f32 / iw as f32;
            let sy = rect.h as f32 / ih as f32;
            let s = if matches!(mode, Fit::Contain) {
                sx.min(sy)
            } else {
                sx.max(sy)
            };
            let nw = (iw as f32 * s).round().max(1.0) as u32;
            let nh = (ih as f32 * s).round().max(1.0) as u32;
            Some(pixel_ops::resize(img, nw, nh, ResampleFilter::Triangle))
        }
    };

    let (src, sw, sh) = match &scaled {
        Some(s) => (s, s.width(), s.height()),
        None => (img, iw, ih),
    };

    let centered = matches!(mode, Fit::Contain | Fit::Cover);
    let (dst_x, dst_y) = if centered {
        (
            rect.x.saturating_add(rect.w.saturating_sub(sw) / 2) as i64,
            rect.y.saturating_add(rect.h.saturating_sub(sh) / 2) as i64,
        )
    } else {
        (rect.x as i64, rect.y as i64)
    };

    if sw <= rect.w && sh <= rect.h {
        pixel_ops::overlay(canvas, src, dst_x, dst_y);
    } else {
        let crop_x = if centered {
            sw.saturating_sub(rect.w) / 2
        } else {
            0
        };
        let crop_y = if centered {
            sh.saturating_sub(rect.h) / 2
        } else {
            0
        };
        let cw = rect.w.min(sw);
        let ch = rect.h.min(sh);
        let cropped = pixel_ops::crop(src, crop_x, crop_y, cw, ch);
        let (dst_x, dst_y) = if centered {
            (
                rect.x.saturating_add(rect.w.saturating_sub(cw) / 2) as i64,
                rect.y.saturating_add(rect.h.saturating_sub(ch) / 2) as i64,
            )
        } else {
            (rect.x as i64, rect.y as i64)
        };
        pixel_ops::overlay(canvas, &cropped, dst_x, dst_y);
    }
}

/// Paint a text leaf at the rect's top-left. The rasterizer is given
/// `(rect.w, rect.h)` so glyph height shrinks to fit either axis.
/// Wrap in [`super::node::Node::Align`] to reposition.
pub(super) fn render_text(spec: &TextSpec, rect: Rect, canvas: &mut Bitmap) {
    let (buf, w, h) = spec.rasterize(rect.w, rect.h);
    if w == 0 || h == 0 {
        return;
    }
    let Some(img) = Bitmap::from_raw(w, h, buf) else {
        return;
    };
    pixel_ops::overlay(canvas, &img, rect.x as i64, rect.y as i64);
}
