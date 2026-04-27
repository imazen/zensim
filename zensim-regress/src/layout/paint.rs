//! Low-level pixel rendering — image and text leaves, plus the rect
//! drawing primitives. Called by both the per-modifier paint paths
//! ([`super::node::Node::paint`]) and the public render entry points.

use image::{Rgba, RgbaImage, imageops};

use super::color::Color;
use super::geom::Rect;
use super::sizing::Fit;
use super::text::TextSpec;

pub fn fill_rect(img: &mut RgbaImage, rect: Rect, color: Color) {
    let px = Rgba(color);
    let img_w = img.width();
    let img_h = img.height();
    let x_end = rect.x.saturating_add(rect.w).min(img_w);
    let y_end = rect.y.saturating_add(rect.h).min(img_h);
    for y in rect.y..y_end {
        for x in rect.x..x_end {
            img.put_pixel(x, y, px);
        }
    }
}

pub fn draw_rect_border(img: &mut RgbaImage, rect: Rect, color: Color) {
    if rect.w == 0 || rect.h == 0 {
        return;
    }
    let px = Rgba(color);
    let img_w = img.width();
    let img_h = img.height();
    let x_end = rect.x.saturating_add(rect.w).min(img_w);
    let y_end = rect.y.saturating_add(rect.h).min(img_h);
    for x in rect.x..x_end {
        if rect.y < img_h {
            img.put_pixel(x, rect.y, px);
        }
        let bot = y_end.saturating_sub(1);
        if bot < img_h && bot > rect.y {
            img.put_pixel(x, bot, px);
        }
    }
    for y in rect.y..y_end {
        if rect.x < img_w {
            img.put_pixel(rect.x, y, px);
        }
        let right = x_end.saturating_sub(1);
        if right < img_w && right > rect.x {
            img.put_pixel(right, y, px);
        }
    }
}

/// Paint an image leaf into `rect` with the given [`Fit`] mode.
///
/// `Fit::None` and `Fit::Stretch` (no resize needed) place the image at
/// the rect's top-left; `Fit::Contain` and `Fit::Cover` center within
/// the rect.
pub(super) fn render_image(img: &RgbaImage, mode: Fit, rect: Rect, canvas: &mut RgbaImage) {
    if rect.w == 0 || rect.h == 0 {
        return;
    }
    let (iw, ih) = (img.width(), img.height());
    if iw == 0 || ih == 0 {
        return;
    }
    let scaled: Option<RgbaImage> = match mode {
        Fit::None => None,
        Fit::Stretch if (iw, ih) != (rect.w, rect.h) => Some(imageops::resize(
            img,
            rect.w,
            rect.h,
            imageops::FilterType::Lanczos3,
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
            Some(imageops::resize(
                img,
                nw,
                nh,
                imageops::FilterType::Lanczos3,
            ))
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
        imageops::overlay(canvas, src, dst_x, dst_y);
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
        let cropped = imageops::crop_imm(src, crop_x, crop_y, cw, ch).to_image();
        let (dst_x, dst_y) = if centered {
            (
                rect.x.saturating_add(rect.w.saturating_sub(cw) / 2) as i64,
                rect.y.saturating_add(rect.h.saturating_sub(ch) / 2) as i64,
            )
        } else {
            (rect.x as i64, rect.y as i64)
        };
        imageops::overlay(canvas, &cropped, dst_x, dst_y);
    }
}

/// Paint a text leaf at the rect's top-left. The rasterizer is given
/// `(rect.w, rect.h)` so glyph height shrinks to fit either axis.
/// Wrap in [`super::node::Node::Align`] to reposition.
pub(super) fn render_text(spec: &TextSpec, rect: Rect, canvas: &mut RgbaImage) {
    let (buf, w, h) = spec.rasterize(rect.w, rect.h);
    if w == 0 || h == 0 {
        return;
    }
    let Some(img) = RgbaImage::from_raw(w, h, buf) else {
        return;
    };
    imageops::overlay(canvas, &img, rect.x as i64, rect.y as i64);
}
