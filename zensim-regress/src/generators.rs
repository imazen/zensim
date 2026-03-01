//! Procedural test image generators for visual regression testing.
//!
//! These generators produce deterministic RGBA pixel buffers suitable for
//! use with [`ChecksumManager::check_pixels()`](crate::manager::ChecksumManager::check_pixels)
//! and other regression testing APIs.
//!
//! Each generator targets specific image characteristics:
//! - [`checkerboard`] — multi-frequency edges, good for edge detection and scaling tests
//! - [`mandelbrot`] — fractal structure with features at every scale
//! - [`value_noise`] — multi-octave noise texture, organic gradients
//! - [`color_blocks`] — 16 saturated colors in a 4x4 grid, tests color space handling
//! - [`gradient`] — smooth per-channel gradients, good for banding/quantization tests
//! - [`solid`] — uniform color, useful as a baseline or for format round-trip tests

/// Multi-frequency checkerboard with slight color variation.
///
/// Produces strong multi-scale edges. Good for testing resize, sharpen,
/// and edge-aware operations.
pub fn checkerboard(w: u32, h: u32, freq: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let tile = ((x * freq / w) + (y * freq / h)) % 2;
            let v = if tile == 0 { 240u8 } else { 16u8 };
            let r = v;
            let g = v.wrapping_add((x % 17) as u8);
            let b = v.wrapping_add((y % 13) as u8);
            let off = ((y * w + x) * 4) as usize;
            buf[off] = r;
            buf[off + 1] = g;
            buf[off + 2] = b;
            buf[off + 3] = 255;
        }
    }
    buf
}

/// Mandelbrot set with escape-time coloring.
///
/// Rich fractal structure activates features at all scales.
/// Colors use bit-manipulation for a varied palette.
pub fn mandelbrot(w: u32, h: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (w * h * 4) as usize];
    let max_iter = 256u32;
    for y in 0..h {
        for x in 0..w {
            let cx = -2.0 + (x as f64) * 2.5 / (w as f64);
            let cy = -1.25 + (y as f64) * 2.5 / (h as f64);
            let mut zx = 0.0f64;
            let mut zy = 0.0f64;
            let mut iter = 0u32;
            while zx * zx + zy * zy <= 4.0 && iter < max_iter {
                let tmp = zx * zx - zy * zy + cx;
                zy = 2.0 * zx * zy + cy;
                zx = tmp;
                iter += 1;
            }
            let off = ((y * w + x) * 4) as usize;
            if iter == max_iter {
                buf[off] = 0;
                buf[off + 1] = 0;
                buf[off + 2] = 0;
            } else {
                buf[off] = (iter.wrapping_mul(7) & 255) as u8;
                buf[off + 1] = (iter.wrapping_mul(13) & 255) as u8;
                buf[off + 2] = (iter.wrapping_mul(19) & 255) as u8;
            }
            buf[off + 3] = 255;
        }
    }
    buf
}

/// Integer hash noise with bilinear interpolation, 3 octaves.
///
/// Multi-scale texture useful for high-frequency feature testing.
/// `seed` controls the noise pattern — different seeds produce
/// different images with similar statistical properties.
pub fn value_noise(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let hash = |x: i32, y: i32, s: u32| -> u8 {
        let mut h = (x as u32).wrapping_mul(374761393)
            ^ (y as u32).wrapping_mul(668265263)
            ^ s.wrapping_mul(1274126177);
        h = h.wrapping_mul(h).wrapping_add(h);
        h ^= h >> 16;
        h = h.wrapping_mul(2654435769);
        (h >> 24) as u8
    };

    let noise_layer = |px: u32, py: u32, grid: u32, ch_seed: u32| -> u8 {
        let gx = px / grid;
        let gy = py / grid;
        let fx = px % grid;
        let fy = py % grid;

        let v00 = hash(gx as i32, gy as i32, ch_seed) as u32;
        let v10 = hash(gx as i32 + 1, gy as i32, ch_seed) as u32;
        let v01 = hash(gx as i32, gy as i32 + 1, ch_seed) as u32;
        let v11 = hash(gx as i32 + 1, gy as i32 + 1, ch_seed) as u32;

        let top = v00 * (grid - fx) + v10 * fx;
        let bot = v01 * (grid - fx) + v11 * fx;
        let val = (top * (grid - fy) + bot * fy) / (grid * grid);
        val.min(255) as u8
    };

    let mut buf = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let mut r = 0u16;
            let mut g = 0u16;
            let mut b = 0u16;
            for (grid, weight) in [(16u32, 4u16), (8, 2), (4, 1)] {
                r += noise_layer(x, y, grid, seed) as u16 * weight;
                g += noise_layer(x, y, grid, seed.wrapping_add(1)) as u16 * weight;
                b += noise_layer(x, y, grid, seed.wrapping_add(2)) as u16 * weight;
            }
            let off = ((y * w + x) * 4) as usize;
            buf[off] = (r / 7).min(255) as u8;
            buf[off + 1] = (g / 7).min(255) as u8;
            buf[off + 2] = (b / 7).min(255) as u8;
            buf[off + 3] = 255;
        }
    }
    buf
}

/// 4x4 grid of 16 maximally diverse saturated colors.
///
/// Strong color diversity tests color space handling and
/// cross-channel correlation features.
pub fn color_blocks(w: u32, h: u32) -> Vec<u8> {
    #[rustfmt::skip]
    let colors: [[u8; 3]; 16] = [
        [255, 0, 0],     [0, 255, 0],     [0, 0, 255],     [255, 255, 0],
        [255, 0, 255],   [0, 255, 255],   [255, 128, 0],   [128, 0, 255],
        [0, 128, 255],   [255, 0, 128],   [128, 255, 0],   [0, 255, 128],
        [64, 0, 128],    [128, 64, 0],    [0, 128, 64],    [192, 192, 64],
    ];
    let mut buf = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let bx = (x * 4) / w;
            let by = (y * 4) / h;
            let idx = (by * 4 + bx) as usize;
            let off = ((y * w + x) * 4) as usize;
            buf[off] = colors[idx][0];
            buf[off + 1] = colors[idx][1];
            buf[off + 2] = colors[idx][2];
            buf[off + 3] = 255;
        }
    }
    buf
}

/// Smooth per-channel gradient.
///
/// Channel values vary independently across x and y:
/// - R increases with x
/// - G increases with y
/// - B increases with x+y (diagonal)
///
/// Good for banding, quantization, and color space round-trip tests.
pub fn gradient(w: u32, h: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let off = ((y * w + x) * 4) as usize;
            buf[off] = ((x * 255) / w.max(1)) as u8;
            buf[off + 1] = ((y * 255) / h.max(1)) as u8;
            buf[off + 2] = (((x + y) * 255) / (w + h).max(1)) as u8;
            buf[off + 3] = 255;
        }
    }
    buf
}

/// Uniform solid color.
///
/// Useful as a baseline for delta testing, or for format
/// round-trip verification.
pub fn solid(w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
    let pixel = [r, g, b, a];
    pixel.iter().copied().cycle().take((w * h * 4) as usize).collect()
}

/// Apply an off-by-N delta to every Nth pixel's red channel.
///
/// Produces a controlled, deterministic perturbation for testing
/// tolerance thresholds.
pub fn off_by_n(rgba: &[u8], delta: u8, every_nth: usize) -> Vec<u8> {
    assert!(rgba.len() % 4 == 0, "RGBA byte length must be a multiple of 4");
    rgba.chunks(4)
        .enumerate()
        .flat_map(|(i, px)| {
            if every_nth > 0 && i % every_nth == 0 {
                [px[0].saturating_add(delta), px[1], px[2], px[3]]
            } else {
                [px[0], px[1], px[2], px[3]]
            }
        })
        .collect()
}

/// Convert RGBA8 sRGB data to an arbitrary interleaved pixel format.
///
/// Takes RGBA8 output from any generator and converts it to `target` via
/// [`zenpixels::RowConverter`]. Panics if conversion fails (test helper —
/// should always succeed for formats returned by [`supported_formats`]).
#[cfg(feature = "zenpixels")]
pub fn generate_in_format(
    rgba8_data: &[u8],
    width: u32,
    height: u32,
    target: zenpixels::PixelDescriptor,
) -> Vec<u8> {
    assert!(
        rgba8_data.len() == (width as usize) * (height as usize) * 4,
        "generate_in_format: expected {} bytes, got {}",
        (width as usize) * (height as usize) * 4,
        rgba8_data.len(),
    );

    let source = zenpixels::PixelDescriptor::RGBA8_SRGB;
    let converter = zenpixels::RowConverter::new(source, target)
        .unwrap_or_else(|e| panic!("generate_in_format: cannot convert RGBA8_SRGB → {target:?}: {e}"));

    if converter.is_identity() {
        return rgba8_data.to_vec();
    }

    let src_bpp = source.bytes_per_pixel();
    let dst_bpp = target.bytes_per_pixel();
    let src_stride = width as usize * src_bpp;
    let dst_stride = width as usize * dst_bpp;
    let mut buf = vec![0u8; height as usize * dst_stride];

    for y in 0..height as usize {
        let src_row = &rgba8_data[y * src_stride..y * src_stride + src_stride];
        let dst_row = &mut buf[y * dst_stride..y * dst_stride + dst_stride];
        converter.convert_row(src_row, dst_row, width);
    }
    buf
}

/// All interleaved pixel formats that zensim-regress supports.
///
/// Returns descriptors for every combination of:
/// - Channel type: U8, U16, F32
/// - Layout: Gray, GrayAlpha, Rgb, Rgba, Bgra
/// - Transfer: sRGB (for U8/U16), Linear (for F32)
///
/// Plus RGBX and BGRX variants (U8 sRGB with undefined alpha).
/// Total: 30+ descriptors.
#[cfg(feature = "zenpixels")]
pub fn supported_formats() -> Vec<zenpixels::PixelDescriptor> {
    use zenpixels::PixelDescriptor as PD;
    vec![
        // U8 sRGB
        PD::GRAY8_SRGB,
        PD::GRAYA8_SRGB,
        PD::RGB8_SRGB,
        PD::RGBA8_SRGB,
        PD::BGRA8_SRGB,
        PD::RGBX8_SRGB,
        PD::BGRX8_SRGB,
        // U16 sRGB
        PD::GRAY16_SRGB,
        PD::GRAYA16_SRGB,
        PD::RGB16_SRGB,
        PD::RGBA16_SRGB,
        // F32 Linear
        PD::GRAYF32_LINEAR,
        PD::GRAYAF32_LINEAR,
        PD::RGBF32_LINEAR,
        PD::RGBAF32_LINEAR,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkerboard_dimensions() {
        let buf = checkerboard(64, 32, 8);
        assert_eq!(buf.len(), 64 * 32 * 4);
    }

    #[test]
    fn checkerboard_all_opaque() {
        let buf = checkerboard(16, 16, 4);
        for px in buf.chunks_exact(4) {
            assert_eq!(px[3], 255);
        }
    }

    #[test]
    fn mandelbrot_dimensions() {
        let buf = mandelbrot(128, 128);
        assert_eq!(buf.len(), 128 * 128 * 4);
    }

    #[test]
    fn mandelbrot_has_black_center() {
        let buf = mandelbrot(128, 128);
        // The Mandelbrot set center (-0.5, 0.0) should be black.
        // At w=128, x=48 maps to cx ≈ -1.06 which is in the main cardioid.
        // Check a known-in-set pixel: center of image (x=76, y=64) maps to cx≈-0.516
        let x = 76;
        let y = 64;
        let off = (y * 128 + x) * 4;
        assert_eq!(buf[off], 0);
        assert_eq!(buf[off + 1], 0);
        assert_eq!(buf[off + 2], 0);
    }

    #[test]
    fn value_noise_deterministic() {
        let a = value_noise(32, 32, 42);
        let b = value_noise(32, 32, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn value_noise_different_seeds() {
        let a = value_noise(32, 32, 1);
        let b = value_noise(32, 32, 2);
        assert_ne!(a, b);
    }

    #[test]
    fn color_blocks_has_diversity() {
        let buf = color_blocks(64, 64);
        // Collect unique RGB tuples (ignoring alpha).
        let mut unique = std::collections::HashSet::new();
        for px in buf.chunks_exact(4) {
            unique.insert((px[0], px[1], px[2]));
        }
        // Should have at least 16 distinct colors.
        assert!(unique.len() >= 16, "expected >=16 unique colors, got {}", unique.len());
    }

    #[test]
    fn gradient_extremes() {
        let buf = gradient(256, 256);
        // Top-left: R=0, G=0, B=0
        assert_eq!(buf[0], 0);
        assert_eq!(buf[1], 0);
        assert_eq!(buf[2], 0);
        // Bottom-right: R≈255, G≈255, B≈255
        let off = (255 * 256 + 255) * 4;
        assert!(buf[off] > 250);
        assert!(buf[off + 1] > 250);
    }

    #[test]
    fn solid_uniform() {
        let buf = solid(8, 8, 128, 64, 32, 200);
        for px in buf.chunks_exact(4) {
            assert_eq!(px, [128, 64, 32, 200]);
        }
    }

    #[test]
    fn off_by_n_modifies_red_only() {
        let base = solid(4, 4, 100, 100, 100, 255);
        let perturbed = off_by_n(&base, 5, 2);
        // Pixel 0: modified (R=105)
        assert_eq!(perturbed[0], 105);
        assert_eq!(perturbed[1], 100);
        assert_eq!(perturbed[2], 100);
        // Pixel 1: unchanged
        assert_eq!(perturbed[4], 100);
        // Pixel 2: modified
        assert_eq!(perturbed[8], 105);
    }

    #[test]
    fn off_by_n_saturates() {
        let base = solid(2, 2, 254, 100, 100, 255);
        let perturbed = off_by_n(&base, 10, 1);
        // Should saturate at 255, not wrap.
        assert_eq!(perturbed[0], 255);
    }
}
