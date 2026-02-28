#![allow(dead_code)]

/// Multi-frequency checkerboard. Strong multi-scale edges.
pub fn gen_checkerboard(w: usize, h: usize, freq: usize) -> Vec<[u8; 3]> {
    let mut pixels = vec![[0u8; 3]; w * h];
    for y in 0..h {
        for x in 0..w {
            let tile = ((x * freq / w) + (y * freq / h)) % 2;
            let v = if tile == 0 { 240u8 } else { 16u8 };
            let r = v;
            let g = v.wrapping_add((x % 17) as u8);
            let b = v.wrapping_add((y % 13) as u8);
            pixels[y * w + x] = [r, g, b];
        }
    }
    pixels
}

/// Mandelbrot escape-time coloring with bit-manipulation palette.
/// Rich fractal structure activates features at all scales.
pub fn gen_mandelbrot(w: usize, h: usize) -> Vec<[u8; 3]> {
    let mut pixels = vec![[0u8; 3]; w * h];
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
            if iter == max_iter {
                pixels[y * w + x] = [0, 0, 0];
            } else {
                pixels[y * w + x] = [
                    (iter.wrapping_mul(7) & 255) as u8,
                    (iter.wrapping_mul(13) & 255) as u8,
                    (iter.wrapping_mul(19) & 255) as u8,
                ];
            }
        }
    }
    pixels
}

/// Integer hash noise with bilinear interpolation, 3 octaves.
/// Multi-scale texture activates high-frequency features.
pub fn gen_value_noise(w: usize, h: usize, seed: u32) -> Vec<[u8; 3]> {
    let hash = |x: i32, y: i32, s: u32| -> u8 {
        let mut h = (x as u32).wrapping_mul(374761393)
            ^ (y as u32).wrapping_mul(668265263)
            ^ s.wrapping_mul(1274126177);
        h = h.wrapping_mul(h).wrapping_add(h);
        h ^= h >> 16;
        h = h.wrapping_mul(2654435769);
        (h >> 24) as u8
    };

    let noise_layer = |px: usize, py: usize, grid: usize, ch_seed: u32| -> u8 {
        let gx = px / grid;
        let gy = py / grid;
        let fx = (px % grid) as u32;
        let fy = (py % grid) as u32;
        let g = grid as u32;

        let v00 = hash(gx as i32, gy as i32, ch_seed) as u32;
        let v10 = hash(gx as i32 + 1, gy as i32, ch_seed) as u32;
        let v01 = hash(gx as i32, gy as i32 + 1, ch_seed) as u32;
        let v11 = hash(gx as i32 + 1, gy as i32 + 1, ch_seed) as u32;

        let top = v00 * (g - fx) + v10 * fx;
        let bot = v01 * (g - fx) + v11 * fx;
        let val = (top * (g - fy) + bot * fy) / (g * g);
        val.min(255) as u8
    };

    let mut pixels = vec![[0u8; 3]; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut r = 0u16;
            let mut g = 0u16;
            let mut b = 0u16;
            for (grid, weight) in [(16, 4u16), (8, 2), (4, 1)] {
                r += noise_layer(x, y, grid, seed) as u16 * weight;
                g += noise_layer(x, y, grid, seed.wrapping_add(1)) as u16 * weight;
                b += noise_layer(x, y, grid, seed.wrapping_add(2)) as u16 * weight;
            }
            pixels[y * w + x] = [
                (r / 7).min(255) as u8,
                (g / 7).min(255) as u8,
                (b / 7).min(255) as u8,
            ];
        }
    }
    pixels
}

/// 4x4 grid of 16 maximally diverse saturated colors.
/// Drives XYB B-channel features hard.
pub fn gen_color_blocks(w: usize, h: usize) -> Vec<[u8; 3]> {
    #[rustfmt::skip]
    let colors: [[u8; 3]; 16] = [
        [255, 0, 0],     [0, 255, 0],     [0, 0, 255],     [255, 255, 0],
        [255, 0, 255],   [0, 255, 255],   [255, 128, 0],   [128, 0, 255],
        [0, 128, 255],   [255, 0, 128],   [128, 255, 0],   [0, 255, 128],
        [64, 0, 128],    [128, 64, 0],    [0, 128, 64],    [192, 192, 64],
    ];
    let mut pixels = vec![[0u8; 3]; w * h];
    for y in 0..h {
        for x in 0..w {
            let bx = (x * 4) / w;
            let by = (y * 4) / h;
            let idx = by * 4 + bx;
            pixels[y * w + x] = colors[idx];
        }
    }
    pixels
}

// ─── Format conversion helpers ─────────────────────────────────────────

/// sRGB u8 → linear f32 (exact formula, no LUT).
#[inline]
pub fn srgb_to_linear(v: u8) -> f32 {
    let s = v as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

pub fn to_srgb8_rgb(pixels: &[[u8; 3]], w: usize, h: usize) -> (Vec<u8>, usize) {
    let stride = w * 3;
    let mut buf = vec![0u8; h * stride];
    for y in 0..h {
        for x in 0..w {
            let p = pixels[y * w + x];
            let off = y * stride + x * 3;
            buf[off] = p[0];
            buf[off + 1] = p[1];
            buf[off + 2] = p[2];
        }
    }
    (buf, stride)
}

pub fn to_srgb8_rgba(pixels: &[[u8; 3]], w: usize, h: usize) -> (Vec<u8>, usize) {
    let stride = w * 4;
    let mut buf = vec![0u8; h * stride];
    for y in 0..h {
        for x in 0..w {
            let p = pixels[y * w + x];
            let off = y * stride + x * 4;
            buf[off] = p[0];
            buf[off + 1] = p[1];
            buf[off + 2] = p[2];
            buf[off + 3] = 255;
        }
    }
    (buf, stride)
}

pub fn to_srgb8_bgra(pixels: &[[u8; 3]], w: usize, h: usize) -> (Vec<u8>, usize) {
    let stride = w * 4;
    let mut buf = vec![0u8; h * stride];
    for y in 0..h {
        for x in 0..w {
            let p = pixels[y * w + x];
            let off = y * stride + x * 4;
            buf[off] = p[2]; // B
            buf[off + 1] = p[1]; // G
            buf[off + 2] = p[0]; // R
            buf[off + 3] = 255; // A
        }
    }
    (buf, stride)
}

pub fn to_linear_f32_rgba(pixels: &[[u8; 3]], w: usize, h: usize) -> (Vec<u8>, usize) {
    let stride = w * 16;
    let mut buf = vec![0u8; h * stride];
    for y in 0..h {
        for x in 0..w {
            let p = pixels[y * w + x];
            let off = y * stride + x * 16;
            let r = srgb_to_linear(p[0]);
            let g = srgb_to_linear(p[1]);
            let b = srgb_to_linear(p[2]);
            let a: f32 = 1.0;
            buf[off..off + 4].copy_from_slice(&r.to_ne_bytes());
            buf[off + 4..off + 8].copy_from_slice(&g.to_ne_bytes());
            buf[off + 8..off + 12].copy_from_slice(&b.to_ne_bytes());
            buf[off + 12..off + 16].copy_from_slice(&a.to_ne_bytes());
        }
    }
    (buf, stride)
}

pub fn to_srgb16_rgba(pixels: &[[u8; 3]], w: usize, h: usize) -> (Vec<u8>, usize) {
    let stride = w * 8;
    let mut buf = vec![0u8; h * stride];
    for y in 0..h {
        for x in 0..w {
            let p = pixels[y * w + x];
            let off = y * stride + x * 8;
            let r = (p[0] as u16) * 257;
            let g = (p[1] as u16) * 257;
            let b = (p[2] as u16) * 257;
            let a: u16 = 65535; // fully opaque
            buf[off..off + 2].copy_from_slice(&r.to_ne_bytes());
            buf[off + 2..off + 4].copy_from_slice(&g.to_ne_bytes());
            buf[off + 4..off + 6].copy_from_slice(&b.to_ne_bytes());
            buf[off + 6..off + 8].copy_from_slice(&a.to_ne_bytes());
        }
    }
    (buf, stride)
}

/// sRGB f16 RGBA: stores sRGB-space values as f16 (not linear).
pub fn to_srgb_f16_rgba(pixels: &[[u8; 3]], w: usize, h: usize) -> (Vec<u8>, usize) {
    let stride = w * 8;
    let mut buf = vec![0u8; h * stride];
    for y in 0..h {
        for x in 0..w {
            let p = pixels[y * w + x];
            let off = y * stride + x * 8;
            // Store sRGB values (not linear) — SrgbF16Rgba does TRC decode internally
            let r = half::f16::from_f32(p[0] as f32 / 255.0);
            let g = half::f16::from_f32(p[1] as f32 / 255.0);
            let b = half::f16::from_f32(p[2] as f32 / 255.0);
            let a = half::f16::from_f32(1.0);
            buf[off..off + 2].copy_from_slice(&r.to_ne_bytes());
            buf[off + 2..off + 4].copy_from_slice(&g.to_ne_bytes());
            buf[off + 4..off + 6].copy_from_slice(&b.to_ne_bytes());
            buf[off + 6..off + 8].copy_from_slice(&a.to_ne_bytes());
        }
    }
    (buf, stride)
}

// ─── Distortion functions ──────────────────────────────────────────────

/// Separable box blur with given radius.
pub fn distort_blur(src: &[[u8; 3]], w: usize, h: usize, radius: usize) -> Vec<[u8; 3]> {
    let mut tmp = vec![[0u16; 3]; w * h];
    let mut dst = vec![[0u8; 3]; w * h];
    let r = radius as i32;
    let diam = (2 * radius + 1) as u16;

    for y in 0..h {
        for x in 0..w {
            let mut sum = [0u16; 3];
            for dx in -r..=r {
                let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                for c in 0..3 {
                    sum[c] += src[y * w + sx][c] as u16;
                }
            }
            tmp[y * w + x] = sum;
        }
    }

    for y in 0..h {
        for x in 0..w {
            let mut sum = [0u32; 3];
            for dy in -r..=r {
                let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                for c in 0..3 {
                    sum[c] += tmp[sy * w + x][c] as u32;
                }
            }
            let d2 = (diam as u32) * (diam as u32);
            dst[y * w + x] = [
                (sum[0] / d2) as u8,
                (sum[1] / d2) as u8,
                (sum[2] / d2) as u8,
            ];
        }
    }
    dst
}

/// Unsharp mask: dst = src + alpha * (src - blur(src)).
pub fn distort_sharpen(src: &[[u8; 3]], w: usize, h: usize) -> Vec<[u8; 3]> {
    let blurred = distort_blur(src, w, h, 2);
    let alpha = 2.0f32;
    let mut dst = vec![[0u8; 3]; w * h];
    for i in 0..w * h {
        for c in 0..3 {
            let s = src[i][c] as f32;
            let b = blurred[i][c] as f32;
            let v = s + alpha * (s - b);
            dst[i][c] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    dst
}

/// Color shift: R+20, G-15, B+30 (saturating).
pub fn distort_color_shift(src: &[[u8; 3]], w: usize, h: usize) -> Vec<[u8; 3]> {
    let mut dst = vec![[0u8; 3]; w * h];
    for i in 0..w * h {
        dst[i] = [
            src[i][0].saturating_add(20),
            src[i][1].saturating_sub(15),
            src[i][2].saturating_add(30),
        ];
    }
    dst
}

/// 8x8 block averaging + boundary offset. Simulates JPEG blocking.
pub fn distort_block_artifacts(src: &[[u8; 3]], w: usize, h: usize) -> Vec<[u8; 3]> {
    let mut dst = src.to_vec();

    let bw = w.div_ceil(8);
    let bh = h.div_ceil(8);
    for by in 0..bh {
        for bx in 0..bw {
            let x0 = bx * 8;
            let y0 = by * 8;
            let x1 = (x0 + 8).min(w);
            let y1 = (y0 + 8).min(h);
            let count = ((x1 - x0) * (y1 - y0)) as u32;

            let mut sum = [0u32; 3];
            for y in y0..y1 {
                for x in x0..x1 {
                    for c in 0..3 {
                        sum[c] += src[y * w + x][c] as u32;
                    }
                }
            }
            let avg = [
                (sum[0] / count) as u8,
                (sum[1] / count) as u8,
                (sum[2] / count) as u8,
            ];

            for y in y0..y1 {
                for x in x0..x1 {
                    for c in 0..3 {
                        dst[y * w + x][c] = ((src[y * w + x][c] as u16 + avg[c] as u16) / 2) as u8;
                    }
                }
            }
        }
    }

    for y in 0..h {
        for x in 0..w {
            if x % 8 == 0 || y % 8 == 0 {
                dst[y * w + x][0] = dst[y * w + x][0].saturating_add(8);
                dst[y * w + x][1] = dst[y * w + x][1].saturating_sub(4);
                dst[y * w + x][2] = dst[y * w + x][2].saturating_add(6);
            }
        }
    }
    dst
}
