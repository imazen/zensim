//! Cross-platform score consistency tests for zensim.
//!
//! Validates that hardcoded reference scores match across all 7 CI platforms,
//! that all 6 PixelFormat variants produce equivalent scores, that all 156
//! features activate on synthetic test images, and that results are deterministic.
//!
//! Run with: `cargo test -p zensim --all-features --test cross_platform`

use zensim::{PixelFormat, RgbSlice, StridedBytes, Zensim, ZensimProfile};

// ─── Procedural image generators ───────────────────────────────────────────

/// Multi-frequency checkerboard. Strong multi-scale edges.
fn gen_checkerboard(w: usize, h: usize, freq: usize) -> Vec<[u8; 3]> {
    let mut pixels = vec![[0u8; 3]; w * h];
    for y in 0..h {
        for x in 0..w {
            let tile = ((x * freq / w) + (y * freq / h)) % 2;
            let v = if tile == 0 { 240u8 } else { 16u8 };
            // Add slight color variation per axis
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
fn gen_mandelbrot(w: usize, h: usize) -> Vec<[u8; 3]> {
    let mut pixels = vec![[0u8; 3]; w * h];
    let max_iter = 256u32;
    for y in 0..h {
        for x in 0..w {
            // Map to [-2.0, 0.5] x [-1.25, 1.25]
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
fn gen_value_noise(w: usize, h: usize, seed: u32) -> Vec<[u8; 3]> {
    // Integer hash function (deterministic)
    let hash = |x: i32, y: i32, s: u32| -> u8 {
        let mut h = (x as u32).wrapping_mul(374761393)
            ^ (y as u32).wrapping_mul(668265263)
            ^ s.wrapping_mul(1274126177);
        h = h.wrapping_mul(h).wrapping_add(h);
        h ^= h >> 16;
        h = h.wrapping_mul(2654435769);
        (h >> 24) as u8
    };

    // Bilinear interpolation of hash noise at given grid spacing
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
            // 3 octaves: grid 16, 8, 4 with weights 4, 2, 1
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
fn gen_color_blocks(w: usize, h: usize) -> Vec<[u8; 3]> {
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

// ─── Distortion functions ──────────────────────────────────────────────────

/// Separable box blur with given radius.
/// Activates: det_*, hf_energy_loss, hf_mag_loss.
fn distort_blur(src: &[[u8; 3]], w: usize, h: usize, radius: usize) -> Vec<[u8; 3]> {
    let mut tmp = vec![[0u16; 3]; w * h];
    let mut dst = vec![[0u8; 3]; w * h];
    let r = radius as i32;
    let diam = (2 * radius + 1) as u16;

    // Horizontal pass
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

    // Vertical pass
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
/// Activates: art_*, hf_energy_gain.
fn distort_sharpen(src: &[[u8; 3]], w: usize, h: usize) -> Vec<[u8; 3]> {
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
/// Activates B-channel features.
fn distort_color_shift(src: &[[u8; 3]], w: usize, h: usize) -> Vec<[u8; 3]> {
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
/// Activates art_* at all scales.
fn distort_block_artifacts(src: &[[u8; 3]], w: usize, h: usize) -> Vec<[u8; 3]> {
    let mut dst = src.to_vec();

    // Block averaging within 8x8 blocks
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

            // Blend: 50% original + 50% block average
            for y in y0..y1 {
                for x in x0..x1 {
                    for c in 0..3 {
                        dst[y * w + x][c] = ((src[y * w + x][c] as u16 + avg[c] as u16) / 2) as u8;
                    }
                }
            }
        }
    }

    // Add boundary artifacts: offset pixels at block edges
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

// ─── Format conversion helpers ─────────────────────────────────────────────

/// sRGB u8 → linear f32 (exact formula, no LUT).
#[inline]
fn srgb_to_linear(v: u8) -> f32 {
    let s = v as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

fn to_srgb8_rgb(pixels: &[[u8; 3]], w: usize, h: usize) -> (Vec<u8>, usize) {
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

fn to_srgb8_rgba(pixels: &[[u8; 3]], w: usize, h: usize) -> (Vec<u8>, usize) {
    let stride = w * 4;
    let mut buf = vec![0u8; h * stride];
    for y in 0..h {
        for x in 0..w {
            let p = pixels[y * w + x];
            let off = y * stride + x * 4;
            buf[off] = p[0];
            buf[off + 1] = p[1];
            buf[off + 2] = p[2];
            buf[off + 3] = 255; // fully opaque
        }
    }
    (buf, stride)
}

fn to_srgb8_bgra(pixels: &[[u8; 3]], w: usize, h: usize) -> (Vec<u8>, usize) {
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

fn to_linear_f32_rgb(pixels: &[[u8; 3]], w: usize, h: usize) -> (Vec<u8>, usize) {
    let stride = w * 12; // 3 * f32
    let mut buf = vec![0u8; h * stride];
    for y in 0..h {
        for x in 0..w {
            let p = pixels[y * w + x];
            let off = y * stride + x * 12;
            let r = srgb_to_linear(p[0]);
            let g = srgb_to_linear(p[1]);
            let b = srgb_to_linear(p[2]);
            buf[off..off + 4].copy_from_slice(&r.to_ne_bytes());
            buf[off + 4..off + 8].copy_from_slice(&g.to_ne_bytes());
            buf[off + 8..off + 12].copy_from_slice(&b.to_ne_bytes());
        }
    }
    (buf, stride)
}

fn to_linear_f32_rgba(pixels: &[[u8; 3]], w: usize, h: usize) -> (Vec<u8>, usize) {
    let stride = w * 16; // 4 * f32
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

fn to_linear_f32_bgra(pixels: &[[u8; 3]], w: usize, h: usize) -> (Vec<u8>, usize) {
    let stride = w * 16; // 4 * f32
    let mut buf = vec![0u8; h * stride];
    for y in 0..h {
        for x in 0..w {
            let p = pixels[y * w + x];
            let off = y * stride + x * 16;
            let r = srgb_to_linear(p[0]);
            let g = srgb_to_linear(p[1]);
            let b = srgb_to_linear(p[2]);
            let a: f32 = 1.0;
            buf[off..off + 4].copy_from_slice(&b.to_ne_bytes()); // B
            buf[off + 4..off + 8].copy_from_slice(&g.to_ne_bytes()); // G
            buf[off + 8..off + 12].copy_from_slice(&r.to_ne_bytes()); // R
            buf[off + 12..off + 16].copy_from_slice(&a.to_ne_bytes()); // A
        }
    }
    (buf, stride)
}

// ─── Test pair generation ──────────────────────────────────────────────────

struct TestPair {
    name: &'static str,
    source: Vec<[u8; 3]>,
    distorted: Vec<[u8; 3]>,
}

fn generate_test_pairs(w: usize, h: usize) -> Vec<TestPair> {
    let checker = gen_checkerboard(w, h, 8);
    let mandel = gen_mandelbrot(w, h);
    let noise = gen_value_noise(w, h, 42);
    let blocks = gen_color_blocks(w, h);

    vec![
        TestPair {
            name: "checkerboard+blur",
            distorted: distort_blur(&checker, w, h, 3),
            source: checker.clone(),
        },
        TestPair {
            name: "checkerboard+sharpen",
            distorted: distort_sharpen(&checker, w, h),
            source: checker,
        },
        TestPair {
            name: "mandelbrot+blur",
            distorted: distort_blur(&mandel, w, h, 3),
            source: mandel.clone(),
        },
        TestPair {
            name: "mandelbrot+color_shift",
            distorted: distort_color_shift(&mandel, w, h),
            source: mandel,
        },
        TestPair {
            name: "noise+blur",
            distorted: distort_blur(&noise, w, h, 3),
            source: noise.clone(),
        },
        TestPair {
            name: "noise+block_artifacts",
            distorted: distort_block_artifacts(&noise, w, h),
            source: noise,
        },
        TestPair {
            name: "color_blocks+color_shift",
            distorted: distort_color_shift(&blocks, w, h),
            source: blocks.clone(),
        },
        TestPair {
            name: "color_blocks+sharpen",
            distorted: distort_sharpen(&blocks, w, h),
            source: blocks,
        },
    ]
}

// ─── Tests ─────────────────────────────────────────────────────────────────

/// Hardcoded reference scores validated across all 7 CI platforms.
/// Tolerance: ±1e-5 on the 0-100 scale (~55× headroom over observed x86↔ARM divergence).
#[test]
fn hardcoded_reference_scores() {
    const W: usize = 128;
    const H: usize = 128;
    const TOLERANCE: f64 = 1e-5;
    let z = Zensim::new(ZensimProfile::latest());
    let pairs = generate_test_pairs(W, H);

    // Midpoints between x86-64 and ARM CI scores (max half-range: 1.8e-7).
    #[allow(clippy::excessive_precision)]
    let expected: &[(&str, f64)] = &[
        ("checkerboard+blur", 0.0),
        ("checkerboard+sharpen", 36.594_580_282_021_894),
        ("mandelbrot+blur", 25.908_168_435_115_862),
        ("mandelbrot+color_shift", 52.156_362_712_514_31),
        ("noise+blur", 68.196_356_880_808_14),
        ("noise+block_artifacts", 62.534_969_411_414_69),
        ("color_blocks+color_shift", 38.519_226_827_523_10),
        ("color_blocks+sharpen", 18.263_318_750_293_17),
    ];

    let mut failures = Vec::new();
    for (pair, &(name, expected_score)) in pairs.iter().zip(expected.iter()) {
        assert_eq!(pair.name, name, "Test pair order mismatch");
        let src = RgbSlice::new(&pair.source, W, H);
        let dst = RgbSlice::new(&pair.distorted, W, H);
        let result = z.compute(&src, &dst).expect("compute failed");

        let diff = (result.score - expected_score).abs();
        println!(
            "  {name:30} score={:.15}  expected={expected_score:.15}  diff={diff:.2e}",
            result.score,
        );
        if diff > TOLERANCE {
            failures.push(format!(
                "{name}: score {:.15} differs from expected {expected_score:.15} by {diff:.2e} (>{TOLERANCE})",
                result.score,
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "Score mismatches:\n{}",
        failures.join("\n")
    );
}

/// All 6 PixelFormat variants produce equivalent scores.
/// sRGB↔f32: ±0.15, same-encoding reorder: ±0.01.
#[test]
fn pixel_format_equivalence() {
    const W: usize = 128;
    const H: usize = 128;
    let z = Zensim::new(ZensimProfile::latest());
    let src_pixels = gen_mandelbrot(W, H);
    let dst_pixels = distort_blur(&src_pixels, W, H, 3);

    // Reference: RgbSlice
    let ref_src = RgbSlice::new(&src_pixels, W, H);
    let ref_dst = RgbSlice::new(&dst_pixels, W, H);
    let ref_result = z.compute(&ref_src, &ref_dst).expect("ref compute failed");

    // Format converters and their tolerances
    type Converter = fn(&[[u8; 3]], usize, usize) -> (Vec<u8>, usize);

    struct FormatTest {
        name: &'static str,
        format: PixelFormat,
        converter: Converter,
        tolerance: f64,
    }

    let formats = [
        FormatTest {
            name: "Srgb8Rgb",
            format: PixelFormat::Srgb8Rgb,
            converter: to_srgb8_rgb,
            tolerance: 0.01,
        },
        FormatTest {
            name: "Srgb8Rgba",
            format: PixelFormat::Srgb8Rgba,
            converter: to_srgb8_rgba,
            tolerance: 0.01,
        },
        FormatTest {
            name: "Srgb8Bgra",
            format: PixelFormat::Srgb8Bgra,
            converter: to_srgb8_bgra,
            tolerance: 0.01,
        },
        FormatTest {
            name: "LinearF32Rgb",
            format: PixelFormat::LinearF32Rgb,
            converter: to_linear_f32_rgb,
            tolerance: 0.15,
        },
        FormatTest {
            name: "LinearF32Rgba",
            format: PixelFormat::LinearF32Rgba,
            converter: to_linear_f32_rgba,
            tolerance: 0.15,
        },
        FormatTest {
            name: "LinearF32Bgra",
            format: PixelFormat::LinearF32Bgra,
            converter: to_linear_f32_bgra,
            tolerance: 0.15,
        },
    ];

    println!("  Reference (RgbSlice): score={:.6}", ref_result.score);

    for fmt in &formats {
        let (src_buf, src_stride) = (fmt.converter)(&src_pixels, W, H);
        let (dst_buf, dst_stride) = (fmt.converter)(&dst_pixels, W, H);
        let src = StridedBytes::new(&src_buf, W, H, src_stride, fmt.format);
        let dst = StridedBytes::new(&dst_buf, W, H, dst_stride, fmt.format);
        let result = z.compute(&src, &dst).expect("compute failed");

        let diff = (result.score - ref_result.score).abs();
        println!(
            "  {:20} score={:.6}  diff={diff:.6}  (tol={:.2})",
            fmt.name, result.score, fmt.tolerance,
        );
        assert!(
            diff <= fmt.tolerance,
            "{}: score {:.6} differs from reference {:.6} by {diff:.6} (>{:.2})",
            fmt.name,
            result.score,
            ref_result.score,
            fmt.tolerance,
        );
    }
}

/// All 156 features must be non-trivial (max > 1e-6 across all 8 test pairs).
#[cfg(feature = "training")]
#[test]
fn feature_coverage() {
    const W: usize = 128;
    const H: usize = 128;
    const NUM_FEATURES: usize = 156;
    let z = Zensim::new(ZensimProfile::latest());
    let pairs = generate_test_pairs(W, H);

    let mut max_per_feature = vec![0.0f64; NUM_FEATURES];

    for pair in &pairs {
        let src = RgbSlice::new(&pair.source, W, H);
        let dst = RgbSlice::new(&pair.distorted, W, H);
        let result = z
            .compute_all_features(&src, &dst)
            .expect("compute_all_features failed");

        assert_eq!(
            result.features.len(),
            NUM_FEATURES,
            "Expected {NUM_FEATURES} features, got {}",
            result.features.len(),
        );

        for (i, &f) in result.features.iter().enumerate() {
            max_per_feature[i] = max_per_feature[i].max(f.abs());
        }
    }

    let feature_names = [
        "ssim_mean",
        "ssim_4th",
        "ssim_2nd",
        "art_mean",
        "art_4th",
        "art_2nd",
        "det_mean",
        "det_4th",
        "det_2nd",
        "mse",
        "hf_energy_loss",
        "hf_mag_loss",
        "hf_energy_gain",
    ];

    let mut dead_features = Vec::new();
    for (i, &max_val) in max_per_feature.iter().enumerate() {
        let scale = i / 39;
        let within = i % 39;
        let ch = within / 13;
        let fi = within % 13;
        let ch_name = ["X", "Y", "B"][ch];
        let f_name = feature_names[fi];

        if max_val <= 1e-6 {
            dead_features.push(format!(
                "  feat[{i:3}] s{scale} {ch_name} {f_name:16} max={max_val:.2e}"
            ));
        }
    }

    if !dead_features.is_empty() {
        panic!(
            "{} of {NUM_FEATURES} features never exceeded 1e-6:\n{}",
            dead_features.len(),
            dead_features.join("\n"),
        );
    }

    println!("  All {NUM_FEATURES} features activated (max > 1e-6)");
}

/// Basic sanity: identical=100, blur<100, heavier blur=lower score, all in [0,100].
#[test]
fn score_sanity_checks() {
    const W: usize = 128;
    const H: usize = 128;
    let z = Zensim::new(ZensimProfile::latest());
    let source = gen_mandelbrot(W, H);

    // Identical images must score exactly 100.0
    let src = RgbSlice::new(&source, W, H);
    let identical = z.compute(&src, &src).expect("compute failed");
    println!(
        "  identical: score={:.15} raw_dist={:.15e}",
        identical.score, identical.raw_distance,
    );
    assert_eq!(
        identical.score, 100.0,
        "Identical images must score exactly 100.0, got {:.15} (raw_dist={:.15e})",
        identical.score, identical.raw_distance,
    );

    // Light blur → < 100
    let light_blur = distort_blur(&source, W, H, 1);
    let dst = RgbSlice::new(&light_blur, W, H);
    let light_result = z.compute(&src, &dst).expect("compute failed");
    assert!(
        light_result.score < 100.0,
        "Light blur should score < 100, got {}",
        light_result.score,
    );
    println!("  light blur (r=1): {:.6}", light_result.score);

    // Heavy blur → lower than light blur
    let heavy_blur = distort_blur(&source, W, H, 5);
    let dst = RgbSlice::new(&heavy_blur, W, H);
    let heavy_result = z.compute(&src, &dst).expect("compute failed");
    assert!(
        heavy_result.score < light_result.score,
        "Heavy blur ({:.4}) should be lower than light blur ({:.4})",
        heavy_result.score,
        light_result.score,
    );
    println!("  heavy blur (r=5): {:.6}", heavy_result.score);

    // All scores in [0, 100]
    let pairs = generate_test_pairs(W, H);
    for pair in &pairs {
        let src = RgbSlice::new(&pair.source, W, H);
        let dst = RgbSlice::new(&pair.distorted, W, H);
        let result = z.compute(&src, &dst).expect("compute failed");
        assert!(
            (0.0..=100.0).contains(&result.score),
            "{}: score {:.4} outside [0, 100]",
            pair.name,
            result.score,
        );
    }
    println!("  All scores in [0, 100] range");
}

/// Same computation 3x → bit-exact score, raw_distance, and features.
#[test]
fn determinism_same_platform() {
    const W: usize = 128;
    const H: usize = 128;
    let z = Zensim::new(ZensimProfile::latest());
    let pairs = generate_test_pairs(W, H);

    for pair in &pairs {
        let src = RgbSlice::new(&pair.source, W, H);
        let dst = RgbSlice::new(&pair.distorted, W, H);

        let r1 = z.compute(&src, &dst).expect("compute 1 failed");
        let r2 = z.compute(&src, &dst).expect("compute 2 failed");
        let r3 = z.compute(&src, &dst).expect("compute 3 failed");

        // Scores must be bit-exact
        assert_eq!(
            r1.score.to_bits(),
            r2.score.to_bits(),
            "{}: score not deterministic (run 1 vs 2): {} vs {}",
            pair.name,
            r1.score,
            r2.score,
        );
        assert_eq!(
            r1.score.to_bits(),
            r3.score.to_bits(),
            "{}: score not deterministic (run 1 vs 3): {} vs {}",
            pair.name,
            r1.score,
            r3.score,
        );

        // raw_distance must be bit-exact
        assert_eq!(
            r1.raw_distance.to_bits(),
            r2.raw_distance.to_bits(),
            "{}: raw_distance not deterministic",
            pair.name,
        );

        // All features must be bit-exact
        for (i, ((f1, f2), f3)) in r1
            .features
            .iter()
            .zip(r2.features.iter())
            .zip(r3.features.iter())
            .enumerate()
        {
            assert_eq!(
                f1.to_bits(),
                f2.to_bits(),
                "{}: feature[{i}] not deterministic (run 1 vs 2)",
                pair.name,
            );
            assert_eq!(
                f1.to_bits(),
                f3.to_bits(),
                "{}: feature[{i}] not deterministic (run 1 vs 3)",
                pair.name,
            );
        }
    }
    println!("  All 8 pairs × 3 runs bit-exact");
}

/// Identical images must score exactly 100.0 with raw_distance=0.0 and all-zero features.
/// Tests all 4 generators with separately-allocated copies (not same pointer).
#[test]
fn identical_images_score_100() {
    const W: usize = 128;
    const H: usize = 128;
    let z = Zensim::new(ZensimProfile::latest());

    let images: &[(&str, Vec<[u8; 3]>)] = &[
        ("checkerboard", gen_checkerboard(W, H, 8)),
        ("mandelbrot", gen_mandelbrot(W, H)),
        ("value_noise", gen_value_noise(W, H, 42)),
        ("color_blocks", gen_color_blocks(W, H)),
    ];

    for (name, pixels) in images {
        // Separate copy so we're not relying on pointer identity
        let copy = pixels.clone();
        let src = RgbSlice::new(pixels, W, H);
        let dst = RgbSlice::new(&copy, W, H);
        let result = z.compute(&src, &dst).expect("compute failed");

        println!(
            "  {name:20} score={:.15}  raw_dist={:.2e}  max_feat={:.2e}",
            result.score,
            result.raw_distance,
            result
                .features
                .iter()
                .map(|f| f.abs())
                .fold(0.0f64, f64::max),
        );
        assert_eq!(
            result.score, 100.0,
            "{name}: identical images must score exactly 100.0, got {:.15}",
            result.score,
        );
        assert_eq!(
            result.raw_distance, 0.0,
            "{name}: identical images must have raw_distance=0.0, got {:.2e}",
            result.raw_distance,
        );
        assert!(
            result.features.iter().all(|&f| f == 0.0),
            "{name}: identical images must have all-zero features",
        );
    }
}
