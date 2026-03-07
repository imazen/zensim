//! Comparison benchmarks: zensim vs ssimulacra2 and butteraugli implementations.
//!
//! Benchmark groups, deliberately separated for fair comparison:
//!
//! - `compute`: pure in-memory computation, no I/O. Apples-to-apples.
//!   Includes C++ ssimulacra2/butteraugli via FFI when built with libjxl (see build.rs).
//! - `end_to_end`: includes process spawn + PPM read. Only for the C++ binary
//!   when FFI is not available. Set `SSIMULACRA2_BIN` to enable.

use criterion::{Criterion, criterion_group, criterion_main};
use imgref::Img;
use rgb::RGB8;
use zensim::{RgbSlice, Zensim, ZensimProfile};

#[cfg(has_cpp_ssimulacra2)]
unsafe extern "C" {
    fn ssimulacra2_from_srgb(
        src_rgb: *const u8,
        dst_rgb: *const u8,
        width: usize,
        height: usize,
    ) -> f64;
}

#[cfg(has_cpp_butteraugli)]
unsafe extern "C" {
    fn butteraugli_from_linear_planes(
        src0: *const f32,
        src1: *const f32,
        src2: *const f32,
        dst0: *const f32,
        dst1: *const f32,
        dst2: *const f32,
        width: usize,
        height: usize,
    ) -> f64;
}

const SIZES: &[(&str, usize, usize)] = &[
    ("512x512", 512, 512),
    ("1280x720", 1280, 720),
    ("1920x1080", 1920, 1080),
    ("2560x1440", 2560, 1440),
    ("3840x2160", 3840, 2160),
];

fn make_test_images(width: usize, height: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let n = width * height;
    let src: Vec<[u8; 3]> = (0..n)
        .map(|i| {
            let x = ((i % width) * 255 / width) as u8;
            let y = ((i / width) * 255 / height) as u8;
            [x, y, x.wrapping_add(y)]
        })
        .collect();
    let dst: Vec<[u8; 3]> = src
        .iter()
        .map(|&[r, g, b]| [r.saturating_add(5), g.saturating_add(3), b])
        .collect();
    (src, dst)
}

fn make_f32_srgb(pixels: &[[u8; 3]]) -> Vec<[f32; 3]> {
    pixels
        .iter()
        .map(|&[r, g, b]| [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0])
        .collect()
}

/// Convert packed sRGB u8 to 3 planar linear-light f32 buffers.
fn srgb_to_linear_planes(pixels: &[[u8; 3]]) -> [Vec<f32>; 3] {
    let n = pixels.len();
    let mut p0 = Vec::with_capacity(n);
    let mut p1 = Vec::with_capacity(n);
    let mut p2 = Vec::with_capacity(n);
    for &[r, g, b] in pixels {
        p0.push(linear_srgb::default::srgb_u8_to_linear(r));
        p1.push(linear_srgb::default::srgb_u8_to_linear(g));
        p2.push(linear_srgb::default::srgb_u8_to_linear(b));
    }
    [p0, p1, p2]
}

/// Pure computation benchmarks — no I/O, no process spawning.
/// All implementations receive pre-allocated pixel buffers in memory.
fn bench_compute(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute");

    for &(label, w, h) in SIZES {
        let (src, dst) = make_test_images(w, h);

        // zensim (multi-threaded, default)
        let z = Zensim::new(ZensimProfile::latest());
        group.bench_function(format!("zensim/{label}"), |b| {
            b.iter(|| {
                let s = RgbSlice::new(std::hint::black_box(&src), w, h);
                let d = RgbSlice::new(std::hint::black_box(&dst), w, h);
                z.compute(&s, &d).unwrap()
            })
        });

        // zensim (single-threaded)
        let z_st = Zensim::new(ZensimProfile::latest()).with_parallel(false);
        group.bench_function(format!("zensim_st/{label}"), |b| {
            b.iter(|| {
                let s = RgbSlice::new(std::hint::black_box(&src), w, h);
                let d = RgbSlice::new(std::hint::black_box(&dst), w, h);
                z_st.compute(&s, &d).unwrap()
            })
        });

        // C++ ssimulacra2 via FFI (libjxl, the reference implementation)
        #[cfg(has_cpp_ssimulacra2)]
        {
            let src_flat: &[u8] = bytemuck::cast_slice(&src);
            let dst_flat: &[u8] = bytemuck::cast_slice(&dst);
            group.bench_function(format!("cpp_ssimulacra2/{label}"), |b| {
                b.iter(|| {
                    let score = unsafe {
                        ssimulacra2_from_srgb(
                            std::hint::black_box(src_flat.as_ptr()),
                            std::hint::black_box(dst_flat.as_ptr()),
                            w,
                            h,
                        )
                    };
                    assert!(score > -900.0, "ssimulacra2 FFI failed: {score}");
                    score
                })
            });
        }

        // ssimulacra2 (rust-av, faithful port of C++ libjxl)
        {
            use ssimulacra2::{
                ColorPrimaries, Rgb, TransferCharacteristic, compute_frame_ssimulacra2,
            };
            let src_f32 = make_f32_srgb(&src);
            let dst_f32 = make_f32_srgb(&dst);
            group.bench_function(format!("ssimulacra2_rs/{label}"), |b| {
                b.iter(|| {
                    let s = Rgb::new(
                        std::hint::black_box(src_f32.clone()),
                        w,
                        h,
                        TransferCharacteristic::SRGB,
                        ColorPrimaries::BT709,
                    )
                    .unwrap();
                    let d = Rgb::new(
                        std::hint::black_box(dst_f32.clone()),
                        w,
                        h,
                        TransferCharacteristic::SRGB,
                        ColorPrimaries::BT709,
                    )
                    .unwrap();
                    compute_frame_ssimulacra2(s, d).unwrap()
                })
            });
        }

        // fast-ssim2 (imazen, SIMD-accelerated ssimulacra2)
        group.bench_function(format!("fast_ssim2/{label}"), |b| {
            b.iter(|| {
                let s = Img::new(std::hint::black_box(src.as_slice()), w, h);
                let d = Img::new(std::hint::black_box(dst.as_slice()), w, h);
                fast_ssim2::compute_ssimulacra2(s, d).unwrap()
            })
        });

        // C++ butteraugli via FFI (libjxl, the reference implementation)
        #[cfg(has_cpp_butteraugli)]
        {
            let src_lin = srgb_to_linear_planes(&src);
            let dst_lin = srgb_to_linear_planes(&dst);
            group.bench_function(format!("cpp_butteraugli/{label}"), |b| {
                b.iter(|| {
                    let score = unsafe {
                        butteraugli_from_linear_planes(
                            std::hint::black_box(src_lin[0].as_ptr()),
                            std::hint::black_box(src_lin[1].as_ptr()),
                            std::hint::black_box(src_lin[2].as_ptr()),
                            std::hint::black_box(dst_lin[0].as_ptr()),
                            std::hint::black_box(dst_lin[1].as_ptr()),
                            std::hint::black_box(dst_lin[2].as_ptr()),
                            w,
                            h,
                        )
                    };
                    assert!(score > -900.0, "butteraugli FFI failed: {score}");
                    score
                })
            });
        }

        // butteraugli (imazen, pure Rust port of libjxl butteraugli)
        {
            let src_rgb8: &[RGB8] = bytemuck::cast_slice(&src);
            let dst_rgb8: &[RGB8] = bytemuck::cast_slice(&dst);
            group.bench_function(format!("butteraugli_rs/{label}"), |b| {
                b.iter(|| {
                    let s = Img::new(std::hint::black_box(src_rgb8), w, h);
                    let d = Img::new(std::hint::black_box(dst_rgb8), w, h);
                    butteraugli::butteraugli(s, d, &butteraugli::ButteraugliParams::default())
                        .unwrap()
                })
            });
        }
    }

    group.finish();
}

fn write_ppm(pixels: &[[u8; 3]], w: usize, h: usize, path: &std::path::Path) {
    use zenpnm::Unstoppable;
    let flat: &[u8] = bytemuck::cast_slice(pixels);
    let encoded = zenpnm::encode_ppm(
        flat,
        w as u32,
        h as u32,
        zenpnm::PixelLayout::Rgb8,
        Unstoppable,
    )
    .expect("PPM encode failed");
    std::fs::write(path, &encoded).expect("write PPM failed");
}

fn find_ssimulacra2_bin() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("SSIMULACRA2_BIN") {
        let path = std::path::PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    for candidate in ["/usr/local/bin/ssimulacra2", "/usr/bin/ssimulacra2"] {
        let path = std::path::PathBuf::from(candidate);
        if path.exists() {
            return Some(path);
        }
    }
    which::which("ssimulacra2").ok()
}

/// End-to-end benchmarks — includes process spawn + PPM file I/O.
///
/// These numbers are NOT directly comparable to the `compute` group.
/// Measures total wall time including process startup, PPM decode, computation,
/// and result output. Set SSIMULACRA2_BIN env var to enable.
fn bench_end_to_end(c: &mut Criterion) {
    let Some(bin) = find_ssimulacra2_bin() else {
        eprintln!(
            "note: C++ ssimulacra2 binary not found. Set SSIMULACRA2_BIN to enable. \
             Skipping end-to-end benchmarks."
        );
        return;
    };

    let tmpdir = std::env::temp_dir().join("zensim_bench_compare");
    std::fs::create_dir_all(&tmpdir).ok();

    let mut group = c.benchmark_group("end_to_end");

    for &(label, w, h) in SIZES {
        let (src, dst) = make_test_images(w, h);
        let src_path = tmpdir.join(format!("src_{label}.ppm"));
        let dst_path = tmpdir.join(format!("dst_{label}.ppm"));
        write_ppm(&src, w, h, &src_path);
        write_ppm(&dst, w, h, &dst_path);

        group.bench_function(format!("cpp_ssimulacra2/{label}"), |b| {
            b.iter(|| {
                let output = std::process::Command::new(&bin)
                    .arg(&src_path)
                    .arg(&dst_path)
                    .output()
                    .expect("failed to run ssimulacra2");
                assert!(output.status.success(), "ssimulacra2 failed");
            })
        });
    }

    group.finish();
    std::fs::remove_dir_all(&tmpdir).ok();
}

criterion_group!(compute, bench_compute);
criterion_group!(end_to_end, bench_end_to_end);
criterion_main!(compute, end_to_end);
