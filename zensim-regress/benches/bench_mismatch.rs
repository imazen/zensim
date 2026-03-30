//! Benchmarks for dimension-mismatch scoring and transform detection.
//!
//! Tracks the overhead of `check_regression_resized` and `detect_transform`
//! across different mismatch categories and image sizes to prevent regressions.

use zensim::{RgbaSlice, Zensim, ZensimProfile};
use zensim_regress::testing::{
    RegressionTolerance, check_regression, check_regression_resized, detect_transform,
};

fn gradient_rgba(w: u32, h: u32) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255) / w.max(1)) as u8;
            let g = ((y * 127) / h.max(1)) as u8;
            let b = (((x + y * 2) * 63) / (w + h).max(1)) as u8;
            rgba.extend_from_slice(&[r, g, b, 255]);
        }
    }
    rgba
}

fn noise_rgba(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    let mut s = seed;
    for _ in 0..(w * h) {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        let r = (s >> 16) as u8;
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        let g = (s >> 16) as u8;
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        let b = (s >> 16) as u8;
        rgba.extend_from_slice(&[r, g, b, 255]);
    }
    rgba
}

fn px(rgba: &[u8]) -> Vec<[u8; 4]> {
    rgba.chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect()
}

fn bench_baseline(suite: &mut zenbench::Suite) {
    suite.group("check_regression_baseline", |group| {
        let z = Zensim::new(ZensimProfile::latest());
        let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
        let src = gradient_rgba(256, 256);
        let dst = gradient_rgba(256, 256);
        let src_px = px(&src);
        let dst_px = px(&dst);

        group.bench("256x256_identical", move |b| {
            b.iter(|| {
                let s = RgbaSlice::new(std::hint::black_box(&src_px), 256, 256);
                let d = RgbaSlice::new(std::hint::black_box(&dst_px), 256, 256);
                check_regression(&z, &s, &d, &tol).unwrap()
            })
        });
    });
}

fn bench_detect_unrelated(suite: &mut zenbench::Suite) {
    suite.group("detect_transform_unrelated", |group| {
        // Worst case: completely different images, should exit fast
        {
            let z = Zensim::new(ZensimProfile::latest());
            let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
            let exp = gradient_rgba(256, 256);
            let act = noise_rgba(256, 256, 42);
            group.bench("256x256", move |b| {
                b.iter(|| {
                    detect_transform(
                        &z,
                        std::hint::black_box(&exp),
                        std::hint::black_box(&act),
                        256, 256, -100.0, &tol,
                    )
                })
            });
        }
        {
            let z = Zensim::new(ZensimProfile::latest());
            let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
            let exp = gradient_rgba(600, 450);
            let act = noise_rgba(600, 450, 42);
            group.bench("600x450", move |b| {
                b.iter(|| {
                    detect_transform(
                        &z,
                        std::hint::black_box(&exp),
                        std::hint::black_box(&act),
                        600, 450, -100.0, &tol,
                    )
                })
            });
        }
    });
}

fn bench_detect_flipped(suite: &mut zenbench::Suite) {
    suite.group("detect_transform_flipped", |group| {
        {
            let z = Zensim::new(ZensimProfile::latest());
            let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
            let exp = gradient_rgba(256, 256);
            let img = image::RgbaImage::from_raw(256, 256, exp.clone()).unwrap();
            let flipped = image::imageops::flip_horizontal(&img).into_raw();
            group.bench("256x256_hflip", move |b| {
                b.iter(|| {
                    detect_transform(
                        &z,
                        std::hint::black_box(&exp),
                        std::hint::black_box(&flipped),
                        256, 256, -100.0, &tol,
                    )
                })
            });
        }
        {
            let z = Zensim::new(ZensimProfile::latest());
            let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
            let exp = gradient_rgba(600, 450);
            let img = image::RgbaImage::from_raw(600, 450, exp.clone()).unwrap();
            let flipped = image::imageops::flip_horizontal(&img).into_raw();
            group.bench("600x450_hflip", move |b| {
                b.iter(|| {
                    detect_transform(
                        &z,
                        std::hint::black_box(&exp),
                        std::hint::black_box(&flipped),
                        600, 450, -100.0, &tol,
                    )
                })
            });
        }
    });
}

fn bench_resized_orientation(suite: &mut zenbench::Suite) {
    suite.group("resized_orientation_swap", |group| {
        {
            let z = Zensim::new(ZensimProfile::latest());
            let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
            let exp = gradient_rgba(256, 384);
            let img = image::RgbaImage::from_raw(256, 384, exp.clone()).unwrap();
            let rotated = image::imageops::rotate90(&img);
            let (rw, rh) = rotated.dimensions();
            let rot_rgba = rotated.into_raw();
            group.bench("256x384_rot90", move |b| {
                b.iter(|| {
                    check_regression_resized(
                        &z,
                        std::hint::black_box(&exp), 256, 384,
                        std::hint::black_box(&rot_rgba), rw, rh,
                        &tol,
                    ).unwrap()
                })
            });
        }
        {
            let z = Zensim::new(ZensimProfile::latest());
            let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
            let exp = gradient_rgba(600, 450);
            let img = image::RgbaImage::from_raw(600, 450, exp.clone()).unwrap();
            let rotated = image::imageops::rotate90(&img);
            let (rw, rh) = rotated.dimensions();
            let rot_rgba = rotated.into_raw();
            group.bench("600x450_rot90", move |b| {
                b.iter(|| {
                    check_regression_resized(
                        &z,
                        std::hint::black_box(&exp), 600, 450,
                        std::hint::black_box(&rot_rgba), rw, rh,
                        &tol,
                    ).unwrap()
                })
            });
        }
    });
}

fn bench_resized_off_by_one(suite: &mut zenbench::Suite) {
    suite.group("resized_off_by_one", |group| {
        {
            let z = Zensim::new(ZensimProfile::latest());
            let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
            let exp = gradient_rgba(256, 256);
            let act = gradient_rgba(257, 255);
            group.bench("256x256_vs_257x255", move |b| {
                b.iter(|| {
                    check_regression_resized(
                        &z,
                        std::hint::black_box(&exp), 256, 256,
                        std::hint::black_box(&act), 257, 255,
                        &tol,
                    ).unwrap()
                })
            });
        }
    });
}

fn bench_resized_large(suite: &mut zenbench::Suite) {
    suite.group("resized_large_diff_unrelated", |group| {
        {
            let z = Zensim::new(ZensimProfile::latest());
            let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
            let exp = gradient_rgba(600, 450);
            let act = noise_rgba(300, 200, 99);
            group.bench("600x450_vs_300x200", move |b| {
                b.iter(|| {
                    check_regression_resized(
                        &z,
                        std::hint::black_box(&exp), 600, 450,
                        std::hint::black_box(&act), 300, 200,
                        &tol,
                    ).unwrap()
                })
            });
        }
    });
}

zenbench::main!(
    bench_baseline,
    bench_detect_unrelated,
    bench_detect_flipped,
    bench_resized_orientation,
    bench_resized_off_by_one,
    bench_resized_large,
);
