//! Benchmarks for dimension-mismatch scoring and transform detection.

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
        for &(label, w, h) in &[
            ("256x256", 256u32, 256u32),
            ("600x450", 600, 450),
            ("3840x2160", 3840, 2160),
        ] {
            let z = Zensim::new(ZensimProfile::latest());
            let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
            let src = gradient_rgba(w, h);
            let dst = gradient_rgba(w, h);
            let src_px = px(&src);
            let dst_px = px(&dst);

            group.bench(label, move |b| {
                b.iter(|| {
                    let s = RgbaSlice::new(std::hint::black_box(&src_px), w as usize, h as usize);
                    let d = RgbaSlice::new(std::hint::black_box(&dst_px), w as usize, h as usize);
                    check_regression(&z, &s, &d, &tol).unwrap()
                })
            });
        }
    });
}

fn bench_detect_unrelated(suite: &mut zenbench::Suite) {
    suite.group("detect_transform_unrelated", |group| {
        for &(label, w, h) in &[
            ("256x256", 256u32, 256u32),
            ("600x450", 600, 450),
            ("3840x2160", 3840, 2160),
        ] {
            let z = Zensim::new(ZensimProfile::latest());
            let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
            let exp = gradient_rgba(w, h);
            let act = noise_rgba(w, h, 42);
            group.bench(label, move |b| {
                b.iter(|| {
                    detect_transform(
                        &z,
                        std::hint::black_box(&exp),
                        std::hint::black_box(&act),
                        w,
                        h,
                        -100.0,
                        &tol,
                    )
                })
            });
        }
    });
}

fn bench_detect_flipped(suite: &mut zenbench::Suite) {
    suite.group("detect_transform_flipped", |group| {
        for &(label, w, h) in &[
            ("256x256", 256u32, 256u32),
            ("600x450", 600, 450),
            ("3840x2160", 3840, 2160),
        ] {
            let z = Zensim::new(ZensimProfile::latest());
            let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
            let exp = gradient_rgba(w, h);
            let img = image::RgbaImage::from_raw(w, h, exp.clone()).unwrap();
            let flipped = image::imageops::flip_horizontal(&img).into_raw();
            group.bench(label, move |b| {
                b.iter(|| {
                    detect_transform(
                        &z,
                        std::hint::black_box(&exp),
                        std::hint::black_box(&flipped),
                        w,
                        h,
                        -100.0,
                        &tol,
                    )
                })
            });
        }
    });
}

fn bench_resized_orientation(suite: &mut zenbench::Suite) {
    suite.group("resized_orientation_swap", |group| {
        for &(label, w, h) in &[
            ("256x384", 256u32, 384u32),
            ("600x450", 600, 450),
            ("2160x3840", 2160, 3840),
        ] {
            let z = Zensim::new(ZensimProfile::latest());
            let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
            let exp = gradient_rgba(w, h);
            let img = image::RgbaImage::from_raw(w, h, exp.clone()).unwrap();
            let rotated = image::imageops::rotate90(&img);
            let (rw, rh) = rotated.dimensions();
            let rot_rgba = rotated.into_raw();
            group.bench(label, move |b| {
                b.iter(|| {
                    check_regression_resized(
                        &z,
                        std::hint::black_box(&exp),
                        w,
                        h,
                        std::hint::black_box(&rot_rgba),
                        rw,
                        rh,
                        &tol,
                    )
                    .unwrap()
                })
            });
        }
    });
}

fn bench_resized_off_by_one(suite: &mut zenbench::Suite) {
    suite.group("resized_off_by_one", |group| {
        for &(label, w, h) in &[
            ("256x256", 256u32, 256u32),
            ("600x450", 600, 450),
            ("3840x2160", 3840, 2160),
        ] {
            let z = Zensim::new(ZensimProfile::latest());
            let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
            let exp = gradient_rgba(w, h);
            let act = gradient_rgba(w + 1, h - 1);
            group.bench(label, move |b| {
                b.iter(|| {
                    check_regression_resized(
                        &z,
                        std::hint::black_box(&exp),
                        w,
                        h,
                        std::hint::black_box(&act),
                        w + 1,
                        h - 1,
                        &tol,
                    )
                    .unwrap()
                })
            });
        }
    });
}

fn bench_resized_large(suite: &mut zenbench::Suite) {
    suite.group("resized_large_diff", |group| {
        for &(label, ew, eh, aw, ah) in &[
            ("600x450_vs_300x200", 600u32, 450u32, 300u32, 200u32),
            ("3840x2160_vs_1920x1080", 3840, 2160, 1920, 1080),
        ] {
            let z = Zensim::new(ZensimProfile::latest());
            let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);
            let exp = gradient_rgba(ew, eh);
            let act = noise_rgba(aw, ah, 99);
            group.bench(label, move |b| {
                b.iter(|| {
                    check_regression_resized(
                        &z,
                        std::hint::black_box(&exp),
                        ew,
                        eh,
                        std::hint::black_box(&act),
                        aw,
                        ah,
                        &tol,
                    )
                    .unwrap()
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
