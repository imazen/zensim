use criterion::{Criterion, criterion_group, criterion_main};
use zensim::{RgbSlice, Zensim, ZensimProfile};

fn bench_zensim_512x512(c: &mut Criterion) {
    let width = 512;
    let height = 512;
    let n = width * height;

    let src: Vec<[u8; 3]> = (0..n)
        .map(|i| {
            let x = (i % width) as u8;
            let y = (i / width) as u8;
            [x, y, x.wrapping_add(y)]
        })
        .collect();

    let dst: Vec<[u8; 3]> = src
        .iter()
        .map(|&[r, g, b]| [r.saturating_add(10), g.saturating_add(5), b])
        .collect();

    let z = Zensim::new(ZensimProfile::latest());
    c.bench_function("zensim_512x512", |b| {
        b.iter(|| {
            let s = RgbSlice::new(std::hint::black_box(&src), width, height);
            let d = RgbSlice::new(std::hint::black_box(&dst), width, height);
            z.compute(&s, &d).unwrap()
        })
    });
}

fn bench_zensim_256x256(c: &mut Criterion) {
    let width = 256;
    let height = 256;
    let n = width * height;

    let src: Vec<[u8; 3]> = (0..n)
        .map(|i| {
            let x = (i % width) as u8;
            let y = (i / width) as u8;
            [x, y, x.wrapping_add(y)]
        })
        .collect();

    let dst: Vec<[u8; 3]> = src
        .iter()
        .map(|&[r, g, b]| [r.saturating_add(20), g.saturating_add(10), b])
        .collect();

    let z = Zensim::new(ZensimProfile::latest());
    c.bench_function("zensim_256x256", |b| {
        b.iter(|| {
            let s = RgbSlice::new(std::hint::black_box(&src), width, height);
            let d = RgbSlice::new(std::hint::black_box(&dst), width, height);
            z.compute(&s, &d).unwrap()
        })
    });
}

fn bench_zensim_320x240(c: &mut Criterion) {
    let width = 320;
    let height = 240;
    let n = width * height;

    let src: Vec<[u8; 3]> = (0..n)
        .map(|i| {
            let x = (i % width) as u8;
            let y = (i / width) as u8;
            [x, y, x.wrapping_add(y)]
        })
        .collect();

    let dst: Vec<[u8; 3]> = src
        .iter()
        .map(|&[r, g, b]| [r.saturating_add(15), g.saturating_add(8), b])
        .collect();

    let z = Zensim::new(ZensimProfile::latest());
    c.bench_function("zensim_320x240", |b| {
        b.iter(|| {
            let s = RgbSlice::new(std::hint::black_box(&src), width, height);
            let d = RgbSlice::new(std::hint::black_box(&dst), width, height);
            z.compute(&s, &d).unwrap()
        })
    });
}

fn bench_zensim_1920x1080(c: &mut Criterion) {
    let width = 1920;
    let height = 1080;
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
        .map(|&[r, g, b]| [r.saturating_add(8), g.saturating_add(4), b])
        .collect();

    let z = Zensim::new(ZensimProfile::latest());
    c.bench_function("zensim_1920x1080", |b| {
        b.iter(|| {
            let s = RgbSlice::new(std::hint::black_box(&src), width, height);
            let d = RgbSlice::new(std::hint::black_box(&dst), width, height);
            z.compute(&s, &d).unwrap()
        })
    });
}

fn bench_zensim_3840x2160(c: &mut Criterion) {
    let width = 3840;
    let height = 2160;
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

    let z = Zensim::new(ZensimProfile::latest());
    c.bench_function("zensim_3840x2160", |b| {
        b.iter(|| {
            let s = RgbSlice::new(std::hint::black_box(&src), width, height);
            let d = RgbSlice::new(std::hint::black_box(&dst), width, height);
            z.compute(&s, &d).unwrap()
        })
    });
}

fn bench_zensim_500x375(c: &mut Criterion) {
    let width = 500;
    let height = 375;
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
        .map(|&[r, g, b]| [r.saturating_add(12), g.saturating_add(6), b])
        .collect();

    let z = Zensim::new(ZensimProfile::latest());
    c.bench_function("zensim_500x375", |b| {
        b.iter(|| {
            let s = RgbSlice::new(std::hint::black_box(&src), width, height);
            let d = RgbSlice::new(std::hint::black_box(&dst), width, height);
            z.compute(&s, &d).unwrap()
        })
    });
}

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

fn bench_precomputed_512x512(c: &mut Criterion) {
    let (src, dst) = make_test_images(512, 512);
    let z = Zensim::new(ZensimProfile::latest());

    c.bench_function("precompute_ref_512x512", |b| {
        let s = RgbSlice::new(&src, 512, 512);
        b.iter(|| z.precompute_reference(std::hint::black_box(&s)).unwrap())
    });

    let s = RgbSlice::new(&src, 512, 512);
    let pre = z.precompute_reference(&s).unwrap();
    c.bench_function("with_ref_512x512", |b| {
        let d = RgbSlice::new(&dst, 512, 512);
        b.iter(|| {
            z.compute_with_ref(std::hint::black_box(&pre), std::hint::black_box(&d))
                .unwrap()
        })
    });
}

fn bench_precomputed_1280x720(c: &mut Criterion) {
    let (src, dst) = make_test_images(1280, 720);
    let z = Zensim::new(ZensimProfile::latest());

    c.bench_function("zensim_1280x720", |b| {
        let s = RgbSlice::new(&src, 1280, 720);
        let d = RgbSlice::new(&dst, 1280, 720);
        b.iter(|| {
            z.compute(std::hint::black_box(&s), std::hint::black_box(&d))
                .unwrap()
        })
    });

    c.bench_function("precompute_ref_1280x720", |b| {
        let s = RgbSlice::new(&src, 1280, 720);
        b.iter(|| z.precompute_reference(std::hint::black_box(&s)).unwrap())
    });

    let s = RgbSlice::new(&src, 1280, 720);
    let pre = z.precompute_reference(&s).unwrap();
    c.bench_function("with_ref_1280x720", |b| {
        let d = RgbSlice::new(&dst, 1280, 720);
        b.iter(|| {
            z.compute_with_ref(std::hint::black_box(&pre), std::hint::black_box(&d))
                .unwrap()
        })
    });
}

fn bench_precomputed_1920x1080(c: &mut Criterion) {
    let (src, dst) = make_test_images(1920, 1080);
    let z = Zensim::new(ZensimProfile::latest());

    c.bench_function("precompute_ref_1920x1080", |b| {
        let s = RgbSlice::new(&src, 1920, 1080);
        b.iter(|| z.precompute_reference(std::hint::black_box(&s)).unwrap())
    });

    let s = RgbSlice::new(&src, 1920, 1080);
    let pre = z.precompute_reference(&s).unwrap();
    c.bench_function("with_ref_1920x1080", |b| {
        let d = RgbSlice::new(&dst, 1920, 1080);
        b.iter(|| {
            z.compute_with_ref(std::hint::black_box(&pre), std::hint::black_box(&d))
                .unwrap()
        })
    });
}

fn bench_precomputed_3840x2160(c: &mut Criterion) {
    let (src, dst) = make_test_images(3840, 2160);
    let z = Zensim::new(ZensimProfile::latest());

    c.bench_function("precompute_ref_3840x2160", |b| {
        let s = RgbSlice::new(&src, 3840, 2160);
        b.iter(|| z.precompute_reference(std::hint::black_box(&s)).unwrap())
    });

    let s = RgbSlice::new(&src, 3840, 2160);
    let pre = z.precompute_reference(&s).unwrap();
    c.bench_function("with_ref_3840x2160", |b| {
        let d = RgbSlice::new(&dst, 3840, 2160);
        b.iter(|| {
            z.compute_with_ref(std::hint::black_box(&pre), std::hint::black_box(&d))
                .unwrap()
        })
    });
}

fn bench_precomputed_7680x4320(c: &mut Criterion) {
    let (src, dst) = make_test_images(7680, 4320);
    let z = Zensim::new(ZensimProfile::latest());

    c.bench_function("zensim_7680x4320", |b| {
        let s = RgbSlice::new(&src, 7680, 4320);
        let d = RgbSlice::new(&dst, 7680, 4320);
        b.iter(|| {
            z.compute(std::hint::black_box(&s), std::hint::black_box(&d))
                .unwrap()
        })
    });

    c.bench_function("precompute_ref_7680x4320", |b| {
        let s = RgbSlice::new(&src, 7680, 4320);
        b.iter(|| z.precompute_reference(std::hint::black_box(&s)).unwrap())
    });

    let s = RgbSlice::new(&src, 7680, 4320);
    let pre = z.precompute_reference(&s).unwrap();
    c.bench_function("with_ref_7680x4320", |b| {
        let d = RgbSlice::new(&dst, 7680, 4320);
        b.iter(|| {
            z.compute_with_ref(std::hint::black_box(&pre), std::hint::black_box(&d))
                .unwrap()
        })
    });
}

criterion_group!(
    benches,
    bench_zensim_512x512,
    bench_zensim_256x256,
    bench_zensim_320x240,
    bench_zensim_500x375,
    bench_zensim_1920x1080,
    bench_zensim_3840x2160,
    bench_precomputed_512x512,
    bench_precomputed_1280x720,
    bench_precomputed_1920x1080,
    bench_precomputed_3840x2160,
    bench_precomputed_7680x4320,
);

#[cfg(feature = "training")]
fn bench_zensim_512x512_masked(c: &mut Criterion) {
    let width = 512;
    let height = 512;
    let n = width * height;

    let src: Vec<[u8; 3]> = (0..n)
        .map(|i| {
            let x = (i % width) as u8;
            let y = (i / width) as u8;
            [x, y, x.wrapping_add(y)]
        })
        .collect();

    let dst: Vec<[u8; 3]> = src
        .iter()
        .map(|&[r, g, b]| [r.saturating_add(10), g.saturating_add(5), b])
        .collect();

    let mut config = zensim::ZensimConfig::default();
    config.compute_all_features = true;

    c.bench_function("zensim_512x512_masked", |b| {
        b.iter(|| {
            zensim::compute_zensim_with_config(
                std::hint::black_box(&src),
                std::hint::black_box(&dst),
                width,
                height,
                config,
            )
            .unwrap()
        })
    });
}

#[cfg(feature = "zenresize")]
fn bench_downscale_filters(c: &mut Criterion) {
    use zensim::{DownscaleFilter, ZensimConfig};

    for &(label, w, h) in &[("512x512", 512, 512), ("1920x1080", 1920, 1080)] {
        let (src, dst) = make_test_images(w, h);

        for &(filter_name, filter) in &[
            ("box", DownscaleFilter::Box2x2),
            ("mitchell", DownscaleFilter::Mitchell),
            ("lanczos", DownscaleFilter::Lanczos),
        ] {
            let mut config = ZensimConfig::default();
            config.downscale_filter = filter;
            config.compute_all_features = true;
            c.bench_function(&format!("zensim_{label}_{filter_name}"), |b| {
                b.iter(|| {
                    zensim::compute_zensim_with_config(
                        std::hint::black_box(&src),
                        std::hint::black_box(&dst),
                        w,
                        h,
                        config,
                    )
                    .unwrap()
                })
            });
        }
    }
}

#[cfg(feature = "training")]
fn bench_zensim_extended(c: &mut Criterion) {
    use zensim::ZensimConfig;

    for &(label, w, h) in &[("512x512", 512, 512), ("1920x1080", 1920, 1080)] {
        let (src, dst) = make_test_images(w, h);

        let mut config_basic = ZensimConfig::default();
        config_basic.compute_all_features = true;
        c.bench_function(&format!("basic_{label}"), |b| {
            b.iter(|| {
                zensim::compute_zensim_with_config(
                    std::hint::black_box(&src),
                    std::hint::black_box(&dst),
                    w,
                    h,
                    config_basic,
                )
                .unwrap()
            })
        });

        let mut config_ext = ZensimConfig::default();
        config_ext.compute_all_features = true;
        config_ext.extended_features = true;
        c.bench_function(&format!("extended_{label}"), |b| {
            b.iter(|| {
                zensim::compute_zensim_with_config(
                    std::hint::black_box(&src),
                    std::hint::black_box(&dst),
                    w,
                    h,
                    config_ext,
                )
                .unwrap()
            })
        });
    }
}

#[cfg(feature = "training")]
criterion_group!(
    training_benches,
    bench_zensim_512x512_masked,
    bench_zensim_extended,
);

#[cfg(feature = "zenresize")]
criterion_group!(zenresize_benches, bench_downscale_filters,);

#[cfg(all(not(feature = "training"), not(feature = "zenresize")))]
criterion_main!(benches);
#[cfg(all(feature = "training", not(feature = "zenresize")))]
criterion_main!(benches, training_benches);
#[cfg(all(not(feature = "training"), feature = "zenresize"))]
criterion_main!(benches, zenresize_benches);
#[cfg(all(feature = "training", feature = "zenresize"))]
criterion_main!(benches, training_benches, zenresize_benches);
