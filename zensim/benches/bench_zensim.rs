use criterion::{Criterion, criterion_group, criterion_main};

fn bench_zensim_512x512(c: &mut Criterion) {
    // Create synthetic test images
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

    c.bench_function("zensim_512x512", |b| {
        b.iter(|| {
            zensim::compute_zensim(
                std::hint::black_box(&src),
                std::hint::black_box(&dst),
                width,
                height,
            )
            .unwrap()
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

    c.bench_function("zensim_256x256", |b| {
        b.iter(|| {
            zensim::compute_zensim(
                std::hint::black_box(&src),
                std::hint::black_box(&dst),
                width,
                height,
            )
            .unwrap()
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

    c.bench_function("zensim_320x240", |b| {
        b.iter(|| {
            zensim::compute_zensim(
                std::hint::black_box(&src),
                std::hint::black_box(&dst),
                width,
                height,
            )
            .unwrap()
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

    c.bench_function("zensim_1920x1080", |b| {
        b.iter(|| {
            zensim::compute_zensim(
                std::hint::black_box(&src),
                std::hint::black_box(&dst),
                width,
                height,
            )
            .unwrap()
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

    c.bench_function("zensim_3840x2160", |b| {
        b.iter(|| {
            zensim::compute_zensim(
                std::hint::black_box(&src),
                std::hint::black_box(&dst),
                width,
                height,
            )
            .unwrap()
        })
    });
}

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

    let config = zensim::ZensimConfig {
        masking_strength: 4.0,
        compute_all_features: true,
        ..Default::default()
    };

    c.bench_function("zensim_512x512_masked", |b| {
        b.iter(|| {
            zensim::compute_zensim_with_config(
                std::hint::black_box(&src),
                std::hint::black_box(&dst),
                width,
                height,
                config.clone(),
            )
            .unwrap()
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

    c.bench_function("zensim_500x375", |b| {
        b.iter(|| {
            zensim::compute_zensim(
                std::hint::black_box(&src),
                std::hint::black_box(&dst),
                width,
                height,
            )
            .unwrap()
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

    c.bench_function("precompute_ref_512x512", |b| {
        b.iter(|| zensim::precompute_reference(std::hint::black_box(&src), 512, 512).unwrap())
    });

    let pre = zensim::precompute_reference(&src, 512, 512).unwrap();
    c.bench_function("with_ref_512x512", |b| {
        b.iter(|| {
            zensim::compute_zensim_with_ref(
                std::hint::black_box(&pre),
                std::hint::black_box(&dst),
                512,
                512,
            )
            .unwrap()
        })
    });
}

fn bench_precomputed_1280x720(c: &mut Criterion) {
    let (src, dst) = make_test_images(1280, 720);

    c.bench_function("zensim_1280x720", |b| {
        b.iter(|| {
            zensim::compute_zensim(
                std::hint::black_box(&src),
                std::hint::black_box(&dst),
                1280,
                720,
            )
            .unwrap()
        })
    });

    c.bench_function("precompute_ref_1280x720", |b| {
        b.iter(|| zensim::precompute_reference(std::hint::black_box(&src), 1280, 720).unwrap())
    });

    let pre = zensim::precompute_reference(&src, 1280, 720).unwrap();
    c.bench_function("with_ref_1280x720", |b| {
        b.iter(|| {
            zensim::compute_zensim_with_ref(
                std::hint::black_box(&pre),
                std::hint::black_box(&dst),
                1280,
                720,
            )
            .unwrap()
        })
    });
}

fn bench_precomputed_1920x1080(c: &mut Criterion) {
    let (src, dst) = make_test_images(1920, 1080);

    c.bench_function("precompute_ref_1920x1080", |b| {
        b.iter(|| {
            zensim::precompute_reference(std::hint::black_box(&src), 1920, 1080).unwrap()
        })
    });

    let pre = zensim::precompute_reference(&src, 1920, 1080).unwrap();
    c.bench_function("with_ref_1920x1080", |b| {
        b.iter(|| {
            zensim::compute_zensim_with_ref(
                std::hint::black_box(&pre),
                std::hint::black_box(&dst),
                1920,
                1080,
            )
            .unwrap()
        })
    });
}

fn bench_precomputed_3840x2160(c: &mut Criterion) {
    let (src, dst) = make_test_images(3840, 2160);

    c.bench_function("precompute_ref_3840x2160", |b| {
        b.iter(|| {
            zensim::precompute_reference(std::hint::black_box(&src), 3840, 2160).unwrap()
        })
    });

    let pre = zensim::precompute_reference(&src, 3840, 2160).unwrap();
    c.bench_function("with_ref_3840x2160", |b| {
        b.iter(|| {
            zensim::compute_zensim_with_ref(
                std::hint::black_box(&pre),
                std::hint::black_box(&dst),
                3840,
                2160,
            )
            .unwrap()
        })
    });
}

fn bench_precomputed_7680x4320(c: &mut Criterion) {
    let (src, dst) = make_test_images(7680, 4320);

    c.bench_function("zensim_7680x4320", |b| {
        b.iter(|| {
            zensim::compute_zensim(
                std::hint::black_box(&src),
                std::hint::black_box(&dst),
                7680,
                4320,
            )
            .unwrap()
        })
    });

    c.bench_function("precompute_ref_7680x4320", |b| {
        b.iter(|| {
            zensim::precompute_reference(std::hint::black_box(&src), 7680, 4320).unwrap()
        })
    });

    let pre = zensim::precompute_reference(&src, 7680, 4320).unwrap();
    c.bench_function("with_ref_7680x4320", |b| {
        b.iter(|| {
            zensim::compute_zensim_with_ref(
                std::hint::black_box(&pre),
                std::hint::black_box(&dst),
                7680,
                4320,
            )
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
criterion_group!(training_benches, bench_zensim_512x512_masked,);

#[cfg(not(feature = "training"))]
criterion_main!(benches);
#[cfg(feature = "training")]
criterion_main!(benches, training_benches);
