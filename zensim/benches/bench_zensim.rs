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

criterion_group!(
    benches,
    bench_zensim_512x512,
    bench_zensim_256x256,
    bench_zensim_320x240,
    bench_zensim_1920x1080,
    bench_zensim_3840x2160,
);
criterion_main!(benches);
