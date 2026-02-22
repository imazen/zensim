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
                criterion::black_box(&src),
                criterion::black_box(&dst),
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
                criterion::black_box(&src),
                criterion::black_box(&dst),
                width,
                height,
            )
            .unwrap()
        })
    });
}

criterion_group!(benches, bench_zensim_512x512, bench_zensim_256x256);
criterion_main!(benches);
