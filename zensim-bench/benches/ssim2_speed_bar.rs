// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.
//! **Does SSIMULACRA2 go faster when you give it threads?** — the one question
//! `zensim/benches/extract_paths_bench.rs`'s ssim2 arm cannot answer, because
//! `fast-ssim2`'s parallelism is a cargo feature and cargo features are
//! per-build, not per-arm (`benchmarks/ssim2_replacement_bar_2026-08-31.md`).
//!
//! fast-ssim2 parallelises its Gaussian blur — the dominant kernel — only under
//! its optional `rayon` feature, which is OFF in this crate's default dep line.
//! So: build twice, once with `--features ssim2-rayon`, and read the delta. The
//! `zensim_B` arm is present in BOTH builds purely as the cross-build anchor —
//! it is untouched by the feature, so if it moves, the box moved.
//!
//! zenbench rather than criterion because the two arms must be interleaved:
//! criterion's isolated back-to-back runs bake the box's thermal/neighbour
//! state into a head-to-head against an external opponent, which is precisely
//! the bias zenbench's randomized round-robin exists to remove.
//! `benches/bench_compare.rs` keeps the wider implementation matrix (C++ FFI,
//! rust-av port); this is the paired instrument for the speed row of the exam.
//!
//! Run:
//! ```text
//! for T in 1 8 16; do RAYON_NUM_THREADS=$T cargo bench --bench ssim2_speed_bar -p zensim-bench; done
//! for T in 1 8 16; do RAYON_NUM_THREADS=$T cargo bench --bench ssim2_speed_bar -p zensim-bench --features ssim2-rayon; done
//! ```
//! `ZEN_S2_SIZES` (default `576,1152,2304`), `ZEN_S2_ROUNDS`, `ZEN_S2_WALL_S`
//! keep a matrix run from holding zenbench's exclusive lock for hours.
use imgref::Img;
use zensim::{RgbSlice, Zensim, ZensimProfile};

/// Byte-identical to `zensim/benches/extract_paths_bench.rs::test_pair` — the
/// content family the attribution tests and `fold_pools_bench` also use, so
/// this bench and that one feed their kernels the same pixels.
fn test_pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let mut src = Vec::with_capacity(w * h);
    let mut dst = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let base = ((x * 255) / w) as u8;
            let tex = (((x * 7 + y * 13) % 32) * 3) as u8;
            let edge = if (y / 16) % 2 == 0 { 40 } else { 0 };
            let px = [
                base.wrapping_add(tex),
                base.wrapping_add(edge),
                (255 - base).wrapping_add(tex / 2),
            ];
            src.push(px);
            let q = |v: u8| (v / 12) * 12;
            let mut d = [q(px[0]), q(px[1]), q(px[2])];
            if x < w / 2 && y < h / 2 {
                d[0] = d[0].saturating_add(18);
            }
            dst.push(d);
        }
    }
    (src, dst)
}

fn env_usize(key: &str, dflt: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(dflt)
}

fn main() {
    let sizes: Vec<usize> = std::env::var("ZEN_S2_SIZES")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![576, 1152, 2304]);
    let max_r = env_usize("ZEN_S2_ROUNDS", 200);
    let min_r = max_r.min(env_usize("ZEN_S2_MIN_ROUNDS", 8));
    let wall_s = env_usize("ZEN_S2_WALL_S", 120) as u64;

    let zb: &'static Zensim = Box::leak(Box::new(Zensim::new(ZensimProfile::B)));

    println!(
        "# ssim2_speed_bar: RAYON_NUM_THREADS={} ssim2_rayon_feature={}",
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "<unset>".into()),
        cfg!(feature = "ssim2-rayon")
    );

    let result = zenbench::run(|suite| {
        for &n in &sizes {
            let (src, dst) = test_pair(n, n);
            let src_s: &'static [[u8; 3]] = Box::leak(src.into_boxed_slice());
            let dst_s: &'static [[u8; 3]] = Box::leak(dst.into_boxed_slice());
            suite.compare(format!("ssim2_bar_{n}"), |group| {
                group
                    .config()
                    .max_rounds(max_r)
                    .min_rounds(min_r)
                    .max_wall_time(std::time::Duration::from_secs(wall_s));
                group.bench("fast_ssim2", move |b| {
                    b.iter(move || {
                        let s = Img::new(src_s, n, n);
                        let d = Img::new(dst_s, n, n);
                        zenbench::black_box(fast_ssim2::compute_ssimulacra2(s, d).unwrap())
                    })
                });
                group.bench("zensim_B", move |b| {
                    b.iter(move || {
                        let s = RgbSlice::new(src_s, n, n);
                        let d = RgbSlice::new(dst_s, n, n);
                        zenbench::black_box(zb.compute(&s, &d).unwrap().score())
                    })
                });
            });
        }
    });
    let _ = result;
}
