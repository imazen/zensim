//! SIMD-tier isolation: the native top tier vs the same code forced to scalar.
//!
//! `bench_zensim.rs` measures absolute throughput and `bench_compare.rs`
//! measures zensim against other metrics. Neither can tell you whether the
//! SIMD paths are earning their keep — a kernel slower than its own scalar
//! fallback is invisible in both. This bench runs the identical pipeline with
//! the native SIMD token disabled, which is the comparison that can expose a
//! bad kernel. (The same gap in linear-srgb was hiding a real regression.)
//!
//! Run: `cargo bench -p zensim-bench --bench tier_isolation`
//! Do NOT build with `-C target-cpu=native`: that pins the tier at compile
//! time, after which it cannot be disabled and this bench skips rather than
//! silently reporting the SIMD path under both labels.

use criterion::{Criterion, criterion_group, criterion_main};
use zensim::{RgbSlice, Zensim, ZensimProfile};

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") {
    "neon"
} else {
    "v3(avx2)"
};

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(enabled: bool) -> bool {
    TierToken::dangerously_disable_token_process_wide(!enabled).is_ok()
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_enabled: bool) -> bool {
    false
}

fn make_pair(width: usize, height: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
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
    (src, dst)
}

/// zensim builds a multi-scale pyramid, so the ratio of vectorised kernel work
/// to per-scale overhead shifts with size. One size cannot tell you whether the
/// SIMD kernels hold up across the range.
const SIZES: &[(&str, usize, usize)] = &[
    ("256x256", 256, 256),
    ("512x512", 512, 512),
    ("1920x1080", 1920, 1080),
];

fn bench_tiers(c: &mut Criterion) {
    if !set_simd(true) || !set_simd(false) {
        eprintln!(
            "[tier_isolation] no toggleable SIMD tier on this target, or the tier is \
             compile-time guaranteed (drop -C target-cpu=native, ensure \
             archmage/testable_dispatch). Skipping."
        );
        return;
    }
    set_simd(true);
    eprintln!("[tier_isolation] comparing {TIER_NAME} vs forced scalar");

    let z = Zensim::new(ZensimProfile::B);
    for &(label, w, h) in SIZES {
        let (src, dst) = make_pair(w, h);
        let mut group = c.benchmark_group(format!("zensim/{label}"));
        for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
            group.bench_function(arm, |b| {
                set_simd(simd);
                b.iter(|| {
                    let s = RgbSlice::new(std::hint::black_box(&src), w, h);
                    let d = RgbSlice::new(std::hint::black_box(&dst), w, h);
                    z.compute(&s, &d).unwrap()
                })
            });
        }
        set_simd(true);
        group.finish();
    }
    set_simd(true);
}

criterion_group!(benches, bench_tiers);
criterion_main!(benches);
