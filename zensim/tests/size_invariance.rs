//! Size-invariance regression gate (2026-06-06).
//!
//! Sub-64px images are reflect(mirror)-padded to the 4-scale pyramid minimum
//! inside `compute`, so:
//!   1. a CONSTANT color difference scores identically at every size
//!      (the hard invariant — must not vary with size), and
//!   2. images score all the way down to 1×1 with NO error.

use zensim::{RgbSlice, Zensim, ZensimProfile};

fn solid(n: usize, c: [u8; 3]) -> Vec<[u8; 3]> {
    vec![c; n * n]
}

fn profiles() -> [ZensimProfile; 2] {
    [ZensimProfile::A, ZensimProfile::PreviewV0_2]
}

#[test]
fn solid_color_score_is_size_invariant() {
    let refc = [100u8, 120, 140];
    let distc = [112u8, 116, 150]; // constant per-channel delta
    for profile in profiles() {
        let z = Zensim::new(profile);
        let mut scores = Vec::new();
        for n in [1usize, 2, 4, 8, 16, 32, 48, 63, 64, 128, 256] {
            let r = solid(n, refc);
            let d = solid(n, distc);
            let s = z
                .compute(&RgbSlice::new(&r, n, n), &RgbSlice::new(&d, n, n))
                .expect("solid pair should score at every size")
                .score();
            scores.push(s);
        }
        let (mn, mx) = scores
            .iter()
            .fold((f64::MAX, f64::MIN), |(a, b), &s| (a.min(s), b.max(s)));
        assert!(
            mx - mn < 0.05,
            "{profile:?}: a constant color difference must score identically at \
             every size; got range {:.5} across {scores:?}",
            mx - mn
        );
    }
}

#[test]
fn scores_down_to_1x1_without_error() {
    for profile in profiles() {
        let z = Zensim::new(profile);
        for n in [1usize, 2, 3, 5, 7, 11, 15, 31, 47, 63] {
            let r = solid(n, [10, 20, 30]);
            let d = solid(n, [40, 50, 60]);
            assert!(
                z.compute(&RgbSlice::new(&r, n, n), &RgbSlice::new(&d, n, n))
                    .is_ok(),
                "{profile:?}: compute must succeed at {n}x{n} (reflect-pad fallback)"
            );
        }
    }
}

#[test]
fn non_square_sub64_scores() {
    // 1:3 and degenerate non-square tiny images must also score.
    let z = Zensim::new(ZensimProfile::A);
    for (w, h) in [(1usize, 3usize), (3, 9), (8, 40), (40, 8), (1, 1), (2, 200)] {
        let r = vec![[100u8, 110, 120]; w * h];
        let d = vec![[130u8, 90, 140]; w * h];
        assert!(
            z.compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
                .is_ok(),
            "compute must succeed at {w}x{h}"
        );
    }
}

#[test]
fn large_images_unchanged_by_padding() {
    // The pad path is a NO-OP at >= 64px: a 64x64 and 65x65 score normally.
    let z = Zensim::new(ZensimProfile::A);
    for n in [64usize, 65, 100] {
        let r = solid(n, [50, 60, 70]);
        let d = solid(n, [55, 58, 75]);
        let s = z
            .compute(&RgbSlice::new(&r, n, n), &RgbSlice::new(&d, n, n))
            .unwrap()
            .score();
        assert!(s.is_finite() && (0.0..=100.0).contains(&s), "{n}x{n} -> {s}");
    }
}
