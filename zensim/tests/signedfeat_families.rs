// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! **SIGNEDFEAT families** (`texgain` f1825..1836, `satsign` f1837..1852), appended after the restored cuts.
//!
//! Its own test executable: `for_each_token_permutation` mutates process-wide SIMD dispatch.
//!
//! Gates:
//! * f0..f1824 are `to_bits`-identical with the new families on vs off, on every dispatchable tier,
//!   serial and MT8, tight and strided input;
//! * each new family is `to_bits`-identical whether the layout stops at it or carries the later one;
//! * new slots are identical between serial and MT8, between tight and strided input, repeatable;
//! * identity: every new slot is exactly zero on an identical pair;
//! * semantic controls: texture added moves `texgain` and not `hf_mag_loss`; blur the reverse; a chroma
//!   boost moves `sat_gain`/`gsat_gain` and not the loss half; desaturation the reverse; a pure
//!   luminance change leaves the saturation slots tiny against a chroma change of the same size class.

#![cfg(all(feature = "training", feature = "feature-regime-v2"))]

mod common;

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zensim::feature_v2::{V1PoolsMode, V2NewFeatureToggles, V2Scratch};
use zensim::source::StridedBytes;
use zensim::{PixelFormat, RgbSlice, Zensim, ZensimProfile};

const PREFIX: usize = 1825;
const TEX: std::ops::Range<usize> = 1825..1837;
const SAT: std::ops::Range<usize> = 1837..1853;
const FULL: usize = 1853;

fn base() -> V2NewFeatureToggles {
    V2NewFeatureToggles {
        append_block: true,
        append2_block: true,
        csfw_block: true,
        dvifm_block: true,
        rev4_gridblk: true,
        rev4_ringbasis: true,
        rev4_tailhist: true,
        rev4_arttype: true,
        gmsbank: true,
        mapdev: true,
        z1max: true,
        gmsnative: true,
        dvifmgate: true,
        v1_pools: V1PoolsMode::Full,
        ..V2NewFeatureToggles::default()
    }
}

fn with(texgain: bool, satsign: bool) -> V2NewFeatureToggles {
    V2NewFeatureToggles {
        texgain,
        satsign,
        ..base()
    }
}

fn pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let r = common::generators::gen_value_noise(w, h, 0xC0FFEE ^ w as u32);
    let d = common::generators::distort_block_artifacts(&r, w, h);
    (r, d)
}

fn extract(
    src: &[[u8; 3]],
    dst: &[[u8; 3]],
    w: usize,
    h: usize,
    toggles: V2NewFeatureToggles,
    parallel: bool,
) -> Vec<f64> {
    let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(parallel);
    let mut scratch = V2Scratch::new();
    z.compute_folded720_append_features_streaming(
        &RgbSlice::new(src, w, h),
        &RgbSlice::new(dst, w, h),
        toggles,
        &mut scratch,
    )
    .expect("streaming walk computes")
    .features()
    .to_vec()
}

fn extract_strided(
    src: &[[u8; 3]],
    dst: &[[u8; 3]],
    w: usize,
    h: usize,
    toggles: V2NewFeatureToggles,
) -> Vec<f64> {
    let stride = w * 3 + 13;
    let pad = |px: &[[u8; 3]]| {
        let mut out = vec![0xA5u8; stride * h];
        for y in 0..h {
            for x in 0..w {
                out[y * stride + x * 3..][..3].copy_from_slice(&px[y * w + x]);
            }
        }
        out
    };
    let (ps, pd) = (pad(src), pad(dst));
    let z = Zensim::new(ZensimProfile::codec_target());
    let mut scratch = V2Scratch::new();
    z.compute_folded720_append_features_streaming(
        &StridedBytes::try_new(&ps, w, h, stride, PixelFormat::Srgb8Rgb).expect("strided src"),
        &StridedBytes::try_new(&pd, w, h, stride, PixelFormat::Srgb8Rgb).expect("strided dst"),
        toggles,
        &mut scratch,
    )
    .expect("streaming walk computes")
    .features()
    .to_vec()
}

fn bits_eq(a: &[f64], b: &[f64], range: std::ops::Range<usize>, what: &str) {
    for i in range {
        assert_eq!(a[i].to_bits(), b[i].to_bits(), "{what}: f{i}");
    }
}

#[test]
fn signedfeat_prefix_independence_and_invariances() {
    for (w, h) in [(200, 200), (127, 288), (64, 64), (97, 73)] {
        let (src, dst) = pair(w, h);
        let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let tag = format!("{w}x{h} {}", perm.label);
            let off = extract(&src, &dst, w, h, with(false, false), false);
            assert_eq!(off.len(), PREFIX);
            let all = extract(&src, &dst, w, h, with(true, true), false);
            assert_eq!(all.len(), FULL, "{tag} full width");
            bits_eq(&off, &all, 0..PREFIX, &format!("{tag} prefix"));

            // Layout independence: texgain's layout stops at f1836.
            let t = extract(&src, &dst, w, h, with(true, false), false);
            assert_eq!(t.len(), TEX.end);
            bits_eq(&t, &all, TEX, &format!("{tag} texgain@1837"));
            bits_eq(&t, &off, 0..PREFIX, &format!("{tag} prefix (texgain only)"));

            // The new slots are live (not all zero) on a distorted pair.
            assert!(
                all[TEX].iter().any(|v| *v != 0.0),
                "{tag}: texgain all zero"
            );
            assert!(
                all[SAT].iter().any(|v| *v != 0.0),
                "{tag}: satsign all zero"
            );
            assert!(
                all[PREFIX..]
                    .iter()
                    .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            );

            // Serial vs MT8, tight vs strided, repeatability.
            let mt = extract(&src, &dst, w, h, with(true, true), true);
            bits_eq(&mt, &all, 0..FULL, &format!("{tag} MT vs serial"));
            let st = extract_strided(&src, &dst, w, h, with(true, true));
            bits_eq(&st, &all, 0..FULL, &format!("{tag} strided vs tight"));
            let again = extract(&src, &dst, w, h, with(true, true), false);
            bits_eq(&again, &all, 0..FULL, &format!("{tag} repeat"));
        });
    }
}

#[test]
fn signedfeat_identity_pair_is_exactly_zero() {
    let (w, h) = (97, 73);
    let (src, _) = pair(w, h);
    let f = extract(&src, &src, w, h, with(true, true), false);
    for (k, v) in f[PREFIX..FULL].iter().enumerate() {
        assert_eq!(*v, 0.0, "f{} nonzero on an identity pair", PREFIX + k);
    }
}

fn natural(w: usize, h: usize) -> Vec<[u8; 3]> {
    // Value noise carries texture AND colour; blend it with a smooth ramp so chroma is moderate.
    let n = common::generators::gen_value_noise(w, h, 7);
    n.iter()
        .enumerate()
        .map(|(i, p)| {
            let (x, y) = (i % w, i / w);
            let r = (x * 255 / w) as u8;
            let g = (y * 255 / h) as u8;
            [
                ((u16::from(p[0]) + u16::from(r)) / 2) as u8,
                ((u16::from(p[1]) + u16::from(g)) / 2) as u8,
                ((u16::from(p[2]) + 128) / 2) as u8,
            ]
        })
        .collect()
}

/// Push each pixel's chroma away from / toward its own gray by `k` (in 1/16 units; 16 = identity).
fn scale_chroma(px: &[[u8; 3]], k: i32) -> Vec<[u8; 3]> {
    px.iter()
        .map(|p| {
            let y = (299 * i32::from(p[0]) + 587 * i32::from(p[1]) + 114 * i32::from(p[2])) / 1000;
            let f = |c: u8| (y + (i32::from(c) - y) * k / 16).clamp(0, 255) as u8;
            [f(p[0]), f(p[1]), f(p[2])]
        })
        .collect()
}

fn scale_luma(px: &[[u8; 3]], k: i32) -> Vec<[u8; 3]> {
    px.iter()
        .map(|p| {
            let f = |c: u8| (i32::from(c) * k / 16).clamp(0, 255) as u8;
            [f(p[0]), f(p[1]), f(p[2])]
        })
        .collect()
}

#[test]
fn signedfeat_semantic_controls() {
    let (w, h) = (192, 160);
    let src = natural(w, h);
    let walk = |d: &[[u8; 3]]| extract(&src, d, w, h, with(true, true), false);
    // Existing slots used as the contrast: v2 block base 372, 29 signals per cell, hf_mag_loss = local 8.
    let hf_mag_loss = |f: &[f64], scale: usize, ch: usize| f[372 + (scale * 3 + ch) * 29 + 8];
    let tex = |f: &[f64], scale: usize, ch: usize| f[1825 + scale * 3 + ch];
    // sat slots per scale: [sat_gain, sat_loss, gsat_gain, gsat_loss].
    let sat = |f: &[f64], scale: usize, k: usize| f[1837 + scale * 4 + k];

    // Texture ADDED (noise) → texgain up, hf_mag_loss ~ 0 at scale 0 Y.
    let noisy: Vec<[u8; 3]> = src
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let n = ((i * 2654435761usize) >> 27) as i32 % 25 - 12;
            [0, 1, 2].map(|c| (i32::from(p[c]) + n).clamp(0, 255) as u8)
        })
        .collect();
    let fn_ = walk(&noisy);
    // Texture REMOVED (blur) → texgain ~ 0, hf_mag_loss up.
    let blurred = common::generators::distort_blur(&src, w, h, 2);
    let fb = walk(&blurred);
    eprintln!(
        "noise: texgain {:.4} hf_mag_loss {:.4}; blur: texgain {:.4} hf_mag_loss {:.4} (s0, Y)",
        tex(&fn_, 0, 1),
        hf_mag_loss(&fn_, 0, 1),
        tex(&fb, 0, 1),
        hf_mag_loss(&fb, 0, 1)
    );
    // Per-pixel pooled halves are both nonzero on any real change (pixelwise |HF| is noisy), so the
    // control is the ORDER of the two halves: added texture favours the gain, removed texture the loss.
    assert!(
        tex(&fn_, 0, 1) > 2.0 * hf_mag_loss(&fn_, 0, 1),
        "noise: gain must dominate loss"
    );
    assert!(
        hf_mag_loss(&fb, 0, 1) > 2.0 * tex(&fb, 0, 1),
        "blur: loss must dominate gain"
    );

    // Chroma boost / cut.
    let boost = walk(&scale_chroma(&src, 24)); // 1.5x
    let cut = walk(&scale_chroma(&src, 8)); // 0.5x
    for scale in 0..4 {
        assert!(
            sat(&boost, scale, 0) > 20.0 * sat(&boost, scale, 1).max(1e-9),
            "boost s{scale} per-pixel"
        );
        assert!(
            sat(&boost, scale, 2) > 0.0 && sat(&boost, scale, 3) == 0.0,
            "boost s{scale} global"
        );
        assert!(
            sat(&cut, scale, 1) > 20.0 * sat(&cut, scale, 0).max(1e-9),
            "cut s{scale} per-pixel"
        );
        assert!(
            sat(&cut, scale, 3) > 0.0 && sat(&cut, scale, 2) == 0.0,
            "cut s{scale} global"
        );
    }
    // Pure luminance scaling is a much smaller saturation event than a chroma change of the same size class.
    let dim = walk(&scale_luma(&src, 12)); // 0.75x
    let lum_sat: f64 = (0..4).map(|s| sat(&dim, s, 0) + sat(&dim, s, 1)).sum();
    let cut_sat: f64 = (0..4).map(|s| sat(&cut, s, 0) + sat(&cut, s, 1)).sum();
    eprintln!("luminance 0.75x: sum sat_gain+sat_loss = {lum_sat:.4}; chroma 0.5x: {cut_sat:.4}");
    assert!(
        lum_sat < cut_sat,
        "luminance change out-scores the chroma cut: {lum_sat} vs {cut_sat}"
    );
}
