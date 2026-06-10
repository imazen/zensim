//! Behavioral tests for the public PU (HDR) entry points:
//! `Zensim::compute_pu_linear` (interleaved, primary) and
//! `Zensim::compute_pu_linear_planar`.
//!
//! These pin entry-point *behavior* — layout equivalence, stride handling,
//! sub-pyramid padding, ordering, and input validation — not constants.
//! Perceptual validity is pinned externally by the UPIQ run
//! (`benchmarks/upiq_pu_validation_2026-06-01.md`).

use zensim::{Zensim, ZensimError, ZensimProfile};

/// Deterministic synthetic HDR scene: smooth gradients + a bright highlight,
/// spanning ~0.05..4000 cd/m² so both ends of the PU curve are exercised.
fn hdr_scene(w: usize, h: usize) -> Vec<[f32; 3]> {
    let mut px = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / (w - 1) as f32;
            let fy = y as f32 / (h - 1) as f32;
            let base = 0.05 + 180.0 * fx + 40.0 * fy;
            // Specular highlight in the upper-right quadrant.
            let d2 = (fx - 0.8).powi(2) + (fy - 0.25).powi(2);
            let highlight = 4000.0 * (-d2 * 60.0).exp();
            px.push([
                base + highlight,
                base * 0.9 + highlight * 0.95,
                base * 0.7 + highlight * 0.8,
            ]);
        }
    }
    px
}

/// A visibly distorted version: luminance compression + chroma shift.
fn distorted_scene(w: usize, h: usize) -> Vec<[f32; 3]> {
    hdr_scene(w, h)
        .into_iter()
        .map(|[r, g, b]| [r * 0.55 + 20.0, g * 0.75, b * 1.3 + 5.0])
        .collect()
}

fn interleave(px: &[[f32; 3]], w: usize, h: usize, stride: usize, fill: f32) -> Vec<f32> {
    assert!(stride >= 3 * w);
    let mut out = vec![fill; stride * h];
    for y in 0..h {
        for x in 0..w {
            let p = px[y * w + x];
            out[y * stride + 3 * x..y * stride + 3 * x + 3].copy_from_slice(&p);
        }
    }
    out
}

fn planar(px: &[[f32; 3]]) -> [Vec<f32>; 3] {
    let mut planes: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::with_capacity(px.len()));
    for p in px {
        for c in 0..3 {
            planes[c].push(p[c]);
        }
    }
    planes
}

fn planar_score(z: &Zensim, r: &[[f32; 3]], d: &[[f32; 3]], w: usize, h: usize) -> f64 {
    let rp = planar(r);
    let dp = planar(d);
    z.compute_pu_linear_planar([&rp[0], &rp[1], &rp[2]], [&dp[0], &dp[1], &dp[2]], w, h, w)
        .expect("planar")
        .score()
}

// Above the 64px pyramid floor — the unpadded path.
const W: usize = 96;
const H: usize = 72;

#[test]
fn interleaved_and_planar_agree_exactly() {
    let z = Zensim::new(ZensimProfile::codec_target());
    let (r, d) = (hdr_scene(W, H), distorted_scene(W, H));

    let ri = interleave(&r, W, H, 3 * W, 0.0);
    let di = interleave(&d, W, H, 3 * W, 0.0);
    let s_int = z
        .compute_pu_linear(&ri, &di, W, H, 3 * W, 3 * W)
        .expect("interleaved")
        .score();

    // Same f32 values reach the same kernel in the same order — the two
    // layouts must agree bit-for-bit, not approximately.
    let s_pla = planar_score(&z, &r, &d, W, H);
    assert_eq!(s_int, s_pla, "interleaved {s_int} != planar {s_pla}");
}

#[test]
fn row_padding_is_ignored() {
    let z = Zensim::new(ZensimProfile::codec_target());
    let (r, d) = (hdr_scene(W, H), distorted_scene(W, H));

    let tight_r = interleave(&r, W, H, 3 * W, 0.0);
    let tight_d = interleave(&d, W, H, 3 * W, 0.0);
    let s_tight = z
        .compute_pu_linear(&tight_r, &tight_d, W, H, 3 * W, 3 * W)
        .expect("tight")
        .score();

    // Padded rows with NaN poison in the gap: if any padding element were
    // read, the score would be NaN-contaminated, not equal.
    let pad_r = interleave(&r, W, H, 3 * W + 7, f32::NAN);
    let pad_d = interleave(&d, W, H, 3 * W + 13, f32::NAN);
    let s_pad = z
        .compute_pu_linear(&pad_r, &pad_d, W, H, 3 * W + 7, 3 * W + 13)
        .expect("padded")
        .score();

    assert_eq!(s_tight, s_pad, "strided {s_pad} != tight {s_tight}");
}

#[test]
fn identical_beats_distorted() {
    let z = Zensim::new(ZensimProfile::codec_target());
    let (r, d) = (hdr_scene(W, H), distorted_scene(W, H));
    let ri = interleave(&r, W, H, 3 * W, 0.0);
    let di = interleave(&d, W, H, 3 * W, 0.0);

    let s_same = z
        .compute_pu_linear(&ri, &ri, W, H, 3 * W, 3 * W)
        .expect("identical pair")
        .score();
    let s_dist = z
        .compute_pu_linear(&ri, &di, W, H, 3 * W, 3 * W)
        .expect("distorted pair")
        .score();

    assert!(
        s_same > s_dist,
        "identical pair ({s_same}) must outscore distorted pair ({s_dist})"
    );
    assert!(
        s_same > 99.0,
        "identical pair should score ~100, got {s_same}"
    );
}

#[test]
fn sub_pyramid_sizes_score_via_reflect_pad() {
    // 32×24 is below the 64px pyramid floor: the PU funnel must reflect-pad
    // like the SDR funnel instead of failing in the MLP with a short feature
    // vector, and both layouts must keep agreeing on the padded path.
    let (w, h) = (32, 24);
    let z = Zensim::new(ZensimProfile::codec_target());
    let (r, d) = (hdr_scene(w, h), distorted_scene(w, h));
    let ri = interleave(&r, w, h, 3 * w, 0.0);
    let di = interleave(&d, w, h, 3 * w, 0.0);

    let s_same = z
        .compute_pu_linear(&ri, &ri, w, h, 3 * w, 3 * w)
        .expect("identical sub-64 pair")
        .score();
    let s_dist = z
        .compute_pu_linear(&ri, &di, w, h, 3 * w, 3 * w)
        .expect("distorted sub-64 pair")
        .score();
    assert!(s_same > 99.0, "identical sub-64 pair scored {s_same}");
    assert!(s_same > s_dist, "{s_same} vs {s_dist}");

    let s_pla = planar_score(&z, &r, &d, w, h);
    assert_eq!(s_dist, s_pla, "padded path diverged across layouts");
}

#[test]
fn input_validation() {
    let z = Zensim::new(ZensimProfile::codec_target());
    let r = interleave(&hdr_scene(W, H), W, H, 3 * W, 0.0);

    // Stride below 3*width.
    assert_eq!(
        z.compute_pu_linear(&r, &r, W, H, 3 * W - 1, 3 * W)
            .unwrap_err(),
        ZensimError::InvalidStride
    );
    // Slice shorter than the last row requires.
    let short = &r[..r.len() - 1];
    assert_eq!(
        z.compute_pu_linear(short, &r, W, H, 3 * W, 3 * W)
            .unwrap_err(),
        ZensimError::InvalidDataLength
    );
    // Zero-sized refused (sub-64px is reflect-padded, not refused).
    assert_eq!(
        z.compute_pu_linear(&r, &r, 0, H, 0, 0).unwrap_err(),
        ZensimError::ImageTooSmall
    );
}
