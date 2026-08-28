//! Issue #17, remaining task: verify `GamutMapping::Preserve` against REAL
//! codec output rather than the synthetic solid-colour fixture in
//! `icc_coverage.rs`.
//!
//! Scenario (the issue's failure case, run through an actual encoder):
//! a wide-gamut source is JPEG-encoded twice with zenjpeg —
//!
//! - **faithful**: the wide-gamut code values are encoded as-is (the
//!   codec is gamut-agnostic; the saturated signal survives, modulo
//!   quantisation noise);
//! - **clipped**: the pipeline destructively clips the linear signal to
//!   the sRGB gamut BEFORE encoding (BT.2020/P3 linear → sRGB linear →
//!   clamp [0,1] → back to the source primaries → encode).
//!
//! Both decodes are scored against the uncompressed source, tagged with
//! the source primaries. Under the default `Clip` the two scores are
//! indistinguishable (the metric clamps the reference the same way the
//! bad pipeline did — the regression is MASKED); under `Preserve` the
//! clipped encode scores clearly below the faithful one (DETECTED).
//!
//! Gated on `custom-profiles` for the same reason as `icc_coverage.rs`:
//! the assertions need the correct-by-construction linear-bounded
//! profile. Run with:
//! `cargo test -p zensim --features custom-profiles --test gamut_real_codec`
#![cfg(feature = "custom-profiles")]

use enough::Unstoppable;
use zensim::profile::ProfileParams;
use zensim::{ColorPrimaries, GamutMapping, PixelFormat, StridedBytes, Zensim, ZensimProfile};

fn linear_bounded_params() -> &'static ProfileParams {
    use std::sync::OnceLock;
    static P: OnceLock<ProfileParams> = OnceLock::new();
    P.get_or_init(|| ProfileParams::builder().bounded_squash(true).build())
}

fn zensim() -> Zensim {
    Zensim::new(ZensimProfile::Custom {
        params: linear_bounded_params(),
        name: "zensim-linear-bounded",
    })
}

/// Display P3 linear → sRGB linear (same values as `zensim::color`, which
/// keeps them private).
#[rustfmt::skip]
const P3_TO_SRGB: [[f64; 3]; 3] = [
    [ 1.224_940_2,   -0.224_940_2,   0.0         ],
    [-0.042_056_955,  1.042_056_9,   0.0         ],
    [-0.019_637_555, -0.078_636_04,  1.098_273_6 ],
];

/// BT.2020 linear → sRGB linear.
#[rustfmt::skip]
const BT2020_TO_SRGB: [[f64; 3]; 3] = [
    [ 1.660_491,   -0.587_641_1, -0.072_849_9 ],
    [-0.124_550_5,  1.132_899_9, -0.008_349_4 ],
    [-0.018_151_0, -0.100_578_6,  1.118_729_6 ],
];

fn matrix_for(p: ColorPrimaries) -> [[f64; 3]; 3] {
    match p {
        ColorPrimaries::DisplayP3 => P3_TO_SRGB,
        ColorPrimaries::Bt2020 => BT2020_TO_SRGB,
        other => panic!("test only covers wide-gamut primaries, got {other:?}"),
    }
}

fn mul(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for (o, row) in out.iter_mut().zip(m) {
        *o = row[0] * v[0] + row[1] * v[1] + row[2] * v[2];
    }
    out
}

/// Cofactor inverse of a 3×3 (the test needs the sRGB → source direction).
fn invert(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let a = m;
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    assert!(det.abs() > 1e-9, "singular gamut matrix");
    let inv = 1.0 / det;
    [
        [
            (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * inv,
            (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv,
            (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv,
        ],
        [
            (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * inv,
            (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv,
            (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv,
        ],
        [
            (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * inv,
            (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv,
            (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv,
        ],
    ]
}

fn srgb_decode(v: u8) -> f64 {
    let c = v as f64 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn srgb_encode(l: f64) -> u8 {
    let l = l.clamp(0.0, 1.0);
    let c = if l <= 0.003_130_8 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    (c * 255.0).round().clamp(0.0, 255.0) as u8
}

/// A wide-gamut test card: saturated primary-coloured bands with a
/// luminance ramp and a fine stripe texture, so the JPEG has real work to
/// do and the saturated signal covers most of the frame. Code values are
/// in the SOURCE primaries' transfer (sRGB curve, as `PixelFormat::Srgb8Rgb`
/// with `with_color_primaries` interprets them).
fn wide_gamut_card(w: usize, h: usize) -> Vec<[u8; 3]> {
    let mut px = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let band = (y * 4) / h; // 4 horizontal bands
            let ramp = 0.55 + 0.45 * (x as f64 / (w - 1) as f64);
            let stripe = if (x / 3 + y / 5) % 2 == 0 { 1.0 } else { 0.92 };
            let v = srgb_encode(ramp * stripe);
            let low = srgb_encode(0.04 * ramp);
            px.push(match band {
                0 => [v, low, low],         // saturated red
                1 => [low, v, low],         // saturated green
                2 => [v, v / 2, low],       // saturated orange
                _ => [low, v, v / 3 + low], // saturated teal-green
            });
        }
    }
    px
}

/// The destructive pipeline: clip the source's linear signal to the sRGB
/// gamut and express the result back in the source primaries.
fn clip_to_srgb_gamut(src: &[[u8; 3]], primaries: ColorPrimaries) -> Vec<[u8; 3]> {
    let to_srgb = matrix_for(primaries);
    let from_srgb = invert(&to_srgb);
    src.iter()
        .map(|p| {
            let lin = [srgb_decode(p[0]), srgb_decode(p[1]), srgb_decode(p[2])];
            let s = mul(&to_srgb, lin);
            let clipped = [
                s[0].clamp(0.0, 1.0),
                s[1].clamp(0.0, 1.0),
                s[2].clamp(0.0, 1.0),
            ];
            let back = mul(&from_srgb, clipped);
            [
                srgb_encode(back[0]),
                srgb_encode(back[1]),
                srgb_encode(back[2]),
            ]
        })
        .collect()
}

fn jpeg_roundtrip(pixels: &[[u8; 3]], w: usize, h: usize, quality: u8) -> Vec<[u8; 3]> {
    use zenjpeg::decoder::Decoder;
    use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
    let flat: Vec<u8> = pixels.iter().flat_map(|p| p.iter().copied()).collect();
    // 4:4:4 so chroma subsampling does not desaturate the card on its own.
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::None);
    let mut enc = config
        .encode_from_bytes(w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .expect("zenjpeg encoder init");
    enc.push_packed(&flat, Unstoppable).expect("zenjpeg push");
    let bytes = enc.finish().expect("zenjpeg finish");

    let result = Decoder::new()
        .decode(&bytes, Unstoppable)
        .expect("zenjpeg decode");
    let (dw, dh) = result.dimensions();
    assert_eq!((dw as usize, dh as usize), (w, h));
    let out = result.pixels_u8().expect("u8 jpeg output");
    out.as_chunks::<3>()
        .0
        .iter()
        .map(|c| [c[0], c[1], c[2]])
        .collect()
}

fn source<'a>(
    bytes: &'a [u8],
    w: usize,
    h: usize,
    primaries: ColorPrimaries,
    mapping: GamutMapping,
) -> StridedBytes<'a> {
    StridedBytes::new(bytes, w, h, w * 3, PixelFormat::Srgb8Rgb)
        .with_color_primaries(primaries)
        .with_gamut_mapping(mapping)
}

fn flat(px: &[[u8; 3]]) -> Vec<u8> {
    px.iter().flat_map(|p| p.iter().copied()).collect()
}

struct Scores {
    clip_faithful: f64,
    clip_clipped: f64,
    preserve_faithful: f64,
    preserve_clipped: f64,
}

fn run(primaries: ColorPrimaries, quality: u8) -> Scores {
    let (w, h) = (96, 96);
    let src = wide_gamut_card(w, h);
    let clipped_src = clip_to_srgb_gamut(&src, primaries);
    // The clip must actually change something, or the scenario is vacuous.
    let changed = src.iter().zip(&clipped_src).filter(|(a, b)| a != b).count();
    assert!(
        changed > w * h / 2,
        "{primaries:?}: the sRGB-gamut clip touched only {changed}/{} pixels — card is not wide-gamut enough",
        w * h
    );

    let faithful = jpeg_roundtrip(&src, w, h, quality);
    let clipped = jpeg_roundtrip(&clipped_src, w, h, quality);

    let (s, f, c) = (flat(&src), flat(&faithful), flat(&clipped));
    let z = zensim();
    let score = |m: GamutMapping, dist: &[u8]| -> f64 {
        z.compute(
            &source(&s, w, h, primaries, m),
            &source(dist, w, h, primaries, m),
        )
        .unwrap()
        .score()
    };
    let scores = Scores {
        clip_faithful: score(GamutMapping::Clip, &f),
        clip_clipped: score(GamutMapping::Clip, &c),
        preserve_faithful: score(GamutMapping::Preserve, &f),
        preserve_clipped: score(GamutMapping::Preserve, &c),
    };
    println!(
        "  {primaries:?} zenjpeg q{quality}: Clip faithful={:.4} clipped={:.4} | Preserve faithful={:.4} clipped={:.4}",
        scores.clip_faithful,
        scores.clip_clipped,
        scores.preserve_faithful,
        scores.preserve_clipped
    );
    scores
}

fn assert_masked_vs_detected(primaries: ColorPrimaries, s: &Scores) {
    // Sanity: the FAITHFUL JPEG output scores like ordinary lossy output —
    // close to, but not identical to, the source — under either mode.
    for (name, v) in [
        ("clip_faithful", s.clip_faithful),
        ("clip_clipped", s.clip_clipped),
        ("preserve_faithful", s.preserve_faithful),
    ] {
        assert!(
            v > 60.0 && v < 100.0,
            "{primaries:?} {name}: expected a sane lossy-JPEG score in (60, 100), got {v}"
        );
    }
    assert!(
        s.preserve_clipped > 0.0 && s.preserve_clipped < 100.0,
        "{primaries:?} preserve_clipped out of range: {}",
        s.preserve_clipped
    );
    // MASKED under Clip: the encode that destroyed the wide-gamut signal
    // is never seen as a LOSS relative to the faithful one — the metric's
    // own clamp reproduces the bad pipeline. Measured with zenjpeg:
    // BT.2020 faithful 91.40 vs clipped 91.69 (q95) and 82.70 vs 86.74
    // (q75) — the pre-clipped source is an EASIER JPEG, so the regression
    // reads as an IMPROVEMENT; Display P3 91.34 vs 89.65 (q95) — the 8-bit
    // re-quantisation of the clipped card costs ~1.7 points, an order of
    // magnitude under the Preserve separation. The one-sided bound is the
    // claim: any deficit the Clip mode shows is inside the codec-noise
    // band, never the gamut loss itself.
    let clip_deficit = (s.clip_faithful - s.clip_clipped).max(0.0);
    assert!(
        clip_deficit < 2.5,
        "{primaries:?}: under Clip the pre-encode gamut clip should be masked \
         (faithful − clipped < 2.5), got faithful={:.4} clipped={:.4}",
        s.clip_faithful,
        s.clip_clipped
    );
    // DETECTED under Preserve: the clipped encode scores far below the
    // faithful one — well outside the JPEG noise band. Measured with
    // zenjpeg: BT.2020 89.99 → 15.37 (q95) / 78.58 → 15.37 (q75),
    // Display P3 91.31 → 54.72 (q95).
    let preserve_gap = s.preserve_faithful - s.preserve_clipped;
    assert!(
        preserve_gap > 10.0,
        "{primaries:?}: under Preserve the gamut clip must be detectable \
         (faithful − clipped > 10), got faithful={:.4} clipped={:.4} gap={preserve_gap:.4}",
        s.preserve_faithful,
        s.preserve_clipped
    );
    // The Preserve separation dominates whatever deficit Clip mode saw.
    assert!(
        preserve_gap > 5.0 * clip_deficit,
        "{primaries:?}: Preserve separation ({preserve_gap:.4}) should dominate Clip deficit ({clip_deficit:.4})"
    );
    // The detection is not a global offset on wide-gamut content: the
    // faithful encode scores about the same in both modes (the JPEG noise
    // is the only loss the metric sees either way).
    let faithful_mode_gap = (s.clip_faithful - s.preserve_faithful).abs();
    assert!(
        faithful_mode_gap < 5.0,
        "{primaries:?}: faithful encode should score alike in both modes, \
         Clip={:.4} Preserve={:.4}",
        s.clip_faithful,
        s.preserve_faithful
    );
}

#[test]
fn zenjpeg_bt2020_gamut_clip_masked_by_clip_detected_by_preserve() {
    let s = run(ColorPrimaries::Bt2020, 95);
    assert_masked_vs_detected(ColorPrimaries::Bt2020, &s);
}

#[test]
fn zenjpeg_display_p3_gamut_clip_masked_by_clip_detected_by_preserve() {
    let s = run(ColorPrimaries::DisplayP3, 95);
    assert_masked_vs_detected(ColorPrimaries::DisplayP3, &s);
}

/// The detection must survive a lossier encode: at q75 the JPEG noise is
/// larger, but the pre-encode gamut clip is still the dominant loss under
/// `Preserve` and still invisible under `Clip`.
#[test]
fn zenjpeg_bt2020_gamut_clip_detected_at_q75() {
    let s = run(ColorPrimaries::Bt2020, 75);
    assert_masked_vs_detected(ColorPrimaries::Bt2020, &s);
}

/// Codec noise alone (no gamut clip) scores the same under both modes on
/// in-gamut content: `Preserve` is not a blanket penalty on JPEG output.
#[test]
fn zenjpeg_in_gamut_content_scores_alike_under_both_modes() {
    let (w, h) = (96, 96);
    // Desaturated version of the card: comfortably inside sRGB gamut even
    // after the BT.2020 → sRGB matrix.
    let src: Vec<[u8; 3]> = wide_gamut_card(w, h)
        .iter()
        .map(|p| {
            let m = |v: u8| 100 + (v as u16 * 60 / 255) as u8;
            [m(p[0]), m(p[1]), m(p[2])]
        })
        .collect();
    let primaries = ColorPrimaries::Bt2020;
    let clipped = clip_to_srgb_gamut(&src, primaries);
    let changed = src.iter().zip(&clipped).filter(|(a, b)| a != b).count();
    assert!(
        changed <= w * h / 100,
        "in-gamut fixture must be (almost) untouched by the sRGB clip, {changed} px changed"
    );
    let dist = jpeg_roundtrip(&src, w, h, 85);
    let (s, d) = (flat(&src), flat(&dist));
    let z = zensim();
    let clip = z
        .compute(
            &source(&s, w, h, primaries, GamutMapping::Clip),
            &source(&d, w, h, primaries, GamutMapping::Clip),
        )
        .unwrap()
        .score();
    let preserve = z
        .compute(
            &source(&s, w, h, primaries, GamutMapping::Preserve),
            &source(&d, w, h, primaries, GamutMapping::Preserve),
        )
        .unwrap()
        .score();
    println!("  in-gamut zenjpeg q85: Clip={clip:.4} Preserve={preserve:.4}");
    assert!(
        (clip - preserve).abs() < 0.2,
        "in-gamut JPEG output must score alike in both modes: clip={clip} preserve={preserve}"
    );
}
