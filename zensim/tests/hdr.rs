//! `Zensim::compute_hdr`: input contract and SIMD-tier identity.
//!
//! The tier test runs in its own test binary because
//! `for_each_token_permutation` disables tiers process-wide; every test here
//! holds `lock_token_testing()` so none of them observes a partial tier set.

mod fixtures {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/split_parity/hdr.rs"
    ));
}

use archmage::testing::{CompileTimePolicy, for_each_token_permutation, lock_token_testing};
use fixtures::{HdrCase, gen_hdr_image, hdr_cases};
use zensim::{
    AlphaMode, ColorPrimaries, HdrEncoding, PixelFormat, StridedBytes, Zensim, ZensimError,
    ZensimProfile,
};

fn encoding(c: &HdrCase) -> HdrEncoding {
    match c.enc {
        0 => HdrEncoding::Linear,
        1 => HdrEncoding::Pq { peak_nits: 1000.0 },
        2 => HdrEncoding::Pq {
            peak_nits: 10_000.0,
        },
        _ => HdrEncoding::Hlg {
            peak_nits: 1000.0,
            ambient_lux: 5.0,
        },
    }
}

fn compute(c: &HdrCase) -> zensim::ZensimResult {
    let format = if c.srgb16 {
        PixelFormat::Srgb16Rgba
    } else {
        PixelFormat::LinearF32Rgba
    };
    let primaries = if c.bt2020 {
        ColorPrimaries::Bt2020
    } else {
        ColorPrimaries::Srgb
    };
    let (src, src_stride) = gen_hdr_image(c, false);
    let (dst, dst_stride) = gen_hdr_image(c, true);
    let src = StridedBytes::try_with_alpha_mode(
        &src,
        c.width,
        c.height,
        src_stride,
        format,
        AlphaMode::Opaque,
    )
    .unwrap()
    .with_color_primaries(primaries);
    let dst = StridedBytes::try_with_alpha_mode(
        &dst,
        c.width,
        c.height,
        dst_stride,
        format,
        AlphaMode::Opaque,
    )
    .unwrap()
    .with_color_primaries(primaries);
    Zensim::new(ZensimProfile::A)
        .with_parallel(c.parallel)
        .compute_hdr(&src, &dst, encoding(c))
        .unwrap()
}

fn features(c: &HdrCase) -> Vec<f64> {
    compute(c).features().to_vec()
}

#[test]
fn hdr_features_identical_across_simd_tiers() {
    let _token = lock_token_testing();
    // Every 64×64 fixture covers the encoding/container/primaries/stride states.
    for c in hdr_cases()
        .iter()
        .filter(|c| c.width == 64 && c.height == 64)
    {
        let base = features(c);
        let mut ran = 0usize;
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            ran += 1;
            for (i, (a, b)) in base.iter().zip(features(c).iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{} {perm:?}: slot {i} differs across tiers",
                    c.enc_name
                );
            }
        });
        assert!(ran > 0, "no tier ran: {report:?}");
    }
}

#[test]
fn sdr_pixel_formats_are_refused() {
    let _token = lock_token_testing();
    let px = vec![128u8; 64 * 64 * 3];
    let img = StridedBytes::try_new(&px, 64, 64, 64 * 3, PixelFormat::Srgb8Rgb).unwrap();
    let err = Zensim::new(ZensimProfile::A)
        .compute_hdr(&img, &img, HdrEncoding::Linear)
        .unwrap_err();
    assert!(
        matches!(err, ZensimError::UnsupportedPixelFormat),
        "{err:?}"
    );
}

/// Scores were captured through the research BakeScorer, never through this port.
#[test]
fn a_hdr_scores_match_the_research_cube_root_reference_bit_for_bit() {
    let _token = lock_token_testing();
    let cases = hdr_cases();
    let rows: Vec<_> = include_str!("fixtures/hdr_a_cube_root_203.tsv")
        .lines()
        .skip(2)
        .collect();
    assert_eq!(rows.len(), cases.len());
    assert_eq!(rows.len(), 166);
    for (i, (c, row)) in cases.iter().zip(rows).enumerate() {
        let columns: Vec<_> = row.split('\t').collect();
        assert_eq!(columns.len(), 5);
        assert_eq!(columns[0].parse::<usize>().unwrap(), i);
        assert_eq!(columns[1], c.enc_name);
        assert_eq!(columns[2], fixtures::hdr_case_id(c));
        let expected = u64::from_str_radix(columns[3], 16).unwrap();
        let got = compute(c).score();
        assert_eq!(
            got.to_bits(),
            expected,
            "HDR case {i}: {c:?}, got {got:.17}"
        );
    }
}

mod sdr_fixtures {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/split_parity/pairs.rs"
    ));
}

/// An SDR pair displayed at 203 cd/m² white retains its score through HDR.
#[test]
fn a_hdr_at_reference_white_is_continuous_with_sdr() {
    let _token = lock_token_testing();
    let pairs = sdr_fixtures::split_parity_pairs();
    let ids = [
        "64x64-photo-noise",
        "64x64-edges-blur",
        "65x63-text-blocks",
        "128x96-photo-posterize",
        "200x150-edges-shift",
        "200x150-photo-blur",
        "256x256-text-noise",
        "256x256-photo-pixels",
        "333x251-edges-blocks",
        "333x251-photo-identity",
        "16x16-photo-noise",
        "31x17-text-shift",
        "512x384-photo-blocks",
        "1024x768-photo-noise",
    ];
    let lut: Vec<f32> = (0..=255u8)
        .map(|v| {
            let c = v as f32 / 255.0;
            let linear = if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            };
            linear * 203.0
        })
        .collect();
    let mut max_gap = 0.0f64;
    for id in ids {
        let p = pairs.iter().find(|p| p.id == id).unwrap();
        let src = zensim::RgbSlice::new(bytemuck::cast_slice(&p.src), p.width, p.height);
        let dst = zensim::RgbSlice::new(bytemuck::cast_slice(&p.dst), p.width, p.height);
        let nits = |rgb: &[u8]| -> Vec<[f32; 4]> {
            rgb.as_chunks::<3>()
                .0
                .iter()
                .map(|p| {
                    [
                        lut[p[0] as usize],
                        lut[p[1] as usize],
                        lut[p[2] as usize],
                        1.0,
                    ]
                })
                .collect()
        };
        let ns = nits(&p.src);
        let nd = nits(&p.dst);
        let linear = |data| {
            StridedBytes::try_with_alpha_mode(
                data,
                p.width,
                p.height,
                p.width * 16,
                PixelFormat::LinearF32Rgba,
                AlphaMode::Opaque,
            )
            .unwrap()
        };
        let hs = linear(bytemuck::cast_slice(&ns));
        let hd = linear(bytemuck::cast_slice(&nd));
        for parallel in [false, true] {
            let z = Zensim::new(ZensimProfile::A).with_parallel(parallel);
            let sdr = z.compute(&src, &dst).unwrap().score();
            let hdr = z
                .compute_hdr(&hs, &hd, HdrEncoding::Linear)
                .unwrap()
                .score();
            let gap = (hdr - sdr).abs();
            max_gap = max_gap.max(gap);
            assert!(
                gap <= 1e-3,
                "{id} parallel={parallel}: SDR={sdr:.17} HDR={hdr:.17} gap={gap:.17}"
            );
        }
    }
    println!("203 cd/m² continuity: 28 comparisons, max absolute gap {max_gap:.17}");
}
