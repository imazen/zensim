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

fn features(c: &HdrCase) -> Vec<f64> {
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
    Zensim::new(ZensimProfile::E)
        .with_parallel(c.parallel)
        .compute_hdr(&src, &dst, encoding(c))
        .unwrap()
        .features()
        .to_vec()
}

#[test]
fn hdr_features_identical_across_simd_tiers() {
    let _token = lock_token_testing();
    // One 64×64 case per encoding covers every decode and PU-XYB path.
    for c in hdr_cases()
        .iter()
        .filter(|c| c.width == 64 && c.height == 64)
        .take(8)
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
    let err = Zensim::new(ZensimProfile::E)
        .compute_hdr(&img, &img, HdrEncoding::Linear)
        .unwrap_err();
    assert!(
        matches!(err, ZensimError::UnsupportedPixelFormat),
        "{err:?}"
    );
}
