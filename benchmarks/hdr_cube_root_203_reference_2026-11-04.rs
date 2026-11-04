mod fixtures {
    include!(env!("ZENSIM_HDR_FIXTURES"));
}
use fixtures::{gen_hdr_image, hdr_case_id, hdr_cases};
use zensim::BakeScorer;
use zensim::feature_v2::{HdrEncoding, HdrFrontEnd};
use zensim::source::{AlphaMode, ColorPrimaries, PixelFormat, StridedBytes};
fn main() {
    let bytes = std::fs::read(std::env::args().nth(1).expect("A bake path")).unwrap();
    let model = zenpredict::Model::from_bytes(&bytes).unwrap();
    println!(
        "# research BakeScorer; CubeRoot unit_nits=203; A bake d5e2a56db35c32620d9f9d9c19ba3bc78bd8fea2b8cd933cde8447428d85e62a"
    );
    println!("index\tencoding\tcase\tscore_bits\tscore");
    for (i, c) in hdr_cases().iter().enumerate() {
        let enc = match c.enc {
            0 => HdrEncoding::Linear,
            1 => HdrEncoding::Pq { peak_nits: 1000.0 },
            2 => HdrEncoding::Pq { peak_nits: 10000.0 },
            _ => HdrEncoding::Hlg {
                peak_nits: 1000.0,
                ambient_lux: 5.0,
            },
        };
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
        let (src, ss) = gen_hdr_image(c, false);
        let (dst, ds) = gen_hdr_image(c, true);
        let src = StridedBytes::try_with_alpha_mode(
            &src,
            c.width,
            c.height,
            ss,
            format,
            AlphaMode::Opaque,
        )
        .unwrap()
        .with_color_primaries(primaries);
        let dst = StridedBytes::try_with_alpha_mode(
            &dst,
            c.width,
            c.height,
            ds,
            format,
            AlphaMode::Opaque,
        )
        .unwrap()
        .with_color_primaries(primaries);
        let mut scorer = BakeScorer::new(&model)
            .unwrap()
            .with_parallel(c.parallel)
            .with_hdr_front_end(HdrFrontEnd::CubeRoot { unit_nits: 203.0 });
        let score = scorer.compute_hdr(&src, &dst, enc, None).unwrap();
        println!(
            "{i}\t{}\t{}\t{:016x}\t{score:.17}",
            c.enc_name,
            hdr_case_id(c),
            score.to_bits()
        );
    }
}
