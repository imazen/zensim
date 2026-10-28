//! Cross-path contracts for cached scoring and encoder steering maps.
#[path = "../examples/path_audit/fixture.rs"]
mod fixture;
use fixture::pair;

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zensim::{
    AlphaMode, DiffmapOptions, DiffmapWeighting, PixelFormat, RgbSlice, StridedBytes, Zensim,
    ZensimProfile, ZensimScratch,
};

#[test]
fn rev7_cached_and_reused_scratch_scores_match_streaming_across_tiers() {
    let report = for_each_token_permutation(CompileTimePolicy::Fail, |perm| {
        for (w, h) in [(31, 17), (65, 63), (257, 193)] {
            let (src, dst) = pair(w, h);
            let src = RgbSlice::new(&src, w, h);
            let dst = RgbSlice::new(&dst, w, h);
            for profile in [ZensimProfile::E, ZensimProfile::ECodec] {
                for parallel in [false, true] {
                    let z = Zensim::new(profile).with_parallel(parallel);
                    let direct = z.compute(&src, &dst).unwrap();
                    let pre = z.precompute_reference(&src).unwrap();
                    let cached = z.compute_with_ref(&pre, &dst).unwrap();
                    let mut scratch = ZensimScratch::new();
                    // Prime scratch with another candidate before checking reuse.
                    z.compute_with_ref_into(&pre, &src, &mut scratch).unwrap();
                    let reused = z.compute_with_ref_into(&pre, &dst, &mut scratch).unwrap();
                    for (route, score, raw, features) in [
                        (
                            "cached",
                            cached.score(),
                            cached.raw_distance(),
                            cached.features(),
                        ),
                        (
                            "scratch",
                            reused.score(),
                            reused.raw_distance(),
                            reused.features(),
                        ),
                    ] {
                        assert_eq!(
                            score.to_bits(),
                            direct.score().to_bits(),
                            "{profile:?} {w}x{h} parallel={parallel} {} {route}",
                            perm.label
                        );
                        assert_eq!(
                            features.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            direct
                                .features()
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>(),
                            "{profile:?} {w}x{h} {} {route} features",
                            perm.label
                        );
                        assert_eq!(
                            raw.to_bits(),
                            direct.raw_distance().to_bits(),
                            "{profile:?} {w}x{h} {} {route} raw",
                            perm.label
                        );
                    }
                }
            }
        }
    });
    assert!(report.permutations_run > 0);
}

#[test]
fn rev7_planar_and_interleaved_steering_maps_match_with_stride_and_postprocessing() {
    let report = for_each_token_permutation(CompileTimePolicy::Fail, |perm| {
        let mut mismatches = Vec::new();
        for (w, h) in [(31, 17), (65, 63), (257, 193)] {
            let (src, dst) = pair(w, h);
            let rgba = |v: &[[u8; 3]]| {
                v.iter()
                    .map(|p| {
                        [
                            p[0] as f32 / 255.0,
                            p[1] as f32 / 255.0,
                            p[2] as f32 / 255.0,
                            1.0,
                        ]
                    })
                    .collect::<Vec<_>>()
            };
            let src = rgba(&src);
            let dst = rgba(&dst);
            let image = |v| {
                StridedBytes::with_alpha_mode(
                    v,
                    w,
                    h,
                    w * 16,
                    PixelFormat::LinearF32Rgba,
                    AlphaMode::Opaque,
                )
            };
            let source = image(bytemuck::cast_slice(&src));
            let distorted = image(bytemuck::cast_slice(&dst));
            let stride = w + 7;
            // Poison padding: any accidental read changes the result or produces NaN.
            let mut planes = [
                vec![f32::NAN; stride * h],
                vec![f32::NAN; stride * h],
                vec![f32::NAN; stride * h],
            ];
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        planes[c][y * stride + x] = dst[y * w + x][c];
                    }
                }
            }
            for profile in [ZensimProfile::E, ZensimProfile::ECodec] {
                for parallel in [false, true] {
                    let z = Zensim::new(profile).with_parallel(parallel);
                    let pre = z.precompute_reference(&source).unwrap();
                    for (weighting, masking, sqrt) in [
                        DiffmapWeighting::Trained,
                        DiffmapWeighting::Balanced,
                        DiffmapWeighting::Custom([0.2, 0.6, 0.2]),
                    ]
                    .into_iter()
                    .flat_map(|weighting| {
                        [(false, false), (true, false), (false, true), (true, true)]
                            .map(move |(masking, sqrt)| (weighting, masking, sqrt))
                    }) {
                        let options = DiffmapOptions {
                            weighting,
                            masking_strength: masking.then_some(4.0),
                            sqrt,
                            include_edge_mse: true,
                            include_hf: true,
                        };
                        let packed = z
                            .compute_with_ref_and_diffmap(&pre, &distorted, options)
                            .unwrap();
                        let planar = z
                            .compute_with_ref_and_diffmap_linear_planar(
                                &pre,
                                [&planes[0], &planes[1], &planes[2]],
                                w,
                                h,
                                stride,
                                options,
                            )
                            .unwrap();
                        let context = format!(
                            "{profile:?} {w}x{h} parallel={parallel} masking={masking} sqrt={sqrt} weighting={weighting:?} {}",
                            perm.label
                        );
                        if packed.score().to_bits() != planar.score().to_bits() {
                            mismatches.push(format!(
                                "{context}: score {} {}",
                                packed.score().to_bits(),
                                planar.score().to_bits()
                            ));
                        }
                        assert_eq!((planar.width(), planar.height()), (w, h), "{context}");
                        assert_eq!(packed.diffmap().len(), w * h, "{context}");
                        assert_eq!(planar.diffmap().len(), w * h, "{context}");
                        if let Some(i) = packed
                            .diffmap()
                            .iter()
                            .zip(planar.diffmap())
                            .position(|(a, b)| a.to_bits() != b.to_bits())
                        {
                            mismatches.push(format!(
                                "{context}: map[{i}] {} {}",
                                packed.diffmap()[i].to_bits(),
                                planar.diffmap()[i].to_bits()
                            ));
                        }
                    }
                }
            }
        }
        assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
    });
    assert!(report.permutations_run > 0);
}
