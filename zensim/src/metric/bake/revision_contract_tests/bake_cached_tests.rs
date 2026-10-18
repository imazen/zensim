use super::*;

#[test]
fn cached_score_preserves_complete_results_and_reuse() {
    let name = "metric::bake::revision_contract_tests::cached_score::cached_score_preserves_complete_results_and_reuse";
    if !run_at_revision("5", name, "PERF3_CACHED_SCORE_CHILD") {
        return;
    }
    use crate::source::{PixelFormat, StridedBytes};
    let model = rev5_by_model();
    for threads in [1, 8, 16] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                for parallel in [false, true] {
                    for (w, h) in [(17, 31), (64, 64), (137, 301), (256, 256), (513, 129)] {
                        let (src, dst) = pair(w, h);
                        let stride = w * 3 + 17;
                        let pad = |pixels: &[[u8; 3]]| {
                            let mut bytes = vec![231; stride * h];
                            for y in 0..h {
                                for x in 0..w {
                                    bytes[y * stride + x * 3..y * stride + x * 3 + 3]
                                        .copy_from_slice(&pixels[y * w + x]);
                                }
                            }
                            bytes
                        };
                        let (rp, dp) = (pad(&src), pad(&dst));
                        let rs = StridedBytes::new(&rp, w, h, stride, PixelFormat::Srgb8Rgb);
                        let ds = StridedBytes::new(&dp, w, h, stride, PixelFormat::Srgb8Rgb);
                        let mut scorer = crate::BakeScorer::new(&model)
                            .unwrap()
                            .with_parallel(parallel);
                        let reference = scorer.precompute_reference(&rs).unwrap();
                        if w >= 64 && h >= 64 {
                            assert!(
                                crate::feature_v2::retained_ref_feed_admits(
                                    &reference,
                                    &ds,
                                    &scorer.plan().unwrap()
                                ),
                                "the cached path must actually run"
                            );
                        }
                        let mut uncached = crate::BakeScorer::new(&model)
                            .unwrap()
                            .with_parallel(parallel);
                        for d in [&ds, &rs, &ds] {
                            let a = uncached.compute(&rs, d, None).unwrap();
                            let b = scorer.compute_with_ref(&rs, &reference, d, None).unwrap();
                            assert_eq!(
                                a.score().to_bits(),
                                b.score().to_bits(),
                                "{w}x{h}/{threads}"
                            );
                            assert_eq!(a.raw_distance().to_bits(), b.raw_distance().to_bits());
                            assert_eq!(a.is_identical(), b.is_identical());
                            assert_eq!(a.profile(), b.profile());
                            assert_eq!(
                                a.features().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                b.features().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                            );
                            assert_eq!(
                                a.mean_offset().map(f64::to_bits),
                                b.mean_offset().map(f64::to_bits)
                            );
                        }
                        let wrong = crate::source::RgbSlice::new(&src[..w], w, 1);
                        assert!(
                            scorer
                                .compute_with_ref(&wrong, &reference, &wrong, None)
                                .is_err()
                        );
                    }
                }
            });
    }
    println!("PERF3_CACHED_SCORE_CHILD complete");
}

#[test]
fn cached_reference_checks_padded_limit_before_pixel_access() {
    let name = "metric::bake::revision_contract_tests::cached_score::cached_reference_checks_padded_limit_before_pixel_access";
    if !run_at_revision("5", name, "PERF3_CACHED_LIMIT_CHILD") {
        return;
    }
    struct DimensionsOnly(usize, usize);
    impl crate::ImageSource for DimensionsOnly {
        fn width(&self) -> usize {
            self.0
        }
        fn height(&self) -> usize {
            self.1
        }
        fn pixel_format(&self) -> crate::source::PixelFormat {
            crate::source::PixelFormat::Srgb8Rgb
        }
        fn alpha_mode(&self) -> crate::source::AlphaMode {
            crate::source::AlphaMode::Opaque
        }
        fn row_bytes(&self, _: usize) -> &[u8] {
            panic!("oversized padded reference must be rejected before reading pixels")
        }
    }
    let model = rev5_by_model();
    let scorer = crate::BakeScorer::new(&model).unwrap();
    assert!(scorer.plan().unwrap().compute.sampling.is_none());
    for source in [DimensionsOnly(1, 1_875_001), DimensionsOnly(1_875_001, 1)] {
        assert!(matches!(
            scorer.precompute_reference(&source),
            Err(crate::ZensimError::ImageTooLarge)
        ));
    }
    println!("PERF3_CACHED_LIMIT_CHILD complete");
}
