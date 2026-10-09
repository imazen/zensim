//! Ordered batch parity with partial strips and retained scratch of varying sizes.
use super::*;
use crate::{PixelFormat, RgbSlice, StridedBytes};

#[test]
fn ordered_batches_match_serial_with_reused_strided_scratch() {
    const SENTINEL: &str = "REV5_BATCH_PARITY_OK";
    if !crate::ssim_form::run_at_revision(
        "5",
        "feature_v2::rev5_scaling_tests::ordered_batches_match_serial_with_reused_strided_scratch",
        SENTINEL,
    ) {
        return;
    }
    let toggles = V2NewFeatureToggles {
        formula_revision: FormulaRevision::Rev5,
        v1_pools: V1PoolsMode::Peaks,
        ..Default::default()
    };
    for threads in [8, 16, 32] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let mut scratch = V2Scratch::new();
        for (w, h, identity) in [
            (257, 1025, false),
            (511, 259, false),
            (129, 513, true),
            (65, 129, false),
        ] {
            let src: Vec<[u8; 3]> = (0..w * h)
                .map(|i| {
                    let x = i % w;
                    let y = i / w;
                    [
                        (x * 17 + y * 31) as u8,
                        ((x * 7) ^ (y * 13)) as u8,
                        (x * y + 91) as u8,
                    ]
                })
                .collect();
            let dst: Vec<[u8; 3]> = src
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    if identity {
                        p
                    } else {
                        [
                            p[0].wrapping_add((i % 11) as u8),
                            p[1] & 0xf0,
                            p[2].saturating_sub(7),
                        ]
                    }
                })
                .collect();
            let expected = compute_folded720_streaming_impl(
                &RgbSlice::new(&src, w, h),
                &RgbSlice::new(&dst, w, h),
                None,
                false,
                toggles,
                &mut V2Scratch::new(),
                None,
            )
            .unwrap()
            .into_features();
            let stride = w * 3 + 17;
            let mut padded_src = vec![0; stride * h];
            let mut padded_dst = vec![0; stride * h];
            for y in 0..h {
                for x in 0..w {
                    let at = y * stride + x * 3;
                    padded_src[at..at + 3].copy_from_slice(&src[y * w + x]);
                    padded_dst[at..at + 3].copy_from_slice(&dst[y * w + x]);
                }
            }
            let source =
                StridedBytes::try_new(&padded_src, w, h, stride, PixelFormat::Srgb8Rgb).unwrap();
            let distorted =
                StridedBytes::try_new(&padded_dst, w, h, stride, PixelFormat::Srgb8Rgb).unwrap();
            let actual = pool
                .install(|| {
                    compute_folded720_streaming_impl(
                        &source,
                        &distorted,
                        None,
                        true,
                        toggles,
                        &mut scratch,
                        None,
                    )
                })
                .unwrap()
                .into_features();
            assert!(
                !scratch.rev5_jobs.is_empty(),
                "the batch route must execute"
            );
            assert_eq!(expected.len(), actual.len());
            for (i, (a, b)) in expected.iter().zip(actual.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{threads}T {w}x{h} identity={identity} feature {i}"
                );
            }
        }
    }
    println!("{SENTINEL}");
}
