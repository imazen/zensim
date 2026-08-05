//! Appendix N B-N1 measurement (zenbench, paired/interleaved): the fused
//! folded-944 score+attribution entry vs the score-only folded-944
//! extraction, plus the loop-shape two-call baseline — all serial (the jxl
//! loop runs `with_parallel(false)`), reference precomputed once.
//!
//! Run: `cargo bench --bench fused944_bench -p zensim \
//!        --features custom-profiles,feature-regime-v2`

use zensim::profile::ProfileParams;
use zensim::{DiffmapWeighting, RgbSlice, Zensim, ZensimProfile};

/// Deterministic textured pair (the attribution tests' content family).
fn test_pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let mut src = Vec::with_capacity(w * h);
    let mut dst = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let base = ((x * 255) / w) as u8;
            let tex = (((x * 7 + y * 13) % 32) * 3) as u8;
            let edge = if (y / 16) % 2 == 0 { 40 } else { 0 };
            let px = [
                base.wrapping_add(tex),
                base.wrapping_add(edge),
                (255 - base).wrapping_add(tex / 2),
            ];
            src.push(px);
            let q = |v: u8| (v / 12) * 12;
            let mut d = [q(px[0]), q(px[1]), q(px[2])];
            if x < w / 2 && y < h / 2 {
                d[0] = d[0].saturating_add(18);
            }
            dst.push(d);
        }
    }
    (src, dst)
}

fn main() {
    let (w, h) = (576usize, 576usize);
    let (src, dst) = test_pair(w, h);
    let params = ProfileParams::builder()
        .skip_score_mapping(true)
        .extrapolate_score(true)
        .extended_features(true)
        .build();
    let params: &'static ProfileParams = Box::leak(Box::new(params));
    let z: &'static Zensim = Box::leak(Box::new(
        Zensim::new(ZensimProfile::Custom {
            params,
            name: "fused944-bench",
        })
        .with_parallel(false),
    ));
    let src_static: &'static [[u8; 3]] = Box::leak(src.into_boxed_slice());
    let dst_static: &'static [[u8; 3]] = Box::leak(dst.into_boxed_slice());
    let rs = RgbSlice::new(src_static, w, h);
    let pre: &'static zensim::PrecomputedReference =
        Box::leak(Box::new(z.precompute_reference(&rs).unwrap()));
    let mut s944 = vec![0.0f64; 944];
    for (k, v) in s944.iter_mut().enumerate() {
        *v = if k % 3 == 0 { -1.0 } else { -0.25 } * (1.0 + (k % 7) as f64 * 0.1);
    }
    let s944: &'static [f64] = Box::leak(s944.into_boxed_slice());
    let toggles944 = zensim::feature_v2::V2NewFeatureToggles {
        append_block: true,
        append2_block: true,
        ..Default::default()
    };

    let result = zenbench::run(|suite| {
        suite.compare("folded944_compare_576", |group| {
            group.config().max_rounds(200);

            // Score-only: the canonical streaming 944 extraction (the
            // loop's score pass; scratch reused like the batch shape).
            group.bench("score_only_extraction", move |b| {
                let mut scratch = zensim::feature_v2::V2Scratch::new();
                b.iter(move || {
                    let rsv = RgbSlice::new(src_static, w, h);
                    let dsv = RgbSlice::new(dst_static, w, h);
                    let v2 = z
                        .compute_folded720_features_streaming(&rsv, &dsv, toggles944, &mut scratch)
                        .unwrap();
                    zenbench::black_box(v2.features()[943]);
                })
            });

            // The fused entry: score features + full attribution map.
            group.bench("fused_score_and_map", move |b| {
                let mut sess = zensim::Fused944Session::new();
                b.iter(move || {
                    let rsv = RgbSlice::new(src_static, w, h);
                    let dsv = RgbSlice::new(dst_static, w, h);
                    let (r, v2, a) = z
                        .compute_folded944_score_and_attribution(&rsv, pre, &dsv, s944, &mut sess)
                        .unwrap();
                    zenbench::black_box((
                        r.score(),
                        v2.features()[943],
                        a.query_rect(0, 0, 32, 32),
                    ));
                })
            });

            // The loop's CURRENT two-call folded-class baseline (Trained
            // diffmap for redistribution + extraction for score) — the
            // B-N2 reference shape.
            group.bench("loop_today_two_call", move |b| {
                let mut scratch = zensim::feature_v2::V2Scratch::new();
                b.iter(move || {
                    let rsv = RgbSlice::new(src_static, w, h);
                    let dsv = RgbSlice::new(dst_static, w, h);
                    let dm = z
                        .compute_with_ref_and_diffmap(pre, &dsv, DiffmapWeighting::Trained)
                        .unwrap();
                    let v2 = z
                        .compute_folded720_features_streaming(&rsv, &dsv, toggles944, &mut scratch)
                        .unwrap();
                    zenbench::black_box((dm.diffmap()[0], v2.features()[943]));
                })
            });
        });
    });
    let _ = result;
}
