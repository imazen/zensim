//! Phase-2 item 4: per-feature-group marginal cost, at the gate size
//! (1024x1024, single-thread) only -- kept as its own small bench (rather
//! than folded into `v2_speed_baseline`'s 4-size sweep) so it runs in
//! seconds, not minutes, and can be re-run cheaply while tuning toggles.
//!
//! ```sh
//! cargo bench -p zensim --features training,feature-regime-v2 --bench v2_feature_group_cost
//! ```

#[path = "../examples/support/zen_io.rs"]
mod zen_io;

use std::sync::Arc;
use zenbench::prelude::*;
use zensim::feature_v2::V2NewFeatureToggles;
use zensim::{RgbSlice, Zensim, ZensimProfile};

const SIZE: usize = 1024;
const REF_PATH: &str = "/mnt/v/output/zensim/diffmap-coherence-2026-07-18/city.png";
const DIST_PATH: &str = "/mnt/v/output/zensim/diffmap-coherence-2026-07-18/city_q50.jpg";

fn all_off() -> V2NewFeatureToggles {
    V2NewFeatureToggles {
        gradient_features: false,
        transducer_bank: false,
        blockiness: false,
        // Deprecate-by-mask toggle (default off) — listed explicitly so this
        // bench keeps compiling when new toggles are added with defaults.
        ..V2NewFeatureToggles::default()
    }
}

fn bench_groups(suite: &mut Suite) {
    let (r_px, rw, rh) = zen_io::decode_rgb8(std::path::Path::new(REF_PATH));
    let (d_px, dw, dh) = zen_io::decode_rgb8(std::path::Path::new(DIST_PATH));
    let rs: Arc<Vec<[u8; 3]>> = Arc::new(zen_io::resize_rgb8(&r_px, rw, rh, SIZE, SIZE));
    let ds: Arc<Vec<[u8; 3]>> = Arc::new(zen_io::resize_rgb8(&d_px, dw, dh, SIZE, SIZE));
    let pixels = (SIZE * SIZE) as u64;

    let variants: &[(&str, V2NewFeatureToggles)] = &[
        ("v0_none_of_the_7_new", all_off()),
        (
            "v1_plus_gradient_group",
            V2NewFeatureToggles {
                gradient_features: true,
                ..all_off()
            },
        ),
        (
            "v2_plus_transducer_bank",
            V2NewFeatureToggles {
                transducer_bank: true,
                ..all_off()
            },
        ),
        (
            "v3_plus_blockiness",
            V2NewFeatureToggles {
                blockiness: true,
                ..all_off()
            },
        ),
        ("v4_all_7_new_features", V2NewFeatureToggles::default()),
    ];

    suite.group("group_cost_1024x1024_1thread", |g| {
        g.throughput(Throughput::Elements(pixels));
        for &(name, toggles) in variants {
            let (rs, ds) = (Arc::clone(&rs), Arc::clone(&ds));
            g.bench(name, move |b| {
                let (rs, ds) = (Arc::clone(&rs), Arc::clone(&ds));
                b.with_input(move || (Arc::clone(&rs), Arc::clone(&ds)))
                    .run(move |(rs, ds)| {
                        let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
                        let source = RgbSlice::new(&rs, SIZE, SIZE);
                        let distorted = RgbSlice::new(&ds, SIZE, SIZE);
                        let res = z
                            .compute_v2_features_with_toggles(&source, &distorted, toggles)
                            .unwrap();
                        std::hint::black_box(res.features().len());
                        (rs, ds)
                    })
            });
        }
    });
}

zenbench::main!(bench_groups);
