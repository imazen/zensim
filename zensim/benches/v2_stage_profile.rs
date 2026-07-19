//! Phase-2 item 2: where-the-time-goes stage profile of the v1 hot path.
//!
//! Rather than adding new instrumentation (a diagnostic feature, or a
//! callgrind harness -- both real options the brief offered), this reuses
//! `ZensimConfig`'s EXISTING block toggles as an existing harness: `basic`
//! (peaks are always fused into the same SIMD kernel, so "basic-only" is
//! not obtainable separately -- see `metric.rs` `FeatureView::new`'s doc,
//! "basic-only 156-element vectors are no longer generated"), `+masked`
//! (`extended_features`), `+IW` (`compute_iw_features`). The marginal delta
//! between successive configs IS the masked-block cost and the IW-block
//! cost respectively -- zero new code, reuses the same knobs
//! `Zensim::compute_extended_features` already exposes.
//!
//! ```sh
//! cargo bench -p zensim --features training,feature-regime-v2 --bench v2_stage_profile
//! ```
//!
//! `docs/FEATURE_V2_SPEC_2026-07-18.md` SS A.12 turns these numbers into a
//! where-the-time-goes percentage table (basic+peak / masked-delta /
//! IW-delta) at both a representative and the gate size.

#[path = "../examples/support/zen_io.rs"]
mod zen_io;

use std::sync::Arc;
use zenbench::prelude::*;
use zensim::{ZensimConfig, compute_zensim_with_config};

const SIZES: &[usize] = &[256, 576, 1024, 2048];
const REF_PATH: &str = "/mnt/v/output/zensim/diffmap-coherence-2026-07-18/city.png";
const DIST_PATH: &str = "/mnt/v/output/zensim/diffmap-coherence-2026-07-18/city_q50.jpg";

// ZensimConfig is #[non_exhaustive] -- mutate fields on a Default instance.
fn cfg_basic_peak() -> ZensimConfig {
    let mut cfg = ZensimConfig::default();
    cfg.compute_all_features = true;
    cfg.extended_features = false;
    cfg.compute_iw_features = false;
    cfg.allow_multithreading = false;
    cfg
}
fn cfg_plus_masked() -> ZensimConfig {
    let mut cfg = cfg_basic_peak();
    cfg.extended_features = true;
    cfg.compute_iw_features = false;
    cfg
}
fn cfg_plus_iw() -> ZensimConfig {
    let mut cfg = cfg_basic_peak();
    cfg.extended_features = true;
    cfg.compute_iw_features = true;
    cfg
}

fn bench_stages(suite: &mut Suite) {
    let (r_px, rw, rh) = zen_io::decode_rgb8(std::path::Path::new(REF_PATH));
    let (d_px, dw, dh) = zen_io::decode_rgb8(std::path::Path::new(DIST_PATH));

    for &size in SIZES {
        let rs: Arc<Vec<[u8; 3]>> = Arc::new(zen_io::resize_rgb8(&r_px, rw, rh, size, size));
        let ds: Arc<Vec<[u8; 3]>> = Arc::new(zen_io::resize_rgb8(&d_px, dw, dh, size, size));
        let pixels = (size * size) as u64;

        suite.group(format!("stage_{size}x{size}"), |g| {
            g.throughput(Throughput::Elements(pixels));

            {
                let (rs, ds) = (Arc::clone(&rs), Arc::clone(&ds));
                g.bench("228_basic_plus_peak", move |b| {
                    let (rs, ds) = (Arc::clone(&rs), Arc::clone(&ds));
                    b.with_input(move || (Arc::clone(&rs), Arc::clone(&ds)))
                        .run(move |(rs, ds)| {
                            let res =
                                compute_zensim_with_config(&rs, &ds, size, size, cfg_basic_peak()).unwrap();
                            std::hint::black_box(res.features().len());
                            (rs, ds)
                        })
                });
            }

            {
                let (rs, ds) = (Arc::clone(&rs), Arc::clone(&ds));
                g.bench("300_plus_masked", move |b| {
                    let (rs, ds) = (Arc::clone(&rs), Arc::clone(&ds));
                    b.with_input(move || (Arc::clone(&rs), Arc::clone(&ds)))
                        .run(move |(rs, ds)| {
                            let res =
                                compute_zensim_with_config(&rs, &ds, size, size, cfg_plus_masked()).unwrap();
                            std::hint::black_box(res.features().len());
                            (rs, ds)
                        })
                });
            }

            {
                let (rs, ds) = (Arc::clone(&rs), Arc::clone(&ds));
                g.bench("372_plus_iw", move |b| {
                    let (rs, ds) = (Arc::clone(&rs), Arc::clone(&ds));
                    b.with_input(move || (Arc::clone(&rs), Arc::clone(&ds)))
                        .run(move |(rs, ds)| {
                            let res =
                                compute_zensim_with_config(&rs, &ds, size, size, cfg_plus_iw()).unwrap();
                            std::hint::black_box(res.features().len());
                            (rs, ds)
                        })
                });
            }
        });
    }
}

zenbench::main!(bench_stages);
