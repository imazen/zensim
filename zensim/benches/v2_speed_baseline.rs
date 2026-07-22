//! Phase-2 item 1: v1 372-feature vs v2 feature-extraction wall-time baseline.
//!
//! Uses zenbench (interleaved round-robin, paired statistics, hardware
//! timers) per project convention -- NOT a hand-rolled `Instant::now()`
//! loop. Decode/resize (via zen* crates: zenpng, zenjpeg, zenresize -- NOT
//! the `image` crate) happens ONCE, before any group runs; each bench's
//! `with_input` clones the already-decoded buffer (cheap `Vec::clone`, not
//! a re-decode) so the timed region is extraction only.
//!
//! Four sizes (256^2, 576^2 = the gb82 `city` reference's native size,
//! 1024^2, 2048^2), single- and multi-thread, v1 (372-feature: basic, peak,
//! masked, and IW all on -- `compute_zensim_with_config` with
//! `extended_features`, `compute_iw_features`, and `compute_all_features`
//! all `true`) vs v2 (`Zensim::compute_v2_features`). No
//! `-C target-cpu=native` in the invocation used for the numbers in
//! `docs/FEATURE_V2_SPEC_2026-07-18.md` SS A.12.
//!
//! ```sh
//! cargo bench -p zensim --features training,feature-regime-v2 --bench v2_speed_baseline
//! ```
//!
//! zenbench's own `mean ± MAD` (median absolute deviation) is reported
//! (its terminal/JSON convention), not a hand-rolled median-of-N -- see
//! docs/FEATURE_V2_SPEC_2026-07-18.md SS A.12 for the OLS alpha+beta*pixels
//! fit computed from these numbers.

#[path = "../examples/support/zen_io.rs"]
mod zen_io;

use std::sync::Arc;
use zenbench::prelude::*;
use zensim::feature_v2::{V2NewFeatureToggles, V2Scratch};
use zensim::{RgbSlice, Zensim, ZensimConfig, ZensimProfile, compute_zensim_with_config};

const SIZES: &[usize] = &[256, 576, 1024, 2048];
const REF_PATH: &str = "/mnt/v/output/zensim/diffmap-coherence-2026-07-18/city.png";
const DIST_PATH: &str = "/mnt/v/output/zensim/diffmap-coherence-2026-07-18/city_q50.jpg";

// ZensimConfig is #[non_exhaustive] (constructible only via Default from
// outside the crate) -- mutate fields on a Default instance rather than a
// struct-literal-with-spread.
fn v1_config(parallel: bool) -> ZensimConfig {
    let mut cfg = ZensimConfig::default();
    cfg.compute_all_features = true;
    cfg.extended_features = true;
    cfg.compute_iw_features = true;
    cfg.allow_multithreading = parallel;
    cfg
}

fn bench_extraction(suite: &mut Suite) {
    let (r_px, rw, rh) = zen_io::decode_rgb8(std::path::Path::new(REF_PATH));
    let (d_px, dw, dh) = zen_io::decode_rgb8(std::path::Path::new(DIST_PATH));

    for &size in SIZES {
        // Arc: zenbench's `with_input` closures must be `'static`, so a
        // loop-local `Vec` can't be borrowed into them. Arc gives cheap
        // (refcount) sharing across the 4 benches in this group instead of
        // 4x full-buffer clones per group.
        let rs: Arc<Vec<[u8; 3]>> = Arc::new(zen_io::resize_rgb8(&r_px, rw, rh, size, size));
        let ds: Arc<Vec<[u8; 3]>> = Arc::new(zen_io::resize_rgb8(&d_px, dw, dh, size, size));
        let pixels = (size * size) as u64;

        suite.group(format!("extract_{size}x{size}"), |g| {
            g.throughput(Throughput::Elements(pixels));
            // Perf pass (§A.13): this box is shared with several other
            // concurrent agent sessions (real, observed ambient load during
            // the perf pass — `uptime`/`ps` showed 10+ concurrent `claude`
            // processes). The default 5 min_rounds / 10s max_time produced
            // "only 4 rounds" warnings and visible run-to-run drift (v1's
            // own number moved 66.6ms -> 70.5ms +-5.2ms between back-to-back
            // invocations). Widen both so a single invocation is trustworthy
            // even under shared-box noise.
            g.config()
                .min_rounds(15)
                .max_time(std::time::Duration::from_secs(30));

            {
                let (rs, ds) = (Arc::clone(&rs), Arc::clone(&ds));
                g.bench("v1_372feat_1thread", move |b| {
                    let (rs, ds) = (Arc::clone(&rs), Arc::clone(&ds));
                    b.with_input(move || (Arc::clone(&rs), Arc::clone(&ds)))
                        .run(move |(rs, ds)| {
                            let cfg = v1_config(false);
                            let res =
                                compute_zensim_with_config(&rs, &ds, size, size, cfg).unwrap();
                            std::hint::black_box(res.features().len());
                            (rs, ds)
                        })
                });
            }

            {
                let (rs, ds) = (Arc::clone(&rs), Arc::clone(&ds));
                g.bench("v1_372feat_Nthread", move |b| {
                    let (rs, ds) = (Arc::clone(&rs), Arc::clone(&ds));
                    b.with_input(move || (Arc::clone(&rs), Arc::clone(&ds)))
                        .run(move |(rs, ds)| {
                            let cfg = v1_config(true);
                            let res =
                                compute_zensim_with_config(&rs, &ds, size, size, cfg).unwrap();
                            std::hint::black_box(res.features().len());
                            (rs, ds)
                        })
                });
            }

            {
                let (rs, ds) = (Arc::clone(&rs), Arc::clone(&ds));
                g.bench("v2_bounded_1thread", move |b| {
                    let (rs, ds) = (Arc::clone(&rs), Arc::clone(&ds));
                    b.with_input(move || (Arc::clone(&rs), Arc::clone(&ds)))
                        .run(move |(rs, ds)| {
                            let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
                            let source = RgbSlice::new(&rs, size, size);
                            let distorted = RgbSlice::new(&ds, size, size);
                            let res = z.compute_v2_features(&source, &distorted).unwrap();
                            std::hint::black_box(res.features().len());
                            (rs, ds)
                        })
                });
            }

            {
                let (rs, ds) = (Arc::clone(&rs), Arc::clone(&ds));
                g.bench("v2_bounded_Nthread", move |b| {
                    let (rs, ds) = (Arc::clone(&rs), Arc::clone(&ds));
                    b.with_input(move || (Arc::clone(&rs), Arc::clone(&ds)))
                        .run(move |(rs, ds)| {
                            let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(true);
                            let source = RgbSlice::new(&rs, size, size);
                            let distorted = RgbSlice::new(&ds, size, size);
                            let res = z.compute_v2_features(&source, &distorted).unwrap();
                            std::hint::black_box(res.features().len());
                            (rs, ds)
                        })
                });
            }

            // Ref-reuse pass: the amortized sweep steady state — reference
            // prepared ONCE (outside the timed region), scratch reused
            // across iterations. This is the per-pair cost a ref-grouped
            // extraction driver actually pays; v2_bounded_1thread minus
            // this = the reference-side share the prepared path amortizes
            // away.
            {
                let (rs, ds) = (Arc::clone(&rs), Arc::clone(&ds));
                g.bench("v2_with_ref_1thread", move |b| {
                    let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
                    let prepared = {
                        let source = RgbSlice::new(&rs, size, size);
                        Arc::new(z.prepare_v2_reference(&source).unwrap())
                    };
                    let ds = Arc::clone(&ds);
                    b.with_input(move || (Arc::clone(&prepared), Arc::clone(&ds), V2Scratch::new()))
                        .run(move |(prepared, ds, mut scratch)| {
                            let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
                            let distorted = RgbSlice::new(&ds, size, size);
                            let res = z
                                .compute_v2_features_with_ref_and_scratch(
                                    &prepared,
                                    &distorted,
                                    V2NewFeatureToggles::default(),
                                    &mut scratch,
                                )
                                .unwrap();
                            std::hint::black_box(res.features().len());
                            (prepared, ds, scratch)
                        })
                });
            }

            // The one-time reference-preparation cost (reflect-pad skip +
            // XYB convert + 3-level pyramid). Amortization math:
            // with_ref + prepare/N vs v2_bounded, N = variants per ref.
            {
                let rs = Arc::clone(&rs);
                g.bench("v2_prepare_ref_1thread", move |b| {
                    let rs = Arc::clone(&rs);
                    b.with_input(move || Arc::clone(&rs)).run(move |rs| {
                        let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
                        let source = RgbSlice::new(&rs, size, size);
                        let prepared = z.prepare_v2_reference(&source).unwrap();
                        std::hint::black_box(prepared.width());
                        rs
                    })
                });
            }
        });
    }
}

zenbench::main!(bench_extraction);
