//! **The v1-372 feature vector must be a pure function of its (ref, dist)
//! pair — same width, same bits, alone or in any batch.**
//!
//! Registered defect: `docs/DATASET_HISTORY.md` §3.26 / `benchmarks/
//! r1b_keyed_rebuild_2026-08-30.md` §8.5(d). Re-extracting the R1b eval
//! slices at v1-372 produced RAGGED tables — 453/6,953 imazen26,
//! 422/6,142 nonphoto, 493/7,717 hfnlproxy rows came out **279** wide
//! (3 scales × 3 ch × 31) instead of **372** (4 scales), carrying real
//! values. §3.26 attributed it to the BATCH ("not a pure function of the
//! pair"); that attribution is itself retracted by this gate — see the
//! module test list below and `benchmarks/v1_width_defect_2026-08-30.md`.
//!
//! MECHANISM (measured, not argued). The scale loop walks
//! `w = simd_padded_width(width)`, `h = height` and stops at `w < 8 ||
//! h < 8` (`streaming.rs::compute_multiscale_stats_streaming`), so a
//! 4-scale pyramid needs `simd_padded_width(W) >= 64 && H >= 64`.
//! `compute_with_config_inner` (every `Zensim::compute*`) guarantees that
//! by reflect-padding any sub-64 side to `MIN_PYRAMID_DIM` first. The
//! `training`-gated free functions `compute_zensim_with_config` /
//! `compute_zensim_with_ref_and_config` did NOT, so they silently emitted
//! a short vector for `W < 49` or `H < 64` — the exact size classes the
//! R1b slices are full of (`43x64`, `64x48`, `96x54`, `36x64`, …).
//!
//! The size table below is the REAL one: every distinct `scale<W>x<H>`
//! class observed among the 20,812 R1b slice rows, split by the width the
//! defective build emitted. `simd_padded_width` is `(W+15)&!15` (+16 when
//! that is >= 512 and an even multiple of 16), which is why `54x96` was
//! full (54 -> 64) while `48x64` was short (48 -> 48), and why the defect
//! is asymmetric in W vs H.

#![cfg(feature = "training")]

mod common;

use common::generators::{distort_block_artifacts, gen_value_noise};
use zensim::{RgbSlice, Zensim, ZensimConfig, ZensimProfile, compute_zensim_with_config};

/// Full v1 `with-iw` 372-feature config — the one both v1-372 extractors
/// (`zensim-bench extract_features_372col`, `zensim/examples/v2_ab_extract`
/// at `ZENSIM_AB_MODE=v1`) use.
fn v1_config() -> ZensimConfig {
    let mut cfg = ZensimConfig::default();
    cfg.compute_all_features = true;
    cfg.extended_features = true;
    cfg.compute_iw_features = true;
    // Match the extractors: inner multithreading off, outer loop parallel.
    cfg.allow_multithreading = false;
    cfg
}

/// `(w, h, emitted_width_before_the_fix)` for every distinct size class
/// observed in the three R1b slices. 279 = the short (3-scale) rows.
///
/// Sizes with 279 are exactly `simd_padded_width(w) < 64 || h < 64`.
const R1B_SIZE_CLASSES: &[(usize, usize, usize)] = &[
    // --- short classes (all 1,368 short rows across the three slices) ---
    (43, 64, 279),
    (64, 43, 279),
    (48, 64, 279),
    (64, 48, 279),
    (44, 64, 279),
    (36, 64, 279),
    (42, 64, 279),
    (96, 54, 279),
    (64, 45, 279),
    (64, 36, 279),
    (64, 55, 279),
    (47, 64, 279),
    (41, 64, 279),
    // --- full classes that LOOK like they should be short but are not:
    //     54 and 62 both pad up to 64, so only H decides.
    (54, 96, 372),
    (62, 96, 372),
    (64, 64, 372),
    (49, 64, 372),
    // --- ordinary classes -------------------------------------------------
    (128, 96, 372),
    (512, 384, 372),
    (96, 128, 372),
    (1024, 768, 372),
];

fn pair(w: usize, h: usize, seed: u32) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let r = gen_value_noise(w, h, seed);
    let d = distort_block_artifacts(&r, w, h);
    (r, d)
}

fn v1_features(w: usize, h: usize, seed: u32) -> Vec<f64> {
    let (r, d) = pair(w, h, seed);
    compute_zensim_with_config(&r, &d, w, h, v1_config())
        .unwrap_or_else(|e| panic!("v1 compute failed at {w}x{h}: {e:?}"))
        .features()
        .to_vec()
}

/// THE GATE. `compute_zensim_with_config` must emit 372 at every size the
/// R1b slices contain — including the 13 classes that produced 279.
#[test]
fn v1_free_fn_emits_372_at_every_r1b_size_class() {
    let mut short = Vec::new();
    for &(w, h, was) in R1B_SIZE_CLASSES {
        let n = v1_features(w, h, 0xC0FFEE).len();
        if n != 372 {
            short.push(format!("{w}x{h}: emitted {n} (pre-fix {was})"));
        }
    }
    assert!(
        short.is_empty(),
        "v1-372 must be 372 wide at EVERY size; short at: {short:?}"
    );
}

/// The same width contract on the reference-reuse (grouped-flow) entry.
#[test]
fn v1_with_ref_entry_emits_372_at_every_r1b_size_class() {
    use zensim::{compute_zensim_with_ref_and_config, precompute_reference_with_scales};
    let cfg = v1_config();
    let mut bad = Vec::new();
    for &(w, h, _) in R1B_SIZE_CLASSES {
        let (r, d) = pair(w, h, 0x51DE);
        let pre = precompute_reference_with_scales(&r, w, h, cfg.num_scales)
            .unwrap_or_else(|e| panic!("precompute failed at {w}x{h}: {e:?}"));
        let n = compute_zensim_with_ref_and_config(&pre, &d, w, h, cfg)
            .unwrap_or_else(|e| panic!("with-ref compute failed at {w}x{h}: {e:?}"))
            .features()
            .len();
        if n != 372 {
            bad.push(format!("{w}x{h}: {n}"));
        }
    }
    assert!(bad.is_empty(), "with-ref entry emitted short vectors: {bad:?}");
}

/// The product path was never affected — kept as the differential that
/// localizes the defect to the training free functions.
#[test]
fn product_compute_path_was_always_372() {
    let z = Zensim::new(ZensimProfile::codec_target());
    for &(w, h, _) in R1B_SIZE_CLASSES {
        let (r, d) = pair(w, h, 0xB0B);
        let n = z
            .compute_extended_features(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
            .unwrap_or_else(|e| panic!("compute_extended_features failed at {w}x{h}: {e:?}"))
            .features()
            .len();
        assert_eq!(n, 372, "product path must be 372 wide at {w}x{h}");
    }
}

/// §3.26's headline claim, tested directly: run each pair ALONE, then in a
/// 5-pair batch, then in the full batch, then in the full batch REVERSED,
/// then through a rayon-parallel batch. Every composition must produce
/// BIT-IDENTICAL feature vectors — the width is a function of the pair.
#[test]
fn width_and_bits_are_independent_of_batch_composition() {
    let sizes: Vec<(usize, usize)> = R1B_SIZE_CLASSES.iter().map(|&(w, h, _)| (w, h)).collect();

    // (1) alone — one process-order-isolated compute per pair.
    let alone: Vec<Vec<f64>> = sizes.iter().map(|&(w, h)| v1_features(w, h, 7)).collect();

    // (2) five-at-a-time batches (§3.26's "5 of them run alone").
    let mut batched5: Vec<Vec<f64>> = Vec::new();
    for chunk in sizes.chunks(5) {
        for &(w, h) in chunk {
            batched5.push(v1_features(w, h, 7));
        }
    }

    // (3) the full batch, in order; (4) the full batch, reversed.
    let full: Vec<Vec<f64>> = sizes.iter().map(|&(w, h)| v1_features(w, h, 7)).collect();
    let mut rev: Vec<Vec<f64>> = sizes
        .iter()
        .rev()
        .map(|&(w, h)| v1_features(w, h, 7))
        .collect();
    rev.reverse();

    for (i, &(w, h)) in sizes.iter().enumerate() {
        assert_eq!(
            alone[i].len(),
            372,
            "{w}x{h} must be 372 wide run alone (got {})",
            alone[i].len()
        );
        assert_eq!(alone[i], batched5[i], "{w}x{h}: alone != 5-batch");
        assert_eq!(alone[i], full[i], "{w}x{h}: alone != full batch");
        assert_eq!(alone[i], rev[i], "{w}x{h}: alone != reversed full batch");
    }
}

/// Batch composition under the extractors' real shape: an OUTER rayon
/// `par_iter` over pairs with zensim's inner threading off. §3.26 measured
/// `RAYON_NUM_THREADS` 1/2/8 giving the same short count; this asserts the
/// stronger property — the bits do not move at all.
#[cfg(feature = "threads")]
#[test]
fn outer_parallel_batch_matches_sequential_bit_for_bit() {
    use rayon::prelude::*;
    let sizes: Vec<(usize, usize)> = R1B_SIZE_CLASSES.iter().map(|&(w, h, _)| (w, h)).collect();
    let seq: Vec<Vec<f64>> = sizes.iter().map(|&(w, h)| v1_features(w, h, 11)).collect();
    let par: Vec<Vec<f64>> = sizes
        .par_iter()
        .map(|&(w, h)| v1_features(w, h, 11))
        .collect();
    assert_eq!(seq, par, "outer-parallel batch must be bit-identical");
    assert!(par.iter().all(|f| f.len() == 372));
}

/// The scale-count rule, stated as an executable contract so a future
/// change to `simd_padded_width` or `MIN_PYRAMID_DIM` cannot re-open the
/// hole silently: every side from 1..=130 (square and both oblong
/// orientations against 64) is 372 wide.
#[test]
fn every_small_size_is_372_wide() {
    let mut short = Vec::new();
    // The free fn contracts `ImageTooSmall` below 8px (unchanged by this fix);
    // the pyramid hole lived in 8..64, which this sweeps exhaustively.
    for n in 8usize..=130 {
        for &(w, h) in &[(n, n), (n, 64), (64, n), (n, 96), (96, n)] {
            let len = v1_features(w, h, 0x5EED).len();
            if len != 372 {
                short.push(format!("{w}x{h}={len}"));
            }
        }
    }
    assert!(short.is_empty(), "short vectors at: {short:?}");
}

/// The sub-64 sizes the R1b slices actually contain, used to sweep the
/// PUBLIC v1 surface below.
const SUB64: &[(usize, usize)] = &[(43, 64), (64, 48), (96, 54), (48, 64), (36, 64), (62, 96)];

/// **Public-surface audit, executable.** Every v1 entry point that walks the
/// pyramid must handle a sub-`MIN_PYRAMID_DIM` side — no panic, no short
/// vector. Found `compute_with_ref_into` (a PRODUCT API, not training-gated)
/// asserting `scale 0 width mismatch` on 43x64: it fed an unpadded distorted
/// to a reflect-padded `PrecomputedReference`.
#[test]
fn every_public_v1_entry_handles_sub64_sides() {
    let z = Zensim::new(ZensimProfile::codec_target());
    for &(w, h) in SUB64 {
        let (r, d) = pair(w, h, 0xA11CE);
        let rs = RgbSlice::new(&r, w, h);
        let ds = RgbSlice::new(&d, w, h);

        // Buffered family.
        assert!(z.compute(&rs, &ds).is_ok(), "compute at {w}x{h}");
        let ext = z.compute_extended_features(&rs, &ds).unwrap();
        assert_eq!(ext.features().len(), 372, "compute_extended_features {w}x{h}");
        let all = z.compute_all_features(&rs, &ds).unwrap();
        assert_eq!(all.features().len(), 372, "compute_all_features {w}x{h}");

        // Precomputed-reference family.
        let pre = z.precompute_reference(&rs).unwrap();
        assert!(z.compute_with_ref(&pre, &ds).is_ok(), "compute_with_ref {w}x{h}");
        let mut scratch = zensim::ZensimScratch::default();
        assert!(
            z.compute_with_ref_into(&pre, &ds, &mut scratch).is_ok(),
            "compute_with_ref_into {w}x{h}"
        );

        // Strip family (routes sub-64 back to the buffered path).
        assert!(
            z.compute_streaming_strips_default(&rs, &ds).is_ok(),
            "compute_streaming_strips_default {w}x{h}"
        );
        assert!(
            z.compute_with_ref_streaming_strips_default(&pre, &ds).is_ok(),
            "compute_with_ref_streaming_strips_default {w}x{h}"
        );

        // Diffmap family.
        assert!(
            z.compute_with_diffmap(&rs, &ds, zensim::DiffmapWeighting::default())
                .is_ok(),
            "compute_with_diffmap {w}x{h}"
        );
        assert!(
            z.compute_with_ref_and_diffmap(&pre, &ds, zensim::DiffmapWeighting::default())
                .is_ok(),
            "compute_with_ref_and_diffmap {w}x{h}"
        );

        // Training free functions.
        assert_eq!(
            compute_zensim_with_config(&r, &d, w, h, v1_config())
                .unwrap()
                .features()
                .len(),
            372,
            "compute_zensim_with_config {w}x{h}"
        );
    }
}

/// `compute_with_ref_into` (scratch-reusing encoder-loop entry) must agree
/// bit-for-bit with `compute_with_ref` — including at sub-64 sizes, where it
/// previously panicked.
#[test]
fn compute_with_ref_into_matches_compute_with_ref_at_sub64() {
    let z = Zensim::new(ZensimProfile::codec_target());
    let mut scratch = zensim::ZensimScratch::default();
    for &(w, h) in SUB64 {
        let (r, d) = pair(w, h, 0xD00D);
        let rs = RgbSlice::new(&r, w, h);
        let ds = RgbSlice::new(&d, w, h);
        let pre = z.precompute_reference(&rs).unwrap();
        let a = z.compute_with_ref(&pre, &ds).unwrap();
        let b = z.compute_with_ref_into(&pre, &ds, &mut scratch).unwrap();
        assert_eq!(
            a.score().to_bits(),
            b.score().to_bits(),
            "{w}x{h}: compute_with_ref_into score != compute_with_ref"
        );
        assert_eq!(a.features(), b.features(), "{w}x{h}: feature mismatch");
    }
}

// ---------------------------------------------------------------------------
// Purity w.r.t. the RAYON POOL SIZE.
//
// The same "pure function of its pair" contract, on the other axis that broke
// it. Measured 2026-08-30 (`benchmarks/v1_extractor_drift_2026-08-30.md`):
// built at `58e6f8d8` — the commit the canonical 2026-05-15 372-col tables
// record as their own build — the v1-372 vector's masked (f228..300) and IW
// (f300..372) blocks are a function of `RAYON_NUM_THREADS`. On the 504
// KonJND-JPEG pairs, T=1 / 2 / 8 / 28 give four DIFFERENT output files, and
// T=1 vs T=28 moves 100 % of rows on 100 % of the 144 masked/IW slots, by up
// to |Δ| 0.086 — while basic (f0..156) and peaks (f156..228) stay inside the
// golden tolerance on every row.
//
// Two commits produced it together:
//   * `2dab8f30` (2026-05-17) — the activity map for masked/IW read
//     `bufs.mu1` at strip-OVERLAP rows, which the fused V-blur never writes;
//     the contents there were whatever the buffer-reuse cascade left (zero,
//     the previous channel's source, the previous strip's mask). Replaced by
//     a per-channel `H_blur(src)` reference. (`docs/PRINCIPLED_ACTIVITY.md`.)
//   * `6af83b60` (2026-06-09) — the band layout was
//     `rayon::current_num_threads().min(total_strips)`, so the thread count
//     chose where those overlap rows fell. Now geometry-only.
//
// basic/peaks survived because the strip aggregator's own 1e-6 parity gate
// bounds them; masked/IW had no such gate because they were reading data no
// gate covered. This test is that missing gate, stated on the whole vector.
// ---------------------------------------------------------------------------

/// The v1-372 vector must not depend on how many threads rayon happens to
/// have — on the free-function extractor path or on the product `compute`.
///
/// Sizes are chosen to span several strips (`STRIP_INNER = 32`), which is
/// the only regime where a band layout exists to disagree about.
#[cfg(feature = "threads")]
#[test]
fn v1_372_is_bit_identical_across_rayon_pool_sizes() {
    let mut cfg = v1_config();
    // The product default: zensim's OWN band parallelism on. `v1_config`
    // turns it off to match the extractors' outer-parallel shape; the
    // thread-count dependence lived in the inner layout, so this test must
    // exercise it.
    cfg.allow_multithreading = true;

    let z = Zensim::new(ZensimProfile::codec_target());

    for &(w, h) in &[(256usize, 256usize), (96, 320), (320, 96), (64, 512)] {
        let (r, d) = pair(w, h, 0xBADC0DE);
        let mut ref_free: Option<Vec<f64>> = None;
        let mut ref_prod: Option<(u64, Vec<f64>)> = None;

        for threads in [1usize, 2, 3, 5, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("build rayon pool");

            let free = pool.install(|| {
                compute_zensim_with_config(&r, &d, w, h, cfg)
                    .unwrap_or_else(|e| panic!("v1 compute failed at {w}x{h}: {e:?}"))
                    .features()
                    .to_vec()
            });
            assert_eq!(free.len(), 372, "{w}x{h} @ {threads} threads: not 372 wide");
            match &ref_free {
                None => ref_free = Some(free),
                Some(first) => assert_eq!(
                    *first, free,
                    "{w}x{h}: compute_zensim_with_config moved between 1 and {threads} \
                     rayon threads — the v1-372 vector is not a pure function of its pair"
                ),
            }

            let prod = pool.install(|| {
                let rs = RgbSlice::new(&r, w, h);
                let ds = RgbSlice::new(&d, w, h);
                let out = z.compute(&rs, &ds).expect("product compute");
                (out.score().to_bits(), out.features().to_vec())
            });
            match &ref_prod {
                None => ref_prod = Some(prod),
                Some(first) => assert_eq!(
                    *first, prod,
                    "{w}x{h}: Zensim::compute moved between 1 and {threads} rayon threads"
                ),
            }
        }
    }
}

/// The same invariant on the block that actually broke, stated separately so
/// a failure names it: masked (f228..300) + IW (f300..372) must be
/// thread-invariant. These are the 144 slots that moved on 100 % of rows at
/// `58e6f8d8`.
#[cfg(feature = "threads")]
#[test]
fn v1_masked_and_iw_blocks_are_thread_invariant() {
    let mut cfg = v1_config();
    cfg.allow_multithreading = true;
    let (w, h) = (256usize, 256usize);
    let (r, d) = pair(w, h, 0x5CA1E);

    let run = |threads: usize| -> Vec<f64> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("build rayon pool")
            .install(|| {
                compute_zensim_with_config(&r, &d, w, h, cfg)
                    .expect("v1 compute")
                    .features()[228..372]
                    .to_vec()
            })
    };
    let one = run(1);
    for threads in [2usize, 4, 7, 16] {
        assert_eq!(
            one,
            run(threads),
            "masked+IW (f228..372) moved between 1 and {threads} rayon threads"
        );
    }
}
