//! **Feature-calculation invariants** — the assertable half of the 2026-09-05
//! feature-defect audit (`docs/FEATURE_DEFECTS_AUDIT_2026-09-05.md`).
//!
//! The audit asked one question — *do we have bugs in feature calculations?* —
//! and answered it by measuring invariants an extractor is supposed to satisfy.
//! Everything in this file is a property that MEASURED TRUE at the audit's HEAD
//! and that a future change must not break silently. The census half (which
//! slots are non-monotone under a distortion ladder, how far the tiers drift)
//! lives in `zensim/examples/feature_invariant_probe.rs`, because its
//! deliverable is a list rather than a bar.
//!
//! What is deliberately NOT here: a bar on a number that is a MODEL property
//! (a bake's score at near-identity) or a DATA property (a stored root's era).
//! Those move when a bake or a root is rotated and would make this file a
//! tripwire for unrelated work; they are registered in
//! `benchmarks/eval_annotations.json` instead.
//!
//! Existing neighbours, none of which this file duplicates:
//! `v1_golden_bytes.rs` (pinned byte fixtures), `v1_feature_width_pure_function.rs`
//! (width is a pure function of `(W,H)`; thread-invariance of masked/IW),
//! `fold_engine_parity.rs` (`ZensimResult` bit-identity across engines and
//! rayon pool sizes 1..16), `feature_v2::tests::v1_372_bit_exact_to_fold_at_every_width`
//! (the in-crate feature-block comparison). This file adds the invariants none
//! of them state: identity response, degenerate-input finiteness, the
//! fabricate-vs-compute disagreement, and the 28-thread rung.

#![cfg(all(feature = "training", feature = "feature-regime-v2"))]

use zensim::feature_v2::{V1FreeExtras, V1PoolsMode, V2NewFeatureToggles, V2Scratch};
use zensim::{RgbSlice, Zensim, ZensimConfig, ZensimProfile, compute_zensim_with_config};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Geometry set spanning the classes that have historically broken feature
/// extraction: tight vs non-tight width (`simd_padded_width`, the option-C
/// class), odd widths, the `h = 93` row-group class, and both sides of
/// `H_TILE_WIDTH = 1024` including a 1-column remainder tile.
const GEOMS: &[(usize, usize)] = &[
    (64, 64),
    (96, 64),
    (100, 96),
    (127, 93),
    (129, 96),
    (200, 150),
    (255, 96),
    (512, 96),
    (576, 96),
    (592, 80),
    (1153, 72),
    (2049, 40),
];

fn value_noise(w: usize, h: usize, seed: u32) -> Vec<[u8; 3]> {
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    let lat = 16usize;
    let (lw, lh) = (w / lat + 2, h / lat + 2);
    let grid: Vec<[f32; 3]> = (0..lw * lh)
        .map(|_| {
            [
                (next() % 256) as f32,
                (next() % 256) as f32,
                (next() % 256) as f32,
            ]
        })
        .collect();
    let _ = lh;
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let (gx, gy) = (x / lat, y / lat);
            let (fx, fy) = ((x % lat) as f32 / lat as f32, (y % lat) as f32 / lat as f32);
            let mut px = [0u8; 3];
            for c in 0..3 {
                let a = grid[gy * lw + gx][c];
                let b = grid[gy * lw + gx + 1][c];
                let cc = grid[(gy + 1) * lw + gx][c];
                let d = grid[(gy + 1) * lw + gx + 1][c];
                let top = a + (b - a) * fx;
                let bot = cc + (d - cc) * fx;
                let v = top + (bot - top) * fy;
                let speckle = ((x * 7 + y * 13 + c * 29) % 17) as f32 - 8.0;
                px[c] = (v + speckle).clamp(0.0, 255.0) as u8;
            }
            out.push(px);
        }
    }
    out
}

fn small_blur(src: &[[u8; 3]], w: usize, h: usize) -> Vec<[u8; 3]> {
    let mut out = src.to_vec();
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0u32; 3];
            let mut n = 0u32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let xx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let yy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    for c in 0..3 {
                        acc[c] += src[yy * w + xx][c] as u32;
                    }
                    n += 1;
                }
            }
            for c in 0..3 {
                out[y * w + x][c] = (acc[c] / n) as u8;
            }
        }
    }
    out
}

/// The one product 944 walk: all pools live, append + append2 on, and the
/// free/class-C tranche requested.
fn fold944(
    r: &[[u8; 3]],
    d: &[[u8; 3]],
    w: usize,
    h: usize,
    pools: V1PoolsMode,
    free: V1FreeExtras,
    v1_only: bool,
) -> Vec<f64> {
    let toggles = V2NewFeatureToggles {
        v1_pools: pools,
        append_block: !v1_only,
        append2_block: !v1_only,
        v1_only,
        free_extras: free,
        ..Default::default()
    };
    let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
    let mut scratch = V2Scratch::new();
    z.compute_folded720_features_streaming(
        &RgbSlice::new(r, w, h),
        &RgbSlice::new(d, w, h),
        toggles,
        &mut scratch,
    )
    .expect("944 fold walk")
    .features()
    .to_vec()
}

/// The "156+free" walk: v1 blocks computed, the v2-era blocks DECLARED (so the
/// layout is 944 and the free slots at `f720+` exist) but their compute
/// skipped, with the free tranche filling what it can for ~free.
fn free_set_walk(r: &[[u8; 3]], d: &[[u8; 3]], w: usize, h: usize, free: V1FreeExtras) -> Vec<f64> {
    let toggles = V2NewFeatureToggles {
        v1_pools: V1PoolsMode::Peaks,
        append_block: true,
        append2_block: true,
        v1_only: true,
        free_extras: free,
        ..Default::default()
    };
    let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
    let mut scratch = V2Scratch::new();
    z.compute_folded720_features_streaming(
        &RgbSlice::new(r, w, h),
        &RgbSlice::new(d, w, h),
        toggles,
        &mut scratch,
    )
    .expect("free-set walk")
    .features()
    .to_vec()
}

/// The config both v1-372 extractors use.
fn v1_config() -> ZensimConfig {
    let mut cfg = ZensimConfig::default();
    cfg.compute_all_features = true;
    cfg.extended_features = true;
    cfg.compute_iw_features = true;
    cfg.allow_multithreading = false;
    cfg
}

// The 944 block boundaries, and the two blocks' local strides.
const V2_BASE: usize = 372;
const V2_STRIDE: usize = 29;
const APPEND_BASE: usize = 720;
const APPEND_STRIDE: usize = 17;
const APPEND2_BASE: usize = 924;
const APPEND2_STRIDE: usize = 5;

fn v2_local(slot: usize) -> Option<usize> {
    (V2_BASE..APPEND_BASE)
        .contains(&slot)
        .then(|| (slot - V2_BASE) % V2_STRIDE)
}
fn append_local(slot: usize) -> Option<usize> {
    (APPEND_BASE..APPEND2_BASE)
        .contains(&slot)
        .then(|| (slot - APPEND_BASE) % APPEND_STRIDE)
}
fn append2_local(slot: usize) -> Option<usize> {
    (slot >= APPEND2_BASE).then(|| (slot - APPEND2_BASE) % APPEND2_STRIDE)
}

// ---------------------------------------------------------------------------
// (a) IDENTITY
// ---------------------------------------------------------------------------

/// **The identity vector at 944 is NOT the zero vector, and every slot that is
/// non-zero falls into exactly one of three registered classes.**
///
/// `docs/DATASET_HISTORY.md` §3.36 measured "286 of 944 slots non-zero" on the
/// 39-row real-image 944-POOLS probe and left the classification prose-only.
/// This gate turns it into a structural bar: a non-zero identity slot must be
///
/// 1. **REFERENCE-ONLY** — `GRAD_SRC_MEAN` (append local 16) and
///    `LUMA_MEAN_REF` (append2 local 2) are functions of the reference alone,
///    so `∂f/∂dist ≡ 0` and a non-zero value on `ref == dist` is CORRECT.
/// 2. **`PJND_FRAGILITY`** (v2 local 21) — a known formula artifact on an
///    undistorted pair (`benchmarks/free_features_classC_2026-09-04.md` §6.3;
///    DATASET_HISTORY §3.33 point 3, which measured the constant `1.0` a
///    v1-only walk produces). Registered, not fixed.
/// 3. **floating-point residue** — everything else, bounded here at 2e-3.
///
/// The bar that matters is (3): a NEW large residue at identity means a
/// difference feature stopped cancelling, which is exactly the shape of a
/// feature-calculation bug. The class-1 and class-2 slots are enumerated by
/// LOCAL INDEX rather than by absolute slot so the gate keeps working if the
/// scale count or the block bases move.
#[test]
fn identity_nonzero_slots_are_reference_only_pjnd_or_fp_residue() {
    use zensim::feature_v2::{idx, idx_append, idx_append2};
    const RESIDUE_BAR: f64 = 2e-3;
    let mut classes = (0usize, 0usize, 0usize);
    for &(w, h) in GEOMS {
        let r = value_noise(w, h, 0xC0FFEE);
        let f = fold944(
            &r,
            &r,
            w,
            h,
            V1PoolsMode::Full,
            V1FreeExtras::RawMomentsPlusBoundedErr,
            false,
        );
        assert_eq!(f.len(), 944, "{w}x{h}: width moved");
        for (i, &v) in f.iter().enumerate() {
            if v == 0.0 {
                continue;
            }
            let reference_only = append_local(i) == Some(idx_append::GRAD_SRC_MEAN)
                || append2_local(i) == Some(idx_append2::LUMA_MEAN_REF);
            let pjnd = v2_local(i) == Some(idx::PJND_FRAGILITY);
            if reference_only {
                classes.0 += 1;
            } else if pjnd {
                classes.1 += 1;
            } else {
                classes.2 += 1;
                assert!(
                    v.abs() <= RESIDUE_BAR,
                    "{w}x{h}: f{i} = {v:e} on an IDENTICAL pair. It is not a \
                     registered reference-only slot (GRAD_SRC_MEAN / \
                     LUMA_MEAN_REF), not PJND_FRAGILITY, and exceeds the {RESIDUE_BAR:e} \
                     floating-point-residue bar — a difference feature has \
                     stopped cancelling at zero distortion. See \
                     docs/FEATURE_DEFECTS_AUDIT_2026-09-05.md."
                );
            }
        }
    }
    assert!(
        classes.0 > 0 && classes.1 > 0 && classes.2 > 0,
        "all three identity classes must be exercised, saw {classes:?} — if a \
         class went empty the gate stopped testing what it claims to"
    );
}

/// **The 372 identity vector is FABRICATED, not computed — and it disagrees
/// with what the same code computes on the same pixels.**
///
/// Both product-facing SDR entries short-circuit `source == distorted` before
/// any walk: `metric.rs::identical_result` (behind every `Zensim::compute*`)
/// and the free function `compute_zensim_with_config` (behind both v1-372
/// extractors) each synthesise `(score = 100, raw_distance = 0, all-zero
/// features)`. So "the 372 identity vector is the zero vector" — a property
/// `zensim-validate`'s `dial_addressability` constant states as measured — has
/// never been a measurement of the extractor at that width; it is a property
/// of the short-circuit.
///
/// This gate states both halves so neither can drift silently: the fabricated
/// payload IS all-zero, and the computed one is NOT.
#[test]
fn identity_is_fabricated_by_the_short_circuit_and_differs_from_the_computed_vector() {
    for &(w, h) in &[(200usize, 150usize), (127, 93)] {
        let r = value_noise(w, h, 0xC0FFEE);

        // (i) the product entry fabricates.
        let z = Zensim::new(ZensimProfile::codec_target());
        let res = z
            .compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&r, w, h))
            .expect("compute");
        assert_eq!(res.score(), 100.0, "{w}x{h}: short-circuit score moved");
        assert!(
            res.features().iter().all(|&v| v == 0.0),
            "{w}x{h}: the fabricated identity payload is no longer all-zero"
        );

        // (ii) the v1-372 extractor entry fabricates the same way.
        let ext = compute_zensim_with_config(&r, &r, w, h, v1_config()).expect("v1-372");
        assert!(
            ext.features().iter().all(|&v| v == 0.0),
            "{w}x{h}: compute_zensim_with_config's identity payload is no longer all-zero"
        );

        // (iii) the same v1 block, COMPUTED, is not all-zero. If this ever
        // becomes all-zero the fabrication is redundant and can be deleted;
        // until then the two disagree and the disagreement is the finding.
        let computed = fold944(&r, &r, w, h, V1PoolsMode::Full, V1FreeExtras::Off, true);
        let nonzero = computed[..372].iter().filter(|&&v| v != 0.0).count();
        assert!(
            nonzero > 0,
            "{w}x{h}: the computed v1 identity block is all-zero, so the \
             fabricated short-circuit payload is now redundant — delete the \
             fabrication rather than keeping two answers to one question"
        );
        let worst = computed[..372].iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        assert!(
            worst <= 2e-3,
            "{w}x{h}: computed v1 identity residue grew to {worst:e} on f0..372"
        );
    }
}

// ---------------------------------------------------------------------------
// (b) DETERMINISM
// ---------------------------------------------------------------------------

/// Repeated computation in ONE process must be bit-identical. Cheap, and it
/// catches the class of defect a thread sweep cannot: uninitialised scratch
/// that happens to hold the previous call's bytes.
#[test]
fn repeat_runs_in_one_process_are_bit_identical() {
    for &(w, h) in GEOMS {
        let r = value_noise(w, h, 0xC0FFEE);
        let d = small_blur(&r, w, h);
        let mut base: Option<Vec<u64>> = None;
        for rep in 0..4 {
            let bits: Vec<u64> = fold944(
                &r,
                &d,
                w,
                h,
                V1PoolsMode::Full,
                V1FreeExtras::RawMomentsPlusBoundedErr,
                false,
            )
            .iter()
            .map(|v| v.to_bits())
            .collect();
            match &base {
                None => base = Some(bits),
                Some(b) => {
                    if let Some(i) = (0..b.len()).find(|&i| b[i] != bits[i]) {
                        panic!("{w}x{h}: f{i} moved between run 0 and run {rep}");
                    }
                }
            }
        }
    }
}

/// **The 28-thread rung.** `fold_engine_parity` sweeps rayon pools 1/2/3/8/16;
/// this box has 32 hardware threads and the fleet nodes have more, so the
/// band-parallel fan-out has a reachable configuration none of those cover.
/// Band merging is bit-exact only because it is SEQUENTIAL IN BAND ORDER
/// (`((0+b0)+b1)+…`) — a scheduler change that reorders the merge would land
/// exactly here.
#[cfg(feature = "threads")]
#[test]
fn folded944_is_bit_identical_at_28_threads() {
    let cells = [(200usize, 150usize), (127, 93), (576, 96), (1153, 72)];
    for &(w, h) in &cells {
        let r = value_noise(w, h, 0xC0FFEE);
        let d = small_blur(&r, w, h);
        let mut base: Option<Vec<u64>> = None;
        for threads in [1usize, 28] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("build rayon pool");
            let bits: Vec<u64> = pool.install(|| {
                let toggles = V2NewFeatureToggles {
                    v1_pools: V1PoolsMode::Full,
                    append_block: true,
                    append2_block: true,
                    free_extras: V1FreeExtras::RawMomentsPlusBoundedErr,
                    ..Default::default()
                };
                let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(true);
                let mut scratch = V2Scratch::new();
                z.compute_folded720_features_streaming(
                    &RgbSlice::new(&r, w, h),
                    &RgbSlice::new(&d, w, h),
                    toggles,
                    &mut scratch,
                )
                .expect("944 fold walk")
                .features()
                .iter()
                .map(|v| v.to_bits())
                .collect()
            });
            match &base {
                None => base = Some(bits),
                Some(b) => {
                    if let Some(i) = (0..b.len()).find(|&i| b[i] != bits[i]) {
                        panic!(
                            "{w}x{h}: f{i} moved between 1 and {threads} threads ({:e} -> {:e})",
                            f64::from_bits(b[i]),
                            f64::from_bits(bits[i])
                        );
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// (c) ENGINE PARITY, through the entry points the EXTRACTORS use
// ---------------------------------------------------------------------------

/// **`compute_zensim_with_config` (the v1-372 extractor entry) and the fold's
/// own v1 block are BIT-EXACT at every geometry.**
///
/// `feature_v2::tests::v1_372_bit_exact_to_fold_at_every_width` proves this
/// in-crate against the internal walk. This states it at the INTEGRATION
/// boundary — the public free function both v1-372 extractors actually call —
/// because that is the path a training table is built through, and the two can
/// be wired apart (they were: `f9fac41e` fixed a reflect-pad this entry did
/// not do and the internal one did).
///
/// It is also the standing correction to `docs/FEATURE_SET_IDS.md` §1 row 9
/// ("the v1-372 `f0..155` is NOT the 944 fold's `f0..155`, 156 of 156 slots
/// differ, max abs 1.0214"). That measurement compared two STORED instruments
/// built in different extractor eras. In one process at one commit, on the
/// same pixels, they agree to the bit.
#[test]
fn extractor_entry_is_bit_exact_to_the_fold_v1_block() {
    for &(w, h) in GEOMS {
        let r = value_noise(w, h, 0xC0FFEE);
        let d = small_blur(&r, w, h);
        let a = compute_zensim_with_config(&r, &d, w, h, v1_config())
            .expect("v1-372")
            .features()
            .to_vec();
        assert_eq!(a.len(), 372, "{w}x{h}: v1-372 width moved");
        let b = fold944(&r, &d, w, h, V1PoolsMode::Full, V1FreeExtras::Off, true);
        for i in 0..372 {
            assert_eq!(
                a[i].to_bits(),
                b[i].to_bits(),
                "{w}x{h}: f{i} differs between the v1-372 extractor entry \
                 ({:e}) and the fold's v1 block ({:e})",
                a[i],
                b[i]
            );
        }
    }
}

/// **Turning the v2-era blocks on must not move v1's own slots.** `v1_only` is
/// documented as PURE COMPUTE-SKIPPING; this states it at the 944 request the
/// product actually makes, over the geometry set, so a future phase-A
/// restructure that leaks into the v1 band replay fails here.
#[test]
fn v2_era_blocks_do_not_move_the_v1_block() {
    for &(w, h) in GEOMS {
        let r = value_noise(w, h, 0xC0FFEE);
        let d = small_blur(&r, w, h);
        let only = fold944(&r, &d, w, h, V1PoolsMode::Full, V1FreeExtras::Off, true);
        let full = fold944(
            &r,
            &d,
            w,
            h,
            V1PoolsMode::Full,
            V1FreeExtras::RawMomentsPlusBoundedErr,
            false,
        );
        for i in 0..372 {
            assert_eq!(
                only[i].to_bits(),
                full[i].to_bits(),
                "{w}x{h}: f{i} moved when the v2-era blocks were enabled \
                 ({:e} -> {:e}) — v1_only is supposed to be pure compute-skipping",
                only[i],
                full[i]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// (h) DEGENERATE INPUTS
// ---------------------------------------------------------------------------

/// **No NaN and no Inf on pathological inputs, at any width, on either route.**
///
/// Flat planes drive every variance denominator to zero; all-black and
/// all-white sit on both ends of the transfer function; a single lit pixel in
/// a black field maximises every peak/max-pooled slot. These are the inputs
/// that make a ratio-form feature divide by zero, and a NaN in a feature
/// vector poisons a whole model forward pass silently (it compares unequal to
/// itself, so a `to_bits()` gate elsewhere would still pass).
#[test]
fn degenerate_inputs_produce_no_nan_or_inf() {
    type Case = (
        &'static str,
        fn(usize, usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>),
    );
    fn flat_off_by_one(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
        (vec![[128, 128, 128]; w * h], vec![[129, 129, 129]; w * h])
    }
    fn black_vs_white(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
        (vec![[0, 0, 0]; w * h], vec![[255, 255, 255]; w * h])
    }
    fn black_one_lit(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
        let r = vec![[0u8, 0, 0]; w * h];
        let mut d = r.clone();
        d[0] = [255, 255, 255];
        (r, d)
    }
    fn white_one_dark(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
        let r = vec![[255u8, 255, 255]; w * h];
        let mut d = r.clone();
        d[w * h / 2] = [0, 0, 0];
        (r, d)
    }
    fn saturated(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
        let r = (0..w * h)
            .map(|i| [255u8, 0, if i % 2 == 0 { 255 } else { 0 }])
            .collect();
        let d = (0..w * h)
            .map(|i| [0u8, 255, if i % 2 == 0 { 0 } else { 255 }])
            .collect();
        (r, d)
    }
    let cases: &[Case] = &[
        ("flat_gray_off_by_one", flat_off_by_one),
        ("all_black_vs_all_white", black_vs_white),
        ("all_black_vs_one_lit_pixel", black_one_lit),
        ("all_white_vs_one_dark_pixel", white_one_dark),
        ("saturated_opposing_channels", saturated),
    ];
    for &(name, mk) in cases {
        for &(w, h) in &[(64usize, 64usize), (127, 93), (200, 150), (1153, 72)] {
            let (r, d) = mk(w, h);
            for (route, pools, free, v1o) in [
                (
                    "fold944_full",
                    V1PoolsMode::Full,
                    V1FreeExtras::RawMomentsPlusBoundedErr,
                    false,
                ),
                ("fold_v1only", V1PoolsMode::Full, V1FreeExtras::Off, true),
            ] {
                let f = fold944(&r, &d, w, h, pools, free, v1o);
                if let Some(i) = f.iter().position(|v| !v.is_finite()) {
                    panic!("{name} {w}x{h} {route}: f{i} = {} is not finite", f[i]);
                }
            }
            // and through the product entry, which also has to produce a score
            let z = Zensim::new(ZensimProfile::codec_target());
            let res = z
                .compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
                .expect("compute");
            if let Some(i) = res.features().iter().position(|v| !v.is_finite()) {
                panic!(
                    "{name} {w}x{h} product: f{i} = {} is not finite",
                    res.features()[i]
                );
            }
            assert!(
                res.score().is_finite(),
                "{name} {w}x{h}: score is not finite ({})",
                res.score()
            );
        }
    }
}

/// Sub-64 inputs must reflect-pad to a full 4-scale pyramid at BOTH product
/// entries rather than emitting a short vector (the `f9fac41e` defect,
/// DATASET_HISTORY §3.26). `v1_feature_width_pure_function.rs` pins the width
/// for `compute_zensim_with_config`; this adds `Zensim::compute` and the
/// degenerate content that made the pyramid walk stop early.
#[test]
fn sub64_inputs_still_emit_a_full_width_vector_on_both_entries() {
    for &(w, h) in &[(43usize, 64usize), (64, 43), (48, 64), (64, 48), (36, 64)] {
        let r = value_noise(w, h, 7);
        let d = small_blur(&r, w, h);
        let a = compute_zensim_with_config(&r, &d, w, h, v1_config())
            .expect("v1-372")
            .features()
            .len();
        assert_eq!(a, 372, "{w}x{h}: extractor entry emitted {a} features");
        let z = Zensim::new(ZensimProfile::codec_target());
        let b = z
            .compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
            .expect("compute")
            .features()
            .len();
        assert_eq!(b, 372, "{w}x{h}: Zensim::compute emitted {b} features");
    }
}

// ---------------------------------------------------------------------------
// (i) THE RAW-MOMENT ROUTE — the emission gate that was written as `==`
// ---------------------------------------------------------------------------

/// **`RawMomentsPlusBoundedErr` is a strict superset of `RawMoments`.**
///
/// The emission gate was once `toggles.free_extras == V1FreeExtras::RawMoments`,
/// so asking for the superset silently zeroed all 40 raw-moment slots and
/// nothing failed. That is fixed; this states the property at the integration
/// boundary as well as in-crate, because the failure mode is a training table
/// full of structural zeros rather than an error.
#[test]
fn class_c_request_still_carries_the_raw_moments() {
    for &(w, h) in &[(200usize, 150usize), (127, 93)] {
        let r = value_noise(w, h, 0xC0FFEE);
        let d = small_blur(&r, w, h);
        // The free set lives at f720+, so the append blocks must be DECLARED
        // for those slots to exist at all — see
        // `free_extras_are_silently_inert_without_the_append_block_declaration`
        // below, which pins that trap.
        let moments = free_set_walk(&r, &d, w, h, V1FreeExtras::RawMoments);
        let superset = free_set_walk(&r, &d, w, h, V1FreeExtras::RawMomentsPlusBoundedErr);
        assert_eq!(moments.len(), 944, "{w}x{h}: free-set width moved");
        let live: Vec<usize> = (720..944).filter(|&i| moments[i] != 0.0).collect();
        assert!(
            !live.is_empty(),
            "{w}x{h}: the RawMoments route emitted nothing above f720 — the gate is vacuous"
        );
        for i in live {
            assert_eq!(
                moments[i].to_bits(),
                superset[i].to_bits(),
                "{w}x{h}: f{i} moved between RawMoments ({:e}) and \
                 RawMomentsPlusBoundedErr ({:e}) — the superset must carry the \
                 subset unchanged",
                moments[i],
                superset[i]
            );
        }
        // and the superset must add slots, not merely reproduce the subset
        let n_sub = (0..944).filter(|&i| moments[i] != 0.0).count();
        let n_sup = (0..944).filter(|&i| superset[i] != 0.0).count();
        assert!(
            n_sup > n_sub,
            "{w}x{h}: RawMomentsPlusBoundedErr populated {n_sup} slots, no more \
             than RawMoments' {n_sub} — the class-C tranche is not being emitted"
        );
    }
}

/// **`V1FreeExtras` is SILENTLY INERT unless the append block is also
/// declared** — measured, and pinned here so it is a documented contract
/// rather than a landmine.
///
/// `append_block` does double duty: it declares the LAYOUT (720 → 924, and
/// with `append2_block` → 944) *and* it enables the append COMPUTE. The free
/// set's raw-moment slots all live at `f720+`. So a caller that asks for
/// `V1FreeExtras::RawMoments` on a `v1_only` walk **without** setting
/// `append_block` gets a 720-wide vector in which those slots do not exist —
/// no error, no warning, and a feature-count identical to `V1FreeExtras::Off`.
/// That is precisely the failure shape of the `==`-vs-`!=` emission-gate
/// defect this module's sibling test covers, reached by a different route, and
/// it is how this very test file was written wrong on its first pass.
///
/// The class-C tranche is only PARTLY affected: its twelve v2-348 `MSE` cells
/// live at `f372..719`, inside the 720 layout, so they survive; its twelve
/// `LUM_*_ERR` append cells do not.
#[test]
fn free_extras_are_silently_inert_without_the_append_block_declaration() {
    let (w, h) = (200usize, 150usize);
    let r = value_noise(w, h, 0xC0FFEE);
    let d = small_blur(&r, w, h);
    let walk = |append: bool, append2: bool, free: V1FreeExtras| -> Vec<f64> {
        let toggles = V2NewFeatureToggles {
            v1_pools: V1PoolsMode::Peaks,
            append_block: append,
            append2_block: append2,
            v1_only: true,
            free_extras: free,
            ..Default::default()
        };
        let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
        let mut scratch = V2Scratch::new();
        z.compute_folded720_features_streaming(
            &RgbSlice::new(&r, w, h),
            &RgbSlice::new(&d, w, h),
            toggles,
            &mut scratch,
        )
        .expect("walk")
        .features()
        .to_vec()
    };
    let off_720 = walk(false, false, V1FreeExtras::Off);
    let moments_720 = walk(false, false, V1FreeExtras::RawMoments);
    assert_eq!(off_720.len(), 720, "layout without append_block moved");
    assert_eq!(moments_720.len(), 720, "layout without append_block moved");
    let n_off = off_720.iter().filter(|&&v| v != 0.0).count();
    let n_mom = moments_720.iter().filter(|&&v| v != 0.0).count();
    assert_eq!(
        n_off, n_mom,
        "requesting RawMoments without append_block now populates {n_mom} slots \
         against Off's {n_off}. If the raw moments became reachable at 720 this \
         is an improvement — update this gate and the audit doc. If some OTHER \
         block moved, that is a defect."
    );

    // With the append blocks declared, the SAME request is not inert.
    let off_944 = walk(true, true, V1FreeExtras::Off);
    let moments_944 = walk(true, true, V1FreeExtras::RawMoments);
    assert_eq!(moments_944.len(), 944);
    let above_off = off_944[720..].iter().filter(|&&v| v != 0.0).count();
    let above_mom = moments_944[720..].iter().filter(|&&v| v != 0.0).count();
    assert_eq!(
        above_off, 0,
        "a v1_only walk with free extras Off must leave f720+ at structural zeros"
    );
    assert!(
        above_mom > 0,
        "RawMoments populated nothing above f720 even with append declared"
    );
}
