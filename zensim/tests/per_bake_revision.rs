// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! **Phase 5 — revision is a PER-BAKE declaration, not a process switch.**
//!
//! ## Why this had to change, measured
//!
//! The revision lane's F5 fix (`c433c7b7`) is not free. `bake_block_profile`
//! shows all three shipped 944 bakes — `c_sdr_mlp944_corrmix`,
//! `c_hdr_l1t1944`, `c_sdr_purity944` — reading the full
//! `GLOBAL_DMEAN`/`CGAIN`/`CLOSS` set, 33 slots each. A GLOBAL flip of
//! `SHIPPED_REVISION` to `Rev2` would therefore move **22 of 33 inputs per
//! bake** and silently re-price every one of them, so `Rev2` could not ship
//! until Profile C was refit. That is the wrong shape for a contract that is
//! supposed to "just work for any feature revision".
//!
//! So a bake DECLARES its revision (`zentrain.formula_revision`; absent means
//! `Rev1`, the registry default), the plan computes the declared revision,
//! and two bakes at different revisions coexist in one process — each getting
//! its own arithmetic, in its own extraction.
//!
//! ## What is in scope, honestly
//!
//! `Rev2` has two halves. Its `paired_global_contrast` half is a FINALISER
//! parameter, so it threads per-walk and is what these tests exercise — and
//! it is also the half the shipped 944 bakes actually read. Its luma-form
//! half (`ssim_form::active_luma_form`) is a `OnceLock` read inside the SIMD
//! kernels; making that per-request is a change to the kernel dispatch, which
//! this lane does not own. A per-bake `Rev2` therefore gets rev2's
//! global-contrast arithmetic and the process's luma form. That is stated
//! rather than papered over, and `mixed_revision_profiles_get_no_plan` pins
//! the one place the limitation is load-bearing.

#![cfg(all(feature = "training", feature = "feature-regime-v2"))]

use zensim::feature_v2::{V1FreeExtras, V1PoolsMode, V2NewFeatureToggles, V2Scratch};
use zensim::{RgbSlice, Zensim, ZensimProfile};

mod common;

fn pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let r = common::generators::gen_value_noise(w, h, 0xC0FFEE);
    let d = common::generators::distort_block_artifacts(&r, w, h);
    (r, d)
}

/// Extract the 944 free-set walk at an explicit revision.
fn extract_at(rev_two: bool, w: usize, h: usize) -> Vec<f64> {
    let (s, d) = pair(w, h);
    let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
    let mut scratch = V2Scratch::new();
    let toggles = V2NewFeatureToggles {
        append2_block: true,
        v1_only: true,
        v1_pools: V1PoolsMode::Peaks,
        free_extras: V1FreeExtras::RawMoments,
        formula_revision: if rev_two {
            zensim::feature_v2::FormulaRevision::Rev2
        } else {
            zensim::feature_v2::FormulaRevision::Rev1
        },
        ..V2NewFeatureToggles::default()
    };
    z.compute_folded720_append_features_streaming(
        &RgbSlice::new(&s, w, h),
        &RgbSlice::new(&d, w, h),
        toggles,
        &mut scratch,
    )
    .expect("walk")
    .into_features()
}

/// **The coexistence property.** Two extractions at two declared revisions,
/// in ONE process, differ on exactly the slots the revision moves and agree
/// everywhere else.
///
/// This is what lets `Rev2` ship without touching Profile C: C keeps
/// declaring `Rev1` and keeps its numbers; a bake refit at `Rev2` declares
/// `Rev2` and gets the fixed arithmetic; neither is affected by the other.
#[test]
fn two_revisions_coexist_in_one_process_and_differ_only_where_the_revision_moves() {
    for (w, h) in [(64usize, 64usize), (128, 96), (200, 150)] {
        let r1 = extract_at(false, w, h);
        let r2 = extract_at(true, w, h);
        assert_eq!(r1.len(), r2.len());

        let moved: Vec<usize> = (0..r1.len())
            .filter(|&i| r1[i].to_bits() != r2[i].to_bits())
            .collect();
        assert!(
            !moved.is_empty(),
            "at {w}x{h} the two revisions produced IDENTICAL vectors — either \
             the per-bake revision is not reaching the finaliser, or the \
             fixture does not exercise the moved slots, and neither is a \
             passing test"
        );
        // Everything that moved must be a GLOBAL_* slot. The rest of the
        // vector is untouched by this revision, and a value that moved
        // outside it would mean the revision leaked.
        use zensim::feature_v2::{FEATURES_PER_CHANNEL_APPEND, idx_append};
        let append_base = 372 + 4 * 3 * zensim::feature_v2::FEATURES_PER_CHANNEL_V2_TOTAL;
        for &i in &moved {
            assert!(
                i >= append_base,
                "f{i} moved between revisions at {w}x{h} but is below the \
                 append block — the revision leaked outside its blast radius"
            );
            let local = (i - append_base) % FEATURES_PER_CHANNEL_APPEND;
            assert!(
                local == idx_append::GLOBAL_DMEAN
                    || local == idx_append::GLOBAL_CGAIN
                    || local == idx_append::GLOBAL_CLOSS,
                "f{i} (append-local {local}) moved between revisions at \
                 {w}x{h} but is not a GLOBAL_* slot — the revision leaked \
                 outside its declared blast radius"
            );
        }
        // And the process default must be Rev1's answer, so nothing that
        // scores today changes.
        let dflt = {
            let (s, d) = pair(w, h);
            let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
            let mut scratch = V2Scratch::new();
            z.compute_folded720_append_features_streaming(
                &RgbSlice::new(&s, w, h),
                &RgbSlice::new(&d, w, h),
                V2NewFeatureToggles {
                    append2_block: true,
                    v1_only: true,
                    v1_pools: V1PoolsMode::Peaks,
                    free_extras: V1FreeExtras::RawMoments,
                    ..V2NewFeatureToggles::default()
                },
                &mut scratch,
            )
            .expect("walk")
            .into_features()
        };
        for (i, (a, b)) in dflt.iter().zip(r1.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "f{i}: the DEFAULT toggles must compute the shipped revision \
                 (Rev1) at {w}x{h}"
            );
        }
    }
}

/// **The blast radius, MEASURED and pinned in both directions.**
///
/// Over the whole 944 vector the revision moves **11 slots**, every one of
/// them a `GLOBAL_CGAIN` or `GLOBAL_CLOSS`, and **`GLOBAL_DMEAN` never moves
/// anywhere**. That is narrower than the name `paired_global_contrast` might
/// suggest and it is correct: the fix is to the paired CONTRAST estimate, not
/// to the mean.
///
/// Asserted over the whole vector rather than at one hand-picked cell,
/// because a `CGAIN` can be legitimately `0.0` in both revisions on a given
/// fixture — an earlier draft pinned (scale 0, Y) and failed for exactly that
/// reason, on a fixture where that cell had no gain to move.
#[test]
fn the_revision_moves_only_the_global_contrast_pair() {
    use zensim::feature_v2::{
        FEATURES_PER_CHANNEL_APPEND, FEATURES_PER_CHANNEL_V2_TOTAL, idx_append,
    };
    let (w, h) = (128usize, 96usize);
    let r1 = extract_at(false, w, h);
    let r2 = extract_at(true, w, h);
    let base = 372 + 4 * 3 * FEATURES_PER_CHANNEL_V2_TOTAL;
    let mut moved_locals = Vec::new();
    for i in 0..r1.len() {
        if r1[i].to_bits() == r2[i].to_bits() {
            continue;
        }
        assert!(i >= base, "f{i} moved but is below the append block");
        let local = (i - base) % FEATURES_PER_CHANNEL_APPEND;
        assert_ne!(
            local,
            idx_append::GLOBAL_DMEAN,
            "f{i} GLOBAL_DMEAN moved — the revision fixes the paired CONTRAST \
             estimate, not the mean"
        );
        assert!(
            local == idx_append::GLOBAL_CGAIN || local == idx_append::GLOBAL_CLOSS,
            "f{i} (append-local {local}) moved but is not a GLOBAL_* contrast \
             slot — the revision leaked outside its blast radius"
        );
        moved_locals.push(local);
    }
    assert_eq!(
        moved_locals.len(),
        11,
        "the measured blast radius on this fixture is 11 slots; a different \
         count means the revision widened or narrowed"
    );
}
