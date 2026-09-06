// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! **Phase 4's round-trip gate** (`docs/PLAN_FEATURE_SYSTEM_2026-09-05.md`
//! phase 4, G4.1 + G4.2): table → bake → serve → the same vector, per
//! registered layout.
//!
//! ## What this actually proves, and why it needed a real bake
//!
//! A `Layout` unit test can show that a dense vector holds the same values as
//! its `w944` twin. It cannot show that the RUNTIME serves a dense bake the
//! right numbers, because that path goes through
//! `metric::forward_one_bake_with_codec` → `feature_layout::declared_layout`
//! → the gather → `prep_bake_input_f32`, and the interesting failure is
//! silent: before the layout existed, a 265-input bake handed a 944-wide
//! caller vector took the `n_inputs < features.len()` PREFIX branch and was
//! served `f0..265` — `basic` plus 37 slots of `masked` — plausible numbers,
//! wrong features, no error.
//!
//! So both arms here are REAL ZNPR v3 bakes built through the mandated JSON
//! pipeline (`zenpredict_bake::bake_from_json_str`, CLAUDE.md "JSON pipeline
//! mandate"), carrying weights that make the forward pass a **checksum of
//! which features arrived**: each live input gets a distinct weight, so any
//! permutation or truncation changes the output. The dense arm declares its
//! layout the way a real dense-table bake would — a `zentrain.feature_set_id`
//! metadata entry whose slot count equals its caller width.

#![cfg(all(
    feature = "training",
    feature = "feature-regime-v2",
    feature = "custom-profiles"
))]

use zensim::feature_set_id::{ComputeParts, ComputeToken, FeatureSetId, SlotSet};
use zensim::research;
use zensim::{RgbSlice, Zensim, ZensimProfile};

mod common;

/// The free-set arm's slot set: `basic + peaks + moments`, 265 ids at the 944
/// layout. The campaign's own producer set, so the round-trip is over a shape
/// that actually exists rather than an invented one.
fn free_set_slots() -> SlotSet {
    SlotSet::from_ranges([(0, 228)])
        .union(&research::family_slots(ComputeToken::Moments))
        .clipped_to(944)
}

fn compute_parts() -> ComputeParts {
    ComputeParts::EMPTY
        .with(ComputeToken::Basic)
        .with(ComputeToken::Peaks)
        .with(ComputeToken::Moments)
}

/// A 1-layer identity ZNPR v3 bake of `n_inputs` width whose weight at input
/// `i` is `weights[i]`, optionally carrying a `zentrain.feature_set_id`.
///
/// Built through the mandated JSON pipeline — never hand-rolled wire bytes.
fn bake(n_inputs: usize, weights: &[f32], declared_set: Option<&str>) -> Vec<u8> {
    assert_eq!(weights.len(), n_inputs);
    let arr = |v: &[f32]| -> String {
        let mut s = String::from("[");
        for (k, x) in v.iter().enumerate() {
            if k > 0 {
                s.push(',');
            }
            s.push_str(&x.to_string());
        }
        s.push(']');
        s
    };
    let metadata = match declared_set {
        Some(id) => format!(
            r#""metadata": [{{"key": "zentrain.feature_set_id", "type": "utf8", "text": "{id}"}}],"#
        ),
        None => String::new(),
    };
    let json = format!(
        r#"{{
            "schema_hash": 1,
            "scaler_mean": {mean},
            "scaler_scale": {scale},
            {metadata}
            "layers": [
                {{
                    "in_dim": {n},
                    "out_dim": 1,
                    "activation": "identity",
                    "dtype": "f32",
                    "weights": {w},
                    "biases": [0.0]
                }}
            ]
        }}"#,
        n = n_inputs,
        mean = arr(&vec![0.0f32; n_inputs]),
        scale = arr(&vec![1.0f32; n_inputs]),
        w = arr(weights),
    );
    zenpredict_bake::bake_from_json_str(&json).expect("synthetic bake must build")
}

/// Distinct per-id weights, so the forward pass is a CHECKSUM of which
/// features arrived at which position. A permutation or a truncation changes
/// the answer; equal weights would hide both.
fn weight_for(id: usize) -> f32 {
    // Spread over a few decades so no two ids' contributions cancel, and keep
    // the magnitudes small enough that the f32 sum stays well-conditioned.
    (1.0 + (id % 97) as f32) * 1e-3 + (id / 97) as f32 * 7e-2
}

/// **G4.1/G4.2 end to end** — a DENSE bake and its `w944` twin, served by the
/// runtime over the same pixels, produce the same score; and the dense arm is
/// served the RIGHT features, which the pre-layout prefix branch would not
/// have done.
#[test]
fn a_dense_bake_and_its_w944_twin_score_identically() {
    let slots = free_set_slots();
    assert_eq!(slots.len(), 265);

    // The SPARSE arm: 944 inputs, live only on the 265 ids, zero elsewhere.
    let mut w944 = vec![0.0f32; 944];
    for id in slots.iter_slots() {
        w944[id] = weight_for(id);
    }
    let sparse_bytes = bake(944, &w944, None);

    // The DENSE arm: 265 inputs, one per id in ascending order, carrying the
    // SAME weight for the SAME id — so equal scores mean the same features
    // reached the same weights.
    let dense_w: Vec<f32> = slots.iter_slots().map(weight_for).collect();
    let dense_id = FeatureSetId::from_slots(compute_parts(), 265, "era2r4", &slots)
        .expect("a dense feature-set id");
    let dense_bytes = bake(265, &dense_w, Some(&dense_id.to_string()));

    // The runtime must resolve the two to different layouts — the whole point.
    let sparse_model = zensim::research::Request::for_bake_bytes(&sparse_bytes);
    assert!(sparse_model.is_ok(), "the w944 bake must be readable");

    for (w, h) in [(64usize, 64usize), (128, 96), (200, 150)] {
        let r = common::generators::gen_value_noise(w, h, 0xC0FFEE);
        let d = common::generators::distort_block_artifacts(&r, w, h);
        let (rs, ds) = (RgbSlice::new(&r, w, h), RgbSlice::new(&d, w, h));

        let sparse = score_with(&sparse_bytes, &rs, &ds);
        let dense = score_with(&dense_bytes, &rs, &ds);

        // Not a tolerance: the two forwards sum the SAME products, in the
        // same ascending-id order, so they must agree to the bit.
        assert_eq!(
            sparse.to_bits(),
            dense.to_bits(),
            "the dense bake scored differently at {w}x{h}: w944 {sparse:.17e} \
             vs dense265 {dense:.17e} — the gather placed the wrong features"
        );

        // NEGATIVE CONTROL: a 265-input bake that does NOT declare its layout
        // is served the identity PREFIX (`f0..265`), which is a different set
        // of features and must therefore score differently. Without this the
        // test above would pass just as happily if the gather were a no-op
        // and both arms were being truncated.
        let undeclared = bake(265, &dense_w, None);
        let prefix = score_with(&undeclared, &rs, &ds);
        assert_ne!(
            prefix.to_bits(),
            dense.to_bits(),
            "the undeclared 265-input bake scored the SAME as the declared \
             one at {w}x{h} — either the gather is inert or the prefix \
             happens to be the free set, and neither is a passing test"
        );
    }
}

/// Score one pair through a custom profile carrying `bytes` as its bake.
fn score_with(bytes: &[u8], r: &RgbSlice<'_>, d: &RgbSlice<'_>) -> f64 {
    // Leak the bytes: `ProfileParams::mlp` takes a `fn() -> &'static [u8]`,
    // and a test fixture's lifetime is the process.
    let leaked: &'static [u8] = Box::leak(bytes.to_vec().into_boxed_slice());
    let params: &'static zensim::profile::ProfileParams = Box::leak(Box::new(
        zensim::profile::ProfileParams::builder()
            .weights(zensim::WEIGHTS)
            .mlp({
                // A closure cannot coerce to `fn()`, so route through a
                // thread-local the fn pointer reads. One bake in flight at a
                // time, which is what this test does.
                BAKE.with(|b| b.set(leaked));
                fn get() -> &'static [u8] {
                    BAKE.with(|b| b.get())
                }
                get
            })
            .skip_score_mapping(true)
            .build(),
    ));
    let z = Zensim::new(ZensimProfile::Custom {
        name: "dense-round-trip",
        params,
    })
    .with_parallel(false);
    z.compute(r, d).expect("custom profile must score").score()
}

thread_local! {
    static BAKE: std::cell::Cell<&'static [u8]> = const { std::cell::Cell::new(&[]) };
}

/// **A dense bake REFUSES a feature vector that does not reach its ids.**
///
/// `Layout::gather` writes the structural `0.0` for an id the walk did not
/// reach, which is right for a DECLARED GAP and wrong for a caller who handed
/// over a short vector: every id a dense bake declares is an id it reads, so
/// filling one is exactly the "a consumer cannot tell this zero from a
/// measured zero" failure the dense contract exists to end.
///
/// Found by running the batched FD-gradient gate after shipped `B` went dense
/// (2026-09-06): `score_features_with_profile(B, &vec![0.1; 156])` used to
/// SUCCEED — the 156-wide row was gathered into `B`'s 95 ids, every one above
/// `f155` becoming a silent zero — so a caller probing widths took 156 as the
/// bake's width and differentiated a function of mostly-zeros.
///
/// The POSITIVE CONTROL is in the same test: at the walk width the same bake
/// scores finitely. Without it the refusal could be a blanket rejection.
#[test]
fn a_dense_bake_refuses_a_feature_vector_that_does_not_reach_its_ids() {
    let slots = free_set_slots();
    let dense_w: Vec<f32> = slots.iter_slots().map(weight_for).collect();
    let dense_id = FeatureSetId::from_slots(compute_parts(), 265, "era2r4", &slots)
        .expect("a dense feature-set id");
    let bytes = bake(265, &dense_w, Some(&dense_id.to_string()));
    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    let params: &'static zensim::profile::ProfileParams = Box::leak(Box::new(
        zensim::profile::ProfileParams::builder()
            .weights(zensim::WEIGHTS)
            .mlp({
                BAKE.with(|b| b.set(leaked));
                fn get() -> &'static [u8] {
                    BAKE.with(|b| b.get())
                }
                get
            })
            .skip_score_mapping(true)
            .build(),
    ));
    let profile = ZensimProfile::Custom {
        name: "dense-short-row",
        params,
    };

    // The highest declared id is f941, so the walk width is 942.
    let reach = slots.iter_slots().max().expect("non-empty") + 1;
    assert_eq!(reach, 942);

    for short in [156usize, 265, 372, 720, 941] {
        let err = zensim::score_features_with_profile(profile, &vec![0.1f64; short], 576, 576)
            .expect_err("a row shorter than the declared reach must be REFUSED, not zero-filled");
        assert!(
            format!("{err:?}").contains("does not reach"),
            "{short}: refused, but not with the named reason: {err:?}"
        );
        assert!(
            zensim::score_features_fd_gradient_with_profile(
                profile,
                &vec![0.1f64; short],
                576,
                576
            )
            .is_err(),
            "{short}: the batched FD entry must refuse for the same reason"
        );
    }

    // POSITIVE CONTROL: at the reach width, both entries work.
    let ok = zensim::score_features_with_profile(profile, &vec![0.1f64; reach], 576, 576)
        .expect("a row that reaches every declared id must score");
    assert!(ok.is_finite(), "score must be finite, got {ok}");
    let g =
        zensim::score_features_fd_gradient_with_profile(profile, &vec![0.1f64; reach], 576, 576)
            .expect("the batched FD entry must accept the same row");
    assert_eq!(g.len(), reach);
    assert!(
        g.iter().any(|v| *v != 0.0),
        "the gradient must be nonzero somewhere, else the control proves nothing"
    );
}
