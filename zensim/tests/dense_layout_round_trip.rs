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
    let dense_id = FeatureSetId::from_slots_with_layout(compute_parts(), 265, "era2r4", &slots)
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
    let dense_id = FeatureSetId::from_slots_with_layout(compute_parts(), 265, "era2r4", &slots)
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

/// A model can select the full-resolution Y subset without a public toggle.
/// Compare pixel serving and research extraction to full-feature inference.
#[test]
fn fullres_y_subset_bake_matches_full_inputs() {
    check_v1_subset_bake(false, false);
}

#[test]
fn coarse_pool_subset_bake_matches_full_inputs() {
    check_v1_subset_bake(true, false);
    check_v1_subset_bake(true, true);
}

fn check_v1_subset_bake(weighted: bool, fine: bool) {
    let width = if weighted { 372 } else { 228 };
    let ids: Vec<usize> = (0..width)
        .filter(|&i| i < 228 || fine || matches!(i,264..=299|336..=371))
        .filter(|&i| !matches!(i,228..=233|240..=245|300..=305|312..=317))
        .filter(|&i| !matches!(i, 0..=12 | 26..=38 | 156..=161 | 168..=173))
        .collect();
    let n = ids.len();
    let recipe = serde_json::json!({
        "schema_hash": 1, "scaler_mean": vec![0.0; n], "scaler_scale": vec![1.0; n],
        "metadata": [
            {"key":"zentrain.feature_ids","type":"utf8","text":ids.iter().map(usize::to_string).collect::<Vec<_>>().join("\n")},
            {"key":"zentrain.formula_revision","type":"utf8","text":std::env::var("ZENSIM_FORMULA_REV").unwrap_or_else(|_| "1".into())}
        ],
        "layers": [{"in_dim": n, "out_dim": 1, "activation":"identity", "dtype":"f32",
                    "weights": (0..n).map(|i| (i+1) as f32 / 1000.0).collect::<Vec<_>>(), "biases":[0.0]}]
    });
    let bytes = zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap();
    let model = zenpredict::Model::from_bytes(&bytes).unwrap();
    let mut scorer = zensim::BakeScorer::new(&model).unwrap();
    let (w, h) = (97, 131);
    let src = vec![[127u8; 3]; w * h];
    let mut dst = src.clone();
    for (i, p) in dst.iter_mut().enumerate() {
        if (i % w) % 8 == 0 || (i / w) % 8 == 0 {
            *p = [165, 90, 180];
        }
    }
    let (rs, ds) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));
    let full = research::extract(
        &research::Request::for_slots(SlotSet::from_slots(0..width), width).with_parallel(true),
        &rs,
        &ds,
    )
    .unwrap();
    let sub = research::extract(
        &research::Request::for_slots(SlotSet::from_slots(ids.iter().copied()), width)
            .with_parallel(true),
        &rs,
        &ds,
    )
    .unwrap();
    let served = scorer.compute(&rs, &ds, None).unwrap();
    for &id in &ids {
        assert_eq!(
            served.features()[id].to_bits(),
            full.values()[id].to_bits(),
            "served f{id}"
        );
        assert_eq!(
            sub.values()[id].to_bits(),
            full.values()[id].to_bits(),
            "research f{id}"
        );
    }
    let expected = scorer
        .score_features(full.values(), w as u32, h as u32, None)
        .unwrap();
    assert_eq!(served.score().to_bits(), expected.to_bits());
    let pre = scorer.precompute_reference(&rs).unwrap();
    let mapped = scorer
        .compute_with_ref_and_attribution(
            &rs,
            &pre,
            &ds,
            None,
            &mut zensim::Fused944Session::new(),
            8,
        )
        .unwrap();
    assert_eq!(mapped.result().score().to_bits(), served.score().to_bits());
    for &id in &ids {
        assert_eq!(
            mapped.result().features()[id].to_bits(),
            full.values()[id].to_bits(),
            "mapped f{id}"
        );
    }
    // Max features have no additive density; retain the owner's explicit
    // coverage report instead of claiming the whole 190-input model maps.
    assert!(
        mapped
            .unsupported_feature_ids()
            .iter()
            .all(|id| ids.contains(id))
    );
    eprintln!(
        "subset density omissions: {:?}; refinement omissions: {:?}",
        mapped.unsupported_feature_ids(),
        mapped.unsupported_refinement_feature_ids()
    );
    assert_eq!(sub.emitted(), &SlotSet::from_slots(ids.iter().copied()));
    assert!(
        sub.feature_set_id().is_none(),
        "family-only shorthand cannot encode this subset"
    );
    assert_eq!(scorer.compute(&rs, &rs, None).unwrap().score(), 100.0);
}

#[test]
fn sampling_contracts_serve_and_spatialize_through_public_api() {
    for mode in ["y", "xyb"] {
        let ids: Vec<usize> = (0..228)
            .filter(|&i| mode != "y" || !matches!(i,0..=12|26..=38|156..=161|168..=173))
            .collect();
        for filter in ["triangle", "mitchell", "robidouxsharp"] {
            for ratio in ["3/2", "2", "3"] {
                let tag = format!("v1:{mode}:{filter}:{ratio}");
                let n = ids.len();
                let recipe = serde_json::json!({"schema_hash":1,"scaler_mean":vec![0.;n],"scaler_scale":vec![1.;n],
                    "metadata":[{"key":"zentrain.feature_ids","type":"utf8","text":ids.iter().map(usize::to_string).collect::<Vec<_>>().join("\n")},
                    {"key":"zentrain.formula_revision","type":"utf8","text":std::env::var("ZENSIM_FORMULA_REV").unwrap_or_else(|_|"1".into())},
                    {"key":"zentrain.sampling","type":"utf8","text":tag}],
                    "layers":[{"in_dim":n,"out_dim":1,"activation":"identity","dtype":"f32","weights":vec![-0.1;n],"biases":[100.]}]});
                let bytes = zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap();
                let model = zenpredict::Model::from_bytes(&bytes).unwrap();
                let mut scorer = zensim::BakeScorer::new(&model).unwrap();
                for (w, h) in [(17, 9), (97, 131), (257, 259)] {
                    let src: Vec<_> = (0..w * h)
                        .map(|i| {
                            [
                                (i % 211) as u8,
                                ((i * 7) % 239) as u8,
                                ((i * 13) % 251) as u8,
                            ]
                        })
                        .collect();
                    let mut dst = src.clone();
                    for y in h / 3..(h / 3 + 8).min(h) {
                        for x in w / 3..(w / 3 + 8).min(w) {
                            dst[y * w + x] = [255, 0, 128];
                        }
                    }
                    let rs = RgbSlice::new(&src, w, h);
                    let ds = RgbSlice::new(&dst, w, h);
                    let direct = scorer.compute(&rs, &ds, None).unwrap();
                    let cached = scorer
                        .score_features(direct.features(), w as u32, h as u32, None)
                        .unwrap();
                    assert_eq!(direct.score().to_bits(), cached.to_bits(), "{tag}");
                    let pre = scorer.precompute_reference(&rs).unwrap();
                    assert!(
                        zensim::Zensim::new(zensim::ZensimProfile::B)
                            .compute_with_ref(&pre, &ds)
                            .is_err()
                    );
                    assert!(
                        scorer
                            .compute_hdr(&rs, &ds, zensim::feature_v2::HdrEncoding::Linear, None)
                            .is_err()
                    );

                    let legacy = zensim::Zensim::new(zensim::ZensimProfile::B);
                    assert!(
                        legacy
                            .compute_with_ref_score_and_attribution(&pre, &ds, &[0.0; 156])
                            .is_err()
                    );
                    assert!(
                        legacy
                            .compute_with_ref_score_and_attribution_binned(
                                &pre,
                                &ds,
                                &[0.0; 156],
                                8
                            )
                            .is_err()
                    );
                    let mapped = scorer
                        .compute_with_ref_and_attribution(
                            &rs,
                            &pre,
                            &ds,
                            None,
                            &mut zensim::Fused944Session::new(),
                            8,
                        )
                        .unwrap();
                    assert_eq!(
                        mapped.result().features(),
                        direct.features(),
                        "{tag} {w}x{h}"
                    );
                    assert_eq!(mapped.result().score(), direct.score(), "{tag}");
                    assert_eq!(
                        mapped.result().raw_distance(),
                        direct.raw_distance(),
                        "{tag}"
                    );
                    assert!(mapped.attribution().density().iter().all(|v| v.is_finite()));
                    assert!(mapped.refinement_gain(0, 0, w, h).is_finite());
                    assert!(
                        mapped.unsupported_refinement_feature_ids().is_empty(),
                        "{tag}: {:?}",
                        mapped.unsupported_refinement_feature_ids()
                    );
                    assert_eq!(scorer.compute(&rs, &rs, None).unwrap().score(), 100.);
                    let wrong = zensim::Zensim::new(zensim::ZensimProfile::B)
                        .precompute_reference(&rs)
                        .unwrap();
                    assert!(
                        scorer
                            .compute_with_ref_and_attribution(
                                &rs,
                                &wrong,
                                &ds,
                                None,
                                &mut zensim::Fused944Session::new(),
                                8
                            )
                            .is_err()
                    );
                }

                let bytes_static: &'static [u8] = Box::leak(bytes.clone().into_boxed_slice());
                BAKE.with(|b| b.set(bytes_static));
                fn sampling_fixture() -> &'static [u8] {
                    BAKE.with(|b| b.get())
                }
                let params = Box::leak(Box::new(
                    zensim::profile::ProfileParams::builder()
                        .weights(zensim::WEIGHTS)
                        .mlp(sampling_fixture)
                        .skip_score_mapping(true)
                        .build(),
                ));
                let legacy = Zensim::new(ZensimProfile::Custom {
                    name: "sampling-refusal",
                    params,
                });
                let r = vec![[100, 120, 140]; 64 * 64];
                let d = vec![[110, 125, 130]; 64 * 64];
                assert!(
                    legacy
                        .compute(&RgbSlice::new(&r, 64, 64), &RgbSlice::new(&d, 64, 64))
                        .is_err()
                );
                let compatible = [
                    zenpredict::Model::from_bytes(&bytes).unwrap(),
                    zenpredict::Model::from_bytes(&bytes).unwrap(),
                ];
                assert!(zensim::BakeScorer::ensemble(&compatible, None).is_ok());
                let other_tag = if ratio == "2" {
                    format!("v1:{mode}:{filter}:3")
                } else {
                    format!("v1:{mode}:{filter}:2")
                };
                let other =
                    zenpredict_bake::append_metadata_utf8(&bytes, "zentrain.sampling", &other_tag)
                        .unwrap();
                let incompatible = [
                    zenpredict::Model::from_bytes(&bytes).unwrap(),
                    zenpredict::Model::from_bytes(&other).unwrap(),
                ];
                assert!(zensim::BakeScorer::ensemble(&incompatible, None).is_err());
                let bad =
                    zenpredict_bake::append_metadata_utf8(&bytes, "zentrain.sampling", "unknown")
                        .unwrap();
                let bad = zenpredict::Model::from_bytes(&bad).unwrap();
                assert!(zensim::BakeScorer::new(&bad).is_err());
            }
        }
    }
}
