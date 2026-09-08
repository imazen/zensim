//! Candidate serving/evaluation contract, including independent expected-value
//! fixtures and comparison with the pre-migration evaluator.
use serde_json::json;
use zenpredict::{Model, Predictor};
use zensim::{BakeScorer, RgbSlice, Zensim, ZensimProfile};
use zensim_validate::{bake_runtime as old, output_calibration_spline as spline};

fn linear(metadata: serde_json::Value) -> Model {
    linear_bias(metadata, -5.)
}

fn linear_bias(metadata: serde_json::Value, bias: f32) -> Model {
    let recipe = json!({
        "schema_hash": 1,
        "scaler_mean": [0.,0.], "scaler_scale": [1.,1.],
        "metadata": metadata,
        "layers": [{"in_dim":2, "out_dim":1,"activation":"identity",
                    "dtype":"f32", "weights":[2.,3.], "biases":[bias]}]
    });
    Model::from_bytes(&zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap()).unwrap()
}

#[test]
fn declared_ids_are_gathered_and_short_or_malformed_rows_refuse() {
    let m = linear(json!([{"key":"zentrain.feature_ids","type":"utf8","text":"1 3"}]));
    let mut s = BakeScorer::new(&m).unwrap();
    assert_eq!(
        s.score_features(&[99., 2., 99., 7.], 64, 64, None).unwrap(),
        20.
    );
    assert_eq!(
        s.score_features(&[99., -2., 99., -7.], 64, 64, None)
            .unwrap(),
        -30.
    );
    assert!(s.score_features(&[99., 2.], 64, 64, None).is_err());
    for ids in ["1 1", "3 1", "1", "bad", "1 65535"] {
        let m = linear(json!([{"key":"zentrain.feature_ids","type":"utf8","text":ids}]));
        assert!(BakeScorer::new(&m).is_err(), "accepted {ids}");
    }
}

fn assert_gradient(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        // Predictor arithmetic is f32; the finite probe is performed in f64.
        assert!((got - want).abs() < 0.002, "gradient[{i}]: {got} vs {want}");
    }
}

#[test]
fn candidate_sensitivities_preserve_feature_ids_and_negative_scores() {
    let model = linear(json!([{"key":"zentrain.feature_ids","type":"utf8","text":"1 3"}]));
    let mut scorer = BakeScorer::new(&model).unwrap();
    for row in [[99., 2., 99., 7.], [99., -2., 99., -7.]] {
        let gradient = scorer
            .score_features_fd_gradient(&row, 64, 64, None)
            .unwrap();
        assert_gradient(&gradient, &[0., 2., 0., 3.]);
    }
    assert!(
        scorer
            .score_features_fd_gradient(&[1., 2.], 64, 64, None)
            .is_err()
    );
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX] {
        // Also reject nonfinite values in an unconsumed slot, rather than
        // manufacturing a sensitivity from undefined inputs/probes.
        assert!(
            scorer
                .score_features_fd_gradient(&[bad, 2., 99., 7.], 64, 64, None)
                .is_err()
        );
    }
    assert_gradient(
        &scorer
            .score_features_fd_gradient(&[99., 2., 99., 7.], 64, 64, None)
            .unwrap(),
        &[0., 2., 0., 3.],
    );
}

#[test]
fn candidate_sensitivities_include_ensemble_and_corruption_discontinuities() {
    let models = [linear(json!([])), linear_bias(json!([]), 5.)];
    let mut weighted = BakeScorer::ensemble(&models, Some(&[0.25, 0.75])).unwrap();
    assert_gradient(
        &weighted
            .score_features_fd_gradient(&[5., 5.], 64, 64, None)
            .unwrap(),
        &[2., 3.],
    );
    let mut gated = BakeScorer::ensemble(&models, None)
        .unwrap()
        .with_linear_corruption_head(&models[0], 20.)
        .unwrap();
    assert_gradient(
        &gated
            .score_features_fd_gradient(&[4., 4.], 64, 64, None)
            .unwrap(),
        &[0., 0.],
    );
    assert_gradient(
        &gated
            .score_features_fd_gradient(&[-2., -7.], 64, 64, None)
            .unwrap(),
        &[2., 3.],
    );
    // Exactly at the deadband, central probes straddle a score jump. Return
    // the finite secant, visibly unlike the ungated gradient; do not silently
    // ignore the companion or label this an analytic derivative.
    let crossing = gated
        .score_features_fd_gradient(&[5., 5.], 64, 64, None)
        .unwrap();
    assert!((crossing[0] - 2501.).abs() < 0.002);
    assert!((crossing[1] - 2501.5).abs() < 0.002);
}

#[test]
fn malformed_present_metadata_is_not_treated_as_absent() {
    assert!(BakeScorer::new(&linear(json!([]))).is_ok());
    for key in [
        "zentrain.per_sample_alpha_head",
        "zentrain.hybrid_head",
        "zentrain.minmax_monotone_head",
        "zentrain.tanh_output_head",
        "zentrain.output_calibration_spline",
        "zentrain.per_codec_calibration",
    ] {
        let m = linear(json!([{"key":key,"type":"bytes","hex":"00"}]));
        assert!(BakeScorer::new(&m).is_err(), "accepted malformed {key}");
    }
}

#[test]
fn ensemble_and_corruption_composition_are_returned_by_the_surface() {
    let models = [linear(json!([])), linear_bias(json!([]), 5.)];
    let mut mean = BakeScorer::ensemble(&models, None).unwrap();
    assert_eq!(mean.score_features(&[5., 5.], 64, 64, None).unwrap(), 25.);
    let mut weighted = BakeScorer::ensemble(&models, Some(&[0.25, 0.75])).unwrap();
    assert_eq!(
        weighted.score_features(&[5., 5.], 64, 64, None).unwrap(),
        27.5
    );
    assert!(BakeScorer::ensemble(&models, Some(&[0.25, 0.25])).is_err());
    assert!(BakeScorer::ensemble(&[], None).is_err());
    let mut inactive = BakeScorer::ensemble(&models, None)
        .unwrap()
        .with_linear_corruption_head(&models[0], 20.)
        .unwrap();
    assert_eq!(
        inactive.score_features(&[5., 5.], 64, 64, None).unwrap(),
        25.
    );
    let mut active = BakeScorer::ensemble(&models, None)
        .unwrap()
        .with_linear_corruption_head(&models[0], 20.0001)
        .unwrap();
    assert_eq!(active.score_features(&[5., 5.], 64, 64, None).unwrap(), 0.);
    assert_eq!(
        active.score_features(&[-1., -1.], 64, 64, None).unwrap(),
        -5.
    );
}

#[test]
fn spline_endpoints_negative_tail_and_upper_cap_use_the_runtime_contract() {
    let mut payload = 2u32.to_le_bytes().to_vec();
    for v in [0.0f32, 0., 10., 100.] {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    let hex: String = payload.iter().map(|b| format!("{b:02x}")).collect();
    let model =
        linear(json!([{"key":"zentrain.output_calibration_spline","type":"bytes","hex":hex}]));
    let mut s = BakeScorer::new(&model).unwrap();
    // Network is 2*x + 3*y - 5; spline maps [0,10] to [0,100].
    for (x, expected) in [(0., -50.), (2.5, 0.), (5., 50.), (7.5, 100.), (10., 100.)] {
        assert_eq!(s.score_features(&[x, 0.], 64, 64, None).unwrap(), expected);
    }
    for (row, expected) in [
        ([-1., 1.], [20., 30.]),
        ([5., 1.], [20., 30.]),
        ([10., 10.], [0., 0.]),
    ] {
        assert_gradient(
            &s.score_features_fd_gradient(&row, 64, 64, None).unwrap(),
            &expected,
        );
    }
}

#[test]
fn shipped_bakes_match_explicit_forward_composition_on_full_feature_rows() {
    let bakes: &[&[u8]] = &[
        include_bytes!("../../zensim/weights/v47_strict_qat_native_byid_2026-09-06.bin"),
        include_bytes!(
            "../../zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_byid_2026-09-06.bin"
        ),
        include_bytes!("../../zensim/weights/bhdr_linear_shaped_cvvdpmix_byid_2026-09-06.bin"),
        include_bytes!("../../zensim/weights/d_sdr_add156_id100_negrich_dial_byid_2026-09-06.bin"),
    ];
    for bytes in bakes {
        let m = Model::from_bytes(bytes).unwrap();
        let mut surface = BakeScorer::new(&m).unwrap();
        let mut predictor = Predictor::new(&m);
        let alpha = old::extract_per_sample_alpha_head(&m);
        let hybrid = old::extract_hybrid_head(&m);
        let pin = old::extract_tanh_output_head_scale(&m);
        let spl = spline::extract(&m);
        let gather = old::CallerGather::for_model(&m);
        let mut scratch = vec![0.; m.caller_input_width()];
        let ids = zensim::declared_feature_ids(&m).unwrap();
        for i in 0..256 {
            let mut row = vec![0.; 372];
            for (j, id) in ids.iter().enumerate() {
                let u = ((i * 47 + j * 13) % 101) as f64 / 25. - 2.;
                row[*id as usize] =
                    f64::from(m.scaler_mean()[j]) + u * f64::from(m.scaler_scale()[j]);
            }
            gather.fill(&mut scratch, &row);
            let out = if m.has_nontrivial_feature_transforms() {
                predictor.predict_transformed(&scratch)
            } else {
                predictor.predict(&scratch)
            }
            .unwrap();
            let form = zensim::det_math::active_pow_form();
            let raw = if let Some((wa, ba, rw, rb, red, b, p)) = &alpha {
                zensim::score_math::per_sample_alpha_head(
                    out,
                    &zensim::score_math::PerSampleAlphaParams {
                        w_alpha: wa,
                        b_alpha: *ba,
                        rank_w: rw,
                        rank_b: *rb,
                        reducer_w: *red,
                        reducer_b: *b,
                        p_norm: *p,
                    },
                    form,
                )
            } else if let Some((rw, rb, a, red, b, p)) = &hybrid {
                zensim::score_math::hybrid_head(
                    out,
                    &zensim::score_math::HybridHeadParams {
                        rank_w: rw,
                        rank_b: *rb,
                        alpha_logit: *a,
                        reducer_w: *red,
                        reducer_b: *b,
                        p_norm: *p,
                    },
                    form,
                )
            } else {
                out[0] as f64
            };
            let pinned = pin.map_or(raw, |p| zensim::score_math::tanh_output_pin(raw, p, form));
            let expected = spl.as_ref().map_or(pinned, |s| spline::apply(pinned, s));
            let got = surface.score_features(&row, 64, 64, None).unwrap();
            assert_eq!(got.to_bits(), expected.to_bits(), "row {i}");
        }
    }
}

#[test]
fn pixel_and_cached_feature_surfaces_match_named_profile() {
    let profile = ZensimProfile::D;
    let bytes =
        include_bytes!("../../zensim/weights/d_sdr_add156_id100_negrich_dial_byid_2026-09-06.bin");
    let model = Model::from_bytes(bytes).unwrap();
    let mut surface = BakeScorer::new(&model).unwrap();
    let src: Vec<[u8; 3]> = (0..96 * 80)
        .map(|i| {
            [
                (i % 239) as u8,
                ((i * 7) % 251) as u8,
                ((i * 13) % 253) as u8,
            ]
        })
        .collect();
    let dst: Vec<[u8; 3]> = src.iter().map(|p| [p[0] / 2, p[1], p[2]]).collect();
    let r = RgbSlice::new(&src, 96, 80);
    let d = RgbSlice::new(&dst, 96, 80);
    let expected = Zensim::new(profile).compute(&r, &d).unwrap();
    assert_eq!(
        surface.compute(&r, &d, None).unwrap().score().to_bits(),
        expected.score().to_bits()
    );
    assert_eq!(
        surface
            .score_features(expected.features(), 96, 80, None)
            .unwrap()
            .to_bits(),
        expected.score().to_bits()
    );
    assert_eq!(surface.compute(&r, &r, None).unwrap().score(), 100.);
    // An arbitrary zero feature row must still run the model, not claim identity.
    let linear = linear(json!([]));
    assert_eq!(
        BakeScorer::new(&linear)
            .unwrap()
            .score_features(&[0., 0.], 96, 80, None)
            .unwrap(),
        -5.
    );
}

#[test]
fn formula_revision_is_selected_per_bake_and_unknown_or_mixed_revisions_refuse() {
    let one = linear(json!([{"key":"zentrain.formula_revision","type":"utf8","text":"1"}]));
    let two = linear(json!([{"key":"zentrain.formula_revision","type":"utf8","text":"2"}]));
    assert!(BakeScorer::ensemble(&[one, two], None).is_err());
    for bad in ["", "3", "rev0", "second"] {
        let model = linear(json!([{"key":"zentrain.formula_revision","type":"utf8","text":bad}]));
        assert!(BakeScorer::new(&model).is_err());
    }
}

#[test]
fn dense_minmax_head_uses_transform_clamp_pin_and_spline_in_order() {
    let mut payload = Vec::new();
    for v in [2u32, 2, 2] {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    for v in [2f32, 1., -1., 3., 1., -1., 4., 0., 1., -2., 4., -3.] {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    let hex: String = payload.iter().map(|b| format!("{b:02x}")).collect();
    let model = linear(json!([
        {"key":"zentrain.feature_ids","type":"utf8","text":"1 3"},
        {"key":"zentrain.minmax_monotone_head","type":"bytes","hex":hex},
        {"key":"zentrain.tanh_output_head","type":"bytes","hex":"00002041"}
    ]));
    let mut scorer = BakeScorer::new(&model).unwrap();
    for (x, y, raw) in [
        (0., 0., 1.),
        (1., 2., 3.),
        (100., 0., 17.),
        (-100., 0., -4.),
    ] {
        let got = scorer
            .score_features(&[99., x, 99., y], 0, 0, None)
            .unwrap();
        let expected =
            zensim::score_math::tanh_output_pin(raw, 10., zensim::det_math::active_pow_form());
        assert_eq!(got.to_bits(), expected.to_bits());
    }
    // Here the active min-max piece is x-y+4. The independent pin
    // derivative checks that sensitivities follow the replacement head,
    // rather than the unused MLP's [2, 3] weights.
    let logistic = 1. / (1. + (-3.0_f64 / 10.).exp());
    let pin_slope = 10. * logistic * (1. - logistic);
    assert_gradient(
        &scorer
            .score_features_fd_gradient(&[99., 1., 99., 2.], 0, 0, None)
            .unwrap(),
        &[0., pin_slope, 0., -pin_slope],
    );
}

#[test]
fn hdr_pixel_surface_matches_the_canonical_pu_features() {
    use zensim::feature_v2::{HdrEncoding, V1PoolsMode, V2NewFeatureToggles, V2Scratch};
    use zensim::{AlphaMode, PixelFormat, StridedBytes};
    let bytes = |gain: f32| -> Vec<u8> {
        (0..64 * 64)
            .flat_map(|i| {
                let v = (i % 53) as f32 * 3. + 0.1;
                [v * gain, v * 0.8 * gain, v * 1.2 * gain, 1.]
                    .into_iter()
                    .flat_map(f32::to_ne_bytes)
            })
            .collect()
    };
    let a = bytes(1.);
    let b = bytes(0.8);
    let r = StridedBytes::with_alpha_mode(
        &a,
        64,
        64,
        64 * 16,
        PixelFormat::LinearF32Rgba,
        AlphaMode::Opaque,
    );
    let d = StridedBytes::with_alpha_mode(
        &b,
        64,
        64,
        64 * 16,
        PixelFormat::LinearF32Rgba,
        AlphaMode::Opaque,
    );
    let model = linear(json!([]));
    let mut scorer = BakeScorer::new(&model).unwrap();
    let extracted = Zensim::new(ZensimProfile::B)
        .compute_folded720_features_hdr(
            &r,
            &d,
            HdrEncoding::Linear,
            V2NewFeatureToggles {
                v1_only: true,
                v1_pools: V1PoolsMode::Peaks,
                ..Default::default()
            },
            &mut V2Scratch::new(),
        )
        .unwrap();
    let expected = scorer
        .score_features(extracted.features(), 64, 64, None)
        .unwrap();
    assert_eq!(
        scorer
            .compute_hdr(&r, &d, HdrEncoding::Linear, None)
            .unwrap()
            .to_bits(),
        expected.to_bits()
    );
    assert_eq!(
        scorer
            .compute_hdr(&r, &r, HdrEncoding::Linear, None)
            .unwrap(),
        100.
    );
    // Wide and dense C bakes consume the same canonical PU features.
    let canonical = Zensim::new(ZensimProfile::B)
        .compute_folded720_features_hdr(
            &r,
            &d,
            HdrEncoding::Linear,
            V2NewFeatureToggles {
                v1_pools: V1PoolsMode::Full,
                append_block: true,
                append2_block: true,
                csfw_block: true,
                ..Default::default()
            },
            &mut V2Scratch::new(),
        )
        .unwrap();
    for (wide, dense) in [
        (
            include_bytes!("../../zensim/weights/c_sdr_purity944_2026-08-29.bin").as_slice(),
            include_bytes!("../../zensim/weights/c_sdr_purity944_byid_2026-09-07.bin").as_slice(),
        ),
        (
            include_bytes!("../../zensim/weights/c_hdr_l1t1944_2026-08-29.bin").as_slice(),
            include_bytes!("../../zensim/weights/c_hdr_l1t1944_byid_2026-09-07.bin").as_slice(),
        ),
    ] {
        let models = [
            Model::from_bytes(wide).unwrap(),
            Model::from_bytes(dense).unwrap(),
        ];
        let mut values = Vec::new();
        for m in &models {
            let mut scorer = BakeScorer::new(m).unwrap();
            let pixel = scorer
                .compute_hdr(&r, &d, HdrEncoding::Linear, None)
                .unwrap();
            let cached = scorer
                .score_features(canonical.features(), 64, 64, None)
                .unwrap();
            assert_eq!(pixel.to_bits(), cached.to_bits());
            values.push(pixel);
        }
        assert_eq!(values[0].to_bits(), values[1].to_bits());
    }
    let srgb = vec![[123u8; 3]; 64 * 64];
    let wrong = RgbSlice::new(&srgb, 64, 64);
    assert!(
        scorer
            .compute_hdr(&wrong, &wrong, HdrEncoding::Linear, None)
            .is_err()
    );
}

#[test]
fn codec_affine_and_final_disposition_are_part_of_the_returned_score() {
    let mut payload = 1u32.to_le_bytes().to_vec();
    payload.extend_from_slice(&4u32.to_le_bytes());
    payload.extend_from_slice(b"jpeg");
    payload.extend_from_slice(&5f32.to_le_bytes());
    payload.extend_from_slice(&2f32.to_le_bytes());
    let hex: String = payload.iter().map(|x| format!("{x:02x}")).collect();
    let model = linear(json!([{"key":"zentrain.per_codec_calibration","type":"bytes","hex":hex}]));
    let mut scorer = BakeScorer::new(&model).unwrap();
    assert_eq!(
        scorer
            .score_features(&[5., 5.], 0, 0, Some("mozjpeg"))
            .unwrap(),
        45.
    );
    assert_eq!(
        scorer
            .score_features(&[5., 5.], 0, 0, Some("unknown"))
            .unwrap(),
        20.
    );
    assert_eq!(
        scorer
            .score_features(&[-2., -7.], 0, 0, Some("JPEG"))
            .unwrap(),
        -55.
    );
    assert_gradient(
        &scorer
            .score_features_fd_gradient(&[-2., -7.], 0, 0, Some("JPEG"))
            .unwrap(),
        &[4., 6.],
    );
    assert_gradient(
        &scorer
            .score_features_fd_gradient(&[-2., -7.], 0, 0, Some("unknown"))
            .unwrap(),
        &[2., 3.],
    );
    let params = zensim_validate::bake_runtime::post_mode_params("clamp").unwrap();
    let mut clamped = scorer.with_score_disposition(&params).unwrap();
    assert_eq!(
        clamped
            .score_features(&[-2., -7.], 0, 0, Some("jpg"))
            .unwrap(),
        0.
    );
    assert_gradient(
        &clamped
            .score_features_fd_gradient(&[-2., -7.], 0, 0, Some("jpg"))
            .unwrap(),
        &[0., 0.],
    );
    assert!(zensim_validate::bake_runtime::post_mode_params("clmap").is_err());
}

fn spatial_pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let src: Vec<_> = (1..=w * h)
        .map(|i| {
            [
                (i % 239) as u8,
                ((i * 7) % 251) as u8,
                ((i * 13) % 253) as u8,
            ]
        })
        .collect();
    let dst = src
        .iter()
        .enumerate()
        .map(|(i, p)| [p[0] / 2, p[1].saturating_add((i % 17) as u8), p[2]])
        .collect();
    (src, dst)
}

#[test]
fn candidate_attribution_matches_served_features_scores_and_reuses_sessions() {
    let bakes: &[&[u8]] = &[
        include_bytes!("../../zensim/weights/v47_strict_qat_native_byid_2026-09-06.bin"),
        include_bytes!(
            "../../zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_byid_2026-09-06.bin"
        ),
        include_bytes!("../../zensim/weights/c_sdr_purity944_byid_2026-09-07.bin"),
        include_bytes!("../../zensim/weights/d_sdr_add156_id100_negrich_dial_byid_2026-09-06.bin"),
    ];
    // Reuse across candidates and geometries, including C -> basic-only D.
    let mut session = zensim::Fused944Session::new();
    for (w, h) in [(96, 80), (71, 65), (31, 47), (1, 1)] {
        let (src, dst) = spatial_pair(w, h);
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        for (i, bytes) in bakes.iter().enumerate() {
            let model = Model::from_bytes(bytes).unwrap();
            let mut scorer = BakeScorer::new(&model).unwrap();
            let expected = scorer.compute(&rs, &ds, Some("jpeg")).unwrap();
            let pre = scorer.precompute_reference(&rs).unwrap();
            let full = scorer
                .compute_with_ref_and_attribution(&rs, &pre, &ds, Some("jpeg"), &mut session, 1)
                .unwrap();
            let binned = scorer
                .compute_with_ref_and_attribution(&rs, &pre, &ds, Some("jpeg"), &mut session, 8)
                .unwrap();
            for result in [&full, &binned] {
                assert_eq!(
                    result.result().score().to_bits(),
                    expected.score().to_bits(),
                    "bake {i}, {w}x{h}"
                );
                assert_eq!(
                    result.result().features(),
                    expected.features(),
                    "bake {i}, {w}x{h}"
                );
                assert_eq!(
                    result.result().raw_distance().to_bits(),
                    expected.raw_distance().to_bits()
                );
                assert!(!result.has_corruption_gate());
                if i >= 2 {
                    assert!(
                        result.unsupported_feature_ids().is_empty(),
                        "bake {i}: {:?}",
                        result.unsupported_feature_ids()
                    );
                }
                assert!(result.attribution().density().iter().all(|x| x.is_finite()));
            }
            for y in (0..h).step_by(8) {
                for x in (0..w).step_by(8) {
                    let a = full
                        .attribution()
                        .query_rect(x, y, (x + 8).min(w), (y + 8).min(h));
                    let b = binned
                        .attribution()
                        .query_rect(x, y, (x + 8).min(w), (y + 8).min(h));
                    assert!(
                        (a - b).abs() <= 1e-5 * a.abs().max(1e-6),
                        "bake {i}, {w}x{h} bin {x},{y}: {a} vs {b}"
                    );
                }
            }
            let repeat = scorer
                .compute_with_ref_and_attribution(&rs, &pre, &ds, Some("jpeg"), &mut session, 8)
                .unwrap();
            assert_eq!(
                repeat.attribution().density(),
                binned.attribution().density()
            );
            assert_eq!(repeat.sensitivities(), binned.sensitivities());
            let identity = scorer
                .compute_with_ref_and_attribution(&rs, &pre, &rs, None, &mut session, 8)
                .unwrap();
            assert_eq!(identity.result().score(), 100.);
            assert_eq!(identity.result().raw_distance(), 0.);
            assert!(identity.result().features().iter().all(|x| *x == 0.));
            assert!(identity.attribution().density().iter().all(|x| *x == 0.));
            assert!(identity.sensitivities().iter().all(|x| *x == 0.));
        }
    }
}

#[test]
fn candidate_attribution_reports_unsupported_terms_and_complete_gating() {
    let (src, dst) = spatial_pair(96, 80);
    let rs = RgbSlice::new(&src, 96, 80);
    let ds = RgbSlice::new(&dst, 96, 80);
    let mut session = zensim::Fused944Session::new();
    let model = linear(json!([{"key":"zentrain.feature_ids","type":"utf8","text":"3 156"}]));
    let mut scorer = BakeScorer::new(&model).unwrap();
    let pre = scorer.precompute_reference(&rs).unwrap();
    let scored = scorer
        .compute_with_ref_and_attribution(&rs, &pre, &ds, None, &mut session, 8)
        .unwrap();
    assert_eq!(scored.unsupported_feature_ids(), &[156]);
    assert!(scored.result().score() < 0.);
    assert_eq!(
        scored.result().score(),
        scorer.compute(&rs, &ds, None).unwrap().score()
    );

    // Reference-only luma and an HDR-gated SDR zero have exactly zero
    // distorted-image contributions, even though the model reads them.
    let refs = linear(json!([{"key":"zentrain.feature_ids","type":"utf8","text":"926 927"}]));
    let mut ref_scorer = BakeScorer::new(&refs).unwrap();
    let scored = ref_scorer
        .compute_with_ref_and_attribution(&rs, &pre, &ds, None, &mut session, 8)
        .unwrap();
    assert!(scored.unsupported_feature_ids().is_empty());
    assert!(scored.attribution().density().iter().all(|x| *x == 0.));

    let models = [linear_bias(json!([]), 5.), linear_bias(json!([]), 15.)];
    let mut gated = BakeScorer::ensemble(&models, Some(&[0.25, 0.75]))
        .unwrap()
        .with_linear_corruption_head(&model, 20.)
        .unwrap();
    let scored = gated
        .compute_with_ref_and_attribution(&rs, &pre, &ds, None, &mut session, 8)
        .unwrap();
    assert!(scored.has_corruption_gate());
    assert_eq!(scored.result().score(), 0.);
    assert_eq!(
        scored.result().score(),
        gated.compute(&rs, &ds, None).unwrap().score()
    );
    assert!(scored.attribution().density().iter().all(|x| *x == 0.));
    assert!(
        gated
            .compute_with_ref_and_attribution(&rs, &pre, &ds, None, &mut session, 0)
            .is_err()
    );
}

#[test]
fn accelerated_candidate_sensitivities_match_sequential_complete_surface() {
    let models = [
        Model::from_bytes(include_bytes!(
            "../../zensim/weights/c_sdr_purity944_byid_2026-09-07.bin"
        ))
        .unwrap(),
        Model::from_bytes(include_bytes!(
            "../../zensim/weights/d_sdr_add156_id100_negrich_dial_byid_2026-09-06.bin"
        ))
        .unwrap(),
    ];
    let (src, dst) = spatial_pair(96, 80);
    let mut scorer = BakeScorer::ensemble(&models, Some(&[0.75, 0.25])).unwrap();
    let row = scorer
        .compute(
            &RgbSlice::new(&src, 96, 80),
            &RgbSlice::new(&dst, 96, 80),
            Some("jpeg"),
        )
        .unwrap()
        .features()
        .to_vec();
    fn check(scorer: &mut BakeScorer<'_>, row: &[f64]) {
        // Deliberately call the public scalar surface twice for EVERY ID,
        // including unread ones. This is the unoptimized independent oracle.
        let mut expected = vec![0.; row.len()];
        for (i, out) in expected.iter_mut().enumerate() {
            let eps = (row[i].abs() * 1e-3).max(1e-5);
            let mut up = row.to_vec();
            let mut down = row.to_vec();
            up[i] += eps;
            down[i] -= eps;
            *out = (scorer.score_features(&up, 96, 80, Some("jpeg")).unwrap()
                - scorer.score_features(&down, 96, 80, Some("jpeg")).unwrap())
                / (2. * eps);
        }
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let actual = pool
                .install(|| scorer.score_features_fd_gradient(row, 96, 80, Some("jpeg")))
                .unwrap();
            for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "thread count {threads}, feature {i}"
                );
            }
        }
    }
    check(&mut scorer, &row);
    let head = linear(json!([]));
    let mut gated = scorer.with_linear_corruption_head(&head, 20.).unwrap();
    check(&mut gated, &row);
}
