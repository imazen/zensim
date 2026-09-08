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
    let params = zensim_validate::bake_runtime::post_mode_params("clamp").unwrap();
    let mut clamped = scorer.with_score_disposition(&params).unwrap();
    assert_eq!(
        clamped
            .score_features(&[-2., -7.], 0, 0, Some("jpg"))
            .unwrap(),
        0.
    );
    assert!(zensim_validate::bake_runtime::post_mode_params("clmap").is_err());
}
