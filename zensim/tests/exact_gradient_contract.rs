//! Public loaded-model gradient contracts, including the E34 review reproductions.
use zensim::BakeScorer;

fn linear(weights: &[f64]) -> Vec<u8> {
    let n = weights.len();
    let spec = serde_json::json!({
        "schema_hash":1,"scaler_mean":vec![0.;n],"scaler_scale":vec![1.;n],
        "layers":[{"in_dim":n,"out_dim":1,"activation":"identity","dtype":"f32",
            "weights":weights,"biases":[0]}]
    });
    zenpredict_bake::bake_from_json_str(&spec.to_string()).unwrap()
}

#[test]
fn identity_widths_and_constant_size_axes_match_public_score_shapes() {
    let bytes = linear(&[2., 1., 1., 1., 1.]);
    let model = zenpredict::Model::from_bytes(&bytes).unwrap();
    let mut scorer = BakeScorer::new(&model).unwrap();
    for (name, features, score, expected) in [
        (
            "exact",
            vec![3., 4., 5., 6., 7.],
            28.0_f64,
            vec![2., 1., 1., 1., 1.],
        ),
        (
            "prefix",
            vec![3., 4., 5., 6., 7., 8.],
            28.,
            vec![2., 1., 1., 1., 1., 0.],
        ),
        ("size_axes", vec![3.], 30., vec![2.]),
    ] {
        let before = scorer.score_features(&features, 64, 64, None).unwrap();
        assert_eq!(before.to_bits(), score.to_bits(), "{name}");
        let gradient = scorer
            .score_features_fd_gradient(&features, 64, 64, None)
            .unwrap();
        assert_eq!(gradient, expected, "{name}");
        assert_eq!(
            before.to_bits(),
            scorer
                .score_features(&features, 64, 64, None)
                .unwrap()
                .to_bits()
        );
        println!("{name} score={before} gradient={gradient:?} score_bits_unchanged=true");
    }
    let short = [3., 4.];
    assert!(scorer.score_features(&short, 64, 64, None).is_err());
    assert!(
        scorer
            .score_features_fd_gradient(&short, 64, 64, None)
            .is_err()
    );
    println!("short input: score and gradient return errors");
}

fn transformed(transform: &str, params: Option<&str>) -> Vec<u8> {
    let bytes = linear(&[2.]);
    let mut bytes =
        zenpredict_bake::append_metadata_utf8(&bytes, "zentrain.feature_transforms", transform)
            .unwrap();
    if let Some(params) = params {
        bytes = zenpredict_bake::append_metadata_utf8(
            &bytes,
            "zentrain.feature_transform_params",
            params,
        )
        .unwrap();
    }
    bytes
}

#[test]
fn winsor_p99_default_and_supplied_bounds_have_flat_derivatives() {
    for (params, samples) in [
        (None, vec![(-2., -4., 2.), (0., 0., 2.), (2., 4., 2.)]),
        (
            Some("1"),
            vec![
                (-0.01, 2., 0.),
                (0., 2., 0.),
                (0.01, 2., 0.),
                (1., 2., 0.),
                (2., 4., 2.),
            ],
        ),
        (
            Some("1,3"),
            vec![
                (0., 2., 0.),
                (1., 2., 0.),
                (2., 4., 2.),
                (3., 6., 0.),
                (4., 6., 0.),
            ],
        ),
    ] {
        let bytes = transformed("winsor_p99", params);
        let model = zenpredict::Model::from_bytes(&bytes).unwrap();
        let mut scorer = BakeScorer::new(&model).unwrap();
        for (x, expected_score, expected_gradient) in samples {
            let score = scorer.score_features(&[x], 64, 64, None).unwrap();
            let gradient = scorer
                .score_features_fd_gradient(&[x], 64, 64, None)
                .unwrap();
            assert_eq!(score, expected_score, "{params:?} x={x}");
            assert_eq!(gradient, [expected_gradient], "{params:?} x={x}");
            assert_eq!(
                score.to_bits(),
                scorer.score_features(&[x], 64, 64, None).unwrap().to_bits()
            );
            println!(
                "winsor_p99 params={params:?} x={x} score={score} gradient={gradient:?} score_bits_unchanged=true"
            );
        }
    }
}

#[test]
fn stacked_winsor_transforms_keep_missing_parameter_fallbacks() {
    for (transform, expected) in [
        ("winsor_then_log", 2.),
        ("winsor_then_log1p", 1.),
        ("winsor_then_signed_cbrt", 2. / 3.),
    ] {
        let mut plain_bits = None;
        for params in [None, Some("3"), Some("3,5")] {
            let bytes = transformed(transform, params);
            let model = zenpredict::Model::from_bytes(&bytes).unwrap();
            let mut scorer = BakeScorer::new(&model).unwrap();
            let score = scorer.score_features(&[1.], 64, 64, None).unwrap();
            let gradient = scorer
                .score_features_fd_gradient(&[1.], 64, 64, None)
                .unwrap();
            assert_eq!(
                gradient,
                [if params == Some("3,5") { 0. } else { expected }]
            );
            assert_eq!(
                score.to_bits(),
                scorer
                    .score_features(&[1.], 64, 64, None)
                    .unwrap()
                    .to_bits()
            );
            match params {
                None => plain_bits = Some(score.to_bits()),
                Some("3") => assert_eq!(Some(score.to_bits()), plain_bits),
                _ => {}
            }
        }
    }
}

// The reviewer’s original public examples, asserting corrected behavior.
use std::panic::{AssertUnwindSafe, catch_unwind};
#[test]
fn reviewer_public_api_probe() {
    let spec = r#"{"schema_hash":1,"scaler_mean":[0,0,0,0,0],"scaler_scale":[1,1,1,1,1],"layers":[{"in_dim":5,"out_dim":1,"activation":"identity","dtype":"f32","weights":[2,1,1,1,1],"biases":[0]}]}"#;
    let bytes = zenpredict_bake::bake_from_json_str(spec).unwrap();
    let model = zenpredict::Model::from_bytes(&bytes).unwrap();
    let mut scorer = zensim::BakeScorer::new(&model).unwrap();
    println!(
        "size_axes score={:?}",
        scorer.score_features(&[3.], 64, 64, None)
    );
    let result = catch_unwind(AssertUnwindSafe(|| {
        scorer.score_features_fd_gradient(&[3.], 64, 64, None)
    }));
    println!("size_axes gradient={result:?}");
    assert_eq!(result.unwrap().unwrap(), [2.]);

    // The loader and runtime explicitly support WinsorP99's missing-high fallback.
    let spec = r#"{"schema_hash":1,"scaler_mean":[0],"scaler_scale":[1],"layers":[{"in_dim":1,"out_dim":1,"activation":"identity","dtype":"f32","weights":[2],"biases":[0]}]}"#;
    let bytes = zenpredict_bake::bake_from_json_str(spec).unwrap();
    let bytes =
        zenpredict_bake::append_metadata_utf8(&bytes, "zentrain.feature_transforms", "winsor_p99")
            .unwrap();
    let bytes =
        zenpredict_bake::append_metadata_utf8(&bytes, "zentrain.feature_transform_params", "1")
            .unwrap();
    let model = zenpredict::Model::from_bytes(&bytes).unwrap();
    let mut scorer = zensim::BakeScorer::new(&model).unwrap();
    let score = scorer.score_features(&[0.], 64, 64, None).unwrap();
    let up = scorer.score_features(&[0.01], 64, 64, None).unwrap();
    let down = scorer.score_features(&[-0.01], 64, 64, None).unwrap();
    let gradient = scorer
        .score_features_fd_gradient(&[0.], 64, 64, None)
        .unwrap();
    println!(
        "winsor_one_bound score={score} up={up} down={down} fd={} exact={gradient:?}",
        (up - down) / 0.02
    );
    assert_eq!(score, 2.);
    assert_eq!(up, down);
    assert_eq!(gradient, [0.]);
}
