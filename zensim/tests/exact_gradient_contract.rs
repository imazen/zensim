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
