//! Exact reverse derivative of the loaded mathematical score graph.
//! Scalar f64 accumulation is deliberate: gradients have one arithmetic order
//! across inference dispatch tiers. The ordinary score forward is untouched.
use super::*;
use zenpredict::{Activation, FeatureTransform, WeightStorage};

type Node = usize;
struct Value {
    value: f64,
    parents: Vec<(Node, f64)>,
}
#[derive(Default)]
struct Tape {
    nodes: Vec<Value>,
}
impl Tape {
}

#[allow(clippy::too_many_arguments)] // Mirrors the two loaded pool-head parameter records.

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn f32_f16_i8_storage_and_dispatch_have_exact_derivatives() {
        if !crate::ssim_form::run_in_own_process(
            "metric::bake::gradient::tests::f32_f16_i8_storage_and_dispatch_have_exact_derivatives",
            "D12_STORAGE_DISPATCH_OK",
            None,
        ) {
            return;
        }
        let _lock = archmage::testing::lock_token_testing();
        for dtype in ["f32", "f16", "i8"] {
            let spec = serde_json::json!({"schema_hash":1,"scaler_mean":[0.,0.],"scaler_scale":[2.,4.],
                "metadata":[{"key":"zentrain.feature_ids","type":"utf8","text":"13\n52"}],
                "layers":[{"in_dim":2,"out_dim":3,"activation":"relu","dtype":dtype,"weights":[1.,-1.,2.,0.5,2.,-1.],"biases":[1.,-1.,0.5]},
                {"in_dim":3,"out_dim":1,"activation":"identity","dtype":dtype,"weights":[-1.,-2.,-0.25],"biases":[100.]}]});
            let bytes = zenpredict_bake::bake_from_json_str(&spec.to_string()).unwrap();
            let model = crate::mlp::Model::from_bytes(&bytes).unwrap();
            let mut row = vec![0.; 372];
            row[13] = 3.;
            row[52] = 2.;
            let mut bits = None;
            let report = archmage::testing::for_each_token_permutation(
                archmage::testing::CompileTimePolicy::Warn,
                |_| {
                    let mut scorer = BakeScorer::new(&model).unwrap();
                    let before = scorer.score_features(&row, 96, 96, None).unwrap().to_bits();
                    let gradient = scorer
                        .score_features_fd_gradient(&row, 96, 96, None)
                        .unwrap();
                    assert_eq!(
                        before,
                        scorer.score_features(&row, 96, 96, None).unwrap().to_bits()
                    );
                    if dtype != "i8" {
                        assert_eq!(gradient[13], -0.75);
                        assert_eq!(gradient[52], -0.0625);
                    }
                    let current: Vec<_> = gradient.iter().map(|x| x.to_bits()).collect();
                    if let Some(bits) = &bits {
                        assert_eq!(&current, bits);
                    } else {
                        bits = Some(current);
                    }
                },
            );
            assert!(report.permutations_run > 0);
        }
        println!("D12_STORAGE_DISPATCH_OK");
    }
    #[test]
    fn transform_derivatives_match_independent_closed_forms_and_boundaries() {
        let cases = [
            (FeatureTransform::Log, 2., vec![], 0.5),
            (FeatureTransform::Log1p, 3., vec![], 0.25),
            (FeatureTransform::SignedLog1p, -3., vec![], 0.25),
            (FeatureTransform::SignedSqrt, -4., vec![], 0.25),
            (FeatureTransform::SignedCbrt, -8., vec![], 1. / 12.),
            (FeatureTransform::SoftSign, -3., vec![1.], 1. / 16.),
            (FeatureTransform::SoftClip, 3., vec![1., 2.], 0.5),
            (FeatureTransform::SignedPow, -4., vec![2.], 8.),
            (FeatureTransform::ClipThenLog1p, 4., vec![1.], 0.25),
            (FeatureTransform::ClipThenLog1p, 0., vec![1.], 0.),
            (FeatureTransform::WinsorP99, 4., vec![1., 3.], 0.),
            (FeatureTransform::WinsorThenLog, 2., vec![1., 3.], 0.5),
            (FeatureTransform::WinsorThenLog1p, 3., vec![1., 4.], 0.25),
            (
                FeatureTransform::WinsorThenSignedCbrt,
                8.,
                vec![1., 27.],
                1. / 12.,
            ),
            (
                FeatureTransform::SignedCbrtThenWinsor,
                8.,
                vec![1., 3.],
                1. / 12.,
            ),
            (
                FeatureTransform::ClipThenLog1pThenWinsor,
                4.,
                vec![1., 0., 2.],
                0.25,
            ),
            (FeatureTransform::QuantileBins, 2., vec![1., 3.], 0.),
            (FeatureTransform::YeoJohnson, 3., vec![0.], 0.25),
            (FeatureTransform::YeoJohnson, -3., vec![2.], 0.25),
            (FeatureTransform::YeoJohnson, 1e-6, vec![1e8], 1.),
        ];
        for (tf, x, p, expected) in cases {
            let mut t = Tape::default();
            let x = t.constant(x);
            let y = transform(&mut t, x, tf, &p).unwrap()[0];
            assert!(
                (t.reverse(y, 1).unwrap()[0] - expected).abs() < 1e-12,
                "{tf:?}"
            );
        }
        let mut t = Tape::default();
        let x = t.constant(0.25);
        let y = transform(&mut t, x, FeatureTransform::Sinusoidal, &[1.]).unwrap();
        assert_eq!(y.len(), 2);
        assert!((t.reverse(y[1], 1).unwrap()[0] + 2. * std::f64::consts::PI).abs() < 1e-12);
        assert!(
            transform(&mut t, x, FeatureTransform::Drop, &[])
                .unwrap()
                .is_empty()
        );
        let mut t = Tape::default();
        let x = t.constant(0.);
        let y = transform(&mut t, x, FeatureTransform::SignedSqrt, &[]).unwrap()[0];
        assert!(
            t.reverse(y, 1).is_err(),
            "a singular active derivative must not be fabricated"
        );
    }
    #[test]
    fn per_sample_gate_chain_rule_and_spline_caps_are_exact() {
        let mut t = Tape::default();
        let h = [t.constant(1.), t.constant(3.)];
        let logit = t.affine(&h, &[1., -1.], 2.);
        let gate = t.sigmoid(logit, 20.);
        let y = pool(&mut t, &h, &[1., 1.], 0., &[1.; 4], 0., 2., gate);
        let g = t.reverse(y, 2).unwrap();
        let r = 5_f64.sqrt();
        assert!((g[0] + 1. / r).abs() < 1e-12);
        assert!((g[1] - (2. + 0.75 / r + r / 4.)).abs() < 1e-12);
        let s = OutputCalibrationSpline {
            xs: vec![0., 1., 2.],
            ys: vec![20., 50., 80.],
            derivs: vec![30.; 3],
        };
        for (x, expected) in [(-2., 0.), (-1., 30.), (0.5, 30.), (2.5, 30.), (3., 0.)] {
            assert_eq!(spline_slope(x, &s), expected);
        }
    }
    #[test]
    fn pool_heads_match_closed_form_on_nonuniform_hidden_values() {
        // h=[1,3]: mu=2, sigma=1, max=3, RMS=sqrt(5).
        // Constant alpha=1/2, rank=sum(h), pool=mu+sigma+max+RMS.
        let mut t = Tape::default();
        let h = [t.constant(1.), t.constant(3.)];
        let gate = t.constant(0.5);
        let y = pool(&mut t, &h, &[1., 1.], 0., &[1.; 4], 0., 2., gate);
        let g = t.reverse(y, 2).unwrap();
        assert!((g[0] - (0.5 + 0.25 / 5_f64.sqrt())).abs() < 1e-12);
        assert!((g[1] - (1.5 + 0.75 / 5_f64.sqrt())).abs() < 1e-12);
    }
}
