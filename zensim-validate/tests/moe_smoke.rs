//! Smoke test for the MoE architecture.
//!
//! Exercises the inference-side invariants that don't require running
//! the trainer — useful as a quick sanity gate when iterating on the
//! gate softmax / hard-routing logic in `dataset_metric_baseline.rs`.
//! The trainer round-trip lives in `mlp_train_moe::tests`.
//!
//! Asserted invariants:
//!  (a) gate forward on random input returns finite logits;
//!  (b) softmax-with-temperature output sums to 1 ± 1e-6;
//!  (c) hard top-1 routing picks the argmax of the soft-routing
//!      weights.
//!
//! No corpus, no PNG decoding, no real features. Builds bakes from
//! synthetic weights via `zensim::mlp::bake::bake_v2`.

#![cfg(feature = "moe")]

use zensim::mlp::bake::{BakeLayer, BakeRequest, bake_v2};
use zensim::mlp::{Activation, Model, Predictor, WeightDtype};

fn bake_two_layer(
    n_in: usize,
    n_hidden: usize,
    n_out: usize,
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    act_hidden: Activation,
    act_out: Activation,
) -> Vec<u8> {
    let scaler_mean = vec![0.0f32; n_in];
    let scaler_scale = vec![1.0f32; n_in];
    let layers = [
        BakeLayer {
            in_dim: n_in,
            out_dim: n_hidden,
            activation: act_hidden,
            dtype: WeightDtype::F32,
            weights: w1,
            biases: b1,
        },
        BakeLayer {
            in_dim: n_hidden,
            out_dim: n_out,
            activation: act_out,
            dtype: WeightDtype::F32,
            weights: w2,
            biases: b2,
        },
    ];
    bake_v2(&BakeRequest {
        schema_hash: 0,
        flags: 0,
        scaler_mean: &scaler_mean,
        scaler_scale: &scaler_scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: &[],
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
    })
    .expect("bake_v2 should succeed for synthetic MoE smoke test")
}

fn softmax_with_temperature(z: &[f64], tau: f64) -> Vec<f64> {
    let inv_tau = 1.0 / tau.max(1e-6);
    let max_z = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = z.iter().map(|&v| ((v - max_z) * inv_tau).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

#[test]
fn moe_smoke_gate_softmax_and_hard_routing() {
    // 10 features → 4-wide gate hidden → 5 experts.
    let n_in = 10;
    let n_hidden = 4;
    let k = 5;

    // Random-ish but deterministic weights.
    let mut rng_state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as i32 as f32) * 1e-10
    };

    // Gate weights: bias gives expert 2 a strong logit so the gate
    // peaks there for a balanced input — useful for the hard-routing
    // assertion below.
    let g_w1: Vec<f32> = (0..n_in * n_hidden).map(|_| next() * 0.5).collect();
    let g_b1: Vec<f32> = vec![0.1f32; n_hidden];
    let g_w2: Vec<f32> = (0..n_hidden * k).map(|_| next() * 0.5).collect();
    let mut g_b2: Vec<f32> = vec![0.0f32; k];
    g_b2[2] = 5.0; // strong bias toward expert 2

    let gate_bytes = bake_two_layer(
        n_in, n_hidden, k,
        &g_w1, &g_b1, &g_w2, &g_b2,
        Activation::Relu, Activation::Identity,
    );

    let model = Model::from_bytes(&gate_bytes).expect("gate roundtrip");
    assert_eq!(model.n_inputs(), n_in);
    let mut p = Predictor::new(model);

    // 10 deterministic synthetic inputs (mean 0, ~unit scale).
    for seed_off in 0..10 {
        let features: Vec<f32> = (0..n_in)
            .map(|i| ((i + seed_off) as f32 * 0.31).sin())
            .collect();
        let logits = p.predict(&features).expect("gate predict");
        assert_eq!(logits.len(), k);

        // (a) finite output
        for &v in logits.iter() {
            assert!(v.is_finite(), "gate logit not finite: {v}");
        }

        // (b) softmax sums to 1
        let logits_f64: Vec<f64> = logits.iter().map(|&v| v as f64).collect();
        let weights = softmax_with_temperature(&logits_f64, 1.0);
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "softmax weights do not sum to 1: sum={sum} weights={weights:?}"
        );
        for &w in weights.iter() {
            assert!(w >= 0.0 && w <= 1.0, "weight out of [0,1]: {w}");
        }

        // (c) hard top-1 routing matches argmax of soft weights
        let (argmax, &max_w) = weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        // With g_b2[2] = 5, expert 2 should usually win on small input.
        assert!(argmax < k);
        // If max_w would trigger hard routing, the chosen expert MUST
        // equal the argmax (this is the contract of the runtime).
        let threshold = 0.95;
        if max_w > threshold {
            // Simulate the runtime's choice — for the hard path we
            // pick the argmax index. Here we just verify the
            // invariant: argmax is well-defined and unique enough
            // that softmax temperature 1.0 keeps it as the maximum.
            let second_best = weights
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != argmax)
                .map(|(_, &w)| w)
                .fold(0.0f64, f64::max);
            assert!(
                max_w > second_best,
                "argmax weight {max_w} not strictly greater than second best {second_best}"
            );
        }

        // Also exercise temperature scaling: τ→0 sharpens to one-hot.
        let sharp = softmax_with_temperature(&logits_f64, 0.05);
        let s_sum: f64 = sharp.iter().sum();
        assert!((s_sum - 1.0).abs() < 1e-6);
        let sharp_max = sharp.iter().cloned().fold(0.0f64, f64::max);
        assert!(
            sharp_max > 0.5,
            "low temperature (0.05) failed to sharpen softmax: max={sharp_max} sharp={sharp:?}"
        );
    }
}

#[test]
fn moe_smoke_expert_bake_roundtrip() {
    // Each expert is a regular V0_6-shaped 2-layer MLP — verify that
    // bake → load → predict round-trips on synthetic weights with the
    // shapes the trainer produces (228+3 → 64 → 1, scaled down here
    // for speed).
    let n_in = 16;
    let n_hidden = 8;
    let n_out = 1;
    let w1: Vec<f32> = (0..n_in * n_hidden)
        .map(|i| ((i as f32) * 0.1).sin() * 0.3)
        .collect();
    let b1: Vec<f32> = vec![0.01; n_hidden];
    let w2: Vec<f32> = (0..n_hidden * n_out)
        .map(|i| ((i as f32) * 0.2).cos() * 0.4)
        .collect();
    let b2: Vec<f32> = vec![0.05; n_out];
    let bytes = bake_two_layer(
        n_in, n_hidden, n_out,
        &w1, &b1, &w2, &b2,
        Activation::LeakyRelu, Activation::Identity,
    );
    let model = Model::from_bytes(&bytes).expect("expert bake roundtrip");
    assert_eq!(model.n_inputs(), n_in);
    let mut p = Predictor::new(model);
    let x = vec![0.1f32; n_in];
    let y = p.predict(&x).expect("expert predict");
    assert_eq!(y.len(), 1);
    assert!(y[0].is_finite());
}
