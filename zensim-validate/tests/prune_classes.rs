//! The three classes of "dead" input, and which two are prunable.
//!
//! This is the correctness test for `zensim_validate::prune`. The
//! fixture bake carries one input of every shape the classifier has to
//! separate, and the load-bearing assertion is the **negative** one:
//! input 3 is constant across the whole evaluation corpus, has no
//! forcing transform, and has a live weight — the exact profile of the
//! "never populated in training" columns `bake_contrib` reports — and
//! it must survive pruning untouched. Pruning it would be a silent
//! behaviour change on any corpus that does exercise it, which is the
//! failure mode this module exists to prevent.

use zenpredict::{Activation, FeatureTransform, MetadataType, Model, Predictor, WeightDtype};
use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake};
use zensim_validate::prune::{self, DropReason};

#[repr(C, align(16))]
struct Aligned(Vec<u8>);

const N_RAW: usize = 6;
const OUT: usize = 3;

/// Layer-0 rows, row-major `N_RAW × OUT`. Rows 1 and 4 are exactly
/// zero (class 1); every other row is live.
const W0: [f32; N_RAW * OUT] = [
    0.75, -1.25, 0.5, // 0: live, identity                 -> RETAIN
    0.0, 0.0, 0.0, // 1: DEAD weights, identity            -> class 1
    -0.4, 0.9, 1.1, // 2: live, winsor_p99[3,3] (pinned)   -> class 2
    0.3, 0.6, -0.2, // 3: live, identity, corpus-constant  -> RETAIN (class 3)
    0.0, 0.0, 0.0, // 4: DEAD weights AND pinned transform -> class 1 (wins)
    -0.8, 0.2, 0.45, // 5: live, winsor_p99[0,1] (a range) -> RETAIN
];
const B0: [f32; OUT] = [0.11, -0.22, 0.33];
const W1: [f32; OUT] = [1.4, -0.6, 0.9];
const B1: [f32; 1] = [0.5];

const MEAN: [f32; N_RAW] = [0.5, -3.0, 2.0, 7.0, 0.25, -1.0];
const SCALE: [f32; N_RAW] = [2.0, 0.25, 1.5, 3.0, 0.5, 1.25];

const TRANSFORMS: &str = "identity\nidentity\nwinsor_p99\nidentity\nwinsor_p99\nwinsor_p99";
const PARAMS: &str = "\n\n3,3\n\n0,0\n0,1";

/// Input 3 is pinned to this value everywhere in the eval corpus — the
/// "inert on this corpus" trap. Nothing in the bake says so.
const CORPUS_CONST_INPUT_3: f32 = 7.5;

fn probes() -> Vec<[f32; N_RAW]> {
    vec![
        [0.0, 0.0, 0.0, CORPUS_CONST_INPUT_3, 0.0, 0.0],
        [1.0, 2.0, 3.0, CORPUS_CONST_INPUT_3, 4.0, 0.5],
        [-4.5, 17.25, 0.125, CORPUS_CONST_INPUT_3, -2.0, 0.9],
        [1e5, -1e-5, 42.0, CORPUS_CONST_INPUT_3, 100.0, 0.1],
        [-0.25, -0.5, -0.75, CORPUS_CONST_INPUT_3, -9.0, 1.0],
    ]
}

fn build(
    mean: &[f32],
    scale: &[f32],
    w0: &[f32],
    b0: &[f32],
    transforms: &str,
    params: &str,
) -> Vec<u8> {
    let layers = [
        BakeLayer {
            in_dim: mean.len(),
            out_dim: OUT,
            activation: Activation::LeakyRelu,
            dtype: WeightDtype::F32,
            weights: w0,
            biases: b0,
        },
        BakeLayer {
            in_dim: OUT,
            out_dim: 1,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &W1,
            biases: &B1,
        },
    ];
    let md = [
        BakeMetadataEntry {
            key: zenpredict::keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: transforms.as_bytes(),
        },
        BakeMetadataEntry {
            key: zenpredict::keys::FEATURE_TRANSFORM_PARAMS,
            kind: MetadataType::Utf8,
            value: params.as_bytes(),
        },
    ];
    bake(&BakeRequest {
        schema_hash: 0x5eed_0000_5eed_0000,
        flags: 0,
        scaler_mean: mean,
        scaler_scale: scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: &md,
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: false,
        hu_permutations: None,
    })
    .expect("bake fixture")
}

fn fixture() -> Aligned {
    Aligned(build(&MEAN, &SCALE, &W0, &B0, TRANSFORMS, PARAMS))
}

/// Run a plan through the same transformation `bake_dial_refit pack`
/// applies, and return the pruned bake's bytes.
fn apply(model: &Model, plan: &prune::PrunePlan) -> Aligned {
    let w0 = prune::prune_layer0_weights(plan, &W0, OUT);
    let b0 = prune::prune_layer0_biases(plan, &B0);
    let mean = prune::prune_input_array(plan, model.scaler_mean());
    let scale = prune::prune_input_array(plan, model.scaler_scale());
    let (t_txt, p_txt) = prune::transform_metadata(plan);
    Aligned(build(&mean, &scale, &w0, &b0, &t_txt, &p_txt))
}

fn plan_for(model: &Model, prune_constants: bool) -> prune::PrunePlan {
    let l0 = prune::Layer0View {
        in_dim: N_RAW,
        out_dim: OUT,
        weights: &W0,
        biases: &B0,
        is_i8: false,
    };
    prune::plan(model, &l0, prune_constants).expect("plan")
}

#[test]
fn classifies_all_three_classes_from_the_bake_alone() {
    let bytes = fixture();
    let model = Model::from_bytes(&bytes.0).expect("load fixture");
    let plan = plan_for(&model, true);

    let dropped: Vec<usize> = plan.drop.iter().map(|(k, _, _)| *k).collect();
    assert_eq!(
        dropped,
        vec![1, 2, 4],
        "exactly inputs 1, 2 and 4 are prunable"
    );

    // Class 1 — exactly-zero weight rows. Input 4 is BOTH weight-dead
    // and transform-pinned; weight-dead wins because it is the strictly
    // stronger (bit-identical) guarantee.
    assert_eq!(plan.drop[0].2, DropReason::WeightDead, "input 1");
    assert_eq!(plan.drop[2].2, DropReason::WeightDead, "input 4");

    // Class 2 — the transform pins input 2 to 3.0 for every input.
    match plan.drop[1].2 {
        DropReason::ForcedConstant {
            value,
            standardized,
        } => {
            assert_eq!(value, 3.0);
            assert_eq!(standardized, (3.0 - MEAN[2]) / SCALE[2]);
        }
        other => panic!("input 2 should be transform-forced, got {other:?}"),
    }

    // Class 3 — input 3 is constant across every probe, but nothing in
    // the BAKE says so. It keeps its column.
    assert!(plan.keep.contains(&3), "class-3 input 3 MUST be retained");
    assert_eq!(plan.keep, vec![0, 3, 5]);
    assert_eq!(plan.class_counts(), (2, 1));
    assert_eq!(plan.raw_width, N_RAW, "caller width never changes");
    assert_eq!(plan.n_inputs_after, 3);
    assert!(!plan.is_bit_identical(), "class 2 present");
}

#[test]
fn a_corpus_constant_input_with_a_live_weight_is_never_pruned() {
    // The load-bearing negative. Input 3 is indistinguishable from a
    // dead column in any corpus report — every score is the same, so an
    // ablation measures |Δ| = 0 — yet removing it changes the model for
    // every input the corpus did not contain. Prove that (a) the pruner
    // keeps it, and (b) it really is live: perturbing it moves the score.
    let bytes = fixture();
    let model = Model::from_bytes(&bytes.0).expect("load fixture");
    let plan = plan_for(&model, true);
    assert!(plan.keep.contains(&3));

    let mut pred = Predictor::new(&model);
    let mut a = probes()[1];
    let base = pred.predict_transformed(&a).expect("predict")[0];
    a[3] = CORPUS_CONST_INPUT_3 + 50.0;
    let moved = pred.predict_transformed(&a).expect("predict")[0];
    assert_ne!(
        base.to_bits(),
        moved.to_bits(),
        "input 3 has a live weight — a corpus that exercises it gets a different score, \
         which is exactly why corpus-inertness is not a pruning license"
    );
}

#[test]
fn class_one_only_pruning_is_bit_identical() {
    let bytes = fixture();
    let model = Model::from_bytes(&bytes.0).expect("load fixture");
    let plan = plan_for(&model, false);

    assert_eq!(plan.class_counts(), (2, 0), "class 2 disabled");
    assert!(plan.is_bit_identical());
    assert_eq!(plan.keep, vec![0, 2, 3, 5]);
    assert!(
        plan.bias_delta.iter().all(|&d| d == 0.0),
        "nothing folds when only zero rows are dropped"
    );

    let pruned = apply(&model, &plan);
    let pruned_model = Model::from_bytes(&pruned.0).expect("load pruned");
    assert_eq!(pruned_model.caller_input_width(), N_RAW);
    assert_eq!(pruned_model.n_inputs(), 4);

    let mut full = Predictor::new(&model);
    let mut small = Predictor::new(&pruned_model);
    for p in probes() {
        let a = full.predict_transformed(&p).expect("full")[0];
        let b = small.predict_transformed(&p).expect("pruned")[0];
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "class-1 pruning must be BIT-identical at {p:?}: {a} vs {b}"
        );
    }
}

#[test]
fn class_two_pruning_reproduces_the_full_model() {
    let bytes = fixture();
    let model = Model::from_bytes(&bytes.0).expect("load fixture");
    let plan = plan_for(&model, true);
    let pruned = apply(&model, &plan);
    let pruned_model = Model::from_bytes(&pruned.0).expect("load pruned");

    assert_eq!(pruned_model.caller_input_width(), N_RAW);
    assert_eq!(pruned_model.n_inputs(), 3);
    assert_eq!(
        pruned_model.feature_transforms().expect("present"),
        &[
            FeatureTransform::Identity,
            FeatureTransform::Drop,
            FeatureTransform::Drop,
            FeatureTransform::Identity,
            FeatureTransform::Drop,
            FeatureTransform::WinsorP99,
        ]
    );

    let mut full = Predictor::new(&model);
    let mut small = Predictor::new(&pruned_model);
    for p in probes() {
        let a = full.predict_transformed(&p).expect("full")[0];
        let b = small.predict_transformed(&p).expect("pruned")[0];
        assert!(
            (a - b).abs() <= 1e-5 * a.abs().max(1.0),
            "constant-fold prune diverged at {p:?}: {a} vs {b}"
        );
    }
}

#[test]
fn pruning_is_idempotent() {
    // Re-planning an already-pruned bake must find nothing left to do.
    // This is what makes `pack` safe to re-run on its own output.
    let bytes = fixture();
    let model = Model::from_bytes(&bytes.0).expect("load fixture");
    let plan = plan_for(&model, true);
    let pruned = apply(&model, &plan);
    let pruned_model = Model::from_bytes(&pruned.0).expect("load pruned");

    let w0 = prune::prune_layer0_weights(&plan, &W0, OUT);
    let b0 = prune::prune_layer0_biases(&plan, &B0);
    let l0 = prune::Layer0View {
        in_dim: plan.n_inputs_after,
        out_dim: OUT,
        weights: &w0,
        biases: &b0,
        is_i8: false,
    };
    let again = prune::plan(&pruned_model, &l0, true).expect("re-plan");
    assert!(
        again.is_noop(),
        "second pass should find nothing: {:?}",
        again.drop
    );
    assert_eq!(again.raw_width, N_RAW);
}

#[test]
fn a_pruned_bake_refuses_a_vector_sized_by_n_inputs() {
    let bytes = fixture();
    let model = Model::from_bytes(&bytes.0).expect("load fixture");
    let plan = plan_for(&model, true);
    let pruned = apply(&model, &plan);
    let pruned_model = Model::from_bytes(&pruned.0).expect("load pruned");
    let mut pred = Predictor::new(&pruned_model);
    assert!(
        pred.predict_transformed(&[1.0, 2.0, 3.0]).is_err(),
        "n_inputs()-sized vector must fail loud, never score a prefix"
    );
    assert!(pred.predict_transformed(&probes()[0]).is_ok());
}

#[test]
fn i8_layer0_refuses_constant_folding_but_allows_weight_dead() {
    // Removing a NONZERO row can change the per-output max-abs i8 scale
    // and therefore every other weight's quantization, so class 2 is
    // refused on i8. Class 1 rows are all-zero and cannot hold the max,
    // so they stay safe.
    let bytes = fixture();
    let model = Model::from_bytes(&bytes.0).expect("load fixture");
    let l0 = prune::Layer0View {
        in_dim: N_RAW,
        out_dim: OUT,
        weights: &W0,
        biases: &B0,
        is_i8: true,
    };
    assert!(
        prune::plan(&model, &l0, true).is_err(),
        "class 2 on i8 layer 0 must be refused"
    );
    let p = prune::plan(&model, &l0, false).expect("class 1 on i8 is fine");
    assert_eq!(p.class_counts(), (2, 0));
    assert!(p.is_bit_identical());
}
