//! `--monotone-cbc` correct-by-construction projection mechanism test.
//!
//! Validates the trainer's `monotone_cbc=true` path on a small synthetic
//! dataset (no OOM risk; safesyn-sized N² allocations don't apply at
//! this scale). After training, the shipped bake's weight signs must be
//! exactly:
//!   - encoder `w1` (and `w2_enc` for 2-layer): all values `≥ 0`
//!   - rank-head `rank_w` (and `w_skip` if present): all values `≤ 0`
//!   - per-sample-α gate: `w_alpha = 0`, `b_alpha = 30`
//!
//! With the increasing tanh pin (`100·σ(y_pre/scale)`) wrapping
//! `y_pre = rank_w · h + rank_b` (and `h = LeakyReLU(w1·x + b1)`), this
//! sign pattern makes the score:
//!   1. Bounded `[0, 100]` (the tanh pin guarantees it).
//!   2. Monotone non-increasing in every non-negative input feature
//!      (`x → h` ascends because `w1 ≥ 0` and LeakyReLU is monotone;
//!      `h → y_pre` descends because `rank_w ≤ 0`; tanh-pin is
//!      monotone-increasing in its argument).
//!
//! Combined: a heavier-distortion input (larger non-negative features)
//! produces a lower score, by construction, on the ENTIRE input domain
//! — not just the training manifold. This is the V39 failure mode
//! (off-manifold inversion) closed structurally.

use zenpredict::{Model, WeightStorage};
use zensim_validate::mlp_train::{MlpHyperparams, TrainingGroup, ValidationPolicy, train_mlp};

fn synthetic_group<'a>(
    n_rows: usize,
    n_features: usize,
    features_buf: &'a mut Vec<Vec<f64>>,
    scores_buf: &'a mut Vec<f64>,
    feat_refs: &'a mut Vec<&'a [f64]>,
) -> TrainingGroup<'a> {
    // Synthetic: each row is a non-negative feature vector; the human
    // score decreases as the feature magnitude increases (a "distortion
    // magnitude" stand-in). The trainer should learn `score ∝ -‖x‖`.
    for i in 0..n_rows {
        let t = i as f64 / n_rows as f64;
        let row: Vec<f64> = (0..n_features)
            .map(|d| t * (1.0 + (d as f64 * 0.13).sin().abs()))
            .collect();
        features_buf.push(row);
        scores_buf.push(100.0 - 95.0 * t);
    }
    for row in features_buf.iter() {
        feat_refs.push(row.as_slice());
    }
    TrainingGroup {
        name: "synth_monotone".to_string(),
        human_scores: scores_buf,
        features: feat_refs,
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
    }
}

fn run_train(monotone_cbc: bool) -> Vec<u8> {
    let n_rows = 128;
    let n_features = 16;
    let mut features_buf: Vec<Vec<f64>> = Vec::with_capacity(n_rows);
    let mut scores_buf: Vec<f64> = Vec::with_capacity(n_rows);
    let mut feat_refs: Vec<&[f64]> = Vec::with_capacity(n_rows);
    let group = synthetic_group(
        n_rows,
        n_features,
        &mut features_buf,
        &mut scores_buf,
        &mut feat_refs,
    );

    let mut hp = MlpHyperparams::default();
    hp.n_hidden = 8;
    hp.n_epochs = 5;
    hp.pairs_per_epoch = 200;
    hp.initial_lr = 1e-2;
    hp.seed = 42;
    hp.per_sample_alpha_head = true;
    hp.mse_weight = 1.0;
    hp.ranknet_weight = 1.0;
    hp.tanh_output_head_scale = 20.0;
    hp.minibatch_size = 8;
    hp.validation_policy = ValidationPolicy::Min;
    hp.monotone_cbc = monotone_cbc;
    hp.log_every = 100;
    hp.parallel_batch = false;

    let mut log: Vec<String> = Vec::new();
    train_mlp(std::slice::from_ref(&group), n_features, &hp, &mut log)
}

#[test]
fn monotone_cbc_projection_signs_exact() {
    let bake = run_train(true);
    let model = Model::from_bytes(&bake).expect("bake parses");

    // n_features × n_hidden encoder, then n_hidden × 1 head (rank).
    let n_features = model.n_inputs();
    let n_hidden = model.n_outputs();
    let layer_views: Vec<_> = model.layers().collect();
    assert!(
        layer_views.len() >= 1,
        "bake has at least one layer; got {}",
        layer_views.len()
    );

    // Layer 0 = encoder. Weights must be ≥ 0 under monotone_cbc.
    let min_w1 = match &layer_views[0].weights {
        WeightStorage::F32(w) => w.iter().cloned().fold(f32::INFINITY, f32::min),
        WeightStorage::F16(_) | WeightStorage::I8 { .. } => {
            panic!("expected F32 encoder weights for monotone_cbc bake (out_dtype=f32)")
        }
    };
    assert!(
        min_w1 >= -1e-6,
        "monotone_cbc=true: encoder w1 must be ≥ 0; min={min_w1}"
    );

    // rank_w is in the metadata payload (per-sample-α head). All entries ≤ 0.
    let md = model.metadata();
    let entry = md
        .get("zentrain.per_sample_alpha_head")
        .expect("monotone_cbc bake has per_sample_alpha_head metadata");
    let mut floats = Vec::with_capacity(2 * n_hidden + 8);
    for chunk in entry.value.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let w_alpha = &floats[..n_hidden];
    let b_alpha = floats[n_hidden];
    let rank_w = &floats[n_hidden + 1..2 * n_hidden + 1];

    let max_rank = rank_w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_rank <= 1e-6,
        "monotone_cbc=true: rank_w must be ≤ 0; max={max_rank}"
    );
    let max_alpha = w_alpha.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_alpha <= 1e-6,
        "monotone_cbc=true: w_alpha must be exactly 0; |max|={max_alpha}"
    );
    assert!(
        (b_alpha - 30.0).abs() < 1e-3,
        "monotone_cbc=true: b_alpha must be 30; got {b_alpha}"
    );

    // Sanity: trainer didn't crash; n_features is what we passed.
    assert_eq!(n_features, 16);
    assert_eq!(n_hidden, 8);
}

#[test]
fn monotone_cbc_off_does_not_project() {
    // With monotone_cbc=false the trainer is free to learn any signs.
    // We don't enforce sign invariants here; we only check the bake
    // SHIPS (i.e. monotone_cbc machinery doesn't silently apply when
    // disabled).
    let bake = run_train(false);
    let model = Model::from_bytes(&bake).expect("bake parses");
    let md = model.metadata();
    let entry = md
        .get("zentrain.per_sample_alpha_head")
        .expect("bake has per_sample_alpha_head metadata");
    let n_hidden = model.n_outputs();
    let mut floats = Vec::with_capacity(2 * n_hidden + 8);
    for chunk in entry.value.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let b_alpha = floats[n_hidden];
    // b_alpha=30 is the monotone_cbc force-value; with monotone_cbc=false
    // it should differ (default init is 0 → small training drift).
    assert!(
        (b_alpha - 30.0).abs() > 1.0,
        "monotone_cbc=false: b_alpha should NOT be force-set to 30; got {b_alpha}"
    );
}
