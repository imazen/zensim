//! `--nonneg-distance`: the dial's identity and no-cell-above-identity rows are
//! made STRUCTURAL, and these tests are the proof.
//!
//! `benchmarks/dial_addressability_gate_2026-09-04.md` §10.3 proves that when
//! real grid cells out-rank a perfect copy in RAW space, no monotone output
//! spline can satisfy both C2 (no flat/clamp dead-zone) and C6 (no cell above
//! identity): pin identity at 100 and those cells cap, leave it below and they
//! out-score identity. It is a weights defect. The fastclass2 228 MLP has it
//! badly — identity dial 90.9368 with **1,642 of 9,593** grid cells above
//! identity (17.1 %, worse than shipped B's 6.01 %).
//!
//! Under this flag `raw(x) = pin − g(x)` with `g ≥ 0` and `g(0⃗) = 0`
//! bit-exactly, so `raw(0⃗)` is the argmax of `raw` over the ENTIRE input space
//! and the either/or cannot arise. These tests assert exactly that, through the
//! same runtime `bake_verdict` scores with, at every weight dtype that ships.
//!
//! They double as the train/serve gate for `Activation::Relu`: the guarantee
//! depends on `h ≥ 0`, which is only true if the bake's declared activation is
//! the one that was trained with. A bake that said `LeakyRelu` while the fit
//! used ReLU would let `h` go negative, `w₂·h` go positive, and `raw` exceed
//! the pin — which these tests would catch.

use zenpredict::{Activation, Model, Predictor, WeightDtype};
use zensim_validate::bake_runtime::score_with_bake_alloc;
use zensim_validate::mlp_train::{
    FeatureRows, GroupLossMode, MlpHyperparams, TrainingGroup, ValidationPolicy, train_mlp_strategy,
};

const N_FEATURES: usize = 14;
const PIN: f64 = 100.0;

fn rng_stream(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 * (1.0 / (1u64 << 53) as f64) - 0.5) * 3.0
    }
}

/// Train a small `--nonneg-distance` bake and return its bytes.
fn train_nonneg(dtype: WeightDtype, pin: f64, nonneg: bool) -> Vec<u8> {
    let mut next = rng_stream(20260906);
    let w: Vec<f64> = (0..N_FEATURES)
        .map(|i| (i as f64 - N_FEATURES as f64 / 2.0) * 0.3)
        .collect();
    let mut feats: Vec<Vec<f64>> = Vec::new();
    let mut quality: Vec<f64> = Vec::new();
    for _ in 0..220 {
        // Distortion-shaped inputs: non-negative magnitudes, like the real
        // feature block, so the fit is representative rather than adversarial.
        let x: Vec<f64> = (0..N_FEATURES).map(|_| next().abs()).collect();
        let y: f64 = -x
            .iter()
            .zip(w.iter())
            .map(|(a, b)| a * b.abs())
            .sum::<f64>();
        feats.push(x);
        quality.push(y);
    }
    let lo = quality.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = quality.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    for q in quality.iter_mut() {
        *q = (*q - lo) / (hi - lo);
    }
    let feats_ref: Vec<&[f64]> = feats.iter().map(|v| v.as_slice()).collect();
    let mut groups = [TrainingGroup {
        name: "synth".to_string(),
        human_scores: &quality,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::Both,
    }];
    let hyper = MlpHyperparams {
        nonneg_distance: nonneg,
        nonneg_pin: pin,
        n_hidden: 10,
        n_epochs: 40,
        pairs_per_epoch: 800,
        initial_lr: 0.005,
        log_every: 10_000,
        early_stop_patience: 0,
        validation_policy: ValidationPolicy::Mean,
        mse_weight: 1.0,
        out_dtype: dtype,
        seed: 4004,
        ..Default::default()
    };
    let mut log = Vec::new();
    train_mlp_strategy(
        &mut groups,
        N_FEATURES,
        &hyper,
        &mut log,
        None,
        None,
        None,
        None,
        None,
        None,
    )
}

fn score(bytes: &'static [u8], x: &[f64]) -> f64 {
    let model = Model::from_bytes(bytes).expect("bake loads");
    let n = model.caller_input_width();
    let mut p = Predictor::new(&model);
    score_with_bake_alloc(&mut p, false, None, None, None, None, n, x)
}

/// `raw(0⃗) == pin`, BIT-exactly, at every dtype that ships.
///
/// Bit-exact and not "within tolerance": `0·w` is exactly `0` in f32, f16 and
/// i8 alike, `ReLU(0) = 0`, `dot(0⃗, w) = 0`, and the frozen output bias passes
/// through untouched. If this ever needs a tolerance, the guarantee is gone.
#[test]
fn nonneg_distance_output_is_exactly_the_pin_on_the_zero_vector() {
    for dtype in [WeightDtype::F32, WeightDtype::F16, WeightDtype::I8] {
        let bytes: &'static [u8] = Box::leak(train_nonneg(dtype, PIN, true).into_boxed_slice());
        let zero = vec![0.0f64; N_FEATURES];
        let got = score(bytes, &zero);
        assert_eq!(
            got.to_bits(),
            PIN.to_bits(),
            "{dtype:?}: raw(0⃗) must be the pin BIT-exactly; got {got}"
        );
    }
}

/// A non-pinned pin is still exact — the constant is data, not a magic number.
#[test]
fn nonneg_distance_pin_is_exact_at_a_non_default_value() {
    let bytes: &'static [u8] =
        Box::leak(train_nonneg(WeightDtype::F32, 42.5, true).into_boxed_slice());
    let got = score(bytes, &[0.0f64; N_FEATURES]);
    assert_eq!(got.to_bits(), 42.5f64.to_bits(), "got {got}");
}

/// `raw(x) ≤ pin` for EVERY input, so no cell can out-score a perfect copy.
///
/// The sweep deliberately includes inputs far outside anything the fit saw —
/// large magnitudes, both signs, subnormals — because C6 is a claim about the
/// whole input space, not about the eval grid. Non-finite inputs are excluded:
/// `0 · inf = NaN` is a caller bug with its own story, not a counterexample to
/// this one.
#[test]
fn nonneg_distance_output_never_exceeds_the_pin() {
    let bytes: &'static [u8] =
        Box::leak(train_nonneg(WeightDtype::F32, PIN, true).into_boxed_slice());
    let model = Model::from_bytes(bytes).expect("bake loads");
    let n = model.caller_input_width();
    let mut p = Predictor::new(&model);
    let mut next = rng_stream(777);
    let scales = [1.0f64, 1e-30, 1e-6, 3.0, 1e3, 1e12];
    let mut lowest = f64::INFINITY;
    let mut n_checked = 0usize;
    let mut n_at_pin = 0usize;
    for scale in scales {
        for _ in 0..20_000 {
            let x: Vec<f64> = (0..N_FEATURES).map(|_| next() * scale).collect();
            let y = score_with_bake_alloc(&mut p, false, None, None, None, None, n, &x);
            if y.is_finite() {
                lowest = lowest.min(y);
                if y == PIN {
                    n_at_pin += 1;
                }
            }
            n_checked += 1;
            // A bare `y <= PIN` would also reject NaN (every ordered comparison
            // against NaN is false in IEEE 754), but via the generic "exceeds
            // the pin" message below — say what is required of a non-finite
            // result explicitly instead. (Review 2026-09-06.)
            assert!(
                y.is_finite(),
                "raw({x:?}) = {y} is not finite — the guarantee is only meaningful \
                 on a finite output"
            );
            assert!(
                y <= PIN,
                "raw({x:?}) = {y} exceeds the pin {PIN} — the C6 guarantee is not held"
            );
        }
    }
    assert_eq!(n_checked, 120_000);
    // The pin is ATTAINED, not merely an upper bound: an input whose every
    // firing hidden unit has a zeroed output weight — or which turns every unit
    // off — lands exactly on it. That is the boundary being tight, which is what
    // makes `raw(0⃗)` the argmax rather than merely an upper bound.
    assert!(
        n_at_pin > 0,
        "no probe attained the pin — the bound may be loose"
    );
    // Sanity: the network must actually MOVE, or the test would pass vacuously
    // on a constant one.
    assert!(
        lowest < PIN - 1.0,
        "every probe returned within 1.0 of the pin (lowest {lowest}) — the \
         network is effectively constant and this test proved nothing"
    );
}

/// `--leaky-alpha 0` must reach the wire format's ReLU byte. Before 2026-09-06
/// every emitter hard-coded `LeakyRelu` and the runtime applied a fixed 0.01
/// slope, so the trained and served functions differed with no warning.
#[test]
fn leaky_alpha_zero_bakes_relu_and_the_default_bakes_leaky_relu() {
    let relu: &'static [u8] =
        Box::leak(train_nonneg(WeightDtype::F32, PIN, true).into_boxed_slice());
    let m = Model::from_bytes(relu).expect("loads");
    assert_eq!(
        m.layer(0).activation,
        Activation::Relu,
        "--nonneg-distance forces leaky_alpha = 0, which must bake as ReLU"
    );
    let leaky: &'static [u8] =
        Box::leak(train_nonneg(WeightDtype::F32, PIN, false).into_boxed_slice());
    let m2 = Model::from_bytes(leaky).expect("loads");
    assert_eq!(
        m2.layer(0).activation,
        Activation::LeakyRelu,
        "the default 0.01 slope must still bake as LeakyRelu — byte-identity"
    );
}

/// A slope the wire format cannot express must fail loud, not silently serve a
/// different function.
#[test]
#[should_panic(expected = "not representable in the ZNPR wire format")]
fn unrepresentable_leaky_alpha_fails_loud() {
    let feats = vec![vec![0.5f64; N_FEATURES]; 8];
    let quality = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let feats_ref: Vec<&[f64]> = feats.iter().map(|v| v.as_slice()).collect();
    let mut groups = [TrainingGroup {
        name: "g".to_string(),
        human_scores: &quality,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::Rank,
    }];
    let hyper = MlpHyperparams {
        leaky_alpha: 0.2,
        n_hidden: 4,
        n_epochs: 1,
        pairs_per_epoch: 4,
        log_every: 10_000,
        early_stop_patience: 0,
        validation_policy: ValidationPolicy::Mean,
        ..Default::default()
    };
    let mut log = Vec::new();
    train_mlp_strategy(
        &mut groups,
        N_FEATURES,
        &hyper,
        &mut log,
        None,
        None,
        None,
        None,
        None,
        None,
    );
}

fn refuse_with(hyper: MlpHyperparams) {
    let feats = vec![vec![0.5f64; N_FEATURES]; 8];
    let quality = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let feats_ref: Vec<&[f64]> = feats.iter().map(|v| v.as_slice()).collect();
    let mut groups = [TrainingGroup {
        name: "g".to_string(),
        human_scores: &quality,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::Rank,
    }];
    let mut log = Vec::new();
    train_mlp_strategy(
        &mut groups,
        N_FEATURES,
        &hyper,
        &mut log,
        None,
        None,
        None,
        None,
        None,
        None,
    );
}

fn refusal_base() -> MlpHyperparams {
    MlpHyperparams {
        nonneg_distance: true,
        n_hidden: 4,
        n_epochs: 1,
        pairs_per_epoch: 4,
        log_every: 10_000,
        early_stop_patience: 0,
        validation_policy: ValidationPolicy::Mean,
        ..Default::default()
    }
}

#[test]
#[should_panic(expected = "incompatible with --skip-connection")]
fn nonneg_distance_refuses_skip_connection() {
    refuse_with(MlpHyperparams {
        skip_connection: true,
        ..refusal_base()
    });
}

#[test]
#[should_panic(expected = "plain n_features → n_hidden → 1")]
fn nonneg_distance_refuses_the_alpha_head() {
    refuse_with(MlpHyperparams {
        per_sample_alpha_head: true,
        ..refusal_base()
    });
}

#[test]
#[should_panic(expected = "1-hidden-layer plain path only")]
fn nonneg_distance_refuses_depth_two() {
    refuse_with(MlpHyperparams {
        n_hidden_layers: 2,
        ..refusal_base()
    });
}
