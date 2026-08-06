//! C2 wrong-regime guard (opus-review, campaign appendix W): the library
//! predicate behind `bake_verdict`'s `--regime 944` refusal.
//!
//! The class this kills has two published instances: the `ebothg_m504` board
//! row (a wrong-root read that sat on the board), and appendix U.R0 — shipped
//! B (a 372-input bake with 49 structurally-used lines in f156-371) scored at
//! `--regime 944`, where that block is STRUCTURAL ZEROS, reading CID22 0.3862
//! against its true 0.8764. No error, no warning, a plausible number half a
//! point low. The guard turns the whole class into a loud refusal.
//!
//! Contract under test (`zensim_validate::block_profile::folded_root_conflict`):
//! - a bake with structural use of f156-371 ⇒ `Some(reason)` (refuse);
//! - an f0-155-only user (ADD156's shape) ⇒ `None` (safe — the bridge cell
//!   T.R4 measured ≤0.0008 cross-root drift for exactly this shape);
//! - a genuine folded-regime bake (944-wide, block exact-zero — what the
//!   944 trainers emit, 216/216) ⇒ `None`.

use zenpredict::{Activation, Model, WeightDtype};
use zenpredict_bake::{BakeLayer, BakeRequest, bake};
use zensim_validate::block_profile::folded_root_conflict;

const OUT: usize = 2;
const W1: [f32; OUT] = [0.7, -0.3];
const B1: [f32; 1] = [0.1];

/// Single hidden layer + identity head over `n_raw` caller lines; layer-0
/// weight on caller line `k` iff `live(k)`.
fn build_model(n_raw: usize, live: impl Fn(usize) -> bool) -> Model {
    let mut w0 = vec![0.0f32; n_raw * OUT];
    for k in 0..n_raw {
        if live(k) {
            for o in 0..OUT {
                w0[k * OUT + o] = 0.05 + ((k * 31 + o * 7) % 97) as f32 / 100.0;
            }
        }
    }
    let b0 = vec![0.0f32; OUT];
    let mean = vec![0.0f32; n_raw];
    let scale = vec![1.0f32; n_raw];
    let layers = [
        BakeLayer {
            in_dim: n_raw,
            out_dim: OUT,
            activation: Activation::LeakyRelu,
            dtype: WeightDtype::F32,
            weights: &w0,
            biases: &b0,
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
    let bytes = bake(&BakeRequest {
        schema_hash: 0x5eed_0000_c105_0000,
        flags: 0,
        scaler_mean: &mean,
        scaler_scale: &scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: &[],
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: false,
        hu_permutations: None,
    })
    .expect("bake fixture");
    Model::from_bytes(&bytes).expect("parse fixture")
}

#[test]
fn a_372_bake_using_the_zeroed_block_is_refused() {
    // Shipped-B shape: 372-wide, structural weight inside f156-371.
    let m = build_model(372, |k| k < 30 || (200..220).contains(&k));
    let verdict = folded_root_conflict(&m).expect("well-formed fixture");
    let why = verdict.expect("must conflict: 20 used lines in f156-371");
    assert!(
        why.contains("20 caller line(s)"),
        "reason names the used count: {why}"
    );
    assert!(
        why.contains("STRUCTURAL ZEROS"),
        "reason names the mechanism: {why}"
    );
}

#[test]
fn an_f0_155_only_bake_is_safe() {
    // ADD156's shape: 372-wide, weight only below f156. The folded root's
    // zeros never reach a used line, so the cross-root read is the bridge
    // case T.R4 verified — no refusal.
    let m = build_model(372, |k| k < 156 && k % 5 == 0);
    assert_eq!(folded_root_conflict(&m).expect("well-formed"), None);
}

#[test]
fn a_genuine_folded_944_bake_is_safe() {
    // What the 944 trainers emit: full width, f156-371 exactly zero
    // (216/216), everything else live.
    let m = build_model(944, |k| !(156..372).contains(&k));
    assert_eq!(folded_root_conflict(&m).expect("well-formed"), None);
}

#[test]
fn a_single_used_line_in_the_block_is_enough_to_refuse() {
    // The guard is structural, not proportional — one live line inside the
    // zeroed block already makes the folded read a different function.
    let m = build_model(372, |k| k < 10 || k == 300);
    let why = folded_root_conflict(&m)
        .expect("well-formed")
        .expect("one used line conflicts");
    assert!(why.contains("1 caller line(s)"), "{why}");
}
