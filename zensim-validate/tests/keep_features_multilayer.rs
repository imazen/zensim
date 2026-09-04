//! End-to-end coverage for the 2026-09-04 owner-fix
//! (`benchmarks/fastclass_distill_wave_2026-09-04.md` §7.6/§7.7): before
//! it, `train_mlp_per_sample_alpha_head` never read `INPUT_KEEP_MASK`, so
//! `--keep-features` on that path (the ONLY owner of two-or-more-hidden-
//! layer training via `--n-hidden-layers`) shipped a bake whose "dropped"
//! columns still carried live, randomly-initialized layer-1 weights —
//! trained on data that happened to be zeroed (so the loss never
//! noticed), but wrong for any caller that supplies a real value there.
//! `zensim_mlp_train.rs`'s CLI guard refused the combination outright
//! rather than ship that silently-wrong bake (see
//! `feature_subset_guard_tests` in that binary for the guard's own
//! failing-first coverage).
//!
//! Uses `INPUT_KEEP_MASK` (the same process global `--keep-features`
//! drives at the CLI layer) rather than calling `zero_masked_w1_rows`
//! directly, so this exercises the actual wiring inside
//! `train_mlp_per_sample_alpha_head`/`train_mlp_strategy`, not just the
//! already-covered primitive
//! (`zensim_validate::mlp_train::group_l1_tests::keep_mask_pins_only_dropped_rows`,
//! an internal `#[cfg(test)]` unit test).
//!
//! **Why this lives in `tests/` and not `src/mlp_train/mod.rs`'s
//! `#[cfg(test)] mod tests`**: it was originally a `src/`-internal unit
//! test module, and it broke two UNRELATED pre-existing tests
//! (`train_mlp_pwrc_disabled_matches_legacy`,
//! `strategy_smoke_all_active_end_to_end`) the first time it ran inside
//! the same test binary — `cargo test`'s default thread-pool parallelism
//! let this suite's `INPUT_KEEP_MASK` mutation race those tests' own
//! `train_mlp`/`train_mlp_strategy` calls (confirmed from the failure
//! logs: their SECOND of two "must reproduce identically" training calls
//! picked up a stray `[keep-features] pinned 3 of N ...` log line the
//! first call never got). `INPUT_KEEP_MASK` is a `static Mutex` inside
//! the library, so every `#[test]` fn in the unit-test binary shares ONE
//! copy of it with zero synchronization by default — safe as long as
//! nothing ever writes `Some(_)` to it, which is exactly what changed
//! here. Each file under `tests/` compiles to its OWN separate process,
//! so it links its own copy of every `static` in the crate: moving these
//! tests here makes the mutation invisible to the unit-test binary (and
//! to any other integration-test binary) without touching a single
//! pre-existing test or adding a new dependency. Within THIS binary,
//! `KEEP_FEATURES_MASK_TEST_LOCK` still serializes the three tests below
//! against each other (nothing else in this file touches the mask).

use zenpredict::{FeatureTransform, Model, WeightStorage};
use zensim_validate::mlp_train::{
    FeatureRows, GroupLossMode, INPUT_KEEP_MASK, MlpHyperparams, TrainingGroup, ValidationPolicy,
    train_mlp_strategy,
};
use zensim_validate::prune;

/// Serializes this file's three tests against each other (they are the
/// only tests that set `INPUT_KEEP_MASK`, and process isolation from
/// every OTHER test binary is what the module doc comment above is
/// about); see there for the full story.
static KEEP_FEATURES_MASK_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

const N_FEATURES: usize = 6;
// Keep {0, 2, 5}; drop {1, 3, 4}.
const KEEP: [bool; N_FEATURES] = [true, false, true, false, false, true];

/// SplitMix64 — the trainer's own PRNG (`zensim_validate::mlp_train`'s
/// private `SplitMix64`) reimplemented here verbatim. It is `pub(crate)`
/// in the library, so an external integration test cannot import it;
/// duplicating ~15 lines of a well-known, fixed PRNG algorithm is cheaper
/// and lower-risk than widening that type's visibility just for a test
/// fixture. This is a source of pseudo-random synthetic data ONLY —
/// nothing here depends on it matching the trainer's own draws.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
    }

    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64_unit().max(1e-12);
        let u2 = self.next_f64_unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn synthetic_group(n_rows: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = SplitMix64::new(seed);
    let mut feats: Vec<Vec<f64>> = Vec::with_capacity(n_rows);
    let mut scores: Vec<f64> = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let x: Vec<f64> = (0..N_FEATURES).map(|_| rng.next_normal()).collect();
        // Target correlates with the KEPT features only (0, 2, 5) —
        // isolates "does keep_features apply the mask" from "does the
        // net learn on its own to ignore uninformative features".
        let y = x[0] * 0.6 - x[2] * 0.4 + x[5] * 0.2 + rng.next_normal() * 0.05;
        feats.push(x);
        scores.push(y);
    }
    (feats, scores)
}

/// Train either the plain path (`per_sample_alpha_head = false,
/// n_hidden_layers = 1`) or the per-sample-α head at the requested
/// architecture, under whatever `INPUT_KEEP_MASK` the caller has set.
///
/// `raw_zero_mask`, when set, zeroes the raw value of every `false`
/// column in a COPY of `feats` before training — mirroring what
/// `zensim_mlp_train.rs`'s `--keep-features` handling does to `loaded`
/// BEFORE the scaler is computed (see its "FEATURE-SUBSET ablation"
/// comment). This is NOT optional when `INPUT_KEEP_MASK` is also set:
/// `zero_masked_w1_rows` only keeps a row at exact zero for the rest of
/// training because the CLI guarantees the matching raw column is ALSO
/// exactly zero (so the gradient into that row is exactly zero every
/// step); pinning the layer-1 row alone, with the raw column still
/// carrying real signal, lets Adam drift it away from zero during
/// training — a first version of this test skipped the raw zeroing and
/// the row visibly drifted (`dropped_plain` came back `[]` instead of
/// the masked indices).
fn train(
    feats: &[Vec<f64>],
    scores: &[f64],
    per_sample_alpha_head: bool,
    n_hidden_layers: usize,
    raw_zero_mask: Option<&[bool; N_FEATURES]>,
) -> Vec<u8> {
    let mut feats_owned;
    let feats: &[Vec<f64>] = match raw_zero_mask {
        None => feats,
        Some(mask) => {
            feats_owned = feats.to_vec();
            for row in &mut feats_owned {
                for (d, keep) in mask.iter().enumerate() {
                    if !keep {
                        row[d] = 0.0;
                    }
                }
            }
            &feats_owned
        }
    };
    let feats_ref: Vec<&[f64]> = feats.iter().map(|v| v.as_slice()).collect();
    let mut groups = [TrainingGroup {
        name: "synth".to_string(),
        human_scores: scores,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    }];
    let hp = MlpHyperparams {
        n_hidden: 8,
        n_hidden_layers,
        n_epochs: 3,
        pairs_per_epoch: 50,
        per_sample_alpha_head,
        tanh_output_head_scale: if per_sample_alpha_head { 20.0 } else { 0.0 },
        seed: 5,
        log_every: 1000,
        early_stop_patience: 0,
        validation_policy: ValidationPolicy::Mean,
        ..Default::default()
    };
    let mut log = Vec::new();
    train_mlp_strategy(
        &mut groups,
        N_FEATURES,
        &hp,
        &mut log,
        None,
        None,
        None,
        None,
        None,
        None,
    )
}

/// Load a bake and return (a) its caller-facing width both ways, and (b)
/// the sorted set of raw input indices `zensim_validate::prune` would
/// drop as class-1 weight-dead — i.e. exactly the layer-1 rows that came
/// out of training as bit-exact zero.
fn caller_width_and_dropped_raw_indices(bytes: &[u8]) -> (usize, usize, Vec<usize>) {
    let model = Model::from_bytes(bytes).expect("bake parses via zenpredict");
    let layer0 = model.layer(0);
    assert_eq!(
        layer0.in_dim, N_FEATURES,
        "layer-0 in_dim must be the raw feature width, unaffected by --keep-features"
    );
    let weights = match layer0.weights {
        WeightStorage::F32(w) => w,
        other => panic!("expected F32 layer-0 weights in this test's bakes, got {other:?}"),
    };
    let l0 = prune::Layer0View {
        in_dim: layer0.in_dim,
        out_dim: layer0.out_dim,
        weights,
        biases: layer0.biases,
        is_i8: false,
    };
    // `prune_constants = false`: this test's bakes carry no
    // feature-transform metadata, so only class 1 (weight-dead) is
    // reachable — restricting to it keeps the assertion scoped to
    // exactly what `--keep-features` produces (bit-exact zero rows), not
    // incidentally-forced-constant columns.
    let plan = prune::plan(&model, &l0, false).expect("prune plan builds");
    let mut dropped_raw: Vec<usize> = plan.drop.iter().map(|&(_, raw, _)| raw).collect();
    dropped_raw.sort_unstable();
    // FeatureTransform::Drop semantics: plan() marks every dropped raw
    // line's post-prune transform as Drop, and nothing else.
    for (i, t) in plan.transforms.iter().enumerate() {
        let is_dropped = dropped_raw.contains(&i);
        assert_eq!(
            *t == FeatureTransform::Drop,
            is_dropped,
            "raw input {i}: FeatureTransform::Drop must be set iff prune dropped it"
        );
    }
    (model.caller_input_width(), model.n_inputs(), dropped_raw)
}

#[test]
fn keep_features_masks_identical_rows_on_plain_and_2layer_alpha_head_paths() {
    let _guard = KEEP_FEATURES_MASK_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (feats, scores) = synthetic_group(80, 11);

    *INPUT_KEEP_MASK.lock().unwrap() = Some(KEEP.to_vec());
    let bake_plain = train(&feats, &scores, false, 1, Some(&KEEP));
    let bake_2layer = train(&feats, &scores, true, 2, Some(&KEEP));
    *INPUT_KEEP_MASK.lock().unwrap() = None;

    let expected_dropped: Vec<usize> = (0..N_FEATURES).filter(|&i| !KEEP[i]).collect();

    let (raw_width_plain, n_inputs_plain, dropped_plain) =
        caller_width_and_dropped_raw_indices(&bake_plain);
    let (raw_width_2layer, n_inputs_2layer, dropped_2layer) =
        caller_width_and_dropped_raw_indices(&bake_2layer);

    // caller_input_width() / n_inputs() are unaffected by --keep-features
    // (masking zeroes rows; it does not narrow the declared width) and
    // must match between the plain and multi-layer architectures.
    assert_eq!(raw_width_plain, N_FEATURES);
    assert_eq!(n_inputs_plain, N_FEATURES);
    assert_eq!(raw_width_2layer, N_FEATURES);
    assert_eq!(n_inputs_2layer, N_FEATURES);

    assert_eq!(
        dropped_plain, expected_dropped,
        "plain path did not drop exactly the requested columns"
    );
    assert_eq!(
        dropped_2layer, expected_dropped,
        "per_sample_alpha_head 2-layer path did not drop exactly the requested columns \
         (pre-fix: train_mlp_per_sample_alpha_head never applied the mask, so this would be \
         empty or contain live near-zero-but-nonzero rows instead of bit-exact zeros)"
    );
    // The literal "bit-identical layer-1 input selection" claim: same
    // mask, same raw indices dropped, regardless of architecture.
    assert_eq!(dropped_plain, dropped_2layer);
}

#[test]
fn keep_features_masks_identical_rows_on_plain_and_1layer_alpha_head_paths() {
    // Same claim at n_hidden_layers = 1: per_sample_alpha_head's w1 is
    // n_features x n_hidden regardless of use_2layer, so the fix covers
    // this architecture too (see keep_features_unsupported_flag in
    // zensim_mlp_train.rs, which now allows both).
    let _guard = KEEP_FEATURES_MASK_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (feats, scores) = synthetic_group(80, 12);

    *INPUT_KEEP_MASK.lock().unwrap() = Some(KEEP.to_vec());
    let bake_plain = train(&feats, &scores, false, 1, Some(&KEEP));
    let bake_1layer_alpha = train(&feats, &scores, true, 1, Some(&KEEP));
    *INPUT_KEEP_MASK.lock().unwrap() = None;

    let expected_dropped: Vec<usize> = (0..N_FEATURES).filter(|&i| !KEEP[i]).collect();
    let (_, _, dropped_plain) = caller_width_and_dropped_raw_indices(&bake_plain);
    let (_, _, dropped_1layer_alpha) = caller_width_and_dropped_raw_indices(&bake_1layer_alpha);

    assert_eq!(dropped_plain, expected_dropped);
    assert_eq!(dropped_1layer_alpha, expected_dropped);
    assert_eq!(dropped_plain, dropped_1layer_alpha);
}

/// Default behavior (no `--keep-features`) is untouched: with
/// `INPUT_KEEP_MASK` at its default `None`, the 2-layer per-sample-α path
/// drops nothing — `zero_masked_w1_rows` returns 0 and never touches
/// `w1`, exactly like the plain path's pre-existing call.
#[test]
fn no_mask_set_drops_nothing_on_2layer_alpha_head_path() {
    let _guard = KEEP_FEATURES_MASK_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (feats, scores) = synthetic_group(80, 13);

    assert!(
        INPUT_KEEP_MASK.lock().unwrap().is_none(),
        "test lock should guarantee no mask leaked in from another test"
    );
    let bake_2layer = train(&feats, &scores, true, 2, None);
    let (raw_width, n_inputs, dropped) = caller_width_and_dropped_raw_indices(&bake_2layer);
    assert_eq!(raw_width, N_FEATURES);
    assert_eq!(n_inputs, N_FEATURES);
    assert!(
        dropped.is_empty(),
        "no --keep-features set: no layer-1 row should be pinned to zero, got {dropped:?}"
    );
}
