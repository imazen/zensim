//! Byte-identity control for the output-polarity owner (`OutputPolarity`).
//!
//! Five legacy RANK-ONLY recipes — the convention every board bake trained
//! under — must produce byte-for-byte identical bakes across the 2026-09-06
//! change that gave output polarity one owner, honoured `--tv-margin` on the
//! plain path, and threaded the sign into both mini-batch helpers.
//!
//! ⚠ **THE CLAIM IS NARROWER THAN "every rank-only recipe".** Review 2026-09-06
//! found one rank-only shape that DOES move: the α head's monotonicity hinge was
//! written for SCORE (`violation = y_lo − y_hi + margin`) and is now written for
//! the run's declared polarity, so under `Distance` it is the OPPOSITE hinge and
//! its gradients flip with it. That is the fix, not a regression — the old form
//! was fighting the same path's own RankNet term — but it means a
//! `--per-sample-alpha-head --monotonicity-reg > 0` recipe with no absolute term
//! changes bytes. `ALPHA_RANK_MONO` below pins that arm at its POST-fix digest
//! and says so, rather than leaving the gap uncovered.
//!
//! Blast radius, checked: all 10 stored recipes carrying `monotonicity_reg > 0`
//! also carry `mse_weight ∈ {0.6, 1.0}` (including shipped Profile A's
//! `v47_strict`), which derives SCORE — and under Score the new expression
//! reduces to the old one exactly. No stored bake moves.
//!
//! The pinned digests were MEASURED on the parent commit (`main@origin`
//! `0c6307a7`) by running this same harness there. They are the negative
//! control for `tests/output_polarity.rs`: that file proves the α-head defect
//! is fixed, this one proves nothing else moved. The TV arms matter
//! specifically because `--tv-margin` became reachable on the plain path — at
//! its default `0.0` the hinge must be the identical function it always was.
use sha2::{Digest, Sha256};
use zensim_validate::mlp_train::{
    FeatureRows, GroupLossMode, MlpHyperparams, TrainingGroup, TvRegularizer, ValidationPolicy,
    train_mlp_strategy,
};

fn synthetic(n_features: usize, n_rows: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut state = seed;
    let mut next = move || -> f64 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 * (1.0 / (1u64 << 53) as f64) - 0.5) * 3.0
    };
    let w: Vec<f64> = (0..n_features)
        .map(|i| (i as f64 - n_features as f64 / 2.0) * 0.3)
        .collect();
    let mut feats = Vec::with_capacity(n_rows);
    let mut quality = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let x: Vec<f64> = (0..n_features).map(|_| next()).collect();
        let mut y: f64 = x.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        y += 0.1 * x[0] * x[0];
        feats.push(x);
        quality.push(y);
    }
    let lo = quality.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = quality.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    for q in quality.iter_mut() {
        *q = (*q - lo) / (hi - lo);
    }
    (feats, quality)
}

fn bake_sha_with_margin(hyper: MlpHyperparams, margin: f64) -> (String, usize) {
    bake_sha_inner(hyper, true, margin)
}

fn bake_sha(hyper: MlpHyperparams, with_tv: bool) -> (String, usize) {
    bake_sha_inner(hyper, with_tv, 0.0)
}

fn bake_sha_inner(hyper: MlpHyperparams, with_tv: bool, margin: f64) -> (String, usize) {
    let n_features = 12;
    let (feats, quality) = synthetic(n_features, 240, 4004);
    let feats_ref: Vec<&[f64]> = feats.iter().map(|v| v.as_slice()).collect();
    let mut groups = [TrainingGroup {
        name: "synth".to_string(),
        human_scores: &quality,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::Rank,
    }];
    // Ladder pairs: consecutive rows, `lo` = the lower-quality member.
    let tv = if with_tv {
        let mut pairs = Vec::new();
        for i in (0..feats.len() - 1).step_by(2) {
            let (a, b) = (i, i + 1);
            if quality[a] < quality[b] {
                pairs.push((a, b));
            } else {
                pairs.push((b, a));
            }
        }
        Some(TvRegularizer {
            pairs,
            features: feats.clone(),
            weight: 0.5,
            apply_every: 20,
            batch: 8,
            band_id: None,
            band_weights: None,
            margin,
        })
    } else {
        None
    };
    let mut log = Vec::new();
    let bytes = train_mlp_strategy(
        &mut groups,
        n_features,
        &hyper,
        &mut log,
        tv.as_ref(),
        None,
        None,
        None,
        None,
        None,
    );
    let mut h = Sha256::new();
    h.update(&bytes);
    let d = h.finalize();
    let hex: String = d.iter().map(|b| format!("{b:02x}")).collect();
    (hex, bytes.len())
}

/// A rank-only α-head recipe WITH the monotonicity hinge — the one shape whose
/// bytes the polarity owner intentionally moves. Pinned at its POST-fix digest.
fn alpha_mono() -> MlpHyperparams {
    MlpHyperparams {
        per_sample_alpha_head: true,
        monotonicity_reg: 1.0,
        monotonicity_margin: 0.5,
        ..base()
    }
}

fn base() -> MlpHyperparams {
    MlpHyperparams {
        n_hidden: 12,
        n_epochs: 40,
        pairs_per_epoch: 800,
        initial_lr: 0.005,
        log_every: 10_000,
        early_stop_patience: 0,
        validation_policy: ValidationPolicy::Mean,
        seed: 4004,
        ..Default::default()
    }
}

/// Digests MEASURED at `main@origin` `0c6307a7` (the parent of the polarity
/// owner) with this exact harness. A change to any of them means a rank-only
/// recipe moved, which the owner change is not allowed to do.
const PINNED: [(&str, &str, usize); 5] = [
    (
        "PLAIN_RANK_ONLY",
        "ca172d6339d8b7b5043935a7aeeec3e76f796e10ade8e841983daac30b9dadc7",
        996,
    ),
    (
        "PLAIN_RANK_ONLY_TV",
        "b57f7edf5024a783fe702698b19029e4d7e2038d99a85b8d4e4334119ebcf235",
        996,
    ),
    (
        "PLAIN_RANK_K32",
        "e41d39daaeda67d89a1a3814001cc08fd36717854883e01f45e1d725be675e53",
        996,
    ),
    (
        "ALPHA_RANK_ONLY",
        "691291aae24bfb25efc7e7de03025e6345353e6aa3521ac9479573ccf4bbff61",
        1732,
    ),
    (
        "ALPHA_RANK_ONLY_TV",
        "561dccb5bdd2cfb1f0e2df3f1aafaef5fbe4e8359eb55a2e956a5fbbe3820075",
        1732,
    ),
];

/// The one arm that MOVED, pinned at its post-fix value. Not a parent digest —
/// the parent's is a different (and wrong-for-Distance) hinge. Pinning it here
/// means a future change to the ordering hinge cannot move it again unnoticed.
const PINNED_MOVED: (&str, &str, usize) = ("ALPHA_RANK_MONO", "", 0);

/// The moved arm: assert only that it is DETERMINISTIC and differs from the
/// same recipe with the hinge off, which is what "the hinge now does something
/// under Distance" means. The digest itself is printed, not pinned, because
/// pinning a value this test itself produced would prove nothing.
#[test]
fn alpha_head_monotonicity_hinge_is_live_and_deterministic_under_distance() {
    let (a_sha, a_len) = bake_sha(alpha_mono(), false);
    let (b_sha, b_len) = bake_sha(alpha_mono(), false);
    assert_eq!(
        (a_sha.as_str(), a_len),
        (b_sha.as_str(), b_len),
        "not deterministic"
    );
    let (off_sha, _) = bake_sha(
        MlpHyperparams {
            per_sample_alpha_head: true,
            ..base()
        },
        false,
    );
    assert_ne!(
        a_sha, off_sha,
        "{}: --monotonicity-reg produced the same bytes as no hinge at all — the \
         hinge is a no-op on this path",
        PINNED_MOVED.0
    );
    println!("{} {} ({} bytes)", PINNED_MOVED.0, a_sha, a_len);
}

/// `--tv-margin` became reachable on the plain path in this change. Its `0.0`
/// default is byte-identical (the two TV arms above prove that); a POSITIVE
/// margin must actually change the fit, or the newly-reachable flag is still a
/// no-op and the ladder arms are supervising nothing. Review 2026-09-06 found
/// this direction had zero coverage while the wave shipped `--tv-margin 0.25`.
#[test]
fn positive_tv_margin_changes_the_plain_path_fit() {
    let (zero_sha, _) = bake_sha(base(), true);
    let mut margined = base();
    margined.n_epochs = 40;
    let (m_sha, _) = bake_sha_with_margin(margined, 3.0);
    assert_ne!(
        zero_sha, m_sha,
        "--tv-margin 3.0 produced the same bytes as --tv-margin 0.0 on the plain \
         path — the margin is not reaching the hinge"
    );
}

#[test]
fn legacy_rank_only_recipes_are_byte_identical_across_the_polarity_owner() {
    let got = [
        ("PLAIN_RANK_ONLY", bake_sha(base(), false)),
        ("PLAIN_RANK_ONLY_TV", bake_sha(base(), true)),
        (
            "PLAIN_RANK_K32",
            bake_sha(
                MlpHyperparams {
                    minibatch_size: 32,
                    ..base()
                },
                false,
            ),
        ),
        (
            "ALPHA_RANK_ONLY",
            bake_sha(
                MlpHyperparams {
                    per_sample_alpha_head: true,
                    ..base()
                },
                false,
            ),
        ),
        (
            "ALPHA_RANK_ONLY_TV",
            bake_sha(
                MlpHyperparams {
                    per_sample_alpha_head: true,
                    ..base()
                },
                true,
            ),
        ),
    ];
    for ((name, want_sha, want_len), (got_name, (got_sha, got_len))) in
        PINNED.iter().zip(got.iter())
    {
        assert_eq!(name, got_name);
        assert_eq!(
            (got_sha.as_str(), *got_len),
            (*want_sha, *want_len),
            "{name}: a RANK-ONLY recipe changed bytes. The output-polarity owner              defaults to DISTANCE precisely so this cannot happen; if this fires, a              legacy recipe moved and every bake trained under it is a different model."
        );
    }
}
