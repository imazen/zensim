//! End-to-end tier parity: a tiny MLP training run must produce
//! **byte-identical** serialized weights and **identical epoch log lines**
//! under every SIMD token permutation the host offers (AVX-512 → AVX2 →
//! scalar dispatch). Negative control: on the pre-fix tree the AVX-512
//! and scalar tiers take different arithmetic paths and this test fails.
//!
//! The permutations exercise the REAL dispatch paths — `simd_mlp`,
//! `adam_simd`, and `simd_encoder` all resolve tokens through
//! `X64V*Token::summon()`/`incant!`, which the permutation guard disables
//! in turn. (archmage's process-wide token disable is scoped to each
//! permutation and restored after.)

#![cfg(target_arch = "x86_64")]

use archmage::SimdToken;
use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zensim_validate::mlp_train::{
    FeatureRows, GroupLossMode, MlpHyperparams, TrainingGroup, train_mlp,
};

/// Tiny deterministic RNG (SplitMix64-free — that type is `pub(crate)`).
struct Xs(u64);
impl Xs {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn normal(&mut self) -> f64 {
        // Sum-of-4-uniforms approx — plenty for training data.
        let mut u = || (self.next_u64() >> 33) as f64 / (u32::MAX as f64) - 0.5;
        u() + u() + u() + u()
    }
}

fn run_tiny_train(per_sample_alpha_head: bool) -> (Vec<u8>, Vec<String>) {
    // n_hidden=20: nh%8=4 deliberately hits the v4-vs-v3 tail boundary
    // (v4 would mul+add [16,20) where v3 fuses them), so the plain path
    // gets a three-way v4/v3/scalar divergence pre-fix, not just the
    // scalar-vs-fused one that f32 quantization can mask at 16×8.
    let n_features = 24usize;
    let n_rows = 240usize;
    let mut rng = Xs::new(0x7E57_C0DE);
    let true_w: Vec<f64> = (0..n_features)
        .map(|i| (i as f64 - 8.0) * 0.25 + rng.normal() * 0.1)
        .collect();
    let mut features_owned: Vec<Vec<f64>> = Vec::with_capacity(n_rows);
    let mut targets: Vec<f64> = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let x: Vec<f64> = (0..n_features).map(|_| rng.normal()).collect();
        let mut y: f64 = x.iter().zip(true_w.iter()).map(|(a, b)| a * b).sum();
        y += 0.1 * x[0] * x[0];
        y += rng.normal() * 0.05;
        features_owned.push(x);
        targets.push(y);
    }
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

    let group = TrainingGroup {
        name: "tierparity".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };

    let hyper = MlpHyperparams {
        n_hidden: 20,
        n_epochs: 4,
        pairs_per_epoch: 600,
        initial_lr: 0.005,
        log_every: 100,
        early_stop_patience: 0,
        per_sample_alpha_head,
        ..Default::default()
    };
    let mut log = Vec::new();
    let bytes = train_mlp(&mut [group], n_features, &hyper, &mut log);
    (bytes, log)
}

/// Epoch lines end in a `| t=<secs>` wall-clock field. Timing is the one
/// thing tiers are *expected* to change — a scalar fallback legitimately
/// prints `t=0.1s` where AVX-512 prints `t=0.0s`. Strip it so the numeric
/// columns (loss, val, srocc, plcc, pwrc) are what get compared.
fn strip_time(line: &str) -> &str {
    match line.rfind(" | t=") {
        Some(i) => &line[..i],
        None => line,
    }
}

/// Core assertion: run the full trainer once per permutation and compare
/// `(weight bytes, epoch log lines)` against the first permutation's
/// baseline — any SIMD-tier arithmetic divergence shows up as a diff.
#[test]
fn train_bit_identical_across_all_host_tiers() {
    if archmage::X64V3Token::summon().is_none() {
        // No AVX2 on this host — there is no canonical tier to pin to;
        // tier parity is still checked for whatever tiers exist via the
        // same baseline-compare below.
        eprintln!("no v3 token; comparing whatever tiers are present");
    }
    for alpha_head in [false, true] {
        let mut baseline: Option<(Vec<u8>, Vec<String>)> = None;
        let mut failures = Vec::new();
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let cur = run_tiny_train(alpha_head);
            eprintln!(
                "perm [{}] bytes_sha={:016x} loglines={}",
                perm.label,
                {
                    let mut h = 0xcbf29ce484222325u64;
                    for &b in &cur.0 {
                        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
                    }
                    h
                },
                cur.1.len()
            );
            if let Some(b) = &baseline {
                if cur.0 != b.0 {
                    failures.push(format!(
                        "alpha_head={alpha_head} WEIGHT BYTES diverged under {}",
                        perm.label
                    ));
                }
                let lines_eq = cur.1.len() == b.1.len()
                    && cur
                        .1
                        .iter()
                        .zip(b.1.iter())
                        .all(|(a, b)| strip_time(a) == strip_time(b));
                if !lines_eq {
                    let diffs: Vec<String> = cur
                        .1
                        .iter()
                        .zip(b.1.iter())
                        .enumerate()
                        .filter(|(_, (a, b))| strip_time(a) != strip_time(b))
                        .map(|(i, (a, b))| format!("  line{i}: perm=`{a}` vs base=`{b}`"))
                        .collect();
                    failures.push(format!(
                        "alpha_head={alpha_head} EPOCH LINES diverged under {}\n{}",
                        perm.label,
                        diffs.join("\n")
                    ));
                }
            } else {
                baseline = Some(cur);
            }
        });
        eprintln!(
            "alpha_head={alpha_head}: permutations run: {}",
            report.permutations_run
        );
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
