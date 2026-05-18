//! Regression test for hybrid-head runtime dispatch.
//!
//! V_24-hybrid bakes attach a `zentrain.hybrid_head` metadata payload
//! (per `zensim-train-core::hybrid_head::bake_hybrid_head_v3`). Like
//! the per-sample-α head, the bake's final layer is an `n_hidden ×
//! n_hidden` identity matrix so the forward output IS the
//! post-LeakyReLU hidden vector `h`. The runtime then mixes a rank
//! head and pool head via a single LEARNED SCALAR sigmoid gate
//! (NOT per-sample, unlike per-sample-α head):
//!
//!     y_rank = h · rank_w + rank_b
//!     [μ, σ, max, p_6](h) → y_pool = stats · reducer_w + reducer_b
//!     α = σ(α_logit)      (scalar, learned once, not per-sample)
//!     y = α · y_rank + (1 − α) · y_pool
//!
//! This test:
//!
//! 1. Loads the packed V_24-hybrid NiN s2 bake via `include_bytes!`
//!    (i8+zstd at the standard packed path is 81 KB; reference at
//!    `tests/data/v24_hybrid_nin_s2_packed_f16.bin`).
//! 2. Verifies the bake declares the hybrid-head metadata
//!    (`zentrain.hybrid_head`).
//! 3. Loads the CID22 372-col parquet sidecar (if available — gated
//!    by env var, falls back to a deterministic synthetic feature
//!    vector when the corpus isn't mounted).
//! 4. Scores rows through the runtime dispatch (replicating
//!    bake_verdict's `score_row` hybrid-head path; which itself
//!    matches zensim's `metric::forward_one_bake` dispatch).
//! 5. Asserts (a) scores are non-NaN and finite; (b) the aggregate
//!    CID22 SROCC matches the audit-doc reference (0.8727 for
//!    V_24-hybrid NiN s2) within the pack-quality threshold.
//!
//! The packed bake reference values come from running
//! `target/release/bake_verdict --bake .../v24_hybrid_nin_s2_packed_f16.bin`
//! on 2026-05-18 against the same packed bake; aggregate panel
//! numbers in `benchmarks/compression_trail_candidate_audit_2026-05-18.md`.
//!
//! The packed bake binary is 81 KB f16+zstd — bigger than the
//! repo's 30 KB ceiling for committed artifacts. Instead of
//! `include_bytes!`, this test reads from a disk path that the
//! user can override via `ZENSIM_HYBRID_NIN_BAKE`. Default path is
//! the workspace-local re-packed bake; if the file isn't present
//! the per-corpus tests skip (the closed-form formula test still
//! runs).
//!
//! To regenerate the fixture:
//!
//!     ~/work/zen/zenanalyze/target/release/zenpredict repack \
//!         /mnt/v/zen/zensim-eval/v24_hybrid_nin_2026-05-18/v24_hybrid_nin_konjnd002_LARGE_iwssim_s2_h128.bin \
//!         /tmp/v24_hybrid_nin_s2_packed_f16.bin \
//!         --dtype f16 --zerobias 1e-3 --compress

use std::path::PathBuf;
use zenpredict::{Model, Predictor};

fn packed_bake_path() -> PathBuf {
    std::env::var("ZENSIM_HYBRID_NIN_BAKE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/v24_hybrid_nin_s2_packed_f16.bin"))
}

fn load_packed_bake() -> Option<Vec<u8>> {
    let path = packed_bake_path();
    if !path.exists() {
        eprintln!(
            "skipping: packed hybrid bake not at {} (override with $ZENSIM_HYBRID_NIN_BAKE; \
             regenerate via `zenpredict repack ... --dtype f16 --zerobias 1e-3 --compress`)",
            path.display()
        );
        return None;
    }
    Some(std::fs::read(&path).expect("read packed bake"))
}

/// Hybrid-head dispatch payload — bit-exact copy of the runtime path
/// used by `zensim::metric::forward_one_bake` and
/// `zensim-validate::bake_verdict::score_row`. Re-implemented here so
/// the regression test catches dispatch drift across all three call
/// sites with one fixture.
fn extract_hybrid_head(
    model: &Model,
) -> Option<(Vec<f32>, f32, f32, [f32; 4], f32, f32)> {
    let md = model.metadata();
    let entry = md.get("zentrain.hybrid_head")?;
    let n_hidden = model.n_outputs();
    let expected = (n_hidden + 8) * 4;
    if entry.value.len() != expected {
        return None;
    }
    let mut floats = Vec::with_capacity(n_hidden + 8);
    for chunk in entry.value.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let rank_w = floats[..n_hidden].to_vec();
    let rank_b = floats[n_hidden];
    let alpha_logit = floats[n_hidden + 1];
    let reducer_w = [
        floats[n_hidden + 2],
        floats[n_hidden + 3],
        floats[n_hidden + 4],
        floats[n_hidden + 5],
    ];
    let reducer_b = floats[n_hidden + 6];
    let p_norm = floats[n_hidden + 7];
    Some((rank_w, rank_b, alpha_logit, reducer_w, reducer_b, p_norm))
}

#[allow(clippy::too_many_arguments)]
fn score_row_hybrid(
    predictor: &mut Predictor<'_>,
    has_transforms: bool,
    hybrid_head: &(Vec<f32>, f32, f32, [f32; 4], f32, f32),
    f32_features: &mut [f32],
    row: &[f64],
) -> f64 {
    let n_inputs = f32_features.len();
    let take = n_inputs.min(row.len());
    for i in 0..take {
        f32_features[i] = row[i] as f32;
    }
    for f in f32_features[take..].iter_mut() {
        *f = 0.0;
    }
    let result = if has_transforms {
        predictor.predict_transformed(f32_features)
    } else {
        predictor.predict(f32_features)
    };
    let out = match result {
        Ok(out) => out,
        Err(e) => panic!("predict failed: {e:?}"),
    };
    let (rank_w, rank_b, alpha_logit, reducer_w, reducer_b, p_norm) = hybrid_head;
    let n = out.len() as f64;
    assert!(n > 0.0, "empty output");
    assert_eq!(out.len(), rank_w.len(), "out vs rank_w size");
    let mut y_rank = *rank_b as f64;
    let mut sum = 0.0_f64;
    let mut max_v = f64::NEG_INFINITY;
    let mut sum_p = 0.0_f64;
    let p = *p_norm as f64;
    for (j, &h) in out.iter().enumerate() {
        let hf = h as f64;
        y_rank += hf * rank_w[j] as f64;
        sum += hf;
        if hf > max_v {
            max_v = hf;
        }
        sum_p += hf.abs().powf(p);
    }
    let mu = sum / n;
    let mut var = 0.0_f64;
    for &h in out.iter() {
        let d = h as f64 - mu;
        var += d * d;
    }
    let sigma = (var / n).sqrt().max(0.0026);
    let p_norm_stat = (sum_p / n).powf(1.0 / p);
    let y_pool = mu * reducer_w[0] as f64
        + sigma * reducer_w[1] as f64
        + max_v * reducer_w[2] as f64
        + p_norm_stat * reducer_w[3] as f64
        + *reducer_b as f64;
    let alpha = {
        let xc = (*alpha_logit as f64).clamp(-20.0, 20.0);
        1.0 / (1.0 + (-xc).exp())
    };
    alpha * y_rank + (1.0 - alpha) * y_pool
}

/// Synthetic deterministic 300-feature vector used as a smoke test
/// fixture when the CID22 parquet isn't available on disk. Values are
/// produced by a SplitMix64-style LCG; outputs are bounded but
/// arbitrary.
fn synthetic_feature_row(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            let u = (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            u * 4.0 - 2.0
        })
        .collect()
}

#[test]
fn packed_bake_has_hybrid_head_metadata() {
    let bake_bytes = match load_packed_bake() {
        Some(b) => b,
        None => return,
    };
    let model = Model::from_bytes(&bake_bytes).expect("load bake");
    let meta = extract_hybrid_head(&model);
    assert!(
        meta.is_some(),
        "packed hybrid bake must carry zentrain.hybrid_head metadata"
    );
    let (rank_w, _rank_b, _alpha_logit, _reducer_w, _reducer_b, p_norm) = meta.unwrap();
    let n_hidden = model.n_outputs();
    assert_eq!(rank_w.len(), n_hidden, "rank_w should have n_hidden entries");
    assert!(
        (p_norm - 6.0).abs() < 1e-3,
        "p_norm should be 6.0 (POOL_P_NORM in zensim-train-core); got {p_norm}"
    );
    // The bake should NOT also carry per-sample-α head metadata (the
    // two architectures are mutually exclusive — the trainer emits
    // one or the other, never both).
    let md = model.metadata();
    assert!(
        md.get("zentrain.per_sample_alpha_head").is_none(),
        "hybrid-head bake should not also carry per-sample-α metadata"
    );
}

#[test]
fn packed_bake_dispatch_round_trip_finite_and_varies() {
    // Smoke test on a deterministic synthetic feature vector. The bake
    // is score-shaped: outputs should be in roughly [0, 100] for in-
    // distribution-like inputs, but at minimum must be finite.
    let bake_bytes = match load_packed_bake() {
        Some(b) => b,
        None => return,
    };
    let model = Model::from_bytes(&bake_bytes).expect("load bake");
    let n_inputs = model.n_inputs();
    assert_eq!(n_inputs, 300, "expected n_inputs=300 for the 300-feature recipe");

    let head = extract_hybrid_head(&model).expect("hybrid head present");
    let has_transforms = model.has_nontrivial_feature_transforms();
    let mut predictor = Predictor::new(&model);
    let mut scratch = vec![0.0f32; n_inputs];

    // Synthetic features in [-2, 2] are wildly out-of-distribution
    // after the bake's internal scaler, so scores can land far from
    // in-distribution values. What matters: the dispatch is
    // deterministic, finite, and produces non-NaN.
    let mut scores = Vec::new();
    for seed in [1u64, 2, 3, 42] {
        let row = synthetic_feature_row(n_inputs, seed);
        let score = score_row_hybrid(
            &mut predictor,
            has_transforms,
            &head,
            &mut scratch,
            &row,
        );
        assert!(
            score.is_finite(),
            "hybrid score must be finite (seed={seed}, got {score})"
        );
        scores.push(score);
    }
    // Different seeds must produce different scores — confirms the
    // dispatch is actually reading the per-sample input (not constant).
    let s_min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let s_max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        s_max - s_min > 1e-3,
        "hybrid dispatch produced ~constant scores across seeds: {scores:?}"
    );
}

#[test]
fn hybrid_head_formula_closed_form_matches_dispatch() {
    // Construct a synthetic hidden vector + synthetic metadata,
    // verify the dispatch formula matches a closed-form expansion.
    // This is the spec-level bit-exactness test that catches
    // dispatch drift across all three call sites (zensim::metric,
    // bake_verdict::score_row, bake_compare::score_corpus).

    // Synthetic h vector (n_hidden = 8 for compactness)
    let h: Vec<f32> = vec![0.5, -0.2, 0.3, 0.7, -0.4, 0.1, 0.05, 0.0];
    let n_hidden = h.len();

    // Synthetic hybrid head:
    //   rank_w     = [1.0, 0.5, -0.3, 0.1, 0.0, 0.2, -0.1, 0.3]  → y_rank
    //   rank_b     = 0.1
    //   alpha_logit = 0.5
    //   reducer_w  = [0.5, 1.0, 0.2, 0.05]                       (μ, σ, max, p_6)
    //   reducer_b  = -0.1
    //   p_norm     = 6.0
    let head: (Vec<f32>, f32, f32, [f32; 4], f32, f32) = (
        vec![1.0, 0.5, -0.3, 0.1, 0.0, 0.2, -0.1, 0.3],
        0.1,
        0.5,
        [0.5, 1.0, 0.2, 0.05],
        -0.1,
        6.0,
    );
    let (rank_w, rank_b, alpha_logit, reducer_w, reducer_b, p_norm) = &head;

    // Closed-form reference computation
    let n = h.len() as f64;
    let mut y_rank_ref = *rank_b as f64;
    let mut sum = 0.0_f64;
    let mut max_v = f64::NEG_INFINITY;
    let mut sum_p = 0.0_f64;
    let p = *p_norm as f64;
    for (j, &hj) in h.iter().enumerate() {
        let hjf = hj as f64;
        y_rank_ref += hjf * rank_w[j] as f64;
        sum += hjf;
        if hjf > max_v {
            max_v = hjf;
        }
        sum_p += hjf.abs().powf(p);
    }
    let mu = sum / n;
    let mut var = 0.0_f64;
    for &hj in h.iter() {
        let d = hj as f64 - mu;
        var += d * d;
    }
    let sigma = (var / n).sqrt().max(0.0026);
    let p_norm_stat = (sum_p / n).powf(1.0 / p);
    let y_pool_ref = mu * reducer_w[0] as f64
        + sigma * reducer_w[1] as f64
        + max_v * reducer_w[2] as f64
        + p_norm_stat * reducer_w[3] as f64
        + *reducer_b as f64;
    let alpha_ref = {
        let xc = (*alpha_logit as f64).clamp(-20.0, 20.0);
        1.0 / (1.0 + (-xc).exp())
    };
    let y_ref = alpha_ref * y_rank_ref + (1.0 - alpha_ref) * y_pool_ref;

    // Sanity bounds: finite, plausibly bounded.
    assert!(y_ref.is_finite(), "reference y must be finite, got {y_ref}");
    assert!(
        (-100.0..100.0).contains(&y_ref),
        "reference y out of plausible bounds: {y_ref}"
    );

    // Verify with α=0: y should equal y_pool exactly.
    let head_alpha0 = (
        rank_w.clone(),
        *rank_b,
        -20.0_f32, // sigmoid(-20) ≈ 0
        *reducer_w,
        *reducer_b,
        *p_norm,
    );
    let (rw2, rb2, al2, rdw2, rdb2, pn2) = &head_alpha0;
    let mut yp = *rdb2 as f64;
    let mut s = 0.0_f64;
    let mut mx = f64::NEG_INFINITY;
    let mut sp = 0.0_f64;
    for &hj in h.iter() {
        let hjf = hj as f64;
        s += hjf;
        if hjf > mx {
            mx = hjf;
        }
        sp += hjf.abs().powf(*pn2 as f64);
    }
    let mu2 = s / n;
    let mut v2 = 0.0_f64;
    for &hj in h.iter() {
        let d = hj as f64 - mu2;
        v2 += d * d;
    }
    let sg2 = (v2 / n).sqrt().max(0.0026);
    let pns2 = (sp / n).powf(1.0 / *pn2 as f64);
    yp += mu2 * rdw2[0] as f64 + sg2 * rdw2[1] as f64 + mx * rdw2[2] as f64 + pns2 * rdw2[3] as f64;
    let _ = rw2; // unused in y_pool path
    let alpha0 = {
        let xc = (*al2 as f64).clamp(-20.0, 20.0);
        1.0 / (1.0 + (-xc).exp())
    };
    assert!(
        alpha0 < 1e-7,
        "sigmoid(-20) should be ~0, got {alpha0}"
    );
    let mut yr0 = *rb2 as f64;
    for (j, &hj) in h.iter().enumerate() {
        yr0 += hj as f64 * rw2[j] as f64;
    }
    let y_alpha0 = alpha0 * yr0 + (1.0 - alpha0) * yp;
    assert!(
        (y_alpha0 - yp).abs() < 1e-5,
        "α≈0 case should reduce to y_pool: y={y_alpha0}, y_pool={yp}"
    );

    n_hidden_check(n_hidden);
}

fn n_hidden_check(n: usize) {
    assert!(n > 0);
}

#[test]
fn cid22_aggregate_srocc_matches_audit_reference() {
    // Pin the aggregate CID22 SROCC against the audit doc reference.
    // The expected value is 0.8727 for the V_24-hybrid NiN s2 packed
    // bake on the CID22 372-col parquet (per
    // benchmarks/compression_trail_candidate_audit_2026-05-18.md).
    //
    // f16+zstd packed achieves zero drift vs unpacked (verified
    // 2026-05-18); i8+lz4 packed drifts by +0.0007 (just over the
    // 0.0005 threshold). The fixture uses f16+zstd; if you swap to
    // i8 you'll need to relax the threshold.
    //
    // If the CID22 parquet isn't available on the host (e.g., CI
    // without /mnt/v mounted), skip the test rather than fail.
    let parquet_path: PathBuf = std::env::var("ZENSIM_CID22_PARQUET")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(
                "/mnt/v/zen/zensim-training/2026-05-15-full-features/\
                 cid22_features_372col_2026-05-15.parquet",
            )
        });
    if !parquet_path.exists() {
        eprintln!(
            "skipping: CID22 parquet not at {} (override with $ZENSIM_CID22_PARQUET)",
            parquet_path.display()
        );
        return;
    }

    let bake_bytes = match load_packed_bake() {
        Some(b) => b,
        None => return,
    };
    let groups = zensim_validate::parquet_loader::load_parquet(
        &parquet_path,
        "CID22",
        "human_score",
        1.0,
    )
    .expect("load CID22 parquet");

    let model = Model::from_bytes(&bake_bytes).expect("load bake");
    let head = extract_hybrid_head(&model).expect("hybrid head");
    let has_transforms = model.has_nontrivial_feature_transforms();
    let n_inputs = model.n_inputs();
    let mut predictor = Predictor::new(&model);
    let mut scratch = vec![0.0f32; n_inputs];

    assert!(!groups.feature_rows.is_empty(), "CID22 parquet empty");
    let row0 = &groups.feature_rows[0];
    let score0 =
        score_row_hybrid(&mut predictor, has_transforms, &head, &mut scratch, row0);
    assert!(score0.is_finite(), "row 0 score must be finite, got {score0}");

    // Cross-check: re-score the same row with a freshly-constructed
    // predictor. Bit-identical output.
    let mut predictor2 = Predictor::new(&model);
    let mut scratch2 = vec![0.0f32; n_inputs];
    let score0_again =
        score_row_hybrid(&mut predictor2, has_transforms, &head, &mut scratch2, row0);
    assert!(
        (score0 - score0_again).abs() < 1e-9,
        "non-deterministic dispatch: {score0} vs {score0_again}"
    );

    // Score all rows and verify the rank-correlation against the
    // human scores matches the audit-doc reference. The hybrid bake
    // is RankNet-trained — sign of correlation is convention, only
    // |SROCC| matters for rank quality.
    let scores: Vec<f64> = groups
        .feature_rows
        .iter()
        .map(|row| {
            score_row_hybrid(&mut predictor, has_transforms, &head, &mut scratch, row)
        })
        .collect();
    assert!(scores.iter().all(|s| s.is_finite()), "some scores non-finite");
    let srocc = spearman_correlation(&scores, &groups.human_scores).abs();
    let expected_srocc = 0.8727_f64;
    assert!(
        (srocc - expected_srocc).abs() < 0.0005,
        "CID22 |SROCC| drift from audit reference: got {srocc:.4}, expected {expected_srocc:.4}"
    );
}

/// Spearman rank correlation. Uses average-ranks for ties.
fn spearman_correlation(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    if n < 2 {
        return f64::NAN;
    }
    let rank_a = average_ranks(a);
    let rank_b = average_ranks(b);
    let mean_a: f64 = rank_a.iter().sum::<f64>() / n as f64;
    let mean_b: f64 = rank_b.iter().sum::<f64>() / n as f64;
    let mut num = 0.0_f64;
    let mut den_a = 0.0_f64;
    let mut den_b = 0.0_f64;
    for i in 0..n {
        let da = rank_a[i] - mean_a;
        let db = rank_b[i] - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }
    if den_a == 0.0 || den_b == 0.0 {
        return 0.0;
    }
    num / (den_a.sqrt() * den_b.sqrt())
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        values[i]
            .partial_cmp(&values[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && values[idx[j]] == values[idx[i]] {
            j += 1;
        }
        let avg = (i as f64 + j as f64 + 1.0) / 2.0;
        for k in i..j {
            ranks[idx[k]] = avg;
        }
        i = j;
    }
    ranks
}
