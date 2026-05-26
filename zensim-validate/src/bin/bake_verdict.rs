//! Instant V_X bake evaluator — loads pre-extracted features from
//! parquet sidecars, scores a ZNPR v3 bake, and emits the full
//! Mohammadi 2025 panel (aggregate + 10-band) per held-out corpus.
//!
//! Replaces the per-bake compute path in
//! `zensim-bench/examples/dataset_metric_baseline.rs` for the case
//! where image features have already been extracted (T10.1). Old
//! path: re-decode images + recompute baseline metrics + score
//! MLP per pair, ~15-20 min for the full 5-corpus held-out set.
//! New path: read parquet sidecars + MLP forward only, <5 s wall.
//!
//! Inputs (T10.1 outputs):
//!     /mnt/v/zen/zensim-training/2026-05-15-full-features/
//!         aic3_features_372col_2026-05-15.parquet
//!         cid22_features_372col_2026-05-15.parquet
//!         kadid_features_372col_2026-05-15.parquet
//!         konjnd_features_372col_2026-05-15.parquet
//!         tid_features_372col_2026-05-15.parquet
//!
//! Each parquet carries 374 columns: `ref_basename, human_score, f0..f371`.
//! `human_score` is on the corpus's own normalized scale (matches the
//! convention `dataset_metric_baseline.rs` uses internally — KADID
//! `(DMOS-1)/4` in [0,1], TID `MOS/9` in [0,1], CID22 `MCOS/100` in
//! [0,1], KonJND `mean_threshold` in raw units, AIC-3 raw `score.jnd`
//! in [-3,0]). SROCC / KROCC / PWRC are rank-invariant so polarity
//! and scale don't matter; PLCC / Z-RMSE absorb scale via the
//! 4-parameter logistic rescale (Mohammadi 2025 convention).
//!
//! Usage:
//!     bake_verdict --bake <path>
//!                  [--corpora cid22,kadid,tid,konjnd,aic3]
//!                  [--output <path.md>]
//!                  [--features-root /mnt/v/zen/zensim-training/2026-05-15-full-features]
//!
//! Verification: when invoked with the V_22-IW v2 calibrated bake
//! (`zensim/weights/v0_22_iw_v2_calibrated_2026-05-16.bin`), the
//! aggregate SROCC values match the dataset_metric_baseline log at
//! `benchmarks/v0_22_iw_v2_seed1_2026-05-16_eval_full.log` to within
//! 1e-3. The full numbers come from the SAME features that the
//! baseline path computes per pair; the only difference is that we
//! read them from parquet instead of recomputing.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use zenpredict::{Model, Predictor};

use zensim_validate::panel::{compute_panel, rescale_logistic};
use zensim_validate::parquet_loader;

// ============================================================================
// Stat functions live in `zenstats::panel` (re-exported via
// `zensim_validate::panel`). bake_verdict previously carried a
// byte-identical inline copy of ranks/spearman/pearson/kendall_tau/
// outlier_ratio/pwrc/z_rmse/rescale_logistic etc. — that drifted from
// the canonical home (the 2026-05-26 paper-correct OR + PWRC rewrite
// only landed in panel.rs, so this binary's "panel" output was
// silently the older proxy math until this commit). All call sites
// route through `panel::compute_panel` now; only `ds_auc` below
// remains bake_verdict-specific (no panel.rs equivalent yet).
// ============================================================================

/// DS-AUC (G9, Mohammadi 2025 § VII): Area Under the ROC curve for
/// classifying stimulus pairs as "same" vs "different" perceptual quality.
///
/// Without parsed 2AFC response data, we use a practical proxy: a pair
/// (i, j) is labeled "different" when |human[i] − human[j]| exceeds
/// `diff_threshold` (in human-score units), "same" otherwise. The metric's
/// |score[i] − score[j]| is the classifier score. AUC measures how well
/// the metric's score-gap separates the same/different pairs.
///
/// Returns AUC in [0, 1]; 0.5 = chance. Subsamples pairs when n is large
/// to keep the O(n²) pair enumeration tractable.
fn ds_auc(predicted: &[f64], human: &[f64], diff_threshold: f64) -> f64 {
    let n = predicted.len();
    if n < 4 || human.len() != n {
        return f64::NAN;
    }
    // Cap pair count: with n up to ~10k, full O(n²) is 100M pairs.
    // Subsample to ~200k pairs deterministically via stride.
    let max_pairs = 200_000usize;
    let total_pairs = n * (n - 1) / 2;
    let stride = (total_pairs / max_pairs).max(1);

    // Collect (metric_gap, is_different) labels.
    let mut samples: Vec<(f64, bool)> = Vec::new();
    let mut pair_idx = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if pair_idx % stride == 0 {
                let metric_gap = (predicted[i] - predicted[j]).abs();
                let human_gap = (human[i] - human[j]).abs();
                if metric_gap.is_finite() && human_gap.is_finite() {
                    samples.push((metric_gap, human_gap > diff_threshold));
                }
            }
            pair_idx += 1;
        }
    }
    if samples.len() < 2 {
        return f64::NAN;
    }
    // AUC via rank-sum (Mann-Whitney U). Sort by metric_gap, sum ranks
    // of the "different" class.
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let n_diff = samples.iter().filter(|s| s.1).count();
    let n_same = samples.len() - n_diff;
    if n_diff == 0 || n_same == 0 {
        return f64::NAN;
    }
    // Average-rank for ties, then U-statistic.
    let mut rank_sum_diff = 0.0f64;
    let mut k = 0usize;
    while k < samples.len() {
        let mut m = k;
        while m + 1 < samples.len() && samples[m + 1].0 == samples[k].0 {
            m += 1;
        }
        // ranks k+1 .. m+1 (1-based), average:
        let avg_rank = ((k + 1) + (m + 1)) as f64 / 2.0;
        for s in &samples[k..=m] {
            if s.1 {
                rank_sum_diff += avg_rank;
            }
        }
        k = m + 1;
    }
    let u = rank_sum_diff - (n_diff * (n_diff + 1)) as f64 / 2.0;
    u / (n_diff as f64 * n_same as f64)
}


// ============================================================================
// Bake scoring helpers
// ============================================================================

/// Per-sample α head dispatch payload — parsed from the bake's
/// `zentrain.per_sample_alpha_head` metadata. Layout matches
/// `zensim-train-core::per_sample_alpha_head::bake_per_sample_alpha_head_v3`
/// (and zensim's runtime in `zensim::metric::forward_one_bake`).
type PerSampleAlphaHeadDispatch = (Vec<f32>, f32, Vec<f32>, f32, [f32; 4], f32, f32);

/// Hybrid-head dispatch payload — parsed from the bake's
/// `zentrain.hybrid_head` metadata. Layout matches
/// `zensim-train-core::hybrid_head::bake_hybrid_head_v3`.
///
/// `(rank_w, rank_b, alpha_logit, reducer_w, reducer_b, p_norm)`.
type HybridHeadDispatch = (Vec<f32>, f32, f32, [f32; 4], f32, f32);

/// Read the `zentrain.tanh_output_head` metadata payload, if any.
/// Returns the sigmoid pin scale (`f32 LE`, single value) or `None`
/// when the key is absent / payload malformed.
///
/// EXP-CROSS-CODEC-V4 (2026-05-19). When present, the final score
/// returned from `score_row` is wrapped as `100·σ(y_pre/scale)`,
/// matching `zensim::metric::apply_tanh_output_pin` bit-exactly.
fn extract_tanh_output_head_scale(model: &Model) -> Option<f64> {
    let md = model.metadata();
    let entry = md.get("zentrain.tanh_output_head")?;
    if entry.value.len() != 4 {
        return None;
    }
    let scale = f32::from_le_bytes([
        entry.value[0],
        entry.value[1],
        entry.value[2],
        entry.value[3],
    ]) as f64;
    if scale.is_finite() && scale > 0.0 {
        Some(scale)
    } else {
        None
    }
}

/// Read the `zentrain.per_sample_alpha_head` metadata payload, if any.
/// Returns `Some((W_α, b_α, rank_w, rank_b, reducer_w, reducer_b, p_norm))`.
fn extract_per_sample_alpha_head(model: &Model) -> Option<PerSampleAlphaHeadDispatch> {
    let md = model.metadata();
    let entry = md.get("zentrain.per_sample_alpha_head")?;
    let n_hidden = model.n_outputs();
    let expected = (2 * n_hidden + 8) * 4;
    if entry.value.len() != expected {
        return None;
    }
    let mut floats = Vec::with_capacity(2 * n_hidden + 8);
    for chunk in entry.value.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let w_alpha = floats[..n_hidden].to_vec();
    let b_alpha = floats[n_hidden];
    let rank_w = floats[n_hidden + 1..2 * n_hidden + 1].to_vec();
    let rank_b = floats[2 * n_hidden + 1];
    let reducer_w = [
        floats[2 * n_hidden + 2],
        floats[2 * n_hidden + 3],
        floats[2 * n_hidden + 4],
        floats[2 * n_hidden + 5],
    ];
    let reducer_b = floats[2 * n_hidden + 6];
    let p_norm = floats[2 * n_hidden + 7];
    Some((
        w_alpha, b_alpha, rank_w, rank_b, reducer_w, reducer_b, p_norm,
    ))
}

/// Read the `zentrain.hybrid_head` metadata payload, if any.
/// Returns `Some((rank_w, rank_b, alpha_logit, reducer_w, reducer_b, p_norm))`.
fn extract_hybrid_head(model: &Model) -> Option<HybridHeadDispatch> {
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

/// Score one row through the loaded MLP. Caller pre-allocates the
/// `f32_features` scratch buffer to avoid the per-row allocation
/// that would otherwise dominate runtime (~10 ms × 19k pairs ≈ 3
/// min if we reallocated every call).
///
/// When the bake carries `zentrain.per_sample_alpha_head` metadata
/// (V_24-per-sample-α architecture), the forward output is treated
/// as the hidden vector `h` (the bake's layer 2 is an identity
/// passthrough) and the runtime mixes a rank-head + pool-head pair
/// via a per-sample sigmoid gate. Bit-exact match with
/// `zensim::metric::forward_one_bake`'s dispatch.
///
/// When the bake carries `zentrain.hybrid_head` metadata (V_24-hybrid
/// architecture), the same idea but the α gate is a single learned
/// SCALAR (not per-sample).
fn score_row(
    predictor: &mut Predictor<'_>,
    has_transforms: bool,
    per_sample_alpha_head: Option<&PerSampleAlphaHeadDispatch>,
    hybrid_head: Option<&HybridHeadDispatch>,
    tanh_pin_scale: Option<f64>,
    output_spline: Option<&zensim_validate::output_calibration_spline::OutputCalibrationSpline>,
    f32_features: &mut [f32],
    row: &[f64],
) -> f64 {
    let n_inputs = f32_features.len();
    let take = n_inputs.min(row.len());
    for i in 0..take {
        f32_features[i] = row[i] as f32;
    }
    // Pad with zeros if the parquet is wider than the bake (unlikely
    // — all T10.1 parquets are 372-wide and bakes are ≤ 372).
    for f in &mut f32_features[take..] {
        *f = 0.0;
    }
    let result = if has_transforms {
        predictor.predict_transformed(f32_features)
    } else {
        predictor.predict(f32_features)
    };
    let y_pre = match result {
        Ok(out) => {
            // Per-sample-α head dispatch — out is the hidden vector h.
            if let Some((w_alpha, b_alpha, rank_w, rank_b, reducer_w, reducer_b, p_norm)) =
                per_sample_alpha_head
            {
                let n = out.len() as f64;
                if n <= 0.0 || out.len() != rank_w.len() || out.len() != w_alpha.len() {
                    return f64::NAN;
                }
                let mut y_rank = *rank_b as f64;
                let mut alpha_logit = *b_alpha as f64;
                let mut sum = 0.0f64;
                let mut max_v = f64::NEG_INFINITY;
                let mut sum_p = 0.0f64;
                let p = *p_norm as f64;
                for (j, &h) in out.iter().enumerate() {
                    let hf = h as f64;
                    y_rank += hf * rank_w[j] as f64;
                    alpha_logit += hf * w_alpha[j] as f64;
                    sum += hf;
                    if hf > max_v {
                        max_v = hf;
                    }
                    sum_p += hf.abs().powf(p);
                }
                let mu = sum / n;
                let mut var = 0.0f64;
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
                    let xc = alpha_logit.clamp(-20.0, 20.0);
                    1.0 / (1.0 + (-xc).exp())
                };
                alpha * y_rank + (1.0 - alpha) * y_pool
            } else if let Some((rank_w, rank_b, alpha_logit, reducer_w, reducer_b, p_norm)) =
                hybrid_head
            {
                // Hybrid-head dispatch — out is the hidden vector h,
                // α is a learned scalar (not per-sample).
                let n = out.len() as f64;
                if n <= 0.0 || out.len() != rank_w.len() {
                    return f64::NAN;
                }
                let mut y_rank = *rank_b as f64;
                let mut sum = 0.0f64;
                let mut max_v = f64::NEG_INFINITY;
                let mut sum_p = 0.0f64;
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
                let mut var = 0.0f64;
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
            } else {
                out.first().copied().map(|v| v as f64).unwrap_or(f64::NAN)
            }
        }
        Err(_) => f64::NAN,
    };
    // EXP-CROSS-CODEC-V4 (2026-05-19): tanh-pinned [0, 100] output
    // wrap. Bit-exact with `zensim::metric::apply_tanh_output_pin`.
    let y_after_pin = if let Some(scale) = tanh_pin_scale {
        if !y_pre.is_nan() {
            let xc = (y_pre / scale).clamp(-30.0, 30.0);
            let s = 1.0 / (1.0 + (-xc).exp());
            100.0 * s
        } else {
            y_pre
        }
    } else {
        y_pre
    };
    // EXP-CROSS-CODEC-V9 (2026-05-20): post-network PCHIP spline
    // calibration. Bit-exact with `zensim::metric::apply_output_calibration_spline`.
    if let Some(spline) = output_spline {
        if !y_after_pin.is_nan() {
            return zensim_validate::output_calibration_spline::apply(y_after_pin, spline);
        }
    }
    y_after_pin
}

// ============================================================================
// Corpus registry
// ============================================================================

#[derive(Clone, Debug)]
struct Corpus {
    name: &'static str,
    /// Display name in tables (matches dataset_metric_baseline.rs
    /// for diff-friendliness across the two binaries).
    display: &'static str,
    /// Parquet path (slot under `<features_root>/`).
    filename: &'static str,
    /// Per-band partitioning enabled? AIC-3 has 600 pairs in a JND
    /// step grid; rank-based per-band stats collapse to 0 on shared
    /// scores (see dataset_metric_baseline.rs comment at L454-471).
    enable_per_band: bool,
}

const CORPORA: &[Corpus] = &[
    Corpus {
        name: "cid22",
        display: "CID22",
        filename: "cid22_features_372col_2026-05-15.parquet",
        enable_per_band: true,
    },
    Corpus {
        name: "kadid",
        display: "KADIK10k",
        filename: "kadid_features_372col_2026-05-15.parquet",
        enable_per_band: true,
    },
    Corpus {
        name: "tid",
        display: "TID2013",
        filename: "tid_features_372col_2026-05-15.parquet",
        enable_per_band: true,
    },
    Corpus {
        name: "konjnd",
        display: "KonJND-1k (full)",
        filename: "konjnd_features_372col_2026-05-15.parquet",
        // KonJND `human_score` here is `mean_threshold` (raw,
        // unit unclear from extract_features_372col.rs but
        // appears to be a per-pair JND threshold in [22, 70]).
        // 10-band-on-[0,1] partitioning doesn't apply; skip.
        enable_per_band: false,
    },
    Corpus {
        name: "aic3",
        display: "AIC-3 CTC",
        filename: "aic3_features_372col_2026-05-15.parquet",
        // AIC-3 = JND step grid (see comment above + L454-471
        // of dataset_metric_baseline.rs); per-band aggregate
        // is misleading.
        enable_per_band: false,
    },
    Corpus {
        name: "aic4",
        display: "AIC-4 sample",
        // AIC-4 sample (5 source × 6 codecs × 10 dlevels = 300 pairs).
        // `human_score` = reconstructed JND units (signed, ~0..6 range);
        // same convention as AIC-3. Like AIC-3 this is a JND step grid
        // so per-band aggregate on [0, 1] doesn't apply.
        filename: "aic4_features_372col_2026-05-20.parquet",
        enable_per_band: false,
    },
];

fn parse_corpora_arg(arg: &str) -> Result<Vec<&'static Corpus>, String> {
    let mut out: Vec<&'static Corpus> = Vec::new();
    for name in arg.split(',') {
        let key = name.trim().to_lowercase();
        let found = CORPORA.iter().find(|c| c.name == key);
        match found {
            Some(c) => {
                if !out.iter().any(|existing| existing.name == c.name) {
                    out.push(c);
                }
            }
            None => {
                return Err(format!(
                    "unknown corpus {key:?} — known: {}",
                    CORPORA.iter().map(|c| c.name).collect::<Vec<_>>().join(",")
                ));
            }
        }
    }
    Ok(out)
}

// ============================================================================
// CLI parsing
// ============================================================================

struct Args {
    bake: PathBuf,
    corpora: Vec<&'static Corpus>,
    output: Option<PathBuf>,
    features_root: PathBuf,
    /// Diagnostic: dump per-row `human<TAB>pred` (parquet row order) to this
    /// path. Used by the AIC-3 CVVDP-feature spike to compute per-ref SROCC
    /// (which the aggregate panel does not split out).
    per_pair_output: Option<PathBuf>,
}

fn print_usage() {
    eprintln!(
        "bake_verdict — instant V_X bake eval from pre-extracted parquet features\n\
\n\
USAGE:\n\
    bake_verdict --bake <path>\n\
                 [--corpora cid22,kadid,tid,konjnd,aic3,aic4]\n\
                 [--output <path.md>]\n\
                 [--features-root /mnt/v/zen/zensim-training/2026-05-15-full-features]\n\
\n\
DEFAULTS:\n\
    --corpora       all 6 (cid22,kadid,tid,konjnd,aic3,aic4)\n\
    --output        stdout\n\
    --features-root /mnt/v/zen/zensim-training/2026-05-15-full-features\n"
    );
}

fn parse_args() -> Result<Args, String> {
    let mut bake: Option<PathBuf> = None;
    let mut corpora: Option<Vec<&'static Corpus>> = None;
    let mut output: Option<PathBuf> = None;
    let mut per_pair_output: Option<PathBuf> = None;
    let mut features_root: PathBuf =
        PathBuf::from("/mnt/v/zen/zensim-training/2026-05-15-full-features");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bake" => {
                let v = args.next().ok_or("--bake requires <path>")?;
                bake = Some(PathBuf::from(v));
            }
            "--corpora" => {
                let v = args.next().ok_or("--corpora requires comma list")?;
                corpora = Some(parse_corpora_arg(&v)?);
            }
            "--output" => {
                let v = args.next().ok_or("--output requires <path>")?;
                output = Some(PathBuf::from(v));
            }
            "--features-root" => {
                let v = args.next().ok_or("--features-root requires <path>")?;
                features_root = PathBuf::from(v);
            }
            "--per-pair-output" => {
                let v = args.next().ok_or("--per-pair-output requires <path>")?;
                per_pair_output = Some(PathBuf::from(v));
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                return Err(format!("unknown arg: {other}"));
            }
        }
    }
    let bake = bake.ok_or("--bake is required (path to ZNPR v3 bake)")?;
    let corpora = corpora.unwrap_or_else(|| CORPORA.iter().collect());
    Ok(Args {
        bake,
        corpora,
        output,
        features_root,
        per_pair_output,
    })
}

// ============================================================================
// Per-corpus pipeline
// ============================================================================

struct CorpusResult {
    display: &'static str,
    n: usize,
    srocc: f64,
    plcc: f64,
    krocc: f64,
    or_ratio: f64,
    pwrc: f64,
    z_rmse: f64,
    ds_auc: f64,
    /// Logistic-rescaled scores in [0,100] dial space (for G1 range check).
    rescaled_scores: Vec<f64>,
    body: String,
}

fn aggregate_panel(scores: &[f64], humans: &[f64]) -> (f64, f64, f64, f64, f64, f64, f64) {
    // Canonical 6-stat panel comes from `compute_panel` (re-exported
    // from `zenstats::panel` via `zensim_validate::panel`). Pre-2026-05-26
    // this binary had a parallel inline copy of every stat — those
    // inline copies were never updated when panel.rs's OR + PWRC were
    // rewritten to the paper-correct ITU-T P.1401 + Mohammadi SA-ST AUC
    // forms (commit 83e7ff70). Every bake_verdict output before this
    // commit therefore reported the older proxy OR + PWRC despite the
    // `panel` binary's output being paper-correct on the same fixture.
    // DS-AUC stays bake_verdict-local (no panel equivalent yet).
    let p = compute_panel(scores, humans);
    // DS-AUC threshold: 1 std of human scores marks "perceptually different".
    let h_mean: f64 = humans.iter().sum::<f64>() / humans.len().max(1) as f64;
    let h_std = (humans.iter().map(|x| (x - h_mean).powi(2)).sum::<f64>()
        / humans.len().max(1) as f64)
        .sqrt();
    let ds_raw = ds_auc(scores, humans, h_std);
    // Orientation-correct: larger metric-gap should mean "more different".
    let ds = if ds_raw.is_finite() {
        ds_raw.max(1.0 - ds_raw)
    } else {
        ds_raw
    };
    (p.srocc, p.plcc, p.krocc, p.or_ratio, p.pwrc, p.z_rmse, ds)
}

fn render_corpus(
    corpus: &Corpus,
    features_root: &Path,
    has_transforms: bool,
    n_inputs: usize,
    model: &Model,
    per_pair_output: Option<&Path>,
) -> Result<CorpusResult, String> {
    let path = features_root.join(corpus.filename);
    let g = parquet_loader::load_parquet(&path, corpus.display, "human_score", 1.0)
        .map_err(|e| format!("load {} parquet: {e}", corpus.display))?;
    let humans = g.human_scores;
    let per_sample_alpha_head = extract_per_sample_alpha_head(model);
    let hybrid_head = extract_hybrid_head(model);
    let tanh_pin_scale = extract_tanh_output_head_scale(model);
    let output_spline = zensim_validate::output_calibration_spline::extract(model);
    let mut predictor = Predictor::new(model);

    // Score every row. f32 scratch buffer reused across all rows
    // to avoid the per-row allocation that would otherwise dominate
    // wall time on the bigger corpora (KADID has 10k rows × 372 f32s).
    let mut scratch = vec![0.0f32; n_inputs];
    let scores: Vec<f64> = g
        .feature_rows
        .iter()
        .map(|row| {
            score_row(
                &mut predictor,
                has_transforms,
                per_sample_alpha_head.as_ref(),
                hybrid_head.as_ref(),
                tanh_pin_scale,
                output_spline.as_ref(),
                &mut scratch,
                row,
            )
        })
        .collect();

    let n = scores.len();

    // Diagnostic per-pair dump (parquet row order): `human<TAB>pred`.
    if let Some(path) = per_pair_output {
        let mut s = String::from("human\tpred\n");
        for (h, p) in humans.iter().zip(scores.iter()) {
            s.push_str(&format!("{h}\t{p}\n"));
        }
        std::fs::write(path, s).map_err(|e| format!("write per-pair output: {e}"))?;
        eprintln!("  wrote per-pair predictions to {}", path.display());
    }

    let (srocc, plcc, krocc, or_, pw, z, ds) = aggregate_panel(&scores, &humans);

    let mut body = String::new();
    body.push_str(&format!("\n## {} (n={})\n\n", corpus.display, n));
    body.push_str("### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)\n\n");
    body.push_str("| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |\n");
    body.push_str("|---|---:|---:|---:|---:|---:|---:|---:|\n");
    body.push_str(&format!(
        "| V_X bake | {srocc:.4} | {plcc:.4} | {krocc:.4} | {or_:.4} | {pw:.4} | {z:.3} | {ds:.4} |\n"
    ));
    body.push('\n');
    body.push_str(
        "_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from \
parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), \
not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because \
saturation regions dominate the residual._\n",
    );

    if corpus.enable_per_band {
        body.push('\n');
        body.push_str(&format!(
            "### {} 10-band full Mohammadi panel (PRIMARY release gate)\n\n",
            corpus.display
        ));
        body.push_str("| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |\n");
        body.push_str("|---|---|--:|---:|---:|---:|---:|---:|---:|---:|\n");
        // Per-band cuts: every corpus that hits this branch
        // (CID22 / KADID / TID) has human_score normalized
        // into [0, 1] per the feature-extractor convention.
        // Width-10 grid on the 0-100 scale → width-0.10 on [0, 1].
        for band_idx in 0..10 {
            let lo = band_idx as f64 * 0.10;
            let hi = lo + 0.10;
            let label = format!("B{band_idx}");
            let range_label = if band_idx == 9 {
                format!("[{:.2}, 1.00]", lo)
            } else {
                format!("[{:.2}, {:.2})", lo, hi)
            };
            let idxs: Vec<usize> = humans
                .iter()
                .enumerate()
                .filter_map(|(i, &h)| {
                    if band_idx == 9 {
                        (h >= lo).then_some(i)
                    } else {
                        (h >= lo && h < hi).then_some(i)
                    }
                })
                .collect();
            if idxs.len() < 4 {
                body.push_str(&format!(
                    "| {label} | {range_label} | {} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |\n",
                    idxs.len()
                ));
                continue;
            }
            let h_b: Vec<f64> = idxs.iter().map(|&i| humans[i]).collect();
            let s_b: Vec<f64> = idxs.iter().map(|&i| scores[i]).collect();
            let (b_srocc, b_plcc, b_krocc, b_or, b_pwrc, b_z, _b_ds) = aggregate_panel(&s_b, &h_b);
            let rescaled = rescale_logistic(&s_b, &h_b);
            let mae: f64 = rescaled
                .iter()
                .zip(h_b.iter())
                .map(|(r, h)| (r - h).abs())
                .sum::<f64>()
                / idxs.len() as f64;
            let noisy = if idxs.len() < 30 { " ⚠" } else { "" };
            body.push_str(&format!(
                "| {label}{noisy} | {range_label} | {} | {b_srocc:.4} | {b_plcc:.4} | {b_krocc:.4} | {b_or:.4} | {b_pwrc:.4} | {b_z:.3} | {mae:.4} |\n",
                idxs.len()
            ));
        }
        body.push('\n');
        body.push_str(
            "_⚠ marks bands with n < 30 — point estimates are noisy (CI widths \
exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically \
distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale \
per Mohammadi 2025._\n",
        );
    } else {
        body.push('\n');
        body.push_str(&format!(
            "_Per-band breakdown skipped for {} — the corpus uses a JND step grid (AIC-3) \
or a raw threshold scale (KonJND) that doesn't partition cleanly into the \
CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing \
read on this corpus._\n",
            corpus.display
        ));
    }

    Ok(CorpusResult {
        display: corpus.display,
        n,
        srocc,
        plcc,
        krocc,
        or_ratio: or_,
        pwrc: pw,
        z_rmse: z,
        ds_auc: ds,
        rescaled_scores: scores.clone(),
        body,
    })
}

/// Soft gate: linear ramp from `floor` (score 0.0) to `target` (score 1.0).
/// Direction-aware: if `target < floor`, lower values score higher.
fn soft_gate(value: f64, floor: f64, target: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    if (target - floor).abs() < 1e-12 {
        return if value >= target { 1.0 } else { 0.0 };
    }
    ((value - floor) / (target - floor)).clamp(0.0, 1.0)
}

/// Percentile of a slice (linear interpolation, p in [0,100]).
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = (p / 100.0) * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = rank - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() -> ExitCode {
    let t0 = Instant::now();
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("bake_verdict: {e}");
            print_usage();
            return ExitCode::from(2);
        }
    };
    eprintln!(
        "bake_verdict — bake={}  features-root={}  corpora={}",
        args.bake.display(),
        args.features_root.display(),
        args.corpora
            .iter()
            .map(|c| c.name)
            .collect::<Vec<_>>()
            .join(",")
    );

    let bake_bytes = match std::fs::read(&args.bake) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "bake_verdict: failed to read bake {}: {e}",
                args.bake.display()
            );
            return ExitCode::from(1);
        }
    };
    let model = match Model::from_bytes(&bake_bytes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("bake_verdict: failed to parse ZNPR bake: {e:?}");
            return ExitCode::from(1);
        }
    };
    let n_inputs = model.n_inputs();
    let has_transforms = model.has_nontrivial_feature_transforms();
    let has_per_sample_alpha = extract_per_sample_alpha_head(&model).is_some();
    let has_hybrid_head = extract_hybrid_head(&model).is_some();
    eprintln!(
        "bake: n_inputs={n_inputs}  feature_transforms={}  per_sample_alpha_head={}  hybrid_head={}",
        if has_transforms { "yes" } else { "no" },
        if has_per_sample_alpha { "yes" } else { "no" },
        if has_hybrid_head { "yes" } else { "no" }
    );

    let mut buf = String::new();
    buf.push_str("# bake_verdict — instant V_X eval\n\n");
    buf.push_str(&format!("- Bake: `{}`\n", args.bake.display()));
    buf.push_str(&format!(
        "- Feature parquets: `{}`\n",
        args.features_root.display()
    ));
    buf.push_str(&format!("- Bake n_inputs: {n_inputs}\n"));
    buf.push_str(&format!(
        "- Feature transforms: {}\n",
        if has_transforms {
            "yes (uses predict_transformed)"
        } else {
            "no"
        }
    ));

    let mut results: Vec<CorpusResult> = Vec::new();
    for corpus in &args.corpora {
        // Per-pair dump only meaningful when a single corpus is selected
        // (one output path → one corpus); pass through for all, last wins.
        match render_corpus(
            corpus,
            &args.features_root,
            has_transforms,
            n_inputs,
            &model,
            args.per_pair_output.as_deref(),
        ) {
            Ok(r) => results.push(r),
            Err(e) => {
                eprintln!("bake_verdict: {e}");
                return ExitCode::from(1);
            }
        }
    }

    // One-row summary across all corpora at the top.
    buf.push_str("\n## Summary (one row per corpus)\n\n");
    buf.push_str(
        "| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC | geomean3 |\n",
    );
    buf.push_str("|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for r in &results {
        let g3 = (r.srocc * r.plcc * r.pwrc).cbrt();
        buf.push_str(&format!(
            "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.3} | {:.4} | {:.4} |\n",
            r.display, r.n, r.srocc, r.plcc, r.krocc, r.or_ratio, r.pwrc, r.z_rmse, r.ds_auc, g3
        ));
    }
    // ── CODEC_TARGET_GOALS.md scorecard ──────────────────────────────
    // Measurable from held-out corpus scores alone. Goals needing
    // external q-sweep / cross-codec data (G3, G4, G10) are flagged.
    {
        let find = |name: &str| results.iter().find(|r| r.display.contains(name));
        let cid22 = find("CID22");
        let konjnd = find("KonJND");
        let aic3 = find("AIC-3");

        // G1: dynamic range — pool all dial-space scores, check p5/p95.
        let mut pooled: Vec<f64> = results
            .iter()
            .flat_map(|r| r.rescaled_scores.iter().copied())
            .filter(|x| x.is_finite())
            .collect();
        pooled.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let (p5, p95) = if pooled.is_empty() {
            (f64::NAN, f64::NAN)
        } else {
            (percentile(&pooled, 5.0), percentile(&pooled, 95.0))
        };
        // Scores are 0-100 dial; goal: p5 ≤ 25, p95 ≥ 85.
        let g1 = soft_gate(p5, 50.0, 25.0).min(soft_gate(p95, 50.0, 85.0));

        // G5: HF rank — KonJND SROCC (floor 0.70, target 0.85) + AIC-3.
        let g5_konjnd = konjnd
            .map(|r| soft_gate(r.srocc, 0.70, 0.85))
            .unwrap_or(0.0);
        let g5_aic3 = aic3.map(|r| soft_gate(r.srocc, 0.70, 0.85)).unwrap_or(0.0);
        let g5 = (g5_konjnd + g5_aic3) / 2.0;

        // G7: CID22 SROCC ≥ 0.85 (advisory).
        let g7 = cid22.map(|r| soft_gate(r.srocc, 0.80, 0.85)).unwrap_or(0.0);

        // G8: Z-RMSE — lower is better. AIC-3 floor 0.80, target 0.50.
        let g8 = aic3.map(|r| soft_gate(r.z_rmse, 0.80, 0.50)).unwrap_or(0.0);

        // G9: DS-AUC — AIC-3 floor 0.70, target 0.85.
        let g9 = aic3.map(|r| soft_gate(r.ds_auc, 0.70, 0.85)).unwrap_or(0.0);

        buf.push_str("\n## CODEC_TARGET_GOALS.md scorecard (measurable subset)\n\n");
        buf.push_str("| Goal | Measure | Value | Soft score |\n");
        buf.push_str("|---|---|---:|---:|\n");
        buf.push_str(&format!(
            "| G1 dynamic range | pooled p5≤25 ∧ p95≥85 | p5={p5:.1} p95={p95:.1} | {g1:.2} |\n"
        ));
        buf.push_str(&format!(
            "| G5 HF rank | KonJND+AIC-3 SROCC ≥0.70 | {:.3} / {:.3} | {g5:.2} |\n",
            konjnd.map(|r| r.srocc).unwrap_or(f64::NAN),
            aic3.map(|r| r.srocc).unwrap_or(f64::NAN),
        ));
        buf.push_str(&format!(
            "| G7 CID22 rank | SROCC ≥0.85 (advisory) | {:.4} | {g7:.2} |\n",
            cid22.map(|r| r.srocc).unwrap_or(f64::NAN),
        ));
        buf.push_str(&format!(
            "| G8 Z-RMSE | AIC-3 ≤0.80 | {:.3} | {g8:.2} |\n",
            aic3.map(|r| r.z_rmse).unwrap_or(f64::NAN),
        ));
        buf.push_str(&format!(
            "| G9 DS-AUC | AIC-3 ≥0.70 | {:.4} | {g9:.2} |\n",
            aic3.map(|r| r.ds_auc).unwrap_or(f64::NAN),
        ));
        // Weighted composite per the doc's priority order (G1=3, G8=2.5,
        // G5=1.5, G9=1, G7=0.5). G2/G3/G4/G6/G10/G11 need external data.
        let weighted =
            (3.0 * g1 + 2.5 * g8 + 1.5 * g5 + 1.0 * g9 + 0.5 * g7) / (3.0 + 2.5 + 1.5 + 1.0 + 0.5);
        buf.push_str(&format!(
            "\n**Weighted goal score (measurable subset): {weighted:.3}**\n\n"
        ));
        buf.push_str(
            "_G2 (JND anchor), G3 (monotonicity), G4 (cross-codec), G6 (MF \
band coverage), G10 (per-source), G11 (display) require external q-sweep / \
cross-codec / multi-PPD data not present in the held-out feature parquets. \
Run the dedicated q-sweep harness for those._\n",
        );
    }

    for r in &results {
        buf.push_str(&r.body);
    }

    let elapsed = t0.elapsed();
    buf.push_str(&format!(
        "\n---\nWall time: {:.2}s ({} pair rows scored across {} corpora).\n",
        elapsed.as_secs_f64(),
        results.iter().map(|r| r.n).sum::<usize>(),
        results.len()
    ));

    if let Some(out_path) = args.output {
        if let Some(parent) = out_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match File::create(&out_path) {
            Ok(mut f) => {
                if let Err(e) = f.write_all(buf.as_bytes()) {
                    eprintln!("bake_verdict: failed to write {}: {e}", out_path.display());
                    return ExitCode::from(1);
                }
                eprintln!("wrote verdict to {}", out_path.display());
            }
            Err(e) => {
                eprintln!("bake_verdict: failed to create {}: {e}", out_path.display());
                return ExitCode::from(1);
            }
        }
    } else {
        print!("{buf}");
    }

    eprintln!("bake_verdict: complete in {:.2}s", elapsed.as_secs_f64());
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ds_auc_perfect_separation() {
        // Metric gap perfectly tracks human gap → AUC = 1.0
        let human = vec![0.0, 0.0, 1.0, 1.0];
        let pred = vec![0.0, 0.0, 1.0, 1.0];
        let auc = ds_auc(&pred, &human, 0.5);
        assert!(
            auc > 0.95,
            "perfect separation should give AUC≈1, got {auc}"
        );
    }

    #[test]
    fn ds_auc_random_is_chance() {
        // Constant metric → can't separate → AUC ≈ 0.5
        let human = vec![0.0, 0.3, 0.6, 0.9, 0.2, 0.7];
        let pred = vec![0.5; 6];
        let auc = ds_auc(&pred, &human, 0.4);
        // Constant predictions: all gaps are 0, ties → AUC = 0.5
        assert!(
            (auc - 0.5).abs() < 0.01 || auc.is_nan(),
            "constant metric should give AUC≈0.5, got {auc}"
        );
    }

    #[test]
    fn ds_auc_handles_degenerate() {
        // All same human score → no "different" pairs → NaN
        let human = vec![0.5; 5];
        let pred = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let auc = ds_auc(&pred, &human, 0.4);
        assert!(
            auc.is_nan(),
            "no different-pairs should give NaN, got {auc}"
        );
    }
}
