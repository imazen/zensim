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
//! (`zensim-experimental/weights/v0_22_iw_v2_calibrated_2026-05-16.bin`), the
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
            if pair_idx.is_multiple_of(stride) {
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
// Bake scoring helpers — DEDUP-M (2026-05-26):
// `PerSampleAlphaHeadDispatch`, `HybridHeadDispatch`, `extract_*` helpers,
// and `score_row` were factored into `zensim_validate::bake_runtime`.
// Six bins (this one + qsweep_eval + preview_stats_demo + ensemble_score_rows
// + score_pair_with_bake + predict_features_with_bake) used to carry
// ~90-95 % shared local copies. The factored runtime is bit-exact (f32 ±1e-6
// on representative parquet rows; see benchmarks/dedup_M_score_row_evidence/).
// ============================================================================

use zensim_validate::bake_runtime::{
    extract_hybrid_head, extract_per_sample_alpha_head, extract_tanh_output_head_scale, score_row,
};

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
    /// DIAL-panel grid parquet (`image_id, codec, q, f0..f371`). Default
    /// is the canonical densified multi-codec grid; override with
    /// `--dial-grid` or `ZENSIM_DIAL_GRID`. When the file is absent the
    /// dial panel is skipped with a loud note (it cannot be recomputed
    /// without the stored feature grid — fetch from R2 eval-grids/).
    dial_grid: PathBuf,
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
    let mut dial_grid: PathBuf = std::env::var("ZENSIM_DIAL_GRID")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(
                "/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29.parquet",
            )
        });
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
            "--dial-grid" => {
                let v = args.next().ok_or("--dial-grid requires <path>")?;
                dial_grid = PathBuf::from(v);
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
        dial_grid,
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

/// DIAL panel — the codec-target half of the eval, run on the densified
/// multi-codec q-sweep grid. For each `(image_id, codec)` curve sorted by
/// `q`, counts strict-decrease violations + ties → monotonicity / tied
/// rate (G3); pools dial-space scores → p5/p95 (G1). This is the metric
/// `bake_verdict` previously lacked — a bake can win the rank panel and
/// still be a broken dial. Returns the markdown section, or a loud SKIPPED
/// note when the stored grid is absent (it can't be recomputed without the
/// feature grid — see docs/EVAL_PANEL_REQUIREMENT.md).
fn dial_panel(model: &Model, has_transforms: bool, n_inputs: usize, grid_path: &Path) -> String {
    if !grid_path.exists() {
        return format!(
            "\n## DIAL panel — ⚠ SKIPPED (grid not found)\n\n\
             Dial grid `{}` is absent. The DIAL panel (G1 range / G3 monotonicity\n\
             / codec reach) is MANDATORY per docs/EVAL_PANEL_REQUIREMENT.md — fetch\n\
             the stored feature grid from `s3://zentrain/eval-grids/` (or set\n\
             `--dial-grid` / `ZENSIM_DIAL_GRID`) and re-run. A rank-only verdict is\n\
             a regression.\n",
            grid_path.display()
        );
    }
    let grid = match parquet_loader::load_dial_grid(&grid_path.to_path_buf()) {
        Ok(g) => g,
        Err(e) => return format!("\n## DIAL panel — ⚠ FAILED to load grid\n\n`{e}`\n"),
    };

    // Score every grid row through the SAME dispatch path render_corpus uses.
    let per_sample_alpha_head = extract_per_sample_alpha_head(model);
    let hybrid_head = extract_hybrid_head(model);
    let tanh_pin_scale = extract_tanh_output_head_scale(model);
    let output_spline = zensim_validate::output_calibration_spline::extract(model);
    let mut predictor = Predictor::new(model);
    let mut scratch = vec![0.0f32; n_inputs];
    let scores: Vec<f64> = grid
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

    // Group rows into (image_id, codec) curves, carrying (q, score, row_idx).
    // row_idx lets us compare adjacent cells' feature vectors: when a codec
    // SATURATES (e.g. zenjpeg/webp produce byte-identical encodes for q99.25
    // vs q99.9), the features are identical and the bake MUST score them
    // identically — that is the codec's quality ceiling, not a bake dead-zone,
    // so it must not count against the bake's flat/clamp gate.
    use std::collections::BTreeMap;
    /// Per `(image, codec)` curve: `(q, score, row_idx)` cells.
    type CurveMap = BTreeMap<(String, String), Vec<(f64, f64, usize)>>;
    let mut curves: CurveMap = BTreeMap::new();
    for (i, &score) in scores.iter().enumerate() {
        curves
            .entry((grid.image_id[i].clone(), grid.codec[i].clone()))
            .or_default()
            .push((grid.q[i], score, i));
    }
    // Adjacent cells are "codec-saturated" when their 372-feature vectors are
    // near-identical (the codec emitted the same image at two different q) —
    // detected by L-inf distance below FEAT_EPS (small margin for GPU-extract
    // ULP noise).
    let feat_eq = |a: usize, b: usize| -> bool {
        const FEAT_EPS: f64 = 1e-5;
        let ra = &grid.feature_rows[a];
        let rb = &grid.feature_rows[b];
        ra.len() == rb.len()
            && ra
                .iter()
                .zip(rb.iter())
                .all(|(x, y)| (x - y).abs() <= FEAT_EPS)
    };

    // Per-codec native-param extremes + dial score at the representable
    // min/max codec config. `codec_param` is integer quality for q-codecs,
    // butteraugli distance for JXL (`param_kind` labels which). We report
    // the score at the LOWEST-quality and HIGHEST-quality representable
    // config (note: for distance, HIGHER distance = LOWER quality), so the
    // table shows the dial's reach at each codec's quality endpoints.
    struct ParamExtremes {
        kind: String,
        lo_param: f64,
        hi_param: f64,
        // median score at the worst-quality and best-quality endpoints
        score_at_worst: Vec<f64>,
        score_at_best: Vec<f64>,
    }
    let mut pext: BTreeMap<String, ParamExtremes> = BTreeMap::new();
    for c in grid.codec.iter() {
        pext.entry(c.clone()).or_insert_with(|| ParamExtremes {
            kind: "q".to_string(),
            lo_param: f64::INFINITY,
            hi_param: f64::NEG_INFINITY,
            score_at_worst: Vec::new(),
            score_at_best: Vec::new(),
        });
    }
    for i in 0..scores.len() {
        let e = pext.get_mut(&grid.codec[i]).unwrap();
        e.kind = grid.param_kind[i].clone();
        e.lo_param = e.lo_param.min(grid.codec_param[i]);
        e.hi_param = e.hi_param.max(grid.codec_param[i]);
    }
    // second pass: collect scores at the param extremes per codec
    for (i, &score) in scores.iter().enumerate() {
        let e = pext.get_mut(&grid.codec[i]).unwrap();
        let p = grid.codec_param[i];
        // worst quality = highest distance OR lowest q; best = the opposite
        let (worst_param, best_param) = if e.kind == "distance" {
            (e.hi_param, e.lo_param)
        } else {
            (e.lo_param, e.hi_param)
        };
        if (p - worst_param).abs() <= 1e-9 {
            e.score_at_worst.push(score);
        }
        if (p - best_param).abs() <= 1e-9 {
            e.score_at_best.push(score);
        }
    }
    let median = |v: &mut Vec<f64>| -> f64 {
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[v.len() / 2]
    };

    // FIVE mutually-exclusive outcomes per adjacent-q pair (as quality rises),
    // so we never conflate a codec limitation, a metric dead-zone, sub-JND
    // noise, and a real ranking error:
    //   1. forward         — Δ >  MATERIAL_INV : clear quality increase (good)
    //   2. inversion       — Δ < -MATERIAL_INV : dial ran BACKWARDS by a
    //                          user-visible amount — a real ranking error. GATED.
    //   3. codec-saturated — adjacent features near-identical : the CODEC emitted
    //                          the same image at two different q (zenjpeg/webp
    //                          quality ceiling). The bake MUST score identical
    //                          inputs identically — NOT a bake defect, NOT gated.
    //   4. flat/clamp      — features DIFFER but |Δ| ≤ 1e-9 : the bake collapsed
    //                          distinct inputs to one score — a real metric
    //                          dead-zone (what V0_5-Balanced suffered). GATED.
    //   5. sub-resolution  — the rest (0 < |Δ| ≤ MATERIAL_INV, distinct features):
    //                          dial moved < half a point. EXPECTED on the dense
    //                          near-lossless grid (sub-JND configs); NOT gated.
    // MATERIAL_INV = 0.5 score-pt, below any user-targetable dial precision.
    const MATERIAL_INV: f64 = 0.5;
    // per-codec: [pairs, material_inversions, flat_clamp, n_curves]
    let mut tot_pairs = 0usize;
    let mut tot_fwd = 0usize; // Δ > MATERIAL_INV — clear quality increase
    let mut tot_inv = 0usize; // strict (any backwards > 1e-9) — diagnostic
    let mut tot_inv_material = 0usize; // backwards by > MATERIAL_INV — gate
    let mut tot_flat = 0usize; // distinct features, |Δ| ≤ 1e-9 — metric dead-zone — gate
    let mut tot_codec_sat = 0usize; // identical features — codec quality ceiling — not gated
    let mut tot_subres = 0usize; // 1e-9 < |Δ| ≤ MATERIAL_INV — expected oversampling
    let mut inv_mags: Vec<f64> = Vec::new(); // magnitudes of strict inversions
    let mut per_codec: BTreeMap<String, [usize; 4]> = BTreeMap::new();
    for ((_img, codec), pts) in curves.iter_mut() {
        pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let entry = per_codec.entry(codec.clone()).or_default();
        entry[3] += 1;
        for w in pts.windows(2) {
            let (_, s0, i0) = w[0];
            let (_, s1, i1) = w[1];
            tot_pairs += 1;
            entry[0] += 1;
            let delta = s1 - s0;
            if delta < -1e-9 {
                tot_inv += 1; // strict backwards (diagnostic, all magnitudes)
                inv_mags.push(-delta);
            }
            // five mutually-exclusive buckets summing to tot_pairs:
            if delta > MATERIAL_INV {
                tot_fwd += 1; // clear quality increase
            } else if delta < -MATERIAL_INV {
                tot_inv_material += 1; // material backwards (gate)
                entry[1] += 1;
            } else if feat_eq(i0, i1) {
                tot_codec_sat += 1; // codec emitted identical image — not the bake's fault
            } else if delta.abs() <= 1e-9 {
                tot_flat += 1; // distinct inputs, identical score — metric dead-zone (gate)
                entry[2] += 1;
            } else {
                tot_subres += 1; // 1e-9 < |Δ| ≤ MATERIAL_INV (expected; not gated)
            }
        }
    }
    // forward strict-increase rate; inversion rate; tied rate — three rates
    // that sum to 1. "monotonicity" (G3) = 1 - inversion rate (ties are not
    // inversions but are reported on their own line). The gate uses the
    // MATERIAL inversion count (backwards by > MATERIAL_INV score units);
    // sub-MATERIAL backwards wiggles fold into the tied/dead-zone bucket.
    let inv_rate = if tot_pairs > 0 {
        tot_inv_material as f64 / tot_pairs as f64
    } else {
        f64::NAN
    };
    // strict (any backwards > 1e-9) — diagnostic only, shows how much of the
    // strict count is sub-MATERIAL noise.
    let inv_rate_strict = if tot_pairs > 0 {
        tot_inv as f64 / tot_pairs as f64
    } else {
        f64::NAN
    };
    let mono = 1.0 - inv_rate;
    // flat/clamp dead-zone rate (literal |Δ|≤1e-9) — the gated tie metric.
    let flat = if tot_pairs > 0 {
        tot_flat as f64 / tot_pairs as f64
    } else {
        f64::NAN
    };
    // sub-resolution moves (0 < |Δ| ≤ MATERIAL) — informational, grid-density
    // dependent, not gated.
    let subres = if tot_pairs > 0 {
        tot_subres as f64 / tot_pairs as f64
    } else {
        f64::NAN
    };
    // codec-saturated pairs (adjacent features identical — codec quality
    // ceiling) — informational, NOT gated against the bake.
    let codec_sat = if tot_pairs > 0 {
        tot_codec_sat as f64 / tot_pairs as f64
    } else {
        f64::NAN
    };
    let forward = if tot_pairs > 0 {
        tot_fwd as f64 / tot_pairs as f64
    } else {
        f64::NAN
    };
    // median + p90 magnitude of strict backwards steps — characterizes whether
    // the strict inversions are noise wiggles or real reversals.
    inv_mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let inv_mag_med = if inv_mags.is_empty() {
        0.0
    } else {
        inv_mags[inv_mags.len() / 2]
    };
    let inv_mag_p90 = if inv_mags.is_empty() {
        0.0
    } else {
        percentile(&inv_mags, 90.0)
    };

    // G1 dynamic range on the codec grid: pool all scores, p5/p95.
    let mut pooled: Vec<f64> = scores.iter().copied().filter(|x| x.is_finite()).collect();
    pooled.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (p5, p95) = if pooled.is_empty() {
        (f64::NAN, f64::NAN)
    } else {
        (percentile(&pooled, 5.0), percentile(&pooled, 95.0))
    };
    let g1 = soft_gate(p5, 50.0, 25.0).min(soft_gate(p95, 50.0, 85.0));
    let g3 = soft_gate(mono, 0.90, 0.93).min(soft_gate(flat, 0.10, 0.05));

    let mut s = String::new();
    s.push_str("\n## DIAL panel (codec-target G1/G3 — densified multi-codec q-sweep)\n\n");
    s.push_str(&format!(
        "Grid: `{}` — {} rows, {} curves across {} codec families.\n\n",
        grid_path.display(),
        scores.len(),
        curves.len(),
        per_codec.len()
    ));
    s.push_str("| metric | value | gate | pass |\n|---|--:|---|:--:|\n");
    s.push_str(&format!(
        "| forward strict-increase | {forward:.4} | — | |\n"
    ));
    s.push_str(&format!(
        "| forward sub-resolution (≤{MATERIAL_INV}pt move) | {subres:.4} | — (dense-grid) | |\n"
    ));
    s.push_str(&format!(
        "| **inversions** (backwards > {MATERIAL_INV}pt) | {inv_rate:.4} | G3 ≤ 0.07 | {} |\n",
        if inv_rate <= 0.07 { "✓" } else { "✗" }
    ));
    s.push_str(&format!(
        "| ↳ strict backwards (any > 1e-9) | {inv_rate_strict:.4} | — (noise diag) | |\n"
    ));
    s.push_str(&format!(
        "| ↳ backwards-step magnitude med / p90 | {inv_mag_med:.2} / {inv_mag_p90:.2} | score-pts | |\n"
    ));
    s.push_str(&format!(
        "| codec-saturated (identical encode) | {codec_sat:.4} | — (codec ceiling) | |\n"
    ));
    s.push_str(&format!(
        "| flat / clamp dead-zone (distinct feats, \\|Δ\\|≤1e-9) | {flat:.4} | G3 ≤ 0.05 | {} |\n",
        if flat <= 0.05 { "✓" } else { "✗" }
    ));
    s.push_str(&format!(
        "| monotonicity (1 − inversions) | {mono:.4} | G3 ≥ 0.93 | {} |\n",
        if mono >= 0.93 { "✓" } else { "✗" }
    ));
    s.push_str(&format!(
        "| dial p5 / p95 | {p5:.1} / {p95:.1} | G1 p5≤25 ∧ p95≥85 | {} |\n",
        if p5 <= 25.0 && p95 >= 85.0 {
            "✓"
        } else {
            "✗"
        }
    ));
    s.push_str(&format!(
        "| G1 soft / G3 soft | {g1:.2} / {g3:.2} | (1.0 = full pass) | |\n\n"
    ));
    s.push_str("Per-codec inversions / flat-clamp + representable config range:\n\n");
    s.push_str(
        "| codec | param | min..max | n_curves | n_pairs | inversions | flat | monotonicity | score @worst→@best |\n",
    );
    s.push_str("|---|---|---|--:|--:|--:|--:|--:|---|\n");
    for (codec, c) in &per_codec {
        let inv = if c[0] > 0 {
            c[1] as f64 / c[0] as f64
        } else {
            f64::NAN
        };
        let m = if c[0] > 0 {
            1.0 - c[1] as f64 / c[0] as f64
        } else {
            f64::NAN
        };
        let t = if c[0] > 0 {
            c[2] as f64 / c[0] as f64
        } else {
            f64::NAN
        };
        let e = pext.get_mut(codec);
        let (kind, range, dial) = match e {
            Some(e) => {
                let w = median(&mut e.score_at_worst);
                let b = median(&mut e.score_at_best);
                let rng = if e.kind == "distance" {
                    // distance: report the full representable distance span
                    format!("{:.2}..{:.2}", e.lo_param, e.hi_param)
                } else {
                    format!("{:.0}..{:.0}", e.lo_param, e.hi_param)
                };
                (e.kind.clone(), rng, format!("{w:.1} → {b:.1}"))
            }
            None => ("q".to_string(), "—".to_string(), "—".to_string()),
        };
        s.push_str(&format!(
            "| {codec} | {kind} | {range} | {} | {} | {inv:.4} | {t:.4} | {m:.4} | {dial} |\n",
            c[3], c[0]
        ));
    }
    s.push_str(
        "\n_`param`/`min..max` = the native codec config axis and its representable range \
         in the grid (integer quality for q-codecs; butteraugli distance for JXL — lower \
         distance = higher quality). `score @worst→@best` = median dial score at the \
         lowest- and highest-quality representable config (for distance, worst = max \
         distance). **inversions** = fraction of adjacent-q pairs where the score went \
         BACKWARDS by more than 0.5 score-pt (higher quality scored materially lower — a \
         real ranking error; the gated metric); **flat** = distinct-feature pairs with \
         identical output (\\|Δ\\|≤1e-9 — a metric dead-zone). Pairs where the CODEC emitted \
         an identical image at two q (near-identical features — zenjpeg/webp quality \
         ceiling) are split into a separate **codec-saturated** bucket and are NOT counted \
         as a bake dead-zone. The aggregate table additionally breaks out the strict \
         (any-backwards) rate and the backwards-step magnitude distribution, plus a \
         sub-resolution bucket (0<\\|Δ\\|≤0.5 pt) that is EXPECTED on the densified \
         near-lossless grid (adjacent configs are sub-JND apart, so the dial correctly \
         barely moves) and is NOT gated. monotonicity = 1 − inversions. Densified grid: \
         q0 + step-1 q90→100 + fractional near-lossless q for q-codecs (96.5..99.9) + JND \
         zone + jxl-in-butteraugli-distance (0→0.3 step .025, 0.3→1 step .05, 1→3 step .2, \
         13→25 step 2; q-equiv = 100 − 4·distance)._\n",
    );
    s
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

    // ── DIAL panel (codec-target G1/G3) — runs every time, native Rust ──
    // The second mandatory half of the eval (docs/EVAL_PANEL_REQUIREMENT.md):
    // monotonicity + tied + dial range on the densified multi-codec grid.
    buf.push_str(&dial_panel(
        &model,
        has_transforms,
        n_inputs,
        &args.dial_grid,
    ));

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
