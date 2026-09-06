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
//!     /mnt/v/zen/zensim-training/2026-08-30-full-features-372/  (default since
//!     2026-08-30; file names are deliberately the 2026-05-15 ones — the ROOT
//!     carries the date. The old root stays valid as a STORED-ERA read.)
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
//!                  [--features-root /mnt/v/zen/zensim-training/2026-08-30-full-features-372]
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

use rayon::prelude::*;
use zenpredict::{Model, Predictor};

use zensim_validate::bands;
/// The G-ADDR owner — also the home of the 2026-09-05 two-reference inversion
/// rule (`encoder_inversion`), which the DIAL census and the contract share.
use zensim_validate::dial_addressability as gaddr;
use zensim_validate::eval_report;
use zensim_validate::eval_roots::{
    DEFAULT_FEATURES_ROOT_372, KONJND_JPEG504_372_SLOTS, era_of, resolve_slot,
};
use zensim_validate::panel::{
    Orientation, PerGroupSrocc, compute_panel, per_group_srocc, rescale_logistic, spearman,
};
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
    //
    // Parallel over `i` (the outer pair index). Each `i` owns the contiguous
    // pair-index run `[base_i, base_i + (n-1-i))`, so its stride hits are a
    // pure function of `i` — the same pairs are selected as the sequential
    // sweep, and concatenating the per-`i` outputs in ascending `i` rebuilds
    // the identical `samples` vector. (It is then sorted, so even the order
    // would not matter; keeping it identical costs nothing and keeps the
    // tie-averaging below reading the same runs.)
    let mut samples: Vec<(f64, bool)> = (0..n)
        .into_par_iter()
        .map(|i| {
            // pair_idx of (i, i+1) = sum_{k<i} (n-1-k) = i*(2n-i-1)/2
            let base = i * (2 * n - i - 1) / 2;
            let mut local: Vec<(f64, bool)> = Vec::new();
            for j in (i + 1)..n {
                let pair_idx = base + (j - i - 1);
                if pair_idx.is_multiple_of(stride) {
                    let metric_gap = (predicted[i] - predicted[j]).abs();
                    let human_gap = (human[i] - human[j]).abs();
                    if metric_gap.is_finite() && human_gap.is_finite() {
                        local.push((metric_gap, human_gap > diff_threshold));
                    }
                }
            }
            local
        })
        // Collect the per-`i` runs first, then concatenate in ascending `i`.
        // Explicit rather than `.flatten()` so the ordering guarantee is
        // syntactic, not a property of rayon's collect.
        .collect::<Vec<Vec<(f64, bool)>>>()
        .into_iter()
        .flatten()
        .collect();
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
    extract_hybrid_head, extract_minmax_head, extract_per_sample_alpha_head,
    extract_tanh_output_head_scale, score_row, score_row_minmax,
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
    /// Parquet path.
    ///
    /// **Normally RELATIVE — a slot under `<features_root>/`.** An absolute
    /// path here silently opts the corpus OUT of `--features-root`, because
    /// `Path::join` discards its argument's root when the argument is
    /// absolute. That is a real hazard, not a style point: a reproduction
    /// pointed at a different root would load *some* corpora from there and
    /// the rest from wherever this string says, without being told.
    ///
    /// So an absolute slot must be listed in [`PINNED_OUTSIDE_FEATURES_ROOT`]
    /// with a reason, and `corpus_slots_are_relative_or_declared_pinned`
    /// fails the build otherwise. The provenance block prints every resolved
    /// path + sha256, so the mix is at least visible in the report.
    filename: &'static str,
    /// Slots preferred OVER [`Self::filename`], tried in order; the first that
    /// exists under the features root wins, and `filename` is the last resort.
    ///
    /// Exists because a corpus can have more than one ruler on disk under the
    /// same name. Resolving to `filename` when this list is non-empty means a
    /// preferred ruler was MISSING, which
    /// [`resolve_372_slot`] announces loudly — a number read on a superseded
    /// or diluted ruler must never be indistinguishable from the real one
    /// (ADD156 ship audit, defect D2).
    ///
    /// Empty for every corpus with a single ruler.
    preferred_slots: &'static [&'static str],
    /// Per-band partitioning enabled? AIC-3 has 600 pairs in a JND
    /// step grid; rank-based per-band stats collapse to 0 on shared
    /// scores (see dataset_metric_baseline.rs comment at L454-471).
    enable_per_band: bool,
}

/// The DIAL-panel grid this binary means when nobody says otherwise.
///
/// # Why this is a named constant with a pinned hash
///
/// Three dial grids sit in that directory and they are NOT interchangeable:
///
/// | file | sha256 | state |
/// |---|---|---|
/// | `dial_grid_372col_2026-05-29.parquet` | `f1156924…` | **two known defects** |
/// | `…_quarantined.parquet` | `b5d27f21…` | fixes defect 1, superseded |
/// | `…_quarantined_v2.parquet` | `6546c43e…` | fixes both — **this one** |
///
/// The original carries (1) the w11 extraction corruption — 9 of 115 ladders
/// with bit-constant garbage in the masked/IW block (f228..f371 at 34..489 vs
/// a healthy 0.003..0.025), from the `zensim-gpu` odd-dimension pathology, so
/// *"any per-ladder dial number on them (any bake, any date since 2026-05-29)
/// is garbage-input scoring"*; and (2) 33 JXL cells at butteraugli distance
/// 0.025 encoded before jxl-encoder `eeb52735`, measured at 37× the distortion
/// of the healthy d≥0.05 ceiling — backwards from the monotone near-lossless
/// trend. `benchmarks/eval_grids_2026-05-29.pointer.md` documents both and says
/// to use `_quarantined_v2`.
///
/// **Until 2026-07-15 the default here was the original.** Both defects were
/// found, documented, and *fixed by building the quarantined grids* — and the
/// default was never switched, so every default run kept scoring against the
/// bad one. Zero code referenced `_quarantined_v2`; only prose did. That is the
/// same shape as the three other rules this repo declared and never checked
/// (see `docs/REPRODUCIBILITY.md` §5) — the fix existed, the wiring did not.
///
/// The hash is pinned so a swapped or rebuilt grid cannot pass unnoticed:
/// `bake_verdict` warns when the grid it loaded is not this one. It warns
/// rather than fails because `--dial-grid` is a legitimate experiment knob —
/// but an unremarked swap is exactly how the corrupt grid stayed default for
/// six weeks.
const CANONICAL_DIAL_GRID: &str = "/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29_quarantined_v2.parquet";

/// sha256 of [`CANONICAL_DIAL_GRID`] (4,424 rows; both defects dropped).
const CANONICAL_DIAL_GRID_SHA256: &str =
    "6546c43e6d9572dcf0740c6346cd604fd8cd3ff01ee2f7031aca998fd8fec2bd";

/// Corpora whose slot is deliberately absolute, i.e. NOT under
/// `--features-root`. Each needs a reason; the test below rejects any absolute
/// slot that is not listed here.
///
/// EMPTY since 2026-08-05 — the goal state its own doc always named. The
/// last entry was `hf_nearlossless`, pinned because its parquet lives in the
/// canonical-2026-07-15 set rather than the 2026-05-15-full-features root.
/// Its slot is now root-relative like every other corpus: the parquet is
/// symlinked into the default root (the same pattern the 720 root uses for
/// the nonphoto/imazen26 NN-joined evals), a missing file is a HARD error
/// with a restore command (never a silent drop), and the 944-era HF-NL axis
/// has its own root-relative corpus (`hfnlproxy`, SOTA-944 §1b) — so nothing
/// the pin protected is unguarded. Keep the machinery: any future absolute
/// slot must be declared here with a reason.
#[allow(dead_code)] // test-only consumer (the slot audit below); empty in the goal state
const PINNED_OUTSIDE_FEATURES_ROOT: &[(&str, &str)] = &[];

/// 720-regime (`feature-regime-v2`) feature-parquet slot for a corpus: the
/// v1-372 space (`f0..f371`) ++ the appended v2-348 block (`f372..f719`).
/// `None` = no 720 extraction exists yet, so the corpus is skipped under
/// `--regime 720`. All live under [`DEFAULT_FEATURES_ROOT_720`]; the
/// nonphoto/imazen26 NN-joined evals are symlinked into that root.
///
/// The 372 loader is already width-dynamic (`parquet_loader::load_parquet`
/// counts contiguous `f0..fN`), so 720 support is purely this filename map +
/// the regime flag — the scorer reads the bake's own `n_inputs`.
fn slot_720(name: &str) -> Option<&'static str> {
    Some(match name {
        "cid22" => "ext_cid22val.parquet",
        "kadid" => "ext_kadid.parquet",
        "tid" => "ext_tid.parquet",
        "csiq" => "ext_csiq.parquet",
        "live" => "ext_live.parquet",
        "konjnd" => "ext_konjnd_jpeg_val.parquet",
        "aic3" => "ext_aic3.parquet",
        "aic4" => "ext_aic4.parquet",
        "nonphoto" => "ext_nonphoto_720_nn_full.parquet",
        "imazen26" => "ext_imazen26_720_nn_full.parquet",
        "sdr25" => "ext_sdr25.parquet",
        // 944-root-only (SOTA-944 §1b): the near-lossless-band TEST-view
        // proxy; under roots without the slice the load fails loud and the
        // corpus is skipped, same pattern as sdr25 under the 720 root.
        "hfnlproxy" => "ext_hfnlproxy.parquet",
        // pipal, hf_nearlossless: no 720 extraction yet.
        _ => return None,
    })
}

/// What extraction regime does a features root declare? Reads
/// `<root>/_MANIFEST.json`'s top-level `"regime"` string (written by
/// `scripts/canonical_corpus/promote_ext944_canonical.py` since 2026-08-30).
/// `None` = the root says nothing, which callers must treat as the historical
/// folded default — never as "safe".
fn root_declared_regime(root: &Path) -> Option<String> {
    let txt = std::fs::read_to_string(root.join("_MANIFEST.json")).ok()?;
    // Deliberately a narrow scan rather than a serde model: this file has
    // several historical shapes and the guard only needs one string.
    let key = "\"regime\"";
    let i = txt.find(key)?;
    let rest = &txt[i + key.len()..];
    let c = rest.find(':')?;
    let after = &rest[c + 1..];
    let q1 = after.find('"')?;
    let after = &after[q1 + 1..];
    let q2 = after.find('"')?;
    Some(after[..q2].to_string())
}

/// Root-aware slot resolution for `--regime 720`-class roots. The 944 root
/// (`ext944-canonical-2026-08-01`) carries the canonical-test-view slices
/// `ext_nonphoto.parquet` / `ext_imazen26.parquet` (built by
/// `scripts/canonical_corpus/build_eval_slices_944.py` per the FULL_EVAL
/// "924-era eval slices" rule — origin-{7,9} TEST views, class-filtered);
/// the 720 root keeps its legacy NN-joined tables. Prefer the modern name
/// when it exists under THIS root, else fall back to the legacy filename —
/// so one binary serves both roots and a missing corpus still fails loud
/// downstream (never a silent skip).
/// Resolve a corpus's slot under a 372 features root — a thin adapter over
/// [`zensim_validate::eval_roots::resolve_slot`], which owns the rule and the
/// KonJND ruler list that `bake_compare` also reads.
///
/// Both `render_corpus` and the load/validate path resolve through here, so
/// the two cannot drift (they previously carried a "keep these two in sync"
/// comment and an inlined `corpus.filename`).
fn resolve_372_slot(corpus: &Corpus, root: &Path) -> zensim_validate::eval_roots::ResolvedSlot {
    resolve_slot(corpus.preferred_slots, corpus.filename, root)
}

/// State the ruler, the same way the features-root era line states the era.
///
/// Emitted for every corpus with more than one ruler on disk, so a KonJND
/// number is never again published without the file it was read on being
/// visible in the run's own output (D2).
fn ruler_lines(corpora: &[&Corpus], root: &Path) -> Vec<String> {
    corpora
        .iter()
        .filter(|c| !c.preferred_slots.is_empty())
        .map(|c| {
            let r = resolve_372_slot(c, root);
            if r.degraded {
                format!(
                    "bake_verdict: ⚠ corpus ruler — {} read on `{}`, a LAST-RESORT slot: none \
                     of its preferred rulers ({}) exist under {}. For KonJND that file is the \
                     DILUTED 1,008-ref build (JPEG+BPG) while 720/944-class rows score the \
                     JPEG-504 half — the two rulers can rank two models in OPPOSITE order. \
                     Do not compare this number across classes.",
                    c.display,
                    r.file,
                    c.preferred_slots.join(", "),
                    root.display()
                )
            } else {
                format!(
                    "bake_verdict: corpus ruler — {} read on `{}`",
                    c.display, r.file
                )
            }
        })
        .collect()
}

fn slot_720_file(name: &str, root: &Path) -> Option<String> {
    let legacy = slot_720(name)?;
    if matches!(name, "nonphoto" | "imazen26") {
        let modern = format!("ext_{name}.parquet");
        if root.join(&modern).exists() {
            return Some(modern);
        }
    }
    Some(legacy.to_string())
}

/// Default `--features-root` for `--regime 720`.
const DEFAULT_FEATURES_ROOT_720: &str = zensim_validate::eval_roots::FEATURES_ROOT_720;
/// Default dial + corruption grids for `--regime 720` (720-wide re-extractions).
const DEFAULT_DIAL_GRID_720: &str =
    "/mnt/v/output/zensim/v2-eval-720-2026-07-22/dial_grid_720col_2026-07-22.parquet";
const DEFAULT_CORRUPTION_GRID_720: &str =
    "/mnt/v/output/zensim/v2-eval-720-2026-07-22/corruption_grid_720col_2026-07-22.parquet";
/// Default multi-metric per-pair source for `--full-json`: the KADIS-720 metric
/// parquet carries `f0..f719` plus `score_{ssim2,butteraugli_max,cvvdp,...}` per
/// cell, so a bake's `(prediction, {ssim2, butter, cvvdp})` scatter can be
/// sampled from it. A <=372-input bake scores over its first `n_inputs`
/// features, so this source works in both regimes.
const DEFAULT_PERPAIR_METRICS: &str =
    "/mnt/v/zen/zensim-training/kadis-720-2026-07-24/kadis700k_720.parquet";
/// Metric columns requested from the per-pair source (a superset — only those
/// present are emitted). Mapped to the schema keys ssim2 / butter / cvvdp.
const PERPAIR_METRIC_COLS: &[(&str, &str)] = &[
    ("score_ssim2_gpu", "ssim2"),
    ("score_butteraugli_max_gpu", "butter"),
    ("score_cvvdp_cpu_imazen_v0_1_0", "cvvdp"),
];

// ============================================================================
// `--regime 944` — the SOTA-944 campaign invocation as ONE preset
// ============================================================================
//
// The SOTA-944 campaign (benchmarks/sota944_campaign_2026-08-03.md §0) ran
// every cell through one frozen bake_verdict invocation: the ext944 canonical
// feature root, the 944 dial + corruption grids, and a 12-corpus list. That
// invocation was assembled BY HAND in two places (scripts/sota944_verdict.sh
// and scripts/run_full_eval.sh), and a third hand-assembled wrapper with a
// SHORTER `--corpora` list is exactly how the published `EM4_mask2_kw0.15_s42`
// HF-NL number came to be wrong (the corpus was silently absent from that
// run, see the campaign doc's "Corrections" section). This preset makes the
// bare invocation correct: `bake_verdict --bake X --regime 944` resolves
// everything below, and a corpus can only be dropped by an EXPLICIT
// `--corpora` override.
//
// The 944 regime shares the `--regime 720` slot mechanics (`slot_720_file`
// resolves the modern `ext_imazen26`/`ext_nonphoto` TEST-view slices under
// this root); only the data roots, the per-pair source, the default corpus
// list, and the reported regime label differ.

/// Default `--features-root` for `--regime 944` (the ext944 canonical legs +
/// TEST-view eval slices; `_MANIFEST.json` carries per-file sha256s).
const DEFAULT_FEATURES_ROOT_944: &str = zensim_validate::eval_roots::FEATURES_ROOT_944;
/// Default dial + corruption grids for `--regime 944`.
const DEFAULT_DIAL_GRID_944: &str =
    "/mnt/v/output/zensim/v2-eval-944-2026-08-01/dial_grid_944col_2026-08-01.parquet";
const DEFAULT_CORRUPTION_GRID_944: &str =
    "/mnt/v/output/zensim/v2-eval-944-2026-08-01/corruption_grid_944col_2026-08-01.parquet";
/// Default multi-metric per-pair source for `--regime 944` (`--full-json`
/// scatter block): the kadis-944 sibling of [`DEFAULT_PERPAIR_METRICS`].
/// The 720-wide default cannot serve a 944 bake (n_features < n_inputs ⇒ the
/// block is skipped with a warning), which is why run_full_eval.sh's 944 case
/// swapped it explicitly. Overridable; a non-existent path skips the block.
const DEFAULT_PERPAIR_METRICS_944: &str =
    "/mnt/v/zen/zensim-training/kadis-944-2026-08-01/kadis700k_944.parquet";
/// The FROZEN SOTA-944 campaign corpus list, in the campaign's report order —
/// what `scripts/sota944_verdict.sh` passed on every one of the campaign's
/// cells. This is the default `--corpora` under `--regime 944`, so a bare
/// invocation cannot silently omit a corpus (the wrong-published-numbers
/// class). Changing this list is a CONSCIOUS act: the
/// `regime_944_default_corpora_match_the_frozen_campaign_list` test pins it.
const SOTA944_CORPORA: &str =
    "cid22,kadid,tid,konjnd,aic3,aic4,csiq,live,sdr25,imazen26,nonphoto,hfnlproxy";

const CORPORA: &[Corpus] = &[
    Corpus {
        name: "cid22",
        display: "CID22",
        filename: "cid22_features_372col_2026-05-15.parquet",
        preferred_slots: &[],
        enable_per_band: true,
    },
    Corpus {
        name: "kadid",
        display: "KADIK10k",
        filename: "kadid_features_372col_2026-05-15.parquet",
        preferred_slots: &[],
        enable_per_band: true,
    },
    Corpus {
        name: "tid",
        display: "TID2013",
        filename: "tid_features_372col_2026-05-15.parquet",
        preferred_slots: &[],
        enable_per_band: true,
    },
    Corpus {
        name: "csiq",
        display: "CSIQ",
        // CSIQ (30 refs × 6 distortions incl. JPEG/JPEG2000; 866 pairs). DMOS
        // stored as `human_score = 1 − DMOS` → quality-oriented [0,1] (matches
        // kadid/tid). Classic FR benchmark; added 2026-07-18 (FR-corpus expansion).
        filename: "csiq_features_372col_2026-07-18.parquet",
        preferred_slots: &[],
        enable_per_band: true,
    },
    Corpus {
        name: "pipal",
        display: "PIPAL",
        // PIPAL (200 refs × ~109 GAN/restoration distortions; 21,800 pairs; ELO MOS
        // → human_score quality-oriented [0,1]). Held-out for our bakes (train set was
        // safesyn/cid22_train/kadid/tid). A DISTINCT, harder axis: algorithmic/GAN
        // distortions, not compression — winner ≈0.62 vs 0.96 on CSIQ. Added 2026-07-18.
        filename: "pipal_features_372col_2026-07-18.parquet",
        preferred_slots: &[],
        enable_per_band: false,
    },
    Corpus {
        name: "live",
        display: "LIVE-R2",
        // LIVE IQA Release 2 (Sheikh 2006): 29 refs × {jp2k,jpeg,wn,gblur,fastfading};
        // 779 real distortions. Realigned DMOS → `human_score = 1 − dmos_new/100`
        // (quality-oriented [0,1]); per-sample `sigma` (dmos_std) carried for Z-RMSE.
        // Classic FR compression benchmark (JPEG + JPEG2000). HELD-OUT for our bakes.
        // Added 2026-07-18 (FR-corpus expansion). Pairs builder:
        // scripts/canonical_corpus/build_fr_corpus_pairs.py live.
        filename: "live_features_372col_2026-07-18.parquet",
        preferred_slots: &[],
        enable_per_band: true,
    },
    Corpus {
        name: "konjnd",
        display: "KonJND-1k",
        // ⚠ TWO RULERS SHARE THIS SLOT (ADD156 ship audit, defect D2).
        // `konjnd_features_372col_2026-05-15.parquet` holds all 1,008 refs —
        // the JPEG **and BPG** halves — while every 720/944-class row scores
        // the JPEG-504 half only. Reading the diluted file INVERTS
        // cross-model comparisons: on the same root, same binary, same code
        // path, ADD156 reads 0.4462 and shipped B 0.6497 (B by +0.204),
        // while on the JPEG-504 ruler ADD156 reads 0.5332 and B 0.5194
        // (ADD156 by +0.014). `filename` is kept as the LAST-RESORT slot so
        // a root that only has the diluted file still produces a number —
        // but it is announced loudly, never silently.
        filename: "konjnd_features_372col_2026-05-15.parquet",
        // Same ruler, root-specific dates: the 2026-08-30 roots carry the
        // 08-30 build, the 2026-05-15 root the 08-29 one. First hit wins.
        preferred_slots: KONJND_JPEG504_372_SLOTS,
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
        preferred_slots: &[],
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
        preferred_slots: &[],
        enable_per_band: false,
    },
    Corpus {
        name: "nonphoto",
        display: "imazen-26 non-photo (held-out)",
        // THE non-photographic axis (added 2026-07-15). All other corpora are
        // photographic/near-photographic, so a bake blind to screen/UI/document/
        // line-art/AI-gen content scores ~identically on them — the blindness is
        // INVISIBLE (see §8.34/§8.35). This is the standing detector: a held-out
        // val-origin ({1,3,5}) 10k subsample of bigcodec/imazen-26 (real modern-codec
        // distortions on 21 diverse content categories); `human_score` = ssim2/100, so
        // SROCC here = rank-agreement with ssim2 on diverse content. A bake that
        // ranks photographic content well but craters here (§8.33/B ≈ 0.856 vs a
        // diverse-trained ≈ 0.93) is content-blind — the G-NP gate flags it.
        // RELATIVE (fixed 2026-07-15). This slot used to be the absolute path
        // `/mnt/v/zen/zensim-training/2026-05-15-full-features/nonphoto_...parquet`
        // with a note that `Path::join` discards `features_root` for an absolute
        // argument — i.e. `--features-root` silently did not apply here. The file
        // is IN the default root, so the absolute path bought nothing and cost the
        // flag its meaning: a reproduction pointed at another root got this corpus
        // from the original one without being told.
        filename: "nonphoto_features_372col_2026-07-15.parquet",
        preferred_slots: &[],
        enable_per_band: false,
    },
    Corpus {
        name: "imazen26",
        display: "imazen-26 real-codec (held-out, ssim2)",
        // BROAD ssim2-agreement axis (2026-07-16, user "eval on imazen26"): a
        // 120k stride subsample of the canonical-picker-2026-06-27 held-out TEST
        // split (origin {7,9}) across 4 real lossy codecs (zenjpeg/zenavif/zenjxl/
        // zenwebp); `human_score` = score_ssim2, so SROCC here = rank-agreement
        // with ssim2 on real modern-codec output. The nonphoto slot's all-content,
        // all-codec sibling. In the default features root.
        filename: "imazen26_test_120k_2026-07-16.parquet",
        preferred_slots: &[],
        enable_per_band: false,
    },
    Corpus {
        name: "sdr25",
        display: "JPEG-AI SDR25 (HQ-zone human)",
        // High-quality-zone HUMAN data (q75-100 triplets) — one of the
        // CLAUDE.md "untapped local human datasets". Never trained on;
        // added 2026-07-29 as a SELECTION/eval leg (the bimodal-seed
        // campaigns need an oracle that is neither a training group nor a
        // product gate). 720-root has no extraction — the 924 root does
        // (ext_sdr25.parquet); under the default root the load fails loud
        // and the corpus is skipped.
        filename: "ext_sdr25.parquet",
        preferred_slots: &[],
        enable_per_band: false,
    },
    Corpus {
        name: "hfnlproxy",
        display: "HF-NL-proxy (944 TEST views, ssim2>=91 band)",
        // SOTA-944 §1b registered HF-NL substitute: the true hf_nearlossless
        // corpus exists only at 372 (bitstreams unpersisted), so the 944
        // near-lossless axis reads the ssim2>=91 band of the held-out
        // bigcodec TEST views, per-ref (READ per_ref_mean, NOT pooled —
        // same confound as the hf corpus's 0.204-pooled/0.916-per-ref).
        // 944-root-only; built by build_eval_slices_944.py.
        filename: "ext_hfnlproxy.parquet",
        preferred_slots: &[],
        enable_per_band: false,
    },
    Corpus {
        name: "hf_nearlossless",
        display: "HF near-lossless (held-out refs)",
        // THE near-lossless axis (added 2026-07-15). 72% of its cells sit above
        // ssim2 95, where safesyn has 1.86% — so it covers the regime the
        // training distribution lacks, and §8.39 measured that a bake trained
        // without it ranks this regime BACKWARDS (per-ref -0.144, 60% of refs
        // negative) while every pooled number looks fine.
        //
        // IT IS AN ssim2 SELF-TARGET, NOT HUMAN DATA (measured 2026-09-01,
        // `benchmarks/ssim2_replacement_bar_2026-08-31.md` APPENDIX A). Its
        // `human_score` column IS `ssim2_gpu / 100` — exactly, in float
        // equality, on 1200/1200 rows. So a number here is AGREEMENT with
        // ssim2 and can never be a win over it: scored with ssim2 as the
        // predictor this axis reads pooled SROCC 1.0000 and per-ref mean
        // 1.0000. Read it as a health indicator alongside `nonphoto` /
        // `imazen26` / `hfnlproxy`, never as evidence of superiority.
        //
        // It is also 372-only FOREVER, not "not yet": the 1,200 distorted JXL
        // bitstreams it would have to be re-extracted from were never persisted
        // (`encoded_filename` blank on every row of the sweep's own pareto.tsv;
        // both `refit/distorted/` mirrors empty). The parquet is byte-identical
        // at all four roots that serve it — copied forward, re-extracted never.
        //
        // READ THE per-ref COLUMN, NOT POOLED. The ladder moves ~0.92 ssim2 pts
        // within an image against ~6 pts of cross-image spread, so pooled SROCC
        // reads +0.204 where per-ref reads +0.916 on the same rows — pooled is
        // dominated by between-image scale and is nearly blind to the ladder.
        // Same confound as the documented AIC-3 "0.79 pooled / 0.93 per-ref".
        //
        // 50 refs x 6 distortions, ref-level split (the 150 train refs are
        // disjoint), so this is honest holdout for a bake trained on
        // hf_nearlossless_train.parquet.
        //
        // Root-relative since 2026-08-05 (was the last absolute slot — see
        // PINNED_OUTSIDE_FEATURES_ROOT). The canonical bytes stay in
        // canonical-2026-07-15/train/; the default root reaches them via a
        // symlink (sha-verified identical), same pattern as the 720 root's
        // nonphoto/imazen26 slots. A root without the file fails LOUD with
        // the R2 restore command — the axis cannot be silently dropped.
        filename: "hf_nearlossless_val.parquet",
        preferred_slots: &[],
        // Everything here lives in ssim2 91..100, i.e. one width-10 band. A
        // 10-band split would put all 300 rows in B9 and report nine empties.
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
    /// `--ensemble a.bin,b.bin,...`: score an equal-weight ENSEMBLE of bakes as
    /// one model (mean of the members' raw predictions per row) through every
    /// panel this binary reports — rank, per-reference, dial, corruption. Empty
    /// = plain single-bake mode. When given, `--bake` defaults to member 0 (the
    /// provenance anchor); an explicit `--bake` must be one of the members.
    ///
    /// See [`Ensemble`] for the averaging contract and the k=1 identity.
    ensemble: Vec<PathBuf>,
    /// `--ensemble-weights w1,w2,...`: a CONVEX blend of the members instead of
    /// the equal-weight mean. Same length as `--ensemble`, non-negative, at
    /// least one strictly positive; normalised to sum 1 at parse time (the
    /// normalised vector is what the report and `--full-json` publish).
    ///
    /// `None` keeps the pre-existing unweighted `(Σ v_i)/k` accumulation
    /// **verbatim**, so every ensemble scored before this flag existed
    /// reproduces bit-for-bit. Three behavioural identities were MEASURED on
    /// the CID22 pools slice (4,292 rows) when the flag landed and are the
    /// gates that make the weighted path trustworthy:
    /// `--ensemble-weights 0.5,0.5` == the unweighted mean (bit-identical,
    /// 4292/4292 rows), `1,0` == a plain `--bake` on member 0 (bit-identical),
    /// `0,1` == a plain `--bake` on member 1 (bit-identical). The parse-side
    /// half is pinned by `ensemble_weights_parse_validate_and_normalise` and
    /// `ensemble_weights_refusals_are_loud`.
    ensemble_weights: Option<Vec<f64>>,
    corpora: Vec<&'static Corpus>,
    output: Option<PathBuf>,
    features_root: PathBuf,
    /// `feature-regime-v2`: score the 720-wide bake against 720-wide corpora
    /// (via [`slot_720`]). Selected by `--regime 720` (and by `--regime 944`,
    /// which shares the slot mechanics).
    regime_720: bool,
    /// `--regime 944`: the SOTA-944 campaign preset — same slot mechanics as
    /// 720, but the defaults resolve to the ext944 roots/grids/per-pair source
    /// and the frozen campaign corpus list (see [`SOTA944_CORPORA`]). Only
    /// affects DEFAULTS + the reported regime label; explicit flags win.
    regime_944: bool,
    /// Diagnostic: dump per-row `human<TAB>pred` (parquet row order) to this
    /// path. Used by the AIC-3 CVVDP-feature spike to compute per-ref SROCC
    /// (which the aggregate panel does not split out).
    per_pair_output: Option<PathBuf>,
    /// With `--per-pair-output`: append a third `ref` column (the loader's
    /// interned per-reference group id) so per-ref statistics can be computed
    /// downstream without re-deriving grouping from row order. Additive and
    /// opt-in — the 2-column default is exact-unpacked by three committed
    /// consumers (metric_compare_report / bake_report / bandwise_dashboard).
    /// Fails loud if the dumped corpus carries no ref identity.
    /// (SOTA-944 appendix O, 2026-08-05.)
    per_pair_refs: bool,
    /// DIAL-panel grid parquet (`image_id, codec, q, f0..f371`). Default
    /// is the canonical densified multi-codec grid; override with
    /// `--dial-grid` or `ZENSIM_DIAL_GRID`. When the file is absent the
    /// dial panel is skipped with a loud note (it cannot be recomputed
    /// without the stored feature grid — fetch from R2 eval-grids/).
    dial_grid: PathBuf,
    /// Self-contained HTML report path. When set, the full report (every
    /// section printed to console) is ALSO rendered to a single browsable
    /// HTML file with a table-of-contents, styled tables, and inline-SVG
    /// charts — no external assets. The "big html report" half of the
    /// unified metric-eval command.
    html: Option<PathBuf>,
    /// `--json <path>`: machine-readable per-corpus panel for the comparative
    /// dashboard. The structured counterpart to `--output` markdown; consumers
    /// parse this instead of the report.
    json: Option<PathBuf>,
    /// Severity-ramp feature grid (`image_path, q, feat_0..`) where
    /// `q = dist_type*10 + severity_level`. Enables the severity-ramp
    /// monotonicity section (distortion dial). The grid's feature regime
    /// MUST match the bake (PU21-u8 vs PU-linear). Absent → section skipped.
    ramp_grid: Option<PathBuf>,
    /// Reference bake for the per-zone dial-agreement section (§8.20):
    /// scores both bakes on the dial grid, buckets by the reference bake's
    /// dial in 5-pt zones, and reports the candidate's mean-Δ / RMSE / rank
    /// per zone. Absent → section skipped.
    compare: Option<PathBuf>,
    /// Corruption-gate grid parquet (`entry, f0..`). Defaults to the
    /// canonical grid; the section auto-runs when the file is present and
    /// the feature count matches the bake, else skips silently.
    corruption_grid: PathBuf,
    /// `--dial-peer-scores <label>=<tsv>`: run the DIAL panel (and its
    /// ladder-inversion zones) on EXTERNALLY SUPPLIED per-cell scores
    /// instead of the bake's forward. The TSV is the exact shape
    /// `ZENSIM_DIAL_PRED_OUT` dumps — `image_id\tcodec\tq\tpred` — so the
    /// dump/read pair round-trips, which is how the mode is gated.
    ///
    /// Why it exists: a REFERENCE METRIC (ssim2, butteraugli, cvvdp) has no
    /// bake, so before this the dial panel could not be run on one at all and
    /// the board's peer rows carried a hand-rolled, self-described
    /// "presentation-grade" monotonicity with `tied_pct: null` and no zones.
    /// "Is zensim's dial more monotone than ssim2's?" is a first-order
    /// product question and it had no owner-computed answer. This makes the
    /// opponent measurable on the SAME instrument as every bake — same
    /// five-bucket rule, same materiality threshold, same zone cuts —
    /// which is the only way the comparison means anything.
    ///
    /// The rank panel still describes `--bake`; only the DIAL section
    /// changes. Every rendering of that section is stamped with the peer
    /// label, and `--full-json` / `--fulleval` are REFUSED in this mode so a
    /// peer dial can never be written into a bake's board row.
    dial_peer_scores: Option<(String, PathBuf)>,
    /// `--inversion-truth single|agree`: which reference set decides whether a
    /// material backwards rung counts against the DIAL.
    ///
    /// USER RULING 2026-09-05 — *"for inversions, we should choose say ssim2
    /// and butter and only flag true inversions where they agree"*. Under
    /// `agree` (the default), a backwards rung on a pair where BOTH reference
    /// metrics independently call the higher setting worse is charged to the
    /// ENCODER and leaves `mono` / the zone census. `single` reproduces the
    /// pre-ruling reading and is retained so a published number stays
    /// reproducible. The rule itself lives in
    /// [`zensim_validate::dial_addressability::encoder_inversion`] — one
    /// function, so the G-ADDR contract and this census cannot drift.
    inversion_truth: gaddr::InversionTruth,
    /// `--reference-truth <tsv>[:variant]`: the per-cell reference table
    /// `agree` needs (`image_id/codec/q/ssim2/butteraugli`, butteraugli in
    /// DISTANCE units). Absent, `agree` is NOT MEASURABLE and the run falls
    /// back to `single` LOUDLY — never silently, because a missing table would
    /// otherwise look like a clean dial.
    reference_truth: Option<(PathBuf, gaddr::ButteraugliVariant)>,
    /// `--encoder-inversion-census <tsv>`: write EVERY encoder inversion the
    /// instrument holds, independent of any bake.
    ///
    /// The per-run table in the DIAL panel lists only the pairs THIS scorer
    /// also inverted on; a codec tracking issue wants the whole set, because a
    /// pair the encoder ran backwards on is the codec's defect whether or not
    /// one particular dial happened to notice. Same rule, same margins, same
    /// `gaddr::encoder_inversion` — only the filter differs.
    encoder_inversion_census: Option<PathBuf>,
    /// `--negtail-peer-scores <label>=<tsv>`: the NEGATIVE-TAIL probe's scores
    /// for a peer, as `entry\tpred`. Companion to `--dial-peer-scores` — it is
    /// what makes G-ADDR's floor axes measurable for a reference metric, which
    /// is a prerequisite for pinning the regression bars to one.
    ///
    /// **Why it is REQUIRED in peer mode rather than optional.** Without it a
    /// peer run measured the grid axes from the peer and the tail axes from
    /// `--bake` — a chimera nobody asked for and nothing labelled. The G-ADDR
    /// axes are now all-or-nothing per run: in peer mode an unsupplied probe
    /// axis is NOT MEASURED, never silently filled from the bake.
    negtail_peer_scores: Option<(String, PathBuf)>,
    /// `--identity-peer-scores <label>=<tsv>`: the IDENTITY probe's scores for
    /// a peer, as `entry\tpred`. Same all-or-nothing rule as
    /// `--negtail-peer-scores`; additionally the "no grid cell out-scores a
    /// perfect copy" count is only meaningful when the grid scores and the
    /// identity score come from the SAME scorer.
    identity_peer_scores: Option<(String, PathBuf)>,
    /// `--gaddr-json <path>`: write ONLY the G-ADDR block (the
    /// `dial_addressability` verdict, full f64 precision) to a standalone JSON
    /// file. Allowed in peer mode — unlike `--full-json`/`--fulleval` it is not
    /// a board row and every field in it describes the same scorer the DIAL
    /// section describes, stamped with `scorer`.
    ///
    /// This is how a REFERENCE pin set is derived: run the gate on the peer,
    /// read the `measured` block at full precision, append it to the registry.
    /// Deriving pins by re-implementing the percentile/tail math outside the
    /// owner would be exactly the duplication the no-duplicate-implementations
    /// rule forbids, and the numbers would drift the first time either moved.
    gaddr_json: Option<PathBuf>,
    /// `--gaddr-reference <name>`: grade the G-ADDR REGRESSION tier against a
    /// named pin set instead of the active one. Default (and the only value a
    /// ship decision may use) is
    /// `dial_addressability::ACTIVE_REFERENCE` = `peer_ssim2`.
    ///
    /// `shipped_b` reproduces the retired pre-2026-09-04 grading. It exists so
    /// "how would this candidate have graded under the old bars?" is a
    /// MEASUREMENT — same comparators, same rows, same owner — rather than a
    /// table someone re-derived by hand beside the gate. Every report says
    /// which pin set it used, in the caption and in the JSON.
    gaddr_reference: Option<String>,
    /// `--gaddr-tail-pins <product|retired>`: which NEGATIVE-TAIL pin set
    /// grades the G-ADDR tail rows. Default `product` — the absolute product
    /// range from the **USER RULING 2026-09-05** (*"the negative tail bar is
    /// entirely arbitrary. below -5-50"*), emitted as `A7r`/`A8r`/`A9r`.
    /// `retired` reproduces the pre-2026-09-05 grading exactly (`A7`/`A8`/`A9`
    /// barred against `peer_ssim2`'s own depth on the same probe), which is
    /// what every G-ADDR number published before that date was graded on.
    /// A1-A6 are unaffected by either value.
    gaddr_tail_pins: Option<String>,
    /// `--gaddr-value-pins <report|hard>`: which TIER the six dial-VALUE rows
    /// `A1`-`A6` sit on. `report` (default since the USER RULING 2026-09-05)
    /// measures and prints them but lets them gate nothing; `hard` restores
    /// the pre-ruling grading, where they are REGRESSION rows. The CONTRACT
    /// tier is identical either way, so the board's NOT-SHIPPABLE badge cannot
    /// move with this flag.
    gaddr_value_pins: Option<String>,
    /// `--gaddr-grid-truth <tsv>`: the REFERENCE metric's own per-cell scores
    /// on the dial grid (`image_id\tcodec\tq\tpred` — the same table
    /// `--dial-peer-scores` reads). Fills the per-codec-family `A9r` columns,
    /// which need per-ROW reference truth. Under `--floor-rule distinct`
    /// (the default) `A7r`'s bar comes from the registry and needs no flag;
    /// under `resolvable`/`spaced` this is REQUIRED — those windows are
    /// chosen from the mentor's own per-cell values, and their bar is the
    /// mentor's LIVE fraction on this same instrument (never a registry
    /// read) — so `A7r`, not only `A9r`, depends on it there.
    gaddr_grid_truth: Option<PathBuf>,
    /// `--floor-rule <distinct|resolvable|spaced>`: which window `A7r` tests
    /// per ladder. OWNER-EXTENSION, opt-in (2026-09-06) — see
    /// `zensim_validate::dial_addressability::FloorRule` and
    /// `benchmarks/ladder_floor_resolution_2026-09-05.md`. Default
    /// `distinct` is the pinned rule and is BYTE-IDENTICAL to every prior
    /// `bake_verdict` invocation. `resolvable` and `spaced` require
    /// `--gaddr-grid-truth` (their window AND their bar both come from the
    /// mentor's own per-cell scores) and are refused without it.
    floor_rule: Option<String>,
    /// `--floor-margin <f64>`: the `resolvable` rule's minimum `|Δ mentor|`
    /// between two SELECTED steps (ssim2 points on the registered
    /// instruments). Default 0.5, matching
    /// `FloorRule::RESOLVABLE_MARGIN_DEFAULT`. Ignored by `distinct`/`spaced`.
    floor_margin: Option<f64>,
    /// `--corruption-head <bake.bin>`: companion corruption-head bake — the
    /// shipping design's corruption owner (at 924 the dial's own ordering is
    /// broken by design, distributional; the head trained on negrich carries
    /// the gate). Scored on the same corruption grid; the dial-alone numbers
    /// stay in the report for honesty. Absent → dial-only, unchanged.
    corruption_head: Option<PathBuf>,
    /// `--corruption-head-threshold <score>`: the DEPLOY deadband, in the
    /// head bake's own OUTPUT units. The registered composition is
    /// `final = min(perceptual, gate)` with `gate = 100` unless
    /// `P(corruption) > T`; a head baked to emit `100*(1-P)` turns that
    /// into `head_score < 100*(1-T)`, so `T = 0.9` is `10.0` here.
    corruption_head_threshold: f64,
    /// `--negtail-probe <parquet>`: pinned negative-tail probe (`entry, f0..`)
    /// for the G-ADDR floor axis — rows whose reference metric is genuinely
    /// negative, so "does the dial still go below zero, and as deep as the
    /// shipped dial does" is a MEASUREMENT rather than an assumption. Absent
    /// → those axes are NOT MEASURED (never a silent pass).
    negtail_probe: Option<PathBuf>,
    /// `--identity-probe <parquet>`: pinned identity probe (`entry, f0..`)
    /// whose rows are `ref == dist` for the dial grid's OWN reference images,
    /// so "no codec output out-scores a perfect copy of its own reference" is
    /// checked per reference rather than against a global constant. Absent
    /// → those axes are NOT MEASURED.
    identity_probe: Option<PathBuf>,
    /// `--full-json <path>`: the unified "full-eval" JSON consumed by
    /// `scripts/run_full_eval.sh` + the summer-gauntlet dashboard. Richer than
    /// `--json` (which stays a stable per-corpus panel): rank map + dial
    /// (mono/tied/reach/dynamic_range) + corruption + a sampled multi-metric
    /// `per_pair` block. `m3_coherence` is left null for the wrapper to inject.
    full_json: Option<PathBuf>,
    /// `--fulleval <path>`: emit the COMPLETE fulleval-schema JSON directly —
    /// the `--full-json` content PLUS every M3 field the wrapper used to add
    /// (`m3_coherence`/`m3_n`/`m3_dropped_mass_pct`/`m3a_coherence`/`m3a_n`)
    /// as explicit nulls, so the emitted file IS a schema-complete
    /// `*.fulleval.json` and `run_full_eval.sh`'s jq step only ever INJECTS
    /// measured values into existing keys. This binary stays free of image
    /// I/O — the M3/M3a measurement remains `diffmap_block_coherence`'s job.
    fulleval: Option<PathBuf>,
    /// Human-readable model name embedded in `--full-json` (else the bake stem).
    name: Option<String>,
    /// Multi-metric per-pair source for `--full-json` (default [`DEFAULT_PERPAIR_METRICS`]).
    /// Provides the ssim2 / butter / cvvdp scatter. Set to a non-existent path
    /// to skip the metric per-pair block.
    perpair_metrics: PathBuf,
    /// Max rows sampled per corpus for the `per_pair` block (default 5000).
    perpair_cap: usize,
    /// `--allow-unpopulated-slots`: override the DEFAULT refusal when a bake
    /// reads slots the table does not populate AND both feature-set ids are
    /// stored rather than inferred.
    ///
    /// The refusal it lifts is not about names: it fires only when both sides
    /// carry a recorded id, i.e. when "this table lacks those columns" is a
    /// measurement. Pass this to state that feeding structural fill to those
    /// columns is intentional — an ablation, say — and expect the numbers to
    /// be about the fill, not about the model.
    allow_unpopulated_slots: bool,
    /// `--cross-regime`: override the wrong-regime refusal. By default a bake
    /// that structurally uses f156-371 is REFUSED at the `--regime 944` root,
    /// because the folded ext944 extraction feeds that block as structural
    /// zeros and the resulting numbers are plausible-looking garbage (the
    /// `ebothg_m504` board row; appendix U.R0's shipped-B 0.3862-vs-0.8764).
    /// Passing this flag states the cross-regime read is intentional; the
    /// report banner records it.
    cross_regime: bool,
    /// `--require-feature-set-match`: REFUSE (exit 2) when the bake's
    /// feature-set id disagrees with the features root's on ANY axis — width,
    /// populated slots, or extractor era (`docs/FEATURE_SET_IDS.md` §4).
    /// Default OFF: today every legacy bake reads era `unknown` because it
    /// carries no `zentrain.feature_set_id`, so the strict mode would refuse
    /// the whole board. The mismatches are always REPORTED; this flag turns
    /// the report into a gate, and becomes the default once new artifacts
    /// carry ids (design §7 step 3).
    require_feature_set_match: bool,
    /// `--print-features-root`: resolve the EVAL features root FROM THE BAKE
    /// (`zensim_validate::feature_set::resolve_features_root`), print it on
    /// stdout with the rule that produced it on stderr, and exit. Scores
    /// nothing and reads no corpus.
    ///
    /// This exists because `scripts/run_full_eval.sh` hard-coded one root per
    /// regime, so a bake trained at a non-default root (A3b, era-2xradius-4)
    /// could only be mis-read or skipped — see the module note on
    /// `resolve_features_root`. The derivation lives in the feature-set owner;
    /// this flag is only the shell's way in, so there is still ONE rule.
    print_features_root: bool,
}

fn print_usage() {
    eprintln!(
        "bake_verdict — instant V_X bake eval from pre-extracted parquet features\n\
\n\
USAGE:\n\
    bake_verdict --bake <path>\n\
                 [--corpora cid22,kadid,tid,konjnd,aic3,aic4]\n\
                 [--output <path.md>] [--html <path.html>]\n\
                 [--ramp-grid <path.parquet>] [--compare <ref-bake.bin>]\n\
                 [--features-root /mnt/v/zen/zensim-training/2026-08-30-full-features-372]\n\
                 [--ensemble a.bin,b.bin [--ensemble-weights 0.7,0.3]]\n\
\n\
DEFAULTS:\n\
    --corpora       all 6 (cid22,kadid,tid,konjnd,aic3,aic4)\n\
    --output        stdout (markdown console report)\n\
    --html          none (also emit a self-contained big HTML report)\n\
    --ramp-grid     none (severity-ramp monotonicity section)\n\
    --compare       none (per-zone dial-agreement vs a reference bake)\n\
    --corruption-grid canonical grid (negative-tail gate; auto if present)\n\
    --corruption-head none (companion head bake scored on the corruption grid)\n\
    --corruption-head-threshold 10.0 (deploy deadband in the head bake's output units)\n\
    --features-root /mnt/v/zen/zensim-training/2026-08-30-full-features-372\n\
                    (the CURRENT-extractor 372 root, default since 2026-08-30; the\n\
                    2026-05-15 root stays on disk as a valid STORED-ERA read)\n\
\n\
REGIMES (--regime 372|720|944):\n\
    372  (default) v1 feature space, 372col corpora\n\
    720  folded+append feature space; swaps unset root/grid defaults to the\n\
         720-wide variants and keeps only corpora with a 720 extraction\n\
    944  THE SOTA-944 campaign invocation as one preset: ext944 root, 944\n\
         dial/corruption grids, kadis-944 per-pair source, and the frozen\n\
         12-corpus campaign list — a bare `--bake X --regime 944` is the\n\
         complete, correct evaluation. Explicit flags override the preset.\n\
\n\
    A bake that structurally USES f156-371 is REFUSED at `--regime 944`\n\
    (the folded root zeroes that block; the numbers would be garbage —\n\
    score it at `--regime 372` instead). `--cross-regime` overrides after\n\
    stating the read is intentional.\n\
\n\
FEATURE-SET IDS (docs/FEATURE_SET_IDS.md):\n\
    Every run prints the bake's feature-set id and the root's, and reports\n\
    every disagreement (width / populated slots / extractor era). Counts are\n\
    aliases only: `944` alone has named seven different feature sets.\n\
    --require-feature-set-match turns the report into a REFUSAL (exit 2).\n\
\n\
G-ADDR FLOOR RULE (--floor-rule distinct|resolvable|spaced, owner-extension):\n\
    distinct    (default) the pinned A7r rule — literal positions 0..=K by\n\
                quality; bar is the registry-pinned mentor fraction. Every\n\
                prior invocation is `distinct` and stays byte-identical.\n\
    resolvable  skip forward past any step the mentor cannot itself resolve\n\
                (|Δ mentor| < --floor-margin, default 0.5) until K+1\n\
                mentor-resolvable steps are collected. REQUIRES\n\
                --gaddr-grid-truth (window AND bar are both LIVE-computed\n\
                from the mentor's own per-cell scores — never the registry).\n\
    spaced      the lowest setting plus the steps nearest +2 and +5 mentor\n\
                points above it. Also REQUIRES --gaddr-grid-truth.\n\
    See benchmarks/ladder_floor_resolution_2026-09-05.md for the report this\n\
    generalizes.\n"
    );
}

fn parse_args() -> Result<Args, String> {
    parse_args_from(std::env::args().skip(1))
}

/// Testable core of [`parse_args`]: same semantics, arguments injected. The
/// `--regime 944` preset-resolution tests drive this directly.
fn parse_args_from(args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut bake: Option<PathBuf> = None;
    let mut ensemble: Vec<PathBuf> = Vec::new();
    let mut ensemble_weights: Option<Vec<f64>> = None;
    let mut corpora: Option<Vec<&'static Corpus>> = None;
    let mut output: Option<PathBuf> = None;
    let mut per_pair_output: Option<PathBuf> = None;
    let mut per_pair_refs = false;
    let mut html: Option<PathBuf> = None;
    let mut json: Option<PathBuf> = None;
    let mut full_json: Option<PathBuf> = None;
    let mut fulleval: Option<PathBuf> = None;
    let mut name: Option<String> = None;
    let mut perpair_metrics: PathBuf = PathBuf::from(DEFAULT_PERPAIR_METRICS);
    let mut perpair_cap: usize = 5000;
    let mut ramp_grid: Option<PathBuf> = None;
    let mut compare: Option<PathBuf> = None;
    let mut corruption_grid: PathBuf = std::env::var("ZENSIM_CORRUPTION_GRID")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(
                "/mnt/v/output/zensim/eval_panels_2026-05-29/corruption_grid_372col_2026-05-28.parquet",
            )
        });
    let mut corruption_head: Option<PathBuf> = None;
    let mut corruption_head_threshold: f64 = 10.0;
    let mut negtail_probe: Option<PathBuf> = None;
    let mut identity_probe: Option<PathBuf> = None;
    let mut dial_peer_scores: Option<(String, PathBuf)> = None;
    let mut inversion_truth = gaddr::InversionTruth::Agree;
    let mut reference_truth: Option<(PathBuf, gaddr::ButteraugliVariant)> = None;
    let mut encoder_inversion_census: Option<PathBuf> = None;
    let mut negtail_peer_scores: Option<(String, PathBuf)> = None;
    let mut identity_peer_scores: Option<(String, PathBuf)> = None;
    let mut gaddr_json: Option<PathBuf> = None;
    let mut gaddr_reference: Option<String> = None;
    let mut gaddr_tail_pins: Option<String> = None;
    let mut gaddr_value_pins: Option<String> = None;
    let mut gaddr_grid_truth: Option<PathBuf> = None;
    let mut floor_rule: Option<String> = None;
    let mut floor_margin: Option<f64> = None;
    let mut features_root: PathBuf = PathBuf::from(DEFAULT_FEATURES_ROOT_372);
    let mut dial_grid: PathBuf = std::env::var("ZENSIM_DIAL_GRID")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(CANONICAL_DIAL_GRID));
    // `--regime 720`/`--regime 944` swap the defaults below to their wide
    // variants, but only when the user did not set them explicitly (tracked
    // here) — explicit flags always win over the preset.
    let mut regime_720 = false;
    let mut regime_944 = false;
    let mut print_features_root = false;
    let mut cross_regime = false;
    let mut allow_unpopulated_slots = false;
    let mut require_feature_set_match = false;
    let mut features_root_set = false;
    let mut dial_grid_set = std::env::var("ZENSIM_DIAL_GRID").is_ok();
    let mut corruption_grid_set = std::env::var("ZENSIM_CORRUPTION_GRID").is_ok();
    let mut perpair_metrics_set = false;
    let mut args = args;
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
                features_root_set = true;
            }
            "--regime" => {
                let v = args.next().ok_or("--regime requires 372|720|944")?;
                match v.as_str() {
                    "372" => {
                        regime_720 = false;
                        regime_944 = false;
                    }
                    "720" => {
                        regime_720 = true;
                        regime_944 = false;
                    }
                    "944" => {
                        regime_720 = true;
                        regime_944 = true;
                    }
                    other => return Err(format!("--regime must be 372|720|944, got {other:?}")),
                }
            }
            "--dial-grid" => {
                let v = args.next().ok_or("--dial-grid requires <path>")?;
                dial_grid = PathBuf::from(v);
                dial_grid_set = true;
            }
            "--dial-peer-scores" => {
                let v = args
                    .next()
                    .ok_or("--dial-peer-scores requires <label>=<path>")?;
                let (label, path) = v
                    .split_once('=')
                    .ok_or("--dial-peer-scores must be <label>=<path>")?;
                if label.trim().is_empty() {
                    return Err("--dial-peer-scores label must be non-empty".into());
                }
                dial_peer_scores = Some((label.to_string(), PathBuf::from(path)));
            }
            "--encoder-inversion-census" => {
                let v = args
                    .next()
                    .ok_or("--encoder-inversion-census requires <path>")?;
                encoder_inversion_census = Some(PathBuf::from(v));
            }
            "--inversion-truth" => {
                let v = args
                    .next()
                    .ok_or("--inversion-truth requires single|agree")?;
                inversion_truth = gaddr::InversionTruth::parse(&v)?;
            }
            "--reference-truth" => {
                let v = args
                    .next()
                    .ok_or("--reference-truth requires <tsv>[:variant]")?;
                // `path:variant` — split on the LAST colon so an absolute path
                // with a drive/colon in it still parses.
                let (path, variant) = match v.rsplit_once(':') {
                    Some((p, tag)) if gaddr::ButteraugliVariant::parse(tag).is_ok() => {
                        (p.to_string(), gaddr::ButteraugliVariant::parse(tag)?)
                    }
                    _ => (v.clone(), gaddr::ButteraugliVariant::PNorm3),
                };
                reference_truth = Some((PathBuf::from(path), variant));
            }
            "--gaddr-reference" => {
                let v = args.next().ok_or("--gaddr-reference requires <name>")?;
                gaddr_reference = Some(v);
            }
            "--gaddr-tail-pins" => {
                let v = args
                    .next()
                    .ok_or("--gaddr-tail-pins requires <product|retired>")?;
                // Validate at PARSE time: an unknown value must never fall
                // through to the default, which would grade an arm against
                // bars nobody selected.
                zensim_validate::dial_addressability::TailPins::parse(&v)?;
                gaddr_tail_pins = Some(v);
            }
            "--gaddr-value-pins" => {
                let v = args
                    .next()
                    .ok_or("--gaddr-value-pins requires <report|hard>")?;
                // Validate here so a typo fails at argv, not silently at grading.
                zensim_validate::dial_addressability::ValuePins::parse(&v)?;
                gaddr_value_pins = Some(v);
            }
            "--gaddr-grid-truth" => {
                let v = args.next().ok_or("--gaddr-grid-truth requires <tsv>")?;
                gaddr_grid_truth = Some(PathBuf::from(v));
            }
            "--floor-rule" => {
                let v = args
                    .next()
                    .ok_or("--floor-rule requires <distinct|resolvable|spaced>")?;
                // Validate at PARSE time (margin doesn't matter for the tag
                // check — the real margin, if any, is resolved after the
                // whole command line is read). An unknown value must never
                // fall through to `distinct`, which would silently grade
                // against a rule nobody selected.
                zensim_validate::dial_addressability::FloorRule::parse(
                    &v,
                    zensim_validate::dial_addressability::FloorRule::RESOLVABLE_MARGIN_DEFAULT,
                )?;
                floor_rule = Some(v);
            }
            "--floor-margin" => {
                let v = args.next().ok_or("--floor-margin requires <f64>")?;
                let m: f64 = v
                    .parse()
                    .map_err(|_| format!("--floor-margin: not a number: {v:?}"))?;
                floor_margin = Some(m);
            }
            "--gaddr-json" => {
                let v = args.next().ok_or("--gaddr-json requires <path>")?;
                gaddr_json = Some(PathBuf::from(v));
            }
            "--negtail-peer-scores" | "--identity-peer-scores" => {
                let which = arg.clone();
                let v = args
                    .next()
                    .ok_or_else(|| format!("{which} requires <label>=<path>"))?;
                let (label, path) = v
                    .split_once('=')
                    .ok_or_else(|| format!("{which} must be <label>=<path>"))?;
                if label.trim().is_empty() {
                    return Err(format!("{which} label must be non-empty"));
                }
                let slot = if which == "--negtail-peer-scores" {
                    &mut negtail_peer_scores
                } else {
                    &mut identity_peer_scores
                };
                *slot = Some((label.to_string(), PathBuf::from(path)));
            }
            "--per-pair-output" => {
                let v = args.next().ok_or("--per-pair-output requires <path>")?;
                per_pair_output = Some(PathBuf::from(v));
            }
            "--per-pair-refs" => {
                per_pair_refs = true;
            }
            "--html" => {
                let v = args.next().ok_or("--html requires <path>")?;
                html = Some(PathBuf::from(v));
            }
            "--json" => {
                let v = args.next().ok_or("--json requires <path>")?;
                json = Some(PathBuf::from(v));
            }
            "--full-json" => {
                let v = args.next().ok_or("--full-json requires <path>")?;
                full_json = Some(PathBuf::from(v));
            }
            "--fulleval" => {
                let v = args.next().ok_or("--fulleval requires <path>")?;
                fulleval = Some(PathBuf::from(v));
            }
            "--name" => {
                let v = args.next().ok_or("--name requires <string>")?;
                name = Some(v);
            }
            "--perpair-metrics" => {
                let v = args.next().ok_or("--perpair-metrics requires <path>")?;
                perpair_metrics = PathBuf::from(v);
                perpair_metrics_set = true;
            }
            "--perpair-cap" => {
                let v = args.next().ok_or("--perpair-cap requires <n>")?;
                perpair_cap = v
                    .parse()
                    .map_err(|e| format!("--perpair-cap must be an integer: {e}"))?;
            }
            "--ramp-grid" => {
                let v = args.next().ok_or("--ramp-grid requires <path>")?;
                ramp_grid = Some(PathBuf::from(v));
            }
            "--compare" => {
                let v = args.next().ok_or("--compare requires <ref-bake path>")?;
                compare = Some(PathBuf::from(v));
            }
            "--corruption-grid" => {
                let v = args.next().ok_or("--corruption-grid requires <path>")?;
                corruption_grid = PathBuf::from(v);
                corruption_grid_set = true;
            }
            "--corruption-head-threshold" => {
                let v = args
                    .next()
                    .ok_or("--corruption-head-threshold requires <score>")?;
                corruption_head_threshold = v
                    .parse()
                    .map_err(|_| "--corruption-head-threshold must be a number")?;
            }
            "--corruption-head" => {
                let v = args.next().ok_or("--corruption-head requires <bake.bin>")?;
                corruption_head = Some(PathBuf::from(v));
            }
            "--negtail-probe" => {
                let v = args.next().ok_or("--negtail-probe requires <parquet>")?;
                negtail_probe = Some(PathBuf::from(v));
            }
            "--identity-probe" => {
                let v = args.next().ok_or("--identity-probe requires <parquet>")?;
                identity_probe = Some(PathBuf::from(v));
            }
            "--print-features-root" => {
                print_features_root = true;
            }
            "--require-feature-set-match" => {
                require_feature_set_match = true;
            }
            "--cross-regime" => {
                cross_regime = true;
            }
            "--allow-unpopulated-slots" => {
                allow_unpopulated_slots = true;
            }
            "--ensemble" => {
                let v = args
                    .next()
                    .ok_or("--ensemble requires a comma-separated list")?;
                ensemble = v
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(PathBuf::from)
                    .collect();
                if ensemble.is_empty() {
                    return Err("--ensemble list is empty".to_string());
                }
            }
            "--ensemble-weights" => {
                let v = args
                    .next()
                    .ok_or("--ensemble-weights requires a comma-separated list")?;
                let mut w: Vec<f64> = Vec::new();
                for tok in v.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    let x: f64 = tok
                        .parse()
                        .map_err(|e| format!("--ensemble-weights: {tok:?} is not a number: {e}"))?;
                    if !x.is_finite() || x < 0.0 {
                        return Err(format!(
                            "--ensemble-weights: weights must be finite and >= 0, got {x}"
                        ));
                    }
                    w.push(x);
                }
                if w.is_empty() {
                    return Err("--ensemble-weights list is empty".to_string());
                }
                ensemble_weights = Some(w);
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
    // Weights are meaningless without members, must match them 1:1, and must
    // not be all-zero (which would score a constant 0 and look like a model).
    // Normalised here so the ONE published vector is the one that scored.
    let ensemble_weights = match ensemble_weights {
        None => None,
        Some(w) => {
            if ensemble.is_empty() {
                return Err("--ensemble-weights requires --ensemble".to_string());
            }
            if w.len() != ensemble.len() {
                return Err(format!(
                    "--ensemble-weights has {} entries but --ensemble has {} members",
                    w.len(),
                    ensemble.len()
                ));
            }
            let sum: f64 = w.iter().sum();
            if sum.is_nan() || sum <= 0.0 {
                return Err("--ensemble-weights must not sum to zero".to_string());
            }
            Some(w.iter().map(|x| x / sum).collect::<Vec<f64>>())
        }
    };
    // `--ensemble` supplies its own provenance anchor when `--bake` is absent.
    // An explicit `--bake` alongside `--ensemble` must name a member, so the
    // report's provenance header can never describe a bake that is not in the
    // scored function.
    if let Some(b) = &bake
        && !ensemble.is_empty()
        && !ensemble.contains(b)
    {
        return Err(format!(
            "--bake {} is not one of the --ensemble members; omit --bake to use member 0 \
             as the provenance anchor",
            b.display()
        ));
    }
    let bake = match bake {
        Some(b) => b,
        None => ensemble
            .first()
            .cloned()
            .ok_or("--bake is required (path to ZNPR v3 bake)")?,
    };
    // --regime 944: the SOTA-944 campaign preset. Everything the campaign
    // invocation needed resolves here so a bare run cannot silently omit a
    // corpus/grid; each piece yields to an explicit flag.
    if regime_944 {
        if !features_root_set {
            features_root = PathBuf::from(DEFAULT_FEATURES_ROOT_944);
        }
        if !dial_grid_set {
            dial_grid = PathBuf::from(DEFAULT_DIAL_GRID_944);
        }
        if !corruption_grid_set {
            corruption_grid = PathBuf::from(DEFAULT_CORRUPTION_GRID_944);
        }
        if !perpair_metrics_set {
            perpair_metrics = PathBuf::from(DEFAULT_PERPAIR_METRICS_944);
        }
        if corpora.is_none() {
            corpora = Some(
                parse_corpora_arg(SOTA944_CORPORA)
                    .expect("SOTA944_CORPORA names only registered corpora (test-pinned)"),
            );
        }
    } else if regime_720 {
        // --regime 720: swap unset defaults to the 720-wide variants.
        if !features_root_set {
            features_root = PathBuf::from(DEFAULT_FEATURES_ROOT_720);
        }
        if !dial_grid_set {
            dial_grid = PathBuf::from(DEFAULT_DIAL_GRID_720);
        }
        if !corruption_grid_set {
            corruption_grid = PathBuf::from(DEFAULT_CORRUPTION_GRID_720);
        }
    }
    let mut corpora = corpora.unwrap_or_else(|| CORPORA.iter().collect());
    if regime_720 {
        let before = corpora.len();
        corpora.retain(|c| slot_720(c.name).is_some());
        let skipped = before - corpora.len();
        if skipped > 0 {
            eprintln!(
                "bake_verdict --regime 720: skipped {skipped} corpora with no 720 extraction \
                 (pipal/hf_nearlossless)"
            );
        }
    }
    // A peer dial belongs to the peer, not to `--bake`. Writing it into a
    // fulleval JSON would put another metric's dial under a bake's name on
    // the board — refuse rather than trust a convention.
    // A peer PROBE table describes the peer for exactly the same reason, and it
    // is only meaningful when the grid it is reported beside is the same
    // scorer's — otherwise the G-ADDR headline mixes two scorers under one
    // verdict.
    if (negtail_peer_scores.is_some() || identity_peer_scores.is_some())
        && dial_peer_scores.is_none()
    {
        return Err(
            "--negtail-peer-scores / --identity-peer-scores require --dial-peer-scores: \
                    a peer's tail or identity reported beside the BAKE's dial grid is two \
                    scorers under one G-ADDR headline, which is not a measurement of either."
                .into(),
        );
    }
    if (negtail_peer_scores.is_some() || identity_peer_scores.is_some())
        && (full_json.is_some() || fulleval.is_some())
    {
        return Err(
            "--negtail-peer-scores / --identity-peer-scores cannot be combined with \
                    --full-json/--fulleval: the G-ADDR block would describe the peer while \
                    every other block describes --bake."
                .into(),
        );
    }
    if dial_peer_scores.is_some() && (full_json.is_some() || fulleval.is_some()) {
        return Err(
            "--dial-peer-scores cannot be combined with --full-json/--fulleval: the DIAL \
             block would describe the peer while every other block describes --bake. Read \
             the peer dial from the markdown report (--output) instead."
                .into(),
        );
    }
    // `resolvable`/`spaced` choose their window from the MENTOR's own
    // per-cell values and their bar is the mentor's LIVE fraction — both
    // need --gaddr-grid-truth. Refuse rather than silently reading every
    // codec as NOT MEASURED, which would look like a quiet pass at a glance.
    if let Some(fr) = floor_rule.as_deref() {
        let needs_mentor = zensim_validate::dial_addressability::FloorRule::parse(
            fr,
            floor_margin.unwrap_or(
                zensim_validate::dial_addressability::FloorRule::RESOLVABLE_MARGIN_DEFAULT,
            ),
        )
        .expect("validated at argument-parse time")
        .needs_mentor_truth();
        if needs_mentor && gaddr_grid_truth.is_none() {
            return Err(format!(
                "--floor-rule {fr} requires --gaddr-grid-truth: its window AND its bar are \
                 both computed from the mentor's own per-cell scores on this instrument — \
                 unlike `distinct`, there is no registry pin to fall back on."
            ));
        }
    }
    Ok(Args {
        bake,
        ensemble,
        ensemble_weights,
        corpora,
        output,
        features_root,
        regime_720,
        regime_944,
        print_features_root,
        per_pair_output,
        per_pair_refs,
        dial_grid,
        html,
        json,
        ramp_grid,
        compare,
        corruption_grid,
        corruption_head,
        corruption_head_threshold,
        negtail_probe,
        identity_probe,
        dial_peer_scores,
        inversion_truth,
        reference_truth,
        encoder_inversion_census,
        negtail_peer_scores,
        identity_peer_scores,
        gaddr_json,
        gaddr_reference,
        gaddr_tail_pins,
        gaddr_value_pins,
        gaddr_grid_truth,
        floor_rule,
        floor_margin,
        full_json,
        fulleval,
        name,
        perpair_metrics,
        perpair_cap,
        cross_regime,
        allow_unpopulated_slots,
        require_feature_set_match,
    })
}

/// Score every row of a feature grid through the same dispatch path
/// `render_corpus` / `dial_panel` use. Reused by the severity-ramp and
/// per-zone sections so their numbers match the rest of the report
/// exactly.
/// Rows per rayon task in [`score_grid_one`]. Big enough that the
/// per-chunk `Predictor::new` + scratch allocation is amortized (a
/// 944-input forward is ~120k MACs, so 512 rows ≈ 60 MFLOP of work per
/// task), small enough that a 12-corpus run with corpora ALSO in flight
/// still load-balances.
const SCORE_CHUNK_ROWS: usize = 512;

/// Below this many rows the sequential path is used — spawning tasks for
/// a 100-row grid costs more than the forward does.
const SCORE_PARALLEL_MIN_ROWS: usize = 2048;

fn score_grid_one(
    model: &Model,
    has_transforms: bool,
    n_inputs: usize,
    rows: &[Vec<f64>],
) -> Vec<f64> {
    let per_sample_alpha_head = extract_per_sample_alpha_head(model);
    let hybrid_head = extract_hybrid_head(model);
    let tanh_pin_scale = extract_tanh_output_head_scale(model);
    let output_spline = zensim_validate::output_calibration_spline::extract(model);
    let minmax_head = extract_minmax_head(model);

    // One row's prediction reads only that row (the `Predictor`'s scratch
    // is fully overwritten per call, and every head/spline handle above is
    // read-only), so splitting the rows across threads and writing each
    // result back at its own index is BIT-IDENTICAL to the sequential
    // loop — no accumulation order changes, no shared mutable state.
    // `scripts/verify_verdict_identity.sh` gates that claim on every
    // numeric field of a full `--full-json`.
    let score_range = |predictor: &mut Predictor<'_>,
                       scratch: &mut Vec<f32>,
                       src: &[Vec<f64>],
                       dst: &mut [f64]| {
        for (out, row) in dst.iter_mut().zip(src.iter()) {
            *out = match minmax_head.as_ref() {
                // Min-max bakes REPLACE the layer forward — bypass the Predictor.
                Some(mm) => {
                    score_row_minmax(model, mm, tanh_pin_scale, output_spline.as_ref(), row)
                }
                None => score_row(
                    predictor,
                    has_transforms,
                    per_sample_alpha_head.as_ref(),
                    hybrid_head.as_ref(),
                    tanh_pin_scale,
                    output_spline.as_ref(),
                    scratch,
                    row,
                ),
            };
        }
    };

    let mut out = vec![0.0f64; rows.len()];
    if rows.len() < SCORE_PARALLEL_MIN_ROWS || rayon::current_num_threads() <= 1 {
        let mut predictor = Predictor::new(model);
        let mut scratch = vec![0.0f32; n_inputs];
        score_range(&mut predictor, &mut scratch, rows, &mut out);
        return out;
    }
    zensim_validate::parallel::init();
    out.par_chunks_mut(SCORE_CHUNK_ROWS)
        .zip(rows.par_chunks(SCORE_CHUNK_ROWS))
        .for_each(|(dst, src)| {
            let mut predictor = Predictor::new(model);
            let mut scratch = vec![0.0f32; n_inputs];
            score_range(&mut predictor, &mut scratch, src, dst);
        });
    out
}

/// An equal-weight ensemble of ZNPR bakes scored as ONE model: every row's
/// prediction is the arithmetic mean of the members' RAW predictions.
///
/// Why it lives here rather than in a script (CLAUDE.md "one owner per task"):
/// every endpoint this binary reports — the Mohammadi panel, per-reference
/// SROCC, the dial mono/tied panel, corruption — must come from the SAME code
/// path for an ensemble as for a single bake, or the two are not comparable.
/// Averaging per-pair dumps outside the binary can reproduce the rank panel but
/// **cannot** reach the dial grid (no `human` column) or the per-reference
/// grouping (no `ref_id` in the dump), which are frozen bar rows.
///
/// **Raw-then-recalibrate order.** Members are averaged AFTER each member's own
/// output spline is applied (`score_grid_one` is the full shipped forward), i.e.
/// in each member's score units. That is the sound order for a *rank* endpoint:
/// each member's spline is monotone, so it cannot change that member's own
/// ranking, and the mean is taken in a common, comparable unit. A single shared
/// recalibration of the ensemble is the packaging step (`bake_dial_refit`),
/// not something this evaluation needs — SROCC is invariant to it.
///
/// **Single-member identity is structural**: `score_rows` short-circuits to
/// exactly `score_grid_one` for k=1, so a 1-member ensemble reproduces a plain
/// `--bake` run bit-for-bit (no `0.0 + x`, no `x / 1.0` rounding surface).
struct Ensemble {
    models: Vec<Model>,
    /// Per-member `has_nontrivial_feature_transforms()` — members are allowed
    /// to differ here; the dispatch is per-member, never the primary's.
    has_transforms: Vec<bool>,
    n_inputs: usize,
    /// `None` = the historical equal-weight mean, accumulated exactly as before
    /// so pre-flag ensembles reproduce bit-for-bit. `Some(w)` = a convex blend
    /// with `Σw = 1` (normalised at parse time). Same ordering as `models`.
    weights: Option<Vec<f64>>,
}

impl Ensemble {
    /// Member 0 — the provenance anchor + the bake whose architecture the
    /// report's metadata blocks describe.
    fn primary(&self) -> &Model {
        &self.models[0]
    }

    fn len(&self) -> usize {
        self.models.len()
    }

    /// Mean (or convex blend) of the members' raw predictions, row by row.
    fn score_rows(&self, rows: &[Vec<f64>]) -> Vec<f64> {
        if self.models.len() == 1 && self.weights.is_none() {
            // Bit-identical to the single-bake path by construction.
            return score_grid_one(&self.models[0], self.has_transforms[0], self.n_inputs, rows);
        }
        if let Some(w) = &self.weights {
            let mut acc = vec![0.0f64; rows.len()];
            for ((m, &tf), &wi) in self
                .models
                .iter()
                .zip(self.has_transforms.iter())
                .zip(w.iter())
            {
                if wi == 0.0 {
                    continue;
                }
                for (a, v) in acc
                    .iter_mut()
                    .zip(score_grid_one(m, tf, self.n_inputs, rows))
                {
                    *a += wi * v;
                }
            }
            return acc;
        }
        let mut acc = vec![0.0f64; rows.len()];
        for (m, &tf) in self.models.iter().zip(self.has_transforms.iter()) {
            for (a, v) in acc
                .iter_mut()
                .zip(score_grid_one(m, tf, self.n_inputs, rows))
            {
                *a += v;
            }
        }
        let k = self.models.len() as f64;
        acc.iter().map(|a| a / k).collect()
    }
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
    /// Signed SROCC (NOT abs'd). `srocc` above is `|·|` per the Mohammadi panel
    /// convention; this preserves polarity so a globally-inverted bake reads
    /// negative instead of masquerading as a healthy positive. The dashboard
    /// shows it for the MOS corpora; the abs form is reserved for the JND
    /// corpora whose target sign is genuinely ambiguous.
    srocc_signed: f64,
    /// Marginal bootstrap 95% CI `(lo, hi)` of `|SROCC|` — makes a mid-cluster
    /// ranking gap legible as real-or-noise (the bare point estimate cannot).
    srocc_ci: (f64, f64),
    /// Logistic-rescaled scores in [0,100] dial space (for G1 range check).
    rescaled_scores: Vec<f64>,
    /// Corpus key (lowercase, e.g. "cid22") for `--full-json` keying and the
    /// mos-vs-jnd per_pair classification.
    name: &'static str,
    /// Human target per pair, aligned 1:1 with `rescaled_scores` — the y-axis
    /// of the `--full-json` per_pair scatter (mos for MOS corpora, jnd for the
    /// JND corpora).
    humans: Vec<f64>,
    /// Per-reference SROCC summary, when the corpus carries ref identity.
    per_ref: Option<PerGroupSrocc>,
    /// Interned per-reference group id per row (aligned 1:1 with
    /// `rescaled_scores`/`humans`), kept solely for the opt-in
    /// `--per-pair-refs` dump. `None` when the corpus has no ref identity.
    ref_ids: Option<Vec<u32>>,
    /// 10-band panel rows (`None` when per-band doesn't apply to the corpus).
    /// Captured alongside the markdown table so `--full-json` can carry the
    /// band structure the interactive dashboard renders (bands used to live
    /// only in verdict.md).
    bands: Option<Vec<BandRow>>,
    body: String,
}

/// One row of the 10-band Mohammadi panel, mirrored into `--full-json`.
struct BandRow {
    band: String,
    lo: f64,
    hi: f64,
    n: usize,
    /// Realised target span (max − min) of the rows inside the band. Published
    /// beside `n` because both failures the fixed-decile grid produced were
    /// invisible in the statistic alone: a band whose target sd was 4.4× tighter
    /// than its neighbours, and an SROCC whose sign was opposite to the value
    /// printed. Range restriction attenuates a band's correlation toward zero
    /// while leaving its noise alone, so a band's number cannot be read without
    /// its span.
    span: f64,
    /// `Some(reason)` when the band does not clear
    /// [`zensim_validate::bands`]'s usability floors. A NOT-MEASURED band
    /// publishes NO statistics (every stat field is NaN → null) — it is never a
    /// measured zero, and it is never silently dropped.
    not_measured_reason: Option<String>,
    srocc: f64,
    /// The band's Spearman WITH ITS SIGN. `srocc` above is `.abs()` (the
    /// polarity-tolerant aggregate convention), which on a band tail hides the
    /// one thing the tail is asked about: a band whose ordering has COLLAPSED
    /// scores identically to one that is correctly ordered, and a band that is
    /// more deeply INVERTED scores HIGHER. `freeze_check`'s F8 documents itself
    /// as signed ("collapse must hurt") and its `B3 >= 0.0` clause is only
    /// meaningful against a signed value. Emitted additively so no existing
    /// consumer of `srocc` changes.
    srocc_signed: f64,
    plcc: f64,
    krocc: f64,
    or_ratio: f64,
    pwrc: f64,
    z_rmse: f64,
    mae: f64,
}

/// Minimum rows per reference for its ladder to be rankable. A 2-row SROCC is
/// +/-1 by construction and only adds noise at the extremes.
///
/// The stat itself lives in `zenstats::panel::per_group_srocc` — the canonical
/// home for statistical math. This binary only chooses the grouping (by
/// reference image) and renders the result.
const PER_REF_MIN_ROWS: usize = 3;

/// KADID/TID are 100% train==val pair-overlap in the zensim training corpus
/// (docs + project MEMORY): their held-out SROCC rewards memorization, not
/// generalization skill, so the eval must MARK them rather than let a reader
/// rank a bake by numbers that reward overfit. This is the runtime flag the
/// report + `--full-json` lacked — the fact previously lived only in a code
/// comment (see `benchmarks/stats_correctness_review_2026-07-26.md` §4b).
fn train_eq_val(name: &str) -> bool {
    matches!(name, "kadid" | "tid")
}

/// Corpora whose `human_score` is QUALITY-oriented, so a NEGATIVE signed SROCC is a
/// genuine ranking inversion and must never render as a high score.
///
/// **Why this exists (2026-08-04).** The report printed `|SROCC|` for every corpus. On
/// 2026-08-04 the ext-lineage KADID eval tables were found to store `human_score =
/// (5−dmos)/4` — the inverse of the canonical `(dmos−1)/4` — so 110 of 188 board bakes
/// were anti-correlated with KADID's real human MOS while every one of them rendered as
/// a positive magnitude, and a wave-8 gate (`KADID ≥ 0.70`) was passed by the three
/// most-inverted arms. An unsigned display cannot show that. See
/// `benchmarks/sota944_campaign_2026-08-03.md` REGISTERED APPENDIX F.
///
/// `konjnd` is deliberately EXCLUDED: its validation target is a mean-PJND threshold, so
/// its SROCC is *structurally* negative on at-PJND pairs and `|SROCC|` is the correct
/// reading there (see the project memory note on konjnd's two `human_score` meanings).
/// Everything else in `CORPORA` is quality-oriented.
fn sign_is_meaningful(name: &str) -> bool {
    !matches!(name, "konjnd")
}

/// Rendered SROCC cell: the SIGNED value on quality-oriented corpora (with a loud
/// inversion marker when negative), `|SROCC|` on `konjnd` where the sign carries no
/// meaning. The JSON keeps both `srocc` and `srocc_signed` unchanged — this is the
/// human-facing surface only.
fn srocc_cell(name: &str, srocc: f64, srocc_signed: f64) -> String {
    if !sign_is_meaningful(name) {
        return format!("{srocc:.4}");
    }
    if srocc_signed < 0.0 {
        format!("**{srocc_signed:+.4} ⛔INVERTED**")
    } else {
        format!("{srocc_signed:+.4}")
    }
}

/// JSON-safe float: non-finite → `None` (serialized as null) so NaN band/dial
/// stats can't produce invalid JSON or a silent 0.
fn nan_null(v: f64) -> Option<f64> {
    v.is_finite().then_some(v)
}

/// Marginal bootstrap 95% CI of |SROCC| for one corpus: resample the
/// `(score, human)` PAIRS with replacement `N_BOOT` times, take |SROCC| of
/// each resample, and return the (2.5th, 97.5th) percentiles.
///
/// **Why this exists.** The single-bake report prints SROCC to 3 decimals with
/// no uncertainty, which silently invites reading a 0.882-vs-0.880 gap as an
/// ordering. It usually isn't: at n≈4292 the 95% CI half-width is ≈0.006, so
/// adjacent bakes in the mid-cluster are a statistical tie. The CI makes that
/// visible (paired A-vs-B significance still lives in `bake_compare`; this is
/// the marginal per-corpus band the dashboard renders as a tie-guide).
///
/// The STATISTIC is `zenstats::panel::spearman` — only the resampling loop is
/// here, so this is orchestration, not a duplicate stat (per the no-duplication
/// rule). Deterministic: a fixed-seed SplitMix64/xorshift keyed off `n` so a
/// bake's CI is reproducible run-to-run without a new `rand` dependency.
fn bootstrap_srocc_ci(scores: &[f64], humans: &[f64]) -> (f64, f64) {
    fn xs(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }
    let n = scores.len().min(humans.len());
    if n < 8 {
        return (f64::NAN, f64::NAN);
    }
    const N_BOOT: usize = 1000;
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15 ^ (n as u64).wrapping_mul(0x2545_F491_4F6C_DD1D);
    // Draw every replicate's index set SEQUENTIALLY from the one xorshift
    // stream — that is what makes the CI reproducible, and it is the only
    // part that must stay serial. The `spearman` evaluations are independent
    // per replicate, so they run on rayon and land back at their own index:
    // same 1000 values, same order, same percentiles, bit-for-bit.
    //
    // Generated a BLOCK at a time so the index buffer stays ~5 MB instead of
    // `N_BOOT × n` (45 MB on the largest corpus, ×12 corpora now that the
    // corpus loop is parallel too).
    const BOOT_BLOCK: usize = 128;
    let mut rs: Vec<f64> = vec![0.0; N_BOOT];
    let mut draws: Vec<u32> = vec![0; BOOT_BLOCK * n];
    let mut done = 0usize;
    while done < N_BOOT {
        let this = BOOT_BLOCK.min(N_BOOT - done);
        for slot in draws[..this * n].iter_mut() {
            *slot = (xs(&mut state) % n as u64) as u32;
        }
        rs[done..done + this]
            .par_iter_mut()
            .zip(draws[..this * n].par_chunks(n))
            .for_each(|(out, idxs)| {
                let bs: Vec<f64> = idxs.iter().map(|&i| scores[i as usize]).collect();
                let bh: Vec<f64> = idxs.iter().map(|&i| humans[i as usize]).collect();
                *out = spearman(&bs, &bh).abs();
            });
        done += this;
    }
    rs.sort_by(f64::total_cmp);
    let pct = |p: f64| -> f64 {
        let pos = p * (N_BOOT as f64 - 1.0);
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(N_BOOT - 1);
        let frac = pos - lo as f64;
        rs[lo] * (1.0 - frac) + rs[hi] * frac
    };
    (pct(0.025), pct(0.975))
}

/// Canonical PRODUCT-WEIGHTED ranking composite over the per-corpus results —
/// the single "which bake ranks best for the product?" number. Called by BOTH
/// the scorecard and the `--full-json` writer, and the dashboard READS the
/// emitted value rather than re-deriving one, so there is exactly one formula
/// (the 2026-07-26 review found two composites that could disagree). Weights
/// center the product axes (CID22 gold MOS + imazen26 real-codec ssim2 +
/// non-photo) over held-out human JND; **KADID/TID are excluded** (train==val
/// memorization). |SROCC| per corpus; a corpus absent from the run drops from
/// both numerator and denominator. Matched byte-for-byte in `gauntlet.py`'s
/// fallback only — the primary path reads this value from the JSON.
fn product_composite(results: &[CorpusResult]) -> f64 {
    let term = |sub: &str, w: f64| -> Option<(f64, f64)> {
        results
            .iter()
            .find(|r| r.display.contains(sub))
            .map(|r| (w * r.srocc, w))
    };
    let terms = [
        term("CID22", 1.00),
        term("real-codec", 0.50),
        term("non-photo", 0.30),
        term("KonJND", 0.20),
        term("AIC-3", 0.10),
        term("AIC-4", 0.05),
    ];
    let (num, den) = terms
        .iter()
        .flatten()
        .fold((0.0f64, 0.0f64), |(n, d), (x, y)| (n + x, d + y));
    if den > 0.0 { num / den } else { f64::NAN }
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
/// The DIAL ship-gate numbers, returned alongside the markdown so `--json`
/// (and the comparative dashboard) can carry the codec-dial verdict — a bake
/// can win every rank corpus and still be a broken dial (non-monotonic, no
/// range), which is exactly why the two-panel eval is mandatory.
#[derive(Clone)]
struct DialMetrics {
    /// G3: monotonicity = 1 − material-inversion rate (gate ≥ 0.93).
    /// Under the 2026-09-05 two-reference ruling this counts only the
    /// DIAL-attributed inversions; see [`InversionTruthReport`].
    mono: f64,
    /// Which inversion reading produced `mono`, and the encoder-attributed
    /// half. Always present so a reader can never mistake "the rule did not
    /// run" for "no encoder inversions".
    inversion_truth: InversionTruthReport,
    /// G1 dial dynamic range percentiles (gate p5 ≤ 25 ∧ p95 ≥ 85).
    p5: f64,
    p95: f64,
    /// Flat/clamp dead-zone rate (informational).
    flat: f64,
    /// Full representable dial reach across the codec grid = pooled max − min
    /// (the G4 "cross-codec reach"). `--full-json`'s `dynamic_range` is the
    /// robust p95 − p5; `reach` is the total span (outliers included).
    reach: f64,
    /// The pooled extremes `reach` is built from. Published separately because
    /// the G-ADDR gate bars them INDEPENDENTLY: a candidate can hold `reach`
    /// while trading ceiling for floor, and that trade is exactly what the
    /// era-corrected anchors do (floor up, ceiling down).
    min: f64,
    max: f64,
    /// G-ADDR — the dial ADDRESSABILITY verdict against the shipped product
    /// dial's own end-of-range behaviour on this same grid. `None` only when
    /// the dial panel itself did not run.
    addr: Option<zensim_validate::dial_addressability::Verdict>,
    /// Per-codec dial stats (codec, n_curves, n_pairs, monotonicity, tied) —
    /// the headline mono/flat pool across codecs, which can mask one broken
    /// family; `--full-json` carries the breakout so the dashboard shows it.
    per_codec: Vec<PerCodecDial>,
    /// Per-codec aggregated dial CURVE for plotting: at each grid param q,
    /// the (p25, median, p75) of the dial score across that codec's image
    /// ladders. Compact (~40-60 pts/codec) yet enough to draw the dial shape.
    curves: Vec<CodecCurve>,
    /// Ladder inversions split by codec x quality zone and by content class x
    /// quality zone (`None` only when the grid is absent). The pooled `mono`
    /// above answers "does the dial run backwards"; this answers "WHERE".
    zones: Option<DialZones>,
}

#[derive(Clone)]
struct PerCodecDial {
    codec: String,
    n_curves: usize,
    n_pairs: usize,
    mono: f64,
    tied: f64,
}

#[derive(Clone)]
struct CodecCurve {
    codec: String,
    /// (q, p25, median, p75) sorted by q.
    pts: Vec<(f64, f64, f64, f64)>,
}

/// Quality-zone edges on the grid's normalised `q` axis (0..100), shared by
/// every codec including JXL (whose native param is a butteraugli distance —
/// the grid carries the normalised `q` beside it).
///
/// The zones are production regimes, not statistics: below 50 is aggressive
/// web compression, 50..85 is ordinary web quality, 85 and above is the
/// high-fidelity / near-lossless band where a codec loop spends most of its
/// iterations and where every learned metric is weakest.
const ZONE_EDGES: [f64; 2] = [50.0, 85.0];

/// A backwards step this large (dial score points) is a real ranking error
/// rather than sub-JND noise — the threshold the G3 gate has always used,
/// lifted to module scope so the zone split and the JSON emitter quote the
/// SAME number the gate counts.
const MATERIAL_INV_PT: f64 = 0.5;

/// Zone label for one adjacent-q PAIR, keyed on the pair's q MIDPOINT (the
/// decision "is the next step up actually better" is made between the two
/// rungs, so the midpoint is the operating point that meets the user).
fn zone_of(q_mid: f64) -> &'static str {
    if q_mid < ZONE_EDGES[0] {
        "q<50"
    } else if q_mid < ZONE_EDGES[1] {
        "q50-85"
    } else {
        "q>=85"
    }
}

/// The three zone labels in report order.
const ZONE_LABELS: [&str; 3] = ["q<50", "q50-85", "q>=85"];

/// Adjacent-rung outcome counts for one (split-key, zone) cell of the ladder
/// grid. The six buckets are the SAME five mutually-exclusive outcomes the
/// pooled G3 gate uses (plus the strict-inversion diagnostic), so a cell's
/// numbers add up to `n_pairs` and reconcile with `mono`/`tied` exactly.
#[derive(Clone, Default)]
struct ZoneCell {
    n_pairs: usize,
    forward: usize,
    /// Material backwards rungs charged to the DIAL (the operative count).
    inv_material: usize,
    /// Material backwards rungs charged to the ENCODER — both reference
    /// metrics independently call the higher setting worse. Reported, never
    /// gated against the dial. Always 0 under `--inversion-truth single`.
    inv_encoder: usize,
    /// Material backwards rungs whose attribution is NOT MEASURABLE (the
    /// reference table has no row for one endpoint). These stay in
    /// `inv_material` — unknown is never an exemption — and are counted here
    /// so a thin truth table cannot look like a clean dial.
    inv_unknown: usize,
    inv_strict: usize,
    flat: usize,
    codec_sat: usize,
    subres: usize,
    /// Magnitudes (dial points) of the MATERIAL backwards steps in this cell.
    inv_mags: Vec<f64>,
}

/// Ladder-level (whole-curve) outcomes for one (split-key, zone) cell.
#[derive(Clone, Default)]
struct ZoneLadders {
    /// Ladders with at least 3 rungs inside the zone (the population).
    n: usize,
    /// Ladders carrying at least one MATERIAL backwards rung inside the zone.
    with_inv: usize,
    /// Ladders whose zone ENDPOINTS are backwards: the score at the zone's
    /// best-quality rung is more than MATERIAL below the score at its
    /// worst-quality rung. This is the "ranks the ladder backwards" statement
    /// — a codec loop handed such a ladder walks the wrong way.
    endpoint_backwards: usize,
}

/// The 2026-09-05 two-reference inversion ruling, as it applied to ONE run.
///
/// `mono` in [`DialMetrics`] is the DIAL-attributed number; `mono_single` is
/// the pre-ruling one. Both travel together so a published figure always says
/// which reading it is, and a reader can see the size of the exemption rather
/// than inferring it.
#[derive(Clone, Default)]
struct InversionTruthReport {
    /// `single` or `agree` — the reading that produced `DialMetrics::mono`.
    /// This is the EFFECTIVE reading: a requested `agree` with no usable
    /// reference table degrades to `single` and reports `single` here.
    effective: String,
    /// What the caller asked for, before any fallback.
    requested: String,
    /// The reference table actually read, and its butteraugli variant.
    source: Option<String>,
    variant: Option<String>,
    n_reference_cells: usize,
    /// Material backwards rungs charged to the ENCODER (both refs agree).
    n_encoder: usize,
    /// Material backwards rungs whose attribution is NOT MEASURABLE; these
    /// stay charged to the dial.
    n_unknown: usize,
    /// `1 − (dial + encoder) / pairs` — the pre-ruling monotonicity.
    mono_single: f64,
    /// The human sentence the markdown prints, including the loud fallback.
    note: String,
    /// Every encoder-attributed pair: `(image, codec, q_lo, q_hi, dial drop,
    /// Δssim2, Δbutteraugli)` — the codec bug-report evidence.
    encoder_pairs: Vec<(String, String, f64, f64, f64, f64, f64)>,
}

impl InversionTruthReport {
    /// The "no dial panel ran at all" value — distinct from a run that read a
    /// truth table and found nothing, which carries a real `effective` tag.
    const EMPTY: Self = Self {
        effective: String::new(),
        requested: String::new(),
        source: None,
        variant: None,
        n_reference_cells: 0,
        n_encoder: 0,
        n_unknown: 0,
        mono_single: f64::NAN,
        note: String::new(),
        encoder_pairs: Vec::new(),
    };
}

/// Ladder-inversion breakdown: the pooled dial monotonicity split by the two
/// axes a reader meets it on — which codec, and which quality zone — plus the
/// same split by reviewed content class.
#[derive(Clone)]
struct DialZones {
    /// The dial grid this split was cut from — a zone rate is only comparable
    /// across cells cut from the same ladders, and the board carries cells
    /// measured on several different grids.
    grid: String,
    /// Rows in that grid (the ladder population's size).
    n_grid_rows: usize,
    /// `(split_kind, key, zone)` -> counts. `split_kind` is `codec`, `class`
    /// or `all`.
    cells: Vec<(String, String, String, ZoneCell, ZoneLadders)>,
    /// Reviewed content class -> number of grid images carrying it (including
    /// `unclassified`, which is reported and never merged).
    class_images: std::collections::BTreeMap<String, usize>,
    /// The worst individual ladders BY NAME — `(image_id, codec, class, zone,
    /// endpoint delta, rungs, worst backwards step)`. The aggregate rates say
    /// how often; this says WHICH reference image, which is the only form a
    /// reader can go and look at.
    worst_ladders: Vec<(String, String, String, String, f64, usize, f64)>,
}

impl DialMetrics {
    #[allow(clippy::declare_interior_mutable_const)]
    const NAN: Self = Self {
        mono: f64::NAN,
        inversion_truth: InversionTruthReport::EMPTY,
        p5: f64::NAN,
        p95: f64::NAN,
        flat: f64::NAN,
        reach: f64::NAN,
        min: f64::NAN,
        max: f64::NAN,
        addr: None,
        per_codec: Vec::new(),
        curves: Vec::new(),
        zones: None,
    };
}

/// Read an external per-cell dial score table (`image_id\tcodec\tq\tpred`,
/// the shape `ZENSIM_DIAL_PRED_OUT` writes) and align it to `grid` row order.
///
/// The join key is `(image_id, codec, round(q, 4))`. `q` is rounded because
/// the grid stores it as f32-widened f64 for some codecs while a TSV carries
/// the decimal text — the JXL ladder's q-equivalent `100 − 4·distance` is the
/// case that needs it. Fails LOUD on any unmatched grid row rather than
/// scoring a partial grid: a dial panel over a subset of ladders is a
/// different measurement, not a smaller one.
fn load_peer_dial_scores(path: &Path, grid: &parquet_loader::DialGrid) -> Result<Vec<f64>, String> {
    let txt = std::fs::read_to_string(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let mut lut: std::collections::HashMap<(String, String, i64), f64> =
        std::collections::HashMap::new();
    let key = |img: &str, codec: &str, q: f64| -> (String, String, i64) {
        (
            img.to_string(),
            codec.to_string(),
            (q * 10_000.0).round() as i64,
        )
    };
    let mut lines = txt.lines();
    let header = lines.next().ok_or_else(|| format!("{path:?}: empty"))?;
    let cols: Vec<&str> = header.split('\t').collect();
    let ci = |n: &str| cols.iter().position(|c| c.trim() == n);
    let (i_img, i_codec, i_q, i_pred) = (
        ci("image_id").ok_or_else(|| format!("{path:?}: missing image_id column"))?,
        ci("codec").ok_or_else(|| format!("{path:?}: missing codec column"))?,
        ci("q").ok_or_else(|| format!("{path:?}: missing q column"))?,
        ci("pred").ok_or_else(|| format!("{path:?}: missing pred column"))?,
    );
    let mut n_rows = 0usize;
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() <= i_pred.max(i_q).max(i_codec).max(i_img) {
            return Err(format!("{path:?}: short row {line:?}"));
        }
        let q: f64 = f[i_q]
            .trim()
            .parse()
            .map_err(|_| format!("{path:?}: bad q {:?}", f[i_q]))?;
        let v: f64 = f[i_pred]
            .trim()
            .parse()
            .map_err(|_| format!("{path:?}: bad pred {:?}", f[i_pred]))?;
        lut.insert(key(f[i_img].trim(), f[i_codec].trim(), q), v);
        n_rows += 1;
    }
    let mut out = Vec::with_capacity(grid.image_id.len());
    let mut missing: Vec<String> = Vec::new();
    for i in 0..grid.image_id.len() {
        match lut.get(&key(&grid.image_id[i], &grid.codec[i], grid.q[i])) {
            Some(&v) => out.push(v),
            None => {
                if missing.len() < 5 {
                    missing.push(format!(
                        "({}, {}, {})",
                        grid.image_id[i], grid.codec[i], grid.q[i]
                    ));
                }
                out.push(f64::NAN);
            }
        }
    }
    let n_missing = out.iter().filter(|v| v.is_nan()).count();
    if n_missing > 0 {
        return Err(format!(
            "{path:?}: {n_missing} of {} grid rows have no score (table had {n_rows} rows). \
             First unmatched: {}. A dial panel over a subset of ladders is a DIFFERENT \
             measurement — refusing rather than reporting one.",
            grid.image_id.len(),
            missing.join(", ")
        ));
    }
    Ok(out)
}

/// Peer scores for a G-ADDR PROBE, joined to the probe parquet's own `entry`
/// labels. TSV shape `entry\tpred`, mirroring `--dial-peer-scores`'s
/// `image_id\tcodec\tq\tpred`.
///
/// Partial coverage is REFUSED, not padded: a tail depth computed over a subset
/// of an all-negative-truth probe is a different measurement, and the whole
/// point of the probe is that its population is pinned by sha256.
fn load_peer_probe_scores(path: &Path, entries: &[String]) -> Result<Vec<f64>, String> {
    let txt = std::fs::read_to_string(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let mut lines = txt.lines();
    let header = lines.next().ok_or_else(|| format!("{path:?}: empty"))?;
    let cols: Vec<&str> = header.split('\t').collect();
    let ci = |n: &str| cols.iter().position(|c| c.trim() == n);
    let (i_entry, i_pred) = (
        ci("entry").ok_or_else(|| format!("{path:?}: missing entry column"))?,
        ci("pred").ok_or_else(|| format!("{path:?}: missing pred column"))?,
    );
    let mut lut: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let mut n_rows = 0usize;
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() <= i_pred.max(i_entry) {
            return Err(format!("{path:?}: short row {line:?}"));
        }
        let v: f64 = f[i_pred]
            .trim()
            .parse()
            .map_err(|_| format!("{path:?}: bad pred {:?}", f[i_pred]))?;
        lut.insert(f[i_entry].trim().to_string(), v);
        n_rows += 1;
    }
    let mut out = Vec::with_capacity(entries.len());
    let mut missing: Vec<String> = Vec::new();
    for e in entries {
        match lut.get(e) {
            Some(&v) => out.push(v),
            None => {
                if missing.len() < 5 {
                    missing.push(e.clone());
                }
                out.push(f64::NAN);
            }
        }
    }
    if !missing.is_empty() {
        return Err(format!(
            "{path:?}: {} of {} probe rows have no score (table had {n_rows} rows). First \
             unmatched: {}. A probe axis over a subset of its pinned population is a DIFFERENT \
             measurement — refusing rather than reporting one.",
            out.iter().filter(|v| v.is_nan()).count(),
            entries.len(),
            missing.join(", ")
        ));
    }
    Ok(out)
}

/// The G-ADDR probe inputs, already scored by the caller (so the probes go
/// through the SAME `Ensemble` forward every other section uses).
struct AddrProbes {
    /// `(measure, probe sha256)` for the negative-tail probe.
    negtail: Option<(zensim_validate::dial_addressability::NegTailMeasure, String)>,
    /// `(entry -> identity dial, probe sha256)`. `entry` is the dial grid's
    /// `image_id`, so the "above identity" check is PER REFERENCE.
    identity: Option<(Vec<(String, f64)>, String)>,
}

#[allow(clippy::too_many_arguments)]
fn dial_panel(
    ens: &Ensemble,
    grid_path: &Path,
    peer: Option<&(String, PathBuf)>,
    grid_sha256: &str,
    probes: &AddrProbes,
    gaddr_reference: &str,
    gaddr_tail_pins: zensim_validate::dial_addressability::TailPins,
    gaddr_grid_truth: Option<&PathBuf>,
    floor_rule: zensim_validate::dial_addressability::FloorRule,
    value_pins: zensim_validate::dial_addressability::ValuePins,
    inversion_truth: zensim_validate::dial_addressability::InversionTruth,
    reference_truth: Option<&(
        PathBuf,
        zensim_validate::dial_addressability::ButteraugliVariant,
    )>,
    encoder_inversion_census: Option<&PathBuf>,
) -> (String, DialMetrics) {
    if !grid_path.exists() {
        return (
            format!(
                "\n## DIAL panel — ⚠ SKIPPED (grid not found)\n\n\
             Dial grid `{}` is absent. The DIAL panel (G1 range / G3 monotonicity\n\
             / codec reach) is MANDATORY per docs/EVAL_PANEL_REQUIREMENT.md — fetch\n\
             the stored feature grid from `s3://zentrain/eval-grids/` (or set\n\
             `--dial-grid` / `ZENSIM_DIAL_GRID`) and re-run. A rank-only verdict is\n\
             a regression.\n",
                grid_path.display()
            ),
            DialMetrics::NAN,
        );
    }
    let dt = zensim_validate::perf_trace::PerfTrace::new("  dial");
    let grid = match parquet_loader::load_dial_grid(&grid_path.to_path_buf()) {
        Ok(g) => g,
        Err(e) => {
            return (
                format!("\n## DIAL panel — ⚠ FAILED to load grid\n\n`{e}`\n"),
                DialMetrics::NAN,
            );
        }
    };

    dt.mark("dial grid parquet load");
    // Score every grid row through the SAME dispatch path render_corpus uses —
    // UNLESS a peer score table was supplied, in which case the scores come
    // from it and everything downstream (buckets, gates, zones) is identical.
    // That identity is the point: it is what makes an external metric's dial
    // comparable to a bake's.
    let scores: Vec<f64> = match peer {
        None => ens.score_rows(&grid.feature_rows),
        Some((label, path)) => match load_peer_dial_scores(path, &grid) {
            Ok(v) => v,
            Err(e) => {
                return (
                    format!(
                        "\n## DIAL panel — ⚠ FAILED to load peer scores (`{label}`)\n\n`{e}`\n"
                    ),
                    DialMetrics::NAN,
                );
            }
        },
    };
    dt.mark("dial forward (score_rows / peer table)");

    // Optional per-cell prediction dump for external joins against
    // reference-metric sidecars (zone-consistency / HQ-zone instruments).
    // TSV: image_id, codec, q, pred. Enabled via ZENSIM_DIAL_PRED_OUT=<path>.
    if let Ok(path) = std::env::var("ZENSIM_DIAL_PRED_OUT") {
        let mut s = String::from("image_id\tcodec\tq\tpred\n");
        for (i, &sc) in scores.iter().enumerate() {
            use std::fmt::Write as _;
            let _ = writeln!(
                s,
                "{}\t{}\t{}\t{}",
                grid.image_id[i], grid.codec[i], grid.q[i], sc
            );
        }
        match std::fs::write(&path, s) {
            Ok(()) => eprintln!("  wrote dial per-cell predictions to {path}"),
            Err(e) => eprintln!("  dial pred dump failed ({path}): {e}"),
        }
    }

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
    const MATERIAL_INV: f64 = MATERIAL_INV_PT;
    // per-codec: [pairs, material_inversions, flat_clamp, n_curves]
    let mut tot_pairs = 0usize;
    let mut tot_fwd = 0usize; // Δ > MATERIAL_INV — clear quality increase
    let mut tot_inv = 0usize; // strict (any backwards > 1e-9) — diagnostic
    let mut tot_inv_material = 0usize; // backwards by > MATERIAL_INV, DIAL-attributed — gate
    // Two-reference inversion truth (2026-09-05 ruling). `truth` is None when
    // the operative reading is not measurable on this instrument; the run then
    // falls back to `single` and SAYS SO in the panel.
    let mut tot_inv_encoder = 0usize; // charged to the ENCODER, not gated
    let mut tot_inv_unknown = 0usize; // attribution NOT MEASURABLE — stays dial
    let mut tot_flat = 0usize; // distinct features, |Δ| ≤ 1e-9 — metric dead-zone — gate
    let mut tot_codec_sat = 0usize; // identical features — codec quality ceiling — not gated
    let mut tot_subres = 0usize; // 1e-9 < |Δ| ≤ MATERIAL_INV — expected oversampling
    let mut inv_mags: Vec<f64> = Vec::new(); // magnitudes of strict inversions
    let mut per_codec: BTreeMap<String, [usize; 4]> = BTreeMap::new();
    // Ladder-inversion split (2026-08-31): the same five outcomes, bucketed by
    // (codec, zone) and (content class, zone). Nothing above changes — these
    // accumulate alongside so the split ALWAYS reconciles with the pooled gate.
    // The reference-truth table for the 2026-09-05 two-reference ruling. Loaded
    // ONCE, here, so a parse failure is reported before any counting starts.
    // `truth_note` carries the exact sentence the panel prints — including the
    // loud NOT-MEASURABLE fallback, which must never be silent.
    let (truth, truth_note): (Option<gaddr::ReferenceTruth>, String) = match (
        inversion_truth,
        reference_truth,
    ) {
        (gaddr::InversionTruth::Single, _) => (
            None,
            "inversion truth: `single` — EVERY material backwards rung is charged to the dial \
             (the pre-2026-09-05 reading)."
                .to_string(),
        ),
        (gaddr::InversionTruth::Agree, Some((path, variant))) => {
            match gaddr::ReferenceTruth::from_tsv(path, *variant) {
                Ok(t) => {
                    let n = t.len();
                    (
                        Some(t),
                        format!(
                            "inversion truth: `agree` — a material backwards rung is charged to \
                             the ENCODER only where ssim2 (≤ −{:.1} pt) AND butteraugli-{} \
                             (≥ +{:.2} distance) BOTH call the higher setting worse. Reference \
                             table `{}` ({n} cells).",
                            gaddr::ENCODER_SSIM2_MARGIN_PT,
                            variant.tag(),
                            variant.margin(),
                            path.display()
                        ),
                    )
                }
                Err(e) => (
                    None,
                    format!(
                        "inversion truth: **NOT MEASURABLE** — `agree` was requested but the \
                         reference table could not be read ({e}). FALLING BACK to `single`; \
                         every material backwards rung below is charged to the dial."
                    ),
                ),
            }
        }
        (gaddr::InversionTruth::Agree, None) => (
            None,
            "inversion truth: **NOT MEASURABLE** — `agree` is the operative reading but no \
             `--reference-truth` table was supplied for this instrument, so no pair's \
             attribution can be decided. FALLING BACK to `single`: every material backwards \
             rung below is charged to the dial."
                .to_string(),
        ),
    };
    /// One ENCODER-attributed pair: `(image, codec, q_lo, q_hi, dial drop,
    /// (Δssim2, Δbutteraugli))`. This is the codec bug-report evidence.
    type EncoderPair = (String, String, f64, f64, f64, (f64, f64));
    let mut encoder_pairs: Vec<EncoderPair> = Vec::new();
    // The BAKE-INDEPENDENT census: every pair the instrument's own two
    // references agree the encoder ran backwards on, whether or not this
    // scorer noticed. Same rule, same margins — only the filter differs.
    let mut census_rows: Vec<(String, String, f64, f64, f64, f64)> = Vec::new();
    let mut zcells: BTreeMap<(String, String, String), ZoneCell> = BTreeMap::new();
    let mut zladders: BTreeMap<(String, String, String), ZoneLadders> = BTreeMap::new();
    let mut class_images: BTreeMap<String, std::collections::BTreeSet<String>> = BTreeMap::new();
    let mut named_ladders: Vec<(String, String, String, String, f64, usize, f64)> = Vec::new();
    for ((img, codec), pts) in curves.iter_mut() {
        pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let entry = per_codec.entry(codec.clone()).or_default();
        entry[3] += 1;
        let class = zensim_validate::dial_content::class_of(img).to_string();
        class_images
            .entry(class.clone())
            .or_default()
            .insert(img.clone());
        // split keys this ladder contributes to (`all` is the reconciliation row)
        let keys: [(String, String); 3] = [
            ("all".to_string(), "all".to_string()),
            ("codec".to_string(), codec.clone()),
            ("class".to_string(), class.clone()),
        ];
        // per-zone rung indices for the ladder-level (endpoint) statistic
        let mut zone_rungs: BTreeMap<&'static str, Vec<(f64, f64)>> = BTreeMap::new();
        let mut zone_has_inv: BTreeMap<&'static str, bool> = BTreeMap::new();
        for &(q, sc, _) in pts.iter() {
            zone_rungs.entry(zone_of(q)).or_default().push((q, sc));
        }
        for w in pts.windows(2) {
            let (q0, s0, i0) = w[0];
            let (q1, s1, i1) = w[1];
            tot_pairs += 1;
            entry[0] += 1;
            let delta = s1 - s0;
            let zone = zone_of(0.5 * (q0 + q1));
            let strict = delta < -1e-9;
            // five mutually-exclusive buckets summing to tot_pairs:
            //   0 forward | 1 material inversion | 2 codec-saturated
            //   3 flat/clamp dead-zone | 4 sub-resolution
            let bucket: u8 = if delta > MATERIAL_INV {
                0
            } else if delta < -MATERIAL_INV {
                1
            } else if feat_eq(i0, i1) {
                2
            } else if delta.abs() <= 1e-9 {
                3
            } else {
                4
            };
            if strict {
                tot_inv += 1; // strict backwards (diagnostic, all magnitudes)
                inv_mags.push(-delta);
            }
            if encoder_inversion_census.is_some()
                && let Some(t) = truth.as_ref()
                && t.pair_is_encoder_inversion(img, codec, q0, q1) == Some(true)
            {
                let (ds, db) = t
                    .pair_deltas(img, codec, q0, q1)
                    .unwrap_or((f64::NAN, f64::NAN));
                census_rows.push((img.clone(), codec.clone(), q0, q1, ds, db));
            }
            // 2026-09-05 ruling: a material backwards rung on a pair where BOTH
            // reference metrics call the higher setting worse is the ENCODER's,
            // not the dial's. `None` (no reference row) is NOT an exemption.
            let attrib: Option<bool> = if bucket == 1 {
                truth
                    .as_ref()
                    .map(|t| t.pair_is_encoder_inversion(img, codec, q0, q1))
                    .unwrap_or(Some(false))
            } else {
                None
            };
            let encoder_pair = attrib == Some(true);
            let unknown_pair = bucket == 1 && attrib.is_none();
            match bucket {
                0 => tot_fwd += 1, // clear quality increase
                1 if encoder_pair => {
                    tot_inv_encoder += 1; // the CODEC ran backwards — reported, not gated
                    encoder_pairs.push((
                        img.clone(),
                        codec.clone(),
                        q0,
                        q1,
                        -delta,
                        truth
                            .as_ref()
                            .and_then(|t| t.pair_deltas(img, codec, q0, q1))
                            .unwrap_or((f64::NAN, f64::NAN)),
                    ));
                }
                1 => {
                    tot_inv_material += 1; // material backwards, DIAL-attributed (gate)
                    if unknown_pair {
                        tot_inv_unknown += 1;
                    }
                    entry[1] += 1;
                    zone_has_inv.insert(zone, true);
                }
                2 => tot_codec_sat += 1, // codec emitted identical image — not the bake's fault
                3 => {
                    tot_flat += 1; // distinct inputs, identical score — metric dead-zone (gate)
                    entry[2] += 1;
                }
                _ => tot_subres += 1, // 1e-9 < |Δ| ≤ MATERIAL_INV (expected; not gated)
            }
            for (kind, key) in keys.iter() {
                let c = zcells
                    .entry((kind.clone(), key.clone(), zone.to_string()))
                    .or_default();
                c.n_pairs += 1;
                if strict {
                    c.inv_strict += 1;
                }
                match bucket {
                    0 => c.forward += 1,
                    1 if encoder_pair => c.inv_encoder += 1,
                    1 => {
                        c.inv_material += 1;
                        if unknown_pair {
                            c.inv_unknown += 1;
                        }
                        c.inv_mags.push(-delta);
                    }
                    2 => c.codec_sat += 1,
                    3 => c.flat += 1,
                    _ => c.subres += 1,
                }
            }
        }
        // ladder-level: a zone counts this ladder once it holds >= 3 rungs, and
        // the ladder is "endpoint backwards" there when its best-quality rung
        // scores MATERIALLY below its worst-quality rung — a loop pointed at
        // that zone walks the wrong way regardless of the per-rung wiggles.
        for (zone, rungs) in zone_rungs.iter() {
            if rungs.len() < 3 {
                continue;
            }
            let lo = rungs.first().copied().unwrap_or((0.0, 0.0));
            let hi = rungs.last().copied().unwrap_or((0.0, 0.0));
            let backwards = hi.1 - lo.1 < -MATERIAL_INV;
            let had_inv = *zone_has_inv.get(zone).unwrap_or(&false);
            if backwards || had_inv {
                // worst step inside THIS zone of THIS ladder
                let mut worst_step = 0.0f64;
                for w in rungs.windows(2) {
                    let d = w[1].1 - w[0].1;
                    if d < -worst_step {
                        worst_step = -d;
                    }
                }
                named_ladders.push((
                    img.clone(),
                    codec.clone(),
                    class.clone(),
                    zone.to_string(),
                    hi.1 - lo.1,
                    rungs.len(),
                    worst_step,
                ));
            }
            for (kind, key) in keys.iter() {
                let l = zladders
                    .entry((kind.clone(), key.clone(), zone.to_string()))
                    .or_default();
                l.n += 1;
                if had_inv {
                    l.with_inv += 1;
                }
                if backwards {
                    l.endpoint_backwards += 1;
                }
            }
        }
    }
    let dial_zones = {
        // RECONCILIATION (load-bearing): the split_kind=="all" cells must
        // re-derive the pooled gate exactly. A split that quietly disagrees
        // with the number it claims to explain is the failure mode this whole
        // panel exists to prevent, so it aborts rather than ships.
        let a_pairs: usize = zcells
            .iter()
            .filter(|((k, _, _), _)| k == "all")
            .map(|(_, c)| c.n_pairs)
            .sum();
        let a_inv: usize = zcells
            .iter()
            .filter(|((k, _, _), _)| k == "all")
            .map(|(_, c)| c.inv_material)
            .sum();
        let a_flat: usize = zcells
            .iter()
            .filter(|((k, _, _), _)| k == "all")
            .map(|(_, c)| c.flat)
            .sum();
        assert_eq!(
            (a_pairs, a_inv, a_flat),
            (tot_pairs, tot_inv_material, tot_flat),
            "dial zone split does not reconcile with the pooled G3 counters"
        );
        let mut cells: Vec<(String, String, String, ZoneCell, ZoneLadders)> = Vec::new();
        let mut keyset: std::collections::BTreeSet<(String, String, String)> =
            zcells.keys().cloned().collect();
        keyset.extend(zladders.keys().cloned());
        for k in keyset {
            let c = zcells.get(&k).cloned().unwrap_or_default();
            let l = zladders.get(&k).cloned().unwrap_or_default();
            cells.push((k.0.clone(), k.1.clone(), k.2.clone(), c, l));
        }
        // most-backwards endpoint first, then deepest single reversal
        named_ladders.sort_by(|a, b| {
            a.4.partial_cmp(&b.4)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.6.partial_cmp(&a.6).unwrap_or(std::cmp::Ordering::Equal))
        });
        named_ladders.truncate(12);
        DialZones {
            grid: grid_path.display().to_string(),
            n_grid_rows: scores.len(),
            worst_ladders: named_ladders,
            cells,
            class_images: class_images
                .into_iter()
                .map(|(k, v)| (k, v.len()))
                .collect(),
        }
    };
    // forward strict-increase rate; inversion rate; tied rate — three rates
    // that sum to 1. "monotonicity" (G3) = 1 - inversion rate (ties are not
    // inversions but are reported on their own line). The gate uses the
    // MATERIAL inversion count (backwards by > MATERIAL_INV score units);
    // sub-MATERIAL backwards wiggles fold into the tied/dead-zone bucket.
    //
    // 2026-09-05 RULING: `tot_inv_material` now holds only the DIAL-attributed
    // material inversions — encoder-attributed pairs were counted into
    // `tot_inv_encoder` above and never entered it. So `mono` is the dial's own
    // monotonicity under `agree`, and byte-identical to the pre-ruling number
    // under `single` (where `tot_inv_encoder` is 0 by construction).
    let inv_rate_all_material = if tot_pairs > 0 {
        (tot_inv_material + tot_inv_encoder) as f64 / tot_pairs as f64
    } else {
        f64::NAN
    };
    let mono_single = 1.0 - inv_rate_all_material;
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
    // Full cross-codec reach = pooled max − min (sorted above).
    let reach = if pooled.is_empty() {
        f64::NAN
    } else {
        pooled[pooled.len() - 1] - pooled[0]
    };
    let g1 = soft_gate(p5, 50.0, 25.0).min(soft_gate(p95, 50.0, 85.0));
    let g3 = soft_gate(mono, 0.90, 0.93).min(soft_gate(flat, 0.10, 0.05));

    let mut s = String::new();
    match peer {
        None => s.push_str(
            "\n## DIAL panel (codec-target G1/G3 — densified multi-codec q-sweep)\n\n",
        ),
        Some((label, path)) => s.push_str(&format!(
            "\n## DIAL panel — **PEER `{label}`** (codec-target G1/G3 — densified multi-codec q-sweep)\n\n\
             > ⚠ **These dial numbers describe `{label}`, NOT the `--bake` this report's RANK \
             panel describes.** Per-cell scores were read from `{}` and every downstream rule \
             (five-bucket split, {MATERIAL_INV_PT}-pt materiality, zone cuts, ladder accounting) is \
             the one every bake takes — that identity is the whole point of the mode. \
             `--full-json`/`--fulleval` are refused here so this can never reach a board row.\n\n",
            path.display()
        )),
    }
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
    // 2026-09-05 ruling: show the encoder-attributed half beside the dial's own
    // number, ALWAYS — including when it is zero, so a reader can tell "no
    // encoder inversions" from "the rule did not run".
    if tot_inv_encoder > 0 || truth.is_some() {
        s.push_str(&format!(
            "| ↳ charged to the ENCODER (both refs agree) | {} | — (codec, not dial) | |\n",
            tot_inv_encoder
        ));
        s.push_str(&format!(
            "| ↳ monotonicity under `single` (pre-ruling) | {mono_single:.4} | (reported) | |\n"
        ));
    }
    if tot_inv_unknown > 0 {
        s.push_str(&format!(
            "| ↳ attribution NOT MEASURABLE (kept on the dial) | {} | — | |\n",
            tot_inv_unknown
        ));
    }
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
    // The bake-independent census, if asked for. Written even when EMPTY: a
    // zero-row file with a header is the honest record that the rule ran and
    // found nothing, which an absent file cannot say.
    if let Some(path) = encoder_inversion_census {
        let variant = truth
            .as_ref()
            .map(|t| t.variant().tag())
            .unwrap_or("<not measurable>");
        let mut out = String::new();
        out.push_str(&format!(
            "# encoder-inversion census, rule `two-reference-agree-2026-09-05`\n\
             # grid = {}\n# reference table = {}\n\
             # ssim2 margin = {:.1} pt; butteraugli-{} margin = {:.2} distance\n\
             # BAKE-INDEPENDENT: every adjacent DISTINCT-setting pair both references\n\
             # agree the higher setting is worse on. This is the CODEC's behaviour,\n\
             # not any dial's.\n",
            grid_path.display(),
            truth.as_ref().map(|t| t.source()).unwrap_or("<none>"),
            gaddr::ENCODER_SSIM2_MARGIN_PT,
            variant,
            truth
                .as_ref()
                .map(|t| t.variant().margin())
                .unwrap_or(f64::NAN),
        ));
        out.push_str("image_id\tcodec\tq_lo\tq_hi\td_ssim2\td_butteraugli\n");
        let mut rows = census_rows.clone();
        rows.sort_by(|a, b| {
            (a.1.as_str(), a.0.as_str(), a.2)
                .partial_cmp(&(b.1.as_str(), b.0.as_str(), b.2))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (img, codec, q0, q1, ds, db) in rows.iter() {
            out.push_str(&format!("{img}\t{codec}\t{q0}\t{q1}\t{ds:.6}\t{db:.6}\n"));
        }
        match std::fs::write(path, &out) {
            Ok(()) => eprintln!(
                "encoder-inversion census: {} pairs -> {}",
                census_rows.len(),
                path.display()
            ),
            Err(e) => eprintln!("encoder-inversion census: FAILED to write {path:?}: {e}"),
        }
    }
    // The 2026-09-05 two-reference ruling: which reading this run used, and —
    // when the operative reading ran — the pairs it charged to the ENCODER,
    // named, so a codec tracking issue can be written from this table alone.
    s.push_str(&format!("\n_{truth_note}_\n"));
    if !encoder_pairs.is_empty() {
        s.push_str(&format!(
            "\n**ENCODER-attributed inversions ({} pairs)** — both reference metrics \
             independently call the higher setting worse, so these are the CODEC running \
             backwards, not the dial. File or update the codec's tracking issue.\n\n",
            encoder_pairs.len()
        ));
        s.push_str(
            "| image | codec | step | dial drop | Δssim2 | Δbutteraugli |\n|---|---|---|--:|--:|--:|\n",
        );
        let mut rows = encoder_pairs.clone();
        rows.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
        for (img, codec, q0, q1, drop, (ds, db)) in rows.iter().take(25) {
            s.push_str(&format!(
                "| `{img}` | {codec} | q{q0}→q{q1} | −{drop:.2} | {ds:+.3} | {db:+.4} |\n"
            ));
        }
        if rows.len() > 25 {
            s.push_str(&format!(
                "\n_{} further pairs omitted; the full set is in `--full-json` \
                 `dial.inversion_truth.encoder_pairs`._\n",
                rows.len() - 25
            ));
        }
    }
    // Ladder inversions WHERE they happen: codec x quality zone, then content
    // class x quality zone. The pooled `inversions` row above is one number for
    // a codec loop; this is the number for the loop the reader is actually
    // running, at the quality it is running at.
    s.push_str(
        "\nLadder inversions by quality zone (rung pairs keyed on the pair's q midpoint):\n\n",
    );
    s.push_str(
        "| split | key | zone | rung pairs | inversions | inv rate | worst step | flat | ladders | ≥1 inv | ends backwards |\n",
    );
    s.push_str("|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|\n");
    {
        let mut rows: Vec<&(String, String, String, ZoneCell, ZoneLadders)> =
            dial_zones.cells.iter().collect();
        // report order: all, then codec rows, then class rows; zones ascending
        let kind_rank = |k: &str| match k {
            "all" => 0,
            "codec" => 1,
            _ => 2,
        };
        let zone_rank = |z: &str| ZONE_LABELS.iter().position(|x| *x == z).unwrap_or(9);
        rows.sort_by(|a, b| {
            kind_rank(&a.0)
                .cmp(&kind_rank(&b.0))
                .then(a.1.cmp(&b.1))
                .then(zone_rank(&a.2).cmp(&zone_rank(&b.2)))
        });
        for (kind, key, zone, c, l) in rows {
            let rate = if c.n_pairs > 0 {
                c.inv_material as f64 / c.n_pairs as f64
            } else {
                f64::NAN
            };
            let flat_r = if c.n_pairs > 0 {
                c.flat as f64 / c.n_pairs as f64
            } else {
                f64::NAN
            };
            let worst = c
                .inv_mags
                .iter()
                .copied()
                .fold(0.0f64, |a, b| if b > a { b } else { a });
            let lad = if l.n > 0 {
                format!(
                    "{} | {} | {}",
                    l.n,
                    format_args!("{:.0}%", 100.0 * l.with_inv as f64 / l.n as f64),
                    format_args!("{:.0}%", 100.0 * l.endpoint_backwards as f64 / l.n as f64)
                )
            } else {
                "0 | — | —".to_string()
            };
            s.push_str(&format!(
                "| {kind} | {key} | {zone} | {} | {} | {rate:.4} | {worst:.1} | {flat_r:.4} | {lad} |\n",
                c.n_pairs, c.inv_material
            ));
        }
        let classes: Vec<String> = dial_zones
            .class_images
            .iter()
            .map(|(k, v)| format!("{k} {v}"))
            .collect();
        s.push_str(&format!(
            "\n_Zones cut on the grid's normalised q axis at {:?}: `q<50` aggressive web \
             compression, `q50-85` ordinary web quality, `q>=85` high-fidelity / \
             near-lossless. A rung pair is assigned by its q MIDPOINT. `inversions` counts \
             the SAME material (>{MATERIAL_INV}pt backwards) events the pooled G3 gate \
             counts, so the `all` rows sum to the pooled numbers exactly. `ladders` = \
             (image, codec) curves with ≥3 rungs in the zone; `≥1 inv` = share carrying at \
             least one material backwards rung there; **`ends backwards`** = share whose \
             best-quality rung in the zone scores materially BELOW its worst-quality rung \
             — a codec loop pointed at that zone walks the wrong way. Content classes are \
             a recorded HAND review of the grid's reference images \
             (`benchmarks/dial_grid_content_classes_2026-08-31.tsv`), image counts: {}. \
             Read the n column: a class with 3 images carries 12 ladders and its rate is \
             coarse._\n",
            ZONE_EDGES,
            classes.join(", ")
        ));
    }

    // Per-codec breakout + aggregated dial curves for --full-json (the
    // markdown table above prints the same numbers; this is the structured
    // twin the dashboard reads — never re-derived downstream).
    let per_codec_json: Vec<PerCodecDial> = per_codec
        .iter()
        .map(|(codec, c)| PerCodecDial {
            codec: codec.clone(),
            n_curves: c[3],
            n_pairs: c[0],
            mono: if c[0] > 0 {
                1.0 - c[1] as f64 / c[0] as f64
            } else {
                f64::NAN
            },
            tied: if c[0] > 0 {
                c[2] as f64 / c[0] as f64
            } else {
                f64::NAN
            },
        })
        .collect();
    // Bucket scores by (codec, q) — q keyed at 1e-3 resolution (the densified
    // grid's finest step is 0.025) — then per-q p25/median/p75 across ladders.
    let mut by_cq: BTreeMap<String, BTreeMap<i64, Vec<f64>>> = BTreeMap::new();
    for ((_img, codec), pts) in curves.iter() {
        let m = by_cq.entry(codec.clone()).or_default();
        for &(q, score, _) in pts.iter() {
            m.entry((q * 1000.0).round() as i64)
                .or_default()
                .push(score);
        }
    }
    let pctl = |v: &[f64], p: f64| -> f64 {
        let pos = p * (v.len() as f64 - 1.0);
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(v.len() - 1);
        let frac = pos - lo as f64;
        v[lo] * (1.0 - frac) + v[hi] * frac
    };
    let curve_json: Vec<CodecCurve> = by_cq
        .into_iter()
        .map(|(codec, qs)| CodecCurve {
            codec,
            pts: qs
                .into_iter()
                .map(|(qk, mut ss)| {
                    ss.sort_by(f64::total_cmp);
                    (
                        qk as f64 / 1000.0,
                        pctl(&ss, 0.25),
                        pctl(&ss, 0.50),
                        pctl(&ss, 0.75),
                    )
                })
                .collect(),
        })
        .collect();

    // ── G-ADDR: the dial ADDRESSABILITY gate ──────────────────────────────
    // Runs on the SAME pooled score vector G1 reads, plus the SAME ladder
    // accounting G3 reads, so no number here can disagree with the table
    // above about the same event. See `zensim_validate::dial_addressability`
    // for why the FLOOR axis is the hard one (the anchor's `max(ssim2, 0)`
    // clamp is what deletes the in-distribution evidence below zero).
    let (dmin, dmax) = if pooled.is_empty() {
        (f64::NAN, f64::NAN)
    } else {
        (pooled[0], pooled[pooled.len() - 1])
    };
    let addr_measure = gaddr::GridMeasure::from_pooled(&scores, mono, flat);
    // Identity: MEASURED 2026-09-04 — `ref == dist` gives the all-zero feature
    // vector for every image, so the identity dial is a SCALAR property of the
    // bake, not a per-image one. Every grid cell is therefore compared against
    // the probe's `dial_max` (the most permissive identity value), which makes
    // a nonzero above-identity count unambiguous.
    let identity_measure = probes.identity.as_ref().map(|(rows, sha)| {
        let mut dials: Vec<f64> = rows.iter().map(|(_, v)| *v).collect();
        dials.sort_by(f64::total_cmp);
        let bars = gaddr::fixed_bars();
        let n_outside = dials
            .iter()
            .filter(|d| **d < bars.identity_lo || **d > bars.identity_hi)
            .count();
        let idl = dials.last().copied().unwrap_or(f64::NAN);
        let mut n_above = 0usize;
        let mut worst: Option<(String, String, f64, f64, f64)> = None;
        if idl.is_finite() {
            for (i, &sc) in scores.iter().enumerate() {
                if sc - idl > gaddr::ABOVE_IDENTITY_SLACK {
                    n_above += 1;
                    if worst.as_ref().is_none_or(|w| sc > w.3) {
                        worst = Some((
                            grid.image_id[i].clone(),
                            grid.codec[i].clone(),
                            grid.q[i],
                            sc,
                            idl,
                        ));
                    }
                }
            }
        }
        let med = if dials.is_empty() {
            f64::NAN
        } else {
            dials[dials.len() / 2]
        };
        (
            gaddr::IdentityMeasure {
                n: dials.len(),
                dial_min: dials.first().copied().unwrap_or(f64::NAN),
                dial_median: med,
                dial_max: idl,
                n_outside_band: n_outside,
                n_above_identity: n_above,
                n_grid_cells_compared: scores.len(),
                n_grid_cells_total: scores.len(),
                worst,
            },
            sha.clone(),
        )
    });
    // ── PER-CODEC FLOOR REPRESENTABILITY (A7r) ──
    // USER RULING 2026-09-05 (operative): "i care that the lowest configurable
    // settings per codec are representable, not that negative fifty is in that
    // specifically." The dial grid is the only instrument on hand that carries
    // codec identity AND quality ladders — the negative-tail probes carry
    // neither — so the per-codec rows are built here, from the SAME pooled
    // score vector G1 and A1-A6 read. Per-ROW reference truth is optional and
    // feeds only the report-only column folded in from the dropped A9r.
    let grid_truth: Option<Vec<f64>> =
        gaddr_grid_truth.and_then(|p| match load_peer_dial_scores(p, &grid) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!(
                    "bake_verdict: --gaddr-grid-truth {} failed to load: {e} — the per-codec \
                     report columns will read NOT MEASURED",
                    p.display()
                );
                None
            }
        });
    let instrument_name = grid_path
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| grid_path.display().to_string());
    let floor_measure = gaddr::FloorMeasure::from_grid_with_rule(
        &instrument_name,
        &grid.image_id,
        &grid.codec,
        &grid.q,
        &scores,
        grid_truth.as_deref(),
        gaddr::report_floor_threshold(),
        floor_rule,
    );
    // `resolvable`/`spaced` have no registry entry for A7r — they are
    // report-derived windows, never pinned — so their bar is the MENTOR
    // scored AGAINST ITS OWN per-cell truth, through the IDENTICAL
    // `from_grid_with_rule` call the candidate went through. This is what
    // makes the bar "computed through the owner", never a re-implementation
    // beside it. `distinct` needs none of this — its bar is the registry.
    let mentor_floor_measure: Option<gaddr::FloorMeasure> = if floor_rule.needs_mentor_truth() {
        grid_truth.as_deref().map(|gt| {
            gaddr::FloorMeasure::from_grid_with_rule(
                &instrument_name,
                &grid.image_id,
                &grid.codec,
                &grid.q,
                gt,
                Some(gt),
                gaddr::report_floor_threshold(),
                floor_rule,
            )
        })
    } else {
        None
    };
    let addr_verdict = gaddr::evaluate_full(
        gaddr_reference,
        gaddr_tail_pins,
        grid_sha256,
        &grid_path.display().to_string(),
        &addr_measure,
        probes.negtail.as_ref().map(|(m, sha)| (m, sha.as_str())),
        identity_measure.as_ref().map(|(m, sha)| (m, sha.as_str())),
        Some(&floor_measure),
        gaddr::FloorRuleContext {
            rule: floor_rule,
            mentor: mentor_floor_measure.as_ref(),
        },
        value_pins,
    );
    s.push_str(&gaddr::render_markdown(&addr_verdict));

    (
        s,
        DialMetrics {
            mono,
            inversion_truth: InversionTruthReport {
                effective: if truth.is_some() {
                    gaddr::InversionTruth::Agree.tag().to_string()
                } else {
                    gaddr::InversionTruth::Single.tag().to_string()
                },
                requested: inversion_truth.tag().to_string(),
                source: truth.as_ref().map(|t| t.source().to_string()),
                variant: truth.as_ref().map(|t| t.variant().tag().to_string()),
                n_reference_cells: truth.as_ref().map(|t| t.len()).unwrap_or(0),
                n_encoder: tot_inv_encoder,
                n_unknown: tot_inv_unknown,
                mono_single,
                note: truth_note.clone(),
                encoder_pairs: encoder_pairs
                    .iter()
                    .map(|(i, c, a, b, d, (ds, db))| (i.clone(), c.clone(), *a, *b, *d, *ds, *db))
                    .collect(),
            },
            p5,
            p95,
            flat,
            reach,
            min: dmin,
            max: dmax,
            addr: Some(addr_verdict),
            per_codec: per_codec_json,
            curves: curve_json,
            zones: Some(dial_zones),
        },
    )
}

/// One provenance row: `(display name, resolved path, sha256, byte size)`.
/// Named because the hashing pre-pass collects `Result<Option<_>, String>` of
/// it across rayon, and the bare tuple in that position is unreadable.
type CorpusProv = (String, PathBuf, String, u64);

fn render_corpus(
    corpus: &Corpus,
    features_root: &Path,
    regime_720: bool,
    ens: &Ensemble,
) -> Result<CorpusResult, String> {
    let fname = if regime_720 {
        slot_720_file(corpus.name, features_root)
            .ok_or_else(|| format!("no 720 slot for corpus {}", corpus.name))?
    } else {
        resolve_372_slot(corpus, features_root).file
    };
    let path = features_root.join(&fname);
    let ct = zensim_validate::perf_trace::PerfTrace::new("  corpus");
    let mut g = parquet_loader::load_parquet(&path, corpus.display, "human_score", 1.0)
        .map_err(|e| format!("load {} parquet: {e}", corpus.display))?;
    ct.mark("parquet load");
    let humans = std::mem::take(&mut g.human_scores);
    let ref_ids = g.ref_ids.take();

    // Score every row through the ensemble (k=1 ⇒ the single-bake path,
    // bit-identically). The f32 scratch buffer is reused across rows inside
    // `score_grid_one` to avoid the per-row allocation that would otherwise
    // dominate wall time on the bigger corpora (KADID has 10k rows × 372 f32s).
    let scores: Vec<f64> = ens.score_rows(&g.feature_rows);
    // Release the feature matrix the instant the forward is done. It is by
    // far the largest allocation in a corpus (11 356 rows × 944 f64 ≈ 86 MB
    // on the biggest one) and NOTHING below reads it — while the stats tail
    // that follows is the longest phase. Holding it there is what made the
    // now-parallel corpus loop peak at 2.9 GB instead of 0.9 GB.
    drop(g);
    ct.mark("MLP forward (score_rows)");

    let n = scores.len();

    // (The diagnostic per-pair dump moved to `main` when the corpus loop went
    // parallel — see the `per_pair_output` block there. Same bytes, same
    // "last corpus wins" rule, now deterministic.)

    let (srocc, plcc, krocc, or_, pw, z, ds) = aggregate_panel(&scores, &humans);
    ct.mark("aggregate_panel (srocc/plcc/krocc/OR/PWRC/Z-RMSE/DS-AUC)");
    // Grouping by reference image is this binary's call; the statistic is
    // zenstats' (the canonical stats home — never re-derive it here).
    // Orientation::Auto, not the default-signed reading: `compute_panel`
    // reports `srocc.abs()`, so a distance-shaped bake (every raw RankNet bake
    // before its spline — pred range here runs NEGATIVE) shows a healthy pooled
    // number while its signed per-ref reads as a total inversion. Measured
    // 2026-07-15 before the fix: pooled +0.8842 beside per-ref -0.9596 / "100%
    // backwards" on CID22 for a bake whose ladders were all CORRECT. Auto
    // resolves polarity once from the pooled sign, so this column and the panel
    // it sits next to cannot contradict each other — while a ladder that
    // disagrees with the bake's own polarity still reports negative.
    let per_ref = ref_ids.as_ref().and_then(|r| {
        // Orientation::Auto infers polarity from the POOLED sign, so on a corpus
        // where the bake is globally inverted it silently re-points the per-ref
        // stat at the inversion and prints "+0.95 / 0% backwards" — which reads as
        // "every ladder correct" when every ladder is backwards. On a
        // quality-oriented corpus the truth direction is KNOWN, so pin it: an
        // inverted bake then shows a negative per-ref mean and a high %bwd, which
        // is the whole point of the stat. `konjnd` keeps Auto — its validation
        // target is a PJND threshold whose sign is structurally negative.
        // (2026-08-04, benchmarks/sota944_campaign_2026-08-03.md APPENDIX F.)
        let orient = if sign_is_meaningful(corpus.name) {
            Orientation::HigherIsBetter
        } else {
            Orientation::Auto
        };
        per_group_srocc(&scores, &humans, r, PER_REF_MIN_ROWS, orient)
    });
    // Signed SROCC (polarity-preserving) + marginal bootstrap CI. `aggregate_panel`
    // returns `|SROCC|`; a globally-inverted bake would hide behind that abs, so
    // keep the sign here. The CI resolves whether a 3-decimal ranking gap is real.
    let srocc_signed = spearman(&scores, &humans);
    ct.mark("per-ref SROCC + signed SROCC");
    let srocc_ci = bootstrap_srocc_ci(&scores, &humans);
    ct.mark("bootstrap SROCC CI (1000 resamples)");

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
        "_Z-RMSE + OR use corpus-wide σ. Per-stimulus observer σ is NOT in the eval \
feature parquets, so `zenstats`' per-sample forms (`z_rmse_per_sample` / \
`outlier_ratio_per_sample`, Mohammadi Eq 6 / P.1401) are unreachable here. The σ \
EXISTS in the sources (TID `mos_std.txt`; KADID `raw_crowdsource_data.csv`; CID22 SOS) \
and the loader already supports it (`metric_sigmas`, `ZENSIM_SIGMA_MSE`) — joining it \
into the eval parquets is a queued re-extraction. Non-blocking: OR is now a catastrophe \
GATE (G-OR), not a ranking column, so the corpus-σ approximation no longer feeds any \
ordering. Rescale is 4-parameter logistic (Mohammadi 2025), not affine — affine inflates \
Z-RMSE on nonlinear metrics ~30× because saturation regions dominate the residual._\n",
    );

    if let Some(pr) = &per_ref {
        body.push('\n');
        body.push_str("### Per-reference SROCC (within-image rank agreement)\n\n");
        body.push_str("| n refs | mean | median | % refs backwards | % refs perfect |\n");
        body.push_str("|---:|---:|---:|---:|---:|\n");
        body.push_str(&format!(
            "| {} | {:+.4} | {:+.4} | {:.0}% | {:.0}% |\n",
            pr.n_groups,
            pr.mean,
            pr.median,
            pr.frac_negative * 100.0,
            pr.frac_perfect * 100.0,
        ));
        body.push('\n');
        body.push_str(&format!(
            "_Aggregate SROCC here is {srocc:+.4} vs per-ref mean {:+.4} (gap {:+.4}). \
The aggregate mixes two questions: does the bake order ONE image's distortions \
correctly, and does it put different images on a common scale? A large gap means \
the pooled number is dominated by cross-image scale, not ranking — the documented \
AIC-3 confound (0.79 pooled / 0.93 per-ref). **'% refs backwards' is the one to \
read**: a bake can post a healthy pooled SROCC while ranking most individual \
ladders in reverse, which is invisible to every pooled and per-band stat above \
(§8.39 measured exactly that: -0.144 per-ref, 60% backwards, on a corpus the \
pooled panel had no complaint about). Refs with <3 rows or a degenerate target \
are excluded._\n",
            pr.mean,
            pr.mean - srocc,
        ));
    }

    let mut band_rows: Vec<BandRow> = Vec::new();
    if corpus.enable_per_band {
        body.push('\n');
        body.push_str(&format!(
            "### {} per-band full Mohammadi panel (PRIMARY release gate)\n\n",
            corpus.display
        ));
        body.push_str(
            "| Band | range | n | span | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |\n",
        );
        body.push_str("|---|---|--:|--:|---:|---:|---:|---:|---:|---:|---:|\n");
        // Per-band cuts: every corpus that hits this branch has human_score
        // normalized into [0, 1] per the feature-extractor convention.
        //
        // The edges come from `zensim_validate::bands` — the single owner —
        // and are a function of the TARGET column alone, so every model
        // evaluated on a corpus gets identical bands and the cross-bake band
        // table stays comparable. See that module for why the fixed-decile
        // grid was replaced (campaign appendix V).
        //
        // Bands are disjoint row subsets, each running its own
        // `aggregate_panel` (whose Kendall + PWRC are O(n_band²)) — nothing is
        // shared, so they run on rayon and are emitted in band order below.
        let band_defs = bands::merged_bands(&humans);
        let banded: Vec<(String, BandRow)> = band_defs
            .par_iter()
            .map(|bd| {
            let label = bd.label.clone();
            let range_label = bd.range_label();
            let idxs = bd.members(&humans);
            let span = if idxs.is_empty() {
                0.0
            } else {
                let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
                for &i in &idxs {
                    mn = mn.min(humans[i]);
                    mx = mx.max(humans[i]);
                }
                mx - mn
            };
            // NOT-MEASURED is an explicit state with a reason, never a
            // silently dropped row and never a measured zero: a band that
            // cannot resolve must not be rankable by anything downstream.
            let reason = bands::not_measured_reason(idxs.len(), span);
            if reason.is_some() || idxs.len() < 4 {
                let why = reason.clone().unwrap_or_else(|| {
                    format!("n={}: too few pairs for a panel", idxs.len())
                });
                let md = format!(
                    "| {label} | {range_label} | {} | {span:.4} | NOT-MEASURED ({why}) | | | | | | |\n",
                    idxs.len()
                );
                return (
                    md,
                    BandRow {
                        band: label,
                        lo: bd.lo,
                        hi: bd.hi,
                        n: idxs.len(),
                        span,
                        not_measured_reason: Some(why),
                        srocc: f64::NAN,
                        srocc_signed: f64::NAN,
                        plcc: f64::NAN,
                        krocc: f64::NAN,
                        or_ratio: f64::NAN,
                        pwrc: f64::NAN,
                        z_rmse: f64::NAN,
                        mae: f64::NAN,
                    },
                );
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
            // The band table renders the SIGNED Spearman through the same
            // `srocc_cell` surface the aggregate row uses, so an inverted tail
            // is loud instead of indistinguishable from a healthy one.
            let b_srocc_signed = zensim_validate::panel::spearman(&h_b, &s_b);
            let md = format!(
                "| {label} | {range_label} | {} | {span:.4} | {} | {b_plcc:.4} | {b_krocc:.4} | {b_or:.4} | {b_pwrc:.4} | {b_z:.3} | {mae:.4} |\n",
                idxs.len(),
                srocc_cell(corpus.name, b_srocc, b_srocc_signed)
            );
            (
                md,
                BandRow {
                    band: label,
                    lo: bd.lo,
                    hi: bd.hi,
                    n: idxs.len(),
                    span,
                    not_measured_reason: None,
                    srocc: b_srocc,
                    srocc_signed: b_srocc_signed,
                    plcc: b_plcc,
                    krocc: b_krocc,
                    or_ratio: b_or,
                    pwrc: b_pwrc,
                    z_rmse: b_z,
                    mae,
                },
            )
            })
            .collect();
        for (md, row) in banded {
            body.push_str(&md);
            band_rows.push(row);
        }
        body.push('\n');
        body.push_str(
            "_Bands are cut by `zensim_validate::bands` (scheme \
`merged-decile-2026-08-06`): fixed deciles accumulated into the finest \
partition whose every band holds n ≥ 1000 pairs spanning ≥ 0.08 of target. \
A band that cannot clear both floors is NOT-MEASURED with its reason — read \
that as 'not measured', never as zero. **Read a band with its `span`**: range \
restriction attenuates a narrow band's correlation toward 0 while leaving its \
noise alone, so band values are not comparable ACROSS bands. SROCC is signed. \
MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._\n",
        );
        // Legacy 4-band CID22 Table-5 cuts alongside the per-band grid
        // (CLAUDE.md per-band mandate: report both on the CID22 corpus).
        if corpus.name == "cid22" {
            body.push_str(&eval_report::four_band_section(&scores, &humans));
        }
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

    ct.mark("10-band panel + markdown");
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
        srocc_signed,
        srocc_ci,
        bands: corpus.enable_per_band.then_some(band_rows),
        rescaled_scores: scores.clone(),
        name: corpus.name,
        humans: humans.clone(),
        per_ref,
        ref_ids,
        body,
    })
}

/// Render the `--per-pair-output` TSV. 2 columns by default; `refs`
/// (the loader's interned group ids, aligned 1:1) appends the third `ref`
/// column for downstream per-ref statistics (`--per-pair-refs`).
/// Factored out of `main` so the format is unit-testable.
fn format_per_pair(humans: &[f64], preds: &[f64], refs: Option<&[u32]>) -> String {
    let mut s = String::from(if refs.is_some() {
        "human\tpred\tref\n"
    } else {
        "human\tpred\n"
    });
    match refs {
        Some(r) => {
            for ((h, p), g) in humans.iter().zip(preds.iter()).zip(r.iter()) {
                s.push_str(&format!("{h}\t{p}\t{g}\n"));
            }
        }
        None => {
            for (h, p) in humans.iter().zip(preds.iter()) {
                s.push_str(&format!("{h}\t{p}\n"));
            }
        }
    }
    s
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

/// The RULER a verdict was produced on, recorded INSIDE the artifact.
///
/// # Why this exists
///
/// Until 2026-08-30 a `--full-json` verdict carried the corpus *values* but
/// never the *root* they came from. `regime` is not that fact — it is a
/// campaign flag string, and it reads `"720"` cosmetically on board JSONs that
/// are not 720 reads — so "which ruler produced this row?" was answerable only
/// by re-running the bake against each candidate root and diffing per-pair
/// predictions. That is not hypothetical: it is exactly the archaeology that
/// established **7 of 9** apparently-stored-era board rows were `--regime 720`
/// ext720 reads and only two were genuine stored-root reads
/// (`benchmarks/eval372_current_root_2026-08-30.md` §7). One field would have
/// answered it in a grep.
///
/// It matters most where it is easiest to get wrong: the `--regime 372` default
/// root MOVED on 2026-08-30 (`a25d1b80`), so the same flagless command means a
/// different ruler before and after that commit, and the era shift between the
/// two 372 roots is model-specific — a number read on one cannot be corrected
/// into the other, only re-verdicted.
///
/// # Cost
///
/// Zero extra hashing: `corpus_prov` is the vector the provenance pre-pass
/// already built, so this reuses those sha256s. No statistic is recomputed.
/// `corpus_prov` also carries the dial grid, which lives OUTSIDE the features
/// root — entries are filtered by path prefix so this block describes the
/// features root and nothing else (the dial grid keeps its own provenance row
/// in the markdown header, and its era caveat is registry-tracked as
/// `dial372-grid-thread-dependent-era-2026-08-30`).
/// The `feature_set` block of `--full-json` (`docs/FEATURE_SET_IDS.md` §6.1).
///
/// Recomputes nothing the board could get wrong: the ids come from
/// `zensim_validate::feature_set`, which owns the derivation, and the
/// mismatch list is exactly what `check` returned.
fn feature_set_block(model: &Model, root: &Path) -> serde_json::Value {
    use zensim_validate::feature_set;
    let era = feature_set::bake_declared_training_set(model)
        .map(|id| id.era().to_string())
        .unwrap_or_else(|| zensim::feature_set_id::ERA_UNKNOWN.to_string());
    let bake = feature_set::bake_feature_set_ref(model, &era).ok();
    let table = feature_set::root_feature_set_ref(root);
    let mismatches: Vec<serde_json::Value> = match (&bake, &table) {
        (Some(b), Some(t)) => feature_set::check(b, t)
            .into_iter()
            .map(|m| serde_json::json!({"kind": format!("{:?}", m.kind), "detail": m.detail}))
            .collect(),
        _ => Vec::new(),
    };
    serde_json::json!({
        "bake": bake.as_ref().map(|b| b.id.to_string()),
        "bake_slots": bake.as_ref().map(|b| b.slots.to_string()),
        "bake_era_source": if era == zensim::feature_set_id::ERA_UNKNOWN {
            "not declared (no zentrain.feature_set_id in the bake)"
        } else {
            "bake metadata zentrain.feature_set_id"
        },
        "table": table.as_ref().map(|t| t.id.to_string()),
        "table_source": table.as_ref().map(|t| t.source.clone()),
        // An INFERRED table id is evidence about the root's NAME, never about
        // its BYTES — consumers must badge it.
        "inferred": table.as_ref().map(|t| t.inferred).unwrap_or(true),
        "mismatches": mismatches,
    })
}

fn features_root_block(
    root: &Path,
    corpus_prov: &[(String, PathBuf, String, u64)],
) -> serde_json::Value {
    let manifest = root.join("_MANIFEST.json");
    let manifest_sha = zensim_validate::train_manifest::sha256_file(&manifest).ok();
    let files: Vec<serde_json::Value> = corpus_prov
        .iter()
        .filter(|(_, path, _, _)| path.starts_with(root))
        .map(|(name, path, sha, bytes)| {
            serde_json::json!({
                "name": name,
                "file": path
                    .file_name()
                    .map(|f| f.to_string_lossy().to_string())
                    .unwrap_or_default(),
                "sha256": sha,
                "bytes": bytes,
            })
        })
        .collect();
    serde_json::json!({
        "path": root.display().to_string(),
        // The registered one-line era label. An unregistered root is reported as
        // unknown rather than guessed — a wrong era label is worse than none.
        "era": era_of(root),
        // The root's own manifest: null when it has none (an unmanaged root).
        "manifest_sha256": manifest_sha,
        "declared_regime": root_declared_regime(root),
        // Exactly the files under this root that this run read, by content.
        "corpus_files": files,
    })
}

/// Render the provenance header: everything needed to reproduce every number
/// in this report, named by CONTENT rather than by location.
///
/// # Why this exists
///
/// Until 2026-07-15 a verdict's header was:
///
/// ```text
/// - Bake: `/mnt/v/output/zensim/r7_rust/seed7_hf0.bin`
/// - Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
/// ```
///
/// That is not citable. It names a file on one machine's scratch volume, with
/// no hash, no row count, and no code version. Six months later the path is
/// gone and the number is unfalsifiable — which is precisely the failure mode
/// `zensim/weights/manifests/README.md` was written about (the V32
/// recipe-archaeology incident: a recipe reconstructed from prose alone scored
/// CID22 0.295 against a documented 0.8879).
///
/// It also could not answer "which corpus?" when two plausibly-named ones
/// exist. They did: `2026-05-15-full-features/cid22_features_372col…` (this
/// binary's default UNTIL 2026-08-30 — the current-extractor root reuses the
/// same file name, so the ROOT is the only thing that disambiguates; sha
/// `a1050ace…` is the 2026-05-15 one) and `canonical-2026-05-21/val/cid22.parquet`
/// (the canonical set per CLAUDE.md, sha `6eea0825…`). Measured 2026-07-15:
/// same 4,292 rows in the same order, `human_score` and all 372 features
/// byte-identical — the canonical one merely adds 21 target columns. So no
/// past number is wrong. But the report could not TELL you that, and "the
/// numbers happen to agree" is a fact someone had to go measure rather than
/// read.
///
/// Hashing costs ~10 ms per corpus against a ~3.5 s run. There is no reason
/// for a number in this program's output not to name its inputs.
fn provenance_block(bake: &Path, corpora: &[(String, PathBuf, String, u64)]) -> String {
    let mut s = String::new();

    let (bake_sha, bake_bytes) = match zensim_validate::train_manifest::sha256_file(bake) {
        Ok(sha) => (sha, std::fs::metadata(bake).map(|m| m.len()).unwrap_or(0)),
        Err(e) => (format!("UNHASHABLE ({e})"), 0),
    };

    // Code version. `git describe`-free: the commit + dirty flag is what a
    // reproduction needs. A dirty tree is reported LOUDLY rather than
    // silently — a number from uncommitted code is not reproducible by
    // anyone else, and that is exactly the thing a paper must not hide.
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);

    s.push_str("## Provenance\n\n");
    s.push_str(
        "_Every number below is reproducible from this block: inputs are named by \
         sha256, not by path._\n\n",
    );
    s.push_str("| input | sha256 | size | note |\n|---|---|--:|---|\n");
    s.push_str(&format!(
        "| bake `{}` | `{}` | {} B | the scored model |\n",
        bake.file_name().unwrap_or_default().to_string_lossy(),
        &bake_sha[..16.min(bake_sha.len())],
        bake_bytes
    ));
    for (name, path, sha, bytes) in corpora {
        s.push_str(&format!(
            "| corpus **{}** | `{}` | {} B | `{}` |\n",
            name,
            &sha[..16.min(sha.len())],
            bytes,
            path.display()
        ));
    }
    s.push_str(&format!(
        "| code | `{commit}`{} | — | zensim @ git HEAD |\n",
        if dirty { " **+DIRTY**" } else { "" }
    ));
    s.push('\n');
    if dirty {
        s.push_str(
            "> ⚠ **The working tree was DIRTY when this ran.** These numbers came from code \
             that is not committed, so nobody else can reproduce them — including you, later. \
             Commit and re-run before citing anything here.\n\n",
        );
    }
    s
}

/// `--print-features-root`: the EVAL features root DERIVED FROM THE BAKE, on
/// stdout, with the rule that produced it on stderr.
///
/// The regime default is passed in as `args.features_root` — `parse_args`
/// already resolved it (and an explicit `--features-root` already overrode
/// it), so this mode reports the same value a scoring run would have used
/// whenever the bake itself names nothing better. Exit 3 when the root cannot
/// be determined: a caller must refuse rather than default silently.
fn print_features_root(args: &Args) -> ExitCode {
    let bytes = match std::fs::read(&args.bake) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "bake_verdict --print-features-root: read {}: {e}",
                args.bake.display()
            );
            return ExitCode::from(3);
        }
    };
    let model = match Model::from_bytes(&bytes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "bake_verdict --print-features-root: parse {}: {e:?}",
                args.bake.display()
            );
            return ExitCode::from(3);
        }
    };
    // `explicit` is deliberately None here: in this mode `--features-root` is
    // the FALLBACK to report when the bake names nothing better, not an
    // override. That matters for `--regime 924`, where the wrapper supplies
    // the ext924 root as a script default — treating it as an override would
    // make a 924-regime bake trained at some other registered root
    // un-rerootable. A genuine caller override belongs to the caller
    // (run_full_eval.sh's 4th argument / $ZENSIM_FEATURES_ROOT), which simply
    // does not run this probe.
    match zensim_validate::feature_set::resolve_features_root(&model, &args.features_root, None) {
        Ok(r) => {
            eprintln!(
                "bake_verdict --print-features-root: {} :: era {} :: {}",
                r.source.describe(),
                era_of(&r.root),
                r.root.display()
            );
            println!("{}", r.root.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!(
                "bake_verdict --print-features-root: REFUSING — {e}\n  \
                 (never silently defaults: an undeterminable root is an error, \
                 not the regime default)"
            );
            ExitCode::from(3)
        }
    }
}

fn main() -> ExitCode {
    let t0 = Instant::now();
    // Phase timing, printed only under ZENSIM_PERF_TRACE=1 (see
    // `zensim_validate::perf_trace`). Off by default and output-neutral.
    let pt = zensim_validate::perf_trace::PerfTrace::new("bake_verdict");
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("bake_verdict: {e}");
            print_usage();
            return ExitCode::from(2);
        }
    };
    // `--print-features-root`: resolve the root FROM THE BAKE and exit. Runs
    // BEFORE the era note below on purpose — that note describes the root the
    // run will SCORE on, and this mode scores nothing.
    if args.print_features_root {
        return print_features_root(&args);
    }
    // Every verdict states the ruler it was produced on (2026-08-30): the 372 default
    // moved to the current-extractor root, and a number read on one era cannot be
    // corrected into the other (the shift is model-specific — see zensim_validate::eval_roots).
    eprintln!(
        "bake_verdict: features-root era — {} :: {}",
        era_of(&args.features_root),
        args.features_root.display()
    );
    // Same discipline for the RULER as for the era: a corpus with two files on
    // disk under one slot must say which one produced the number (D2).
    if !args.regime_720 {
        for line in ruler_lines(&args.corpora, &args.features_root) {
            eprintln!("{line}");
        }
    }
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

    // Member list: `--ensemble` when given, else the single `--bake`. k=1 runs
    // the byte-for-byte original path (see [`Ensemble::score_rows`]).
    let members: Vec<PathBuf> = if args.ensemble.is_empty() {
        vec![args.bake.clone()]
    } else {
        args.ensemble.clone()
    };
    let mut models: Vec<Model> = Vec::with_capacity(members.len());
    for p in &members {
        let bytes = match std::fs::read(p) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("bake_verdict: failed to read bake {}: {e}", p.display());
                return ExitCode::from(1);
            }
        };
        match Model::from_bytes(&bytes) {
            Ok(m) => models.push(m),
            Err(e) => {
                eprintln!(
                    "bake_verdict: failed to parse ZNPR bake {}: {e:?}",
                    p.display()
                );
                return ExitCode::from(1);
            }
        }
    }
    // Members must agree on input width — averaging predictions from models
    // that read different feature regimes is the column-mixing failure mode
    // this repo bans outright, so it fails loud rather than truncating.
    let n_inputs = models[0].caller_input_width();
    if let Some((p, m)) = members
        .iter()
        .zip(models.iter())
        .find(|(_, m)| m.caller_input_width() != n_inputs)
    {
        eprintln!(
            "bake_verdict: ensemble member {} has n_inputs={} but member 0 has {} — \
             refusing to average across feature regimes",
            p.display(),
            m.n_inputs(),
            n_inputs
        );
        return ExitCode::from(2);
    }
    // C2 wrong-regime guard (appendix W): the `--regime 944` root is a FOLDED
    // extraction whose f156-371 block is structural zeros. A bake that
    // structurally uses that block would score to plausible-looking garbage
    // (ebothg_m504; appendix U.R0's shipped-B CID22 0.3862 vs true 0.8764) —
    // refuse, unless `--cross-regime` states the read is intentional.
    // R1b (2026-08-30): the guard's premise — "a `--regime 944` root feeds
    // f156-371 as structural zeros" — is TRUE of every folded root but FALSE of
    // the all-live pools roots (`folded720append2pools` / `…carriers`), which
    // are 944 wide with that block populated. Ask the ROOT what regime it is
    // instead of assuming, so the guard refuses exactly when the block really
    // is zeros. Unknown / unmarked roots keep the old (refusing) behaviour.
    let root_regime = root_declared_regime(&args.features_root);
    let root_block_live = root_regime
        .as_deref()
        .is_some_and(|r| r.ends_with("pools") || r.ends_with("carriers"));
    if root_block_live {
        eprintln!(
            "bake_verdict: features-root declares regime {} — f156-371 is LIVE at \
             this root, wrong-regime refusal not applicable",
            root_regime.as_deref().unwrap_or("?")
        );
    }
    if args.regime_944 && !args.cross_regime && !root_block_live {
        for (p, m) in members.iter().zip(models.iter()) {
            match zensim_validate::block_profile::folded_root_conflict(m) {
                Ok(None) => {}
                Ok(Some(why)) => {
                    eprintln!(
                        "bake_verdict: REFUSING wrong-regime read of {}: {why}.\n\
                         Score this bake at its native root (`--regime 372`), or pass \
                         `--cross-regime` if feeding zeros to that block is intentional.",
                        p.display()
                    );
                    return ExitCode::from(2);
                }
                Err(e) => {
                    eprintln!(
                        "bake_verdict: cannot block-profile {} for the wrong-regime \
                         guard (malformed transform table?): {e}",
                        p.display()
                    );
                    return ExitCode::from(2);
                }
            }
        }
    } else if args.regime_944 && args.cross_regime && !root_block_live {
        eprintln!("bake_verdict: --cross-regime set — wrong-regime refusal disabled by caller");
    }
    // FEATURE-SET IDS (docs/FEATURE_SET_IDS.md). The guard above is the same
    // check hand-specialised to ONE block (f156-371) at ONE regime (944);
    // this generalises it to every block at every regime, and adds the era
    // axis the width can never carry. It REPORTS by default and REFUSES under
    // `--require-feature-set-match` — see the flag's doc for why the strict
    // mode is not yet the default.
    {
        use zensim_validate::feature_set;
        // **G5.1 — `--regime N` is a DERIVED, PRINTED alias, not a hidden
        // switch.** The flag selects defaults (root, grids, corpora, label);
        // what it MEANS is a slot set, and until now that meaning lived only
        // in the defaults it picked. Resolving it through the registry and
        // printing it makes the alias auditable — and makes a mismatch
        // between "what the flag says" and "what the root is" visible in the
        // same three lines as everything else.
        let regime_name = if args.regime_944 {
            "944"
        } else if args.regime_720 {
            "720"
        } else {
            "372"
        };
        // The flag's meaning is DERIVED, not asserted: it selects a default
        // root, that root's `_MANIFEST.json` declares an extraction regime,
        // and the registry says which slot set that regime is. Reading the
        // chain rather than hard-coding `372 -> v1-372` keeps the printed
        // claim tied to the artifact on disk — if the default root's manifest
        // changes, so does this line.
        let default_root: &Path = match regime_name {
            "944" => Path::new(DEFAULT_FEATURES_ROOT_944),
            "720" => Path::new(DEFAULT_FEATURES_ROOT_720),
            _ => Path::new(zensim_validate::eval_roots::DEFAULT_FEATURES_ROOT_372),
        };
        let declared = root_declared_regime(default_root);
        // A root manifest's `"regime"` is by convention `<key> (human prose)`
        // — the 372 root declares `v1-372 (extended+iw, num_scales=4, ...)`
        // — so the registry key is the leading token. Both forms are tried
        // rather than assuming, and a root with no manifest `regime` at all
        // (the 944 campaign root today) reports NOT ESTABLISHED rather than
        // being guessed at from its path.
        let key = declared
            .as_deref()
            .map(|r| r.split([' ', '(']).next().unwrap_or(r).trim().to_string());
        match key
            .as_deref()
            .and_then(|k| feature_set::registry().regime(k).map(|x| (k, x)))
            .or_else(|| {
                declared
                    .as_deref()
                    .and_then(|r| feature_set::registry().regime(r).map(|x| (r, x)))
            }) {
            Some((r, (parts, width, slots))) => eprintln!(
                "bake_verdict: --regime {regime_name} means regime {r} = {parts}@w{width} \
                 ({} slots, #{}) — derived from {}'s manifest, not hard-coded",
                slots.len(),
                zensim::feature_set_id::hex8(slots.hash8()),
                default_root.display()
            ),
            None => eprintln!(
                "bake_verdict: --regime {regime_name} — its default root declares regime {} and \
                 the registry has no entry for it, so the flag's meaning in COLUMNS is NOT \
                 established (never read that as a match)",
                declared.as_deref().unwrap_or("<none>")
            ),
        }
        let root_ref = feature_set::root_feature_set_ref(&args.features_root);
        match &root_ref {
            Some(r) => eprintln!(
                "bake_verdict: feature-set — table {} [{}{}]",
                r.id,
                r.source,
                if r.inferred { ", INFERRED" } else { "" }
            ),
            None => eprintln!(
                "bake_verdict: feature-set — table id NOT ESTABLISHED for {} \
                 (no _MANIFEST.json feature_set_id, and the path/regime are not in \
                 benchmarks/feature_sets_registry.json) — treat every id comparison below \
                 as unavailable, never as a pass",
                args.features_root.display()
            ),
        }
        let mut refuse = false;
        for (p, m) in members.iter().zip(models.iter()) {
            let era = feature_set::bake_declared_training_set(m)
                .map(|id| id.era().to_string())
                .unwrap_or_else(|| zensim::feature_set_id::ERA_UNKNOWN.to_string());
            match feature_set::bake_feature_set_ref(m, &era) {
                Err(e) => {
                    eprintln!(
                        "bake_verdict: feature-set — cannot derive id for {}: {e}",
                        p.display()
                    );
                    refuse |= args.require_feature_set_match;
                }
                Ok(b) => {
                    eprintln!(
                        "bake_verdict: feature-set — bake  {} [{}]",
                        b.id,
                        p.display()
                    );
                    // **PER-BAKE REVISION.** A bake declares the formula
                    // revision it was trained against
                    // (`zentrain.formula_revision`; absent = the shipped
                    // one), and this build computes one revision per walk. A
                    // bake asking for a revision this build is not computing
                    // would be scored on arithmetic it never saw — measured
                    // as 22 of 33 inputs per shipped 944 bake for the F5 fix
                    // — so it is REFUSED rather than reported.
                    let bake_rev = zensim::feature_v2::bake_formula_revision_public(m);
                    let build_rev = zensim::feature_v2::active_formula_revision();
                    if bake_rev != build_rev {
                        eprintln!(
                            "bake_verdict: REFUSING — {} declares formula revision {bake_rev:?} \
                             and this build computes {build_rev:?}. The arithmetic differs on \
                             the slots that revision moves, so every number would be about a \
                             formula the bake was not trained on. Re-run with \
                             ZENSIM_FORMULA_REV set to match, or score a bake of this build's \
                             revision.",
                            p.display()
                        );
                        return ExitCode::from(2);
                    }
                    if let Some(r) = &root_ref {
                        for mm in feature_set::check(&b, r) {
                            let loud = mm.kind != feature_set::MismatchKind::EraUnknown;
                            // **G5.2 — the `--regime 944` silent-mis-scoring
                            // bug, closed at the root rather than at one
                            // block.** `SlotsNotPopulated` is not a
                            // disagreement about NAMES; it is the statement
                            // that the bake reads columns this table does not
                            // contain, so every number below it would be
                            // computed from structural fill. That is the
                            // recorded instance (shipped B, CID22 **0.3862**
                            // against its true **0.8764**), and it is not a
                            // judgement call.
                            //
                            // It refuses by DEFAULT — but only when BOTH ids
                            // are stored rather than inferred. An INFERRED id
                            // is "evidence about the artifact's NAME, never
                            // about its BYTES" (`FEATURE_SET_IDS.md` §2.3), so
                            // refusing on one would be refusing on a guess,
                            // and most roots on disk today are inferred. When
                            // either side is inferred this stays a report, and
                            // `--require-feature-set-match` still escalates it
                            // as before.
                            //
                            // The pre-existing `folded_root_conflict` guard
                            // above is this same check hand-specialised to ONE
                            // block at ONE regime; it is kept because it fires
                            // on INFERRED roots too, where this one will not.
                            let unpopulated =
                                mm.kind == feature_set::MismatchKind::SlotsNotPopulated;
                            let both_stored = !b.inferred && !r.inferred;
                            eprintln!(
                                "bake_verdict: feature-set {} — {mm}",
                                if loud { "MISMATCH" } else { "note" }
                            );
                            if unpopulated && both_stored && !args.allow_unpopulated_slots {
                                eprintln!(
                                    "bake_verdict: REFUSING — the bake reads slots this table \
                                     does not populate, and BOTH ids are stored (not inferred), \
                                     so this is a measured fact rather than a guess about names. \
                                     Every number from this pairing would be computed from \
                                     structural fill. Score at the matching root, or pass \
                                     --allow-unpopulated-slots to state that feeding fill to \
                                     those columns is intentional."
                                );
                                return ExitCode::from(2);
                            }
                            refuse |= args.require_feature_set_match;
                        }
                    } else {
                        refuse |= args.require_feature_set_match;
                    }
                }
            }
        }
        if refuse {
            eprintln!(
                "bake_verdict: REFUSING — --require-feature-set-match is set and the bake's \
                 feature-set id does not match the table's. Score at the matching root, or drop \
                 the flag to get the report without the gate."
            );
            return ExitCode::from(2);
        }
    }
    let ens = Ensemble {
        has_transforms: models
            .iter()
            .map(|m| m.has_nontrivial_feature_transforms())
            .collect(),
        n_inputs,
        models,
        weights: args.ensemble_weights.clone(),
    };
    if let Some(w) = &args.ensemble_weights {
        eprintln!(
            "bake_verdict: WEIGHTED ensemble — {}",
            w.iter()
                .zip(args.ensemble.iter())
                .map(|(wi, p)| {
                    format!(
                        "{:.4}*{}",
                        wi,
                        p.file_name().and_then(|s| s.to_str()).unwrap_or("member")
                    )
                })
                .collect::<Vec<_>>()
                .join(" + ")
        );
    }
    let model = ens.primary();
    let has_transforms = ens.has_transforms[0];
    let has_per_sample_alpha = extract_per_sample_alpha_head(model).is_some();
    let has_hybrid_head = extract_hybrid_head(model).is_some();
    eprintln!(
        "bake: n_inputs={n_inputs}{}  feature_transforms={}  per_sample_alpha_head={}  hybrid_head={}",
        // A dead-column-PRUNED bake forwards fewer layer-0 rows than the
        // feature width it accepts; surface both so the reader can see it.
        if model.n_inputs() != n_inputs {
            format!(" (PRUNED: layer0_in_dim={})", model.n_inputs())
        } else {
            String::new()
        },
        if has_transforms { "yes" } else { "no" },
        if has_per_sample_alpha { "yes" } else { "no" },
        if has_hybrid_head { "yes" } else { "no" }
    );
    if ens.len() > 1 {
        eprintln!(
            "ENSEMBLE: {} members, equal weight, mean of raw predictions:\n  {}",
            ens.len(),
            members
                .iter()
                .map(|p| p
                    .file_name()
                    .map(|f| f.to_string_lossy().to_string())
                    .unwrap_or_else(|| p.display().to_string()))
                .collect::<Vec<_>>()
                .join("\n  ")
        );
    }

    // PRE-PASS: resolve, existence-check, and HASH every corpus before any
    // scoring. Two reasons it runs first rather than inside the scoring loop:
    // a missing corpus fails before we spend a single row of work, and the
    // provenance header below can name every input by content hash.
    //
    // NO GRACEFUL SKIP (CLAUDE.md): a missing corpus FAILS LOUD with an R2 fetch hint,
    // never silently drops an axis (a run that skips the non-photo corpus would hide the
    // very content-blindness this eval exists to catch). All eval corpora are mirrored to
    // s3://zentrain/eval-corpora/ — the error tells the caller how to restore them.
    //
    // Hashing is `rayon`-parallel across corpora and results are re-assembled
    // in the ORIGINAL corpus order, so the provenance block is byte-identical
    // to the sequential pass; only the wall clock changes (each hash re-reads
    // the whole parquet off the slow 9p `/mnt/v` mount, and 12 of those
    // serialized was 1.3 s of pure IO latency).
    zensim_validate::parallel::init();
    let hashed: Vec<Result<Option<CorpusProv>, String>> = args
        .corpora
        .par_iter()
        .map(|corpus| {
            // Regime-aware slot: --regime 720 loads the 720-wide ext_*.parquet
            // (via slot_720), NOT the 372col filename. The 372 arm resolves
            // through the shared `resolve_372_slot`, which render_corpus also
            // calls — so the two can no longer drift.
            let cfname = if args.regime_720 {
                match slot_720_file(corpus.name, &args.features_root) {
                    Some(f) => f,
                    None => return Ok(None), // filtered already; belt-and-braces
                }
            } else {
                resolve_372_slot(corpus, &args.features_root).file
            };
            let cpath = args.features_root.join(&cfname);
            if !cpath.exists() {
                return Err(format!(
                    "bake_verdict: MISSING corpus {} at {} — do NOT skip; restore it:\n  \
                     aws s3 cp s3://zentrain/eval-corpora/{} {} --endpoint-url \
                     https://338ad3b06716695d6e2c81c864e387d8.r2.cloudflarestorage.com",
                    corpus.display,
                    cpath.display(),
                    Path::new(&cfname).file_name().unwrap().to_string_lossy(),
                    cpath.display(),
                ));
            }
            match zensim_validate::train_manifest::sha256_file(&cpath) {
                Ok(sha) => {
                    let bytes = std::fs::metadata(&cpath).map(|m| m.len()).unwrap_or(0);
                    Ok(Some((corpus.display.to_string(), cpath, sha, bytes)))
                }
                Err(e) => Err(format!(
                    "bake_verdict: cannot hash corpus {}: {e}",
                    cpath.display()
                )),
            }
        })
        .collect();
    let mut corpus_prov: Vec<(String, PathBuf, String, u64)> = Vec::new();
    for r in hashed {
        match r {
            Ok(Some(entry)) => corpus_prov.push(entry),
            Ok(None) => {}
            Err(msg) => {
                eprintln!("{msg}");
                return ExitCode::from(2);
            }
        }
    }

    // The DIAL grid is an INPUT to half this report's numbers, so it belongs in
    // the provenance block exactly as much as a corpus does. It was omitted
    // until 2026-07-15 — and the dial grid is the input where it mattered MOST:
    // three versions exist, they genuinely differ, and the default was the one
    // with two known defects. A verdict could not say which it used.
    if args.dial_grid.exists()
        && let Ok(sha) = zensim_validate::train_manifest::sha256_file(&args.dial_grid)
    {
        let bytes = std::fs::metadata(&args.dial_grid)
            .map(|m| m.len())
            .unwrap_or(0);
        let label = if sha == CANONICAL_DIAL_GRID_SHA256 {
            "dial grid (canonical)".to_string()
        } else {
            eprintln!(
                "bake_verdict: WARNING — dial grid {} (sha {}) is NOT the canonical grid \
                     (sha {}).\n  The canonical grid is {}\n  The original 2026-05-29 grid carries \
                     TWO known defects: 9/115 ladders of w11 extraction garbage, and 33 pre-fix \
                     JXL cells at d=0.025. Per-ladder dial numbers on those are garbage-input \
                     scoring. See benchmarks/eval_grids_2026-05-29.pointer.md.",
                args.dial_grid.display(),
                &sha[..16],
                &CANONICAL_DIAL_GRID_SHA256[..16],
                CANONICAL_DIAL_GRID,
            );
            "dial grid ⚠ **NOT CANONICAL**".to_string()
        };
        corpus_prov.push((label, args.dial_grid.clone(), sha, bytes));
    }

    pt.mark("corpus pre-pass (existence + sha256 of every input)");

    let mut buf = String::new();
    buf.push_str("# bake_verdict — instant V_X eval\n\n");
    buf.push_str(&provenance_block(&args.bake, &corpus_prov));
    buf.push_str(&format!("- Bake n_inputs: {n_inputs}\n"));
    buf.push_str(&format!(
        "- Feature transforms: {}\n",
        if has_transforms {
            "yes (uses predict_transformed)"
        } else {
            "no"
        }
    ));

    // ── Overlap the `--full-json` KADIS per-pair read with the corpora ──
    // It is the single most expensive IO in the run (4.5 s of a 40 s
    // baseline: a bounded window out of a 2.5 GB parquet, over the 64 KB-msize
    // 9p `/mnt/v` mount) and it depends on NOTHING but the args, so it starts
    // now and is joined at its use site inside the `--full-json` block. Same
    // file, same `read_cap`, same rows, same order — pure scheduling.
    let perpair_job: Option<
        std::thread::JoinHandle<Result<parquet_loader::PerPairSample, String>>,
    > = if (args.full_json.is_some() || args.fulleval.is_some()) && args.perpair_metrics.exists() {
        let path = args.perpair_metrics.clone();
        let read_cap = args
            .perpair_cap
            .saturating_mul(8)
            .clamp(args.perpair_cap, 40_000);
        Some(std::thread::spawn(move || {
            let cols: Vec<&str> = PERPAIR_METRIC_COLS.iter().map(|(c, _)| *c).collect();
            parquet_loader::load_perpair_sample(&path, &cols, read_cap)
        }))
    } else {
        None
    };

    // Corpora are INDEPENDENT: each loads its own parquet, scores it, and
    // runs its own panel — nothing crosses between them until the summary
    // table below. Running them on rayon and re-assembling in the original
    // `args.corpora` order therefore changes wall time only (12 serialized
    // corpora were 31.2 s of a 40.0 s run; the two 6 s corpora dominate the
    // parallel form). Every per-corpus number is computed by exactly the
    // same code on exactly the same rows.
    zensim_validate::parallel::init();
    let rendered: Vec<Result<CorpusResult, String>> = args
        .corpora
        .par_iter()
        .map(|corpus| render_corpus(corpus, &args.features_root, args.regime_720, &ens))
        .collect();
    let mut results: Vec<CorpusResult> = Vec::with_capacity(rendered.len());
    for r in rendered {
        match r {
            Ok(r) => results.push(r),
            Err(e) => {
                eprintln!("bake_verdict: {e}");
                return ExitCode::from(1);
            }
        }
    }
    pt.mark("all corpora (load+score+stats, parallel)");

    // Per-pair dump. Hoisted out of `render_corpus` when the corpus loop went
    // parallel: it is only meaningful with a single `--corpora` selection, and
    // the previous "every corpus writes, last one wins" ordering would have
    // become nondeterministic. Writing the LAST corpus's rows here preserves
    // the old semantics exactly, and deterministically.
    if let Some(path) = args.per_pair_output.as_deref()
        && let Some(last) = results.last()
    {
        let refs = if args.per_pair_refs {
            match last.ref_ids.as_deref() {
                Some(r) => Some(r),
                None => {
                    eprintln!(
                        "bake_verdict: --per-pair-refs but corpus '{}' carries no ref identity",
                        last.name
                    );
                    return ExitCode::from(1);
                }
            }
        } else {
            None
        };
        let s = format_per_pair(&last.humans, &last.rescaled_scores, refs);
        if let Err(e) = std::fs::write(path, s) {
            eprintln!("bake_verdict: write per-pair output: {e}");
            return ExitCode::from(1);
        }
        eprintln!("  wrote per-pair predictions to {}", path.display());
    }

    // One-row summary across all corpora at the top.
    buf.push_str("\n## Summary (one row per corpus)\n\n");
    buf.push_str(
        "| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC | geomean3 | per-ref | %bwd |\n",
    );
    buf.push_str("|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for r in &results {
        let g3 = (r.srocc * r.plcc * r.pwrc).cbrt();
        let (pr, bwd) = match &r.per_ref {
            Some(p) => (
                format!("{:+.4}", p.mean),
                format!("{:.0}%", p.frac_negative * 100.0),
            ),
            None => ("—".to_string(), "—".to_string()),
        };
        // Mark train==val corpora (KADID/TID): their SROCC rewards memorization,
        // not held-out skill, so it must not be read as a generalization number.
        let mut disp = if train_eq_val(r.name) {
            format!("{} ⚠t=v", r.display)
        } else {
            r.display.to_string()
        };
        if sign_is_meaningful(r.name) && r.srocc_signed < 0.0 {
            disp.push_str(" ⛔INV");
        }
        buf.push_str(&format!(
            "| {} | {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.3} | {:.4} | {:.4} | {} | {} |\n",
            disp,
            r.n,
            srocc_cell(r.name, r.srocc, r.srocc_signed),
            r.plcc,
            r.krocc,
            r.or_ratio,
            r.pwrc,
            r.z_rmse,
            r.ds_auc,
            g3,
            pr,
            bwd
        ));
    }
    buf.push('\n');
    buf.push_str(
        "_`per-ref` is the mean within-image SROCC and `%bwd` the share of references \
whose distortion ladder is ranked BACKWARDS. Read them against the pooled SROCC: a wide \
gap means the pooled number is carried by cross-image scale rather than ranking (the \
AIC-3 0.79-pooled / 0.93-per-ref confound). A high `%bwd` next to a healthy SROCC is the \
failure §8.39 found and no pooled or per-band stat can see. `—` = corpus carries no ref \
identity. The SROCC column is **SIGNED** on every quality-oriented corpus and \
**⛔INVERTED** marks a bake that is ANTI-CORRELATED with that corpus's human labels — a \
backwards ranker, never a high scorer (`konjnd` alone prints |SROCC|, whose sign is \
structurally negative on at-PJND pairs). **⚠t=v** marks KADID/TID, whose 100% train==val pair-overlap makes their SROCC \
a memorization number — not held-out generalization; do not rank a bake by them._\n",
    );
    // Per-corpus SROCC at a glance (inline-SVG; renders in the HTML report).
    if !results.is_empty() {
        let labels: Vec<String> = results.iter().map(|r| r.display.to_string()).collect();
        let sroccs: Vec<f64> = results
            .iter()
            .map(|r| {
                100.0
                    * if sign_is_meaningful(r.name) {
                        r.srocc_signed
                    } else {
                        r.srocc
                    }
            })
            .collect();
        buf.push('\n');
        buf.push_str(&eval_report::svg_bars(
            "Per-corpus SIGNED SROCC ×100 (rank agreement with human MOS; negative = INVERTED)",
            &labels,
            &sroccs,
            0.0,
            100.0,
            85.0,
            true,
        ));
        buf.push('\n');
    }
    // ── CODEC_TARGET_GOALS.md scorecard ──────────────────────────────
    // Measurable from held-out corpus scores alone. Goals needing
    // external q-sweep / cross-codec data (G3, G4, G10) are flagged.
    // `gates_json` mirrors the scorecard into `--full-json` (same values,
    // never re-derived downstream).
    let gates_json: Option<serde_json::Value>;
    {
        let find = |name: &str| results.iter().find(|r| r.display.contains(name));
        let cid22 = find("CID22");
        let konjnd = find("KonJND");
        let aic3 = find("AIC-3");
        let nonphoto = find("non-photo");
        let imazen26 = find("real-codec");

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

        // G-NP: non-photo content rank (added 2026-07-15). Held-out imazen-26 diverse
        // content vs ssim2. Floor 0.85, target 0.93 — a content-BLIND bake (photographic-
        // only training) craters to ~0.856 here while photographic corpora stay high, so
        // this is the standing detector for "non-photo content crashes" (§8.34/§8.35).
        // Only scored when the nonphoto corpus is in the run (default set includes it).
        let gnp = nonphoto.map(|r| soft_gate(r.srocc, 0.85, 0.93));
        // G-IM26: ssim2-agreement on real-codec output (added 2026-07-16, user
        // "make imazen-26 a first-class gate"). The broad sibling of G-NP over all
        // content + all 4 real lossy codecs. B (excluded from bigcodec training)
        // scores ~0.84 here; an MLP that absorbed bigcodec reaches ~0.95 — the
        // second axis to G7's CID22. Floor 0.85, target 0.95.
        let gim26 = imazen26.map(|r| soft_gate(r.srocc, 0.85, 0.95));

        // Canonical product-weighted ranking composite (single source: the
        // `product_composite` helper; the --full-json writer + the dashboard read
        // the SAME formula). CID22 + imazen26 + non-photo centered, KADID/TID out.
        let product_comp = product_composite(&results);

        // ── G-OR: OR is a CATASTROPHE FLOOR, not a ranker ───────────────────
        // Measured across 15 bakes (stats review §2a): OR std ≤0.018 on every
        // corpus, ≈0 for all real models — it only lifts on a broken bake. So it
        // earns a pass/fail gate on the WORST-corpus OR, never a ranking column.
        // Per-sample σ is unavailable in the parquets, so OR runs the lenient
        // corpus-σ fallback; the 0.10 fail point reflects that leniency.
        let max_or = results
            .iter()
            .map(|r| r.or_ratio)
            .filter(|x| x.is_finite())
            .fold(0.0_f64, f64::max);
        let g_or = soft_gate(1.0 - max_or, 0.90, 0.98); // pass when worst OR ≲ 0.10

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
        if let Some(r) = nonphoto {
            // Crash vs weak: a diverse-trained bake ranks non-photo content ≈0.93; a
            // photographic-only bake degrades to ≈0.86 (content-weak); a truly broken bake
            // (garbage IW features, unclipped outliers) collapses toward 0 (crash).
            let flag = if r.srocc < 0.50 {
                " ⚠⚠ NON-PHOTO CRASH (garbage ranking)"
            } else if r.srocc < 0.88 {
                " ⚠ content-weak (< diverse-trained ~0.93)"
            } else {
                ""
            };
            buf.push_str(&format!(
                "| G-NP non-photo rank | imazen-26 SROCC ≥0.85 (target 0.93) | {:.4} | {:.2} |{flag}\n",
                r.srocc,
                gnp.unwrap_or(0.0),
            ));
        }
        if let Some(r) = imazen26 {
            // The ssim2-agreement axis on real codec output. B ≈ 0.84 (never
            // trained on bigcodec); bigcodec-absorbing MLPs ≈ 0.95.
            let flag = if r.srocc < 0.85 {
                " ⚠ weak ssim2-agreement on real-codec output"
            } else {
                ""
            };
            buf.push_str(&format!(
                "| G-IM26 ssim2-agree | imazen-26 real-codec SROCC ≥0.85 (target 0.95) | {:.4} | {:.2} |{flag}\n",
                r.srocc,
                gim26.unwrap_or(0.0),
            ));
        }
        buf.push_str(&format!(
            "| G-OR catastrophe floor | worst-corpus OR ≤0.10 (floor, NOT a ranker) | {max_or:.4} | {g_or:.2} |\n"
        ));
        // Weighted GATE score = "is it shippable?" — dial + calibration + the
        // first-class real-codec/non-photo gates. gim26/gnp were computed-but-
        // EXCLUDED before the 2026-07-26 review (§4a); added at weight 1 each
        // (+ G-OR at 0.5) so the gate reflects every first-class axis. This is a
        // DIFFERENT question from `product_composite` (the "which ranks best?"
        // ranking number) — both are emitted to --full-json; neither is silently
        // authoritative.
        let g_np_v = gnp.unwrap_or(0.0);
        let g_im26_v = gim26.unwrap_or(0.0);
        let weighted = (3.0 * g1
            + 2.5 * g8
            + 1.5 * g5
            + 1.0 * g9
            + 1.0 * g_im26_v
            + 1.0 * g_np_v
            + 0.5 * g7
            + 0.5 * g_or)
            / (3.0 + 2.5 + 1.5 + 1.0 + 1.0 + 1.0 + 0.5 + 0.5);
        buf.push_str(&format!(
            "\n**Product composite (ranking, KADID/TID excluded): {product_comp:.4}**  \
— CID22·1.0 + imazen26·0.5 + non-photo·0.3 + KonJND·0.2 + AIC3·0.1 + AIC4·0.05, / Σweights.\n"
        ));
        buf.push_str(&format!(
            "\n**Weighted goal score (shippability gate): {weighted:.3}**\n\n"
        ));
        gates_json = Some(serde_json::json!({
            "g1_dynamic_range": g1, "g5_hf_rank": g5, "g7_cid22": g7,
            "g8_zrmse": g8, "g9_ds_auc": g9,
            "g_np_nonphoto": gnp, "g_im26_realcodec": gim26,
            "g_or_catastrophe": g_or, "max_or": max_or,
            "p5": p5, "p95": p95,
            "weighted_goal": weighted,
        }));
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
    // ── G-ADDR probes: score them through the SAME `Ensemble` forward every
    // other section uses, so the gate can never disagree with the panel about
    // what this bake emits. A probe whose feature width does not match the
    // bake is REFUSED loudly rather than silently dropped: a quietly absent
    // axis is exactly the failure this gate exists to prevent.
    //
    // PEER MODE IS ALL-OR-NOTHING per axis. When `--dial-peer-scores` supplies
    // the grid, a probe axis is scored from its OWN `--*-peer-scores` table or
    // it is NOT MEASURED — it is never quietly filled in from `--bake`. Mixing
    // them would report a peer's grid reach beside the bake's tail depth under
    // one headline, and the "no cell out-scores a perfect copy" count would
    // compare one scorer's grid against another's identity, which is not a
    // measurement of anything.
    let peer_grid_label: Option<&str> = args.dial_peer_scores.as_ref().map(|(l, _)| l.as_str());
    #[allow(clippy::type_complexity)]
    let probe_scores = |path: &PathBuf,
                        what: &str,
                        peer: Option<&(String, PathBuf)>|
     -> Option<(Vec<String>, Vec<f64>, String, Option<Vec<f64>>)> {
        if !path.exists() {
            eprintln!(
                "bake_verdict: {what} probe {} is absent — G-ADDR axis NOT MEASURED",
                path.display()
            );
            return None;
        }
        let g = match parquet_loader::load_labeled_grid(path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!(
                    "bake_verdict: {what} probe {} failed to load: {e} — G-ADDR axis NOT MEASURED",
                    path.display()
                );
                return None;
            }
        };
        let sha = zensim_validate::train_manifest::sha256_file(path).unwrap_or_default();
        match (peer, peer_grid_label) {
            // Peer scores supplied: the probe parquet contributes only its
            // pinned row labels; the values come from the peer's table, so the
            // feature width is irrelevant (a reference metric has no bake).
            (Some((label, ppath)), _) => match load_peer_probe_scores(ppath, &g.label) {
                Ok(v) => {
                    eprintln!(
                        "bake_verdict: {what} probe scored from PEER `{label}` ({})",
                        ppath.display()
                    );
                    Some((g.label, v, sha, g.truth_ssim2))
                }
                Err(e) => {
                    eprintln!(
                        "bake_verdict: {what} peer probe `{label}` failed: {e} — G-ADDR axis \
                         NOT MEASURED"
                    );
                    None
                }
            },
            // Grid came from a peer but this probe did not: refuse to mix.
            (None, Some(label)) => {
                eprintln!(
                    "bake_verdict: DIAL grid is peer `{label}` but no --{}-peer-scores was \
                     supplied — G-ADDR axis NOT MEASURED (refusing to score this probe with \
                     --bake while the grid describes `{label}`)",
                    if what.starts_with("neg") {
                        "negtail"
                    } else {
                        "identity"
                    }
                );
                None
            }
            (None, None) if g.n_features == n_inputs => {
                let scores = ens.score_rows(&g.feature_rows);
                Some((g.label, scores, sha, g.truth_ssim2))
            }
            (None, None) => {
                eprintln!(
                    "bake_verdict: {what} probe {} has {} feature columns; bake expects {} — \
                     G-ADDR axis NOT MEASURED",
                    path.display(),
                    g.n_features,
                    n_inputs
                );
                None
            }
        }
    };
    let addr_probes = AddrProbes {
        negtail: args.negtail_probe.as_ref().and_then(|p| {
            probe_scores(p, "negative-tail", args.negtail_peer_scores.as_ref()).map(
                |(_lbl, dial, sha, truth)| {
                    // The probe's OWN reference truth drives A8r's
                    // reachability guard. It comes from the probe PARQUET even
                    // in peer mode — the instrument's truth is a property of
                    // the instrument, not of whichever scorer is being graded
                    // — and `load_labeled_grid` reorders nothing, so the two
                    // vectors stay row-aligned.
                    if truth.is_none() {
                        eprintln!(
                            "bake_verdict: negative-tail probe {} carries no `ssim2_gpu` truth \
                             column — G-ADDR A8r NOT MEASURED (no unit-safe fallback)",
                            p.display()
                        );
                    }
                    (
                        zensim_validate::dial_addressability::NegTailMeasure::from_scores_and_truth(
                            &dial,
                            truth.as_deref(),
                        ),
                        sha,
                    )
                },
            )
        }),
        identity: args.identity_probe.as_ref().and_then(|p| {
            probe_scores(p, "identity", args.identity_peer_scores.as_ref())
                .map(|(lbl, dial, sha, _)| (lbl.into_iter().zip(dial).collect::<Vec<_>>(), sha))
        }),
    };
    let dial_grid_sha =
        zensim_validate::train_manifest::sha256_file(&args.dial_grid).unwrap_or_default();
    let (dial_md, dial_metrics) = dial_panel(
        &ens,
        &args.dial_grid,
        args.dial_peer_scores.as_ref(),
        &dial_grid_sha,
        &addr_probes,
        args.gaddr_reference
            .as_deref()
            .unwrap_or(zensim_validate::dial_addressability::ACTIVE_REFERENCE),
        args.gaddr_tail_pins
            .as_deref()
            .map(|v| {
                zensim_validate::dial_addressability::TailPins::parse(v)
                    .expect("validated at argument-parse time")
            })
            .unwrap_or_default(),
        args.gaddr_grid_truth.as_ref(),
        args.floor_rule
            .as_deref()
            .map(|v| {
                zensim_validate::dial_addressability::FloorRule::parse(
                    v,
                    args.floor_margin.unwrap_or(
                        zensim_validate::dial_addressability::FloorRule::RESOLVABLE_MARGIN_DEFAULT,
                    ),
                )
                .expect("validated at argument-parse time")
            })
            // No `--floor-rule` ⇒ the OPERATIVE rule, owned by the registry's
            // active pin set (`resolvable`, margin 0.5, since the 2026-09-05
            // ruling) — NOT a hardcoded `Distinct`.
            .unwrap_or_else(zensim_validate::dial_addressability::operative_floor_rule),
        args.gaddr_value_pins
            .as_deref()
            .map(|v| {
                zensim_validate::dial_addressability::ValuePins::parse(v)
                    .expect("validated at argument-parse time")
            })
            .unwrap_or_default(),
        args.inversion_truth,
        args.reference_truth.as_ref(),
        args.encoder_inversion_census.as_ref(),
    );
    buf.push_str(&dial_md);
    pt.mark("DIAL panel (grid load + score + mono/tied)");

    if let Some(path) = args.gaddr_json.as_ref() {
        match dial_metrics.addr.as_ref() {
            Some(v) => {
                let mut j = zensim_validate::dial_addressability::to_json(v);
                // Stamp WHOSE dial this describes. In peer mode that is the
                // peer, not `--bake`, and a file that does not say so is a
                // mislabelled measurement waiting to be quoted.
                if let Some(obj) = j.as_object_mut() {
                    // `to_json` already carries `"floor_rule"` (the tag); add
                    // the numeric params actually used, so a `resolvable`/
                    // `spaced` bar is fully reproducible from the JSON alone
                    // without re-deriving them from the CLI invocation.
                    obj.insert(
                        "floor_rule_params".into(),
                        serde_json::json!({
                            "margin": args.floor_margin.unwrap_or(
                                zensim_validate::dial_addressability::FloorRule::RESOLVABLE_MARGIN_DEFAULT,
                            ),
                            "near_lo": zensim_validate::dial_addressability::FloorRule::SPACED_NEAR_LO_DEFAULT,
                            "near_hi": zensim_validate::dial_addressability::FloorRule::SPACED_NEAR_HI_DEFAULT,
                        }),
                    );
                    obj.insert(
                        "scorer".into(),
                        serde_json::json!(match args.dial_peer_scores.as_ref() {
                            Some((label, path)) => serde_json::json!({
                                "kind": "peer",
                                "label": label,
                                "dial_cells": path.display().to_string(),
                                "negtail_peer": args.negtail_peer_scores.as_ref()
                                    .map(|(_, p)| p.display().to_string()),
                                "identity_peer": args.identity_peer_scores.as_ref()
                                    .map(|(_, p)| p.display().to_string()),
                            }),
                            None => serde_json::json!({
                                "kind": "bake",
                                "label": args.bake.display().to_string(),
                            }),
                        }),
                    );
                }
                match std::fs::write(path, serde_json::to_string_pretty(&j).unwrap_or_default()) {
                    Ok(()) => eprintln!("wrote G-ADDR json to {}", path.display()),
                    Err(e) => eprintln!("--gaddr-json write failed ({}): {e}", path.display()),
                }
            }
            None => eprintln!(
                "--gaddr-json: the DIAL panel produced no G-ADDR verdict (grid missing or \
                 unreadable) — nothing written"
            ),
        }
    }

    let basename = |p: &Path| -> String {
        p.file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| p.display().to_string())
    };

    // ── Severity-ramp monotonicity (distortion dial) — opt-in ──────────
    if let Some(ramp_path) = &args.ramp_grid {
        match parquet_loader::load_ramp_grid(ramp_path) {
            Ok(grid) if grid.n_features == n_inputs => {
                let dial = ens.score_rows(&grid.feature_rows);
                let images: Vec<String> =
                    grid.image.iter().map(|p| basename(Path::new(p))).collect();
                let stats = eval_report::severity_ramp(&images, &grid.q, &dial, 0.5);
                buf.push_str(&eval_report::severity_ramp_section(
                    &stats,
                    &ramp_path.display().to_string(),
                ));
            }
            Ok(grid) => {
                buf.push_str(&format!(
                    "\n## Severity-ramp monotonicity — ⚠ SKIPPED (feature-count mismatch)\n\n\
                     Ramp grid `{}` has {} feature columns but the bake expects {}. The grid's \
                     feature regime must match the bake (PU21-u8 vs PU-linear). Point `--ramp-grid` \
                     at a regime-consistent parquet.\n",
                    ramp_path.display(),
                    grid.n_features,
                    n_inputs
                ));
            }
            Err(e) => {
                buf.push_str(&format!(
                    "\n## Severity-ramp monotonicity — ⚠ FAILED to load grid\n\n`{e}`\n"
                ));
            }
        }
    }

    // ── Per-zone dial agreement vs a reference bake (§8.20) — opt-in ────
    if let Some(cmp_path) = &args.compare {
        let ref_res = std::fs::read(cmp_path)
            .map_err(|e| format!("read reference bake {}: {e}", cmp_path.display()))
            .and_then(|b| {
                Model::from_bytes(&b).map_err(|e| format!("parse reference bake: {e:?}"))
            });
        match ref_res {
            Ok(ref_model) if args.dial_grid.exists() => {
                match parquet_loader::load_dial_grid(&args.dial_grid) {
                    Ok(grid) => {
                        let cand = ens.score_rows(&grid.feature_rows);
                        let ref_tf = ref_model.has_nontrivial_feature_transforms();
                        let ref_n = ref_model.caller_input_width();
                        let refs = score_grid_one(&ref_model, ref_tf, ref_n, &grid.feature_rows);
                        let zone = eval_report::zone_buckets(&cand, &refs, 5.0);
                        buf.push_str(&eval_report::zone_bucket_section(
                            &zone,
                            &basename(&args.bake),
                            &basename(cmp_path),
                            &args.dial_grid.display().to_string(),
                        ));
                    }
                    Err(e) => buf.push_str(&format!(
                        "\n## Per-zone dial agreement — ⚠ FAILED to load dial grid\n\n`{e}`\n"
                    )),
                }
            }
            Ok(_) => buf.push_str(&format!(
                "\n## Per-zone dial agreement — ⚠ SKIPPED (dial grid absent)\n\n\
                 The reference bake `{}` loaded, but the dial grid `{}` is missing — the \
                 per-zone comparison scores both bakes on the dial grid.\n",
                cmp_path.display(),
                args.dial_grid.display()
            )),
            Err(e) => buf.push_str(&format!(
                "\n## Per-zone dial agreement — ⚠ FAILED\n\n`{e}`\n"
            )),
        }
    }

    // ── Corruption gate (negative-tail ranking) — auto when grid present ──
    // Hoisted out of the block so `--full-json` can carry the gate result.
    let mut corruption_stats: Option<eval_report::CorruptionStats> = None;
    let mut corruption_head_stats: Option<eval_report::CorruptionStats> = None;
    let mut corruption_deploy_stats: Option<eval_report::CorruptionStats> = None;
    if args.corruption_grid.exists() {
        match parquet_loader::load_labeled_grid(&args.corruption_grid) {
            Ok(grid) => {
                // Hoisted so the DEPLOY composition below can reuse the dial
                // scores; `None` when the dial could not read this grid, which
                // is exactly when the composition is undefined.
                let mut dial_scores: Option<Vec<f64>> = None;
                if grid.n_features == n_inputs {
                    let dial = ens.score_rows(&grid.feature_rows);
                    let stats = eval_report::corruption_gate(&grid.label, &dial);
                    dial_scores = Some(dial);
                    buf.push_str(&eval_report::corruption_gate_section(
                        &stats,
                        &args.corruption_grid.display().to_string(),
                        "Corruption gate (negative-tail ranking)",
                    ));
                    corruption_stats = Some(stats);
                } else {
                    buf.push_str(&format!(
                        "\n## Corruption gate — ⚠ SKIPPED (feature-count mismatch)\n\n\
                         Grid `{}` has {} feature columns; bake expects {}.\n",
                        args.corruption_grid.display(),
                        grid.n_features,
                        n_inputs
                    ));
                }
                // Companion corruption head (`--corruption-head`): the shipping
                // design routes corruption to a separate head (the 924 dial's
                // own ordering is broken by design — distributional), so the
                // freeze gate reads the HEAD's numbers; the dial-alone section
                // above stays for honesty.
                if let Some(head_path) = &args.corruption_head {
                    match std::fs::read(head_path)
                        .map_err(|e| e.to_string())
                        .and_then(|b| Model::from_bytes(&b).map_err(|e| format!("{e:?}")))
                    {
                        Ok(head) if head.caller_input_width() == grid.n_features => {
                            let head_tf = head.has_nontrivial_feature_transforms();
                            let head_n = head.caller_input_width();
                            let scores = score_grid_one(&head, head_tf, head_n, &grid.feature_rows);
                            let stats = eval_report::corruption_gate(&grid.label, &scores);
                            buf.push_str(&eval_report::corruption_gate_section(
                                &stats,
                                &args.corruption_grid.display().to_string(),
                                &format!(
                                    "Corruption gate — companion head `{}` (the shipping owner)",
                                    basename(head_path)
                                ),
                            ));
                            corruption_head_stats = Some(stats);
                            // DEPLOY composition. The head is not a second
                            // ranker - the registered design is
                            // `final = min(perceptual, gate)`, so a row the
                            // head flags is forced to 0 and can no longer
                            // out-rank its own honest anchor. This is the
                            // number the product sees; the two sections above
                            // are each half of it.
                            let thr = args.corruption_head_threshold;
                            if let Some(dial) = dial_scores.as_ref() {
                                let composed: Vec<f64> = dial
                                    .iter()
                                    .zip(scores.iter())
                                    .map(|(&d, &h)| if h < thr { d.min(0.0) } else { d })
                                    .collect();
                                let cstats = eval_report::corruption_gate(&grid.label, &composed);
                                let title = format!(
                                    "Corruption gate - DEPLOY composition `min(dial, gate)` (head `{}`, deadband head_score < {thr})",
                                    basename(head_path)
                                );
                                buf.push_str(&eval_report::corruption_gate_section(
                                    &cstats,
                                    &args.corruption_grid.display().to_string(),
                                    &title,
                                ));
                                corruption_deploy_stats = Some(cstats);
                            } else {
                                buf.push_str(
                                    "\n## Corruption gate (deploy) - SKIPPED\n\nThe dial could not read this grid, so `min(dial, gate)` is undefined.\n",
                                );
                            }
                        }
                        Ok(head) => buf.push_str(&format!(
                            "\n## Corruption gate (head) — ⚠ SKIPPED (feature-count mismatch)\n\n\
                             Grid `{}` has {} feature columns; head `{}` expects {}.\n",
                            args.corruption_grid.display(),
                            grid.n_features,
                            head_path.display(),
                            head.caller_input_width()
                        )),
                        Err(e) => buf.push_str(&format!(
                            "\n## Corruption gate (head) — ⚠ FAILED to load `{}`\n\n`{e}`\n",
                            head_path.display()
                        )),
                    }
                }
            }
            Err(e) => buf.push_str(&format!(
                "\n## Corruption gate — ⚠ FAILED to load grid\n\n`{e}`\n"
            )),
        }
    } else if args.corruption_head.is_some() {
        buf.push_str(&format!(
            "\n## Corruption gate (head) — ⚠ SKIPPED (grid absent)\n\n\
             `--corruption-head` given but the corruption grid `{}` is missing.\n",
            args.corruption_grid.display()
        ));
    }

    for r in &results {
        buf.push_str(&r.body);
    }

    pt.mark("ramp/zone/corruption panels");

    // ── Related specialized evals (tracked-down historical set) ──────────
    // Everything the default eval does NOT run inline — because it needs a
    // second bake, a spline-internals gate, or an HDR/UPIQ regime corpus —
    // is listed here with the exact command, so this report points at the
    // full historical eval set rather than silently omitting it.
    buf.push_str("\n## Related specialized evals (run separately)\n\n");
    buf.push_str(
        "| eval | tool | when |\n|---|---|---|\n\
         | A-vs-B decisive (MRR + bootstrap CI) | `bake_compare --a <bake> --b <ref>` | comparing two bakes for a ship decision |\n\
         | G-RANGE tail gate (raw preds outside spline domain) | `bake_dial_refit gate --bake <bake> --corpus <parquet>` | a bake with an output spline (dial reach) |\n\
         | cross-codec JND consistency (stddev at JND targets) | `scripts/v_next/cross_codec_jnd_eval.py` | dial precision across codec families |\n\
         | UPIQ within-study / cross-domain seam (SDR↔HDR) | `scripts/hdr/upiq_panel.py`, `scripts/hdr/upiq_crossdomain_instrument.py` | an HDR bake / SDR-HDR alignment |\n\
         | sub-domain identity (R1: HDR-path ≡ SDR limit) | `scripts/hdr/ga_identity_report.py` | an HDR bake vs its SDR counterpart |\n\n\
         _These need a second bake, spline internals, or an HDR/UPIQ regime \
         corpus, so they stay separate tools; the rank + dial + ramp + zone + \
         corruption sections above are the always-applicable core._\n",
    );

    let elapsed = t0.elapsed();
    buf.push_str(&format!(
        "\n---\nWall time: {:.2}s ({} pair rows scored across {} corpora).\n",
        elapsed.as_secs_f64(),
        results.iter().map(|r| r.n).sum::<usize>(),
        results.len()
    ));

    // Machine-readable panel for the comparative dashboard — the structured
    // counterpart to the markdown, so consumers never parse the report. Bake
    // sha ties each row back to its manifest (the reproducibility spine).
    if let Some(json_path) = &args.json {
        #[derive(serde::Serialize)]
        struct CorpusJson {
            display: String,
            n: usize,
            srocc: f64,
            plcc: f64,
            krocc: f64,
            or_ratio: f64,
            pwrc: f64,
            z_rmse: f64,
            ds_auc: f64,
            per_ref_mean: Option<f64>,
            per_ref_frac_negative: Option<f64>,
        }
        #[derive(serde::Serialize)]
        struct DialJson {
            monotonicity: f64,
            p5: f64,
            p95: f64,
            flat: f64,
            /// G3 pass = monotonicity ≥ 0.93; G1 pass = p5 ≤ 25 ∧ p95 ≥ 85.
            g3_pass: bool,
            g1_pass: bool,
        }
        #[derive(serde::Serialize)]
        struct VerdictJson {
            bake: String,
            bake_sha256: String,
            n_inputs: usize,
            corpora: Vec<CorpusJson>,
            /// The codec-dial ship gate — a bake can win every rank corpus and
            /// still be a broken (non-monotonic, no-range) dial.
            dial: DialJson,
        }
        let bake_sha = zensim_validate::train_manifest::sha256_file(&args.bake).unwrap_or_default();
        let vj = VerdictJson {
            bake: args.bake.display().to_string(),
            bake_sha256: bake_sha,
            n_inputs,
            corpora: results
                .iter()
                .map(|r| CorpusJson {
                    display: r.display.to_string(),
                    n: r.n,
                    srocc: r.srocc,
                    plcc: r.plcc,
                    krocc: r.krocc,
                    or_ratio: r.or_ratio,
                    pwrc: r.pwrc,
                    z_rmse: r.z_rmse,
                    ds_auc: r.ds_auc,
                    per_ref_mean: r.per_ref.as_ref().map(|p| p.mean),
                    per_ref_frac_negative: r.per_ref.as_ref().map(|p| p.frac_negative),
                })
                .collect(),
            dial: DialJson {
                monotonicity: dial_metrics.mono,
                p5: dial_metrics.p5,
                p95: dial_metrics.p95,
                flat: dial_metrics.flat,
                g3_pass: dial_metrics.mono >= 0.93,
                g1_pass: dial_metrics.p5 <= 25.0 && dial_metrics.p95 >= 85.0,
            },
        };
        if let Some(parent) = json_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match serde_json::to_string_pretty(&vj) {
            Ok(s) => {
                if let Err(e) = std::fs::write(json_path, s) {
                    eprintln!("bake_verdict: failed to write {}: {e}", json_path.display());
                    return ExitCode::from(1);
                }
                eprintln!("wrote json panel to {}", json_path.display());
            }
            Err(e) => {
                eprintln!("bake_verdict: json serialize failed: {e}");
                return ExitCode::from(1);
            }
        }
    }

    // ── Unified "full-eval" JSON (scripts/run_full_eval.sh + dashboard) ──────
    // A superset of `--json`: rank as a map keyed by corpus, dial with
    // mono/tied/reach/dynamic_range, the corruption gate, and a sampled
    // multi-metric `per_pair` block for the scatter panels. `m3_coherence` is
    // left null — the wrapper computes it (diffmap_block_coherence) and injects
    // it, so this binary stays free of image I/O. `--fulleval` emits the same
    // content with ALL five M3 slots pre-nulled (schema-complete fulleval file).
    if args.full_json.is_some() || args.fulleval.is_some() {
        use serde_json::{Map, Value, json};
        // "model": the bake's own architecture + in/out modifiers, read from the
        // loaded ZNPR (zenpredict is the single parser — no byte-poking anywhere
        // else). Structured twin of `zenpredict inspect` for the dashboard's
        // per-model details card. Head/spline extraction is a cheap re-read.
        let model_block = {
            // `model` is member 0 (`Ensemble::primary`). In ensemble mode the
            // architecture fields therefore describe the ANCHOR member — an
            // ensemble has no single ZNPR to introspect — and the identity
            // fields below (`kind`/`members`/`member_names`/`anchor`) mark it
            // as an ensemble AT THE SOURCE, so the dashboard's Model-details
            // card cannot misattribute the anchor's architecture to the whole.
            // Schema matches scripts/promote_ensemble_fulleval.py + the
            // gauntlet's readers (`model.kind === 'ensemble'`, `model.members`,
            // `model.member_names`).
            let alpha = extract_per_sample_alpha_head(model).is_some();
            let hybrid = extract_hybrid_head(model).is_some();
            let minmax = extract_minmax_head(model).is_some();
            let tanh = extract_tanh_output_head_scale(model);
            let spline = zensim_validate::output_calibration_spline::extract(model);
            let layers: Vec<Value> = model
                .layers()
                .map(|l| {
                    let dtype = match l.weights {
                        zenpredict::WeightStorage::F32(_) => "f32",
                        zenpredict::WeightStorage::F16 { .. } => "f16",
                        zenpredict::WeightStorage::I8 { .. } => "i8",
                    };
                    json!({
                        "in": l.in_dim, "out": l.out_dim,
                        "activation": format!("{:?}", l.activation),
                        "dtype": dtype,
                    })
                })
                .collect();
            // Non-identity per-feature INPUT transforms (idx, kind, params).
            let transforms: Vec<Value> = match model.feature_transforms() {
                Some(ts) => {
                    let params = model.feature_transform_params();
                    ts.iter()
                        .enumerate()
                        .filter(|(_, t)| !matches!(t, zenpredict::FeatureTransform::Identity))
                        .map(|(i, t)| {
                            let p: Vec<f32> =
                                params.and_then(|ps| ps.get(i).cloned()).unwrap_or_default();
                            json!({"idx": i, "kind": format!("{t:?}"), "params": p})
                        })
                        .collect()
                }
                None => Vec::new(),
            };
            let mut mb = json!({
                "n_inputs": n_inputs,
                "n_outputs": model.n_outputs(),
                "n_layers": model.n_layers(),
                "znpr_version": model.version(),
                "file_bytes": std::fs::metadata(&args.bake).map(|m| m.len()).unwrap_or(0),
                "layers": layers,
                "scaler": {"present": !model.scaler_mean().is_empty(), "n": model.scaler_mean().len()},
                "feature_transforms": transforms,
                "n_feature_bounds": model.feature_bounds().len(),
                // OUTPUT modifier: the dial calibration spline (knots for plotting).
                "output_spline": spline.map(|sp| json!({
                    "n_knots": sp.xs.len(), "xs": sp.xs, "ys": sp.ys,
                })),
                "heads": {
                    "per_sample_alpha": alpha, "hybrid": hybrid, "minmax": minmax,
                    "tanh_pin_scale": tanh.and_then(nan_null),
                },
                "n_output_specs": model.output_specs().len(),
                "n_discrete_sets": model.discrete_sets().len(),
                "metadata_keys": model.metadata().iter().map(|e| e.key.to_string()).collect::<Vec<_>>(),
            });
            let o = mb.as_object_mut().expect("model block is an object");
            if args.ensemble.is_empty() {
                o.insert("kind".into(), json!("single"));
            } else {
                let stems: Vec<String> = args
                    .ensemble
                    .iter()
                    .map(|p| {
                        p.file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("member")
                            .to_string()
                    })
                    .collect();
                o.insert("kind".into(), json!("ensemble"));
                o.insert("members".into(), json!(args.ensemble.len()));
                o.insert("member_names".into(), json!(stems));
                o.insert("anchor".into(), json!(basename(&args.bake)));
                // The NORMALISED vector that actually scored; `null` = the
                // historical equal-weight mean. Published so a board cell can
                // never show a blend under a weight it was not scored at.
                o.insert(
                    "member_weights".into(),
                    match &args.ensemble_weights {
                        Some(w) => json!(w),
                        None => serde_json::Value::Null,
                    },
                );
            }
            mb
        };
        // Even-stride down to `cap` indices across [0,len) — keeps the scatter
        // spread across the corpus rather than truncating to a prefix.
        let stride = |len: usize, cap: usize| -> Vec<usize> {
            if cap == 0 || len <= cap {
                (0..len).collect()
            } else {
                (0..cap).map(|k| k * len / cap).collect()
            }
        };
        let bake_sha = zensim_validate::train_manifest::sha256_file(&args.bake).unwrap_or_default();
        let name = args.name.clone().unwrap_or_else(|| {
            args.bake
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("bake")
                .to_string()
        });
        let regime = if args.regime_944 {
            "944"
        } else if args.regime_720 {
            "720"
        } else {
            "372"
        };

        // rank: { "<corpus>": {n, srocc, plcc, krocc, or, pwrc, z_rmse} }
        let mut rank = Map::new();
        for r in &results {
            rank.insert(
                r.name.to_string(),
                json!({
                    "n": r.n,
                    "srocc": r.srocc,
                    // Signed (polarity-preserving) SROCC — a negative here means the
                    // bake is globally inverted, which `srocc` (abs) hides.
                    "srocc_signed": r.srocc_signed,
                    // Marginal bootstrap 95% CI of |SROCC| → the dashboard tie-band.
                    "srocc_ci": [r.srocc_ci.0, r.srocc_ci.1],
                    "plcc": r.plcc,
                    "krocc": r.krocc,
                    "or": r.or_ratio,
                    "pwrc": r.pwrc,
                    "z_rmse": r.z_rmse,
                    // Per-reference backwards-ladder share (null when the corpus
                    // carries no ref identity) — the one signal no pooled stat sees.
                    "frac_negative": r.per_ref.as_ref().map(|p| p.frac_negative),
                    "per_ref_mean": r.per_ref.as_ref().map(|p| p.mean),
                    "per_ref_n": r.per_ref.as_ref().map(|p| p.n_groups),
                    // KADID/TID are train==val (memorization) — flagged, not hidden.
                    "train_eq_val": train_eq_val(r.name),
                    // 10-band panel (null for JND/step-grid corpora). NaN → null
                    // via serde_json's f64 handling for bands with n < 4.
                    "bands": r.bands.as_ref().map(|bs| bs.iter().map(|b| json!({
                        "band": b.band,
                        // `lo` is −inf on the bottom band and `hi` +inf on the
                        // top; JSON has no infinity, so an open end serialises
                        // as null and MUST be read as "unbounded", never as a
                        // missing value.
                        "lo": nan_null(b.lo), "hi": nan_null(b.hi),
                        "n": b.n, "span": b.span,
                        "not_measured_reason": b.not_measured_reason,
                        "srocc": nan_null(b.srocc),
                        "srocc_signed": nan_null(b.srocc_signed),
                        "plcc": nan_null(b.plcc),
                        "krocc": nan_null(b.krocc), "or": nan_null(b.or_ratio),
                        "pwrc": nan_null(b.pwrc), "z_rmse": nan_null(b.z_rmse),
                        "mae": nan_null(b.mae),
                    })).collect::<Vec<_>>()),
                    // The scheme that cut those bands, so a reader never has to
                    // guess where the edges came from or why a band is absent.
                    // Edges are a function of the corpus target column alone —
                    // identical for every model — which is what makes the
                    // cross-bake band table comparable.
                    "band_scheme": r.bands.as_ref().map(|_| json!({
                        "name": "merged-decile-2026-08-06",
                        "base_bands": bands::BASE_BANDS,
                        "n_min": bands::N_MIN,
                        "span_min": bands::SPAN_MIN,
                        "doc": "campaign appendix V: fixed deciles accumulated \
                                into the finest partition whose every band has \
                                n >= n_min AND target span >= span_min",
                    })),
                }),
            );
        }

        // dial: mono/tied/reach/dynamic_range (+ raw p5/p95 for context) +
        // the per-codec breakout and aggregated per-codec curves for plotting.
        let dynamic_range = dial_metrics.p95 - dial_metrics.p5;
        // Hoisted out of the `dial` json! below: serde_json's json_internal
        // recurses once per key AND once per nesting level, so keeping this
        // deep sub-tree inline made adding a single key to `dial` blow the
        // macro recursion limit. Same value, one expansion shallower.
        let itr = &dial_metrics.inversion_truth;
        let dial_zones_json = dial_metrics.zones.as_ref().map(|z| json!({
                "scheme": "ladder-inversion-2026-08-31",
                // The grid this split was cut from. Recorded because the board
                // carries cells measured on SEVERAL dial grids (the 2026-05-29
                // 372 grid and its two quarantines, 720/924/944, the carriers
                // POOLS grid, the era-2 tiling variants) and a zone rate is
                // only comparable across cells cut from the same ladders.
                "grid": z.grid,
                "n_grid_rows": z.n_grid_rows,
                "zone_edges": ZONE_EDGES,
                "material_pt": MATERIAL_INV_PT,
                "min_rungs_per_ladder_zone": 3,
                "class_table": "benchmarks/dial_grid_content_classes_2026-08-31.tsv",
                "class_images": z.class_images,
                "worst_ladders": z.worst_ladders.iter().map(|(img, cd, cl, zn, dend, n, worst)| json!({
                    "image_id": img, "codec": cd, "class": cl, "zone": zn,
                    "end_delta": dend, "n_rungs": n, "worst_step": worst,
                })).collect::<Vec<_>>(),
                "cells": z.cells.iter().map(|(kind, key, zone, c, l)| {
                    let mut mags = c.inv_mags.clone();
                    mags.sort_by(f64::total_cmp);
                    json!({
                        "split": kind, "key": key, "zone": zone,
                        "n_pairs": c.n_pairs,
                        "forward": c.forward,
                        "inv_material": c.inv_material,
                        // 2026-09-05 ruling: material backwards rungs the two
                        // reference metrics AGREE are the encoder's, split out.
                        // `inv_material` above is DIAL-attributed only.
                        "inv_encoder": c.inv_encoder,
                        "inv_unknown": c.inv_unknown,
                        "inv_strict": c.inv_strict,
                        "flat": c.flat,
                        "codec_sat": c.codec_sat,
                        "subres": c.subres,
                        "inv_rate": if c.n_pairs > 0 {
                            Some(c.inv_material as f64 / c.n_pairs as f64) } else { None },
                        "flat_rate": if c.n_pairs > 0 {
                            Some(c.flat as f64 / c.n_pairs as f64) } else { None },
                        "inv_mag_med": if mags.is_empty() { None } else {
                            Some(mags[mags.len() / 2]) },
                        "inv_mag_max": mags.last().copied(),
                        "n_ladders": l.n,
                        "ladders_with_inv": l.with_inv,
                        "ladders_ends_backwards": l.endpoint_backwards,
                        "frac_ladders_with_inv": if l.n > 0 {
                            Some(l.with_inv as f64 / l.n as f64) } else { None },
                        "frac_ladders_ends_backwards": if l.n > 0 {
                            Some(l.endpoint_backwards as f64 / l.n as f64) } else { None },
                    })
                }).collect::<Vec<_>>(),
            }));
        let dial = json!({
            "mono_pct": dial_metrics.mono,
            "tied_pct": dial_metrics.flat,
            "reach": dial_metrics.reach,
            "dynamic_range": dynamic_range,
            "p5": dial_metrics.p5,
            "p95": dial_metrics.p95,
            "min": nan_null(dial_metrics.min),
            "max": nan_null(dial_metrics.max),
            // G-ADDR: floor + ceiling addressability against the SHIPPED
            // product dial on this same grid. `null` only when the dial panel
            // did not run; an unregistered grid still emits a block, with
            // every end-of-range axis `not_measured` (never a pass).
            "addressability": dial_metrics.addr.as_ref()
                .map(zensim_validate::dial_addressability::to_json),
            "per_codec": dial_metrics.per_codec.iter().map(|c| json!({
                "codec": c.codec, "n_curves": c.n_curves, "n_pairs": c.n_pairs,
                "mono": nan_null(c.mono), "tied": nan_null(c.tied),
            })).collect::<Vec<_>>(),
            // curves: {codec: [[q, p25, median, p75], ...]} sorted by q.
            "curves": dial_metrics.curves.iter().map(|c| (
                c.codec.clone(),
                c.pts.iter().map(|&(q, p25, med, p75)| vec![q, p25, med, p75]).collect::<Vec<_>>(),
            )).collect::<std::collections::BTreeMap<_, _>>(),
            // zones: ladder inversions split by codec x quality zone and by
            // reviewed content class x quality zone — the WHERE behind
            // `mono_pct`. Same events, same MATERIAL threshold, so the
            // split_kind=="all" rows reconcile with mono_pct exactly.
            "zones": dial_zones_json,
            // inversion_truth: the 2026-09-05 two-reference ruling as it
            // applied to THIS run. `mono` above is the DIAL-attributed number
            // under `effective == "agree"`; `mono_single` is the pre-ruling
            // one. Always emitted, so a reader can never mistake "the rule did
            // not run" for "no encoder inversions".
            "inversion_truth": {
                "scheme": "two-reference-agree-2026-09-05",
                "effective": itr.effective,
                "requested": itr.requested,
                "reference_table": itr.source,
                "butteraugli_variant": itr.variant,
                "n_reference_cells": itr.n_reference_cells,
                "ssim2_margin_pt": gaddr::ENCODER_SSIM2_MARGIN_PT,
                "butteraugli_margin": itr.variant.as_deref().and_then(
                    |v| gaddr::ButteraugliVariant::parse(v).ok()).map(|v| v.margin()),
                "n_encoder_attributed": itr.n_encoder,
                "n_attribution_unknown": itr.n_unknown,
                "mono_dial": nan_null(dial_metrics.mono),
                "mono_single": nan_null(itr.mono_single),
                "note": itr.note,
                "encoder_pairs": itr.encoder_pairs.iter().map(
                    |(img, cd, q0, q1, drop, ds, db)| json!({
                        "image_id": img, "codec": cd, "q_lo": q0, "q_hi": q1,
                        "dial_drop": drop, "d_ssim2": ds, "d_butteraugli": db,
                    })).collect::<Vec<_>>(),
            },
        });

        // corruption: the real bake_verdict gate (score(corruption) < score(q20)
        // pass-rate) — not a detection/false-positive ROC. null when no grid.
        let corruption = match &corruption_stats {
            Some(cs) => json!({
                "n_triples": cs.n_triples,
                "pass_q20": cs.pass_q20,
                "pass_q10": cs.pass_q10,
                "per_family": cs.per_family.iter().map(|(fam, rate, n)| json!({
                    "family": fam, "pass_rate": rate, "n": n,
                })).collect::<Vec<_>>(),
            }),
            None => Value::Null,
        };

        // corruption_head: the same gate evaluated on the companion head bake
        // (`--corruption-head`) — the shipping design's corruption owner.
        // null when no head was given (or it failed to load / mismatched).
        let corruption_head = match &corruption_head_stats {
            Some(cs) => json!({
                "head": args.corruption_head.as_ref().map(|p| basename(p)),
                "n_triples": cs.n_triples,
                "pass_q20": cs.pass_q20,
                "pass_q10": cs.pass_q10,
                "per_family": cs.per_family.iter().map(|(fam, rate, n)| json!({
                    "family": fam, "pass_rate": rate, "n": n,
                })).collect::<Vec<_>>(),
            }),
            None => Value::Null,
        };

        // corruption_deploy: the registered `final = min(perceptual, gate)`
        // composition — the product-visible gate, distinct from either half.
        let corruption_deploy = match &corruption_deploy_stats {
            Some(cs) => json!({
                "head": args.corruption_head.as_ref().map(|p| basename(p)),
                "threshold": args.corruption_head_threshold,
                "n_triples": cs.n_triples,
                "pass_q20": cs.pass_q20,
                "pass_q10": cs.pass_q10,
            }),
            None => Value::Null,
        };

        // per_pair: {corpus: {pred, <mos|jnd>}} for the rank corpora + a
        // {kadis: {pred, ssim2, butter, cvvdp}} block from the metric parquet.
        // `sdr25` added 2026-08-04 (campaign Appendix I): its target is `q_jnd`,
        // a JND distance from the pristine original, so the per-pair axis is JND
        // and was being labelled "MOS (human)" on the dashboard scatter. The
        // dashboard resolves whichever key a bake carries (gauntlet.py
        // REFERENCES), so board JSONs emitted BEFORE this fix still carry sdr25
        // under "mos" and render fine — only newly-emitted ones are relabelled.
        let jnd_prefixes = ["aic3", "aic4", "konjnd", "sdr25"];
        let mut per_pair = Map::new();
        for r in &results {
            let idx = stride(r.rescaled_scores.len(), args.perpair_cap);
            let pred: Vec<f64> = idx.iter().map(|&i| r.rescaled_scores[i]).collect();
            let tgt: Vec<f64> = idx.iter().map(|&i| r.humans[i]).collect();
            let key = if jnd_prefixes.iter().any(|j| r.name.starts_with(j)) {
                "jnd"
            } else {
                "mos"
            };
            per_pair.insert(r.name.to_string(), json!({ "pred": pred, key: tgt }));
        }
        // KADIS multi-metric per_pair. Read a bounded window (≤40k rows) then
        // stride to the cap for source diversity; score the bake's first
        // n_inputs features (the frozen v1 block for a 372 bake).
        if let Some(job) = perpair_job {
            let perpair_load = job
                .join()
                .unwrap_or_else(|_| Err("per-pair loader thread panicked".to_string()));
            pt.mark("--full-json kadis per-pair (join background load)");
            match perpair_load {
                Ok(sample) if sample.n_features >= n_inputs => {
                    let idx = stride(sample.feature_rows.len(), args.perpair_cap);
                    let rows: Vec<Vec<f64>> = idx
                        .iter()
                        .map(|&i| sample.feature_rows[i][..n_inputs].to_vec())
                        .collect();
                    let pred = ens.score_rows(&rows);
                    let mut obj = Map::new();
                    obj.insert("pred".into(), json!(pred));
                    for (col, key) in PERPAIR_METRIC_COLS {
                        if let Some((_, vals)) = sample.metrics.iter().find(|(n, _)| n == col) {
                            let s: Vec<f64> = idx.iter().map(|&i| vals[i]).collect();
                            obj.insert((*key).to_string(), json!(s));
                        }
                    }
                    per_pair.insert("kadis".to_string(), Value::Object(obj));
                }
                Ok(sample) => eprintln!(
                    "bake_verdict --full-json: perpair-metrics has {} features, bake needs {} — skipping kadis per_pair",
                    sample.n_features, n_inputs
                ),
                Err(e) => {
                    eprintln!("bake_verdict --full-json: perpair-metrics load failed: {e}")
                }
            }
        } else {
            eprintln!(
                "bake_verdict --full-json: perpair-metrics {} absent — no ssim2/butter/cvvdp scatter",
                args.perpair_metrics.display()
            );
        }

        // Reproduction provenance: embedded zentrain.repro > .spec.json sidecar > null.
        let repro_value: Value = {
            let embedded = model
                .metadata()
                .get_utf8("zentrain.repro")
                .ok()
                .and_then(|s| serde_json::from_str::<Value>(s).ok());
            match embedded {
                Some(mut v) => {
                    if let Some(o) = v.as_object_mut() {
                        o.insert("source".into(), json!("embedded"));
                    }
                    v
                }
                None => {
                    let mut sc = args.bake.clone().into_os_string();
                    sc.push(".spec.json");
                    match std::fs::read_to_string(PathBuf::from(sc))
                        .ok()
                        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
                    {
                        Some(mut v) => {
                            if let Some(o) = v.as_object_mut() {
                                o.insert("source".into(), json!("sidecar"));
                            }
                            eprintln!(
                                "bake_verdict: no embedded zentrain.repro (legacy bake) — using .spec.json sidecar"
                            );
                            v
                        }
                        None => {
                            eprintln!(
                                "bake_verdict: ⚠ NO reproduction provenance (no embedded zentrain.repro, no .spec.json) — this bake is irreproducible without archaeology"
                            );
                            Value::Null
                        }
                    }
                }
            }
        };
        // `zentrain.sample_coverage` — WHAT THE RUN'S SAMPLER TOUCHED
        // (2026-09-04). A separate bake metadata key, because it is a
        // MEASUREMENT rather than provenance: a bake without it is still
        // reproducible. It is surfaced as `repro.sample_coverage` because
        // that is where the board already reads it from
        // (`gauntlet.py`: `(o.get("repro") or {}).get("sample_coverage")`),
        // so a bake trained with it renders without a board change.
        //
        // ABSENT IS ABSENT. Every bake trained before this landed carries
        // no block; the field is simply not inserted, and the board renders
        // NOT MEASURED — never a zero, which would read as "this run
        // covered nothing".
        let repro_value: Value = match model.metadata().get_utf8("zentrain.sample_coverage") {
            Ok(s) => match (serde_json::from_str::<Value>(s), repro_value) {
                (Ok(cov), Value::Object(mut o)) => {
                    o.insert("sample_coverage".into(), cov);
                    Value::Object(o)
                }
                // No repro object to attach to (a bake with coverage but no
                // provenance is not a shape the trainer can emit, but do not
                // silently drop a measurement if one ever appears).
                (Ok(cov), other) => {
                    if other.is_null() {
                        json!({ "source": "sample_coverage_only", "sample_coverage": cov })
                    } else {
                        other
                    }
                }
                (Err(e), other) => {
                    eprintln!(
                        "bake_verdict: zentrain.sample_coverage is present but unparseable ({e}) — reporting NOT MEASURED rather than a partial value"
                    );
                    other
                }
            },
            Err(_) => repro_value,
        };
        let full = json!({
            "bake": args.bake.display().to_string(),
            "bake_sha256": bake_sha,
            "name": name,
            "regime": regime,
            // WHICH RULER produced every `rank`/`per_pair` number below: the
            // resolved features root, its registered era label, its manifest
            // sha + declared regime, and the per-corpus files actually read.
            // `regime` above is a campaign flag string and is NOT this fact.
            "features_root": features_root_block(&args.features_root, &corpus_prov),
            // FEATURE-SET IDS (docs/FEATURE_SET_IDS.md). `regime`/`n_inputs`
            // above are LEGACY ALIASES — a width, not an identity ("944" alone
            // has named seven feature sets). This block is the identity: the
            // bake's CONSUMER id (families read, at its caller width, hashed
            // over the exact read set), the table's PRODUCER id, whether the
            // table's id had to be INFERRED, and every disagreement between
            // them. The board reads this; it recomputes nothing.
            "feature_set": feature_set_block(model, &args.features_root),
            "n_inputs": n_inputs,
            // Architecture + in/out modifiers (transforms, winsor bounds, spline,
            // heads) — the structured `zenpredict inspect`.
            "model": model_block,
            // Reproduction instructions. Preferred source: the `zentrain.repro`
            // entry EMBEDDED in the bake bytes (mandatory for new trainer output —
            // inseparable from the model). Fallback for legacy bakes: the
            // `<bake>.spec.json` sidecar (argv + groups; separable, so flagged).
            // null = irreproducible without archaeology — the report warns.
            "repro": repro_value,
            // Canonical product-weighted ranking composite (single Rust source;
            // the dashboard READS this, never re-derives it). KADID/TID excluded.
            "composite": product_composite(&results),
            // CODEC_TARGET_GOALS scorecard values (same numbers as the report's
            // scorecard table; null when the run computed no gates).
            "gates": gates_json.clone().unwrap_or(Value::Null),
            // Injected by run_full_eval.sh from diffmap_block_coherence --bake.
            "m3_coherence": Value::Null,
            "rank": rank,
            "dial": dial,
            "corruption": corruption,
            "corruption_head": corruption_head,
            "corruption_deploy": corruption_deploy,
            "per_pair": per_pair,
        });
        let write_json = |path: &Path, value: &Value, label: &str| -> bool {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            match serde_json::to_string_pretty(value) {
                Ok(s) => {
                    if let Err(e) = std::fs::write(path, s) {
                        eprintln!("bake_verdict: failed to write {}: {e}", path.display());
                        return false;
                    }
                    eprintln!("wrote {label} to {}", path.display());
                    true
                }
                Err(e) => {
                    eprintln!("bake_verdict: {label} serialize failed: {e}");
                    false
                }
            }
        };
        if let Some(json_path) = &args.full_json
            && !write_json(json_path, &full, "full-eval json")
        {
            return ExitCode::from(1);
        }
        if let Some(fe_path) = &args.fulleval {
            // Schema-complete fulleval: the same content with every M3 slot
            // present as an explicit null. The M3/M3a values are measured by
            // `diffmap_block_coherence` (image I/O — not this binary's job);
            // run_full_eval.sh injects them INTO these keys, so a fulleval
            // file's key set no longer depends on which wrapper produced it.
            let mut fe = full.clone();
            let o = fe.as_object_mut().expect("fulleval root is an object");
            for key in [
                "m3_coherence",
                "m3_n",
                "m3_dropped_mass_pct",
                "m3a_coherence",
                "m3a_n",
            ] {
                o.entry(key).or_insert(Value::Null);
            }
            if !write_json(fe_path, &fe, "fulleval json") {
                return ExitCode::from(1);
            }
        }
    }

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

    // Self-contained big HTML report (the "html as well as console" half).
    if let Some(html_path) = &args.html {
        let title = format!(
            "metric eval — {}",
            args.bake
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_default()
        );
        let html = eval_report::markdown_to_html(&buf, &title);
        if let Some(parent) = html_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match std::fs::write(html_path, &html) {
            Ok(()) => eprintln!(
                "wrote HTML report to {} ({} KB)",
                html_path.display(),
                html.len() / 1024
            ),
            Err(e) => {
                eprintln!(
                    "bake_verdict: failed to write HTML {}: {e}",
                    html_path.display()
                );
                return ExitCode::from(1);
            }
        }
    }

    pt.mark("--full-json / --output / --html write");
    pt.finish();
    // `elapsed` is captured before the markdown tail (line ~2691) so the
    // report's own "Wall time" row stays comparable with historical
    // verdicts; `pt.total()` is the honest end-to-end number. They differ
    // by the whole `--full-json` block, which is why the trace exists.
    eprintln!(
        "bake_verdict: complete in {:.2}s (report-timer; {:.2}s end-to-end)",
        elapsed.as_secs_f64(),
        pt.total()
    );
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::load_peer_dial_scores;
    use zensim_validate::parquet_loader::DialGrid;

    /// A three-row two-ladder grid, including a fractional q (the JXL
    /// q-equivalent `100 − 4·distance` shape) so the key rounding is exercised.
    fn tiny_grid() -> DialGrid {
        DialGrid {
            image_id: vec!["imgA".into(), "imgA".into(), "imgB".into()],
            codec: vec!["jpeg".into(), "jpeg".into(), "jxl".into()],
            q: vec![10.0, 90.0, 99.9],
            codec_param: vec![10.0, 90.0, 0.025],
            param_kind: vec!["q".into(), "q".into(), "distance".into()],
            feature_rows: vec![vec![0.0], vec![0.0], vec![0.0]],
            n_features: 1,
        }
    }

    fn write_tmp(name: &str, body: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "bv_peer_dial_{}_{}_{name}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&p, body).unwrap();
        p
    }

    /// The join must align to GRID ROW ORDER, not file order, and must not
    /// care about extra rows the table carries for cells this grid dropped
    /// (the stored reference-metric tables cover a superset of every grid).
    #[test]
    fn peer_dial_scores_align_to_grid_row_order_and_tolerate_extra_rows() {
        let f = write_tmp(
            "ok.tsv",
            // deliberately shuffled, plus two cells absent from the grid
            "image_id\tcodec\tq\tpred\n\
             imgB\tjxl\t99.9\t3.5\n\
             imgZ\tjpeg\t50\t999\n\
             imgA\tjpeg\t90\t2.5\n\
             imgA\tjpeg\t10\t1.5\n\
             imgA\twebp\t10\t999\n",
        );
        let v = load_peer_dial_scores(&f, &tiny_grid()).expect("join");
        assert_eq!(v, vec![1.5, 2.5, 3.5], "scores must follow grid row order");
        let _ = std::fs::remove_file(&f);
    }

    /// A partial table is a DIFFERENT measurement, not a smaller one: a dial
    /// panel over some of the ladders would silently report a better (or
    /// worse) monotonicity than the grid actually contains. Refuse, loudly.
    #[test]
    fn peer_dial_scores_refuse_a_partially_covered_grid() {
        let f = write_tmp(
            "partial.tsv",
            "image_id\tcodec\tq\tpred\nimgA\tjpeg\t10\t1.5\n",
        );
        let e = load_peer_dial_scores(&f, &tiny_grid()).expect_err("must refuse");
        assert!(e.contains("2 of 3"), "must name the shortfall, got: {e}");
        assert!(
            e.contains("imgA, jpeg, 90") || e.contains("imgB, jxl"),
            "must name an unmatched cell, got: {e}"
        );
        let _ = std::fs::remove_file(&f);
    }

    /// Column order is not part of the contract — the header names are.
    #[test]
    fn peer_dial_scores_locate_columns_by_name_not_position() {
        let f = write_tmp(
            "reordered.tsv",
            "pred\tq\timage_id\tcodec\n\
             1.5\t10\timgA\tjpeg\n\
             2.5\t90\timgA\tjpeg\n\
             3.5\t99.9\timgB\tjxl\n",
        );
        let v = load_peer_dial_scores(&f, &tiny_grid()).expect("join");
        assert_eq!(v, vec![1.5, 2.5, 3.5]);
        let _ = std::fs::remove_file(&f);
    }

    /// Zone edges are a PRODUCT statement (aggressive / ordinary / near-lossless),
    /// so pin them and pin the midpoint rule the split is keyed on.
    #[test]
    fn zone_labels_partition_the_q_axis_at_the_documented_edges() {
        assert_eq!(super::ZONE_EDGES, [50.0, 85.0]);
        assert_eq!(super::zone_of(0.0), "q<50");
        assert_eq!(super::zone_of(49.999), "q<50");
        assert_eq!(super::zone_of(50.0), "q50-85");
        assert_eq!(super::zone_of(84.999), "q50-85");
        assert_eq!(super::zone_of(85.0), "q>=85");
        assert_eq!(super::zone_of(100.0), "q>=85");
        // every label the report orders on is reachable
        for z in super::ZONE_LABELS {
            assert!(
                [
                    super::zone_of(10.0),
                    super::zone_of(60.0),
                    super::zone_of(95.0)
                ]
                .contains(&z)
            );
        }
    }

    /// The zone split counts the SAME events the G3 gate counts; if that
    /// threshold ever moves, both must move together.
    #[test]
    fn material_threshold_is_shared_with_the_gate() {
        assert_eq!(super::MATERIAL_INV_PT, 0.5);
    }

    /// The content-class table must cover the dial grid's images. This asserts
    /// the coupling (bake_verdict reads it) without loading the 25 MB grid: the
    /// table's own module test pins the counts, this pins that bake_verdict
    /// resolves through it and that an unknown id is reported, not merged.
    #[test]
    fn content_class_lookup_is_wired_and_reports_unknowns() {
        assert_eq!(
            zensim_validate::dial_content::class_of("5a9b3b963f852e20_512sq"),
            "text_lineart"
        );
        assert_eq!(
            zensim_validate::dial_content::class_of("not_in_the_grid"),
            zensim_validate::dial_content::UNCLASSIFIED
        );
    }

    /// `--per-pair-refs` (appendix O): the 2-column default is byte-stable
    /// (three committed consumers exact-unpack it) and the opt-in 3-column
    /// form appends the interned ref id, aligned 1:1.
    #[test]
    fn per_pair_format_default_two_columns_refs_opt_in_three() {
        let humans = [0.91_f64, 0.99];
        let preds = [88.5_f64, 97.25];
        assert_eq!(
            super::format_per_pair(&humans, &preds, None),
            "human\tpred\n0.91\t88.5\n0.99\t97.25\n"
        );
        assert_eq!(
            super::format_per_pair(&humans, &preds, Some(&[7u32, 7])),
            "human\tpred\tref\n0.91\t88.5\t7\n0.99\t97.25\t7\n"
        );
    }

    /// An ANTI-CORRELATED bake must never RENDER as a high scorer.
    ///
    /// Regression gate for the 2026-08-04 finding (campaign APPENDIX F): the ext-lineage
    /// KADID eval tables stored `human_score` inverted, so 110 of 188 board bakes were
    /// backwards on KADID while every report printed a positive magnitude. `|SROCC|`
    /// display is what made that invisible for six weeks.
    #[test]
    fn inverted_corpus_renders_as_inverted_not_as_a_high_score() {
        // quality-oriented corpus, bake is backwards on it
        let cell = srocc_cell("kadid", 0.9464, -0.9464);
        assert!(
            cell.contains("-0.9464"),
            "signed value must be shown, got {cell}"
        );
        assert!(
            cell.contains("INVERTED"),
            "inversion must be marked, got {cell}"
        );
        assert!(
            !cell.starts_with("0.9"),
            "an inverted bake must not render as a bare positive magnitude: {cell}"
        );
        // same corpus, correct direction
        let ok = srocc_cell("kadid", 0.9464, 0.9464);
        assert_eq!(ok, "+0.9464");
        assert!(!ok.contains("INVERTED"));
        // konjnd's SROCC is STRUCTURALLY negative (validation target is a PJND
        // threshold) — |SROCC| is correct there and must NOT be flagged.
        let jnd = srocc_cell("konjnd", 0.4308, -0.4308);
        assert_eq!(
            jnd, "0.4308",
            "konjnd must keep |SROCC| with no inversion marker"
        );
        assert!(!sign_is_meaningful("konjnd"));
        assert!(sign_is_meaningful("kadid") && sign_is_meaningful("tid"));
        assert!(sign_is_meaningful("cid22") && sign_is_meaningful("csiq"));
    }

    use super::*;

    /// The G-ADDR floor registry must hold a row for the grid this binary
    /// defaults to. Without it every default-flag verdict reads NOT MEASURABLE
    /// and the addressability gate is decorative — the exact failure mode the
    /// "absent is never passed" rule exists to make visible rather than to
    /// make silent.
    #[test]
    fn canonical_dial_grid_has_a_g_addr_floor_row() {
        use zensim_validate::dial_addressability as gaddr;
        assert!(
            gaddr::floor_for_grid(CANONICAL_DIAL_GRID_SHA256, gaddr::ACTIVE_REFERENCE).is_some(),
            "benchmarks/dial_addressability_floor_2026-09-04.json must register the canonical \
             dial grid (sha {}) for the ACTIVE reference `{}`. If the canonical grid is \
             rotated, measure the REFERENCE METRIC on the new one \
             (`bake_verdict --dial-peer-scores … --gaddr-json`) and APPEND a row — never edit \
             an existing row, and never drop the gate to unblock.",
            &CANONICAL_DIAL_GRID_SHA256[..16],
            gaddr::ACTIVE_REFERENCE
        );
        assert!(
            gaddr::floor_for_grid(CANONICAL_DIAL_GRID_SHA256, gaddr::INCUMBENT_REFERENCE).is_some(),
            "the incumbent row must survive too — it is the `incumbent` column and the grading \
             of every pre-2026-09-04 verdict"
        );
    }

    /// A verdict must name its inputs by CONTENT, not by path.
    ///
    /// Before 2026-07-15 the header said `- Bake: /mnt/v/output/.../seed7_hf0.bin`
    /// and nothing else — a path on one machine's scratch volume, no hash, no
    /// code version. That is not a citable artifact, and it could not answer
    /// "which corpus?" when two plausibly-named CID22 parquets existed
    /// (`a1050ace…` vs the canonical `6eea0825…` — measured identical in rows
    /// and all 372 features, but the report could not TELL you that).
    ///
    /// The bake sha is what links a verdict to its manifest (`grep -rl <sha>
    /// zensim/weights/manifests/`), which is what makes the whole chain —
    /// number → verdict → bake → recipe → input hashes — close. If this test
    /// fails, that chain is broken; fix the emission, do not weaken the test.
    #[test]
    fn provenance_names_every_input_by_sha256() {
        let dir = std::env::temp_dir().join("zensim_prov_test");
        std::fs::create_dir_all(&dir).unwrap();
        let bake = dir.join("fake_bake.bin");
        std::fs::write(&bake, b"not a real bake, but it hashes").unwrap();
        let corpus = dir.join("cid22.parquet");
        std::fs::write(&corpus, b"not a real parquet either").unwrap();

        let bake_sha = zensim_validate::train_manifest::sha256_file(&bake).unwrap();
        let corpus_sha = zensim_validate::train_manifest::sha256_file(&corpus).unwrap();
        let prov = provenance_block(
            &bake,
            &[("CID22".to_string(), corpus.clone(), corpus_sha.clone(), 25)],
        );

        assert!(
            prov.contains(&bake_sha[..16]),
            "the bake's sha256 must appear — it is the join key from a verdict to its \
             manifest. Got:\n{prov}"
        );
        assert!(
            prov.contains(&corpus_sha[..16]),
            "each corpus's sha256 must appear — otherwise 'which corpus produced this \
             number?' is unanswerable from the artifact. Got:\n{prov}"
        );
        assert!(prov.contains("CID22"), "corpus display name must appear");
        assert!(
            prov.contains("fake_bake.bin"),
            "the bake filename must appear alongside its hash"
        );
        // Code version: either a real short sha, or an explicit "unknown" —
        // never silently absent.
        assert!(
            prov.contains("git HEAD"),
            "the code version must appear: a number from unknown code is not \
             reproducible. Got:\n{prov}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `--features-root` must mean what it says.
    ///
    /// An absolute slot silently opts a corpus out of the flag (`Path::join`
    /// drops the root), so a reproduction pointed elsewhere would load a
    /// SILENT MIX: some corpora from the new root, the absolute ones from the
    /// original. `nonphoto` was exactly this until 2026-07-15 — and for no
    /// benefit, since its file was in the default root all along.
    ///
    /// Absolute is still allowed where the corpus genuinely lives in another
    /// root, but it must be declared with a reason. Undeclared absolute slots
    /// fail here.
    ///
    /// Absoluteness is judged platform-independently: slots point into the
    /// dev box's `/mnt/v` world, and `Path::is_absolute()` calls a rooted,
    /// prefix-less `/...` string RELATIVE on Windows. That asymmetry is what
    /// made the 2026-07..08 windows-x64 CI red unique: the then-pinned
    /// `hf_nearlossless` slot counted as absolute on Unix (both halves
    /// pass) but as relative on Windows, so ONLY windows fired the
    /// stale-exemption half against a pin that was load-bearing everywhere
    /// else. `slot_is_absolute` closes that hole for good.
    #[test]
    /// **D2 (ADD156 ship audit, `benchmarks/add156_ship_audit_2026-08-31.md`).**
    /// The default KonJND ruler must be the JPEG-**504** file, not the diluted
    /// 1,008-ref build that also lives in the same directory.
    ///
    /// `konjnd_features_372col_2026-05-15.parquet` holds all 1,008 refs — the
    /// JPEG *and BPG* halves — while every 720/944-class row scores the JPEG-504
    /// half. Reading the diluted file does not merely shift the number, it
    /// **inverts** cross-model comparisons. Measured on the same root, same
    /// binary, same code path (this audit, replicating the report exactly):
    ///
    /// | ruler | ADD156 | shipped `B` | winner |
    /// |---|---:|---:|---|
    /// | diluted 1,008 (the OLD default) | 0.4462 | 0.6497 | `B` by +0.204 |
    /// | JPEG-504 (the new default) | 0.5332 | 0.5194 | ADD156 by +0.014 |
    ///
    /// This test pins the default so the diluted file cannot drift back in.
    fn konjnd_default_ruler_is_jpeg504_not_the_diluted_file() {
        let konjnd = CORPORA
            .iter()
            .find(|c| c.name == "konjnd")
            .expect("konjnd is a registered corpus");

        // The preferred list is the shared owner's, so `bake_compare` reads the
        // same ruler — a second copy of these filenames is exactly how one
        // binary would go on publishing the diluted number.
        assert_eq!(
            konjnd.preferred_slots, KONJND_JPEG504_372_SLOTS,
            "konjnd must resolve through the shared ruler list"
        );
        assert!(
            konjnd.preferred_slots.iter().all(|f| f.contains("jpeg504")),
            "every preferred konjnd ruler must be a JPEG-504 build"
        );
        // The diluted file stays as the LAST-RESORT slot (a root that only has
        // it still produces a number), never as the preferred one.
        assert!(
            konjnd.filename.contains("konjnd_features_372col"),
            "the diluted build remains the declared fallback"
        );

        // Under the shipped default root the resolution must land on JPEG-504.
        let root = Path::new(DEFAULT_FEATURES_ROOT_372);
        if root.exists() {
            let r = resolve_372_slot(konjnd, root);
            assert!(
                r.file.contains("jpeg504") && !r.degraded,
                "default root {} resolved konjnd to `{}` (degraded={}) — expected a \
                 JPEG-504 ruler",
                root.display(),
                r.file,
                r.degraded
            );
        }

        // A root with neither preferred ruler falls back, and says so.
        let empty = std::env::temp_dir().join("zensim-d2-no-ruler-here");
        let r = resolve_372_slot(konjnd, &empty);
        assert_eq!(r.file, konjnd.filename);
        assert!(
            r.degraded,
            "falling back past every preferred ruler must flag"
        );
        let lines = ruler_lines(&[konjnd], &empty);
        assert_eq!(lines.len(), 1);
        assert!(
            lines[0].contains("LAST-RESORT") && lines[0].contains("DILUTED"),
            "the fallback must be announced, not silent; got: {}",
            lines[0]
        );

        // A single-ruler corpus is unaffected and prints nothing.
        let cid22 = CORPORA.iter().find(|c| c.name == "cid22").unwrap();
        assert!(cid22.preferred_slots.is_empty());
        assert_eq!(resolve_372_slot(cid22, &empty).file, cid22.filename);
        assert!(ruler_lines(&[cid22], &empty).is_empty());
    }

    #[test]
    fn corpus_slots_are_relative_or_declared_pinned() {
        fn slot_is_absolute(slot: &str) -> bool {
            Path::new(slot).is_absolute() || slot.starts_with('/')
        }
        let undeclared: Vec<_> = CORPORA
            .iter()
            .filter(|c| slot_is_absolute(c.filename))
            .filter(|c| {
                !PINNED_OUTSIDE_FEATURES_ROOT
                    .iter()
                    .any(|(name, _)| *name == c.name)
            })
            .map(|c| format!("  {} -> {}", c.name, c.filename))
            .collect();
        assert!(
            undeclared.is_empty(),
            "corpus slot(s) are absolute but not declared in \
             PINNED_OUTSIDE_FEATURES_ROOT:\n{}\n\n\
             An absolute slot silently ignores --features-root, so a reproduction pointed at \
             another root loads a MIX without being told. Either make it relative (nonphoto's \
             absolute path bought nothing — the file was in the default root all along), or \
             add it to PINNED_OUTSIDE_FEATURES_ROOT with the reason it must live elsewhere.",
            undeclared.join("\n")
        );

        // ...and the declarations must not rot: every pinned name must exist
        // and still actually be absolute, so a fixed slot cannot leave a stale
        // exemption behind that would re-permit the hazard later.
        for (name, _reason) in PINNED_OUTSIDE_FEATURES_ROOT {
            let c = CORPORA.iter().find(|c| c.name == *name).unwrap_or_else(|| {
                panic!("PINNED_OUTSIDE_FEATURES_ROOT names {name}, which is not a corpus")
            });
            assert!(
                slot_is_absolute(c.filename),
                "{name} is declared pinned-outside-features-root but its slot is relative now. \
                 Remove the stale exemption — leaving it lets a future absolute slot pass unnoticed."
            );
        }
    }

    /// The default dial grid must be the quarantined-v2 one, and its pinned
    /// hash must match the file.
    ///
    /// Both defects in the original grid were found, documented, and fixed by
    /// BUILDING the quarantined siblings — and the default was never switched,
    /// so from 2026-05-29 to 2026-07-15 every default run scored its dial panel
    /// against a grid with 9/115 garbage ladders and 33 pre-fix JXL cells.
    /// Zero code referenced `_quarantined_v2`; only prose did. This test is the
    /// wiring that prose could not provide.
    ///
    /// Skips when the grid is not on disk (CI does not mount `/mnt/v`), so it
    /// verifies the pin wherever the data exists and never fails for its
    /// absence.
    #[test]
    fn canonical_dial_grid_is_the_quarantined_v2_grid() {
        assert!(
            CANONICAL_DIAL_GRID.contains("_quarantined_v2"),
            "the canonical dial grid must be the v2-quarantined one: the original carries the \
             w11 extraction corruption (9/115 ladders) AND 33 pre-fix JXL cells at d=0.025, and \
             the _quarantined (v1) sibling fixes only the first. See \
             benchmarks/eval_grids_2026-05-29.pointer.md. Got: {CANONICAL_DIAL_GRID}"
        );
        let p = Path::new(CANONICAL_DIAL_GRID);
        if !p.exists() {
            eprintln!("skip: {CANONICAL_DIAL_GRID} not on this host (CI has no /mnt/v)");
            return;
        }
        let actual = zensim_validate::train_manifest::sha256_file(p).expect("hash dial grid");
        assert_eq!(
            actual, CANONICAL_DIAL_GRID_SHA256,
            "the canonical dial grid's bytes changed. If it was legitimately rebuilt, update \
             CANONICAL_DIAL_GRID_SHA256 *and* the pointer doc in the same commit — do NOT just \
             bump the constant, because the pin is the only thing that noticed."
        );
    }

    /// An unhashable bake must degrade loudly inside the report rather than
    /// panic the run or silently omit the row.
    #[test]
    fn provenance_reports_an_unhashable_bake_instead_of_panicking() {
        let prov = provenance_block(Path::new("/nonexistent/nope.bin"), &[]);
        assert!(prov.contains("UNHASHABLE"), "got:\n{prov}");
    }

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

    // ── --regime 944 preset (the one-command SOTA-944 evaluation) ──────────

    fn parse(argv: &[&str]) -> Args {
        parse_args_from(argv.iter().map(|s| s.to_string())).expect("parse")
    }

    /// `--ensemble-weights` is parsed, validated and NORMALISED at parse time,
    /// so the vector the report publishes is the vector that scored.
    #[test]
    fn ensemble_weights_parse_validate_and_normalise() {
        // absent by default — the historical equal-weight path
        let a = parse(&["--bake", "x.bin", "--ensemble", "x.bin,y.bin"]);
        assert!(a.ensemble_weights.is_none());
        // normalised to sum 1
        let a = parse(&[
            "--bake",
            "x.bin",
            "--ensemble",
            "x.bin,y.bin",
            "--ensemble-weights",
            "3,1",
        ]);
        let w = a.ensemble_weights.expect("weights");
        assert_eq!(w.len(), 2);
        assert!((w[0] - 0.75).abs() < 1e-12 && (w[1] - 0.25).abs() < 1e-12);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // a zero member is legal (it is a declared ablation), all-zero is not
        let a = parse(&[
            "--bake",
            "x.bin",
            "--ensemble",
            "x.bin,y.bin",
            "--ensemble-weights",
            "1,0",
        ]);
        assert_eq!(a.ensemble_weights.expect("weights"), vec![1.0, 0.0]);
    }

    /// Every refusal is loud. A weight vector that cannot describe the members
    /// must never be silently truncated, padded or ignored.
    #[test]
    fn ensemble_weights_refusals_are_loud() {
        let bad: &[(&[&str], &str)] = &[
            (
                &["--bake", "x.bin", "--ensemble-weights", "0.5,0.5"],
                "requires --ensemble",
            ),
            (
                &[
                    "--bake",
                    "x.bin",
                    "--ensemble",
                    "x.bin,y.bin",
                    "--ensemble-weights",
                    "0.5",
                ],
                "entries but --ensemble has",
            ),
            (
                &[
                    "--bake",
                    "x.bin",
                    "--ensemble",
                    "x.bin,y.bin",
                    "--ensemble-weights",
                    "0,0",
                ],
                "must not sum to zero",
            ),
            (
                &[
                    "--bake",
                    "x.bin",
                    "--ensemble",
                    "x.bin,y.bin",
                    "--ensemble-weights",
                    "-1,2",
                ],
                "must be finite and >= 0",
            ),
            (
                &[
                    "--bake",
                    "x.bin",
                    "--ensemble",
                    "x.bin,y.bin",
                    "--ensemble-weights",
                    "a,2",
                ],
                "is not a number",
            ),
        ];
        for (argv, needle) in bad {
            let e = match parse_args_from(argv.iter().map(|s| s.to_string())) {
                Ok(_) => panic!("must refuse: {argv:?}"),
                Err(e) => e,
            };
            assert!(e.contains(needle), "expected {needle:?} in {e:?}");
        }
    }

    /// **G5.2's flag half** — the unpopulated-slots refusal is ON by default
    /// and is opt-OUT, mirroring `--cross-regime`.
    ///
    /// The refusal itself is narrower than the flag name suggests, and
    /// deliberately so: it fires only when BOTH feature-set ids are STORED,
    /// because an inferred id is evidence about a name and not about bytes.
    /// Refusing on a guess would break every root on disk that has no
    /// recorded id, which is most of them.
    #[test]
    fn allow_unpopulated_slots_defaults_off_and_parses_on() {
        let a = parse(&["--bake", "x.bin"]);
        assert!(
            !a.allow_unpopulated_slots,
            "the refusal must be the default"
        );
        let a = parse(&["--bake", "x.bin", "--allow-unpopulated-slots"]);
        assert!(a.allow_unpopulated_slots);
    }

    /// C2 wrong-regime guard (appendix W): the refusal is the DEFAULT and the
    /// override is an explicit, greppable flag. The behavioral half lives in
    /// `tests/cross_regime_guard.rs` against
    /// `block_profile::folded_root_conflict`.
    #[test]
    fn cross_regime_defaults_off_and_parses_on() {
        let a = parse(&["--bake", "x.bin", "--regime", "944"]);
        assert!(!a.cross_regime, "the refusal must be the default");
        let a = parse(&["--bake", "x.bin", "--regime", "944", "--cross-regime"]);
        assert!(a.cross_regime);
    }

    /// A bare `--bake X --regime 944` must resolve the ENTIRE campaign
    /// invocation — roots, grids, per-pair source, and the corpus list. The
    /// class of error this kills: a wrapper (or a bare run) with a shorter
    /// corpus list silently producing no number for an axis, which is how the
    /// published `EM4_mask2_kw0.15_s42` HF-NL cell came to be wrong (see the
    /// campaign doc's Corrections section — `rank.hfnlproxy` was null because
    /// the corpus was never in that run's `--corpora`).
    #[test]
    fn regime_944_preset_resolves_the_full_campaign_invocation() {
        let a = parse(&["--bake", "/x/b.bin", "--regime", "944"]);
        assert!(a.regime_720, "944 shares the 720 slot mechanics");
        assert!(a.regime_944);
        assert_eq!(a.features_root, PathBuf::from(DEFAULT_FEATURES_ROOT_944));
        assert_eq!(a.dial_grid, PathBuf::from(DEFAULT_DIAL_GRID_944));
        assert_eq!(
            a.corruption_grid,
            PathBuf::from(DEFAULT_CORRUPTION_GRID_944)
        );
        assert_eq!(
            a.perpair_metrics,
            PathBuf::from(DEFAULT_PERPAIR_METRICS_944)
        );
        let names: Vec<&str> = a.corpora.iter().map(|c| c.name).collect();
        let expected: Vec<&str> = SOTA944_CORPORA.split(',').collect();
        assert_eq!(
            names, expected,
            "the bare 944 invocation must evaluate exactly the frozen campaign corpus list"
        );
    }

    /// The default 944 corpus list IS the frozen campaign list — spelled out
    /// literally here so neither the const nor the corpus registry can drift
    /// without a conscious edit to this test (and to the campaign doc, which
    /// this list mirrors: scripts/sota944_verdict.sh's §0 invocation).
    /// R1b (2026-08-30): the wrong-regime guard must ask the ROOT what regime
    /// it is, not assume that every `--regime 944` root feeds f156-371 as
    /// zeros. An all-live pools root is 944 wide WITH that block populated;
    /// refusing there blocks a correct read. An unmarked root must still read
    /// as folded (the refusing default) — silence is never "safe".
    #[test]
    fn root_declared_regime_reads_the_manifest_and_defaults_to_none() {
        let d = std::env::temp_dir().join(format!(
            "r1b_regime_probe_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        assert_eq!(root_declared_regime(&d), None, "no manifest => no claim");

        std::fs::write(
            d.join("_MANIFEST.json"),
            r#"{"description":"x","driver":"v2_ab_extract (folded720append2pools)",
                "regime":"folded720append2pools",
                "regime_purity":"never column-mix","entries":[]}"#,
        )
        .unwrap();
        let r = root_declared_regime(&d).expect("regime present");
        assert_eq!(r, "folded720append2pools");
        assert!(r.ends_with("pools"), "pools roots feed the block LIVE");

        // `regime_purity` must not be mistaken for `regime`, and a folded root
        // must NOT read as live.
        std::fs::write(
            d.join("_MANIFEST.json"),
            r#"{"regime_purity":"never column-mix","regime":"folded720append2"}"#,
        )
        .unwrap();
        let r = root_declared_regime(&d).expect("regime present");
        assert_eq!(r, "folded720append2");
        assert!(!r.ends_with("pools") && !r.ends_with("carriers"));

        // A manifest with no `regime` key at all (every pre-2026-08-30 root).
        std::fs::write(
            d.join("_MANIFEST.json"),
            r#"{"description":"old","entries":[]}"#,
        )
        .unwrap();
        assert_eq!(root_declared_regime(&d), None);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn regime_944_default_corpora_match_the_frozen_campaign_list() {
        assert_eq!(
            SOTA944_CORPORA,
            "cid22,kadid,tid,konjnd,aic3,aic4,csiq,live,sdr25,imazen26,nonphoto,hfnlproxy",
            "SOTA944_CORPORA changed. That list is the campaign's frozen §0 invocation \
             (benchmarks/sota944_campaign_2026-08-03.md); update the campaign doc's repro \
             note in the same commit or revert."
        );
        // Every name must resolve a 720-slot file (the mechanism 944 rides on),
        // so none of the preset corpora can be silently dropped by the
        // slot-filter belt-and-braces.
        for name in SOTA944_CORPORA.split(',') {
            assert!(
                slot_720(name).is_some(),
                "preset corpus {name:?} has no slot_720 entry — it would be silently \
                 filtered out of every --regime 944 run"
            );
        }
    }

    /// Explicit flags override every piece of the preset, exactly as
    /// `--regime 720` behaves.
    #[test]
    fn regime_944_respects_explicit_overrides() {
        let a = parse(&[
            "--bake",
            "/x/b.bin",
            "--regime",
            "944",
            "--features-root",
            "/tmp/other-root",
            "--dial-grid",
            "/tmp/dial.parquet",
            "--corruption-grid",
            "/tmp/corr.parquet",
            "--perpair-metrics",
            "/tmp/pp.parquet",
            "--corpora",
            "cid22,konjnd",
        ]);
        assert_eq!(a.features_root, PathBuf::from("/tmp/other-root"));
        assert_eq!(a.dial_grid, PathBuf::from("/tmp/dial.parquet"));
        assert_eq!(a.corruption_grid, PathBuf::from("/tmp/corr.parquet"));
        assert_eq!(a.perpair_metrics, PathBuf::from("/tmp/pp.parquet"));
        let names: Vec<&str> = a.corpora.iter().map(|c| c.name).collect();
        assert_eq!(names, vec!["cid22", "konjnd"]);
    }

    /// THE DEFAULT ROOT IS PINNED (2026-08-30 flip). A flagless `bake_verdict` must read
    /// the CURRENT-extractor 372 root; a silent revert to `2026-05-15-full-features` would
    /// change every future 372-regime number without touching a call site, and the era
    /// shift is model-specific so no correction factor could undo it
    /// (`benchmarks/eval372_current_root_2026-08-30.md`). The stored root stays a valid
    /// STORED-ERA read — it is just no longer what "no flag" means.
    /// The `--full-json` `features_root` block must name the root the run
    /// ACTUALLY resolved — in every regime, including the ones that swap the
    /// root out from under an unset `--features-root`. Without this, the era of
    /// a stored verdict is recoverable only by re-running and diffing
    /// predictions, which is precisely how 7 of 9 board rows spent a day being
    /// misattributed to the stored 372 root when they were ext720 reads.
    #[test]
    fn full_json_features_root_matches_the_resolved_root_in_every_regime() {
        for args in [
            vec!["--bake", "/x/b.bin"],
            vec!["--bake", "/x/b.bin", "--regime", "720"],
            vec!["--bake", "/x/b.bin", "--regime", "944"],
            vec!["--bake", "/x/b.bin", "--features-root", "/x/some-root"],
            vec![
                "--bake",
                "/x/b.bin",
                "--regime",
                "944",
                "--features-root",
                "/x/explicit-wins",
            ],
        ] {
            let a = parse(&args);
            let blk = features_root_block(&a.features_root, &[]);
            assert_eq!(
                blk["path"].as_str().unwrap(),
                a.features_root.display().to_string(),
                "features_root.path must be the RESOLVED root for {args:?}"
            );
            assert_eq!(
                blk["era"].as_str().unwrap(),
                era_of(&a.features_root),
                "era label must come from the registry, for {args:?}"
            );
            assert!(
                blk.get("corpus_files").is_some() && blk.get("manifest_sha256").is_some(),
                "the block's shape is part of the contract"
            );
        }
        // The three registered roots must be distinguishable from the block
        // alone — that is the whole point of recording it.
        let d = features_root_block(&parse(&["--bake", "/x/b.bin"]).features_root, &[]);
        let s720 = features_root_block(
            &parse(&["--bake", "/x/b.bin", "--regime", "720"]).features_root,
            &[],
        );
        let stored = features_root_block(
            Path::new(zensim_validate::eval_roots::STORED_FEATURES_ROOT_2026_05_15),
            &[],
        );
        assert_ne!(d["path"], s720["path"]);
        assert_ne!(d["path"], stored["path"]);
        assert_ne!(d["era"], stored["era"]);
        assert!(stored["era"].as_str().unwrap().contains("STORED-ERA"));
    }

    /// The block describes the features root and ONLY it: `corpus_prov` also
    /// carries the dial grid, which lives elsewhere and is a different artifact
    /// with its own era caveat.
    #[test]
    fn features_root_block_lists_only_files_under_that_root() {
        let root = std::env::temp_dir().join(format!(
            "zensim_frblock_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("_MANIFEST.json"),
            r#"{"description":"probe","regime":"v1-372 probe","corpora":{}}"#,
        )
        .unwrap();
        let inside = root.join("cid22_features_372col.parquet");
        std::fs::write(&inside, b"corpus-bytes").unwrap();
        let outside = std::env::temp_dir().join("some_dial_grid_not_under_root.parquet");
        std::fs::write(&outside, b"dial-bytes").unwrap();

        let prov = vec![
            (
                "CID22".to_string(),
                inside.clone(),
                "aaa".to_string(),
                12u64,
            ),
            (
                "dial grid (canonical)".to_string(),
                outside.clone(),
                "bbb".to_string(),
                10u64,
            ),
        ];
        let blk = features_root_block(&root, &prov);

        assert_eq!(blk["path"].as_str().unwrap(), root.display().to_string());
        assert_eq!(blk["declared_regime"].as_str().unwrap(), "v1-372 probe");
        assert_eq!(
            blk["manifest_sha256"].as_str().unwrap(),
            zensim_validate::train_manifest::sha256_file(&root.join("_MANIFEST.json")).unwrap(),
            "the root's manifest is named by content, not by path"
        );
        let files = blk["corpus_files"].as_array().unwrap();
        assert_eq!(files.len(), 1, "the dial grid is not a features-root file");
        assert_eq!(files[0]["name"].as_str().unwrap(), "CID22");
        assert_eq!(
            files[0]["file"].as_str().unwrap(),
            "cid22_features_372col.parquet"
        );
        assert_eq!(files[0]["sha256"].as_str().unwrap(), "aaa");

        // An unmanaged root is honestly null, never a fabricated hash.
        let bare = root.join("kon504");
        std::fs::create_dir_all(&bare).unwrap();
        let b = features_root_block(&bare, &[]);
        assert!(b["manifest_sha256"].is_null() && b["declared_regime"].is_null());
        assert!(b["era"].as_str().unwrap().contains("UNKNOWN"));

        std::fs::remove_dir_all(&root).ok();
        std::fs::remove_file(&outside).ok();
    }

    #[test]
    fn default_features_root_is_the_current_extractor_372_root() {
        let a = parse(&["--bake", "/x/b.bin"]);
        assert!(!a.regime_720 && !a.regime_944);
        assert_eq!(a.features_root, PathBuf::from(DEFAULT_FEATURES_ROOT_372));
        assert_eq!(
            a.features_root,
            PathBuf::from("/mnt/v/zen/zensim-training/2026-08-30-full-features-372")
        );
        assert_ne!(
            a.features_root,
            PathBuf::from(zensim_validate::eval_roots::STORED_FEATURES_ROOT_2026_05_15)
        );
        // …and the run line can name the era it just pinned.
        assert!(era_of(&a.features_root).contains("current-extractor 372"));
    }

    /// An explicit `--features-root` still wins, and the note follows it — that is how a
    /// deliberate stored-era read stays available and stays LABELED.
    #[test]
    fn explicit_features_root_overrides_the_default_and_relabels_the_era() {
        let a = parse(&[
            "--bake",
            "/x/b.bin",
            "--features-root",
            zensim_validate::eval_roots::STORED_FEATURES_ROOT_2026_05_15,
        ]);
        assert_eq!(
            a.features_root,
            PathBuf::from(zensim_validate::eval_roots::STORED_FEATURES_ROOT_2026_05_15)
        );
        assert!(era_of(&a.features_root).contains("STORED-ERA 372"));
    }

    /// `--regime 720` behavior is unchanged by the 944 addition: 720 defaults,
    /// no 944 label, filtered all-corpora default.
    #[test]
    fn regime_720_defaults_unchanged() {
        let a = parse(&["--bake", "/x/b.bin", "--regime", "720"]);
        assert!(a.regime_720);
        assert!(!a.regime_944);
        assert_eq!(a.features_root, PathBuf::from(DEFAULT_FEATURES_ROOT_720));
        assert_eq!(a.dial_grid, PathBuf::from(DEFAULT_DIAL_GRID_720));
        assert_eq!(
            a.corruption_grid,
            PathBuf::from(DEFAULT_CORRUPTION_GRID_720)
        );
        // Default per-pair source stays the kadis-720 one under --regime 720.
        assert_eq!(a.perpair_metrics, PathBuf::from(DEFAULT_PERPAIR_METRICS));
        // Default corpora = every corpus with a 720 slot.
        let names: Vec<&str> = a.corpora.iter().map(|c| c.name).collect();
        let expected: Vec<&str> = CORPORA
            .iter()
            .filter(|c| slot_720(c.name).is_some())
            .map(|c| c.name)
            .collect();
        assert_eq!(names, expected);
    }

    /// Preset path/corpus resolution against the REAL data roots: every file
    /// the bare `--regime 944` invocation would read must exist. Runs only
    /// where the canonical roots are mounted (CI has no /mnt/v — same skip
    /// convention as `canonical_dial_grid_is_the_quarantined_v2_grid`).
    #[test]
    fn regime_944_preset_files_exist_on_canonical_roots() {
        let root = Path::new(DEFAULT_FEATURES_ROOT_944);
        if !root.exists() {
            eprintln!("skip: {DEFAULT_FEATURES_ROOT_944} not on this host (CI has no /mnt/v)");
            return;
        }
        let a = parse(&["--bake", "/x/b.bin", "--regime", "944"]);
        for c in &a.corpora {
            let fname = slot_720_file(c.name, &a.features_root)
                .unwrap_or_else(|| panic!("no slot for preset corpus {}", c.name));
            let p = a.features_root.join(&fname);
            assert!(
                p.exists(),
                "preset corpus {} resolves to {} which does not exist — the bare \
                 --regime 944 invocation would fail loud on it (fix the root or the slot)",
                c.name,
                p.display()
            );
        }
        assert!(a.dial_grid.exists(), "944 dial grid missing");
        assert!(a.corruption_grid.exists(), "944 corruption grid missing");
        assert!(
            a.perpair_metrics.exists(),
            "kadis-944 per-pair source missing"
        );
    }

    /// A band whose ordering is INVERTED must be distinguishable from one that
    /// is correctly ordered. `srocc` cannot do it — the panel convention takes
    /// `.abs()`, so both read the same and a DEEPER inversion reads HIGHER.
    /// This is exactly what `freeze_check`'s F8 band-tail gate consumes, while
    /// documenting itself as signed ("collapse must hurt") and carrying a
    /// `B3 >= 0.0` clause that an absolute value can never fail.
    ///
    /// Fails without the `srocc_signed` band field.
    #[test]
    fn band_srocc_signed_separates_inverted_from_healthy_tails() {
        let humans = [0.91_f64, 0.93, 0.95, 0.97, 0.99];
        let healthy = [10.0_f64, 20.0, 30.0, 40.0, 50.0]; // rank-aligned
        let inverted = [50.0_f64, 40.0, 30.0, 20.0, 10.0]; // exactly reversed

        let h_signed = zensim_validate::panel::spearman(&humans, &healthy);
        let i_signed = zensim_validate::panel::spearman(&humans, &inverted);
        assert!(
            (h_signed - 1.0).abs() < 1e-12,
            "healthy signed = {h_signed}"
        );
        assert!(
            (i_signed + 1.0).abs() < 1e-12,
            "inverted signed = {i_signed}"
        );

        // The abs'd form the board and F8 read is IDENTICAL for the two.
        let (h_abs, ..) = aggregate_panel(&healthy, &humans);
        let (i_abs, ..) = aggregate_panel(&inverted, &humans);
        assert!(
            (h_abs - i_abs).abs() < 1e-12,
            "the abs'd band srocc cannot tell them apart ({h_abs} vs {i_abs}) — \
             which is the whole reason srocc_signed exists"
        );

        // ...and the rendered cell must flag the inversion loudly.
        let cell = srocc_cell("cid22", i_abs, i_signed);
        assert!(
            cell.contains("INVERTED"),
            "inverted band rendered as {cell:?}"
        );
        assert!(!srocc_cell("cid22", h_abs, h_signed).contains("INVERTED"));
    }
}
