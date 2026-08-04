//! qsweep_eval (PreviewV0_5Tuner experiment, 2026-05-18)
//!
//! Scores a q-sweep feature CSV with one or more bake bytes, then
//! computes three tuner-specific evaluation criteria:
//!
//! 1. **Monotonicity**: for each (image_id, codec) curve sorted by `q`,
//!    count adjacent-q pairs where the bake's score(q+δ) ≤ score(q).
//!    Report the violation rate per bake.
//! 2. **Calibration linearity**: per [0,10), [10,20), …, [90,100] band on
//!    `q` (or `score_target` if provided), compute RMSE of bake_score vs
//!    a reference target (default: `q` mapped to [0..100] for JPEG).
//! 3. **Score spread**: per-q histogram of the bake's predictions (min,
//!    p5, median, p95, max). Helps diagnose dead zones (clamp ties at 0
//!    or 100).
//!
//! Each bake gets its own table. Output is markdown.
//!
//! Usage (CSV/manifest + bake — re-forwards a bake over feature rows):
//!   qsweep_eval --features qsweep_features.csv \
//!               --manifest qsweep_manifest.tsv \
//!               --bake tuner=path/to/tuner.bin \
//!               --bake compression=path/to/compression.bin \
//!               --bake balanced=path/to/balanced.bin \
//!               --out report.md
//!
//! The features CSV is the output of `extract_features_372col
//! --corpus qsweep`. The manifest TSV (same layout the extractor
//! consumed) provides `image_id`, `codec`, `q` for each row in input
//! order — the CSV's `human_score` column carries `q` already, but
//! the manifest gives the codec string explicitly.
//!
//! Usage (`--parquet` — score is ALREADY a column, no bake re-forward):
//!   qsweep_eval --parquet cells.a.parquet \
//!               --col-ref ref_filename --col-codec codec --col-q q \
//!               --score A=pred_a \
//!               --score B=pred_b@cells.b.parquet \
//!               --score ssim2=score_ssim2 \
//!               --tag zenjpeg_lossy \
//!               --summary-tsv summary.tsv --out report.md
//!
//! In `--parquet` mode the four grouping/score columns are read straight
//! from parquet — the score column is taken as-is (no Model forward). Each
//! `--score LABEL=COLUMN[@FILE]` becomes one report column (analogous to a
//! `--bake`); `@FILE` overrides the default `--parquet` file so A and B can
//! live in sibling parquets. `--col-score COLUMN` is shorthand for a single
//! `--score COLUMN=COLUMN` spec (matches the minimal 4-flag invocation).
//!
//! **Dial semantics.** Picker-sweep parquets carry MANY encoder-knob
//! variants per (ref, codec, q) cell (unique `knob_tuple_json` per row),
//! so "the score at q" is the per-cell aggregate across knobs (median by
//! default, `--agg mean|none`). Curves are then (ref, codec) → sorted-by-q
//! aggregated points, fed to the SAME [`compute_mono`] core as the bake
//! path. A curve with < 2 distinct q (e.g. a lossless codec, single q=0)
//! or zero rank variance (all scores equal, e.g. ssim2≡100) is DEGENERATE:
//! it is excluded from the rate and reported as "no rank variance" rather
//! than a divide-by-zero or a fake 100% monotonic.
//!
//! Direct-main commit, no PR — produces a tuner-trail eval that
//! complements bake_verdict's rank-trail Mohammadi panel.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array, StringArray};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use zenpredict::{Model, Predictor};

// DEDUP-M (2026-05-26): `score_row` + `extract_*` helpers moved to
// `zensim_validate::bake_runtime`. Bit-exact, f32 ±1e-6 on representative
// inputs (see benchmarks/dedup_M_score_row_evidence/).
use zensim_validate::bake_runtime::{
    extract_hybrid_head, extract_per_sample_alpha_head, extract_tanh_output_head_scale, score_row,
};

/// Selected input mode, parsed from the CLI.
///
/// - `Csv`: the historical path — forward one or more bakes over a
///   feature CSV + manifest TSV. Each bake spec is `(name, path,
///   post_mode)`, `post_mode` one of:
///   - "raw"        — emit the bake's raw output (no clamp).
///   - "clamp"      — clamp(raw, 0, 100).
///   - "mapped"     — score = 100 - 18 * max(raw,0)^0.7  (distance bakes).
///   - "mapped:A,B" — score = 100 - A * max(raw,0)^B.
///
///   (v0_3 / V_18 ship → "mapped"; V_22 / V_24 / tuner → "clamp".)
/// - `Parquet`: the score is an existing parquet column, read as-is.
enum Mode {
    Csv {
        features: PathBuf,
        manifest: PathBuf,
        bakes: Vec<(String, PathBuf, String)>,
        out: Option<PathBuf>,
    },
    Parquet {
        col_ref: String,
        col_codec: String,
        col_q: String,
        agg: Agg,
        specs: Vec<ScoreSpec>,
        tag: Option<String>,
        summary_tsv: Option<PathBuf>,
        out: Option<PathBuf>,
    },
}

fn parse_args() -> Mode {
    let mut args = std::env::args().skip(1);
    // Shared / CSV-mode state.
    let mut features: Option<PathBuf> = None;
    let mut manifest: Option<PathBuf> = None;
    let mut bakes: Vec<(String, PathBuf, String)> = Vec::new();
    let mut out: Option<PathBuf> = None;
    // Parquet-mode state.
    let mut parquet_default: Option<PathBuf> = None;
    let mut col_ref = "ref_filename".to_string();
    let mut col_codec = "codec".to_string();
    let mut col_q = "q".to_string();
    let mut agg = Agg::Median;
    let mut tag: Option<String> = None;
    let mut summary_tsv: Option<PathBuf> = None;
    // Deferred score specs (resolve default file after the full parse).
    let mut raw_specs: Vec<(String, String, Option<PathBuf>)> = Vec::new();
    let mut saw_parquet_flag = false;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--features" => features = Some(args.next().expect("--features VALUE").into()),
            "--manifest" => manifest = Some(args.next().expect("--manifest VALUE").into()),
            "--bake" => {
                // NAME=PATH[:MODE]
                let spec = args.next().expect("--bake NAME=PATH[:MODE]");
                let mut parts = spec.splitn(2, '=');
                let name = parts.next().expect("name").to_string();
                let rest = parts.next().expect("path[:mode]");
                // The path may contain `:`; allow `:mode` only if rest
                // ends in `:raw`, `:clamp`, or `:mapped[A,B]`.
                let (path_str, mode) = if let Some(idx) = rest.rfind(":mapped") {
                    (&rest[..idx], rest[idx + 1..].to_string())
                } else if let Some(idx) = rest.rfind(":clamp") {
                    (&rest[..idx], rest[idx + 1..].to_string())
                } else if let Some(idx) = rest.rfind(":raw") {
                    (&rest[..idx], rest[idx + 1..].to_string())
                } else {
                    // No explicit mode: default to "clamp" (matches V_22/V_24/tuner).
                    (rest, "clamp".to_string())
                };
                bakes.push((name, PathBuf::from(path_str), mode));
            }
            "--parquet" => {
                parquet_default = Some(args.next().expect("--parquet VALUE").into());
                saw_parquet_flag = true;
            }
            "--col-ref" => col_ref = args.next().expect("--col-ref VALUE"),
            "--col-codec" => col_codec = args.next().expect("--col-codec VALUE"),
            "--col-q" => col_q = args.next().expect("--col-q VALUE"),
            "--agg" => agg = Agg::parse(&args.next().expect("--agg VALUE")),
            "--tag" => tag = Some(args.next().expect("--tag VALUE")),
            "--summary-tsv" => summary_tsv = Some(args.next().expect("--summary-tsv VALUE").into()),
            "--col-score" => {
                // Shorthand: LABEL == COLUMN, default file.
                let col = args.next().expect("--col-score VALUE");
                raw_specs.push((col.clone(), col, None));
            }
            "--score" => {
                // LABEL=COLUMN[@FILE]
                let spec = args.next().expect("--score LABEL=COLUMN[@FILE]");
                let (label, rhs) = spec
                    .split_once('=')
                    .unwrap_or_else(|| panic!("--score expects LABEL=COLUMN[@FILE], got {spec:?}"));
                let (column, file) = match rhs.split_once('@') {
                    Some((c, f)) => (c.to_string(), Some(PathBuf::from(f))),
                    None => (rhs.to_string(), None),
                };
                raw_specs.push((label.to_string(), column, file));
            }
            "--out" => out = Some(args.next().expect("--out VALUE").into()),
            other => panic!("unknown arg: {other}"),
        }
    }

    // Parquet mode when a parquet file or any score spec was given.
    if saw_parquet_flag || !raw_specs.is_empty() {
        let specs: Vec<ScoreSpec> = raw_specs
            .into_iter()
            .map(|(label, column, file)| {
                let file = file.or_else(|| parquet_default.clone()).unwrap_or_else(|| {
                    panic!("score {label:?} has no file: pass --parquet FILE or LABEL=COL@FILE")
                });
                ScoreSpec {
                    label,
                    column,
                    file,
                }
            })
            .collect();
        assert!(
            !specs.is_empty(),
            "--parquet mode needs at least one --score / --col-score"
        );
        return Mode::Parquet {
            col_ref,
            col_codec,
            col_q,
            agg,
            specs,
            tag,
            summary_tsv,
            out,
        };
    }

    Mode::Csv {
        features: features.expect("--features REQUIRED (or use --parquet mode)"),
        manifest: manifest.expect("--manifest REQUIRED (or use --parquet mode)"),
        bakes,
        out,
    }
}

fn apply_post(raw: f64, mode: &str) -> f64 {
    if raw.is_nan() {
        return f64::NAN;
    }
    match mode {
        "raw" => raw,
        // EXP-CROSS-CODEC-V10 (2026-05-20): explicit no-clamp mode.
        "extrapolate" => raw,
        "clamp" => raw.clamp(0.0, 100.0),
        m if m.starts_with("mapped") => {
            // mapped or mapped:A,B
            let (a, b) = if let Some(rest) = m.strip_prefix("mapped:") {
                let mut it = rest.splitn(2, ',');
                let a: f64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(18.0);
                let b: f64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.7);
                (a, b)
            } else {
                (18.0, 0.7)
            };
            let d = raw.max(0.0);
            (100.0 - a * d.powf(b)).clamp(0.0, 100.0)
        }
        _ => raw.clamp(0.0, 100.0),
    }
}

fn load_features_csv(path: &PathBuf) -> (Vec<f64>, Vec<Vec<f64>>, Vec<String>) {
    let f = File::open(path).expect("open features CSV");
    let r = BufReader::new(f);
    let mut header: Option<Vec<String>> = None;
    let mut human_idx = 0;
    let mut f0_idx = 0;
    let mut ref_idx = 0;
    let mut human_scores = Vec::new();
    let mut feature_rows = Vec::new();
    let mut ref_basenames = Vec::new();
    for line in r.lines().map_while(|x| x.ok()) {
        if header.is_none() {
            let hdr: Vec<String> = line.split(',').map(|s| s.to_string()).collect();
            ref_idx = hdr
                .iter()
                .position(|c| c == "ref_basename")
                .expect("ref_basename");
            human_idx = hdr
                .iter()
                .position(|c| c == "human_score")
                .expect("human_score");
            f0_idx = hdr.iter().position(|c| c == "f0").expect("f0");
            header = Some(hdr);
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() <= f0_idx {
            continue;
        }
        let ref_b = cols[ref_idx].to_string();
        let h: f64 = cols[human_idx].parse().unwrap_or(f64::NAN);
        let feats: Vec<f64> = cols[f0_idx..]
            .iter()
            .map(|s| s.parse().unwrap_or(0.0))
            .collect();
        ref_basenames.push(ref_b);
        human_scores.push(h);
        feature_rows.push(feats);
    }
    (human_scores, feature_rows, ref_basenames)
}

fn load_manifest_tsv(path: &PathBuf) -> Vec<(String, String, f64)> {
    let f = File::open(path).expect("open manifest TSV");
    let r = BufReader::new(f);
    let mut rows = Vec::new();
    let mut first = true;
    for line in r.lines().map_while(|x| x.ok()) {
        if first {
            first = false;
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 5 {
            continue;
        }
        let image_id = cols[2].to_string();
        let codec = cols[3].to_string();
        let q: f64 = cols[4].parse().unwrap_or(f64::NAN);
        rows.push((image_id, codec, q));
    }
    rows
}

#[derive(Debug, Clone)]
struct BakeReport {
    name: String,
    monotonicity_rate: f64,
    tied_rate: f64,
    n_curves: usize,
    n_pairs: usize,
    n_violations: usize,
    n_ties: usize,
    /// Curves with < 2 distinct-q points (no dial axis, e.g. lossless q≡0).
    n_deg_single: usize,
    /// Curves with ≥ 2 points but zero rank variance (all scores equal).
    n_deg_flat: usize,
    score_per_q: BTreeMap<u32, Vec<f64>>, // q → all scores at that q across images
    band_rmse: Vec<(f64, f64, usize, f64)>, // (band_lo, band_hi, n, rmse(score vs target=q))
}

/// A set of dial curves: each `(image_id, codec)` maps to its `(q, score)`
/// points. The shared unit both input paths feed to [`compute_mono`].
type Curves = BTreeMap<(String, String), Vec<(f64, f64)>>;

/// Result of the shared dial-monotonicity core over a set of curves.
///
/// A curve is `Vec<(q, score)>` for one `(image_id, codec)` group. This
/// counts, per curve sorted by q, adjacent-q pairs that strictly decrease
/// (violation) or tie, and classifies degenerate curves separately so a
/// dial with no rank variance is never reported as a fake 100% monotonic.
#[derive(Debug, Clone, Default)]
struct MonoResult {
    n_curves: usize,     // curves that carry a real dial (≥2 q AND rank variance)
    n_pairs: usize,      // adjacent-q comparisons across those curves
    n_violations: usize, // strict score(q+δ) < score(q)
    n_ties: usize,       // score(q+δ) == score(q)
    n_deg_single: usize, // curves with < 2 distinct-q points
    n_deg_flat: usize,   // curves with ≥2 points but all scores equal
}

impl MonoResult {
    /// Monotonicity rate = 1 − violations/pairs. NaN when there are no
    /// pairs (every curve degenerate) — the caller renders that as
    /// "no rank variance", never a fake 100%.
    fn monotonicity_rate(&self) -> f64 {
        if self.n_pairs > 0 {
            1.0 - self.n_violations as f64 / self.n_pairs as f64
        } else {
            f64::NAN
        }
    }
    fn tied_rate(&self) -> f64 {
        if self.n_pairs > 0 {
            self.n_ties as f64 / self.n_pairs as f64
        } else {
            f64::NAN
        }
    }
}

/// The single dial-monotonicity core, shared by the bake path and the
/// `--parquet` path. Each curve is sorted by q; adjacent-q pairs are
/// classified as strict-decrease (violation), tie, or increase. Curves
/// with fewer than two distinct-q points, or with zero rank variance
/// (all scores equal), are counted as degenerate and excluded from the
/// rate — they carry no dial to be monotonic over.
fn compute_mono(curves: &Curves) -> MonoResult {
    let mut r = MonoResult::default();
    for v in curves.values() {
        let mut v = v.clone();
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        if v.len() < 2 {
            r.n_deg_single += 1;
            continue;
        }
        // Rank variance: does any score differ from the first? A flat
        // curve (e.g. ssim2≡100 on lossless) has none — degenerate.
        let s0 = v[0].1;
        if v.iter().all(|p| p.1 == s0) {
            r.n_deg_flat += 1;
            continue;
        }
        r.n_curves += 1;
        for w in v.windows(2) {
            r.n_pairs += 1;
            // q increases → score should strictly increase.
            // STRICT violation = decrease; TIE = equal (recorded separately).
            if w[1].1 < w[0].1 {
                r.n_violations += 1;
            } else if w[1].1 == w[0].1 {
                r.n_ties += 1;
            }
        }
    }
    r
}

/// Per-q spread histogram (q → all representative scores at that q).
fn score_per_q_hist(points: &[(f64, f64)]) -> BTreeMap<u32, Vec<f64>> {
    let mut m: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
    for (q, s) in points {
        if s.is_nan() {
            continue;
        }
        m.entry(q.round() as u32).or_default().push(*s);
    }
    m
}

/// Calibration RMSE of `score` vs `q` per [b·10, (b+1)·10) band, target=q.
fn band_rmse_vs_q(points: &[(f64, f64)]) -> Vec<(f64, f64, usize, f64)> {
    let mut band_rmse = Vec::new();
    for b in 0..10u32 {
        let lo = (b * 10) as f64;
        let hi = ((b + 1) * 10) as f64;
        let mut ss = 0.0;
        let mut n = 0usize;
        for (q, s) in points {
            if *q < lo || *q >= hi || s.is_nan() {
                continue;
            }
            ss += (s - q).powi(2);
            n += 1;
        }
        let rmse = if n > 0 {
            (ss / n as f64).sqrt()
        } else {
            f64::NAN
        };
        band_rmse.push((lo, hi, n, rmse));
    }
    band_rmse
}

fn evaluate_bake(
    name: &str,
    bake_path: &PathBuf,
    post_mode: &str,
    feature_rows: &[Vec<f64>],
    manifest: &[(String, String, f64)],
) -> BakeReport {
    let bake_bytes = std::fs::read(bake_path).expect("read bake");
    let model = Model::from_bytes(&bake_bytes).expect("parse ZNPR");
    let n_inputs = model.caller_input_width();
    let has_transforms = model.has_nontrivial_feature_transforms();
    let per_sample_alpha = extract_per_sample_alpha_head(&model);
    let hybrid = extract_hybrid_head(&model);
    let tanh_pin_scale = extract_tanh_output_head_scale(&model);
    let output_spline = zensim_validate::output_calibration_spline::extract(&model);
    eprintln!(
        "{name}: n_inputs={n_inputs} transforms={has_transforms} per-α={} hybrid={} tanh-pin={} spline={}",
        if per_sample_alpha.is_some() {
            "yes"
        } else {
            "no"
        },
        if hybrid.is_some() { "yes" } else { "no" },
        if let Some(s) = tanh_pin_scale {
            format!("scale={s:.3}")
        } else {
            "no".to_string()
        },
        if let Some(s) = &output_spline {
            format!("{}-knot", s.xs.len())
        } else {
            "no".to_string()
        }
    );

    let mut predictor = Predictor::new(&model);
    let mut buf = vec![0.0f32; n_inputs];
    let scores: Vec<f64> = feature_rows
        .iter()
        .map(|row| {
            let raw = score_row(
                &mut predictor,
                has_transforms,
                per_sample_alpha.as_ref(),
                hybrid.as_ref(),
                tanh_pin_scale,
                output_spline.as_ref(),
                &mut buf,
                row,
            );
            apply_post(raw, post_mode)
        })
        .collect();
    assert_eq!(
        scores.len(),
        manifest.len(),
        "feature rows / manifest length mismatch"
    );

    // Group by (image_id, codec); each group's (q, score). The bake path
    // has one row per (image_id, codec, q) already, so no per-cell
    // aggregation is needed — feed the raw curves to the shared core.
    let mut curves: Curves = BTreeMap::new();
    let mut points: Vec<(f64, f64)> = Vec::with_capacity(manifest.len());
    for (i, (image_id, codec, q)) in manifest.iter().enumerate() {
        let s = scores[i];
        if s.is_nan() {
            continue;
        }
        curves
            .entry((image_id.clone(), codec.clone()))
            .or_default()
            .push((*q, s));
        points.push((*q, s));
    }
    let mono = compute_mono(&curves);

    // Score-per-q histogram + calibration RMSE (target = q), from raw rows.
    let score_per_q = score_per_q_hist(&points);
    let band_rmse = band_rmse_vs_q(&points);

    BakeReport {
        name: name.to_string(),
        monotonicity_rate: mono.monotonicity_rate(),
        tied_rate: mono.tied_rate(),
        n_curves: mono.n_curves,
        n_pairs: mono.n_pairs,
        n_violations: mono.n_violations,
        n_ties: mono.n_ties,
        n_deg_single: mono.n_deg_single,
        n_deg_flat: mono.n_deg_flat,
        score_per_q,
        band_rmse,
    }
}

/// Render a rate as a 4-dp fraction, or `no-var` when NaN (every curve
/// degenerate → no dial → no rank variance to measure).
fn fmt_rate(x: f64) -> String {
    if x.is_nan() {
        "no-var".to_string()
    } else {
        format!("{x:.4}")
    }
}

// ----------------------------------------------------------------------
// `--parquet` mode: score is an existing column, no bake re-forward.
// ----------------------------------------------------------------------

/// Per-(ref, codec, q) aggregation of the many knob-variant rows into one
/// representative "score at q".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Agg {
    Median,
    Mean,
    None,
}

impl Agg {
    fn parse(s: &str) -> Agg {
        match s.to_ascii_lowercase().as_str() {
            "median" => Agg::Median,
            "mean" => Agg::Mean,
            "none" => Agg::None,
            other => panic!("unknown --agg '{other}' (expected median|mean|none)"),
        }
    }
    fn label(self) -> &'static str {
        match self {
            Agg::Median => "median",
            Agg::Mean => "mean",
            Agg::None => "none(first)",
        }
    }
    fn reduce(self, v: &mut [f64]) -> f64 {
        match self {
            Agg::Median => {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n = v.len();
                if n == 0 {
                    f64::NAN
                } else if n % 2 == 1 {
                    v[n / 2]
                } else {
                    0.5 * (v[n / 2 - 1] + v[n / 2])
                }
            }
            Agg::Mean => {
                if v.is_empty() {
                    f64::NAN
                } else {
                    v.iter().sum::<f64>() / v.len() as f64
                }
            }
            Agg::None => *v.first().unwrap_or(&f64::NAN),
        }
    }
}

/// One report column in `--parquet` mode: a label, the parquet column to
/// read as the score, and the file it lives in.
#[derive(Debug, Clone)]
struct ScoreSpec {
    label: String,
    column: String,
    file: PathBuf,
}

/// Read a `f64` from an arrow column regardless of f32/f64/i32/i64 storage.
fn col_as_f64(col: &dyn Array) -> Vec<f64> {
    let n = col.len();
    if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
        (0..n)
            .map(|i| if a.is_null(i) { f64::NAN } else { a.value(i) })
            .collect()
    } else if let Some(a) = col.as_any().downcast_ref::<Float32Array>() {
        (0..n)
            .map(|i| {
                if a.is_null(i) {
                    f64::NAN
                } else {
                    a.value(i) as f64
                }
            })
            .collect()
    } else if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        (0..n)
            .map(|i| {
                if a.is_null(i) {
                    f64::NAN
                } else {
                    a.value(i) as f64
                }
            })
            .collect()
    } else if let Some(a) = col.as_any().downcast_ref::<Int32Array>() {
        (0..n)
            .map(|i| {
                if a.is_null(i) {
                    f64::NAN
                } else {
                    a.value(i) as f64
                }
            })
            .collect()
    } else {
        vec![f64::NAN; n]
    }
}

/// Read a `String` from an arrow column (Utf8, or a numeric column stringified).
fn col_as_string(col: &dyn Array) -> Vec<String> {
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        (0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    String::new()
                } else {
                    a.value(i).to_string()
                }
            })
            .collect()
    } else {
        col_as_f64(col)
            .into_iter()
            .map(|v| format!("{v}"))
            .collect()
    }
}

/// Columns read from a single parquet file: the shared (ref, codec, q)
/// keys plus one score vector per requested score column.
struct ParquetCols {
    refs: Vec<String>,
    codecs: Vec<String>,
    qs: Vec<f64>,
    scores: BTreeMap<String, Vec<f64>>,
}

/// Read `col_ref`/`col_codec`/`col_q` and each of `score_cols` from a
/// parquet file, projecting only those columns (the files carry 500+
/// columns; we touch 4-6). All returned vectors are row-aligned.
fn read_parquet_cols(
    path: &Path,
    col_ref: &str,
    col_codec: &str,
    col_q: &str,
    score_cols: &[String],
) -> Result<ParquetCols, String> {
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("{}: parquet open: {e}", path.display()))?;
    let schema = builder.schema().clone();
    let parquet_schema = builder.parquet_schema().clone();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

    let find = |c: &str| -> Result<usize, String> {
        names
            .iter()
            .position(|&x| x == c)
            .ok_or_else(|| format!("{}: column {c:?} not found", path.display()))
    };
    let ref_idx = find(col_ref)?;
    let codec_idx = find(col_codec)?;
    let q_idx = find(col_q)?;
    // De-dup requested score columns while keeping insertion order.
    let mut uniq_scores: Vec<String> = Vec::new();
    for c in score_cols {
        if !uniq_scores.iter().any(|x| x == c) {
            uniq_scores.push(c.clone());
        }
    }
    let score_idx: Vec<usize> = uniq_scores
        .iter()
        .map(|c| find(c))
        .collect::<Result<_, _>>()?;

    // Projection over the arrow-flat leaf indices (feature parquets are
    // primitive-typed, so arrow col idx == parquet leaf idx).
    let mut wanted = vec![ref_idx, codec_idx, q_idx];
    wanted.extend_from_slice(&score_idx);
    let mask = ProjectionMask::leaves(&parquet_schema, wanted.iter().copied());
    let reader = builder
        .with_projection(mask)
        .with_batch_size(16384)
        .build()
        .map_err(|e| format!("{}: build reader: {e}", path.display()))?;

    // Projected batches are ordered by ascending original column index.
    let mut sorted = wanted.clone();
    sorted.sort_unstable();
    let pos = |orig: usize| sorted.iter().position(|&p| p == orig).unwrap();
    let p_ref = pos(ref_idx);
    let p_codec = pos(codec_idx);
    let p_q = pos(q_idx);
    let p_scores: Vec<usize> = score_idx.iter().map(|&i| pos(i)).collect();

    let mut refs = Vec::new();
    let mut codecs = Vec::new();
    let mut qs = Vec::new();
    let mut scores: Vec<Vec<f64>> = vec![Vec::new(); uniq_scores.len()];
    for batch_res in reader {
        let batch = batch_res.map_err(|e| format!("{}: read batch: {e}", path.display()))?;
        refs.extend(col_as_string(batch.column(p_ref)));
        codecs.extend(col_as_string(batch.column(p_codec)));
        qs.extend(col_as_f64(batch.column(p_q)));
        for (k, &pk) in p_scores.iter().enumerate() {
            scores[k].extend(col_as_f64(batch.column(pk)));
        }
    }
    let scores_map = uniq_scores.into_iter().zip(scores).collect();
    Ok(ParquetCols {
        refs,
        codecs,
        qs,
        scores: scores_map,
    })
}

/// Aggregate raw (ref, codec, q, score) rows into per-(ref, codec) curves:
/// one representative score per distinct q (median/mean over knob variants).
/// Returns the curves plus the flattened aggregated points (for the
/// histogram + band RMSE, which then describe the dial spread across
/// images, not the raw within-q knob spread).
fn build_curves_aggregated(
    refs: &[String],
    codecs: &[String],
    qs: &[f64],
    scores: &[f64],
    agg: Agg,
) -> (Curves, Vec<(f64, f64)>) {
    // (ref, codec, q-bits) → all scores in that cell.
    let mut cells: BTreeMap<(String, String, u64), Vec<f64>> = BTreeMap::new();
    let n = refs.len().min(codecs.len()).min(qs.len()).min(scores.len());
    for i in 0..n {
        let s = scores[i];
        let q = qs[i];
        if s.is_nan() || q.is_nan() {
            continue;
        }
        cells
            .entry((refs[i].clone(), codecs[i].clone(), q.to_bits()))
            .or_default()
            .push(s);
    }
    let mut curves: Curves = BTreeMap::new();
    let mut points: Vec<(f64, f64)> = Vec::with_capacity(cells.len());
    for ((r, c, qbits), mut v) in cells {
        let q = f64::from_bits(qbits);
        let rep = agg.reduce(&mut v);
        curves.entry((r, c)).or_default().push((q, rep));
        points.push((q, rep));
    }
    (curves, points)
}

/// Build a [`BakeReport`] for one score column read straight from parquet.
fn evaluate_parquet_score(
    label: &str,
    refs: &[String],
    codecs: &[String],
    qs: &[f64],
    scores: &[f64],
    agg: Agg,
) -> BakeReport {
    let (curves, points) = build_curves_aggregated(refs, codecs, qs, scores, agg);
    let mono = compute_mono(&curves);
    let score_per_q = score_per_q_hist(&points);
    let band_rmse = band_rmse_vs_q(&points);
    BakeReport {
        name: label.to_string(),
        monotonicity_rate: mono.monotonicity_rate(),
        tied_rate: mono.tied_rate(),
        n_curves: mono.n_curves,
        n_pairs: mono.n_pairs,
        n_violations: mono.n_violations,
        n_ties: mono.n_ties,
        n_deg_single: mono.n_deg_single,
        n_deg_flat: mono.n_deg_flat,
        score_per_q,
        band_rmse,
    }
}

/// Append one machine-readable summary row per report to a TSV (creating
/// it with a header when absent). `tag` labels the run (e.g. the codec).
fn append_summary_tsv(path: &Path, tag: &str, reports: &[BakeReport]) -> std::io::Result<()> {
    let write_header = !path.exists();
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    if write_header {
        writeln!(
            f,
            "tag\tlabel\tn_curves\tn_pairs\tn_violations\tn_ties\tn_deg_single\tn_deg_flat\tmono_rate\ttied_rate"
        )?;
    }
    for r in reports {
        writeln!(
            f,
            "{tag}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            r.name,
            r.n_curves,
            r.n_pairs,
            r.n_violations,
            r.n_ties,
            r.n_deg_single,
            r.n_deg_flat,
            fmt_rate(r.monotonicity_rate),
            fmt_rate(r.tied_rate),
        )?;
    }
    Ok(())
}

fn render_report(reports: &[BakeReport], n_total_pairs: usize) -> String {
    let mut s = String::new();
    s.push_str("# qsweep_eval — PreviewV0_5Tuner trail eval\n\n");
    s.push_str("Per-bake monotonicity (on JPEG q-sweep 50 imgs × 19 q values),\n");
    s.push_str("score-per-q histogram, calibration RMSE per [0,10), [10,20), … band on `q`.\n\n");
    s.push_str(&format!("- Total manifest rows: {}\n\n", n_total_pairs));

    s.push_str("## Monotonicity summary\n\n");
    s.push_str("Strict-decrease violation rate (lower = better). Ties (clamp-flat ");
    s.push_str("regions, often score=0 or score=100) are reported separately — they ");
    s.push_str("don't count as inversions but they ARE dead zones a user-facing dial ");
    s.push_str("can't binary-search through.\n\n");
    s.push_str("`degenerate` = curves with no dial to be monotonic over: ");
    s.push_str("`<2 distinct q` (e.g. a lossless codec at a single q) or `flat` ");
    s.push_str("(zero rank variance, e.g. ssim2≡100). These are EXCLUDED from the ");
    s.push_str("rate — an all-degenerate column reports `no rank variance`, never a ");
    s.push_str("fake 100%.\n\n");
    s.push_str("| Bake | n_curves | n_adj_pairs | strict_violations | tied | degenerate (single/flat) | monotonicity_rate | tied_rate |\n");
    s.push_str("|---|--:|--:|--:|--:|--:|---:|---:|\n");
    for r in reports {
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} ({}/{}) | {} | {} |\n",
            r.name,
            r.n_curves,
            r.n_pairs,
            r.n_violations,
            r.n_ties,
            r.n_deg_single + r.n_deg_flat,
            r.n_deg_single,
            r.n_deg_flat,
            fmt_rate(r.monotonicity_rate),
            fmt_rate(r.tied_rate),
        ));
    }
    s.push('\n');

    s.push_str("## Score-per-q histogram (median / p25 / p75)\n\n");
    s.push_str("Each row: q value → (median, p25, p75, min, max).\n\n");
    for r in reports {
        s.push_str(&format!("### {}\n\n", r.name));
        s.push_str("| q | n | min | p25 | median | p75 | max |\n");
        s.push_str("|--:|--:|---:|---:|---:|---:|---:|\n");
        for (q, v) in &r.score_per_q {
            let mut sorted = v.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = sorted.len();
            let p = |frac: f64| {
                let i = ((n as f64 * frac).round() as usize).min(n.saturating_sub(1));
                sorted[i]
            };
            s.push_str(&format!(
                "| {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |\n",
                q,
                n,
                sorted[0],
                p(0.25),
                p(0.5),
                p(0.75),
                sorted[n - 1]
            ));
        }
        s.push('\n');
    }

    s.push_str("## Calibration linearity (RMSE per band, target = q)\n\n");
    s.push_str("`score - q` RMSE per [b·10, (b+1)·10) band. Low RMSE = score tracks q linearly. ");
    s.push_str("NOTE: zensim does NOT have a constraint that score=q on JPEG; this RMSE is ");
    s.push_str("a proxy for cross-image consistency — a tuner with low RMSE per band gives ");
    s.push_str("the user a JPEG q-targeting tool whose zensim output is predictable.\n\n");
    s.push_str("| Band | range | n |");
    for r in reports {
        s.push_str(&format!(" {} |", r.name));
    }
    s.push_str("\n|---|---|--:|");
    for _ in reports {
        s.push_str("---:|");
    }
    s.push('\n');
    // All reports share the same 10 bands; iterate using the first report's band list.
    if let Some(first) = reports.first() {
        for b_idx in 0..first.band_rmse.len() {
            let (lo, hi, n, _) = first.band_rmse[b_idx];
            s.push_str(&format!(
                "| B{} | [{}, {}) | {} |",
                b_idx, lo as i32, hi as i32, n
            ));
            for r in reports {
                let v = r.band_rmse[b_idx].3;
                if v.is_nan() {
                    s.push_str(" n/a |");
                } else {
                    s.push_str(&format!(" {:.2} |", v));
                }
            }
            s.push('\n');
        }
    }
    s.push('\n');

    s
}

fn main() -> ExitCode {
    match parse_args() {
        Mode::Csv {
            features,
            manifest,
            bakes,
            out,
        } => run_csv(&features, &manifest, &bakes, out.as_deref()),
        Mode::Parquet {
            col_ref,
            col_codec,
            col_q,
            agg,
            specs,
            tag,
            summary_tsv,
            out,
        } => run_parquet(
            &col_ref,
            &col_codec,
            &col_q,
            agg,
            &specs,
            tag.as_deref(),
            summary_tsv.as_deref(),
            out.as_deref(),
        ),
    }
}

/// CSV/manifest + bake path (unchanged behavior — bakes are re-forwarded).
fn run_csv(
    features_path: &PathBuf,
    manifest_path: &PathBuf,
    bakes: &[(String, PathBuf, String)],
    out_path: Option<&Path>,
) -> ExitCode {
    eprintln!("loading features {}", features_path.display());
    let (_human, feature_rows, _refs) = load_features_csv(features_path);
    eprintln!("loaded {} feature rows", feature_rows.len());
    eprintln!("loading manifest {}", manifest_path.display());
    let manifest = load_manifest_tsv(manifest_path);
    eprintln!("loaded {} manifest rows", manifest.len());
    if feature_rows.len() != manifest.len() {
        eprintln!(
            "WARNING: feature_rows.len()={} != manifest.len()={}",
            feature_rows.len(),
            manifest.len()
        );
    }

    let mut reports = Vec::new();
    for (name, path, mode) in bakes {
        eprintln!(
            "evaluating bake '{name}' (mode={mode}) from {}",
            path.display()
        );
        let r = evaluate_bake(name, path, mode, &feature_rows, &manifest);
        eprintln!(
            "  {}: monotonicity={} ({}/{} curves)",
            r.name,
            fmt_rate(r.monotonicity_rate),
            r.n_curves,
            r.n_pairs
        );
        reports.push(r);
    }

    let s = render_report(&reports, manifest.len());
    emit(&s, out_path);
    ExitCode::SUCCESS
}

/// `--parquet` path: score is an existing column, no bake re-forward.
#[allow(clippy::too_many_arguments)]
fn run_parquet(
    col_ref: &str,
    col_codec: &str,
    col_q: &str,
    agg: Agg,
    specs: &[ScoreSpec],
    tag: Option<&str>,
    summary_tsv: Option<&Path>,
    out_path: Option<&Path>,
) -> ExitCode {
    eprintln!(
        "parquet dial-monotonicity: agg={} keys=({col_ref},{col_codec},{col_q}) tag={}",
        agg.label(),
        tag.unwrap_or("-")
    );
    // Group specs by file so each parquet is read once.
    let mut by_file: BTreeMap<PathBuf, Vec<usize>> = BTreeMap::new();
    for (i, s) in specs.iter().enumerate() {
        by_file.entry(s.file.clone()).or_default().push(i);
    }
    // Reports must come out in the order the specs were given.
    let mut reports: Vec<Option<BakeReport>> = vec![None; specs.len()];
    let mut total_rows = 0usize;
    for (file, spec_ids) in &by_file {
        let cols: Vec<String> = spec_ids.iter().map(|&i| specs[i].column.clone()).collect();
        eprintln!("reading {} (cols: {})", file.display(), cols.join(", "));
        let pc = match read_parquet_cols(file, col_ref, col_codec, col_q, &cols) {
            Ok(pc) => pc,
            Err(e) => {
                eprintln!("ERROR: {e}");
                return ExitCode::FAILURE;
            }
        };
        total_rows += pc.refs.len();
        for &i in spec_ids {
            let spec = &specs[i];
            let scores = pc
                .scores
                .get(&spec.column)
                .expect("score column present after read");
            let r = evaluate_parquet_score(&spec.label, &pc.refs, &pc.codecs, &pc.qs, scores, agg);
            eprintln!(
                "  {} [{}]: mono={} tied={} ({} dial curves, {} degenerate)",
                spec.label,
                spec.column,
                fmt_rate(r.monotonicity_rate),
                fmt_rate(r.tied_rate),
                r.n_curves,
                r.n_deg_single + r.n_deg_flat,
            );
            reports[i] = Some(r);
        }
    }
    let reports: Vec<BakeReport> = reports.into_iter().map(|r| r.unwrap()).collect();

    if let Some(tsv) = summary_tsv {
        let run_tag = tag.unwrap_or("run");
        if let Err(e) = append_summary_tsv(tsv, run_tag, &reports) {
            eprintln!(
                "WARNING: failed to append summary TSV {}: {e}",
                tsv.display()
            );
        } else {
            eprintln!("appended {} rows to {}", reports.len(), tsv.display());
        }
    }

    let s = render_report(&reports, total_rows);
    emit(&s, out_path);
    ExitCode::SUCCESS
}

/// Write the rendered markdown to `out_path` or stdout.
fn emit(s: &str, out_path: Option<&Path>) {
    if let Some(out) = out_path {
        let mut f = File::create(out).expect("create out");
        f.write_all(s.as_bytes()).expect("write");
        eprintln!("wrote {}", out.display());
    } else {
        print!("{s}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_curve(pairs: &[(f64, f64)]) -> Curves {
        let mut m = BTreeMap::new();
        m.insert(("img".to_string(), "codec".to_string()), pairs.to_vec());
        m
    }

    #[test]
    fn strictly_increasing_curve_has_zero_violations() {
        let c = one_curve(&[(5.0, 10.0), (30.0, 20.0), (50.0, 40.0), (90.0, 80.0)]);
        let m = compute_mono(&c);
        assert_eq!(m.n_curves, 1);
        assert_eq!(m.n_pairs, 3);
        assert_eq!(m.n_violations, 0);
        assert_eq!(m.n_ties, 0);
        assert_eq!(m.n_deg_single, 0);
        assert_eq!(m.n_deg_flat, 0);
        assert_eq!(m.monotonicity_rate(), 1.0);
        assert_eq!(m.tied_rate(), 0.0);
    }

    #[test]
    fn out_of_order_input_still_zero_violations_after_sort() {
        // Same curve, rows shuffled — compute_mono sorts by q internally.
        let c = one_curve(&[(90.0, 80.0), (5.0, 10.0), (50.0, 40.0), (30.0, 20.0)]);
        let m = compute_mono(&c);
        assert_eq!(m.n_violations, 0);
        assert_eq!(m.n_pairs, 3);
    }

    #[test]
    fn one_violation_in_four_step_curve() {
        // 20 -> 15 is a strict decrease.
        let c = one_curve(&[(5.0, 10.0), (30.0, 20.0), (50.0, 15.0), (90.0, 40.0)]);
        let m = compute_mono(&c);
        assert_eq!(m.n_pairs, 3);
        assert_eq!(m.n_violations, 1);
        assert!((m.monotonicity_rate() - (1.0 - 1.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn flat_curve_is_degenerate_not_fake_100() {
        // Constant score across distinct q: zero rank variance. Must be
        // reported as "no-var", NEVER a fake 100% monotonic.
        let c = one_curve(&[(5.0, 100.0), (30.0, 100.0), (90.0, 100.0)]);
        let m = compute_mono(&c);
        assert_eq!(m.n_curves, 0);
        assert_eq!(m.n_pairs, 0);
        assert_eq!(m.n_deg_flat, 1);
        assert_eq!(m.n_deg_single, 0);
        assert!(
            m.monotonicity_rate().is_nan(),
            "flat curve must be no-var, not 100%"
        );
        assert_eq!(fmt_rate(m.monotonicity_rate()), "no-var");
    }

    #[test]
    fn single_q_curve_is_degenerate_single() {
        // One distinct q (e.g. a lossless codec at q=0) -> no dial axis.
        let c = one_curve(&[(0.0, 50.0)]);
        let m = compute_mono(&c);
        assert_eq!(m.n_deg_single, 1);
        assert_eq!(m.n_curves, 0);
        assert!(m.monotonicity_rate().is_nan());
        assert!(m.tied_rate().is_nan());
    }

    #[test]
    fn median_aggregation_collapses_knobs_per_q() {
        // Two knob variants per q at each of 3 q; median over knobs, then mono.
        let refs = vec!["a".to_string(); 6];
        let codecs = vec!["c".to_string(); 6];
        let qs = vec![5.0, 5.0, 30.0, 30.0, 90.0, 90.0];
        // per-q medians: q5 -> 11, q30 -> 22, q90 -> 42 (strictly increasing)
        let scores = vec![10.0, 12.0, 20.0, 24.0, 40.0, 44.0];
        let (curves, points) = build_curves_aggregated(&refs, &codecs, &qs, &scores, Agg::Median);
        assert_eq!(curves.len(), 1);
        let v = &curves[&("a".to_string(), "c".to_string())];
        assert_eq!(v.len(), 3, "3 distinct q after knob collapse");
        assert_eq!(points.len(), 3);
        let m = compute_mono(&curves);
        assert_eq!(m.n_curves, 1);
        assert_eq!(m.n_pairs, 2);
        assert_eq!(m.n_violations, 0);
        assert_eq!(m.n_ties, 0);
    }

    #[test]
    fn synthetic_parquet_roundtrip_dial() {
        use arrow::array::{Float64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        // 1 ref, 1 codec, 3 distinct q, 2 knob rows each. Increasing dial.
        let refs = vec!["r0"; 6];
        let codecs = vec!["cx"; 6];
        let qv = vec![5.0, 5.0, 30.0, 30.0, 90.0, 90.0];
        let sv = vec![10.0, 12.0, 20.0, 25.0, 40.0, 45.0];
        let schema = Arc::new(Schema::new(vec![
            Field::new("ref_filename", DataType::Utf8, false),
            Field::new("codec", DataType::Utf8, false),
            Field::new("q", DataType::Float64, false),
            Field::new("score_x", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(refs)),
                Arc::new(StringArray::from(codecs)),
                Arc::new(Float64Array::from(qv)),
                Arc::new(Float64Array::from(sv)),
            ],
        )
        .unwrap();
        let path = std::env::temp_dir().join(format!(
            "qsweep_test_{}_{}.parquet",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        {
            let f = File::create(&path).unwrap();
            let mut w = ArrowWriter::try_new(f, schema, None).unwrap();
            w.write(&batch).unwrap();
            w.close().unwrap();
        }
        let pc = read_parquet_cols(
            &path,
            "ref_filename",
            "codec",
            "q",
            &["score_x".to_string()],
        )
        .unwrap();
        assert_eq!(pc.refs.len(), 6);
        let scores = &pc.scores["score_x"];
        let r = evaluate_parquet_score("A", &pc.refs, &pc.codecs, &pc.qs, scores, Agg::Median);
        assert_eq!(r.n_curves, 1);
        assert_eq!(r.n_pairs, 2);
        assert_eq!(r.n_violations, 0);
        assert_eq!(r.monotonicity_rate, 1.0);
        assert_eq!(r.n_deg_single, 0);
        assert_eq!(r.n_deg_flat, 0);
        std::fs::remove_file(&path).ok();
    }
}
