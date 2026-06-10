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
//! Usage:
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
//! Direct-main commit, no PR — produces a tuner-trail eval that
//! complements bake_verdict's rank-trail Mohammadi panel.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use zenpredict::{Model, Predictor};

// DEDUP-M (2026-05-26): `score_row` + `extract_*` helpers moved to
// `zensim_validate::bake_runtime`. Bit-exact, f32 ±1e-6 on representative
// inputs (see benchmarks/dedup_M_score_row_evidence/).
use zensim_validate::bake_runtime::{
    extract_hybrid_head, extract_per_sample_alpha_head, extract_tanh_output_head_scale, score_row,
};

/// `(features, manifest, bakes, out)` CLI args, where each bake is a
/// per-bake spec `(name, path, post_mode)`. `post_mode` is one of:
///   - "raw"        — emit the bake's raw output (no clamp).
///   - "clamp"      — clamp(raw, 0, 100).
///   - "mapped"     — score = 100 - 18 * max(raw,0)^0.7  (distance bakes).
///   - "mapped:A,B" — score = 100 - A * max(raw,0)^B.
///
/// The choice depends on the bake's runtime profile:
///   - v0_3 / V_18 ship: "mapped" (raw is distance, runtime maps it).
///   - V_22 + V_24 + tuner: "clamp" (raw IS score, `skip_score_mapping=true`).
type ParsedArgs = (
    PathBuf,
    PathBuf,
    Vec<(String, PathBuf, String)>,
    Option<PathBuf>,
);

fn parse_args() -> ParsedArgs {
    let mut args = std::env::args().skip(1);
    let mut features: Option<PathBuf> = None;
    let mut manifest: Option<PathBuf> = None;
    let mut bakes: Vec<(String, PathBuf, String)> = Vec::new();
    let mut out: Option<PathBuf> = None;
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
            "--out" => out = Some(args.next().expect("--out VALUE").into()),
            other => panic!("unknown arg: {other}"),
        }
    }
    (
        features.expect("--features REQUIRED"),
        manifest.expect("--manifest REQUIRED"),
        bakes,
        out,
    )
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
    score_per_q: BTreeMap<u32, Vec<f64>>, // q → all scores at that q across images
    band_rmse: Vec<(f64, f64, usize, f64)>, // (band_lo, band_hi, n, rmse(score vs target=q))
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
    let n_inputs = model.n_inputs();
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

    // Group by (image_id, codec); each group sorted by q.
    let mut curves: BTreeMap<(String, String), Vec<(f64, f64)>> = BTreeMap::new();
    for (i, (image_id, codec, q)) in manifest.iter().enumerate() {
        let s = scores[i];
        if s.is_nan() {
            continue;
        }
        curves
            .entry((image_id.clone(), codec.clone()))
            .or_default()
            .push((*q, s));
    }
    let mut n_curves = 0usize;
    let mut n_pairs = 0usize;
    let mut n_violations = 0usize; // strict score(q+δ) < score(q)
    let mut n_ties = 0usize; // score(q+δ) == score(q) (often clamp artifacts)
    for v in curves.values_mut() {
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        if v.len() < 2 {
            continue;
        }
        n_curves += 1;
        for w in v.windows(2) {
            n_pairs += 1;
            // q increases → score should strictly increase.
            // STRICT violation = decrease; TIE = equal (recorded separately).
            if w[1].1 < w[0].1 {
                n_violations += 1;
            } else if w[1].1 == w[0].1 {
                n_ties += 1;
            }
        }
    }
    let monotonicity_rate = if n_pairs > 0 {
        // Rate counts strict decreases ONLY as violations; ties are
        // reported separately so we can see clamp-induced flatlines.
        1.0 - n_violations as f64 / n_pairs as f64
    } else {
        f64::NAN
    };
    let tied_rate = if n_pairs > 0 {
        n_ties as f64 / n_pairs as f64
    } else {
        f64::NAN
    };

    // Score-per-q histogram.
    let mut score_per_q: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
    for (i, (_image_id, _codec, q)) in manifest.iter().enumerate() {
        let s = scores[i];
        if s.is_nan() {
            continue;
        }
        let qk = q.round() as u32;
        score_per_q.entry(qk).or_default().push(s);
    }

    // Calibration linearity: target = q (we want score≈q for JPEG since
    // higher q → higher zensim). Per band [b*10, (b+1)*10), compute RMSE
    // of score vs q. Bands 0..10 (B0..B9) on q.
    let mut band_rmse: Vec<(f64, f64, usize, f64)> = Vec::new();
    for b in 0..10u32 {
        let lo = (b * 10) as f64;
        let hi = ((b + 1) * 10) as f64;
        let mut ss = 0.0;
        let mut n = 0usize;
        for (i, (_id, _c, q)) in manifest.iter().enumerate() {
            if *q < lo || *q >= hi {
                continue;
            }
            let s = scores[i];
            if s.is_nan() {
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

    BakeReport {
        name: name.to_string(),
        monotonicity_rate,
        tied_rate,
        n_curves,
        n_pairs,
        n_violations,
        n_ties,
        score_per_q,
        band_rmse,
    }
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
    s.push_str("| Bake | n_curves | n_adj_pairs | strict_violations | tied | monotonicity_rate | tied_rate |\n");
    s.push_str("|---|--:|--:|--:|--:|---:|---:|\n");
    for r in reports {
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {:.4} | {:.4} |\n",
            r.name,
            r.n_curves,
            r.n_pairs,
            r.n_violations,
            r.n_ties,
            r.monotonicity_rate,
            r.tied_rate
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
    let (features_path, manifest_path, bakes, out_path) = parse_args();
    eprintln!("loading features {}", features_path.display());
    let (_human, feature_rows, _refs) = load_features_csv(&features_path);
    eprintln!("loaded {} feature rows", feature_rows.len());
    eprintln!("loading manifest {}", manifest_path.display());
    let manifest = load_manifest_tsv(&manifest_path);
    eprintln!("loaded {} manifest rows", manifest.len());
    if feature_rows.len() != manifest.len() {
        eprintln!(
            "WARNING: feature_rows.len()={} != manifest.len()={}",
            feature_rows.len(),
            manifest.len()
        );
    }

    let mut reports = Vec::new();
    for (name, path, mode) in &bakes {
        eprintln!(
            "evaluating bake '{name}' (mode={mode}) from {}",
            path.display()
        );
        let r = evaluate_bake(name, path, mode, &feature_rows, &manifest);
        eprintln!(
            "  {}: monotonicity={:.4} ({}/{} curves)",
            r.name, r.monotonicity_rate, r.n_curves, r.n_pairs
        );
        reports.push(r);
    }

    let s = render_report(&reports, manifest.len());
    if let Some(out) = out_path {
        let mut f = File::create(&out).expect("create out");
        f.write_all(s.as_bytes()).expect("write");
        eprintln!("wrote {}", out.display());
    } else {
        print!("{s}");
    }
    ExitCode::SUCCESS
}
