//! Extract 372-column zensim feature CSVs for arbitrary corpora.
//!
//! Used by the 2026-05-15 full-feature sweep that re-creates the training
//! ingest for V_20a IW / V_20b distortion-manifold bakes. The historical
//! 372-col CSV used to train V_20a was lost; this binary rebuilds it
//! deterministically against current source so downstream trainers can
//! consume a single canonical schema.
//!
//! Output schema (matches `/mnt/v/zen/zensim-training/2026-05-14-clean/`):
//!   ref_basename, human_score, f0, f1, ..., f371
//!
//! Features are computed via `compute_zensim_with_config` with
//! `extended_features = true` and `compute_iw_features = true` so the
//! emitted vector is 4 scales × 3 channels × 31 features per channel
//! = 372 columns (basic + peaks + masked + IW).
//!
//! Usage:
//!   cargo run --release -p zensim-bench --example extract_features_372col -- \
//!     --corpus konjnd \
//!     --path /mnt/v/datasets/KonJND-1k/KonJND-1k \
//!     --out  /mnt/v/zen/zensim-training/2026-05-15-full-features/konjnd_features_372col_2026-05-15.csv
//!
//!   cargo run --release -p zensim-bench --example extract_features_372col -- \
//!     --corpus aic3 \
//!     --path /mnt/v/dataset/aic3_ctc_epfl/decoded/info.csv \
//!     --out  /mnt/v/zen/zensim-training/2026-05-15-full-features/aic3_features_372col_2026-05-15.csv

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use zensim::{ZensimConfig, compute_zensim_with_config};

#[derive(Debug, Clone)]
struct Pair {
    reference: PathBuf,
    distorted: PathBuf,
    human_score: f64,
    ref_basename: String,
    /// Optional named extra target columns emitted alongside
    /// `human_score` (e.g., IW-SSIM on safesyn). Each entry becomes a
    /// CSV column between `human_score` and `f0`. The trainer's
    /// `--target-column NAME` flag (T1.1) selects which column is the
    /// regression target.
    extra_targets: Vec<(String, f64)>,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut corpus: Option<String> = None;
    let mut path: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut max_pairs: usize = usize::MAX;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--corpus" => corpus = Some(args.next().unwrap()),
            "--path" => path = Some(args.next().unwrap().into()),
            "--out" => out = Some(args.next().unwrap().into()),
            "--max-pairs" => max_pairs = args.next().unwrap().parse().unwrap(),
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let corpus = corpus.expect("--corpus REQUIRED (konjnd or aic3)");
    let path = path.expect("--path REQUIRED");
    let out = out.expect("--out REQUIRED");

    let pairs: Vec<Pair> = match corpus.as_str() {
        "konjnd" => load_konjnd(&path, max_pairs),
        "konjnd_full" => load_konjnd_full(&path, max_pairs),
        "aic3" => load_aic3(&path, max_pairs),
        "safesyn" => load_safesyn(&path, max_pairs),
        "qsweep" => load_qsweep_tsv(&path, max_pairs),
        _ => {
            eprintln!(
                "--corpus must be one of: konjnd, konjnd_full, aic3, safesyn, qsweep (got {corpus:?})"
            );
            std::process::exit(2);
        }
    };

    let n_total = pairs.len();
    eprintln!("Loaded {n_total} pairs from {corpus}");
    if n_total == 0 {
        eprintln!("no pairs loaded; exiting");
        std::process::exit(3);
    }

    let started = std::time::Instant::now();
    let progress = AtomicUsize::new(0);
    let log_every = (n_total / 20).max(1);

    let scored: Vec<Option<(String, f64, Vec<(String, f64)>, Vec<f64>)>> = pairs
        .par_iter()
        .map(|kp| {
            let p = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if p.is_multiple_of(log_every) {
                let elapsed = started.elapsed().as_secs_f64();
                let rate = p as f64 / elapsed;
                let eta = (n_total - p) as f64 / rate;
                eprintln!("  {corpus} {p}/{n_total} ({rate:.1}/s, ETA {eta:.0}s)");
            }
            extract_features(kp)
        })
        .collect();

    let mut rows: Vec<(String, f64, Vec<(String, f64)>, Vec<f64>)> =
        scored.into_iter().flatten().collect();
    eprintln!(
        "scored {}/{} pairs in {:.1}s",
        rows.len(),
        n_total,
        started.elapsed().as_secs_f64()
    );

    let n_feat = rows.first().map(|r| r.3.len()).unwrap_or(0);
    if n_feat != 372 {
        eprintln!(
            "WARNING: expected 372 features per pair, got {n_feat} — \
             check ZensimConfig extended/iw flags + image dimensions"
        );
    }

    // Header layout: ref_basename, human_score, <extra-target columns…>, f0..f<n-1>.
    // Extra target column names come from the first row's `extra_targets`; every row
    // is asserted to carry the same set in the same order (loader contract).
    let extra_names: Vec<String> = rows
        .first()
        .map(|r| r.2.iter().map(|(n, _)| n.clone()).collect())
        .unwrap_or_default();

    // Write CSV
    use std::io::{BufWriter, Write};
    if let Some(parent) = out.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).expect("create output dir");
        }
    }
    let f = std::fs::File::create(&out).expect("create output CSV");
    let mut w = BufWriter::with_capacity(1 << 20, f);
    write!(w, "ref_basename,human_score").unwrap();
    for name in &extra_names {
        write!(w, ",{name}").unwrap();
    }
    for i in 0..n_feat {
        write!(w, ",f{i}").unwrap();
    }
    writeln!(w).unwrap();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    for (ref_name, human, extras, feats) in &rows {
        write!(w, "{ref_name},{human}").unwrap();
        for (_, v) in extras {
            write!(w, ",{v}").unwrap();
        }
        for v in feats {
            write!(w, ",{v}").unwrap();
        }
        writeln!(w).unwrap();
    }
    w.flush().unwrap();
    eprintln!(
        "Wrote {} rows × {n_feat} features to {}",
        rows.len(),
        out.display()
    );
}

fn extract_features(kp: &Pair) -> Option<(String, f64, Vec<(String, f64)>, Vec<f64>)> {
    let src_img = image::open(&kp.reference).ok()?.to_rgb8();
    let dst_img = image::open(&kp.distorted).ok()?.to_rgb8();
    let (w, h) = src_img.dimensions();
    let (dw, dh) = dst_img.dimensions();
    if w != dw || h != dh {
        return None;
    }
    let src_pixels: Vec<[u8; 3]> = src_img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let dst_pixels: Vec<[u8; 3]> = dst_img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let w_us = w as usize;
    let h_us = h as usize;
    if w_us < 8 || h_us < 8 {
        return None;
    }
    let mut config = ZensimConfig::default();
    config.extended_features = true;
    config.compute_iw_features = true;
    let result = compute_zensim_with_config(&src_pixels, &dst_pixels, w_us, h_us, config).ok()?;
    let features: Vec<f64> = result.features().to_vec();
    Some((
        kp.ref_basename.clone(),
        kp.human_score,
        kp.extra_targets.clone(),
        features,
    ))
}

fn load_konjnd(base: &Path, max: usize) -> Vec<Pair> {
    let csv_path = base.join("subjective_ratings.csv");
    let mut rdr = match csv::Reader::from_path(&csv_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to open {}: {e}", csv_path.display());
            return Vec::new();
        }
    };
    let mut pairs = Vec::new();
    for record in rdr.records().flatten() {
        if record.len() < 5 {
            continue;
        }
        let image_id = record.get(0).unwrap_or("");
        let comp = record.get(1).unwrap_or("");
        let mean_threshold: f64 = match record.get(3).unwrap_or("").parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let stem = image_id.trim_end_matches(".png");
        if stem.is_empty() {
            continue;
        }
        let level = mean_threshold.round().clamp(1.0, 100.0) as u32;
        let (subdir, ext) = match comp {
            "JPEG" => ("jpeg", "jpg"),
            "BPG" => ("bpg", "png"),
            _ => continue,
        };
        let dist_name = format!("{stem}_{comp}_{level:03}.{ext}");
        let ref_path = base.join("source_image").join(image_id);
        let dist_path = base.join(subdir).join(&dist_name);
        if !dist_path.exists() {
            continue;
        }
        pairs.push(Pair {
            reference: ref_path,
            distorted: dist_path,
            human_score: mean_threshold,
            ref_basename: image_id.to_string(),
            extra_targets: Vec::new(),
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}

/// Full KonJND-1k loader: reads `konjnd_full_scored.csv` which lists
/// every (source × codec × quality) variant (~76k pairs) along with the
/// metric scores. We emit one row per pair, using `gpu_ssimulacra2 / 100`
/// as the `human_score` anchor — matching the convention used in the
/// existing `/mnt/v/zen/zensim-training/2026-05-14-clean/konjnd_aligned_features.csv`.
/// The score is NOT a real human MOS for these pairs; the canonical
/// 1008-source human-PJND anchors live in `subjective_ratings.csv`
/// (loaded by the `konjnd` corpus type).
fn load_konjnd_full(csv_path: &Path, max: usize) -> Vec<Pair> {
    let mut rdr = match csv::Reader::from_path(csv_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to open {}: {e}", csv_path.display());
            return Vec::new();
        }
    };
    let mut pairs = Vec::new();
    for record in rdr.records().flatten() {
        if record.len() < 5 {
            continue;
        }
        let src_path = record.get(0).unwrap_or("");
        let dist_path = record.get(1).unwrap_or("");
        // gpu_ssimulacra2 / 100 → 0..1 score anchor
        let score_norm: f64 = match record.get(4).and_then(|s| s.parse::<f64>().ok()) {
            Some(v) => v / 100.0,
            None => continue,
        };
        if src_path.is_empty() || dist_path.is_empty() {
            continue;
        }
        let ref_pb = PathBuf::from(src_path);
        let basename = ref_pb
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        pairs.push(Pair {
            reference: ref_pb,
            distorted: PathBuf::from(dist_path),
            human_score: score_norm,
            ref_basename: basename,
            extra_targets: Vec::new(),
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}

/// AIC-3 CTC dataset loader. The info.csv columns are:
///   score.jnd, codec, img.number, img.name, quality, quality.selected, method
/// Reference image lives at `<dataset_root>/original/<img.name>.png` and
/// each distorted file at `<dataset_root>/decoded/<img.name>/<codec>_<img.name>_<quality>.png`.
/// `--path` should be the info.csv path; the dataset root is its grandparent.
fn load_aic3(csv_path: &Path, max: usize) -> Vec<Pair> {
    let mut rdr = match csv::Reader::from_path(csv_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to open {}: {e}", csv_path.display());
            return Vec::new();
        }
    };
    // info.csv lives at <root>/decoded/info.csv → grandparent of csv = <root>
    let root: PathBuf = csv_path
        .parent()
        .and_then(|d| d.parent())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let original_dir = root.join("original");
    let decoded_dir = root.join("decoded");
    let mut pairs = Vec::new();
    for record in rdr.records().flatten() {
        if record.len() < 5 {
            continue;
        }
        let score_jnd: f64 = match record.get(0).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let codec = record.get(1).unwrap_or("");
        let img_name = record.get(3).unwrap_or("");
        let quality = record.get(4).unwrap_or("");
        if codec.is_empty() || img_name.is_empty() || quality.is_empty() {
            continue;
        }
        let ref_path = original_dir.join(format!("{img_name}.png"));
        let dist_name = format!("{codec}_{img_name}_{quality}.png");
        let dist_path = decoded_dir.join(img_name).join(&dist_name);
        if !ref_path.exists() || !dist_path.exists() {
            continue;
        }
        pairs.push(Pair {
            reference: ref_path,
            distorted: dist_path,
            human_score: score_jnd,
            ref_basename: format!("{img_name}.png"),
            extra_targets: Vec::new(),
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}

/// Safesyn (safe-synthetic) loader with IW-SSIM target column.
///
/// Reads the enriched safesyn TSV at
/// `/mnt/v/zen/zensim-training/<date>/safesyn_with_iwssim.csv`
/// (produced by `scripts/v_next/merge_iwssim_into_safesyn.py`).
/// Schema: `source_path, decoded_path, codec, quality, width, height,
/// gpu_ssimulacra2, gpu_butteraugli, cpu_ssimulacra2, cpu_butteraugli,
/// size_bytes, run_id, dssim, iwssim`.
///
/// Emits one Pair per row with:
/// - `reference = source_path`
/// - `distorted = decoded_path`
/// - `ref_basename = source_path file stem`
/// - `human_score = cpu_ssimulacra2 / 100` (legacy ssim2 target in [0, 1])
/// - `extra_targets = [("iwssim", iwssim_value)]` — emitted in the
///   output CSV as a column between `human_score` and `f0`. The
///   trainer's `--target-column iwssim` flag (T1.1) selects this
///   column as the regression target instead of `human_score`.
///
/// This is the V_22-IW training-data input. The 196 086-pair safesyn
/// corpus is the only large-N corpus that carries an IW-SSIM target
/// (computed via `scripts/v_next/compute_iwssim_on_safesyn.py`).
fn load_safesyn(csv_path: &Path, max: usize) -> Vec<Pair> {
    let mut rdr = match csv::Reader::from_path(csv_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to open {}: {e}", csv_path.display());
            return Vec::new();
        }
    };
    // Header positions are stable (this file is generated, not
    // hand-edited), but look them up by name to be robust to future
    // column reorderings.
    let header = rdr
        .headers()
        .map(|h| h.iter().map(String::from).collect::<Vec<_>>())
        .unwrap_or_default();
    let pos = |name: &str| -> Option<usize> { header.iter().position(|c| c == name) };
    let src_col = match pos("source_path") {
        Some(i) => i,
        None => {
            eprintln!("{}: missing source_path column", csv_path.display());
            return Vec::new();
        }
    };
    let dst_col = match pos("decoded_path") {
        Some(i) => i,
        None => {
            eprintln!("{}: missing decoded_path column", csv_path.display());
            return Vec::new();
        }
    };
    let cpu_ssim2_col = match pos("cpu_ssimulacra2") {
        Some(i) => i,
        None => {
            eprintln!("{}: missing cpu_ssimulacra2 column", csv_path.display());
            return Vec::new();
        }
    };
    // Half the safesyn rows have empty cpu_ssimulacra2 (zenavif / zenjxl
    // codec families were scored GPU-only). Both metrics are in
    // score_zensim units, so use cpu when present, fall back to gpu.
    let gpu_ssim2_col = match pos("gpu_ssimulacra2") {
        Some(i) => i,
        None => {
            eprintln!("{}: missing gpu_ssimulacra2 column", csv_path.display());
            return Vec::new();
        }
    };
    let iwssim_col = match pos("iwssim") {
        Some(i) => i,
        None => {
            eprintln!("{}: missing iwssim column", csv_path.display());
            return Vec::new();
        }
    };
    let mut pairs = Vec::new();
    for record in rdr.records().flatten() {
        let src_path = record.get(src_col).unwrap_or("");
        let dst_path = record.get(dst_col).unwrap_or("");
        if src_path.is_empty() || dst_path.is_empty() {
            continue;
        }
        let ssim2_raw: f64 = match record
            .get(cpu_ssim2_col)
            .and_then(|s| s.parse::<f64>().ok())
            .or_else(|| {
                record
                    .get(gpu_ssim2_col)
                    .and_then(|s| s.parse::<f64>().ok())
            }) {
            Some(v) => v,
            None => continue,
        };
        let iwssim: f64 = match record.get(iwssim_col).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let ref_pb = PathBuf::from(src_path);
        let basename = ref_pb
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        // Legacy convention: `human_score` is the ssim2 target divided
        // by 100 so it lands in [0, 1]. The trainer multiplies by 100
        // (default --target-scale) to recover score_zensim units.
        pairs.push(Pair {
            reference: ref_pb,
            distorted: PathBuf::from(dst_path),
            human_score: ssim2_raw / 100.0,
            ref_basename: basename,
            extra_targets: vec![("iwssim".to_string(), iwssim)],
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}

/// Generic q-sweep TSV loader for the `PreviewV0_5Tuner` evaluation
/// harness (2026-05-18). Reads a TSV with the columns:
///
/// ```text
///   ref_path  dist_path  image_id  codec  q
/// ```
///
/// where the first row is the header. `image_id` becomes
/// `ref_basename` (the field the trainer's downstream tooling
/// groups on); `q` is loaded into `human_score` (so the eval can
/// pivot by quality at scoring time without re-parsing); `codec`
/// becomes an extra target column. Monotonicity is measured
/// downstream by sorting per (`ref_basename`, `codec`) by `human_score`
/// (= q) and counting score(q+δ) ≤ score(q) inversions.
fn load_qsweep_tsv(path: &Path, max: usize) -> Vec<Pair> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    let f = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("failed to open {}: {e}", path.display());
            return Vec::new();
        }
    };
    let r = BufReader::new(f);
    let mut lines = r.lines();
    let _header = match lines.next() {
        Some(Ok(h)) => h,
        _ => return Vec::new(),
    };
    let mut pairs = Vec::new();
    for line in lines.flatten() {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 5 {
            continue;
        }
        let ref_path = cols[0];
        let dist_path = cols[1];
        let image_id = cols[2];
        let codec = cols[3];
        let q: f64 = match cols[4].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        pairs.push(Pair {
            reference: PathBuf::from(ref_path),
            distorted: PathBuf::from(dist_path),
            human_score: q,
            ref_basename: image_id.to_string(),
            extra_targets: vec![(
                "codec".to_string(),
                codec
                    .bytes()
                    .fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64))
                    as f64,
            )],
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}
