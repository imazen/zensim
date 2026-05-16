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
        _ => {
            eprintln!("--corpus must be one of: konjnd, konjnd_full, aic3 (got {corpus:?})");
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

    let scored: Vec<Option<(String, f64, Vec<f64>)>> = pairs
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

    let mut rows: Vec<(String, f64, Vec<f64>)> = scored.into_iter().flatten().collect();
    eprintln!(
        "scored {}/{} pairs in {:.1}s",
        rows.len(),
        n_total,
        started.elapsed().as_secs_f64()
    );

    let n_feat = rows.first().map(|r| r.2.len()).unwrap_or(0);
    if n_feat != 372 {
        eprintln!(
            "WARNING: expected 372 features per pair, got {n_feat} — \
             check ZensimConfig extended/iw flags + image dimensions"
        );
    }

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
    for i in 0..n_feat {
        write!(w, ",f{i}").unwrap();
    }
    writeln!(w).unwrap();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    for (ref_name, human, feats) in &rows {
        write!(w, "{ref_name},{human}").unwrap();
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

fn extract_features(kp: &Pair) -> Option<(String, f64, Vec<f64>)> {
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
    Some((kp.ref_basename.clone(), kp.human_score, features))
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
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}
