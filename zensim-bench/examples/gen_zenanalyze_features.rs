//! Generate a per-reference-image zenanalyze feature TSV for V0_6 training.
//!
//! Walks one or more dataset CSVs, dedups by reference image stem
//! (file basename without extension — what zensim-validate uses as
//! `ref_keys` for cache lookups), decodes each reference PNG/JPG once,
//! runs zenanalyze with the requested feature set, and writes one row
//! per unique stem: `stem<TAB>feat1<TAB>feat2<TAB>...`.
//!
//! Output is a stable sidecar that `zensim-validate` consumes via
//! `--mlp-zenanalyze-tsv`. Computing analyze features is per-reference
//! (not per-pair), so caching pays for itself: synthetic-v2 has ~3,587
//! unique references vs 218k pairs.
//!
//! Source-path columns supported (auto-detected): `source_path` (synthetic
//! CSVs), `ref_path`, `reference`. For directory-based datasets (KADID,
//! TID, KonJND), pre-build a CSV mapping or pass a CSV-style index.
//!
//! Usage:
//!   gen_zenanalyze_features \
//!     --csv /mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv \
//!     --features Variance,EdgeDensity,LumaHistogramEntropy,ChromaComplexity \
//!     --output /mnt/v/output/zensim/synthetic-v2/zenanalyze_features.tsv
//!
//! Multiple CSVs (writes a unioned TSV; latest stem wins on collision):
//!   gen_zenanalyze_features --csv a.csv --csv b.csv --features ... --output union.tsv

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

fn parse_features(spec: &str) -> Vec<AnalysisFeature> {
    let mut out = Vec::new();
    for raw in spec.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        // Match by both PascalCase variant name (Variance) and snake_case
        // wire name (variance). Linear scan over SUPPORTED is O(n) but
        // n=~70 and this is one-shot startup.
        let mut found = None;
        for f in FeatureSet::SUPPORTED.iter() {
            let pascal = format!("{:?}", f);
            if pascal.eq_ignore_ascii_case(raw) || f.name().eq_ignore_ascii_case(raw) {
                found = Some(f);
                break;
            }
        }
        match found {
            Some(f) => out.push(f),
            None => {
                eprintln!("error: unknown zenanalyze feature: {raw}");
                std::process::exit(2);
            }
        }
    }
    out
}

/// (stem, full_path) pairs, deduped by stem.
fn collect_unique_sources(csv_paths: &[PathBuf]) -> Vec<(String, String)> {
    let mut by_stem: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    for csv_path in csv_paths {
        let f = File::open(csv_path).unwrap_or_else(|e| {
            eprintln!("could not open CSV {}: {e}", csv_path.display());
            std::process::exit(1);
        });
        let mut rdr = BufReader::new(f);
        let mut header = String::new();
        rdr.read_line(&mut header).expect("read header");
        let cols: Vec<&str> = header.trim().split(',').collect();
        let src_idx = cols
            .iter()
            .position(|c| *c == "source_path" || *c == "ref_path" || *c == "reference")
            .unwrap_or(0);
        for line in rdr.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            let cols: Vec<&str> = line.split(',').collect();
            if let Some(p) = cols.get(src_idx) {
                let trimmed = p.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let stem = Path::new(trimmed)
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();
                if stem.is_empty() {
                    continue;
                }
                by_stem.entry(stem).or_insert_with(|| trimmed.to_string());
            }
        }
    }
    by_stem.into_iter().collect()
}

/// Walk a directory of reference images, returning (stem, full_path).
fn collect_dir_sources(dir: &Path) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("could not open dir {}: {e}", dir.display());
            return out;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase());
        match ext.as_deref() {
            Some("png" | "jpg" | "jpeg" | "bmp" | "tiff" | "tif" | "webp") => {}
            _ => continue,
        }
        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        out.push((stem, path.to_string_lossy().to_string()));
    }
    out.sort();
    out
}

fn decode_rgb8(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    Some((rgb.into_raw(), w, h))
}

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    args.remove(0);

    let mut csv_paths: Vec<PathBuf> = Vec::new();
    let mut dir_paths: Vec<PathBuf> = Vec::new();
    let mut features_spec: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut threads: usize = 0;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--csv" => {
                csv_paths.push(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--dir" => {
                dir_paths.push(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--features" => {
                features_spec = Some(args[i + 1].clone());
                i += 2;
            }
            "--output" => {
                output = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--threads" => {
                threads = args[i + 1].parse().unwrap_or(0);
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }

    if csv_paths.is_empty() && dir_paths.is_empty() {
        eprintln!("error: at least one --csv or --dir required");
        std::process::exit(2);
    }
    let features_spec = features_spec.expect("--features required");
    let output = output.expect("--output required");

    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok();
    }

    let features = parse_features(&features_spec);
    let mut needed = FeatureSet::new();
    for &f in &features {
        needed = needed.with(f);
    }
    let query = AnalysisQuery::new(needed);

    eprintln!(
        "csvs={} dirs={} features={} output={}",
        csv_paths.len(),
        dir_paths.len(),
        features
            .iter()
            .map(|f| format!("{:?}", f))
            .collect::<Vec<_>>()
            .join(","),
        output.display()
    );

    // Union sources from CSVs and directories. Deduplicate by stem.
    let mut by_stem: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    if !csv_paths.is_empty() {
        for (s, p) in collect_unique_sources(&csv_paths) {
            by_stem.entry(s).or_insert(p);
        }
    }
    for d in &dir_paths {
        for (s, p) in collect_dir_sources(d) {
            by_stem.entry(s).or_insert(p);
        }
    }
    let sources: Vec<(String, String)> = by_stem.into_iter().collect();
    eprintln!("found {} unique reference stems", sources.len());

    let _ = BTreeSet::<()>::new(); // keep BTreeSet import live for refactors

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let out = File::create(&output).expect("create output tsv");
    let out = Mutex::new(BufWriter::new(out));

    // Header row.
    {
        let mut w = out.lock().unwrap();
        write!(w, "stem\tsource_path").unwrap();
        for f in &features {
            write!(w, "\t{}", f.name()).unwrap();
        }
        writeln!(w).unwrap();
    }

    let total = sources.len();
    let done = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let started = std::time::Instant::now();
    let progress_every = (total / 50).max(50);

    sources.par_iter().for_each(|(stem, src_path)| {
        let path = Path::new(src_path);
        let res = decode_rgb8(path);
        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        if n % progress_every == 0 || n == total {
            let elapsed = started.elapsed().as_secs_f32();
            let rate = n as f32 / elapsed.max(1e-3);
            let eta = (total - n) as f32 / rate.max(1e-3);
            eprintln!(
                "  [{:>5.0}s] {}/{} ({:.1}/s, ETA {:.0}s)",
                elapsed, n, total, rate, eta
            );
        }
        let (rgb, w, h) = match res {
            Some(t) => t,
            None => {
                failed.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };
        let r = zenanalyze::analyze_features_rgb8(&rgb, w, h, &query);
        let mut line = String::new();
        line.push_str(stem);
        line.push('\t');
        line.push_str(src_path);
        for f in &features {
            line.push('\t');
            match r.get_f32(*f) {
                Some(v) => line.push_str(&format!("{v}")),
                None => line.push_str("NaN"),
            }
        }
        line.push('\n');
        let mut wlk = out.lock().unwrap();
        wlk.write_all(line.as_bytes()).expect("tsv write");
    });

    let elapsed = started.elapsed().as_secs_f32();
    let f = failed.load(Ordering::Relaxed);
    eprintln!(
        "done {}/{} (failed {}) in {:.1}s",
        total - f,
        total,
        f,
        elapsed
    );

    out.into_inner().unwrap().flush().expect("tsv flush");
}
