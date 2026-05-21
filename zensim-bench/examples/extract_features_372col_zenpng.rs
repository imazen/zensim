//! Extract 372-column zensim features for zenpng cells by RE-ENCODING
//! each (ref, q, knob) on the fly. zenpng is near-lossless; encoded files
//! are not preserved on disk in the canonical sweep store, so this binary
//! reproduces them deterministically from the same EncodeConfig used by
//! `zen-metrics-cli/src/sweep/encode.rs::encode_png` (knobs:
//! `compression` ∈ {balanced, high, thorough}, `near_lossless_bits` ∈ 0..=3).
//!
//! Input TSV columns:
//!   ref_path  dist_marker  unique_id  human_score  compression  near_lossless_bits
//!
//! `dist_marker` is unused (we re-encode in memory); kept for column
//! alignment with the broader pairs.tsv schema.
//!
//! Output: CSV matching `extract_features_372col` schema — ref_basename
//! (= unique_id), human_score, f0..f371.
//!
//! Usage:
//!   cargo run --release -p zensim-bench --features training \
//!     --example extract_features_372col_zenpng -- \
//!     --path /tmp/large_pairs_zenpng.tsv \
//!     --out  /tmp/large_372feat_zenpng.csv

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use enough::Unstoppable;
use imgref::ImgRef;
use rayon::prelude::*;
use zenpng::{Compression, EncodeConfig};
use zensim::{ZensimConfig, compute_zensim_with_config};

#[derive(Debug, Clone)]
struct PngPair {
    reference: PathBuf,
    unique_id: String,
    human_score: f64,
    compression: String,
    near_lossless_bits: u8,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut path: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut max_pairs: usize = usize::MAX;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--path" => path = Some(args.next().unwrap().into()),
            "--out" => out = Some(args.next().unwrap().into()),
            "--max-pairs" => max_pairs = args.next().unwrap().parse().unwrap(),
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let path = path.expect("--path REQUIRED");
    let out = out.expect("--out REQUIRED");

    let pairs = load_zenpng_tsv(&path, max_pairs);
    let n_total = pairs.len();
    eprintln!("Loaded {n_total} zenpng pairs");
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
                eprintln!("  zenpng {p}/{n_total} ({rate:.1}/s, ETA {eta:.0}s)");
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
        eprintln!("WARNING: expected 372 features, got {n_feat}");
    }

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

fn extract_features(kp: &PngPair) -> Option<(String, f64, Vec<f64>)> {
    let src_img = image::open(&kp.reference).ok()?.to_rgb8();
    let (w, h) = src_img.dimensions();
    let w_us = w as usize;
    let h_us = h as usize;
    if w_us < 8 || h_us < 8 {
        return None;
    }

    // Build zenpng EncodeConfig matching the sweep knobs
    let comp = match kp.compression.as_str() {
        "balanced" => Compression::Balanced,
        "high" => Compression::High,
        "thorough" => Compression::Thorough,
        other => {
            eprintln!("ERR unknown compression preset: {other} for {}", kp.unique_id);
            return None;
        }
    };
    let cfg = EncodeConfig::default()
        .with_compression(comp)
        .with_near_lossless_bits(kp.near_lossless_bits);

    // Encode
    let src_pixels_rgb: Vec<rgb::Rgb<u8>> =
        src_img.pixels().map(|p| rgb::Rgb { r: p.0[0], g: p.0[1], b: p.0[2] }).collect();
    let img_ref = ImgRef::new(&src_pixels_rgb, w_us, h_us);
    let encoded = zenpng::encode_rgb8(img_ref, None, &cfg, &Unstoppable, &Unstoppable);
    let bytes = match encoded {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ERR zenpng encode failed for {}: {e}", kp.unique_id);
            return None;
        }
    };

    // Decode the PNG back to RGB8
    let dst_img = match image::load_from_memory(&bytes) {
        Ok(im) => im.to_rgb8(),
        Err(e) => {
            eprintln!("ERR decode failed for {}: {e}", kp.unique_id);
            return None;
        }
    };
    let (dw, dh) = dst_img.dimensions();
    if dw != w || dh != h {
        eprintln!("ERR dim mismatch for {}: src={}x{} dst={}x{}", kp.unique_id, w, h, dw, dh);
        return None;
    }
    let src_pixels: Vec<[u8; 3]> = src_img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let dst_pixels: Vec<[u8; 3]> = dst_img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let mut config = ZensimConfig::default();
    config.extended_features = true;
    config.compute_iw_features = true;
    let result = compute_zensim_with_config(&src_pixels, &dst_pixels, w_us, h_us, config).ok()?;
    let features: Vec<f64> = result.features().to_vec();
    Some((kp.unique_id.clone(), kp.human_score, features))
}

fn load_zenpng_tsv(path: &Path, max: usize) -> Vec<PngPair> {
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
    let _header = lines.next();
    let mut pairs = Vec::new();
    for line in lines.flatten() {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 6 {
            continue;
        }
        let ref_path = cols[0];
        let unique_id = cols[2];
        let human_score: f64 = match cols[3].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let compression = cols[4].to_string();
        let near_lossless_bits: u8 = match cols[5].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        pairs.push(PngPair {
            reference: PathBuf::from(ref_path),
            unique_id: unique_id.to_string(),
            human_score,
            compression,
            near_lossless_bits,
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}
