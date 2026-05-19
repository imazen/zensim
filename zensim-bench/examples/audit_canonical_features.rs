//! Extract first 100 features for a TSV list of (ref_path, dist_path) pairs.
//!
//! Used by /home/lilith/work/zen/_ml-inventory-2026-05-20/10-canonical-build-audit.md to
//! re-extract features against current zensim main and byte-compare against the
//! canonical-2026-05-21 parquet's stored f0..f99 columns.
//!
//! Mode: extended_features=true + compute_iw_features=true (372-feature mode).
//! We emit only f0..f99 since 100 features is sufficient to detect build drift.
//!
//! Input TSV (with header `ref_path<TAB>dist_path`):
//!     /path/to/ref.png<TAB>/path/to/dist.png
//!
//! Output (stdout JSON-lines):
//!     {"idx":0,"ref":"...","dist":"...","feats":[...100 floats...]}
//!     {"idx":1,"ref":"...","dist":"...","feats":[...]}
//!     ...
//!
//! Errors HARD on any decode/extraction failure — never silently skip.

use std::path::PathBuf;

use rayon::prelude::*;
use zensim::{ZensimConfig, compute_zensim_with_config};

fn main() {
    let mut args = std::env::args().skip(1);
    let mut pairs_tsv: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--pairs" => pairs_tsv = Some(args.next().unwrap().into()),
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let pairs_tsv = pairs_tsv.expect("--pairs REQUIRED");

    let body = std::fs::read_to_string(&pairs_tsv).expect("read pairs tsv");
    let mut lines = body.lines();
    let header = lines.next().unwrap_or("");
    if !header.starts_with("ref_path\tdist_path") {
        eprintln!("expected header `ref_path\\tdist_path...`, got: {header:?}");
        std::process::exit(2);
    }
    let pairs: Vec<(usize, String, String)> = lines
        .enumerate()
        .filter_map(|(i, line)| {
            let mut cols = line.split('\t');
            let r = cols.next()?.to_string();
            let d = cols.next()?.to_string();
            Some((i, r, d))
        })
        .collect();
    eprintln!("loaded {} pairs from {}", pairs.len(), pairs_tsv.display());

    let results: Vec<(usize, String, String, Vec<f64>)> = pairs
        .par_iter()
        .map(|(i, ref_path, dist_path)| {
            let feats = extract(ref_path, dist_path).unwrap_or_else(|e| {
                eprintln!(
                    "HARD FAIL idx={i} ref={ref_path} dist={dist_path}: {e}"
                );
                std::process::exit(3);
            });
            (*i, ref_path.clone(), dist_path.clone(), feats)
        })
        .collect();

    let mut sorted = results;
    sorted.sort_by_key(|(i, ..)| *i);
    for (i, r, d, feats) in &sorted {
        let feats100: Vec<String> = feats.iter().take(100).map(|v| format!("{v:.10e}")).collect();
        println!(
            "{{\"idx\":{i},\"ref\":\"{r}\",\"dist\":\"{d}\",\"feats\":[{}]}}",
            feats100.join(",")
        );
    }
}

fn extract(ref_path: &str, dist_path: &str) -> Result<Vec<f64>, String> {
    let src_img = image::ImageReader::open(ref_path)
        .map_err(|e| format!("open ref: {e}"))?
        .with_guessed_format()
        .map_err(|e| format!("guess ref format: {e}"))?
        .decode()
        .map_err(|e| format!("decode ref: {e}"))?
        .to_rgb8();
    let dst_img = image::ImageReader::open(dist_path)
        .map_err(|e| format!("open dist: {e}"))?
        .with_guessed_format()
        .map_err(|e| format!("guess dist format: {e}"))?
        .decode()
        .map_err(|e| format!("decode dist: {e}"))?
        .to_rgb8();
    let (w, h) = src_img.dimensions();
    let (dw, dh) = dst_img.dimensions();
    if w != dw || h != dh {
        return Err(format!(
            "dim mismatch: ref={w}x{h} dist={dw}x{dh}"
        ));
    }
    if w < 8 || h < 8 {
        return Err(format!("image too small: {w}x{h}"));
    }
    let src_pixels: Vec<[u8; 3]> = src_img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let dst_pixels: Vec<[u8; 3]> = dst_img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let mut config = ZensimConfig::default();
    config.extended_features = true;
    config.compute_iw_features = true;
    let result =
        compute_zensim_with_config(&src_pixels, &dst_pixels, w as usize, h as usize, config)
            .map_err(|e| format!("compute_zensim_with_config: {e:?}"))?;
    Ok(result.features().to_vec())
}
