// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! v2-feature extractor for the trainability A/B (docs/V2_TRAINABILITY_AB_2026-07-19.md).
//!
//! Reads a pairs TSV (`ref_path`, `dist_path`, `human_score`, extra columns ignored)
//! and writes the FULL 264-feature v2 vector per pair as CSV
//! (`ref_basename,human_score,f0..f263`) — the same shape
//! `extract_features_372col --corpus pairs-tsv` emits for the v1 arm, so both
//! arms feed `zensim_mlp_train` identically (CSV is auto-detected by extension).
//!
//! ```sh
//! cargo run --release -p zensim --features feature-regime-v2,threads \
//!   --example v2_ab_extract -- pairs.tsv out_v2.csv
//! ```

#[path = "support/zen_io.rs"]
mod zen_io;

use std::path::PathBuf;

use rayon::prelude::*;
use zensim::{RgbSlice, Zensim, ZensimProfile};

struct Pair {
    ref_path: PathBuf,
    dist_path: PathBuf,
    human_score: f64,
}

fn load_pairs_tsv(path: &str) -> Vec<Pair> {
    let text = std::fs::read_to_string(path).expect("read pairs tsv");
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let idx = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("pairs tsv missing column {name:?}"))
    };
    let (ri, di, hi) = (idx("ref_path"), idx("dist_path"), idx("human_score"));
    lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            Pair {
                ref_path: PathBuf::from(c[ri]),
                dist_path: PathBuf::from(c[di]),
                human_score: c[hi].parse().expect("human_score"),
            }
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: v2_ab_extract <pairs.tsv> <out.csv>");
        std::process::exit(2);
    }
    let pairs = load_pairs_tsv(&args[0]);
    eprintln!("{} pairs from {}", pairs.len(), args[0]);

    let mut n_feat_seen = std::sync::atomic::AtomicUsize::new(0);
    let rows: Vec<String> = pairs
        .par_iter()
        .enumerate()
        .filter_map(|(i, p)| {
            if i % 1000 == 0 {
                eprintln!("progress: {i}/{}", pairs.len());
            }
            if !p.ref_path.exists() || !p.dist_path.exists() {
                eprintln!("SKIP missing: {:?} / {:?}", p.ref_path, p.dist_path);
                return None;
            }
            let (r_px, rw, rh) = zen_io::decode_rgb8(&p.ref_path);
            let (d_px, dw, dh) = zen_io::decode_rgb8(&p.dist_path);
            if (rw, rh) != (dw, dh) {
                eprintln!("SKIP dim mismatch: {:?}", p.dist_path);
                return None;
            }
            let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
            let result = match z
                .compute_v2_features(&RgbSlice::new(&r_px, rw, rh), &RgbSlice::new(&d_px, dw, dh))
            {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP compute error {:?}: {e:?}", p.dist_path);
                    return None;
                }
            };
            let f = result.features();
            n_feat_seen.store(f.len(), std::sync::atomic::Ordering::Relaxed);
            let base = p
                .ref_path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let mut row = format!("{base},{}", p.human_score);
            for v in f {
                row.push(',');
                row.push_str(&format!("{v}"));
            }
            Some(row)
        })
        .collect();

    let n_feat = *n_feat_seen.get_mut();
    let mut out = String::from("ref_basename,human_score");
    for k in 0..n_feat {
        out.push_str(&format!(",f{k}"));
    }
    out.push('\n');
    for r in &rows {
        out.push_str(r);
        out.push('\n');
    }
    std::fs::write(&args[1], out).expect("write out csv");
    eprintln!("wrote {} rows x {n_feat} features to {}", rows.len(), args[1]);
}
