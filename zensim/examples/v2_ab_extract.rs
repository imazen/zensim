// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! APPEND-ONLY extended feature extractor for the v2 backfill
//! (docs/V2_TRAINABILITY_AB_2026-07-19.md; feature-numbering directive
//! 2026-07-19: new v2 features occupy indices AFTER all v1 features).
//!
//! Reads a pairs TSV (`ref_path`, `dist_path`, `human_score`, extra columns
//! ignored) and writes the EXTENDED vector per pair as CSV
//! (`ref_basename,human_score,f0..f719`): the FROZEN v1-372 block
//! (`compute_zensim_with_config`, extended+iw) at f0..f371, THEN the v2-348
//! block (`compute_v2_features`) relabeled at f372..f719. Both blocks are
//! computed on the SAME zen_io-decoded pixels in one pass — no join, no key,
//! no ordering risk (the two-file join hit an unrecoverable collision:
//! `(ref,human_score)` is not unique for kadid/aic3).
//!
//! Slice the output: v1-only = f0..f371, v2-only = f372..f719, and any
//! deprecated feature is MASKED (its column zeroed) rather than dropped —
//! indices stay stable per the append-only directive.
//!
//! NOTE: the v1-372 block here is zen_io-decoded (zenpng/zenjpeg/zenbitmaps),
//! so it may differ sub-ULP from the canonical image-crate v1 parquets. That
//! is irrelevant to this experiment: the v1 and v2 blocks are on IDENTICAL
//! pixels, which is what the append-only comparison needs.
//!
//! ```sh
//! cargo run --release -p zensim --features feature-regime-v2,threads \
//!   --example v2_ab_extract -- pairs.tsv ext_out.csv
//! ```

#[path = "support/zen_io.rs"]
mod zen_io;

use std::path::PathBuf;

use rayon::prelude::*;
use zensim::{RgbSlice, Zensim, ZensimConfig, ZensimProfile, compute_zensim_with_config};

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

    // ZENSIM_AB_MODE: "ext" (default, v1-372 ++ v2-348 = 720) | "v1" (372 only)
    // | "v2" (348 only). For the clean 3-way timing bench — same binary, same
    // decode path, only the compute set changes.
    let mode = std::env::var("ZENSIM_AB_MODE").unwrap_or_else(|_| "ext".into());
    let (do_v1, do_v2) = match mode.as_str() {
        "v1" => (true, false),
        "v2" => (false, true),
        "none" => (false, false), // decode-only, for timing decomposition
        _ => (true, true),
    };

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
            let mut combined: Vec<f64> = Vec::new();
            // FROZEN v1-372 block (extended + iw = 372 features), same config
            // extract_features_372col uses.
            if do_v1 {
                let mut cfg = ZensimConfig::default();
                cfg.extended_features = true;
                cfg.compute_iw_features = true;
                let v1 = match compute_zensim_with_config(&r_px, &d_px, rw, rh, cfg) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("SKIP v1 compute error {:?}: {e:?}", p.dist_path);
                        return None;
                    }
                };
                combined.extend_from_slice(v1.features());
            }
            // v2-348 block, same pixels.
            if do_v2 {
                let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
                let v2 = match z
                    .compute_v2_features(&RgbSlice::new(&r_px, rw, rh), &RgbSlice::new(&d_px, dw, dh))
                {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("SKIP v2 compute error {:?}: {e:?}", p.dist_path);
                        return None;
                    }
                };
                combined.extend_from_slice(v2.features());
            }
            n_feat_seen.store(combined.len(), std::sync::atomic::Ordering::Relaxed);
            let base = p
                .ref_path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let mut row = format!("{base},{}", p.human_score);
            for v in &combined {
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
