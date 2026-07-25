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
//! Ref-reuse pass: pairs are GROUPED BY REFERENCE (default) — each group
//! decodes its reference once and prepares the v2 reference pyramid once
//! (`Zensim::prepare_v2_reference`), so per-pair work is distorted-side
//! only. Output rows keep the input TSV's order and are byte-identical to
//! the ungrouped path (`ZENSIM_AB_GROUPED=0`, the pre-reuse flow kept for
//! A/B timing).
//!
//! ```sh
//! cargo run --release -p zensim --features feature-regime-v2,threads \
//!   --example v2_ab_extract -- pairs.tsv ext_out.csv
//! ```

#[path = "support/zen_io.rs"]
mod zen_io;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use zensim::feature_v2::{V2NewFeatureToggles, V2Scratch};
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
    // Fold-investigation modes (2026-07-24): "v1e" = v1 extended-only (300,
    // IW skipped) | "v1s" = v1 standard (228, masked+IW skipped). Same v1
    // code path, only the config flags change — isolates the activity-path
    // (masked/IW pool) share of v1c for the fold-basic-156-into-v2 design.
    // "fold" = the folded-720 ONE-pass extraction (v1 basic in the v2 walk,
    // f156..371 = 0, v2-348) — `Zensim::compute_folded720_features*`.
    let mode = std::env::var("ZENSIM_AB_MODE").unwrap_or_else(|_| "ext".into());
    let do_fold = mode == "fold";
    let (do_v1, do_v2) = match mode.as_str() {
        "v1" | "v1e" | "v1s" => (true, false),
        "v2" => (false, true),
        "none" | "fold" => (false, false), // fold has its own branch below
        _ => (true, true),
    };

    // ZENSIM_AB_GROUPED: "1"/unset (default) = ref-grouped extraction with
    // prepared-reference reuse; "0" = the original per-pair flow (each pair
    // re-decodes its reference and rebuilds both ref pyramids) — kept for
    // A/B timing of the reuse win itself. Both produce byte-identical CSVs.
    let grouped = std::env::var("ZENSIM_AB_GROUPED")
        .map(|v| v != "0")
        .unwrap_or(true);

    let n_feat_seen = AtomicUsize::new(0);
    let n_done = AtomicUsize::new(0);
    let progress = |k: usize| {
        if k % 1000 == 0 {
            eprintln!("progress: {k}/{}", pairs.len());
        }
    };

    // Per-pair body shared by both flows: decode the distorted image, run
    // the requested blocks, and render the CSV row. `prepared`/`scratch`
    // are Some only on the grouped path.
    let score_pair = |p: &Pair,
                      r_px: &[[u8; 3]],
                      rw: usize,
                      rh: usize,
                      prepared: Option<&zensim::feature_v2::V2PreparedReference>,
                      scratch: &mut V2Scratch|
     -> Option<String> {
        if !p.dist_path.exists() {
            eprintln!("SKIP missing: {:?} / {:?}", p.ref_path, p.dist_path);
            return None;
        }
        let (d_px, dw, dh) = zen_io::decode_rgb8(&p.dist_path);
        if (rw, rh) != (dw, dh) {
            eprintln!("SKIP dim mismatch: {:?}", p.dist_path);
            return None;
        }
        let mut combined: Vec<f64> = Vec::new();
        // FROZEN v1-372 block (extended + iw = 372 features), same config
        // extract_features_372col uses. Inner multithreading OFF — the
        // outer rayon loop already saturates cores (thread count never
        // changes v1's output bytes, only speed).
        if do_v1 {
            let mut cfg = ZensimConfig::default();
            cfg.extended_features = mode != "v1s";
            cfg.compute_iw_features = mode != "v1s" && mode != "v1e";
            cfg.allow_multithreading = false;
            let v1 = match compute_zensim_with_config(r_px, &d_px, rw, rh, cfg) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP v1 compute error {:?}: {e:?}", p.dist_path);
                    return None;
                }
            };
            combined.extend_from_slice(v1.features());
        }
        // v2-348 block, same pixels — through the prepared reference when
        // the grouped path supplies one (bit-identical either way).
        if do_v2 {
            let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
            let distorted = RgbSlice::new(&d_px, dw, dh);
            let v2 = match prepared {
                Some(pre) => z.compute_v2_features_with_ref_and_scratch(
                    pre,
                    &distorted,
                    V2NewFeatureToggles::default(),
                    scratch,
                ),
                None => z.compute_v2_features(&RgbSlice::new(r_px, rw, rh), &distorted),
            };
            let v2 = match v2 {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP v2 compute error {:?}: {e:?}", p.dist_path);
                    return None;
                }
            };
            combined.extend_from_slice(v2.features());
        }
        // Folded-720 ONE-pass block: v1 basic (f0..156) + zeros (f156..372)
        // + v2-348 (f372..720), emitted by a single v2-walk pass.
        if do_fold {
            let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
            let distorted = RgbSlice::new(&d_px, dw, dh);
            let folded = match prepared {
                Some(pre) => z.compute_folded720_features_with_ref_and_scratch(
                    pre,
                    &distorted,
                    V2NewFeatureToggles::default(),
                    scratch,
                ),
                None => z.compute_folded720_features(&RgbSlice::new(r_px, rw, rh), &distorted),
            };
            let folded = match folded {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP fold compute error {:?}: {e:?}", p.dist_path);
                    return None;
                }
            };
            combined.extend_from_slice(folded.features());
        }
        n_feat_seen.store(combined.len(), Ordering::Relaxed);
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
    };

    let rows: Vec<String> = if grouped {
        // Group pair indices by reference path, preserving first-seen
        // order; rows are re-emitted in input order below.
        let mut group_of: HashMap<PathBuf, usize> = HashMap::new();
        let mut groups: Vec<(PathBuf, Vec<usize>)> = Vec::new();
        for (i, p) in pairs.iter().enumerate() {
            let gid = *group_of.entry(p.ref_path.clone()).or_insert_with(|| {
                groups.push((p.ref_path.clone(), Vec::new()));
                groups.len() - 1
            });
            groups[gid].1.push(i);
        }
        eprintln!(
            "{} groups ({}x mean reuse) — ref-grouped extraction",
            groups.len(),
            pairs.len() / groups.len().max(1)
        );

        // LPT scheduling: a group runs sequentially on one rayon worker, so
        // without ordering the biggest (large-image × many-variant) groups
        // land last and straggle the whole run. Sort by estimated cost
        // (variant count × reference file size ∝ pixels) descending; row
        // order is restored by the index sort below either way.
        groups.sort_by_key(|(ref_path, idxs)| {
            std::cmp::Reverse(
                idxs.len() as u64 * std::fs::metadata(ref_path).map(|m| m.len()).unwrap_or(1),
            )
        });

        let mut indexed: Vec<(usize, String)> = groups
            .par_iter()
            .map(|(ref_path, idxs)| -> Vec<(usize, String)> {
                if !ref_path.exists() {
                    for &i in idxs {
                        eprintln!(
                            "SKIP missing: {:?} / {:?}",
                            pairs[i].ref_path, pairs[i].dist_path
                        );
                        progress(n_done.fetch_add(1, Ordering::Relaxed));
                    }
                    return Vec::new();
                }
                let (r_px, rw, rh) = zen_io::decode_rgb8(ref_path);
                // Prepare the v2 reference pyramid ONCE per group —
                // with cached ref-side moments unless ZENSIM_AB_MOMENTS=0
                // (both bit-identical; moments trade ~2x prepared memory
                // for skipping the ref V-blur + activity chain per pair).
                let want_moments = std::env::var("ZENSIM_AB_MOMENTS")
                    .map(|v| v != "0")
                    .unwrap_or(true);
                let prepared = if do_v2 || do_fold {
                    let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
                    let r = if want_moments {
                        z.prepare_v2_reference_with_moments(&RgbSlice::new(&r_px, rw, rh))
                    } else {
                        z.prepare_v2_reference(&RgbSlice::new(&r_px, rw, rh))
                    };
                    match r {
                        Ok(p) => Some(p),
                        Err(e) => {
                            for &i in idxs {
                                eprintln!("SKIP v2 prepare error {:?}: {e:?}", pairs[i].dist_path);
                                progress(n_done.fetch_add(1, Ordering::Relaxed));
                            }
                            return Vec::new();
                        }
                    }
                } else {
                    None
                };
                // Variants fan out in parallel WITHIN the group too (rayon
                // work-stealing handles the nesting): corpora with few
                // references and many variants — the sweep shape — would
                // otherwise be wall-bounded by the largest single group.
                // The prepared pyramid + decoded reference are shared by
                // `&`; each rayon split gets its own `V2Scratch`.
                idxs.par_iter()
                    .map_init(V2Scratch::new, |scratch, &i| {
                        let row = score_pair(&pairs[i], &r_px, rw, rh, prepared.as_ref(), scratch)
                            .map(|row| (i, row));
                        progress(n_done.fetch_add(1, Ordering::Relaxed));
                        row
                    })
                    .flatten()
                    .collect()
            })
            .flatten()
            .collect();
        indexed.sort_unstable_by_key(|(i, _)| *i);
        indexed.into_iter().map(|(_, row)| row).collect()
    } else {
        pairs
            .par_iter()
            .filter_map(|p| {
                progress(n_done.fetch_add(1, Ordering::Relaxed));
                if !p.ref_path.exists() {
                    eprintln!("SKIP missing: {:?} / {:?}", p.ref_path, p.dist_path);
                    return None;
                }
                let (r_px, rw, rh) = zen_io::decode_rgb8(&p.ref_path);
                let mut scratch = V2Scratch::new();
                score_pair(p, &r_px, rw, rh, None, &mut scratch)
            })
            .collect()
    };

    let n_feat = n_feat_seen.load(Ordering::Relaxed);
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
    eprintln!(
        "wrote {} rows x {n_feat} features to {}",
        rows.len(),
        args[1]
    );
}
