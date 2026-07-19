//! Phase-2 item 6: helpfulness (correlation) screen for the 7 new A.10
//! candidates on KADID-10k and TID2013.
//!
//! Extracts v2 features for every (reference, distorted) pair, aggregates
//! each NEW feature (mean across all 4 scales x 3 channels) to a single
//! scalar per pair, and writes a wide TSV per corpus. Correlation itself is
//! computed by the canonical `panel` binary (`zensim-validate/src/bin/
//! panel.rs`, `--col-predicted <feature> --col-target human_score`) --
//! this tool does NOT hand-roll Spearman, per the no-duplicate-
//! implementations policy.
//!
//! Parallelized via rayon (`threads` feature, default-on) since this is a
//! ~13,000-pair local batch (not a multi-machine fleet job -- KADID+TID
//! fit comfortably in a single-box run; genuinely at-scale corpus work
//! belongs in zenmetrics, per project convention -- this is a "screen" as
//! the phase-2 brief itself names it, not a production pipeline).
//!
//! ```sh
//! cargo run --release -p zensim --features feature-regime-v2 \
//!   --example v2_helpfulness_screen -- kadid /mnt/v/dataset/kadid10k out_kadid.tsv
//! cargo run --release -p zensim --features feature-regime-v2 \
//!   --example v2_helpfulness_screen -- tid /mnt/v/dataset/tid2013 out_tid.tsv
//! ```

#[path = "support/zen_io.rs"]
mod zen_io;

use rayon::prelude::*;
use std::path::{Path, PathBuf};
use zensim::feature_v2::{FEATURES_PER_CHANNEL_V2_TOTAL, idx};
use zensim::{RgbSlice, Zensim, ZensimProfile};

struct Pair {
    ref_path: PathBuf,
    dist_path: PathBuf,
    human_score: f64,
}

fn load_kadid(root: &Path) -> Vec<Pair> {
    let csv = std::fs::read_to_string(root.join("dmos.csv")).expect("read dmos.csv");
    let images = root.join("images");
    csv.lines()
        .skip(1) // header: dist_img,ref_img,dmos,var
        .filter_map(|line| {
            let mut cols = line.split(',');
            let dist = cols.next()?;
            let refimg = cols.next()?;
            let dmos: f64 = cols.next()?.parse().ok()?;
            Some(Pair {
                ref_path: images.join(refimg),
                dist_path: images.join(dist),
                human_score: dmos,
            })
        })
        .collect()
}

fn load_tid(root: &Path) -> Vec<Pair> {
    let txt =
        std::fs::read_to_string(root.join("mos_with_names.txt")).expect("read mos_with_names.txt");
    let dist_dir = root.join("distorted_images_png");
    let ref_dir = root.join("reference_images_png");
    txt.lines()
        .filter_map(|line| {
            let mut it = line.split_whitespace();
            let mos: f64 = it.next()?.parse().ok()?;
            let bmp_name = it.next()?; // e.g. "I01_01_1.bmp" or "i01_01_2.bmp"
            let stem = bmp_name.strip_suffix(".bmp")?;
            let png_name = format!("{stem}.png");
            // Reference id: the 3 chars before the first '_', normalized to
            // uppercase 'I' (reference_images_png only has uppercase names,
            // e.g. "I01.png", even though distorted filenames mix case).
            let ref_num = stem.split('_').next()?; // "I01" or "i01"
            let ref_upper = format!("I{}", &ref_num[1..]);
            Some(Pair {
                ref_path: ref_dir.join(format!("{ref_upper}.png")),
                dist_path: dist_dir.join(png_name),
                human_score: mos,
            })
        })
        .collect()
}

/// Mean of one v2 feature index across all 4 scales x 3 channels.
fn agg(features: &[f64], n_scales: usize, local_idx: usize) -> f64 {
    let mut sum = 0.0;
    let mut n = 0;
    for scale in 0..n_scales {
        for ch in 0..3 {
            let base =
                scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
            sum += features[base + local_idx];
            n += 1;
        }
    }
    sum / n as f64
}

const NEW_FEATURES: &[(&str, usize)] = &[
    ("gms", idx::GMS),
    ("pjnd_transducer_low_k", idx::PJND_TRANSDUCER_LOW_K),
    ("pjnd_transducer_high_k", idx::PJND_TRANSDUCER_HIGH_K),
    ("blockiness", idx::BLOCKINESS),
    ("ringing", idx::RINGING),
    ("banding", idx::BANDING),
    ("edge_width_change", idx::EDGE_WIDTH_CHANGE),
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        eprintln!("usage: v2_helpfulness_screen <kadid|tid> <dataset_root> <out.tsv>");
        std::process::exit(2);
    }
    let (corpus, root, out_path) = (&args[0], PathBuf::from(&args[1]), &args[2]);

    let pairs = match corpus.as_str() {
        "kadid" => load_kadid(&root),
        "tid" => load_tid(&root),
        other => {
            eprintln!("unknown corpus {other:?}, expected kadid|tid");
            std::process::exit(2);
        }
    };
    eprintln!("{} pairs loaded from {:?}", pairs.len(), root);

    let rows: Vec<String> = pairs
        .par_iter()
        .enumerate()
        .filter_map(|(i, p)| {
            if i % 500 == 0 {
                eprintln!("progress: {i}/{}", pairs.len());
            }
            if !p.ref_path.exists() || !p.dist_path.exists() {
                eprintln!("SKIP missing file: {:?} or {:?}", p.ref_path, p.dist_path);
                return None;
            }
            let (r_px, rw, rh) = zen_io::decode_rgb8(&p.ref_path);
            let (d_px, dw, dh) = zen_io::decode_rgb8(&p.dist_path);
            if (rw, rh) != (dw, dh) {
                eprintln!("SKIP dimension mismatch: {:?}", p.dist_path);
                return None;
            }
            let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
            let source = RgbSlice::new(&r_px, rw, rh);
            let distorted = RgbSlice::new(&d_px, dw, dh);
            let result = match z.compute_v2_features(&source, &distorted) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("SKIP compute error on {:?}: {e:?}", p.dist_path);
                    return None;
                }
            };
            let features = result.features();
            let n_scales = result.n_scales();
            let mut row = format!("{}\t{}", p.dist_path.display(), p.human_score);
            for &(_name, local_idx) in NEW_FEATURES {
                row.push('\t');
                row.push_str(&agg(features, n_scales, local_idx).to_string());
            }
            Some(row)
        })
        .collect();

    let mut header = "image\thuman_score".to_string();
    for &(name, _) in NEW_FEATURES {
        header.push('\t');
        header.push_str(name);
    }
    let mut out = String::new();
    out.push_str(&header);
    out.push('\n');
    for row in &rows {
        out.push_str(row);
        out.push('\n');
    }
    std::fs::write(out_path, out).expect("write output tsv");
    eprintln!("wrote {} rows to {out_path}", rows.len());
}
