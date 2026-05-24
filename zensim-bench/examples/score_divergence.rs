//! Print zensim scores for a list of (ref, dist) pairs.
//!
//! Built twice — once with `--features training` and once with
//! `--features training,zensim/iir-blur` — and the outputs diffed to measure
//! IIR-vs-box-blur score divergence on real photographs.
//!
//! Usage:
//!   cargo run --release -p zensim-bench --example score_divergence --features training -- \
//!     /mnt/v/dataset/kadid10k/images/I01_01_01.png \
//!     /mnt/v/dataset/kadid10k/images/I01_01_05.png \
//!     ...

use std::env;
use std::path::PathBuf;

use image::GenericImageView;
use zensim::{ZensimConfig, compute_zensim_with_config};

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 || args.len() % 2 != 0 {
        eprintln!("usage: score_divergence ref1 dist1 [ref2 dist2 ...]");
        std::process::exit(2);
    }

    // Force extended_features + compute_iw_features so the activity-map blur
    // path (the one we swapped for IIR) is actually exercised.
    let mut cfg = ZensimConfig::default();
    cfg.extended_features = true;
    cfg.compute_iw_features = true;

    println!("# ref\tdist\tn_feat\tfeat0\tfeat100\tfeat200\tfeat250\tfeat300\tfeat350");
    for chunk in args.chunks_exact(2) {
        let ref_path = PathBuf::from(&chunk[0]);
        let dst_path = PathBuf::from(&chunk[1]);

        let r = image::open(&ref_path).unwrap_or_else(|e| panic!("open {ref_path:?}: {e}"));
        let d = image::open(&dst_path).unwrap_or_else(|e| panic!("open {dst_path:?}: {e}"));

        let (w, h) = r.dimensions();
        let (w2, h2) = d.dimensions();
        if (w, h) != (w2, h2) {
            eprintln!(
                "DIM MISMATCH {}x{} vs {}x{} for {:?}",
                w, h, w2, h2, ref_path
            );
            continue;
        }

        let r_rgb = r.to_rgb8().into_raw();
        let d_rgb = d.to_rgb8().into_raw();
        let r_pixels: &[[u8; 3]] = bytemuck::cast_slice(&r_rgb);
        let d_pixels: &[[u8; 3]] = bytemuck::cast_slice(&d_rgb);

        let result =
            compute_zensim_with_config(r_pixels, d_pixels, w as usize, h as usize, cfg)
                .expect("compute");
        let feats = result.features();
        let probe = |i: usize| feats.get(i).copied().unwrap_or(f64::NAN);
        println!(
            "{}\t{}\t{}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}",
            ref_path.file_name().unwrap().to_string_lossy(),
            dst_path.file_name().unwrap().to_string_lossy(),
            feats.len(),
            probe(0),
            probe(100),
            probe(200),
            probe(250),
            probe(300),
            probe(350),
        );
    }
}
