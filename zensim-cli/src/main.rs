use clap::Parser;
use std::path::PathBuf;
use zensim::{RgbSlice, Zensim, ZensimProfile};

#[derive(Parser)]
#[command(name = "zensim", about = "Fast psychovisual image similarity metric")]
struct Args {
    /// Reference (source) image
    source: PathBuf,
    /// Distorted image to compare
    distorted: PathBuf,
    /// Print raw feature vector
    #[arg(long, default_value = "false")]
    features: bool,
}

fn main() {
    let args = Args::parse();

    let src_img = image::open(&args.source)
        .unwrap_or_else(|e| {
            eprintln!("Failed to open source image {:?}: {}", args.source, e);
            std::process::exit(1);
        })
        .to_rgb8();

    let dst_img = image::open(&args.distorted)
        .unwrap_or_else(|e| {
            eprintln!("Failed to open distorted image {:?}: {}", args.distorted, e);
            std::process::exit(1);
        })
        .to_rgb8();

    let (w, h) = src_img.dimensions();
    let (dw, dh) = dst_img.dimensions();

    if w != dw || h != dh {
        eprintln!("Image dimensions don't match: {}x{} vs {}x{}", w, h, dw, dh);
        std::process::exit(1);
    }

    let src_pixels: Vec<[u8; 3]> = src_img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let dst_pixels: Vec<[u8; 3]> = dst_img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();

    let z = Zensim::new(ZensimProfile::latest());
    let src = RgbSlice::new(&src_pixels, w as usize, h as usize);
    let dst = RgbSlice::new(&dst_pixels, w as usize, h as usize);

    let start = std::time::Instant::now();
    let result = if args.features {
        z.compute_all_features(&src, &dst)
    } else {
        z.compute(&src, &dst)
    }
    .unwrap_or_else(|e| {
        eprintln!("Error computing zensim: {}", e);
        std::process::exit(1);
    });
    let elapsed = start.elapsed();

    println!(
        "score: {:.4}  raw_distance: {:.6}  time: {:.3}ms  ({}x{})",
        result.score,
        result.raw_distance,
        elapsed.as_secs_f64() * 1000.0,
        w,
        h
    );

    if args.features {
        let fpc = 13;
        let ch_names = ["X", "Y", "B"];
        let feat_names = [
            "ssim_mean",
            "ssim_4th",
            "ssim_2nd",
            "art_mean",
            "art_4th",
            "art_2nd",
            "det_mean",
            "det_4th",
            "det_2nd",
            "mse",
            "hf_energy_loss",
            "hf_mag_loss",
            "hf_energy_gain",
        ];
        let fpc_scale = fpc * 3;
        for (si, chunk) in result.features.chunks(fpc_scale).enumerate() {
            println!("Scale {}:", si);
            for (ci, ch_feats) in chunk.chunks(fpc).enumerate() {
                let nonzero: Vec<String> = ch_feats
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.abs() > 1e-12)
                    .map(|(fi, v)| {
                        let name = feat_names.get(fi).copied().unwrap_or("?");
                        format!("{}={:.6}", name, v)
                    })
                    .collect();
                if !nonzero.is_empty() {
                    println!("  {}: {}", ch_names[ci], nonzero.join(", "));
                }
            }
        }
    }
}
