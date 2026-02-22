use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "zensim", about = "Fast psychovisual image similarity metric")]
struct Args {
    /// Reference (source) image
    source: PathBuf,
    /// Distorted image to compare
    distorted: PathBuf,
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

    // Convert to &[[u8; 3]]
    let src_pixels: Vec<[u8; 3]> = src_img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let dst_pixels: Vec<[u8; 3]> = dst_img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();

    let start = std::time::Instant::now();
    let result = zensim::compute_zensim(&src_pixels, &dst_pixels, w as usize, h as usize)
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
}
