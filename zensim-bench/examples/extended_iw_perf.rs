//! Runtime cost of the four (extended_features, compute_iw_features)
//! permutations.
//!
//! Calls `compute_zensim_with_config` with all four flag combinations
//! on a single (ref, dist) pair, repeats N times per config, reports
//! min/median/mean wall time in milliseconds, plus the feature count
//! per config.
//!
//! Usage:
//!   cargo run --release --example extended_iw_perf -p zensim-bench --features training -- \
//!     --ref /mnt/v/dataset/kadid10k/images/I01_01_01.png \
//!     --dist /mnt/v/dataset/kadid10k/images/I01_01_03.png \
//!     --iters 20
//!
//! Defaults: --iters 20, builds a deterministic 512×512 synthetic pair
//! if no --ref / --dist given.

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use zensim::{compute_zensim_with_config, ZensimConfig};

fn main() {
    let mut args = env::args().skip(1);
    let mut ref_path: Option<PathBuf> = None;
    let mut dist_path: Option<PathBuf> = None;
    let mut iters: usize = 20;
    let mut size: usize = 512;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--ref" => ref_path = Some(args.next().expect("--ref PATH").into()),
            "--dist" => dist_path = Some(args.next().expect("--dist PATH").into()),
            "--iters" => iters = args.next().expect("--iters N").parse().expect("usize"),
            "--size" => size = args.next().expect("--size N").parse().expect("usize"),
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }

    // Load or synthesize images.
    let (src_pixels, dst_pixels, w, h) = match (&ref_path, &dist_path) {
        (Some(rp), Some(dp)) => {
            let src = image::open(rp).expect("open ref").to_rgb8();
            let dst = image::open(dp).expect("open dist").to_rgb8();
            let (w, h) = src.dimensions();
            let (dw, dh) = dst.dimensions();
            assert_eq!((w, h), (dw, dh), "ref/dist dims mismatch");
            let src_pix: Vec<[u8; 3]> = src.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
            let dst_pix: Vec<[u8; 3]> = dst.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
            (src_pix, dst_pix, w as usize, h as usize)
        }
        _ => {
            // Deterministic synthetic: gradient + noisy gradient.
            let n = size * size;
            let mut src = Vec::with_capacity(n);
            let mut dst = Vec::with_capacity(n);
            for y in 0..size {
                for x in 0..size {
                    let r = ((x * 255) / size) as u8;
                    let g = ((y * 255) / size) as u8;
                    let b = (((x + y) * 255) / (2 * size)) as u8;
                    src.push([r, g, b]);
                    // Add a small deterministic perturbation to simulate compression noise.
                    let drift = (((x as i32 - y as i32).abs() % 17) - 8) as i16;
                    let cl = |v: u8, d: i16| (v as i16 + d).clamp(0, 255) as u8;
                    dst.push([cl(r, drift), cl(g, -drift / 2), cl(b, drift / 3)]);
                }
            }
            (src, dst, size, size)
        }
    };

    println!("# Extended-features + IW runtime cost (4 permutations)");
    println!();
    println!("Image: {w}×{h} ({} pixels), iters per config: {iters}", w * h);
    if let (Some(r), Some(d)) = (&ref_path, &dist_path) {
        println!("Ref:  {}", r.display());
        println!("Dist: {}", d.display());
    } else {
        println!("Synthetic deterministic gradient + perturbation");
    }
    println!();

    let perms: &[(bool, bool, &str)] = &[
        (false, false, "Standard      (228 features)"),
        (true, false, "Extended only (300 features = +masked)"),
        (false, true, "IW only       (300 features = +IW)"),
        (true, true, "Extended + IW (372 features = +masked +IW)"),
    ];

    println!("| Config | n_features | min ms | median ms | mean ms | × vs standard |");
    println!("|---|---:|---:|---:|---:|---:|");

    let mut baseline_mean_ms: Option<f64> = None;

    for &(ext, iw, label) in perms {
        // Warm-up (1 untimed call).
        {
            let mut cfg = ZensimConfig::default();
            cfg.extended_features = ext;
            cfg.compute_iw_features = iw;
            cfg.allow_multithreading = true;
            let _ = compute_zensim_with_config(&src_pixels, &dst_pixels, w, h, cfg);
        }

        let mut times: Vec<f64> = Vec::with_capacity(iters);
        let mut n_features = 0;
        for _ in 0..iters {
            let mut cfg = ZensimConfig::default();
            cfg.extended_features = ext;
            cfg.compute_iw_features = iw;
            cfg.allow_multithreading = true;
            let t0 = Instant::now();
            let result = compute_zensim_with_config(&src_pixels, &dst_pixels, w, h, cfg)
                .expect("compute");
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            times.push(elapsed);
            n_features = result.features().len();
        }
        times.sort_by(|a, b| a.total_cmp(b));
        let min = times[0];
        let median = times[times.len() / 2];
        let mean: f64 = times.iter().sum::<f64>() / times.len() as f64;
        let rel = match baseline_mean_ms {
            Some(b) if b > 0.0 => mean / b,
            _ => 1.0,
        };
        if baseline_mean_ms.is_none() {
            baseline_mean_ms = Some(mean);
        }
        println!(
            "| {label} | {n_features} | {min:.2} | {median:.2} | {mean:.2} | {rel:.2}× |"
        );
    }

    println!();
    println!("_All times are wall-clock per `compute_zensim_with_config` call, single (ref, dist) pair, parallel=true (rayon)._");
}
