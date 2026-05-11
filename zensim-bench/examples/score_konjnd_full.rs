//! Score the full 76,104-pair KonJND-1k per-q sweep with fast-ssim2 to
//! produce a paired CSV `(source_path, decoded_path, codec, quality,
//! gpu_ssimulacra2, gpu_butteraugli, dssim)` in the same shape as the
//! safesyn training CSV. Required to reconstruct the V0_5 training mix
//! (per Tick 76: KonJND_train as 4th training group was the missing
//! ingredient).
//!
//! Usage:
//!   cargo run --release -p zensim-bench --example score_konjnd_full -- \
//!     --konjnd /mnt/v/datasets/KonJND-1k/KonJND-1k \
//!     --out /mnt/v/datasets/KonJND-1k/konjnd_full_scored.csv

use butteraugli::ButteraugliParams;
use imgref::Img;
use rayon::prelude::*;
use rgb::RGB8;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

fn main() {
    let mut args = std::env::args().skip(1);
    let mut konjnd_dir: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--konjnd" => konjnd_dir = Some(args.next().unwrap().into()),
            "--out" => out_path = Some(args.next().unwrap().into()),
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let konjnd = konjnd_dir.expect("--konjnd is required");
    let out = out_path.expect("--out is required");

    // Enumerate all (source, codec, q) triples from the file system.
    let jpeg_dir = konjnd.join("jpeg");
    let bpg_dir = konjnd.join("bpg");
    let src_dir = konjnd.join("source_image");

    let mut triples: Vec<(String, String, u32, PathBuf, PathBuf)> = Vec::new();
    // JPEG: SRC0001_JPEG_001.jpg ... SRC1008_JPEG_050.jpg
    for entry in std::fs::read_dir(&jpeg_dir).expect("jpeg dir") {
        let entry = entry.expect("jpeg entry");
        let name = entry.file_name();
        let s = name.to_string_lossy().to_string();
        // Pattern: SRC####_JPEG_###.jpg
        let parts: Vec<&str> = s.trim_end_matches(".jpg").split('_').collect();
        if parts.len() != 3 || parts[1] != "JPEG" {
            continue;
        }
        let src_stem = parts[0]; // SRC0001
        let q: u32 = match parts[2].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let src_path = src_dir.join(format!("{src_stem}.png"));
        triples.push((
            src_stem.to_string(),
            "JPEG".to_string(),
            q,
            src_path,
            entry.path(),
        ));
    }
    // BPG: SRC####_BPG_###.png
    for entry in std::fs::read_dir(&bpg_dir).expect("bpg dir") {
        let entry = entry.expect("bpg entry");
        let name = entry.file_name();
        let s = name.to_string_lossy().to_string();
        let parts: Vec<&str> = s.trim_end_matches(".png").split('_').collect();
        if parts.len() != 3 || parts[1] != "BPG" {
            continue;
        }
        let src_stem = parts[0];
        let q: u32 = match parts[2].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let src_path = src_dir.join(format!("{src_stem}.png"));
        triples.push((
            src_stem.to_string(),
            "BPG".to_string(),
            q,
            src_path,
            entry.path(),
        ));
    }
    eprintln!("Found {} triples (KonJND full per-q sweep)", triples.len());

    let writer = Mutex::new(std::fs::File::create(&out).expect("create out"));
    {
        let mut w = writer.lock().unwrap();
        writeln!(
            w,
            "source_path,decoded_path,codec,quality,gpu_ssimulacra2,gpu_butteraugli,dssim"
        )
        .unwrap();
    }

    let started = std::time::Instant::now();
    let progress = AtomicUsize::new(0);
    let n_total = triples.len();
    triples.par_iter().for_each(|(src_stem, codec, q, src_path, dist_path)| {
        let p = progress.fetch_add(1, Ordering::Relaxed) + 1;
        if p.is_multiple_of(500) {
            let elapsed = started.elapsed().as_secs_f64();
            let rate = p as f64 / elapsed;
            let eta = (n_total - p) as f64 / rate;
            eprintln!("  {p}/{n_total} ({rate:.1}/s, ETA {eta:.0}s)");
        }
        let s_img_o = match image::open(src_path) {
            Ok(i) => i.to_rgb8(),
            Err(_) => return,
        };
        let d_img_o = match image::open(dist_path) {
            Ok(i) => i.to_rgb8(),
            Err(_) => return,
        };
        let (w, h) = s_img_o.dimensions();
        let (dw, dh) = d_img_o.dimensions();
        if w != dw || h != dh {
            return;
        }
        let src_pixels: Vec<[u8; 3]> =
            s_img_o.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
        let dst_pixels: Vec<[u8; 3]> =
            d_img_o.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
        let s_img = Img::new(src_pixels.as_slice(), w as usize, h as usize);
        let d_img = Img::new(dst_pixels.as_slice(), w as usize, h as usize);
        let ssim2 = fast_ssim2::compute_ssimulacra2(s_img, d_img).ok();
        let src_rgb8: &[RGB8] = bytemuck::cast_slice(&src_pixels);
        let dst_rgb8: &[RGB8] = bytemuck::cast_slice(&dst_pixels);
        let s_b = Img::new(src_rgb8, w as usize, h as usize);
        let d_b = Img::new(dst_rgb8, w as usize, h as usize);
        let bp = ButteraugliParams::default();
        let butter = butteraugli::butteraugli(s_b, d_b, &bp).ok().map(|b| b.score);
        let dssim = if let (Some(s2), _) = (ssim2, ()) {
            ((100.0 - s2) / 100.0).clamp(0.0, 1.0)
        } else {
            f64::NAN
        };

        let mut w = writer.lock().unwrap();
        writeln!(
            w,
            "{},{},{},{},{},{},{}",
            src_path.display(),
            dist_path.display(),
            codec,
            q,
            ssim2.unwrap_or(f64::NAN),
            butter.unwrap_or(f64::NAN),
            dssim,
        )
        .unwrap();
    });
    eprintln!(
        "Done {}/{} in {:.1}s -> {}",
        n_total,
        n_total,
        started.elapsed().as_secs_f64(),
        out.display()
    );
}
