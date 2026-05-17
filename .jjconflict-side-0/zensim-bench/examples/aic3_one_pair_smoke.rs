//! Debug smoke test for the AIC-3 0-valid eval issue (T4.3).
//!
//! Reads the first AIC-3 row from info.csv, manually re-implements
//! the load + process flow, and prints which step fails. Built as
//! an example so it runs with the same crate features as
//! dataset_metric_baseline.
//!
//! Usage:
//!   cargo run --release -p zensim-bench --example aic3_one_pair_smoke --features training

use std::path::PathBuf;

use butteraugli::ButteraugliParams;
use imgref::Img;
use rayon::prelude::*;
use rgb::RGB8;
use zensim::{ZensimConfig, compute_zensim_with_config};

fn main() {
    let info_csv = PathBuf::from("/mnt/v/dataset/aic3_ctc_epfl/decoded/info.csv");
    let root = info_csv.parent().and_then(|p| p.parent()).unwrap().to_path_buf();
    let original_dir = root.join("original");
    let decoded_dir = root.join("decoded");

    // Build the pair list FIRST (sequentially), then PROCESS in parallel —
    // matching dataset_metric_baseline's pattern exactly.
    let mut pairs: Vec<(PathBuf, PathBuf, String)> = Vec::new();
    let mut rdr = csv::Reader::from_path(&info_csv).expect("open info.csv");
    for record in rdr.records().flatten() {
        if record.len() < 5 {
            continue;
        }
        let codec = record.get(1).unwrap_or("");
        let img_name = record.get(3).unwrap_or("");
        let quality = record.get(4).unwrap_or("");
        if codec.is_empty() || img_name.is_empty() || quality.is_empty() {
            continue;
        }
        let ref_path = original_dir.join(format!("{img_name}.png"));
        let dist_name = format!("{codec}_{img_name}_{quality}.png");
        let dist_path = decoded_dir.join(img_name).join(&dist_name);
        if !ref_path.exists() || !dist_path.exists() {
            continue;
        }
        pairs.push((ref_path, dist_path, img_name.to_string()));
    }
    println!("Loaded {} AIC-3 pairs", pairs.len());

    // Process via rayon par_iter — matches dataset_metric_baseline exactly.
    let results: Vec<Option<(usize, f64)>> = pairs.par_iter().enumerate().map(|(idx, (ref_path, dist_path, _img_name))| {
        let src_img = image::open(ref_path).ok()?.to_rgb8();
        let dst_img = image::open(dist_path).ok()?.to_rgb8();
        let (w, h) = src_img.dimensions();
        let (dw, dh) = dst_img.dimensions();
        if w != dw || h != dh {
            return None;
        }
        let src_pixels: Vec<[u8; 3]> = src_img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
        let dst_pixels: Vec<[u8; 3]> = dst_img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
        let w_us = w as usize;
        let h_us = h as usize;
        if w_us < 8 || h_us < 8 {
            return None;
        }
        let mut cfg = ZensimConfig::default();
        cfg.extended_features = true;
        cfg.compute_iw_features = true;
        let result = compute_zensim_with_config(&src_pixels, &dst_pixels, w_us, h_us, cfg).ok()?;
        let s_img = Img::new(src_pixels.as_slice(), w_us, h_us);
        let d_img = Img::new(dst_pixels.as_slice(), w_us, h_us);
        let _ssim2 = fast_ssim2::compute_ssimulacra2(s_img, d_img).ok()?;
        let src_rgb8: &[RGB8] = bytemuck::cast_slice(&src_pixels);
        let dst_rgb8: &[RGB8] = bytemuck::cast_slice(&dst_pixels);
        let s_b = Img::new(src_rgb8, w_us, h_us);
        let d_b = Img::new(dst_rgb8, w_us, h_us);
        let _butter = butteraugli::butteraugli(s_b, d_b, &ButteraugliParams::default()).ok()?;
        Some((idx, result.score()))
    }).collect();
    let n_valid = results.iter().filter(|r| r.is_some()).count();
    let n_none = results.iter().filter(|r| r.is_none()).count();
    println!("par_iter results: {n_valid} valid, {n_none} None");
    return;

    // Original sequential test (unreachable after the return above)
    let mut attempted = 0usize;
    let mut img_open_ok = 0usize;
    let mut dim_match_ok = 0usize;
    let mut compute_ok = 0usize;
    let mut feat_count_distrib: std::collections::HashMap<usize, usize> = Default::default();

    for record in rdr.records().flatten() {
        if record.len() < 5 {
            continue;
        }
        let codec = record.get(1).unwrap_or("");
        let img_name = record.get(3).unwrap_or("");
        let quality = record.get(4).unwrap_or("");
        if codec.is_empty() || img_name.is_empty() || quality.is_empty() {
            continue;
        }
        let ref_path = original_dir.join(format!("{img_name}.png"));
        let dist_name = format!("{codec}_{img_name}_{quality}.png");
        let dist_path = decoded_dir.join(img_name).join(&dist_name);
        if !ref_path.exists() || !dist_path.exists() {
            continue;
        }

        attempted += 1;

        let src_img = match image::open(&ref_path) {
            Ok(img) => img.to_rgb8(),
            Err(e) => {
                eprintln!("[fail open ref] {}: {}", ref_path.display(), e);
                continue;
            }
        };
        let dst_img = match image::open(&dist_path) {
            Ok(img) => img.to_rgb8(),
            Err(e) => {
                eprintln!("[fail open dst] {}: {}", dist_path.display(), e);
                continue;
            }
        };
        img_open_ok += 1;

        let (w, h) = src_img.dimensions();
        let (dw, dh) = dst_img.dimensions();
        if w != dw || h != dh {
            eprintln!(
                "[fail dim mismatch] {}: ref={}x{}, dist={}x{}",
                img_name, w, h, dw, dh
            );
            continue;
        }
        dim_match_ok += 1;

        let src_pixels: Vec<[u8; 3]> =
            src_img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
        let dst_pixels: Vec<[u8; 3]> =
            dst_img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
        let w_us = w as usize;
        let h_us = h as usize;

        let mut cfg = ZensimConfig::default();
        cfg.extended_features = true;
        cfg.compute_iw_features = true;
        match compute_zensim_with_config(&src_pixels, &dst_pixels, w_us, h_us, cfg) {
            Ok(result) => {
                compute_ok += 1;
                let feats = result.features();
                *feat_count_distrib.entry(feats.len()).or_insert(0) += 1;
                if attempted <= 5 {
                    println!(
                        "[ok zensim] {} ({}x{}): {} features, score={:.4}",
                        img_name,
                        w,
                        h,
                        feats.len(),
                        result.score()
                    );
                }
            }
            Err(e) => {
                eprintln!(
                    "[fail compute] {} ({}x{}): {:?}",
                    img_name, w, h, e
                );
                continue;
            }
        }

        // Test fast-ssim2 forward
        let s_img = Img::new(src_pixels.as_slice(), w_us, h_us);
        let d_img = Img::new(dst_pixels.as_slice(), w_us, h_us);
        match fast_ssim2::compute_ssimulacra2(s_img, d_img) {
            Ok(s) => {
                if attempted <= 5 {
                    println!("[ok ssim2] {} → {:.4}", img_name, s);
                }
            }
            Err(e) => {
                eprintln!("[fail ssim2] {} ({}x{}): {:?}", img_name, w, h, e);
                continue;
            }
        }

        // Test butteraugli forward
        let src_rgb8: &[RGB8] = bytemuck::cast_slice(&src_pixels);
        let dst_rgb8: &[RGB8] = bytemuck::cast_slice(&dst_pixels);
        let s_b = Img::new(src_rgb8, w_us, h_us);
        let d_b = Img::new(dst_rgb8, w_us, h_us);
        let bp = ButteraugliParams::default().with_compute_diffmap(true);
        match butteraugli::butteraugli(s_b, d_b, &bp) {
            Ok(_b) => {
                if attempted <= 5 {
                    println!("[ok butter] {}", img_name);
                }
            }
            Err(e) => {
                eprintln!("[fail butter] {} ({}x{}): {:?}", img_name, w, h, e);
                continue;
            }
        }
    }

    println!();
    println!("=== Summary (first {} valid rows) ===", attempted);
    println!("  image::open OK:    {}", img_open_ok);
    println!("  dim match OK:      {}", dim_match_ok);
    println!("  compute OK:        {}", compute_ok);
    println!("  feature counts:");
    for (n, c) in &feat_count_distrib {
        println!("    {} features: {} pairs", n, c);
    }
}
