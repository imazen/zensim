//! Imageflow visual regression checksum analysis.
//!
//! This integration test loads pairs of old/new checksum images from
//! imageflow's visual regression test history and runs `classify()` on each pair.
//!
//! Run with:
//! ```bash
//! CHECKSUM_MANIFEST=/tmp/checksum_manifest.json \
//! IMAGE_CACHE=/tmp/imageflow-analysis/images \
//!   cargo test -p zensim --all-features --test imageflow_checksums -- --ignored --nocapture
//! ```

mod common;

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::panic;
use std::path::{Path, PathBuf};

use zensim::source::RgbaSlice;
use zensim::{Zensim, ZensimProfile};

/// Ground truth categories for commits where we know the root cause.
fn ground_truth() -> HashMap<&'static str, &'static str> {
    let mut m = HashMap::new();
    m.insert("d6af6e13", "TransferFunction"); // linear-to-sRGB LUT improvement
    m.insert("7db3e38d", "Mixed"); // dep update + Rust rendering
    m.insert("39269e3d", "AlphaCompositing"); // matte compositing fix
    m.insert("20d2fd4b", "TransferFunction"); // gAMA handling fix
    m.insert("01f33a25", "Mixed"); // linux-arm / Rust graphics
    m.insert("91acf694", "Geometric"); // FitPad fix
    m.insert("29b8603c", "ColorSpaceMatrix"); // color parsing (rgba/argb issue)
    m.insert("9d6b40c6", "Resampling"); // new transpose system for scaling
    m.insert("3db8caa9", "Hashing"); // image hashing change (same images)
    m.insert("11050483", "Hashing"); // hashing change (bitmaps identical)
    m.insert("77fdb819", "Compression"); // lower file sizes
    m.insert("3542b52d", "Rendering"); // rounded corners features
    m.insert("d5f439c7", "AlphaCompositing"); // fix 24-bit PNG with zero alpha
    m.insert("a9b8f5dc", "AlphaCompositing"); // matte support for encoders
    m.insert("46115d2b", "Platform"); // platform-specific PNG difference
    m.insert("fd3399c8", "Bugfix"); // WebPDecoder bug fix
    m.insert("0859c3bf", "Bugfix"); // watermark layout bug fix
    m.insert("3787d913", "API"); // JSON API cleanup
    m.insert("079d0075", "Platform"); // visual test checksums for linux laptop
    m.insert("2cd06cf4", "Platform"); // benchmark on new platform
    m.insert("93c1e21f", "NewTests"); // add visual tests
    m
}


fn checksum_to_filename(checksum: &str) -> String {
    if checksum.contains('_') {
        // Bitmap: append .png
        format!("{}.png", checksum)
    } else if checksum.contains('.') {
        // Encoded: already has extension
        checksum.to_string()
    } else {
        // Bare hash: append .png
        format!("{}.png", checksum)
    }
}

fn load_image_rgba(path: &Path) -> Option<(Vec<[u8; 4]>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    let pixels: Vec<[u8; 4]> = rgba.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
    Some((pixels, w, h))
}

#[test]
#[ignore]
fn imageflow_checksum_analysis() {
    let manifest_path =
        std::env::var("CHECKSUM_MANIFEST").unwrap_or("/tmp/checksum_manifest.json".to_string());
    let image_cache =
        std::env::var("IMAGE_CACHE").unwrap_or("/tmp/imageflow-analysis/images".to_string());
    let output_path = std::env::var("ANALYSIS_OUTPUT")
        .unwrap_or("/tmp/imageflow_checksum_analysis.tsv".to_string());

    let manifest_str = fs::read_to_string(&manifest_path)
        .unwrap_or_else(|e| panic!("Cannot read manifest at {}: {}", manifest_path, e));

    let manifest: serde_json::Value = serde_json::from_str(&manifest_str)
        .unwrap_or_else(|e| panic!("Cannot parse manifest JSON: {}", e));

    let gt = ground_truth();
    let zensim = Zensim::new(ZensimProfile::latest());
    let image_dir = PathBuf::from(&image_cache);

    let mut tsv = String::new();
    writeln!(
        tsv,
        "commit\tcommit_msg\ttest_name\tscore\traw_distance\tdominant_category\tconfidence\t\
         tf\tcsm\tswap\tquant\talpha\tnoise\tblur\tringing\tcolor_shift\t\
         max_abs_delta_r\tmax_abs_delta_g\tmax_abs_delta_b\t\
         frac_identical\tfrac_diff_gt1\t\
         mean_delta_r\tmean_delta_g\tmean_delta_b\t\
         ground_truth\tmatch"
    )
    .unwrap();

    let entries = manifest["manifest"].as_array().expect("manifest array");
    let mut total_pairs = 0;
    let mut compared = 0;
    let mut skipped_missing = 0;
    let mut skipped_dim_mismatch = 0;
    let mut skipped_webp = 0;
    let mut matches = 0;
    let mut mismatches = 0;
    let mut no_gt = 0;

    for entry in entries {
        let commit = entry["commit_hash"].as_str().unwrap();
        let msg = entry["commit_message"].as_str().unwrap();
        let pairs = entry["changed_pairs"].as_array().unwrap();

        for pair in pairs {
            total_pairs += 1;
            let test_name = pair["test_name"].as_str().unwrap();
            let old_cs = pair["old_checksum"].as_str().unwrap();
            let new_cs = pair["new_checksum"].as_str().unwrap();

            let old_file = image_dir.join(checksum_to_filename(old_cs));
            let new_file = image_dir.join(checksum_to_filename(new_cs));

            // Skip WebP
            if old_file.extension().is_some_and(|e| e == "webp")
                || new_file.extension().is_some_and(|e| e == "webp")
            {
                skipped_webp += 1;
                continue;
            }

            // Load images
            let old_img = match load_image_rgba(&old_file) {
                Some(img) => img,
                None => {
                    skipped_missing += 1;
                    continue;
                }
            };
            let new_img = match load_image_rgba(&new_file) {
                Some(img) => img,
                None => {
                    skipped_missing += 1;
                    continue;
                }
            };

            // Check dimensions match
            if old_img.1 != new_img.1 || old_img.2 != new_img.2 {
                skipped_dim_mismatch += 1;
                eprintln!(
                    "  DIM MISMATCH: {} [{}] {}x{} vs {}x{}",
                    test_name, commit, old_img.1, old_img.2, new_img.1, new_img.2
                );
                continue;
            }

            let (w, h) = (old_img.1 as usize, old_img.2 as usize);

            // Skip very small images that trigger blur kernel panics
            if w < 32 || h < 32 {
                eprintln!("  TOO SMALL: {} [{}] {}x{}", test_name, commit, w, h);
                skipped_dim_mismatch += 1;
                continue;
            }

            let old_src = RgbaSlice::new(&old_img.0, w, h);
            let new_src = RgbaSlice::new(&new_img.0, w, h);

            let result = match panic::catch_unwind(panic::AssertUnwindSafe(|| {
                zensim.classify(&old_src, &new_src)
            })) {
                Ok(Ok(r)) => r,
                Ok(Err(e)) => {
                    eprintln!("  ERROR: {} [{}]: {}", test_name, commit, e);
                    continue;
                }
                Err(_) => {
                    eprintln!("  PANIC: {} [{}] {}x{}", test_name, commit, w, h);
                    continue;
                }
            };

            compared += 1;
            let dom_cat = result.classification.dominant;
            let dom_conf = result.classification.confidence;
            let ec = &result.classification;
            let ds = &result.delta_stats;

            let frac_identical = if ds.pixel_count > 0 {
                1.0 - (ds.pixels_differing as f64 / ds.pixel_count as f64)
            } else {
                1.0
            };
            let frac_diff_gt1 = if ds.pixel_count > 0 {
                ds.pixels_differing_by_more_than_1 as f64 / ds.pixel_count as f64
            } else {
                0.0
            };

            let gt_label = gt.get(&commit[..8]).copied().unwrap_or("?");
            let dom_str = format!("{:?}", dom_cat);
            let is_match = if gt_label == "?" {
                no_gt += 1;
                "?"
            } else if gt_label == "Mixed"
                || gt_label == "Hashing"
                || gt_label == "NewTests"
                || gt_label == "Platform"
                || gt_label == "Geometric"
                || gt_label == "Resampling"
                || gt_label == "Rendering"
                || gt_label == "Compression"
                || gt_label == "API"
                || gt_label == "Bugfix"
            {
                // These categories aren't in our classifier — note but don't score
                no_gt += 1;
                "n/a"
            } else if dom_str == gt_label {
                matches += 1;
                "Y"
            } else {
                mismatches += 1;
                "N"
            };

            writeln!(
                tsv,
                "{}\t{}\t{}\t{:.2}\t{:.6}\t{:?}\t{:.3}\t\
                 {:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t\
                 {:.4}\t{:.4}\t{:.4}\t\
                 {:.4}\t{:.4}\t\
                 {:.6}\t{:.6}\t{:.6}\t\
                 {}\t{}",
                commit,
                msg.chars().take(50).collect::<String>(),
                test_name,
                result.result.score,
                result.result.raw_distance,
                dom_cat,
                dom_conf,
                ec.transfer_function,
                ec.color_space_matrix,
                ec.channel_swap,
                ec.quantization,
                ec.alpha_compositing,
                ec.pixel_noise,
                ec.blur,
                ec.ringing,
                ec.color_shift,
                ds.max_abs_delta[0],
                ds.max_abs_delta[1],
                ds.max_abs_delta[2],
                frac_identical,
                frac_diff_gt1,
                ds.mean_delta[0],
                ds.mean_delta[1],
                ds.mean_delta[2],
                gt_label,
                is_match,
            )
            .unwrap();
        }
    }

    // Write TSV
    fs::write(&output_path, &tsv)
        .unwrap_or_else(|e| panic!("Cannot write output to {}: {}", output_path, e));

    // Print summary
    println!("\n=== Imageflow Checksum Analysis Summary ===");
    println!("Total pairs in manifest: {}", total_pairs);
    println!("Compared: {}", compared);
    println!("Skipped (missing image): {}", skipped_missing);
    println!("Skipped (dim mismatch): {}", skipped_dim_mismatch);
    println!("Skipped (WebP): {}", skipped_webp);
    println!();
    println!("Ground truth matches: {}", matches);
    println!("Ground truth mismatches: {}", mismatches);
    println!("No ground truth: {}", no_gt);
    println!();
    println!("Output written to: {}", output_path);

    // Print per-commit summary
    println!("\n=== Per-Commit Breakdown ===");
    for entry in entries {
        let commit = entry["commit_hash"].as_str().unwrap();
        let msg = entry["commit_message"].as_str().unwrap();
        let pairs = entry["changed_pairs"].as_array().unwrap();
        if pairs.is_empty() {
            continue;
        }
        let gt_label = gt.get(&commit[..8]).copied().unwrap_or("?");
        println!(
            "{} {} ({} pairs) [gt: {}]",
            &commit[..8],
            &msg.chars().take(60).collect::<String>(),
            pairs.len(),
            gt_label
        );
    }
}
