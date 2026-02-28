//! Imageflow visual regression checksum analysis.
//!
//! Downloads old/new image pairs from imageflow's S3 bucket and runs
//! `classify()` on each pair. The manifest of pairs and ground truth
//! analysis is committed at `tests/fixtures/imageflow_checksum_pairs.json`.
//!
//! Run with:
//! ```bash
//! cargo test -p zensim --all-features --test imageflow_checksums -- --ignored --nocapture
//! ```
//!
//! Images are cached in `$IMAGEFLOW_CACHE` (default: `/tmp/imageflow-checksum-images/`).

mod common;

use std::fmt::Write as FmtWrite;
use std::fs;
use std::panic;
use std::path::{Path, PathBuf};
use std::process::Command;

use zensim::source::RgbaSlice;
use zensim::{Zensim, ZensimProfile};

// ─── Image download ─────────────────────────────────────────────────────

fn try_curl(url: &str, dest: &Path) -> Result<(), String> {
    let status = Command::new("curl")
        .args(["-fSL", "-o"])
        .arg(dest)
        .arg(url)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .status();
    match status {
        Ok(s) if s.success() => Ok(()),
        _ => Err("curl failed".into()),
    }
}

fn try_wget(url: &str, dest: &Path) -> Result<(), String> {
    let status = Command::new("wget")
        .args(["-q", "-O"])
        .arg(dest)
        .arg(url)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .status();
    match status {
        Ok(s) if s.success() => Ok(()),
        _ => Err("wget failed".into()),
    }
}

fn try_powershell(url: &str, dest: &Path) -> Result<(), String> {
    let script = "param($u,$o) Invoke-WebRequest -Uri $u -OutFile $o";
    let status = Command::new("powershell")
        .args(["-NoProfile", "-Command", script, url])
        .arg(dest)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .status();
    match status {
        Ok(s) if s.success() => Ok(()),
        _ => Err("powershell failed".into()),
    }
}

fn download(url: &str, dest: &Path) -> Result<(), String> {
    try_curl(url, dest)
        .or_else(|_| try_wget(url, dest))
        .or_else(|_| try_powershell(url, dest))
}

/// Ensure an image file exists at `cache_dir/filename`, downloading from S3 if needed.
fn ensure_image(s3_base: &str, checksum: &str, cache_dir: &Path) -> Option<PathBuf> {
    let filename = checksum_to_filename(checksum);
    let dest = cache_dir.join(&filename);

    if dest.exists() {
        return Some(dest);
    }

    let url = format!("{}/{}", s3_base, filename);
    match download(&url, &dest) {
        Ok(()) => Some(dest),
        Err(e) => {
            eprintln!("  DOWNLOAD FAILED: {} ({})", filename, e);
            // Clean up partial download
            let _ = fs::remove_file(&dest);
            None
        }
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────

fn checksum_to_filename(checksum: &str) -> String {
    if checksum.contains('_') {
        format!("{}.png", checksum)
    } else if checksum.contains('.') {
        checksum.to_string()
    } else {
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

// ─── Test ───────────────────────────────────────────────────────────────

#[test]
#[ignore]
fn imageflow_checksum_analysis() {
    // Load manifest from committed fixture
    let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("imageflow_checksum_pairs.json");

    let manifest_str = fs::read_to_string(&manifest_path)
        .unwrap_or_else(|e| panic!("Cannot read manifest at {}: {}", manifest_path.display(), e));
    let manifest: serde_json::Value = serde_json::from_str(&manifest_str)
        .unwrap_or_else(|e| panic!("Cannot parse manifest JSON: {}", e));

    let s3_base = manifest["s3_base"].as_str().expect("s3_base");
    let cache_dir = PathBuf::from(
        std::env::var("IMAGEFLOW_CACHE")
            .unwrap_or_else(|_| "/tmp/imageflow-checksum-images".to_string()),
    );
    fs::create_dir_all(&cache_dir).expect("create cache dir");

    let output_path = std::env::var("ANALYSIS_OUTPUT")
        .unwrap_or_else(|_| "/tmp/imageflow_checksum_analysis.tsv".to_string());

    let zensim = Zensim::new(ZensimProfile::latest());
    let commits = manifest["commits"].as_array().expect("commits array");

    let mut tsv = String::new();
    writeln!(
        tsv,
        "commit\tchange_type\ttest_name\tscore\traw_distance\tdominant_category\tconfidence\t\
         tf\tcsm\tswap\tquant\talpha\tnoise\tblur\tringing\tcolor_shift\t\
         max_abs_delta_r\tmax_abs_delta_g\tmax_abs_delta_b\t\
         frac_identical\tfrac_diff_gt1\t\
         mean_delta_r\tmean_delta_g\tmean_delta_b\t\
         expected_delta\tcommit_description"
    )
    .unwrap();

    let mut total_pairs = 0;
    let mut compared = 0;
    let mut skipped_download = 0;
    let mut skipped_dim_mismatch = 0;
    let mut skipped_webp = 0;
    let mut skipped_small = 0;

    for commit_entry in commits {
        let commit = commit_entry["commit"].as_str().unwrap();
        let change_type = commit_entry["change_type"].as_str().unwrap();
        let description = commit_entry["description"].as_str().unwrap();
        let expected_delta = commit_entry["expected_delta"].as_str().unwrap();
        let pairs = commit_entry["pairs"].as_array().unwrap();

        for pair in pairs {
            total_pairs += 1;
            let arr = pair.as_array().unwrap();
            let test_name = arr[0].as_str().unwrap();
            let old_cs = arr[1].as_str().unwrap();
            let new_cs = arr[2].as_str().unwrap();

            // Skip WebP
            let old_fn = checksum_to_filename(old_cs);
            let new_fn = checksum_to_filename(new_cs);
            if old_fn.ends_with(".webp") || new_fn.ends_with(".webp") {
                skipped_webp += 1;
                continue;
            }

            // Download/load images
            let old_path = match ensure_image(s3_base, old_cs, &cache_dir) {
                Some(p) => p,
                None => {
                    skipped_download += 1;
                    continue;
                }
            };
            let new_path = match ensure_image(s3_base, new_cs, &cache_dir) {
                Some(p) => p,
                None => {
                    skipped_download += 1;
                    continue;
                }
            };

            let old_img = match load_image_rgba(&old_path) {
                Some(img) => img,
                None => {
                    eprintln!("  DECODE FAILED: {} (old: {})", test_name, old_path.display());
                    skipped_download += 1;
                    continue;
                }
            };
            let new_img = match load_image_rgba(&new_path) {
                Some(img) => img,
                None => {
                    eprintln!("  DECODE FAILED: {} (new: {})", test_name, new_path.display());
                    skipped_download += 1;
                    continue;
                }
            };

            // Check dimensions match
            if old_img.1 != new_img.1 || old_img.2 != new_img.2 {
                skipped_dim_mismatch += 1;
                continue;
            }

            let (w, h) = (old_img.1 as usize, old_img.2 as usize);

            // Skip very small images
            if w < 32 || h < 32 {
                skipped_small += 1;
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

            writeln!(
                tsv,
                "{}\t{}\t{}\t{:.2}\t{:.6}\t{:?}\t{:.3}\t\
                 {:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t\
                 {:.4}\t{:.4}\t{:.4}\t\
                 {:.4}\t{:.4}\t\
                 {:.6}\t{:.6}\t{:.6}\t\
                 {}\t{}",
                commit,
                change_type,
                test_name,
                result.result.score,
                result.result.raw_distance,
                ec.dominant,
                ec.confidence,
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
                expected_delta,
                description.chars().take(80).collect::<String>(),
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
    println!("Compared:                {}", compared);
    println!("Skipped (download/decode): {}", skipped_download);
    println!("Skipped (dim mismatch):  {}", skipped_dim_mismatch);
    println!("Skipped (too small):     {}", skipped_small);
    println!("Skipped (WebP):          {}", skipped_webp);

    // Per-commit summary
    println!("\n=== Per-Commit Breakdown ===");
    for entry in commits {
        let commit = entry["commit"].as_str().unwrap();
        let change_type = entry["change_type"].as_str().unwrap();
        let msg = entry["message"].as_str().unwrap();
        let pairs = entry["pairs"].as_array().unwrap();
        if pairs.is_empty() {
            continue;
        }
        println!(
            "  {} [{:20}] {:3} pairs  {}",
            &commit[..8],
            change_type,
            pairs.len(),
            &msg.chars().take(60).collect::<String>(),
        );
    }

    println!("\nOutput: {}", output_path);
}
