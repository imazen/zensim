//! Meta-regression: take two directories of named PNGs (a baseline +
//! an actual) and run [`MontageOptions::render`] on each matching pair.
//! Emits the regression montages into a third directory.
//!
//! ```text
//! cargo run -p zensim-regress --example meta_montage -- \
//!     /mnt/v/output/zensim-regress/api_main \
//!     /mnt/v/output/zensim-regress/api_branch \
//!     /mnt/v/output/zensim-regress/api_meta
//! ```
//!
//! The output is itself a montage gallery — using zensim-regress's
//! diff_image API to visualize the pixel differences between the two
//! sets of zensim-regress outputs. Eating our own dog food.

use std::env;
use std::path::{Path, PathBuf};

use zensim_regress::Bitmap;
use zensim_regress::diff_image::{AnnotationText, MontageOptions};

fn main() {
    let mut args = env::args().skip(1);
    let baseline = PathBuf::from(
        args.next()
            .expect("usage: meta_montage <baseline_dir> <actual_dir> <output_dir>"),
    );
    let actual = PathBuf::from(args.next().expect("missing <actual_dir>"));
    let out = PathBuf::from(args.next().expect("missing <output_dir>"));
    std::fs::create_dir_all(&out).expect("create output dir");

    let baseline_label = baseline
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("BASELINE")
        .to_uppercase();
    let actual_label = actual
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("ACTUAL")
        .to_uppercase();

    let mut baseline_files: Vec<_> = std::fs::read_dir(&baseline)
        .expect("read baseline dir")
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("png"))
        .map(|e| e.file_name())
        .collect();
    baseline_files.sort();

    let mut clean = 0u32;
    let mut differ = 0u32;
    let mut missing = 0u32;
    for name in baseline_files {
        let n = name.to_string_lossy();
        let a_path = baseline.join(&name);
        let b_path = actual.join(&name);
        if !b_path.exists() {
            println!("{n:<32} MISSING in {actual_label}");
            missing += 1;
            continue;
        }
        let a = match Bitmap::open(&a_path) {
            Ok(b) => b,
            Err(e) => {
                println!("{n:<32} ERROR loading baseline: {e}");
                continue;
            }
        };
        let b = match Bitmap::open(&b_path) {
            Ok(b) => b,
            Err(e) => {
                println!("{n:<32} ERROR loading actual: {e}");
                continue;
            }
        };

        // Compute basic stats so we can annotate the montage with the
        // regression verdict — same logic the regression harness uses,
        // but stripped down: pixel count differing + max channel delta.
        let dim_a = (a.width(), a.height());
        let dim_b = (b.width(), b.height());
        let stats = if dim_a == dim_b {
            Some(pixel_stats(a.as_raw(), b.as_raw()))
        } else {
            None
        };

        let title = format!("{n}  —  {baseline_label} vs {actual_label}");
        let mut ann = AnnotationText::empty().with_title(&title);
        let dims_match = dim_a == dim_b;
        let verdict_label = verdict(&stats, dims_match);
        match (stats, dims_match) {
            (Some(s), _) if s.diff_pixels == 0 => {
                ann.primary_lines = vec![(
                    format!("PASS  byte-identical ({}×{})", dim_a.0, dim_a.1),
                    [80, 220, 80, 255],
                )];
                clean += 1;
            }
            (Some(s), _) => {
                ann.primary_lines = vec![
                    ("DIFFERS".to_string(), [255, 200, 80, 255]),
                    (
                        format!(
                            "{:.2}% pixels differ  max Δ = {}  avg Δ = {:.3}",
                            s.pct_diff, s.max_d, s.avg_d
                        ),
                        [255, 200, 80, 255],
                    ),
                ];
                differ += 1;
            }
            (None, _) => {
                ann.primary_lines = vec![(
                    format!(
                        "DIM-MISMATCH  {}×{} vs {}×{}",
                        dim_a.0, dim_a.1, dim_b.0, dim_b.1
                    ),
                    [255, 80, 80, 255],
                )];
                differ += 1;
            }
        }

        let mut opts = MontageOptions::default();
        opts.expected_label = Some(baseline_label.clone());
        opts.actual_label = Some(actual_label.clone());
        opts.amplification = 10;

        let m = opts.render(&a, &b, &ann);
        let out_path = out.join(&name);
        m.save(&out_path).expect("save meta montage");
        println!("{n:<32} {verdict_label} → {}", out_path.display());
    }

    println!("\n{clean} clean, {differ} differ, {missing} missing");
}

struct PixelStats {
    diff_pixels: u64,
    max_d: u32,
    avg_d: f64,
    pct_diff: f64,
}

fn pixel_stats(a: &[u8], b: &[u8]) -> PixelStats {
    let mut diff_pixels: u64 = 0;
    let mut sum_d: u64 = 0;
    let mut max_d: u32 = 0;
    let n = a.len() / 4;
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let d = (0..4)
            .map(|i| (pa[i] as i32 - pb[i] as i32).unsigned_abs())
            .max()
            .unwrap_or(0);
        if d > 0 {
            diff_pixels += 1;
        }
        max_d = max_d.max(d);
        sum_d += d as u64;
    }
    PixelStats {
        diff_pixels,
        max_d,
        avg_d: (sum_d as f64) / (n.max(1) as f64),
        pct_diff: 100.0 * (diff_pixels as f64) / (n.max(1) as f64),
    }
}

fn verdict(stats: &Option<PixelStats>, dims_match: bool) -> &'static str {
    match (stats, dims_match) {
        (Some(s), _) if s.diff_pixels == 0 => "✓ identical",
        (Some(_), _) => "Δ differs",
        (None, false) => "‖ dim-mismatch",
        _ => "?",
    }
}

#[allow(dead_code)]
fn _bitmap_open_check(p: &Path) {
    let _ = Bitmap::open(p);
}
