use calamine::{Reader, Xlsx};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "zensim-validate", about = "Validate zensim against human quality ratings")]
struct Args {
    /// Dataset directory (e.g., /mnt/v/dataset/tid2013)
    #[arg(long)]
    dataset: PathBuf,

    /// Dataset type
    #[arg(long, value_enum)]
    format: DatasetFormat,

    /// Max images to process (0 = all)
    #[arg(long, default_value = "0")]
    max_images: usize,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum DatasetFormat {
    Tid2013,
    Kadid10k,
    Csiq,
}

/// A single reference-distorted pair with human score.
#[derive(Debug, Clone)]
struct ImagePair {
    reference: PathBuf,
    distorted: PathBuf,
    /// Human subjective score (higher = better quality, normalized to 0-1)
    human_score: f64,
}

fn main() {
    let args = Args::parse();

    let pairs = match args.format {
        DatasetFormat::Tid2013 => load_tid2013(&args.dataset),
        DatasetFormat::Kadid10k => load_kadid10k(&args.dataset),
        DatasetFormat::Csiq => load_csiq(&args.dataset),
    };

    let pairs = if args.max_images > 0 && args.max_images < pairs.len() {
        pairs[..args.max_images].to_vec()
    } else {
        pairs
    };

    println!("Loaded {} image pairs", pairs.len());
    println!("Computing zensim scores...");

    let pb = ProgressBar::new(pairs.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({per_sec}) {msg}")
            .unwrap(),
    );

    // Compute zensim scores in parallel
    let results: Vec<(f64, f64)> = pairs
        .par_iter()
        .map(|pair| {
            let score = compute_pair_score(pair);
            pb.inc(1);
            (pair.human_score, score)
        })
        .collect();

    pb.finish_with_message("done");

    // Filter out failures
    let valid: Vec<(f64, f64)> = results
        .into_iter()
        .filter(|&(_, s)| s.is_finite())
        .collect();

    println!("\nSuccessfully computed {}/{} pairs", valid.len(), pairs.len());

    if valid.len() < 3 {
        println!("Too few valid results for correlation analysis");
        return;
    }

    // Compute correlation metrics
    let human_scores: Vec<f64> = valid.iter().map(|&(h, _)| h).collect();
    let metric_scores: Vec<f64> = valid.iter().map(|&(_, m)| m).collect();

    let srocc = spearman_correlation(&human_scores, &metric_scores);
    let plcc = pearson_correlation(&human_scores, &metric_scores);
    let krocc = kendall_correlation(&human_scores, &metric_scores);

    println!("\n=== Correlation with Human Ratings ===");
    println!("SROCC (Spearman):  {:.4}", srocc);
    println!("PLCC  (Pearson):   {:.4}", plcc);
    println!("KROCC (Kendall):   {:.4}", krocc);
    println!();

    // Show score range
    let min_m = metric_scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_m = metric_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_m: f64 = metric_scores.iter().sum::<f64>() / metric_scores.len() as f64;
    println!("Metric score range: {:.2} to {:.2}, mean: {:.2}", min_m, max_m, mean_m);
}

fn compute_pair_score(pair: &ImagePair) -> f64 {
    let src = match image::open(&pair.reference) {
        Ok(img) => img.to_rgb8(),
        Err(e) => {
            eprintln!("Failed to open {:?}: {}", pair.reference, e);
            return f64::NAN;
        }
    };
    let dst = match image::open(&pair.distorted) {
        Ok(img) => img.to_rgb8(),
        Err(e) => {
            eprintln!("Failed to open {:?}: {}", pair.distorted, e);
            return f64::NAN;
        }
    };

    let (w, h) = src.dimensions();
    let (dw, dh) = dst.dimensions();
    if w != dw || h != dh {
        eprintln!("Dimension mismatch: {:?}", pair.distorted);
        return f64::NAN;
    }

    let src_pixels: Vec<[u8; 3]> = src.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let dst_pixels: Vec<[u8; 3]> = dst.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();

    match zensim::compute_zensim(&src_pixels, &dst_pixels, w as usize, h as usize) {
        Ok(r) => r.score,
        Err(e) => {
            eprintln!("Error on {:?}: {}", pair.distorted, e);
            f64::NAN
        }
    }
}

// ===== Dataset loaders =====

fn load_tid2013(base: &Path) -> Vec<ImagePair> {
    let mos_path = base.join("mos_with_names.txt");
    if !mos_path.exists() {
        // Try alternative locations
        let alt = base.join("mos.txt");
        if alt.exists() {
            return load_tid2013_mos(&alt, base);
        }
        eprintln!("Cannot find mos_with_names.txt in {:?}", base);
        eprintln!("Expected: <base>/mos_with_names.txt with lines like: <mos> <filename>");
        return vec![];
    }
    load_tid2013_mos(&mos_path, base)
}

fn load_tid2013_mos(mos_path: &Path, base: &Path) -> Vec<ImagePair> {
    let content = std::fs::read_to_string(mos_path).expect("Failed to read MOS file");
    let mut pairs = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Format: "mos_value filename" e.g., "5.16 I01_01_1.bmp"
        let parts: Vec<&str> = line.splitn(2, char::is_whitespace).collect();
        if parts.len() < 2 {
            continue;
        }
        let mos: f64 = match parts[0].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let filename = parts[1].trim();

        // Extract reference image number: I01_01_1.bmp → reference_images/I01.bmp
        let ref_num = &filename[..3]; // "I01"
        let ref_path = base.join("reference_images").join(format!("{}.BMP", ref_num));
        let ref_path_lower = base.join("reference_images").join(format!("{}.bmp", ref_num));
        let ref_path = if ref_path.exists() {
            ref_path
        } else if ref_path_lower.exists() {
            ref_path_lower
        } else {
            // Try without subdirectory
            base.join(format!("{}.BMP", ref_num))
        };

        let dist_path = base.join("distorted_images").join(filename);
        let dist_path = if dist_path.exists() {
            dist_path
        } else {
            base.join(filename)
        };

        // TID2013 MOS: 0-9 scale (higher = better)
        pairs.push(ImagePair {
            reference: ref_path,
            distorted: dist_path,
            human_score: mos / 9.0, // Normalize to 0-1
        });
    }

    pairs
}

fn load_kadid10k(base: &Path) -> Vec<ImagePair> {
    let dmos_path = base.join("dmos.csv");
    if !dmos_path.exists() {
        eprintln!("Cannot find dmos.csv in {:?}", base);
        return vec![];
    }

    let mut rdr = csv::Reader::from_path(&dmos_path).expect("Failed to open dmos.csv");
    let mut pairs = Vec::new();

    for result in rdr.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };
        // KADID-10k CSV format: dist_img, ref_img, dmos, std
        if record.len() < 3 {
            continue;
        }

        let dist_name = &record[0];
        let ref_name = &record[1];
        let dmos: f64 = match record[2].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        let ref_path = base.join("images").join(ref_name);
        let dist_path = base.join("images").join(dist_name);

        // KADID DMOS: 1-5 scale (higher = better quality)
        pairs.push(ImagePair {
            reference: ref_path,
            distorted: dist_path,
            human_score: (dmos - 1.0) / 4.0, // Normalize to 0-1
        });
    }

    pairs
}

fn load_csiq(base: &Path) -> Vec<ImagePair> {
    // CSIQ uses an Excel file for DMOS
    let dmos_path = base.join("csiq.DMOS.xlsx");
    if !dmos_path.exists() {
        eprintln!("Cannot find csiq.DMOS.xlsx in {:?}", base);
        eprintln!("Looking for alternative formats...");

        // Try CSV version
        let csv_path = base.join("csiq_dmos.csv");
        if csv_path.exists() {
            return load_csiq_csv(&csv_path, base);
        }
        return vec![];
    }

    load_csiq_xlsx(&dmos_path, base)
}

fn load_csiq_xlsx(xlsx_path: &Path, base: &Path) -> Vec<ImagePair> {
    let mut workbook: Xlsx<_> = calamine::open_workbook(xlsx_path)
        .expect("Failed to open CSIQ DMOS xlsx");

    let mut pairs = Vec::new();

    // CSIQ DMOS xlsx has sheets per distortion type
    let sheet_names: Vec<String> = workbook.sheet_names().to_vec();

    // Distortion type mapping to directory names
    let distortion_dirs: HashMap<&str, &str> = [
        ("JPEG", "jpeg"),
        ("jpeg", "jpeg"),
        ("JPEG 2000", "jpeg2000"),
        ("jpeg2000", "jpeg2000"),
        ("AWGN", "awgn"),
        ("awgn", "awgn"),
        ("Gaussian Blur", "blur"),
        ("blur", "blur"),
        ("fnoise", "fnoise"),
        ("contrast", "contrast"),
    ].into_iter().collect();

    for sheet_name in &sheet_names {
        if let Ok(range) = workbook.worksheet_range(sheet_name) {
            let dist_dir = distortion_dirs.get(sheet_name.as_str())
                .copied()
                .unwrap_or(sheet_name.as_str());

            for row in range.rows().skip(1) { // Skip header
                if row.len() < 2 {
                    continue;
                }
                // Try to parse image name and DMOS
                let img_name = match &row[0] {
                    calamine::Data::String(s) => s.clone(),
                    _ => continue,
                };
                let dmos = match &row[1] {
                    calamine::Data::Float(f) => *f,
                    _ => continue,
                };

                let ref_path = base.join("src_imgs").join(format!("{}.png", img_name));
                let dist_path = base.join("dst_imgs").join(dist_dir).join(format!("{}.png", img_name));

                // CSIQ DMOS: 0-1 scale (lower = better, higher = more distortion)
                pairs.push(ImagePair {
                    reference: ref_path,
                    distorted: dist_path,
                    human_score: 1.0 - dmos, // Invert so higher = better
                });
            }
        }
    }

    pairs
}

fn load_csiq_csv(csv_path: &Path, base: &Path) -> Vec<ImagePair> {
    let mut rdr = csv::Reader::from_path(csv_path).expect("Failed to open CSIQ CSV");
    let mut pairs = Vec::new();

    for result in rdr.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };
        if record.len() < 3 {
            continue;
        }
        let dist_name = &record[0];
        let ref_name = &record[1];
        let dmos: f64 = match record[2].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        pairs.push(ImagePair {
            reference: base.join("src_imgs").join(ref_name),
            distorted: base.join("dst_imgs").join(dist_name),
            human_score: 1.0 - dmos,
        });
    }

    pairs
}

// ===== Correlation statistics =====

fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    let rx = ranks(x);
    let ry = ranks(y);
    pearson_correlation(&rx, &ry)
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut cov = 0.0f64;
    let mut var_x = 0.0f64;
    let mut var_y = 0.0f64;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

fn kendall_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let mut concordant: i64 = 0;
    let mut discordant: i64 = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let x_diff = x[i] - x[j];
            let y_diff = y[i] - y[j];
            let product = x_diff * y_diff;
            if product > 0.0 {
                concordant += 1;
            } else if product < 0.0 {
                discordant += 1;
            }
        }
    }

    let total = concordant + discordant;
    if total == 0 {
        return 0.0;
    }

    (concordant - discordant) as f64 / total as f64
}

fn ranks(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut result = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && indexed[j].1 == indexed[i].1 {
            j += 1;
        }
        // Average rank for ties
        let avg_rank = (i + j) as f64 / 2.0 + 0.5;
        for k in i..j {
            result[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    result
}
