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

    /// Train weights: run Nelder-Mead optimization to find best feature weights
    #[arg(long, default_value = "false")]
    train: bool,

    /// Output features CSV for external analysis
    #[arg(long)]
    features_csv: Option<PathBuf>,
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

    // When training, compute all features (no skipping) so weights can be freely optimized
    let compute_all = args.train;

    // Compute zensim scores + features in parallel
    let results: Vec<(f64, zensim::ZensimResult)> = pairs
        .par_iter()
        .map(|pair| {
            let result = compute_pair_result(pair, compute_all);
            pb.inc(1);
            (pair.human_score, result)
        })
        .collect();

    pb.finish_with_message("done");

    // Filter out failures (NaN score)
    let valid: Vec<(f64, zensim::ZensimResult)> = results
        .into_iter()
        .filter(|(_, r)| r.score.is_finite())
        .collect();

    println!("\nSuccessfully computed {}/{} pairs", valid.len(), pairs.len());

    if valid.len() < 3 {
        println!("Too few valid results for correlation analysis");
        return;
    }

    // Output features CSV if requested
    if let Some(ref csv_path) = args.features_csv {
        write_features_csv(csv_path, &valid);
    }

    let human_scores: Vec<f64> = valid.iter().map(|(h, _)| *h).collect();
    let metric_scores: Vec<f64> = valid.iter().map(|(_, r)| r.score).collect();

    let srocc = spearman_correlation(&human_scores, &metric_scores);
    let plcc = pearson_correlation(&human_scores, &metric_scores);
    let krocc = kendall_correlation(&human_scores, &metric_scores);

    println!("\n=== Correlation with Human Ratings (default weights) ===");
    println!("SROCC (Spearman):  {:.4}", srocc);
    println!("PLCC  (Pearson):   {:.4}", plcc);
    println!("KROCC (Kendall):   {:.4}", krocc);

    let min_m = metric_scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_m = metric_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_m: f64 = metric_scores.iter().sum::<f64>() / metric_scores.len() as f64;
    println!("Metric score range: {:.2} to {:.2}, mean: {:.2}", min_m, max_m, mean_m);

    let raw_dists: Vec<f64> = valid.iter().map(|(_, r)| r.raw_distance).collect();
    let min_d = raw_dists.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_d = raw_dists.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_d: f64 = raw_dists.iter().sum::<f64>() / raw_dists.len() as f64;
    let mut sorted_d = raw_dists.clone();
    sorted_d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p10 = sorted_d[sorted_d.len() / 10];
    let p50 = sorted_d[sorted_d.len() / 2];
    let p90 = sorted_d[sorted_d.len() * 9 / 10];
    println!("Raw distance: min={:.3}, p10={:.3}, p50={:.3}, p90={:.3}, max={:.3}, mean={:.3}\n",
        min_d, p10, p50, p90, max_d, mean_d);

    // Train weights if requested
    if args.train {
        let feature_vecs: Vec<&[f64]> = valid.iter().map(|(_, r)| r.features.as_slice()).collect();
        let n_features = feature_vecs[0].len();
        println!("Training weights on {} pairs with {} features...", valid.len(), n_features);

        let best_weights = train_weights(&human_scores, &feature_vecs, n_features);

        // Evaluate with trained weights
        let trained_scores: Vec<f64> = feature_vecs.iter()
            .map(|f| zensim::score_from_features(f, &best_weights).0)
            .collect();

        let srocc_t = spearman_correlation(&human_scores, &trained_scores);
        let plcc_t = pearson_correlation(&human_scores, &trained_scores);
        let krocc_t = kendall_correlation(&human_scores, &trained_scores);

        println!("\n=== Correlation with Trained Weights ===");
        println!("SROCC (Spearman):  {:.4}", srocc_t);
        println!("PLCC  (Pearson):   {:.4}", plcc_t);
        println!("KROCC (Kendall):   {:.4}", krocc_t);

        // Print weights for embedding in code
        println!("\n// Trained weights ({} values):", best_weights.len());
        println!("const TRAINED_WEIGHTS: [f64; {}] = [", best_weights.len());
        for (i, w) in best_weights.iter().enumerate() {
            if i % 6 == 0 {
                let scale = i / 18;
                let ch = (i % 18) / 6;
                let ch_name = ["X", "Y", "B"][ch];
                print!("    // Scale {} Channel {}\n    ", scale, ch_name);
            }
            print!("{:.6}, ", w);
            if i % 6 == 5 {
                println!();
            }
        }
        println!("];");
    }
}

fn compute_pair_result(pair: &ImagePair, compute_all_features: bool) -> zensim::ZensimResult {
    let nan_result = zensim::ZensimResult {
        score: f64::NAN,
        raw_distance: f64::NAN,
        features: vec![],
    };

    let src = match image::open(&pair.reference) {
        Ok(img) => img.to_rgb8(),
        Err(_) => return nan_result,
    };
    let dst = match image::open(&pair.distorted) {
        Ok(img) => img.to_rgb8(),
        Err(_) => return nan_result,
    };

    let (w, h) = src.dimensions();
    let (dw, dh) = dst.dimensions();
    if w != dw || h != dh {
        return nan_result;
    }

    let src_pixels: Vec<[u8; 3]> = src.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let dst_pixels: Vec<[u8; 3]> = dst.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();

    let config = zensim::ZensimConfig {
        compute_all_features,
        ..Default::default()
    };
    match zensim::compute_zensim_with_config(&src_pixels, &dst_pixels, w as usize, h as usize, config) {
        Ok(r) => r,
        Err(_) => nan_result,
    }
}

fn write_features_csv(path: &Path, data: &[(f64, zensim::ZensimResult)]) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("Failed to create features CSV");

    // Header
    let n_features = data[0].1.features.len();
    write!(f, "human_score,metric_score,raw_distance").unwrap();
    for i in 0..n_features {
        write!(f, ",f{}", i).unwrap();
    }
    writeln!(f).unwrap();

    // Data
    for (human, result) in data {
        write!(f, "{},{},{}", human, result.score, result.raw_distance).unwrap();
        for feat in &result.features {
            write!(f, ",{}", feat).unwrap();
        }
        writeln!(f).unwrap();
    }
    println!("Wrote features to {:?}", path);
}

/// Train weights using coordinate descent with random restarts.
/// Maximizes SROCC between predicted scores and human scores.
fn train_weights(human_scores: &[f64], features: &[&[f64]], n_features: usize) -> Vec<f64> {
    let mut best_weights = vec![1.0; n_features];
    let mut best_srocc = -1.0f64;

    // Compute feature ranges for additive step sizing
    let mut feat_max = vec![0.0f64; n_features];
    for feats in features.iter() {
        for (i, &f) in feats.iter().enumerate() {
            feat_max[i] = feat_max[i].max(f.abs());
        }
    }

    // Try several random starting points
    let n_restarts = 10;
    let mut rng_state = 42u64;

    let mut next_rand = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng_state >> 33) as f64 / (u32::MAX as f64)
    };

    for restart in 0..n_restarts {
        let mut weights = if restart == 0 {
            // Start from current embedded weights (good initial point)
            let embedded = zensim::FEATURES_PER_SCALE; // just for count
            let _ = embedded;
            vec![1.0 / n_features as f64; n_features]
        } else {
            // Sparse random: only activate ~30% of features
            (0..n_features)
                .map(|_| {
                    if next_rand() < 0.3 {
                        next_rand() * 10.0
                    } else {
                        0.0
                    }
                })
                .collect()
        };

        // Coordinate descent: optimize one weight at a time
        for iter in 0..50 {
            let mut improved = false;
            let step_scale = if iter < 20 { 1.0 } else { 0.5 };

            for dim in 0..n_features {
                let current_val = weights[dim];
                let mut best_val = current_val;
                let mut best_local_srocc = eval_srocc(human_scores, features, &weights);

                // Multiplicative steps
                for &mult in &[0.0, 0.5, 1.5, 2.0, 3.0, 0.1, 5.0, 10.0] {
                    weights[dim] = current_val * mult * step_scale + current_val * (1.0 - step_scale);
                    if weights[dim] < 0.0 { weights[dim] = 0.0; }
                    let srocc = eval_srocc(human_scores, features, &weights);
                    if srocc > best_local_srocc {
                        best_local_srocc = srocc;
                        best_val = weights[dim];
                        improved = true;
                    }
                }

                // Additive steps based on feature range
                if feat_max[dim] > 0.0 {
                    let target_contrib = 1.0; // target contribution ~1.0
                    let base_step = target_contrib / feat_max[dim];
                    for &mult in &[0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0] {
                        weights[dim] = base_step * mult * step_scale;
                        let srocc = eval_srocc(human_scores, features, &weights);
                        if srocc > best_local_srocc {
                            best_local_srocc = srocc;
                            best_val = weights[dim];
                            improved = true;
                        }
                    }
                }

                weights[dim] = best_val;
            }

            if !improved {
                break;
            }
        }

        // Enforce non-negativity
        for w in weights.iter_mut() {
            *w = w.max(0.0);
        }

        let srocc = eval_srocc(human_scores, features, &weights);
        if srocc > best_srocc {
            best_srocc = srocc;
            best_weights = weights;
        }
        println!("  Restart {}: SROCC = {:.4}", restart, srocc);
    }

    println!("Best training SROCC: {:.4}", best_srocc);
    best_weights
}

fn eval_srocc(human_scores: &[f64], features: &[&[f64]], weights: &[f64]) -> f64 {
    let predicted: Vec<f64> = features.iter()
        .map(|f| zensim::score_from_features(f, weights).0)
        .collect();
    spearman_correlation(human_scores, &predicted)
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

        // Extract reference image number: I01_01_1.bmp → reference_images/I01.BMP
        // Filenames have inconsistent case (I01 vs i01), references are uppercase
        let ref_num = filename[..3].to_uppercase();
        let ref_path = base.join("reference_images").join(format!("{}.BMP", ref_num));

        // Distorted images also have inconsistent case
        let dist_path = base.join("distorted_images").join(filename);
        let dist_path = if dist_path.exists() {
            dist_path
        } else {
            // Try case variations
            let upper = base.join("distorted_images").join(filename.to_uppercase());
            if upper.exists() { upper } else { dist_path }
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

    // Use 'all_by_image' sheet which has all pairs
    let range = workbook.worksheet_range("all_by_image")
        .expect("Failed to read 'all_by_image' sheet");

    // Mapping from xlsx distortion type to (directory name, filename label)
    let dist_map: HashMap<&str, (&str, &str)> = [
        ("noise", ("awgn", "AWGN")),
        ("blur", ("blur", "BLUR")),
        ("contrast", ("contrast", "contrast")),
        ("fnoise", ("fnoise", "fnoise")),
        ("jpeg", ("jpeg", "JPEG")),
        ("jpeg 2000", ("jpeg2000", "jpeg2000")),
    ].into_iter().collect();

    // Calamine strips leading empty columns. Actual layout:
    // [0]=image, [1]=dst_idx, [2]=dst_type, [3]=dst_lev, [4]=dmos_std, [5]=dmos
    for row in range.rows() {
        if row.len() < 6 {
            continue;
        }

        let img_name = match &row[0] {
            calamine::Data::String(s) => s.clone(),
            calamine::Data::Float(f) => format!("{}", *f as i64),
            calamine::Data::Int(i) => format!("{}", i),
            _ => continue,
        };

        // Skip header row
        if img_name == "image" {
            continue;
        }

        let dst_type = match &row[2] {
            calamine::Data::String(s) => s.clone(),
            _ => continue,
        };

        let dst_lev = match &row[3] {
            calamine::Data::Float(f) => *f as i64,
            calamine::Data::Int(i) => *i,
            _ => continue,
        };

        let dmos = match &row[5] {
            calamine::Data::Float(f) => *f,
            _ => continue,
        };

        let (dir_name, file_label) = match dist_map.get(dst_type.as_str()) {
            Some(v) => *v,
            None => {
                eprintln!("Unknown CSIQ distortion type: {}", dst_type);
                continue;
            }
        };

        let ref_path = base.join(format!("{}.png", img_name));
        let dist_path = base.join(dir_name)
            .join(format!("{}.{}.{}.png", img_name, file_label, dst_lev));

        // CSIQ DMOS: 0-1 scale (lower = better quality, higher = more distortion)
        pairs.push(ImagePair {
            reference: ref_path,
            distorted: dist_path,
            human_score: 1.0 - dmos, // Invert so higher = better
        });
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
