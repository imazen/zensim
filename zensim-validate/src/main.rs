use calamine::{Reader, Xlsx};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "zensim-validate",
    about = "Validate zensim against human quality ratings"
)]
struct Args {
    /// Dataset directory (e.g., /mnt/v/dataset/tid2013)
    #[arg(long)]
    dataset: PathBuf,

    /// Dataset type
    #[arg(long, value_enum)]
    format: DatasetFormat,

    /// Additional datasets for combined training (format: type:path,type:path)
    #[arg(long)]
    also: Option<String>,

    /// Max images to process (0 = all)
    #[arg(long, default_value = "0")]
    max_images: usize,

    /// Train weights: run Nelder-Mead optimization to find best feature weights
    #[arg(long, default_value = "false")]
    train: bool,

    /// Output features CSV for external analysis
    #[arg(long)]
    features_csv: Option<PathBuf>,

    /// Box blur passes (1, 2, or 3, default: 1). 1 = rectangular, 2 = triangular, 3 ≈ Gaussian.
    #[arg(long, default_value = "1")]
    blur_passes: u8,

    /// Box blur radius at scale 0 (default: 5, giving 11-pixel kernel)
    #[arg(long, default_value = "5")]
    blur_radius: usize,

    /// During training, only tune weights that are already nonzero in WEIGHTS.
    /// Prevents activating new channels (which would add blur operations).
    #[arg(long, default_value = "false")]
    sparse: bool,

    /// Force compute_all_features=true even without --train.
    /// Useful for exporting full feature vectors for offline analysis.
    #[arg(long, default_value = "false")]
    compute_all: bool,

    /// K-fold cross-validation by reference image (e.g., --cross-validate 5)
    #[arg(long)]
    cross_validate: Option<usize>,

    /// Leave-one-dataset-out cross-validation (requires --also)
    #[arg(long, default_value = "false")]
    leave_one_out: bool,

    /// Local contrast masking strength (0 = disabled, 2-8 typical)
    #[arg(long, default_value = "0")]
    masking: f32,

    /// Number of downscale levels (default: 4, max: 6)
    #[arg(long, default_value = "4")]
    num_scales: usize,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum DatasetFormat {
    Tid2013,
    Kadid10k,
    Csiq,
    Pipal,
    Cid22,
}

/// A single reference-distorted pair with human score.
#[derive(Debug, Clone)]
struct ImagePair {
    reference: PathBuf,
    distorted: PathBuf,
    /// Human subjective score (higher = better quality, normalized to 0-1)
    human_score: f64,
}

/// A dataset with precomputed features, ready for CV splitting.
struct DatasetWithFeatures {
    name: String,
    human_scores: Vec<f64>,
    features: Vec<Vec<f64>>,
    ref_keys: Vec<String>,
}

fn main() {
    let args = Args::parse();

    if args.leave_one_out && args.also.is_none() {
        eprintln!("--leave-one-out requires --also to specify additional datasets");
        std::process::exit(1);
    }

    let cv_mode = args.cross_validate.is_some() || args.leave_one_out;
    let compute_all = args.train || args.compute_all || cv_mode;
    let blur_passes = args.blur_passes;
    let blur_radius = args.blur_radius;
    let masking_strength = args.masking;
    let num_scales = args.num_scales;

    // Load and compute primary dataset
    let primary = load_and_compute(
        &format!("{:?}", args.format),
        args.format,
        &args.dataset,
        args.max_images,
        compute_all,
        blur_passes,
        blur_radius,
        masking_strength,
        num_scales,
    );

    let n_features = if primary.features.is_empty() {
        eprintln!("No valid results from primary dataset");
        return;
    } else {
        primary.features[0].len()
    };

    // Build frozen mask (expanded to match feature count)
    let embedded_w = expand_embedded_weights(n_features);
    let frozen: Vec<bool> = if args.sparse {
        let fpc = 13;
        embedded_w
            .iter()
            .enumerate()
            .map(|(i, w)| {
                let pos = i % fpc;
                // Keep new features (ssim_2nd=2, art_2nd=5, det_2nd=8, contrast_increase=12)
                // and existing trainable features (mse=9, variance_loss=10, texture_loss=11) unfrozen
                let is_new_or_trainable = matches!(pos, 2 | 5 | 8 | 9 | 10 | 11 | 12);
                !is_new_or_trainable && w.abs() < 0.001
            })
            .collect()
    } else {
        vec![false; n_features]
    };
    if args.sparse {
        let active = frozen.iter().filter(|f| !**f).count();
        println!(
            "Sparse mode: optimizing {} of {} weights (freezing {} at zero)",
            active,
            n_features,
            n_features - active
        );
    }

    // Load additional datasets if specified
    let mut all_datasets = vec![primary];
    if let Some(ref also_str) = args.also {
        for spec in also_str.split(',') {
            let parts: Vec<&str> = spec.splitn(2, ':').collect();
            if parts.len() != 2 {
                eprintln!("Invalid --also format: {}. Expected type:path", spec);
                continue;
            }
            let fmt = match parts[0] {
                "tid2013" => DatasetFormat::Tid2013,
                "kadid10k" => DatasetFormat::Kadid10k,
                "csiq" => DatasetFormat::Csiq,
                "pipal" => DatasetFormat::Pipal,
                "cid22" => DatasetFormat::Cid22,
                _ => {
                    eprintln!("Unknown format: {}", parts[0]);
                    continue;
                }
            };
            let ds = load_and_compute(
                parts[0],
                fmt,
                Path::new(parts[1]),
                0,
                compute_all,
                blur_passes,
                blur_radius,
                masking_strength,
                num_scales,
            );
            all_datasets.push(ds);
        }
    }

    // Cross-validation modes
    if let Some(k) = args.cross_validate {
        if k < 2 {
            eprintln!("--cross-validate requires K >= 2");
            std::process::exit(1);
        }
        for ds in &all_datasets {
            run_kfold_cv(ds, k, n_features, &frozen);
        }
        return;
    }

    if args.leave_one_out {
        run_leave_one_out(&all_datasets, n_features, &frozen);
        return;
    }

    // Normal mode: report correlations on primary dataset
    let ds = &all_datasets[0];
    report_embedded_correlations(ds);

    // Output features CSV if requested
    if let Some(ref csv_path) = args.features_csv {
        write_features_csv(csv_path, &ds.human_scores, &ds.features);
    }

    // Train weights if requested
    if args.train {
        let dataset_groups: Vec<(String, Vec<f64>, Vec<Vec<f64>>)> = all_datasets
            .iter()
            .map(|ds| {
                (
                    ds.name.clone(),
                    ds.human_scores.clone(),
                    ds.features.clone(),
                )
            })
            .collect();

        if dataset_groups.len() == 1 {
            let feats: Vec<&[f64]> = dataset_groups[0].2.iter().map(|v| v.as_slice()).collect();
            println!(
                "Training weights on {} pairs with {} features...",
                dataset_groups[0].1.len(),
                n_features
            );
            let best_weights = train_weights(&dataset_groups[0].1, &feats, n_features, &frozen);
            print_trained_results(&dataset_groups[0].1, &feats, &best_weights);
        } else {
            println!(
                "\nMulti-dataset training on {} datasets...",
                dataset_groups.len()
            );
            let best_weights = train_weights_multi(&dataset_groups, n_features, &frozen);

            for (name, h, f) in &dataset_groups {
                let feats: Vec<&[f64]> = f.iter().map(|v| v.as_slice()).collect();
                let trained_scores: Vec<f64> = feats
                    .iter()
                    .map(|feat| zensim::score_from_features(feat, &best_weights).0)
                    .collect();
                let srocc = spearman_correlation(h, &trained_scores);
                println!("  {}: SROCC = {:.4}", name, srocc);
            }
            print_weights(&best_weights);
        }
    }
}

/// Extract reference image key from a pair's reference path (file stem).
fn reference_key(pair: &ImagePair) -> String {
    pair.reference
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default()
}

/// Load a dataset and compute all features in parallel.
#[allow(clippy::too_many_arguments)]
fn load_and_compute(
    name: &str,
    format: DatasetFormat,
    path: &Path,
    max_images: usize,
    compute_all: bool,
    blur_passes: u8,
    blur_radius: usize,
    masking_strength: f32,
    num_scales: usize,
) -> DatasetWithFeatures {
    let pairs = match format {
        DatasetFormat::Tid2013 => load_tid2013(path),
        DatasetFormat::Kadid10k => load_kadid10k(path),
        DatasetFormat::Csiq => load_csiq(path),
        DatasetFormat::Pipal => load_pipal(path),
        DatasetFormat::Cid22 => load_cid22(path),
    };

    let pairs = if max_images > 0 && max_images < pairs.len() {
        pairs[..max_images].to_vec()
    } else {
        pairs
    };

    println!("Loading {}: {} image pairs...", name, pairs.len());

    let pb = ProgressBar::new(pairs.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({per_sec}) {msg}")
            .unwrap(),
    );

    let results: Vec<(String, f64, zensim::ZensimResult)> = pairs
        .par_iter()
        .map(|pair| {
            let key = reference_key(pair);
            let result = compute_pair_result(
                pair,
                compute_all,
                blur_passes,
                blur_radius,
                masking_strength,
                num_scales,
            );
            pb.inc(1);
            (key, pair.human_score, result)
        })
        .collect();

    pb.finish_with_message("done");

    let mut human_scores = Vec::new();
    let mut features = Vec::new();
    let mut ref_keys = Vec::new();
    let mut n_valid = 0;

    for (key, hs, result) in results {
        if result.score.is_finite() {
            human_scores.push(hs);
            features.push(result.features);
            ref_keys.push(key);
            n_valid += 1;
        }
    }

    println!("  {} valid pairs from {}", n_valid, name);

    DatasetWithFeatures {
        name: name.to_string(),
        human_scores,
        features,
        ref_keys,
    }
}

/// Expand embedded WEIGHTS (156 entries, 13 features/ch) to match a wider feature layout.
/// When extra scales are used, pads with zeros for extra scale features.
fn expand_embedded_weights(n_features: usize) -> Vec<f64> {
    let embedded = &zensim::WEIGHTS;
    if n_features == embedded.len() {
        return embedded.to_vec();
    }

    let mut expanded = vec![0.0; n_features];

    // Copy what fits from embedded weights
    let copy_len = n_features.min(embedded.len());
    expanded[..copy_len].copy_from_slice(&embedded[..copy_len]);

    expanded
}

/// Report correlations using embedded WEIGHTS.
fn report_embedded_correlations(ds: &DatasetWithFeatures) {
    let ew = expand_embedded_weights(ds.features[0].len());
    let metric_scores: Vec<f64> = ds
        .features
        .iter()
        .map(|f| zensim::score_from_features(f, &ew).0)
        .collect();

    let srocc = spearman_correlation(&ds.human_scores, &metric_scores);
    let plcc = pearson_correlation(&ds.human_scores, &metric_scores);
    let krocc = kendall_correlation(&ds.human_scores, &metric_scores);

    println!(
        "\n=== {} — Correlation with Human Ratings (embedded weights) ===",
        ds.name
    );
    println!("SROCC (Spearman):  {:.4}", srocc);
    println!("PLCC  (Pearson):   {:.4}", plcc);
    println!("KROCC (Kendall):   {:.4}", krocc);

    let min_m = metric_scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_m = metric_scores
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let mean_m: f64 = metric_scores.iter().sum::<f64>() / metric_scores.len() as f64;
    println!(
        "Metric score range: {:.2} to {:.2}, mean: {:.2}",
        min_m, max_m, mean_m
    );

    let raw_dists: Vec<f64> = ds
        .features
        .iter()
        .map(|f| zensim::score_from_features(f, &ew).1)
        .collect();
    let min_d = raw_dists.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_d = raw_dists.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_d: f64 = raw_dists.iter().sum::<f64>() / raw_dists.len() as f64;
    let mut sorted_d = raw_dists.clone();
    sorted_d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p10 = sorted_d[sorted_d.len() / 10];
    let p50 = sorted_d[sorted_d.len() / 2];
    let p90 = sorted_d[sorted_d.len() * 9 / 10];
    println!(
        "Raw distance: min={:.3}, p10={:.3}, p50={:.3}, p90={:.3}, max={:.3}, mean={:.3}\n",
        min_d, p10, p50, p90, max_d, mean_d
    );
}

/// Deterministic shuffle + round-robin split of reference keys into K folds.
fn make_folds(ref_keys: &[String], k: usize, seed: u64) -> Vec<Vec<String>> {
    // Collect unique keys preserving discovery order for determinism
    let mut seen = HashMap::new();
    let mut unique_keys = Vec::new();
    for key in ref_keys {
        if seen.insert(key.clone(), ()).is_none() {
            unique_keys.push(key.clone());
        }
    }

    // Fisher-Yates shuffle with LCG
    let mut rng_state = seed;
    let mut next_rand = |bound: usize| -> usize {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng_state >> 33) as usize) % bound
    };

    for i in (1..unique_keys.len()).rev() {
        let j = next_rand(i + 1);
        unique_keys.swap(i, j);
    }

    // Round-robin into folds
    let mut folds: Vec<Vec<String>> = (0..k).map(|_| Vec::new()).collect();
    for (i, key) in unique_keys.into_iter().enumerate() {
        folds[i % k].push(key);
    }

    folds
}

/// Run K-fold cross-validation on a single dataset, splitting by reference image.
fn run_kfold_cv(ds: &DatasetWithFeatures, k: usize, n_features: usize, frozen: &[bool]) {
    let folds = make_folds(&ds.ref_keys, k, 42);

    // Build a lookup: ref_key → set of indices
    let mut key_to_indices: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, key) in ds.ref_keys.iter().enumerate() {
        key_to_indices.entry(key.as_str()).or_default().push(i);
    }

    let n_refs: usize = folds.iter().map(|f| f.len()).sum();
    println!(
        "\n=== {}: {}-fold CV ({} reference images, {} pairs) ===",
        ds.name,
        k,
        n_refs,
        ds.human_scores.len()
    );

    let embedded_w = expand_embedded_weights(n_features);
    let mut trained_sroccs = Vec::new();
    let mut embedded_sroccs = Vec::new();

    for (fold_idx, fold_keys) in folds.iter().enumerate() {
        // Test set = this fold's ref keys
        let test_keys: std::collections::HashSet<&str> =
            fold_keys.iter().map(|s| s.as_str()).collect();

        let mut train_h = Vec::new();
        let mut train_f: Vec<Vec<f64>> = Vec::new();
        let mut test_h = Vec::new();
        let mut test_f: Vec<Vec<f64>> = Vec::new();

        for (i, key) in ds.ref_keys.iter().enumerate() {
            if test_keys.contains(key.as_str()) {
                test_h.push(ds.human_scores[i]);
                test_f.push(ds.features[i].clone());
            } else {
                train_h.push(ds.human_scores[i]);
                train_f.push(ds.features[i].clone());
            }
        }

        // Train on K-1 folds
        let train_slices: Vec<&[f64]> = train_f.iter().map(|v| v.as_slice()).collect();
        let weights = train_weights(&train_h, &train_slices, n_features, frozen);

        // Evaluate trained weights on held-out fold
        let test_slices: Vec<&[f64]> = test_f.iter().map(|v| v.as_slice()).collect();
        let trained_srocc = eval_srocc(&test_h, &test_slices, &weights);
        trained_sroccs.push(trained_srocc);

        // Evaluate embedded WEIGHTS on held-out fold (baseline)
        let embedded_srocc = eval_srocc(&test_h, &test_slices, &embedded_w);
        embedded_sroccs.push(embedded_srocc);

        println!(
            "  Fold {}/{}: train={} test={} | trained SROCC={:.4} | embedded SROCC={:.4}",
            fold_idx + 1,
            k,
            train_h.len(),
            test_h.len(),
            trained_srocc,
            embedded_srocc,
        );
    }

    print_cv_summary(&ds.name, &trained_sroccs, &embedded_sroccs);
}

/// Leave-one-dataset-out cross-validation.
fn run_leave_one_out(datasets: &[DatasetWithFeatures], n_features: usize, frozen: &[bool]) {
    println!(
        "\n=== Leave-one-dataset-out CV ({} datasets) ===",
        datasets.len()
    );

    let embedded_w = expand_embedded_weights(n_features);
    let mut trained_sroccs = Vec::new();
    let mut embedded_sroccs = Vec::new();

    for held_out_idx in 0..datasets.len() {
        // Train on all except held-out
        let mut train_groups: Vec<(String, Vec<f64>, Vec<Vec<f64>>)> = Vec::new();
        for (i, ds) in datasets.iter().enumerate() {
            if i != held_out_idx {
                train_groups.push((
                    ds.name.clone(),
                    ds.human_scores.clone(),
                    ds.features.clone(),
                ));
            }
        }

        let weights = if train_groups.len() == 1 {
            let feats: Vec<&[f64]> = train_groups[0].2.iter().map(|v| v.as_slice()).collect();
            train_weights(&train_groups[0].1, &feats, n_features, frozen)
        } else {
            train_weights_multi(&train_groups, n_features, frozen)
        };

        // Evaluate on held-out dataset
        let test = &datasets[held_out_idx];
        let test_slices: Vec<&[f64]> = test.features.iter().map(|v| v.as_slice()).collect();
        let trained_srocc = eval_srocc(&test.human_scores, &test_slices, &weights);
        trained_sroccs.push(trained_srocc);

        let embedded_srocc = eval_srocc(&test.human_scores, &test_slices, &embedded_w);
        embedded_sroccs.push(embedded_srocc);

        let train_names: Vec<&str> = train_groups.iter().map(|(n, _, _)| n.as_str()).collect();
        println!(
            "  Held out {}: trained on [{}] → SROCC={:.4} | embedded SROCC={:.4}",
            test.name,
            train_names.join(", "),
            trained_srocc,
            embedded_srocc,
        );
    }

    print_cv_summary("LODO", &trained_sroccs, &embedded_sroccs);
}

fn print_cv_summary(label: &str, trained_sroccs: &[f64], embedded_sroccs: &[f64]) {
    let n = trained_sroccs.len() as f64;
    let trained_mean = trained_sroccs.iter().sum::<f64>() / n;
    let embedded_mean = embedded_sroccs.iter().sum::<f64>() / n;

    let trained_std = (trained_sroccs
        .iter()
        .map(|x| (x - trained_mean).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    let embedded_std = (embedded_sroccs
        .iter()
        .map(|x| (x - embedded_mean).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();

    println!("\n  {} summary:", label);
    println!(
        "    Trained:  mean SROCC = {:.4} ± {:.4}",
        trained_mean, trained_std
    );
    println!(
        "    Embedded: mean SROCC = {:.4} ± {:.4}",
        embedded_mean, embedded_std
    );

    let overfit_gap = trained_mean - embedded_mean;
    if overfit_gap.abs() > 0.001 {
        println!(
            "    Gap (trained - embedded): {:+.4}  {}",
            overfit_gap,
            if overfit_gap > 0.01 {
                "← trained weights generalize better on holdout"
            } else if overfit_gap < -0.01 {
                "← embedded weights generalize better on holdout"
            } else {
                ""
            }
        );
    }
}

fn compute_pair_result(
    pair: &ImagePair,
    compute_all_features: bool,
    blur_passes: u8,
    blur_radius: usize,
    masking_strength: f32,
    num_scales: usize,
) -> zensim::ZensimResult {
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
        blur_passes,
        blur_radius,
        masking_strength,
        num_scales,
    };
    match zensim::compute_zensim_with_config(
        &src_pixels,
        &dst_pixels,
        w as usize,
        h as usize,
        config,
    ) {
        Ok(r) => r,
        Err(_) => nan_result,
    }
}

fn write_features_csv(path: &Path, human_scores: &[f64], features: &[Vec<f64>]) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("Failed to create features CSV");

    let n_features = features[0].len();
    write!(f, "human_score,metric_score,raw_distance").unwrap();
    for i in 0..n_features {
        write!(f, ",f{}", i).unwrap();
    }
    writeln!(f).unwrap();

    let ew = expand_embedded_weights(n_features);
    for (human, feat) in human_scores.iter().zip(features) {
        let (score, raw) = zensim::score_from_features(feat, &ew);
        write!(f, "{},{},{}", human, score, raw).unwrap();
        for v in feat {
            write!(f, ",{}", v).unwrap();
        }
        writeln!(f).unwrap();
    }
    println!("Wrote features to {:?}", path);
}

/// Train weights using coordinate descent with random restarts.
/// Maximizes SROCC between predicted scores and human scores.
fn train_weights(
    human_scores: &[f64],
    features: &[&[f64]],
    n_features: usize,
    frozen: &[bool],
) -> Vec<f64> {
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
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng_state >> 33) as f64 / (u32::MAX as f64)
    };

    for restart in 0..n_restarts {
        let mut weights = if restart == 0 {
            // Start from current embedded weights (best known solution)
            expand_embedded_weights(n_features)
        } else if restart == 1 {
            // Start from uniform weights (respecting frozen mask)
            (0..n_features)
                .map(|i| {
                    if frozen[i] {
                        0.0
                    } else {
                        1.0 / n_features as f64
                    }
                })
                .collect()
        } else {
            // Sparse random: only activate ~30% of non-frozen features
            (0..n_features)
                .map(|i| {
                    if frozen[i] {
                        0.0
                    } else if next_rand() < 0.3 {
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
                if frozen[dim] {
                    weights[dim] = 0.0;
                    continue;
                }
                let current_val = weights[dim];
                let mut best_val = current_val;
                let mut best_local_srocc = eval_srocc(human_scores, features, &weights);

                // Multiplicative steps
                for &mult in &[0.0, 0.5, 1.5, 2.0, 3.0, 0.1, 5.0, 10.0] {
                    weights[dim] =
                        current_val * mult * step_scale + current_val * (1.0 - step_scale);
                    if weights[dim] < 0.0 {
                        weights[dim] = 0.0;
                    }
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

fn print_trained_results(human_scores: &[f64], feats: &[&[f64]], weights: &[f64]) {
    let trained_scores: Vec<f64> = feats
        .iter()
        .map(|f| zensim::score_from_features(f, weights).0)
        .collect();

    let srocc = spearman_correlation(human_scores, &trained_scores);
    let plcc = pearson_correlation(human_scores, &trained_scores);
    let krocc = kendall_correlation(human_scores, &trained_scores);

    println!("\n=== Correlation with Trained Weights ===");
    println!("SROCC (Spearman):  {:.4}", srocc);
    println!("PLCC  (Pearson):   {:.4}", plcc);
    println!("KROCC (Kendall):   {:.4}", krocc);
    print_weights(weights);
}

fn print_weights(weights: &[f64]) {
    let features_per_ch = zensim::FEATURES_PER_SCALE / 3;
    println!("\n// Trained weights ({} values):", weights.len());
    println!("const TRAINED_WEIGHTS: [f64; {}] = [", weights.len());
    for (i, w) in weights.iter().enumerate() {
        if i % features_per_ch == 0 {
            let scale = i / zensim::FEATURES_PER_SCALE;
            let ch = (i % zensim::FEATURES_PER_SCALE) / features_per_ch;
            let ch_name = ["X", "Y", "B"][ch];
            print!("    // Scale {} Channel {}\n    ", scale, ch_name);
        }
        print!("{:.6}, ", w);
        if i % features_per_ch == features_per_ch - 1 {
            println!();
        }
    }
    println!("];");
}

/// Train weights on multiple datasets, maximizing average SROCC.
#[allow(clippy::type_complexity)]
fn train_weights_multi(
    datasets: &[(String, Vec<f64>, Vec<Vec<f64>>)],
    n_features: usize,
    frozen: &[bool],
) -> Vec<f64> {
    let mut best_weights = vec![1.0; n_features];
    let mut best_avg_srocc = -1.0f64;

    // Prepare feature slices for each dataset
    let dataset_slices: Vec<(Vec<f64>, Vec<Vec<f64>>)> = datasets
        .iter()
        .map(|(_, h, f)| (h.clone(), f.clone()))
        .collect();

    // Compute feature ranges across all datasets
    let mut feat_max = vec![0.0f64; n_features];
    for (_, feats) in &dataset_slices {
        for f in feats {
            for (i, &val) in f.iter().enumerate() {
                feat_max[i] = feat_max[i].max(val.abs());
            }
        }
    }

    let eval_multi = |weights: &[f64]| -> f64 {
        let mut sum_srocc = 0.0;
        for (human, feats) in &dataset_slices {
            let feat_slices: Vec<&[f64]> = feats.iter().map(|v| v.as_slice()).collect();
            sum_srocc += eval_srocc(human, &feat_slices, weights);
        }
        sum_srocc / dataset_slices.len() as f64
    };

    let n_restarts = 10;
    let mut rng_state = 42u64;
    let mut next_rand = || -> f64 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng_state >> 33) as f64 / (u32::MAX as f64)
    };

    for restart in 0..n_restarts {
        let mut weights = if restart == 0 {
            expand_embedded_weights(n_features)
        } else if restart == 1 {
            (0..n_features)
                .map(|i| {
                    if frozen[i] {
                        0.0
                    } else {
                        1.0 / n_features as f64
                    }
                })
                .collect()
        } else {
            (0..n_features)
                .map(|i| {
                    if frozen[i] {
                        0.0
                    } else if next_rand() < 0.3 {
                        next_rand() * 10.0
                    } else {
                        0.0
                    }
                })
                .collect()
        };

        for iter in 0..50 {
            let mut improved = false;
            let step_scale = if iter < 20 { 1.0 } else { 0.5 };

            for dim in 0..n_features {
                if frozen[dim] {
                    weights[dim] = 0.0;
                    continue;
                }
                let current_val = weights[dim];
                let mut best_val = current_val;
                let mut best_local = eval_multi(&weights);

                for &mult in &[0.0, 0.5, 1.5, 2.0, 3.0, 0.1, 5.0, 10.0] {
                    weights[dim] =
                        current_val * mult * step_scale + current_val * (1.0 - step_scale);
                    if weights[dim] < 0.0 {
                        weights[dim] = 0.0;
                    }
                    let score = eval_multi(&weights);
                    if score > best_local {
                        best_local = score;
                        best_val = weights[dim];
                        improved = true;
                    }
                }

                if feat_max[dim] > 0.0 {
                    let base_step = 1.0 / feat_max[dim];
                    for &mult in &[0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0] {
                        weights[dim] = base_step * mult * step_scale;
                        let score = eval_multi(&weights);
                        if score > best_local {
                            best_local = score;
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

        for w in weights.iter_mut() {
            *w = w.max(0.0);
        }

        let avg = eval_multi(&weights);
        if avg > best_avg_srocc {
            best_avg_srocc = avg;
            best_weights = weights;
        }
        println!("  Restart {}: avg SROCC = {:.4}", restart, avg);
    }

    println!("Best multi-dataset avg SROCC: {:.4}", best_avg_srocc);
    best_weights
}

fn eval_srocc(human_scores: &[f64], features: &[&[f64]], weights: &[f64]) -> f64 {
    let predicted: Vec<f64> = features
        .iter()
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
        let ref_path = base
            .join("reference_images")
            .join(format!("{}.BMP", ref_num));

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
    let mut workbook: Xlsx<_> =
        calamine::open_workbook(xlsx_path).expect("Failed to open CSIQ DMOS xlsx");

    let mut pairs = Vec::new();

    // Use 'all_by_image' sheet which has all pairs
    let range = workbook
        .worksheet_range("all_by_image")
        .expect("Failed to read 'all_by_image' sheet");

    // Mapping from xlsx distortion type to (directory name, filename label)
    let dist_map: HashMap<&str, (&str, &str)> = [
        ("noise", ("awgn", "AWGN")),
        ("blur", ("blur", "BLUR")),
        ("contrast", ("contrast", "contrast")),
        ("fnoise", ("fnoise", "fnoise")),
        ("jpeg", ("jpeg", "JPEG")),
        ("jpeg 2000", ("jpeg2000", "jpeg2000")),
    ]
    .into_iter()
    .collect();

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
        let dist_path = base
            .join(dir_name)
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

fn load_pipal(base: &Path) -> Vec<ImagePair> {
    let label_dir = base.join("Train_Label");
    let ref_dir = base.join("Train_Ref");

    if !label_dir.exists() {
        eprintln!("Cannot find Train_Label/ in {:?}", base);
        return vec![];
    }
    if !ref_dir.exists() {
        eprintln!("Cannot find Train_Ref/ in {:?}", base);
        return vec![];
    }

    // Distorted images are split across Distortion_1..4
    let dist_dirs: Vec<PathBuf> = (1..=4)
        .map(|i| base.join(format!("Distortion_{}", i)))
        .filter(|d| d.exists())
        .collect();

    if dist_dirs.is_empty() {
        eprintln!("Cannot find any Distortion_N/ directories in {:?}", base);
        return vec![];
    }

    // Read MOS range to normalize: PIPAL uses Elo-like scores (~900-1850)
    let mut all_scores: Vec<(PathBuf, PathBuf, f64, String)> = Vec::new();

    let mut label_files: Vec<_> = std::fs::read_dir(&label_dir)
        .expect("Failed to read Train_Label directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "txt"))
        .collect();
    label_files.sort_by_key(|e| e.file_name());

    for entry in &label_files {
        let label_path = entry.path();
        let ref_stem = label_path.file_stem().unwrap().to_string_lossy();
        let ref_path = ref_dir.join(format!("{}.bmp", ref_stem));

        if !ref_path.exists() {
            continue;
        }

        let content = match std::fs::read_to_string(&label_path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.splitn(2, ',').collect();
            if parts.len() != 2 {
                continue;
            }
            let dist_name = parts[0].trim();
            let mos: f64 = match parts[1].trim().parse() {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Find the distorted image across distortion directories
            let dist_path = dist_dirs
                .iter()
                .map(|d| d.join(dist_name))
                .find(|p| p.exists());

            if let Some(dist_path) = dist_path {
                all_scores.push((ref_path.clone(), dist_path, mos, ref_stem.to_string()));
            }
        }
    }

    if all_scores.is_empty() {
        eprintln!("No valid PIPAL pairs found");
        return vec![];
    }

    // Normalize MOS to 0-1: PIPAL uses Elo scores where higher = better
    let min_mos = all_scores
        .iter()
        .map(|(_, _, m, _)| *m)
        .fold(f64::INFINITY, f64::min);
    let max_mos = all_scores
        .iter()
        .map(|(_, _, m, _)| *m)
        .fold(f64::NEG_INFINITY, f64::max);
    let range = (max_mos - min_mos).max(1.0);

    let pairs: Vec<ImagePair> = all_scores
        .into_iter()
        .map(|(reference, distorted, mos, _)| ImagePair {
            reference,
            distorted,
            human_score: (mos - min_mos) / range,
        })
        .collect();

    println!(
        "  PIPAL: {} pairs, MOS range {:.1}..{:.1}",
        pairs.len(),
        min_mos,
        max_mos
    );

    pairs
}

fn load_cid22(base: &Path) -> Vec<ImagePair> {
    let csv_path = base.join("CID22_validation_set.csv");
    if !csv_path.exists() {
        eprintln!("Cannot find CID22_validation_set.csv in {:?}", base);
        return vec![];
    }

    let mut rdr =
        csv::Reader::from_path(&csv_path).expect("Failed to open CID22_validation_set.csv");
    let mut pairs = Vec::new();

    for result in rdr.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };
        // CID22 CSV: reference_img, distorted_img, encoder, setting, bpp, MCOS, RMOS, Elo, nb_pc_opinions
        if record.len() < 6 {
            continue;
        }

        let ref_name = &record[0];
        let dist_name = &record[1];
        let encoder = &record[2];

        // Skip self-reference rows
        if encoder == "Reference" {
            continue;
        }

        let mcos: f64 = match record[5].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        let ref_path = base.join(ref_name);
        let dist_path = base.join(dist_name);

        // MCOS: 0-100 scale (higher = better quality)
        pairs.push(ImagePair {
            reference: ref_path,
            distorted: dist_path,
            human_score: mcos / 100.0,
        });
    }

    println!("  CID22: {} pairs", pairs.len());
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
