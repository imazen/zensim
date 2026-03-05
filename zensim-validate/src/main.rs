#![allow(clippy::needless_range_loop)] // Training loops index parallel arrays by shared index

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
    /// Dataset directory (e.g., ./datasets/tid2013)
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

    /// Compute extended features (25 per channel instead of 13).
    /// Adds masked SSIM/edge/MSE + max/p95 percentile features.
    #[arg(long, default_value = "false")]
    extended_features: bool,

    /// Masking strength for extended features (default: 4.0).
    /// Only used when --extended-features is set.
    #[arg(long, default_value = "4.0")]
    extended_masking_strength: f32,

    /// Downscale filter for pyramid construction: box, mitchell, lanczos
    #[arg(long, default_value = "box")]
    downscale_filter: String,

    /// Local contrast masking strength (0 = disabled, 2-8 typical)
    #[arg(long, default_value = "0")]
    masking: f32,

    /// Number of downscale levels (default: 4, max: 6)
    #[arg(long, default_value = "4")]
    num_scales: usize,

    /// Load custom weights from file (one weight per line, 156 values).
    /// Evaluates these weights against the dataset(s) instead of the embedded weights.
    #[arg(long)]
    weights_file: Option<PathBuf>,

    /// Target metric for synthetic datasets (selects which column to use as ground truth)
    #[arg(long, value_enum)]
    target_metric: Option<TargetMetric>,

    /// Feature cache file path. Auto-derived from dataset path if omitted.
    #[arg(long)]
    feature_cache: Option<PathBuf>,

    /// Force recompute features, ignoring any existing cache.
    #[arg(long, default_value = "false")]
    recompute: bool,

    /// Directory for training run logs. Defaults to directory containing the dataset.
    #[arg(long)]
    log_dir: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum TargetMetric {
    /// GPU SSIMULACRA2 (ssimulacra2-cuda)
    GpuSsim2,
    /// GPU Butteraugli (butteraugli-cuda)
    GpuButteraugli,
    /// CPU SSIMULACRA2 (fast-ssim2)
    CpuSsim2,
    /// CPU Butteraugli (butteraugli crate)
    CpuButteraugli,
    /// DSSIM (structural dissimilarity)
    Dssim,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum DatasetFormat {
    Tid2013,
    Kadid10k,
    Csiq,
    Pipal,
    Cid22,
    KonfigIqa,
    Synthetic,
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

struct CacheConfig {
    num_scales: u32,
    blur_passes: u8,
    blur_radius: u32,
    masking_bits: u32,
}

/// Cached features without human scores (which are target-metric-dependent).
struct CachedFeatures {
    name: String,
    features: Vec<Vec<f64>>,
    ref_keys: Vec<String>,
    /// Original pair indices (which pairs from the dataset produced valid features).
    valid_indices: Vec<u32>,
}

fn save_feature_cache(
    path: &Path,
    ds: &DatasetWithFeatures,
    valid_indices: &[u32],
    config: &CacheConfig,
) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);

    // Magic + version (v2: no human_scores, target-metric-independent)
    f.write_all(b"ZSFC")?;
    f.write_all(&2u32.to_le_bytes())?;

    // Validation fields
    f.write_all(&config.num_scales.to_le_bytes())?;
    f.write_all(&[config.blur_passes])?;
    f.write_all(&config.blur_radius.to_le_bytes())?;
    f.write_all(&config.masking_bits.to_le_bytes())?;

    let n_pairs = ds.features.len() as u64;
    let n_features = if ds.features.is_empty() {
        0u64
    } else {
        ds.features[0].len() as u64
    };
    f.write_all(&n_pairs.to_le_bytes())?;
    f.write_all(&n_features.to_le_bytes())?;

    // Dataset name
    let name_bytes = ds.name.as_bytes();
    f.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
    f.write_all(name_bytes)?;

    // Valid pair indices (which original pairs produced non-NaN features)
    for &idx in valid_indices {
        f.write_all(&idx.to_le_bytes())?;
    }

    // Features (flat row-major) — no human_scores, those are target-dependent
    for row in &ds.features {
        for &v in row {
            f.write_all(&v.to_le_bytes())?;
        }
    }

    // Ref keys
    for key in &ds.ref_keys {
        let kb = key.as_bytes();
        f.write_all(&(kb.len() as u32).to_le_bytes())?;
        f.write_all(kb)?;
    }

    f.flush()?;
    Ok(())
}

fn load_feature_cache(
    path: &Path,
    config: &CacheConfig,
) -> std::io::Result<Option<CachedFeatures>> {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };

    let mut pos = 0usize;

    let read_bytes = |pos: &mut usize, n: usize| -> std::io::Result<&[u8]> {
        if *pos + n > data.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "truncated cache file",
            ));
        }
        let slice = &data[*pos..*pos + n];
        *pos += n;
        Ok(slice)
    };

    // Magic
    let magic = read_bytes(&mut pos, 4)?;
    if magic != b"ZSFC" {
        eprintln!("Feature cache: invalid magic, recomputing");
        return Ok(None);
    }

    // Version
    let version = u32::from_le_bytes(read_bytes(&mut pos, 4)?.try_into().unwrap());
    if version != 2 {
        eprintln!(
            "Feature cache: version {} (expected 2), recomputing",
            version
        );
        return Ok(None);
    }

    // Validation fields
    let num_scales = u32::from_le_bytes(read_bytes(&mut pos, 4)?.try_into().unwrap());
    let blur_passes = read_bytes(&mut pos, 1)?[0];
    let blur_radius = u32::from_le_bytes(read_bytes(&mut pos, 4)?.try_into().unwrap());
    let masking_bits = u32::from_le_bytes(read_bytes(&mut pos, 4)?.try_into().unwrap());

    if num_scales != config.num_scales {
        eprintln!(
            "Feature cache: num_scales mismatch (cache={}, current={}), recomputing",
            num_scales, config.num_scales
        );
        return Ok(None);
    }
    if blur_passes != config.blur_passes {
        eprintln!(
            "Feature cache: blur_passes mismatch (cache={}, current={}), recomputing",
            blur_passes, config.blur_passes
        );
        return Ok(None);
    }
    if blur_radius != config.blur_radius {
        eprintln!(
            "Feature cache: blur_radius mismatch (cache={}, current={}), recomputing",
            blur_radius, config.blur_radius
        );
        return Ok(None);
    }
    if masking_bits != config.masking_bits {
        eprintln!(
            "Feature cache: masking mismatch (cache={}, current={}), recomputing",
            f32::from_bits(masking_bits),
            f32::from_bits(config.masking_bits)
        );
        return Ok(None);
    }

    let n_pairs = u64::from_le_bytes(read_bytes(&mut pos, 8)?.try_into().unwrap()) as usize;
    let n_features = u64::from_le_bytes(read_bytes(&mut pos, 8)?.try_into().unwrap()) as usize;

    // Name
    let name_len = u32::from_le_bytes(read_bytes(&mut pos, 4)?.try_into().unwrap()) as usize;
    let name = String::from_utf8_lossy(read_bytes(&mut pos, name_len)?).to_string();

    // Valid pair indices
    let mut valid_indices = Vec::with_capacity(n_pairs);
    for _ in 0..n_pairs {
        valid_indices.push(u32::from_le_bytes(
            read_bytes(&mut pos, 4)?.try_into().unwrap(),
        ));
    }

    // Features (no human_scores in v2)
    let mut features = Vec::with_capacity(n_pairs);
    for _ in 0..n_pairs {
        let mut row = Vec::with_capacity(n_features);
        for _ in 0..n_features {
            row.push(f64::from_le_bytes(
                read_bytes(&mut pos, 8)?.try_into().unwrap(),
            ));
        }
        features.push(row);
    }

    // Ref keys
    let mut ref_keys = Vec::with_capacity(n_pairs);
    for _ in 0..n_pairs {
        let klen = u32::from_le_bytes(read_bytes(&mut pos, 4)?.try_into().unwrap()) as usize;
        let key = String::from_utf8_lossy(read_bytes(&mut pos, klen)?).to_string();
        ref_keys.push(key);
    }

    Ok(Some(CachedFeatures {
        name,
        features,
        ref_keys,
        valid_indices,
    }))
}

fn git_describe() -> String {
    std::process::Command::new("git")
        .args(["describe", "--always", "--dirty"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn utc_timestamp() -> String {
    std::process::Command::new("date")
        .args(["-u", "+%Y%m%dT%H%M%S"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| {
            format!(
                "{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            )
        })
}

fn log_line(msg: &str, log: &mut Vec<String>) {
    println!("{}", msg);
    log.push(msg.to_string());
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
    let extended_features = args.extended_features;
    let extended_masking_strength = args.extended_masking_strength;
    let downscale_filter = match args.downscale_filter.as_str() {
        "box" => zensim::DownscaleFilter::Box2x2,
        #[cfg(feature = "zenresize")]
        "mitchell" => zensim::DownscaleFilter::Mitchell,
        #[cfg(feature = "zenresize")]
        "lanczos" => zensim::DownscaleFilter::Lanczos,
        #[cfg(feature = "zenresize")]
        s if s.starts_with("mitchell-blur") => {
            let sigma = s
                .strip_prefix("mitchell-blur")
                .and_then(|rest| rest.strip_prefix(':'))
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(0.5);
            zensim::DownscaleFilter::MitchellBlur(sigma)
        }
        other => {
            eprintln!(
                "Unknown downscale filter: {other}. Options: box, mitchell, lanczos, mitchell-blur[:sigma] (requires zenresize feature)"
            );
            std::process::exit(1);
        }
    };

    let cache_config = CacheConfig {
        num_scales: num_scales as u32,
        blur_passes,
        blur_radius: blur_radius as u32,
        masking_bits: masking_strength.to_bits(),
    };

    // Load and compute primary dataset (with optional caching)
    // Timestamped cache: saves produce `dataset.csv.features.YYYYMMDD_HHMMSS.bin`,
    // loads glob for `dataset.csv.features.*.bin` and pick the newest.
    let auto_cache_save_path = |dataset_path: &Path| -> PathBuf {
        let now = chrono::Local::now();
        let mut p = dataset_path.as_os_str().to_owned();
        p.push(format!(".features.{}.bin", now.format("%Y%m%d_%H%M%S")));
        PathBuf::from(p)
    };
    let find_latest_cache = |dataset_path: &Path| -> Option<PathBuf> {
        let mut pattern = dataset_path.as_os_str().to_owned();
        pattern.push(".features.*.bin");
        let pattern_str = pattern.to_string_lossy();
        let mut matches: Vec<PathBuf> = glob::glob(&pattern_str)
            .ok()?
            .filter_map(|r| r.ok())
            .collect();
        // Sort by filename descending — ISO timestamps sort lexicographically
        matches.sort();
        matches.pop()
    };
    // Also support legacy non-timestamped cache files
    let auto_cache_legacy_path = |dataset_path: &Path| -> PathBuf {
        let mut p = dataset_path.as_os_str().to_owned();
        p.push(".features.bin");
        PathBuf::from(p)
    };
    let find_cache_to_load = |dataset_path: &Path| -> Option<PathBuf> {
        // Prefer newest timestamped cache, fall back to legacy non-timestamped
        if let Some(p) = find_latest_cache(dataset_path) {
            return Some(p);
        }
        let legacy = auto_cache_legacy_path(dataset_path);
        if legacy.exists() { Some(legacy) } else { None }
    };

    let primary = if compute_all && !args.recompute {
        let explicit_cache = args.feature_cache.clone();
        let load_path = explicit_cache
            .clone()
            .or_else(|| find_cache_to_load(&args.dataset));
        let cache_start = std::time::Instant::now();
        let cached_result =
            load_path
                .as_ref()
                .and_then(|p| match load_feature_cache(p, &cache_config) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("Warning: failed to read cache {:?}: {}", p, e);
                        None
                    }
                });
        // Save path: explicit if given, otherwise timestamped
        let save_path = explicit_cache
            .clone()
            .unwrap_or_else(|| auto_cache_save_path(&args.dataset));
        match cached_result {
            Some(cached) => {
                // Reload pairs for fresh human_scores (target-metric-dependent)
                let pairs = load_pairs(
                    args.format,
                    &args.dataset,
                    args.max_images,
                    args.target_metric,
                );

                // Check if cache covers all pairs — if not, extract missing ones
                let max_cached_idx =
                    cached.valid_indices.iter().copied().max().unwrap_or(0) as usize;
                let cached_count = cached.valid_indices.len();

                if max_cached_idx < pairs.len().saturating_sub(1) && pairs.len() > cached_count {
                    // Incremental: cache is from a smaller dataset, extract new pairs
                    let cached_set: std::collections::HashSet<u32> =
                        cached.valid_indices.iter().copied().collect();
                    let new_pairs: Vec<(usize, ImagePair)> = pairs
                        .iter()
                        .enumerate()
                        .filter(|(idx, p)| {
                            !cached_set.contains(&(*idx as u32)) && !p.human_score.is_nan()
                        })
                        .map(|(idx, p)| (idx, p.clone()))
                        .collect();
                    let n_new = new_pairs.len();

                    println!(
                        "Cache has {} pairs, dataset has {} — extracting {} new pairs",
                        cached_count,
                        pairs.len(),
                        n_new
                    );

                    let ds = build_dataset_from_cache(cached, &pairs);
                    if n_new == 0 {
                        ds
                    } else {
                        // Extract features for new pairs using same logic as load_and_compute
                        let config = zensim::ZensimConfig {
                            compute_all_features: compute_all,
                            extended_features,
                            extended_masking_strength,
                            blur_passes,
                            blur_radius,
                            masking_strength,
                            num_scales,
                            downscale_filter,
                            score_mapping_a: 18.0,
                            score_mapping_b: 0.7,
                            ..Default::default()
                        };
                        let nan_result = zensim::ZensimResult {
                            score: f64::NAN,
                            raw_distance: f64::NAN,
                            features: vec![],
                            profile: zensim::ZensimProfile::PreviewV0_1,
                            mean_offset: [f64::NAN; 3],
                        };

                        // Group new pairs by reference
                        let mut by_ref: std::collections::BTreeMap<
                            PathBuf,
                            Vec<(usize, ImagePair)>,
                        > = std::collections::BTreeMap::new();
                        for (idx, pair) in new_pairs {
                            by_ref
                                .entry(pair.reference.clone())
                                .or_default()
                                .push((idx, pair));
                        }

                        let pb = ProgressBar::new(n_new as u64);
                        pb.set_style(
                            ProgressStyle::default_bar()
                                .template(
                                    "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({per_sec}) {msg}",
                                )
                                .unwrap(),
                        );
                        let progress_ctr = std::sync::atomic::AtomicU64::new(0);
                        let start_t = std::time::Instant::now();
                        let log_int = (n_new / 20).max(1000) as u64;

                        let ref_groups: Vec<(PathBuf, Vec<(usize, ImagePair)>)> =
                            by_ref.into_iter().collect();

                        let group_results: Vec<
                            Vec<(usize, String, f64, zensim::ZensimResult)>,
                        > = ref_groups
                            .par_iter()
                            .map(|(ref_path, group)| {
                                let fail = |grp: &[(usize, ImagePair)]| -> Vec<_> {
                                    progress_ctr.fetch_add(grp.len() as u64, std::sync::atomic::Ordering::Relaxed);
                                    pb.inc(grp.len() as u64);
                                    grp.iter()
                                        .map(|(idx, pair)| {
                                            (
                                                *idx,
                                                reference_key(pair),
                                                pair.human_score,
                                                nan_result.clone(),
                                            )
                                        })
                                        .collect()
                                };

                                let src_img = match image::open(ref_path) {
                                    Ok(img) => img.to_rgb8(),
                                    Err(_) => return fail(group),
                                };
                                let (w, h) = src_img.dimensions();
                                let src_pixels: Vec<[u8; 3]> = src_img
                                    .pixels()
                                    .map(|p| [p.0[0], p.0[1], p.0[2]])
                                    .collect();

                                let needs_full = config.masking_strength > 0.0;
                                let precomputed = if !needs_full {
                                    match zensim::precompute_reference_with_scales(
                                        &src_pixels,
                                        w as usize,
                                        h as usize,
                                        num_scales,
                                    ) {
                                        Ok(p) => Some(p),
                                        Err(_) => return fail(group),
                                    }
                                } else {
                                    None
                                };

                                group
                                    .par_iter()
                                    .map(|(idx, pair)| {
                                        let key = reference_key(pair);
                                        let result = match image::open(&pair.distorted) {
                                            Ok(img) => {
                                                let dst = img.to_rgb8();
                                                let (dw, dh) = dst.dimensions();
                                                if dw != w || dh != h {
                                                    nan_result.clone()
                                                } else {
                                                    let dst_pixels: Vec<[u8; 3]> = dst
                                                        .pixels()
                                                        .map(|p| [p.0[0], p.0[1], p.0[2]])
                                                        .collect();
                                                    if let Some(ref pre) = precomputed {
                                                        zensim::compute_zensim_with_ref_and_config(
                                                            pre,
                                                            &dst_pixels,
                                                            w as usize,
                                                            h as usize,
                                                            config,
                                                        )
                                                        .unwrap_or_else(|_| nan_result.clone())
                                                    } else {
                                                        zensim::compute_zensim_with_config(
                                                            &src_pixels,
                                                            &dst_pixels,
                                                            w as usize,
                                                            h as usize,
                                                            config,
                                                        )
                                                        .unwrap_or_else(|_| nan_result.clone())
                                                    }
                                                }
                                            }
                                            Err(_) => nan_result.clone(),
                                        };
                                        let prev = progress_ctr.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                        pb.inc(1);
                                        let cur = prev + 1;
                                        if prev / log_int != cur / log_int {
                                            let el = start_t.elapsed().as_secs_f64();
                                            let rate = cur as f64 / el;
                                            let eta = (n_new as f64 - cur as f64) / rate;
                                            eprintln!("  [{:.0}s] {}/{} new pairs ({:.1}%), {:.0}/s, ETA {:.0}s", el, cur, n_new, cur as f64 / n_new as f64 * 100.0, rate, eta);
                                        }
                                        (*idx, key, pair.human_score, result)
                                    })
                                    .collect()
                            })
                            .collect();

                        pb.finish_with_message("done");
                        eprintln!(
                            "  New pairs extracted: {} in {:.1}s",
                            n_new,
                            start_t.elapsed().as_secs_f64()
                        );

                        // Merge cached + new features
                        let mut all_human_scores = ds.human_scores;
                        let mut all_features = ds.features;
                        let mut all_ref_keys = ds.ref_keys;
                        let mut all_valid_indices: Vec<u32> = cached_set.into_iter().collect();

                        let mut new_results: Vec<_> = group_results.into_iter().flatten().collect();
                        new_results.sort_by_key(|(idx, _, _, _)| *idx);

                        let mut n_new_valid = 0usize;
                        for (idx, key, hs, result) in new_results {
                            if result.score.is_finite() {
                                all_human_scores.push(hs);
                                all_features.push(result.features);
                                all_ref_keys.push(key);
                                all_valid_indices.push(idx as u32);
                                n_new_valid += 1;
                            }
                        }
                        all_valid_indices.sort();

                        println!(
                            "  Incremental: {} new valid pairs (total {})",
                            n_new_valid,
                            all_features.len()
                        );

                        let merged = DatasetWithFeatures {
                            name: ds.name,
                            human_scores: all_human_scores,
                            features: all_features,
                            ref_keys: all_ref_keys,
                        };

                        // Save updated cache (new timestamped file)
                        if let Err(e) = save_feature_cache(
                            &save_path,
                            &merged,
                            &all_valid_indices,
                            &cache_config,
                        ) {
                            eprintln!("Warning: failed to save updated cache: {}", e);
                        } else {
                            println!("Saved updated feature cache to {:?}", save_path);
                        }
                        merged
                    }
                } else {
                    let ds = build_dataset_from_cache(cached, &pairs);
                    println!(
                        "Loaded {} pairs ({} features) from cache {:?} ({:.1}s)",
                        ds.human_scores.len(),
                        if ds.features.is_empty() {
                            0
                        } else {
                            ds.features[0].len()
                        },
                        load_path.as_deref().unwrap_or(Path::new("?")),
                        cache_start.elapsed().as_secs_f64()
                    );
                    ds
                }
            }
            None => {
                let (ds, valid_indices) = load_and_compute(
                    &format!("{:?}", args.format),
                    args.format,
                    &args.dataset,
                    args.max_images,
                    compute_all,
                    blur_passes,
                    blur_radius,
                    masking_strength,
                    num_scales,
                    args.target_metric,
                    extended_features,
                    extended_masking_strength,
                    downscale_filter,
                );
                if let Err(e) = save_feature_cache(&save_path, &ds, &valid_indices, &cache_config) {
                    eprintln!("Warning: failed to save feature cache: {}", e);
                } else {
                    println!("Saved feature cache to {:?}", save_path);
                }
                ds
            }
        }
    } else {
        let (ds, valid_indices) = load_and_compute(
            &format!("{:?}", args.format),
            args.format,
            &args.dataset,
            args.max_images,
            compute_all,
            blur_passes,
            blur_radius,
            masking_strength,
            num_scales,
            args.target_metric,
            extended_features,
            extended_masking_strength,
            downscale_filter,
        );
        if compute_all {
            let save_path = args
                .feature_cache
                .clone()
                .unwrap_or_else(|| auto_cache_save_path(&args.dataset));
            if let Err(e) = save_feature_cache(&save_path, &ds, &valid_indices, &cache_config) {
                eprintln!("Warning: failed to save feature cache: {}", e);
            } else {
                println!("Saved feature cache to {:?}", save_path);
            }
        }
        ds
    };

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
                "konfig-iqa" | "konfig" => DatasetFormat::KonfigIqa,
                "synthetic" | "synth" => DatasetFormat::Synthetic,
                _ => {
                    eprintln!("Unknown format: {}", parts[0]);
                    continue;
                }
            };
            let also_path = Path::new(parts[1]);
            let also_load = find_cache_to_load(also_path);
            let also_save = auto_cache_save_path(also_path);
            let ds = if compute_all && !args.recompute {
                let t = std::time::Instant::now();
                let also_cached =
                    also_load
                        .as_ref()
                        .and_then(|p| match load_feature_cache(p, &cache_config) {
                            Ok(c) => c,
                            Err(e) => {
                                eprintln!("Warning: failed to read cache {:?}: {}", p, e);
                                None
                            }
                        });
                match also_cached {
                    Some(cached) => {
                        let pairs = load_pairs(fmt, also_path, 0, args.target_metric);
                        let ds = build_dataset_from_cache(cached, &pairs);
                        println!(
                            "Loaded {} pairs from cache {:?} ({:.1}s)",
                            ds.human_scores.len(),
                            also_load.as_deref().unwrap_or(Path::new("?")),
                            t.elapsed().as_secs_f64()
                        );
                        ds
                    }
                    None => {
                        let (ds, valid_indices) = load_and_compute(
                            parts[0],
                            fmt,
                            also_path,
                            0,
                            compute_all,
                            blur_passes,
                            blur_radius,
                            masking_strength,
                            num_scales,
                            args.target_metric,
                            extended_features,
                            extended_masking_strength,
                            downscale_filter,
                        );
                        if let Err(e) =
                            save_feature_cache(&also_save, &ds, &valid_indices, &cache_config)
                        {
                            eprintln!("Warning: failed to save feature cache: {}", e);
                        } else {
                            println!("Saved feature cache to {:?}", also_save);
                        }
                        ds
                    }
                }
            } else {
                let (ds, valid_indices) = load_and_compute(
                    parts[0],
                    fmt,
                    also_path,
                    0,
                    compute_all,
                    blur_passes,
                    blur_radius,
                    masking_strength,
                    num_scales,
                    args.target_metric,
                    extended_features,
                    extended_masking_strength,
                    downscale_filter,
                );
                if compute_all {
                    if let Err(e) =
                        save_feature_cache(&also_save, &ds, &valid_indices, &cache_config)
                    {
                        eprintln!("Warning: failed to save feature cache: {}", e);
                    } else {
                        println!("Saved feature cache to {:?}", also_save);
                    }
                }
                ds
            };
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
    let mut training_log: Vec<String> = Vec::new();
    report_embedded_correlations(ds, &mut training_log);

    // Evaluate custom weights if provided
    if let Some(ref weights_path) = args.weights_file {
        let custom_weights = load_weights_file(weights_path);
        if custom_weights.len() != n_features {
            eprintln!(
                "Weights file has {} values, expected {}",
                custom_weights.len(),
                n_features
            );
            std::process::exit(1);
        }
        let msg = format!("\n=== Custom weights from {:?} ===", weights_path);
        log_line(&msg, &mut training_log);
        for ds in &all_datasets {
            let feats: Vec<&[f64]> = ds.features.iter().map(|v| v.as_slice()).collect();
            let custom_scores: Vec<f64> = feats
                .iter()
                .map(|f| zensim::score_from_features(f, &custom_weights).0)
                .collect();
            let srocc = spearman_correlation(&ds.human_scores, &custom_scores);
            let plcc = pearson_correlation(&ds.human_scores, &custom_scores);
            let krocc = kendall_correlation(&ds.human_scores, &custom_scores);
            log_line(
                &format!(
                    "  {}: SROCC={:.4}  PLCC={:.4}  KROCC={:.4}",
                    ds.name, srocc, plcc, krocc
                ),
                &mut training_log,
            );
        }
    }

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

        let best_weights = if dataset_groups.len() == 1 {
            let feats: Vec<&[f64]> = dataset_groups[0].2.iter().map(|v| v.as_slice()).collect();
            log_line(
                &format!(
                    "Training weights on {} pairs with {} features...",
                    dataset_groups[0].1.len(),
                    n_features
                ),
                &mut training_log,
            );
            let best_weights = train_weights(
                &dataset_groups[0].1,
                &feats,
                n_features,
                &frozen,
                &mut training_log,
            );
            print_trained_results(
                &dataset_groups[0].1,
                &feats,
                &best_weights,
                &mut training_log,
            );
            best_weights
        } else {
            log_line(
                &format!(
                    "\nMulti-dataset training on {} datasets...",
                    dataset_groups.len()
                ),
                &mut training_log,
            );
            let best_weights =
                train_weights_multi(&dataset_groups, n_features, &frozen, &mut training_log);

            for (name, h, f) in &dataset_groups {
                let feats: Vec<&[f64]> = f.iter().map(|v| v.as_slice()).collect();
                let trained_scores: Vec<f64> = feats
                    .iter()
                    .map(|feat| zensim::score_from_features(feat, &best_weights).0)
                    .collect();
                let srocc = spearman_correlation(h, &trained_scores);
                log_line(
                    &format!("  {}: SROCC = {:.4}", name, srocc),
                    &mut training_log,
                );
            }
            print_weights(&best_weights, &mut training_log);
            save_weights_file(&best_weights, "/tmp/zensim_trained_weights.txt");
            best_weights
        };

        // Auto-save training log and weights
        let log_dir = args
            .log_dir
            .unwrap_or_else(|| args.dataset.parent().unwrap_or(Path::new(".")).join("runs"));
        if let Err(e) = std::fs::create_dir_all(&log_dir) {
            eprintln!("Warning: failed to create log dir {:?}: {}", log_dir, e);
        } else {
            let timestamp = utc_timestamp();
            let target_suffix = match args.target_metric {
                Some(TargetMetric::GpuSsim2) => "_gpu_ssim2",
                Some(TargetMetric::GpuButteraugli) => "_gpu_butteraugli",
                Some(TargetMetric::CpuSsim2) => "_cpu_ssim2",
                Some(TargetMetric::CpuButteraugli) => "_cpu_butteraugli",
                Some(TargetMetric::Dssim) => "_dssim",
                None => "",
            };

            // Write log file
            let log_path = log_dir.join(format!("train_{}{}.txt", timestamp, target_suffix));
            {
                use std::io::Write;
                if let Ok(mut f) = std::fs::File::create(&log_path) {
                    let _ = writeln!(f, "# Training run: {}", timestamp);
                    let _ = writeln!(f, "# Git: {}", git_describe());
                    let _ = writeln!(
                        f,
                        "# CLI: {}",
                        std::env::args().collect::<Vec<_>>().join(" ")
                    );
                    let _ = writeln!(
                        f,
                        "# Dataset: {} ({} pairs, {} features)",
                        ds.name,
                        ds.human_scores.len(),
                        n_features
                    );
                    let _ = writeln!(f);
                    for line in &training_log {
                        let _ = writeln!(f, "{}", line);
                    }
                    println!("Saved training log to {:?}", log_path);
                }
            }

            // Write weights file
            let weights_path = log_dir.join(format!("weights_{}{}.txt", timestamp, target_suffix));
            {
                use std::io::Write;
                if let Ok(mut f) = std::fs::File::create(&weights_path) {
                    for w in &best_weights {
                        let _ = writeln!(f, "{:.10}", w);
                    }
                    println!("Saved weights to {:?}", weights_path);
                }
            }
        }
    }
}

/// Load pairs from a dataset (for human_scores) without computing features.
/// Used when features are loaded from cache but human_scores need fresh loading.
fn load_pairs(
    format: DatasetFormat,
    path: &Path,
    max_images: usize,
    target_metric: Option<TargetMetric>,
) -> Vec<ImagePair> {
    let pairs = match format {
        DatasetFormat::Tid2013 => load_tid2013(path),
        DatasetFormat::Kadid10k => load_kadid10k(path),
        DatasetFormat::Csiq => load_csiq(path),
        DatasetFormat::Pipal => load_pipal(path),
        DatasetFormat::Cid22 => load_cid22(path),
        DatasetFormat::KonfigIqa => load_konfig_iqa(path),
        DatasetFormat::Synthetic => load_synthetic(path, target_metric),
    };
    if max_images > 0 && max_images < pairs.len() {
        pairs[..max_images].to_vec()
    } else {
        pairs
    }
}

/// Build a DatasetWithFeatures from cached features + freshly loaded pairs.
/// Uses valid_indices to look up human_scores from the original pairs list.
/// Filters out any cached entries whose indices exceed the pairs list
/// (can happen when the cache was built from a larger CSV than the current
/// target metric supports, e.g., DSSIM skips rows without dssim scores).
fn build_dataset_from_cache(cached: CachedFeatures, pairs: &[ImagePair]) -> DatasetWithFeatures {
    let n_pairs = pairs.len();
    let mut human_scores = Vec::with_capacity(cached.valid_indices.len());
    let mut features = Vec::with_capacity(cached.valid_indices.len());
    let mut ref_keys = Vec::with_capacity(cached.valid_indices.len());

    let mut nan_skipped = 0usize;
    for (i, &idx) in cached.valid_indices.iter().enumerate() {
        if (idx as usize) < n_pairs {
            let score = pairs[idx as usize].human_score;
            // Skip NaN placeholder pairs (rows without target metric, e.g. missing dssim)
            if score.is_nan() {
                nan_skipped += 1;
                continue;
            }
            human_scores.push(score);
            features.push(cached.features[i].clone());
            ref_keys.push(cached.ref_keys[i].clone());
        }
    }
    if nan_skipped > 0 {
        eprintln!(
            "  Skipped {nan_skipped} cached entries with NaN human_score (missing target metric)"
        );
    }

    DatasetWithFeatures {
        name: cached.name,
        human_scores,
        features,
        ref_keys,
    }
}

fn load_weights_file(path: &Path) -> Vec<f64> {
    let content = std::fs::read_to_string(path).expect("Failed to read weights file");
    content
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
                return None;
            }
            // Handle Rust array format: "0.123456, " or plain "0.123456"
            let cleaned = trimmed.trim_end_matches(',').trim();
            cleaned.parse::<f64>().ok()
        })
        .collect()
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
    target_metric: Option<TargetMetric>,
    extended_features: bool,
    extended_masking_strength: f32,
    downscale_filter: zensim::DownscaleFilter,
) -> (DatasetWithFeatures, Vec<u32>) {
    let pairs = match format {
        DatasetFormat::Tid2013 => load_tid2013(path),
        DatasetFormat::Kadid10k => load_kadid10k(path),
        DatasetFormat::Csiq => load_csiq(path),
        DatasetFormat::Pipal => load_pipal(path),
        DatasetFormat::Cid22 => load_cid22(path),
        DatasetFormat::KonfigIqa => load_konfig_iqa(path),
        DatasetFormat::Synthetic => load_synthetic(path, target_metric),
    };

    let pairs = if max_images > 0 && max_images < pairs.len() {
        pairs[..max_images].to_vec()
    } else {
        pairs
    };

    let total_pairs = pairs.len();
    println!("Loading {}: {} image pairs...", name, total_pairs);

    let pb = ProgressBar::new(total_pairs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({per_sec}) {msg}")
            .unwrap(),
    );

    // Group pairs by reference image path for precomputed-reference reuse
    let mut by_ref: std::collections::BTreeMap<&Path, Vec<(usize, &ImagePair)>> =
        std::collections::BTreeMap::new();
    for (idx, pair) in pairs.iter().enumerate() {
        by_ref
            .entry(pair.reference.as_path())
            .or_default()
            .push((idx, pair));
    }
    let n_refs = by_ref.len();
    println!(
        "  {} unique references, {:.1} distorted/ref avg",
        n_refs,
        pairs.len() as f64 / n_refs as f64
    );

    let ref_groups: Vec<(&Path, Vec<(usize, &ImagePair)>)> = by_ref.into_iter().collect();

    let nan_result = zensim::ZensimResult {
        score: f64::NAN,
        raw_distance: f64::NAN,
        features: vec![],
        profile: zensim::ZensimProfile::PreviewV0_1,
        mean_offset: [f64::NAN; 3],
    };

    let config = zensim::ZensimConfig {
        compute_all_features: compute_all,
        extended_features,
        extended_masking_strength,
        blur_passes,
        blur_radius,
        masking_strength,
        num_scales,
        downscale_filter,
        score_mapping_a: 18.0,
        score_mapping_b: 0.7,
        ..Default::default()
    };

    let fpc = if config.extended_features {
        zensim::FEATURES_PER_CHANNEL_EXTENDED
    } else {
        zensim::FEATURES_PER_CHANNEL_BASIC
    };
    let total_features = num_scales * 3 * fpc;
    eprintln!(
        "  Config: scales={}, blur_passes={}, blur_radius={}, extended={}, masking={:.1}, downscale={:?}",
        num_scales, blur_passes, blur_radius, extended_features, masking_strength, downscale_filter,
    );
    eprintln!(
        "  Features: {} per channel × 3 channels × {} scales = {} total",
        fpc, num_scales, total_features,
    );
    if config.extended_features {
        eprintln!(
            "  Extended: masked_strength={:.1}, path=streaming",
            extended_masking_strength
        );
    }

    // Process reference groups in parallel
    let progress_counter = std::sync::atomic::AtomicU64::new(0);
    let start_time = std::time::Instant::now();
    let log_interval = (total_pairs / 20).max(1000) as u64; // ~5% increments

    let group_results: Vec<Vec<(usize, String, f64, zensim::ZensimResult)>> = ref_groups
        .par_iter()
        .map(|(ref_path, group)| {
            let fail = |grp: &[(usize, &ImagePair)]| -> Vec<_> {
                let prev = progress_counter
                    .fetch_add(grp.len() as u64, std::sync::atomic::Ordering::Relaxed);
                pb.inc(grp.len() as u64);
                let new = prev + grp.len() as u64;
                if prev / log_interval != new / log_interval {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let rate = new as f64 / elapsed;
                    let eta = (total_pairs as f64 - new as f64) / rate;
                    eprintln!(
                        "  [{:.0}s] {}/{} pairs ({:.1}%), {:.0}/s, ETA {:.0}s",
                        elapsed,
                        new,
                        total_pairs,
                        new as f64 / total_pairs as f64 * 100.0,
                        rate,
                        eta,
                    );
                }
                grp.iter()
                    .map(|(idx, pair)| {
                        (
                            *idx,
                            reference_key(pair),
                            pair.human_score,
                            nan_result.clone(),
                        )
                    })
                    .collect()
            };

            // Load reference image once
            let src_img = match image::open(ref_path) {
                Ok(img) => img.to_rgb8(),
                Err(_) => return fail(group),
            };
            let (w, h) = src_img.dimensions();
            let src_pixels: Vec<[u8; 3]> =
                src_img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();

            // Masking requires the full-image path;
            // extended features are handled by the streaming path.
            let needs_full_path = config.masking_strength > 0.0;

            // Precompute reference XYB + downscale pyramid (only used on fast path)
            let precomputed = if !needs_full_path {
                match zensim::precompute_reference_with_scales(
                    &src_pixels,
                    w as usize,
                    h as usize,
                    num_scales,
                ) {
                    Ok(p) => Some(p),
                    Err(_) => return fail(group),
                }
            } else {
                None
            };

            // Compare each distorted image against the reference (parallel)
            group
                .par_iter()
                .map(|(idx, pair)| {
                    let key = reference_key(pair);
                    let result = match image::open(&pair.distorted) {
                        Ok(img) => {
                            let dst = img.to_rgb8();
                            let (dw, dh) = dst.dimensions();
                            if dw != w || dh != h {
                                nan_result.clone()
                            } else {
                                let dst_pixels: Vec<[u8; 3]> =
                                    dst.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
                                if let Some(ref pre) = precomputed {
                                    zensim::compute_zensim_with_ref_and_config(
                                        pre,
                                        &dst_pixels,
                                        w as usize,
                                        h as usize,
                                        config,
                                    )
                                    .unwrap_or_else(|_| nan_result.clone())
                                } else {
                                    zensim::compute_zensim_with_config(
                                        &src_pixels,
                                        &dst_pixels,
                                        w as usize,
                                        h as usize,
                                        config,
                                    )
                                    .unwrap_or_else(|_| nan_result.clone())
                                }
                            }
                        }
                        Err(_) => nan_result.clone(),
                    };
                    let prev = progress_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    pb.inc(1);
                    let new = prev + 1;
                    if prev / log_interval != new / log_interval {
                        let elapsed = start_time.elapsed().as_secs_f64();
                        let rate = new as f64 / elapsed;
                        let eta = (total_pairs as f64 - new as f64) / rate;
                        eprintln!(
                            "  [{:.0}s] {}/{} pairs ({:.1}%), {:.0}/s, ETA {:.0}s",
                            elapsed,
                            new,
                            total_pairs,
                            new as f64 / total_pairs as f64 * 100.0,
                            rate,
                            eta,
                        );
                    }
                    (*idx, key, pair.human_score, result)
                })
                .collect()
        })
        .collect();

    let total_elapsed = start_time.elapsed().as_secs_f64();
    pb.finish_with_message("done");
    eprintln!(
        "  Feature extraction: {} pairs in {:.1}s ({:.0}/s)",
        total_pairs,
        total_elapsed,
        total_pairs as f64 / total_elapsed,
    );

    // Flatten and sort back to original pair order
    let mut results: Vec<(usize, String, f64, zensim::ZensimResult)> =
        group_results.into_iter().flatten().collect();
    results.sort_by_key(|(idx, _, _, _)| *idx);

    let mut human_scores = Vec::new();
    let mut features = Vec::new();
    let mut ref_keys = Vec::new();
    let mut valid_indices = Vec::new();
    let mut n_valid = 0;

    for (idx, key, hs, result) in results {
        if result.score.is_finite() {
            human_scores.push(hs);
            features.push(result.features);
            ref_keys.push(key);
            valid_indices.push(idx as u32);
            n_valid += 1;
        }
    }

    println!("  {} valid pairs from {}", n_valid, name);

    (
        DatasetWithFeatures {
            name: name.to_string(),
            human_scores,
            features,
            ref_keys,
        },
        valid_indices,
    )
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
fn report_embedded_correlations(ds: &DatasetWithFeatures, log: &mut Vec<String>) {
    let ew = expand_embedded_weights(ds.features[0].len());
    let metric_scores: Vec<f64> = ds
        .features
        .iter()
        .map(|f| zensim::score_from_features(f, &ew).0)
        .collect();

    let srocc = spearman_correlation(&ds.human_scores, &metric_scores);
    let plcc = pearson_correlation(&ds.human_scores, &metric_scores);
    let krocc = kendall_correlation(&ds.human_scores, &metric_scores);

    log_line(
        &format!(
            "\n=== {} — Correlation with Human Ratings (embedded weights) ===",
            ds.name
        ),
        log,
    );
    log_line(&format!("SROCC (Spearman):  {:.4}", srocc), log);
    log_line(&format!("PLCC  (Pearson):   {:.4}", plcc), log);
    log_line(&format!("KROCC (Kendall):   {:.4}", krocc), log);

    let min_m = metric_scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_m = metric_scores
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let mean_m: f64 = metric_scores.iter().sum::<f64>() / metric_scores.len() as f64;
    log_line(
        &format!(
            "Metric score range: {:.2} to {:.2}, mean: {:.2}",
            min_m, max_m, mean_m
        ),
        log,
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
    log_line(
        &format!(
            "Raw distance: min={:.3}, p10={:.3}, p50={:.3}, p90={:.3}, max={:.3}, mean={:.3}\n",
            min_d, p10, p50, p90, max_d, mean_d
        ),
        log,
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
        let weights = train_weights(&train_h, &train_slices, n_features, frozen, &mut Vec::new());

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
            train_weights(
                &train_groups[0].1,
                &feats,
                n_features,
                frozen,
                &mut Vec::new(),
            )
        } else {
            train_weights_multi(&train_groups, n_features, frozen, &mut Vec::new())
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
    log: &mut Vec<String>,
) -> Vec<f64> {
    let n_train = human_scores.len();
    let n_scales = (n_features as f64 / zensim::FEATURES_PER_SCALE as f64).max(1.0);

    // --- Transpose features scaled by 1/n_scales for cache-friendly per-dim access ---
    let mut features_t = vec![vec![0.0f64; n_train]; n_features];
    for (pair_idx, feats) in features.iter().enumerate() {
        for (dim, &val) in feats.iter().enumerate() {
            features_t[dim][pair_idx] = val / n_scales;
        }
    }

    // Feature ranges for additive step sizing
    let mut feat_max = vec![0.0f64; n_features];
    for dim in 0..n_features {
        for &f in &features_t[dim] {
            feat_max[dim] = feat_max[dim].max(f.abs());
        }
    }

    // --- O(1) Pearson infrastructure for pre-filtering ---
    let mut sum_ft = vec![0.0f64; n_features];
    let mut sum_fth = vec![0.0f64; n_features];
    let mut sum_ft2 = vec![0.0f64; n_features];
    for dim in 0..n_features {
        for i in 0..n_train {
            let f = features_t[dim][i];
            let h = human_scores[i];
            sum_ft[dim] += f;
            sum_fth[dim] += f * h;
            sum_ft2[dim] += f * f;
        }
    }
    let n = n_train as f64;
    let sum_h: f64 = human_scores.iter().sum();
    let sum_h2: f64 = human_scores.iter().map(|h| h * h).sum();
    let denom_h = n * sum_h2 - sum_h * sum_h;

    let pearson_neg_dist = |s_d: f64, s_dh: f64, s_d2: f64| -> f64 {
        let denom_d = n * s_d2 - s_d * s_d;
        if denom_d <= 0.0 || denom_h <= 0.0 {
            return -1.0;
        }
        -(n * s_dh - s_d * sum_h) / (denom_d * denom_h).sqrt()
    };

    // --- Spearman infrastructure for confirmation ---
    let human_ranks = ranks(human_scores);
    let mean_rank = (n + 1.0) / 2.0;
    let var_hr: f64 = human_ranks.iter().map(|r| (r - mean_rank).powi(2)).sum();

    let mut indexed: Vec<(usize, f64)> = Vec::with_capacity(n_train);
    let mut dist_ranks = vec![0.0f64; n_train];
    let mut trial_dist = vec![0.0f64; n_train];

    let fast_srocc = |distances: &[f64],
                      indexed: &mut Vec<(usize, f64)>,
                      dist_ranks: &mut [f64]|
     -> f64 {
        indexed.clear();
        indexed.extend(distances.iter().copied().enumerate());
        indexed.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut i = 0;
        while i < n_train {
            let mut j = i + 1;
            while j < n_train && indexed[j].1 == indexed[i].1 {
                j += 1;
            }
            let avg_rank = (i + j) as f64 / 2.0 + 0.5;
            for k in i..j {
                dist_ranks[indexed[k].0] = avg_rank;
            }
            i = j;
        }
        let mut cov = 0.0f64;
        let mut var_dr = 0.0f64;
        for i in 0..n_train {
            let ddr = dist_ranks[i] - mean_rank;
            let dhr = human_ranks[i] - mean_rank;
            cov += ddr * dhr;
            var_dr += ddr * ddr;
        }
        if var_dr <= 0.0 || var_hr <= 0.0 {
            return -1.0;
        }
        -(cov / (var_dr * var_hr).sqrt())
    };

    let mut best_weights = vec![1.0; n_features];
    let mut best_srocc = -1.0f64;

    let n_restarts = 10;
    let mut rng_state = 42u64;
    let mut next_rand = || -> f64 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng_state >> 33) as f64 / (u32::MAX as f64)
    };

    let start_time = std::time::Instant::now();

    for restart in 0..n_restarts {
        // Diverse starting points: embedded, uniform, perturbations of both, sparse random
        let mut weights: Vec<f64> = match restart {
            0 => expand_embedded_weights(n_features),
            1 => (0..n_features)
                .map(|i| {
                    if frozen[i] {
                        0.0
                    } else {
                        1.0 / n_features as f64
                    }
                })
                .collect(),
            2..=4 => {
                // Perturbations of embedded weights
                let base = expand_embedded_weights(n_features);
                base.iter()
                    .enumerate()
                    .map(|(i, &w)| {
                        if frozen[i] {
                            0.0
                        } else {
                            (w * (1.0 + (next_rand() - 0.5) * 0.6)).max(0.0)
                        }
                    })
                    .collect()
            }
            5..=7 => {
                // Perturbations of uniform weights
                let base = 1.0 / n_features as f64;
                (0..n_features)
                    .map(|i| {
                        if frozen[i] {
                            0.0
                        } else {
                            (base * (1.0 + (next_rand() - 0.5) * 1.0)).max(0.0)
                        }
                    })
                    .collect()
            }
            _ => {
                // Sparse random (30% non-zero)
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
            }
        };

        // Compute initial distances = Σ_j weights[j] * features_t[j][i]
        let mut distances = vec![0.0f64; n_train];
        for dim in 0..n_features {
            let w = weights[dim];
            if w != 0.0 {
                for i in 0..n_train {
                    distances[i] += w * features_t[dim][i];
                }
            }
        }

        // Pearson running sums
        let mut sum_d: f64 = distances.iter().sum();
        let mut sum_dh: f64 = distances
            .iter()
            .zip(human_scores.iter())
            .map(|(d, h)| d * h)
            .sum();
        let mut sum_d2: f64 = distances.iter().map(|d| d * d).sum();

        for iter in 0..50 {
            let mut improved = false;
            let step_scale = if iter < 20 { 1.0 } else { 0.5 };

            for dim in 0..n_features {
                if frozen[dim] {
                    weights[dim] = 0.0;
                    continue;
                }
                let old_w = weights[dim];

                // O(n) once per dim: sum of distance * feature for this dim
                let sum_dft: f64 = distances
                    .iter()
                    .zip(features_t[dim].iter())
                    .map(|(d, f)| d * f)
                    .sum();

                // Generate all trial weights
                let mut trials: Vec<f64> = Vec::with_capacity(20);
                for &mult in &[0.0, 0.5, 1.5, 2.0, 3.0, 0.1, 5.0, 10.0] {
                    trials.push((old_w * mult * step_scale + old_w * (1.0 - step_scale)).max(0.0));
                }
                if feat_max[dim] > 0.0 {
                    let base_step = 1.0 / feat_max[dim];
                    for &mult in &[0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0] {
                        trials.push(base_step * mult * step_scale);
                    }
                }

                // --- Phase 1: O(1) Pearson pre-filter to find top candidates ---
                let current_pearson = pearson_neg_dist(sum_d, sum_dh, sum_d2);
                let mut candidates: Vec<(f64, f64)> = Vec::with_capacity(4); // (pearson, weight)
                candidates.push((current_pearson, old_w));

                for &trial_w in &trials {
                    let delta = trial_w - old_w;
                    if delta == 0.0 {
                        continue;
                    }
                    let s_d = sum_d + delta * sum_ft[dim];
                    let s_dh = sum_dh + delta * sum_fth[dim];
                    let s_d2 = sum_d2 + 2.0 * delta * sum_dft + delta * delta * sum_ft2[dim];
                    let p = pearson_neg_dist(s_d, s_dh, s_d2);
                    candidates.push((p, trial_w));
                }

                // Sort by Pearson descending, take top 4 unique weights
                candidates
                    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                candidates.dedup_by(|a, b| (a.1 - b.1).abs() < 1e-12);
                candidates.truncate(4);

                // --- Phase 2: Spearman confirmation on top candidates ---
                let current_srocc = fast_srocc(&distances, &mut indexed, &mut dist_ranks);
                let mut best_local = current_srocc;
                let mut best_w = old_w;

                for &(_, trial_w) in &candidates {
                    let delta = trial_w - old_w;
                    if delta.abs() < 1e-15 {
                        continue;
                    }
                    for i in 0..n_train {
                        trial_dist[i] = distances[i] + delta * features_t[dim][i];
                    }
                    let s = fast_srocc(&trial_dist, &mut indexed, &mut dist_ranks);
                    if s > best_local {
                        best_local = s;
                        best_w = trial_w;
                    }
                }

                // Commit the best weight
                if best_w != old_w {
                    let delta = best_w - old_w;
                    for i in 0..n_train {
                        distances[i] += delta * features_t[dim][i];
                    }
                    // Update Pearson running sums
                    sum_d += delta * sum_ft[dim];
                    sum_dh += delta * sum_fth[dim];
                    sum_d2 += 2.0 * delta * sum_dft + delta * delta * sum_ft2[dim];
                    weights[dim] = best_w;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }

        for w in weights.iter_mut() {
            *w = w.max(0.0);
        }

        let srocc = fast_srocc(&distances, &mut indexed, &mut dist_ranks);
        if srocc > best_srocc {
            best_srocc = srocc;
            best_weights = weights;
        }
        log_line(
            &format!(
                "  Restart {}: SROCC = {:.4} [{:.1}s]",
                restart,
                srocc,
                start_time.elapsed().as_secs_f64()
            ),
            log,
        );
    }

    // Normalize weights so distances are in a useful range for distance_to_score().
    // Spearman is rank-invariant so uniform scaling preserves all correlation metrics.
    // Target: median distance ≈ 1.7 (matching embedded weights' p50).
    let mut dists: Vec<f64> = features
        .iter()
        .map(|f| {
            let mut d = 0.0f64;
            for (dim, &val) in f.iter().enumerate() {
                d += best_weights[dim] * val;
            }
            d / n_scales
        })
        .collect();
    dists.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50 = dists[dists.len() / 2];
    if p50 > 0.0 {
        let target_p50 = 1.7;
        let scale = target_p50 / p50;
        for w in &mut best_weights {
            *w *= scale;
        }
        log_line(
            &format!(
                "  Normalized weights: p50 distance {:.3} → {:.3} (scale={:.6})",
                p50, target_p50, scale
            ),
            log,
        );
    }

    log_line(
        &format!(
            "Best training SROCC: {:.4} ({:.1}s total)",
            best_srocc,
            start_time.elapsed().as_secs_f64()
        ),
        log,
    );
    best_weights
}

fn print_trained_results(
    human_scores: &[f64],
    feats: &[&[f64]],
    weights: &[f64],
    log: &mut Vec<String>,
) {
    let trained_scores: Vec<f64> = feats
        .iter()
        .map(|f| zensim::score_from_features(f, weights).0)
        .collect();

    let srocc = spearman_correlation(human_scores, &trained_scores);
    let plcc = pearson_correlation(human_scores, &trained_scores);
    let krocc = kendall_correlation(human_scores, &trained_scores);

    log_line("\n=== Correlation with Trained Weights ===", log);
    log_line(&format!("SROCC (Spearman):  {:.4}", srocc), log);
    log_line(&format!("PLCC  (Pearson):   {:.4}", plcc), log);
    log_line(&format!("KROCC (Kendall):   {:.4}", krocc), log);
    print_weights(weights, log);
    save_weights_file(weights, "/tmp/zensim_trained_weights.txt");
}

fn save_weights_file(weights: &[f64], path: &str) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("Failed to create weights file");
    for w in weights {
        writeln!(f, "{:.10}", w).unwrap();
    }
    println!("Saved weights to {}", path);
}

fn print_weights(weights: &[f64], log: &mut Vec<String>) {
    let features_per_ch = zensim::FEATURES_PER_SCALE / 3;
    log_line(
        &format!("\n// Trained weights ({} values):", weights.len()),
        log,
    );
    log_line(
        &format!("const TRAINED_WEIGHTS: [f64; {}] = [", weights.len()),
        log,
    );
    for (i, w) in weights.iter().enumerate() {
        if i % features_per_ch == 0 {
            let scale = i / zensim::FEATURES_PER_SCALE;
            let ch = (i % zensim::FEATURES_PER_SCALE) / features_per_ch;
            let ch_name = ["X", "Y", "B"][ch];
            // For weights, use print! for inline formatting but also capture to log
            print!("    // Scale {} Channel {}\n    ", scale, ch_name);
        }
        print!("{:.6}, ", w);
        if i % features_per_ch == features_per_ch - 1 {
            println!();
        }
    }
    println!("];");

    // Also capture the full weight array as a single log block
    let mut block = String::new();
    for (i, w) in weights.iter().enumerate() {
        if i % features_per_ch == 0 {
            let scale = i / zensim::FEATURES_PER_SCALE;
            let ch = (i % zensim::FEATURES_PER_SCALE) / features_per_ch;
            let ch_name = ["X", "Y", "B"][ch];
            block.push_str(&format!("    // Scale {} Channel {}\n    ", scale, ch_name));
        }
        block.push_str(&format!("{:.6}, ", w));
        if i % features_per_ch == features_per_ch - 1 {
            block.push('\n');
        }
    }
    block.push_str("];");
    log.push(block);
}

/// Train weights on multiple datasets, maximizing blended Spearman.
/// Uses Spearman on subsampled datasets for training, validates on full data per restart.
#[allow(clippy::type_complexity)]
fn train_weights_multi(
    datasets: &[(String, Vec<f64>, Vec<Vec<f64>>)],
    n_features: usize,
    frozen: &[bool],
    log: &mut Vec<String>,
) -> Vec<f64> {
    let start_time = std::time::Instant::now();
    let n_datasets = datasets.len();
    let n_scales = (n_features as f64 / zensim::FEATURES_PER_SCALE as f64).max(1.0);

    // Per-dataset training state
    struct DatasetState {
        n_train: usize,
        features_t: Vec<Vec<f64>>, // [dim][pair], scaled by 1/n_scales
        human_ranks: Vec<f64>,
        mean_rank: f64,
        var_hr: f64,
    }

    let max_train = 15_000usize;
    let mut ds_states: Vec<DatasetState> = Vec::with_capacity(n_datasets);
    let mut feat_max = vec![0.0f64; n_features];

    for (name, human_scores, feats) in datasets {
        let n_full = human_scores.len();
        let (train_human, train_features): (Vec<f64>, Vec<&[f64]>) = if n_full > max_train {
            let mut rng = 12345u64;
            let mut indices: Vec<usize> = (0..n_full).collect();
            for i in 0..max_train {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let j = i + (rng >> 33) as usize % (n_full - i);
                indices.swap(i, j);
            }
            indices.truncate(max_train);
            indices.sort_unstable();
            let h: Vec<f64> = indices.iter().map(|&i| human_scores[i]).collect();
            let f: Vec<&[f64]> = indices.iter().map(|&i| feats[i].as_slice()).collect();
            log_line(
                &format!("  {}: subsampled {}/{} pairs", name, max_train, n_full),
                log,
            );
            (h, f)
        } else {
            (
                human_scores.clone(),
                feats.iter().map(|v| v.as_slice()).collect(),
            )
        };
        let n_train = train_human.len();

        let mut features_t = vec![vec![0.0f64; n_train]; n_features];
        for (pair_idx, f) in train_features.iter().enumerate() {
            for (dim, &val) in f.iter().enumerate() {
                let scaled = val / n_scales;
                features_t[dim][pair_idx] = scaled;
                feat_max[dim] = feat_max[dim].max(scaled.abs());
            }
        }

        let human_ranks = ranks(&train_human);
        let nf = n_train as f64;
        let mean_rank = (nf + 1.0) / 2.0;
        let var_hr: f64 = human_ranks.iter().map(|r| (r - mean_rank).powi(2)).sum();

        ds_states.push(DatasetState {
            n_train,
            features_t,
            human_ranks,
            mean_rank,
            var_hr,
        });
    }

    // Reusable buffers (sized to max dataset)
    let max_n = ds_states.iter().map(|ds| ds.n_train).max().unwrap_or(0);
    let mut indexed: Vec<(usize, f64)> = Vec::with_capacity(max_n);
    let mut dist_ranks = vec![0.0f64; max_n];
    let mut trial_dist = vec![0.0f64; max_n];

    // Spearman(scores, human) = -Spearman(distances, human) for a single dataset
    let fast_srocc = |distances: &[f64],
                      ds: &DatasetState,
                      indexed: &mut Vec<(usize, f64)>,
                      dist_ranks: &mut [f64]|
     -> f64 {
        let n = ds.n_train;
        indexed.clear();
        indexed.extend(distances[..n].iter().copied().enumerate());
        indexed.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            while j < n && indexed[j].1 == indexed[i].1 {
                j += 1;
            }
            let avg_rank = (i + j) as f64 / 2.0 + 0.5;
            for k in i..j {
                dist_ranks[indexed[k].0] = avg_rank;
            }
            i = j;
        }
        let mut cov = 0.0f64;
        let mut var_dr = 0.0f64;
        for i in 0..n {
            let ddr = dist_ranks[i] - ds.mean_rank;
            let dhr = ds.human_ranks[i] - ds.mean_rank;
            cov += ddr * dhr;
            var_dr += ddr * ddr;
        }
        if var_dr <= 0.0 || ds.var_hr <= 0.0 {
            return -1.0;
        }
        -(cov / (var_dr * ds.var_hr).sqrt())
    };

    // Blended objective across datasets
    let eval_multi = |all_distances: &[Vec<f64>],
                      indexed: &mut Vec<(usize, f64)>,
                      dist_ranks: &mut [f64]|
     -> f64 {
        let mut min_s = f64::INFINITY;
        let mut sum_s = 0.0;
        for (k, ds) in ds_states.iter().enumerate() {
            let s = fast_srocc(&all_distances[k], ds, indexed, dist_ranks);
            sum_s += s;
            min_s = min_s.min(s);
        }
        0.5 * sum_s / n_datasets as f64 + 0.5 * min_s
    };

    let mut best_weights = vec![1.0; n_features];
    let mut best_avg_srocc = -1.0f64;

    let n_restarts = 20;
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
        } else if restart <= 5 {
            let base = expand_embedded_weights(n_features);
            base.iter()
                .enumerate()
                .map(|(i, &w)| {
                    if frozen[i] {
                        0.0
                    } else {
                        let pert = 1.0 + (next_rand() - 0.5) * 0.4;
                        (w * pert).max(0.0)
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

        // Per-dataset distances
        let mut all_distances: Vec<Vec<f64>> = ds_states
            .iter()
            .map(|ds| {
                let mut d = vec![0.0f64; ds.n_train];
                for dim in 0..n_features {
                    let w = weights[dim];
                    if w != 0.0 {
                        for i in 0..ds.n_train {
                            d[i] += w * ds.features_t[dim][i];
                        }
                    }
                }
                d
            })
            .collect();

        for iter in 0..80 {
            let mut improved = false;
            let step_scale = if iter < 30 {
                1.0
            } else if iter < 60 {
                0.5
            } else {
                0.25
            };

            for dim in 0..n_features {
                if frozen[dim] {
                    weights[dim] = 0.0;
                    continue;
                }
                let old_w = weights[dim];
                let current_obj = eval_multi(&all_distances, &mut indexed, &mut dist_ranks);
                let mut best_obj = current_obj;
                let mut best_w = old_w;

                // Generate trials
                let mut trials: Vec<f64> = Vec::with_capacity(22);
                for &mult in &[0.0, 0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0, 0.1, 5.0, 10.0] {
                    trials.push((old_w * mult * step_scale + old_w * (1.0 - step_scale)).max(0.0));
                }
                if feat_max[dim] > 0.0 {
                    let base_step = 1.0 / feat_max[dim];
                    for &mult in &[0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0] {
                        trials.push(base_step * mult * step_scale);
                    }
                }

                for &trial_w in &trials {
                    let delta = trial_w - old_w;
                    if delta == 0.0 {
                        continue;
                    }

                    // Evaluate blended Spearman with trial weight
                    let mut min_s = f64::INFINITY;
                    let mut sum_s = 0.0;
                    for (k, ds) in ds_states.iter().enumerate() {
                        // Compute trial distances for this dataset
                        for i in 0..ds.n_train {
                            trial_dist[i] = all_distances[k][i] + delta * ds.features_t[dim][i];
                        }
                        let s = fast_srocc(&trial_dist, ds, &mut indexed, &mut dist_ranks);
                        sum_s += s;
                        min_s = min_s.min(s);
                    }
                    let obj = 0.5 * sum_s / n_datasets as f64 + 0.5 * min_s;
                    if obj > best_obj {
                        best_obj = obj;
                        best_w = trial_w;
                    }
                }

                if best_w != old_w {
                    let delta = best_w - old_w;
                    for (k, ds) in ds_states.iter().enumerate() {
                        for i in 0..ds.n_train {
                            all_distances[k][i] += delta * ds.features_t[dim][i];
                        }
                    }
                    weights[dim] = best_w;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }

        for w in weights.iter_mut() {
            *w = w.max(0.0);
        }

        // Validate with Spearman on full datasets
        let mut min_srocc = f64::INFINITY;
        let mut sum_srocc = 0.0;
        for (_, human, feats) in datasets {
            let feat_slices: Vec<&[f64]> = feats.iter().map(|v| v.as_slice()).collect();
            let s = eval_srocc(human, &feat_slices, &weights);
            sum_srocc += s;
            min_srocc = min_srocc.min(s);
        }
        let obj = 0.5 * sum_srocc / n_datasets as f64 + 0.5 * min_srocc;
        let avg = sum_srocc / n_datasets as f64;

        if obj > best_avg_srocc {
            best_avg_srocc = obj;
            best_weights = weights;
        }
        log_line(
            &format!(
                "  Restart {}: obj={:.4}  avg SROCC={:.4} [{:.1}s]",
                restart,
                obj,
                avg,
                start_time.elapsed().as_secs_f64()
            ),
            log,
        );
    }

    log_line(
        &format!(
            "Best multi-dataset objective: {:.4} ({:.1}s total)",
            best_avg_srocc,
            start_time.elapsed().as_secs_f64()
        ),
        log,
    );
    for (name, human, feats) in datasets {
        let feat_slices: Vec<&[f64]> = feats.iter().map(|v| v.as_slice()).collect();
        let s = eval_srocc(human, &feat_slices, &best_weights);
        log_line(&format!("  {}: SROCC = {:.4}", name, s), log);
    }
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

fn load_konfig_iqa(base: &Path) -> Vec<ImagePair> {
    // KonFiG-IQA uses DCR (Degradation Category Rating) from EXP_III.
    // Raw data: individual worker ratings per (source, distortion, level).
    // We aggregate to mean DCR, then invert to quality (higher = better).
    let csv_path = base.join("DATA/EXP_III/data3.csv");
    if !csv_path.exists() {
        eprintln!("Cannot find DATA/EXP_III/data3.csv in {:?}", base);
        return vec![];
    }

    let mut rdr = csv::Reader::from_path(&csv_path).expect("Failed to open data3.csv");

    // Aggregate raw ratings: mean DCR per (source, distortion_type, level)
    let mut ratings: HashMap<(String, String, u32), Vec<u32>> = HashMap::new();
    for result in rdr.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };
        // Columns: Source, Distortion Type, BoostType, Distortion Level, HIT id,
        //          Assignment id, Worker id, Answer, Time
        if record.len() < 8 {
            continue;
        }
        let source = record[0].to_string();
        let dist_type = record[1].to_string();
        let level: u32 = match record[3].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let answer: u32 = match record[7].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        ratings
            .entry((source, dist_type, level))
            .or_default()
            .push(answer);
    }

    let mut pairs = Vec::new();
    for ((source, dist_type, level), vals) in &ratings {
        let mean_dcr: f64 = vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64;
        // DCR: 0=imperceptible, 4=very annoying. Invert to quality 0-1.
        let quality = (4.0 - mean_dcr) / 4.0;

        // Image path: IMAGES/PartA/{source}/{distortion}/{source}_{distortion}_{level}.png
        let dist_path = base
            .join("IMAGES/PartA")
            .join(source)
            .join(dist_type)
            .join(format!("{source}_{dist_type}_{level}.png"));
        // Reference: IMAGES/reference_images/{source}_0.png
        let ref_path = base
            .join("IMAGES/reference_images")
            .join(format!("{source}_0.png"));

        if !dist_path.exists() {
            continue;
        }

        pairs.push(ImagePair {
            reference: ref_path,
            distorted: dist_path,
            human_score: quality,
        });
    }

    pairs.sort_by(|a, b| a.distorted.cmp(&b.distorted));
    println!(
        "  KonFiG-IQA: {} pairs from {} ratings",
        pairs.len(),
        ratings.len()
    );
    pairs
}

/// Load synthetic training CSV produced by generate_zensim_training.
///
/// Load synthetic training CSV with header-based column lookup.
///
/// Supports both old format (ssimulacra2, butteraugli columns) and new format
/// (gpu_ssimulacra2, gpu_butteraugli, cpu_ssimulacra2, cpu_butteraugli columns).
///
/// Scores are normalized to 0-1 (higher = better quality):
/// - SSIM2: `((ssim2 + 50) / 150).clamp(0, 1)` — monotonic, preserves SROCC
/// - Butteraugli: `1 / (1 + ba)` — monotonically decreasing, maps (0,inf) → (0,1]
fn load_synthetic(csv_path: &Path, target_metric: Option<TargetMetric>) -> Vec<ImagePair> {
    let metric = target_metric.unwrap_or_else(|| {
        eprintln!(
            "Warning: --target-metric not specified for synthetic dataset, defaulting to gpu-ssim2"
        );
        TargetMetric::GpuSsim2
    });

    let mut rdr = csv::Reader::from_path(csv_path).unwrap_or_else(|e| {
        eprintln!("Failed to open CSV {}: {}", csv_path.display(), e);
        std::process::exit(1);
    });

    // Header-based column lookup
    let headers = rdr.headers().unwrap().clone();
    let col = |name: &str| -> Option<usize> { headers.iter().position(|h| h == name) };
    let source_col = col("source_path").expect("CSV missing source_path column");
    let decoded_col = col("decoded_path").expect("CSV missing decoded_path column");

    // Resolve the metric column name based on target + available headers
    let metric_col = match metric {
        TargetMetric::GpuSsim2 => col("gpu_ssimulacra2").or_else(|| col("ssimulacra2")),
        TargetMetric::GpuButteraugli => col("gpu_butteraugli").or_else(|| col("butteraugli")),
        TargetMetric::CpuSsim2 => col("cpu_ssimulacra2"),
        TargetMetric::CpuButteraugli => col("cpu_butteraugli"),
        TargetMetric::Dssim => col("dssim"),
    }
    .unwrap_or_else(|| {
        eprintln!("CSV missing column for {:?}", metric);
        eprintln!(
            "Available columns: {:?}",
            headers.iter().collect::<Vec<_>>()
        );
        std::process::exit(1);
    });

    let is_ssim2 = matches!(metric, TargetMetric::GpuSsim2 | TargetMetric::CpuSsim2);
    let is_dssim = matches!(metric, TargetMetric::Dssim);

    let mut pairs = Vec::new();
    let mut skipped = 0usize;

    for result in rdr.records() {
        let record = match result {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  CSV parse error: {}", e);
                skipped += 1;
                continue;
            }
        };

        let source_path = PathBuf::from(&record[source_col]);
        let decoded_path = PathBuf::from(&record[decoded_col]);

        let raw_score: f64 = match record[metric_col].parse() {
            Ok(v) => v,
            Err(_) => {
                // Insert NaN placeholder to preserve row indices for cache alignment
                pairs.push(ImagePair {
                    reference: source_path,
                    distorted: decoded_path,
                    human_score: f64::NAN,
                });
                skipped += 1;
                continue;
            }
        };

        // Normalize to 0-1, higher = better
        let score = if is_ssim2 {
            ((raw_score + 50.0) / 150.0).clamp(0.0, 1.0)
        } else if is_dssim {
            // DSSIM: 0 = identical, ~0.1 = poor. Scale by 100 for better spread.
            1.0 / (1.0 + 100.0 * raw_score)
        } else {
            1.0 / (1.0 + raw_score)
        };

        if !score.is_finite() {
            // Insert NaN placeholder to preserve row indices for cache alignment
            pairs.push(ImagePair {
                reference: source_path,
                distorted: decoded_path,
                human_score: f64::NAN,
            });
            skipped += 1;
            continue;
        }

        pairs.push(ImagePair {
            reference: source_path,
            distorted: decoded_path,
            human_score: score,
        });
    }

    if skipped > 0 {
        eprintln!("  Synthetic: skipped {} invalid rows", skipped);
    }
    println!("  Synthetic: {} pairs, target={:?}", pairs.len(), metric,);
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
