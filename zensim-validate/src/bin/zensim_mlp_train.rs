//! Standalone binary that wraps `zensim-validate::mlp_train::train_mlp`.
//!
//! Reads one or more `--group NAME:CSV:TRAIN_W:VAL_W` CSVs (each in
//! the trainer-compatible shape `ref_basename, human_score, f0..f227`)
//! and trains the multi-group RankNet MLP. Writes ZNPR v2 bytes to
//! `--out PATH`.
//!
//! ## V0_16 SHIP recipe (current default)
//!
//! Defaults are tuned to the V0_16 ship recipe (CID22 SROCC 0.8919 on
//! 4292 pairs, +0.0024 over fast-ssim2). To reproduce V0_16:
//!
//!     zensim_mlp_train \
//!       --group safesyn:/tmp/safe_synth_clean_features.csv:1.0:0.0 \
//!       --group kadid:/path/to/kadid_features.csv:0.3:1.0 \
//!       --group tid:/path/to/tid_features.csv:0.3:1.0 \
//!       --tv-pairs-file /tmp/combined_purged_tv_pairs_bands.tsv \
//!       --out benchmarks/rust_v0_X_$(date -u +%Y-%m-%d).bin
//!
//! All other flags (--hidden 128, --tv-weight 20, --epochs 300,
//! --val-policy min, --lr 1e-3, --max-features 228, --seed 1) default
//! to the V0_16 ship values; override only when running cycle
//! experiments. See zensim/CONTEXT-HANDOFF.md for the full recipe
//! provenance (raw bake md5 b3f5fc59, calibrated baf3fdcb).

use clap::Parser;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

#[path = "../mlp_train.rs"]
#[allow(dead_code)] // some helpers are unused in this binary
mod mlp_train;

use mlp_train::{
    MlpHyperparams, TrainingGroup, TvRegularizer, ValidationPolicy, train_mlp_with_tv,
};

#[derive(Parser)]
#[command(
    name = "zensim_mlp_train",
    about = "Rust RankNet MLP trainer (defaults match V0_16 ship recipe — see benchmarks/recipe_v0_16.sh for the full invocation)"
)]
struct Args {
    /// Group spec: NAME:CSV_PATH:TRAIN_WEIGHT:VAL_WEIGHT. Repeat for
    /// each dataset. CSV header must include `ref_basename`,
    /// `human_score`, and `f0..f<N-1>`. `human_score` is in [0, 1] and
    /// is multiplied by 100 internally to match `score_zensim` scale.
    #[arg(long, required = true)]
    group: Vec<String>,

    /// Number of hidden units in the single hidden layer. Default 128
    /// matches the V0_16 ship recipe. Other tested architectures:
    /// h=32 (V0_4 placeholder, too small), h=64 (V0_5, AIC-4-friendly
    /// but -0.01 CID22), h=192/256 (overfit, worse on both axes).
    #[arg(long, default_value_t = 128)]
    hidden: usize,

    /// Training epochs.
    #[arg(long, default_value_t = 300)]
    epochs: usize,

    /// Pair updates per epoch (Rust trainer's defining parameter — V0_5 used 50,000).
    #[arg(long, default_value_t = 50_000)]
    pairs_per_epoch: usize,

    /// Initial learning rate. Cosine annealing with 50-epoch period
    /// is built into mlp_train::train_mlp.
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,

    /// L2 regularization on layer weights (not biases).
    #[arg(long, default_value_t = 1e-5)]
    l2: f64,

    /// LeakyReLU negative slope.
    #[arg(long, default_value_t = 0.01)]
    leaky_alpha: f64,

    /// Validation policy: "min" (worst per-group SROCC, V0_5 default)
    /// or "mean".
    #[arg(long, default_value = "min")]
    val_policy: String,

    /// Random seed. Default 1 matches V0_16 ship.
    #[arg(long, default_value_t = 1)]
    seed: u64,

    /// Log every N epochs.
    #[arg(long, default_value_t = 10)]
    log_every: usize,

    /// Early-stop patience (epochs of no validation improvement). 0 disables.
    #[arg(long, default_value_t = 50)]
    early_stop_patience: usize,

    /// Output path for the trained ZNPR v2 bake.
    #[arg(long)]
    out: PathBuf,

    /// Cap features at the first N columns. Default 228 matches the
    /// V0_4 zensim runtime input width. CSVs may include extended
    /// features at f228..f299 (per-pair training-side features) that
    /// the runtime can't supply; capping at 228 keeps train/inference
    /// feature spaces aligned. Pass `--max-features 300` only for
    /// content-classifier or research bakes that don't ship as runtime
    /// weights.
    #[arg(long, default_value_t = 228)]
    max_features: usize,

    /// TV-regularizer pair indices TSV. Two columns: lo_trainer_idx,
    /// hi_trainer_idx. Indices reference rows in the concatenated
    /// trainer-feature space (group 0 first, then group 1, etc.).
    /// Penalty per pair: max(0, pred[hi] - pred[lo]).
    #[arg(long)]
    tv_pairs_file: Option<PathBuf>,

    /// TV regularizer weight (loss multiplier). 5-30 typical range.
    /// 0 disables TV even if --tv-pairs-file is given.
    #[arg(long, default_value_t = 0.0)]
    tv_weight: f64,

    /// Apply TV update every N RankNet pair updates. 50 default
    /// (gives ~1000 TV steps/epoch at pairs_per_epoch=50000).
    #[arg(long, default_value_t = 50)]
    tv_apply_every: usize,

    /// TV mini-batch size per update.
    #[arg(long, default_value_t = 32)]
    tv_batch: usize,

    /// Per-band TV weights `[B0, B1, B2, B3]`. When set, the TV
    /// pairs file MUST include a `band_id` column (use
    /// `regen_tv_pairs.py --emit-bands`). Pair-specific weight
    /// replaces the flat `--tv-weight`. Example:
    /// `--tv-band-weights 10,20,10,10` pushes B1 (medium-quality
    /// band) harder than other bands.
    #[arg(long, value_parser = parse_band_weights)]
    tv_band_weights: Option<[f64; 4]>,

    /// Cycle-9 row-weight boost for B0 + B1 (low-quality) rows
    /// during per-step RankNet pair sampling. Multiplies the per-row
    /// sampling weight: full boost for human_score<50, sqrt(boost)
    /// for 50..65, 1.0 elsewhere. Default 1.0 = no-op. Composes
    /// multiplicatively with --mid-q-boost. Python-side cycle-9
    /// findings: boost=1.5 closed B0 SROCC by +0.011 with mild B2
    /// regression at single seed; multi-seed verification showed
    /// 5-seed mean was within noise so it didn't ship as default —
    /// useful for B0-targeted ablations.
    #[arg(long, default_value_t = 1.0)]
    low_q_boost: f64,

    /// Cycle-12 row-weight boost for B1 + B2 (medium-quality) rows
    /// during per-step RankNet pair sampling. Multiplies per-row
    /// sampling weight for human_score in [50, 90). Default 1.0 =
    /// no-op. Python-side cycle-12 finding: boost=1.5 was a
    /// σ-tightener (4x tighter seed-to-seed CID22 variance vs
    /// baseline 0.0068 → 0.0016) with small +0.003 mean lift; useful
    /// for downstream codec orchestrators needing stable per-image
    /// ranking. Boost=2.0 plateaus (no additional mean lift, σ
    /// widens again).
    #[arg(long, default_value_t = 1.0)]
    mid_q_boost: f64,

    /// Optional path to dump the trainer log for the run.
    #[arg(long)]
    log_path: Option<PathBuf>,
}

struct LoadedGroup {
    name: String,
    train_w: f64,
    val_w: f64,
    human_scores: Vec<f64>,
    feature_rows: Vec<Vec<f64>>,
    n_features: usize,
}

fn parse_group_spec(spec: &str) -> Result<(String, PathBuf, f64, f64), String> {
    let parts: Vec<&str> = spec.splitn(4, ':').collect();
    if parts.len() != 4 {
        return Err(format!("expected NAME:PATH:TRAIN_W:VAL_W, got {spec:?}"));
    }
    let train_w: f64 = parts[2].parse().map_err(|e| format!("bad train_w: {e}"))?;
    let val_w: f64 = parts[3].parse().map_err(|e| format!("bad val_w: {e}"))?;
    Ok((
        parts[0].to_string(),
        PathBuf::from(parts[1]),
        train_w,
        val_w,
    ))
}

fn load_csv(path: &PathBuf, name: &str) -> Result<LoadedGroup, String> {
    let file = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let mut rdr = BufReader::new(file);
    let mut header = String::new();
    rdr.read_line(&mut header)
        .map_err(|e| format!("header read {path:?}: {e}"))?;
    let cols: Vec<&str> = header.trim_end().split(',').collect();
    let score_idx = cols
        .iter()
        .position(|&c| c == "human_score")
        .ok_or_else(|| format!("{path:?}: missing human_score column"))?;
    let f0 = cols
        .iter()
        .position(|&c| c == "f0")
        .ok_or_else(|| format!("{path:?}: missing f0 column"))?;
    // Find consecutive f0..f<N-1> columns. Stop at the first non-f<idx> column or end of header.
    let mut n_features = 0usize;
    while f0 + n_features < cols.len() {
        let expected = format!("f{}", n_features);
        if cols[f0 + n_features] != expected {
            break;
        }
        n_features += 1;
    }
    if n_features == 0 {
        return Err(format!("{path:?}: no fN columns found"));
    }
    let mut human_scores = Vec::new();
    let mut feature_rows = Vec::new();
    for (lineno, line) in rdr.lines().enumerate() {
        let line = line.map_err(|e| format!("read line {}: {e}", lineno + 2))?;
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < f0 + n_features {
            return Err(format!(
                "{path:?} line {}: expected ≥{} fields, got {}",
                lineno + 2,
                f0 + n_features,
                fields.len()
            ));
        }
        // Multiply by 100 to match `score_zensim` scale (consistent with Python's load_human_csv).
        let score: f64 = fields[score_idx]
            .parse::<f64>()
            .map_err(|e| format!("{path:?} line {}: bad human_score: {e}", lineno + 2))?
            * 100.0;
        let mut row = Vec::with_capacity(n_features);
        for i in 0..n_features {
            row.push(
                fields[f0 + i]
                    .parse::<f64>()
                    .map_err(|e| format!("{path:?} line {}: bad f{i}: {e}", lineno + 2))?,
            );
        }
        human_scores.push(score);
        feature_rows.push(row);
    }
    println!(
        "  {name}: loaded {} pairs × {n_features} features from {path:?}",
        human_scores.len()
    );
    Ok(LoadedGroup {
        name: name.to_string(),
        train_w: 0.0,
        val_w: 0.0,
        human_scores,
        feature_rows,
        n_features,
    })
}

fn parse_band_weights(s: &str) -> Result<[f64; 4], String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 4 {
        return Err(format!(
            "expected 4 comma-separated floats for B0,B1,B2,B3; got {}",
            parts.len()
        ));
    }
    let mut out = [0.0; 4];
    for (i, p) in parts.iter().enumerate() {
        out[i] = p
            .trim()
            .parse()
            .map_err(|e| format!("bad weight at index {i}: {e}"))?;
    }
    Ok(out)
}

fn main() {
    let args = Args::parse();

    let val_policy = match args.val_policy.to_lowercase().as_str() {
        "min" => ValidationPolicy::Min,
        "mean" => ValidationPolicy::Mean,
        other => {
            eprintln!("--val-policy must be 'min' or 'mean', got {other:?}");
            std::process::exit(2);
        }
    };

    // Load all groups, infer n_features from the first.
    let mut loaded: Vec<LoadedGroup> = Vec::new();
    let mut n_features = 0usize;
    for spec in &args.group {
        let (name, path, train_w, val_w) = parse_group_spec(spec).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2);
        });
        let mut g = load_csv(&path, &name).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
        g.train_w = train_w;
        g.val_w = val_w;
        let cap = args.max_features;
        if g.n_features > cap {
            for row in &mut g.feature_rows {
                row.truncate(cap);
            }
            g.n_features = cap;
        }
        if n_features == 0 {
            n_features = g.n_features;
        } else if n_features != g.n_features {
            eprintln!(
                "{name}: n_features={} disagrees with first group's {n_features}",
                g.n_features
            );
            std::process::exit(1);
        }
        loaded.push(g);
    }
    if loaded.is_empty() {
        eprintln!("no groups loaded");
        std::process::exit(2);
    }

    // Build TrainingGroups (borrows feature rows from `loaded`).
    let feat_refs: Vec<Vec<&[f64]>> = loaded
        .iter()
        .map(|g| g.feature_rows.iter().map(|r| r.as_slice()).collect())
        .collect();
    let groups: Vec<TrainingGroup> = loaded
        .iter()
        .zip(feat_refs.iter())
        .map(|(g, fr)| TrainingGroup {
            name: g.name.clone(),
            human_scores: &g.human_scores,
            features: fr.as_slice(),
            train_weight: g.train_w,
            validation_weight: g.val_w,
        })
        .collect();

    let hyperparams = MlpHyperparams {
        n_hidden: args.hidden,
        n_epochs: args.epochs,
        pairs_per_epoch: args.pairs_per_epoch,
        initial_lr: args.lr,
        leaky_alpha: args.leaky_alpha,
        seed: args.seed,
        log_every: args.log_every,
        l2_lambda: args.l2,
        early_stop_patience: args.early_stop_patience,
        validation_policy: val_policy,
        low_q_boost: args.low_q_boost,
        mid_q_boost: args.mid_q_boost,
    };

    println!(
        "Training: {} groups, {n_features} features, {hyperparams:?}",
        groups.len()
    );

    // Build optional TV regularizer: load pairs file, concatenate
    // all-group features into a flat row index space (trainer row =
    // group 0 rows [0..n0], then group 1 [n0..n0+n1], etc.).
    let tv_regularizer: Option<TvRegularizer> = if let Some(tv_path) = &args.tv_pairs_file
        && args.tv_weight > 0.0
    {
        // Concatenate features in group order to build the flat feature index.
        let mut all_features: Vec<Vec<f64>> = Vec::new();
        for g in &loaded {
            for row in &g.feature_rows {
                all_features.push(row.clone());
            }
        }
        // Load pairs file.
        let f = File::open(tv_path).unwrap_or_else(|e| {
            eprintln!("open {tv_path:?}: {e}");
            std::process::exit(1);
        });
        let rdr = BufReader::new(f);
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        let mut bands: Vec<u8> = Vec::new();
        let mut tv_has_bands = false;
        for (i, line) in rdr.lines().enumerate() {
            let line = line.unwrap_or_else(|e| {
                eprintln!("read TV pair line {}: {e}", i + 1);
                std::process::exit(1);
            });
            if i == 0 && line.starts_with("lo_") {
                tv_has_bands = line.contains("band_id");
                continue; // header
            }
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() < 2 {
                continue;
            }
            let lo: usize = parts[0].parse().unwrap_or_else(|e| {
                eprintln!("bad lo idx line {}: {e}", i + 1);
                std::process::exit(1);
            });
            let hi: usize = parts[1].parse().unwrap_or_else(|e| {
                eprintln!("bad hi idx line {}: {e}", i + 1);
                std::process::exit(1);
            });
            if lo >= all_features.len() || hi >= all_features.len() {
                continue; // ignore out-of-range pairs
            }
            pairs.push((lo, hi));
            if tv_has_bands && parts.len() >= 3 {
                let band: u8 = parts[2].parse().unwrap_or(2).min(3);
                bands.push(band);
            }
        }
        let band_id = if tv_has_bands && bands.len() == pairs.len() {
            Some(bands)
        } else {
            None
        };
        let band_weights = args.tv_band_weights;
        println!(
            "Loaded {} TV pairs from {tv_path:?} (weight {}, every {}, batch {}, bands={}, band_weights={:?})",
            pairs.len(),
            args.tv_weight,
            args.tv_apply_every,
            args.tv_batch,
            band_id.is_some(),
            band_weights,
        );
        Some(TvRegularizer {
            pairs,
            features: all_features,
            weight: args.tv_weight,
            apply_every: args.tv_apply_every,
            batch: args.tv_batch,
            band_id,
            band_weights,
        })
    } else {
        None
    };

    let mut log: Vec<String> = Vec::new();
    let bake_bytes = train_mlp_with_tv(
        &groups,
        n_features,
        &hyperparams,
        &mut log,
        tv_regularizer.as_ref(),
    );

    std::fs::write(&args.out, &bake_bytes).unwrap_or_else(|e| {
        eprintln!("write {:?}: {e}", args.out);
        std::process::exit(1);
    });
    println!("Wrote {} bytes to {:?}", bake_bytes.len(), args.out);

    if let Some(log_path) = &args.log_path {
        let mut f = File::create(log_path).unwrap_or_else(|e| {
            eprintln!("create {log_path:?}: {e}");
            std::process::exit(1);
        });
        for line in &log {
            writeln!(f, "{line}").unwrap_or_else(|e| {
                eprintln!("write log: {e}");
                std::process::exit(1);
            });
        }
        println!("Wrote log ({} lines) to {log_path:?}", log.len());
    } else {
        for line in &log {
            println!("{line}");
        }
    }
}
