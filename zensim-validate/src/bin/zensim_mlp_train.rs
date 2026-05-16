//! Standalone binary that wraps `zensim-validate::mlp_train::train_mlp`.
//!
//! Reads one or more `--group NAME:CSV:TRAIN_W:VAL_W` CSVs (each in
//! the trainer-compatible shape `ref_basename, human_score, f0..f227`)
//! and trains the multi-group RankNet MLP. Writes ZNPR v3 bytes to
//! `--out PATH`.
//!
//! ## Canonical V0_18-methodology recipe (2026-05-14, RECOMMENDED)
//!
//! V0_18 is a **3-way concat ensemble** of three 228→128→1 single
//! MLPs averaged at the output (0.65/0.30/0.05 mix). Each component
//! has its own seed and TV config. To reproduce V0_19+ on the
//! canonical KADID/TID-perceptually-cleaned 2026-05-14 corpus:
//!
//! ```sh
//! CLEAN=/mnt/v/zen/zensim-training/2026-05-14-clean
//!
//! # Component 1: V0_16-base equivalent (no TV regularizer, seed=1).
//! zensim_mlp_train \
//!   --group safesyn:$CLEAN/safe_synth_v19_clean_features.csv:1.0:0.0 \
//!   --group kadid:$CLEAN/kadid_features.csv:0.3:1.0 \
//!   --group tid:$CLEAN/tid_features.csv:0.3:1.0 \
//!   --group konjnd:$CLEAN/konjnd_aligned_features.csv:0.5:1.0 \
//!   --hidden 128 --epochs 300 --seed 1 \
//!   --out benchmarks/v0_X_base_seed1_$(date -u +%Y-%m-%d).bin
//!
//! # Component 2: cycle-14 TV-regularized, seed=1.
//! zensim_mlp_train \
//!   <same 4 groups> \
//!   --hidden 128 --epochs 300 --seed 1 \
//!   --tv-pairs-file $CLEAN/tv_pairs_bands.tsv \
//!   --tv-weight 1.0 --tv-band-weights 10,30,10,30 \
//!   --tv-apply-every 50 --tv-batch 32 \
//!   --out benchmarks/v0_X_cycle14_s1_$(date -u +%Y-%m-%d).bin
//!
//! # Component 3: cycle-14 TV-regularized, seed=42.
//! zensim_mlp_train \
//!   <same 4 groups + same TV flags but> \
//!   --seed 42 \
//!   --out benchmarks/v0_X_cycle14_s42_$(date -u +%Y-%m-%d).bin
//!
//! # Concat three into one wide ensemble bake:
//! cargo run --release -p zensim-validate --bin concat_three_way -- \
//!   --base benchmarks/v0_X_base_seed1_...bin \
//!   --s1   benchmarks/v0_X_cycle14_s1_...bin \
//!   --s42  benchmarks/v0_X_cycle14_s42_...bin \
//!   --out  benchmarks/v0_X_concat_3way_$(date -u +%Y-%m-%d).bin
//!
//! # Affine-calibrate (α=28.0366, β=-5.0738 inherits from V0_16 lineage):
//! python3 scripts/v_next/affine_calibrate_znpr_v2.py \
//!   --in-bake  benchmarks/v0_X_concat_3way_...bin \
//!   --out-bake zensim/weights/v0_X_$(date -u +%Y-%m-%d)_f32.bin \
//!   --alpha 28.0366 --beta -5.0738
//!
//! # I8 re-quantize:
//! cargo run --release -p zensim-bench --example quant_compare -- \
//!   zensim/weights/v0_X_$(date -u +%Y-%m-%d)_f32.bin /tmp/quant
//! ```
//!
//! ## Contamination guard
//!
//! Every `--group <name>:<csv>:...` call passes the CSV through
//! `contamination_guard::scrub_csv_or_die` before training begins.
//! The 149-basename KADID+TID overlap blocklist is embedded at
//! compile time via `include_str!` from
//! `benchmarks/contamination_blocklist_2026-05-14.txt` — even if a
//! stale CSV is on disk somewhere, the trainer refuses to use it.
//! Filenames containing `CONTAMINATED` are rejected on sight before
//! any row scan.
//!
//! ## Defaults
//!
//! All other flags (--hidden 128, --epochs 300, --val-policy min,
//! --lr 1e-3, --max-features 228, --seed 1) default to the V0_16/V0_18
//! ship values; override only for cycle experiments. See
//! `benchmarks/v0_18_methodology_2026-05-13.md` for the canonical
//! per-bake methodology + reproduction.

use clap::Parser;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

#[path = "../mlp_train.rs"]
#[allow(dead_code)] // some helpers are unused in this binary
mod mlp_train;

#[path = "../contamination_guard.rs"]
mod contamination_guard;

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

    /// V0_20a row-weight boost for B3 (visually-lossless, human_score
    /// ≥ 90) rows during per-step RankNet pair sampling. Multiplies
    /// per-row sampling weight for human_score >= 90. Default 1.0 =
    /// no-op. The B3 band is small (e.g. ~43/4292 CID22 pairs, ~486/10125
    /// KADID), so val_policy=min selecting on KADID/TID/KonJND
    /// structurally underfits the visually-lossless tail. Recommended
    /// for any IW-feature (Wang 2011) or HF-targeted training where
    /// B3 SROCC is a ship gate: 2.0–4.0 depending on B3 sample density.
    /// Composes multiplicatively but B3 is mutually exclusive with B0-B2,
    /// so the multiplier only takes effect on B3 rows. Added 2026-05-14
    /// after V0_20a sweep revealed B3 underfit at default 1.0.
    #[arg(long, default_value_t = 1.0)]
    high_q_boost: f64,

    /// Optional path to dump the trainer log for the run.
    #[arg(long)]
    log_path: Option<PathBuf>,

    /// Output weight dtype for the baked ZNPR v2: `f32`, `f16`, or `i8`.
    /// Default `f32` matches every shipped bake through V0_17. `i8` cuts
    /// the bin to ~26% with per-output f32 scales; verified on V0_18
    /// (no measurable SROCC change vs V0_17 across KADID/TID/CID22/AIC).
    /// `f16` cuts to ~50% with zero SROCC change. Inference cost: F16
    /// adds a per-weight bit-shift dequant; I8 adds one fma per output
    /// after the inner loop. Both are negligible at zensim's 88K-param
    /// scale.
    #[arg(long, default_value = "f32")]
    out_dtype: String,

    /// V0_20 input-shaping research: apply a `FeatureTransform` to a
    /// specific feature column BEFORE the trainer's scaler runs. Format
    /// is `<token>:<feature_idx>[:<params>]` where:
    ///
    /// - `<token>` is one of `identity`, `log`, `log1p`, `signed_log1p`,
    ///   `signed_sqrt`, `signed_cbrt`, `clip_then_log1p`, `winsor_p99`,
    ///   `quantile_bins`.
    /// - `<feature_idx>` is in `[0, n_features)`.
    /// - `<params>` (optional) is comma-separated f32 values; required
    ///   for the parameterized variants:
    ///   - `clip_then_log1p:<idx>:<epsilon>` — 1 param
    ///   - `winsor_p99:<idx>:<p1>,<p99>` — 2 params
    ///   - `quantile_bins:<idx>:<e0>,<e1>,...,<e_{N-1}>` — N edges
    ///
    /// Repeat the flag for each transformed feature; features not listed
    /// default to `identity`.
    ///
    /// The trainer applies the transform to feature_rows in-place after
    /// CSV load, then trains as usual. The bake emits both
    /// `zentrain.feature_transforms` and (when any feature has params)
    /// `zentrain.feature_transform_params` metadata; zenpredict 0.2.0+
    /// runtime applies the same transform via
    /// `Predictor::predict_transformed`.
    ///
    /// Examples:
    /// - `--feature-transform signed_log1p:42` — no params
    /// - `--feature-transform winsor_p99:42:1.5,98.5` — clamp to [1.5, 98.5]
    /// - `--feature-transform clip_then_log1p:42:0.001` — ε=0.001
    ///
    /// See V0_20 design doc:
    /// `benchmarks/v0_20_v0_21_design_2026-05-14.md`.
    #[arg(long, value_name = "TOKEN:IDX[:PARAMS]")]
    feature_transform: Vec<String>,

    /// V0_20 input-shaping: auto-load per-feature transforms from a
    /// greedy-screen TSV (winning transform per feature with
    /// `lift >= --auto-transforms-min-lift`). Default-on path for any
    /// MLP training going forward — winsor_p99 / signed_cbrt /
    /// signed_sqrt / etc. applied per feature where the screen says
    /// they beat identity.
    ///
    /// TSV format (per `scripts/v_next/v0_20_feature_transform_greedy_screen.py`):
    /// columns `feat_idx, best_transform, params_csv, baseline_pearson,
    /// transformed_pearson, lift, baseline_spearman, n_samples`.
    ///
    /// When combined with `--feature-transform` flags, the auto-loaded
    /// set is applied first and per-flag overrides may set a feature
    /// back to identity (use `--feature-transform identity:<idx>`) or
    /// switch its transform. Conflicts (auto-loaded + flag both
    /// non-identity, non-matching) are an error.
    ///
    /// Recommended default: `--auto-transforms benchmarks/v0_20_feature_transform_greedy_screen_2026-05-15.tsv`
    /// for any new MLP bake. Per the methodology critique
    /// (CLAUDE.md "SROCC-only verdicts BANNED" section), this is the
    /// least-controversial training-side win measured to date —
    /// V_20 IS (98 transforms) wins KADID + TID across the full
    /// Mohammadi panel.
    #[arg(long, value_name = "PATH")]
    auto_transforms: Option<PathBuf>,

    /// Min Pearson-lift threshold for `--auto-transforms`. Features in
    /// the screen TSV with `lift < this` are left as identity. Default
    /// 0.05 matches the V_20 IS adopted set (98 of 228 features at
    /// idx < 228, 139 at idx < 300).
    ///
    /// Lower values (0.02) admit more transforms but include borderline
    /// ones the trainer may not need. D3 sweep showed lift ≥ 0.10 cuts
    /// to 60 transforms and gives back half the B3 lift — don't trim
    /// aggressively without re-testing.
    #[arg(long, default_value_t = 0.05, value_name = "FLOAT")]
    auto_transforms_min_lift: f64,
}

struct LoadedGroup {
    name: String,
    train_w: f64,
    val_w: f64,
    human_scores: Vec<f64>,
    feature_rows: Vec<Vec<f64>>,
    n_features: usize,
}

/// Load per-feature transforms from a screen TSV (output of
/// `scripts/v_next/v0_20_feature_transform_greedy_screen.py`). Populates
/// `transforms` and `params` in place for every row where `lift >=
/// min_lift` AND `feat_idx < n_features`. Returns the count of
/// transforms actually loaded (i.e., non-identity entries written).
///
/// TSV columns expected: feat_idx, best_transform, params_csv,
/// baseline_pearson, transformed_pearson, lift, baseline_spearman,
/// n_samples (header row is required).
fn load_auto_transforms_from_screen(
    tsv_path: &PathBuf,
    min_lift: f64,
    n_features: usize,
    transforms: &mut [zenpredict::FeatureTransform],
    params: &mut [Vec<f32>],
) -> usize {
    let file = match File::open(tsv_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!(
                "--auto-transforms: cannot open {}: {e}",
                tsv_path.display()
            );
            std::process::exit(2);
        }
    };
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let header_line = match lines.next() {
        Some(Ok(l)) => l,
        _ => {
            eprintln!("--auto-transforms: empty TSV {}", tsv_path.display());
            std::process::exit(2);
        }
    };
    let header: Vec<&str> = header_line.split('\t').collect();
    let col = |name: &str| -> usize {
        header.iter().position(|c| c.trim() == name).unwrap_or_else(|| {
            eprintln!(
                "--auto-transforms: missing column {name:?} in TSV header {:?}",
                header
            );
            std::process::exit(2);
        })
    };
    let idx_col = col("feat_idx");
    let xform_col = col("best_transform");
    let params_col = col("params_csv");
    let lift_col = col("lift");

    let mut loaded = 0usize;
    // map_while stops on Err — flatten() would infinite-loop on a
    // sticky read error (per clippy::lines_filter_map_ok).
    for line in lines.map_while(Result::ok) {
        if line.trim().is_empty() {
            continue;
        }
        let cells: Vec<&str> = line.split('\t').collect();
        if cells.len() <= idx_col.max(xform_col).max(params_col).max(lift_col) {
            continue;
        }
        let Ok(idx) = cells[idx_col].trim().parse::<usize>() else {
            continue;
        };
        if idx >= n_features {
            continue;
        }
        let Ok(lift) = cells[lift_col].trim().parse::<f64>() else {
            continue;
        };
        if lift < min_lift {
            continue;
        }
        let token = cells[xform_col].trim();
        if token == "identity" || token.is_empty() {
            continue;
        }
        let transform = match zenpredict::FeatureTransform::from_token(token) {
            Ok(t) => t,
            Err(_) => {
                eprintln!(
                    "--auto-transforms: unknown transform token {token:?} at idx {idx} (lift {lift:.4}); skipping"
                );
                continue;
            }
        };
        let params_str = cells[params_col].trim();
        let parsed_params: Vec<f32> = if params_str.is_empty() {
            Vec::new()
        } else {
            params_str
                .split(',')
                .filter_map(|s| s.trim().parse::<f32>().ok())
                .collect()
        };
        transforms[idx] = transform;
        params[idx] = parsed_params;
        loaded += 1;
    }
    loaded
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
        // 2026-05-14 contamination guard: refuse to load any CSV that
        // contains a KADID/TID-overlap basename. Exits 2 with a loud
        // message if the input is contaminated. See
        // `crate::contamination_guard::BLOCKLIST_TXT` for the 149
        // basenames committed at
        // benchmarks/contamination_blocklist_2026-05-14.txt.
        contamination_guard::scrub_csv_or_die(&path).unwrap_or_else(|e| {
            eprintln!("contamination_guard read error on {}: {e}", path.display());
            std::process::exit(1);
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

    // V0_20 input-shaping: parse `--feature-transform TOKEN:IDX[:PARAMS]`
    // repeating flags into a per-feature transform list. Default
    // is all-Identity. Apply transforms to feature_rows in-place so
    // the trainer's scaler is fit to the post-transform distribution.
    //
    // Parameterized variants (clip_then_log1p / winsor_p99 / quantile_bins)
    // accept comma-separated f32 params after the third colon. Per-feature
    // params accumulate in a parallel `Vec<Vec<f32>>`; an empty inner vec
    // means "no params for this feature".
    //
    // When `--auto-transforms` is set, pre-populate the transform list
    // from the screen TSV first; then `--feature-transform` flags
    // override per-feature (or extend if the auto set didn't cover
    // that feature).
    let (feature_transforms, feature_transform_params): (
        Option<Vec<zenpredict::FeatureTransform>>,
        Option<Vec<Vec<f32>>>,
    ) = if args.feature_transform.is_empty() && args.auto_transforms.is_none() {
        (None, None)
    } else {
        let mut transforms = vec![zenpredict::FeatureTransform::Identity; n_features];
        let mut params: Vec<Vec<f32>> = vec![Vec::new(); n_features];

        // Step 1: load auto-transforms from screen TSV if specified.
        if let Some(tsv_path) = &args.auto_transforms {
            let n_loaded = load_auto_transforms_from_screen(
                tsv_path,
                args.auto_transforms_min_lift,
                n_features,
                &mut transforms,
                &mut params,
            );
            eprintln!(
                "--auto-transforms: loaded {n_loaded} transforms from {} (min-lift={})",
                tsv_path.display(),
                args.auto_transforms_min_lift,
            );
        }

        // Step 2: apply explicit --feature-transform flags (may override
        // auto-loaded entries when the flag's transform matches the auto
        // value; conflicts error out below).
        for spec in &args.feature_transform {
            // splitn(3) — allow PARAMS to contain ':' if needed
            // (we don't currently support that, but splitn(3) keeps
            // the door open and isolates the token + idx cleanly).
            let parts: Vec<&str> = spec.splitn(3, ':').collect();
            if parts.len() < 2 {
                eprintln!("--feature-transform expects TOKEN:IDX[:PARAMS] (got {spec:?})");
                std::process::exit(2);
            }
            let token = parts[0];
            let idx_str = parts[1];
            let params_str = parts.get(2).copied().unwrap_or("");
            let idx: usize = idx_str.parse().unwrap_or_else(|_| {
                eprintln!("--feature-transform: bad idx {idx_str:?}");
                std::process::exit(2);
            });
            if idx >= n_features {
                eprintln!("--feature-transform: idx {idx} >= n_features {n_features}");
                std::process::exit(2);
            }
            let transform = zenpredict::FeatureTransform::from_token(token).unwrap_or_else(|_| {
                eprintln!(
                    "--feature-transform: unknown token {token:?} (valid: \
                     identity, log, log1p, signed_log1p, signed_sqrt, \
                     signed_cbrt, clip_then_log1p, winsor_p99, quantile_bins)"
                );
                std::process::exit(2);
            });
            if transforms[idx] != zenpredict::FeatureTransform::Identity
                && transforms[idx] != transform
            {
                eprintln!(
                    "--feature-transform: feature {idx} already set to {:?}, \
                     can't override with {transform:?}",
                    transforms[idx]
                );
                std::process::exit(2);
            }
            let parsed_params: Vec<f32> = if params_str.is_empty() {
                Vec::new()
            } else {
                params_str
                    .split(',')
                    .map(|s| {
                        s.trim().parse::<f32>().unwrap_or_else(|_| {
                            eprintln!("--feature-transform: bad param {s:?} in {spec:?}");
                            std::process::exit(2);
                        })
                    })
                    .collect()
            };
            // Validate param count matches variant expectation
            match transform {
                zenpredict::FeatureTransform::ClipThenLog1p => {
                    if !parsed_params.is_empty() && parsed_params.len() != 1 {
                        eprintln!(
                            "--feature-transform clip_then_log1p:{idx}: expected 1 param (epsilon), got {}",
                            parsed_params.len()
                        );
                        std::process::exit(2);
                    }
                }
                zenpredict::FeatureTransform::WinsorP99 => {
                    if !parsed_params.is_empty() && parsed_params.len() != 2 {
                        eprintln!(
                            "--feature-transform winsor_p99:{idx}: expected 2 params (p1, p99), got {}",
                            parsed_params.len()
                        );
                        std::process::exit(2);
                    }
                    if parsed_params.len() == 2 && parsed_params[0] > parsed_params[1] {
                        eprintln!(
                            "--feature-transform winsor_p99:{idx}: p1={} > p99={}; pass them in (low, high) order",
                            parsed_params[0], parsed_params[1]
                        );
                        std::process::exit(2);
                    }
                }
                zenpredict::FeatureTransform::QuantileBins => {
                    if parsed_params.len() < 2 && !parsed_params.is_empty() {
                        eprintln!(
                            "--feature-transform quantile_bins:{idx}: need at least 2 edges (got {})",
                            parsed_params.len()
                        );
                        std::process::exit(2);
                    }
                    // Validate edges sorted ascending
                    for w in parsed_params.windows(2) {
                        if w[0] > w[1] {
                            eprintln!(
                                "--feature-transform quantile_bins:{idx}: edges not sorted ascending"
                            );
                            std::process::exit(2);
                        }
                    }
                }
                _ => {
                    // Non-parameterized variants ignore params; warn if any provided
                    if !parsed_params.is_empty() {
                        eprintln!(
                            "warning: --feature-transform {token}:{idx} ignores params {parsed_params:?} \
                             (this variant doesn't consume any)"
                        );
                    }
                }
            }
            transforms[idx] = transform;
            params[idx] = parsed_params;
        }
        // Apply per-feature transforms in-place across every group's
        // feature_rows. The trainer's scaler sees the transformed
        // distribution and the bake's metadata records the list so
        // the runtime applies the same transform before its scaler.
        let summary = transforms
            .iter()
            .enumerate()
            .filter(|(_, t)| **t != zenpredict::FeatureTransform::Identity)
            .map(|(i, t)| {
                if params[i].is_empty() {
                    format!("f{i}={}", t.as_token())
                } else {
                    let pstr = params[i]
                        .iter()
                        .map(|v| format!("{v}"))
                        .collect::<Vec<_>>()
                        .join(",");
                    format!("f{i}={}({pstr})", t.as_token())
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        println!("feature_transforms: {summary}");
        for g in &mut loaded {
            for row in &mut g.feature_rows {
                for (i, t) in transforms.iter().enumerate() {
                    if *t != zenpredict::FeatureTransform::Identity {
                        row[i] = t.apply_with_params(row[i] as f32, &params[i]) as f64;
                    }
                }
            }
        }
        let any_params = params.iter().any(|p| !p.is_empty());
        (
            Some(transforms),
            if any_params { Some(params) } else { None },
        )
    };

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

    let out_dtype = match args.out_dtype.to_ascii_lowercase().as_str() {
        "f32" => zenpredict::WeightDtype::F32,
        "f16" => zenpredict::WeightDtype::F16,
        "i8" => zenpredict::WeightDtype::I8,
        other => {
            eprintln!("--out-dtype must be one of f32 / f16 / i8, got {other:?}");
            std::process::exit(2);
        }
    };

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
        high_q_boost: args.high_q_boost,
        out_dtype,
        feature_transforms: feature_transforms.clone(),
        feature_transform_params: feature_transform_params.clone(),
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
