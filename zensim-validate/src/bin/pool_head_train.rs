//! EX-2 std-pool head trainer binary.
//!
//! Trains a single 228 → n_hidden → pool[μ, σ, max, p_6] → 4→1 reducer
//! pipeline on parquet feature corpora, emits a ZNPR v3 bake whose
//! `predict()` returns the hidden vector + `zentrain.pool_head_reducer`
//! metadata carries `[w_μ, w_σ, w_max, w_p6, b, p_norm]`.
//!
//! Runs alongside `zensim_mlp_train` — does NOT modify it. Same group
//! spec format (`--group NAME:PATH:TRAIN_W:VAL_W`), same target-column
//! flag, smaller subset of hyperparameters.
//!
//! Reference: `~/work/zen/zensim/PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md`
//! §3 (pool: μ + σ + max + p_norm; GMSD, IW-SSIM, Butteraugli) and
//! §8 (EX-2 std-pool head experiment slate).

use clap::Parser;
use std::fs;
use std::path::PathBuf;

use zensim_train_core::pool_head::{
    PoolHeadHparams, bake_pool_head_v3, train_pool_head,
};
use zensim_train_core::TrainingGroup;
use zensim_validate::parquet_loader;

#[derive(Parser, Debug)]
#[command(name = "pool_head_train")]
#[command(
    about = "EX-2 std-pool head MLP trainer. Replaces the final scalar with [μ,σ,max,p_6] + 4→1 reducer."
)]
struct Args {
    /// Group spec: `NAME:PATH:TRAIN_W:VAL_W`. Repeat for multiple groups.
    /// PATH can be .parquet or .csv (.csv ignored — parquet only).
    #[arg(long, required = true)]
    group: Vec<String>,

    /// Output bake path (ZNPR v3 bytes).
    #[arg(long)]
    out: PathBuf,

    /// Hidden vector width.
    #[arg(long, default_value_t = 128)]
    hidden: usize,

    /// Number of epochs.
    #[arg(long, default_value_t = 200)]
    epochs: usize,

    /// RankNet pair samples per epoch.
    #[arg(long, default_value_t = 50_000)]
    pairs_per_epoch: usize,

    /// Adam initial learning rate.
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,

    /// L2 weight decay on layer weights (not biases).
    #[arg(long, default_value_t = 1e-5)]
    l2: f64,

    /// LeakyReLU negative slope.
    #[arg(long, default_value_t = 0.01)]
    alpha: f64,

    /// PRNG seed.
    #[arg(long, default_value_t = 3)]
    seed: u64,

    /// Target column name in the parquet (`human_score`, `iwssim`,
    /// `mix_cv40_iw60`, etc.).
    #[arg(long, default_value = "mix_cv40_iw60")]
    target_column: String,

    /// Scalar multiplier applied to the target value at load time.
    #[arg(long, default_value_t = 100.0)]
    target_scale: f64,

    /// Max number of features to use from each group (truncates if
    /// the parquet carries more). Defaults to 228 to match the V_X
    /// embedded feature width.
    #[arg(long, default_value_t = 228)]
    max_features: usize,
}

#[derive(Debug)]
struct GroupSpec {
    name: String,
    path: PathBuf,
    train_w: f64,
    val_w: f64,
}

fn parse_group(spec: &str) -> Result<GroupSpec, String> {
    let parts: Vec<&str> = spec.splitn(4, ':').collect();
    if parts.len() != 4 {
        return Err(format!(
            "group spec {spec:?} must have 4 colon-separated parts NAME:PATH:TRAIN_W:VAL_W"
        ));
    }
    let train_w: f64 = parts[2]
        .parse()
        .map_err(|e| format!("group {spec:?} train_w parse: {e}"))?;
    let val_w: f64 = parts[3]
        .parse()
        .map_err(|e| format!("group {spec:?} val_w parse: {e}"))?;
    Ok(GroupSpec {
        name: parts[0].to_string(),
        path: PathBuf::from(parts[1]),
        train_w,
        val_w,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let specs: Vec<GroupSpec> = args
        .group
        .iter()
        .map(|s| parse_group(s))
        .collect::<Result<Vec<_>, _>>()?;
    if specs.is_empty() {
        return Err("at least one --group required".into());
    }

    // Load every group via the parquet loader.
    let mut owned_groups = Vec::with_capacity(specs.len());
    let mut n_features_seen: Option<usize> = None;
    for spec in &specs {
        if !spec.path.exists() {
            return Err(format!("group {:?}: missing path {:?}", spec.name, spec.path).into());
        }
        let mut g = parquet_loader::load_parquet(
            &spec.path,
            &spec.name,
            &args.target_column,
            args.target_scale,
        )
        .map_err(|e| format!("load {:?}: {e}", spec.path))?;
        // Truncate per-row feature width to --max-features.
        if g.n_features > args.max_features {
            for row in &mut g.feature_rows {
                row.truncate(args.max_features);
            }
            g.n_features = args.max_features;
        }
        match n_features_seen {
            None => n_features_seen = Some(g.n_features),
            Some(seen) if seen != g.n_features => {
                return Err(format!(
                    "feature-width mismatch between groups: {} has {}, prior groups have {}",
                    spec.name, g.n_features, seen
                )
                .into());
            }
            _ => {}
        }
        eprintln!(
            "[load] group={} path={:?} n_pairs={} n_features={} train_w={} val_w={}",
            spec.name,
            spec.path,
            g.human_scores.len(),
            g.n_features,
            spec.train_w,
            spec.val_w
        );
        owned_groups.push((g, spec.train_w, spec.val_w));
    }
    let n_features = n_features_seen.expect("at least one group");

    // Build TrainingGroup refs over the owned data.
    let mut group_refs_storage: Vec<Vec<&[f64]>> = Vec::with_capacity(owned_groups.len());
    for (g, _, _) in &owned_groups {
        let row_refs: Vec<&[f64]> = g.feature_rows.iter().map(|v| v.as_slice()).collect();
        group_refs_storage.push(row_refs);
    }
    let training_groups: Vec<TrainingGroup<'_>> = owned_groups
        .iter()
        .zip(group_refs_storage.iter())
        .map(|((g, train_w, val_w), refs)| TrainingGroup {
            name: g.name.clone(),
            human_scores: g.human_scores.as_slice(),
            features: refs.as_slice(),
            train_weight: *train_w,
            validation_weight: *val_w,
        })
        .collect();

    let hp = PoolHeadHparams {
        n_hidden: args.hidden,
        n_epochs: args.epochs,
        pairs_per_epoch: args.pairs_per_epoch,
        initial_lr: args.lr,
        leaky_alpha: args.alpha,
        seed: args.seed,
        l2_lambda: args.l2,
    };
    eprintln!("[train] hparams={hp:?} n_features={n_features}");
    let t0 = std::time::Instant::now();
    let model = train_pool_head(&training_groups, &hp, n_features);
    let dt = t0.elapsed();
    eprintln!(
        "[train] done dt={:.1}s reducer_w={:?} reducer_b={}",
        dt.as_secs_f64(),
        model.reducer_w,
        model.reducer_b
    );

    let bytes = bake_pool_head_v3(&model);
    eprintln!("[bake] {} bytes → {:?}", bytes.len(), args.out);
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {parent:?}: {e}"))?;
    }
    fs::write(&args.out, &bytes).map_err(|e| format!("write {:?}: {e}", args.out))?;

    Ok(())
}
