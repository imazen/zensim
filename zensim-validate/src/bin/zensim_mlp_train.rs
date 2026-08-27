//! Standalone binary that wraps `zensim-validate::mlp_train::train_mlp`.
//!
//! Reads one or more `--group NAME:CSV:TRAIN_W:VAL_W` CSVs (each in
//! the trainer-compatible shape `ref_basename, <target>, f0..f227`)
//! and trains the multi-group RankNet MLP. Writes ZNPR v3 bytes to
//! `--out PATH`.
//!
//! The target column defaults to `human_score` (legacy ssim2-derived
//! score, multiplied by 100 to match score_zensim scale). Use
//! `--target-column NAME` to retarget the regression — `iwssim`
//! (already available on the safesyn corpus) or future
//! `cvvdp_pycvvdp_v054` once the corpus carries it. Pair with
//! `--target-scale` if the new column is not in `[0, 1]`. This breaks
//! the ssim2-target training bias documented in `CLAUDE.md > SROCC-only
//! verdicts BANNED + ssim2-target training bias` (2026-05-15).
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

use clap::parser::ValueSource;
use clap::{CommandFactory, FromArgMatches, Parser};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use zensim_validate::mlp_train;
use zensim_validate::mlp_train::{GroupLossMode, TripletPool, train_mlp_strategy};
use zensim_validate::train_manifest;

#[path = "../contamination_guard.rs"]
mod contamination_guard;

use mlp_train::{
    AnchorRows, EquivPairs, KonjndAggregationPool, MlpHyperparams, TrainingGroup, TvRegularizer,
    ValidationPolicy,
};

#[derive(Parser)]
#[command(
    name = "zensim_mlp_train",
    about = "Rust RankNet MLP trainer (defaults match V0_16 ship recipe — see benchmarks/recipe_v0_16.sh for the full invocation)"
)]
struct Args {
    /// Group spec: NAME:PATH:TRAIN_WEIGHT:VAL_WEIGHT[:withinref].
    /// Repeat for each dataset. Header must include `ref_basename`,
    /// `human_score`, and `f0..f<N-1>` (or `feat_0..`). `human_score`
    /// is in [0, 1] and is multiplied by 100 internally to match
    /// `score_zensim` scale.
    ///
    /// Required unless `--manifest` is given (the manifest's `groups`
    /// array supplies the group set). Passing any explicit `--group`
    /// alongside `--manifest` REPLACES the manifest's groups entirely.
    ///
    /// The optional 5th field `withinref` draws every RankNet pair from
    /// WITHIN one reference image instead of uniformly across the group.
    /// Use it when the group's signal is a per-reference distortion
    /// ladder: cross-image pairs otherwise teach between-image scale and
    /// drown the ladder out. On the near-lossless HF corpus the ladder
    /// moves ~0.92 ssim2 points within an image versus ~6 points between
    /// images, so uniform pairing leaves it ~1/7th of the gradient.
    /// Requires ref identity on the source (`ref_basename` or
    /// `image_path`); the trainer exits 2 rather than silently falling
    /// back to uniform pairing.
    #[arg(long, required_unless_present = "manifest")]
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

    /// Validation policy: "min" (worst per-group score, V0_5 default)
    /// or "mean". Applied AFTER per-group multi-stat aggregation.
    #[arg(long, default_value = "min")]
    val_policy: String,

    /// Per-group stat aggregation for checkpoint selection:
    ///   srocc    — legacy single-stat (backward compat)
    ///   geomean3 — geometric mean of (SROCC, PLCC, PWRC) [DEFAULT]
    ///   harmean3 — harmonic mean of same
    ///   min3     — min of (SROCC, PLCC, PWRC)
    ///
    /// Mohammadi 2025 shows SROCC alone is "the single most misleading
    /// practice" in IQA evaluation. geomean3 captures rank accuracy,
    /// calibration linearity, and perceptual weighting — the three
    /// cheaply-computable axes — in a single checkpoint score.
    #[arg(long, default_value = "geomean3")]
    val_aggregate: String,

    /// Random seed. Default 1 matches V0_16 ship.
    #[arg(long, default_value_t = 1)]
    seed: u64,

    /// Log every N epochs.
    #[arg(long, default_value_t = 10)]
    log_every: usize,

    /// Early-stop patience (epochs of no validation improvement). 0 disables.
    #[arg(long, default_value_t = 50)]
    early_stop_patience: usize,

    /// Output path for the trained ZNPR v3 bake. Required unless
    /// `--manifest` is given (the manifest supplies it from `[bake].file`).
    #[arg(long, required_unless_present = "manifest")]
    out: Option<PathBuf>,

    /// Cap features at the first N columns. **Default 372** — the
    /// with-iw runtime width that `ZensimProfile::A` (v47) ships and the
    /// canonical 372-col validation corpora supply. The legacy 228/300
    /// regimes are BANNED for ship bakes: a narrower cap silently drops
    /// the extended + IW-pool blocks (f228..f371), producing a bake that
    /// can't be validated against the 372-col corpora and mis-aligns the
    /// train/inference feature space (this footgun shipped a 228-input
    /// probe on 2026-06-30). Values < 372 are rejected unless
    /// `--allow-narrow-features` is also passed (research/classifier only,
    /// never a runtime weight).
    #[arg(long, default_value_t = 372)]
    max_features: usize,

    /// Scale-mass regularizer (E-M6 remedy): L2 multiplier on layer-1 rows of
    /// COARSE-scale inputs (basic s2/s3 f78-155, v2 s2/s3 f546-719, append
    /// s2/s3 f822-923). >1 pushes gradient mass toward fine scales so the
    /// ModelSensitivity fold keeps spatial resolution (coarse concentration
    /// collapses the map to 1/8-res: M3 0.11-0.25 at M2 0.99). 1.0 = uniform
    /// L2, bit-identical legacy behavior.
    #[arg(long, default_value_t = 1.0)]
    coarse_l2_mult: f64,

    /// Decoupled (AdamW-style) weight-decay RATE on coarse-scale layer-1 rows,
    /// applied after each Adam step (bypasses Adam's rescaling, which
    /// neutralizes coupled L2 — measured 2026-07-29). Effective per-step decay
    /// = lr * rate * coarse_l2_mult (so pair with --coarse-l2-mult as the
    /// row marker/multiplier; mult defaults to marking coarse rows at 1x when
    /// only the rate is given). 0.0 = off.
    #[arg(long, default_value_t = 0.0)]
    coarse_decay: f64,

    /// Escape hatch for `--max-features < 372` (BANNED by default —
    /// narrow bakes can't ship as runtime weights). Research/content-
    /// classifier use only; never produces a `ZensimProfile` bake.
    #[arg(long, default_value_t = false)]
    allow_narrow_features: bool,

    /// FEATURE-SUBSET ablation: restrict the fit to this set of input
    /// indices. Everything else is dropped — its raw column is zeroed
    /// before the scaler runs (so its standardized value is exactly 0.0)
    /// and its layer-1 row is pinned to 0.0 at init, which makes the run an
    /// *exact* K-wide fit that still draws the same pairs and the same init
    /// normals as the full-width run at the same seed. The baked layer-1
    /// rows come out exactly zero ⇒ prunable by `bake_dial_refit pack` into
    /// a genuinely narrower, faster bake with identical predictions.
    ///
    /// SPEC is either an inline comma-separated list (`0,5,17`) or a path to
    /// a file of whitespace/comma-separated indices (`#` comments allowed).
    /// Indices must be `< --max-features`; duplicates are collapsed.
    #[arg(long, value_name = "SPEC")]
    keep_features: Option<String>,

    /// GROUP-LASSO (group-L1 / ℓ2,1) strength over layer-1 input columns:
    /// penalty `λ · Σ_k ‖W1[k,:]‖₂`, applied as a DECOUPLED proximal
    /// block-soft-threshold after each Adam step (threshold `lr·λ`). Drives
    /// whole input columns to EXACTLY zero, so the fit *learns* its own
    /// feature subset in one run instead of ranking post-hoc. A coupled
    /// subgradient would be both neutralized by Adam's rescaling and unable
    /// to reach exact zero — see `apply_group_l1`. 0.0 = off (bit-identical
    /// to every historical recipe).
    #[arg(long, default_value_t = 0.0, value_name = "LAMBDA")]
    group_l1: f64,

    /// TV-regularizer pair indices TSV. Two columns: lo_trainer_idx,
    /// hi_trainer_idx. Indices reference rows in the concatenated
    /// trainer-feature space (group 0 first, then group 1, etc.).
    /// Penalty per pair: `max(0, pred[hi] - pred[lo])`.
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

    /// Anti-collapse margin for the within-ladder TV hinge. Penalty
    /// becomes `max(0, y_harsher - y_milder + margin)`, forcing a
    /// minimum per-step gap between adjacent severity levels. 0.0 =
    /// pure hinge (can collapse the ladder flat under high weight).
    /// A small positive value (raw-output units) spreads the ladder,
    /// preserving dynamic range + analytic-corpus rank while keeping
    /// monotonicity. Only affects the --per-sample-alpha-head path.
    #[arg(long, default_value_t = 0.0)]
    tv_margin: f64,

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

    /// Name of the column in each group CSV to use as the regression
    /// target. Default `human_score` matches every V_X bake through
    /// V_20 — the column carries the ssim2-derived score per pair.
    ///
    /// **Why this flag exists**: every V_X bake trained against the
    /// default produces an ssim2-shaped output surface, and any
    /// SROCC-against-ssim2-derived-MOS evaluation favors that shape by
    /// construction (the "ssim2-target training bias" documented in
    /// `CLAUDE.md > SROCC-only verdicts BANNED`). To get an
    /// ssim2-independent bake, train against a different target column
    /// — `iwssim` (Wang & Li 2011, available on safesyn at
    /// `/mnt/v/zen/zensim-training/2026-05-16/safesyn_with_iwssim.csv`)
    /// or `cvvdp_pycvvdp_v054` (Mantiuk, coming via zenmetrics CVVDP
    /// sweep) once features are extracted onto a corpus carrying that
    /// column.
    ///
    /// The CSV header must contain the named column; missing column =
    /// hard error. Pair this with `--target-scale` if the new column
    /// is not in `[0, 1]` — the trainer multiplies the loaded value by
    /// `--target-scale` to bring it to `score_zensim` (0..100) units,
    /// matching the legacy `human_score * 100` convention. The per-row
    /// boost thresholds (`--low-q-boost` at < 50, `--mid-q-boost` at
    /// 50..90, `--high-q-boost` at ≥ 90) operate on the scaled value,
    /// so they still align with band cutoffs after the scale.
    ///
    /// Examples:
    /// - `--target-column iwssim` (IW-SSIM ∈ [0, 1], default scale 100)
    /// - `--target-column cpu_ssimulacra2 --target-scale 1.0`
    ///   (already in score units; pass scale 1.0 to avoid x100)
    /// - `--target-column cvvdp_jod --target-scale 10.0`
    ///   (CVVDP JOD ∈ [0, 10]; x10 brings to 0..100 band-cutoff space)
    #[arg(long, default_value = "human_score", value_name = "NAME")]
    target_column: String,

    /// Multiplier applied to the loaded `--target-column` value before
    /// training. Default 100.0 matches the legacy `human_score * 100`
    /// convention (human_score is in [0, 1]; multiplied to land in the
    /// 0..100 `score_zensim` band-cutoff space).
    ///
    /// Set to 1.0 if the target column is already in `score_zensim`
    /// units (e.g., `cpu_ssimulacra2`). Set to 10.0 for CVVDP JOD
    /// (∈ [0, 10]) to bring to 0..100. The per-row boost thresholds
    /// (`--low-q-boost`, `--mid-q-boost`, `--high-q-boost`) and the
    /// validation SROCC reporting both operate on the scaled value.
    #[arg(long, default_value_t = 100.0, value_name = "FLOAT")]
    target_scale: f64,

    /// T8.1 (2026-05-16): mini-batch SGD size. Default 1 = per-pair
    /// Adam (bit-identical to legacy / V_18 / V_22-IW trainer). When
    /// > 1, the trainer accumulates K RankNet pair gradients between
    /// > each Adam step, with a final-flush at epoch end if
    /// > `pairs_per_epoch % K != 0`.
    ///
    /// **Convergence implications**: less noisy gradients than per-pair
    /// SGD. Usually helps generalization. Can hurt regularization on
    /// small datasets — Adam's bias correction `(1 - β^t)` decays K×
    /// slower because `t` increments K× less often. Mathematically
    /// still correct, just on a different schedule.
    ///
    /// **Determinism**: same `--seed N` produces same sample sequence
    /// regardless of K (only Adam call cadence changes). Bake bytes
    /// for `--minibatch-size 1` are bit-identical to the legacy trainer.
    ///
    /// Recommended K-ablation: K ∈ {1, 8, 64, 256} on the target
    /// recipe before flipping. See `benchmarks/trainer_perf_analysis_2026-05-16.md`.
    #[arg(long, default_value_t = 1, value_name = "K")]
    minibatch_size: usize,

    /// **DEPRECATED**: rayon parallel-batch is now ALWAYS on when
    /// `--minibatch-size > 1` (no behavior change vs sequential —
    /// bit-identical bake bytes per the T8.2 determinism gate). The
    /// flag is kept for backwards compatibility but has no effect.
    /// Removed CLAUDE.md "fast mode on by default" gate, 2026-05-17.
    #[arg(long, default_value_t = true, hide = true)]
    parallel_batch: bool,

    /// Enable PWRC-aligned pair weighting (Wu et al. 2018, IEEE TIP
    /// DOI 10.1109/TIP.2018.2799331; reference MATLAB at
    /// <https://github.com/wqb-uestc/PWRC>). When set, each drawn
    /// RankNet pair's loss and gradient are multiplied by the
    /// label-only PWRC weight `exp(max(MOS_a, MOS_b) / 100)` (closed-
    /// form default), and pairs with `|ΔMOS| < --pwrc-sensory-
    /// threshold` are dropped as perceptually tied.
    ///
    /// **Off (default)**: bit-identical to the pre-PWRC trainer —
    /// every pair contributes weight 1.0, no threshold dropping.
    ///
    /// **Determinism note**: enabling PWRC does not change the RNG
    /// stream; same `--seed N` yields the same `(group, ia, ib)`
    /// draw sequence, with some pairs filtered out by the sensory
    /// threshold. The Adam mini-batch counter advances only for
    /// gradient-contributing draws.
    ///
    /// Use `--pwrc-band-weights` to override the closed-form weight
    /// with an explicit per-band schedule (e.g., to invert the Wu
    /// 2018 direction and upweight B0..B5, the zensim "low-q
    /// priority" target per CLAUDE.md "B0..B5 lift is the dominant
    /// priority"). See `benchmarks/v0_X_pwrc_design_2026-05-17.md`
    /// for design rationale + expected ablation behavior.
    #[arg(long, default_value_t = false)]
    pwrc_pair_weight: bool,

    /// PWRC sensory threshold `T` (Wu et al. 2018) on the same scale
    /// as `human_score` (= `score_zensim` 0..100 after `--target-
    /// scale`). Pairs with `|MOS_a - MOS_b| < T` are dropped from
    /// training as perceptually tied. Default 5.0 matches the
    /// published recommendation (~5 % of a 100-unit MOS range).
    ///
    /// Set to 0.0 to disable threshold-based dropping while keeping
    /// the per-pair weighting active. Active only when `--pwrc-pair-
    /// weight` is set.
    #[arg(long, default_value_t = 5.0, value_name = "FLOAT")]
    pwrc_sensory_threshold: f64,

    /// Optional explicit per-band weight vector for `--pwrc-pair-
    /// weight`. Comma-separated f64 values; each entry's bin is
    /// `[i*100/N, (i+1)*100/N)` for `N = len(values)`. Example:
    ///
    /// - `--pwrc-band-weights 5,4,3,2,1.5,1,1,1,1,1` (10 bands, B0..B9)
    ///   upweights low-q (zensim "B0..B5 lift" target).
    /// - `--pwrc-band-weights 1,1.2,1.5,2.0` (4 bands, [0,25),
    ///   [25,50), [50,75), [75,100]) approximates the closed-form
    ///   Wu 2018 schedule with coarse bins.
    ///
    /// When omitted, the trainer uses the closed-form `exp(max_MOS /
    /// 100)` weight from the Wu 2018 paper (label-only piece; see
    /// `MlpHyperparams::pwrc_pair_weight` doc). Active only when
    /// `--pwrc-pair-weight` is set.
    #[arg(long, value_parser = parse_pwrc_band_weights, value_name = "W0,W1,...")]
    pwrc_band_weights: Option<Vec<f64>>,

    /// Norm-in-Norm + RankNet hybrid loss weight `β` (Li, Jiang, Jiang
    /// 2020, "Norm-in-Norm Loss with Faster Convergence and Better
    /// Performance for Image Quality Assessment", ACM MM, arXiv:
    /// 2008.03889; reference impl
    /// <https://github.com/lidq92/LinearityIQA>).
    ///
    /// **Default 0.0 = pure RankNet** (bit-identical bake bytes to the
    /// legacy / pre-NiN trainer at any `--minibatch-size`). When set
    /// to a positive value (paper recommends 0.1), an auxiliary loss
    /// term is added on top of every mini-batch's RankNet gradients:
    ///
    /// ```text
    ///   total = ranknet + norm_in_norm_weight · norm_in_norm_loss
    /// ```
    ///
    /// The Norm-in-Norm loss is computed over the 2K predictions
    /// generated by the K pair forwards in the mini-batch (with
    /// `ε = 1e-8`):
    ///
    /// ```text
    ///   ŝ_n = (Ŝ - mean(Ŝ)) / (||Ŝ - mean(Ŝ)||_q + ε)
    ///   s_n = (-MOS - mean(-MOS)) / (||-MOS - mean(-MOS)||_q + ε)
    ///   loss_NiN = (||ŝ_n - s_n||_p / scale)^p
    ///   scale    = 2^max(1, 1/q) · N^max(0, 1/p - 1/q)
    /// ```
    ///
    /// Per Li 2020 Table 2 last row (KonIQ-10k headline result), the
    /// recommended hybrid is `β = 0.1, p = 1, q = 2` — that
    /// configuration lifted SROCC 0.928 → 0.937 and PLCC 0.928 →
    /// 0.947 vs RankNet alone.
    ///
    /// **Requires `--minibatch-size >= 16`** for stable batch
    /// statistics. The trainer errors out at K < 16 when this is set.
    #[arg(long, default_value_t = 0.0, value_name = "FLOAT")]
    norm_in_norm_weight: f64,

    /// Inner-norm exponent `p` for the Norm-in-Norm loss (Li 2020).
    /// Default 1.0 per Table 1 (best single-loss config) and the
    /// recommended (β=0.1, p=1, q=2) hybrid. Set to 2.0 alongside
    /// `--norm-in-norm-q 2.0` to recover the PLCC-induced special
    /// case (paper Section 2.2, Eqn 14): `loss ∝ (1 − PLCC)`.
    ///
    /// Only meaningful when `--norm-in-norm-weight > 0`.
    #[arg(long, default_value_t = 1.0, value_name = "FLOAT")]
    norm_in_norm_p: f64,

    /// Outer (denominator) q-norm exponent for the Norm-in-Norm loss
    /// (Li 2020). Default 2.0 recovers the z-score-equivalent
    /// normalization of the reference impl.
    ///
    /// Only meaningful when `--norm-in-norm-weight > 0`.
    #[arg(long, default_value_t = 2.0, value_name = "FLOAT")]
    norm_in_norm_q: f64,

    /// EX-2 std-pool head (`PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md §3`).
    /// When set, the trainer replaces the standard `n_hidden → 1`
    /// linear output with `pool[μ, σ, max, p_6] → 4 → 1` reducer
    /// (GMSD's std-pooling + Butteraugli p-norm + IW-style pooling).
    /// The bake emits a passthrough second layer + a
    /// `zentrain.pool_head_reducer` metadata key carrying
    /// `[w_μ, w_σ, w_max, w_p6, b, p_norm]`. Runtime detects the
    /// metadata and routes through `pool_head::forward_pool_head`.
    ///
    /// **Current limitations** (v0 prod wire-in, 2026-05-18):
    /// - Pool-head backprop is scalar (SIMD parity work queued; see
    ///   `MlpHyperparams::pool_head` doc).
    /// - `--norm-in-norm-weight > 0` is incompatible (errors out).
    /// - Parallel-batch is silently ignored (sequential mini-batch
    ///   path; pool-head per-pair work is small enough that the K=256
    ///   recipe still trains in ~25 min wall at h=128).
    ///
    /// **What composes** (verified in unit tests):
    /// - `--minibatch-size K` (sequential gradient accumulation, Adam
    ///   step every K pairs + final-flush).
    /// - `--pwrc-pair-weight` + `--pwrc-sensory-threshold` +
    ///   `--pwrc-band-weights`.
    /// - TV regularizer (`--tv-pairs-file` + `--tv-weight`).
    /// - L2 (`--l2`) on layer-1 weights and reducer weights.
    /// - Low/Mid/High-q row boosts (`--low-q-boost` etc).
    /// - Cosine LR (50-epoch period), early stop on val SROCC.
    #[arg(long, default_value_t = false)]
    pool_head: bool,

    /// EX-2 follow-up: hybrid pool + rank head. When set, the trainer
    /// runs BOTH the standard `n_hidden → 1` rank-net head AND the
    /// `pool[μ,σ,max,p_6] → 4 → 1` reducer on the same encoder, then
    /// blends via a sigmoid-bounded learned `α` (`y = α·y_rank + (1−α)·y_pool`).
    /// Mutually exclusive with `--pool-head`. The bake emits a
    /// passthrough second layer + a `zentrain.hybrid_head` metadata key
    /// carrying `[rank_w[n_hidden] | rank_b | α_logit | reducer_w[4] |
    /// reducer_b | p_norm]`. Runtime detects the key and routes through
    /// `hybrid_head::apply_hybrid_head_runtime`.
    ///
    /// **Current v0 limitations** (2026-05-18):
    /// - `--norm-in-norm-weight > 0` not yet composed (trainer panics).
    /// - Parallel-batch flag silently ignored (sequential mini-batch).
    /// - SIMD backprop queued.
    ///
    /// **What composes**:
    /// - `--minibatch-size K`, `--pwrc-pair-weight`, `--pwrc-sensory-threshold`,
    ///   `--pwrc-band-weights`, TV regularizer, L2, low/mid/high-q boosts.
    #[arg(long, default_value_t = false)]
    hybrid_head: bool,

    /// EX-2 follow-up²: per-sample α head. Replaces the scalar α in
    /// `--hybrid-head` with a learned function `α(x) = sigmoid(W_α · h +
    /// b_α)` predicted from the encoder's hidden vector. Lets the
    /// model assign α per-pair so photo-like inputs (CID22-shaped)
    /// pull α toward rank-dominant while JND-step-grid inputs
    /// (KonJND-shaped) pull α toward pool-dominant.
    ///
    /// Mutually exclusive with `--pool-head` AND `--hybrid-head`. The
    /// bake metadata key changes to `zentrain.per_sample_alpha_head`
    /// with payload `[W_α[n_hidden] | b_α | rank_w[n_hidden] |
    /// rank_b | reducer_w[4] | reducer_b | p_norm]` (size
    /// `4·(2·n_hidden + 8)`).
    ///
    /// **What composes**: NiN composition (mandatory for V_22-LARGE
    /// recipe), `--minibatch-size K`, PWRC weights, L2 on layer-1 +
    /// rank_w + reducer_w + W_α (b_α unregularized),
    /// low/mid/high-q row boosts. **What is omitted**: TV regularizer
    /// (skipped on this path — V_22 recipe doesn't use TV).
    #[arg(long, default_value_t = false)]
    per_sample_alpha_head: bool,

    /// Add a learned input→output skip connection (372→1 linear)
    /// alongside the MLP. Output = MLP(x) + skip(x). Lets features
    /// with a direct linear relationship to quality bypass the
    /// hidden-layer bottleneck. ~375 extra params.
    #[arg(long, default_value_t = false)]
    skip_connection: bool,

    /// Number of hidden layers. 1 = 372→128→heads (default).
    /// 2 = 372→128→64→heads (second layer width = n_hidden/2).
    #[arg(long, default_value_t = 1)]
    n_hidden_layers: usize,

    /// `PreviewV0_5Tuner` MSE auxiliary loss weight (2026-05-18).
    /// Default `0.0` = pure RankNet. When `> 0`, adds an auxiliary
    /// regression loss `mse_weight·(y - target)²` (averaged across
    /// the 2·pairs_per_epoch predictions per epoch) on top of the
    /// RankNet pair gradient. Only wired on `--per-sample-alpha-head`;
    /// not yet composed with `--norm-in-norm-weight > 0`.
    #[arg(long, default_value_t = 0.0)]
    mse_weight: f64,

    /// Weight MSE loss by 1/max(σ, 0.05)² where σ is per-row metric
    /// disagreement (std of normalized cvvdp, iwssim, ssim2). Directly
    /// optimizes Z-RMSE — errors on high-consensus stimuli cost more.
    /// Requires parquet inputs with cvvdp_score + iwssim + ssim2_gpu columns.
    #[arg(long, default_value_t = false)]
    sigma_weighted_mse: bool,

    /// STRATEGY: EMA decay of weights (0 = off; typical 0.999). Bake
    /// snapshots the EMA copies — seed-variance reduction.
    #[arg(long, default_value_t = 0.0)]
    ema_decay: f64,
    /// STRATEGY: probability a pair is re-drawn as a hard pair.
    #[arg(long, default_value_t = 0.0)]
    hard_pair_frac: f64,
    /// |Δ target| ceiling defining a hard pair (native target units).
    #[arg(long, default_value_t = 0.05)]
    hard_pair_max_delta: f64,
    /// STRATEGY: target-quantile bands for stratified row sampling (0 = off).
    #[arg(long, default_value_t = 0)]
    stratified_bands: usize,
    /// STRATEGY: GroupDRO temperature over per-group mean loss (0 = off).
    #[arg(long, default_value_t = 0.0)]
    dro_eta: f64,
    /// STRATEGY: ListMLE listwise loss weight (0 = off).
    #[arg(long, default_value_t = 0.0)]
    listwise_weight: f64,
    /// Rows per listwise list.
    #[arg(long, default_value_t = 8)]
    listwise_size: usize,
    /// Fraction of steps run as listwise steps when active.
    #[arg(long, default_value_t = 0.15)]
    listwise_frac: f64,
    /// STRATEGY: ordered-probit triplet NLL weight on raw human triplets (0 = off).
    #[arg(long, default_value_t = 0.0)]
    triplet_weight: f64,
    /// Fraction of steps run as triplet steps when active.
    #[arg(long, default_value_t = 0.2)]
    triplet_frac: f64,
    /// Triplet indecision threshold τ (model-score units).
    #[arg(long, default_value_t = 0.6)]
    triplet_tau: f64,
    /// Triplet observer noise σ (model-score units).
    #[arg(long, default_value_t = 1.0)]
    triplet_sigma: f64,
    /// Triplet stimuli parquet/CSV (ref_basename + q_jnd/level + f0..fN rows).
    #[arg(long)]
    triplet_stimuli: Option<String>,
    /// Triplet responses TSV: left_idx, right_idx, response(0=left-more-distorted,1=right,2=notsure).
    #[arg(long)]
    triplet_responses: Option<String>,

    /// `PreviewV0_5Tuner` RankNet pair-loss weight (2026-05-18).
    /// Default `1.0` matches legacy behavior. Set to `0.0` to disable
    /// the RankNet pair loss entirely — use with `--mse-weight > 0`
    /// for pure-MSE training.
    #[arg(long, default_value_t = 1.0)]
    ranknet_weight: f64,

    /// `PreviewV0_5Tuner` monotonicity-reg weight (2026-05-18). When
    /// `> 0`, penalizes pairs whose predicted ordering disagrees
    /// with the target ordering via a quadratic hinge
    /// `w · max(0, y_low - y_high + margin)²`. Only wired on
    /// `--per-sample-alpha-head`.
    #[arg(long, default_value_t = 0.0)]
    monotonicity_reg: f64,

    /// **Correct-by-construction monotone mode.** Projects encoder
    /// weights ≥0 and head weights ≤0 + forces α≡1 after every Adam step
    /// so the bake is bounded `[0,100]` + monotone↓ in distortion BY
    /// CONSTRUCTION (codec goals G1+G3). Only on `--per-sample-alpha-head`
    /// with `--tanh-output-head-scale > 0`. Pair with
    /// `--skip-connection false`. Default off.
    #[arg(long, default_value_t = false)]
    monotone_cbc: bool,

    /// Per-feature sign mask TSV for `--monotone-cbc`: columns
    /// `feat_idx, sign_mask` where `sign_mask == "pin_geq0"` pins that
    /// input feature's W1 column ≥0 and anything else leaves it
    /// free/dropped. Produced by `tests/feature_distortion_direction.rs`
    /// (`benchmarks/feature_sign_mask_2026-05-26.tsv`). Without this,
    /// `--monotone-cbc` pins ALL features ≥0 (collapses the dial by
    /// mis-constraining the ~72 sign-flip features). Default: none.
    #[arg(long)]
    monotone_feature_mask: Option<PathBuf>,

    /// With `--monotone-feature-mask`: DROP the non-pinned (sign-flip)
    /// features (W1 columns → 0) so the bake is STRICTLY monotone in the
    /// sign-safe subset only. Without it (partial), non-pinned features
    /// stay free — full signal but monotone only in the safe subset.
    #[arg(long, default_value_t = false)]
    monotone_strict: bool,

    /// Soft-monotone-keep-72 mode (#39 followup #2, 2026-05-28):
    /// HARD-project the pinned-feature W1 columns to ≥0 after every Adam
    /// step (matches the final bake projection exactly), while leaving
    /// the unpinned ("sign-flip") features FREE throughout training and
    /// bake. This replicates the MVP-Python BVLS bounds behavior in the
    /// Rust trainer.
    ///
    /// Orthogonal to `--monotone-strict` — when this flag is set,
    /// `--monotone-strict`'s "drop the 72 unpinned" behavior is
    /// SUPPRESSED (the 72 stay free regardless).
    ///
    /// Recipe field: `monotone_pin_during_training` (TOML). Default off.
    #[arg(long, default_value_t = false)]
    monotone_pin_during_training: bool,

    /// Quantization-aware FINE-TUNE: train the LAST N epochs with a
    /// straight-through estimator (forward uses f16-rounded + zerobiased
    /// weights, Adam updates the f32 master) so the shipped packed bake ==
    /// the validated net. 0 = off (default; pure f32). Pairs with
    /// `--out-dtype f16`. Recipe field: `qat_fine_tune_epochs`.
    #[arg(long, default_value_t = 0)]
    qat_fine_tune_epochs: usize,

    /// QAT zerobias threshold (relative to per-layer max, matching the
    /// bake-time zerobias). Default 0.005.
    #[arg(long, default_value_t = 0.005)]
    qat_tau: f64,

    /// Per-epoch group-eval row cap (0 = full/historical). >0 forwards a
    /// deterministic stride sample per oversized group for the per-epoch
    /// diagnostics/selection — the iteration-speed lever for multi-million-
    /// row groups. No RNG; training bytes unchanged.
    #[arg(long, default_value_t = 0)]
    group_eval_cap: usize,

    /// `PreviewV0_5Tuner` monotonicity-reg margin (2026-05-18). The
    /// penalty activates only when the predicted gap is below
    /// `+margin` relative to perfect ordering. Default `0.0` =
    /// activate on any strict inversion.
    #[arg(long, default_value_t = 0.0)]
    monotonicity_margin: f64,

    /// `PreviewV0_5TunerV2` cross-codec JND anchor parquet (2026-05-19).
    /// Path to a parquet with at minimum (`f0..f<N-1>`, `anchor_weight`).
    /// The target column is the constant `--anchor-target-score`
    /// regardless of any `human_score` column in the file (we override
    /// after loading). One row = one anchor pair (source, codec-at-PJND).
    /// Weighted higher anchor_weight values are sampled preferentially.
    /// Default empty = no anchor data; ignored if --anchor-loss-weight 0.
    #[arg(long)]
    anchor_parquet: Option<PathBuf>,

    /// `PreviewV0_5TunerV2` cross-codec JND anchor loss weight
    /// (2026-05-19). Default `0.0` = no anchor stepping. When `> 0`,
    /// every pair-step with probability `--anchor-step-p` samples one
    /// anchor row and applies MSE against `--anchor-target-score`. Only
    /// wired on the per-sample-α head and minibatch-size 1.
    #[arg(long, default_value_t = 0.0)]
    anchor_loss_weight: f64,

    /// `PreviewV0_5TunerV2` target score the anchor rows regress to
    /// (2026-05-19). Default `63.0` matches CID22 paper Table 4 PJND
    /// calibration. Per-row anchor weights multiply this target's MSE
    /// contribution.
    #[arg(long, default_value_t = 63.0)]
    anchor_target_score: f64,

    /// `PreviewV0_5TunerV2` probability that each pair-step also
    /// performs an anchor step (2026-05-19). Default `0.10` = 10 % of
    /// pair-steps trigger an anchor row. Higher values give the anchor
    /// loss more bandwidth at the cost of rank-loss bandwidth.
    #[arg(long, default_value_t = 0.10)]
    anchor_step_p: f64,

    /// EXP-CROSS-CODEC-METRIC equivalence pair parquet (2026-05-19).
    /// Path to a parquet built by
    /// `scripts/v_next/build_cross_codec_equivalence.py` with columns:
    /// `ref_basename, codec_a, q_a, codec_b, q_b, butter_level,
    /// butter_a, butter_b, row_weight, fa_0..fa_<N-1>, fb_0..fb_<N-1>`.
    /// Each row is a cross-codec equivalence pair: features_a and
    /// features_b come from distortions at q values picked to match
    /// the same butter level under different codecs. During training
    /// these pairs apply a shape-free `(y_a − y_b)²` loss so the
    /// metric learns to score them identically.
    /// Default empty = no equiv data; ignored if --cross-codec-eq-weight 0.
    #[arg(long)]
    cross_codec_eq_parquet: Option<PathBuf>,

    /// EXP-CROSS-CODEC-METRIC equivalence pair loss weight (2026-05-19).
    /// Default `0.0` = no equivalence stepping. When `> 0`, every
    /// pair-step with probability `--cross-codec-eq-step-p` samples one
    /// (A, B) equivalence pair and applies `w · (y_a - y_b)²`. Only
    /// wired on the per-sample-α head and minibatch-size 1.
    #[arg(long, default_value_t = 0.0)]
    cross_codec_eq_weight: f64,

    /// EXP-CROSS-CODEC-METRIC probability that each pair-step also
    /// performs an equivalence step (2026-05-19). Default `0.10`.
    #[arg(long, default_value_t = 0.10)]
    cross_codec_eq_step_p: f64,

    /// EXP-CROSS-CODEC-V3 rank-preserve regularizer weight (2026-05-19).
    /// When `> 0` AND the equiv parquet supplies `butter_a` / `butter_b`,
    /// each equivalence step adds a RankNet-style sigmoid loss weighted
    /// by `|butter_a − butter_b|` that pushes the network to keep
    /// `sign(score_a − score_b)` aligned with `sign(butter_b − butter_a)`
    /// (since LOWER butter = HIGHER quality and the bake's `mix_cv40_iw60`
    /// target is HIGHER = HIGHER quality). Prevents collapse of equiv
    /// outputs to a point mass when `cross_codec_eq_weight` is high.
    /// Default `0.0` = no rank-preserve.
    #[arg(long, default_value_t = 0.0)]
    cross_codec_rank_preserve_weight: f64,

    /// EXP-CROSS-CODEC-V3 dynamic-range floor regularizer weight
    /// (2026-05-19). When `> 0`, every pair-step with probability
    /// `--dynamic-range-step-p` triggers a q-sweep probe: sample
    /// `--dynamic-range-probe-n` random rows from the equiv pool's A-side,
    /// forward each through the per-sample-α head, compute σ across the
    /// outputs, and penalize the network if σ < threshold:
    /// `L = w · max(0, σ_threshold − σ_obs)²`. Directly addresses the
    /// "collapse to constant" failure mode by structurally requiring
    /// per-batch output spread. Default `0.0` = off. Requires equiv
    /// pool data (A-side acts as the q-sweep substrate).
    #[arg(long, default_value_t = 0.0)]
    dynamic_range_floor_weight: f64,

    /// EXP-CROSS-CODEC-V3 dynamic-range floor σ threshold (score units).
    /// Probes whose σ_obs ≥ threshold contribute 0 penalty; below
    /// threshold the penalty is quadratic. Default `15.0` targets a
    /// ~45-50 score-unit p25-p75 spread.
    #[arg(long, default_value_t = 15.0)]
    dynamic_range_sigma_threshold: f64,

    /// EXP-CROSS-CODEC-V3 dynamic-range floor step probability per
    /// pair-step. Default `0.05` = 5% of pair-steps trigger a probe,
    /// ~2500 probes per epoch at `--pairs-per-epoch 50000`.
    #[arg(long, default_value_t = 0.05)]
    dynamic_range_step_p: f64,

    /// EXP-CROSS-CODEC-V3 dynamic-range floor probe sample size.
    /// Default `40` = 8 refs × 5 q values conceptually. Affects the
    /// σ estimator's variance.
    #[arg(long, default_value_t = 40)]
    dynamic_range_probe_n: usize,

    /// EXP-CROSS-CODEC-V4 tanh-pinned [0, 100] output head scale
    /// (2026-05-19). When `> 0`, wraps the per-sample-α head's raw
    /// output `y_pre = α·y_rank + (1−α)·y_pool` in a sigmoid pin:
    ///
    /// `y_score = 100 · σ(y_pre / scale)`
    ///
    /// Default `0.0` = legacy linear output (post-hoc affine needed
    /// to reach [0, 100]). Recommended `10.0` so active linear region
    /// `y_pre ∈ [−30, 30]` maps to `[5, 95]` score units. The bake
    /// receives a `zentrain.tanh_output_head` metadata entry; the
    /// zensim runtime applies the matching sigmoid pin at inference.
    /// Only wired on `--per-sample-alpha-head`. Requires `--minibatch-size 1`.
    #[arg(long, default_value_t = 0.0)]
    tanh_output_head_scale: f64,

    /// GPU trainer dispatch (task #166, 2026-05-19). When set to a value
    /// other than `""` / `cpu`, routes the per-sample-α head training
    /// loop through `zensim-train-gpu` instead of the CPU SIMD path.
    /// Acceptable: `cuda`, `wgpu`, `gpucpu` (cubecl-cpu — parity only).
    /// Requires the matching cargo feature (`--features gpu-cuda` etc).
    ///
    /// **Phase 1 restrictions** (panic-on-conflict at startup):
    /// - `--per-sample-alpha-head` must be set
    /// - `--tv-pairs-file` must NOT be set
    /// - `--norm-in-norm-weight` must be 0 (Phase 3+ port)
    /// - PWRC band weights / boost flags ignored on the GPU path
    /// - Empty = CPU (default).
    ///
    /// Phase 2 (task #169, 2026-05-19): anchor + cross-codec-eq +
    /// rank-preserve + σ-floor aux losses are GPU-ported. NiN is the
    /// remaining gap.
    #[arg(long, default_value = "")]
    gpu_runtime: String,

    /// K samples per aux-loss fire on GPU. Default 32. Higher = larger
    /// kernel launches (better GPU saturation, more sampling variance
    /// reduction); lower = closer to the CPU K=1 path. Ignored when
    /// no aux losses are active.
    #[arg(long, default_value_t = 32)]
    gpu_minibatch_k_aux: usize,

    /// EXP-V11-D-PJND-DOMINANT (2026-05-20, task #198) KonJND-PJND
    /// passthrough anchor parquet. Same schema as `--anchor-parquet`
    /// (uses `f0..f<N-1>` features), but loaded with a constant
    /// per-row weight of `1.0`; per-row target defaults to
    /// `--pjnd-passthrough-target-score`. Intended for
    /// `canonical-2026-05-21/train/konjnd-dense.parquet` (20,160
    /// rows × 372 features). Empty = no second anchor pool.
    #[arg(long)]
    pjnd_passthrough_parquet: Option<PathBuf>,

    /// EXP-V11-D-PJND-DOMINANT passthrough anchor loss weight. Default
    /// `0.0` = off. When `> 0`, the PJND anchor fires alongside the
    /// primary `--anchor-parquet` anchor at probability
    /// `--pjnd-passthrough-step-p`. Only wired on the per-sample-α head.
    #[arg(long, default_value_t = 0.0)]
    pjnd_passthrough_weight: f64,

    /// EXP-V11-D-PJND-DOMINANT step probability. Default `0.30` (per
    /// V11-D spec: anchor fires often enough to dominate cross-codec-eq).
    /// Only effective when `--pjnd-passthrough-weight > 0`.
    #[arg(long, default_value_t = 0.30)]
    pjnd_passthrough_step_p: f64,

    /// EXP-V11-D-PJND-DOMINANT global target score every PJND-passthrough
    /// row regresses to. Default `80.0` (V10 maps PJND ssim2≈63 to
    /// score=80). Only effective when `--pjnd-passthrough-weight > 0`.
    #[arg(long, default_value_t = 80.0)]
    pjnd_passthrough_target_score: f64,

    /// KONJND-AGGREGATION-HEAD (2026-05-24, task #4) konjnd-dense
    /// parquet path. Requires the parquet to carry `ref_basename`,
    /// `pjnd_target`, and `f0..fN` columns — the canonical
    /// `canonical-2026-05-21/train/konjnd-dense.parquet` does.
    /// Loaded with per-ref grouping; the aggregation step samples K
    /// refs per fire, S rows per ref, forwards K·S times, computes K
    /// aggregate means, and applies MSE against the per-ref
    /// pjnd_target. Empty = no aggregation pool. See
    /// `docs/KONJND_AGGREGATION_HEAD_DESIGN_2026-05-24.md`.
    #[arg(long)]
    konjnd_aggregation_parquet: Option<PathBuf>,

    /// KONJND-AGGREGATION-HEAD loss weight. Default `0.0` = off.
    /// When `> 0`, the aggregation step fires per pair-step with
    /// probability `--konjnd-aggregation-step-p`. Only wired on the
    /// per-sample-α head. Structurally different from
    /// `--pjnd-passthrough-weight`: regresses the pooled per-ref
    /// mean against pjnd_target, not each row independently.
    #[arg(long, default_value_t = 0.0)]
    konjnd_aggregation_weight: f64,

    /// KONJND-AGGREGATION-HEAD step probability. Default `0.30`.
    /// Only effective when `--konjnd-aggregation-weight > 0`.
    #[arg(long, default_value_t = 0.30)]
    konjnd_aggregation_step_p: f64,

    /// KONJND-AGGREGATION-HEAD rows-per-ref (S). Number of distortion
    /// levels sampled per ref per aggregation step. Default `5`.
    /// Total forwards per aggregation step = K·S.
    #[arg(long, default_value_t = 5)]
    konjnd_aggregation_samples_per_ref: usize,

    /// KONJND-AGGREGATION-HEAD refs-per-step (K). Number of refs
    /// picked per aggregation step. Default `8`.
    #[arg(long, default_value_t = 8)]
    konjnd_aggregation_refs_per_step: usize,

    /// Reproduce-this input: load a shipped bake's TOML manifest
    /// (`zensim/weights/manifests/*.toml`) and reconstruct the training
    /// run it records. The manifest's structured `[training]` fields,
    /// `groups` array, `auto_transforms` / `anchor_parquet` paths, and
    /// `--out` (from `[bake].file`) become the run config; every
    /// referenced `[inputs.<name>]` file is sha256-verified before
    /// training begins (FAILS LOUD on drift — that is the whole point of
    /// reproduce-exactly).
    ///
    /// **Precedence**: the manifest provides DEFAULTS; any flag you also
    /// pass explicitly on the command line OVERRIDES the manifest value.
    /// So `--manifest foo.toml --seed 99` reproduces foo's recipe but
    /// with seed 99. `--group` is special: if you pass any explicit
    /// `--group`, your groups fully replace the manifest's group set
    /// (they don't merge).
    ///
    /// Post-training `steps` recorded in the manifest (spline injection,
    /// affine calibration, etc.) are PRINTED but NOT executed — they
    /// shell to external scripts the trainer doesn't run; apply them by
    /// hand after the bake is produced.
    #[arg(long, value_name = "PATH")]
    manifest: Option<PathBuf>,

    /// Escape hatch for `--manifest`: downgrade input-file sha256
    /// mismatches from a hard error to a warning. OFF by default. With
    /// this set, a reproduce run proceeds even if a referenced input
    /// drifted — but the produced bake will NOT match the shipped one,
    /// so only use it when you knowingly intend to retrain on changed
    /// data. Missing input files still error regardless.
    #[arg(long, default_value_t = false)]
    manifest_allow_sha_drift: bool,
}

/// CLI parser for `--pwrc-band-weights W0,W1,...` — accepts any
/// non-empty comma-separated list of finite, non-negative f64s.
fn parse_pwrc_band_weights(s: &str) -> Result<Vec<f64>, String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.is_empty() {
        return Err("expected at least one band weight".to_string());
    }
    let mut out = Vec::with_capacity(parts.len());
    for (i, p) in parts.iter().enumerate() {
        let v: f64 = p
            .trim()
            .parse()
            .map_err(|e| format!("band {i}: parse '{p}': {e}"))?;
        if !v.is_finite() || v < 0.0 {
            return Err(format!("band {i}: weight {v} must be finite and >= 0"));
        }
        out.push(v);
    }
    Ok(out)
}

struct LoadedGroup {
    name: String,
    train_w: f64,
    val_w: f64,
    human_scores: Vec<f64>,
    /// ONE flat row-major buffer per group (`n_rows × n_features`,
    /// `n_rows == human_scores.len()`), NOT per-row `Vec`s. Deliberate, and
    /// load-bearing for memory: the trainer takes this buffer and
    /// standardizes it in place (`FeatureRows::Releasable`), so a lane
    /// never holds two copies of the feature matrix — and per-row `Vec`s
    /// measurably defeat that, because 779k ~7.5 KB chunks freed out of the
    /// loaders' interleaved arenas never return to the OS (full-recipe RSS
    /// moved only ~0.4 GB when a first version freed rows individually).
    /// One large buffer per group frees for real. See
    /// `benchmarks/trainer_mem_release_2026-08-04.md`.
    feature_rows: Vec<f64>,
    metric_sigmas: Option<Vec<f64>>,
    n_features: usize,
    /// Dense per-row ref identity from the parquet loader; `None` for
    /// CSV groups (the CSV loaders don't retain `ref_basename`) and for
    /// parquets carrying neither `ref_basename` nor `image_path`.
    /// Required by `within_ref` — see `GroupSpec::within_ref`.
    ref_ids: Option<Vec<u32>>,
    /// Draw RankNet pairs WITHIN a reference image rather than uniformly
    /// across the group. See `TrainingGroup::ref_ids`.
    within_ref: bool,
    /// Which loss terms this group contributes. See `GroupLossMode`.
    loss_mode: GroupLossMode,
    /// Absolute path of the source parquet/CSV — REPRODUCTION identity,
    /// embedded into the bake's `zentrain.repro` (paths in argv can be
    /// worker-relative and die with the worker dir; this one is canonical).
    source_path: String,
    /// sha256 of the source file — the content identity that outlives moves.
    source_sha256: String,
}

/// Flatten a per-row table into one row-major buffer, dropping each row
/// `Vec` as it is consumed. Values and their order are identical to the
/// per-row form (`flat[i * width + d] == rows[i][d]` for `d < width`).
///
/// `width` must be ≤ every row's length; rows longer than `width` are
/// truncated (this is where the old per-row `row.truncate(cap)` semantics
/// live now). Dropping rows as we go matters: the freed same-size chunks
/// are immediately reused by the NEXT group's loader rows, so the per-row
/// stage's footprint stays bounded by one group instead of the whole
/// dataset.
fn flatten_rows(rows: Vec<Vec<f64>>, width: usize) -> Vec<f64> {
    let mut flat: Vec<f64> = Vec::with_capacity(rows.len() * width);
    for row in rows {
        assert!(
            row.len() >= width,
            "flatten_rows: row has {} features, need ≥ {width}",
            row.len()
        );
        flat.extend_from_slice(&row[..width]);
    }
    flat
}

impl From<zensim_validate::parquet_loader::OwnedLoadedGroupFlat> for LoadedGroup {
    /// The zero-copy conversion: the loader already emitted the flat
    /// row-major buffer this struct stores, so the ~1.9 GB rows+flat
    /// flatten transient of the `OwnedLoadedGroup` path never exists.
    /// See `benchmarks/trainer_mem_release_2026-08-04.md` ("next lever").
    fn from(o: zensim_validate::parquet_loader::OwnedLoadedGroupFlat) -> Self {
        Self {
            name: o.name,
            train_w: o.train_w,
            val_w: o.val_w,
            human_scores: o.human_scores,
            feature_rows: o.features_flat,
            metric_sigmas: o.metric_sigmas,
            n_features: o.n_features,
            ref_ids: o.ref_ids,
            within_ref: false,
            loss_mode: GroupLossMode::default(),
            source_path: String::new(),
            source_sha256: String::new(),
        }
    }
}

/// Parse `--keep-features SPEC` into a sorted, de-duplicated index list.
///
/// SPEC is either an inline comma-separated list (`0,5,17`) or a path to a
/// file of whitespace/comma-separated indices; `#` starts a line comment.
/// Every index must be `< n_features`. An empty set is an error (it would
/// train a constant model, silently).
fn parse_keep_features(spec: &str, n_features: usize) -> Result<Vec<usize>, String> {
    let text = if std::path::Path::new(spec).is_file() {
        std::fs::read_to_string(spec).map_err(|e| format!("cannot read {spec:?}: {e}"))?
    } else {
        spec.to_string()
    };
    let mut idx: Vec<usize> = Vec::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("");
        for tok in line.split([',', ' ', '\t']).filter(|t| !t.is_empty()) {
            let v: usize = tok
                .trim()
                .parse()
                .map_err(|_| format!("bad index {tok:?} (want a non-negative integer)"))?;
            if v >= n_features {
                return Err(format!(
                    "index {v} >= --max-features {n_features}; the subset must live inside the \
                     declared feature width"
                ));
            }
            idx.push(v);
        }
    }
    idx.sort_unstable();
    idx.dedup();
    if idx.is_empty() {
        return Err("empty index set — refusing to train a zero-input model".into());
    }
    Ok(idx)
}

/// Count layer-1 input rows that are **exactly** zero in a baked model, i.e.
/// inputs the fit dropped (pinned by `--keep-features`, or learned away by
/// `--group-l1`). Exact zeros are what `bake_dial_refit pack` prunes, so this
/// is the honest "live width" of the bake. Returns `(live, dead)`.
fn count_live_l0_rows(bake: &[u8]) -> Option<(usize, usize)> {
    let model = zenpredict::Model::from_bytes(bake).ok()?;
    let l = model.layer(0);
    let (in_dim, out_dim) = (l.in_dim, l.out_dim);
    let is_zero = |i: usize| -> bool {
        (0..out_dim).all(|j| match &l.weights {
            zenpredict::WeightStorage::F32(w) => w[i * out_dim + j] == 0.0,
            zenpredict::WeightStorage::F16(w) => {
                zenpredict::f16_bits_to_f32(w[i * out_dim + j]) == 0.0
            }
            zenpredict::WeightStorage::I8 { weights, .. } => weights[i * out_dim + j] == 0,
        })
    };
    let dead = (0..in_dim).filter(|&i| is_zero(i)).count();
    Some((in_dim - dead, dead))
}

/// Dispatch to the right loader based on file extension. `.parquet` ->
/// load_parquet (T8.10), anything else (typically `.csv`) -> load_csv
/// (the parallel mmap+memchr+fast_float loader from T8.9).
fn load_group_dispatch(
    path: &PathBuf,
    name: &str,
    target_column: &str,
    target_scale: f64,
) -> Result<LoadedGroup, String> {
    let is_parquet = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.eq_ignore_ascii_case("parquet"))
        .unwrap_or(false);
    if is_parquet {
        // Flat emission: the loader fills ONE pre-reserved row-major buffer,
        // so the per-row stage (and the rows+flat flatten transient) never
        // exists. `From<OwnedLoadedGroupFlat>` is a field move.
        zensim_validate::parquet_loader::load_parquet_flat(path, name, target_column, target_scale)
            .map(LoadedGroup::from)
    } else {
        load_csv(path, name, target_column, target_scale)
    }
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
            eprintln!("--auto-transforms: cannot open {}: {e}", tsv_path.display());
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
        header
            .iter()
            .position(|c| c.trim() == name)
            .unwrap_or_else(|| {
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

/// Parse `NAME:PATH:TRAIN_W:VAL_W` with an optional 5th field of
/// comma-separated flags.
///
/// Accepted flags:
/// - `withinref` — draw this group's RankNet pairs within a single
///   reference image instead of uniformly across the group (see
///   `TrainingGroup::ref_ids`).
/// - `rank` (default) / `mse` / `both` — which loss terms this group
///   contributes (see `GroupLossMode`). `mse` and `both` require
///   `--mse-weight`.
///
/// Adding the field is backward compatible: a 5th field previously made
/// `val_w` parsing fail, so no existing spec can carry one.
fn parse_group_spec(
    spec: &str,
) -> Result<(String, PathBuf, f64, f64, bool, GroupLossMode), String> {
    let parts: Vec<&str> = spec.splitn(5, ':').collect();
    if parts.len() < 4 {
        return Err(format!(
            "expected NAME:PATH:TRAIN_W:VAL_W[:flag,flag], got {spec:?}"
        ));
    }
    let train_w: f64 = parts[2].parse().map_err(|e| format!("bad train_w: {e}"))?;
    let val_w: f64 = parts[3].parse().map_err(|e| format!("bad val_w: {e}"))?;
    let mut within_ref = false;
    let mut loss_mode: Option<GroupLossMode> = None;
    for flag in parts.get(4).into_iter().flat_map(|f| f.split(',')) {
        match flag {
            "withinref" => within_ref = true,
            "rank" | "mse" | "both" => {
                let m = match flag {
                    "rank" => GroupLossMode::Rank,
                    "mse" => GroupLossMode::Mse,
                    _ => GroupLossMode::Both,
                };
                if loss_mode.replace(m).is_some() {
                    return Err(format!(
                        "more than one loss mode in {spec:?} (pick one of rank/mse/both)"
                    ));
                }
            }
            other => {
                return Err(format!(
                    "unknown group flag {other:?} in {spec:?} \
                     (accepted: withinref, rank, mse, both)"
                ));
            }
        }
    }
    Ok((
        parts[0].to_string(),
        PathBuf::from(parts[1]),
        train_w,
        val_w,
        within_ref,
        loss_mode.unwrap_or_default(),
    ))
}

/// Sequential CSV loader — kept ONLY as the bit-identical reference for
/// the equivalence test (`csv_load_equivalence_tests`). Single-threaded,
/// uses `f64::from_str` via stdlib `.parse::<f64>()`.
///
/// The parallel loader is the production path (see `load_csv`). This
/// function is `#[cfg(test)]`-gated so the release binary never carries
/// the dead code.
#[cfg(test)]
fn load_csv_sequential(
    path: &PathBuf,
    name: &str,
    target_column: &str,
    target_scale: f64,
) -> Result<LoadedGroup, String> {
    let file = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let mut rdr = BufReader::new(file);
    let mut header = String::new();
    rdr.read_line(&mut header)
        .map_err(|e| format!("header read {path:?}: {e}"))?;
    let cols: Vec<&str> = header.trim_end().split(',').collect();
    let score_idx = cols
        .iter()
        .position(|&c| c == target_column)
        .ok_or_else(|| format!("{path:?}: missing target column {target_column:?}"))?;
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
        // Multiply by `target_scale` (default 100.0) to match `score_zensim` units.
        // The legacy default brought `human_score ∈ [0, 1]` to 0..100; new
        // target columns may already be in 0..100 (`--target-scale 1.0`) or
        // need a different multiplier (CVVDP JOD ∈ [0, 10] → 10.0).
        let score: f64 = fields[score_idx].parse::<f64>().map_err(|e| {
            format!(
                "{path:?} line {}: bad target column {target_column:?}: {e}",
                lineno + 2
            )
        })? * target_scale;
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
        feature_rows: flatten_rows(feature_rows, n_features),
        metric_sigmas: None,
        n_features,
        ref_ids: None,
        within_ref: false,
        loss_mode: GroupLossMode::default(),
        source_path: String::new(),
        source_sha256: String::new(),
    })
}

/// Parallel CSV loader.
///
/// Production path. mmap's the file, finds line boundaries via memchr,
/// partitions rows into rayon chunks, and parses each field with
/// `fast_float2::parse::<f64, _>` (5-10× faster than stdlib
/// `f64::from_str` per the crate's README). Order is preserved.
///
/// Empty lines are skipped (matches sequential behavior). Field-count
/// and parse errors surface with the same shape as the sequential
/// loader so downstream error handling is unchanged.
pub(crate) fn load_csv(
    path: &PathBuf,
    name: &str,
    target_column: &str,
    target_scale: f64,
) -> Result<LoadedGroup, String> {
    let file = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    // SAFETY: We're reading a file that's not concurrently mutated by
    // any in-process writer; the trainer treats inputs as read-only.
    // mmap is the standard fast-CSV-load pattern.
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("mmap {path:?}: {e}"))?;
    let bytes: &[u8] = &mmap;

    // 1) Find header end + parse header.
    let header_end = memchr::memchr(b'\n', bytes)
        .ok_or_else(|| format!("{path:?}: empty file / no header newline"))?;
    let header_str = std::str::from_utf8(&bytes[..header_end])
        .map_err(|e| format!("{path:?}: non-UTF8 header: {e}"))?;
    let cols: Vec<&str> = header_str.trim_end_matches('\r').split(',').collect();
    let score_idx = cols
        .iter()
        .position(|&c| c == target_column)
        .ok_or_else(|| format!("{path:?}: missing target column {target_column:?}"))?;
    let f0 = cols
        .iter()
        .position(|&c| c == "f0")
        .ok_or_else(|| format!("{path:?}: missing f0 column"))?;
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
    let min_fields = f0 + n_features;

    // 2) Build a Vec of (start, end) byte offsets for every body line,
    // including empty ones — so index-in-vec is a faithful proxy for
    // absolute file line number, matching the sequential loader's
    // `lineno + 2` error semantics. memchr_iter scans at ~10 GB/s —
    // this is effectively free.
    let body_start = header_end + 1;
    let mut line_ranges: Vec<(usize, usize)> = Vec::with_capacity(
        // rough estimate to skip a few realloc rounds
        (bytes.len() - body_start) / 256 + 1,
    );
    let mut cursor = body_start;
    for nl in memchr::memchr_iter(b'\n', &bytes[body_start..]) {
        let end = body_start + nl;
        line_ranges.push((cursor, end));
        cursor = end + 1;
    }
    // Tail without trailing newline.
    if cursor < bytes.len() {
        line_ranges.push((cursor, bytes.len()));
    }

    // 3) Parse rows in parallel rayon chunks. Each chunk produces its
    // own (scores, rows, source-line-numbers-for-errors) and we merge
    // chunks in source order via `flat_map_iter`.
    //
    // Chunk size of 1024 gives ~200 chunks for the safesyn corpus (~196k
    // rows) and amortizes scheduling cost over enough work.
    const CHUNK_SIZE: usize = 1024;

    #[derive(Default)]
    struct ChunkOut {
        scores: Vec<f64>,
        rows: Vec<Vec<f64>>,
    }

    let chunks: Vec<Result<ChunkOut, String>> = line_ranges
        .par_chunks(CHUNK_SIZE)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let mut out = ChunkOut {
                scores: Vec::with_capacity(chunk.len()),
                rows: Vec::with_capacity(chunk.len()),
            };
            for (i_in_chunk, &(start, end)) in chunk.iter().enumerate() {
                let line = &bytes[start..end];
                // Strip trailing '\r' (CRLF files).
                let line = if line.last() == Some(&b'\r') {
                    &line[..line.len() - 1]
                } else {
                    line
                };
                if line.is_empty() {
                    continue;
                }
                // Real (1-based, header-inclusive) line number for
                // error messages — matches sequential `lineno + 2`
                // semantics (lineno is 0-based index into rdr.lines()
                // which starts AFTER header, plus 1-based + header).
                let lineno_for_err = chunk_idx * CHUNK_SIZE + i_in_chunk + 2;

                // Locate the comma positions we care about. We need
                // fields[score_idx], fields[f0..f0+n_features]. Use
                // memchr_iter to walk commas, picking out the slices.
                //
                // Strategy: index into the comma list by position.
                // Build a small Vec<usize> of comma positions then
                // slice it. This keeps the hot loop branch-free and
                // matches the sequential `split(',').collect()`
                // behavior bit-for-bit (we slice the same byte ranges).
                let mut commas: Vec<usize> = Vec::with_capacity(min_fields + 4);
                for c in memchr::memchr_iter(b',', line) {
                    commas.push(c);
                }
                // Sequential checked `fields.len() < min_fields`
                // (== commas+1 < min_fields).
                let n_fields = commas.len() + 1;
                if n_fields < min_fields {
                    return Err(format!(
                        "{path:?} line {lineno_for_err}: expected ≥{min_fields} fields, got {n_fields}"
                    ));
                }
                let field = |idx: usize| -> &[u8] {
                    let s = if idx == 0 { 0 } else { commas[idx - 1] + 1 };
                    let e = if idx < commas.len() { commas[idx] } else { line.len() };
                    &line[s..e]
                };
                let score_bytes = field(score_idx);
                let score: f64 = fast_float2::parse::<f64, _>(score_bytes).map_err(|e| {
                    format!(
                        "{path:?} line {lineno_for_err}: bad target column {target_column:?}: {e}"
                    )
                })? * target_scale;
                let mut row: Vec<f64> = Vec::with_capacity(n_features);
                for fi in 0..n_features {
                    let fb = field(f0 + fi);
                    let v: f64 = fast_float2::parse::<f64, _>(fb).map_err(|e| {
                        format!("{path:?} line {lineno_for_err}: bad f{fi}: {e}")
                    })?;
                    row.push(v);
                }
                out.scores.push(score);
                out.rows.push(row);
            }
            Ok(out)
        })
        .collect();

    // 4) Merge chunks in source order, propagating the first error.
    // Pre-compute total capacity to avoid Vec growth.
    let mut total = 0usize;
    for c in &chunks {
        match c {
            Ok(o) => total += o.scores.len(),
            Err(e) => return Err(e.clone()),
        }
    }
    let mut human_scores: Vec<f64> = Vec::with_capacity(total);
    let mut feature_rows: Vec<Vec<f64>> = Vec::with_capacity(total);
    for c in chunks {
        let mut o = c.unwrap();
        human_scores.append(&mut o.scores);
        feature_rows.append(&mut o.rows);
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
        feature_rows: flatten_rows(feature_rows, n_features),
        metric_sigmas: None,
        n_features,
        ref_ids: None,
        within_ref: false,
        loss_mode: GroupLossMode::default(),
        source_path: String::new(),
        source_sha256: String::new(),
    })
}

/// Load a cross-codec equivalence parquet for EXP-CROSS-CODEC-METRIC.
///
/// Schema: ref_basename, codec_a, q_a, codec_b, q_b, butter_level,
///         butter_a, butter_b, row_weight, fa_0..fa_<M-1>, fb_0..fb_<M-1>
///
/// Truncates each fa_*/fb_* feature row to `max_features` (matches the
/// trainer's `--max-features`). Returns `(features_a, features_b,
/// row_weights, butter_diff)` as owned vectors. `butter_diff[i] =
/// butter_a[i] − butter_b[i]` (LOWER butter = HIGHER quality so a
/// positive Δb means A is quality-worse than B). `butter_diff` is
/// always populated when butter_a/butter_b columns are present; empty
/// otherwise (callers detect via `butter_diff.is_empty()`).
/// `(features_a, features_b, row_weights, butter_diff)` columns.
type EquivColumns = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>);

fn load_equiv_parquet(path: &PathBuf, max_features: usize) -> Result<EquivColumns, String> {
    use arrow::array::{Array, Float32Array, Float64Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = std::fs::File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("{path:?}: parquet open: {e}"))?;
    let schema = builder.schema().clone();

    let arrow_fields = schema.fields();
    let n_arrow_cols = arrow_fields.len();

    // Locate row_weight.
    let rw_idx = arrow_fields
        .iter()
        .position(|f| f.name() == "row_weight")
        .ok_or_else(|| format!("{path:?}: missing row_weight column"))?;

    // EXP-CROSS-CODEC-V3: locate butter_a / butter_b for the
    // rank-preserve regularizer. Optional — schema versions without
    // them just return an empty butter_diff vec.
    let butter_a_idx = arrow_fields.iter().position(|f| f.name() == "butter_a");
    let butter_b_idx = arrow_fields.iter().position(|f| f.name() == "butter_b");

    // Locate fa_0 and count consecutive fa_i.
    let fa0_idx = arrow_fields
        .iter()
        .position(|f| f.name() == "fa_0")
        .ok_or_else(|| format!("{path:?}: missing fa_0 column"))?;
    let mut n_fa = 0usize;
    while fa0_idx + n_fa < n_arrow_cols {
        let expected = format!("fa_{}", n_fa);
        if arrow_fields[fa0_idx + n_fa].name() != &expected {
            break;
        }
        n_fa += 1;
    }
    if n_fa == 0 {
        return Err(format!("{path:?}: no fa_N columns"));
    }

    // Locate fb_0 and count consecutive fb_i.
    let fb0_idx = arrow_fields
        .iter()
        .position(|f| f.name() == "fb_0")
        .ok_or_else(|| format!("{path:?}: missing fb_0 column"))?;
    let mut n_fb = 0usize;
    while fb0_idx + n_fb < n_arrow_cols {
        let expected = format!("fb_{}", n_fb);
        if arrow_fields[fb0_idx + n_fb].name() != &expected {
            break;
        }
        n_fb += 1;
    }
    if n_fb != n_fa {
        return Err(format!("{path:?}: fa width {} != fb width {}", n_fa, n_fb));
    }

    let n_features = n_fa.min(max_features);
    if n_features == 0 {
        return Err(format!("{path:?}: zero features after truncation"));
    }

    let reader = builder
        .with_batch_size(8192)
        .build()
        .map_err(|e| format!("{path:?}: parquet build: {e}"))?;

    let mut features_a: Vec<Vec<f64>> = Vec::new();
    let mut features_b: Vec<Vec<f64>> = Vec::new();
    let mut row_weights: Vec<f64> = Vec::new();
    let mut butter_diff: Vec<f64> = Vec::new();
    let butter_avail = butter_a_idx.is_some() && butter_b_idx.is_some();

    // Inline helper: read a f32/f64 column into Vec<f64>.
    fn read_f64_col(col: &dyn arrow::array::Array, n_rows: usize) -> Result<Vec<f64>, String> {
        match col.data_type() {
            arrow::datatypes::DataType::Float64 => {
                let a = col.as_any().downcast_ref::<Float64Array>().unwrap();
                Ok((0..n_rows).map(|i| a.value(i)).collect())
            }
            arrow::datatypes::DataType::Float32 => {
                let a = col.as_any().downcast_ref::<Float32Array>().unwrap();
                Ok((0..n_rows).map(|i| a.value(i) as f64).collect())
            }
            other => Err(format!("col dtype {other:?} unsupported")),
        }
    }

    for batch_res in reader {
        let batch = batch_res.map_err(|e| format!("{path:?}: read batch: {e}"))?;
        let n_rows = batch.num_rows();
        if n_rows == 0 {
            continue;
        }

        // row_weight
        let rw_col = batch.column(rw_idx);
        let rw_vec: Vec<f64> = match rw_col.data_type() {
            arrow::datatypes::DataType::Float64 => {
                let a = rw_col.as_any().downcast_ref::<Float64Array>().unwrap();
                (0..n_rows).map(|i| a.value(i)).collect()
            }
            arrow::datatypes::DataType::Float32 => {
                let a = rw_col.as_any().downcast_ref::<Float32Array>().unwrap();
                (0..n_rows).map(|i| a.value(i) as f64).collect()
            }
            other => {
                return Err(format!("{path:?}: row_weight dtype {other:?} unsupported"));
            }
        };
        row_weights.extend(rw_vec);

        // EXP-CROSS-CODEC-V3: butter_a / butter_b → butter_diff.
        if butter_avail {
            let ba_col = batch.column(butter_a_idx.unwrap());
            let bb_col = batch.column(butter_b_idx.unwrap());
            let ba = read_f64_col(ba_col.as_ref(), n_rows)
                .map_err(|e| format!("{path:?}: butter_a {e}"))?;
            let bb = read_f64_col(bb_col.as_ref(), n_rows)
                .map_err(|e| format!("{path:?}: butter_b {e}"))?;
            for i in 0..n_rows {
                butter_diff.push(ba[i] - bb[i]);
            }
        }

        // fa_0..fa_<n_features-1>
        let mut a_per_col: Vec<Vec<f64>> = Vec::with_capacity(n_features);
        for i in 0..n_features {
            let col = batch.column(fa0_idx + i);
            let v: Vec<f64> = match col.data_type() {
                arrow::datatypes::DataType::Float64 => {
                    let a = col.as_any().downcast_ref::<Float64Array>().unwrap();
                    (0..n_rows).map(|j| a.value(j)).collect()
                }
                arrow::datatypes::DataType::Float32 => {
                    let a = col.as_any().downcast_ref::<Float32Array>().unwrap();
                    (0..n_rows).map(|j| a.value(j) as f64).collect()
                }
                other => {
                    return Err(format!("{path:?}: fa_{i} dtype {other:?} unsupported"));
                }
            };
            a_per_col.push(v);
        }
        features_a.extend(
            (0..n_rows).map(|row| a_per_col.iter().map(|col| col[row]).collect::<Vec<f64>>()),
        );

        // fb_0..fb_<n_features-1>
        let mut b_per_col: Vec<Vec<f64>> = Vec::with_capacity(n_features);
        for i in 0..n_features {
            let col = batch.column(fb0_idx + i);
            let v: Vec<f64> = match col.data_type() {
                arrow::datatypes::DataType::Float64 => {
                    let a = col.as_any().downcast_ref::<Float64Array>().unwrap();
                    (0..n_rows).map(|j| a.value(j)).collect()
                }
                arrow::datatypes::DataType::Float32 => {
                    let a = col.as_any().downcast_ref::<Float32Array>().unwrap();
                    (0..n_rows).map(|j| a.value(j) as f64).collect()
                }
                other => {
                    return Err(format!("{path:?}: fb_{i} dtype {other:?} unsupported"));
                }
            };
            b_per_col.push(v);
        }
        features_b.extend(
            (0..n_rows).map(|row| b_per_col.iter().map(|col| col[row]).collect::<Vec<f64>>()),
        );
    }

    Ok((features_a, features_b, row_weights, butter_diff))
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

/// `true` when the user passed `--<id>` explicitly on the command line
/// (as opposed to clap supplying its default). Explicit flags take
/// precedence over manifest values.
fn explicit(matches: &clap::ArgMatches, id: &str) -> bool {
    matches.value_source(id) == Some(ValueSource::CommandLine)
}

/// Overlay a parsed manifest onto `args` as DEFAULTS: a manifest field
/// is applied only when the corresponding flag was NOT given explicitly
/// on the command line (per `matches`). `--group` is replace-not-merge:
/// any explicit `--group` keeps the CLI groups; otherwise the manifest's
/// groups become `args.group`.
///
/// Returns the manifest's post-training `steps` so `main` can surface
/// them (the trainer cannot run them — they shell to external scripts).
fn apply_manifest_to_args(
    args: &mut Args,
    matches: &clap::ArgMatches,
    cfg: &train_manifest::ManifestConfig,
) -> Vec<String> {
    // --group: replace-not-merge. Only adopt manifest groups when the
    // user passed none explicitly.
    if !explicit(matches, "group") && !cfg.groups.is_empty() {
        args.group = cfg.groups.iter().map(|g| g.to_group_spec()).collect();
    }

    macro_rules! set_if_default {
        ($field:ident, $id:literal, $val:expr) => {
            if !explicit(matches, $id) {
                if let Some(v) = $val {
                    args.$field = v;
                }
            }
        };
    }

    set_if_default!(hidden, "hidden", cfg.hidden);
    set_if_default!(epochs, "epochs", cfg.epochs);
    set_if_default!(pairs_per_epoch, "pairs_per_epoch", cfg.pairs_per_epoch);
    set_if_default!(lr, "lr", cfg.lr);
    set_if_default!(l2, "l2", cfg.l2);
    set_if_default!(leaky_alpha, "leaky_alpha", cfg.leaky_alpha);
    set_if_default!(seed, "seed", cfg.seed);
    set_if_default!(val_policy, "val_policy", cfg.val_policy.clone());
    set_if_default!(val_aggregate, "val_aggregate", cfg.val_aggregate.clone());
    set_if_default!(max_features, "max_features", cfg.max_features);
    set_if_default!(minibatch_size, "minibatch_size", cfg.minibatch_size);
    set_if_default!(out_dtype, "out_dtype", cfg.out_dtype.clone());
    set_if_default!(target_column, "target_column", cfg.target_column.clone());
    set_if_default!(target_scale, "target_scale", cfg.target_scale);
    set_if_default!(n_hidden_layers, "n_hidden_layers", cfg.n_hidden_layers);
    set_if_default!(
        per_sample_alpha_head,
        "per_sample_alpha_head",
        cfg.per_sample_alpha_head
    );
    set_if_default!(mse_weight, "mse_weight", cfg.mse_weight);
    set_if_default!(ranknet_weight, "ranknet_weight", cfg.ranknet_weight);
    set_if_default!(monotonicity_reg, "monotonicity_reg", cfg.monotonicity_reg);
    set_if_default!(
        monotonicity_margin,
        "monotonicity_margin",
        cfg.monotonicity_margin
    );
    set_if_default!(
        tanh_output_head_scale,
        "tanh_output_head_scale",
        cfg.tanh_output_head_scale
    );
    set_if_default!(
        anchor_loss_weight,
        "anchor_loss_weight",
        cfg.anchor_loss_weight
    );
    set_if_default!(anchor_step_p, "anchor_step_p", cfg.anchor_step_p);
    set_if_default!(
        anchor_target_score,
        "anchor_target_score",
        cfg.anchor_target_score
    );

    // Masked-monotone recipe fields (bools + the mask path).
    set_if_default!(monotone_cbc, "monotone_cbc", cfg.monotone_cbc);
    set_if_default!(monotone_strict, "monotone_strict", cfg.monotone_strict);
    set_if_default!(
        monotone_pin_during_training,
        "monotone_pin_during_training",
        cfg.monotone_pin_during_training
    );
    set_if_default!(
        qat_fine_tune_epochs,
        "qat_fine_tune_epochs",
        cfg.qat_fine_tune_epochs
    );
    set_if_default!(qat_tau, "qat_tau", cfg.qat_tau);
    set_if_default!(group_eval_cap, "group_eval_cap", cfg.group_eval_cap);
    set_if_default!(
        konjnd_aggregation_weight,
        "konjnd_aggregation_weight",
        cfg.konjnd_aggregation_weight
    );
    if args.konjnd_aggregation_parquet.is_none() {
        args.konjnd_aggregation_parquet = cfg
            .konjnd_aggregation_parquet
            .clone()
            .map(std::path::PathBuf::from);
    }
    set_if_default!(
        konjnd_aggregation_step_p,
        "konjnd_aggregation_step_p",
        cfg.konjnd_aggregation_step_p
    );
    set_if_default!(ema_decay, "ema_decay", cfg.ema_decay);
    set_if_default!(hard_pair_frac, "hard_pair_frac", cfg.hard_pair_frac);
    set_if_default!(
        hard_pair_max_delta,
        "hard_pair_max_delta",
        cfg.hard_pair_max_delta
    );
    set_if_default!(stratified_bands, "stratified_bands", cfg.stratified_bands);
    set_if_default!(dro_eta, "dro_eta", cfg.dro_eta);
    set_if_default!(listwise_weight, "listwise_weight", cfg.listwise_weight);
    set_if_default!(listwise_size, "listwise_size", cfg.listwise_size);
    set_if_default!(listwise_frac, "listwise_frac", cfg.listwise_frac);
    set_if_default!(triplet_weight, "triplet_weight", cfg.triplet_weight);
    set_if_default!(triplet_frac, "triplet_frac", cfg.triplet_frac);
    set_if_default!(triplet_tau, "triplet_tau", cfg.triplet_tau);
    set_if_default!(triplet_sigma, "triplet_sigma", cfg.triplet_sigma);
    if args.triplet_stimuli.is_none() {
        args.triplet_stimuli = cfg.triplet_stimuli.clone();
    }
    if args.triplet_responses.is_none() {
        args.triplet_responses = cfg.triplet_responses.clone();
    }

    // Path-valued options (already resolved to absolute/relative-to-manifest).
    if !explicit(matches, "auto_transforms") && cfg.auto_transforms.is_some() {
        args.auto_transforms = cfg.auto_transforms.clone();
    }
    if !explicit(matches, "anchor_parquet") && cfg.anchor_parquet.is_some() {
        args.anchor_parquet = cfg.anchor_parquet.clone();
    }
    if !explicit(matches, "monotone_feature_mask") && cfg.monotone_feature_mask.is_some() {
        args.monotone_feature_mask = cfg.monotone_feature_mask.clone();
    }
    // --out: clap makes it required-unless-manifest, but with --manifest
    // we supply it from [bake].file when not given explicitly.
    if !explicit(matches, "out") && cfg.out.is_some() {
        args.out = cfg.out.clone();
    }

    cfg.post_training_steps.clone()
}

fn main() {
    // We parse via ArgMatches (not Args::parse) so --manifest can apply
    // its recorded fields as DEFAULTS while letting explicit CLI flags
    // win. `value_source(id) == CommandLine` tells us which flags the
    // user actually typed.
    let matches = Args::command().get_matches();
    let mut args = Args::from_arg_matches(&matches).unwrap_or_else(|e| e.exit());

    // The bake sha256 the manifest CLAIMS, hoisted out of the manifest block so
    // the write site can check the claim against what we actually produced.
    // `None` when no manifest, or when the manifest records no sha — a RECIPE
    // (the normal case since the 2026-07-15 collapse) describes how to train,
    // not one bake, so it makes no claim and none is checked. See the
    // verification right after `fs::write`.
    let mut manifest_claimed_sha: Option<(String, PathBuf)> = None;

    if let Some(manifest_path) = args.manifest.clone() {
        let cfg = train_manifest::parse_manifest(&manifest_path).unwrap_or_else(|e| {
            eprintln!("--manifest {}: {e}", manifest_path.display());
            std::process::exit(2);
        });
        manifest_claimed_sha = cfg
            .bake_sha256
            .clone()
            .map(|sha| (sha, manifest_path.clone()));

        // Reproduce-exactly gate #0: trainer version. The 2026-07-01 v47
        // reproduction proved training is DETERMINISTIC (pinned tree →
        // byte-identical bake) and that trainer drift alone breaks it
        // (current-main trainer produced a 57 KB collapsed bake from the
        // same manifest + data). When the manifest records the commit that
        // produced the bake, a mismatching trainer fails loud.
        if let Some(want) = cfg.trainer_commit.as_deref() {
            let head = std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
            match head {
                Some(h) if h.starts_with(want) || want.starts_with(h.as_str()) => {
                    eprintln!("[manifest] trainer_commit OK ({want})");
                }
                Some(h) => {
                    // HEAD moved past the recorded commit — that's fine as
                    // long as no TRAINER SOURCE changed in between (docs /
                    // manifests / benchmarks commits move HEAD constantly).
                    // The behavior-relevant question is source drift.
                    let src_unchanged = std::process::Command::new("git")
                        .args([
                            "diff",
                            "--quiet",
                            &format!("{want}..{h}"),
                            "--",
                            "zensim-validate/src",
                            "zensim-train-core/src",
                            "zensim/src",
                        ])
                        .status()
                        .map(|s| s.success())
                        .unwrap_or(false);
                    if src_unchanged {
                        eprintln!(
                            "[manifest] trainer_commit OK: HEAD {h} differs from \
                             recorded {want} but trainer sources are unchanged \
                             between them"
                        );
                    } else {
                        eprintln!(
                            "[manifest] trainer_commit MISMATCH: manifest records {want}, \
                             this trainer is built from {h}, and trainer sources CHANGED \
                             between them.\n\
                             Reproduce-exactly requires the recorded trainer: \
                             `jj workspace add ../<repo>--pin -r {want}` and build there.\n\
                             Pass --manifest-allow-sha-drift to override (the produced \
                             bake will NOT match the shipped one)."
                        );
                        if !args.manifest_allow_sha_drift {
                            std::process::exit(2);
                        }
                    }
                }
                None => eprintln!(
                    "[manifest] trainer_commit {want} recorded but git HEAD \
                     unavailable — cannot verify trainer version"
                ),
            }
        }

        // Load-bearing reproduce-exactly gate: verify every recorded
        // input file's sha256 BEFORE we touch the trainer. A drift means
        // the produced bake won't match the shipped one — fail loud.
        match train_manifest::verify_inputs(&cfg.inputs, args.manifest_allow_sha_drift) {
            Ok(warnings) => {
                for w in &warnings {
                    eprintln!("[manifest] WARNING: {w}");
                }
                eprintln!(
                    "[manifest] verified {} input file(s) from {}",
                    cfg.inputs.len(),
                    manifest_path.display()
                );
            }
            Err(e) => {
                eprintln!("[manifest] {e}");
                std::process::exit(2);
            }
        }

        let steps = apply_manifest_to_args(&mut args, &matches, &cfg);
        if !steps.is_empty() {
            eprintln!(
                "[manifest] {} post-training step(s) recorded — NOT executed by the trainer; \
                 apply them by hand after the bake is produced:",
                steps.len()
            );
            for s in &steps {
                eprintln!("[manifest]   - {s}");
            }
        }
    }

    // Resolve the output path now that the manifest (if any) has been
    // applied. clap's `required_unless_present = "manifest"` guarantees
    // an explicit `--out` whenever `--manifest` is absent; when present,
    // the manifest's `[bake].file` should have filled it in.
    let out_path: PathBuf = args.out.clone().unwrap_or_else(|| {
        eprintln!(
            "--out is required: neither an explicit --out nor a manifest [bake].file was provided"
        );
        std::process::exit(2);
    });

    let val_policy = match args.val_policy.to_lowercase().as_str() {
        "min" => ValidationPolicy::Min,
        "mean" => ValidationPolicy::Mean,
        "goals" => ValidationPolicy::Goals,
        other => {
            eprintln!("--val-policy must be 'min', 'mean', or 'goals'; got {other:?}");
            std::process::exit(2);
        }
    };

    let val_aggregate: zensim_validate::panel::ValAggregate =
        args.val_aggregate.parse().unwrap_or_else(|e| {
            eprintln!("--val-aggregate: {e}");
            std::process::exit(2);
        });

    // DATA-INTEGRITY GUARD (2026-05-25, task #215): refuse to TRAIN on a mock
    // target column. The kadid/tid iwssim corruption happened because a
    // validation-only mock column (a verbatim copy of human_score) leaked into
    // training parquets after its "mock" filename qualifier was renamed away.
    // Mock columns are now suffixed `_MOCK_VAL_ONLY`; any `--target-column`
    // matching `*mock*` (case-insensitive) is a validation-only signal and
    // must never carry training gradient.
    if args.target_column.to_ascii_lowercase().contains("mock") {
        let any_train = args
            .group
            .iter()
            .filter_map(|s| parse_group_spec(s).ok())
            .any(|(_, _, train_w, _, _, _)| train_w > 0.0);
        if any_train {
            eprintln!(
                "DATA-INTEGRITY: --target-column {:?} is a MOCK (validation-only) column \
                 but at least one --group has train_weight > 0. A mock target is a copy of \
                 human_score and must never carry training gradient (this is the 2026-05-25 \
                 iwssim leak). Set all training groups' train_weight to 0.0, or pick a real \
                 target column.",
                args.target_column
            );
            std::process::exit(2);
        }
    }

    // BANNED: narrow feature caps. 228/300 silently drop the extended +
    // IW-pool blocks (f228..f371) and produce a bake that can't be validated
    // against the canonical 372-col corpora (the 2026-06-30 footgun). The
    // with-iw 372 width is the ship floor; only an explicit research opt-in
    // may go narrower (never a runtime weight).
    if args.max_features < 372 && !args.allow_narrow_features {
        eprintln!(
            "BANNED: --max-features {} < 372. The 228/300 regimes are not \
             shippable — a narrow cap drops the extended + IW-pool features \
             (f228..f371) and mis-aligns the train/inference space. Use \
             --max-features 372 (the v47/A with-iw width), or pass \
             --allow-narrow-features for a research/classifier bake that never \
             ships as a ZensimProfile weight.",
            args.max_features
        );
        std::process::exit(2);
    }

    // Load all groups, infer n_features from the first.
    let mut loaded: Vec<LoadedGroup> = Vec::new();
    let mut n_features = 0usize;
    for spec in &args.group {
        let (name, path, train_w, val_w, within_ref, loss_mode) = parse_group_spec(spec)
            .unwrap_or_else(|e| {
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
        let mut g = load_group_dispatch(&path, &name, &args.target_column, args.target_scale)
            .unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1);
            });
        g.train_w = train_w;
        g.val_w = val_w;
        // MANDATORY reproduction identity: canonical absolute path + content
        // sha256 of every training input, embedded into the bake's
        // `zentrain.repro`. argv alone is not enough — its paths can be
        // worker-relative and die with the worker dir; the sha256 outlives
        // moves and renames. Hashing 100s of MB adds seconds to a
        // minutes-long train run; failure to hash is loud but non-fatal.
        g.source_path = std::fs::canonicalize(&path)
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| path.display().to_string());
        g.source_sha256 = train_manifest::sha256_file(&path).unwrap_or_else(|e| {
            eprintln!("[repro] warning: could not sha256 {}: {e}", path.display());
            String::from("unhashed")
        });
        // Within-ref pairing needs per-row ref identity. If the group asked
        // for it but the loader could not supply ids (CSV input, or a
        // parquet with neither `ref_basename` nor `image_path`), FAIL —
        // silently falling back to cross-image draws is precisely the
        // confound the flag exists to avoid, and it would be invisible in
        // the logs.
        if within_ref && g.ref_ids.is_none() {
            eprintln!(
                "group {name:?}: ':withinref' requires per-row reference identity, but \
                 {} supplies none (CSV inputs never do; a parquet needs a `ref_basename` \
                 or `image_path` column). Refusing to train: falling back to cross-image \
                 pairs would silently reintroduce the scale confound that within-ref \
                 exists to prevent.",
                path.display()
            );
            std::process::exit(2);
        }
        g.within_ref = within_ref;
        g.loss_mode = loss_mode;
        let cap = args.max_features;
        if g.n_features > cap {
            // Narrow the flat buffer to the first `cap` features of each
            // row — same kept values as the old per-row `row.truncate(cap)`.
            let old_nf = g.n_features;
            let n_rows = g.human_scores.len();
            let mut narrowed: Vec<f64> = Vec::with_capacity(n_rows * cap);
            for r in 0..n_rows {
                narrowed.extend_from_slice(&g.feature_rows[r * old_nf..r * old_nf + cap]);
            }
            g.feature_rows = narrowed;
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
            let nf = g.n_features;
            for row in g.feature_rows.chunks_mut(nf) {
                for (i, t) in transforms.iter().enumerate() {
                    if *t != zenpredict::FeatureTransform::Identity {
                        row[i] = t.apply_with_params(row[i] as f32, &params[i]) as f64;
                    }
                }
            }
            // `sweep_nan_inf` only reads — hand it row views of the flat
            // buffer (the `.to_vec()` that used to be here deep-cloned the
            // whole group, up to ~1.6 GB on the largest wave-10 leg, for a
            // checker that never mutates).
            if let Err(e) = mlp_train::sweep_nan_inf(
                g.feature_rows.chunks(nf),
                &transforms,
                &format!("group '{}'", g.name),
            ) {
                eprintln!("FATAL: {e}");
                std::process::exit(1);
            }
        }
        let any_params = params.iter().any(|p| !p.is_empty());
        (
            Some(transforms),
            if any_params { Some(params) } else { None },
        )
    };

    // FEATURE-SUBSET ablation (`--keep-features`): zero the raw values of
    // every dropped column, in every loaded group, BEFORE the scaler is
    // computed. A constant-zero column standardizes to exactly 0.0
    // (`mean = 0`, `(0 − 0)/s = 0`), which makes its layer-1 gradient
    // exactly 0.0 for the whole run; the trainer core pins the matching
    // layer-1 rows to 0.0 once at init (`INPUT_KEEP_MASK`). Net effect: an
    // exact K-wide fit, at zero per-step cost, with exactly-zero baked rows.
    let keep_mask: Option<Vec<bool>> = match args.keep_features.as_deref() {
        None => None,
        Some(spec) => {
            let idx = match parse_keep_features(spec, args.max_features) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("FATAL: --keep-features: {e}");
                    std::process::exit(1);
                }
            };
            let mut mask = vec![false; args.max_features];
            for &i in &idx {
                mask[i] = true;
            }
            let n_keep = mask.iter().filter(|&&k| k).count();
            println!(
                "keep_features: {n_keep} of {} inputs kept ({} dropped)",
                args.max_features,
                args.max_features - n_keep
            );
            for g in &mut loaded {
                for row in g.feature_rows.chunks_mut(g.n_features) {
                    for (d, keep) in mask.iter().enumerate() {
                        if !keep && d < row.len() {
                            row[d] = 0.0;
                        }
                    }
                }
            }
            *zensim_validate::mlp_train::INPUT_KEEP_MASK.lock().unwrap() = Some(mask.clone());
            Some(mask)
        }
    };

    let out_dtype = match args.out_dtype.to_ascii_lowercase().as_str() {
        "f32" => zenpredict::WeightDtype::F32,
        "f16" => zenpredict::WeightDtype::F16,
        "i8" => zenpredict::WeightDtype::I8,
        other => {
            eprintln!("--out-dtype must be one of f32 / f16 / i8, got {other:?}");
            std::process::exit(2);
        }
    };

    // Parse the per-feature monotone sign mask (if supplied).
    let monotone_feature_pin: Option<Vec<bool>> = args.monotone_feature_mask.as_ref().map(|path| {
        let txt = std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("--monotone-feature-mask read error {}: {e}", path.display());
            std::process::exit(2);
        });
        let mut pin = vec![true; n_features];
        for (li, line) in txt.lines().enumerate() {
            if li == 0 {
                continue; // header
            }
            let mut cols = line.split('\t');
            let idx: usize = match cols.next().and_then(|s| s.trim().parse().ok()) {
                Some(i) => i,
                None => continue,
            };
            let mask = cols.next().unwrap_or("").trim();
            if idx < n_features {
                pin[idx] = mask == "pin_geq0";
            }
        }
        let n_pin = pin.iter().filter(|&&p| p).count();
        eprintln!(
            "monotone-feature-mask: {n_pin} pinned ≥0 / {} {} of {n_features} (from {})",
            n_features - n_pin,
            if args.monotone_strict {
                "DROPPED"
            } else {
                "free"
            },
            path.display()
        );
        pin
    });

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
        val_aggregate,
        low_q_boost: args.low_q_boost,
        mid_q_boost: args.mid_q_boost,
        high_q_boost: args.high_q_boost,
        out_dtype,
        feature_transforms: feature_transforms.clone(),
        feature_transform_params: feature_transform_params.clone(),
        minibatch_size: args.minibatch_size.max(1),
        parallel_batch: args.parallel_batch,
        pwrc_pair_weight: args.pwrc_pair_weight,
        pwrc_sensory_threshold: args.pwrc_sensory_threshold,
        pwrc_band_weights: args.pwrc_band_weights.clone(),
        norm_in_norm_weight: args.norm_in_norm_weight,
        norm_in_norm_p: args.norm_in_norm_p,
        norm_in_norm_q: args.norm_in_norm_q,
        pool_head: args.pool_head,
        hybrid_head: args.hybrid_head,
        per_sample_alpha_head: args.per_sample_alpha_head,
        skip_connection: args.skip_connection,
        n_hidden_layers: args.n_hidden_layers,
        mse_weight: args.mse_weight,
        sigma_weighted_mse: args.sigma_weighted_mse,
        ema_decay: args.ema_decay,
        hard_pair_frac: args.hard_pair_frac,
        hard_pair_max_delta: args.hard_pair_max_delta,
        stratified_bands: args.stratified_bands,
        dro_eta: args.dro_eta,
        listwise_weight: args.listwise_weight,
        listwise_size: args.listwise_size,
        listwise_frac: args.listwise_frac,
        triplet_weight: args.triplet_weight,
        triplet_frac: args.triplet_frac,
        triplet_tau: args.triplet_tau,
        triplet_sigma: args.triplet_sigma,
        ranknet_weight: args.ranknet_weight,
        monotonicity_reg: args.monotonicity_reg,
        monotone_cbc: args.monotone_cbc,
        monotone_feature_pin,
        monotone_strict: args.monotone_strict,
        monotone_pin_during_training: args.monotone_pin_during_training,
        qat_fine_tune_epochs: args.qat_fine_tune_epochs,
        qat_tau: args.qat_tau,
        group_eval_cap: args.group_eval_cap,
        monotonicity_margin: args.monotonicity_margin,
        anchor_loss_weight: args.anchor_loss_weight,
        anchor_target_score: args.anchor_target_score,
        anchor_step_p: args.anchor_step_p,
        cross_codec_eq_weight: args.cross_codec_eq_weight,
        cross_codec_eq_step_p: args.cross_codec_eq_step_p,
        cross_codec_rank_preserve_weight: args.cross_codec_rank_preserve_weight,
        dynamic_range_floor_weight: args.dynamic_range_floor_weight,
        dynamic_range_sigma_threshold: args.dynamic_range_sigma_threshold,
        dynamic_range_step_p: args.dynamic_range_step_p,
        dynamic_range_probe_n: args.dynamic_range_probe_n,
        tanh_output_head_scale: args.tanh_output_head_scale,
        pjnd_passthrough_weight: args.pjnd_passthrough_weight,
        pjnd_passthrough_step_p: args.pjnd_passthrough_step_p,
        pjnd_passthrough_target_score: args.pjnd_passthrough_target_score,
        konjnd_aggregation_weight: args.konjnd_aggregation_weight,
        konjnd_aggregation_step_p: args.konjnd_aggregation_step_p,
        konjnd_aggregation_samples_per_ref: args.konjnd_aggregation_samples_per_ref,
        konjnd_aggregation_refs_per_step: args.konjnd_aggregation_refs_per_step,
    };

    println!(
        "Training: {} groups, {n_features} features, {hyperparams:?}",
        loaded.len()
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
            for row in g.feature_rows.chunks(g.n_features) {
                all_features.push(row.to_vec());
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
            margin: args.tv_margin,
        })
    } else {
        None
    };

    // PreviewV0_5TunerV2 (2026-05-19): load optional anchor parquet.
    // The parquet uses `anchor_weight` as the target column (we override
    // the target score in the trainer); we keep the column-based loader
    // call to reuse the same parquet machinery — the loaded
    // `human_scores` field becomes the per-row anchor weight vector.
    //
    // V5 extension (2026-05-19): when the parquet has a `target_score`
    // column (multi-band piecewise anchors), load it and pass per-row
    // targets to the trainer. V4-style single-band parquets without
    // that column fall back to `--anchor-target-score`.
    let (anchor_feat_storage, anchor_row_weights, anchor_target_scores): (
        Vec<Vec<f64>>,
        Vec<f64>,
        Option<Vec<f64>>,
    ) = if let Some(anchor_path) = &args.anchor_parquet {
        if args.anchor_loss_weight <= 0.0 {
            eprintln!(
                "WARNING: --anchor-parquet set but --anchor-loss-weight is 0; \
                 anchor data will be ignored."
            );
        }
        let loader = zensim_validate::parquet_loader::load_parquet(
            anchor_path,
            "jnd_anchor",
            "anchor_weight",
            1.0,
        )
        .unwrap_or_else(|e| {
            eprintln!("anchor parquet load failed: {e}");
            std::process::exit(1);
        });
        let cap = n_features;
        let mut feat: Vec<Vec<f64>> = loader.feature_rows;
        for row in &mut feat {
            row.truncate(cap);
        }
        // Apply feature transforms in-place if active (same as groups).
        if let Some(ts) = &feature_transforms {
            let pp = feature_transform_params.as_ref();
            for row in &mut feat {
                for (i, t) in ts.iter().enumerate() {
                    if *t != zenpredict::FeatureTransform::Identity {
                        let p = pp.map(|v| v[i].as_slice()).unwrap_or(&[][..]);
                        row[i] = t.apply_with_params(row[i] as f32, p) as f64;
                    }
                }
            }
            if let Err(e) =
                mlp_train::sweep_nan_inf(feat.iter().map(|r| r.as_slice()), ts, "anchor parquet")
            {
                eprintln!("FATAL: {e}");
                std::process::exit(1);
            }
        }
        // V5: optional per-row `target_score` column. Row ordering
        // matches `load_parquet` (same sequential batch scan).
        let target_scores = zensim_validate::parquet_loader::load_optional_scalar_column(
            anchor_path,
            "target_score",
        )
        .unwrap_or_else(|e| {
            eprintln!("anchor target_score load failed: {e}");
            std::process::exit(1);
        });
        if let Some(ref ts) = target_scores {
            if ts.len() != feat.len() {
                eprintln!(
                    "anchor target_score column length ({}) != feature rows ({}); \
                     refusing to mix shapes",
                    ts.len(),
                    feat.len(),
                );
                std::process::exit(1);
            }
            // Per-row target span summary.
            let mut ts_sorted = ts.clone();
            ts_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let (mn, mx) = (
                ts_sorted.first().copied().unwrap_or(0.0),
                ts_sorted.last().copied().unwrap_or(0.0),
            );
            let med = ts_sorted[ts_sorted.len() / 2];
            eprintln!(
                "anchor parquet: loaded {} rows from {:?} (PER-ROW target_score: min={:.1} median={:.1} max={:.1}, weight={}, step_p={})",
                feat.len(),
                anchor_path,
                mn,
                med,
                mx,
                args.anchor_loss_weight,
                args.anchor_step_p,
            );
        } else {
            eprintln!(
                "anchor parquet: loaded {} rows from {:?} (V4-style global target={}, weight={}, step_p={})",
                feat.len(),
                anchor_path,
                args.anchor_target_score,
                args.anchor_loss_weight,
                args.anchor_step_p,
            );
        }
        (feat, loader.human_scores, target_scores)
    } else {
        (Vec::new(), Vec::new(), None)
    };
    let anchor_feat_refs: Vec<&[f64]> = anchor_feat_storage.iter().map(|r| r.as_slice()).collect();
    let anchor_loaded: Option<AnchorRows<'_>> =
        if args.anchor_parquet.is_some() && !anchor_feat_storage.is_empty() {
            Some(AnchorRows {
                name: "jnd_anchor".to_string(),
                features: anchor_feat_refs.as_slice(),
                row_weights: anchor_row_weights.as_slice(),
                target_scores: anchor_target_scores.as_deref(),
            })
        } else {
            None
        };

    // EXP-CROSS-CODEC-METRIC equivalence pair loader (2026-05-19).
    // Schema: ref_basename, codec_a, q_a, codec_b, q_b, butter_level,
    //         butter_a, butter_b, row_weight, fa_0..fa_<M-1>, fb_0..fb_<M-1>
    // We discard the bookkeeping columns and just collect the standardized
    // (features_a, features_b) pairs + row_weight.
    let (equiv_a_storage, equiv_b_storage, equiv_row_weights, equiv_butter_diff): EquivColumns =
        if let Some(equiv_path) = &args.cross_codec_eq_parquet {
            if args.cross_codec_eq_weight <= 0.0 {
                eprintln!(
                    "WARNING: --cross-codec-eq-parquet set but \
                 --cross-codec-eq-weight is 0; equiv data will be ignored."
                );
            }
            match load_equiv_parquet(equiv_path, n_features) {
                Ok((a, b, w, bd)) => {
                    eprintln!(
                        "equiv parquet: loaded {} pairs from {:?} (weight={}, step_p={}, butter_diff={})",
                        a.len(),
                        equiv_path,
                        args.cross_codec_eq_weight,
                        args.cross_codec_eq_step_p,
                        if bd.is_empty() {
                            "absent".to_string()
                        } else {
                            format!("{}", bd.len())
                        },
                    );
                    (a, b, w, bd)
                }
                Err(e) => {
                    eprintln!("equiv parquet load failed: {e}");
                    std::process::exit(1);
                }
            }
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };
    let equiv_a_refs: Vec<&[f64]> = equiv_a_storage.iter().map(|r| r.as_slice()).collect();
    let equiv_b_refs: Vec<&[f64]> = equiv_b_storage.iter().map(|r| r.as_slice()).collect();
    let equiv_loaded: Option<EquivPairs<'_>> =
        if args.cross_codec_eq_parquet.is_some() && !equiv_a_storage.is_empty() {
            Some(EquivPairs {
                name: "cross_codec_eq".to_string(),
                features_a: equiv_a_refs.as_slice(),
                features_b: equiv_b_refs.as_slice(),
                row_weights: equiv_row_weights.as_slice(),
                butter_diff: equiv_butter_diff.as_slice(),
            })
        } else {
            None
        };

    // EXP-V11-D-PJND-DOMINANT (2026-05-20, task #198) — second anchor
    // pool from konjnd-dense.parquet. The canonical konjnd-dense
    // schema has `pjnd_target` (KonJND PJND threshold in ssim2 space)
    // but the V11-D recipe uses a CONSTANT target=80.0 (the V10 PJND
    // calibration point). Per-row weight is constant 1.0 (no
    // preferential sampling on KonJND).
    //
    // We reuse `load_parquet` with `human_score` as the target column
    // (any present score column works — it's discarded). After load
    // we OVERRIDE row_weights to 1.0/row and drop the per-row
    // target_scores so the global `--pjnd-passthrough-target-score`
    // is used uniformly.
    let (pjnd_feat_storage, pjnd_row_weights): (Vec<Vec<f64>>, Vec<f64>) = if let Some(pjnd_path) =
        &args.pjnd_passthrough_parquet
    {
        if args.pjnd_passthrough_weight <= 0.0 {
            eprintln!(
                "WARNING: --pjnd-passthrough-parquet set but \
                 --pjnd-passthrough-weight is 0; pjnd anchor data will be ignored."
            );
        }
        let loader = zensim_validate::parquet_loader::load_parquet(
            pjnd_path,
            "pjnd_anchor",
            "human_score",
            1.0,
        )
        .unwrap_or_else(|e| {
            eprintln!("pjnd-passthrough parquet load failed: {e}");
            std::process::exit(1);
        });
        let cap = n_features;
        let mut feat: Vec<Vec<f64>> = loader.feature_rows;
        for row in &mut feat {
            row.truncate(cap);
        }
        // Apply feature transforms in-place if active (same as groups).
        if let Some(ts) = &feature_transforms {
            let pp = feature_transform_params.as_ref();
            for row in &mut feat {
                for (i, t) in ts.iter().enumerate() {
                    if *t != zenpredict::FeatureTransform::Identity {
                        let p = pp.map(|v| v[i].as_slice()).unwrap_or(&[][..]);
                        row[i] = t.apply_with_params(row[i] as f32, p) as f64;
                    }
                }
            }
            if let Err(e) = mlp_train::sweep_nan_inf(
                feat.iter().map(|r| r.as_slice()),
                ts,
                "pjnd-passthrough parquet",
            ) {
                eprintln!("FATAL: {e}");
                std::process::exit(1);
            }
        }
        // V11-D: constant per-row weight 1.0 (no preferential sampling).
        let weights = vec![1.0_f64; feat.len()];
        eprintln!(
            "pjnd-passthrough parquet: loaded {} rows from {:?} (CONSTANT row_weight=1.0, global target_score={}, weight={}, step_p={})",
            feat.len(),
            pjnd_path,
            args.pjnd_passthrough_target_score,
            args.pjnd_passthrough_weight,
            args.pjnd_passthrough_step_p,
        );
        (feat, weights)
    } else {
        (Vec::new(), Vec::new())
    };
    let pjnd_feat_refs: Vec<&[f64]> = pjnd_feat_storage.iter().map(|r| r.as_slice()).collect();
    let pjnd_anchor_loaded: Option<AnchorRows<'_>> =
        if args.pjnd_passthrough_parquet.is_some() && !pjnd_feat_storage.is_empty() {
            Some(AnchorRows {
                name: "pjnd_passthrough".to_string(),
                features: pjnd_feat_refs.as_slice(),
                row_weights: pjnd_row_weights.as_slice(),
                target_scores: None, // global fallback to --pjnd-passthrough-target-score
            })
        } else {
            None
        };

    // KONJND-AGGREGATION-HEAD (task #4, 2026-05-24) — load the
    // per-source-grouped konjnd-dense pool. Same canonical parquet as
    // pjnd_passthrough but read via the aggregation-aware loader so
    // we get the per-ref grouping metadata. Applies feature transforms
    // in-place; standardization happens inside the trainer using the
    // primary-stream scaler (same invariant as anchor + pjnd anchors).
    // FAIL-LOUD (2026-07-03): weight>0 with no pool parquet was a SILENT
    // no-op — wave-4 kagg cells reproduced base byte-identically. A recipe
    // that asks for the aggregation head must provide its data.
    if args.konjnd_aggregation_weight > 0.0 && args.konjnd_aggregation_parquet.is_none() {
        panic!(
            "--konjnd-aggregation-weight > 0 requires --konjnd-aggregation-parquet (silent no-op guard)"
        );
    }
    let konjnd_agg_owned: Option<zensim_validate::parquet_loader::OwnedKonjndAggregationPool> =
        if let Some(p) = &args.konjnd_aggregation_parquet {
            if args.konjnd_aggregation_weight <= 0.0 {
                eprintln!(
                    "WARNING: --konjnd-aggregation-parquet set but \
                     --konjnd-aggregation-weight is 0; aggregation pool will be ignored."
                );
            }
            let mut pool = zensim_validate::parquet_loader::load_konjnd_aggregation_pool(
                p,
                "konjnd_aggregation",
            )
            .unwrap_or_else(|e| {
                eprintln!("konjnd-aggregation parquet load failed: {e}");
                std::process::exit(1);
            });
            // Truncate to current --max-features.
            let cap = n_features;
            for row in &mut pool.feature_rows {
                row.truncate(cap);
            }
            // Apply feature transforms in-place (same as groups + pjnd).
            if let Some(ts) = &feature_transforms {
                let pp = feature_transform_params.as_ref();
                for row in &mut pool.feature_rows {
                    for (i, t) in ts.iter().enumerate() {
                        if *t != zenpredict::FeatureTransform::Identity {
                            let p = pp.map(|v| v[i].as_slice()).unwrap_or(&[][..]);
                            row[i] = t.apply_with_params(row[i] as f32, p) as f64;
                        }
                    }
                }
                if let Err(e) = mlp_train::sweep_nan_inf(
                    pool.feature_rows.iter().map(|r| r.as_slice()),
                    ts,
                    "konjnd-aggregation parquet",
                ) {
                    eprintln!("FATAL: {e}");
                    std::process::exit(1);
                }
            }
            eprintln!(
                "konjnd-aggregation parquet: loaded {} rows / {} refs (weight={}, step_p={}, S={}, K={})",
                pool.feature_rows.len(),
                pool.ref_ranges.len(),
                args.konjnd_aggregation_weight,
                args.konjnd_aggregation_step_p,
                args.konjnd_aggregation_samples_per_ref,
                args.konjnd_aggregation_refs_per_step,
            );
            Some(pool)
        } else {
            None
        };
    // Build the borrowing-view that the trainer consumes. We must
    // build feature row-refs in a separately-named slot so the
    // borrows stay live across the train_mlp call.
    let konjnd_agg_feat_refs: Vec<&[f64]> = konjnd_agg_owned
        .as_ref()
        .map(|p| p.feature_rows.iter().map(|r| r.as_slice()).collect())
        .unwrap_or_default();
    let konjnd_agg_loaded: Option<KonjndAggregationPool<'_>> =
        konjnd_agg_owned.as_ref().map(|p| KonjndAggregationPool {
            name: p.name.clone(),
            features: konjnd_agg_feat_refs.as_slice(),
            ref_ranges: p.ref_ranges.as_slice(),
            ref_pjnd_target: p.ref_pjnd_target.as_slice(),
            ref_weight: p.ref_weight.as_slice(),
        });

    let mut log: Vec<String> = Vec::new();

    // GPU trainer dispatch (task #166, 2026-05-19). Only active when
    // the user passes `--gpu-runtime <name>` AND the recipe is
    // GPU-supported (per-sample-α head, no aux losses, no TV).
    let gpu_runtime_str = args.gpu_runtime.trim().to_ascii_lowercase();
    let want_gpu = !gpu_runtime_str.is_empty() && gpu_runtime_str != "cpu";
    // Scale-mass regularizer: build the per-feature L2 multiplier and hand it
    // to the trainer via the module global (uniform when mult == 1.0). Coarse
    // scales are s2/s3 of each region: basic 78..156, v2 546..720, append
    // 822..924 (scale-major layouts: basic 39/scale, v2 87/scale, append
    // 51/scale). The v1-pool 156..372 is left at 1.0 (structural zeros in the
    // folded regimes; real pools in v1 bakes are not scale-separable here).
    if args.coarse_decay > 0.0 {
        *zensim_validate::mlp_train::COARSE_DECAY_RATE
            .lock()
            .unwrap() = args.coarse_decay;
        println!(
            "[coarse-decay] decoupled decay ON: rate {}",
            args.coarse_decay
        );
    }
    if args.coarse_l2_mult != 1.0 || args.coarse_decay > 0.0 {
        let nf = args.max_features;
        // When only --coarse-decay is given, mark coarse rows at 2.0 so the
        // decay gate (m > 1) engages; the rate absorbs the scaling.
        let eff = if args.coarse_l2_mult != 1.0 {
            args.coarse_l2_mult
        } else {
            2.0
        };
        let mut mult = vec![1.0f64; nf];
        let coarse = |r: core::ops::Range<usize>, mult: &mut Vec<f64>| {
            for i in r {
                if i < nf {
                    mult[i] = eff;
                }
            }
        };
        coarse(78..156, &mut mult);
        coarse(546..720, &mut mult);
        coarse(822..924, &mut mult);
        *zensim_validate::mlp_train::L2_FEATURE_MULT.lock().unwrap() =
            Some(std::sync::Arc::new(mult));
        println!(
            "[coarse-l2] scale-mass regularizer ON: mult {} on basic-s2/s3 + v2-s2/s3 + append-s2/s3",
            args.coarse_l2_mult
        );
    }
    if args.group_l1 > 0.0 {
        *zensim_validate::mlp_train::GROUP_L1_LAMBDA.lock().unwrap() = args.group_l1;
        println!("[group-l1] group-lasso prox ON: lambda {}", args.group_l1);
    }
    // Both feature-subset knobs are wired ONLY on the plain
    // `n_features → n_hidden → 1` strategy path (the one every 944-regime
    // recipe uses). The head architectures keep their layer-1 weights in a
    // different owner (`pool_head` / `hybrid_head` / `per_sample_alpha_head`)
    // and would silently ignore the flag — fail loud instead of shipping a
    // bake whose spec claims a subset it never trained.
    if args.keep_features.is_some() || args.group_l1 > 0.0 {
        let unsupported = [
            (args.pool_head, "--pool-head"),
            (args.hybrid_head, "--hybrid-head"),
            (args.per_sample_alpha_head, "--per-sample-alpha-head"),
            (args.n_hidden_layers >= 2, "--n-hidden-layers >= 2"),
            (want_gpu, "--gpu-runtime"),
        ];
        for (hit, name) in unsupported {
            if hit {
                eprintln!(
                    "FATAL: --keep-features / --group-l1 are implemented on the plain \
                     n_features→n_hidden→1 path only; {name} routes layer-1 weights through a \
                     different owner and would silently ignore them."
                );
                std::process::exit(1);
            }
        }
    }

    // Build TrainingGroups. This MOVES the raw feature rows under a mutable
    // borrow of `loaded`: the trainer releases each row as it standardizes
    // it (see `TrainingGroup::features`), which is what keeps a lane's RSS
    // at one copy of the features instead of two. Constructed last, and
    // immediately before the training call, so every reader of `loaded`
    // above — the TV concatenation, the provenance/spec blocks — still sees
    // the rows; the borrow ends at the call and `loaded`'s metadata is
    // readable again for the repro sidecars below.
    let mut groups: Vec<TrainingGroup> = loaded
        .iter_mut()
        .map(|g| TrainingGroup {
            name: g.name.clone(),
            human_scores: &g.human_scores,
            features: mlp_train::FeatureRows::Releasable {
                n_rows: g.human_scores.len(),
                n_features: g.n_features,
                data: &mut g.feature_rows,
            },
            metric_sigmas: g.metric_sigmas.as_deref(),
            train_weight: g.train_w,
            validation_weight: g.val_w,
            // Only surface ref ids when the group opted in: `Some` IS the
            // within-ref switch in the trainer core.
            loss_mode: g.loss_mode,
            ref_ids: if g.within_ref {
                g.ref_ids.as_deref()
            } else {
                None
            },
        })
        .collect();

    let bake_bytes = if want_gpu {
        if !args.per_sample_alpha_head {
            eprintln!(
                "--gpu-runtime requires --per-sample-alpha-head (Phase 1 GPU MVP only ports the \
                 per-sample-α head)."
            );
            std::process::exit(2);
        }
        if tv_regularizer.is_some() {
            eprintln!("--gpu-runtime is incompatible with --tv-pairs-file (Phase 2 work).");
            std::process::exit(2);
        }
        if hyperparams.norm_in_norm_weight > 0.0 {
            eprintln!(
                "--gpu-runtime is incompatible with --norm-in-norm-weight \
                 (Phase 2+ work). Run with the default CPU path."
            );
            std::process::exit(2);
        }
        // Phase 2 (2026-05-19, task #169): anchor, cross-codec-eq,
        // rank-preserve, and σ-floor aux losses are GPU-supported via
        // dedicated CubeCL kernels in `zensim-train-gpu`. We thread the
        // anchor/equiv data through `train_per_sample_alpha_head_gpu_with_aux`.
        let runtime = match gpu_runtime_str.as_str() {
            "cuda" => zensim_train_gpu::GpuRuntime::Cuda,
            "wgpu" => zensim_train_gpu::GpuRuntime::Wgpu,
            "gpucpu" => zensim_train_gpu::GpuRuntime::Cpu,
            other => {
                eprintln!(
                    "--gpu-runtime must be one of cuda / wgpu / gpucpu / cpu (empty), got {other:?}"
                );
                std::process::exit(2);
            }
        };
        let gpu_hp = zensim_train_gpu::GpuHparams {
            n_hidden: hyperparams.n_hidden,
            n_epochs: hyperparams.n_epochs,
            pairs_per_epoch: hyperparams.pairs_per_epoch,
            minibatch_k: hyperparams.minibatch_size.max(512),
            initial_lr: hyperparams.initial_lr,
            leaky_alpha: hyperparams.leaky_alpha,
            seed: hyperparams.seed,
            l2_lambda: hyperparams.l2_lambda,
            mse_weight: hyperparams.mse_weight,
            ranknet_weight: hyperparams.ranknet_weight,
            monotonicity_reg: hyperparams.monotonicity_reg,
            monotonicity_margin: hyperparams.monotonicity_margin,
            tanh_output_head_scale: hyperparams.tanh_output_head_scale,
            anchor_loss_weight: hyperparams.anchor_loss_weight,
            anchor_step_p: hyperparams.anchor_step_p,
            cross_codec_eq_weight: hyperparams.cross_codec_eq_weight,
            cross_codec_eq_step_p: hyperparams.cross_codec_eq_step_p,
            cross_codec_rank_preserve_weight: hyperparams.cross_codec_rank_preserve_weight,
            dynamic_range_floor_weight: hyperparams.dynamic_range_floor_weight,
            dynamic_range_probe_n: hyperparams.dynamic_range_probe_n,
            dynamic_range_sigma_threshold: hyperparams.dynamic_range_sigma_threshold,
            dynamic_range_step_p: hyperparams.dynamic_range_step_p,
            minibatch_k_aux: args.gpu_minibatch_k_aux.max(1),
            pjnd_passthrough_weight: hyperparams.pjnd_passthrough_weight,
            pjnd_passthrough_step_p: hyperparams.pjnd_passthrough_step_p,
        };
        log.push(format!(
            "GPU trainer: runtime={gpu_runtime_str:?} hparams={gpu_hp:?}"
        ));
        // Remap `mlp_train::TrainingGroup` to `zensim_train_core::TrainingGroup`
        // (same shape, different type identity per crate boundary). The core
        // type still takes a borrowed row table, so materialize one here —
        // 16 B/row, and only on the GPU path.
        let core_feat_refs: Vec<Vec<&[f64]>> =
            groups.iter().map(|g| g.features.iter().collect()).collect();
        let core_groups: Vec<zensim_train_core::TrainingGroup<'_>> = groups
            .iter()
            .zip(core_feat_refs.iter())
            .map(|(g, fr)| zensim_train_core::TrainingGroup {
                name: g.name.clone(),
                human_scores: g.human_scores,
                features: fr.as_slice(),
                metric_sigmas: g.metric_sigmas,
                train_weight: g.train_weight,
                validation_weight: g.validation_weight,
            })
            .collect();

        // Phase 2 aux: rehydrate anchor + equiv as GpuAnchorRows / GpuEquivPairs.
        // The CPU AnchorRows has `target_scores: Option<&[f64]>` (with a
        // global fallback); the GPU variant requires per-row targets so
        // we materialize a constant vector when the parquet didn't ship
        // a `target_score` column.
        let gpu_anchor_targets_owned: Option<Vec<f64>> = if let Some(a) = anchor_loaded.as_ref() {
            if a.target_scores.is_some() {
                None
            } else {
                Some(vec![args.anchor_target_score; a.features.len()])
            }
        } else {
            None
        };
        let gpu_anchor: Option<zensim_train_gpu::GpuAnchorRows<'_>> =
            anchor_loaded
                .as_ref()
                .map(|a| zensim_train_gpu::GpuAnchorRows {
                    name: a.name.clone(),
                    features: a.features,
                    row_weights: a.row_weights,
                    target_scores: a
                        .target_scores
                        .unwrap_or(gpu_anchor_targets_owned.as_deref().unwrap_or(&[])),
                });
        let gpu_equiv: Option<zensim_train_gpu::GpuEquivPairs<'_>> =
            equiv_loaded
                .as_ref()
                .map(|e| zensim_train_gpu::GpuEquivPairs {
                    name: e.name.clone(),
                    features_a: e.features_a,
                    features_b: e.features_b,
                    row_weights: e.row_weights,
                    butter_diff: e.butter_diff,
                });
        // EXP-V11-D-PJND-DOMINANT — GPU needs per-row targets (no global
        // fallback on the GPU side). Materialize from
        // --pjnd-passthrough-target-score when pjnd_anchor_loaded
        // ships no `target_scores` slice.
        let gpu_pjnd_targets_owned: Option<Vec<f64>> = if let Some(pa) = pjnd_anchor_loaded.as_ref()
        {
            if pa.target_scores.is_some() {
                None
            } else {
                Some(vec![args.pjnd_passthrough_target_score; pa.features.len()])
            }
        } else {
            None
        };
        let gpu_pjnd_anchor: Option<zensim_train_gpu::GpuAnchorRows<'_>> = pjnd_anchor_loaded
            .as_ref()
            .map(|pa| zensim_train_gpu::GpuAnchorRows {
                name: pa.name.clone(),
                features: pa.features,
                row_weights: pa.row_weights,
                target_scores: pa
                    .target_scores
                    .unwrap_or(gpu_pjnd_targets_owned.as_deref().unwrap_or(&[])),
            });
        let res = zensim_train_gpu::train_per_sample_alpha_head_gpu_with_aux_pjnd(
            &core_groups,
            &gpu_hp,
            n_features,
            runtime,
            gpu_anchor.as_ref(),
            gpu_equiv.as_ref(),
            gpu_pjnd_anchor.as_ref(),
        );
        log.push(format!(
            "GPU trainer: {} batches in {:.2} s ({:.1} batches/s)",
            res.n_batches,
            res.wall_seconds,
            res.n_batches as f64 / res.wall_seconds
        ));
        if gpu_hp.tanh_output_head_scale > 0.0 {
            zensim_train_core::per_sample_alpha_head::bake_per_sample_alpha_head_v3_with_tanh_and_transforms(
                &res.model,
                gpu_hp.tanh_output_head_scale,
                feature_transforms.as_deref(),
                feature_transform_params.as_deref(),
                None, // spline fitted by the Rust trainer's post-training step
            )
        } else {
            zensim_train_core::per_sample_alpha_head::bake_per_sample_alpha_head_v3(&res.model)
        }
    } else {
        // STRATEGY-2026-07-02: optional raw-human-triplet pool.
        let triplet_pool: Option<TripletPool> =
            match (&args.triplet_stimuli, &args.triplet_responses) {
                (Some(stim), Some(resp)) if args.triplet_weight > 0.0 => {
                    let mut pool = TripletPool::default();
                    let mut rdr = csv::Reader::from_path(stim)
                        .unwrap_or_else(|e| panic!("triplet stimuli {stim}: {e}"));
                    let headers = rdr.headers().expect("stimuli headers").clone();
                    let f0 = headers
                        .iter()
                        .position(|h| h == "f0")
                        .expect("stimuli CSV needs f0..fN columns");
                    for rec in rdr.records() {
                        let rec = rec.expect("stimuli row");
                        let feats: Vec<f64> = (f0..headers.len())
                            .map(|i| rec.get(i).and_then(|v| v.parse().ok()).unwrap_or(0.0))
                            .collect();
                        pool.features.push(feats);
                    }
                    for (ln, line) in std::fs::read_to_string(resp)
                        .unwrap_or_else(|e| panic!("triplet responses {resp}: {e}"))
                        .lines()
                        .enumerate()
                    {
                        if line.is_empty() || line.starts_with('#') {
                            continue;
                        }
                        let mut it = line.split('\t');
                        let (l, r, a) = (
                            it.next().and_then(|v| v.parse::<u32>().ok()),
                            it.next().and_then(|v| v.parse::<u32>().ok()),
                            it.next().and_then(|v| v.parse::<u8>().ok()),
                        );
                        match (l, r, a) {
                            (Some(l), Some(r), Some(a))
                                if (l as usize) < pool.features.len()
                                    && (r as usize) < pool.features.len()
                                    && a <= 2 =>
                            {
                                pool.responses.push((l, r, a));
                            }
                            _ => panic!(
                                "triplet responses line {}: malformed or out of range",
                                ln + 1
                            ),
                        }
                    }
                    eprintln!(
                        "triplet pool: {} stimuli x {} features, {} responses",
                        pool.features.len(),
                        pool.features.first().map(|f| f.len()).unwrap_or(0),
                        pool.responses.len()
                    );
                    Some(pool)
                }
                _ => None,
            };
        train_mlp_strategy(
            &mut groups,
            n_features,
            &hyperparams,
            &mut log,
            tv_regularizer.as_ref(),
            anchor_loaded.as_ref(),
            equiv_loaded.as_ref(),
            pjnd_anchor_loaded.as_ref(),
            konjnd_agg_loaded.as_ref(),
            triplet_pool.as_ref(),
        )
    };

    // Honest live width of what we just trained: how many layer-1 input rows
    // are EXACTLY zero. `--keep-features` pins them; `--group-l1` learns them
    // away. Exact zeros are what `bake_dial_refit pack` prunes, so this is
    // the number that turns into a narrower, faster shipping bake.
    let live_l0_rows: Option<usize> = match count_live_l0_rows(&bake_bytes) {
        Some((live, dead)) => {
            if keep_mask.is_some() || args.group_l1 > 0.0 {
                println!(
                    "[live-width] layer-1 inputs: {live} live, {dead} exactly-zero \
                     (prunable by `bake_dial_refit pack`)"
                );
            }
            Some(live)
        }
        None => None,
    };

    // ── MANDATORY reproduction provenance ──────────────────────────────
    // Assembled BEFORE baking, embedded INTO the bake bytes as the
    // `zentrain.repro` metadata entry at the single write choke-point below
    // (covers every arch/GPU/CPU path), and merged into the .spec.json
    // sidecar. The embedded copy is the one that matters: a bake copied
    // without its sidecar keeps its reproduction instructions. Not optional,
    // no flag: an unreproducible bake is a defect (2026-07-27 user directive).
    let repro_json: String = {
        // Trainer source identity: HEAD of the source dir at TRAIN time
        // (honest label — the binary may have been built at an older commit).
        let src_dir = env!("CARGO_MANIFEST_DIR");
        // git first (colocated primary checkout); jj fallback for secondary
        // jj workspaces, which carry .jj but no .git.
        let git_head = std::process::Command::new("git")
            .args(["-C", src_dir, "rev-parse", "--short=12", "HEAD"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .or_else(|| {
                // cwd-based: jj searches upward from the crate dir to the
                // workspace root (--repository would need the root itself).
                std::process::Command::new("jj")
                    .current_dir(src_dir)
                    .args(["log", "--no-graph", "-r", "@-", "-T", "commit_id.short(12)"])
                    .output()
                    .ok()
                    .filter(|o| o.status.success())
                    .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            })
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown".into());
        let inputs: Vec<serde_json::Value> = loaded
            .iter()
            .map(|g| {
                serde_json::json!({
                    "name": g.name, "path": g.source_path, "sha256": g.source_sha256,
                    "rows": g.human_scores.len(), "n_features": g.n_features,
                    "train_w": g.train_w, "val_w": g.val_w,
                    "loss_mode": format!("{:?}", g.loss_mode), "within_ref": g.within_ref,
                })
            })
            .collect();
        serde_json::json!({
            "schema": 1,
            "tool": "zensim_mlp_train",
            // argv reproduces every hyperparameter + transform verbatim.
            "argv": std::env::args().collect::<Vec<String>>(),
            "cwd": std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_default(),
            // Structured duplicates of the load-bearing knobs (greppable
            // without argv parsing):
            "seed": args.seed,
            "epochs": args.epochs,
            "pairs_per_epoch": args.pairs_per_epoch,
            "n_hidden_layers": args.n_hidden_layers,
            "target_column": args.target_column,
            "target_scale": args.target_scale,
            "max_features": args.max_features,
            "keep_features_n": keep_mask.as_ref().map(|m| m.iter().filter(|&&k| k).count()),
            "group_l1": args.group_l1,
            "algorithm": "RankNet pairwise (+per-group loss modes; see argv for full hyperparams)",
            // Content-addressed inputs: the canonical reproduction identity.
            "inputs": inputs,
            "best_val": *zensim_validate::mlp_train::LAST_BEST_VAL.lock().unwrap(),
            "trainer_source_dir": src_dir,
            "trainer_head_at_train": git_head,
            "hostname": std::env::var("HOSTNAME").unwrap_or_default(),
            "timestamp_epoch": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        })
        .to_string()
    };

    // MANDATORY: embed reproduction provenance into the bake bytes themselves.
    // append_metadata_utf8 is zenpredict-bake's section-level splice with
    // score/byte identity guarantees (weights untouched — no requantization).
    // Failure here is FATAL by design: shipping an unreproducible bake is a
    // defect, not a degraded mode.
    let bake_bytes =
        zenpredict_bake::append_metadata_utf8(&bake_bytes, "zentrain.repro", &repro_json)
            .unwrap_or_else(|e| {
                eprintln!("FATAL: could not embed zentrain.repro into the bake: {e:?}");
                std::process::exit(4);
            });
    std::fs::write(&out_path, &bake_bytes).unwrap_or_else(|e| {
        eprintln!("write {out_path:?}: {e}");
        std::process::exit(1);
    });
    println!("Wrote {} bytes to {out_path:?}", bake_bytes.len());

    // Provenance sidecar `<bake>.spec.json` — so downstream tooling (bandwise
    // dashboard honesty matrix, bake_verdict train-vs-heldout labeling) never has
    // to GUESS which corpora a bake trained on. Derived from the ACTUAL train_w>0
    // groups (the desync-proof source of truth); a missing sidecar is what renders
    // as "unknown". Raw group names are emitted (kadid / tid / konjnd_dense /
    // bigcodec …) — the dashboard's prefix-tolerant `_trained()` resolves them to
    // the CHEAT/val-split twins. 2026-07-18.
    {
        let mut train_corpora: Vec<String> = loaded
            .iter()
            .filter(|g| g.train_w > 0.0)
            .map(|g| g.name.clone())
            .collect();
        train_corpora.sort();
        train_corpora.dedup();
        // Full group provenance (name/weights/loss/within-ref) so the recipe reproduces.
        let groups: Vec<serde_json::Value> = loaded
            .iter()
            .map(|g| {
                serde_json::json!({
                    "name": g.name, "train_w": g.train_w, "val_w": g.val_w,
                    "loss_mode": format!("{:?}", g.loss_mode), "within_ref": g.within_ref,
                })
            })
            .collect();
        // argv is the EXACT reproduction command (captures every hyperparam + feature-transform
        // verbatim — the "how was this trained" question answered without archaeology).
        let spec = serde_json::json!({
            "train_corpora": train_corpora,
            "groups": groups,
            // Content-addressed input identity (path + sha256 + rows) — same
            // block as the bake-embedded zentrain.repro, duplicated here so
            // the sidecar alone still reproduces (and legacy tooling that
            // reads only spec.json gets the full story).
            "inputs": loaded.iter().map(|g| serde_json::json!({
                "name": g.name, "path": g.source_path, "sha256": g.source_sha256,
                "rows": g.human_scores.len(),
            })).collect::<Vec<_>>(),
            "seed": args.seed,
            "keep_features_n": keep_mask.as_ref().map(|m| m.iter().filter(|&&k| k).count()),
            "group_l1": args.group_l1,
            "live_l0_rows": live_l0_rows,
            "best_val": *zensim_validate::mlp_train::LAST_BEST_VAL.lock().unwrap(),
            "repro_embedded": true,
            "argv": std::env::args().collect::<Vec<String>>(),
            "cwd": std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_default(),
            "timestamp_epoch": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            "note": "auto-emitted by zensim_mlp_train; argv is the exact reproduction command",
        });
        let mut spec_os = out_path.clone().into_os_string();
        spec_os.push(".spec.json");
        let spec_path = PathBuf::from(spec_os);
        match std::fs::write(
            &spec_path,
            serde_json::to_string_pretty(&spec).unwrap_or_default(),
        ) {
            Ok(()) => println!(
                "[spec] wrote provenance sidecar {spec_path:?} ({} train corpora, argv captured)",
                train_corpora.len()
            ),
            Err(e) => eprintln!("[spec] warning: could not write {spec_path:?}: {e}"),
        }
    }

    // REPRODUCE-EXACTLY, the output half. `verify_inputs` has always checked
    // that the manifest's INPUTS are the bytes it claims; nothing checked its
    // OUTPUT until 2026-07-15, and the parser did not even read
    // `[bake].sha256` (RawBake carried only `file`, so serde dropped it).
    //
    // The cost of that asymmetry, measured: 128 of 142 manifests recorded
    // `d0ef7a30…` — the shipped Profile A bake — because each new experiment
    // was forked from `v47_strict_qat.toml` and never updated the outcome
    // fields. Exactly one of them described that bake. This check is what
    // would have caught all 128 AT CREATION rather than two months later:
    // fork a manifest, train, and it says "you claimed X, you produced Y".
    //
    // Silent when the manifest makes no claim — a recipe (no `[bake].sha256`)
    // describes how to train, not one bake. Only a claim gets checked.
    if let Some((claimed, manifest_path)) = &manifest_claimed_sha {
        match train_manifest::sha256_file(&out_path) {
            Ok(actual) if &actual == claimed => {
                println!(
                    "[manifest] REPRODUCED: bake sha256 matches {}'s [bake].sha256 ({})",
                    manifest_path.display(),
                    &actual[..16]
                );
            }
            Ok(actual) if args.manifest_allow_sha_drift => {
                // Don't tell someone to pass the flag they just passed.
                eprintln!(
                    "[manifest] WARNING: bake sha256 drift ALLOWED via \
                     --manifest-allow-sha-drift: {} claims {claimed}, this run produced \
                     {actual}. The bake on disk is NOT the one that manifest describes — \
                     do not cite the manifest's [eval] for it.",
                    manifest_path.display()
                );
            }
            Ok(actual) => {
                eprintln!(
                    "[manifest] bake sha256 MISMATCH\n  \
                     manifest: {}\n  \
                     claimed:  {claimed}\n  \
                     produced: {actual}\n\
                     The manifest describes a DIFFERENT bake than this run produced. Either:\n  \
                     (a) you forked this manifest for a new experiment and kept the parent's \
                     outcome fields — recompute [bake].sha256/file_bytes/[eval] from YOUR bake, \
                     or drop them (a recipe makes no claim); or\n  \
                     (b) this was meant to reproduce the recorded bake and did not — check \
                     trainer_commit, the input shas, and the seed before trusting either artifact.\n\
                     Pass --manifest-allow-sha-drift to downgrade this to a warning.",
                    manifest_path.display()
                );
                std::process::exit(3);
            }
            Err(e) => eprintln!("[manifest] cannot hash produced bake to verify: {e}"),
        }
    }

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

    // Auto-evaluate: run bake_verdict on the output bake after every training run.
    // Produces the full Mohammadi panel (SROCC+PLCC+KROCC+OR+PWRC+Z-RMSE)
    // on all held-out corpora so every bake gets an honest verdict.
    let self_exe = std::env::current_exe().ok();
    let verdict_bin = self_exe
        .as_ref()
        .and_then(|p| p.parent())
        .map(|dir| dir.join("bake_verdict"));
    if let Some(ref vb) = verdict_bin {
        if vb.exists() {
            println!("\n--- bake_verdict (auto-eval) ---");
            let mut cmd = std::process::Command::new(vb);
            cmd.arg("--bake").arg(&out_path);
            // Regime must FOLLOW the bake's input width — a 944-class bake
            // verdicted at the default 372 root silently mis-scores (the
            // ebothg_m504 wrong-root class; zensim CLAUDE.md Known Bugs).
            // Found live 2026-08-27: the HDR-944 L1 bakes' auto-verdicts ran
            // regime-372. Known widths map to their regime; anything else
            // keeps the default and says so.
            match args.max_features {
                720 => { cmd.arg("--regime").arg("720"); }
                944 => { cmd.arg("--regime").arg("944"); }
                372 => {}
                w => eprintln!(
                    "bake_verdict auto-eval: width {w} has no registered --regime; using the 372 default (verify the root)"
                ),
            }
            // Always write verdict file alongside the bake
            let verdict_path = out_path.with_extension("verdict.md");
            cmd.arg("--output").arg(&verdict_path);
            // Inherit stdout/stderr so user sees the eval live
            cmd.stdout(std::process::Stdio::inherit());
            cmd.stderr(std::process::Stdio::inherit());
            match cmd.status() {
                Ok(s) if s.success() => {
                    println!("Verdict written to {:?}", verdict_path);
                }
                Ok(s) => eprintln!("bake_verdict exited with {s}"),
                Err(e) => eprintln!("bake_verdict failed to run: {e}"),
            }
        } else {
            eprintln!(
                "bake_verdict not found at {:?} — build with: \
                 cargo build --release --bin bake_verdict -p zensim-validate",
                vb
            );
        }
    }
}

#[cfg(test)]
mod csv_load_equivalence_tests {
    //! Equivalence tests for `load_csv` (parallel/mmap/fast-float)
    //! vs `load_csv_sequential` (BufReader + stdlib `f64::from_str`).
    //!
    //! Verifies bit-identical output on a real training-corpus CSV
    //! (the smaller TID file at /mnt/v with 3000 rows × 372 features).
    //! Skipped at compile-time on environments without that path.
    use super::{LoadedGroup, load_csv, load_csv_sequential};
    use std::path::PathBuf;

    fn tid_csv() -> Option<PathBuf> {
        let p = PathBuf::from(
            "/mnt/v/zen/zensim-training/2026-05-16/v2/tid_features_iwssim_log_372col.csv",
        );
        if p.exists() { Some(p) } else { None }
    }

    fn assert_groups_eq(par: &LoadedGroup, seq: &LoadedGroup) {
        assert_eq!(par.name, seq.name, "name");
        assert_eq!(par.n_features, seq.n_features, "n_features");
        assert_eq!(
            par.human_scores.len(),
            seq.human_scores.len(),
            "human_scores len: parallel={} sequential={}",
            par.human_scores.len(),
            seq.human_scores.len()
        );
        assert_eq!(
            par.feature_rows.len(),
            seq.feature_rows.len(),
            "feature_rows len (flat)"
        );

        // Bit-identical scalar comparison. fast_float2 advertises
        // IEEE-754 round-to-nearest correctness on every input
        // that stdlib f64::from_str handles, so these should be
        // exact equals across the corpus.
        for (i, (a, b)) in par
            .human_scores
            .iter()
            .zip(seq.human_scores.iter())
            .enumerate()
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "human_score[{}] differs: parallel={} sequential={}",
                i,
                a,
                b
            );
        }
        // feature_rows is one flat row-major buffer per group; compare
        // element-wise (index / n_features recovers the row for messages).
        let nf = par.n_features.max(1);
        for (idx, (a, b)) in par
            .feature_rows
            .iter()
            .zip(seq.feature_rows.iter())
            .enumerate()
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "feature_row[{}][{}] differs: parallel={} sequential={}",
                idx / nf,
                idx % nf,
                a,
                b
            );
        }
    }

    #[test]
    fn parallel_matches_sequential_iwssim_log_target() {
        let path = match tid_csv() {
            Some(p) => p,
            None => {
                eprintln!(
                    "SKIP: TID iwssim CSV not present (run on a workstation with /mnt/v mounted)"
                );
                return;
            }
        };
        let par = load_csv(&path, "tid", "iwssim_log_norm", 1.0).expect("parallel loader");
        let seq =
            load_csv_sequential(&path, "tid", "iwssim_log_norm", 1.0).expect("sequential loader");
        assert_groups_eq(&par, &seq);
    }

    #[test]
    fn parallel_matches_sequential_default_target_with_scale() {
        // Exercises the `target_scale` multiplier path and a different
        // target column (`human_score` is in [0, 1] in this corpus and
        // typically multiplied by 100.0). fast-float and stdlib should
        // still agree bit-for-bit.
        let path = match tid_csv() {
            Some(p) => p,
            None => {
                eprintln!("SKIP: TID iwssim CSV not present");
                return;
            }
        };
        // The 2026-05-16 v2 CSV's first metric column is iwssim_log_norm,
        // but the file also carries a `human_score` column (per the
        // training-CSV format documented at the top of this binary).
        // If absent, the loader will error and the test fails loudly.
        let target = "human_score";
        let par_res = load_csv(&path, "tid", target, 100.0);
        let seq_res = load_csv_sequential(&path, "tid", target, 100.0);
        match (par_res, seq_res) {
            (Ok(par), Ok(seq)) => assert_groups_eq(&par, &seq),
            (Err(e1), Err(e2)) => assert_eq!(e1, e2, "errors must match"),
            (Ok(_), Err(e)) => panic!("parallel succeeded but sequential errored: {e}"),
            (Err(e), Ok(_)) => panic!("sequential succeeded but parallel errored: {e}"),
        }
    }
}
