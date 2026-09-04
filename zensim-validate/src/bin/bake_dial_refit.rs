//! `bake_dial_refit` — Rust migration of the ZNPR bake output-spline /
//! anchor-refit family (retires `scripts/v_next/{dense_dial_refit_b,
//! shared_anchor_refit, bhdr_bottom_extend, winsorize_bake,
//! bake_outlier_gate}.py`).
//!
//! Only the genuinely-missing capability lives here — the anchor-driven
//! spline-REFIT math + the G-RANGE tail gate. Everything a Rust crate
//! already owns is reused, never re-implemented:
//!
//! * bake read  → [`zenpredict::Model`] (`scaler_mean`, `layer`, `metadata`)
//! * bake emit  → [`zenpredict_bake::bake`] (THE canonical v3 serializer)
//! * PCHIP eval → [`zensim_validate::output_calibration_spline`]
//! * Z-RMSE/OR/SROCC → [`zensim_validate::panel`] (= `zenstats::panel`)
//!
//! Full non-duplication analysis + byte-parity de-risk:
//! `benchmarks/bake_refit_rust_migration_2026-07-05.md`.
//!
//! ## Subcommands
//!
//! * `extend-top`  — reproduces `dense_dial_refit_b.py` (the SHIPPED-B
//!   producer): extend ONLY the output spline's TOP above its top knot by
//!   the concave saturation `score(r)=100−(100−y0)·e^{−k(r−x0)}`, with `k`
//!   from a robust log-OLS `log(100−t) ≈ logA − k·raw` on the anchor's
//!   `target > band_min` rows. Bottom + in-distribution knots and all
//!   weights/scaler/transform metadata are kept VERBATIM ⇒ rank-invariant.
//! * `shared-anchor` — reproduces `shared_anchor_refit.py`: refit the WHOLE
//!   output spline to an anchor (percentile-edge bins + per-bin median +
//!   monotone filter + neg-tail dedup — faithful to `lp.fit_spline_knots`;
//!   note the crate's existing `fit_monotone_spline` uses *equal-count*
//!   bins, a different strategy).
//! * `bottom-extend` — reproduces `bhdr_bottom_extend.py`: prepend a
//!   `(floor_raw, 0.0)` bottom knot to cover the raw floor (rank-invariant).
//! * `add-winsor` — reproduces `winsorize_bake.py`: add 372 `winsor_p99`
//!   feature-transform guards computed as per-feature `[p_lo, p_hi]` on a
//!   fit corpus (identity within bounds ⇒ rank-invariant on in-distribution
//!   rows, bounds the extrapolating tail).
//! * `gate` — the HARD **G-RANGE** gate (fraction of raw preds outside the
//!   spline knot domain — the tail detector SROCC is blind to) + advisory
//!   Z-RMSE / outlier-ratio / SROCC vs a reference column. Originally a
//!   linear-only port of `bake_outlier_gate.py`; since 2026-08-04 the
//!   forward routes through the shared `bake_runtime` production dispatch,
//!   so it is evaluable for EVERY bake class (MLPs, α/hybrid heads,
//!   min-max) — closing the campaign-long "G-RANGE NOT EVALUABLE" gap on
//!   MLP candidates. Computes NO PWRC (the `zenstats` `sa_st_curve` O(n²)
//!   allocation OOMs broad corpora — see the module's PWRC note).
//! * `pack` — reproduces `pack_and_calibrate.py` (the STANDARD non-QAT
//!   packing path): per-layer zerobias (`--protect-last` exempts the final
//!   layer, keeping it f32) + dtype quantization, then refit the output
//!   spline ON THE PACKED network. The pack-THEN-calibrate order is
//!   load-bearing: quantization preserves rank but shifts raw outputs, so
//!   a spline fit on the f32 net maps the packed net's identity to the
//!   wrong dial value (identity drops 97.8→93.4 observed).
//! * `strip` — reproduces `strip_spline_metadata.py`: drop one metadata
//!   entry (default the output-calibration spline), everything else
//!   re-emitted verbatim.
//! * `fit-lasso` — reproduces `linear_projections_2026-07-03.py`'s
//!   `MixGram.lasso` + `bake_candidate` (the SHIPPED-BHdr producer, task
//!   #68): lasso coordinate descent on a FROZEN feature-Gram `.npz`, f16
//!   pack, dial spline fit on the PACKED forward over the anchor `.npz`,
//!   one bake out. Unlike the other subcommands this CREATES a bake instead
//!   of editing one; it lives here (rather than a new bin) because this bin
//!   is the Rust home of the `scripts/v_next` bake-family ports and the
//!   plumbing it must share — `emit_linear`, `fit_spline_knots`,
//!   `spline_payload`, `SPLINE_KEY` — is this bin's (a second bin would
//!   duplicate `emit_linear`, violating the one-owner rule). Fit math lives
//!   in [`zensim_validate::gram_lasso`], npz reading in
//!   [`zensim_validate::npz`]. Accepts MULTIPLE `--gram`+`--weight` pairs
//!   (MixGram multi-group accumulation) and a `--anchor-parquet`
//!   alternative to the frozen npz anchor (E-LIN linear-924 campaign).
//! * `gram` — builds a per-corpus raw-moment Gram npz (`S`, `s`, `q_<t>`,
//!   `Y1_<t>`, `n`) from a feature parquet via
//!   `parquet_loader::stream_parquet_rows` (memory-capped, f64,
//!   deterministic file order) — the artifact `fit-lasso` consumes. One
//!   corpus per invocation; arms combine grams with `--weight`s.
//!   Pre-registration: `benchmarks/linear924_phase1_2026-08-01.md`.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use arrow::array::{Array, Float32Array, Float64Array};
use arrow::record_batch::RecordBatch;
use clap::{Args, Parser, Subcommand};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use zenpredict::{Activation, MetadataType, Model, WeightDtype, WeightStorage, f16_bits_to_f32};
use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake};
use zensim_validate::block_profile;
use zensim_validate::output_calibration_spline as spline;
use zensim_validate::panel::{outlier_ratio, rescale_logistic, spearman, z_rmse};
use zensim_validate::prune;
// Dial-spline fitting lives in the shared `dial_spline` module (2026-07-16) so
// this linear tool AND the min-max bake path fit the [0,100] dial identically.
use zensim_validate::dial_spline::{fit_spline_knots, percentile_linear, spline_payload};

/// Metadata key for the PCHIP output-calibration spline payload. Matches
/// the private `KEY` in `output_calibration_spline` (which does not
/// re-export it) and `zenpredict::keys::FEATURE_TRANSFORM*`.
const SPLINE_KEY: &str = "zentrain.output_calibration_spline";
use zenpredict::keys::{
    FEATURE_TRANSFORM_PARAMS as FEATURE_TRANSFORM_PARAMS_KEY,
    FEATURE_TRANSFORMS as FEATURE_TRANSFORMS_KEY,
};

#[derive(Parser)]
#[command(
    name = "bake_dial_refit",
    about = "Refit a ZNPR bake's output-calibration spline / feature-winsor guard (Rust port of the scripts/v_next refit family)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Fit + inject an output-calibration spline for an ARBITRARY spline-less
    /// bake (incl. MLPs) — the generic sibling of `shared-anchor`, which is
    /// linear-only. Forwards through `zenpredict::Predictor` (transform-safe
    /// `predict_transformed` dispatch), fits the shared `fit_spline_knots`,
    /// re-emits every layer VERBATIM (f32/f16 dtypes preserved) with only the
    /// spline metadata entry added. Rank-invariant by construction.
    /// Added 2026-07-18 to give the Ebothg winner MLP a dial (scorecard P1).
    AddSpline(AddSplineArgs),
    /// Extend ONLY the output spline's top by a training-fitted concave
    /// saturation (reproduces `dense_dial_refit_b.py`).
    ExtendTop(ExtendTopArgs),
    /// Refit the WHOLE output spline to an anchor (reproduces
    /// `shared_anchor_refit.py`).
    SharedAnchor(SharedAnchorArgs),
    /// Prepend a `(floor_raw, 0.0)` bottom knot (reproduces
    /// `bhdr_bottom_extend.py`).
    BottomExtend(BottomExtendArgs),
    /// Add per-feature `winsor_p99` transform guards from a fit corpus
    /// (reproduces `winsorize_bake.py`).
    AddWinsor(AddWinsorArgs),
    /// G-RANGE tail gate + advisory Z-RMSE/OR/SROCC (reproduces
    /// `bake_outlier_gate.py`).
    Gate(GateArgs),
    /// Per-layer zerobias + dtype pack, THEN spline refit on the packed
    /// network (reproduces `pack_and_calibrate.py`, the STANDARD non-QAT
    /// packing path).
    Pack(PackArgs),
    /// Drop a metadata entry (default: the output-calibration spline) and
    /// re-emit everything else verbatim (reproduces
    /// `strip_spline_metadata.py`).
    Strip(StripArgs),
    /// Append ONE metadata entry via zenpredict-bake's section splice
    /// (weights byte-untouched; strip's inverse). Added 2026-08-28 to stamp
    /// H-TRAJ checkpoint dumps with `zentrain.repro` post-hoc.
    AppendMeta(AppendMetaArgs),
    /// Lasso-CD fit on a frozen feature-Gram npz, f16 pack, anchor spline,
    /// and bake — the Rust-native BHdr fit chain (reproduces
    /// `linear_projections_2026-07-03.py` `fit` plus `finalize` for one
    /// gram/lambda, bit-exactly).
    FitLasso(Box<FitLassoArgs>),
    /// Forward an arbitrary bake (incl. MLPs) over a feature parquet through
    /// the PRODUCTION predictor (transform-safe) and write `ref_basename\tpred`
    /// TSV — the teacher-labeling / preds-dump owner (SOTA-944 amendment 3
    /// distillation arm; reuses add-spline's forward machinery).
    Predict(PredictArgs),
    /// Blend two fit-npz heads (from `fit-lasso --emit-fit-npz`) in raw
    /// output space with per-head z-normalization over the anchor rows —
    /// the Profile-B multi-head mechanism (SOTA-944 §4). Collapses to ONE
    /// identity-scaler linear layer (B's shipped scaler shape), f16-packs,
    /// fits the shared spline on the packed forward, emits canonically.
    BlendHeads(BlendHeadsArgs),
    /// One-pass monotone-transform screen (SOTA-944 §3b): per feature ×
    /// candidate {identity, log1p, signed_cbrt}, |Pearson r(t(x), y)| on
    /// stride-sampled corpus rows; emits the winner-per-feature TSV
    /// (`feat_idx`/`best_transform`/`params_csv`) consumed by
    /// `gram --transforms-tsv` + `fit-lasso --transforms-tsv`.
    ScreenTransforms(ScreenTransformsArgs),
    /// REFIT the winsor windows of an EXISTING `--feature-transform` token
    /// list on current data, holding the token→feature ASSIGNMENT fixed
    /// (SOTA-944 wave 8, amendment 9 §9.1). Reuses this binary's
    /// `add-winsor` fit rule verbatim — `percentile_linear` at
    /// `[--lo-pct, --hi-pct]` (owner defaults 0.1 / 99.9) plus the same
    /// `lo == hi == 0 → hi = 1e-9` degenerate guard — over the POOLED rows
    /// of every `--parquet`. Emits the refit token list in the input's
    /// order (so a training argv diffs token-for-token) plus an audit TSV.
    RefitWinsor(RefitWinsorArgs),
    /// Build a per-corpus raw-moment feature Gram (`S = Σxxᵀ`, `s = Σx`,
    /// `q_t = Σx·y_t`, `Y1_t = Σy_t`, `n`) from a feature parquet, streamed
    /// through `parquet_loader::stream_parquet_rows` (memory-capped, f64,
    /// deterministic row order) into the `.npz` layout `fit-lasso` consumes.
    /// E-LIN linear-924 campaign (`benchmarks/linear924_phase1_2026-08-01.md`).
    Gram(GramArgs),
}

// --------------------------------------------------------------------------
// shared bake I/O
// --------------------------------------------------------------------------

/// A single-layer identity linear bake: 372→1 with an f16 weight layer.
/// Every refit target in the Python family is this shape (linear pick /
/// winsor / bhdr / raw). Multi-layer bakes are rejected loudly.
struct LinearBake {
    scaler_mean: Vec<f32>,
    scaler_scale: Vec<f32>,
    weights: Vec<f32>,
    bias: f32,
}

fn load_linear(model: &Model) -> LinearBake {
    assert_eq!(
        model.n_layers(),
        1,
        "bake_dial_refit expects a single-layer linear bake (got {} layers)",
        model.n_layers()
    );
    let l0 = model.layer(0);
    assert_eq!(l0.out_dim, 1, "expected out_dim=1 (got {})", l0.out_dim);
    let weights = dequant_out1(&l0.weights);
    LinearBake {
        scaler_mean: model.scaler_mean().to_vec(),
        scaler_scale: model.scaler_scale().to_vec(),
        weights,
        bias: l0.biases[0],
    }
}

/// Dequantize an `out_dim == 1` layer's weight storage to f32. f16 round-
/// trips exactly (`f16→f32→f16` is identity), so re-emitting as f16
/// reproduces the source bytes.
fn dequant_out1(w: &WeightStorage) -> Vec<f32> {
    match w {
        WeightStorage::F32(w) => w.to_vec(),
        WeightStorage::F16(w) => w.iter().map(|b| f16_bits_to_f32(*b)).collect(),
        WeightStorage::I8 { weights, scales } => {
            weights.iter().map(|q| *q as f32 * scales[0]).collect()
        }
    }
}

/// An owned metadata entry (so the bytes outlive the borrow into
/// [`BakeMetadataEntry`]).
struct OwnedMeta {
    key: String,
    kind: MetadataType,
    value: Vec<u8>,
}

/// Copy every metadata entry of `model` verbatim, in order.
fn clone_metadata(model: &Model) -> Vec<OwnedMeta> {
    model
        .metadata()
        .iter()
        .map(|e| OwnedMeta {
            key: e.key.to_string(),
            kind: e.kind,
            value: e.value.to_vec(),
        })
        .collect()
}

/// Emit a single-layer identity f16 linear bake through the canonical
/// serializer. `compressed: true` and the field set match what the Python
/// pipeline (`bake_from_json` → `bake`) produced, so a same-input run is
/// byte-reproducible.
fn emit_linear(
    out: &Path,
    scaler_mean: &[f32],
    scaler_scale: &[f32],
    weights: &[f32],
    bias: f32,
    metadata: &[OwnedMeta],
) -> std::io::Result<usize> {
    let biases = [bias];
    let layers = [BakeLayer {
        in_dim: scaler_mean.len(),
        out_dim: 1,
        activation: Activation::Identity,
        dtype: WeightDtype::F16,
        weights,
        biases: &biases,
    }];
    let md: Vec<BakeMetadataEntry<'_>> = metadata
        .iter()
        .map(|m| BakeMetadataEntry {
            key: &m.key,
            kind: m.kind,
            value: &m.value,
        })
        .collect();
    let bytes = bake(&BakeRequest {
        schema_hash: 0,
        flags: 0,
        scaler_mean,
        scaler_scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: &md,
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: true,
        hu_permutations: None,
    })
    .expect("serialize ZNPR v3 bake");
    std::fs::write(out, &bytes)?;
    Ok(bytes.len())
}

/// Read the parsed spline knots (`x`, `y`, both f64 widened from the f32
/// wire) from a bake. Reuses the crate's shared parser.
fn read_spline(model: &Model) -> (Vec<f64>, Vec<f64>) {
    let s = spline::extract(model).expect("bake has no valid output_calibration_spline");
    (s.xs, s.ys)
}

// --------------------------------------------------------------------------
// f64 calibration forward (winsor / identity)
// --------------------------------------------------------------------------

/// Per-feature forward op. The fit-side forward runs in **f64** (the
/// runtime `Predictor` is f32); the Python fit did too, so this reproduces
/// its `k` — and hence the knots — byte-exactly.
enum FwOp {
    Identity,
    /// `np.clip(x, lo, hi)` semantics — matches `FeatureTransform::WinsorP99`.
    Winsor(f64, f64),
    /// `sign(x) * |x|^(1/3)` — matches `FeatureTransform::SignedCbrt` (the
    /// shaped-944 W-LIN class; added 2026-08-29 so the dial pass covers it).
    SignedCbrt,
    /// `ln(1 + x)` — matches `FeatureTransform::Log1p`.
    Log1p,
}

impl FwOp {
    #[inline]
    fn apply(&self, x: f64) -> f64 {
        match *self {
            FwOp::Identity => x,
            FwOp::Winsor(lo, hi) => {
                if x < lo {
                    lo
                } else if x > hi {
                    hi
                } else {
                    x
                }
            }
            FwOp::SignedCbrt => {
                let s = if x >= 0.0 { 1.0 } else { -1.0 };
                s * x.abs().cbrt()
            }
            FwOp::Log1p => x.ln_1p(),
        }
    }
}

/// Build the per-feature forward ops from the bake's transform metadata.
/// Absent metadata ⇒ all-identity (the baker omits the entry when every
/// transform is identity). `identity` / `winsor_p99` / `signed_cbrt` /
/// `log1p` are supported in the f64 fit-forward (the last two added
/// 2026-08-29 for the shaped-944 W-LIN class); any other token errors
/// (the yeo-johnson HDR path stays in the research Python).
fn build_fw_ops(model: &Model, n: usize) -> Result<Vec<FwOp>, String> {
    let md = model.metadata();
    let Some(t) = md.get(zenpredict::keys::FEATURE_TRANSFORMS) else {
        return Ok((0..n).map(|_| FwOp::Identity).collect());
    };
    let tokens: Vec<&str> = std::str::from_utf8(t.value)
        .map_err(|e| format!("feature_transforms not UTF-8: {e}"))?
        .split('\n')
        .collect();
    let params_txt = md
        .get(zenpredict::keys::FEATURE_TRANSFORM_PARAMS)
        .map(|p| std::str::from_utf8(p.value).unwrap_or(""))
        .unwrap_or("");
    let params: Vec<&str> = params_txt.split('\n').collect();
    if tokens.len() != n {
        return Err(format!(
            "feature_transforms has {} tokens, expected {n}",
            tokens.len()
        ));
    }
    let mut ops = Vec::with_capacity(n);
    for (i, tok) in tokens.iter().enumerate() {
        match tok.trim() {
            "identity" | "" => ops.push(FwOp::Identity),
            "winsor_p99" => {
                let row = params.get(i).copied().unwrap_or("");
                let mut it = row.split(',');
                let lo: f64 = it
                    .next()
                    .and_then(|s| s.trim().parse().ok())
                    .ok_or_else(|| {
                        format!("winsor_p99 feature {i}: missing lo param in {row:?}")
                    })?;
                let hi: f64 = it
                    .next()
                    .and_then(|s| s.trim().parse().ok())
                    .ok_or_else(|| {
                        format!("winsor_p99 feature {i}: missing hi param in {row:?}")
                    })?;
                ops.push(FwOp::Winsor(lo, hi));
            }
            "signed_cbrt" => ops.push(FwOp::SignedCbrt),
            "log1p" => ops.push(FwOp::Log1p),
            other => {
                return Err(format!(
                    "f64 fit-forward supports identity/winsor_p99/signed_cbrt/log1p; feature {i} \
                     has {other:?} (yeo-johnson-class HDR bakes stay in the research Python)"
                ));
            }
        }
    }
    Ok(ops)
}

/// Raw (pre-spline) forward of one feature row in f64:
/// `bias + Σ_j ((op_j(x_j) − μ_j) / σ_j) · w_j`. Naive left-to-right
/// accumulation — verified (see migration doc) to reproduce the shipped
/// bake's `k` byte-exactly.
fn forward_raw(row: &[f64], ops: &[FwOp], lin: &LinearBake) -> f64 {
    let mut acc = lin.bias as f64;
    for j in 0..row.len() {
        let x = ops[j].apply(row[j]);
        let z = (x - lin.scaler_mean[j] as f64) / lin.scaler_scale[j] as f64;
        acc += z * lin.weights[j] as f64;
    }
    acc
}

// --------------------------------------------------------------------------
// parquet feature reader
// --------------------------------------------------------------------------

/// Read column `idx` of `batch` as f64 regardless of f32/f64 storage
/// (nulls → NaN). Mirrors `rescore_parquet::col_f64`.
fn col_f64(batch: &RecordBatch, idx: usize) -> Vec<f64> {
    let c = batch.column(idx);
    if let Some(a) = c.as_any().downcast_ref::<Float64Array>() {
        (0..a.len())
            .map(|i| if a.is_null(i) { f64::NAN } else { a.value(i) })
            .collect()
    } else if let Some(a) = c.as_any().downcast_ref::<Float32Array>() {
        (0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    f64::NAN
                } else {
                    a.value(i) as f64
                }
            })
            .collect()
    } else {
        vec![f64::NAN; batch.num_rows()]
    }
}

/// Read `<prefix>0..<prefix>{n_feat-1}` + `target_col` from a parquet into
/// row-major feature rows + the target vector.
fn read_features(
    path: &Path,
    prefix: &str,
    n_feat: usize,
    target_col: &str,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let file = File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("parquet reader");
    let schema = builder.schema().clone();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    let feat_idx: Vec<usize> = (0..n_feat)
        .map(|i| {
            let want = format!("{prefix}{i}");
            names
                .iter()
                .position(|&x| x == want)
                .unwrap_or_else(|| panic!("feature column {want} not found in {path:?}"))
        })
        .collect();
    let tgt_idx = names
        .iter()
        .position(|&x| x == target_col)
        .unwrap_or_else(|| panic!("target column {target_col} not found in {path:?}"));

    let reader = builder.build().expect("batch reader");
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut tgt: Vec<f64> = Vec::new();
    for batch in reader {
        let batch = batch.expect("read batch");
        let nrows = batch.num_rows();
        let cols: Vec<Vec<f64>> = feat_idx.iter().map(|&i| col_f64(&batch, i)).collect();
        let tcol = col_f64(&batch, tgt_idx);
        for r in 0..nrows {
            rows.push(cols.iter().map(|c| c[r]).collect());
            tgt.push(tcol[r]);
        }
    }
    (rows, tgt)
}

// --------------------------------------------------------------------------
// small numeric helpers (percentile / median / OLS)
// --------------------------------------------------------------------------

/// Ordinary least squares `y ≈ a + b·x` via the closed-form normal
/// equations (2×2). Verified equivalent to numpy `lstsq` to 1e-15 for the
/// well-conditioned `[1, pred]` design (see migration doc).
fn ols(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxx: f64 = x.iter().map(|v| v * v).sum();
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| a * b).sum();
    let b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let a = (sy - b * sx) / n;
    (a, b)
}

// --------------------------------------------------------------------------
// subcommand: extend-top  (reproduces dense_dial_refit_b.py)
// --------------------------------------------------------------------------

#[derive(Args)]
struct ExtendTopArgs {
    /// Winsor bake whose spline TOP is extended (bottom + in-distribution
    /// knots + weights/scaler/transforms kept verbatim).
    #[arg(long = "in")]
    input: PathBuf,
    /// Output bake path.
    #[arg(long)]
    out: PathBuf,
    /// Multiband anchor parquet (`f0..fN` + `--target-col`).
    #[arg(
        long,
        default_value = "/mnt/v/zen/zensim-training/canonical-2026-05-21/train/multiband_anchor_dial100.parquet"
    )]
    anchor: PathBuf,
    /// Per-row target-score column in the anchor.
    #[arg(long, default_value = "target_score")]
    target_col: String,
    /// Feature-column prefix in the anchor.
    #[arg(long, default_value = "f")]
    feat_prefix: String,
    /// Fit the saturation decay on rows with `target > band_min`.
    #[arg(long, default_value_t = 70.0)]
    band_min: f64,
    /// Number of top knots to append.
    #[arg(long, default_value_t = 12)]
    n_knots: usize,
    /// Score-completion tolerance: `r_far` is where the saturation reaches
    /// `100·(1 − far_eps)`.
    #[arg(long, default_value_t = 1e-4)]
    far_eps: f64,
}

fn cmd_extend_top(a: &ExtendTopArgs) -> Result<(), String> {
    let bytes = std::fs::read(&a.input).map_err(|e| format!("read {:?}: {e}", a.input))?;
    let model = Model::from_bytes(&bytes).map_err(|e| format!("parse bake: {e:?}"))?;
    let lin = load_linear(&model);
    let n = lin.scaler_mean.len();
    let ops = build_fw_ops(&model, n)?;

    let (mut xs, mut ys) = read_spline(&model);
    let x0 = *xs.last().unwrap();
    let y0 = *ys.last().unwrap();
    eprintln!(
        "input spline: {} knots, domain [{:.3},{:.3}] y [{:.1},{:.1}] (bottom + in-distribution kept VERBATIM)",
        xs.len(),
        xs[0],
        x0,
        ys[0],
        y0
    );

    let (feats, tgt) = read_features(&a.anchor, &a.feat_prefix, n, &a.target_col);
    // preds over band rows; robust log-OLS log(100−t) ≈ logA − k·raw.
    let mut band_pred = Vec::new();
    let mut band_logy = Vec::new();
    for (row, &t) in feats.iter().zip(&tgt) {
        if t > a.band_min {
            band_pred.push(forward_raw(row, &ops, &lin));
            band_logy.push((100.0 - t).max(1e-3).ln());
        }
    }
    if band_pred.len() < 2 {
        return Err(format!(
            "only {} rows with target>{} — need >=2 for the saturation fit",
            band_pred.len(),
            a.band_min
        ));
    }
    let (_a_int, slope) = ols(&band_pred, &band_logy);
    let k = -slope;
    if k <= 0.0 || k.is_nan() {
        return Err(format!("saturation fit gave non-decaying k={k}"));
    }

    // r_far where score reaches ~100·(1−far_eps); append n_knots on
    // np.linspace(x0+(r_far−x0)/n, r_far, n). Guard: strictly increasing
    // x and y (never lowers a knot).
    let r_far = x0 + (-a.far_eps.ln()) / k;
    let start = x0 + (r_far - x0) / a.n_knots as f64;
    let step = (r_far - start) / (a.n_knots as f64 - 1.0);
    let mut added = 0usize;
    for i in 0..a.n_knots {
        // numpy linspace forces the last sample exactly to the stop value.
        let r = if i == a.n_knots - 1 {
            r_far
        } else {
            i as f64 * step + start
        };
        let y = 100.0 - (100.0 - y0) * (-k * (r - x0)).exp();
        if r > *xs.last().unwrap() + 1e-7 && y > *ys.last().unwrap() {
            xs.push(r);
            ys.push(y);
            added += 1;
        }
    }
    eprintln!(
        "  top extension: k={k:.4} (n={}); {x0:.3}->{r_far:.2}, +{added} knots, y-top {:.2}; final {} knots",
        band_pred.len(),
        ys.last().unwrap(),
        xs.len()
    );

    // re-emit: transforms + params verbatim, new spline (input order is
    // [transforms, params, spline] → replace-in-place preserves it).
    let metadata = metadata_with_spline(&model, &spline_payload(&xs, &ys));
    let sz = emit_linear(
        &a.out,
        &lin.scaler_mean,
        &lin.scaler_scale,
        &lin.weights,
        lin.bias,
        &metadata,
    )
    .map_err(|e| format!("write {:?}: {e}", a.out))?;
    eprintln!("emitted {:?} ({sz} B)", a.out);
    Ok(())
}

/// Clone `model`'s metadata verbatim, replacing the spline entry's value
/// with `new_spline` (in place, preserving order). Appends a spline entry
/// if the bake had none.
fn metadata_with_spline(model: &Model, new_spline: &[u8]) -> Vec<OwnedMeta> {
    let mut md = clone_metadata(model);
    if let Some(e) = md.iter_mut().find(|e| e.key == SPLINE_KEY) {
        e.value = new_spline.to_vec();
    } else {
        md.push(OwnedMeta {
            key: SPLINE_KEY.to_string(),
            kind: MetadataType::Bytes,
            value: new_spline.to_vec(),
        });
    }
    md
}

// --------------------------------------------------------------------------
// subcommand: add-spline  (generic, MLP-capable — 2026-07-18)
// --------------------------------------------------------------------------

#[derive(Args)]
struct AddSplineArgs {
    #[arg(long = "in")]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// Anchor parquet (`f0..fN` + `--target-col`).
    #[arg(long)]
    anchor: PathBuf,
    #[arg(long, default_value = "target_score")]
    target_col: String,
    #[arg(long, default_value = "f")]
    feat_prefix: String,
    /// Multiply the anchor target by this before fitting.
    #[arg(long, default_value_t = 1.0)]
    target_scale: f64,
    /// Percentile-edge count for `fit_spline_knots`.
    #[arg(long, default_value_t = 18)]
    n_edges: usize,
}

fn cmd_add_spline(a: &AddSplineArgs) -> Result<(), String> {
    let bytes = std::fs::read(&a.input).map_err(|e| format!("read {:?}: {e}", a.input))?;
    let model = Model::from_bytes(&bytes).map_err(|e| format!("parse bake: {e:?}"))?;
    if spline::extract(&model).is_some() {
        return Err(
            "bake already carries an output-calibration spline — add-spline is for \
             spline-less bakes (use shared-anchor semantics for a refit, which would \
             otherwise compound two splines)"
                .into(),
        );
    }
    if !model.feature_bounds().is_empty() {
        return Err(format!(
            "bake carries {} feature_bounds entries; add-spline does not round-trip \
             bounds yet — extend emit_full first (fail-loud beats silent drop)",
            model.feature_bounds().len()
        ));
    }
    let n_in = model.caller_input_width();

    // Forward the anchor through the PRODUCTION predictor (transform-safe).
    let (feats, tgt) = read_features(&a.anchor, &a.feat_prefix, n_in, &a.target_col);
    let transformed = model.has_nontrivial_feature_transforms();
    let mut predictor = zenpredict::Predictor::new(&model);
    let mut preds = Vec::with_capacity(feats.len());
    let mut xbuf = vec![0f32; n_in];
    for row in &feats {
        for (d, s) in xbuf.iter_mut().zip(row.iter()) {
            *d = *s as f32;
        }
        let out = if transformed {
            predictor.predict_transformed(&xbuf)
        } else {
            predictor.predict(&xbuf)
        }
        .map_err(|e| format!("predictor forward: {e:?}"))?;
        preds.push(out[0] as f64);
    }
    let tgt_scaled: Vec<f64> = tgt.iter().map(|&t| t * a.target_scale).collect();

    let (cx, cy) = fit_spline_knots(&preds, &tgt_scaled, a.n_edges, true);
    if cx.len() < 2 {
        return Err(format!("anchor fit produced only {} knots (<2)", cx.len()));
    }
    eprintln!(
        "add-spline: n={} anchor rows, raw pred range [{:.3}, {:.3}], {} knots, dial y-range [{:.1}, {:.1}]",
        preds.len(),
        preds.iter().cloned().fold(f64::INFINITY, f64::min),
        preds.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        cx.len(),
        cy[0],
        cy[cy.len() - 1]
    );

    let metadata = metadata_with_spline(&model, &spline_payload(&cx, &cy));
    let sz =
        emit_full(&a.out, &model, &metadata, 0).map_err(|e| format!("write {:?}: {e}", a.out))?;
    eprintln!("emitted {:?} ({sz} B)", a.out);
    Ok(())
}

/// Re-emit an arbitrary bake VERBATIM (every layer, dtype-preserving) with
/// replacement metadata — the MLP-capable sibling of [`emit_linear`]. f16
/// weights round-trip exactly; i8 layers error (extend when needed).
/// `schema_hash`: `add-spline` passes 0 (its established output bytes predate
/// the parameter — do not change them); `strip` preserves the input's.
fn emit_full(
    out: &Path,
    model: &Model,
    metadata: &[OwnedMeta],
    schema_hash: u64,
) -> std::io::Result<usize> {
    struct OwnedLayer {
        in_dim: usize,
        out_dim: usize,
        activation: Activation,
        dtype: WeightDtype,
        weights: Vec<f32>,
        biases: Vec<f32>,
    }
    let mut owned: Vec<OwnedLayer> = Vec::with_capacity(model.n_layers());
    for l in model.layers() {
        let (dtype, weights) = match &l.weights {
            WeightStorage::F32(w) => (WeightDtype::F32, w.to_vec()),
            WeightStorage::F16(w) => (
                WeightDtype::F16,
                w.iter().map(|b| f16_bits_to_f32(*b)).collect(),
            ),
            WeightStorage::I8 { .. } => {
                return Err(std::io::Error::other(
                    "emit_full: i8 layers not supported yet (extend the dequant like repack)",
                ));
            }
        };
        owned.push(OwnedLayer {
            in_dim: l.in_dim,
            out_dim: l.out_dim,
            activation: l.activation,
            dtype,
            weights,
            biases: l.biases.to_vec(),
        });
    }
    let layers: Vec<BakeLayer<'_>> = owned
        .iter()
        .map(|l| BakeLayer {
            in_dim: l.in_dim,
            out_dim: l.out_dim,
            activation: l.activation,
            dtype: l.dtype,
            weights: &l.weights,
            biases: &l.biases,
        })
        .collect();
    let md: Vec<BakeMetadataEntry<'_>> = metadata
        .iter()
        .map(|m| BakeMetadataEntry {
            key: &m.key,
            kind: m.kind,
            value: &m.value,
        })
        .collect();
    let bytes = bake(&BakeRequest {
        schema_hash,
        flags: 0,
        scaler_mean: model.scaler_mean(),
        scaler_scale: model.scaler_scale(),
        layers: &layers,
        feature_bounds: &[],
        metadata: &md,
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: true,
        hu_permutations: None,
    })
    .expect("serialize ZNPR v3 bake");
    std::fs::write(out, &bytes)?;
    Ok(bytes.len())
}

// --------------------------------------------------------------------------
// subcommand: shared-anchor  (reproduces shared_anchor_refit.py core)
// --------------------------------------------------------------------------

#[derive(Args)]
struct SharedAnchorArgs {
    #[arg(long = "in")]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// Anchor parquet (`f0..fN` + `--target-col`).
    #[arg(long)]
    anchor: PathBuf,
    #[arg(long, default_value = "target_score")]
    target_col: String,
    #[arg(long, default_value = "f")]
    feat_prefix: String,
    /// Multiply the anchor target by this before fitting (e.g. 100 to map a
    /// `human_score` in [0,1] onto the 0..100 dial).
    #[arg(long, default_value_t = 1.0)]
    target_scale: f64,
    /// Percentile-edge count for `fit_spline_knots` (Python default 18).
    #[arg(long, default_value_t = 18)]
    n_edges: usize,
}

fn cmd_shared_anchor(a: &SharedAnchorArgs) -> Result<(), String> {
    let bytes = std::fs::read(&a.input).map_err(|e| format!("read {:?}: {e}", a.input))?;
    let model = Model::from_bytes(&bytes).map_err(|e| format!("parse bake: {e:?}"))?;
    let lin = load_linear(&model);
    let n = lin.scaler_mean.len();
    let ops = build_fw_ops(&model, n)?;

    let (feats, tgt) = read_features(&a.anchor, &a.feat_prefix, n, &a.target_col);
    let preds: Vec<f64> = feats.iter().map(|r| forward_raw(r, &ops, &lin)).collect();
    let tgt_scaled: Vec<f64> = tgt.iter().map(|&t| t * a.target_scale).collect();

    let (cx, cy) = fit_spline_knots(&preds, &tgt_scaled, a.n_edges, true);
    if cx.len() < 2 {
        return Err(format!(
            "anchor refit produced only {} knots (<2)",
            cx.len()
        ));
    }
    eprintln!(
        "shared-anchor refit: {} knots, dial y-range [{:.1}, {:.1}]",
        cx.len(),
        cy[0],
        cy[cy.len() - 1]
    );

    let metadata = metadata_with_spline(&model, &spline_payload(&cx, &cy));
    let sz = emit_linear(
        &a.out,
        &lin.scaler_mean,
        &lin.scaler_scale,
        &lin.weights,
        lin.bias,
        &metadata,
    )
    .map_err(|e| format!("write {:?}: {e}", a.out))?;
    eprintln!("emitted {:?} ({sz} B)", a.out);
    Ok(())
}

// --------------------------------------------------------------------------
// subcommand: bottom-extend  (reproduces bhdr_bottom_extend.py)
// --------------------------------------------------------------------------

#[derive(Args)]
struct BottomExtendArgs {
    #[arg(long = "in")]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// Raw value of the new bottom knot (score 0). Must be `< current
    /// bottom knot`.
    #[arg(long, default_value_t = 0.0)]
    floor_raw: f64,
}

fn cmd_bottom_extend(a: &BottomExtendArgs) -> Result<(), String> {
    let bytes = std::fs::read(&a.input).map_err(|e| format!("read {:?}: {e}", a.input))?;
    let model = Model::from_bytes(&bytes).map_err(|e| format!("parse bake: {e:?}"))?;
    let lin = load_linear(&model);
    let (mut xs, mut ys) = read_spline(&model);
    if a.floor_raw >= xs[0] {
        return Err(format!(
            "floor-raw {} must be < bottom knot {}",
            a.floor_raw, xs[0]
        ));
    }
    let nk_before = xs.len();
    xs.insert(0, a.floor_raw);
    ys.insert(0, 0.0);

    let metadata = metadata_with_spline(&model, &spline_payload(&xs, &ys));
    let sz = emit_linear(
        &a.out,
        &lin.scaler_mean,
        &lin.scaler_scale,
        &lin.weights,
        lin.bias,
        &metadata,
    )
    .map_err(|e| format!("write {:?}: {e}", a.out))?;
    eprintln!(
        "bottom-extended: {nk_before}->{} knots, new bottom ({},0.0); {sz} B -> {:?}",
        xs.len(),
        a.floor_raw,
        a.out
    );
    Ok(())
}

// --------------------------------------------------------------------------
// subcommand: add-winsor  (reproduces winsorize_bake.py)
// --------------------------------------------------------------------------

#[derive(Args)]
struct AddWinsorArgs {
    /// Raw-space linear bake (must have NO feature transforms).
    #[arg(long = "in")]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// Parquet whose `f0..fN` columns the winsor bounds are computed from.
    #[arg(long)]
    fit_corpus: PathBuf,
    #[arg(long, default_value = "f")]
    feat_prefix: String,
    #[arg(long, default_value_t = 0.1)]
    lo_pct: f64,
    #[arg(long, default_value_t = 99.9)]
    hi_pct: f64,
    /// If given, assert the output sha256 begins with this hex prefix.
    #[arg(long)]
    expect_sha256: Option<String>,
    /// COMPOSE mode (SOTA-944 §3d): accept an input whose transforms are the
    /// parameterless monotone set {identity, log1p, signed_cbrt} and emit
    /// the winsor-composed runtime tokens (`winsor_p99` /
    /// `winsor_then_log1p` / `winsor_then_signed_cbrt`) with RAW-space
    /// bounds — for a monotone t, winsor-then-t with raw bounds ≡
    /// t-then-winsor with transformed bounds, exactly. Default OFF keeps
    /// the raw-bake byte-repro paths (B lineage) untouched.
    #[arg(long)]
    compose: bool,
}

fn cmd_add_winsor(a: &AddWinsorArgs) -> Result<(), String> {
    let bytes = std::fs::read(&a.input).map_err(|e| format!("read {:?}: {e}", a.input))?;
    let model = Model::from_bytes(&bytes).map_err(|e| format!("parse bake: {e:?}"))?;
    let has_transforms = model.metadata().iter().any(|e| e.key.contains("transform"));
    if has_transforms && !a.compose {
        return Err(
            "input already has feature transforms — winsorize a RAW bake (or pass --compose \
             for a shaped bake with {identity, log1p, signed_cbrt} transforms)"
                .into(),
        );
    }
    // Compose mode: read the existing token list (must be the parameterless
    // monotone set) and REPLACE the transform metadata with the composed
    // winsor forms; everything else (spline, repro, weights) is carried.
    let compose_toks: Option<Vec<String>> = if a.compose && has_transforms {
        let toks_txt = model
            .metadata()
            .iter()
            .find(|e| e.key == zenpredict::keys::FEATURE_TRANSFORMS)
            .and_then(|e| std::str::from_utf8(e.value).ok())
            .ok_or("compose: input transforms metadata unreadable")?
            .to_string();
        let toks: Vec<String> = toks_txt.lines().map(|s| s.to_string()).collect();
        for (j, t) in toks.iter().enumerate() {
            if !matches!(t.as_str(), "identity" | "log1p" | "signed_cbrt") {
                return Err(format!(
                    "compose: feat {j} has transform {t:?} — only identity/log1p/signed_cbrt \
                     compose with winsor"
                ));
            }
        }
        Some(toks)
    } else {
        None
    };
    let lin = load_linear(&model);
    let n = lin.scaler_mean.len();
    if let Some(toks) = &compose_toks
        && toks.len() != n
    {
        return Err(format!(
            "compose: transform count {} != n_features {n}",
            toks.len()
        ));
    }

    // per-feature [lo_pct, hi_pct]; zero-constant features get [0, 1e-9].
    // read_features needs a target column but add-winsor uses none — point
    // it at f0 (always present) and drop the returned target.
    let throwaway_target = format!("{}0", a.feat_prefix);
    let (feats, _) = read_features(&a.fit_corpus, &a.feat_prefix, n, &throwaway_target);
    let mut lo = vec![0.0f64; n];
    let mut hi = vec![0.0f64; n];
    for j in 0..n {
        let mut col: Vec<f64> = feats.iter().map(|r| r[j]).collect();
        col.sort_by(f64::total_cmp);
        // Shared with `refit-winsor` so the two windows rules cannot diverge
        // (arithmetic byte-identical to the inlined form it replaced).
        let (l, h) = winsor_window(&col, a.lo_pct, a.hi_pct);
        lo[j] = l;
        hi[j] = h;
    }

    let transforms_txt = match &compose_toks {
        None => vec!["winsor_p99"; n].join("\n"),
        Some(toks) => toks
            .iter()
            .map(|t| match t.as_str() {
                "identity" => "winsor_p99",
                "log1p" => "winsor_then_log1p",
                "signed_cbrt" => "winsor_then_signed_cbrt",
                _ => unreachable!("validated above"),
            })
            .collect::<Vec<_>>()
            .join("\n"),
    };
    let params_txt = (0..n)
        .map(|j| format!("{},{}", lo[j], hi[j]))
        .collect::<Vec<_>>()
        .join("\n");
    // metadata order: transforms, params, then everything the raw bake had
    // (incl. its spline) verbatim. In compose mode the input's OWN transform
    // entries are superseded by the composed ones — never duplicated.
    let mut metadata = vec![
        OwnedMeta {
            key: zenpredict::keys::FEATURE_TRANSFORMS.to_string(),
            kind: MetadataType::Utf8,
            value: transforms_txt.into_bytes(),
        },
        OwnedMeta {
            key: zenpredict::keys::FEATURE_TRANSFORM_PARAMS.to_string(),
            kind: MetadataType::Utf8,
            value: params_txt.into_bytes(),
        },
    ];
    metadata.extend(clone_metadata(&model).into_iter().filter(|e| {
        !(compose_toks.is_some()
            && (e.key == zenpredict::keys::FEATURE_TRANSFORMS
                || e.key == zenpredict::keys::FEATURE_TRANSFORM_PARAMS))
    }));

    let sz = emit_linear(
        &a.out,
        &lin.scaler_mean,
        &lin.scaler_scale,
        &lin.weights,
        lin.bias,
        &metadata,
    )
    .map_err(|e| format!("write {:?}: {e}", a.out))?;

    let out_bytes = std::fs::read(&a.out).map_err(|e| format!("re-read {:?}: {e}", a.out))?;
    let got = sha256_hex(&out_bytes);
    eprintln!(
        "winsorized {:?} -> {:?} ({sz} B); {n} winsor-guard transforms (composed when --compose), fit [p{},p{}]\n  sha256 {got}",
        a.input, a.out, a.lo_pct, a.hi_pct
    );
    if let Some(expect) = &a.expect_sha256 {
        if got.starts_with(expect) {
            eprintln!("  BYTE-REPRODUCED (matches expected {expect})");
        } else {
            return Err(format!("sha mismatch: expected {expect}, got {got}"));
        }
    }
    Ok(())
}

// --------------------------------------------------------------------------
// subcommand: gate  (reproduces bake_outlier_gate.py)
// --------------------------------------------------------------------------

#[derive(Args)]
struct GateArgs {
    #[arg(long)]
    bake: PathBuf,
    /// Broad corpus (`f0..fN` + `--ref-col`).
    #[arg(long)]
    corpus: PathBuf,
    /// Reference-metric column for the advisory Z-RMSE / OR / SROCC.
    #[arg(long, default_value = "human_score")]
    ref_col: String,
    #[arg(long, default_value = "f")]
    feat_prefix: String,
    /// G-RANGE hard-gate threshold: fail if more than this fraction of raw
    /// preds fall outside the spline knot domain.
    #[arg(long, default_value_t = 1e-4)]
    range_frac: f64,
}

fn cmd_gate(a: &GateArgs) -> Result<bool, String> {
    use zensim_validate::bake_runtime::{
        extract_hybrid_head, extract_minmax_head, extract_per_sample_alpha_head,
        extract_tanh_output_head_scale, score_row, score_row_minmax,
    };
    let bytes = std::fs::read(&a.bake).map_err(|e| format!("read {:?}: {e}", a.bake))?;
    let model = Model::from_bytes(&bytes).map_err(|e| format!("parse bake: {e:?}"))?;
    let n = model.caller_input_width();
    let sp = spline::extract(&model).ok_or("bake has no output_calibration_spline")?;
    let klo = sp.xs[0];
    let khi = sp.xs[sp.xs.len() - 1];

    let (feats, refv) = read_features(&a.corpus, &a.feat_prefix, n, &a.ref_col);

    // Forward through the SHARED production scoring path (`bake_runtime` —
    // the same per-sample-α / hybrid / min-max / tanh-pin dispatch
    // bake_verdict uses), with the output spline DISABLED so `raw` is
    // exactly the value the spline maps (post-head, post-tanh-pin). This
    // makes the gate evaluable for EVERY bake class. It replaces a
    // linear-only local forward that asserted `n_layers == 1` — the reason
    // G-RANGE read "NOT EVALUABLE (inherited MLP tool gap)" for every MLP
    // candidate of the SOTA-944 campaign.
    let per_sample_alpha = extract_per_sample_alpha_head(&model);
    let hybrid = extract_hybrid_head(&model);
    let minmax = extract_minmax_head(&model);
    let tanh_pin = extract_tanh_output_head_scale(&model);
    let has_transforms = model.has_nontrivial_feature_transforms();
    let mut predictor = zenpredict::Predictor::new(&model);
    let mut scratch = vec![0f32; n];
    let raw: Vec<f64> = feats
        .iter()
        .map(|row| match minmax.as_ref() {
            // Min-max bakes REPLACE the layer forward — bypass the Predictor
            // (same branch shape as bake_verdict::score_grid_one).
            Some(mm) => score_row_minmax(&model, mm, tanh_pin, None, row),
            None => score_row(
                &mut predictor,
                has_transforms,
                per_sample_alpha.as_ref(),
                hybrid.as_ref(),
                tanh_pin,
                None,
                &mut scratch,
                row,
            ),
        })
        .collect();
    let dial: Vec<f64> = raw.iter().map(|&r| spline::apply(r, &sp)).collect();
    let ntotal = raw.len();

    // G-RANGE — the HARD gate. Raw preds outside [klo, khi] extrapolate
    // (uncapped downward off the bottom knot) — the tail SROCC hides.
    let below = raw.iter().filter(|&&r| r < klo).count();
    let above = raw.iter().filter(|&&r| r > khi).count();
    let range_fail = (below + above) as f64 > ntotal as f64 * a.range_frac;

    let bn = a
        .bake
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("bake");
    let cn = a
        .corpus
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("corpus");
    let rawmin = raw.iter().copied().fold(f64::INFINITY, f64::min);
    let rawmax = raw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!("\n=== outlier gate: {bn} on {cn} (n={ntotal}) ===");
    println!("knot domain [{klo:.3}, {khi:.3}]  |  raw pred [{rawmin:.2}, {rawmax:.2}]");
    println!(
        "[HARD] G-RANGE   below-knot {below} ({:.3}%)  above-knot {above} ({:.3}%)  -> {} (gate: <{:.3}% extrapolating)",
        100.0 * below as f64 / ntotal as f64,
        100.0 * above as f64 / ntotal as f64,
        if range_fail { "FAIL" } else { "PASS" },
        100.0 * a.range_frac
    );

    // advisory panel vs the reference metric — Z-RMSE + OR after a 4PL
    // rescale + polarity-tolerant SROCC. NO PWRC (zenstats sa_st_curve is
    // O(n²)-memory and OOMs broad corpora).
    let rescaled = rescale_logistic(&dial, &refv);
    let zr = z_rmse(&rescaled, &refv);
    let outr = outlier_ratio(&rescaled, &refv);
    let sr = spearman(&dial, &refv).abs();
    println!(
        "[adv]  G-ZRMSE   {zr:.3} vs {} (lower=better; a tail inflates this while SROCC stays flat)",
        a.ref_col
    );
    println!(
        "[adv]  G-SROCC   {sr:.4}  (rank — near-INVARIANT to the tail; that's why it hides the bug)"
    );
    println!("[adv]  G-OUTRATIO {outr:.4}  (fraction outside +/-1.96 sigma of rescaled reference)");

    println!(
        "\nVERDICT: {}  [HARD gate = G-RANGE only; the rest are advisory]",
        if range_fail {
            "FAIL (blocks ship)"
        } else {
            "PASS (ship-eligible)"
        }
    );
    Ok(range_fail)
}

// --------------------------------------------------------------------------
// subcommand: pack  (reproduces pack_and_calibrate.py — STANDARD non-QAT path)
// --------------------------------------------------------------------------

#[derive(Args)]
struct PackArgs {
    /// Input bake (multi-layer MLPs supported; f32/f16 layers).
    #[arg(long = "in")]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// Bulk-layer weight dtype after packing (`f16`, `f32`, or `i8`).
    #[arg(long, default_value = "f16")]
    dtype: String,
    /// Zero every bulk-layer weight with |w| < tau (per-layer zerobias).
    /// Default: 0.005 for dense MLPs — but 0 when a SPARSE-CLASS bake is
    /// detected (the flat 0.005 would kill >10% of its live layer-0 lines
    /// outright; it cost ADD156 −0.0069 CID22 and wiped GL4_s2501 57→3 live
    /// rows). Pass a value explicitly to override the detection either way.
    #[arg(long = "zerobias-bulk")]
    zerobias_bulk: Option<f64>,
    /// Exempt the LAST layer from zerobias AND keep it f32 (identity-
    /// critical layers, e.g. the per-sample-alpha passthrough, are tiny in
    /// bytes but precision-sensitive).
    #[arg(long)]
    protect_last: bool,
    /// Keep only the last of any run of y<=1e-6 spline knots (negative-tail
    /// dedup in `fit_spline_knots`) — i.e. PRESERVE the dial's negative tail.
    ///
    /// **You almost always want this.** Without it, a run of `y ≈ 0` knots
    /// leaves the spline's bottom segment FLAT at zero: every prediction in
    /// that range maps to exactly `0.0` and the extrapolation below the bottom
    /// knot has slope 0, so inputs worse than the worst codec output — which
    /// the product contract says must score BELOW 0 — pin to a dead zone.
    /// Measured on ADD156: dial p5 `−12.4334` → `0.0000` and up to −0.021
    /// SROCC (LIVE 0.9602 → 0.9397), all of it restored by this flag.
    ///
    /// When the choice would change the fitted spline, `pack` REFUSES to run
    /// without one of `--neg-tail` / `--no-neg-tail` (ADD156 audit, D4).
    #[arg(long)]
    neg_tail: bool,
    /// Explicitly accept the FLAT-bottom spline (the pre-2026-08-31 default):
    /// keep every `y <= 1e-6` knot and let the negative tail collapse to a
    /// dead zone at 0.
    ///
    /// Exists so a historical bake can still be reproduced byte-for-byte. It
    /// is never the right choice for a new ship candidate.
    #[arg(long, conflicts_with = "neg_tail")]
    no_neg_tail: bool,
    /// Anchor parquet the packed-network spline is fit on
    /// (`f0..fN` + `--target-col`).
    #[arg(
        long,
        default_value = "/mnt/v/zen/zensim-training/canonical-2026-05-21/train/multiband_anchor_dial100.parquet"
    )]
    anchor: PathBuf,
    #[arg(long, default_value = "target_score")]
    target_col: String,
    #[arg(long, default_value = "f")]
    feat_prefix: String,
    /// Advisory verification parquet (post-spline SROCC + calibration
    /// percentiles). Pass `--verify none` to skip.
    #[arg(
        long,
        default_value = "/mnt/v/zen/zensim-training/2026-08-30-full-features-372/cid22_features_372col_2026-05-15.parquet"
    )]
    verify: String,
    #[arg(long, default_value = "human_score")]
    verify_col: String,
    /// Multiply the verify column by this before the SROCC (CID22 stores
    /// MCOS/100, so the default re-scales to 0..100).
    #[arg(long, default_value_t = 100.0)]
    verify_scale: f64,
    /// If given, assert the output sha256 begins with this hex prefix.
    #[arg(long)]
    expect_sha256: Option<String>,
    /// Skip dead-column pruning (on by default). Pruning drops layer-0
    /// inputs whose weight row is exactly zero, and inputs the bake's own
    /// transform pins to a constant, declaring `drop` on those raw lines so
    /// the CALLER's feature width is unchanged. Pass this to reproduce a
    /// pre-2026-08-04 bake byte-for-byte.
    #[arg(long)]
    no_prune: bool,
    /// Restrict pruning to class 1 (exactly-zero weight rows), which is
    /// bit-identical for every input including NaN. Without this, class 2
    /// (transform-forced-constant, folded into the layer-0 bias) is also
    /// pruned — exact in real arithmetic, but a NaN feature on such a
    /// column no longer propagates to the output.
    #[arg(long)]
    no_prune_constants: bool,
    /// Max allowed |Δ| between pre- and post-prune anchor scores. Class-1
    /// pruning is asserted BIT-identical regardless of this; the tolerance
    /// only applies once class-2 bias folding is in play.
    #[arg(long, default_value_t = 1e-4)]
    prune_identity_tol: f64,
}

/// One packed layer, owned (weights dequantized to f32).
struct PackLayer {
    in_dim: usize,
    out_dim: usize,
    activation: Activation,
    dtype: WeightDtype,
    weights: Vec<f32>,
    biases: Vec<f32>,
}

/// Per-layer zerobias + dtype assignment — the pure core of `pack`.
/// Returns the packed layers + per-layer `(index, zeroed, total)` counts.
///
/// Parity note (byte-identity with `pack_and_calibrate.py`): the Python
/// compared `|w| < tau` on weights that had round-tripped through
/// `zenpredict inspect` JSON — i.e. on `float(shortest_repr(w_f32))`, which
/// is NOT always the same f64 as `w_f32 as f64`. The comparison here does
/// the same string round-trip so threshold-straddling weights mask
/// identically. The weights passed on are the original f32s (the Python's
/// JSON f64 → baker f32 narrowing recovers exactly these), so the emitted
/// bytes match.
/// Per-layer (kept, zeroed, total) weight counts reported by [`pack_layers`].
type PackCounts = Vec<(usize, usize, usize)>;

fn pack_layers(
    model: &Model,
    dtype: WeightDtype,
    tau: f64,
    protect_last: bool,
) -> Result<(Vec<PackLayer>, PackCounts), String> {
    let n_layers = model.n_layers();
    let mut packed = Vec::with_capacity(n_layers);
    let mut counts = Vec::with_capacity(n_layers);
    for (li, l) in model.layers().enumerate() {
        let mut weights: Vec<f32> = match &l.weights {
            WeightStorage::F32(w) => w.to_vec(),
            WeightStorage::F16(w) => w.iter().map(|b| f16_bits_to_f32(*b)).collect(),
            WeightStorage::I8 { .. } => {
                return Err(format!(
                    "pack: layer {li} is i8 — repacking an already-i8 bake is \
                     lossy; start from the f32/f16 original"
                ));
            }
        };
        let is_last = li == n_layers - 1;
        let layer_tau = if protect_last && is_last { 0.0 } else { tau };
        let mut zeroed = 0usize;
        if layer_tau > 0.0 {
            for w in weights.iter_mut() {
                // Python-pipeline parity: threshold on the shortest-repr
                // string round-trip (see fn doc).
                let wj: f64 = format!("{w}").parse().unwrap_or(*w as f64);
                if wj.abs() < layer_tau {
                    *w = 0.0;
                    zeroed += 1;
                }
            }
        }
        counts.push((li, zeroed, weights.len()));
        packed.push(PackLayer {
            in_dim: l.in_dim,
            out_dim: l.out_dim,
            activation: l.activation,
            dtype: if protect_last && is_last {
                WeightDtype::F32
            } else {
                dtype
            },
            weights,
            biases: l.biases.to_vec(),
        });
    }
    Ok((packed, counts))
}

/// Serialize packed layers + metadata through the canonical serializer,
/// returning the bytes. `schema_hash` is preserved from the input;
/// `flags: 0` and `compressed: true` match the Python pipeline verbatim.
fn emit_packed(
    schema_hash: u64,
    scaler_mean: &[f32],
    scaler_scale: &[f32],
    layers: &[PackLayer],
    metadata: &[OwnedMeta],
) -> Vec<u8> {
    let bl: Vec<BakeLayer<'_>> = layers
        .iter()
        .map(|l| BakeLayer {
            in_dim: l.in_dim,
            out_dim: l.out_dim,
            activation: l.activation,
            dtype: l.dtype,
            weights: &l.weights,
            biases: &l.biases,
        })
        .collect();
    let md: Vec<BakeMetadataEntry<'_>> = metadata
        .iter()
        .map(|m| BakeMetadataEntry {
            key: &m.key,
            kind: m.kind,
            value: &m.value,
        })
        .collect();
    bake(&BakeRequest {
        schema_hash,
        flags: 0,
        scaler_mean,
        scaler_scale,
        layers: &bl,
        feature_bounds: &[],
        metadata: &md,
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: true,
        hu_permutations: None,
    })
    .expect("serialize ZNPR v3 bake")
}

/// Forward `feats` through a bake's FULL runtime dispatch (per-sample-α /
/// hybrid head / tanh pin / output spline when present) — the shared
/// `bake_runtime::score_with_bake_alloc` path, bit-exact with
/// `predict_features_with_bake`. Two pipeline-parity details:
/// * features are narrowed to f32 first (the Python packed a f32 matrix);
/// * each score is round-tripped through its `%.6f` print (the Python fit
///   knots on the subprocess's 6-decimal stdout).
fn forward_scored_6dec(bytes: &[u8], feats: &[Vec<f64>]) -> Result<Vec<f64>, String> {
    use zensim_validate::bake_runtime::{
        extract_hybrid_head, extract_per_sample_alpha_head, extract_tanh_output_head_scale,
        score_with_bake_alloc,
    };
    let model = Model::from_bytes(bytes).map_err(|e| format!("parse packed bake: {e:?}"))?;
    // Caller width, NOT n_inputs(): a pruned bake's layer-0 in_dim is
    // smaller than the vector its callers hand it.
    let n_inputs = model.caller_input_width();
    let has_transforms = model.has_nontrivial_feature_transforms();
    let psa = extract_per_sample_alpha_head(&model);
    let hyb = extract_hybrid_head(&model);
    let pin = extract_tanh_output_head_scale(&model);
    let sp = spline::extract(&model);
    let mut predictor = zenpredict::Predictor::new(&model);
    let mut out = Vec::with_capacity(feats.len());
    let mut row_f64 = vec![0f64; n_inputs];
    for row in feats {
        if row.len() != n_inputs {
            return Err(format!(
                "feature row has {} values, bake expects {n_inputs}",
                row.len()
            ));
        }
        for (d, s) in row_f64.iter_mut().zip(row.iter()) {
            *d = (*s as f32) as f64;
        }
        let y = score_with_bake_alloc(
            &mut predictor,
            has_transforms,
            psa.as_ref(),
            hyb.as_ref(),
            pin,
            sp.as_ref(),
            n_inputs,
            &row_f64,
        );
        out.push(round_6dec(y));
    }
    Ok(out)
}

/// `%.6f`-print round-trip (see [`forward_scored_6dec`]).
fn round_6dec(y: f64) -> f64 {
    format!("{y:.6}").parse().unwrap_or(y)
}

/// Replace (or append) a UTF-8 metadata entry, preserving the position
/// of an existing key so the metadata blob's order is otherwise stable.
fn set_meta_utf8(md: &mut Vec<OwnedMeta>, key: &str, value: Vec<u8>) {
    match md.iter_mut().find(|m| m.key == key) {
        Some(slot) => {
            slot.kind = MetadataType::Utf8;
            slot.value = value;
        }
        None => md.push(OwnedMeta {
            key: key.to_string(),
            kind: MetadataType::Utf8,
            value,
        }),
    }
}

/// THE identity gate. Pruning is only ever allowed to be a storage +
/// inference optimization, so the pre- and post-prune networks must
/// score the anchor corpus the same. Two strictnesses:
///
/// * **class 1 only** ⇒ demand exact equality of every score. The
///   dropped rows are exactly zero, so `fma(x, 0, acc) == acc` and the
///   bits cannot move.
/// * **class 2 present** ⇒ the constant fold reorders one `f32` sum, so
///   demand `|Δ| <= --prune-identity-tol` and report the worst case.
///
/// Fails loud either way. A pruner that quietly changes predictions is

/// What packing did to the dial's NEGATIVE TAIL — the half of "identity" the
/// prune gate structurally cannot see (ADD156 ship audit, defect **D4**).
///
/// `check_prune_identity` compares the NETWORK's raw outputs on in-domain
/// anchor rows. The tail lives in the output-calibration SPLINE, which `pack`
/// refits from scratch, so a run could delete the tail entirely and still
/// print nothing but "PASS — BIT-identical". This report closes that gap.
#[derive(Debug, PartialEq)]
struct DialTailReport {
    /// Leading spline knots pinned at `y <= 1e-6`. More than one ⇒ the bottom
    /// segment is flat and the extrapolation below it has slope 0.
    flat_bottom_knots: usize,
    /// Anchor rows the INPUT bake scored below zero.
    in_negative: usize,
    /// ...of which the packed bake pins to (approximately) zero.
    pinned_to_zero: usize,
    in_p5: f64,
    out_p5: f64,
}

impl DialTailReport {
    fn tail_deleted(&self) -> bool {
        self.flat_bottom_knots > 1 || (self.in_negative > 0 && self.pinned_to_zero > 0)
    }
    fn render(&self) -> String {
        if self.tail_deleted() {
            format!(
                "dial tail: ⚠ DELETED — {} leading spline knots sit at y<=1e-6 (flat bottom, \
                 slope 0 below it) and {}/{} anchor rows the INPUT bake scored below zero are \
                 pinned to zero by the packed bake; dial p5 {:.4} -> {:.4}. Negative zensim \
                 values MUST work: inputs worse than the worst codec output score BELOW 0. \
                 Re-pack with --neg-tail.",
                self.flat_bottom_knots,
                self.pinned_to_zero,
                self.in_negative,
                self.in_p5,
                self.out_p5
            )
        } else {
            format!(
                "dial tail: PRESERVED — bottom knot is not a flat run ({} at y<=1e-6), {} \
                 negative anchor rows survive packing; dial p5 {:.4} -> {:.4}",
                self.flat_bottom_knots,
                self.in_negative - self.pinned_to_zero,
                self.in_p5,
                self.out_p5
            )
        }
    }
}

fn dial_tail_report(
    cx: &[f64],
    cy: &[f64],
    in_scores: &[f64],
    out_scores: &[f64],
) -> DialTailReport {
    let _ = cx;
    let flat_bottom_knots = cy.iter().take_while(|&&y| y <= 1e-6).count();
    let mut in_negative = 0usize;
    let mut pinned_to_zero = 0usize;
    for (&i, &o) in in_scores.iter().zip(out_scores.iter()) {
        if i < 0.0 {
            in_negative += 1;
            if o.abs() <= 1e-6 {
                pinned_to_zero += 1;
            }
        }
    }
    let p5 = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        zensim_validate::dial_spline::percentile_linear(&s, 5.0)
    };
    DialTailReport {
        flat_bottom_knots,
        in_negative,
        pinned_to_zero,
        in_p5: p5(in_scores),
        out_p5: p5(out_scores),
    }
}

/// the exact failure this whole module exists to make impossible.
fn check_prune_identity(
    plan: &prune::PrunePlan,
    reference: &[f64],
    pruned: &[f64],
    a: &PackArgs,
) -> Result<(), String> {
    if reference.len() != pruned.len() {
        return Err(format!(
            "prune identity gate: score-count mismatch {} vs {}",
            reference.len(),
            pruned.len()
        ));
    }
    let mut worst = 0f64;
    let mut worst_at = 0usize;
    let mut n_diff = 0usize;
    for (i, (&r, &p)) in reference.iter().zip(pruned.iter()).enumerate() {
        if r.to_bits() != p.to_bits() {
            n_diff += 1;
            let d = (r - p).abs();
            if d > worst {
                worst = d;
                worst_at = i;
            }
        }
    }
    if plan.is_bit_identical() {
        if n_diff != 0 {
            return Err(format!(
                "PRUNE IDENTITY GATE FAILED: {n_diff}/{} anchor scores changed under \
                 class-1-only pruning, which MUST be bit-identical (worst |Δ|={worst:.6e} \
                 at row {worst_at}). Refusing to write the bake.",
                reference.len()
            ));
        }
        eprintln!(
            "prune identity gate: PASS — all {} anchor scores BIT-identical (class 1 only)",
            reference.len()
        );
        return Ok(());
    }
    if worst > a.prune_identity_tol {
        return Err(format!(
            "PRUNE IDENTITY GATE FAILED: worst |Δ|={worst:.6e} at anchor row {worst_at} \
             exceeds --prune-identity-tol {:.1e} ({n_diff}/{} scores moved). Refusing to \
             write the bake; re-run with --no-prune-constants for the bit-identical subset.",
            a.prune_identity_tol,
            reference.len()
        ));
    }
    eprintln!(
        "prune identity gate: PASS — {n_diff}/{} anchor scores moved, worst |Δ|={worst:.3e} \
         (<= {:.1e}); class-2 bias folding reorders one f32 sum",
        reference.len(),
        a.prune_identity_tol
    );
    Ok(())
}

/// The dense-MLP zerobias default `pack` has always shipped (v47 / packed30k
/// provenance — byte-repro of those bakes relies on this value resolving when
/// no explicit tau is given and the bake is dense).
const ZEROBIAS_DENSE_DEFAULT: f64 = 0.005;
/// Sparse-class trigger: auto-default tau to 0 when the dense default would
/// kill MORE than this fraction of live layer-0 lines. Measured population
/// (2026-08-06): dense MLPs (v47 76.6% live, C/EM4 ~71% live) kill ~0 whole
/// lines; ADD156 kills 13/26 = 50%; GL4_s2501 killed 54/57 = 95%.
const SPARSE_KILL_FRAC: f64 = 0.10;

/// C8 default resolution (appendix W): explicit flag always wins; otherwise
/// dense default, downgraded to 0 for the sparse class. Second return =
/// "auto-sparse fired" (for the loud note).
fn resolve_zerobias(explicit: Option<f64>, killed: usize, live: usize) -> (f64, bool) {
    if let Some(v) = explicit {
        return (v, false);
    }
    let frac = if live == 0 {
        0.0
    } else {
        killed as f64 / live as f64
    };
    if frac > SPARSE_KILL_FRAC {
        (0.0, true)
    } else {
        (ZEROBIAS_DENSE_DEFAULT, false)
    }
}

fn cmd_pack(a: &PackArgs) -> Result<(), String> {
    let dtype = match a.dtype.as_str() {
        "f16" => WeightDtype::F16,
        "f32" => WeightDtype::F32,
        "i8" => WeightDtype::I8,
        other => return Err(format!("--dtype must be f16|f32|i8 (got {other:?})")),
    };
    let bytes = std::fs::read(&a.input).map_err(|e| format!("read {:?}: {e}", a.input))?;
    let model = Model::from_bytes(&bytes).map_err(|e| format!("parse bake: {e:?}"))?;
    // The Python JSON pipeline silently DROPPED these sections. Refuse
    // instead — a pack that strips output_specs ships a broken bake.
    if !model.feature_bounds().is_empty()
        || !model.output_specs().is_empty()
        || !model.discrete_sets().is_empty()
        || !model.sparse_overrides().is_empty()
    {
        return Err(
            "bake carries feature_bounds/output_specs/discrete_sets/sparse_overrides — \
             pack does not round-trip those sections yet (extend emit_packed first; \
             fail-loud beats the Python's silent drop)"
                .into(),
        );
    }
    if model.flags() != 0 {
        eprintln!(
            "warning: input flags=0x{:x} are reset to 0 (pipeline parity with pack_and_calibrate.py)",
            model.flags()
        );
    }

    // C8 sparse-class default (appendix W): the flat 0.005 zerobias is
    // calibrated for dense 100-500 KB MLPs. On a sparse fit it kills whole
    // live lines — measured twice (T.R11: ADD156 −0.0069 CID22; J.R3:
    // GL4_s2501 wiped 57→3 rows) — so when no explicit tau is given, probe
    // what 0.005 WOULD kill and default to 0 on the sparse class.
    let (killed, live) =
        block_profile::zerobias_line_kill_fraction(&model, ZEROBIAS_DENSE_DEFAULT)?;
    let (zerobias_bulk, auto_sparse) = resolve_zerobias(a.zerobias_bulk, killed, live);
    eprintln!(
        "zerobias line-kill preview @ {ZEROBIAS_DENSE_DEFAULT}: {killed}/{live} live layer-0 lines"
    );
    if auto_sparse {
        eprintln!(
            "pack: SPARSE-CLASS bake — the default zerobias {ZEROBIAS_DENSE_DEFAULT} would kill \
             {killed} of {live} live layer-0 lines outright; defaulting --zerobias-bulk to 0 \
             (measured damage on this class: T.R11 −0.0069 CID22, J.R3 57→3 rows). \
             Pass --zerobias-bulk {ZEROBIAS_DENSE_DEFAULT} explicitly to override."
        );
    }
    let (mut packed, counts) = pack_layers(&model, dtype, zerobias_bulk, a.protect_last)?;
    eprintln!(
        "per-layer zerobias (zeroed/total): {}",
        counts
            .iter()
            .map(|(li, z, t)| format!("L{li}:{z}/{t}"))
            .collect::<Vec<_>>()
            .join(" ")
    );

    // metadata = input order minus the spline; the refit spline is appended
    // at the END (matches the Python's `md2 = md + [spline]`).
    let mut md_nospline: Vec<OwnedMeta> = clone_metadata(&model)
        .into_iter()
        .filter(|m| m.key != SPLINE_KEY)
        .collect();

    // The un-pruned packed network — kept as the identity-gate reference
    // even when pruning fires, so the gate compares like for like (both
    // sides post-zerobias, pre-spline).
    let mut scaler_mean = model.scaler_mean().to_vec();
    let mut scaler_scale = model.scaler_scale().to_vec();
    let reference_bytes = emit_packed(
        model.schema_hash(),
        &scaler_mean,
        &scaler_scale,
        &packed,
        &md_nospline,
    );

    // Caller width never changes — pruning is invisible to callers, which
    // is the entire contract. Read the anchor at this width both times.
    let n_in = model.caller_input_width();
    let (feats, tgt) = read_features(&a.anchor, &a.feat_prefix, n_in, &a.target_col);
    let reference_preds = forward_scored_6dec(&reference_bytes, &feats)?;

    // ── dead-column pruning (zerobias → PRUNE → dtype/quantize → spline) ──
    //
    // Zerobias is what creates most weight-dead columns, so the plan is
    // built on the POST-zerobias weights. The spline still lands last, on
    // the final packed net, preserving QUANTIZE-then-CALIBRATE.
    let plan = if a.no_prune {
        None
    } else {
        let l0 = prune::Layer0View {
            in_dim: packed[0].in_dim,
            out_dim: packed[0].out_dim,
            weights: &packed[0].weights,
            biases: &packed[0].biases,
            is_i8: matches!(packed[0].dtype, WeightDtype::I8),
        };
        let p = prune::plan(&model, &l0, !a.no_prune_constants).map_err(|e| e.to_string())?;
        if p.is_noop() { None } else { Some(p) }
    };

    if let Some(p) = &plan {
        // WeightDtype is #[non_exhaustive]; a future dtype falls back to
        // f32 sizing, which only affects the reported byte count.
        let dtype_bytes = match packed[0].dtype {
            WeightDtype::F16 => 2,
            WeightDtype::I8 => 1,
            _ => 4,
        };
        eprint!("{}", p.report(packed[0].out_dim, dtype_bytes));

        let out_dim = packed[0].out_dim;
        packed[0].weights = prune::prune_layer0_weights(p, &packed[0].weights, out_dim);
        packed[0].biases = prune::prune_layer0_biases(p, &packed[0].biases);
        packed[0].in_dim = p.n_inputs_after;
        scaler_mean = prune::prune_input_array(p, &scaler_mean);
        scaler_scale = prune::prune_input_array(p, &scaler_scale);

        // Rewrite the two line-aligned transform metadata entries in place
        // (or append them when the bake carried none).
        let (t_txt, p_txt) = prune::transform_metadata(p);
        set_meta_utf8(&mut md_nospline, FEATURE_TRANSFORMS_KEY, t_txt.into_bytes());
        set_meta_utf8(
            &mut md_nospline,
            FEATURE_TRANSFORM_PARAMS_KEY,
            p_txt.into_bytes(),
        );
    } else if !a.no_prune {
        eprintln!("prune: no dead columns found (every layer-0 input has a live weight)");
    }

    // 1. packed network WITHOUT spline -> its raw (tanh-pin) outputs.
    // With no plan the reference IS the packed net, so reuse it rather than
    // re-baking and re-forwarding identical bytes.
    let preds = match &plan {
        Some(p) => {
            let bytes = emit_packed(
                model.schema_hash(),
                &scaler_mean,
                &scaler_scale,
                &packed,
                &md_nospline,
            );
            let scored = forward_scored_6dec(&bytes, &feats)?;
            check_prune_identity(p, &reference_preds, &scored, a)?;
            scored
        }
        None => reference_preds,
    };
    let pmin = preds.iter().copied().fold(f64::INFINITY, f64::min);
    let pmax = preds.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "packed tanh-pin range [{pmin:.4},{pmax:.4}] corr={:.4}",
        zensim_validate::panel::pearson(&preds, &tgt)
    );

    // 2. fit spline ON THE PACKED NETWORK (re-anchors identity).
    //
    // D4 (ADD156 ship audit): omitting `--neg-tail` here used to SILENTLY
    // delete the dial's negative tail. The prune identity gate above cannot
    // see it — it compares the NETWORK's raw outputs on in-domain anchor rows
    // and reports "BIT-identical" while the spline underneath is being refit
    // with a flat bottom. So when the choice is material, refuse rather than
    // pick one for the caller.
    if let Some((n_flat, n_dedup)) =
        zensim_validate::dial_spline::neg_tail_is_material(&preds, &tgt, 18)
    {
        if !a.neg_tail && !a.no_neg_tail {
            return Err(format!(
                "the --neg-tail choice CHANGES this bake's spline ({n_flat} knots without the \
                 dedup vs {n_dedup} with it) and there is no safe default, so pack refuses to \
                 guess.\n  \
                 Without the dedup the bottom segment is FLAT at y=0: every prediction in that \
                 range maps to exactly 0.0 and the extrapolation below the bottom knot has \
                 slope 0, so the NEGATIVE TAIL the product contract requires is deleted. \
                 Measured on ADD156: dial p5 -12.4334 -> 0.0000 and up to -0.021 SROCC \
                 (LIVE 0.9602 -> 0.9397).\n  \
                 Pass --neg-tail to preserve the tail (what every shipped packed bake used), \
                 or --no-neg-tail to accept the flat bottom deliberately (reproduces the \
                 pre-2026-08-31 default byte-for-byte)."
            ));
        }
    }
    let (cx, cy) = fit_spline_knots(&preds, &tgt, 18, a.neg_tail);
    if cx.len() < 2 {
        return Err(format!(
            "packed-network spline fit produced only {} knots (<2)",
            cx.len()
        ));
    }

    // 3. inject the spline -> final.
    let mut md_final = md_nospline;
    md_final.push(OwnedMeta {
        key: SPLINE_KEY.to_string(),
        kind: MetadataType::Bytes,
        value: spline_payload(&cx, &cy),
    });
    let final_bytes = emit_packed(
        model.schema_hash(),
        &scaler_mean,
        &scaler_scale,
        &packed,
        &md_final,
    );
    std::fs::write(&a.out, &final_bytes).map_err(|e| format!("write {:?}: {e}", a.out))?;
    // D4 second half: the identity story must COVER THE DIAL TAIL. The prune
    // gate speaks only for the network on in-domain anchor rows; a run that
    // deleted the tail used to print nothing but "BIT-identical". Score the
    // input and the packed bake END-TO-END (through their own splines) and
    // state what happened to the negative tail, on every pack.
    match forward_scored_6dec(&bytes, &feats) {
        Ok(in_scores) => match forward_scored_6dec(&final_bytes, &feats) {
            Ok(out_scores) => {
                let rep = dial_tail_report(&cx, &cy, &in_scores, &out_scores);
                eprintln!("{}", rep.render());
            }
            Err(e) => eprintln!("dial tail: NOT CHECKED — scoring the packed bake failed: {e}"),
        },
        Err(e) => eprintln!("dial tail: NOT CHECKED — scoring the input bake failed: {e}"),
    }
    eprintln!(
        "packed {:?} ({} B) -> {:?} ({} B); {} spline knots, dial y-range [{:.1},{:.1}]",
        a.input,
        bytes.len(),
        a.out,
        final_bytes.len(),
        cx.len(),
        cy[0],
        cy[cy.len() - 1]
    );

    // advisory verification: post-spline (unclamped) SROCC + calibration
    // percentiles on the verify corpus.
    if a.verify != "none" {
        let vpath = PathBuf::from(&a.verify);
        let (vfeats, vref) = read_features(&vpath, &a.feat_prefix, n_in, &a.verify_col);
        let vpred = forward_scored_6dec(&final_bytes, &vfeats)?;
        let vref_scaled: Vec<f64> = vref.iter().map(|&t| t * a.verify_scale).collect();
        let mut sorted = vpred.clone();
        sorted.sort_by(f64::total_cmp);
        eprintln!(
            "verify {}: SROCC (post-spline)={:.4}  cal pctl p5={:.1} p95={:.1}  (n={})",
            vpath.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
            spearman(&vpred, &vref_scaled),
            percentile_linear(&sorted, 5.0),
            percentile_linear(&sorted, 95.0),
            vpred.len()
        );
    }

    let got = sha256_hex(&final_bytes);
    eprintln!("  sha256 {got}");
    if let Some(expect) = &a.expect_sha256 {
        if got.starts_with(expect) {
            eprintln!("  BYTE-REPRODUCED (matches expected {expect})");
        } else {
            return Err(format!("sha mismatch: expected {expect}, got {got}"));
        }
    }
    Ok(())
}

// --------------------------------------------------------------------------
// subcommand: strip  (reproduces strip_spline_metadata.py)
// --------------------------------------------------------------------------

#[derive(Args)]
struct StripArgs {
    #[arg(long = "in")]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// Metadata key to remove.
    #[arg(long, default_value = SPLINE_KEY)]
    key: String,
}

#[derive(Args)]
struct AppendMetaArgs {
    #[arg(long = "in")]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// Metadata key to add (refused if already present — append, never
    /// overwrite; strip first to replace).
    #[arg(long)]
    key: String,
    /// File whose UTF-8 contents become the value.
    #[arg(long)]
    value_file: PathBuf,
}

fn cmd_append_meta(a: &AppendMetaArgs) -> Result<(), String> {
    let bytes = std::fs::read(&a.input).map_err(|e| format!("read {:?}: {e}", a.input))?;
    let model = Model::from_bytes(&bytes).map_err(|e| format!("parse bake: {e:?}"))?;
    if model.metadata().iter().any(|m| m.key == a.key) {
        return Err(format!(
            "metadata key {:?} already present in {:?} — append never overwrites;              `strip --key` first to replace",
            a.key, a.input
        ));
    }
    let value = std::fs::read_to_string(&a.value_file)
        .map_err(|e| format!("read {:?}: {e}", a.value_file))?;
    let out_bytes = zenpredict_bake::append_metadata_utf8(&bytes, &a.key, &value)
        .map_err(|e| format!("append_metadata_utf8: {e:?}"))?;
    std::fs::write(&a.out, &out_bytes).map_err(|e| format!("write {:?}: {e}", a.out))?;
    eprintln!(
        "appended {:?} ({} B value) -> {:?} ({} B)",
        a.key,
        value.len(),
        a.out,
        out_bytes.len()
    );
    Ok(())
}

fn cmd_strip(a: &StripArgs) -> Result<(), String> {
    let bytes = std::fs::read(&a.input).map_err(|e| format!("read {:?}: {e}", a.input))?;
    let model = Model::from_bytes(&bytes).map_err(|e| format!("parse bake: {e:?}"))?;
    if !model.feature_bounds().is_empty()
        || !model.output_specs().is_empty()
        || !model.discrete_sets().is_empty()
        || !model.sparse_overrides().is_empty()
    {
        return Err(
            "bake carries feature_bounds/output_specs/discrete_sets/sparse_overrides — \
             strip does not round-trip those sections yet (fail-loud beats the \
             Python pipeline's silent drop)"
                .into(),
        );
    }
    let before = model.metadata().len();
    let metadata: Vec<OwnedMeta> = clone_metadata(&model)
        .into_iter()
        .filter(|m| m.key != a.key)
        .collect();
    if metadata.len() == before {
        return Err(format!(
            "metadata key {:?} not present in {:?} (keys: {})",
            a.key,
            a.input,
            model
                .metadata()
                .iter()
                .map(|e| e.key)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    // schema_hash preserved, flags 0 + compressed true — the same pipeline
    // contract as `pack` (and the Python it replaces).
    let sz = emit_full(&a.out, &model, &metadata, model.schema_hash())
        .map_err(|e| format!("write {:?}: {e}", a.out))?;
    eprintln!(
        "stripped {:?} ({} -> {} metadata entries) -> {:?} ({sz} B)",
        a.key,
        before,
        metadata.len(),
        a.out
    );
    Ok(())
}

// --------------------------------------------------------------------------
// subcommand: fit-lasso  (reproduces linear_projections_2026-07-03.py
// `MixGram.lasso` + `bake_candidate` — the shipped-BHdr fit chain, task #68)
// --------------------------------------------------------------------------

#[derive(Args)]
struct FitLassoArgs {
    /// Frozen per-group Gram .npz (keys `<space>__S`, `<space>__s`,
    /// `<space>__n`, `<space>__q_<target>`, `<space>__Y1_<target>` — the
    /// exact artifact the Python `cmd_gram` / the `gram` subcommand wrote).
    /// READ-ONLY; bit-exactness vs a Python fit requires consuming the same
    /// frozen artifact, never re-assembling its Gram from parquets (BLAS
    /// accumulation order differs). REPEATABLE: multiple grams accumulate
    /// `S += w·S_z` in argv order (MixGram multi-group semantics); pair each
    /// with a `--weight`.
    #[arg(long, required = true)]
    gram: Vec<PathBuf>,
    /// Feature-space prefix inside the gram / anchor npz.
    #[arg(long, default_value = "shaped")]
    space: String,
    /// Target column name (selects `__q_<target>` / `__Y1_<target>`).
    #[arg(long, default_value = "human_score")]
    target: String,
    /// Per-gram target-name override (REPEATABLE, paired with `--gram` in
    /// order) for mixing corpora whose grams store their q/Y1 under
    /// different column names but the same target CLASS (E-LIN: legs =
    /// `human_score`, kadis = `score_ssim2_gpu`, bigcodec = `score_ssim2`).
    /// Omit entirely to use `--target` for every gram.
    #[arg(long)]
    gram_target: Vec<String>,
    /// Mix weight per `--gram`, in the same order (1.0 = exact
    /// pass-through, the shipped-BHdr single-group case). Omit entirely for
    /// all-1.0; otherwise give exactly one per gram.
    #[arg(long)]
    weight: Vec<f64>,
    /// L1 penalty on the mean-loss scale (Python `lasso{lam}`).
    #[arg(long)]
    lam: f64,
    /// Coordinate-descent sweep cap (Python `n_sweeps`).
    #[arg(long, default_value_t = 200)]
    n_sweeps: usize,
    /// Sweep max-|Δw| convergence threshold (Python `tol`).
    #[arg(long, default_value_t = 1e-10)]
    tol: f64,
    /// Zero weights with |w| < tau BEFORE the f16 pack (Python `--taus`).
    #[arg(long, default_value_t = 0.0)]
    tau: f64,
    /// Anchor .npz (`raw`/`shaped` f32 matrices + `y`) — the dial spline is
    /// fit on the PACKED forward over these rows. Mutually exclusive with
    /// `--anchor-parquet`.
    #[arg(long)]
    anchor: Option<PathBuf>,
    /// Parquet anchor alternative (REPEATABLE): rows are loaded through
    /// `parquet_loader::load_parquet` (THE loader owner), stride-sampled
    /// (`--anchor-stride`, paired per parquet, default 1), concatenated in
    /// argv order; `y` = `--anchor-target` × `--anchor-scale`, optionally
    /// clamped to ≥ `--anchor-clip-min`. Features are f32-rounded first so
    /// the packed forward matches the npz-anchor numeric path (and the f32
    /// runtime).
    #[arg(long)]
    anchor_parquet: Vec<PathBuf>,
    /// Target column for `--anchor-parquet` rows.
    #[arg(long)]
    anchor_target: Option<String>,
    /// Scale applied to the anchor target (e.g. 100 for [0,1]-scale legs).
    #[arg(long, default_value_t = 1.0)]
    anchor_scale: f64,
    /// Row stride per `--anchor-parquet`, same order (rows 0, s, 2s, …).
    /// Omit entirely for all-1.
    #[arg(long)]
    anchor_stride: Vec<usize>,
    /// Clamp anchor targets to at least this value (post-scale).
    #[arg(long, allow_negative_numbers = true)]
    anchor_clip_min: Option<f64>,
    /// Transform screen TSV (`feat_idx`/`best_transform`/`params_csv`)
    /// providing the `zentrain.feature_transform*` metadata TEXT. Required
    /// for `--space shaped`; the params are re-emitted in CPython float-repr
    /// form because that text is part of the bake bytes.
    #[arg(long)]
    transforms_tsv: Option<PathBuf>,
    /// Output bake path.
    #[arg(long)]
    out: PathBuf,
    /// Parity gate 1: a Python `fits/*.npz` (`w`/`bias`/`mu`/`sd`) that the
    /// Rust fit must match BIT-EXACTLY (errors on any f64 mismatch).
    #[arg(long)]
    parity_fit: Option<PathBuf>,
    /// If given, assert the output sha256 begins with this hex prefix.
    #[arg(long)]
    expect_sha256: Option<String>,
    /// Solver: `lasso` (default; L1 coordinate descent) or `bvls`
    /// (box-constrained CD — the B kon-head class; SOTA-944 §3e/§4). With
    /// `bvls`, `--lam` is ignored and `--bounds-tsv` supplies the box.
    #[arg(long, default_value = "lasso")]
    solver: String,
    /// Sign-mask TSV for `--solver bvls` (`feat_idx`/`sign_mask` with
    /// `pin_geq0`/`free` rows — `benchmarks/feature_sign_mask_2026-05-26.tsv`).
    /// Features absent from the TSV (e.g. f372+ at 944) are FREE.
    #[arg(long)]
    bounds_tsv: Option<PathBuf>,
    /// Also write the pre-pack fit as an npz (`w`/`bias`/`mu`/`sd`, f64) —
    /// the head artifact `blend-heads` consumes.
    #[arg(long)]
    emit_fit_npz: Option<PathBuf>,
    /// Embed a `zentrain.repro` metadata entry (argv + gram shas + code
    /// commit) via the canonical `zenpredict_bake::append_metadata_utf8`.
    /// OPT-IN so the byte-repro paths (BHdr `--expect-sha256`) stay
    /// byte-identical; the SOTA-944 driver always passes it (embed failure
    /// is fatal, exit-4 class).
    #[arg(long)]
    embed_repro: bool,
    /// Take the LEADING `n_feat` columns of a wider `--anchor-parquet` — the
    /// opt-in companion of `gram --max-feat`, for the case where the same table
    /// served the gram at a narrower width (hybrid lane PART II: a 372-wide
    /// student fit on the leading block of a 944-wide root). Refused unless the
    /// anchor is strictly wider; never pads.
    #[arg(long)]
    anchor_prefix: bool,
    /// Coordinate-slice file (newline-separated feature indices): the
    /// ADD156-class `w[out-of-slice] = 0` constraint — CD sweeps ONLY these
    /// coordinates (SOTA-944 §3a spatializable slices). Omit for all.
    #[arg(long)]
    slice_file: Option<PathBuf>,
}

/// Parse a newline-separated feature-index slice file (comments `#` ok).
fn load_slice_file(path: &Path, n_feat: usize) -> Result<Vec<usize>, String> {
    let txt = std::fs::read_to_string(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let mut idx = Vec::new();
    for (ln, line) in txt.lines().enumerate() {
        let t = line.split('#').next().unwrap_or("").trim();
        if t.is_empty() {
            continue;
        }
        let i: usize = t
            .parse()
            .map_err(|e| format!("{path:?}:{}: bad index {t:?}: {e}", ln + 1))?;
        if i >= n_feat {
            return Err(format!("{path:?}:{}: index {i} >= n_feat {n_feat}", ln + 1));
        }
        idx.push(i);
    }
    if idx.is_empty() {
        return Err(format!("{path:?}: empty slice"));
    }
    Ok(idx)
}

/// Load a sign-mask TSV into z-space box bounds (`pin_geq0` → [0, +inf);
/// `free`/absent → (−inf, +inf)). Sign is preserved by standardization
/// (`w_raw = w_z / sd`, `sd > 0`), so raw-space pins ARE z-space pins.
fn load_sign_bounds(path: &Path, n_feat: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    let mut lo = vec![f64::NEG_INFINITY; n_feat];
    let hi = vec![f64::INFINITY; n_feat];
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(path)
        .map_err(|e| format!("open bounds TSV {path:?}: {e}"))?;
    let headers = rdr
        .headers()
        .map_err(|e| format!("bounds TSV header: {e}"))?
        .clone();
    let c_idx = headers
        .iter()
        .position(|h| h == "feat_idx")
        .ok_or("bounds TSV missing feat_idx")?;
    let c_mask = headers
        .iter()
        .position(|h| h == "sign_mask")
        .ok_or("bounds TSV missing sign_mask")?;
    let mut n_pinned = 0usize;
    for rec in rdr.records() {
        let rec = rec.map_err(|e| format!("bounds TSV row: {e}"))?;
        let fi: usize = rec
            .get(c_idx)
            .ok_or("bounds TSV row missing feat_idx")?
            .trim()
            .parse()
            .map_err(|e| format!("bad feat_idx: {e}"))?;
        if fi >= n_feat {
            continue;
        }
        match rec.get(c_mask).unwrap_or("").trim() {
            "pin_geq0" => {
                lo[fi] = 0.0;
                n_pinned += 1;
            }
            "free" | "" => {}
            other => return Err(format!("bounds TSV feat {fi}: unknown sign_mask {other:?}")),
        }
    }
    eprintln!("  bounds: {n_pinned} features pinned >= 0, rest free (of {n_feat})");
    Ok((lo, hi))
}

/// Embed `zentrain.repro` into an emitted bake through the canonical
/// append owner (self-checked loadable; byte/score-identity gated by
/// zenpredict-bake's own tests). Fatal on failure — a campaign bake
/// without embedded repro is an invalid artifact (SOTA-944 bar).
fn embed_repro_into(out: &Path, repro_json: &str) -> Result<(), String> {
    let bytes = std::fs::read(out).map_err(|e| format!("re-read {out:?}: {e}"))?;
    let appended = zenpredict_bake::append_metadata_utf8(&bytes, "zentrain.repro", repro_json)
        .map_err(|e| format!("embed zentrain.repro FAILED (fatal): {e:?}"))?;
    std::fs::write(out, &appended).map_err(|e| format!("write {out:?}: {e}"))?;
    Ok(())
}

/// Parse the yeo-johnson screen TSV exactly like the Python loaders
/// (`train_v02_bvls_shaped.load_transforms`): rows keyed by `feat_idx`,
/// missing features stay `identity` with no params, `params_csv` split on
/// `,` and parsed as f64 (Python `float()` and Rust `str::parse` are both
/// correctly rounded, so the VALUES are identical; the repr difference is
/// handled at emission by `py_repr_f64`).
fn load_transform_screen(
    path: &Path,
    n_feat: usize,
) -> Result<(Vec<String>, Vec<Vec<f64>>), String> {
    let mut toks = vec!["identity".to_string(); n_feat];
    let mut params = vec![Vec::<f64>::new(); n_feat];
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(path)
        .map_err(|e| format!("open screen TSV {path:?}: {e}"))?;
    let headers = rdr
        .headers()
        .map_err(|e| format!("screen TSV header: {e}"))?
        .clone();
    let col = |name: &str| {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| format!("screen TSV missing column {name:?}"))
    };
    let c_idx = col("feat_idx")?;
    let c_tok = col("best_transform")?;
    let c_par = col("params_csv")?;
    for rec in rdr.records() {
        let rec = rec.map_err(|e| format!("screen TSV row: {e}"))?;
        let fi: usize = rec
            .get(c_idx)
            .ok_or("screen TSV row missing feat_idx")?
            .trim()
            .parse()
            .map_err(|e| format!("bad feat_idx: {e}"))?;
        if fi >= n_feat {
            continue;
        }
        toks[fi] = rec.get(c_tok).unwrap_or("").to_string();
        let cp = rec.get(c_par).unwrap_or("");
        if !cp.trim().is_empty() {
            params[fi] = cp
                .split(',')
                .map(|t| {
                    t.trim()
                        .parse::<f64>()
                        .map_err(|e| format!("bad param {t:?} for feat {fi}: {e}"))
                })
                .collect::<Result<Vec<f64>, String>>()?;
        }
    }
    Ok((toks, params))
}

fn cmd_fit_lasso(a: &FitLassoArgs) -> Result<(), String> {
    use zensim_validate::gram_lasso::{
        GramGroup, f16_bits_to_f64, f64_to_f16_bits, py_repr_f64, standardize_gram_multi,
    };
    use zensim_validate::npz::Npz;

    // 1. frozen Gram moments → standardized system (MixGram.__init__;
    // multiple grams accumulate `S += w·S_z` in argv order).
    let weights: Vec<f64> = if a.weight.is_empty() {
        vec![1.0; a.gram.len()]
    } else if a.weight.len() == a.gram.len() {
        a.weight.clone()
    } else {
        return Err(format!(
            "--weight count {} != --gram count {} (omit for all-1.0, else one per gram)",
            a.weight.len(),
            a.gram.len()
        ));
    };
    let key = |suffix: &str| format!("{}__{suffix}", a.space);
    struct LoadedGram {
        s_arr: zensim_validate::npz::NpyArray,
        s_vec: zensim_validate::npz::NpyArray,
        q_arr: zensim_validate::npz::NpyArray,
        y1: f64,
        n_rows: f64,
    }
    let mut loaded: Vec<LoadedGram> = Vec::with_capacity(a.gram.len());
    let mut n_feat = 0usize;
    for (gi, gpath) in a.gram.iter().enumerate() {
        let gram = Npz::open(gpath)?;
        let gtarget: &str = if a.gram_target.is_empty() {
            &a.target
        } else if a.gram_target.len() == a.gram.len() {
            &a.gram_target[gi]
        } else {
            return Err(format!(
                "--gram-target count {} != --gram count {} (omit entirely, or one per gram)",
                a.gram_target.len(),
                a.gram.len()
            ));
        };
        let s_arr = gram.get(&key("S"))?;
        let s_vec = gram.get(&key("s"))?;
        let q_arr = gram.get(&key(&format!("q_{gtarget}")))?;
        let y1 = gram.get(&key(&format!("Y1_{gtarget}")))?.scalar_f64()?;
        let n_rows = gram.get(&key("n"))?.scalar_f64()?;
        let nf = *s_vec
            .shape
            .first()
            .ok_or("gram `__s` must be 1-d, got 0-d")?;
        if s_arr.shape != [nf, nf] {
            return Err(format!(
                "gram `__S` shape {:?} != ({nf}, {nf}) in {gpath:?}",
                s_arr.shape
            ));
        }
        if gi == 0 {
            n_feat = nf;
        } else if nf != n_feat {
            return Err(format!(
                "gram width mismatch: {gpath:?} has n_feat {nf}, first gram has {n_feat}"
            ));
        }
        loaded.push(LoadedGram {
            s_arr,
            s_vec,
            q_arr,
            y1,
            n_rows,
        });
    }
    let mut groups: Vec<GramGroup<'_>> = Vec::with_capacity(loaded.len());
    for (g, w) in loaded.iter().zip(&weights) {
        groups.push(GramGroup {
            weight: *w,
            s_mat: g.s_arr.f64s()?,
            s_vec: g.s_vec.f64s()?,
            q: g.q_arr.f64s()?,
            y1: g.y1,
            n_rows: g.n_rows,
        });
    }
    let sg = standardize_gram_multi(n_feat, &groups)?;

    // 2. solver: lasso coordinate descent (MixGram.lasso) or the BVLS-class
    // box-constrained CD (SOTA-944 §3e — sign-mask bounds).
    let slice_idx: Option<Vec<usize>> = match &a.slice_file {
        Some(p) => {
            let idx = load_slice_file(p, n_feat)?;
            eprintln!(
                "  slice: {} of {n_feat} coordinates active ({:?})",
                idx.len(),
                p
            );
            Some(idx)
        }
        None => None,
    };
    let w = match a.solver.as_str() {
        "lasso" => zensim_validate::gram_lasso::lasso_cd_slice(
            &sg,
            a.lam,
            a.n_sweeps,
            a.tol,
            slice_idx.as_deref(),
        ),
        "bvls" => {
            let (lo, hi) = match &a.bounds_tsv {
                Some(p) => load_sign_bounds(p, n_feat)?,
                None => (vec![f64::NEG_INFINITY; n_feat], vec![f64::INFINITY; n_feat]),
            };
            zensim_validate::gram_lasso::box_cd_slice(
                &sg,
                &lo,
                &hi,
                a.n_sweeps,
                a.tol,
                slice_idx.as_deref(),
            )
        }
        other => return Err(format!("--solver must be lasso|bvls, got {other:?}")),
    };
    let bias = sg.ybar;
    let n_active_pre = w.iter().filter(|v| v.abs() > 1e-7).count();
    eprintln!(
        "{}(lam={}) on {} gram(s) {:?} w={:?} [{} space, target {}]: W={:.0} act={n_active_pre} bias={bias:.6}",
        a.solver,
        a.lam,
        a.gram.len(),
        a.gram,
        weights,
        a.space,
        a.target,
        sg.w_total
    );

    // Head artifact for blend-heads: pre-pack f64 fit (w/bias/mu/sd).
    if let Some(npz_path) = &a.emit_fit_npz {
        use zensim_validate::npz::{NpzF64Entry, write_npz_f64};
        let shape1 = [n_feat];
        let bias_s = [bias];
        write_npz_f64(
            npz_path,
            &[
                NpzF64Entry {
                    name: "w",
                    shape: &shape1,
                    data: &w,
                },
                NpzF64Entry {
                    name: "bias",
                    shape: &[],
                    data: &bias_s,
                },
                NpzF64Entry {
                    name: "mu",
                    shape: &shape1,
                    data: &sg.mu,
                },
                NpzF64Entry {
                    name: "sd",
                    shape: &shape1,
                    data: &sg.sd,
                },
            ],
        )?;
        eprintln!("  fit npz -> {npz_path:?}");
    }

    // 3. parity gate 1: bit-exact w/bias/mu/sd vs the Python fit npz.
    if let Some(pf) = &a.parity_fit {
        let fit = Npz::open(pf)?;
        let pw_arr = fit.get("w")?;
        let pmu_arr = fit.get("mu")?;
        let psd_arr = fit.get("sd")?;
        let pbias = fit.get("bias")?.scalar_f64()?;
        let mut bad = 0usize;
        let mut check = |name: &str, ours: &[f64], theirs: &[f64]| {
            if ours.len() != theirs.len() {
                eprintln!("  parity {name}: length {} != {}", ours.len(), theirs.len());
                bad += 1;
                return;
            }
            for (i, (o, t)) in ours.iter().zip(theirs).enumerate() {
                if o.to_bits() != t.to_bits() {
                    if bad < 8 {
                        eprintln!(
                            "  parity {name}[{i}]: rust {o:?} ({:#018x}) != python {t:?} ({:#018x})",
                            o.to_bits(),
                            t.to_bits()
                        );
                    }
                    bad += 1;
                }
            }
        };
        check("w", &w, pw_arr.f64s()?);
        check("mu", &sg.mu, pmu_arr.f64s()?);
        check("sd", &sg.sd, psd_arr.f64s()?);
        if pbias.to_bits() != bias.to_bits() {
            eprintln!(
                "  parity bias: rust {bias:?} ({:#018x}) != python {pbias:?} ({:#018x})",
                bias.to_bits(),
                pbias.to_bits()
            );
            bad += 1;
        }
        if bad > 0 {
            return Err(format!(
                "parity gate 1 FAILED: {bad} f64 mismatches vs {pf:?}"
            ));
        }
        eprintln!("  parity gate 1 PASS: w/bias/mu/sd bit-exact vs {pf:?}");
    }

    // 4. tau prune + f16 pack (bake_candidate: `w[|w| < tau] = 0`, then
    // `w.astype(f16).astype(f64)` — single-rounding f64→f16).
    let mut w = w;
    if a.tau > 0.0 {
        for v in &mut w {
            if v.abs() < a.tau {
                *v = 0.0;
            }
        }
    }
    let wp: Vec<f64> = w
        .iter()
        .map(|v| f16_bits_to_f64(f64_to_f16_bits(*v)))
        .collect();
    let n_active = wp.iter().filter(|v| v.abs() > 0.0).count();

    // 5. dial spline on the PACKED forward over the anchor (npz matrix, or
    // parquet rows through the canonical loader).
    let (xaf, ya, rows): (Vec<f32>, Vec<f64>, usize) = match (&a.anchor, a.anchor_parquet.len()) {
        (Some(_), n) if n > 0 => {
            return Err("--anchor and --anchor-parquet are mutually exclusive".into());
        }
        (Some(anchor_path), _) => {
            let anchor = Npz::open(anchor_path)?;
            let xa = anchor.get(&a.space)?;
            let (rows, cols) = match xa.shape[..] {
                [r, c] => (r, c),
                _ => {
                    return Err(format!(
                        "anchor {:?} entry {:?} must be 2-d, got {:?}",
                        anchor_path, a.space, xa.shape
                    ));
                }
            };
            if cols != n_feat {
                return Err(format!(
                    "anchor width {cols} != gram n_feat {n_feat} — wrong anchor for this gram"
                ));
            }
            let xaf = xa.f32s()?.to_vec();
            let ya = anchor.get("y")?.f64s()?.to_vec();
            if ya.len() != rows {
                return Err(format!("anchor y len {} != rows {rows}", ya.len()));
            }
            (xaf, ya, rows)
        }
        (None, 0) => {
            return Err("one of --anchor / --anchor-parquet is required".into());
        }
        (None, np) => {
            let target = a.anchor_target.as_ref().ok_or(
                "--anchor-target is required with --anchor-parquet (the dial target column)",
            )?;
            let strides: Vec<usize> = if a.anchor_stride.is_empty() {
                vec![1; np]
            } else if a.anchor_stride.len() == np {
                a.anchor_stride.clone()
            } else {
                return Err(format!(
                    "--anchor-stride count {} != --anchor-parquet count {np}",
                    a.anchor_stride.len()
                ));
            };
            if strides.contains(&0) {
                return Err("--anchor-stride must be >= 1".into());
            }
            // SHAPED space + raw parquet rows: the gram (and therefore the
            // weights) live in transform space, so the anchor forward MUST
            // apply the same per-feature transforms the runtime will (the
            // npz-anchor path stored pre-shaped matrices; parquet rows are
            // raw). Applied at f32 via zenpredict's own math — identical to
            // the deployed `predict_transformed` feature path.
            let anchor_appliers: Option<Vec<(zenpredict::FeatureTransform, Vec<f32>)>> =
                if a.space == "shaped" {
                    let tsv = a.transforms_tsv.as_ref().ok_or(
                        "--transforms-tsv is required for --space shaped with --anchor-parquet \
                         (the anchor forward must mirror the shaped gram)",
                    )?;
                    Some(screen_appliers(tsv, n_feat)?)
                } else {
                    None
                };
            let mut xaf: Vec<f32> = Vec::new();
            let mut ya: Vec<f64> = Vec::new();
            for (path, &stride) in a.anchor_parquet.iter().zip(&strides) {
                let g = zensim_validate::parquet_loader::load_parquet(
                    path,
                    "anchor",
                    target,
                    a.anchor_scale,
                )?;
                if g.n_features != n_feat {
                    // `--anchor-prefix` is the opt-in companion of
                    // `gram --max-feat`: the SAME table served the gram at a
                    // narrower width, so its leading `n_feat` columns are the
                    // fit space by construction. Opt-in and loud, because a
                    // silent prefix of an unrelated table would score garbage.
                    if a.anchor_prefix && g.n_features > n_feat {
                        eprintln!(
                            "  --anchor-prefix: anchor {path:?} is {} wide, taking the leading \
                             {n_feat} columns (the gram's width)",
                            g.n_features
                        );
                    } else {
                        return Err(format!(
                            "anchor parquet {path:?} width {} != gram n_feat {n_feat}\
                             {}",
                            g.n_features,
                            if g.n_features > n_feat {
                                " — pass --anchor-prefix if this is the same table the gram                                  was built from with --max-feat"
                            } else {
                                ""
                            }
                        ));
                    }
                }
                let mut taken = 0usize;
                let mut ri = 0usize;
                while ri < g.feature_rows.len() {
                    // f32-round the features so the packed forward matches
                    // the npz-anchor numeric path (and the f32 runtime).
                    match &anchor_appliers {
                        Some(ap) => xaf.extend(
                            g.feature_rows[ri][..n_feat]
                                .iter()
                                .zip(ap.iter())
                                .map(|(v, (t, ps))| t.apply_with_params(*v as f32, ps)),
                        ),
                        None => xaf.extend(g.feature_rows[ri][..n_feat].iter().map(|v| *v as f32)),
                    }
                    let mut y = g.human_scores[ri];
                    if let Some(clip) = a.anchor_clip_min
                        && y < clip
                    {
                        y = clip;
                    }
                    ya.push(y);
                    taken += 1;
                    ri += stride;
                }
                eprintln!(
                    "  anchor: {taken} rows (stride {stride}) from {path:?} [target {target} x{}]",
                    a.anchor_scale
                );
            }
            let rows = ya.len();
            if rows < 50 {
                return Err(format!(
                    "anchor has only {rows} rows — too few to fit an 18-knot dial spline"
                ));
            }
            (xaf, ya, rows)
        }
    };
    let xaf: &[f32] = &xaf;
    let mut preds = vec![0.0f64; rows];
    for (r, pred) in preds.iter_mut().enumerate() {
        let row = &xaf[r * n_feat..(r + 1) * n_feat];
        let mut acc = 0.0f64;
        for j in 0..n_feat {
            // mirrors `(Xa - mu) / sd @ w`: widen, subtract, divide, multiply
            // — each rounded once; sequential sum (see the parity note in the
            // repro doc: knots are f32-rounded, which absorbs the BLAS
            // accumulation-order difference).
            acc += (row[j] as f64 - sg.mu[j]) / sg.sd[j] * wp[j];
        }
        *pred = acc + bias;
    }
    let (cx, cy) = fit_spline_knots(&preds, &ya, 18, true);

    // 6. metadata: transforms text (shaped space) + spline payload, in the
    // exact order bake_candidate emits them.
    let mut metadata: Vec<OwnedMeta> = Vec::new();
    if a.space == "shaped" {
        let tsv = a.transforms_tsv.as_ref().ok_or(
            "--transforms-tsv is required for --space shaped (the transform \
             metadata text ships inside the bake bytes)",
        )?;
        let (toks, params) = load_transform_screen(tsv, n_feat)?;
        metadata.push(OwnedMeta {
            key: zenpredict::keys::FEATURE_TRANSFORMS.to_string(),
            kind: MetadataType::Utf8,
            value: toks.join("\n").into_bytes(),
        });
        let params_txt = params
            .iter()
            .map(|row| {
                row.iter()
                    .map(|p| py_repr_f64(*p))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .collect::<Vec<_>>()
            .join("\n");
        metadata.push(OwnedMeta {
            key: zenpredict::keys::FEATURE_TRANSFORM_PARAMS.to_string(),
            kind: MetadataType::Utf8,
            value: params_txt.into_bytes(),
        });
    }
    if !cx.is_empty() {
        metadata.push(OwnedMeta {
            key: SPLINE_KEY.to_string(),
            kind: MetadataType::Bytes,
            value: spline_payload(&cx, &cy),
        });
    }

    // 7. emit through the shared canonical serializer (same BakeRequest
    // shape as the Python JSON pipeline: f16 layer, compressed, flags 0).
    let mu32: Vec<f32> = sg.mu.iter().map(|v| *v as f32).collect();
    let sd32: Vec<f32> = sg.sd.iter().map(|v| *v as f32).collect();
    let w32: Vec<f32> = wp.iter().map(|v| *v as f32).collect();
    let sz = emit_linear(&a.out, &mu32, &sd32, &w32, bias as f32, &metadata)
        .map_err(|e| format!("write {:?}: {e}", a.out))?;

    // Embedded repro (SOTA-944 bar): argv + per-gram sha256 + solver, via
    // the canonical append owner. Fatal on failure.
    if a.embed_repro {
        let argv: Vec<String> = std::env::args().collect();
        let mut gram_shas = Vec::with_capacity(a.gram.len());
        for g in &a.gram {
            let sha = zensim_validate::train_manifest::sha256_file(g)
                .map_err(|e| format!("sha256 {g:?}: {e:?}"))?;
            gram_shas.push(format!(
                "{{\"gram\":{:?},\"sha256\":\"{sha}\"}}",
                g.display().to_string()
            ));
        }
        let repro = format!(
            "{{\"tool\":\"bake_dial_refit fit-lasso\",\"solver\":{:?},\"argv\":{:?},\"grams\":[{}]}}",
            a.solver,
            argv,
            gram_shas.join(",")
        );
        embed_repro_into(&a.out, &repro)?;
        eprintln!("  zentrain.repro embedded ({} B)", repro.len());
    }

    let out_bytes = std::fs::read(&a.out).map_err(|e| format!("re-read {:?}: {e}", a.out))?;
    let got = sha256_hex(&out_bytes);
    eprintln!(
        "fit-lasso -> {:?} ({sz} B): act={n_active} (pre-pack {n_active_pre}), {} spline knots, dial y-range [{:.2},{:.2}]\n  sha256 {got}",
        a.out,
        cx.len(),
        cy.first().copied().unwrap_or(f64::NAN),
        cy.last().copied().unwrap_or(f64::NAN)
    );
    if let Some(expect) = &a.expect_sha256 {
        if got.starts_with(expect) {
            eprintln!("  BYTE-REPRODUCED (matches expected {expect})");
        } else {
            return Err(format!("sha mismatch: expected {expect}, got {got}"));
        }
    }
    Ok(())
}

// --------------------------------------------------------------------------
// subcommand: predict  (transform-safe bake forward over a parquet)
// --------------------------------------------------------------------------

#[derive(Args)]
struct PredictArgs {
    /// Bake to forward (any ZNPR v3, incl. MLPs). With `--ensemble` this is
    /// optional; when both are given it must name one of the members.
    #[arg(long)]
    bake: Option<PathBuf>,
    /// Comma-separated ZNPR bakes scored as ONE equal-weight ensemble: every
    /// row's prediction is the arithmetic mean of the members' raw predictions.
    ///
    /// **⚠ Only with `--score-units` does this mirror `bake_verdict`'s
    /// `Ensemble::score_rows` contract.** Without it the accumulation happens
    /// in RAW network units, which is a DIFFERENT blend whenever the members'
    /// raw scales differ (measured: |SROCC| 0.5218 vs 0.5019 on the `HYA`
    /// pair at w = 0.84). The paragraph below describes the intended
    /// contract, which `--score-units` implements; the historical default
    /// does not, and is kept only so stored recipes reproduce.
    ///
    /// This mirrors `bake_verdict`'s `Ensemble::score_rows` contract exactly —
    /// same averaging order (after each member's own output spline, i.e. in
    /// each member's score units), same k=1 short-circuit (one member runs the
    /// byte-for-byte single-bake path, no `0.0 + x` / `x / 1.0` rounding
    /// surface), same loud failure when members disagree on `n_inputs`.
    /// Averaging TSVs in a script instead is the duplication CLAUDE.md bans:
    /// the teacher a distillation trains against must come from the same
    /// forward the evaluation used.
    #[arg(long, value_delimiter = ',')]
    ensemble: Vec<PathBuf>,
    /// Convex member weights (same length as `--ensemble`, non-negative, at
    /// least one positive; normalised to sum 1). Omitted = the historical
    /// equal-weight mean, whose accumulation is left verbatim. Mirrors
    /// `bake_verdict --ensemble-weights` so a TEACHER target is produced by the
    /// IDENTICAL forward the evaluation scores — which is what the `--ensemble`
    /// doc above already requires and could not deliver for a weighted blend.
    #[arg(long, value_delimiter = ',')]
    ensemble_weights: Vec<f64>,
    /// Feature parquet (f0..fN / feat_0.. + ref_basename).
    #[arg(long)]
    corpus: PathBuf,
    /// Output TSV path (`row_idx<TAB>pred`, file row order — positional
    /// alignment is the join contract).
    ///
    /// **Units: RAW network output by default** — the head/tanh-pin/output-
    /// spline dispatch is NOT applied. Pass `--score-units` for the
    /// `bake_verdict` scoring units. See `--score-units` for why the default
    /// is raw and why it matters for `--ensemble`.
    #[arg(long)]
    out: PathBuf,
    /// Emit **scoring units** instead of raw network output: run the SAME
    /// post-network dispatch `bake_verdict` runs — per-sample-α / hybrid /
    /// min-max head, tanh pin, and the bake's own output-calibration spline —
    /// through the shared owner (`zensim_validate::bake_runtime::score_row`,
    /// which is literally the function `bake_verdict`'s scorer calls).
    ///
    /// **Why this flag exists (measured 2026-09-04).** Without it, a `k >= 2`
    /// `--ensemble` averages members in RAW units while `bake_verdict
    /// --ensemble` averages them in SCORE units, so the two tools disagree on
    /// every blend — and the doc on `--ensemble` claimed they could not. On
    /// the `HYA` pair over 504 KonJND rows the divergence is |SROCC| 0.5390
    /// (bake_verdict) vs 0.5073 (raw) at w = 0.5, and 0.5218 vs 0.5019 at
    /// w = 0.84, because the members' raw scales differ 17x (7.18 vs 0.41)
    /// while their score-unit scales do not. It is invisible at k = 1: a
    /// monotone spline is rank-invariant, so every single-bake SROCC agrees.
    ///
    /// **Default is OFF and byte-identical to every historical invocation**,
    /// because existing teacher-build recipes store affine bounds fitted in
    /// raw units (e.g. `lo = -13.996, hi = 12.711`) and flipping the default
    /// would silently move them. New blended-teacher builds should pass it.
    #[arg(long, default_value_t = false)]
    score_units: bool,
}

fn cmd_predict(a: &PredictArgs) -> Result<(), String> {
    let members: Vec<PathBuf> = if a.ensemble.is_empty() {
        vec![
            a.bake
                .clone()
                .ok_or("predict needs --bake or --ensemble".to_string())?,
        ]
    } else {
        if let Some(b) = a.bake.as_ref()
            && !a.ensemble.contains(b)
        {
            return Err(format!(
                "--bake {} is not one of the --ensemble members; omit --bake",
                b.display()
            ));
        }
        a.ensemble.clone()
    };

    let mut models: Vec<Model> = Vec::with_capacity(members.len());
    for p in &members {
        let bytes = std::fs::read(p).map_err(|e| format!("read {p:?}: {e}"))?;
        models.push(Model::from_bytes(&bytes).map_err(|e| format!("parse bake {p:?}: {e:?}"))?);
    }
    // CALLER width, not the internal layer-0 width (fixed 2026-09-01, hybrid
    // lane). Two facts made the old `n_inputs()` reading wrong on both counts:
    //
    //  * it SIZED the input buffer, so a dead-column-PRUNED bake
    //    (`n_inputs` 667, `caller_input_width` 944) was handed the first 667
    //    columns of a 944-wide row — a prefix, which the repo's own pruning
    //    rule forbids in as many words ("size every feature vector by
    //    caller_input_width()"); and
    //  * it REFUSED a pruned + unpruned pair whose caller widths agree, which
    //    `bake_verdict`'s `Ensemble` accepts — so this function did not in fact
    //    "mirror bake_verdict's contract exactly" as its own doc comment
    //    claims. It does now: same field, same refusal.
    let n_in = models[0].caller_input_width();
    for (p, m) in members.iter().zip(models.iter()).skip(1) {
        if m.caller_input_width() != n_in {
            return Err(format!(
                "ensemble member {p:?} has caller_input_width={} but member 0 has {n_in} — \
                 averaging across feature regimes is the column-mixing this repo bans",
                m.caller_input_width()
            ));
        }
    }
    // Weights: same validation and normalisation as `bake_verdict`.
    let weights: Option<Vec<f64>> = if a.ensemble_weights.is_empty() {
        None
    } else {
        if a.ensemble.is_empty() {
            return Err("--ensemble-weights requires --ensemble".into());
        }
        if a.ensemble_weights.len() != members.len() {
            return Err(format!(
                "--ensemble-weights has {} entries but --ensemble has {} members",
                a.ensemble_weights.len(),
                members.len()
            ));
        }
        if a.ensemble_weights
            .iter()
            .any(|w| !w.is_finite() || *w < 0.0)
        {
            return Err("--ensemble-weights must be finite and >= 0".into());
        }
        let sum: f64 = a.ensemble_weights.iter().sum();
        if !(sum > 0.0) {
            return Err("--ensemble-weights must not sum to zero".into());
        }
        Some(a.ensemble_weights.iter().map(|w| w / sum).collect())
    };
    let g =
        zensim_validate::parquet_loader::load_parquet(&a.corpus, "predict", "human_score", 1.0)?;

    // Per-member forward over every row. k=1 takes exactly the original path
    // (single accumulator, single divide-by-one is skipped below).
    let k = models.len();
    let mut acc = vec![0f64; g.feature_rows.len()];
    for (mi, model) in models.iter().enumerate() {
        let wi = weights.as_ref().map(|w| w[mi]);
        if wi == Some(0.0) {
            continue;
        }
        let transformed = model.has_nontrivial_feature_transforms();
        let mut predictor = zenpredict::Predictor::new(model);
        let mut xbuf = vec![0f32; n_in];
        // `--score-units`: the SAME post-network dispatch `bake_verdict`'s
        // scorer runs, through the shared owner — nothing re-implemented here.
        let (psa, hyb, tanh, ospline, mmh) = if a.score_units {
            (
                zensim_validate::bake_runtime::extract_per_sample_alpha_head(model),
                zensim_validate::bake_runtime::extract_hybrid_head(model),
                zensim_validate::bake_runtime::extract_tanh_output_head_scale(model),
                zensim_validate::output_calibration_spline::extract(model),
                zensim_validate::bake_runtime::extract_minmax_head(model),
            )
        } else {
            (None, None, None, None, None)
        };
        for (i, row) in g.feature_rows.iter().enumerate() {
            let p0: f64 = if a.score_units {
                match mmh.as_ref() {
                    Some(mm) => zensim_validate::bake_runtime::score_row_minmax(
                        model,
                        mm,
                        tanh,
                        ospline.as_ref(),
                        row,
                    ),
                    None => zensim_validate::bake_runtime::score_row(
                        &mut predictor,
                        transformed,
                        psa.as_ref(),
                        hyb.as_ref(),
                        tanh,
                        ospline.as_ref(),
                        &mut xbuf,
                        row,
                    ),
                }
            } else {
                let take = n_in.min(row.len());
                for (d, s) in xbuf[..take].iter_mut().zip(row[..take].iter()) {
                    *d = *s as f32;
                }
                let p = if transformed {
                    predictor.predict_transformed(&xbuf)
                } else {
                    predictor.predict(&xbuf)
                }
                .map_err(|e| format!("predictor forward: {e:?}"))?;
                p[0] as f64
            };
            match wi {
                Some(w) => acc[i] += w * p0,
                None if k == 1 => acc[i] = p0,
                None => acc[i] += p0,
            }
        }
    }
    if k > 1 && weights.is_none() {
        let kf = k as f64;
        for v in acc.iter_mut() {
            *v /= kf;
        }
    }

    let mut out = String::with_capacity(acc.len() * 24);
    out.push_str("row_idx\tpred\n");
    for (i, v) in acc.iter().enumerate() {
        out.push_str(&format!("{i}\t{v:?}\n"));
    }
    std::fs::write(&a.out, &out).map_err(|e| format!("write {:?}: {e}", a.out))?;
    eprintln!(
        "predict: {} rows x k={k}{} -> {:?} (caller_input_width {n_in}, units={})",
        g.feature_rows.len(),
        match &weights {
            Some(w) => format!(
                " weighted [{}]",
                w.iter()
                    .map(|x| format!("{x:.4}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            None => String::new(),
        },
        a.out,
        if a.score_units { "score" } else { "raw" }
    );
    Ok(())
}

// --------------------------------------------------------------------------
// subcommand: blend-heads  (the Profile-B multi-head raw blend — SOTA-944 §4)
// --------------------------------------------------------------------------

#[derive(Args)]
struct BlendHeadsArgs {
    /// Head fit npz (`w`/`bias`/`mu`/`sd` from `fit-lasso --emit-fit-npz`).
    /// Exactly TWO, in order: head1 (weight `--alpha`), head2 (1 − alpha).
    #[arg(long, required = true)]
    head: Vec<PathBuf>,
    /// Convex blend weight of head 1 (B lineage: 0.8 · cid + 0.2 · kon).
    #[arg(long)]
    alpha: f64,
    /// Anchor parquet(s) — the z-norm sample AND the dial-spline anchor
    /// (same registered anchor as fit-lasso).
    #[arg(long, required = true)]
    anchor_parquet: Vec<PathBuf>,
    #[arg(long)]
    anchor_target: String,
    #[arg(long, default_value_t = 1.0)]
    anchor_scale: f64,
    #[arg(long)]
    anchor_stride: Vec<usize>,
    #[arg(long, allow_negative_numbers = true)]
    anchor_clip_min: Option<f64>,
    /// Transform screen TSV — REQUIRED when the heads were fit on shaped
    /// grams (the anchor forward + bake metadata must mirror the space).
    #[arg(long)]
    transforms_tsv: Option<PathBuf>,
    /// Output bake path.
    #[arg(long)]
    out: PathBuf,
    /// Also write the collapsed blend as a fit npz (`w`/`bias`/`mu`/`sd`, f64,
    /// PRE-f16-pack) so a SECOND `blend-heads` pass can compose a third head on
    /// it — the registered 3-way mechanism. `mu` is all-zero and `sd` all-one, so
    /// this head's own raw-x collapse (`w/sd`, `bias − Σ μ·w/sd`) reproduces the
    /// blend exactly; the emitted vector is the pre-pack f64 blend, matching what
    /// `fit-lasso --emit-fit-npz` writes.
    #[arg(long)]
    emit_fit_npz: Option<PathBuf>,
    /// Embed zentrain.repro (argv + head shas). Fatal on failure.
    #[arg(long)]
    embed_repro: bool,
}

fn cmd_blend_heads(a: &BlendHeadsArgs) -> Result<(), String> {
    use zensim_validate::gram_lasso::{f16_bits_to_f64, f64_to_f16_bits, py_repr_f64};
    use zensim_validate::npz::Npz;

    if a.head.len() != 2 {
        return Err(format!(
            "--head must be given exactly twice, got {}",
            a.head.len()
        ));
    }
    if !(0.0..=1.0).contains(&a.alpha) {
        return Err(format!("--alpha must be in [0,1], got {}", a.alpha));
    }

    // 1. load heads; collapse each to raw-x space: p(x) = Σ x_j·(w_j/sd_j) + (bias − Σ μ_j·w_j/sd_j)
    struct Head {
        a_raw: Vec<f64>,
        c: f64,
    }
    let mut heads: Vec<Head> = Vec::with_capacity(2);
    let mut n_feat = 0usize;
    for hp in &a.head {
        let npz = Npz::open(hp)?;
        let w = npz.get("w")?.f64s()?.to_vec();
        let mu = npz.get("mu")?.f64s()?.to_vec();
        let sd = npz.get("sd")?.f64s()?.to_vec();
        let bias = npz.get("bias")?.scalar_f64()?;
        if n_feat == 0 {
            n_feat = w.len();
        } else if w.len() != n_feat {
            return Err(format!("head width mismatch: {} vs {n_feat}", w.len()));
        }
        let mut a_raw = vec![0.0f64; n_feat];
        let mut c = bias;
        for j in 0..n_feat {
            if w[j] != 0.0 {
                a_raw[j] = w[j] / sd[j];
                c -= mu[j] * w[j] / sd[j];
            }
        }
        heads.push(Head { a_raw, c });
    }

    // 2. anchor rows (f32-rounded; shaped transforms mirrored when given).
    let anchor_appliers: Option<Vec<(zenpredict::FeatureTransform, Vec<f32>)>> =
        match &a.transforms_tsv {
            Some(tsv) => Some(screen_appliers(tsv, n_feat)?),
            None => None,
        };
    let strides: Vec<usize> = if a.anchor_stride.is_empty() {
        vec![1; a.anchor_parquet.len()]
    } else if a.anchor_stride.len() == a.anchor_parquet.len() {
        a.anchor_stride.clone()
    } else {
        return Err("--anchor-stride count != --anchor-parquet count".into());
    };
    let mut xaf: Vec<f32> = Vec::new();
    let mut ya: Vec<f64> = Vec::new();
    for (path, &stride) in a.anchor_parquet.iter().zip(&strides) {
        let g = zensim_validate::parquet_loader::load_parquet(
            path,
            "anchor",
            &a.anchor_target,
            a.anchor_scale,
        )?;
        if g.n_features != n_feat {
            return Err(format!(
                "anchor parquet {path:?} width {} != head width {n_feat}",
                g.n_features
            ));
        }
        let mut ri = 0usize;
        while ri < g.feature_rows.len() {
            match &anchor_appliers {
                Some(ap) => xaf.extend(
                    g.feature_rows[ri]
                        .iter()
                        .zip(ap.iter())
                        .map(|(v, (t, ps))| t.apply_with_params(*v as f32, ps)),
                ),
                None => xaf.extend(g.feature_rows[ri].iter().map(|v| *v as f32)),
            }
            let mut y = g.human_scores[ri];
            if let Some(clip) = a.anchor_clip_min
                && y < clip
            {
                y = clip;
            }
            ya.push(y);
            ri += stride;
        }
    }
    let rows = ya.len();
    if rows < 50 {
        return Err(format!("anchor has only {rows} rows — too few"));
    }

    // 3. per-head anchor preds -> z-norm moments (population sd).
    let pred_of = |h: &Head, row: &[f32]| -> f64 {
        let mut acc = 0.0f64;
        for (x, aj) in row.iter().zip(&h.a_raw) {
            if *aj != 0.0 {
                acc += *x as f64 * *aj;
            }
        }
        acc + h.c
    };
    let mut znorm: Vec<(f64, f64)> = Vec::with_capacity(2);
    for h in &heads {
        let mut s1 = 0.0f64;
        let mut s2 = 0.0f64;
        for r in 0..rows {
            let p = pred_of(h, &xaf[r * n_feat..(r + 1) * n_feat]);
            s1 += p;
            s2 += p * p;
        }
        let m = s1 / rows as f64;
        let var = (s2 / rows as f64 - m * m).max(0.0);
        let s = var.sqrt();
        if s < 1e-12 {
            return Err("head has ~zero anchor-pred variance — cannot z-norm".into());
        }
        znorm.push((m, s));
        eprintln!("  head z-norm: mean {m:.6} sd {s:.6}");
    }

    // 4. collapse the z-normed convex blend to ONE identity-scaler layer:
    //    r(x) = α·(p1−m1)/s1 + (1−α)·(p2−m2)/s2
    let (m1, s1) = znorm[0];
    let (m2, s2) = znorm[1];
    let mut a_blend = vec![0.0f64; n_feat];
    for (j, ab) in a_blend.iter_mut().enumerate() {
        *ab = a.alpha * heads[0].a_raw[j] / s1 + (1.0 - a.alpha) * heads[1].a_raw[j] / s2;
    }
    let c_blend = a.alpha * (heads[0].c - m1) / s1 + (1.0 - a.alpha) * (heads[1].c - m2) / s2;

    // 5a. optional head artifact: the PRE-pack f64 blend in fit-npz form, so a
    //     second pass can treat this blend as one head (mu=0, sd=1 makes the
    //     reader's raw-x collapse the identity on these numbers).
    if let Some(npz_path) = &a.emit_fit_npz {
        use zensim_validate::npz::{NpzF64Entry, write_npz_f64};
        let shape1 = [n_feat];
        let bias_s = [c_blend];
        let mu0 = vec![0.0f64; n_feat];
        let sd1 = vec![1.0f64; n_feat];
        write_npz_f64(
            npz_path,
            &[
                NpzF64Entry {
                    name: "w",
                    shape: &shape1,
                    data: &a_blend,
                },
                NpzF64Entry {
                    name: "bias",
                    shape: &[],
                    data: &bias_s,
                },
                NpzF64Entry {
                    name: "mu",
                    shape: &shape1,
                    data: &mu0,
                },
                NpzF64Entry {
                    name: "sd",
                    shape: &shape1,
                    data: &sd1,
                },
            ],
        )?;
        eprintln!("  blend fit npz -> {npz_path:?}");
    }

    // 5. f16 pack, spline on the PACKED forward (QUANTIZE-then-CALIBRATE).
    let wp: Vec<f64> = a_blend
        .iter()
        .map(|v| f16_bits_to_f64(f64_to_f16_bits(*v)))
        .collect();
    let n_active = wp.iter().filter(|v| v.abs() > 0.0).count();
    let mut preds = vec![0.0f64; rows];
    for (r, pred) in preds.iter_mut().enumerate() {
        let row = &xaf[r * n_feat..(r + 1) * n_feat];
        let mut acc = 0.0f64;
        for j in 0..n_feat {
            if wp[j] != 0.0 {
                acc += row[j] as f64 * wp[j];
            }
        }
        *pred = acc + c_blend;
    }
    let (cx, cy) = fit_spline_knots(&preds, &ya, 18, true);

    // 6. metadata (transforms when shaped, spline), identity scaler emit.
    let mut metadata: Vec<OwnedMeta> = Vec::new();
    if let Some(tsv) = &a.transforms_tsv {
        let (toks, params) = load_transform_screen(tsv, n_feat)?;
        metadata.push(OwnedMeta {
            key: zenpredict::keys::FEATURE_TRANSFORMS.to_string(),
            kind: MetadataType::Utf8,
            value: toks.join("\n").into_bytes(),
        });
        let params_txt = params
            .iter()
            .map(|row| {
                row.iter()
                    .map(|p| py_repr_f64(*p))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .collect::<Vec<_>>()
            .join("\n");
        metadata.push(OwnedMeta {
            key: zenpredict::keys::FEATURE_TRANSFORM_PARAMS.to_string(),
            kind: MetadataType::Utf8,
            value: params_txt.into_bytes(),
        });
    }
    if !cx.is_empty() {
        metadata.push(OwnedMeta {
            key: SPLINE_KEY.to_string(),
            kind: MetadataType::Bytes,
            value: spline_payload(&cx, &cy),
        });
    }
    let zeros = vec![0.0f32; n_feat];
    let ones = vec![1.0f32; n_feat];
    let w32: Vec<f32> = wp.iter().map(|v| *v as f32).collect();
    let sz = emit_linear(&a.out, &zeros, &ones, &w32, c_blend as f32, &metadata)
        .map_err(|e| format!("write {:?}: {e}", a.out))?;

    if a.embed_repro {
        let argv: Vec<String> = std::env::args().collect();
        let mut head_shas = Vec::new();
        for h in &a.head {
            let sha = zensim_validate::train_manifest::sha256_file(h)
                .map_err(|e| format!("sha256 {h:?}: {e:?}"))?;
            head_shas.push(format!(
                "{{\"head\":{:?},\"sha256\":\"{sha}\"}}",
                h.display().to_string()
            ));
        }
        let repro = format!(
            "{{\"tool\":\"bake_dial_refit blend-heads\",\"alpha\":{},\"argv\":{:?},\"heads\":[{}]}}",
            a.alpha,
            argv,
            head_shas.join(",")
        );
        embed_repro_into(&a.out, &repro)?;
        eprintln!("  zentrain.repro embedded ({} B)", repro.len());
    }

    let out_bytes = std::fs::read(&a.out).map_err(|e| format!("re-read {:?}: {e}", a.out))?;
    eprintln!(
        "blend-heads -> {:?} ({sz} B): alpha {}, act={n_active}, {} knots, dial [{:.2},{:.2}]\n  sha256 {}",
        a.out,
        a.alpha,
        cx.len(),
        cy.first().copied().unwrap_or(f64::NAN),
        cy.last().copied().unwrap_or(f64::NAN),
        sha256_hex(&out_bytes)
    );
    Ok(())
}

// --------------------------------------------------------------------------
// subcommand: screen-transforms  (one-pass monotone screen — SOTA-944 §3b)
// --------------------------------------------------------------------------

#[derive(Args)]
struct ScreenTransformsArgs {
    /// Fit parquet(s), stride-sampled (paired `--stride`, default 1).
    #[arg(long, required = true)]
    parquet: Vec<PathBuf>,
    #[arg(long)]
    stride: Vec<usize>,
    #[arg(long, default_value = "human_score")]
    target: String,
    #[arg(long, default_value_t = 1.0)]
    target_scale: f64,
    #[arg(long, allow_negative_numbers = true)]
    target_clip_min: Option<f64>,
    /// Required |Pearson r| lift over identity to switch a feature off
    /// identity (registered: 0.005).
    #[arg(long, default_value_t = 0.005)]
    switch_threshold: f64,
    /// Output screen TSV (`feat_idx`/`best_transform`/`params_csv`).
    #[arg(long)]
    out: PathBuf,
}

fn cmd_screen_transforms(a: &ScreenTransformsArgs) -> Result<(), String> {
    use zensim_validate::parquet_loader::stream_parquet_rows;
    const CANDS: [&str; 3] = ["identity", "log1p", "signed_cbrt"];
    let cands: Vec<zenpredict::FeatureTransform> = CANDS
        .iter()
        .map(|t| zenpredict::FeatureTransform::from_token(t).expect("known token"))
        .collect();

    let strides: Vec<usize> = if a.stride.is_empty() {
        vec![1; a.parquet.len()]
    } else if a.stride.len() == a.parquet.len() {
        a.stride.clone()
    } else {
        return Err("--stride count != --parquet count".into());
    };

    let mut n_feat = 0usize;
    // per (feature, cand): n, Σt, Σt², Σty ; plus global Σy, Σy², n.
    let mut st: Vec<[f64; 3]> = Vec::new(); // [Σt, Σt², Σty] per (j, c)
    let mut sy = 0.0f64;
    let mut sy2 = 0.0f64;
    let mut n_rows_used = 0usize;
    for (path, &stride) in a.parquet.iter().zip(&strides) {
        let sha = zensim_validate::train_manifest::sha256_file(path)
            .map_err(|e| format!("sha256 {path:?}: {e:?}"))?;
        eprintln!("screen: {path:?} stride {stride}\n  sha256 {sha}");
        let mut global_idx = 0usize;
        stream_parquet_rows(
            path,
            &[a.target.as_str()],
            a.target_scale,
            &mut |features, n_rows, targets| {
                if n_feat == 0 {
                    n_feat = features.len() / n_rows;
                    st = vec![[0.0f64; 3]; n_feat * cands.len()];
                } else if features.len() / n_rows != n_feat {
                    return Err("screen: parquet width mismatch across inputs".into());
                }
                for r in 0..n_rows {
                    if !global_idx.is_multiple_of(stride) {
                        global_idx += 1;
                        continue;
                    }
                    global_idx += 1;
                    let x = &features[r * n_feat..(r + 1) * n_feat];
                    let mut y = targets[0][r];
                    if let Some(clip) = a.target_clip_min
                        && y < clip
                    {
                        y = clip;
                    }
                    sy += y;
                    sy2 += y * y;
                    n_rows_used += 1;
                    for (j, &xv) in x.iter().enumerate() {
                        for (c, t) in cands.iter().enumerate() {
                            let tv = t.apply(xv as f32) as f64;
                            let e = &mut st[j * cands.len() + c];
                            e[0] += tv;
                            e[1] += tv * tv;
                            e[2] += tv * y;
                        }
                    }
                }
                Ok(())
            },
        )?;
    }
    if n_rows_used < 100 {
        return Err(format!("screen: only {n_rows_used} rows — too few"));
    }
    let n = n_rows_used as f64;
    let var_y = (sy2 / n - (sy / n) * (sy / n)).max(0.0);
    let mut out = String::from("feat_idx\tbest_transform\tparams_csv\tr_identity\tr_best\n");
    let mut switched = 0usize;
    for j in 0..n_feat {
        let r_of = |c: usize| -> f64 {
            let e = &st[j * cands.len() + c];
            let var_t = (e[1] / n - (e[0] / n) * (e[0] / n)).max(0.0);
            let cov = e[2] / n - (e[0] / n) * (sy / n);
            if var_t <= 1e-24 || var_y <= 1e-24 {
                0.0
            } else {
                cov / (var_t.sqrt() * var_y.sqrt())
            }
        };
        let r_id = r_of(0);
        let mut best_c = 0usize;
        let mut best_r = r_id;
        for c in 1..cands.len() {
            let r = r_of(c);
            if r.abs() > best_r.abs() {
                best_r = r;
                best_c = c;
            }
        }
        // switch only on a real lift over identity (registered threshold)
        let (tok, r_out) = if best_c != 0 && best_r.abs() >= r_id.abs() + a.switch_threshold {
            switched += 1;
            (CANDS[best_c], best_r)
        } else {
            (CANDS[0], r_id)
        };
        out.push_str(&format!("{j}\t{tok}\t\t{r_id:.6}\t{r_out:.6}\n"));
    }
    std::fs::write(&a.out, &out).map_err(|e| format!("write {:?}: {e}", a.out))?;
    eprintln!(
        "screen-transforms -> {:?}: {n_rows_used} rows, {n_feat} feats, {switched} switched off identity (threshold {})",
        a.out, a.switch_threshold
    );
    Ok(())
}

// --------------------------------------------------------------------------
// subcommand: gram  (per-corpus raw-moment Gram builder for fit-lasso;
// E-LIN linear-924 — benchmarks/linear924_phase1_2026-08-01.md)
// --------------------------------------------------------------------------

#[derive(Args)]
struct GramArgs {
    /// Feature parquet (consecutive `f<i>` / `feat_<i>` columns + the
    /// target column(s)).
    #[arg(long)]
    parquet: PathBuf,
    /// Target column(s); each gets `__q_<target>` + `__Y1_<target>` entries.
    #[arg(long = "target", required = true)]
    targets: Vec<String>,
    /// Scale applied to every target (e.g. 100 for [0,1]-scale legs).
    #[arg(long, default_value_t = 1.0)]
    target_scale: f64,
    /// Clamp targets to at least this value (POST-scale). Registered E-LIN
    /// policy: -100 (MSE magnitude protection for catastrophic tails).
    #[arg(long, allow_negative_numbers = true)]
    target_clip_min: Option<f64>,
    /// Feature-space prefix for the npz keys (raw features ⇒ "raw").
    #[arg(long, default_value = "raw")]
    space: String,
    /// Fail unless the parquet's feature width is exactly this (regime
    /// purity guard — 924 for the folded+append campaign).
    #[arg(long)]
    expect_n_feat: Option<usize>,
    /// Accumulate only the FIRST `n` feature columns, i.e. fit a NARROWER
    /// student on a wider table. The resulting bake declares `n` as its caller
    /// input width, which is the whole point: a 372-input model needs the
    /// 372-class walk and a 944-input one needs the 944 walk, whatever its
    /// weights are zero on. Refuses `n` larger than the table (never pads),
    /// and prints the truncation so it can never be silent.
    ///
    /// Registered 2026-09-01 (hybrid lane PART II): distilling a 944 teacher
    /// into the 156-compute class needs the STUDENT at 372 width while the
    /// TEACHER target is only computable on a 944 root. `--slice-file` on
    /// `fit-lasso` zeroes coefficients; it cannot narrow the declared width.
    #[arg(long)]
    max_feat: Option<usize>,
    /// Transform screen TSV (`feat_idx`/`best_transform`/`params_csv`):
    /// apply per-feature monotone transforms DURING accumulation via
    /// zenpredict's own `FeatureTransform` f32 apply (fit space == the f32
    /// runtime's, zero math duplication). Use with `--space shaped`; the
    /// SAME TSV goes to `fit-lasso --transforms-tsv` so the bake carries
    /// the transforms (SOTA-944 §3b).
    #[arg(long)]
    transforms_tsv: Option<PathBuf>,
    /// Per-corpus min-max target normalization (B's kon-head anchor-target
    /// mechanism, `minmax01_bounds`): a first pass computes this corpus's
    /// [q0.001, q0.999] of the (scaled) target; the accumulation pass then
    /// uses `y' = clip((y − lo)/(hi − lo), 0, 1)`. The q/Y1 keys are stored
    /// under `<target>__mm01`. Mutually exclusive with `--target-clip-min`.
    #[arg(long)]
    target_minmax01: bool,
    /// Output `.npz` path.
    #[arg(long)]
    out: PathBuf,
}

/// numpy-default (linear-interpolation) quantile on a SORTED slice —
/// mirrors `np.quantile(x, q)` for the minmax01 bounds.
fn np_quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let h = q * (sorted.len() as f64 - 1.0);
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
    }
}

/// Parsed per-feature transform appliers from a screen TSV: zenpredict's
/// own enum + f32 params, applied f32-in/f32-out exactly like the runtime.
fn screen_appliers(
    path: &Path,
    n_feat: usize,
) -> Result<Vec<(zenpredict::FeatureTransform, Vec<f32>)>, String> {
    let (toks, params) = load_transform_screen(path, n_feat)?;
    toks.iter()
        .zip(&params)
        .enumerate()
        .map(|(i, (tok, ps))| {
            let t = zenpredict::FeatureTransform::from_token(tok)
                .map_err(|e| format!("screen TSV feat {i}: bad transform {tok:?}: {e:?}"))?;
            Ok((t, ps.iter().map(|p| *p as f32).collect()))
        })
        .collect()
}

/// Accumulate `S = Σxxᵀ` (upper triangle + mirror), `s = Σx`,
/// `q_t = Σx·y_t`, `Y1_t = Σy_t`, `n` over every row, in file order,
/// single-threaded f64 — the canonical deterministic accumulation for
/// E-LIN grams. The upper-triangle skip of `x_i == 0` rows is bit-exact
/// vs the naive triple loop (adding `±0.0·x_j` never changes a finite
/// accumulator that starts at +0.0), verified by
/// `tests/gram_stream_equivalence.rs`.
fn cmd_gram(a: &GramArgs) -> Result<(), String> {
    use zensim_validate::npz::{NpzF64Entry, write_npz_f64};
    use zensim_validate::parquet_loader::stream_parquet_rows;

    let sha = zensim_validate::train_manifest::sha256_file(&a.parquet)
        .map_err(|e| format!("sha256 {:?}: {e:?}", a.parquet))?;
    eprintln!("gram: input {:?}\n  sha256 {sha}", a.parquet);

    let target_refs: Vec<&str> = a.targets.iter().map(|s| s.as_str()).collect();

    // Per-feature shaped-space appliers (SOTA-944 §3b): zenpredict's own
    // transform math at f32, widened back to f64 for accumulation.
    let appliers: Option<Vec<(zenpredict::FeatureTransform, Vec<f32>)>> = None;
    let mut appliers = appliers; // sized on first batch (needs n_feat)

    // --target-minmax01 pre-pass: this corpus's own [q0.001, q0.999] of the
    // scaled target (B's kon-head `minmax01_bounds`). One extra stream of
    // the parquet — targets only are kept.
    let mm01: Option<Vec<(f64, f64)>> = if a.target_minmax01 {
        if a.target_clip_min.is_some() {
            return Err("--target-minmax01 and --target-clip-min are mutually exclusive".into());
        }
        let mut collected: Vec<Vec<f64>> = vec![Vec::new(); a.targets.len()];
        stream_parquet_rows(
            &a.parquet,
            &target_refs,
            a.target_scale,
            &mut |_f, n_rows, targets| {
                for (t, tv) in targets.iter().enumerate() {
                    collected[t].extend_from_slice(&tv[..n_rows]);
                }
                Ok(())
            },
        )?;
        let mut bounds = Vec::with_capacity(a.targets.len());
        for (t, vals) in collected.iter_mut().enumerate() {
            vals.sort_by(f64::total_cmp);
            let lo = np_quantile_sorted(vals, 0.001);
            let hi = np_quantile_sorted(vals, 0.999);
            if hi <= lo {
                return Err(format!(
                    "minmax01 bounds degenerate for target {} ([{lo}, {hi}])",
                    a.targets[t]
                ));
            }
            eprintln!(
                "  minmax01 {}: [q0.001, q0.999] = [{lo:.6}, {hi:.6}]",
                a.targets[t]
            );
            bounds.push((lo, hi));
        }
        Some(bounds)
    } else {
        None
    };

    let mut n_feat = 0usize;
    let mut s_mat: Vec<f64> = Vec::new();
    let mut s_vec: Vec<f64> = Vec::new();
    let mut q: Vec<Vec<f64>> = Vec::new();
    let mut y1: Vec<f64> = Vec::new();
    let mut clipped: Vec<usize> = vec![0; a.targets.len()];
    let mut ymin = vec![f64::INFINITY; a.targets.len()];
    let mut ymax = vec![f64::NEG_INFINITY; a.targets.len()];
    let mut ysum = vec![0.0f64; a.targets.len()];

    let mut shaped_row: Vec<f64> = Vec::new();
    let info = stream_parquet_rows(
        &a.parquet,
        &target_refs,
        a.target_scale,
        &mut |features, n_rows, targets| {
            if n_feat == 0 {
                n_feat = features.len() / n_rows;
                if let Some(exp) = a.expect_n_feat
                    && n_feat != exp
                {
                    return Err(format!(
                        "regime guard: {:?} has {n_feat} features, expected {exp}",
                        a.parquet
                    ));
                }
                if let Some(cap) = a.max_feat {
                    if cap > n_feat {
                        return Err(format!(
                            "--max-feat {cap} exceeds the table's {n_feat} columns; \
                             this never pads"
                        ));
                    }
                    if cap < n_feat {
                        eprintln!(
                            "  --max-feat: TRUNCATING {n_feat} -> {cap} columns \
                             (the bake's declared caller width will be {cap})"
                        );
                    }
                    n_feat = cap;
                }
                s_mat = vec![0.0f64; n_feat * n_feat];
                s_vec = vec![0.0f64; n_feat];
                q = vec![vec![0.0f64; n_feat]; a.targets.len()];
                y1 = vec![0.0f64; a.targets.len()];
                if let Some(tsv) = &a.transforms_tsv {
                    appliers = Some(screen_appliers(tsv, n_feat)?);
                    shaped_row = vec![0.0f64; n_feat];
                }
            }
            // Row stride is the TABLE's width; `n_feat` may be a prefix cap.
            let stride = features.len() / n_rows;
            for r in 0..n_rows {
                let raw = &features[r * stride..r * stride + n_feat];
                let x: &[f64] = if let Some(ap) = &appliers {
                    for (dst, (src, (t, ps))) in
                        shaped_row.iter_mut().zip(raw.iter().zip(ap.iter()))
                    {
                        *dst = t.apply_with_params(*src as f32, ps) as f64;
                    }
                    &shaped_row
                } else {
                    raw
                };
                // S upper triangle: row-sequential rank-1 update. The inner
                // loop is contiguous over j for auto-vectorization.
                for i in 0..n_feat {
                    let xi = x[i];
                    if xi != 0.0 {
                        let base = i * n_feat;
                        for j in i..n_feat {
                            s_mat[base + j] += xi * x[j];
                        }
                    }
                }
                for (acc, v) in s_vec.iter_mut().zip(x) {
                    *acc += *v;
                }
                for (t, tv) in targets.iter().enumerate() {
                    let mut y = tv[r];
                    if let Some(bounds) = &mm01 {
                        let (lo, hi) = bounds[t];
                        let z = (y - lo) / (hi - lo);
                        if !(0.0..=1.0).contains(&z) {
                            clipped[t] += 1;
                        }
                        y = z.clamp(0.0, 1.0);
                    } else if let Some(clip) = a.target_clip_min
                        && y < clip
                    {
                        y = clip;
                        clipped[t] += 1;
                    }
                    let qt = &mut q[t];
                    for (acc, v) in qt.iter_mut().zip(x) {
                        *acc += *v * y;
                    }
                    y1[t] += y;
                    if y < ymin[t] {
                        ymin[t] = y;
                    }
                    if y > ymax[t] {
                        ymax[t] = y;
                    }
                    ysum[t] += y;
                }
            }
            Ok(())
        },
    )?;

    // Mirror the upper triangle (bitwise-symmetric by construction, matching
    // the numpy X'X property the lasso comment documents).
    for i in 0..n_feat {
        for j in (i + 1)..n_feat {
            s_mat[j * n_feat + i] = s_mat[i * n_feat + j];
        }
    }

    let n_rows_f = info.n_rows as f64;
    let mut entries: Vec<NpzF64Entry<'_>> = Vec::new();
    let key_s = format!("{}__S", a.space);
    let key_sv = format!("{}__s", a.space);
    let key_n = format!("{}__n", a.space);
    // minmax01 targets get a distinct key suffix so a fit can never mix a
    // normalized q with an unnormalized one silently.
    let tsuffix = if a.target_minmax01 { "__mm01" } else { "" };
    let key_q: Vec<String> = a
        .targets
        .iter()
        .map(|t| format!("{}__q_{t}{tsuffix}", a.space))
        .collect();
    let key_y1: Vec<String> = a
        .targets
        .iter()
        .map(|t| format!("{}__Y1_{t}{tsuffix}", a.space))
        .collect();
    let shape2 = [n_feat, n_feat];
    let shape1 = [n_feat];
    entries.push(NpzF64Entry {
        name: &key_s,
        shape: &shape2,
        data: &s_mat,
    });
    entries.push(NpzF64Entry {
        name: &key_sv,
        shape: &shape1,
        data: &s_vec,
    });
    entries.push(NpzF64Entry {
        name: &key_n,
        shape: &[],
        data: std::slice::from_ref(&n_rows_f),
    });
    for t in 0..a.targets.len() {
        entries.push(NpzF64Entry {
            name: &key_q[t],
            shape: &shape1,
            data: &q[t],
        });
        entries.push(NpzF64Entry {
            name: &key_y1[t],
            shape: &[],
            data: std::slice::from_ref(&y1[t]),
        });
    }
    write_npz_f64(&a.out, &entries)?;

    eprintln!(
        "gram -> {:?}: {} rows x {n_feat} feats [space {}]",
        a.out, info.n_rows, a.space
    );
    for (t, name) in a.targets.iter().enumerate() {
        eprintln!(
            "  target {name} (x{}, clip>={}): min {:.3} max {:.3} mean {:.3} clipped {}",
            a.target_scale,
            a.target_clip_min
                .map(|c| c.to_string())
                .unwrap_or_else(|| "-inf".into()),
            ymin[t],
            ymax[t],
            ysum[t] / n_rows_f,
            clipped[t]
        );
    }
    Ok(())
}

// --------------------------------------------------------------------------
// sha256 (for add-winsor's byte-repro assertion)
// --------------------------------------------------------------------------

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    let d = h.finalize();
    let mut s = String::with_capacity(64);
    for b in d {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// --------------------------------------------------------------------------
// subcommand: refit-winsor  (SOTA-944 wave 8, amendment 9 §9.1)
// --------------------------------------------------------------------------

#[derive(Args)]
struct RefitWinsorArgs {
    /// Fit parquet(s). Rows are POOLED with equal per-row weight.
    #[arg(long, required = true)]
    parquet: Vec<PathBuf>,
    /// The inherited token list: one `--feature-transform` VALUE per line,
    /// i.e. `<token>:<idx>[:<params_csv>]`, in the order the trainer
    /// receives them. Blank lines and `#` comments are ignored.
    #[arg(long)]
    base_tokens: PathBuf,
    #[arg(long, default_value_t = 0.1)]
    lo_pct: f64,
    #[arg(long, default_value_t = 99.9)]
    hi_pct: f64,
    /// Refit token list — same order, non-winsor lines byte-verbatim.
    #[arg(long)]
    out_tokens: PathBuf,
    /// Audit TSV (idx / token / old_lo / old_hi / new_lo / new_hi / n /
    /// degenerate_old / degenerate_new / changed / fit_lo / fit_hi / selected).
    /// `new_*` is the EMITTED window (= inherited when the index is not
    /// selected); `fit_*` is always what the pooled fit produced.
    #[arg(long)]
    out_tsv: PathBuf,
    /// Target column name — needed only to drive the streaming loader.
    #[arg(long, default_value = "human_score")]
    target: String,
    /// Which winsor indices the newly-fit window is APPLIED to, classified by
    /// the INHERITED window (`degenerate` == `old_lo == old_hi`). Indices that
    /// are not selected keep their inherited window and are emitted
    /// byte-verbatim from `--base-tokens`. The fit itself always covers every
    /// winsor index, so `fit_lo`/`fit_hi` are recorded either way.
    /// (SOTA-944 wave 9, amendment 10 §10.2.)
    #[arg(long, value_enum, default_value_t = RefitClass::All)]
    refit_class: RefitClass,
    /// Explicit comma-separated index subset to apply the fit to. Mutually
    /// exclusive with a non-default `--refit-class`.
    #[arg(long)]
    refit_indices: Option<String>,
}

#[derive(Copy, Clone, PartialEq, Eq, clap::ValueEnum)]
enum RefitClass {
    /// Every winsor index (the default; wave-8 behaviour).
    All,
    /// Only indices whose inherited window is degenerate (`old_lo == old_hi`).
    Degenerate,
    /// Only indices whose inherited window is non-degenerate.
    Nondegenerate,
}

/// `add-winsor`'s window rule, factored out so both callers share it:
/// linear-interpolated percentiles plus the degenerate `[0,0] → [0,1e-9]`
/// guard. `sorted` must be ascending.
fn winsor_window(sorted: &[f64], lo_pct: f64, hi_pct: f64) -> (f64, f64) {
    let lo = percentile_linear(sorted, lo_pct);
    let mut hi = percentile_linear(sorted, hi_pct);
    if lo == 0.0 && hi == 0.0 {
        hi = 1e-9;
    }
    (lo, hi)
}

/// Split a `--feature-transform` value into `(token, idx, params_csv)`.
/// `signed_cbrt:61:` → `("signed_cbrt", 61, "")`.
fn parse_ft_token(line: &str) -> Result<(String, usize, String), String> {
    let mut it = line.splitn(3, ':');
    let tok = it.next().unwrap_or_default().to_string();
    let idx_s = it
        .next()
        .ok_or_else(|| format!("token {line:?}: missing :<idx>"))?;
    let idx: usize = idx_s
        .trim()
        .parse()
        .map_err(|e| format!("token {line:?}: bad idx {idx_s:?}: {e}"))?;
    Ok((tok, idx, it.next().unwrap_or("").to_string()))
}

/// Decide which winsor indices the newly-fit window is APPLIED to
/// (amendment 10 §10.2). Pure: no I/O, no fit — it reads only the inherited
/// token list, so the classification is a property of the base screen.
///
/// `entries` are `(original_line, token, idx, params_csv)` in file order;
/// `winsor_idx` is the `winsor_p99` index list in first-appearance order.
fn select_refit_indices(
    entries: &[(String, String, usize, String)],
    winsor_idx: &[usize],
    class: RefitClass,
    explicit: Option<&str>,
) -> Result<std::collections::HashSet<usize>, String> {
    let explicit = explicit.map(str::trim).filter(|s| !s.is_empty());
    if explicit.is_some() && class != RefitClass::All {
        return Err(
            "--refit-indices and a non-default --refit-class are mutually exclusive".into(),
        );
    }
    if let Some(csv) = explicit {
        let mut set = std::collections::HashSet::new();
        for part in csv.split(',') {
            let p = part.trim();
            if p.is_empty() {
                continue;
            }
            let i: usize = p
                .parse()
                .map_err(|e| format!("--refit-indices: bad index {p:?}: {e}"))?;
            if !winsor_idx.contains(&i) {
                return Err(format!("--refit-indices: {i} is not a winsor_p99 index"));
            }
            set.insert(i);
        }
        if set.is_empty() {
            return Err("--refit-indices selected nothing".into());
        }
        return Ok(set);
    }
    if class == RefitClass::All {
        return Ok(winsor_idx.iter().copied().collect());
    }
    let want_deg = class == RefitClass::Degenerate;
    let mut set = std::collections::HashSet::new();
    for (line, tok, idx, params) in entries {
        if tok != "winsor_p99" {
            continue;
        }
        // A class selector needs a well-defined inherited window; an
        // unparseable one would silently land in the complement.
        let old: Vec<f64> = params
            .split(',')
            .map(|s| {
                s.trim().parse::<f64>().map_err(|e| {
                    format!("token {line:?}: --refit-class needs 2 numeric params: {e}")
                })
            })
            .collect::<Result<_, _>>()?;
        if old.len() != 2 {
            return Err(format!(
                "token {line:?}: --refit-class needs exactly 2 params, got {}",
                old.len()
            ));
        }
        if (old[0] == old[1]) == want_deg {
            set.insert(*idx);
        }
    }
    if set.is_empty() {
        return Err(format!(
            "--refit-class {} selected no indices",
            if want_deg {
                "degenerate"
            } else {
                "nondegenerate"
            }
        ));
    }
    Ok(set)
}

fn cmd_refit_winsor(a: &RefitWinsorArgs) -> Result<(), String> {
    use zensim_validate::parquet_loader::stream_parquet_rows;

    let txt = std::fs::read_to_string(&a.base_tokens)
        .map_err(|e| format!("read {:?}: {e}", a.base_tokens))?;
    // (original line, token, idx, old params) in file order.
    let mut entries: Vec<(String, String, usize, String)> = Vec::new();
    for raw in txt.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (tok, idx, params) = parse_ft_token(line)?;
        entries.push((line.to_string(), tok, idx, params));
    }
    if entries.is_empty() {
        return Err(format!("{:?} contains no transform tokens", a.base_tokens));
    }

    // The winsor indices, in first-appearance order; duplicates are an error
    // (the trainer would silently keep the last).
    let mut winsor_idx: Vec<usize> = Vec::new();
    for (line, tok, idx, _) in &entries {
        if tok == "winsor_p99" {
            if winsor_idx.contains(idx) {
                return Err(format!("duplicate winsor index {idx} ({line:?})"));
            }
            winsor_idx.push(*idx);
        }
    }
    if winsor_idx.is_empty() {
        return Err("no winsor_p99 tokens to refit".into());
    }
    let slot_of: std::collections::HashMap<usize, usize> = winsor_idx
        .iter()
        .enumerate()
        .map(|(s, &i)| (i, s))
        .collect();

    // Which indices the fit is APPLIED to (amendment 10 §10.2). The fit below
    // always covers every winsor index; this only decides emission.
    let selected = select_refit_indices(
        &entries,
        &winsor_idx,
        a.refit_class,
        a.refit_indices.as_deref(),
    )?;
    eprintln!(
        "refit-winsor: applying the fit to {} of {} winsor indices",
        selected.len(),
        winsor_idx.len()
    );

    // Pool the requested columns only — full-width rows are streamed and
    // dropped, so peak memory is one batch plus |winsor_idx| columns.
    let mut vals: Vec<Vec<f64>> = vec![Vec::new(); winsor_idx.len()];
    let mut n_feat_seen = 0usize;
    let mut n_rows_total = 0usize;
    for path in &a.parquet {
        let sha = zensim_validate::train_manifest::sha256_file(path)
            .map_err(|e| format!("sha256 {path:?}: {e:?}"))?;
        let mut n_rows_here = 0usize;
        stream_parquet_rows(
            path,
            &[a.target.as_str()],
            1.0,
            &mut |features, n_rows, _targets| {
                let n_feat = features.len() / n_rows;
                if n_feat_seen == 0 {
                    n_feat_seen = n_feat;
                    if let Some(&bad) = winsor_idx.iter().find(|&&i| i >= n_feat) {
                        return Err(format!(
                            "winsor index {bad} >= feature width {n_feat} in {path:?}"
                        ));
                    }
                } else if n_feat != n_feat_seen {
                    return Err(format!(
                        "feature width {n_feat} in {path:?} != {n_feat_seen} from earlier input"
                    ));
                }
                for r in 0..n_rows {
                    let row = &features[r * n_feat..(r + 1) * n_feat];
                    for (slot, &j) in winsor_idx.iter().enumerate() {
                        vals[slot].push(row[j]);
                    }
                }
                n_rows_here += n_rows;
                Ok(())
            },
        )?;
        eprintln!("refit-winsor: {path:?} rows {n_rows_here}\n  sha256 {sha}");
        n_rows_total += n_rows_here;
    }
    eprintln!(
        "refit-winsor: pooled {n_rows_total} rows, width {n_feat_seen}, {} winsor indices",
        winsor_idx.len()
    );

    // Windows.
    let mut new_win: Vec<(f64, f64)> = Vec::with_capacity(winsor_idx.len());
    for col in vals.iter_mut() {
        col.sort_by(f64::total_cmp);
        new_win.push(winsor_window(col, a.lo_pct, a.hi_pct));
    }

    // Emit tokens (input order) + audit TSV.
    let mut out_tok = String::new();
    let mut tsv = format!(
        "# refit-winsor  lo_pct={} hi_pct={}  pooled_rows={n_rows_total}  width={n_feat_seen}\n\
         # inputs: {}\n\
         idx\ttoken\told_lo\told_hi\tnew_lo\tnew_hi\tn\tdegenerate_old\tdegenerate_new\tchanged\tfit_lo\tfit_hi\tselected\n",
        a.lo_pct,
        a.hi_pct,
        a.parquet
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(" "),
    );
    let (mut n_changed, mut n_deg_old, mut n_deg_new) = (0usize, 0usize, 0usize);
    for (line, tok, idx, params) in &entries {
        if tok != "winsor_p99" {
            out_tok.push_str(line);
            out_tok.push('\n');
            continue;
        }
        let slot = slot_of[idx];
        let (flo, fhi) = new_win[slot];
        let old: Vec<f64> = params
            .split(',')
            .filter_map(|s| s.trim().parse::<f64>().ok())
            .collect();
        let (olo, ohi) = if old.len() == 2 {
            (old[0], old[1])
        } else {
            (f64::NAN, f64::NAN)
        };
        // Not selected ⇒ the inherited line is emitted BYTE-VERBATIM, so an
        // inherited window can never be silently reformatted.
        let is_sel = selected.contains(idx);
        let (lo, hi) = if is_sel { (flo, fhi) } else { (olo, ohi) };
        let deg_old = olo == ohi;
        let deg_new = lo == hi;
        let changed = !(olo == lo && ohi == hi);
        n_deg_old += usize::from(deg_old);
        n_deg_new += usize::from(deg_new);
        n_changed += usize::from(changed);
        if is_sel {
            out_tok.push_str(&format!("winsor_p99:{idx}:{lo:e},{hi:e}\n"));
        } else {
            out_tok.push_str(line);
            out_tok.push('\n');
        }
        tsv.push_str(&format!(
            "{idx}\t{tok}\t{olo:e}\t{ohi:e}\t{lo:e}\t{hi:e}\t{}\t{}\t{}\t{}\t{flo:e}\t{fhi:e}\t{}\n",
            vals[slot].len(),
            u8::from(deg_old),
            u8::from(deg_new),
            u8::from(changed),
            u8::from(is_sel),
        ));
    }
    std::fs::write(&a.out_tokens, &out_tok)
        .map_err(|e| format!("write {:?}: {e}", a.out_tokens))?;
    std::fs::write(&a.out_tsv, &tsv).map_err(|e| format!("write {:?}: {e}", a.out_tsv))?;
    eprintln!(
        "refit-winsor: {} tokens ({} winsor, {} applied) -> {:?}; changed {n_changed}, degenerate old {n_deg_old} new {n_deg_new}; audit {:?}",
        entries.len(),
        winsor_idx.len(),
        selected.len(),
        a.out_tokens,
        a.out_tsv,
    );
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result: Result<bool, String> = match &cli.cmd {
        Cmd::AddSpline(a) => cmd_add_spline(a).map(|_| false),
        Cmd::ExtendTop(a) => cmd_extend_top(a).map(|_| false),
        Cmd::SharedAnchor(a) => cmd_shared_anchor(a).map(|_| false),
        Cmd::BottomExtend(a) => cmd_bottom_extend(a).map(|_| false),
        Cmd::AddWinsor(a) => cmd_add_winsor(a).map(|_| false),
        Cmd::Gate(a) => cmd_gate(a),
        Cmd::Pack(a) => cmd_pack(a).map(|_| false),
        Cmd::Strip(a) => cmd_strip(a).map(|_| false),
        Cmd::AppendMeta(a) => cmd_append_meta(a).map(|_| false),
        Cmd::FitLasso(a) => cmd_fit_lasso(a).map(|_| false),
        Cmd::Gram(a) => cmd_gram(a).map(|_| false),
        Cmd::Predict(a) => cmd_predict(a).map(|_| false),
        Cmd::BlendHeads(a) => cmd_blend_heads(a).map(|_| false),
        Cmd::ScreenTransforms(a) => cmd_screen_transforms(a).map(|_| false),
        Cmd::RefitWinsor(a) => cmd_refit_winsor(a).map(|_| false),
    };
    match result {
        Ok(gate_failed) => {
            if gate_failed {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// C8 (appendix W): explicit tau always wins; auto-default is 0.005 for
    /// dense bakes and 0 for the sparse class (kill fraction > 10%). The
    /// measured population anchoring the threshold: dense MLPs ≈0%, ADD156
    /// 50% (13/26), GL4_s2501 95% (54/57).
    #[test]
    fn zerobias_default_resolves_dense_vs_sparse() {
        // explicit flag wins, both directions, regardless of kill stats
        assert_eq!(resolve_zerobias(Some(0.005), 54, 57), (0.005, false));
        assert_eq!(resolve_zerobias(Some(0.0), 0, 300), (0.0, false));
        // dense: essentially no whole-line kills → historical default
        assert_eq!(
            resolve_zerobias(None, 0, 285),
            (ZEROBIAS_DENSE_DEFAULT, false)
        );
        assert_eq!(
            resolve_zerobias(None, 28, 285),
            (ZEROBIAS_DENSE_DEFAULT, false)
        );
        // sparse: ADD156- and GL-class kill fractions → auto 0, loudly
        assert_eq!(resolve_zerobias(None, 13, 26), (0.0, true));
        assert_eq!(resolve_zerobias(None, 54, 57), (0.0, true));
        // degenerate: no live lines cannot divide-by-zero
        assert_eq!(
            resolve_zerobias(None, 0, 0),
            (ZEROBIAS_DENSE_DEFAULT, false)
        );
    }

    /// Build a tiny 3→1 identity f16 bake with a monotone spline. No
    /// feature transforms (all-identity), scaler is identity.
    fn tiny_bake(knots: &[(f32, f32)]) -> Vec<u8> {
        let mut payload = (knots.len() as u32).to_le_bytes().to_vec();
        for (x, y) in knots {
            payload.extend_from_slice(&x.to_le_bytes());
            payload.extend_from_slice(&y.to_le_bytes());
        }
        let weights = [1.0f32, 0.5, -0.25];
        let biases = [0.0f32];
        let meta = [BakeMetadataEntry {
            key: SPLINE_KEY,
            kind: MetadataType::Bytes,
            value: &payload,
        }];
        bake(&BakeRequest {
            schema_hash: 0,
            flags: 0,
            scaler_mean: &[0.0; 3],
            scaler_scale: &[1.0; 3],
            layers: &[BakeLayer {
                in_dim: 3,
                out_dim: 1,
                activation: Activation::Identity,
                dtype: WeightDtype::F16,
                weights: &weights,
                biases: &biases,
            }],
            feature_bounds: &[],
            metadata: &meta,
            output_specs: &[],
            discrete_sets: &[],
            sparse_overrides: &[],
            feature_order: None,
            output_order: None,
            compressed: false,
            hu_permutations: None,
        })
        .unwrap()
    }

    fn assert_strictly_monotone(xs: &[f64], ys: &[f64]) {
        for w in xs.windows(2) {
            assert!(
                w[1] > w[0],
                "x not strictly increasing: {} !> {}",
                w[1],
                w[0]
            );
        }
        for w in ys.windows(2) {
            assert!(
                w[1] > w[0],
                "y not strictly increasing: {} !> {}",
                w[1],
                w[0]
            );
        }
    }

    /// Rows whose raw preds (= 1.25·t) land strictly inside the base spline
    /// interior [x_lo, x_hi] — where a monotone spline is strictly
    /// increasing (no top-cap ties), so a rank-preserving refit gives an
    /// exact SROCC of 1.
    fn interior_rows() -> Vec<[f64; 3]> {
        (0..20)
            .map(|i| {
                let t = -1.5 + 0.15 * i as f64;
                [t, t, t]
            })
            .collect()
    }

    fn dial_over(
        model: &Model,
        sp: &spline::OutputCalibrationSpline,
        rows: &[[f64; 3]],
    ) -> Vec<f64> {
        let lin = load_linear(model);
        let ops = build_fw_ops(model, 3).unwrap();
        rows.iter()
            .map(|r| spline::apply(forward_raw(&r[..], &ops, &lin), sp))
            .collect()
    }

    #[test]
    fn extend_top_stays_monotone_and_rank_invariant() {
        let bytes = tiny_bake(&[(-2.0, 0.0), (0.0, 50.0), (2.0, 90.0)]);
        let model = Model::from_bytes(&bytes).unwrap();
        let rows = interior_rows();
        let sp_before = spline::extract(&model).unwrap();
        let before = dial_over(&model, &sp_before, &rows);

        // extend-top's exact knot-append math (k fixed for the test).
        let (mut xs, mut ys) = read_spline(&model);
        let (x0, y0) = (*xs.last().unwrap(), *ys.last().unwrap());
        let k = 1.5f64;
        let r_far = x0 + (-(1e-4f64).ln()) / k;
        let start = x0 + (r_far - x0) / 12.0;
        let step = (r_far - start) / 11.0;
        for i in 0..12 {
            let r = if i == 11 {
                r_far
            } else {
                i as f64 * step + start
            };
            let y = 100.0 - (100.0 - y0) * (-k * (r - x0)).exp();
            if r > *xs.last().unwrap() + 1e-7 && y > *ys.last().unwrap() {
                xs.push(r);
                ys.push(y);
            }
        }
        // (1) the full extended spline (incl. the appended top) is strictly monotone.
        assert!(xs.len() > 3, "no knots were appended");
        assert_strictly_monotone(&xs, &ys);

        // (2) forwarding before/after over the interior feature set is rank-exact.
        let sp_after = spline::parse_payload(&spline_payload(&xs, &ys)).unwrap();
        let after = dial_over(&model, &sp_after, &rows);
        let sr = spearman(&before, &after);
        assert!(
            (sr - 1.0).abs() < 1e-12,
            "extend-top changed rank: SROCC={sr}"
        );
    }

    #[test]
    fn bottom_extend_stays_monotone_and_rank_invariant() {
        let bytes = tiny_bake(&[(-2.0, 10.0), (0.0, 50.0), (2.0, 90.0)]);
        let model = Model::from_bytes(&bytes).unwrap();
        let rows = interior_rows();
        let sp_before = spline::extract(&model).unwrap();
        let before = dial_over(&model, &sp_before, &rows);

        let (mut xs, mut ys) = read_spline(&model);
        xs.insert(0, -4.0);
        ys.insert(0, 0.0);
        assert_strictly_monotone(&xs, &ys);

        let sp_after = spline::parse_payload(&spline_payload(&xs, &ys)).unwrap();
        let after = dial_over(&model, &sp_after, &rows);
        let sr = spearman(&before, &after);
        assert!(
            (sr - 1.0).abs() < 1e-12,
            "bottom-extend changed rank: SROCC={sr}"
        );
    }

    #[test]
    /// **D4 (ADD156 ship audit, `benchmarks/add156_ship_audit_2026-08-31.md`).**
    /// `pack` without `--neg-tail` used to SILENTLY delete the dial's negative
    /// tail while the prune identity gate went on reporting "BIT-identical" —
    /// because that gate compares the NETWORK's raw outputs on in-domain anchor
    /// rows, and the damage is in the output-calibration SPLINE, which `pack`
    /// refits from scratch underneath it.
    ///
    /// Half one pins the pathology; half two asserts the two new guards.
    fn d4_flat_bottom_spline_deletes_the_negative_tail_and_is_now_caught() {
        use zensim_validate::dial_spline::{fit_spline_knots, neg_tail_is_material};
        use zensim_validate::output_calibration_spline as ocs;

        // An anchor whose low end is CLAMPED at zero — the shape every real
        // dial anchor has, because the target is a bounded [0,100] score.
        // Several low bins therefore share a median target of 0.0.
        let preds: Vec<f64> = (0..600).map(|i| -1.0 + i as f64 * 0.005).collect();
        let tgt: Vec<f64> = preds
            .iter()
            .map(|&p| if p <= 0.4 { 0.0 } else { (p - 0.4) * 120.0 })
            .collect();

        // (1) THE PATHOLOGY. Without the dedup the fit keeps the whole run of
        // y≈0 knots, so the bottom segment is flat and everything below it
        // maps to exactly 0.0 — a dead zone, not a tail.
        let (fx, fy) = fit_spline_knots(&preds, &tgt, 18, false);
        let (dx, dy) = fit_spline_knots(&preds, &tgt, 18, true);
        let flat_run = fy.iter().take_while(|&&y| y <= 1e-6).count();
        assert!(
            flat_run > 1,
            "fixture no longer produces a flat-zero knot run ({flat_run})"
        );
        assert_eq!(
            dy.iter().take_while(|&&y| y <= 1e-6).count(),
            1,
            "the dedup must leave exactly one y≈0 knot"
        );

        let payload_flat = spline_payload(&fx, &fy);
        let payload_tail = spline_payload(&dx, &dy);
        let sp_flat = ocs::parse_payload(&payload_flat).expect("flat spline parses");
        let sp_tail = ocs::parse_payload(&payload_tail).expect("tail spline parses");

        // Below the bottom knot the flat spline is pinned at exactly 0.0...
        let deep = dx[0] - 0.5;
        assert_eq!(
            ocs::apply(deep, &sp_flat),
            0.0,
            "flat-bottom spline must reproduce the dead zone this test exists to pin"
        );
        // ...while the deduped one keeps a real slope and goes NEGATIVE, which
        // is the product contract ("inputs worse than the worst codec output
        // score BELOW 0; do NOT clamp at 0").
        assert!(
            ocs::apply(deep, &sp_tail) < -1e-3,
            "deduped spline must extrapolate below zero, got {}",
            ocs::apply(deep, &sp_tail)
        );

        // (2a) GUARD ONE: the choice is detected as material, so `pack` can
        // refuse instead of silently picking the tail-deleting side.
        let material = neg_tail_is_material(&preds, &tgt, 18);
        assert_eq!(
            material,
            Some((fx.len(), dx.len())),
            "a flat-zero run must be reported as a MATERIAL --neg-tail choice"
        );
        // An anchor with no clamped run leaves the choice immaterial, so an
        // ordinary pack is not gratuitously blocked.
        let clean_tgt: Vec<f64> = preds.iter().map(|&p| (p + 2.0) * 20.0).collect();
        assert_eq!(neg_tail_is_material(&preds, &clean_tgt, 18), None);

        // (2b) GUARD TWO: the identity report now COVERS the tail, so a
        // deleted tail can never again pass as "bit-identical".
        let in_scores = vec![-12.0, -3.0, 5.0, 40.0, 90.0];
        let killed = vec![0.0, 0.0, 5.0, 40.0, 90.0];
        let kept = vec![-12.0, -3.0, 5.0, 40.0, 90.0];
        let rep_bad = dial_tail_report(&fx, &fy, &in_scores, &killed);
        assert!(rep_bad.tail_deleted());
        assert_eq!((rep_bad.in_negative, rep_bad.pinned_to_zero), (2, 2));
        assert!(
            rep_bad.render().contains("DELETED") && rep_bad.render().contains("--neg-tail"),
            "the report must say what happened and how to fix it: {}",
            rep_bad.render()
        );
        let rep_ok = dial_tail_report(&dx, &dy, &in_scores, &kept);
        assert!(!rep_ok.tail_deleted());
        assert!(rep_ok.render().contains("PRESERVED"));
    }

    #[test]
    fn fit_spline_knots_is_monotone() {
        // synthetic: pred uniform, target a monotone-with-noise function.
        let preds: Vec<f64> = (0..500).map(|i| i as f64 * 0.2).collect();
        let tgt: Vec<f64> = preds
            .iter()
            .enumerate()
            .map(|(i, &p)| p * 0.9 + if i % 7 == 0 { 1.5 } else { 0.0 })
            .collect();
        let (cx, cy) = fit_spline_knots(&preds, &tgt, 18, true);
        assert!(cx.len() >= 2, "too few knots: {}", cx.len());
        // x strictly increasing, y non-decreasing (fit_spline_knots contract).
        for w in cx.windows(2) {
            assert!(w[1] > w[0], "knot x not increasing");
        }
        for w in cy.windows(2) {
            assert!(w[1] >= w[0], "knot y decreased");
        }
    }

    #[test]
    fn winsor_op_clips_like_np_clip() {
        let op = FwOp::Winsor(-1.0, 2.0);
        assert_eq!(op.apply(-3.0), -1.0);
        assert_eq!(op.apply(0.5), 0.5);
        assert_eq!(op.apply(9.0), 2.0);
    }

    #[test]
    fn signed_cbrt_and_log1p_ops_match_canonical_math() {
        // sign(x)·|x|^(1/3), matching zenpredict FeatureTransform::SignedCbrt.
        assert!((FwOp::SignedCbrt.apply(8.0) - 2.0).abs() < 1e-12);
        assert!((FwOp::SignedCbrt.apply(-8.0) + 2.0).abs() < 1e-12);
        assert_eq!(FwOp::SignedCbrt.apply(0.0), 0.0);
        // ln(1+x), matching FeatureTransform::Log1p.
        assert!((FwOp::Log1p.apply(std::f64::consts::E - 1.0) - 1.0).abs() < 1e-12);
        assert_eq!(FwOp::Log1p.apply(0.0), 0.0);
    }

    /// A 2-layer f32 MLP bake (3→2 leaky-relu, 2→1 identity) with weights
    /// straddling the zerobias threshold in both layers.
    fn two_layer_bake() -> Vec<u8> {
        let w0 = [0.004f32, -0.0049, 0.006, -1.0, 0.5, 0.0051];
        let b0 = [0.1f32, -0.2];
        let w1 = [0.003f32, 0.8];
        let b1 = [0.05f32];
        bake(&BakeRequest {
            schema_hash: 0xabc123,
            flags: 0,
            scaler_mean: &[0.0; 3],
            scaler_scale: &[1.0; 3],
            layers: &[
                BakeLayer {
                    in_dim: 3,
                    out_dim: 2,
                    activation: Activation::LeakyRelu,
                    dtype: WeightDtype::F32,
                    weights: &w0,
                    biases: &b0,
                },
                BakeLayer {
                    in_dim: 2,
                    out_dim: 1,
                    activation: Activation::Identity,
                    dtype: WeightDtype::F32,
                    weights: &w1,
                    biases: &b1,
                },
            ],
            feature_bounds: &[],
            metadata: &[],
            output_specs: &[],
            discrete_sets: &[],
            sparse_overrides: &[],
            feature_order: None,
            output_order: None,
            compressed: false,
            hu_permutations: None,
        })
        .unwrap()
    }

    #[test]
    fn pack_layers_zerobias_protect_last() {
        let bytes = two_layer_bake();
        let model = Model::from_bytes(&bytes).unwrap();

        // protect-last: bulk layer zerobias'd + f16, last layer untouched f32.
        let (packed, counts) = pack_layers(&model, WeightDtype::F16, 0.005, true).unwrap();
        assert_eq!(counts, vec![(0, 2, 6), (1, 0, 2)]);
        assert_eq!(packed[0].dtype as u8, WeightDtype::F16 as u8);
        assert_eq!(packed[0].weights, vec![0.0, 0.0, 0.006, -1.0, 0.5, 0.0051]);
        assert_eq!(packed[1].dtype as u8, WeightDtype::F32 as u8);
        assert_eq!(packed[1].weights, vec![0.003, 0.8]); // 0.003 SURVIVES (protected)
        assert_eq!(packed[0].biases, vec![0.1, -0.2]); // biases never zerobias'd
        assert_eq!(packed[1].biases, vec![0.05]);

        // without protect-last: the last layer gets the bulk treatment too.
        let (packed, counts) = pack_layers(&model, WeightDtype::F16, 0.005, false).unwrap();
        assert_eq!(counts, vec![(0, 2, 6), (1, 1, 2)]);
        assert_eq!(packed[1].dtype as u8, WeightDtype::F16 as u8);
        assert_eq!(packed[1].weights, vec![0.0, 0.8]);
    }

    /// The whole point of pack-THEN-calibrate: fitting the spline on the
    /// PACKED network's outputs re-anchors identity exactly, where a spline
    /// fit on the pre-pack outputs would map the packed net's (shifted)
    /// identity output to the wrong dial value.
    #[test]
    fn pack_then_calibrate_reanchors_identity() {
        let bytes = two_layer_bake();
        let model = Model::from_bytes(&bytes).unwrap();
        let (packed, _) = pack_layers(&model, WeightDtype::F16, 0.005, false).unwrap();
        let nospline = emit_packed(
            model.schema_hash(),
            model.scaler_mean(),
            model.scaler_scale(),
            &packed,
            &[],
        );
        // schema_hash survives the re-emit.
        assert_eq!(
            Model::from_bytes(&nospline).unwrap().schema_hash(),
            0xabc123
        );

        // synthetic anchor: rows spanning the packed net's output range,
        // target = the PACKED net's own output min-max mapped to [0,100]
        // (the identity-dial relation the spline must re-anchor).
        let feats: Vec<Vec<f64>> = (0..200)
            .map(|i| {
                let t = -2.0 + 0.02 * i as f64;
                vec![t, t, t]
            })
            .collect();
        let preds = forward_scored_6dec(&nospline, &feats).unwrap();
        let (lo, hi) = preds
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |a, &p| {
                (a.0.min(p), a.1.max(p))
            });
        let tgt: Vec<f64> = preds
            .iter()
            .map(|&p| (p - lo) / (hi - lo) * 100.0)
            .collect();
        let (cx, cy) = fit_spline_knots(&preds, &tgt, 18, false);
        assert!(cx.len() >= 2);
        // the fitted spline maps the packed net's own top-of-range output to
        // (approximately) the top target — identity is re-anchored on the
        // PACKED outputs, not the f32 originals.
        let sp = spline::parse_payload(&spline_payload(&cx, &cy)).unwrap();
        let top_pred = preds.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mapped_top = spline::apply(top_pred, &sp);
        assert!(
            mapped_top > 90.0,
            "packed top output should map near the top target (got {mapped_top:.2})"
        );
    }

    #[test]
    fn six_decimal_roundtrip_matches_printed_pipe() {
        assert_eq!(round_6dec(0.123456789), 0.123457);
        assert_eq!(round_6dec(-1e-7), 0.0); // "-0.000000" parses to -0.0 == 0.0
        assert_eq!(round_6dec(97.5), 97.5);
        assert!(round_6dec(f64::NAN).is_nan());
    }

    // ── gate: MLP-capable forward (the SOTA-944 "G-RANGE NOT EVALUABLE" fix) ──

    /// A 2-layer MLP bake (3→2 LeakyReLU, 2→1 identity) CARRYING an output
    /// spline — the bake class `gate` used to panic on (`load_linear`
    /// asserts single-layer). All-ones layer-0 weights + 0.5 layer-1 weights
    /// make the forward weight-layout-independent: for rows `[t,t,t]` with
    /// `t ≥ 0`, both hidden units are `3t` (LeakyReLU passthrough) and
    /// `raw = 0.5·3t + 0.5·3t = 3t`.
    fn two_layer_bake_with_spline(knots: &[(f32, f32)]) -> Vec<u8> {
        let mut payload = (knots.len() as u32).to_le_bytes().to_vec();
        for (x, y) in knots {
            payload.extend_from_slice(&x.to_le_bytes());
            payload.extend_from_slice(&y.to_le_bytes());
        }
        let w0 = [1.0f32; 6];
        let b0 = [0.0f32; 2];
        let w1 = [0.5f32; 2];
        let b1 = [0.0f32];
        let meta = [BakeMetadataEntry {
            key: SPLINE_KEY,
            kind: MetadataType::Bytes,
            value: &payload,
        }];
        bake(&BakeRequest {
            schema_hash: 0,
            flags: 0,
            scaler_mean: &[0.0; 3],
            scaler_scale: &[1.0; 3],
            layers: &[
                BakeLayer {
                    in_dim: 3,
                    out_dim: 2,
                    activation: Activation::LeakyRelu,
                    dtype: WeightDtype::F32,
                    weights: &w0,
                    biases: &b0,
                },
                BakeLayer {
                    in_dim: 2,
                    out_dim: 1,
                    activation: Activation::Identity,
                    dtype: WeightDtype::F32,
                    weights: &w1,
                    biases: &b1,
                },
            ],
            feature_bounds: &[],
            metadata: &meta,
            output_specs: &[],
            discrete_sets: &[],
            sparse_overrides: &[],
            feature_order: None,
            output_order: None,
            compressed: false,
            hu_permutations: None,
        })
        .unwrap()
    }

    /// 8-row corpus parquet `[t,t,t]` for `t = 0.1·(i+1)` with a correlated
    /// `human_score` (= t), under the `f<i>` prefix `read_features` expects.
    fn write_gate_corpus(path: &Path) {
        use std::sync::Arc;

        use arrow::array::ArrayRef;
        use arrow::datatypes::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;

        let ts: Vec<f64> = (0..8).map(|i| 0.1 * (i + 1) as f64).collect();
        let fields = vec![
            Field::new("human_score", DataType::Float64, false),
            Field::new("f0", DataType::Float64, false),
            Field::new("f1", DataType::Float64, false),
            Field::new("f2", DataType::Float64, false),
        ];
        let schema = Arc::new(Schema::new(fields));
        let cols: Vec<ArrayRef> = vec![
            Arc::new(Float64Array::from(ts.clone())),
            Arc::new(Float64Array::from(ts.clone())),
            Arc::new(Float64Array::from(ts.clone())),
            Arc::new(Float64Array::from(ts)),
        ];
        let batch = RecordBatch::try_new(schema.clone(), cols).unwrap();
        let f = File::create(path).unwrap();
        let mut w = ArrowWriter::try_new(f, schema, None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    fn gate_args(bake: PathBuf, corpus: PathBuf) -> GateArgs {
        GateArgs {
            bake,
            corpus,
            ref_col: "human_score".into(),
            feat_prefix: "f".into(),
            range_frac: 1e-4,
        }
    }

    /// `gate` on a 2-layer MLP — the invocation that used to PANIC in
    /// `load_linear` ("expects a single-layer linear bake"), which is why
    /// G-RANGE read "NOT EVALUABLE (inherited MLP tool gap)" for every MLP
    /// candidate of the SOTA-944 campaign. Wide knots → every raw pred
    /// (3t ∈ [0.3, 2.4]) in-domain → PASS; narrow knots [1.0, 1.5] → 6/8
    /// rows extrapolate → FAIL. Both verdicts prove the forward is real,
    /// not a degenerate all-inside path.
    #[test]
    fn gate_evaluates_two_layer_mlp_bakes() {
        let dir = std::env::temp_dir().join("zensim_gate_mlp_test");
        std::fs::create_dir_all(&dir).unwrap();
        let corpus = dir.join("gate_fixture.parquet");
        write_gate_corpus(&corpus);

        let wide = dir.join("mlp_wide.bin");
        std::fs::write(
            &wide,
            two_layer_bake_with_spline(&[(-10.0, 0.0), (10.0, 100.0)]),
        )
        .unwrap();
        let failed = cmd_gate(&gate_args(wide, corpus.clone()))
            .expect("gate must run on a 2-layer MLP bake (panicked before the bake_runtime fix)");
        assert!(
            !failed,
            "raw preds 0.3..2.4 all inside [-10,10] — must PASS"
        );

        let narrow = dir.join("mlp_narrow.bin");
        std::fs::write(
            &narrow,
            two_layer_bake_with_spline(&[(1.0, 20.0), (1.5, 80.0)]),
        )
        .unwrap();
        let failed = cmd_gate(&gate_args(narrow, corpus)).expect("gate runs");
        assert!(
            failed,
            "6/8 raw preds extrapolate outside [1.0, 1.5] — must FAIL"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The linear class still gates correctly through the shared path
    /// (continuity for every pre-fix caller: raw = 1.25·t stays inside wide
    /// knots → PASS; a narrow domain above the range → FAIL).
    #[test]
    fn winsor_window_matches_owner_rule_and_degenerate_guard() {
        // percentile_linear semantics on a known ladder.
        let v: Vec<f64> = (0..=100).map(f64::from).collect();
        let (lo, hi) = winsor_window(&v, 0.1, 99.9);
        assert!((lo - 0.1).abs() < 1e-12, "lo {lo}");
        assert!((hi - 99.9).abs() < 1e-12, "hi {hi}");
        // all-zero column → the [0,0] guard fires, and NOT to a plain [0,0].
        let z = vec![0.0f64; 32];
        assert_eq!(winsor_window(&z, 0.1, 99.9), (0.0, 1e-9));
        // constant-nonzero stays degenerate (guard is [0,0]-only) — the
        // refit reports it rather than silently widening it.
        let c = vec![7.0f64; 32];
        assert_eq!(winsor_window(&c, 0.1, 99.9), (7.0, 7.0));
        // a column that is 99.9%+ zeros still collapses — the honest
        // outcome the refit must REPORT, not hide.
        let mut sparse = vec![0.0f64; 10_000];
        sparse[9_999] = 5.0;
        let (slo, shi) = winsor_window(&sparse, 0.1, 99.9);
        assert_eq!((slo, shi), (0.0, 1e-9));
    }

    #[test]
    fn ft_token_round_trip() {
        assert_eq!(
            parse_ft_token("winsor_p99:100:1.46128e-06,0.000106967").unwrap(),
            (
                "winsor_p99".to_string(),
                100,
                "1.46128e-06,0.000106967".to_string()
            )
        );
        // trailing-colon parameterless form (the signed_cbrt shape)
        assert_eq!(
            parse_ft_token("signed_cbrt:61:").unwrap(),
            ("signed_cbrt".to_string(), 61, String::new())
        );
        // degenerate zero window parses as two params, not as "no params"
        let (_, _, p) = parse_ft_token("winsor_p99:731:0,0").unwrap();
        let nums: Vec<f64> = p.split(',').map(|s| s.parse().unwrap()).collect();
        assert_eq!(nums, vec![0.0, 0.0]);
        assert!(
            parse_ft_token("winsor_p99").is_err(),
            "missing idx must error"
        );
        assert!(
            parse_ft_token("winsor_p99:xx:0,0").is_err(),
            "bad idx must error"
        );
    }

    /// The wave-9 window-subset selector (amendment 10 §10.2). Classification
    /// is by the INHERITED window, so it is a property of the base screen and
    /// never of the fit — which is what makes `degenerate ⊎ nondegenerate =
    /// all` a checkable identity rather than a hope.
    #[test]
    fn refit_class_partitions_by_inherited_window() {
        // A miniature of the real screen: 2 degenerate append-ish indices, 3
        // non-degenerate fold-ish ones, and a parameterless passthrough.
        let lines = [
            "winsor_p99:9:2.5e-16,6.6e-05",
            "signed_cbrt:61:",
            "winsor_p99:731:0,0",
            "winsor_p99:155:1.1e-09,0.1638",
            "winsor_p99:748:0,0",
            "winsor_p99:100:1.46128e-06,0.000106967",
        ];
        let entries: Vec<(String, String, usize, String)> = lines
            .iter()
            .map(|l| {
                let (t, i, p) = parse_ft_token(l).unwrap();
                (l.to_string(), t, i, p)
            })
            .collect();
        let widx: Vec<usize> = entries
            .iter()
            .filter(|(_, t, _, _)| t == "winsor_p99")
            .map(|(_, _, i, _)| *i)
            .collect();
        assert_eq!(widx, vec![9, 731, 155, 748, 100]);

        let sel = |c, e| select_refit_indices(&entries, &widx, c, e).unwrap();
        let all = sel(RefitClass::All, None);
        let deg = sel(RefitClass::Degenerate, None);
        let non = sel(RefitClass::Nondegenerate, None);

        assert_eq!(all.len(), 5, "All selects every winsor index");
        assert_eq!(deg, [731usize, 748].into_iter().collect());
        assert_eq!(non, [9usize, 155, 100].into_iter().collect());
        // the registered identity: disjoint, and their union is `all`
        assert!(deg.is_disjoint(&non), "classes must not overlap");
        assert_eq!(
            deg.union(&non)
                .copied()
                .collect::<std::collections::HashSet<_>>(),
            all,
            "degenerate + nondegenerate must exactly cover all"
        );

        // explicit subset
        assert_eq!(
            sel(RefitClass::All, Some(" 731 , 100 ")),
            [731usize, 100].into_iter().collect()
        );
        // and its guards
        assert!(
            select_refit_indices(&entries, &widx, RefitClass::Degenerate, Some("731")).is_err(),
            "--refit-indices with a non-default class must be rejected"
        );
        assert!(
            select_refit_indices(&entries, &widx, RefitClass::All, Some("61")).is_err(),
            "a non-winsor index must be rejected, not silently ignored"
        );
        assert!(
            select_refit_indices(&entries, &widx, RefitClass::All, Some("9999")).is_err(),
            "an absent index must be rejected"
        );
        // an unparseable inherited window has no defined class — error, never
        // a silent landing in the complement.
        let bad: Vec<(String, String, usize, String)> = vec![(
            "winsor_p99:9:junk".into(),
            "winsor_p99".into(),
            9,
            "junk".into(),
        )];
        assert!(
            select_refit_indices(&bad, &[9], RefitClass::Degenerate, None).is_err(),
            "unparseable params must error under a class selector"
        );
        // An empty class is an error too: a screen with no degenerate windows
        // asked for `degenerate` is a user mistake, not an empty refit.
        let nodeg: Vec<(String, String, usize, String)> = vec![(
            "winsor_p99:9:1,2".into(),
            "winsor_p99".into(),
            9,
            "1,2".into(),
        )];
        assert!(select_refit_indices(&nodeg, &[9], RefitClass::Degenerate, None).is_err());
    }

    #[test]
    fn gate_still_evaluates_linear_bakes() {
        let dir = std::env::temp_dir().join("zensim_gate_linear_test");
        std::fs::create_dir_all(&dir).unwrap();
        let corpus = dir.join("gate_fixture.parquet");
        write_gate_corpus(&corpus);

        // tiny_bake weights [1.0, 0.5, -0.25] → raw = 1.25·t ∈ [0.125, 1.0].
        let wide = dir.join("lin_wide.bin");
        std::fs::write(&wide, tiny_bake(&[(-100.0, 0.0), (100.0, 100.0)])).unwrap();
        let failed = cmd_gate(&gate_args(wide, corpus.clone())).expect("gate runs");
        assert!(!failed, "linear raw preds inside [-100,100] — must PASS");

        let above = dir.join("lin_above.bin");
        std::fs::write(&above, tiny_bake(&[(5.0, 0.0), (10.0, 100.0)])).unwrap();
        let failed = cmd_gate(&gate_args(above, corpus)).expect("gate runs");
        assert!(failed, "all linear raw preds below knot 5.0 — must FAIL");
        std::fs::remove_dir_all(&dir).ok();
    }
}
