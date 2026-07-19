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
//! * `gate` — reproduces `bake_outlier_gate.py`: the HARD **G-RANGE** gate
//!   (fraction of raw preds outside the spline knot domain — the tail
//!   detector SROCC is blind to) + advisory Z-RMSE / outlier-ratio / SROCC
//!   vs a reference column. Computes NO PWRC (the `zenstats` `sa_st_curve`
//!   O(n²) allocation OOMs broad corpora — see the module's PWRC note).

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use arrow::array::{Array, Float32Array, Float64Array};
use arrow::record_batch::RecordBatch;
use clap::{Args, Parser, Subcommand};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use zenpredict::{Activation, MetadataType, Model, WeightDtype, WeightStorage, f16_bits_to_f32};
use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake};
use zensim_validate::output_calibration_spline as spline;
use zensim_validate::panel::{outlier_ratio, rescale_logistic, spearman, z_rmse};
// Dial-spline fitting lives in the shared `dial_spline` module (2026-07-16) so
// this linear tool AND the min-max bake path fit the [0,100] dial identically.
use zensim_validate::dial_spline::{fit_spline_knots, percentile_linear, spline_payload};

/// Metadata key for the PCHIP output-calibration spline payload. Matches
/// the private `KEY` in `output_calibration_spline` (which does not
/// re-export it) and `zenpredict::keys::FEATURE_TRANSFORM*`.
const SPLINE_KEY: &str = "zentrain.output_calibration_spline";

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
        }
    }
}

/// Build the per-feature forward ops from the bake's transform metadata.
/// Absent metadata ⇒ all-identity (the baker omits the entry when every
/// transform is identity). Only `identity` / `winsor_p99` are supported in
/// the f64 fit-forward; any other token errors (the yeo-johnson / shaped
/// HDR path stays in the research Python, per the migration doc).
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
            other => {
                return Err(format!(
                    "f64 fit-forward supports identity/winsor_p99 only; feature {i} has {other:?} \
                     (shaped/HDR bakes stay in the research Python)"
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
    let n_in = model.n_inputs();

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
    let sz = emit_full(&a.out, &model, &metadata).map_err(|e| format!("write {:?}: {e}", a.out))?;
    eprintln!("emitted {:?} ({sz} B)", a.out);
    Ok(())
}

/// Re-emit an arbitrary bake VERBATIM (every layer, dtype-preserving) with
/// replacement metadata — the MLP-capable sibling of [`emit_linear`]. f16
/// weights round-trip exactly; i8 layers error (extend when needed).
fn emit_full(out: &Path, model: &Model, metadata: &[OwnedMeta]) -> std::io::Result<usize> {
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
        schema_hash: 0,
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
}

fn cmd_add_winsor(a: &AddWinsorArgs) -> Result<(), String> {
    let bytes = std::fs::read(&a.input).map_err(|e| format!("read {:?}: {e}", a.input))?;
    let model = Model::from_bytes(&bytes).map_err(|e| format!("parse bake: {e:?}"))?;
    if model.metadata().iter().any(|e| e.key.contains("transform")) {
        return Err("input already has feature transforms — winsorize a RAW bake".into());
    }
    let lin = load_linear(&model);
    let n = lin.scaler_mean.len();

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
        lo[j] = percentile_linear(&col, a.lo_pct);
        hi[j] = percentile_linear(&col, a.hi_pct);
        if lo[j] == 0.0 && hi[j] == 0.0 {
            hi[j] = 1e-9;
        }
    }

    let transforms_txt = vec!["winsor_p99"; n].join("\n");
    let params_txt = (0..n)
        .map(|j| format!("{},{}", lo[j], hi[j]))
        .collect::<Vec<_>>()
        .join("\n");
    // metadata order: transforms, params, then everything the raw bake had
    // (incl. its spline) verbatim.
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
    metadata.extend(clone_metadata(&model));

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
        "winsorized {:?} -> {:?} ({sz} B); {n} winsor_p99 transforms, fit [p{},p{}]\n  sha256 {got}",
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
    let bytes = std::fs::read(&a.bake).map_err(|e| format!("read {:?}: {e}", a.bake))?;
    let model = Model::from_bytes(&bytes).map_err(|e| format!("parse bake: {e:?}"))?;
    let lin = load_linear(&model);
    let n = lin.scaler_mean.len();
    let ops = build_fw_ops(&model, n)?;
    let sp = spline::extract(&model).ok_or("bake has no output_calibration_spline")?;
    let klo = sp.xs[0];
    let khi = sp.xs[sp.xs.len() - 1];

    let (feats, refv) = read_features(&a.corpus, &a.feat_prefix, n, &a.ref_col);
    let raw: Vec<f64> = feats.iter().map(|r| forward_raw(r, &ops, &lin)).collect();
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

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result: Result<bool, String> = match &cli.cmd {
        Cmd::AddSpline(a) => cmd_add_spline(a).map(|_| false),
        Cmd::ExtendTop(a) => cmd_extend_top(a).map(|_| false),
        Cmd::SharedAnchor(a) => cmd_shared_anchor(a).map(|_| false),
        Cmd::BottomExtend(a) => cmd_bottom_extend(a).map(|_| false),
        Cmd::AddWinsor(a) => cmd_add_winsor(a).map(|_| false),
        Cmd::Gate(a) => cmd_gate(a),
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
}
