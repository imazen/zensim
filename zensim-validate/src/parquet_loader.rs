//! Parquet loader for zensim feature training data.
//!
//! Produces an [`OwnedLoadedGroup`] that mirrors `LoadedGroup` in
//! `zensim-validate/src/bin/zensim_mlp_train.rs` field-for-field.
//!
//! The schema mirrors the CSV layout the trainer already understands:
//! a `target_column` (e.g. `iwssim_log_norm`) plus consecutive
//! `f0, f1, ..., f<N-1>` feature columns. Any other columns in the file
//! (image paths, codec names, etc.) are ignored.
//!
//! Wiring into the trainer binary is deliberately deferred — the user
//! will write a thin `From<OwnedLoadedGroup> for LoadedGroup` shim
//! during merge once the parallel-CSV agent and SIMD-fusion agent
//! also land. The conversion is trivial: identical field shapes.
//!
//! Equivalence to the sequential CSV loader is verified by
//! `tests/parquet_load_equivalence.rs`.

use std::fs::File;
use std::path::PathBuf;

use arrow::array::{
    Array, Float32Array, Float64Array, Int32Array, Int64Array, StringArray, UInt32Array,
    UInt64Array,
};
use arrow::datatypes::DataType;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// Owned mirror of the trainer's `LoadedGroup` struct.
///
/// The trainer (in `bin/zensim_mlp_train.rs`) defines `LoadedGroup`
/// privately; reproducing the shape here lets the parquet loader stay
/// completely additive (no edits to the trainer source). Merging the
/// two requires a one-line `From` shim, written by the user during the
/// agent-worktree merge.
#[derive(Debug)]
pub struct OwnedLoadedGroup {
    pub name: String,
    pub train_w: f64,
    pub val_w: f64,
    pub human_scores: Vec<f64>,
    pub feature_rows: Vec<Vec<f64>>,
    pub n_features: usize,
    /// Per-row metric-disagreement σ, computed from the normalized std
    /// of available metric columns (cvvdp_score, iwssim, ssim2_gpu).
    /// When all three are present and non-null, σ = std([norm_cvvdp,
    /// norm_iwssim, norm_ssim2]) per row. The trainer can use this as
    /// a per-pair weight in the MSE loss: loss = ((pred - target) / σ)².
    /// None if the metric columns are not available.
    pub metric_sigmas: Option<Vec<f64>>,
}

/// Load a zensim training feature parquet file.
///
/// Reads the `target_column` (multiplied by `target_scale` to match
/// the trainer's score units) plus `f0..f<N-1>` consecutive feature
/// columns. Other columns are skipped. Returns an
/// [`OwnedLoadedGroup`] with `train_w` / `val_w` set to `0.0` —
/// callers fill those in from the group-spec parser.
///
/// Errors mirror the CSV loader's phrasing so the binary's error
/// handling stays uniform after merge:
/// - missing `target_column` -> `"<path>: missing target column ..."`
/// - missing `f0` column     -> `"<path>: missing f0 column"`
/// - no `f0..` columns       -> `"<path>: no fN columns found"`
/// - empty file              -> `"<path>: empty file ..."`
pub fn load_parquet(
    path: &PathBuf,
    name: &str,
    target_column: &str,
    target_scale: f64,
) -> Result<OwnedLoadedGroup, String> {
    let file = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("{path:?}: parquet open: {e}"))?;
    let schema = builder.schema().clone();
    let parquet_schema = builder.parquet_schema().clone();

    // Build a (leaf-index, column-name) view for projection + dispatch.
    // We use the Arrow `schema` for top-level names (these match the
    // CSV header names) and `parquet_schema` for the leaf projection
    // mask (Parquet schema is column-tree shaped, Arrow is flat for
    // primitive types — which is what zensim features always are).
    let arrow_fields = schema.fields();
    let n_arrow_cols = arrow_fields.len();

    // Locate the target column by name.
    let score_arrow_idx = arrow_fields
        .iter()
        .position(|f| f.name() == target_column)
        .ok_or_else(|| format!("{path:?}: missing target column {target_column:?}"))?;

    // Locate f0 column, then count consecutive f<i> columns. Mirrors
    // the CSV loader's scan precisely so the same source data
    // produces the same `n_features`.
    let f0_arrow_idx = arrow_fields
        .iter()
        .position(|f| f.name() == "f0")
        .ok_or_else(|| format!("{path:?}: missing f0 column"))?;
    let mut n_features = 0usize;
    while f0_arrow_idx + n_features < n_arrow_cols {
        let expected = format!("f{}", n_features);
        if arrow_fields[f0_arrow_idx + n_features].name() != &expected {
            break;
        }
        n_features += 1;
    }
    if n_features == 0 {
        return Err(format!("{path:?}: no fN columns found"));
    }

    // Project just the columns we need. For primitive (non-nested)
    // schemas — which zensim feature parquets always are — the
    // arrow-column index maps 1:1 to a parquet leaf index. We use
    // `ProjectionMask::leaves` indexed by these arrow positions.
    let mut wanted: Vec<usize> = Vec::with_capacity(n_features + 1);
    wanted.push(score_arrow_idx);
    for i in 0..n_features {
        wanted.push(f0_arrow_idx + i);
    }
    let mask = ProjectionMask::leaves(&parquet_schema, wanted.iter().copied());

    let reader = builder
        .with_projection(mask)
        .with_batch_size(16384)
        .build()
        .map_err(|e| format!("{path:?}: parquet build reader: {e}"))?;

    // The projected reader emits batches whose column order matches the
    // projection mask's leaf ordering — which is ascending by original
    // column index, not the order we passed in. Pre-compute the post-
    // projection index of (target, f0..f<N-1>) so we don't have to look
    // up by name per batch.
    let mut sorted_wanted = wanted.clone();
    sorted_wanted.sort_unstable();
    let proj_score_idx = sorted_wanted
        .iter()
        .position(|&i| i == score_arrow_idx)
        .expect("score idx must be in projection");
    let proj_feature_indices: Vec<usize> = (0..n_features)
        .map(|i| {
            sorted_wanted
                .iter()
                .position(|&p| p == f0_arrow_idx + i)
                .expect("feature idx must be in projection")
        })
        .collect();

    let mut human_scores: Vec<f64> = Vec::new();
    let mut feature_rows: Vec<Vec<f64>> = Vec::new();

    for batch_res in reader {
        let batch = batch_res.map_err(|e| format!("{path:?}: parquet read batch: {e}"))?;
        let n_rows = batch.num_rows();
        if n_rows == 0 {
            continue;
        }

        // Extract target column as f64 (multiply by target_scale).
        let score_col = batch.column(proj_score_idx);
        let score_iter: Box<dyn Iterator<Item = f64>> = match score_col.data_type() {
            DataType::Float64 => {
                let a = score_col
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("Float64 dispatch");
                // Materialize into a Vec then iterate — avoids lifetime
                // gymnastics across the row-build loop.
                let v: Vec<f64> = (0..n_rows).map(|i| a.value(i) * target_scale).collect();
                Box::new(v.into_iter())
            }
            DataType::Float32 => {
                let a = score_col
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .expect("Float32 dispatch");
                let v: Vec<f64> = (0..n_rows)
                    .map(|i| a.value(i) as f64 * target_scale)
                    .collect();
                Box::new(v.into_iter())
            }
            other => {
                return Err(format!(
                    "{path:?}: target column {target_column:?} has unsupported dtype {other:?} (need Float32/Float64)",
                ));
            }
        };
        human_scores.extend(score_iter);

        // Extract feature columns. We pre-materialize per-batch into
        // batch-row-major Vec<Vec<f64>> to keep the per-row push hot.
        // Materializing once per batch (not per cell) lets the loop
        // be column-wise, which is friendlier to the columnar arrays.
        let mut per_col_f64: Vec<Vec<f64>> = Vec::with_capacity(n_features);
        for &pi in &proj_feature_indices {
            let col = batch.column(pi);
            // Feature columns may be Float64/Float32 (the common case)
            // OR an integer type (Int32/Int64/UInt32/UInt64). Some
            // feature extractors emit count-style features (e.g.
            // KonJND's f12) as integers; the trainer / runtime
            // consumes them as f64 anyway, so widen silently.
            let v: Vec<f64> = match col.data_type() {
                DataType::Float64 => {
                    let a = col.as_any().downcast_ref::<Float64Array>().unwrap();
                    (0..n_rows).map(|i| a.value(i)).collect()
                }
                DataType::Float32 => {
                    let a = col.as_any().downcast_ref::<Float32Array>().unwrap();
                    (0..n_rows).map(|i| a.value(i) as f64).collect()
                }
                DataType::Int64 => {
                    let a = col.as_any().downcast_ref::<Int64Array>().unwrap();
                    (0..n_rows).map(|i| a.value(i) as f64).collect()
                }
                DataType::Int32 => {
                    let a = col.as_any().downcast_ref::<Int32Array>().unwrap();
                    (0..n_rows).map(|i| a.value(i) as f64).collect()
                }
                DataType::UInt64 => {
                    let a = col.as_any().downcast_ref::<UInt64Array>().unwrap();
                    (0..n_rows).map(|i| a.value(i) as f64).collect()
                }
                DataType::UInt32 => {
                    let a = col.as_any().downcast_ref::<UInt32Array>().unwrap();
                    (0..n_rows).map(|i| a.value(i) as f64).collect()
                }
                other => {
                    return Err(format!(
                        "{path:?}: feature column has unsupported dtype {other:?} (need Float32/Float64/Int32/Int64/UInt32/UInt64)",
                    ));
                }
            };
            per_col_f64.push(v);
        }

        // Transpose per-column f64 vectors into row-major Vec<Vec<f64>>.
        feature_rows.reserve(n_rows);
        for row_i in 0..n_rows {
            let mut row = Vec::with_capacity(n_features);
            for col in &per_col_f64 {
                row.push(col[row_i]);
            }
            feature_rows.push(row);
        }
    }

    if human_scores.is_empty() {
        return Err(format!("{path:?}: empty file / no rows"));
    }

    // Compute per-row metric-disagreement σ from available metric columns.
    // Uses the normalized std of (cvvdp_score, iwssim, ssim2_gpu) when
    // all three are present. This is the natural per-pair "confidence"
    // signal: where metrics agree, the quality judgment is easy (low σ);
    // where they disagree, it's ambiguous (high σ).
    let metric_sigmas: Option<Vec<f64>> = None;
    #[allow(unreachable_code)]
    if false {
        let _metric_sigmas_disabled = {
            let cv_col = load_optional_scalar_column(path, "cvvdp_score")
                .ok()
                .flatten();
            let iw_col = load_optional_scalar_column(path, "iwssim").ok().flatten();
            let s2_col = load_optional_scalar_column(path, "ssim2_gpu")
                .ok()
                .flatten();
            match (cv_col, iw_col, s2_col) {
                (Some(cv), Some(iw), Some(s2)) if cv.len() == human_scores.len() => {
                    let cv_min = cv.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let cv_max = cv.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let iw_min = iw.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let iw_max = iw.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let s2_min = s2.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let s2_max = s2.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let cv_range = (cv_max - cv_min).max(1e-9);
                    let iw_range = (iw_max - iw_min).max(1e-9);
                    let s2_range = (s2_max - s2_min).max(1e-9);
                    let sigmas: Vec<f64> = (0..cv.len())
                        .map(|i| {
                            let cn = (cv[i] - cv_min) / cv_range;
                            let in_ = (iw[i] - iw_min) / iw_range;
                            let sn = (s2[i] - s2_min) / s2_range;
                            let mean = (cn + in_ + sn) / 3.0;
                            let var =
                                ((cn - mean).powi(2) + (in_ - mean).powi(2) + (sn - mean).powi(2))
                                    / 3.0;
                            var.sqrt()
                        })
                        .collect();
                    eprintln!(
                        "  {name}: metric_sigmas computed from cvvdp/iwssim/ssim2 (mean={:.4}, p5={:.4}, p95={:.4})",
                        sigmas.iter().sum::<f64>() / sigmas.len() as f64,
                        {
                            let mut s = sigmas.clone();
                            s.sort_by(|a, b| a.total_cmp(b));
                            s[s.len() / 20]
                        },
                        {
                            let mut s = sigmas.clone();
                            s.sort_by(|a, b| a.total_cmp(b));
                            s[s.len() * 19 / 20]
                        },
                    );
                    Some(sigmas)
                }
                _ => None,
            }
        };
    } // end of disabled block

    println!(
        "  {name}: loaded {} pairs × {n_features} features from {path:?}",
        human_scores.len()
    );

    Ok(OwnedLoadedGroup {
        name: name.to_string(),
        train_w: 0.0,
        val_w: 0.0,
        human_scores,
        feature_rows,
        n_features,
        metric_sigmas,
    })
}

/// A multi-codec q-sweep grid for the DIAL panel: per row, the source
/// `image_id`, the `codec` family, the quality `q` (or q-equivalent for
/// distance-parameterized codecs), and the 372-feature vector. Rows are
/// grouped by `(image_id, codec)` and sorted by `q` to measure dial
/// monotonicity / tied-rate / per-q span across codec configurations.
#[derive(Debug)]
pub struct DialGrid {
    pub image_id: Vec<String>,
    pub codec: Vec<String>,
    pub q: Vec<f64>,
    pub feature_rows: Vec<Vec<f64>>,
    pub n_features: usize,
}

/// Load the dial-grid parquet (`image_id`, `codec`, `q`, `f0..f<N-1>`)
/// produced by `scripts/v_next/build_qsweep_expanded.py` and consolidated
/// to `eval-grids/dial_grid_372col_*.parquet`. Columns are located by
/// name (order-free); `q` may be Float32/Float64/Int. Powers the
/// `bake_verdict` DIAL panel (codec-target G1/G3/G4).
pub fn load_dial_grid(path: &PathBuf) -> Result<DialGrid, String> {
    let file = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("{path:?}: parquet open: {e}"))?;
    let schema = builder.schema().clone();
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().to_string()).collect();
    let idx = |n: &str| names.iter().position(|x| x == n);
    let img_i = idx("image_id").ok_or_else(|| format!("{path:?}: missing image_id column"))?;
    let codec_i = idx("codec").ok_or_else(|| format!("{path:?}: missing codec column"))?;
    let q_i = idx("q").ok_or_else(|| format!("{path:?}: missing q column"))?;
    let mut feat_idx = Vec::new();
    let mut fi = 0usize;
    while let Some(p) = idx(&format!("f{fi}")) {
        feat_idx.push(p);
        fi += 1;
    }
    let n_features = feat_idx.len();
    if n_features == 0 {
        return Err(format!("{path:?}: no fN feature columns found"));
    }
    let reader = builder
        .build()
        .map_err(|e| format!("{path:?}: parquet build reader: {e}"))?;

    let col_f64 = |col: &dyn Array, n_rows: usize| -> Result<Vec<f64>, String> {
        match col.data_type() {
            DataType::Float64 => {
                let a = col.as_any().downcast_ref::<Float64Array>().unwrap();
                Ok((0..n_rows).map(|i| a.value(i)).collect())
            }
            DataType::Float32 => {
                let a = col.as_any().downcast_ref::<Float32Array>().unwrap();
                Ok((0..n_rows).map(|i| a.value(i) as f64).collect())
            }
            DataType::Int64 => {
                let a = col.as_any().downcast_ref::<Int64Array>().unwrap();
                Ok((0..n_rows).map(|i| a.value(i) as f64).collect())
            }
            DataType::Int32 => {
                let a = col.as_any().downcast_ref::<Int32Array>().unwrap();
                Ok((0..n_rows).map(|i| a.value(i) as f64).collect())
            }
            other => Err(format!("unsupported numeric dtype {other:?}")),
        }
    };

    let mut image_id = Vec::new();
    let mut codec = Vec::new();
    let mut q = Vec::new();
    let mut feature_rows = Vec::new();
    for batch_res in reader {
        let batch = batch_res.map_err(|e| format!("{path:?}: parquet read batch: {e}"))?;
        let n_rows = batch.num_rows();
        if n_rows == 0 {
            continue;
        }
        let img_arr = batch
            .column(img_i)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| format!("{path:?}: image_id not Utf8"))?;
        let codec_arr = batch
            .column(codec_i)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| format!("{path:?}: codec not Utf8"))?;
        let q_v = col_f64(batch.column(q_i).as_ref(), n_rows)
            .map_err(|e| format!("{path:?}: q column {e}"))?;
        let per_col: Vec<Vec<f64>> = feat_idx
            .iter()
            .map(|&pi| col_f64(batch.column(pi).as_ref(), n_rows))
            .collect::<Result<_, _>>()
            .map_err(|e| format!("{path:?}: feature column {e}"))?;
        for r in 0..n_rows {
            image_id.push(img_arr.value(r).to_string());
            codec.push(codec_arr.value(r).to_string());
            q.push(q_v[r]);
            feature_rows.push((0..n_features).map(|c| per_col[c][r]).collect());
        }
    }
    Ok(DialGrid {
        image_id,
        codec,
        q,
        feature_rows,
        n_features,
    })
}

/// Load an optional scalar f64 column from a parquet file by name.
///
/// Returns `Ok(None)` if the column doesn't exist in the schema. Returns
/// `Ok(Some(values))` if the column is present and has Float32/Float64
/// dtype. Used by the EXP-CROSS-CODEC-V5 trainer to pick up the
/// `target_score` column from multi-band anchor parquets — when absent
/// (V4-style single-band parquets), the trainer falls back to the
/// `--anchor-target-score` CLI default.
///
/// Row ordering matches the order rows appear in `load_parquet`'s
/// returned `feature_rows` — both readers scan the parquet sequentially
/// with the same batch ordering.
pub fn load_optional_scalar_column(
    path: &PathBuf,
    column: &str,
) -> Result<Option<Vec<f64>>, String> {
    let file = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("{path:?}: parquet open: {e}"))?;
    let schema = builder.schema().clone();
    let parquet_schema = builder.parquet_schema().clone();

    let arrow_fields = schema.fields();
    let arrow_idx = match arrow_fields.iter().position(|f| f.name() == column) {
        Some(i) => i,
        None => return Ok(None),
    };

    let mask = ProjectionMask::leaves(&parquet_schema, [arrow_idx]);
    let reader = builder
        .with_projection(mask)
        .with_batch_size(16384)
        .build()
        .map_err(|e| format!("{path:?}: parquet build reader: {e}"))?;

    let mut values: Vec<f64> = Vec::new();
    for batch_res in reader {
        let batch = batch_res.map_err(|e| format!("{path:?}: parquet read batch: {e}"))?;
        let n_rows = batch.num_rows();
        if n_rows == 0 {
            continue;
        }
        let col = batch.column(0);
        match col.data_type() {
            DataType::Float64 => {
                let a = col.as_any().downcast_ref::<Float64Array>().unwrap();
                values.extend((0..n_rows).map(|i| a.value(i)));
            }
            DataType::Float32 => {
                let a = col.as_any().downcast_ref::<Float32Array>().unwrap();
                values.extend((0..n_rows).map(|i| a.value(i) as f64));
            }
            other => {
                return Err(format!(
                    "{path:?}: column {column:?} has unsupported dtype {other:?} (need Float32/Float64)",
                ));
            }
        }
    }

    Ok(Some(values))
}

/// KONJND-AGGREGATION-HEAD owned data pool (task #4, 2026-05-24).
///
/// Same row-major feature layout as [`OwnedLoadedGroup`] plus the
/// per-ref grouping metadata needed by the aggregation training step:
/// `ref_ranges[i] = (start_row, n_rows)` into `feature_rows` for ref i,
/// `ref_pjnd_target[i]` is that ref's per-source-constant target.
///
/// Constructed by [`load_konjnd_aggregation_pool`]. The trainer wraps
/// this in a `KonjndAggregationPool<'_>` (in `mlp_train.rs`) with
/// borrowed slices for the training-step hot path.
#[derive(Debug)]
pub struct OwnedKonjndAggregationPool {
    pub name: String,
    /// Flat row storage, one entry per (ref, distortion-level) pair.
    /// Length: `Σ ref_ranges[i].1`. Aligned with `ref_ranges`.
    pub feature_rows: Vec<Vec<f64>>,
    pub n_features: usize,
    /// Per-ref slice into `feature_rows`: `(start_row, n_rows)`.
    /// Length: n_refs.
    pub ref_ranges: Vec<(usize, usize)>,
    /// Per-ref pjnd_target (per-source-constant value). Length: n_refs.
    pub ref_pjnd_target: Vec<f64>,
    /// Per-ref training weight (defaults to 1.0). Length: n_refs.
    pub ref_weight: Vec<f64>,
}

/// Load konjnd-dense into a per-source-grouped aggregation pool.
///
/// Requires the parquet to carry `ref_basename` (utf8) and
/// `pjnd_target` (Float32/Float64) columns in addition to `f0..fN`.
/// Rows with the same `ref_basename` are grouped together; the
/// per-ref `pjnd_target` is taken from the first row in each group
/// (and asserted to be uniform across the group, since canonical
/// konjnd-dense always satisfies this — see
/// `benchmarks/recovery_phase3b_falsification_2026-05-21.md`).
///
/// Output rows are concatenated in ref-sorted order so the
/// per-ref ranges are contiguous and dense.
///
/// Errors:
/// - missing `ref_basename` column → `"<path>: missing ref_basename column"`
/// - missing `pjnd_target` column  → `"<path>: missing pjnd_target column"`
/// - non-uniform `pjnd_target` within a ref → returns the offending ref.
pub fn load_konjnd_aggregation_pool(
    path: &PathBuf,
    name: &str,
) -> Result<OwnedKonjndAggregationPool, String> {
    let file = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("{path:?}: parquet open: {e}"))?;
    let schema = builder.schema().clone();
    let parquet_schema = builder.parquet_schema().clone();
    let arrow_fields = schema.fields();
    let n_arrow_cols = arrow_fields.len();

    let ref_arrow_idx = arrow_fields
        .iter()
        .position(|f| f.name() == "ref_basename")
        .ok_or_else(|| format!("{path:?}: missing ref_basename column"))?;
    let pjnd_arrow_idx = arrow_fields
        .iter()
        .position(|f| f.name() == "pjnd_target")
        .ok_or_else(|| format!("{path:?}: missing pjnd_target column"))?;
    let f0_arrow_idx = arrow_fields
        .iter()
        .position(|f| f.name() == "f0")
        .ok_or_else(|| format!("{path:?}: missing f0 column"))?;

    let mut n_features = 0usize;
    while f0_arrow_idx + n_features < n_arrow_cols {
        let expected = format!("f{}", n_features);
        if arrow_fields[f0_arrow_idx + n_features].name() != &expected {
            break;
        }
        n_features += 1;
    }
    if n_features == 0 {
        return Err(format!("{path:?}: no fN columns found"));
    }

    let mut wanted: Vec<usize> = Vec::with_capacity(n_features + 2);
    wanted.push(ref_arrow_idx);
    wanted.push(pjnd_arrow_idx);
    for i in 0..n_features {
        wanted.push(f0_arrow_idx + i);
    }
    let mask = ProjectionMask::leaves(&parquet_schema, wanted.iter().copied());

    let reader = builder
        .with_projection(mask)
        .with_batch_size(16384)
        .build()
        .map_err(|e| format!("{path:?}: parquet build reader: {e}"))?;

    let mut sorted_wanted = wanted.clone();
    sorted_wanted.sort_unstable();
    let proj_ref_idx = sorted_wanted
        .iter()
        .position(|&i| i == ref_arrow_idx)
        .expect("ref idx must be in projection");
    let proj_pjnd_idx = sorted_wanted
        .iter()
        .position(|&i| i == pjnd_arrow_idx)
        .expect("pjnd idx must be in projection");
    let proj_feature_indices: Vec<usize> = (0..n_features)
        .map(|i| {
            sorted_wanted
                .iter()
                .position(|&p| p == f0_arrow_idx + i)
                .expect("feature idx must be in projection")
        })
        .collect();

    // Flat per-row accumulators. After full scan we group by ref_basename
    // and emit ref_ranges in sorted order.
    let mut all_refs: Vec<String> = Vec::new();
    let mut all_pjnd: Vec<f64> = Vec::new();
    let mut all_rows: Vec<Vec<f64>> = Vec::new();

    for batch_res in reader {
        let batch = batch_res.map_err(|e| format!("{path:?}: parquet read batch: {e}"))?;
        let n_rows = batch.num_rows();
        if n_rows == 0 {
            continue;
        }

        let ref_col = batch.column(proj_ref_idx);
        let ref_arr = ref_col
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                format!(
                    "{path:?}: ref_basename column has unsupported dtype {:?} (need Utf8)",
                    ref_col.data_type()
                )
            })?;
        for i in 0..n_rows {
            all_refs.push(ref_arr.value(i).to_string());
        }

        let pjnd_col = batch.column(proj_pjnd_idx);
        match pjnd_col.data_type() {
            DataType::Float64 => {
                let a = pjnd_col.as_any().downcast_ref::<Float64Array>().unwrap();
                all_pjnd.extend((0..n_rows).map(|i| a.value(i)));
            }
            DataType::Float32 => {
                let a = pjnd_col.as_any().downcast_ref::<Float32Array>().unwrap();
                all_pjnd.extend((0..n_rows).map(|i| a.value(i) as f64));
            }
            other => {
                return Err(format!(
                    "{path:?}: pjnd_target column has unsupported dtype {other:?} (need Float32/Float64)",
                ));
            }
        }

        // Per-batch feature transpose, same shape as load_parquet.
        let mut per_col_f64: Vec<Vec<f64>> = Vec::with_capacity(n_features);
        for &pi in &proj_feature_indices {
            let col = batch.column(pi);
            let v: Vec<f64> = match col.data_type() {
                DataType::Float64 => {
                    let a = col.as_any().downcast_ref::<Float64Array>().unwrap();
                    (0..n_rows).map(|i| a.value(i)).collect()
                }
                DataType::Float32 => {
                    let a = col.as_any().downcast_ref::<Float32Array>().unwrap();
                    (0..n_rows).map(|i| a.value(i) as f64).collect()
                }
                DataType::Int64 => {
                    let a = col.as_any().downcast_ref::<Int64Array>().unwrap();
                    (0..n_rows).map(|i| a.value(i) as f64).collect()
                }
                DataType::Int32 => {
                    let a = col.as_any().downcast_ref::<Int32Array>().unwrap();
                    (0..n_rows).map(|i| a.value(i) as f64).collect()
                }
                DataType::UInt64 => {
                    let a = col.as_any().downcast_ref::<UInt64Array>().unwrap();
                    (0..n_rows).map(|i| a.value(i) as f64).collect()
                }
                DataType::UInt32 => {
                    let a = col.as_any().downcast_ref::<UInt32Array>().unwrap();
                    (0..n_rows).map(|i| a.value(i) as f64).collect()
                }
                other => {
                    return Err(format!(
                        "{path:?}: feature column has unsupported dtype {other:?}",
                    ));
                }
            };
            per_col_f64.push(v);
        }
        for row_i in 0..n_rows {
            let mut row = Vec::with_capacity(n_features);
            for col in &per_col_f64 {
                row.push(col[row_i]);
            }
            all_rows.push(row);
        }
    }

    if all_rows.is_empty() {
        return Err(format!("{path:?}: empty file / no rows"));
    }
    if all_refs.len() != all_rows.len() || all_pjnd.len() != all_rows.len() {
        return Err(format!(
            "{path:?}: column length mismatch — refs={} pjnd={} rows={}",
            all_refs.len(),
            all_pjnd.len(),
            all_rows.len()
        ));
    }

    // Group by ref_basename in sort order (deterministic across runs).
    // Build per-ref index lists then concatenate rows in that order.
    let mut sorted_indices: Vec<usize> = (0..all_rows.len()).collect();
    sorted_indices.sort_by(|&a, &b| all_refs[a].cmp(&all_refs[b]));

    let mut feature_rows: Vec<Vec<f64>> = Vec::with_capacity(all_rows.len());
    let mut ref_ranges: Vec<(usize, usize)> = Vec::new();
    let mut ref_pjnd_target: Vec<f64> = Vec::new();
    let mut cursor = 0usize;
    let mut i = 0usize;
    while i < sorted_indices.len() {
        let ref_name = &all_refs[sorted_indices[i]];
        let start = cursor;
        let mut count = 0usize;
        let group_pjnd = all_pjnd[sorted_indices[i]];
        while i < sorted_indices.len() && all_refs[sorted_indices[i]] == *ref_name {
            // Assert pjnd_target is uniform within the ref group.
            let p = all_pjnd[sorted_indices[i]];
            if (p - group_pjnd).abs() > 1e-9 {
                return Err(format!(
                    "{path:?}: non-uniform pjnd_target within ref {ref_name:?} ({group_pjnd} vs {p})",
                ));
            }
            // Move the row into feature_rows. We mem::take to avoid clones.
            let row = std::mem::take(&mut all_rows[sorted_indices[i]]);
            feature_rows.push(row);
            cursor += 1;
            count += 1;
            i += 1;
        }
        ref_ranges.push((start, count));
        ref_pjnd_target.push(group_pjnd);
    }

    let ref_weight = vec![1.0_f64; ref_ranges.len()];

    println!(
        "  {name}: konjnd-aggregation loaded {} rows × {n_features} features grouped into {} refs from {path:?}",
        feature_rows.len(),
        ref_ranges.len()
    );

    Ok(OwnedKonjndAggregationPool {
        name: name.to_string(),
        feature_rows,
        n_features,
        ref_ranges,
        ref_pjnd_target,
        ref_weight,
    })
}
