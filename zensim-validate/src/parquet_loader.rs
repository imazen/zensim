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
    Array, Float32Array, Float64Array, Int32Array, Int64Array, UInt32Array, UInt64Array,
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
    })
}
