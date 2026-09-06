//! Parquet loader for zensim feature training data.
//!
//! Produces an [`OwnedLoadedGroup`] that mirrors `LoadedGroup` in
//! `zensim-validate/src/bin/zensim_mlp_train.rs` field-for-field.
//!
//! A `target_column` (e.g. `iwssim_log_norm`) plus consecutive feature
//! columns, named either `f0, f1, ...` (the canonical-corpus / CSV
//! convention) or `feat_0, feat_1, ...` (what zenmetrics sidecars and
//! the pareto-sweep extractor emit). Both name the same 372-wide with-iw
//! space. `ref_basename` / `image_path` is read as per-row reference
//! identity when present; everything else in the file is ignored.
//!
//! THIS IS THE ONLY FEATURE-PARQUET LOADER. Per the "NO DUPLICATE
//! IMPLEMENTATIONS" rule in CLAUDE.md, do not read a feature parquet
//! anywhere else (in Rust or Python) on the way to training/eval — if
//! this can't express what you need, extend it here.
//!
//! Two materializing shapes over ONE shared scan ([`load_parquet_impl`]):
//! [`load_parquet`] emits per-row `Vec`s ([`OwnedLoadedGroup`] — the
//! compat shape every non-trainer consumer keeps), and
//! [`load_parquet_flat`] emits one flat row-major buffer
//! ([`OwnedLoadedGroupFlat`] — the memory-shaped variant the trainer
//! adopts; see its doc for the ~1.9 GB load-transient rationale). Values
//! are bit-identical between the shapes (`tests/parquet_flat_equivalence.rs`).
//!
//! Wired into the trainer binary via `From<OwnedLoadedGroupFlat> for
//! LoadedGroup`. Equivalence to the sequential CSV loader is verified by
//! `tests/parquet_load_equivalence.rs`; ref-identity + prefix handling by
//! `tests/within_ref_pairing.rs`.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use arrow::array::{
    Array, Float32Array, Float64Array, Int32Array, Int64Array, LargeStringArray, StringArray,
    UInt32Array, UInt64Array,
};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// Rows per transpose block (see [`load_parquet`]). 1024 × 944 columns is a
/// ~7.7 MB f64 staging buffer — small enough to sit in L3 next to a dozen
/// concurrent loaders, large enough that the per-block column downcast is
/// amortized over 1024 forward-walked values.
const TRANSPOSE_BLOCK_ROWS: usize = 1024;

/// Append `block_len` values of `col` starting at `start`, widened to f64,
/// to `out`. Accepts the dtypes zensim feature parquets actually carry:
/// Float64/Float32 (the common case) plus the integer widths some
/// extractors emit for count-style features (e.g. KonJND's f12).
fn col_block_to_f64(
    path: &Path,
    col: &dyn Array,
    start: usize,
    block_len: usize,
    out: &mut Vec<f64>,
) -> Result<(), String> {
    let end = start + block_len;
    match col.data_type() {
        DataType::Float64 => {
            let a = col.as_any().downcast_ref::<Float64Array>().unwrap();
            out.extend((start..end).map(|i| a.value(i)));
        }
        DataType::Float32 => {
            let a = col.as_any().downcast_ref::<Float32Array>().unwrap();
            out.extend((start..end).map(|i| a.value(i) as f64));
        }
        DataType::Int64 => {
            let a = col.as_any().downcast_ref::<Int64Array>().unwrap();
            out.extend((start..end).map(|i| a.value(i) as f64));
        }
        DataType::Int32 => {
            let a = col.as_any().downcast_ref::<Int32Array>().unwrap();
            out.extend((start..end).map(|i| a.value(i) as f64));
        }
        DataType::UInt64 => {
            let a = col.as_any().downcast_ref::<UInt64Array>().unwrap();
            out.extend((start..end).map(|i| a.value(i) as f64));
        }
        DataType::UInt32 => {
            let a = col.as_any().downcast_ref::<UInt32Array>().unwrap();
            out.extend((start..end).map(|i| a.value(i) as f64));
        }
        other => {
            return Err(format!(
                "{path:?}: feature column has unsupported dtype {other:?} (need Float32/Float64/Int32/Int64/UInt32/UInt64)",
            ));
        }
    }
    Ok(())
}

/// Owned mirror of the trainer's `LoadedGroup` struct.
///
/// The trainer (in `bin/zensim_mlp_train.rs`) defines `LoadedGroup`
/// privately; this owned shape crosses the lib/bin boundary and converts
/// via the `From<OwnedLoadedGroup> for LoadedGroup` shim there. Keep the
/// two field-for-field: adding a field here means adding it there.
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
    /// Per-row reference-image identity, densely numbered `0..n_refs`.
    /// Read from `ref_basename` when present, else `image_path` (the
    /// zenmetrics sidecar / pareto-sweep convention). `None` when the
    /// parquet carries neither.
    ///
    /// This is what makes WITHIN-REF pair sampling possible. It matters
    /// because a RankNet pair drawn across images teaches "image A
    /// scores above image B" — a cross-image *scale* fact — while a pair
    /// drawn within one image teaches "this distortion is worse than
    /// that one", the actual ranking task. On corpora whose within-image
    /// ladder is small next to the between-image spread, the cross-image
    /// draw drowns the signal: on the post-jxl-fix near-lossless corpus
    /// the ssim2 ladder moves ~0.92 pts within an image against ~6 pts
    /// between images, which is why its pooled SROCC reads +0.204 while
    /// its per-ref SROCC reads +0.916 (the same confound as the
    /// documented AIC-3 "0.79 pooled / 0.93 per-ref").
    pub ref_ids: Option<Vec<u32>>,
}

/// [`OwnedLoadedGroup`] with the features as ONE flat row-major buffer
/// instead of per-row `Vec`s — the memory-shaped variant for consumers
/// (the trainer) that flatten anyway.
///
/// `features_flat[i * n_features + d]` is feature `d` of row `i`;
/// `n_rows == human_scores.len()`. Values and their order are IDENTICAL
/// to [`OwnedLoadedGroup::feature_rows`] for the same file
/// (`features_flat[i * n_features + d] == feature_rows[i][d]`, bit for
/// bit) — asserted by `tests/parquet_flat_equivalence.rs`.
///
/// Why this exists (`benchmarks/trainer_mem_release_2026-08-04.md`, "next
/// lever"): the per-row shape costs a load transient — the largest group
/// exists as ~7.5 KB row chunks AND as the trainer's flat copy during
/// flattening (~1.9 GB extra peak on the wave-10/11 recipes), and the
/// freed row chunks are interior free-list holes glibc cannot return.
/// Emitting flat at the loader means the matrix is the only allocation:
/// pre-reserved once from the parquet footer's row count, filled in
/// place, never copied.
#[derive(Debug)]
pub struct OwnedLoadedGroupFlat {
    pub name: String,
    pub train_w: f64,
    pub val_w: f64,
    pub human_scores: Vec<f64>,
    /// Row-major `n_rows × n_features`, tightly packed.
    pub features_flat: Vec<f64>,
    pub n_features: usize,
    /// See [`OwnedLoadedGroup::metric_sigmas`].
    pub metric_sigmas: Option<Vec<f64>>,
    /// See [`OwnedLoadedGroup::ref_ids`].
    pub ref_ids: Option<Vec<u32>>,
}

/// Feature-matrix destination for [`load_parquet_impl`] — same scan, same
/// per-block transpose walk, two emission shapes.
enum FeatureStore {
    Rows(Vec<Vec<f64>>),
    Flat(Vec<f64>),
}

/// Fields shared by both loader shapes (everything except the features).
struct LoadedCommon {
    human_scores: Vec<f64>,
    n_features: usize,
    metric_sigmas: Option<Vec<f64>>,
    ref_ids: Option<Vec<u32>>,
}

/// Load a zensim training feature parquet file.
///
/// Reads the `target_column` (multiplied by `target_scale` to match the
/// trainer's score units) plus the consecutive feature columns, accepting
/// either the `f<i>` or `feat_<i>` prefix. Also reads `ref_basename` (else
/// `image_path`) into [`OwnedLoadedGroup::ref_ids`] when present. Other
/// columns are skipped. Returns an [`OwnedLoadedGroup`] with `train_w` /
/// `val_w` set to `0.0` — callers fill those in from the group-spec parser.
///
/// Errors mirror the CSV loader's phrasing so the binary's error
/// handling stays uniform:
/// - missing `target_column`   -> `"<path>: missing target column ..."`
/// - missing both prefixes     -> `"<path>: missing f0 / feat_0 column"`
/// - no feature columns        -> `"<path>: no <prefix>N columns found"`
/// - empty file                -> `"<path>: empty file ..."`
///
/// **THE owner of "where are this table's feature columns".**
///
/// Returns `(prefix, arrow index of `<prefix>0`, n_features)` for a table whose
/// feature columns are the CONTIGUOUS run `<prefix>0..<prefix>{n-1}` sitting at
/// consecutive arrow positions — which is every table on disk today, and for
/// which this is byte-for-byte the walk each loader used to inline.
///
/// **It REFUSES a gapped id set instead of truncating at the gap**, and that is
/// the whole reason it exists. Each of the five loaders used to stop at the
/// first name that did not match, so a DENSE-BY-ID table — `f0..f155,
/// f372..f943`, which `rescore_parquet --densify` now produces — would have
/// loaded as a 156-wide table with no error at all, and every number computed
/// from it would have been about a different feature space. A silent truncation
/// in a loader is the same defect class as a positional slice in a scorer: the
/// numbers come back, and they are about the wrong columns.
///
/// Reading a dense table is a separate, registered step (the row has to become
/// id-indexed end to end, not merely wider). Until it lands, "refuse" is the
/// correct answer and "truncate" never was.
fn feature_column_run(
    path: &std::path::Path,
    arrow_fields: &[std::sync::Arc<arrow::datatypes::Field>],
) -> Result<(&'static str, usize, usize), String> {
    let n_arrow_cols = arrow_fields.len();
    let (prefix, f0) = ["f", "feat_"]
        .iter()
        .find_map(|p| {
            arrow_fields
                .iter()
                .position(|f| f.name() == &format!("{p}0"))
                .map(|i| (*p, i))
        })
        .ok_or_else(|| format!("{path:?}: missing f0 / feat_0 column"))?;
    let mut n = 0usize;
    while f0 + n < n_arrow_cols {
        let expected = format!("{prefix}{n}");
        if arrow_fields[f0 + n].name() != &expected {
            break;
        }
        n += 1;
    }
    if n == 0 {
        return Err(format!("{path:?}: no {prefix}N columns found"));
    }
    // The GAP check: any `<prefix><id>` column with `id >= n` means the run
    // stopped early on a table that HAS more feature columns — a dense-by-id
    // layout, not a narrow table.
    let beyond: Vec<usize> = arrow_fields
        .iter()
        .filter_map(|f| {
            f.name()
                .strip_prefix(prefix)
                .and_then(|r| r.parse::<usize>().ok())
        })
        .filter(|id| *id >= n)
        .collect();
    if !beyond.is_empty() {
        let lo = beyond.iter().copied().min().unwrap_or(0);
        let hi = beyond.iter().copied().max().unwrap_or(0);
        return Err(format!(
            "{path:?}: feature columns are NOT the contiguous run {prefix}0..{prefix}{} — \
             {} more column(s) exist at ids {prefix}{lo}..{prefix}{hi}, so this is a DENSE-BY-ID \
             table. Loading it as {n}-wide would silently be about a different feature space. \
             Score it at its wide source, or wait for the id-indexed loader.",
            n - 1,
            beyond.len(),
        ));
    }
    Ok((prefix, f0, n))
}

/// The name-indexed sibling of [`feature_column_run`], for the loaders that
/// find each column by NAME rather than by consecutive arrow position (the
/// dial grid, the corruption grid, the per-pair metric table). Same contract:
/// the contiguous run today, byte-for-byte, and a LOUD REFUSAL on a gapped
/// (dense-by-id) table rather than a silent stop at the gap.
fn feature_column_run_by_name(
    path: &std::path::Path,
    names: &[String],
    prefix: &str,
) -> Result<Vec<usize>, String> {
    let idx = |n: &str| names.iter().position(|x| x == n);
    let mut out = Vec::new();
    let mut fi = 0usize;
    while let Some(p) = idx(&format!("{prefix}{fi}")) {
        out.push(p);
        fi += 1;
    }
    if out.is_empty() {
        return Err(format!("{path:?}: no {prefix}N feature columns found"));
    }
    let beyond: Vec<usize> = names
        .iter()
        .filter_map(|n| n.strip_prefix(prefix).and_then(|r| r.parse::<usize>().ok()))
        .filter(|id| *id >= out.len())
        .collect();
    if !beyond.is_empty() {
        let lo = beyond.iter().copied().min().unwrap_or(0);
        let hi = beyond.iter().copied().max().unwrap_or(0);
        return Err(format!(
            "{path:?}: feature columns are NOT the contiguous run {prefix}0..{prefix}{} — \
             {} more column(s) exist at ids {prefix}{lo}..{prefix}{hi}, so this is a \
             DENSE-BY-ID grid. Loading it as {}-wide would silently be about a different \
             feature space.",
            out.len() - 1,
            beyond.len(),
            out.len(),
        ));
    }
    Ok(out)
}

pub fn load_parquet(
    path: &PathBuf,
    name: &str,
    target_column: &str,
    target_scale: f64,
) -> Result<OwnedLoadedGroup, String> {
    let (common, store) = load_parquet_impl(path, name, target_column, target_scale, false)?;
    let feature_rows = match store {
        FeatureStore::Rows(rows) => rows,
        FeatureStore::Flat(_) => unreachable!("load_parquet_impl(flat=false) must emit Rows"),
    };
    Ok(OwnedLoadedGroup {
        name: name.to_string(),
        train_w: 0.0,
        val_w: 0.0,
        human_scores: common.human_scores,
        feature_rows,
        n_features: common.n_features,
        metric_sigmas: common.metric_sigmas,
        ref_ids: common.ref_ids,
    })
}

/// [`load_parquet`], emitting the features as ONE flat row-major buffer
/// (see [`OwnedLoadedGroupFlat`] for why). Identical column conventions,
/// identical values in identical order, identical error phrasing — the
/// only difference is the feature-matrix shape and its allocation
/// profile (one pre-reserved buffer instead of `n_rows` row `Vec`s).
pub fn load_parquet_flat(
    path: &PathBuf,
    name: &str,
    target_column: &str,
    target_scale: f64,
) -> Result<OwnedLoadedGroupFlat, String> {
    let (common, store) = load_parquet_impl(path, name, target_column, target_scale, true)?;
    let features_flat = match store {
        FeatureStore::Flat(flat) => flat,
        FeatureStore::Rows(_) => unreachable!("load_parquet_impl(flat=true) must emit Flat"),
    };
    Ok(OwnedLoadedGroupFlat {
        name: name.to_string(),
        train_w: 0.0,
        val_w: 0.0,
        human_scores: common.human_scores,
        features_flat,
        n_features: common.n_features,
        metric_sigmas: common.metric_sigmas,
        ref_ids: common.ref_ids,
    })
}

/// Shared body of [`load_parquet`] / [`load_parquet_flat`]. One scan, one
/// per-block transpose walk; `flat` selects the emission shape.
fn load_parquet_impl(
    path: &PathBuf,
    name: &str,
    target_column: &str,
    target_scale: f64,
    flat: bool,
) -> Result<(LoadedCommon, FeatureStore), String> {
    let file = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("{path:?}: parquet open: {e}"))?;
    let schema = builder.schema().clone();
    let parquet_schema = builder.parquet_schema().clone();
    // Total row count from the parquet footer — used to pre-reserve the
    // flat feature buffer in one allocation.
    let total_rows_meta = builder.metadata().file_metadata().num_rows().max(0) as usize;

    // Build a (leaf-index, column-name) view for projection + dispatch.
    // We use the Arrow `schema` for top-level names (these match the
    // CSV header names) and `parquet_schema` for the leaf projection
    // mask (Parquet schema is column-tree shaped, Arrow is flat for
    // primitive types — which is what zensim features always are).
    let arrow_fields = schema.fields();

    // Locate the target column by name.
    let score_arrow_idx = arrow_fields
        .iter()
        .position(|f| f.name() == target_column)
        .ok_or_else(|| format!("{path:?}: missing target column {target_column:?}"))?;

    // Locate the first feature column, then count consecutive ones.
    // Mirrors the CSV loader's scan precisely so the same source data
    // produces the same `n_features`.
    //
    // TWO PREFIXES are accepted. `f<i>` is the canonical-corpus /
    // CSV-loader convention (canonical-2026-05-21 train+val). `feat_<i>`
    // is what zenmetrics sidecars and the pareto-sweep extractor emit
    // (e.g. the post-jxl-fix near-lossless corpus). Both name the same
    // 372-wide with-iw feature space; only the header text differs, so
    // rejecting one of them just forces a rename-copy of the parquet.
    let (prefix, f0_arrow_idx, n_features) = feature_column_run(path, arrow_fields)?;
    let _ = prefix;

    // Optional reference-identity column, for within-ref pair sampling.
    // `ref_basename` is the canonical-corpus convention; `image_path` is
    // the sidecar / pareto-sweep one. Absent on neither -> ref_ids None.
    let ref_arrow_idx = ["ref_basename", "image_path"]
        .iter()
        .find_map(|c| arrow_fields.iter().position(|f| f.name() == c));

    // Project just the columns we need. For primitive (non-nested)
    // schemas — which zensim feature parquets always are — the
    // arrow-column index maps 1:1 to a parquet leaf index. We use
    // `ProjectionMask::leaves` indexed by these arrow positions.
    let mut wanted: Vec<usize> = Vec::with_capacity(n_features + 2);
    wanted.push(score_arrow_idx);
    if let Some(r) = ref_arrow_idx {
        wanted.push(r);
    }
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
    let proj_ref_idx = ref_arrow_idx.map(|r| {
        sorted_wanted
            .iter()
            .position(|&p| p == r)
            .expect("ref idx must be in projection")
    });

    let mut human_scores: Vec<f64> = Vec::new();
    // Flat: pre-reserve the whole matrix from the parquet footer's row
    // count, so the buffer is allocated ONCE and never realloc-copied —
    // at 944 features × 200k+ rows a growth-doubling realloc would
    // transiently hold ~1.5× the matrix, which defeats the point of the
    // flat shape. (The count is a hint: if a corrupt footer under-reports,
    // the Vec still grows correctly.)
    let mut store = if flat {
        FeatureStore::Flat(Vec::with_capacity(
            total_rows_meta.saturating_mul(n_features),
        ))
    } else {
        FeatureStore::Rows(Vec::new())
    };
    // Dense ref numbering, assigned in first-seen order so the ids are
    // deterministic for a given file.
    let mut ref_ids: Vec<u32> = Vec::new();
    let mut ref_lookup: HashMap<String, u32> = HashMap::new();
    // Column-major staging for one row block; reused across blocks + batches.
    let mut per_col_scratch: Vec<f64> = Vec::with_capacity(n_features * TRANSPOSE_BLOCK_ROWS);

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

        // Extract the reference-identity column, mapping each distinct
        // string to a dense u32 in first-seen order.
        if let Some(pi) = proj_ref_idx {
            let col = batch.column(pi);
            let as_str: Box<dyn Fn(usize) -> String> = match col.data_type() {
                DataType::Utf8 => {
                    let a = col.as_any().downcast_ref::<StringArray>().unwrap().clone();
                    Box::new(move |i| a.value(i).to_string())
                }
                DataType::LargeUtf8 => {
                    let a = col
                        .as_any()
                        .downcast_ref::<LargeStringArray>()
                        .unwrap()
                        .clone();
                    Box::new(move |i| a.value(i).to_string())
                }
                other => {
                    return Err(format!(
                        "{path:?}: ref column has unsupported dtype {other:?} (need Utf8/LargeUtf8)",
                    ));
                }
            };
            for i in 0..n_rows {
                let key = as_str(i);
                let next = ref_lookup.len() as u32;
                let id = *ref_lookup.entry(key).or_insert(next);
                ref_ids.push(id);
            }
        }

        // Columnar -> row-major, one ROW BLOCK at a time.
        //
        // The column-major staging buffer is what keeps the transpose
        // cache-friendly (each source array is walked forward once), but
        // staging a WHOLE 16 384-row batch of 944 columns costs 124 MB of
        // scratch per open loader — and once the corpus loop went parallel
        // that is 12 of them at once. Blocking the transpose at
        // [`TRANSPOSE_BLOCK_ROWS`] keeps the same forward walk with a ~7.7 MB
        // working set that fits in L3, independent of the parquet batch size.
        // Values, order, and the resulting features are unchanged — and
        // identical between the two emission shapes: both walk
        // `(r, c) -> per_col_scratch[c * block_len + r]` in the same order,
        // so `flat[i * n_features + d] == rows[i][d]` bit for bit.
        if let FeatureStore::Rows(rows) = &mut store {
            rows.reserve(n_rows);
        }
        let mut block_start = 0usize;
        while block_start < n_rows {
            let block_len = TRANSPOSE_BLOCK_ROWS.min(n_rows - block_start);
            per_col_scratch.clear();
            for &pi in &proj_feature_indices {
                let col = batch.column(pi);
                // Feature columns may be Float64/Float32 (the common case)
                // OR an integer type (Int32/Int64/UInt32/UInt64). Some
                // feature extractors emit count-style features (e.g.
                // KonJND's f12) as integers; the trainer / runtime
                // consumes them as f64 anyway, so widen silently.
                col_block_to_f64(
                    path,
                    col.as_ref(),
                    block_start,
                    block_len,
                    &mut per_col_scratch,
                )?;
            }
            match &mut store {
                FeatureStore::Rows(rows) => {
                    for r in 0..block_len {
                        let mut row = Vec::with_capacity(n_features);
                        for c in 0..n_features {
                            row.push(per_col_scratch[c * block_len + r]);
                        }
                        rows.push(row);
                    }
                }
                FeatureStore::Flat(flat_buf) => {
                    for r in 0..block_len {
                        for c in 0..n_features {
                            flat_buf.push(per_col_scratch[c * block_len + r]);
                        }
                    }
                }
            }
            block_start += block_len;
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
    // STRATEGY-2026-07-02: re-enabled behind ZENSIM_SIGMA_MSE=1 (the
    // sigma_weighted_mse flag's data side). Column names accept BOTH the
    // canonical-2026-05-21 scheme (cvvdp_score/iwssim/ssim2_gpu) and the
    // mm6 multimetric scheme (score_cvvdp/score_iwssim/score_dssim).
    let sigma_on = std::env::var("ZENSIM_SIGMA_MSE")
        .map(|v| v == "1")
        .unwrap_or(false);
    let mut metric_sigmas: Option<Vec<f64>> = None;
    if sigma_on {
        metric_sigmas = {
            let cv_col = load_optional_scalar_column(path, "cvvdp_score")
                .ok()
                .flatten()
                .or_else(|| {
                    load_optional_scalar_column(path, "score_cvvdp")
                        .ok()
                        .flatten()
                });
            let iw_col = load_optional_scalar_column(path, "iwssim")
                .ok()
                .flatten()
                .or_else(|| {
                    load_optional_scalar_column(path, "score_iwssim")
                        .ok()
                        .flatten()
                });
            let s2_col = load_optional_scalar_column(path, "ssim2_gpu")
                .ok()
                .flatten()
                .or_else(|| {
                    load_optional_scalar_column(path, "score_dssim")
                        .ok()
                        .flatten()
                });
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
        if metric_sigmas.is_none() {
            eprintln!(
                "  WARNING: ZENSIM_SIGMA_MSE=1 but no metric columns found — sigmas unavailable for this group"
            );
        }
    }

    println!(
        "  {name}: loaded {} pairs × {n_features} features ({prefix}0..{prefix}{}) from {path:?}{}",
        human_scores.len(),
        n_features - 1,
        match ref_lookup.len() {
            0 => String::new(),
            n => format!(" [{n} refs]"),
        }
    );

    Ok((
        LoadedCommon {
            human_scores,
            n_features,
            metric_sigmas,
            ref_ids: if ref_ids.is_empty() {
                None
            } else {
                Some(ref_ids)
            },
        },
        store,
    ))
}

/// Batch callback for [`stream_parquet_rows`]: `(features_row_major,
/// n_rows, targets_per_column)`.
pub type StreamBatchFn<'a> = dyn FnMut(&[f64], usize, &[Vec<f64>]) -> Result<(), String> + 'a;

/// Summary of one [`stream_parquet_rows`] pass.
#[derive(Debug)]
pub struct StreamInfo {
    pub n_rows: usize,
    pub n_features: usize,
}

/// STREAMING row visitor over a feature parquet — the memory-capped sibling
/// of [`load_parquet`] for accumulation passes (Gram moments, running
/// statistics) where materializing every row (`Vec<Vec<f64>>`) would need
/// tens of GB. Same column conventions as `load_parquet` (THE loader owner —
/// extend here, never re-read feature parquets elsewhere): consecutive
/// `f<i>` / `feat_<i>` feature columns, named target columns (multiple
/// allowed, each scaled by `target_scale`), Float32/Float64/int widening.
///
/// Determinism: single-threaded, batches in file order, rows in batch order
/// — a given file yields one exact visit sequence.
///
/// Contract differences vs `load_parquet` (accumulation passes must not
/// silently absorb bad data):
/// * errors on any NULL in a projected column (feature or target);
/// * errors on any non-finite feature or target value;
/// * no ref-identity / sigma handling (not needed for moment accumulation).
///
/// `on_batch(features, n_rows, targets)`: `features` is row-major
/// `n_rows × n_features` (buffer reused between batches), `targets[t]` has
/// `n_rows` scaled values for `target_columns[t]`.
pub fn stream_parquet_rows(
    path: &PathBuf,
    target_columns: &[&str],
    target_scale: f64,
    on_batch: &mut StreamBatchFn<'_>,
) -> Result<StreamInfo, String> {
    let file = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("{path:?}: parquet open: {e}"))?;
    let schema = builder.schema().clone();
    let parquet_schema = builder.parquet_schema().clone();
    let arrow_fields = schema.fields();

    // Target columns by name (same error phrasing as load_parquet).
    let mut target_arrow_idx: Vec<usize> = Vec::with_capacity(target_columns.len());
    for t in target_columns {
        let i = arrow_fields
            .iter()
            .position(|f| f.name() == t)
            .ok_or_else(|| format!("{path:?}: missing target column {t:?}"))?;
        target_arrow_idx.push(i);
    }

    // First feature column + consecutive count — mirrors load_parquet.
    let (prefix, f0_arrow_idx, n_features) = feature_column_run(path, arrow_fields)?;
    let _ = prefix;

    let mut wanted: Vec<usize> = Vec::with_capacity(n_features + target_columns.len());
    wanted.extend(target_arrow_idx.iter().copied());
    for i in 0..n_features {
        wanted.push(f0_arrow_idx + i);
    }
    let mask = ProjectionMask::leaves(&parquet_schema, wanted.iter().copied());
    let reader = builder
        .with_projection(mask)
        .with_batch_size(16384)
        .build()
        .map_err(|e| format!("{path:?}: parquet build reader: {e}"))?;

    // Post-projection indices (projection emits ascending original order).
    let mut sorted_wanted = wanted.clone();
    sorted_wanted.sort_unstable();
    sorted_wanted.dedup();
    let proj_of = |orig: usize| -> usize {
        sorted_wanted
            .iter()
            .position(|&p| p == orig)
            .expect("index must be in projection")
    };
    let proj_target_indices: Vec<usize> = target_arrow_idx.iter().map(|&i| proj_of(i)).collect();
    let proj_feature_indices: Vec<usize> =
        (0..n_features).map(|i| proj_of(f0_arrow_idx + i)).collect();

    // Column extractor shared by targets and features: any null or
    // non-finite value is a hard error (accumulation contract).
    let col_to_f64 = |batch: &RecordBatch,
                      pi: usize,
                      colname: &str,
                      scale: f64,
                      out: &mut Vec<f64>|
     -> Result<(), String> {
        let col = batch.column(pi);
        if col.null_count() > 0 {
            return Err(format!(
                "{path:?}: column {colname:?} has {} NULLs — accumulation passes reject nulls",
                col.null_count()
            ));
        }
        let n_rows = batch.num_rows();
        out.clear();
        out.reserve(n_rows);
        match col.data_type() {
            DataType::Float64 => {
                let a = col.as_any().downcast_ref::<Float64Array>().unwrap();
                out.extend((0..n_rows).map(|i| a.value(i) * scale));
            }
            DataType::Float32 => {
                let a = col.as_any().downcast_ref::<Float32Array>().unwrap();
                out.extend((0..n_rows).map(|i| a.value(i) as f64 * scale));
            }
            DataType::Int64 => {
                let a = col.as_any().downcast_ref::<Int64Array>().unwrap();
                out.extend((0..n_rows).map(|i| a.value(i) as f64 * scale));
            }
            DataType::Int32 => {
                let a = col.as_any().downcast_ref::<Int32Array>().unwrap();
                out.extend((0..n_rows).map(|i| a.value(i) as f64 * scale));
            }
            DataType::UInt64 => {
                let a = col.as_any().downcast_ref::<UInt64Array>().unwrap();
                out.extend((0..n_rows).map(|i| a.value(i) as f64 * scale));
            }
            DataType::UInt32 => {
                let a = col.as_any().downcast_ref::<UInt32Array>().unwrap();
                out.extend((0..n_rows).map(|i| a.value(i) as f64 * scale));
            }
            other => {
                return Err(format!(
                    "{path:?}: column {colname:?} has unsupported dtype {other:?} (need Float32/Float64/int)",
                ));
            }
        }
        if let Some(bad) = out.iter().position(|v| !v.is_finite()) {
            return Err(format!(
                "{path:?}: column {colname:?} has non-finite value {} at in-batch row {bad}",
                out[bad]
            ));
        }
        Ok(())
    };

    let mut total_rows = 0usize;
    let mut features_rm: Vec<f64> = Vec::new();
    let mut targets: Vec<Vec<f64>> = vec![Vec::new(); target_columns.len()];
    let mut col_buf: Vec<f64> = Vec::new();
    for batch_res in reader {
        let batch = batch_res.map_err(|e| format!("{path:?}: parquet read batch: {e}"))?;
        let n_rows = batch.num_rows();
        if n_rows == 0 {
            continue;
        }
        for (t, &pi) in proj_target_indices.iter().enumerate() {
            col_to_f64(&batch, pi, target_columns[t], target_scale, &mut targets[t])?;
        }
        // Row-major feature buffer: fill column-wise into strided slots.
        features_rm.clear();
        features_rm.resize(n_rows * n_features, 0.0);
        for (fi, &pi) in proj_feature_indices.iter().enumerate() {
            col_to_f64(&batch, pi, &format!("{prefix}{fi}"), 1.0, &mut col_buf)?;
            for (r, v) in col_buf.iter().enumerate() {
                features_rm[r * n_features + fi] = *v;
            }
        }
        on_batch(&features_rm, n_rows, &targets)?;
        total_rows += n_rows;
    }
    if total_rows == 0 {
        return Err(format!("{path:?}: empty file / no rows"));
    }
    Ok(StreamInfo {
        n_rows: total_rows,
        n_features,
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
    /// Native codec-config parameter: integer quality for q-codecs, or
    /// butteraugli distance for distance-parameterized codecs (JXL).
    /// Falls back to `q` if the parquet predates the column.
    pub codec_param: Vec<f64>,
    /// Per-row label for `codec_param`: "q" or "distance". Falls back to
    /// "q" if the parquet predates the column.
    pub param_kind: Vec<String>,
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
    let names: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect();
    let idx = |n: &str| names.iter().position(|x| x == n);
    let img_i = idx("image_id").ok_or_else(|| format!("{path:?}: missing image_id column"))?;
    let codec_i = idx("codec").ok_or_else(|| format!("{path:?}: missing codec column"))?;
    let q_i = idx("q").ok_or_else(|| format!("{path:?}: missing q column"))?;
    // Optional native codec-param columns (added 2026-05-29). Fall back to q.
    let cp_i = idx("codec_param");
    let pk_i = idx("param_kind");
    let feat_idx = feature_column_run_by_name(path, &names, "f")?;
    let n_features = feat_idx.len();
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
    let mut codec_param = Vec::new();
    let mut param_kind = Vec::new();
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
        // Native codec param: read codec_param column if present, else fall
        // back to q (older grids). param_kind likewise defaults to "q".
        let cp_v = match cp_i {
            Some(i) => col_f64(batch.column(i).as_ref(), n_rows)
                .map_err(|e| format!("{path:?}: codec_param column {e}"))?,
            None => q_v.clone(),
        };
        let pk_v: Vec<String> =
            match pk_i.and_then(|i| batch.column(i).as_any().downcast_ref::<StringArray>()) {
                Some(a) => (0..n_rows).map(|r| a.value(r).to_string()).collect(),
                None => vec!["q".to_string(); n_rows],
            };
        let per_col: Vec<Vec<f64>> = feat_idx
            .iter()
            .map(|&pi| col_f64(batch.column(pi).as_ref(), n_rows))
            .collect::<Result<_, _>>()
            .map_err(|e| format!("{path:?}: feature column {e}"))?;
        for r in 0..n_rows {
            image_id.push(img_arr.value(r).to_string());
            codec.push(codec_arr.value(r).to_string());
            q.push(q_v[r]);
            codec_param.push(cp_v[r]);
            param_kind.push(pk_v[r].clone());
            feature_rows.push((0..n_features).map(|c| per_col[c][r]).collect());
        }
    }
    Ok(DialGrid {
        image_id,
        codec,
        q,
        codec_param,
        param_kind,
        feature_rows,
        n_features,
    })
}

/// A distortion severity-ramp grid for the severity-ramp monotonicity
/// section: per row the source `image` basename, the `q` code (which
/// encodes `dist_type * 10 + severity_level`, levels 1..5), and the
/// feature vector. Groups by `(image, dist_type)` and checks the dial is
/// non-increasing as severity rises. Mirrors the schema produced by the
/// kadis-distort / kadis-hdr feature sidecars (`image_path`, `q`,
/// `feat_0..feat_<N-1>`).
#[derive(Debug)]
pub struct RampGrid {
    pub image: Vec<String>,
    pub q: Vec<f64>,
    pub feature_rows: Vec<Vec<f64>>,
    pub n_features: usize,
}

/// Load a severity-ramp feature grid. The image column is located as
/// `image_path` → `image_id` → `ref_basename` (first present); the
/// feature columns as `feat_0..` (zenmetrics convention) → `f0..` (dial
/// convention). `q` may be Float32/Float64/Int. Column order is free.
///
/// Used by the `severity-ramp monotonicity` section of the unified
/// metric-eval report (`bake_verdict --ramp-grid`). The grid's feature
/// regime (PU21-u8 vs PU-linear) MUST match the bake being scored — the
/// caller is responsible for pointing at the regime-consistent parquet.
pub fn load_ramp_grid(path: &PathBuf) -> Result<RampGrid, String> {
    let file = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("{path:?}: parquet open: {e}"))?;
    let schema = builder.schema().clone();
    let names: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect();
    let idx = |n: &str| names.iter().position(|x| x == n);
    let img_i = idx("image_path")
        .or_else(|| idx("image_id"))
        .or_else(|| idx("ref_basename"))
        .ok_or_else(|| {
            format!("{path:?}: missing image column (image_path/image_id/ref_basename)")
        })?;
    let q_i = idx("q").ok_or_else(|| format!("{path:?}: missing q column"))?;
    // Feature columns: prefer the zenmetrics `feat_N` naming, fall back to
    // the dial-grid `fN` naming.
    let feat_prefix = if idx("feat_0").is_some() {
        "feat_"
    } else {
        "f"
    };
    let feat_idx = feature_column_run_by_name(path, &names, feat_prefix)?;
    let n_features = feat_idx.len();
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

    let mut image = Vec::new();
    let mut q = Vec::new();
    let mut feature_rows: Vec<Vec<f64>> = Vec::new();
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
            .ok_or_else(|| format!("{path:?}: image column not Utf8"))?;
        let q_v = col_f64(batch.column(q_i).as_ref(), n_rows)
            .map_err(|e| format!("{path:?}: q column {e}"))?;
        let per_col: Vec<Vec<f64>> = feat_idx
            .iter()
            .map(|&pi| col_f64(batch.column(pi).as_ref(), n_rows))
            .collect::<Result<_, _>>()
            .map_err(|e| format!("{path:?}: feature column {e}"))?;
        for r in 0..n_rows {
            image.push(img_arr.value(r).to_string());
            q.push(q_v[r]);
            feature_rows.push((0..n_features).map(|c| per_col[c][r]).collect());
        }
    }
    Ok(RampGrid {
        image,
        q,
        feature_rows,
        n_features,
    })
}

/// A labeled feature grid: per row a string `label` and the feature
/// vector. Powers the corruption-gate section (label = the `entry`
/// column, e.g. `gb82_dog__aliasing__frac2__op100__corruption`).
#[derive(Debug)]
pub struct LabeledGrid {
    pub label: Vec<String>,
    pub feature_rows: Vec<Vec<f64>>,
    pub n_features: usize,
    /// The probe's own REFERENCE TRUTH, read from `ssim2_gpu` **only**, row
    /// aligned with `label`. `None` when the column is absent.
    ///
    /// Deliberately single-named: `negtail_probe_944_era2r4_foldapp2.parquet`
    /// stores its truth as `human_score_norm`, a ÷100 quantity, and silently
    /// accepting that would put G-ADDR's −50 product bar 100× off. A probe
    /// whose truth is under another name or in another unit reads `None`, and
    /// the axes that need it read NOT MEASURED.
    pub truth_ssim2: Option<Vec<f64>>,
}

/// Load a labeled feature grid (`<label_col>`, `f0..`/`feat_0..`). The
/// label column is located as `entry` → `image_path` → `image_id` →
/// `ref_basename` (first present). Column order is free. Used by the
/// corruption-gate section of the unified metric-eval report.
pub fn load_labeled_grid(path: &PathBuf) -> Result<LabeledGrid, String> {
    let file = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("{path:?}: parquet open: {e}"))?;
    let schema = builder.schema().clone();
    let names: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect();
    let idx = |n: &str| names.iter().position(|x| x == n);
    let lbl_i = idx("entry")
        .or_else(|| idx("image_path"))
        .or_else(|| idx("image_id"))
        .or_else(|| idx("ref_basename"))
        .ok_or_else(|| {
            format!("{path:?}: missing label column (entry/image_path/image_id/ref_basename)")
        })?;
    let truth_i = idx("ssim2_gpu");
    let feat_prefix = if idx("feat_0").is_some() {
        "feat_"
    } else {
        "f"
    };
    let feat_idx = feature_column_run_by_name(path, &names, feat_prefix)?;
    let n_features = feat_idx.len();
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
            other => Err(format!("unsupported feature dtype {other:?}")),
        }
    };
    let mut label = Vec::new();
    let mut feature_rows: Vec<Vec<f64>> = Vec::new();
    let mut truth: Vec<f64> = Vec::new();
    for batch_res in reader {
        let batch = batch_res.map_err(|e| format!("{path:?}: parquet read batch: {e}"))?;
        let n_rows = batch.num_rows();
        if n_rows == 0 {
            continue;
        }
        let lbl_arr = batch
            .column(lbl_i)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| format!("{path:?}: label column not Utf8"))?;
        let per_col: Vec<Vec<f64>> = feat_idx
            .iter()
            .map(|&pi| col_f64(batch.column(pi).as_ref(), n_rows))
            .collect::<Result<_, _>>()
            .map_err(|e| format!("{path:?}: feature column {e}"))?;
        // `r` indexes both `lbl_arr` and every `per_col[c]` — a range loop
        // is the natural shape (same idiom as `load_dial_grid`).
        let truth_col = match truth_i {
            Some(ti) => Some(
                col_f64(batch.column(ti).as_ref(), n_rows)
                    .map_err(|e| format!("{path:?}: ssim2_gpu column {e}"))?,
            ),
            None => None,
        };
        #[allow(clippy::needless_range_loop)]
        for r in 0..n_rows {
            label.push(lbl_arr.value(r).to_string());
            feature_rows.push((0..n_features).map(|c| per_col[c][r]).collect());
            if let Some(t) = truth_col.as_ref() {
                truth.push(t[r]);
            }
        }
    }
    let truth_ssim2 = if truth_i.is_some() && truth.len() == label.len() {
        Some(truth)
    } else {
        None
    };
    Ok(LabeledGrid {
        label,
        feature_rows,
        n_features,
        truth_ssim2,
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

    let ref_arrow_idx = arrow_fields
        .iter()
        .position(|f| f.name() == "ref_basename")
        .ok_or_else(|| format!("{path:?}: missing ref_basename column"))?;
    let pjnd_arrow_idx = arrow_fields
        .iter()
        .position(|f| f.name() == "pjnd_target")
        .ok_or_else(|| format!("{path:?}: missing pjnd_target column"))?;
    let (_prefix, f0_arrow_idx, n_features) = feature_column_run(path, arrow_fields)?;

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

/// A row-capped, multi-metric sample for per-pair scatter/diagnostic use.
///
/// Unlike [`load_parquet`] (one `target_column`, ALL rows), this reads the
/// feature block plus any of `metric_columns` that are present, stopping once
/// `max_rows` rows are collected. It exists so a per-pair panel can sample a
/// bake's `(prediction, {ssim2, butteraugli, cvvdp, ...})` scatter from a
/// multi-gigabyte metric parquet (e.g. the 2.7 GB KADIS-720
/// `kadis700k_720.parquet`) without materializing the whole file — the reader
/// projects only the requested columns and short-circuits at `max_rows`.
///
/// Absent metric columns are silently skipped (they appear in
/// [`PerPairSample::metrics`] only when present), so a caller may request a
/// superset and use whatever the file carries.
pub struct PerPairSample {
    pub feature_rows: Vec<Vec<f64>>,
    pub n_features: usize,
    /// `(column_name, values)` for each requested metric column that existed,
    /// aligned 1:1 (by row) with `feature_rows`. Order follows `metric_columns`.
    pub metrics: Vec<(String, Vec<f64>)>,
}

/// Widen any supported numeric Arrow column to `Vec<f64>` (first `n_rows`).
/// Feature and metric columns are always numeric; ints are widened silently
/// (some extractors emit count-style features as integers).
fn numeric_col_to_f64(path: &Path, col: &dyn Array, n_rows: usize) -> Result<Vec<f64>, String> {
    Ok(match col.data_type() {
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
                "{path:?}: numeric column has unsupported dtype {other:?} (need Float32/Float64/Int32/Int64/UInt32/UInt64)"
            ));
        }
    })
}

/// See [`PerPairSample`]. `metric_columns` is a superset request; only present
/// columns are returned. Reads at most `max_rows` rows (the first `max_rows` in
/// file order).
pub fn load_perpair_sample(
    path: &Path,
    metric_columns: &[&str],
    max_rows: usize,
) -> Result<PerPairSample, String> {
    let file = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("{path:?}: parquet open: {e}"))?;
    let schema = builder.schema().clone();
    let parquet_schema = builder.parquet_schema().clone();
    let arrow_fields = schema.fields();

    // Feature block: `f<i>` or `feat_<i>`, consecutive from index 0 (same scan
    // as `load_parquet`, so `n_features` matches).
    let (prefix, f0_arrow_idx, n_features) = feature_column_run(path, arrow_fields)?;
    let _ = prefix;

    // Metric columns actually present, preserving request order.
    let present_metrics: Vec<(String, usize)> = metric_columns
        .iter()
        .filter_map(|name| {
            arrow_fields
                .iter()
                .position(|f| f.name() == name)
                .map(|i| (name.to_string(), i))
        })
        .collect();

    // Projection = features ++ present metric columns.
    let mut wanted: Vec<usize> = Vec::with_capacity(n_features + present_metrics.len());
    for i in 0..n_features {
        wanted.push(f0_arrow_idx + i);
    }
    for (_, i) in &present_metrics {
        wanted.push(*i);
    }
    let mask = ProjectionMask::leaves(&parquet_schema, wanted.iter().copied());

    let reader = builder
        .with_projection(mask)
        .with_batch_size(16384)
        .build()
        .map_err(|e| format!("{path:?}: parquet build reader: {e}"))?;

    // Projection emits columns in ascending original-index order; map each
    // wanted column to its post-projection position.
    let mut sorted_wanted = wanted.clone();
    sorted_wanted.sort_unstable();
    let pos = |orig: usize| {
        sorted_wanted
            .iter()
            .position(|&p| p == orig)
            .expect("wanted idx must be in projection")
    };
    let proj_feature_indices: Vec<usize> = (0..n_features).map(|i| pos(f0_arrow_idx + i)).collect();
    let proj_metric_indices: Vec<usize> = present_metrics.iter().map(|(_, i)| pos(*i)).collect();

    let mut feature_rows: Vec<Vec<f64>> = Vec::new();
    let mut metric_vals: Vec<Vec<f64>> = vec![Vec::new(); present_metrics.len()];

    for batch_res in reader {
        if feature_rows.len() >= max_rows {
            break;
        }
        let batch = batch_res.map_err(|e| format!("{path:?}: parquet read batch: {e}"))?;
        let full_rows = batch.num_rows();
        if full_rows == 0 {
            continue;
        }
        // Cap the final batch so the total never exceeds `max_rows`.
        let n_rows = full_rows.min(max_rows - feature_rows.len());

        // Features → per-column f64, then transpose to row-major.
        let mut per_col: Vec<Vec<f64>> = Vec::with_capacity(n_features);
        for &pi in &proj_feature_indices {
            per_col.push(numeric_col_to_f64(path, batch.column(pi).as_ref(), n_rows)?);
        }
        for row_i in 0..n_rows {
            let mut row = Vec::with_capacity(n_features);
            for col in &per_col {
                row.push(col[row_i]);
            }
            feature_rows.push(row);
        }
        // Metric columns.
        for (mi, &pi) in proj_metric_indices.iter().enumerate() {
            let v = numeric_col_to_f64(path, batch.column(pi).as_ref(), n_rows)?;
            metric_vals[mi].extend(v);
        }
    }

    if feature_rows.is_empty() {
        return Err(format!("{path:?}: empty file / no rows"));
    }
    let metrics = present_metrics
        .into_iter()
        .map(|(name, _)| name)
        .zip(metric_vals)
        .collect();
    Ok(PerPairSample {
        feature_rows,
        n_features,
        metrics,
    })
}

/// Target scores + optional reference identity for ONE parquet, with **no
/// feature columns read**.
///
/// [`load_parquet`] always projects `f0..fN`, which on a 944-wide, 190k-row
/// corpus is ~1.4 GB of I/O plus a transpose. The pair sampler
/// (`crate::mlp_train::sampling`) needs neither: a drawn subset is a
/// function of row COUNTS, group weights and the RNG only, and the coverage
/// descriptors need just the target column and the reference id. This is
/// that read — same column conventions as [`load_parquet`] (target by name;
/// `ref_basename` then `image_path` for identity; dense first-seen ref
/// numbering; `Float32`/`Float64` targets scaled by `target_scale`) so a
/// light read and a full read agree row-for-row.
pub struct ScoresAndRefs {
    /// Row count. Equals [`OwnedLoadedGroup::human_scores`]`.len()`, which
    /// is the `n` the sampler draws modulo.
    pub n_rows: usize,
    /// Target column × `target_scale`.
    pub human_scores: Vec<f64>,
    /// Dense reference ids in first-seen order; `None` when the file
    /// carries neither `ref_basename` nor `image_path`.
    pub ref_ids: Option<Vec<u32>>,
}

/// Read only the target + reference columns of a feature parquet.
pub fn load_scores_and_refs(
    path: &Path,
    target_column: &str,
    target_scale: f64,
) -> Result<ScoresAndRefs, String> {
    let file = File::open(path).map_err(|e| format!("{path:?}: open: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("{path:?}: parquet open: {e}"))?;
    let arrow_schema = builder.schema().clone();
    let parquet_schema = builder.parquet_schema().clone();
    let arrow_fields = arrow_schema.fields();

    let score_arrow_idx = arrow_fields
        .iter()
        .position(|f| f.name() == target_column)
        .ok_or_else(|| format!("{path:?}: missing target column {target_column:?}"))?;
    let ref_arrow_idx = ["ref_basename", "image_path"]
        .iter()
        .find_map(|c| arrow_fields.iter().position(|f| f.name() == c));

    let mut wanted: Vec<usize> = vec![score_arrow_idx];
    if let Some(r) = ref_arrow_idx {
        wanted.push(r);
    }
    let mask = ProjectionMask::leaves(&parquet_schema, wanted.iter().copied());
    let reader = builder
        .with_projection(mask)
        .with_batch_size(16384)
        .build()
        .map_err(|e| format!("{path:?}: parquet build reader: {e}"))?;

    // The projected reader emits columns in ASCENDING ORIGINAL-INDEX order,
    // not in the order they were requested — the same rule `load_parquet`
    // encodes just above. Assuming request order silently reads the wrong
    // column whenever the reference column precedes the target in the
    // schema, which it does on every `ext_*` view.
    let mut sorted_wanted = wanted.clone();
    sorted_wanted.sort_unstable();
    let proj_score_idx = sorted_wanted
        .iter()
        .position(|&i| i == score_arrow_idx)
        .expect("score idx must be in projection");
    let proj_ref_idx = ref_arrow_idx.map(|r| {
        sorted_wanted
            .iter()
            .position(|&p| p == r)
            .expect("ref idx must be in projection")
    });

    let mut human_scores: Vec<f64> = Vec::new();
    let mut ref_ids: Vec<u32> = Vec::new();
    let mut ref_lookup: HashMap<String, u32> = HashMap::new();

    for batch_res in reader {
        let batch = batch_res.map_err(|e| format!("{path:?}: parquet read batch: {e}"))?;
        let n_rows = batch.num_rows();
        if n_rows == 0 {
            continue;
        }
        let score_col = batch.column(proj_score_idx);
        match score_col.data_type() {
            DataType::Float64 => {
                let a = score_col
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("Float64 dispatch");
                human_scores.extend((0..n_rows).map(|i| a.value(i) * target_scale));
            }
            DataType::Float32 => {
                let a = score_col
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .expect("Float32 dispatch");
                human_scores.extend((0..n_rows).map(|i| a.value(i) as f64 * target_scale));
            }
            other => {
                return Err(format!(
                    "{path:?}: target column {target_column:?} has unsupported dtype {other:?} (need Float32/Float64)",
                ));
            }
        }
        if let Some(pi) = proj_ref_idx {
            let col = batch.column(pi);
            let as_str: Box<dyn Fn(usize) -> String> = match col.data_type() {
                DataType::Utf8 => {
                    let a = col.as_any().downcast_ref::<StringArray>().unwrap().clone();
                    Box::new(move |i| a.value(i).to_string())
                }
                DataType::LargeUtf8 => {
                    let a = col
                        .as_any()
                        .downcast_ref::<LargeStringArray>()
                        .unwrap()
                        .clone();
                    Box::new(move |i| a.value(i).to_string())
                }
                other => {
                    return Err(format!(
                        "{path:?}: ref column has unsupported dtype {other:?} (need Utf8/LargeUtf8)",
                    ));
                }
            };
            for i in 0..n_rows {
                let key = as_str(i);
                let next = ref_lookup.len() as u32;
                let id = *ref_lookup.entry(key).or_insert(next);
                ref_ids.push(id);
            }
        }
    }

    Ok(ScoresAndRefs {
        n_rows: human_scores.len(),
        human_scores,
        ref_ids: if ref_ids.is_empty() {
            None
        } else {
            Some(ref_ids)
        },
    })
}

#[cfg(test)]
mod feature_column_run_tests {
    use super::*;
    use arrow::datatypes::{DataType, Field};
    use std::path::Path;
    use std::sync::Arc;

    fn fields(names: &[&str]) -> Vec<Arc<Field>> {
        names
            .iter()
            .map(|n| Arc::new(Field::new(*n, DataType::Float64, false)))
            .collect()
    }

    /// The contiguous run is found exactly as the five inlined walks found it —
    /// same prefix, same start index, same count.
    #[test]
    fn a_contiguous_table_reads_exactly_as_before() {
        let f = fields(&["ref_basename", "human_score", "f0", "f1", "f2", "extra"]);
        assert_eq!(
            feature_column_run(Path::new("t.parquet"), &f),
            Ok(("f", 2, 3))
        );
        let g = fields(&["id", "feat_0", "feat_1"]);
        assert_eq!(
            feature_column_run(Path::new("t.parquet"), &g),
            Ok(("feat_", 1, 2))
        );
        let names: Vec<String> = ["a", "f0", "f1", "f2"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            feature_column_run_by_name(Path::new("t.parquet"), &names, "f"),
            Ok(vec![1, 2, 3])
        );
    }

    /// **The reason both helpers exist.** A DENSE-BY-ID table (`f0..f155`,
    /// `f372..`) used to load as a 156-wide table with no error, and every
    /// number computed from it would have been about a different feature
    /// space. It must REFUSE, and the message must name the gap.
    #[test]
    fn a_gapped_dense_by_id_table_is_refused_not_truncated() {
        let mut names: Vec<&str> = vec!["ref_basename", "human_score"];
        let owned: Vec<String> = (0..4)
            .map(|i| format!("f{i}"))
            .chain((8..11).map(|i| format!("f{i}")))
            .collect();
        names.extend(owned.iter().map(String::as_str));
        let f = fields(&names);
        let err = feature_column_run(Path::new("dense.parquet"), &f)
            .expect_err("a gapped table must be refused");
        assert!(err.contains("NOT the contiguous run"), "{err}");
        assert!(
            err.contains("f8..f10"),
            "the message must name the gap: {err}"
        );

        let by_name: Vec<String> = names.iter().map(|s| s.to_string()).collect();
        let err2 = feature_column_run_by_name(Path::new("dense.parquet"), &by_name, "f")
            .expect_err("the by-name sibling must refuse too");
        assert!(err2.contains("NOT the contiguous run"), "{err2}");

        // NEGATIVE CONTROL: without the columns past the gap, the SAME table
        // loads as 4-wide — which is what made the truncation invisible.
        let tight = fields(&["ref_basename", "human_score", "f0", "f1", "f2", "f3"]);
        assert_eq!(
            feature_column_run(Path::new("tight.parquet"), &tight),
            Ok(("f", 2, 4))
        );
    }

    /// A table with no feature columns at all is still the old error, not the
    /// new one — "absent" and "gapped" are different findings.
    #[test]
    fn no_feature_columns_is_the_old_error() {
        let f = fields(&["ref_basename", "human_score"]);
        let err = feature_column_run(Path::new("t.parquet"), &f).expect_err("no f0");
        assert!(err.contains("missing f0 / feat_0"), "{err}");
    }
}
