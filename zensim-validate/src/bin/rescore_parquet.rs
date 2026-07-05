//! Rescore a sweep parquet's per-cell feature vectors under a chosen
//! [`zensim::ZensimProfile`], replacing (or adding) a score column.
//!
//! Purpose: codec-picker retraining targets a *specific* zensim profile.
//! A sweep parquet carries `feat_0..feat_<N-1>` (the 372-feature vector)
//! per (image, codec, q, knob) cell plus its original `score_zensim` from
//! whatever profile scored the sweep. To retrain a picker against, e.g.,
//! `ZensimProfile::A` (v47), we rescore every cell from its stored feature
//! vector — `score_features_with_profile` is bit-exact with a full
//! `compute()` and needs **no re-encode** (for the 372-input A bake the
//! width/height args are inert; they only matter for size-axis bakes).
//!
//! Streams RecordBatches so the 370 MB `encoded_bytes` column passes
//! through without loading the whole file into memory.
//!
//! ```text
//! rescore_parquet --input sweep.parquet --output sweep_a.parquet \
//!     [--profile a] [--score-col score_zensim] [--feat-prefix feat_]
//! ```

use std::fs::File;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float32Array, Float64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use zensim::ZensimProfile;

fn arg(flag: &str, default: Option<&str>) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == flag)
        .and_then(|i| a.get(i + 1).cloned())
        .or_else(|| default.map(str::to_string))
}

fn parse_profile(s: &str) -> ZensimProfile {
    match s.to_ascii_lowercase().as_str() {
        "a" | "zensim-a" => ZensimProfile::A,
        "b" | "zensim-b" => ZensimProfile::B,
        "bhdr" | "b-hdr" | "zensim-b-hdr" => ZensimProfile::BHdr,
        other => {
            eprintln!("unknown --profile '{other}', defaulting to A");
            ZensimProfile::A
        }
    }
}

/// Read column `idx` of `batch` as `f64` regardless of f32/f64 storage.
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

fn main() {
    let input = arg("--input", None).expect("--input <parquet> required");
    let output = arg("--output", None).expect("--output <parquet> required");
    let profile = parse_profile(&arg("--profile", Some("a")).unwrap());
    let score_col = arg("--score-col", Some("score_zensim")).unwrap();
    let feat_prefix = arg("--feat-prefix", Some("feat_")).unwrap();

    let file = File::open(&input).expect("open input");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("parquet reader");
    let in_schema = builder.schema().clone();

    // Locate feat_0..feat_<N-1> (consecutive) by name.
    let names: Vec<&str> = in_schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    let mut feat_idx = Vec::new();
    let mut n = 0usize;
    loop {
        let want = format!("{feat_prefix}{n}");
        match names.iter().position(|&x| x == want) {
            Some(p) => {
                feat_idx.push(p);
                n += 1;
            }
            None => break,
        }
    }
    assert!(
        !feat_idx.is_empty(),
        "no {feat_prefix}0.. feature columns found"
    );
    eprintln!(
        "[rescore] {} features ({feat_prefix}0..{feat_prefix}{}); scoring under {} -> column '{score_col}'",
        feat_idx.len(),
        feat_idx.len() - 1,
        profile.name()
    );

    // Output schema: same as input, but ensure `score_col` exists as Float64.
    let score_pos = names.iter().position(|&x| x == score_col);
    let out_fields: Vec<Field> = {
        let mut v: Vec<Field> = in_schema
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        match score_pos {
            Some(p) => v[p] = Field::new(&score_col, DataType::Float64, false),
            None => v.push(Field::new(&score_col, DataType::Float64, false)),
        }
        v
    };
    let out_schema = Arc::new(Schema::new(out_fields));

    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
        .build();
    let out_file = File::create(&output).expect("create output");
    let mut writer =
        ArrowWriter::try_new(out_file, out_schema.clone(), Some(props)).expect("writer");

    let reader = builder.build().expect("batch reader");
    let mut rows = 0usize;
    for batch in reader {
        let batch = batch.expect("read batch");
        let nrows = batch.num_rows();
        // Gather per-row feature vectors (column-major -> row-major).
        let cols: Vec<Vec<f64>> = feat_idx.iter().map(|&i| col_f64(&batch, i)).collect();
        let mut scores = Vec::with_capacity(nrows);
        let mut feats = vec![0.0f64; feat_idx.len()];
        for r in 0..nrows {
            for (k, c) in cols.iter().enumerate() {
                feats[k] = c[r];
            }
            // width/height inert for the 372-input A bake; pass a sane default.
            let s = zensim::score_features_with_profile(profile, &feats, 1024, 1024)
                .unwrap_or(f64::NAN);
            scores.push(s);
        }
        let score_arr: ArrayRef = Arc::new(Float64Array::from(scores));

        // Rebuild the batch with the replaced/added score column.
        let mut out_cols: Vec<ArrayRef> = batch.columns().to_vec();
        match score_pos {
            Some(p) => out_cols[p] = score_arr,
            None => out_cols.push(score_arr),
        }
        let out_batch = RecordBatch::try_new(out_schema.clone(), out_cols).expect("rebuild batch");
        writer.write(&out_batch).expect("write batch");
        rows += nrows;
        if rows % 50_000 < nrows {
            eprintln!("[rescore] {rows} rows…");
        }
    }
    writer.close().expect("close writer");
    eprintln!("[rescore] done: {rows} rows -> {output}");
}
