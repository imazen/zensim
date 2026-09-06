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

#[allow(deprecated)] // `a` stays selectable for rescoring against the deprecated v47/A bake
fn parse_profile(s: &str) -> ZensimProfile {
    match s.to_ascii_lowercase().as_str() {
        "a" | "zensim-a" => ZensimProfile::A,
        "b" | "zensim-b" => ZensimProfile::B,
        "bhdr" | "b-hdr" | "zensim-b-hdr" => ZensimProfile::BHdr,
        other => {
            eprintln!("unknown --profile '{other}', defaulting to the codec-target profile (B)");
            ZensimProfile::codec_target()
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

/// **`--densify` — the table half of the dense feature-id contract.**
///
/// Rewrites a feature parquet so it stores exactly the ids it POPULATES:
/// absent-id columns dropped, every kept column bit-identical, row order
/// identical, and a sidecar manifest recording the source sha256, the output
/// sha256, the kept id list and the dropped one.
///
/// **The populated set comes from the DECLARATION, never from a value scan,
/// and that is the whole design.** A scan can only say "these columns are
/// zero on THIS corpus" — which is prune class 3, the one
/// `zensim_validate::prune` refuses to act on. MEASURED on the postC 372 eval
/// root, where every id is genuinely populated: a scan-only converter would
/// have dropped `f25` from `aic3` (600 rows), `f12` from `konjnd` (1,008) and
/// EIGHT columns from `ext_sdr25` (50 rows) — small-corpus accidents, not
/// structural absences, and dropping them would make those tables unreadable
/// by every bake that reads those ids.
///
/// So the scan is a GATE, not the source: every column the declaration says is
/// absent must be all-zero across every row, or the conversion REFUSES. That
/// is the direction that cannot be wrong — the declaration decides what to
/// drop, and the data has to agree.
///
/// ```text
/// rescore_parquet --densify --input wide.parquet --output dense.parquet \
///     --keep-ids 0-155,372-719   [--feat-prefix f] [--scan-only]
/// ```
///
/// `--scan-only` writes nothing and reports the census (every column's
/// all-zero verdict), which is how the measurements above were taken.
fn densify_main(input: &str, output: Option<&str>) {
    let feat_prefix = arg("--feat-prefix", Some("f")).unwrap();
    let scan_only = std::env::args().any(|a| a == "--scan-only");
    let keep_spec = arg("--keep-ids", None);
    if !scan_only && keep_spec.is_none() {
        eprintln!(
            "[densify] --keep-ids <slot-set> is REQUIRED unless --scan-only: the populated set \
             is a DECLARATION, and deriving it from a value scan would drop columns that are \
             merely zero on this corpus (measured: aic3 f25, konjnd f12, ext_sdr25 x8)."
        );
        std::process::exit(2);
    }
    let keep = keep_spec.as_deref().map(|spec| {
        zensim::feature_set_id::SlotSet::parse(spec).unwrap_or_else(|| {
            eprintln!("[densify] --keep-ids {spec:?} is not a slot set (e.g. \"0-155,372-719\")");
            std::process::exit(2);
        })
    });

    let file = File::open(input).expect("open input");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("parquet reader");
    let in_schema = builder.schema().clone();
    let names: Vec<String> = in_schema
        .fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect();

    // Feature columns BY ID, located by name — never by a contiguous walk, so
    // an already-dense table reads correctly here.
    let mut feat: Vec<(usize, usize)> = Vec::new(); // (id, column index)
    for (i, n) in names.iter().enumerate() {
        if let Some(rest) = n.strip_prefix(feat_prefix.as_str())
            && let Ok(id) = rest.parse::<usize>()
        {
            feat.push((id, i));
        }
    }
    feat.sort_unstable();
    assert!(
        !feat.is_empty(),
        "no {feat_prefix}<id> feature columns found"
    );

    // FULL-COLUMN SCAN — every row of every feature column, never a sample.
    let mut all_zero: Vec<bool> = vec![true; feat.len()];
    let mut n_rows = 0usize;
    let reader = builder.build().expect("build reader");
    for batch in reader {
        let batch = batch.expect("read batch");
        n_rows += batch.num_rows();
        for (k, &(_, ci)) in feat.iter().enumerate() {
            if !all_zero[k] {
                continue;
            }
            if col_f64(&batch, ci).iter().any(|v| *v != 0.0) {
                all_zero[k] = false;
            }
        }
    }

    let zero_ids: Vec<usize> = feat
        .iter()
        .zip(&all_zero)
        .filter(|(_, z)| **z)
        .map(|((id, _), _)| *id)
        .collect();
    eprintln!(
        "[densify] {input}: {} rows, {} feature columns (f{}..f{}), {} all-zero: {:?}",
        n_rows,
        feat.len(),
        feat[0].0,
        feat[feat.len() - 1].0,
        zero_ids.len(),
        zero_ids
    );

    let Some(keep) = keep else {
        eprintln!("[densify] --scan-only: nothing written");
        return;
    };

    // THE GATE: every id the declaration drops must be all-zero here. The
    // declaration decides; the data must agree.
    let dropped: Vec<usize> = feat
        .iter()
        .map(|(id, _)| *id)
        .filter(|id| !keep.contains(*id))
        .collect();
    let live_dropped: Vec<usize> = dropped
        .iter()
        .copied()
        .filter(|id| {
            feat.iter()
                .position(|(i, _)| i == id)
                .is_some_and(|k| !all_zero[k])
        })
        .collect();
    if !live_dropped.is_empty() {
        eprintln!(
            "[densify] REFUSING: the declaration drops {} id(s) that carry NONZERO values in \
             this table: {live_dropped:?}. Either the declaration is wrong or the table is not \
             what it claims — both are findings, neither is a column to delete.",
            live_dropped.len()
        );
        std::process::exit(2);
    }
    // The mirror check: an id the declaration KEEPS that this table does not
    // have at all cannot be written, and silently emitting a zero column for
    // it would manufacture exactly the fill this contract exists to remove.
    let missing: Vec<usize> = keep
        .iter_slots()
        .filter(|id| !feat.iter().any(|(i, _)| i == id))
        .collect();
    if !missing.is_empty() {
        eprintln!(
            "[densify] REFUSING: the declaration keeps {} id(s) this table has no column for: \
             {missing:?}",
            missing.len()
        );
        std::process::exit(2);
    }

    let output = output.expect("--output required");
    let keep_cols: Vec<(usize, usize)> = feat
        .iter()
        .copied()
        .filter(|(id, _)| keep.contains(*id))
        .collect();
    let passthrough: Vec<usize> = (0..names.len())
        .filter(|i| !feat.iter().any(|(_, ci)| ci == i))
        .collect();
    let out_fields: Vec<Field> = passthrough
        .iter()
        .map(|&i| in_schema.field(i).clone())
        .chain(keep_cols.iter().map(|&(_, ci)| in_schema.field(ci).clone()))
        .collect();
    let out_schema = Arc::new(Schema::new(out_fields));
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
        .build();
    let out_file = File::create(output).expect("create output");
    let mut writer =
        ArrowWriter::try_new(out_file, out_schema.clone(), Some(props)).expect("arrow writer");

    // Second pass. Columns are moved by REFERENCE — the arrays are not
    // rebuilt, so a kept column's bytes cannot change.
    let file = File::open(input).expect("re-open input");
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .expect("parquet reader")
        .build()
        .expect("build reader");
    let mut written = 0usize;
    for batch in reader {
        let batch = batch.expect("read batch");
        let cols: Vec<ArrayRef> = passthrough
            .iter()
            .map(|&i| batch.column(i).clone())
            .chain(keep_cols.iter().map(|&(_, ci)| batch.column(ci).clone()))
            .collect();
        written += batch.num_rows();
        writer
            .write(&RecordBatch::try_new(out_schema.clone(), cols).expect("out batch"))
            .expect("write batch");
    }
    writer.close().expect("close writer");
    assert_eq!(written, n_rows, "row count changed");

    let manifest = format!(
        "{{\n  \"tool\": \"rescore_parquet --densify\",\n  \"source\": {:?},\n  \"source_sha256\": {:?},\n  \"output\": {:?},\n  \"output_sha256\": {:?},\n  \"rows\": {},\n  \"kept_ids\": {:?},\n  \"n_kept\": {},\n  \"dropped_ids\": {:?},\n  \"n_dropped\": {},\n  \"gate\": \"every dropped id verified all-zero across every row (full-column scan)\"\n}}\n",
        input,
        sha256_file(input),
        output,
        sha256_file(output),
        n_rows,
        keep_cols.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        keep_cols.len(),
        dropped,
        dropped.len(),
    );
    let mpath = format!("{output}.densify.json");
    std::fs::write(&mpath, manifest).expect("write manifest");
    eprintln!(
        "[densify] wrote {output} ({} of {} feature columns kept) + {mpath}",
        keep_cols.len(),
        feat.len()
    );
}

/// sha256 of a file, hex. Used only for the densify manifest.
fn sha256_file(path: &str) -> String {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path).unwrap_or_default();
    let mut h = Sha256::new();
    h.update(&bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn main() {
    if std::env::args().any(|a| a == "--densify") {
        let input = arg("--input", None).expect("--input <parquet> required");
        let output = arg("--output", None);
        densify_main(&input, output.as_deref());
        return;
    }
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
