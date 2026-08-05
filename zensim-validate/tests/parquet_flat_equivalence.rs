//! `load_parquet_flat` must produce BIT-identical values to `load_parquet`
//! — `flat.features_flat[i * nf + d] == rows.feature_rows[i][d]` for every
//! element, plus identical scores / ref ids / feature count. The two shapes
//! share one scan (`load_parquet_impl`), so this gate is what lets the
//! trainer adopt the flat path while 10+ other consumer binaries keep the
//! per-row shape: any divergence in the emission walk fails here, not in a
//! full-recipe bake diff.
//!
//! Two layers:
//!
//! * `flat_matches_rows_on_generated_parquet` — always runs. Writes a
//!   synthetic parquet (via the same arrow/parquet crates the loader reads
//!   with) into `CARGO_TARGET_TMPDIR`, shaped to cross every boundary the
//!   emission loop has: multiple row groups (so the reader yields several
//!   batches), batch lengths that are NOT multiples of the 1024-row
//!   transpose block (tail blocks), all six accepted feature dtypes
//!   (Float64/Float32/Int64/Int32/UInt64/UInt32 widening), non-finite
//!   values (NaN / ±Inf — `load_parquet` tolerates them; only the
//!   streaming loader rejects), negatives, denormals, a target scale ≠ 1,
//!   and a `ref_basename` column (dense first-seen id numbering).
//!
//! * `flat_matches_rows_on_real_fixture` — `#[ignore]`d, run explicitly
//!   (fixtures live on /mnt/v which CI boxes lack; same convention as
//!   `parquet_load_equivalence.rs`):
//!
//!       cargo test -p zensim-validate --test parquet_flat_equivalence -- \
//!           --ignored --nocapture
//!
//!   Checks the TID 372-col fixture, then any extra real parquet named by
//!   `ZENSIM_FLAT_EQ_EXTRA=path[:target_column]` (target defaults to
//!   `human_score`) — used to run the gate over a canonical 944-wide
//!   training leg.

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, StringArray, UInt32Array,
    UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;

use zensim_validate::parquet_loader::{load_parquet, load_parquet_flat};

/// Load `path` through BOTH shapes and assert bit-identity of everything.
fn assert_flat_matches_rows(path: &PathBuf, target_column: &str, target_scale: f64) {
    let rows = load_parquet(path, "rows_arm", target_column, target_scale)
        .unwrap_or_else(|e| panic!("load_parquet failed on {path:?}: {e}"));
    let flat = load_parquet_flat(path, "flat_arm", target_column, target_scale)
        .unwrap_or_else(|e| panic!("load_parquet_flat failed on {path:?}: {e}"));

    assert_eq!(flat.n_features, rows.n_features, "n_features must match");
    let nf = rows.n_features;
    let n_rows = rows.human_scores.len();
    assert_eq!(
        flat.human_scores.len(),
        n_rows,
        "row count must match ({path:?})"
    );
    assert_eq!(
        flat.features_flat.len(),
        n_rows * nf,
        "flat buffer must be exactly n_rows × n_features"
    );
    // The flat path pre-reserves from the parquet footer's row count: the
    // buffer must have been allocated ONCE (capacity == final length), or
    // the memory rationale for the flat shape is silently defeated.
    assert_eq!(
        flat.features_flat.capacity(),
        n_rows * nf,
        "flat buffer must be pre-reserved exactly (footer num_rows × n_features)"
    );

    // Bit-level equality (to_bits: NaN payloads included) — f64 == would
    // pass -0.0 vs 0.0 and fail NaN vs NaN; bits are the honest contract.
    for i in 0..n_rows {
        assert_eq!(
            flat.human_scores[i].to_bits(),
            rows.human_scores[i].to_bits(),
            "human_scores[{i}] diverged ({path:?})"
        );
        let row = &rows.feature_rows[i];
        assert_eq!(row.len(), nf, "rows arm row {i} width");
        for (d, rv) in row.iter().enumerate() {
            assert_eq!(
                flat.features_flat[i * nf + d].to_bits(),
                rv.to_bits(),
                "feature [{i}][{d}] diverged: flat={:e} rows={:e} ({path:?})",
                flat.features_flat[i * nf + d],
                rv,
            );
        }
    }
    assert_eq!(flat.ref_ids, rows.ref_ids, "ref_ids must match ({path:?})");
    // metric_sigmas: both arms ran under the same ZENSIM_SIGMA_MSE state in
    // this process, so the Option shape + values must agree.
    match (&flat.metric_sigmas, &rows.metric_sigmas) {
        (None, None) => {}
        (Some(a), Some(b)) => {
            assert_eq!(a.len(), b.len(), "sigma length");
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "metric_sigmas[{i}] diverged");
            }
        }
        (a, b) => panic!(
            "metric_sigmas presence diverged: flat={:?} rows={:?}",
            a.is_some(),
            b.is_some()
        ),
    }
    eprintln!(
        "flat==rows OK: {path:?} — {n_rows} rows × {nf} features, {} ref ids, bit-identical",
        rows.ref_ids.as_ref().map(|r| r.len()).unwrap_or(0)
    );
}

/// Deterministic LCG in [0, 1) — same generator family as the adam tests.
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as u32) as f64 / u32::MAX as f64
}

#[test]
fn flat_matches_rows_on_generated_parquet() {
    let n_rows = 20_000usize;
    let mut st = 0xF1A7u64;

    // f0: Float64 with hostile values sprinkled in (NaN, ±Inf, -0.0,
    // denormal) — load_parquet must carry them through both shapes.
    let f0: Vec<f64> = (0..n_rows)
        .map(|i| match i % 997 {
            0 => f64::NAN,
            1 => f64::INFINITY,
            2 => f64::NEG_INFINITY,
            3 => -0.0,
            4 => 4.9e-324, // smallest positive denormal
            _ => lcg(&mut st) * 2.0 - 1.0,
        })
        .collect();
    let f1: Vec<f32> = (0..n_rows).map(|_| (lcg(&mut st) * 100.0) as f32).collect();
    let f2: Vec<i64> = (0..n_rows)
        .map(|_| (lcg(&mut st) * 1e6) as i64 - 500_000)
        .collect();
    let f3: Vec<i32> = (0..n_rows).map(|_| (lcg(&mut st) * 255.0) as i32).collect();
    let f4: Vec<u64> = (0..n_rows).map(|_| (lcg(&mut st) * 1e9) as u64).collect();
    let f5: Vec<u32> = (0..n_rows).map(|_| (lcg(&mut st) * 4096.0) as u32).collect();
    let f6: Vec<f64> = (0..n_rows).map(|_| lcg(&mut st) * 1e-6).collect();
    let target: Vec<f64> = (0..n_rows).map(|_| lcg(&mut st) * 100.0).collect();
    let refs: Vec<String> = (0..n_rows).map(|i| format!("ref_{:03}", i % 37)).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("ref_basename", DataType::Utf8, false),
        Field::new("target_score", DataType::Float64, false),
        Field::new("f0", DataType::Float64, false),
        Field::new("f1", DataType::Float32, false),
        Field::new("f2", DataType::Int64, false),
        Field::new("f3", DataType::Int32, false),
        Field::new("f4", DataType::UInt64, false),
        Field::new("f5", DataType::UInt32, false),
        Field::new("f6", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(refs)) as ArrayRef,
            Arc::new(Float64Array::from(target)),
            Arc::new(Float64Array::from(f0)),
            Arc::new(Float32Array::from(f1)),
            Arc::new(Int64Array::from(f2)),
            Arc::new(Int32Array::from(f3)),
            Arc::new(UInt64Array::from(f4)),
            Arc::new(UInt32Array::from(f5)),
            Arc::new(Float64Array::from(f6)),
        ],
    )
    .expect("build record batch");

    // 7000-row row groups: the reader emits ≤7000-row batches, none a
    // multiple of the 1024-row transpose block (7000 % 1024 = 856, final
    // group 6000 % 1024 = 880) — every batch ends in a tail block.
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&dir).expect("create target tmpdir");
    let path = dir.join("flat_equivalence_fixture.parquet");
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(7000))
        .build();
    let file = std::fs::File::create(&path).expect("create fixture parquet");
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("open writer");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close writer");

    // Scale ≠ 1 so the target multiply is exercised identically in both arms.
    assert_flat_matches_rows(&path, "target_score", 0.01);
    let _ = std::fs::remove_file(&path);
}

const TID_PARQUET: &str =
    "/mnt/v/zen/zensim-training/2026-05-16/v2/tid_features_iwssim_log_372col.parquet";

/// Real-fixture arm — run explicitly with `-- --ignored` on a box that has
/// the /mnt/v fixtures (documented in the module header).
#[test]
#[ignore]
fn flat_matches_rows_on_real_fixture() {
    assert_flat_matches_rows(&PathBuf::from(TID_PARQUET), "iwssim_log_norm", 1.0);
    // Optional second corpus: ZENSIM_FLAT_EQ_EXTRA=path[:target_column]
    // (an absent var only skips the EXTRA file; the TID assertion above has
    // already run unconditionally — the skip decision is the caller's, per
    // the no-graceful-skip rule).
    if let Ok(spec) = std::env::var("ZENSIM_FLAT_EQ_EXTRA") {
        let (p, tc) = match spec.rsplit_once(':') {
            Some((p, tc)) if !tc.contains('/') => (p.to_string(), tc.to_string()),
            _ => (spec, "human_score".to_string()),
        };
        assert_flat_matches_rows(&PathBuf::from(p), &tc, 1.0);
    }
}
