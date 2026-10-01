//! `load_parquet_flat_f32` (compact f32, column-projected) must hold EXACTLY
//! the values `load_parquet_flat` holds for the selected columns — widening an
//! f32 to f64 is exact, so `flat[i * nf + d] == compact[i * k + j] as f64` bit
//! for bit (NaN payloads, ±Inf, -0.0 and denormals included) — and it must
//! DECLINE, reading nothing, whenever the compact shape cannot represent the
//! table exactly. A decline is what lets the trainer fall back to the dense
//! f64 loader with behaviour unchanged.

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;

use zensim_validate::parquet_loader::{load_parquet_flat, load_parquet_flat_f32};

const N_ROWS: usize = 20_000;
const N_FEATURES: usize = 40;
/// Feature column that is written as Float64 (every other one is Float32).
const F64_COLUMN: usize = 7;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as u32) as f64 / u32::MAX as f64
}

fn write_fixture(name: &str) -> PathBuf {
    let mut st = 0x5EED_F32A_u64;
    let mut fields = vec![
        Field::new("ref_basename", DataType::Utf8, false),
        Field::new("human_score", DataType::Float64, false),
    ];
    let mut cols: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(
            (0..N_ROWS)
                .map(|i| format!("ref_{:03}", i % 37))
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            (0..N_ROWS)
                .map(|_| lcg(&mut st) * 100.0)
                .collect::<Vec<_>>(),
        )),
    ];
    for d in 0..N_FEATURES {
        if d == F64_COLUMN {
            fields.push(Field::new(format!("f{d}"), DataType::Float64, false));
            cols.push(Arc::new(Float64Array::from(
                (0..N_ROWS).map(|_| lcg(&mut st)).collect::<Vec<_>>(),
            )));
        } else {
            fields.push(Field::new(format!("f{d}"), DataType::Float32, false));
            let v: Vec<f32> = (0..N_ROWS)
                .map(|i| match (d + i) % 991 {
                    0 => f32::NAN,
                    1 => f32::INFINITY,
                    2 => f32::NEG_INFINITY,
                    3 => -0.0,
                    4 => 1.4e-45, // smallest positive denormal
                    _ => (lcg(&mut st) * 200.0 - 100.0) as f32,
                })
                .collect();
            cols.push(Arc::new(Float32Array::from(v)));
        }
    }
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), cols).expect("batch");
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&dir).expect("tmpdir");
    let path = dir.join(name);
    // Row groups of 7000: batches that end in a tail block of the 1024-row
    // transpose walk.
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(7000))
        .build();
    let mut w = ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), schema, Some(props))
        .expect("writer");
    w.write(&batch).expect("write");
    w.close().expect("close");
    path
}

#[test]
fn compact_f32_holds_exactly_the_selected_columns_of_the_flat_load() {
    let path = write_fixture("flat_f32_projection_fixture.parquet");
    let flat = load_parquet_flat(&path, "flat", "human_score", 0.5).expect("flat load");
    let subset: Vec<u32> = vec![0, 1, 5, 6, 8, 20, 21, 39];
    let c = load_parquet_flat_f32(&path, "compact", "human_score", 0.5, &subset, N_FEATURES)
        .expect("compact load")
        .expect("Float32 subset must not decline");

    assert_eq!(c.n_features, N_FEATURES);
    assert_eq!(c.kept, subset);
    assert_eq!(c.human_scores.len(), N_ROWS);
    assert_eq!(c.data.len(), N_ROWS * subset.len());
    // Pre-reserved exactly: the resident-memory point of the compact shape.
    assert_eq!(c.data.capacity(), N_ROWS * subset.len());
    for i in 0..N_ROWS {
        assert_eq!(c.human_scores[i].to_bits(), flat.human_scores[i].to_bits());
        for (j, &d) in subset.iter().enumerate() {
            let got = (c.data[i * subset.len() + j] as f64).to_bits();
            let want = flat.features_flat[i * N_FEATURES + d as usize].to_bits();
            assert_eq!(got, want, "row {i} feature {d}");
        }
    }
    assert_eq!(c.ref_ids, flat.ref_ids);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn compact_f32_caps_the_logical_width_like_the_trainer_truncate() {
    let path = write_fixture("flat_f32_projection_cap.parquet");
    let subset: Vec<u32> = vec![0, 3, 9];
    let c = load_parquet_flat_f32(&path, "compact", "human_score", 1.0, &subset, 10)
        .expect("compact load")
        .expect("ids below the cap must not decline");
    assert_eq!(
        c.n_features, 10,
        "logical width = min(file width, max_width)"
    );
    // An id at/after the cap is outside the logical table: decline.
    let none = load_parquet_flat_f32(&path, "compact", "human_score", 1.0, &[0, 10], 10)
        .expect("decline is Ok(None), not an error");
    assert!(none.is_none());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn compact_f32_declines_instead_of_approximating() {
    let path = write_fixture("flat_f32_projection_decline.parquet");
    let load = |subset: &[u32], width: usize| {
        load_parquet_flat_f32(&path, "compact", "human_score", 1.0, subset, width)
            .expect("decline is Ok(None), not an error")
    };
    // A selected column that is Float64 in the file.
    assert!(load(&[0, F64_COLUMN as u32, 9], N_FEATURES).is_none());
    // The same file with that column NOT selected is fine.
    assert!(load(&[0, 9], N_FEATURES).is_some());
    // Out of range, unsorted, duplicated, empty.
    assert!(load(&[0, N_FEATURES as u32], N_FEATURES).is_none());
    assert!(load(&[5, 3], N_FEATURES).is_none());
    assert!(load(&[3, 3], N_FEATURES).is_none());
    assert!(load(&[], N_FEATURES).is_none());
    let _ = std::fs::remove_file(&path);
}
