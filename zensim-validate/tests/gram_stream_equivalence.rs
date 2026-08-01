//! E-LIN linear-924 fit-chain guards (`benchmarks/linear924_phase1_2026-08-01.md`):
//!
//! 1. `stream_parquet_rows` visits exactly the rows `load_parquet` loads
//!    (bit-identical features + scaled target, same order).
//! 2. The `gram` accumulation strategy (upper-triangle, skip `x_i == 0`,
//!    row-sequential) is BIT-EXACT against a naive full triple-loop direct
//!    computation.
//! 3. `standardize_gram_multi` over two per-corpus grams equals the
//!    single-group standardization of the pooled weighted moments
//!    (accumulated in the same order), bit for bit.
//! 4. The streaming contract fails loudly on NULLs and non-finite values.
//!
//! Fixtures are written into the cargo target dir (never /tmp) as parquet
//! via the same arrow/parquet crates the loader uses.

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{Float32Array, Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;

use zensim_validate::gram_lasso::{GramGroup, standardize_gram, standardize_gram_multi};
use zensim_validate::parquet_loader::{load_parquet, stream_parquet_rows};

const N_FEAT: usize = 7;

fn scratch_dir() -> PathBuf {
    let base = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".into());
    let dir = PathBuf::from(base).join("gram-stream-fixtures");
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    dir
}

/// Deterministic pseudo-random doubles (xorshift; no rand dep).
struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        // ~[-2, 2] with full mantissa churn
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 4.0 - 2.0
    }
}

/// Rows for a synthetic corpus: some exact zeros (exercises the skip-zero
/// path), one all-zero column (exercises the sd<1e-9 guard downstream).
fn make_rows(n_rows: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = Rng(seed | 1);
    let mut rows = Vec::with_capacity(n_rows);
    let mut ys = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        let mut row = Vec::with_capacity(N_FEAT);
        for f in 0..N_FEAT {
            // f==3: structural-zero column (f156..f371 analog);
            // (r+f)%5==0: scattered exact zeros (skip-zero path).
            let v = if f == 3 || (r + f) % 5 == 0 {
                0.0
            } else {
                rng.next_f64()
            };
            row.push(v);
        }
        rows.push(row);
        ys.push(rng.next_f64() * 60.0 - 10.0);
    }
    (rows, ys)
}

fn write_fixture(path: &PathBuf, rows: &[Vec<f64>], ys: &[f64], f32_feats: bool) {
    let mut fields: Vec<Field> = vec![
        Field::new("ref_basename", DataType::Utf8, false),
        Field::new("human_score", DataType::Float64, false),
    ];
    for i in 0..N_FEAT {
        let dt = if f32_feats {
            DataType::Float32
        } else {
            DataType::Float64
        };
        fields.push(Field::new(format!("f{i}"), dt, false));
    }
    let schema = Arc::new(Schema::new(fields));
    let mut cols: Vec<Arc<dyn arrow::array::Array>> = Vec::new();
    let refs: Vec<String> = (0..rows.len()).map(|r| format!("ref{}", r / 4)).collect();
    cols.push(Arc::new(StringArray::from(refs)));
    cols.push(Arc::new(Float64Array::from(ys.to_vec())));
    for i in 0..N_FEAT {
        if f32_feats {
            cols.push(Arc::new(Float32Array::from(
                rows.iter().map(|r| r[i] as f32).collect::<Vec<f32>>(),
            )));
        } else {
            cols.push(Arc::new(Float64Array::from(
                rows.iter().map(|r| r[i]).collect::<Vec<f64>>(),
            )));
        }
    }
    let batch = RecordBatch::try_new(schema.clone(), cols).expect("batch");
    let file = std::fs::File::create(path).expect("create fixture");
    let mut w = ArrowWriter::try_new(file, schema, None).expect("writer");
    w.write(&batch).expect("write");
    w.close().expect("close");
}

/// Naive full triple-loop raw moments (the reference the gram subcommand's
/// upper-triangle + skip-zero accumulation must match bitwise).
fn direct_moments(
    rows: &[Vec<f64>],
    ys: &[f64],
    scale: f64,
    clip: Option<f64>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64, f64) {
    let n = N_FEAT;
    let mut s_mat = vec![0.0f64; n * n];
    let mut s_vec = vec![0.0f64; n];
    let mut q = vec![0.0f64; n];
    let mut y1 = 0.0f64;
    for (row, y0) in rows.iter().zip(ys) {
        let mut y = y0 * scale;
        if let Some(c) = clip
            && y < c
        {
            y = c;
        }
        for i in 0..n {
            for j in 0..n {
                s_mat[i * n + j] += row[i] * row[j];
            }
            s_vec[i] += row[i];
            q[i] += row[i] * y;
        }
        y1 += y;
    }
    (s_mat, s_vec, q, y1, rows.len() as f64)
}

/// The gram subcommand's accumulation strategy, reproduced over the
/// streamed batches: upper triangle with the `x_i == 0` skip, then mirror.
fn streamed_moments(
    path: &PathBuf,
    scale: f64,
    clip: Option<f64>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64, f64) {
    let n = N_FEAT;
    let mut s_mat = vec![0.0f64; n * n];
    let mut s_vec = vec![0.0f64; n];
    let mut q = vec![0.0f64; n];
    let mut y1 = 0.0f64;
    let info = stream_parquet_rows(path, &["human_score"], scale, &mut |feats,
                                                                        n_rows,
                                                                        tgts|
     -> Result<
        (),
        String,
    > {
        for r in 0..n_rows {
            let x = &feats[r * n..(r + 1) * n];
            for i in 0..n {
                let xi = x[i];
                if xi != 0.0 {
                    for j in i..n {
                        s_mat[i * n + j] += xi * x[j];
                    }
                }
            }
            for (acc, v) in s_vec.iter_mut().zip(x) {
                *acc += *v;
            }
            let mut y = tgts[0][r];
            if let Some(c) = clip
                && y < c
            {
                y = c;
            }
            for (acc, v) in q.iter_mut().zip(x) {
                *acc += *v * y;
            }
            y1 += y;
        }
        Ok(())
    })
    .expect("stream");
    assert_eq!(info.n_features, N_FEAT);
    for i in 0..n {
        for j in (i + 1)..n {
            s_mat[j * n + i] = s_mat[i * n + j];
        }
    }
    (s_mat, s_vec, q, y1, info.n_rows as f64)
}

#[test]
fn stream_matches_load_parquet_bitwise() {
    let (rows, ys) = make_rows(257, 0xE11A);
    let path = scratch_dir().join("stream_eq_f64.parquet");
    write_fixture(&path, &rows, &ys, false);

    let loaded = load_parquet(&path, "fixture", "human_score", 100.0).expect("load");
    let mut streamed_feats: Vec<Vec<f64>> = Vec::new();
    let mut streamed_y: Vec<f64> = Vec::new();
    stream_parquet_rows(
        &path,
        &["human_score"],
        100.0,
        &mut |feats, n_rows, tgts| {
            for r in 0..n_rows {
                streamed_feats.push(feats[r * N_FEAT..(r + 1) * N_FEAT].to_vec());
                streamed_y.push(tgts[0][r]);
            }
            Ok(())
        },
    )
    .expect("stream");

    assert_eq!(loaded.feature_rows.len(), streamed_feats.len());
    for (a, b) in loaded.feature_rows.iter().zip(&streamed_feats) {
        for (x, y) in a.iter().zip(b) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }
    for (a, b) in loaded.human_scores.iter().zip(&streamed_y) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    std::fs::remove_file(&path).ok();
}

#[test]
fn gram_accumulation_bit_exact_vs_direct() {
    // f64 features + a scale + clip exercising the registered E-LIN policy.
    let (rows, ys) = make_rows(311, 0x924);
    let path = scratch_dir().join("gram_eq_f64.parquet");
    write_fixture(&path, &rows, &ys, false);
    let (ds, dv, dq, dy1, dn) = direct_moments(&rows, &ys, 100.0, Some(-100.0));
    let (ss, sv, sq, sy1, sn) = streamed_moments(&path, 100.0, Some(-100.0));
    assert_eq!(dn, sn);
    assert_eq!(dy1.to_bits(), sy1.to_bits(), "Y1");
    for k in 0..N_FEAT * N_FEAT {
        assert_eq!(ds[k].to_bits(), ss[k].to_bits(), "S[{k}]");
    }
    for k in 0..N_FEAT {
        assert_eq!(dv[k].to_bits(), sv[k].to_bits(), "s[{k}]");
        assert_eq!(dq[k].to_bits(), sq[k].to_bits(), "q[{k}]");
    }
    std::fs::remove_file(&path).ok();

    // f32-stored features widen exactly, so the equality must ALSO hold
    // when the parquet stores Float32 (the kadis-924 layout).
    let rows32: Vec<Vec<f64>> = rows
        .iter()
        .map(|r| r.iter().map(|v| *v as f32 as f64).collect())
        .collect();
    let path32 = scratch_dir().join("gram_eq_f32.parquet");
    write_fixture(&path32, &rows32, &ys, true);
    let (ds, dv, dq, dy1, dn) = direct_moments(&rows32, &ys, 1.0, None);
    let (ss, sv, sq, sy1, sn) = streamed_moments(&path32, 1.0, None);
    assert_eq!(dn, sn);
    assert_eq!(dy1.to_bits(), sy1.to_bits());
    for k in 0..N_FEAT * N_FEAT {
        assert_eq!(ds[k].to_bits(), ss[k].to_bits(), "S32[{k}]");
    }
    for k in 0..N_FEAT {
        assert_eq!(dv[k].to_bits(), sv[k].to_bits());
        assert_eq!(dq[k].to_bits(), sq[k].to_bits());
    }
    std::fs::remove_file(&path32).ok();
}

#[test]
fn multi_gram_equals_pooled_weighted_accumulation() {
    let (rows_a, ys_a) = make_rows(120, 0xA);
    let (rows_b, ys_b) = make_rows(85, 0xB);
    let (sa, va, qa, y1a, na) = direct_moments(&rows_a, &ys_a, 100.0, Some(-100.0));
    let (sb, vb, qb, y1b, nb) = direct_moments(&rows_b, &ys_b, 1.0, Some(-100.0));
    let (wa, wb) = (1.0, 0.1);

    let multi = standardize_gram_multi(
        N_FEAT,
        &[
            GramGroup {
                weight: wa,
                s_mat: &sa,
                s_vec: &va,
                q: &qa,
                y1: y1a,
                n_rows: na,
            },
            GramGroup {
                weight: wb,
                s_mat: &sb,
                s_vec: &vb,
                q: &qb,
                y1: y1b,
                n_rows: nb,
            },
        ],
    )
    .expect("multi");

    // Pool the weighted moments in the SAME order (acc = wa*v; acc += wb*v)
    // and standardize through the single-group path with weight 1.0.
    let pool = |a: &[f64], b: &[f64]| -> Vec<f64> {
        a.iter().zip(b).map(|(x, y)| wa * x + wb * y).collect()
    };
    let sp = pool(&sa, &sb);
    let vp = pool(&va, &vb);
    let qp = pool(&qa, &qb);
    let y1p = wa * y1a + wb * y1b;
    let np = wa * na + wb * nb;
    let single = standardize_gram(N_FEAT, 1.0, &sp, &vp, &qp, y1p, np).expect("single");

    assert_eq!(multi.ybar.to_bits(), single.ybar.to_bits());
    assert_eq!(multi.w_total.to_bits(), single.w_total.to_bits());
    for k in 0..N_FEAT {
        assert_eq!(multi.mu[k].to_bits(), single.mu[k].to_bits(), "mu[{k}]");
        assert_eq!(multi.sd[k].to_bits(), single.sd[k].to_bits(), "sd[{k}]");
        assert_eq!(multi.c[k].to_bits(), single.c[k].to_bits(), "c[{k}]");
    }
    for k in 0..N_FEAT * N_FEAT {
        assert_eq!(multi.g[k].to_bits(), single.g[k].to_bits(), "g[{k}]");
    }
}

#[test]
fn stream_rejects_nonfinite_and_missing() {
    let (mut rows, ys) = make_rows(40, 0xF00D);
    rows[17][2] = f64::NAN;
    let path = scratch_dir().join("stream_nan.parquet");
    write_fixture(&path, &rows, &ys, false);
    let err = stream_parquet_rows(&path, &["human_score"], 1.0, &mut |_, _, _| Ok(()))
        .expect_err("NaN must be rejected");
    assert!(err.contains("non-finite"), "got: {err}");
    std::fs::remove_file(&path).ok();

    let (rows, ys) = make_rows(10, 0xBEEF);
    let path2 = scratch_dir().join("stream_missing.parquet");
    write_fixture(&path2, &rows, &ys, false);
    let err = stream_parquet_rows(&path2, &["not_a_column"], 1.0, &mut |_, _, _| Ok(()))
        .expect_err("missing target must error");
    assert!(err.contains("missing target column"), "got: {err}");
    std::fs::remove_file(&path2).ok();
}
