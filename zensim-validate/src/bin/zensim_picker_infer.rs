//! Run a per-codec picker bake on (features_parquet, T) and emit
//! `ref_basename | T | predicted_q | q_rounded` rows.
//!
//! Usage:
//!   zensim_picker_infer \
//!       --bake zensim/weights/picker_zenjpeg_2026-05-19.bin \
//!       --features /mnt/v/zen/picker-training/2026-05-19/sources_zenanalyze_features.parquet \
//!       --t-values 30,50,63,70,80,90 \
//!       --q-grid 5,10,15,...,95 \
//!       [--ref-basenames file] \
//!       --out /tmp/picker_zenjpeg_predictions.tsv

use anyhow::{Context, Result, anyhow};
use arrow::array::{Array, Float32Array, Float64Array, StringArray};
use arrow::datatypes::DataType;
use clap::Parser;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use zenpredict::{Model, Predictor};

#[derive(Parser, Debug)]
struct Args {
    /// Picker bake (ZNPR v3).
    #[arg(long)]
    bake: PathBuf,

    /// Source features parquet
    /// (ref_basename + feat_0..feat_NNN columns).
    #[arg(long)]
    features: PathBuf,

    /// Comma-separated T values (zensim_tuner targets).
    #[arg(long, default_value = "30,50,63,70,80,90")]
    t_values: String,

    /// Comma-separated q grid for rounding (must match picker training).
    #[arg(long, default_value = "5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95")]
    q_grid: String,

    /// Optional file with newline-separated ref_basenames to filter.
    #[arg(long)]
    ref_basenames: Option<PathBuf>,

    /// Output TSV.
    #[arg(long)]
    out: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let bake_bytes = std::fs::read(&args.bake)
        .with_context(|| format!("read bake {:?}", args.bake))?;
    let model = Model::from_bytes(&bake_bytes)
        .map_err(|e| anyhow!("load bake: {e:?}"))?;
    let mut predictor = Predictor::new(&model);
    let n_in = predictor.n_inputs();
    eprintln!("loaded bake, n_inputs={}", n_in);

    let t_values: Vec<f32> = args
        .t_values
        .split(',')
        .map(|s| s.trim().parse::<f32>().context("parse T"))
        .collect::<Result<_>>()?;
    let q_grid: Vec<f32> = args
        .q_grid
        .split(',')
        .map(|s| s.trim().parse::<f32>().context("parse q"))
        .collect::<Result<_>>()?;

    let filter: Option<HashSet<String>> = if let Some(p) = &args.ref_basenames {
        let s = std::fs::read_to_string(p).with_context(|| format!("read {p:?}"))?;
        let v: HashSet<String> = s.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect();
        eprintln!("filtering to {} ref_basenames", v.len());
        Some(v)
    } else {
        None
    };

    // Load features parquet.
    let file = File::open(&args.features)
        .with_context(|| format!("open {:?}", args.features))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let parquet_schema = builder.parquet_schema().clone();
    let arrow_fields = schema.fields();

    let ref_idx = arrow_fields
        .iter()
        .position(|f| f.name() == "ref_basename")
        .ok_or_else(|| anyhow!("missing ref_basename"))?;
    let mut feat_indices: Vec<(usize, String)> = arrow_fields
        .iter()
        .enumerate()
        .filter_map(|(i, f)| {
            let n = f.name();
            if n.starts_with("feat_") {
                Some((i, n.clone()))
            } else {
                None
            }
        })
        .collect();
    feat_indices.sort_by_key(|(i, _)| *i);
    let n_features = feat_indices.len();
    if n_features + 1 != n_in {
        return Err(anyhow!(
            "n_features ({}) + 1 ≠ bake n_inputs ({})",
            n_features, n_in
        ));
    }

    let mut wanted: Vec<usize> = vec![ref_idx];
    for (i, _) in &feat_indices {
        wanted.push(*i);
    }
    let mask = ProjectionMask::leaves(&parquet_schema, wanted.iter().copied());
    let mut sorted_wanted = wanted.clone();
    sorted_wanted.sort_unstable();
    let pos = |orig: usize| -> usize { sorted_wanted.iter().position(|&i| i == orig).unwrap() };
    let proj_ref = pos(ref_idx);
    let proj_feats: Vec<usize> = feat_indices.iter().map(|(i, _)| pos(*i)).collect();

    let reader = builder
        .with_projection(mask)
        .with_batch_size(16384)
        .build()?;

    let mut out_file = File::create(&args.out)?;
    writeln!(out_file, "ref_basename\tT\tpredicted_q\tq_rounded")?;

    let mut total_rows = 0;

    for batch_res in reader {
        let batch = batch_res?;
        let n_rows = batch.num_rows();
        if n_rows == 0 {
            continue;
        }
        let ref_col = batch
            .column(proj_ref)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow!("ref_basename not String"))?;

        // Materialize feature columns as Vec<Vec<f32>> per batch row.
        let mut feat_cols: Vec<Vec<f32>> = Vec::with_capacity(n_features);
        for &pi in &proj_feats {
            let col = batch.column(pi);
            let v: Vec<f32> = match col.data_type() {
                DataType::Float32 => {
                    let a = col.as_any().downcast_ref::<Float32Array>().unwrap();
                    (0..n_rows).map(|i| a.value(i)).collect()
                }
                DataType::Float64 => {
                    let a = col.as_any().downcast_ref::<Float64Array>().unwrap();
                    (0..n_rows).map(|i| a.value(i) as f32).collect()
                }
                other => return Err(anyhow!("feat dtype {other:?}")),
            };
            feat_cols.push(v);
        }

        for row in 0..n_rows {
            let basename = ref_col.value(row);
            if let Some(f) = &filter {
                if !f.contains(basename) {
                    continue;
                }
            }
            // Build feature vector.
            let mut feats = Vec::with_capacity(n_in);
            for col in &feat_cols {
                feats.push(col[row]);
            }
            for &t in &t_values {
                let mut full = feats.clone();
                full.push(t);
                let pred = predictor.predict(&full).map_err(|e| anyhow!("predict: {e:?}"))?;
                let q_pred = pred[0];
                // Round to nearest q in grid.
                let mut best_q = q_grid[0];
                let mut best_d = (q_grid[0] - q_pred).abs();
                for &q in &q_grid {
                    let d = (q - q_pred).abs();
                    if d < best_d {
                        best_d = d;
                        best_q = q;
                    }
                }
                writeln!(out_file, "{}\t{:.1}\t{:.3}\t{}", basename, t, q_pred, best_q as i32)?;
                total_rows += 1;
            }
        }
    }
    eprintln!("wrote {} predictions to {:?}", total_rows, args.out);
    Ok(())
}
