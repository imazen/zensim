//! Extract 372-feature zensim parquet sidecar from omni R2 multi-codec data.
//!
//! Task #192 (V11-FEATURE-EXTRACT-372): the omni sweep at
//! `s3://zentrain/omni-multi-codec-2026-05-19/` shipped 300-feat
//! zensim_features sidecars (no IW-pool block). This binary re-extracts
//! the full 372-feature vector (basic + peak + masked + IW pool) from the
//! preserved encoded variants on R2, joining back to the omni sidecar's
//! ssim2 / cvvdp / iwssim / butteraugli columns.
//!
//! Usage:
//! ```bash
//! cargo build --release --features extract-omni \
//!   --example extract_features_372col_omni -p zensim-bench
//! ./target/release/examples/extract_features_372col_omni \
//!   --omni-dir /mnt/v/zen/zensim-training/2026-05-20-r2-omni/multi-codec/omni \
//!   --encoded-dir /mnt/v/zen/zensim-training/2026-05-20-r2-omni/multi-codec/encoded \
//!   --sources-dir /mnt/v/input/zensim/sources \
//!   --out /mnt/v/zen/zensim-training/2026-05-20-v11-substrate/multi_codec_372col.parquet
//! ```
//!
//! Output schema (one row per omni cell, joinable to omni by image_path+codec+q+knob_tuple_json):
//!   image_path : string
//!   codec      : string
//!   q          : int32
//!   knob_tuple_json : string
//!   ref_basename : string (basename of image_path)
//!   score_zensim_gpu, score_ssim2_gpu, score_butteraugli_max_gpu,
//!   score_butteraugli_pnorm3_gpu, score_cvvdp_imazen_v0_0_1,
//!   score_dssim_gpu, score_iwssim_gpu : f64 (carried from omni)
//!   f0..f371 : float32

#![cfg(feature = "extract-omni")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{Array, ArrayRef, Float32Array, Float64Array, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use zensim::{ZensimConfig, compute_zensim_with_config};

#[derive(Debug, Clone)]
struct Cell {
    image_path: String,
    codec: String,
    q: i32,
    knob_tuple_json: String,
    encoded_filename: String,
    chunk_id: String,
    score_zensim_gpu: f64,
    score_ssim2_gpu: f64,
    score_butteraugli_max_gpu: f64,
    score_butteraugli_pnorm3_gpu: f64,
    score_cvvdp_imazen_v0_0_1: f64,
    score_dssim_gpu: f64,
    score_iwssim_gpu: f64,
}

#[derive(Debug, Clone)]
struct ExtractedRow {
    cell: Cell,
    ref_basename: String,
    features: Vec<f32>,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut omni_dir: Option<PathBuf> = None;
    let mut encoded_dir: Option<PathBuf> = None;
    let mut sources_dir: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut max_cells: usize = usize::MAX;
    let mut shard_size: usize = 10_000;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--omni-dir" => omni_dir = Some(args.next().unwrap().into()),
            "--encoded-dir" => encoded_dir = Some(args.next().unwrap().into()),
            "--sources-dir" => sources_dir = Some(args.next().unwrap().into()),
            "--out" => out = Some(args.next().unwrap().into()),
            "--max-cells" => max_cells = args.next().unwrap().parse().unwrap(),
            "--shard-size" => shard_size = args.next().unwrap().parse().unwrap(),
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let omni_dir = omni_dir.expect("--omni-dir REQUIRED");
    let encoded_dir = encoded_dir.expect("--encoded-dir REQUIRED");
    let sources_dir = sources_dir.expect("--sources-dir REQUIRED");
    let out = out.expect("--out REQUIRED");

    eprintln!("Loading omni sidecars from {}", omni_dir.display());
    let cells = load_omni_cells(&omni_dir, max_cells);
    let n_total = cells.len();
    eprintln!("Loaded {n_total} cells from omni sidecars");
    if n_total == 0 {
        eprintln!("no cells; exiting");
        std::process::exit(3);
    }

    let started = std::time::Instant::now();
    let progress = AtomicUsize::new(0);
    let log_every = (n_total / 100).max(1);
    let n_errors = AtomicUsize::new(0);

    // Process in shards so we can stream parquet writes for memory-bounded runs.
    let mut shard_idx: usize = 0;
    let total_shards = (n_total + shard_size - 1) / shard_size;
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).expect("create output dir");
    }
    // Open the parquet writer with the canonical schema once; append shards.
    let schema = output_schema();
    let file = std::fs::File::create(&out).expect("create output parquet");
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(15).unwrap()))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
        .expect("init parquet writer");

    for chunk in cells.chunks(shard_size) {
        shard_idx += 1;
        let extracted: Vec<ExtractedRow> = chunk
            .par_iter()
            .filter_map(|cell| {
                let p = progress.fetch_add(1, Ordering::Relaxed) + 1;
                if p.is_multiple_of(log_every) {
                    let elapsed = started.elapsed().as_secs_f64();
                    let rate = p as f64 / elapsed;
                    let eta = (n_total - p) as f64 / rate.max(1e-6);
                    eprintln!(
                        "  {p}/{n_total} ({rate:.1}/s, ETA {eta:.0}s) shard={shard_idx}/{total_shards}"
                    );
                }
                match extract_one(cell, &encoded_dir, &sources_dir) {
                    Ok(row) => Some(row),
                    Err(e) => {
                        n_errors.fetch_add(1, Ordering::Relaxed);
                        if n_errors.load(Ordering::Relaxed) < 20 {
                            eprintln!(
                                "  err: {} {} q={}: {}",
                                cell.image_path, cell.codec, cell.q, e
                            );
                        }
                        None
                    }
                }
            })
            .collect();

        if extracted.is_empty() {
            eprintln!("  shard {shard_idx}/{total_shards}: 0 successful rows, skipping write");
            continue;
        }
        let batch = build_record_batch(&schema, &extracted);
        writer.write(&batch).expect("write parquet batch");
        eprintln!("  shard {shard_idx}/{total_shards}: wrote {} rows", extracted.len());
    }

    writer.close().expect("close parquet writer");
    eprintln!(
        "Done in {:.1}s. Total {} cells, {} errors. Output: {}",
        started.elapsed().as_secs_f64(),
        n_total,
        n_errors.load(Ordering::Relaxed),
        out.display()
    );
}

fn extract_one(
    cell: &Cell,
    encoded_dir: &Path,
    sources_dir: &Path,
) -> Result<ExtractedRow, Box<dyn std::error::Error>> {
    let ref_basename = Path::new(&cell.image_path)
        .file_name()
        .ok_or("image_path has no basename")?
        .to_string_lossy()
        .to_string();
    let ref_path = sources_dir.join(&ref_basename);
    let dist_path = encoded_dir.join(&cell.chunk_id).join(&cell.encoded_filename);
    if !ref_path.exists() {
        return Err(format!("reference image missing: {}", ref_path.display()).into());
    }
    if !dist_path.exists() {
        return Err(format!("encoded image missing: {}", dist_path.display()).into());
    }
    // Sniff format before opening — `image::open` will return a confusing
    // error for AVIF / JXL (no decoder available in image 0.25 default
    // features), so we short-circuit with a clearer "unsupported codec".
    let ext = dist_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "avif" | "jxl" | "heic" | "heif" => {
            return Err(format!(
                "codec {ext} not supported by image-0.25 default features; skip"
            )
            .into());
        }
        _ => {}
    }
    let src_img = image::open(&ref_path)
        .map_err(|e| format!("decode ref {}: {}", ref_path.display(), e))?
        .to_rgb8();
    let dst_img = image::open(&dist_path)
        .map_err(|e| format!("decode dist {}: {}", dist_path.display(), e))?
        .to_rgb8();
    let (w, h) = src_img.dimensions();
    let (dw, dh) = dst_img.dimensions();
    if w != dw || h != dh {
        return Err(format!("dim mismatch ref={w}x{h} dist={dw}x{dh}").into());
    }
    let w_us = w as usize;
    let h_us = h as usize;
    if w_us < 8 || h_us < 8 {
        return Err("image too small (< 8 px)".into());
    }
    let src_pixels: Vec<[u8; 3]> =
        src_img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let dst_pixels: Vec<[u8; 3]> =
        dst_img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let mut config = ZensimConfig::default();
    config.extended_features = true;
    config.compute_iw_features = true;
    let result = compute_zensim_with_config(&src_pixels, &dst_pixels, w_us, h_us, config)
        .map_err(|e| format!("compute_zensim: {e:?}"))?;
    let features: Vec<f32> = result.features().iter().map(|&v| v as f32).collect();
    if features.len() != 372 {
        return Err(format!("expected 372 features, got {}", features.len()).into());
    }
    Ok(ExtractedRow {
        cell: cell.clone(),
        ref_basename,
        features,
    })
}

fn load_omni_cells(omni_dir: &Path, max_cells: usize) -> Vec<Cell> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let mut paths: Vec<PathBuf> = std::fs::read_dir(omni_dir)
        .expect("read omni-dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "parquet").unwrap_or(false))
        .collect();
    paths.sort();
    eprintln!("  {} omni parquet sidecars", paths.len());

    let mut out = Vec::new();
    for path in paths {
        let file = match std::fs::File::open(&path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("WARN open {}: {e}", path.display());
                continue;
            }
        };
        let builder = match ParquetRecordBatchReaderBuilder::try_new(file) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("WARN parquet builder {}: {e}", path.display());
                continue;
            }
        };
        let reader = match builder.build() {
            Ok(r) => r,
            Err(e) => {
                eprintln!("WARN parquet reader {}: {e}", path.display());
                continue;
            }
        };
        for batch_res in reader {
            let batch = match batch_res {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("WARN batch {}: {e}", path.display());
                    continue;
                }
            };
            // Resolve column indices once per batch
            let cols: HashMap<&str, usize> = batch
                .schema()
                .fields()
                .iter()
                .enumerate()
                .map(|(i, f)| (f.name().as_str(), i))
                .collect::<HashMap<_, _>>()
                .into_iter()
                .map(|(k, v)| (Box::leak(k.to_string().into_boxed_str()) as &str, v))
                .collect();
            macro_rules! col_str {
                ($name:expr, $row:expr) => {{
                    let c = batch.column(*cols.get($name).expect($name));
                    c.as_any()
                        .downcast_ref::<StringArray>()
                        .expect($name)
                        .value($row)
                        .to_string()
                }};
            }
            macro_rules! col_i32 {
                ($name:expr, $row:expr) => {{
                    let c = batch.column(*cols.get($name).expect($name));
                    if let Some(a) = c.as_any().downcast_ref::<Int32Array>() {
                        a.value($row)
                    } else if let Some(a) = c.as_any().downcast_ref::<arrow::array::Int64Array>() {
                        a.value($row) as i32
                    } else {
                        panic!("col {} not int", $name)
                    }
                }};
            }
            macro_rules! col_f64 {
                ($name:expr, $row:expr) => {{
                    let idx = match cols.get($name) {
                        Some(&i) => i,
                        None => 0,
                    };
                    if !cols.contains_key($name) {
                        f64::NAN
                    } else {
                        let c = batch.column(idx);
                        if let Some(a) = c.as_any().downcast_ref::<Float64Array>() {
                            if a.is_null($row) { f64::NAN } else { a.value($row) }
                        } else if let Some(a) = c.as_any().downcast_ref::<Float32Array>() {
                            if a.is_null($row) { f64::NAN } else { a.value($row) as f64 }
                        } else {
                            f64::NAN
                        }
                    }
                }};
            }
            for row in 0..batch.num_rows() {
                if out.len() >= max_cells {
                    break;
                }
                out.push(Cell {
                    image_path: col_str!("image_path", row),
                    codec: col_str!("codec", row),
                    q: col_i32!("q", row),
                    knob_tuple_json: col_str!("knob_tuple_json", row),
                    encoded_filename: col_str!("encoded_filename", row),
                    chunk_id: col_str!("chunk_id", row),
                    score_zensim_gpu: col_f64!("score_zensim_gpu", row),
                    score_ssim2_gpu: col_f64!("score_ssim2_gpu", row),
                    score_butteraugli_max_gpu: col_f64!("score_butteraugli_max_gpu", row),
                    score_butteraugli_pnorm3_gpu: col_f64!(
                        "score_butteraugli_pnorm3_gpu",
                        row
                    ),
                    score_cvvdp_imazen_v0_0_1: col_f64!("score_cvvdp_imazen_v0_0_1", row),
                    score_dssim_gpu: col_f64!("score_dssim_gpu", row),
                    score_iwssim_gpu: col_f64!("score_iwssim_gpu", row),
                });
            }
            if out.len() >= max_cells {
                break;
            }
        }
        if out.len() >= max_cells {
            break;
        }
    }
    out
}

fn output_schema() -> Arc<Schema> {
    let mut fields: Vec<Field> = Vec::with_capacity(12 + 372);
    fields.push(Field::new("image_path", DataType::Utf8, false));
    fields.push(Field::new("codec", DataType::Utf8, false));
    fields.push(Field::new("q", DataType::Int32, false));
    fields.push(Field::new("knob_tuple_json", DataType::Utf8, false));
    fields.push(Field::new("ref_basename", DataType::Utf8, false));
    fields.push(Field::new("score_zensim_gpu", DataType::Float64, true));
    fields.push(Field::new("score_ssim2_gpu", DataType::Float64, true));
    fields.push(Field::new("score_butteraugli_max_gpu", DataType::Float64, true));
    fields.push(Field::new("score_butteraugli_pnorm3_gpu", DataType::Float64, true));
    fields.push(Field::new("score_cvvdp_imazen_v0_0_1", DataType::Float64, true));
    fields.push(Field::new("score_dssim_gpu", DataType::Float64, true));
    fields.push(Field::new("score_iwssim_gpu", DataType::Float64, true));
    for i in 0..372 {
        fields.push(Field::new(format!("f{i}"), DataType::Float32, false));
    }
    Arc::new(Schema::new(fields))
}

fn build_record_batch(schema: &Arc<Schema>, rows: &[ExtractedRow]) -> RecordBatch {
    let n = rows.len();
    let image_path: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.cell.image_path.as_str()),
    ));
    let codec: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.cell.codec.as_str()),
    ));
    let q: ArrayRef = Arc::new(Int32Array::from_iter_values(
        rows.iter().map(|r| r.cell.q),
    ));
    let knob: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.cell.knob_tuple_json.as_str()),
    ));
    let ref_bn: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.ref_basename.as_str()),
    ));

    let mk_f64 = |getter: fn(&Cell) -> f64| -> ArrayRef {
        let v: Vec<Option<f64>> = rows
            .iter()
            .map(|r| {
                let x = getter(&r.cell);
                if x.is_nan() { None } else { Some(x) }
            })
            .collect();
        Arc::new(Float64Array::from(v))
    };
    let s_zensim = mk_f64(|c| c.score_zensim_gpu);
    let s_ssim2 = mk_f64(|c| c.score_ssim2_gpu);
    let s_butmax = mk_f64(|c| c.score_butteraugli_max_gpu);
    let s_butp3 = mk_f64(|c| c.score_butteraugli_pnorm3_gpu);
    let s_cvvdp = mk_f64(|c| c.score_cvvdp_imazen_v0_0_1);
    let s_dssim = mk_f64(|c| c.score_dssim_gpu);
    let s_iwssim = mk_f64(|c| c.score_iwssim_gpu);

    let mut feat_cols: Vec<ArrayRef> = Vec::with_capacity(372);
    for i in 0..372 {
        let v: Vec<f32> = rows.iter().map(|r| r.features[i]).collect();
        feat_cols.push(Arc::new(Float32Array::from(v)));
    }

    let mut cols: Vec<ArrayRef> = Vec::with_capacity(12 + 372);
    cols.push(image_path);
    cols.push(codec);
    cols.push(q);
    cols.push(knob);
    cols.push(ref_bn);
    cols.push(s_zensim);
    cols.push(s_ssim2);
    cols.push(s_butmax);
    cols.push(s_butp3);
    cols.push(s_cvvdp);
    cols.push(s_dssim);
    cols.push(s_iwssim);
    cols.extend(feat_cols);

    // Drop unused `n` warning lint
    let _ = n;
    RecordBatch::try_new(schema.clone(), cols).expect("build record batch")
}
