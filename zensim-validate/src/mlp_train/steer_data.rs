//! Label-free K-table admission and reuse of the canonical fx1 transformation.
use arrow::array::{Array, Float32Array, ListArray, UInt32Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::{collections::BTreeMap, path::Path};

pub(super) struct Image {
    pub base: Vec<f64>,
    pub repairs: Vec<Vec<f64>>,
    pub sizes: Vec<u32>,
}

pub(super) fn load(
    path: &Path,
    derived: Option<&Path>,
    width: usize,
    mean: &[f64],
    scale: &[f64],
) -> Result<Vec<Image>, String> {
    let sidecar = path.with_extension("parquet.manifest.json");
    for p in [
        path.to_path_buf(),
        path.canonicalize().map_err(|e| e.to_string())?,
        sidecar.clone(),
        sidecar.canonicalize().map_err(|e| e.to_string())?,
    ] {
        if p.components().any(|c| {
            let c = c.as_os_str().to_string_lossy().to_ascii_lowercase();
            c.contains("_sealed")
                || c.contains("holdout")
                || c.contains("terminal")
                || c.starts_with("labels__")
        }) {
            return Err("protected K input ancestry".into());
        }
    }
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&sidecar).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if manifest["schema"] != "e34-steer-pairs-v1"
        || manifest["formula_revision"] != 5
        || manifest["tier"] != "v3"
        || manifest["labels_read"] != false
        || manifest["build_complete"] != true
        || manifest["base_images"] != 1536
        || manifest["origins"] != 128
        || manifest["eligibility_receipt_sha256"]
            .as_str()
            .is_none_or(|x| x.len() != 64)
        || manifest["extractor_sha256"]
            .as_str()
            .is_none_or(|x| x.len() != 64)
    {
        return Err(
            "K needs the completed registered label-free Rev5/v3 augmentation receipt".into(),
        );
    }
    if manifest["table_sha256"]
        != crate::train_manifest::sha256_file(path).map_err(|e| e.to_string())?
    {
        return Err("K table digest mismatch".into());
    }
    let raw_width = manifest["width"].as_u64().ok_or("K missing raw width")? as usize;
    let derived = derived
        .map(crate::derived_inputs::Fx1Declaration::load)
        .transpose()?
        .map(|(d, _)| d);
    if width != raw_width + derived.as_ref().map_or(0, |d| d.products.len()) {
        return Err("K layout disagrees with rank inputs".into());
    }
    let keep = super::input_keep_mask();
    if keep.as_ref().is_some_and(|k| k.len() != width) {
        return Err("K keep-mask width".into());
    }
    let transform = |array: &dyn Array| -> Result<Vec<f64>, String> {
        let row = array
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or("K list must contain f32")?;
        if row.len() != raw_width || row.null_count() != 0 {
            return Err("K raw row width/null".into());
        }
        let mut flat: Vec<f64> = row.values().iter().map(|&x| f64::from(x)).collect();
        if let Some(d) = &derived {
            d.extend_dense(&mut flat, 1, raw_width)?;
        }
        for i in 0..width {
            if keep.as_ref().is_none_or(|k| k[i]) {
                if !flat[i].is_finite() {
                    return Err(format!("K nonfinite consumed input {i}"));
                }
                flat[i] = (flat[i] - mean[i]) / scale[i].max(1e-12);
            } else {
                flat[i] = 0.;
            }
        }
        Ok(flat)
    };
    let builder = ParquetRecordBatchReaderBuilder::try_new(
        std::fs::File::open(path).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let names: Vec<_> = builder
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    if names != ["base_id", "block_size", "base_features", "repair_features"] {
        return Err("K unexpected columns; labels are forbidden".into());
    }
    let reader = builder
        .with_batch_size(256)
        .build()
        .map_err(|e| e.to_string())?;
    let mut images: BTreeMap<u32, Image> = BTreeMap::new();
    let mut rows = 0usize;
    for batch in reader {
        let b = batch.map_err(|e| e.to_string())?;
        if b.columns().iter().any(|a| a.null_count() != 0) {
            return Err("K null column".into());
        }
        let ids = b
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or("K base_id type")?;
        let sizes = b
            .column(1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or("K block_size type")?;
        let bases = b
            .column(2)
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or("K base_features type")?;
        let repairs = b
            .column(3)
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or("K repair_features type")?;
        for r in 0..b.num_rows() {
            if ![8, 16, 32, 64].contains(&sizes.value(r)) {
                return Err("K block size".into());
            }
            let base = transform(bases.value(r).as_ref())?;
            let repair = transform(repairs.value(r).as_ref())?;
            let entry = images.entry(ids.value(r)).or_insert_with(|| Image {
                base: base.clone(),
                repairs: Vec::new(),
                sizes: Vec::new(),
            });
            if base
                .iter()
                .zip(&entry.base)
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err("K inconsistent base".into());
            }
            // Builder removes bit-identical raw rows. An arm may consume only a
            // subset and thus see zero gains; those are counted during training.
            entry.repairs.push(repair);
            entry.sizes.push(sizes.value(r));
            rows += 1;
        }
    }
    if images.len() != 1536 || manifest["repair_rows"] != rows || rows > 98304 {
        return Err("K image/repair count differs from receipt or registration".into());
    }
    for image in images.values() {
        for size in [8, 16, 32, 64] {
            if image.sizes.iter().filter(|&&s| s == size).count() > 16 {
                return Err("K exceeds per-size repair cap".into());
            }
        }
    }
    Ok(images.into_values().collect())
}
