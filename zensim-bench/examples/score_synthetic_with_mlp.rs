//! Score the synthetic training set with a saved MLP bake, for
//! smoothness analysis of a quality metric.
//!
//! For each pair in the synthetic CSV (which already has source_path,
//! codec, quality, gpu_ssimulacra2 columns), this loads the precomputed
//! feature cache, runs the MLP forward, and writes:
//!   source_path,codec,quality,ssim2_score,predicted_distance
//!
//! Downstream Python groups by (source_path, codec) and computes
//! smoothness statistics on the predicted_distance vs quality sweep
//! per group. Smoothness matters when the metric is a human-facing
//! quality target — users tune to a specific value and expect small
//! quality changes to map to small metric changes.
//!
//! Usage:
//!   score_synthetic_with_mlp \
//!     --csv /mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv \
//!     --features-cache .../features.20260308_162434.bin \
//!     --bake .../v05_mlp_ssim2_synvalidation_*.bin \
//!     [--zenanalyze-tsv .../zenanalyze_union_v1.tsv] \
//!     [--zenanalyze-features dct_compressibility_y,...] \
//!     --output /tmp/synth_scored_v05.csv

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rayon::prelude::*;
use zenpredict::{Model, Predictor};

#[derive(Debug, Clone)]
struct CsvPair {
    source_path: String,
    codec: String,
    quality: u32,
    ssim2: f64,
}

fn load_csv(path: &Path) -> Vec<CsvPair> {
    let f = File::open(path).expect("open csv");
    let mut rdr = BufReader::new(f);
    let mut header = String::new();
    rdr.read_line(&mut header).expect("read header");
    let cols: Vec<&str> = header.trim().split(',').collect();
    let i_src = cols.iter().position(|c| *c == "source_path").expect("source_path");
    let i_codec = cols.iter().position(|c| *c == "codec").expect("codec");
    let i_q = cols.iter().position(|c| *c == "quality").expect("quality");
    let i_ssim2 = cols.iter().position(|c| *c == "gpu_ssimulacra2").expect("gpu_ssimulacra2");
    let mut pairs = Vec::new();
    for line in rdr.lines() {
        let line = line.unwrap_or_default();
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        let src = cols.get(i_src).map(|s| s.to_string()).unwrap_or_default();
        let cod = cols.get(i_codec).map(|s| s.to_string()).unwrap_or_default();
        let q: u32 = cols.get(i_q).and_then(|s| s.parse().ok()).unwrap_or(0);
        let ss: f64 = cols.get(i_ssim2).and_then(|s| s.parse().ok()).unwrap_or(f64::NAN);
        pairs.push(CsvPair {
            source_path: src,
            codec: cod,
            quality: q,
            ssim2: ss,
        });
    }
    pairs
}

/// Read the zensim-validate feature cache binary directly. Format is
/// the v3 layout from main.rs save_feature_cache: ZSFC + version + 4
/// validation u32s + n_pairs(u32) + n_features(u16) + dataset name +
/// valid_indices + features (f32 LE, row-major).
fn load_features(path: &Path) -> (Vec<u32>, Vec<Vec<f32>>, usize) {
    let mut f = File::open(path).expect("open features cache");
    let mut data = Vec::new();
    f.read_to_end(&mut data).expect("read cache");
    let mut pos = 0usize;
    let take = |p: &mut usize, n: usize| -> Vec<u8> {
        let v = data[*p..*p + n].to_vec();
        *p += n;
        v
    };
    let magic = take(&mut pos, 4);
    assert_eq!(&magic, b"ZSFC", "bad magic");
    let version = u32::from_le_bytes(take(&mut pos, 4).try_into().unwrap());
    assert!(version == 3, "expected version 3 cache, got {version}");
    let _num_scales = u32::from_le_bytes(take(&mut pos, 4).try_into().unwrap());
    let _blur_passes = take(&mut pos, 1)[0];
    let _blur_radius = u32::from_le_bytes(take(&mut pos, 4).try_into().unwrap());
    let _reserved = u32::from_le_bytes(take(&mut pos, 4).try_into().unwrap());
    let n_pairs = u32::from_le_bytes(take(&mut pos, 4).try_into().unwrap()) as usize;
    let n_features = u16::from_le_bytes(take(&mut pos, 2).try_into().unwrap()) as usize;
    let name_len = u16::from_le_bytes(take(&mut pos, 2).try_into().unwrap()) as usize;
    let _name = take(&mut pos, name_len);
    let mut valid_indices = Vec::with_capacity(n_pairs);
    for _ in 0..n_pairs {
        valid_indices.push(u32::from_le_bytes(take(&mut pos, 4).try_into().unwrap()));
    }
    let mut features = Vec::with_capacity(n_pairs);
    for _ in 0..n_pairs {
        let mut row = Vec::with_capacity(n_features);
        for _ in 0..n_features {
            row.push(f32::from_le_bytes(take(&mut pos, 4).try_into().unwrap()));
        }
        features.push(row);
    }
    (valid_indices, features, n_features)
}

#[derive(Debug, Clone)]
struct Augment {
    table: Arc<HashMap<String, Vec<f64>>>,
    selected: Vec<usize>,
    n_extras: usize,
}

fn load_zenanalyze_tsv(path: &Path, feature_spec: &str) -> Augment {
    let data = std::fs::read_to_string(path).expect("read tsv");
    let mut lines = data.lines();
    let header = lines.next().unwrap_or("");
    let cols: Vec<&str> = header.split('\t').collect();
    assert!(cols.len() > 2 && cols[0] == "stem" && cols[1] == "source_path", "bad TSV header");
    let names: Vec<String> = cols[2..].iter().map(|s| s.to_string()).collect();
    let mut selected = Vec::new();
    for raw in feature_spec.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        let i = names.iter().position(|n| n.eq_ignore_ascii_case(raw)).unwrap_or_else(|| {
            eprintln!("feature {raw} not in TSV; available: {names:?}");
            std::process::exit(2);
        });
        selected.push(i);
    }
    let n_feat = names.len();
    let mut table = HashMap::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() != 2 + n_feat {
            continue;
        }
        let stem = cols[0].to_string();
        let mut vals = Vec::with_capacity(n_feat);
        for c in &cols[2..] {
            let v: f64 = c.parse().unwrap_or(0.0);
            vals.push(if v.is_finite() { v } else { 0.0 });
        }
        table.insert(stem, vals);
    }
    Augment {
        n_extras: selected.len(),
        selected,
        table: Arc::new(table),
    }
}

fn stem_of(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string()
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut csv: Option<PathBuf> = None;
    let mut cache: Option<PathBuf> = None;
    let mut bake: Option<PathBuf> = None;
    let mut tsv: Option<PathBuf> = None;
    let mut feats: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut tier: usize = 228;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--csv" => { csv = Some(args[i+1].clone().into()); i += 2; }
            "--features-cache" => { cache = Some(args[i+1].clone().into()); i += 2; }
            "--bake" => { bake = Some(args[i+1].clone().into()); i += 2; }
            "--zenanalyze-tsv" => { tsv = Some(args[i+1].clone().into()); i += 2; }
            "--zenanalyze-features" => { feats = Some(args[i+1].clone()); i += 2; }
            "--output" => { output = Some(args[i+1].clone().into()); i += 2; }
            "--tier" => { tier = args[i+1].parse().unwrap(); i += 2; }
            other => { eprintln!("unknown arg {other}"); std::process::exit(2); }
        }
    }
    let _ = args.drain(..);
    let csv = csv.expect("--csv");
    let cache = cache.expect("--features-cache");
    let bake = bake.expect("--bake");
    let output = output.expect("--output");

    eprintln!("loading csv {}", csv.display());
    let pairs = load_csv(&csv);
    eprintln!("loaded {} pairs from CSV", pairs.len());

    eprintln!("loading feature cache {}", cache.display());
    let (valid_indices, features, n_loaded) = load_features(&cache);
    eprintln!("loaded {} cache rows × {} features", features.len(), n_loaded);

    let augment = if let (Some(tsv_p), Some(spec)) = (tsv.as_ref(), feats.as_ref()) {
        eprintln!("loading zenanalyze TSV {}", tsv_p.display());
        Some(load_zenanalyze_tsv(tsv_p, spec))
    } else {
        None
    };

    let bake_bytes: Vec<u8> = std::fs::read(&bake).expect("read bake");
    let bake_static: &'static [u8] = Box::leak(bake_bytes.into_boxed_slice());
    let model = Model::from_bytes(bake_static).expect("load model");
    let n_inputs = model.n_inputs();
    let n_extras = augment.as_ref().map(|a| a.n_extras).unwrap_or(0);
    let needed_base = n_inputs - n_extras;
    eprintln!("model n_inputs={n_inputs}, base={needed_base}, n_extras={n_extras}");
    assert!(needed_base <= tier, "model wants more base features than tier {tier}");

    let n_pairs = features.len();
    let augment_ref = augment.as_ref();

    // Score in parallel.
    let scored: Vec<Option<(usize, f32)>> = features
        .par_iter()
        .enumerate()
        .map(|(cache_idx, row)| {
            let pair_idx = valid_indices[cache_idx] as usize;
            if pair_idx >= pairs.len() {
                return None;
            }
            let src_path = &pairs[pair_idx].source_path;
            let mut feats: Vec<f32> = row[..tier.min(row.len())].to_vec();
            if let Some(a) = augment_ref {
                let key = stem_of(src_path);
                let extras = a.table.get(&key);
                for &idx in &a.selected {
                    let v: f32 = extras.map(|v| v[idx] as f32).unwrap_or(0.0);
                    feats.push(v);
                }
            }
            if feats.len() < n_inputs {
                return None;
            }
            let model = Model::from_bytes(bake_static).expect("model");
            let mut p = Predictor::new(model);
            let pred = p.predict(&feats[..n_inputs]).expect("predict")[0];
            Some((pair_idx, pred))
        })
        .collect();

    let mut out = BufWriter::new(File::create(&output).expect("create output"));
    writeln!(out, "source_path,codec,quality,ssim2_score,predicted_distance").unwrap();
    let mut n_ok = 0usize;
    for s in scored {
        let (pair_idx, pred) = match s {
            Some(v) => v,
            None => continue,
        };
        let p = &pairs[pair_idx];
        writeln!(out, "{},{},{},{},{}", p.source_path, p.codec, p.quality, p.ssim2, pred).unwrap();
        n_ok += 1;
    }
    eprintln!("wrote {n_ok}/{n_pairs} pairs to {}", output.display());
    let _ = bake;
}
