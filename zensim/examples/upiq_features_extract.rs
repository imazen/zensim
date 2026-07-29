// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! Full-vector UPIQ feature extractor for the HDR cross-route
//! commensurability study
//! (`benchmarks/hdr_dmean_commensurability_2026-07-29.md`).
//!
//! Same pair-enumeration protocol as `upiq_hdr924_score` (the chunk-2 V4
//! gate scorer — iterate `upiq_subjective_scores.csv` in file order), but
//! emits the FULL feature vector per condition (944 or 956 per `--mode`)
//! plus the fixed PreviewV0_2 readout over the folded 228 prefix, for
//! BOTH strata:
//!
//! - `is_hdr==1` (narwaria/korshunov, absolute-luminance EXR): loaded
//!   with the control scorer's loader (`image::open().to_rgb32f()`,
//!   values preserved as cd/m²) and run through the declared-HDR
//!   streaming route (`compute_folded720_append2_features_hdr`,
//!   `HdrEncoding::Linear`, csfw toggle per `--mode` — the
//!   `v2_ab_extract` `foldapp2hdr100`/`foldcsfwhdr100` call shape).
//! - `is_hdr==0` (tid2013/live, u8 sRGB PNG): zen_io-decoded and run
//!   through the SDR streaming route
//!   (`compute_folded720_append_features_streaming`, append2 toggle on,
//!   csfw toggle per `--mode` — the `foldapp2`/`foldcsfw` call shape).
//!
//! Nothing here is fitted on anything: extraction + a fixed SDR-trained
//! readout only.
//!
//! Usage:
//!   upiq_features_extract [--images DIR] [--subjective CSV] [--out CSV]
//!                         [--mode 944|956] [--subset hdr|sdr|all]

#[path = "support/zen_io.rs"]
mod zen_io;

use std::path::Path;

use rayon::prelude::*;
use zensim::feature_v2::{HdrEncoding, V2NewFeatureToggles, V2Scratch};
use zensim::source::{AlphaMode, ImageSource, PixelFormat};
use zensim::{RgbSlice, Zensim, ZensimProfile};

/// Absolute-linear cd/m² source over interleaved RGBA f32.
struct NitsImage {
    data: Vec<[f32; 4]>,
    w: usize,
    h: usize,
}

impl ImageSource for NitsImage {
    fn width(&self) -> usize {
        self.w
    }
    fn height(&self) -> usize {
        self.h
    }
    fn pixel_format(&self) -> PixelFormat {
        PixelFormat::LinearF32Rgba
    }
    fn row_bytes(&self, y: usize) -> &[u8] {
        bytemuck::cast_slice(&self.data[y * self.w..(y + 1) * self.w])
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
    fn is_hdr(&self) -> bool {
        true
    }
}

/// Load an absolute-luminance EXR (values preserved, HDR magnitudes
/// kept) — byte-for-byte the `upiq_hdr924_score` loader.
fn load_exr(path: &Path) -> Result<NitsImage, String> {
    let rgb = image::open(path)
        .map_err(|e| format!("{path:?}: {e}"))?
        .to_rgb32f();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let raw = rgb.into_raw();
    let data = (0..w * h)
        .map(|i| [raw[3 * i], raw[3 * i + 1], raw[3 * i + 2], 1.0])
        .collect();
    Ok(NitsImage { data, w, h })
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

struct Row {
    cid: String,
    dataset: String,
    is_hdr: bool,
    jod: f64,
    test_rel: String,
    ref_rel: String,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let images = arg(&args, "--images")
        .unwrap_or_else(|| "/mnt/v/datasets/upiq_extracted/upiq_dataset/images".into());
    let subjective = arg(&args, "--subjective")
        .unwrap_or_else(|| "/mnt/v/datasets/upiq/upiq_subjective_scores.csv".into());
    let out = arg(&args, "--out").unwrap_or_else(|| "/tmp/upiq_features.csv".into());
    let mode = arg(&args, "--mode").unwrap_or_else(|| "956".into());
    let subset = arg(&args, "--subset").unwrap_or_else(|| "all".into());
    let csfw_on = match mode.as_str() {
        "944" => false,
        "956" => true,
        other => panic!("--mode must be 944 or 956, got {other}"),
    };

    let text = std::fs::read_to_string(&subjective).expect("read subjective csv");
    let mut lines = text.lines();
    let headers: Vec<&str> = lines.next().expect("header").split(',').collect();
    let col = |name: &str| {
        headers
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("missing column {name}"))
    };
    let (c_cid, c_ds, c_hdr, c_test, c_ref, c_jod) = (
        col("condition_id"),
        col("dataset"),
        col("is_hdr"),
        col("test_file"),
        col("reference_file"),
        col("JOD"),
    );

    let rows: Vec<Row> = lines
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| {
            let rec: Vec<&str> = line.split(',').collect();
            let is_hdr = rec[c_hdr] == "1";
            match subset.as_str() {
                "hdr" if !is_hdr => return None,
                "sdr" if is_hdr => return None,
                _ => {}
            }
            Some(Row {
                cid: rec[c_cid].to_string(),
                dataset: rec[c_ds].to_string(),
                is_hdr,
                jod: rec[c_jod].parse().expect("JOD"),
                test_rel: rec[c_test].to_string(),
                ref_rel: rec[c_ref].to_string(),
            })
        })
        .collect();
    eprintln!(
        "{} conditions (subset={subset}, mode={mode} csfw={})",
        rows.len(),
        csfw_on
    );

    let weights = zensim::WEIGHTS;

    // Per-row extraction (no ref cache: deterministic, order-free).
    let mut indexed: Vec<(usize, String, usize)> = rows
        .par_iter()
        .enumerate()
        .map_init(V2Scratch::new, |scratch, (i, row)| {
            let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
            let ref_path = Path::new(&images).join(&row.ref_rel);
            let dist_path = Path::new(&images).join(&row.test_rel);
            let res = if row.is_hdr {
                let r = match load_exr(&ref_path) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("REF FAIL {e}");
                        return None;
                    }
                };
                let d = match load_exr(&dist_path) {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("DIST FAIL {e}");
                        return None;
                    }
                };
                if (r.w, r.h) != (d.w, d.h) {
                    eprintln!("SKIP dim mismatch: {}", row.cid);
                    return None;
                }
                z.compute_folded720_append2_features_hdr(
                    &r,
                    &d,
                    HdrEncoding::Linear,
                    V2NewFeatureToggles {
                        csfw_block: csfw_on,
                        ..V2NewFeatureToggles::default()
                    },
                    scratch,
                )
            } else {
                let (r_px, rw, rh) = zen_io::decode_rgb8(&ref_path);
                let (d_px, dw, dh) = zen_io::decode_rgb8(&dist_path);
                if (rw, rh) != (dw, dh) {
                    eprintln!("SKIP dim mismatch: {}", row.cid);
                    return None;
                }
                z.compute_folded720_append_features_streaming(
                    &RgbSlice::new(&r_px, rw, rh),
                    &RgbSlice::new(&d_px, dw, dh),
                    V2NewFeatureToggles {
                        append2_block: true,
                        csfw_block: csfw_on,
                        ..V2NewFeatureToggles::default()
                    },
                    scratch,
                )
            };
            let res = match res {
                Ok(res) => res,
                Err(e) => {
                    eprintln!("SCORE FAIL {}: {e:?}", row.cid);
                    return None;
                }
            };
            let feats = res.features();
            let (score, _) =
                zensim::try_score_from_features(&feats[..228], weights).expect("readout");
            let mut line = format!(
                "{},{},{},{},{score}",
                row.cid,
                row.dataset,
                if row.is_hdr { 1 } else { 0 },
                row.jod
            );
            for v in feats {
                line.push(',');
                line.push_str(&format!("{v}"));
            }
            Some((i, line, feats.len()))
        })
        .flatten()
        .collect();
    indexed.sort_unstable_by_key(|(i, _, _)| *i);

    let n_feat = indexed.first().map(|(_, _, n)| *n).unwrap_or(0);
    assert!(
        indexed.iter().all(|(_, _, n)| *n == n_feat),
        "inconsistent feature widths"
    );
    let mut w = String::from("condition_id,dataset,is_hdr,jod,score228");
    for k in 0..n_feat {
        w.push_str(&format!(",f{k}"));
    }
    w.push('\n');
    for (_, line, _) in &indexed {
        w.push_str(line);
        w.push('\n');
    }
    std::fs::write(&out, w).expect("write out");
    eprintln!("done: {} rows x {n_feat} features -> {out}", indexed.len());
}
