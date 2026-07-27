// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! V4 gate scorer (`benchmarks/hdr_streaming_gates_2026-07-27.md`): run the
//! STREAMING HDR route (`compute_folded720_append_features_hdr`,
//! `HdrEncoding::Linear`) over the UPIQ HDR codec subset and emit the CSV
//! `scripts/upiq_eval.py` consumes — the same pair-enumeration protocol as
//! `zensim-validate/src/bin/upiq_pu_score.rs` (the 2026-06-01 control).
//!
//! Scalar readout: `try_score_from_features` with the PreviewV0_2 linear
//! weights over the folded vector's first 228 slots (v1 basic-156 + the
//! zeroed deprecated peak block) — a FIXED SDR-trained readout, nothing
//! fitted on UPIQ. Comparable against the recorded
//! `zensim-PU (PreviewV0_2, X=4) = 0.687` leg (same weight family; the
//! peak block is live there and zeroed here — stated in the gate doc).
//!
//! Usage mirrors the control:
//!   upiq_hdr924_score [--images DIR] [--subjective CSV] [--out CSV]

use std::collections::HashMap;
use std::path::Path;

use zensim::feature_v2::{HdrEncoding, V2NewFeatureToggles, V2Scratch};
use zensim::source::{AlphaMode, ImageSource, PixelFormat};
use zensim::{Zensim, ZensimProfile};

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

/// Load an absolute-luminance EXR (values preserved, HDR magnitudes kept)
/// — the control scorer's loader, emitting the interleaved source form.
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let images = arg(&args, "--images")
        .unwrap_or_else(|| "/mnt/v/datasets/upiq_extracted/upiq_dataset/images".into());
    let subjective = arg(&args, "--subjective")
        .unwrap_or_else(|| "/mnt/v/datasets/upiq/upiq_subjective_scores.csv".into());
    let out = arg(&args, "--out").unwrap_or_else(|| "/tmp/zensim_hdr924_scores.csv".into());

    let text = std::fs::read_to_string(&subjective).expect("read subjective csv");
    let mut lines = text.lines();
    let headers: Vec<&str> = lines.next().expect("header").split(',').collect();
    let col = |name: &str| {
        headers
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("missing column {name}"))
    };
    let (c_cid, c_ds, c_hdr, c_test, c_ref) = (
        col("condition_id"),
        col("dataset"),
        col("is_hdr"),
        col("test_file"),
        col("reference_file"),
    );

    let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
    let mut scratch = V2Scratch::new();
    let weights = zensim::WEIGHTS;

    let mut ref_cache: HashMap<String, NitsImage> = HashMap::new();
    let mut rows: Vec<(String, f64)> = Vec::new();
    let (mut ok, mut skip, mut err) = (0usize, 0usize, 0usize);

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let rec: Vec<&str> = line.split(',').collect();
        let dataset = rec[c_ds];
        if dataset != "narwaria" && dataset != "korshunov" {
            continue;
        }
        if rec[c_hdr] != "1" {
            continue;
        }
        let cid = rec[c_cid].to_string();
        let test_path = Path::new(&images).join(rec[c_test]);
        let ref_rel = rec[c_ref].to_string();
        let ref_path = Path::new(&images).join(&ref_rel);

        if !ref_cache.contains_key(&ref_rel) {
            match load_exr(&ref_path) {
                Ok(img) => {
                    ref_cache.insert(ref_rel.clone(), img);
                }
                Err(e) => {
                    eprintln!("REF FAIL {e}");
                    err += 1;
                    continue;
                }
            }
        }
        let r = &ref_cache[&ref_rel];
        let d = match load_exr(&test_path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("DIST FAIL {e}");
                err += 1;
                continue;
            }
        };
        if d.w != r.w || d.h != r.h {
            skip += 1;
            continue;
        }
        let res = match z.compute_folded720_append_features_hdr(
            r,
            &d,
            HdrEncoding::Linear,
            V2NewFeatureToggles::default(),
            &mut scratch,
        ) {
            Ok(res) => res,
            Err(e) => {
                eprintln!("SCORE FAIL {cid}: {e:?}");
                err += 1;
                continue;
            }
        };
        let (score, _) =
            zensim::try_score_from_features(&res.features()[..228], weights).expect("readout");
        rows.push((cid, score));
        ok += 1;
        if ok % 50 == 0 {
            eprintln!("scored {ok} pairs…");
        }
    }

    let mut w = String::from("condition_id,zensim_hdr924\n");
    for (cid, s) in &rows {
        w.push_str(&format!("{cid},{s}\n"));
    }
    std::fs::write(&out, w).expect("write out");
    eprintln!("done: {ok} scored, {skip} dim-skipped, {err} errored → {out}");
}
