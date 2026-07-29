// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! SI-HDR full-vector feature extractor for the SI-HDR transfer study
//! (`benchmarks/sihdr_transfer_2026-07-29.md`).
//!
//! Extracts the 944 (or 956) feature vector for SI-HDR reconstruction
//! vs reference EXR pairs under the study's registered display model
//! (`/mnt/v/output/zensim/sihdr-transfer-2026-07-29/PROTOCOL.md`):
//!
//! - Per (scene, clip): `e = 1 / percentile_clip(maxRGB(ref))` (linear
//!   interpolation, non-finite pixels treated as 0), the paper's "3% or
//!   5% of pixels saturated" exposure (SIGGRAPH '22 §3.2).
//! - Reference: `nits = clamp(v * e * 100, 0, 1000)` per channel.
//! - Reconstruction: `nits = clamp(v * 100, 0, 1000)` per channel (the
//!   input-frame convention: recon value 1.0 = the scene's saturation
//!   point = 100 cd/m² on the experiment display; upper clamp = the
//!   PA32UCX 1000 cd/m² peak, per the paper's metric protocol).
//! - Route: declared-HDR streaming
//!   (`compute_folded720_append2_features_hdr`, `HdrEncoding::Linear`),
//!   csfw per `--mode` — the same call shape as `upiq_features_extract`'s
//!   HDR arm.
//!
//! Manifest TSV (no header): `cid <TAB> clip <TAB> ref_path <TAB> dist_path`
//! where `clip` is 95 or 97. Output CSV:
//! `cid,clip,e_scale,nonfinite_ref,nonfinite_dist,crop_x,crop_y,score228,f0..`
//! (crop_x/crop_y = the center-crop offsets of PROTOCOL deviation #2,
//! 0,0 when dimensions matched exactly).
//!
//! `--probe` skips extraction and prints per-pair pixel diagnostics
//! (dims, e under maxRGB vs BT.709-luminance percentile conventions,
//! frame-scale median ratio) for the registered pilot probe.
//!
//! Nothing here is fitted on anything: extraction + the fixed
//! SDR-trained readout only.

use std::io::Write as _;
use std::path::Path;

use rayon::prelude::*;
use zensim::feature_v2::{HdrEncoding, V2NewFeatureToggles, V2Scratch};
use zensim::source::{AlphaMode, ImageSource, PixelFormat};
use zensim::{Zensim, ZensimProfile};

/// Absolute-linear cd/m² source over interleaved RGBA f32 (the
/// `upiq_features_extract` `NitsImage`, verbatim).
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

/// Raw linear EXR pixels (non-finite → 0, counted) — pre-display-map.
struct RawExr {
    rgb: Vec<f32>,
    w: usize,
    h: usize,
    nonfinite: usize,
}

fn load_exr_raw(path: &Path) -> Result<RawExr, String> {
    let rgb_img = image::open(path)
        .map_err(|e| format!("{path:?}: {e}"))?
        .to_rgb32f();
    let (w, h) = (rgb_img.width() as usize, rgb_img.height() as usize);
    let mut rgb = rgb_img.into_raw();
    let mut nonfinite = 0usize;
    for v in &mut rgb {
        if !v.is_finite() {
            *v = 0.0;
            nonfinite += 1;
        }
    }
    Ok(RawExr {
        rgb,
        w,
        h,
        nonfinite,
    })
}

/// Percentile (0..100) with linear interpolation over a copied buffer.
fn percentile(values: &[f32], p: f64) -> f32 {
    assert!(!values.is_empty());
    let mut v = values.to_vec();
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = p / 100.0 * (v.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = (idx - lo as f64) as f32;
    v[lo] + frac * (v[hi] - v[lo])
}

/// The registered display map: `clamp(v * scale, 0, 1000)` per channel.
fn to_display(raw: &RawExr, scale: f32) -> NitsImage {
    let n = raw.w * raw.h;
    let data = (0..n)
        .map(|i| {
            let m = |v: f32| (v * scale).clamp(0.0, 1000.0);
            [
                m(raw.rgb[3 * i]),
                m(raw.rgb[3 * i + 1]),
                m(raw.rgb[3 * i + 2]),
                1.0,
            ]
        })
        .collect();
    NitsImage {
        data,
        w: raw.w,
        h: raw.h,
    }
}

/// Registered harness repair (PROTOCOL.md deviation #2): center-crop a
/// reconstruction that is strictly larger than its reference in BOTH
/// dimensions (drtmo's symmetric input padding: offset (128, 64) on the
/// 2016×1536 → 1888×1280 case, NCC-verified). Returns the (x, y) offsets.
fn center_crop_to(raw: &mut RawExr, w: usize, h: usize) -> (usize, usize) {
    assert!(raw.w > w && raw.h > h);
    let ox = (raw.w - w) / 2;
    let oy = (raw.h - h) / 2;
    let mut rgb = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        let start = ((oy + y) * raw.w + ox) * 3;
        rgb.extend_from_slice(&raw.rgb[start..start + w * 3]);
    }
    raw.rgb = rgb;
    raw.w = w;
    raw.h = h;
    (ox, oy)
}

fn max_rgb(raw: &RawExr) -> Vec<f32> {
    (0..raw.w * raw.h)
        .map(|i| {
            raw.rgb[3 * i]
                .max(raw.rgb[3 * i + 1])
                .max(raw.rgb[3 * i + 2])
        })
        .collect()
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

struct Row {
    cid: String,
    clip: f64,
    ref_path: String,
    dist_path: String,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let manifest = arg(&args, "--manifest").expect("--manifest TSV required");
    let out = arg(&args, "--out").unwrap_or_else(|| "sihdr_features.csv".into());
    let mode = arg(&args, "--mode").unwrap_or_else(|| "944".into());
    let probe = args.iter().any(|a| a == "--probe");
    let csfw_on = match mode.as_str() {
        "944" => false,
        "956" => true,
        other => panic!("--mode must be 944 or 956, got {other}"),
    };

    let rows: Vec<Row> = std::fs::read_to_string(&manifest)
        .expect("read manifest")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert_eq!(f.len(), 4, "manifest row: cid\\tclip\\tref\\tdist ({l})");
            Row {
                cid: f[0].to_string(),
                clip: f[1].parse().expect("clip percentile"),
                ref_path: f[2].to_string(),
                dist_path: f[3].to_string(),
            }
        })
        .collect();
    eprintln!(
        "{} pairs (mode={mode} csfw={csfw_on} probe={probe})",
        rows.len()
    );

    if probe {
        // Pixel-statistics pilot (no extraction, no JOD contact).
        for row in &rows {
            let r = match load_exr_raw(Path::new(&row.ref_path)) {
                Ok(r) => r,
                Err(e) => {
                    println!("PROBE {} REF_FAIL {e}", row.cid);
                    continue;
                }
            };
            let d = match load_exr_raw(Path::new(&row.dist_path)) {
                Ok(d) => d,
                Err(e) => {
                    println!("PROBE {} DIST_FAIL {e}", row.cid);
                    continue;
                }
            };
            let mrgb = max_rgb(&r);
            let sat_maxrgb = percentile(&mrgb, row.clip);
            let luma: Vec<f32> = (0..r.w * r.h)
                .map(|i| {
                    0.2126 * r.rgb[3 * i] + 0.7152 * r.rgb[3 * i + 1] + 0.0722 * r.rgb[3 * i + 2]
                })
                .collect();
            let sat_luma = percentile(&luma, row.clip);
            let e = 1.0 / sat_maxrgb;
            let med_ref = percentile(&r.rgb, 50.0);
            let med_dist = percentile(&d.rgb, 50.0);
            println!(
                "PROBE {} ref={}x{} dist={}x{} nf_ref={} nf_dist={} sat_maxrgb={sat_maxrgb:.6e} \
                 sat_luma={sat_luma:.6e} e={e:.6e} med_ref={med_ref:.6e} med_dist={med_dist:.6e} \
                 frame_ratio_med={:.4}",
                row.cid,
                r.w,
                r.h,
                d.w,
                d.h,
                r.nonfinite,
                d.nonfinite,
                med_dist / (med_ref * e),
            );
        }
        return;
    }

    let weights = zensim::WEIGHTS;
    let mut indexed: Vec<(usize, String, usize)> = rows
        .par_iter()
        .enumerate()
        .map_init(V2Scratch::new, |scratch, (i, row)| {
            let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
            let r = match load_exr_raw(Path::new(&row.ref_path)) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("REF FAIL {e}");
                    return None;
                }
            };
            let mut d = match load_exr_raw(Path::new(&row.dist_path)) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("DIST FAIL {e}");
                    return None;
                }
            };
            let mut crop = (0usize, 0usize);
            if (r.w, r.h) != (d.w, d.h) {
                if d.w > r.w && d.h > r.h {
                    crop = center_crop_to(&mut d, r.w, r.h);
                    eprintln!(
                        "CROP {}: {}x{} -> {}x{} at ({},{})",
                        row.cid,
                        r.w + 2 * crop.0,
                        r.h + 2 * crop.1,
                        r.w,
                        r.h,
                        crop.0,
                        crop.1
                    );
                } else {
                    eprintln!(
                        "SKIP dim mismatch {}: {}x{} vs {}x{}",
                        row.cid, r.w, r.h, d.w, d.h
                    );
                    return None;
                }
            }
            let sat = percentile(&max_rgb(&r), row.clip);
            if !(sat.is_finite() && sat > 0.0) {
                eprintln!("SKIP degenerate saturation point {}: {sat}", row.cid);
                return None;
            }
            let e_scale = 1.0 / sat;
            let (nf_r, nf_d) = (r.nonfinite, d.nonfinite);
            let ref_img = to_display(&r, e_scale * 100.0);
            let dist_img = to_display(&d, 100.0);
            drop((r, d));
            let res = z.compute_folded720_append2_features_hdr(
                &ref_img,
                &dist_img,
                HdrEncoding::Linear,
                V2NewFeatureToggles {
                    csfw_block: csfw_on,
                    ..V2NewFeatureToggles::default()
                },
                scratch,
            );
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
                "{},{},{e_scale},{nf_r},{nf_d},{},{},{score}",
                row.cid, row.clip, crop.0, crop.1
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
    let expected = if csfw_on { 956 } else { 944 };
    assert_eq!(n_feat, expected, "feature width vs --mode");
    let f = std::fs::File::create(&out).expect("create out");
    let mut w = std::io::BufWriter::new(f);
    let mut header =
        String::from("cid,clip,e_scale,nonfinite_ref,nonfinite_dist,crop_x,crop_y,score228");
    for k in 0..n_feat {
        header.push_str(&format!(",f{k}"));
    }
    writeln!(w, "{header}").expect("write header");
    for (_, line, _) in &indexed {
        writeln!(w, "{line}").expect("write row");
    }
    w.flush().expect("flush");
    eprintln!("done: {} rows x {n_feat} features -> {out}", indexed.len());
}
