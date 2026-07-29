// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! HDR-VDC viewing-condition feature extractor for the hdrvdc conditions
//! study (`benchmarks/hdrvdc_conditions_2026-07-29.md`).
//!
//! Extracts the 944 feature vector for HDR-VDC ref/test frame pairs under
//! the study's registered configurations
//! (`/mnt/v/output/zensim/hdrvdc-conditions-2026-07-29/PROTOCOL.md`):
//!
//! - Inputs are rgb48 PNGs of FULL-RANGE PQ (SMPTE ST 2084) code values,
//!   BT.2020 primaries, produced by the registered ffmpeg decode chain
//!   (display-frame reconstruction; Lanczos display upscale already
//!   applied). Code values = u16 / 65535, fed as `LinearF32Rgba`.
//! - `dim=1` rows reproduce the experiment's dimming shader (stimulus
//!   reconstruction, applied to BOTH ref and test): PQ-decode at spec
//!   peak (10 000 cd/m², no display clamp) → linear ÷ 8 → PQ-re-encode,
//!   in f64 with exact ST 2084 constants — the same chain as the official
//!   `gfxdisp/HDR-VDC` display-model example (minus its G2 LUT, which is
//!   replaced by the zensim front-end display model under test).
//! - Route: declared-HDR streaming
//!   (`compute_folded720_append2_features_hdr`,
//!   `HdrEncoding::Pq { peak_nits }` per manifest row — the front-end
//!   display model under test), csfw OFF (mode 944), same call shape as
//!   `sihdr_features_extract`.
//!
//! Manifest TSV (no header):
//! `key <TAB> dim(0|1) <TAB> peak_nits <TAB> ref_png <TAB> dist_png`.
//! Output CSV: `key,dim,peak_nits,score228,f0..f943`.
//!
//! Nothing here is fitted on anything: extraction + the fixed
//! SDR-trained readout only.

use std::io::Write as _;
use std::path::Path;

use rayon::prelude::*;
use zensim::feature_v2::{HdrEncoding, V2NewFeatureToggles, V2Scratch};
use zensim::source::{AlphaMode, ImageSource, PixelFormat};
use zensim::{Zensim, ZensimProfile};

/// PQ code-value source over interleaved RGBA f32 (code values in [0,1];
/// the route's `LinearF32Rgba` + `HdrEncoding::Pq` contract).
struct CodeImage {
    data: Vec<[f32; 4]>,
    w: usize,
    h: usize,
}

impl ImageSource for CodeImage {
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

// SMPTE ST 2084 constants (exact rationals), f64.
const PQ_M1: f64 = 2610.0 / 16384.0;
const PQ_M2: f64 = 2523.0 / 4096.0 * 128.0;
const PQ_C1: f64 = 3424.0 / 4096.0;
const PQ_C2: f64 = 2413.0 / 4096.0 * 32.0;
const PQ_C3: f64 = 2392.0 / 4096.0 * 32.0;

/// ST 2084 EOTF: code value [0,1] → luminance cd/m² (spec peak 10 000,
/// no display clamp — the dimming shader operates pre-display).
fn pq_eotf(v: f64) -> f64 {
    let v = v.clamp(0.0, 1.0);
    let p = v.powf(1.0 / PQ_M2);
    let num = (p - PQ_C1).max(0.0);
    let den = PQ_C2 - PQ_C3 * p;
    10000.0 * (num / den).powf(1.0 / PQ_M1)
}

/// ST 2084 inverse EOTF: luminance cd/m² → code value [0,1].
fn pq_oetf(nits: f64) -> f64 {
    let y = (nits / 10000.0).max(0.0);
    let ym = y.powf(PQ_M1);
    ((PQ_C1 + PQ_C2 * ym) / (1.0 + PQ_C3 * ym)).powf(PQ_M2)
}

/// The experiment's dimming shader on one code value: linear ÷ `factor`.
fn dim_code(v: f64, factor: f64) -> f64 {
    pq_oetf(pq_eotf(v) / factor)
}

/// Load an rgb48 PNG of full-range PQ code values; optionally apply the
/// dimming shader per channel.
fn load_code_png(path: &Path, dim: bool) -> Result<CodeImage, String> {
    let img = image::open(path)
        .map_err(|e| format!("{path:?}: {e}"))?
        .to_rgb16();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let raw = img.into_raw();
    const INV: f64 = 1.0 / 65535.0;
    let data = (0..w * h)
        .map(|i| {
            let mut px = [0f32; 4];
            for c in 0..3 {
                let v = raw[3 * i + c] as f64 * INV;
                let v = if dim { dim_code(v, 8.0) } else { v };
                px[c] = v as f32;
            }
            px[3] = 1.0;
            px
        })
        .collect();
    Ok(CodeImage { data, w, h })
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

struct Row {
    key: String,
    dim: bool,
    peak_nits: f32,
    ref_path: String,
    dist_path: String,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let manifest = arg(&args, "--manifest").expect("--manifest TSV required");
    let out = arg(&args, "--out").unwrap_or_else(|| "hdrvdc_features.csv".into());

    let rows: Vec<Row> = std::fs::read_to_string(&manifest)
        .expect("read manifest")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert_eq!(
                f.len(),
                5,
                "manifest row: key\\tdim\\tpeak_nits\\tref\\tdist ({l})"
            );
            Row {
                key: f[0].to_string(),
                dim: match f[1] {
                    "0" => false,
                    "1" => true,
                    other => panic!("dim must be 0|1, got {other}"),
                },
                peak_nits: f[2].parse().expect("peak_nits"),
                ref_path: f[3].to_string(),
                dist_path: f[4].to_string(),
            }
        })
        .collect();
    eprintln!("{} manifest rows", rows.len());

    let weights = zensim::WEIGHTS;
    let mut indexed: Vec<(usize, String, usize)> = rows
        .par_iter()
        .enumerate()
        .map_init(V2Scratch::new, |scratch, (i, row)| {
            let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
            let ref_img = match load_code_png(Path::new(&row.ref_path), row.dim) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("REF FAIL {}: {e}", row.key);
                    return None;
                }
            };
            let dist_img = match load_code_png(Path::new(&row.dist_path), row.dim) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("DIST FAIL {}: {e}", row.key);
                    return None;
                }
            };
            if (ref_img.w, ref_img.h) != (dist_img.w, dist_img.h) {
                eprintln!(
                    "SKIP dim mismatch {}: {}x{} vs {}x{}",
                    row.key, ref_img.w, ref_img.h, dist_img.w, dist_img.h
                );
                return None;
            }
            let res = z.compute_folded720_append2_features_hdr(
                &ref_img,
                &dist_img,
                HdrEncoding::Pq {
                    peak_nits: row.peak_nits,
                },
                V2NewFeatureToggles::default(),
                scratch,
            );
            let res = match res {
                Ok(res) => res,
                Err(e) => {
                    eprintln!("SCORE FAIL {}: {e:?}", row.key);
                    return None;
                }
            };
            let feats = res.features();
            let (score, _) =
                zensim::try_score_from_features(&feats[..228], weights).expect("readout");
            let mut line = format!(
                "{},{},{},{score}",
                row.key,
                if row.dim { 1 } else { 0 },
                row.peak_nits
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
    assert_eq!(n_feat, 944, "feature width (mode 944)");
    let f = std::fs::File::create(&out).expect("create out");
    let mut w = std::io::BufWriter::new(f);
    let mut header = String::from("key,dim,peak_nits,score228");
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
