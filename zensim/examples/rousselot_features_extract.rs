// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! Rousselot HDdtb/4Kdtb full-vector feature extractor for the chroma
//! blind-spot validation study
//! (`benchmarks/rousselot_chroma_validation_2026-07-29.md`).
//!
//! Extracts the 944 feature vector for Radiance-.hdr reference/distorted
//! pairs under the study's registered display model
//! (`/mnt/v/output/zensim/rousselot-chroma-2026-07-29/PROTOCOL.md`):
//!
//! - `nits = raw × scale` per channel, `scale` = 179 (the Radiance/pfstools
//!   luminous-efficacy convention; the compressed files of both sets
//!   ceiling at raw 55.75 = RGBE-quantized 10000/179, the 10-bit PQ
//!   container maximum). `--nits-scale` overrides for the registered K=100
//!   robustness leg.
//! - clamp to [0, 1000] cd/m² (BVM-X300 measured peak; the papers' own
//!   metric protocol crops to this range).
//! - BT.2020 primaries fed as-is (route contract "primaries taken as-is";
//!   registered deviation with rationale in PROTOCOL.md).
//! - Route: declared-HDR streaming
//!   (`compute_folded720_append2_features_hdr`, `HdrEncoding::Linear`,
//!   csfw OFF — mode 944), default toggles otherwise. Same call shape as
//!   `sihdr_features_extract`.
//!
//! Manifest TSV (no header): `id <TAB> ref_path <TAB> dist_path`.
//! Output CSV: `id,nonfinite_ref,nonfinite_dist,clampfrac_ref,
//! clampfrac_dist,score228,f0..f943`. Dimension mismatches are drops with
//! reasons on stderr — never silently substituted.
//!
//! Nothing here is fitted on anything: extraction + the fixed SDR-trained
//! readout only.

use std::io::Write as _;
use std::path::Path;

use rayon::prelude::*;
use zensim::feature_v2::{HdrEncoding, V2NewFeatureToggles, V2Scratch};
use zensim::source::{AlphaMode, ImageSource, PixelFormat};
use zensim::{Zensim, ZensimProfile};

/// Absolute-linear cd/m² source over interleaved RGBA f32 (the
/// `sihdr_features_extract` `NitsImage`, verbatim).
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

/// Raw linear Radiance pixels (non-finite → 0, counted) — pre-display-map.
struct RawHdr {
    rgb: Vec<f32>,
    w: usize,
    h: usize,
    nonfinite: usize,
}

fn load_hdr_raw(path: &Path) -> Result<RawHdr, String> {
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
    Ok(RawHdr {
        rgb,
        w,
        h,
        nonfinite,
    })
}

/// The registered display map: `clamp(raw * nits_scale, 0, 1000)` per
/// channel. Returns the image and the fraction of channel-values that hit
/// the 1000 cd/m² upper clamp (diagnostic).
fn to_display(raw: &RawHdr, nits_scale: f32) -> (NitsImage, f64) {
    let n = raw.w * raw.h;
    let mut clamped = 0usize;
    let data = (0..n)
        .map(|i| {
            let mut m = |v: f32| {
                let nits = v * nits_scale;
                if nits >= 1000.0 {
                    clamped += 1;
                    1000.0
                } else if nits < 0.0 {
                    0.0
                } else {
                    nits
                }
            };
            [
                m(raw.rgb[3 * i]),
                m(raw.rgb[3 * i + 1]),
                m(raw.rgb[3 * i + 2]),
                1.0,
            ]
        })
        .collect();
    let frac = clamped as f64 / (3 * n) as f64;
    (
        NitsImage {
            data,
            w: raw.w,
            h: raw.h,
        },
        frac,
    )
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

struct Row {
    id: String,
    ref_path: String,
    dist_path: String,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let manifest = arg(&args, "--manifest").expect("--manifest TSV required");
    let out = arg(&args, "--out").unwrap_or_else(|| "rousselot_features.csv".into());
    let nits_scale: f32 = arg(&args, "--nits-scale")
        .map(|s| s.parse().expect("--nits-scale f32"))
        .unwrap_or(179.0);

    let rows: Vec<Row> = std::fs::read_to_string(&manifest)
        .expect("read manifest")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert_eq!(f.len(), 3, "manifest row: id\\tref\\tdist ({l})");
            Row {
                id: f[0].to_string(),
                ref_path: f[1].to_string(),
                dist_path: f[2].to_string(),
            }
        })
        .collect();
    eprintln!("{} pairs (nits_scale={nits_scale})", rows.len());

    let weights = zensim::WEIGHTS;
    let mut indexed: Vec<(usize, String, usize)> = rows
        .par_iter()
        .enumerate()
        .map_init(V2Scratch::new, |scratch, (i, row)| {
            let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
            let r = match load_hdr_raw(Path::new(&row.ref_path)) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("REF FAIL {}: {e}", row.id);
                    return None;
                }
            };
            let d = match load_hdr_raw(Path::new(&row.dist_path)) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("DIST FAIL {}: {e}", row.id);
                    return None;
                }
            };
            if (r.w, r.h) != (d.w, d.h) {
                eprintln!(
                    "SKIP dim mismatch {}: {}x{} vs {}x{}",
                    row.id, r.w, r.h, d.w, d.h
                );
                return None;
            }
            let (nf_r, nf_d) = (r.nonfinite, d.nonfinite);
            let (ref_img, clamp_r) = to_display(&r, nits_scale);
            let (dist_img, clamp_d) = to_display(&d, nits_scale);
            drop((r, d));
            let res = z.compute_folded720_append2_features_hdr(
                &ref_img,
                &dist_img,
                HdrEncoding::Linear,
                V2NewFeatureToggles {
                    csfw_block: false,
                    ..V2NewFeatureToggles::default()
                },
                scratch,
            );
            let res = match res {
                Ok(res) => res,
                Err(e) => {
                    eprintln!("SCORE FAIL {}: {e:?}", row.id);
                    return None;
                }
            };
            let feats = res.features();
            let (score, _) =
                zensim::try_score_from_features(&feats[..228], weights).expect("readout");
            let mut line = format!("{},{nf_r},{nf_d},{clamp_r},{clamp_d},{score}", row.id);
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
    assert_eq!(n_feat, 944, "feature width must be 944 (csfw OFF)");
    let f = std::fs::File::create(&out).expect("create out");
    let mut w = std::io::BufWriter::new(f);
    let mut header =
        String::from("id,nonfinite_ref,nonfinite_dist,clampfrac_ref,clampfrac_dist,score228");
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
