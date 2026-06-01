//! Score the UPIQ HDR subset with zensim's PU21 HDR front-end and emit a CSV
//! the `scripts/upiq_eval.py` harness can validate against the JOD truth.
//!
//! UPIQ EXRs are stored in absolute photometric units (cd/m²), so they feed
//! `Zensim::compute_pu_linear_planar` directly — no display model / PQ decode
//! needed. Output: `condition_id,zensim_hdr` per scored pair.
//!
//! Usage:
//!   upiq_pu_score \
//!     --images /mnt/v/datasets/upiq_extracted/upiq_dataset/images \
//!     --subjective /mnt/v/datasets/upiq/upiq_subjective_scores.csv \
//!     --out /tmp/zensim_hdr_scores.csv \
//!     [--corpus narwaria,korshunov] [--hdr-only]

use std::collections::HashMap;
use std::path::Path;

use zensim::{Zensim, ZensimProfile};

struct Rgb {
    w: usize,
    h: usize,
    r: Vec<f32>,
    g: Vec<f32>,
    b: Vec<f32>,
}

/// Load an absolute-luminance EXR into planar f32 RGB (values preserved, HDR
/// magnitudes > 1 kept). Uses the `image` crate's EXR decoder.
fn load_exr_rgb(path: &Path) -> Result<Rgb, String> {
    let rgb = image::open(path)
        .map_err(|e| format!("{path:?}: {e}"))?
        .to_rgb32f();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let raw = rgb.into_raw(); // interleaved [r,g,b, …], len = w*h*3
    let n = w * h;
    let mut r = vec![0.0f32; n];
    let mut g = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    for i in 0..n {
        r[i] = raw[3 * i];
        g[i] = raw[3 * i + 1];
        b[i] = raw[3 * i + 2];
    }
    Ok(Rgb { w, h, r, g, b })
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let images = arg(&args, "--images")
        .unwrap_or_else(|| "/mnt/v/datasets/upiq_extracted/upiq_dataset/images".into());
    let subjective = arg(&args, "--subjective")
        .unwrap_or_else(|| "/mnt/v/datasets/upiq/upiq_subjective_scores.csv".into());
    let out = arg(&args, "--out").unwrap_or_else(|| "/tmp/zensim_hdr_scores.csv".into());
    let corpora: Vec<String> = arg(&args, "--corpus")
        .map(|s| s.split(',').map(str::to_string).collect())
        .unwrap_or_else(|| vec!["narwaria".into(), "korshunov".into()]);
    let hdr_only = args.iter().any(|a| a == "--hdr-only") || true; // default HDR subset

    // Score each pair under several profiles in one EXR-decode pass (decode is
    // the bottleneck). PreviewV0_2 = linear cube-root-tuned weights;
    // A = 372-feature MLP. Lets us see which feature aggregation ranks PU-XYB
    // best without re-decoding.
    let profiles: [(&str, Zensim); 3] = [
        ("zensim_v02", Zensim::new(ZensimProfile::PreviewV0_2)),
        ("zensim_v01", Zensim::new(ZensimProfile::PreviewV0_1)),
        ("zensim_a", Zensim::new(ZensimProfile::A)),
    ];

    let mut rdr = csv::Reader::from_path(&subjective).expect("open subjective csv");
    let headers = rdr.headers().expect("headers").clone();
    let col = |name: &str| headers.iter().position(|h| h == name).expect(name);
    let (c_cid, c_ds, c_hdr, c_test, c_ref) = (
        col("condition_id"),
        col("dataset"),
        col("is_hdr"),
        col("test_file"),
        col("reference_file"),
    );

    // Cache reference images (one ref → many distorted).
    let mut ref_cache: HashMap<String, Rgb> = HashMap::new();
    let mut rows: Vec<(String, Vec<f64>)> = Vec::new();
    let (mut ok, mut skip, mut err) = (0usize, 0usize, 0usize);

    for rec in rdr.records() {
        let rec = rec.expect("record");
        let dataset = &rec[c_ds];
        let is_hdr = &rec[c_hdr] == "1";
        if !corpora.iter().any(|c| c == dataset) {
            continue;
        }
        if hdr_only && !is_hdr {
            continue;
        }
        let cid = rec[c_cid].to_string();
        let test_path = Path::new(&images).join(&rec[c_test]);
        let ref_rel = rec[c_ref].to_string();
        let ref_path = Path::new(&images).join(&ref_rel);

        if !ref_cache.contains_key(&ref_rel) {
            match load_exr_rgb(&ref_path) {
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
        let d = match load_exr_rgb(&test_path) {
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
        let mut scores = Vec::with_capacity(profiles.len());
        let mut failed = false;
        for (_, z) in &profiles {
            match z.compute_pu_linear_planar([&r.r, &r.g, &r.b], [&d.r, &d.g, &d.b], r.w, r.h, r.w) {
                Ok(res) => scores.push(res.score()),
                Err(e) => {
                    eprintln!("SCORE FAIL {cid}: {e:?}");
                    failed = true;
                    break;
                }
            }
        }
        if failed {
            err += 1;
            continue;
        }
        rows.push((cid, scores));
        ok += 1;
        if ok % 50 == 0 {
            eprintln!("scored {ok} pairs…");
        }
    }

    let mut w = String::from("condition_id");
    for (name, _) in &profiles {
        w.push(',');
        w.push_str(name);
    }
    w.push('\n');
    for (cid, scores) in &rows {
        w.push_str(cid);
        for s in scores {
            w.push_str(&format!(",{s}"));
        }
        w.push('\n');
    }
    std::fs::write(&out, w).expect("write out");
    eprintln!("done: {ok} scored, {skip} dim-skipped, {err} errored → {out}");
}
