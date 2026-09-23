//! `e5a_render_score` — per-item scorer for the E5A rendering-regression lane.
//!
//! Reads the same pairs TSV every arm uses (`ref_path`, `dist_path` plus
//! passthrough keys) and emits the arms that have no existing owner:
//!
//! | column | definition |
//! |---|---|
//! | `maxabs` | max over pixels×channels of \|Δu8\| |
//! | `psnr` | PSNR (dB) over all RGB channels; `inf` when identical |
//! | `t1_lin_max` | max of the linear-light max-channel error map |
//! | `t2_lin_q999` | 99.9th percentile of that map (linear interp) |
//! | `t3_lin_q99` | 99th percentile |
//! | `t4_enc_max` | max of the u8 error map (== maxabs) |
//! | `t5_enc_q999` | 99.9th percentile of the u8 map |
//! | `gmsd` | zenmetrics `gmsd` GMSD score |
//! | `zensim_b` | `Zensim::compute_with_diffmap` profile codec-target (B) |
//!
//! With `--maps <dir>` per-item maps are written as little-endian f32:
//! `<key>__u8.f32` (per-pixel max-channel |Δ| u8 — the `maxabs`/changed-mask
//! map), `<key>__lin.f32` (same map in linear light — the `testlin` candidate
//! map), `<key>__gmsd.f32` (GMS map, `map_dims` geometry), `<key>__zensim_b.f32`
//! (zensim-B diffmap). sha256 of each is recorded in the TSV.
//!
//! Decode goes through `zenpng` + `zenpixels-convert`, the same owner the
//! generator uses.
//!
//! ```sh
//! cargo run --release -p zensim-bench --example e5a_render_score -- \
//!   --pairs pairs.tsv --output e5a_scores.tsv --maps maps/ [--threads N]
//! ```

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use enough::Unstoppable;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use zenpixels_convert::PixelBufferConvertTypedExt;
use zensim::{DiffmapWeighting, RgbSlice, Zensim, ZensimProfile};

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Decode a PNG to packed RGB8 via the imazen owners (same path as
/// `m3_fixture_gen::read_png_rgb8`).
fn read_png_rgb8(path: &Path) -> Result<(Vec<u8>, u32, u32), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let out = zenpng::decode(&bytes, &zenpng::PngDecodeConfig::default(), &Unstoppable)
        .map_err(|e| format!("zenpng decode {path:?}: {e:?}"))?;
    let (w, h) = (out.info.width, out.info.height);
    let rgb = out.pixels.to_rgb8();
    let view = rgb.as_imgref();
    let mut px = Vec::with_capacity(w as usize * h as usize * 3);
    for row in view.rows() {
        for p in row {
            px.extend_from_slice(&[p.r, p.g, p.b]);
        }
    }
    if px.len() != w as usize * h as usize * 3 {
        return Err(format!("{path:?}: decoded {} bytes != {w}x{h}x3", px.len()));
    }
    Ok((px, w, h))
}

/// Per-pixel max-channel |Δ| maps in u8 and linear-light f32 domains.
fn error_maps(a: &[u8], b: &[u8], w: usize, h: usize) -> (Vec<u8>, Vec<f32>) {
    let n = w * h;
    let mut e8 = vec![0u8; n];
    let mut el = vec![0f32; n];
    for (i, (pa, pb)) in a
        .as_chunks::<3>()
        .0
        .iter()
        .zip(b.as_chunks::<3>().0)
        .enumerate()
    {
        let mut m8 = 0u8;
        let mut ml = 0f32;
        for c in 0..3 {
            let d = (pa[c] as i32 - pb[c] as i32).unsigned_abs() as u8;
            m8 = m8.max(d);
            let la = linear_srgb::precise::srgb_to_linear_f64(pa[c] as f64 / 255.0) as f32;
            let lb = linear_srgb::precise::srgb_to_linear_f64(pb[c] as f64 / 255.0) as f32;
            ml = ml.max((la - lb).abs());
        }
        e8[i] = m8;
        el[i] = ml;
    }
    (e8, el)
}

/// Empirical quantile of `map` at q with linear interpolation (numpy
/// `linear`/`inclusive` convention on the sorted sample).
fn quantile_sorted(map: &[f64], q: f64) -> f64 {
    if map.is_empty() {
        return f64::NAN;
    }
    let pos = q * (map.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(map.len() - 1);
    map[lo] + (map[hi] - map[lo]) * (pos - lo as f64)
}

fn map_quantile(map_u8: &[u8], q: f64) -> f64 {
    let mut v: Vec<f64> = map_u8.iter().map(|&x| x as f64).collect();
    v.sort_by(|a, b| a.total_cmp(b));
    quantile_sorted(&v, q)
}

fn map_quantile_f32(map: &[f32], q: f64) -> f64 {
    let mut v: Vec<f64> = map.iter().map(|&x| x as f64).collect();
    v.sort_by(|a, b| a.total_cmp(b));
    quantile_sorted(&v, q)
}

struct Row {
    keys: Vec<String>,
    ref_path: String,
    dist_path: String,
    key: String,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut pairs, mut output, mut maps, mut threads) = (None, None, None, None);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--pairs" => pairs = Some(args.next().expect("--pairs VALUE").into()),
            "--output" => output = Some(args.next().expect("--output VALUE").into()),
            "--maps" => maps = Some(args.next().expect("--maps VALUE").into()),
            "--threads" => threads = Some(args.next().expect("--threads VALUE").parse().unwrap()),
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }
    let pairs: PathBuf = pairs.expect("--pairs is required");
    let output: PathBuf = output.expect("--output is required");
    if let Some(t) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .expect("rayon init");
    }
    let maps_dir: Option<PathBuf> = maps;
    if let Some(d) = &maps_dir {
        std::fs::create_dir_all(d).expect("create --maps dir");
    }

    let text = std::fs::read_to_string(&pairs).expect("read pairs TSV");
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("empty pairs TSV").split('\t').collect();
    let find = |n: &str| {
        header
            .iter()
            .position(|c| *c == n)
            .unwrap_or_else(|| panic!("missing '{n}' column in {header:?}"))
    };
    let ref_idx = find("ref_path");
    let dist_idx = find("dist_path");
    let key_idx: Vec<usize> = (0..header.len())
        .filter(|i| *i != ref_idx && *i != dist_idx)
        .collect();
    let key_col = header.iter().position(|c| *c == "key");

    let rows: Vec<Row> = lines
        .map(|ln| {
            let p: Vec<&str> = ln.split('\t').collect();
            Row {
                keys: key_idx.iter().map(|k| p[*k].to_string()).collect(),
                ref_path: p[ref_idx].to_string(),
                dist_path: p[dist_idx].to_string(),
                key: key_col.map(|i| p[i].to_string()).unwrap_or_default(),
            }
        })
        .collect();
    let n_total = rows.len();
    eprintln!("e5a_render_score: {n_total} pairs");

    let writer = Mutex::new(std::fs::File::create(&output).expect("create output"));
    {
        let mut w = writer.lock().unwrap();
        let mut cols: Vec<String> = key_idx.iter().map(|k| header[*k].to_string()).collect();
        cols.extend(
            [
                "maxabs",
                "psnr",
                "t1_lin_max",
                "t2_lin_q999",
                "t3_lin_q99",
                "t4_enc_max",
                "t5_enc_q999",
                "gmsd",
                "gmsd_mean",
                "zensim_b",
                "anchor_lin_mean",
                "u8_map_sha256",
                "lin_map_sha256",
                "gmsd_map_sha256",
                "zensim_b_map_sha256",
            ]
            .iter()
            .map(|s| (*s).to_string()),
        );
        writeln!(w, "{}", cols.join("\t")).unwrap();
    }

    let zensim_b = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
    let started = std::time::Instant::now();
    let progress = AtomicUsize::new(0);
    let failures = AtomicUsize::new(0);
    let maps_dir = &maps_dir;
    rows.par_iter().for_each(|row| {
        let p = progress.fetch_add(1, Ordering::Relaxed) + 1;
        if p.is_multiple_of(200) || p == n_total {
            let elapsed = started.elapsed().as_secs_f64();
            let rate = p as f64 / elapsed;
            eprintln!(
                "  {p}/{n_total} ({rate:.1}/s, ETA {:.0}s)",
                (n_total - p) as f64 / rate
            );
        }
        let (a, w, h) = match read_png_rgb8(Path::new(&row.ref_path)) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("DECODE FAIL ref {}: {e}", row.ref_path);
                failures.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };
        let (b, w2, h2) = match read_png_rgb8(Path::new(&row.dist_path)) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("DECODE FAIL dist {}: {e}", row.dist_path);
                failures.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };
        if (w, h) != (w2, h2) {
            eprintln!("DIMS DIFFER {} vs {}", row.ref_path, row.dist_path);
            failures.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let (wu, hu) = (w as usize, h as usize);
        let (e8, el) = error_maps(&a, &b, wu, hu);
        let maxabs = e8.iter().copied().max().unwrap_or(0) as f64;
        let mse = a
            .iter()
            .zip(&b)
            .map(|(x, y)| {
                let d = *x as f64 - *y as f64;
                d * d
            })
            .sum::<f64>()
            / a.len() as f64;
        let psnr = if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0 * 255.0 / mse).log10()
        };
        let t1 = el.iter().copied().fold(0f32, f32::max) as f64;
        let t2 = map_quantile_f32(&el, 0.999);
        let t3 = map_quantile_f32(&el, 0.99);
        let t4 = maxabs;
        let t5 = map_quantile(&e8, 0.999);
        let anchor = el.iter().map(|&v| v as f64).sum::<f64>() / el.len() as f64;
        let dump = |suffix: &str, map: &[f32]| -> String {
            match maps_dir {
                Some(d) => {
                    let bytes: Vec<u8> = map.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let s = sha(&bytes);
                    std::fs::write(d.join(format!("{}{}", row.key, suffix)), &bytes)
                        .expect("write map");
                    s
                }
                None => String::new(),
            }
        };
        let u8_sha = dump(
            "__u8.f32",
            &e8.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        );
        let lin_sha = dump("__lin.f32", &el);
        // GMSD + GMS map (map_dims = (w/2)*(h/2), per crate contract).
        let (gmw, gmh) = gmsd::map_dims(wu, hu);
        let mut gmap = vec![0f32; gmw * gmh];
        let mut gr = vec![0f32; wu * hu];
        let mut gd = vec![0f32; wu * hu];
        let gscore = gmsd::rgb8_to_gray(&a, wu, hu, wu * 3, &mut gr)
            .and_then(|_| gmsd::rgb8_to_gray(&b, wu, hu, wu * 3, &mut gd))
            .ok()
            .and_then(|_| {
                let r = gmsd::GrayImage::packed(&gr, wu, hu).ok()?;
                let d = gmsd::GrayImage::packed(&gd, wu, hu).ok()?;
                gmsd::gmsd_with_map(r, d, &mut gmap).ok()
            });
        let (gmsd_v, gmsd_mean_v) = gscore
            .map(|s| (s.gmsd, s.mean_gms))
            .unwrap_or((f64::NAN, f64::NAN));
        let gmsd_map_sha = dump("__gmsd.f32", &gmap);
        // zensim B + diffmap.
        let rs = RgbSlice::new(a.as_chunks::<3>().0, wu, hu);
        let ds = RgbSlice::new(b.as_chunks::<3>().0, wu, hu);
        let (zb, zb_map_sha) = match zensim_b.compute_with_diffmap(
            &rs,
            &ds,
            DiffmapWeighting::default(),
        ) {
            Ok(r) => (r.score(), dump("__zensim_b.f32", r.diffmap())),
            Err(e) => {
                eprintln!("ZENSIM_B FAIL {}: {e:?}", row.dist_path);
                failures.fetch_add(1, Ordering::Relaxed);
                (f64::NAN, String::new())
            }
        };
        let mut w = writer.lock().unwrap();
        writeln!(
            w,
            "{}\t{maxabs:.17e}\t{psnr:.17e}\t{t1:.17e}\t{t2:.17e}\t{t3:.17e}\t{t4:.17e}\t{t5:.17e}\t{gmsd_v:.17e}\t{gmsd_mean_v:.17e}\t{zb:.17e}\t{anchor:.17e}\t{u8_sha}\t{lin_sha}\t{gmsd_map_sha}\t{zb_map_sha}",
            row.keys.join("\t")
        )
        .unwrap();
    });
    let f = failures.load(Ordering::Relaxed);
    eprintln!("e5a_render_score: done in {:.1}s, {f} failures", started.elapsed().as_secs_f64());
    if f > 0 {
        std::process::exit(1);
    }
}
