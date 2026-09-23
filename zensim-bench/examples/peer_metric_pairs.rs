//! `peer_metric_pairs` — score a pairs TSV with OUR implementations of the
//! peer metrics, so a third party's published column can be compared against
//! the number our crate actually returns.
//!
//! Written for the JPEG AIC2026 read (2026-09-19), whose shipped score table
//! carries a `SSIMULACRA2` column and a `proposal-Butteraugli` column. Those
//! are other people's runs of those methods; this binary produces ours on the
//! same pixels, and the difference is a fact about implementations, decoding
//! and versions — not about either method being wrong.
//!
//! Input is the SAME TSV `score_pairs_tuner` reads (header must contain
//! `ref_path` and `dist_path`; every other column is carried through), so one
//! pair list drives both the zensim roster and the peer roster.
//!
//! Output TSV columns: the passthrough keys, then
//! `ssim2` (`fast_ssim2::compute_ssimulacra2`) and the butteraugli family from
//! a single `butteraugli::butteraugli` call with the diffmap on —
//! `butter_max` (the crate's own `score`), plus the libjxl-style averaged
//! p-norms at p = 1, 2, 3 (`ComputeDistanceP`, `lib/extras/metrics.cc`).
//! Emitting all four is deliberate: which norm a published "Butteraugli"
//! column used is not stated, so it is identified by agreement rather than
//! assumed.
//!
//! Decoding goes through `shared/zen_decode.rs`, the crate's decode owner —
//! imazen codecs only, and a decode failure is a loud error, never a dropped
//! row.
//!
//! ```sh
//! cargo run --release -p zensim-bench --example peer_metric_pairs -- \
//!   --pairs pairs.tsv --output peer_scores.tsv [--threads N]
//! ```

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use butteraugli::ButteraugliParams;
use imgref::Img;
use rayon::prelude::*;
use rgb::RGB8;
use sha2::{Digest, Sha256};

#[path = "shared/zen_decode.rs"]
mod zen_decode;

/// libjxl-style p-norm averaging at p, 2p, 4p. Mirrors
/// `gen_butteraugli_3norm.rs`'s owner of the same formula
/// (libjxl `lib/extras/metrics.cc` `ComputeDistanceP`).
fn libjxl_pnorm(diffmap: &[f32], p: f64) -> f64 {
    if diffmap.is_empty() {
        return f64::NAN;
    }
    let mut sum1 = [0.0_f64; 3];
    for &v in diffmap {
        let d = v as f64;
        let mut acc = d.powf(p);
        sum1[0] += acc;
        acc *= acc;
        sum1[1] += acc;
        acc *= acc;
        sum1[2] += acc;
    }
    let one_per_pixels = 1.0 / diffmap.len() as f64;
    let mut v = 0.0_f64;
    for (i, &s) in sum1.iter().enumerate() {
        let exponent = 1.0 / (p * (1u32 << i) as f64);
        v += (one_per_pixels * s).powf(exponent);
    }
    v / 3.0
}

struct Row {
    keys: Vec<String>,
    ref_path: String,
    dist_path: String,
    /// `key` column value (or row index) — used to name `--diffmaps` files.
    key: String,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut pairs: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut threads: Option<usize> = None;
    // E5A: optional per-item butteraugli diffmap dump (raw LE f32, named
    // `<key>__butter.f32` when the row carries a `key` column, else row index).
    let mut diffmaps: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--pairs" => pairs = Some(args.next().expect("--pairs VALUE").into()),
            "--output" => output = Some(args.next().expect("--output VALUE").into()),
            "--threads" => threads = Some(args.next().expect("--threads VALUE").parse().unwrap()),
            "--diffmaps" => diffmaps = Some(args.next().expect("--diffmaps VALUE").into()),
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let pairs = pairs.expect("--pairs is required");
    let output = output.expect("--output is required");
    if let Some(d) = &diffmaps {
        std::fs::create_dir_all(d).expect("create --diffmaps dir");
    }
    if let Some(t) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .expect("rayon init");
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
        .enumerate()
        .map(|(i, ln)| {
            let p: Vec<&str> = ln.split('\t').collect();
            Row {
                keys: key_idx.iter().map(|k| p[*k].to_string()).collect(),
                ref_path: p[ref_idx].to_string(),
                dist_path: p[dist_idx].to_string(),
                key: key_col
                    .map(|k| p[k].to_string())
                    .filter(|k| !k.is_empty())
                    .unwrap_or_else(|| format!("row{i:04}")),
            }
        })
        .collect();
    let n_total = rows.len();
    eprintln!("peer_metric_pairs: {n_total} pairs");

    let writer = Mutex::new(std::fs::File::create(&output).expect("create output"));
    {
        let mut w = writer.lock().unwrap();
        let mut cols: Vec<String> = key_idx.iter().map(|k| header[*k].to_string()).collect();
        cols.extend(
            [
                "ssim2",
                "butter_max",
                "butter_p1",
                "butter_p2",
                "butter_p3",
                "butter_map_sha256",
            ]
            .iter()
            .map(|s| (*s).to_string()),
        );
        writeln!(w, "{}", cols.join("\t")).unwrap();
    }

    let started = std::time::Instant::now();
    let progress = AtomicUsize::new(0);
    let failures = AtomicUsize::new(0);
    rows.par_iter().for_each(|row| {
        let p = progress.fetch_add(1, Ordering::Relaxed) + 1;
        if p.is_multiple_of(500) || p == n_total {
            let elapsed = started.elapsed().as_secs_f64();
            let rate = p as f64 / elapsed;
            eprintln!(
                "  {p}/{n_total} ({rate:.1}/s, ETA {:.0}s)",
                (n_total - p) as f64 / rate
            );
        }
        let s = match zen_decode::decode_rgb8_path(Path::new(&row.ref_path)) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("DECODE FAIL ref {}: {e}", row.ref_path);
                failures.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };
        let d = match zen_decode::decode_rgb8_path(Path::new(&row.dist_path)) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("DECODE FAIL dist {}: {e}", row.dist_path);
                failures.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };
        if s.width != d.width || s.height != d.height {
            eprintln!(
                "DIM MISMATCH {} ({}x{}) vs {} ({}x{})",
                row.ref_path, s.width, s.height, row.dist_path, d.width, d.height
            );
            failures.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let (w_us, h_us) = (s.width as usize, s.height as usize);
        let src: &[[u8; 3]] = s.pixels.as_chunks::<3>().0;
        let dst: &[[u8; 3]] = d.pixels.as_chunks::<3>().0;
        let ssim2 =
            fast_ssim2::compute_ssimulacra2(Img::new(src, w_us, h_us), Img::new(dst, w_us, h_us))
                .unwrap_or(f64::NAN);
        let src_rgb8: &[RGB8] = bytemuck::cast_slice(src);
        let dst_rgb8: &[RGB8] = bytemuck::cast_slice(dst);
        let bp = ButteraugliParams::default().with_compute_diffmap(true);
        let (bmax, p1, p2, p3, bmap_sha) = match butteraugli::butteraugli(
            Img::new(src_rgb8, w_us, h_us),
            Img::new(dst_rgb8, w_us, h_us),
            &bp,
        ) {
            Ok(b) => {
                let dm = b.diffmap.as_ref().map(|m| m.buf().to_vec());
                let mut sha = String::new();
                if let (Some(dir), Some(map)) = (&diffmaps, &dm) {
                    let bytes: Vec<u8> = map.iter().flat_map(|v| v.to_le_bytes()).collect();
                    sha = format!("{:x}", Sha256::digest(&bytes));
                    std::fs::write(dir.join(format!("{}__butter.f32", row.key)), &bytes)
                        .expect("write butter diffmap");
                }
                let f = |p: f64| dm.as_ref().map_or(f64::NAN, |m| libjxl_pnorm(m, p));
                (b.score, f(1.0), f(2.0), f(3.0), sha)
            }
            Err(e) => {
                eprintln!("BUTTERAUGLI FAIL {}: {e:?}", row.dist_path);
                failures.fetch_add(1, Ordering::Relaxed);
                (f64::NAN, f64::NAN, f64::NAN, f64::NAN, String::new())
            }
        };
        let mut w = writer.lock().unwrap();
        writeln!(
            w,
            "{}\t{ssim2:.17e}\t{bmax:.17e}\t{p1:.17e}\t{p2:.17e}\t{p3:.17e}\t{bmap_sha}",
            row.keys.join("\t")
        )
        .unwrap();
    });

    eprintln!(
        "wrote {:?} in {:.1}s ({} failures)",
        output,
        started.elapsed().as_secs_f64(),
        failures.load(Ordering::Relaxed)
    );
}
