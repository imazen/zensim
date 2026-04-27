//! Real-codec validation of the scale-0 proxy signal for streaming AQ control
//! (zenjpeg #113 / zensim #16, Option B).
//!
//! Companion to `scale0_correlation.rs`, which used synthetic block-quant
//! distortion. This file replays *real codec output* (mozjpeg, zenjpeg sRGB,
//! zenjpeg XYB, zenwebp, zenavif, zenjxl) from the safe-synthetic training
//! ledger and asks the same question:
//!
//!   Does `dot(features[0..39], WEIGHTS_PREVIEW_V0_1[0..39])` (i.e. scale-0
//!   only, basic features only) track the canonical multi-scale raw distance
//!   well enough to drive a per-strip PI controller?
//!
//! The blocking concern is that real codecs produce structured chroma noise,
//! ringing, and (for VarDCT codecs) frequency-domain quant patterns that
//! synthetic block-quant doesn't reproduce. If chroma-aggressive codecs
//! (zenwebp lossy, zenjpeg-XYB) decorrelate the proxy from canonical, the
//! fixed-shape controller breaks.
//!
//! Methodology:
//!   - Read the ledger CSV at /mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv.
//!   - Stratify by (codec, quality-band) where bands are q<=30, 30<q<=60, 60<q<=90, q>90.
//!   - Within each cell, pick ~10 distinct source images and 4 quality levels each.
//!   - Decode source PNG + decoded PNG, run `Zensim::compute`, record canonical
//!     raw_distance and scale-0 proxy.
//!   - Report pooled SROCC + Pearson, per-codec SROCC + Pearson, and intra-image
//!     SROCC over the q-sweep.
//!
//! Cap total runtime around ~15 minutes via the SAMPLES_PER_CELL knob.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;
use zensim::{RgbaSlice, Zensim, ZensimProfile, profile::WEIGHTS_PREVIEW_V0_1};

const CSV_PATH: &str = "/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv";
const SCALE0_BASIC_LEN: usize = 13 * 3; // 39

/// Distinct quality-band labels. Each cell is (codec × band).
const Q_BANDS: &[(&str, u32, u32)] = &[
    ("low",  0,   30),
    ("mid",  31,  60),
    ("high", 61,  90),
    ("near", 91,  100),
];

/// Per-cell sampling. ~10 images × ~4 q-levels per image = ~40 pairs/cell.
/// 6 codecs × 4 bands × ~40 = ~960 pairs total.
const IMAGES_PER_CELL: usize = 10;
const QS_PER_IMAGE_IN_BAND: usize = 4;

fn scale0_proxy(features: &[f64]) -> f64 {
    let n = SCALE0_BASIC_LEN.min(features.len());
    features[..n]
        .iter()
        .zip(WEIGHTS_PREVIEW_V0_1[..n].iter())
        .map(|(f, w)| f * w)
        .sum()
}

/// Spearman rank correlation utilities (copied from synthetic example).
fn rank(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && values[idx[j + 1]] == values[idx[i]] {
            j += 1;
        }
        let avg = ((i + j) as f64) / 2.0 + 1.0;
        for k in i..=j {
            ranks[idx[k]] = avg;
        }
        i = j + 1;
    }
    ranks
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    if a.len() < 2 {
        return f64::NAN;
    }
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for i in 0..a.len() {
        let da = a[i] - ma;
        let db = b[i] - mb;
        num += da * db;
        va += da * da;
        vb += db * db;
    }
    if va == 0.0 || vb == 0.0 {
        return f64::NAN;
    }
    num / (va.sqrt() * vb.sqrt())
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let ra = rank(a);
    let rb = rank(b);
    pearson(&ra, &rb)
}

#[derive(Clone, Debug)]
struct LedgerRow {
    source_path: String,
    decoded_path: String,
    codec: String,
    quality: u32,
    #[allow(dead_code)]
    width: u32,
    #[allow(dead_code)]
    height: u32,
}

fn parse_csv(path: &str) -> Vec<LedgerRow> {
    let f = File::open(path).expect("cannot open CSV");
    let r = BufReader::new(f);
    let mut out = Vec::new();
    for (i, line) in r.lines().enumerate() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        if i == 0 {
            continue; // header
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 6 {
            continue;
        }
        let q: u32 = match parts[3].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let w: u32 = match parts[4].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let h: u32 = match parts[5].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        out.push(LedgerRow {
            source_path: parts[0].to_string(),
            decoded_path: parts[1].to_string(),
            codec: parts[2].to_string(),
            quality: q,
            width: w,
            height: h,
        });
    }
    out
}

fn band_for(q: u32) -> Option<&'static str> {
    for (label, lo, hi) in Q_BANDS {
        if q >= *lo && q <= *hi {
            return Some(*label);
        }
    }
    None
}

/// Stratified sampling: codec × band → pick IMAGES_PER_CELL distinct source images
/// (each image contributes up to QS_PER_IMAGE_IN_BAND quality levels in that band).
fn sample(rows: &[LedgerRow]) -> Vec<LedgerRow> {
    // Group rows by (codec, band) → source_path → list of LedgerRow
    let mut groups: BTreeMap<(String, &'static str), BTreeMap<String, Vec<LedgerRow>>> =
        BTreeMap::new();
    for r in rows {
        let band = match band_for(r.quality) {
            Some(b) => b,
            None => continue,
        };
        groups
            .entry((r.codec.clone(), band))
            .or_default()
            .entry(r.source_path.clone())
            .or_default()
            .push(r.clone());
    }
    // Deterministic LCG so the script reproduces exactly.
    let mut seed: u64 = 0x5A5A_C0DE_DEAD_BEEF;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };
    let mut out = Vec::new();
    for ((codec, band), per_image) in groups {
        let mut images: Vec<String> = per_image.keys().cloned().collect();
        // Shuffle deterministically.
        for i in (1..images.len()).rev() {
            let j = (next() as usize) % (i + 1);
            images.swap(i, j);
        }
        images.truncate(IMAGES_PER_CELL);
        for img in &images {
            let mut rs = per_image.get(img).cloned().unwrap_or_default();
            // Sort by quality, evenly subsample QS_PER_IMAGE_IN_BAND.
            rs.sort_by_key(|r| r.quality);
            if rs.len() > QS_PER_IMAGE_IN_BAND {
                let stride = rs.len() as f64 / QS_PER_IMAGE_IN_BAND as f64;
                let mut picked = Vec::with_capacity(QS_PER_IMAGE_IN_BAND);
                for k in 0..QS_PER_IMAGE_IN_BAND {
                    let idx = ((k as f64 + 0.5) * stride) as usize;
                    picked.push(rs[idx.min(rs.len() - 1)].clone());
                }
                // Dedup by quality
                picked.sort_by_key(|r| r.quality);
                picked.dedup_by_key(|r| r.quality);
                rs = picked;
            }
            for r in rs {
                out.push(r);
            }
        }
        eprintln!(
            "  cell ({:>30}, {:>4}): {} images sampled",
            codec,
            band,
            images.len()
        );
    }
    out
}

fn load_rgba(path: &str) -> Option<(Vec<[u8; 4]>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    let pixels = rgba.pixels().map(|p| [p[0], p[1], p[2], p[3]]).collect();
    Some((pixels, w, h))
}

#[derive(Clone, Debug)]
struct Measurement {
    codec: String,
    band: &'static str,
    source: String,
    quality: u32,
    raw_distance: f64,
    score: f64,
    proxy: f64,
}

fn main() {
    let t_total = Instant::now();
    eprintln!("scale0_correlation_real: real-codec proxy validation");
    eprintln!("CSV: {}", CSV_PATH);
    if !Path::new(CSV_PATH).exists() {
        eprintln!("ERROR: CSV not found at {}", CSV_PATH);
        std::process::exit(1);
    }
    let rows = parse_csv(CSV_PATH);
    eprintln!("  parsed {} ledger rows", rows.len());

    let sampled = sample(&rows);
    eprintln!("  sampled {} pairs after stratification", sampled.len());

    let z = Zensim::new(ZensimProfile::PreviewV0_1);

    println!("# scale0_correlation_real — measurements (header)");
    println!("# codec,band,source,quality,raw_distance,score,proxy");

    let mut meas: Vec<Measurement> = Vec::new();
    let mut errors = 0usize;
    let mut last_log = Instant::now();

    for (i, row) in sampled.iter().enumerate() {
        // Periodic progress.
        if last_log.elapsed().as_secs() >= 15 {
            eprintln!(
                "  progress: {}/{} ({:.1}s elapsed, {} errors)",
                i,
                sampled.len(),
                t_total.elapsed().as_secs_f32(),
                errors
            );
            last_log = Instant::now();
        }

        let src = match load_rgba(&row.source_path) {
            Some(v) => v,
            None => {
                errors += 1;
                continue;
            }
        };
        let dst = match load_rgba(&row.decoded_path) {
            Some(v) => v,
            None => {
                errors += 1;
                continue;
            }
        };
        if src.1 != dst.1 || src.2 != dst.2 {
            errors += 1;
            continue;
        }
        let s = RgbaSlice::new(&src.0, src.1 as usize, src.2 as usize);
        let d = RgbaSlice::new(&dst.0, dst.1 as usize, dst.2 as usize);
        let res = match z.compute(&s, &d) {
            Ok(r) => r,
            Err(_) => {
                errors += 1;
                continue;
            }
        };
        let proxy = scale0_proxy(res.features());
        let band = band_for(row.quality).unwrap_or("?");
        let m = Measurement {
            codec: row.codec.clone(),
            band,
            source: row.source_path.clone(),
            quality: row.quality,
            raw_distance: res.raw_distance(),
            score: res.score(),
            proxy,
        };
        let src_short = std::path::Path::new(&m.source)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?");
        println!(
            "{},{},{},{},{:.6e},{:.4},{:.6e}",
            m.codec, m.band, src_short, m.quality, m.raw_distance, m.score, m.proxy
        );
        meas.push(m);
    }

    eprintln!(
        "  done: {} measurements, {} errors, {:.1}s",
        meas.len(),
        errors,
        t_total.elapsed().as_secs_f32()
    );

    // ---------- Pooled correlation ----------
    println!();
    println!("# ---------- POOLED ----------");
    let raws: Vec<f64> = meas.iter().map(|m| m.raw_distance).collect();
    let proxies: Vec<f64> = meas.iter().map(|m| m.proxy).collect();
    let pooled_srocc = spearman(&raws, &proxies);
    let pooled_pearson = pearson(&raws, &proxies);
    println!(
        "# pooled SROCC = {:.4}  Pearson = {:.4}  n = {}",
        pooled_srocc,
        pooled_pearson,
        meas.len()
    );

    // ---------- Per-codec ----------
    println!();
    println!("# ---------- PER-CODEC ----------");
    println!("# codec,n,srocc,pearson,intra_image_mean_srocc,intra_image_p10_srocc,n_images_below_0.9");
    let mut codec_set: Vec<String> = meas.iter().map(|m| m.codec.clone()).collect();
    codec_set.sort();
    codec_set.dedup();
    let mut summary_rows: Vec<(String, usize, f64, f64, f64, f64, usize)> = Vec::new();
    for codec in &codec_set {
        let rs: Vec<&Measurement> = meas.iter().filter(|m| &m.codec == codec).collect();
        if rs.len() < 3 {
            continue;
        }
        let raws: Vec<f64> = rs.iter().map(|m| m.raw_distance).collect();
        let proxies: Vec<f64> = rs.iter().map(|m| m.proxy).collect();
        let s = spearman(&raws, &proxies);
        let p = pearson(&raws, &proxies);

        // Intra-image SROCC (rank q-sweep within each source).
        let mut by_src: BTreeMap<String, Vec<&Measurement>> = BTreeMap::new();
        for m in &rs {
            by_src.entry(m.source.clone()).or_default().push(*m);
        }
        let mut intra: Vec<f64> = Vec::new();
        for (_src, mut list) in by_src {
            if list.len() < 3 {
                continue;
            }
            list.sort_by_key(|m| m.quality);
            let r: Vec<f64> = list.iter().map(|m| m.raw_distance).collect();
            let p2: Vec<f64> = list.iter().map(|m| m.proxy).collect();
            let s = spearman(&r, &p2);
            if s.is_finite() {
                intra.push(s);
            }
        }
        intra.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean_intra = if intra.is_empty() {
            f64::NAN
        } else {
            intra.iter().sum::<f64>() / intra.len() as f64
        };
        let p10_intra = if intra.is_empty() {
            f64::NAN
        } else {
            let i = (intra.len() as f64 * 0.10) as usize;
            intra[i.min(intra.len() - 1)]
        };
        let n_below = intra.iter().filter(|x| **x < 0.9).count();
        println!(
            "{},{},{:.4},{:.4},{:.4},{:.4},{}",
            codec,
            rs.len(),
            s,
            p,
            mean_intra,
            p10_intra,
            n_below
        );
        summary_rows.push((codec.clone(), rs.len(), s, p, mean_intra, p10_intra, n_below));
    }

    // ---------- Per quality band ----------
    println!();
    println!("# ---------- PER BAND (across all codecs) ----------");
    println!("# band,n,srocc,pearson");
    for (label, _, _) in Q_BANDS {
        let rs: Vec<&Measurement> = meas.iter().filter(|m| m.band == *label).collect();
        if rs.len() < 3 {
            continue;
        }
        let raws: Vec<f64> = rs.iter().map(|m| m.raw_distance).collect();
        let proxies: Vec<f64> = rs.iter().map(|m| m.proxy).collect();
        let s = spearman(&raws, &proxies);
        let p = pearson(&raws, &proxies);
        println!("{},{},{:.4},{:.4}", label, rs.len(), s, p);
    }

    // ---------- Per (codec × band) ----------
    println!();
    println!("# ---------- PER (CODEC × BAND) ----------");
    println!("# codec,band,n,srocc,pearson");
    for codec in &codec_set {
        for (label, _, _) in Q_BANDS {
            let rs: Vec<&Measurement> = meas
                .iter()
                .filter(|m| &m.codec == codec && m.band == *label)
                .collect();
            if rs.len() < 3 {
                continue;
            }
            let raws: Vec<f64> = rs.iter().map(|m| m.raw_distance).collect();
            let proxies: Vec<f64> = rs.iter().map(|m| m.proxy).collect();
            let s = spearman(&raws, &proxies);
            let p = pearson(&raws, &proxies);
            println!("{},{},{},{:.4},{:.4}", codec, label, rs.len(), s, p);
        }
    }

    // ---------- Worst-case intra-image dump ----------
    println!();
    println!("# ---------- WORST INTRA-IMAGE CASES (SROCC < 0.9) ----------");
    println!("# codec,source,quality,raw_distance,proxy,intra_srocc");
    let mut worst: Vec<(String, String, f64, Vec<&Measurement>)> = Vec::new();
    for codec in &codec_set {
        let rs: Vec<&Measurement> = meas.iter().filter(|m| &m.codec == codec).collect();
        let mut by_src: BTreeMap<String, Vec<&Measurement>> = BTreeMap::new();
        for m in &rs {
            by_src.entry(m.source.clone()).or_default().push(*m);
        }
        for (src, mut list) in by_src {
            if list.len() < 3 {
                continue;
            }
            list.sort_by_key(|m| m.quality);
            let r: Vec<f64> = list.iter().map(|m| m.raw_distance).collect();
            let p2: Vec<f64> = list.iter().map(|m| m.proxy).collect();
            let s = spearman(&r, &p2);
            if s.is_finite() && s < 0.9 {
                worst.push((codec.clone(), src, s, list));
            }
        }
    }
    worst.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    for (codec, src, s, list) in worst.iter().take(20) {
        let src_short = std::path::Path::new(src)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?");
        for m in list {
            println!(
                "{},{},{},{:.6e},{:.6e},{:.4}",
                codec, src_short, m.quality, m.raw_distance, m.proxy, s
            );
        }
    }

    // ---------- Negative-correlation hunt ----------
    println!();
    println!("# ---------- NEGATIVE-CORRELATION SCAN ----------");
    let mut neg_count = 0usize;
    let mut neg_codecs: BTreeMap<String, usize> = BTreeMap::new();
    let mut total_intra = 0usize;
    for codec in &codec_set {
        let rs: Vec<&Measurement> = meas.iter().filter(|m| &m.codec == codec).collect();
        let mut by_src: BTreeMap<String, Vec<&Measurement>> = BTreeMap::new();
        for m in &rs {
            by_src.entry(m.source.clone()).or_default().push(*m);
        }
        for (_src, mut list) in by_src {
            if list.len() < 3 {
                continue;
            }
            list.sort_by_key(|m| m.quality);
            let r: Vec<f64> = list.iter().map(|m| m.raw_distance).collect();
            let p2: Vec<f64> = list.iter().map(|m| m.proxy).collect();
            let s = spearman(&r, &p2);
            total_intra += 1;
            if s.is_finite() && s < 0.0 {
                neg_count += 1;
                *neg_codecs.entry(codec.clone()).or_default() += 1;
            }
        }
    }
    println!(
        "# intra-image SROCC < 0 cases: {}/{} (across all codecs)",
        neg_count, total_intra
    );
    for (codec, count) in neg_codecs {
        println!("#   {}: {}", codec, count);
    }

    println!();
    println!("# ---------- SUMMARY ----------");
    println!("# Pooled SROCC = {:.4}, Pearson = {:.4}, n = {}", pooled_srocc, pooled_pearson, meas.len());
    println!("# Synthetic baseline (block-quant): pooled SROCC = 0.978, intra-image >=0.9 on 29/30");
    println!("#");
    println!("# Per-codec table (markdown):");
    println!("# | codec | n | SROCC | Pearson | mean intra-img SROCC | p10 intra-img | n imgs <0.9 |");
    println!("# |---|---|---|---|---|---|---|");
    for (codec, n, s, p, mi, p10, nb) in &summary_rows {
        println!("# | {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {} |", codec, n, s, p, mi, p10, nb);
    }

    eprintln!(
        "scale0_correlation_real complete in {:.1}s",
        t_total.elapsed().as_secs_f32()
    );
}
