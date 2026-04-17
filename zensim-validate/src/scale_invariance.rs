//! Scale-invariance analyzer for `--scale-invariance` mode.
//!
//! Ingests the CSV produced by `coefficient/examples/generate_scale_pyramid`,
//! computes zensim per row, groups by (source × codec × quality × distortion),
//! and fits `score = α + β · log2(pixel_count)` per metric per group.
//!
//! See `imazen/zensim` issue #12 for the full protocol.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct PyramidRow {
    pub source_id: String,
    pub level_idx: u32,
    pub level_w: u32,
    pub level_h: u32,
    pub pixel_count: u64,
    pub codec: String,
    pub quality: u32,
    pub distortion_family: String,
    pub distortion_param: String,
    pub ref_path: PathBuf,
    pub decoded_path: PathBuf,
    pub gpu_ssimulacra2: f64,
    pub gpu_butteraugli: f64,
    pub dssim: f64,
}

#[derive(Debug, Clone)]
pub struct EnrichedRow {
    pub row: PyramidRow,
    pub zensim_score: f64,
    pub zensim_raw_distance: f64,
}

#[derive(Debug, Clone)]
pub struct GroupKey {
    pub source_id: String,
    pub codec: String,
    pub quality: u32,
    pub distortion_family: String,
    pub distortion_param: String,
}

impl GroupKey {
    fn as_csv_prefix(&self) -> String {
        format!(
            "{},{},{},{},{}",
            self.source_id, self.codec, self.quality, self.distortion_family, self.distortion_param,
        )
    }
}

/// Linear least-squares fit of y = α + β·x.
#[derive(Debug, Clone, Copy)]
pub struct Fit {
    pub alpha: f64,
    pub beta: f64,
    pub r_squared: f64,
    pub n: usize,
}

fn linear_fit(xs: &[f64], ys: &[f64]) -> Option<Fit> {
    let n = xs.len();
    if n < 2 || n != ys.len() {
        return None;
    }
    let xm = xs.iter().sum::<f64>() / n as f64;
    let ym = ys.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        num += (x - xm) * (y - ym);
        den += (x - xm).powi(2);
    }
    if den == 0.0 {
        return None;
    }
    let beta = num / den;
    let alpha = ym - beta * xm;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let yhat = alpha + beta * x;
        ss_res += (y - yhat).powi(2);
        ss_tot += (y - ym).powi(2);
    }
    let r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        // All y equal — perfect fit by constant; reportable as 1.0.
        1.0
    };
    Some(Fit {
        alpha,
        beta,
        r_squared,
        n,
    })
}

pub fn run(csv_path: &Path, out_dir: Option<&Path>, weights: &[f64]) -> anyhow::Result<()> {
    let out_dir = out_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| csv_path.parent().unwrap_or(Path::new(".")).to_path_buf());
    fs::create_dir_all(&out_dir)?;

    eprintln!("Loading pyramid CSV: {}", csv_path.display());
    let rows = load_csv(csv_path)?;
    eprintln!("Loaded {} rows", rows.len());
    if rows.is_empty() {
        anyhow::bail!("no rows in CSV");
    }

    eprintln!("Computing zensim per row...");
    let enriched = compute_zensim_per_row(rows, weights)?;
    let kept = enriched.iter().filter(|r| r.zensim_score.is_finite()).count();
    eprintln!("Computed: {} valid / {} rows", kept, enriched.len());

    let enriched_csv = out_dir.join("scale_pyramid_with_zensim.csv");
    write_enriched_csv(&enriched_csv, &enriched)?;
    eprintln!("Wrote: {}", enriched_csv.display());

    let groups = group_by_key(&enriched);
    eprintln!(
        "Grouped into {} (source × codec × quality × distortion) groups",
        groups.len()
    );

    type Getter = fn(&EnrichedRow) -> f64;
    let metrics: [(&str, bool, Getter); 4] = [
        ("zensim_score", true, |r| r.zensim_score),
        ("gpu_ssimulacra2", true, |r| r.row.gpu_ssimulacra2),
        ("gpu_butteraugli", false, |r| r.row.gpu_butteraugli),
        ("dssim", false, |r| r.row.dssim),
    ];

    let fits_csv = out_dir.join("scale_pyramid_fits.csv");
    let mut fw = fs::File::create(&fits_csv)?;
    writeln!(
        fw,
        "source_id,codec,quality,distortion_family,distortion_param,metric,higher_is_better,n,alpha,beta_per_octave,r_squared"
    )?;
    let mut all_fits: Vec<(GroupKey, &'static str, bool, Fit)> = Vec::new();
    for (key, members) in &groups {
        // x = log2(pixel_count); y = metric value at that level.
        let xs: Vec<f64> = members.iter().map(|r| (r.row.pixel_count as f64).log2()).collect();
        for (mname, higher_better, getter) in &metrics {
            let ys: Vec<f64> = members.iter().map(getter).collect();
            // Skip if any NaN/Inf.
            if !ys.iter().all(|v| v.is_finite()) {
                continue;
            }
            if let Some(fit) = linear_fit(&xs, &ys) {
                writeln!(
                    fw,
                    "{},{},{},{:.6},{:.6},{:.6},{:.6}",
                    key.as_csv_prefix(),
                    mname,
                    higher_better,
                    fit.n,
                    fit.alpha,
                    fit.beta,
                    fit.r_squared,
                )?;
                all_fits.push((key.clone(), *mname, *higher_better, fit));
            }
        }
    }
    drop(fw);
    eprintln!("Wrote: {} ({} fits)", fits_csv.display(), all_fits.len());

    // Aggregate per (metric, distortion_family).
    let summary = aggregate(&all_fits);
    print_summary(&summary);

    let html_path = out_dir.join("scale_pyramid_report.html");
    write_html_report(&html_path, &summary, &all_fits)?;
    eprintln!("Wrote: {}", html_path.display());

    Ok(())
}

fn load_csv(path: &Path) -> anyhow::Result<Vec<PyramidRow>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();
    let col = |name: &str| -> anyhow::Result<usize> {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| anyhow::anyhow!("missing column: {}", name))
    };
    let i_src = col("source_id")?;
    let i_li = col("level_idx")?;
    let i_lw = col("level_w")?;
    let i_lh = col("level_h")?;
    let i_pc = col("pixel_count")?;
    let i_codec = col("codec")?;
    let i_q = col("quality")?;
    let i_fam = col("distortion_family")?;
    let i_par = col("distortion_param")?;
    let i_ref = col("ref_path")?;
    let i_dec = col("decoded_path")?;
    let i_s2 = col("gpu_ssimulacra2")?;
    let i_ba = col("gpu_butteraugli")?;
    let i_dssim = col("dssim")?;

    let mut out = Vec::new();
    for record in rdr.records() {
        let r = record?;
        let parse_f = |s: &str| s.parse::<f64>().unwrap_or(f64::NAN);
        let parse_u32 = |s: &str| s.parse::<u32>().unwrap_or(0);
        let parse_u64 = |s: &str| s.parse::<u64>().unwrap_or(0);
        out.push(PyramidRow {
            source_id: r.get(i_src).unwrap_or("").to_string(),
            level_idx: parse_u32(r.get(i_li).unwrap_or("0")),
            level_w: parse_u32(r.get(i_lw).unwrap_or("0")),
            level_h: parse_u32(r.get(i_lh).unwrap_or("0")),
            pixel_count: parse_u64(r.get(i_pc).unwrap_or("0")),
            codec: r.get(i_codec).unwrap_or("").to_string(),
            quality: parse_u32(r.get(i_q).unwrap_or("0")),
            distortion_family: r.get(i_fam).unwrap_or("").to_string(),
            distortion_param: r.get(i_par).unwrap_or("").to_string(),
            ref_path: PathBuf::from(r.get(i_ref).unwrap_or("")),
            decoded_path: PathBuf::from(r.get(i_dec).unwrap_or("")),
            gpu_ssimulacra2: parse_f(r.get(i_s2).unwrap_or("")),
            gpu_butteraugli: parse_f(r.get(i_ba).unwrap_or("")),
            dssim: parse_f(r.get(i_dssim).unwrap_or("")),
        });
    }
    Ok(out)
}

/// Computes zensim score per row. Groups by (source_id, level_idx) so each
/// reference image is loaded and precomputed exactly once.
fn compute_zensim_per_row(
    rows: Vec<PyramidRow>,
    weights: &[f64],
) -> anyhow::Result<Vec<EnrichedRow>> {
    // Group row indices by (source_id, level_idx).
    let mut by_level: HashMap<(String, u32), Vec<usize>> = HashMap::new();
    for (idx, row) in rows.iter().enumerate() {
        by_level
            .entry((row.source_id.clone(), row.level_idx))
            .or_default()
            .push(idx);
    }

    let mut out: Vec<EnrichedRow> = rows
        .iter()
        .map(|r| EnrichedRow {
            row: r.clone(),
            zensim_score: f64::NAN,
            zensim_raw_distance: f64::NAN,
        })
        .collect();

    let groups: Vec<((String, u32), Vec<usize>)> = by_level.into_iter().collect();
    let total = groups.len();
    let progress = std::sync::atomic::AtomicUsize::new(0);

    let results: Vec<Vec<(usize, f64, f64)>> = groups
        .par_iter()
        .map(|((src_id, level_idx), indices)| {
            let mut local: Vec<(usize, f64, f64)> = Vec::with_capacity(indices.len());
            let first = &rows[indices[0]];
            let lw = first.level_w as usize;
            let lh = first.level_h as usize;
            let ref_path = &first.ref_path;
            let ref_img = match image::open(ref_path) {
                Ok(i) => i.to_rgb8(),
                Err(e) => {
                    eprintln!("skip {}/L{}: open ref {}: {}", src_id, level_idx, ref_path.display(), e);
                    let n = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if (n + 1).is_multiple_of(10) || n + 1 == total {
                        eprintln!("  zensim refs: {}/{}", n + 1, total);
                    }
                    return local;
                }
            };
            let (rw, rh) = ref_img.dimensions();
            if rw as usize != lw || rh as usize != lh {
                eprintln!(
                    "skip {}/L{}: ref {}x{} != declared {}x{}",
                    src_id, level_idx, rw, rh, lw, lh
                );
                let n = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if (n + 1).is_multiple_of(10) || n + 1 == total {
                    eprintln!("  zensim refs: {}/{}", n + 1, total);
                }
                return local;
            }
            let ref_pixels: Vec<[u8; 3]> = ref_img
                .pixels()
                .map(|p| [p.0[0], p.0[1], p.0[2]])
                .collect();
            // num_scales=4 matches the standard zensim profile / weight layout.
            let precomputed = match zensim::precompute_reference_with_scales(&ref_pixels, lw, lh, 4) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("skip {}/L{}: precompute: {}", src_id, level_idx, e);
                    let n = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if (n + 1).is_multiple_of(10) || n + 1 == total {
                        eprintln!("  zensim refs: {}/{}", n + 1, total);
                    }
                    return local;
                }
            };
            let mut config = zensim::ZensimConfig::default();
            config.num_scales = 4;
            for &i in indices {
                let row = &rows[i];
                let dec_img = match image::open(&row.decoded_path) {
                    Ok(im) => im.to_rgb8(),
                    Err(_) => continue,
                };
                let (dw, dh) = dec_img.dimensions();
                if dw as usize != lw || dh as usize != lh {
                    continue;
                }
                let dec_pixels: Vec<[u8; 3]> = dec_img
                    .pixels()
                    .map(|p| [p.0[0], p.0[1], p.0[2]])
                    .collect();
                let result = match zensim::compute_zensim_with_ref_and_config(
                    &precomputed,
                    &dec_pixels,
                    lw,
                    lh,
                    config,
                ) {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                // The result's score uses the embedded weights; for custom
                // weights we re-score from the feature vector.
                let (score, raw) = zensim::score_from_features(result.features(), weights);
                local.push((i, score, raw));
            }
            let n = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if (n + 1).is_multiple_of(10) || n + 1 == total {
                eprintln!("  zensim refs: {}/{}", n + 1, total);
            }
            local
        })
        .collect();

    for batch in results {
        for (i, score, raw) in batch {
            out[i].zensim_score = score;
            out[i].zensim_raw_distance = raw;
        }
    }

    Ok(out)
}

fn write_enriched_csv(path: &Path, rows: &[EnrichedRow]) -> anyhow::Result<()> {
    let mut f = fs::File::create(path)?;
    writeln!(
        f,
        "source_id,level_idx,level_w,level_h,pixel_count,codec,quality,distortion_family,distortion_param,ref_path,decoded_path,gpu_ssimulacra2,gpu_butteraugli,dssim,zensim_score,zensim_raw_distance"
    )?;
    for r in rows {
        writeln!(
            f,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            r.row.source_id,
            r.row.level_idx,
            r.row.level_w,
            r.row.level_h,
            r.row.pixel_count,
            r.row.codec,
            r.row.quality,
            r.row.distortion_family,
            r.row.distortion_param,
            r.row.ref_path.display(),
            r.row.decoded_path.display(),
            r.row.gpu_ssimulacra2,
            r.row.gpu_butteraugli,
            r.row.dssim,
            r.zensim_score,
            r.zensim_raw_distance,
        )?;
    }
    Ok(())
}

fn group_by_key(rows: &[EnrichedRow]) -> Vec<(GroupKey, Vec<EnrichedRow>)> {
    // BTreeMap → deterministic ordering.
    let mut groups: BTreeMap<String, (GroupKey, Vec<EnrichedRow>)> = BTreeMap::new();
    for r in rows {
        let key = GroupKey {
            source_id: r.row.source_id.clone(),
            codec: r.row.codec.clone(),
            quality: r.row.quality,
            distortion_family: r.row.distortion_family.clone(),
            distortion_param: r.row.distortion_param.clone(),
        };
        let id = key.as_csv_prefix();
        groups
            .entry(id)
            .or_insert_with(|| (key, Vec::new()))
            .1
            .push(r.clone());
    }
    // Within each group, sort by level_idx so x is monotone.
    let mut out: Vec<(GroupKey, Vec<EnrichedRow>)> = groups.into_values().collect();
    for (_k, members) in &mut out {
        members.sort_by_key(|r| r.row.level_idx);
    }
    // Drop groups with fewer than 2 levels — slope undefined.
    out.retain(|(_k, m)| m.len() >= 2);
    out
}

#[derive(Debug, Clone)]
pub struct PerMetricFamilySummary {
    pub metric: String,
    pub distortion_family: String,
    pub n_groups: usize,
    pub median_abs_beta: f64,
    pub iqr_abs_beta: (f64, f64),
    pub p95_abs_beta: f64,
    pub mean_signed_beta: f64,
    pub median_r_squared: f64,
}

fn aggregate(fits: &[(GroupKey, &'static str, bool, Fit)]) -> Vec<PerMetricFamilySummary> {
    let mut by_key: HashMap<(String, String), Vec<(f64, f64)>> = HashMap::new();
    for (k, m, _hb, f) in fits {
        by_key
            .entry((m.to_string(), k.distortion_family.clone()))
            .or_default()
            .push((f.beta, f.r_squared));
    }
    let mut out: Vec<PerMetricFamilySummary> = Vec::new();
    for ((metric, family), pairs) in by_key {
        let mut abs_betas: Vec<f64> = pairs.iter().map(|(b, _)| b.abs()).collect();
        abs_betas.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut r2s: Vec<f64> = pairs.iter().map(|(_, r)| *r).collect();
        r2s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = abs_betas.len();
        let q = |s: &[f64], pct: f64| -> f64 {
            if s.is_empty() {
                return f64::NAN;
            }
            let idx = ((pct / 100.0) * (s.len() - 1) as f64).round() as usize;
            s[idx.min(s.len() - 1)]
        };
        let mean_signed = pairs.iter().map(|(b, _)| *b).sum::<f64>() / n.max(1) as f64;
        out.push(PerMetricFamilySummary {
            metric,
            distortion_family: family,
            n_groups: n,
            median_abs_beta: q(&abs_betas, 50.0),
            iqr_abs_beta: (q(&abs_betas, 25.0), q(&abs_betas, 75.0)),
            p95_abs_beta: q(&abs_betas, 95.0),
            mean_signed_beta: mean_signed,
            median_r_squared: q(&r2s, 50.0),
        });
    }
    out.sort_by(|a, b| {
        a.distortion_family
            .cmp(&b.distortion_family)
            .then(a.metric.cmp(&b.metric))
    });
    out
}

fn print_summary(summary: &[PerMetricFamilySummary]) {
    println!("\n=== Scale-invariance: |β| (score change per octave of pixel count) ===");
    println!(
        "{:<18} {:<18} {:>6} {:>10} {:>20} {:>10} {:>12} {:>10}",
        "distortion", "metric", "n", "med|β|", "IQR|β|", "p95|β|", "mean β", "med R²"
    );
    for s in summary {
        println!(
            "{:<18} {:<18} {:>6} {:>10.4} {:>20} {:>10.4} {:>12.4} {:>10.3}",
            s.distortion_family,
            s.metric,
            s.n_groups,
            s.median_abs_beta,
            format!("[{:.3},{:.3}]", s.iqr_abs_beta.0, s.iqr_abs_beta.1),
            s.p95_abs_beta,
            s.mean_signed_beta,
            s.median_r_squared,
        );
    }
    println!();
    println!("Interpretation guide (per-metric thresholds are provisional):");
    println!("  |β| < 0.5  per octave of pixels → effectively scale-invariant");
    println!("  0.5..2.0   per octave           → measurable drift, document");
    println!("  ≥ 2.0      per octave           → significant bias, investigate");
    println!("Note: zensim & ssim2 are 0..100 scale, dssim is 0..~1, butteraugli ~0..15.");
    println!("      Compare each metric's |β| against its own scale.");
}

fn write_html_report(
    path: &Path,
    summary: &[PerMetricFamilySummary],
    all_fits: &[(GroupKey, &'static str, bool, Fit)],
) -> anyhow::Result<()> {
    let mut html = String::new();
    html.push_str("<!doctype html><html><head><meta charset=\"utf-8\">");
    html.push_str("<title>Scale-invariance report</title>");
    html.push_str("<style>");
    html.push_str("body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#222}");
    html.push_str("h1,h2{border-bottom:1px solid #ddd;padding-bottom:0.2em}");
    html.push_str("table{border-collapse:collapse;margin:1em 0;font-size:0.92em}");
    html.push_str("th,td{padding:6px 10px;border:1px solid #ddd;text-align:right}");
    html.push_str("th{background:#f4f4f4;text-align:center}");
    html.push_str("td:first-child,td:nth-child(2){text-align:left}");
    html.push_str("tr.bias-low td{background:#eaffea}");
    html.push_str("tr.bias-mid td{background:#fff7d6}");
    html.push_str("tr.bias-high td{background:#ffe1e1}");
    html.push_str("code{background:#f0f0f0;padding:1px 4px;border-radius:3px}");
    html.push_str(".note{color:#666;font-size:0.9em;margin-top:1em}");
    html.push_str("</style></head><body>");
    html.push_str("<h1>Scale-invariance evaluation</h1>");
    html.push_str("<p>For each (source × codec × quality × distortion) group, fit ");
    html.push_str("<code>score = α + β · log<sub>2</sub>(pixel_count)</code>. ");
    html.push_str("β is the score change per doubling of pixel count. ");
    html.push_str("|β| ≈ 0 = scale-invariant.</p>");
    html.push_str(
        "<p class=\"note\">Per-metric scales differ: zensim &amp; SSIMULACRA2 are 0–100, \
         butteraugli is ~0–15, DSSIM is ~0–1. Compare each metric against its own scale, \
         not across metrics.</p>",
    );

    html.push_str("<h2>Summary by (distortion family × metric)</h2>");
    html.push_str("<table><thead><tr>");
    for h in [
        "distortion family",
        "metric",
        "n groups",
        "median |β|",
        "IQR |β|",
        "p95 |β|",
        "mean signed β",
        "median R²",
    ] {
        html.push_str(&format!("<th>{}</th>", h));
    }
    html.push_str("</tr></thead><tbody>");
    for s in summary {
        let class = if s.median_abs_beta < 0.5 {
            "bias-low"
        } else if s.median_abs_beta < 2.0 {
            "bias-mid"
        } else {
            "bias-high"
        };
        html.push_str(&format!(
            "<tr class=\"{}\"><td>{}</td><td>{}</td><td>{}</td><td>{:.4}</td><td>[{:.3}, {:.3}]</td><td>{:.4}</td><td>{:+.4}</td><td>{:.3}</td></tr>",
            class,
            s.distortion_family,
            s.metric,
            s.n_groups,
            s.median_abs_beta,
            s.iqr_abs_beta.0,
            s.iqr_abs_beta.1,
            s.p95_abs_beta,
            s.mean_signed_beta,
            s.median_r_squared,
        ));
    }
    html.push_str("</tbody></table>");

    // Worst-case examples per metric (top 10 |β|, R² ≥ 0.8 to filter noise).
    html.push_str("<h2>Worst-case slopes per metric (R² ≥ 0.8)</h2>");
    let metrics: Vec<&str> = summary
        .iter()
        .map(|s| s.metric.as_str())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    let mut metrics = metrics;
    metrics.sort();
    for m in metrics {
        html.push_str(&format!("<h3>{}</h3>", m));
        let mut filtered: Vec<&(GroupKey, &'static str, bool, Fit)> = all_fits
            .iter()
            .filter(|(_, mn, _, fit)| *mn == m && fit.r_squared >= 0.8)
            .collect();
        filtered.sort_by(|a, b| {
            b.3.beta
                .abs()
                .partial_cmp(&a.3.beta.abs())
                .unwrap()
        });
        let top: Vec<_> = filtered.into_iter().take(10).collect();
        if top.is_empty() {
            html.push_str("<p><em>No fits with R² ≥ 0.8.</em></p>");
            continue;
        }
        html.push_str("<table><thead><tr>");
        for h in ["source", "codec", "quality", "family", "param", "β/oct", "R²"] {
            html.push_str(&format!("<th>{}</th>", h));
        }
        html.push_str("</tr></thead><tbody>");
        for (k, _mn, _hb, fit) in top {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:+.3}</td><td>{:.3}</td></tr>",
                k.source_id, k.codec, k.quality, k.distortion_family, k.distortion_param,
                fit.beta, fit.r_squared,
            ));
        }
        html.push_str("</tbody></table>");
    }

    html.push_str(
        "<p class=\"note\">Generated by <code>zensim-validate --scale-invariance</code>. \
         See imazen/zensim issue #12 for protocol.</p>",
    );
    html.push_str("</body></html>");
    let mut f = fs::File::create(path)?;
    f.write_all(html.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_fit_recovers_exact_line() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        let ys = [3.0, 5.0, 7.0, 9.0];
        let fit = linear_fit(&xs, &ys).unwrap();
        assert!((fit.beta - 2.0).abs() < 1e-12, "beta {}", fit.beta);
        assert!((fit.alpha - 1.0).abs() < 1e-12, "alpha {}", fit.alpha);
        assert!((fit.r_squared - 1.0).abs() < 1e-12, "R² {}", fit.r_squared);
    }

    #[test]
    fn linear_fit_constant_y_zero_slope() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        let ys = [5.0, 5.0, 5.0, 5.0];
        let fit = linear_fit(&xs, &ys).unwrap();
        assert!(fit.beta.abs() < 1e-12);
        assert!((fit.alpha - 5.0).abs() < 1e-12);
        // All-equal y → ss_tot = 0; we report R² = 1 (constant fits perfectly).
        assert_eq!(fit.r_squared, 1.0);
    }

    #[test]
    fn linear_fit_rejects_too_few_points() {
        assert!(linear_fit(&[1.0], &[2.0]).is_none());
        assert!(linear_fit(&[], &[]).is_none());
    }
}

