//! Fit each metric map's placement on butteraugli's scale for jxl-encoder's
//! loop, on the encodes zenmetrics' `loop_norm_calibrate --save-pairs` wrote,
//! by zenmetrics' documented method (`crates/zenmetrics-browser/examples/
//! loop_norm_calibrate.rs`): per 8×8 block, the map's local error (the
//! module's `loop_map_values` at scale 1, exponent 1) raised to `exponent`
//! and reduced by the 16-norm × 1.2 tile norm; for each exponent the scale is
//! the mean log ratio to butteraugli's block distance over blocks where both
//! are positive and butteraugli's is at least 0.05; the exponent with the
//! smallest log RMSE wins.
//!
//! ```text
//! ZENSIM_WORKBENCH_BAKES=... cargo run --release --example loop_calibrate -- PAIRS_DIR OUT.tsv
//! ```
//!
//! Fit: the CID22-512 *training* images. Held out: the gb82-sc screenshots.
//! The pair set's CID22-512 validation images are left out entirely (they are
//! a protected population in zensim's data splits).

use std::io::Write as _;

const EXPONENTS: [f64; 22] = [
    0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75,
    2.0, 2.5, 3.0, 4.0,
];
const FLOOR: f64 = 0.05;
const K_TILE_NORM: f64 = 1.2;

fn tile_norms(values: &[f32], width: usize, height: usize, exponent: f64) -> Vec<f64> {
    let (bx, by) = (width.div_ceil(8), height.div_ceil(8));
    let mut sum = vec![0f64; bx * by];
    let mut count = vec![0u32; bx * by];
    for y in 0..height {
        for x in 0..width {
            let b = (y / 8) * bx + x / 8;
            sum[b] += (f64::from(values[y * width + x]).max(0.0).powf(exponent)).powi(16);
            count[b] += 1;
        }
    }
    sum.iter()
        .zip(&count)
        .map(|(&s, &n)| K_TILE_NORM * (s / f64::from(n)).powf(1.0 / 16.0))
        .collect()
}

/// `(log RMSE, Pearson r of logs, blocks)` of `scale · m` against `b`.
fn fit_stats(pairs: &[(Vec<f64>, Vec<f32>)], scale: f64) -> (f64, f64, usize) {
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for (m, b) in pairs {
        for (&mv, &bv) in m.iter().zip(b) {
            if mv > 0.0 && f64::from(bv) >= FLOOR {
                xs.push((scale * mv).ln());
                ys.push(f64::from(bv).ln());
            }
        }
    }
    let n = xs.len();
    if n < 2 {
        return (f64::NAN, f64::NAN, n);
    }
    let rmse = (xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();
    let (mx, my) = (
        xs.iter().sum::<f64>() / n as f64,
        ys.iter().sum::<f64>() / n as f64,
    );
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (x, y) in xs.iter().zip(&ys) {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx).powi(2);
        syy += (y - my).powi(2);
    }
    (rmse, sxy / (sxx * syy).sqrt(), n)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().expect("PAIRS_DIR"));
    let out = std::path::PathBuf::from(args.next().expect("OUT.tsv"));
    let index = std::fs::read_to_string(dir.join("pairs.tsv")).expect("pairs.tsv");
    let mut rows = Vec::new();
    for line in index.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        let (idx, image) = (f[0].to_string(), f[2].to_string());
        let set = if image.contains("CID22-512/training/") {
            "fit"
        } else if image.starts_with("gb82-sc/") {
            "hold"
        } else {
            continue; // CID22-512 validation: protected, never read
        };
        let (w, h): (usize, usize) = (f[4].parse().expect("width"), f[5].parse().expect("height"));
        rows.push((idx, set, w, h));
    }
    let read = |p: std::path::PathBuf| {
        std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
    };
    let mut tsv = String::from(
        "metric\tmap\tscale\texponent\tfit_log_rmse\tfit_log_r\tfit_blocks\thold_log_rmse\thold_log_r\thold_blocks\n",
    );
    let metrics: Vec<String> =
        serde_json::from_str(&zensim_workbench::available_metrics()).expect("metrics");
    for metric in &metrics {
        for map in zensim_workbench::map_names() {
            // Per pair: the map's local error (scale 1, exponent 1) and butteraugli's block distances.
            let pairs: Vec<(&str, Vec<f32>, Vec<f32>, usize, usize)> = rows
                .iter()
                .map(|(idx, set, w, h)| {
                    let r = read(dir.join(format!("{idx}.ref.rgba")));
                    let d = read(dir.join(format!("{idx}.dist.rgba")));
                    let (values, _) = zensim_workbench::loop_map_values(
                        metric,
                        &map,
                        &r,
                        &d,
                        *w as u32,
                        *h as u32,
                        w * 4,
                        1.0,
                        1.0,
                    )
                    .expect("loop map");
                    let b: Vec<f32> = read(dir.join(format!("{idx}.butter.f32")))
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes(*c))
                        .collect();
                    (*set, values, b, *w, *h)
                })
                .collect();
            let mut best: Option<(f64, f64, f64)> = None;
            for &e in &EXPONENTS {
                let fit: Vec<(Vec<f64>, Vec<f32>)> = pairs
                    .iter()
                    .filter(|p| p.0 == "fit")
                    .map(|p| (tile_norms(&p.1, p.3, p.4, e), p.2.clone()))
                    .collect();
                let (mut s, mut n) = (0.0, 0usize);
                for (m, b) in &fit {
                    for (&mv, &bv) in m.iter().zip(b) {
                        if mv > 0.0 && f64::from(bv) >= FLOOR {
                            s += f64::from(bv).ln() - mv.ln();
                            n += 1;
                        }
                    }
                }
                if n == 0 {
                    continue;
                }
                let scale = (s / n as f64).exp();
                let (rmse, _, _) = fit_stats(&fit, scale);
                if best.is_none_or(|b| rmse < b.0) {
                    best = Some((rmse, scale, e));
                }
            }
            let Some((_, scale, e)) = best else {
                eprintln!("{metric}/{map}: no positive blocks");
                continue;
            };
            let set = |name: &str| -> Vec<(Vec<f64>, Vec<f32>)> {
                pairs
                    .iter()
                    .filter(|p| p.0 == name)
                    .map(|p| (tile_norms(&p.1, p.3, p.4, e), p.2.clone()))
                    .collect()
            };
            let (fr, fp, fnn) = fit_stats(&set("fit"), scale);
            let (hr, hp, hn) = fit_stats(&set("hold"), scale);
            tsv.push_str(&format!(
                "{metric}\t{map}\t{scale:.6e}\t{e}\t{fr:.4}\t{fp:.4}\t{fnn}\t{hr:.4}\t{hp:.4}\t{hn}\n"
            ));
            eprintln!(
                "{metric}/{map}: scale {scale:.4e} exponent {e} · fit rmse {fr:.3} r {fp:.3} · held-out rmse {hr:.3} r {hp:.3}"
            );
        }
    }
    std::fs::File::create(&out)
        .and_then(|mut f| f.write_all(tsv.as_bytes()))
        .expect("write tsv");
}
