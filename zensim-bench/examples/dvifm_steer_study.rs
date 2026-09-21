//! DVIFM spatial-steering study — S2 (finite-edit gain prediction) and S3
//! (two-judge severity agreement) on a stratified joint-core-v1 TRAIN sample.
//!
//! Lane: `dvifm-steer-2026-09-20`. Question: is the DVIFM block field a better
//! spatial-steering substrate than the current attribution-density map?
//!
//! * **S2** — finite interventions on pixel-size rectangles
//!   (16/32/64/128/256 px grids): copy reference pixels into the distorted
//!   image there, rescore through `BakeScorer` (the same owner the closed
//!   loop uses), and rank-correlate each map's predicted rect mass against
//!   the actual `ΔS`. Candidates: DVIFM combined field (Σ_l ε_l/n_l, the
//!   F1-normalized pooled mass), each single DVIFM level, the shipped
//!   attribution density, `ScoredAttribution::refinement_gain` (the
//!   non-additive prediction), and per-pixel SSE (the codec-default bar a
//!   steering map must beat). Also measures two-map diagnostics the prompt
//!   requires: **additivity** (ΔS of two disjoint 64-px rects edited jointly
//!   vs the sum of their individual ΔS) and the **pyramid leak** (fraction
//!   of |Δfield| mass a 64-px edit deposits OUTSIDE the rect, per DVIFM
//!   level and for the attribution density).
//! * **S3** — judge agreement: per-block severity rankings vs two
//!   independent judges — a sum-preserving fast-ssim2 score-mass field
//!   (replicated scalar aggregation over the public `Ssimulacra2Reference`
//!   scale planes, parity-checked against `compare()`) and the butteraugli
//!   diffmap — plus the judge-agreement set (top-fraction under BOTH).
//!
//! ```sh
//! cargo run --release -p zensim-bench --features "training zen-decode" \
//!   --example dvifm_steer_study -- \
//!   --pairs-tsv /mnt/v/output/zensim/joint-core-v1/pairs/pairs_core.tsv \
//!   --out-dir /mnt/v/output/zensim/dvifm-steer-2026-09-20
//! ```
//!
//! Outputs (all capped; no full caches): `pairs.jsonl` (one scalar row per
//! pair), `blocks.bin` + `rects.bin` (f32-LE sidecars the rows index into),
//! `summary.json` (aggregate + provenance + adapter-parity maxima).
//! NO GRACEFUL SKIPS: a pair that fails to decode or score is a hard error;
//! only sampling eligibility (leg list, size floor) drops rows, and those
//! are counted in the summary.

#[path = "shared/zen_decode.rs"]
mod zen_decode;

use butteraugli::{ButteraugliParams, Img, RGB8, butteraugli};
use fast_ssim2::{Blur, Ssimulacra2Reference};
use imgref::ImgRef;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};
use zenstats::{kendall_tau, pearson, spearman};
use zensim::feature_set_id::SlotSet;
use zensim::research::{DvifmFieldMap, Request};
use zensim::{BakeScorer, Fused944Session, RgbSlice};

// ===========================================================================
// fast-ssim2 0.8.2 score composition — the published ssimulacra2/libjxl
// weight table and score polynomial, replicated scalar-side so the per-pixel
// severity field and the pooled score share one arithmetic. `score()` walks
// WEIGHT *linearly* in (channel, scale_iter, norm, term) order, so with
// scales_n < 6 the effective weight for a (c, s) pair is NOT the nominal
// 6-scale layout cell — replicate the linear walk exactly.
// ===========================================================================

const SSIM2_WEIGHT: [f64; 108] = [
    0.0, 0.000_737_660_670_740_658_6, 0.0,
    0.0, 0.000_779_348_168_286_730_9, 0.0,
    0.0, 0.000_437_115_573_010_737_9, 0.0,
    1.104_172_642_665_734_6, 0.000_662_848_341_292_71, 0.000_152_316_327_837_187_52,
    0.0, 0.001_640_643_745_659_975_4, 0.0,
    1.842_245_552_053_929_8, 11.441_172_603_757_666, 0.0,
    0.000_798_910_943_601_516_3, 0.000_176_816_438_078_653, 0.0,
    1.878_759_497_954_638_7, 10.949_069_906_051_42, 0.0,
    0.000_728_934_699_150_807_2, 0.967_793_708_062_683_3, 0.0,
    0.000_140_034_242_854_358_84, 0.998_176_697_785_496_7, 0.000_319_497_559_344_350_53,
    0.000_455_099_211_379_206_3, 0.0, 0.0,
    0.001_364_876_616_324_339_8, 0.0, 0.0,
    0.0, 0.0, 0.0,
    7.466_890_328_078_848, 0.0, 17.445_833_984_131_262,
    0.000_623_560_163_404_146_6, 0.0, 0.0,
    6.683_678_146_179_332, 0.000_377_244_079_796_112_96, 1.027_889_937_768_264,
    225.205_153_008_492_74, 0.0, 0.0,
    19.213_238_186_143_016, 0.001_140_152_458_661_836_1, 0.001_237_755_635_509_985,
    176.393_175_984_506_94, 0.0, 0.0,
    24.433_009_998_704_76, 0.285_208_026_121_177_57, 0.000_448_543_692_383_340_8,
    0.0, 0.0, 0.0,
    34.779_063_444_837_72, 44.835_625_328_877_896, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.000_868_055_657_329_169_8, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.000_531_319_187_435_874_7, 0.0,
    0.000_165_338_141_613_791_12, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.000_417_917_180_325_133_6, 0.001_729_082_823_472_283_3, 0.0,
    0.002_082_700_584_663_643_7, 0.0, 0.0,
    8.826_982_764_996_862, 23.192_433_439_989_26, 0.0,
    95.108_049_881_108_6, 0.986_397_803_440_068_2, 0.983_438_279_246_535_3,
    0.001_228_640_504_827_849_3, 171.266_725_589_730_7, 0.980_785_887_243_537_9,
    0.0, 0.0, 0.0,
    0.000_513_006_458_899_067_9, 0.0, 0.000_108_540_578_584_115_37,
];

const SSIM2_C2: f32 = 0.0009;

/// Per-scale (channel × scale) scalar aggregates + the per-pixel severity
/// field at that scale's resolution. `avg` matches `simd_ops`'s
/// `plane_averages` layout: `avg_ssim[c*2+n]`, `avg_edge[c*4+{0..3}]`.
struct Ssim2Scale {
    width: usize,
    height: usize,
    /// Per-pixel score-mass share at THIS scale's resolution — sums to the
    /// scale's total weighted contribution (all channels).
    severity: Vec<f64>,
    avg_ssim: [f64; 6],
    avg_edge: [f64; 12],
}

/// The replicated-aggregation ssim2 product: a scale-0 score-mass density
/// (each pixel's share of the pooled score) plus the replicated score for
/// parity checking against `Ssimulacra2Reference::compare`.
struct Ssim2Field {
    density: Vec<f64>,
    score: f64,
}

/// Build the per-pixel severity field for one (reference, distorted) pair
/// by replicating `ssim_map_simd`/`edge_diff_map_simd` scalar-side over the
/// two `Ssimulacra2Reference` scale planes (`new(dist)` runs the identical
/// pipeline as `compare_with`'s distorted side). Severity is the
/// sum-preserving decomposition of the score's weighted terms: L1 terms
/// contribute `w·x_i/N`, L4 terms `w·x_i⁴/(N·r4³)` where `r4 = (mean x⁴)^{1/4}`
/// — each sums back to exactly `w·mean`/`w·r4`, the pooled score term.
fn ssim2_field(
    ref_img: ImgRef<'_, [u8; 3]>,
    dist_img: ImgRef<'_, [u8; 3]>,
) -> Result<Ssim2Field, String> {
    let r1 = Ssimulacra2Reference::new(ref_img).map_err(|e| format!("ssim2 ref: {e}"))?;
    let r2 = Ssimulacra2Reference::new(dist_img).map_err(|e| format!("ssim2 dist: {e}"))?;
    let ns = r1.num_scales();
    let (w, h) = (ref_img.width(), ref_img.height());
    let mut density = vec![0.0f64; w * h];
    let mut scales: Vec<Ssim2Scale> = Vec::with_capacity(ns);
    let mut blur = Blur::new(w, h);
    let mut mul = [
        vec![0.0f32; w * h],
        vec![0.0f32; w * h],
        vec![0.0f32; w * h],
    ];

    for s in 0..ns {
        let v1 = r1.scale_planes(s).expect("scale index in range");
        let v2 = r2.scale_planes(s).expect("scale index in range");
        let (sw, sh) = (v1.width, v1.height);
        let n = sw * sh;
        // σ12 = blur(img1·img2) — the one cross term compare_with needs
        // that neither side's ScalePlanesView carries.
        for c in 0..3 {
            mul[c].truncate(n);
            for i in 0..n {
                mul[c][i] = v1.img1_planar[c][i] * v2.img1_planar[c][i];
            }
        }
        blur.shrink_to(sw, sh);
        let sigma12 = blur.blur(&mul);

        let mut severity = vec![0.0f64; n];
        let mut avg_ssim = [0.0f64; 6];
        let mut avg_edge = [0.0f64; 12];
        let nf = n as f64;
        for c in 0..3 {
            let mut sum_d = 0.0f64;
            let mut sum_d4 = 0.0f64;
            let mut sum_a = 0.0f64;
            let mut sum_a4 = 0.0f64;
            let mut sum_dl = 0.0f64;
            let mut sum_dl4 = 0.0f64;
            // Scalar replication of ssim_map_inner / edge_diff_map_inner —
            // identical formulas, sequential accumulation (the SIMD kernels
            // chunk by 8 lanes; the difference is float-associativity noise
            // only, bounded by the compare() parity gate below).
            for i in 0..n {
                let mu1 = v1.mu1[c][i] as f64;
                let mu2 = v2.mu1[c][i] as f64;
                let s11 = v1.sigma1_sq[c][i] as f64;
                let s22 = v2.sigma1_sq[c][i] as f64;
                let s12 = sigma12[c][i] as f64;
                let mu_diff = mu1 - mu2;
                let num_m = (-mu_diff).mul_add(mu_diff, 1.0);
                let num_s = 2.0f64.mul_add(s12 - mu1 * mu2, f64::from(SSIM2_C2));
                let den_s = (s11 - mu1 * mu1) + (s22 - mu2 * mu2) + f64::from(SSIM2_C2);
                let d = (1.0 - (num_m * num_s) / den_s).max(0.0);
                let d1 = (1.0 + (v2.img1_planar[c][i] as f64 - mu2).abs())
                    / (1.0 + (v1.img1_planar[c][i] as f64 - mu1).abs())
                    - 1.0;
                let artifact = d1.max(0.0);
                let detail = (-d1).max(0.0);
                sum_d += d;
                sum_d4 += d * d * d * d;
                sum_a += artifact;
                sum_a4 += artifact.powi(4);
                sum_dl += detail;
                sum_dl4 += detail.powi(4);
            }
            avg_ssim[c * 2] = sum_d / nf;
            avg_ssim[c * 2 + 1] = (sum_d4 / nf).sqrt().sqrt();
            avg_edge[c * 4] = sum_a / nf;
            avg_edge[c * 4 + 1] = (sum_a4 / nf).sqrt().sqrt();
            avg_edge[c * 4 + 2] = sum_dl / nf;
            avg_edge[c * 4 + 3] = (sum_dl4 / nf).sqrt().sqrt();
            // Second pass with the pooled r4 values — sum-preserving mass.
            let w_at = |n_idx: usize, m: usize| SSIM2_WEIGHT[((c * ns + s) * 2 + n_idx) * 3 + m];
            let w00 = w_at(0, 0);
            let w01 = w_at(0, 1);
            let w02 = w_at(0, 2);
            let w10 = w_at(1, 0);
            let w11 = w_at(1, 1);
            let w12 = w_at(1, 2);
            let r4d = avg_ssim[c * 2 + 1];
            let r4a = avg_edge[c * 4 + 1];
            let r4l = avg_edge[c * 4 + 3];
            for i in 0..n {
                let mu1 = v1.mu1[c][i] as f64;
                let mu2 = v2.mu1[c][i] as f64;
                let s11 = v1.sigma1_sq[c][i] as f64;
                let s22 = v2.sigma1_sq[c][i] as f64;
                let s12 = sigma12[c][i] as f64;
                let mu_diff = mu1 - mu2;
                let num_m = (-mu_diff).mul_add(mu_diff, 1.0);
                let num_s = 2.0f64.mul_add(s12 - mu1 * mu2, f64::from(SSIM2_C2));
                let den_s = (s11 - mu1 * mu1) + (s22 - mu2 * mu2) + f64::from(SSIM2_C2);
                let d = (1.0 - (num_m * num_s) / den_s).max(0.0);
                let d1 = (1.0 + (v2.img1_planar[c][i] as f64 - mu2).abs())
                    / (1.0 + (v1.img1_planar[c][i] as f64 - mu1).abs())
                    - 1.0;
                let artifact = d1.max(0.0);
                let detail = (-d1).max(0.0);
                let mut sev = w00 * d + w01 * artifact + w02 * detail;
                if r4d > 0.0 {
                    sev += w10 * d * d * d * d / (nf * r4d * r4d * r4d);
                }
                if r4a > 0.0 {
                    sev += w11 * artifact.powi(4) / (nf * r4a * r4a * r4a);
                }
                if r4l > 0.0 {
                    sev += w12 * detail.powi(4) / (nf * r4l * r4l * r4l);
                }
                severity[i] += sev;
            }
        }
        scales.push(Ssim2Scale {
            width: sw,
            height: sh,
            severity,
            avg_ssim,
            avg_edge,
        });
    }

    // Replicated `Msssim::score()` — linear WEIGHT walk, then the published
    // polynomial + power transform, verbatim.
    let mut ssim = 0.0f64;
    let mut i = 0usize;
    for c in 0..3 {
        for scale in &scales {
            for n in 0..2 {
                ssim = SSIM2_WEIGHT[i].mul_add(scale.avg_ssim[c * 2 + n].abs(), ssim);
                i += 1;
                ssim = SSIM2_WEIGHT[i].mul_add(scale.avg_edge[c * 4 + n].abs(), ssim);
                i += 1;
                ssim = SSIM2_WEIGHT[i].mul_add(scale.avg_edge[c * 4 + n + 2].abs(), ssim);
                i += 1;
            }
        }
    }
    ssim *= 0.956_238_261_683_484_4_f64;
    ssim = (6.248_496_625_763_138e-5 * ssim * ssim).mul_add(
        ssim,
        2.326_765_642_916_932f64.mul_add(ssim, -0.020_884_521_182_843_837 * ssim * ssim),
    );
    let score = if ssim > 0.0 {
        ssim.powf(0.627_633_646_783_138_7).mul_add(-10.0, 100.0)
    } else {
        100.0
    };

    // Splat each scale's per-pixel mass onto its scale-0 footprint — every
    // scale-s cell's mass is divided over its VALID source pixels (the
    // ceil-divided border cells have a clipped footprint), preserving total
    // mass exactly.
    for (s, sc) in scales.iter().enumerate() {
        let step = 1usize << s;
        for sy in 0..sc.height {
            let y0 = sy * step;
            let y1 = ((sy + 1) * step).min(h);
            for sx in 0..sc.width {
                let x0 = sx * step;
                let x1 = ((sx + 1) * step).min(w);
                let count = (x1 - x0) * (y1 - y0);
                let share = sc.severity[sy * sc.width + sx] / count as f64;
                for y in y0..y1 {
                    for x in x0..x1 {
                        density[y * w + x] += share;
                    }
                }
            }
        }
    }
    Ok(Ssim2Field { density, score })
}

// ===========================================================================
// TRAIN manifest
// ===========================================================================

#[derive(Debug)]
struct PairRow {
    ref_path: PathBuf,
    dist_path: PathBuf,
    leg: String,
    codec: String,
    q: String,
    band: String,
    ref_basename: String,
    group: String,
}

fn load_pairs(tsv: &Path) -> Vec<PairRow> {
    let text = std::fs::read_to_string(tsv).expect("read pairs tsv");
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 || line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        assert!(f.len() >= 10, "pairs row {i} has {} columns", f.len());
        rows.push(PairRow {
            ref_path: PathBuf::from(f[0]),
            dist_path: PathBuf::from(f[1]),
            leg: f[3].into(),
            codec: f[4].into(),
            q: f[5].into(),
            band: f[6].into(),
            ref_basename: f[8].into(),
            group: f[9].into(),
        });
    }
    rows
}

/// Deterministic stratified sample: SDR legs only (the `hdr` leg is excluded
/// — the B bake and DVIFM default plane are SDR), per-(leg, codec) caps from
/// `CAPS`, stride-sampled over (band, ref_basename, q, dist) sort so refs,
/// quality rungs and bands all spread.
fn stratified_sample(rows: &[PairRow], cap: usize) -> Vec<PairRow> {
    const CAPS: &[(&str, &str, usize)] = &[
        ("fresh_imazen26", "zenjpeg-420-e2", 56),
        ("fresh_imazen26", "zenwebp-m4", 56),
        ("fresh_imazen26", "zenavif-s6", 56),
        ("fresh_imazen26", "zenjxl-e7", 56),
        ("fresh_safesyn", "zenjpeg-420-e2", 18),
        ("fresh_safesyn", "zenwebp-m4", 18),
        ("fresh_safesyn", "zenavif-s6", 18),
        ("fresh_safesyn", "zenjxl-e7", 18),
        ("cid22", "aom", 20),
        ("cid22", "cld_avif", 20),
        ("cid22", "cld_heic", 15),
        ("cid22", "cld_jp2", 15),
        ("cid22", "cld_webp", 15),
        ("cid22", "libjxl", 15),
        ("cid22", "mozjpeg", 10),
        ("cid22", "vis_avif", 6),
        ("human", "images", 30),
        ("human", "distorted_images_png", 30),
        ("konfig", "jnd-levels", 10),
    ];
    let mut picked: Vec<&PairRow> = Vec::new();
    for &(leg, codec, group_cap) in CAPS {
        let mut cell: Vec<&PairRow> = rows
            .iter()
            .filter(|r| r.leg == leg && r.codec == codec)
            .collect();
        cell.sort_by(|a, b| {
            (&a.band, &a.ref_basename, &a.q, &a.dist_path)
                .cmp(&(&b.band, &b.ref_basename, &b.q, &b.dist_path))
        });
        let k = group_cap.min(cell.len());
        if k == cell.len() {
            picked.extend(cell);
        } else {
            for i in 0..k {
                let idx = (i * cell.len()) / k; // even coverage of the sorted cell
                picked.push(cell[idx]);
            }
        }
    }
    let mut picked: Vec<PairRow> = picked
        .into_iter()
        .map(|r| PairRow {
            ref_path: r.ref_path.clone(),
            dist_path: r.dist_path.clone(),
            leg: r.leg.clone(),
            codec: r.codec.clone(),
            q: r.q.clone(),
            band: r.band.clone(),
            ref_basename: r.ref_basename.clone(),
            group: r.group.clone(),
        })
        .collect();
    if picked.len() > cap {
        let n = picked.len();
        let keep: std::collections::HashSet<usize> =
            (0..cap).map(|i| (i * n) / cap).collect();
        picked = picked
            .into_iter()
            .enumerate()
            .filter_map(|(i, r)| keep.contains(&i).then_some(r))
            .collect();
    }
    picked
}

// ===========================================================================
// helpers
// ===========================================================================

fn sha256_hex(path: &Path) -> String {
    let bytes = std::fs::read(path).expect("read for sha256");
    Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn write_f32(file: &mut std::fs::File, vals: &[f64]) -> (u64, u64) {
    let mut buf = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        buf.extend_from_slice(&(*v as f32).to_le_bytes());
    }
    let off = file.metadata().map(|m| m.len()).unwrap_or(0);
    file.write_all(&buf).expect("write f32 sidecar");
    (off, buf.len() as u64)
}

/// Spearman with a NaN/degenerate guard — identical to zenstats but the
/// caller decides what "no mass anywhere" means (spearman() returns 0.0 on
/// constant input; we additionally report it as 0 with a flag upstream).
fn srocc(a: &[f64], b: &[f64]) -> f64 {
    spearman(a, b)
}

/// Least-squares `y ≈ slope·x + intercept` — the calibration pair reported
/// alongside rank agreement (map mass has its own units; the slope is how
/// many score points a unit of predicted mass is worth at THIS granularity).
fn linfit(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len().min(y.len()) as f64;
    if n < 2.0 {
        return (0.0, 0.0);
    }
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut sxy = 0.0f64;
    let mut sxx = 0.0f64;
    for i in 0..x.len().min(y.len()) {
        sxy += (x[i] - mx) * (y[i] - my);
        sxx += (x[i] - mx) * (x[i] - mx);
    }
    if sxx < 1e-30 {
        return (0.0, my);
    }
    let slope = sxy / sxx;
    (slope, my - slope * mx)
}

/// Fraction of `truth`'s top-`k` items that are also in `pred`'s top-`k`
/// (ties broken by index order — same rule for both lists).
fn precision_at_k(pred: &[f64], truth: &[f64], k: usize) -> f64 {
    let n = pred.len().min(truth.len());
    if k == 0 || n == 0 {
        return 0.0;
    }
    let k = k.min(n);
    let mut pi: Vec<usize> = (0..n).collect();
    let mut ti = pi.clone();
    pi.sort_by(|&a, &b| pred[b].total_cmp(&pred[a]).then(a.cmp(&b)));
    ti.sort_by(|&a, &b| truth[b].total_cmp(&truth[a]).then(a.cmp(&b)));
    let in_top: std::collections::HashSet<usize> = ti[..k].iter().copied().collect();
    let hit = pi[..k].iter().filter(|i| in_top.contains(i)).count();
    hit as f64 / k as f64
}

/// Indices of the top `ceil(n·frac)` items of `v` (stable tie-break by idx).
fn top_frac_set(v: &[f64], frac: f64) -> std::collections::HashSet<usize> {
    let n = v.len();
    let k = ((n as f64) * frac).ceil() as usize;
    let k = k.clamp(1, n);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    idx[..k].iter().copied().collect()
}

// ===========================================================================

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pairs_tsv: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut bake: Option<PathBuf> = None;
    let mut max_pairs = 240usize;
    let mut max_interventions = 256usize;
    let mut top_frac = 0.10f64;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--pairs-tsv" => pairs_tsv = Some(PathBuf::from(&args[i + 1])),
            "--out-dir" => out_dir = Some(PathBuf::from(&args[i + 1])),
            "--bake" => bake = Some(PathBuf::from(&args[i + 1])),
            "--max-pairs" => max_pairs = args[i + 1].parse().unwrap(),
            "--max-interventions" => max_interventions = args[i + 1].parse().unwrap(),
            "--top-frac" => top_frac = args[i + 1].parse().unwrap(),
            other => panic!("unknown arg {other}"),
        }
        i += 2;
    }
    let pairs_tsv = pairs_tsv.expect("--pairs-tsv required");
    let out_dir = out_dir.expect("--out-dir required");
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let default_bake = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_byid_2026-09-06.bin");
    let bake = bake.unwrap_or(default_bake);
    assert!(bake.exists(), "bake missing: {}", bake.display());

    let model_bytes = std::fs::read(&bake).expect("read bake");
    let bake_sha = sha256_hex(&bake);
    let model = zenpredict::Model::from_bytes(&model_bytes).expect("parse bake");
    let pairs_sha = sha256_hex(&pairs_tsv);

    let rows = load_pairs(&pairs_tsv);
    let picked = stratified_sample(&rows, max_pairs);
    eprintln!(
        "dvifm_steer_study: {} pairs sampled of {} manifest rows",
        picked.len(),
        rows.len()
    );

    // The w986 request carries the DVIFM block-field side channel through the
    // SAME walk that emits pooled features (the steering field derives from
    // the pump's own block records — no second map pipeline).
    let req = Request::for_slots(SlotSet::from_ranges([(0, 986)]), 986)
        .collect_dvifm_fields(true);

    let mut jsonl = std::fs::File::create(out_dir.join("pairs.jsonl")).expect("pairs.jsonl");
    let mut blocks_bin = std::fs::File::create(out_dir.join("blocks.bin")).expect("blocks.bin");
    let mut rects_bin = std::fs::File::create(out_dir.join("rects.bin")).expect("rects.bin");

    let mut n_done = 0usize;
    let mut n_skipped_size = 0usize;
    let mut parity_max = 0.0f64;
    let t_start = std::time::Instant::now();

    for (pi, row) in picked.iter().enumerate() {
        let refd = zen_decode::decode_rgb8_path(&row.ref_path)
            .unwrap_or_else(|e| panic!("{}: {e}", row.ref_path.display()));
        let distd = zen_decode::decode_rgb8_path(&row.dist_path)
            .unwrap_or_else(|e| panic!("{}: {e}", row.dist_path.display()));
        let (w, h) = (refd.width as usize, refd.height as usize);
        assert_eq!(
            (w, h),
            (distd.width as usize, distd.height as usize),
            "dim mismatch {}",
            row.dist_path.display()
        );
        let nbx = w / 5;
        let nby = h / 5;
        if w < 40 || h < 40 {
            n_skipped_size += 1;
            eprintln!("  [{pi}] skip {w}x{h} (lattice too small) {}", row.dist_path.display());
            continue;
        }
        let rpx: Vec<[u8; 3]> = refd
            .pixels
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let dpx: Vec<[u8; 3]> = distd
            .pixels
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let rs = RgbSlice::new(&rpx, w, h);
        let ds = RgbSlice::new(&dpx, w, h);

        // --- control map + base score through the BakeScorer attribution
        // surface — the same owner `diffmap_block_coherence` uses, minus the
        // `SteeringSession` integrity-head gate (which would refuse rows the
        // study must still measure; the gate flag is recorded per pair).
        let mut scorer = BakeScorer::new(&model).expect("servable bake");
        let pre = scorer.precompute_reference(&rs).expect("ref precompute");
        let mut session = Fused944Session::new();
        let sa = scorer
            .compute_with_ref_and_attribution(&rs, &pre, &ds, None, &mut session, 1)
            .expect("score + attribution");
        let base_score = sa.result().score();
        let attr_map = sa.attribution().clone();

        // --- DVIFM field via the same walk that emits pooled features
        let ext = zensim::research::extract(&req, &rs, &ds)
            .unwrap_or_else(|e| panic!("research extract {}: {e:?}", row.dist_path.display()));
        let fields: &DvifmFieldMap = ext
            .dvifm_fields()
            .expect("collect_dvifm_fields(true) must produce fields");
        let n_levels = fields.levels().len();
        assert_eq!(n_levels, 5, "DVIFM has five pyramid levels");

        // --- block-lattice vectors (scale-0 full 5×5 blocks)
        let nb = nbx * nby;
        let mut dvifm_comb = vec![0.0f64; nb];
        let mut dvifm_lvl = vec![vec![0.0f64; nb]; n_levels];
        let mut attr_blk = vec![0.0f64; nb];
        let mut sse_blk = vec![0.0f64; nb];
        for by in 0..nby {
            for bx in 0..nbx {
                let b = by * nbx + bx;
                let (x0, y0, x1, y1) = (bx * 5, by * 5, bx * 5 + 5, by * 5 + 5);
                for (l, lv) in fields.levels().iter().enumerate() {
                    let q = lv.query_eps_scale0(x0 as u32, y0 as u32, x1 as u32, y1 as u32)
                        / lv.n_blocks() as f64;
                    dvifm_lvl[l][b] = q;
                    dvifm_comb[b] += q;
                }
                attr_blk[b] = attr_map.query_rect(x0, y0, x1, y1);
                let mut s = 0.0f64;
                for y in y0..y1 {
                    for x in x0..x1 {
                        let i = y * w + x;
                        for c in 0..3 {
                            let d = rpx[i][c] as f64 - dpx[i][c] as f64;
                            s += d * d;
                        }
                    }
                }
                sse_blk[b] = s;
            }
        }

        // --- S3 judges
        let ref_img = Img::new(rpx.clone(), w, h);
        let dist_img = Img::new(dpx.clone(), w, h);
        let s2f = ssim2_field(ref_img.as_ref(), dist_img.as_ref())
            .unwrap_or_else(|e| panic!("ssim2 field {}: {e}", row.dist_path.display()));
        let ssim2_ref_score = Ssimulacra2Reference::new(ref_img.as_ref())
            .expect("ssim2 ref")
            .compare(dist_img.as_ref())
            .expect("ssim2 compare");
        let parity = (s2f.score - ssim2_ref_score).abs();
        parity_max = parity_max.max(parity);
        assert!(
            parity < 1e-2,
            "ssim2 adapter parity {} exceeds 1e-2 on {}",
            parity,
            row.dist_path.display()
        );
        let mut ssim2_blk = vec![0.0f64; nb];
        for by in 0..nby {
            for bx in 0..nbx {
                let mut m = 0.0f64;
                for y in (by * 5)..(by * 5 + 5) {
                    for x in (bx * 5)..(bx * 5 + 5) {
                        m += s2f.density[y * w + x];
                    }
                }
                ssim2_blk[by * nbx + bx] = m;
            }
        }
        let bpx: Vec<RGB8> = bytemuck::cast_slice(&rpx).to_vec();
        let bdx: Vec<RGB8> = bytemuck::cast_slice(&dpx).to_vec();
        let bres = butteraugli(
            Img::new(bpx, w, h).as_ref(),
            Img::new(bdx, w, h).as_ref(),
            &ButteraugliParams::default().with_compute_diffmap(true),
        )
        .unwrap_or_else(|e| panic!("butteraugli {}: {e}", row.dist_path.display()));
        let dmap = bres.diffmap.as_ref().expect("diffmap requested");
        let mut butter_blk = vec![0.0f64; nb];
        for by in 0..nby {
            for bx in 0..nbx {
                let mut m = 0.0f64;
                for y in (by * 5)..(by * 5 + 5) {
                    for x in (bx * 5)..(bx * 5 + 5) {
                        m += dmap.buf()[y * w + x] as f64;
                    }
                }
                butter_blk[by * nbx + bx] = m;
            }
        }

        // --- S2 finite interventions over PIXEL-size rect grids
        // (16/32/64/128/256 px — the prompt's size axis; rects need not be
        // lattice-aligned: predictions go through the fractional/area-weighted
        // queries, edits are pixel-exact).
        const RECT_SIZES: [usize; 5] = [16, 32, 64, 128, 256];
        const ADD_SIZE: usize = 64; // additivity + pyramid-leak probe size
        const MAX_ADD_PAIRS: usize = 12;
        const MAX_LEAK_RECTS: usize = 4;

        // rect_records: (size_idx, x0, y0, x1, y1)
        let mut rect_records: Vec<(usize, usize, usize, usize, usize)> = Vec::new();
        let mut size_rects: Vec<Vec<usize>> = vec![Vec::new(); RECT_SIZES.len()];
        for (si, &s) in RECT_SIZES.iter().enumerate() {
            if w < 2 * s || h < 2 * s {
                continue;
            }
            let nx = w / s;
            let ny = h / s;
            let mut grid: Vec<(usize, usize, usize, usize)> =
                Vec::with_capacity(nx * ny);
            for ry in 0..ny {
                for rx in 0..nx {
                    grid.push((rx * s, ry * s, (rx + 1) * s, (ry + 1) * s));
                }
            }
            let per_size_cap = max_interventions / RECT_SIZES.len();
            if grid.len() > per_size_cap {
                let stride = grid.len() as f64 / per_size_cap as f64;
                let mut sub = Vec::with_capacity(per_size_cap);
                for k in 0..per_size_cap {
                    sub.push(grid[(k as f64 * stride) as usize]);
                }
                grid = sub;
            }
            for r in grid {
                size_rects[si].push(rect_records.len());
                rect_records.push((si, r.0, r.1, r.2, r.3));
            }
        }

        let mut pred_dvifm = Vec::with_capacity(rect_records.len());
        let mut pred_dvifm_lvl: Vec<Vec<f64>> =
            vec![Vec::with_capacity(rect_records.len()); n_levels];
        let mut pred_attr = Vec::with_capacity(rect_records.len());
        let mut pred_refgain = Vec::with_capacity(rect_records.len());
        let mut pred_sse = Vec::with_capacity(rect_records.len());
        let mut delta_s = Vec::with_capacity(rect_records.len());
        for &(_, x0, y0, x1, y1) in &rect_records {
            let mut dv = 0.0f64;
            for (l, lv) in fields.levels().iter().enumerate() {
                let q = lv.query_eps_scale0(x0 as u32, y0 as u32, x1 as u32, y1 as u32)
                    / lv.n_blocks() as f64;
                pred_dvifm_lvl[l].push(q);
                dv += q;
            }
            pred_dvifm.push(dv);
            pred_attr.push(attr_map.query_rect(x0, y0, x1, y1));
            pred_refgain.push(sa.refinement_gain(x0, y0, x1, y1));
            let mut s = 0.0f64;
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = y * w + x;
                    for c in 0..3 {
                        let d = rpx[i][c] as f64 - dpx[i][c] as f64;
                        s += d * d;
                    }
                }
            }
            pred_sse.push(s);
            let mut edited = dpx.clone();
            for y in y0..y1 {
                for x in x0..x1 {
                    edited[y * w + x] = rpx[y * w + x];
                }
            }
            let s_edit = scorer
                .compute(&rs, &RgbSlice::new(&edited, w, h), None)
                .expect("intervention rescore")
                .score();
            delta_s.push(s_edit - base_score);
        }

        // --- S2 metrics, per size and pooled
        let metrics_for = |idx: &[usize]| {
            let take = |v: &[f64]| idx.iter().map(|&k| v[k]).collect::<Vec<_>>();
            let (pd, pa, pr, ps, ds) = (
                take(&pred_dvifm),
                take(&pred_attr),
                take(&pred_refgain),
                take(&pred_sse),
                take(&delta_s),
            );
            json!({
                "n": idx.len(),
                "srocc_dvifm": srocc(&pd, &ds),
                "srocc_attr": srocc(&pa, &ds),
                "srocc_refgain": srocc(&pr, &ds),
                "srocc_sse": srocc(&ps, &ds),
                "srocc_dvifm_levels": (0..n_levels)
                    .map(|l| srocc(&take(&pred_dvifm_lvl[l]), &ds))
                    .collect::<Vec<_>>(),
                "pearson_dvifm": pearson(&pd, &ds),
                "pearson_attr": pearson(&pa, &ds),
                "kendall_dvifm": kendall_tau(&pd, &ds),
                "kendall_attr": kendall_tau(&pa, &ds),
                "fit_dvifm": linfit(&pd, &ds),
                "fit_attr": linfit(&pa, &ds),
                "fit_sse": linfit(&ps, &ds),
                "precision_q_dvifm": precision_at_k(&pd, &ds, idx.len() / 4),
                "precision_q_attr": precision_at_k(&pa, &ds, idx.len() / 4),
                "mean_delta_s": ds.iter().sum::<f64>() / ds.len().max(1) as f64,
                "frac_negative_delta_s": ds.iter().filter(|&&d| d < 0.0).count() as f64
                    / ds.len().max(1) as f64,
            })
        };
        let all_idx: Vec<usize> = (0..rect_records.len()).collect();
        let mut s2 = metrics_for(&all_idx);
        let mut by_size = serde_json::Map::new();
        for (si, &s) in RECT_SIZES.iter().enumerate() {
            if !size_rects[si].is_empty() {
                by_size.insert(s.to_string(), metrics_for(&size_rects[si]));
            }
        }
        s2["by_size"] = json!(by_size);
        s2["sizes"] = json!(RECT_SIZES);

        // --- additivity: disjoint non-adjacent rect pairs at ADD_SIZE.
        // residual = ΔS(A∪B) − (ΔS_A + ΔS_B); a strictly local+additive map
        // predicts 0.
        let add_si = RECT_SIZES.iter().position(|&s| s == ADD_SIZE).unwrap();
        let mut add_tests = Vec::new();
        'outer: for (ai, &ka) in size_rects[add_si].iter().enumerate() {
            for &kb in &size_rects[add_si][ai + 1..] {
                let (_, ax0, ay0, ax1, ay1) = rect_records[ka];
                let (_, bx0, by0, bx1, by1) = rect_records[kb];
                // non-adjacent: touching corners allowed, shared edge is not
                let edge_touch =
                    (ax1 == bx0 || bx1 == ax0) && ay0.max(by0) < ay1.min(by1)
                        || (ay1 == by0 || by1 == ay0) && ax0.max(bx0) < ax1.min(bx1);
                if edge_touch {
                    continue;
                }
                let mut edited = dpx.clone();
                for y in ay0..ay1 {
                    for x in ax0..ax1 {
                        edited[y * w + x] = rpx[y * w + x];
                    }
                }
                for y in by0..by1 {
                    for x in bx0..bx1 {
                        edited[y * w + x] = rpx[y * w + x];
                    }
                }
                let ds_joint = scorer
                    .compute(&rs, &RgbSlice::new(&edited, w, h), None)
                    .expect("additivity rescore")
                    .score()
                    - base_score;
                add_tests.push(json!({
                    "a": [ax0, ay0, ax1, ay1], "b": [bx0, by0, bx1, by1],
                    "ds_a": delta_s[ka], "ds_b": delta_s[kb],
                    "ds_joint": ds_joint,
                    "residual": ds_joint - (delta_s[ka] + delta_s[kb]),
                }));
                if add_tests.len() >= MAX_ADD_PAIRS {
                    break 'outer;
                }
            }
        }
        s2["add_tests"] = json!(add_tests);

        // --- pyramid leak: re-run the field owner + attribution on the edited
        // image; report the fraction of |Δfield| mass OUTSIDE the edited rect.
        // A perfectly local map scores 0; pyramid spreading leaks mass into
        // coarse blocks outside R.
        let mut leak_lv = vec![Vec::<f64>::new(); n_levels];
        let mut leak_attr = Vec::<f64>::new();
        let leak_idx: Vec<usize> = {
            let pool = &size_rects[add_si];
            let stride = (pool.len() as f64 / MAX_LEAK_RECTS as f64).max(1.0);
            (0..MAX_LEAK_RECTS.min(pool.len()))
                .map(|k| pool[(k as f64 * stride) as usize])
                .collect()
        };
        for &k in &leak_idx {
            let (_, x0, y0, x1, y1) = rect_records[k];
            let mut edited = dpx.clone();
            for y in y0..y1 {
                for x in x0..x1 {
                    edited[y * w + x] = rpx[y * w + x];
                }
            }
            let ers = RgbSlice::new(&edited, w, h);
            let ext2 = zensim::research::extract(&req, &rs, &ers)
                .expect("leak re-extract");
            let fields2 = ext2.dvifm_fields().expect("leak fields");
            for (l, lv2) in fields2.levels().iter().enumerate() {
                let lv1 = &fields.levels()[l];
                let (nby_l, nbx_l) = lv2.block_grid();
                let scale = (1u64 << l) as usize;
                let (mut inside, mut outside) = (0.0f64, 0.0f64);
                for by in 0..nby_l {
                    for bx in 0..nbx_l {
                        let b = (by * nbx_l + bx) as usize;
                        let d = (lv2.eps_blocks()[b] - lv1.eps_blocks()[b]).abs();
                        // block's scale-0 footprint vs the edit rect
                        let (fx0, fy0) = (bx as usize * 5 * scale, by as usize * 5 * scale);
                        let (fx1, fy1) =
                            ((fx0 + 5 * scale).min(w), (fy0 + 5 * scale).min(h));
                        let overlaps =
                            fx0 < x1 && fx1 > x0 && fy0 < y1 && fy1 > y0;
                        if overlaps {
                            inside += d;
                        } else {
                            outside += d;
                        }
                    }
                }
                leak_lv[l].push(outside / (inside + outside + 1e-30));
            }
            let mut sess2 = Fused944Session::new();
            let sa2 = scorer
                .compute_with_ref_and_attribution(&rs, &pre, &ers, None, &mut sess2, 1)
                .expect("leak re-attribution");
            let (mut inside, mut outside) = (0.0f64, 0.0f64);
            let (a1, a2) = (attr_map.density(), sa2.attribution().density());
            for y in 0..h {
                for x in 0..w {
                    let d = (a2[y * w + x] - a1[y * w + x]).abs() as f64;
                    if x >= x0 && x < x1 && y >= y0 && y < y1 {
                        inside += d;
                    } else {
                        outside += d;
                    }
                }
            }
            leak_attr.push(outside / (inside + outside + 1e-30));
        }
        s2["leak_n"] = json!(leak_idx.len());
        let mean_or_null =
            |v: &[f64]| (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64);
        let max_or_null = |v: &[f64]| v.iter().copied().reduce(f64::max);
        s2["leak_dvifm_levels"] =
            json!(leak_lv.iter().map(|v| mean_or_null(v)).collect::<Vec<_>>());
        s2["leak_dvifm_levels_max"] =
            json!(leak_lv.iter().map(|v| max_or_null(v)).collect::<Vec<_>>());
        s2["leak_attr"] = json!(mean_or_null(&leak_attr));
        s2["leak_attr_max"] = json!(max_or_null(&leak_attr));

        // --- S3 metrics
        let ssim2_top = top_frac_set(&ssim2_blk, top_frac);
        let butter_top = top_frac_set(&butter_blk, top_frac);
        let agree: std::collections::HashSet<usize> =
            ssim2_top.intersection(&butter_top).copied().collect();
        let agree_ind: Vec<f64> = (0..nb)
            .map(|b| if agree.contains(&b) { 1.0 } else { 0.0 })
            .collect();
        let s3 = json!({
            "srocc_dvifm_ssim2": srocc(&dvifm_comb, &ssim2_blk),
            "srocc_attr_ssim2": srocc(&attr_blk, &ssim2_blk),
            "srocc_dvifm_butter": srocc(&dvifm_comb, &butter_blk),
            "srocc_attr_butter": srocc(&attr_blk, &butter_blk),
            "srocc_dvifm_levels_ssim2": (0..n_levels).map(|l| srocc(&dvifm_lvl[l], &ssim2_blk)).collect::<Vec<_>>(),
            "srocc_dvifm_levels_butter": (0..n_levels).map(|l| srocc(&dvifm_lvl[l], &butter_blk)).collect::<Vec<_>>(),
            "srocc_judges": srocc(&ssim2_blk, &butter_blk),
            "agree_n": agree.len(),
            "agree_frac": agree.len() as f64 / nb as f64,
            "srocc_dvifm_agree": srocc(&dvifm_comb, &agree_ind),
            "srocc_attr_agree": srocc(&attr_blk, &agree_ind),
            "precision_agree_dvifm": precision_at_k(&dvifm_comb, &agree_ind, agree.len()),
            "precision_agree_attr": precision_at_k(&attr_blk, &agree_ind, agree.len()),
            "kendall_dvifm_ssim2": kendall_tau(&dvifm_comb, &ssim2_blk),
            "kendall_attr_ssim2": kendall_tau(&attr_blk, &ssim2_blk),
        });

        // --- sidecar writes (f32 LE; row carries offsets)
        let mut block_vals = Vec::with_capacity(nb * 9);
        for b in 0..nb {
            block_vals.push(dvifm_comb[b]);
            for l in 0..n_levels {
                block_vals.push(dvifm_lvl[l][b]);
            }
            block_vals.push(attr_blk[b]);
            block_vals.push(ssim2_blk[b]);
            block_vals.push(butter_blk[b]);
        }
        let (blk_off, blk_len) = write_f32(&mut blocks_bin, &block_vals);
        // per-rect: size_idx, x0, y0, x1, y1, dvifm_comb, dvifm_l0..l4, attr,
        // refgain, sse, delta_s — 15 f32/rect; the aggregator derives
        // calibrated relative error by size/level from these.
        let mut rect_vals = Vec::with_capacity(rect_records.len() * 15);
        for k in 0..rect_records.len() {
            let (si, x0, y0, x1, y1) = rect_records[k];
            rect_vals.push(si as f64);
            rect_vals.push(x0 as f64);
            rect_vals.push(y0 as f64);
            rect_vals.push(x1 as f64);
            rect_vals.push(y1 as f64);
            rect_vals.push(pred_dvifm[k]);
            for lvl in &pred_dvifm_lvl {
                rect_vals.push(lvl[k]);
            }
            rect_vals.push(pred_attr[k]);
            rect_vals.push(pred_refgain[k]);
            rect_vals.push(pred_sse[k]);
            rect_vals.push(delta_s[k]);
        }
        let (rct_off, rct_len) = write_f32(&mut rects_bin, &rect_vals);

        let rec = json!({
            "pair": pi,
            "ref": row.ref_path, "dist": row.dist_path,
            "leg": row.leg, "codec": row.codec, "q": row.q,
            "band": row.band, "group": row.group, "ref_basename": row.ref_basename,
            "w": w, "h": h, "nbx": nbx, "nby": nby,
            "n_rects": rect_records.len(),
            "base_score": base_score,
            "ssim2_score_repl": s2f.score, "ssim2_score_ref": ssim2_ref_score,
            "ssim2_parity_abs": parity, "butter_score": bres.score,
            "unsupported_ids": sa.unsupported_feature_ids(),
            "corruption_gate": sa.has_corruption_gate(),
            "dvifm_f1": fields.levels().iter().map(|l| l.f1()).collect::<Vec<_>>(),
            "s2": s2, "s3": s3,
            "blocks_off": blk_off, "blocks_len": blk_len,
            "rects_off": rct_off, "rects_len": rct_len,
        });
        serde_json::to_writer(&mut jsonl, &rec).expect("write row");
        jsonl.write_all(b"\n").unwrap();
        jsonl.flush().unwrap();
        n_done += 1;
        if n_done % 10 == 0 {
            eprintln!(
                "  {n_done}/{} pairs  {:.0}s elapsed  parity_max={parity_max:.2e}",
                picked.len(),
                t_start.elapsed().as_secs_f64()
            );
        }
    }

    let summary = json!({
        "study": "dvifm-steer-2026-09-20",
        "pairs_tsv": pairs_tsv, "pairs_tsv_sha256": pairs_sha,
        "bake": bake, "bake_sha256": bake_sha,
        "n_manifest": rows.len(), "n_sampled": picked.len(),
        "n_done": n_done, "n_skipped_size": n_skipped_size,
        "rect_sizes": [16usize, 32, 64, 128, 256],
        "max_interventions": max_interventions,
        "top_frac": top_frac,
        "ssim2_parity_max": parity_max,
        "elapsed_s": t_start.elapsed().as_secs_f64(),
    });
    std::fs::write(
        out_dir.join("summary.json"),
        serde_json::to_string_pretty(&summary).unwrap(),
    )
    .expect("summary.json");
    eprintln!("done: {n_done} pairs in {:.0}s", t_start.elapsed().as_secs_f64());
}
