//! Comprehensive metric-eval report sections + a self-contained HTML
//! renderer — the pieces that turn `bake_verdict` from a rank+dial
//! evaluator into the single "default metric evaluation" command that
//! runs every eval/stat/bucket we have historically run on a bake and
//! emits both console (markdown) and a big self-contained HTML report.
//!
//! Sections provided here (each appends markdown, some with inline-SVG
//! charts, so the console and the HTML share one source of truth):
//!   - **severity-ramp monotonicity** — port of
//!     `scripts/hdr/severity_ramp_monotonicity.py`: for each
//!     `(ref, dist_type)` 5-level severity ramp, a correct dial is
//!     non-increasing as severity rises.
//!   - **per-zone bucket vs a reference bake** — the §8.20 dial-space
//!     zone panel: bucket rows by the reference bake's dial in `step`-pt
//!     zones and report the candidate's mean-Δ / RMSE / rank per zone.
//!   - **legacy 4-band CID22 cuts** — the 2023 CID22 Table-5 partition
//!     (B0<50 / B1 50-65 / B2 65-90 / B3 ≥90), reported alongside the
//!     10-band grid per the CLAUDE.md per-band mandate.
//!   - **`markdown_to_html`** — a self-contained, theme-aware GFM-subset
//!     renderer (tables, headers with a table-of-contents, inline SVG
//!     passthrough) so the whole report becomes one browsable file with
//!     no external assets.
//!
//! ALL correlation math routes through [`crate::panel`] (a thin re-export
//! of the canonical `zenstats::panel`) — nothing statistical is
//! re-implemented here.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::panel::compute_panel;

/// Distortion types whose `dist_param` is signed (U-shaped score by
/// design — over/under-shoot around the reference). Excluded from the
/// monotone denominator, reported separately. Matches the SIGNED set in
/// `severity_ramp_monotonicity.py`.
pub const SIGNED_DIST_TYPES: [u32; 3] = [7, 18, 25];

/// The two half-ramps of a signed (U-shaped) distortion, as `level` sequences
/// ordered from the identity (minimum |dist_param|) OUTWARD. A signed type
/// sweeps its parameter positive → 0 (identity) → negative, so quality is
/// U-shaped in `level` but MONOTONE in |dist_param| — folding at the identity
/// turns each U into two proper severity ramps (quality must fall as the
/// distortion magnitude rises in either direction). The generator's per-level
/// `dist_param` is fixed by kadis-distort, so we encode the |dist_param|
/// ordering directly rather than parsing `knob_tuple_json`:
///   d7  color_saturate_hsv : params +.40 +.20 +.10  .00 −.40  (identity L4)
///   d18 mean_shift         : params +.15 +.08  .00 −.08 −.15  (identity L3)
///   d25 contrast           : params +.30 +.15  .00 −.40 −.60  (identity L3)
/// Returns `[positive_arm, negative_arm]`, each a level sequence starting at
/// the identity. Keep in lockstep with [`SIGNED_DIST_TYPES`].
fn signed_fold_arms(ty: u32) -> Option<[&'static [u32]; 2]> {
    match ty {
        7 => Some([&[4, 3, 2, 1], &[4, 5]]),
        18 => Some([&[3, 2, 1], &[3, 4, 5]]),
        25 => Some([&[3, 2, 1], &[3, 4, 5]]),
        _ => None,
    }
}

/// The identity level (minimum |dist_param|) for a signed type. Its dial
/// should be ≈100 (the undistorted reference); a low value flags either a
/// non-identity param=0 image or a metric calibration gap on that distortion.
fn signed_identity_level(ty: u32) -> Option<u32> {
    match ty {
        7 => Some(4),
        18 | 25 => Some(3),
        _ => None,
    }
}

// ============================================================================
// Severity-ramp monotonicity
// ============================================================================

/// Per-type ramp tally: `(dist_type, monotone_fraction, n_ramps)`.
#[derive(Debug, Clone)]
pub struct RampType {
    pub dist_type: u32,
    pub monotone_frac: f64,
    pub n: usize,
}

/// Result of the severity-ramp monotonicity check.
#[derive(Debug, Clone)]
pub struct RampStats {
    /// Ramps counted (5-level, non-signed).
    pub n_ramps: usize,
    /// Signed/U-shaped ramps folded at their identity (count of source ramps).
    pub n_signed: usize,
    /// Fraction of ramps non-increasing within `eps` slack (the headline;
    /// UNSIGNED ramps only — signed folded arms are reported separately so
    /// the 90% gate stays comparable to history).
    pub pct_monotone: f64,
    /// Fraction strictly decreasing (every step down).
    pub pct_strict: f64,
    /// Mean worst forward-inversion magnitude (dial pts) over the
    /// non-monotone ramps. 0 when all ramps are monotone.
    pub mean_worst_inv: f64,
    /// Per-type monotone fraction, worst first (unsigned types).
    pub per_type: Vec<RampType>,
    /// Tie slack used (dial points).
    pub eps: f64,
    /// Total folded half-ramps from signed types (2 per source ramp).
    pub n_signed_arms: usize,
    /// Fraction of folded signed half-ramps non-increasing within `eps`.
    pub pct_signed_monotone: f64,
    /// Per signed-type folded-arm monotone fraction.
    pub signed_per_type: Vec<RampType>,
    /// Per signed-type mean identity-level dial (should be ≈100):
    /// `(dist_type, mean_identity_dial, n)`.
    pub signed_identity: Vec<(u32, f64, usize)>,
}

/// Severity-ramp monotonicity. `q` encodes `dist_type * 10 + level`
/// (level 1..5). For each `(image, dist_type)` ramp with all 5 levels,
/// the dial must be non-increasing (higher severity → lower quality) as
/// level rises, within `eps` dial-points of tie slack. Signed types
/// ([`SIGNED_DIST_TYPES`]) are U-shaped in level, so instead of discarding
/// them they are FOLDED at their identity (min |dist_param|) into two
/// half-ramps each (see [`signed_fold_arms`]) and reported separately — the
/// unsigned headline stays comparable to history while the signed types stay
/// relevant (with an identity-dial fidelity check).
///
/// `image` is any stable per-source key (basename); `dial` is the bake's
/// score for the same row.
pub fn severity_ramp(image: &[String], q: &[f64], dial: &[f64], eps: f64) -> RampStats {
    // Group dial by (image, dist_type) → {level: dial}.
    let mut ramps: BTreeMap<(String, u32), BTreeMap<u32, f64>> = BTreeMap::new();
    for i in 0..dial.len() {
        let qi = q[i] as i64;
        if qi < 0 {
            continue;
        }
        let ty = (qi / 10) as u32;
        let lv = (qi % 10) as u32;
        ramps
            .entry((image[i].clone(), ty))
            .or_default()
            .insert(lv, dial[i]);
    }
    let mut n_ramps = 0usize;
    let mut n_signed = 0usize;
    let mut mono = 0usize;
    let mut strict = 0usize;
    let mut inv_mags: Vec<f64> = Vec::new();
    // per-type: [monotone, total]
    let mut per_type: BTreeMap<u32, [usize; 2]> = BTreeMap::new();
    // signed folded-arm tallies (kept separate from the unsigned headline).
    let mut n_signed_arms = 0usize;
    let mut signed_mono = 0usize;
    let mut signed_per_type: BTreeMap<u32, [usize; 2]> = BTreeMap::new();
    let mut signed_identity: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
    // Check one monotone (non-increasing within eps) sequence; return (ok, strict).
    let check = |seq: &[f64]| -> (bool, bool, f64) {
        let diffs: Vec<f64> = seq.windows(2).map(|w| w[1] - w[0]).collect();
        let ok = diffs.iter().all(|&d| d <= eps);
        let st = diffs.iter().all(|&d| d < 0.0);
        let worst = diffs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (ok, st, worst)
    };
    for ((_img, ty), levels) in &ramps {
        if levels.len() < 5 {
            continue;
        }
        if let Some(arms) = signed_fold_arms(*ty) {
            // Fold the U at its identity: each arm is a half-ramp from the
            // identity outward and must be non-increasing as |param| rises.
            n_signed += 1;
            for arm in arms {
                let seq: Vec<f64> = arm
                    .iter()
                    .filter_map(|lv| levels.get(lv).copied())
                    .collect();
                if seq.len() < 2 {
                    continue;
                }
                let (ok, _st, _worst) = check(&seq);
                n_signed_arms += 1;
                if ok {
                    signed_mono += 1;
                }
                let e = signed_per_type.entry(*ty).or_default();
                e[0] += ok as usize;
                e[1] += 1;
            }
            if let Some(idlv) = signed_identity_level(*ty)
                && let Some(&d) = levels.get(&idlv)
            {
                signed_identity.entry(*ty).or_default().push(d);
            }
            continue;
        }
        // Unsigned severity ramp: sorted-by-level dial sequence.
        let seq: Vec<f64> = levels.values().copied().collect();
        let (ok, st, worst) = check(&seq);
        n_ramps += 1;
        if ok {
            mono += 1;
        }
        if st {
            strict += 1;
        }
        let e = per_type.entry(*ty).or_default();
        e[0] += ok as usize;
        e[1] += 1;
        if !ok {
            inv_mags.push(worst);
        }
    }
    let to_types = |m: &BTreeMap<u32, [usize; 2]>| -> Vec<RampType> {
        let mut v: Vec<RampType> = m
            .iter()
            .map(|(&ty, c)| RampType {
                dist_type: ty,
                monotone_frac: if c[1] > 0 {
                    c[0] as f64 / c[1] as f64
                } else {
                    f64::NAN
                },
                n: c[1],
            })
            .collect();
        v.sort_by(|a, b| {
            a.monotone_frac
                .partial_cmp(&b.monotone_frac)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        v
    };
    let signed_identity_v: Vec<(u32, f64, usize)> = signed_identity
        .iter()
        .map(|(&ty, v)| (ty, v.iter().sum::<f64>() / v.len().max(1) as f64, v.len()))
        .collect();
    RampStats {
        n_ramps,
        n_signed,
        pct_monotone: if n_ramps > 0 {
            mono as f64 / n_ramps as f64
        } else {
            f64::NAN
        },
        pct_strict: if n_ramps > 0 {
            strict as f64 / n_ramps as f64
        } else {
            f64::NAN
        },
        mean_worst_inv: if inv_mags.is_empty() {
            0.0
        } else {
            inv_mags.iter().sum::<f64>() / inv_mags.len() as f64
        },
        per_type: to_types(&per_type),
        eps,
        n_signed_arms,
        pct_signed_monotone: if n_signed_arms > 0 {
            signed_mono as f64 / n_signed_arms as f64
        } else {
            f64::NAN
        },
        signed_per_type: to_types(&signed_per_type),
        signed_identity: signed_identity_v,
    }
}

/// Render the severity-ramp section as markdown + an inline-SVG per-type
/// monotonicity bar chart.
pub fn severity_ramp_section(stats: &RampStats, grid_label: &str) -> String {
    let mut s = String::new();
    s.push_str("\n## Severity-ramp monotonicity (distortion dial)\n\n");
    s.push_str(&format!(
        "Grid: `{}` — {} unsigned five-level ramps; {} signed U-shaped ramps \
         (types {:?}) folded at their identity into {} half-ramps (reported below).\n\n",
        grid_label, stats.n_ramps, stats.n_signed, SIGNED_DIST_TYPES, stats.n_signed_arms
    ));
    s.push_str("| metric | value | gate | pass |\n|---|--:|---|:--:|\n");
    let mono_pass = stats.pct_monotone >= 0.90;
    let _ = writeln!(
        s,
        "| monotone (ε={:.1}pt slack) | {:.1}% | ≥ 90% | {} |",
        stats.eps,
        100.0 * stats.pct_monotone,
        if mono_pass { "✓" } else { "✗" }
    );
    let _ = writeln!(
        s,
        "| strictly decreasing | {:.1}% | — | |",
        100.0 * stats.pct_strict
    );
    let _ = writeln!(
        s,
        "| mean worst-inversion | {:.2} dial pts | — | |",
        stats.mean_worst_inv
    );
    s.push('\n');
    // Inline SVG: per-type monotone fraction (worst 12).
    let show: Vec<&RampType> = stats.per_type.iter().take(12).collect();
    if !show.is_empty() {
        let labels: Vec<String> = show.iter().map(|t| format!("d{}", t.dist_type)).collect();
        let values: Vec<f64> = show.iter().map(|t| 100.0 * t.monotone_frac).collect();
        s.push_str(&svg_bars(
            "Per-type monotone % (worst first)",
            &labels,
            &values,
            0.0,
            100.0,
            90.0,
            true,
        ));
        s.push('\n');
        s.push_str("| dist_type | monotone % | n |\n|---|--:|--:|\n");
        for t in &show {
            let _ = writeln!(
                s,
                "| d{} | {:.0}% | {} |",
                t.dist_type,
                100.0 * t.monotone_frac,
                t.n
            );
        }
        s.push('\n');
    }
    // Signed U-shaped types, folded at their identity into half-ramps.
    if stats.n_signed_arms > 0 {
        s.push_str("\n**Signed U-shaped types (folded at identity — |dist_param| ramps):**\n\n");
        let _ = writeln!(
            s,
            "Overall folded-arm monotone: **{:.1}%** ({} half-ramps). A signed \
             distortion sweeps its parameter +→0→−, so quality is U-shaped in level \
             but must fall monotonically as |dist_param| rises from the identity.\n",
            100.0 * stats.pct_signed_monotone,
            stats.n_signed_arms
        );
        s.push_str("| dist_type | folded monotone % | half-ramps | identity dial (→100) |\n");
        s.push_str("|---|--:|--:|--:|\n");
        for t in &stats.signed_per_type {
            let ident = stats
                .signed_identity
                .iter()
                .find(|(ty, _, _)| *ty == t.dist_type)
                .map(|(_, d, _)| *d);
            let ident_s = match ident {
                Some(d) => format!("{d:.1}"),
                None => "—".to_string(),
            };
            let _ = writeln!(
                s,
                "| d{} | {:.0}% | {} | {} |",
                t.dist_type,
                100.0 * t.monotone_frac,
                t.n,
                ident_s
            );
        }
        s.push('\n');
    }
    s.push_str(
        "_A correct dial is non-increasing as distortion severity rises (level 1→5). \
         Signed U-shaped types are folded at their identity (min |dist_param|) into two \
         half-ramps each — relevant, not discarded — with an identity-dial fidelity check \
         (should be ≈100). `mean worst-inversion` is the mean over non-monotone UNSIGNED \
         ramps of the largest wrong-direction step._\n",
    );
    s
}

// ============================================================================
// Per-zone bucket comparison vs a reference bake (§8.20)
// ============================================================================

/// One zone bucket of the dial-space comparison.
#[derive(Debug, Clone)]
pub struct ZoneRow {
    pub lo: f64,
    pub hi: f64,
    pub n: usize,
    /// mean(candidate − reference) in the bucket.
    pub mean_delta: f64,
    /// RMSE(candidate, reference) in the bucket.
    pub rmse: f64,
    /// SROCC(candidate, reference) in the bucket (NaN when n < 4 or the
    /// bucket range is degenerate).
    pub srocc: f64,
    /// candidate p5 / p95 in the bucket (dial reach within the zone).
    pub cand_p5: f64,
    pub cand_p95: f64,
}

/// Result of the per-zone bucket comparison.
#[derive(Debug, Clone)]
pub struct ZoneStats {
    pub rows: Vec<ZoneRow>,
    pub agg_srocc: f64,
    pub agg_rmse: f64,
    pub agg_n: usize,
    pub step: f64,
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() {
        return f64::NAN;
    }
    (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
}

fn pctile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = (p / 100.0) * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let f = rank - lo as f64;
        sorted[lo] * (1.0 - f) + sorted[hi] * f
    }
}

/// Bucket rows by the `reference` bake's dial into `step`-point zones over
/// [0, 100] and report, per zone, the `candidate`'s agreement with the
/// reference: mean-Δ, RMSE, and within-zone SROCC. `candidate` and
/// `reference` are the two bakes' scores on the SAME rows (e.g. the dial
/// grid). This is the §8.20 "bucket stats per 5 points of B" panel.
pub fn zone_buckets(candidate: &[f64], reference: &[f64], step: f64) -> ZoneStats {
    let n = candidate.len().min(reference.len());
    let mut rows = Vec::new();
    let n_zones = (100.0 / step).ceil() as usize;
    for z in 0..n_zones {
        let lo = z as f64 * step;
        let hi = lo + step;
        let idxs: Vec<usize> = (0..n)
            .filter(|&i| {
                reference[i].is_finite()
                    && candidate[i].is_finite()
                    && reference[i] >= lo
                    && (reference[i] < hi || (z == n_zones - 1 && reference[i] <= hi))
            })
            .collect();
        if idxs.is_empty() {
            rows.push(ZoneRow {
                lo,
                hi,
                n: 0,
                mean_delta: f64::NAN,
                rmse: f64::NAN,
                srocc: f64::NAN,
                cand_p5: f64::NAN,
                cand_p95: f64::NAN,
            });
            continue;
        }
        let cand: Vec<f64> = idxs.iter().map(|&i| candidate[i]).collect();
        let refe: Vec<f64> = idxs.iter().map(|&i| reference[i]).collect();
        let mean_delta =
            cand.iter().zip(&refe).map(|(c, r)| c - r).sum::<f64>() / cand.len() as f64;
        let srocc = if cand.len() >= 4 {
            compute_panel(&cand, &refe).srocc
        } else {
            f64::NAN
        };
        let mut cs = cand.clone();
        cs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        rows.push(ZoneRow {
            lo,
            hi,
            n: idxs.len(),
            mean_delta,
            rmse: rmse(&cand, &refe),
            srocc,
            cand_p5: pctile(&cs, 5.0),
            cand_p95: pctile(&cs, 95.0),
        });
    }
    let cand_all: Vec<f64> = (0..n).map(|i| candidate[i]).collect();
    let ref_all: Vec<f64> = (0..n).map(|i| reference[i]).collect();
    let agg_srocc = if n >= 4 {
        compute_panel(&cand_all, &ref_all).srocc
    } else {
        f64::NAN
    };
    ZoneStats {
        rows,
        agg_srocc,
        agg_rmse: rmse(&cand_all, &ref_all),
        agg_n: n,
        step,
    }
}

/// Render the per-zone bucket section as markdown + an inline-SVG mean-Δ
/// chart.
pub fn zone_bucket_section(
    stats: &ZoneStats,
    cand_label: &str,
    ref_label: &str,
    grid_label: &str,
) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "\n## Per-zone dial agreement — `{cand_label}` vs reference `{ref_label}`\n\n"
    ));
    s.push_str(&format!(
        "Grid: `{}` — {} rows bucketed by **{}**'s dial in {:.0}-pt zones. Aggregate: \
         SROCC(cand,ref) = **{:.4}**, RMSE = **{:.2}** dial pts (n={}).\n\n",
        grid_label,
        stats.agg_n,
        ref_label,
        stats.step,
        stats.agg_srocc,
        stats.agg_rmse,
        stats.agg_n
    ));
    s.push_str(&format!(
        "| {ref_label} zone | n | mean Δ (cand−ref) | RMSE | SROCC | cand p5..p95 |\n\
         |---|--:|--:|--:|--:|---|\n"
    ));
    for r in &stats.rows {
        if r.n == 0 {
            let _ = writeln!(s, "| [{:.0},{:.0}) | 0 | — | — | — | — |", r.lo, r.hi);
            continue;
        }
        let noisy = if r.n < 30 { " ⚠" } else { "" };
        let srocc_s = if r.srocc.is_nan() {
            "n/a".to_string()
        } else {
            format!("{:.3}", r.srocc)
        };
        let _ = writeln!(
            s,
            "| [{:.0},{:.0}){} | {} | {:+.2} | {:.2} | {} | {:.1}..{:.1} |",
            r.lo, r.hi, noisy, r.n, r.mean_delta, r.rmse, srocc_s, r.cand_p5, r.cand_p95
        );
    }
    s.push('\n');
    let labels: Vec<String> = stats
        .rows
        .iter()
        .filter(|r| r.n > 0)
        .map(|r| format!("{:.0}", r.lo))
        .collect();
    let values: Vec<f64> = stats
        .rows
        .iter()
        .filter(|r| r.n > 0)
        .map(|r| r.mean_delta)
        .collect();
    if !values.is_empty() {
        let vmax = values
            .iter()
            .cloned()
            .fold(0.0f64, |m, v| m.max(v.abs()))
            .max(5.0);
        s.push_str(&svg_bars(
            &format!("mean Δ (cand − ref) per {ref_label} zone — 0 = perfect agreement"),
            &labels,
            &values,
            -vmax,
            vmax,
            0.0,
            false,
        ));
        s.push('\n');
    }
    s.push_str(
        "_A candidate that tracks the reference across the full dial has mean Δ ≈ 0 and \
         high within-zone SROCC in every zone. A collapsed top zone (large negative Δ at \
         high reference dial) means the candidate can't reach the reference's near-lossless \
         ceiling — the classic sparse-high-anchor failure. ⚠ marks n < 30 (noisy)._\n",
    );
    s
}

// ============================================================================
// Legacy 4-band CID22 cuts (2023 Table 5 partition)
// ============================================================================

/// Render the legacy 4-band CID22 cuts (B0<50 / B1 50-65 / B2 65-90 /
/// B3 ≥90) on a `[0,1]`-normalized `human_score` corpus (CID22). Reported
/// alongside the 10-band grid per the CLAUDE.md per-band mandate.
pub fn four_band_section(scores: &[f64], humans: &[f64]) -> String {
    // Cuts on the 0..100 MCOS scale → 0..1 normalized.
    let cuts: [(&str, f64, f64); 4] = [
        ("B0 <50", 0.00, 0.50),
        ("B1 50-65", 0.50, 0.65),
        ("B2 65-90", 0.65, 0.90),
        ("B3 ≥90", 0.90, 1.0001),
    ];
    let mut s = String::new();
    s.push_str("\n### CID22 legacy 4-band cuts (2023 paper Table 5)\n\n");
    s.push_str(
        "| Band | n | SROCC | PLCC | KROCC | PWRC | Z-RMSE |\n|---|--:|--:|--:|--:|--:|--:|\n",
    );
    for (label, lo, hi) in cuts {
        let idxs: Vec<usize> = humans
            .iter()
            .enumerate()
            .filter_map(|(i, &h)| (h >= lo && h < hi).then_some(i))
            .collect();
        if idxs.len() < 4 {
            let _ = writeln!(
                s,
                "| {label} | {} | n/a | n/a | n/a | n/a | n/a |",
                idxs.len()
            );
            continue;
        }
        let h: Vec<f64> = idxs.iter().map(|&i| humans[i]).collect();
        let sc: Vec<f64> = idxs.iter().map(|&i| scores[i]).collect();
        let p = compute_panel(&sc, &h);
        let noisy = if idxs.len() < 30 { " ⚠" } else { "" };
        let _ = writeln!(
            s,
            "| {label}{noisy} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.3} |",
            idxs.len(),
            p.srocc,
            p.plcc,
            p.krocc,
            p.pwrc,
            p.z_rmse
        );
    }
    s.push_str("\n_Kept for comparison with the 2023 CID22 paper; the 10-band grid above is the primary release gate._\n");
    s
}

// ============================================================================
// Corruption gate (negative-tail ranking)
// ============================================================================

/// Region tokens used in the corruption-grid entry names, in the position
/// right after the distortion family.
const CORRUPTION_REGIONS: [&str; 6] = ["whole", "frac2", "frac4", "sq64", "sq16", "sq8"];

/// Result of the corruption-gate check.
#[derive(Debug, Clone)]
pub struct CorruptionStats {
    /// Corruption entries with a matching q20 anchor.
    pub n_triples: usize,
    /// Fraction where score(corruption) < score(q20) — the gate.
    pub pass_q20: f64,
    /// Fraction where score(corruption) < score(q10) — a stricter anchor.
    pub pass_q10: f64,
    /// Per-family `(family, pass_q20_fraction, n)`, worst first.
    pub per_family: Vec<(String, f64, usize)>,
}

fn corruption_family(key: &str) -> String {
    let toks: Vec<&str> = key.split("__").collect();
    // family is the token immediately before the region token.
    for (i, t) in toks.iter().enumerate() {
        if CORRUPTION_REGIONS.contains(t) && i > 0 {
            return toks[i - 1].to_string();
        }
    }
    // fall back to the second token (after the ref name).
    toks.get(1)
        .map(|s| s.to_string())
        .unwrap_or_else(|| key.to_string())
}

/// Corruption gate: for each corruption entry, a structurally-broken decode
/// must rank BELOW an honestly-lossy q20 encode — `score(corruption) <
/// score(q20)`. `label` is the `entry` column
/// (`<ref>__<fam>__<region>__<sev>__{corruption,q20,q10}`); `dial` is the
/// bake's score for the same row. Higher dial = better quality, so the
/// gate is a strict `<`.
pub fn corruption_gate(label: &[String], dial: &[f64]) -> CorruptionStats {
    // key = entry minus the trailing __{corruption,q20,q10} → {kind: dial}.
    let mut groups: BTreeMap<String, BTreeMap<&str, f64>> = BTreeMap::new();
    for (i, l) in label.iter().enumerate() {
        for kind in ["corruption", "q20", "q10"] {
            let suffix = format!("__{kind}");
            if let Some(key) = l.strip_suffix(&suffix) {
                groups
                    .entry(key.to_string())
                    .or_default()
                    .insert(kind, dial[i]);
                break;
            }
        }
    }
    let mut n_triples = 0usize;
    let mut pass20 = 0usize;
    let mut n10 = 0usize;
    let mut pass10 = 0usize;
    // per family: [pass20, total20]
    let mut fam: BTreeMap<String, [usize; 2]> = BTreeMap::new();
    for (key, kinds) in &groups {
        let corr = kinds.get("corruption");
        if let (Some(&c), Some(&q20)) = (corr, kinds.get("q20")) {
            n_triples += 1;
            let ok = c < q20;
            if ok {
                pass20 += 1;
            }
            let e = fam.entry(corruption_family(key)).or_default();
            e[0] += ok as usize;
            e[1] += 1;
        }
        if let (Some(&c), Some(&q10)) = (corr, kinds.get("q10")) {
            n10 += 1;
            if c < q10 {
                pass10 += 1;
            }
        }
    }
    let mut per_family: Vec<(String, f64, usize)> = fam
        .into_iter()
        .map(|(k, c)| {
            (
                k,
                if c[1] > 0 {
                    c[0] as f64 / c[1] as f64
                } else {
                    f64::NAN
                },
                c[1],
            )
        })
        .collect();
    per_family.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    CorruptionStats {
        n_triples,
        pass_q20: if n_triples > 0 {
            pass20 as f64 / n_triples as f64
        } else {
            f64::NAN
        },
        pass_q10: if n10 > 0 {
            pass10 as f64 / n10 as f64
        } else {
            f64::NAN
        },
        per_family,
    }
}

/// Render the corruption-gate section as markdown + an inline-SVG per-family
/// pass-rate chart.
pub fn corruption_gate_section(stats: &CorruptionStats, grid_label: &str, title: &str) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "\n## {title}\n");
    s.push_str(&format!(
        "Grid: `{}` — {} corruption entries. A structurally-broken decode MUST rank below an \
         honestly-lossy encode: `score(corruption) < score(q20)`.\n\n",
        grid_label, stats.n_triples
    ));
    s.push_str("| gate | pass % | note | pass |\n|---|--:|---|:--:|\n");
    let _ = writeln!(
        s,
        "| corruption < q20 (honest low-quality) | {:.1}% | negative-tail ranking | {} |",
        100.0 * stats.pass_q20,
        if stats.pass_q20 >= 0.95 { "✓" } else { "✗" }
    );
    let _ = writeln!(
        s,
        "| corruption < q10 (stricter anchor) | {:.1}% | should be near-total | |",
        100.0 * stats.pass_q10
    );
    s.push('\n');
    let show: Vec<&(String, f64, usize)> = stats.per_family.iter().take(12).collect();
    if !show.is_empty() {
        let labels: Vec<String> = show.iter().map(|(f, _, _)| f.clone()).collect();
        let values: Vec<f64> = show.iter().map(|(_, f, _)| 100.0 * f).collect();
        s.push_str(&svg_bars(
            "Per-family corruption<q20 pass % (worst first)",
            &labels,
            &values,
            0.0,
            100.0,
            95.0,
            true,
        ));
        s.push('\n');
        s.push_str("| family | pass % | n |\n|---|--:|--:|\n");
        for (f, frac, n) in &show {
            let _ = writeln!(s, "| {f} | {:.0}% | {n} |", 100.0 * frac);
        }
        s.push('\n');
    }
    s.push_str(
        "_The negative tail the regression-test use case depends on: a broken decode must not \
         outscore an honest q20. A low pass rate here means the metric can be fooled by \
         structural corruption. (Butteraugli-max historically wins this gate 2-4× over MLP \
         bakes — see the corruption-corpus note.)_\n",
    );
    s
}

// ============================================================================
// Inline SVG bar chart (self-contained, theme-aware via currentColor)
// ============================================================================

/// A horizontal bar chart as an inline `<svg>` string. `good_high` colors
/// bars green when the value passes `threshold` in the good direction
/// (values ≥ threshold when `good_high`, else ≤ threshold); `false` +
/// `threshold=0` gives a diverging chart centered on zero (used for
/// mean-Δ). Values are clamped into `[vmin, vmax]`.
pub fn svg_bars(
    title: &str,
    labels: &[String],
    values: &[f64],
    vmin: f64,
    vmax: f64,
    threshold: f64,
    good_high: bool,
) -> String {
    let n = labels.len().min(values.len());
    if n == 0 {
        return String::new();
    }
    let row_h = 20i32;
    let top = 26i32;
    let left = 62i32;
    let bar_w = 360i32;
    let width = left + bar_w + 60;
    let height = top + n as i32 * row_h + 8;
    let span = (vmax - vmin).max(1e-9);
    let x_of = |v: f64| -> f64 { left as f64 + (v.clamp(vmin, vmax) - vmin) / span * bar_w as f64 };
    let zero_x = x_of(threshold.clamp(vmin, vmax));
    let mut s = String::new();
    let _ = write!(
        s,
        "<svg viewBox=\"0 0 {width} {height}\" width=\"{width}\" height=\"{height}\" \
         role=\"img\" class=\"zsv-chart\" xmlns=\"http://www.w3.org/2000/svg\">"
    );
    let _ = write!(
        s,
        "<text x=\"6\" y=\"16\" class=\"zsv-title\">{}</text>",
        html_escape(title)
    );
    // baseline / threshold line
    let _ = write!(
        s,
        "<line x1=\"{zx:.1}\" y1=\"{y0}\" x2=\"{zx:.1}\" y2=\"{y1}\" class=\"zsv-axis\"/>",
        zx = zero_x,
        y0 = top - 4,
        y1 = top + n as i32 * row_h
    );
    for i in 0..n {
        let y = top + i as i32 * row_h;
        let v = values[i];
        let pass = if good_high {
            v >= threshold
        } else {
            v <= threshold
        };
        let cls = if !good_high && threshold == 0.0 {
            // diverging: color by sign
            if v >= 0.0 { "zsv-pos" } else { "zsv-neg" }
        } else if pass {
            "zsv-good"
        } else {
            "zsv-bad"
        };
        let xv = x_of(v);
        let (bx, bw) = if xv >= zero_x {
            (zero_x, xv - zero_x)
        } else {
            (xv, zero_x - xv)
        };
        let val = format!("{v:.1}");
        let _ = write!(
            s,
            "<text x=\"{lx}\" y=\"{ty}\" class=\"zsv-lbl\">{lbl}</text>\
             <rect x=\"{bx:.1}\" y=\"{ry}\" width=\"{bw:.1}\" height=\"{bh}\" class=\"{cls}\"/>\
             <text x=\"{vx:.1}\" y=\"{ty}\" class=\"zsv-val\">{val}</text>",
            lx = 6,
            ty = y + 14,
            lbl = html_escape(&labels[i]),
            bx = bx,
            ry = y + 3,
            bw = bw.max(0.5),
            bh = row_h - 6,
            cls = cls,
            vx = left as f64 + bar_w as f64 + 4.0,
        );
    }
    s.push_str("</svg>");
    s
}

// ============================================================================
// Markdown → self-contained HTML
// ============================================================================

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Inline markdown → HTML for one text run: `**bold**`, `` `code` ``,
/// `_em_`. Everything else escaped. Raw `<...>` runs (SVG) are handled by
/// the block layer, not here.
fn inline_md(s: &str) -> String {
    // Tokenize on backticks first so ** and _ inside code are literal.
    let mut out = String::new();
    let mut chars = s.chars().peekable();
    let mut buf = String::new();
    let flush = |buf: &mut String, out: &mut String| {
        // apply ** and _ on the escaped buffer
        let esc = html_escape(buf);
        let bolded = apply_wrap(&esc, "**", "strong");
        let emmed = apply_wrap(&bolded, "_", "em");
        out.push_str(&emmed);
        buf.clear();
    };
    while let Some(c) = chars.next() {
        if c == '`' {
            flush(&mut buf, &mut out);
            let mut code = String::new();
            for cc in chars.by_ref() {
                if cc == '`' {
                    break;
                }
                code.push(cc);
            }
            out.push_str("<code>");
            out.push_str(&html_escape(&code));
            out.push_str("</code>");
        } else {
            buf.push(c);
        }
    }
    flush(&mut buf, &mut out);
    out
}

/// Wrap paired `delim` runs in `<tag>...</tag>`. Odd trailing delims are
/// left literal.
fn apply_wrap(s: &str, delim: &str, tag: &str) -> String {
    let parts: Vec<&str> = s.split(delim).collect();
    // Need ≥3 parts and an odd count (fully-paired delimiters) to wrap;
    // an even count means an unpaired trailing delim — leave literal.
    if parts.len() < 3 || parts.len().is_multiple_of(2) {
        return s.to_string();
    }
    let mut out = String::new();
    for (i, p) in parts.iter().enumerate() {
        // Even indices are the runs between/around pairs (literal); odd
        // indices are the wrapped spans. The last index is even (odd
        // part-count) so trailing text stays literal.
        if i.is_multiple_of(2) {
            out.push_str(p);
        } else {
            let _ = write!(out, "<{tag}>{p}</{tag}>");
        }
    }
    out
}

fn slugify(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

/// Cell class for a table cell: green for ✓/pass, red for ✗/fail.
fn cell_class(cell: &str) -> &'static str {
    let t = cell.trim();
    if t == "✓" || t.contains("PASS") {
        "zc-good"
    } else if t == "✗" || t.contains("FAIL") {
        "zc-bad"
    } else {
        ""
    }
}

/// Render a GFM-subset markdown document to a single self-contained,
/// theme-aware HTML string: `#`/`##`/`###` headers (with anchors + a
/// generated table of contents), pipe tables (with pass/fail cell
/// coloring), `---` rules, blank-line paragraphs, inline `**`/`_`/`` ` ``,
/// and raw `<svg>`/`<...>` blocks passed through verbatim. No external
/// assets, no scripts.
pub fn markdown_to_html(md: &str, title: &str) -> String {
    let lines: Vec<&str> = md.lines().collect();
    let mut body = String::new();
    let mut toc: Vec<(u8, String, String)> = Vec::new();
    let mut i = 0usize;
    let mut in_svg = false;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim_start();

        // Raw HTML / SVG passthrough (possibly multi-line).
        if in_svg || trimmed.starts_with('<') {
            body.push_str(line);
            body.push('\n');
            if line.contains("</svg>") {
                in_svg = false;
            } else if trimmed.starts_with("<svg") && !line.contains("</svg>") {
                in_svg = true;
            }
            i += 1;
            continue;
        }

        // Headers.
        if let Some(rest) = trimmed.strip_prefix("### ") {
            let slug = slugify(rest);
            toc.push((3, rest.to_string(), slug.clone()));
            let _ = writeln!(body, "<h3 id=\"{slug}\">{}</h3>", inline_md(rest));
            i += 1;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("## ") {
            let slug = slugify(rest);
            toc.push((2, rest.to_string(), slug.clone()));
            let _ = writeln!(body, "<h2 id=\"{slug}\">{}</h2>", inline_md(rest));
            i += 1;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("# ") {
            let slug = slugify(rest);
            let _ = writeln!(body, "<h1 id=\"{slug}\">{}</h1>", inline_md(rest));
            i += 1;
            continue;
        }

        // Horizontal rule.
        if trimmed == "---" {
            body.push_str("<hr/>\n");
            i += 1;
            continue;
        }

        // Table: a header row `| ... |` followed by a separator `|---|`.
        if trimmed.starts_with('|')
            && i + 1 < lines.len()
            && lines[i + 1].trim_start().starts_with('|')
            && lines[i + 1].contains("---")
        {
            let header = split_row(trimmed);
            i += 2; // skip header + separator
            body.push_str("<table><thead><tr>");
            for h in &header {
                let _ = write!(body, "<th>{}</th>", inline_md(h));
            }
            body.push_str("</tr></thead><tbody>");
            while i < lines.len() && lines[i].trim_start().starts_with('|') {
                let cells = split_row(lines[i].trim_start());
                body.push_str("<tr>");
                for c in &cells {
                    let cls = cell_class(c);
                    if cls.is_empty() {
                        let _ = write!(body, "<td>{}</td>", inline_md(c));
                    } else {
                        let _ = write!(body, "<td class=\"{cls}\">{}</td>", inline_md(c));
                    }
                }
                body.push_str("</tr>");
                i += 1;
            }
            body.push_str("</tbody></table>\n");
            continue;
        }

        // Blank line → paragraph break.
        if trimmed.is_empty() {
            i += 1;
            continue;
        }

        // Ordinary paragraph (accumulate consecutive non-blank, non-special).
        let mut para = String::new();
        while i < lines.len() {
            let l = lines[i].trim_start();
            if l.is_empty()
                || l.starts_with('|')
                || l.starts_with('#')
                || l.starts_with('<')
                || l == "---"
            {
                break;
            }
            if !para.is_empty() {
                para.push(' ');
            }
            para.push_str(l);
            i += 1;
        }
        let _ = writeln!(body, "<p>{}</p>", inline_md(&para));
    }

    // Table of contents from h2/h3.
    let mut toc_html = String::new();
    if !toc.is_empty() {
        toc_html.push_str("<nav class=\"ztoc\"><div class=\"ztoc-h\">Contents</div><ul>");
        for (lvl, text, slug) in &toc {
            let cls = if *lvl == 3 { " class=\"sub\"" } else { "" };
            let _ = write!(
                toc_html,
                "<li{cls}><a href=\"#{slug}\">{}</a></li>",
                html_escape(text)
            );
        }
        toc_html.push_str("</ul></nav>");
    }

    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
<title>{title}</title><style>{css}</style></head>\
<body><div class=\"wrap\">{toc}<main class=\"doc\">{body}</main></div></body></html>",
        title = html_escape(title),
        css = REPORT_CSS,
        toc = toc_html,
        body = body,
    )
}

/// Split a markdown table row `| a | b |` into trimmed cells.
fn split_row(line: &str) -> Vec<String> {
    let t = line.trim().trim_start_matches('|').trim_end_matches('|');
    t.split('|').map(|c| c.trim().to_string()).collect()
}

/// Theme-aware, self-contained stylesheet for the report.
const REPORT_CSS: &str = r#"
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#e2e2e2;--acc:#1f6fb2;
--th:#f4f6f8;--good:#137333;--goodbg:#e6f4ea;--bad:#c5221f;--badbg:#fce8e6;
--pos:#1f6fb2;--neg:#c5221f;--axis:#999;--code:#f0f2f4;}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e6e6e6;--muted:#9aa0a6;
--line:#2a2e35;--acc:#63a4d8;--th:#1c1f26;--good:#7ee2a8;--goodbg:#123524;
--bad:#f2a5a2;--badbg:#3a1a19;--pos:#63a4d8;--neg:#f2a5a2;--axis:#666;--code:#1c1f26;}}
:root[data-theme=dark]{--bg:#14161a;--fg:#e6e6e6;--muted:#9aa0a6;--line:#2a2e35;
--acc:#63a4d8;--th:#1c1f26;--good:#7ee2a8;--goodbg:#123524;--bad:#f2a5a2;
--badbg:#3a1a19;--pos:#63a4d8;--neg:#f2a5a2;--axis:#666;--code:#1c1f26;}
:root[data-theme=light]{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#e2e2e2;
--acc:#1f6fb2;--th:#f4f6f8;--good:#137333;--goodbg:#e6f4ea;--bad:#c5221f;
--badbg:#fce8e6;--pos:#1f6fb2;--neg:#c5221f;--axis:#999;--code:#f0f2f4;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{display:flex;gap:0;align-items:flex-start;max-width:1180px;margin:0 auto;}
.ztoc{position:sticky;top:0;max-height:100vh;overflow:auto;width:230px;flex:0 0 230px;
padding:18px 12px;border-right:1px solid var(--line);font-size:13px;}
.ztoc-h{font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;
font-size:11px;margin-bottom:8px;}
.ztoc ul{list-style:none;margin:0;padding:0}
.ztoc li{margin:2px 0}.ztoc li.sub{padding-left:12px;font-size:12px}
.ztoc a{color:var(--fg);text-decoration:none;opacity:.85}
.ztoc a:hover{color:var(--acc);opacity:1}
.doc{flex:1 1 auto;min-width:0;padding:22px 30px 80px;}
h1{font-size:24px;margin:.2em 0 .5em;border-bottom:2px solid var(--line);padding-bottom:.25em}
h2{font-size:19px;margin:1.6em 0 .5em;border-bottom:1px solid var(--line);padding-bottom:.2em}
h3{font-size:16px;margin:1.2em 0 .4em;color:var(--muted)}
p{margin:.5em 0}
code{background:var(--code);padding:.08em .35em;border-radius:3px;
font:12.5px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}
hr{border:0;border-top:1px solid var(--line);margin:1.6em 0}
table{border-collapse:collapse;margin:.6em 0;font-size:13px;width:auto;
display:block;overflow-x:auto;max-width:100%}
th,td{border:1px solid var(--line);padding:4px 9px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{background:var(--th);position:sticky;top:0}
tbody tr:nth-child(even){background:color-mix(in srgb,var(--th) 45%,transparent)}
td.zc-good{background:var(--goodbg);color:var(--good);font-weight:700;text-align:center}
td.zc-bad{background:var(--badbg);color:var(--bad);font-weight:700;text-align:center}
.zsv-chart{display:block;margin:.6em 0;max-width:100%;height:auto;
background:color-mix(in srgb,var(--th) 40%,transparent);border:1px solid var(--line);border-radius:5px}
.zsv-title{fill:var(--muted);font:600 11px sans-serif}
.zsv-lbl{fill:var(--fg);font:11px ui-monospace,monospace}
.zsv-val{fill:var(--muted);font:10px ui-monospace,monospace}
.zsv-axis{stroke:var(--axis);stroke-width:1;stroke-dasharray:2 2}
.zsv-good{fill:var(--good)}.zsv-bad{fill:var(--bad)}
.zsv-pos{fill:var(--pos)}.zsv-neg{fill:var(--neg)}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramp_perfect_monotone() {
        // one ramp, dist_type 1, levels 1..5 strictly decreasing.
        let img = vec!["a".to_string(); 5];
        let q = vec![11.0, 12.0, 13.0, 14.0, 15.0];
        let dial = vec![90.0, 80.0, 70.0, 60.0, 50.0];
        let st = severity_ramp(&img, &q, &dial, 0.5);
        assert_eq!(st.n_ramps, 1);
        assert!((st.pct_monotone - 1.0).abs() < 1e-9);
        assert!((st.pct_strict - 1.0).abs() < 1e-9);
        assert_eq!(st.mean_worst_inv, 0.0);
    }

    #[test]
    fn ramp_inversion_counted() {
        // level 3 jumps UP by 8 → non-monotone, worst inversion 8.
        let img = vec!["a".to_string(); 5];
        let q = vec![11.0, 12.0, 13.0, 14.0, 15.0];
        let dial = vec![90.0, 80.0, 88.0, 60.0, 50.0];
        let st = severity_ramp(&img, &q, &dial, 0.5);
        assert_eq!(st.n_ramps, 1);
        assert!((st.pct_monotone - 0.0).abs() < 1e-9);
        assert!((st.mean_worst_inv - 8.0).abs() < 1e-9);
    }

    #[test]
    fn ramp_signed_excluded() {
        // dist_type 7 is signed → excluded from denominator.
        let img = vec!["a".to_string(); 5];
        let q = vec![71.0, 72.0, 73.0, 74.0, 75.0];
        let dial = vec![50.0, 90.0, 10.0, 99.0, 5.0];
        let st = severity_ramp(&img, &q, &dial, 0.5);
        assert_eq!(st.n_ramps, 0);
        assert_eq!(st.n_signed, 1);
    }

    #[test]
    fn ramp_eps_slack_ties() {
        // tiny upward wiggles within eps count as monotone.
        let img = vec!["a".to_string(); 5];
        let q = vec![11.0, 12.0, 13.0, 14.0, 15.0];
        let dial = vec![90.0, 90.2, 89.9, 89.8, 89.7];
        let st = severity_ramp(&img, &q, &dial, 0.5);
        assert!((st.pct_monotone - 1.0).abs() < 1e-9);
        assert!((st.pct_strict - 0.0).abs() < 1e-9); // not strict (a +0.2 step)
    }

    #[test]
    fn zone_perfect_agreement() {
        // candidate == reference → mean Δ 0, agg srocc 1, rmse 0.
        let refe: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let cand = refe.clone();
        let z = zone_buckets(&cand, &refe, 5.0);
        assert!((z.agg_srocc - 1.0).abs() < 1e-9);
        assert!(z.agg_rmse < 1e-9);
        for r in &z.rows {
            if r.n > 0 {
                assert!(r.mean_delta.abs() < 1e-9);
            }
        }
    }

    #[test]
    fn zone_constant_offset() {
        // candidate = reference − 10 → mean Δ −10 in every zone.
        let refe: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let cand: Vec<f64> = refe.iter().map(|r| r - 10.0).collect();
        let z = zone_buckets(&cand, &refe, 5.0);
        for r in &z.rows {
            if r.n >= 2 {
                assert!((r.mean_delta + 10.0).abs() < 1e-9, "zone Δ should be −10");
            }
        }
        assert!((z.agg_rmse - 10.0).abs() < 1e-9);
    }

    #[test]
    fn md_table_renders() {
        let md = "## Head\n\n| a | b |\n|---|---|\n| 1 | ✓ |\n";
        let html = markdown_to_html(md, "t");
        assert!(html.contains("<h2 id=\"head\">"));
        assert!(html.contains("<table>"));
        assert!(html.contains("zc-good")); // ✓ colored
        assert!(html.contains("<!doctype html>"));
    }

    #[test]
    fn md_svg_passthrough() {
        let md = "text\n<svg viewBox=\"0 0 1 1\"><rect/></svg>\nmore";
        let html = markdown_to_html(md, "t");
        assert!(html.contains("<svg viewBox"));
        assert!(html.contains("<rect/>"));
    }

    #[test]
    fn inline_bold_code() {
        let h = inline_md("a **bold** and `code` end");
        assert_eq!(h, "a <strong>bold</strong> and <code>code</code> end");
    }

    #[test]
    fn corruption_gate_basic() {
        // two triples; one passes (corruption < q20), one fails.
        let label = vec![
            "r__aliasing__whole__op100__corruption".to_string(),
            "r__aliasing__whole__op100__q20".to_string(),
            "r__aliasing__whole__op100__q10".to_string(),
            "r__ringing__sq64__op50__corruption".to_string(),
            "r__ringing__sq64__op50__q20".to_string(),
        ];
        // aliasing: corr 30 < q20 60 → pass. ringing: corr 70 > q20 55 → fail.
        let dial = vec![30.0, 60.0, 40.0, 70.0, 55.0];
        let st = corruption_gate(&label, &dial);
        assert_eq!(st.n_triples, 2);
        assert!((st.pass_q20 - 0.5).abs() < 1e-9);
        // family parse: aliasing passes 100%, ringing 0%.
        let fams: std::collections::HashMap<_, _> = st
            .per_family
            .iter()
            .map(|(f, p, _)| (f.clone(), *p))
            .collect();
        assert!((fams["aliasing"] - 1.0).abs() < 1e-9);
        assert!((fams["ringing"] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn svg_bars_smoke() {
        let s = svg_bars(
            "t",
            &["a".into(), "b".into()],
            &[10.0, 90.0],
            0.0,
            100.0,
            50.0,
            true,
        );
        assert!(s.starts_with("<svg"));
        assert!(s.contains("zsv-good")); // 90 ≥ 50
        assert!(s.contains("zsv-bad")); // 10 < 50
    }
}
