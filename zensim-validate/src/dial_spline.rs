//! Output-calibration dial-spline fitting — the single owner of the
//! percentile-edge `fit_spline_knots` and its helpers.
//!
//! Extracted from `bake_dial_refit` (2026-07-16) so BOTH the linear
//! anchor-refit tool AND the min-max bake path (`train_minmax --dial-anchor`)
//! fit the [0,100] output dial with the same code. The functions are moved
//! verbatim — a faithful port of `linear_projections_2026-07-03.py::
//! fit_spline_knots` — so bake_dial_refit's behavior (and its
//! `fit_spline_knots_is_monotone` test) is unchanged.
//!
//! The wire format matches `zensim::metric`'s
//! `zentrain.output_calibration_spline` parser: `[u32 n_knots, n_knots × (x:
//! f32, y: f32)]`, knots strictly increasing in x.

/// Encode PCHIP knots into the `zentrain.output_calibration_spline` payload:
/// `[u32 n_knots, n_knots × (x: f32 LE, y: f32 LE)]`.
pub fn spline_payload(xs: &[f64], ys: &[f64]) -> Vec<u8> {
    let nk = xs.len();
    let mut p = Vec::with_capacity(4 + 8 * nk);
    p.extend_from_slice(&(nk as u32).to_le_bytes());
    for i in 0..nk {
        p.extend_from_slice(&(xs[i] as f32).to_le_bytes());
        p.extend_from_slice(&(ys[i] as f32).to_le_bytes());
    }
    p
}

/// `numpy.percentile(sorted, p)` with linear interpolation. `sorted` MUST be
/// ascending.
pub fn percentile_linear(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let rank = p / 100.0 * (n as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = rank - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// `np.median` of the values at `idx` — linear-interp 50th percentile.
pub fn median_at(idx: &[usize], vals: &[f64]) -> f64 {
    let mut v: Vec<f64> = idx.iter().map(|&i| vals[i]).collect();
    v.sort_by(f64::total_cmp);
    percentile_linear(&v, 50.0)
}

/// A quantile bin becomes a knot at its `(median pred, median target)` iff it
/// holds >= 2 rows (matches `fit_spline_knots`).
pub fn push_bin(mask: &[usize], preds: &[f64], tgt: &[f64], kx: &mut Vec<f64>, ky: &mut Vec<f64>) {
    if mask.len() >= 2 {
        kx.push(median_at(mask, preds));
        ky.push(median_at(mask, tgt));
    }
}

/// Does the `neg_tail` choice CHANGE the fitted spline for this anchor?
///
/// Returns `Some((n_knots_without_dedup, n_knots_with_dedup))` when it does —
/// i.e. when the fit produced a RUN of more than one knot at `y <= 1e-6`.
///
/// **Why this matters (ADD156 ship audit, defect D4).** A run of `y ≈ 0` knots
/// makes the spline's bottom segment FLAT at zero, so every prediction in that
/// x-range maps to exactly `0.0` and the linear extrapolation below the bottom
/// knot has slope 0. The negative tail — which the product contract requires to
/// work, because inputs worse than the worst codec output must score BELOW 0 —
/// is silently deleted. `neg_tail` dedups the run down to its last knot, which
/// restores the slope.
///
/// Measured cost of getting this wrong on ADD156: dial p5 `−12.4334` →
/// `0.0000`, and up to **−0.021 SROCC** (LIVE 0.9602 → 0.9397; CSIQ, KADID,
/// PIPAL and TID all moved too). `--neg-tail` restored every corpus exactly.
///
/// `None` = the choice is immaterial for this anchor and either setting emits
/// the same knots.
pub fn neg_tail_is_material(preds: &[f64], tgt: &[f64], n_edges: usize) -> Option<(usize, usize)> {
    let (kx_keep, _) = fit_spline_knots(preds, tgt, n_edges, false);
    let (kx_dedup, _) = fit_spline_knots(preds, tgt, n_edges, true);
    if kx_keep.len() == kx_dedup.len() {
        return None;
    }
    Some((kx_keep.len(), kx_dedup.len()))
}

/// Faithful port of `linear_projections_2026-07-03.py::fit_spline_knots`:
/// percentile-EDGE bins (edges at `linspace(1,99,n_edges)` percentiles),
/// per-bin median `(pred, target)` knots, a strictly-increasing-x /
/// non-decreasing-y monotone filter, and the neg-tail dedup (keep only the
/// last of any run of `y<=1e-6` knots).
pub fn fit_spline_knots(
    preds: &[f64],
    tgt: &[f64],
    n_edges: usize,
    neg_tail: bool,
) -> (Vec<f64>, Vec<f64>) {
    let mut sorted = preds.to_vec();
    sorted.sort_by(f64::total_cmp);
    let edges: Vec<f64> = (0..n_edges)
        .map(|i| {
            let p = 1.0 + 98.0 * (i as f64) / (n_edges as f64 - 1.0);
            percentile_linear(&sorted, p)
        })
        .collect();

    let mut kx: Vec<f64> = Vec::new();
    let mut ky: Vec<f64> = Vec::new();
    let below: Vec<usize> = (0..preds.len()).filter(|&i| preds[i] < edges[0]).collect();
    push_bin(&below, preds, tgt, &mut kx, &mut ky);
    for e in 0..n_edges - 1 {
        let m: Vec<usize> = (0..preds.len())
            .filter(|&i| preds[i] >= edges[e] && preds[i] < edges[e + 1])
            .collect();
        push_bin(&m, preds, tgt, &mut kx, &mut ky);
    }
    let hi: Vec<usize> = (0..preds.len())
        .filter(|&i| preds[i] >= edges[n_edges - 1])
        .collect();
    push_bin(&hi, preds, tgt, &mut kx, &mut ky);

    // strictly-increasing-x, non-decreasing-y monotone filter.
    let mut cx = vec![kx[0]];
    let mut cy = vec![ky[0]];
    for i in 1..kx.len() {
        if kx[i] > cx[cx.len() - 1] + 1e-7 && ky[i] >= cy[cy.len() - 1] {
            cx.push(kx[i]);
            cy.push(ky[i]);
        }
    }
    if neg_tail {
        // The run this dedups is a run of ZERO knots — the y == 0 plateau a
        // CLAMPED anchor (`target_score = max(ssim2, 0)`) produces. The test
        // must therefore be `|y| <= 1e-6`, not `y <= 1e-6`.
        //
        // MEASURED 2026-09-04 (D-id100 lane): with an UNCLAMPED anchor target
        // (`ssim2_gpu`, which the multiband anchor already carries at full
        // depth) `y <= 1e-6` matches every genuinely NEGATIVE knot too, so the
        // dedup deleted the entire negative tail and kept only its shallowest
        // member. On a 4,021-row anchor holding 2,147 negative rows spanning
        // ssim2 −1437.97 … −0.74, the fitted bottom knot came back at
        // y = −12.16 (10 knots survived); the deep evidence was discarded.
        // Because the dial's OOD floor is `ys[0] − (ys[n−1] − ys[0])`, that
        // capped the whole negative tail at −124.33.
        //
        // BYTE-INERT for every clamped anchor: when no `y` is negative,
        // `y <= 1e-6` and `|y| <= 1e-6` select the same indices. Gated by
        // `neg_tail_dedup_is_byte_inert_on_a_clamped_anchor` +
        // `neg_tail_dedup_keeps_genuinely_negative_knots` below.
        let zeros: Vec<usize> = (0..cy.len()).filter(|&i| cy[i].abs() <= 1e-6).collect();
        if zeros.len() > 1 {
            let drop: std::collections::HashSet<usize> =
                zeros[..zeros.len() - 1].iter().copied().collect();
            let fx: Vec<f64> = (0..cx.len())
                .filter(|i| !drop.contains(i))
                .map(|i| cx[i])
                .collect();
            let fy: Vec<f64> = (0..cy.len())
                .filter(|i| !drop.contains(i))
                .map(|i| cy[i])
                .collect();
            return (fx, fy);
        }
    }
    (cx, cy)
}

#[cfg(test)]
mod neg_tail_dedup_tests {
    use super::*;

    /// Build an anchor whose targets are the CLAMPED form (`max(y, 0)`) with a
    /// long zero plateau at the bottom — the shape every shipped recipe fits on.
    fn clamped_anchor(n: usize) -> (Vec<f64>, Vec<f64>) {
        let preds: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        // Truth ramps from -60 to 100; the stored target clamps the negatives.
        let tgt: Vec<f64> = preds.iter().map(|p| (-60.0 + 160.0 * p).max(0.0)).collect();
        (preds, tgt)
    }

    /// The fix must not move a single knot for a clamped anchor: when no `y` is
    /// negative, `y <= 1e-6` and `|y| <= 1e-6` select the same indices. This is
    /// what makes the change inert for every recipe already on disk.
    #[test]
    fn neg_tail_dedup_is_byte_inert_on_a_clamped_anchor() {
        for n in [200usize, 1000, 2000] {
            let (preds, tgt) = clamped_anchor(n);
            assert!(tgt.iter().all(|&y| y >= 0.0), "fixture must be clamped");
            let (kx, ky) = fit_spline_knots(&preds, &tgt, 18, true);
            // Reference: the pre-fix predicate, applied to the same pre-dedup knots.
            let (rx, ry) = {
                let (cx, cy) = fit_spline_knots(&preds, &tgt, 18, false);
                let zeros: Vec<usize> = (0..cy.len()).filter(|&i| cy[i] <= 1e-6).collect();
                if zeros.len() > 1 {
                    let drop: std::collections::HashSet<usize> =
                        zeros[..zeros.len() - 1].iter().copied().collect();
                    (
                        (0..cx.len())
                            .filter(|i| !drop.contains(i))
                            .map(|i| cx[i])
                            .collect(),
                        (0..cy.len())
                            .filter(|i| !drop.contains(i))
                            .map(|i| cy[i])
                            .collect(),
                    )
                } else {
                    (cx, cy)
                }
            };
            assert_eq!(kx.len(), rx.len(), "knot count moved at n={n}");
            for i in 0..kx.len() {
                assert_eq!(kx[i].to_bits(), rx[i].to_bits(), "kx[{i}] moved at n={n}");
                assert_eq!(ky[i].to_bits(), ry[i].to_bits(), "ky[{i}] moved at n={n}");
            }
        }
    }

    /// With an UNCLAMPED anchor the dedup must keep the negative knots. The
    /// pre-fix predicate collapsed the whole run down to its shallowest member,
    /// which is what capped the dial's negative reach (the OOD floor is
    /// `ys[0] - (ys[n-1] - ys[0])`, so a shallow `ys[0]` is a shallow floor).
    #[test]
    fn neg_tail_dedup_keeps_genuinely_negative_knots() {
        let n = 2000usize;
        let preds: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let tgt: Vec<f64> = preds.iter().map(|p| -400.0 + 500.0 * p).collect();
        assert!(tgt.iter().any(|&y| y < -100.0), "fixture must go deep");
        let (kx, ky) = fit_spline_knots(&preds, &tgt, 18, true);
        assert_eq!(kx.len(), ky.len());
        let n_neg = ky.iter().filter(|&&y| y < -1e-6).count();
        assert!(
            n_neg > 1,
            "the negative tail must survive the dedup, got {n_neg} negative knots: {ky:?}"
        );
        assert!(
            ky[0] < -100.0,
            "the bottom knot must carry the anchor's deep evidence, got {}",
            ky[0]
        );
        // The pre-fix predicate is the negative control: it keeps exactly one.
        let (_, cy) = fit_spline_knots(&preds, &tgt, 18, false);
        let zeros: Vec<usize> = (0..cy.len()).filter(|&i| cy[i] <= 1e-6).collect();
        assert!(
            zeros.len() > 1,
            "control: the pre-fix predicate must have had a run to collapse"
        );
    }
}
