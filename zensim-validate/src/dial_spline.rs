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
        let zeros: Vec<usize> = (0..cy.len()).filter(|&i| cy[i] <= 1e-6).collect();
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
