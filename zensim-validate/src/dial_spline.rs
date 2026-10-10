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
    assert_eq!(xs.len(), ys.len(), "spline coordinate lengths differ");
    let nk = xs.len();
    let mut p = Vec::with_capacity(4 + 8 * nk);
    p.extend_from_slice(&(nk as u32).to_le_bytes());
    for i in 0..nk {
        p.extend_from_slice(&(xs[i] as f32).to_le_bytes());
        p.extend_from_slice(&(ys[i] as f32).to_le_bytes());
    }
    assert!(
        crate::output_calibration_spline::parse_payload(&p).is_some(),
        "spline is invalid after f32 serialization (non-finite or collapsed knots)"
    );
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

/// The percentile-edge binning shared by [`fit_spline_knots`] and
/// [`fit_identity_pinned_knots`]: per-bin median `(pred, target)` candidates
/// before any monotone filter. `None` for an unusable anchor.
fn bin_knots(preds: &[f64], tgt: &[f64], n_edges: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    // Invalid/empty anchors cannot produce a bake. Callers already refuse a
    // fit with fewer than two knots; do not panic indexing empty bins.
    if preds.len() != tgt.len()
        || preds.len() < 2
        || n_edges < 2
        || preds.iter().chain(tgt).any(|v| !v.is_finite())
    {
        return None;
    }
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

    if kx.is_empty() {
        return None;
    }
    Some((kx, ky))
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
    let Some((kx, ky)) = bin_knots(preds, tgt, n_edges) else {
        return (Vec::new(), Vec::new());
    };

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

/// E33 registered output stage (`benchmarks/e33_registration_2026-10-09.md`
/// §8), for a `--nonneg-distance` network whose raw output never exceeds
/// `pin` and equals it exactly on a perfect copy.
///
/// 1. [`bin_knots`] candidates, filtered STRICTLY (`x > x_last + 1e-7` and
///    `y > y_last + 1e-6`), so no interior segment is flat.
/// 2. Identity knot `(pin, 100)` appended; refused unless the last fitted
///    knot lies below it in both coordinates. The runtime's upper branch is
///    then reached only at `raw == pin` and returns exactly 100.
/// 3. Tail knot `(x_t, y_t)` prepended on the fitted bottom secant,
///    `x_t = x_0 − tail_factor·(pin − x_0)`, so the first two segments are
///    collinear and the PCHIP endpoint slope is that secant (> 0). Below
///    `x_t` the score keeps falling linearly until the unchanged OOD floor.
pub fn fit_identity_pinned_knots(
    preds: &[f64],
    tgt: &[f64],
    n_edges: usize,
    pin: f64,
    tail_factor: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if !(pin.is_finite() && tail_factor.is_finite() && tail_factor > 0.0) {
        return Err(
            "identity-pinned spline: pin and tail factor must be finite, factor > 0".into(),
        );
    }
    let (kx, ky) = bin_knots(preds, tgt, n_edges)
        .ok_or("identity-pinned spline: unusable calibration rows")?;
    let mut cx = vec![kx[0]];
    let mut cy = vec![ky[0]];
    for i in 1..kx.len() {
        if kx[i] > cx[cx.len() - 1] + 1e-7 && ky[i] > cy[cy.len() - 1] + 1e-6 {
            cx.push(kx[i]);
            cy.push(ky[i]);
        }
    }
    if cx.len() < 2 {
        return Err(format!(
            "identity-pinned spline: only {} strictly increasing knot(s); the network does not \
             rank the calibration rows in the target's direction",
            cx.len()
        ));
    }
    let (x_last, y_last) = (cx[cx.len() - 1], cy[cy.len() - 1]);
    if !(x_last < pin && y_last < 100.0) {
        return Err(format!(
            "identity-pinned spline: last fitted knot ({x_last}, {y_last}) is not below the \
             identity knot ({pin}, 100)"
        ));
    }
    let s0 = (cy[1] - cy[0]) / (cx[1] - cx[0]);
    let x_t = cx[0] - tail_factor * (pin - cx[0]);
    let y_t = cy[0] - s0 * (cx[0] - x_t);
    let mut xs = Vec::with_capacity(cx.len() + 2);
    let mut ys = Vec::with_capacity(cx.len() + 2);
    xs.push(x_t);
    ys.push(y_t);
    xs.extend_from_slice(&cx);
    ys.extend_from_slice(&cy);
    xs.push(pin);
    ys.push(100.0);
    Ok((xs, ys))
}

/// Measured properties of an identity-pinned spline (registration §8 K1–K3),
/// evaluated on the f32-serialized knots through the serving owner.
#[derive(Clone, Debug)]
pub struct IdentityPinnedChecks {
    /// Raw output below which the runtime's OOD floor engages.
    pub x_floor: f64,
    /// Score at the floor.
    pub floor: f64,
    /// PCHIP slope at the identity knot (reported, not gated).
    pub identity_slope: f64,
    /// Bottom fitted knot (the tail starts below it).
    pub x0: f64,
    /// Number of dense-grid points evaluated for K2.
    pub k2_points: usize,
}

/// K1 (every stored derivative except the identity knot's is > 0), K2
/// (served score strictly decreasing in `g = pin − raw` over 10⁶ log-spaced
/// points on `[1e-6, pin − x_floor]`, plus `g = 0` and every knot ± 1 ulp;
/// ulp-adjacent probes must not increase, since f64 cannot resolve a strict
/// decrease of `slope · 1 ulp`)
/// and K3 (`spline(pin) == 100.0`). Errors name the first failure.
pub fn check_identity_pinned(payload: &[u8], pin: f64) -> Result<IdentityPinnedChecks, String> {
    let sp = crate::output_calibration_spline::parse_payload(payload)
        .ok_or("identity-pinned spline: payload does not parse")?;
    let n = sp.xs.len();
    if let Some(k) = (0..n - 1).find(|&k| sp.derivs[k] <= 0.0) {
        return Err(format!(
            "K1: PCHIP derivative at knot {k} (x = {}) is {} (must be > 0)",
            sp.xs[k], sp.derivs[k]
        ));
    }
    let apply = |x: f64| crate::output_calibration_spline::apply(x, &sp);
    let at_pin = apply(pin);
    if at_pin.to_bits() != 100.0f64.to_bits() {
        return Err(format!("K3: spline(pin) = {at_pin:?}, not exactly 100"));
    }
    if (sp.xs[n - 1] - pin).abs() > 0.0 {
        return Err(format!(
            "K3: top knot x = {} is not the pin {pin}",
            sp.xs[n - 1]
        ));
    }
    let floor = sp.ys[0] - (sp.ys[n - 1] - sp.ys[0]);
    let x_floor = sp.xs[0] - (sp.ys[0] - floor) / sp.derivs[0];
    let span = pin - x_floor;
    let mut gs: Vec<f64> = Vec::with_capacity(1_000_000 + 4 * n + 1);
    gs.push(0.0);
    const N: usize = 1_000_000;
    let (lo, hi) = (1e-6f64.ln(), span.ln());
    for i in 0..N {
        gs.push((lo + (hi - lo) * i as f64 / (N - 1) as f64).exp());
    }
    for &x in &sp.xs {
        for xx in [
            x,
            f64::from_bits(x.to_bits() + 1),
            f64::from_bits(x.to_bits() - 1),
        ] {
            let g = pin - xx;
            if (0.0..=span).contains(&g) {
                gs.push(g);
            }
        }
    }
    gs.sort_by(f64::total_cmp);
    gs.dedup();
    // Strict decrease is required between points the f64 score can resolve:
    // the log grid and every knot neighbourhood against its grid neighbours.
    // Probes within a few ulps of each other (a knot and its ±1-ulp twins)
    // must not INCREASE; a change of `slope · 1 ulp` is below the score's own
    // f64 resolution, so equality there is not a flat segment.
    let mut prev: Option<(f64, f64)> = None;
    for &g in &gs {
        let y = apply(pin - g);
        if let Some((pg, py)) = prev {
            let (x, px) = (pin - g, pin - pg);
            let ulp_close = (x - px).abs() <= 4.0 * f64::EPSILON * x.abs().max(px.abs()).max(1.0);
            let ok = if ulp_close { y <= py } else { y < py };
            if !ok {
                return Err(format!(
                    "K2: served score not strictly decreasing between g = {pg} ({py}) and g = {g} ({y})"
                ));
            }
        }
        prev = Some((g, y));
    }
    Ok(IdentityPinnedChecks {
        x_floor,
        floor,
        identity_slope: sp.derivs[n - 1],
        x0: sp.xs[1],
        k2_points: gs.len(),
    })
}

#[cfg(test)]
mod identity_pinned_tests {
    use super::*;

    fn anchor() -> (Vec<f64>, Vec<f64>) {
        // A raw distribution that stays below the pin, ranked like the target.
        let preds: Vec<f64> = (0..4000).map(|i| 15.0 + 77.0 * i as f64 / 4000.0).collect();
        let tgt: Vec<f64> = preds
            .iter()
            .map(|p| 30.0 + 0.7 * (p - 15.0) + 3.0 * (p / 9.0).sin())
            .collect();
        (preds, tgt)
    }

    #[test]
    fn identity_pinned_spline_passes_its_checks() {
        let (p, t) = anchor();
        let (xs, ys) = fit_identity_pinned_knots(&p, &t, 18, 100.0, 2.0).unwrap();
        assert_eq!(*xs.last().unwrap(), 100.0);
        assert_eq!(*ys.last().unwrap(), 100.0);
        assert!(ys.windows(2).all(|w| w[1] > w[0]));
        let payload = spline_payload(&xs, &ys);
        let c = check_identity_pinned(&payload, 100.0).unwrap();
        assert!(c.x_floor < xs[0]);
        assert!(c.k2_points > 1_000_000);
    }

    #[test]
    fn the_existing_fit_can_leave_a_zero_endpoint_slope_and_k1_catches_it() {
        // Shallow first segment, steep second: pchip_endpoint clamps d0 to 0,
        // which flattens the whole lower tail.
        let xs = [10.0, 20.0, 21.0, 100.0];
        let ys = [40.0, 40.5, 90.0, 100.0];
        let payload = spline_payload(&xs, &ys);
        assert!(
            check_identity_pinned(&payload, 100.0)
                .unwrap_err()
                .starts_with("K1")
        );
    }

    #[test]
    fn refusals() {
        let (p, t) = anchor();
        // Pin below the calibration rows' top knot.
        assert!(fit_identity_pinned_knots(&p, &t, 18, 50.0, 2.0).is_err());
        // Anti-ranked network.
        let rev: Vec<f64> = t.iter().map(|y| -y).collect();
        assert!(fit_identity_pinned_knots(&p, &rev, 18, 100.0, 2.0).is_err());
        assert!(fit_identity_pinned_knots(&p, &t, 18, 100.0, 0.0).is_err());
    }
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

#[cfg(test)]
mod historical_fit_tests {
    use super::*;

    #[test]
    fn v47_recorded_fitter_knots_and_f32_payload_match() {
        let v: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/v47_spline_fit.json")).unwrap();
        let numbers = |v: &serde_json::Value| {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect::<Vec<_>>()
        };
        let preds = numbers(&v["predictions"]);
        let targets = numbers(&v["targets"]);
        for case in v["cases"].as_array().unwrap() {
            let (xs, ys) =
                fit_spline_knots(&preds, &targets, 18, case["neg_tail"].as_bool().unwrap());
            let expected_x = numbers(&case["xs"]);
            assert_eq!(xs.len(), expected_x.len());
            for (a, b) in xs.iter().zip(expected_x) {
                assert!((a - b).abs() < 1e-14);
            }
            assert_eq!(ys, numbers(&case["ys"]));
            let hex = spline_payload(&xs, &ys)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>();
            assert_eq!(hex, case["payload_hex"].as_str().unwrap());
        }
    }

    #[test]
    fn invalid_anchor_cannot_produce_knots() {
        for (preds, tgt, edges) in [
            (vec![], vec![], 18),
            (vec![1., 2.], vec![1.], 18),
            (vec![1., 2.], vec![1., 2.], 1),
            (vec![1., f64::NAN], vec![1., 2.], 18),
            (vec![1., 2.], vec![1., f64::INFINITY], 18),
        ] {
            assert!(fit_spline_knots(&preds, &tgt, edges, true).0.is_empty());
        }
    }

    #[test]
    #[should_panic(expected = "invalid after f32 serialization")]
    fn distinct_f64_knots_that_collapse_in_the_bake_are_refused() {
        spline_payload(&[1.0, 1.0 + 1e-9], &[0.0, 100.0]);
    }
}
