//! STRATEGY-2026-07-02: training/sampling strategy primitives, implemented as
//! pure functions so each is verified against a REFERENCE implementation
//! (scipy/numpy constants + finite differences) independently of the trainer.
//!
//! Test-failure convention (user directive 2026-07-02): a mismatch here is
//! flagged as **"IMPL BUG (not strategy)"** — the algorithm's merit is judged
//! by fleet ablation, but only after these tests prove the implementation
//! matches the math.
//!
//! Contents:
//! - [`listmle_loss_grad`] — ListMLE (Plackett–Luce NLL) listwise loss.
//! - [`triplet_probit_loss_dgrad`] — ordered-probit triplet NLL (KonFiG /
//!   AIC-3 raw-response likelihood; same model as the SDR25 reconstruction).
//! - [`EmaState`] — per-epoch exponential moving average of weight tensors.
//! - [`dro_reweight`] — GroupDRO-style multiplicative group weights.
//! - [`build_bands`] — equal-count target-quantile bands for stratified
//!   sampling.
//! - [`hard_pair_ok`] — hard-pair acceptance predicate.

/// Standard normal CDF via erf (no external dep; |err| < 1.2e-7 which is
/// far below the 1e-6 test tolerance vs scipy).
fn phi(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Abramowitz–Stegun 7.1.26 erf approximation (|err| ≤ 1.5e-7), sign-folded.
fn erf(x: f64) -> f64 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    s * y
}

fn pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// ListMLE (Plackett–Luce NLL) over one list. `targets` define the ground
/// truth order (descending = best first, stable ties); `scores` are the
/// model outputs. Returns `(loss, dL/dscores)` aligned with the INPUT order.
///
/// Reference: Xia et al. 2008 "Listwise Approach to Learning to Rank";
/// verified against a numpy implementation (see tests).
pub fn listmle_loss_grad(scores: &[f64], targets: &[f64]) -> (f64, Vec<f64>) {
    let n = scores.len();
    assert_eq!(n, targets.len());
    if n < 2 {
        return (0.0, vec![0.0; n]);
    }
    // stable argsort by descending target
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        targets[b]
            .partial_cmp(&targets[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let s_sorted: Vec<f64> = order.iter().map(|&i| scores[i]).collect();
    let mut loss = 0.0;
    let mut grad_sorted = vec![0.0f64; n];
    for i in 0..n {
        let suf = &s_sorted[i..];
        let m = suf.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lse = m + suf.iter().map(|&v| (v - m).exp()).sum::<f64>().ln();
        loss += -(s_sorted[i] - lse);
        for (j, &v) in suf.iter().enumerate() {
            grad_sorted[i + j] += (v - lse).exp();
        }
        grad_sorted[i] -= 1.0;
    }
    let mut grad = vec![0.0f64; n];
    for (pos, &orig) in order.iter().enumerate() {
        grad[orig] = grad_sorted[pos];
    }
    (loss, grad)
}

/// Ordered-probit triplet NLL for one raw human response.
///
/// Convention (KonFiG / SDR25, trap-verified 2026-07-02): the pivot is the
/// pristine original and the response names the side judged **more
/// distorted**. With model QUALITY scores `ya`, `yb` and `delta = yb − ya`:
///
/// ```text
/// P(resp=0, "left more distorted")  = Φ(( delta − τ)/σ)   // ya < yb expected
/// P(resp=1, "right more distorted") = Φ((−delta − τ)/σ)
/// P(resp=2, "not sure")             = 1 − P0 − P1
/// ```
///
/// Returns `(loss, dL/d(delta))`; the caller maps to per-side gradients as
/// `dL/dya = −dL/ddelta`, `dL/dyb = +dL/ddelta`. Verified against scipy
/// constants + finite differences (tests below).
pub fn triplet_probit_loss_dgrad(ya: f64, yb: f64, tau: f64, sigma: f64, resp: u8) -> (f64, f64) {
    let d = yb - ya;
    let z1 = (d - tau) / sigma;
    let z2 = (-d - tau) / sigma;
    let pl = phi(z1);
    let pr = phi(z2);
    let pn = (1.0 - pl - pr).max(1e-12);
    let dpl_dd = pdf(z1) / sigma;
    let dpr_dd = -pdf(z2) / sigma;
    let (p, dp_dd) = match resp {
        0 => (pl.max(1e-12), dpl_dd),
        1 => (pr.max(1e-12), dpr_dd),
        _ => (pn, -dpl_dd - dpr_dd),
    };
    (-p.ln(), -dp_dd / p)
}

/// Per-epoch exponential moving average over an arbitrary set of weight
/// tensors (SWA-flavored: updated once per epoch, so `decay` is per-epoch —
/// typical 0.9). The bake snapshot and validation both read the EMA copies
/// so the shipped bytes equal the validated net.
pub struct EmaState {
    pub decay: f64,
    pub tensors: Vec<Vec<f64>>,
    pub scalars: Vec<f64>,
    initialized: bool,
}

impl EmaState {
    pub fn new(decay: f64) -> Self {
        Self { decay, tensors: Vec::new(), scalars: Vec::new(), initialized: false }
    }

    /// Fold the current live weights into the average. First call copies.
    pub fn update(&mut self, tensors: &[&[f64]], scalars: &[f64]) {
        if !self.initialized {
            self.tensors = tensors.iter().map(|t| t.to_vec()).collect();
            self.scalars = scalars.to_vec();
            self.initialized = true;
            return;
        }
        let d = self.decay;
        for (e, t) in self.tensors.iter_mut().zip(tensors) {
            for (ev, &tv) in e.iter_mut().zip(t.iter()) {
                *ev = d * *ev + (1.0 - d) * tv;
            }
        }
        for (ev, &sv) in self.scalars.iter_mut().zip(scalars) {
            *ev = d * *ev + (1.0 - d) * sv;
        }
    }
}

/// GroupDRO-style multiplicative reweighting: `w_i ∝ base_i · exp(η·(L_i −
/// max L))` (max-shifted for numerical stability), normalized to sum 1.
/// η = 0 returns the normalized base weights (pure no-op), so the flag
/// default cannot change sampling.
pub fn dro_reweight(base_w: &[f64], mean_losses: &[f64], eta: f64) -> Vec<f64> {
    assert_eq!(base_w.len(), mean_losses.len());
    let lmax = mean_losses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut w: Vec<f64> = base_w
        .iter()
        .zip(mean_losses)
        .map(|(&b, &l)| b * (eta * (l - lmax)).exp())
        .collect();
    let s: f64 = w.iter().sum();
    if s > 0.0 {
        for v in &mut w {
            *v /= s;
        }
    }
    w
}

/// Equal-count target-quantile bands: rows sorted by target, chunked into
/// `n_bands` contiguous runs. Sampling band-uniform then row-uniform gives
/// every target decile equal gradient exposure regardless of the corpus's
/// band histogram.
pub fn build_bands(targets: &[f64], n_bands: usize) -> Vec<Vec<usize>> {
    let n = targets.len();
    if n_bands <= 1 || n < n_bands * 2 {
        return vec![(0..n).collect()];
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        targets[a]
            .partial_cmp(&targets[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let per = n / n_bands;
    let mut bands: Vec<Vec<usize>> = Vec::with_capacity(n_bands);
    for b in 0..n_bands {
        let lo = b * per;
        let hi = if b == n_bands - 1 { n } else { (b + 1) * per };
        bands.push(idx[lo..hi].to_vec());
    }
    bands
}

/// Hard-pair predicate: is (a, b) a near-threshold pair?
#[inline]
pub fn hard_pair_ok(ta: f64, tb: f64, max_delta: f64) -> bool {
    (ta - tb).abs() <= max_delta
}

#[cfg(test)]
mod tests {
    use super::*;

    const BUG: &str = "IMPL BUG (not strategy): mismatch vs reference implementation";

    #[test]
    fn triplet_probit_matches_scipy_constants() {
        // Constants computed 2026-07-02 with scipy.stats.norm (see the
        // session log): (ya, yb, tau, sigma, resp) -> (loss, dL/ddelta).
        let cases = [
            (0.2, 0.9, 0.5, 1.0, 0u8, 0.546004353723, -0.675073179790),
            (0.8, 0.3, 0.5, 1.0, 1u8, 0.693147180560, 0.797884560803),
            (0.5, 0.55, 0.5, 1.0, 2u8, 0.961065599223, 0.045970699950),
            (0.1, 0.4, 0.6, 1.5, 0u8, 0.865739522682, -0.619610565390),
        ];
        for (ya, yb, tau, s, r, l_ref, d_ref) in cases {
            let (l, d) = triplet_probit_loss_dgrad(ya, yb, tau, s, r);
            assert!((l - l_ref).abs() < 1e-6, "{BUG}: triplet loss {l} != scipy {l_ref}");
            assert!((d - d_ref).abs() < 1e-6, "{BUG}: triplet dL/dd {d} != scipy {d_ref}");
        }
    }

    #[test]
    fn triplet_probit_gradient_matches_finite_difference() {
        // NOTE: cases are chosen with z-arguments away from 0 — the A&S erf
        // approximation sign-folds at x=0, so central differences OF THE
        // APPROXIMATED loss are wrong by ~1e-3 exactly at the fold while the
        // analytic gradient (exact pdf form) is right — proven by the scipy
        // constants test above, which pins a z=0 case to 1e-6. First caught
        // 2026-07-02 when the (0.8,0.3,resp=1) case put z2 exactly at 0.
        let eps = 1e-5;
        for (ya, yb, tau, s, r) in
            [(0.2, 0.9, 0.5, 1.0, 0u8), (0.9, 0.3, 0.5, 1.0, 1u8), (0.5, 0.75, 0.5, 1.0, 2u8)]
        {
            let (_, d) = triplet_probit_loss_dgrad(ya, yb, tau, s, r);
            let (lp, _) = triplet_probit_loss_dgrad(ya, yb + eps, tau, s, r);
            let (lm, _) = triplet_probit_loss_dgrad(ya, yb - eps, tau, s, r);
            let fd = (lp - lm) / (2.0 * eps);
            assert!((d - fd).abs() < 1e-4, "{BUG}: triplet analytic {d} != FD {fd}");
        }
    }

    #[test]
    fn listmle_matches_numpy_constants() {
        // numpy reference computed 2026-07-02 (session log).
        let (l, g) = listmle_loss_grad(&[0.3, -0.1, 0.7], &[0.9, 0.2, 0.5]);
        assert!((l - 1.522351179676).abs() < 1e-9, "{BUG}: listmle loss {l}");
        let gref = [-0.683758941775, 0.522008239580, 0.161750702195];
        for (a, b) in g.iter().zip(gref) {
            assert!((a - b).abs() < 1e-9, "{BUG}: listmle grad {a} != {b}");
        }
        let (l4, g4) = listmle_loss_grad(&[1.0, 0.0, 0.5, 0.2], &[0.1, 0.9, 0.5, 0.7]);
        assert!((l4 - 4.380082469056).abs() < 1e-9, "{BUG}: listmle4 loss {l4}");
        let g4ref = [1.521459532275, -0.848218215202, -0.077188146163, -0.596053170911];
        for (a, b) in g4.iter().zip(g4ref) {
            assert!((a - b).abs() < 1e-9, "{BUG}: listmle4 grad {a} != {b}");
        }
    }

    #[test]
    fn listmle_gradient_matches_finite_difference() {
        let scores = [0.4, -0.3, 0.9, 0.1, -0.6];
        let targets = [0.2, 0.9, 0.4, 0.8, 0.1];
        let (_, g) = listmle_loss_grad(&scores, &targets);
        let eps = 1e-6;
        for i in 0..scores.len() {
            let mut sp = scores;
            sp[i] += eps;
            let (lp, _) = listmle_loss_grad(&sp, &targets);
            sp[i] -= 2.0 * eps;
            let (lm, _) = listmle_loss_grad(&sp, &targets);
            let fd = (lp - lm) / (2.0 * eps);
            assert!((g[i] - fd).abs() < 1e-5, "{BUG}: listmle grad[{i}] {} != FD {fd}", g[i]);
        }
    }

    #[test]
    fn ema_matches_closed_form() {
        let mut e = EmaState::new(0.9);
        e.update(&[&[1.0, 2.0]], &[10.0]); // init copy
        e.update(&[&[3.0, 4.0]], &[20.0]);
        e.update(&[&[5.0, 6.0]], &[30.0]);
        // closed form: ((1*.9 + 3*.1)*.9 + 5*.1) etc.
        let t0 = (1.0f64 * 0.9 + 3.0 * 0.1) * 0.9 + 5.0 * 0.1;
        let t1 = (2.0f64 * 0.9 + 4.0 * 0.1) * 0.9 + 6.0 * 0.1;
        let s0 = (10.0f64 * 0.9 + 20.0 * 0.1) * 0.9 + 30.0 * 0.1;
        assert!((e.tensors[0][0] - t0).abs() < 1e-12, "{BUG}: ema tensor {} != {t0}", e.tensors[0][0]);
        assert!((e.tensors[0][1] - t1).abs() < 1e-12, "{BUG}");
        assert!((e.scalars[0] - s0).abs() < 1e-12, "{BUG}: ema scalar");
    }

    #[test]
    fn dro_matches_closed_form_and_eta0_is_noop() {
        let base = [1.0, 2.0, 1.0];
        let losses = [0.5, 1.5, 1.0];
        let w = dro_reweight(&base, &losses, 2.0);
        // manual: exp(2*(L - 1.5)) = [e^-2, 1, e^-1]; base-mult then norm
        let raw = [1.0 * (-2.0f64).exp(), 2.0 * 1.0, 1.0 * (-1.0f64).exp()];
        let s: f64 = raw.iter().sum();
        for (a, b) in w.iter().zip(raw.iter().map(|v| v / s)) {
            assert!((a - b).abs() < 1e-12, "{BUG}: dro weight {a} != {b}");
        }
        let w0 = dro_reweight(&base, &losses, 0.0);
        let s0: f64 = base.iter().sum();
        for (a, b) in w0.iter().zip(base.iter().map(|v| v / s0)) {
            assert!((a - b).abs() < 1e-12, "{BUG}: dro eta=0 must be a no-op");
        }
    }

    #[test]
    fn bands_are_equal_count_and_ordered() {
        let targets: Vec<f64> = (0..103).map(|i| ((i * 37) % 103) as f64 / 103.0).collect();
        let bands = build_bands(&targets, 10);
        assert_eq!(bands.len(), 10, "{BUG}");
        let total: usize = bands.iter().map(|b| b.len()).sum();
        assert_eq!(total, 103, "{BUG}: bands must partition rows");
        for w in bands.windows(2) {
            let max_lo = w[0].iter().map(|&i| targets[i]).fold(f64::NEG_INFINITY, f64::max);
            let min_hi = w[1].iter().map(|&i| targets[i]).fold(f64::INFINITY, f64::min);
            assert!(max_lo <= min_hi + 1e-12, "{BUG}: bands must be target-ordered");
        }
        for b in &bands[..9] {
            assert_eq!(b.len(), 10, "{BUG}: equal-count bands (except last)");
        }
    }

    #[test]
    fn hard_pair_predicate() {
        assert!(hard_pair_ok(0.50, 0.53, 0.05), "{BUG}");
        assert!(!hard_pair_ok(0.50, 0.60, 0.05), "{BUG}");
    }

    #[test]
    fn phi_matches_scipy() {
        // scipy.stats.norm.cdf reference points
        for (x, r) in [(0.0, 0.5), (1.0, 0.8413447460685429), (-1.5, 0.06680720126885807)] {
            assert!((phi(x) - r).abs() < 2e-7, "{BUG}: phi({x}) {} != {r}", phi(x));
        }
    }
}
