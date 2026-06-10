//! Norm-in-Norm loss (Li, Jiang, Jiang 2020, ACM MM, arXiv:2008.03889).
//!
//! Implements the **hybrid auxiliary loss** added on top of the existing
//! RankNet pairwise loss in [`crate::mlp_train`]. Per the paper's Table 2
//! (KonIQ-10k), the best published configuration is
//! `loss = ranknet + 0.1 · norm_in_norm` with `p = 1, q = 2`, lifting
//! SROCC from 0.928 → 0.937 vs RankNet alone.
//!
//! ## Mathematical formulation
//!
//! For a batch of `N` predictions `Ŝ` and `N` MOS labels `S`:
//!
//! 1. **Centering**: `ŝ = Ŝ - mean(Ŝ)`, `s = S - mean(S)`
//! 2. **Q-norm normalization** with `ε = 1e-8`:
//!    - `Z = (Σ |ŝ_k|^q)^(1/q)` (the `q`-norm of centered preds)
//!    - `ẑ = Z + ε`
//!    - `ŝ_n = ŝ / ẑ`, `s_n = s / (||s||_q + ε)`
//! 3. **Loss** (matching the reference impl's `exponent=True` default,
//!    `detach=False`):
//!    - `Δ = ŝ_n - s_n`
//!    - `scale = 2^max(1, 1/q) · N^max(0, 1/p - 1/q)`
//!    - `L = (||Δ||_p / scale)^p = ||Δ||_p^p / scale^p`
//!
//! For the canonical `(p=1, q=2)` setting: `scale = 2 · √N`, so
//! `L = (Σ |Δ_i|) / (2 √N)`.
//!
//! For the PLCC-induced special case `(p=q=2)`: `scale = 2`, so
//! `L = (Σ Δ_i²) / 4` — proportional to `(1 - PLCC)` per paper Eqn 14.
//!
//! ## Closed-form gradient (verified vs autograd by the gradient-check test)
//!
//! Let `g_i = dL/dΔ_i = (p / scale^p) · |Δ_i|^(p-1) · sign(Δ_i)`.
//! Let `u_j = (1 / Z^(q-1)) · |ŝ_j|^(q-1) · sign(ŝ_j)` (so for q=2:
//! `u_j = ŝ_j / Z`).
//! Let `dot_gs = Σ_i g_i · ŝ_i`.
//!
//! Then `Ŝ` flows into the loss via both `ŝ` (mean-centering) AND `ẑ`
//! (the q-norm of `ŝ`):
//!
//! ```text
//!   dL/dŜ_j = (g_j - mean(g)) / ẑ  −  (u_j - mean(u)) · dot_gs / ẑ²
//! ```
//!
//! The `mean(g)` and `mean(u)` corrections come from the centering
//! step (`dŝ_i/dŜ_j = δ_ij - 1/N`); the second term carries the
//! through-the-denominator chain. Paper's reference impl
//! (`https://github.com/lidq92/LinearityIQA/blob/master/IQAloss.py`)
//! relies on autograd, which produces exactly this expression — the
//! commented-out `gb`/`gab` corrections in their source match the
//! `(u - mean(u)) · dot_gs / ẑ²` term derived here.
//!
//! ## NaN / zero-variance protection
//!
//! When the batch has zero variance in predictions (`Z = 0`), the `ε`
//! in `ẑ = Z + ε` keeps `1/ẑ` finite. When `Δ_i = 0` for some `i`
//! with `p < 1`, the gradient `|Δ_i|^(p-1)` blows up; for `p ≥ 1` it's
//! either 1 (p=1) or 0 (p>1). The implementation guards `0^0 = 0` for
//! the `p=1` case where `sign(0) = 0`.
//!
//! ## When to use
//!
//! The cleanest reading: each "minibatch" of `K` RankNet pairs gives
//! `2K` predictions (`y_a` and `y_b` from each pair). Compute
//! Norm-in-Norm loss over these `2K` (point, label) pairs in parallel
//! to the RankNet loss. Both backpropagate through the same shared
//! MLP. The per-prediction gradients **add**.
//!
//! Mini-batch path is **required** for Norm-in-Norm stability — batch
//! statistics are meaningless at `N < ~16`. The trainer enforces
//! `minibatch_size ≥ 16` whenever `norm_in_norm_weight > 0`.

/// Numerical-stability constant — matches the reference impl
/// (`eps = 1e-8` at file scope in `IQAloss.py`).
pub const EPS: f64 = 1e-8;

/// Recommended hybrid weighting per Li 2020 Table 2 (last row,
/// KonIQ-10k headline result).
#[allow(dead_code)] // paper-reference default, kept importable for recipes
pub const PAPER_HYBRID_BETA: f64 = 0.1;

/// Recommended `p` per Li 2020 Table 1 (best single-loss config).
#[allow(dead_code)] // paper-reference default, kept importable for recipes
pub const PAPER_P: f64 = 1.0;

/// Recommended `q` per Li 2020 (the `q=2` choice is z-score-equivalent
/// centering + scaling; matches the reference impl default).
#[allow(dead_code)] // paper-reference default, kept importable for recipes
pub const PAPER_Q: f64 = 2.0;

/// Compute the Norm-in-Norm loss AND its per-prediction gradient in
/// one pass.
///
/// `preds[i]` is the i-th model output, `labels[i]` is the matching
/// MOS / target. Returns `(loss, grad)` where `grad[i] = dL/d preds[i]`.
///
/// `p` and `q` are the inner / outer norm exponents (paper defaults
/// `p=1, q=2`; PLCC-induced special case is `p=q=2`).
///
/// **Zero-variance safe**: when `Z = ||ŝ||_q ≤ ε` the `ε` in the
/// denominator keeps gradients finite (preds are uniform → `ŝ ≈ 0`
/// → `ŝ_n ≈ 0`, but `Δ ≠ 0` if `s_n` has signal → loss is positive
/// but gradient is small and bounded).
///
/// **Sample-count gate**: with `N < 2` the loss is 0 and the gradient
/// is all zeros — single-sample batch statistics are undefined. The
/// caller (trainer) enforces `minibatch_size ≥ 16` separately.
pub fn compute_norm_in_norm_loss_and_grad(
    preds: &[f64],
    labels: &[f64],
    p: f64,
    q: f64,
) -> (f64, Vec<f64>) {
    let n = preds.len();
    assert_eq!(
        n,
        labels.len(),
        "preds and labels length mismatch ({} vs {})",
        n,
        labels.len()
    );
    if n < 2 {
        return (0.0, vec![0.0; n]);
    }

    // 1. Center preds and labels.
    let m_hat = preds.iter().sum::<f64>() / n as f64;
    let m = labels.iter().sum::<f64>() / n as f64;
    let s_hat: Vec<f64> = preds.iter().map(|&x| x - m_hat).collect();
    let s_lbl: Vec<f64> = labels.iter().map(|&x| x - m).collect();

    // 2. Q-norm of centered preds (the term that flows into the
    //    denominator AND the chain-rule second term).
    let z_pred = q_norm(&s_hat, q);
    let z_pred_e = z_pred + EPS;
    let z_lbl = q_norm(&s_lbl, q);
    let z_lbl_e = z_lbl + EPS;

    // 3. Normalized vectors and residual.
    let mut s_hat_n = vec![0.0f64; n];
    let mut s_lbl_n = vec![0.0f64; n];
    let mut delta = vec![0.0f64; n];
    for i in 0..n {
        s_hat_n[i] = s_hat[i] / z_pred_e;
        s_lbl_n[i] = s_lbl[i] / z_lbl_e;
        delta[i] = s_hat_n[i] - s_lbl_n[i];
    }

    // 4. Loss = (||delta||_p / scale)^p = ||delta||_p^p / scale^p.
    //    scale = 2^max(1, 1/q) · N^max(0, 1/p - 1/q).
    let inv_q = 1.0 / q;
    let inv_p = 1.0 / p;
    let scale = 2f64.powf(inv_q.max(1.0)) * (n as f64).powf((inv_p - inv_q).max(0.0));
    let scale_p = scale.powf(p);
    let abs_delta_p: f64 = delta.iter().map(|&d| d.abs().powf(p)).sum();
    let loss = abs_delta_p / scale_p;

    // 5. Per-Δ gradient g_i = dL/dΔ_i = (p / scale^p) · |Δ_i|^(p-1) · sign(Δ_i).
    //    For p = 1 this is sign(Δ_i)/scale; for p = 2 it's 2·Δ_i/scale².
    //    Convention 0^0 = 0 keeps g_i = 0 at Δ_i = 0 for p = 1 (where
    //    sign(0) = 0 anyway), and avoids NaN for general p ≥ 1.
    let mut g = vec![0.0f64; n];
    for i in 0..n {
        let ad = delta[i].abs();
        if ad == 0.0 {
            g[i] = 0.0;
        } else {
            // |d|^(p-1) · sign(d) = |d|^(p-1) · (d / |d|) — but compute
            // directly via powf to avoid the explicit div for p=1.
            let pow_term = ad.powf(p - 1.0);
            g[i] = (p / scale_p) * pow_term * delta[i].signum();
        }
    }

    // 6. u_j for the denominator-chain correction:
    //    u_j = (1/Z^(q-1)) · |ŝ_j|^(q-1) · sign(ŝ_j).
    //    For q = 2: u_j = ŝ_j / Z (cheap path).
    //    Z_safe protects against Z = 0 (uniform-pred batch) — when ŝ
    //    is all zero, u_j = 0 by construction so we just need a
    //    non-zero divisor.
    let z_safe = z_pred.max(EPS);
    let mut u = vec![0.0f64; n];
    if (q - 2.0).abs() < 1e-12 {
        // Fast path for the canonical q = 2.
        for j in 0..n {
            u[j] = s_hat[j] / z_safe;
        }
    } else {
        let z_pow = z_safe.powf(q - 1.0);
        for j in 0..n {
            let abs_sj = s_hat[j].abs();
            if abs_sj == 0.0 {
                u[j] = 0.0;
            } else {
                u[j] = abs_sj.powf(q - 1.0) * s_hat[j].signum() / z_pow;
            }
        }
    }

    // 7. Aggregates needed for the gradient assembly.
    let mean_g = g.iter().sum::<f64>() / n as f64;
    let mean_u = u.iter().sum::<f64>() / n as f64;
    let dot_gs: f64 = g.iter().zip(s_hat.iter()).map(|(&a, &b)| a * b).sum();

    // 8. dL/dŜ_j = (g_j - mean_g)/ẑ - (u_j - mean_u) · dot_gs / ẑ².
    let mut grad = vec![0.0f64; n];
    let inv_z = 1.0 / z_pred_e;
    let inv_z_sq = inv_z * inv_z;
    for j in 0..n {
        grad[j] = (g[j] - mean_g) * inv_z - (u[j] - mean_u) * dot_gs * inv_z_sq;
    }

    (loss, grad)
}

fn q_norm(v: &[f64], q: f64) -> f64 {
    // For q = 2 the explicit `sqrt(sum_sq)` path avoids the powf round-trip
    // and matches the reference impl's `torch.norm(..., p=2)` precision.
    if (q - 2.0).abs() < 1e-12 {
        let sum_sq: f64 = v.iter().map(|&x| x * x).sum();
        sum_sq.sqrt()
    } else {
        let s: f64 = v.iter().map(|&x| x.abs().powf(q)).sum();
        s.powf(1.0 / q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Numerical gradient check: compare the analytical closed-form
    /// gradient against a centered finite-difference approximation on a
    /// tiny batch. Tolerance 1e-5 relative.
    #[test]
    fn gradient_check_p1_q2() {
        let preds: Vec<f64> = vec![0.3, -0.4, 0.7, 0.1, -0.2, 0.5, -0.6, 0.8];
        let labels: Vec<f64> = vec![10.0, 5.0, 80.0, 30.0, 15.0, 60.0, 8.0, 95.0];
        let p = 1.0;
        let q = 2.0;
        let (l0, grad) = compute_norm_in_norm_loss_and_grad(&preds, &labels, p, q);
        // Sanity: loss must be finite and non-negative.
        assert!(l0.is_finite() && l0 >= 0.0, "loss not finite/non-neg: {l0}");

        let h = 1e-6;
        for i in 0..preds.len() {
            let mut plus = preds.clone();
            let mut minus = preds.clone();
            plus[i] += h;
            minus[i] -= h;
            let (lp, _) = compute_norm_in_norm_loss_and_grad(&plus, &labels, p, q);
            let (lm, _) = compute_norm_in_norm_loss_and_grad(&minus, &labels, p, q);
            let num = (lp - lm) / (2.0 * h);
            let ana = grad[i];
            let rel = (num - ana).abs() / (num.abs().max(ana.abs()).max(1e-8));
            assert!(
                rel < 1e-5,
                "grad mismatch at i={i}: ana={ana:.8} num={num:.8} rel={rel:.2e}"
            );
        }
    }

    /// Same gradient check at p=q=2 (PLCC-induced special case). The
    /// gradient formula collapses to the squared-error form per paper
    /// Eqn 14; verifying it matches autograd is the most important
    /// correctness sanity for the special case.
    #[test]
    fn gradient_check_p2_q2() {
        let preds: Vec<f64> = vec![1.0, -0.5, 2.0, 0.3, -1.2, 0.8, -0.7, 1.5, 0.2, -0.9];
        let labels: Vec<f64> = vec![50.0, 20.0, 90.0, 40.0, 5.0, 70.0, 15.0, 85.0, 35.0, 10.0];
        let p = 2.0;
        let q = 2.0;
        let (l0, grad) = compute_norm_in_norm_loss_and_grad(&preds, &labels, p, q);
        assert!(l0.is_finite() && l0 >= 0.0);

        let h = 1e-6;
        for i in 0..preds.len() {
            let mut plus = preds.clone();
            let mut minus = preds.clone();
            plus[i] += h;
            minus[i] -= h;
            let (lp, _) = compute_norm_in_norm_loss_and_grad(&plus, &labels, p, q);
            let (lm, _) = compute_norm_in_norm_loss_and_grad(&minus, &labels, p, q);
            let num = (lp - lm) / (2.0 * h);
            let ana = grad[i];
            let rel = (num - ana).abs() / (num.abs().max(ana.abs()).max(1e-8));
            assert!(
                rel < 1e-4,
                "grad mismatch at i={i}: ana={ana:.8} num={num:.8} rel={rel:.2e}"
            );
        }
    }

    /// Zero-variance preds (all identical) must not NaN. The eps in
    /// the denominator (`ẑ = Z + ε` with Z = 0) saves the computation;
    /// the gradient is small but finite (the centered preds are all
    /// zero, so `u = 0` and the second chain-term vanishes; the first
    /// term `(g_j - mean(g))/ẑ` is non-zero because labels still vary).
    #[test]
    fn zero_variance_batch_safe() {
        let preds: Vec<f64> = vec![0.5; 16];
        let labels: Vec<f64> = (0..16).map(|i| i as f64 * 5.0).collect();
        let (l, grad) = compute_norm_in_norm_loss_and_grad(&preds, &labels, 1.0, 2.0);
        assert!(l.is_finite(), "loss NaN on zero-var preds: {l}");
        for (i, &g) in grad.iter().enumerate() {
            assert!(g.is_finite(), "grad[{i}] NaN on zero-var preds: {g}");
        }
    }

    /// Single-sample batch is a degenerate case — return (0, [0]).
    /// The trainer enforces minibatch_size ≥ 16 separately, so this
    /// just guards against accidental misuse.
    #[test]
    fn single_sample_returns_zero() {
        let (l, g) = compute_norm_in_norm_loss_and_grad(&[1.0], &[50.0], 1.0, 2.0);
        assert_eq!(l, 0.0);
        assert_eq!(g, vec![0.0]);
    }

    /// Gradient sum should be (approximately) zero — the loss is
    /// invariant to a uniform shift of all predictions (because of
    /// the mean-centering step). So `Σ_j dL/dŜ_j = 0`.
    #[test]
    fn gradient_sums_to_zero_under_centering_invariance() {
        let preds: Vec<f64> = vec![0.3, -0.4, 0.7, 0.1, -0.2, 0.5, -0.6, 0.8, 0.0, 0.4];
        let labels: Vec<f64> = vec![10.0, 5.0, 80.0, 30.0, 15.0, 60.0, 8.0, 95.0, 50.0, 45.0];
        for &(p, q) in &[(1.0, 2.0), (2.0, 2.0), (1.5, 2.0)] {
            let (_, grad) = compute_norm_in_norm_loss_and_grad(&preds, &labels, p, q);
            let s: f64 = grad.iter().sum();
            assert!(
                s.abs() < 1e-9,
                "gradient does not sum to zero at p={p} q={q}: sum={s:.2e}"
            );
        }
    }
}
