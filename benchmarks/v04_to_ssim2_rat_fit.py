#!/usr/bin/env python3
"""Rational polynomial fit z → SSIM2 score using scipy + Sanathanan-Koerner.

Same input/target as poly_shift.py — but fits P(z)/Q(z) instead of P(z).
"""
import bisect
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

V02_ANCHORS = [
    (-59.8119, 100.0000), (-46.3979, 86.3832), (-39.8782, 82.4838),
    (-35.2807, 79.5798), (-30.7110, 76.5164), (-25.9624, 73.4357),
    (-20.0988, 69.9280), (-12.7256, 65.9895), (-3.0700, 61.2748),
    (4.6772, 55.6898),   (10.5595, 48.6972),  (17.3930, 40.9081),
    (24.9022, 32.4060),  (33.9656, 22.3016),  (45.3017, 11.4260),
    (57.9030, -0.1502),  (72.9362, -12.2887), (92.7248, -28.9030),
    (124.7492, -50.6577),(176.7873, -81.7489),(623.7492, -100.0000),
]
V02_X = [a[0] for a in V02_ANCHORS]


def v02_calib(d):
    if d <= V02_X[0]:
        return V02_ANCHORS[0][1]
    if d >= V02_X[-1]:
        return V02_ANCHORS[-1][1]
    i = bisect.bisect_right(V02_X, d)
    x0, y0 = V02_ANCHORS[i - 1]
    x1, y1 = V02_ANCHORS[i]
    return y0 + (d - x0) / (x1 - x0) * (y1 - y0)


def fit_rational(z, y, p_deg, q_deg, max_iter=30, tol=1e-10):
    """Sanathanan–Koerner iterative reweighted least squares for P(z)/Q(z).

    P has p_deg+1 free coefficients; Q has constant term fixed at 1, so q_deg
    free coefficients (q[1..q_deg]). Total free params: p_deg+1+q_deg.
    """
    n = len(z)
    # Reset per-call state.
    if hasattr(fit_rational, "_last_p"):
        del fit_rational._last_p
    if hasattr(fit_rational, "_last_q"):
        del fit_rational._last_q
    # Initial weight: uniform (equivalent to fitting y*Q(z) = P(z) by linear LS).
    w = np.ones(n)
    p = np.zeros(p_deg + 1)
    q = np.concatenate(([1.0], np.zeros(q_deg)))  # constant=1.
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    last_err = float("inf")
    for it in range(max_iter):
        # Linearized: (1/w) * (P(z) − y * Q_aux(z)) = 0
        # Build design matrix [z^0 .. z^p | -y z^1 .. -y z^q] (constant of Q is fixed at 1)
        cols_p = np.stack([z**k for k in range(p_deg + 1)], axis=1)  # (n, p_deg+1)
        cols_q = np.stack([-y * z**(k + 1) for k in range(q_deg)], axis=1)  # (n, q_deg)
        A = np.hstack([cols_p, cols_q])
        # Right-hand side: y * Q(z) approx P(z) → with Q's constant 1, b = y * 1
        b = y.copy()
        # Apply weighting w_i = 1/|Q(z_i)|.
        Wsqrt = np.diag(np.sqrt(w))
        # Use np.linalg.lstsq on weighted system.
        Aw = A * np.sqrt(w)[:, None]
        bw = b * np.sqrt(w)
        coeffs, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        p = coeffs[: p_deg + 1]
        q = np.concatenate(([1.0], coeffs[p_deg + 1 :]))
        # Compute new weights.
        Q_z = np.polyval(q[::-1], z)  # np.polyval expects high-to-low
        # Avoid division by zero.
        denom = np.where(np.abs(Q_z) < 1e-9, 1e-9, np.abs(Q_z))
        w_new = 1.0 / denom**2
        # Convergence: change in p and q.
        delta_p = np.linalg.norm(p - getattr(fit_rational, "_last_p", np.zeros_like(p)))
        delta_q = np.linalg.norm(q - getattr(fit_rational, "_last_q", np.zeros_like(q)))
        fit_rational._last_p = p.copy()
        fit_rational._last_q = q.copy()
        # Compute error.
        P_z = np.polyval(p[::-1], z)
        y_hat = P_z / Q_z
        rmse = np.sqrt(((y_hat - y) ** 2).mean())
        if abs(last_err - rmse) < tol:
            break
        last_err = rmse
        w = w_new
    return p, q, rmse, it + 1


def eval_rational(p, q, x):
    """Evaluate P(x)/Q(x) at scalar or array x."""
    P = np.polyval(p[::-1], x)
    Q = np.polyval(q[::-1], x)
    return P / Q


def main():
    df = pd.read_csv("/mnt/v/output/zensim/per_pair_v04_ssim2_holdout.csv")
    df = df[df["dataset"] != "KonJND-1k"].copy()
    df["z"] = df["v04_distance"].apply(v02_calib)
    z = df["z"].values
    y = df["fast_ssim2_score"].values
    print(f"n = {len(z)} pairs")
    print()

    # Polynomial baseline (degrees 1..7).
    print(f"{'shape':<25} {'RMSE':>8} {'monotonic [-100,100]':>22}")
    print("-" * 60)
    for deg in range(1, 8):
        coeffs = np.polyfit(z, y, deg=deg)
        yhat = np.polyval(coeffs, z)
        rmse = np.sqrt(((yhat - y) ** 2).mean())
        zs = np.linspace(-100, 100, 2001)
        ys_eval = np.polyval(coeffs, zs)
        diffs = np.diff(ys_eval)
        mono = bool((diffs >= -1e-9).all()) or bool((diffs <= 1e-9).all())
        print(f"{'poly p(' + str(deg) + ') z->ssim2':<25} {rmse:>8.3f} {str(mono):>22}")

    print()
    # Rational P/Q at various (p_deg, q_deg).
    for p_deg, q_deg in [(2, 2), (3, 2), (2, 3), (3, 3), (4, 3), (3, 4), (4, 4), (5, 4), (4, 5)]:
        try:
            p, q, rmse, niter = fit_rational(z, y, p_deg, q_deg)
            zs = np.linspace(-100, 100, 2001)
            yseval = eval_rational(p, q, zs)
            diffs = np.diff(yseval)
            mono = bool((diffs >= -1e-9).all()) or bool((diffs <= 1e-9).all())
            print(f"{'rat (' + str(p_deg) + '/' + str(q_deg) + ') z->ssim2':<25} {rmse:>8.3f} {str(mono):>22}  iter={niter}")
        except Exception as e:
            print(f"rat ({p_deg}/{q_deg}) FAILED: {e}")

    # Try the best rational fit, dump its coefficients in a Rust-friendly form.
    print()
    p_best, q_best, rmse_best, _ = fit_rational(z, y, 4, 4)
    print(f"Best rational (4/4) coefficients (lowest-degree-first):")
    print(f"  P = {[f'{c:.6e}' for c in p_best.tolist()]}")
    print(f"  Q = {[f'{c:.6e}' for c in q_best.tolist()]}")
    print(f"  RMSE = {rmse_best:.4f}")


if __name__ == "__main__":
    main()
