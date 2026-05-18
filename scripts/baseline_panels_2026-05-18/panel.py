"""Mohammadi 2025 statistical panel computation in pure Python.

Mirrors the Rust reference in `zensim-validate/src/panel.rs` /
`zensim-validate/src/bin/bake_verdict.rs`. Used by the baseline-panel
extraction pipeline to compute SROCC / PLCC / KROCC / OR / PWRC /
Z-RMSE for ssim2, cvvdp, iwssim per validation corpus.

Polarity is handled by `.abs()` on rank-based stats (SROCC, KROCC,
PWRC) and by a 4-parameter logistic fit before PLCC / Z-RMSE — same
convention as the Rust reference.

Validates against Mohammadi 2025 anchor CSV Z-RMSE values:
    SSIMULACRA2 = 47.63 ± 0.5
    IW-SSIM     = 31.51 ± 0.5
    CVVDP       =  9.45 ± 0.5
    PSNR-Y      = 13.36 ± 0.5
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import kendalltau, rankdata, spearmanr


@dataclass
class PanelStats:
    n: int
    srocc: float
    plcc: float
    krocc: float
    or_ratio: float
    pwrc: float
    z_rmse: float


def _spearman_abs(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    r, _ = spearmanr(a, b)
    if math.isnan(r):
        return 0.0
    return abs(r)


def _kendall_abs(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    r, _ = kendalltau(a, b, variant="b")
    if math.isnan(r):
        return 0.0
    return abs(r)


def _outlier_ratio(predicted: np.ndarray, target: np.ndarray) -> float:
    """OR per bake_verdict.rs: residuals after polarity-aligned z-score
    standardization of predicted vs target; OR = fraction of residuals
    > 2σ of the residual distribution.
    """
    n = len(predicted)
    if n < 4:
        return float("nan")
    mean_p = predicted.mean()
    mean_t = target.mean()
    sd_p = max(predicted.std(ddof=0), 1e-12)
    sd_t = max(target.std(ddof=0), 1e-12)
    corr = np.corrcoef(predicted, target)[0, 1]
    polarity = -1.0 if corr < 0 else 1.0
    zp = polarity * (predicted - mean_p) / sd_p
    zt = (target - mean_t) / sd_t
    resid = np.abs(zp - zt)
    mean_r = resid.mean()
    sd_r = max(resid.std(ddof=0), 1e-12)
    return float(np.mean(np.abs(resid - mean_r) > 2.0 * sd_r))


def _pwrc(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson-weighted rank correlation per Mohammadi 2025: rank
    transform both inputs, weight by distance from rank midpoint,
    then weighted Pearson on the ranks.
    """
    n = len(a)
    if n < 4:
        return 0.0
    ra = rankdata(a, method="average") - 1.0  # 0-based ranks
    rb = rankdata(b, method="average") - 1.0
    mid = (n - 1) / 2.0
    max_dev = max(mid, 1e-12)
    w = np.abs(ra - mid) / max_dev
    wsum = w.sum()
    if wsum < 1e-12:
        return 0.0
    mean_a = (w * ra).sum() / wsum
    mean_b = (w * rb).sum() / wsum
    xa = ra - mean_a
    xb = rb - mean_b
    num = (w * xa * xb).sum()
    da = (w * xa * xa).sum()
    db = (w * xb * xb).sum()
    den = math.sqrt(da * db)
    if den < 1e-12:
        return 0.0
    return abs(num / den)


def _logistic_4p(b, x):
    """4-parameter logistic in the convention used by bake_verdict.rs
    rescale_logistic: pred = b[1] + (b[0] - b[1]) / (1 + exp(-(x - b[2]) / b[3]))
    """
    b3 = b[3] if abs(b[3]) > 1e-12 else (1e-12 if b[3] >= 0 else -1e-12)
    arg = -(x - b[2]) / b3
    arg = np.clip(arg, -700, 700)
    return b[1] + (b[0] - b[1]) / (1.0 + np.exp(arg))


def _logistic_residuals(b, x, y):
    return _logistic_4p(b, x) - y


def _fit_logistic_4p(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Multi-start LM fit for the 4-parameter logistic. Mirrors the
    13 starts in bake_verdict.rs rescale_logistic. Returns the best
    parameter vector by SSR.
    """
    n = len(predicted)
    if n < 4:
        return None
    mean_p = predicted.mean()
    std_p = max(predicted.std(ddof=0), 1e-12)
    t_min, t_max = target.min(), target.max()
    p_min, p_max = predicted.min(), predicted.max()
    t_span = max(abs(t_max - t_min), 1.0)
    tail = 1000.0 * t_span
    corr = np.corrcoef(predicted, target)[0, 1]
    b4_sign = -1.0 if corr < 0 else 1.0
    b3_high = p_max + 25.0 * std_p
    b3_low = p_min - 25.0 * std_p
    starts = [
        [t_max, t_min, mean_p, math.copysign(max(std_p * b4_sign, 1e-3), b4_sign)],
        [t_max, t_min, mean_p, math.copysign(std_p * 0.1 * b4_sign, b4_sign)],
        [t_max, t_min, mean_p, math.copysign(std_p * 10.0 * b4_sign, b4_sign)],
        [t_max, t_min, mean_p + std_p, math.copysign(std_p * b4_sign, b4_sign)],
        [t_max, t_min, mean_p - std_p, math.copysign(std_p * b4_sign, b4_sign)],
        [-tail, t_max, mean_p, math.copysign(max(std_p * b4_sign, 1e-3), b4_sign)],
        [t_max, -tail, mean_p, math.copysign(max(-std_p * b4_sign, 1e-3), b4_sign)],
        [tail, t_min, mean_p, math.copysign(max(std_p * b4_sign, 1e-3), b4_sign)],
        [t_min, tail, mean_p, math.copysign(max(-std_p * b4_sign, 1e-3), b4_sign)],
        [-tail, t_max, b3_high, math.copysign(max(std_p * b4_sign, 1e-3), b4_sign)],
        [t_max, -tail, b3_low, math.copysign(max(-std_p * b4_sign, 1e-3), b4_sign)],
        [tail, t_min, b3_low, math.copysign(max(std_p * b4_sign, 1e-3), b4_sign)],
        [t_min, tail, b3_high, math.copysign(max(-std_p * b4_sign, 1e-3), b4_sign)],
    ]
    best = None
    best_cost = math.inf
    for s in starts:
        try:
            res = least_squares(
                _logistic_residuals,
                s,
                args=(predicted, target),
                method="lm",
                max_nfev=2000,
            )
            if res.cost < best_cost and math.isfinite(res.cost):
                best_cost = res.cost
                best = res.x
        except Exception:
            continue
    return best


def _rescale_logistic(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """4-parameter logistic rescale, with fallback to affine if the
    LM solve fails or produces non-finite values. Polarity is absorbed
    into the b[3] sign convention.
    """
    n = len(predicted)
    if n < 4:
        return predicted.copy()
    if not (np.isfinite(predicted).all() and np.isfinite(target).all()):
        return _rescale_affine(predicted, target)
    b = _fit_logistic_4p(predicted, target)
    if b is None:
        return _rescale_affine(predicted, target)
    fit = _logistic_4p(b, predicted)
    if not np.isfinite(fit).all():
        return _rescale_affine(predicted, target)
    return fit


def _rescale_affine(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    n = len(predicted)
    mean_p = predicted.mean()
    mean_t = target.mean()
    cov = ((predicted - mean_p) * (target - mean_t)).sum()
    var_p = ((predicted - mean_p) ** 2).sum()
    if var_p < 1e-12:
        return predicted.copy()
    b = cov / var_p
    a = mean_t - b * mean_p
    return a + b * predicted


def _z_rmse(rescaled: np.ndarray, target: np.ndarray, sigma: np.ndarray | None) -> float:
    """Z-RMSE per Mohammadi 2025: when per-sample sigma is provided,
    z = (pred - target) / sigma per row; else fall back to corpus-wide
    sigma. The bake_verdict.rs reference uses corpus-wide sigma always
    (per-sample sigma "unavailable from parquet sidecars"). We support
    per-sample sigma for the AIC-3 anchor and CID22 (if bootstrap σ
    available), corpus-wide sigma elsewhere.
    """
    n = len(rescaled)
    if n < 2 or len(target) != n:
        return float("nan")
    diff = rescaled - target
    if sigma is not None:
        sigma = np.where(sigma > 1e-9, sigma, 1e-9)
        z = diff / sigma
    else:
        sigma_global = max(target.std(ddof=0), 1e-9)
        z = diff / sigma_global
    finite = np.isfinite(z)
    if not finite.any():
        return float("nan")
    return float(math.sqrt((z[finite] ** 2).mean()))


def compute_panel(
    scores: np.ndarray,
    humans: np.ndarray,
    sigma: np.ndarray | None = None,
) -> PanelStats:
    """Compute the full Mohammadi 2025 panel for one (scores, humans)
    pair. `sigma` is optional per-sample σ (length-n array); when None
    Z-RMSE falls back to corpus-wide σ.
    """
    scores = np.asarray(scores, dtype=float)
    humans = np.asarray(humans, dtype=float)
    finite = np.isfinite(scores) & np.isfinite(humans)
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        finite &= np.isfinite(sigma)
    scores = scores[finite]
    humans = humans[finite]
    if sigma is not None:
        sigma = sigma[finite]
    n = len(scores)
    if n == 0:
        return PanelStats(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    srocc = _spearman_abs(humans, scores)
    krocc = _kendall_abs(humans, scores)
    pw = _pwrc(humans, scores)
    or_ = _outlier_ratio(scores, humans)
    rescaled = _rescale_logistic(scores, humans)
    plcc = abs(np.corrcoef(rescaled, humans)[0, 1]) if n >= 2 else float("nan")
    z = _z_rmse(rescaled, humans, sigma)
    return PanelStats(n=n, srocc=srocc, plcc=plcc, krocc=krocc, or_ratio=or_, pwrc=pw, z_rmse=z)


def validate_against_anchor() -> dict:
    """Validate the implementation against the Mohammadi 2025 anchor
    CSV at
    /mnt/v/input/datasets/aic3/EvaluationMetrics/Anchor_assessment_on_PTC_full_resolution_Aug_3_2025.csv

    Returns a dict with computed Z-RMSE per metric and the paper's
    expected values. The Rust reference asserts within 0.5 — same
    tolerance here.
    """
    import csv

    path = "/mnt/v/input/datasets/aic3/EvaluationMetrics/Anchor_assessment_on_PTC_full_resolution_Aug_3_2025.csv"
    rows = list(csv.DictReader(open(path)))
    mos = np.array([float(r["distortion"]) for r in rows])
    sigma = np.array([float(r["std_bootstrap"]) for r in rows])
    out = {}
    for key, paper in (
        ("SSIMULACRA2", 47.63),
        ("iw_ssim", 31.51),
        ("CVVDP", 9.45),
        ("psnry", 13.36),
    ):
        vals = np.array([float(r[key]) for r in rows])
        rescaled = _rescale_logistic(vals, mos)
        z = _z_rmse(rescaled, mos, sigma)
        out[key] = (z, paper, abs(z - paper))
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(validate_against_anchor(), indent=2))
