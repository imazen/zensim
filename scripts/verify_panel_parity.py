#!/usr/bin/env python3
"""Cross-check selected canonical Rust panel statistics with Python references.

The 36-case gate compares current panel definitions at tolerance 1e-9.
Both references consume Rust's --emit-rescaled predictions, so this verifies
statistics conditional on that fit, NOT the logistic optimizer end to end.
SciPy supplies rank/correlation checks; PWRC and OR share Python helpers across
the two reference columns and are not two independent implementations.

Current PWRC is SA-ST AUC and OR thresholds logistic-rescaled residuals at
1.96 times target sigma. The retired mohammadi_eval.py used different PWRC/OR
instruments; a PASS here does not establish equivalence with that historical
script or authorize arbitrary statistical substitutions. Global and per-sample
Z-RMSE use different normalizers; the latter is not the global-panel gate.

The companion Rust tests pin small goldens in normal CI. Full cross-language
integration tests are ignored by default and must be invoked explicitly.

Usage:
  python3 scripts/verify_panel_parity.py [--bin path/to/panel] [--tol 1e-9]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import kendalltau, pearsonr, spearmanr


# ----------------------------------------------------------------------
# scipy_ref — SciPy correlations plus current panel PWRC/OR helpers
# ----------------------------------------------------------------------

def _logistic_4param(b, x):
    b4 = max(abs(b[3]), 1e-8) * (1 if b[3] >= 0 else -1)
    arg = np.clip(-(x - b[2]) / b4, -500, 500)
    return b[1] + (b[0] - b[1]) / (1 + np.exp(arg))


def _fit_logistic(pred, target):
    def residuals(b):
        return _logistic_4param(b, pred) - target

    b0 = [max(target), min(target), float(np.median(pred)), float(np.std(pred))]
    res = least_squares(residuals, b0, method="lm", max_nfev=5000)
    return _logistic_4param(res.x, pred)


def scipy_ref(pred, target, sigma, rescaled_rust=None):
    pred = np.asarray(pred, float)
    target = np.asarray(target, float)
    s, _ = spearmanr(pred, target)
    k, _ = kendalltau(pred, target)
    # PLCC / OR / Z-RMSE depend on the logistic rescale. When the Rust
    # bin's exact rescaled scores are provided (--emit-rescaled), use
    # them so the gate isolates the stat math from the optimizer.
    rescaled = np.asarray(rescaled_rust, float) if rescaled_rust is not None else _fit_logistic(pred, target)
    p, _ = pearsonr(rescaled, target)
    # PWRC = SA-ST AUC (Mohammadi 2025 § VII), computed on the
    # logistic-rescaled prediction so anti-correlated raw pred is
    # polarity-handled by the rescale (perfect anti-rank → rescaled is
    # positively correlated → SA-ST AUC = 1).
    pw = _pwrc_sa_st_auc_panel(list(rescaled), list(target))
    # OR = P.1401 / Mohammadi Eq 2-4: τ = 1.96·σ_target on the rescaled
    # residual. (mohammadi_eval.py:56 used 2σ on the residual STD —
    # close but not identical. The paper text gives 1.96·σ on the
    # target column; we follow the paper.)
    out = _outlier_ratio_panel(list(rescaled), list(target))
    # Z-RMSE per-sample (mohammadi_eval.py:82)
    if sigma is not None:
        sg = np.asarray(sigma, float)
        valid = sg > 0
        z = (rescaled[valid] - target[valid]) / sg[valid]
        zr = float(np.sqrt(np.mean(z ** 2)))
    else:
        zr = float("nan")
    return {
        "srocc": abs(s),
        "plcc": abs(p),
        "krocc": abs(k),
        "or": out,
        "pwrc": pw,
        "z_rmse_scipy": zr,
    }


# ----------------------------------------------------------------------
# panel_def_ref — a faithful pure-Python mirror of panel.rs's EXACT
# definitions. This is what `zen_stats.py` would carry if a Python
# mirror is ever needed; it exists here to prove the Rust home is
# reproducible in Python to <= 1e-9 across ALL six stats (including OR
# and the global Z-RMSE, where scipy_ref intentionally differs).
# ----------------------------------------------------------------------

def _ranks(v):
    # Mirror panel.rs `ranks` (panel.rs:30): average ranks for ties,
    # 1e-12 tie tolerance, 0-based mean-centered ranks.
    n = len(v)
    idx = sorted(range(n), key=lambda i: v[i])
    r = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(v[idx[j]] - v[idx[i]]) < 1e-12:
            j += 1
        avg = (i + j - 1) / 2.0
        for k in range(i, j):
            r[idx[k]] = avg
        i = j
    return r


def _spearman(a, b):
    n = len(a)
    if n < 2:
        return 0.0
    ra, rb = _ranks(a), _ranks(b)
    mean = (n - 1) / 2.0
    num = da = db = 0.0
    for i in range(n):
        xa = ra[i] - mean
        xb = rb[i] - mean
        num += xa * xb
        da += xa * xa
        db += xb * xb
    den = math.sqrt(da * db)
    return 0.0 if den < 1e-12 else num / den


def _pearson(a, b):
    n = len(a)
    if n < 2:
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    num = da = db = 0.0
    for i in range(n):
        xa = a[i] - ma
        xb = b[i] - mb
        num += xa * xb
        da += xa * xa
        db += xb * xb
    den = math.sqrt(da * db)
    return 0.0 if den < 1e-12 else num / den


def _kendall(a, b):
    n = len(a)
    if n < 2:
        return 0.0
    c = d = ta = tb = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if abs(da) < 1e-12 and abs(db) < 1e-12:
                continue
            elif abs(da) < 1e-12:
                ta += 1
            elif abs(db) < 1e-12:
                tb += 1
            elif da * db > 0:
                c += 1
            else:
                d += 1
    den = math.sqrt((c + d + ta) * (c + d + tb))
    return 0.0 if den < 1e-12 else (c - d) / den


def _pwrc_sa_st_auc_panel(scores, humans, n_points=128):
    """Mirror panel.rs `pwrc_sa_st_auc` — Mohammadi 2025 § VII SA-ST
    Sorting-Accuracy AUC.

    For each Sensory Threshold ST ∈ [0, max_subj_gap], count the
    fraction of (i,j) pairs with |humans_gap| > ST that the metric
    ranks correctly (sign(humans_diff) == sign(scores_diff)). The
    return value is the trapezoidal AUC normalised by max_subj_gap,
    so perfect rank → 1 and perfect anti-rank → 0.
    """
    n = min(len(scores), len(humans))
    if n < 2 or n_points < 2:
        return 0.0
    pairs = []  # (gap, correct)
    for i in range(n):
        for j in range(i + 1, n):
            dh = humans[j] - humans[i]
            ds = scores[j] - scores[i]
            if not math.isfinite(dh) or not math.isfinite(ds) or dh == 0.0 or ds == 0.0:
                continue
            correct = (dh > 0.0) == (ds > 0.0)
            pairs.append((abs(dh), correct))
    if not pairs:
        return 0.0
    st_max = max(g for g, _ in pairs)
    if st_max <= 0.0:
        return 0.0
    curve = []
    for k in range(n_points):
        frac = k / (n_points - 1)
        st = frac * st_max
        active = 0
        correct = 0
        for gap, ok in pairs:
            if gap > st:
                active += 1
                if ok:
                    correct += 1
        if active == 0:
            sa = curve[-1][1] if curve else 0.0
        else:
            sa = correct / active
        curve.append((st, sa))
    # Trapezoidal AUC, normalise by st_max.
    auc = 0.0
    for (st0, sa0), (st1, sa1) in zip(curve[:-1], curve[1:]):
        dt = st1 - st0
        if dt > 0.0:
            auc += 0.5 * (sa0 + sa1) * dt
    return auc / st_max


def _outlier_ratio_panel(pred, target):
    """Mirror panel.rs `outlier_ratio` — ITU-T P.1401 § C.4 / Mohammadi
    2025 Eq 2-4. Fraction of stimuli where |pred - target| > 1.96·σ_target.

    Caller MUST pass `pred` already on `target`'s scale (i.e., 4-param-
    logistic-rescaled). panel.rs `compute_panel` does this internally.
    """
    n = min(len(pred), len(target))
    if n < 2:
        return float("nan")
    mt = sum(target) / n
    vt = sum((x - mt) ** 2 for x in target) / n
    sigma = max(math.sqrt(vt), 1e-12)
    tau = 1.96 * sigma
    outliers = 0
    counted = 0
    for i in range(n):
        r = pred[i] - target[i]
        if not math.isfinite(r):
            continue
        if abs(r) > tau:
            outliers += 1
        counted += 1
    if counted == 0:
        return float("nan")
    return outliers / counted


def _z_rmse_global_panel(pred, target):
    # Mirror panel.rs:193 — divide by target's GLOBAL σ.
    n = len(pred)
    if n < 2:
        return float("nan")
    mt = sum(target) / n
    vt = sum((x - mt) ** 2 for x in target) / n
    sigma = max(math.sqrt(vt), 1e-9)
    ss = 0.0
    cnt = 0
    for i in range(n):
        z = (pred[i] - target[i]) / sigma
        if math.isfinite(z):
            ss += z * z
            cnt += 1
    return float("nan") if cnt == 0 else math.sqrt(ss / cnt)


def _rescale_affine(pred, target):
    n = len(pred)
    mp = sum(pred) / n
    mt = sum(target) / n
    cov = vp = 0.0
    for i in range(n):
        dp = pred[i] - mp
        dt = target[i] - mt
        cov += dp * dt
        vp += dp * dp
    b = 0.0 if abs(vp) < 1e-12 else cov / vp
    a = mt - b * mp
    return [a + b * p for p in pred]


def panel_def_ref(pred, target, sigma, rescaled_rust=None):
    """Compute the panel using panel.rs's EXACT definitions.

    SROCC / KROCC / PWRC / OR are rescale-independent and computed
    directly from panel.rs's formulas (note: SROCC/KROCC/PWRC are
    .abs(), and panel.rs weights PWRC by the human/target ranks; OR
    panel.rs:129 is the polarity-aligned z-score residual).

    PLCC and the global Z-RMSE depend on the 4-param logistic rescale.
    Reimplementing panel.rs's 13-start Levenberg-Marquardt in Python to
    1e-9 is impractical, so this reference uses panel.rs's OWN rescaled
    scores (`rescaled_rust`, from the bin's `--emit-rescaled` path) when
    provided — that removes the optimizer difference and isolates the
    stat math. PLCC = |pearson(rescaled, target)|; Z-RMSE = global-σ
    z_rmse(rescaled, target).
    """
    s = abs(_spearman(target, pred))
    k = abs(_kendall(target, pred))
    rescaled = list(rescaled_rust) if rescaled_rust is not None \
        else list(_fit_logistic(np.asarray(pred, float), np.asarray(target, float)))
    plcc = abs(_pearson(rescaled, list(target)))
    # PWRC + OR are computed on the LOGISTIC-RESCALED prediction —
    # both depend on having pred on target's scale (panel.rs `compute_panel`
    # convention). Anti-correlated raw pred + rescaled → positively-
    # correlated rescaled → SA-ST AUC = 1 (paper-correct polarity).
    pw = _pwrc_sa_st_auc_panel(rescaled, list(target))
    out = _outlier_ratio_panel(rescaled, list(target))
    zr_global = _z_rmse_global_panel(rescaled, list(target))
    return {
        "srocc": s,
        "plcc": plcc,
        "krocc": k,
        "pwrc": pw,
        "or": out,
        "z_rmse": zr_global,
    }


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def gen_case(seed, n, kind):
    rng = np.random.default_rng(seed)
    target = rng.uniform(0, 100, n)
    if kind == "linear_noisy":
        pred = 0.9 * target + rng.normal(0, 8, n) + 3
    elif kind == "saturating":
        pred = 100 / (1 + np.exp(-(target - 50) / 12)) + rng.normal(0, 5, n)
    elif kind == "distance_shaped":  # anti-correlated (low=good)
        pred = 30 - 0.25 * target + rng.normal(0, 2, n)
    elif kind == "weak":
        pred = 0.3 * target + rng.normal(0, 25, n)
    else:
        raise ValueError(kind)
    sigma = rng.uniform(2, 15, n)
    return pred, target, sigma


def _write_tsv(pred, target, sigma):
    f = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False)
    f.write("predicted\ttarget\tsigma\n")
    # repr(float(...)) is the shortest round-trippable decimal — Rust's
    # f64 parser reads it back bit-identically. (numpy's own repr emits
    # "np.float64(...)" which Rust can't parse, so cast to builtin float.)
    for p, t, s in zip(pred, target, sigma):
        f.write(f"{float(p)!r}\t{float(t)!r}\t{float(s)!r}\n")
    f.close()
    return f.name


def run_rust(bin_path, tsv_path):
    out = subprocess.run(
        [bin_path, "--input", tsv_path, "--json"],
        capture_output=True, text=True, timeout=120, check=True,
    )
    return json.loads(out.stdout)["groups"][0]  # ALL


def run_rust_rescaled(bin_path, tsv_path):
    """Return panel.rs's exact 4-param-logistic-rescaled predicted
    column (finite rows of the ALL group), via --emit-rescaled."""
    out = subprocess.run(
        [bin_path, "--input", tsv_path, "--emit-rescaled"],
        capture_output=True, text=True, timeout=120, check=True,
    )
    return [float(x) for x in out.stdout.split()]


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--bin", default=None,
                    help="path to the `panel` binary (default: search target/{release,debug})")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    bin_path = args.bin
    if bin_path is None:
        for cand in (
            os.path.join(here, "target", "release", "panel"),
            os.path.join(here, "target", "debug", "panel"),
        ):
            if os.path.exists(cand):
                bin_path = cand
                break
    if bin_path is None or not os.path.exists(bin_path):
        print("ERROR: `panel` binary not found — build with "
              "`cargo build -p zensim-validate --bin panel` first", file=sys.stderr)
        return 2

    # panel_def_ref is the AUTHORITATIVE gate: it reproduces panel.rs's
    # exact definitions in Python and (using panel.rs's own rescaled
    # scores for the logistic-dependent stats) gates all six to <= tol.
    panel_def_gated = ["srocc", "plcc", "krocc", "pwrc", "or", "z_rmse"]
    # SciPy cross-checks correlations. PWRC/OR reuse the same Python helpers
    # as panel_def_ref. Preserve the existing gate set; OR is report-only
    # in this column and gated against panel_def_ref.
    scipy_gated = ["srocc", "plcc", "krocc", "pwrc"]

    cases = [
        (s, n, kind)
        for s in (1, 2, 3)
        for n in (40, 120, 400)
        for kind in ("linear_noisy", "saturating", "distance_shaped", "weak")
    ]

    # Track max divergence per (reference, stat).
    max_div_scipy = {k: 0.0 for k in ["srocc", "plcc", "krocc", "or", "pwrc"]}
    max_div_paneldef = {k: 0.0 for k in panel_def_gated}
    n_cases = 0

    for seed, n, kind in cases:
        pred, target, sigma = gen_case(seed, n, kind)
        tsv = _write_tsv(pred, target, sigma)
        try:
            rust = run_rust(bin_path, tsv)
            rescaled_rust = run_rust_rescaled(bin_path, tsv)
        finally:
            os.unlink(tsv)
        # Both references use panel.rs's own rescaled scores so the
        # logistic-optimizer difference is removed and the gate tests
        # the stat math, not the curve-fit convergence point.
        sc = scipy_ref(pred, target, sigma, rescaled_rust=rescaled_rust)
        pd = panel_def_ref(pred, target, sigma, rescaled_rust=rescaled_rust)
        n_cases += 1

        for k in max_div_scipy:
            if k in rust and rust[k] is not None and not math.isnan(sc[k]):
                max_div_scipy[k] = max(max_div_scipy[k], abs(rust[k] - sc[k]))
        for k in panel_def_gated:
            rv = rust[k]
            if rv is not None and not math.isnan(pd[k]):
                max_div_paneldef[k] = max(max_div_paneldef[k], abs(rv - pd[k]))

    tol = args.tol
    print(f"# panel parity cross-check — {n_cases} synthetic cases "
          f"(seeds 1-3 x n in {{40,120,400}} x 4 shapes), tol={tol:g}")
    print()
    print("## vs panel_def_ref (faithful pure-Python mirror of panel.rs definitions)")
    print("## (uses Rust --emit-rescaled scores for all fit-dependent statistics)")
    print(f"{'stat':<10} {'max_div':>14} {'gate':>8}")
    fail = False
    for k in panel_def_gated:
        ok = max_div_paneldef[k] <= tol
        flag = "" if ok else "  <-- FAIL"
        print(f"{k:<10} {max_div_paneldef[k]:>14.3e} {'GATED':>8}{flag}")
        if not ok:
            fail = True
    print()
    print("## vs scipy_ref (SciPy correlations; shared Python PWRC/OR helpers)")
    print(f"{'stat':<10} {'max_div':>14} {'gate':>8}")
    for k in ["srocc", "plcc", "krocc", "pwrc", "or"]:
        is_gated = k in scipy_gated
        ok = (max_div_scipy[k] <= tol) if is_gated else True
        gate = "GATED" if is_gated else "report"
        flag = "" if ok else "  <-- FAIL"
        note = ""
        if k == "or":
            note = "  (same Python OR helper as panel_def_ref; report-only here)"
        print(f"{k:<10} {max_div_scipy[k]:>14.3e} {gate:>8}{flag}{note}")
        if is_gated and not ok:
            fail = True

    print()
    if fail:
        print("RESULT: FAIL — a GATED stat diverged > tol. This means panel.rs "
              "and the Python reference have a real algorithmic difference "
              "(tie-handling / NaN-drop / formula) that must be reconciled.")
        return 1
    print("RESULT: PASS — every GATED stat agrees to <= tol on these fixtures, "
          "conditional on Rust's logistic-rescaled predictions.")
    print()
    print("LIMIT: this gate does not independently validate the logistic fit, "
          "all edge cases, or the retired historical instrument. Keep the "
          "Python reference as a check on the actual Rust owner.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
