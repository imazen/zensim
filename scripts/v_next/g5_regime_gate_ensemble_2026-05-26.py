#!/usr/bin/env python3
"""
G5 regime-routed 2-bake ensemble — offline combiner test.

Hypothesis (CODEC_TARGET_GOALS.md G5): a 2-bake ensemble that ROUTES
near-lossless / high-fidelity (HF, KonJND-like) inputs to a KonJND-
specialist and everything else to the shipped V39 general bake can get
**both** KonJND SROCC >= 0.70 AND CID22/KADIK/TID/AIC-3/AIC-4 within
-0.01 of V39.

This tests OPTION (b) from the brief: a transparent score-gated blend,
no runtime change required to test the hypothesis. We score every pair
through BOTH bakes (per-pair TSVs from `ensemble_score_rows`, bit-exact
with the runtime), then combine per-pair offline and compute the full
Mohammadi panel per corpus vs V39.

The HF regime is where V39 predicts HIGH quality (near-lossless): that's
exactly where KonJND lives (JND thresholds are near-visually-lossless)
and where V39's rank collapses. The gate g(x) is a soft sigmoid on V39's
own prediction level:

    g = sigmoid((v39_pred - center) / width)        # in [0,1], HF -> 1
    combined = g * specialist + (1 - g) * v39

g(x) depends ONLY on V39's per-pair prediction (a scalar), so there is no
feature classifier to train and no risk of corpus-label leakage. We sweep
(center, width) and also try a hard threshold + a max() blend as controls.

Usage:
    python3 scripts/v_next/g5_regime_gate_ensemble_2026-05-26.py \
        --specialist w01 [--center 70 --width 8]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Dedup-K (2026-05-26): the local srocc/krocc/plcc/z_rmse/pwrc/panel
# implementations (each a small scipy/np-only re-roll) are replaced
# with the canonical `scripts.lib.zen_stats.panel` shim, which calls
# the Rust `panel` binary — same code path as `bake_verdict` /
# `bake_compare`. This eliminates 5+1 duplicate stat impls and
# guarantees this script's ship/no-ship verdicts agree bit-exactly
# (≤1e-9) with the canonical IQA-stats home.
import sys as _sys
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))
from scripts.lib import zen_stats as _zs  # noqa: E402

SCORES = Path("/mnt/v/output/zensim/g5_ensemble_2026-05-26/scores")
CORPORA = ["cid22", "kadid", "tid", "konjnd", "aic3", "aic4"]
OTHER5 = ["cid22", "kadid", "tid", "aic3", "aic4"]


def load(corpus, which):
    return pd.read_csv(SCORES / f"{corpus}_{which}.tsv", sep="\t")


def _drop_nonfinite(h, p):
    h = np.asarray(h, float)
    p = np.asarray(p, float)
    valid = np.isfinite(h) & np.isfinite(p)
    return h[valid], p[valid]


def panel(pred, human):
    """Full Mohammadi panel via the canonical Rust shim.

    Returns dict with keys {n, srocc, plcc, krocc, pwrc, z}.
    """
    h, p = _drop_nonfinite(human, pred)
    if len(p) < 3:
        return dict(n=0, srocc=float("nan"), plcc=float("nan"),
                    krocc=float("nan"), pwrc=float("nan"), z=float("nan"))
    s = _zs.panel(p.tolist(), h.tolist())
    # zen_stats.panel returns 'z_rmse'; this script's downstream code
    # uses key 'z'. Translate at the boundary to keep callers stable.
    return dict(n=int(len(p)), srocc=s["srocc"], plcc=s["plcc"],
                krocc=s["krocc"], pwrc=s["pwrc"], z=s["z_rmse"])


# Single-stat convenience shims, used by 1-2 call-sites elsewhere.
def srocc(h, p):
    return panel(p, h)["srocc"]


def krocc(h, p):
    return panel(p, h)["krocc"]


def plcc(pred, human):
    return panel(pred, human)["plcc"]


def z_rmse(pred, human):
    return panel(pred, human)["z"]


def pwrc(human, pred):
    return panel(pred, human)["pwrc"]


def combine(v39, spec, mode, center, width):
    """Return combined per-pair scores."""
    v = np.asarray(v39, float)
    s = np.asarray(spec, float)
    if mode == "sigmoid":
        g = 1.0 / (1.0 + np.exp(-(v - center) / width))
        return g * s + (1.0 - g) * v
    if mode == "hard":
        return np.where(v >= center, s, v)
    if mode == "max":
        return np.maximum(v, s)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specialist", default="w01", help="bake tag (w002/w005/w01)")
    ap.add_argument("--mode", default="sigmoid", choices=["sigmoid", "hard", "max"])
    ap.add_argument("--center", type=float, default=70.0)
    ap.add_argument("--width", type=float, default=8.0)
    ap.add_argument("--sweep", action="store_true", help="sweep center/width grid")
    args = ap.parse_args()

    data = {}
    for c in CORPORA:
        v39 = load(c, "v39")
        spec = load(c, args.specialist)
        assert len(v39) == len(spec), f"{c}: row mismatch"
        data[c] = dict(human=v39["human"].values, v39=v39["score"].values, spec=spec["score"].values)

    # Baselines
    base_v39 = {c: panel(data[c]["v39"], data[c]["human"]) for c in CORPORA}
    base_spec = {c: panel(data[c]["spec"], data[c]["human"]) for c in CORPORA}

    print(f"=== specialist={args.specialist} ===")
    print(f"{'corpus':8s} {'V39':>8s} {'spec':>8s}")
    for c in CORPORA:
        print(f"{c:8s} {base_v39[c]['srocc']:8.4f} {base_spec[c]['srocc']:8.4f}")

    def eval_combo(mode, center, width):
        rows = {}
        for c in CORPORA:
            comb = combine(data[c]["v39"], data[c]["spec"], mode, center, width)
            rows[c] = panel(comb, data[c]["human"])
        return rows

    if args.sweep:
        print("\n=== SWEEP (sigmoid gate on V39 prediction level) ===")
        print(f"{'center':>7s} {'width':>6s} | " + " ".join(f"{c[:5]:>6s}" for c in CORPORA) +
              " | konjnd>=.70? others within -.01?")
        best = None
        for center in [40, 50, 55, 60, 65, 70, 75, 80, 85]:
            for width in [3, 5, 8, 12, 20]:
                rows = eval_combo("sigmoid", center, width)
                kon = rows["konjnd"]["srocc"]
                others_ok = all(rows[c]["srocc"] >= base_v39[c]["srocc"] - 0.01 for c in OTHER5)
                min_other_delta = min(rows[c]["srocc"] - base_v39[c]["srocc"] for c in OTHER5)
                flag = "PASS" if (kon >= 0.70 and others_ok) else ""
                line = (f"{center:7.0f} {width:6.0f} | " +
                        " ".join(f"{rows[c]['srocc']:6.3f}" for c in CORPORA) +
                        f" | kon={kon:.3f} minΔoth={min_other_delta:+.3f} {flag}")
                print(line)
                score = (kon if others_ok else kon - 5.0)  # rank passers first
                if best is None or score > best[0]:
                    best = (score, center, width, kon, others_ok, min_other_delta)
        print(f"\nBEST: center={best[1]} width={best[2]} konjnd={best[3]:.3f} "
              f"others_ok={best[4]} minΔoth={best[5]:+.3f}")
        # Also max() control
        rows_max = eval_combo("max", 0, 1)
        print("\n=== max(V39, spec) control ===")
        print(" ".join(f"{c}={rows_max[c]['srocc']:.3f}" for c in CORPORA))
        return

    # Single config: full panel
    rows = eval_combo(args.mode, args.center, args.width)
    print(f"\n=== ensemble ({args.mode} center={args.center} width={args.width}) "
          f"full panel vs V39 ===")
    print(f"{'corpus':8s} {'n':>5s} | {'SROCC':>7s} {'(V39)':>7s} {'Δ':>7s} | "
          f"{'PLCC':>6s} {'PWRC':>6s} {'Z-RMSE':>6s}")
    for c in CORPORA:
        e = rows[c]
        v = base_v39[c]
        d = e["srocc"] - v["srocc"]
        print(f"{c:8s} {e['n']:5d} | {e['srocc']:7.4f} {v['srocc']:7.4f} {d:+7.4f} | "
              f"{e['plcc']:6.3f} {e['pwrc']:6.3f} {e['z']:6.3f}")
    kon = rows["konjnd"]["srocc"]
    others_ok = all(rows[c]["srocc"] >= base_v39[c]["srocc"] - 0.01 for c in OTHER5)
    print(f"\nKonJND={kon:.4f} (>=0.70? {kon >= 0.70}) | others within -0.01 of V39? {others_ok}")
    print(f"VERDICT: {'PASS' if (kon >= 0.70 and others_ok) else 'FAIL'}")


if __name__ == "__main__":
    main()
