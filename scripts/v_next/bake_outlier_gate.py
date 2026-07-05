#!/usr/bin/env python3
# DEPRECATED (2026-07-05): migrated to Rust. Use
#   target/release/bake_dial_refit gate --bake <bin> --corpus <parquet> [--ref-col <col>]
# (zensim-validate/src/bin/bake_dial_refit.rs). The Rust gate reuses
# zenstats::panel for Z-RMSE/OR/SROCC and computes NO PWRC (OOM-safe), matching
# this script's light_panel. Kept for provenance. See
# benchmarks/bake_refit_rust_migration_2026-07-05.md.
"""Outlier / calibration eval gate for a linear ZNPR bake — catches the
failure modes a rank-only (SROCC) panel is BLIND to.

Motivation (2026-07-05): the shipped Profile-B SDR bake scored a healthy
CID22 SROCC 0.873 while emitting raw predictions to -1131 on ~0.4% of a
broad corpus (a heavy-tailed feature, f155, on tiny dark-screen renditions).
SROCC is rank-based and near-invariant to a monotone tail explosion, so it
hid the pathology completely. The dial then extrapolated those raw values
below the bottom spline knot to absurd negatives (the webp -80 finding).

This gate runs a bake over a BROAD corpus and reports the stats that DO
move under a tail:

  G-RANGE   : raw-pred distribution vs the spline knot domain — the fraction
              of rows whose raw pred falls OUTSIDE [knot_lo, knot_hi] and so
              extrapolates. This is the direct tail detector. Hard gate.
  G-ZRMSE   : Z-RMSE vs human (Mohammadi 2025) — calibration error after a
              4PL rescale. Unlike SROCC it PENALIZES a few wild misses.
  G-XMETRIC : cross-metric disagreement — per-row |z(dial) - z(human)|;
              flags the pairs where the bake most disagrees with the
              reference metric, and lists the worst offenders by name.
              (Sanity-checking against another metric is what a human would
              do; this automates it.)
  G-OUTRATIO: outlier ratio (fraction of rescaled preds outside +/-2 sigma).

The forward is reconstructed faithfully by importing the SAME transform code
(`apply_transform`) the bakes are built with, then validated against the
Rust runtime's CID22 (--validate) so the numbers are trustworthy.

  usage: bake_outlier_gate.py --bake B.bin [--corpus P.parquet] [--validate]
"""
import argparse
import importlib.util
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath("."))
from scipy.optimize import curve_fit  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


def _logistic4(x, a, b, c, d):
    return a + (b - a) / (1.0 + np.exp(-(x - c) / (d if abs(d) > 1e-9 else 1e-9)))


def light_panel(pred, human):
    """SROCC + Z-RMSE + outlier-ratio WITHOUT the O(n^2) PWRC in zen_stats.panel
    (which OOMs/hangs past ~10k rows — the sa_st_curve all-pairs bug). Z-RMSE is
    the sigma-normalized RMSE after a 4PL rescale (corpus-wide sigma when no
    per-sample sigma is available); OR is the fraction outside +/-2 sigma."""
    pred = np.asarray(pred, float); human = np.asarray(human, float)
    srocc = abs(spearmanr(pred, human).statistic)
    # 4PL rescale pred -> human units, sign-robust via median init
    pn = (pred - pred.min()) / max(1e-9, pred.max() - pred.min())
    try:
        p0 = [float(human.min()), float(human.max()), float(np.median(pn)), 0.2]
        popt, _ = curve_fit(_logistic4, pn, human, p0=p0, maxfev=6000)
        fit = _logistic4(pn, *popt)
    except Exception:
        fit = pred
    resid = fit - human
    sigma = human.std() + 1e-9
    z_rmse = float(np.sqrt(np.mean((resid / sigma) ** 2)))
    outr = float(np.mean(np.abs(resid) > 2 * sigma))
    return dict(srocc=srocc, z_rmse=z_rmse, **{"or": outr})

SPEC = importlib.util.spec_from_file_location(
    "lp", Path(__file__).parent / "linear_projections_2026-07-03.py")
lp = importlib.util.module_from_spec(SPEC)
sys.modules["lp"] = lp
SPEC.loader.exec_module(lp)

BAKER = os.path.expanduser("~/work/zen/zenanalyze/target/release/zenpredict")
PROBE = Path("/mnt/v/output/zensim-multicodec-probe")
FF = Path("/mnt/v/zen/zensim-training/2026-05-15-full-features")


def load_bake(binpath):
    """Extract scaler / weights / transforms / spline from a ZNPR bake."""
    ins = json.loads(subprocess.run(
        [BAKER, "inspect", binpath, "--weights"], capture_output=True, text=True).stdout)
    mu = np.array(ins["scaler_mean"], dtype=float)
    sd = np.array(ins["scaler_scale"], dtype=float)
    layer = ins["layers"][0]
    w = np.array(layer["weights"], dtype=float)
    b = float(layer["biases"][0])
    md = {m["key"]: m for m in ins["metadata"]}
    transforms = tparams = None
    if "zentrain.feature_transforms" in md:
        toks = md["zentrain.feature_transforms"]["value_text"].split("\n")
        praw = md["zentrain.feature_transform_params"]["value_text"].split("\n")
        transforms = toks
        tparams = [[float(x) for x in r.split(",")] if r else [] for r in praw]
    sp = md["zentrain.output_calibration_spline"]
    raw = bytes.fromhex(sp.get("value_hex") or sp.get("hex"))
    n = struct.unpack("<I", raw[:4])[0]
    knots = np.array([struct.unpack("<ff", raw[4 + 8 * i:12 + 8 * i]) for i in range(n)])
    return dict(mu=mu, sd=sd, w=w, b=b, transforms=transforms, tparams=tparams,
                knot_x=knots[:, 0], knot_y=knots[:, 1])


def raw_forward(bk, F):
    X = F
    if bk["transforms"] is not None:
        X = np.column_stack([
            lp.apply_transform(bk["transforms"][i], bk["tparams"][i], F[:, i])
            for i in range(F.shape[1])])
    return ((X - bk["mu"]) / np.where(bk["sd"] == 0, 1, bk["sd"])) @ bk["w"] + bk["b"]


def dial(bk, raw):
    # product-runtime spline: monotone PCHIP interp, upper cap 100, lower uncapped
    y = np.interp(raw, bk["knot_x"], bk["knot_y"])
    below = raw < bk["knot_x"][0]
    if below.any():  # linear downward extrapolation off the bottom knot
        s = (bk["knot_y"][1] - bk["knot_y"][0]) / (bk["knot_x"][1] - bk["knot_x"][0])
        y = np.where(below, bk["knot_y"][0] + s * (raw - bk["knot_x"][0]), y)
    return np.minimum(y, 100.0)


def loadF(fp, pref="f", tcol="human_score"):
    import pyarrow.parquet as pq
    t = pq.read_table(fp)
    cols = sorted([c for c in t.schema.names
                   if c.startswith(pref) and c[len(pref):].split("_")[-1].isdigit()],
                  key=lambda c: int("".join(ch for ch in c if ch.isdigit())))[:372]
    F = np.column_stack([np.asarray(t[c], dtype=float) for c in cols])
    names = t["ref_basename"].to_pylist() if "ref_basename" in t.schema.names else [str(i) for i in range(len(F))]
    return F, np.asarray(t[tcol], dtype=float), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", required=True)
    ap.add_argument("--corpus", default=str(PROBE / "bigcodec_valdigits_2026-07-02.parquet"),
                    help="broad corpus with f0..f371 + a reference column (default: bigcodec_val 147k)")
    ap.add_argument("--ref-col", default="human_score",
                    help="reference-metric column for the cross-metric check. Prefer an "
                         "INDEPENDENT metric (e.g. butteraugli_max) over the bake's own "
                         "training target — a self-derived reference shares the bake's biases.")
    ap.add_argument("--validate", action="store_true",
                    help="cross-check the numpy forward against the Rust runtime CID22")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    bk = load_bake(a.bake)
    klo, khi = float(bk["knot_x"][0]), float(bk["knot_x"][-1])

    if a.validate:
        Fc, hc, _ = loadF(FF / "cid22_features_372col_2026-05-15.parquet")
        np_srocc = light_panel(dial(bk, raw_forward(bk, Fc)), hc)["srocc"]
        print(f"[validate] numpy-forward CID22 SROCC = {np_srocc:.4f} "
              f"(compare to bake_verdict runtime; must match to ~1e-3)")

    F, hum, names = loadF(a.corpus, tcol=a.ref_col)
    raw = raw_forward(bk, F)
    dl = dial(bk, raw)
    n = len(raw)

    # G-RANGE — the HARD gate. This is the one that would have blocked the raw-B
    # ship: raw preds outside the spline knot domain extrapolate (uncapped
    # downward), turning a heavy-tailed feature into an absurd dial value.
    below = int((raw < klo).sum()); above = int((raw > khi).sum())
    range_fail = below + above > n * 1e-4
    print(f"\n=== outlier gate: {os.path.basename(a.bake)} on {os.path.basename(a.corpus)} (n={n:,}) ===")
    print(f"knot domain [{klo:.3f}, {khi:.3f}]  |  raw pred [{raw.min():.2f}, {raw.max():.2f}]")
    print(f"[HARD] G-RANGE   below-knot {below} ({100*below/n:.3f}%)  above-knot {above} ({100*above/n:.3f}%)"
          f"  -> {'FAIL' if range_fail else 'PASS'} (gate: <0.01% extrapolating)")

    # G-ZRMSE + G-OUTRATIO (advisory — vs the reference metric)
    p = light_panel(dl, hum)
    print(f"[adv]  G-ZRMSE   {p['z_rmse']:.3f} vs {a.ref_col} (lower=better; a tail inflates this while SROCC stays flat)")
    print(f"[adv]  G-SROCC   {p['srocc']:.4f}  (rank — near-INVARIANT to the tail; that's why it hid the bug)")
    print(f"[adv]  G-OUTRATIO {p['or']:.4f}  (fraction outside +/-2 sigma of rescaled reference)")

    # G-XMETRIC: per-row disagreement vs the reference metric. CAVEAT: when the
    # reference is the bake's own training target (here human_score = ssim2/100)
    # it (a) shares the bake's biases and (b) has FLOOR spikes (ssim2 clamped to
    # 0 on tiny/heavily-distorted renditions) that produce false alarms. We
    # exclude reference-floored rows (human in {0,1}) so the flag reflects
    # disagreement vs a VALID reference. For a true independent sanity check,
    # run against butteraugli/an unrelated metric, not the training target.
    floored = (hum <= 0.0) | (hum >= 1.0)
    n_floor = int(floored.sum())
    valid = ~floored

    def z(v):
        return (v - v.mean()) / (v.std() + 1e-9)
    zd, zh = z(dl[valid]), z(hum[valid])
    disagree = np.abs(zd - zh)
    nv = int(valid.sum())
    flagged = int((disagree > 3).sum())
    print(f"[adv]  G-XMETRIC {n_floor} reference-floored rows excluded ({100*n_floor/n:.2f}% — clamp "
          f"artifact when ref is self-derived, NOT a bake fault). On {nv:,} valid rows:")
    print(f"          disagreement>3sigma: {flagged} ({100*flagged/max(nv,1):.3f}%)"
          f"  (advisory: surfaces content classes where the bake and reference metric diverge)")
    vnames = [nm for nm, keep in zip(names, valid) if keep]
    vraw, vdl, vhum = raw[valid], dl[valid], hum[valid]
    order = np.argsort(-disagree)[:a.top]
    print(f"  worst {a.top} offenders vs valid reference (name | raw | dial | ref | |z-diff|):")
    for i in order:
        print(f"    {vnames[i][:44]:<44} {vraw[i]:>9.2f} {vdl[i]:>7.1f} {vhum[i]:>6.3f} {disagree[i]:>6.2f}")

    verdict = "FAIL (blocks ship)" if range_fail else "PASS (ship-eligible)"
    print(f"\nVERDICT: {verdict}  [HARD gate = G-RANGE only; the rest are advisory review signals]")
    sys.exit(1 if range_fail else 0)


if __name__ == "__main__":
    main()
