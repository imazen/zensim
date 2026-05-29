#!/usr/bin/env python3
"""Sweep round 2 (#48): the higher-value levers the round-1 sweep didn't try.

Round 1 showed Cell 5's recipe is the BVLS-linear optimum (mean-g3 0.8145).
Round 2 tries levers that could break the linear ceiling:
  L1. low-q band weighting    — upweight low/mid-q rows (CLAUDE.md: compression
                                product decisions live in low-q; weight them more)
  L2. larger corpus           — add cvvdp_iwssim_LARGE (73k extra rows)
  L3. soft concordance weight — weight each row by ssim2↔cvvdp agreement strength
                                instead of a hard ±2-bucket keep/drop
  L4. mix_ss_iw (round-1 CID22 record) re-confirmed + its full panel
  L5. mix_ss_iw + low-q weighting (stack the two best signals)

Bakes + full-panel-scores each via bake_verdict. Reuses round-1 infra.
"""
from __future__ import annotations
import csv, json, struct, subprocess, sys, tempfile, os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import lsq_linear

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v02_372feat_sweep import (  # reuse round-1 helpers
    N_FEATURES, TRAIN_DIR, ANCHOR_PQ, OUT_DIR, BAKER, CORPORA, CELL5,
    load_mask, to_distance, concordance_mask, fit_bvls, fit_spline, emit_bake, verdict,
)


def main():
    pin = load_mask()
    print("loading safesyn (+ssim2/cvvdp/iwssim)...", file=sys.stderr)
    p = TRAIN_DIR / "safesyn.parquet"
    t = pq.read_table(p, columns=[f"f{i}" for i in range(N_FEATURES)] + ["ssim2_gpu", "cvvdp_score", "iwssim"])
    X = np.column_stack([np.asarray(t[f"f{i}"].combine_chunks().to_numpy(), dtype=np.float64) for i in range(N_FEATURES)])
    s2 = np.asarray(t["ssim2_gpu"].combine_chunks().to_numpy(), dtype=np.float64)
    cv = np.asarray(t["cvvdp_score"].combine_chunks().to_numpy(), dtype=np.float64)
    iw = np.asarray(t["iwssim"].combine_chunks().to_numpy(), dtype=np.float64)
    fin = np.all(np.isfinite(X), axis=1) & np.isfinite(s2) & np.isfinite(cv) & np.isfinite(iw)
    X, s2, cv, iw = X[fin], s2[fin], cv[fin], iw[fin]
    print(f"safesyn: {len(X)} rows", file=sys.stderr)

    # Anchor
    at = pq.read_table(ANCHOR_PQ, columns=[f"f{i}" for i in range(N_FEATURES)] + ["human_score"])
    Xa = np.column_stack([np.asarray(at[f"f{i}"].combine_chunks().to_numpy(), dtype=np.float64) for i in range(N_FEATURES)])
    ya = np.asarray(at["human_score"].combine_chunks().to_numpy(), dtype=np.float64)
    ka = np.isfinite(ya) & np.all(np.isfinite(Xa), axis=1)
    Xa, ya = Xa[ka], ya[ka]

    d_s2 = to_distance(s2, "ssim2_gpu")
    cm = concordance_mask(s2, cv)  # round-1 default concordance

    # low-q weight: ssim2_gpu low (=more distortion) → higher weight.
    # ssim2 score 0..100; low-q rows (ssim2 < 50) get weight 2.0, high-q 1.0, linear ramp.
    lowq_w = np.clip(2.0 - (s2 / 100.0) * 1.5, 0.5, 2.0)

    # soft concordance weight: 1 / (1 + bucket-distance). closer agreement → higher weight.
    nb = 20
    s2r = np.argsort(np.argsort(s2)) / len(s2)
    cvr = np.argsort(np.argsort(cv)) / len(s2)
    bd = np.abs(np.clip((s2r * nb).astype(int), 0, nb - 1) - np.clip((cvr * nb).astype(int), 0, nb - 1))
    soft_w = 1.0 / (1.0 + bd)

    cells = [
        # label,            target_dist,                          row_mask, row_weight
        ("lowq_weight",     d_s2,                                  cm,   lowq_w[cm]),
        ("soft_concord",    d_s2,                                  None, soft_w),
        ("mix_ss_iw_conf",  0.5 * d_s2 + 0.5 * to_distance(iw, "iwssim"), concordance_mask(s2, iw), None),
        ("mix_ss_iw_lowq",  0.5 * d_s2 + 0.5 * to_distance(iw, "iwssim"), concordance_mask(s2, iw), None),  # +lowq below
    ]

    print(f"\n{'cell':18s} {'active':>7s} {'CID22':>7s} {'KADID':>7s} {'TID':>7s} {'KonJND':>7s} {'AIC3':>7s} {'AIC4':>7s} {'meanG3':>7s} {'vsCell5':>8s}", file=sys.stderr)
    print("-" * 100, file=sys.stderr)
    rows = []
    for label, yd, mask, rw in cells:
        if mask is None:
            Xtr, ytr = X, yd
            w = rw
        else:
            Xtr, ytr = X[mask], yd[mask]
            w = rw  # already masked where provided
        if label == "mix_ss_iw_lowq":
            m2 = concordance_mask(s2, iw)
            Xtr, ytr = X[m2], (0.5 * d_s2 + 0.5 * to_distance(iw, "iwssim"))[m2]
            w = lowq_w[m2]
        coef, bias, mu, sd, nact = fit_bvls(Xtr, ytr, pin, sw=w)
        Xa_z = (Xa - mu) / sd
        raw_a = Xa_z @ coef + bias
        spline = fit_spline(raw_a, ya)
        bake_path = OUT_DIR / f"{label}.bin"
        if not emit_bake(coef, bias, mu, sd, spline, bake_path):
            print(f"{label:18s}  BAKE FAILED", file=sys.stderr); continue
        v = verdict(bake_path)
        if not v:
            print(f"{label:18s}  VERDICT FAILED", file=sys.stderr); continue
        sr = {c: v.get(c, {}).get("srocc", float("nan")) for c in CORPORA}
        g3 = [v.get(c, {}).get("g3", float("nan")) for c in CORPORA]
        mg = float(np.nanmean(g3))
        dl = mg - CELL5["mean_g3"]
        flag = "  WIN" if (dl > 0 and sr["cid22"] >= 0.86) else ""
        print(f"{label:18s} {nact:7d} {sr['cid22']:7.4f} {sr['kadid']:7.4f} {sr['tid']:7.4f} "
              f"{sr['konjnd']:7.4f} {sr['aic3']:7.4f} {sr['aic4']:7.4f} {mg:7.4f} {dl:+8.4f}{flag}", file=sys.stderr)
        rows.append({"cell": label, "n_active": nact, "mean_g3": mg, "delta": dl,
                     **{f"srocc_{c}": sr[c] for c in CORPORA}})

    out = Path("/tmp/v02_372feat_sweep2_summary.tsv")
    with open(out, "w") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            w.writeheader(); w.writerows(rows)
    print(f"\nsummary → {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
