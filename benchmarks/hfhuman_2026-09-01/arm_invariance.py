#!/usr/bin/env python3
"""Arm-invariance table: does a verdict survive the boosted-vs-native choice?

`btc_displayed` and `btc_native` serve the IDENTICAL 504,960 responses over the
IDENTICAL 3,960 triplets; the only thing that differs is which pixels the
metric is shown — the boosted rendering the worker actually saw (2x magnified,
~1.8x distortion-amplified) or the same region at native scale and amplitude,
cropped from the CTC encode it was rendered from.

A verdict that flips between them is a property of the reading, not of the
model. This emits, per (subset, scorer), the two paired-bootstrap verdicts and
whether they agree. Nothing is computed here beyond reading the sign of an
interval `panel --pairwise` already produced.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

ORDER = ["peer_ssim2", "Q7b_pools_g0.2_a0.2_b0.97", "W10L9PH_s4004_packed",
         "W10L9P_s4005_packed", "ADD156", "B"]
SUBSETS = ["all", "cross_codec", "same_codec", "vs_original",
           "study:aic3_btc", "study:sdr25_btc"]


def verdict(r) -> str:
    lo, hi = float(r["d_ssim2_question_lo"]), float(r["d_ssim2_question_hi"])
    return "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    rows = [r for r in csv.DictReader(open(d / "pairwise_results.tsv"), delimiter="\t")
            if "#raw" not in r["scorer"]]
    idx = {(r["arm"], r["subset"], r["scorer"]): r for r in rows}
    out, flips = [], 0
    for sub in SUBSETS:
        for sc in ORDER:
            n = idx.get(("btc_native", sub, sc))
            p = idx.get(("btc_displayed", sub, sc))
            if not (n and p):
                continue
            vn, vp = verdict(n), verdict(p)
            flips += vn != vp
            out.append({
                "subset": sub, "scorer": sc,
                "n_groups": int(n["n_groups"]), "n_responses": float(n["n_responses"]),
                "d_native": float(n["d_ssim2_question"]),
                "native_lo": float(n["d_ssim2_question_lo"]),
                "native_hi": float(n["d_ssim2_question_hi"]),
                "verdict_native": vn,
                "d_displayed": float(p["d_ssim2_question"]),
                "displayed_lo": float(p["d_ssim2_question_lo"]),
                "displayed_hi": float(p["d_ssim2_question_hi"]),
                "verdict_displayed": vp,
                "arm_invariant": vn == vp,
            })
    outp = d / "arm_invariance.tsv"
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), delimiter="\t")
        w.writeheader()
        for r in out:
            w.writerow(r)
    for r in out:
        print(f"{r['subset']:18s} {r['scorer']:30s} {r['d_native']:+.4f} {r['verdict_native']:5s} | "
              f"{r['d_displayed']:+.4f} {r['verdict_displayed']:5s} {'' if r['arm_invariant'] else '** FLIPS'}")
    print(f"\n{flips} of {len(out)} (subset, scorer) verdicts FLIP with the reading -> {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
