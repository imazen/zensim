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

Generalised 2026-09-01: `--arm-a` / `--arm-b` name ANY two arms and `--results`
may be given more than once, so the same reading can also be compared across
INDEPENDENT corpora -- e.g. `btc_native` vs `iptc_native`, two native readings
of two different studies. The `*_native` / `*_displayed` column names denote
reading A and reading B; the `arm_a` / `arm_b` columns say which arms those are.
The defaults reproduce the original three-arm invocation.
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
    ap.add_argument("--results", action="append", default=None,
                    help="pairwise_results TSV; repeatable; default <dir>/pairwise_results.tsv")
    ap.add_argument("--arm-a", default="btc_native")
    ap.add_argument("--arm-b", default="btc_displayed")
    ap.add_argument("--subsets", default=None, help="comma list; default the six BTC subsets")
    ap.add_argument("--out", default="arm_invariance.tsv")
    a = ap.parse_args()
    d = Path(a.dir)
    srcs = [Path(x) if Path(x).is_absolute() else d / x
            for x in (a.results or ["pairwise_results.tsv"])]
    rows = [r for s_ in srcs for r in csv.DictReader(open(s_), delimiter="\t")
            if "#raw" not in r["scorer"]]
    subsets = [x for x in a.subsets.split(",")] if a.subsets else SUBSETS
    idx = {(r["arm"], r["subset"], r["scorer"]): r for r in rows}
    out, flips = [], 0
    for sub in subsets:
        for sc in ORDER:
            n = idx.get((a.arm_a, sub, sc))
            p = idx.get((a.arm_b, sub, sc))
            if not (n and p):
                continue
            vn, vp = verdict(n), verdict(p)
            flips += vn != vp
            out.append({
                "arm_a": a.arm_a, "arm_b": a.arm_b, "subset": sub, "scorer": sc,
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
    outp = d / a.out
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), delimiter="\t")
        w.writeheader()
        for r in out:
            w.writerow(r)
    for r in out:
        print(f"{r['subset']:18s} {r['scorer']:30s} {r['d_native']:+.4f} {r['verdict_native']:5s} | "
              f"{r['d_displayed']:+.4f} {r['verdict_displayed']:5s} {'' if r['arm_invariant'] else '** FLIPS'}")
    print(f"\n{flips} of {len(out)} (subset, scorer) verdicts FLIP between "
          f"{a.arm_a} and {a.arm_b} -> {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
