#!/usr/bin/env python3
"""Bjøntegaard BD-rate for the #38 diffmap RD comparison.

Reads the `loops_{butteraugli,dssim}.tsv` scored by `zen-metrics batch` over
the `zensim_diffmap_rd` harness manifests (columns: label, image, distance,
bytes, + metric columns). Computes BD-rate (% bytes at equal quality,
integrated over the overlapping quality range) of each challenger loop vs
`zensim_default`, per image + mean, on each independent quality axis.

Negative BD-rate = challenger uses FEWER bytes at equal quality = better.
Quality axes: dssim (independent of all 3 loops) + butteraugli_pnorm3
(independent of zensim+cvvdp; circular for the butteraugli loop — discount).

Usage: python3 scripts/v_next/bd_rate_diffmap.py [results_dir]
  default results_dir = /mnt/v/output/zensim/diffmap-rd-2026-05-27
"""
import csv, collections, sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/v/output/zensim/diffmap-rd-2026-05-27"
BASE = "zensim_default"
CHALLENGERS = ["cvvdp", "butteraugli"]


def load(m):
    with open(f"{OUT}/loops_{m}.tsv") as f:
        return {" ".join([r["label"], r["image"], r["distance"]]): r
                for r in csv.DictReader(f, delimiter="\t")}


def bd_rate(rate_a, q_a, rate_b, q_b):
    """% bytes(B) vs bytes(A) at equal quality. x=quality (lower=better),
    y=log10(rate); poly-fit (deg<=2), integrate diff over the quality overlap."""
    A = sorted(zip(q_a, np.log10(rate_a)))
    B = sorted(zip(q_b, np.log10(rate_b)))
    qa, ya = [p[0] for p in A], [p[1] for p in A]
    qb, yb = [p[0] for p in B], [p[1] for p in B]
    lo, hi = max(min(qa), min(qb)), min(max(qa), max(qb))
    if hi <= lo:
        return None
    pa = np.polyfit(qa, ya, min(2, len(qa) - 1))
    pb = np.polyfit(qb, yb, min(2, len(qb) - 1))
    Pa, Pb = np.polyint(pa), np.polyint(pb)
    int_a = np.polyval(Pa, hi) - np.polyval(Pa, lo)
    int_b = np.polyval(Pb, hi) - np.polyval(Pb, lo)
    return (10 ** ((int_b - int_a) / (hi - lo)) - 1) * 100.0


def main():
    ba, ds = load("butteraugli"), load("dssim")
    D = collections.defaultdict(list)
    for k, r in ba.items():
        dcol = [c for c in ds[k] if "dssim" in c.lower()][0]
        D[(r["label"], r["image"])].append(
            (int(r["bytes"]), float(ds[k][dcol]), float(r["butteraugli_pnorm3"])))
    for k in D:
        D[k].sort()
    images = sorted({im for (_, im) in D})

    print("BD-rate vs zensim_default (NEG = challenger fewer bytes at equal quality = better):")
    for qaxis, idx in [("dssim", 1), ("butteraugli", 2)]:
        print(f"\n  axis={qaxis}:   {'image':12s}" + "".join(f"{c:>12s}" for c in CHALLENGERS))
        agg = collections.defaultdict(list)
        for im in images:
            base = D[(BASE, im)]
            rb, qb = [p[0] for p in base], [p[idx] for p in base]
            row = f"             {im:12s}"
            for chal in CHALLENGERS:
                c = D[(chal, im)]
                bd = bd_rate(rb, qb, [p[0] for p in c], [p[idx] for p in c])
                if bd is not None:
                    agg[chal].append(bd)
                    row += f"{bd:+11.1f}%"
                else:
                    row += f"{'n/a':>12s}"
            print(row)
        print(f"             {'MEAN':12s}" + "".join(f"{np.mean(agg[c]):+11.1f}%" for c in CHALLENGERS))


if __name__ == "__main__":
    main()
