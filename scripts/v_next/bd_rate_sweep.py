#!/usr/bin/env python3
"""Per-content-class BD-rate for the #38 diffmap SWEEP (2026-05-27).

Compares the butteraugli loop vs the zensim-v47 loop (the cvvdp loop is
unbuildable after the zenmetrics cvvdp-cpu→cvvdp rename), judged by THREE
independent authorities: ssim2, butteraugli, cvvdp. ssim2 replaces dssim per
user request — note it is SEMI-circular for the zensim loop (v47 trained on
ssim2-derived targets), so read the ssim2 axis on the zensim loop with that
caveat; butteraugli + cvvdp are fully independent of zensim.

Reads sweep_{ssim2,butteraugli,cvvdp}.tsv (zen-metrics batch over the sweep
manifests; cols include label, image, corpus[=class], distance, bytes).
Negative BD-rate = challenger (butteraugli loop) uses fewer bytes at equal
quality = better than zensim-v47.

Usage: python3 scripts/v_next/bd_rate_sweep.py [sweep_dir]
"""
import csv, collections, sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/v/output/zensim/diffmap-sweep-2026-05-27"
BASE = "zensim_v47"
CHALLENGERS = ["cvvdp", "butteraugli"]


def load(m):
    with open(f"{OUT}/sweep_{m}.tsv") as f:
        return {" ".join([r["label"], r["image"], r["distance"]]): r
                for r in csv.DictReader(f, delimiter="\t")}


def col(row, key):
    return float(row[[c for c in row if key in c.lower()][0]])


def bd_rate(rate_a, q_a, rate_b, q_b):
    A = sorted(zip(q_a, np.log10(rate_a))); B = sorted(zip(q_b, np.log10(rate_b)))
    qa, ya = [p[0] for p in A], [p[1] for p in A]
    qb, yb = [p[0] for p in B], [p[1] for p in B]
    lo, hi = max(min(qa), min(qb)), min(max(qa), max(qb))
    if hi <= lo:
        return None
    deg = min(3, len(qa) - 1)
    Pa = np.polyint(np.polyfit(qa, ya, deg)); Pb = np.polyint(np.polyfit(qb, yb, deg))
    ia = np.polyval(Pa, hi) - np.polyval(Pa, lo); ib = np.polyval(Pb, hi) - np.polyval(Pb, lo)
    return (10 ** ((ib - ia) / (hi - lo)) - 1) * 100.0


def main():
    ss, ba, cv = load("ssim2"), load("butteraugli"), load("cvvdp")
    # per (loop, image): class + sorted (bytes, ssim2[lower=better via 100-x], butter_pnorm3, cvvdp[10-jod])
    D = collections.defaultdict(list); cls = {}
    for k, r in ba.items():
        # ssim2: higher=better (0..100) → 100-x so lower=better, like the others
        q_ss = 100.0 - col(ss[k], "ssim2")
        q_cv = 10.0 - col(cv[k], "cvvdp")
        D[(r["label"], r["image"])].append((int(r["bytes"]), q_ss, col(r, "butteraugli_pnorm3"), q_cv))
        cls[r["image"]] = r["corpus"]
    for kk in D:
        D[kk].sort()
    images = sorted({im for (_, im) in D})
    classes = sorted(set(cls.values()))

    print(f"BD-rate vs {BASE} (NEG = challenger fewer bytes at equal quality = better).")
    print("[C] = circular (butteraugli axis↔butteraugli loop, cvvdp axis↔cvvdp loop) — discount.")
    print("ssim2 is SEMI-circular for the zensim base (v47 trained on ssim2-derived targets).\n")
    for qaxis, idx in [("ssim2", 1), ("butteraugli", 2), ("cvvdp", 3)]:
        print(f"  === axis={qaxis} ===")
        for chal_name in CHALLENGERS:
            circ = (qaxis == "butteraugli" and chal_name == "butteraugli") or (qaxis == "cvvdp" and chal_name == "cvvdp")
            per_class = collections.defaultdict(list)
            for im in images:
                base = D.get((BASE, im)); chal = D.get((chal_name, im))
                if not base or not chal:
                    continue
                v = bd_rate([p[0] for p in base], [p[idx] for p in base],
                            [p[0] for p in chal], [p[idx] for p in chal])
                if v is not None:
                    per_class[cls[im]].append(v)
            line = f"    {chal_name + (' [C]' if circ else ''):18s}"
            for c in classes:
                if per_class[c]:
                    line += f"  {c}: {np.mean(per_class[c]):+6.1f}% (n{len(per_class[c])})"
            allx = [v for c in classes for v in per_class[c]]
            if allx:
                line += f"  ALL: {np.mean(allx):+6.1f}%"
            print(line)
        print()


if __name__ == "__main__":
    main()
