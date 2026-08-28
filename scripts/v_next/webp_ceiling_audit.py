#!/usr/bin/env python3
"""Gate-provenance audit instrument (2026-08-28, user directive: "missing by
hairs is sus... unless webp qualities not addressable at top end we can
adjust gate, but give more info").

Question: is the G-GRAN webp reach bar (91.9 = incumbent reach − 1)
demanding a value ABOVE webp's true top-end quality? Reach is a CALIBRATION
property, not monotone-good: if peers say webp's top knob delivers less
than 91.9-equivalent quality, the bar rewards over-reporting inherited
from the incumbent (whose top zone holds its 228 identity violations).

Method: per bake, forward the stored 944 dial grid, fit the campaign's
optimal-class monotone translation model->ssim2 (loop_proxy.qmap, weighted
PAVA + tail extrapolation), then per codec at the TOP knob cell of each
image: implied honest reach = map^{-1}(median top-cell ssim2). Compare with
the bake's actual reach (median emission at top cells). Butteraugli as a
second witness. Peers are model-independent ground truth here."""
import numpy as np, json, sys
import loop_proxy as lp

BAKES = {
    "incumbent_s4003": "/mnt/v/output/zensim/bakes/sota944/bakes/W10L9_s4003_packed.bin",
    "A_PH_s4004":      "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W10L9PH_s4004_packed.bin",
    "w11_s4014_e050":  "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W11J_s4014_ckpts/ckpt_epoch050_s4014_packed.bin",
    "w11_s4012_e080":  "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W11J_s4012_ckpts/ckpt_epoch080_s4012_packed.bin",
    "w11_s4014_final": "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W11J_s4014_ckpts/W11J_s4014_s4014_packed.bin",
}
import os
for n, p in BAKES.items():
    assert os.path.exists(p), f"missing bake {n}: {p}"

X, imgs, codecs, prm = lp.load_grid()
ss = lp.load_peer("dialgrid_ssim2_gpu.tsv", "ssim2_gpu")
import csv
bcol = [c for c in csv.DictReader(open(f"{lp.REFM}/dialgrid_butteraugli_gpu.tsv"),
        delimiter="\t").fieldnames if "max" in c or "butter" in c][0]
bt = lp.load_peer("dialgrid_butteraugli_gpu.tsv", bcol, neg=True)

keys = [(im, "zen" + c, round(float(p), 4)) for im, c, p in zip(imgs, codecs, prm)]
have = np.array([k in ss for k in keys])
sv = np.array([ss.get(k, np.nan) for k in keys])
bv = np.array([-bt[k] if k in bt else np.nan for k in keys])  # raw butter, lower=better
print(f"grid {len(keys)} cells, ssim2 join {have.sum()}", file=sys.stderr)

# top-knob cell per (image, codec)
top = {}
for i, (im, c, p) in enumerate(zip(imgs, codecs, prm)):
    k = (im, c)
    if k not in top or p > prm[top[k]]:
        top[k] = i
top_idx = {c: np.array([i for (im, cc), i in top.items() if cc == c and have[i]])
           for c in ("jxl", "avif", "jpeg", "webp")}

print("\n=== PEER TRUTH AT THE TOP KNOB (model-independent) ===")
print(f"{'codec':6} {'n':>3} {'ssim2 p25/p50/p75':>22} {'butter p50':>11}")
for c, ti in top_idx.items():
    q = np.percentile(sv[ti], [25, 50, 75])
    print(f"{c:6} {len(ti):3d} {q[0]:6.2f}/{q[1]:6.2f}/{q[2]:6.2f}        {np.nanmedian(bv[ti]):8.3f}")

print("\n=== PER-BAKE REACH vs IMPLIED-HONEST REACH (map^-1 of top-cell ssim2) ===")
mfit = have  # fit translation on every joined cell
for name, path in BAKES.items():
    pr = lp.forward(path, X)
    m = lp.qmap(pr[mfit], sv[mfit], imgs[mfit])
    # numeric inverse over the emission axis
    gx = np.linspace(pr.min() - 5, pr.max() + 15, 4001)
    gy = np.asarray(m(gx))
    def inv(y):
        j = np.searchsorted(gy, y)
        return float(gx[min(j, len(gx) - 1)])
    row = [name]
    for c in ("jxl", "avif", "jpeg", "webp"):
        ti = top_idx[c]
        reach = float(np.median(pr[ti]))
        honest = inv(float(np.median(sv[ti])))
        row.append(f"{c}: reach {reach:5.2f} honest {honest:5.2f} stretch {reach-honest:+5.2f}")
    print("  ".join(row))
    if name in ("incumbent_s4003", "A_PH_s4004"):
        # what ssim2 level does the webp bar 91.9 demand under this bake's map?
        lvl = float(m(91.9))
        ti = top_idx["webp"]
        att = float((sv[ti] >= lvl).mean())
        print(f"    -> under {name}'s map, webp bar 91.9 demands ssim2 >= {lvl:.2f}; "
              f"fraction of images whose webp TOP cell attains it: {att:.2%}")
