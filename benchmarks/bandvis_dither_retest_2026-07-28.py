#!/usr/bin/env python3
"""BANDVIS dst-side dither retest on grain-pathology-2026-07-28.

Decides whether the +5%-CPU dst-activity-plane fix (append2 gates doc
REMAINDERS #3, the pinned V3(b) limitation) is warranted, using the real
posterize+dither corpus instead of the 256-square synthetic ramp:

  (1) masking-suppression ratios GAIN_dithered/GAIN_undithered at matched
      (source, size, bits), per dither type and scale;
  (2) absolute cross-fire magnitude of GAIN on dithered pairs vs the
      real-banding GAIN levels from the LIVE-YT-Banding run;
  (3) GAIN/LOSS polarity across the other three families
      (denoise_regrain, jxl_noise, av1_grain pn-on/off);
  (4) trainer-side separability: can existing append-block Y lanes
      (CONTRAST_GAIN, MSCN_DIFF, TEXTURE_DISSIM, ...) separate
      dither-dst from banding-dst pairs in a 2-feature view?

Inputs (all pre-existing; this script generates nothing):
  --dataset  /mnt/v/output/grain-pathology-2026-07-28  (pairs.tsv +
             features_944.csv, row-aligned; see its _MANIFEST.json)
  --lyb-per-video  the committed LYB eval's per-video table (optional;
             absolute-scale overlay skipped if absent)

Companion doc: benchmarks/bandvis_dither_retest_2026-07-28.md
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

# 944 layout (zensim feature_v2): v1 0..372 | v2 372..720 | append 720..924
# | append2 924..944 (scale*5 + local, Y-only). Channel order B=0, Y=1, X=2.
V2_BASE, APP_BASE, APP2_BASE = 372, 720, 924
V2_PER_CH, APP_PER_CH = 29, 17


def app_idx(scale: int, ch: int, local: int) -> int:
    return APP_BASE + scale * 3 * APP_PER_CH + ch * APP_PER_CH + local


def app2_idx(scale: int, local: int) -> int:
    return APP2_BASE + scale * 5 + local


# append-block Y-lane locals (feature_v2.rs idx_append)
MSCN_DIFF_MEAN, MSCN_DIFF_L2 = 5, 6
CONTRAST_GAIN, CONTRAST_LOSS, TEXTURE_DISSIM = 7, 8, 9
GMS_DEV2, ART_DEV2 = 10, 11

EPS_GAIN = 1e-4  # = C_BV; below this the undithered GAIN is noise-floor


def q(x, p):
    return float(np.percentile(np.asarray(x, dtype=float), p))


def dist(x):
    x = np.asarray(x, dtype=float)
    return (f"n={len(x)} med={q(x, 50):.4f} " f"[{q(x, 25):.4f}, {q(x, 75):.4f}]")


def auc(pos, neg):
    """Rank AUC (Mann-Whitney with average ranks)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    rp = ranks[: len(pos)].sum()
    u = rp - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",
                    default="/mnt/v/output/grain-pathology-2026-07-28")
    ap.add_argument("--lyb-per-video",
                    default=os.path.expanduser("~/tmp/lyb-out/lyb_per_video.csv"))
    ap.add_argument("--out-dir",
                    default=os.path.expanduser("~/tmp/bandvis-retest"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = list(csv.DictReader(open(f"{args.dataset}/pairs.tsv"),
                               delimiter="\t"))
    F = np.loadtxt(f"{args.dataset}/features_944.csv", delimiter=",",
                   skiprows=1, usecols=range(2, 946))
    assert F.shape == (len(rows), 944), F.shape
    # row alignment: ref_basename column must match pairs.tsv ref stem
    with open(f"{args.dataset}/features_944.csv") as fh:
        rd = csv.reader(fh)
        next(rd)
        for i, r in enumerate(rd):
            stem = os.path.basename(rows[i]["ref_path"]).rsplit(".", 1)[0]
            assert r[0] == stem, (i, r[0], stem)
    print(f"loaded {len(rows)} aligned pairs x 944")

    gain = {s: F[:, app2_idx(s, 0)] for s in range(4)}
    loss = {s: F[:, app2_idx(s, 1)] for s in range(4)}

    # ---- index posterize rows by (id, size, bits) -> {mode: row}
    post = defaultdict(dict)
    for i, r in enumerate(rows):
        if r["family"] != "posterize_dither":
            continue
        b, mode = r["variant"].split("_", 1)  # b3_none -> b3, none
        post[(r["id"], r["size"], int(b[1:]))][mode] = i

    # ================= (1) suppression ratios =================
    print("\n== (1) GAIN_dithered / GAIN_undithered at matched "
          "(source,size,bits) ==")
    print(f"   (groups with GAIN_undithered < {EPS_GAIN} excluded; "
          "ratio >1 = dither FIRES, <1 = dither masks)")
    ratio_rows = []
    for dith in ("bayer", "fs"):
        for s in range(4):
            ratios, excl = [], 0
            for key, m in sorted(post.items()):
                if "none" not in m or dith not in m:
                    continue
                g0 = gain[s][m["none"]]
                if g0 < EPS_GAIN:
                    excl += 1
                    continue
                ratios.append(gain[s][m[dith]] / g0)
            fr = float(np.mean(np.asarray(ratios) > 1.0))
            print(f"  {dith:5s} s{s}: {dist(ratios)} frac>1={fr:.2f} "
                  f"(excl {excl})")
            for key, r_ in zip(
                    [k for k, m in sorted(post.items())
                     if "none" in m and dith in m
                     and gain[s][m['none']] >= EPS_GAIN], ratios):
                ratio_rows.append([*key, dith, s, r_])
    with open(f"{args.out_dir}/suppression_ratios.csv", "w") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "size", "bits", "dither", "scale", "ratio"])
        w.writerows(ratio_rows)

    # ================= (2) absolute cross-fire =================
    print("\n== (2) absolute GAIN by (bits, mode), pooled sizes, "
          "median per scale ==")
    hdr = "  bits mode  " + "  ".join(f"s{s}_med" for s in range(4))
    print(hdr)
    for bits in (6, 5, 4, 3):
        for mode in ("none", "bayer", "fs"):
            idx = [m[mode] for k, m in post.items()
                   if k[2] == bits and mode in m]
            if not idx:
                continue
            meds = "  ".join(f"{q(gain[s][idx], 50):6.4f}" for s in range(4))
            print(f"  b{bits}  {mode:5s} {meds}  (n={len(idx)})")
    print("\n  -- per size, dithered rows only (bayer+fs), GAIN med "
          "[q25,q75] --")
    for size in ("native", "s1024"):
        for s in range(4):
            idx = [i for i, r in enumerate(rows)
                   if r["family"] == "posterize_dither"
                   and r["size"] == size and not r["variant"].endswith("none")]
            print(f"  {size:6s} s{s}: {dist(gain[s][idx])}")

    if os.path.exists(args.lyb_per_video):
        lyb = list(csv.DictReader(open(args.lyb_per_video)))
        print(f"\n  -- LYB real-AV1-banding overlay ({len(lyb)} distorted "
              "videos, frame-sampled) --")
        dith_all = [i for i, r in enumerate(rows)
                    if r["family"] == "posterize_dither"
                    and not r["variant"].endswith("none")]
        for s in range(4):
            g = [float(r[f"bv_gain_s{s}"]) for r in lyb]
            lo = [float(r[f"bv_loss_s{s}"]) for r in lyb]
            exc50 = float((gain[s][dith_all] >= q(g, 50)).mean())
            exc90 = float((gain[s][dith_all] >= q(g, 90)).mean())
            print(f"  LYB s{s}: GAIN {dist(g)} p90={q(g, 90):.4f} | "
                  f"LOSS med={q(lo, 50):.4f} | dithered-pairs GAIN >= "
                  f"LYB-p50: {exc50:.1%}, >= p90: {exc90:.1%}")
    else:
        print("\n  (LYB per-video table absent — overlay skipped)")

    # LOSS on the posterize modes (does dither corrupt the workhorse?)
    print("\n  -- posterize family LOSS medians per scale (workhorse "
          "corruption check) --")
    for mode in ("none", "bayer", "fs"):
        idx = [m[mode] for m in post.values() if mode in m]
        lo = "  ".join(f"{q(loss[s][idx], 50):6.4f}" for s in range(4))
        print(f"   {mode:5s} LOSS {lo} (n={len(idx)})")

    # ================= (3) other families =================
    print("\n== (3) per-family GAIN/LOSS medians per scale "
          "(pooled sizes) ==")
    fams = {
        "denoise_regrain": ["den", "rg05", "rg10", "rg20"],
        "jxl_noise": ["d1_plain", "d1_noise", "d1_photon3200",
                      "d3_photon800", "d1_denoise"],
        "av1_grain": ["q80_pn0", "q80_pn24", "q128_pn0", "q128_pn24",
                      "q128_pn48"],
    }
    for fam, variants in fams.items():
        print(f"  -- {fam} --")
        for v in variants:
            idx = [i for i, r in enumerate(rows)
                   if r["family"] == fam and r["variant"] == v]
            g = "  ".join(f"{q(gain[s][idx], 50):6.4f}" for s in range(4))
            lo = "  ".join(f"{q(loss[s][idx], 50):6.4f}" for s in range(4))
            print(f"   {v:13s} GAIN {g} | LOSS {lo} (n={len(idx)})")

    # av1 paired deltas pn-on minus pn-off at matched (id,size,q)
    print("\n  -- av1_grain paired deltas (grain-on minus grain-off) --")
    av1 = defaultdict(dict)
    for i, r in enumerate(rows):
        if r["family"] == "av1_grain":
            qq, pn = r["variant"].split("_")
            av1[(r["id"], r["size"], qq)][pn] = i
    for qq, pn in (("q80", "pn24"), ("q128", "pn24"), ("q128", "pn48")):
        dg = {s: [] for s in range(4)}
        dl = {s: [] for s in range(4)}
        for key, m in av1.items():
            if key[2] != qq or pn not in m or "pn0" not in m:
                continue
            for s in range(4):
                dg[s].append(gain[s][m[pn]] - gain[s][m["pn0"]])
                dl[s].append(loss[s][m[pn]] - loss[s][m["pn0"]])
        g = "  ".join(f"{q(dg[s], 50):+7.4f}" for s in range(4))
        lo = "  ".join(f"{q(dl[s], 50):+7.4f}" for s in range(4))
        print(f"   {qq} {pn}-pn0: dGAIN {g} | dLOSS {lo} "
              f"(n={len(dg[0])})")

    # ================= (4) trainer-side separability =================
    print("\n== (4) dither-dst vs undithered-dst separability (AUC; "
          "posterize family) ==")
    dith_idx = [i for i, r in enumerate(rows)
                if r["family"] == "posterize_dither"
                and not r["variant"].endswith("none")]
    none_idx = [i for i, r in enumerate(rows)
                if r["family"] == "posterize_dither"
                and r["variant"].endswith("none")]
    cands = [(f"bandvis_gain_s{s}", gain[s]) for s in range(4)]
    for name, local in [("contrast_gain", CONTRAST_GAIN),
                        ("contrast_loss", CONTRAST_LOSS),
                        ("mscn_diff", MSCN_DIFF_MEAN),
                        ("mscn_diff_l2", MSCN_DIFF_L2),
                        ("texture_dissim", TEXTURE_DISSIM),
                        ("gms_dev2", GMS_DEV2),
                        ("art_dev2", ART_DEV2)]:
        for s in range(4):
            cands.append((f"{name}_s{s}", F[:, app_idx(s, 1, local)]))
    print(f"   pos = dithered (n={len(dith_idx)}), "
          f"neg = undithered (n={len(none_idx)}); AUC 0.5 = inseparable")
    best = []
    for name, col in cands:
        a = auc(col[dith_idx], col[none_idx])
        best.append((abs(a - 0.5), name, a))
    for _, name, a in sorted(best, reverse=True):
        print(f"   {name:18s} AUC={a:.3f}")

    # ROC points for the top-3 non-BANDVIS gate lanes at Youden max.
    # Direction-aware: if AUC < 0.5 the lane is inverted (dithered LOW);
    # gate with <= t in that case.
    non_bv = sorted((d, n, a) for d, n, a in best
                    if not n.startswith("bandvis"))[::-1][:3]
    cd = dict(cands)
    print()
    for _, gname, ga in non_bv:
        gcol = cd[gname]
        inv = ga < 0.5
        col = -gcol if inv else gcol
        ths = np.unique(np.concatenate([col[dith_idx], col[none_idx]]))
        besty, bt, tpr_b, fpr_b = -1, None, None, None
        for t in ths:
            tpr = float((col[dith_idx] >= t).mean())
            fpr = float((col[none_idx] >= t).mean())
            yj = tpr - fpr
            if yj > besty:
                besty, bt, tpr_b, fpr_b = yj, t, tpr, fpr
        d_ = "dithered-LOW (<=t)" if inv else "dithered-HIGH (>=t)"
        print(f"   gate {gname}: AUC {ga:.3f} [{d_}]; Youden t="
              f"{-bt if inv else bt:.4f}: catches {tpr_b:.1%} of dithered "
              f"@ {fpr_b:.1%} false-flag on undithered")
        r = np.corrcoef(gcol[dith_idx], gain[1][dith_idx])[0, 1]
        print(f"     corr({gname}, bandvis_gain_s1) within dithered: "
              f"{r:+.3f}")


if __name__ == "__main__":
    main()
