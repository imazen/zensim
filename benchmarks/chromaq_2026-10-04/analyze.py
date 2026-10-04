"""CHROMAQ agreement analysis: zenjpeg plane_tables chroma ladders, judges vs zensim bakes.

  python3 analyze.py <sweep444.tsv> <sweep420.tsv> <bakescores.tsv> <out.json>

Inputs: zenmetrics sweep TSVs (score_* columns), and serve_custom_bake --pairs output whose dist_path joins to
the sweep's distorted filename. Every score is oriented higher = better (butteraugli negated).
"""

import csv
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

from scipy.stats import kendalltau, spearmanr

JUDGES = {"cvvdp": ("score_cvvdp_cpu_imazen_v0_1_0_standard_4k", 1), "butter_p3": ("score_butteraugli_pnorm3", -1),
          "butter_max": ("score_butteraugli_max", -1), "vmaf": ("score_vmaf", 1), "ssim2": ("score_ssim2", 1),
          "zensim_B": ("score_zensim", 1), "psnr_y": ("score_psnr_y", 1)}


def family(knobs: dict) -> tuple:
    pt = knobs["plane_tables"]
    y, cb, cr = pt["m"]
    mask = pt.get("chroma_mask", "all")
    sub = knobs["subsampling"]
    if sub == "420":
        return ("s420", max(cb, cr))
    if y == 0 and cb == 0 and cr == 0:
        return ("base", 0)
    if cb == 0 and cr == 0:
        return ("luma", y)
    if y == 1:
        return ("mix_y1", cb)
    if cb == cr:
        return ({"all": "chroma", "lf": "chroma_lf", "hf": "chroma_hf"}[mask], cb)
    return ("cb_only", cb) if cr == 0 else ("cr_only", cr)


def load(paths, bake_tsv):
    rows = []
    for p in paths:
        for r in csv.DictReader(open(p), delimiter="\t"):
            k = json.loads(r["knob_tuple_json"])
            fam, lvl = family(k)
            d = {"img": os.path.basename(r["image_path"]), "fam": fam, "lvl": float(lvl), "bytes": int(r["encoded_bytes"]),
                 "dist": r["encoded_filename"].rsplit(".", 1)[0] + ".png"}
            for name, (col, sign) in JUDGES.items():
                v = r.get(col, "")
                d[name] = sign * float(v) if v not in ("", None) else math.nan
            rows.append(d)
    bake = {}
    with open(bake_tsv) as f:
        rd = csv.reader(f, delimiter="\t")
        hdr = next(rd)
        names = [os.path.basename(h)[:-4] for h in hdr[2:]]
        for r in rd:
            bake[os.path.basename(r[1])] = dict(zip(names, map(float, r[2:])))
    sets = sorted({n.split("-")[0] for n in names})
    for d in rows:
        b = bake.get(d["dist"])
        if b is None:
            raise SystemExit(f"no bake score for {d['dist']}")
        for s in sets:
            vals = [v for n, v in b.items() if n.split("-")[0] == s]
            d[s] = st.fmean(vals)
            for n, v in b.items():
                if n.split("-")[0] == s:
                    d[f"{n}"] = v
    return rows, sets, names


def srocc(a, b):
    ok = [(x, y) for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    if len(ok) < 3:
        return math.nan
    return float(spearmanr([x for x, _ in ok], [y for _, y in ok]).statistic)


def per_image(rows, metric_a, metric_b, fams):
    by = defaultdict(list)
    for d in rows:
        if d["fam"] in fams:
            by[d["img"]].append(d)
    vals = [srocc([d[metric_a] for d in v], [d[metric_b] for d in v]) for v in by.values()]
    vals = [v for v in vals if not math.isnan(v)]
    return (st.fmean(vals), min(vals)) if vals else (math.nan, math.nan)


def monotone(rows, metric, fams):
    """Per (image, family) ladder: share with Kendall tau == -1 vs level (strictly worse as the table coarsens)."""
    by = defaultdict(list)
    for d in rows:
        if d["fam"] in fams:
            by[(d["img"], d["fam"])].append(d)
    taus, viol = [], 0
    for v in by.values():
        v = sorted(v, key=lambda d: d["lvl"])
        xs, ys = [d["lvl"] for d in v], [d[metric] for d in v]
        t = kendalltau(xs, ys).statistic
        taus.append(t)
        viol += sum(1 for a, b in zip(ys, ys[1:]) if b > a)
    return {"mean_tau": st.fmean(taus), "ladders": len(taus), "upticks": viol}


def cross_family(rows, metric, judge, fam_a="luma", fam_b=("chroma", "chroma_lf", "chroma_hf", "cb_only", "cr_only")):
    """Within each image, every (luma cell, chroma cell) pair: share ordered the same way by metric and judge.
    This isolates the chroma-vs-luma exchange rate; within-family ordering is excluded."""
    by = defaultdict(lambda: ([], []))
    for d in rows:
        if d["fam"] == fam_a:
            by[d["img"]][0].append(d)
        elif d["fam"] in fam_b:
            by[d["img"]][1].append(d)
    agree = total = 0
    for la, ch in by.values():
        for a in la:
            for c in ch:
                dj = a[judge] - c[judge]
                dm = a[metric] - c[metric]
                if dj == 0 or dm == 0 or math.isnan(dj) or math.isnan(dm):
                    continue
                total += 1
                agree += (dj > 0) == (dm > 0)
    return agree / total if total else math.nan, total


def main() -> int:
    s444, s420, bake_tsv, out = sys.argv[1:5]
    rows, sets, names = load([s444, s420], bake_tsv)
    chroma_f = {"chroma", "chroma_lf", "chroma_hf", "cb_only", "cr_only"}
    all_f = chroma_f | {"luma", "base", "mix_y1", "s420"}
    metrics = list(JUDGES) + sets
    res = {"n_cells": len(rows), "images": len({d["img"] for d in rows}), "sets": sets, "bakes": names}
    # Luma control: in chroma-only families luma must not move.
    res["luma_control"] = {m: {"chroma_cells_range": [min(d[m] for d in rows if d["fam"] in chroma_f),
                                                      max(d[m] for d in rows if d["fam"] in chroma_f)]}
                           for m in ("psnr_y", "vmaf")}
    res["monotone"] = {m: {f: monotone(rows, m, {f}) for f in sorted({d["fam"] for d in rows} - {"base"})} for m in metrics}
    judges = ["cvvdp", "butter_p3", "butter_max", "vmaf", "ssim2"]
    res["per_image_srocc"] = {}
    for scope, fams in (("chroma_only", chroma_f), ("all", all_f), ("luma_only", {"luma", "base"})):
        res["per_image_srocc"][scope] = {f"{m}~{j}": per_image(rows, m, j, fams | {"base"}) for m in sets + ["zensim_B"]
                                         for j in judges}
        res["per_image_srocc"][scope].update({f"{a}~{b}": per_image(rows, a, b, fams | {"base"})
                                              for i, a in enumerate(judges) for b in judges[i + 1:]})
    # Consensus judges: per image, the mean of within-image ranks of the named judges (the owner's "3" = CVVDP,
    # butteraugli, VMAF; and the colour-aware pair CVVDP + butteraugli).
    from scipy.stats import rankdata
    by_img = defaultdict(list)
    for d in rows:
        by_img[d["img"]].append(d)
    for name, js in (("consensus3", ("cvvdp", "butter_p3", "vmaf")), ("consensus_colour", ("cvvdp", "butter_p3"))):
        for v in by_img.values():
            ranks = [rankdata([d[j] for d in v]) for j in js]
            for i, d in enumerate(v):
                d[name] = float(sum(r[i] for r in ranks) / len(js))
    judges_c = judges + ["consensus3", "consensus_colour"]
    for scope, fams in (("chroma_only", chroma_f), ("all", all_f), ("luma_only", {"luma", "base"})):
        res["per_image_srocc"][scope].update({f"{m}~{j}": per_image(rows, m, j, fams | {"base"})
                                              for m in sets + ["zensim_B", "ssim2", "psnr_y"] for j in ("consensus3", "consensus_colour")})
    res["cross_family"] = {f"{m}~{j}": cross_family(rows, m, j) for m in sets + judges + ["zensim_B"] for j in judges_c if m != j}
    # Pooled (across images) SROCC on chroma-only cells: does the metric order chroma damage across content as the judges do?
    res["pooled_srocc_chroma"] = {f"{m}~{j}": srocc([d[m] for d in rows if d["fam"] in chroma_f],
                                                   [d[j] for d in rows if d["fam"] in chroma_f])
                                  for m in sets + judges + ["zensim_B"] for j in judges if m != j}
    # Ladder medians per family and level (for curves).
    curves = defaultdict(dict)
    for f in sorted({d["fam"] for d in rows}):
        for lvl in sorted({d["lvl"] for d in rows if d["fam"] == f}):
            sel = [d for d in rows if d["fam"] == f and d["lvl"] == lvl]
            curves[f][str(lvl)] = {m: st.median(d[m] for d in sel) for m in metrics} | {"bytes": st.median(d["bytes"] for d in sel)}
    res["curves"] = curves
    json.dump(res, open(out, "w"), indent=1)
    with open(out.replace(".json", ".rows.tsv"), "w") as f:
        cols = ["img", "fam", "lvl", "bytes", "dist"] + metrics + names
        f.write("\t".join(cols) + "\n")
        for d in rows:
            f.write("\t".join(str(d[c]) for c in cols) + "\n")
    print(json.dumps({k: res[k] for k in ("n_cells", "images", "sets")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
