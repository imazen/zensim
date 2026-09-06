#!/usr/bin/env python3
"""Summarize the BEST-OF-ALL wave: per-arm k-seed means, spreads, and the
full dial-gate table, read from the artifacts and RE-DERIVING NOTHING.

Every number here is copied out of a `bake_verdict --full-json` or
`--gaddr-json` file. The stats owner is `zenstats` via `bake_verdict`; this
script only aggregates across seeds (mean, min, max) and formats.

Usage: bestofall_report.py [--out <dir>] [--md <path>]
"""
import argparse
import json
import os
import re
import statistics
import sys
from collections import defaultdict

ARMS = ["A_plain", "B_nonneg", "C_lad05", "D_lad20", "E_plainlad", "F_nonneg32"]
ARM_DESC = {
    "A_plain": "CONTROL — the fastclass2 winner's recipe, unchanged",
    "B_nonneg": "+ --nonneg-distance (architecture only)",
    "C_lad05": "+ --nonneg-distance + ladder hinge @ tv-weight 0.5",
    "D_lad20": "+ --nonneg-distance + ladder hinge @ tv-weight 2.0",
    "E_plainlad": "control + ladder hinge @ 0.5 (isolates the LOSS)",
    "F_nonneg32": "B at --hidden 32",
}
CORPORA = ["cid22", "konjnd", "aic3", "tid", "kadid", "csiq", "live", "hfnlproxy",
           "imazen26", "nonphoto"]
# Shipped D era-2 (`zensim/weights/d_sdr_add156_id100_negrich_dial_byid_2026-09-06.bin`),
# MEASURED BY THIS LANE on the postC root with the SAME `bake_verdict`
# invocation the wave's arms use — not transcribed. Its rank values reproduce
# `benchmarks/rev2_d_arms_2026-09-06.md` §12.3 to six digits, which is what says
# the reference is the same one.
#
# ⚠ The COMPOSITE does NOT match that record's 0.8064387598449834, and should
# not: `product_composite` is a function of the corpus SET a run scored, and
# that record's run used a different one. Comparing a candidate against the
# number below is apples-to-apples; comparing it against 0.8064 is not.
SHIPPED_D = {
    "cid22": 0.8633299325973021, "konjnd": 0.5367043689366323,
    "aic3": 0.7769958141384993, "csiq": 0.9016689158701747,
    "live": 0.9602891256047629, "tid": 0.823691890065306,
    "kadid": 0.8080635746122159, "hfnlproxy": 0.4920983413616811,
    "composite": 0.8244399655192828,
}
MENTOR_FLOORS = {
    "avif-rav1e": 0.6410256410256411, "avif-svt": 1.0,
    "jpeg": 0.6666666666666666, "jxl": 0.9615384615384616, "webp": 1.0,
}


def load(out):
    cells = {}
    vdir = os.path.join(out, "verdicts")
    gdir = os.path.join(out, "gaddr")
    if not os.path.isdir(vdir):
        return cells
    for fn in sorted(os.listdir(vdir)):
        m = re.match(r"(.+)_s(\d+)\.fulleval\.json$", fn)
        if not m:
            continue
        arm, seed = m.group(1), m.group(2)
        d = json.load(open(os.path.join(vdir, fn)))
        g = None
        gp = os.path.join(gdir, f"gaddr_{arm}_s{seed}.json")
        if os.path.exists(gp):
            g = json.load(open(gp))
        cells[(arm, seed)] = (d, g)
    return cells


def rank_of(d, corpus):
    r = d.get("rank", {}).get(corpus)
    if not r:
        return None
    v = r.get("srocc_signed")
    return abs(v) if (v is not None and corpus == "konjnd") else v


def composite_of(d):
    for k in ("product_composite", "composite", "balanced_composite"):
        v = d.get(k)
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, dict):
            for kk in ("product_composite", "value", "product"):
                if isinstance(v.get(kk), (int, float)):
                    return v[kk]
    return None


def fmt(v, n=4):
    return "—" if v is None else f"{v:.{n}f}"


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    return statistics.fmean(vals), (max(vals) - min(vals)), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/v/output/zensim/best-of-all-2026-09-06")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()

    cells = load(a.out)
    if not cells:
        sys.exit(f"no fullevals under {a.out}/verdicts")

    by_arm = defaultdict(list)
    for (arm, seed), v in sorted(cells.items()):
        by_arm[arm].append((seed, v))

    lines = []
    w = lines.append
    w("## Arm table — k-seed means (spread = max − min over seeds)\n")
    w("| arm | k | CID22 | KonJND \\| | AIC-3 | composite | contract | A7r | identity outside | above-identity |")
    w("|---|--:|--:|--:|--:|--:|:--:|--:|--:|--:|")
    for arm in ARMS:
        rows = by_arm.get(arm, [])
        if not rows:
            continue
        cid = agg([rank_of(d, "cid22") for _s, (d, _g) in rows])
        kon = agg([rank_of(d, "konjnd") for _s, (d, _g) in rows])
        aic = agg([rank_of(d, "aic3") for _s, (d, _g) in rows])
        comp = agg([composite_of(d) for _s, (d, _g) in rows])
        cstate, a7r, c5, c6 = [], [], [], []
        for _s, (_d, g) in rows:
            if not g:
                continue
            chk = {c["id"]: c for c in g["checks"]}
            npass = sum(1 for i in "C1 C2 C3 C4 C5 C6".split()
                        if chk.get(i, {}).get("state") == "pass")
            cstate.append(npass)
            a7r.append(chk.get("A7r", {}).get("measured"))
            c5.append(chk.get("C5", {}).get("measured"))
            c6.append(chk.get("C6", {}).get("measured"))
        w(f"| **{arm}** | {len(rows)} | {fmt(cid[0])} ±{fmt(cid[1])} | {fmt(kon[0])} ±{fmt(kon[1])} "
          f"| {fmt(aic[0])} ±{fmt(aic[1])} | {fmt(comp[0])} ±{fmt(comp[1])} "
          f"| {min(cstate) if cstate else '—'}–{max(cstate) if cstate else '—'}/6 "
          f"| {fmt(agg(a7r)[0], 1)} | {fmt(agg(c5)[0], 1)} | {fmt(agg(c6)[0], 1)} |")
    w("")
    w(f"Shipped D (era-2), MEASURED BY THIS LANE on the same root and the same "
      f"invocation: CID22 **{SHIPPED_D['cid22']:.5f}**, KonJND "
      f"|{SHIPPED_D['konjnd']:.5f}|, AIC-3 {SHIPPED_D['aic3']:.5f}, composite "
      f"{SHIPPED_D['composite']:.5f}. Its published contract is **6/6** and its "
      f"A7r **0 of 5**.\n")

    w("## Per-codec resolvable floors (bar = the mentor's own value; a codec passes at ≥ bar)\n")
    hdr = "| arm | " + " | ".join(MENTOR_FLOORS) + " |"
    w(hdr)
    w("|---|" + "--:|" * len(MENTOR_FLOORS))
    w("| _mentor bar_ | " + " | ".join(f"{v:.4f}" for v in MENTOR_FLOORS.values()) + " |")
    for arm in ARMS:
        rows = by_arm.get(arm, [])
        per = defaultdict(list)
        for _s, (_d, g) in rows:
            if not g:
                continue
            for c in g["measured"].get("codec_floor", []):
                per[c["codec"]].append(c["represented_frac"])
        if not per:
            continue
        cellsv = []
        for codec, bar in MENTOR_FLOORS.items():
            m = agg(per.get(codec, []))[0]
            mark = "" if m is None else ("" if m >= bar else " ✗")
            cellsv.append(("—" if m is None else f"{m:.4f}") + mark)
        w(f"| {arm} | " + " | ".join(cellsv) + " |")
    w("")

    w("## Full rank panel (k-seed means)\n")
    w("| arm | " + " | ".join(CORPORA) + " |")
    w("|---|" + "--:|" * len(CORPORA))
    for arm in ARMS:
        rows = by_arm.get(arm, [])
        if not rows:
            continue
        vals = [fmt(agg([rank_of(d, c) for _s, (d, _g) in rows])[0]) for c in CORPORA]
        w(f"| {arm} | " + " | ".join(vals) + " |")
    w("")

    # The constraint cost, at matched seeds.
    w("## The constraint cost, at MATCHED seeds\n")
    w("| pair | axis | Δ (mean over matched seeds) | per-seed |")
    w("|---|---|--:|---|")
    for lo, hi, label in [("A_plain", "B_nonneg", "architecture"),
                          ("A_plain", "E_plainlad", "ladder loss alone"),
                          ("B_nonneg", "C_lad05", "ladder on top of the architecture"),
                          ("A_plain", "C_lad05", "both")]:
        for corpus in ("cid22", "konjnd", "aic3"):
            da, db = dict(by_arm.get(lo, [])), dict(by_arm.get(hi, []))
            seeds = sorted(set(da) & set(db))
            if not seeds:
                continue
            per = [rank_of(db[s][0], corpus) - rank_of(da[s][0], corpus)
                   for s in seeds
                   if rank_of(da[s][0], corpus) is not None
                   and rank_of(db[s][0], corpus) is not None]
            if not per:
                continue
            w(f"| {hi} − {lo} ({label}) | {corpus} | {statistics.fmean(per):+.5f} | "
              + ", ".join(f"{p:+.5f}" for p in per) + " |")
    w("")

    txt = "\n".join(lines)
    print(txt)
    if a.md:
        with open(a.md, "w") as f:
            f.write(txt + "\n")


if __name__ == "__main__":
    main()
