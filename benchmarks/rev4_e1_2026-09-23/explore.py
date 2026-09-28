#!/usr/bin/env python3
"""Rev4 E1 — EXPLORATORY (not preregistered): what do the losing bands share?

The program doc's "refuted" branch asks what the losing bands have in common.
This splits each band's within-reference pairs into same-codec (or, on CSIQ,
same-distortion-type) and cross-codec pairs, and reports pairwise accuracy per
split. Statistic, bootstrap and draws are the registered ones (analyze.py);
only the pair filter is new, so every number here is exploratory.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze as A  # noqa: E402

CODEC = {
    "cid22a": lambda s: s.split("/")[2],
    "csiq": lambda s: s.split(".")[1],
    "aic3": lambda s: s.split("_")[0],
    "aic4crop": lambda s: s.split("_")[1],
    "aic4full": lambda s: s.split("_")[1],
}
FOCUS = ["ssim2", "B", "C", "D", "R915_fast", "R915_rich"]


def run(rows, idx, metrics, rd, scratch, tag, panel_bin, want):
    codec = {i: CODEC[tag.split("__")[0]](rows[i]["stim"]) for i in idx}
    by_ref = {}
    for i in idx:
        by_ref.setdefault(rows[i]["ref"], []).append(i)
    pairs = []
    for r, ii in by_ref.items():
        for a in range(len(ii)):
            for b in range(a + 1, len(ii)):
                i, j = ii[a], ii[b]
                ti, tj = float(rows[i]["t"]), float(rows[j]["t"])
                if ti == tj:
                    continue
                same = codec[i] == codec[j]
                if (want == "same") != same:
                    continue
                pairs.append((r, i, j, "left" if ti < tj else "right"))
    if not pairs:
        return None, 0
    groups = sorted({p[0] for p in pairs})
    gidx = {g: k for k, g in enumerate(groups)}
    man = f"{scratch}/{tag}__{want}__resample.tsv"
    with open(man, "w") as f:
        f.write("POINT\t*\n")
        for b, dr in enumerate(rd):
            s = [str(gidx[r]) for r in dr if r in gidx]
            if s:
                f.write(f"B{b}\t{','.join(s)}\n")
    out = {}
    for m in metrics:
        rf = f"{scratch}/{tag}__{want}__{m}.tsv"
        with open(rf, "w") as f:
            f.write("group\ts_left\ts_right\tchoice\tweight\n")
            for (r, i, j, ch) in pairs:
                f.write(f"{r}\t{rows[i][m]}\t{rows[j][m]}\t{ch}\t1\n")
        p = subprocess.run([panel_bin, "--pairwise", rf, "--resample", man], capture_output=True, text=True, check=True)
        lines = p.stdout.strip().split("\n")
        hdr = lines[0].split("\t")
        boot = [float("nan")] * len(rd)
        point = float("nan")
        for l in lines[1:]:
            rec = dict(zip(hdr, l.split("\t")))
            if rec["label"] == "POINT":
                point = float(rec["acc_response"])
            else:
                boot[int(rec["label"][1:])] = float(rec["acc_response"])
        out[m] = {"point": point, "boot": boot}
    return out, len(pairs)


def main():
    full = json.load(open("/var/tmp/rev4-e1/e1_full.json"))
    scratch = "/var/tmp/rev4-e1/explore_scratch"
    os.makedirs(scratch, exist_ok=True)
    panel_bin = A.zen_stats._find_panel_bin()
    res = {"note": "EXPLORATORY - not preregistered", "corpora": {}}
    for c in CODEC:
        rows, cols = A.load("/var/tmp/rev4-e1/tables", c)
        t = [float(r["t"]) for r in rows]
        bl, _ = A.bands_for(c, t)
        refs = sorted({r["ref"] for r in rows})
        rd = A.draws(refs, 2000, A.SEED)
        res["corpora"][c] = {}
        for b in ["ALL"] + sorted(set(bl)):
            idx = [i for i in range(len(rows)) if b == "ALL" or bl[i] == b]
            best = full["corpora"][c]["bands"][b]["pairwise"]["best_peer"]
            metrics = [m for m in dict.fromkeys(FOCUS + [best]) if m in cols]
            ent = {}
            for want in ("same", "cross"):
                st, n = run(rows, idx, metrics, rd, scratch, f"{c}__{b}", panel_bin, want)
                if st is None:
                    continue
                e = {"n_pairs": n, "best_peer": best, "acc": {m: round(v["point"], 4) for m, v in st.items()}, "d": {}}
                for z in metrics:
                    if z in A.PEERS:
                        continue
                    for p in dict.fromkeys(["ssim2", best]):
                        d = [x - y for x, y in zip(st[z]["boot"], st[p]["boot"]) if not (math.isnan(x) or math.isnan(y))]
                        e["d"][f"{z}-{p}"] = [round(st[z]["point"] - st[p]["point"], 4),
                                              round(A.pct(d, .025), 4), round(A.pct(d, .975), 4)]
                ent[want] = e
            res["corpora"][c][b] = ent
            print(c, b, {w: v["n_pairs"] for w, v in ent.items()}, flush=True)
    json.dump(res, open("/var/tmp/rev4-e1/explore.json", "w"), indent=1)


if __name__ == "__main__":
    main()
