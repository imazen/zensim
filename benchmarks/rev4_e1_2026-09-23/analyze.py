#!/usr/bin/env python3
"""Rev4 E1 — per-band statistics on the assembled stimulus tables.

Registered in benchmarks/rev4_e1_prereg_2026-09-23.md (§4 bands, §5 statistics,
§6 decision rule). No statistic is computed in this file:

* band SROCC       -> `panel --batch --stats srocc` (scripts/lib/zen_stats)
* pairwise accuracy -> `panel --pairwise --resample` (zensim_validate::pairwise)

This file owns only the bands (functions of the target), the pair lists, the
reference-clustered resample draws (the caller owns the RNG, per the owner's
contract) and the percentile read of the owner's per-resample outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
from lib import zen_stats  # noqa: E402

PEERS = ["ssim2", "butter_p3", "butter_max", "iwssim", "cvvdp_4k", "cvvdp_fhd", "dssim",
         "pub_iwssim", "pub_msssim", "pub_psnry", "pub_cvvdp"]
CONTEXT = ["ctx_ssim", "ctx_vmafneg", "ctx_hdrvdp2", "ctx_hdrvdp3"]
ZENSIM = ["B", "C", "D", "R915_fast", "R915_rich", "V0_2"]
JND = {"aic3", "aic4crop", "aic4full", "sdr25"}
MOS = {"cid22a", "csiq"}
CORPORA = ["cid22a", "csiq", "konjnd404", "aic3", "aic4crop", "aic4full", "sdr25"]
SEED = 20260923


def load(tables, c):
    rows = list(csv.DictReader(open(f"{tables}/{c}.tsv"), delimiter="\t"))
    cols = [k for k in rows[0] if k not in ("stim", "ref", "t")]
    cols = [k for k in cols if all(r[k] != "" for r in rows)]
    return rows, cols


def bands_for(c, t):
    """Band label per row (prereg §4). Target-only."""
    if c in JND:
        out = []
        for v in t:
            d = -v
            out.append("NT" if d < 1 else ("MID" if d < 2 else "LOW"))
        return out, {"NT": "d<1 JND", "MID": "1<=d<2", "LOW": "d>=2"}
    if c in MOS:
        e = np.percentile(np.asarray(t, dtype=float), [20, 40, 60, 80])
        out = []
        for v in t:
            k = int(sum(v > x for x in e))  # value equal to an edge -> lower band
            out.append(f"Q{k + 1}")
        return out, {"edges_p20_p40_p60_p80": [float(x) for x in e]}
    assert c == "konjnd404"
    return ["NT"] * len(t), {"NT": "whole corpus (every stimulus at PJND)"}


def band_role(c, b):
    if c in MOS:
        return {"Q5": "NT", "Q4": "MID", "Q3": "MID", "Q2": "LOW", "Q1": "LOW"}[b]
    return b


def draws(refs, B, seed):
    rng = random.Random(seed)
    n = len(refs)
    return [[refs[rng.randrange(n)] for _ in range(n)] for _ in range(B)]


def pct(v, q):
    v = sorted(x for x in v if not math.isnan(x))
    if not v:
        return float("nan")
    return v[min(len(v) - 1, max(0, int(q * len(v))))]


def band_srocc(rows, idx, metrics, ref_draws):
    """Point + per-resample orientation-aligned SROCC for each metric (owner: panel)."""
    if len(idx) < 3:
        return None
    by_ref = {}
    for j, i in enumerate(idx):
        by_ref.setdefault(rows[i]["ref"], []).append(j)
    bases = {"t": [float(rows[i]["t"]) for i in idx]}
    for m in metrics:
        bases[m] = [float(rows[i][m]) for i in idx]
    jobs = [(f"{m}|P", m, "t", None) for m in metrics]
    sel = []
    for b, dr in enumerate(ref_draws):
        s = [j for r in dr for j in by_ref.get(r, [])]
        sel.append(s)
        if len(s) >= 3:
            for m in metrics:
                jobs.append((f"{m}|{b}", m, "t", s))
    res = zen_stats.panel_batch_indexed(bases, jobs, stats="srocc")
    out = {m: {"point": float("nan"), "boot": [float("nan")] * len(ref_draws)} for m in metrics}
    for r in res:
        m, k = r["label"].split("|")
        v = r["srocc_signed"]
        v = float("nan") if v is None else float(v)
        if k == "P":
            out[m]["point"] = v
        else:
            out[m]["boot"][int(k)] = v
    return out


def pairwise(rows, idx, metrics, ref_draws, scratch, tag, panel_bin):
    """Within-reference pairs, both in the band, target ties dropped (owner: panel --pairwise)."""
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
                pairs.append((r, i, j, "left" if ti < tj else "right"))
    if not pairs:
        return None, 0, 0
    groups = sorted({p[0] for p in pairs})
    gidx = {g: k for k, g in enumerate(groups)}
    man = f"{scratch}/{tag}__resample.tsv"
    with open(man, "w") as f:
        f.write("POINT\t*\n")
        for b, dr in enumerate(ref_draws):
            s = [str(gidx[r]) for r in dr if r in gidx]
            if s:
                f.write(f"B{b}\t{','.join(s)}\n")
    out = {}
    for m in metrics:
        rf = f"{scratch}/{tag}__{m}.tsv"
        with open(rf, "w") as f:
            f.write("group\ts_left\ts_right\tchoice\tweight\n")
            for (r, i, j, ch) in pairs:
                f.write(f"{r}\t{rows[i][m]}\t{rows[j][m]}\t{ch}\t1\n")
        p = subprocess.run([panel_bin, "--pairwise", rf, "--resample", man], capture_output=True, text=True)
        if p.returncode != 0:
            raise SystemExit(f"panel --pairwise failed ({tag}/{m}):\n{p.stderr[-2000:]}")
        lines = p.stdout.strip().split("\n")
        hdr = lines[0].split("\t")
        recs = [dict(zip(hdr, l.split("\t"))) for l in lines[1:]]
        boot = [float("nan")] * len(ref_draws)
        point = tie = float("nan")
        for rec in recs:
            if rec["label"] == "POINT":
                point = float(rec["acc_response"])
                tie = float(rec["tie_rate"])
            else:
                boot[int(rec["label"][1:])] = float(rec["acc_response"])
        out[m] = {"point": point, "boot": boot, "tie_rate": tie}
    return out, len(pairs), len(groups)


def summarize(stat, metrics, peers, zmods):
    """Points, CIs, and paired deltas (zensim - peer) for every zensim x peer."""
    s = {}
    for m in metrics:
        v = stat[m]
        s[m] = {"point": v["point"], "ci": [pct(v["boot"], 0.025), pct(v["boot"], 0.975)]}
        if "tie_rate" in v:
            s[m]["tie_rate"] = v["tie_rate"]
    best = None
    if peers:
        best = max(peers, key=lambda p: (-1e9 if math.isnan(stat[p]["point"]) else stat[p]["point"]))
    deltas = {}
    for z in zmods:
        for p in peers:
            d = [a - b for a, b in zip(stat[z]["boot"], stat[p]["boot"]) if not (math.isnan(a) or math.isnan(b))]
            deltas[f"{z}-{p}"] = {"point": stat[z]["point"] - stat[p]["point"], "ci": [pct(d, 0.025), pct(d, 0.975)],
                                  "p_gt0": (sum(x > 0 for x in d) / len(d)) if d else float("nan"), "n_boot": len(d)}
    return s, best, deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="/var/tmp/rev4-e1/tables")
    ap.add_argument("--scratch", default="/var/tmp/rev4-e1/scratch")
    ap.add_argument("--out", default="/var/tmp/rev4-e1/e1_full.json")
    ap.add_argument("-B", type=int, default=2000)
    ap.add_argument("--corpora", default=",".join(CORPORA))
    a = ap.parse_args()
    os.makedirs(a.scratch, exist_ok=True)
    panel_bin = zen_stats._find_panel_bin()
    result = {"seed": SEED, "B": a.B, "panel_bin": panel_bin, "corpora": {}}
    for c in a.corpora.split(","):
        rows, cols = load(a.tables, c)
        metrics = [m for m in cols]
        peers = [m for m in metrics if m in PEERS]
        zmods = [m for m in metrics if m in ZENSIM]
        t = [float(r["t"]) for r in rows]
        bl, bdef = bands_for(c, t)
        refs = sorted({r["ref"] for r in rows})
        rd = draws(refs, a.B, SEED)
        cres = {"n_rows": len(rows), "n_refs": len(refs), "band_def": bdef, "metrics": metrics,
                "peers": peers, "zensim": zmods, "bands": {}}
        band_names = sorted(set(bl), key=lambda b: {"NT": 0, "MID": 1, "LOW": 2}.get(b, 0) if c not in MOS else -int(b[1]))
        for b in ["ALL"] + band_names:
            idx = [i for i in range(len(rows)) if b == "ALL" or bl[i] == b]
            ent = {"role": "GLOBAL" if b == "ALL" else band_role(c, b), "n_stim": len(idx),
                   "n_refs": len({rows[i]["ref"] for i in idx}),
                   "t_range": [min(t[i] for i in idx), max(t[i] for i in idx)]}
            sr = band_srocc(rows, idx, metrics, rd)
            if sr is not None:
                s, best, deltas = summarize(sr, metrics, peers, zmods)
                ent["srocc"] = {"metrics": s, "best_peer": best, "deltas": deltas}
            if c != "konjnd404":
                pw, npairs, ngroups = pairwise(rows, idx, metrics, rd, a.scratch, f"{c}__{b}", panel_bin)
                ent["n_pairs"], ent["n_pair_refs"] = npairs, ngroups
                if pw is not None:
                    s, best, deltas = summarize(pw, metrics, peers, zmods)
                    ent["pairwise"] = {"metrics": s, "best_peer": best, "deltas": deltas}
            cres["bands"][b] = ent
            print(f"{c} {b}: n_stim={ent['n_stim']} refs={ent['n_refs']} pairs={ent.get('n_pairs')}", flush=True)
        result["corpora"][c] = cres
    json.dump(result, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
