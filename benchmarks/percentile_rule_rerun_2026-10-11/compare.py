#!/usr/bin/env python3
"""Compare bake_compare outputs under the order-statistic and type 7 percentile rules.

For every (pair, corpus, cell) where cell is the corpus aggregate or a band with
a decisive outcome: report agree counts and decision under both rules, every CI
bound whose sign relative to 0 changes, and the bound shift relative to the CI
width. Also checks reproduction of the recorded reports on CID22 / AIC-3
(SROCC_A, SROCC_B, aggregate decision) — the other corpora were run on SELECT
rows only and are not comparable to the recorded full-table runs.
"""
import json, os, re, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
pairs = json.load(open(os.path.join(here, "pairs.json")))
STATS = ["srocc", "plcc", "krocc", "or", "pwrc", "z_rmse"]

def cells(d):
    for c in d["corpora"]:
        yield c["display"], "aggregate", c["aggregate"]
        for b in c.get("per_band") or []:
            if b.get("outcome"):
                yield c["display"], b["label"], b["outcome"]

def recorded_row(report, corpus):
    t = subprocess.check_output(["git", "-C", os.path.expanduser("~/work/zen/zensim"), "show", "origin/main:" + report]).decode()
    for line in t.splitlines():
        if line.startswith(f"| {corpus} |"):
            f = [x.strip() for x in line.strip("|").split("|")]
            return {"srocc_a": float(f[2]), "srocc_b": float(f[3]), "decision": f[-1]}
    return None

rows, flips, shifts, repro = [], [], [], []
n_cells = 0
for p in pairs:
    name = p["report"].split("/")[-1][:-3]
    fo = os.path.join(here, "out", f"{name}__orderstat.json")
    ft = os.path.join(here, "out", f"{name}__type7.json")
    if not (os.path.exists(fo) and os.path.exists(ft)):
        continue
    o, t = json.load(open(fo)), json.load(open(ft))
    to = {(c, k): v for c, k, v in cells(t)}
    for corpus, cell, vo in cells(o):
        vt = to[(corpus, cell)]
        n_cells += 1
        if (vo["agreement_a"], vo["agreement_b"], vo["decision"]) != (vt["agreement_a"], vt["agreement_b"], vt["decision"]):
            flips.append((name, corpus, cell, vo["agreement_a"], vo["agreement_b"], vo["decision"], vt["agreement_a"], vt["agreement_b"], vt["decision"]))
        for s, (lo_o, hi_o), (lo_t, hi_t) in zip(STATS, vo["ci_delta"], vt["ci_delta"]):
            if None in (lo_o, hi_o, lo_t, hi_t):
                continue
            w = hi_o - lo_o
            if w > 0:
                shifts.append((abs(lo_t - lo_o) / w, abs(hi_t - hi_o) / w))
            for bo, bt, side in ((lo_o, lo_t, "lo"), (hi_o, hi_t, "hi")):
                if (bo > 0) != (bt > 0) or (bo < 0) != (bt < 0):
                    rows.append((name, corpus, cell, s, side, bo, bt))
    for corpus in ("CID22", "AIC-3 CTC"):
        rec = recorded_row(p["report"], corpus)
        agg = next(v for c, k, v in cells(o) if c == corpus and k == "aggregate")
        if rec:
            repro.append((name, corpus, rec["srocc_a"], round(agg["panel_a"]["srocc"], 4), rec["srocc_b"], round(agg["panel_b"]["srocc"], 4), rec["decision"], agg["decision"]))

out = []
out.append(f"cells compared: {n_cells}")
out.append(f"cells whose agree counts or decision differ between rules: {len(flips)}")
for f in flips:
    out.append("  FLIP " + " | ".join(map(str, f)))
out.append(f"CI bounds that cross 0 between rules: {len(rows)}")
for r in rows:
    out.append("  CROSS " + " | ".join(map(str, r)))
if shifts:
    lo = sorted(s[0] for s in shifts); hi = sorted(s[1] for s in shifts)
    q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))]
    out.append(f"bound shift / CI width over {len(shifts)} intervals: lower median {q(lo,.5):.4f} p99 {q(lo,.99):.4f} max {lo[-1]:.4f}; upper median {q(hi,.5):.4f} p99 {q(hi,.99):.4f} max {hi[-1]:.4f}")
out.append("reproduction of recorded CID22 / AIC-3 aggregates (recorded vs re-run, order-statistic binary):")
for r in repro:
    out.append("  " + " | ".join(map(str, r)))
print("\n".join(out))
