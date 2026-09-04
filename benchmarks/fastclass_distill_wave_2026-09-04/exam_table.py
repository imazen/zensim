#!/usr/bin/env python3
"""fastclass wave — assemble the per-arm exam table by READING fullevals.

Computes NO statistic. Every number is read out of the `bake_verdict
--full-json` the owner already wrote; the only arithmetic is the mean and
min/max ACROSS SEEDS of an arm, which is reported as such (k, mean, range),
never as a single "the" value. The W1/W2 CIs come from the exam's own
`paired_perref_boot.py`, not from here.

Usage: exam_table.py [--dir DIR] [--arms "C0 D1 ..."] [--seeds "4004 ..."]
"""
import json, os, sys, argparse, statistics

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/mnt/v/output/zensim/fastclass-2026-09-04")
ap.add_argument("--arms", default="C0 D1 D2 D3 D4 E1 F1")
ap.add_argument("--seeds", default="4004 4005 4006")
ap.add_argument("--tsv", action="store_true")
a = ap.parse_args()

# ssim2's own measured values on this exam's rulers (the opponent row; read
# from the exam doc §3.1 / the peer tables, not recomputed here).
# ssim2's own measured pooled |SROCC| on each ruler, READ from the exam doc
# §3.1's `ssim2` column (current-root values, the ones its paired CIs were
# computed on). This column is ORIENTATION ONLY — the authoritative signed
# difference and its CI come from `paired_perref_boot.py`, which recomputes
# ssim2 from the peer table on exactly the rows being compared.
SSIM2 = {"cid22": 0.8894, "konjnd": 0.5272, "csiq": 0.9047,
         "live": 0.9599, "aic3": 0.7970, "aic4": 0.9127}
AXES = ["cid22", "konjnd", "csiq", "live", "aic3", "aic4"]

def q85(fe):
    """The q>=85 zone cell — the near-lossless ladder row W3 weights."""
    for c in fe.get("dial", {}).get("zones", {}).get("cells", []):
        if c.get("split") == "all" and "85" in str(c.get("zone")) and ">" in str(c.get("zone")):
            return c
    return {}

rows = {}
for arm in a.arms.split():
    for s in a.seeds.split():
        p = os.path.join(a.dir, f"{arm}_s{s}.fulleval.json")
        if not os.path.exists(p):
            continue
        fe = json.load(open(p))
        r = fe["rank"]; z = q85(fe)
        rows.setdefault(arm, []).append({
            "seed": s,
            **{k: r[k]["srocc"] for k in AXES},
            "nonphoto": r["nonphoto"]["srocc"], "imazen26": r["imazen26"]["srocc"],
            "composite": fe["composite"],
            "mono": fe["dial"]["mono_pct"],
            "p5": fe["dial"]["p5"], "p95": fe["dial"]["p95"],
            "q85_inv": z.get("frac_ladders_with_inv"),
            "q85_bkw": z.get("frac_ladders_ends_backwards"),
            "m3a": (fe.get("m3_coherence") or {}).get("m3a"),
            "sha": fe.get("bake_sha256", "")[:12],
        })

cols = AXES + ["nonphoto", "imazen26", "composite", "mono", "q85_inv", "q85_bkw"]
sep = "\t" if a.tsv else "  "

def fmt(v, w=8):
    if v is None: return ("—").rjust(w)
    return f"{v:.4f}".rjust(w)

print("# PER-SEED (every cell read from the arm's own fulleval.json)")
hdr = ["arm_seed"] + cols
print(sep.join(h.rjust(9) for h in hdr))
for arm, rs in rows.items():
    for r in rs:
        print(sep.join([f"{arm}_s{r['seed']}".rjust(9)] + [fmt(r[c], 9) for c in cols]))

print("\n# PER-ARM, k seeds: mean [min..max]  (spread is the seed spread, not a CI)")
print(sep.join(["arm".rjust(5), "k".rjust(2)] +
               [c.rjust(22) for c in cols]))
summary = {}
for arm, rs in rows.items():
    cells = []
    summary[arm] = {}
    for c in cols:
        vs = [r[c] for r in rs if r[c] is not None]
        if not vs:
            cells.append("—".rjust(22)); continue
        m = statistics.fmean(vs)
        summary[arm][c] = {"mean": m, "min": min(vs), "max": max(vs), "k": len(vs)}
        cells.append(f"{m:.4f} [{min(vs):.4f}..{max(vs):.4f}]".rjust(22))
    print(sep.join([arm.rjust(5), str(len(rs)).rjust(2)] + cells))

if "C0" in summary:
    print("\n# Δ vs C0 (control = A4b recipe verbatim), mean over seeds")
    print(sep.join(["arm".rjust(5)] + [c.rjust(9) for c in cols]))
    for arm in summary:
        if arm == "C0": continue
        d = []
        for c in cols:
            if c in summary[arm] and c in summary["C0"]:
                d.append(f"{summary[arm][c]['mean'] - summary['C0'][c]['mean']:+.4f}".rjust(9))
            else:
                d.append("—".rjust(9))
        print(sep.join([arm.rjust(5)] + d))

print("\n# vs ssim2 (mean over seeds − ssim2's own measured value on the same ruler)")
print(sep.join(["arm".rjust(5)] + [c.rjust(9) for c in AXES]))
for arm in summary:
    print(sep.join([arm.rjust(5)] +
        [f"{summary[arm][c]['mean'] - SSIM2[c]:+.4f}".rjust(9) if c in summary[arm]
         else "—".rjust(9) for c in AXES]))
json.dump(summary, open(os.path.join(a.dir, "exam_summary.json"), "w"), indent=1)
print(f"\n# wrote {os.path.join(a.dir, 'exam_summary.json')}")
