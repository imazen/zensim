#!/usr/bin/env python3
"""Score a strategy-ablation wave into one comparison table.

Runs bake_verdict on every ab_<variant>_s<seed> bin found in --bins-dir,
parses the aggregate panel rows, and prints per-variant CID22/KonJND SROCC
(mean over seeds ± spread) with deltas vs ab_base. Optionally cross-checks
box-vs-fleet bins for byte equality (same manifest, two machines — the free
determinism check).

  usage: ablation_scoreboard.py --bins-dir DIR [--fleet-dir DIR2] [--out MD]
"""
import argparse, glob, hashlib, os, re, subprocess, sys
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--bins-dir", required=True)
ap.add_argument("--fleet-dir")
ap.add_argument("--verdict-bin", default="./target/release/bake_verdict")
ap.add_argument("--out")
a = ap.parse_args()

def score(bake):
    out = f"/tmp/{os.path.basename(bake)}.verdict.md"
    subprocess.run([a.verdict_bin, "--bake", bake, "--output", out],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    r = {}
    if os.path.exists(out):
        for line in open(out):
            m = re.match(r"\| (CID22|KonJND[^|]*|AIC-3[^|]*) \|[^|]*\| ([0-9.+-]+) \|", line)
            if m:
                key = m.group(1).split()[0].split("-")[0].lower()
                r[key] = float(m.group(2))
    return r

cells = defaultdict(dict)   # variant -> seed -> scores
shas = {}
for b in sorted(glob.glob(os.path.join(a.bins_dir, "ab_*.bin"))):
    m = re.match(r"ab_(.+)_s(\d+)\.bin", os.path.basename(b))
    if not m:
        continue
    variant, seed = m.group(1), m.group(2)
    cells[variant][seed] = score(b)
    shas[(variant, seed)] = hashlib.sha256(open(b, "rb").read()).hexdigest()

det = []
if a.fleet_dir:
    for (variant, seed), h in shas.items():
        fb = glob.glob(os.path.join(a.fleet_dir, f"ab_{variant}_s{seed}", "*.bin")) + \
             glob.glob(os.path.join(a.fleet_dir, f"ab_{variant}_s{seed}.bin"))
        if fb:
            fh = hashlib.sha256(open(fb[0], "rb").read()).hexdigest()
            det.append((variant, seed, "IDENTICAL" if fh == h else "DIVERGENT"))

lines = []
lines.append(f"| variant | CID22 (seeds) | mean | ΔCID22 vs base | KonJND mean | ΔKonJND |")
lines.append("|---|---|---|---|---|---|")
base_cid = base_kon = None
rows = []
for v in sorted(cells, key=lambda x: (x != "base", x)):
    cids = [cells[v][s].get("cid22") for s in sorted(cells[v]) if cells[v][s].get("cid22") is not None]
    kons = [cells[v][s].get("konjnd") for s in sorted(cells[v]) if cells[v][s].get("konjnd") is not None]
    if not cids:
        rows.append((v, None, None, None))
        continue
    mc = sum(cids) / len(cids)
    mk = sum(kons) / len(kons) if kons else float("nan")
    if v == "base":
        base_cid, base_kon = mc, mk
    rows.append((v, cids, mc, mk))
for v, cids, mc, mk in rows:
    if cids is None:
        lines.append(f"| {v} | (no bins) | | | | |")
        continue
    seeds_str = "/".join(f"{c:.4f}" for c in cids)
    dc = f"{mc - base_cid:+.4f}" if base_cid is not None else "-"
    dk = f"{mk - base_kon:+.4f}" if base_kon is not None and mk == mk else "-"
    lines.append(f"| {v} | {seeds_str} | {mc:.4f} | {dc} | {mk:.4f} | {dk} |")
if det:
    lines.append("")
    lines.append("Cross-machine determinism (box vs fleet):")
    for v, s, verdict in sorted(det):
        lines.append(f"- {v}_s{s}: {verdict}")
text = "\n".join(lines)
print(text)
if a.out:
    open(a.out, "w").write(text + "\n")
