#!/usr/bin/env python3
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/bandvis-loo-2026-07-28/harness/collect_loo944.py
# sha256(source): ca6c95011d45032eeccdd469fc0b3fa3d0e8359b49ff4c44cc6b988f5842369a
# build_commit:  b1d4bc257e57f7c3215ec8a237e9f87cdad8e35f
# Protocol doc:  benchmarks/bandvis_loo_944_2026-07-28.md
# Everything below the marker line is BYTE-IDENTICAL to the source file
# (verify: strip through the marker, sha256 the rest). Do NOT extend this
# file — it is an archival record of the exact as-run analysis (it may call
# scipy directly; it predates the stats-rule batch migration and is kept
# verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
# Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
# FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
# in the artifact dir are the stored equivalents (see ../README.md).
# ===== byte-identical source below this line ================================
"""Collect the 944 LOO verdicts into the gate tables. Sign convention matches
e2_optimal_model_720_vs_372_2026-07-23.md: delta = drop − lin944 per corpus in
|SROCC|; POSITIVE = the family HURT (dropping it improved held-out ranking).
Σ is over the pinned e2 corpus subset intersected with the 8 available legs
(pin720/sigma_pin.json documents the pin)."""
import json
from pathlib import Path

import numpy as np

OUT = Path("/mnt/v/output/zensim/bandvis-loo-2026-07-28")
V = OUT / "verdicts944"
PIN = json.load(open(OUT / "pin720" / "sigma_pin.json"))

# fit-weight introspection: which append2 slots did BVLS actually use?
FIT = np.load(OUT / "linear-probe-944" / "fits" / "twin_n944.npz")
W = FIT["w"].astype(np.float64)
blocks = {"v1basic f0..155": (0, 156), "foldslots f156..371": (156, 372),
          "v2-348 f372..719": (372, 720), "append204 f720..923": (720, 924),
          "append2 f924..943": (924, 944)}
wlines = [f"BVLS active weights (|w|>1e-7) total {int((np.abs(W) > 1e-7).sum())}/944:"]
for nm, (a, b) in blocks.items():
    wlines.append(f"  {nm}: {int((np.abs(W[a:b]) > 1e-7).sum())}/{b-a}")
names = {0: "gain", 1: "loss", 2: "lumref", 3: "hl1", 4: "hl2"}
wlines.append("append2 slot weights (f: w):")
for s in range(4):
    row = "  s%d " % s + "  ".join(
        f"{names[l]}={W[924 + s*5 + l]:+.5f}" for l in range(5))
    wlines.append(row)
WEIGHT_TXT = "\n".join(wlines)
print(WEIGHT_TXT)
print()


def sroccs(path: Path) -> dict[str, float]:
    d = json.load(open(path))
    return {c["display"]: float(c["srocc"]) for c in d["corpora"]}


full = sroccs(V / "lin944.json")
corpora = list(full)
sigma_set = [c for c in PIN["subset"] if c in corpora]
missing = [c for c in PIN["subset"] if c not in corpora]

fams = sorted(p.stem[5:] for p in V.glob("drop_*.json"))
rows = []
for f in fams:
    d = sroccs(V / f"drop_{f}.json")
    delta = {c: d[c] - full[c] for c in corpora}
    rows.append((f, delta, sum(delta[c] for c in sigma_set)))

rows.sort(key=lambda r: -r[2])
lines = []
lines.append("lin944 full-model |SROCC| per corpus:")
lines.append("  " + "  ".join(f"{c}={full[c]:.4f}" for c in corpora))
lines.append("")
lines.append(f"Σ subset (pinned from e2, available legs): {', '.join(sigma_set)}")
if missing:
    lines.append(f"  (pinned-but-unavailable at 944: {', '.join(missing)})")
lines.append("")
hdr = "family".ljust(26) + "Σ".rjust(9) + "".join(c.rjust(9) for c in corpora)
lines.append(hdr)
for f, delta, s in rows:
    lines.append(f.ljust(26) + f"{s:+.4f}".rjust(9)
                 + "".join(f"{delta[c]:+.4f}".rjust(9) for c in corpora))
txt = WEIGHT_TXT + "\n\n" + "\n".join(lines)
print("\n".join(lines))
(OUT / "loo944_table.txt").write_text(txt + "\n")

import csv
with open(OUT / "loo944_deltas.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["family", "sigma_pinned_subset"] + corpora)
    for f, delta, s in rows:
        w.writerow([f, f"{s:+.5f}"] + [f"{delta[c]:+.5f}" for c in corpora])
json.dump({"full_lin944": full, "sigma_subset": sigma_set,
           "missing_from_pin": missing,
           "deltas": {f: d for f, d, _ in rows},
           "sigma": {f: s for f, _, s in rows}},
          open(OUT / "loo944_deltas.json", "w"), indent=1)
print(f"\nwrote {OUT}/loo944_table.txt + loo944_deltas.csv + loo944_deltas.json")
