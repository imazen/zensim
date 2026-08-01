#!/usr/bin/env python3
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/csfw-g6-loo-2026-07-29/harness/collect_loo956.py
# sha256(source): c2ee99ca11f216e9a7c455c6bd0198fb6e29b1b44059bdd0fcc3b2cbdcb1dcef
# build_commit:  7bfd511de78f85e8fcd618df15716ca56575bb60
# Protocol doc:  benchmarks/csfw_g6_loo_2026-07-29.md
# Everything below the marker line is BYTE-IDENTICAL to the source file
# (verify: strip through the marker, sha256 the rest). Do NOT extend this
# file — it is an archival record of the exact as-run analysis (it may call
# scipy directly; it predates the stats-rule batch migration and is kept
# verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
# Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
# FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
# in the artifact dir are the stored equivalents (see ../README.md).
# ===== byte-identical source below this line ================================
"""Collect the 956 LOO verdicts into the gate tables. Sign convention matches
e2_optimal_model_720_vs_372_2026-07-23.md: delta = drop − lin956 per corpus in
|SROCC|; POSITIVE = the family HURT (dropping it improved held-out ranking).
Σ is over the pinned e2 corpus subset intersected with the 8 available legs
(pin720/sigma_pin.json, copied from the bandvis wave, documents the pin).
Robustness cuts per the bandvis doc: Σ_pinned (6 legs), Σ_all8 (adds
AIC-3/AIC-4, outside the pin per e2), Σ_clean4 (CID22, CSIQ, LIVE, KonJND —
train-guards excluded)."""
import csv
import json
from pathlib import Path

import numpy as np

OUT = Path("/mnt/v/output/zensim/csfw-g6-loo-2026-07-29")
V = OUT / "verdicts956"
PIN = json.load(open(OUT / "pin720" / "sigma_pin.json"))
CLEAN4 = ["CID22", "CSIQ", "LIVE-R2", "KonJND-1k (full)"]

# fit-weight introspection: which append2/csfw slots did BVLS actually use?
FIT = np.load(OUT / "linear-probe-956" / "fits" / "twin_n956.npz")
W = FIT["w"].astype(np.float64)
blocks = {"v1basic f0..155": (0, 156), "foldslots f156..371": (156, 372),
          "v2-348 f372..719": (372, 720), "append204 f720..923": (720, 924),
          "append2 f924..943": (924, 944), "csfw f944..955": (944, 956)}
wlines = [f"BVLS active weights (|w|>1e-7) total {int((np.abs(W) > 1e-7).sum())}/956:"]
for nm, (a, b) in blocks.items():
    wlines.append(f"  {nm}: {int((np.abs(W[a:b]) > 1e-7).sum())}/{b-a}")
a2names = {0: "gain", 1: "loss", 2: "lumref", 3: "hl1", 4: "hl2"}
wlines.append("append2 slot weights (f: w):")
for s in range(4):
    wlines.append("  s%d " % s + "  ".join(
        f"{a2names[l]}={W[924 + s*5 + l]:+.5f}" for l in range(5)))
csnames = {0: "w_dmean", 1: "w_cgain", 2: "w_closs"}
wlines.append("csfw slot weights (f: w):")
for s in range(4):
    wlines.append("  s%d " % s + "  ".join(
        f"{csnames[l]}={W[944 + s*3 + l]:+.5f}" for l in range(3)))
WEIGHT_TXT = "\n".join(wlines)
print(WEIGHT_TXT)
print()


def sroccs(path: Path) -> dict[str, float]:
    d = json.load(open(path))
    return {c["display"]: float(c["srocc"]) for c in d["corpora"]}


full = sroccs(V / "lin956.json")
corpora = list(full)
sigma_set = [c for c in PIN["subset"] if c in corpora]
missing = [c for c in PIN["subset"] if c not in corpora]
clean_set = [c for c in CLEAN4 if c in corpora]

fams = sorted(p.stem[5:] for p in V.glob("drop_*.json"))
rows = []
for f in fams:
    d = sroccs(V / f"drop_{f}.json")
    delta = {c: d[c] - full[c] for c in corpora}
    rows.append((f, delta,
                 sum(delta[c] for c in sigma_set),
                 sum(delta[c] for c in corpora),
                 sum(delta[c] for c in clean_set)))

rows.sort(key=lambda r: -r[2])
lines = []
lines.append("lin956 full-model |SROCC| per corpus:")
lines.append("  " + "  ".join(f"{c}={full[c]:.4f}" for c in corpora))
lines.append("")
lines.append(f"Σ subset (pinned from e2, available legs): {', '.join(sigma_set)}")
if missing:
    lines.append(f"  (pinned-but-unavailable at 956: {', '.join(missing)})")
lines.append(f"Σ_all8 adds AIC-3/AIC-4 (outside the e2 pin); "
             f"Σ_clean4 = {', '.join(clean_set)} (train-guards excluded)")
lines.append("")
hdr = ("family".ljust(28) + "Σpin".rjust(9) + "Σall8".rjust(9) + "Σcln4".rjust(9)
       + "".join(c.rjust(9) for c in corpora))
lines.append(hdr)
for f, delta, s6, s8, s4 in rows:
    lines.append(f.ljust(28) + f"{s6:+.4f}".rjust(9) + f"{s8:+.4f}".rjust(9)
                 + f"{s4:+.4f}".rjust(9)
                 + "".join(f"{delta[c]:+.4f}".rjust(9) for c in corpora))
txt = WEIGHT_TXT + "\n\n" + "\n".join(lines)
print("\n".join(lines))
(OUT / "loo956_table.txt").write_text(txt + "\n")

with open(OUT / "loo956_deltas.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["family", "sigma_pinned_subset", "sigma_all8", "sigma_clean4"]
               + corpora)
    for f, delta, s6, s8, s4 in rows:
        w.writerow([f, f"{s6:+.5f}", f"{s8:+.5f}", f"{s4:+.5f}"]
                   + [f"{delta[c]:+.5f}" for c in corpora])
json.dump({"full_lin956": full, "sigma_subset": sigma_set,
           "missing_from_pin": missing, "clean4": clean_set,
           "deltas": {f: d for f, d, *_ in rows},
           "sigma": {f: s for f, _, s, _, _ in rows},
           "sigma_all8": {f: s for f, _, _, s, _ in rows},
           "sigma_clean4": {f: s for f, _, _, _, s in rows}},
          open(OUT / "loo956_deltas.json", "w"), indent=1)
print(f"\nwrote {OUT}/loo956_table.txt + loo956_deltas.csv + loo956_deltas.json")
