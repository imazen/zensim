#!/usr/bin/env python3
"""S4+C2 frozen proxy-gap gate (plan doc "S4+C2 DESIGN RULING 2026-08-27").

Measures, on the 39-image dial-grid jxl DISTANCE ladders (decoded pixels,
no parquet regime hazard), the gap between zensimA-ladder targets and
C-bake-ladder targets:
  (i)  median |Δ seed position| ≤ 2 ladder grid-steps at each t in {70,80,88}
  (ii) seed-bracket slope-sign agreement ≥ 90% of measurable cells
Photo/screen split: PENDING a registered class mapping for the 39 refs
(reported per-image here so the split is a relabel, not a rerun).

Scoring: canonical forward `predict_features_with_bake` over row-aligned
feature CSVs from v2_ab_extract (v1=372 for A, foldapp2=944 for C).
Seed = first downward crossing of t in increasing distance; linear interp;
grid-step position = fractional index in the ladder's own knot grid.
"""
import csv
import json
import statistics
import struct
import subprocess
import sys

O = "/mnt/v/zen/zensim-training/s4c2-2026-08-27"
D = "/mnt/v/output/zensim/dial-grid-pixels-2026-07-27"
FWD = "target/release/predict_features_with_bake"
BAKE_A = "zensim/weights/v47_strict_qat_native_2026-05-27.bin"
BAKE_C = "zensim/weights/c_sdr_mlp944_corrmix_2026-08-05.bin"
TARGETS = (70.0, 80.0, 88.0)

def load_feats(path, nf):
    rows, order = [], []
    with open(path) as f:
        r = csv.reader(f)
        hdr = next(r)
        assert hdr[2] == "f0" and len(hdr) == nf + 2, (path, len(hdr))
        for row in r:
            order.append(row[0])
            rows.append([float(x) for x in row[2:]])
    return order, rows

def forward(bake, rows, nf):
    blob = struct.pack("<II", nf, len(rows))
    for r in rows:
        blob += struct.pack(f"<{nf}f", *r)
    p = f"{O}/_fwd_{nf}.blob"
    open(p, "wb").write(blob)
    out = subprocess.run([FWD, "--bake", bake, "--features-file", p],
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"forward failed ({bake}): {out.stderr[-400:]}")
    scores = [float(x) for x in out.stdout.split()]
    assert len(scores) == len(rows)
    return scores

# --- row-aligned drv + distance from pairs manifest ---
drv = list(csv.DictReader(open(f"{O}/drv_jxl_mnt.tsv"), delimiter="\t"))
dist_of = {}
for row in csv.DictReader(open(f"{D}/pairs_jxl.tsv"), delimiter="\t"):
    base = row["dist_path"].rsplit("/", 1)[1]
    k = row["knob_tuple_json"]
    v = json.loads(k)
    if isinstance(v, str):  # double-encoded variant
        v = json.loads(v)
    dist_of[base] = v["distance"]
o372, f372 = load_feats(f"{O}/probe_jxl_372.csv", 372)
o944, f944 = load_feats(f"{O}/probe_jxl_944.csv", 944)
assert len(drv) == len(f372) == len(f944)

sA = forward(BAKE_A, f372, 372)
sC = forward(BAKE_C, f944, 944)

ladders = {}
for i, row in enumerate(drv):
    ref = row["ref_path"].rsplit("/", 1)[1]
    base = row["dist_path"].rsplit("/", 1)[1]
    ladders.setdefault(ref, []).append((dist_of[base], sA[i], sC[i]))

def seed(pts, t, col):
    """First downward crossing of t as d increases -> (d_seed, frac_idx, slope_sign) or flag."""
    ss = [p[col] for p in pts]
    if ss[0] < t:
        return None, None, None, "below_at_d0"
    for i in range(len(ss) - 1):
        if ss[i] >= t > ss[i + 1]:
            fr = (ss[i] - t) / (ss[i] - ss[i + 1])
            d = pts[i][0] + fr * (pts[i + 1][0] - pts[i][0])
            return d, i + fr, -1 if ss[i + 1] < ss[i] else 1, "ok"
    return None, None, None, "never_below"

per_image, summary = [], {}
for t in TARGETS:
    deltas, signs_agree, n_flag = [], 0, {"ok": 0, "below_at_d0": 0, "never_below": 0, "one_sided": 0}
    for ref, pts in sorted(ladders.items()):
        pts = sorted(pts)
        dA, iA, sgA, flA = seed(pts, t, 1)
        dC, iC, sgC, flC = seed(pts, t, 2)
        if flA == "ok" and flC == "ok":
            n_flag["ok"] += 1
            deltas.append(abs(iA - iC))
            signs_agree += (sgA == sgC)
            per_image.append((int(t), ref, round(dA, 4), round(dC, 4), round(abs(iA - iC), 3), flA, flC))
        else:
            key = flA if flA != "ok" else flC
            n_flag[key if key in n_flag else "one_sided"] = n_flag.get(key, 0) + 1
            per_image.append((int(t), ref, dA and round(dA, 4), dC and round(dC, 4), None, flA, flC))
    med = statistics.median(deltas) if deltas else None
    summary[f"t{int(t)}"] = {
        "n_measurable": len(deltas), "flags": n_flag,
        "median_abs_delta_steps": round(med, 3) if med is not None else None,
        "p90_abs_delta_steps": round(sorted(deltas)[int(0.9 * len(deltas))], 3) if deltas else None,
        "slope_sign_agreement": round(signs_agree / len(deltas), 4) if deltas else None,
        "gate_i_pass": (med is not None and med <= 2.0),
        "gate_ii_pass": (len(deltas) > 0 and signs_agree / len(deltas) >= 0.90),
    }

result = {
    "bakes": {"A": BAKE_A, "C": BAKE_C},
    "n_refs": len(ladders), "n_cells": len(drv),
    "summary": summary,
    "photo_screen_split": "PENDING registered 39-image class mapping (relabel of the per-image table, not a rerun)",
    "verdict": "PASS" if all(v["gate_i_pass"] and v["gate_ii_pass"] for v in summary.values()) else "FAIL",
}
json.dump(result, open(f"{O}/proxy_gap_gate.json", "w"), indent=1)
with open(f"{O}/proxy_gap_per_image.tsv", "w") as f:
    f.write("t\tref\td_seed_A\td_seed_C\tabs_delta_steps\tflag_A\tflag_C\n")
    for r in per_image:
        f.write("\t".join("" if x is None else str(x) for x in r) + "\n")
print(json.dumps(result, indent=1)[:1800])
