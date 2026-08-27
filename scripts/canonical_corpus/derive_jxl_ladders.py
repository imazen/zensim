#!/usr/bin/env python3
"""S4+C2 wave: per-(variant, cell) jxl q->zensim ladders from the canonical
zenjxl_lossy picker parquets. Emits the regressor TARGET table.

Honesty notes baked into the output (do not silently paper over):
- The source grid is 9 q-points {5,25,30,40,50,60,70,80,90} — NOT dense, and
  q>90 does not exist. Every t-crossing is linear interp between bracketing
  knots; a target above the ladder max is flagged `above`, NEVER extrapolated.
- Raw knots (q[], score[], bytes[]) ride along as list columns — the fit can
  re-derive any functional (incl. distance-space elasticity) without re-reading
  the 350 MB canonicals.
- Holdout is NOT enforced here: corpus9 / dial-39 exclusion happens at fit,
  keyed on the carried origin_id / content_source / content_image_sha.
- slope_dscore_dlogq_at{t} is a NEUTRAL local slope (bracketing finite diff in
  log-q); the controller-exponent interpretation (jxl §5.1) is the fit's job.
"""
import hashlib
import json
import math
import os
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

SRC = "/mnt/v/output/canonical-picker-2026-06-27/zenjxl_lossy"
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/v/zen/zensim-training/s4c2-2026-08-27"
TARGETS = (70.0, 80.0, 88.0)
COLS = ["split", "origin_id", "variant_name", "cell", "q", "score_zensim",
        "encoded_bytes", "width", "height", "content_source",
        "content_image_sha", "content_class", "size_class"]

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(1 << 20):
            h.update(c)
    return h.hexdigest()

def cross(qs, ss, t):
    """First upward crossing of t. -> (q_seed, slope_dscore_dlogq, flag)"""
    if t <= ss[0]:
        return None, None, "below"
    if t > max(ss):
        return None, None, "above"
    hits = []
    for i in range(len(ss) - 1):
        if ss[i] < t <= ss[i + 1]:
            hits.append(i)
    if not hits:
        return None, None, "nonmono_no_bracket"  # t inside range but only downward crossings
    i = hits[0]
    frac = (t - ss[i]) / (ss[i + 1] - ss[i])
    q_seed = qs[i] + frac * (qs[i + 1] - qs[i])
    slope = (ss[i + 1] - ss[i]) / (math.log(qs[i + 1]) - math.log(qs[i]))
    return q_seed, slope, ("ok" if len(hits) == 1 else "multi_crossing")

groups = {}
in_shas, in_rows = {}, {}
for split in ("train", "validate", "test"):
    p = f"{SRC}/{split}.parquet"
    in_shas[split] = sha256(p)
    t = pq.read_table(p, columns=COLS)
    in_rows[split] = t.num_rows
    d = t.to_pydict()
    for j in range(t.num_rows):
        k = (d["split"][j], d["origin_id"][j], d["variant_name"][j], d["cell"][j])
        g = groups.setdefault(k, {"pts": [], "meta": None})
        g["pts"].append((float(d["q"][j]), float(d["score_zensim"][j]), int(d["encoded_bytes"][j])))
        if g["meta"] is None:
            g["meta"] = {c: d[c][j] for c in ("width", "height", "content_source",
                                              "content_image_sha", "content_class", "size_class")}

out = {c: [] for c in ["split", "origin_id", "variant_name", "cell", "n_points",
                       "n_inversions", "max_inversion", "q_knots", "score_knots",
                       "bytes_knots", "width", "height", "content_source",
                       "content_image_sha", "content_class", "size_class"]}
for t in TARGETS:
    for pre in ("q_seed_t", "slope_dscore_dlogq_t", "flag_t"):
        out[f"{pre}{int(t)}"] = []

dupes = 0
for (split, origin, variant, cell), g in groups.items():
    pts = sorted(g["pts"])
    qs = [p[0] for p in pts]
    if len(set(qs)) != len(qs):  # duplicate q in a group — should not happen
        dupes += 1
        continue
    ss = [p[1] for p in pts]
    bs = [p[2] for p in pts]
    inv = [ss[i] - ss[i + 1] for i in range(len(ss) - 1) if ss[i + 1] < ss[i] - 1e-9]
    out["split"].append(split); out["origin_id"].append(origin)
    out["variant_name"].append(variant); out["cell"].append(cell)
    out["n_points"].append(len(qs)); out["n_inversions"].append(len(inv))
    out["max_inversion"].append(max(inv) if inv else 0.0)
    out["q_knots"].append(qs); out["score_knots"].append(ss); out["bytes_knots"].append(bs)
    for c, v in g["meta"].items():
        out[c].append(v)
    for t in TARGETS:
        q_seed, slope, flag = cross(qs, ss, t)
        out[f"q_seed_t{int(t)}"].append(q_seed)
        out[f"slope_dscore_dlogq_t{int(t)}"].append(slope)
        out[f"flag_t{int(t)}"].append(flag)

os.makedirs(OUT_DIR, exist_ok=True)
table = pa.table(out)
outp = f"{OUT_DIR}/jxl_ladders_9pt.parquet"
pq.write_table(table, outp, compression="zstd")

def count(col):
    from collections import Counter
    return dict(Counter(out[col]))

head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
manifest = {
    "what": "S4+C2 regressor target table: per-(split,origin,variant,cell) jxl q->zensim ladders",
    "build_commit": head,
    "generator": "scripts/canonical_corpus/derive_jxl_ladders.py",
    "inputs": {s: {"path": f"{SRC}/{s}.parquet", "sha256": in_shas[s], "rows": in_rows[s]}
               for s in in_shas},
    "rows": table.num_rows,
    "q_grid": [5.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
    "density_note": "9-point grid, no q>90; pre-registration's 'q-dense' was wrong; "
                    "t=88 crossings interpolate the q80-q90 gap; 'above' = unreachable, never extrapolated",
    "holdout_note": "corpus9/dial-39 exclusion enforced at FIT (id + dHash), not here",
    "dupes_skipped": dupes,
    "flags": {f"t{int(t)}": count(f"flag_t{int(t)}") for t in TARGETS},
    "split_rows": count("split"),
}
json.dump(manifest, open(f"{OUT_DIR}/_MANIFEST.json", "w"), indent=1)
print(json.dumps({k: manifest[k] for k in ("rows", "flags", "split_rows", "dupes_skipped")}, indent=1))
print("->", outp)
