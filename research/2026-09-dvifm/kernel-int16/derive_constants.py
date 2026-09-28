#!/usr/bin/env python3
"""Derive screen constants from a block cache — TRAIN fit rows only.

Per level: C0 = 10th percentile of C~ (= min(C~_ref, C~_dist) per block,
the F2 convention, at g=1 with the edge discount); F2 centres = {10,30,
50,70,90}% quantiles of ln(min(C~,)+1e-6). Usage: derive_constants.py
<cache.bin> <out-spec.json> <band>
"""
import sys, json
import numpy as np
sys.path.insert(0, 'tools')
from pathlib import Path
from dvifm_cache import load_index, read_records, contrast_g, load_spec

bin_path, out_path, band = sys.argv[1], sys.argv[2], sys.argv[3]
# Optional 4th arg: a fitted spec whose per-level g/P/… are carried into
# the output (F2 centres re-derived "at the baked g" per the design).
base_spec = load_spec(sys.argv[4]) if len(sys.argv) > 4 else None
index = load_index(bin_path)
train = [e for e in index if e["row_index"] < 8000]
print(f"{len(train)} train rows of {len(index)}")

acc = [[] for _ in range(5)]          # per-level min-C~ arrays
for k, e in enumerate(train):
    recs = read_records(bin_path, e)
    for l in range(5):
        r = recs[l]
        if r.shape[0]:
            g_l = base_spec["levels"][l]["g"] if base_spec else 1.0
            cs = contrast_g(r, 0, g_l, True)
            cd = contrast_g(r, 1, g_l, True)
            acc[l].append(np.minimum(cs, cd))
    if (k + 1) % 1000 == 0:
        print(f"  {k+1}/{len(train)}", flush=True)

levels = []
report = {}
for l in range(5):
    c = np.concatenate(acc[l])
    ell = np.log(c + 1e-6)
    centres = [float(np.quantile(ell, q)) for q in (0.10, 0.30, 0.50, 0.70, 0.90)]
    if base_spec:
        lv = dict(base_spec["levels"][l])
        lv["f2_centers"] = centres
        lv["band"] = band
        if not np.isfinite(lv.get("c_hi", np.inf)):
            lv["c_hi"] = None
        levels.append(lv)
        c0 = lv["c0"]
    else:
        c0 = float(np.quantile(c, 0.10))
        levels.append({"g": 1.0, "p": 1.0, "c0": c0, "beta": 0.65, "sharp": 4.0,
                       "c_hi": None, "f2_centers": centres, "band": band,
                       "edge": True})
    report[l] = {"n_blocks": int(c.size), "c0_p10": float(np.quantile(c, 0.10)),
                 "c0_used": c0, "f2_centres": centres,
                 "g_used": levels[-1]["g"],
                 "ctilde_min": float(c.min()), "ctilde_med": float(np.median(c)),
                 "ctilde_p90": float(np.quantile(c, 0.9))}
    print(f"L{l}: n={c.size} c0={c0:.6f} centres={[round(x,3) for x in centres]}")

prov = {"cache": bin_path, "train_row_rule": "row_index < 8000",
        "band": band, "per_level": report}
if base_spec:
    prov["base_spec"] = sys.argv[4]
    prov["g/p/c0/beta/sharp"] = "carried from base_spec (fitted)"
    prov["f2_centres_rule"] = "re-derived at the fitted per-level g"
else:
    prov.update({"g": 1.0, "p": 1.0, "beta": 0.65, "sharp": 4.0, "edge": True})
spec = {"schema": "dvifm-spec-v1",
        "note": f"screen constants derived from {bin_path} TRAIN rows (row_index<8000); "
                "C0 = p10 of min-C~ per level; F2 centres = {10,30,50,70,90}% of ln(min C~+1e-6)",
        "provenance": prov,
        "levels": levels}
Path(out_path).write_text(json.dumps(spec, indent=1) + "\n")
print("wrote", out_path)
