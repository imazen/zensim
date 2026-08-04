#!/usr/bin/env python3
"""KADID per-distortion-type SROCC breakdown from bake_contrib score dumps.

Registered in benchmarks/sota944_campaign_2026-08-03.md, "REGISTERED
APPENDIX — bake_contrib" §C.4. Row identity is the appendix's §C.0
verified arithmetic: every era's kadid parquet is in sorted-`dist_img`
order (0/10,125 mismatches vs /mnt/v/dataset/kadid10k/dmos.csv), so
`dist_type = (row_idx % 125) // 5 + 1` (81 refs × 25 types × 5 levels).
Stats come from the canonical owner (`panel --batch` via
scripts/lib/zen_stats.panel_batch) — nothing hand-rolled.

Usage:
  ZEN_PANEL_BIN=<panel> python3 scripts/contrib_kadid_types.py \
      --scores name=path/to/run.scores.tsv ... --out out.tsv

Each scores TSV is a bake_contrib --dump-scores file whose kadid corpus
rows cover the FULL 10,125-row parquet (row_idx = parquet row index).
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.lib.zen_stats import panel_batch  # noqa: E402

# KADID-10k published type map (Lin/Hosu/Saupe, QoMEX 2019, Table 1) and
# the §C.4 registered family grouping.
TYPE_NAMES = {
    1: "gaussian_blur", 2: "lens_blur", 3: "motion_blur",
    4: "color_diffusion", 5: "color_shift", 6: "color_quantization",
    7: "color_saturation_1", 8: "color_saturation_2",
    9: "jpeg2000", 10: "jpeg",
    11: "white_noise", 12: "white_noise_color_comp", 13: "impulse_noise",
    14: "multiplicative_noise", 15: "denoise",
    16: "brighten", 17: "darken", 18: "mean_shift",
    19: "jitter", 20: "non_eccentricity_patch", 21: "pixelate",
    22: "quantization", 23: "color_block", 24: "high_sharpen",
    25: "contrast_change",
}
GROUPS = {
    **{t: "blur" for t in (1, 2, 3)},
    **{t: "color" for t in (4, 5, 6, 7, 8)},
    **{t: "compression" for t in (9, 10)},
    **{t: "noise" for t in (11, 12, 13, 14, 15)},
    **{t: "brightness" for t in (16, 17, 18)},
    **{t: "spatial" for t in (19, 20, 21)},
    **{t: "sharp_contrast" for t in (22, 23, 24, 25)},
}
GROUP_ORDER = ["blur", "color", "compression", "noise", "brightness",
               "spatial", "sharp_contrast"]


def load_kadid_scores(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["corpus"] != "kadid":
                continue
            rows.append((int(r["row_idx"]), float(r["target"]),
                         float(r["baseline_score"])))
    rows.sort()
    if len(rows) != 10125:
        raise SystemExit(f"{path}: kadid rows {len(rows)} != 10125 "
                         "(per-type join needs the FULL corpus)")
    if [r[0] for r in rows] != list(range(10125)):
        raise SystemExit(f"{path}: kadid row_idx not the identity range")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", action="append", required=True,
                    metavar="name=path")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    models = []
    for spec in args.scores:
        name, path = spec.split("=", 1)
        models.append((name, load_kadid_scores(path)))

    jobs = []
    labels = []
    for name, rows in models:
        # per type
        for t in range(1, 26):
            idx = [i for i in range(10125) if (i % 125) // 5 + 1 == t]
            x = [rows[i][2] for i in idx]
            y = [rows[i][1] for i in idx]
            jobs.append((f"{name}|type{t}", x, y))
            labels.append((name, "type", t))
        # per group (pooled across the group's types)
        for g in GROUP_ORDER:
            idx = [i for i in range(10125)
                   if GROUPS[(i % 125) // 5 + 1] == g]
            x = [rows[i][2] for i in idx]
            y = [rows[i][1] for i in idx]
            jobs.append((f"{name}|group_{g}", x, y))
            labels.append((name, "group", g))
        # full-corpus cross-check vs the bake_contrib baseline SROCC
        jobs.append((f"{name}|all", [r[2] for r in rows],
                     [r[1] for r in rows]))
        labels.append((name, "all", "all"))

    res = panel_batch(jobs, stats="srocc")
    by = {}
    for (name, kind, key), r in zip(labels, res):
        by[(name, kind, key)] = abs(float(r["srocc"]))

    model_names = [m[0] for m in models]
    with open(args.out, "w") as f:
        f.write("kind\tkey\tname\tgroup\tn\t" +
                "\t".join(f"srocc_{m}" for m in model_names) + "\n")
        for t in range(1, 26):
            n = sum(1 for i in range(10125) if (i % 125) // 5 + 1 == t)
            f.write(f"type\t{t}\t{TYPE_NAMES[t]}\t{GROUPS[t]}\t{n}\t" +
                    "\t".join(f"{by[(m, 'type', t)]:.4f}"
                              for m in model_names) + "\n")
        for g in GROUP_ORDER:
            n = sum(1 for i in range(10125)
                    if GROUPS[(i % 125) // 5 + 1] == g)
            f.write(f"group\t{g}\t{g}\t{g}\t{n}\t" +
                    "\t".join(f"{by[(m, 'group', g)]:.4f}"
                              for m in model_names) + "\n")
        f.write("all\tall\tall\tall\t10125\t" +
                "\t".join(f"{by[(m, 'all', 'all')]:.4f}"
                          for m in model_names) + "\n")
    print(f"wrote {args.out}")
    for m in model_names:
        print(f"  {m}: all={by[(m, 'all', 'all')]:.4f}")


if __name__ == "__main__":
    main()
