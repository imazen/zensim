#!/usr/bin/env python3
"""geometry lane Stage-2 — standalone gate fit on geometry-variant caches.

Per cell: paired-seed TRAIN-fit subset (the same row subset feeds every
cell at a seed), the gate arm only (`fit_forms.fit_gate` — grid-scanned κ
on the shared GRID_C0 axis + golden refine + head fit; the curve arm's
8-start Adam refinement is NOT run), then dev-leg scoring on that cell's
own caches. Baseline = the joint-core production caches (bin121 + local +
n5 — records are constants-independent, so the canonical cache IS the
baseline cell's cache).

Usage:
  fit_geometry.py <cells.json> <fit-pairs.tsv> <outdir>
      [--seeds 9201,9207,9211] [--rows 1024]

cells.json:
  {"<cell>": {"fit":  {"ycbcr_y": bin, "ycbcr_cb": bin, "ycbcr_cr": bin},
              "dev":  {"<leg>": {"ycbcr_y": bin, ...}, ...}}}

Every dev leg carries a pairs TSV at <leg>.pairs.tsv next to the cells
file's "pairs" map: {"<leg>": "<pairs.tsv>"} — y = human_score*100 (the
target_0_100 convention; verified identical to the verdict lane's
eval_meta/*_y.json on kadid135/codec_dev/safesyn_sub).
"""
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '/mnt/v/output/zensim/dvifm-screen2d-2026-09-19/tools')
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import fit_standalone as fs
import fit_forms as ff
from fit_core import RowView, PlaneCacheF16

PLANES = ("ycbcr_y", "ycbcr_cb", "ycbcr_cr")
XYB = ("xyb_x", "xyb_y", "xyb_b")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def y_of(pairs_tsv):
    """target_0_100 convention: joint-core train legs already carry
    human_score on 0..100 (signed rows included); the dev legs carry
    0..1 — scale only the latter."""
    rows = list(csv.DictReader(open(pairs_tsv), delimiter="\t"))
    y = np.asarray([float(r["human_score"]) for r in rows])
    if float(np.max(np.abs(y))) <= 1.5:
        y = y * 100.0
    return y


def main():
    args = sys.argv[1:]
    cells_path, pairs_tsv, outdir = args[0], args[1], args[2]
    seeds = [9201, 9207, 9211]
    n_rows = 1024
    i = 3
    while i < len(args):
        if args[i] == "--seeds":
            seeds = [int(x) for x in args[i + 1].split(",")]
            i += 2
        elif args[i] == "--rows":
            n_rows = int(args[i + 1])
            i += 2
        else:
            raise SystemExit(f"unknown arg {args[i]}")
    cells = json.loads(Path(cells_path).read_text())
    Path(outdir).mkdir(parents=True, exist_ok=True)
    y_all = y_of(pairs_tsv)
    legs = next(v for k, v in cells.items() if k != "__pairs__").get("dev", {})
    leg_y = {}
    leg_pairs = cells.get("__pairs__", {})
    for leg in legs:
        leg_y[leg] = y_of(leg_pairs[leg])
    results = {"seeds": seeds, "n_rows": n_rows, "fit_pairs": pairs_tsv,
               "cells_file": cells_path, "cells": {}}
    rpath = Path(outdir, "geometry_results.json")
    for seed in seeds:
        rng = np.random.default_rng(seed)
        sel = sorted(rng.choice(len(y_all), size=n_rows,
                                replace=False).tolist())
        y = y_all[sel]
        for cell, spec in cells.items():
            if cell == "__pairs__":
                continue
            planes = tuple(spec.get("planes", PLANES))
            # Level count = nonempty index levels — the zensim-pooling
            # arm is 4 box scales (production NUM_SCALES) with the 5th
            # index slot zero-padded for schema shape; everything else
            # the canonical 5. fs.LEVELS is read dynamically throughout
            # the stack, so patching it scopes the fit correctly.
            first_idx = fs.load_index(spec["fit"][planes[0]])[0]
            rec_counts = first_idx["level_records"]
            fs.LEVELS = 1 + max(l for l, r in enumerate(rec_counts) if r > 0)
            log(f"seed {seed} cell {cell}: gate fit on {planes} "
                f"({fs.LEVELS} levels)")
            caches = {p: RowView(spec["fit"][p], sel) for p in planes}
            t0 = time.time()
            art = ff.fit_gate(planes, caches, y,
                              ff.gate_init_cells(planes, caches))
            art["wall_s"] = time.time() - t0
            rec = {"fit": {k: v for k, v in art["fit"].items()
                           if k != "history"},
                   "wall_s": art["wall_s"],
                   "cells": [{"mode": c["mode"], "kappa": c.get("kappa")}
                             for p in planes for c in art["cells"][p]],
                   "legs": {}}
            Path(outdir, f"gate_{cell}_s{seed}.json").write_text(
                json.dumps({**ff.arm_to_json(art, planes),
                            "provenance": {"seed": seed, "rows": n_rows,
                                           "cell": cell,
                                           "planes": list(planes),
                                           "fit_pairs": pairs_tsv,
                                           "fit_caches": spec["fit"],
                                           "sel": sel}},
                           indent=1) + "\n")
            for leg, bins in spec.get("dev", {}).items():
                t_caches = {p: PlaneCacheF16(bins[p]) for p in planes}
                ty = leg_y[leg]
                n_cache = next(iter(t_caches.values())).n_rows
                assert n_cache == len(ty), (leg, n_cache, len(ty))
                E, yhat = ff.score_arm(art, planes, t_caches)
                m = ff.metrics_of(E, ty, art["params"].map_abl)
                np.savez(Path(outdir, f"scores_{leg}_{cell}_s{seed}.npz"),
                         E=E, yhat=yhat, y=ty)
                rec["legs"][leg] = m
            results["cells"].setdefault(cell, {})[str(seed)] = rec
            rpath.write_text(json.dumps(results, indent=1) + "\n")
            log(f"seed {seed} cell {cell} DONE "
                f"({art['wall_s']:.0f}s fit)")
    log("ALL DONE")


if __name__ == "__main__":
    main()
