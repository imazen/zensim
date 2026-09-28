#!/usr/bin/env python3
"""Score a FROZEN X2 arm artefact on eval legs -> metrics.py score CSVs.

  score_dvifm.py <arm-json> <tag> <tasks-json> <outdir>
    [--emit-ref-grouped]

<arm-json>: gate_s*.json / prior_s*.json (arm_to_json shape) OR a
  fit_standalone curve artefact (schema v2 with levels/level_weights/map).
<tasks-json>: like score_tasks_x1.json — name, per-plane bins, fmt,
  optional rows subset, y json ({y, ref}).
Emits <outdir>/<tag>__on__<leg>.csv (ref_path,dist_path,target,score,E)
plus <outdir>/<tag>__on__<leg>.npz (E, yhat, y) for downstream bootstrap.
Task entries may carry "pairs" so the CSV rows keep ref/dist paths.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/lilith/work/zen/zensim/tools/joint_core')
sys.path.insert(0, '/mnt/v/output/zensim/dvifm-screen2d-2026-09-19/tools')
import numpy as np
import fit_forms as ff


def load_arm(path):
    j = json.loads(Path(path).read_text())
    if "cells" in j:                      # gate/prior arm artefact
        return ff.arm_from_json(j), j
    if "levels" in j:                     # curve fit_standalone artefact
        return ff.curve_arm(j), j
    raise SystemExit(f"unrecognised arm artefact {path}")


def main():
    arm_path, tag, tasks_path, outdir = sys.argv[1:5]
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    arm, raw = load_arm(arm_path)
    planes = tuple(raw["planes"])
    tasks = json.loads(Path(tasks_path).read_text())
    for task in tasks:
        name = task["name"]
        caches = ff.load_task_caches(task, planes)
        yj = json.loads(Path(task["y"]).read_text())
        y = np.asarray(yj["y"], dtype=np.float64)
        ref = yj["ref"]
        E, yhat = ff.score_arm(arm, planes, caches)
        assert len(E) == len(y) == len(ref), (len(E), len(y), len(ref))
        npz = outdir / f"{tag}__on__{name}.npz"
        np.savez(npz, E=E, yhat=yhat, y=y)
        csvp = outdir / f"{tag}__on__{name}.csv"
        pairs = task.get("pairs")
        dists = None
        if pairs:
            pr = list(csv.DictReader(open(pairs), delimiter="\t"))
            if "rows" in task:
                idx = (np.load(task["rows"])
                       if task["rows"].endswith(".npy")
                       else np.asarray(json.loads(
                           Path(task["rows"]).read_text())))
                pr = [pr[int(i)] for i in idx]
            dists = [r["dist_path"] for r in pr]
        with open(csvp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ref_path", "dist_path", "target", "score", "E"])
            for i in range(len(E)):
                w.writerow([ref[i], dists[i] if dists else "",
                            y[i], yhat[i], E[i]])
        m = ff.metrics_of(E, y, arm["params"].map_abl)
        print(f"{name}: n={len(E)} SROCC {m['srocc']:.4f} "
              f"KROCC {m['krocc']:.4f} PLCC {m['plcc']:.4f} "
              f"MSE {m['mse']:.3f}", flush=True)


if __name__ == "__main__":
    main()
