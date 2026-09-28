#!/usr/bin/env python3
"""Score all X2 arm artefacts on the X2 eval tasks -> npz + leg metrics
merged into x2_results.json.

  score_x2.py <x2_outdir> <tasks.json>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/lilith/work/zen/zensim/tools/joint_core')
sys.path.insert(0, '/mnt/v/output/zensim/dvifm-screen2d-2026-09-19/tools')
import numpy as np
import fit_forms as ff


def load_arm(path):
    j = json.loads(Path(path).read_text())
    if "cells" in j:
        return ff.arm_from_json(j), tuple(j["planes"])
    return ff.curve_arm(j), tuple(j["planes"])


def main():
    outdir = Path(sys.argv[1])
    tasks = json.loads(Path(sys.argv[2]).read_text())
    rpath = outdir / "x2_results.json"
    res = json.loads(rpath.read_text())
    seeds = [str(s) for s in res["seeds"]]
    for task in tasks:
        name = task["name"]
        yj = json.loads(Path(task["y"]).read_text())
        ty = np.asarray(yj["y"], dtype=np.float64)
        (outdir / "scores" / name).mkdir(parents=True, exist_ok=True)
        for arm_name in ("curve", "gate", "prior"):
            for s in seeds:
                art_path = outdir / f"{arm_name}_s{s}.json"
                if not art_path.exists():
                    print(f"MISS {art_path}")
                    continue
                arm, pl = load_arm(art_path)
                caches = ff.load_task_caches(task, pl)
                E, yhat = ff.score_arm(arm, pl, caches)
                assert len(E) == len(ty), (name, arm_name, s,
                                           len(E), len(ty))
                m = ff.metrics_of(E, ty, arm["params"].map_abl)
                np.savez(outdir / "scores" / name /
                         f"{arm_name}_s{s}.npz", E=E, yhat=yhat, y=ty)
                rec = res["arms"].setdefault(arm_name, {}).setdefault(s, {})
                rec[name] = m
                print(f"{name} {arm_name} s{s}: SROCC {m['srocc']:.4f} "
                      f"KROCC {m['krocc']:.4f} PLCC {m['plcc']:.4f}",
                      flush=True)
        rpath.write_text(json.dumps(res, indent=1) + "\n")
    print("SCORE X2 DONE")


if __name__ == "__main__":
    main()
