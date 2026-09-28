#!/usr/bin/env python3
"""X2 decision: compare the three constant forms on identical fit rows +
eval legs, emit the verdict JSON used to pick the frozen X1 form.

  x2_decide.py <x2_results.json> <out.json>

Composite per (arm,seed): mean SROCC over the X2 eval legs. Decision rule
from the lane: if the curve does not beat the gate by more than seed
noise, the gate wins (cheaper); prior preferred only if it ties the gate.
"""
import json
import sys
from pathlib import Path

import numpy as np

LEGS = ["kadid135", "konfig_val", "codec_dev", "human_dev", "cid22_dev",
        "safesyn_sub"]


def composite(rec, legs):
    vals = [rec[l]["srocc"] for l in legs if l in rec]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    res = json.loads(Path(sys.argv[1]).read_text())
    seeds = [str(s) for s in res["seeds"]]
    arms = res["arms"]
    out = {"seeds": seeds, "legs": LEGS, "arms": {}}
    for arm, per in arms.items():
        comp = [composite(per[s], LEGS) for s in seeds]
        fit_mse = [per[s]["fit"]["mse"] for s in seeds]
        fit_srocc = [per[s]["fit"]["srocc"] for s in seeds]
        wall = [per[s].get("wall_s") or 0.0 for s in seeds]
        leg_m = {}
        for l in LEGS:
            if all(l in per[s] for s in seeds):
                leg_m[l] = {
                    "srocc": [per[s][l]["srocc"] for s in seeds],
                    "krocc": [per[s][l]["krocc"] for s in seeds],
                    "plcc": [per[s][l]["plcc"] for s in seeds],
                    "mse": [per[s][l]["mse"] for s in seeds],
                    "srocc_mean": float(np.mean(
                        [per[s][l]["srocc"] for s in seeds])),
                    "srocc_std": float(np.std(
                        [per[s][l]["srocc"] for s in seeds])),
                }
        out["arms"][arm] = {
            "composite_srocc": comp,
            "composite_mean": float(np.mean(comp)),
            "composite_std": float(np.std(comp)),
            "fit_mse": fit_mse, "fit_srocc": fit_srocc,
            "wall_s": wall, "legs": leg_m,
        }
    # paired deltas
    d = {}
    for a in ("curve", "prior"):
        if a in arms and "gate" in arms:
            dif = [composite(arms[a][s], LEGS)
                   - composite(arms["gate"][s], LEGS)
                   for s in seeds]
            d[f"{a}-gate"] = {
                "delta_per_seed": dif,
                "delta_mean": float(np.mean(dif)),
                "delta_std": float(np.std(dif)),
            }
    out["deltas"] = d
    gate_c = out["arms"]["gate"]["composite_mean"]
    curve_c = out["arms"].get("curve", {}).get("composite_mean", -9)
    prior_c = out["arms"].get("prior", {}).get("composite_mean", -9)
    gate_noise = out["arms"]["gate"]["composite_std"]
    curve_edge = (out["deltas"].get("curve-gate", {})
                  .get("delta_mean", -9))
    # rule: curve must beat gate by > 2*seed-std to earn its params;
    # else cheaper wins; prior over gate only if it ties within noise.
    if curve_edge > 2.0 * gate_noise:
        winner = "curve"
    elif prior_c >= gate_c - gate_noise:
        winner = "prior"
    else:
        winner = "gate"
    out["decision"] = {
        "winner": winner,
        "rule": "curve wins only if delta_mean(curve-gate) > 2*std(gate "
                "composite); else the cheaper of gate/prior within one "
                "std of the gate composite wins; gate otherwise",
        "gate_composite": gate_c, "curve_composite": curve_c,
        "prior_composite": prior_c, "gate_std": gate_noise,
        "curve_minus_gate": curve_edge,
    }
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["decision"], indent=1))


if __name__ == "__main__":
    main()
