#!/usr/bin/env python3
"""Decide the dvifmish variant screen (rule fixed before any screen result).

Per arm and seed: human composite = mean global SROCC over the two human
selection legs (CID22-A, KonFiG validation). Teacher legs (codec_dev,
safesyn_dev; SSIMULACRA2 labels) are reported beside it and never decide.
σ = the mean over arms of the seed SD of the composite. An arm SURVIVES when
its mean composite is at least (best mean − 2σ); the baseline is always
carried. Paired per-seed deltas against the baseline are reported too.

Usage: screen_decide.py <screen work dir> [--json out.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

HUMAN = ("cid22a", "konfig_val")
TEACHER = ("codec_dev", "safesyn_dev")
BASE = "base"


def main():
    work = Path(sys.argv[1])
    ev = {leg: json.loads((work / "eval" / f"{leg}.json").read_text()) for leg in HUMAN + TEACHER}
    arms = sorted({m.rsplit("-s", 1)[0].removeprefix("screen-")
                   for m in ev["cid22a"]["models"]})
    seeds = sorted({int(m.rsplit("-s", 1)[1]) for m in ev["cid22a"]["models"]})

    def sro(leg, arm, s):
        return ev[leg]["models"][f"screen-{arm}-s{s}"]["srocc"]

    comp = {a: np.array([np.mean([sro(l, a, s) for l in HUMAN]) for s in seeds]) for a in arms}
    teach = {a: {l: float(np.mean([sro(l, a, s) for s in seeds])) for l in TEACHER} for a in arms}
    legs = {a: {l: float(np.mean([sro(l, a, s) for s in seeds])) for l in HUMAN} for a in arms}
    sigma = float(np.mean([np.std(v, ddof=1) for v in comp.values()]))
    best = max(arms, key=lambda a: comp[a].mean())
    bar = comp[best].mean() - 2 * sigma
    rows = []
    for a in sorted(arms, key=lambda a: -comp[a].mean()):
        d = comp[a] - comp[BASE]
        rows.append({"arm": a, "composite_mean": float(comp[a].mean()),
                     "composite_sd": float(np.std(comp[a], ddof=1)),
                     "per_seed": comp[a].tolist(), "vs_base_mean": float(d.mean()),
                     "vs_base_per_seed": d.tolist(), **{f"{l}": legs[a][l] for l in HUMAN},
                     **{f"{l}": teach[a][l] for l in TEACHER},
                     "survives": bool(comp[a].mean() >= bar or a == BASE)})
    res = {"rule": __doc__.split("Usage")[0].strip(), "seeds": seeds, "sigma": sigma,
           "best": best, "bar": bar, "arms": rows}
    print(f"sigma (mean seed SD) {sigma:.4f}; best {best} {comp[best].mean():.4f}; bar {bar:.4f}")
    print(f"{'arm':16s} {'human':>7s} {'sd':>6s} {'vs base':>8s} {'cid22a':>7s} {'konfig':>7s} "
          f"{'codec*':>7s} {'safesyn*':>8s}  survives")
    for r in rows:
        print(f"{r['arm']:16s} {r['composite_mean']:7.4f} {r['composite_sd']:6.4f} "
              f"{r['vs_base_mean']:+8.4f} {r['cid22a']:7.4f} {r['konfig_val']:7.4f} "
              f"{r['codec_dev']:7.4f} {r['safesyn_dev']:8.4f}  {'yes' if r['survives'] else 'no'}")
    print("(* teacher legs: SSIMULACRA2 labels, reported only)")
    if len(sys.argv) > 3 and sys.argv[2] == "--json":
        Path(sys.argv[3]).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
