#!/usr/bin/env python3
"""Fit the X2-winning constant form on the UNION of the X2 seed subsets
and emit the single frozen X1 artefact.

  fit_winner.py <form:gate|prior|curve> <x2_results.json> <out-json>
    ycbcr_y=<bin> ycbcr_cb=<bin> ycbcr_cr=<bin>
    [--init-spec spec.json] [--surfaces-dir dir]

Union rows = sorted(set(union of every seed's `sel` recorded in
x2_results.json)) — the union of all rows any X2 arm saw, in file order.
Gate/prior fit via fit_forms.fit_gate/fit_prior (same machinery as X2);
curve via fs.fit_variant (identical pipeline, writes a v2 artefact).
"""
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/lilith/work/zen/zensim/tools/joint_core')
sys.path.insert(0, '/mnt/v/output/zensim/dvifm-screen2d-2026-09-19/tools')
import numpy as np
import fit_standalone as fs
import fit_forms as ff
from fit_core import RowView

PAIRS = '/mnt/v/output/zensim/joint-core-v1/pairs/pairs_dvifm_sdr.tsv'


def main():
    form, results_path, out_json = sys.argv[1:4]
    planes = {}
    init_spec = ff.DEFAULT_SPEC
    surfaces = None
    for a in sys.argv[4:]:
        if a == "--init-spec":
            pass
        elif a.startswith("--init-spec="):
            init_spec = a.split("=", 1)[1]
        elif a.startswith("--surfaces-dir="):
            surfaces = a.split("=", 1)[1]
        else:
            p, _, path = a.partition("=")
            planes[p] = path
    res = json.loads(Path(results_path).read_text())
    sel = sorted({i for s in res["seeds"]
                  for i in res["arms"]["gate"][str(s)]["sel"]})
    rows_meta = [r for r in csv.DictReader(open(PAIRS), delimiter="\t")]
    y = np.asarray([float(rows_meta[i]["human_score"]) for i in sel])
    pl = tuple(planes.keys())
    caches = {p: RowView(b, sel) for p, b in planes.items()}
    prov = {"form": form, "fit_rows": len(sel),
            "row_sample": "union of X2 seed subsets",
            "seeds": res["seeds"], "n_rows_x2": res["n_rows"],
            "pairs_tsv": PAIRS, "caches": planes}
    t0 = time.time()
    if form == "gate":
        art = ff.fit_gate(pl, caches, y, ff.gate_init_cells(pl, caches))
        art["wall_s"] = time.time() - t0
        out = {**ff.arm_to_json(art, pl), "provenance": prov,
               "form": "gate"}
    elif form == "prior":
        art = ff.fit_prior(pl, caches, y, ff.prior_cells(pl, caches))
        art["wall_s"] = time.time() - t0
        out = {**ff.arm_to_json(art, pl), "provenance": prov,
               "form": "prior"}
    elif form == "curve":
        init_levels = json.loads(Path(init_spec).read_text())["levels"]
        sdir = surfaces or str(Path(out_json).parent / "surfaces_winner")
        Path(sdir).mkdir(parents=True, exist_ok=True)
        art = fs.fit_variant("x1_curve_frozen", pl, caches, y,
                             init_levels, out_json, sdir, prov)
        print(f"curve frozen -> {out_json} "
              f"({time.time() - t0:.1f}s)")
        return 0
    else:
        raise SystemExit(f"unknown form {form}")
    out["wall_s"] = time.time() - t0
    Path(out_json).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{form} frozen -> {out_json} ({out['wall_s']:.1f}s, "
          f"fit mse {out['fit']['mse']:.4f}, "
          f"srocc {out['fit']['srocc']:.4f})")
    return 0


if __name__ == "__main__":
    main()
