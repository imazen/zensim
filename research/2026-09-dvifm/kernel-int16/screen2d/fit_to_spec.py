#!/usr/bin/env python3
"""Emit a dvifm-spec-v1 JSON per plane from a standalone fit artefact —
the `--dvifm-spec` input for Part-C extractions.

The fit artefact carries (g, p, c0, beta, sharp) per (plane, level);
the extractor spec additionally needs c_hi / f2_centers / band / edge,
taken verbatim from the K1 spec (the pyramid/feature geometry is not
fitted — only the visibility constants are).

usage: fit_to_spec.py <fit.json> <k1_spec.json> <out_dir>
  -> <out_dir>/dvifm_<plane>.json per plane in the artefact
"""
import json
import sys
from pathlib import Path

PLANE_KEY = {"ycbcr_y": "ycbcr_y", "ycbcr_cb": "ycbcr_cb",
             "ycbcr_cr": "ycbcr_cr", "xyb_y": "xyb_y"}


def main() -> int:
    fit = json.loads(Path(sys.argv[1]).read_text())
    k1 = json.loads(Path(sys.argv[2]).read_text())
    outdir = Path(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)
    k1lv = k1["levels"]
    for plane in fit["planes"]:
        levels = []
        for l, lv in enumerate(fit["levels"][plane]):
            levels.append({
                "g": lv["g"], "p": lv["p"], "c0": lv["c0"],
                "beta": lv["beta"], "sharp": lv["sharp"],
                "c_hi": k1lv[l].get("c_hi"),
                "f2_centers": k1lv[l]["f2_centers"],
                "band": k1lv[l].get("band", "local"),
                "edge": k1lv[l].get("edge", True)})
        spec = {"schema": "dvifm-spec-v1",
                "note": (f"Part-D fitted constants ({fit.get('variant')}) "
                         f"on plane {plane}; f2_centers/band/edge from K1"),
                "input_plane": PLANE_KEY[plane],
                "levels": levels,
                "provenance": {"fit": sys.argv[1],
                               "fit_mse": fit.get("fit_mse")}}
        out = outdir / f"dvifm_{plane}.json"
        out.write_text(json.dumps(spec, indent=1) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
