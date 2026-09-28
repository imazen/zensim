#!/usr/bin/env python3
"""Wrap the Phase-2 K1 constants spec as a standalone fit artefact so the
uniform `score` path can emit per-row scores without refitting.

K1 has no learned head: level weights are uniform, channel weights are the
degenerate single-plane mix, and the output map is taken from the
eval-consts artefact of a TRAIN-side domain (Amendment-1 map family
A*exp(-lam*E)+B — see fits/<dom>_k1.json) so that no eval-domain label is
consumed.

usage: k1_to_fit.py <k1-spec.json> <evalconsts.json> <out.json>
"""
import json
import sys
from pathlib import Path

spec_p, ec_p, out_p = sys.argv[1], sys.argv[2], Path(sys.argv[3])
spec = json.loads(Path(spec_p).read_text())
ec = json.loads(Path(ec_p).read_text())
assert "map" in ec, "eval-consts artefact must carry the v2 map"
plane = spec.get("input_plane", "xyb_y")
levels = spec["levels"]
assert len(levels) == 5
fit = {
    "schema": "dvifm-standalone-fit-v2",
    "variant": "k1-control",
    "planes": [plane],
    "levels": {plane: [
        {"g": lv["g"], "p": lv["p"], "c0": lv["c0"], "beta": lv["beta"],
         "sharp": lv["sharp"]} for lv in levels]},
    "level_weights": {plane: [0.2] * 5},
    # single-plane artefact: channel mix is overridden to (1,0) in
    # forward_E; store a finite log-domain pair regardless
    "channel_weights": [2.0 / 3.0, 1.0 / 6.0],
    "map": ec["map"],
    "fit_mse": ec.get("fit_mse"),
    "provenance": {
        "note": "K1 spec wrapped unmodified for scoring; output map from "
                "the CID22-A eval-consts fit (TRAIN-side only).",
        "spec": spec_p,
        "eval_consts": ec_p,
    },
}
out_p.write_text(json.dumps(fit, indent=1) + "\n")
print(f"wrote {out_p} (plane={plane}, map={ec['map']})")
