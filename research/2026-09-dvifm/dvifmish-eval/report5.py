#!/usr/bin/env python3
"""Markdown tables for the dvifmish §5 evaluation, from `table5.py eval` JSON.

A models file (JSON list) names every row and its exposure, so no cell can be
printed without its role:
  {"id": "talk-faithful-luma", "label": "...", "family": "dvifm"|"peer",
   "planes": "Y'"|"Y'CbCr"|..., "fitted_on": "...",
   "exposure": {"<set>": "fit"|"partly"|"held-out"|"overlap"|"exposed"}}
Sets missing from "exposure" default to "held-out" and are printed as such.

Usage: report5.py <work> <models.json> <out.md> [--math float|int] [--repro DIR]

--repro DIR: read every DVIFM row from a `repro/run.sh` output (DIR/results and
DIR/scores, the published route: datasets as distributed, decoded by the CLI)
instead of from `table5.py`; peers still come from <work>.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

MARK = {"fit": "ᶠ", "partly": "ᵖ", "overlap": "ᵒ", "exposed": "ˣ", "held-out": ""}
LEGEND = ("ᶠ fit-domain (fitted on these pairs) · ᵖ partly fit-domain (some pairs were in "
          "the fitting set) · ᵒ scene overlap (the model was fitted on TID2013, whose scenes are "
          "crops of NNCD's) · ˣ previously used for model selection · unmarked: held-out")
TALK = [("tid2013_codec", "TID2013 JPEG+J2K"), ("kadid10k_nt_codec", "KADID-10k JPEG+J2K"),
        ("nncd", "NNCD"), ("aic4_ptc", "AIC-4 crops"), ("cid22", "CID22")]
FULL = [("tid2013_full", "TID2013 all"), ("kadid10k_nt", "KADID-10k all"),
        ("aic4_full", "AIC-4 full-res")]


REPRO = None  # set from --repro


def load(work, name, path):
    if REPRO is not None and path != "peers":
        return load_repro(name, path)
    p = work / "eval" / f"{name}__{path}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_repro(name, math):
    """repro/evaluate.py `table` output, reshaped to eval_scores.py's layout."""
    p = REPRO / "results" / f"{name}__{math}.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return {"models": {k: {"srocc": v["srocc"], "krocc": v["krocc"],
                           "per_codec": {"srocc_mean": v["per_codec"]["srocc"]},
                           "per_source": {"srocc_mean": v["per_source"]["srocc"]}}
                       for k, v in r["presets"].items()}}


def scores_dir(work):
    return (REPRO if REPRO is not None else work) / "scores"


def cell(res, model, role, per=None):
    if res is None or model not in res["models"]:
        return "–"
    d = res["models"][model]
    if per:
        v = d[per]["srocc_mean"]
        return "–" if v is None else f"{v:.3f}{MARK[role]}"
    # eval_scores.py already orients every score toward its label (−E for quality
    # labels, +E for AIC-4's JND distortion), so a sign flip is a real disagreement
    s, k = d["srocc"], d["krocc"]
    return f"{s:.3f} / {k:.3f}{MARK[role]}"


def cid22_set(m):
    """Models fitted on CID22-49 are read on all 49 (fit-domain); everyone else on B(23)."""
    return "cid22_49" if m["exposure"].get("cid22_49") == "fit" else "cid22b23"


def table(work, models, sets, math, per=None):
    head = "| Model | Planes | Fitted on | " + " | ".join(t for _, t in sets) + " |"
    lines = [head, "|" + "---|" * (3 + len(sets))]
    for m in models:
        path = math if m["family"] == "dvifm" else "peers"
        row = [m["label"], m["planes"], m["fitted_on"]]
        for key, _ in sets:
            name = cid22_set(m) if key == "cid22" else key
            role = m["exposure"].get(name, "held-out")
            row.append(cell(load(work, name, path), m["id"], role, per))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def float_int(work, models):
    """Largest relative difference of E and of SROCC between the two paths."""
    out = ["| Preset | Sets | max rel. ΔE | median rel. ΔE | max ΔSROCC |", "|---|---|---|---|---|"]
    for m in models:
        if m["family"] != "dvifm":
            continue
        rel, dsr, nsets, float_only = [], [], 0, 0
        for name in ("tid2013_full", "kadid10k_nt", "nncd", "aic4_ptc", "aic4_full", "cid22_49"):
            fa = scores_dir(work) / name / "float" / f"{m['id']}.tsv"
            ia = scores_dir(work) / name / "int" / f"{m['id']}.tsv"
            if fa.exists() and not ia.exists():
                float_only += 1
            if not (fa.exists() and ia.exists()):
                continue
            ef = np.array([float(r["distortion"]) for r in csv.DictReader(open(fa), delimiter="\t")])
            ei = np.array([float(r["distortion"]) for r in csv.DictReader(open(ia), delimiter="\t")])
            ok = ef > 0
            rel.extend((np.abs(ei[ok] - ef[ok]) / ef[ok]).tolist())
            rf, ri = load(work, name, "float"), load(work, name, "int")
            if rf and ri and m["id"] in rf["models"] and m["id"] in ri["models"]:
                dsr.append(abs(rf["models"][m["id"]]["srocc"] - ri["models"][m["id"]]["srocc"]))
            nsets += 1
        if rel:
            out.append(f"| `{m['id']}` | {nsets} | {max(rel):.2e} | {float(np.median(rel)):.2e} | "
                       f"{max(dsr) if dsr else float('nan'):.4f} |")
        elif float_only:
            out.append(f"| `{m['id']}` | – | integer path refused (float only) | | |")
    return "\n".join(out)


def main():
    global REPRO
    work, models_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    opts = dict(zip(sys.argv[4::2], sys.argv[5::2]))
    math = opts.get("--math", "float")
    if "--repro" in opts:
        REPRO = Path(opts["--repro"])
    models = json.loads(models_path.read_text())
    parts = [
        f"<!-- generated by report5.py from {REPRO or work} ({math} path; peers from {work}) -->",
        "### The talk's test sets (global SROCC / KROCC)", "", table(work, models, TALK, math), "",
        LEGEND, "",
        "### Full sets (global SROCC / KROCC)", "", table(work, models, FULL, math), "",
        "### Secondary: mean SROCC within each codec or distortion type", "",
        table(work, models, TALK + FULL[:2], math, per="per_codec"), "",
        "### Secondary: mean SROCC within each source image", "",
        table(work, models, TALK + FULL[:2], math, per="per_source"), "",
        "### Float and integer paths", "", float_int(work, models), "",
    ]
    out.write_text("\n".join(parts) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
