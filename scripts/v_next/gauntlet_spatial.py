"""Spatial A/B view for the gauntlet renderer; consumes recorded Rust results.

No scoring or correlation implementation. Images are copied byte-for-byte,
checked against the experiment manifest. Block repairs are browser previews
of the exact recorded replacement, not newly evaluated image variants.
"""
import hashlib
import json
import math
import shutil
from pathlib import Path


def build_spatial_gallery(root, out):
    root, out = Path(root), Path(out)
    if out.exists():
        raise ValueError(f"refusing to overwrite an existing gallery: {out}")
    manifest = json.loads((root / "SPATIAL_CASES.json").read_text())
    matrix = json.loads((root / "SPATIAL_MATRIX.json").read_text())
    if not matrix["success"] or any(r["exit_code"] != 0 for r in matrix["rows"]):
        raise ValueError("spatial matrix is incomplete")
    cases = {c["name"]: dict(c) for c in manifest["cases"]}
    # Validate the entire evidence packet before creating any output.
    for case in cases.values():
        for side in ("reference", "distorted"):
            path = Path(case[side])
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != case[side + "_sha256"]:
                raise ValueError(f"image hash mismatch: {path}")
            case[side + "_url"] = f"images/{actual}.png"
    cells, reports = [], {}
    seen = set()
    for row in matrix["rows"]:
        model, case_name, block = row["model"], row["case"], row["block"]
        if (model, case_name, block) in seen:
            raise ValueError("duplicate model/case/block cell")
        seen.add((model, case_name, block))
        key = f"{model}--{case_name}--b{block}"
        path = root / "spatial" / (key + ".json")
        report_bytes = path.read_bytes()
        d = json.loads(report_bytes)
        if d["schema"] != "zensim-finite-rectangle-coherence-v1" or d["block_size"] != block:
            raise ValueError(f"unexpected report: {path}")
        if len(d["blocks"]) != d["pixel_interventions"]:
            raise ValueError(f"incomplete block array: {path}")
        if d.get("refinement_unsupported_ids") or not d.get("refinement_available"):
            raise ValueError(f"unsupported refinement in {path}")
        # This gallery is an inspection of complete rectangular partitions.
        w, h = d["width"], d["height"]
        bounds = {tuple(b["bounds"]) for b in d["blocks"]}
        expected = {(x, y, min(x + block, w), min(y + block, h))
                    for y in range(0, h, block) for x in range(0, w, block)}
        if bounds != expected or len(bounds) != len(d["blocks"]):
            raise ValueError(f"block partition mismatch: {path}")
        for b in d["blocks"]:
            for k in ("score_delta", "refinement_gain", "linearized_gain", "density_gain"):
                if not isinstance(b[k], (int, float)) or not math.isfinite(b[k]):
                    raise ValueError(f"nonfinite block gain: {path}")
        valid = lambda v: isinstance(v, (int, float)) and math.isfinite(v)
        fail_m2 = not valid(d["m2"]) or d["m2"] < .99
        fail_m3f = not valid(d["m3f"]) or d["m3f"] < .70
        case = cases[case_name]
        cells.append(dict(id=key, model=model, case=case_name, block=block,
                          width=w, height=h, m2=d["m2"], m3a=d["m3a"], m3f=d["m3f"],
                          fail_m2=fail_m2, fail_m3f=fail_m3f,
                          base_score=d["base_score"], n=len(d["blocks"]),
                          reference=case["reference_url"], distorted=case["distorted_url"],
                          data=f"data/{key}.json", report_sha256=hashlib.sha256(report_bytes).hexdigest()))
        # Preserve numerical evidence; avoid exposing absolute filesystem paths.
        d["models"] = [dict(m, path=Path(m["path"]).name) for m in d["models"]]
        reports[key] = d
    out.parent.mkdir(parents=True, exist_ok=True)
    assets = out.parent
    (assets / "images").mkdir(exist_ok=True)
    (assets / "data").mkdir(exist_ok=True)
    for c in cases.values():
        for side in ("reference", "distorted"):
            shutil.copyfile(c[side], assets / c[side + "_url"])
    for key, data in reports.items():
        (assets / "data" / (key + ".json")).write_text(json.dumps(data, separators=(",", ":")) + "\n")
    payload = dict(cells=cells, total=len(cells), failed=sum(c["fail_m2"] or c["fail_m3f"] for c in cells),
                   thresholds=dict(m2=.99, m3f=.70), seed=manifest["seed"],
                   spatial_cases_sha256=hashlib.sha256((root / "SPATIAL_CASES.json").read_bytes()).hexdigest())
    (assets / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    template = Path(__file__).with_name("gauntlet_spatial.html").read_text()
    embedded = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    out.write_text(template.replace("__SPATIAL_DATA__", embedded))
    return out, out.stat().st_size, payload["failed"], len(cells)
