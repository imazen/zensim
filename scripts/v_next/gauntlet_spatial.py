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
                   evidence_label=manifest.get("evidence_label", "ZENSIM · DEVELOPMENT EVIDENCE"),
                   scope_note=manifest.get("scope_note", "Recorded development checks; see the source experiment for dataset roles and model provenance."),
                   spatial_cases_sha256=hashlib.sha256((root / "SPATIAL_CASES.json").read_bytes()).hexdigest())
    (assets / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    template = Path(__file__).with_name("gauntlet_spatial.html").read_text()
    embedded = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    out.write_text(template.replace("__SPATIAL_DATA__", embedded))
    return out, out.stat().st_size, payload["failed"], len(cells)


def build_native_gallery(root, out):
    """Render saved native probes without pretending they are rectangle repairs.

    Consumes the native mode of diffmap_block_coherence and canonical panel
    assessments. No statistics, scoring, image modification or model selection.
    """
    from html import escape

    root, out = Path(root), Path(out)
    if out.exists():
        raise ValueError(f"refusing to overwrite an existing gallery: {out}")
    read = lambda name: json.loads((root / name).read_text())
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest, result = read("REPLAY.json"), read("RESULT.json")
    primary, baseline = read("ASSESSMENT.json"), read("BASELINE_ASSESSMENT.json")
    consensus = read("CONSENSUS.json")
    if (manifest["role"] != "train" or result["role"] != "train"
            or result["schema"] != "zensim-native-map-replay-result-v1"
            or result["manifest_sha256"] != digest(root / "REPLAY.json")
            or primary["result_sha256"] != digest(root / "RESULT.json")
            or baseline["result_sha256"] != digest(root / "baseline-result.json")):
        raise ValueError("native gallery evidence identity mismatch")
    cases = {c["id"]: c for c in manifest["cases"]}
    if len(cases) != len(manifest["cases"]) or set(cases) != {c["id"] for c in result["cases"]}:
        raise ValueError("native gallery case coverage mismatch")
    if any(c["role"] != "train" or any(p["role"] != "train" for p in c["probes"])
           for c in cases.values()):
        raise ValueError("native gallery TRAIN cases only")
    peer = {m["model"]: m for m in consensus["models"]}
    models = primary["models"] + baseline["models"]
    if set(peer) != {m["model"] for m in models}:
        raise ValueError("native gallery model coverage mismatch")
    rows = []
    for m in models:
        fields = [m["m2_min"], m["native_mass_rank_median"], m["native_mass_rank_min"],
                  m["density_byte_rank_median"]]
        if not all(math.isfinite(v) for v in fields):
            raise ValueError("nonfinite native gallery statistic")
        rows.append("<tr><td>" + escape(m["model"]) + "</td><td>"
                    + ("complete" if m["complete_map_gate_eligible"] else "PARTIAL — hard maxima omitted")
                    + f"</td><td>{m['m2_min']:.3f}</td><td>{m['native_mass_rank_median']:.3f} / "
                    + f"{m['native_mass_rank_min']:.3f}</td><td>{m['density_byte_rank_median']:.3f}</td>"
                    + f"<td>{peer[m['model']]['conflicts']} / {peer[m['model']]['consensus_cases']}</td></tr>")
    assets, examples = {}, []
    for example in consensus["worst_per_model"]:
        case = cases[example["cell"]]
        if case["role"] != "train":
            raise ValueError("native gallery TRAIN examples only")
        probe = next(p for p in case["probes"] if p["name"] == example["probe"])
        assessment = next(c for c in primary["cases"] + baseline["cases"]
                          if c["id"] == example["cell"] and c["model"] == example["model"])
        sample = next(p for p in assessment["samples"] if p["name"] == example["probe"])
        if (sample["delta"] != example["score_delta"]
                or sample["ssim2_delta"] != example["ssim2_delta"]
                or sample["ba_quality_delta"] != example["ba_quality_delta"]):
            raise ValueError("native gallery example differs from measured outcome")
        views = []
        for label, item in [("Reference", case["reference"]),
                            ("Baseline encode", case["probes"][0]), ("Native intervention", probe)]:
            path = Path(item["path"])
            if digest(path) != item["sha256"]:
                raise ValueError(f"native gallery PNG hash mismatch: {path}")
            rel = f"images/{item['sha256']}.png"
            assets[rel] = path
            views.append(f'<figure><a href="{rel}"><img loading="lazy" src="{rel}" alt="{label}"></a>'
                         f'<figcaption>{label} · {case["width"]}×{case["height"]}</figcaption></figure>')
        examples.append("<section><h2>" + escape(example["model"]) + "</h2><p>"
                        + escape(example["cell"] + " / " + example["probe"])
                        + f" · Zensim Δ {example['score_delta']:+.4f}; SSIM2 Δ {example['ssim2_delta']:+.4f}; "
                        + f"Butteraugli Δ {-example['ba_quality_delta']:+.6f}.</p>"
                        + '<div class="views">' + "".join(views) + "</div></section>")
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Zensim native JXL comparison — TRAIN, no qualified model</title>
<style>body{font:15px/1.5 system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#182338;background:#f7f9fc}
h1{font-size:1.7rem}h2{font-size:1.1rem;overflow-wrap:anywhere}.notice{padding:1rem;background:#fff0d1;border-left:5px solid #c18410}
.table{overflow-x:auto}table{border-collapse:collapse;background:white;width:100%;font-size:13px}td,th{padding:.6rem;text-align:left;border-bottom:1px solid #d6ddea}td:first-child{overflow-wrap:anywhere;max-width:260px}
section{background:white;padding:1rem;margin:1.5rem 0}.views{display:flex;gap:1rem;align-items:flex-start}.views figure{margin:0;flex:1;min-width:0}.views img{max-width:100%;height:auto}figcaption{color:#536078}.plot{width:100%;height:auto}a{color:#125cba}@media(max-width:700px){.views{flex-wrap:wrap}.views figure{min-width:45%}}</style>
<h1>Native JXL steering comparison · September 14, 2026</h1>
<p class="notice"><strong>TRAIN development only. No model qualifies.</strong> Three source families, two sizes, two distances; twelve cells and 408 retained native bitstreams, freshly decoded. The reserved document family is excluded. These are actual encoder interventions, not reference-pixel replacements.</p>
<p>Complete Rust models score every output. Maps are predicted from baseline pixels before probe scoring. Hard-max additive maps are explicitly partial. Native correlations are mechanism diagnostics, not the original repair gates or a matched-rate–distortion win.</p>
<div class="table"><table><thead><tr><th>Model</th><th>Additive map coverage</th><th>Native M2 minimum</th><th>Mass/response rank median / min</th><th>Density/gain-per-byte rank median</th><th>Robust peer conflicts</th></tr></thead><tbody>"""
    page += "".join(rows) + """</tbody></table></div>
<p>Peer conflicts require both SSIM2 and Butteraugli to agree beyond the registered margins and Zensim to move oppositely by more than .1. These peers are not human truth. D uses its frozen revision1; the six new ensembles use revision3.</p>
<p><a href="native_map_replay_2026-09-14.md">Full report and limitations</a> · <a href="native_map_replay_2026-09-14.results.json">Results and evidence hashes</a> · <a href="FILES.json">Evidence index</a></p>
<h2>Raw native response scatter</h2><a href="native_scatter.svg"><img class="plot" src="native_scatter.svg" alt="Per-model and per-image raw attribution mass versus actual native quantizer response"></a>
<h2>Exact A/B failure examples</h2><p>Largest signed peer-consensus conflict for each affected model. Selected illustrations, not a representative population. Click an image for original-size PNG bytes. No synthetic repair or image resizing is stored.</p>"""
    page += "".join(examples) + "</html>\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    for rel, path in assets.items():
        dest = out.parent / rel
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, dest)
    shutil.copyfile(root / "native_scatter.svg", out.parent / "native_scatter.svg")
    out.write_text(page)
    return out, out.stat().st_size, len(examples), len(models)
