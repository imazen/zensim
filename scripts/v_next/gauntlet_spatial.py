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


def build_integrity_gallery(root, out):
    """Show every active TRAIN control using the recorded scalar and PNG audits."""
    from html import escape
    from corruption_gate_eval import integrity_summary

    root, out = Path(root), Path(out)
    if out.exists():
        raise ValueError(f"refusing to overwrite an existing gallery: {out}")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    read = lambda name: json.loads((root / name).read_text())
    admission, assessment = read("ADMISSION.json"), read("ASSESSMENT.json")
    if (admission["schema"] != "integrity-train-diagnostic-v1"
            or any(r["role"] != "train" for r in admission["records"])
            or assessment["admission_sha256"] != sha(root / "ADMISSION.json")
            or assessment["audit_sha256"] != sha(root / "audit.jsonl")):
        raise ValueError("integrity gallery requires bound TRAIN evidence")

    def audits(name):
        rows = [json.loads(line) for line in (root / name).read_text().splitlines()]
        keyed = {int(r["human_score"]): r for r in rows}
        if len(keyed) != len(rows) or any(int(r["human_score"]) != r["human_score"] for r in rows):
            raise ValueError("duplicate/noninteger integrity gallery key")
        return keyed

    original, png, prepared = audits("audit.jsonl"), audits("gallery-audit.jsonl"), audits("prepared-audit.jsonl")
    checked = integrity_summary(admission["records"], original)
    if any(checked[k] != assessment[k] for k in checked):
        raise ValueError("integrity gallery assessment differs from owner")
    active = {r["index"] for r in checked["rows"] if r["active"]}
    if set(png) != active or not active.issubset(prepared):
        raise ValueError("integrity gallery must show every active control")
    metadata = {r["index"]: r for r in admission["records"]}
    examples, assets = [], {}
    for key in sorted(active):
        a, p, m = original[key], png[key], metadata[key]
        for field in ("reference_pixels_sha256", "distorted_pixels_sha256", "base_score",
                      "head_probability", "pixel_composed_score", "model_inputs"):
            if a[field] != p[field] or a[field] != prepared[key][field]:
                raise ValueError("gallery PNG/prepared evidence differs from scored pixels")
        if prepared[key]["prepared_steering"]["status"] != "REJECTED_INTEGRITY":
            raise ValueError("active gallery control lacks prepared rejection")
        images = []
        for side in ("reference", "distorted"):
            path = Path(p[side])
            if sha(path) != p[side + "_file_sha256"]:
                raise ValueError("integrity gallery image hash mismatch")
            rel = "images/" + p[side + "_file_sha256"] + ".png"
            assets[rel] = path
            images.append(f'<figure><figcaption>{side}</figcaption><a href="{rel}"><img loading="lazy" src="{rel}" alt="{side}, control {key}"></a></figure>')
        examples.append(f'<article id="case-{key}"><h2>{escape(m["origin"])} · {escape(m["codec"])} q{m["knob"]:g} · row {key}</h2>'
                        f'<p>{escape(m["content_class"])} · TRAIN {escape(m["fit_role"])}. '
                        f'Probability {a["head_probability"]:.6f} &gt; {a["head_threshold"]:g}; '
                        f'base {a["base_score"]:.4f}; composed {a["pixel_composed_score"]:.4f}. '
                        'Prepared steering: REJECTED_INTEGRITY.</p><div class="pair">' + "".join(images) + '</div></article>')
    codec_rows = []
    for codec, result in checked["by_codec"].items():
        rate = result["honest_activation"]
        codec_rows.append(f'<tr><td>{escape(codec)}</td><td>{rate["count"]}/{rate["n"]}</td><td>{rate["rate"]:.3%}</td><td>{result["honest_lowered"]}</td></tr>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Integrity head: honest TRAIN controls</title><style>
body{font:16px system-ui;max-width:1200px;margin:auto;padding:24px;background:#f5f6f8;color:#182230}
article{background:white;padding:18px;margin:24px 0;border:1px solid #ccd3db}h2{font-size:19px}
.pair{display:flex;flex-wrap:wrap;gap:20px}figure{margin:0;flex:1;min-width:240px}img{max-width:100%;height:auto;min-width:192px}
td,th{text-align:left;padding:8px 20px 8px 0}a{color:#1457a5}</style>
<h1>Frozen integrity head on honest TRAIN controls</h1>
<p><strong>TRAIN diagnostic only. No model qualifies.</strong> Every observed activation is shown below, using exact decoded pixels.
These controls retain their preregistered valid labels. Successful encoding does not prove freedom from bugs; visual concerns remain explicit in the report.</p>
<p>The scalar minimum can conceal an active corruption head when the perceptual score is already lower. Prepared steering still rejects the input.</p>
<p><a href="REPORT.md">Full report and limitations</a> · <a href="RESULTS.json">Measured results</a> · <a href="replay.zip">Replay evidence</a></p>
<table><thead><tr><th>Codec</th><th>Active / reconstructed controls</th><th>Rate</th><th>Scalar scores lowered</th></tr></thead><tbody>'''
    page += "".join(codec_rows) + '</tbody></table><p>JXL includes product encodes and distinct native interventions. Images may be enlarged by the browser; click for original PNGs.</p>'
    page += "".join(examples) + '</html>\n'
    out.parent.mkdir(parents=True, exist_ok=True)
    for rel, source in assets.items():
        dest = out.parent / rel
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, dest)
    out.write_text(page)
    return out, len(examples)


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
    assessments. Only aggregates recorded cell correlations; no new correlation
    calculation, scoring, image modification or model selection.
    """
    from html import escape
    from statistics import median

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
    context = read("GALLERY.json") if (root / "GALLERY.json").exists() else {}
    report_stem = context.get("report_stem", "native_map_replay_2026-09-14")
    if not isinstance(report_stem, str) or not report_stem.replace("_", "").replace("-", "").isalnum():
        raise ValueError("native gallery report stem must be a plain filename")
    population = (f"{len({c['family'] for c in cases.values()})} source families, "
                  f"{len(cases)} cells and {sum(len(c['probes']) for c in cases.values())} "
                  "native bitstreams, freshly decoded. ")
    population += context.get("population_note", "Source admission and exclusions are recorded in the report.")
    peer = {m["model"]: m for m in consensus["models"]}
    models = primary["models"] + baseline["models"]
    if set(peer) != {m["model"] for m in models}:
        raise ValueError("native gallery model coverage mismatch")
    rows = []
    for m in models:
        model_cases = [c for c in primary["cases"] + baseline["cases"]
                       if c["model"] == m["model"]]
        if len(model_cases) != len(cases) or {c["id"] for c in model_cases} != set(cases):
            raise ValueError("native gallery correlation cell coverage mismatch")
        peer_ranks = []
        for key in ("ssim2_delta", "ba_quality_delta"):
            values = [c["mass_ranks"][key] for c in model_cases]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                raise ValueError("nonfinite native gallery peer correlation")
            peer_ranks.append(median(values))
        fields = [m["m2_min"], m["native_mass_rank_median"], m["native_mass_rank_min"],
                  m["density_byte_rank_median"]]
        if not all(math.isfinite(v) for v in fields):
            raise ValueError("nonfinite native gallery statistic")
        rows.append("<tr><td>" + escape(m["model"]) + "</td><td>"
                    + ("complete" if m["complete_map_gate_eligible"] else "PARTIAL — hard maxima omitted")
                    + f"</td><td>{m['m2_min']:.3f}</td><td>{m['native_mass_rank_median']:.3f} / "
                    + f"{m['native_mass_rank_min']:.3f}</td><td>{m['density_byte_rank_median']:.3f}</td>"
                    + f"<td>{peer_ranks[0]:.3f}</td><td>{peer_ranks[1]:.3f}</td>"
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
            views.append(f'<figure><a href="{rel}"><img loading="eager" src="{rel}" alt="{label}" '
                         f'width="{case["width"]}" height="{case["height"]}"></a>'
                         f'<figcaption>{label} · {case["width"]}×{case["height"]}</figcaption></figure>')
        examples.append("<section><h2>" + escape(example["model"]) + "</h2><p>"
                        + escape(example["cell"] + " / " + example["probe"])
                        + f" · Zensim Δ {example['score_delta']:+.4f}; SSIM2 Δ {example['ssim2_delta']:+.4f}; "
                        + f"Butteraugli Δ {-example['ba_quality_delta']:+.6f}.</p>"
                        + '<div class="views">' + "".join(views) + "</div></section>")
    diagnostic_figures = []
    for figure in context.get("diagnostic_figures", []):
        name = figure["file"]
        if (not isinstance(name, str) or Path(name).name != name
                or Path(name).suffix not in (".svg", ".png")):
            raise ValueError("diagnostic figure must be a local SVG/PNG filename")
        path = root / name
        if digest(path) != figure["sha256"]:
            raise ValueError("diagnostic figure hash mismatch")
        assets[name] = path
        diagnostic_figures.append(
            '<section><h2>' + escape(figure["title"]) + '</h2><p>'
            + escape(figure["caption"]) + '</p><a href="' + escape(name)
            + '"><img class="plot" src="' + escape(name) + '" alt="'
            + escape(figure["title"]) + '"></a></section>')
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Zensim native JXL comparison — TRAIN, no qualified model</title>
<style>body{font:15px/1.5 system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#182338;background:#f7f9fc}
h1{font-size:1.7rem}h2{font-size:1.1rem;overflow-wrap:anywhere}.notice{padding:1rem;background:#fff0d1;border-left:5px solid #c18410}
.table{overflow-x:auto}table{border-collapse:collapse;background:white;width:100%;font-size:13px}td,th{padding:.6rem;text-align:left;border-bottom:1px solid #d6ddea}td:first-child{overflow-wrap:anywhere;max-width:260px}
section{background:white;padding:1rem;margin:1.5rem 0}.views{display:flex;gap:1rem;align-items:flex-start}.views figure{margin:0;flex:1;min-width:0}.views img{max-width:100%;height:auto}figcaption{color:#536078}.plot{width:100%;height:auto}a{color:#125cba}@media(max-width:700px){.views{flex-wrap:wrap}.views figure{min-width:45%}}</style>
<h1>Native JXL steering comparison · September 14, 2026</h1>
<p class="notice"><strong>TRAIN development only. No model qualifies.</strong> NATIVE_POPULATION These are actual encoder interventions, not reference-pixel replacements.</p>
<p>Complete Rust models score every output. Maps are predicted from baseline pixels before probe scoring. Hard-max additive maps are explicitly partial. Native correlations are mechanism diagnostics, not the original repair gates or a matched-rate–distortion win.</p>
DIAGNOSTIC_FIGURES<div class="table"><table><thead><tr><th>Model</th><th>Additive map coverage</th><th>Native M2 minimum</th><th>Own-score mass/response rank median / min</th><th>Own-score density/gain-per-byte rank median</th><th>SSIM2 mass/response rank median</th><th>Butteraugli quality mass/response rank median</th><th>Robust peer conflicts</th></tr></thead><tbody>"""
    page += "".join(rows) + """</tbody></table></div>
<p>Own-score columns measure internal consistency. The SSIM2 and Butteraugli columns compare map mass with those peers' measured quality responses; each column is the median of the recorded cell correlations. High internal consistency does not establish perceptual quality or encoding benefit. Peer conflicts require both SSIM2 and Butteraugli to agree beyond the registered margins and Zensim to move oppositely by more than .1. These peers are not human truth; consult the report for their use in training. D uses its frozen revision1; candidate ensembles use revision3.</p>
<p><a href="native_map_replay_2026-09-14.md">Full report and limitations</a> · <a href="native_map_replay_2026-09-14.results.json">Results and evidence hashes</a> · <a href="FILES.json">Evidence index</a></p>
<h2>Raw native response scatter</h2><a href="native_scatter.svg"><img class="plot" src="native_scatter.svg" alt="Per-model and per-image raw attribution mass versus actual native quantizer response"></a>
<h2>Exact A/B failure examples</h2><p>Largest signed peer-consensus conflict for each affected model. Selected illustrations, not a representative population. Click an image for original-size PNG bytes. No synthetic repair or image resizing is stored.</p>"""
    page = page.replace("NATIVE_POPULATION", escape(population))
    page = page.replace("DIAGNOSTIC_FIGURES", "".join(diagnostic_figures))
    page = page.replace("native_map_replay_2026-09-14.md", report_stem + ".md")
    page = page.replace("native_map_replay_2026-09-14.results.json", report_stem + ".results.json")
    page += "".join(examples) + "</html>\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    for rel, path in assets.items():
        dest = out.parent / rel
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, dest)
    shutil.copyfile(root / "native_scatter.svg", out.parent / "native_scatter.svg")
    out.write_text(page)
    return out, out.stat().st_size, len(examples), len(models)
