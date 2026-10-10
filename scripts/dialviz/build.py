#!/usr/bin/env python3
"""Build the dialviz static site: how zensim is evaluated, from its sources of truth.

    python3 scripts/dialviz/build.py --out <dir> [--zenanalyze ../zenanalyze] [--integrity-gates GATES.json]

Reads only repository markdown, JSON summaries and Rust/Python source (plus the
sibling zenanalyze checkout). Opens no label table and no protected payload.
Fails with SourceShapeError when a source no longer has the structure a reader
expects. Design: docs/DIALVIZ_DESIGN_2026-10-10.md.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dialviz import model, pages  # noqa: E402
from dialviz.mdparse import SourceShapeError  # noqa: E402
from dialviz.sources import REPO, Ctx  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"

# The brief's checklist per section; each item names the model data that covers it.
BRIEF = {
    "1. Wanted properties": [
        ("identity at exactly 100", lambda m: _prop(m, "identity")),
        ("smooth approach to 100 (N1–N3)", lambda m: _prop(m, "near-identity")),
        ("monotone ladders", lambda m: _prop(m, "monotone")),
        ("no ties (C2)", lambda m: _prop(m, "no-ties")),
        ("dial addressability (C1–C6, G-DIAL)", lambda m: _prop(m, "addressability")),
        ("target accuracy", lambda m: _prop(m, "target-accuracy")),
        ("codec floors", lambda m: _prop(m, "codec-floors")),
        ("cross-codec consistency at PJND (measured state)", lambda m: False),
        ("negative tails", lambda m: _prop(m, "negative-tails")),
        ("steering (G-STEER M2/M3f)", lambda m: _prop(m, "steering")),
        ("HDR behaviour", lambda m: _prop(m, "hdr")),
        ("runtime and memory", lambda m: _prop(m, "runtime")),
        ("integrity head", lambda m: _prop(m, "integrity")),
        ("ranking vs human data", lambda m: _prop(m, "ranking")),
    ],
    "2. Unwanted properties and gates": [
        (u, (lambda uid: lambda m: any(p["id"] == uid for p in m["unwanted"]))(uid))
        for u, uid in [("identity gaps", "identity-gaps"), ("floor ties", "floor-ties"), ("non-monotone ladders", "non-monotone"),
                       ("saturation", "saturation"), ("unidentified tables", "unidentified-tables"), ("feature drift", "feature-drift"),
                       ("data leakage/exposure", "leakage")]
    ] + [("every release gate with pass/fail/blocked/rule-pending state", lambda m: len(m["gates"]["gates"]) >= 15)],
    "3. Evaluation and combination": [
        ("registered experiments and decision rules", lambda m: len(m["experiments"]) > 20),
        ("E21 as-good: signed, per-source, W2 guards", lambda m: "as_good" in m["evaluation"]),
        ("adoption rules", lambda m: "e28_decision" in m["evaluation"]),
        ("per-source and pooled statistics", lambda m: "sdr_decision" in m["evaluation"]),
        ("Borda/JOD panels (rule and code)", lambda m: "consensus" in m["evaluation"]),
        ("Borda/JOD panel values", lambda m: bool(m["v40_hdr"]["borda"])),
        ("data roles and splits (TRAIN/SELECT/VAL/TEST/TERMINAL)", lambda m: len(m["splits"]["datasets"]) > 10),
        ("exposure ledger", lambda m: len(m["splits"]["ledger"]) > 30),
        ("release-gate matrix", lambda m: bool(m["crosswalk"])),
        ("selection vs qualification", lambda m: bool(m["freeze"]["gates"])),
    ],
    "4. zensim features": [
        ("feature_defs families and IDs", lambda m: len(m["features"]["blocks"]) >= 19),
        ("forms Difference/Similarity/ReferenceOnly", lambda m: any(s["form"] for s in m["features"]["slots"])),
        ("scale × channel layout", lambda m: bool(m["features"]["slots"])),
        ("formula revisions Rev1–Rev5", lambda m: len(m["features"]["revisions"]) >= 5),
        ("named sets: by_v2fy 420", lambda m: len(m["named_sets"]["by_v2fy"]) == 420),
        ("named sets: E33 arms A/C incl. products", lambda m: len(m["named_sets"]["fx1_table"]) == 410),
        ("feature_sets_registry.json", lambda m: len(m["featuresets"]["sets"]) > 40),
    ],
    "5. zenanalyze features": [
        ("features_table catalogue", lambda m: len(m["za"]["features"]) >= 100),
        ("versions", lambda m: m["za"]["version"] is not None),
        ("which routers pin which features (drift state)", lambda m: any(p["pins"] for p in m["pickers"])),
        ("derived KEEP_FEATURES configs resolved statically",
         lambda m: not any(p["kind"].endswith("not resolved)") for p in m["pickers"])),
    ],
}


def _prop(m, pid):
    return any(p["id"] == pid for p in m["wanted"])


def coverage(m) -> list[dict]:
    out = []
    for section, items in BRIEF.items():
        done = [name for name, f in items if f(m)]
        miss = [name for name, f in items if not f(m)]
        out.append({"section": section, "covered": len(done), "total": len(items), "missing": "; ".join(miss) or "—"})
    return out


def search_index(m) -> list[dict]:
    idx = []
    for p in m["wanted"]:
        idx.append({"k": "property", "t": p["title"], "d": p["definition"][:140], "u": f'property/{p["id"]}.html'})
    for p in m["unwanted"]:
        idx.append({"k": "defect", "t": p["title"], "d": p["definition"][:140], "u": f'gates.html#{p["id"]}'})
    for g in m["gates"]["gates"]:
        idx.append({"k": "gate", "t": g["name"], "d": re.sub(r"\*\*|`", "", g["status_text"])[:140], "u": f'gate/{g["id"]}.html',
                    "x": re.sub(r"\*\*|`", "", g["pass_rule"])[:400]})
    for e in m["experiments"]:
        idx.append({"k": "experiment", "t": f'{e["label"]} {e["title"][:90]}', "d": e["registration"],
                    "u": f'experiment/{e["scheme"]}-{e["id"]}.html'})
    for b in m["features"]["blocks"]:
        idx.append({"k": "family", "t": b["family"], "d": f'slots {b["lo"]}–{b["hi"]}, {len(b["signals"])} signals',
                    "u": f'feature/{b["family"]}.html', "x": " ".join(s["name"] for s in b["signals"])})
    for d in m["splits"]["datasets"]:
        idx.append({"k": "dataset", "t": d["name"], "d": re.sub(r"\*\*|`", "", d["tier"])[:120], "u": "splits.html#registry"})
    for f in m["za"]["features"]:
        idx.append({"k": "zenanalyze", "t": f["name"], "d": f'id {f["id"]} · {f["doc"][:100]}', "u": "zenanalyze.html#drift"})
    for b in m["bugs"]:
        idx.append({"k": "bug", "t": re.sub(r"\*\*|`", "", b["title"])[:110], "d": f'{b["date"]} · {b["status"]}', "u": "gates.html#bugs"})
    for k, t in [("evaluation.html#e21", "E21 as-good rule"), ("evaluation.html#stats", "Statistics owners (panel, zenstats)"),
                 ("evaluation.html#borda", "Borda / JOD consensus"), ("evaluation.html#selection", "Selection vs qualification"),
                 ("evaluation.html#contract", "September 8 production contract"), ("splits.html#ledger", "Exposure ledger"),
                 ("features.html#named", "Named feature sets (by_v2fy, E33)"), ("features.html#revisions", "Formula revisions Rev1–Rev5"),
                 ("features.html#registry", "Feature-set registry")]:
        idx.append({"k": "section", "t": t, "d": "", "u": k})
    return idx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", required=True, type=Path, help="output directory (replaced pages are rewritten in place)")
    ap.add_argument("--zenanalyze", type=Path, default=REPO.parent / "zenanalyze", help="zenanalyze checkout (read only)")
    ap.add_argument("--integrity-gates", type=Path, default=None,
                    help="optional ZCTH TRAIN-refit GATES.json (lives outside the repository)")
    args = ap.parse_args(argv)
    ctx = Ctx(REPO, args.zenanalyze)
    try:
        m = model.build(ctx, args.integrity_gates)
    except SourceShapeError as e:
        print(f"dialviz: source shape changed: {e}", file=sys.stderr)
        return 2
    cov = coverage(m)
    files: dict[str, str] = {"index.html": pages.page_index(m), "evaluation.html": pages.page_evaluation(m),
                             "splits.html": pages.page_splits(m), "zenanalyze.html": pages.page_zenanalyze(m)}
    for f in (pages.page_properties, pages.page_gates, pages.page_experiments, pages.page_features):
        files.update(f(m))
    files["sources.html"] = pages.page_sources(m, cov)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("property", "gate", "experiment", "feature", "assets"):
        (out / sub).mkdir(exist_ok=True)
    for rel, html in files.items():
        (out / rel).write_text(html, encoding="utf-8")
    for a in ASSETS.iterdir():
        shutil.copyfile(a, out / "assets" / a.name)
    (out / "search.js").write_text("window.DIALVIZ_INDEX=" + json.dumps(search_index(m), ensure_ascii=False) + ";\n", encoding="utf-8")
    slim = {k: v for k, v in m.items() if k not in ("features",)}
    slim["features"] = {k: v for k, v in m["features"].items() if k != "slots"}
    (out / "model.json").write_text(json.dumps(slim, indent=1, default=str), encoding="utf-8")
    (out / "coverage.json").write_text(json.dumps(cov, indent=1), encoding="utf-8")
    print(f"dialviz: wrote {len(files)} pages to {out} from {len(m['sources'])} sources")
    for c in cov:
        print(f"  {c['section']}: {c['covered']}/{c['total']}" + (f"  missing: {c['missing']}" if c["missing"] != "—" else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
