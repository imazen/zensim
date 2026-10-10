"""Readers for registered experiments and their decision/result records.

Two numbering schemes exist and are kept apart:
  * the Rev4 program, docs/REV4_EXPERIMENTS_2026-09-23.md (E1–E6, 2026-09-23);
  * the featpot design log (E1–E33): E1–E24 registered in the
    scripts/rev4_featpot/e*.py docstrings, E25 in the Rev5 spec addendum,
    E26–E33 and KADID TERMINAL in benchmarks/*registration*.md.

Outcomes are read from the recorded decision/result JSON files; arms from each
schema are normalized into one shape. Nothing is recomputed.
"""
from __future__ import annotations

import ast
import re

from .mdparse import SourceShapeError, headings, section

REV4_PROGRAM = "docs/REV4_EXPERIMENTS_2026-09-23.md"
FEATPOT_DIR = "scripts/rev4_featpot"

# Decision/result records and the schema each must carry. Arms are normalized below.
RESULTS = {
    "E24": ("benchmarks/e24_rev5_decision_2026-10-04.json", None),
    "E25": ("benchmarks/e25_2026-10-04/decision.json", "featpot-e25-decision-v1"),
    "E27": ("benchmarks/e27_result_summary_2026-10-05.json", "e27-registered-decision-v1"),
    "E28": ("benchmarks/e28_result_summary_2026-10-07.json", "e28-decision-v1"),
    "E30": ("benchmarks/e30_result_summary_2026-10-07.json", "e30-result-summary-v1"),
    "V40": ("benchmarks/v40_result_summary_2026-10-09.json", None),
    "V40HDR": ("benchmarks/v40_hdr_result_summary_2026-10-09.json", "v40-hdr-result-summary-v1"),
}
V40_STUDY = {"e29": "E29", "e31": "E31", "e32": "E32"}


def _ps(per_source: dict) -> dict:
    out = {}
    for k, v in (per_source or {}).items():
        if isinstance(v, dict):
            if "delta" in v:
                out[k] = {"delta": v["delta"], "se": v.get("se"), "n": v.get("n")}
            elif "signed" in v and isinstance(v["signed"], dict):
                out[k] = {"delta": v["signed"].get("delta"), "se": v["signed"].get("se"), "n": v["signed"].get("n")}
        elif isinstance(v, (int, float)):
            out[k] = {"delta": v, "se": None, "n": None}
    return out


def _arm(exp, name, *, spec=None, signed=None, se=None, worst=None, per_source=None, w2=None, w2_se=None,
         as_good=None, passes=None, verdict=None, reason=None, seed_deltas=None, extra=None, path=None):
    return {"experiment": exp, "arm": name, "spec": spec, "signed": signed, "se": se, "worst": worst,
            "per_source": _ps(per_source), "w2": w2, "w2_se": w2_se, "as_good": as_good, "passes": passes,
            "verdict": verdict, "reason": reason, "seed_deltas": seed_deltas or [], "extra": extra or {}, "path": path}


def results(ctx) -> dict:
    out: dict[str, dict] = {}
    for key, (rel, schema) in RESULTS.items():
        d = ctx.json(rel, "experiment_results")
        if schema and d.get("schema") != schema:
            raise SourceShapeError(f"{rel}: schema {d.get('schema')!r}, expected {schema!r}")
        arms = []
        if key == "E24":
            for n, a in d["arms"].items():
                arms.append(_arm("E24", n, spec=a["spec"], signed=a["signed_mean"], se=a["signed_se"], worst=a["signed_worst"],
                                 per_source=a["per_source"], w2=a["w2"], w2_se=a["w2_se"], as_good=a["as_good"], path=rel))
            out["E24"] = {"path": rel, "arms": arms, "adopted": None, "control": d.get("control")}
        elif key == "E25":
            a = d["primary"]
            arms.append(_arm("E25", "rev4-weights@rev5", spec=a["spec"], signed=a["signed_mean"], se=a["signed_se"],
                             worst=a["signed_worst"], per_source=a["per_source"], w2=a["w2"], w2_se=a["w2_se"],
                             as_good=a["as_good"], path=rel))
            out["E25"] = {"path": rel, "arms": arms, "adopted": None, "status": d.get("status")}
        elif key == "E27":
            for n, a in d["arms"].items():
                s = a["sdr"]
                arms.append(_arm("E27", n, spec=s["spec"], signed=s["signed_mean"], se=s["signed_se"], worst=s["signed_worst"],
                                 per_source=s["per_source"], w2=s["w2"], w2_se=s["w2_se"], as_good=s["as_good"],
                                 passes=a.get("passes"), extra={"hdr_pass": a.get("hdr_pass"),
                                                                "gate_checks": d.get("gate_checks", {}).get(n, {}),
                                                                "within_reference": a.get("within_reference"),
                                                                "pooled": a.get("pooled")}, path=rel))
            out["E27"] = {"path": rel, "arms": arms, "adopted": d.get("adopt"), "status": d.get("status"),
                          "gate_checks": d.get("gate_checks")}
        elif key == "E28":
            for n, a in d["arms"].items():
                arms.append(_arm("E28", n, spec=a["spec"], signed=a["signed_mean"], se=a["signed_se"], worst=a["signed_worst"],
                                 per_source=a["per_source"], w2=a["w2"], w2_se=a["w2_se"], as_good=a["as_good"],
                                 passes=a.get("passes"), extra={"pooled": a.get("pooled"), "recipe_signal": a.get("recipe_signal")},
                                 path=rel))
            out["E28"] = {"path": rel, "arms": arms, "adopted": d.get("adopted"), "label": d.get("label"),
                          "pooled_rule": d.get("pooled_rule")}
        elif key == "E30":
            n = d["nA3"]
            arms.append(_arm("E30", "nA3", spec=n.get("spec"), signed=n["signed"]["mean"], se=n["signed"]["se"],
                             worst=n["signed"].get("worst"), per_source=d.get("per_source"),
                             w2=n.get("w2_type_worst3", {}).get("mean"), w2_se=n.get("w2_type_worst3", {}).get("se"),
                             extra={"report_only": True, "scope": d.get("scope")}, path=rel))
            out["E30"] = {"path": rel, "arms": arms, "adopted": None, "scope": d.get("scope"), "status": d.get("status")}
        elif key == "V40":
            if "assessments" not in d or "arms" not in d or "guard_rules" not in d:
                raise SourceShapeError(f"{rel}: expected assessments/arms/guard_rules")
            for n, a in d["arms"].items():
                rd = a.get("registered_decision") or {}
                exp = V40_STUDY.get(rd.get("study"))
                if exp is None:
                    raise SourceShapeError(f"{rel}: arm {n} has unknown study {rd.get('study')!r}")
                sg = rd.get("signed") or {}
                w2 = rd.get("w2") or {}
                arm = _arm(exp, n, signed=sg.get("delta"), se=sg.get("se"), per_source=rd.get("per_source"),
                           w2=w2.get("delta"), w2_se=w2.get("se"), as_good=rd.get("as_good"), verdict=a.get("verdict"),
                           reason=a.get("reason"), seed_deltas=sg.get("seed_deltas"),
                           extra={"guards": rd.get("guards"), "adopt": rd.get("adopt")}, path=rel)
                out.setdefault(exp, {"path": rel, "arms": [], "adopted": None, "guard_rules": d["guard_rules"]})["arms"].append(arm)
        elif key == "V40HDR":
            if d.get("study") not in (None, "e29") and "e29" not in str(d.get("study")):
                raise SourceShapeError(f"{rel}: expected the E29 HDR study, got {d.get('study')!r}")
            out.setdefault("E29", {"path": rel, "arms": [], "adopted": None})
            out["E29"]["adopted"] = d.get("adopt")
            out["E29"]["adopted_from"] = rel
        ctx.count(rel, len(arms) or len(d.get("arms", {})))
    return out


def _first_heading(text: str) -> tuple[int, str]:
    for n, lvl, t in headings(text):
        if lvl == 1:
            return n, t
    raise SourceShapeError("no level-1 heading")


def registry(ctx) -> list[dict]:
    exps = []
    # Rev4 program
    text = ctx.text(REV4_PROGRAM, "experiments")
    for n, lvl, t in headings(text):
        m = re.match(r"^(E\d+)\s+—\s+(.*)$", t)
        if lvl == 2 and m:
            exps.append({"scheme": "rev4", "id": f"rev4-{m.group(1)}", "label": f"Rev4 {m.group(1)}", "title": m.group(2),
                         "registration": REV4_PROGRAM, "line": n, "date": "2026-09-23"})
    if not any(e["scheme"] == "rev4" for e in exps):
        raise SourceShapeError(f"{REV4_PROGRAM}: no `## E<n> — ` headings")
    # featpot design log: script docstrings
    for path in ctx.glob(f"{FEATPOT_DIR}/e[0-9]*.py"):
        src = ctx.text(path, "experiments")
        doc = ast.get_docstring(ast.parse(src)) or ""
        first = doc.strip().splitlines()[0] if doc.strip() else ""
        m = re.match(r"^Design log (E\d+[′″]*)(?: method \d+)?(?:\s*\(([^)]*)\))?\s*[:—-]?\s*(.*)$", first)
        if not m:
            continue
        reg = re.search(r"registered (\d{4}-\d{2}-\d{2})", first)
        exps.append({"scheme": "featpot", "id": m.group(1).replace("′", "p").replace("″", "pp"), "label": m.group(1),
                     "title": doc.strip().split("\n\n")[0].replace("\n", " "), "registration": path, "line": 1,
                     "date": reg.group(1) if reg else None, "docstring": doc})
    # E25: Rev5 spec addendum D
    spec = "benchmarks/rev5_spec_2026-10-04.md"
    st = ctx.text(spec, "experiments")
    ln, body = section(st, r"^Addendum D .*E25", spec)
    exps.append({"scheme": "featpot", "id": "E25", "label": "E25", "title": "Rev4-trained weights at Rev5 arithmetic",
                 "registration": spec, "line": ln, "date": "2026-10-05", "rule_text": body.strip()})
    # E26–E33, KADID TERMINAL
    for path in ctx.glob("benchmarks/e[23][0-9]_*registration*.md") + ["benchmarks/kadid_terminal_registration_2026-10-07.md"]:
        txt = ctx.text(path, "experiments")
        ln, title = _first_heading(txt)
        m = re.match(r"^(E\d+|KADID TERMINAL)\s+—\s+(.*)$", title)
        if not m:
            raise SourceShapeError(f"{path}: title does not start with `E<n> — `")
        rule_line, rule = None, None
        for n, lvl, t in headings(txt):
            if re.search(r"decision rule|seed-paired statistic|assessment and limits|^9\. decision", t, re.I):
                rule_line, rule = section(txt, re.escape(t), path)
                break
        reg = re.search(r"(\d{4}-\d{2}-\d{2})", title) or re.search(r"(\d{4}-\d{2}-\d{2})", path)
        status = None
        sm = re.search(r"^Status:\s*(.*)$", txt, re.M)
        if sm:
            status = sm.group(1).strip()
        eid = m.group(1).replace(" ", "-")
        exps.append({"scheme": "featpot", "id": eid, "label": m.group(1), "title": re.sub(r"\s*\(registered[^)]*\)\s*$", "", m.group(2)),
                     "registration": path, "line": ln, "date": reg.group(1) if reg else None,
                     "rule_line": rule_line, "rule_text": rule.strip() if rule else None, "status_text": status})
    # de-duplicate featpot ids (a docstring E-number may also have a registration file)
    seen = {}
    for e in exps:
        k = (e["scheme"], e["id"])
        if k in seen and e["registration"].endswith(".md"):
            seen[k] = e
        elif k not in seen:
            seen[k] = e
    out = list(seen.values())
    if len([e for e in out if e["scheme"] == "featpot"]) < 15:
        raise SourceShapeError("featpot design log: fewer than 15 experiments found")
    return out
