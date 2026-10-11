"""Release readiness of the current candidate: measured gate states and what blocks the rest.

Reads QUAL-A (the release gates run on E33 candidate A, with C and seed 0 alongside) and the later
STEERPATH landed-state re-measurement, which supersedes QUAL-A's spatial-cost and memory rows. The
release gate map (`sources.release_gates`) is an earlier snapshot; this module is the current state.

Blocker root causes are assigned by fixed patterns over each not-run gate's "Needs" text (`CAUSES`);
the matched phrase is kept as evidence, and a gate that matches no pattern refuses the build.
"""
from __future__ import annotations

import re

from .mdparse import SourceShapeError, section, table_with_headers

QUAL_A_JSON = "benchmarks/qual_a_2026-10-10.json"
QUAL_A_MD = "benchmarks/qual_a_2026-10-10.md"
STEERPATH = "benchmarks/steerpath_perf_2026-10-10.md"
MODELS = ("a", "c", "seed0")
MODEL_LABEL = {"a": "A (candidate)", "c": "C", "seed0": "seed 0 (production)"}

# Display groups for the measured gates, matched on the gate name in order.
FAMILIES = [
    ("Build and surface", r"^(Table provenance|Rust surface|ZCTH)"),
    ("Dial and contract", r"^(G-DIAL|C\d|Negative tails)"),
    ("Identity", r"^(Identity|Near-identity)"),
    ("Steering", r"^G-STEER"),
    ("Runtime and memory", r"^(Scalar performance|Spatial cost|Memory)"),
]

# Root causes behind the gates that could not run. Pattern over the "Needs" cell; a gate may have several.
CAUSES = [
    ("owner", "Owner rule or decision", r"\bowner\b[^.;]*"),
    ("composition", "Final model frozen and bound", r"(?:frozen|final) (?:composition|model|bytes|receipt)[^.;]*|final shipping[^.;]*"),
    ("data", "Authorized human or EVAL data", r"[^.;]*(?:exposure|EVAL (?:root|corruption|corpora)|human EVAL|independent human|human HDR)[^.;]*"),
    ("binding", "Instrument registration and binding", r"[^.;]*(?:registration|bound via|bound to|parameteri[sz]ed|instrumentation|packet|bundle)[^.;]*"),
    ("authorization", "Authorization to read or touch", r"[^.;]*authori[sz]ation[^.;]*"),
    ("release", "Release mechanics", r"[^.;]*(?:Platform CI|README|publication)[^.;]*"),
]

CAUSE_SHORT = {"owner": "Owner", "composition": "Final model", "data": "Human/EVAL data", "binding": "Binding work",
               "authorization": "Authorization", "release": "Release"}


# The not-run "State" cell is a ';'-separated list of these words; anything else refuses the build.
# A gate's band is the first listed kind it carries, in this order.
NOT_RUN_STATES = [("forbidden", ("forbidden in this lane", "forbidden")), ("blocked", ("blocked",)),
                  ("rule_pending", ("rule pending",)), ("not_measured", ("not measured",)), ("outstanding", ("outstanding",))]


def _not_run_kind(state: str, gate: str) -> tuple[str, set]:
    words = set()
    for token in (t.strip().lower() for t in state.split(";")):
        hit = next((k for k, spellings in NOT_RUN_STATES if token in spellings), None)
        if hit is None:
            raise SourceShapeError(f"{QUAL_A_MD}: not-run gate {gate!r} has unrecognised state {token!r}")
        words.add(hit)
    if not words:
        raise SourceShapeError(f"{QUAL_A_MD}: not-run gate {gate!r} has an empty state")
    return next(k for k, _ in NOT_RUN_STATES if k in words), words


def _family(name: str) -> str:
    for fam, rx in FAMILIES:
        if re.search(rx, name):
            return fam
    raise SourceShapeError(f"{QUAL_A_JSON}: gate {name!r} matches no display family")


def _steerpath(ctx) -> dict:
    """STEERPATH's landed-state table: {(size, model): {spatial, spatial_state, rss, rss_bar, rss_state}}."""
    text = ctx.text(STEERPATH, "readiness")
    ln, body = section(text, r"^Landed state", STEERPATH)
    t = table_with_headers(body, STEERPATH, ["size", "model", "spatial, map p95 / uncached p95"])
    out = {}
    names = {"A": "a", "C": "c", "seed 0": "seed0"}
    for row, line in zip(t.rows, t.row_lines):
        if len(row) != 7 or row[1] not in names or row[3] not in ("PASS", "FAIL") or row[6] not in ("PASS", "FAIL"):
            raise SourceShapeError(f"{STEERPATH}: landed-state row {row!r} has an unexpected shape")
        out[(row[0], names[row[1]])] = {"spatial": row[2], "spatial_state": row[3].lower(), "rss": row[4],
                                        "rss_bar": row[5], "rss_state": row[6].lower(), "line": ln + line}
    if len(out) != 6:
        raise SourceShapeError(f"{STEERPATH}: expected 6 landed-state rows (2 sizes x 3 models), got {len(out)}")
    ctx.count(STEERPATH, len(out))
    return {"rows": out, "line": ln}


def read(ctx) -> dict:
    d = ctx.json(QUAL_A_JSON, "readiness")
    if set(d.get("models", {})) != set(MODELS) or not isinstance(d.get("gates"), dict):
        raise SourceShapeError(f"{QUAL_A_JSON}: expected models {MODELS} and a gates object")
    sp = _steerpath(ctx)
    gates = []
    for name, g in d["gates"].items():
        if "bar" not in g or any(not isinstance(g.get(mk), dict) or g[mk].get("state") not in ("pass", "fail") for mk in MODELS):
            raise SourceShapeError(f"{QUAL_A_JSON}: gate {name!r} lacks a bar or a pass/fail state per model")
        states = {mk: {"state": g[mk]["state"], "detail": g[mk]["detail"], "source": QUAL_A_JSON} for mk in MODELS}
        sm = re.match(r"^(Spatial cost|Memory) (\d+)x\2\b", name)
        superseded = None
        if sm:
            size = f"{sm.group(2)}²"
            for mk in MODELS:
                r = sp["rows"].get((size, mk))
                if r is None:
                    raise SourceShapeError(f"{STEERPATH}: no landed-state row for {size} {mk}")
                if sm.group(1) == "Spatial cost":
                    states[mk] = {"state": r["spatial_state"], "detail": r["spatial"], "source": STEERPATH, "line": r["line"],
                                  "before": g[mk]["detail"]}
                else:
                    qbar = re.search(r"= ([\d,]+) KiB", g["bar"])
                    if not qbar or qbar.group(1).replace(",", "") != r["rss_bar"].replace(",", ""):
                        raise SourceShapeError(f"{STEERPATH}: {size} {mk} memory bar {r['rss_bar']!r} differs from "
                                               f"QUAL-A's bar {g['bar']!r}")
                    states[mk] = {"state": r["rss_state"], "detail": f'score+map {r["rss"]} KiB (bar {r["rss_bar"]})',
                                  "source": STEERPATH, "line": r["line"], "before": g[mk]["detail"]}
            superseded = STEERPATH
        gates.append({"name": name, "bar": g["bar"], "family": _family(name), "states": states, "superseded_by": superseded})
    ctx.count(QUAL_A_JSON, len(gates))

    text = ctx.text(QUAL_A_MD, "readiness")
    ln, body = section(text, r"^Gates not run", QUAL_A_MD)
    t = table_with_headers(body, QUAL_A_MD, ["Gate", "State", "Needs"])
    not_run = []
    for row, line in zip(t.rows, t.row_lines):
        if len(row) != 3:
            raise SourceShapeError(f"{QUAL_A_MD}: not-run row {row!r} is not Gate | State | Needs")
        gate, state, needs = row
        causes = []
        for cid, label, rx in CAUSES:
            hit = re.search(rx, needs, re.I)
            if hit:
                causes.append({"id": cid, "label": label, "evidence": hit.group(0).strip(" ,")})
        if not causes:
            raise SourceShapeError(f"{QUAL_A_MD}: not-run gate {gate!r} matches no root-cause pattern")
        kind, words = _not_run_kind(state, gate)
        not_run.append({"gate": gate, "state": state, "kind": kind, "rule_pending": "rule_pending" in words, "needs": needs,
                        "causes": causes, "line": ln + line})
    if len(not_run) < 5:
        raise SourceShapeError(f"{QUAL_A_MD}: fewer than 5 not-run gates parsed")
    ctx.count(QUAL_A_MD, len(not_run))
    return {"path": QUAL_A_JSON, "md": QUAL_A_MD, "steerpath": STEERPATH, "models": d["models"], "gates": gates,
            "not_run": not_run, "not_run_line": ln, "causes": [{"id": c, "label": l} for c, l, _ in CAUSES]}


# Every release-gate-map row (exact bold name) and the QUAL-A gates that cover it: (map row, measured-gate patterns,
# not-run-gate patterns). A row with neither is shown on the board as "in the gate map, not in QUAL-A". The build
# refuses an unknown map row, a missing map row, a pattern that matches nothing, or a QUAL-A gate no row covers.
GATE_MAP_COVERAGE = [
    ("E30 / D1 population", [], []),
    ("Table provenance", [r"^Table provenance"], [r"^Table provenance"]),
    ("Rust surface / final identity", [r"^Rust surface", r"^Identity"], []),
    ("G-RANK / board axes", [], [r"^G-RANK"]),
    ("G-DIAL", [r"^G-DIAL"], []),
    ("G-ADDR / five codec floors", [], [r"^G-ADDR"]),
    ("Negative tails / identity", [r"^Negative tails", r"^C\d", r"^Identity", r"^Near-identity"], []),
    ("G-STEER", [r"^G-STEER"], []),
    ("STEERCODEC / CHROMAQ", [], [r"^STEERCODEC"]),
    ("G-RD / spatial value", [], [r"^G-RD"]),
    ("G-TARGET", [], [r"^G-TARGET"]),
    ("Integrity / ZCTH v4", [r"^ZCTH v4"], [r"^Integrity detection"]),
    ("Rev5 correctness", [], []),
    ("Runtime/memory / work census", [r"^(Scalar performance|Spatial cost|Memory)"], [r"^Runtime work census"]),
    ("Input/serving / bake-format / API", [], []),
    ("HDR scope", [], [r"^HDR scope"]),
    ("KADID TERMINAL / D2", [], [r"^KADID TERMINAL"]),
]

# QUAL-A gates with no gate-map row at all (they stay on the board in their own band).
QUAL_A_ONLY = [r"^Release mechanics"]


def coverage(r: dict, gate_map: dict) -> list[dict]:
    """Match every release-gate-map row to the QUAL-A gates that cover it (see GATE_MAP_COVERAGE)."""
    rows = {g["name"]: g for g in gate_map["gates"]}
    known = [name for name, _, _ in GATE_MAP_COVERAGE]
    for name in rows:
        if name not in known:
            raise SourceShapeError(f"{gate_map['path']}: gate-map row {name!r} has no GATE_MAP_COVERAGE entry")
    measured = [g["name"] for g in r["gates"]]
    not_run = [n["gate"] for n in r["not_run"]]
    used_m, used_n, out = set(), set(), []
    for name, mpats, npats in GATE_MAP_COVERAGE:
        if name not in rows:
            raise SourceShapeError(f"{gate_map['path']}: GATE_MAP_COVERAGE names {name!r}, which the gate map no longer has")
        hits_m, hits_n = [], []
        for pats, pool, hits, used in ((mpats, measured, hits_m, used_m), (npats, not_run, hits_n, used_n)):
            for pat in pats:
                found = [x for x in pool if re.search(pat, x)]
                if not found:
                    raise SourceShapeError(f"GATE_MAP_COVERAGE {name!r}: pattern {pat!r} matches no QUAL-A gate")
                hits.extend(x for x in found if x not in hits)
                used.update(found)
        g = rows[name]
        out.append({"name": name, "line": g["line"], "map_status": g["status"], "status_text": g["status_text"],
                    "measured": hits_m, "not_run": hits_n, "in_qual_a": bool(hits_m or hits_n)})
    for pat in QUAL_A_ONLY:
        found = [x for x in measured + not_run if re.search(pat, x)]
        if not found:
            raise SourceShapeError(f"QUAL_A_ONLY pattern {pat!r} matches no QUAL-A gate")
        used_m.update(x for x in found if x in measured)
        used_n.update(x for x in found if x in not_run)
    for pool, used, what in ((measured, used_m, "measured"), (not_run, used_n, "not-run")):
        for x in pool:
            if x not in used:
                raise SourceShapeError(f"QUAL-A {what} gate {x!r} is covered by no gate-map row in GATE_MAP_COVERAGE")
    return out


def blocked_breakdown(cov: list[dict]) -> dict:
    """The gate map's blocked rows, by what QUAL-A now says about them."""
    blocked = [c for c in cov if c["map_status"] == "blocked"]
    return {"blocked": len(blocked),
            "measured": [c["name"] for c in blocked if c["measured"]],
            "not_run_only": [c["name"] for c in blocked if not c["measured"] and c["not_run"]],
            "not_in_qual_a": [c["name"] for c in blocked if not c["in_qual_a"]]}


# Numeric checks drawn as bullet rows: (gate name regex, label, value regex on the detail, bar regex on the bar text,
# direction). Values and bars are both read from the record; nothing is recomputed.
BULLETS = [
    (r"^G-STEER", "G-STEER cases passed", r"^(\d+)/\d+", r"all (\d+) pass", "ge", ""),
    (r"^G-DIAL", "G-DIAL p5", r"p5 ([\d.]+)", r"p5 ≤ ([\d.]+)", "le", ""),
    (r"^G-DIAL", "G-DIAL p95", r"p95 ([\d.]+)", r"p95 ≥ ([\d.]+)", "ge", ""),
    (r"^G-DIAL", "G-DIAL monotonicity", r"mono ([\d.]+)", r"monotonicity ≥ (\.\d+|[\d.]+)", "ge", ""),
    (r"^C2 ", "C2 ladder ties", r"/ ([\d.]+)$", r"ties ≤ (\.\d+|[\d.]+)", "le", ""),
    (r"^Near-identity", "N1 one-pixel minimum", r"one-pixel min ([\d.]+)", r"one-pixel ≥ ([\d.]+)", "ge", ""),
    (r"^Near-identity", "N3 nonincreasing ladders", r"ladders (\d+)/", r"≥ (\d+)/\d+ ladders", "ge", ""),
    (r"^Scalar performance 1024", "Uncached p95, 1024²", r"^([\d.]+) ms", r"≤ ([\d.]+) ms", "le", " ms"),
    (r"^Scalar performance 2048", "Uncached p95, 2048²", r"^([\d.]+) ms", r"≤ ([\d.]+) ms", "le", " ms"),
    (r"^Spatial cost 1024", "Score+map ÷ uncached, 1024²", r"= ([\d.]+)×", r"≤ ([\d.]+)×", "le", "×"),
    (r"^Spatial cost 2048", "Score+map ÷ uncached, 2048²", r"= ([\d.]+)×", r"≤ ([\d.]+)×", "le", "×"),
    (r"^Memory 1024", "Map peak RSS, 1024² (KiB)", r"score\+map ([\d,]+) KiB", r"= ([\d,]+) KiB", "le", ""),
    (r"^Memory 2048", "Map peak RSS, 2048² (KiB)", r"score\+map ([\d,]+) KiB", r"= ([\d,]+) KiB", "le", ""),
]


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def bullets(r: dict) -> list[dict]:
    out = []
    for grx, label, vrx, brx, direction, unit in BULLETS:
        g = next((x for x in r["gates"] if re.search(grx, x["name"])), None)
        if g is None:
            raise SourceShapeError(f"readiness bullet {label!r}: no gate matches /{grx}/")
        bm = re.search(brx, g["bar"])
        if not bm:
            raise SourceShapeError(f"readiness bullet {label!r}: bar {g['bar']!r} lacks /{brx}/")
        vals = {}
        for mk in MODELS:
            vm = re.search(vrx, g["states"][mk]["detail"])
            if not vm:
                raise SourceShapeError(f"readiness bullet {label!r}: {mk} detail {g['states'][mk]['detail']!r} lacks /{vrx}/")
            vals[mk] = _num(vm.group(1))
        bar = _num(bm.group(1))
        # per-check state: the recorded value against the recorded bar; the gate's own state can combine several checks
        out.append({"label": label, "gate": g["name"], "bar": bar, "dir": direction, "unit": unit, "values": vals,
                    "states": {mk: "pass" if (v <= bar if direction == "le" else v >= bar) else "fail" for mk, v in vals.items()},
                    "gate_states": {mk: g["states"][mk]["state"] for mk in MODELS}, "source": g["states"]["a"]["source"]})
    return out
