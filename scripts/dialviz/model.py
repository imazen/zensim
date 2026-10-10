"""Assemble every reader's output into one model the page renderers consume."""
from __future__ import annotations

import ast
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import catalogue, experiments_src, feature_defs_src, featuresets_src, quotes, sources, zenanalyze_src
from .mdparse import SourceShapeError, table_with_headers

TERMINAL_OWNER = "scripts/rev4_featpot/_terminal_owner.py"
FREEZE_CHECK = "zensim-validate/src/bin/freeze_check.rs"
NEARID = "benchmarks/nearid_2026-10-09.md"
STEERFIX = "benchmarks/steerfix_2026-10-09.md"
SPEEDQ3 = "benchmarks/rev5_speedq3_2026-10-08.md"
V40_HDR = "benchmarks/v40_hdr_result_summary_2026-10-09.md"
V40_BORDA = "benchmarks/v40_hdr_borda_report_2026-10-09.json"

# Release-gate row -> names used by the other instruments for the same gate. Authored crosswalk;
# test_dialviz checks every name exists in its instrument and every instrument name is mapped.
CROSSWALK = {
    "e30-d1-population": {"terminal": [], "freeze": [], "contract": []},
    "table-provenance": {"terminal": ["Table provenance"], "freeze": ["Table provenance"], "contract": []},
    "rust-surface-final-identity": {"terminal": ["Rust surface"], "freeze": ["Rust surface"],
                                    "contract": ["Input and serving correctness"]},
    "g-rank-board-axes": {"terminal": ["G-RANK"], "freeze": ["G-RANK"], "contract": ["Human ranking"]},
    "g-dial": {"terminal": ["G-DIAL"], "freeze": ["G-DIAL"], "contract": ["Dial and addressability"]},
    "g-addr-five-codec-floors": {"terminal": ["G-ADDR", "codec floors"],
                                 "freeze": ["Ladder evidence", "G-ADDR contract", "G-ADDR regression", "Codec floor: avif-rav1e",
                                            "Codec floor: avif-svt", "Codec floor: jpeg", "Codec floor: jxl", "Codec floor: webp"],
                                 "contract": ["Dial and addressability"]},
    "negative-tails-identity": {"terminal": ["negative tails/identity"], "freeze": [], "contract": ["Dial and addressability"]},
    "g-steer": {"terminal": ["G-STEER"], "freeze": ["G-STEER"], "contract": ["Spatial value"]},
    "steercodec-chromaq": {"terminal": [], "freeze": [], "contract": []},
    "g-rd-spatial-value": {"terminal": ["G-RD"], "freeze": ["G-RD"], "contract": ["Spatial value"]},
    "g-target": {"terminal": ["G-TARGET"], "freeze": ["G-TARGET"],
                 "contract": ["One-shot targeting", "Two-shot targeting", "Three-shot targeting", "Target coverage and cost"]},
    "integrity-zcth-v4": {"terminal": ["integrity"], "freeze": [], "contract": ["Corruption"]},
    "rev5-correctness": {"terminal": ["Rev5 correctness"], "freeze": [], "contract": []},
    "runtime-memory-work-census": {"terminal": ["runtime/memory"], "freeze": [],
                                   "contract": ["Scalar performance", "Spatial and memory cost"]},
    "input-serving-bake-format-api": {"terminal": ["input/serving"], "freeze": [], "contract": ["Input and serving correctness"]},
    "hdr-scope": {"terminal": ["HDR scope"], "freeze": [], "contract": []},
    "kadid-terminal-d2": {"terminal": [], "freeze": [], "contract": []},
}


def terminal_gates(ctx) -> dict:
    src = ctx.text(TERMINAL_OWNER, "terminal_gates")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "GATES" for t in node.targets):
            vals = ast.literal_eval(node.value)
            if not (isinstance(vals, tuple) and all(isinstance(v, str) for v in vals) and len(vals) >= 10):
                raise SourceShapeError(f"{TERMINAL_OWNER}: GATES is not a tuple of >=10 names")
            ctx.count(TERMINAL_OWNER, len(vals))
            return {"path": TERMINAL_OWNER, "line": node.lineno, "gates": list(vals)}
    raise SourceShapeError(f"{TERMINAL_OWNER}: no top-level GATES assignment")


def freeze_check_gates(ctx) -> dict:
    src = ctx.text(FREEZE_CHECK, "freeze_check_gates")
    m = re.search(r"^fn qualification_report\(.*?^\}", src, re.M | re.S)
    if not m:
        raise SourceShapeError(f"{FREEZE_CHECK}: fn qualification_report not found")
    body = m.group(0)
    line = src[:m.start()].count("\n") + 1
    names: list[str] = []
    for mm in re.finditer(r'add\(\s*"([^"]+)"|\("(G-ADDR [a-z]+)",\s*"[a-z]+"\)|for codec in \[([^\]]+)\]|for gate in \[([^\]]+)\]', body):
        if mm.group(1):
            names.append(mm.group(1))
        elif mm.group(2):
            names.append(mm.group(2))
        elif mm.group(3):
            fmt = re.search(r'format!\("([^"{]*)\{codec\}"\)', body)
            if not fmt:
                raise SourceShapeError(f"{FREEZE_CHECK}: codec floor label format not found")
            names += [fmt.group(1) + c for c in re.findall(r'"([^"]+)"', mm.group(3))]
        elif mm.group(4):
            names += re.findall(r'"([^"]+)"', mm.group(4))
    if len(names) < 10:
        raise SourceShapeError(f"{FREEZE_CHECK}: only {len(names)} qualification checks parsed")
    floors = []
    fm = re.search(r"^mod balanced \{(.*?)^\}", src, re.M | re.S)
    if fm:
        for c in re.finditer(r"pub const ([A-Z0-9_]+): f64 = ([-\d.]+);\s*(?://\s*(.*))?", fm.group(1)):
            floors.append({"name": c.group(1), "value": float(c.group(2)), "comment": (c.group(3) or "").strip()})
    ctx.count(FREEZE_CHECK, len(names))
    return {"path": FREEZE_CHECK, "line": line, "gates": names, "floors": floors}


def nearid_table(ctx) -> dict:
    text = ctx.text(NEARID, "nearid")
    t = table_with_headers(text, NEARID, ["Model", "Highest nonidentical", "One-pixel ±1 range", "Nonincreasing ladders"])
    rows = []
    for r, ln in zip(t.rows, t.row_lines):
        def rng(s):
            a, b = re.split(r"\s*[–-]\s*", s.strip())
            return float(a), float(b)
        lo, hi = rng(r[t.col("One-pixel ±1 range")])
        glo, ghi = rng(r[t.col("Per-reference gap from 100")])
        num, den = r[t.col("Nonincreasing ladders")].split("/")
        rows.append({"model": r[0], "highest": float(r[t.col("Highest nonidentical")]), "onepx_lo": lo, "onepx_hi": hi,
                     "gap_lo": glo, "gap_hi": ghi, "ladders": int(num), "ladders_of": int(den), "line": ln})
    ctx.count(NEARID, len(rows))
    return {"path": NEARID, "line": t.line, "rows": rows}


def steerfix_table(ctx) -> dict:
    text = ctx.text(STEERFIX, "steerfix")
    t = table_with_headers(text, STEERFIX, ["Case", "Verdict", "Pre-floor + replay M2 / M3f", "Reason"])
    rows = []
    for r, ln in zip(t.rows, t.row_lines):
        m2, m3 = [float(x) for x in r[2].split("/")]
        rows.append({"case": r[0], "verdict": r[1], "m2": m2, "m3f": m3, "reason": r[3], "line": ln})
    ctx.count(STEERFIX, len(rows))
    return {"path": STEERFIX, "line": t.line, "rows": rows}


def speedq3_table(ctx) -> dict:
    text = ctx.text(SPEEDQ3, "speedq3")
    t = table_with_headers(text, SPEEDQ3, ["tier", "faster", "slower", "inconclusive"])
    rows = [{"tier": r[0], "faster": int(r[1]), "slower": int(r[2]), "inconclusive": int(r[3]), "line": ln}
            for r, ln in zip(t.rows, t.row_lines)]
    ctx.count(SPEEDQ3, len(rows))
    return {"path": SPEEDQ3, "line": t.line, "rows": rows}


def v40_hdr(ctx) -> dict:
    text = ctx.text(V40_HDR, "v40_hdr")
    t = table_with_headers(text, V40_HDR, ["Arm", "Endpoint", "Teacher", "Delta", "SE", "n"])
    rows = [{"arm": r[0], "endpoint": r[1], "teacher": r[2], "delta": float(r[3]), "se": float(r[4]), "n": int(r[5]), "line": ln}
            for r, ln in zip(t.rows, t.row_lines)]
    p = table_with_headers(text, V40_HDR, ["Arm", "HDR pass", "SDR as_good", "Combined passes"])
    passes = {r[0]: {"hdr": r[1] == "True", "sdr": r[2] == "True", "combined": r[3] == "True"} for r in p.rows}
    b = ctx.json(V40_BORDA, "v40_borda")
    if b.get("report_only") is not True or not isinstance(b.get("panels"), dict):
        raise SourceShapeError(f"{V40_BORDA}: expected report_only true and a panels object")
    panels = {arm: [float(v) for v in cells.values()] for arm, cells in b["panels"].items()}
    ctx.count(V40_HDR, len(rows))
    ctx.count(V40_BORDA, sum(len(v) for v in panels.values()))
    return {"path": V40_HDR, "line": t.line, "rows": rows, "passes": passes, "borda_path": V40_BORDA, "borda": panels}


def integrity_gates(ctx, path: Path | None) -> dict | None:
    """Optional: the SHIPPATH6 ZCTH v4 TRAIN refit GATES.json, which lives outside the repository."""
    if path is None or not path.is_file():
        return None
    rel = str(path)
    data = path.read_bytes()
    import hashlib
    ctx.read_log[rel] = {"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "readers": ["integrity_gates"], "entities": 0}
    d = json.loads(data)
    if not isinstance(d.get("gates"), dict) or not all(isinstance(v, bool) for v in d["gates"].values()):
        raise SourceShapeError(f"{rel}: gates must be an object of booleans")
    ctx.read_log[rel]["entities"] = len(d["gates"])
    return {"path": rel, "scope": d.get("scope"), "gates": d["gates"], "summary": d.get("summary", {})}


def git_head(repo: Path) -> str | None:
    """Commit of the checkout being read: jj's working-copy parent (works in secondary jj workspaces), else git HEAD."""
    for cmd in (["jj", "--ignore-working-copy", "log", "-r", "@-", "--no-graph", "-T", "commit_id"],
                ["git", "rev-parse", "HEAD"]):
        try:
            out = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
            if re.fullmatch(r"[0-9a-f]{40}", out):
                return out
        except (OSError, subprocess.CalledProcessError):
            continue
    return None


def dirty(repo: Path) -> bool | None:
    """True when the working copy holds changes beyond the reported commit."""
    for cmd in (["jj", "diff", "--name-only"], ["git", "status", "--porcelain", "--untracked-files=no"]):
        try:
            return bool(subprocess.run(cmd, cwd=repo, capture_output=True, text=True, check=True).stdout.strip())
        except (OSError, subprocess.CalledProcessError):
            continue
    return None


def build(ctx, integrity_path: Path | None = None) -> dict:
    m: dict = {"built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "commit": git_head(ctx.repo),
               "zenanalyze_commit": ctx.za_tree.commit if ctx.za_tree else git_head(ctx.zenanalyze),
               "zenanalyze_rev": ctx.zenanalyze_rev, "zenanalyze_date": ctx.za_tree.date if ctx.za_tree else None,
               "zenanalyze_checkout": git_head(ctx.zenanalyze), "dirty": dirty(ctx.repo)}
    m["gates"] = sources.release_gates(ctx)
    m["scorecard"] = sources.scorecard(ctx)
    m["bugs"] = sources.known_bugs(ctx)
    m["peers"] = sources.paper_gates(ctx)
    m["splits"] = sources.data_splits(ctx)
    m["terminal"] = terminal_gates(ctx)
    m["freeze"] = freeze_check_gates(ctx)
    m["nearid"] = nearid_table(ctx)
    m["steerfix"] = steerfix_table(ctx)
    m["speedq3"] = speedq3_table(ctx)
    m["integrity"] = integrity_gates(ctx, integrity_path)
    m["v40_hdr"] = v40_hdr(ctx)
    m["results"] = experiments_src.results(ctx)
    m["experiments"] = experiments_src.registry(ctx)
    m["features"] = feature_defs_src.read(ctx)
    m["featuresets"] = featuresets_src.registry(ctx)
    m["named_sets"] = featuresets_src.named_sets(ctx)
    m["za"] = zenanalyze_src.catalogue(ctx)
    m["pickers"] = zenanalyze_src.pickers(ctx, m["za"])
    m["crosswalk"] = CROSSWALK
    gate_ids = {g["id"] for g in m["gates"]["gates"]}
    if set(CROSSWALK) != gate_ids:
        raise SourceShapeError(f"crosswalk/release-gate mismatch: {sorted(set(CROSSWALK) ^ gate_ids)}")

    def resolve_all(p, keys):
        out = dict(p)
        for k in keys:
            out[k] = [quotes.resolve(ctx, loc) for loc in p.get(k, [])]
        return out

    m["wanted"] = [resolve_all(p, ("rules", "owners", "state")) for p in catalogue.WANTED]
    m["unwanted"] = [resolve_all(p, ("evidence",)) for p in catalogue.UNWANTED]
    for p in m["unwanted"]:
        rx = re.compile(p["bugs"], re.I)
        p["bug_hits"] = [b for b in m["bugs"] if rx.search(b["title"])]
    for p in m["wanted"] + m["unwanted"]:
        for gid in p.get("gates", []) + p.get("catches", []):
            if gid not in gate_ids:
                raise SourceShapeError(f"catalogue property {p['id']} names unknown gate {gid}")
    m["evaluation"] = {k: quotes.resolve(ctx, loc) for k, loc in catalogue.EVALUATION.items()}
    m["sources"] = sorted(ctx.read_log.values(), key=lambda e: e["path"])
    return m
