"""Readers for zenanalyze's feature catalogue and the picker artifacts that pin it.

Sources live in the sibling zenanalyze checkout and are only read:
  src/feature.rs                         features_table! rows and tier FeatureSet consts
  benchmarks/feature_qualified_names.tsv current `name@hex8` identities (golden-tested there)
  Cargo.toml                             crate version
  zenpicker/benchmarks/*.bin             ZNPR v3 routers (metadata key zentrain.feature_columns)
  benchmarks/metapicker_v1_feature_slots_*.tsv, benchmarks/*.manifest.json, zentrain/examples/*.py
"""
from __future__ import annotations

import ast
import hashlib
import re
import struct

from .mdparse import SourceShapeError

FEATURE_RS = "zenanalyze:src/feature.rs"
QUALIFIED = "zenanalyze:benchmarks/feature_qualified_names.tsv"

_ROW = re.compile(r"^\s*([A-Z][A-Za-z0-9]*)\s*=\s*(\d+)\s*:\s*(\w+)\s*=>\s*([a-z0-9_]+)\s*,\s*$")


def catalogue(ctx) -> dict:
    text = ctx.text(FEATURE_RS, "zenanalyze_catalogue")
    m = re.search(r"^features_table!\s*\{\s*$", text, re.M)
    if not m:
        raise SourceShapeError(f"{FEATURE_RS}: no `features_table! {{` invocation")
    start_line = text[:m.start()].count("\n") + 1
    lines = text.splitlines()
    feats = []
    doc: list[str] = []
    attrs: list[str] = []
    header = None
    in_decl = 0
    for n in range(start_line, len(lines)):
        raw = lines[n]
        s = raw.strip()
        if s == "}" and not in_decl:
            break
        if in_decl:
            attrs[-1] += " " + s
            in_decl += s.count("(") - s.count(")")
            continue
        if s.startswith("// ----") or s.startswith("// ===="):
            header = s.strip("/ -=").strip()
            continue
        if s.startswith("///"):
            doc.append(s[3:].strip())
            continue
        if s.startswith("#[") or s.startswith("@decl"):
            attrs.append(s)
            in_decl = s.count("(") - s.count(")") if s.startswith("@decl") else 0
            continue
        mm = _ROW.match(raw)
        if mm:
            a = " ".join(attrs)
            cfg = re.findall(r'cfg\(feature\s*=\s*"([^"]+)"\)', a)
            dep = re.search(r'deprecated\(\s*since\s*=\s*"([^"]+)"\s*,\s*note\s*=\s*"(.*?)"\s*\)', a, re.S)
            feats.append({"variant": mm.group(1), "id": int(mm.group(2)), "ty": mm.group(3), "name": mm.group(4),
                          "doc": " ".join(doc), "cfg": cfg, "deprecated": bool(dep),
                          "deprecated_note": re.sub(r"\\\s+", "", dep.group(2)) if dep else None,
                          "section": header, "line": n + 1})
            doc, attrs = [], []
            continue
        if s and not s.startswith("//"):
            raise SourceShapeError(f"{FEATURE_RS}:{n + 1}: unrecognised features_table! line {s[:80]!r}")
    if len(feats) < 100:
        raise SourceShapeError(f"{FEATURE_RS}: only {len(feats)} features_table! rows parsed")
    ids = [f["id"] for f in feats]
    if len(set(ids)) != len(ids):
        raise SourceShapeError(f"{FEATURE_RS}: duplicate feature ids")
    # tier sets
    tiers: dict[str, list[str]] = {}
    for mm in re.finditer(r"^pub\(crate\) const ([A-Z0-9_]+): FeatureSet = (\{.*?^\};|FeatureSet::new\(\)[^;]*;|[^;]*;)",
                          text, re.M | re.S):
        body = mm.group(2)
        tiers[mm.group(1)] = re.findall(r"AnalysisFeature::([A-Za-z0-9]+)", body)
    for tname in ("TIER1_FULL_FEATURES", "TIER2_FEATURES", "TIER3_FEATURES", "PALETTE_FULL_FEATURES"):
        if not tiers.get(tname):
            raise SourceShapeError(f"{FEATURE_RS}: tier set {tname} missing or empty")
    unions = {}
    for mm in re.finditer(r"^pub\(crate\) const ([A-Z0-9_]+): FeatureSet = ([A-Z0-9_]+)\.union\(([A-Z0-9_]+)\);", text, re.M):
        unions[mm.group(1)] = [mm.group(2), mm.group(3)]
    by_variant = {f["variant"]: f for f in feats}
    for tname, members in tiers.items():
        for v in members:
            if v in by_variant:
                by_variant[v].setdefault("tiers", []).append(tname)
    retired = []
    rm = re.search(r"const RESERVED_RETIRED_IDS: &\[u16\] = &\[(.*?)\];", text, re.S)
    if not rm:
        raise SourceShapeError(f"{FEATURE_RS}: RESERVED_RETIRED_IDS not found")
    for line in rm.group(1).splitlines():
        code = line.split("//")[0]
        retired += [int(x) for x in re.findall(r"\d+", code)]
    rm_line = text[:rm.start()].count("\n") + 1
    # qualified identities
    qual = {}
    for n, line in enumerate(ctx.text(QUALIFIED, "zenanalyze_catalogue").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2 or not re.fullmatch(r"[a-z0-9_]+@[0-9a-f]{8}", parts[1]) or parts[1].split("@")[0] != parts[0]:
            raise SourceShapeError(f"{QUALIFIED}:{n}: expected `name<TAB>name@hex8`")
        qual[parts[0]] = parts[1]
    for f in feats:
        f["qualified"] = qual.get(f["name"])
    missing_q = [f["name"] for f in feats if f["qualified"] is None]
    cargo = ctx.text("zenanalyze:Cargo.toml", "zenanalyze_catalogue")
    vm = re.search(r'^version\s*=\s*"([^"]+)"', cargo, re.M)
    lib = ctx.text("zenanalyze:src/lib.rs", "zenanalyze_catalogue")
    fdv = re.search(r"FEATURE_DEFS_VERSION:\s*u32\s*=\s*(\d+)", lib)
    ctx.count(FEATURE_RS, len(feats))
    ctx.count(QUALIFIED, len(qual))
    return {"features": sorted(feats, key=lambda f: f["id"]), "tiers": tiers, "unions": unions,
            "retired_ids": sorted(set(retired)), "retired_line": rm_line, "qualified": qual,
            "missing_qualified": missing_q, "version": vm.group(1) if vm else None,
            "feature_defs_version": int(fdv.group(1)) if fdv else None, "table_line": start_line}


# --------------------------------------------------------------------------- picker pins

def _znpr_utf8(data: bytes, key: str, rel: str) -> str:
    if data[:4] != b"ZNPR":
        raise SourceShapeError(f"{rel}: not a ZNPR file")
    k = key.encode()
    i = data.find(bytes([len(k)]) + k)
    if i < 0:
        raise SourceShapeError(f"{rel}: metadata key {key} not found")
    pos = i + 1 + len(k)
    kind = data[pos]
    (vlen,) = struct.unpack_from("<I", data, pos + 1)
    if kind != 1:
        raise SourceShapeError(f"{rel}: metadata {key} has wire type {kind}, expected 1 (utf8)")
    return data[pos + 5:pos + 5 + vlen].decode("utf-8")


def _drift(pins: list[str], cat: dict) -> list[dict]:
    """Classify each pinned feature against the current catalogue."""
    names = {f["name"] for f in cat["features"]}
    out = []
    for p in pins:
        bare = p[5:] if p.startswith("feat_") else p
        if "@" in bare:
            nm, h = bare.split("@", 1)
            cur = cat["qualified"].get(nm)
            if nm not in names:
                st = "missing"
            elif cur != bare:
                st = "hash-changed"
            else:
                st = "current"
            out.append({"pin": p, "name": nm, "state": st, "current": cur})
        else:
            if bare in names:
                st = "current-unversioned"
            elif re.fullmatch(r"\d+", bare):
                st = "positional"
            else:
                st = "not-a-catalogue-feature"
            out.append({"pin": p, "name": bare, "state": st, "current": cat["qualified"].get(bare)})
    return out


def pickers(ctx, cat: dict) -> list[dict]:
    out = []
    for rel in ctx.glob("zenanalyze:zenpicker/benchmarks/*.bin"):
        stem = rel.rsplit("/", 1)[1].rsplit(".", 1)[0]
        data = ctx.bytes(rel, "zenanalyze_pickers")
        try:
            cols = _znpr_utf8(data, "zentrain.feature_columns", rel).split("\n")
        except SourceShapeError as e:
            out.append({"name": stem, "path": rel, "kind": "ZNPR router", "pins": [], "error": str(e)})
            continue
        cols = [c for c in cols if c]
        ctx.count(rel, len(cols))
        out.append({"name": stem, "path": rel, "kind": "ZNPR router (shipped via include_bytes!)",
                    "pins": _drift(cols, cat)})
    for rel in ctx.glob("zenanalyze:benchmarks/metapicker_v1_feature_slots_*.tsv"):
        stem = rel.rsplit("/", 1)[1].rsplit(".", 1)[0]
        pins = []
        for line in ctx.text(rel, "zenanalyze_pickers").splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if parts[0] == "slot":
                continue
            if len(parts) < 3:
                raise SourceShapeError(f"{rel}: expected slot<TAB>input_name<TAB>zenanalyze_feature")
            if parts[2] and parts[2] != "-":
                pins.append(parts[2])
        ctx.count(rel, len(pins))
        out.append({"name": stem, "path": rel, "kind": "metapicker slot map (bake off-git)", "pins": _drift(pins, cat)})
    for rel in ctx.glob("zenanalyze:benchmarks/*.manifest.json"):
        d = ctx.json(rel, "zenanalyze_pickers")
        cols = d.get("feat_cols") or d.get("feature_columns")
        if not isinstance(cols, list):
            continue
        ctx.count(rel, len(cols))
        out.append({"name": rel.rsplit("/", 1)[1].replace(".manifest.json", ""), "path": rel, "kind": "legacy picker manifest",
                    "pins": _drift(cols, cat)})
    literal: dict[str, list[str]] = {}
    parsed: dict[str, tuple[str, ast.Module]] = {}
    for rel in ctx.glob("zenanalyze:zentrain/examples/*.py"):
        try:
            parsed[rel.rsplit("/", 1)[1][:-3]] = (rel, ast.parse(ctx.text(rel, "zenanalyze_pickers")))
        except SyntaxError:
            continue

    def assigned(tree, name):
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return node
        return None

    def str_list(node):
        if isinstance(node, ast.List) and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
            return [e.value for e in node.elts]
        return None

    for stem, (rel, tree) in parsed.items():
        node = assigned(tree, "KEEP_FEATURES")
        if node is not None and str_list(node.value) is not None:
            literal[stem] = str_list(node.value)
    for stem, (rel, tree) in parsed.items():
        node = assigned(tree, "KEEP_FEATURES")
        if node is None:
            continue
        cols, kind = None, None
        if stem in literal:
            cols, kind = literal[stem], "zentrain config (literal KEEP_FEATURES)"
        else:
            base = None
            for imp in tree.body:
                if isinstance(imp, ast.ImportFrom) and any(a.name == "KEEP_FEATURES" and a.asname == "BASE_KEEP_FEATURES"
                                                           for a in imp.names):
                    base = literal.get(imp.module)
            v = node.value
            drop = assigned(tree, "DROP_FEATURES")
            if base is not None and isinstance(v, ast.ListComp) and drop is not None and str_list(drop.value) is not None:
                d = set(str_list(drop.value))
                cols, kind = [f for f in base if f not in d], "zentrain config (base KEEP_FEATURES minus DROP_FEATURES)"
            elif (base is not None and isinstance(v, ast.BinOp) and isinstance(v.op, ast.Add)
                  and str_list(v.right) is not None):
                cols, kind = list(base) + str_list(v.right), "zentrain config (base KEEP_FEATURES plus additions)"
            elif isinstance(v, (ast.Call, ast.IfExp)):
                kind = "zentrain config (dynamic: reads the features TSV header at run time)"
            else:
                kind = "zentrain config (derived KEEP_FEATURES, not resolved)"
        if cols is not None:
            ctx.count(rel, len(cols))
        out.append({"name": stem, "path": rel, "kind": kind, "pins": _drift(cols, cat) if cols else [], "line": node.lineno})
    for rel in ctx.glob("zenanalyze:zentrain/testdata/*.manifest.json"):
        d = ctx.json(rel, "zenanalyze_pickers")
        cols = d.get("feat_cols") or d.get("feature_columns")
        if isinstance(cols, list):
            ctx.count(rel, len(cols))
            out.append({"name": rel.rsplit("/", 1)[1].replace(".manifest.json", ""), "path": rel, "kind": "zentrain test manifest",
                        "pins": _drift(cols, cat)})
    if not any(o["kind"].startswith("ZNPR") and o["pins"] for o in out):
        raise SourceShapeError("zenpicker routers: no ZNPR router with feature_columns found")
    return out
