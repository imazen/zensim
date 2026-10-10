"""Reader for zensim's feature registry (`zensim/src/feature_defs.rs`).

The registry is `pub(crate)` Rust with no dump tool, so this reader evaluates
the `SignalDef` constructor calls the way the compiler would for the fields
the site shows: each `const fn` constructor's `SignalDef { .. }` body maps its
parameters to fields (with `..base` spreads and delegating calls followed),
`use` aliases resolve to enum variants, and `BLOCKS` plus the replication
arithmetic (`BlockDef::width`, `def_at`) place every signal at its slot ids.

Anything the reader does not understand raises `SourceShapeError`. The tests
cross-check its output against independent sources (registry JSON slot ranges,
registered layout widths, the E33 `fx1` table).
"""
from __future__ import annotations

import re

from .mdparse import SourceShapeError

FEATURE_DEFS = "zensim/src/feature_defs.rs"
N_SCALES = 4
CHANNELS = ["x", "y", "b"]


def strip_comments(src: str) -> str:
    """Remove // and /* */ comments, keeping string literals and line structure."""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1])
            i = j + 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i)
            out.append("\n" * src[i:j].count("\n"))
            i = n if j < 0 else j + 2
        elif c == "'" and i + 2 < n and src[i + 2] == "'":
            out.append(src[i:i + 3])
            i += 3
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _match_close(s: str, i: int) -> int:
    """Index of the bracket closing s[i] (one of ({[), skipping strings."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = [pairs[s[i]]]
    j = i + 1
    while j < len(s):
        c = s[j]
        if c == '"':
            j += 1
            while s[j] != '"':
                j += 2 if s[j] == "\\" else 1
        elif c in pairs:
            stack.append(pairs[c])
        elif c in ")]}":
            if c != stack.pop():
                raise SourceShapeError(f"{FEATURE_DEFS}: unbalanced bracket near offset {j}")
            if not stack:
                return j
        j += 1
    raise SourceShapeError(f"{FEATURE_DEFS}: unterminated bracket at offset {i}")


def _split_top(s: str, sep: str = ",") -> list[str]:
    parts, depth, cur, i = [], 0, [], 0
    while i < len(s):
        c = s[i]
        if c == '"':
            j = i + 1
            while s[j] != '"':
                j += 2 if s[j] == "\\" else 1
            cur.append(s[i:j + 1])
            i = j + 1
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


_FN = re.compile(r"\bconst fn ([a-z0-9_]+)\s*\(")


def _parse_fns(code: str) -> dict:
    """const fn name -> {params: [...], body: str} for fns returning SignalDef."""
    fns = {}
    for m in _FN.finditer(code):
        po = code.index("(", m.end() - 1)
        pc = _match_close(code, po)
        rest = code[pc + 1:]
        rm = re.match(r"\s*->\s*SignalDef\s*\{", rest)
        if not rm:
            continue
        bo = pc + 1 + rm.end() - 1
        bc = _match_close(code, bo)
        params = [p.split(":")[0].strip() for p in _split_top(code[po + 1:pc]) if p.strip()]
        fns[m.group(1)] = {"params": params, "body": code[bo + 1:bc].strip(), "span": (m.start(), bc)}
    return fns


def _uses(code: str) -> dict:
    """`use A::B as C;` / `use A::{B, C as D};` -> alias -> 'A::B'."""
    al = {}
    for m in re.finditer(r"\buse\s+([A-Za-z0-9_:]+?)(?:::\{([^}]*)\}|\s+as\s+([A-Za-z0-9_]+))?\s*;", code):
        path, group, alias = m.group(1), m.group(2), m.group(3)
        if group is not None:
            for item in _split_top(group):
                im = re.match(r"([A-Za-z0-9_]+)(?:\s+as\s+([A-Za-z0-9_]+))?$", item.strip())
                if im:
                    al[im.group(2) or im.group(1)] = f"{path}::{im.group(1)}"
        elif alias:
            al[alias] = path
        else:
            al[path.rsplit("::", 1)[-1]] = path
    return al


_POOLED = re.compile(r"^if uses_pooled_root\(statistic\)\s*\{\s*Some\((\w+)\)\s*\}\s*else\s*\{\s*None\s*\}$", re.S)
_V2DEF = re.compile(r"^match defect\s*\{\s*Some\(d\)\s*=>\s*Some\(d\),\s*None\s*=>\s*\{\s*(if .*?)\s*\}\s*,?\s*\}$", re.S)


class _Eval:
    def __init__(self, top_fns: dict):
        self.top = top_fns

    def call(self, expr: str, local_fns: dict, aliases: dict) -> dict:
        m = re.match(r"^([a-z0-9_]+)\s*\((.*)\)$", expr.strip(), re.S)
        if not m:
            raise SourceShapeError(f"{FEATURE_DEFS}: not a constructor call: {expr[:80]!r}")
        name = m.group(1)
        fn = local_fns.get(name) or self.top.get(name)
        if fn is None:
            raise SourceShapeError(f"{FEATURE_DEFS}: unknown SignalDef constructor {name}()")
        args = _split_top(m.group(2))
        if len(args) != len(fn["params"]):
            raise SourceShapeError(f"{FEATURE_DEFS}: {name}() takes {len(fn['params'])} args, got {len(args)}")
        env = dict(zip(fn["params"], args))
        fields = self._body(fn["body"], env, local_fns, aliases)
        fields["_ctor"] = name
        return fields

    def _subst(self, expr: str, env: dict) -> str:
        return re.sub(r"\b([a-z_][a-z0-9_]*)\b", lambda mm: env.get(mm.group(1), mm.group(1)), expr)

    def _body(self, body: str, env: dict, local_fns: dict, aliases: dict) -> dict:
        base: dict = {}
        lm = re.match(r"^let base = (.*?\));\s*(.*)$", body, re.S)
        if lm:
            base = self.call(self._subst(lm.group(1), env), local_fns, aliases)
            body = lm.group(2).strip()
        if body.startswith("SignalDef"):
            o = body.index("{")
            inner = body[o + 1:_match_close(body, o)]
            fields = dict(base)
            for part in _split_top(inner):
                if part.startswith(".."):
                    continue
                if ":" in part and not part.startswith('"'):
                    k, v = part.split(":", 1)
                    k, v = k.strip(), v.strip()
                else:
                    k, v = part.strip(), part.strip()
                fields[k] = self._value(k, v, env, aliases)
            return fields
        if re.match(r"^[a-z0-9_]+\s*\(", body):
            return self.call(self._subst(body, env), local_fns, aliases)
        raise SourceShapeError(f"{FEATURE_DEFS}: constructor body not understood: {body[:80]!r}")

    def _value(self, key: str, v: str, env: dict, aliases: dict):
        if v in env:
            v = env[v]
        v = v.strip()
        if key in ("defect",):
            pm = _POOLED.match(v)
            if pm:
                return pm.group(1) if _stat_tok(env.get("statistic", ""), aliases) in ("L4", "L8") else None
            vm = _V2DEF.match(v)
            if vm:
                d = env.get("defect", "None").strip()
                if d.startswith("Some("):
                    return d[5:-1]
                return self._value("defect", vm.group(1), env, aliases)
            if v.startswith("Some("):
                return _resolve(self._subst(v[5:-1], env), aliases)
            if v == "None":
                return None
            return _resolve(v, aliases)
        if key == "revisions":
            return "derived" if v.startswith("if ") else v
        if key in ("name",):
            if not (v.startswith('"') and v.endswith('"')):
                raise SourceShapeError(f"{FEATURE_DEFS}: signal name is not a string literal: {v!r}")
            return v[1:-1]
        if key == "block_local":
            if not v.isdigit():
                raise SourceShapeError(f"{FEATURE_DEFS}: block_local is not an integer literal: {v!r}")
            return int(v)
        if key == "deprecated":
            return v == "true"
        return _resolve(v, aliases)


def _resolve(v: str, aliases: dict) -> str:
    v = v.strip()
    m = re.match(r"^Family::Supported\((.*)\)$", v)
    if m:
        return _resolve(m.group(1), aliases)
    if v in aliases:
        v = aliases[v]
    head = v.split("::")[0]
    if head in aliases and "::" in v:
        v = aliases[head] + "::" + v.split("::", 1)[1]
    return v.rsplit("::", 1)[-1]


def _stat_tok(v: str, aliases: dict) -> str:
    return _resolve(v, aliases) if v else ""


_TOKEN_STR = {  # enum variant -> as_str token, checked against the as_str tables below
    "Statistic": None, "Form": None, "Direction": None, "CostClass": None, "KernelId": None,
}


def _as_str_tables(code: str) -> dict:
    """enum name -> {Variant: token} from `impl X { fn as_str(..) { match self { Self::V => "tok", .. } } }`."""
    out = {}
    for m in re.finditer(r"impl ([A-Za-z]+) \{", code):
        o = code.index("{", m.end() - 1)
        c = _match_close(code, o)
        body = code[o:c]
        am = re.search(r"fn as_str\(self\) -> &'static str \{(.*)", body, re.S)
        if not am:
            continue
        pairs = dict(re.findall(r"(?:Self|[A-Za-z]+)::([A-Za-z0-9]+)(?:\([^)]*\))?\s*=>\s*\"([a-z0-9_]+)\"", am.group(1)))
        if pairs:
            out.setdefault(m.group(1), {}).update(pairs)
    return out


def read(ctx) -> dict:
    raw = ctx.text(FEATURE_DEFS, "feature_defs")
    code = strip_comments(raw)
    top_fns = _parse_fns(code)
    statics = {}
    ev = _Eval({k: v for k, v in top_fns.items()})
    for m in re.finditer(r"pub\(crate\) static ([A-Z0-9_]+): \[SignalDef; (\d+)\] = ", code):
        name, count = m.group(1), int(m.group(2))
        o = m.end()
        if code[o] == "{":
            c = _match_close(code, o)
            block = code[o + 1:c]
            local_fns = _parse_fns(block)
            aliases = _uses(block)
            # the array literal is the last top-level `[`..`]` in the block
            depth, arr_open = 0, None
            j = 0
            while j < len(block):
                ch = block[j]
                if ch == '"':
                    j += 1
                    while block[j] != '"':
                        j += 2 if block[j] == "\\" else 1
                elif ch == "{" or ch == "(":
                    depth += 1
                elif ch == "}" or ch == ")":
                    depth -= 1
                elif ch == "[" and depth == 0:
                    arr_open = j
                    j = _match_close(block, j)
                j += 1
            if arr_open is None:
                raise SourceShapeError(f"{FEATURE_DEFS}: static {name} has no array literal")
            arr = block[arr_open + 1:_match_close(block, arr_open)]
        elif code[o] == "[":
            local_fns, aliases = {}, {}
            arr = code[o + 1:_match_close(code, o)]
        else:
            continue
        sigs = [ev.call(e, local_fns, aliases) for e in _split_top(arr)]
        if len(sigs) != count:
            raise SourceShapeError(f"{FEATURE_DEFS}: static {name} declares {count} signals, parsed {len(sigs)}")
        for k, s in enumerate(sigs):
            if s.get("block_local") != k:
                raise SourceShapeError(f"{FEATURE_DEFS}: {name}[{k}] has block_local {s.get('block_local')}")
        line = raw[:raw.find(f"static {name}:")].count("\n") + 1
        statics[name] = {"signals": sigs, "line": line}
    # BLOCKS
    bm = re.search(r"pub\(crate\) static BLOCKS: &\[BlockDef\] = &\[", code)
    if not bm:
        raise SourceShapeError(f"{FEATURE_DEFS}: BLOCKS not found")
    bo = code.index("[", bm.end() - 1)
    blocks = []
    for part in _split_top(code[bo + 1:_match_close(code, bo)]):
        pm = re.match(r"BlockDef\s*\{\s*family:\s*(.*?),\s*signals:\s*&([A-Z0-9_]+),\s*replication:\s*Replication::(\w+),?\s*\}$",
                      part, re.S)
        if not pm:
            raise SourceShapeError(f"{FEATURE_DEFS}: BlockDef entry not understood: {part[:80]!r}")
        fam = _resolve(pm.group(1), {}).lower()
        if pm.group(2) not in statics:
            raise SourceShapeError(f"{FEATURE_DEFS}: BLOCKS references unknown static {pm.group(2)}")
        blocks.append({"family": fam, "static": pm.group(2), "replication": pm.group(3)})
    tables = _as_str_tables(code)
    for enum in ("Statistic", "Form", "Direction", "CostClass", "KernelId"):
        if enum not in tables:
            raise SourceShapeError(f"{FEATURE_DEFS}: as_str table for {enum} not found")
    widths = re.search(r"REGISTERED_LAYOUT_WIDTHS: &\[usize\] = &\[([\d,\s]+)\]", code)
    if not widths:
        raise SourceShapeError(f"{FEATURE_DEFS}: REGISTERED_LAYOUT_WIDTHS not found")
    registered_widths = [int(x) for x in re.findall(r"\d+", widths.group(1))]

    def tok(enum, v):
        if v is None:
            return None
        t = tables[enum].get(v)
        if t is None:
            raise SourceShapeError(f"{FEATURE_DEFS}: {enum}::{v} has no as_str token")
        return t

    slots, families = [], []
    base = 0
    for b in blocks:
        sigs = statics[b["static"]]["signals"]
        per = len(sigs)
        rep = b["replication"]
        cells: list[tuple[int, str, int]] = []  # (scale, channel, local) in slot order
        if rep == "PerChannel":
            cells = [(s, CHANNELS[c], l) for s in range(N_SCALES) for c in range(3) for l in range(per)]
        elif rep == "PerScale":
            cells = [(s, "s", l) for s in range(N_SCALES) for l in range(per)]
        elif rep == "Flat":
            cells = [(0, "s", l) for l in range(per)]
        elif rep == "NativeXb":
            cells = [(0, ch, l) for ch in ("x", "b") for l in range(per)]
        elif rep == "GmsbankChroma":
            if per != 25:
                raise SourceShapeError(f"{FEATURE_DEFS}: GmsbankChroma expects 25 signals, found {per}")
            cells = [(0, "y", l) for l in range(15)]
            for s in range(1, N_SCALES):
                cells += [(s, ch, l) for ch in CHANNELS for l in range(15)]
                cells += [(s, "s", 15 + l) for l in range(10)]
        else:
            raise SourceShapeError(f"{FEATURE_DEFS}: unknown replication {rep}")
        fam_slots = []
        for k, (scale, ch, local) in enumerate(cells):
            s = sigs[local]
            sid = base + k
            fam_slots.append(sid)
            slots.append({
                "id": sid, "family": b["family"], "signal": s["name"], "local": local, "scale": scale, "channel": ch,
                "name": f'{b["family"]}_{s["name"]}_s{scale}_{ch}',
                "statistic": tok("Statistic", s.get("statistic")), "form": tok("Form", s.get("form")),
                "direction": tok("Direction", s.get("direction")), "cost": tok("CostClass", s.get("cost")),
                "kernel": tok("KernelId", s.get("kernel")), "deprecated": s.get("deprecated", False),
                "defect": s.get("defect"),
            })
        families.append({"family": b["family"], "static": b["static"], "replication": rep,
                         "line": statics[b["static"]]["line"], "lo": base, "hi": base + len(cells) - 1,
                         "width": len(cells), "signals": [
                             {"local": s["block_local"], "name": s["name"], "statistic": tok("Statistic", s.get("statistic")),
                              "form": tok("Form", s.get("form")), "direction": tok("Direction", s.get("direction")),
                              "cost": tok("CostClass", s.get("cost")), "kernel": tok("KernelId", s.get("kernel")),
                              "defect": s.get("defect"), "deprecated": s.get("deprecated", False),
                              "ctor": s["_ctor"]} for s in sigs]})
        base += len(cells)
    full_width = base
    if full_width not in registered_widths:
        raise SourceShapeError(f"{FEATURE_DEFS}: computed full width {full_width} is not a registered layout width")
    # FormulaRevision variants with doc comments and era tokens
    revs = []
    fm = re.search(r"pub enum FormulaRevision \{(.*?)\n\}", raw, re.S)
    if not fm:
        raise SourceShapeError(f"{FEATURE_DEFS}: enum FormulaRevision not found")
    doc: list[str] = []
    for line in fm.group(1).splitlines():
        s = line.strip()
        if s.startswith("///"):
            doc.append(s[3:].strip())
        elif re.match(r"^(Rev\d+),?$", s):
            revs.append({"name": s.rstrip(","), "doc": " ".join(doc)})
            doc = []
        elif s.startswith("#[") or not s:
            continue
    em = re.search(r"fn era_tokens\(self\) -> &'static \[&'static str\] \{(.*?)\n    \}", code, re.S)
    if em:
        for rv in revs:
            mm = re.search(rf"Self::{rv['name']}\s*=>\s*&\[([^\]]*)\]", em.group(1))
            rv["eras_added"] = re.findall(r'"([a-z0-9_]+)"', mm.group(1)) if mm else None
            if mm is None:
                mm2 = re.search(rf"Self::{rv['name']}\s*=>\s*([A-Z_0-9]+)", em.group(1))
                rv["eras_const"] = mm2.group(1) if mm2 else None
    defects = {}
    for m in re.finditer(r"const (DEFECT_[A-Z0-9_]+): Defect = Defect \{\s*id:\s*\"([^\"]+)\",\s*note:\s*(.*?),\s*\};", code, re.S):
        note = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(3)))
        defects[m.group(1)] = {"id": m.group(2), "note": re.sub(r"\\\n\s*", "", note)}
    ctx.count(FEATURE_DEFS, len(slots))
    return {"path": FEATURE_DEFS, "blocks": families, "slots": slots, "full_width": full_width,
            "registered_widths": registered_widths, "revisions": revs, "defects": defects,
            "n_scales": N_SCALES}
