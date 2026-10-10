"""Source readers: each parses one committed source of truth and checks its shape.

Readers return plain dicts/lists. They never open label tables or protected
payloads; they read repository markdown, JSON summaries and Rust source.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .mdparse import (SourceShapeError, first_line_matching, headings, section, split_row, strip_md,
                      table_with_headers, tables)

REPO = Path(__file__).resolve().parents[2]


class Ctx:
    """Tracks every file read (path, sha256, git blob id, reader, entity count) for the Sources page.

    zensim paths are read from the checkout being built. `zenanalyze:` paths are
    read from a git commit of the sibling repository (`zenanalyze_rev`, default
    `origin/main`), never from whatever its working copy holds; pass
    `zenanalyze_rev="worktree"` to read the working copy instead.
    """

    def __init__(self, repo: Path = REPO, zenanalyze: Path | None = None, zenanalyze_rev: str = "origin/main"):
        self.repo = repo
        self.zenanalyze = zenanalyze or (repo.parent / "zenanalyze")
        self.zenanalyze_rev = zenanalyze_rev
        self._za_tree = None
        self.read_log: dict[str, dict] = {}

    @property
    def za_tree(self):
        """The zenanalyze commit being read (None when reading the working copy)."""
        if self.zenanalyze_rev == "worktree":
            return None
        if self._za_tree is None:
            from .gitref import GitTree, git_dir_of
            self._za_tree = GitTree(git_dir_of(self.zenanalyze), self.zenanalyze_rev)
        return self._za_tree

    def _abs(self, rel: str) -> Path:
        if rel.startswith("zenanalyze:"):
            return self.zenanalyze / rel.split(":", 1)[1]
        return self.repo / rel

    def _raw(self, rel: str) -> bytes:
        if rel.startswith("zenanalyze:") and self.za_tree is not None:
            return self.za_tree.read(rel.split(":", 1)[1])
        p = self._abs(rel)
        if not p.is_file():
            raise SourceShapeError(f"missing source file {rel}")
        return p.read_bytes()

    def bytes(self, rel: str, reader: str) -> bytes:
        from .gitref import blob_id
        data = self._raw(rel)
        ent = self.read_log.setdefault(rel, {"path": rel, "sha256": hashlib.sha256(data).hexdigest(),
                                             "blob": blob_id(data), "readers": [], "entities": 0})
        if reader not in ent["readers"]:
            ent["readers"].append(reader)
        return data

    def text(self, rel: str, reader: str) -> str:
        return self.bytes(rel, reader).decode("utf-8")

    def json(self, rel: str, reader: str):
        return json.loads(self.text(rel, reader))

    def count(self, rel: str, n: int):
        self.read_log[rel]["entities"] += n

    def exists(self, rel: str) -> bool:
        if rel.startswith("zenanalyze:") and self.za_tree is not None:
            return self.za_tree.exists(rel.split(":", 1)[1])
        return self._abs(rel).exists()

    def glob(self, pattern: str) -> list[str]:
        if pattern.startswith("zenanalyze:"):
            pat = pattern.split(":", 1)[1]
            if self.za_tree is not None:
                return ["zenanalyze:" + p for p in self.za_tree.glob(pat)]
            return sorted("zenanalyze:" + str(p.relative_to(self.zenanalyze)) for p in self.zenanalyze.glob(pat))
        return sorted(str(p.relative_to(self.repo)) for p in self.repo.glob(pattern))


def find_symbol(ctx: Ctx, rel: str, pattern: str) -> int:
    """1-based line of the first match of `pattern` in a code file (CodeRef resolution)."""
    n, _ = first_line_matching(ctx.text(rel, "code-ref"), pattern, rel)
    return n


# --------------------------------------------------------------------------- release gates

RELEASE_GATE_MAP = "benchmarks/release_gate_map_2026-10-07.md"
GATE_COLUMNS = ["Gate / owner command", "Required inputs and pass rule", "Status for production", "Protected reads"]


def _gate_id(name: str) -> str:
    s = strip_md(name).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:48].rstrip("-")


def classify_gate_status(status_md: str, rule_md: str) -> dict:
    """Fixed classification rules for the release gate map's status cell.

    - status: `blocked` if any bold clause contains "blocked"; else `ready` if a
      bold clause starts with "ready"; else `done` if one starts with "done".
    - components: the bold clauses split on ';' into done / ready / blocked parts.
    - rule_pending: an "Owner decision:" clause, or a rule cell stating that no
      numerical/acceptance rule is registered.
    """
    bold = re.findall(r"\*\*(.+?)\*\*", status_md)
    clauses = [c.strip() for b in bold for c in b.split(";") if c.strip()]
    comps = {"done": [], "ready": [], "blocked": []}
    for c in clauses:
        cl = c.lower()
        if cl.startswith("owner decision"):
            continue
        if "blocked" in cl:
            comps["blocked"].append(c)
        elif cl.startswith("ready"):
            comps["ready"].append(c)
        elif cl.startswith("done"):
            comps["done"].append(c)
    if comps["blocked"]:
        status = "blocked"
    elif comps["ready"]:
        status = "ready"
    elif comps["done"]:
        status = "done"
    else:
        raise SourceShapeError(f"release gate status has no classifiable bold clause: {status_md[:120]!r}")
    both = status_md + " " + rule_md
    rule_pending = bool(re.search(r"\*\*Owner decision:?\*\*", both)) or bool(
        re.search(r"no (numerical|additional [^.]*?) (pass/adoption rule|acceptance threshold)", both))
    owner_decisions = []
    for cell in (status_md, rule_md):
        owner_decisions += re.findall(r"\*\*Owner decision:?\*\*\s*([^*]+?)(?=\*\*|$)", cell)
    return {"status": status, "components": comps, "rule_pending": rule_pending,
            "owner_decisions": [o.strip() for o in owner_decisions]}


def release_gates(ctx: Ctx) -> dict:
    text = ctx.text(RELEASE_GATE_MAP, "release_gates")
    t = table_with_headers(text, RELEASE_GATE_MAP, GATE_COLUMNS)
    ci = {name: t.col(name) for name in GATE_COLUMNS}
    gates = []
    for row, line in zip(t.rows, t.row_lines):
        if len(row) != len(t.headers):
            raise SourceShapeError(f"{RELEASE_GATE_MAP}:{line}: expected {len(t.headers)} cells, got {len(row)}")
        first = row[ci["Gate / owner command"]]
        m = re.match(r"\*\*(.+?)\*\*:?\s*(.*)", first, re.S)
        if not m:
            raise SourceShapeError(f"{RELEASE_GATE_MAP}:{line}: gate cell lacks a bold name")
        name = m.group(1).rstrip(":").strip()
        cls = classify_gate_status(row[ci["Status for production"]], row[ci["Required inputs and pass rule"]])
        gates.append({
            "id": _gate_id(name), "name": name, "owner_command": m.group(2).strip(),
            "pass_rule": row[ci["Required inputs and pass rule"]],
            "status_text": row[ci["Status for production"]],
            "protected_reads": row[ci["Protected reads"]],
            "line": line, **cls,
        })
    ctx.count(RELEASE_GATE_MAP, len(gates))
    blocked_line, _ = first_line_matching(text, r"^\*\*Blocked:\*\*", RELEASE_GATE_MAP)
    para = []
    for l in text.splitlines()[blocked_line - 1:]:
        if not l.strip():
            break
        para.append(l.strip())
    blocked = " ".join(para)
    _, qual_par = section(text, r"^Gate inventory$", RELEASE_GATE_MAP)
    qual = re.search(r"The current `freeze_check::qualification_report`.*?(?:\n\n|$)", qual_par, re.S)
    return {"path": RELEASE_GATE_MAP, "gates": gates, "headline": blocked, "headline_line": blocked_line,
            "qualification_note": qual.group(0).strip() if qual else None}


# --------------------------------------------------------------------------- scorecard

SCORECARD = "docs/MODEL_SELECTION_SCORECARD.md"


def scorecard(ctx: Ctx) -> dict:
    text = ctx.text(SCORECARD, "scorecard")
    contract = table_with_headers(text, SCORECARD, ["Requirement", "Release bar / measurement"])
    five = table_with_headers(text, SCORECARD, ["gate", "question", "instrument", "pass bar (SDR)"])
    reqs = [{"requirement": strip_md(r[0]), "bar": r[1], "line": ln} for r, ln in zip(contract.rows, contract.row_lines)]
    gates5 = [{"gate": strip_md(r[0]), "question": r[1], "instrument": r[2], "bar": r[3], "line": ln}
              for r, ln in zip(five.rows, five.row_lines)]
    disp_line, disp = first_line_matching(text, r"^Current disposition:", SCORECARD)
    # the disposition paragraph runs to the next blank line
    lines = text.splitlines()
    para = []
    for l in lines[disp_line - 1:]:
        if not l.strip():
            break
        para.append(l.strip())
    if len(reqs) < 8 or len(gates5) != 5:
        raise SourceShapeError(f"{SCORECARD}: expected >=8 contract rows and 5 exam gates, got {len(reqs)}/{len(gates5)}")
    ctx.count(SCORECARD, len(reqs) + len(gates5))
    return {"path": SCORECARD, "contract": reqs, "contract_line": contract.line, "exam": gates5,
            "exam_line": five.line, "disposition": " ".join(para), "disposition_line": disp_line}


# --------------------------------------------------------------------------- known bugs

def known_bugs(ctx: Ctx) -> list[dict]:
    text = ctx.text("CLAUDE.md", "known_bugs")
    start, body = section(text, r"^Known Bugs$", "CLAUDE.md")
    out = []
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        l = lines[i]
        if re.match(r"^[*-] ", l):
            entry = [l[2:].strip()]
            j = i + 1
            while j < len(lines) and not re.match(r"^[*-] ", lines[j]) and not lines[j].startswith("## "):
                entry.append(lines[j].strip())
                j += 1
            full = " ".join(x for x in entry if x)
            m = re.match(r"\*\*(\d{4}-\d{2}-\d{2})\s*[—-]\s*(.+?)\*\*(.*)", full)
            if m:
                date, head, rest = m.group(1), m.group(2), m.group(3)
            else:
                m2 = re.match(r"(\d{4}-\d{2}-\d{2})\s+(.*)", full)
                if not m2:
                    raise SourceShapeError(f"CLAUDE.md:{start + i + 1}: Known Bugs entry without a date")
                date, head, rest = m2.group(1), m2.group(2)[:160], m2.group(2)
            hu = (head + " " + rest[:200]).upper()
            if "NOT A BUG" in hu:
                st = "info"
            elif re.search(r"\bOPEN\b", head.upper()):
                st = "open"
            elif re.search(r"\b(FIXED|RESOLVED|AMENDED|SUPERSEDED)\b", hu):
                st = "fixed"
            elif re.search(r"\bOPEN\b", hu):
                st = "open"
            else:
                st = "info"
            out.append({"date": date, "title": head, "body": rest.strip(), "status": st, "line": start + i + 1})
            i = j
        else:
            i += 1
    if len(out) < 10:
        raise SourceShapeError("CLAUDE.md: Known Bugs section has fewer than 10 entries")
    ctx.count("CLAUDE.md", len(out))
    return out


# --------------------------------------------------------------------------- paper gates (peer G-ADDR)

PAPER_GATES = "benchmarks/paper_gates_2026-09-23.json"
GADDR_ROWS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7r", "A8r", "C1", "C2", "C3", "C4", "C5", "C6"]


def paper_gates(ctx: Ctx) -> dict:
    d = ctx.json(PAPER_GATES, "paper_gates")
    if d.get("schema") != "paper-gates-2026-09-23-v1":
        raise SourceShapeError(f"{PAPER_GATES}: schema {d.get('schema')!r}")
    scorers = []
    for name, s in d["scorers"].items():
        g = s.get("gaddr") or {}
        reading = "native" if "native" in g else None
        if reading is None:
            raise SourceShapeError(f"{PAPER_GATES}: scorer {name} lacks gaddr.native")
        states = g["native"]["states"]
        missing = [k for k in GADDR_ROWS if k not in states]
        if missing:
            raise SourceShapeError(f"{PAPER_GATES}: scorer {name} lacks states {missing}")
        scorers.append({"name": name, "orientation": s["orientation"], "identity_exact": s.get("identity_exact"),
                        "lsb_center": s.get("lsb-center-g_min_med_max"), "lsb_all": s.get("lsb-all_min_med_max"),
                        "native": g["native"], "s100": g.get("s100")})
    ctx.count(PAPER_GATES, len(scorers))
    return {"path": PAPER_GATES, "instrument": d["instrument"], "scorers": scorers,
            "aic2026": d.get("aic2026"), "review": d.get("review")}


# --------------------------------------------------------------------------- data splits

DATA_SPLITS = "docs/DATA_SPLITS.md"


def data_splits(ctx: Ctx) -> dict:
    text = ctx.text(DATA_SPLITS, "data_splits")
    reg = table_with_headers(text, DATA_SPLITS, ["Dataset", "Tier", "Our split"])
    ci = {k: reg.col(k) for k in ("Dataset", "Tier", "Our split")}
    leak = None
    for h in reg.headers:
        if h.strip().lower().startswith("leakage"):
            leak = reg.headers.index(h)
    datasets = []
    for r, ln in zip(reg.rows, reg.row_lines):
        if len(r) < 3:
            continue
        name = strip_md(r[ci["Dataset"]])
        short = re.split(r"\s*\(", name, maxsplit=1)[0].strip()
        datasets.append({"name": short, "full": r[ci["Dataset"]], "tier": r[ci["Tier"]], "split": r[ci["Our split"]],
                         "leakage": r[leak] if leak is not None and leak < len(r) else "", "line": ln})
    ledger = []
    kinds = r"(Exposure ledger|Exposure receipt|Ruling|Finding|Addendum|Admission update|Label-source correction|LODO rotation|Post-read update|registered|report|complete|preparation|E\d+)"
    for line, lvl, title in headings(text):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", title)
        if not m or lvl < 2:
            continue
        if line < 600:  # policy sections carry dates in their titles too; the ledger starts after §8
            continue
        k = re.search(kinds, title)
        kind = k.group(1) if k else "entry"
        if kind.startswith("E") and kind[1:].isdigit():
            kind = "experiment"
        ledger.append({"date": m.group(1), "title": title, "line": line, "level": lvl, "kind": kind})
    roles_line, _ = section(text, r"^September 14 clarification", DATA_SPLITS)
    policy = [{"line": n, "level": l, "title": t} for n, l, t in headings(text) if n < 640 and l <= 3]
    if len(datasets) < 15 or len(ledger) < 30:
        raise SourceShapeError(f"{DATA_SPLITS}: registry/ledger too small ({len(datasets)}/{len(ledger)})")
    ctx.count(DATA_SPLITS, len(datasets) + len(ledger))
    return {"path": DATA_SPLITS, "datasets": datasets, "registry_line": reg.line, "ledger": ledger,
            "policy": policy, "clarification_line": roles_line}


# --------------------------------------------------------------------------- experiment decisions

def decision_json(ctx: Ctx, rel: str) -> dict:
    d = ctx.json(rel, "experiment_decision")
    arms = d.get("arms")
    if not isinstance(arms, dict) or not arms:
        raise SourceShapeError(f"{rel}: no arms object")
    out_arms = []
    for name, a in arms.items():
        ps = a.get("per_source")
        if ps is not None and not isinstance(ps, dict):
            raise SourceShapeError(f"{rel}: arm {name} per_source is not an object")
        out_arms.append({"name": name, "spec": a.get("spec"), "signed_mean": a.get("signed_mean"),
                         "signed_se": a.get("signed_se"), "signed_worst": a.get("signed_worst"),
                         "per_source": ps or {}, "w2": a.get("w2"), "w2_se": a.get("w2_se"),
                         "as_good": a.get("as_good"), "passes": a.get("passes"), "raw": a})
    ctx.count(rel, len(out_arms))
    return {"path": rel, "schema": d.get("schema"), "arms": out_arms, "adopted": d.get("adopted", d.get("adopt")),
            "control": d.get("control"), "status": d.get("status"), "raw": d}
