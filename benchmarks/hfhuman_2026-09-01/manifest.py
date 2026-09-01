#!/usr/bin/env python3
"""`_MANIFEST.json` for an hfhuman artifact dir. Provenance only -- computes nothing.

Same schema as the lane's parent manifest (name/date/repo/build_commit/doc/what/
split_rule/regimes/regime_purity/gates/inputs/arms/files), so both directories
read identically. `--extra key=value` adds flat annotations.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--what", required=True)
    ap.add_argument("--doc", default="benchmarks/hfhuman_2026-09-01.md")
    ap.add_argument("--date", default="2026-09-01")
    ap.add_argument("--commit", default=None, help="build commit; default = git HEAD of --repo")
    ap.add_argument("--extra", action="append", default=[])
    a = ap.parse_args()
    d, repo = Path(a.dir), Path(a.repo)
    commit = a.commit or subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    man = {
        "name": a.name, "date": a.date, "repo": "github.com/imazen/zensim",
        "build_commit": commit, "doc": a.doc, "what": a.what,
        "split_rule": ("jpeg-aic-family-holdout-2026-09-01 (benchmarks/eval_annotations.json) "
                       "— HOLDOUT-ONLY, family-wide, membership by content. NO training leg "
                       "was produced from this family, by design."),
        "regimes": {"v1": "true 372 (pools live)", "foldapp2": "944",
                    "foldapp2pools": "944-pools (Q7b's root)"},
        "regime_purity": "each bake is scored ONLY on its own regime; never column-mix",
        "gates": "see stimuli_manifest.json['gates'] (G5..G8) + the doc's appendix A",
    }
    for kv in a.extra:
        k, v = kv.split("=", 1)
        man[k] = v
    sm = d / "stimuli_manifest.json"
    if sm.exists():
        j = json.loads(sm.read_text())
        man["arms"] = j.get("arms", {})
        man["inputs"] = j.get("inputs", {})
        man["iptc_identification"] = j.get("iptc_identification", {})
    man["files"] = {}
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.name != "_MANIFEST.json":
            man["files"][str(p.relative_to(d))] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    (d / "_MANIFEST.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    print(f"-> {d/'_MANIFEST.json'}  ({len(man['files'])} files, build_commit {commit[:8]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
