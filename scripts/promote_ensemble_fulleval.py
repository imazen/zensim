#!/usr/bin/env python3
"""promote_ensemble_fulleval.py — publish an ENSEMBLE `bake_verdict --full-json` verdict
onto the summer-gauntlet board WITHOUT recomputing a single statistic.

An equal-weight ensemble (`bake_verdict --ensemble a.bin,b.bin,...`) is scored by exactly
the same program/corpora/grids as a single bake, so its verdict JSON is already the
fulleval schema — except for two things this script fixes:

  1. **M3 / M3a are NOT COMPUTABLE for an ensemble.** `run_full_eval.sh` normally injects
     them by running `diffmap_block_coherence --bake <one ZNPR>`; an ensemble has no single
     ZNPR, so the instrument cannot be pointed at it. Both keys are emitted as explicit
     JSON **null** — never 0.0, never a placeholder (the dashboard renders null as an
     em-dash; a 0.0 would read as "measured, and terrible").
  2. **The `model` block describes member 0 only** (`bake_verdict` introspects
     `Ensemble::primary` — see its own comment). Left unmarked, the board's Model-details
     card shows one member's architecture/seed/repro as if it were the shipped artifact.
     This script stamps `model.kind="ensemble"` + `model.members=k` + the member list, and
     `gauntlet.py` renders an `ens×k` marker wherever the bake is named.

Everything that is a NUMBER is carried through byte-identically: `rank`, `dial`,
`corruption`, `corruption_head`, `gates`, `composite`, `per_pair`, `n_inputs`, `regime`
are re-serialized from the parsed source and asserted equal to it before writing. The
source verdict's path + sha256 land in `source_verdict` so any board number chains back
to the committed verdict file (docs/REPRODUCIBILITY.md).

Usage:
    promote_ensemble_fulleval.py --verdict <stem.full.json> --name <board-name> \
        (--members A,B,C | --members-file <tsv>) [--out-dir DIR] [--dry-run]

`--members-file` reads the FIRST tab/whitespace-separated column of each non-`#` line
(the frozen-registration format of benchmarks/wave5_e3_members.txt).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

DEFAULT_OUT = Path("/mnt/v/output/zensim/reports/fulleval")

# Blocks that must survive promotion untouched — every statistic on the board.
CARRIED = ("rank", "dial", "corruption", "corruption_head", "gates", "composite",
           "per_pair", "n_inputs", "regime", "bake", "bake_sha256", "repro")


def read_members_file(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line.split()[0])
    return out


def promote(verdict: Path, name: str, members: list[str], out_dir: Path,
            dry_run: bool = False) -> Path:
    src_bytes = verdict.read_bytes()
    src = json.loads(src_bytes)
    if not members:
        raise SystemExit("promote: --members / --members-file produced an empty list")

    doc = copy.deepcopy(src)
    doc["name"] = name

    # (1) M3 / M3a: explicit null. An ensemble has no single ZNPR for
    # diffmap_block_coherence to load, so the coherence instruments are
    # NOT-MEASURED — which is a different statement from "measured low".
    # If a future ensemble-aware instrument fills them in the source, carry it.
    doc["m3_coherence"] = src.get("m3_coherence")
    doc["m3a_coherence"] = src.get("m3a_coherence")
    if doc["m3_coherence"] is None:
        doc.pop("m3_n", None)
        doc.pop("m3_dropped_mass_pct", None)
    if doc["m3a_coherence"] is None:
        doc.pop("m3a_n", None)

    # (2) mark the model block as an ensemble; its architecture/repro fields
    # describe the ANCHOR member (bake_verdict introspects Ensemble::primary).
    model = doc.get("model")
    if not isinstance(model, dict):
        raise SystemExit(f"promote: {verdict} has no `model` block — not a full-json verdict")
    model["kind"] = "ensemble"
    model["members"] = len(members)
    model["member_names"] = list(members)
    model["anchor"] = Path(str(src.get("bake", ""))).name or None

    doc["source_verdict"] = {
        "path": str(verdict),
        "sha256": hashlib.sha256(src_bytes).hexdigest(),
        "name": src.get("name"),
    }

    # Byte-identity gate on every statistic: promotion relabels, it never rescores.
    for k in CARRIED:
        a = json.dumps(src.get(k), sort_keys=True, separators=(",", ":"))
        b = json.dumps(doc.get(k), sort_keys=True, separators=(",", ":"))
        if a != b:
            raise SystemExit(f"promote: block `{k}` changed during promotion — refusing to write")

    out = out_dir / f"{name}.fulleval.json"
    if dry_run:
        print(f"[dry-run] would write {out}  (k={len(members)}, "
              f"cid22={src.get('rank', {}).get('cid22', {}).get('srocc')})")
        return out
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc))
    print(f"wrote {out}  k={len(members)}  cid22={src.get('rank', {}).get('cid22', {}).get('srocc')}  "
          f"m3={doc['m3_coherence']} m3a={doc['m3a_coherence']}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdict", required=True, type=Path,
                    help="the ensemble's bake_verdict --full-json output")
    ap.add_argument("--name", required=True, help="board name (fulleval JSON `name`)")
    ap.add_argument("--members", default=None,
                    help="comma-separated member stems (the FROZEN registration list)")
    ap.add_argument("--members-file", default=None, type=Path,
                    help="file whose first column per non-# line is a member stem")
    ap.add_argument("--out-dir", default=DEFAULT_OUT, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    if bool(a.members) == bool(a.members_file):
        ap.error("give exactly one of --members / --members-file")
    members = ([m for m in a.members.split(",") if m] if a.members
               else read_members_file(a.members_file))
    if not a.verdict.exists():
        raise SystemExit(f"promote: verdict not found: {a.verdict}")
    promote(a.verdict, a.name, members, a.out_dir, a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
