#!/usr/bin/env python3
"""promote_fulleval.py — publish a `bake_verdict --full-json` verdict (single bake OR
ensemble) onto the summer-gauntlet board WITHOUT recomputing a single statistic.

(Generalized 2026-08-04 from promote_ensemble_fulleval.py: the coverage audit found the
sota944 campaign produced ~149 verdict cells while the board showed ~45 — every non-ensemble
grid cell had no promotion path. Single-bake promotion is now the default; `--members`
switches on the ensemble stamping.)

A verdict is scored by exactly the same program/corpora/grids as every promoted cell, so
its JSON is already the fulleval schema — promotion RELABELS and annotates, it never
rescores. What it adds per mode:

  ALL bakes:
  1. `name` = the board name; `source_verdict` = {path, sha256, name} so any board number
     chains back to the committed verdict file (docs/REPRODUCIBILITY.md).
  2. M3 / M3a carried from the verdict when present, else explicit JSON **null** — never
     0.0, never a placeholder (the dashboard renders null as an em-dash; a 0.0 would read
     as "measured, and terrible"). `--carry-coherence-from <existing.fulleval.json>` fills
     nulls from an already-measured board file for the SAME bake (sha-gated) — the
     "carry m3a where measured" rule for re-promotions.
  3. `--strip-per-pair` (registered board-size rule 2026-08-04): grid-interior cells drop
     the per-pair scatter arrays AFTER the integrity gate passes — all scalar stats
     (rank/bands/dial/gates/corruption) stay; `per_pair_stripped: true` records it and
     the full data remains in `source_verdict.path`. The curated headline set keeps
     per_pair (list: `scripts/v_next/gauntlet.py` CURATED_BOARD — the one owner).

  ENSEMBLES (`--members` / `--members-file`):
  4. **M3 / M3a are NOT COMPUTABLE for an ensemble** (`diffmap_block_coherence --bake`
     loads one ZNPR; an ensemble has no single ZNPR) — nulls per (2).
  5. **The `model` block describes member 0 only** (`bake_verdict` introspects
     `Ensemble::primary`). This stamps `model.kind="ensemble"` + `model.members=k` + the
     member list, and `gauntlet.py` renders an `ens×k` marker wherever the bake is named.

  GRAFT mode (`--graft-into <board.fulleval.json>`): copy the `corruption_head` block from
  a `*_corrjoint.full.json` verdict into an already-promoted board file whose
  corruption_head is null — sha-gated (same bake), every other key byte-identical, source
  recorded in `corruption_head_source`. This is how corrjoint re-verdicts fold under the
  plain board name without losing the richer full-eval content (measured M3a, kadis
  per-pair) the plain file already carries.

Everything that is a NUMBER is carried through byte-identically: `rank`, `dial`,
`corruption`, `corruption_head`, `gates`, `composite`, `per_pair`, `n_inputs`, `regime`
are re-serialized from the parsed source and asserted equal to it before writing (the
per-pair strip happens after that gate, and only replaces the block with an explicit
stripped marker).

Usage:
    promote_fulleval.py --verdict <stem.full.json> --name <board-name> \
        [--members A,B,C | --members-file <tsv>] [--strip-per-pair] \
        [--carry-coherence-from <existing.fulleval.json>] [--out-dir DIR] [--dry-run]
    promote_fulleval.py --verdict <stem_corrjoint.full.json> --graft-into <board.fulleval.json> \
        [--dry-run]

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

# Coherence-instrument fields that may be carried from an existing measured fulleval
# (the m3a-carry rule); each is only filled when the verdict's value is null/absent.
COHERENCE_FIELDS = ("m3_coherence", "m3_n", "m3_dropped_mass_pct", "m3a_coherence", "m3a_n")


def _jc(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def read_members_file(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line.split()[0])
    return out


def promote(verdict: Path, name: str, members: list[str] | None, out_dir: Path,
            dry_run: bool = False, strip_per_pair: bool = False,
            carry_coherence_from: Path | None = None) -> Path:
    src_bytes = verdict.read_bytes()
    src = json.loads(src_bytes)

    doc = copy.deepcopy(src)
    doc["name"] = name

    # M3 / M3a: verdict value when measured, else explicit null (see docstring 2/4).
    doc["m3_coherence"] = src.get("m3_coherence")
    doc["m3a_coherence"] = src.get("m3a_coherence")

    # Optional carry from an already-measured board file for the SAME bake.
    if carry_coherence_from is not None:
        prev_bytes = carry_coherence_from.read_bytes()
        prev = json.loads(prev_bytes)
        if prev.get("bake_sha256") != src.get("bake_sha256"):
            raise SystemExit(
                f"promote: --carry-coherence-from bake_sha256 mismatch "
                f"({carry_coherence_from} is a different bake) — refusing")
        carried = []
        for k in COHERENCE_FIELDS:
            if doc.get(k) is None and prev.get(k) is not None:
                doc[k] = prev[k]
                carried.append(k)
        if carried:
            doc["coherence_source"] = {
                "path": str(carry_coherence_from),
                "sha256": hashlib.sha256(prev_bytes).hexdigest(),
                "fields": carried,
            }

    if doc["m3_coherence"] is None:
        doc.pop("m3_n", None)
        doc.pop("m3_dropped_mass_pct", None)
    if doc["m3a_coherence"] is None:
        doc.pop("m3a_n", None)

    model = doc.get("model")
    if not isinstance(model, dict):
        raise SystemExit(f"promote: {verdict} has no `model` block — not a full-json verdict")

    if members is not None:
        # ENSEMBLE: mark the model block; its architecture/repro fields describe the
        # ANCHOR member (bake_verdict introspects Ensemble::primary).
        if not members:
            raise SystemExit("promote: --members / --members-file produced an empty list")
        model["kind"] = "ensemble"
        model["members"] = len(members)
        model["member_names"] = list(members)
        model["anchor"] = Path(str(src.get("bake", ""))).name or None
        # An ensemble has no single ZNPR: the coherence instruments are NOT-MEASURED —
        # a different statement from "measured low". (A carry still applies if a future
        # ensemble-aware instrument fills them in the source.)

    doc["source_verdict"] = {
        "path": str(verdict),
        "sha256": hashlib.sha256(src_bytes).hexdigest(),
        "name": src.get("name"),
    }

    # Byte-identity gate on every statistic: promotion relabels, it never rescores.
    for k in CARRIED:
        if _jc(src.get(k)) != _jc(doc.get(k)):
            raise SystemExit(f"promote: block `{k}` changed during promotion — refusing to write")

    if strip_per_pair:
        # AFTER the integrity gate: registered board-size rule — grid-interior cells
        # carry every scalar stat but no embedded scatter arrays. The full per-pair
        # data stays in source_verdict.path (never deleted).
        doc["per_pair"] = {}
        doc["per_pair_stripped"] = True

    out = out_dir / f"{name}.fulleval.json"
    k_note = f"k={len(members)}  " if members else ""
    cid = src.get("rank", {}).get("cid22", {}).get("srocc")
    if dry_run:
        print(f"[dry-run] would write {out}  ({k_note}cid22={cid}"
              f"{'  per_pair STRIPPED' if strip_per_pair else ''})")
        return out
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc))
    print(f"wrote {out}  {k_note}cid22={cid}  m3={doc['m3_coherence']} m3a={doc['m3a_coherence']}"
          f"{'  per_pair STRIPPED' if strip_per_pair else ''}")
    return out


def graft_corruption_head(board: Path, verdict: Path, dry_run: bool = False) -> bool:
    """Copy `corruption_head` from a corrjoint verdict into an existing board file
    (same bake only). Returns True when the file was (or would be) updated."""
    board_bytes = board.read_bytes()
    bdoc = json.loads(board_bytes)
    v_bytes = verdict.read_bytes()
    v = json.loads(v_bytes)
    ch = v.get("corruption_head")
    if not isinstance(ch, dict):
        raise SystemExit(f"graft: {verdict} carries no corruption_head block")
    if bdoc.get("bake_sha256") != v.get("bake_sha256"):
        raise SystemExit(f"graft: bake_sha256 mismatch — {board.name} is not the same bake "
                         f"as {verdict.name}; refusing")
    if isinstance(bdoc.get("corruption_head"), dict):
        print(f"graft: {board.name} already has corruption_head — unchanged")
        return False
    doc = dict(bdoc)
    doc["corruption_head"] = ch
    doc["corruption_head_source"] = {
        "path": str(verdict),
        "sha256": hashlib.sha256(v_bytes).hexdigest(),
        "name": v.get("name"),
    }
    # Integrity: every pre-existing key other than the grafted block is byte-identical.
    for k in bdoc:
        if k == "corruption_head":
            continue
        if _jc(bdoc[k]) != _jc(doc[k]):
            raise SystemExit(f"graft: block `{k}` changed — refusing to write")
    if dry_run:
        print(f"[dry-run] would graft corruption_head ({v.get('name')}) into {board}")
        return True
    board.write_text(json.dumps(doc))
    print(f"grafted corruption_head into {board}  (pass_q20={ch.get('pass_q20')})")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdict", required=True, type=Path,
                    help="the bake_verdict --full-json output to promote (or graft from)")
    ap.add_argument("--name", default=None, help="board name (fulleval JSON `name`)")
    ap.add_argument("--members", default=None,
                    help="ENSEMBLE ONLY: comma-separated member stems (the FROZEN registration list)")
    ap.add_argument("--members-file", default=None, type=Path,
                    help="ENSEMBLE ONLY: file whose first column per non-# line is a member stem")
    ap.add_argument("--strip-per-pair", action="store_true",
                    help="drop per-pair scatter arrays (grid-interior cells; registered size rule)")
    ap.add_argument("--carry-coherence-from", default=None, type=Path,
                    help="existing fulleval (same bake) whose measured M3/M3a fill this verdict's nulls")
    ap.add_argument("--graft-into", default=None, type=Path,
                    help="GRAFT mode: existing board fulleval to receive --verdict's corruption_head")
    ap.add_argument("--out-dir", default=DEFAULT_OUT, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    if not a.verdict.exists():
        raise SystemExit(f"promote: verdict not found: {a.verdict}")

    if a.graft_into is not None:
        if a.name or a.members or a.members_file or a.strip_per_pair or a.carry_coherence_from:
            ap.error("--graft-into takes only --verdict (and --dry-run)")
        if not a.graft_into.exists():
            raise SystemExit(f"graft: board file not found: {a.graft_into}")
        graft_corruption_head(a.graft_into, a.verdict, a.dry_run)
        return 0

    if not a.name:
        ap.error("--name is required (except in --graft-into mode)")
    if a.members and a.members_file:
        ap.error("give at most one of --members / --members-file")
    members = None
    if a.members:
        members = [m for m in a.members.split(",") if m]
    elif a.members_file:
        members = read_members_file(a.members_file)
    promote(a.verdict, a.name, members, a.out_dir, a.dry_run,
            strip_per_pair=a.strip_per_pair, carry_coherence_from=a.carry_coherence_from)
    return 0


if __name__ == "__main__":
    sys.exit(main())
