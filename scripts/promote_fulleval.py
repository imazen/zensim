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

  Surgical modes added by the board-integrity pass (2026-08-04) — all share the same
  everything-else-byte-identical gate:
  * `--graft-into <board> --graft-rank <corpus>`: graft `rank.<corpus>` from a same-bake
    verdict into a board file that lacks it (the era-bridge hfnlproxy fill); provenance in
    `rank_graft_sources.<corpus>`.
  * `--mark-dominated <board> --dominated-by A,B`: write `dominated_by` + `dominance`
    provenance (strict same-class Pareto trim; empty list clears). The board renders
    dominated cells dimmed + default-off; files are NEVER deleted.
  * `--set-block-profile <board>`: run `bake_block_profile --json` on the fulleval's own
    bake (sha-gated against `bake_sha256`) and store the static feature-block usage
    fingerprint as `block_profile` (f156-371 were ZEROED by the folded regimes — slots
    preserved per the append-only discipline, not removed).

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
import math
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


def _load_board(board: Path):
    b = board.read_bytes()
    return json.loads(b)


def _write_board_gated(board: Path, before: dict, after: dict, allowed_keys: set[str],
                       dry_run: bool, what: str) -> bool:
    """Integrity gate shared by all surgical modes: every key other than
    `allowed_keys` must be byte-identical between before/after."""
    for k in set(before) | set(after):
        if k in allowed_keys:
            continue
        if _jc(before.get(k)) != _jc(after.get(k)):
            raise SystemExit(f"{what}: block `{k}` changed — refusing to write")
    if dry_run:
        print(f"[dry-run] would {what} in {board}")
        return True
    board.write_text(json.dumps(after))
    print(f"{what}: wrote {board}")
    return True


def graft_rank_corpus(board: Path, verdict: Path, corpus: str, dry_run: bool = False) -> bool:
    """Copy `rank.<corpus>` from a same-bake verdict into a board fulleval that
    lacks it (the era-bridge hfnlproxy fill path, board-integrity pass
    2026-08-04). Sha-gated; every other key byte-identical; provenance in
    `rank_graft_sources.<corpus>`."""
    bdoc = _load_board(board)
    v_bytes = verdict.read_bytes()
    v = json.loads(v_bytes)
    blk = (v.get("rank") or {}).get(corpus)
    if not isinstance(blk, dict):
        raise SystemExit(f"graft-rank: {verdict} carries no rank.{corpus} block")
    if bdoc.get("bake_sha256") != v.get("bake_sha256"):
        raise SystemExit(f"graft-rank: bake_sha256 mismatch — {board.name} is not the same "
                         f"bake as {verdict.name}; refusing")
    if isinstance((bdoc.get("rank") or {}).get(corpus), dict):
        print(f"graft-rank: {board.name} already has rank.{corpus} — unchanged")
        return False
    doc = copy.deepcopy(bdoc)
    doc.setdefault("rank", {})[corpus] = blk
    srcs = dict(doc.get("rank_graft_sources") or {})
    srcs[corpus] = {"path": str(verdict), "sha256": hashlib.sha256(v_bytes).hexdigest(),
                    "name": v.get("name")}
    doc["rank_graft_sources"] = srcs
    # rank changed only by ADDING the corpus key — assert the rest of rank.
    for k in bdoc.get("rank") or {}:
        if _jc((bdoc["rank"] or {}).get(k)) != _jc(doc["rank"].get(k)):
            raise SystemExit(f"graft-rank: rank.{k} changed — refusing to write")
    return _write_board_gated(board, bdoc, doc, {"rank", "rank_graft_sources"},
                              dry_run, f"graft rank.{corpus} ({v.get('name')})")


def reslice_rank(board: Path, verdict: Path, corpora, dry_run: bool = False) -> bool:
    """Replace `rank.<corpus>` blocks with a same-bake verdict scored on the
    FAMILY-AWARE re-sliced eval tables (registered program 2026-08-28, user
    decision: structural re-slice + full-board rescore; zensim
    benchmarks/imazen26_dhash_audit_2026-08-27.md ★REGISTERED section).

    A REPLACEMENT graft: sha-gated to the same bake, provenance per corpus in
    `rank_graft_sources.<corpus>` carrying the reslice tag + the superseded
    srocc, and the everything-else-byte-identical gate of the other surgical
    modes (only rank.<named corpora> + rank_graft_sources may change)."""
    bdoc = _load_board(board)
    v_bytes = verdict.read_bytes()
    v = json.loads(v_bytes)
    if bdoc.get("bake_sha256") != v.get("bake_sha256"):
        raise SystemExit(f"reslice-rank: bake_sha256 mismatch — {board.name} vs {verdict.name}; refusing")
    doc = copy.deepcopy(bdoc)
    srcs = dict(doc.get("rank_graft_sources") or {})
    changed = []
    for corpus in corpora:
        blk = (v.get("rank") or {}).get(corpus)
        if not isinstance(blk, dict):
            print(f"reslice-rank: {verdict.name} has no rank.{corpus} — skipped")
            continue
        old = (bdoc.get("rank") or {}).get(corpus)
        doc.setdefault("rank", {})[corpus] = blk
        srcs[corpus] = {"path": str(verdict), "sha256": hashlib.sha256(v_bytes).hexdigest(),
                        "name": v.get("name"), "reslice": "family-aware-2026-08-28",
                        "superseded_srocc": (old or {}).get("srocc")}
        changed.append(corpus)
    if not changed:
        return False
    doc["rank_graft_sources"] = srcs
    for k in set(bdoc.get("rank") or {}) | set(doc.get("rank") or {}):
        if k in changed:
            continue
        if _jc((bdoc.get("rank") or {}).get(k)) != _jc((doc.get("rank") or {}).get(k)):
            raise SystemExit(f"reslice-rank: rank.{k} changed — refusing to write")
    return _write_board_gated(board, bdoc, doc, {"rank", "rank_graft_sources"},
                              dry_run, f"reslice rank.{{{','.join(changed)}}}")


def repair_rank_orientation(board: Path, verdict: Path, corpus: str,
                            dry_run: bool = False) -> bool:
    """Replace `rank.<corpus>` with a fresh same-bake verdict's block when the
    stored block was produced BEFORE the per-ref orientation pin (`730a386e`,
    2026-08-04 16:49) and the bake's pooled signed SROCC is negative — the
    Orientation::Auto pooled-sign flip (SOTA-944 appendix O finding: 80 board
    cells carried `per_ref_mean`/`frac_negative` sign-flipped vs the pinned
    quality-orientation convention).

    This is a CORRECTION, not a re-measurement, and the gates enforce that:
    the fresh block must reproduce every orientation-INDEPENDENT field of the
    stored block to exact float equality (srocc, srocc_signed, plcc, krocc,
    or, pwrc, z_rmse, n, per_ref_n, srocc_ci, ...); only `per_ref_mean` and
    `frac_negative` (the per_group_srocc orientation-dependent outputs) may
    differ, and `per_ref_mean` must be an exact sign flip. Provenance lands in
    `rank_graft_sources.<corpus>` with the superseded value."""
    ORIENT_DEP = {"per_ref_mean", "frac_negative"}
    bdoc = _load_board(board)
    v_bytes = verdict.read_bytes()
    v = json.loads(v_bytes)
    blk = (v.get("rank") or {}).get(corpus)
    if not isinstance(blk, dict):
        raise SystemExit(f"repair-rank: {verdict} carries no rank.{corpus} block")
    if bdoc.get("bake_sha256") != v.get("bake_sha256"):
        raise SystemExit(f"repair-rank: bake_sha256 mismatch — {board.name} is not the same "
                         f"bake as {verdict.name}; refusing")
    old = (bdoc.get("rank") or {}).get(corpus)
    if not isinstance(old, dict):
        raise SystemExit(f"repair-rank: {board.name} has no rank.{corpus} to repair "
                         f"(use --graft-rank for the fill path)")
    for k in set(old) | set(blk):
        if k in ORIENT_DEP:
            continue
        if _jc(old.get(k)) != _jc(blk.get(k)):
            raise SystemExit(f"repair-rank: rank.{corpus}.{k} differs between stored and fresh "
                             f"verdict — this is not an orientation-only correction; refusing")
    o, n = old.get("per_ref_mean"), blk.get("per_ref_mean")
    if o is None or n is None or abs(n + o) > 1e-9 or abs(n - o) <= 1e-12:
        raise SystemExit(f"repair-rank: per_ref_mean stored={o} fresh={n} is not an exact sign "
                         f"flip; refusing (nothing to repair, or a different defect)")
    doc = copy.deepcopy(bdoc)
    doc["rank"][corpus] = blk
    srcs = dict(doc.get("rank_graft_sources") or {})
    srcs[corpus] = {"path": str(verdict), "sha256": hashlib.sha256(v_bytes).hexdigest(),
                    "name": v.get("name"),
                    "repair": "per-ref-orientation-pin-730a386e",
                    "superseded_per_ref_mean": o,
                    "superseded_frac_negative": old.get("frac_negative")}
    doc["rank_graft_sources"] = srcs
    for k in bdoc.get("rank") or {}:
        if k != corpus and _jc((bdoc["rank"] or {}).get(k)) != _jc(doc["rank"].get(k)):
            raise SystemExit(f"repair-rank: rank.{k} changed — refusing to write")
    return _write_board_gated(board, bdoc, doc, {"rank", "rank_graft_sources"},
                              dry_run, f"repair rank.{corpus} orientation ({v.get('name')}: "
                                       f"{o:+.4f} -> {n:+.4f})")


def rebuild_bands(board: Path, corpora: list[str], dry_run: bool = False) -> bool:
    """Recut `rank.<corpus>.bands` under the current band scheme, from the
    cell's OWN stored per-pair (campaign appendix V).

    This is a RE-CUT, not a re-measurement: the predictions and the human
    targets are the ones the cell was published with, so nothing is re-scored
    and no bake is loaded. Only where the cuts fall changes — and with it the
    per-band statistics, which are computed by the canonical `panel --batch`
    owner (zero stat math here).

    Only cells that still carry per-pair can be recut. A cell whose per-pair was
    stripped by the board-size rule keeps its legacy bands and is reported as
    such; `freeze_check` prints those as ABSENT rather than scoring them, and
    `benchmarks/eval_annotations.json` carries the reason.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.band_reliability import scheme_merge, band_members  # noqa: E402
    from scripts.lib import zen_stats  # noqa: E402
    import numpy as np  # noqa: E402

    # Floors + scheme name come from the OWNER's committed parity fixture, so a
    # rebuild can never be cut on constants that differ from what ships.
    fixture = Path(__file__).resolve().parent.parent / "benchmarks/appendixV/band_scheme_parity.tsv"
    n_min = span_min = None
    for line in fixture.read_text().splitlines():
        if line.startswith("# floors:"):
            for tok in line.split():
                if tok.startswith("n_min="):
                    n_min = int(tok.split("=")[1])
                elif tok.startswith("span_min="):
                    span_min = float(tok.split("=")[1])
    if n_min is None or span_min is None:
        raise SystemExit("rebuild-bands: floors not found in the parity fixture")
    scheme_name = "merged-decile-2026-08-06"

    bdoc = _load_board(board)
    doc = copy.deepcopy(bdoc)
    touched, skipped = [], []
    for corpus in corpora:
        blk = (doc.get("rank") or {}).get(corpus)
        if not isinstance(blk, dict) or not blk.get("bands"):
            continue
        pp = (doc.get("per_pair") or {}).get(corpus)
        tkey = next((k for k in ("mos", "jnd") if pp and pp.get(k)), None) if pp else None
        if not tkey or not pp.get("pred"):
            continue  # per-pair stripped: leave the legacy bands in place
        t = np.asarray(pp[tkey], dtype=float)
        p = np.asarray(pp["pred"], dtype=float)
        if t.shape != p.shape:
            raise SystemExit(f"rebuild-bands: {board.name} {corpus} per-pair shape mismatch")
        # The recut must cover the SAME rows the corpus aggregate covers, or the
        # band n's and `rank.<c>.n` describe different populations. KADID's
        # stored per-pair is a 5,000-row subsample of its 10,125 — recutting it
        # would publish bands on half the corpus under a header claiming all of
        # it. Skipped and annotated instead (appendix V confound 4).
        n_corpus = blk.get("n")
        if isinstance(n_corpus, int) and n_corpus != int(t.size):
            skipped.append(f"{corpus}(per-pair {t.size} != rank.n {n_corpus})")
            continue

        defs = scheme_merge(t, n_min, span_min)
        jobs, meta = [], []
        for label, lo, hi in defs:
            idx = band_members(t, lo, hi)
            span = float(t[idx].max() - t[idx].min()) if idx.size else 0.0
            span = span if math.isfinite(span) else 0.0
            meta.append((label, lo, hi, idx, span))
            if idx.size >= 4:
                jobs.append((label.replace("-", "_"), p[idx], t[idx]))
        stats = {r["label"]: r for r in zen_stats.panel_batch(jobs, stats="full")} if jobs else {}

        rows = []
        for label, lo, hi, idx, span in meta:
            n = int(idx.size)
            reason = _not_measured_reason(n, span, n_min, span_min)
            # JSON has no infinity: an open end serialises as null and MUST be
            # read as "unbounded", never as a missing value. (Python's json
            # module would otherwise emit bare `-Infinity`, which is not JSON
            # and which the Rust reader rejects outright.)
            row = {
                "band": label,
                "lo": None if math.isinf(lo) else lo,
                "hi": None if math.isinf(hi) else hi,
                "n": n,
                "span": span,
                "not_measured_reason": reason,
            }
            st = stats.get(label.replace("-", "_"))
            keys = ("srocc", "srocc_signed", "plcc", "krocc", "or", "pwrc", "z_rmse", "mae")
            if reason is None and st is not None:
                row.update({k: (None if not math.isfinite(st[k]) else st[k])
                            for k in keys})
            else:
                row.update({k: None for k in keys})
            rows.append(row)
        blk["bands"] = rows
        blk["band_scheme"] = {
            "name": scheme_name,
            "base_bands": 10,
            "n_min": n_min,
            "span_min": span_min,
            "doc": "campaign appendix V: fixed deciles accumulated into the finest "
                   "partition whose every band has n >= n_min AND target span >= span_min",
            "recut_from": "stored per_pair (no rescore; predictions unchanged)",
        }
        touched.append(f"{corpus}:{len(rows)}")

    if skipped:
        print(f"rebuild-bands: {board.name} — SKIPPED {'; '.join(skipped)} "
              "(legacy bands kept; freeze_check reports them ABSENT)")
    if not touched:
        print(f"rebuild-bands: {board.name} — no recuttable corpus (per-pair stripped?)")
        return False
    return _write_board_gated(board, bdoc, doc, {"rank"}, dry_run,
                              f"recut bands [{','.join(touched)}]")


def _not_measured_reason(n: int, span: float, n_min: int, span_min: float):
    """Mirror of `zensim_validate::bands::not_measured_reason` (the owner).
    Kept string-identical so a recut cell reads the same as a freshly emitted
    one; the parity fixture gates the EDGES, this gates the wording."""
    if n == 0:
        return "empty: no pairs in this target range"
    if n < n_min and span < span_min:
        return (f"n={n} < {n_min} and span={span:.4f} < {span_min}: "
                "too few pairs AND too narrow to resolve")
    if n < n_min:
        return f"n={n} < {n_min}: too few pairs to rank models"
    if span < span_min:
        return (f"span={span:.4f} < {span_min}: range-restricted, "
                "correlation attenuated toward 0")
    return None


def mark_dominated(board: Path, dominated_by: list[str], rule: str,
                   dry_run: bool = False) -> bool:
    """Write `dominated_by` (+ `dominance` provenance) into a board fulleval —
    the Pareto-trim mechanism (board-integrity pass 2026-08-04). NEVER deletes
    or alters any statistic; the board renders dominated cells dimmed +
    default-off. Empty list clears the mark."""
    bdoc = _load_board(board)
    doc = copy.deepcopy(bdoc)
    if dominated_by:
        doc["dominated_by"] = sorted(dominated_by)
        doc["dominance"] = {"rule": rule, "date": "2026-08-04"}
    else:
        doc.pop("dominated_by", None)
        doc.pop("dominance", None)
    if _jc(bdoc.get("dominated_by")) == _jc(doc.get("dominated_by")) and \
       _jc(bdoc.get("dominance")) == _jc(doc.get("dominance")):
        print(f"mark-dominated: {board.name} unchanged")
        return False
    return _write_board_gated(board, bdoc, doc, {"dominated_by", "dominance"},
                              dry_run, f"mark dominated_by={sorted(dominated_by)}")


def set_block_profile(board: Path, bbp_bin: str, dry_run: bool = False) -> bool:
    """Compute the static feature-block usage fingerprint from the fulleval's
    own bake bytes (`bake_block_profile --json`, sha-gated against
    `bake_sha256`) and store it as `block_profile`. Missing bake file ⇒
    reported and skipped (block_profile stays absent — never fabricated)."""
    import subprocess
    bdoc = _load_board(board)
    bake = bdoc.get("bake")
    if not bake or not Path(bake).exists():
        print(f"block-profile: {board.name}: bake missing on disk ({bake}) — skipped")
        return False
    want_sha = bdoc.get("bake_sha256")
    got_sha = hashlib.sha256(Path(bake).read_bytes()).hexdigest()
    if want_sha and got_sha != want_sha:
        raise SystemExit(f"block-profile: {board.name}: bake sha mismatch on disk "
                         f"({got_sha[:12]} != {want_sha[:12]}) — refusing")
    r = subprocess.run([bbp_bin, "--bake", bake, "--json"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"block-profile: {board.name}: {bbp_bin} rc={r.returncode}: "
              f"{r.stderr.strip()} — skipped")
        return False
    prof = json.loads(r.stdout)
    doc = copy.deepcopy(bdoc)
    doc["block_profile"] = prof
    if _jc(bdoc.get("block_profile")) == _jc(prof):
        print(f"block-profile: {board.name} unchanged")
        return False
    return _write_board_gated(board, bdoc, doc, {"block_profile"}, dry_run,
                              "set block_profile")


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
    ap.add_argument("--verdict", default=None, type=Path,
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
    ap.add_argument("--reslice-rank", default=None, metavar="CORPORA",
                    help="comma list: REPLACE rank.<corpus> blocks from --verdict "
                         "(family-aware re-slice graft, sha-gated; use with --graft-into)")
    ap.add_argument("--graft-rank", default=None, metavar="CORPUS",
                    help="with --graft-into: graft rank.<CORPUS> from --verdict instead of "
                         "corruption_head (era-bridge hfnlproxy fill; sha-gated)")
    ap.add_argument("--repair-rank-orientation", default=None, metavar="CORPUS",
                    help="with --graft-into: replace a PRE-orientation-pin rank.<CORPUS> block "
                         "with --verdict's pinned block (sha-gated; every orientation-independent "
                         "field must be float-identical and per_ref_mean an exact sign flip; "
                         "appendix O per-ref flip repair)")
    ap.add_argument("--mark-dominated", default=None, type=Path, metavar="BOARD_JSON",
                    help="DOMINANCE mode: board fulleval to receive dominated_by (see --dominated-by)")
    ap.add_argument("--dominated-by", default="",
                    help="comma list of same-class dominator names (empty = clear the mark)")
    ap.add_argument("--dominance-rule", default="strict-pareto-2026-08-04",
                    help="rule id recorded in the `dominance` provenance block")
    ap.add_argument("--rebuild-bands", default=None, type=Path, metavar="BOARD_JSON",
                    help="recut rank.<corpus>.bands under the current band scheme "
                         "from the cell's own stored per-pair (campaign appendix V); "
                         "no rescore, no bake load")
    ap.add_argument("--band-corpora", default="cid22,csiq,kadid,live,tid",
                    help="with --rebuild-bands: which banded corpora to recut")
    ap.add_argument("--set-block-profile", default=None, type=Path, metavar="BOARD_JSON",
                    help="BLOCK-PROFILE mode: compute the static feature-block fingerprint from "
                         "the fulleval's bake bytes (bake_block_profile --json, sha-gated) and "
                         "store it as `block_profile`")
    ap.add_argument("--bbp-bin", default=None,
                    help="bake_block_profile binary (default: $BBP_BIN or "
                         "target/release/bake_block_profile next to this repo)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    if a.mark_dominated is not None:
        if a.verdict or a.name or a.members or a.members_file:
            ap.error("--mark-dominated takes only --dominated-by/--dominance-rule/--dry-run")
        if not a.mark_dominated.exists():
            raise SystemExit(f"mark-dominated: board file not found: {a.mark_dominated}")
        dom = [x for x in a.dominated_by.split(",") if x]
        mark_dominated(a.mark_dominated, dom, a.dominance_rule, a.dry_run)
        return 0

    if a.rebuild_bands is not None:
        if a.verdict or a.name or a.members or a.members_file:
            ap.error("--rebuild-bands takes only --band-corpora/--dry-run")
        if not a.rebuild_bands.exists():
            raise SystemExit(f"rebuild-bands: board file not found: {a.rebuild_bands}")
        rebuild_bands(a.rebuild_bands, a.band_corpora.split(","), a.dry_run)
        return 0

    if a.set_block_profile is not None:
        if a.verdict or a.name or a.members or a.members_file:
            ap.error("--set-block-profile takes only --bbp-bin/--dry-run")
        if not a.set_block_profile.exists():
            raise SystemExit(f"block-profile: board file not found: {a.set_block_profile}")
        import os
        bbp = a.bbp_bin or os.environ.get("BBP_BIN") or str(
            Path(__file__).resolve().parent.parent / "target/release/bake_block_profile")
        set_block_profile(a.set_block_profile, bbp, a.dry_run)
        return 0

    if a.verdict is None or not a.verdict.exists():
        raise SystemExit(f"promote: verdict not found: {a.verdict}")

    if a.graft_into is not None:
        if a.name or a.members or a.members_file or a.strip_per_pair or a.carry_coherence_from:
            ap.error("--graft-into takes only --verdict/--graft-rank/--reslice-rank/--repair-rank-orientation "
                     "(and --dry-run)")
        if a.graft_rank and a.repair_rank_orientation:
            ap.error("--graft-rank and --repair-rank-orientation are mutually exclusive")
        if not a.graft_into.exists():
            raise SystemExit(f"graft: board file not found: {a.graft_into}")
        if a.reslice_rank:
            reslice_rank(a.graft_into, a.verdict, a.reslice_rank.split(","), a.dry_run)
        elif a.repair_rank_orientation:
            repair_rank_orientation(a.graft_into, a.verdict, a.repair_rank_orientation, a.dry_run)
        elif a.graft_rank:
            graft_rank_corpus(a.graft_into, a.verdict, a.graft_rank, a.dry_run)
        else:
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
