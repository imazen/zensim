#!/usr/bin/env python3
"""promote_sota944_board.py — put EVERY SOTA-944 campaign verdict cell on the gauntlet board.

The coverage fix (dashboard overhaul, 2026-08-04): the campaign produced ~149 result cells
under /mnt/v/output/zensim/bakes/sota944/verdicts/ while the board showed ~45 — every number
cited in benchmarks/sota944_campaign_2026-08-03.md must be findable on the board. This is a
CALLER of the one promotion owner (scripts/promote_fulleval.py — it computes nothing); the
policy encoded here is:

  * EXCLUDED (instrument/duplicate cells, not campaign results): LOO_* (masked-root
    ablation rescores), REPROCHK_* / C_co3arepro_s1301 (bit-identical reproduction
    checks), XBUILDCHK_* (fresh-build checks), W5GATE_* / W6GATE_* (k=1 identity gates),
    *_recheck (post-cleanup re-runs), SMOKE (harness smoke test, different corpora).
  * *_corrjoint verdicts fold UNDER THE PLAIN BOARD NAME: every one of them re-verdicts a
    bake that is already promoted, so its corruption_head block is GRAFTED into the
    existing board file (sha-gated; the richer full-eval content — measured M3a, kadis
    per-pair — is untouched).
  * Existing board names stay stable (RENAMED maps the campaign stems that were promoted
    under sota944_* names); new cells get consistent `sota944_<stem>` names
    (W6 ensembles follow wave-5's convention: `sota944_ens_<stem-sans-W6_>`).
  * W6 G-E cells are ENSEMBLES — promoted with the frozen §6.2 member lists (mirrors
    scripts/wave6_konjnd_ensemble.sh, the runner) so the board marks them ens×k and
    never presents the anchor member's architecture as a shippable artifact.
  * Registered size rule: per-pair scatter is kept only for the curated headline set
    (gauntlet.py CURATED_BOARD — the one owner, imported here); grid-interior cells are
    promoted --strip-per-pair (all scalar stats stay; the full data remains in the
    source verdict, which promotion records by path + sha256).

Ends with a COVERAGE GATE: every non-excluded verdict stem must map to an existing board
file, or the run fails. A mapping index lands at <out-dir>/_sota944_board_map.tsv.

Usage:
    promote_sota944_board.py [--verdicts DIR] [--out-dir DIR] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # promote_fulleval
sys.path.insert(0, str(Path(__file__).resolve().parent / "v_next"))  # gauntlet (CURATED)

from promote_fulleval import graft_corruption_head, promote  # noqa: E402
from gauntlet import CURATED  # noqa: E402

DEFAULT_VD = Path("/mnt/v/output/zensim/bakes/sota944/verdicts")
DEFAULT_OUT = Path("/mnt/v/output/zensim/reports/fulleval")

# Instrument / duplicate cells — excluded from the board (documented in the docstring).
EXCLUDE = re.compile(r"^(LOO_|REPROCHK|XBUILDCHK|W5GATE|W6GATE)|_recheck$|^SMOKE$"
                     r"|^C_co3arepro_s1301$")

# Campaign stems already promoted under a DIFFERENT (stable) board name.
RENAMED = {
    "A_bvls_X_AM5_w": "sota944_winner_A_bvls_X_AM5",
    "C_em944_s31": "sota944_C_em944_s31",
    "C_nt944_s223": "sota944_nt223",
    "W5_E1_k2": "sota944_ens_E1_k2",
    "W5_E1_k3": "sota944_ens_E1_k3",
    "W5_E1_k5": "sota944_ens_E1_k5",
    "W5_E1_k8": "sota944_ens_E1_k8",
    "W5_E2_diverse5": "sota944_ens_E2_diverse5",
    "W5_E3_all51": "sota944_ens_E3_all51",
}

# Frozen W6 §6.2 G-E membership (mirrors scripts/wave6_konjnd_ensemble.sh).
W6_MEMBERS = {
    "W6_GE1_konpair": ["C_co3a_s1301", "C_co4_s1307"],
    "W6_GE2_trio": ["C_co3a_s1301", "C_co3a_s1307", "C_em944_s31"],
    "W6_GE3_balanced5": ["C_co3a_s1301", "C_co3a_s1319", "C_co3a_s1307",
                          "C_em944_s31", "C_co4_s1307"],
    "W6_GE4_konfloor5": ["C_em944_s31", "C_co3a_s1307", "C_co4_s1307",
                          "C_co4_s1301", "C_co3a_s1327"],
    "W6_GE5_w5plus3": ["C_co3a_s1301", "C_co2a_s1307", "C_co3a_s1319", "C_co1b_s1303",
                        "C_em944_s31", "C_co3a_s1307", "C_co4_s1307", "C_em944_s127"],
}


def board_name(stem: str, out_dir: Path) -> str:
    if stem in RENAMED:
        return RENAMED[stem]
    if stem in W6_MEMBERS:
        return "sota944_ens_" + stem[len("W6_"):]
    if (out_dir / f"{stem}.fulleval.json").exists():
        return stem                      # promoted under the plain name (wave-3/4 cells)
    return f"sota944_{stem}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdicts", default=DEFAULT_VD, type=Path)
    ap.add_argument("--out-dir", default=DEFAULT_OUT, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    stems = sorted(p.name[:-len(".full.json")] for p in a.verdicts.glob("*.full.json"))
    rows, promoted, kept, grafted, excluded = [], 0, 0, 0, 0

    # Pass 1 — plain cells (corrjoint handled in pass 2).
    for stem in stems:
        if stem.endswith("_corrjoint"):
            continue
        if EXCLUDE.search(stem):
            excluded += 1
            rows.append((stem, "(excluded: instrument/duplicate)", "excluded"))
            continue
        target = board_name(stem, a.out_dir)
        out = a.out_dir / f"{target}.fulleval.json"
        if out.exists():
            kept += 1
            rows.append((stem, target, "kept (already promoted)"))
            continue
        promote(a.verdicts / f"{stem}.full.json", target,
                members=W6_MEMBERS.get(stem), out_dir=a.out_dir, dry_run=a.dry_run,
                strip_per_pair=target not in CURATED)
        promoted += 1
        rows.append((stem, target, "promoted" + ("" if target in CURATED
                                                 else " (per_pair stripped)")))

    # Pass 2 — corrjoint grafts into the (now guaranteed-present) plain board files.
    for stem in stems:
        if not stem.endswith("_corrjoint"):
            continue
        plain = stem[:-len("_corrjoint")]
        target = board_name(plain, a.out_dir)
        out = a.out_dir / f"{target}.fulleval.json"
        if not out.exists():
            raise SystemExit(f"corrjoint graft target missing: {out} (from {stem})")
        if graft_corruption_head(out, a.verdicts / f"{stem}.full.json", a.dry_run):
            grafted += 1
        rows.append((stem, target, "corruption_head grafted"))

    # COVERAGE GATE: every non-excluded stem maps to a board file.
    missing = []
    for stem in stems:
        if stem.endswith("_corrjoint") or EXCLUDE.search(stem):
            continue
        target = board_name(stem, a.out_dir)
        if not (a.out_dir / f"{target}.fulleval.json").exists() and not a.dry_run:
            missing.append((stem, target))
    if missing:
        raise SystemExit(f"COVERAGE GATE FAILED — {len(missing)} cells not on the board: "
                         + ", ".join(f"{s}->{t}" for s, t in missing[:10]))

    if not a.dry_run:
        idx = a.out_dir / "_sota944_board_map.tsv"
        idx.write_text("# verdict_stem\tboard_name\taction\n"
                       + "\n".join("\t".join(r) for r in rows) + "\n")
        print(f"map -> {idx}")
    n_board = len(list(a.out_dir.glob("*.fulleval.json")))
    print(f"promoted={promoted} kept={kept} grafted={grafted} excluded={excluded} "
          f"| board files now: {n_board}{' (dry-run: unchanged)' if a.dry_run else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
