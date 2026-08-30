#!/usr/bin/env python3
"""promote_era372_board.py — put the CURRENT-EXTRACTOR 372 verdicts on the gauntlet board.

Round 4b (benchmarks/eval372_current_root_2026-08-30.md) re-verdicted the 372-class roster
on a NEW dated eval root built with today's extractor
(/mnt/v/zen/zensim-training/2026-08-30-full-features-372, _MANIFEST.json build_commit
ea16c7ee), because the 2026-05-15 root's masked/IW block (f156-371) was a function of
RAYON_NUM_THREADS and does not reproduce at its own build commit (docs/DATASET_HISTORY.md
§3.27). The shift is MODEL-SPECIFIC — 0.00000 on the three basic-only controls, |0.489| on
cl_tfm's KonJND — so there is no correction factor and 41 orderings flip.

This is a CALLER of the one promotion owner (scripts/promote_fulleval.py — it recomputes
NOTHING); the policy encoded here is:

  * NAMING: `<stored-era board name>` + `gauntlet.ERA372_CUR_SUFFIX` ("@cur372"). Same stem,
    era suffix — so the two halves of a pair sort together, share every `family_of` prefix
    rule, and a reader can never mistake one for the other. `@` appears in no other board
    name, so the suffix test is unambiguous. (This supersedes the `__r372cur` spelling
    *suggested* in eval372_current_root_2026-08-30.md §6.2 — nothing was promoted under it.)
  * The STORED-ERA ROW IS NEVER OVERWRITTEN. It stays on the board, flagged by
    benchmarks/eval_annotations.json (eval372-stored-root-thread-dependent-2026-08-30 for
    the six f156-371 users; eval372-basic-only-bakes-era-independent-2026-08-30 for the
    three measured-immune controls). A byte gate below re-reads every stored-era file after
    the run and fails if one changed.
  * PAIR GATE: where a stored-era row exists, its `bake_sha256` must equal the verdict's —
    an era pair is the SAME BAKE read on two rulers, or it is not a pair.
  * M3/M3a are carried from the stored-era row when measured (`--carry-coherence-from`, the
    registered carry rule): coherence is a property of the BAKE measured by
    diffmap_block_coherence on images, not a function of the 372 eval root. Provenance lands
    in `coherence_source`. No stored row ⇒ null ⇒ the board renders NOT MEASURED.
  * `block_profile` is (re)computed from the bake bytes (`bake_block_profile`, sha-gated) so
    the board's "uses f156-371" chip — the exact discriminator between the invalidated and
    the era-immune halves of this roster — is populated on the new rows too.
  * Registered size rule: per-pair scatter is kept only for the curated headline set
    (gauntlet.py CURATED_BOARD — the one owner, imported here); the rest are promoted
    --strip-per-pair (every scalar stat stays; the full data remains in the source verdict,
    which promotion records by path + sha256).

Two roster members have NO stored-era board row (`mlp_2L_diverse_H128`, `v02_bvls_shaped`)
— their `@cur372` row is the only one, and no stored-era counterpart is created, because
that would add fresh un-annotated extinct-era numbers to the board.

A mapping index lands at <out-dir>/_era372_board_map.tsv.

Usage:
    promote_era372_board.py [--json-dir DIR] [--out-dir DIR] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))             # promote_fulleval
sys.path.insert(0, str(Path(__file__).resolve().parent / "v_next"))  # gauntlet (CURATED)

from promote_fulleval import promote, set_block_profile  # noqa: E402
from gauntlet import CURATED, ERA372_CUR_SUFFIX  # noqa: E402

DEFAULT_JSON = Path("/mnt/v/output/zensim/eval372-roster-2026-08-30/json")
DEFAULT_OUT = Path("/mnt/v/output/zensim/reports/fulleval")
DEFAULT_BBP = str(Path(__file__).resolve().parents[1] / "target" / "release" / "bake_block_profile")

# FROZEN roster map: round-4b verdict label -> stored-era board stem.
# Established by bake_sha256 identity against the board (not by name similarity — the
# verdicts' own `name` field is the bake stem, which differs from the board name on 8 of 11).
# `None` = no stored-era board row exists for this bake.
ROSTER = [
    ("B_shipped",           "b_sdr_linear_cid80_inclwinsor_dense_dial"),
    ("blend_2L_H128",       "mlp_2L_diverse_H128"),                      # no stored-era row
    ("cl_tfm_LQ_MLP",       "cl_tfm_corruption_LQ_MLP_s13"),
    ("v02_bvls_NO_shaping", "v02_bvls_NO_shaping"),
    ("v02_bvls_shaped",     "v02_bvls_shaped"),                          # no stored-era row
    ("v47A_strict_QAT",     "v47_strict_QAT_native"),
    ("T_b_lam1e-3",         "T_appT_b372_lam1e-3"),
    ("BHdr_sdr_route",      "bhdr_linear_shaped_cvvdpmix"),
    ("ADD156",              "ADD156_safesyn_only_raw_lasso"),            # era-immune control
    ("Ebothg_scr05",        "Ebothg_scr0_5_dial"),                       # era-immune control
    ("winner_dial",         "winner_dial_Ebothg_hfgain_winsor_dial"),    # era-immune control
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-dir", default=DEFAULT_JSON, type=Path)
    ap.add_argument("--out-dir", default=DEFAULT_OUT, type=Path)
    ap.add_argument("--bbp-bin", default=DEFAULT_BBP)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    # Snapshot every stored-era file we are pairing against, so we can prove afterwards
    # that promotion did not touch one (the "never overwrite a stored-era row" rule).
    stored_before = {}
    for _, stem in ROSTER:
        sp = a.out_dir / f"{stem}.fulleval.json"
        if sp.exists():
            stored_before[stem] = _sha(sp)

    rows, n_curated, n_stripped, n_carried, n_blk = [], 0, 0, 0, 0
    for label, stem in ROSTER:
        verdict = a.json_dir / f"{label}_new.json"
        if not verdict.exists():
            raise SystemExit(f"missing current-era verdict: {verdict}")
        target = stem + ERA372_CUR_SUFFIX
        if target == stem:
            raise SystemExit("era suffix is empty — refusing to overwrite a stored-era row")
        v = json.loads(verdict.read_text())
        stored = a.out_dir / f"{stem}.fulleval.json"

        carry = None
        pairing = "unpaired (no stored-era row)"
        if stem in stored_before:
            sdoc = json.loads(stored.read_text())
            if sdoc.get("bake_sha256") != v.get("bake_sha256"):
                raise SystemExit(
                    f"PAIR GATE FAILED: {stem} board row is a different bake than "
                    f"{verdict.name} ({sdoc.get('bake_sha256')} vs {v.get('bake_sha256')})")
            pairing = "paired"
            if sdoc.get("m3_coherence") is not None or sdoc.get("m3a_coherence") is not None:
                carry = stored

        strip = target not in CURATED
        out = promote(verdict, target, members=None, out_dir=a.out_dir,
                      dry_run=a.dry_run, strip_per_pair=strip,
                      carry_coherence_from=carry)
        n_curated += 0 if strip else 1
        n_stripped += 1 if strip else 0
        blk = False
        if not a.dry_run:
            doc = json.loads(out.read_text())
            if doc.get("coherence_source"):
                n_carried += 1
            blk = set_block_profile(out, a.bbp_bin, dry_run=False)
            n_blk += 1 if blk else 0
        cid = ((v.get("rank") or {}).get("cid22") or {}).get("srocc")
        rows.append((label, target, pairing,
                     "curated" if not strip else "grid-interior (per_pair stripped)",
                     "coherence carried" if carry is not None else "coherence NOT MEASURED",
                     "block_profile set" if blk else "block_profile absent",
                     f"{cid:.5f}" if isinstance(cid, (int, float)) else "?"))

    # NEVER-OVERWRITE gate: every stored-era file byte-identical to its pre-run snapshot.
    if not a.dry_run:
        for stem, sha in stored_before.items():
            now = _sha(a.out_dir / f"{stem}.fulleval.json")
            if now != sha:
                raise SystemExit(f"STORED-ERA ROW CHANGED: {stem}.fulleval.json ({sha} -> {now})")
        print(f"never-overwrite gate PASS: {len(stored_before)} stored-era rows byte-identical")

        idx = a.out_dir / "_era372_board_map.tsv"
        idx.write_text("# verdict_label\tboard_name\tpairing\tcuration\tcoherence\tblock_profile\tcid22\n"
                       + "\n".join("\t".join(r) for r in rows) + "\n")
        print(f"map -> {idx}")

    n_board = len(list(a.out_dir.glob("*.fulleval.json")))
    print(f"promoted={len(rows)} curated={n_curated} stripped={n_stripped} "
          f"coherence_carried={n_carried} block_profile={n_blk} | board files now: {n_board}"
          f"{' (dry-run: unchanged)' if a.dry_run else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
