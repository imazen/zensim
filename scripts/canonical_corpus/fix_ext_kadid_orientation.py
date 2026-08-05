#!/usr/bin/env python3
"""Rebuild the ext-lineage `ext_kadid.parquet` with its target orientation CORRECTED.

**What was wrong.** `build_fr_corpus_pairs.build_kadid()` applied the standard
invert-a-DMOS reflex to KADID's `dmos` column, which is a MOS in disguise: the raw
crowdsourced DCR ratings FALL with severity (4.0789 -> 2.0072 across levels 1-5, over
349,800 ratings), so the column was already quality-oriented. The ext roots therefore
stored `human_score = (5 - dmos)/4`, the exact inverse of the canonical `(dmos - 1)/4`.
Determination: `benchmarks/sota944_campaign_2026-08-03.md` REGISTERED APPENDIX F.
Remediation registration (this script's contract): REGISTERED APPENDIX H, section H.1.

**The transform.** `human_score := 1 - human_score`, and nothing else. That is an exact
algebraic identity, not a re-derivation:

    1 - (5 - dmos)/4  ==  (dmos - 1)/4

Verified numerically against the independently-built canonical root
(`canonical-2026-05-21/train/kadid.parquet`): over all 10,125 rows in identical
`ref_basename` order, `ext.human_score + canonical.human_score == 1` **exactly**
(min == max == 1.0), so `1 - ext` reproduces the canonical target to within one ULP.

**Every other column is carried through byte-identically** -- same values, same dtype,
same column order, same single row group, same ZSTD codec. Only `human_score` moves.

**File placement (registered, H.1).** The corrected table takes the CANONICAL name
`ext_kadid.parquet`; the inverted original is PRESERVED, never deleted, as
`ext_kadid_INVERTED_2026-08-04.parquet` beside it. Rationale: a gate nobody can turn
green is a gate that gets ignored -- leaving the canonical name inverted means every
future recipe must remember to override the path, which is exactly the failure mode
that let this bug live six weeks.

**Registered hazard.** Re-running any pre-2026-08-05 bake's embedded `zentrain.repro`
argv verbatim will now train against the CORRECTED table and will NOT reproduce that
bake. The repro's per-input `sha256` is the discriminator; the substitution needed
(`ext_kadid.parquet` -> `ext_kadid_INVERTED_2026-08-04.parquet`) is recorded in the dir
`_MANIFEST.json`, in `benchmarks/eval_annotations.json`, and in `docs/DATA_SPLITS.md`.

Usage:
    fix_ext_kadid_orientation.py [--root DIR]... [--dry-run] [--force]

Default roots are the three ext lineages. Idempotent: a root whose corrected table is
already in place (orientation OK and the preserved original present) is skipped.

Exit 0 = every requested root corrected or already correct; 1 = a post-write check
failed; 2 = usage/IO error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_ROOTS = [
    "/mnt/v/zen/zensim-training/ext720-canonical-2026-07-22",
    "/mnt/v/zen/zensim-training/ext924-canonical-2026-07-27",
    "/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01",
]
NAME = "ext_kadid.parquet"
PRESERVED = "ext_kadid_INVERTED_2026-08-04.parquet"
# The independently-built, correctly-oriented root -- used as a cross-check only.
CANONICAL_REF = "/mnt/v/zen/zensim-training/canonical-2026-05-21/train/kadid.parquet"


def sha256_file(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def cross_check(fixed: np.ndarray, refs: list[str]) -> dict:
    """Compare the corrected target against the canonical root (informational)."""
    try:
        can = pq.read_table(CANONICAL_REF, columns=["ref_basename", "human_score"]).to_pydict()
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": str(e)}
    if can["ref_basename"] != refs:
        return {"available": False, "reason": "ref_basename order differs; positional compare unsafe"}
    c = np.asarray(can["human_score"], dtype=float)
    return {
        "available": True,
        "path": CANONICAL_REF,
        "max_abs_diff": float(np.abs(fixed - c).max()),
        "bit_identical": bool(np.array_equal(fixed, c)),
    }


def fix_root(root: Path, dry_run: bool, force: bool) -> dict:
    src = root / NAME
    keep = root / PRESERVED
    rec: dict = {"root": str(root), "table": str(src)}
    if not src.exists():
        rec.update(status="MISSING")
        return rec

    f = pq.ParquetFile(src)
    meta = f.metadata
    rec.update(rows=meta.num_rows, cols=meta.num_columns, row_groups=meta.num_row_groups,
               compression=meta.row_group(0).column(0).compression)
    tbl = f.read()
    hs = np.asarray(tbl["human_score"].to_pylist(), dtype=float)
    refs = tbl["ref_basename"].to_pylist()

    # Already corrected? `human_score` here spans [0,1]; the inverted form's min is
    # 0.0175 and its max is exactly 1.0, the corrected form's min is exactly 0.0.
    # Do NOT infer from the range -- ask the orientation gate's own ground truth by
    # checking whether the preserved original exists and differs.
    if keep.exists() and not force:
        prev = pq.read_table(keep, columns=["human_score"])["human_score"].to_pylist()
        if np.allclose(np.asarray(prev, dtype=float), 1.0 - hs, atol=0, rtol=0):
            rec.update(status="ALREADY-CORRECTED", preserved=str(keep),
                       preserved_sha256=sha256_file(keep), corrected_sha256=sha256_file(src))
            return rec

    fixed = 1.0 - hs
    rec["target_range_before"] = [float(hs.min()), float(hs.max())]
    rec["target_range_after"] = [float(fixed.min()), float(fixed.max())]
    rec["cross_check_vs_canonical"] = cross_check(fixed, refs)

    if dry_run:
        rec.update(status="DRY-RUN")
        return rec

    # 1. Preserve the original bytes verbatim, and prove the copy is byte-identical.
    orig_sha = sha256_file(src)
    if not keep.exists():
        shutil.copy2(src, keep)
    keep_sha = sha256_file(keep)
    if keep_sha != orig_sha:
        rec.update(status="FAIL", reason=f"preserved copy sha {keep_sha} != original {orig_sha}")
        return rec
    rec["preserved"] = str(keep)
    rec["inverted_sha256"] = orig_sha

    # 2. Write the corrected table: every column carried through, human_score replaced.
    idx = tbl.schema.get_field_index("human_score")
    out = tbl.set_column(idx, pa.field("human_score", pa.float64()), pa.array(fixed, pa.float64()))
    tmp = src.with_suffix(".parquet.tmp")
    pq.write_table(out, tmp, compression="zstd", compression_level=15)
    tmp.replace(src)
    rec["corrected_sha256"] = sha256_file(src)

    # 3. Post-write verification -- read it back and re-derive.
    back = pq.read_table(src)
    b_hs = np.asarray(back["human_score"].to_pylist(), dtype=float)
    orig = pq.read_table(keep)
    ok_target = bool(np.array_equal(b_hs, fixed))
    ok_refs = back["ref_basename"].to_pylist() == refs
    ok_schema = back.schema.names == tbl.schema.names
    # Every non-target column must be unchanged, value for value.
    changed = [n for n in tbl.schema.names
               if n != "human_score" and not back[n].equals(orig[n])]
    rec.update(readback_target_exact=ok_target, readback_refs_identical=ok_refs,
               readback_schema_identical=ok_schema, other_columns_changed=changed)
    rec["status"] = "OK" if (ok_target and ok_refs and ok_schema and not changed) else "FAIL"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", default=None,
                    help="ext root dir (repeatable); default = the three ext lineages")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-apply even if a preserved original is already present")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    roots = [Path(r) for r in (a.root or DEFAULT_ROOTS)]
    recs = [fix_root(r, a.dry_run, a.force) for r in roots]
    report = {
        "tool": "fix_ext_kadid_orientation.py",
        "registration": "benchmarks/sota944_campaign_2026-08-03.md REGISTERED APPENDIX H, H.1",
        "transform": "human_score := 1 - human_score  (== (dmos-1)/4 given the stored (5-dmos)/4)",
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "roots": recs,
    }
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        for r in recs:
            print(f"{r['status']:18s} {r['table']}")
            for k in ("rows", "target_range_before", "target_range_after",
                      "inverted_sha256", "corrected_sha256", "other_columns_changed",
                      "cross_check_vs_canonical", "reason"):
                if k in r:
                    print(f"    {k}: {r[k]}")
    bad = [r for r in recs if r["status"] in ("FAIL", "MISSING")]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
