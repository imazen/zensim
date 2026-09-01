#!/usr/bin/env python3
"""wave-r4 A6 gate: do the 196k big leg's JPEG rows reproduce the 111k leg?

`ext_safesyn_full.parquet` (111,068 rows) is exactly the JPEG-family subset of
safesyn, extracted by the same binary, same mode, same build. The 196k leg
contains those same 111,068 pairs pointing at the same `.jpg` bitstreams, plus
85,018 avif/jxl/webp rows. So the JPEG subset MUST reproduce -- if it does not,
the difference is caused by something other than the added rows (reference
grouping, ordering, a stale binary) and must be reported, not averaged away.

The join is an EXACT KEY join, not positional:
  * the 111k parquet is row-aligned with `safesyn_jpeg_FULL_pairs_ab.tsv`
    (MEASURED 111068/111068 on ref_basename AND human_score exact), whose
    `dist_path` is the `.jpg` bitstream path and is unique;
  * the 196k CSV is row-aligned with `pairs_safesyn_big.tsv` (the extractor
    preserves input TSV order), whose JPEG rows carry that same `.jpg` path.
So `dist_path` keys both sides. Coverage is asserted at 111068/111068.

NOTE the human_score caveat, measured and reported by the stager: our labels
come from `canonical-2026-05-21/train/safesyn.parquet` while the 111k leg's come
from a different vintage (only 29,290/111,068 bit-exact). So human_score is
deliberately NOT part of the key and NOT compared -- only f0..f943 are, and
features do not depend on the label.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path("/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01")
BIG_CSV = Path(
    os.environ.get(
        "BIGLEG_CSV", "/mnt/v/output/zensim/waver4-run-2026-09-01/ext_safesyn_big.csv"
    )
)
BIG_PAIRS = Path(os.environ.get("BIGLEG_PAIRS", ROOT / "pairs" / "pairs_safesyn_big.tsv"))
REF_PARQUET = ROOT / "ext_safesyn_full.parquet"
REF_TSV = Path("/mnt/v/output/zensim/v2-ab-2026-07-19/safesyn_jpeg_FULL_pairs_ab.tsv")
OUT = Path(os.environ.get("BIGLEG_GATE_OUT", "/mnt/v/output/zensim/waver4-run-2026-09-01"))


def die(m: str) -> None:
    print(f"ABORT: {m}", file=sys.stderr)
    sys.exit(2)


def main() -> None:
    for p in (BIG_CSV, BIG_PAIRS, REF_PARQUET, REF_TSV):
        if not p.exists():
            die(f"missing {p}")

    # ---- reference side --------------------------------------------------
    ref_dists: list[str] = []
    ref_refs: list[str] = []
    with open(REF_TSV, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ref_dists.append(row["dist_path"])
            ref_refs.append(row["ref_path"])
    tbl = pq.read_table(REF_PARQUET)
    if tbl.num_rows != len(ref_dists):
        die(f"ref parquet {tbl.num_rows} != ref tsv {len(ref_dists)}")
    rb = tbl.column("ref_basename").to_pylist()
    ok = sum(
        1
        for i in range(len(ref_refs))
        if os.path.splitext(os.path.basename(ref_refs[i]))[0] == rb[i]
    )
    if ok != len(ref_refs):
        die(f"reference row-alignment gate {ok}/{len(ref_refs)}")
    ref_idx = {d: i for i, d in enumerate(ref_dists)}

    # ---- big side --------------------------------------------------------
    big_dists: list[str] = []
    with open(BIG_PAIRS, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            big_dists.append(row["dist_path"])

    # Which big-leg rows are the JPEG subset, and where each lands in the
    # reference table. Established BEFORE reading the CSV so only those rows are
    # ever materialised (196,086 x 946 as Python strings would be ~185M objects).
    pairs = [(k, ref_idx[d]) for k, d in enumerate(big_dists) if d in ref_idx]
    coverage = f"{len(pairs)}/{tbl.num_rows}"
    if len(pairs) != tbl.num_rows:
        die(f"JPEG-subset coverage {coverage} -- the 196k leg does not contain the 111k leg")
    slot = {k: n for n, (k, _) in enumerate(pairs)}
    ref_rows_idx = np.array([i for _, i in pairs], dtype=np.int64)
    codecs = [big_dists[k].split("/")[-2] for k, _ in pairs]

    with open(BIG_CSV, newline="") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        fcols = [c for c in hdr if c.startswith("f")]
        if len(fcols) != 944:
            die(f"big csv has {len(fcols)} feature cols, expected 944")
        fpos = [hdr.index(c) for c in fcols]
        A = np.empty((len(pairs), len(fcols)), dtype=np.float64)
        seen = 0
        for k, row in enumerate(rd):
            seen += 1
            n = slot.get(k)
            if n is not None:
                A[n] = [float(row[j]) for j in fpos]
    if seen != len(big_dists):
        die(f"big csv {seen} rows != big pairs {len(big_dists)}")

    max_abs = 0.0
    max_rel = 0.0
    max_abs_at = ""
    n_cells = 0
    n_diff = 0
    rows_diff_mask = np.zeros(len(pairs), dtype=bool)

    # column-blocked so peak memory stays ~O(block * rows), not 944 * rows
    BLOCK = 64
    for start in range(0, len(fcols), BLOCK):
        names = fcols[start : start + BLOCK]
        b = np.column_stack(
            [tbl.column(c).to_numpy(zero_copy_only=False)[ref_rows_idx] for c in names]
        )
        a = A[:, start : start + len(names)]
        d = np.abs(a - b)
        n_cells += d.size
        nz = d != 0.0
        n_diff += int(nz.sum())
        rows_diff_mask |= nz.any(axis=1)
        if nz.any():
            fl = int(np.argmax(d))
            r, c = divmod(fl, d.shape[1])
            if d[r, c] > max_abs:
                max_abs = float(d[r, c])
                max_abs_at = f"{names[c]} codec={codecs[r]}"
            s = np.maximum(np.abs(a), np.abs(b))
            ok = s > 1e-12
            if ok.any():
                max_rel = max(max_rel, float((d[ok] / s[ok]).max()))

    n_rows_diff = int(rows_diff_mask.sum())
    per_codec_diff: dict[str, int] = {}
    for k, bad in zip(range(len(pairs)), rows_diff_mask):
        if bad:
            per_codec_diff[codecs[k]] = per_codec_diff.get(codecs[k], 0) + 1

    res = {
        "gate": "wave-r4 A6 JPEG-subset agreement (196k big leg vs 111k ext_safesyn_full)",
        "join": "EXACT KEY on dist_path (.jpg bitstream); ref side row-aligned to its pairs TSV",
        "coverage": coverage,
        "n_cells_compared": n_cells,
        "n_cells_differing": n_diff,
        "n_rows_differing": n_rows_diff,
        "per_codec_rows_differing": per_codec_diff,
        "max_abs": max_abs,
        "max_abs_at": max_abs_at,
        "max_rel": max_rel,
        "verdict": "BYTE-IDENTICAL" if n_diff == 0 else "DIFFERS",
        "note": "human_score is deliberately excluded (different label vintage; see stager)",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate_safesyn_big_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    if n_diff != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
