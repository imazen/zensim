#!/usr/bin/env python3
"""Emit the REFERENCE-TRUTH sidecar the 2026-09-05 two-reference inversion rule
reads (`bake_verdict --reference-truth <tsv>:<variant>`).

USER RULING 2026-09-05, verbatim: *"for inversions, we should choose say ssim2
and butter and only flag true inversions where they agree, and we can then file
or update tracking issues on codecs for when they are nonmonotonic."*

The rule itself lives in ONE place — `zensim_validate::dial_addressability::
encoder_inversion` — and this script computes NOTHING: it only reshapes an
instrument's already-persisted per-cell reference metrics into the flat table
that rule reads. Shape:

    image_id \t codec \t q \t ssim2 \t butteraugli

`butteraugli` is in DISTANCE units (higher = worse) of the variant named on the
command line, because the margin is variant-specific (pnorm3 tracks ssim2's
direction on 94.30 % of the ladder instrument's adjacent pairs; max on 75.27 %).
The variant is NOT inferred from the column name at read time — the caller
states it, and a table generated here for one variant must be passed with that
same variant.

An instrument that does not persist both references cannot produce this table,
and the two-reference reading is then NOT MEASURABLE on it — which
`bake_verdict` reports loudly rather than silently degrading.
"""
import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq

VARIANTS = {"pnorm3": "score_butteraugli_pnorm3", "max": "score_butteraugli_max"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, type=Path,
                    help="the instrument's FULL archive parquet (every step, "
                         "with `saturated` and both butteraugli columns)")
    ap.add_argument("--variant", default="pnorm3", choices=sorted(VARIANTS))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--keep-saturated", action="store_true",
                    help="emit duplicate-setting rows too. Off by default: the "
                         "dial grid holds only DISTINCT settings, so a "
                         "saturated row can never be a ladder endpoint there.")
    a = ap.parse_args()

    bcol = VARIANTS[a.variant]
    need = ["image_id", "codec", "q", "score_ssim2", bcol]
    t = pq.read_table(a.full, columns=need + ["saturated"])
    d = t.to_pydict()
    rows = []
    for i in range(t.num_rows):
        if d["saturated"][i] and not a.keep_saturated:
            continue
        rows.append((d["image_id"][i], d["codec"][i], float(d["q"][i]),
                     float(d["score_ssim2"][i]), float(d[bcol][i])))
    if not rows:
        print(f"{a.full}: no rows survived", file=sys.stderr)
        return 2
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        # A leading `#` line is skipped by the reader, so the variant travels
        # with the bytes as well as on the command line.
        f.write(f"# reference-truth sidecar; butteraugli variant = {a.variant} "
                f"(DISTANCE units); source = {a.full}\n")
        f.write("image_id\tcodec\tq\tssim2\tbutteraugli\n")
        for img, cod, q, s2, b in rows:
            f.write(f"{img}\t{cod}\t{q!r}\t{s2!r}\t{b!r}\n")
    print(f"{a.out}: {len(rows)} cells, butteraugli variant {a.variant}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
