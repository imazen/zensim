#!/usr/bin/env python3
"""Merge the IW-SSIM target parquet sidecar into the safesyn source TSV.

The IW-SSIM corpus compute (`compute_iwssim_on_safesyn.py`) produced
a parquet at
`/mnt/v/output/zensim/synthetic-v2/iwssim_targets_safesyn_<DATE>.parquet`
with columns `source_path, decoded_path, iwssim` — one row per
(ref, dist) pair from the 196086-row safesyn source TSV.

This script left-joins that parquet onto the safesyn source TSV
(`training_safe_synthetic.csv`) by `(source_path, decoded_path)` and
writes an enriched CSV with the original columns plus a new
`iwssim` column. Output is the canonical input for the V_22-IW
training pipeline: every (ref, dist) pair has both legacy ssim2
targets AND an IW-SSIM target available, so the trainer's
`--target-column` flag (T1.1, queued) can switch freely.

## Why ssim2-target training is structurally biased

Per `CLAUDE.md "SROCC-only verdicts BANNED + ssim2-target training
bias"`, every V_X bake trained against `cpu_ssimulacra2` produces an
ssim2-shaped output surface. Evaluating that bake via SROCC against
ssim2-derived MOS-equivalents favors ssim2-shaped surfaces by
construction. To break the loop, the training corpus needs
non-ssim2 target columns. IW-SSIM is the first such column (this
script); CVVDP is the second (zenmetrics CVVDP PINNED TASK
in flight at 77 % per `~/work/zen/zenmetrics/CLAUDE.md`).

## Usage

```sh
# Default: read today's IW-SSIM parquet + write enriched CSV under
# /mnt/v/zen/zensim-training/<DATE>/
python3 scripts/v_next/merge_iwssim_into_safesyn.py

# Custom paths
python3 scripts/v_next/merge_iwssim_into_safesyn.py \\
    --source-tsv /mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv \\
    --iwssim-parquet /mnt/v/output/zensim/synthetic-v2/iwssim_targets_safesyn_2026-05-16.parquet \\
    --out /mnt/v/zen/zensim-training/2026-05-16/safesyn_with_iwssim.csv
```

## Output schema

All columns from the source TSV (`source_path, decoded_path, codec,
quality, width, height, gpu_ssimulacra2, gpu_butteraugli,
cpu_ssimulacra2, cpu_butteraugli, size_bytes, run_id, dssim`) plus:

  - `iwssim` — Wang & Li 2011 IW-SSIM per pair, in `[0, 1]`. NaN
    for pairs that the IW-SSIM compute skipped (e.g., dimension
    mismatch).

Row count = source TSV row count (left-join). Pairs missing from the
parquet get NaN in the `iwssim` column.

## Next steps (queued tasks)

After this script lands an enriched CSV, the V_22-IW critical path
continues:

- **T1.1 (#50)**: Trainer `--target-column NAME` flag — switches
  the regression target from `cpu_ssimulacra2` → `iwssim`.
- **T1.3 (#52)**: Train V_22-IW seed=1 against IW-SSIM target.

Note: the enriched CSV at the canonical path above is the SOURCE
TSV (paths + metric columns, no features). For the trainer to use
it, features must be extracted from the (source_path, decoded_path)
PNG pairs. The existing 138872-row features CSV at
`/mnt/v/zen/zensim-training/2026-05-14-clean/safe_synth_v19_clean_features.csv`
is post-purge subset of an older 196086-row source — re-extracting
features against this script's 2026-05-16 enriched TSV will produce
a 196086-row features CSV with both `human_score` (legacy ssim2)
and `iwssim` columns available.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


def main() -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source-tsv",
        type=Path,
        default=Path("/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv"),
    )
    ap.add_argument(
        "--iwssim-parquet",
        type=Path,
        default=Path(
            f"/mnt/v/output/zensim/synthetic-v2/iwssim_targets_safesyn_{today}.parquet"
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(f"/mnt/v/zen/zensim-training/{today}/safesyn_with_iwssim.csv"),
    )
    args = ap.parse_args()

    if not args.source_tsv.is_file():
        print(f"ERROR: source TSV missing at {args.source_tsv}", file=sys.stderr)
        return 2
    if not args.iwssim_parquet.is_file():
        print(
            f"ERROR: IW-SSIM parquet missing at {args.iwssim_parquet}",
            file=sys.stderr,
        )
        return 2

    print(f"source TSV: {args.source_tsv}")
    print(f"IW-SSIM parquet: {args.iwssim_parquet}")
    print(f"output: {args.out}")
    print()

    # Build the lookup: (source_path, decoded_path) → iwssim
    table = pq.read_table(args.iwssim_parquet)
    src_col = table.column("source_path").to_pylist()
    dst_col = table.column("decoded_path").to_pylist()
    iw_col = table.column("iwssim").to_pylist()
    lookup: dict[tuple[str, str], float] = dict(
        zip(zip(src_col, dst_col), iw_col, strict=True)
    )
    print(f"loaded {len(lookup):,} IW-SSIM pairs from parquet")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Stream through the source TSV; emit enriched rows.
    n_rows = 0
    n_matched = 0
    n_missing = 0
    with args.source_tsv.open() as fin, args.out.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        if (
            "source_path" not in (reader.fieldnames or [])
            or "decoded_path" not in (reader.fieldnames or [])
        ):
            print(
                f"ERROR: source TSV missing source_path / decoded_path columns; got {reader.fieldnames}",
                file=sys.stderr,
            )
            return 2
        out_fields = list(reader.fieldnames) + ["iwssim"]
        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()
        for row in reader:
            n_rows += 1
            key = (row["source_path"], row["decoded_path"])
            iw = lookup.get(key)
            if iw is None:
                n_missing += 1
                row["iwssim"] = ""
            else:
                n_matched += 1
                row["iwssim"] = f"{iw:.6f}"
            writer.writerow(row)

    print()
    print(f"wrote {args.out}")
    print(f"  source rows: {n_rows:,}")
    print(f"  matched   :  {n_matched:,} ({100 * n_matched / max(n_rows, 1):.2f} %)")
    print(f"  missing   :  {n_missing:,} (empty iwssim column)")
    if n_missing > 0 and n_missing < 100:
        print("  → small missing set; spot-check the input/output paths match")
    elif n_missing >= 100:
        print(
            "  → large missing set; verify the parquet was generated against"
            " the same source TSV"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
