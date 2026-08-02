#!/usr/bin/env python3
"""Backfill 720 features for the CORRUPTION GRID eval instrument (2026-07-22).

Context: docs/V2_EXPERIMENT_PLAN_2026-07-20.md ("eval: corruption grid" row).
Unlike the dial grid, the corruption grid's PIXELS were never deleted — all
2,016 cells (672 base entries x 3 kind variants {corruption,q10,q20}) plus the
single shared reference live at
`/mnt/v/output/zensim/corruption_gate/*.png` (per `_MANIFEST.json` +
`gb82_dog__reference.png`). So this is a PURE RE-EXTRACTION: no re-encode
needed. The original 372-feature grid was CPU-extracted via
`extract_features_372col --corpus pairs`
(see `benchmarks/eval_grids_2026-05-29.pointer.md`), so this backfill should
match to near-ULP (both CPU, both on the exact same pixels) — unlike the
dial grid, which crosses GPU (original) vs CPU (backfill) backends.

Two subcommands, mirroring the dial-grid sibling script:

  build-pairs   Build the pairs.tsv (ref_path = the single shared reference,
                dist_path = each entry's PNG, human_score = row index) that
                v2_ab_extract needs.

  finalize      Join v2_ab_extract's output back onto `entry` (via the row
                index carried in `human_score`), verify f0..f371 against the
                stored corruption_grid parquet, and write the 720-wide
                parquet.

Usage:
  python3 backfill_corruption_grid_720.py build-pairs \
      --corruption-dir /mnt/v/output/zensim/corruption_gate \
      --original /path/to/corruption_grid_372col_2026-05-28.parquet \
      --out-pairs corruption_pairs.tsv

  # then, separately (CPU step, via run-heavy):
  #   v2_ab_extract corruption_pairs.tsv ext_raw.csv

  python3 backfill_corruption_grid_720.py finalize \
      --ext-csv ext_raw.csv \
      --original /path/to/corruption_grid_372col_2026-05-28.parquet \
      --out ext_corruption_grid.parquet
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def cmd_build_pairs(args):
    orig = pq.read_table(args.original, columns=["entry"]).to_pandas()
    entries = orig["entry"].tolist()
    ref = os.path.join(args.corruption_dir, "gb82_dog__reference.png")
    if not os.path.exists(ref):
        raise SystemExit(f"reference missing: {ref}")

    lines = ["ref_path\tdist_path\thuman_score"]
    missing = []
    for i, e in enumerate(entries):
        dist = os.path.join(args.corruption_dir, f"{e}.png")
        if not os.path.exists(dist):
            missing.append(e)
            continue
        lines.append(f"{ref}\t{dist}\t{i}")
    print(f"{len(entries)} entries; {len(missing)} missing dist files", file=sys.stderr)
    if missing:
        print("MISSING (first 10):", missing[:10], file=sys.stderr)
    with open(args.out_pairs, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines) - 1} pairs -> {args.out_pairs}", file=sys.stderr)


def cmd_finalize(args):
    ext = pd.read_csv(args.ext_csv)
    n_feat = len([c for c in ext.columns if c.startswith("f")])
    assert n_feat == 720, f"expected 720 f-cols, got {n_feat}"
    ext["idx"] = ext["human_score"].round().astype(int)

    orig = pq.read_table(args.original).to_pandas()
    orig = orig.reset_index().rename(columns={"index": "idx"})
    print(f"original grid: {len(orig)} rows", file=sys.stderr)

    joined = orig.merge(ext, on="idx", how="left", suffixes=("_orig", "_new"), indicator=True)  # joinsafety-ok: idx = reset_index() row id vs the harness row-index round-tripped through human_score (same run); indicator + matched/missing accounting + 372-feature near-ULP verify AFTER the join — historical as-run script, predates the gate
    matched = joined[joined["_merge"] == "both"].copy()
    missing = joined[joined["_merge"] == "left_only"].copy()
    print(
        f"JOIN RESULT: {len(matched)}/{len(orig)} matched "
        f"({100 * len(matched) / len(orig):.1f}%); {len(missing)} MISSING (dropped)",
        file=sys.stderr,
    )
    if len(missing):
        print("missing entries (first 10):", missing["entry"].head(10).tolist(), file=sys.stderr)

    f_orig_cols = [f"f{i}_orig" for i in range(372)]
    f_new_cols = [f"f{i}_new" for i in range(372)]
    orig_mat = matched[f_orig_cols].to_numpy(dtype=np.float64)
    new_mat = matched[f_new_cols].to_numpy(dtype=np.float64)
    absdiff = np.abs(orig_mat - new_mat)
    l2 = np.sqrt((absdiff**2).sum(axis=1))
    max_abs = absdiff.max(axis=1)
    matched["_verify_l2"] = l2
    matched["_verify_max_abs"] = max_abs

    print(
        f"\n372-feature verify (both CPU, same pixels — expect near-ULP):\n"
        f"  L2   median={np.median(l2):.4g}  p90={np.quantile(l2, 0.9):.4g}  "
        f"p99={np.quantile(l2, 0.99):.4g}  max={l2.max():.4g}\n"
        f"  max_abs median={np.median(max_abs):.4g}  max={max_abs.max():.4g}",
        file=sys.stderr,
    )
    flagged = matched[matched["_verify_l2"] > args.flag_l2]
    print(
        f"{len(flagged)}/{len(matched)} rows flagged (L2 > {args.flag_l2}) as material drift",
        file=sys.stderr,
    )
    if len(flagged):
        print(flagged["entry"].head(20).tolist(), file=sys.stderr)
    matched["drift_flag"] = (matched["_verify_l2"] > args.flag_l2).astype(int)

    out_cols = {
        "entry": matched["entry"].to_numpy(),
        "drift_flag": matched["drift_flag"].to_numpy(),
        "verify_l2_372": matched["_verify_l2"].to_numpy(),
    }
    for i in range(372):
        out_cols[f"f{i}"] = matched[f"f{i}_orig"].to_numpy(dtype=np.float32)
    for i in range(348):
        # f372..f719 exist ONLY in the extraction output, not in `orig` (372-wide),
        # so pandas merge leaves them unsuffixed (see the dial-grid sibling script).
        out_cols[f"f{372 + i}"] = matched[f"f{372 + i}"].to_numpy(dtype=np.float32)

    pq.write_table(pa.table(out_cols), args.out, compression="zstd")
    print(f"wrote {len(matched)} rows x 720 features -> {args.out}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    bp = sub.add_parser("build-pairs")
    bp.add_argument("--corruption-dir", required=True)
    bp.add_argument("--original", required=True)
    bp.add_argument("--out-pairs", required=True)
    bp.set_defaults(func=cmd_build_pairs)

    fz = sub.add_parser("finalize")
    fz.add_argument("--ext-csv", required=True)
    fz.add_argument("--original", required=True)
    fz.add_argument("--out", required=True)
    fz.add_argument("--flag-l2", type=float, default=0.5)
    fz.set_defaults(func=cmd_finalize)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
