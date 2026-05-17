#!/usr/bin/env python3
"""Convert the V_20 greedy-screen TSV into `--feature-transform` CLI args.

Reads the output of `v0_20_feature_transform_greedy_screen.py`, picks
features with lift >= --min-lift (and skips Identity winners), and
emits one `--feature-transform TOKEN:IDX[:PARAMS]` flag per winner.

## Usage

  python3 scripts/v_next/v0_20_screen_to_trainer_args.py \\
    --screen-tsv benchmarks/v0_20_feature_transform_greedy_screen_2026-05-15.tsv \\
    --min-lift 0.05 \\
    > /tmp/v0_20_top_transforms.txt

  # Then feed into trainer (Bash):
  xargs -a /tmp/v0_20_top_transforms.txt zensim_mlp_train \\
    --group safesyn:/mnt/v/.../safe_synth_v19_clean_features.csv:1.0:0.0 \\
    [...] \\
    --out benchmarks/v0_20_input_shaping_$(date -u +%Y-%m-%d).bin

The TSV from the screen has columns:
  feat_idx, best_transform, params_csv, baseline_pearson,
  transformed_pearson, lift, baseline_spearman, n_samples

Optional `--max-features N` cap helps for first-pass experiments.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-tsv", required=True, type=Path)
    ap.add_argument(
        "--min-lift",
        type=float,
        default=0.05,
        help="Only emit features with lift >= this. Default 0.05 (98 features).",
    )
    ap.add_argument(
        "--max-features",
        type=int,
        default=0,
        help="Cap at top-N by lift (0 = no cap, emit all eligible).",
    )
    ap.add_argument(
        "--max-feature-idx",
        type=int,
        default=228,
        help="Skip features with idx >= this. Default 228 = the runtime input "
        "width (V_18 ship uses 228 features). Pass 300 for research bakes "
        "that include the extended-feature columns.",
    )
    args = ap.parse_args()

    rows: list[dict] = []
    with args.screen_tsv.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            try:
                idx = int(row["feat_idx"])
                lift = float(row["lift"])
                token = row["best_transform"]
            except (KeyError, ValueError):
                continue
            if idx >= args.max_feature_idx:
                continue
            if token == "identity":
                continue
            if lift < args.min_lift:
                continue
            rows.append({"idx": idx, "lift": lift, "token": token, "params": row["params_csv"]})

    rows.sort(key=lambda r: -r["lift"])
    if args.max_features > 0:
        rows = rows[: args.max_features]

    for row in rows:
        if row["params"]:
            print(f"--feature-transform {row['token']}:{row['idx']}:{row['params']}")
        else:
            print(f"--feature-transform {row['token']}:{row['idx']}")

    print(
        f"# emitted {len(rows)} flags (min-lift={args.min_lift}, max-feature-idx={args.max_feature_idx})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
