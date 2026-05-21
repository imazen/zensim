#!/usr/bin/env python3
"""Measure bake-output cross-codec stddev at JND pivot.

For each bake in a directory, load the cross-codec equivalence parquet,
score both A-side and B-side feature vectors through the bake, then
compute stddev of bake output across codecs at the same pivot per image.
Lower stddev = the bake assigns more consistent quality to encodes at
the same pivot — i.e. the cross-codec equivalence mechanism is working.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def score_bake(bake_path: Path, features: np.ndarray) -> np.ndarray:
    """Score features through a bake using zenpredict CLI.

    Writes features to a TSV, runs zenpredict with bake, reads back scores.
    Slow but doesn't require Python bindings.

    For this script we approximate by loading the bake's MLP weights and
    forward-passing in NumPy.
    """
    raise NotImplementedError("use direct numpy forward")


def load_bake_predict(bake_path: Path) -> tuple[callable, dict]:
    """Run bake_verdict --output-per-pair on a known feature set."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bake-dir",
        type=Path,
        nargs="+",
        required=True,
        help="one or more dirs containing .bin bakes",
    )
    parser.add_argument(
        "--cvvdp-equiv",
        type=Path,
        default=Path("/mnt/v/zen/zensim-training/2026-05-20-v12-cvvdp-substrate/cross_codec_equivalence_cvvdp_372col.parquet"),
    )
    parser.add_argument(
        "--ssim2-equiv",
        type=Path,
        default=Path("/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/cross_codec_equivalence_ssim2_372col_v4.parquet"),
    )
    parser.add_argument(
        "--n-features", type=int, default=372,
    )
    args = parser.parse_args()

    # Use bake_verdict's per-pair output to score features
    # Build a (features_a, features_b) → score_a, score_b pipeline via
    # a small "anchor parquet" trick: scoring each row through the bake
    # gives us the predicted score. We use a held-out validation harness
    # for this and report per-pivot stddev.

    # For each equiv parquet, extract feature_a + feature_b matrices
    print("Loading cvvdp equiv parquet...")
    cv_pairs = pq.read_table(args.cvvdp_equiv).to_pandas()
    print(f"  rows: {len(cv_pairs)}")

    print("Loading ssim2 equiv parquet...")
    ss_pairs = pq.read_table(args.ssim2_equiv).to_pandas()
    print(f"  rows: {len(ss_pairs)}")

    # For each bake, write per-pivot stddev measure via a helper:
    # the bake's score on each row's fa_* and fb_* features.
    # We invoke a Rust binary that scores a feature parquet through a bake.
    # If not available, we approximate by loading bake weights directly.

    # Simpler: build a temp "anchor parquet" with feature_cols from fa_*,
    # then run bake_verdict —but bake_verdict only does corpus eval, not
    # arbitrary features. Use zenpredict CLI's score path.

    # The cleanest available is to load the bake as a ZNPR v3 and call
    # Predictor::predict via a small Rust binary. Not built yet.
    #
    # For now, use a workaround: run bake_verdict on a synthetic corpus
    # via a Python predict path. Build a NumPy MLP forward from the bake.

    # Use a separate helper that calls zenpredict-bake.
    # Defer to a separate analysis using bake_verdict's per-pair output.

    print()
    print("WARN: bake-output cross-codec stddev needs a Rust score helper.")
    print("Reporting substrate stddev only (see measure_cc_std_jnd.py).")


if __name__ == "__main__":
    main()
