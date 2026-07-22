#!/usr/bin/env python3
"""Backfill 720 features for the DIAL GRID eval instrument (2026-07-22).

Context: docs/V2_EXPERIMENT_PLAN_2026-07-20.md ("eval: dial grid" row) — the
stored `dial_grid_372col_2026-05-29.parquet` (4,817 cells) carries feature
VECTORS ONLY (image_id, codec, q, codec_param, param_kind, f0..f371); the
pixels used to build it were never persisted, and the original build used
`zenmetrics sweep --metric zensim-gpu` (GPU zensim extraction). As of
2026-07-19 GPU zensim SCORING is fully disabled (panics) — see
zenmetrics-cli/src/metrics/mod.rs `MetricKind::ZensimGpu => panic!(...)` —
so a straight rerun of the original recipe is no longer possible, and CPU
`sweep --feature-output` hardcodes 300 features regardless of
`--zensim-features-regime` (`sweep/run.rs` `want_features_cpu` path calls
`run_zensim_with_features` -> `zensim::score_with_features`, NOT the
regime-aware `run_zensim_features`). So this backfill re-encodes the exact
documented grid (`scripts/v_next/build_qsweep_expanded.py`'s QGRID /
JXL_DISTANCES / jxl_q_equiv, reproduced verbatim below) via `zenmetrics sweep`
for ENCODE+DECODE ONLY (no zensim scoring/features requested from sweep at
all — just `--distorted-out-dir` + `--pairs-tsv`), then extracts the full
720-wide vector (v1-372 ++ v2-348) via zensim's own CPU
`examples/v2_ab_extract` on the resulting (ref, dist) pixel pairs. Finally it
JOINS back onto the ORIGINAL 4,817-row identity (image_id, codec, q,
codec_param) to verify f0..f371 parity and append f372..f719.

Two subcommands:

  build-pairs   Merge the per-codec `zenmetrics sweep --pairs-tsv` outputs
                into ONE pairs.tsv for v2_ab_extract, with a side parquet
                mapping row-index -> (image_id, codec, q_raw, knob_tuple_json)
                so identity survives v2_ab_extract's ref-basename-keyed CSV
                (which collapses per-cell identity when many cells share one
                reference — see the corruption-grid sibling script for the
                same reason). The pairs.tsv's `human_score` column is
                REPURPOSED to carry the row index (a plain float), NOT an
                actual human score — v2_ab_extract only threads it through
                verbatim, so this needs zero changes to that tool.

  finalize      Read v2_ab_extract's output CSV + the side mapping + the
                ORIGINAL dial_grid parquet; compute (q, codec_param,
                param_kind) per row exactly as the original build script did;
                join to the original by (image_id, codec, rounded q/param);
                verify f0..f371 drift; write the final 720-wide parquet.
                Never fabricates: any original row with no re-encoded match
                is reported and dropped, not synthesized.

Usage:
  python3 backfill_dial_grid_720.py build-pairs \
      --codec-dir jpeg:/path/to/jpeg --codec-dir webp:/path/to/webp \
      --codec-dir avif:/path/to/avif --codec-dir jxl:/path/to/jxl \
      --out-pairs merged_pairs.tsv --out-map merged_map.parquet

  # then, separately (heavy CPU step, run via run-heavy):
  #   v2_ab_extract merged_pairs.tsv ext_raw.csv

  python3 backfill_dial_grid_720.py finalize \
      --ext-csv ext_raw.csv --map merged_map.parquet \
      --original /path/to/dial_grid_372col_2026-05-29.parquet \
      --out ext_dial_grid.parquet
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---- verbatim from scripts/v_next/build_qsweep_expanded.py (2026-05-29) ----
JXL_Q_K = 4.0


def jxl_q_equiv(d: float) -> float:
    return max(0.0, min(100.0, 100.0 - JXL_Q_K * float(d)))


# codec CLI name -> short name used in the stored grid's `codec` column
CODEC_SHORT = {"zenjpeg": "jpeg", "zenwebp": "webp", "zenavif": "avif", "zenjxl": "jxl"}


def image_id_of(ref_path: str) -> str:
    return os.path.splitext(os.path.basename(ref_path))[0]


def cmd_build_pairs(args):
    codec_dirs = {}
    for spec in args.codec_dir:
        name, path = spec.split(":", 1)
        codec_dirs[name] = path

    rows = []  # (image_id, codec_short, q_raw, knob_tuple_json, ref_path, dist_path)
    for name, d in codec_dirs.items():
        pairs_tsv = os.path.join(d, "pairs.tsv")
        if not os.path.exists(pairs_tsv):
            print(f"WARN: {pairs_tsv} missing, skipping {name}", file=sys.stderr)
            continue
        with open(pairs_tsv, newline="") as f:
            r = csv.DictReader(f, delimiter="\t")
            n = 0
            for rec in r:
                rows.append(
                    (
                        image_id_of(rec["ref_path"]),
                        CODEC_SHORT.get(rec["codec"], rec["codec"]),
                        float(rec["q"]),
                        rec["knob_tuple_json"],
                        rec["ref_path"],
                        rec["dist_path"],
                    )
                )
                n += 1
        print(f"{name}: {n} pairs from {pairs_tsv}", file=sys.stderr)

    print(f"total merged pairs: {len(rows)}", file=sys.stderr)

    with open(args.out_pairs, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ref_path", "dist_path", "human_score"])
        for i, r in enumerate(rows):
            w.writerow([r[4], r[5], i])

    map_tbl = pa.table(
        {
            "idx": pa.array(range(len(rows)), type=pa.int64()),
            "image_id": pa.array([r[0] for r in rows]),
            "codec": pa.array([r[1] for r in rows]),
            "q_raw": pa.array([r[2] for r in rows], type=pa.float64()),
            "knob_tuple_json": pa.array([r[3] for r in rows]),
        }
    )
    pq.write_table(map_tbl, args.out_map, compression="zstd")
    print(f"wrote {args.out_pairs} ({len(rows)} rows) + {args.out_map}", file=sys.stderr)


def cmd_finalize(args):
    ext = pd.read_csv(args.ext_csv)
    n_feat = len([c for c in ext.columns if c.startswith("f")])
    assert n_feat == 720, f"expected 720 f-cols in {args.ext_csv}, got {n_feat}"
    ext["idx"] = ext["human_score"].round().astype(int)

    mp = pq.read_table(args.map).to_pandas()
    merged = mp.merge(ext, on="idx", how="left", suffixes=("", "_ext"))
    n_unmatched_extract = merged["f0"].isna().sum()
    if n_unmatched_extract:
        print(
            f"WARN: {n_unmatched_extract} pairs had NO v2_ab_extract output row "
            "(decode/compute error inside v2_ab_extract) — dropped",
            file=sys.stderr,
        )
    merged = merged.dropna(subset=["f0"]).copy()

    # Derive (q, codec_param, param_kind) exactly as build_qsweep_expanded.py did.
    def derive(row):
        if row["codec"] == "jxl":
            try:
                d = json.loads(row["knob_tuple_json"])["distance"]
            except Exception:
                return pd.Series({"q": np.nan, "codec_param": np.nan, "param_kind": "distance"})
            return pd.Series(
                {"q": jxl_q_equiv(d), "codec_param": float(d), "param_kind": "distance"}
            )
        else:
            return pd.Series(
                {"q": row["q_raw"], "codec_param": row["q_raw"], "param_kind": "q"}
            )

    derived = merged.apply(derive, axis=1)
    merged["q"] = derived["q"]
    merged["codec_param"] = derived["codec_param"]
    merged["param_kind"] = derived["param_kind"]
    merged = merged.dropna(subset=["q", "codec_param"])

    orig = pq.read_table(args.original).to_pandas()
    print(f"original grid: {len(orig)} rows", file=sys.stderr)
    print(f"re-encoded (post v2_ab_extract, pre-join): {len(merged)} rows", file=sys.stderr)

    # Join key: (image_id, codec, rounded q, rounded codec_param). Both sides
    # are float64 derived from the SAME literal grid constants, so rounding to
    # 6 dp only guards against JSON/float-formatting round-trip noise, not a
    # real tolerance band.
    ROUND = 6
    orig["_k_q"] = orig["q"].round(ROUND)
    orig["_k_p"] = orig["codec_param"].round(ROUND)
    merged["_k_q"] = merged["q"].round(ROUND)
    merged["_k_p"] = merged["codec_param"].round(ROUND)

    key_cols = ["image_id", "codec", "_k_q", "_k_p"]
    joined = orig.merge(
        merged, on=key_cols, how="left", suffixes=("_orig", "_new"), indicator=True
    )

    matched = joined[joined["_merge"] == "both"].copy()
    missing = joined[joined["_merge"] == "left_only"].copy()
    print(
        f"\nJOIN RESULT: {len(matched)}/{len(orig)} original rows matched "
        f"({100 * len(matched) / len(orig):.1f}%); {len(missing)} MISSING (dropped, not fabricated)",
        file=sys.stderr,
    )
    if len(missing):
        print("per-codec missing counts:", file=sys.stderr)
        print(missing["codec"].value_counts().to_string(), file=sys.stderr)

    # Verify f0..f371 parity (cross-backend: original = zensim-gpu 372
    # extraction 2026-05-29; backfill = CPU-only, since zensim-gpu SCORING is
    # now fully disabled). Report per-row L2/max-abs diff; do NOT silently
    # treat large-diff rows as ok — flag them.
    f_orig_cols = [f"f{i}_orig" for i in range(372)]
    f_new_cols = [f"f{i}_new" for i in range(372)]
    orig_mat = matched[f_orig_cols].to_numpy(dtype=np.float64)
    new_mat = matched[f_new_cols].to_numpy(dtype=np.float64)
    absdiff = np.abs(orig_mat - new_mat)
    max_abs = absdiff.max(axis=1)
    l2 = np.sqrt((absdiff**2).sum(axis=1))
    matched["_verify_max_abs"] = max_abs
    matched["_verify_l2"] = l2

    print("\nPer-codec 372-feature drift (original GPU extraction vs backfill CPU re-extraction):", file=sys.stderr)
    for c in sorted(matched["codec"].unique()):
        sub = matched[matched["codec"] == c]
        print(
            f"  {c:5s} n={len(sub):5d}  L2 median={sub['_verify_l2'].median():.4g} "
            f"p90={sub['_verify_l2'].quantile(0.9):.4g} p99={sub['_verify_l2'].quantile(0.99):.4g} "
            f"max={sub['_verify_l2'].max():.4g}  max_abs median={sub['_verify_max_abs'].median():.4g} "
            f"max={sub['_verify_max_abs'].max():.4g}",
            file=sys.stderr,
        )

    flag_thresh = args.flag_l2
    flagged = matched[matched["_verify_l2"] > flag_thresh]
    print(
        f"\n{len(flagged)}/{len(matched)} rows flagged (L2 > {flag_thresh}) as material drift "
        "(encoder-version drift / known contamination, not backfill error) — "
        "kept in output with drift=1, NOT dropped unless --drop-flagged",
        file=sys.stderr,
    )
    if len(flagged):
        print(flagged.groupby("codec").size().to_string(), file=sys.stderr)

    matched["drift_flag"] = (matched["_verify_l2"] > flag_thresh).astype(int)

    if args.drop_flagged:
        before = len(matched)
        matched = matched[matched["drift_flag"] == 0]
        print(f"--drop-flagged: dropped {before - len(matched)} flagged rows", file=sys.stderr)

    # Assemble output: identity + f0..f371 (STORED, i.e. the original/GPU
    # values — unchanged, so downstream consumers see no shift vs the
    # existing 372 panel) ++ f372..f719 (NEW, from the backfill).
    out_cols = {
        "image_id": matched["image_id"].to_numpy(),
        "codec": matched["codec"].to_numpy(),
        "q": matched["q_orig"].to_numpy(dtype=np.float64),
        "codec_param": matched["codec_param_orig"].to_numpy(dtype=np.float64),
        "param_kind": matched["param_kind_orig"].to_numpy(),
        "drift_flag": matched["drift_flag"].to_numpy(),
        "verify_l2_372": matched["_verify_l2"].to_numpy(),
    }
    for i in range(372):
        out_cols[f"f{i}"] = matched[f"f{i}_orig"].to_numpy(dtype=np.float32)
    for i in range(348):
        # f372..f719 exist ONLY in the `merged` (re-encoded) frame, not in
        # `orig` (which is 372-wide) — pandas merge only appends _orig/_new
        # suffixes to columns present on BOTH sides, so these stay unsuffixed.
        out_cols[f"f{372 + i}"] = matched[f"f{372 + i}"].to_numpy(dtype=np.float32)

    pq.write_table(pa.table(out_cols), args.out, compression="zstd")
    print(f"\nwrote {len(matched)} rows x 720 features -> {args.out}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    bp = sub.add_parser("build-pairs")
    bp.add_argument("--codec-dir", action="append", required=True, help="name:dir, repeatable")
    bp.add_argument("--out-pairs", required=True)
    bp.add_argument("--out-map", required=True)
    bp.set_defaults(func=cmd_build_pairs)

    fz = sub.add_parser("finalize")
    fz.add_argument("--ext-csv", required=True)
    fz.add_argument("--map", required=True)
    fz.add_argument("--original", required=True)
    fz.add_argument("--out", required=True)
    fz.add_argument(
        "--flag-l2",
        type=float,
        default=0.5,
        help="per-row L2(f0..f371 diff) above which a matched row is flagged as material "
        "drift (known contamination / encoder-version drift) rather than benign "
        "cross-backend (GPU-original vs CPU-backfill) noise. Default 0.5 chosen from the "
        "observed noise floor (~0.01-0.05) vs known-contaminated cells (>1) — see the "
        "benchmark note for the calibration.",
    )
    fz.add_argument("--drop-flagged", action="store_true", help="drop flagged rows instead of keeping them with drift_flag=1")
    fz.set_defaults(func=cmd_finalize)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
