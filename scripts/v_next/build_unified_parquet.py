#!/usr/bin/env python3
"""Build unified training parquet from v15r/v15rc/v13/v14/v12 sweeps.

For each sweep, joins the per-source `.features.parquet` (300 zensim features
+ knob keys) with its matching `.tsv` (encoded_bytes, encode_ms, decode_ms,
score_{zensim,ssim2,butteraugli_max,butteraugli_pnorm3}) on the composite
key (basename(image_path), codec, q, knob_tuple_json).

Adds:
- `sweep_id`         — e.g. "v15r", "v15rc", "v12"
- `image_basename`   — strips /workspace/sweep/stage-*/ prefix from image_path
- `content_class`    — joined from features_v15r_combined.tsv (v15r/v15rc only)
- `corpus_features_*`— 33 named zenanalyze features per source (v15r/v15rc only)

Output: /mnt/v/zen/zensim-training/2026-05-07/unified/training_unified.parquet

Usage:
    python3 build_unified_parquet.py [--sweeps v15r,v15rc,v13,v14,v12]
                                     [--limit-files N]   # debug: only N TSVs per sweep
                                     [--out PATH]
"""
import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Route every join through join_safety — see zensim/scripts/canonical_corpus/.
# The two pre-W44 raw `tsv.merge(...)` + `df.merge(corpus_features, on=
# "image_basename")` sites were the literal Mode-B "ref-only broadcast" shape
# that produced the 2026-05-25 kadid/tid metric corruption.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "scripts" / "canonical_corpus")
)
from join_safety import (  # noqa: E402
    attach_per_source_features,
    guard_metric_table,
    safe_metric_join,
)

ROOT = Path("/mnt/v/zen/zensim-training/2026-05-07")
DEFAULT_OUT = ROOT / "unified" / "training_unified.parquet"
CORPUS_FEATURES_TSV = ROOT / "v15r-prep" / "features_v15r_combined.tsv"

# Per-sweep TSV+parquet locations, by (sweep_id, codec) → directory
SWEEP_LAYOUT = {
    ("v15r",  "zenjpeg"): (
        ROOT / "v15r-prep" / "data" / "zenjpeg",   # TSVs (synced earlier)
        ROOT / "v15r" / "zenjpeg",                  # feature parquets
    ),
    ("v15rc", "zenjpeg"): (
        ROOT / "v15rc" / "zenjpeg",                # both TSVs + parquets co-located
        ROOT / "v15rc" / "zenjpeg",
    ),
    ("v13",   "zenjpeg"): (
        ROOT / "v13" / "zenjpeg",
        ROOT / "v13" / "zenjpeg",
    ),
    ("v14",   "zenpng"): (
        ROOT / "v14" / "zenpng",
        ROOT / "v14" / "zenpng",
    ),
    ("v12",   "zenavif"): (
        ROOT / "v12" / "zenavif",
        ROOT / "v12" / "zenavif",
    ),
    ("v12",   "zenjxl"): (
        ROOT / "v12" / "zenjxl",
        ROOT / "v12" / "zenjxl",
    ),
    ("v12",   "zenwebp"): (
        ROOT / "v12" / "zenwebp",
        ROOT / "v12" / "zenwebp",
    ),
}

WORKSPACE_PREFIX_RE = re.compile(r"^/workspace/sweep/stage-[^/]+/")


def normalize_basename(image_path: str) -> str:
    """Strip /workspace/sweep/stage-*/ prefix from image path. Returns basename."""
    p = WORKSPACE_PREFIX_RE.sub("", image_path)
    return os.path.basename(p)


def load_corpus_features() -> pd.DataFrame:
    """Load 33-feature zenanalyze TSV and build a basename→features lookup."""
    if not CORPUS_FEATURES_TSV.exists():
        print(f"WARN: corpus features TSV missing: {CORPUS_FEATURES_TSV}")
        return pd.DataFrame()
    df = pd.read_csv(CORPUS_FEATURES_TSV, sep="\t")
    df["image_basename"] = df["image_path"].map(normalize_basename)
    keep = ["image_basename", "content_class", "size_class", "width", "height"]
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    keep += feat_cols
    out = df[keep].drop_duplicates(subset=["image_basename"]).rename(
        columns={c: f"corpus_{c}" for c in feat_cols}
    )
    print(f"  corpus features: {len(out)} unique basenames, {len(feat_cols)} feat_* cols")
    return out


NUMERIC_TSV_COLS = (
    "encoded_bytes", "encode_ms", "decode_ms",
    "score_zensim", "score_ssim2",
    "score_butteraugli_max", "score_butteraugli_pnorm3",
    "score_ssim2_gpu",
    "score_butteraugli_max_gpu", "score_butteraugli_pnorm3_gpu",
)


CANONICAL_METRIC_COLS = (
    "score_zensim", "score_ssim2",
    "score_butteraugli_max", "score_butteraugli_pnorm3",
)


def coalesce_gpu_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Some workers emitted `score_ssim2_gpu` / `score_butteraugli_*_gpu` while
    others emitted CPU column names. Coalesce into a single canonical name and
    record which path the row used in `metric_runtime` ("gpu" | "cpu" | "mixed").
    """
    runtime = pd.Series(["cpu"] * len(df), index=df.index)
    for canonical in CANONICAL_METRIC_COLS:
        gpu_col = canonical + "_gpu"
        if gpu_col in df.columns and canonical in df.columns:
            # Prefer non-null value from either column.
            df[canonical] = df[canonical].where(df[canonical].notna(), df[gpu_col])
            runtime = runtime.where(df[gpu_col].isna(), "gpu")
            df = df.drop(columns=[gpu_col])
        elif gpu_col in df.columns:
            df = df.rename(columns={gpu_col: canonical})
            runtime = pd.Series(["gpu"] * len(df), index=df.index)
    df["metric_runtime"] = runtime
    return df


def join_pair(tsv_path: Path, parquet_path: Path) -> pd.DataFrame | None:
    """Join one (.tsv, .features.parquet) pair on the composite key.

    Forces numeric metric columns to float64 (empty strings on encode/decode
    failures become NaN) and coalesces `score_*_gpu` aliases so streaming
    concat across files keeps a stable pyarrow schema.
    """
    if not tsv_path.exists() or not parquet_path.exists():
        return None
    tsv = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    for col in NUMERIC_TSV_COLS:
        if col in tsv.columns:
            tsv[col] = pd.to_numeric(tsv[col], errors="coerce")
    feat = pq.read_table(parquet_path).to_pandas()

    key = ["image_path", "codec", "q", "knob_tuple_json"]
    # The features table is keyed on the full per-pair key; we merge that
    # alongside whatever metric columns came in via the TSV. Use a raw
    # `target.merge(source, on=key)` here (NOT safe_metric_join, which is
    # for a SINGLE metric column at a time) because `feat` carries many
    # feat_* columns — but assert metric-side uniqueness on the full key
    # so a duplicated feature row can't silently broadcast.
    feat_dupes = int(feat.duplicated(subset=key).sum())
    if feat_dupes:
        raise RuntimeError(
            f"join_pair({tsv_path.name}): features parquet has {feat_dupes} "
            f"rows duplicated on {key} — averaging or first-wins broadcast "
            f"would silently corrupt the per-pair attach. Re-emit the "
            f"features parquet de-duplicated."
        )
    merged = tsv.merge(feat, on=key, how="inner", suffixes=("", "_feat"))  # joinsafety-ok: per-pair full-key merge with explicit uniqueness guard above
    if "zensim_score" in merged.columns and "score_zensim" in merged.columns:
        merged = merged.drop(columns=["zensim_score"])
    merged = coalesce_gpu_metrics(merged)
    return merged


def build_sweep(sweep_id: str, codec: str, tsv_dir: Path, parq_dir: Path,
                limit_files: int | None) -> pd.DataFrame:
    rows = []
    tsvs = sorted(tsv_dir.glob(f"{codec}-*.tsv"))
    if limit_files:
        tsvs = tsvs[:limit_files]
    for tsv in tsvs:
        # Find matching parquet — basenames identical except .tsv → .features.parquet
        parq = parq_dir / tsv.name.replace(".tsv", ".features.parquet")
        df = join_pair(tsv, parq)
        if df is None or df.empty:
            continue
        df["sweep_id"] = sweep_id
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["image_basename"] = out["image_path"].map(normalize_basename)
    return out


def write_sweep_parquet(out_path: Path, sweep_id: str, codec: str,
                        tsv_dir: Path, parq_dir: Path,
                        corpus_features: pd.DataFrame,
                        limit_files: int | None) -> int:
    """Streaming write: process one TSV/parquet pair at a time, append to a
    ParquetWriter so we never hold the full sweep in RAM."""
    tsvs = sorted(tsv_dir.glob(f"{codec}-*.tsv"))
    if limit_files:
        tsvs = tsvs[:limit_files]
    if not tsvs:
        return 0
    writer: pq.ParquetWriter | None = None
    total = 0
    try:
        for i, tsv in enumerate(tsvs):
            parq = parq_dir / tsv.name.replace(".tsv", ".features.parquet")
            df = join_pair(tsv, parq)
            if df is None or df.empty:
                continue
            df["sweep_id"] = sweep_id
            df["image_basename"] = df["image_path"].map(normalize_basename)
            if not corpus_features.empty:
                # Legitimate per-source attach (image_basename → many
                # distortions per ref). attach_per_source_features enforces
                # source-side uniqueness on image_basename so a duplicated
                # source row cannot silently broadcast different feature
                # values onto the same ref's distortions.
                df = attach_per_source_features(
                    df, corpus_features, "image_basename", how="left"
                )
            tbl = pa.Table.from_pandas(df, preserve_index=False)
            # Post-attach guard: catches mock leak (Mode A) + any ssim2/cvvdp/
            # butter/dssim column constant-per-ref (Mode B broadcast signature).
            guard_metric_table(
                f"build_unified[{sweep_id}/{codec}]", tbl,
                source_key="image_basename",
            )
            if writer is None:
                writer = pq.ParquetWriter(out_path, tbl.schema,
                                          compression="zstd",
                                          compression_level=9)
            writer.write_table(tbl)
            total += len(df)
            if (i + 1) % 50 == 0:
                print(f"    [{sweep_id}/{codec}] {i+1}/{len(tsvs)} files, "
                      f"{total:,} rows so far", flush=True)
    finally:
        if writer is not None:
            writer.close()
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", default="v15r,v15rc,v13,v14,v12",
                    help="Comma-separated sweep ids to include")
    ap.add_argument("--limit-files", type=int, default=None,
                    help="Per-sweep cap on TSV/parquet pairs (debug)")
    ap.add_argument("--out-dir", default=str(ROOT / "unified"),
                    help="Per-sweep parquet output dir")
    args = ap.parse_args()

    wanted = set(args.sweeps.split(","))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building per-sweep unified parquets → {out_dir}")
    print("Loading corpus features (zenanalyze 33-feature TSV) ...")
    corpus_features = load_corpus_features()

    summary: list[tuple[str, str, int, str]] = []
    for (sweep_id, codec), (tsv_dir, parq_dir) in SWEEP_LAYOUT.items():
        if sweep_id not in wanted:
            continue
        out_path = out_dir / f"unified_{sweep_id}_{codec}.parquet"
        print(f"\n[{sweep_id}/{codec}] → {out_path.name}")
        rows = write_sweep_parquet(out_path, sweep_id, codec, tsv_dir, parq_dir,
                                    corpus_features, args.limit_files)
        if rows == 0:
            print(f"  -> 0 rows (no parquet written)")
            if out_path.exists():
                out_path.unlink()
            continue
        sz_mb = out_path.stat().st_size / 1024 / 1024
        print(f"  -> {rows:,} rows, {sz_mb:.1f} MB")
        summary.append((sweep_id, codec, rows, f"{sz_mb:.1f} MB"))

    print("\nSummary:")
    print(f"  {'sweep':<6} {'codec':<10} {'rows':>12} {'size':>12}")
    for sid, c, r, s in summary:
        print(f"  {sid:<6} {c:<10} {r:>12,} {s:>12}")
    print(f"  {'TOTAL':<17} {sum(r for _,_,r,_ in summary):>12,}")
    return 0 if summary else 1


if __name__ == "__main__":
    sys.exit(main())
