#!/usr/bin/env python3
"""Score-quality diagnostics on the unified v_next parquet(s).

Surfaces three categories of zensim quality issue, all detected from data the
unified parquet already has (no model retrain needed):

1. Monotonicity violations
   - Sort by q within each (image, codec, knob_tuple_json) group.
   - Count adjacent (q_i < q_{i+1}) pairs where score went DOWN by > eps.
   - Report worst offenders + per-codec aggregates.

2. Bumpiness (curve roughness)
   - For each monotone-enough curve, fit a smooth reference (rolling median)
     and compute mean absolute residual of zensim_score from it.
   - Compare to reference metrics (ssim2, butteraugli) on the same curve to
     see if "bumpiness" is a property of the input pair (e.g. codec mode
     switch) vs zensim specifically.

3. Disagreement with reference metrics
   - For every cell, compute residual = zensim - 100*(1 - clamp(butteraugli_max/6, 0, 1))
     and the simpler delta vs ssim2.
   - Emit top-K positive/negative pairs as adversarial-mining seeds (TODO 4.4).

Outputs:
  /mnt/v/zen/zensim-training/2026-05-07/unified/score_quality_summary.tsv
  /mnt/v/zen/zensim-training/2026-05-07/unified/monotonicity_violations.tsv
  /mnt/v/zen/zensim-training/2026-05-07/unified/adversarial_pairs_top_disagree.parquet

Usage:
  python3 analyze_score_quality.py [--input-dir DIR] [--top-k 5000]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DEFAULT_DIR = Path("/mnt/v/zen/zensim-training/2026-05-07/unified")


def load_unified(input_dir: Path, sweeps: list[str] | None = None) -> pd.DataFrame:
    parqs = sorted(input_dir.glob("unified_*.parquet"))
    if sweeps:
        parqs = [p for p in parqs
                 if any(p.name.startswith(f"unified_{s}_") for s in sweeps)]
    if not parqs:
        raise SystemExit(f"no unified parquets in {input_dir} matching {sweeps}")
    print(f"Loading {len(parqs)} parquet shards ...")
    keep_cols: list[str] | None = None
    frames: list[pd.DataFrame] = []
    for p in parqs:
        try:
            cols_in_file = pq.ParquetFile(p).schema.names
        except Exception as e:
            print(f"  SKIP {p.name}: {e}")
            continue
        if keep_cols is None:
            keep_cols = [c for c in cols_in_file if not c.startswith("feat_")]
        keep_now = [c for c in keep_cols if c in cols_in_file]
        df = pq.read_table(p, columns=keep_now).to_pandas()
        df["__shard"] = p.name
        frames.append(df)
        print(f"  {p.name}: {len(df):,} rows")
    if not frames:
        raise SystemExit("no readable parquets")
    full = pd.concat(frames, ignore_index=True)
    print(f"Total: {len(full):,} rows × {full.shape[1]} cols")
    return full


def find_monotonicity_violations(df: pd.DataFrame, eps: float = 0.5) -> pd.DataFrame:
    """For each (image, codec, knob_tuple_json) group, sort by q and find
    adjacent q-pairs where score_zensim went DOWN by > eps."""
    keys = ["sweep_id", "image_basename", "codec", "knob_tuple_json"]
    needed = keys + ["q", "score_zensim", "score_ssim2", "score_butteraugli_max"]
    work = df[[c for c in needed if c in df.columns]].dropna(
        subset=["score_zensim", "q"]
    ).copy()
    work = work.sort_values(keys + ["q"])
    grouped = work.groupby(keys, sort=False)
    rows = []
    for key_vals, g in grouped:
        if len(g) < 2:
            continue
        zs = g["score_zensim"].to_numpy()
        qs = g["q"].to_numpy()
        diffs = np.diff(zs)
        for i, d in enumerate(diffs):
            if d < -eps:
                rec = dict(zip(keys, key_vals))
                rec.update({
                    "q_lo": qs[i], "q_hi": qs[i+1],
                    "zensim_lo": zs[i], "zensim_hi": zs[i+1],
                    "drop": d,
                })
                rows.append(rec)
    return pd.DataFrame(rows)


def bumpiness_per_curve(df: pd.DataFrame) -> pd.DataFrame:
    """For each (image, codec, knobs) curve, compute residual of zensim_score
    from a rolling-median smooth as a roughness proxy."""
    keys = ["sweep_id", "image_basename", "codec", "knob_tuple_json"]
    work = df.dropna(subset=["score_zensim", "q"]).sort_values(keys + ["q"])
    rows = []
    for key_vals, g in work.groupby(keys, sort=False):
        if len(g) < 5:
            continue
        z = g["score_zensim"].to_numpy()
        smooth = pd.Series(z).rolling(3, center=True, min_periods=1).median().to_numpy()
        bumpiness = float(np.mean(np.abs(z - smooth)))
        rec = dict(zip(keys, key_vals))
        rec["n_q"] = len(g)
        rec["bumpiness_mae"] = bumpiness
        rec["zensim_min"] = float(z.min())
        rec["zensim_max"] = float(z.max())
        rows.append(rec)
    return pd.DataFrame(rows)


def disagreement_residual(df: pd.DataFrame) -> pd.DataFrame:
    """delta_ba = zensim_score - 100*(1 - clip(butteraugli_max/6, 0, 1))
       delta_ssim2 = zensim_score - score_ssim2

    Returns whichever residuals are computable from the available columns.
    Some sweep×codec combos lack one or both reference metrics.
    """
    needed = ["score_zensim"]
    has_ba = "score_butteraugli_max" in df.columns
    has_s2 = "score_ssim2" in df.columns
    if not has_ba and not has_s2:
        return pd.DataFrame()
    drop_subset = needed + (["score_butteraugli_max"] if has_ba else []) \
                          + (["score_ssim2"] if has_s2 else [])
    work = df.dropna(subset=drop_subset).copy()
    if has_ba:
        ba_proxy = 100.0 * (1.0 - np.clip(work["score_butteraugli_max"] / 6.0,
                                           0.0, 1.0))
        work["delta_ba"] = work["score_zensim"] - ba_proxy
    if has_s2:
        work["delta_ssim2"] = work["score_zensim"] - work["score_ssim2"]
    cols = ["sweep_id", "image_basename", "codec", "q", "knob_tuple_json",
            "score_zensim"]
    if has_ba:
        cols += ["score_butteraugli_max", "delta_ba"]
    if has_s2:
        cols += ["score_ssim2", "delta_ssim2"]
    return work[cols]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=str(DEFAULT_DIR))
    ap.add_argument("--sweeps", default=None,
                    help="Comma-separated sweep ids to include (default: all readable)")
    ap.add_argument("--top-k", type=int, default=5000)
    ap.add_argument("--mono-eps", type=float, default=0.5,
                    help="Min downward step (in zensim points) to flag")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    sweeps = args.sweeps.split(",") if args.sweeps else None
    df = load_unified(in_dir, sweeps=sweeps)

    print("\n# 1. Monotonicity violations")
    mv = find_monotonicity_violations(df, eps=args.mono_eps)
    if mv.empty:
        print("  no violations found at eps =", args.mono_eps)
    else:
        out = in_dir / "monotonicity_violations.tsv"
        mv.sort_values("drop").to_csv(out, sep="\t", index=False)
        agg = mv.groupby(["sweep_id", "codec"])["drop"].agg(
            count="size", mean_drop="mean", worst_drop="min"
        )
        print(f"  {len(mv):,} violations (eps={args.mono_eps}) → {out.name}")
        print(agg.to_string())

    print("\n# 2. Bumpiness per curve")
    bp = bumpiness_per_curve(df)
    if not bp.empty:
        out = in_dir / "bumpiness_per_curve.tsv"
        bp.sort_values("bumpiness_mae", ascending=False).to_csv(out, sep="\t",
                                                                  index=False)
        agg = bp.groupby(["sweep_id", "codec"])["bumpiness_mae"].agg(
            n="size", mean="mean", p95=lambda s: float(np.quantile(s, 0.95)),
            max="max"
        )
        print(f"  {len(bp):,} curves → {out.name}")
        print(agg.to_string())

    print("\n# 3. zensim ↔ reference-metric disagreement")
    da = disagreement_residual(df)
    if da.empty:
        print("  no reference-metric columns available; skipping")
    else:
        sort_col = "delta_ba" if "delta_ba" in da.columns else "delta_ssim2"
        top_high = da.nlargest(args.top_k, sort_col)
        top_low = da.nsmallest(args.top_k, sort_col)
        adv = pd.concat([top_high, top_low], ignore_index=True)
        out = in_dir / "adversarial_pairs_top_disagree.parquet"
        adv.to_parquet(out, compression="zstd", compression_level=9)
        print(f"  {len(adv):,} adversarial pairs (sorted by {sort_col}) → {out.name}")
        for col in ("delta_ba", "delta_ssim2"):
            if col not in da.columns:
                continue
            print(f"  {col}: mean={da[col].mean():+.2f} "
                  f"std={da[col].std():.2f} "
                  f"p1={np.quantile(da[col], 0.01):+.2f} "
                  f"p99={np.quantile(da[col], 0.99):+.2f}")
            per_codec = da.groupby(["sweep_id", "codec"])[col].agg(
                n="size", mean="mean", std="std",
                p1=lambda s: float(np.quantile(s, 0.01)),
                p99=lambda s: float(np.quantile(s, 0.99)),
            )
            print(per_codec.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
