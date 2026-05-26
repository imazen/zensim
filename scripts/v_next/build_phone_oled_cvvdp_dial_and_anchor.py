#!/usr/bin/env python3
"""Build modern_oled_phone_indoor CVVDP dial-target training parquets +
a phone-CVVDP-DERIVED multi-band spline anchor for zensim-b-phone.

CORRECTION 2 (2026-05-26): the prior zensim-b-phone bake used a
DESKTOP-CVVDP-derived spline anchor (anchors_cvvdp_372col_continuous.parquet,
target_score built from desktop/ssim2 substrate). Its target distribution
(p50≈93.6) did not match the phone-CVVDP dial distribution → broken dial
(G1=0.00, p5=96 p95=138). This script fixes that by building the anchor
FROM the phone-CVVDP training data itself: sample ~N rows stratified across
the full phone-CVVDP-dial range [0,100], set target_score = phone-CVVDP dial.

Inputs (per corpus):
  <scores-dir>/<corpus>_cvvdp_phone.tsv  — phone-CVVDP JOD per row, in the
      SAME row order as the canonical train parquet (produced by
      `zen-metrics batch --display-model modern_oled_phone_indoor`, pairs
      TSV row-aligned to the canonical parquet via build_iphone14 pairs).
  <train-dir>/<corpus>.parquet           — canonical 372-feature parquet.

Outputs to <out-dir>:
  <corpus>_phone_cvvdptgt.parquet   — features + human_score=dial(cvvdp_phone)/100
                                       + cvvdp_phone raw + cvvdp_phone_dial.
  modern_oled_anchor.parquet        — pooled stratified sample, target_score =
                                       phone-CVVDP dial (0..100), anchor_weight,
                                       372 features. Drops into --anchor-parquet.

Dial transform: identical V12 monotone band map used by the canonical
cvvdp anchor family (so it lives in the same 0..100 score units the
trainer's score-shaped output + anchor target_score use). Monotone in
CVVDP-JOD → SROCC vs raw phone-CVVDP is preserved.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# V12 (cvvdp_JOD, dial_score) band table — the canonical CVVDP->dial map
# (scripts/v_next/v12_cvvdp/build_v12_cvvdp_substrate.py) shared by the
# anchor family. Same transform regardless of display model: it maps a
# JOD value to a 0..100 dial. The display model changes the JOD values
# (phone makes artifacts MORE visible -> lower JOD -> lower dial), which
# is exactly the point.
V12_BANDS = [
    (10.00, 100.0),
    (9.95, 95.0),
    (9.85, 90.0),
    (9.65, 80.0),   # JND
    (9.30, 65.0),
    (8.50, 50.0),   # JOD
    (7.50, 35.0),
    (6.50, 20.0),
    (5.00, 10.0),
    (3.00, 0.0),
]
_BX = [b[0] for b in V12_BANDS][::-1]
_BY = [b[1] for b in V12_BANDS][::-1]


def cvvdp_to_dial(cvvdp: np.ndarray) -> np.ndarray:
    """Monotone piecewise-linear CVVDP-JOD -> 0..100 dial (V12 band map).
    np.interp clamps to endpoints (JOD<=3 -> 0, JOD>=10 -> 100)."""
    return np.interp(cvvdp, _BX, _BY)


def load_scores_tsv(path: Path) -> np.ndarray:
    scores = []
    with open(path) as f:
        r = csv.DictReader(f, delimiter="\t")
        col = next(c for c in r.fieldnames if c.startswith("cvvdp"))
        for row in r:
            try:
                scores.append(float(row[col]))
            except (ValueError, TypeError):
                scores.append(np.nan)
    return np.asarray(scores, dtype=float)


def build_dial_parquet(corpus: str, parquet: Path, scores_tsv: Path, out: Path):
    tbl = pq.read_table(str(parquet))
    n = tbl.num_rows
    scores = load_scores_tsv(scores_tsv)
    if len(scores) != n:
        raise RuntimeError(
            f"{corpus}: row count mismatch — parquet {n} vs scores {len(scores)}"
        )
    finite = np.isfinite(scores)
    if finite.sum() < n:
        print(f"  {corpus}: {n - int(finite.sum())} non-finite CVVDP rows dropped")

    dial = cvvdp_to_dial(scores)
    names = tbl.column_names
    cols = {nm: tbl.column(nm) for nm in names}
    cols["human_score"] = pa.array(dial / 100.0)  # trainer *100 -> 0..100
    cols["cvvdp_phone"] = pa.array(scores)
    cols["cvvdp_phone_dial"] = pa.array(dial)
    new_names = list(names)
    for extra in ("cvvdp_phone", "cvvdp_phone_dial"):
        if extra not in new_names:
            new_names.append(extra)
    out_tbl = pa.table({nm: cols[nm] for nm in new_names})
    if finite.sum() < n:
        out_tbl = out_tbl.filter(pa.array(finite))
    pq.write_table(out_tbl, str(out), compression="zstd", compression_level=15)
    print(
        f"  WROTE {out} ({out_tbl.num_rows} rows; "
        f"dial p5={np.nanpercentile(dial[finite],5):.1f} "
        f"p50={np.nanpercentile(dial[finite],50):.1f} "
        f"p95={np.nanpercentile(dial[finite],95):.1f})"
    )
    return out_tbl


def build_anchor(dial_tables: list[pa.Table], n_target: int, n_bins: int,
                 out: Path, seed: int = 17):
    """Stratified sample across the full phone-CVVDP-dial range [0,100].

    Pool all corpora, bin the dial into n_bins uniform-width bins, draw
    ~n_target/n_bins rows from each non-empty bin (uniform within bin),
    so the anchor's target_score distribution spans the WHOLE dial range
    rather than clustering where the corpus happens to concentrate. The
    bake's spline then calibrates against a target distribution that
    matches the phone-CVVDP dial it's emulating.
    """
    rng = np.random.default_rng(seed)
    feat_cols = sorted(
        [c for c in dial_tables[0].column_names if c.startswith("f") and c[1:].isdigit()],
        key=lambda c: int(c[1:]),
    )
    # Pool dial + features across corpora.
    dial_all = []
    feat_all = []
    base_all = []
    for t in dial_tables:
        d = np.asarray(t.column("cvvdp_phone_dial").to_numpy(zero_copy_only=False), dtype=float)
        f = np.column_stack([
            np.asarray(t.column(c).to_numpy(zero_copy_only=False), dtype=np.float64)
            for c in feat_cols
        ])
        b = np.asarray(t.column("ref_basename").to_numpy(zero_copy_only=False))
        dial_all.append(d); feat_all.append(f); base_all.append(b)
    dial_all = np.concatenate(dial_all)
    feat_all = np.concatenate(feat_all, axis=0)
    base_all = np.concatenate(base_all)
    finite = np.isfinite(dial_all) & np.all(np.isfinite(feat_all), axis=1)
    dial_all, feat_all, base_all = dial_all[finite], feat_all[finite], base_all[finite]

    edges = np.linspace(0.0, 100.0, n_bins + 1)
    per_bin = max(1, n_target // n_bins)
    pick_idx = []
    bin_counts = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = np.where((dial_all >= lo) & (dial_all < hi if i < n_bins - 1 else dial_all <= hi))[0]
        bin_counts.append(len(in_bin))
        if len(in_bin) == 0:
            continue
        k = min(per_bin, len(in_bin))
        pick_idx.append(rng.choice(in_bin, size=k, replace=False))
    pick = np.concatenate(pick_idx)
    rng.shuffle(pick)

    cols = {c: pa.array(feat_all[pick, j]) for j, c in enumerate(feat_cols)}
    cols["target_score"] = pa.array(dial_all[pick])
    cols["anchor_weight"] = pa.array(np.ones(len(pick), dtype=np.float64))
    cols["human_score"] = pa.array(dial_all[pick])  # alias, same shape
    cols["ref_basename"] = pa.array(base_all[pick])
    cols["anchor_source"] = pa.array(np.array(["phone_cvvdp_stratified"] * len(pick)))
    order = ["ref_basename", "anchor_source", "human_score", "anchor_weight",
             "target_score"] + feat_cols
    anchor_tbl = pa.table({k: cols[k] for k in order})
    pq.write_table(anchor_tbl, str(out), compression="zstd", compression_level=15)
    ts = dial_all[pick]
    print(
        f"  ANCHOR {out}: {len(pick)} rows; target_score "
        f"p5={np.percentile(ts,5):.1f} p50={np.percentile(ts,50):.1f} "
        f"p95={np.percentile(ts,95):.1f} min={ts.min():.1f} max={ts.max():.1f}"
    )
    print(f"  bin occupancy (n_bins={n_bins}): {bin_counts}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir",
                    default="/mnt/v/zen/zensim-training/canonical-2026-05-21/train")
    ap.add_argument("--scores-dir",
                    default="/mnt/v/output/zensim/iphone14-cvvdp-2026-05-25")
    ap.add_argument("--out-dir",
                    default="/mnt/v/output/zensim/iphone14-cvvdp-2026-05-25")
    ap.add_argument("--corpora", default="kadid,tid")
    ap.add_argument("--anchor-rows", type=int, default=2000)
    ap.add_argument("--anchor-bins", type=int, default=20)
    args = ap.parse_args()

    train = Path(args.train_dir)
    scores = Path(args.scores_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dial_tables = []
    for corpus in args.corpora.split(","):
        corpus = corpus.strip()
        print(f"--- {corpus}")
        t = build_dial_parquet(
            corpus,
            train / f"{corpus}.parquet",
            scores / f"{corpus}_cvvdp_phone.tsv",
            out / f"{corpus}_phone_cvvdptgt.parquet",
        )
        dial_tables.append(t)

    print("--- anchor (phone-CVVDP-derived, stratified)")
    build_anchor(dial_tables, args.anchor_rows, args.anchor_bins,
                 out / "modern_oled_anchor.parquet")


if __name__ == "__main__":
    main()
