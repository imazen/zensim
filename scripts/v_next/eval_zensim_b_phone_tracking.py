#!/usr/bin/env python3
"""Tracking eval for zensim-b-phone: SROCC of bake output vs the held-out
iPhone-14 CVVDP scores it was trained to emulate.

This is the ACTUAL goal metric — "is it a good iPhone-14 CVVDP emulator"
— NOT human-MOS. We also report PLCC + KROCC for the full picture, and
the bake's output distribution (a sanity check on the dial spread).

Run on each corpus's dial-target parquet (features + cvvdp_iphone14 raw).
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr, pearsonr, kendalltau

PREDICT_BIN = "/home/lilith/work/zen/zensim/target/release/predict_features_with_bake"


def bake_predict(bake: str, features: np.ndarray) -> np.ndarray:
    n_rows, n_features = features.shape
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmp = f.name
        f.write(struct.pack("<II", n_features, n_rows))
        f.write(features.astype(np.float32).tobytes())
    try:
        res = subprocess.run(
            [PREDICT_BIN, "--bake", bake, "--features-file", tmp],
            capture_output=True, text=True, timeout=300, check=True,
        )
    finally:
        os.unlink(tmp)
    preds = []
    for line in res.stdout.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            preds.append(float(line))
        except ValueError:
            pass
    return np.asarray(preds, dtype=float)


def eval_corpus(bake: str, parquet: Path, name: str) -> dict:
    tbl = pq.read_table(str(parquet))
    fcols = [c for c in tbl.column_names if c.startswith("f") and c[1:].isdigit()]
    fcols.sort(key=lambda c: int(c[1:]))
    feats = np.column_stack(
        [np.asarray(tbl.column(c).to_numpy(zero_copy_only=False), dtype=np.float32)
         for c in fcols]
    )
    cvvdp = np.asarray(
        tbl.column("cvvdp_iphone14").to_numpy(zero_copy_only=False), dtype=float
    )
    preds = bake_predict(bake, feats)
    if len(preds) != len(cvvdp):
        raise RuntimeError(f"{name}: pred/target len mismatch {len(preds)} vs {len(cvvdp)}")
    m = np.isfinite(preds) & np.isfinite(cvvdp)
    p, c = preds[m], cvvdp[m]
    srocc = spearmanr(p, c).correlation
    plcc = pearsonr(p, c)[0]
    krocc = kendalltau(p, c).correlation
    return {
        "name": name, "n": int(m.sum()),
        "srocc": srocc, "plcc": plcc, "krocc": krocc,
        "pred_p5": float(np.percentile(p, 5)),
        "pred_p50": float(np.percentile(p, 50)),
        "pred_p95": float(np.percentile(p, 95)),
        "pred_min": float(p.min()), "pred_max": float(p.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", required=True)
    ap.add_argument(
        "--dial-dir", default="/mnt/v/output/zensim/iphone14-cvvdp-2026-05-25"
    )
    args = ap.parse_args()

    dial = Path(args.dial_dir)
    print(f"=== zensim-b-phone tracking eval: SROCC vs iPhone-14 CVVDP ===")
    print(f"bake: {args.bake}\n")
    all_p, all_c = [], []
    rows = []
    for corpus in ["kadid", "tid"]:
        r = eval_corpus(args.bake, dial / f"{corpus}_iphone14_cvvdptgt.parquet", corpus)
        rows.append(r)
        print(
            f"{r['name']:>6}: n={r['n']:>5} | SROCC={r['srocc']:.4f} "
            f"PLCC={r['plcc']:.4f} KROCC={r['krocc']:.4f} | "
            f"pred dial p5/p50/p95 = {r['pred_p5']:.1f}/{r['pred_p50']:.1f}/{r['pred_p95']:.1f} "
            f"[{r['pred_min']:.1f},{r['pred_max']:.1f}]"
        )
    print()
    # Pooled
    print("pooled across corpora:")
    pooled = {}
    # Recompute pooled by concatenation
    cat_p, cat_c = [], []
    for corpus in ["kadid", "tid"]:
        tbl = pq.read_table(str(dial / f"{corpus}_iphone14_cvvdptgt.parquet"))
        fcols = sorted([c for c in tbl.column_names if c.startswith("f") and c[1:].isdigit()],
                       key=lambda c: int(c[1:]))
        feats = np.column_stack(
            [np.asarray(tbl.column(c).to_numpy(zero_copy_only=False), dtype=np.float32) for c in fcols]
        )
        cvvdp = np.asarray(tbl.column("cvvdp_iphone14").to_numpy(zero_copy_only=False), dtype=float)
        preds = bake_predict(args.bake, feats)
        cat_p.append(preds); cat_c.append(cvvdp)
    cat_p = np.concatenate(cat_p); cat_c = np.concatenate(cat_c)
    m = np.isfinite(cat_p) & np.isfinite(cat_c)
    print(f"   n={m.sum()} | SROCC={spearmanr(cat_p[m],cat_c[m]).correlation:.4f} "
          f"PLCC={pearsonr(cat_p[m],cat_c[m])[0]:.4f} "
          f"KROCC={kendalltau(cat_p[m],cat_c[m]).correlation:.4f}")


if __name__ == "__main__":
    main()
