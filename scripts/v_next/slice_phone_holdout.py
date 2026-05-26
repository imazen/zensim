#!/usr/bin/env python3
"""Carve a held-out slice (by ref_basename) from the phone-CVVDP dial
parquets for an honest zensim-b-phone tracking eval.

Splits each corpus's *_phone_cvvdptgt.parquet into _train + _holdout by
hashing ref_basename (deterministic, ~holdout-frac of distinct refs go
to holdout). The tracking eval (SROCC of bake output vs held-out
phone-CVVDP) then runs on refs the bake never trained on.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def ref_in_holdout(ref: str, frac: float, salt: str) -> bool:
    h = hashlib.sha256((salt + str(ref)).encode()).hexdigest()
    # map first 8 hex digits to [0,1)
    return (int(h[:8], 16) / 0xFFFFFFFF) < frac


def split_one(path: Path, frac: float, salt: str):
    tbl = pq.read_table(str(path))
    refs = [str(x) for x in tbl.column("ref_basename").to_pylist()]
    distinct = sorted(set(refs))
    hold_refs = {r for r in distinct if ref_in_holdout(r, frac, salt)}
    mask = np.array([r in hold_refs for r in refs])
    train_tbl = tbl.filter(~mask)
    hold_tbl = tbl.filter(mask)
    train_out = path.with_name(path.stem + "_train.parquet")
    hold_out = path.with_name(path.stem + "_holdout.parquet")
    pq.write_table(train_tbl, str(train_out), compression="zstd", compression_level=15)
    pq.write_table(hold_tbl, str(hold_out), compression="zstd", compression_level=15)
    print(
        f"{path.name}: {tbl.num_rows} rows ({len(distinct)} refs) -> "
        f"train {train_tbl.num_rows} ({len(distinct)-len(hold_refs)} refs), "
        f"holdout {hold_tbl.num_rows} ({len(hold_refs)} refs)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dial-dir",
                    default="/mnt/v/output/zensim/iphone14-cvvdp-2026-05-25")
    ap.add_argument("--corpora", default="kadid,tid")
    ap.add_argument("--holdout-frac", type=float, default=0.20)
    ap.add_argument("--salt", default="zensim_b_phone_oled_2026-05-26")
    args = ap.parse_args()
    d = Path(args.dial_dir)
    for c in args.corpora.split(","):
        split_one(d / f"{c.strip()}_phone_cvvdptgt.parquet",
                  args.holdout_frac, args.salt)


if __name__ == "__main__":
    main()
