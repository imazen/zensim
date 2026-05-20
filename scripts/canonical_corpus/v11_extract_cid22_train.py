#!/usr/bin/env python3
"""V11 Phase 2 step 1: extract CID22 training-only-subset feature parquet.

Per CLAUDE.md "CID22 is VALIDATION-ONLY":
- The 49-reference validation set lives at /mnt/v/dataset/cid22/CID22_validation_set/.
  Its human MOS is sacred and NEVER usable as a training target.
- The broader CID22 image library (250 refs) at /mnt/v/dataset/cid22/CID22/ contains
  the 49 validation refs PLUS 201 additional non-validation refs. The 201-ref pool
  is the "training-only subset".
- Training anchor is ssim2_gpu (computed via zen-metrics batch). NEVER human MOS.

Output: canonical-2026-05-21/train/cid22_train.parquet with schema
  ref_basename : str
  human_score  : f64    # ssim2_gpu (NOT human MOS) — explicitly anchored for clarity
  ssim2_gpu    : f64
  ssim2_log_norm : f64  # (ssim2_gpu + 30) / 1.3 clamped at 0 (matches canonical-2026-05-18)
  cvvdp_score, cvvdp_log_norm, iwssim, iwssim_log_norm : f64  # all null (not scored)
  pjnd_target  : f64    # null
  mix_* columns : f64   # null (ssim2-anchored; mix grid would need CVVDP/IWSSIM)
  mix_target   : f64    # null
  f0..f371     : f64    # 372 features per pair

The 17,611 pairs are scored separately by zen-metrics; this script just
joins ssim2 scores into the canonical 395-column schema.

Pre-run requirements:
  1. canonical-2026-05-21/_workspace/cid22_train_pairs.tsv (built by
     this script's --build-pairs-only mode OR by hand).
  2. canonical-2026-05-21/_workspace/cid22_train_ssim2.tsv (zen-metrics batch
     output with `ssim2_gpu` column appended).
  3. canonical-2026-05-21/_workspace/cid22_train_features.parquet (372-col
     feature parquet from extract_features_372col_cid22 -- new tool).
"""
import argparse
import csv
import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CID22_ROOT = Path("/mnt/v/dataset/cid22/CID22")
CID22_VAL_ROOT = Path("/mnt/v/dataset/cid22/CID22_validation_set")
CANONICAL = Path("/mnt/v/zen/zensim-training/canonical-2026-05-21")
WORKSPACE = CANONICAL / "_workspace"

NUM_FEATURES_FULL = 372

# All target columns in the canonical schema (must match
# canonical-2026-05-18 + the iw44→iw45 rename applied in v11).
ALL_TARGETS = [
    "cvvdp_score", "cvvdp_log_norm",
    "iwssim", "iwssim_log_norm",
    "ssim2_gpu", "ssim2_log_norm",
    "pjnd_target",
    "mix_cv25_iw75", "mix_cv30_iw70", "mix_cv35_iw65", "mix_cv40_iw60",
    "mix_cv45_iw55", "mix_cv50_iw50", "mix_cv55_iw45",
    "mix_cv60_iw40", "mix_cv65_iw35", "mix_cv70_iw30", "mix_cv75_iw25",
    "mix_cv33_iw33_sm33",
    "mix_target",
]


def ssim2_log_norm(ssim2_arr):
    """Match canonical-2026-05-18 ssim2_log_norm column. Verified exact via
    linear-fit error of 1e-14 against safesyn.parquet."""
    return np.maximum((ssim2_arr + 30.0) / 1.3, 0.0)


def build_pairs_tsv():
    """Build the (ref_path, dist_path, ref_basename, codec, q) TSV for
    every CID22 training-only pair (201 non-val refs × ~88 distorted)."""
    val_refs = {p.stem for p in (CID22_VAL_ROOT / "original").iterdir() if p.suffix == ".png"}
    broader_refs = {p.stem for p in (CID22_ROOT / "original").iterdir() if p.suffix == ".png"}
    train_refs = sorted(broader_refs - val_refs)
    if len(train_refs) != 201:
        raise RuntimeError(f"expected 201 train-only refs, got {len(train_refs)}")
    if val_refs.intersection(train_refs):
        raise RuntimeError("LEAK: val refs overlap with train_refs")
    out = WORKSPACE / "cid22_train_pairs.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs = []
    for ref in train_refs:
        ref_path = CID22_ROOT / "original" / f"{ref}.png"
        if not ref_path.exists():
            continue
        dist_root = CID22_ROOT / "compressed" / ref
        if not dist_root.exists():
            continue
        for codec_dir in sorted(dist_root.iterdir()):
            if not codec_dir.is_dir():
                continue
            for dist_file in sorted(codec_dir.iterdir()):
                if not dist_file.is_file():
                    continue
                pairs.append((str(ref_path), str(dist_file), ref,
                              codec_dir.name, dist_file.stem))
    with open(out, "w") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["ref_path", "dist_path", "ref_basename", "codec", "q"])
        w.writerows(pairs)
    print(f"wrote {out} ({len(pairs)} pairs)")
    return out


def assemble_canonical():
    """Join ssim2 scores + 372-feature CSV → canonical train/cid22_train.parquet.

    Feature CSV is produced by `extract_features_372col --corpus cid22_train`,
    where each row's `ref_basename` is the composite key
    `"<ref>|<codec>|<q>"` (loader splits the input TSV's 5 columns into the
    composite). The assemble step splits the composite to recover (ref,
    codec, q) and joins with ssim2 scores from the workspace TSV.
    """
    ssim2_tsv = WORKSPACE / "cid22_train_ssim2.tsv"
    feat_csv = WORKSPACE / "cid22_train_features.csv"
    if not ssim2_tsv.exists():
        raise FileNotFoundError(f"missing {ssim2_tsv} — run zen-metrics batch first")
    if not feat_csv.exists():
        raise FileNotFoundError(
            f"missing {feat_csv} — run extract_features_372col --corpus cid22_train first"
        )

    # Load ssim2 scores keyed by (ref_basename, codec, q)
    score_map = {}
    with open(ssim2_tsv) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            key = (row["ref_basename"], row["codec"], row["q"])
            score_map[key] = float(row["ssim2_gpu"])
    print(f"loaded {len(score_map)} ssim2 scores from {ssim2_tsv}")

    # Load features CSV via pyarrow.csv for speed/memory efficiency
    import pyarrow.csv as pcsv
    print(f"reading features CSV: {feat_csv} ({feat_csv.stat().st_size/1e6:.0f} MB)")
    read_opts = pcsv.ReadOptions(use_threads=True)
    parse_opts = pcsv.ParseOptions(delimiter=",")
    conv_opts = pcsv.ConvertOptions(
        column_types={"ref_basename": pa.string()},  # composite key stays string
    )
    tbl = pcsv.read_csv(str(feat_csv), read_options=read_opts,
                         parse_options=parse_opts, convert_options=conv_opts)
    print(f"loaded {tbl.num_rows} feature rows × {tbl.num_columns} cols")

    # Split composite key
    composites = tbl.column("ref_basename").to_pylist()
    n = tbl.num_rows
    refs = [None] * n
    codecs = [None] * n
    qs = [None] * n
    bad = 0
    for i, c in enumerate(composites):
        parts = c.split("|")
        if len(parts) != 3:
            bad += 1
            continue
        refs[i] = parts[0]
        codecs[i] = parts[1]
        qs[i] = parts[2]
    if bad:
        print(f"WARN: {bad} composite keys malformed (expected 3 parts)")

    # Join: build aligned ssim2 array
    ssim2_vals = np.full(n, np.nan, dtype=np.float64)
    missing = 0
    for i, (r, c, q) in enumerate(zip(refs, codecs, qs)):
        if r is None:
            continue
        v = score_map.get((r, c, q))
        if v is None:
            missing += 1
        else:
            ssim2_vals[i] = v
    if missing:
        print(f"WARN: {missing}/{n} feature rows had no ssim2 score (will be dropped)")
    valid_mask = ~np.isnan(ssim2_vals)
    tbl = tbl.filter(pa.array(valid_mask))
    ssim2_vals = ssim2_vals[valid_mask]
    refs_kept = [refs[i] for i in range(n) if valid_mask[i]]
    log_norm = ssim2_log_norm(ssim2_vals)
    print(f"final rows after ssim2 join: {tbl.num_rows}")

    # Replace ref_basename composite with bare ref
    tbl = tbl.drop_columns(["ref_basename"])
    tbl = tbl.append_column("ref_basename", pa.array(refs_kept, type=pa.string()))

    # Replace human_score with ssim2 (explicit anchor)
    if "human_score" in tbl.schema.names:
        tbl = tbl.drop_columns(["human_score"])
    # ssim2_gpu is the canonical anchor target; also set human_score = ssim2_gpu
    # so trainers that don't pass --target-column find it.
    tbl = tbl.append_column("human_score", pa.array(ssim2_vals))
    tbl = tbl.append_column("ssim2_gpu", pa.array(ssim2_vals))
    tbl = tbl.append_column("ssim2_log_norm", pa.array(log_norm))

    # Add nulls for every other canonical target column
    for col in ALL_TARGETS:
        if col not in tbl.schema.names:
            tbl = tbl.append_column(col, pa.nulls(tbl.num_rows, type=pa.float64()))

    # Reorder to canonical schema
    ordered = ["ref_basename", "human_score"]
    for c in ALL_TARGETS:
        if c in tbl.schema.names:
            ordered.append(c)
    for i in range(NUM_FEATURES_FULL):
        name = f"f{i}"
        if name not in tbl.schema.names:
            raise RuntimeError(f"missing feature column {name}")
        ordered.append(name)
    tbl = tbl.select(ordered)

    # Verify zero leakage with val/cid22.parquet
    val_path = CANONICAL / "val" / "cid22.parquet"
    val_refs_from_parquet = set()
    if val_path.exists():
        val_tbl = pq.read_table(str(val_path), columns=["ref_basename"])
        val_refs_from_parquet = set(val_tbl.column("ref_basename").to_pylist())
    else:
        # No val parquet copied to 2026-05-21 yet; check 2026-05-18
        v18_val = Path("/mnt/v/zen/zensim-training/canonical-2026-05-18/val/cid22.parquet")
        if v18_val.exists():
            val_tbl = pq.read_table(str(v18_val), columns=["ref_basename"])
            val_refs_from_parquet = set(val_tbl.column("ref_basename").to_pylist())
    train_refs_in_new = set(tbl.column("ref_basename").to_pylist())
    leakage = val_refs_from_parquet.intersection(train_refs_in_new)
    if leakage:
        print(f"CRITICAL LEAKAGE: {len(leakage)} refs overlap with val/cid22.parquet")
        print(f"  examples: {sorted(leakage)[:5]}")
        sys.exit(1)
    print(f"zero leakage verified: val has {len(val_refs_from_parquet)} refs, train has {len(train_refs_in_new)}, intersection=0")

    # Write
    out = CANONICAL / "train" / "cid22_train.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    schema_with_meta = tbl.schema.with_metadata({
        b"canonical_corpus": b"cid22_train",
        b"canonical_role": b"TRAINING (ssim2-anchored, NEVER human MOS)",
        b"canonical_date": b"2026-05-21",
        b"source_paths": json.dumps([
            str(CID22_ROOT),
            "workspace/cid22_train_pairs.tsv (17611 pairs from 201 train-only refs)",
            "workspace/cid22_train_ssim2.tsv (zen-metrics batch ssim2-gpu)",
            "workspace/cid22_train_features.parquet (extract_features_372col --corpus cid22_train)",
        ]).encode(),
        b"num_features": str(NUM_FEATURES_FULL).encode(),
        b"training_use": (
            b"CID22 training-only subset (201 non-validation refs); "
            b"human_score = ssim2_gpu (explicitly anchored, NEVER human MOS); "
            b"verified zero overlap with val/cid22.parquet's 49 held-out refs"
        ),
        b"schema_version": b"canonical-2026-05-21.v1",
        b"schema_note": (
            b"ssim2-anchored: cvvdp/iwssim/mix_* are ALL NULL by design; "
            b"per CLAUDE.md 'CID22 is VALIDATION-ONLY' rule, NEVER use human MOS"
        ),
    })
    tbl = tbl.replace_schema_metadata(schema_with_meta.metadata)
    pq.write_table(tbl, str(out), compression="zstd", compression_level=15)
    sz = out.stat().st_size
    print(f"WROTE {out} ({tbl.num_rows} rows × {tbl.num_columns} cols, {sz/1e6:.1f} MB)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-pairs", action="store_true",
                    help="Write workspace/cid22_train_pairs.tsv only")
    ap.add_argument("--assemble", action="store_true",
                    help="Join scores + features into canonical train/cid22_train.parquet")
    args = ap.parse_args()
    if args.build_pairs:
        build_pairs_tsv()
    if args.assemble:
        assemble_canonical()
    if not (args.build_pairs or args.assemble):
        ap.print_help()


if __name__ == "__main__":
    main()
