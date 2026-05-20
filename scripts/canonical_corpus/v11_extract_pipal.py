#!/usr/bin/env python3
"""V11 Phase 2 step 5: assemble PIPAL feature parquet from extracted features + ELO MOS labels.

PIPAL has 23,200 training pairs (200 refs × 116 distortions). Per-pair
ELO MOS labels come from Train_Label/A####.txt files (paired with
distorted basename). PIPAL human labels ARE OK to use as training target
(not in CID22's sacred rule scope — PIPAL is an independent IQA corpus).

Inputs (workspace dir):
  - pipal_pairs.tsv     (built by tmp/build_pipal_pairs.py)
  - pipal_elo.tsv       (built alongside, ref_basename + codec + q + elo)
  - pipal_features.csv  (output of extract_features_372col --corpus cid22_train
                         using PIPAL pairs TSV; ref_basename = 'A####|DD|LL')

Output: canonical-2026-05-21/train/pipal.parquet with canonical schema:
  ref_basename : str (bare A#### like 'A0001')
  human_score  : f64 (= elo, normalized to [0, 1] via min-max over corpus)
  pipal_elo    : f64 (raw ELO score for clarity in audit)
  ssim2_*, cvvdp_*, iwssim_*, pjnd_target, mix_* : NULL
  f0..f371     : 372 features
"""
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

CANONICAL = Path("/mnt/v/zen/zensim-training/canonical-2026-05-21")
WORKSPACE = CANONICAL / "_workspace"

NUM_FEATURES_FULL = 372

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
    # PIPAL-specific (NOT a canonical-2026-05-18 col; added for v11)
    "pipal_elo",
]


def main():
    elo_tsv = WORKSPACE / "pipal_elo.tsv"
    feat_csv = WORKSPACE / "pipal_features.csv"
    if not elo_tsv.exists():
        raise FileNotFoundError(elo_tsv)
    if not feat_csv.exists():
        raise FileNotFoundError(feat_csv)

    # ELO map
    elo_map = {}
    with open(elo_tsv) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            elo_map[(row["ref_basename"], row["codec"], row["q"])] = float(row["elo"])
    print(f"loaded {len(elo_map)} ELO scores")

    # Load features
    print(f"reading {feat_csv} ({feat_csv.stat().st_size/1e6:.0f} MB)")
    conv = pcsv.ConvertOptions(column_types={"ref_basename": pa.string()})
    tbl = pcsv.read_csv(str(feat_csv), convert_options=conv)
    print(f"loaded {tbl.num_rows} feature rows × {tbl.num_columns} cols")

    # Split composite key
    composites = tbl.column("ref_basename").to_pylist()
    n = tbl.num_rows
    refs = [None] * n
    elos = np.full(n, np.nan, dtype=np.float64)
    missing = 0
    for i, c in enumerate(composites):
        parts = c.split("|")
        if len(parts) != 3:
            continue
        ref, codec, q = parts
        e = elo_map.get((ref, codec, q))
        refs[i] = ref
        if e is None:
            missing += 1
        else:
            elos[i] = e
    if missing:
        print(f"WARN: {missing}/{n} feature rows had no ELO label")
    valid = ~np.isnan(elos)
    tbl = tbl.filter(pa.array(valid))
    elos = elos[valid]
    refs_kept = [refs[i] for i in range(n) if valid[i]]
    print(f"final rows after ELO join: {tbl.num_rows}")

    # ELO range
    e_min, e_max = float(elos.min()), float(elos.max())
    print(f"ELO range: [{e_min:.2f}, {e_max:.2f}]")
    # Normalize to [0, 1] for human_score (preserve raw in pipal_elo)
    # ELO is rank-like; PIPAL paper has e_min ≈ 1100, e_max ≈ 1700.
    elo_norm = (elos - e_min) / (e_max - e_min)

    tbl = tbl.drop_columns(["ref_basename"])
    tbl = tbl.append_column("ref_basename", pa.array(refs_kept, type=pa.string()))
    if "human_score" in tbl.schema.names:
        tbl = tbl.drop_columns(["human_score"])
    tbl = tbl.append_column("human_score", pa.array(elo_norm))
    tbl = tbl.append_column("pipal_elo", pa.array(elos))

    for col in ALL_TARGETS:
        if col not in tbl.schema.names:
            tbl = tbl.append_column(col, pa.nulls(tbl.num_rows, type=pa.float64()))

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

    out = CANONICAL / "train" / "pipal.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    schema_with_meta = tbl.schema.with_metadata({
        b"canonical_corpus": b"pipal",
        b"canonical_role": b"TRAINING (PIPAL ELO MOS, OK per CLAUDE.md scope)",
        b"canonical_date": b"2026-05-21",
        b"source_paths": json.dumps([
            "/mnt/v/dataset/pipal/",
            "workspace/pipal_pairs.tsv (23,200 pairs, 200 refs × 116 distortions)",
            "workspace/pipal_elo.tsv (per-pair ELO from Train_Label/A####.txt)",
            "workspace/pipal_features.csv (extract_features_372col on BMP)",
        ]).encode(),
        b"num_features": str(NUM_FEATURES_FULL).encode(),
        b"training_use": (
            f"PIPAL 23,200-pair ELO MOS corpus; human_score = (elo - {e_min:.2f}) "
            f"/ ({e_max - e_min:.2f}) in [0,1]; raw ELO preserved as pipal_elo. "
            f"ELO is rank-based - train weight should be moderate (not dominant) "
            f"to avoid disrupting ssim2-shaped target columns from other corpora."
        ).encode(),
        b"schema_version": b"canonical-2026-05-21.v1",
        b"schema_note": (
            b"PIPAL is INDEPENDENT of CID22 sacred rule. Human MOS is OK to train. "
            b"ssim2/cvvdp/iwssim/mix_* are ALL NULL by design."
        ),
    })
    tbl = tbl.replace_schema_metadata(schema_with_meta.metadata)
    pq.write_table(tbl, str(out), compression="zstd", compression_level=15)
    sz = out.stat().st_size
    print(f"WROTE {out} ({tbl.num_rows} rows × {tbl.num_columns} cols, {sz/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
