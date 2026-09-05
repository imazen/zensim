#!/usr/bin/env python3
"""Re-extract the KADIS negrich severe-honest negatives at the CURRENT era.

negrich is the hard-negative half of the corruption head: severe-but-HONEST KADIS
degradations (heavy blur / noise / motion / colour) that look corruption-like in
feature space. `corruption_head_2026-07-24.md` measured it as load-bearing — the
no-negrich head false-positives on **82 %** of severe-honest content — and that
head's manifest says outright that shipping needs negrich "at the current era".

The 2026-07-24 table (`kadis_negrich_srcid.parquet`) is pre-option-C, so its
features are not the ones the runtime produces. The distortions themselves are NOT
regenerable — `project_kadis720_rescore_from_links` records that a kadis-distort
re-run diverges (mean |Δ| 9.8) — so the ONLY correct route is rescore-from-links:
fetch the persisted distorted PNGs and re-extract.

Two gotchas, both measured here:

  * `distorted_url` lives on the **GPU** canonical (2026-07-01), not the
    zensim-only one (2026-06-30) the negrich subset was cut from. They join
    exactly on `(source_id, dist_type, severity_level)` — 280,384 / 280,384 hits,
    0 misses.
  * Every distorted basename embeds `..._zenjpeg_q<N>_...` REGARDLESS of the
    actual distortion — it is a chunk-pipeline naming artifact, not the content.
    `dist_name` is the truth (24 KADIS types; `color_saturate_hsv` is the mode).
    Do not conclude from the filename that these rows are codec output.
  * The PNGs are on R2 only. The LAN store carries `kadis-700k-gpu/canonical/`
    and no `distorted/` prefix, so `ZEN_S3_ENDPOINT` will 404 on every one.

Usage:
  build_negrich_postc.py --pairs pairs.tsv --meta meta.tsv --feats feats.csv \
      --out negrich_372_postC.parquet
"""
import argparse, csv, json, os
import numpy as np, pyarrow as pa, pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True,
                    help="idx/source_id/dist_name/severity_level/score_zensim TSV")
    ap.add_argument("--feats", required=True,
                    help="extract_features_372col CSV (human_score carries idx)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nfeat", type=int, default=372)
    a = ap.parse_args()
    cols = [f"f{i}" for i in range(a.nfeat)]

    meta = {}
    with open(a.meta) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            meta[int(r["idx"])] = r

    feats, order = {}, []
    with open(a.feats) as f:
        for r in csv.DictReader(f):
            idx = int(float(r["human_score"]))
            v = np.array([float(r[c]) for c in cols], dtype=np.float32)
            if not np.all(np.isfinite(v)):
                continue
            feats[idx] = v
            order.append(idx)
    order = sorted(set(order) & set(meta))
    X = np.stack([feats[i] for i in order])
    print(f"rows {len(order)} of {len(meta)} meta / {len(feats)} extracted")

    t = pa.table({
        **{c: pa.array(X[:, j], pa.float32()) for j, c in enumerate(cols)},
        "source_id": pa.array([int(meta[i]["source_id"]) for i in order]),
        "dist_name": pa.array([meta[i]["dist_name"] for i in order]),
        "severity_level": pa.array([int(meta[i]["severity_level"]) for i in order]),
        "score_zensim_stored": pa.array([float(meta[i]["score_zensim"]) for i in order]),
        "is_corruption": pa.array([0] * len(order)),
        "neg_subclass": pa.array(["severe_honest"] * len(order)),
    })
    pq.write_table(t, a.out, compression="zstd")
    print(f"wrote {a.out}  {t.num_rows} rows x {t.num_columns} cols")


if __name__ == "__main__":
    main()
