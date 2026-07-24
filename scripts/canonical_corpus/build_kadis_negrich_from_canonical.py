#!/usr/bin/env python3
"""Regenerate the KADIS negative-rich (negrich) hard-negative sample WITH source_id.

This is the builder that `build_kadis_negrich_2026-07-15.py` records as MISSING:
the original `kadis_sample_negrich.parquet` (266k rows) has no builder, no
`build_commit`, and — the sharp one — dropped `source_id`, so its train/val split
cannot be verified leak-free. That column is right there in the KADIS-700k
canonical; this rebuild restores it.

negrich = the NEGATIVE-RICH (severe) subset of the 700k KADIS canonical — the
severe-but-HONEST degradations (heavy blur/noise/motion/color) whose extreme,
corruption-LOOKING features are the hard negatives the corruption detector needs
to learn the corruption-vs-severe-honest boundary. Selection matches the original
sample's coverage: `score_zensim < --threshold` (default 0 → ~280k rows, ≈ the
original 266k; corresponds to KADIS severity levels 4-5, mean score -16/-57).

Output carries `source_id` + full distortion provenance + a `_MANIFEST.json` with
`build_commit`, so a leak-free source-held-out split is finally possible.

Usage:
  build_kadis_negrich_from_canonical.py [--threshold 0] [--nfeat 372] \
      [--out <dir>/kadis_negrich_srcid.parquet]
"""
import argparse, hashlib, json, os, subprocess
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

SRC = "/mnt/v/datasets/kadis700k/canonical/kadis700k_canonical_2026-06-30.parquet"
OUT_DEFAULT = ("/mnt/v/zen/zensim-training/kadis-negrich-regen-2026-07-24/"
               "kadis_negrich_srcid.parquet")
META = ["source_id", "source_filename", "dist_type", "dist_name",
        "severity_level", "dist_param", "score_zensim"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="keep rows with score_zensim < threshold (negative-rich)")
    ap.add_argument("--nfeat", type=int, default=372)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    # source names features feat_0..feat_371; trainer/corpus use f0..f371.
    src_fcols = [f"feat_{i}" for i in range(a.nfeat)]
    dst_fcols = [f"f{i}" for i in range(a.nfeat)]

    pf = pq.ParquetFile(a.src)
    writer = None
    kept = 0
    srcids = set()
    lv_counts = {}
    for batch in pf.iter_batches(batch_size=131072, columns=src_fcols + META):
        z = np.asarray(batch.column("score_zensim").to_numpy(zero_copy_only=False))
        keep = z < a.threshold
        if not keep.any():
            continue
        idx = pa.array(np.where(keep)[0])
        feats = {d: batch.column(s).take(idx) for s, d in zip(src_fcols, dst_fcols)}
        meta = {c: batch.column(c).take(idx) for c in META}
        n = len(idx)
        t = pa.table({**feats, **meta,
                      "is_corruption": pa.array(np.zeros(n, dtype=np.int8)),
                      "neg_subclass": pa.array(["severe_honest"] * n)})
        if writer is None:
            writer = pq.ParquetWriter(a.out, t.schema, compression="zstd")
        writer.write_table(t)
        kept += n
        for s in meta["source_id"].to_pylist():
            srcids.add(s)
        for lv in meta["severity_level"].to_pylist():
            lv_counts[int(lv)] = lv_counts.get(int(lv), 0) + 1
    if writer:
        writer.close()

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    manifest = {
        "artifact": os.path.basename(a.out),
        "purpose": "KADIS negative-rich (severe-honest) hard negatives for the "
                   "corruption detector — leak-free (source_id restored)",
        "source_parquet": a.src,
        "selection": f"score_zensim < {a.threshold}",
        "rows": kept, "nfeat": a.nfeat,
        "unique_source_ids": len(srcids),
        "severity_level_counts": lv_counts,
        "build_commit": commit,
        "fixes": "provenance gap in kadis_sample_negrich.parquet (no source_id) — "
                 "documented in build_kadis_negrich_2026-07-15.py",
    }
    with open(os.path.join(os.path.dirname(a.out), "_MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"DONE: {a.out}\n  {kept} rows, {len(srcids)} unique source_ids, "
          f"severity {dict(sorted(lv_counts.items()))}\n  manifest written (commit {commit[:8]})")


if __name__ == "__main__":
    main()
