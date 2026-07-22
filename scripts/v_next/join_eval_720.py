#!/usr/bin/env python3
"""Backfill 720 features for the nonphoto / imazen26 ssim2 eval gates by JOINING
their existing 372-feature parquets to the fleet's bigcodec 720 output.

WHY a join (not re-extraction): nonphoto/imazen26 are content-filtered subsets of the
bigcodec (canonical-picker) validate/test CELLS the zensim720 fleet is extracting 720 for
(verified 2026-07-22: eval `ref_basename` == canonical `ref_filename`, 100% overlap). The
eval parquets carry only (ref_basename, human_score=ssim2, f0..f371); the fleet output
carries f0..f719 for the same cells. The 372 block exactly fingerprints a cell, so we match
on (ref_basename, f0..f371) and append f372..f719 — no re-encode, no pixel reconstruction.

Match is EXACT on the rounded 372 block (same v1 extractor + same pixels ⇒ identical bytes
to ULP). Any eval row with no fleet match is REPORTED and DROPPED, never fabricated.

Usage:
  python3 scripts/v_next/join_eval_720.py <fleet_720_parquet_or_glob> <eval_372_parquet> <out_720_parquet>

The fleet 720 parquet is the merged blobs from s3://zentrain/jobs/bf-*/blobs/ (pull + merge
once the fleet run completes). Columns expected: ref_basename (or ref_filename) + f0..f719
(or feat_0..). Run per eval gate (nonphoto, imazen26).
"""
import sys, glob
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROUND = 6  # decimals for the fingerprint match on f0..f371


def fcols(schema, n_expect):
    fs = [c for c in schema.names if (c.startswith("f") and c[1:].isdigit())]
    if not fs:
        fs = [c for c in schema.names if (c.startswith("feat_") and c[5:].isdigit())]
    fs.sort(key=lambda c: int(c.split("_")[-1] if "_" in c else c[1:]))
    return fs


def refcol(schema):
    for c in ("ref_basename", "ref_filename", "image_path"):
        if c in schema.names:
            return c
    raise SystemExit(f"no ref column in {schema.names[:8]}")


def load(path, want_feats):
    t = pq.read_table(path)
    fs = fcols(t.schema, want_feats)
    rc = refcol(t.schema)
    refs = [str(x).replace(".png", "").split(".scale")[0] for x in t.column(rc).to_pylist()]
    mat = np.column_stack([t.column(c).to_numpy() for c in fs]).astype("f8")
    return t, refs, fs, mat


def fingerprint(refs, mat372):
    # (ref_stem, rounded f0..f371) -> row index
    fp = {}
    for i, r in enumerate(refs):
        key = (r, tuple(np.round(mat372[i], ROUND)))
        fp.setdefault(key, i)
    return fp


def main():
    fleet_glob, eval_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
    fleet_files = sorted(glob.glob(fleet_glob)) or [fleet_glob]
    # eval side: 372 features
    et, erefs, efs, emat = load(eval_p, 372)
    if len(efs) != 372:
        raise SystemExit(f"eval parquet has {len(efs)} f-cols, expected 372")
    ehuman = et.column("human_score").to_pylist()

    # fleet side: build fingerprint of the 372 block -> full 720 row, across all shards
    matched = {}  # eval-row-idx -> (f372..f719 list)
    for ff in fleet_files:
        ft, frefs, ffs, fmat = load(ff, 720)
        if len(ffs) < 720:
            print(f"  WARN {ff}: only {len(ffs)} f-cols (<720) — skipping v2 block")
            continue
        fp = fingerprint(frefs, fmat[:, :372])
        for i in range(len(erefs)):
            if i in matched:
                continue
            key = (erefs[i], tuple(np.round(emat[i], ROUND)))
            j = fp.get(key)
            if j is not None:
                matched[i] = fmat[j, 372:720].tolist()

    n = len(erefs)
    hit = len(matched)
    print(f"{eval_p}: matched {hit}/{n} ({100*hit/n:.1f}%) to fleet 720 output")
    if hit < n:
        miss = [erefs[i] for i in range(n) if i not in matched][:10]
        print(f"  UNMATCHED (dropped, first 10 ref stems): {miss}")
    if hit == 0:
        raise SystemExit("0 matches — is the fleet run complete + does it cover these cells?")

    # assemble the 720 output for matched rows only
    keep = sorted(matched)
    cols = {
        "ref_basename": pa.array([et.column(refcol(et.schema))[i].as_py() for i in keep]),
        "human_score": pa.array([ehuman[i] for i in keep]),
    }
    for k in range(372):
        cols[f"f{k}"] = pa.array([emat[i][k] for i in keep], type=pa.float64())
    for k in range(348):
        cols[f"f{372+k}"] = pa.array([matched[i][k] for i in keep], type=pa.float64())
    pq.write_table(pa.table(cols), out_p, compression="zstd")
    print(f"  wrote {len(keep)} rows x 720 features -> {out_p}")


if __name__ == "__main__":
    main()
