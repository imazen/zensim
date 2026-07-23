#!/usr/bin/env python3
"""Memory-safe join: append f372..f719 to an eval 372-parquet from the full
fleet 720 corpus (e.g. tbig_720_full.parquet, 5.7M rows / 29 GB).

The eval set is small (~10k); the corpus is huge. So we fingerprint the EVAL
side (small dict) and STREAM the corpus in row-group batches, matching each
corpus row's (ref_stem, rounded f0..f371) against the eval fingerprints. Never
materializes the corpus in memory. Unmatched eval rows are DROPPED + reported,
never fabricated.

Usage: python3 join_eval_to_corpus_720.py <corpus_720.parquet> <eval_372.parquet> <out_720.parquet>
"""
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROUND = 6


def stem(x):
    return str(x).replace(".png", "").split(".scale")[0] + (
        ".scale" + str(x).split(".scale")[1].replace(".png", "") if ".scale" in str(x) else ""
    )


def fcols(names):
    fs = [c for c in names if c.startswith("f") and c[1:].isdigit()]
    if not fs:
        fs = [c for c in names if c.startswith("feat_") and c.split("_")[-1].isdigit()]
    return sorted(fs, key=lambda c: int(c.split("_")[-1] if "_" in c else c[1:]))


def refcol(names):
    for c in ("ref_basename", "ref_filename", "image_path"):
        if c in names:
            return c
    raise SystemExit(f"no ref col in {names[:8]}")


def main():
    corpus_p, eval_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]

    # --- eval side: fingerprint on (ref_stem, rounded f0..f371) -> eval idx ---
    et = pq.read_table(eval_p)
    ef = fcols(et.schema.names)
    if len(ef) != 372:
        raise SystemExit(f"eval has {len(ef)} f-cols, expected 372")
    erc = refcol(et.schema.names)
    erefs = [stem(x) for x in et.column(erc).to_pylist()]
    emat = np.column_stack([et.column(c).to_numpy() for c in ef]).astype("f8")
    ehuman = et.column("human_score").to_pylist() if "human_score" in et.schema.names else [None] * et.num_rows
    fp = {}
    for i in range(len(erefs)):
        fp.setdefault((erefs[i], tuple(np.round(emat[i], ROUND))), i)
    print(f"eval: {et.num_rows} rows, {len(fp)} distinct fingerprints")

    # --- stream corpus, match, collect f372..f719 for matched eval rows ---
    pf = pq.ParquetFile(corpus_p)
    cf = fcols(pf.schema_arrow.names)
    if len(cf) < 720:
        raise SystemExit(f"corpus has {len(cf)} f-cols (<720)")
    crc = refcol(pf.schema_arrow.names)
    cols_needed = [crc] + cf[:372] + cf[372:720]
    matched = {}  # eval idx -> f372..f719 list
    seen = 0
    for bi in range(pf.num_row_groups):
        rg = pf.read_row_group(bi, columns=cols_needed)
        refs = [stem(x) for x in rg.column(crc).to_pylist()]
        m372 = np.column_stack([rg.column(c).to_numpy() for c in cf[:372]]).astype("f8")
        v2 = np.column_stack([rg.column(c).to_numpy() for c in cf[372:720]]).astype("f8")
        for r in range(len(refs)):
            key = (refs[r], tuple(np.round(m372[r], ROUND)))
            j = fp.get(key)
            if j is not None and j not in matched:
                matched[j] = v2[r].tolist()
        seen += rg.num_rows
        if bi % 20 == 0:
            print(f"  scanned {seen:,} corpus rows, matched {len(matched)}/{et.num_rows}")
        if len(matched) == et.num_rows:
            break

    n = et.num_rows
    print(f"{eval_p}: matched {len(matched)}/{n} ({100*len(matched)/n:.1f}%)")
    if not matched:
        raise SystemExit("0 matches")
    keep = sorted(matched)
    out = {"ref_basename": pa.array([et.column(erc)[i].as_py() for i in keep]),
           "human_score": pa.array([ehuman[i] for i in keep])}
    for k in range(372):
        out[f"f{k}"] = pa.array([emat[i][k] for i in keep], type=pa.float64())
    for k in range(348):
        out[f"f{372+k}"] = pa.array([matched[i][k] for i in keep], type=pa.float64())
    pq.write_table(pa.table(out), out_p, compression="zstd")
    print(f"  wrote {len(keep)} x720 -> {out_p}")


if __name__ == "__main__":
    main()
