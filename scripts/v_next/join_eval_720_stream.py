#!/usr/bin/env python3
"""Exact-fingerprint join of eval-372 gates onto the full fleet 720 index — streaming.

Unlike join_eval_720.py (which fingerprints the huge FLEET side, blowing up memory on
million-row shards), this fingerprints the SMALL eval side (~10k rows) once, then STREAMS
each 720 shard in row-batches, probing every fleet row against the eval fingerprint and
pulling f372..f719 for matches. Peak memory is one row-batch + the 10k-entry index.

Fingerprint key = (ref_stem, round(f0..f371, 6).tobytes()) — same scheme as join_eval_720.py
(exact to ULP for the same v1 extractor + same pixels), just hashed via bytes for speed.
Eval rows sharing a key (duplicate cells) all receive the matched cell's v2 block — correct,
since identical (ref, f0..f371) IS the same cell. Unmatched eval rows are REPORTED + DROPPED,
never fabricated.

  python3 join_eval_720_stream.py '<shard_glob>' <eval_372.parquet> <out_720.parquet>
"""
import sys, glob
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROUND = 6
BATCH = 200_000


def ref_stem(x):
    return str(x).replace(".png", "").split(".scale")[0]


def fcols(names):
    fs = [c for c in names if c.startswith("f") and c[1:].isdigit()]
    if not fs:
        fs = [c for c in names if c.startswith("feat_") and c[5:].isdigit()]
    fs.sort(key=lambda c: int(c.split("_")[-1] if "_" in c else c[1:]))
    return fs


def refcol(names):
    for c in ("ref_basename", "ref_filename", "image_path"):
        if c in names:
            return c
    raise SystemExit(f"no ref column in {names[:8]}")


def key_bytes(refstem, row372):
    # round to 6 decimals in f8, hash the raw bytes prefixed by the ref stem
    return refstem.encode() + b"\x00" + np.round(row372, ROUND).astype("f8").tobytes()


def main():
    shard_glob, eval_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
    shards = sorted(glob.glob(shard_glob)) or [shard_glob]

    # ---- eval side: build fingerprint -> list of eval row indices ----
    et = pq.read_table(eval_p)
    efs = fcols(et.schema.names)
    if len(efs) != 372:
        raise SystemExit(f"eval {eval_p}: {len(efs)} f-cols, expected 372")
    erc = refcol(et.schema.names)
    erefs = [ref_stem(x) for x in et.column(erc).to_pylist()]
    emat = np.column_stack([et.column(c).to_numpy() for c in efs]).astype("f8")
    ehuman = et.column("human_score").to_pylist() if "human_score" in et.schema.names else [None] * len(erefs)
    eorig = et.column(erc).to_pylist()

    fp = {}
    for i in range(len(erefs)):
        fp.setdefault(key_bytes(erefs[i], emat[i]), []).append(i)
    n_eval = len(erefs)
    n_keys = len(fp)
    print(f"eval {eval_p}: {n_eval} rows, {n_keys} distinct fingerprints "
          f"(ceiling {n_eval} rows via key-fill)", flush=True)

    matched = {}   # eval_row_idx -> f372..f719 (list of 348)
    ncols_v2 = None

    # ---- fleet side: stream shards, probe ----
    for si, sh in enumerate(shards):
        pf = pq.ParquetFile(sh)
        sfs = fcols(pf.schema_arrow.names)
        if len(sfs) < 720:
            print(f"  WARN {sh}: {len(sfs)} f-cols (<720) — skip", flush=True)
            continue
        v1 = sfs[:372]
        v2 = sfs[372:720]
        if ncols_v2 is None:
            ncols_v2 = len(v2)
        src = refcol(pf.schema_arrow.names)
        got0 = len(matched)
        for batch in pf.iter_batches(batch_size=BATCH, columns=[src] + sfs):
            d = batch.to_pydict()
            refs = [ref_stem(x) for x in d[src]]
            m1 = np.column_stack([np.asarray(d[c], dtype="f8") for c in v1])
            m1r = np.round(m1, ROUND)
            for j in range(len(refs)):
                if len(matched) >= n_eval:
                    break
                k = refs[j].encode() + b"\x00" + m1r[j].tobytes()
                idxs = fp.get(k)
                if not idxs:
                    continue
                fresh = [ix for ix in idxs if ix not in matched]
                if not fresh:
                    continue
                v2vals = [float(d[c][j]) for c in v2]
                for ix in fresh:
                    matched[ix] = v2vals
            if len(matched) >= n_eval:
                break
        print(f"  [{si+1}/{len(shards)}] {sh.split('/')[-1]}: "
              f"+{len(matched)-got0} matched (total {len(matched)}/{n_eval})", flush=True)
        if len(matched) >= n_eval:
            print("  all eval rows matched — early exit", flush=True)
            break

    hit = len(matched)
    print(f"{eval_p}: matched {hit}/{n_eval} ({100*hit/n_eval:.2f}%)", flush=True)
    if hit < n_eval:
        miss = [erefs[i] for i in range(n_eval) if i not in matched][:10]
        print(f"  UNMATCHED (dropped, first 10 stems): {miss}", flush=True)
    if hit == 0:
        raise SystemExit("0 matches — check shard coverage / fingerprint scheme")

    keep = sorted(matched)
    cols = {
        "ref_basename": pa.array([eorig[i] for i in keep]),
        "human_score": pa.array([ehuman[i] for i in keep]),
    }
    for k in range(372):
        cols[f"f{k}"] = pa.array([float(emat[i][k]) for i in keep], type=pa.float64())
    for k in range(ncols_v2):
        cols[f"f{372+k}"] = pa.array([matched[i][k] for i in keep], type=pa.float64())
    pq.write_table(pa.table(cols), out_p, compression="zstd")
    print(f"  wrote {len(keep)} rows x {372+ncols_v2} features -> {out_p}", flush=True)


if __name__ == "__main__":
    main()
