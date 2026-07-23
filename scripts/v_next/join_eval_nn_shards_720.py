#!/usr/bin/env python3
"""Nearest-variant 720 join over the FULL 54-shard fleet index (drift-robust).

Same principle as join_eval_nn_720.py, generalized from one corpus file to the whole
sharded 720 index, and grouped by ORIGIN stem (what merge_fleet_720.py stored in
ref_basename — o_NNNN, scale already dropped). Exact f0..f371 fingerprinting only reaches
~53% because the eval gates' 372 came from a different (zensim-GPU) extractor than the
fleet's (CPU v2_ab) — the SAME cell's features drift past 1e-6. But within one origin's
fleet variants the eval cell's true variant is the NEAREST in 372-space (drift ~1e-3 ≪
inter-variant spacing), so NN within the origin + a distance threshold recovers it without
matching a wrong cell (too-far rows are DROPPED + flagged, never fabricated).

Output row = the fleet variant's own full 720 (f0..f719, self-consistent on the fleet's
pixels) + the eval's human_score (ssim2 target) + nn_dist (audit).

Usage: python3 join_eval_nn_shards_720.py '<shard_glob>' <eval_372.parquet> <out.parquet> [thresh=0.03]
"""
import sys, glob
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

THRESH = float(sys.argv[4]) if len(sys.argv) > 4 else 0.03


def origin(x):
    # match merge_fleet_720.py's ref_stem: drop .png, drop .scale suffix -> o_NNNN
    return str(x).replace(".png", "").split(".scale")[0]


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
    shard_glob, eval_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
    shards = sorted(glob.glob(shard_glob)) or [shard_glob]

    et = pq.read_table(eval_p)
    ef = fcols(et.schema.names)
    erc = refcol(et.schema.names)
    erefs = [origin(x) for x in et.column(erc).to_pylist()]
    emat = np.column_stack([et.column(c).to_numpy() for c in ef]).astype("f4")
    ehuman = et.column("human_score").to_pylist()
    eval_refset = set(erefs)
    print(f"eval: {et.num_rows} rows, {len(eval_refset)} distinct origins", flush=True)

    # stream all shards, keep only rows whose origin is in the eval set
    by_ref = {}  # origin -> list of 720-vecs
    for si, sh in enumerate(shards):
        pf = pq.ParquetFile(sh)
        cf = fcols(pf.schema_arrow.names)[:720]
        crc = refcol(pf.schema_arrow.names)
        kept = 0
        for bi in range(pf.num_row_groups):
            rg = pf.read_row_group(bi, columns=[crc] + cf)
            refs = [origin(r) for r in rg.column(crc).to_pylist()]
            keepidx = [i for i, r in enumerate(refs) if r in eval_refset]
            if keepidx:
                allf = np.column_stack([rg.column(c).to_numpy() for c in cf]).astype("f4")
                for i in keepidx:
                    by_ref.setdefault(refs[i], []).append(allf[i])
                kept += len(keepidx)
        print(f"  [{si+1}/{len(shards)}] {sh.split('/')[-1]}: kept {kept} "
              f"(origins so far {len(by_ref)}/{len(eval_refset)})", flush=True)
    for r in list(by_ref):
        by_ref[r] = np.vstack(by_ref[r])
    tot_var = sum(v.shape[0] for v in by_ref.values())
    print(f"collected {tot_var:,} fleet variants for {len(by_ref)}/{len(eval_refset)} origins", flush=True)

    keep_rows, dists = [], []
    n_noref = n_toofar = 0
    for i in range(len(erefs)):
        fv = by_ref.get(erefs[i])
        if fv is None:
            n_noref += 1
            continue
        d = np.sqrt(((fv[:, :372] - emat[i]) ** 2).sum(axis=1))
        j = int(d.argmin())
        if d[j] > THRESH:
            n_toofar += 1
            continue
        keep_rows.append((i, fv[j]))
        dists.append(float(d[j]))
    n = len(erefs)
    print(f"matched {len(keep_rows)}/{n} ({100*len(keep_rows)/n:.2f}%)  "
          f"[no-origin={n_noref}, too-far>{THRESH}={n_toofar}]", flush=True)
    if dists:
        print(f"  NN dist: median={np.median(dists):.2e} p95={np.percentile(dists,95):.2e} "
              f"max={max(dists):.2e}", flush=True)
    if not keep_rows:
        raise SystemExit("0 matched")
    out = {"ref_basename": pa.array([et.column(erc)[i].as_py() for i, _ in keep_rows]),
           "human_score": pa.array([ehuman[i] for i, _ in keep_rows]),
           "nn_dist": pa.array(dists, type=pa.float64())}
    F = np.vstack([v for _, v in keep_rows])
    for k in range(720):
        out[f"f{k}"] = pa.array(F[:, k].astype("f8"))
    pq.write_table(pa.table(out), out_p, compression="zstd")
    print(f"  wrote {len(keep_rows)} x720 -> {out_p}", flush=True)


if __name__ == "__main__":
    main()
