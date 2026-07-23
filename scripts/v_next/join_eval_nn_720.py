#!/usr/bin/env python3
"""Nearest-variant join: recover an eval cell's 720 from the fleet corpus even
under decoder drift. Exact f0..f371 fingerprint matching hits ~52% because the
eval parquet's 372 was extracted with a different decoder than the fleet's; the
SAME cell's features differ past 1e-6. But within a reference's ~hundreds of
fleet variants, the eval cell's true variant is the NEAREST in 372-space (drift
~1e-3 ≪ inter-variant q-spacing ~1e-2+), so nearest-neighbour within the ref
uniquely identifies it. A distance THRESHOLD keeps it honest: if the nearest
fleet variant is too far (cell genuinely absent), the eval row is DROPPED+flagged,
never matched to a wrong cell.

Output row = fleet's full self-consistent 720 (f0..f719 on the fleet's decode) +
the eval's human_score(ssim2) + the NN distance (audit). Using the fleet's own
372 (not the eval's) keeps 372 and 348 on the SAME pixels.

Usage: python3 join_eval_nn_720.py <corpus_720.parquet> <eval_372.parquet> <out.parquet> [dist_thresh=0.03]
"""
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

THRESH = float(sys.argv[4]) if len(sys.argv) > 4 else 0.03


def stem(x):
    s = str(x).replace(".png", "")
    return s


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
    et = pq.read_table(eval_p)
    ef = fcols(et.schema.names)
    erc = refcol(et.schema.names)
    erefs = [stem(x) for x in et.column(erc).to_pylist()]
    emat = np.column_stack([et.column(c).to_numpy() for c in ef]).astype("f4")
    ehuman = et.column("human_score").to_pylist()
    eval_refset = set(erefs)
    print(f"eval: {et.num_rows} rows, {len(eval_refset)} distinct refs")

    # stream corpus, keep only eval-ref rows -> per-ref (fleet_372 matrix, fleet_720 matrix)
    pf = pq.ParquetFile(corpus_p)
    cf = fcols(pf.schema_arrow.names)
    crc = refcol(pf.schema_arrow.names)
    by_ref = {}  # ref -> [list of (f372 np, f720 np)]
    seen = 0
    for bi in range(pf.num_row_groups):
        rg = pf.read_row_group(bi, columns=[crc] + cf[:720])
        refs = rg.column(crc).to_pylist()
        keepidx = [i for i, r in enumerate(refs) if stem(r) in eval_refset]
        if keepidx:
            allf = np.column_stack([rg.column(c).to_numpy() for c in cf[:720]]).astype("f4")
            for i in keepidx:
                by_ref.setdefault(stem(refs[i]), []).append(allf[i])
        seen += rg.num_rows
        if bi % 20 == 0:
            print(f"  scanned {seen:,}, kept refs={len(by_ref)}")
    # finalize per-ref arrays
    for r in by_ref:
        by_ref[r] = np.vstack(by_ref[r])  # (nvar, 720)
    print(f"collected fleet variants for {len(by_ref)}/{len(eval_refset)} eval refs")

    keep_rows = []
    dists = []
    n_nomatch_ref = 0
    n_toofar = 0
    for i in range(len(erefs)):
        fv = by_ref.get(erefs[i])
        if fv is None:
            n_nomatch_ref += 1
            continue
        d = np.sqrt(((fv[:, :372] - emat[i]) ** 2).sum(axis=1))
        j = int(d.argmin())
        if d[j] > THRESH:
            n_toofar += 1
            continue
        keep_rows.append((i, fv[j]))
        dists.append(float(d[j]))
    n = len(erefs)
    print(f"matched {len(keep_rows)}/{n} ({100*len(keep_rows)/n:.1f}%)  "
          f"[no-ref={n_nomatch_ref}, too-far>{THRESH}={n_toofar}]")
    if dists:
        print(f"  NN dist: median={np.median(dists):.2e} p95={np.percentile(dists,95):.2e} max={max(dists):.2e}")
    if not keep_rows:
        raise SystemExit("0 matched")
    out = {"ref_basename": pa.array([et.column(erc)[i].as_py() for i, _ in keep_rows]),
           "human_score": pa.array([ehuman[i] for i, _ in keep_rows]),
           "nn_dist": pa.array(dists, type=pa.float64())}
    F = np.vstack([v for _, v in keep_rows])
    for k in range(720):
        out[f"f{k}"] = pa.array(F[:, k].astype("f8"))
    pq.write_table(pa.table(out), out_p, compression="zstd")
    print(f"  wrote {len(keep_rows)} x720 -> {out_p}")


if __name__ == "__main__":
    main()
