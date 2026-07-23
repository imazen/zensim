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

    # Build per-ref (nvar x 720) fleet-variant matrices. TWO-PASS + PREALLOCATED:
    # the naive one-pass (append allf[i] views to a list, then vstack) OOMs at the
    # 5.7M-row T-big scale — allf[i] is a VIEW that pins each row-group's full
    # matrix alive, and the finalize vstack transiently doubles the ~16 GB of kept
    # data. Pass 1 counts variants per ref (ref column only, cheap); pass 2 fills
    # preallocated arrays by direct assignment (a copy into the target — no parent
    # pinning, no vstack). Peak ~= the 720 data once (~16 GB) instead of ~2x+.
    pf = pq.ParquetFile(corpus_p)
    cf = fcols(pf.schema_arrow.names)[:720]
    crc = refcol(pf.schema_arrow.names)

    counts = {}
    seen = 0
    for bi in range(pf.num_row_groups):
        refs = pf.read_row_group(bi, columns=[crc]).column(crc).to_pylist()
        for r in refs:
            s = stem(r)
            if s in eval_refset:
                counts[s] = counts.get(s, 0) + 1
        seen += len(refs)
        if bi % 40 == 0:
            print(f"  [pass1] scanned {seen:,}, refs-seen={len(counts)}", flush=True)
    total_var = sum(counts.values())
    print(f"  [pass1] {len(counts)} eval refs present in corpus, {total_var:,} total variants "
          f"(~{total_var * 720 * 4 / 1e9:.1f} GB @ f4)", flush=True)

    by_ref = {r: np.empty((n, 720), dtype="f4") for r, n in counts.items()}
    pos = {r: 0 for r in counts}
    seen = 0
    for bi in range(pf.num_row_groups):
        rg = pf.read_row_group(bi, columns=[crc] + cf)
        refs = rg.column(crc).to_pylist()
        keepidx = [i for i, r in enumerate(refs) if stem(r) in eval_refset]
        if keepidx:
            allf = np.column_stack([rg.column(c).to_numpy() for c in cf]).astype("f4")
            for i in keepidx:
                s = stem(refs[i])
                by_ref[s][pos[s]] = allf[i]   # assignment copies -> parent freeable
                pos[s] += 1
            del allf
        seen += rg.num_rows
        if bi % 40 == 0:
            print(f"  [pass2] scanned {seen:,}/{pf.metadata.num_rows:,}", flush=True)
    print(f"collected fleet variants for {len(by_ref)}/{len(eval_refset)} eval refs", flush=True)

    # Stream output via ParquetWriter (bounded memory — the eval version's
    # collect-all-then-vstack OOMs at the 2.3M-row T-big scale).
    erefs_full = et.column(erc).to_pylist()
    schema = pa.schema([("ref_basename", pa.string()), ("human_score", pa.float64()),
                        ("nn_dist", pa.float64())] + [(f"f{k}", pa.float64()) for k in range(720)])
    writer = pq.ParquetWriter(out_p, schema, compression="zstd")
    b_ref, b_hum, b_d, b_f = [], [], [], []

    def flush():
        nonlocal b_ref, b_hum, b_d, b_f
        if not b_ref:
            return
        F = np.vstack(b_f)
        cols = [pa.array(b_ref), pa.array(b_hum), pa.array(b_d)] + [pa.array(F[:, k].astype("f8")) for k in range(720)]
        writer.write_table(pa.table(cols, schema=schema))
        b_ref, b_hum, b_d, b_f = [], [], [], []

    matched = dists = n_nomatch_ref = n_toofar = 0
    dist_samp = []
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
        b_ref.append(erefs_full[i]); b_hum.append(ehuman[i]); b_d.append(float(d[j])); b_f.append(fv[j])
        matched += 1
        if len(dist_samp) < 200000:
            dist_samp.append(float(d[j]))
        if len(b_ref) >= 50000:
            flush()
    flush()
    writer.close()
    n = len(erefs)
    print(f"matched {matched}/{n} ({100*matched/n:.1f}%)  [no-ref={n_nomatch_ref}, too-far>{THRESH}={n_toofar}]")
    if dist_samp:
        ds = np.array(dist_samp)
        print(f"  NN dist (sample): median={np.median(ds):.2e} p95={np.percentile(ds,95):.2e} max={ds.max():.2e}")
    if matched == 0:
        raise SystemExit("0 matched")
    print(f"  wrote {matched} x720 -> {out_p}")


if __name__ == "__main__":
    main()
