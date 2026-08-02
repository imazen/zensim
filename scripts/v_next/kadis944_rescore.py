#!/usr/bin/env python3
"""KADIS-700k -> 944 features (folded+append+append2 STREAMING regime,
ZENSIM_AB_MODE=foldapp2) via RESCORE-FROM-LINKS — the 944 adaptation of
kadis924_rescore.py (PLAN_SOTA944 P1, task #9). NEVER regenerate with
kadis-distort (RNG diverges); download the persisted distorted PNGs the
metric scores were computed on.

Differences from the 924 script (single-box shape instead of 24 fleet slots):
  - META = kadis700k_924.parquet itself (distorted_url only is read here;
    metadata/score columns are carried verbatim from the 924 file at merge
    time by merge_kadis944.py, so the gate compares like-for-like).
  - LARGE extractor batches (BATCH_REFS refs -> one pairs.tsv -> ONE
    v2_ab_extract invocation; rayon parallelizes across ref groups) instead
    of one invocation per ref — the per-ref shape needs a fleet to be fast.
  - s5cmd batch downloads (measured 442 obj/s vs boto3-thread 32 obj/s on
    these ~250 KiB objects) + 1-ahead prefetch: batch k+1 downloads while
    batch k extracts, so the run is extraction-bound.
Output rows: distorted_url + f0..f943 (f32) — features only.

Env: ZM944_BIN (required), R2_ENDPOINT + AWS creds (r2-env.sh), CHUNK_I/
CHUNK_N (optional hash-slot split, default 0/1), BATCH_REFS (default 400),
S5_WORKERS (default 128), KADIS944_OUT dir (default ~/tmp/backfill944/kadis944).
Resumable: skips distorted_urls already present in the output chunk parquet.
"""
import csv as _csv
import hashlib
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

META = os.environ.get(
    "KADIS944_META", "/mnt/v/zen/zensim-training/kadis-924-2026-07-27/kadis700k_924.parquet"
)
EXTRACT = os.environ.get("ZM944_BIN") or sys.exit("ABORT: ZM944_BIN env required")
CHUNK_I = int(os.environ.get("CHUNK_I", "0"))
CHUNK_N = int(os.environ.get("CHUNK_N", "1"))
BATCH_REFS = int(os.environ.get("BATCH_REFS", "400"))
S5_WORKERS = int(os.environ.get("S5_WORKERS", "128"))
OUTDIR = os.path.expanduser(os.environ.get("KADIS944_OUT", "~/tmp/backfill944/kadis944"))
OUT = f"{OUTDIR}/out/kadis944_c{CHUNK_I:02d}.parquet"
REF_BUCKET, REF_PREFIX = "zentrain", "kadis-700k/refs/"
LOCAL_REFS = "/mnt/v/datasets/kadis700k/refs"  # local mirror; fall back to R2
FEATCOLS = [f"f{i}" for i in range(944)]
S5CMD = shutil.which("s5cmd") or sys.exit("ABORT: s5cmd not on PATH")
os.environ.setdefault(
    "R2_ENDPOINT", f"https://{os.environ.get('R2_ACCOUNT_ID', '')}.r2.cloudflarestorage.com"
)


def main():
    m = pq.read_table(META, columns=["source_filename", "distorted_url"]).to_pydict()
    by = defaultdict(list)
    for i in range(len(m["source_filename"])):
        by[m["source_filename"][i]].append(i)
    refs = [
        r
        for r in sorted(by)
        if int(hashlib.md5(r.encode()).hexdigest()[:8], 16) % CHUNK_N == CHUNK_I
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        try:
            done = set(
                pq.read_table(OUT, columns=["distorted_url"]).column("distorted_url").to_pylist()
            )
        except Exception:
            pass
    batches = [refs[i : i + BATCH_REFS] for i in range(0, len(refs), BATCH_REFS)]
    print(
        f"kadis944 rescore c{CHUNK_I:02d}/{CHUNK_N}: {len(refs)} refs, "
        f"{len(batches)} batches, {len(done)} cells already done",
        flush=True,
    )

    pool = ThreadPoolExecutor(max_workers=2)
    writer = [None]

    def dl_batch(bi):
        """Download refs+dists for batch bi into OUTDIR/b<bi>; return job list."""
        bdir = f"{OUTDIR}/b{bi}"
        shutil.rmtree(bdir, ignore_errors=True)
        os.makedirs(bdir)
        todo = []  # (ref_path, meta_row_index, dist_path)
        cmds = []
        for fn in batches[bi]:
            idxs = [i for i in by[fn] if m["distorted_url"][i] not in done]
            if not idxs:
                continue
            rp = os.path.join(LOCAL_REFS, fn)
            if not os.path.exists(rp):
                rp = os.path.join(bdir, "r_" + fn.replace("/", "_"))
                cmds.append(f"cp s3://{REF_BUCKET}/{REF_PREFIX}{fn} {rp}\n")
            for i in idxs:
                dp = os.path.join(bdir, f"d{len(todo)}.png")
                cmds.append(f"cp {m['distorted_url'][i]} {dp}\n")
                todo.append((rp, i, dp))
        if not todo:
            return bdir, []
        cmdf = os.path.join(bdir, "s5.txt")
        with open(cmdf, "w") as f:
            f.writelines(cmds)
        r = subprocess.run(
            [S5CMD, "--endpoint-url", os.environ["R2_ENDPOINT"],
             "--numworkers", str(S5_WORKERS), "run", cmdf],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"  s5cmd batch {bi} rc={r.returncode} (missing files skipped): "
                  f"{r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ''}",
                  flush=True)
        jobs = [
            (rp, dp, i)
            for (rp, i, dp) in todo
            if os.path.exists(dp) and os.path.exists(rp)
        ]
        if len(jobs) < len(todo):
            print(f"  batch {bi}: {len(todo) - len(jobs)} cells failed to download",
                  flush=True)
        return bdir, jobs

    def flush(buf_f, buf_u):
        if not buf_u:
            return
        t = pa.table(
            {
                "distorted_url": pa.array(buf_u),
                **{c: pa.array(buf_f[c], pa.float32()) for c in FEATCOLS},
            }
        )
        if writer[0] is None:
            writer[0] = pq.ParquetWriter(OUT, t.schema, compression="zstd")
        writer[0].write_table(t)

    processed = 0
    nxt = pool.submit(dl_batch, 0) if batches else None
    for bi in range(len(batches)):
        bdir, jobs = nxt.result()
        nxt = pool.submit(dl_batch, bi + 1) if bi + 1 < len(batches) else None
        if not jobs:
            shutil.rmtree(bdir, ignore_errors=True)
            continue
        pairs = os.path.join(bdir, "pairs.tsv")
        with open(pairs, "w") as f:
            f.write("ref_path\tdist_path\thuman_score\n")
            for j, (rp, dp, _i) in enumerate(jobs):
                f.write(f"{rp}\t{dp}\t{j}\n")
        fcsv = os.path.join(bdir, "f.csv")
        r = subprocess.run(
            [EXTRACT, pairs, fcsv],
            capture_output=True,
            text=True,
            env={**os.environ, "ZENSIM_AB_MODE": "foldapp2"},
        )
        if r.returncode != 0 or not os.path.exists(fcsv):
            print(f"  SKIP batch {bi}: extract ({r.stderr.strip()[-120:]})", flush=True)
            shutil.rmtree(bdir, ignore_errors=True)
            continue
        buf_f = {c: [] for c in FEATCOLS}
        buf_u = []
        with open(fcsv, newline="") as fh:
            rd = _csv.reader(fh)
            hdr = next(rd)
            fi = [hdr.index(c) for c in FEATCOLS]
            hj = hdr.index("human_score")
            for row in rd:
                j = int(float(row[hj]))
                v = np.array([float(row[x]) for x in fi], np.float32)
                if not np.all(np.isfinite(v)):
                    continue
                _rp, _dp, i = jobs[j]
                for x, c in enumerate(FEATCOLS):
                    buf_f[c].append(float(v[x]))
                buf_u.append(m["distorted_url"][i])
                processed += 1
        flush(buf_f, buf_u)
        shutil.rmtree(bdir, ignore_errors=True)
        print(f"[batch {bi + 1}/{len(batches)}] {processed} cells written", flush=True)
    if writer[0]:
        writer[0].close()
    print(f"DONE: {OUT}  {processed} cells", flush=True)


if __name__ == "__main__":
    main()
