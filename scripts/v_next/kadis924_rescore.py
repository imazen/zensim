#!/usr/bin/env python3
"""KADIS-700k -> 924 features (folded+append STREAMING regime, ZENSIM_AB_MODE=foldapp,
zensim 0b3d16b0) via RESCORE-FROM-LINKS (no regeneration). 924 adaptation of kadis720_rescore.py.

The GPU-metrics canonical (2026-07-01) persisted every distorted PNG on R2
(distorted_url). Those are the GROUND-TRUTH labeled distortions the 7 metric
scores were computed on. So instead of regenerating distortions with kadis-distort
(which uses a different RNG seed -> divergent pixels), download the persisted PNG +
its ref and extract 720 features directly. Correct (exact labeled image), ~5x
faster (no generation), non-duplicative.

Per worker: chunk = refs whose md5 hash falls in this slot. Download ref once,
download each cell's distorted PNG, extract 720 (v2_ab_extract), join the canonical
metadata + all 7 metric scores. Batched row-group writes (crash-safe + no tiny-
row-group bloat). Resumable (skips cells already in the output by distorted_url).
"""
import os, sys, subprocess, shutil, tempfile, hashlib, csv as _csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
import boto3
from botocore.config import Config

DL_THREADS = int(os.environ.get("DL_THREADS", "8"))

HOME = os.path.expanduser("~")
EXTRACT = f"{HOME}/kadis924/v2_ab_extract"
META = f"{HOME}/kadis924/{os.environ.get('META_FILE', 'kadis700k_meta.parquet')}"
CHUNK_I = int(os.environ.get("CHUNK_I", "0"))
CHUNK_N = int(os.environ.get("CHUNK_N", "1"))
OUT = f"{HOME}/kadis924/out/kadis924_c{CHUNK_I:02d}.parquet"
REF_BUCKET, REF_PREFIX = "zentrain", "kadis-700k/refs/"
FEATCOLS = [f"f{i}" for i in range(924)]
SCORECOLS = ['score_butteraugli_max_gpu', 'score_butteraugli_pnorm3_gpu',
             'score_cvvdp_cpu_imazen_v0_1_0', 'score_dssim_gpu', 'score_iwssim_gpu',
             'score_ssim2_gpu', 'score_zensim_gpu']
METACOLS = ['source_filename', 'dist_type', 'dist_name', 'severity_level', 'dist_param', 'source_id']
BATCH_REFS = 100

s3 = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT"],
                  config=Config(max_pool_connections=32, retries={"max_attempts": 5}))


def parse_s3(url):
    b, k = url[5:].split("/", 1)
    return b, k


def main():
    m = pq.read_table(META).to_pydict()
    by = defaultdict(list)
    for i in range(len(m["source_filename"])):
        by[m["source_filename"][i]].append(i)
    refs = [r for r in sorted(by)
            if int(hashlib.md5(r.encode()).hexdigest()[:8], 16) % CHUNK_N == CHUNK_I]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        try:
            done = set(pq.read_table(OUT, columns=["distorted_url"]).column("distorted_url").to_pylist())
        except Exception:
            pass
    print(f"kadis924 rescore: {len(refs)} refs, {len(m['source_filename'])} cells total; "
          f"{len(done)} cells already done", flush=True)
    writer = [None]
    buf_f = {c: [] for c in FEATCOLS}
    buf_m = {c: [] for c in METACOLS + SCORECOLS + ["distorted_url"]}

    def flush():
        if not buf_m["distorted_url"]:
            return
        t = pa.table({**{c: pa.array(buf_f[c], pa.float32()) for c in FEATCOLS},
                      **{c: pa.array(buf_m[c]) for c in buf_m}})
        if writer[0] is None:
            writer[0] = pq.ParquetWriter(OUT, t.schema, compression="zstd")
        writer[0].write_table(t)
        for c in buf_f:
            buf_f[c].clear()
        for c in buf_m:
            buf_m[c].clear()

    processed = 0
    pool = ThreadPoolExecutor(max_workers=DL_THREADS)
    with tempfile.TemporaryDirectory(dir=f"{HOME}/kadis924") as tmp:
        for k, fn in enumerate(refs):
            idxs = [i for i in by[fn] if m["distorted_url"][i] not in done]
            if not idxs:
                continue
            refp = os.path.join(tmp, "ref.png")
            try:
                s3.download_file(REF_BUCKET, REF_PREFIX + fn, refp)
            except Exception as e:
                print(f"  SKIP ref {fn}: {e}", flush=True)
                continue
            gen = os.path.join(tmp, "g")
            shutil.rmtree(gen, ignore_errors=True)
            os.makedirs(gen)

            def _dl(ji):
                j, i = ji
                dp = os.path.join(gen, f"d{j}.png")
                b, key = parse_s3(m["distorted_url"][i])
                try:
                    s3.download_file(b, key, dp)
                    return (j, i, dp)
                except Exception:
                    return None

            got = [r for r in pool.map(_dl, list(enumerate(idxs))) if r]
            labels = []
            with open(os.path.join(tmp, "pairs.tsv"), "w") as f:
                f.write("ref_path\tdist_path\thuman_score\n")
                for (j, i, dp) in got:
                    f.write(f"{refp}\t{dp}\t{j}\n")
                    labels.append((j, i))
            if not labels:
                shutil.rmtree(gen, ignore_errors=True)
                continue
            fcsv = os.path.join(tmp, "f.csv")
            r = subprocess.run([EXTRACT, os.path.join(tmp, "pairs.tsv"), fcsv],
                               capture_output=True, text=True,
                               env={**os.environ, "ZENSIM_AB_MODE": "foldapp"})
            shutil.rmtree(gen, ignore_errors=True)
            if r.returncode != 0 or not os.path.exists(fcsv):
                print(f"  SKIP {fn}: extract ({r.stderr.strip()[:80]})", flush=True)
                continue
            with open(fcsv) as fh:
                rd = _csv.reader(fh)
                hdr = next(rd)
                if any(c not in hdr for c in FEATCOLS):
                    continue
                fi = [hdr.index(c) for c in FEATCOLS]
                hj = hdr.index("human_score")
                feats = {int(float(row[hj])): np.array([float(row[x]) for x in fi], np.float32)
                         for row in rd}
            for (j, i) in labels:
                v = feats.get(j)
                if v is None or not np.all(np.isfinite(v)):
                    continue
                for x, c in enumerate(FEATCOLS):
                    buf_f[c].append(float(v[x]))
                for c in METACOLS + SCORECOLS:
                    buf_m[c].append(m[c][i])
                buf_m["distorted_url"].append(m["distorted_url"][i])
                processed += 1
            if (k + 1) % BATCH_REFS == 0:
                flush()
                print(f"[{k+1}/{len(refs)}] refs, {processed} cells written", flush=True)
        flush()
    if writer[0]:
        writer[0].close()
    print(f"DONE: {OUT}  {processed} cells", flush=True)


if __name__ == "__main__":
    main()
