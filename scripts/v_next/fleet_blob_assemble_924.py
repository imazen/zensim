#!/usr/bin/env python3
"""Assemble the bf924 wave's per-cell blobs into ONE 924-feature fleet parquet
(tbig_924_full.parquet) — the 924 counterpart of fleet_blob_fetch_720.py's
fetch-all path, sized for the FULL 490,173-cell / ~5.74M-row wave.

Stages (run in order; each is resumable):
  sync-ledgers  s5cmd-sync every bf924-* ledger dir locally (small parquets).
  scan          done rows -> one (pool, job_id, output_sha, image_path, codec)
                parquet, deduped by job_id keep-first (re-executed cells can
                carry order-permuted = different-sha blobs; any one is valid —
                the blob's record SET is identical).
  fetch         bounded-thread blob fetch (64-128 workers — R2 throttles past
                that, measured 2026-07-22), JSONL parse, KEEP ONLY
                kind=="feature" && regime=="folded720append" && len==924
                (error records are counted per class, never fabricated),
                incremental ParquetWriter flushes every --flush rows,
                checkpointed by job_id so an interrupted fetch resumes.

Output schema (mirrors tbig_720_full minus its vestigial score column):
  image_path, codec, encode_sha, pool, f0..f923 (float64).

Usage:
  python3 fleet_blob_assemble_924.py sync-ledgers --work ~/tmp/bf924_join
  python3 fleet_blob_assemble_924.py scan        --work ~/tmp/bf924_join
  python3 fleet_blob_assemble_924.py fetch       --work ~/tmp/bf924_join \
      --out /mnt/v/output/zensim/tbig-924-2026-07-27/tbig_924_full.parquet --workers 96
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

N_FEAT = 924
REGIME = "folded720append"
BUCKET = "zentrain"


def envs():
    e = dict(os.environ)
    for line in open(os.path.expanduser("~/.config/cloudflare/r2-credentials")):
        line = line.strip()
        if line.startswith("R2_") and "=" in line:
            k, v = line.split("=", 1)
            e[k] = v.strip().strip('"').strip("'")
    e["EP"] = "https://%s.r2.cloudflarestorage.com" % e["R2_ACCOUNT_ID"]
    e["AWS_ACCESS_KEY_ID"] = e["R2_ACCESS_KEY_ID"]
    e["AWS_SECRET_ACCESS_KEY"] = e["R2_SECRET_ACCESS_KEY"]
    e["AWS_REGION"] = "auto"
    return e


E = envs()


def runlist() -> list[str]:
    r = subprocess.run(
        ["s5cmd", "--endpoint-url", E["EP"], "cat", f"s3://{BUCKET}/jobs/_pool924/runlist.tsv"],
        env=E, capture_output=True,
    )
    return [ln.split("\t")[0] for ln in r.stdout.decode().splitlines() if ln.strip()]


def cmd_sync_ledgers(work: Path) -> None:
    ld = work / "ledgers"
    ld.mkdir(parents=True, exist_ok=True)
    for run in runlist():
        d = ld / run
        d.mkdir(exist_ok=True)
        r = subprocess.run(
            ["s5cmd", "--endpoint-url", E["EP"], "sync",
             f"s3://{BUCKET}/jobs/{run}/ledger/*", str(d) + "/"],
            env=E, capture_output=True,
        )
        n = len(list(d.glob("*.parquet")))
        print(f"{run}: {n} ledger shards", flush=True)
    print("sync-ledgers done")


def cmd_scan(work: Path) -> None:
    ld = work / "ledgers"
    seen: set[str] = set()
    out_rows = {k: [] for k in ("pool", "job_id", "output_sha", "image_path", "codec")}
    for run_dir in sorted(ld.iterdir()):
        if not run_dir.is_dir():
            continue
        t = ds.dataset(str(run_dir), format="parquet").to_table(
            columns=["job_id", "status", "output_sha", "image_path", "codec"]
        )
        n_new = 0
        for jid, st, sha, ip, cod in zip(
            t.column("job_id").to_pylist(), t.column("status").to_pylist(),
            t.column("output_sha").to_pylist(), t.column("image_path").to_pylist(),
            t.column("codec").to_pylist(),
        ):
            if st == "done" and jid not in seen and sha:
                seen.add(jid)
                out_rows["pool"].append(run_dir.name)
                out_rows["job_id"].append(jid)
                out_rows["output_sha"].append(sha)
                out_rows["image_path"].append(ip)
                out_rows["codec"].append(cod)
                n_new += 1
        print(f"{run_dir.name}: +{n_new} (cum {len(seen)})", flush=True)
    pq.write_table(pa.table(out_rows), work / "matched_ledger.parquet", compression="zstd")
    print(f"scan done: {len(seen)} distinct done jobs -> matched_ledger.parquet")


def bounded_map(executor, fn, items, in_flight):
    it = iter(items)
    futures = set()
    while True:
        while len(futures) < in_flight:
            try:
                futures.add(executor.submit(fn, next(it)))
            except StopIteration:
                break
        if not futures:
            return
        done_set, futures = wait(futures, return_when=FIRST_COMPLETED)
        for f in done_set:
            yield f.result()


def cmd_fetch(work: Path, out: Path, workers: int, flush_every: int) -> None:
    import boto3
    from botocore.config import Config

    t = pq.read_table(work / "matched_ledger.parquet")
    jobs = list(zip(*[t.column(c).to_pylist()
                      for c in ("pool", "job_id", "output_sha", "image_path", "codec")]))
    ckpt = work / "fetched_jobids.txt"
    done_ids = set()
    if ckpt.exists():
        done_ids = set(ckpt.read_text().split())
        print(f"resume: {len(done_ids)} jobs already fetched")
    todo = [j for j in jobs if j[1] not in done_ids]
    print(f"fetch: {len(todo)} of {len(jobs)} jobs remaining; workers={workers}")

    s3 = boto3.client(
        "s3", endpoint_url=E["EP"], aws_access_key_id=E["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=E["R2_SECRET_ACCESS_KEY"], region_name="auto",
        config=Config(max_pool_connections=workers + 8, retries={"max_attempts": 6}),
    )

    def get(job):
        pool, jid, sha, ip, cod = job
        try:
            body = s3.get_object(Bucket=BUCKET, Key=f"jobs/{pool}/blobs/{sha}")["Body"].read()
            return (job, body, None)
        except Exception as e:  # noqa: BLE001
            return (job, None, str(e)[:120])

    names = ["image_path", "codec", "encode_sha", "pool"] + [f"f{i}" for i in range(N_FEAT)]
    schema = pa.schema(
        [(n, pa.string()) for n in names[:4]] + [(n, pa.float64()) for n in names[4:]]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    mode_append = out.exists() and len(done_ids) > 0
    writer = pq.ParquetWriter(
        str(out) + (".resume.parquet" if mode_append else ""), schema, compression="zstd"
    )
    buf = {n: [] for n in names}
    stats = {"rows": 0, "blobs": 0, "fetch_err": 0, "bad_record": 0, "error_record": 0}
    t0 = time.time()
    ck = open(ckpt, "a")

    def flush():
        if buf["image_path"]:
            writer.write_table(pa.table(
                {n: pa.array(buf[n], type=schema.field(n).type) for n in names}, schema=schema))
            for n in names:
                buf[n].clear()
        ck.flush()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for job, body, err in bounded_map(ex, get, todo, workers * 2):
            pool, jid, sha, ip, cod = job
            if err is not None:
                stats["fetch_err"] += 1
                print(f"FETCH-ERR {pool}/{sha}: {err}", flush=True)
                continue
            n_before = stats["rows"]
            for line in body.splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    stats["bad_record"] += 1
                    continue
                if r.get("kind") != "feature" or r.get("regime") != REGIME:
                    stats["error_record"] += 1
                    continue
                f = r.get("features")
                if not isinstance(f, list) or len(f) != N_FEAT:
                    stats["bad_record"] += 1
                    continue
                buf["image_path"].append(r.get("image_path", ip))
                buf["codec"].append(r.get("codec", cod))
                buf["encode_sha"].append(r.get("encode_sha", ""))
                buf["pool"].append(pool)
                for i, v in enumerate(f):
                    buf[f"f{i}"].append(float(v))
                stats["rows"] += 1
            stats["blobs"] += 1
            ck.write(jid + "\n")
            if stats["rows"] - n_before == 0:
                print(f"EMPTY-BLOB {pool}/{sha} (no feature records)", flush=True)
            if len(buf["image_path"]) >= flush_every:
                flush()
            if stats["blobs"] % 5000 == 0:
                el = time.time() - t0
                print(f"[{el:6.0f}s] blobs={stats['blobs']} rows={stats['rows']} "
                      f"({stats['blobs'] / el:.0f} blobs/s) err={stats['fetch_err']} "
                      f"skip={stats['error_record']}+{stats['bad_record']}", flush=True)
    flush()
    writer.close()
    ck.close()
    print(f"DONE {json.dumps(stats)}")
    if mode_append:
        print("NOTE: resume output written alongside as .resume.parquet — concatenate "
              "with the prior part via pyarrow before the join.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sync-ledgers", "scan", "fetch"])
    ap.add_argument("--work", type=Path, default=Path.home() / "tmp/bf924_join")
    ap.add_argument("--out", type=Path,
                    default=Path("/mnt/v/output/zensim/tbig-924-2026-07-27/tbig_924_full.parquet"))
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--flush", type=int, default=250_000)
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    if a.stage == "sync-ledgers":
        cmd_sync_ledgers(a.work)
    elif a.stage == "scan":
        cmd_scan(a.work)
    else:
        cmd_fetch(a.work, a.out, a.workers, a.flush)


if __name__ == "__main__":
    main()
