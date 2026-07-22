#!/usr/bin/env python3
"""Fetch the zenfleet job-system's per-cell 720-feature blobs for a target set
of source-image stems, for the nonphoto/imazen26 eval-720 backfill
(docs/V2_EXPERIMENT_PLAN_2026-07-20.md).

Context: the T-big bigcodec fleet backfill (concurrent zenmetrics session,
2026-07-19..22, ~56 `bf-*` job pools under `s3://zentrain/jobs/`) does NOT
expose a consolidated `(ref, f0..f719)` parquet — its `ledger/*.parquet`
chunks are job-tracking metadata ONLY (job_id/image_path/codec/q=-1/
knob_tuple_json="scorefile"/output_sha/status/...): q and knob_tuple_json are
NOT populated for score_file jobs (the actual codec/q identity lives inside
the RESULT blob's `encode_sha` filename, not the ledger row). The features
themselves live in per-job blobs at `s3://zentrain/jobs/<pool>/blobs/<sha>`,
one blob per completed job, addressed by content hash — NOT globally
shared across pools (`s3://zentrain/blobs/` is a different, ~empty prefix;
verified 2026-07-22 a same-sha lookup there 404s). A "done" ledger status
does not guarantee a valid feature blob either — some pools (e.g.
`bf-zenjpeg-lossy` observed 2026-07-22) have 100% "done" rows whose blobs are
`{"kind":"metric",...,"error":"...403 Forbidden..."}` records from a since-
patched R2-permissions bug, not feature vectors.

So this backfill is a 3-stage pipeline:
  1. `scan-ledgers`  — locally-synced ledger parquets (`aws s3 sync` each
     `bf-*/ledger/` dir first — NOT done by this script, see the wrapper
     shell block below) are scanned, filtered to rows whose image_path's
     ref-stem (`.replace(".png","").split(".scale")[0]`) is in the target
     stem set AND status=="done", tagged with their source pool (needed to
     find the blob — blobs are per-pool, not global), and written to one
     parquet.
  2. `fetch-blobs`   — dedupes to (pool, output_sha), fetches each via boto3
     (R2/S3). Each blob is JSONL, not a single JSON object — one job batches
     results for ~5-12 variants of the same source image, one JSON record per
     line (discovered 2026-07-22: a naive `json.loads(body)` on the whole
     blob throws "Extra data" on every multi-variant blob). Parses each line,
     KEEPS ONLY `kind=="feature"` records with `regime=="v2-ab"` and a
     720-length `features` array (anything else — error records like
     `{"kind":"metric","error":"...403 Forbidden..."}` from the since-patched
     R2-permissions bug, other kinds, malformed JSON — is counted and
     dropped, never fabricated), and writes
     image_path/codec/encode_sha/f0..f719 to a parquet. Checkpoints every
     `--checkpoint` blobs so a long fetch surviving an interruption doesn't
     lose all progress. Use `--exclude-pool` to skip pools already known to
     be 100% error blobs (bf-avif, bf-zenjpeg-lossy as of 2026-07-22).
  3. `join`          — hands the resulting 720-wide fleet-side table to
     `join_eval_720.py`'s fingerprint-join logic (ref_basename + rounded
     f0..f371 exact match) against an eval 372 parquet (nonphoto/imazen26).

Rate note (measured 2026-07-22, R2, this box): 64 boto3 threads sustain
~140 req/s; 256 threads regress to ~110 req/s (R2-side throttling) — use
64-128, not more.

Usage:
  # (wrapper, not this script) for p in $(cat pools.txt); do aws s3 sync \
  #   s3://zentrain/jobs/$p/ledger/ ledgers/$p/; done
  python3 fleet_blob_fetch_720.py scan-ledgers --ledgers-dir ~/tmp/fleet_ledgers \
      --target-stems-json ~/tmp/eval_target_stems.json --out matched_ledger.parquet

  python3 fleet_blob_fetch_720.py fetch-blobs --matched matched_ledger.parquet \
      --bucket zentrain --out fleet_features_720.parquet --workers 96

  python3 fleet_blob_fetch_720.py join --fleet fleet_features_720.parquet \
      --eval-372 nonphoto_features_372col_2026-07-15.parquet \
      --out ext_nonphoto_720.parquet

RECOMMENDED PATH — `fetch-and-match` fuses steps 2+3: at ~12 records/blob and
hundreds of thousands of blobs, materializing ALL fetched features (step 2's
plain `fetch-blobs`) before joining is a real memory/time problem (measured
2026-07-22: 378,172 blobs x ~12 rows/blob x 720 float64 cols would be ~26 GB
just for the raw feature matrix, and the naive checkpoint-by-rewriting-the-
whole-table approach is quadratic in elapsed time as it grows). Since we
already know the target fingerprints in advance (the eval-372 parquets),
`fetch-and-match` builds the (refstem, rounded-f0..371) -> eval-row-index
lookup ONCE, then matches each fetched record against it inline as blobs
stream in, keeping ONLY matches — bounded by the eval row counts (~20k), not
the fleet's total variant count (millions). It also exits early once every
eval row across all `--eval` inputs has been matched.

  python3 fleet_blob_fetch_720.py fetch-and-match \
      --matched matched_ledger.parquet \
      --exclude-pool bf-avif --exclude-pool bf-zenjpeg-lossy \
      --eval nonphoto:nonphoto_features_372col_2026-07-15.parquet \
      --eval imazen26:imazen26_test_120k_2026-07-16.parquet \
      --out-dir /mnt/v/output/zensim/v2-eval-720-2026-07-22 --workers 96
"""
import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def refstem(x: str) -> str:
    return str(x).replace(".png", "").split(".scale")[0]


def bounded_map(executor, fn, items, in_flight):
    """Apply `fn` over `items` via `executor`, keeping at most `in_flight` futures
    alive at once (NOT len(items)).

    BUG THIS FIXES (measured 2026-07-22): `{executor.submit(fn, x): x for x in items}`
    followed by `as_completed(that_dict)` keeps every Future — including COMPLETED
    ones — alive in the dict for the entire run, because the dict comprehension never
    removes an entry once its future finishes. A `concurrent.futures.Future` caches its
    result internally after completion, so with 378,172 pre-submitted futures each
    returning ~12 records of 720 floats, RSS grew ~69 MB/s (10.7 GB -> 13.3 GB in 37s)
    with no plateau in sight — an OOM on this 58 GiB box within minutes. Bounding the
    in-flight set AND `del`-ing each future the instant its result is consumed lets
    completed futures (and their cached results) be garbage collected immediately.
    """
    it = iter(items)
    futs = {}
    for item in itertools.islice(it, in_flight):
        futs[executor.submit(fn, item)] = None
    while futs:
        done, _ = wait(futs.keys(), return_when=FIRST_COMPLETED)
        for fut in done:
            del futs[fut]
            yield fut.result()
            nxt = next(it, None)
            if nxt is not None:
                futs[executor.submit(fn, nxt)] = None


def cmd_scan_ledgers(args):
    # --all: no stem filter — consolidate EVERY done ledger row (the T-big
    # full-merge path). Otherwise filter to the target stems as before.
    target_stems = None
    if not args.all:
        if not args.target_stems_json:
            raise SystemExit("scan-ledgers: pass --target-stems-json or --all")
        with open(args.target_stems_json) as f:
            target_stems = set(json.load(f))
        print(f"{len(target_stems)} target stems", file=sys.stderr)
    else:
        print("scan mode: ALL stems (no target filter)", file=sys.stderr)

    cols = ["image_path", "codec", "q", "knob_tuple_json", "output_sha", "status"]
    pools = sorted(
        p for p in os.listdir(args.ledgers_dir) if os.path.isdir(os.path.join(args.ledgers_dir, p))
    )
    print(f"{len(pools)} pools", file=sys.stderr)

    t0 = time.time()
    chunks = []
    for pool in pools:
        d = os.path.join(args.ledgers_dir, pool)
        if not any(fn.endswith(".parquet") for fn in os.listdir(d)):
            continue
        dataset = ds.dataset(d, format="parquet")
        for batch in dataset.scanner(columns=cols).to_batches():
            if batch.num_rows == 0:
                continue
            df = batch.to_pandas()
            df["_stem"] = df["image_path"].map(refstem)
            if target_stems is None:
                m = df[df["status"] == "done"]
            else:
                m = df[df["_stem"].isin(target_stems) & (df["status"] == "done")]
            if len(m):
                m = m.copy()
                m["pool"] = pool
                chunks.append(m)
        print(f"  {pool}: cumulative = {sum(len(x) for x in chunks)}", file=sys.stderr, flush=True)

    matched = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    print(f"TOTAL matched: {len(matched)} in {time.time() - t0:.1f}s", file=sys.stderr)
    matched.to_parquet(args.out)
    print(f"wrote {args.out}", file=sys.stderr)


def cmd_fetch_blobs(args):
    import boto3
    from botocore.config import Config

    with open(os.path.expanduser(args.r2_credentials)) as f:
        for line in f:
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip('"'))

    endpoint = f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(max_pool_connections=args.workers, retries={"max_attempts": 3}),
    )

    df = pd.read_parquet(args.matched)
    if args.exclude_pool:
        before = len(df)
        df = df[~df["pool"].isin(set(args.exclude_pool))]
        print(f"excluded pools {args.exclude_pool}: {before - len(df)} rows dropped", file=sys.stderr)
    todo = df.drop_duplicates(subset=["pool", "output_sha"])[["pool", "output_sha"]]
    print(f"{len(todo)} distinct (pool, output_sha) blobs to fetch (each is JSONL, ~5-12 variant records)", file=sys.stderr)

    def fetch_one(pool, sha):
        """A blob is JSONL — one job batches results for MULTIPLE variants of
        the same source image (typically ~12 lines; ~5 for zpng), one JSON
        object per line. Returns (status_counts_dict, list_of_ok_records)."""
        try:
            obj = s3.get_object(Bucket=args.bucket, Key=f"jobs/{pool}/blobs/{sha}")
            body = obj["Body"].read()
        except Exception as e:
            return ({"fetch_error": 1}, [])
        counts = {}
        ok_records = []
        for line in body.strip().split(b"\n"):
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                counts["json_error"] = counts.get("json_error", 0) + 1
                continue
            if rec.get("kind") != "feature":
                k = f"kind_{rec.get('kind', 'unknown')}"
                counts[k] = counts.get(k, 0) + 1
                continue
            if rec.get("regime") != "v2-ab":
                k = f"regime_{rec.get('regime')}"
                counts[k] = counts.get(k, 0) + 1
                continue
            feats = rec.get("features")
            if not isinstance(feats, list) or len(feats) != 720:
                counts["bad_feature_len"] = counts.get("bad_feature_len", 0) + 1
                continue
            counts["ok"] = counts.get("ok", 0) + 1
            ok_records.append(rec)
        return (counts, ok_records)

    counts = {}
    results = []
    t0 = time.time()
    n_done = 0

    def checkpoint():
        if not results:
            return
        tbl = _records_to_table(results)
        pq.write_table(tbl, args.out, compression="zstd")
        print(
            f"  checkpoint: {len(results)} feature rows written to {args.out} "
            f"({n_done} blobs fetched, {time.time() - t0:.0f}s elapsed)",
            file=sys.stderr,
        )

    items = list(zip(todo["pool"], todo["output_sha"]))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for blob_counts, ok_records in bounded_map(
            ex, lambda ps: fetch_one(ps[0], ps[1]), items, in_flight=args.workers * 4
        ):
            for k, v in blob_counts.items():
                counts[k] = counts.get(k, 0) + v
            results.extend(ok_records)
            n_done += 1
            if n_done % args.checkpoint == 0:
                rate = n_done / (time.time() - t0)
                print(
                    f"[{n_done}/{len(todo)}] rate={rate:.1f} blobs/s  {len(results)} feature rows so far  "
                    f"status_counts={counts}",
                    file=sys.stderr,
                    flush=True,
                )
                checkpoint()

    checkpoint()
    print(f"\nFINAL status counts: {counts}", file=sys.stderr)
    print(f"feature rows: {len(results)} / {len(todo)} fetched ({100 * len(results) / max(len(todo), 1):.1f}%)", file=sys.stderr)


def _records_to_table(records):
    n = len(records)
    image_path = [r["image_path"] for r in records]
    codec = [r["codec"] for r in records]
    encode_sha = [r.get("encode_sha", "") for r in records]
    feats = np.array([r["features"] for r in records], dtype=np.float64)
    cols = {
        "image_path": pa.array(image_path),
        "codec": pa.array(codec),
        "encode_sha": pa.array(encode_sha),
    }
    for i in range(720):
        cols[f"f{i}"] = pa.array(feats[:, i], type=pa.float64())
    return pa.table(cols)


def _records_to_table_full(records):
    """fetch-all row shape: identity (image_path/codec/encode_sha/pool) +
    optional zensim_score + f0..f719. Separate from `_records_to_table` so
    the eval fetch paths keep their exact schema."""
    cols = {
        "image_path": pa.array([r["image_path"] for r in records]),
        "codec": pa.array([r["codec"] for r in records]),
        "encode_sha": pa.array([r.get("encode_sha", "") for r in records]),
        "pool": pa.array([r["_pool"] for r in records]),
        "zensim_score": pa.array(
            [r.get("zensim_score") for r in records], type=pa.float64()
        ),
    }
    feats = np.array([r["features"] for r in records], dtype=np.float64)
    for i in range(720):
        cols[f"f{i}"] = pa.array(feats[:, i], type=pa.float64())
    return pa.table(cols)


def cmd_fetch_all(args):
    """Streaming full merge of EVERY feature record: fetch each distinct
    (pool, output_sha) blob, keep kind=feature/regime=v2-ab/len=720 rows,
    dedupe on encode_sha ACROSS pools (poisoned early pools produce only
    error rows, which drop at parse; codec-retry pools like bf-zjl2 then
    supply the real rows — first good record per encode_sha wins, dupes
    counted), and append to ONE zstd parquet via row-group streaming
    (constant memory — `fetch-blobs`'s all-in-RAM checkpoint rewrite is
    exactly what this mode exists to avoid at ~1M-blob scale).

    Resume: flushed blobs are logged to `<out>.fetched`; rerunning skips
    them (a crash loses only the unflushed buffer, which refetches)."""
    import boto3
    from botocore.config import Config

    with open(os.path.expanduser(args.r2_credentials)) as f:
        for line in f:
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip('"'))

    endpoint = f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(max_pool_connections=args.workers, retries={"max_attempts": 3}),
    )

    df = pd.read_parquet(args.matched)
    if args.exclude_pool:
        before = len(df)
        df = df[~df["pool"].isin(set(args.exclude_pool))]
        print(f"excluded pools {args.exclude_pool}: {before - len(df)} rows dropped", file=sys.stderr)
    todo = df.drop_duplicates(subset=["pool", "output_sha"])[["pool", "output_sha"]]

    fetched_log = args.out + ".fetched"
    already = set()
    if os.path.exists(fetched_log):
        with open(fetched_log) as f:
            already = {line.strip() for line in f if line.strip()}
        print(f"resume: {len(already)} blobs already flushed", file=sys.stderr)
    items = [
        (p, s) for p, s in zip(todo["pool"], todo["output_sha"]) if f"{p}/{s}" not in already
    ]
    print(
        f"{len(items)} blobs to fetch ({len(todo)} total distinct (pool, output_sha))",
        file=sys.stderr,
    )

    def fetch_one(pool, sha):
        try:
            obj = s3.get_object(Bucket=args.bucket, Key=f"jobs/{pool}/blobs/{sha}")
            body = obj["Body"].read()
        except Exception:
            return (pool, sha, {"fetch_error": 1}, [])
        counts = {}
        ok = []
        for line in body.strip().split(b"\n"):
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                counts["json_error"] = counts.get("json_error", 0) + 1
                continue
            if rec.get("kind") != "feature":
                k = "error_row" if rec.get("error") else f"kind_{rec.get('kind', 'unknown')}"
                counts[k] = counts.get(k, 0) + 1
                continue
            if rec.get("regime") != "v2-ab":
                counts[f"regime_{rec.get('regime')}"] = counts.get(f"regime_{rec.get('regime')}", 0) + 1
                continue
            feats = rec.get("features")
            if not isinstance(feats, list) or len(feats) != 720:
                counts["bad_feature_len"] = counts.get("bad_feature_len", 0) + 1
                continue
            counts["ok"] = counts.get("ok", 0) + 1
            rec["_pool"] = pool
            ok.append(rec)
        return (pool, sha, counts, ok)

    counts = {}
    per_pool_ok = {}
    seen = set()
    dupes = 0
    buffer = []
    buffer_blobs = []
    writer = None
    rows_written = 0
    t0 = time.time()
    n_done = 0

    def flush():
        nonlocal writer, rows_written, buffer, buffer_blobs
        if buffer:
            tbl = _records_to_table_full(buffer)
            if writer is None:
                writer = pq.ParquetWriter(args.out, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            rows_written += len(buffer)
            buffer = []
        if buffer_blobs:
            with open(fetched_log, "a") as f:
                for b in buffer_blobs:
                    f.write(b + "\n")
            buffer_blobs = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for pool, sha, blob_counts, ok in bounded_map(
            ex, lambda ps: fetch_one(ps[0], ps[1]), items, in_flight=args.workers * 4
        ):
            for k, v in blob_counts.items():
                counts[k] = counts.get(k, 0) + v
            if blob_counts.get("ok"):
                per_pool_ok[pool] = per_pool_ok.get(pool, 0) + blob_counts["ok"]
            for rec in ok:
                key = rec.get("encode_sha", "")
                if key in seen:
                    dupes += 1
                    continue
                seen.add(key)
                buffer.append(rec)
            buffer_blobs.append(f"{pool}/{sha}")
            n_done += 1
            if len(buffer) >= args.rows_per_group:
                flush()
            if n_done % args.checkpoint == 0:
                rate = n_done / max(time.time() - t0, 1e-9)
                eta = (len(items) - n_done) / max(rate, 1e-9) / 60
                print(
                    f"[{n_done}/{len(items)}] rate={rate:.1f} blobs/s eta={eta:.0f}m "
                    f"rows={rows_written + len(buffer)} dupes={dupes} counts={counts}",
                    file=sys.stderr,
                    flush=True,
                )

    flush()
    if writer is not None:
        writer.close()
    print(f"\nFINAL: {rows_written} rows -> {args.out}", file=sys.stderr)
    print(f"dupes (same encode_sha, cross re-run/pool): {dupes}", file=sys.stderr)
    print(f"status counts: {counts}", file=sys.stderr)
    print("per-pool ok rows:", json.dumps(per_pool_ok, indent=0, sort_keys=True), file=sys.stderr)


ROUND = 6


def _build_fingerprint_index(eval_path):
    """(refstem, rounded f0..f371 tuple) -> row index, plus the raw table for
    later output assembly. Same fingerprint scheme as join_eval_720.py."""

    def fcols(schema):
        fs = [c for c in schema.names if c.startswith("f") and c[1:].isdigit()]
        if not fs:
            fs = [c for c in schema.names if c.startswith("feat_") and c[5:].isdigit()]
        fs.sort(key=lambda c: int(c.split("_")[-1] if "_" in c else c[1:]))
        return fs

    def refcol(schema):
        for c in ("ref_basename", "ref_filename", "image_path"):
            if c in schema.names:
                return c
        raise SystemExit(f"no ref column in {schema.names[:8]}")

    t = pq.read_table(eval_path)
    fs = fcols(t.schema)
    assert len(fs) == 372, f"{eval_path}: expected 372 f-cols, got {len(fs)}"
    rc = refcol(t.schema)
    refs = [refstem(x) for x in t.column(rc).to_pylist()]
    mat = np.column_stack([t.column(c).to_numpy() for c in fs]).astype("f8")
    fp = {}
    for i in range(len(refs)):
        key = (refs[i], tuple(np.round(mat[i], ROUND)))
        fp.setdefault(key, i)
    return {"table": t, "refcol": rc, "fcols": fs, "fp": fp, "n": len(refs)}


def cmd_fetch_and_match(args):
    import boto3
    from botocore.config import Config

    with open(os.path.expanduser(args.r2_credentials)) as f:
        for line in f:
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip('"'))

    endpoint = f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(max_pool_connections=args.workers, retries={"max_attempts": 3}),
    )

    evals = {}
    total_eval_rows = 0
    for spec in args.eval:
        name, path = spec.split(":", 1)
        evals[name] = _build_fingerprint_index(path)
        total_eval_rows += evals[name]["n"]
        print(f"eval '{name}': {evals[name]['n']} rows, {len(evals[name]['fp'])} distinct fingerprints", file=sys.stderr)

    matched = {name: {} for name in evals}  # name -> {row_idx: features[372:720]}
    total_matched = 0

    df = pd.read_parquet(args.matched)
    if args.exclude_pool:
        before = len(df)
        df = df[~df["pool"].isin(set(args.exclude_pool))]
        print(f"excluded pools {args.exclude_pool}: {before - len(df)} rows dropped", file=sys.stderr)
    todo = df.drop_duplicates(subset=["pool", "output_sha"])[["pool", "output_sha"]]
    print(f"{len(todo)} distinct (pool, output_sha) blobs to scan", file=sys.stderr)

    def fetch_one(pool, sha):
        try:
            obj = s3.get_object(Bucket=args.bucket, Key=f"jobs/{pool}/blobs/{sha}")
            body = obj["Body"].read()
        except Exception:
            return []
        out = []
        for line in body.strip().split(b"\n"):
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("kind") != "feature" or rec.get("regime") != "v2-ab":
                continue
            feats = rec.get("features")
            if not isinstance(feats, list) or len(feats) != 720:
                continue
            out.append((rec["image_path"], feats))
        return out

    t0 = time.time()
    n_blobs_done = 0
    n_records_seen = 0
    stop = False

    def write_outputs():
        for name, idx in evals.items():
            keep = sorted(matched[name])
            if not keep:
                print(f"  {name}: 0 matches, skipping output", file=sys.stderr)
                continue
            t = idx["table"]
            rc = idx["refcol"]
            fs = idx["fcols"]
            human = t.column("human_score").to_pylist() if "human_score" in t.schema.names else [None] * idx["n"]
            cols = {
                "ref_basename": pa.array([t.column(rc)[i].as_py() for i in keep]),
                "human_score": pa.array([human[i] for i in keep]),
            }
            emat = np.column_stack([t.column(c).to_numpy() for c in fs]).astype("f8")
            for k in range(372):
                cols[f"f{k}"] = pa.array([emat[i][k] for i in keep], type=pa.float64())
            for k in range(348):
                cols[f"f{372+k}"] = pa.array([matched[name][i][k] for i in keep], type=pa.float64())
            out_path = os.path.join(args.out_dir, f"ext_{name}_720.parquet")
            pq.write_table(pa.table(cols), out_path, compression="zstd")
            print(f"  {name}: {len(keep)}/{idx['n']} matched ({100*len(keep)/idx['n']:.1f}%) -> {out_path}", file=sys.stderr)

    items = list(zip(todo["pool"], todo["output_sha"]))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for recs in bounded_map(ex, lambda ps: fetch_one(ps[0], ps[1]), items, in_flight=args.workers * 4):
            if stop:
                break
            n_blobs_done += 1
            for image_path, feats in recs:
                n_records_seen += 1
                key_ref = refstem(image_path)
                farr = np.array(feats[:372])
                key = (key_ref, tuple(np.round(farr, ROUND)))
                for name, idx in evals.items():
                    j = idx["fp"].get(key)
                    if j is not None and j not in matched[name]:
                        matched[name][j] = feats[372:720]
                        total_matched += 1
            if n_blobs_done % args.report_every == 0:
                rate = n_blobs_done / (time.time() - t0)
                print(
                    f"[{n_blobs_done}/{len(todo)}] rate={rate:.1f} blobs/s  "
                    f"records_seen={n_records_seen}  matched={total_matched}/{total_eval_rows}  "
                    f"elapsed={time.time()-t0:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                write_outputs()
            if total_matched >= total_eval_rows:
                print("ALL eval rows matched -- stopping early", file=sys.stderr)
                stop = True

    print(f"\nFINAL: {n_blobs_done} blobs scanned, {n_records_seen} feature records seen, "
          f"{total_matched}/{total_eval_rows} eval rows matched, {time.time()-t0:.0f}s total", file=sys.stderr)
    write_outputs()


def cmd_join(args):
    """Fingerprint-join a fleet 720 table onto an eval 372 parquet — same
    logic as join_eval_720.py, inlined here so this script is self-contained
    once fetch-blobs has produced the fleet-side table."""
    ROUND = 6

    def fcols(schema):
        fs = [c for c in schema.names if c.startswith("f") and c[1:].isdigit()]
        fs.sort(key=lambda c: int(c[1:]))
        return fs

    def refcol(schema):
        for c in ("ref_basename", "ref_filename", "image_path"):
            if c in schema.names:
                return c
        raise SystemExit(f"no ref column in {schema.names[:8]}")

    ft = pq.read_table(args.fleet)
    ffs = fcols(ft.schema)
    assert len(ffs) == 720, f"fleet table has {len(ffs)} f-cols, expected 720"
    frefs = [refstem(x) for x in ft.column(refcol(ft.schema)).to_pylist()]
    fmat = np.column_stack([ft.column(c).to_numpy() for c in ffs]).astype("f8")
    fp = {}
    for i, r in enumerate(frefs):
        key = (r, tuple(np.round(fmat[i, :372], ROUND)))
        fp.setdefault(key, i)

    et = pq.read_table(args.eval_372)
    efs = fcols(et.schema)
    assert len(efs) == 372, f"eval parquet has {len(efs)} f-cols, expected 372"
    erefs = [refstem(x) for x in et.column(refcol(et.schema)).to_pylist()]
    emat = np.column_stack([et.column(c).to_numpy() for c in efs]).astype("f8")
    ehuman = et.column("human_score").to_pylist() if "human_score" in et.schema.names else [None] * len(erefs)

    matched = {}
    for i in range(len(erefs)):
        key = (erefs[i], tuple(np.round(emat[i], ROUND)))
        j = fp.get(key)
        if j is not None:
            matched[i] = fmat[j, 372:720]

    n = len(erefs)
    hit = len(matched)
    print(f"{args.eval_372}: matched {hit}/{n} ({100 * hit / max(n,1):.1f}%) to fleet 720 output", file=sys.stderr)
    if hit < n:
        miss = [erefs[i] for i in range(n) if i not in matched][:10]
        print(f"  UNMATCHED (dropped, first 10): {miss}", file=sys.stderr)
    if hit == 0:
        print("0 matches — nothing to write", file=sys.stderr)
        return

    keep = sorted(matched)
    cols = {
        "ref_basename": pa.array([et.column(refcol(et.schema))[i].as_py() for i in keep]),
        "human_score": pa.array([ehuman[i] for i in keep]),
    }
    for k in range(372):
        cols[f"f{k}"] = pa.array([emat[i][k] for i in keep], type=pa.float64())
    for k in range(348):
        cols[f"f{372+k}"] = pa.array([matched[i][k] for i in keep], type=pa.float64())
    pq.write_table(pa.table(cols), args.out, compression="zstd")
    print(f"wrote {len(keep)} rows x 720 -> {args.out}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("scan-ledgers")
    s1.add_argument("--ledgers-dir", required=True)
    s1.add_argument("--target-stems-json")
    s1.add_argument("--all", action="store_true", help="no stem filter — every done row (T-big full merge)")
    s1.add_argument("--out", required=True)
    s1.set_defaults(func=cmd_scan_ledgers)

    s2 = sub.add_parser("fetch-blobs")
    s2.add_argument("--matched", required=True)
    s2.add_argument("--bucket", default="zentrain")
    s2.add_argument("--out", required=True)
    s2.add_argument("--workers", type=int, default=96)
    s2.add_argument("--checkpoint", type=int, default=10000)
    s2.add_argument("--r2-credentials", default="~/.config/cloudflare/r2-credentials")
    s2.add_argument(
        "--exclude-pool",
        action="append",
        default=[],
        help="pool name to skip entirely (repeatable) -- e.g. bf-avif / bf-zenjpeg-lossy, which "
        "were found 2026-07-22 to be 100%% error blobs from a since-patched R2-permissions bug "
        "(objstore get on canonical/2026-06-27/.../encodes/... returned 403 Forbidden); every "
        "other pool sampled (bf-zjl2 and all -tN shards) was 100%% valid kind=\"feature\" records",
    )
    s2.set_defaults(func=cmd_fetch_blobs)

    s3_ = sub.add_parser("join")
    s3_.add_argument("--fleet", required=True)
    s3_.add_argument("--eval-372", required=True)
    s3_.add_argument("--out", required=True)
    s3_.set_defaults(func=cmd_join)

    s5_ = sub.add_parser(
        "fetch-all",
        help="streaming FULL merge: every feature record -> one zstd parquet (T-big write-back)",
    )
    s5_.add_argument("--matched", required=True)
    s5_.add_argument("--bucket", default="zentrain")
    s5_.add_argument("--out", required=True)
    s5_.add_argument("--workers", type=int, default=96)
    s5_.add_argument("--checkpoint", type=int, default=5000)
    s5_.add_argument("--rows-per-group", type=int, default=50000)
    s5_.add_argument("--exclude-pool", action="append")
    s5_.add_argument("--r2-credentials", default="~/.config/cloudflare/r2-credentials")
    s5_.set_defaults(func=cmd_fetch_all)

    s4 = sub.add_parser("fetch-and-match", help="RECOMMENDED: fuses fetch-blobs+join, bounded memory")
    s4.add_argument("--matched", required=True)
    s4.add_argument("--bucket", default="zentrain")
    s4.add_argument("--eval", action="append", required=True, help="name:path.parquet, repeatable")
    s4.add_argument("--out-dir", required=True)
    s4.add_argument("--workers", type=int, default=96)
    s4.add_argument("--report-every", type=int, default=5000)
    s4.add_argument("--r2-credentials", default="~/.config/cloudflare/r2-credentials")
    s4.add_argument("--exclude-pool", action="append", default=[])
    s4.set_defaults(func=cmd_fetch_and_match)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
