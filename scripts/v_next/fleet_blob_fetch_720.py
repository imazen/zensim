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
     (R2/S3), parses the JSON body, KEEPS ONLY `kind=="feature"` records with
     `regime=="v2-ab"` and a 720-length `features` array (anything else —
     error blobs, other kinds, malformed JSON — is counted and dropped, never
     fabricated), and writes image_path/codec/encode_sha/f0..f719 to a
     parquet. Checkpoints every `--checkpoint` blobs so a long fetch surviving
     an interruption doesn't lose all progress.
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
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def refstem(x: str) -> str:
    return str(x).replace(".png", "").split(".scale")[0]


def cmd_scan_ledgers(args):
    with open(args.target_stems_json) as f:
        target_stems = set(json.load(f))
    print(f"{len(target_stems)} target stems", file=sys.stderr)

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
    todo = df.drop_duplicates(subset=["pool", "output_sha"])[["pool", "output_sha"]]
    print(f"{len(todo)} distinct (pool, output_sha) to fetch", file=sys.stderr)

    def fetch_one(pool, sha):
        try:
            obj = s3.get_object(Bucket=args.bucket, Key=f"jobs/{pool}/blobs/{sha}")
            body = obj["Body"].read()
        except Exception as e:
            return ("fetch_error", str(e), None)
        try:
            rec = json.loads(body)
        except Exception as e:
            return ("json_error", str(e), None)
        if rec.get("kind") != "feature":
            return (f"kind_{rec.get('kind', 'unknown')}", rec.get("error", "")[:120], None)
        if rec.get("regime") != "v2-ab":
            return (f"regime_{rec.get('regime')}", "", None)
        feats = rec.get("features")
        if not isinstance(feats, list) or len(feats) != 720:
            return ("bad_feature_len", str(len(feats) if isinstance(feats, list) else type(feats)), None)
        return ("ok", None, rec)

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
            f"({n_done} fetched, {time.time() - t0:.0f}s elapsed)",
            file=sys.stderr,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, r.pool, r.output_sha): (r.pool, r.output_sha) for r in todo.itertuples()}
        for fut in as_completed(futs):
            status, detail, rec = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if status == "ok":
                results.append(rec)
            n_done += 1
            if n_done % args.checkpoint == 0:
                rate = n_done / (time.time() - t0)
                print(
                    f"[{n_done}/{len(todo)}] rate={rate:.1f}/s status_counts={counts}",
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
    s1.add_argument("--target-stems-json", required=True)
    s1.add_argument("--out", required=True)
    s1.set_defaults(func=cmd_scan_ledgers)

    s2 = sub.add_parser("fetch-blobs")
    s2.add_argument("--matched", required=True)
    s2.add_argument("--bucket", default="zentrain")
    s2.add_argument("--out", required=True)
    s2.add_argument("--workers", type=int, default=96)
    s2.add_argument("--checkpoint", type=int, default=10000)
    s2.add_argument("--r2-credentials", default="~/.config/cloudflare/r2-credentials")
    s2.set_defaults(func=cmd_fetch_blobs)

    s3_ = sub.add_parser("join")
    s3_.add_argument("--fleet", required=True)
    s3_.add_argument("--eval-372", required=True)
    s3_.add_argument("--out", required=True)
    s3_.set_defaults(func=cmd_join)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
