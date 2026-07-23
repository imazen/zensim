#!/usr/bin/env python3
"""Merge the zensim720 fleet's raw per-cell blobs into a 720-feature parquet.

The fleet (T-big multi-codec) wrote, per completed cell, a content-addressed blob at
s3://zentrain/jobs/<run>/blobs/<output_sha> — JSONL, one record per encode:
  {"kind":"feature","image_path":"o_NNNN.png.scaleWxH.png","codec":"zenjxl",
   "encode_sha":"...","regime":"v2-ab","features":[<720 floats>]}
The ledger (jobs/<run>/ledger/*.parquet) records which output_sha each done cell produced.

This tool reads a run's ledger -> its distinct output_shas -> fetches+parses each blob ->
emits rows (ref_basename, codec, encode_sha, f0..f719) to a per-run parquet shard. The
shards are later concatenated into the 720 index that join_eval_720.py joins the 372
train/val/eval datasets against (fingerprint match on ref + f0..f371, append f372..f719).

Per-run so it distributes: run one invocation per <run> across boxes.

  python3 merge_fleet_720.py <run>            # e.g. bf-zjxll-t4  -> /mnt/v/.../720shards/<run>.parquet
  python3 merge_fleet_720.py <run> <out.parquet>
"""
import os, sys, json, io
from concurrent.futures import ThreadPoolExecutor
from pyarrow import fs
import pyarrow as pa, pyarrow.parquet as pq

E = dict(os.environ)
def _s3():
    # dev box: root R2 keys (R2_*). fleet node: scoped temp cred (AWS_* + session token).
    if E.get("R2_ACCESS_KEY_ID"):
        return fs.S3FileSystem(access_key=E["R2_ACCESS_KEY_ID"], secret_key=E["R2_SECRET_ACCESS_KEY"],
            endpoint_override="https://%s.r2.cloudflarestorage.com" % E["R2_ACCOUNT_ID"], region="auto")
    ep = E.get("ZEN_R2_ENDPOINT") or ("https://%s.r2.cloudflarestorage.com" % E["R2_ACCOUNT_ID"])
    return fs.S3FileSystem(access_key=E["AWS_ACCESS_KEY_ID"], secret_key=E["AWS_SECRET_ACCESS_KEY"],
        session_token=E.get("AWS_SESSION_TOKEN"), endpoint_override=ep, region="auto")
S3 = _s3()
BUCKET = "zentrain"
NFEAT = 720
THREADS = int(E.get("MERGE_THREADS", "24"))


def ref_stem(image_path):
    # normalize to the join key used by the eval/train 372 parquets: drop .png + .scale suffix
    return str(image_path).replace(".png", "").split(".scale")[0]


def _ledger_shas(path):
    t = pq.read_table(S3.open_input_file(path), columns=["output_sha", "status"])
    return [s for s, st in zip(t.column("output_sha").to_pylist(), t.column("status").to_pylist())
            if s and str(st) == "done"]

def blob_shas(run):
    sel = fs.FileSelector(f"{BUCKET}/jobs/{run}/ledger/", recursive=False)
    paths = [b.path for b in S3.get_file_info(sel) if b.path.endswith(".parquet")]
    shas = set()
    with ThreadPoolExecutor(max_workers=THREADS) as ex:  # ledger can be 1000s of chunks — parallelize
        for chunk in ex.map(_ledger_shas, paths):
            shas.update(chunk)
    return shas


def fetch_records(args):
    run, sha = args
    try:
        raw = S3.open_input_file(f"{BUCKET}/jobs/{run}/blobs/{sha}").read().decode()
    except Exception as e:
        return ("ERR", sha, str(e)[:60])
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        f = o.get("features")
        if not isinstance(f, list) or len(f) != NFEAT:
            return ("BADFEAT", sha, f"len={len(f) if isinstance(f,list) else type(f).__name__}")
        out.append((ref_stem(o.get("image_path", "")), o.get("codec", ""), o.get("encode_sha", ""), f))
    return ("OK", out)


def main():
    run = sys.argv[1]
    out_p = sys.argv[2] if len(sys.argv) > 2 else f"/mnt/v/zen/zensim-training/720shards/{run}.parquet"
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    shas = blob_shas(run)
    print(f"{run}: {len(shas)} distinct done blobs", flush=True)
    refs, cods, encs, feats = [], [], [], [[] for _ in range(NFEAT)]
    errs = bad = 0
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        for i, res in enumerate(ex.map(fetch_records, ((run, s) for s in shas))):
            tag = res[0]
            if tag == "ERR":
                errs += 1; continue
            if tag == "BADFEAT":
                bad += 1
                if bad <= 3: print("  BADFEAT", res[1][:12], res[2], flush=True)
                continue
            for rs, cd, en, f in res[1]:
                refs.append(rs); cods.append(cd); encs.append(en)
                for k in range(NFEAT):
                    feats[k].append(f[k])
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(shas)} blobs, {len(refs)} rows", flush=True)
    cols = {"ref_basename": pa.array(refs), "codec": pa.array(cods), "encode_sha": pa.array(encs)}
    for k in range(NFEAT):
        cols[f"f{k}"] = pa.array(feats[k], type=pa.float64())
    pq.write_table(pa.table(cols), out_p, compression="zstd")
    print(f"{run}: wrote {len(refs)} rows x {NFEAT}f -> {out_p}  (blob errs={errs}, badfeat={bad})", flush=True)


if __name__ == "__main__":
    main()
