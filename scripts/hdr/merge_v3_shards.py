#!/usr/bin/env python3
"""Merge hdr_score_fleet v3 (pu-linear) feature shards into datagen-shaped
sidecar dirs so build_hdr_train_parquets.py consumes them unchanged.

  usage: merge_v3_shards.py <run_id> <n_boxes> <out_datagen_dir>
  e.g.:  merge_v3_shards.py v3june 4 /mnt/v/output/zenmetrics/datagen-v3-june
         merge_v3_shards.py v3hq   6 /mnt/v/output/zenmetrics/datagen-v3-hq

Pulls s3://zentrain/hdr/runs/<run>/box-*/zensim_features.parquet (+ zensim
scores), concatenates, and writes <out>/sidecars/zenjxl/zensim_features.parquet.
The omni (ssim2/cvvdp targets) is NOT touched — the builder reads those from
the ORIGINAL datagen dir; pass this dir via --extra-datagen only for features
by symlinking omni/ and the cvvdp sidecar from the original.
"""
import os, subprocess, sys
import pyarrow as pa
import pyarrow.parquet as pq

run, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
orig = {
    "v3june": "/mnt/v/output/zenmetrics/datagen-2026-06-23-hdr",
    "v3hq": "/mnt/v/output/zenmetrics/datagen-2026-07-03-hdr-hq",
}[run]

env = dict(os.environ)
for line in open(os.path.expanduser("~/.config/cloudflare/r2-credentials")):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1)
        env[k] = v
ep = f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
env["AWS_ACCESS_KEY_ID"] = env["R2_ACCESS_KEY_ID"]
env["AWS_SECRET_ACCESS_KEY"] = env["R2_SECRET_ACCESS_KEY"]
env["AWS_REGION"] = "auto"

sdir = os.path.join(out, "sidecars", "zenjxl")
os.makedirs(sdir, exist_ok=True)
tabs, stabs = [], []
for i in range(n):
    for name, acc in (("zensim_features.parquet", tabs), ("zensim.parquet", stabs)):
        local = f"/tmp/{run}_box{i}_{name}"
        r = subprocess.run(
            ["s5cmd", "--endpoint-url", ep, "cp",
             f"s3://zentrain/hdr/runs/{run}/box-{i}/{name}", local],
            env=env, capture_output=True)
        if r.returncode == 0:
            t = pq.read_table(local)
            assert t.num_rows > 0, f"EMPTY shard {run}/box-{i}/{name}"
            acc.append(t)
        else:
            print(f"MISSING shard: {run}/box-{i}/{name}", file=sys.stderr)
assert len(tabs) == n, f"only {len(tabs)}/{n} feature shards present — do not merge partial"
merged = pa.concat_tables(tabs)
pq.write_table(merged, os.path.join(sdir, "zensim_features.parquet"), compression="zstd")
if stabs:
    pq.write_table(pa.concat_tables(stabs), os.path.join(sdir, "zensim-pulinear-score.parquet"),
                   compression="zstd")
# borrow targets from the original datagen
os.makedirs(os.path.join(out, "omni"), exist_ok=True)
for rel in ("omni/zenjxl.tsv",):
    dst = os.path.join(out, rel)
    if not os.path.exists(dst):
        os.symlink(os.path.join(orig, rel), dst)
cv = os.path.join(orig, "sidecars", "zenjxl", "cvvdp.parquet")
cvd = os.path.join(sdir, "cvvdp.parquet")
if os.path.exists(cv) and not os.path.exists(cvd):
    os.symlink(cv, cvd)
print(f"{run}: merged {merged.num_rows:,} feature rows -> {sdir}")
