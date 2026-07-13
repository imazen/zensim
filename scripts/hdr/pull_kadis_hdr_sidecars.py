#!/usr/bin/env python3
"""Pull + consolidate the kadis-hdr fleet sidecars into datagen-dir shape.

The vast fleet (zenmetrics `onstart_hdr_pairs.sh` + `hdr_pairs_chunk_worker.sh`)
writes per-chunk sidecars to
`s3://codec-corpus/kadis-hdr-2026-07-13/sidecars/<chunk>/<metric>.parquet`
(+ `zensim_features.parquet` + `_DONE`). This script:

  1. s5cmd-syncs the sidecars prefix locally,
  2. verifies every chunk has its `_DONE` sentinel,
  3. concatenates each metric's chunks into ONE parquet at
     `<datagen>/sidecars/kadis-hdr/<metric>.parquet` — the exact layout
     `build_hdr_train_parquets.py --codec kadis-hdr` consumes,
  4. asserts total rows == 11,400 and (basename(image_path), codec, q)
     uniqueness per metric, prints row counts + sha256s for manifest pinning.

Usage:
  pull_kadis_hdr_sidecars.py [--datagen /mnt/v/output/zenmetrics/datagen-2026-07-12-hdr-kadis]
                             [--run-prefix s3://codec-corpus/kadis-hdr-2026-07-13]
                             [--expect-chunks 19] [--expect-rows 11400]

Needs ~/.aws/credentials [r2] profile + R2_ACCOUNT_ID (source
~/.config/cloudflare/r2-credentials).
"""
import argparse
import glob
import hashlib
import os
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--datagen", default="/mnt/v/output/zenmetrics/datagen-2026-07-12-hdr-kadis")
ap.add_argument("--run-prefix", default="s3://codec-corpus/kadis-hdr-2026-07-13")
ap.add_argument("--expect-chunks", type=int, default=19)
ap.add_argument("--expect-rows", type=int, default=11400)
ap.add_argument("--skip-sync", action="store_true", help="reuse an existing local sidecars pull")
a = ap.parse_args()

acct = os.environ.get("R2_ACCOUNT_ID")
assert acct, "R2_ACCOUNT_ID missing — source ~/.config/cloudflare/r2-credentials"
ep = f"https://{acct}.r2.cloudflarestorage.com"

pull = os.path.join(a.datagen, "sidecars-fleet-pull")
os.makedirs(pull, exist_ok=True)
if not a.skip_sync:
    subprocess.run(
        ["s5cmd", "--endpoint-url", ep, "--profile", "r2",
         "sync", f"{a.run_prefix}/sidecars/*", pull + "/"],
        check=True,
    )

done = sorted(glob.glob(os.path.join(pull, "*", "_DONE")))
chunks = sorted(d for d in os.listdir(pull) if os.path.isdir(os.path.join(pull, d)))
missing = [c for c in chunks if not os.path.exists(os.path.join(pull, c, "_DONE"))]
print(f"chunks pulled: {len(chunks)}, _DONE: {len(done)}, missing _DONE: {missing or 'none'}")
assert len(done) == a.expect_chunks and not missing, (
    f"fleet incomplete: {len(done)}/{a.expect_chunks} _DONE — do not consolidate partial output"
)

metrics = ["zensim-gpu", "ssim2-gpu", "cvvdp", "iwssim-gpu", "butteraugli-gpu", "zensim_features"]
outdir = os.path.join(a.datagen, "sidecars", "kadis-hdr")
os.makedirs(outdir, exist_ok=True)
for m in metrics:
    parts = sorted(glob.glob(os.path.join(pull, "*", f"{m}.parquet")))
    assert len(parts) == a.expect_chunks, f"{m}: {len(parts)}/{a.expect_chunks} chunk files"
    t = pa.concat_tables([pq.read_table(p) for p in parts])
    assert t.num_rows == a.expect_rows, f"{m}: {t.num_rows} rows != {a.expect_rows}"
    keys = set(zip(
        (os.path.basename(x) for x in t["image_path"].to_pylist()),
        t["codec"].to_pylist(),
        (float(x) for x in t["q"].to_pylist()),
    ))
    assert len(keys) == a.expect_rows, f"{m}: {len(keys)} unique keys != {a.expect_rows}"
    out = os.path.join(outdir, f"{m}.parquet")
    pq.write_table(t, out, compression="zstd")
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print(f"{m}: {t.num_rows} rows -> {out}  sha256 {sha[:16]}…")

print("consolidation OK — next: build_hdr_train_parquets.py --codec kadis-hdr "
      f"--datagen {a.datagen}")
