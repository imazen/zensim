#!/usr/bin/env python3
"""Promote the bigcodec 944 views: gate all 21 (G-BF1/G-BF2 via
gate_backfill944.py, positional vs the frozen 924 views), then install into the
canonical layout with a _MANIFEST.json (build commits + per-file sha256 + rows
+ gate verdicts). ANY gate FAIL aborts before anything is installed.

Usage:
  python3 promote_bigcodec944.py \
      --staged /mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/bigcodec_staged \
      --old-root /mnt/v/zen/zensim-training/ext924-canonical-2026-07-27/bigcodec \
      --out /mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/bigcodec \
      --zensim-commit d061636262387b8746ffa0c883a73731ce9ab789 \
      --zenmetrics-commit 57b7b9adbd33 \
      --image ghcr.io/imazen/zenfleet-worker:exec-zensim944-57b7b9ad@sha256:ebb4bf361486e87d0a6849967e8e9c5c5925b1df51152cd9f758ad1b84abddee
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

SPLITS = ("train", "validate", "test")
HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.join(HERE, "gate_backfill944.py")


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", required=True)
    ap.add_argument("--old-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zensim-commit", required=True)
    ap.add_argument("--zenmetrics-commit", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--skip-gates", action="store_true",
                    help="install only (gates must have PASSED in a prior run)")
    a = ap.parse_args()

    datasets = sorted(
        d for d in os.listdir(a.staged)
        if os.path.isfile(os.path.join(a.staged, d, "train_944.parquet"))
    )
    print(f"{len(datasets)} datasets staged: {datasets}")
    if len(datasets) * len(SPLITS) != 21:
        print(f"ABORT: expected 21 views, found {len(datasets) * len(SPLITS)}")
        return 1

    gates_dir = os.path.join(a.staged, "gates")
    os.makedirs(gates_dir, exist_ok=True)
    verdicts = {}
    if not a.skip_gates:
        for dset in datasets:
            for split in SPLITS:
                new = os.path.join(a.staged, dset, f"{split}_944.parquet")
                old = os.path.join(a.old_root, dset, f"{split}_924.parquet")
                rep = os.path.join(gates_dir, f"{dset}_{split}.json")
                print(f"== gate {dset}/{split}", flush=True)
                r = subprocess.run(
                    ["python3", GATE, "--new", new, "--old", old, "--report", rep],
                    capture_output=True, text=True,
                )
                sys.stdout.write(r.stdout[-600:])
                ok = r.returncode == 0
                verdicts[f"{dset}/{split}"] = "PASS" if ok else "FAIL"
                if not ok:
                    print(f"ABORT: gate FAIL on {dset}/{split} (see {rep}); "
                          f"stderr: {r.stderr[-300:]}")
                    return 1
        print("ALL 21 GATES PASS")
    else:
        for dset in datasets:
            for split in SPLITS:
                rep = os.path.join(gates_dir, f"{dset}_{split}.json")
                with open(rep) as f:
                    v = json.load(f)["verdict"]
                if v != "PASS":
                    print(f"ABORT: stored gate report {rep} is {v}")
                    return 1
                verdicts[f"{dset}/{split}"] = v

    # install: copy views + gate reports + join report into the canonical layout
    os.makedirs(a.out, exist_ok=True)
    files = {}
    for dset in datasets:
        od = os.path.join(a.out, dset)
        os.makedirs(od, exist_ok=True)
        for split in SPLITS:
            src = os.path.join(a.staged, dset, f"{split}_944.parquet")
            dst = os.path.join(od, f"{split}_944.parquet")
            print(f"install {dst}", flush=True)
            shutil.copy2(src, dst)
            import pyarrow.parquet as pq
            files[f"{dset}/{split}_944.parquet"] = {
                "sha256": sha256_file(dst),
                "bytes": os.path.getsize(dst),
                "rows": pq.ParquetFile(dst).metadata.num_rows,
                "gate": verdicts[f"{dset}/{split}"],
            }
    for extra in ("_JOIN_REPORT.json",):
        src = os.path.join(a.staged, extra)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(a.out, extra))
    if os.path.isdir(gates_dir):
        shutil.copytree(gates_dir, os.path.join(a.out, "gates"), dirs_exist_ok=True)

    manifest = {
        "dataset": "bigcodec 944 split views (SOTA-944 P1 leg 3, bf944 tier-matched fleet wave)",
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "regime": "folded720append2 (944 = 924 ++ append2-20; append2_dst_activity OFF per P1.5 adjudication)",
        "build_commit_zensim": a.zensim_commit,
        "build_commit_zenmetrics": a.zenmetrics_commit,
        "worker_image": a.image,
        "provenance": {
            "recipe": "zensim benchmarks/backfill944_bigcodec_2026-08-02.md",
            "declare": "zenmetrics scripts/jobsys/declare_bf944_tiered.py (SIMD-tier-matched: "
                       "v4 226,818 / v4x 231,836 / neon 31,519 cells by bf924 ledger attribution)",
            "assemble": "zensim scripts/v_next/fleet_blob_assemble_944.py",
            "join": "zensim scripts/canonical_corpus/tbig_join_944.py (924-view row order; "
                    "non-feature columns byte-carried from the frozen 924 views)",
            "gate": "scripts/canonical_corpus/gate_backfill944.py per view (G-BF1 f0..f923 "
                    "bitwise row-for-row vs the 924 view + structural checks; G-BF2 carried columns exact)",
        },
        "regime_purity": "NEVER column-mix these 944 rows with 924/720/v1 parquets.",
        "files": files,
    }
    mp = os.path.join(a.out, "_MANIFEST.json")
    with open(mp, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {mp} ({len(files)} views)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
