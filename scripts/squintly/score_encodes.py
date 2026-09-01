#!/usr/bin/env python3
"""Forward each adjudication candidate over its NATIVE-regime feature table.

One job: turn `(bake, feature parquet)` into `(encoded_filename -> score)`,
using the owner (`predict_features_with_bake`) for every forward pass. No stat
math, no pair logic, no selection — those live in `mine_adjudication_stimuli.py`.

WHY A SEPARATE PASS. A 944-input bake scored against the wrong 944 regime gets
a plausible-looking number back with no warning (zensim CLAUDE.md, the
`--regime 944` hazard). The regime is therefore a REQUIRED, EXPLICIT argument
here and each candidate is pinned to the table its trainer actually read:

    W10L9PH_s4004_packed  944  folded720append2       tbig_944_200k_pure.parquet
    Q7b_pools_...         944  folded720append2pools  tbig_pools944.parquet

Both tables carry `encoded_filename`, which is the join key to the bytes on
disk and to the ssim2 sidecar. Rows are streamed in batches so a 944 x 200k
matrix never has to be resident.

Usage:
    score_encodes.py --spec candidates.json --out-dir <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
# The owner. Never re-implement a bake forward pass in Python.
DEFAULT_FORWARD_BIN = Path.home() / "tmp/squintly-prep/bin/predict_features_with_bake"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def forward(bin_path: Path, bake: Path, mat: np.ndarray, scratch: Path) -> np.ndarray:
    """Shell one batch through the owner. `mat` is (rows, n_features) f32."""
    assert mat.dtype == np.float32 and mat.flags["C_CONTIGUOUS"]
    blob = struct.pack("<II", mat.shape[1], mat.shape[0]) + mat.tobytes(order="C")
    scratch.write_bytes(blob)
    p = subprocess.run(
        [str(bin_path), "--bake", str(bake), "--features-file", str(scratch)],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise SystemExit(
            f"predict_features_with_bake rc={p.returncode} on {bake}\n{p.stderr[-4000:]}"
        )
    out = np.array([float(x) for x in p.stdout.split()], dtype=np.float64)
    if out.shape[0] != mat.shape[0]:
        raise SystemExit(
            f"forward returned {out.shape[0]} scores for {mat.shape[0]} rows ({bake})"
        )
    return out


def score_one(
    name: str,
    bake: Path,
    table: Path,
    n_features: int,
    forward_bin: Path,
    out_dir: Path,
    batch_size: int,
    scratch: Path,
) -> dict:
    cols = [f"f{i}" for i in range(n_features)]
    pf = pq.ParquetFile(table)
    keys: list[str] = []
    scores: list[float] = []
    t0 = time.time()
    n_batches = 0
    zero_block_max = 0.0
    for b in pf.iter_batches(batch_size=batch_size, columns=cols + ["encoded_filename"]):
        mat = np.empty((b.num_rows, n_features), dtype=np.float32)
        for i, c in enumerate(cols):
            mat[:, i] = np.asarray(b[c], dtype=np.float32)
        # Regime fingerprint: the f156..f371 block is structurally zero under
        # `folded720append2` and LIVE under `folded720append2pools`. Recorded so
        # a table/bake regime mismatch is visible in the manifest rather than
        # silently producing plausible numbers.
        if n_features > 372:
            zero_block_max = max(zero_block_max, float(np.abs(mat[:, 156:372]).max()))
        scores.extend(forward(forward_bin, bake, mat, scratch).tolist())
        keys.extend(b["encoded_filename"].to_pylist())
        n_batches += 1
        if n_batches % 20 == 0:
            print(
                f"  [{name}] {len(keys)} rows  {time.time() - t0:.0f}s",
                flush=True,
            )
    out = out_dir / f"scores_{name}.parquet"
    pq.write_table(
        pa.table({"encoded_filename": pa.array(keys), "score": pa.array(scores)}),
        out,
        compression="zstd",
    )
    rec = {
        "model": name,
        "bake": str(bake),
        "bake_sha256": sha256_file(bake),
        "feature_table": str(table),
        "n_features": n_features,
        "rows": len(keys),
        "f156_371_absmax": zero_block_max,
        "regime_fingerprint": (
            "zero-block (folded720append2-like)"
            if zero_block_max == 0.0
            else "live-pools (folded720append2pools-like)"
        ),
        "out": str(out),
        "out_sha256": sha256_file(out),
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"  [{name}] DONE {rec['rows']} rows in {rec['elapsed_s']}s -> {out}", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--forward-bin", type=Path, default=DEFAULT_FORWARD_BIN)
    ap.add_argument("--batch-size", type=int, default=20000)
    a = ap.parse_args()

    if not a.forward_bin.exists():
        raise SystemExit(
            f"forward owner not found at {a.forward_bin}\n"
            "build it: cargo build --release -p zensim-validate --bin predict_features_with_bake"
        )
    a.out_dir.mkdir(parents=True, exist_ok=True)
    scratch = a.out_dir / ".blob.bin"

    spec = json.loads(a.spec.read_text())
    recs = []
    for c in spec["candidates"]:
        print(f"[score] {c['name']}", flush=True)
        recs.append(
            score_one(
                c["name"],
                Path(c["bake"]),
                Path(c["feature_table"]),
                int(c["n_features"]),
                a.forward_bin,
                a.out_dir,
                a.batch_size,
                scratch,
            )
        )
    scratch.unlink(missing_ok=True)
    man = a.out_dir / "_MANIFEST_scores.json"
    man.write_text(
        json.dumps(
            {
                "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "forward_owner": str(a.forward_bin),
                "spec": json.loads(a.spec.read_text()),
                "records": recs,
            },
            indent=1,
        )
    )
    print(f"[score] manifest -> {man}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
