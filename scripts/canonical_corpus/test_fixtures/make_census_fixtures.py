#!/usr/bin/env python3
"""Generate tiny committed parquet fixtures for the metric-column census CI gate.

Produces three small parquets that exercise the two corruption modes + a clean
control, so `audit_metric_columns.py --fail-on-corruption` can be smoke-tested
in CI without access to the real /mnt/v + R2 canonical parquets.

  corrupt_human_copy.parquet  — iwssim is a verbatim copy of human_score (Mode A)
  corrupt_ref_misjoin.parquet — ssim2_gpu constant within every ref group (Mode B)
  clean.parquet               — iwssim & ssim2_gpu vary per pair; correlated-but-
                                not-identical to human_score (passes)

Re-run to regenerate (deterministic seed). Output is committed — each file is a
few KB. Keep the row count small.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parent
N_REFS = 6
N_DIST = 8  # distortions per reference
RNG = np.random.default_rng(20260525)


def _base():
    refs, hs = [], []
    for r in range(N_REFS):
        base = RNG.uniform(20, 90)
        for d in range(N_DIST):
            refs.append(f"ref{r:02d}.png")
            # human_score degrades with distortion index, plus noise
            hs.append(float(np.clip(base - d * 6 + RNG.normal(0, 1.5), 0, 100)))
    return refs, np.asarray(hs, dtype=float)


def _write(name: str, cols: dict[str, np.ndarray | list]):
    arrays, names = [], []
    for k, v in cols.items():
        names.append(k)
        if isinstance(v, list):
            arrays.append(pa.array(v))
        else:
            arrays.append(pa.array(np.asarray(v, dtype=float)))
    tbl = pa.Table.from_arrays(arrays, names=names)
    out = OUT / name
    pq.write_table(tbl, str(out), compression="zstd", compression_level=15)
    print(f"wrote {out} ({tbl.num_rows} rows, {out.stat().st_size} bytes)")


def main():
    refs, hs = _base()
    n = len(hs)

    # Mode A: iwssim ≡ human_score (the leaked mock).
    _write("corrupt_human_copy.parquet", {
        "ref_basename": refs,
        "human_score": hs,
        "iwssim": hs.copy(),  # verbatim copy → HUMAN-COPY
    })

    # Mode B: ssim2_gpu broadcast — one mean value per ref group.
    by_ref_mean = {}
    for r in set(refs):
        by_ref_mean[r] = float(np.mean([hs[i] for i in range(n) if refs[i] == r]))
    ssim2_broadcast = np.asarray([by_ref_mean[r] for r in refs], dtype=float)
    _write("corrupt_ref_misjoin.parquet", {
        "ref_basename": refs,
        "human_score": hs,
        "ssim2_gpu": ssim2_broadcast,  # constant within each ref → REF-MISJOIN
    })

    # Clean control: both metrics vary per pair, correlated-but-not-identical.
    iwssim_clean = np.clip(hs / 100.0 + RNG.normal(0, 0.02, n), 0, 1)
    ssim2_clean = np.clip(hs + RNG.normal(0, 4, n), -50, 100)
    _write("clean.parquet", {
        "ref_basename": refs,
        "human_score": hs,
        "iwssim": iwssim_clean,
        "ssim2_gpu": ssim2_clean,
    })


if __name__ == "__main__":
    main()
