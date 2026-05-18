#!/usr/bin/env python3
"""Build training parquets with `ensemble_teacher` column.

For each canonical training parquet, joins three sidecar TSVs of
ensemble_score_rows output (balanced bake, compression bake, classifier
bake) to produce the Ensemble's per-row teacher score:

    routed_raw  = compression_score if classifier_logit > 0 else balanced_score
    teacher     = soft_clamp(routed_raw) / 100   (so target-scale 100.0 → score)

soft_clamp matches `zensim::metric::soft_clamp_score`:
    soft_clamp(x) = 100 / (1 + exp(-(x - 50) / 20))

Writes new parquet to /mnt/v/zen/zensim-training/2026-05-18-distill-ensemble/
with columns: `human_score` (copy of source), `ensemble_teacher` (new),
`pjnd_target` (copy if present), and `f0..fN-1` features.
"""
import sys
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC_DIR = Path("/mnt/v/zen/zensim-training/canonical-2026-05-18/train")
SCORES_DIR = Path("/tmp/exp_distill_ensemble_scores")
OUT_DIR = Path("/mnt/v/zen/zensim-training/2026-05-18-distill-ensemble")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def soft_clamp(x: np.ndarray) -> np.ndarray:
    # 100 / (1 + exp(-(x - 50)/20))
    z = -(x - 50.0) / 20.0
    z = np.clip(z, -50.0, 50.0)  # avoid overflow in exp
    return 100.0 / (1.0 + np.exp(z))


def load_scores(corpus: str, kind: str) -> np.ndarray:
    p = SCORES_DIR / f"{corpus}_{kind}.tsv"
    arr = np.loadtxt(p, delimiter="\t", skiprows=1, usecols=2)
    return arr


def process(corpus: str) -> None:
    src = SRC_DIR / f"{corpus}.parquet"
    print(f"[{corpus}] reading {src}")
    table = pq.read_table(src)
    n = table.num_rows
    print(f"[{corpus}]   n_rows = {n}")

    bal = load_scores(corpus, "bal")
    cmp_ = load_scores(corpus, "cmp")
    clf = load_scores(corpus, "clf")
    assert len(bal) == n, f"{corpus}: bal {len(bal)} != {n}"
    assert len(cmp_) == n, f"{corpus}: cmp {len(cmp_)} != {n}"
    assert len(clf) == n, f"{corpus}: clf {len(clf)} != {n}"

    routed_raw = np.where(clf > 0.0, cmp_, bal)
    teacher_score = soft_clamp(routed_raw)
    teacher_unit = teacher_score / 100.0  # in [0,1] for target-scale=100

    n_to_compression = int((clf > 0.0).sum())
    print(
        f"[{corpus}]   routed: balanced={n - n_to_compression}, "
        f"compression={n_to_compression} "
        f"(frac_cmp={n_to_compression / n:.4f})"
    )
    print(
        f"[{corpus}]   teacher stats: min={teacher_score.min():.3f} "
        f"max={teacher_score.max():.3f} mean={teacher_score.mean():.3f} "
        f"std={teacher_score.std():.3f}"
    )

    # Build the new parquet: keep only what trainer needs.
    cols = table.column_names
    feature_cols = [c for c in cols if c.startswith("f") and c[1:].isdigit()]
    keep_cols = ["human_score"] + feature_cols
    if "pjnd_target" in cols:
        keep_cols.append("pjnd_target")
    sub = table.select(keep_cols)
    teacher_array = pa.array(teacher_unit.astype(np.float64), type=pa.float64())
    sub = sub.append_column("ensemble_teacher", teacher_array)

    out_path = OUT_DIR / f"{corpus}.parquet"
    print(f"[{corpus}]   writing {out_path}")
    pq.write_table(sub, out_path, compression="zstd", compression_level=15)


def main() -> int:
    for corpus in ["safesyn", "kadid", "tid", "konjnd-dense", "cvvdp_iwssim_LARGE"]:
        process(corpus)
    print("\nDone.")
    print(f"Output parquets in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
