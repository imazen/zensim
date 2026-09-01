#!/usr/bin/env python3
"""a4bkon lane (2026-09-01): build the `ttbig` HYA-teacher-target table K2/K3
need — the leg A4b's own run dropped (wave_r4_2026-09-01.md §3.0.2 point 3:
"no HYA-graft exists for tbig yet"; §24.2 registers this plan before it runs).

Why this is NOT a duplicate of `scripts/canonical_corpus/build_teacher944.py
--graft-from` (the owner for "carry a teacher target onto a same-row feature
table"): that owner requires EXACT row-count and row-order match between
`--graft-from` and `--graft-features`, by design — a silent reorder would be
exactly the wrong-regime-style defect its own docstring warns about. Here the
two tables that exist do NOT share a row count:

    tbig_pools944.parquet        (era-1 pools-944 features)   208,169 rows
    tbig_944_200k_pure.parquet   (wave-r4 root's own view)     192,714 rows

so the owner's positional graft cannot run. This script does the one thing
the owner does not: a KEY join (by `encoded_filename`, the same key
`derive_recipe_views.py` already uses to attach that column) from the wider
table onto the narrower table's own row order. Everything else — the teacher
forward itself — goes through the REAL owner (`bake_dial_refit predict`),
not a re-implementation.

Steps (see §24.2 for the full registration):
  1. Forward HYA_w084 over tbig_pools944.parquet via `bake_dial_refit predict`
     (the owner call `build_teacher944.py`'s own `predict()` helper wraps).
  2. Apply the ALREADY-FIT affine from the existing safesyn teacher build
     (read from its manifest, not refit) so tsafesyn/ttbig share one scale.
  3. Key-join the resulting (encoded_filename, hya_target) onto
     tbig_944_200k_pure.parquet's row order.
  4. Gate: every target row's key must resolve, or ABORT. Nothing padded,
     nothing best-effort joined (derive_recipe_views.py's own discipline).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TBIG_POOLS = Path(
    "/mnt/v/zen/zensim-training/wlin7-pools944-2026-08-30/tbig_pools944.parquet"
)
TARGET_VIEW = Path(
    "/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01/recipe_views/"
    "tbig_944_200k_pure.parquet"
)
AFFINE_MANIFEST = Path(
    "/mnt/v/output/zensim/hybrid-2026-09-01/distill/teach/_MANIFEST.json"
)
TEACHER_MEMBERS = [
    "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W10L9PH_s4004_packed.bin",
    "/mnt/v/output/zensim/wlin7b-2026-08-30/arms/Q7b_pools_g0.2_a0.2_b0.97.bin",
]
TEACHER_WEIGHTS = "0.84,0.16"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--bdr",
        type=Path,
        default=Path("/mnt/v/zen/cargo-targets/waver4/release/bake_dial_refit"),
        help="bake_dial_refit binary (the predict owner) -- the wave-r4 build,"
        " so the forward runs the SAME binary that produced every other"
        " wave-r4 artifact",
    )
    ap.add_argument("--work-dir", type=Path, default=Path("/mnt/v/output/zensim/a4bkon-2026-09-01/ttbig_build"))
    a = ap.parse_args()
    a.work_dir.mkdir(parents=True, exist_ok=True)

    for p in (TBIG_POOLS, TARGET_VIEW, AFFINE_MANIFEST, a.bdr, *[Path(m) for m in TEACHER_MEMBERS]):
        if not p.exists():
            print(f"ABORT: missing required input {p}", file=sys.stderr)
            return 2

    # 1. Forward HYA_w084 over tbig_pools944.parquet (the owner call).
    raw_tsv = a.work_dir / "tbig_pools944_hya_raw.tsv"
    if raw_tsv.exists():
        print(f"[reuse] {raw_tsv}")
    else:
        cmd = [
            str(a.bdr), "predict",
            "--ensemble", ",".join(TEACHER_MEMBERS),
            "--ensemble-weights", TEACHER_WEIGHTS,
            "--corpus", str(TBIG_POOLS),
            "--out", str(raw_tsv),
        ]
        print("$", " ".join(cmd))
        subprocess.run(cmd, check=True)

    raw = np.loadtxt(raw_tsv, delimiter="\t", skiprows=1, usecols=1)
    src = pq.read_table(TBIG_POOLS, columns=["encoded_filename"])
    ef_src = src["encoded_filename"].combine_chunks().to_pylist()
    if len(raw) != len(ef_src):
        print(f"ABORT: forward has {len(raw)} preds, source has {len(ef_src)} keys", file=sys.stderr)
        return 3
    print(f"forward: {len(raw)} rows, raw mean {raw.mean():.6f} [{raw.min():.6f}, {raw.max():.6f}]")

    # 2. Apply the ALREADY-FIT affine (read, not refit).
    affine_man = json.loads(AFFINE_MANIFEST.read_text())
    lo, hi = affine_man["affine"]
    print(f"affine (from {AFFINE_MANIFEST}): lo={lo!r} hi={hi!r}")
    target = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
    clip_frac = float(((raw < lo) | (raw > hi)).mean())
    key_to_target = dict(zip(ef_src, target))
    if len(key_to_target) != len(ef_src):
        print(
            f"NOTE: {len(ef_src)} source rows collapse to {len(key_to_target)} "
            "unique encoded_filename keys -- duplicates take the LAST row's "
            "value (dict overwrite); checked below for target-side impact.",
        )

    # 3. Key-join onto tbig_944_200k_pure.parquet's own row order.
    dst = pq.read_table(TARGET_VIEW)
    ef_dst = dst["encoded_filename"].combine_chunks().to_pylist()
    missing = [k for k in ef_dst if k not in key_to_target]
    # 4. Gate: refuse partial coverage.
    if missing:
        print(
            f"ABORT: {len(missing)}/{len(ef_dst)} target keys have no match in "
            f"the source forward. First 5 missing: {missing[:5]}",
            file=sys.stderr,
        )
        return 4
    print(f"G-KEYJOIN PASS: {len(ef_dst)}/{len(ef_dst)} target rows resolved")

    y = np.array([key_to_target[k] for k in ef_dst], dtype=np.float64)
    feats = [c for c in dst.column_names if c.startswith("f") and c[1:].isdigit()]
    feats.sort(key=lambda c: int(c[1:]))
    carry = [c for c in ("ref_basename", "encoded_filename", "regime") if c in dst.column_names]
    # human_score is REPLACED (the target-view's own human-target column is
    # not carried through -- ttbig's whole purpose is a different target).
    out_t = dst.select(carry + feats).append_column(
        "human_score", pa.array(y, type=pa.float64())
    )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_t, a.out, compression="zstd", compression_level=7)
    man = {
        "file": str(a.out),
        "sha256": sha256(a.out),
        "rows": out_t.num_rows,
        "mode": "key-join-graft (a4bkon lane extension -- see wave_r4_2026-09-01.md §24.2)",
        "mechanism": (
            "features + row order from --graft-features-equivalent "
            f"({TARGET_VIEW}), teacher target from a FRESH forward of HYA_w084 "
            f"over {TBIG_POOLS} (era-1 pools-944, row-count mismatched with the "
            "target so the owner's positional graft() could not run), joined "
            "by encoded_filename KEY (not position). Affine read from the "
            "existing safesyn teacher build's manifest, not refit, so "
            "tsafesyn/ttbig share one target scale."
        ),
        "teacher_members": TEACHER_MEMBERS,
        "teacher_weights": TEACHER_WEIGHTS,
        "affine": [lo, hi],
        "affine_source": str(AFFINE_MANIFEST),
        "target_source_forward": {"path": str(raw_tsv), "corpus": str(TBIG_POOLS)},
        "features_source": {"path": str(TARGET_VIEW), "sha256": sha256(TARGET_VIEW)},
        "clip_frac": clip_frac,
        "target_mean": float(y.mean()),
        "key_join_coverage": f"{len(ef_dst)}/{len(ef_dst)}",
    }
    (a.out.parent / (a.out.name + "._MANIFEST.json")).write_text(json.dumps(man, indent=1))
    print(f"wrote {a.out} ({out_t.num_rows} rows) + manifest, target_mean={y.mean():.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
