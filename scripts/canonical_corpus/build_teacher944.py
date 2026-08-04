#!/usr/bin/env python3
"""Build the three DISTILLATION TEACHER twins at 944 (SOTA-944 amendment 3 / 6).

A "teacher twin" is one of the campaign's three dense training legs with its
`human_score` column REPLACED by a teacher model's prediction, min-max'd into
[0,1] so the trainer's `--target-scale 100` puts it in the same units as the
real legs. The features are carried verbatim, so the student sees identical
rows under a different target.

    twins: safesyn (ext_safesyn_full) · tbig (tbig_944_200k) · kadis (kadis_944_ssim2_50k)

THE TARGET RULE (frozen; re-derived and verified against the committed EM4
teacher before wave 6 registered it):

    raw        = teacher forward over the twin's feature rows
    lo, hi     = quantile(raw_safesyn, 0.001), quantile(raw_safesyn, 0.999)
    human_score= clip((raw - lo) / (hi - lo), 0, 1)

ONE affine, fit on the safesyn twin, applied to all three. Verification: on the
committed `teacher/safesyn_teacher.tsv` this reproduces the stored manifest
affine [-12.95392379951477, 10.061253767967228] and the stored teacher mean
0.6142450490816594 exactly (total clip fraction 0.2017%).

The teacher forward is NOT re-implemented here: it comes from
`bake_dial_refit predict` (`--bake` or `--ensemble`), the owner of "forward a
bake over a parquet". This script only joins the emitted TSV onto the twin and
writes the parquet — no model math, no averaging.

Usage
-----
    build_teacher944.py --tag ensk2 \
        --members /path/a.bin,/path/b.bin \
        --out-dir /mnt/v/output/zensim/bakes/sota944/teacher_ensk2

`--members` with one entry reproduces the single-teacher (EM4) chain.
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

EXT944 = Path("/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01")

TWINS = {
    "safesyn": EXT944 / "ext_safesyn_full.parquet",
    "tbig": Path("/mnt/v/zen/zensim-training/tbig_944_200k.parquet"),
    "kadis": Path(
        "/mnt/v/zen/zensim-training/kadis-944-2026-08-01/kadis_944_ssim2_50k.parquet"
    ),
}
# The affine is fit on this twin and applied to all three.
AFFINE_TWIN = "safesyn"
QLO, QHI = 0.001, 0.999


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def predict(bdr: Path, members: list[str], corpus: Path, out: Path) -> np.ndarray:
    """Forward the teacher over `corpus` via the canonical owner, return raw preds."""
    if out.exists():
        print(f"  [reuse] {out}")
    else:
        cmd = [str(bdr), "predict", "--corpus", str(corpus), "--out", str(out)]
        if len(members) == 1:
            cmd += ["--bake", members[0]]
        else:
            cmd += ["--ensemble", ",".join(members)]
        print("  $", " ".join(cmd))
        subprocess.run(cmd, check=True)
    return np.loadtxt(out, delimiter="\t", skiprows=1, usecols=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="teacher tag, e.g. ensk2")
    ap.add_argument(
        "--members", required=True, help="comma-separated ZNPR bakes (k=1 → single teacher)"
    )
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument(
        "--bdr",
        type=Path,
        default=Path.home() / "tmp/zensimw6-target/release/bake_dial_refit",
        help="bake_dial_refit binary (the predict owner)",
    )
    a = ap.parse_args()

    members = [m for m in a.members.split(",") if m]
    for m in members:
        if not Path(m).is_file():
            print(f"member not found: {m}", file=sys.stderr)
            return 2
    a.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Forward the teacher over every twin (raw predictions).
    raw: dict[str, np.ndarray] = {}
    for name, path in TWINS.items():
        print(f"[{a.tag}/{name}] forward over {path.name}")
        raw[name] = predict(a.bdr, members, path, a.out_dir / f"{name}_teacher.tsv")

    # 2. ONE affine, fit on the AFFINE_TWIN raw preds.
    ref = raw[AFFINE_TWIN]
    lo, hi = float(np.quantile(ref, QLO)), float(np.quantile(ref, QHI))
    print(f"[{a.tag}] affine (q{QLO}, q{QHI}) on {AFFINE_TWIN}: [{lo!r}, {hi!r}]")

    manifest: dict = {
        "tag": a.tag,
        "teacher_members": members,
        "affine": [lo, hi],
        "affine_rule": f"lo=quantile(raw[{AFFINE_TWIN}],{QLO}) hi=quantile(...,{QHI}); "
        "human_score = clip((raw-lo)/(hi-lo),0,1)",
        "predict_owner": "bake_dial_refit predict (--bake | --ensemble)",
        "files": {},
    }

    # 3. Write each twin with human_score replaced by the teacher target.
    for name, path in TWINS.items():
        tbl = pq.read_table(path)
        n = tbl.num_rows
        assert len(raw[name]) == n, f"{name}: {len(raw[name])} preds vs {n} rows"
        t = np.clip((raw[name] - lo) / (hi - lo), 0.0, 1.0)
        clip_frac = float(((raw[name] < lo) | (raw[name] > hi)).mean())

        feats = [c for c in tbl.column_names if c.startswith("f") and c[1:].isdigit()]
        feats.sort(key=lambda c: int(c[1:]))
        keep = ["ref_basename"] + feats
        sub = tbl.select(keep).append_column(
            "human_score", pa.array(t.astype(np.float64), type=pa.float64())
        )
        outp = a.out_dir / f"{name}_teacher944.parquet"
        print(
            f"[{a.tag}/{name}] rows {n} raw_mean {raw[name].mean()!r} "
            f"teacher_mean {t.mean()!r} clip_frac {clip_frac:.6f} -> {outp}"
        )
        pq.write_table(sub, outp, compression="zstd", compression_level=9)
        manifest["files"][name] = {
            "path": str(outp),
            "source": str(path),
            "rows": n,
            "raw_mean": float(raw[name].mean()),
            "teacher_mean": float(t.mean()),
            "clip_frac": clip_frac,
            "sha256": sha256(outp),
        }

    (a.out_dir / "_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"[{a.tag}] manifest -> {a.out_dir / '_MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
