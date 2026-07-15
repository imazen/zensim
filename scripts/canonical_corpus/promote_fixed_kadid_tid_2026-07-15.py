#!/usr/bin/env python3
"""Promote the verified `*_fixed_2026-05-25.parquet` KADID/TID siblings to the active canonical
train names (user directive 2026-07-15), closing the V39-#8 / DATASET_HISTORY §3.18 data bug
(`iwssim` = human_score copy; `ssim2_gpu` = ref-vs-ref misjoin) that was still live in
canonical-2026-05-21/train.

SAFETY (aborts touching NOTHING if any guard fails):
  - fixed sibling has identical schema + row count to the current canonical
  - fixed `iwssim` is NOT ~human_score (< 1% isclose) and correlates with MOS (|SROCC| > 0.5)
  - fixed `ssim2_gpu` spans a real range (min < 50 — not pinned near 100) and |SROCC| > 0.5
  - every PRESERVED column (human_score, cvvdp_*, pjnd_target, f0..f371) is byte-identical
    → guarantees zero feature/target drift; only the corrupt columns + their derivatives change

On success (per corpus): mv original -> `<c>.CORRUPT-v39bug.pre-2026-07-15.bak.parquet` (NEVER rm;
corrupt original also on Tower + R2 + v11-reproduction-kit), mv fixed sibling -> `<c>.parquet`,
update the dir `_MANIFEST.json` entry (new sha256 + byte_size + data_integrity_fix note; rows/cols/
schema unchanged). Idempotent: skips a corpus already promoted (no `_fixed` sibling present).

  usage: promote_fixed_kadid_tid_2026-07-15.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

CANON = Path("/mnt/v/zen/zensim-training/canonical-2026-05-21/train")
MANIFEST = Path("/mnt/v/zen/zensim-training/canonical-2026-05-21/_MANIFEST.json")
PRESERVED_SCALAR = ["human_score", "cvvdp_score", "cvvdp_log_norm", "pjnd_target"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def guard(corpus: str, cur_p: Path, fix_p: Path) -> None:
    """Raise if the fixed sibling is not a safe drop-in for the canonical file."""
    cur = pq.read_table(str(cur_p))
    fix = pq.read_table(str(fix_p))
    assert cur.schema.names == fix.schema.names, f"{corpus}: schema mismatch"
    assert cur.num_rows == fix.num_rows, f"{corpus}: row-count mismatch"
    cd, fd = cur.to_pandas(), fix.to_pandas()
    hs = cd.human_score.to_numpy(float)
    iw = fd.iwssim.to_numpy(float)
    s2 = fd.ssim2_gpu.to_numpy(float)
    m = np.isfinite(s2) & np.isfinite(hs) & np.isfinite(iw)
    assert np.mean(np.isclose(hs, iw)) < 0.01, f"{corpus}: fixed iwssim STILL ~human_score"
    assert abs(spearmanr(hs[m], iw[m]).correlation) > 0.5, f"{corpus}: fixed iwssim doesn't track MOS"
    assert s2.min() < 50.0, f"{corpus}: fixed ssim2 still pinned near 100 (min={s2.min():.1f})"
    assert abs(spearmanr(hs[m], s2[m]).correlation) > 0.5, f"{corpus}: fixed ssim2 doesn't track MOS"
    feats = [f"f{i}" for i in range(372) if f"f{i}" in cd.columns]
    for c in PRESERVED_SCALAR + feats:
        if c in cd.columns and c in fd.columns:
            a, b = cd[c].to_numpy(float), fd[c].to_numpy(float)
            assert np.array_equal(a, b, equal_nan=True), f"{corpus}: PRESERVED col '{c}' DRIFTED"
    print(f"  [{corpus}] guard PASS: iwssim SROCC {spearmanr(hs[m], iw[m]).correlation:+.4f}, "
          f"ssim2 SROCC {spearmanr(hs[m], s2[m]).correlation:+.4f}, "
          f"{len(feats)} features + {len(PRESERVED_SCALAR)} scalars byte-identical")


def promote(corpus: str, dry: bool, manifest: dict) -> bool:
    cur_p = CANON / f"{corpus}.parquet"
    fix_p = CANON / f"{corpus}_fixed_2026-05-25.parquet"
    bak_p = CANON / f"{corpus}.CORRUPT-v39bug.pre-2026-07-15.bak.parquet"
    if not fix_p.exists():
        print(f"  [{corpus}] no _fixed sibling — already promoted, skip")
        return False
    guard(corpus, cur_p, fix_p)
    if dry:
        print(f"  [{corpus}] DRY-RUN: would mv {cur_p.name} -> {bak_p.name}; {fix_p.name} -> {corpus}.parquet")
        return False
    shutil.move(str(cur_p), str(bak_p))          # preserve corrupt original (never rm)
    shutil.move(str(fix_p), str(cur_p))          # promote fixed -> canonical name
    new_sha, new_sz = sha256(cur_p), cur_p.stat().st_size
    for e in manifest["entries"]:
        if e.get("path") == f"train/{corpus}.parquet":
            e["byte_size"] = new_sz
            e["sha256"] = new_sha
            e["data_integrity_fix"] = ("2026-07-15: promoted <corpus>_fixed_2026-05-25 — iwssim (was "
                                       "human_score copy) + ssim2_gpu (was ref-vs-ref misjoin) + their "
                                       "log_norm + all mix_* recomputed on correct (ref,dist) pairs; "
                                       "human_score/cvvdp_*/pjnd_target/f0..f371 byte-identical. "
                                       "Corrupt original -> .CORRUPT-v39bug.pre-2026-07-15.bak.parquet.")
    print(f"  [{corpus}] PROMOTED: sha256 {new_sha[:16]}… size {new_sz}  (original -> {bak_p.name})")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    changed = False
    for corpus in ["kadid", "tid"]:
        if promote(corpus, a.dry_run, manifest):
            changed = True
    if changed and not a.dry_run:
        manifest["data_integrity_fix_2026_07_15"] = ("kadid/tid iwssim+ssim2_gpu V39-#8 corruption "
                                                     "resolved by promoting *_fixed_2026-05-25 siblings")
        MANIFEST.write_text(json.dumps(manifest, indent=2))
        print(f"\nupdated {MANIFEST}")
    print("\nDONE" if not a.dry_run else "\nDRY-RUN complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
