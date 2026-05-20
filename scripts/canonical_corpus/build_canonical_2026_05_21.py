#!/usr/bin/env python3
"""V11 canonical-2026-05-21 build: incremental on canonical-2026-05-18.

Changes vs canonical-2026-05-18:

  1. NEW: `train/cid22_train.parquet` — ssim2-anchored CID22 training-only
     subset, 17,611 pairs from 201 non-validation refs. Built by
     `v11_extract_cid22_train.py --assemble`.

  2. RENAME: in `train/safesyn.parquet`, `train/kadid.parquet`,
     `train/tid.parquet` — column `mix_cv55_iw44` → `mix_cv55_iw45`.
     The typo persisted from the 2026-05-18-v24 source. The column
     CONTENT is unchanged (already populated with the correct
     iw45-weight mix values); only the column NAME is corrected. This
     resolves the v11_retrain_brief naming-collision bug between
     safesyn/kadid/tid (typo) and konjnd-dense (correct).

  3. UNCHANGED (hard-linked from canonical-2026-05-18):
       train/konjnd-dense.parquet, train/cvvdp_iwssim_LARGE.parquet,
       all val/*.parquet, all scores/*.parquet.

  4. NEW (Phase 2 step 5 follow-up): `train/pipal.parquet` if PIPAL
     feature extraction has completed. Optional — only included when the
     PIPAL feature parquet exists.

Layout matches canonical-2026-05-18 with the additions above. A fresh
_MANIFEST.json is emitted at the root.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

OLD_ROOT = Path("/mnt/v/zen/zensim-training/canonical-2026-05-18")
NEW_ROOT = Path("/mnt/v/zen/zensim-training/canonical-2026-05-21")

# Columns to rename safesyn/kadid/tid (typo → correct)
TYPO_RENAME = {"mix_cv55_iw44": "mix_cv55_iw45"}

# All canonical target columns (in canonical schema order, AFTER rename).
ALL_TARGETS = [
    "cvvdp_score", "cvvdp_log_norm",
    "iwssim", "iwssim_log_norm",
    "ssim2_gpu", "ssim2_log_norm",
    "pjnd_target",
    "mix_cv25_iw75", "mix_cv30_iw70", "mix_cv35_iw65", "mix_cv40_iw60",
    "mix_cv45_iw55", "mix_cv50_iw50", "mix_cv55_iw45",
    "mix_cv60_iw40", "mix_cv65_iw35", "mix_cv70_iw30", "mix_cv75_iw25",
    "mix_cv33_iw33_sm33",
    "mix_target",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def rename_typo_columns(src: Path, dest: Path):
    """Read source parquet, rename TYPO_RENAME columns, write to dest with
    the canonical schema metadata preserved + a new 'schema_note' added."""
    print(f"\n--- rename_typo_columns {src.name} → {dest.name}")
    tbl = pq.read_table(str(src))
    new_names = []
    renamed = False
    for n in tbl.schema.names:
        if n in TYPO_RENAME:
            new_names.append(TYPO_RENAME[n])
            renamed = True
        else:
            new_names.append(n)
    if renamed:
        tbl = tbl.rename_columns(new_names)
        print(f"  renamed {list(TYPO_RENAME.keys())} -> {list(TYPO_RENAME.values())}")
    else:
        print("  no rename needed")

    # Reorder so mix_cv55_iw45 lands in the canonical position
    ordered = []
    for c in ["ref_basename", "human_score"]:
        if c in tbl.schema.names:
            ordered.append(c)
    for c in ALL_TARGETS:
        if c in tbl.schema.names:
            ordered.append(c)
    for n in tbl.schema.names:
        if n not in ordered:
            ordered.append(n)
    tbl = tbl.select(ordered)

    # Preserve + augment metadata
    src_meta = tbl.schema.metadata or {}
    meta = {k.decode(): v.decode() for k, v in src_meta.items()}
    meta["schema_version"] = "canonical-2026-05-21.v1"
    note = meta.get("schema_note", "")
    fix_note = "v11 rename: mix_cv55_iw44 → mix_cv55_iw45 (typo fixed in canonical-2026-05-21)"
    meta["schema_note"] = f"{note}; {fix_note}" if note else fix_note

    new_meta = {k.encode(): v.encode() for k, v in meta.items()}
    schema_with_meta = tbl.schema.with_metadata(new_meta)
    tbl = tbl.replace_schema_metadata(schema_with_meta.metadata)

    dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, str(dest), compression="zstd", compression_level=15)
    sz = dest.stat().st_size
    print(f"  WROTE {dest} ({tbl.num_rows} rows × {tbl.num_columns} cols, {sz/1e6:.1f} MB)")


def copy_unchanged(src: Path, dest: Path):
    """Hard-copy (not symlink) so the canonical-2026-05-21 dir is
    self-contained for R2 sync and downstream consumers."""
    print(f"\n--- copy_unchanged {src.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    # Use shutil.copy2 to preserve mtime; same content sha256 as source
    shutil.copy2(str(src), str(dest))
    print(f"  COPIED {dest} ({dest.stat().st_size/1e6:.1f} MB)")


def emit_manifest():
    manifest_entries = []
    for p in sorted(NEW_ROOT.rglob("*.parquet")):
        if "_workspace" in str(p):
            continue
        rel = p.relative_to(NEW_ROOT).as_posix()
        meta = pq.read_metadata(str(p))
        schema = pq.read_schema(str(p))
        target_cols = [n for n in schema.names if n in ALL_TARGETS or n == "human_score"]
        feature_cols = [n for n in schema.names if n.startswith("f") and n[1:].isdigit()]
        kv = schema.metadata or {}
        kvd = {k.decode(): v.decode() for k, v in kv.items()}
        entry = {
            "path": rel,
            "rows": meta.num_rows,
            "columns": len(schema.names),
            "num_features": len(feature_cols),
            "target_columns": target_cols,
            "byte_size": p.stat().st_size,
            "sha256": sha256_file(p),
            "canonical_metadata": kvd,
        }
        manifest_entries.append(entry)
    out = NEW_ROOT / "_MANIFEST.json"
    with open(out, "w") as f:
        json.dump({
            "canonical_date": "2026-05-21",
            "schema_version": "canonical-2026-05-21.v1",
            "schema_changes_vs_2026_05_18": [
                "Renamed mix_cv55_iw44 → mix_cv55_iw45 in train/{safesyn,kadid,tid}.parquet (typo fix)",
                "Added train/cid22_train.parquet (ssim2-anchored CID22 training-only subset, 17,611 pairs)",
                "Added train/pipal.parquet (Phase 2 step 5) — only if PIPAL extraction has completed",
                "All other entries hard-copied from canonical-2026-05-18 (content unchanged)",
            ],
            "entries": manifest_entries,
        }, f, indent=2)
    print(f"\nWROTE _MANIFEST.json ({len(manifest_entries)} entries)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-copy", action="store_true",
                    help="Skip copy of unchanged konjnd-dense/LARGE/val/scores (useful for iteration)")
    args = ap.parse_args()

    NEW_ROOT.mkdir(parents=True, exist_ok=True)
    (NEW_ROOT / "train").mkdir(parents=True, exist_ok=True)
    (NEW_ROOT / "val").mkdir(parents=True, exist_ok=True)
    (NEW_ROOT / "scores").mkdir(parents=True, exist_ok=True)

    # 1) Apply typo rename to safesyn / kadid / tid
    for name in ["safesyn.parquet", "kadid.parquet", "tid.parquet"]:
        rename_typo_columns(OLD_ROOT / "train" / name, NEW_ROOT / "train" / name)

    # 2) Copy unchanged train parquets (no rename needed — these already use mix_cv55_iw45 or don't use it)
    if not args.skip_copy:
        for name in ["konjnd-dense.parquet", "cvvdp_iwssim_LARGE.parquet"]:
            copy_unchanged(OLD_ROOT / "train" / name, NEW_ROOT / "train" / name)
        # Copy val + scores
        for sub in ["val", "scores"]:
            for src in (OLD_ROOT / sub).iterdir():
                if src.suffix == ".parquet":
                    copy_unchanged(src, NEW_ROOT / sub / src.name)

    # 3) cid22_train.parquet is built separately by v11_extract_cid22_train.py --assemble
    cid22_train = NEW_ROOT / "train" / "cid22_train.parquet"
    if cid22_train.exists():
        print(f"\nNOTE: cid22_train.parquet already present at {cid22_train}")
    else:
        print(f"\nNOTE: cid22_train.parquet NOT YET BUILT — run v11_extract_cid22_train.py --assemble first")

    # 4) PIPAL is optional
    pipal = NEW_ROOT / "train" / "pipal.parquet"
    if pipal.exists():
        print(f"NOTE: pipal.parquet present at {pipal}")
    else:
        print(f"NOTE: pipal.parquet not built (Phase 2 step 5 in flight)")

    emit_manifest()


if __name__ == "__main__":
    main()
