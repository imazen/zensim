#!/usr/bin/env python3
"""Build the canonical training/validation parquet set.

Consolidates sprawling /mnt/v/zen/zensim-training/2026-05-{15,17,18}-* dirs
into ONE canonical layout at /mnt/v/zen/zensim-training/canonical-2026-05-18/.

Layout:
  train/safesyn.parquet           - 196,086 rows x 372 features, all targets
  train/kadid.parquet             - 10,125 rows x 372 features, all targets + mix_cv33_iw33_sm33
  train/tid.parquet               - 3,000 rows x 372 features, all targets + mix_cv33_iw33_sm33
  train/konjnd-dense.parquet      - 20,160 rows x 372 features, mix targets + pjnd_target
  train/cvvdp_iwssim_LARGE.parquet - 73,300 rows x 300 features, cvvdp + iwssim + mix_cv40_iw60
  val/cid22.parquet               - 4,292 rows x 372 features, human MCOS
  val/kadid.parquet               - 10,125 rows x 372 features, KADID DMOS
  val/tid.parquet                 - 3,000 rows x 372 features, TID MOS
  val/konjnd.parquet              - 1,008 rows x 372 features, KonJND PJND anchor
  val/aic3.parquet                - 600 rows x 372 features, AIC-3 CTC JND
  val/aic4.parquet                - 300 rows x 372 features, AIC-4 sample reconstructed JND
  scores/cvvdp_imazen_v0_0_1.parquet  - 1,169,500 raw cvvdp scores
  scores/iwssim_imazen.parquet         - 75,300 raw iwssim scores
  scores/ssim2_imazen.parquet          - 55,000 raw ssim2 scores

CANONICAL SCHEMA (training parquets):
  ref_basename:str       - source image identifier
  human_score:f64        - per-corpus native anchor (kadid DMOS / tid MOS / mix synthetic target)
  cvvdp_score:f64        - CVVDP raw score (null if not available)
  cvvdp_log_norm:f64     - CVVDP log-normalized
  iwssim:f64             - IW-SSIM raw
  iwssim_log_norm:f64    - IW-SSIM log-normalized
  ssim2_gpu:f64          - SSIMULACRA2 GPU score (null if not available)
  ssim2_log_norm:f64     - SSIM2 log-normalized
  pjnd_target:f64        - KonJND PJND threshold (null except for konjnd-dense)
  mix_cv25_iw75 ... mix_cv75_iw25 : multi-target weighted columns (cvvdp/iwssim)
  mix_cv33_iw33_sm33:f64 - 3-way mix with ssim2 (only kadid + tid)
  mix_cv40_iw60:f64      - the default trainer mix
  mix_target:f64         - active --target-column (sometimes alias for mix_cv40_iw60)
  f0 .. f371             - 372 features (basic + peak + masked + IW-pool blocks)
                           (LARGE: f0..f299 only — 300 features, no IW-pool block)

Missing columns are filled with explicit nulls, NOT silently dropped.
"""
import json
import hashlib
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from join_safety import (  # noqa: E402
    JoinSafetyError,
    assert_metric_not_constant_per_ref,
    assert_no_leaked_metric_columns,
)

SRC_ROOT = Path("/mnt/v/zen/zensim-training")
CANONICAL_ROOT = Path("/mnt/v/zen/zensim-training/canonical-2026-05-18")
DRY = "--dry" in sys.argv

# All possible target columns across the canonical set
ALL_TARGETS = [
    "cvvdp_score", "cvvdp_log_norm",
    "iwssim", "iwssim_log_norm",
    "ssim2_gpu", "ssim2_log_norm",
    "pjnd_target",
    "mix_cv25_iw75", "mix_cv30_iw70", "mix_cv35_iw65", "mix_cv40_iw60",
    "mix_cv45_iw55", "mix_cv50_iw50", "mix_cv55_iw45", "mix_cv55_iw44",
    "mix_cv60_iw40", "mix_cv65_iw35", "mix_cv70_iw30", "mix_cv75_iw25",
    "mix_cv33_iw33_sm33",
    "mix_target",
]

NUM_FEATURES_FULL = 372  # f0..f371
NUM_FEATURES_LARGE = 300  # f0..f299 (LARGE corpus only)

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def validate_metric_columns(label: str, tbl: pa.Table):
    """DATA-INTEGRITY GUARD (added 2026-05-25, structural fix for task #215).

    Refuse to bake the kadid/tid iwssim-leak + ssim2-misjoin bug into the
    canonical set. Fires on:
      - any MOCK column (mock metric columns must never enter a corpus),
      - any metric column that is a bit-identical/perfect copy of human_score
        (Mode A target leak),
      - ssim2_gpu constant within every reference group (Mode B ref-misjoin).
    safesyn/konjnd-dense/LARGE pass cleanly (real iwssim and/or null ssim2).
    Raises RuntimeError so a rebuild cannot silently re-ship corrupt columns.
    The corrected columns are produced by fix_kadid_tid_apply_scores.py; point
    the source paths at the *_fixed_2026-05-25.parquet siblings once promoted.
    """
    import numpy as np

    names = list(tbl.schema.names)
    try:
        assert_no_leaked_metric_columns(label, names, tbl)
        if "ssim2_gpu" in names and "ref_basename" in names:
            s2 = np.asarray(tbl.column("ssim2_gpu").to_numpy(zero_copy_only=False), dtype=float)
            if np.isfinite(s2).sum() > 100:
                refs = tbl.column("ref_basename").to_pylist()
                assert_metric_not_constant_per_ref(label, refs, s2, "ssim2_gpu")
    except JoinSafetyError as e:
        raise RuntimeError(str(e)) from e


def add_null_columns(tbl: pa.Table, columns: list[str], dtype=pa.float64()) -> pa.Table:
    """Add specified columns as all-null if they don't already exist."""
    existing = set(tbl.schema.names)
    for col in columns:
        if col not in existing:
            arr = pa.nulls(tbl.num_rows, type=dtype)
            tbl = tbl.append_column(col, arr)
    return tbl

def select_canonical_schema(tbl: pa.Table, num_features: int) -> pa.Table:
    """Reorder columns to canonical schema: ref_basename, human_score, all targets, fN..."""
    ordered = ["ref_basename", "human_score"]
    for c in ALL_TARGETS:
        if c in tbl.schema.names:
            ordered.append(c)
    feature_cols = [f"f{i}" for i in range(num_features)]
    for c in feature_cols:
        if c not in tbl.schema.names:
            raise RuntimeError(f"missing feature column {c}")
        ordered.append(c)
    return tbl.select(ordered)

def write_with_metadata(tbl: pa.Table, out: Path, metadata: dict):
    schema_with_meta = tbl.schema.with_metadata({
        k.encode(): json.dumps(v).encode() if not isinstance(v, str) else v.encode()
        for k, v in metadata.items()
    })
    tbl = tbl.replace_schema_metadata(schema_with_meta.metadata)
    if DRY:
        print(f"  DRY: would write {out} ({tbl.num_rows} rows x {tbl.num_columns} cols)")
        return
    pq.write_table(tbl, out, compression="zstd", compression_level=15)
    sz = out.stat().st_size
    print(f"  WROTE {out} ({tbl.num_rows} rows x {tbl.num_columns} cols, {sz/1e6:.1f} MB)")

def build_safesyn():
    src = SRC_ROOT / "2026-05-18-v24" / "safesyn_4target_372col.parquet"
    print(f"\n--- build_safesyn from {src}")
    tbl = pq.read_table(str(src))
    validate_metric_columns(src.name, tbl)
    # Add missing mix_cv33_iw33_sm33 column (only kadid/tid have it)
    tbl = add_null_columns(tbl, ["mix_cv33_iw33_sm33", "pjnd_target"])
    tbl = select_canonical_schema(tbl, NUM_FEATURES_FULL)
    out = CANONICAL_ROOT / "train" / "safesyn.parquet"
    write_with_metadata(tbl, out, {
        "canonical_corpus": "safesyn",
        "canonical_date": "2026-05-18",
        "source_paths": [str(src)],
        "num_features": NUM_FEATURES_FULL,
        "training_use": "synthetic-safe (CID22-leak-purged); train target = any of mix_cv* / cvvdp* / iwssim* / ssim2*",
        "schema_version": "canonical-2026-05-18.v1",
    })

def build_kadid():
    src = SRC_ROOT / "2026-05-18-v24" / "kadid_4target_372col.parquet"
    print(f"\n--- build_kadid from {src}")
    tbl = pq.read_table(str(src))
    validate_metric_columns(src.name, tbl)
    tbl = add_null_columns(tbl, ["pjnd_target"])
    tbl = select_canonical_schema(tbl, NUM_FEATURES_FULL)
    out = CANONICAL_ROOT / "train" / "kadid.parquet"
    write_with_metadata(tbl, out, {
        "canonical_corpus": "kadid",
        "canonical_date": "2026-05-18",
        "source_paths": [str(src)],
        "num_features": NUM_FEATURES_FULL,
        "training_use": "KADID-10k DMOS (1-5 lower=better) as human_score; OK to train",
        "schema_version": "canonical-2026-05-18.v1",
    })

def build_tid():
    src = SRC_ROOT / "2026-05-18-v24" / "tid_4target_372col.parquet"
    print(f"\n--- build_tid from {src}")
    tbl = pq.read_table(str(src))
    validate_metric_columns(src.name, tbl)
    tbl = add_null_columns(tbl, ["pjnd_target"])
    tbl = select_canonical_schema(tbl, NUM_FEATURES_FULL)
    out = CANONICAL_ROOT / "train" / "tid.parquet"
    write_with_metadata(tbl, out, {
        "canonical_corpus": "tid",
        "canonical_date": "2026-05-18",
        "source_paths": [str(src)],
        "num_features": NUM_FEATURES_FULL,
        "training_use": "TID2013 MOS (0-9 higher=better) as human_score; OK to train",
        "schema_version": "canonical-2026-05-18.v1",
    })

def build_konjnd_dense():
    src_mix = SRC_ROOT / "2026-05-18-konjnd-dense" / "konjnd_dense_features_mix_targets_372col.parquet"
    src_pjnd = SRC_ROOT / "2026-05-18-konjnd-dense" / "konjnd_dense_pjndtarget_300col.parquet"
    print(f"\n--- build_konjnd_dense from {src_mix} + pjnd from {src_pjnd}")
    tbl_mix = pq.read_table(str(src_mix))
    tbl_pjnd_score = pq.read_table(str(src_pjnd), columns=["ref_basename", "human_score"])
    # Both are 20,160 rows aligned by position (verified above)
    if tbl_mix.num_rows != tbl_pjnd_score.num_rows:
        raise RuntimeError(f"row mismatch: mix {tbl_mix.num_rows} vs pjnd {tbl_pjnd_score.num_rows}")
    # Verify ref alignment
    m_refs = tbl_mix.column("ref_basename").to_pylist()
    p_refs = tbl_pjnd_score.column("ref_basename").to_pylist()
    if m_refs != p_refs:
        raise RuntimeError("ref_basename order mismatch between konjnd-dense mix & pjnd")
    pjnd_arr = tbl_pjnd_score.column("human_score")
    tbl = tbl_mix.append_column("pjnd_target", pjnd_arr)
    # mix_cv55_iw45 (note: konjnd uses cv55_iw45 not cv55_iw44 — preserve both columns nullably)
    tbl = add_null_columns(tbl, [
        "cvvdp_score", "cvvdp_log_norm",
        "iwssim", "iwssim_log_norm",
        "ssim2_gpu", "ssim2_log_norm",
        "mix_cv33_iw33_sm33",
        "mix_cv55_iw44",
        "mix_target",
    ])
    tbl = select_canonical_schema(tbl, NUM_FEATURES_FULL)
    out = CANONICAL_ROOT / "train" / "konjnd-dense.parquet"
    write_with_metadata(tbl, out, {
        "canonical_corpus": "konjnd-dense",
        "canonical_date": "2026-05-18",
        "source_paths": [str(src_mix), str(src_pjnd)],
        "num_features": NUM_FEATURES_FULL,
        "training_use": "KonJND-1k densified (20 distortions x 1008 refs); pjnd_target = mean PJND compression-q threshold",
        "schema_version": "canonical-2026-05-18.v1",
        "schema_note": "human_score column carries the mix output value, NOT PJND. Use pjnd_target for PJND-anchored training.",
    })

def build_cvvdp_large():
    src = SRC_ROOT / "2026-05-17-cvvdp-merged-trainer" / "cvvdp_iwssim_large_300col_v2.parquet"
    print(f"\n--- build_cvvdp_large from {src}")
    tbl = pq.read_table(str(src))
    # This corpus has 300 features (f0..f299) — no IW-pool block
    tbl = add_null_columns(tbl, [
        "ssim2_gpu", "ssim2_log_norm",
        "pjnd_target",
        "mix_cv25_iw75", "mix_cv30_iw70", "mix_cv35_iw65",
        "mix_cv45_iw55", "mix_cv50_iw50", "mix_cv55_iw45", "mix_cv55_iw44",
        "mix_cv60_iw40", "mix_cv65_iw35", "mix_cv70_iw30", "mix_cv75_iw25",
        "mix_cv33_iw33_sm33",
        "mix_target",
    ])
    tbl = select_canonical_schema(tbl, NUM_FEATURES_LARGE)
    out = CANONICAL_ROOT / "train" / "cvvdp_iwssim_LARGE.parquet"
    write_with_metadata(tbl, out, {
        "canonical_corpus": "cvvdp_iwssim_LARGE",
        "canonical_date": "2026-05-18",
        "source_paths": [str(src)],
        "num_features": NUM_FEATURES_LARGE,
        "training_use": "iwssim-overlap subset (73,300 rows) of the 1.17M-row CVVDP corpus. Use for CVVDP+IW supervision.",
        "schema_version": "canonical-2026-05-18.v1",
        "schema_note": "300 features only (no IW-pool block); f300..f371 absent. Subset of the 1,169,500-row CVVDP score corpus.",
    })

def build_val_corpus(name: str, src_path: Path, schema_note: str):
    print(f"\n--- build_val_{name} from {src_path}")
    tbl = pq.read_table(str(src_path))
    # Validation parquets have just ref_basename, human_score, fN — add nulls for all targets so schema matches train
    tbl = add_null_columns(tbl, ALL_TARGETS)
    tbl = select_canonical_schema(tbl, NUM_FEATURES_FULL)
    out = CANONICAL_ROOT / "val" / f"{name}.parquet"
    write_with_metadata(tbl, out, {
        "canonical_corpus": name,
        "canonical_role": "VALIDATION-ONLY",
        "canonical_date": "2026-05-18",
        "source_paths": [str(src_path)],
        "num_features": NUM_FEATURES_FULL,
        "training_use": "HOLDOUT ONLY. Never train on this parquet's human_score.",
        "schema_version": "canonical-2026-05-18.v1",
        "schema_note": schema_note,
    })

def build_validations():
    feats = SRC_ROOT / "2026-05-15-full-features"
    build_val_corpus("cid22", feats / "cid22_features_372col_2026-05-15.parquet",
                     "human_score = MCOS / 100 (CID22 paper). Gold-standard cross-band evaluation corpus.")
    build_val_corpus("kadid", feats / "kadid_features_372col_2026-05-15.parquet",
                     "human_score = DMOS (1-5, lower=better). Same images as train/kadid.parquet — kept here for integrity audit.")
    build_val_corpus("tid", feats / "tid_features_372col_2026-05-15.parquet",
                     "human_score = MOS (0-9, higher=better). Same images as train/tid.parquet — kept for integrity audit.")
    build_val_corpus("konjnd", feats / "konjnd_features_372col_2026-05-15.parquet",
                     "human_score = mean PJND threshold (compression q). 1008 source-anchor pairs. konjnd-dense is the TRAINING densification of these.")
    build_val_corpus("aic3", feats / "aic3_features_372col_2026-05-15.parquet",
                     "human_score = score.jnd (signed JND units, AIC-3 CTC). HOLDOUT — never train.")
    build_val_corpus("aic4", feats / "aic4_features_372col_2026-05-20.parquet",
                     "human_score = reconstructed JND (signed, AIC-4 sample 5 src × 6 codecs × 10 dlevels = 300 pairs). HOLDOUT — never train.")

def copy_score_sidecars():
    """Copy raw score sidecars to scores/ — preserved as input for any future re-mix."""
    pairs = [
        ("cvvdp_imazen_v0_0_1.parquet", SRC_ROOT / "2026-05-15-cvvdp-r2" / "cvvdp_imazen_consolidated.parquet"),
        ("iwssim_imazen.parquet",       SRC_ROOT / "2026-05-15-cvvdp-r2" / "iwssim_imazen_consolidated.parquet"),
        ("ssim2_imazen.parquet",        SRC_ROOT / "2026-05-18-ssim2"    / "ssim2_imazen_consolidated.parquet"),
    ]
    for dest_name, src in pairs:
        print(f"\n--- copy_score_sidecar {src.name} -> {dest_name}")
        tbl = pq.read_table(str(src))
        out = CANONICAL_ROOT / "scores" / dest_name
        write_with_metadata(tbl, out, {
            "canonical_role": "RAW-SCORE-SIDECAR",
            "canonical_date": "2026-05-18",
            "source_paths": [str(src)],
            "schema_version": "canonical-2026-05-18.v1",
            "schema_note": "Raw (image_path, codec, q, knob_tuple_json, score) sidecars. Use for re-mixing training targets or joining onto any unified parquet.",
        })

def emit_manifest():
    manifest_entries = []
    for p in sorted((CANONICAL_ROOT).rglob("*.parquet")):
        rel = p.relative_to(CANONICAL_ROOT).as_posix()
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
    out = CANONICAL_ROOT / "_MANIFEST.json"
    if DRY:
        print(f"\nDRY: would write _MANIFEST.json with {len(manifest_entries)} entries")
        return
    with open(out, "w") as f:
        json.dump({
            "canonical_date": "2026-05-18",
            "schema_version": "canonical-2026-05-18.v1",
            "entries": manifest_entries,
        }, f, indent=2)
    print(f"\nWROTE _MANIFEST.json ({len(manifest_entries)} entries)")

def main():
    build_safesyn()
    build_kadid()
    build_tid()
    build_konjnd_dense()
    build_cvvdp_large()
    build_validations()
    copy_score_sidecars()
    emit_manifest()

if __name__ == "__main__":
    main()
