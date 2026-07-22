#!/usr/bin/env python3
"""Promote the scattered 720-feature extraction outputs into ONE canonical
dataset directory with a unified manifest (E1 gate: parquet + _MANIFEST.json
with build_commit + DATA_PROVENANCE index).

Sources (both stay in place — nothing is deleted; POINTER.md is dropped in
each dated dir):
  /mnt/v/output/zensim/v2-ab-2026-07-19/        7 corpora (A/B campaign wave)
  /mnt/v/output/zensim/v2-backfill-2026-07-20/  4 corpora (local E1 leg)

Destination:
  /mnt/v/zen/zensim-training/ext720-canonical-2026-07-22/

Idempotent: re-running verifies sha256s and rewrites the manifest/README.
Mirror commands (run separately, see README): R2 s3://zentrain/ + Tower.

Schema of every parquet: ref_basename (utf8), human_score (f64),
f0..f719 (f64) — f0..f371 = frozen v1 with-iw block, f372..f719 = v2-348
bounded block (append-only directive 2026-07-19). All ZSTD. NaN/null-free
(scanned 2026-07-22).
"""

import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

SRC_AB = Path("/mnt/v/output/zensim/v2-ab-2026-07-19")
SRC_BF = Path("/mnt/v/output/zensim/v2-backfill-2026-07-20")
DEST = Path("/mnt/v/zen/zensim-training/ext720-canonical-2026-07-22")

# Per-corpus metadata. The 2026-07-19 wave's extraction ran at zensim main
# @ 6f191264 (recorded in benchmarks/v2_trainability_ab_2026-07-19.md
# "Repro": trainer lineage ea0186a0, "main @ 6f191264 at run time"); its
# kadid/tid/cid22val/csiq/live pair TSVs were not persisted, but each is
# the COMPLETE standard corpus (row count == corpus definition, verified
# below). The 2026-07-20 wave carries its own per-entry manifest
# (zensim_main_commit 9e7516d7), merged in verbatim.
WAVE_AB = {
    "commit": "6f191264",
    "provenance": "benchmarks/v2_trainability_ab_2026-07-19.md (Repro section)",
}
CORPORA = [
    # (file stem, source dir, expected rows, role, target semantics, notes)
    (
        "ext_cid22val",
        SRC_AB,
        4292,
        "H-cid22 (T0 GOLD holdout — NEVER train; MOS ban absolute)",
        "human_score = CID22 MCOS (human)",
        "CID22 49-reference validation set, complete.",
    ),
    (
        "ext_aic3",
        SRC_AB,
        600,
        "H-aic3 (T0 holdout — never train)",
        "human_score = AIC-3 CTC JND-based score (human)",
        "JPEG-AIC-3 committee test corpus, complete (10 refs).",
    ),
    (
        "ext_csiq",
        SRC_AB,
        866,
        "H-csiq (context holdout, general-FR)",
        "human_score = CSIQ DMOS (human)",
        "CSIQ full set.",
    ),
    (
        "ext_live",
        SRC_AB,
        779,
        "H-live (context holdout, general-FR)",
        "human_score = LIVE-R2 DMOS (human)",
        "LIVE Release-2 full set.",
    ),
    (
        "ext_kadid",
        SRC_AB,
        10125,
        "T-kadid (train, GUARD weight only — train==val overlap makes it a memorization guard, not ranking signal)",
        "human_score = KADID-10k DMOS (human)",
        "KADID-10k complete (81 refs x 25 distortions x 5 levels).",
    ),
    (
        "ext_tid",
        SRC_AB,
        3000,
        "T-tid (train, GUARD weight only)",
        "human_score = TID2013 MOS (human)",
        "TID2013 complete (25 refs).",
    ),
    (
        "ext_safesyn_full",
        SRC_AB,
        111068,
        "T-safe JPEG slice (train)",
        "human_score = ssim2-derived label (safesyn convention), NOT human",
        "safesyn JPEG-codec slice; pairs TSV persisted alongside "
        "(safesyn_jpeg_FULL_pairs_ab.tsv). The fleet T-safe multi-codec "
        "extraction supersedes this for full-codec coverage; this slice "
        "remains the valid JPEG-only leg (what the A/B trained on).",
    ),
    (
        "ext_cid22_train201",
        SRC_BF,
        17611,
        "T-cid201 (train; ssim2-anchored, NOT MCOS — legal per CID22 ban)",
        "human_score = ssim2-anchored score, NOT human MOS",
        "cid22_train 201-reference subset; verified 0-overlap with the "
        "49-ref holdout (backfill 2026-07-20).",
    ),
    (
        "ext_aic4",
        SRC_BF,
        300,
        "H-aic4 (T0 holdout — never train)",
        "human_score = AIC-4 reconstructed JND score (human)",
        "AIC-4 sample dataset (5 src x 6 codecs x 10 levels).",
    ),
    (
        "ext_konjnd_jpeg_val",
        SRC_BF,
        504,
        "H-konjnd (near-threshold holdout)",
        "human_score = raw mean-PJND target",
        "KonJND-1k validation split, JPEG half (BPG half: no decoder).",
    ),
    (
        "ext_sdr25",
        SRC_BF,
        50,
        "H-sdr25 (HQ-zone holdout; n=50 — DIRECTIONAL only, not a hard gate)",
        "human_score = ordered-probit collapse of triplet responses",
        "JPEG-AI-SDR25 scoreable pairs; the '95k' figure was triplet "
        "RESPONSES, not pairs (correction e8f7e892).",
    ),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    bf_manifest = json.loads((SRC_BF / "_MANIFEST.json").read_text())
    bf_entries = {Path(e["parquet"]).stem: e for e in bf_manifest["entries"]}

    entries = []
    total_rows = 0
    for stem, src_dir, want_rows, role, target, notes in CORPORA:
        src = src_dir / f"{stem}.parquet"
        dst = DEST / f"{stem}.parquet"
        src_sha = sha256(src)
        if not dst.exists() or sha256(dst) != src_sha:
            shutil.copy2(src, dst)
        dst_sha = sha256(dst)
        assert dst_sha == src_sha, f"copy corrupted for {stem}"

        f = pq.ParquetFile(dst)
        rows = f.metadata.num_rows
        names = f.schema_arrow.names
        n_feat = sum(1 for c in names if c.startswith("f") and c[1:].isdigit())
        assert rows == want_rows, f"{stem}: rows {rows} != expected {want_rows}"
        assert n_feat == 720, f"{stem}: {n_feat} features != 720"
        assert names[:2] == ["ref_basename", "human_score"], f"{stem}: key cols"
        total_rows += rows

        entry = {
            "corpus": stem,
            "parquet": str(dst),
            "sha256": dst_sha,
            "rows": rows,
            "n_features": 720,
            "layout": "f0..f371 = frozen v1 with-iw; f372..f719 = v2-348 bounded (append-only)",
            "role": role,
            "target_semantics": target,
            "notes": notes,
            "source_parquet": str(src),
            "extractor": "zensim/examples/v2_ab_extract.rs (ZENSIM_AB_MODE=ext)",
        }
        if src_dir == SRC_BF and stem in bf_entries:
            b = bf_entries[stem]
            entry["zensim_main_commit"] = b.get("zensim_main_commit")
            entry["pairs_tsv"] = b.get("pairs_tsv")
            entry["builder"] = b.get("builder")
            entry["extraction_date"] = b.get("date")
            entry["upstream_notes"] = b.get("notes")
        else:
            entry["zensim_main_commit"] = WAVE_AB["commit"]
            entry["commit_provenance"] = WAVE_AB["provenance"]
            entry["extraction_date"] = "2026-07-19"
            if stem == "ext_safesyn_full":
                entry["pairs_tsv"] = str(SRC_AB / "safesyn_jpeg_FULL_pairs_ab.tsv")
            elif stem == "ext_aic3":
                entry["pairs_tsv"] = str(SRC_AB / "aic3_pairs_ab.tsv")
            else:
                entry["pairs_tsv"] = (
                    "not persisted (complete standard corpus; row count == corpus definition)"
                )
        entries.append(entry)

    manifest = {
        "description": (
            "CANONICAL 720-feature (v1-372 ++ v2-348) extraction datasets — "
            "local legs, consolidated from v2-ab-2026-07-19 + "
            "v2-backfill-2026-07-20. Fleet legs (T-big bigcodec multi-codec, "
            "T-safe multi-codec) land separately via the zenfleet job system "
            "and are indexed when written back. See "
            "docs/V2_EXPERIMENT_PLAN_2026-07-20.md."
        ),
        "date": str(date.today()),
        "n_corpora": len(entries),
        "total_rows": total_rows,
        "numeric_note": (
            "Both waves extracted PRE pool-SIMD (commits <= 9e7516d7). "
            "Extractions from commit 1e48e7c8 onward differ in v2 pool "
            "features by <= 1.1e-7 rel on AVX-512 hosts (policy 5e-4) — "
            "see benchmarks/v2_ref_reuse_perf_2026-07-21.md."
        ),
        "bans": [
            "CID22-49 human MOS NEVER trains (ext_cid22val is eval-only)",
            "AIC-3 / AIC-4 holdout-only",
            "KADID/TID train at guard weight only (train==val overlap)",
        ],
        "entries": entries,
    }
    (DEST / "_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    readme = ["# ext720-canonical-2026-07-22 — 720-feature extraction datasets (local legs)\n"]
    readme.append(
        "One parquet per corpus: `ref_basename, human_score, f0..f719` "
        "(f64, ZSTD). f0..f371 = frozen v1 with-iw block; f372..f719 = "
        "v2-348 bounded block. NaN/null-free. Full provenance per corpus: "
        "`_MANIFEST.json`.\n"
    )
    readme.append("| corpus | rows | role |")
    readme.append("|---|--:|---|")
    for e in entries:
        readme.append(f"| {e['corpus']} | {e['rows']} | {e['role']} |")
    readme.append(f"\nTotal: {total_rows} rows, {len(entries)} corpora.\n")
    readme.append(
        "Mirrors: `s3://zentrain/ext720-canonical-2026-07-22/` (R2), "
        "`/mnt/tower/output/zensim-ext720-canonical-2026-07-22/` (Tower).\n"
        "Sources (kept in place): `/mnt/v/output/zensim/v2-ab-2026-07-19/`, "
        "`/mnt/v/output/zensim/v2-backfill-2026-07-20/`.\n"
    )
    (DEST / "README.md").write_text("\n".join(readme))

    for src_dir in (SRC_AB, SRC_BF):
        (src_dir / "POINTER.md").write_text(
            f"ext_*.parquet from this directory were PROMOTED to the canonical "
            f"dataset dir on 2026-07-22:\n  {DEST}\n(+ R2/Tower mirrors — see "
            f"its README.md). Use the canonical copies; nothing here was "
            f"deleted.\n"
        )

    print(f"OK: {len(entries)} corpora, {total_rows} rows -> {DEST}")
    for e in entries:
        print(f"  {e['corpus']:24s} {e['rows']:>7} rows  {e['sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
