#!/usr/bin/env python3
"""Promote the post-jxl-fix near-lossless (HF-end) sweep into the canonical set.

WHY THIS CORPUS. It is the ONLY data we hold below butteraugli distance 0.03
that was generated with the FIXED jxl encoder (`eeb52735`, 2026-07-06T06:09Z;
this sweep ran 06:35). 72% of its cells sit above ssim2 95, where canonical
safesyn has only 1.86% (safesyn's `ssim2_gpu`: p50 68.73, p90 91.15, p99
95.89) — so it is the dense high-quality coverage the training distribution
otherwise lacks. §8.39 measured that a bake trained without it ranks this
regime BACKWARDS (held-out per-ref SROCC -0.144, 60% of refs negative), and
that rank supervision here fixes it to +0.950 / 0% negative for -0.004 CID22.

CONTAMINATION VERDICT: CLEAN, no rows filtered. The standing two-clause test
(benchmarks/jxl_nearlossless_contamination_2026-07-15.md, DATASET_HISTORY
§3.22) is: a JXL cell is contaminated iff `distance < 0.03` AND `generated
before 2026-07-06T06:09Z`. This sweep is post-fix, so it is clean at EVERY
distance — clause 2 fails. Its distances (0.005..0.03) would otherwise put
1,000 of 1,200 cells squarely in the contaminated range, which is exactly why
the verdict is recorded here rather than assumed. The sibling `smoke/` sweep IS
pre-fix and broken (ssim2 ~34); it is deliberately not read by this script.

USAGE NOTE THAT IS LOAD-BEARING. Consume this corpus PER-REF / PAIRWISE, never
as an absolute target. Its ssim2 ladder moves only ~0.92 pts within an image
(p10 +0.43, p90 +1.94) against ~6 pts of cross-image spread at a fixed
distance, so pooled SROCC reads +0.204 while per-ref reads +0.916 — the same
confound as the documented AIC-3 "0.79 pooled / 0.93 per-ref". An MSE/absolute
term here fits between-image scale, not the ladder. Train it with
`--group hf_nearlossless:<path>:0.1:0.0:withinref` (the Rust trainer's
within-ref RankNet pairing); 0.1 is where §8.39 measured the signal to
saturate.

Single owner: this is the canonical-corpus builder family. Do not re-implement
the join elsewhere — `join_safety.safe_metric_join` exists so the §3.18 Mode-B
ref-misjoin cannot be reintroduced.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from join_safety import (  # noqa: E402
    JoinSafetyError,
    assert_metric_not_constant_per_ref,
    assert_no_leaked_metric_columns,
    safe_metric_join,
)

SRC = Path("/mnt/v/output/zensim-jxl-nearlossless/refit")
OUT_DIR = Path("/mnt/v/zen/zensim-training/canonical-2026-07-15/train")
OUT = OUT_DIR / "hf_nearlossless.parquet"
N_FEAT = 372

# The fix commit + this sweep's generation time. Both are needed to evaluate
# clause 2 of the contamination test, so both are recorded in the manifest.
JXL_FIX_UTC = "2026-07-06T06:09:17Z"
SWEEP_UTC = "2026-07-06T06:35:00Z"
JOIN_KEYS = ["image_path", "codec", "q", "knob_tuple_json"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def main() -> int:
    feats_path = SRC / "features.parquet"
    pareto_path = SRC / "pareto.tsv"
    for p in (feats_path, pareto_path):
        if not p.exists():
            print(f"FATAL: missing {p}\n  restore from R2 "
                  f"s3://zentrain/jxl-nearlossless-2026-07-06/ — see "
                  f"benchmarks/jxl_nearlossless_corpus_2026-07-06.pointer.md")
            return 1

    ft = pq.read_table(feats_path)
    feat_cols = [f"feat_{i}" for i in range(N_FEAT)]
    missing = [c for c in feat_cols if c not in ft.schema.names]
    if missing:
        print(f"FATAL: features.parquet lacks {len(missing)} feature columns "
              f"(first: {missing[0]}) — expected the 372-wide with-iw regime")
        return 1

    feats = ft.select(JOIN_KEYS + feat_cols + ["zensim_score"]).to_pandas()
    pareto = pd.read_csv(pareto_path, sep="\t")

    # `q` is a float in the parquet (90.0) and a string in the TSV ("90").
    # str(90.0) != "90", so a naive join silently yields ZERO rows — it does
    # not error, it just quietly produces an empty corpus. Normalize both.
    for df in (feats, pareto):
        df["q"] = df["q"].astype(float)

    # safe_metric_join REFUSES to join on a weakened key rather than collapsing
    # to a ref-level mean (the §3.18 Mode-B misjoin that pinned ssim2_gpu ~100
    # on kadid/tid). Here `image_path` identifies the REFERENCE (200 distinct)
    # and `knob_tuple_json` (6 distinct) is the distortion axis, so the full
    # 4-key tuple is genuinely per-pair: 200 x 6 = 1200.
    try:
        merged = safe_metric_join(feats, pareto, JOIN_KEYS, "score_ssim2")
    except JoinSafetyError as e:
        print(f"FATAL: join refused: {e}")
        return 1

    if len(merged) != len(feats):
        print(f"FATAL: join changed row count {len(feats)} -> {len(merged)}")
        return 1
    n_null = int(merged["score_ssim2"].isna().sum())
    if n_null:
        print(f"FATAL: {n_null} rows got no score_ssim2 — join key mismatch")
        return 1

    # ref identity: the source refs lived in a /tmp scratchpad that has since
    # been wiped, so the absolute path is dead. The basename is the stable
    # identity and all 200 are re-findable in /mnt/v/input/zensim/sources/.
    merged["ref_basename"] = merged["image_path"].map(lambda p: Path(p).stem)
    merged["distance"] = merged["knob_tuple_json"].map(
        lambda s: float(json.loads(s)["distance"])
    )

    n_refs = merged["ref_basename"].nunique()
    if n_refs != 200:
        print(f"FATAL: expected 200 refs, got {n_refs}")
        return 1

    # A per-pair metric that is constant within a ref is the misjoin
    # signature. This corpus's ladder is SHALLOW (~0.92 pts p50), so this
    # guard is the difference between "shallow but real" and "broadcast".
    assert_metric_not_constant_per_ref(
        "hf_nearlossless/score_ssim2",
        merged["ref_basename"].to_numpy(),
        merged["score_ssim2"].to_numpy(),
    )

    # Canonical schema uses f<i>; the sidecar extractor emits feat_<i>. The
    # Rust loader now reads both, but the canonical set stays uniform on f<i>.
    out = pd.concat(
        [
            pd.DataFrame(
                {
                    "ref_basename": merged["ref_basename"],
                    "human_score": merged["score_ssim2"].astype("float64"),
                    "ssim2_gpu": merged["score_ssim2"].astype("float64"),
                    "zensim_score": merged["zensim_score"].astype("float64"),
                    "distance": merged["distance"].astype("float64"),
                }
            ),
            merged[[f"feat_{i}" for i in range(N_FEAT)]]
            .astype("float64")
            .rename(columns={f"feat_{i}": f"f{i}" for i in range(N_FEAT)}),
        ],
        axis=1,
    )

    # Mode-A guard: refuse any metric column that is a copy of human_score.
    # ssim2_gpu IS human_score here by construction (the target is ssim2), so
    # it is declared, not smuggled — the guard runs over the rest.
    assert_no_leaked_metric_columns(
        "hf_nearlossless", ["zensim_score"], out,
    )

    # §3.19 IW-explosion context. Not a filter — a recorded observation, so a
    # later winsor decision has the number instead of re-deriving it.
    fmat = out[[f"f{i}" for i in range(N_FEAT)]].to_numpy()
    if not np.isfinite(fmat).all():
        print("FATAL: non-finite features present")
        return 1
    max_abs = float(np.abs(fmat).max())

    lad = merged.groupby("ref_basename")["score_ssim2"].agg(lambda v: v.max() - v.min())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tbl = pa.Table.from_pandas(out, preserve_index=False)
    pq.write_table(tbl, OUT, compression="zstd", compression_level=15)

    # REF-LEVEL split, not row-level. Every ref contributes 6 rows (one per
    # distance); splitting rows would put the same image on both sides and the
    # held-out per-ref readout would be measuring memorization. Deterministic:
    # sort refs, take every 4th to val -> 50 val / 150 train, and NO ref is in
    # both. (Matches the §8.39 Python split, so the two are comparable.)
    refs = sorted(out["ref_basename"].unique())
    val_refs = set(refs[::4])
    is_val = out["ref_basename"].isin(val_refs)
    splits = {"hf_nearlossless_train": out[~is_val], "hf_nearlossless_val": out[is_val]}
    split_meta = {}
    for label, df in splits.items():
        p = OUT_DIR / f"{label}.parquet"
        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False),
            p, compression="zstd", compression_level=15,
        )
        split_meta[label] = {
            "path": str(p),
            "rows": int(len(df)),
            "n_refs": int(df["ref_basename"].nunique()),
            "sha256": sha256_file(p),
        }
    overlap = set(splits["hf_nearlossless_train"]["ref_basename"]) & set(
        splits["hf_nearlossless_val"]["ref_basename"]
    )
    if overlap:
        print(f"FATAL: {len(overlap)} refs leak across the split")
        return 1

    manifest = {
        "build_commit": git_commit(),
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": "hf_nearlossless",
        "role": "train (within-ref rank supervision ONLY — see usage note)",
        "rows": int(len(out)),
        "n_refs": int(n_refs),
        "n_features": N_FEAT,
        "target_column": "human_score (= ssim2, 0..100)",
        "source_paths": [str(feats_path), str(pareto_path)],
        "source_sha256": {
            "features.parquet": sha256_file(feats_path),
            "pareto.tsv": sha256_file(pareto_path),
        },
        "output_sha256": sha256_file(OUT),
        "splits": split_meta,
        "split_rule": "REF-level (not row-level): sorted refs, every 4th -> val "
                      "(50 val / 150 train, zero ref overlap). Row-level would "
                      "put the same image on both sides and the held-out per-ref "
                      "readout would measure memorization. Matches the §8.39 "
                      "Python split so the two are comparable.",
        "jxl_contamination": {
            "verdict": "CLEAN — no rows filtered",
            "test": "contaminated iff distance < 0.03 AND generated before "
                    f"{JXL_FIX_UTC}",
            "clause_1_distance_lt_0.03": int((out["distance"] < 0.03).sum()),
            "clause_2_pre_fix": False,
            "sweep_generated_utc": SWEEP_UTC,
            "jxl_fix_commit": "eeb52735",
            "jxl_fix_utc": JXL_FIX_UTC,
            "note": "post-fix => clean at every distance; clause 2 fails so "
                    "the 1000 sub-0.03 cells are NOT contaminated",
        },
        "usage": {
            "consume": "per-ref / pairwise ONLY",
            "trainer": "--group hf_nearlossless:<path>:0.1:0.0:withinref",
            "why": "pooled SROCC +0.204 vs per-ref +0.916 (cross-image scale "
                   "mixing); within-image ssim2 ladder ~0.92 pts vs ~6 pts "
                   "between images, so an absolute/MSE target fits noise",
            "saturation": "§8.39: rank weight 0.1 is where the signal "
                          "saturates; higher only costs CID22",
        },
        "diagnostics": {
            "max_abs_feature": max_abs,
            "pct_above_ssim2_95": float((out["human_score"] > 95).mean() * 100),
            "within_ref_ladder_span_p10": float(lad.quantile(0.10)),
            "within_ref_ladder_span_p50": float(lad.quantile(0.50)),
            "within_ref_ladder_span_p90": float(lad.quantile(0.90)),
        },
        "provenance": "zensim 14b2f3c4 (post-fix rebuild + re-sweep, 1200/1200 "
                      "cells, 0 failures); encoder jxl-encoder@eeb52735",
    }
    (OUT_DIR / "_MANIFEST_hf_nearlossless.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print(f"wrote {OUT}")
    print(f"  rows={len(out)} refs={n_refs} feats={N_FEAT} "
          f"bytes={OUT.stat().st_size:,}")
    print(f"  sha256={manifest['output_sha256']}")
    print(f"  human_score(ssim2) p50={out['human_score'].median():.2f} "
          f">95: {manifest['diagnostics']['pct_above_ssim2_95']:.1f}%")
    print(f"  within-ref ladder span p10/p50/p90 = "
          f"{lad.quantile(0.10):.2f}/{lad.quantile(0.50):.2f}/"
          f"{lad.quantile(0.90):.2f} ssim2 pts")
    print(f"  max|feat|={max_abs:.1f}")
    print(f"  jxl contamination: {manifest['jxl_contamination']['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
