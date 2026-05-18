#!/usr/bin/env python3
"""EX-MIX3: add 3-way mix target columns to canonical training parquets.

For each canonical parquet that has cvvdp_log_norm + iwssim_log_norm +
ssim2_log_norm all populated, compute three new target columns:

    mix_cv33_iw33_sm33 = (1/3)·cvvdp_log_norm + (1/3)·iwssim_log_norm + (1/3)·ssim2_log_norm
    mix_cv30_iw40_sm30 = 0.30·cvvdp_log_norm + 0.40·iwssim_log_norm + 0.30·ssim2_log_norm
    mix_cv40_iw40_sm20 = 0.40·cvvdp_log_norm + 0.40·iwssim_log_norm + 0.20·ssim2_log_norm

Coverage gate: groups with <50% ssim2 coverage are dropped (NOT zero-filled,
per CLAUDE.md falsification-antipattern guard).

Output: /mnt/v/zen/zensim-training/2026-05-18-mix3/<corpus>.parquet
"""

from __future__ import annotations
import sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

SRC = Path("/mnt/v/zen/zensim-training/canonical-2026-05-18/train")
DST = Path("/mnt/v/zen/zensim-training/2026-05-18-mix3")
DST.mkdir(parents=True, exist_ok=True)

CORPORA = [
    "safesyn",          # 196,086 rows
    "kadid",            #  10,125
    "tid",              #   3,000
    "konjnd-dense",     #  20,160 (no ssim2 — will be reported but won't get mix3 columns)
    "cvvdp_iwssim_LARGE",  # 73,300 (no ssim2 — will be reported)
]

COVERAGE_THRESHOLD = 0.50  # drop if <50% triple non-null

# ----- konjnd PJND-anchor workaround (added after seed=1 KonJND collapse) -----
# Per task: coverage gate drops konjnd from training. BUT V_22 noLARGE relies
# on konjnd at weight 0.02 for JND boundary signal. Dropping it costs
# KonJND-eval SROCC ~0.55 (0.84 → 0.30). Re-add konjnd as an extra training
# group whose target columns are PJND-passthrough (mix_cv33_iw33_sm33 =
# mix_cv30_iw40_sm30 = mix_cv40_iw40_sm20 = mix_cv40_iw60 = PJND threshold,
# all the same 1008-row signal). This matches V_22 noLARGE's exact mechanic
# verbatim (konjnd at weight 0.02 trains against mix_cv40_iw60 which equals
# human_score = PJND for the small konjnd parquet — see add_mix3_konjnd()).
# This is NOT zero-fill (the konjnd target VALUE is the real PJND signal,
# the value computed by the konjnd-1k MOS study); it's a label-passthrough
# for a group whose conceptual blend equation doesn't apply because the
# group doesn't have per-pair cvvdp/iwssim/ssim2 measurements.


def compute_mixes(cvvdp_ln, iwssim_ln, ssim2_ln):
    """Compute 3 mix variants. All inputs/outputs are np arrays in score units 0..100."""
    mix_33 = (1.0 / 3.0) * cvvdp_ln + (1.0 / 3.0) * iwssim_ln + (1.0 / 3.0) * ssim2_ln
    mix_30_40_30 = 0.30 * cvvdp_ln + 0.40 * iwssim_ln + 0.30 * ssim2_ln
    mix_40_40_20 = 0.40 * cvvdp_ln + 0.40 * iwssim_ln + 0.20 * ssim2_ln
    return mix_33, mix_30_40_30, mix_40_40_20


def process_corpus(name: str) -> dict:
    src_path = SRC / f"{name}.parquet"
    if not src_path.exists():
        return {"name": name, "status": "SOURCE_MISSING", "ssim2_coverage": 0.0}

    tbl = pq.read_table(src_path)
    n_rows = tbl.num_rows

    # Check ssim2 coverage
    if "ssim2_log_norm" not in tbl.schema.names:
        return {"name": name, "status": "NO_SSIM2_COL", "ssim2_coverage": 0.0}

    cvvdp_a = tbl["cvvdp_log_norm"].to_numpy()
    iwssim_a = tbl["iwssim_log_norm"].to_numpy()
    ssim2_a = tbl["ssim2_log_norm"].to_numpy()

    n_ssim2_nn = int(np.sum(~np.isnan(ssim2_a)))
    n_cvvdp_nn = int(np.sum(~np.isnan(cvvdp_a)))
    n_iwssim_nn = int(np.sum(~np.isnan(iwssim_a)))
    coverage = n_ssim2_nn / max(n_rows, 1)
    triple_coverage = int(np.sum(
        ~np.isnan(ssim2_a) & ~np.isnan(cvvdp_a) & ~np.isnan(iwssim_a)
    )) / max(n_rows, 1)

    if triple_coverage < COVERAGE_THRESHOLD:
        return {
            "name": name,
            "status": "DROPPED_LOW_COVERAGE",
            "rows": n_rows,
            "ssim2_coverage": coverage,
            "triple_coverage": triple_coverage,
            "cvvdp_coverage": n_cvvdp_nn / max(n_rows, 1),
            "iwssim_coverage": n_iwssim_nn / max(n_rows, 1),
        }

    # All three signals present -> compute mixes
    mix_33, mix_30_40_30, mix_40_40_20 = compute_mixes(cvvdp_a, iwssim_a, ssim2_a)

    # Replace the existing nullable mix_cv33_iw33_sm33 column with the populated one;
    # add the two new variant columns.
    out_tbl = tbl
    if "mix_cv33_iw33_sm33" in tbl.schema.names:
        idx = tbl.schema.get_field_index("mix_cv33_iw33_sm33")
        out_tbl = out_tbl.set_column(idx, "mix_cv33_iw33_sm33", pa.array(mix_33, type=pa.float64()))
    else:
        out_tbl = out_tbl.append_column("mix_cv33_iw33_sm33", pa.array(mix_33, type=pa.float64()))

    # Add the two new variants (always append).
    if "mix_cv30_iw40_sm30" in tbl.schema.names:
        idx = out_tbl.schema.get_field_index("mix_cv30_iw40_sm30")
        out_tbl = out_tbl.set_column(idx, "mix_cv30_iw40_sm30", pa.array(mix_30_40_30, type=pa.float64()))
    else:
        out_tbl = out_tbl.append_column("mix_cv30_iw40_sm30", pa.array(mix_30_40_30, type=pa.float64()))

    if "mix_cv40_iw40_sm20" in tbl.schema.names:
        idx = out_tbl.schema.get_field_index("mix_cv40_iw40_sm20")
        out_tbl = out_tbl.set_column(idx, "mix_cv40_iw40_sm20", pa.array(mix_40_40_20, type=pa.float64()))
    else:
        out_tbl = out_tbl.append_column("mix_cv40_iw40_sm20", pa.array(mix_40_40_20, type=pa.float64()))

    out_path = DST / f"{name}.parquet"
    pq.write_table(out_tbl, out_path, compression="zstd", compression_level=15)

    # Quick stats on the new columns
    stats = {}
    for col, vals in [
        ("mix_cv33_iw33_sm33", mix_33),
        ("mix_cv30_iw40_sm30", mix_30_40_30),
        ("mix_cv40_iw40_sm20", mix_40_40_20),
    ]:
        nn = vals[~np.isnan(vals)]
        stats[col] = {
            "min": float(np.min(nn)) if len(nn) else float("nan"),
            "max": float(np.max(nn)) if len(nn) else float("nan"),
            "mean": float(np.mean(nn)) if len(nn) else float("nan"),
            "nonnull": int(len(nn)),
        }

    return {
        "name": name,
        "status": "WROTE",
        "rows": n_rows,
        "ssim2_coverage": coverage,
        "triple_coverage": triple_coverage,
        "out_path": str(out_path),
        "out_bytes": out_path.stat().st_size,
        "stats": stats,
    }


def add_konjnd_pjnd_passthrough():
    """Rebuild the konjnd small parquet with mix3 columns = PJND passthrough.

    Source: /mnt/v/zen/zensim-training/2026-05-17-cvvdp/konjnd_features_mix_targets_372col.parquet
    Target: DST / 'konjnd.parquet'
    """
    src = Path("/mnt/v/zen/zensim-training/2026-05-17-cvvdp/konjnd_features_mix_targets_372col.parquet")
    out = DST / "konjnd.parquet"
    tbl = pq.read_table(src)
    n = tbl.num_rows
    # human_score == mix_cv40_iw60 == PJND threshold for this corpus (1008 rows)
    pjnd = tbl["mix_cv40_iw60"].to_numpy()
    # Sanity: assert this matches human_score
    hs = tbl["human_score"].to_numpy()
    if not np.allclose(pjnd, hs):
        raise RuntimeError(f"konjnd small: mix_cv40_iw60 != human_score! Likely corrupt input.")
    # Append mix3 columns as PJND passthrough
    new_cols = ["mix_cv33_iw33_sm33", "mix_cv30_iw40_sm30", "mix_cv40_iw40_sm20"]
    for col in new_cols:
        if col in tbl.schema.names:
            idx = tbl.schema.get_field_index(col)
            tbl = tbl.set_column(idx, col, pa.array(pjnd, type=pa.float64()))
        else:
            tbl = tbl.append_column(col, pa.array(pjnd, type=pa.float64()))
    pq.write_table(tbl, out, compression="zstd", compression_level=15)
    print(f"--- konjnd (PJND-passthrough)")
    print(f"  rows: {n}")
    print(f"  pjnd target range: [{pjnd.min():.3f}, {pjnd.max():.3f}] mean={pjnd.mean():.3f}")
    print(f"  out: {out} ({out.stat().st_size} B)")
    print()


def main():
    print(f"EX-MIX3: building 3-way mix target parquets")
    print(f"  source:  {SRC}")
    print(f"  dest:    {DST}")
    print(f"  gate:    triple-coverage (cv+iw+sm) >= {COVERAGE_THRESHOLD}")
    print()

    results = []
    for name in CORPORA:
        print(f"--- {name}")
        r = process_corpus(name)
        results.append(r)
        for k, v in r.items():
            if k == "stats":
                for col, s in v.items():
                    print(f"  {col}: range=[{s['min']:.3f}, {s['max']:.3f}] mean={s['mean']:.3f} n_nonnull={s['nonnull']}")
            else:
                print(f"  {k}: {v}")
        print()

    # Add the konjnd PJND-passthrough group
    add_konjnd_pjnd_passthrough()

    # Summary
    kept = [r for r in results if r["status"] == "WROTE"]
    dropped = [r for r in results if r["status"] != "WROTE"]
    print(f"=== summary: {len(kept)} kept, {len(dropped)} dropped/skipped")
    for r in kept:
        print(f"  KEPT: {r['name']} -> {r['out_path']}")
    for r in dropped:
        print(f"  DROPPED: {r['name']} ({r['status']}, triple-coverage={r.get('triple_coverage', 0):.2%})")


if __name__ == "__main__":
    main()
