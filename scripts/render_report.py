#!/usr/bin/env python3
"""Render V0_7 e1-fill subsampling ablation report.

Reads per-variant per-pair CSVs and computes:
  - |SROCC| of v04_distance vs human_score, by dataset
  - |SROCC| by SSIM2 band (synthetic-side breakdown using the holdout dataset's fast_ssim2_score)

Holdout SSIM2 bands aren't really apples-to-apples for human-MOS datasets,
but they give a sense of low/mid/high-distortion behavior on real images.
"""
import sys
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats


VARIANTS = [
    ("0pct",   "ablation_0pct (V0_6 baseline reproduction)"),
    ("5pct",   "ablation_5pct"),
    ("10pct",  "ablation_10pct"),
    ("20pct",  "ablation_20pct"),
    ("50pct",  "ablation_50pct"),
    ("100pct", "ablation_100pct (V0_7 reproduction)"),
]

EVAL_DIR = Path("/home/lilith/work/zen/zensim--v07-e1-ablation/eval_out")
OUT_PATH = Path("/home/lilith/work/zen/zensim--v07-e1-ablation/benchmarks/v07_e1_subsample_ablation_2026-05-05.md")

# V0_6 baseline numbers from the brief (4metric_overnight_FINAL_2026-05-01.md)
V06_BASELINE = {
    "KADIK10k": 0.8496,
    "TID2013":  0.8416,
    "CID22":    0.8935,
    "KonJND-1k": None,   # not reported in baseline
}

DATASETS = ["KADIK10k", "TID2013", "CID22", "KonJND-1k"]

BANDS = [
    ("25-40", lambda s: 25 <= s < 40),
    ("40-60", lambda s: 40 <= s < 60),
    ("60-75", lambda s: 60 <= s < 75),
    ("75-90", lambda s: 75 <= s < 90),
]


def load(path):
    by_ds = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                h = float(r["human_score"])
                v = float(r["v04_distance"])
                s = float(r["fast_ssim2_score"])
            except Exception:
                continue
            if not (np.isfinite(h) and np.isfinite(v) and np.isfinite(s)):
                continue
            by_ds[r["dataset"]].append((h, v, s))
    return by_ds


def srocc(a, b):
    if len(a) < 3:
        return float("nan"), 0
    return abs(stats.spearmanr(a, b).correlation), len(a)


def per_band_srocc(rows, band_pred):
    h, v, s = zip(*[(h, v, s) for h, v, s in rows if band_pred(s)]) if any(band_pred(s) for h, v, s in rows) else ((), (), ())
    if len(h) < 3:
        return float("nan"), len(h)
    return srocc(list(v), list(h))


def main():
    rows = {}
    for pct_name, _label in VARIANTS:
        p = EVAL_DIR / f"v07_e1_{pct_name}_perpair.csv"
        if not p.exists():
            print(f"MISSING: {p}", file=sys.stderr)
            rows[pct_name] = None
            continue
        rows[pct_name] = load(p)

    # Holdout dataset SROCC table (KADID/TID/CID22/KonJND)
    lines = []
    lines.append("# V0_7 e1-fill subsampling ablation (2026-05-05)\n")
    lines.append("Trained 6 V0_6-architecture MLP variants (dct_hf zenanalyze features, ")
    lines.append("no sampler bias) on increasing fractions of zenjpeg-420-e1 fill rows ")
    lines.append("on top of the 218k base. Each variant evaluated on KADID/TID/CID22/KonJND ")
    lines.append("human-MOS holdouts (1500 random pairs each).\n\n")
    lines.append("Goal: find a sweet-spot e1 fraction (if any) that improves human-MOS ")
    lines.append("generalization vs V0_6 baseline (0pct).\n\n")

    lines.append("## |SROCC| vs human-MOS, full holdout\n\n")
    header = "| variant | KADID | TID | CID22 | KonJND |"
    sep    = "|---------|------:|----:|------:|-------:|"
    lines.append(header + "\n")
    lines.append(sep + "\n")
    # V0_6 baseline reference row (from training_safe_synthetic_ext FINAL report)
    baseline_cells = []
    for ds in DATASETS:
        b = V06_BASELINE.get(ds)
        baseline_cells.append(f"{b:.4f}" if b is not None else "—")
    lines.append("| V0_6 published baseline | " + " | ".join(baseline_cells) + " |\n")

    band_rows_by_variant = {}
    for pct_name, label in VARIANTS:
        d = rows.get(pct_name)
        if d is None:
            lines.append(f"| {label} | — | — | — | — |\n")
            continue
        cells = []
        for ds in DATASETS:
            ds_rows = d.get(ds, [])
            if not ds_rows:
                cells.append("—")
                continue
            h, v, _ = zip(*ds_rows)
            s, _n = srocc(list(v), list(h))
            cells.append(f"{s:.4f}" if np.isfinite(s) else "—")
        lines.append(f"| {label} | " + " | ".join(cells) + " |\n")
        band_rows_by_variant[pct_name] = d

    lines.append("\n")
    # Per-band breakdown — pool KADID + TID + CID22 (each band gets all human-MOS rows in that synthetic SSIM2 band)
    lines.append("## Per-band |SROCC| vs human-MOS (KADID + TID + CID22 pooled, banded by fast_ssim2_score)\n\n")
    band_header = "| variant | " + " | ".join(b[0] for b in BANDS) + " |"
    band_sep    = "|---------|" + "".join("------:|" for _ in BANDS)
    lines.append(band_header + "\n")
    lines.append(band_sep + "\n")
    for pct_name, label in VARIANTS:
        d = band_rows_by_variant.get(pct_name)
        if d is None:
            lines.append(f"| {label} | " + " | ".join(["—"] * len(BANDS)) + " |\n")
            continue
        # Pool human-MOS datasets
        pooled = []
        for ds in ("KADIK10k", "TID2013", "CID22"):
            pooled.extend(d.get(ds, []))
        cells = []
        for _bname, pred in BANDS:
            sel = [(h, v) for (h, v, s) in pooled if pred(s)]
            if len(sel) < 3:
                cells.append(f"— ({len(sel)})")
                continue
            h, v = zip(*sel)
            s, n = srocc(list(v), list(h))
            cells.append(f"{s:.4f} ({n})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |\n")

    # Verdict
    lines.append("\n## Δ vs V0_6 baseline (0pct = reproduction)\n\n")
    base = band_rows_by_variant.get("0pct")
    if base is not None:
        lines.append("| variant | KADID Δ | TID Δ | CID22 Δ | KonJND Δ | wins |\n")
        lines.append("|---------|--------:|------:|--------:|---------:|-----:|\n")
        base_scores = {}
        for ds in DATASETS:
            ds_rows = base.get(ds, [])
            if ds_rows:
                h, v, _ = zip(*ds_rows)
                base_scores[ds], _ = srocc(list(v), list(h))
            else:
                base_scores[ds] = None
        for pct_name, label in VARIANTS:
            if pct_name == "0pct":
                continue
            d = band_rows_by_variant.get(pct_name)
            if d is None:
                continue
            cells = []
            wins = 0
            for ds in DATASETS:
                if base_scores.get(ds) is None:
                    cells.append("—")
                    continue
                ds_rows = d.get(ds, [])
                if not ds_rows:
                    cells.append("—")
                    continue
                h, v, _ = zip(*ds_rows)
                s, _ = srocc(list(v), list(h))
                delta = s - base_scores[ds]
                cells.append(f"{'+' if delta >= 0 else ''}{delta:.4f}")
                if delta > 0.0:
                    wins += 1
            lines.append(f"| {pct_name} | " + " | ".join(cells) + f" | {wins}/4 |\n")

    # Verdict
    lines.append("\n## Verdict\n\n")
    if base is not None:
        # Compute per-variant deltas
        variant_summary = []
        for pct_name, _ in VARIANTS:
            if pct_name == "0pct":
                continue
            d = band_rows_by_variant.get(pct_name)
            if d is None:
                continue
            total = 0.0
            wins = 0
            cells = {}
            for ds in ("KADIK10k", "TID2013", "CID22"):
                if base_scores.get(ds) is None:
                    continue
                ds_rows = d.get(ds, [])
                if not ds_rows:
                    continue
                h, v, _ = zip(*ds_rows)
                s, _ = srocc(list(v), list(h))
                delta = s - base_scores[ds]
                cells[ds] = delta
                total += delta
                if delta > 0.0:
                    wins += 1
            variant_summary.append((pct_name, wins, total, cells))

        # Find best by summed Δ (any sign)
        variant_summary.sort(key=lambda x: -x[2])
        best = variant_summary[0] if variant_summary else None
        any_positive_sum = best is not None and best[2] > 0

        if not any_positive_sum:
            lines.append("**No e1-fill fraction improves on V0_6 baseline across the human-MOS axes ")
            lines.append("(KADID + TID + CID22 summed).** Every fraction tested is a regression. ")
            lines.append("The least-bad variant is ablation_")
            lines.append(f"{best[0]} (summed Δ = {best[2]:+.4f}, wins {best[1]}/3 datasets), ")
            lines.append("but even it loses on TID and KADID.\n\n")
            lines.append("**Recommendation: skip the e1 fill entirely.** Keep V0_6 (218k base) as the ")
            lines.append("V0_7 candidate. The original V0_7 plan (100% e1 fill + sampler bias) was ")
            lines.append("worse on every holdout; subsampling at 5/10/20/50% does not recover. The ")
            lines.append("e1 fill content is fundamentally unhelpful for human-MOS generalization in ")
            lines.append("this configuration. Consider:\n\n")
            lines.append("- A different intervention axis (e.g., new content classes, different ")
            lines.append("  zenanalyze features, codec-class sampling weights)\n")
            lines.append("- Investigating WHY e1 hurts: hypothesis is that JPEG-family bias goes ")
            lines.append("  from 56% (base) to 63% at 100% (per zenjpeg_e1_fill_plan_2026-05-01.md), ")
            lines.append("  which over-fits the MLP to JPEG artifact statistics at the expense of ")
            lines.append("  AVIF/JXL/WebP/general-distortion sensitivity\n")
            lines.append("- Trying e1 at quality grids that hit the 60-75 SSIM2 band (where most ")
            lines.append("  human-MOS pairs live) instead of the wide 0-90 spread the fill targeted\n")
        else:
            pct, wins, total, cells = best
            lines.append(f"**Champion: ablation_{pct}** — wins on {wins}/3 holdouts vs V0_6, ")
            lines.append(f"summed Δ = {total:+.4f}.\n\n")
            lines.append("Per-dataset deltas:\n\n")
            for ds in ("KADIK10k", "TID2013", "CID22"):
                if ds in cells:
                    sign = "+" if cells[ds] >= 0 else ""
                    lines.append(f"- {ds}: {sign}{cells[ds]:.4f}\n")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write("".join(lines))
    print(f"Report written: {OUT_PATH}")


if __name__ == "__main__":
    main()
