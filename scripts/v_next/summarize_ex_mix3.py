#!/usr/bin/env python3
"""EX-MIX3: parse bake_verdict outputs into a 5-seed CI table.

Reads /mnt/v/zen/zensim-eval/ex_mix3_2026-05-18/verdicts/exmix3_<variant>_s<seed>.md
files, extracts per-corpus aggregate SROCC + Z-RMSE + PWRC, computes mean ± std
across 5 seeds per variant.

Writes /mnt/v/zen/zensim-eval/ex_mix3_2026-05-18/SUMMARY_5seed.md
"""

from __future__ import annotations
import re
import sys
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

VERDICT_DIR = Path("/mnt/v/zen/zensim-eval/ex_mix3_2026-05-18/verdicts")
OUT_PATH = Path("/mnt/v/zen/zensim-eval/ex_mix3_2026-05-18/SUMMARY_5seed.md")

VARIANTS = ["cv33_iw33_sm33", "cv30_iw40_sm30", "cv40_iw40_sm20"]
SEEDS = [1, 2, 3, 4, 5]
CORPORA = ["CID22", "KADIK10k", "TID2013", "KonJND-1k (full)", "AIC-3 CTC"]
CORPUS_DISPLAY = {
    "CID22": "CID22",
    "KADIK10k": "KADID",
    "TID2013": "TID",
    "KonJND-1k (full)": "KonJND",
    "AIC-3 CTC": "AIC-3",
}


def parse_verdict(path: Path) -> dict[str, dict[str, float]]:
    """Parse a bake_verdict.md and extract Summary table per-corpus stats."""
    text = path.read_text()
    # Find Summary section
    out = {}
    # Match table rows like:
    # | CID22 | 4292 | 0.8558 | 0.8551 | 0.6651 | 0.0473 | 0.9102 | 0.518 |
    pattern = re.compile(
        r"^\|\s*([A-Za-z0-9\-_\s]+(?:\s*\([^)]+\))?)\s*\|\s*(\d+)\s*\|\s*([\d.\-]+)\s*\|\s*([\d.\-]+)\s*\|\s*([\d.\-]+)\s*\|\s*([\d.\-]+)\s*\|\s*([\d.\-]+)\s*\|\s*([\d.\-]+)\s*\|\s*$",
        re.MULTILINE,
    )
    for m in pattern.finditer(text):
        corpus = m.group(1).strip()
        if corpus not in CORPORA:
            continue
        n = int(m.group(2))
        srocc = float(m.group(3))
        plcc = float(m.group(4))
        krocc = float(m.group(5))
        outlier = float(m.group(6))
        pwrc = float(m.group(7))
        zrmse = float(m.group(8))
        out[corpus] = {
            "n": n, "srocc": srocc, "plcc": plcc, "krocc": krocc,
            "or": outlier, "pwrc": pwrc, "zrmse": zrmse,
        }
    return out


def main():
    # Read every verdict
    data: dict[str, dict[int, dict]] = defaultdict(dict)
    missing = []
    for variant in VARIANTS:
        for seed in SEEDS:
            p = VERDICT_DIR / f"exmix3_{variant}_s{seed}.md"
            if not p.exists():
                missing.append(p)
                continue
            try:
                data[variant][seed] = parse_verdict(p)
            except Exception as e:
                print(f"  PARSE FAIL {p}: {e}", file=sys.stderr)
                missing.append(p)
    if missing:
        print(f"WARN: {len(missing)} missing verdicts:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)

    # Aggregate
    rows = []
    rows.append("# EX-MIX3 5-seed CI summary")
    rows.append("")
    rows.append(f"Source: {VERDICT_DIR}")
    rows.append("")
    rows.append("Means ± σ across 5 seeds per variant. Each cell is the corpus aggregate SROCC; PWRC/Z-RMSE in parentheses where relevant.")
    rows.append("")
    rows.append("## SROCC table")
    rows.append("")
    rows.append("| Variant | CID22 | KADID | TID | KonJND | AIC-3 |")
    rows.append("|---|---|---|---|---|---|")
    for variant in VARIANTS:
        seeds_data = data.get(variant, {})
        cells = []
        for corpus in CORPORA:
            vals = [seeds_data[s][corpus]["srocc"] for s in SEEDS
                    if s in seeds_data and corpus in seeds_data[s]]
            if not vals:
                cells.append("n/a")
            elif len(vals) == 1:
                cells.append(f"{vals[0]:.4f} (n=1)")
            else:
                m = mean(vals)
                s = stdev(vals)
                cells.append(f"{m:.4f}±{s:.4f}")
        rows.append(f"| {variant} | " + " | ".join(cells) + " |")

    rows.append("")
    rows.append("## PWRC table")
    rows.append("")
    rows.append("| Variant | CID22 | KADID | TID | KonJND | AIC-3 |")
    rows.append("|---|---|---|---|---|---|")
    for variant in VARIANTS:
        seeds_data = data.get(variant, {})
        cells = []
        for corpus in CORPORA:
            vals = [seeds_data[s][corpus]["pwrc"] for s in SEEDS
                    if s in seeds_data and corpus in seeds_data[s]]
            if not vals:
                cells.append("n/a")
            elif len(vals) == 1:
                cells.append(f"{vals[0]:.4f} (n=1)")
            else:
                m = mean(vals)
                s = stdev(vals)
                cells.append(f"{m:.4f}±{s:.4f}")
        rows.append(f"| {variant} | " + " | ".join(cells) + " |")

    rows.append("")
    rows.append("## Z-RMSE table")
    rows.append("")
    rows.append("| Variant | CID22 | KADID | TID | KonJND | AIC-3 |")
    rows.append("|---|---|---|---|---|---|")
    for variant in VARIANTS:
        seeds_data = data.get(variant, {})
        cells = []
        for corpus in CORPORA:
            vals = [seeds_data[s][corpus]["zrmse"] for s in SEEDS
                    if s in seeds_data and corpus in seeds_data[s]]
            if not vals:
                cells.append("n/a")
            elif len(vals) == 1:
                cells.append(f"{vals[0]:.4f} (n=1)")
            else:
                m = mean(vals)
                s = stdev(vals)
                cells.append(f"{m:.4f}±{s:.4f}")
        rows.append(f"| {variant} | " + " | ".join(cells) + " |")

    # Baseline (single-seed) for reference
    rows.append("")
    rows.append("## Baseline references (single seed, for context)")
    rows.append("")
    rows.append("| Bake | CID22 | KADID | TID | KonJND | AIC-3 |")
    rows.append("|---|---|---|---|---|---|")
    rows.append("| V_22 noLARGE 372feat s1 | 0.8558 | 0.9336 | 0.8904 | 0.8369 | 0.8107 |")
    rows.append("")
    rows.append("(V_22-LARGE+iwssim 5-seed: CID22 0.8339±0.007, KADID 0.9673, TID 0.9726, KonJND 0.8869, AIC-3 0.7872 per benchmarks/v22_mix_LARGE_iwssim_methodology_2026-05-18.md — but uses 300 features, NOT directly comparable to 372-feat EX-MIX3)")

    OUT_PATH.write_text("\n".join(rows))
    print(f"WROTE: {OUT_PATH}")
    print()
    print("\n".join(rows))


if __name__ == "__main__":
    main()
