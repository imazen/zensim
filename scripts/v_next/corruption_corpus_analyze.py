#!/usr/bin/env python3
"""Analyze the multi-metric corruption-corpus TSV.

For each metric, computes:
  - Gate-pass rate (corruption ranks below q20 OR q10 anchor)
  - Per-family pass rate
  - Per-region pass rate (the "subtlety axis")
  - Discriminative gap: median(corruption - q20) in the metric's natural direction

Also compares the metric to the existing zensim profiles (V0_2, v47, Cell 5)
once those are scored — TODO if the user wants the full comparison.
"""
import sys, csv, os
from collections import defaultdict
from pathlib import Path
import numpy as np

TSV = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/corruption_multimetric_2026-05-28.tsv")

# Metric directions: HIGH = good quality, or HIGH = bad?
DIRECTIONS = {
    # column        : (label,           higher_is_better)
    "ssim2_gpu":         ("ssim2-gpu",   True),
    "butter_max_gpu":    ("butter-max",  False),
    "butter_pnorm3_gpu": ("butter-p3",   False),
    "cvvdp":             ("cvvdp",       True),
    "dssim_gpu":         ("dssim",       False),
}

def gate_pass(corr_val, anchor_val, higher_is_better):
    """Returns True if `corr_val` indicates WORSE quality than `anchor_val`.

    higher_is_better=True (e.g. ssim2):  corr_val < anchor_val == gate_pass
    higher_is_better=False (e.g. butter): corr_val > anchor_val == gate_pass
    """
    if corr_val is None or anchor_val is None:
        return None
    if higher_is_better:
        return corr_val < anchor_val
    return corr_val > anchor_val

def main():
    if not TSV.exists():
        print(f"TSV not found: {TSV}", file=sys.stderr)
        sys.exit(1)

    # Reshape: index by base name (ref__family__region__sev) → {kind: row}
    by_name = defaultdict(dict)
    with open(TSV) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            by_name[row["name"]][row["kind"]] = row

    triples = []
    for name, kinds in by_name.items():
        if "corruption" in kinds and "q20" in kinds and "q10" in kinds:
            triples.append((name, kinds["corruption"], kinds["q20"], kinds["q10"]))

    n = len(triples)
    print(f"# Corruption corpus multi-metric analysis ({n} complete triples)\n")
    print(f"## Gate pass rate by metric — `score(corruption)` vs `score(q20)` and `score(q10)`\n")
    print(f"| metric | dir | pass@q20 | pass@q10 |")
    print(f"|---|---|--:|--:|")
    for col, (label, hib) in DIRECTIONS.items():
        n_pass_q20 = 0
        n_pass_q10 = 0
        n_eval = 0
        for name, c, q20, q10 in triples:
            try:
                cv = float(c[col]); v20 = float(q20[col]); v10 = float(q10[col])
            except (ValueError, KeyError, TypeError):
                continue
            if not (np.isfinite(cv) and np.isfinite(v20) and np.isfinite(v10)):
                continue
            n_eval += 1
            if gate_pass(cv, v20, hib): n_pass_q20 += 1
            if gate_pass(cv, v10, hib): n_pass_q10 += 1
        dir_str = "high=good" if hib else "high=bad"
        if n_eval > 0:
            print(f"| {label} | {dir_str} | {n_pass_q20}/{n_eval} ({100*n_pass_q20/n_eval:.1f}%) | {n_pass_q10}/{n_eval} ({100*n_pass_q10/n_eval:.1f}%) |")

    # Per-region pass rate
    print(f"\n## Per-region pass@q20 (smaller region = harder/subtler corruption)\n")
    # Direction headers
    metric_keys = list(DIRECTIONS.keys())
    metric_labels = [DIRECTIONS[k][0] for k in metric_keys]
    print(f"| region | n | " + " | ".join(metric_labels) + " |")
    print(f"|---|--:|" + "|".join(["--:" for _ in metric_keys]) + "|")
    region_stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    region_n = defaultdict(int)
    for name, c, q20, q10 in triples:
        region = c["region"]
        region_n[region] += 1
        for col in metric_keys:
            try:
                cv = float(c[col]); v20 = float(q20[col])
            except (ValueError, KeyError, TypeError):
                continue
            if not (np.isfinite(cv) and np.isfinite(v20)):
                continue
            label, hib = DIRECTIONS[col]
            if gate_pass(cv, v20, hib):
                region_stats[region][col][0] += 1
            region_stats[region][col][1] += 1
    for region in ["whole", "frac2", "frac4", "sq64", "sq16", "sq8"]:
        if region not in region_stats:
            continue
        row = [region, str(region_n[region])]
        for col in metric_keys:
            pn, total = region_stats[region][col]
            row.append(f"{pn}/{total} ({100*pn/total:.1f}%)" if total > 0 else "n/a")
        print("| " + " | ".join(row) + " |")

    # Per-family pass rate
    print(f"\n## Per-family pass@q20\n")
    fam_stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    fam_n = defaultdict(int)
    for name, c, q20, q10 in triples:
        family = c["family"]
        fam_n[family] += 1
        for col in metric_keys:
            try:
                cv = float(c[col]); v20 = float(q20[col])
            except (ValueError, KeyError, TypeError):
                continue
            if not (np.isfinite(cv) and np.isfinite(v20)):
                continue
            label, hib = DIRECTIONS[col]
            if gate_pass(cv, v20, hib):
                fam_stats[family][col][0] += 1
            fam_stats[family][col][1] += 1
    print(f"| family | n | " + " | ".join(metric_labels) + " |")
    print(f"|---|--:|" + "|".join(["--:" for _ in metric_keys]) + "|")
    for fam in sorted(fam_stats.keys()):
        row = [fam, str(fam_n[fam])]
        for col in metric_keys:
            pn, total = fam_stats[fam][col]
            row.append(f"{100*pn/total:.0f}%" if total > 0 else "n/a")
        print("| " + " | ".join(row) + " |")

    # Discriminative gap: median(corruption - q20) — bigger = better separation
    print(f"\n## Discriminative gap (median magnitude `|corruption - q20|`, "
          f"normalized to anchor range)\n")
    print("Larger = the corruption-score is more clearly separated from the honest-lq anchor. "
          "Normalized by (anchor max - anchor min) so metrics with different natural ranges "
          "compare fairly.\n")
    print("| metric | median gap | p25 gap | p75 gap | "
          "(unit normalized to anchor variability) |")
    print("|---|--:|--:|--:|---|")
    for col, (label, hib) in DIRECTIONS.items():
        gaps = []
        anchor_vals = []
        for name, c, q20, q10 in triples:
            try:
                cv = float(c[col]); v20 = float(q20[col])
            except (ValueError, KeyError, TypeError):
                continue
            if not (np.isfinite(cv) and np.isfinite(v20)):
                continue
            if hib:
                gap = v20 - cv      # corruption should be BELOW; gap > 0 = good
            else:
                gap = cv - v20      # corruption should be ABOVE; gap > 0 = good
            gaps.append(gap)
            anchor_vals.append(v20)
        if not gaps:
            continue
        gaps = np.array(gaps)
        anchor_vals = np.array(anchor_vals)
        anchor_span = np.max(anchor_vals) - np.min(anchor_vals)
        if anchor_span <= 0:
            anchor_span = 1.0
        norm = gaps / anchor_span
        print(f"| {label} | {np.median(norm):+.3f} | {np.percentile(norm, 25):+.3f} | "
              f"{np.percentile(norm, 75):+.3f} | gap units = anchor variability ({anchor_span:.3f}) |")

if __name__ == "__main__":
    main()
