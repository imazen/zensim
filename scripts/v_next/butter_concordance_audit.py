#!/usr/bin/env python3
"""Audit ssim2 ↔ butter ranking concordance per (source, codec) curve.

Per the CID22 paper, ssim2 is less reliable in q < 30 and q > 95 regimes.
A simple cross-check is whether butteraugli agrees on within-curve ranking:
for a series of (source, codec, q1<q2<...<qn) pairs, do ssim2 and butter
both predict the same monotonic ranking?

Concordant curves: monotonic in both metrics.
Discordant curves: ranking disagrees on at least one adjacent-q step.

We compute Spearman SROCC within each curve as a continuous measure and
also count strictly-discordant adjacent-q pairs.

Usage:
  python3 butter_concordance_audit.py \\
    --csv /mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv \\
    --report /tmp/zensim_loop/butter_concordance_audit.tsv

DEPRECATED STAT MATH: the inline `spearman` here is superseded by the
canonical Rust `panel` (zensim-validate/src/bin/panel.rs). For NEW work on
arbitrary (predicted, target) pairs use:
    from scripts.lib.zen_stats import srocc, panel   # shells to Rust `panel`
verified to scipy <= 1e-9 by scripts/verify_panel_parity.py.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def spearman_pair(xs, ys):
    """Simple Spearman SROCC for two ranking series (n usually small)."""
    n = len(xs)
    if n < 2:
        return 1.0
    # Convert to ranks (mean-rank ties)
    def ranks(v):
        sv = sorted(enumerate(v), key=lambda t: t[1])
        r = [0.0] * len(v)
        # average rank assignment for ties (simple, slow)
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1][1] == sv[i][1]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[sv[k][0]] = avg
            i = j + 1
        return r
    rx = ranks(xs)
    ry = ranks(ys)
    mean = (n + 1) / 2.0
    num = sum((rx[i] - mean) * (ry[i] - mean) for i in range(n))
    dx = sum((rx[i] - mean) ** 2 for i in range(n))
    dy = sum((ry[i] - mean) ** 2 for i in range(n))
    den = (dx * dy) ** 0.5
    return num / den if den > 1e-12 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path,
                    help="Per-curve TSV: source_path, codec, n, srocc_ssim2_vs_butter, discordant_adj_pairs")
    ap.add_argument("--ssim2-col", default="gpu_ssimulacra2")
    ap.add_argument("--butter-col", default="gpu_butteraugli")
    args = ap.parse_args()

    grouped = defaultdict(list)
    n_in = 0
    with open(args.csv) as f:
        r = csv.DictReader(f)
        for row in r:
            n_in += 1
            try:
                ssim2 = float(row[args.ssim2_col])
                butter = float(row[args.butter_col])
                q = int(row["quality"])
                src = row["source_path"]
                codec = row["codec"]
            except (KeyError, ValueError):
                continue
            grouped[(src, codec)].append((q, ssim2, butter))

    n_total_curves = 0
    n_concordant_full = 0  # SROCC ≈ -1 (since ssim2 grows with q, butter shrinks → negative)
    n_at_least_one_discord = 0
    total_adj_pairs = 0
    total_discord_adj = 0
    srocc_buckets = [0] * 5  # [-1,-0.9), [-0.9,-0.5), [-0.5,0.5), [0.5,0.9), [0.9,1]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as out:
        out.write("source_path\tcodec\tn\tsrocc_ssim2_vs_butter\tdiscordant_adj_pairs\tadj_pairs\n")
        for (src, codec), rows in grouped.items():
            if len(rows) < 2:
                continue
            rows.sort(key=lambda t: t[0])
            ssim2s = [r[1] for r in rows]
            butters = [r[2] for r in rows]
            s = spearman_pair(ssim2s, butters)
            # adjacent-q concordance: higher q should give HIGHER ssim2 AND LOWER butter
            n_adj = len(rows) - 1
            n_discord = 0
            for i in range(n_adj):
                s_up = ssim2s[i + 1] > ssim2s[i]
                b_down = butters[i + 1] < butters[i]
                # both should be True for full concordance
                if s_up != b_down:
                    n_discord += 1
            out.write(f"{src}\t{codec}\t{len(rows)}\t{s:.4f}\t{n_discord}\t{n_adj}\n")

            n_total_curves += 1
            total_adj_pairs += n_adj
            total_discord_adj += n_discord
            if n_discord == 0:
                n_concordant_full += 1
            else:
                n_at_least_one_discord += 1
            if s <= -0.9: srocc_buckets[0] += 1
            elif s <= -0.5: srocc_buckets[1] += 1
            elif s <= 0.5: srocc_buckets[2] += 1
            elif s <= 0.9: srocc_buckets[3] += 1
            else: srocc_buckets[4] += 1

    print(f"Read {n_in} rows; analyzed {n_total_curves} (source, codec) curves",
          file=sys.stderr)
    print(f"  Curves with NO adjacent-q disagreement: {n_concordant_full} "
          f"({n_concordant_full/n_total_curves*100:.1f}%)", file=sys.stderr)
    print(f"  Curves with ≥1 disagreement: {n_at_least_one_discord} "
          f"({n_at_least_one_discord/n_total_curves*100:.1f}%)", file=sys.stderr)
    print(f"  Adjacent-q pairs: {total_adj_pairs}; discordant: {total_discord_adj} "
          f"({total_discord_adj/total_adj_pairs*100:.2f}%)", file=sys.stderr)
    print(f"  SROCC distribution: ", file=sys.stderr)
    labels = ["[-1.0, -0.9]", "(-0.9, -0.5]", "(-0.5, 0.5]", "(0.5, 0.9]", "(0.9, 1.0]"]
    for lab, c in zip(labels, srocc_buckets):
        print(f"    {lab}: {c} ({c/n_total_curves*100:.1f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
