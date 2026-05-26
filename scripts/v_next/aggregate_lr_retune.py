#!/usr/bin/env python3
"""SPEED-B lr-retune sweep aggregator.

Runs bake_verdict on every cc4v6_lr*.bin in the sweep dir and aggregates
into one summary table: per-lr SROCC (median across seeds) on each corpus,
compared to the K=1 V6 baseline.

Usage: python3 aggregate_lr_retune.py [SWEEP_DIR]

NOTE: this aggregator already shells to `bake_verdict` (the canonical
bake-on-corpus eval) and only re-parses its SROCC; it does NOT compute any
panel stat itself. The `srocc_only` regex/parse below reads bake_verdict's
output — no hand-rolled stat math. (If you ever need the panel on arbitrary
pairs here, use `from scripts.lib.zen_stats import panel`.)
"""
from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from pathlib import Path

SWEEP_DIR_DEFAULT = "/mnt/v/zen/zensim-eval/speed_b_lr_retune_2026-05-19"
BAKE_VERDICT = "/home/lilith/work/zen/zensim/target/release/bake_verdict"

# K=1 baseline — seed 1 only (task's literal "K=1 reference")
K1_S1_BASELINE = {
    "CID22": 0.8770,
    "KADIK10k": 0.7179,
    "TID2013": 0.7542,
    "KonJND-1k (full)": 0.1962,
    "AIC-3 CTC": 0.7961,
}

# K=1 MEDIAN across 3 seeds (the methodologically correct reference):
# seed 1: CID22=0.8770, KADID=0.7179, TID=0.7542, KonJND=0.1962, AIC3=0.7961
# seed 2: CID22=0.8302, KADID=0.4433, TID=0.4819, KonJND=0.0135, AIC3=0.7681
# seed 3: CID22=0.8440, KADID=0.3875, TID=0.4235, KonJND=0.0119, AIC3=0.7809
K1_MEDIAN_BASELINE = {
    "CID22": 0.8440,
    "KADIK10k": 0.4433,
    "TID2013": 0.4819,
    "KonJND-1k (full)": 0.0135,
    "AIC-3 CTC": 0.7809,
}

# K=32 lr=1e-3 reference regression (cc4v6_lr1e-3_s1, expected from speed_b_verify)
K32_LR1E3_REF = {
    "CID22": 0.8236,
    "KADIK10k": 0.5478,
    "TID2013": 0.6635,
    "KonJND-1k (full)": 0.1252,
    "AIC-3 CTC": 0.7821,
}

# Default baseline for the gate (task-spec literal):
K1_BASELINE = K1_S1_BASELINE


SUMMARY_ROW_RE = re.compile(
    r"^\|\s*(?P<corpus>[^|]+?)\s*\|\s*(?P<n>\d+)\s*\|\s*"
    r"(?P<srocc>-?\d+\.\d+)\s*\|\s*"
    r"(?P<plcc>-?\d+\.\d+)\s*\|\s*"
    r"(?P<krocc>-?\d+\.\d+)\s*\|\s*"
    r"(?P<or>-?\d+\.\d+)\s*\|\s*"
    r"(?P<pwrc>-?\d+\.\d+)\s*\|\s*"
    r"(?P<zrmse>-?\d+\.\d+)"
)


def parse_summary(md_path: Path) -> dict[str, dict[str, float]]:
    """Parse bake_verdict markdown summary table — full panel per corpus.

    Returns: corpus → {srocc, plcc, krocc, or, pwrc, zrmse}
    """
    out: dict[str, dict[str, float]] = {}
    in_summary = False
    for line in md_path.read_text().splitlines():
        if line.startswith("## Summary"):
            in_summary = True
            continue
        if in_summary:
            if line.startswith("## "):
                break
            m = SUMMARY_ROW_RE.match(line)
            if m:
                corpus = m.group("corpus").strip()
                out[corpus] = {
                    "srocc": float(m.group("srocc")),
                    "plcc": float(m.group("plcc")),
                    "krocc": float(m.group("krocc")),
                    "or": float(m.group("or")),
                    "pwrc": float(m.group("pwrc")),
                    "zrmse": float(m.group("zrmse")),
                }
    return out


def srocc_only(d: dict[str, dict[str, float]]) -> dict[str, float]:
    """Convenience: extract SROCC dict from full-panel parse."""
    return {c: v["srocc"] for c, v in d.items()}


def run_verdict(bake_path: Path, verdict_path: Path) -> bool:
    if verdict_path.exists() and verdict_path.stat().st_size > 0:
        return True
    print(f"  running bake_verdict on {bake_path.name}", flush=True)
    try:
        subprocess.run(
            [BAKE_VERDICT, "--bake", str(bake_path), "--output", str(verdict_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR {bake_path.name}: {e.stderr}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", nargs="?", default=SWEEP_DIR_DEFAULT)
    args = ap.parse_args()
    sweep_dir = Path(args.sweep_dir)
    verdict_dir = sweep_dir / "verdicts"
    verdict_dir.mkdir(parents=True, exist_ok=True)

    bakes = sorted(sweep_dir.glob("cc4v6_lr*_s*.bin"))
    if not bakes:
        print(f"no bakes found in {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    # (lr_tag, seed, bake_path)
    by_cell: list[tuple[str, int, Path]] = []
    bake_re = re.compile(r"cc4v6_lr(?P<lr>[^_]+)_s(?P<seed>\d+)\.bin")
    for bake in bakes:
        m = bake_re.match(bake.name)
        if not m:
            continue
        by_cell.append((m.group("lr"), int(m.group("seed")), bake))

    # Run bake_verdict on each
    print(f"Found {len(by_cell)} bakes. Running bake_verdict on each...", flush=True)
    results: dict[str, dict[int, dict[str, float]]] = {}
    for lr_tag, seed, bake_path in by_cell:
        verdict_path = verdict_dir / f"{bake_path.stem}.md"
        if not run_verdict(bake_path, verdict_path):
            continue
        panel = parse_summary(verdict_path)
        # Reduce to SROCC dict for downstream median/delta arithmetic
        results.setdefault(lr_tag, {})[seed] = srocc_only(panel)

    # LR decode order — preserve the sweep order
    lr_order = ["1e-3", "1p5e-3", "2p83e-3", "5p66e-3", "8e-3"]
    lr_display = {
        "1e-3": "1.0e-3",
        "1p5e-3": "1.5e-3",
        "2p83e-3": "2.83e-3",
        "5p66e-3": "5.66e-3",
        "8e-3": "8.0e-3",
    }

    corpora = ["CID22", "KADIK10k", "TID2013", "KonJND-1k (full)", "AIC-3 CTC"]

    # === Table 1: per-(lr, seed) SROCC ===
    print("\n## Per-(lr, seed) SROCC (raw, K=32)\n")
    header = "| lr | seed | " + " | ".join(corpora) + " |"
    sep = "|---|---:|" + "|".join("---:" for _ in corpora) + "|"
    print(header)
    print(sep)
    for lr_tag in lr_order:
        for seed in (1, 2, 3):
            row_srocc = results.get(lr_tag, {}).get(seed, {})
            if not row_srocc:
                vals = "n/a"
                row = f"| {lr_display.get(lr_tag, lr_tag)} | {seed} | "
                row += " | ".join("n/a" for _ in corpora) + " |"
            else:
                row = f"| {lr_display.get(lr_tag, lr_tag)} | {seed} | "
                row += " | ".join(f"{row_srocc.get(c, float('nan')):.4f}" for c in corpora) + " |"
            print(row)

    # === Table 2: per-lr median SROCC + deltas vs K=1 baseline ===
    print("\n## Per-lr median SROCC across 3 seeds + Δ vs K=1 reference\n")
    print("Two reference rows: **K=1 s1** (task-spec literal, a lucky outlier seed) "
          "and **K=1 median** across 3 seeds (methodologically correct).\n")
    header = "| lr | " + " | ".join(corpora) + " |"
    sep = "|---|" + "|".join("---:" for _ in corpora) + "|"
    print(header)
    print(sep)
    print("| **K=1 s1 (task reference)** | "
          + " | ".join(f"{K1_S1_BASELINE[c]:.4f}" for c in corpora) + " |")
    print("| **K=1 median (3 seeds)** | "
          + " | ".join(f"{K1_MEDIAN_BASELINE[c]:.4f}" for c in corpora) + " |")
    print("| **K=32 lr=1e-3 verified (speed_b_verify s1)** | "
          + " | ".join(f"{K32_LR1E3_REF[c]:.4f} (Δs1={K32_LR1E3_REF[c] - K1_S1_BASELINE[c]:+.4f})"
                       for c in corpora) + " |")

    pass_table: dict[str, dict[str, bool]] = {}  # lr → corpus → within-tolerance
    median_table: dict[str, dict[str, float]] = {}
    for lr_tag in lr_order:
        cells = results.get(lr_tag, {})
        if not cells:
            continue
        per_corpus = {}
        deltas = {}
        within = {}
        for c in corpora:
            vals = [cells[s].get(c, float("nan")) for s in sorted(cells.keys())
                    if c in cells[s]]
            vals = [v for v in vals if v == v]  # drop NaN
            if vals:
                med = statistics.median(vals)
                per_corpus[c] = med
                deltas[c] = med - K1_BASELINE[c]
                within[c] = abs(deltas[c]) <= 0.01
            else:
                per_corpus[c] = float("nan")
                deltas[c] = float("nan")
                within[c] = False
        median_table[lr_tag] = per_corpus
        pass_table[lr_tag] = within
        row = f"| {lr_display.get(lr_tag, lr_tag)} | "
        row += " | ".join(
            f"{per_corpus[c]:.4f} (Δs1={deltas[c]:+.4f})" for c in corpora) + " |"
        print(row)

    # === Gate decision ===
    print("\n## Winning lr (within ±0.01 of K=1 s1 reference per corpus)\n")
    winners_s1 = []
    for lr_tag in lr_order:
        if lr_tag not in pass_table:
            continue
        all_pass = all(pass_table[lr_tag][c] for c in corpora)
        pass_count = sum(1 for c in corpora if pass_table[lr_tag][c])
        print(f"  lr={lr_display.get(lr_tag, lr_tag)}: "
              f"{pass_count}/5 corpora pass ±0.01 vs K=1 s1 — "
              f"{'WIN' if all_pass else 'fail'}")
        if all_pass:
            winners_s1.append(lr_tag)

    if winners_s1:
        print(f"\nWINNING lr (vs K=1 s1): {[lr_display[w] for w in winners_s1]}")
    else:
        print("\nNO lr passes the ±0.01-vs-s1 gate.")

    # === Alternative gate: vs K=1 median ===
    print("\n## Alternative: Δ vs K=1 median across 3 seeds\n")
    print("Use this when seed 1 of K=1 looks like an outlier draw (e.g. K=1 KADID s1=0.72 "
          "vs s2/s3 medians 0.39/0.44).\n")
    winners_med = []
    for lr_tag in lr_order:
        per_corpus = median_table.get(lr_tag, {})
        if not per_corpus:
            continue
        deltas_med = {c: per_corpus[c] - K1_MEDIAN_BASELINE[c] for c in corpora}
        within_med = {c: abs(deltas_med[c]) <= 0.01 for c in corpora}
        all_pass = all(within_med[c] for c in corpora)
        pass_count = sum(1 for c in corpora if within_med[c])
        delta_str = " ".join(f"{c}={deltas_med[c]:+.3f}" for c in corpora)
        print(f"  lr={lr_display.get(lr_tag, lr_tag)}: "
              f"{pass_count}/5 corpora pass ±0.01 vs K=1 median  | {delta_str}")
        if all_pass:
            winners_med.append(lr_tag)

    if winners_med:
        print(f"\nWINNING lr (vs K=1 median): {[lr_display[w] for w in winners_med]}")
    else:
        print("\nNO lr passes the ±0.01-vs-median gate.")

    # Save median table to disk for easier downstream consumption
    out_tsv = sweep_dir / "lr_retune_summary.tsv"
    with out_tsv.open("w") as f:
        f.write("lr\tseed\t" + "\t".join(corpora) + "\n")
        for lr_tag in lr_order:
            for seed in sorted(results.get(lr_tag, {}).keys()):
                cell = results[lr_tag][seed]
                f.write(f"{lr_display.get(lr_tag, lr_tag)}\t{seed}\t")
                f.write("\t".join(f"{cell.get(c, float('nan')):.4f}" for c in corpora))
                f.write("\n")
    print(f"\nSaved {out_tsv}")


if __name__ == "__main__":
    main()
