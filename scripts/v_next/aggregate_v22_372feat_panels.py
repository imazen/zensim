#!/usr/bin/env python3
"""Aggregate 5-seed CI panels for V_22-372feat experiments + V_22-300 baseline.

Reads per-bake bake_verdict markdown files, parses the Summary table,
and emits a per-corpus mean ± std table for V_22-372feat-5grp,
V_22-372feat-noLARGE, V_22-300-baseline, plus a Pareto-gate verdict.

Output: /mnt/v/zen/zensim-eval/v22_372feat_2026-05-18/SUMMARY_5seed.md
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

OUT_DIR = Path("/mnt/v/zen/zensim-eval/v22_372feat_2026-05-18")
BASELINE_DIR = OUT_DIR / "baselines"

CORPORA = ["CID22", "KADIK10k", "TID2013", "KonJND-1k (full)", "AIC-3 CTC"]


def parse_summary(md_path: Path) -> dict[str, dict[str, float]]:
    """Parse the Summary table; return {corpus: {metric: value}}."""
    rows: dict[str, dict[str, float]] = {}
    if not md_path.exists():
        return rows
    txt = md_path.read_text()
    # Summary table after "## Summary"
    m = re.search(r"^## Summary[^\n]*\n[\s\S]*?\n\n", txt, re.MULTILINE)
    if not m:
        return rows
    block = m.group(0)
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < 8:
            continue
        corpus = cells[0]
        if corpus in ("Corpus", "---") or corpus.startswith("---") or corpus.startswith(":"):
            continue
        try:
            n = int(cells[1])
            srocc = float(cells[2])
            plcc = float(cells[3])
            krocc = float(cells[4])
            outr = float(cells[5])
            pwrc = float(cells[6])
            zrmse = float(cells[7])
        except ValueError:
            continue
        rows[corpus] = {
            "n": n, "SROCC": srocc, "PLCC": plcc, "KROCC": krocc,
            "OR": outr, "PWRC": pwrc, "Z-RMSE": zrmse,
        }
    return rows


def collect_5seed(pattern: str, label_dir: Path | None = None) -> dict[str, list[float]]:
    """Collect 5-seed SROCC/etc panel for one variant.

    Returns dict[corpus] -> list of {metric: value} (one per seed).
    """
    seeds = []
    base = label_dir or OUT_DIR
    for s in (1, 2, 3, 4, 5):
        md = base / pattern.format(seed=s)
        rows = parse_summary(md)
        if rows:
            seeds.append(rows)
    return seeds


def stats(seeds: list[dict[str, dict[str, float]]], corpus: str, metric: str) -> tuple[float, float, int]:
    vals = []
    for s in seeds:
        if corpus in s and metric in s[corpus]:
            vals.append(s[corpus][metric])
    if not vals:
        return (float("nan"), float("nan"), 0)
    return (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0, len(vals))


def fmt_pm(mean: float, std: float, n: int) -> str:
    if n == 0:
        return "n/a"
    return f"{mean:.4f}±{std:.4f}"


def main() -> None:
    variants = [
        ("V_22-300 baseline (ship)",     "baselines", "verdict_v22_baseline_s{seed}.md"),
        ("V_22-372feat 5-group",         None,        "verdict_v22_372feat_s{seed}_h128.md"),
        ("V_22-372feat noLARGE",         None,        "verdict_v22_372feat_noLARGE_s{seed}_h128.md"),
    ]

    panels = {}
    for label, subdir, pat in variants:
        base = OUT_DIR / subdir if subdir else OUT_DIR
        seeds = collect_5seed(pat, base)
        panels[label] = seeds
        print(f"  {label}: {len(seeds)} seeds")

    # Build the markdown table.
    lines: list[str] = []
    lines.append("# V_22-372feat 5-seed CI vs V_22-300 baseline\n")
    lines.append(f"_Generated 2026-05-18. {len(panels)} variants × 5 seeds × 5 corpora._\n")

    # Per-corpus table — SROCC only is the headline; full Mohammadi
    # panel one corpus at a time.
    lines.append("\n## Headline SROCC (5-seed mean ± std)\n")
    lines.append("| Corpus | " + " | ".join(panels.keys()) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(panels)) + "|")
    for corpus in CORPORA:
        cells = []
        for label, seeds in panels.items():
            m, s, n = stats(seeds, corpus, "SROCC")
            cells.append(fmt_pm(m, s, n))
        lines.append(f"| {corpus} | " + " | ".join(cells) + " |")

    # Pareto gate
    lines.append("\n## Pareto gate (target = baseline + delta)\n")
    pareto_gates = {
        "CID22": ("SROCC", +0.005),
        "KADIK10k": ("SROCC", -0.005),  # within −0.005
        "TID2013": ("SROCC", -0.005),
        "KonJND-1k (full)": ("SROCC", 0.0),  # close (even +0.000 is interesting)
        "AIC-3 CTC": ("SROCC", +0.005),
    }
    lines.append("| Corpus | Gate vs baseline | V_22-372feat-5grp | V_22-372feat-noLARGE | Verdict-5grp | Verdict-noLARGE |")
    lines.append("|---|---|---:|---:|---|---|")
    base_seeds = panels.get("V_22-300 baseline (ship)", [])
    for corpus in CORPORA:
        metric, delta = pareto_gates.get(corpus, ("SROCC", 0.0))
        base_m, _, _ = stats(base_seeds, corpus, metric)
        target = base_m + delta
        gate_str = f"{'≥' if delta>=0 else '−≤'} {base_m:.4f}{delta:+.4f} = {target:.4f}"
        verdicts = []
        cells = []
        for var in ("V_22-372feat 5-group", "V_22-372feat noLARGE"):
            v_m, _, _ = stats(panels.get(var, []), corpus, metric)
            cells.append(f"{v_m:.4f}")
            if base_m != base_m:  # NaN
                verdicts.append("?")
            else:
                if delta >= 0:
                    verdicts.append("PASS" if v_m >= target else f"FAIL ({v_m-target:+.4f})")
                else:
                    verdicts.append("PASS" if v_m >= target else f"FAIL ({v_m-target:+.4f})")
        lines.append(f"| {corpus} | {gate_str} | {cells[0]} | {cells[1]} | {verdicts[0]} | {verdicts[1]} |")

    # Full Mohammadi panel per corpus
    for corpus in CORPORA:
        lines.append(f"\n## {corpus} full panel (5-seed mean ± std)\n")
        lines.append("| Variant | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for label, seeds in panels.items():
            cells = []
            for metric in ("SROCC", "PLCC", "KROCC", "OR", "PWRC", "Z-RMSE"):
                m, s, n = stats(seeds, corpus, metric)
                cells.append(fmt_pm(m, s, n))
            lines.append(f"| {label} | " + " | ".join(cells) + " |")

    out = OUT_DIR / "SUMMARY_5seed.md"
    out.write_text("\n".join(lines))
    print(f"\nWrote {out}")
    print("\n".join(lines[:30]))


if __name__ == "__main__":
    main()
