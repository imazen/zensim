#!/usr/bin/env python3
"""Produce the Phase 2 v11 retrain diff vs v11 ship.

Parses two `bake_verdict --output <md>` runs and emits:
- TSV with per-corpus + per-band SROCC/PLCC/KROCC/OR/PWRC/Z-RMSE diff
- A compact markdown summary fragment

Usage:
  python3 scripts/v_next/yj_at_phase2_diff.py \\
    --ship-md benchmarks/yj_autotransforms_retrain_2026-05-25/v11_ship_baseline_verdict.md \\
    --candidate-md benchmarks/yj_autotransforms_retrain_2026-05-25/v11_yj_at_verdict.md \\
    --out-tsv benchmarks/yj_autotransforms_retrain_2026-05-25/bake_verdict_diff.tsv \\
    --out-md benchmarks/yj_autotransforms_retrain_2026-05-25/diff_summary.md
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_aggregate_table(md: str) -> dict[str, dict[str, float]]:
    """Returns {corpus: {SROCC, PLCC, KROCC, OR, PWRC, Z-RMSE}}.

    Reads from the top 'Summary' section.
    """
    result: dict[str, dict[str, float]] = {}
    # Find the "## Summary" section (one row per corpus).
    m = re.search(
        r"## Summary.*?\n\|\s*Corpus.*?\n\|.*?\n((?:\|.*?\n)+?)(?:\n|$)",
        md,
        re.DOTALL,
    )
    if not m:
        return result
    for row in m.group(1).strip().split("\n"):
        cells = [c.strip() for c in row.strip("|").split("|")]
        if len(cells) < 8:
            continue
        corpus = cells[0]
        try:
            result[corpus] = {
                "n": int(cells[1]),
                "SROCC": float(cells[2]),
                "PLCC": float(cells[3]),
                "KROCC": float(cells[4]),
                "OR": float(cells[5]),
                "PWRC": float(cells[6]),
                "Z-RMSE": float(cells[7]),
            }
        except ValueError:
            continue
    return result


def parse_per_band(md: str) -> dict[str, dict[str, dict[str, float]]]:
    """Returns {corpus: {band: {SROCC, PLCC, KROCC, OR, PWRC, Z-RMSE, MAE, n}}}."""
    result: dict[str, dict[str, dict[str, float]]] = {}
    # Find all "## CID22 (n=4292)" headers and their following per-band tables.
    # The per-band header is "### <corpus> 10-band full Mohammadi panel".
    corpus_re = re.compile(r"## ([^(]+?) \(n=\d+\)\n", re.MULTILINE)
    corpus_starts = [(m.start(), m.group(1).strip()) for m in corpus_re.finditer(md)]
    corpus_starts.append((len(md), None))
    for i in range(len(corpus_starts) - 1):
        start, corpus = corpus_starts[i]
        end = corpus_starts[i + 1][0]
        section = md[start:end]
        m = re.search(
            r"10-band full Mohammadi panel.*?\n\|\s*Band.*?\n\|.*?\n((?:\|.*?\n)+?)(?:\n|$)",
            section,
            re.DOTALL,
        )
        if not m:
            continue
        bands: dict[str, dict[str, float]] = {}
        for row in m.group(1).strip().split("\n"):
            cells = [c.strip() for c in row.strip("|").split("|")]
            if len(cells) < 11:
                continue
            band = cells[0].split()[0]  # "B0" from "B0" or "B0 ⚠"
            if cells[3] == "n/a":
                bands[band] = {"n": int(cells[2]) if cells[2].isdigit() else 0,
                               "SROCC": float("nan"), "PLCC": float("nan"),
                               "KROCC": float("nan"), "OR": float("nan"),
                               "PWRC": float("nan"), "Z-RMSE": float("nan"),
                               "MAE": float("nan")}
                continue
            try:
                bands[band] = {
                    "n": int(cells[2]),
                    "SROCC": float(cells[3]),
                    "PLCC": float(cells[4]),
                    "KROCC": float(cells[5]),
                    "OR": float(cells[6]),
                    "PWRC": float(cells[7]),
                    "Z-RMSE": float(cells[8]),
                    "MAE": float(cells[9]),
                }
            except ValueError:
                continue
        if bands:
            result[corpus] = bands
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ship-md", type=Path, required=True)
    ap.add_argument("--candidate-md", type=Path, required=True)
    ap.add_argument("--out-tsv", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args()

    ship = args.ship_md.read_text()
    cand = args.candidate_md.read_text()

    ship_agg = parse_aggregate_table(ship)
    cand_agg = parse_aggregate_table(cand)
    ship_band = parse_per_band(ship)
    cand_band = parse_per_band(cand)

    # TSV
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w") as f:
        f.write("corpus\tband\tn\tstat\tship\tcandidate\tdelta\n")
        # Aggregate
        for corpus in ship_agg:
            if corpus not in cand_agg:
                continue
            for stat in ("SROCC", "PLCC", "KROCC", "OR", "PWRC", "Z-RMSE"):
                s_val = ship_agg[corpus][stat]
                c_val = cand_agg[corpus][stat]
                f.write(f"{corpus}\taggregate\t{ship_agg[corpus]['n']}\t{stat}\t{s_val:.4f}\t{c_val:.4f}\t{c_val - s_val:+.4f}\n")
        # Per-band
        for corpus in ship_band:
            if corpus not in cand_band:
                continue
            for band in sorted(ship_band[corpus].keys()):
                if band not in cand_band[corpus]:
                    continue
                n_band = ship_band[corpus][band]["n"]
                for stat in ("SROCC", "PLCC", "KROCC", "OR", "PWRC", "Z-RMSE", "MAE"):
                    s_val = ship_band[corpus][band][stat]
                    c_val = cand_band[corpus][band][stat]
                    if s_val != s_val or c_val != c_val:  # NaN check
                        continue
                    f.write(f"{corpus}\t{band}\t{n_band}\t{stat}\t{s_val:.4f}\t{c_val:.4f}\t{c_val - s_val:+.4f}\n")

    # Markdown
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Phase 2 diff — v11 retrain (YJ-autotransforms) vs v11 ship")
    lines.append("")
    lines.append("## Per-corpus aggregate Mohammadi panel diff")
    lines.append("")
    lines.append("| Corpus | n | SROCC Δ | PLCC Δ | KROCC Δ | OR Δ | PWRC Δ | Z-RMSE Δ |")
    lines.append("|---|--:|---:|---:|---:|---:|---:|---:|")
    for corpus in sorted(ship_agg.keys()):
        if corpus not in cand_agg:
            continue
        n = ship_agg[corpus]["n"]
        deltas = {s: cand_agg[corpus][s] - ship_agg[corpus][s] for s in ("SROCC", "PLCC", "KROCC", "OR", "PWRC", "Z-RMSE")}
        row = f"| {corpus} | {n} | "
        row += " | ".join(f"{deltas[s]:+.4f}" for s in ("SROCC", "PLCC", "KROCC", "OR", "PWRC", "Z-RMSE"))
        row += " |"
        lines.append(row)
    lines.append("")
    lines.append("Per-corpus raw values (ship → candidate):")
    lines.append("")
    lines.append("| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |")
    lines.append("|---|--:|---:|---:|---:|---:|---:|---:|")
    for corpus in sorted(ship_agg.keys()):
        if corpus not in cand_agg:
            continue
        n = ship_agg[corpus]["n"]
        line = f"| {corpus} (ship) | {n} | "
        line += " | ".join(f"{ship_agg[corpus][s]:.4f}" for s in ("SROCC", "PLCC", "KROCC", "OR", "PWRC", "Z-RMSE"))
        line += " |"
        lines.append(line)
        line = f"| {corpus} (cand) | {n} | "
        line += " | ".join(f"{cand_agg[corpus][s]:.4f}" for s in ("SROCC", "PLCC", "KROCC", "OR", "PWRC", "Z-RMSE"))
        line += " |"
        lines.append(line)
    lines.append("")

    # CID22 10-band table
    for corpus in ("CID22", "KADIK10k", "TID2013"):
        if corpus not in ship_band or corpus not in cand_band:
            continue
        lines.append(f"## {corpus} 10-band SROCC + Z-RMSE diff")
        lines.append("")
        lines.append("| Band | n | SROCC ship | SROCC cand | Δ SROCC | Z-RMSE ship | Z-RMSE cand | Δ Z-RMSE |")
        lines.append("|---|--:|---:|---:|---:|---:|---:|---:|")
        for band in sorted(ship_band[corpus].keys()):
            if band not in cand_band[corpus]:
                continue
            sb = ship_band[corpus][band]
            cb = cand_band[corpus][band]
            if sb["n"] == 0:
                continue
            sroc_s = sb["SROCC"]
            sroc_c = cb["SROCC"]
            zrm_s = sb["Z-RMSE"]
            zrm_c = cb["Z-RMSE"]
            warn = " ⚠" if sb["n"] < 30 else ""
            srow = f"| {band}{warn} | {sb['n']} | "
            if sroc_s != sroc_s:  # NaN
                srow += "n/a | n/a | n/a | "
            else:
                srow += f"{sroc_s:.4f} | {sroc_c:.4f} | {sroc_c - sroc_s:+.4f} | "
            if zrm_s != zrm_s:
                srow += "n/a | n/a | n/a |"
            else:
                srow += f"{zrm_s:.3f} | {zrm_c:.3f} | {zrm_c - zrm_s:+.3f} |"
            lines.append(srow)
        lines.append("")

    args.out_md.write_text("\n".join(lines))
    print(f"wrote {args.out_tsv} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
