#!/usr/bin/env python3
"""UPIQ HDR/SDR validation harness for zensim.

Joins the UPIQ subjective JOD scores with objective metric scores, then runs
the canonical `panel` binary (zensim_validate::panel — the ONLY place IQA
statistics are computed; this script does NOT reimplement any stat math) to
produce the full Mohammadi 2025 panel (SROCC/PLCC/KROCC/OR/PWRC/Z-RMSE),
stratified ALL / HDR / SDR.

Two uses:
  1. Reproduce the published UPIQ baselines (PU-SSIM / PU-FSIM / HDR-VDP-2 /
     PU-PieAPP ...) from `upiq_objective_scores.csv` — confirms the harness and
     establishes the bar zensim-HDR must clear.
  2. Evaluate a zensim-HDR score column (once chunk 2 lands): pass
     `--scores <csv>` with columns `condition_id` and a score column; it is
     joined and run through the same panel alongside the baselines.

Usage:
  upiq_eval.py [--upiq-dir /mnt/v/datasets/upiq] [--panel ./target/release/panel]
               [--scores zensim_hdr.csv --score-col zensim_hdr]
               [--out benchmarks/upiq_baselines_<date>.md]

The UPIQ EXR images are NOT needed for this harness — only the two CSVs, which
ship in the score-only UPIQ distribution. See docs/HDR_PLAN.md §4-5.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import tempfile

# Baseline metric columns in upiq_objective_scores.csv worth reporting, with
# the polarity (all are higher=better against JOD except PSNR-likes, which are
# also higher=better, so every column here is higher=better) and the published
# full-set SROCC-to-JOD from the UPIQ paper / our 2026-05-27 reproduction
# (docs/iqa-methods/evaluation-statistics.md §6). Used only to flag drift.
BASELINE_METRICS = [
    "PU_PieApp",
    "PU_FSIM",
    "HDRVDP2_2",
    "PU_SSIM",
    "PU_PSNR",
    "HDRVQM",
    "FSIM",
    "PSNR",
]
PUBLISHED_SROCC = {  # full-set (ALL), for drift detection only
    "PU_PieApp": 0.945,
    "PU_FSIM": 0.841,
    "HDRVDP2_2": 0.815,
    "PU_SSIM": 0.696,
    "PU_PSNR": 0.660,
}


def load_subjective(path: str) -> dict[str, tuple[float, int]]:
    """condition_id -> (JOD, is_hdr)."""
    out: dict[str, tuple[float, int]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[row["condition_id"]] = (float(row["JOD"]), int(row["is_hdr"]))
            except (KeyError, ValueError):
                continue
    return out


def load_objective(path: str) -> tuple[list[str], dict[str, dict[str, float]]]:
    """condition_id -> {metric: value}; returns (metric_names, table)."""
    table: dict[str, dict[str, float]] = {}
    metrics: list[str] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        metrics = [c for c in (reader.fieldnames or []) if c not in ("condition_id", "dataset")]
        for row in reader:
            cid = row["condition_id"]
            vals = {}
            for m in metrics:
                try:
                    vals[m] = float(row[m])
                except (KeyError, ValueError):
                    pass
            table[cid] = vals
    return metrics, table


def load_extra_scores(path: str, col: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[row["condition_id"]] = float(row[col])
            except (KeyError, ValueError):
                continue
    return out


def run_panel(panel_bin: str, pred: list[float], target: list[float],
              band: list[str]) -> dict[str, dict]:
    """Write a TSV and run `panel --json`; return {label: group_dict}."""
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as tf:
        tf.write("predicted\ttarget\tband\n")
        for p, t, b in zip(pred, target, band):
            tf.write(f"{p}\t{t}\t{b}\n")
        tsv = tf.name
    try:
        res = subprocess.run(
            [panel_bin, "--input", tsv, "--json"],
            capture_output=True, text=True, check=True,
        )
    finally:
        os.unlink(tsv)
    groups = json.loads(res.stdout)["groups"]
    return {g["label"]: g for g in groups}


def eval_metric(panel_bin, name, getter, subj, cids) -> dict[str, dict]:
    pred, target, band = [], [], []
    for cid in cids:
        v = getter(cid)
        if v is None:
            continue
        jod, is_hdr = subj[cid]
        pred.append(v)
        target.append(jod)
        band.append("hdr" if is_hdr else "sdr")
    if len(pred) < 10:
        return {}
    return run_panel(panel_bin, pred, target, band)


def fmt(x):
    return "—" if x is None else f"{x:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upiq-dir", default="/mnt/v/datasets/upiq")
    ap.add_argument("--panel", default="./target/release/panel")
    ap.add_argument("--scores", help="extra CSV with condition_id + a score column")
    ap.add_argument("--score-col", help="name of the score column in --scores")
    ap.add_argument("--out", help="markdown report path (default: stdout)")
    args = ap.parse_args()

    subj = load_subjective(os.path.join(args.upiq_dir, "upiq_subjective_scores.csv"))
    _, obj = load_objective(os.path.join(args.upiq_dir, "upiq_objective_scores.csv"))
    cids = [c for c in subj if c in obj]
    n_hdr = sum(1 for c in cids if subj[c][1])
    n_sdr = len(cids) - n_hdr

    rows = []  # (metric, results_by_label)
    for m in BASELINE_METRICS:
        r = eval_metric(args.panel, m, lambda c, m=m: obj[c].get(m), subj, cids)
        if r:
            rows.append((m, r, PUBLISHED_SROCC.get(m)))

    if args.scores and args.score_col:
        extra = load_extra_scores(args.scores, args.score_col)
        scids = [c for c in subj if c in extra]
        r = eval_metric(args.panel, args.score_col,
                        lambda c: extra.get(c), subj, scids)
        if r:
            rows.append((args.score_col, r, None))

    date = datetime.date.today().isoformat()
    lines = []
    lines.append(f"# UPIQ validation panel — {date}")
    lines.append("")
    lines.append(f"n = {len(cids)} conditions joined ({n_hdr} HDR, {n_sdr} SDR). "
                 "Stats via `zensim_validate::panel` (the `panel` binary); "
                 "no stat math reimplemented. JOD truth = `upiq_subjective_scores.csv`.")
    lines.append("")
    lines.append("SROCC (rank) per stratum; PLCC after 4-param logistic; "
                 "PWRC + Z-RMSE from the full panel. `Δpub` = ALL-SROCC minus "
                 "published full-set baseline (drift check).")
    lines.append("")
    lines.append("| metric | SROCC all | SROCC HDR | SROCC SDR | PLCC all | PWRC all | Z-RMSE all | Δpub |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for m, r, pub in rows:
        a = r.get("ALL", {})
        h = r.get("band=hdr", {})
        s = r.get("band=sdr", {})
        drift = None if pub is None else (a.get("srocc", float("nan")) - pub)
        lines.append(
            f"| {m} | {fmt(a.get('srocc'))} | {fmt(h.get('srocc'))} | "
            f"{fmt(s.get('srocc'))} | {fmt(a.get('plcc'))} | {fmt(a.get('pwrc'))} | "
            f"{fmt(a.get('z_rmse'))} | {fmt(drift)} |"
        )
    lines.append("")
    lines.append("**Bar for zensim-HDR:** clear PU-SSIM / HDR-VDP-2 decisively, "
                 "approach PU-PieAPP (SROCC 0.945). Watch the HDR column — the "
                 "highlight band is where SDR-tuned metrics collapse.")
    report = "\n".join(lines) + "\n"

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(report)
        print(f"wrote {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
