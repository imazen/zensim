#!/usr/bin/env python3
"""Post-hoc α-sweep analysis for V_18 + V_22-IW multi-bake (Option C).

T1.4 from `benchmarks/v0_22_iw_seed1_eval_findings_2026-05-16.md`.

The V_22-IW seed=1 standalone bake was falsified on CID22 by the full
Mohammadi panel (SROCC 0.6122 vs ssim2 0.8895) but won TID by +0.112
SROCC. The two bakes are complementary; Option C from the methodology
doc tests whether a heavy-V_18 multi-bake mix captures TID's win
without losing CID22.

Method: load per-pair CSVs from both bakes' separate `dataset_metric_baseline`
runs, join by (dataset, reference, distorted), compute mix_raw =
α × V_18_v04 + (1 − α) × V_22-IW_v04 at α ∈ {0.5, 0.6, 0.7, 0.8, 0.9,
0.95}, then SROCC(mix_raw, human_score) per corpus per α. The mix is
RAW-output linear, matching the runtime path in `apply_mlp_scoring`.

This is the cheapest possible Option C verification — no profile
changes, no rebuild, just post-hoc SROCC at multiple mix weights.
If a viable α exists (e.g., CID22 ≥ 0.88 AND TID ≥ 0.90), commit to
shipping V_22-IW as the PreviewV0_5 secondary via a proper
ProfileParams update.

Usage:
  python3 scripts/v_next/v0_22_iw_option_c_alpha_sweep.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from scipy.stats import spearmanr


def load_per_pair(csv_path: Path) -> dict[tuple[str, str, str], dict]:
    out: dict[tuple[str, str, str], dict] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["dataset"], row["reference"], row["distorted"])
            out[key] = {
                "human_score": float(row["human_score"]),
                "v04_distance": float(row["v04_distance"]),
                "v02_distance": float(row["v02_distance"]),
                "fast_ssim2_score": float(row["fast_ssim2_score"]),
                "butter_3norm": float(row["butter_3norm"]),
                "codec": row.get("codec", ""),
                "version": row.get("version", ""),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--v18",
        type=Path,
        default=Path(
            "/home/lilith/work/zen/zensim/benchmarks/v0_18_ship_eval_per_pair_2026-05-16.csv"
        ),
    )
    ap.add_argument(
        "--v22iw",
        type=Path,
        default=Path(
            "/home/lilith/work/zen/zensim/benchmarks/v0_22_iw_seed1_2026-05-16_eval_per_pair.csv"
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/home/lilith/work/zen/zensim/benchmarks/v0_22_iw_option_c_alpha_sweep_2026-05-16.md"
        ),
    )
    ap.add_argument(
        "--alphas",
        type=str,
        default="0.0,0.3,0.5,0.6,0.7,0.8,0.9,0.95,1.0",
        help="Comma-separated list of α values (V_18 weight in mix).",
    )
    args = ap.parse_args()

    if not args.v18.is_file():
        print(f"ERROR: V_18 per-pair CSV missing: {args.v18}", file=sys.stderr)
        return 2
    if not args.v22iw.is_file():
        print(f"ERROR: V_22-IW per-pair CSV missing: {args.v22iw}", file=sys.stderr)
        return 2

    alphas = [float(a) for a in args.alphas.split(",")]
    print(f"V_18 per-pair: {args.v18}")
    print(f"V_22-IW per-pair: {args.v22iw}")
    print(f"alphas: {alphas}")
    print()

    v18 = load_per_pair(args.v18)
    v22 = load_per_pair(args.v22iw)
    print(f"V_18 rows: {len(v18):,}  |  V_22-IW rows: {len(v22):,}")

    # Inner join by (dataset, reference, distorted)
    joined: dict[str, list[dict]] = {}
    for key, v18_row in v18.items():
        v22_row = v22.get(key)
        if v22_row is None:
            continue
        dataset = key[0]
        merged = {
            "human_score": v18_row["human_score"],
            "v18_raw": v18_row["v04_distance"],
            "v22_raw": v22_row["v04_distance"],
            "v02_raw": v18_row["v02_distance"],
            "fast_ssim2": v18_row["fast_ssim2_score"],
            "butter": v18_row["butter_3norm"],
        }
        joined.setdefault(dataset, []).append(merged)

    n_joined = sum(len(v) for v in joined.values())
    print(f"joined rows: {n_joined:,} across {len(joined)} datasets")
    for ds, rows in sorted(joined.items()):
        print(f"  {ds}: {len(rows):,}")
    print()

    # Compute SROCC per (dataset, α). For each α, mix = α·V_18 + (1−α)·V_22-IW.
    # Higher V_18_raw = higher distance ⇒ lower quality (in legacy ABI), but
    # V_18 ship has `skip_score_mapping=true`, so its raw IS the score:
    # higher raw = higher quality. Same for V_22-IW (trained against iwssim
    # ∈ [0,1] × 100, output is score-shaped). So we treat both as
    # score-shaped and SROCC vs human_score directly — abs() handles the
    # legacy polarity confusion at the absolute level.

    rows_out: list[dict] = []
    for ds in sorted(joined):
        rows = joined[ds]
        humans = [r["human_score"] for r in rows]
        v18_arr = [r["v18_raw"] for r in rows]
        v22_arr = [r["v22_raw"] for r in rows]
        v02_arr = [r["v02_raw"] for r in rows]
        ssim2_arr = [r["fast_ssim2"] for r in rows]
        butter_arr = [r["butter"] for r in rows]
        srocc_v02 = abs(spearmanr(v02_arr, humans).correlation)
        srocc_ssim2 = abs(spearmanr(ssim2_arr, humans).correlation)
        srocc_butter = abs(spearmanr(butter_arr, humans).correlation)
        srocc_v18 = abs(spearmanr(v18_arr, humans).correlation)
        srocc_v22 = abs(spearmanr(v22_arr, humans).correlation)
        for alpha in alphas:
            mix = [alpha * a + (1 - alpha) * b for a, b in zip(v18_arr, v22_arr)]
            srocc_mix = abs(spearmanr(mix, humans).correlation)
            rows_out.append(
                {
                    "dataset": ds,
                    "n": len(rows),
                    "alpha": alpha,
                    "srocc_mix": srocc_mix,
                    "srocc_v18": srocc_v18,
                    "srocc_v22": srocc_v22,
                    "srocc_v02": srocc_v02,
                    "srocc_ssim2": srocc_ssim2,
                    "srocc_butter": srocc_butter,
                }
            )

    # Print + write markdown.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write("# V_22-IW Option C α-sweep — V_18 ship × V_22-IW seed=1 multi-bake (2026-05-16)\n\n")
        f.write("Post-hoc analysis of `α × V_18_raw + (1−α) × V_22-IW_raw` per-pair mix\n")
        f.write("at α ∈ " + ", ".join(f"{a:.2f}" for a in alphas) + ".\n\n")
        f.write("**Inputs**:\n")
        f.write(f"- V_18 ship per-pair: `{args.v18}`\n")
        f.write(f"- V_22-IW seed=1 per-pair: `{args.v22iw}`\n\n")
        f.write(f"**Joined rows**: {n_joined:,} across {len(joined)} corpora.\n\n")
        for ds in sorted(joined):
            ds_rows = [r for r in rows_out if r["dataset"] == ds]
            n = ds_rows[0]["n"]
            v18 = ds_rows[0]["srocc_v18"]
            v22 = ds_rows[0]["srocc_v22"]
            v02 = ds_rows[0]["srocc_v02"]
            ssim2 = ds_rows[0]["srocc_ssim2"]
            butter = ds_rows[0]["srocc_butter"]
            f.write(f"## {ds} (n = {n:,})\n\n")
            f.write(
                f"Baselines (no mix): V_0_2 = {v02:.4f}, V_18 ship = **{v18:.4f}**, "
                f"V_22-IW = **{v22:.4f}**, fast-ssim2 = {ssim2:.4f}, butter = {butter:.4f}.\n\n"
            )
            f.write("| α (V_18 weight) | SROCC mix | Δ vs V_18 ship | Δ vs fast-ssim2 |\n")
            f.write("|---|---:|---:|---:|\n")
            for r in ds_rows:
                d_v18 = r["srocc_mix"] - v18
                d_ssim2 = r["srocc_mix"] - ssim2
                marker = ""
                if r["alpha"] in (0.0, 1.0):
                    marker = " (= V_22-IW alone)" if r["alpha"] == 0.0 else " (= V_18 alone)"
                f.write(
                    f"| {r['alpha']:.2f}{marker} | {r['srocc_mix']:.4f} | "
                    f"{d_v18:+.4f} | {d_ssim2:+.4f} |\n"
                )
            f.write("\n")

        # Pareto summary across datasets
        f.write("## Cross-corpus Pareto picks\n\n")
        f.write("For each α, list the SROCC per corpus to identify the\n")
        f.write("trade-off frontier.\n\n")
        datasets = sorted(joined)
        f.write("| α | " + " | ".join(datasets) + " |\n")
        f.write("|---|" + "|".join(["---:"] * len(datasets)) + "|\n")
        for alpha in alphas:
            cells = [f"{alpha:.2f}"]
            for ds in datasets:
                r = next(
                    r for r in rows_out if r["dataset"] == ds and r["alpha"] == alpha
                )
                cells.append(f"{r['srocc_mix']:.4f}")
            f.write("| " + " | ".join(cells) + " |\n")
        f.write("\n")

        # Decision aid: per α, count of corpora where mix beats fast-ssim2
        f.write("## Decision aid: how many corpora does each α beat fast-ssim2 on?\n\n")
        f.write("| α | wins vs ssim2 | total | corpora won |\n")
        f.write("|---|--:|--:|---|\n")
        for alpha in alphas:
            wins = []
            for ds in datasets:
                r = next(
                    r for r in rows_out if r["dataset"] == ds and r["alpha"] == alpha
                )
                if r["srocc_mix"] > r["srocc_ssim2"]:
                    wins.append(ds)
            f.write(f"| {alpha:.2f} | {len(wins)} | {len(datasets)} | {', '.join(wins) if wins else '—'} |\n")
        f.write("\n")
        f.write("## Decision aid: how many corpora does each α beat V_18 ship on?\n\n")
        f.write("| α | wins vs V_18 | total | corpora won |\n")
        f.write("|---|--:|--:|---|\n")
        for alpha in alphas:
            wins = []
            for ds in datasets:
                r = next(
                    r for r in rows_out if r["dataset"] == ds and r["alpha"] == alpha
                )
                if r["srocc_mix"] > r["srocc_v18"]:
                    wins.append(ds)
            f.write(f"| {alpha:.2f} | {len(wins)} | {len(datasets)} | {', '.join(wins) if wins else '—'} |\n")
        f.write("\n")

    print(f"wrote {args.out}")
    print()
    print("Next step: read the Pareto table + decision aid. If an α ∈ (0, 1)")
    print("beats V_18 ship on >= 2 of CID22/KADID/TID/AIC-3 (the ship-grade")
    print("corpora) and beats fast-ssim2 on the same, commit to shipping the")
    print("multi-bake as PreviewV0_5. Otherwise, escalate to T1.5 (V_22-IW v2")
    print("with target-distribution transform).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
