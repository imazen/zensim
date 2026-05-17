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
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, pearsonr, spearmanr


def _logistic_4p(x: np.ndarray, b1: float, b2: float, b3: float, b4: float) -> np.ndarray:
    """4-parameter logistic per Mohammadi 2025 / VQEG."""
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(b2 * (x - b3)))) + b4 * x + (b1 / 2.0)


def rescale_logistic(predicted: list[float], target: list[float]) -> list[float]:
    """Map predicted into target's MOS scale via 4-parameter logistic.

    Falls back to identity-z if the curve_fit fails. This matches the
    Rust eval harness's `rescale_logistic` in dataset_metric_baseline.rs.
    """
    pred = np.asarray(predicted, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if len(pred) < 8 or pred.std() < 1e-9 or tgt.std() < 1e-9:
        # Identity-z fallback
        return [(p - pred.mean()) / max(pred.std(), 1e-12) * tgt.std() + tgt.mean() for p in pred]
    # Auto-detect polarity: if Pearson(pred, tgt) negative, flip pred sign
    p_corr = float(np.corrcoef(pred, tgt)[0, 1])
    flipped = pred if p_corr >= 0 else -pred
    b1_0 = float(tgt.max() - tgt.min())
    b2_0 = 1.0 / max(flipped.std(), 1e-6)
    b3_0 = float(flipped.mean())
    b4_0 = 0.0
    try:
        popt, _ = curve_fit(
            _logistic_4p, flipped, tgt, p0=[b1_0, b2_0, b3_0, b4_0], maxfev=5000
        )
        rescaled = _logistic_4p(flipped, *popt)
        if np.any(np.isnan(rescaled)) or np.any(np.isinf(rescaled)):
            raise RuntimeError("nan-bearing rescale")
        return rescaled.tolist()
    except Exception:
        # Identity-z fallback
        return [(p - flipped.mean()) / max(flipped.std(), 1e-12) * tgt.std() + tgt.mean() for p in flipped]


def z_rmse(predicted: list[float], target: list[float]) -> float:
    """σ-normalized RMSE — corpus-wide σ (Mohammadi 2025 form)."""
    pred = np.asarray(predicted, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    sd = max(float(tgt.std()), 1e-12)
    return float(np.sqrt(np.mean(((pred - tgt) / sd) ** 2)))


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

    # Drop rows with NaN V_18 OR V_22-IW raw — V_22-IW had 320 NaN
    # KADID outputs (the degenerate high-q regime per
    # `benchmarks/v0_22_iw_seed1_eval_findings_2026-05-16.md`). NaN
    # propagates through spearmanr → all-NaN result, masking real signal.
    # Per CLAUDE.md "NaN-safe sort" precedent: drop, then report n_used.
    import math
    nan_drops: dict[str, int] = {}
    for ds in list(joined):
        kept = [
            r
            for r in joined[ds]
            if not (
                math.isnan(r["v18_raw"]) or math.isnan(r["v22_raw"])
                or math.isnan(r["human_score"])
            )
        ]
        nan_drops[ds] = len(joined[ds]) - len(kept)
        joined[ds] = kept
    if any(c > 0 for c in nan_drops.values()):
        print("NaN-bearing rows dropped per dataset:")
        for ds, n in sorted(nan_drops.items()):
            print(f"  {ds}: {n}")
        print()

    # Compute SROCC per (dataset, α). For each α, mix = α·V_18 + (1−α)·V_22-IW.
    # Higher V_18_raw = higher distance ⇒ lower quality (in legacy ABI), but
    # V_18 ship has `skip_score_mapping=true`, so its raw IS the score:
    # higher raw = higher quality. Same for V_22-IW (trained against iwssim
    # ∈ [0,1] × 100, output is score-shaped). So we treat both as
    # score-shaped and SROCC vs human_score directly — abs() handles the
    # legacy polarity confusion at the absolute level.

    def zscore(vals: list[float]) -> list[float]:
        n = len(vals)
        m = sum(vals) / n
        var = sum((v - m) ** 2 for v in vals) / n
        sd = max(var**0.5, 1e-12)
        return [(v - m) / sd for v in vals]

    def full_panel(pred: list[float], tgt: list[float]) -> dict[str, float]:
        """Compute SROCC + PLCC + KROCC + PWRC-proxy + Z-RMSE (Mohammadi 2025).

        PLCC is computed AFTER 4-parameter logistic rescale (per Mohammadi
        eq. 5), Z-RMSE is corpus-wide σ-normalized RMSE on the same
        rescaled predictions. PWRC is approximated as Pearson on
        rank-transformed values (the Wang-Liu form lives in the Rust
        harness; this script uses a simple corr-of-ranks as proxy).
        """
        srocc = abs(spearmanr(pred, tgt).correlation)
        krocc = abs(kendalltau(pred, tgt).correlation)
        rescaled = rescale_logistic(pred, tgt)
        plcc = abs(pearsonr(rescaled, tgt).correlation)
        zr = z_rmse(rescaled, tgt)
        # PWRC proxy: Pearson on rank-transformed values, weighted by
        # extremeness — Wang & Liu form, simplified.
        rs_p = np.argsort(np.argsort(np.asarray(pred))).astype(float)
        rs_t = np.argsort(np.argsort(np.asarray(tgt))).astype(float)
        mid = (len(pred) - 1) / 2.0
        w = np.abs(rs_t - mid) / max(mid, 1e-12)
        if w.sum() < 1e-12:
            pwrc = 0.0
        else:
            wm_p = float(np.sum(w * rs_p) / w.sum())
            wm_t = float(np.sum(w * rs_t) / w.sum())
            num = float(np.sum(w * (rs_p - wm_p) * (rs_t - wm_t)))
            d1 = float(np.sum(w * (rs_p - wm_p) ** 2))
            d2 = float(np.sum(w * (rs_t - wm_t) ** 2))
            pwrc = abs(num / max((d1 * d2) ** 0.5, 1e-12))
        return {"srocc": srocc, "plcc": plcc, "krocc": krocc, "pwrc": pwrc, "zrmse": zr}

    rows_out: list[dict] = []
    for ds in sorted(joined):
        rows = joined[ds]
        humans = [r["human_score"] for r in rows]
        v18_arr = [r["v18_raw"] for r in rows]
        v22_arr = [r["v22_raw"] for r in rows]
        v02_arr = [r["v02_raw"] for r in rows]
        ssim2_arr = [r["fast_ssim2"] for r in rows]
        butter_arr = [r["butter"] for r in rows]
        baseline_v02 = full_panel(v02_arr, humans)
        baseline_ssim2 = full_panel(ssim2_arr, humans)
        baseline_butter = full_panel(butter_arr, humans)
        baseline_v18 = full_panel(v18_arr, humans)
        baseline_v22 = full_panel(v22_arr, humans)
        # Per-bake z-normalization before mix — the offline `ensemble_mix`
        # tool's approach (CLAUDE.md V_20 learnings § "Multi-bake runtime").
        v18_z = zscore(v18_arr)
        v22_z = zscore(v22_arr)
        for alpha in alphas:
            mix_raw = [alpha * a + (1 - alpha) * b for a, b in zip(v18_arr, v22_arr)]
            mix_z = [alpha * a + (1 - alpha) * b for a, b in zip(v18_z, v22_z)]
            panel_raw = full_panel(mix_raw, humans)
            panel_z = full_panel(mix_z, humans)
            rows_out.append(
                {
                    "dataset": ds,
                    "n": len(rows),
                    "alpha": alpha,
                    "raw": panel_raw,
                    "z": panel_z,
                    "v18": baseline_v18,
                    "v22": baseline_v22,
                    "v02": baseline_v02,
                    "ssim2": baseline_ssim2,
                    "butter": baseline_butter,
                }
            )

    # Print + write markdown.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write("# V_22-IW Option C α-sweep — V_18 ship × V_22-IW seed=1 multi-bake (2026-05-16)\n\n")
        f.write("Post-hoc full Mohammadi panel sweep of `α × V_18_raw + (1−α) × V_22-IW_raw`\n")
        f.write("per-pair mix at α ∈ " + ", ".join(f"{a:.2f}" for a in alphas) + ".\n\n")
        f.write("Per CLAUDE.md `SROCC-only verdicts BANNED` (2026-05-15) the verdict\n")
        f.write("uses the full panel — SROCC + PLCC + KROCC + PWRC + Z-RMSE — at each\n")
        f.write("(corpus, α). PLCC + Z-RMSE are computed after a 4-parameter logistic\n")
        f.write("rescale (Mohammadi 2025 eq. 5) so they're calibration-aware. PWRC is a\n")
        f.write("Wang-Liu-form proxy: weighted Pearson on rank-transformed values.\n\n")
        f.write("**Inputs**:\n")
        f.write(f"- V_18 ship per-pair: `{args.v18}`\n")
        f.write(f"- V_22-IW seed=1 per-pair: `{args.v22iw}`\n\n")
        f.write(f"**Joined rows**: {n_joined:,} across {len(joined)} corpora.\n\n")
        for ds in sorted(joined):
            ds_rows = [r for r in rows_out if r["dataset"] == ds]
            n = ds_rows[0]["n"]
            v18 = ds_rows[0]["v18"]
            v22 = ds_rows[0]["v22"]
            ssim2 = ds_rows[0]["ssim2"]
            f.write(f"## {ds} (n = {n:,})\n\n")
            f.write("Baselines (no mix), full Mohammadi panel:\n\n")
            f.write("| Metric | SROCC | PLCC | KROCC | PWRC | Z-RMSE |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for lbl, b in [
                ("V_18 ship", v18),
                ("V_22-IW", v22),
                ("V_0_2", ds_rows[0]["v02"]),
                ("fast-ssim2", ssim2),
                ("butter", ds_rows[0]["butter"]),
            ]:
                f.write(
                    f"| {lbl} | {b['srocc']:.4f} | {b['plcc']:.4f} | "
                    f"{b['krocc']:.4f} | {b['pwrc']:.4f} | {b['zrmse']:.3f} |\n"
                )
            f.write("\n")
            for mix_label, mix_key in (("RAW-mix", "raw"), ("Z-mix", "z")):
                f.write(f"### α sweep — {mix_label} ({ds})\n\n")
                f.write(
                    "| α | SROCC | PLCC | KROCC | PWRC | Z-RMSE | Δ SROCC vs V_18 | Δ Z-RMSE vs V_18 |\n"
                )
                f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
                for r in ds_rows:
                    panel = r[mix_key]
                    d_srocc = panel["srocc"] - v18["srocc"]
                    # For Z-RMSE: LOWER is better, so Δ is panel - v18 (positive = worse)
                    d_zr = panel["zrmse"] - v18["zrmse"]
                    marker = ""
                    if r["alpha"] in (0.0, 1.0):
                        marker = " (= V_22-IW alone)" if r["alpha"] == 0.0 else " (= V_18 alone)"
                    f.write(
                        f"| {r['alpha']:.2f}{marker} | {panel['srocc']:.4f} | "
                        f"{panel['plcc']:.4f} | {panel['krocc']:.4f} | "
                        f"{panel['pwrc']:.4f} | {panel['zrmse']:.3f} | "
                        f"{d_srocc:+.4f} | {d_zr:+.3f} |\n"
                    )
                f.write("\n")

        # Cross-corpus picks per stat — Z-RMSE is the load-bearing stat
        # for V_22-IW because the failure mode is calibration-saturation,
        # which SROCC misses but Z-RMSE catches.
        datasets = sorted(joined)
        for stat_label, stat_key, lower_is_better in (
            ("SROCC", "srocc", False),
            ("Z-RMSE (lower = better)", "zrmse", True),
            ("PLCC", "plcc", False),
            ("PWRC", "pwrc", False),
        ):
            for mix_label, mix_key in (("RAW", "raw"), ("Z-NORM", "z")):
                f.write(f"## Cross-corpus {stat_label} ({mix_label} mix)\n\n")
                f.write("| α | " + " | ".join(datasets) + " |\n")
                f.write("|---|" + "|".join(["---:"] * len(datasets)) + "|\n")
                for alpha in alphas:
                    cells = [f"{alpha:.2f}"]
                    for ds in datasets:
                        r = next(
                            r for r in rows_out if r["dataset"] == ds and r["alpha"] == alpha
                        )
                        cells.append(f"{r[mix_key][stat_key]:.4f}")
                    f.write("| " + " | ".join(cells) + " |\n")
                f.write("\n")

        # CLAUDE.md ship gate: ≥3 of 5 stats agree that mix beats V_18.
        f.write("## Multi-stat ship gate per CLAUDE.md (≥3 of 5 stats beat V_18)\n\n")
        f.write("For each (mix, α), count of stats (SROCC, PLCC, KROCC, PWRC, Z-RMSE)\n")
        f.write("where the mix beats V_18 ship on each corpus. Mix-vs-V_18 win:\n")
        f.write("higher SROCC/PLCC/KROCC/PWRC, LOWER Z-RMSE.\n\n")
        for mix_label, mix_key in (("RAW", "raw"), ("Z-NORM", "z")):
            f.write(f"### {mix_label} mix\n\n")
            f.write("| α | " + " | ".join(f"{ds} (stats won)" for ds in datasets) + " | total ship-grade |\n")
            f.write("|---|" + "|".join(["---"] * (len(datasets) + 1)) + "|\n")
            for alpha in alphas:
                cells = [f"{alpha:.2f}"]
                ship_grade_count = 0
                for ds in datasets:
                    r = next(r for r in rows_out if r["dataset"] == ds and r["alpha"] == alpha)
                    panel = r[mix_key]
                    base = r["v18"]
                    won = 0
                    for stat in ("srocc", "plcc", "krocc", "pwrc"):
                        if panel[stat] > base[stat]:
                            won += 1
                    if panel["zrmse"] < base["zrmse"]:
                        won += 1
                    cells.append(f"{won}/5")
                    if won >= 3:
                        ship_grade_count += 1
                cells.append(f"**{ship_grade_count}/{len(datasets)}**" if ship_grade_count > 0 else "—")
                f.write("| " + " | ".join(cells) + " |\n")
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
