#!/usr/bin/env python3
"""EX-MIX3 Pareto verdict: compare mean-of-5-seed per variant against
the V_22 noLARGE baseline using bake_verdict output.

Pareto-win definition (per task: "≥4-of-6 panel agreement on improvement,
no decisive regression on any corpus"):

For each variant, for each corpus, count panel stats (SROCC, PLCC, KROCC,
PWRC, Z-RMSE — 5 stats; OR optional) where the variant's mean beats the
baseline (lower is better for Z-RMSE/OR, higher for the rest).

Ship candidate if:
  - For ≥4-of-6 corpora the variant wins on ≥3-of-5 stats
  - No corpus shows decisive regression (mean Δ-SROCC < -0.01 AND
    mean Δ-PWRC < -0.005 AND mean Δ-Z-RMSE > +0.020)

Decisive regression triggers Pareto FAIL.

Writes /mnt/v/zen/zensim-eval/ex_mix3_2026-05-18/PARETO_VERDICT.md
"""

from __future__ import annotations
import re
import sys
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

OUT_PATH = Path("/mnt/v/zen/zensim-eval/ex_mix3_2026-05-18/PARETO_VERDICT.md")
VERDICT_DIR = Path("/mnt/v/zen/zensim-eval/ex_mix3_2026-05-18/verdicts")
BASELINE_DIR = Path("/mnt/v/zen/zensim-eval/v22_372feat_2026-05-18")

VARIANTS = ["cv33_iw33_sm33", "cv30_iw40_sm30", "cv40_iw40_sm20"]
SEEDS = [1, 2, 3, 4, 5]
CORPORA = ["CID22", "KADIK10k", "TID2013", "KonJND-1k (full)", "AIC-3 CTC"]

# Higher = better
HIGHER_BETTER = {"srocc", "plcc", "krocc", "pwrc"}
# Lower = better
LOWER_BETTER = {"or", "zrmse"}


def parse_verdict(path: Path) -> dict[str, dict[str, float]]:
    """Parse a bake_verdict.md and extract Summary table per-corpus stats."""
    text = path.read_text()
    out = {}
    pattern = re.compile(
        r"^\|\s*([A-Za-z0-9\-_\s]+(?:\s*\([^)]+\))?)\s*\|\s*(\d+)\s*\|\s*([\d.\-]+)\s*\|\s*([\d.\-]+)\s*\|\s*([\d.\-]+)\s*\|\s*([\d.\-]+)\s*\|\s*([\d.\-]+)\s*\|\s*([\d.\-]+)\s*\|\s*$",
        re.MULTILINE,
    )
    for m in pattern.finditer(text):
        corpus = m.group(1).strip()
        if corpus not in CORPORA:
            continue
        out[corpus] = {
            "n": int(m.group(2)),
            "srocc": float(m.group(3)),
            "plcc": float(m.group(4)),
            "krocc": float(m.group(5)),
            "or": float(m.group(6)),
            "pwrc": float(m.group(7)),
            "zrmse": float(m.group(8)),
        }
    return out


def get_baseline_means() -> dict[str, dict[str, float]]:
    """Compute mean-of-5-seed for V_22 noLARGE 372feat baseline."""
    # We need bake_verdict outputs for the 5 baseline seeds. The matrix
    # may have only s3 verdicted by default. Lazily run bake_verdict on
    # baseline seeds if needed.
    out_dir = Path("/mnt/v/zen/zensim-eval/ex_mix3_2026-05-18/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    import subprocess
    seeds_data = []
    for s in [1, 2, 3, 4, 5]:
        bake = BASELINE_DIR / f"v22_372feat_noLARGE_s{s}_h128.bin"
        out = out_dir / f"v22_noLARGE_s{s}.md"
        if not out.exists() and bake.exists():
            subprocess.run([
                "/home/lilith/work/zen/zensim/target/release/bake_verdict",
                "--bake", str(bake),
                "--corpora", "cid22,kadid,tid,konjnd,aic3",
                "--output", str(out),
            ], check=True)
        if out.exists():
            seeds_data.append(parse_verdict(out))
    # Aggregate mean per corpus per stat
    agg = {}
    for corpus in CORPORA:
        agg[corpus] = {}
        for stat in ["srocc", "plcc", "krocc", "or", "pwrc", "zrmse"]:
            vals = [sd[corpus][stat] for sd in seeds_data if corpus in sd]
            if vals:
                agg[corpus][stat] = mean(vals)
                agg[corpus][f"{stat}_std"] = stdev(vals) if len(vals) > 1 else 0.0
    return agg


def get_variant_means(variant: str) -> tuple[dict, int]:
    """Compute mean per variant. Returns (agg, n_seeds)."""
    seeds_data = []
    for s in SEEDS:
        p = VERDICT_DIR / f"exmix3_{variant}_s{s}.md"
        if p.exists():
            try:
                seeds_data.append(parse_verdict(p))
            except Exception as e:
                print(f"  parse fail {p}: {e}", file=sys.stderr)
    if not seeds_data:
        return {}, 0
    agg = {}
    for corpus in CORPORA:
        agg[corpus] = {}
        for stat in ["srocc", "plcc", "krocc", "or", "pwrc", "zrmse"]:
            vals = [sd[corpus][stat] for sd in seeds_data if corpus in sd]
            if vals:
                agg[corpus][stat] = mean(vals)
                agg[corpus][f"{stat}_std"] = stdev(vals) if len(vals) > 1 else 0.0
    return agg, len(seeds_data)


def wins(variant_val: float, baseline_val: float, stat: str) -> bool:
    if stat in HIGHER_BETTER:
        return variant_val > baseline_val
    if stat in LOWER_BETTER:
        return variant_val < baseline_val
    return False


def decisive_regression(variant_agg: dict, baseline_agg: dict, corpus: str) -> tuple[bool, str]:
    """Return (is_decisive_regression, reason)."""
    if corpus not in variant_agg or corpus not in baseline_agg:
        return False, "missing data"
    v = variant_agg[corpus]
    b = baseline_agg[corpus]
    d_srocc = v.get("srocc", 0) - b.get("srocc", 0)
    d_pwrc = v.get("pwrc", 0) - b.get("pwrc", 0)
    d_zrmse = v.get("zrmse", 0) - b.get("zrmse", 0)
    if d_srocc < -0.01 and d_pwrc < -0.005 and d_zrmse > 0.020:
        return True, f"Δsrocc={d_srocc:+.4f}, Δpwrc={d_pwrc:+.4f}, Δzrmse={d_zrmse:+.4f}"
    return False, ""


def main():
    print("Building Pareto verdict...")
    baseline_agg = get_baseline_means()
    if not baseline_agg or not any(baseline_agg.values()):
        print("FAIL: no baseline data — bake_verdict on v22_noLARGE_s* failed?")
        return 1

    out = []
    out.append("# EX-MIX3 Pareto verdict (§ A.9 decisive rule)")
    out.append("")
    out.append("Baseline: V_22 noLARGE 372feat 5-seed mean.")
    out.append("Variant: EX-MIX3 5-seed mean per target column.")
    out.append("")
    out.append("Wins counted: Higher-is-better (SROCC/PLCC/KROCC/PWRC), Lower-is-better (OR/Z-RMSE).")
    out.append("")

    # Baseline summary
    out.append("## Baseline (V_22 noLARGE 372feat 5-seed mean)")
    out.append("")
    out.append("| Corpus | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for c in CORPORA:
        if c not in baseline_agg or not baseline_agg[c]:
            out.append(f"| {c} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        b = baseline_agg[c]
        out.append(f"| {c} | {b.get('srocc', float('nan')):.4f} | {b.get('plcc', float('nan')):.4f} | "
                   f"{b.get('krocc', float('nan')):.4f} | {b.get('or', float('nan')):.4f} | "
                   f"{b.get('pwrc', float('nan')):.4f} | {b.get('zrmse', float('nan')):.4f} |")
    out.append("")

    # Per-variant Δ
    pareto_winners = []
    for variant in VARIANTS:
        v_agg, n_seeds = get_variant_means(variant)
        out.append(f"## EX-MIX3 / {variant} ({n_seeds}/5 seeds)")
        out.append("")
        if not v_agg or n_seeds == 0:
            out.append("**No verdicts yet — train still running or all failed.**")
            out.append("")
            continue
        out.append("| Corpus | ΔSROCC | ΔPLCC | ΔKROCC | ΔOR | ΔPWRC | ΔZ-RMSE | wins/5 |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        any_decisive_regression = False
        wins_per_corpus = {}
        for c in CORPORA:
            if c not in v_agg or not v_agg[c] or c not in baseline_agg:
                out.append(f"| {c} | n/a |")
                continue
            v = v_agg[c]
            b = baseline_agg[c]
            d_srocc = v.get("srocc", 0) - b.get("srocc", 0)
            d_plcc = v.get("plcc", 0) - b.get("plcc", 0)
            d_krocc = v.get("krocc", 0) - b.get("krocc", 0)
            d_or = v.get("or", 0) - b.get("or", 0)
            d_pwrc = v.get("pwrc", 0) - b.get("pwrc", 0)
            d_zrmse = v.get("zrmse", 0) - b.get("zrmse", 0)
            w = 0
            if d_srocc > 0: w += 1
            if d_plcc > 0: w += 1
            if d_krocc > 0: w += 1
            if d_or < 0: w += 1
            if d_pwrc > 0: w += 1
            # Z-RMSE: lower is better -> negative Δ is win
            if d_zrmse < 0: w += 1
            wins_per_corpus[c] = w
            dec, reason = decisive_regression(v_agg, baseline_agg, c)
            if dec:
                any_decisive_regression = True
            tag = " 🚨 decisive regression" if dec else ""
            out.append(f"| {c} | {d_srocc:+.4f} | {d_plcc:+.4f} | {d_krocc:+.4f} | {d_or:+.4f} | "
                       f"{d_pwrc:+.4f} | {d_zrmse:+.4f} | {w}/6{tag} |")
        out.append("")
        # Aggregate: count corpora where variant wins on majority of stats
        corpora_majority_win = sum(1 for c, w in wins_per_corpus.items() if w >= 4)
        if any_decisive_regression:
            verdict = "**PARETO FAIL** (decisive regression on at least one corpus)"
        elif corpora_majority_win >= 4:
            verdict = f"**PARETO WIN** ({corpora_majority_win}/5 corpora wins ≥4 of 6 stats)"
            pareto_winners.append((variant, corpora_majority_win))
        elif corpora_majority_win >= 3:
            verdict = f"PARETO MIXED ({corpora_majority_win}/5 corpora wins ≥4 of 6 stats)"
        else:
            verdict = f"**FALSIFIED** ({corpora_majority_win}/5 corpora wins ≥4 of 6 stats)"
        out.append(f"### Verdict: {verdict}")
        out.append("")

    # Overall
    out.append("## Overall verdict")
    out.append("")
    if pareto_winners:
        # Pick the strongest by corpora_majority_win
        pareto_winners.sort(key=lambda p: -p[1])
        top = pareto_winners[0]
        out.append(f"Ship candidate: **{top[0]}** ({top[1]}/5 corpora wins ≥4 of 6 stats)")
        out.append("")
        out.append(f"Next step: `zenpredict repack` packed bake from best-of-5-seeds.")
    else:
        out.append("**No Pareto winner.** All variants either failed (<4 corpora majority) or hit a decisive regression.")
        out.append("")
        out.append("Root-cause candidates:")
        out.append("- Coverage shrink (EX-MIX3 dropped LARGE + konjnd; smaller training corpus)")
        out.append("- 3-way blend itself doesn't improve over 2-way cv+iw")
        out.append("- ssim2 contribution dilutes iwssim signal (V_22's chosen direction)")

    OUT_PATH.write_text("\n".join(out))
    print(f"WROTE: {OUT_PATH}")
    print()
    print("\n".join(out))


if __name__ == "__main__":
    sys.exit(main())
