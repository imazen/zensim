#!/usr/bin/env python3
"""Per-band low-sample-size analysis.

For each (corpus, band) with n < 100, extracts the SROCC + 95% CI for
every bake we evaluated. Computes:

1. **Sampling ceiling**: theoretical max SROCC at this n given the
   observed sample variance. Derived from the 95% CI's upper bound,
   capped at 1.0.
2. **Inter-observer floor**: from the CID22 paper, inter-observer
   SROCC ≈ 0.93 on the population. At small n, a predictor may
   exceed this by chance.
3. **Best bake**: highest point-estimate SROCC for each band.
4. **Least bad across all low-n bands**: which bake has the highest
   sum of SROCC across low-sample bands.

## Usage

  python3 scripts/v_next/v0_20_low_n_band_analysis.py \\
    --out benchmarks/v0_20_low_n_band_analysis_2026-05-15.md

## Output

Markdown doc with per-band tables + a "least bad" verdict.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

CORPORA = ("KADIK10k", "TID2013", "CID22")

# (bake_label, eval_log_path)
BAKES = [
    ("V_18 ship",                   "benchmarks/v0_18_ship_reference_card_2026-05-14.log"),
    ("V_18 base seed=1",            "benchmarks/v0_18_base_seed1_eval_2026-05-15.log"),
    ("V_20 IS (98)",                "benchmarks/v0_20_input_shaping_eval_2026-05-15.log"),
    ("V_20b manifold",              "benchmarks/v0_20b_seed1_eval_2026-05-15.log"),
    ("D1 3-way concat",             "benchmarks/v0_20_d1_concat_eval_2026-05-15.log"),
    ("D3 lift>=0.10",               "benchmarks/v0_20_input_shaping_lift10_eval_2026-05-15.log"),
    ("V_20_4 multi-bake α=0.4",     "benchmarks/v0_20_4_runtime_eval_2026-05-15.log"),
]

LOW_N_THRESHOLD = 100


def parse_10band(log_path: Path, corpus: str):
    """Returns list of dicts with band/range/n/srocc/ci_lo/ci_hi/ssim2_srocc."""
    if not log_path.exists():
        return None
    text = log_path.read_text()
    # Find the section "### <corpus> 10-band SROCC"
    pat = rf"### {re.escape(corpus)} 10-band SROCC.*?\n(.*?)\n###"
    m = re.search(pat, text, re.DOTALL)
    if not m:
        return None
    block = m.group(1)
    out = []
    # Format: | B0 | [0.00, 0.10) | n | v02 | v04 | ci | ssim2 | ...
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|") or "Band" in line or "---" in line:
            continue
        parts = [c.strip() for c in line.strip("|").split("|")]
        if len(parts) < 7:
            continue
        band = parts[0]
        rng = parts[1]
        try:
            n = int(parts[2])
        except ValueError:
            continue
        # Parse "0.1234" or "n/a"
        def f(s):
            try:
                return float(s)
            except (ValueError, TypeError):
                return None
        srocc = f(parts[4])
        # CI is parts[5] in the format "[0.01, 0.34]"
        ci_lo = ci_hi = None
        if "[" in parts[5]:
            cm = re.match(r"\[\s*([\-\d.]+)\s*,\s*([\-\d.]+)\s*\]", parts[5])
            if cm:
                ci_lo = float(cm.group(1))
                ci_hi = float(cm.group(2))
        ssim2_srocc = f(parts[6])
        out.append({
            "band": band, "range": rng, "n": n, "srocc": srocc,
            "ci_lo": ci_lo, "ci_hi": ci_hi, "ssim2_srocc": ssim2_srocc,
        })
    return out


def sampling_se(r, n):
    """Spearman SROCC standard error approximation: SE(r) = (1-r²)/√(n-1)."""
    if n < 2 or r is None:
        return None
    return (1.0 - r * r) / math.sqrt(n - 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repo-root", default=".", type=Path)
    args = ap.parse_args()

    # Build {(corpus, band): {bake_label: row_dict}}
    band_table: dict[tuple[str, str, int, str], dict[str, dict]] = {}
    for label, log_rel in BAKES:
        log_path = args.repo_root / log_rel
        for corpus in CORPORA:
            rows = parse_10band(log_path, corpus)
            if not rows:
                continue
            for r in rows:
                if r["srocc"] is None:
                    continue
                key = (corpus, r["band"], r["n"], r["range"])
                band_table.setdefault(key, {})[label] = r

    # Filter to low-n bands
    low_n = sorted(
        [k for k in band_table.keys() if k[2] > 0 and k[2] < LOW_N_THRESHOLD],
        key=lambda k: (CORPORA.index(k[0]), k[1]),
    )

    out_lines = ["# Low-sample-size band ceiling analysis (2026-05-15)"]
    out_lines.append("")
    out_lines.append(
        f"All (corpus, band) cells with **n < {LOW_N_THRESHOLD}** across "
        "KADID / TID / CID22 evaluated against every V_X bake."
    )
    out_lines.append("")
    out_lines.append("## What 'ceiling' means here")
    out_lines.append("")
    out_lines.append(
        "Two ceilings are relevant at low n:"
    )
    out_lines.append("")
    out_lines.append(
        "1. **Sample-size ceiling** — the standard error of Spearman r is"
        " `SE(r) ≈ (1 − r²) / √(n − 1)`. The 95% CI is roughly `r ± 1.96·SE`."
        " The CI upper bound is the **highest plausible SROCC at this n**"
        " given what we observed — anything above is sampling noise."
    )
    out_lines.append("")
    out_lines.append(
        "2. **Inter-observer ceiling** — bounded by human-MOS reliability."
        " The CID22 paper (Sneyers / Ben Baruch / Vaxman 2023) reports"
        " inter-observer SROCC ≈ 0.93 on CID22 — no metric can exceed that"
        " on the population, but small-n samples can luck into higher r."
    )
    out_lines.append("")
    out_lines.append(
        "The per-band CI in our eval logs comes from a percentile bootstrap"
        " (BCa or quantile). Cells with n < 30 are marked ⚠ — CI widths"
        " exceed 0.3 SROCC and rankings between bakes are not statistically"
        " distinguishable."
    )
    out_lines.append("")

    # Per-band tables
    out_lines.append("## Per-band SROCC + CI across bakes")
    out_lines.append("")
    for key in low_n:
        corpus, band, n, rng = key
        bakes_data = band_table[key]
        out_lines.append(f"### {corpus} {band} {rng}, n = {n}")
        out_lines.append("")
        # Determine sample ceiling: the highest CI upper bound across all bakes
        ceiling_per_bake = []
        for label, row in bakes_data.items():
            ci_hi = row.get("ci_hi")
            srocc = row.get("srocc")
            if ci_hi is not None:
                ceiling_per_bake.append(ci_hi)
            if srocc is not None and ci_hi is None:
                # Estimate SE
                se = sampling_se(srocc, n)
                if se:
                    ceiling_per_bake.append(min(1.0, srocc + 1.96 * se))
        emp_ceiling = max(ceiling_per_bake) if ceiling_per_bake else None
        # Also static ssim2 baseline
        ssim2_srocc = None
        for row in bakes_data.values():
            if row.get("ssim2_srocc") is not None:
                ssim2_srocc = row["ssim2_srocc"]
                break

        out_lines.append(
            "| Bake | SROCC | 95% CI | SE est. |"
        )
        out_lines.append("|---|---:|---|---:|")
        if ssim2_srocc is not None:
            se = sampling_se(ssim2_srocc, n)
            se_str = f"{se:.3f}" if se else "—"
            out_lines.append(
                f"| _fast-ssim2 (static)_ | {ssim2_srocc:.4f} | — | {se_str} |"
            )
        for label, row in bakes_data.items():
            srocc = row["srocc"]
            ci = ""
            if row.get("ci_lo") is not None and row.get("ci_hi") is not None:
                ci = f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"
            else:
                ci = "—"
            se = sampling_se(srocc, n)
            se_str = f"{se:.3f}" if se else "—"
            out_lines.append(
                f"| {label} | {srocc:.4f} | {ci} | {se_str} |"
            )
        if emp_ceiling is not None:
            out_lines.append("")
            out_lines.append(
                f"**Empirical sample ceiling** (max CI upper bound across bakes): {emp_ceiling:.3f}"
            )
            if n < 30:
                out_lines.append(
                    f"⚠ n < 30 — CI width is too large to discriminate between bakes."
                )
        out_lines.append("")

    # Cross-band ranking — sum of SROCC across low-n bands, per bake
    out_lines.append("## Least bad across all low-n bands")
    out_lines.append("")
    bake_sums: dict[str, list[float]] = {}
    bake_counts: dict[str, int] = {}
    for key in low_n:
        corpus, band, n, rng = key
        if n < 30:
            continue  # Skip very-low-n (rankings are coin-flips)
        for label, row in band_table[key].items():
            srocc = row.get("srocc")
            if srocc is None:
                continue
            bake_sums.setdefault(label, []).append(srocc)
            bake_counts[label] = bake_counts.get(label, 0) + 1
    ranking = sorted(
        ((sum(v) / len(v), label) for label, v in bake_sums.items() if v),
        reverse=True,
    )
    out_lines.append("Aggregating across all `30 ≤ n < 100` bands, sorted by mean SROCC:")
    out_lines.append("")
    out_lines.append("| Rank | Bake | Mean SROCC (low-n bands) | n_bands |")
    out_lines.append("|---:|---|---:|---:|")
    for rank, (mean, label) in enumerate(ranking, 1):
        out_lines.append(
            f"| {rank} | {label} | {mean:.4f} | {bake_counts[label]} |"
        )
    out_lines.append("")
    if ranking:
        winner = ranking[0][1]
        out_lines.append(
            f"**Least bad across low-n bands**: **{winner}** (mean SROCC = {ranking[0][0]:.4f})."
        )
    out_lines.append("")
    out_lines.append(
        "Caveat: 'mean of SROCC' is not a valid statistic for cross-band"
        " ranking — different bands have different n and noise floors. This is"
        " a heuristic for 'overall low-sample behavior'. For a rigorous ranking,"
        " test pairwise via MRR or Wilcoxon paired."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
