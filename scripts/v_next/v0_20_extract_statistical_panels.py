#!/usr/bin/env python3
"""Extract the full statistical-rigor panels (SROCC + PLCC + KROCC + OR
+ PWRC + Z-RMSE) from every V_X eval log and build a side-by-side
comparison table.

CLAUDE.md "Statistical rigor (2026-05-14)" mandates the full panel
for every eval. Each `dataset_metric_baseline` log has 3 panels (per
corpus). This script consolidates across bakes for direct comparison.

## Usage

  python3 scripts/v_next/v0_20_extract_statistical_panels.py \\
    --out benchmarks/v0_20_all_bakes_stat_comparison_2026-05-15.md

## What it parses

For each known eval log, it parses the
"### <corpus> full statistical panel" sections, extracts the V0_4
(bake) row, plus the static fast-ssim2 + V0_2 + butteraugli
baselines for context.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

CORPORA = ("KADIK10k", "TID2013", "CID22")
METRICS = ("SROCC", "PLCC", "KROCC", "OR", "PWRC", "Z-RMSE")

# (bake_label, eval_log_relative_path, brief_description)
BAKES = [
    ("V_18 ship (3-way concat)", "benchmarks/v0_18_ship_reference_card_2026-05-14.log",
     "shipped baseline — affine-calibrated 0.65/0.30/0.05 concat"),
    ("V_18 base seed=1 (single MLP)", "benchmarks/v0_18_base_seed1_eval_2026-05-15.log",
     "apples-to-apples single-MLP comparison baseline"),
    ("V_20 IS (98 transforms, single MLP)", "benchmarks/v0_20_input_shaping_eval_2026-05-15.log",
     "V_20 input-shaping at lift>=0.05"),
    ("V_20b distortion manifold", "benchmarks/v0_20b_seed1_eval_2026-05-15.log",
     "Su 2023 contrastive pre-train + fine-tune"),
    ("D1 3-way concat with transforms", "benchmarks/v0_20_d1_concat_eval_2026-05-15.log",
     "V_20 IS + cycle-14 TV components, 0.65/0.30/0.05 mix"),
    ("D3 tighter transforms (lift>=0.10)", "benchmarks/v0_20_input_shaping_lift10_eval_2026-05-15.log",
     "V_20 input-shaping at lift>=0.10 (60 features)"),
]


def parse_panel(log_path: Path, corpus: str) -> dict[str, dict[str, float]] | None:
    """Extract one corpus's stat panel. Returns dict of metric_row → stat_dict."""
    if not log_path.exists():
        return None
    text = log_path.read_text()
    # Find the section "### <corpus> full statistical panel" and read until next section
    pat = rf"### {re.escape(corpus)} full statistical panel.*?\n(.*?)\n###"
    m = re.search(pat, text, re.DOTALL)
    if not m:
        return None
    block = m.group(1)
    # Parse markdown table lines: `| Metric | SROCC | PLCC | ... |`
    panel: dict[str, dict[str, float]] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|") or "---" in line or "Metric" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 7:
            continue
        row_name = cells[0]
        try:
            vals = {
                "SROCC": float(cells[1]),
                "PLCC": float(cells[2]),
                "KROCC": float(cells[3]),
                "OR": float(cells[4]),
                "PWRC": float(cells[5]),
                "Z-RMSE": float(cells[6]),
            }
        except ValueError:
            continue
        panel[row_name] = vals
    return panel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repo-root", default=".", type=Path)
    args = ap.parse_args()

    rows: list[tuple[str, str, dict[str, dict[str, float]] | None]] = []
    for label, log_rel, desc in BAKES:
        log_path = args.repo_root / log_rel
        panels = {c: parse_panel(log_path, c) for c in CORPORA}
        rows.append((label, log_rel, panels))  # type: ignore[arg-type]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_lines = ["# V_20 full statistical-rigor comparison across all bakes (2026-05-15)"]
    out_lines.append("")
    out_lines.append("Per CLAUDE.md mandate, every eval emits SROCC + PLCC + KROCC + ")
    out_lines.append("OR + PWRC + Z-RMSE. This consolidates the V_X bake row from each")
    out_lines.append("eval log into a single comparison.")
    out_lines.append("")
    out_lines.append("Static baselines for reference: V_2 (linear) + fast-ssim2 +")
    out_lines.append("butteraugli (Z-RMSE notes: corpus-wide σ on KADID/TID/CID22 since")
    out_lines.append("they don't carry bootstrap σ; AIC-3/AIC-4 have per-stimulus σ).")
    out_lines.append("")

    # Build per-corpus tables
    for corpus in CORPORA:
        out_lines.append(f"## {corpus}")
        out_lines.append("")
        # Header
        out_lines.append(
            "| Bake | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |"
        )
        out_lines.append("|---|---:|---:|---:|---:|---:|---:|")

        # Reference rows from first eval log that has the panel
        ref_panel = None
        for _, _, panels in rows:
            if panels.get(corpus):
                ref_panel = panels[corpus]
                break
        if ref_panel:
            for static_row in ("V0_2", "fast-ssim2", "butteraugli"):
                if static_row in ref_panel:
                    r = ref_panel[static_row]
                    out_lines.append(
                        f"| {static_row} (static) | {r['SROCC']:.4f} | {r['PLCC']:.4f} | "
                        f"{r['KROCC']:.4f} | {r['OR']:.4f} | {r['PWRC']:.4f} | "
                        f"{r['Z-RMSE']:.3f} |"
                    )

        # V0_4 (bake) row per V_X bake
        for label, _, panels in rows:
            corpus_panel = panels.get(corpus)
            if corpus_panel and "V0_4 (bake)" in corpus_panel:
                r = corpus_panel["V0_4 (bake)"]
                out_lines.append(
                    f"| **{label}** | **{r['SROCC']:.4f}** | {r['PLCC']:.4f} | "
                    f"{r['KROCC']:.4f} | {r['OR']:.4f} | {r['PWRC']:.4f} | "
                    f"{r['Z-RMSE']:.3f} |"
                )
            else:
                out_lines.append(f"| **{label}** | _(eval log missing or unparseable)_ | | | | | |")
        out_lines.append("")

    # Add interpretation
    out_lines.append("## Reading notes")
    out_lines.append("")
    out_lines.append("- **SROCC** is rank correlation. Calibration-invariant.")
    out_lines.append("- **PLCC** is Pearson on calibrated outputs vs MOS. Sensitive to")
    out_lines.append("  output scale — V_20 IS bakes are RAW (no affine calibration),")
    out_lines.append("  so their PLCC can mislead. V_18 ship is affine-calibrated.")
    out_lines.append("- **KROCC** is Kendall-τ — alternative to SROCC; sometimes more")
    out_lines.append("  stable at low n.")
    out_lines.append("- **OR** = outlier ratio (fraction of predictions outside ±2σ of")
    out_lines.append("  subjective). Lower is better.")
    out_lines.append("- **PWRC** = Pearson-weighted rank correlation (Mohammadi 2025).")
    out_lines.append("- **Z-RMSE** = σ-normalized RMSE on calibrated outputs. Lower is")
    out_lines.append("  better. On KADID/TID/CID22 this uses corpus-wide σ (less")
    out_lines.append("  informative than the AIC-3 per-stimulus form).")
    out_lines.append("")
    out_lines.append("**Caveat for V_20 IS / V_20b / D1 / D3**: PLCC + Z-RMSE on these")
    out_lines.append("rows reflect the bake's RAW output range, not a calibrated 0..100")
    out_lines.append("score. For direct comparison with V_18 ship's PLCC, the V_X bakes")
    out_lines.append("would need affine calibration via")
    out_lines.append("`affine_calibrate` (zensim-validate bin). SROCC + KROCC +")
    out_lines.append("PWRC are calibration-invariant and tell the true ranking story.")

    args.out.write_text("\n".join(out_lines) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
