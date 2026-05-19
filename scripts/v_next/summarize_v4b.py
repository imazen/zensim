#!/usr/bin/env python3
"""EXP-CROSS-CODEC-V4 summary table builder (2026-05-19).

Reads:
  - {V4_DIR}/qsweep_v4b.md → per-bake mono/tied/range/band-rmse
  - {V4_DIR}/cross_codec_t63/<bake>_t63_n20.tsv → T=63 butter_max, butter_p3
  - {V4_DIR}/verdicts/<bake>.md → CID22 SROCC + KonJND SROCC + AIC-3 SROCC
  - {V4_DIR}/v4b_pjnd_check.md → cross-codec score std median

Emits a single combined table to {V4_DIR}/v4b_summary.md.

Gate (per task brief):
  - strict_mono ≥ 0.9378
  - tied ≤ 5 %
  - range ≥ 50
  - T=63 butter_max < 2.5 OR butter_p3 < 2.5
  - cross-codec PJND score std median ≤ 5
If ALL: pass → ship as PreviewV0_5TunerV2.

Usage:
    python3 scripts/v_next/summarize_v4.py /mnt/v/zen/zensim-eval/exp_cross_codec_v4_2026-05-19
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def parse_qsweep(md_path: Path) -> dict[str, dict]:
    """Parse qsweep_v4b.md, return per-bake {mono, tied, range, q5_med, q95_med}.

    qsweep_eval emits a global mono/tied table near the top:
      | Bake | n_curves | n_adj_pairs | strict_violations | tied | monotonicity_rate | tied_rate |
      | baseline_tuner | 50 | 900 | 65 | 4 | 0.9278 | 0.0044 |

    Then per-bake `### <bake_name>` sections with q-distribution table.
    """
    if not md_path.exists():
        return {}
    text = md_path.read_text()

    # Pass 1: parse global mono/tied table.
    mono_by_bake: dict[str, dict] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 7:
            continue
        bake = cells[0]
        if bake in ("Bake", "---") or bake.startswith("---"):
            continue
        try:
            mono = float(cells[5])
            tied = float(cells[6])
            mono_by_bake[bake] = {"strict_mono": mono, "tied": tied}
        except (ValueError, IndexError):
            continue

    # Pass 2: parse per-bake q5/q95 medians.
    parts = re.split(r"^### (\S+)", text, flags=re.MULTILINE)
    out: dict[str, dict] = {}
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        med = {}
        for line in body.splitlines():
            if line.count("|") < 7:
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 7:
                continue
            try:
                q = int(cells[0])
            except ValueError:
                continue
            try:
                m_val = float(cells[4])
            except (ValueError, IndexError):
                continue
            med[q] = m_val
        q5 = med.get(5)
        q95 = med.get(95)
        rng = (q95 - q5) if (q5 is not None and q95 is not None) else None
        out[name] = {
            "q5_med": q5,
            "q95_med": q95,
            "range": rng,
            "strict_mono": mono_by_bake.get(name, {}).get("strict_mono"),
            "tied": mono_by_bake.get(name, {}).get("tied"),
        }
    return out


def parse_cross_codec_t63(v4_dir: Path) -> dict[str, dict]:
    """Read each cross_codec_t63/<bake>_t63_n20.tsv and extract mean butter_max,
    butter_p3 across the n=20 image rows. TSV format from cross_codec_consistency.py:
    columns include pairwise_butter_max_mean, pairwise_butter_p3_mean.
    """
    out = {}
    cc_dir = v4_dir / "cross_codec_t63"
    if not cc_dir.exists():
        return {}
    for tsv in cc_dir.glob("*_t63_n20.tsv"):
        # Bake name is the filename stem minus "_t63_n20"
        name = tsv.stem.replace("_t63_n20", "")
        if not tsv.exists():
            continue
        try:
            with open(tsv) as f:
                lines = [l.strip() for l in f if l.strip()]
            if len(lines) < 2:
                continue
            header = lines[0].split("\t")
            try:
                bmax_col = header.index("pairwise_butter_max_mean")
                bp3_col = header.index("pairwise_butter_p3_mean")
            except ValueError:
                continue
            bmax_vals = []
            bp3_vals = []
            for line in lines[1:]:
                cells = line.split("\t")
                if len(cells) <= max(bmax_col, bp3_col):
                    continue
                try:
                    bmax_vals.append(float(cells[bmax_col]))
                    bp3_vals.append(float(cells[bp3_col]))
                except ValueError:
                    continue
            if bmax_vals:
                out[name] = {
                    "butter_max_mean": sum(bmax_vals) / len(bmax_vals),
                    "butter_p3_mean": sum(bp3_vals) / len(bp3_vals),
                    "n": len(bmax_vals),
                }
        except Exception as e:
            print(f"  parse error on {tsv}: {e}", file=sys.stderr)
    return out


def parse_verdicts(v4_dir: Path) -> dict[str, dict]:
    """Parse bake_verdict outputs for CID22/KADID/TID/KonJND/AIC-3 SROCC."""
    out = {}
    vdir = v4_dir / "verdicts"
    if not vdir.exists():
        return {}
    for md in vdir.glob("cc4v4b_*.md"):
        name = md.stem
        text = md.read_text()
        # The summary table at the top has columns: corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE
        srocc = {}
        for line in text.splitlines():
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 8:
                continue
            corpus = cells[0]
            try:
                val = float(cells[2])
            except ValueError:
                continue
            if corpus in ("CID22", "KADIK10k", "TID2013", "KonJND-1k (full)", "AIC-3 CTC"):
                srocc[corpus] = val
        out[name] = srocc
    return out


def parse_pjnd_check(md_path: Path) -> dict[str, dict]:
    """Parse v4b_pjnd_check.md table for cc_std_median, agg_mean per bake."""
    out = {}
    if not md_path.exists():
        return {}
    text = md_path.read_text()
    in_table = False
    for line in text.splitlines():
        if "| bake | agg_mean" in line:
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            in_table = False
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 7:
            continue
        name = cells[0]
        if name in ("---", "bake"):
            continue
        try:
            out[name] = {
                "agg_mean": float(cells[1]),
                "agg_std": float(cells[2]),
                "cc_std_median": float(cells[3]),
                "cc_std_mean": float(cells[4]),
                "cc_std_p95": float(cells[5]),
            }
        except ValueError:
            pass
    return out


def gate_mono(v): return "PASS" if (v is not None and v >= 0.9378) else "FAIL"
def gate_tied(v): return "PASS" if (v is not None and v <= 0.05) else "FAIL"
def gate_range(v): return "PASS" if (v is not None and v >= 50.0) else "FAIL"
def gate_cc_butter(bmax, bp3):
    return "PASS" if (bmax is not None and bmax < 2.5) or (bp3 is not None and bp3 < 2.5) else "FAIL"
def gate_pjnd_std(v): return "PASS" if (v is not None and v <= 5.0) else "FAIL"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: summarize_v4.py <v4_dir>", file=sys.stderr)
        return 2
    v4_dir = Path(sys.argv[1])

    qsweep = parse_qsweep(v4_dir / "qsweep_v4b.md")
    t63 = parse_cross_codec_t63(v4_dir)
    verdicts = parse_verdicts(v4_dir)
    pjnd = parse_pjnd_check(v4_dir / "v4b_pjnd_check.md")

    # Find all V4 bake names
    bakes = sorted({b.stem for b in v4_dir.glob("cc4v4b_*.bin")})
    if not bakes:
        print("no cc4v4b_*.bin found", file=sys.stderr)
        return 1

    out_md = v4_dir / "v4b_summary.md"
    with open(out_md, "w") as f:
        f.write("# EXP-CROSS-CODEC-V4 summary table (2026-05-19)\n\n")
        f.write("## Tuner-trail gate scorecard\n\n")
        f.write("Gates per V4 ship criteria:\n")
        f.write("- strict_mono ≥ 0.9378\n")
        f.write("- tied ≤ 5%\n")
        f.write("- range ≥ 50\n")
        f.write("- T=63 butter_max < 2.5 OR butter_p3 < 2.5\n")
        f.write("- cross-codec score std per source median ≤ 5.0\n\n")
        f.write("| Bake | mono | tied | range | T63 b_max | T63 b_p3 | PJND std | mono | tied | range | xc | pjnd | ALL |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|:-:|:-:|:-:|:-:|:-:|:-:|\n")
        for b in bakes:
            qs = qsweep.get(b, {})
            t63b = t63.get(b, {})
            pjndb = pjnd.get(b, {})
            mono = qs.get("strict_mono")
            tied = qs.get("tied")
            rng = qs.get("range")
            bmax = t63b.get("butter_max_mean")
            bp3 = t63b.get("butter_p3_mean")
            cc_std = pjndb.get("cc_std_median")
            agg = pjndb.get("agg_mean")
            mono_s = f"{mono:.4f}" if mono is not None else "—"
            tied_s = f"{tied:.4f}" if tied is not None else "—"
            rng_s = f"{rng:.2f}" if rng is not None else "—"
            bmax_s = f"{bmax:.3f}" if bmax is not None else "—"
            bp3_s = f"{bp3:.3f}" if bp3 is not None else "—"
            cc_s = f"{cc_std:.3f}" if cc_std is not None else "—"
            g_mono = gate_mono(mono)
            g_tied = gate_tied(tied)
            g_rng = gate_range(rng)
            g_xc = gate_cc_butter(bmax, bp3)
            g_pjnd = gate_pjnd_std(cc_std)
            g_all = "PASS" if all(g == "PASS" for g in [g_mono, g_tied, g_rng, g_xc, g_pjnd]) else "FAIL"
            check = lambda s: "✓" if s == "PASS" else "✗"
            f.write(
                f"| {b} | {mono_s} | {tied_s} | {rng_s} | {bmax_s} | {bp3_s} | {cc_s} | "
                f"{check(g_mono)} | {check(g_tied)} | {check(g_rng)} | {check(g_xc)} | {check(g_pjnd)} | {g_all} |\n"
            )

        # Verdicts side-by-side
        f.write("\n## Mohammadi SROCC panel per bake\n\n")
        f.write("| Bake | CID22 | KADID | TID | KonJND | AIC-3 |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for b in bakes:
            v = verdicts.get(b, {})
            row = [b]
            for corp in ("CID22", "KADIK10k", "TID2013", "KonJND-1k (full)", "AIC-3 CTC"):
                val = v.get(corp)
                row.append(f"{val:.4f}" if val is not None else "—")
            f.write("| " + " | ".join(row) + " |\n")

        # PJND aggregate stats
        f.write("\n## Multi-codec PJND score aggregate (target 63.0)\n\n")
        f.write("| Bake | agg_mean | agg_std | cc_std_median | cc_std_p95 |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for b in bakes:
            p = pjnd.get(b, {})
            f.write(
                f"| {b} | "
                f"{p.get('agg_mean', float('nan')):.2f} | "
                f"{p.get('agg_std', float('nan')):.2f} | "
                f"{p.get('cc_std_median', float('nan')):.3f} | "
                f"{p.get('cc_std_p95', float('nan')):.3f} |\n"
            )
    print(f"wrote {out_md}")

    # Also emit to stdout for quick inspection.
    print()
    print(out_md.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
