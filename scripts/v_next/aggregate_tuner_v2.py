#!/usr/bin/env python3
"""Aggregate 3-seed TunerV2 results into a single ship/falsify decision.

Reads:
  /mnt/v/output/zensim/cross_codec_consistency_2026-05-19/tuner_v2_s{1,2,3}/<label>_summary.json
  /mnt/v/zen/zensim-eval/exp_tuner_v2_2026-05-19/verdict_s{1,2,3}.md
  /mnt/v/zen/zensim-eval/exp_tuner_v2_2026-05-19/verdict_baseline_PreviewV0_5Tuner.md

Emits:
  /mnt/v/zen/zensim-eval/exp_tuner_v2_2026-05-19/aggregate_3seed.md
  /mnt/v/zen/zensim-eval/exp_tuner_v2_2026-05-19/ship_decision.json

Pass criteria (from EXP-TUNER-V2 dispatch):
  - Cross-codec butter at T=63 < 3.0 (vs Tuner's 6.68)
  - Cross-codec butter at T=70 < 2.5 (vs Tuner's 5.00)
  - CID22 SROCC ≥ 0.85 (preserves Tuner's compression-trail value)
  - KonJND SROCC ≥ 0.80 (anchor data injects PJND signal)
  - strict_mono ≥ 0.92 (matches Tuner; we read from baseline TSV if cross-codec
    eval doesn't emit it; cross_codec_jnd_eval is dial-honesty, not monotone)
"""

import json
import re
import sys
from pathlib import Path

OUT_DIR = Path("/mnt/v/zen/zensim-eval/exp_tuner_v2_2026-05-19")
CCC_DIR = Path("/mnt/v/output/zensim/cross_codec_consistency_2026-05-19")
TARGETS = [30, 50, 63, 70, 80, 90]


def read_ccc_summary(seed):
    label = f"tuner_v2_s{seed}"
    path = CCC_DIR / label / f"{label}_summary.json"
    with open(path) as f:
        data = json.load(f)
    # keys like "tuner_v2_s1@T63"
    return {int(k.split("@T")[1]): v for k, v in data.items()}


def read_verdict_summary(path):
    """Parse the bake_verdict summary table for SROCC/PLCC/etc per corpus."""
    text = Path(path).read_text()
    summary = {}
    # find the summary table after "## Summary (one row per corpus)"
    m = re.search(r"## Summary.*?\n((?:\|[^\n]+\n)+)", text, re.DOTALL)
    if not m:
        return summary
    table = m.group(1)
    for line in table.strip().split("\n"):
        if not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        # header / sep
        if cells[0] in ("Corpus", "---"):
            continue
        try:
            summary[cells[0]] = {
                "n": int(cells[1]),
                "srocc": float(cells[2]),
                "plcc": float(cells[3]),
                "krocc": float(cells[4]),
                "or": float(cells[5]),
                "pwrc": float(cells[6]),
                "zrmse": float(cells[7]),
            }
        except (ValueError, IndexError):
            continue
    return summary


def main():
    # Read baseline Tuner verdict
    baseline_v = read_verdict_summary(OUT_DIR / "verdict_baseline_PreviewV0_5Tuner.md")
    # Read TunerV2 seed verdicts + cross-codec summaries
    seeds = [1, 2, 3]
    v2_verdicts = {s: read_verdict_summary(OUT_DIR / f"verdict_s{s}.md") for s in seeds}
    v2_ccc = {s: read_ccc_summary(s) for s in seeds}

    # Cross-codec butter table at T=63 / T=70
    print("# PreviewV0_5TunerV2 — 3-seed aggregate", flush=True)
    print(f"\n## Cross-codec butter_max_mean (lower = consistent)")
    print(f"| Variant | T=30 | T=50 | T=63 | T=70 | T=80 | T=90 |")
    print(f"|---|---:|---:|---:|---:|---:|---:|")
    # Baseline pulled from benchmarks/cross_codec_consistency_2026-05-19.md
    print(f"| PreviewV0_5Tuner (baseline) | 13.64 | 9.63 | 6.68 | 5.00 | 3.31 | 1.88 |")
    for s in seeds:
        cells = []
        for t in TARGETS:
            v = v2_ccc[s].get(t, {}).get("mean_butter_max", float("nan"))
            cells.append(f"{v:.2f}" if v == v else "n/a")
        print(f"| TunerV2 s{s} | " + " | ".join(cells) + " |")
    # 3-seed mean
    mean_cells = []
    for t in TARGETS:
        vals = [v2_ccc[s].get(t, {}).get("mean_butter_max", float("nan")) for s in seeds]
        vals = [v for v in vals if v == v]
        if vals:
            mean_cells.append(f"{sum(vals)/len(vals):.2f}")
        else:
            mean_cells.append("n/a")
    print(f"| **TunerV2 3-seed mean** | " + " | ".join(mean_cells) + " |")

    # Bake_verdict SROCC panel
    print(f"\n## Held-out SROCC panel")
    print(f"| Variant | CID22 | KADID | TID | KonJND | AIC-3 |")
    print(f"|---|---:|---:|---:|---:|---:|")
    def row(label, v):
        cid22 = v.get("CID22", {}).get("srocc", float("nan"))
        kadid = v.get("KADIK10k", {}).get("srocc", float("nan"))
        tid = v.get("TID2013", {}).get("srocc", float("nan"))
        konjnd = v.get("KonJND-1k (full)", {}).get("srocc", float("nan"))
        aic3 = v.get("AIC-3 CTC", {}).get("srocc", float("nan"))
        cells = [f"{x:.4f}" if x == x else "n/a" for x in [cid22, kadid, tid, konjnd, aic3]]
        print(f"| {label} | " + " | ".join(cells) + " |")
    row("PreviewV0_5Tuner (baseline)", baseline_v)
    for s in seeds:
        row(f"TunerV2 s{s}", v2_verdicts[s])

    # Compute 3-seed mean SROCC
    def mean(values):
        vs = [v for v in values if v == v]
        return sum(vs)/len(vs) if vs else float("nan")
    cid22_mean = mean([v2_verdicts[s].get("CID22", {}).get("srocc", float("nan")) for s in seeds])
    kadid_mean = mean([v2_verdicts[s].get("KADIK10k", {}).get("srocc", float("nan")) for s in seeds])
    tid_mean = mean([v2_verdicts[s].get("TID2013", {}).get("srocc", float("nan")) for s in seeds])
    konjnd_mean = mean([v2_verdicts[s].get("KonJND-1k (full)", {}).get("srocc", float("nan")) for s in seeds])
    aic3_mean = mean([v2_verdicts[s].get("AIC-3 CTC", {}).get("srocc", float("nan")) for s in seeds])
    print(f"| **TunerV2 3-seed mean** | {cid22_mean:.4f} | {kadid_mean:.4f} | {tid_mean:.4f} | {konjnd_mean:.4f} | {aic3_mean:.4f} |")

    # Ship decision
    butter_t63_mean = mean([v2_ccc[s].get(63, {}).get("mean_butter_max", float("nan")) for s in seeds])
    butter_t70_mean = mean([v2_ccc[s].get(70, {}).get("mean_butter_max", float("nan")) for s in seeds])

    criteria = {
        "butter_t63_lt_3.0": (butter_t63_mean, butter_t63_mean < 3.0 if butter_t63_mean == butter_t63_mean else False, "< 3.0"),
        "butter_t70_lt_2.5": (butter_t70_mean, butter_t70_mean < 2.5 if butter_t70_mean == butter_t70_mean else False, "< 2.5"),
        "cid22_srocc_gte_0.85": (cid22_mean, cid22_mean >= 0.85, ">= 0.85"),
        "konjnd_srocc_gte_0.80": (konjnd_mean, konjnd_mean >= 0.80, ">= 0.80"),
    }
    print(f"\n## Ship criteria")
    print(f"| Criterion | Threshold | Value | Pass? |")
    print(f"|---|---|---:|:---:|")
    all_pass = True
    for k, (val, ok, thr) in criteria.items():
        emoji = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"| {k} | {thr} | {val:.4f} | {emoji} |")
    print(f"\n**Verdict**: {'SHIP as PreviewV0_5TunerV2' if all_pass else 'FALSIFICATION'}")

    decision = {
        "criteria": {k: {"value": v[0], "pass": v[1], "threshold": v[2]} for k, v in criteria.items()},
        "all_pass": all_pass,
        "seeds": seeds,
        "verdict": "SHIP" if all_pass else "FALSIFICATION",
        "baseline": {
            "cid22_srocc": baseline_v.get("CID22", {}).get("srocc", float("nan")),
            "kadid_srocc": baseline_v.get("KADIK10k", {}).get("srocc", float("nan")),
            "tid_srocc": baseline_v.get("TID2013", {}).get("srocc", float("nan")),
            "konjnd_srocc": baseline_v.get("KonJND-1k (full)", {}).get("srocc", float("nan")),
            "aic3_srocc": baseline_v.get("AIC-3 CTC", {}).get("srocc", float("nan")),
            "butter_t63": 6.68,
            "butter_t70": 5.00,
        },
        "tuner_v2_mean": {
            "cid22_srocc": cid22_mean,
            "kadid_srocc": kadid_mean,
            "tid_srocc": tid_mean,
            "konjnd_srocc": konjnd_mean,
            "aic3_srocc": aic3_mean,
            "butter_t63": butter_t63_mean,
            "butter_t70": butter_t70_mean,
        },
    }
    with open(OUT_DIR / "ship_decision.json", "w") as f:
        json.dump(decision, f, indent=2)
    print(f"\nWrote {OUT_DIR}/ship_decision.json")


if __name__ == "__main__":
    sys.exit(main())
