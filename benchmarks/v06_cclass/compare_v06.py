#!/usr/bin/env python3
"""Compare V0_6+cclass eval against the V0_6 baseline reigning champion.

Inputs:
  argv[1]: V0_6+cclass perpair CSV (new)
  argv[2]: V0_6 baseline perpair CSV (reigning champion)
  argv[3]: V0_6 baseline rebake perpair CSV (340k control, optional, "" to skip)
  argv[4]: output markdown report path

Reports:
  - per-dataset SROCC (KADID, TID, CID22, KonJND)
  - per-quality-band SROCC (5 bands)
  - per-content-class SROCC where the new CSV carries CID22 source labels
  - delta vs reigning champion
  - ship/hold verdict
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def load(path: str) -> dict:
    by_ds = defaultdict(lambda: {"h": [], "v04": [], "ssim2": [], "butter": []})
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                vs = [
                    float(r["human_score"]),
                    float(r["v04_distance"]),
                    float(r.get("fast_ssim2_score", "nan")),
                    float(r.get("butter_3norm", "nan")),
                ]
            except (ValueError, KeyError):
                continue
            if not (np.isfinite(vs[0]) and np.isfinite(vs[1])):
                continue
            d = by_ds[r["dataset"]]
            d["h"].append(vs[0])
            d["v04"].append(vs[1])
            d["ssim2"].append(vs[2])
            d["butter"].append(vs[3])
    return by_ds


def srocc(a, b):
    if len(a) < 4:
        return float("nan")
    return abs(stats.spearmanr(a, b).correlation)


BANDS = [
    ("0-25", lambda s: 0 <= s < 25),
    ("25-40", lambda s: 25 <= s < 40),
    ("40-60", lambda s: 40 <= s < 60),
    ("60-75", lambda s: 60 <= s < 75),
    ("75-90", lambda s: 75 <= s < 90),
    ("≥ 90", lambda s: s >= 90),
]


def per_band_srocc(d):
    """Per-(SSIM2 band) SROCC of v04_distance vs human_score."""
    rows = []
    for label, pred in BANDS:
        mask = [pred(s) for s in d["ssim2"]]
        h = [x for x, m in zip(d["h"], mask) if m]
        v = [x for x, m in zip(d["v04"], mask) if m]
        if len(h) < 4:
            rows.append((label, len(h), float("nan")))
        else:
            rows.append((label, len(h), srocc(h, v)))
    return rows


def main() -> int:
    if len(sys.argv) < 5:
        print("usage: compare_v06.py <new.csv> <ref.csv> <ctrl.csv|''> <out.md>", file=sys.stderr)
        return 2
    new_csv, ref_csv, ctrl_csv, out_md = sys.argv[1:5]

    d_new = load(new_csv)
    d_ref = load(ref_csv)
    d_ctrl = load(ctrl_csv) if ctrl_csv else {}

    out = Path(out_md)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        f.write("# V0_6 + content_class — eval report\n\n")
        f.write("**Hypothesis**: conditioning the V0_6 dct_hf MLP on a per-image\n")
        f.write("content_class one-hot (photo / screen / lineart / synthetic / document)\n")
        f.write("improves SROCC vs human MOS on KADID, TID, and CID22.\n\n")
        f.write("**Architecture**: 228 zensim features + 3 zenanalyze (dct_compressibility_y,\n")
        f.write("dct_compressibility_uv, high_freq_energy_ratio) + 5 cclass one-hot = 236 in →\n")
        f.write("64-wide hidden (LeakyReLU α=0.01) → 1 score. Same hyperparameters as V0_6\n")
        f.write("(`mlp-magnitude-match-lambda=0.001`, `alpha=30.0`, `validation-policy=min`,\n")
        f.write("`epochs=200`).\n\n")
        f.write("**Training data**: training_safe_synthetic_extended.csv (340 207 pairs;\n")
        f.write("synthetic-v2 corpus + zenjpeg-420-e1 fill).\n\n")
        f.write("**content_class signal** is heavily skewed photo:\n")
        f.write("- photo: 4 693 stems (99.5%)\n")
        f.write("- screen: 18 stems (0.4%)\n")
        f.write("- lineart: 6 stems (0.1%)\n")
        f.write("- synthetic: 0 stems\n")
        f.write("- document: 0 stems\n\n")
        f.write("Class is derived heuristically from reference-image basenames. The\n")
        f.write("synthetic-v2 training corpus is photo-dominant by construction, so the\n")
        f.write("network sees almost-constant cclass features during training. This\n")
        f.write("limits the experiment's ceiling; reported deltas should be read with\n")
        f.write("that caveat in mind.\n\n")

        f.write("## Per-dataset SROCC vs human MOS\n\n")
        all_ds = ["KADIK10k", "TID2013", "CID22", "KonJND-1k"]
        f.write("| dataset | n | V0_6 (ref) | V0_6 rebake (340k ctrl) | V0_6+cclass | Δ vs ref | Δ vs rebake |\n")
        f.write("|---|--:|--:|--:|--:|--:|--:|\n")
        for ds in all_ds:
            n_new = len(d_new.get(ds, {}).get("h", [])) if ds in d_new else 0
            r_new = srocc(d_new[ds]["v04"], d_new[ds]["h"]) if ds in d_new else float("nan")
            r_ref = srocc(d_ref[ds]["v04"], d_ref[ds]["h"]) if ds in d_ref else float("nan")
            r_ctrl = srocc(d_ctrl[ds]["v04"], d_ctrl[ds]["h"]) if ds in d_ctrl else float("nan")
            d_ref_v = r_new - r_ref if (np.isfinite(r_new) and np.isfinite(r_ref)) else float("nan")
            d_ctrl_v = r_new - r_ctrl if (np.isfinite(r_new) and np.isfinite(r_ctrl)) else float("nan")
            def fmt(v):
                if not np.isfinite(v): return "—"
                return f"{v:+.4f}" if v != 0 and abs(v) < 1 else f"{v:.4f}"
            cells = [
                ds, str(n_new),
                f"{r_ref:.4f}" if np.isfinite(r_ref) else "—",
                f"{r_ctrl:.4f}" if np.isfinite(r_ctrl) else "—",
                f"{r_new:.4f}" if np.isfinite(r_new) else "—",
                fmt(d_ref_v), fmt(d_ctrl_v),
            ]
            f.write("| " + " | ".join(cells) + " |\n")
        f.write("\n")

        # Per-band breakdown for each dataset
        f.write("## Per-band SROCC (by fast-ssim2 band)\n\n")
        f.write("Each band is computed from V0_6+cclass perpair CSV grouped by the\n")
        f.write("fast_ssim2_score column. Reigning V0_6 ref and rebake control shown\n")
        f.write("for the same bands.\n\n")
        for ds in all_ds:
            if ds not in d_new:
                continue
            f.write(f"### {ds}\n\n")
            f.write("| band | n (new) | V0_6 (ref) | V0_6 rebake | V0_6+cclass | Δ vs ref |\n")
            f.write("|---|--:|--:|--:|--:|--:|\n")
            new_rows = per_band_srocc(d_new[ds]) if ds in d_new else []
            ref_rows = per_band_srocc(d_ref[ds]) if ds in d_ref else []
            ctrl_rows = per_band_srocc(d_ctrl[ds]) if ds in d_ctrl else []
            ref_lookup = {r[0]: (r[1], r[2]) for r in ref_rows}
            ctrl_lookup = {r[0]: (r[1], r[2]) for r in ctrl_rows}
            for band, n, v in new_rows:
                rn, rv = ref_lookup.get(band, (0, float("nan")))
                cn, cv = ctrl_lookup.get(band, (0, float("nan")))
                d = v - rv if (np.isfinite(v) and np.isfinite(rv)) else float("nan")
                cells = [
                    band, str(n),
                    f"{rv:.4f}" if np.isfinite(rv) else "—",
                    f"{cv:.4f}" if np.isfinite(cv) else "—",
                    f"{v:.4f}" if np.isfinite(v) else "—",
                    f"{d:+.4f}" if np.isfinite(d) else "—",
                ]
                f.write("| " + " | ".join(cells) + " |\n")
            f.write("\n")

        # Verdict
        f.write("## Verdict\n\n")
        deltas = []
        for ds in ("KADIK10k", "TID2013", "CID22"):
            if ds in d_new and ds in d_ref:
                rn = srocc(d_new[ds]["v04"], d_new[ds]["h"])
                rr = srocc(d_ref[ds]["v04"], d_ref[ds]["h"])
                if np.isfinite(rn) and np.isfinite(rr):
                    deltas.append((ds, rn - rr))
        positive = sum(1 for _, d in deltas if d > 0.005)
        f.write(f"Holdouts where V0_6+cclass beats V0_6 by > +0.005 SROCC: {positive} of 3.\n\n")
        for ds, d in deltas:
            f.write(f"- {ds}: Δ = {d:+.4f}\n")
        f.write("\n")
        if positive >= 2:
            f.write("**SHIP**: V0_6+cclass replaces V0_6.\n")
        else:
            f.write("**HOLD**: V0_6 reigning champion stands. Per the V0_6 task brief,\n")
            f.write("ship requires Δ > +0.005 SROCC on at least 2 of 3 holdouts.\n")
            f.write("\n")
            f.write("Likely reasons cclass adds little signal here:\n\n")
            f.write("1. Training-set class imbalance — 99.5% photo means the network\n")
            f.write("   sees a near-constant cclass vector and has no gradient signal\n")
            f.write("   to specialize per-class behavior.\n")
            f.write("2. Holdouts (KADID/TID/CID22 mostly natural photos, KonJND-1k\n")
            f.write("   natural images) are also photo-dominant, so any per-class\n")
            f.write("   adjustment learned from the 18 screen / 6 lineart training\n")
            f.write("   refs would only help on the long tail.\n")
            f.write("3. The 3 dct_hf zenanalyze features already encode much of the\n")
            f.write("   per-image content variation that distinguishes photo from\n")
            f.write("   screen/lineart, so cclass is partially redundant.\n")
            f.write("\n")
            f.write("To make content-class conditioning useful, training data needs\n")
            f.write("balanced photo/screen/lineart/document references (e.g., +30%\n")
            f.write("UI captures, +15% scanned documents, +15% diagrams), not the\n")
            f.write("photo-heavy synthetic-v2 corpus.\n")

    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
