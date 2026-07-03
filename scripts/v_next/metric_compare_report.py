#!/usr/bin/env python3
"""Cross-METRIC comparison report: candidate bake vs Profile-A vs ssim2 on the
same corpora, same visual grammar as bake_report.py (human-vs-score scatter,
4PL display fit), plus a per-corpus stats table (SROCC/PLCC/KROCC from the
canonical zen_stats panel). Bake predictions come from bake_verdict
--per-pair-output; ssim2 points come from (human, score) TSVs produced by
`zenmetrics batch` (columns: last = score, and a named human column).

  usage: metric_compare_report.py --candidate BAKE --label NAME \
             [--a-bake PATH] [--out-root /mnt/v/output/zensim/reports]
"""
import argparse, os, subprocess, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from scripts.lib.zen_stats import panel  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

P = "/mnt/v/output/zensim-multicodec-probe"
CORPORA = [
    ("cid22", "CID22-49 (T0 holdout — ssim2's own validation split)", "human MCOS"),
    ("aic3", "AIC-3 CTC (T0 holdout)", "human JND"),
    ("konjnd", "KonJND-1k (guard/anchor)", "human PJND"),
    ("sdr25", "JPEG-AI-SDR25 (post-ssim2 holdout)", "reconstructed JND"),
]
# ssim2 per-pair TSVs: (path, human_col, score_col, human_negate)
SSIM2_TSV = {
    "cid22": (f"{P}/cid22_ssim2_scores.tsv", "mcos", "ssim2", False),
    "aic3": (f"{P}/aic3_ssim2_scores.tsv", "jnd", "ssim2", False),
    "konjnd": (f"{P}/konjnd_ssim2_scores.tsv", "pjnd", "ssim2", False),
    "sdr25": (f"{P}/sdr25_ssim2.tsv", "q_jnd", "ssim2", True),
}


def logistic4(x, a, b, c, d):
    return a + (b - a) / (1.0 + np.exp(-(x - c) / (d if abs(d) > 1e-9 else 1e-9)))


def bake_pairs(verdict_bin, bake, corpus):
    """(human, pred) via bake_verdict --per-pair-output; sdr25 via the dial
    grid + the value-permutation join is pre-baked into a TSV by the caller."""
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tf:
        pp = tf.name
    subprocess.run([verdict_bin, "--bake", bake, "--corpora", corpus,
                    "--per-pair-output", pp, "--output", os.devnull],
                   capture_output=True, text=True)
    rows = []
    if os.path.exists(pp):
        with open(pp) as f:
            next(f, None)
            for line in f:
                h, p = line.split("\t")
                rows.append((float(h), float(p)))
        os.unlink(pp)
    return rows


def tsv_pairs(path, hcol, scol, negate_h):
    import csv
    rows = []
    for r in csv.DictReader(open(path), delimiter="\t"):
        h = float(r[hcol])
        rows.append((-h if negate_h else h, float(r[scol])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--a-bake", default="zensim/weights/v47_strict_qat_native_2026-05-27.bin")
    ap.add_argument("--sdr25-preds", help="TSV human\tpred for the candidate on SDR25 (pre-joined)")
    ap.add_argument("--sdr25-preds-a", help="same for Profile-A")
    ap.add_argument("--verdict-bin", default="./target/release/bake_verdict")
    ap.add_argument("--out-root", default="/mnt/v/output/zensim/reports")
    a = ap.parse_args()

    out_dir = os.path.join(a.out_root, f"2026-07-03_compare_{a.label}")
    os.makedirs(out_dir, exist_ok=True)
    entries = [(a.label, "#d62728"), ("ProfileA", "#1f6fb2"), ("ssim2", "#2ca02c")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(f"{a.label} vs Profile-A vs SSIMULACRA2 — human vs score", fontsize=13)
    table = ["| corpus | metric | SROCC | PLCC | KROCC | n |", "|---|---|---|---|---|---|"]
    for ax, (key, title, ylab) in zip(axes.flat, CORPORA):
        for name, color in entries:
            if key == "sdr25":
                src = {a.label: a.sdr25_preds, "ProfileA": a.sdr25_preds_a}.get(name)
                if name == "ssim2":
                    rows = tsv_pairs(*SSIM2_TSV[key])
                elif src and os.path.exists(src):
                    rows = [tuple(map(float, l.split("\t"))) for l in open(src).read().splitlines()[1:]]
                else:
                    rows = []
            elif name == "ssim2":
                rows = tsv_pairs(*SSIM2_TSV[key])
            else:
                bake = a.candidate if name == a.label else a.a_bake
                rows = bake_pairs(a.verdict_bin, bake, key)
            if not rows:
                table.append(f"| {key} | {name} | (unavailable) | | | |")
                continue
            hum = np.array([x[0] for x in rows])
            pred = np.array([x[1] for x in rows])
            st = panel(pred.tolist(), hum.tolist())
            table.append(f"| {key} | {name} | {st['srocc']:.4f} | {st['plcc']:.4f} | {st['krocc']:.4f} | {len(rows)} |")
            # normalize pred to [0,1] per metric for overlay comparability
            pn = (pred - pred.min()) / max(1e-9, pred.max() - pred.min())
            ax.scatter(pn, hum, s=4, alpha=0.22, color=color, edgecolors="none",
                       label=f"{name} ρ={st['srocc']:.3f}")
            try:
                p0 = [float(hum.min()), float(hum.max()), float(np.median(pn)), 0.2]
                popt, _ = curve_fit(logistic4, pn, hum, p0=p0, maxfev=8000)
                xs = np.linspace(0, 1, 200)
                ax.plot(xs, logistic4(xs, *popt), color=color, lw=1.4)
            except Exception:
                pass
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("metric score (min-max normalized)")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = os.path.join(out_dir, "compare.png")
    fig.savefig(png, dpi=130)
    open(os.path.join(out_dir, "stats.md"), "w").write("\n".join(table) + "\n")
    html = ["<html><head><title>compare: " + a.label + "</title></head><body style='font-family:sans-serif;max-width:1100px;margin:2em auto'>",
            f"<h1>{a.label} vs Profile-A vs SSIMULACRA2</h1>",
            "<p>CID22-49 is the honest head-to-head (ssim2's own holdout). KonJND: raw PJND targets — guard, not ranking. SDR25 postdates both metrics' tuning.</p>",
            "<img src='compare.png' style='max-width:100%'>", "<h2>Stats (canonical zen_stats panel)</h2><pre>"]
    html += table + ["</pre></body></html>"]
    open(os.path.join(out_dir, "index.html"), "w").write("\n".join(html))
    print("report:", out_dir)
    print("\n".join(table))


if __name__ == "__main__":
    main()
