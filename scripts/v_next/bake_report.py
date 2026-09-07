#!/usr/bin/env python3
"""Per-bake visual report: Cloudinary-style human-vs-score scatterplots
(cf. ssimulacra2's metric_correlation-scatterplots-MCOS-all.svg) over the six
canonical validation corpora, plus the full Mohammadi stat panel per corpus.

THE RULE (user directive 2026-07-02): every result gets one of these, they
collect under /mnt/v/output/zensim/reports/ (browsable at
http://localhost:3300/zensim/reports/), and the viewer index.html is
regenerated on every run. runcells.sh calls this automatically per cell, so
reports are the NORMAL output of training, not an extra step.

Scientific validity notes (rendered into every report):
- CID22 here is the 49-ref HOLDOUT (never trained by zensim; ssim2's own
  validation split) — stricter than the Cloudinary SVG, which includes the
  201 refs ssim2 tuned on.
- KADID/TID panels carry an in-sample banner: they train (w=0.5) in our
  recipes AND ssim2 tuned on them — integrity guards, not rankings.
- Stats come from scripts/lib/zen_stats (the canonical panel — no hand-rolled
  correlation math); the 4PL curve is a display aid fitted per panel.

Usage:
  bake_report.py --bake X.bin [--label NAME]
      [--out-root /mnt/v/output/zensim/reports]
      [--verdict-bin ./target/release/bake_verdict]
"""
import argparse
import datetime
import hashlib
import html
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.zen_stats import panel  # canonical stats — do not hand-roll

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

CORPORA = [
    ("cid22", "CID22-49 holdout (MCOS)", "T0 — honest for zensim AND ssim2"),
    ("imazen26", "imazen-26 real-codec (ssim2)", "T0 — real modern-codec ssim2-agreement (2026-07-16)"),
    ("nonphoto", "imazen-26 non-photo (ssim2)", "T0 — non-photo ssim2-agreement"),
    ("aic3", "AIC-3 CTC (JND)", "T0 holdout"),
    ("aic4", "AIC-4 sample (JND)", "T0 holdout — no recipe-search"),
    ("konjnd", "KonJND-1k (PJND)", "T1 guard/anchor"),
    ("kadid", "KADID-10k (DMOS)", "T1 — IN-SAMPLE (ours w=0.5 + ssim2 tuned on it)"),
    ("tid", "TID2013 (MOS)", "T1 — IN-SAMPLE (ours w=0.5 + ssim2 tuned on it)"),
]


def logistic4(x, a, b, c, d):
    return a + (b - a) / (1.0 + np.exp(-(x - c) / (d if abs(d) > 1e-9 else 1e-9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", required=True)
    ap.add_argument("--label")
    ap.add_argument("--out-root", default="/mnt/v/output/zensim/reports")
    ap.add_argument("--verdict-bin", default="./target/release/bake_verdict")
    a = ap.parse_args()
    label = a.label or os.path.splitext(os.path.basename(a.bake))[0]
    stamp = datetime.date.today().isoformat()
    out_dir = os.path.join(a.out_root, f"{stamp}_{label}")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(21, 10))
    fig.suptitle(
        f"{label} — human score vs bake prediction (per-corpus, 4PL display fit)",
        fontsize=14,
    )
    stats_rows = []
    for ax, (key, title, tier) in zip(axes.flat, CORPORA):
        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tf:
            pp = tf.name
        r = subprocess.run(
            [a.verdict_bin, "--bake", a.bake, "--corpora", key,
             "--per-pair-output", pp, "--output", os.path.join(out_dir, f"{key}.verdict.md")],
            capture_output=True, text=True,
        )
        rows = []
        if os.path.exists(pp):
            with open(pp) as f:
                next(f, None)
                for line in f:
                    h, p = line.split("\t")
                    rows.append((float(h), float(p)))
            os.unlink(pp)
        if not rows:
            ax.set_title(f"{title}\n(unavailable: {r.returncode})", fontsize=10)
            ax.axis("off")
            stats_rows.append((key, title, tier, 0, None))
            continue
        hum = np.array([x[0] for x in rows])
        pred = np.array([x[1] for x in rows])
        st = panel(pred.tolist(), hum.tolist())
        ax.scatter(pred, hum, s=4, alpha=0.25, color="#1f6fb2", edgecolors="none")
        try:
            p0 = [hum.min(), hum.max(), float(np.median(pred)), (pred.max() - pred.min()) / 4 or 1.0]
            popt, _ = curve_fit(logistic4, pred, hum, p0=p0, maxfev=8000)
            xs = np.linspace(pred.min(), pred.max(), 200)
            ax.plot(xs, logistic4(xs, *popt), color="#d1495b", lw=1.6)
        except Exception:
            pass
        ax.set_title(f"{title}  n={len(rows)}", fontsize=10)
        ax.set_xlabel(f"{label} score", fontsize=8)
        ax.set_ylabel("subjective", fontsize=8)
        ax.text(
            0.02, 0.98,
            f"SROCC {st['srocc']:+.4f}\nKROCC {st['krocc']:+.4f}\nPLCC  {st['plcc']:+.4f}\n"
            f"PWRC  {st['pwrc']:+.4f}\nZ-RMSE {st['z_rmse']:.3f}",
            transform=ax.transAxes, va="top", fontsize=7, family="monospace",
            bbox=dict(fc="white", alpha=0.75, ec="none"),
        )
        if "IN-SAMPLE" in tier:
            ax.text(0.98, 0.02, "IN-SAMPLE for ssim2 + trained here",
                    transform=ax.transAxes, ha="right", fontsize=7, color="#a33")
        stats_rows.append((key, title, tier, len(rows), st))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    svg = os.path.join(out_dir, "scatter_all.svg")
    fig.savefig(svg)
    fig.savefig(os.path.join(out_dir, "scatter_all.png"), dpi=110)
    plt.close(fig)

    sha = hashlib.sha256(open(a.bake, "rb").read()).hexdigest()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    json.dump(
        {"bake": os.path.abspath(a.bake), "sha256": sha, "label": label,
         "git": commit, "date": stamp, "cmd": " ".join(sys.argv)},
        open(os.path.join(out_dir, "meta.json"), "w"), indent=1,
    )
    with open(os.path.join(out_dir, "report.html"), "w") as f:
        f.write(f"<!doctype html><meta charset=\"utf-8\"><title>{html.escape(label)}</title>"
                f"<h1>{html.escape(label)}</h1>"
                f"<p>bake sha256 <code>{sha[:16]}…</code> · trainer <code>{commit[:12]}</code> · {stamp}</p>"
                f"<p><b>CID22 = 49-ref holdout</b> (ssim2's own validation split — stricter than the "
                f"Cloudinary all-CID22 SVG). KADID/TID are in-sample integrity guards for both metrics.</p>"
                f"<img src='scatter_all.svg' style='max-width:100%'>"
                f"<h2>Panels</h2><table border=1 cellpadding=4><tr><th>corpus</th><th>tier</th><th>n</th>"
                f"<th>SROCC</th><th>KROCC</th><th>PLCC</th><th>PWRC</th><th>Z-RMSE</th></tr>")
        for key, title, tier, n, st in stats_rows:
            if st is None:
                f.write(f"<tr><td>{title}</td><td>{tier}</td><td>0</td><td colspan=5>unavailable</td></tr>")
            else:
                f.write(f"<tr><td>{title}</td><td>{tier}</td><td>{n}</td>"
                        f"<td>{st['srocc']:+.4f}</td><td>{st['krocc']:+.4f}</td><td>{st['plcc']:+.4f}</td>"
                        f"<td>{st['pwrc']:+.4f}</td><td>{st['z_rmse']:.3f}</td></tr>")
        f.write("</table><p>Per-corpus verdict markdown files sit alongside this page.</p>")

    # regenerate the viewer index over ALL reports
    root = a.out_root
    entries = []
    for d in sorted(os.listdir(root), reverse=True):
        meta = os.path.join(root, d, "meta.json")
        if os.path.isfile(meta):
            m = json.load(open(meta))
            entries.append((d, m))
    with open(os.path.join(root, "index.html"), "w") as f:
        f.write("<!doctype html><title>zensim bake reports</title><h1>zensim bake reports</h1>"
                "<p>One report per result (bake_report.py — auto-run by runcells.sh). "
                "CID22 panels use the 49-ref holdout.</p><ul>")
        for d, m in entries:
            f.write(f"<li><a href='{d}/report.html'>{html.escape(m.get('label', d))}</a> "
                    f"— {m.get('date','')} · sha {m.get('sha256','')[:12]}…</li>")
        f.write("</ul>")
    print(f"report: {out_dir}/report.html")
    print(f"viewer: http://localhost:3300/zensim/reports/")


if __name__ == "__main__":
    main()
