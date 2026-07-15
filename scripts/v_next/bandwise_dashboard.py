#!/usr/bin/env python3
"""Bandwise + every-graph eval dashboard (user 2026-07-15). ONE self-contained HTML comparing
bakes (npz candidates via blend_lib numpy forward + shipped .bin via bake_verdict --per-pair-output)
AND the reference metrics ssim2 / cvvdp / butteraugli, across every held-out corpus.

2026-07-15 rework (user feedback):
  - ZOOMABLE inline SVG charts (not raster PNG), smaller scatter dots, dense scatters subsampled.
  - FIXED the scale bug: comparison plots (calibration overlay, candlestick) use rank-PERCENTILE
    so bakes on different pred scales (B is 0..100, npz candidates are z-space) are comparable —
    the old shared identity diagonal forced the axis and crushed the human range ("baseline
    invisible" / "nonphoto broken" / "last candlestick big" were all this one bug).
  - ADDED ssim2 / cvvdp / butteraugli as reference-metric pseudo-bakes (their own (score, MOS)).
  - ADDED a DATA-PROVENANCE section: train vs val vs held-out, and what is honest skill vs
    "cheat" (KADID/TID = train-overlap memorization; CID22/AIC/KonJND = true held-out).

  usage: bandwise_dashboard.py [--from-search <json>] [--bakes label:npz_or_bin,...] [--out x.html]
"""
import argparse
import io
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import blend_lib as B

REPO = Path.home() / "work/zen/zensim"
BV = str(REPO / "target/release/bake_verdict")
WEIGHTS = REPO / "zensim/weights"
REPORTS = Path("/mnt/v/output/zensim/reports/b_negatives")
REFMET = Path("/mnt/v/output/zensim/reports/refmetrics")
OUTDIR = Path("/mnt/v/output/zensim/dashboards")
plt.rcParams.update({"font.size": 7.5, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.facecolor": "white", "svg.fonttype": "none"})
BANDS = [f"B{i}" for i in range(10)]
SCATTER_MAX = 1500              # subsample dense scatters — keeps zoomable-SVG size responsive
# corpora bandable on a 0..100-normalized human MOS (JND-scale corpora are not)
BANDABLE = {"cid22", "kadid", "tid", "nonphoto"}

# --- data provenance: eval role + is-it-cheat, for the DIVERSE-trained primary candidate ---
# (bake_verdict convention: KADID/TID are train==val pair-overlap -> memorization, not skill.)
PROVENANCE = {
    "cid22":    ("TRUE HELD-OUT", "no — cid22_train is a disjoint ssim2-anchored subset; the 49-ref human-MOS set is sacred", "human MCOS/100", "HONEST — primary skill gate"),
    "kadid":    ("INTEGRITY GUARD", "YES — diverse bakes train on kadid ssim2_gpu (same images)", "human DMOS", "CHEAT — rewards memorization, watch for regressions only"),
    "tid":      ("INTEGRITY GUARD", "YES — diverse bakes train on tid ssim2_gpu (same images)", "human MOS", "CHEAT — rewards memorization, watch for regressions only"),
    "konjnd":   ("TRUE HELD-OUT", "no", "mean PJND threshold", "HONEST — HF/near-lossless (G5)"),
    "aic3":     ("TRUE HELD-OUT", "no", "JND units", "HONEST — compression JND"),
    "aic4":     ("TRUE HELD-OUT", "no", "JND units", "HONEST — compression JND"),
    "nonphoto": ("HELD-OUT (val split)", "diverse bakes train on imazen-26 TRAIN-origin {0,2,4,6,8}; this is VAL-origin {1,3,5} — no rendition leak", "ssim2/100", "HONEST generalization; same corpus family"),
}
TRAIN_INPUTS = [
    ("safesyn", "synthetic-safe tiles (CID22-leak-purged)", "ssim2_gpu", "196k (cap 90k)"),
    ("cid22_train", "CID22 ssim2-anchored subset — NOT the 49-ref MOS holdout", "ssim2_gpu", "~17k"),
    ("kadid", "KADID-10k (ssim2 FIXED §3.18)", "ssim2_gpu", "10,125"),
    ("tid", "TID2013 (ssim2 FIXED §3.18)", "ssim2_gpu", "3,000"),
    ("bigcodec (DIV)", "imazen-26 diverse real-codec, TRAIN-origin", "ssim2/100 (HQ>85 ×0.3)", "2.32M (cap 120k)"),
    ("kadis (NEG)", "KADIS-700k neg-rich (ssim2 negative tail)", "score_ssim2_gpu", "266k (cap 90k, w0.3)"),
]


def svg(fig):
    b = io.StringIO(); fig.savefig(b, format="svg", bbox_inches="tight"); plt.close(fig)
    s = b.getvalue()
    return s[s.index("<svg"):]                    # strip xml/doctype preamble -> inline vector


def _sub(n, k=SCATTER_MAX, seed=0):
    if n <= k:
        return np.arange(n)
    return np.random.RandomState(seed).permutation(n)[:k]


def pctrank(v):
    """rank -> [0,100] percentile; makes different pred scales directly comparable."""
    v = np.asarray(v, float); order = v.argsort(); r = np.empty(len(v)); r[order] = np.arange(len(v))
    return r / max(1, len(v) - 1) * 100.0


def score_npz(npz_path, corpus):
    p = dict(np.load(npz_path))
    X, h = B.load_val(corpus)
    return h, B.forward(p, X)


def score_bin(bin_path, corpus):
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tf:
        out = tf.name
    subprocess.run([BV, "--bake", str(bin_path), "--corpora", corpus, "--per-pair-output", out],
                   capture_output=True, text=True)
    try:
        arr = np.loadtxt(out, delimiter="\t", skiprows=1)
        arr = arr.reshape(1, -1) if arr.ndim == 1 else arr
        return arr[:, 0], arr[:, 1]
    except Exception:
        return None, None


def load_ref_metrics(corpus):
    """{metric_label: (human, pred)} for corpora with computed metric TSVs — MCOS + the metric read
    from the SAME file (zenmetrics passes MCOS through, so they're row-aligned by construction).
    butteraugli negated so higher=better uniformly. Currently CID22 (see refmetrics/)."""
    out = {}
    if corpus != "cid22":
        return out
    import csv

    def load(fn, key, human="MCOS", neg=False):
        p = REFMET / fn
        if not p.exists():
            return None
        rows = list(csv.DictReader(open(p), delimiter="\t"))
        if not rows or human not in rows[0]:
            return None
        name = key if key in rows[0] else next((c for c in rows[0] if key in c), None)  # exact or substring
        if name is None:
            return None
        def num(r, k):
            try:
                return float(r[k])
            except (ValueError, KeyError):
                return np.nan
        hh = np.array([num(r, human) for r in rows]); vv = np.array([num(r, name) for r in rows])
        m = np.isfinite(hh) & np.isfinite(vv)
        return (hh[m], (-vv[m] if neg else vv[m]))
    for lab, fn, key, neg in [("ssim2", "cid22_ssim2.tsv", "ssim2", False),
                              ("cvvdp", "cid22_cvvdp.tsv", "cvvdp", False),
                              ("butteraugli↓", "cid22_butter.tsv", "butteraugli_pnorm3", True)]:
        r = load(fn, key, neg=neg)
        if r is not None:
            out[lab] = r
    return out


def collect(bakes):
    """{corpus: {label: (human, pred, panel, kind)}} — bakes + reference metrics."""
    data = {}
    for corp, (_p, _y, sign, _pb) in B.VAL_CORPORA.items():
        band = corp in BANDABLE
        data[corp] = {}
        for label, kind, path in bakes:
            h, pr = score_npz(path, corp) if kind == "npz" else score_bin(path, corp)
            if h is None or pr is None or len(h) == 0:
                continue
            data[corp][label] = (h, pr, B.panel(pr, h, sign=sign, per_band=band), kind)
        for label, (h, pr) in load_ref_metrics(corp).items():
            data[corp][label] = (h, pr, B.panel(pr, h, sign=+1, per_band=band), "metric")
    return data


# ---------------------------------------------------------------- charts (SVG) ----------------
def fig_scatter(h, pr, label, corp):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    idx = _sub(len(h))
    ax.scatter(h[idx], pr[idx], s=2, alpha=0.28, lw=0, color="#2b6cb0")
    lo, hi = np.nanmin(h), np.nanmax(h)
    span = (hi - lo) or 1
    hn = (h - lo) / span * 100
    xs, ys = [], []
    for q in range(0, 100, 5):
        s = (hn >= q) & (hn < q + 5)
        if s.sum() > 2:
            xs.append(np.median(h[s])); ys.append(np.median(pr[s]))
    if xs:
        ax.plot(xs, ys, "-o", color="crimson", ms=2.4, lw=1.1)   # rank-median trend (the "baseline")
    ax.set_xlim(lo - 0.02 * span, hi + 0.02 * span)              # human range, NOT forced by a diagonal
    sr = B.panel(pr, h, per_band=False)["srocc"]
    ax.set_title(f"{label}\n{corp} (SROCC {sr:+.3f})", fontsize=6.5)
    ax.set_xlabel("human MOS"); ax.set_ylabel("pred")
    return svg(fig)


def fig_bands_grouped(perbake, corp):
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    labels = list(perbake); w = 0.8 / max(1, len(labels))
    for i, lab in enumerate(labels):
        vals = [perbake[lab][2].get("bands", {}).get(b, {}).get("srocc", np.nan) for b in BANDS]
        ax.bar(np.arange(10) + i * w, np.nan_to_num(vals), w, label=lab)
    ax.set_xticks(np.arange(10) + 0.4); ax.set_xticklabels(BANDS, fontsize=6)
    ax.set_ylabel("SROCC"); ax.set_title(f"{corp} — per-band SROCC (B0..B9, higher=better)", fontsize=8)
    ax.legend(fontsize=5, ncol=len(labels)); ax.axhline(0, color="k", lw=0.4)
    return svg(fig)


def fig_calibration(perbake, corp):
    """comparable across bakes: pred rank-percentile (x) vs mean human (y)."""
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    for lab, (h, pr, _pan, _k) in perbake.items():
        pc = pctrank(pr); order = pc.argsort(); pcs, hs = pc[order], h[order]
        edges = np.linspace(0, len(pr), 21).astype(int)
        xs = [pcs[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a]
        ys = [hs[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a]
        ax.plot(xs, ys, "-o", ms=2, lw=1, label=lab)
    ax.set_xlabel("pred percentile"); ax.set_ylabel("mean human MOS")
    ax.set_title(f"{corp} — calibration (rank-normalized)", fontsize=8); ax.legend(fontsize=5)
    return svg(fig)


def fig_residual(perbake, corp):
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    for lab, (h, pr, _pan, _k) in perbake.items():
        A = np.polyfit(pr, h, 1); fitv = np.polyval(A, pr); resid = h - fitv
        idx = _sub(len(h))
        ax.scatter(fitv[idx], resid[idx], s=2, alpha=0.2, lw=0, label=lab)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("fitted (affine pred→human)"); ax.set_ylabel("residual")
    ax.set_title(f"{corp} — residuals", fontsize=8); ax.legend(fontsize=5)
    return svg(fig)


def fig_candlestick(perbake, corp):
    """ONE chart: pred rank-percentile distribution per MOS band, grouped by bake — comparable
    (all on 0..100 percentile), so no bake's box is 'big' merely from a wider raw pred scale."""
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    labels = list(perbake); w = 0.8 / max(1, len(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    for i, lab in enumerate(labels):
        h, pr, _pan, _k = perbake[lab]
        pc = pctrank(pr); lo, hi = np.nanmin(h), np.nanmax(h); hn = (h - lo) / ((hi - lo) or 1) * 100
        data, pos = [], []
        for bi in range(10):
            s = (hn >= bi * 10) & (hn < bi * 10 + 10) if bi < 9 else (hn >= 90)
            if s.sum() > 2:
                data.append(pc[s]); pos.append(bi + i * w)
        if data:
            bp = ax.boxplot(data, positions=pos, widths=w * 0.9, showfliers=False, patch_artist=True)
            for box in bp["boxes"]:
                box.set(facecolor=colors[i], alpha=0.55, lw=0.4)
            for med in bp["medians"]:
                med.set(color="k", lw=0.7)
    ax.set_xticks(np.arange(10) + 0.4); ax.set_xticklabels(BANDS, fontsize=6)
    ax.set_xlabel("MOS band"); ax.set_ylabel("pred percentile")
    ax.set_title(f"{corp} — pred-distribution candlestick per band (rank %, comparable)", fontsize=8)
    ax.legend([plt.Rectangle((0, 0), 1, 1, fc=colors[i], alpha=0.55) for i in range(len(labels))],
              labels, fontsize=5, ncol=len(labels))
    return svg(fig)


def fig_heatmap(data, labels):
    corps = list(B.VAL_CORPORA)
    M = np.full((len(labels), len(corps)), np.nan)
    for j, c in enumerate(corps):
        for i, lab in enumerate(labels):
            if lab in data[c]:
                pan = data[c][lab][2]
                M[i, j] = pan["srocc_abs"] if B.VAL_CORPORA[c][2] < 0 else pan["srocc"]
    fig, ax = plt.subplots(figsize=(1.3 + 0.75 * len(corps), 1.0 + 0.42 * len(labels)))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(corps)))
    ax.set_xticklabels([f"{c}\n{PROVENANCE[c][0].split()[0]}" for c in corps], rotation=30, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(corps)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=6)
    ax.set_title("SROCC — bake × corpus  (|SROCC| for signed JND corpora; green=better)", fontsize=8.5)
    fig.colorbar(im, fraction=0.025)
    return svg(fig)


def fig_trade(data, labels):
    def sr(lab, c):
        if lab not in data[c]:
            return np.nan
        pan = data[c][lab][2]
        return pan["srocc_abs"] if B.VAL_CORPORA[c][2] < 0 else pan["srocc"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.3))
    for ax, (yc, yl) in [(a1, ("nonphoto", "non-photo SROCC")), (a2, ("konjnd", "KonJND |SROCC|"))]:
        for lab in labels:
            x, y = sr(lab, "cid22"), sr(lab, yc)
            if np.isfinite(x) and np.isfinite(y):
                ax.scatter(x, y, s=45)
                ax.annotate(lab, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("CID22 SROCC (true held-out)"); ax.set_ylabel(yl)
        ax.set_title(f"CID22 vs {yc}", fontsize=8)
    fig.suptitle("Operating-point trade map (upper-right = better on both)", fontsize=9)
    return svg(fig)


def fig_composite(data, labels):
    comps = []
    for lab in labels:
        res = {}
        for c in B.VAL_CORPORA:
            res[c] = dict(data[c][lab][2]) if lab in data[c] else {"srocc": np.nan, "srocc_abs": np.nan}
            res[c]["sign"] = B.VAL_CORPORA[c][2]
        comps.append(B.composite(res)[0])
    order = np.argsort(comps)[::-1]
    fig, ax = plt.subplots(figsize=(1.4 + 0.8 * len(labels), 2.9))
    ax.bar(range(len(labels)), [comps[i] for i in order], color="#3182bd")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels([labels[i] for i in order], rotation=35, ha="right", fontsize=6)
    ax.set_ylabel("composite"); ax.set_title("Goal-aware ranking (CID22+0.3·nonphoto+0.2·KonJND+…)", fontsize=7.5)
    return svg(fig)


# ---------------------------------------------------------------- dial ----------------
DIAL_GRID = "/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29_quarantined.parquet"
_dial = {}


def load_dial():
    if not _dial:
        import pyarrow.parquet as pq
        names = [f.name for f in pq.read_schema(DIAL_GRID)]
        pfx = "feat_" if "feat_0" in names else "f"
        t = pq.read_table(DIAL_GRID, columns=["image_id", "codec", "q"] + [f"{pfx}{i}" for i in range(B.N_FEAT)])
        _dial["meta"] = t.select(["image_id", "codec", "q"]).to_pandas()
        _dial["X"] = np.stack([np.asarray(t[f"{pfx}{i}"], dtype=np.float32) for i in range(B.N_FEAT)], 1)
    return _dial["meta"], _dial["X"]


def dial_mono(meta, pred):
    import pandas as pd
    df = meta.copy(); df["pred"] = pred; good = tot = 0
    for _k, g in df.groupby(["image_id", "codec"]):
        p = g.sort_values("q")["pred"].to_numpy()
        if len(p) < 2:
            continue
        d = np.diff(p); good += int((d >= 0).sum()); tot += len(d)
    return good / tot if tot else np.nan


def fig_dial(npz_bakes):
    import pandas as pd
    meta, X = load_dial()
    codecs = sorted(meta["codec"].unique())
    fig, axes = plt.subplots(1, len(codecs), figsize=(2.2 * len(codecs), 2.6))
    axes = np.atleast_1d(axes)
    monos = {}
    for lab, path in npz_bakes:
        pred = B.forward(dict(np.load(path)), X)
        monos[lab] = dial_mono(meta, pred)
        for ax, cod in zip(axes, codecs):
            sel = meta["codec"].to_numpy() == cod
            g = pd.DataFrame({"q": meta["q"].to_numpy()[sel], "p": pred[sel]}).groupby("q")["p"].median()
            ax.plot(g.index, g.values, "-o", ms=2, lw=1, label=lab)
    for ax, cod in zip(axes, codecs):
        ax.set_title(cod, fontsize=7); ax.set_xlabel("q"); ax.legend(fontsize=5)
    axes[0].set_ylabel("median pred (dial)")
    return svg(fig), monos


# ---------------------------------------------------------------- tables ----------------
def agg_stat_table(perbake, corp):
    sign = B.VAL_CORPORA[corp][2]
    head = ["SROCC" + (" |·|" if sign < 0 else ""), "PLCC", "KROCC", "Z-RMSE", "n", "kind"]
    rows = ["<table><thead><tr><th>bake / metric</th>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"]
    for lab, (_h, _p, pan, kind) in perbake.items():
        sr = pan["srocc_abs"] if sign < 0 else pan["srocc"]
        cls = " class='metric'" if kind == "metric" else ""
        vals = [sr, pan["plcc"], pan["krocc"], pan["zrmse"]]
        cells = "".join(f"<td>{v:.4f}</td>" if np.isfinite(v) else "<td>—</td>" for v in vals)
        rows.append(f"<tr{cls}><td class='lbl'>{lab}</td>{cells}<td>{pan['n']}</td><td>{kind}</td></tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


def band_table(perbake, corp):
    rows = ["<table><thead><tr><th>bake / metric</th>" + "".join(f"<th>{b}</th>" for b in BANDS)
            + "<th>agg</th><th>n</th></tr></thead><tbody>"]
    allv = {b: [] for b in BANDS + ["agg"]}
    for lab, (_h, _p, pan, _k) in perbake.items():
        for b in BANDS:
            v = pan.get("bands", {}).get(b, {}).get("srocc", np.nan)
            if np.isfinite(v):
                allv[b].append(v)
        agg = pan["srocc_abs"] if B.VAL_CORPORA[corp][2] < 0 else pan["srocc"]
        if np.isfinite(agg):
            allv["agg"].append(agg)

    def cell(v, col):
        if v is None or not np.isfinite(v):
            return '<td style="color:#999">—</td>'
        vv = allv[col]; lo, hi = (min(vv), max(vv)) if vv else (0, 1)
        t = 0 if hi == lo else max(0, min(1, (v - lo) / (hi - lo)))
        r = int(200 * (1 - t) + 40 * t); g = int(60 * (1 - t) + 160 * t)
        return f'<td style="background:rgb({r},{g},60);color:#fff">{v:.3f}</td>'
    for lab, (_h, _p, pan, kind) in perbake.items():
        agg = pan["srocc_abs"] if B.VAL_CORPORA[corp][2] < 0 else pan["srocc"]
        cls = " class='metric'" if kind == "metric" else ""
        cells = "".join(cell(pan.get("bands", {}).get(b, {}).get("srocc"), b) for b in BANDS)
        rows.append(f"<tr{cls}><td class='lbl'>{lab}</td>{cells}{cell(agg,'agg')}<td>{pan['n']}</td></tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


def provenance_section(bakes):
    b = ["<h2 id='data'>Datasets &amp; provenance — train vs held-out vs \"cheat\"</h2>",
         "<p class='sub'>What the model learned from vs what honestly tests it. <b>Provenance is for "
         "the DIVERSE-trained bakes</b> (2L / §8.35 / ep700 — they train on safesyn+cid22_train+kadid+tid"
         "+bigcodec+kadis). ssim2-only trains on safesyn+cid22_train only, so KADID/TID are honest held-out "
         "for it. Reference metrics (ssim2/cvvdp/butteraugli) train on nothing.</p>"]
    b.append("<h3>Training inputs (what the diverse bakes fit)</h3><table><thead><tr>"
             "<th>corpus</th><th>what</th><th>target</th><th>rows</th></tr></thead><tbody>")
    for name, what, tgt, rows in TRAIN_INPUTS:
        b.append(f"<tr><td class='lbl'>{name}</td><td>{what}</td><td>{tgt}</td><td>{rows}</td></tr>")
    b.append("</tbody></table>")
    b.append("<h3>Evaluation corpora (what the SROCC numbers mean)</h3><table><thead><tr>"
             "<th>corpus</th><th>eval role</th><th>in diverse-bake training?</th><th>human label</th>"
             "<th>honesty</th></tr></thead><tbody>")
    for c, (role, intr, lab, hon) in PROVENANCE.items():
        cheat = "CHEAT" in hon
        badge = ("#c0392b" if cheat else ("#e67e22" if "val split" in role else "#27ae60"))
        b.append(f"<tr><td class='lbl'>{c}</td><td><span class='badge' style='background:{badge}'>{role}</span></td>"
                 f"<td>{intr}</td><td>{lab}</td><td>{hon}</td></tr>")
    b.append("</tbody></table>")
    b.append("<p class='sub'><b style='color:#27ae60'>■ TRUE HELD-OUT</b> = honest skill (CID22 is the "
             "primary gate; AIC-3/AIC-4/KonJND compression/JND). <b style='color:#c0392b'>■ CHEAT / integrity "
             "guard</b> = KADID/TID images are in training (as ssim2 targets); their human-MOS SROCC rewards "
             "MEMORIZATION, not skill — watch only for regressions, never rank on them. "
             "<b style='color:#e67e22'>■ HELD-OUT val split</b> = non-photo is imazen-26 VAL-origin {1,3,5}, "
             "disjoint renditions from the TRAIN-origin the bakes fit (honest generalization, same corpus family).</p>")
    return "".join(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-search", default="/mnt/v/output/zensim/reports/blend/blend_results_r3_2026-07-15.json")
    ap.add_argument("--bakes", default=None)
    ap.add_argument("--out", default=str(OUTDIR / "bandwise_dashboard_2026-07-15.html"))
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    bakes = []
    for lab, p in [("B(shipped)", WEIGHTS / "b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin"),
                   ("§8.35 diverse", REPORTS / "mlp_diverse_depoison_dv0.5kw0.3_2026-07-15.bin")]:
        if Path(p).exists():
            bakes.append((lab, "bin", str(p)))
    if a.bakes:
        for tok in a.bakes.split(","):
            lab, pth = tok.split(":", 1)
            bakes.append((lab, "npz" if pth.endswith(".npz") else "bin", pth))
    elif Path(a.from_search).exists():
        for s in json.loads(Path(a.from_search).read_text()).get("saved", [])[:5]:
            bakes.append((s["label"], "npz", s["npz"]))

    print(f"scoring {len(bakes)} bakes + ref metrics × {len(B.VAL_CORPORA)} corpora ...")
    data = collect(bakes)
    # labels in a stable order: bakes first, then any metric labels present anywhere
    labels = [lab for lab, *_ in bakes]
    for c in data:
        for lab in data[c]:
            if lab not in labels:
                labels.append(lab)

    body = ["<h1>zensim bandwise dashboard — bakes + ssim2/cvvdp/butteraugli × corpus × band</h1>",
            "<p class='sub'>Zoomable SVG. Comparison plots use rank-<b>percentile</b> so bakes on "
            "different pred scales (B=0..100, MLP candidates=z-space, metrics=native) are comparable. "
            "By <code>scripts/v_next/bandwise_dashboard.py</code>.</p>"]
    body.append(provenance_section(bakes))
    body.append("<h2 id='overview'>Cross-corpus overview</h2>")
    body.append(fig_heatmap(data, labels))
    body.append("<div class='row'>" + fig_trade(data, labels) + fig_composite(data, labels) + "</div>")
    npz_bakes = [(lab, path) for lab, kind, path in bakes if kind == "npz"]
    if npz_bakes:
        try:
            dimg, monos = fig_dial(npz_bakes)
            body.append("<h2 id='dial'>Codec-target dial (score vs q — npz candidates)</h2>")
            body.append("<p class='sub'>monotonicity (non-decreasing adjacent-q steps): "
                        + " · ".join(f"<b>{k}</b> {v*100:.1f}%" for k, v in monos.items()) + "</p>")
            body.append(dimg)
        except Exception as e:
            body.append(f"<p class='sub'>dial skipped: {e}</p>")
    body.append("<p class='toc'>corpora: " + " · ".join(f"<a href='#{c}'>{c}</a>" for c in B.VAL_CORPORA) + "</p>")

    for corp in B.VAL_CORPORA:
        perbake = data[corp]
        if not perbake:
            continue
        role = PROVENANCE[corp][0]; cheat = "CHEAT" in PROVENANCE[corp][3]
        color = "#c0392b" if cheat else ("#e67e22" if "val split" in role else "#27ae60")
        body.append(f"<h2 id='{corp}'>{corp} <span class='badge' style='background:{color}'>{role}</span></h2>")
        body.append(f"<p class='sub'>{PROVENANCE[corp][3]}</p>")
        body.append(agg_stat_table(perbake, corp))
        if corp in BANDABLE:
            body.append(band_table(perbake, corp))
            body.append(fig_bands_grouped(perbake, corp))
            body.append(fig_candlestick(perbake, corp))
        body.append("<div class='row'>" + fig_calibration(perbake, corp) + fig_residual(perbake, corp) + "</div>")
        body.append("<div class='row'>" + "".join(
            fig_scatter(h, pr, lab, corp) for lab, (h, pr, _pan, _k) in perbake.items()) + "</div>")

    html = ("<style>body{font:13px system-ui,sans-serif;margin:1.3rem;background:#fff;color:#111}"
            "h1{font-size:1.3rem}h2{font-size:1.05rem;margin-top:1.6rem;border-top:1px solid #ccc;padding-top:.5rem}"
            "h3{font-size:.92rem;margin:.6rem 0 .2rem}.sub,.toc{color:#555;max-width:74rem}"
            ".row{display:flex;flex-wrap:wrap;gap:6px;align-items:flex-start}svg{max-width:100%;height:auto;border:1px solid #eee}"
            "table{border-collapse:collapse;margin:.4rem 0;font-size:11px}th,td{border:1px solid #ddd;padding:2px 6px;text-align:center}"
            "td.lbl{text-align:left;font-weight:600;background:#f4f4f4}tr.metric td{font-style:italic;background:#eef6ff}"
            "tr.metric td.lbl{background:#dbeafe}.badge{color:#fff;padding:.05rem .4rem;border-radius:.2rem;font-size:.7rem;font-weight:600}"
            "code{background:#f0f0f0;padding:.1rem .3rem}a{color:#06c}</style>" + "".join(body))
    Path(a.out).write_text(html)
    print(f"wrote {a.out}  ({len(html)//1024} KB)\n  view: "
          + a.out.replace("/mnt/v/output/", "http://172.23.240.1:3300/"))


if __name__ == "__main__":
    main()
