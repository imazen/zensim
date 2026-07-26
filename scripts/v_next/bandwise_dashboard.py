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
BANDABLE = {"cid22", "kadid", "tid", "nonphoto", "csiq", "live"}

# --- eval-corpus metadata: split-suffixed display, split semantics, cheat-twin, footnote. The
# per-bake honesty is DERIVED from each bake's <path>.spec.json (train_corpora) at render time —
# NOT hardcoded — so the dashboard can't desync from what a bake actually trained on. ---
CORPUS_META = {
    "cid22":    ("cid22_val", "49-ref human-MOS holdout", None,
                 "Held-out human MOS. cid22_train (ssim2-anchored, DIFFERENT images) may be in a bake's "
                 "training — it is disjoint from these 49 refs, so this is NEVER cheat."),
    "kadid":    ("kadid_val", "KADID-10k — val images == train images", "kadid",
                 "TRAIN==VAL image overlap. Honest ONLY for bakes that did NOT train on kadid; "
                 "CHEAT (memorization) for bakes that did (see per-bake honesty matrix)."),
    "tid":      ("tid_val", "TID2013 — val images == train images", "tid",
                 "TRAIN==VAL image overlap. Honest only for bakes that didn't train on tid; CHEAT for those that did."),
    "konjnd":   ("konjnd_val", "KonJND-1k PJND — HF holdout", "konjnd",
                 "CHEAT for bakes that train on konjnd/konjnd_dense (shares the 1008 source refs); "
                 "HELD-OUT otherwise. HF/near-lossless — the G5 weak axis."),
    "aic3":     ("aic3_val", "AIC-3 CTC JND — holdout", None, "Never trained on. Compression JND."),
    "aic4":     ("aic4_val", "AIC-4 JND — holdout", None, "Never trained on. Compression JND."),
    "nonphoto": ("imazen26_val", "imazen-26 non-photo — val-origin {1,3,5}", "bigcodec",
                 "VAL-SPLIT of the bigcodec/imazen-26 corpus: diverse bakes train on TRAIN-origin "
                 "{0,2,4,6,8} (disjoint renditions) → honest generalization, NOT cheat."),
    # FR-corpus expansion 2026-07-18 — all HELD-OUT (our bakes never train on them).
    "live":     ("live_val", "LIVE-R2 — 29-ref FR (JPEG/JP2K/blur/WN/fastfading)", None,
                 "Held-out FR. Sheikh 2006 realigned DMOS → human=1−dmos/100. THE classic "
                 "compression benchmark; never trained on."),
    "csiq":     ("csiq_val", "CSIQ — 30-ref FR (JPEG/JP2K/blur/noise/contrast)", None,
                 "Held-out FR. human=1−DMOS. Classic compression + analytic distortions; never trained on."),
    "pipal":    ("pipal_val", "PIPAL — 200-ref GAN/restoration (ELO MOS)", None,
                 "Held-out. GAN/restoration distortions — a DISTINCT, harder axis (NOT compression); "
                 "never trained on. Expect lower SROCC than the FR compression sets."),
}
TRAIN_CORPUS_DESC = {
    "safesyn": "synthetic-safe tiles (CID22-leak-purged), ssim2_gpu",
    "cid22_train": "CID22 ssim2-anchored subset — NOT the 49-ref MOS holdout",
    "kadid": "KADID-10k (ssim2 FIXED §3.18) — overlaps kadid_val",
    "tid": "TID2013 (ssim2 FIXED §3.18) — overlaps tid_val",
    "bigcodec": "imazen-26 diverse real-codec, TRAIN-origin {0,2,4,6,8}",
    "kadis": "KADIS-700k neg-rich (ssim2 negative tail)",
    "hdr_v3mix": "HDR cvvdp-mix head corpus (7,410 rows) — B's cid head",
}


def bake_train(path):
    """training corpora for a bake, from its <path>.spec.json sidecar (the desync-proof source of
    truth). Returns None if no sidecar (dashboard then shows 'unknown', never a guess)."""
    sp = Path(str(path) + ".spec.json")
    if not sp.exists():
        return None
    try:
        return set(json.loads(sp.read_text()).get("train_corpora", []))
    except Exception:
        return None


# Reference-metric tuning provenance (publicly documented — NOT "trained on nothing").
# `train` = corpora whose EVAL images the metric was fit on → CHEAT there, same rule as bakes.
# Sources: SSIMULACRA2 README (Nelder-Mead-tuned on CID22 201/250 refs + TID2013 + KADID-10k +
# KonFiG-IQA) — so KADID/TID are in-sample, while the CID22-49 val is the held-out remainder and
# is therefore FAIR for ssim2 too (docs/DATA_SPLITS.md §"SSIMULACRA2's own data usage"). cvvdp /
# butteraugli are fit on external psychophysical / proprietary data (none of these corpora);
# IW-SSIM is analytical (no learned parameters).
METRIC_PROVENANCE = {
    "ssim2": ({"kadid", "tid"},
              "SSIMULACRA2 — Nelder-Mead-tuned on CID22 (201/250 refs) + TID2013 + KADID-10k + "
              "KonFiG-IQA. KADID/TID are in-sample (CHEAT); the CID22-49 val is the held-out "
              "remainder (FAIR). Never scoreboard ssim2 on KADID/TID."),
    "cvvdp": (set(),
              "ColorVideoVDP (Mantiuk et al.) — calibrated on psychophysical contrast / JOD "
              "datasets; none of these corpora → held-out everywhere here."),
    "butteraugli↓": (set(),
                     "butteraugli (Google) — tuned on internal data; none of these corpora → "
                     "held-out everywhere here."),
    "iwssim": (set(),
               "IW-SSIM (Wang & Li 2011) — analytical information-content weighting, no learned "
               "parameters → held-out everywhere here."),
}


def _trained(train_set, key):
    """True if the bake trained on `key` — tolerant of raw manifest group names
    (e.g. 'konjnd_dense' satisfies key 'konjnd', 'bigcodec' the nonphoto twin)."""
    return any(t == key or t.startswith(key) for t in train_set)


def honesty(train_set, corpus):
    """(label, color) — per-bake, per-corpus honesty from the ACTUAL train set."""
    if train_set is None:
        return "unknown", "#888"
    twin = CORPUS_META[corpus][2]
    if corpus in ("kadid", "tid", "konjnd") and twin and _trained(train_set, twin):
        return "CHEAT", "#c0392b"
    if corpus == "nonphoto" and twin and _trained(train_set, twin):
        return "val-split", "#e67e22"
    return "HELD-OUT", "#27ae60"


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


# Reference-metric TSVs per corpus: (human_col, {label: (filename, column_key, negate)}).
# human_col is passed through by `zenmetrics batch` from the pairs TSV, so human + metric are
# row-aligned by construction (never joined). butteraugli is negated so higher=better uniformly.
# CID22 = CPU run 2026-07-15; KADID/TID = GPU run (ssim2-gpu / butteraugli-gpu / cvvdp-gpu).
# metric spec: (filename, column_key, negate[, human_col_override]). The 4th element
# overrides the corpus human column when a metric TSV carries its own (the iwssim TSVs use
# "human_score", not the corpus DMOS/MOS). iwssim range [0,1], 1=identical → higher=better.
REF_METRIC_FILES = {
    "cid22": ("MCOS", {
        "ssim2": ("cid22_ssim2.tsv", "ssim2", False),
        "cvvdp": ("cid22_cvvdp.tsv", "cvvdp", False),
        "butteraugli↓": ("cid22_butter.tsv", "butteraugli_pnorm3", True),
        "iwssim": ("cid22_iwssim.tsv", "iwssim_gpu", False),
    }),
    "kadid": ("DMOS", {
        "ssim2": ("kadid_ssim2_gpu.tsv", "ssim2_gpu", False),
        "cvvdp": ("kadid_cvvdp_gpu.tsv", "cvvdp", False),
        "butteraugli↓": ("kadid_butteraugli_gpu.tsv", "butteraugli_pnorm3_gpu", True),
        "iwssim": ("kadid_iwssim.tsv", "iwssim_imazen_v0_0_1", False, "human_score"),
    }),
    "tid": ("MOS", {
        "ssim2": ("tid_ssim2_gpu.tsv", "ssim2_gpu", False),
        "cvvdp": ("tid_cvvdp_gpu.tsv", "cvvdp", False),
        "butteraugli↓": ("tid_butteraugli_gpu.tsv", "butteraugli_pnorm3_gpu", True),
        "iwssim": ("tid_iwssim.tsv", "iwssim_imazen_v0_0_1", False, "human_score"),
    }),
    # held-out corpora: scored fresh on the val-consistent pairs (zenmetrics *-gpu),
    # human passed through from the pairs TSV (konjnd=pjnd, aic3=jnd). butteraugli negated.
    "konjnd": ("pjnd", {
        "ssim2": ("konjnd_ssim2_heldout.tsv", "ssim2", False),
        "cvvdp": ("konjnd_cvvdp_heldout.tsv", "cvvdp", False),
        "butteraugli↓": ("konjnd_butteraugli_heldout.tsv", "butter", True),
        "iwssim": ("konjnd_iwssim_heldout.tsv", "iwssim", False),
    }),
    "aic3": ("jnd", {
        "ssim2": ("aic3_ssim2_heldout.tsv", "ssim2", False),
        "cvvdp": ("aic3_cvvdp_heldout.tsv", "cvvdp", False),
        "butteraugli↓": ("aic3_butteraugli_heldout.tsv", "butter", True),
        "iwssim": ("aic3_iwssim_heldout.tsv", "iwssim", False),
    }),
}


def load_ref_metrics(corpus):
    """{metric_label: (human, pred)} for corpora with computed metric TSVs — the human score and the
    metric are read from the SAME file (zenmetrics passes the human column through), so they are
    row-aligned by construction. Missing files are skipped silently, so a corpus whose metrics
    haven't been computed yet simply shows no reference rows."""
    spec = REF_METRIC_FILES.get(corpus)
    if not spec:
        return {}
    human, files = spec
    import csv

    def load(fn, key, neg=False, human_col=None):
        hcol = human_col or human
        p = REFMET / fn
        if not p.exists():
            return None
        rows = list(csv.DictReader(open(p), delimiter="\t"))
        if not rows or hcol not in rows[0]:
            return None
        name = key if key in rows[0] else next((c for c in rows[0] if key in c), None)  # exact or substring
        if name is None:
            return None
        def num(r, k):
            try:
                return float(r[k])
            except (ValueError, KeyError):
                return np.nan
        hh = np.array([num(r, hcol) for r in rows]); vv = np.array([num(r, name) for r in rows])
        m = np.isfinite(hh) & np.isfinite(vv)
        return (hh[m], (-vv[m] if neg else vv[m]))
    out = {}
    for lab, spec in files.items():
        fn, key, neg = spec[0], spec[1], spec[2]
        hcol = spec[3] if len(spec) > 3 else None
        r = load(fn, key, neg=neg, human_col=hcol)
        if r is not None:
            # konjnd pairs carry raw `pjnd`, whose polarity is inverted vs the val
            # parquet's stored human (the bakes score positive on konjnd via sign=-1);
            # flip the metric human so metric + bake SROCC share one polarity.
            if corpus == "konjnd":
                r = (-r[0], r[1])
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
            # metrics are quality predictors like the bakes → sign identically per corpus
            # (konjnd/aic3/aic4 are sign=-1; hardcoding +1 flipped their metric SROCC)
            data[corp][label] = (h, pr, B.panel(pr, h, sign=sign, per_band=band), "metric")
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
    ax.set_xticklabels([CORPUS_META[c][0] for c in corps], rotation=30, ha="right", fontsize=6.5)
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
def agg_stat_table(perbake, corp, train):
    sign = B.VAL_CORPORA[corp][2]
    head = ["SROCC" + (" |·|" if sign < 0 else ""), "PLCC", "KROCC", "Z-RMSE",
            "SROCC low-tail", "SROCC high-tail", "n", "honesty"]
    rows = ["<table><thead><tr><th>bake / metric</th>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"]
    for lab, (_h, _p, pan, kind) in perbake.items():
        sr = pan["srocc_abs"] if sign < 0 else pan["srocc"]
        cls = " class='metric'" if kind == "metric" else ""
        lt, ht = pan.get("srocc_lowtail", np.nan), pan.get("srocc_hightail", np.nan)
        if sign < 0:
            lt, ht = abs(lt) if np.isfinite(lt) else lt, abs(ht) if np.isfinite(ht) else ht
        vals = [sr, pan["plcc"], pan["krocc"], pan["zrmse"], lt, ht]
        cells = "".join(f"<td>{v:.4f}</td>" if np.isfinite(v) else "<td>—</td>" for v in vals)
        hn, col = honesty(train.get(lab), corp)
        rows.append(f"<tr{cls}><td class='lbl'>{lab}</td>{cells}<td>{pan['n']}</td>"
                    f"<td style='background:{col};color:#fff;font-size:9px'>{hn}</td></tr>")
    rows.append("</tbody></table>")
    rows.append(
        "<p class='sub'><b>low-tail / high-tail SROCC</b> = rank skill on the worst / best 30% by human score. "
        "These are the <b>honest extreme-quality numbers</b>: unlike the width-10 B0/B9 bands below they are "
        "<b>not range-restricted</b>, so they measure real rank skill at each end instead of noise. "
        "<b>Read them against the ssim2 row, not against 1.0</b> — the high-tail is intrinsically harder than "
        "the low-tail for <i>every</i> metric (on CID22, ssim2's own tails are +0.649 / +0.463), because "
        "ranking stimuli humans scored near the top means subtler differences and noisier MOS. A bake whose "
        "high-tail tracks ssim2's is at the corpus's difficulty floor, not defective. Note the high-tail is "
        "<b>not</b> the near-lossless regime: CID22's top 30% by MOS sits at ssim2 ≈75–88, with 0% above 95.</p>")
    return "".join(rows)


def band_table(perbake, corp, train):
    rows = ["<table><thead><tr><th>bake / metric</th>" + "".join(f"<th>{b}</th>" for b in BANDS)
            + "<th>agg</th><th>n</th><th>honesty</th></tr></thead><tbody>"]
    allv = {b: [] for b in BANDS + ["agg"]}
    for lab, (_h, _p, pan, _k) in perbake.items():
        for b in BANDS:
            v = pan.get("bands", {}).get(b, {}).get("srocc", np.nan)
            if np.isfinite(v):
                allv[b].append(v)
        agg = pan["srocc_abs"] if B.VAL_CORPORA[corp][2] < 0 else pan["srocc"]
        if np.isfinite(agg):
            allv["agg"].append(agg)

    # Per-band n / target-spread are corpus properties (identical across bakes) -> one context row.
    any_pan = next((p for _h, _p, p, _k in perbake.values() if p.get("bands")), None)
    ctx = ""
    if any_pan:
        nrow = "".join(
            f"<td style='font-size:9px;color:#666'>n={any_pan['bands'][b]['n']}<br>"
            f"σ={any_pan['bands'][b]['restrict']:.2f}</td>" for b in BANDS)
        ctx = (f"<tr><td class='lbl' style='font-size:9px;color:#666'>band context</td>{nrow}"
               f"<td colspan='3' style='font-size:9px;color:#666'>σ = within-band target spread<br>"
               f"as fraction of full range</td></tr>")

    def cell(v, col, pan=None):
        if v is None or not np.isfinite(v):
            return '<td style="color:#999">—</td>'
        vv = allv[col]; lo, hi = (min(vv), max(vv)) if vv else (0, 1)
        t = 0 if hi == lo else max(0, min(1, (v - lo) / (hi - lo)))
        r = int(200 * (1 - t) + 40 * t); g = int(60 * (1 - t) + 160 * t)
        # Dim + mark bands whose SROCC is untrustworthy (n<30 => CI wider than +-0.3, sign is noise).
        lc = bool(pan and col in pan.get("bands", {}) and pan["bands"][col].get("lowconf"))
        if lc:
            return (f'<td style="background:rgb({r},{g},60);color:#fff;opacity:.45" '
                    f'title="LOW CONFIDENCE: n&lt;30, SROCC CI wider than ±0.3 — sign is noise, not signal">'
                    f'{v:.3f}<sup>?</sup></td>')
        return f'<td style="background:rgb({r},{g},60);color:#fff">{v:.3f}</td>'
    for lab, (_h, _p, pan, kind) in perbake.items():
        agg = pan["srocc_abs"] if B.VAL_CORPORA[corp][2] < 0 else pan["srocc"]
        cls = " class='metric'" if kind == "metric" else ""
        cells = "".join(cell(pan.get("bands", {}).get(b, {}).get("srocc"), b, pan) for b in BANDS)
        hn, col = honesty(train.get(lab), corp)
        rows.append(f"<tr{cls}><td class='lbl'>{lab}</td>{cells}{cell(agg,'agg')}<td>{pan['n']}</td>"
                    f"<td style='background:{col};color:#fff;font-size:9px'>{hn}</td></tr>")
    rows.insert(1, ctx)
    rows.append("</tbody></table>")
    rows.append(
        "<p class='sub'><b>Why per-band SROCC is far below the aggregate — and why B0/B9 can go negative.</b> "
        "Bands are cut on the <i>human</i> score, so within a band the target barely varies (σ≈0.13 of full range "
        "on CID22). Rank correlation under that <b>range restriction</b> is mostly noise: a <i>near-perfect</i> model "
        "(pred = human + 2% noise, aggregate SROCC 0.994) still scores only <b>0.77–0.89</b> per band. So low "
        "per-band values are a property of the binning, not a training failure — and at the extremes (B0: smallest n; "
        "B9: smallest σ, near-lossless piling against the MOS ceiling) the noise routinely crosses zero into "
        "<b>negative</b>. Cells marked <sup>?</sup> have n&lt;30 (CI wider than ±0.3 — the sign is noise). "
        "For real extreme-quality skill read <b>low-tail / high-tail SROCC</b> above, which are not range-restricted.</p>")
    return "".join(rows)


def provenance_section(bakes):
    """DERIVED entirely from each bake's <path>.spec.json — cannot desync from the bakes."""
    train = {lab: bake_train(path) for lab, _k, path in bakes}
    labels = [lab for lab, *_ in bakes]
    # reference metrics carry DOCUMENTED tuning provenance (not "trained on nothing")
    metric_labels = list(METRIC_PROVENANCE.keys())
    for m in metric_labels:
        train[m] = METRIC_PROVENANCE[m][0]
    all_labels = labels + metric_labels
    b = ["<h2 id='data'>Datasets &amp; provenance — bakes from <code>spec.json</code>, metrics from published tuning sets</h2>",
         "<p class='sub'>Honesty is computed <b>per series</b> from what it ACTUALLY trained/tuned on. Bakes: their "
         "<code>&lt;bake&gt;.spec.json</code> sidecar. Reference metrics: their <b>publicly documented</b> tuning "
         "corpora — <b>not</b> \"trained on nothing\". SSIMULACRA2 was Nelder-Mead-tuned on CID22-201 + TID2013 + "
         "KADID-10k + KonFiG, so <b>ssim2 is CHEAT (in-sample) on KADID/TID</b> and must never be scoreboarded there; "
         "the CID22-49 val is the held-out remainder so it is FAIR for ssim2. cvvdp / butteraugli / iwssim were fit "
         "on external or analytical data disjoint from these corpora → held-out here.</p>"]
    # 1) what each bake trained on
    b.append("<h3>What each bake trained on (from its sidecar)</h3><table><thead><tr>"
             "<th>bake</th><th>arch</th><th>train corpora</th></tr></thead><tbody>")
    for lab, _k, path in bakes:
        sp = Path(str(path) + ".spec.json")
        meta = json.loads(sp.read_text()) if sp.exists() else {}
        tc = ", ".join(sorted(meta.get("train_corpora", []))) or "<i>unknown (no sidecar)</i>"
        b.append(f"<tr><td class='lbl'>{lab}</td><td>{meta.get('arch','?')}</td><td>{tc}</td></tr>")
    b.append("</tbody></table>")
    # 1b) reference-metric tuning provenance (publicly documented)
    b.append("<h3>Reference-metric tuning provenance (published, not \"trained on nothing\")</h3>"
             "<table><thead><tr><th>metric</th><th>tuned/fit on (→ CHEAT there)</th><th>source</th>"
             "</tr></thead><tbody>")
    for m in metric_labels:
        tc = ", ".join(sorted(METRIC_PROVENANCE[m][0])) or "<i>none of these corpora</i>"
        b.append(f"<tr class='metric'><td class='lbl'>{m}</td><td>{tc}</td><td>{METRIC_PROVENANCE[m][1]}</td></tr>")
    b.append("</tbody></table>")
    # 2) honesty matrix: corpus × (bake + metric)
    b.append("<h3>Honesty matrix — honest held-out vs CHEAT, per (corpus, series)</h3>"
             "<p class='sub'>Metric columns use each metric's documented tuning set — ssim2 reads CHEAT on "
             "KADID/TID (in-sample) and HELD-OUT on the CID22-49 remainder.</p>")
    b.append("<table><thead><tr><th>eval corpus</th>" + "".join(f"<th>{l}</th>" for l in all_labels)
             + "</tr></thead><tbody>")
    for c in B.VAL_CORPORA:
        cells = ""
        for lab in all_labels:
            hn, col = honesty(train[lab], c)
            cells += f"<td style='background:{col};color:#fff;font-weight:600'>{hn}</td>"
        b.append(f"<tr><td class='lbl'>{CORPUS_META[c][0]}</td>{cells}</tr>")
    b.append("</tbody></table>")
    # 3) training corpora actually used
    used = sorted(set().union(*[t for t in train.values() if t]) if any(train.values()) else set())
    if used:
        b.append("<h3>Training corpora used (across these bakes)</h3><table><thead><tr>"
                 "<th>corpus</th><th>what</th></tr></thead><tbody>")
        for tc in used:
            b.append(f"<tr><td class='lbl'>{tc}</td><td>{TRAIN_CORPUS_DESC.get(tc, '?')}</td></tr>")
        b.append("</tbody></table>")
    # 4) split footnotes
    b.append("<h3>Split footnotes</h3><ol class='sub' style='margin-top:.2rem'>")
    for c in B.VAL_CORPORA:
        disp, split, _twin, foot = CORPUS_META[c]
        b.append(f"<li id='fn-{c}'><b>{disp}</b> — {split}. {foot}</li>")
    b.append("</ol>")
    b.append("<p class='sub'><b style='color:#27ae60'>■ HELD-OUT</b> honest skill · "
             "<b style='color:#c0392b'>■ CHEAT</b> train==val image overlap (memorization — never rank on it) · "
             "<b style='color:#e67e22'>■ val-split</b> disjoint-rendition generalization · "
             "<b style='color:#888'>■ unknown</b> no spec.json sidecar.</p>")
    return "".join(b)


def robustness_section(bakes):
    """Closed-loop / robustness factors READ from the persisted <bake>.metrics.json sidecars
    (emit_bake_metrics.py) — NOT recomputed here. OOD stability, corruption gate, and the TWO
    distinct closed-loop-diffmap properties: `additive` (all-identity layers → diffmap is the
    EXACT per-feature spatial gradient; an MLP is NOT additive) and `basic_input_only` (reads
    only the spatializable basic block f0..155). A bake with no sidecar is flagged, never
    silently dropped (the anti-amnesia contract)."""
    import json as _json
    rows = []
    for lab, kind, path in bakes:
        if kind != "bin":
            continue
        mp = Path(str(path) + ".metrics.json")
        if not mp.exists():
            rows.append((lab, None))
            continue
        try:
            rows.append((lab, _json.load(open(mp))))
        except Exception:
            rows.append((lab, None))
    if not rows:
        return ""
    b = ["<h2 id='robust'>Closed-loop &amp; robustness factors <span class='sub' style='font-weight:400'>"
         "(read from <code>&lt;bake&gt;.metrics.json</code> sidecars — persisted, not recomputed)</span></h2>",
         "<table><thead><tr><th>bake</th><th>OOD |raw| max<br><span class='note'>lower=safer</span></th>"
         "<th>corruption gate<br><span class='note'>separate concern</span></th>"
         "<th>additive<br><span class='note'>exact-gradient diffmap</span></th>"
         "<th>basic-input<br><span class='note'>spatializable f0..155 only</span></th>"
         "<th>dial-mono</th><th>sha256</th><th>eval</th></tr></thead><tbody>"]
    for lab, m in rows:
        if m is None:
            b.append(f"<tr><td class='lbl'>{lab}</td><td colspan='7' style='background:#fadbd8'>"
                     "NO metrics.json sidecar — run emit_bake_metrics.py</td></tr>")
            continue
        e = m["eval"]
        ood = e["ood_max_abs_raw"]
        ood_c = "#fadbd8" if ood > 1000 else "#d5f5e3"
        # closed_loop {additive, basic_input_only, diffmap_basic_fraction}; tolerate old schema.
        cl = e.get("closed_loop") or {}
        add = cl.get("additive")
        bin_ = cl.get("basic_input_only")
        frac = cl.get("diffmap_basic_fraction")
        add_txt = {True: "✓ yes", False: "✗ MLP", None: "?"}[add]
        add_c = "#d5f5e3" if add is True else ("#fff3cd" if add is False else "")
        # for additive bakes, annotate the basic-mass fraction; for MLPs show input-scope only
        bin_txt = ("✓ 156-only" if bin_ else "372-in") + (f" · frac {frac}" if (add and frac is not None) else "")
        bin_c = "#d5f5e3" if bin_ else ""
        b.append(f"<tr><td class='lbl'>{lab}</td>"
                 f"<td style='background:{ood_c}'>{ood:,.0f}</td>"
                 f"<td>{e['corruption_gate_q20']*100:.0f}%</td>"
                 f"<td style='background:{add_c}'>{add_txt}</td>"
                 f"<td style='background:{bin_c}'>{bin_txt}</td>"
                 f"<td>{e['dial']['monotonicity']:.3f}</td>"
                 f"<td><code>{m['bake_sha256'][:12]}</code></td>"
                 f"<td class='note'>{m['tool']['timestamp'][:10]}</td></tr>")
    b.append("</tbody></table>")
    return "".join(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-search", default="/mnt/v/output/zensim/reports/blend/blend_results_r3_2026-07-15.json")
    ap.add_argument("--bakes", default=None)
    # NEW (2026-07-26): interactive summer-gauntlet mode. Reads pre-computed per-bake
    # *.fulleval.json (schema: make_stub_fulleval.py) and emits ONE self-contained, OFFLINE
    # HTML with bake-toggle checkboxes, a sortable scoreboard, and the predicted-vs-reference
    # correlation scatter matrix (MOS/JND/ssim2/butteraugli/cvvdp). See gauntlet.py.
    ap.add_argument("--fulleval-dir", default=None,
                    help="interactive gauntlet mode: dir of *.fulleval.json to compare (offline HTML)")
    ap.add_argument("--best-per-day", default=None,
                    help="optional best_per_day.json giving champion order for --fulleval-dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # ---- interactive gauntlet mode (self-contained offline HTML) --------------------------
    if a.fulleval_dir:
        import os as _os
        import sys as _sys
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from gauntlet import build_html, load_fulleval
        out = a.out or "/mnt/v/output/zensim/reports/summer_gauntlet.html"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        gbakes = load_fulleval(a.fulleval_dir, a.best_per_day)
        _p, size = build_html(gbakes, out)
        print(f"wrote {out}  ({size // 1024} KB)  {len(gbakes)} bakes (interactive gauntlet)\n  view: "
              + out.replace("/mnt/v/output/", "http://localhost:3300/"))
        return

    # ---- legacy per-bake matplotlib bandwise dashboard (unchanged) -------------------------
    a.out = a.out or str(OUTDIR / "bandwise_dashboard_2026-07-15.html")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    bakes = []
    for lab, p in [("B(shipped)", WEIGHTS / "b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin")]:
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
    # per-bake training set (desync-proof, from spec.json). metrics train on nothing -> empty set.
    train = {lab: bake_train(path) for lab, _k, path in bakes}
    for c in data:
        for lab, (_h, _p, _pan, kind) in data[c].items():
            if kind == "metric":
                train.setdefault(lab, METRIC_PROVENANCE.get(lab, (set(), ""))[0])

    body = ["<h1>zensim bandwise dashboard — bakes + ssim2/cvvdp/butteraugli × corpus × band</h1>",
            "<p class='sub'>Zoomable SVG. Comparison plots use rank-<b>percentile</b> so bakes on "
            "different pred scales (B=0..100, MLP candidates=z-space, metrics=native) are comparable. "
            "By <code>scripts/v_next/bandwise_dashboard.py</code>.</p>"]
    body.append(provenance_section(bakes))
    body.append("<h2 id='overview'>Cross-corpus overview</h2>")
    body.append("<p class='sub'>Heatmap columns use split-suffixed names (see provenance). Cells are "
                "SROCC; a bake's number on a corpus it TRAINED on (CHEAT) is memorization — cross-check "
                "the honesty column in each corpus table.</p>")
    body.append(fig_heatmap(data, labels))
    body.append("<div class='row'>" + fig_trade(data, labels) + fig_composite(data, labels) + "</div>")
    body.append(robustness_section(bakes))
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
        disp, split, _twin, foot = CORPUS_META[corp]
        cheats = [lab for lab in perbake if honesty(train.get(lab), corp)[0] == "CHEAT"]
        summary = (f" <span class='badge' style='background:#c0392b'>CHEAT for: {', '.join(cheats)}</span>"
                   if cheats else " <span class='badge' style='background:#27ae60'>held-out for all bakes</span>")
        body.append(f"<h2 id='{corp}'>{disp}{summary}</h2>")
        body.append(f"<p class='sub'>{split}. <a href='#fn-{corp}'>footnote↴</a> — {foot}</p>")
        body.append(agg_stat_table(perbake, corp, train))
        if corp in BANDABLE:
            body.append(band_table(perbake, corp, train))
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
