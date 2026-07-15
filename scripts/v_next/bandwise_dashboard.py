#!/usr/bin/env python3
"""Bandwise + every-graph eval dashboard (user 2026-07-15: "make html output do bandwise reports
and every graph possible"). For a set of bakes (npz candidates scored in numpy via blend_lib, +
shipped .bin bakes scored via bake_verdict --per-pair-output) across every held-out corpus, renders
ONE self-contained HTML: per-corpus bandwise (B0..B9) tables + grouped SROCC bars + per-bake
scatter (pred-vs-MOS + step-5 median + identity) + calibration overlay + residual + candlestick +
per-band overlay, plus a cross-corpus SROCC heatmap and composite ranking. Matplotlib PNGs are
base64-embedded (self-contained; viewable at http://172.23.240.1:3300/zensim/...).

  usage: bandwise_dashboard.py [--from-search <blend_results.json>]
                               [--bakes label:npz_or_bin,...] [--out dashboard.html]
"""
import argparse
import base64
import io
import json
import re
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
OUTDIR = Path("/mnt/v/output/zensim/dashboards")
plt.rcParams.update({"figure.dpi": 96, "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.facecolor": "white"})
BANDS = [f"B{i}" for i in range(10)]


def png(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def score_npz(npz_path, corpus):
    p = dict(np.load(npz_path))  # arrays only (blend_search strips pickled dict fields)
    X, h = B.load_val(corpus)
    return h, B.forward(p, X)


def score_bin(bin_path, corpus):
    """shipped ZNPR .bin -> (human, pred) via bake_verdict --per-pair-output (parquet-row order)."""
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tf:
        out = tf.name
    r = subprocess.run([BV, "--bake", str(bin_path), "--corpora", corpus,
                        "--per-pair-output", out], capture_output=True, text=True)
    try:
        arr = np.loadtxt(out, delimiter="\t", skiprows=1)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr[:, 0], arr[:, 1]   # human, pred
    except Exception:
        return None, None


def score(kind, path, corpus):
    return score_npz(path, corpus) if kind == "npz" else score_bin(path, corpus)


def band100(h):
    lo, hi = np.nanmin(h), np.nanmax(h)
    return (h - lo) / ((hi - lo) or 1) * 100.0


def collect(bakes):
    """{corpus: {label: (human, pred, panel)}}."""
    data = {}
    for corp, (_p, _y, sign, want_band) in B.VAL_CORPORA.items():
        data[corp] = {}
        for label, kind, path in bakes:
            h, pr = score(kind, path, corp)
            if h is None or pr is None or len(h) == 0:
                continue
            pan = B.panel(pr, h, sign=sign, per_band=want_band)
            data[corp][label] = (h, pr, pan)
    return data


def fig_scatter(h, pr, label, corp):
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    ax.scatter(h, pr, s=3, alpha=0.25, lw=0)
    # step-5 median line on MOS
    b100 = band100(h)
    xs, ys = [], []
    for lo in range(0, 100, 5):
        sel = (b100 >= lo) & (b100 < lo + 5)
        if sel.sum() > 2:
            xs.append(np.median(h[sel])); ys.append(np.median(pr[sel]))
    if xs:
        ax.plot(xs, ys, "-o", color="crimson", ms=2, lw=1)
    lim = [min(h.min(), pr.min()), max(h.max(), pr.max())]
    ax.plot(lim, lim, "k--", lw=0.6, alpha=0.5)
    ax.set_title(f"{label}\n{corp}", fontsize=7)
    ax.set_xlabel("human MOS"); ax.set_ylabel("pred")
    return png(fig)


def fig_bands_grouped(perbake, corp):
    """grouped per-band SROCC bar over all bakes."""
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    labels = list(perbake.keys()); nb = len(BANDS); w = 0.8 / max(1, len(labels))
    for i, lab in enumerate(labels):
        pan = perbake[lab][2]
        vals = [pan.get("bands", {}).get(b, {}).get("srocc", np.nan) for b in BANDS]
        ax.bar(np.arange(nb) + i * w, np.nan_to_num(vals), w, label=lab)
    ax.set_xticks(np.arange(nb) + 0.4); ax.set_xticklabels(BANDS, fontsize=6)
    ax.set_ylabel("SROCC"); ax.set_title(f"{corp} — per-band SROCC (B0..B9)", fontsize=8)
    ax.legend(fontsize=5, ncol=len(labels)); ax.axhline(0, color="k", lw=0.4)
    return png(fig)


def fig_calibration(perbake, corp):
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    for lab, (h, pr, _pan) in perbake.items():
        order = np.argsort(pr); prs, hs = pr[order], h[order]
        nb = 20; edges = np.linspace(0, len(pr), nb + 1).astype(int)
        xs = [prs[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a]
        ys = [hs[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a]
        ax.plot(xs, ys, "-o", ms=2, lw=1, label=lab)
    ax.set_xlabel("pred (binned)"); ax.set_ylabel("mean human")
    ax.set_title(f"{corp} — calibration", fontsize=8); ax.legend(fontsize=5)
    return png(fig)


def fig_residual(perbake, corp):
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    for lab, (h, pr, _pan) in perbake.items():
        A = np.polyfit(pr, h, 1); resid = h - np.polyval(A, pr)
        ax.scatter(np.polyval(A, pr), resid, s=2, alpha=0.2, lw=0, label=lab)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("fitted"); ax.set_ylabel("residual (human−fit)")
    ax.set_title(f"{corp} — residuals", fontsize=8); ax.legend(fontsize=5)
    return png(fig)


def fig_candlestick(h, pr, label, corp):
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    b100 = band100(h); data, pos = [], []
    for i in range(10):
        sel = (b100 >= i * 10) & (b100 < i * 10 + 10)
        if sel.sum() > 2:
            data.append(pr[sel]); pos.append(i)
    if data:
        ax.boxplot(data, positions=pos, widths=0.6, showfliers=False)
    ax.set_xticks(range(10)); ax.set_xticklabels(BANDS, fontsize=6)
    ax.set_xlabel("MOS band"); ax.set_ylabel("pred distribution")
    ax.set_title(f"{label} — {corp} candlestick", fontsize=7)
    return png(fig)


def fig_heatmap(data, labels):
    corps = list(B.VAL_CORPORA)
    M = np.full((len(labels), len(corps)), np.nan)
    for j, c in enumerate(corps):
        for i, lab in enumerate(labels):
            if lab in data[c]:
                pan = data[c][lab][2]
                M[i, j] = pan["srocc_abs"] if B.VAL_CORPORA[c][2] < 0 else pan["srocc"]
    fig, ax = plt.subplots(figsize=(1.1 + 0.7 * len(corps), 0.9 + 0.4 * len(labels)))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(corps))); ax.set_xticklabels(corps, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(corps)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=6)
    ax.set_title("SROCC — bake × corpus  (|SROCC| for signed JND corpora)", fontsize=9)
    fig.colorbar(im, fraction=0.025)
    return png(fig)


DIAL_GRID = "/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29_quarantined.parquet"
_dial = {}


def load_dial():
    if not _dial:
        import pyarrow.parquet as pq
        names = [f.name for f in pq.read_schema(DIAL_GRID)]
        pfx = "feat_" if "feat_0" in names else "f"
        t = pq.read_table(DIAL_GRID, columns=["image_id", "codec", "q"]
                          + [f"{pfx}{i}" for i in range(B.N_FEAT)])
        _dial["meta"] = t.select(["image_id", "codec", "q"]).to_pandas()
        _dial["X"] = np.stack([np.asarray(t[f"{pfx}{i}"], dtype=np.float32)
                               for i in range(B.N_FEAT)], 1)
    return _dial["meta"], _dial["X"]


def dial_mono(meta, pred):
    import pandas as pd
    df = meta.copy(); df["pred"] = pred
    good = tot = 0
    for _key, g in df.groupby(["image_id", "codec"]):
        p = g.sort_values("q")["pred"].to_numpy()
        if len(p) < 2:
            continue
        d = np.diff(p); good += int((d >= 0).sum()); tot += len(d)
    return good / tot if tot else np.nan


def fig_dial(npz_bakes):
    """score-vs-q dial per codec (median over images) + monotonicity %, for npz candidates."""
    import pandas as pd
    meta, X = load_dial()
    codecs = sorted(meta["codec"].unique())
    fig, axes = plt.subplots(1, len(codecs), figsize=(2.1 * len(codecs), 2.6))
    if len(codecs) == 1:
        axes = [axes]
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
    return png(fig), monos


def agg_stat_table(perbake, corp):
    """full Mohammadi panel per bake for one corpus (covers JND corpora with no per-band)."""
    sign = B.VAL_CORPORA[corp][2]
    head = ["SROCC" + (" (|·|)" if sign < 0 else ""), "PLCC", "KROCC", "Z-RMSE", "n"]
    rows = ["<table><thead><tr><th>bake</th>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"]
    for lab, (_h, _p, pan) in perbake.items():
        sr = pan["srocc_abs"] if sign < 0 else pan["srocc"]
        vals = [sr, pan["plcc"], pan["krocc"], pan["zrmse"], pan["n"]]
        cells = "".join(f"<td>{v:.4f}</td>" if isinstance(v, float) and np.isfinite(v)
                        else (f"<td>{v}</td>" if not isinstance(v, float) else "<td>—</td>") for v in vals)
        rows.append(f"<tr><td class='lbl'>{lab}</td>{cells}</tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


def fig_trade(data, labels):
    """Pareto trade scatters: CID22 vs non-photo, CID22 vs KonJND|·| — the operating-point map."""
    def sr(lab, c):
        if lab not in data[c]:
            return np.nan
        pan = data[c][lab][2]
        return pan["srocc_abs"] if B.VAL_CORPORA[c][2] < 0 else pan["srocc"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for ax, (yc, yl) in [(a1, ("nonphoto", "non-photo SROCC")), (a2, ("konjnd", "KonJND |SROCC|"))]:
        for lab in labels:
            x, y = sr(lab, "cid22"), sr(lab, yc)
            if np.isfinite(x) and np.isfinite(y):
                ax.scatter(x, y, s=40); ax.annotate(lab, (x, y), fontsize=6,
                                                    xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("CID22 SROCC"); ax.set_ylabel(yl); ax.set_title(f"CID22 vs {yc}", fontsize=8)
    fig.suptitle("Operating-point trade map (upper-right = better on both)", fontsize=9)
    return png(fig)


def fig_composite(data, labels):
    fig, ax = plt.subplots(figsize=(1.2 + 0.7 * len(labels), 2.8))
    comps = []
    for lab in labels:
        res = {c: data[c][lab][2] for c in B.VAL_CORPORA if lab in data[c]}
        for c in B.VAL_CORPORA:
            res.setdefault(c, {"srocc": np.nan, "srocc_abs": np.nan})
            res[c]["sign"] = B.VAL_CORPORA[c][2]
        comps.append(B.composite(res)[0])
    order = np.argsort(comps)[::-1]
    ax.bar([labels[i] for i in order], [comps[i] for i in order], color="steelblue")
    ax.set_ylabel("composite (CID22+0.3·nonphoto+0.2·KonJND+…)"); ax.set_title("Goal-aware ranking", fontsize=8)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels([labels[i] for i in order], rotation=35, ha="right", fontsize=6)
    return png(fig)


def band_table(perbake, corp):
    rows = ["<table><thead><tr><th>bake</th>" + "".join(f"<th>{b}</th>" for b in BANDS)
            + "<th>agg</th><th>n</th></tr></thead><tbody>"]
    # per-column min/max for coloring
    allv = {b: [] for b in BANDS + ["agg"]}
    for lab, (_h, _p, pan) in perbake.items():
        for b in BANDS:
            v = pan.get("bands", {}).get(b, {}).get("srocc", np.nan)
            if np.isfinite(v):
                allv[b].append(v)
        agg = pan["srocc_abs"] if B.VAL_CORPORA[corp][2] < 0 else pan["srocc"]
        if np.isfinite(agg):
            allv["agg"].append(agg)

    def cell(v, col):
        if v is None or not np.isfinite(v):
            return '<td style="color:#666">—</td>'
        vv = allv[col]
        lo, hi = (min(vv), max(vv)) if vv else (0, 1)
        t = 0 if hi == lo else max(0, min(1, (v - lo) / (hi - lo)))
        r = int(200 * (1 - t) + 40 * t); g = int(50 * (1 - t) + 160 * t)
        return f'<td style="background:rgb({r},{g},60);color:#fff">{v:.3f}</td>'

    for lab, (_h, _p, pan) in perbake.items():
        agg = pan["srocc_abs"] if B.VAL_CORPORA[corp][2] < 0 else pan["srocc"]
        cells = "".join(cell(pan.get("bands", {}).get(b, {}).get("srocc"), b) for b in BANDS)
        rows.append(f"<tr><td class='lbl'>{lab}</td>{cells}{cell(agg,'agg')}<td>{pan['n']}</td></tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-search", default="/mnt/v/output/zensim/reports/blend/blend_results_2026-07-15.json")
    ap.add_argument("--bakes", default=None, help="label:path,... (path .npz or .bin) — overrides defaults")
    ap.add_argument("--out", default=str(OUTDIR / "bandwise_dashboard_2026-07-15.html"))
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    bakes = []
    # shipped references (ZNPR .bin, scored via bake_verdict)
    ship = [("B(shipped)", WEIGHTS / "b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin"),
            ("§8.35 diverse", REPORTS / "mlp_diverse_depoison_dv0.5kw0.3_2026-07-15.bin")]
    for lab, p in ship:
        if Path(p).exists():
            bakes.append((lab, "bin", str(p)))
    # blend-search top-K candidates (npz)
    if a.bakes:
        for tok in a.bakes.split(","):
            lab, pth = tok.split(":", 1)
            bakes.append((lab, "npz" if pth.endswith(".npz") else "bin", pth))
    elif Path(a.from_search).exists():
        js = json.loads(Path(a.from_search).read_text())
        for s in js.get("saved", [])[:5]:
            bakes.append((s["label"], "npz", s["npz"]))

    print(f"scoring {len(bakes)} bakes × {len(B.VAL_CORPORA)} corpora ...")
    data = collect(bakes)
    labels = [lab for lab, *_ in bakes]

    body = ["<h1>zensim bandwise dashboard — every bake × corpus × band</h1>",
            "<p class='sub'>Rank SROCC per corpus + 10-band (B0..B9). Signed JND corpora "
            "(konjnd/aic3/aic4) report |SROCC|. Candidates from the optimal-blend search "
            "(blend_search.py) vs shipped B + the §8.35 diverse bake. "
            "Generated by <code>scripts/v_next/bandwise_dashboard.py</code>.</p>"]
    body.append("<h2 id='overview'>Cross-corpus overview</h2>")
    body.append(f'<img src="{fig_heatmap(data, labels)}">')
    body.append("<div class='row'>")
    body.append(f'<img src="{fig_trade(data, labels)}">')
    body.append(f'<img src="{fig_composite(data, labels)}">')
    body.append("</div>")
    # dial curves (npz candidates only — bins' dial lives in bake_verdict --html)
    npz_bakes = [(lab, path) for lab, kind, path in bakes if kind == "npz"]
    if npz_bakes:
        try:
            dimg, monos = fig_dial(npz_bakes)
            body.append("<h2 id='dial'>Codec-target dial (score vs q per codec — npz candidates)</h2>")
            body.append("<p class='sub'>monotonicity (fraction of non-decreasing adjacent-q steps): "
                        + " · ".join(f"<b>{k}</b> {v*100:.1f}%" for k, v in monos.items()) + "</p>")
            body.append(f'<img src="{dimg}">')
        except Exception as e:
            body.append(f"<p class='sub'>dial section skipped: {e}</p>")
    # TOC
    toc = " · ".join(f"<a href='#{c}'>{c}</a>" for c in B.VAL_CORPORA)
    body.append(f"<p class='toc'>corpora: {toc}</p>")

    for corp in B.VAL_CORPORA:
        perbake = data[corp]
        if not perbake:
            continue
        body.append(f"<h2 id='{corp}'>{corp}</h2>")
        body.append(agg_stat_table(perbake, corp))  # full Mohammadi panel (all corpora)
        if B.VAL_CORPORA[corp][3]:  # per-band enabled
            body.append(band_table(perbake, corp))
            body.append(f'<img src="{fig_bands_grouped(perbake, corp)}">')
        body.append("<div class='row'>")
        body.append(f'<img src="{fig_calibration(perbake, corp)}">')
        body.append(f'<img src="{fig_residual(perbake, corp)}">')
        body.append("</div>")
        body.append("<div class='row'>")
        for lab in perbake:
            h, pr, _pan = perbake[lab]
            body.append(f'<img src="{fig_scatter(h, pr, lab, corp)}">')
        body.append("</div>")
        if B.VAL_CORPORA[corp][3]:
            body.append("<div class='row'>")
            for lab in perbake:
                h, pr, _pan = perbake[lab]
                body.append(f'<img src="{fig_candlestick(h, pr, lab, corp)}">')
            body.append("</div>")

    html = ("<style>body{font:13px system-ui,sans-serif;margin:1.4rem;background:#fff;color:#111}"
            "h1{font-size:1.3rem}h2{font-size:1.05rem;margin-top:1.6rem;border-top:1px solid #ccc;padding-top:.5rem}"
            ".sub,.toc{color:#555;max-width:70rem}.row{display:flex;flex-wrap:wrap;gap:6px;align-items:flex-start}"
            "img{max-width:100%;border:1px solid #eee}table{border-collapse:collapse;margin:.4rem 0;font-size:11px}"
            "th,td{border:1px solid #ddd;padding:2px 6px;text-align:center}td.lbl{text-align:left;font-weight:600;background:#f4f4f4}"
            "code{background:#f0f0f0;padding:.1rem .3rem}a{color:#06c}</style>" + "".join(body))
    Path(a.out).write_text(html)
    url = a.out.replace("/mnt/v/output/", "http://172.23.240.1:3300/")
    print(f"wrote {a.out}\n  view: {url}")


if __name__ == "__main__":
    main()
