#!/usr/bin/env python3
"""AIC2026 metric-agreement panel — NOT an accuracy measurement.

JPEG AIC2026 (DaRUS doi:10.18419/DARUS-6156, CC BY-SA 4.0) ships **no human
scores**. It carries 71 objective IQA-method score columns over 9,618
distorted images (70 sources x 17 codec configs x 20 levels, FTIC 6). So every
number this script produces is agreement between metrics, or the behaviour of a
metric along a codec's own quality ladder. Nothing here is accuracy against
human judgment, and no output of this script may be read as one.

Two properties of the dataset that the statistics have to respect:

* **CVVDP placed the levels.** Every ladder was built to be evenly spaced in
  CVVDP-estimated JND, so CVVDP is monotone on these ladders by construction.
  It is reported for completeness and excluded from any monotonicity ranking.
* **Column orientation is mixed** -- `JND_*` and the distance-like metrics rise
  with distortion, the SSIMULACRA2/CVVDP/PSNR/SSIM families fall. Orientation
  is DERIVED here (mean within-ladder Spearman against `distortion_level`) and
  cross-checked against the expected sign, rather than assumed.

Sections, matching `benchmarks/aic2026_agreement_2026-09-19.md`:

  A  per-ladder monotonicity (ours and the peers, absolute + relative
     materiality)
  B  agreement matrix: pooled and mean within-ladder Spearman against all 71
     shipped columns, with a source-level bootstrap on the headline pairs
  C  cross-codec consistency at matched JND -- does one dial value mean the
     same thing on JPG / JXL / AVIF / JPEG-AI / the learned codecs
  D  implementation parity: our fast-ssim2 vs their `SSIMULACRA2`, our
     butteraugli vs their `proposal-Butteraugli`
  E  cropped (840x944) vs full-resolution score shift

Usage:

    python3 scripts/aic2026_agreement.py \\
      --metrics-cropped /mnt/v/datasets/aic2026/metrics_cropped.csv \\
      --metrics-fullres /mnt/v/datasets/aic2026/metrics_fullres.csv \\
      --scores-cropped  <dir>/zensim_scores_cropped.parquet \\
      --scores-fullres  <dir>/zensim_scores_fullres.parquet \\
      --peer-cropped    <dir>/peer_scores_cropped.tsv \\
      --out-md   benchmarks/aic2026_agreement_2026-09-19.md \\
      --out-json benchmarks/aic2026_agreement_2026-09-19.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats

# Key (non-score) columns of the shipped metric tables.
KEY_COLS = [
    "distorted",
    "source",
    "codec",
    "codec_acronym",
    "distortion_level",
    "bpp",
    "codec_id",
    "enc_setting",
    "enc_value",
]

# Codec groupings used in section C. "learned" is where hand-built metrics and
# deep metrics are expected to disagree, which is the interesting cell.
CODEC_GROUPS = {
    "traditional": ["JPG", "JPGR", "JPGL", "J2K", "WEBP", "PNG"],
    "modern-block": ["AVIF", "AVIFQ", "HEVC", "VVC"],
    "jxl": ["JXL", "JXLM"],
    "jpeg-ai": ["JAI", "JAIB"],
    "learned": ["CCW", "CCM", "FTIC"],
}

# Headline peer columns the markdown leads with.
HEADLINE_PEERS = [
    "SSIMULACRA2",
    "CVVDP",
    "JND_CVVDP",
    "JND_SSIMULACRA2",
    "proposal-Butteraugli",
    "DSSIM",
    "IW-SSIM",
    "VMAF",
    "LPIPS-VGG",
    "LPIPS-alex",
    "DISTS",
    "TOPIQ",
    "PSNR-Y",
]

# Deep/learned-feature metrics, contrasted against ours on learned codecs.
DEEP_PEERS = ["LPIPS-VGG", "LPIPS-alex", "DISTS", "TOPIQ", "AHIQ", "DreamSim"]


_LADDER_CACHE: Dict[int, List[np.ndarray]] = {}


def ladder_groups(df: pd.DataFrame) -> List[np.ndarray]:
    """Positional row indices of each (source, codec) ladder, cached per frame.

    The groupby is the expensive part of every within-ladder statistic and the
    partition never changes, so it is computed once. 490 ladders x 71 columns x
    6 models is 200k+ correlations; re-grouping each time turns minutes into
    hours."""
    key = id(df)
    if key not in _LADDER_CACHE:
        codes = df.groupby(["source", "codec_acronym"], sort=False).ngroup().to_numpy()
        order = np.argsort(codes, kind="stable")
        bounds = np.flatnonzero(np.diff(codes[order])) + 1
        _LADDER_CACHE[key] = [
            g for g in np.split(order, bounds) if g.size >= 3
        ]
    return _LADDER_CACHE[key]


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rho, NaN when either side is constant or too short."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    a, b = a[ok], b[ok]
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def derive_orientation(df: pd.DataFrame, col: str) -> float:
    """+1 if the column FALLS with distortion (quality-oriented), -1 if it
    RISES (distance-oriented). Derived, never assumed: the mean within-ladder
    Spearman against `distortion_level`."""
    lv = df["distortion_level"].to_numpy(dtype=float)
    cv = df[col].to_numpy(dtype=float)
    rhos = [spearman(lv[g], cv[g]) for g in ladder_groups(df)]
    rhos = [r for r in rhos if np.isfinite(r)]
    if not rhos:
        return float("nan")
    return -1.0 if float(np.mean(rhos)) > 0 else 1.0


def ladder_monotonicity(
    df: pd.DataFrame, col: str, orient: float, materiality: float
) -> Dict[str, float]:
    """Fraction of adjacent level pairs moving the right way, and how many
    ladders carry an inversion bigger than `materiality` (in the column's own
    units)."""
    good = 0
    total = 0
    bad_ladders = 0
    n_ladders = 0
    worst = 0.0
    lv = df["distortion_level"].to_numpy(dtype=float)
    cv = df[col].to_numpy(dtype=float)
    for idx in ladder_groups(df):
        idx = idx[np.argsort(lv[idx], kind="stable")]
        v = orient * cv[idx]
        if len(v) < 2 or not np.all(np.isfinite(v)):
            continue
        n_ladders += 1
        d = np.diff(v)
        # Oriented to quality: a correct step is non-increasing.
        good += int(np.sum(d <= 0))
        total += len(d)
        up = d[d > 0]
        if up.size and float(up.max()) > materiality:
            bad_ladders += 1
            worst = max(worst, float(up.max()))
    return {
        "pairs_correct_frac": good / total if total else float("nan"),
        "n_pairs": total,
        "n_ladders": n_ladders,
        "ladders_with_material_inversion": bad_ladders,
        "ladders_with_material_inversion_frac": (
            bad_ladders / n_ladders if n_ladders else float("nan")
        ),
        "worst_inversion": worst,
        "materiality": materiality,
    }


def span_p1_p99(series: pd.Series) -> float:
    """A column's robust span on this population. Cross-codec spreads in
    different units are meaningless side by side; spread/span is not."""
    v = series.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 10:
        return float("nan")
    lo, hi = np.percentile(v, [1, 99])
    return float(hi - lo)


def relative_materiality(series: pd.Series) -> float:
    """0.5 units on a 0-100 dial = 0.5 % of the dial's span. The equivalent for
    a column in unknown units is 0.5 % of its robust (p1..p99) span over this
    population. Stated so the peer numbers are comparable, not assumed equal."""
    v = series.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 10:
        return float("nan")
    lo, hi = np.percentile(v, [1, 99])
    return 0.005 * float(hi - lo)


def within_ladder_srocc(df: pd.DataFrame, a: str, b: str) -> float:
    """Mean of the per-ladder Spearman between two columns."""
    av = df[a].to_numpy(dtype=float)
    bv = df[b].to_numpy(dtype=float)
    rhos = [spearman(av[g], bv[g]) for g in ladder_groups(df)]
    rhos = [r for r in rhos if np.isfinite(r)]
    return float(np.mean(rhos)) if rhos else float("nan")


def source_index_groups(df: pd.DataFrame) -> List[np.ndarray]:
    """Positional row indices per SOURCE -- the unit of independence for every
    bootstrap here is the source image, not the row."""
    codes = df.groupby("source", sort=False).ngroup().to_numpy()
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    return list(np.split(order, bounds))


def bootstrap_pooled_srocc_many(
    df: pd.DataFrame, pairs: Sequence[tuple], n_boot: int, seed: int
) -> Dict[tuple, Dict[str, float]]:
    """Percentile CIs for many pooled Spearmans at once: ONE source resample
    per replicate, reused by every pair. Per-pair resampling would repeat the
    same expensive index build dozens of times for no extra information."""
    rng = np.random.default_rng(seed)
    groups = source_index_groups(df)
    cols = {c: df[c].to_numpy(dtype=float) for p in pairs for c in p}
    acc: Dict[tuple, List[float]] = {p: [] for p in pairs}
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        for p in pairs:
            acc[p].append(spearman(cols[p[0]][idx], cols[p[1]][idx]))
    out = {}
    for p, vals in acc.items():
        v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
        if v.size == 0:
            out[p] = {"lo": float("nan"), "hi": float("nan"), "n_boot": 0}
        else:
            lo, hi = np.percentile(v, [2.5, 97.5])
            out[p] = {"lo": float(lo), "hi": float(hi), "n_boot": int(v.size)}
    return out


def jnd_bins(v: pd.Series) -> pd.Series:
    edges = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, np.inf]
    labels = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-2.5", "2.5-3", "3-4", "4+"]
    return pd.cut(v, bins=edges, labels=labels, right=False)


def per_codec_offsets(
    df: pd.DataFrame, model: str, bin_col: str, min_rows: int, n_boot: int, seed: int
) -> Dict[str, Dict[str, float]]:
    """At matched JND, how far each codec's mean model score sits from the
    all-codec mean of the same bin, averaged over bins. A metric that means the
    same thing everywhere has offsets near zero; a large positive offset means
    the model scores that codec HIGHER than the shared JND level implies."""
    work = df[["source", "codec_acronym", model, bin_col]].dropna()
    rng = np.random.default_rng(seed)
    val = work[model].to_numpy(dtype=float)
    bin_code = work[bin_col].astype(str).to_numpy()
    codec = work["codec_acronym"].to_numpy()

    def offsets(idx: np.ndarray) -> Dict[str, float]:
        out: Dict[str, List[float]] = {}
        b = bin_code[idx]
        c = codec[idx]
        v = val[idx]
        for bb in np.unique(b):
            sel = b == bb
            if int(sel.sum()) < min_rows:
                continue
            grand = float(v[sel].mean())
            cc = c[sel]
            vv = v[sel]
            for k in np.unique(cc):
                m = cc == k
                if int(m.sum()) < 5:
                    continue
                out.setdefault(str(k), []).append(float(vv[m].mean()) - grand)
        return {k: float(np.mean(x)) for k, x in out.items() if x}

    point = offsets(np.arange(len(work)))
    groups = source_index_groups(work)
    boot: Dict[str, List[float]] = {k: [] for k in point}
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        for k, v in offsets(idx).items():
            if k in boot:
                boot[k].append(v)
    res = {}
    for k, v in point.items():
        arr = np.asarray(boot[k], dtype=float)
        if arr.size >= 10:
            lo, hi = np.percentile(arr, [2.5, 97.5])
        else:
            lo = hi = float("nan")
        res[k] = {"offset": v, "lo": float(lo), "hi": float(hi)}
    return res


def load_scores(metrics_csv: str, scores_parquet: str) -> pd.DataFrame:
    """Join our per-row scores onto the shipped table by `distorted`."""
    m = pd.read_csv(metrics_csv)
    s = pd.read_parquet(scores_parquet)
    key = "stimulus" if "stimulus" in s.columns else "distorted"
    s = s.rename(columns={key: "distorted"})
    keep = ["distorted", "ref_width", "ref_height"] + [
        c for c in s.columns if c.startswith("score_")
    ]
    out = m.merge(s[keep], on="distorted", how="inner", validate="one_to_one")
    if len(out) != len(m):
        raise SystemExit(
            f"join lost rows: metrics {len(m)} -> joined {len(out)} "
            f"({metrics_csv} x {scores_parquet})"
        )
    return out


def round_floats(obj, nd: int = 6):
    """Round every float in a nested structure. Full f64 text is most of the
    committed JSON's bytes and none of its information -- the statistics here
    are not meaningful past ~6 decimals, and the repo does not carry files
    over 30 KB."""
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else round(obj, nd)
    if isinstance(obj, dict):
        return {k: round_floats(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, nd) for v in obj]
    return obj


def fmt(x: float, nd: int = 4) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics-cropped", required=True)
    ap.add_argument("--metrics-fullres")
    ap.add_argument("--scores-cropped", required=True)
    ap.add_argument("--scores-fullres")
    ap.add_argument("--peer-cropped")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True,
                    help="committed summary JSON (kept small — the headline "
                         "pairs, not the 71-column matrix)")
    ap.add_argument("--out-json-full",
                    help="the complete report including the full 71-column "
                         "agreement matrix; belongs on /mnt/v, not in git")
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260919)
    args = ap.parse_args()

    cropped = load_scores(args.metrics_cropped, args.scores_cropped)
    models = [c for c in cropped.columns if c.startswith("score_")]
    peer_cols = [
        c
        for c in cropped.columns
        if c not in KEY_COLS and not c.startswith("score_") and c not in
        ("ref_width", "ref_height")
    ]
    report: Dict[str, object] = {
        "dataset": "JPEG AIC2026 (DaRUS doi:10.18419/DARUS-6156, CC BY-SA 4.0)",
        "human_labels": False,
        "caveat": (
            "No human scores exist in this release. Every number below is "
            "metric-vs-metric agreement or ladder behaviour. CVVDP placed the "
            "distortion levels, so CVVDP is monotone on these ladders by "
            "construction and is excluded from monotonicity rankings."
        ),
        "n_rows": int(len(cropped)),
        "n_sources": int(cropped["source"].nunique()),
        "n_ladders": int(cropped.groupby(["source", "codec_acronym"]).ngroups),
        "models": models,
        "n_peer_columns": len(peer_cols),
    }

    # --- orientation, derived
    orient = {c: derive_orientation(cropped, c) for c in peer_cols}
    for m in models:
        orient[m] = derive_orientation(cropped, m)
    report["orientation"] = {
        k: ("quality" if v == 1.0 else "distortion" if v == -1.0 else "undetermined")
        for k, v in orient.items()
    }

    # --- A. monotonicity
    mono: Dict[str, Dict[str, float]] = {}
    for c in models + peer_cols:
        rel = relative_materiality(cropped[c])
        mono[c] = {
            "relative": ladder_monotonicity(cropped, c, orient.get(c, 1.0), rel),
        }
    for m in models:
        mono[m]["absolute_0p5"] = ladder_monotonicity(cropped, m, orient[m], 0.5)
    report["A_monotonicity"] = mono

    # --- B. agreement
    agreement: Dict[str, Dict[str, Dict[str, float]]] = {}
    for m in models:
        agreement[m] = {}
        for c in peer_cols:
            sign = orient[m] * orient.get(c, 1.0)
            pooled = spearman(cropped[m], cropped[c])
            within = within_ladder_srocc(cropped, m, c)
            agreement[m][c] = {
                "pooled_srocc_signed": (
                    float(sign * pooled) if np.isfinite(pooled) else float("nan")
                ),
                "within_ladder_srocc_signed": (
                    float(sign * within) if np.isfinite(within) else float("nan")
                ),
            }
    boot_pairs = [
        (m, c) for m in models for c in HEADLINE_PEERS if c in agreement[m]
    ]
    cis = bootstrap_pooled_srocc_many(cropped, boot_pairs, args.n_boot, args.seed)
    for m, c in boot_pairs:
        ci = cis[(m, c)]
        sign = orient[m] * orient.get(c, 1.0)
        lo, hi = float(sign * ci["lo"]), float(sign * ci["hi"])
        agreement[m][c]["pooled_ci95"] = [min(lo, hi), max(lo, hi)]
        agreement[m][c]["n_boot"] = ci["n_boot"]
    report["B_agreement"] = agreement

    # --- B2. deep-metric agreement per codec group (the learned-codec cell)
    groupwise: Dict[str, Dict[str, Dict[str, float]]] = {}
    for gname, acros in CODEC_GROUPS.items():
        sub = cropped[cropped["codec_acronym"].isin(acros)]
        if len(sub) < 50:
            continue
        groupwise[gname] = {}
        for m in models:
            groupwise[gname][m] = {}
            for c in ["SSIMULACRA2", "CVVDP"] + DEEP_PEERS:
                if c not in sub.columns:
                    continue
                sign = orient[m] * orient.get(c, 1.0)
                w = within_ladder_srocc(sub, m, c)
                groupwise[gname][m][c] = (
                    float(sign * w) if np.isfinite(w) else float("nan")
                )
        groupwise[gname]["_n_rows"] = int(len(sub))
    report["B2_group_agreement"] = groupwise

    # --- C. cross-codec consistency at matched JND
    cropped = cropped.copy()
    cropped["bin_cvvdp"] = jnd_bins(cropped["JND_CVVDP"])
    cropped["bin_ssim2"] = jnd_bins(cropped["JND_SSIMULACRA2"])
    consistency: Dict[str, Dict[str, object]] = {}
    for anchor, bcol in (("JND_CVVDP", "bin_cvvdp"), ("JND_SSIMULACRA2", "bin_ssim2")):
        consistency[anchor] = {}
        for m in models:
            offs = per_codec_offsets(cropped, m, bcol, 100, args.n_boot, args.seed)
            vals = [v["offset"] for v in offs.values()]
            span = span_p1_p99(cropped[m])
            spread = float(max(vals) - min(vals)) if vals else float("nan")
            consistency[anchor][m] = {
                "per_codec": offs,
                "spread": spread,
                "span_p1_p99": span,
                "spread_over_span": spread / span if span else float("nan"),
            }
        for c in ["SSIMULACRA2", "CVVDP"] + DEEP_PEERS:
            if c not in cropped.columns:
                continue
            offs = per_codec_offsets(cropped, c, bcol, 100, args.n_boot, args.seed)
            vals = [v["offset"] for v in offs.values()]
            span = span_p1_p99(cropped[c])
            spread = float(max(vals) - min(vals)) if vals else float("nan")
            consistency[anchor][c] = {
                "per_codec": offs,
                "spread": spread,
                "span_p1_p99": span,
                "spread_over_span": spread / span if span else float("nan"),
                "note": "peer column in its own units — compare "
                "spread_over_span, never the raw spread",
            }
    report["C_cross_codec"] = consistency

    # --- D. implementation parity
    parity: Dict[str, object] = {}
    if args.peer_cropped and os.path.exists(args.peer_cropped):
        peer = pd.read_csv(args.peer_cropped, sep="\t")
        key = "stimulus" if "stimulus" in peer.columns else "distorted"
        peer = peer.rename(columns={key: "distorted"})
        j = cropped.merge(peer, on="distorted", how="inner")
        parity["n_rows"] = int(len(j))
        if "SSIMULACRA2" in j.columns and "ssim2" in j.columns:
            d = (j["ssim2"] - j["SSIMULACRA2"]).to_numpy(dtype=float)
            worst = int(np.nanargmax(np.abs(d)))
            parity["ssimulacra2"] = {
                "ours": "fast-ssim2 0.8.0 (fast_ssim2::compute_ssimulacra2)",
                "theirs": "AIC2026 column SSIMULACRA2",
                "max_abs_diff": float(np.nanmax(np.abs(d))),
                "mean_abs_diff": float(np.nanmean(np.abs(d))),
                "median_abs_diff": float(np.nanmedian(np.abs(d))),
                "srocc": spearman(j["ssim2"], j["SSIMULACRA2"]),
                "pearson": float(
                    np.corrcoef(j["ssim2"], j["SSIMULACRA2"])[0, 1]
                ),
                "worst_row": {
                    "distorted": str(j["distorted"].iloc[worst]),
                    "codec": str(j["codec_acronym"].iloc[worst]),
                    "level": int(j["distortion_level"].iloc[worst]),
                    "ours": float(j["ssim2"].iloc[worst]),
                    "theirs": float(j["SSIMULACRA2"].iloc[worst]),
                },
            }
        if "proposal-Butteraugli" in j.columns:
            norms = {}
            for n in ["butter_max", "butter_p1", "butter_p2", "butter_p3"]:
                if n not in j.columns:
                    continue
                d = (j[n] - j["proposal-Butteraugli"]).to_numpy(dtype=float)
                norms[n] = {
                    "max_abs_diff": float(np.nanmax(np.abs(d))),
                    "mean_abs_diff": float(np.nanmean(np.abs(d))),
                    "srocc": spearman(j[n], j["proposal-Butteraugli"]),
                }
            best = min(
                (k for k in norms if np.isfinite(norms[k]["mean_abs_diff"])),
                key=lambda k: norms[k]["mean_abs_diff"],
                default=None,
            )
            best_rank = max(
                (k for k in norms if np.isfinite(norms[k]["srocc"])),
                key=lambda k: norms[k]["srocc"],
                default=None,
            )
            mono = None
            if best_rank:
                o = j.sort_values("proposal-Butteraugli")[best_rank].to_numpy(float)
                d = np.diff(o)
                bands = []
                t = j["proposal-Butteraugli"].to_numpy(float)
                v = j[best_rank].to_numpy(float)
                for lo, hi in [
                    (0.0, 0.01),
                    (0.01, 0.1),
                    (0.1, 1.0),
                    (1.0, 4.0),
                    (4.0, 10.0),
                    (10.0, 1e9),
                ]:
                    sel = (t >= lo) & (t < hi)
                    if int(sel.sum()) < 5:
                        continue
                    bands.append(
                        {
                            "theirs_range": [lo, hi],
                            "n": int(sel.sum()),
                            "median_ours_over_theirs": float(
                                np.median(v[sel] / np.where(t[sel] == 0, np.nan, t[sel]))
                            ),
                            "median_ours": float(np.median(v[sel])),
                        }
                    )
                mono = {
                    "norm": best_rank,
                    "adjacent_pairs": int(d.size),
                    "monotonicity_violations": int((d < 0).sum()),
                    "largest_violation": float(d.min()) if d.size else float("nan"),
                    "kendall_tau": float(
                        stats.kendalltau(t, v).statistic
                    ),
                    "ratio_by_band": bands,
                }
            parity["butteraugli"] = {
                "ours": "butteraugli 0.9 (imazen), four norms",
                "theirs": "AIC2026 column proposal-Butteraugli (norm not stated)",
                "by_norm": norms,
                "best_matching_norm_by_mean_abs_diff": best,
                "best_matching_norm_by_rank": best_rank,
                "monotone_map": mono,
            }
    report["D_parity"] = parity

    # --- E. cropped vs full-resolution
    shift: Dict[str, object] = {}
    if args.metrics_fullres and args.scores_fullres and os.path.exists(
        args.scores_fullres
    ):
        full = load_scores(args.metrics_fullres, args.scores_fullres)
        full["stem"] = full["distorted"].str.replace(r"\.png$", "", regex=True)
        crop = cropped.copy()
        crop["stem"] = (
            crop["distorted"]
            .str.replace(r"^PTC_", "", regex=True)
            .str.replace(r"\.png$", "", regex=True)
        )
        j = crop.merge(full, on="stem", suffixes=("_crop", "_full"))
        j["is_cat3"] = (j["ref_width_full"] != 840) | (j["ref_height_full"] != 944)
        shift["n_joined"] = int(len(j))
        shift["n_cat3_rows"] = int(j["is_cat3"].sum())
        shift["n_cat3_sources"] = int(j.loc[j["is_cat3"], "source_full"].nunique())
        for m in models:
            a, b = f"{m}_crop", f"{m}_full"
            if a not in j.columns or b not in j.columns:
                continue
            d3 = (j.loc[j["is_cat3"], b] - j.loc[j["is_cat3"], a]).to_numpy(float)
            d1 = (j.loc[~j["is_cat3"], b] - j.loc[~j["is_cat3"], a]).to_numpy(float)
            shift[m] = {
                "cat3_mean_full_minus_crop": float(np.nanmean(d3)) if d3.size else None,
                "cat3_median": float(np.nanmedian(d3)) if d3.size else None,
                "cat3_p5_p95": (
                    [float(x) for x in np.nanpercentile(d3, [5, 95])]
                    if d3.size
                    else None
                ),
                "cat3_max_abs": float(np.nanmax(np.abs(d3))) if d3.size else None,
                "same_geometry_mean": float(np.nanmean(d1)) if d1.size else None,
                "same_geometry_max_abs": (
                    float(np.nanmax(np.abs(d1))) if d1.size else None
                ),
                "cat3_srocc_crop_vs_full": spearman(
                    j.loc[j["is_cat3"], a], j.loc[j["is_cat3"], b]
                ),
            }
    report["E_crop_vs_fullres"] = shift

    if args.out_json_full:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.out_json_full)) or ".",
            exist_ok=True,
        )
        with open(args.out_json_full, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, sort_keys=True, default=str)

    # The committed JSON is a SUMMARY: headline pairs, model-level detail, and
    # every peer column reduced to the two numbers the report reads. The full
    # 71-column matrix and the peers' per-codec offsets live in
    # --out-json-full on block storage. Nothing over 30 KB goes into the repo
    # (workspace rule), and full f64 text is bytes without information.
    def slim_mono(block):
        r = block["relative"]
        out = {
            "pairs_correct_frac": r["pairs_correct_frac"],
            "material_inversion_ladders": r["ladders_with_material_inversion"],
            "n_ladders": r["n_ladders"],
            "materiality": r["materiality"],
        }
        if "absolute_0p5" in block:
            out["abs0p5_inversion_ladders"] = block["absolute_0p5"][
                "ladders_with_material_inversion"
            ]
        return out

    summary = {
        k: v
        for k, v in report.items()
        if k
        not in ("A_monotonicity", "B_agreement", "C_cross_codec", "orientation")
    }
    summary["full_report"] = (
        args.out_json_full or "not written (pass --out-json-full)"
    )
    summary["orientation"] = {
        k: v
        for k, v in report["orientation"].items()  # type: ignore[union-attr]
        if k in HEADLINE_PEERS or k in models
    }
    summary["A_monotonicity"] = {
        k: slim_mono(v)
        for k, v in report["A_monotonicity"].items()  # type: ignore[union-attr]
        if k in HEADLINE_PEERS or k in models
    }
    summary["B_agreement"] = {
        m: {
            c: {
                "pooled": v["pooled_srocc_signed"],
                "within": v["within_ladder_srocc_signed"],
                "ci95": v.get("pooled_ci95"),
            }
            for c, v in cols.items()
            if c in HEADLINE_PEERS
        }
        for m, cols in report["B_agreement"].items()  # type: ignore[union-attr]
    }
    summary["C_cross_codec"] = {
        anchor: {
            col: (
                {
                    "spread": block["spread"],
                    "span_p1_p99": block["span_p1_p99"],
                    "spread_over_span": block["spread_over_span"],
                    **(
                        {
                            "per_codec_offset": {
                                k: v["offset"] for k, v in block["per_codec"].items()
                            },
                            "per_codec_ci95": {
                                k: [v["lo"], v["hi"]]
                                for k, v in block["per_codec"].items()
                            },
                        }
                        if col in models and anchor == "JND_CVVDP"
                        else {}
                    ),
                }
            )
            for col, block in cols.items()
        }
        for anchor, cols in report["C_cross_codec"].items()  # type: ignore[union-attr]
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        # indent=0: one entry per line (diffable) with no indent padding,
        # which is what keeps the committed copy under the repo's 30 KB bar.
        json.dump(round_floats(summary, 4), fh, indent=0, sort_keys=True, default=str)

    write_markdown(args.out_md, report, cropped, models, peer_cols, orient)
    print(f"wrote {args.out_json} and {args.out_md}")


def write_markdown(
    path: str,
    report: Dict[str, object],
    cropped: pd.DataFrame,
    models: List[str],
    peer_cols: List[str],
    orient: Dict[str, float],
) -> None:
    """The markdown carries the headline tables; the JSON carries everything."""
    L: List[str] = []
    a = L.append
    a("# JPEG AIC2026 — metric-agreement panel (2026-09-19)")
    a("")
    a(
        "**AIC2026 ships no human scores.** Every number here is agreement "
        "between objective metrics, or the behaviour of a metric along a "
        "codec's own quality ladder. None of it is accuracy against human "
        "judgment, and it must not be reported as such."
    )
    a("")
    a(
        f"Population: {report['n_rows']} distorted images, "
        f"{report['n_sources']} sources, {report['n_ladders']} "
        f"(source, codec) ladders, {report['n_peer_columns']} shipped metric "
        "columns."
    )
    a("")
    a(
        "**CVVDP placed the distortion levels**, so CVVDP is monotone on these "
        "ladders by construction. It appears in the tables for completeness and "
        "is excluded from any monotonicity ranking."
    )
    a("")

    a("## Findings")
    a("")
    mono = report["A_monotonicity"]  # type: ignore[index]
    order = sorted(models, key=lambda m: -mono[m]["relative"]["pairs_correct_frac"])
    a(
        "* **Ladder monotonicity** (A), best to worst over our six: "
        + ", ".join(
            f"`{m}` {mono[m]['relative']['pairs_correct_frac']:.4f}" for m in order
        )
        + ". `SSIMULACRA2` on the same statistic is "
        f"{mono['SSIMULACRA2']['relative']['pairs_correct_frac']:.4f}."
    )
    cc = report["C_cross_codec"]["JND_CVVDP"]  # type: ignore[index]
    corder = sorted(cc, key=lambda k: cc[k]["spread_over_span"])
    a(
        "* **Cross-codec consistency** (C), spread as a fraction of each "
        "column's own span, most consistent first: "
        + ", ".join(f"`{k}` {cc[k]['spread_over_span']:.3f}" for k in corder)
        + ". `CVVDP` is the anchor the bins are cut on, so its near-zero "
        "figure is a definition, not a result."
    )
    worst_codec = {}
    for m in models:
        pc = cc[m]["per_codec"]
        k = min(pc, key=lambda c: pc[c]["offset"])
        worst_codec[m] = (k, pc[k]["offset"])
    a(
        "* **The largest per-codec offset** for each of our models: "
        + ", ".join(
            f"`{m}` {worst_codec[m][0]} {worst_codec[m][1]:+.1f}" for m in models
        )
        + "."
    )
    par = report["D_parity"]  # type: ignore[index]
    if "ssimulacra2" in par:
        a(
            "* **Implementation parity** (D): our `fast-ssim2` and their "
            f"`SSIMULACRA2` differ by mean |Δ| "
            f"{par['ssimulacra2']['mean_abs_diff']:.4f} "
            f"(max {par['ssimulacra2']['max_abs_diff']:.4f}), SROCC "
            f"{par['ssimulacra2']['srocc']:.6f}."
        )
    if "butteraugli" in par and par["butteraugli"].get("monotone_map"):
        mm = par["butteraugli"]["monotone_map"]
        a(
            f"* Our `{mm['norm']}` butteraugli and their "
            "`proposal-Butteraugli` rank the population identically (Kendall "
            f"{mm['kendall_tau']:.7f}) under a monotone but strongly "
            "nonlinear output mapping — the same computation, different units."
        )
    a("")

    a("## A. Per-ladder monotonicity")
    a("")
    a(
        "A step is *correct* when the oriented score does not rise as the "
        "distortion level rises. `material` counts ladders carrying at least "
        "one inversion larger than the materiality threshold. For our models "
        "the absolute threshold is 0.5 score points; the relative threshold, "
        "applied to every column including ours so the comparison is "
        "apples-to-apples, is 0.5 % of that column's own p1..p99 span on this "
        "population."
    )
    a("")
    a("| column | orientation | correct steps | ladders w/ material inversion (rel.) | ladders w/ inversion > 0.5 pt (abs.) |")
    a("|---|---|--:|--:|--:|")
    mono = report["A_monotonicity"]  # type: ignore[index]
    for c in models:
        r = mono[c]["relative"]
        ab = mono[c]["absolute_0p5"]
        a(
            f"| `{c}` | {report['orientation'][c]} | "  # type: ignore[index]
            f"{fmt(r['pairs_correct_frac'])} | "
            f"{r['ladders_with_material_inversion']}/{r['n_ladders']} | "
            f"{ab['ladders_with_material_inversion']}/{ab['n_ladders']} |"
        )
    for c in HEADLINE_PEERS:
        if c not in mono:
            continue
        r = mono[c]["relative"]
        note = " *(levels selected with it)*" if c.endswith("CVVDP") else ""
        a(
            f"| `{c}`{note} | {report['orientation'][c]} | "  # type: ignore[index]
            f"{fmt(r['pairs_correct_frac'])} | "
            f"{r['ladders_with_material_inversion']}/{r['n_ladders']} | — |"
        )
    a("")

    a("## B. Agreement with the shipped columns")
    a("")
    a(
        "Spearman, sign-normalised by each column's derived orientation, so a "
        "positive number always means agreement. `pooled` is over all rows; "
        "`within` is the mean over the 490 ladders. The full 71-column matrix "
        "is in the JSON."
    )
    a("")
    head = " | ".join(f"`{m}`" for m in models)
    a(f"| peer column | {head} |")
    a("|---" * (len(models) + 1) + "|")
    agr = report["B_agreement"]  # type: ignore[index]
    for c in HEADLINE_PEERS:
        if c not in agr[models[0]]:
            continue
        cells = []
        for m in models:
            e = agr[m][c]
            cells.append(
                f"{fmt(e['pooled_srocc_signed'], 3)} / "
                f"{fmt(e['within_ladder_srocc_signed'], 3)}"
            )
        a(f"| `{c}` (pooled / within) | " + " | ".join(cells) + " |")
    a("")

    a("### B2. By codec family — where hand-built and deep metrics part ways")
    a("")
    a(
        "Mean within-ladder Spearman. The `learned` row (Cool-Chic "
        "Wasserstein/MSE, FTIC) is the cell to read: learned-codec artifacts "
        "are where a hand-built metric and a deep metric are expected to "
        "disagree."
    )
    a("")
    gw = report["B2_group_agreement"]  # type: ignore[index]
    for gname, block in gw.items():
        a(f"**{gname}** (n = {block['_n_rows']})")
        a("")
        cols = [c for c in block[models[0]]]
        a("| model | " + " | ".join(f"`{c}`" for c in cols) + " |")
        a("|---" * (len(cols) + 1) + "|")
        for m in models:
            a(
                f"| `{m}` | "
                + " | ".join(fmt(block[m][c], 3) for c in cols)
                + " |"
            )
        a("")

    a("## C. Cross-codec consistency at matched JND")
    a("")
    a(
        "Within a `JND_CVVDP` bin every row is (by the dataset's own anchor) "
        "the same perceptual distance from its reference. A per-codec offset is "
        "the mean model score for that codec in the bin minus the bin's "
        "all-codec mean, averaged over bins, with a source-level bootstrap CI. "
        "Offsets near zero mean one dial value means the same thing on every "
        "codec; a positive offset means the model scores that codec HIGHER than "
        "the shared JND level implies."
    )
    a("")
    cc = report["C_cross_codec"]["JND_CVVDP"]  # type: ignore[index]
    codecs = sorted({k for m in models for k in cc[m]["per_codec"]})
    a("| codec | " + " | ".join(f"`{m}`" for m in models) + " |")
    a("|---" * (len(models) + 1) + "|")
    for cdc in codecs:
        cells = []
        for m in models:
            e = cc[m]["per_codec"].get(cdc)
            cells.append(
                "—"
                if e is None
                else f"{e['offset']:+.2f} [{e['lo']:+.2f}, {e['hi']:+.2f}]"
            )
        a(f"| {cdc} | " + " | ".join(cells) + " |")
    a("")
    a(
        "Spread of those offsets, as a raw value and as a fraction of the "
        "column's own p1..p99 span — the second is the only one comparable "
        "across columns that are not in the same units. Peer columns are "
        "included so the question is 'is this worse than the alternatives', "
        "not 'is this non-zero'."
    )
    a("")
    a("| column | spread | p1..p99 span | spread / span |")
    a("|---|--:|--:|--:|")
    for k in list(models) + [c for c in cc if not c.startswith("score_")]:
        e = cc[k]
        a(
            f"| `{k}` | {fmt(e['spread'], 3)} | {fmt(e['span_p1_p99'], 3)} | "
            f"{fmt(e['spread_over_span'], 4)} |"
        )
    a("")

    a("## D. Implementation parity")
    a("")
    p = report["D_parity"]  # type: ignore[index]
    if not p:
        a("Not run (no peer score table supplied).")
    else:
        if "ssimulacra2" in p:
            s = p["ssimulacra2"]
            a(
                f"**SSIMULACRA2** — ours ({s['ours']}) vs theirs "
                f"({s['theirs']}), n = {p['n_rows']}: mean |Δ| "
                f"{fmt(s['mean_abs_diff'])}, median |Δ| "
                f"{fmt(s['median_abs_diff'])}, max |Δ| "
                f"{fmt(s['max_abs_diff'])}, SROCC {fmt(s['srocc'], 6)}."
            )
            w = s["worst_row"]
            a("")
            a(
                f"Worst row: `{w['distorted']}` ({w['codec']} level "
                f"{w['level']}) — ours {w['ours']:.4f}, theirs "
                f"{w['theirs']:.4f}."
            )
            a("")
        if "butteraugli" in p:
            b = p["butteraugli"]
            a(
                "**Butteraugli** — the shipped column does not state which norm "
                "it is, so it is identified by agreement:"
            )
            a("")
            a("| our norm | mean abs diff | max abs diff | SROCC |")
            a("|---|--:|--:|--:|")
            for n, e in b["by_norm"].items():
                a(
                    f"| `{n}` | {fmt(e['mean_abs_diff'])} | "
                    f"{fmt(e['max_abs_diff'])} | {fmt(e['srocc'], 6)} |"
                )
            a("")
            a(f"Closest by mean |Δ|: `{b['best_matching_norm_by_mean_abs_diff']}`; "
              f"closest by rank: `{b['best_matching_norm_by_rank']}`.")
            a("")
            mm = b.get("monotone_map")
            if mm:
                a(
                    f"Ordering agreement between our `{mm['norm']}` and their "
                    f"column is effectively exact — Kendall tau "
                    f"{mm['kendall_tau']:.7f}, with "
                    f"{mm['monotonicity_violations']} of {mm['adjacent_pairs']} "
                    f"adjacent pairs out of order (largest "
                    f"{abs(mm['largest_violation']):.2e}, numerical noise). The "
                    "two are therefore the same computation under a monotone "
                    "output mapping, NOT the same units:"
                )
                a("")
                a("| their value | n | median ours / theirs | median ours |")
                a("|---|--:|--:|--:|")
                for bd in mm["ratio_by_band"]:
                    lo, hi = bd["theirs_range"]
                    hi_s = "∞" if hi > 1e8 else f"{hi:g}"
                    a(
                        f"| [{lo:g}, {hi_s}) | {bd['n']} | "
                        f"{bd['median_ours_over_theirs']:.3f} | "
                        f"{bd['median_ours']:.4f} |"
                    )
                a("")

    a("## E. Cropped 840×944 vs full resolution")
    a("")
    e = report["E_crop_vs_fullres"]  # type: ignore[index]
    if not e:
        a("Not run (no full-resolution scores supplied).")
    else:
        a(
            f"{e['n_joined']} stimuli joined; {e['n_cat3_rows']} rows over "
            f"{e['n_cat3_sources']} sources whose full-resolution geometry is "
            "larger than the 840×944 crop (category 3)."
        )
        a("")
        a("| model | cat-3 mean (full − crop) | median | p5..p95 | max abs | SROCC crop↔full |")
        a("|---|--:|--:|--:|--:|--:|")
        for m in models:
            if m not in e:
                continue
            r = e[m]
            rng = r["cat3_p5_p95"]
            a(
                f"| `{m}` | {fmt(r['cat3_mean_full_minus_crop'], 3)} | "
                f"{fmt(r['cat3_median'], 3)} | "
                + (f"{rng[0]:.2f} .. {rng[1]:.2f}" if rng else "n/a")
                + f" | {fmt(r['cat3_max_abs'], 3)} | "
                f"{fmt(r['cat3_srocc_crop_vs_full'], 4)} |"
            )
        a("")

    a("## F. Adding AIC2026 to the board — what it would take")
    a("")
    a(
        "Not implemented, deliberately. The board's axis pipeline is built on a "
        "`human_score` column: `bake_verdict` reads one per registered eval "
        "parquet and emits `rank.<corpus>.srocc_signed`, and `gauntlet.py` "
        "consumes that. AIC2026 has no human target, so wiring it in today "
        "would mean writing `JND_CVVDP` into a column named `human_score` — "
        "a fabricated human label in the plumbing, however it is labelled in "
        "the UI. That is the one thing this corpus must not cause."
    )
    a("")
    a("What a clean implementation needs, in order:")
    a("")
    a(
        "1. A target-role concept at ingestion: `bake_verdict` learns a "
        "`target_kind` (`human` | `metric_agreement`) alongside the target "
        "column, and refuses to call a metric column `human_score`."
    )
    a(
        "2. A third exclusion class in `gauntlet.py` beside `CIRCULAR_AXES` and "
        "`TRAIN_EQ_VAL_AXES` — say `METRIC_AGREEMENT_AXES = [\"aic2026\"]` — "
        "excluded from the composite and from `HELD_OUT_HUMAN_AXES`, rendered "
        "with its own badge. The exclusion machinery already exists and is "
        "already embedded into the page (`circularAxes`), so this part is small."
    )
    a(
        "3. A registered eval root carrying the AIC2026 features plus "
        "`JND_CVVDP` and `JND_SSIMULACRA2` as two named non-human targets, with "
        "the CVVDP caveat (it placed the levels) attached in "
        "`eval_annotations.json`."
    )
    a(
        "4. The axis reported as *agreement*, never as rank quality: a model "
        "scoring higher here agrees more with CVVDP, which is a fact about "
        "CVVDP as much as about the model."
    )
    a("")
    a(
        "Until (1) exists the honest place for these numbers is this document "
        "and the committed JSON beside it."
    )
    a("")

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
