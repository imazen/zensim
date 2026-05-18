"""Compute Mohammadi 2025 iwssim panels for CID22 / KADID / TID /
KonJND-1k from per-pair iwssim score TSVs produced by
`zen-metrics batch --metric iwssim-gpu`.

The score TSV layout is `ref_path\tdist_path\tiwssim_gpu` — one row
per scored pair. This script joins each corpus's score TSV to the
canonical human-MOS loaders in `extract_panels.py` (which use the
SAME pairs TSVs we scored against, so the row order is preserved
modulo skipped rows). It emits aggregate + 10-band Mohammadi panels
per corpus and writes markdown patch content to stdout.

Inputs (one TSV per corpus, header `ref_path\tdist_path\tiwssim_gpu`):
- CID22: /mnt/v/zen/zensim-training/2026-05-18-iwssim-panels/cid22_iwssim_scores.tsv
- KADID: .../kadid_iwssim_scores.tsv
- TID:   .../tid_iwssim_scores.tsv
- KonJND_JPEG / KonJND_BPG: .../konjnd_{jpeg,bpg}_iwssim_scores.tsv
- AIC3:  .../aic3_iwssim_scores.tsv (optional anchor cross-check)

Outputs:
- /tmp/iwssim_panels_<corpus>.json — aggregate + 10-band stats
- stdout: markdown patch rows for `baseline_panels_2026-05-18.md`
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from panel import PanelStats, compute_panel  # noqa: E402
from extract_panels import (  # noqa: E402
    CVVDP_SCORE_TSVS,
    CID22_MOS_CSV,
    KADID_DMOS_CSV,
    TID_MOS_TXT,
    TID_MOS_STD_TXT,
    KONJND_SUBJ_CSV,
    AIC3_ANCHOR_CSV,
    load_konjnd_cvvdp,
    per_band_panel,
)


# ---------------------------------------------------------------------------
# Iwssim score TSV paths (produced by zen-metrics batch --metric iwssim-gpu)
# ---------------------------------------------------------------------------

IWSSIM_TSVS = {
    "CID22": "/mnt/v/zen/zensim-training/2026-05-18-iwssim-panels/cid22_iwssim_scores.tsv",
    "KADID": "/mnt/v/zen/zensim-training/2026-05-18-iwssim-panels/kadid_iwssim_scores.tsv",
    "TID": "/mnt/v/zen/zensim-training/2026-05-18-iwssim-panels/tid_iwssim_scores.tsv",
    "KonJND_JPEG": "/mnt/v/zen/zensim-training/2026-05-18-iwssim-panels/konjnd_jpeg_iwssim_scores.tsv",
    "KonJND_BPG": "/mnt/v/zen/zensim-training/2026-05-18-iwssim-panels/konjnd_bpg_iwssim_scores.tsv",
    "AIC3": "/mnt/v/zen/zensim-training/2026-05-18-iwssim-panels/aic3_iwssim_scores.tsv",
}


def load_iwssim_tsv(path: str) -> dict:
    """Read a `(ref_path, dist_path, iwssim_gpu)` TSV. Returns
    dict[dist_path] -> iwssim score.
    """
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            v = r.get("iwssim_gpu")
            if v is None or v == "":
                continue
            try:
                out[r["dist_path"]] = float(v)
            except (KeyError, ValueError, TypeError):
                continue
    return out


# ---------------------------------------------------------------------------
# Per-corpus loaders. Each returns (humans, iwssim, sigma_or_None).
# The dist-path key matches the canonical human-MOS loader in
# extract_panels.py so we can re-use band slicing.
# ---------------------------------------------------------------------------


def load_cid22() -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    iwssim_by = load_iwssim_tsv(IWSSIM_TSVS["CID22"])
    humans, iwssim = [], []
    misses = 0
    with open(CID22_MOS_CSV) as f:
        for r in csv.DictReader(f):
            dist_rel = r["distorted_img"]
            dist_full = os.path.join(
                "/mnt/v/dataset/cid22/CID22_validation_set", dist_rel
            )
            if dist_full not in iwssim_by:
                misses += 1
                continue
            humans.append(float(r["MCOS"]) / 100.0)
            iwssim.append(iwssim_by[dist_full])
    return np.array(humans), np.array(iwssim), None, misses


def load_kadid() -> tuple[np.ndarray, np.ndarray, None, int]:
    iwssim_by = load_iwssim_tsv(IWSSIM_TSVS["KADID"])
    humans, iwssim = [], []
    misses = 0
    with open(KADID_DMOS_CSV) as f:
        for r in csv.DictReader(f):
            dist_path = f"/mnt/v/dataset/kadid10k/images/{r['dist_img']}"
            if dist_path not in iwssim_by:
                misses += 1
                continue
            humans.append((float(r["dmos"]) - 1.0) / 4.0)
            iwssim.append(iwssim_by[dist_path])
    return np.array(humans), np.array(iwssim), None, misses


def load_tid() -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    """TID: re-implement the basename joining from load_tid_cvvdp."""
    mos_by_basename = {}
    sigma_by_basename = {}
    with open(TID_MOS_TXT) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            m = float(parts[0])
            bn = parts[1].lower()
            mos_by_basename[bn] = m
    with open(TID_MOS_STD_TXT) as f:
        stds = [float(line.strip()) for line in f if line.strip()]
    with open(TID_MOS_TXT) as f:
        names = [line.split()[1].lower() for line in f if line.strip()]
    for bn, s in zip(names, stds):
        sigma_by_basename[bn] = s

    iwssim_by_base = {}
    iwssim_by = load_iwssim_tsv(IWSSIM_TSVS["TID"])
    for dist_path, v in iwssim_by.items():
        base = os.path.basename(dist_path).lower()
        # TID pairs TSV uses .png filenames produced from .BMP — match
        # the cvvdp loader's normalization
        if base.endswith(".png"):
            base = base[:-4] + ".bmp"
        iwssim_by_base[base] = v
    humans, iwssim, sigma = [], [], []
    misses = 0
    for bn, m in mos_by_basename.items():
        if bn not in iwssim_by_base:
            misses += 1
            continue
        humans.append(m / 9.0)
        iwssim.append(iwssim_by_base[bn])
        sigma.append(sigma_by_basename.get(bn, float("nan")) / 9.0)
    return np.array(humans), np.array(iwssim), np.array(sigma), misses


def load_konjnd() -> tuple[np.ndarray, np.ndarray, None, int]:
    """KonJND-1k: join 1008 PJND-threshold pairs from
    subjective_ratings.csv. Mirrors load_konjnd_cvvdp logic but for
    iwssim TSVs.
    """
    pjnd_rows = []
    with open(KONJND_SUBJ_CSV) as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 4:
                continue
            try:
                t = float(parts[3])
            except ValueError:
                continue
            pjnd_rows.append((parts[0], parts[1], t))
    iwssim_lookup = {}
    for codec, path in [("JPEG", IWSSIM_TSVS["KonJND_JPEG"]),
                         ("BPG", IWSSIM_TSVS["KonJND_BPG"])]:
        iwssim_lookup.update(load_iwssim_tsv(path))
    humans, iwssim = [], []
    misses = 0
    base = "/mnt/v/datasets/KonJND-1k/KonJND-1k"
    for image_id, comp, t in pjnd_rows:
        stem = image_id.replace(".png", "")
        level = max(1, min(100, round(t)))
        if comp == "JPEG":
            dist = f"{base}/jpeg/{stem}_JPEG_{level:03d}.jpg"
        elif comp == "BPG":
            dist = f"{base}/bpg/{stem}_BPG_{level:03d}.png"
        else:
            continue
        if dist not in iwssim_lookup:
            misses += 1
            continue
        humans.append(t)
        iwssim.append(iwssim_lookup[dist])
    return np.array(humans), np.array(iwssim), None, misses


def load_aic3() -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    """AIC-3 anchor n=600 sweep: humans = `distortion` from anchor CSV
    (per-pair sigma_bootstrap). Joined by basename like cvvdp loader.
    """
    iwssim_by = load_iwssim_tsv(IWSSIM_TSVS["AIC3"])
    iwssim_by_base = {os.path.basename(p): v for p, v in iwssim_by.items()}
    rows = list(csv.DictReader(open(AIC3_ANCHOR_CSV)))
    humans, iwssim, sigma = [], [], []
    misses = 0
    for r in rows:
        bn = r["image_filename"]
        if bn not in iwssim_by_base:
            misses += 1
            continue
        humans.append(float(r["distortion"]))
        iwssim.append(iwssim_by_base[bn])
        sigma.append(float(r["std_bootstrap"]))
    return np.array(humans), np.array(iwssim), np.array(sigma), misses


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def fmt_stat(v, digits=4):
    if v is None or (isinstance(v, float) and (np.isnan(v) or not np.isfinite(v))):
        return "n/a"
    return f"{v:.{digits}f}"


def panel_to_row(label: str, ps: PanelStats | None) -> str:
    if ps is None or ps.n == 0:
        return f"| {label} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
    return (
        f"| {label} | {ps.n} | "
        f"{fmt_stat(ps.srocc)} | {fmt_stat(ps.plcc)} | {fmt_stat(ps.krocc)} | "
        f"{fmt_stat(ps.or_ratio)} | {fmt_stat(ps.pwrc)} | {fmt_stat(ps.z_rmse, 3)} |"
    )


def emit_per_band(label: str, bands: list) -> str:
    lines = [f"#### {label}\n",
             "| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |",
             "|---|---|--:|---:|---:|---:|---:|---:|---:|"]
    for blabel, rng, n, ps in bands:
        if ps is None:
            lines.append(f"| {blabel} | {rng} | {n} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        flag = " ⚠" if ps.n < 30 else ""
        lines.append(
            f"| {blabel}{flag} | {rng} | {ps.n} | "
            f"{fmt_stat(ps.srocc)} | {fmt_stat(ps.plcc)} | {fmt_stat(ps.krocc)} | "
            f"{fmt_stat(ps.or_ratio)} | {fmt_stat(ps.pwrc)} | {fmt_stat(ps.z_rmse, 3)} |"
        )
    return "\n".join(lines) + "\n"


def main():
    out = {}
    print("# iwssim panels (computed 2026-05-18)\n")

    # --- CID22 ---
    if os.path.exists(IWSSIM_TSVS["CID22"]):
        h, s, sig, misses = load_cid22()
        ps = compute_panel(s, h, sig)
        out["CID22"] = {
            "n": len(h), "matched": len(h), "misses": misses,
            "aggregate": ps.__dict__,
        }
        print(f"## CID22 (n={len(h)}, misses={misses})\n")
        print(panel_to_row("iwssim", ps))
        bands = per_band_panel(s, h, sig)
        out["CID22"]["bands"] = [
            {"label": b[0], "range": b[1], "n": b[2],
             "stats": b[3].__dict__ if b[3] else None}
            for b in bands
        ]
        print()
        print(emit_per_band("iwssim (CID22)", bands))

    # --- KADID ---
    if os.path.exists(IWSSIM_TSVS["KADID"]):
        h, s, sig, misses = load_kadid()
        ps = compute_panel(s, h, sig)
        out["KADID"] = {
            "n": len(h), "matched": len(h), "misses": misses,
            "aggregate": ps.__dict__,
        }
        print(f"## KADID-10k (n={len(h)}, misses={misses})\n")
        print(panel_to_row("iwssim", ps))
        bands = per_band_panel(s, h, sig)
        out["KADID"]["bands"] = [
            {"label": b[0], "range": b[1], "n": b[2],
             "stats": b[3].__dict__ if b[3] else None}
            for b in bands
        ]
        print()
        print(emit_per_band("iwssim (KADID-10k)", bands))

    # --- TID ---
    if os.path.exists(IWSSIM_TSVS["TID"]):
        h, s, sig, misses = load_tid()
        # Per the cvvdp loader's TID note: per-stim mos_std contains
        # zeros / near-zeros that blow up per-sample normalization.
        # Use corpus-wide σ (sig=None) for Z-RMSE.
        ps = compute_panel(s, h, None)
        out["TID"] = {
            "n": len(h), "matched": len(h), "misses": misses,
            "aggregate": ps.__dict__, "sigma": "corpus-wide",
        }
        print(f"## TID2013 (n={len(h)}, misses={misses})\n")
        print(panel_to_row("iwssim", ps))
        bands = per_band_panel(s, h, None)
        out["TID"]["bands"] = [
            {"label": b[0], "range": b[1], "n": b[2],
             "stats": b[3].__dict__ if b[3] else None}
            for b in bands
        ]
        print()
        print(emit_per_band("iwssim (TID2013)", bands))

    # --- KonJND ---
    if (os.path.exists(IWSSIM_TSVS["KonJND_JPEG"]) and
        os.path.exists(IWSSIM_TSVS["KonJND_BPG"])):
        h, s, sig, misses = load_konjnd()
        ps = compute_panel(s, h, None)
        out["KonJND"] = {
            "n": len(h), "matched": len(h), "misses": misses,
            "aggregate": ps.__dict__,
        }
        print(f"## KonJND-1k (n={len(h)}, misses={misses})\n")
        print(panel_to_row("iwssim", ps))

    # --- AIC-3 (cross-check) ---
    if os.path.exists(IWSSIM_TSVS["AIC3"]):
        h, s, sig, misses = load_aic3()
        ps = compute_panel(s, h, sig)
        out["AIC3"] = {
            "n": len(h), "matched": len(h), "misses": misses,
            "aggregate": ps.__dict__,
        }
        print(f"## AIC-3 (n={len(h)}, misses={misses}) — cross-check\n")
        print(panel_to_row("iwssim", ps))

    Path("/tmp/iwssim_panels.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote /tmp/iwssim_panels.json")


if __name__ == "__main__":
    main()
