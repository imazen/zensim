#!/usr/bin/env python3
"""Stub generator AND executable schema-contract for the ``*.fulleval.json`` files that
``bandwise_dashboard.py --fulleval-dir`` renders into the interactive summer-gauntlet page.

WHY THIS FILE EXISTS
--------------------
The full-eval JSONs are produced by the eval/discovery session (bake_verdict + the canonical
`panel` binary over the held-out feature parquets + the reference-metric TSVs). The DASHBOARD
session builds the viz *against the schema* before those real JSONs exist. This module is the
single source of truth for that schema — it both (a) documents every field the dashboard reads
and (b) emits realistic fixtures so the viz can be built + tested now. When the real full-eval
runs, it MUST emit the SAME shape (see ``FULLEVAL_SCHEMA`` below); drop the real JSONs into
``--fulleval-dir`` and the dashboard re-renders unchanged.

STATS ARE CANONICAL, NEVER HAND-ROLLED
--------------------------------------
Every SROCC/PLCC/etc. in the emitted JSON comes from ``scripts/lib/zen_stats.panel`` — the thin
shim over the Rust ``panel`` binary that ``bake_verdict`` uses (no scipy/hand-rolled math). The
real full-eval must do the same. Point ``$ZEN_PANEL_BIN`` at a built ``panel`` if ``target/`` is
not populated in this checkout::

    ZEN_PANEL_BIN=/path/to/target/release/panel python3 scripts/v_next/make_stub_fulleval.py

Usage::

    make_stub_fulleval.py [--out-dir /mnt/v/output/zensim/reports/fulleval]
                          [--best-per-day /mnt/v/output/zensim/reports/best_per_day.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.zen_stats import panel  # canonical stats — do NOT hand-roll

# --------------------------------------------------------------------------- schema doc
FULLEVAL_SCHEMA = {
    "bake": "absolute path to the .bin (str)",
    "name": "short display label (str)",
    "regime": "feature regime / arch tag, e.g. 'linear-372' | 'mlp-372' | 'v2-720' (str)",
    "date": "champion date YYYY-MM-DD (str, optional — best_per_day carries it too)",
    "rank": {
        "<corpus>": {
            "n": "int", "srocc": "float", "plcc": "float", "krocc": "float",
            "or": "float (outlier ratio)", "pwrc": "float", "z_rmse": "float",
            # SROCC of pred vs THIS corpus's PRIMARY human/target reference (MOS, JND, or —
            # for the ssim2-anchored non-photo corpus — ssim2). Polarity handled upstream:
            # report abs() for structural-negative JND corpora (dashboard just displays it).
        }
    },
    "dial": {
        "mono_pct": "float 0..1 — monotonicity (1 - inversions) on the densified multi-codec grid",
        "tied_pct": "float 0..1 — fraction of adjacent-q pairs tied (dead-zone)",
        "reach": "float — dial span (p95 - p5) in score points",
        "dynamic_range": "float — pooled p95 (higher end reached); or a {p5,p95} pair",
    },
    "m3_coherence": "float 0..1 — diffmap<->scalar coherence (M3). 1.0 = diffmap reflects scalar",
    "corruption": {
        "detection_t50": "float 0..1 — corruption detection rate at 50% threshold (higher=better)",
        "fp_severe_honest": "float 0..1 — false-positive rate on severe-but-honest inputs (lower=better)",
        "fp_broad_honest": "float 0..1 — false-positive rate on broad honest inputs (lower=better)",
    },
    "per_pair": {
        "<corpus>": {
            "pred": "[float] — bake prediction per pair (REQUIRED)",
            "mos": "[float] — human MOS reference (present for MOS corpora)",
            "jnd": "[float] — human JND/PJND reference (present for JND corpora)",
            "ssim2": "[float] — SSIMULACRA2 reference (optional)",
            "butter": "[float] — butteraugli, higher=better (NEGATED at source) (optional)",
            "cvvdp": "[float] — ColorVideoVDP reference (optional)",
            # arrays are row-aligned; a missing reference key = that reference not computed for
            # this corpus. The dashboard subsamples for embedding, so full-length arrays are fine.
        }
    },
    "scatter": {
        "<corpus>": {
            "<reference>": {  # reference in {mos, jnd, ssim2, butter, cvvdp}
                "srocc": "float — canonical panel SROCC of (pred, reference)",
                "plcc": "float", "n": "int",
                # OPTIONAL but recommended: precomputed by the eval agent via the canonical
                # panel so the dashboard never re-runs O(n^2) PWRC per cell. If absent, the
                # dashboard computes srocc/plcc via the same panel at build time.
            }
        }
    },
}

# corpus -> (primary human reference key or None if ssim2-anchored, [metric reference keys])
CORPUS_REFS = {
    "cid22":    ("mos", ["ssim2", "butter", "cvvdp"]),
    "kadid":    ("mos", ["ssim2", "butter", "cvvdp"]),
    "tid":      ("mos", ["ssim2", "butter", "cvvdp"]),
    "konjnd":   ("jnd", ["ssim2", "butter", "cvvdp"]),
    "aic3":     ("jnd", ["ssim2", "butter", "cvvdp"]),
    "aic4":     ("jnd", []),
    "nonphoto": (None,  ["ssim2"]),   # ssim2 is the anchor/target here
    "live":     ("mos", ["ssim2", "butter", "cvvdp"]),
    "csiq":     ("mos", ["ssim2", "butter", "cvvdp"]),
}
CORPUS_N = {"cid22": 800, "kadid": 500, "tid": 400, "konjnd": 500, "aic3": 400,
            "aic4": 300, "nonphoto": 500, "live": 400, "csiq": 400}

# Four realistic summer champions (values modelled on SESSION-RESUME + a real cid22.verdict.md;
# these are FIXTURES to build the viz — the eval agent overwrites them with measured JSONs).
# per-bake: regime, date, target CID22 human-SROCC, and per-corpus srocc offsets, dial, m3, corruption.
STUBS = [
    dict(name="B_shipped", regime="linear-372", date="2026-07-07",
         srocc=dict(cid22=0.857, kadid=0.90, tid=0.88, konjnd=0.55, aic3=0.61, aic4=0.58,
                    nonphoto=0.881, live=0.94, csiq=0.90),
         dial=dict(mono_pct=0.985, tied_pct=0.031, reach=5.6, dynamic_range=50.1),
         m3_coherence=0.62,
         corruption=dict(detection_t50=0.196, fp_severe_honest=0.11, fp_broad_honest=0.04)),
    dict(name="Ebothg_scr0.5_dial", regime="mlp-372", date="2026-07-18",
         srocc=dict(cid22=0.879, kadid=0.905, tid=0.885, konjnd=0.27, aic3=0.66, aic4=0.60,
                    nonphoto=0.906, live=0.959, csiq=0.917),
         dial=dict(mono_pct=0.985, tied_pct=0.028, reach=6.1, dynamic_range=51.0),
         m3_coherence=0.88,
         corruption=dict(detection_t50=0.61, fp_severe_honest=0.07, fp_broad_honest=0.03)),
    dict(name="winner_dial", regime="mlp-372", date="2026-07-16",
         srocc=dict(cid22=0.894, kadid=0.91, tid=0.89, konjnd=0.34, aic3=0.63, aic4=0.59,
                    nonphoto=0.902, live=0.955, csiq=0.915),
         dial=dict(mono_pct=0.981, tied_pct=0.030, reach=6.0, dynamic_range=50.7),
         m3_coherence=0.72,
         corruption=dict(detection_t50=0.42, fp_severe_honest=0.09, fp_broad_honest=0.05)),
    dict(name="e2_ext720_s1", regime="v2-720", date="2026-07-23",
         srocc=dict(cid22=0.865, kadid=0.90, tid=0.88, konjnd=0.49, aic3=0.72, aic4=0.63,
                    nonphoto=0.898, live=0.951, csiq=0.905),
         dial=dict(mono_pct=0.983, tied_pct=0.033, reach=5.9, dynamic_range=50.6),
         m3_coherence=1.00,
         corruption=dict(detection_t50=0.55, fp_severe_honest=0.08, fp_broad_honest=0.03)),
]


def _synth_pair(rng, n, target_srocc, hi=100.0):
    """Generate (reference, pred) with approx target Spearman via a Gaussian copula.
    reference is mapped onto [0,hi]; pred stays z-space (like the real MLP candidates)."""
    ts = min(0.985, max(0.05, target_srocc))
    rho = 2.0 * np.sin(np.pi * ts / 6.0)                 # Spearman -> Pearson for a Gaussian copula
    z1 = rng.standard_normal(n)
    z2 = rho * z1 + np.sqrt(max(0.0, 1.0 - rho * rho)) * rng.standard_normal(n)
    ref = (z1 - z1.min()) / (np.ptp(z1) or 1) * hi       # anchor on [0,hi]
    pred = z2 * 1.4 - 0.2                                 # z-ish pred scale
    return ref, pred


def build_bake(stub, out_dir):
    rng = np.random.default_rng(abs(hash(stub["name"])) % (2**32))
    per_pair, rank, scatter = {}, {}, {}
    for corp, (human_key, metric_keys) in CORPUS_REFS.items():
        if corp not in stub["srocc"]:
            continue
        n = CORPUS_N[corp]
        tgt = stub["srocc"][corp]
        # anchor reference: MOS/JND if human corpus, else ssim2
        anchor_key = human_key or "ssim2"
        ref, pred = _synth_pair(rng, n, tgt, hi=100.0)
        cols = {"pred": pred.round(4).tolist()}
        cols[anchor_key] = ref.round(4).tolist()
        # metric references: track the same latent quality as the anchor, each with its own
        # noise -> realistic pred-vs-metric spread (metrics agree with the bake, imperfectly).
        for mk in metric_keys:
            if mk == anchor_key:
                continue
            mnoise = rng.uniform(0.12, 0.30)
            mref = ref + rng.normal(0, mnoise * 100, n)
            cols[mk] = np.round(mref, 3).tolist()
        per_pair[corp] = cols

        # --- canonical stats: rank block (pred vs anchor) + per-(corpus,ref) scatter block ---
        anchor = np.asarray(cols[anchor_key], float)
        pr = np.asarray(cols["pred"], float)
        pn = panel(pr.tolist(), anchor.tolist())
        rank[corp] = {"n": int(pn["n"]), "srocc": round(abs(pn["srocc"]), 4),
                      "plcc": round(pn["plcc"], 4), "krocc": round(abs(pn["krocc"]), 4),
                      "or": round(pn["or"], 4), "pwrc": round(abs(pn["pwrc"]), 4),
                      "z_rmse": round(pn["z_rmse"], 3)}
        sc = {}
        for rk in [anchor_key] + [m for m in metric_keys if m != anchor_key]:
            rr = np.asarray(cols[rk], float)
            p2 = panel(pr.tolist(), rr.tolist())
            sc[rk] = {"srocc": round(abs(p2["srocc"]), 4), "plcc": round(p2["plcc"], 4),
                      "n": int(p2["n"])}
        scatter[corp] = sc

    obj = {
        "bake": f"/mnt/v/output/zensim/bakes/{stub['name']}.bin",
        "name": stub["name"], "regime": stub["regime"], "date": stub["date"],
        "rank": rank, "dial": stub["dial"], "m3_coherence": stub["m3_coherence"],
        "corruption": stub["corruption"], "per_pair": per_pair, "scatter": scatter,
        "_stub": True, "_schema": "fulleval/v1",
    }
    p = out_dir / f"{stub['name']}.fulleval.json"
    p.write_text(json.dumps(obj))
    return obj, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/mnt/v/output/zensim/reports/fulleval")
    ap.add_argument("--best-per-day", default="/mnt/v/output/zensim/reports/best_per_day.json")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best = []
    for stub in STUBS:
        obj, p = build_bake(stub, out_dir)
        kb = p.stat().st_size / 1024
        cid = obj["rank"].get("cid22", {}).get("srocc")
        print(f"  wrote {p.name}  ({kb:.0f} KB)  CID22 srocc≈{cid}")
        best.append({"date": stub["date"], "name": stub["name"], "bake_path": obj["bake"],
                     "regime": stub["regime"]})
    best.sort(key=lambda r: r["date"])
    Path(a.best_per_day).write_text(json.dumps(best, indent=2))
    print(f"  wrote {a.best_per_day}  ({len(best)} champions)")
    print("\nschema keys:", ", ".join(FULLEVAL_SCHEMA.keys()))


if __name__ == "__main__":
    main()
