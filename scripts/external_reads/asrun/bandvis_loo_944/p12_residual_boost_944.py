#!/usr/bin/env python3
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/bandvis-loo-2026-07-28/harness/p12_residual_boost_944.py
# sha256(source): 684fa3b6b23330fc4783d532d4937c378c84e499b80d04174b46f8daf8f7ed39
# build_commit:  b1d4bc257e57f7c3215ec8a237e9f87cdad8e35f
# Protocol doc:  benchmarks/bandvis_loo_944_2026-07-28.md
# Everything below the marker line is BYTE-IDENTICAL to the source file
# (verify: strip through the marker, sha256 the rest). Do NOT extend this
# file — it is an archival record of the exact as-run analysis (it may call
# scipy directly; it predates the stats-rule batch migration and is kept
# verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
# Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
# FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
# in the artifact dir are the stored equivalents (see ../README.md).
# ===== byte-identical source below this line ================================
"""P12-style residual-boost of the append2-20 block over the 924-MODEL
residuals — the coordinator's instrument (zensim
scripts/v_next/p12_residual_boost.py, run 2026-07-27) with two labeled
deviations, per the LOO-944 brief:

  1. Stage 1 = the FULL 924 streaming config (f0..f155 ++ f372..f923; the
     f156..f371 fold slots are structurally zero and self-mask via zero
     variance), NOT the 504 config — the brief asks for the marginal value of
     the 20 new columns over the *924-model* residuals.
  2. Stage 2 families = the five append2 slot families (BANDVIS_GAIN,
     BANDVIS_LOSS, LUMA_MEAN_REF, HL_BIN1, HL_BIN2; each 4 scale slots,
     Y-only, f = 924 + scale*5 + local). No E-class members exist in the
     append2 block. HL bins are identically 0 on these SDR legs (their sd
     guard maps them to zero predictors — reported as 0.0 R², structurally).

Everything else is the coordinator's recipe verbatim: V0_2-style BVLS >= 0 on
standardized features, distance-form target, stage-1 fit+frozen on
ext_safesyn_full; stage-2 LINEAR per family, in-sample + deterministic 5-fold
CV R² on the 4 train-legal corpora; holdouts NEVER fit (CID22 MOS ban
absolute) — eval-only column = SROCC(holdout residual, safesyn-fitted family
prediction).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr

D = Path("/mnt/v/zen/zensim-training/ext944-instrument-2026-07-28")
OUTDIR = Path("/mnt/v/output/zensim/bandvis-loo-2026-07-28")
OUT_CSV = OUTDIR / "p12_residual_boost_944_2026-07-28.csv"
OUT_MD = OUTDIR / "p12_residual_boost_944_2026-07-28.md"

N = 944
COLS_924 = list(range(0, 156)) + list(range(372, 924))
A2 = 924

TRAIN_FIT = ["ext_safesyn_full", "ext_cid22_train201", "ext_kadid", "ext_tid"]
HOLDOUT = [
    "ext_cid22val", "ext_aic3", "ext_aic4", "ext_csiq", "ext_live",
    "ext_konjnd_jpeg_val", "ext_sdr25",
]

FAMILIES = {
    "BANDVIS_GAIN": [0],
    "BANDVIS_LOSS": [1],
    "LUMA_MEAN_REF": [2],
    "HL_BIN1": [3],
    "HL_BIN2": [4],
    "BANDVIS_pair(gain+loss)": [0, 1],
    "append2_all": [0, 1, 2, 3, 4],
}


def fam_cols(slots: list[int]) -> list[int]:
    return [A2 + s * 5 + k for s in range(4) for k in slots]


def load(corpus: str) -> tuple[np.ndarray, np.ndarray]:
    t = pq.read_table(D / f"{corpus}.parquet")
    X = np.column_stack(
        [np.asarray(t[f"f{i}"].combine_chunks().to_numpy(), np.float64) for i in range(N)]
    )
    y = np.asarray(t["human_score"].combine_chunks().to_numpy(), np.float64)
    return X, np.max(y) - y  # distance form: high = bad, matches >=0-coef BVLS


def std_fit(X: np.ndarray):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    return mu, sd


def r2(y: np.ndarray, p: np.ndarray) -> float:
    ss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - p) ** 2).sum()) / ss if ss > 0 else 0.0


def lin_fit(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(np.hstack([A, np.ones((len(y), 1))]), y, rcond=None)[0]


def cv_r2(A: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    n = len(y)
    idx = np.arange(n) % k  # deterministic interleave, no RNG
    preds = np.empty(n)
    for f in range(k):
        tr, te = idx != f, idx == f
        w = lin_fit(A[tr], y[tr])
        preds[te] = np.hstack([A[te], np.ones((te.sum(), 1))]) @ w
    return r2(y, preds)


def main() -> None:
    print("stage 1: BVLS >=0 on the 924 config, ext_safesyn_full ...", flush=True)
    Xs, ys = load("ext_safesyn_full")
    X9 = Xs[:, COLS_924]
    mu, sd = std_fit(X9)
    A = np.hstack([(X9 - mu) / sd, np.ones((len(ys), 1))])
    lo = np.zeros(A.shape[1])
    lo[-1] = -np.inf
    res = lsq_linear(A, ys, bounds=(lo, np.full(A.shape[1], np.inf)), max_iter=300)
    w924 = res.x
    active = int((w924[:-1] > 1e-9).sum())
    s1 = A @ w924
    print(f"  active={active}/{len(COLS_924)}  train R2={r2(ys, s1):.4f}  "
          f"SROCC={spearmanr(ys, s1).statistic:.4f}", flush=True)

    def stage1_pred(X: np.ndarray) -> np.ndarray:
        return np.hstack([(X[:, COLS_924] - mu) / sd, np.ones((len(X), 1))]) @ w924

    fam_models: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rows = []
    for corpus in TRAIN_FIT:
        X, y = (Xs, ys) if corpus == "ext_safesyn_full" else load(corpus)
        resid = y - stage1_pred(X)
        base_srocc = spearmanr(y, stage1_pred(X)).statistic
        for fam, slots in FAMILIES.items():
            cols = fam_cols(slots)
            F = X[:, cols]
            fmu, fsd = std_fit(F)
            Fs = (F - fmu) / fsd
            w = lin_fit(Fs, resid)
            pred = np.hstack([Fs, np.ones((len(resid), 1))]) @ w
            rows.append(
                dict(corpus=corpus, family=fam, kind="fit", n=len(y),
                     stage1_srocc=round(float(base_srocc), 4),
                     r2_in=round(r2(resid, pred), 5),
                     r2_cv5=round(cv_r2(Fs, resid), 5))
            )
            if corpus == "ext_safesyn_full":
                fam_models[fam] = (fmu, fsd, w)
    for corpus in HOLDOUT:
        X, y = load(corpus)
        p1 = stage1_pred(X)
        resid = y - p1
        base_srocc = spearmanr(y, p1).statistic
        for fam, slots in FAMILIES.items():
            cols = fam_cols(slots)
            fmu, fsd, w = fam_models[fam]
            pred = np.hstack([(X[:, cols] - fmu) / fsd, np.ones((len(y), 1))]) @ w
            sr = spearmanr(resid, pred).statistic if len(y) > 3 else float("nan")
            rows.append(
                dict(corpus=corpus, family=fam, kind="eval-only", n=len(y),
                     stage1_srocc=round(float(base_srocc), 4),
                     r2_in=float("nan"),
                     r2_cv5=round(float(sr), 4))
            )

    import csv as _csv
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        wtr = _csv.DictWriter(f, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)

    fit_rows = [r_ for r_ in rows if r_["kind"] == "fit"]
    agg: dict[str, float] = {}
    for fam in FAMILIES:
        vals = [r_["r2_cv5"] for r_ in fit_rows if r_["family"] == fam]
        agg[fam] = float(np.mean(vals))
    ranked = sorted(agg.items(), key=lambda kv: -kv[1])
    with open(OUT_MD, "w") as f:
        f.write("# P12-at-924 residual-boost of append2 — 2026-07-28\n\n")
        f.write(f"Stage 1: BVLS >=0 on the FULL 924 config, ext_safesyn_full train "
                f"(active {active}/{len(COLS_924)}, train R2 {r2(ys, s1):.4f}) — "
                "labeled variant of p12_residual_boost.py (stage-1 = 924, not 504).\n"
                "Stage 2: per-append2-family LINEAR on residuals. HL bins are "
                "identically 0 on SDR legs (structural).\n"
                "Holdouts NEVER fit; eval-only column = SROCC(holdout residual, "
                "safesyn-fitted family prediction).\n\n")
        f.write("## Ranked marginal value (mean CV-R2 across the 4 train-legal corpora)\n\n")
        f.write("| rank | append2 family | mean CV-R2 |\n|---|---|---|\n")
        for i, (fam, v) in enumerate(ranked, 1):
            f.write(f"| {i} | {fam} | {v:.5f} |\n")
        f.write(f"\nFull per-corpus table: `{OUT_CSV.name}` (holdout rows eval-only).\n")
    print(f"wrote {OUT_CSV} + {OUT_MD}")
    for i, (fam, v) in enumerate(ranked, 1):
        print(f"  {i}. {fam}: {v:.5f}")


if __name__ == "__main__":
    main()
