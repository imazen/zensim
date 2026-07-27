#!/usr/bin/env python3
"""P12 — RESIDUAL-BOOST instrument (zensim-bake-subset-plans-2026-07-26).

Stage 1: fit the 504 config (f0..f155 folded v1-basic ++ f372..f719 v2-348;
the f156..f371 fold slots are structurally zero) with the V0_2-style BVLS
recipe — standardized features, distance-form target, all-coef ≥ 0 — on the
ext924 canonical TRAIN mass (ext_safesyn_full). Freeze it.

Stage 2: per corpus, residual = y_dist − stage1_pred; fit each APPEND FAMILY
(f720..f923 slots, minus E-class) alone on the residuals — LINEAR least
squares — and report R² (in-sample + 5-fold CV). The per-family R² IS the
marginal information of the family, immune to twin collinearity by
construction (P12's point).

Ban hygiene (manifest roles are authoritative):
  * Stage-2 FITS run only on TRAIN-legal corpora: safesyn_full, cid22_train201,
    kadid (guard), tid (guard).
  * HOLDOUTS (cid22val MOS-ban-absolute, aic3, aic4, csiq, live, konjnd, sdr25)
    are NEVER fit on: we apply the safesyn-fitted family models and report the
    Spearman of family-prediction vs holdout residual (generalizing signal only).

E-class exclusions (doc §0): E1 GRAD_SRC_MEAN(16) — exact alias of
1−PJND_FRAGILITY; E2 LUM_MID_ERR(3) — derived partition bin; E3 ART_DEV2(11),
DET_DEV2(12) — exact functions of v1-basic norms.

Output: benchmarks/p12_residual_boost_<date>.csv + .md (ranked table).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr

D = Path("/mnt/v/zen/zensim-training/ext924-canonical-2026-07-27")
REPO = Path(__file__).resolve().parent.parent.parent
OUT_CSV = REPO / f"benchmarks/p12_residual_boost_{date.today()}.csv"
OUT_MD = REPO / f"benchmarks/p12_residual_boost_{date.today()}.md"

N = 924
COLS_504 = list(range(0, 156)) + list(range(372, 720))
APP_BASE, SCALES, CH, PER = 720, 4, 3, 17

TRAIN_FIT = ["ext_safesyn_full", "ext_cid22_train201", "ext_kadid", "ext_tid"]
HOLDOUT = [
    "ext_cid22val", "ext_aic3", "ext_aic4", "ext_csiq", "ext_live",
    "ext_konjnd_jpeg_val", "ext_sdr25",
]

FAMILIES = {
    "XMASK_TRANSDUCER": [0],
    "LUM_TRANSDUCER": [1],
    "LUM_BINS(dark+bright)": [2, 4],
    "MSCN_DIFF": [5, 6],
    "CONTRAST_GAIN/LOSS": [7, 8],
    "TEXTURE_DISSIM": [9],
    "GMS_DEV2": [10],
    "GLOBAL(dmean+cgain+closs)": [13, 14, 15],
}
E_CLASS = {3, 11, 12, 16}  # excluded everywhere


def fam_cols(slots: list[int]) -> list[int]:
    return [
        APP_BASE + s * CH * PER + c * PER + k
        for s in range(SCALES)
        for c in range(CH)
        for k in slots
    ]


def load(corpus: str) -> tuple[np.ndarray, np.ndarray]:
    t = pq.read_table(D / f"{corpus}.parquet")
    X = np.column_stack(
        [np.asarray(t[f"f{i}"].combine_chunks().to_numpy(), np.float64) for i in range(N)]
    )
    y = np.asarray(t["human_score"].combine_chunks().to_numpy(), np.float64)
    return X, np.max(y) - y  # distance form: high = bad, matches ≥0-coef BVLS


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
    print("stage 1: BVLS 504 on ext_safesyn_full ...")
    Xs, ys = load("ext_safesyn_full")
    X5 = Xs[:, COLS_504]
    mu, sd = std_fit(X5)
    A = np.hstack([(X5 - mu) / sd, np.ones((len(ys), 1))])
    lo = np.zeros(A.shape[1])
    lo[-1] = -np.inf
    res = lsq_linear(A, ys, bounds=(lo, np.full(A.shape[1], np.inf)), max_iter=300)
    w504 = res.x
    active = int((w504[:-1] > 1e-9).sum())
    s1 = A @ w504
    print(f"  active={active}/504  train R²={r2(ys, s1):.4f}  SROCC={spearmanr(ys, s1).statistic:.4f}")

    def stage1_pred(X: np.ndarray) -> np.ndarray:
        return np.hstack([(X[:, COLS_504] - mu) / sd, np.ones((len(X), 1))]) @ w504

    fam_models: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rows = []
    for corpus in TRAIN_FIT:
        X, y = (Xs, ys) if corpus == "ext_safesyn_full" else load(corpus)
        resid = y - stage1_pred(X)
        base_srocc = spearmanr(y, stage1_pred(X)).statistic
        for fam, slots in FAMILIES.items():
            cols = [c for c in fam_cols(slots) if (c - APP_BASE) % PER not in E_CLASS]
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
            cols = [c for c in fam_cols(slots) if (c - APP_BASE) % PER not in E_CLASS]
            fmu, fsd, w = fam_models[fam]
            pred = np.hstack([(X[:, cols] - fmu) / fsd, np.ones((len(y), 1))]) @ w
            sr = spearmanr(resid, pred).statistic if len(y) > 3 else float("nan")
            rows.append(
                dict(corpus=corpus, family=fam, kind="eval-only", n=len(y),
                     stage1_srocc=round(float(base_srocc), 4),
                     r2_in=float("nan"),
                     r2_cv5=round(float(sr), 4))  # holdout column = SROCC(resid, safesyn-family-pred)
            )

    import csv as _csv
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
        f.write(f"# P12 residual-boost — {date.today()}\n\n")
        f.write("Stage 1: BVLS ≥0 on the 504 config, ext_safesyn_full train "
                f"(active {active}/504, train R² {r2(ys, s1):.4f}).\n"
                "Stage 2: per-family LINEAR on residuals (E-class excluded: "
                "GRAD_SRC_MEAN, LUM_MID_ERR, ART_DEV2, DET_DEV2).\n"
                "Holdouts are NEVER fit (CID22 MOS ban absolute): eval-only column = "
                "SROCC(holdout residual, safesyn-fitted family prediction).\n\n")
        f.write("## Ranked marginal value (mean CV-R² across the 4 train-legal corpora)\n\n")
        f.write("| rank | append family | mean CV-R² |\n|---|---|---|\n")
        for i, (fam, v) in enumerate(ranked, 1):
            f.write(f"| {i} | {fam} | {v:.5f} |\n")
        f.write(f"\nFull per-corpus table: `{OUT_CSV.name}` (holdout rows are eval-only).\n")
    print(f"wrote {OUT_CSV.name} + {OUT_MD.name}")
    for i, (fam, v) in enumerate(ranked, 1):
        print(f"  {i}. {fam}: {v:.5f}")


if __name__ == "__main__":
    main()
