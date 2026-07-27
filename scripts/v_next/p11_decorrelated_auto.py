#!/usr/bin/env python3
"""P11 — DECORRELATED-AUTO, plain-mass arm (zensim-bake-subset-plans-2026-07-26).

Procedure (not hand-set): start from 924 minus {f156..f371 fold slots, BANDING,
E1/E2 resolutions}; compute train-mass pairwise |r| (plain mass =
ext_safesyn_full); greedily drop the member of any |r|>0.985 pair with the
worse univariate held-out |SROCC| (held-out scorer = ext_cid22_train201 —
train-legal, ssim2-anchored, NOT MOS); commit the survivor index list as an
artifact. Train a BVLS arm on the survivor set and report holdout SROCCs
vs the 504-config BVLS baseline (holdouts are APPLIED to, never fit — CID22
MOS ban absolute).

The pathology-enriched arm (KADIS negrich severe) requires the W2 kadis-924
extraction — PENDING; the survivor-list diff (the empirical S-class map)
lands when W2 does.

E1/E2 resolutions applied up front: GRAD_SRC_MEAN (app 16) dropped in favor of
its older exact twin PJND_FRAGILITY; LUM_MID_ERR (app 3) dropped (derived
partition bin). E3 deviations stay (P11 does not exclude them; the |r| screen
decides). BANDING (v2 slot 27) excluded per the plan text.

Outputs: benchmarks/p11_survivor_mask_<date>.tsv + p11_decorrelated_auto_<date>.md
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr

D = Path("/mnt/v/zen/zensim-training/ext924-canonical-2026-07-27")
REPO = Path(__file__).resolve().parent.parent.parent
OUT_TSV = REPO / f"benchmarks/p11_survivor_mask_{date.today()}.tsv"
OUT_MD = REPO / f"benchmarks/p11_decorrelated_auto_{date.today()}.md"

N = 924
R_THRESH = 0.985
V2_BASE, V2_PER = 372, 29
APP_BASE, APP_PER = 720, 17
COLS_504 = list(range(0, 156)) + list(range(372, 720))
HOLDOUTS = [
    "ext_cid22val", "ext_aic3", "ext_aic4", "ext_csiq", "ext_live",
    "ext_konjnd_jpeg_val", "ext_sdr25",
]


def cell_cols(base: int, per: int, slot: int) -> set[int]:
    return {base + s * 3 * per + c * per + slot for s in range(4) for c in range(3)}


def load(corpus: str):
    t = pq.read_table(D / f"{corpus}.parquet")
    X = np.column_stack(
        [np.asarray(t[f"f{i}"].combine_chunks().to_numpy(), np.float64) for i in range(N)]
    )
    y = np.asarray(t["human_score"].combine_chunks().to_numpy(), np.float64)
    return X, np.max(y) - y


def bvls(X: np.ndarray, y: np.ndarray):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    A = np.hstack([(X - mu) / sd, np.ones((len(y), 1))])
    lo = np.zeros(A.shape[1])
    lo[-1] = -np.inf
    w = lsq_linear(A, y, bounds=(lo, np.full(A.shape[1], np.inf)), max_iter=300).x
    return mu, sd, w


def main() -> None:
    excl = set(range(156, 372))
    excl |= cell_cols(V2_BASE, V2_PER, 27)   # BANDING
    excl |= cell_cols(APP_BASE, APP_PER, 16)  # E1 GRAD_SRC_MEAN
    excl |= cell_cols(APP_BASE, APP_PER, 3)   # E2 LUM_MID_ERR
    cand = [i for i in range(N) if i not in excl]
    print(f"candidates: {len(cand)} (excluded {len(excl)})")

    Xs, ys = load("ext_safesyn_full")
    Xh, yh = load("ext_cid22_train201")

    Xc = Xs[:, cand]
    sd = Xc.std(0)
    dead = sd < 1e-12
    mu = Xc.mean(0)
    sdn = np.where(dead, 1.0, sd)
    Z = (Xc - mu) / sdn
    C = (Z.T @ Z) / len(Z)
    np.fill_diagonal(C, 0.0)

    srocc = np.zeros(len(cand))
    for j, col in enumerate(cand):
        if not dead[j]:
            srocc[j] = abs(spearmanr(Xh[:, col], yh).statistic)
    srocc = np.nan_to_num(srocc)

    iu, ju = np.triu_indices(len(cand), k=1)
    hot = np.abs(C[iu, ju]) > R_THRESH
    pairs = sorted(
        zip(np.abs(C[iu, ju])[hot], iu[hot], ju[hot]), key=lambda t: -t[0]
    )
    print(f"|r|>{R_THRESH} pairs on plain mass: {len(pairs)}")

    alive = np.ones(len(cand), dtype=bool)
    alive[dead] = False  # zero-variance on train mass carries nothing here
    dropped_for: dict[int, tuple[int, float]] = {}
    for r, a, b in pairs:
        if alive[a] and alive[b]:
            loser, winner = (a, b) if srocc[a] < srocc[b] else (b, a)
            alive[loser] = False
            dropped_for[loser] = (winner, float(r))
    survivors = [cand[j] for j in range(len(cand)) if alive[j]]
    print(f"survivors: {len(survivors)} / {len(cand)}")

    with open(OUT_TSV, "w") as f:
        f.write("feat_idx\tstatus\theldout_abs_srocc\ttwin_kept\tabs_r\n")
        for j, col in enumerate(cand):
            if alive[j]:
                f.write(f"{col}\tkept\t{srocc[j]:.4f}\t\t\n")
            elif j in dropped_for:
                w_, r_ = dropped_for[j]
                f.write(f"{col}\tdropped\t{srocc[j]:.4f}\t{cand[w_]}\t{r_:.4f}\n")
            else:
                f.write(f"{col}\tdead_variance\t{srocc[j]:.4f}\t\t\n")

    print("BVLS arms: 504 baseline vs P11 survivors ...")
    mu5, sd5, w5 = bvls(Xs[:, COLS_504], ys)
    muv, sdv, wv = bvls(Xs[:, survivors], ys)
    a5 = int((w5[:-1] > 1e-9).sum())
    av = int((wv[:-1] > 1e-9).sum())

    def pred(X, cols, mu_, sd_, w_):
        return np.hstack([(X[:, cols] - mu_) / sd_, np.ones((len(X), 1))]) @ w_

    lines = [f"| corpus | 504-BVLS SROCC | P11-BVLS SROCC | Δ |", "|---|---|---|---|"]
    deltas = []
    for c in HOLDOUTS:
        X, y = load(c)
        s5 = spearmanr(y, pred(X, COLS_504, mu5, sd5, w5)).statistic
        sv = spearmanr(y, pred(X, survivors, muv, sdv, wv)).statistic
        deltas.append(sv - s5)
        lines.append(f"| {c} | {s5:.4f} | {sv:.4f} | {sv - s5:+.4f} |")

    with open(OUT_MD, "w") as f:
        f.write(f"# P11 decorrelated-auto (plain-mass arm) — {date.today()}\n\n")
        f.write(f"Candidates {len(cand)} (924 − fold slots − BANDING − E1/E2); "
                f"|r|>{R_THRESH} pairs {len(pairs)}; survivors **{len(survivors)}**. "
                f"Held-out univariate scorer: ext_cid22_train201 (train-legal).\n\n"
                f"BVLS arms on safesyn (≥0, standardized): 504 baseline {a5}/504 active; "
                f"P11 survivors {av}/{len(survivors)} active. Holdouts applied-only "
                f"(never fit; CID22 MOS ban).\n\n")
        f.write("\n".join(lines))
        f.write(f"\n\nMean holdout Δ: {np.mean(deltas):+.4f}. "
                "MLP arm + the pathology-enriched diff (needs W2 kadis-924): PENDING.\n"
                f"Survivor mask: `{OUT_TSV.name}`.\n")
    print(f"wrote {OUT_TSV.name} + {OUT_MD.name}; mean holdout delta {np.mean(deltas):+.4f}")


if __name__ == "__main__":
    main()
