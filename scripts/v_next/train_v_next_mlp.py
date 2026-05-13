#!/usr/bin/env python3
"""Train a 228 → H → 1 MLP perceptual scorer on the unified parquet.

Targets (configurable via --target):
- ssim2          (recommended; aligns zensim with the well-validated SSIMULACRA2)
- butteraugli    (= score_butteraugli_max; classic, but lower CID22 ceiling)
- butteraugli_p3 (= score_butteraugli_pnorm3; smoother, less max-driven)

Validation: source-disjoint split on `image_basename`. 80/10/10.

Architecture (default, matches V0_4 PR #24): 228 → 64 LeakyReLU → 1.

Loss options (--loss):
- mse       Direct regression to target (predict the SSIM2/BA score).
- ranknet   Pairwise (sigmoid) ranking — same-image pairs at different
            distortions; loss aligns with -SROCC at training time.
- mse_rank  Sum: 1.0*MSE + 0.5*RankNet (recommended starting point).

Usage:
    python3 scripts/v_next/train_v_next_mlp.py \\
        --input-dir /mnt/v/zen/zensim-training/2026-05-07/unified \\
        --target ssim2 --loss mse_rank --hidden 64 --epochs 50 \\
        --out-dir /mnt/v/zen/zensim-training/2026-05-07/runs

Outputs:
    runs/<TIMESTAMP>_<TAG>/
        train.log
        model.pt              (PyTorch state dict)
        meta.json             (config + final metrics)
        predictions_val.parquet (image, target, prediction, residual)

The PyTorch model is trained on GPU when available; bake conversion to
ZNPR v2 lives in a separate script (TBD: scripts/v_next/bake_to_znpr.py).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy import stats as sstats


def load_parquets(input_dir: Path, sweeps: list[str]) -> pd.DataFrame:
    parqs = []
    for s in sweeps:
        parqs.extend(sorted(input_dir.glob(f"unified_{s}_*.parquet")))
    if not parqs:
        raise SystemExit(f"no parquets matching sweeps={sweeps} in {input_dir}")
    print(f"Loading {len(parqs)} parquets:", flush=True)
    keep_meta = ["sweep_id", "codec", "image_basename", "q", "knob_tuple_json",
                 "score_zensim", "score_ssim2",
                 "score_butteraugli_max", "score_butteraugli_pnorm3",
                 "metric_runtime", "content_class"]
    feat_cols = [f"feat_{i}" for i in range(228)]
    frames = []
    for p in parqs:
        cols_avail = pq.ParquetFile(p).schema.names
        cols = [c for c in keep_meta + feat_cols if c in cols_avail]
        t0 = time.time()
        df = pq.read_table(p, columns=cols).to_pandas()
        print(f"  {p.name}: {len(df):,} rows × {len(cols)} cols "
              f"({time.time()-t0:.1f}s)", flush=True)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    print(f"Total: {len(full):,} rows", flush=True)
    return full


def load_human_csv(path: Path, dataset_name: str, target_col: str,
                    train_weight: float, val_frac: float = 0.25,
                    seed: int = 0) -> pd.DataFrame:
    """Load a zensim-validate `--features-csv` output and turn it into
    rows compatible with the unified-parquet schema.

    The CSV is `(ref_basename, human_score, metric_score, raw_distance,
    f0..f299)`. We:
    - take only `f0..f227` (the basic + peak features that the V0_4
      runtime expects — extended features at f228..f299 are unused),
    - rename them to `feat_0..feat_227`,
    - scale `human_score` (already in [0,1]) to [0,100] so it's
      comparable with `score_ssim2`,
    - tag the rows with `dataset=<name>`, `train_weight=<weight>`,
      `image_basename = ref_basename`, and synthetic `q=0` /
      `knob_tuple_json="human"` so the downstream split logic still
      works without special-casing.
    - assign each ref to either the train half or val half (`is_val_only`)
      via a per-dataset source-disjoint split with the given `val_frac`.

    The 2026-04-30 V0_4 bake used train_weight=0.3 for human-MOS
    rows alongside synthetic train_weight=1.0 — same recipe here.
    """
    print(f"Loading {dataset_name} {path.name}...", flush=True)
    t0 = time.time()
    if path.suffix == ".parquet":
        df = pq.read_table(path).to_pandas()
    else:
        df = pd.read_csv(path)
    n_orig = len(df)

    feat_cols_csv = [f"f{i}" for i in range(228)]
    if not all(c in df.columns for c in feat_cols_csv):
        raise SystemExit(
            f"{path}: missing one of f0..f227 (got cols: "
            f"{[c for c in df.columns if c.startswith('f')][:5]}...)"
        )

    # Build the unified-schema rows.
    out = pd.DataFrame()
    out["image_basename"] = df["ref_basename"]
    out["sweep_id"] = f"human-{dataset_name}"
    # Cycle-7: preserve real codec/quality from synth CSV when present.
    # Lets TV regularizer work on safe-synthetic rows (which have real
    # q values per quality column). Falls back to V0_16 behavior
    # (codec="human-<name>", q=0) when those columns are absent.
    if "codec_real" in df.columns and "quality" in df.columns:
        out["codec"] = df["codec_real"]
        out["q"] = df["quality"].astype(np.int32)
    else:
        out["codec"] = f"human-{dataset_name}"
        out["q"] = 0
    out["knob_tuple_json"] = "human"
    # human_score is in [0, 1]; scale to [0, 100] to match score_ssim2.
    score_100 = df["human_score"].astype(float) * 100.0
    out["score_zensim"] = score_100
    out["score_ssim2"] = score_100
    out["score_butteraugli_max"] = float("nan")
    out["score_butteraugli_pnorm3"] = float("nan")
    out["metric_runtime"] = float("nan")
    out["content_class"] = ""
    for i in range(228):
        out[f"feat_{i}"] = df[f"f{i}"]

    # Source-disjoint split: assign each unique ref to train or val.
    refs = pd.Series(df["ref_basename"].unique())
    # Python's `hash()` is randomized per process, so use a stable
    # in-process hash so re-runs with the same --seed get identical
    # KADID/TID train/val splits.
    name_seed = sum(ord(c) for c in dataset_name) * 31
    rng = np.random.default_rng(seed + name_seed)
    perm = rng.permutation(len(refs))
    # `val_frac=0` means no val-only split — the rows go through the
    # standard image-basename split via `make_split` like synthetic
    # rows. Use this for the canonical synthetic CSV (added as a
    # human-csv style input but with weight=1.0 and no val carving).
    n_val = int(round(len(refs) * val_frac))
    val_refs = set(refs.iloc[perm[:n_val]]) if n_val > 0 else set()
    out["is_val_only"] = out["image_basename"].isin(val_refs)
    out["train_weight"] = np.where(out["is_val_only"], 0.0, train_weight)
    out["dataset"] = dataset_name
    out["target_human"] = score_100  # explicit copy for V0_6 ranknet groups
    # Optional: preserve `dssim` column for cycle-7 co-training. When the
    # CSV has it, the downstream trainer reads it via `df["dssim"]` when
    # `--dssim-weight > 0`.
    if "dssim" in df.columns:
        out["dssim"] = df["dssim"].astype(np.float32)

    print(f"  {n_orig:,} rows, {len(refs)} unique refs "
          f"(val={n_val} val_only refs); train_w={train_weight}, "
          f"loaded in {time.time()-t0:.1f}s", flush=True)
    return out


def make_split(df: pd.DataFrame, val_frac: float, test_frac: float,
               seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source-disjoint split on image_basename. Returns boolean masks."""
    images = pd.Series(df["image_basename"].unique())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(images))
    n = len(images)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_imgs = set(images.iloc[perm[:n_test]])
    val_imgs = set(images.iloc[perm[n_test:n_test + n_val]])
    is_test = df["image_basename"].isin(test_imgs).to_numpy()
    is_val = df["image_basename"].isin(val_imgs).to_numpy()
    is_train = ~(is_val | is_test)
    print(f"Split: train={is_train.sum():,} ({is_train.mean()*100:.1f}%) "
          f"val={is_val.sum():,} ({is_val.mean()*100:.1f}%) "
          f"test={is_test.sum():,} ({is_test.mean()*100:.1f}%)")
    return is_train, is_val, is_test


def build_arrays(df: pd.DataFrame, target_col: str
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feat_cols = [c for c in df.columns if c.startswith("feat_")
                 and int(c[5:]) < 228]
    feat_cols = sorted(feat_cols, key=lambda c: int(c[5:]))
    n_features = len(feat_cols)
    print(f"Using {n_features} feature columns: feat_0..feat_{n_features-1}",
          flush=True)
    print("Materializing feature matrix (this is the slow part — pandas → "
          "numpy on 2M+ rows × 228 cols)...", flush=True)
    t0 = time.time()
    X_full = df[feat_cols].to_numpy(dtype=np.float32)
    y_full = df[target_col].to_numpy(dtype=np.float32)
    print(f"  done in {time.time()-t0:.1f}s, X shape {X_full.shape}", flush=True)

    # Drop rows with NaN/inf in target or features.
    finite_y = np.isfinite(y_full) & (np.abs(y_full) < 1e6)
    finite_x = np.isfinite(X_full).all(axis=1)
    keep_mask = finite_y & finite_x
    n_drop = int((~keep_mask).sum())
    if n_drop:
        print(f"Dropping {n_drop:,} rows with NaN/inf in target or features",
              flush=True)

    images = df["image_basename"].to_numpy()
    sweep_codec = (df["sweep_id"].astype(str) + ":"
                    + df["codec"].astype(str)).to_numpy()
    return X_full, y_full, images, sweep_codec, keep_mask


def standardize(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature Z-score on the training split, applied to val + test.

    Replaces NaN/inf with 0 after standardization (some features are
    degenerate at certain content classes and don't carry signal).
    Returns (X_train, X_val, X_test, mean, std) — mean/std saved into
    the bake's scaler section so runtime forward pass reproduces the
    same normalization.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    # Floor std away from zero so constant features don't explode after divide.
    std = np.where(std < 1e-6, 1.0, std)
    Xt = (X_train - mean) / std
    Xv = (X_val - mean) / std
    Xs = (X_test - mean) / std
    for arr in (Xt, Xv, Xs):
        arr[~np.isfinite(arr)] = 0.0
    return Xt.astype(np.float32), Xv.astype(np.float32), Xs.astype(np.float32), \
           mean.astype(np.float32), std.astype(np.float32)


class MLP(torch.nn.Module):
    def __init__(self, n_in: int, hidden: list[int], init: str = "kaiming"):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(torch.nn.LeakyReLU(negative_slope=0.01))
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)
        if init == "glorot":
            # Glorot/Xavier-normal init matches the deleted Rust mlp_train.rs
            # which used `std = sqrt(2 / (n_in + n_out))` per layer. PyTorch's
            # default is Kaiming-uniform; Glorot has different fan handling
            # that better matches RankNet's scale assumptions.
            for m in self.net:
                if isinstance(m, torch.nn.Linear):
                    fan_in, fan_out = m.weight.shape[1], m.weight.shape[0]
                    std = (2.0 / (fan_in + fan_out)) ** 0.5
                    torch.nn.init.normal_(m.weight, mean=0.0, std=std)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def ranknet_loss(pred: torch.Tensor, target: torch.Tensor,
                  groups: torch.Tensor, max_total_pairs: int = 4096,
                  low_q_pair_boost: float = 1.0,
                  ) -> torch.Tensor:
    """Pairwise sigmoid loss (Bradley–Terry / RankNet) over same-group pairs.

    Vectorized: sample random index pairs from the batch in a single
    `torch.randint`, filter to same-group i<j, cap, and compute the
    softplus-of-signed-margin in pure tensor ops. Removes the
    per-group Python `for` loop that bottlenecked the previous
    implementation (15% CPU, 22% GPU on a Ryzen 9 + RTX 5070; ~155 s
    per 16k-row epoch on 1.86M rows).

    Trades the original group-uniform-then-pair-uniform weighting for
    pair-uniform sampling — pairs from larger groups contribute
    proportionally more, but since we cap at `max_total_pairs` per
    batch and groups within a batch are roughly equal-sized
    (image_basename × q × codec), the bias is small. Empirically
    SROCC trajectory on a 200k-row smoke matches within 0.001.
    """
    device = pred.device
    n = pred.size(0)
    if n < 2:
        return torch.zeros((), device=device)
    # Oversample candidates by ~num_unique_groups so the same-group
    # filter still leaves us enough pairs. The `groups.unique()` call
    # is cheap (single pass, GPU-side) compared to the previous
    # per-group loop.
    num_unique = int(groups.unique().numel())
    n_candidates = min(max_total_pairs * max(num_unique, 1) * 2,
                       max_total_pairs * 256)
    n_candidates = min(n_candidates, 1_048_576)
    i = torch.randint(0, n, (n_candidates,), device=device)
    j = torch.randint(0, n, (n_candidates,), device=device)
    keep = (groups[i] == groups[j]) & (i < j)
    i = i[keep]
    j = j[keep]
    if i.numel() == 0:
        return torch.zeros((), device=device)
    if i.numel() > max_total_pairs:
        sel = torch.randperm(i.numel(), device=device)[:max_total_pairs]
        i = i[sel]
        j = j[sel]
    p_diff = pred[i] - pred[j]
    t_diff = target[i] - target[j]
    nonzero = t_diff != 0
    if nonzero.sum() == 0:
        return torch.zeros((), device=device)
    i_nz = i[nonzero]
    j_nz = j[nonzero]
    sign = torch.sign(t_diff[nonzero])
    losses = torch.nn.functional.softplus(-sign * p_diff[nonzero])
    if low_q_pair_boost == 1.0:
        return losses.mean()
    # Cycle-9b: weight each pair by max(boost_i, boost_j) where
    # boost = low_q_pair_boost if endpoint.target<50, sqrt(boost) if
    # 50..65, else 1.0. Oversamples low-q ranking signal at the
    # RankNet loss term (which carries most rank-correlation signal).
    sqrt_boost = float(low_q_pair_boost) ** 0.5
    ti = target[i_nz]
    tj = target[j_nz]
    boost_i = torch.where(
        ti < 50.0,
        torch.full_like(ti, float(low_q_pair_boost)),
        torch.where(ti < 65.0,
                    torch.full_like(ti, sqrt_boost),
                    torch.ones_like(ti)))
    boost_j = torch.where(
        tj < 50.0,
        torch.full_like(tj, float(low_q_pair_boost)),
        torch.where(tj < 65.0,
                    torch.full_like(tj, sqrt_boost),
                    torch.ones_like(tj)))
    weights = torch.maximum(boost_i, boost_j)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


def srocc(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    rho, _ = sstats.spearmanr(a, b)
    return float(rho)


def krocc(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    tau, _ = sstats.kendalltau(a, b)
    return float(tau)


@dataclass
class TrainConfig:
    target: str
    loss: str
    hidden: list[int]
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    dropout: float
    rank_weight: float
    seed: int
    tv_weight: float = 0.0  # Per-curve monotonicity penalty weight.
    lr_schedule: str = "constant"  # "constant" | "cosine" — cosine matches the
                                   # deleted Rust mlp_train.rs trainer that
                                   # produced V0_5's CID22 0.8893.
    optimizer: str = "adamw"       # "adamw" | "adam" — Rust trainer used Adam
                                   # (no decoupled weight decay).
    init: str = "kaiming"          # "kaiming" | "glorot" — Rust used Glorot.
    val_policy: str = "mean"       # "mean" | "min" — Rust used `Min` (worst
                                   # per-group SROCC), Python defaulted to mean.
    lr_cycle_period: int = 50      # First cycle length in epochs for
                                   # cosine_cyclic schedule. T_mult=1.
    dssim_weight: float = 0.0      # Cycle-7 dssim co-training auxiliary loss
                                   # weight. When > 0, adds
                                   # `dssim_weight * mse(pred, (1-dssim)*100)`
                                   # to the loss. Targets JPEG-AI deficit
                                   # (AIC-4: V0_16=0.7951, dssim=0.9147).
    low_q_pair_boost: float = 1.0  # Cycle-9b RankNet pair-resampling boost.
                                   # When > 1, weights ranknet softplus loss
                                   # for pairs where either endpoint has
                                   # target < 50 by factor; sqrt(factor)
                                   # for 50..65. Targets B0/B1 SROCC.


def train(cfg: TrainConfig, X_train, y_train, g_train,
          X_val, y_val, g_val, device,
          tv_pairs_train: np.ndarray | None = None,
          w_train: np.ndarray | None = None,
          val_dataset_labels: np.ndarray | None = None,
          dssim_train: np.ndarray | None = None,
          ) -> tuple[MLP, dict]:
    torch.manual_seed(cfg.seed)
    n_in = X_train.shape[1]
    model = MLP(n_in, cfg.hidden, init=cfg.init).to(device)
    if cfg.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                               weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
    # LR scheduler. The deleted Rust mlp_train.rs (PR #29 e613224) used
    # "Adam with cosine annealing" which is hypothesized to be one of the
    # ingredients that gave V0_5 its CID22 0.8893 SROCC; replicate that
    # path when cfg.lr_schedule == "cosine".
    if cfg.lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)
    elif cfg.lr_schedule == "cosine_cyclic":
        # Warm-restart cosine — `T_0` is the first cycle length in
        # epochs. `T_mult=1` keeps subsequent cycles equal-length.
        # The Rust trainer's recovery candidate hypothesis was that
        # cyclic cosine with T_0 ≈ 50 helps escape sharp minima
        # without the full annealing decay starving late training.
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=cfg.lr_cycle_period, T_mult=1, eta_min=cfg.lr * 0.01)
    else:
        sched = None

    # Pre-bake tensors on GPU
    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    gt = torch.from_numpy(g_train).to(device)
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)
    # Per-row training weight: synthetic rows = 1.0, KADID/TID rows
    # default 0.3 to match the 2026-04-30 V0_4 mixed-supervision
    # recipe. Applied only to MSE — RankNet's pair sampling already
    # naturally over-represents larger groups, and weighting pairs
    # would double-count.
    wt = (torch.from_numpy(w_train).to(device).float()
          if w_train is not None else None)
    # dssim co-training target (quality-scaled: (1 - dssim) * 100).
    # When cfg.dssim_weight > 0 and dssim_train provided, the loss adds
    # an MSE term against this target.
    dst = None
    if cfg.dssim_weight > 0 and dssim_train is not None:
        dssim_quality = (1.0 - dssim_train) * 100.0
        dst = torch.from_numpy(dssim_quality.astype(np.float32)).to(device)
        print(f"  dssim co-training: weight={cfg.dssim_weight}, "
              f"target range [{float(dst.min()):.2f}, {float(dst.max()):.2f}]",
              flush=True)

    # Adjacent-q pairs (lower_idx, higher_idx) within each curve for
    # the TV regularizer. `pred[lower] >= pred[higher]` is a violation
    # (worse-q produced higher quality score), penalized as
    # `relu(pred[lower] - pred[higher])`.
    tv_pairs_t = None
    if tv_pairs_train is not None and cfg.tv_weight > 0:
        if len(tv_pairs_train) == 0:
            print(f"  TV regularizer disabled: 0 adjacent-q pairs in training "
                  f"set (likely --human-csv-only mode with synthetic q=0); "
                  f"set --tv-weight=0 to silence this message", flush=True)
        else:
            tv_pairs_t = torch.from_numpy(tv_pairs_train).to(device)
            print(f"  TV regularizer: {len(tv_pairs_t):,} adjacent-q pairs, "
                  f"weight={cfg.tv_weight}", flush=True)

    best = {"epoch": -1, "val_srocc": -1, "val_mse": math.inf,
            "state": None}

    n = Xt.size(0)
    bs = min(cfg.batch_size, n)
    # TV pair sample size per batch — quarter of batch_size keeps the
    # regularizer term the same magnitude as MSE while keeping the
    # fused forward pass tractable on a single GPU.
    tv_bs = (bs // 4) if tv_pairs_t is not None else 0
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            x = Xt[idx]
            y = yt[idx]
            g = gt[idx]

            # Build a fused input: main batch + (TV lo) + (TV hi). One
            # model forward per step instead of three; one backward
            # over the combined output. This was the dominant
            # bottleneck on a 228 → 64 → 1 MLP — Python overhead per
            # `model(...)` call dwarfed the actual matmul.
            if tv_pairs_t is not None:
                # Use actual batch size, not the constant `bs` — the last
                # partial batch is smaller than `bs` and slicing on `bs`
                # over-reads into the TV-pair predictions.
                n_main = x.size(0)
                pair_idx = torch.randint(
                    0, tv_pairs_t.size(0), (tv_bs,), device=device)
                lo = tv_pairs_t[pair_idx, 0]
                hi = tv_pairs_t[pair_idx, 1]
                fused = torch.cat([x, Xt[lo], Xt[hi]], dim=0)
                pred_all = model(fused)
                pred = pred_all[:n_main]
                pred_lo = pred_all[n_main:n_main + tv_bs]
                pred_hi = pred_all[n_main + tv_bs:]
            else:
                pred = model(x)
                pred_lo = pred_hi = None

            if wt is not None:
                w = wt[idx]
                # Per-row weighted MSE; normalize by sum-of-weights so
                # the magnitude stays comparable to unweighted MSE.
                mse = ((pred - y) ** 2 * w).sum() / w.sum().clamp_min(1e-6)
            else:
                mse = torch.mean((pred - y) ** 2)
            if cfg.loss == "mse":
                loss = mse
            elif cfg.loss == "ranknet":
                loss = ranknet_loss(pred, y, g,
                                    low_q_pair_boost=cfg.low_q_pair_boost)
            elif cfg.loss == "mse_rank":
                rk = ranknet_loss(pred, y, g,
                                  low_q_pair_boost=cfg.low_q_pair_boost)
                loss = mse + cfg.rank_weight * rk
            else:
                raise ValueError(cfg.loss)
            if pred_lo is not None:
                tv = torch.relu(pred_lo - pred_hi).mean()
                loss = loss + cfg.tv_weight * tv
            if dst is not None:
                # dssim auxiliary MSE against quality-scaled dssim target.
                # Pulls pred toward (1-dssim)*100; rank-invariant with the
                # ssim2 head since both are quality (higher = better).
                # NaN-aware: only rows that have dssim contribute. Sweep
                # parquet rows (which lack dssim) skip this term cleanly.
                dssim_target = dst[idx]
                mask = ~torch.isnan(dssim_target)
                if mask.any():
                    p_valid = pred[mask]
                    t_valid = dssim_target[mask]
                    dssim_mse = torch.mean((p_valid - t_valid) ** 2)
                    loss = loss + cfg.dssim_weight * dssim_mse
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1
        if sched is not None:
            sched.step()

        model.eval()
        with torch.no_grad():
            pv = model(Xv).detach().cpu().numpy()
        val_mse = float(np.mean((pv - y_val) ** 2))
        val_sr = srocc(pv, y_val)
        val_kr = krocc(pv, y_val)

        # Per-dataset val SROCC — when human-MOS data is mixed in,
        # the global SROCC is dominated by the synthetic majority and
        # hides whether the model actually generalizes to human
        # ratings. Report per-dataset SROCC every epoch and use the
        # *mean* over datasets as the model-selection criterion if
        # multiple datasets are present.
        per_ds_srocc: dict[str, float] = {}
        per_ds_str = ""
        if val_dataset_labels is not None:
            unique_ds = sorted(set(val_dataset_labels.tolist()))
            for ds_name in unique_ds:
                mask = val_dataset_labels == ds_name
                if mask.sum() < 2:
                    continue
                s = srocc(pv[mask], y_val[mask])
                per_ds_srocc[ds_name] = s
            if per_ds_srocc:
                per_ds_str = "  " + "  ".join(
                    f"{k}={v:.4f}" for k, v in per_ds_srocc.items())
        # Selection metric: per `cfg.val_policy`. Min matches the Rust
        # trainer (worst per-group SROCC); mean was the Python default
        # but lets the synthetic majority dominate selection.
        if len(per_ds_srocc) >= 2:
            vals = list(per_ds_srocc.values())
            if cfg.val_policy == "min":
                sel_metric = float(min(vals))
            else:
                sel_metric = float(np.mean(vals))
        else:
            sel_metric = val_sr
        if sel_metric > best["val_srocc"]:
            best = {"epoch": ep, "val_srocc": sel_metric, "val_mse": val_mse,
                    "val_krocc": val_kr,
                    "per_ds_srocc": dict(per_ds_srocc),
                    "state": {k: v.detach().clone().cpu()
                              for k, v in model.state_dict().items()}}
        print(f"  epoch {ep:3d}  train_loss={ep_loss/max(1,n_batches):.4f}  "
              f"val_mse={val_mse:.3f}  val_srocc={val_sr:.4f}  "
              f"val_krocc={val_kr:.4f}  sel={sel_metric:.4f}{per_ds_str}",
              flush=True)
        sys.stdout.flush()

    if best["state"]:
        model.load_state_dict(best["state"])
    metrics = {k: v for k, v in best.items() if k != "state"}
    return model, metrics


def _apply_ssim2_butter_concordance(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows from (image, codec) groups where ssim2 and butter
    rank-orders disagree.

    For each (image_basename, codec) group sorted by q ascending: the
    ssim2 column should monotonically increase (or at least follow the
    same rank-order) as `-butter`. We compute Spearman rank correlation
    within each group; groups with negative or near-zero correlation
    have noisy quality labels — drop them entirely.

    Threshold: keep groups with within-group Spearman(ssim2, -butter) ≥
    0.6. Empirically this drops ~5–15 % of synthetic rows in q-tails
    where the CID22 paper flags ssim2 as less accurate.

    Rows with NaN butter (i.e., human-csv rows where load_human_csv
    set score_butteraugli_max = NaN) are kept unchanged — only
    same-curve compression-sweep rows get filtered.
    """
    if "score_butteraugli_max" not in df.columns:
        return df
    has_both = df["score_ssim2"].notna() & df["score_butteraugli_max"].notna()
    keep_mask = pd.Series(True, index=df.index)
    if has_both.sum() == 0:
        return df
    sub = df[has_both]
    # Group by (image_basename, codec, knob_tuple_json) when columns exist;
    # else fall back to (image_basename, codec).
    group_cols = [c for c in
                  ["image_basename", "codec", "knob_tuple_json"]
                  if c in sub.columns]
    if not group_cols:
        return df
    # For each group, compute Spearman rank corr between ssim2 and -butter.
    grp = sub.groupby(group_cols, sort=False)
    drop_count = 0
    drop_idx_set: list[int] = []
    for keys, idx in grp.indices.items():
        if len(idx) < 3:
            continue  # too few points to estimate rank corr
        vss = sub["score_ssim2"].iloc[idx].values
        vba = sub["score_butteraugli_max"].iloc[idx].values
        # Spearman: rank both, then Pearson on ranks
        from scipy.stats import spearmanr
        try:
            r, _ = spearmanr(vss, -vba)
        except Exception:
            continue
        if r is None or (isinstance(r, float) and (r != r or r < 0.6)):
            real_idx = sub.index[idx]
            drop_idx_set.extend(real_idx.tolist())
            drop_count += len(idx)
    if drop_count:
        keep_mask.loc[drop_idx_set] = False
    return df[keep_mask].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir",
                    default="/mnt/v/zen/zensim-training/2026-05-07/unified")
    ap.add_argument("--sweeps", default="v15r,v15rc",
                    help="Comma-separated sweep ids to train on")
    ap.add_argument("--target", default="ssim2",
                    choices=["ssim2", "butteraugli", "butteraugli_p3", "zensim"])
    ap.add_argument("--loss", default="mse_rank",
                    choices=["mse", "ranknet", "mse_rank"])
    ap.add_argument("--hidden", default="64",
                    help="Comma-separated hidden layer widths")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--rank-weight", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Cap rows for debug runs")
    ap.add_argument("--out-dir",
                    default="/mnt/v/zen/zensim-training/2026-05-07/runs")
    ap.add_argument("--tag", default="v_next")
    ap.add_argument("--dssim-weight", type=float, default=0.0,
                    help="cycle-7 dssim co-training auxiliary loss weight "
                         "(0.0 = disabled, V0_16 behavior; suggested 0.3 to "
                         "target JPEG-AI deficit)")
    ap.add_argument("--tv-pairs-file", type=str, default=None,
                    help="Load pre-built adjacent-q TV pairs from TSV "
                         "(columns lo_trainer_idx + hi_trainer_idx, "
                         "optional band_id). Overrides auto-build from "
                         "training rows. Used to reproduce V0_16's "
                         "combined_purged_tv_pairs_bands.tsv (205,654 pairs).")
    ap.add_argument("--tv-weight", type=float, default=0.0,
                    help="Per-curve monotonicity penalty weight. Penalizes "
                         "adjacent-q score reversals within each "
                         "(image, codec, knob_tuple) curve. 0 disables.")
    ap.add_argument("--lr-schedule", default="constant",
                    choices=["constant", "cosine", "cosine_cyclic"],
                    help="LR schedule. cosine matches the deleted Rust "
                         "mlp_train.rs trainer that produced V0_5's "
                         "CID22 0.8893 SROCC. cosine_cyclic uses "
                         "CosineAnnealingWarmRestarts with T_0 = "
                         "--lr-cycle-period (default 50).")
    ap.add_argument("--lr-cycle-period", type=int, default=50,
                    help="First cycle length (epochs) for "
                         "cosine_cyclic schedule. T_mult=1.")
    ap.add_argument("--class-balance", default="none",
                    choices=["none", "weight"],
                    help="Per-content-class balancing for synth rows. "
                         "weight: inverse-frequency multiplier on each "
                         "row's train_weight (rows with empty "
                         "content_class are left at 1.0).")
    ap.add_argument("--low-q-boost", type=float, default=1.0,
                    help="Multiply train_weight for low-quality rows by this "
                         "factor. Bins by target column: score<50 (B0) gets "
                         "the full multiplier, 50<=score<65 (B1) gets "
                         "sqrt(multiplier). Default 1.0 = no boost. Use "
                         "to address CID22 B0/B1 SROCC ceiling (cycle-9).")
    ap.add_argument("--low-q-pair-boost", type=float, default=1.0,
                    help="Cycle-9b lever: weight RankNet pair losses by max "
                         "boost of the pair's two endpoints (B0 endpoint → "
                         "boost factor, B1 → sqrt boost, else 1.0). Targets "
                         "B0/B1 ranking signal at the rank-loss term rather "
                         "than via MSE row weighting (cycle-9). Default 1.0.")
    ap.add_argument("--optimizer", default="adamw",
                    choices=["adamw", "adam"],
                    help="adam matches the Rust trainer (no decoupled "
                         "weight decay).")
    ap.add_argument("--init", default="kaiming",
                    choices=["kaiming", "glorot"],
                    help="glorot matches the Rust trainer's std=sqrt(2/(in+out)) "
                         "normal init.")
    ap.add_argument("--val-policy", default="mean",
                    choices=["mean", "min"],
                    help="min matches the Rust trainer's ValidationPolicy::Min "
                         "(worst per-group SROCC drives selection).")
    ap.add_argument("--ranknet-group", default="image",
                    choices=["image", "dataset"],
                    help="dataset matches the Rust trainer's per-dataset "
                         "pair sampling — pairs span all rows within a "
                         "dataset (synthetic/kadid/tid), allowing cross-image "
                         "absolute-quality ranking. image (default) restricts "
                         "to same-source-image curves only.")
    ap.add_argument("--concordance-filter", default="none",
                    choices=["none", "ssim2_butter"],
                    help="ssim2_butter drops synthetic rows where the metric "
                         "ranking within an (image, codec) curve disagrees "
                         "between gpu_ssimulacra2 and gpu_butteraugli (bigger "
                         "= worse for butter). Cleans noisy ranking labels per "
                         "the CID22 paper's Table 3 caveat that ssim2 is less "
                         "reliable in the q-extremes. Only applies to rows "
                         "with both columns present (i.e., the unified-parquet "
                         "rows from --sweeps; --human-csv rows have score_ssim2 "
                         "= score_zensim = human_score and are not filtered).")
    ap.add_argument("--human-csv", action="append", default=[],
                    metavar="NAME:PATH:WEIGHT[:VAL_FRAC]",
                    help="Add a CSV produced by `zensim-validate "
                         "--extract-only --features-csv`. Format: "
                         "NAME:PATH:WEIGHT[:VAL_FRAC]. WEIGHT scales the "
                         "MSE term per row (synthetic synth canonical = "
                         "1.0; KADID/TID human-MOS = 0.3 per the V0_4 "
                         "recipe). VAL_FRAC reserves that fraction of "
                         "the dataset's unique refs as val-only (default "
                         "0.25 for human-MOS datasets). Pass 0.0 for the "
                         "canonical synthetic CSV — its rows go through "
                         "the normal image-basename split. Pass 1.0 for "
                         "CID22 holdout / pure validation sets so they "
                         "never appear in training. Repeat the flag for "
                         "multiple datasets.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    target_col = {
        "ssim2": "score_ssim2",
        "butteraugli": "score_butteraugli_max",
        "butteraugli_p3": "score_butteraugli_pnorm3",
        "zensim": "score_zensim",
    }[args.target]

    # When the user passes only --human-csv (no --sweeps), allow skipping
    # the parquet load entirely. The dataframe is built from human-csv rows
    # alone in that case (V0_16 + V0_24 use this path for the safe-synthetic
    # CSV which carries `feat_0..feat_227` directly).
    if args.sweeps and args.sweeps != "NONE":
        df = load_parquets(Path(args.input_dir), args.sweeps.split(","))
    else:
        if not args.human_csv:
            raise SystemExit("Pass either --sweeps or --human-csv")
        print("Sweeps disabled (`--sweeps NONE` or empty); training from --human-csv rows only")
        df = pd.DataFrame()
    if args.max_rows:
        df = df.sample(min(args.max_rows, len(df)), random_state=args.seed)\
               .reset_index(drop=True)
        print(f"Subsampled to {len(df):,} rows")

    # Tag synthetic rows with default dataset / weight / val flag so the
    # human-MOS rows can be concatenated cleanly without a schema clash.
    df["dataset"] = "synthetic"
    df["train_weight"] = 1.0
    df["is_val_only"] = False

    # Concordance filter (per zensim CLAUDE.md "Training goals" #4):
    # within each (image, codec) curve, drop rows where the rank-order of
    # gpu_ssimulacra2 disagrees with the rank-order of gpu_butteraugli
    # (lower butter = better, so concordance = ssim2-rank == reversed-butter-rank).
    # Cleans noisy ranking labels in the q-tails where the CID22 paper
    # flags ssim2 as less reliable. Only applies to synthetic-codec-sweep
    # rows that have both metrics; --human-csv rows are left untouched
    # (they carry score_zensim = human_score and have NaN butter).
    if (args.concordance_filter == "ssim2_butter"
            and "score_butteraugli_max" in df.columns
            and "score_ssim2" in df.columns):
        before = len(df)
        df = _apply_ssim2_butter_concordance(df)
        after = len(df)
        print(f"Concordance filter (ssim2 ↔ butter): "
              f"kept {after:,} / {before:,} rows ({after/before*100:.1f}%)",
              flush=True)

    # Optionally splice in human-rated CSVs (KADID, TID, ...) using the
    # 2026-04-30 V0_4 mixed-supervision recipe — synthetic train_w=1.0,
    # human train_w=0.3, with 25% of each dataset's refs reserved as
    # val-only.
    human_frames = []
    for spec in args.human_csv:
        parts = spec.split(":")
        if len(parts) == 3:
            name, csv_path, weight_str = parts
            val_frac = 0.25
        elif len(parts) == 4:
            name, csv_path, weight_str, val_frac_str = parts
            val_frac = float(val_frac_str)
        else:
            raise SystemExit(
                f"--human-csv expects NAME:PATH:WEIGHT[:VAL_FRAC], got {spec!r}")
        weight = float(weight_str)
        h = load_human_csv(Path(csv_path), name, target_col, weight,
                           val_frac=val_frac, seed=args.seed)
        human_frames.append(h)
    if human_frames:
        n_synth = len(df)
        df = pd.concat([df] + human_frames, ignore_index=True)
        print(f"After human-MOS splice: {n_synth:,} synthetic + "
              f"{len(df) - n_synth:,} human = {len(df):,} total rows")

    # Per-content-class balancing (per zensim CLAUDE.md "Training goals" #1
    # context: CID22 has mixed content classes; CLIC2025-derived synth is
    # ~70% illustration/screen-mixed, ~30% photo). Multiplies train_weight
    # by an inverse-frequency factor so each class contributes equally to
    # the MSE term. Rows with empty content_class (human-MOS) are left at
    # weight 1.0. Sample-mode is not yet implemented — only weight-mode.
    if args.class_balance == "weight":
        counts = df["content_class"].value_counts(dropna=False)
        labelled = counts.drop("", errors="ignore")
        if labelled.empty or labelled.sum() == 0:
            print("class-balance=weight: no non-empty content_class; skipping",
                  flush=True)
        else:
            mean_cnt = labelled.mean()
            cls_weights = (mean_cnt / labelled).to_dict()
            cls_weights[""] = 1.0
            cls_weights[None] = 1.0
            multiplier = df["content_class"].map(cls_weights).fillna(1.0)
            df["train_weight"] = (df["train_weight"]
                                  * multiplier.astype(np.float32))
            print("class-balance=weight applied. Per-class multipliers:",
                  flush=True)
            for c, w in sorted(cls_weights.items(),
                               key=lambda kv: -labelled.get(kv[0], 0)):
                if c in ("", None):
                    continue
                print(f"  {c!r}: count={labelled[c]:,} "
                      f"× {w:.3f}", flush=True)

    if args.low_q_boost != 1.0 and target_col in df.columns:
        boost = float(args.low_q_boost)
        sqrt_boost = boost ** 0.5
        target_vals = df[target_col].to_numpy(dtype=np.float32)
        mult = np.ones_like(target_vals, dtype=np.float32)
        b0_mask = target_vals < 50.0
        b1_mask = (target_vals >= 50.0) & (target_vals < 65.0)
        mult[b0_mask] = boost
        mult[b1_mask] = sqrt_boost
        n_b0 = int(b0_mask.sum())
        n_b1 = int(b1_mask.sum())
        df["train_weight"] = (df["train_weight"]
                              * mult.astype(np.float32))
        print(f"low-q-boost: B0 (score<50) n={n_b0:,} × {boost:.3f}; "
              f"B1 (50..65) n={n_b1:,} × {sqrt_boost:.3f}; "
              f"others × 1.0", flush=True)

    is_tr, is_va, is_te = make_split(df, args.val_frac, args.test_frac, args.seed)
    # Force human val-only rows (is_val_only=True) into val regardless of
    # what make_split picked — they were never candidates for training.
    if df["is_val_only"].any():
        force_val = df["is_val_only"].to_numpy()
        is_tr = is_tr & ~force_val
        is_te = is_te & ~force_val
        is_va = is_va | force_val
    X, y, images, sc, keep_mask = build_arrays(df, target_col)

    # Combine the original split masks with the NaN-drop mask so X/y/g_all
    # are all aligned at the row level.
    is_tr_x = is_tr & keep_mask
    is_va_x = is_va & keep_mask
    is_te_x = is_te & keep_mask

    # RankNet group key. Two options:
    #   image  — per-image curves; only same-source pairs sample. Within-curve
    #            ranking signal only. (Original Python default.)
    #   dataset — per-dataset; any two rows in the same dataset can pair.
    #             Matches the deleted Rust mlp_train.rs which sampled from
    #             {Synthetic, KADID, TID} groups and ranked cross-image
    #             absolute quality. Generalizes better to CID22 (which is
    #             also cross-image absolute MOS).
    print(f"Encoding RankNet groups (key={args.ranknet_group})...", flush=True)
    t0 = time.time()
    if args.ranknet_group == "dataset":
        codes, _ = pd.factorize(df["dataset"], sort=False)
    else:
        codes, _ = pd.factorize(images, sort=False)
    g_all = codes.astype(np.int64)
    print(f"  {len(set(codes.tolist())):,} unique groups "
          f"({time.time()-t0:.1f}s)", flush=True)

    cfg = TrainConfig(
        target=args.target, loss=args.loss,
        hidden=[int(h) for h in args.hidden.split(",")],
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, dropout=args.dropout,
        rank_weight=args.rank_weight, seed=args.seed,
        tv_weight=args.tv_weight, dssim_weight=args.dssim_weight,
        low_q_pair_boost=args.low_q_pair_boost,
        lr_schedule=args.lr_schedule,
        lr_cycle_period=args.lr_cycle_period,
        optimizer=args.optimizer,
        init=args.init,
        val_policy=args.val_policy)

    print(f"Config: {cfg}")
    print(f"Train rows: {is_tr_x.sum():,}  Val rows: {is_va_x.sum():,}  "
          f"Test rows: {is_te_x.sum():,}")

    # Standardize features on the train split. Without this the raw
    # feat_* columns span ~7 orders of magnitude (some 0..1 SSIM means,
    # some MSE values >100, plus rare outliers >700) and the linear
    # init can't escape early gradient explosion.
    Xt, Xv, Xs, scaler_mean, scaler_std = standardize(
        X[is_tr_x], X[is_va_x], X[is_te_x])
    print(f"Standardized features: train mean abs={np.abs(Xt).mean():.3f}, "
          f"std={Xt.std():.3f}")

    # Build adjacent-q TV pairs over training rows. For each
    # (image_basename, codec, knob_tuple_json) curve, sort by q and
    # emit (lower_q_idx, higher_q_idx) pairs. These are *post-mask*
    # local indices — they reference rows in the standardized Xt.
    tv_pairs_train = None
    if args.tv_weight > 0 and args.tv_pairs_file:
        print(f"Loading external TV pairs from {args.tv_pairs_file}...", flush=True)
        t0 = time.time()
        tv_df = pd.read_csv(args.tv_pairs_file, sep="\t")
        if "lo_trainer_idx" not in tv_df.columns or "hi_trainer_idx" not in tv_df.columns:
            raise SystemExit(
                f"--tv-pairs-file {args.tv_pairs_file} missing required "
                f"columns lo_trainer_idx + hi_trainer_idx (got: {list(tv_df.columns)})"
            )
        n_tr = int(is_tr_x.sum())
        lo = tv_df["lo_trainer_idx"].to_numpy(dtype=np.int64)
        hi = tv_df["hi_trainer_idx"].to_numpy(dtype=np.int64)
        in_range = (lo >= 0) & (lo < n_tr) & (hi >= 0) & (hi < n_tr)
        lo = lo[in_range]
        hi = hi[in_range]
        tv_pairs_train = np.stack([lo, hi], axis=1).astype(np.int64)
        print(f"  {len(tv_pairs_train):,} external TV pairs loaded "
              f"({time.time()-t0:.1f}s; {int((~in_range).sum())} rows dropped "
              f"as out-of-range for n_train={n_tr})", flush=True)
    elif args.tv_weight > 0:
        print("Building TV adjacency pairs from training rows...", flush=True)
        t0 = time.time()
        df_tr = df.loc[is_tr_x, ["image_basename", "codec", "knob_tuple_json", "q"]] \
                  .reset_index(drop=True)
        df_tr["local_idx"] = np.arange(len(df_tr), dtype=np.int64)
        df_tr = df_tr.sort_values(
            ["image_basename", "codec", "knob_tuple_json", "q"], kind="stable")
        # Adjacent rows within the same group: shift by 1 and check the
        # group keys still match.
        same_curve = (
            (df_tr["image_basename"].values[1:] == df_tr["image_basename"].values[:-1]) &
            (df_tr["codec"].values[1:] == df_tr["codec"].values[:-1]) &
            (df_tr["knob_tuple_json"].values[1:] == df_tr["knob_tuple_json"].values[:-1]) &
            (df_tr["q"].values[1:] > df_tr["q"].values[:-1])
        )
        lo = df_tr["local_idx"].values[:-1][same_curve]
        hi = df_tr["local_idx"].values[1:][same_curve]
        tv_pairs_train = np.stack([lo, hi], axis=1).astype(np.int64)
        print(f"  {len(tv_pairs_train):,} adjacent-q pairs "
              f"({time.time()-t0:.1f}s)", flush=True)

    # Per-row training weight (1.0 for synthetic, configurable for human
    # rows via the `--human-csv` flag). Applied inside the train loop's
    # MSE term so KADID/TID rows don't get overwhelmed by the synthetic
    # majority. Per-dataset val labels let the trainer report and select
    # on the *mean* over datasets rather than the synthetic-dominated
    # global SROCC.
    w_train = df["train_weight"].to_numpy(dtype=np.float32)[is_tr_x]
    val_dataset_labels = df["dataset"].to_numpy()[is_va_x]

    # dssim co-training: load column from the dataframe if --dssim-weight > 0
    dssim_train = None
    if args.dssim_weight > 0:
        if "dssim" not in df.columns:
            raise SystemExit(
                "--dssim-weight > 0 but no 'dssim' column in training "
                "dataframe; check that --human-csv points to a file with "
                "dssim scores (e.g. training_safe_synthetic.csv has it).")
        dssim_train = df["dssim"].to_numpy(dtype=np.float32)[is_tr_x]
        n_nan = int(np.isnan(dssim_train).sum())
        n_valid = int(np.sum(~np.isnan(dssim_train)))
        print(f"dssim co-training: weight={args.dssim_weight}, "
              f"valid target rows={n_valid:,} / {len(dssim_train):,} "
              f"(NaN={n_nan:,} — masked out, not pushed to quality=100)",
              flush=True)
        # KEEP NaN; the train() loop masks them out of the dssim term so
        # rows without dssim (e.g. sweep parquets) don't get spurious
        # quality=100 targets pulling pred to 100.

    t0 = time.time()
    model, metrics = train(cfg, Xt, y[is_tr_x], g_all[is_tr_x],
                           Xv, y[is_va_x], g_all[is_va_x], device,
                           tv_pairs_train=tv_pairs_train,
                           w_train=w_train,
                           val_dataset_labels=val_dataset_labels,
                           dssim_train=dssim_train)
    train_secs = time.time() - t0

    # Test
    model.eval()
    with torch.no_grad():
        pt = model(torch.from_numpy(Xs).to(device)).cpu().numpy()
    test_mse = float(np.mean((pt - y[is_te_x]) ** 2))
    test_sr = srocc(pt, y[is_te_x])
    test_kr = krocc(pt, y[is_te_x])
    metrics.update({"test_mse": test_mse, "test_srocc": test_sr,
                    "test_krocc": test_kr,
                    "train_secs": round(train_secs, 1),
                    "n_train": int(is_tr_x.sum()),
                    "n_val": int(is_va_x.sum()),
                    "n_test": int(is_te_x.sum())})
    print(f"\nFinal: best_val_srocc={metrics['val_srocc']:.4f} "
          f"test_srocc={test_sr:.4f} test_krocc={test_kr:.4f} "
          f"test_mse={test_mse:.3f}  ({train_secs:.0f}s)")

    # Per-codec breakdown on test
    test_codec = sc[is_te_x]
    print("\nPer-(sweep:codec) test metrics:")
    for sc_id in np.unique(test_codec):
        mask = test_codec == sc_id
        if mask.sum() < 50:
            continue
        a, b = pt[mask], y[is_te_x][mask]
        print(f"  {sc_id:<24} n={mask.sum():>7,}  "
              f"srocc={srocc(a,b):+.4f}  krocc={krocc(a,b):+.4f}  "
              f"mse={float(np.mean((a-b)**2)):.3f}")

    # Save
    ts = time.strftime("%Y%m%dT%H%M%S")
    run_dir = Path(args.out_dir) / f"{ts}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    with open(run_dir / "meta.json", "w") as f:
        json.dump({"config": asdict(cfg), "metrics": metrics,
                   "target_col": target_col,
                   "n_features": int(X.shape[1])}, f, indent=2)
    # Save scaler so the bake step can roundtrip the normalization
    np.savez(run_dir / "scaler.npz", mean=scaler_mean, std=scaler_std)

    # Predictions parquet for downstream analysis
    val_df = df[is_va_x].copy().reset_index(drop=True)
    with torch.no_grad():
        pv = model(torch.from_numpy(Xv).to(device)).cpu().numpy()
    val_df["pred"] = pv
    val_df["target_value"] = y[is_va_x]
    val_df["residual"] = val_df["pred"] - val_df["target_value"]
    keep_pred_cols = ["sweep_id", "codec", "image_basename", "q",
                      "knob_tuple_json", "score_zensim", target_col,
                      "pred", "target_value", "residual"]
    val_df[[c for c in keep_pred_cols if c in val_df.columns]]\
        .to_parquet(run_dir / "predictions_val.parquet",
                    compression="zstd", compression_level=9)
    print(f"\nWrote {run_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
