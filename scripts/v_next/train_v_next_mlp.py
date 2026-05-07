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
    print(f"Loading {len(parqs)} parquets:")
    keep_meta = ["sweep_id", "codec", "image_basename", "q", "knob_tuple_json",
                 "score_zensim", "score_ssim2",
                 "score_butteraugli_max", "score_butteraugli_pnorm3",
                 "metric_runtime", "content_class"]
    feat_cols = [f"feat_{i}" for i in range(228)]
    frames = []
    for p in parqs:
        cols_avail = pq.ParquetFile(p).schema.names
        cols = [c for c in keep_meta + feat_cols if c in cols_avail]
        df = pq.read_table(p, columns=cols).to_pandas()
        print(f"  {p.name}: {len(df):,} rows × {len(cols)} cols")
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    print(f"Total: {len(full):,} rows")
    return full


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
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feat_cols = [c for c in df.columns if c.startswith("feat_")
                 and int(c[5:]) < 228]
    feat_cols = sorted(feat_cols, key=lambda c: int(c[5:]))
    n_features = len(feat_cols)
    print(f"Using {n_features} feature columns: feat_0..feat_{n_features-1}")
    has_y = df[target_col].notna() & df[target_col].abs().lt(1e6)
    has_x = df[feat_cols].notna().all(axis=1)
    keep = has_y & has_x
    print(f"Dropping {(~keep).sum():,} rows with NaN/inf in target or features")
    sub = df[keep]
    X = sub[feat_cols].to_numpy(dtype=np.float32)
    y = sub[target_col].to_numpy(dtype=np.float32)
    images = sub["image_basename"].to_numpy()
    sweep_codec = (sub["sweep_id"] + ":" + sub["codec"]).to_numpy()
    return X, y, images, sweep_codec


class MLP(torch.nn.Module):
    def __init__(self, n_in: int, hidden: list[int]):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(torch.nn.LeakyReLU(negative_slope=0.01))
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def ranknet_loss(pred: torch.Tensor, target: torch.Tensor,
                  groups: torch.Tensor, max_pairs_per_group: int = 64
                  ) -> torch.Tensor:
    """Pairwise sigmoid loss within each group (image_basename id)."""
    device = pred.device
    losses = []
    unique_groups, counts = torch.unique(groups, return_counts=True)
    for g, c in zip(unique_groups.tolist(), counts.tolist()):
        if c < 2:
            continue
        idx = torch.where(groups == g)[0]
        if c > max_pairs_per_group:
            sel = idx[torch.randperm(c, device=device)[:max_pairs_per_group]]
        else:
            sel = idx
        p = pred[sel].unsqueeze(1) - pred[sel].unsqueeze(0)
        t = target[sel].unsqueeze(1) - target[sel].unsqueeze(0)
        sign = torch.sign(t)
        mask = (sign != 0)
        if mask.sum() == 0:
            continue
        # Bradley–Terry / RankNet: -log sigmoid(sign * p)
        l = torch.nn.functional.softplus(-sign[mask] * p[mask])
        losses.append(l.mean())
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


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


def train(cfg: TrainConfig, X_train, y_train, g_train,
          X_val, y_val, g_val, device) -> tuple[MLP, dict]:
    torch.manual_seed(cfg.seed)
    n_in = X_train.shape[1]
    model = MLP(n_in, cfg.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    # Pre-bake tensors on GPU
    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    gt = torch.from_numpy(g_train).to(device)
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    best = {"epoch": -1, "val_srocc": -1, "val_mse": math.inf,
            "state": None}

    n = Xt.size(0)
    bs = min(cfg.batch_size, n)
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
            pred = model(x)
            mse = torch.mean((pred - y) ** 2)
            if cfg.loss == "mse":
                loss = mse
            elif cfg.loss == "ranknet":
                loss = ranknet_loss(pred, y, g)
            elif cfg.loss == "mse_rank":
                rk = ranknet_loss(pred, y, g)
                loss = mse + cfg.rank_weight * rk
            else:
                raise ValueError(cfg.loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1

        model.eval()
        with torch.no_grad():
            pv = model(Xv).detach().cpu().numpy()
        val_mse = float(np.mean((pv - y_val) ** 2))
        val_sr = srocc(pv, y_val)
        val_kr = krocc(pv, y_val)
        if val_sr > best["val_srocc"]:
            best = {"epoch": ep, "val_srocc": val_sr, "val_mse": val_mse,
                    "val_krocc": val_kr,
                    "state": {k: v.detach().clone().cpu()
                              for k, v in model.state_dict().items()}}
        print(f"  epoch {ep:3d}  train_loss={ep_loss/max(1,n_batches):.4f}  "
              f"val_mse={val_mse:.3f}  val_srocc={val_sr:.4f}  "
              f"val_krocc={val_kr:.4f}", flush=True)

    if best["state"]:
        model.load_state_dict(best["state"])
    metrics = {k: v for k, v in best.items() if k != "state"}
    return model, metrics


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
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    target_col = {
        "ssim2": "score_ssim2",
        "butteraugli": "score_butteraugli_max",
        "butteraugli_p3": "score_butteraugli_pnorm3",
        "zensim": "score_zensim",
    }[args.target]

    df = load_parquets(Path(args.input_dir), args.sweeps.split(","))
    if args.max_rows:
        df = df.sample(min(args.max_rows, len(df)), random_state=args.seed)\
               .reset_index(drop=True)
        print(f"Subsampled to {len(df):,} rows")

    is_tr, is_va, is_te = make_split(df, args.val_frac, args.test_frac, args.seed)
    X, y, images, sc = build_arrays(df, target_col)

    # Re-mask after dropna so masks align with X/y
    keep_idx = df.index[df[target_col].notna()
                        & df[[c for c in df.columns
                              if c.startswith("feat_") and int(c[5:]) < 228]]
                        .notna().all(axis=1)]
    is_tr_x = is_tr[keep_idx]
    is_va_x = is_va[keep_idx]
    is_te_x = is_te[keep_idx]

    # Image-id encoding for ranknet groups (only need within-train uniqueness)
    img_to_id = {n: i for i, n in enumerate(np.unique(images))}
    g_all = np.array([img_to_id[n] for n in images], dtype=np.int64)

    cfg = TrainConfig(
        target=args.target, loss=args.loss,
        hidden=[int(h) for h in args.hidden.split(",")],
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, dropout=args.dropout,
        rank_weight=args.rank_weight, seed=args.seed)

    print(f"Config: {cfg}")
    print(f"Train rows: {is_tr_x.sum():,}  Val rows: {is_va_x.sum():,}  "
          f"Test rows: {is_te_x.sum():,}")

    t0 = time.time()
    model, metrics = train(cfg,
                            X[is_tr_x], y[is_tr_x], g_all[is_tr_x],
                            X[is_va_x], y[is_va_x], g_all[is_va_x],
                            device)
    train_secs = time.time() - t0

    # Test
    model.eval()
    with torch.no_grad():
        pt = model(torch.from_numpy(X[is_te_x]).to(device)).cpu().numpy()
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
    # Predictions parquet for downstream analysis
    val_df = df.iloc[keep_idx][is_va_x].copy().reset_index(drop=True)
    with torch.no_grad():
        pv = model(torch.from_numpy(X[is_va_x]).to(device)).cpu().numpy()
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
