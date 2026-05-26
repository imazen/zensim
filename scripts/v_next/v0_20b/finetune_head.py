#!/usr/bin/env python3
"""V_20b Phase 3 — fine-tune a head on top of the contrastive encoder.

Loads `encoder.pt` from Phase 2 (`contrastive_pretrain.py`), freezes
or partially-unfreezes the encoder, attaches a `embedding_dim → 1`
head, trains on labeled corpora (KADID + TID + CID22-train + KonJND)
using RankNet-style pairwise loss to align the output scale with MOS.

Output: full V_20b bake-ready npz with all of (scaler, encoder, head)
weights. Phase 4 (`bake_v3.py`) converts to ZNPR v3.

## Usage

  python3 scripts/v_next/v0_20b/finetune_head.py \\
    --encoder /tmp/v0_20b_encoder.pt.npz \\
    --features-csv kadid:/mnt/v/zen/zensim-training/2026-05-14-clean/kadid_features.csv \\
    --features-csv tid:/mnt/v/zen/zensim-training/2026-05-14-clean/tid_features.csv \\
    --features-csv konjnd:/mnt/v/zen/zensim-training/2026-05-14-clean/konjnd_aligned_features.csv \\
    --freeze-encoder \\
    --epochs 100 \\
    --batch-size 1024 \\
    --pairs-per-epoch 50000 \\
    --out /tmp/v0_20b_full.npz

## Design

- Encoder = 228 → hidden_dim → embedding_dim (frozen by default)
- Head = embedding_dim → 1 (Identity output)
- Loss = RankNet (pairwise sigmoid cross-entropy on score deltas)
- Pair sampling: per-group, draw 2 rows, optimize so higher MOS gets
  higher score
- val_mean = min per-group SROCC, same as Rust trainer

When `--freeze-encoder` is false, encoder learns at lr/10 alongside.

MIGRATION CANDIDATE (stat math): the inline `spearman_abs` is a per-epoch
in-loop monitor, so it can't shell to the Rust `panel` bin mid-training
without per-epoch subprocess overhead. The canonical stat home is
zensim_validate::panel (zensim-validate/src/bin/panel.rs); for the FINAL
held-out report this script should use `from scripts.lib.zen_stats import
panel` (or `bake_verdict` once the head is baked). Keep the in-loop monitor
lightweight but treat its number as indicative, not a ship verdict.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_csv(path: Path, max_features: int):
    t0 = time.perf_counter()
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        score_idx = header.index("human_score")
        feat_idx = [
            header.index(c)
            for c in header
            if c.startswith("f") and c[1:].isdigit()
        ][:max_features]
        mos_list = []
        feat_rows = []
        for row in r:
            try:
                mos = float(row[score_idx]) * 100.0
                feats = [float(row[i]) for i in feat_idx]
            except (ValueError, IndexError):
                continue
            mos_list.append(mos)
            feat_rows.append(feats)
    mos = np.array(mos_list, dtype=np.float32)
    feats = np.array(feat_rows, dtype=np.float32)
    print(
        f"  {path.name}: n={len(mos)} × {feats.shape[1]} feats in {time.perf_counter()-t0:.1f}s"
    )
    return mos, feats


class V0_20b(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        hidden_dim: int,
        embedding_dim: int,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
    ):
        super().__init__()
        self.scaler_mean = nn.Parameter(
            torch.from_numpy(scaler_mean), requires_grad=False
        )
        self.scaler_scale = nn.Parameter(
            torch.from_numpy(scaler_scale), requires_grad=False
        )
        self.encoder_fc0 = nn.Linear(n_inputs, hidden_dim)
        self.encoder_fc1 = nn.Linear(hidden_dim, embedding_dim)
        self.head = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.scaler_mean) / self.scaler_scale
        h = F.leaky_relu(self.encoder_fc0(x), negative_slope=0.01)
        e = self.encoder_fc1(h)
        return self.head(e).squeeze(-1)


def spearman_abs(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return 0.0
    return float(abs(np.corrcoef(ra, rb)[0, 1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, type=Path,
                    help="Encoder npz from contrastive_pretrain.py")
    ap.add_argument(
        "--features-csv",
        action="append",
        required=True,
        help="NAME:PATH per labeled corpus (kadid/tid/konjnd/cid22-train).",
    )
    ap.add_argument("--max-features", type=int, default=228)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--pairs-per-epoch", type=int, default=50_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-encoder-ratio", type=float, default=0.1,
                    help="If unfrozen, encoder lr = lr * this. Default 0.1.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load encoder
    enc = np.load(args.encoder)
    scaler_mean = enc["scaler_mean"]
    scaler_scale = enc["scaler_scale"]
    w0, b0 = enc["w0"], enc["b0"]
    w1, b1 = enc["w1"], enc["b1"]
    n_inputs = w0.shape[1]
    hidden_dim = w0.shape[0]
    embedding_dim = w1.shape[0]
    print(f"loaded encoder: {n_inputs} -> {hidden_dim} -> {embedding_dim}")

    # Load groups
    groups = []
    for spec in args.features_csv:
        name, path = spec.split(":", 1)
        mos, feats = load_csv(Path(path), args.max_features)
        groups.append({"name": name, "mos": mos, "feats": feats})

    device = torch.device(args.device)
    model = V0_20b(n_inputs, hidden_dim, embedding_dim, scaler_mean, scaler_scale).to(
        device
    )
    # Load encoder weights
    with torch.no_grad():
        model.encoder_fc0.weight.copy_(torch.from_numpy(w0))
        model.encoder_fc0.bias.copy_(torch.from_numpy(b0))
        model.encoder_fc1.weight.copy_(torch.from_numpy(w1))
        model.encoder_fc1.bias.copy_(torch.from_numpy(b1))

    if args.freeze_encoder:
        for p in model.encoder_fc0.parameters():
            p.requires_grad = False
        for p in model.encoder_fc1.parameters():
            p.requires_grad = False
        opt = torch.optim.Adam(model.head.parameters(), lr=args.lr)
    else:
        enc_params = list(model.encoder_fc0.parameters()) + list(
            model.encoder_fc1.parameters()
        )
        opt = torch.optim.Adam(
            [
                {"params": enc_params, "lr": args.lr * args.lr_encoder_ratio},
                {"params": model.head.parameters(), "lr": args.lr},
            ]
        )

    # Pre-load features to device per group
    for g in groups:
        g["feats_t"] = torch.from_numpy(g["feats"]).to(device)
        g["mos_t"] = torch.from_numpy(g["mos"]).to(device)

    rng = random.Random(args.seed)

    best_val = -float("inf")
    best_state = None
    for epoch in range(args.epochs):
        t_ep = time.perf_counter()
        total_loss = 0.0
        n_done = 0
        per_epoch_target = args.pairs_per_epoch
        while n_done < per_epoch_target:
            this_n = min(args.batch_size, per_epoch_target - n_done)
            # Sample group + pairs
            g_pick = groups[rng.randrange(len(groups))]
            n_rows = len(g_pick["mos"])
            if n_rows < 2:
                continue
            i = torch.randint(0, n_rows, (this_n,), device=device)
            j = torch.randint(0, n_rows, (this_n,), device=device)
            mask = i != j
            i = i[mask]
            j = j[mask]
            if len(i) == 0:
                continue
            f_i = g_pick["feats_t"][i]
            f_j = g_pick["feats_t"][j]
            mos_i = g_pick["mos_t"][i]
            mos_j = g_pick["mos_t"][j]
            s_i = model(f_i)
            s_j = model(f_j)
            # Higher mos = better. We want s_i > s_j iff mos_i > mos_j.
            # RankNet: P(i>j) = sigmoid(s_i - s_j); target = (mos_i > mos_j).
            target = (mos_i > mos_j).float()
            loss = F.binary_cross_entropy_with_logits(s_i - s_j, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(i)
            n_done += len(i)
        avg = total_loss / max(n_done, 1)
        # Validation: per-group SROCC
        srccs = []
        with torch.no_grad():
            for g in groups:
                preds = model(g["feats_t"]).cpu().numpy()
                srccs.append(spearman_abs(g["mos"], preds))
        val_mean = min(srccs)
        if val_mean > best_val:
            best_val = val_mean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            per_g = " ".join(f"{g['name']}={s:.4f}" for g, s in zip(groups, srccs))
            print(
                f"epoch {epoch:3d} | loss={avg:.4f} | val_min={val_mean:.4f} (best={best_val:.4f}) | {per_g} | t={time.perf_counter()-t_ep:.1f}s"
            )

    # Save full model (encoder + head) as npz
    state = best_state if best_state is not None else model.state_dict()
    payload = {
        "scaler_mean": scaler_mean.astype(np.float32),
        "scaler_scale": scaler_scale.astype(np.float32),
        "encoder_w0": state["encoder_fc0.weight"].numpy().astype(np.float32),
        "encoder_b0": state["encoder_fc0.bias"].numpy().astype(np.float32),
        "encoder_w1": state["encoder_fc1.weight"].numpy().astype(np.float32),
        "encoder_b1": state["encoder_fc1.bias"].numpy().astype(np.float32),
        "head_w": state["head.weight"].numpy().astype(np.float32),
        "head_b": state["head.bias"].numpy().astype(np.float32),
        "n_inputs": np.array([n_inputs], dtype=np.int32),
        "hidden_dim": np.array([hidden_dim], dtype=np.int32),
        "embedding_dim": np.array([embedding_dim], dtype=np.int32),
        "best_val_min": np.array([best_val], dtype=np.float32),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **payload)
    print(f"saved V_20b model to {args.out} (best val_min {best_val:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
