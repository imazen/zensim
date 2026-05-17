#!/usr/bin/env python3
"""V_20b — distortion-manifold contrastive pre-training (Phase 2).

Trains a 228 → 64-dim encoder using triplet loss on unlabeled
(ref, dist) pairs from the safe-synth corpus. Per-image triplets are
sampled by ordering pairs by `human_score` (= ssim2) and selecting
anchor / positive (similar score) / negative (far score).

Output: encoder weights checkpoint (`encoder.pt`) that Phase 3
fine-tunes into a complete head.

## Usage

  python3 scripts/v_next/v0_20b/contrastive_pretrain.py \\
    --features-parquet /mnt/v/zen/zensim-training/2026-05-07/v06-features/safe_synth_ssim2_features.parquet \\
    --max-features 228 \\
    --embedding-dim 64 \\
    --margin 0.5 \\
    --epochs 200 \\
    --batch-size 256 \\
    --triplet-radius-similar 5 \\
    --triplet-radius-far 20 \\
    --out /tmp/v0_20b_encoder.pt

## Design notes

- `human_score` (ssim2) is the distortion-strength proxy because the
  parquet doesn't carry the codec quality setting (`zq`). This is a
  slight self-supervision violation per Su 2023's pure-zq protocol
  but uses information available in our actual data. The encoder
  learns "similar-ssim2-pairs cluster" — fine-tune then maps to MOS.
- Triplet loss with margin α (default 0.5).
- 228 → H → 64 encoder (H = embedding_dim * 2). LeakyReLU. Adam.
- Standardization fit on TRAIN features only; saved alongside.

## Output

`encoder.pt` is a torch state_dict with keys:
  - `scaler_mean: [228]` (f32)
  - `scaler_scale: [228]` (f32)
  - `w0: [228, H]`, `b0: [H]` (LeakyReLU layer)
  - `w1: [H, embedding_dim]`, `b1: [embedding_dim]` (Identity output)
  - `hyperparams`: dict of margin / dim / epochs / etc.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_parquet(path: Path, max_features: int):
    import pyarrow.parquet as pq

    t0 = time.perf_counter()
    tbl = pq.read_table(str(path))
    cols = tbl.column_names
    feat_cols = [c for c in cols if c.startswith("f") and c[1:].isdigit()]
    feat_cols.sort(key=lambda c: int(c[1:]))
    feat_cols = feat_cols[:max_features]
    n = tbl.num_rows
    mos = tbl["human_score"].to_numpy().astype(np.float32)
    refs = tbl["ref_basename"].to_pylist()
    arrs = [tbl[c].to_numpy().astype(np.float32) for c in feat_cols]
    feats = np.stack(arrs, axis=1)
    print(
        f"loaded {path.name}: n={n}, {len(feat_cols)} features in {time.perf_counter() - t0:.1f}s"
    )
    return refs, mos, feats, feat_cols


class TripletSampler:
    """Per-image triplet sampler over rows ordered by `human_score`.

    Within each `ref_basename` group, sorts rows by human_score and
    picks (anchor, positive, negative) where positive is within
    `radius_similar` rank positions of anchor and negative is at
    least `radius_far` rank positions away.

    Each `__iter__` yields `batch_size` triplets per call up to
    `n_batches` total.
    """

    def __init__(
        self,
        refs: list[str],
        mos: np.ndarray,
        radius_similar: int,
        radius_far: int,
        seed: int,
    ) -> None:
        groups: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(refs):
            groups[r].append(i)
        for r, idxs in groups.items():
            idxs.sort(key=lambda i: mos[i])
        self.groups = [v for v in groups.values() if len(v) >= 3]
        self.rng = random.Random(seed)
        self.radius_similar = radius_similar
        self.radius_far = radius_far
        print(
            f"triplet sampler: {len(self.groups)} groups with >=3 rows "
            f"(range {min(len(g) for g in self.groups)}..{max(len(g) for g in self.groups)} rows/group)"
        )

    def sample_batch(self, n: int) -> list[tuple[int, int, int]]:
        out: list[tuple[int, int, int]] = []
        while len(out) < n:
            g = self.rng.choice(self.groups)
            L = len(g)
            anchor_pos = self.rng.randint(0, L - 1)
            # Positive: same group, within radius_similar
            lo = max(0, anchor_pos - self.radius_similar)
            hi = min(L - 1, anchor_pos + self.radius_similar)
            pos_candidates = [p for p in range(lo, hi + 1) if p != anchor_pos]
            if not pos_candidates:
                continue
            pos_pos = self.rng.choice(pos_candidates)
            # Negative: same group, distance >= radius_far
            neg_candidates = [
                p for p in range(L) if abs(p - anchor_pos) >= self.radius_far
            ]
            if not neg_candidates:
                continue
            neg_pos = self.rng.choice(neg_candidates)
            out.append((g[anchor_pos], g[pos_pos], g[neg_pos]))
        return out


class Encoder(nn.Module):
    def __init__(self, n_inputs: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.fc0 = nn.Linear(n_inputs, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.fc0(x), negative_slope=0.01)
        return self.fc1(h)


def fit_scaler(feats: np.ndarray):
    mean = feats.mean(axis=0)
    std = feats.std(axis=0).clip(min=1e-8)
    return mean.astype(np.float32), std.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-parquet", required=True, type=Path)
    ap.add_argument("--max-features", type=int, default=228)
    ap.add_argument("--embedding-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=128,
                    help="Encoder hidden width (default 128).")
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--batches-per-epoch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--triplet-radius-similar", type=int, default=5)
    ap.add_argument("--triplet-radius-far", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    refs, mos, feats, _ = load_parquet(args.features_parquet, args.max_features)
    scaler_mean, scaler_scale = fit_scaler(feats)
    feats_std = (feats - scaler_mean) / scaler_scale
    n_inputs = feats_std.shape[1]
    print(f"n_inputs={n_inputs}, embedding_dim={args.embedding_dim}, hidden={args.hidden_dim}")

    sampler = TripletSampler(
        refs,
        mos,
        radius_similar=args.triplet_radius_similar,
        radius_far=args.triplet_radius_far,
        seed=args.seed,
    )

    device = torch.device(args.device)
    model = Encoder(n_inputs, args.hidden_dim, args.embedding_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    feats_torch = torch.from_numpy(feats_std).to(device)
    print(f"feats on device: {feats_torch.shape} {feats_torch.dtype}")

    best_loss = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        t_ep = time.perf_counter()
        total_loss = 0.0
        n_batches = 0
        for _ in range(args.batches_per_epoch):
            triplets = sampler.sample_batch(args.batch_size)
            a_idx = torch.tensor([t[0] for t in triplets], device=device)
            p_idx = torch.tensor([t[1] for t in triplets], device=device)
            n_idx = torch.tensor([t[2] for t in triplets], device=device)
            a = model(feats_torch[a_idx])
            p = model(feats_torch[p_idx])
            n = model(feats_torch[n_idx])
            d_ap = ((a - p) ** 2).sum(dim=1)
            d_an = ((a - n) ** 2).sum(dim=1)
            loss = F.relu(d_ap - d_an + args.margin).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        avg = total_loss / n_batches
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d} | avg_triplet_loss={avg:.4f} | best={best_loss:.4f} | t={time.perf_counter()-t_ep:.1f}s")

    # Save encoder + scaler + hyperparams
    args.out.parent.mkdir(parents=True, exist_ok=True)
    state = best_state if best_state is not None else model.state_dict()
    payload = {
        "scaler_mean": scaler_mean.astype(np.float32),
        "scaler_scale": scaler_scale.astype(np.float32),
        "w0": state["fc0.weight"].numpy().astype(np.float32),
        "b0": state["fc0.bias"].numpy().astype(np.float32),
        "w1": state["fc1.weight"].numpy().astype(np.float32),
        "b1": state["fc1.bias"].numpy().astype(np.float32),
        "hyperparams": {
            "n_inputs": int(n_inputs),
            "hidden_dim": int(args.hidden_dim),
            "embedding_dim": int(args.embedding_dim),
            "margin": float(args.margin),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "batches_per_epoch": int(args.batches_per_epoch),
            "lr": float(args.lr),
            "triplet_radius_similar": int(args.triplet_radius_similar),
            "triplet_radius_far": int(args.triplet_radius_far),
            "best_triplet_loss": float(best_loss),
        },
    }
    # Save as npz so we can load without torch later (the fine-tune
    # script may run in plain numpy + torch.from_numpy).
    np.savez(args.out, **{k: v for k, v in payload.items() if k != "hyperparams"})
    args.out.with_suffix(".hp.json").write_text(json.dumps(payload["hyperparams"], indent=2))
    print(f"saved encoder to {args.out} (best loss {best_loss:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
