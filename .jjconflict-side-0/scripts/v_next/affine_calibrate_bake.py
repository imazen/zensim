#!/usr/bin/env python3
"""Apply post-hoc affine calibration to a trained MLP run.

The model output `y` is the result of `Linear(in_dim=h, out_dim=1)`
on the penultimate hidden state: `y = W·h + b` (W shape [1, h], b
shape [1]). Composing with `y' = α + β·y` is mathematically
equivalent to setting `W' = β·W` and `b' = β·b + α`. This means the
calibration is a zero-runtime-cost change — we just rewrite the
final-layer weights of `model.pt` and re-bake.

The model's training objective (`mse_rank`) is rank-invariant under
affine output transforms, so this preserves all relative SROCC
numbers exactly while shifting the absolute score range.

Usage:
    python3 affine_calibrate_bake.py \\
        --in-run-dir <path> --alpha A --beta B --out-bake-dir <path>

Output: a new run dir at <out-bake-dir> with model.pt + meta.json
+ scaler.npz cloned from --in-run-dir, but model.pt's final layer
modified. Caller then runs bake_to_znpr.py on that new dir.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-run-dir", required=True, type=Path)
    ap.add_argument("--out-run-dir", required=True, type=Path)
    ap.add_argument("--alpha", type=float, required=True,
                    help="affine offset: y' = alpha + beta*y")
    ap.add_argument("--beta", type=float, required=True,
                    help="affine scale: y' = alpha + beta*y")
    args = ap.parse_args()

    in_dir: Path = args.in_run_dir
    out_dir: Path = args.out_run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clone meta.json + scaler.npz unchanged — calibration only touches
    # the final layer's parameters.
    for fname in ("meta.json", "scaler.npz"):
        src = in_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)

    # Load PyTorch state_dict.
    sd = torch.load(in_dir / "model.pt", map_location="cpu",
                    weights_only=True)

    # Identify the FINAL Linear layer. The MLP class constructs
    # layers as torch.nn.Sequential of Linear/LeakyReLU pairs, so
    # state_dict keys look like:
    #   "net.0.weight" / "net.0.bias"  (first Linear)
    #   "net.2.weight" / "net.2.bias"  (second Linear)
    #   ...
    #   "net.<N>.weight" / "net.<N>.bias"  (final Linear, output dim = 1)
    weight_keys = sorted(
        [k for k in sd.keys() if k.endswith(".weight")],
        key=lambda k: int(k.split(".")[-2]),
    )
    if not weight_keys:
        raise SystemExit(f"No .weight tensors found in {in_dir}/model.pt")

    final_w_key = weight_keys[-1]
    final_b_key = final_w_key[:-len(".weight")] + ".bias"
    if final_b_key not in sd:
        raise SystemExit(f"Expected matching bias for {final_w_key}")

    W = sd[final_w_key]
    b = sd[final_b_key]
    if W.shape[0] != 1:
        raise SystemExit(
            f"Final layer out_dim != 1 (got {tuple(W.shape)}). Calibration "
            f"only sensible for scalar regression heads."
        )

    print(f"Final layer: {final_w_key} {tuple(W.shape)}, "
          f"{final_b_key} {tuple(b.shape)}")
    print(f"Before: bias={b.item():.4f}, weight_norm={W.norm().item():.4f}")

    # Apply y' = alpha + beta * (W·h + b) = (beta*W)·h + (beta*b + alpha)
    sd[final_w_key] = args.beta * W
    sd[final_b_key] = args.beta * b + args.alpha

    print(f"After:  bias={sd[final_b_key].item():.4f}, "
          f"weight_norm={sd[final_w_key].norm().item():.4f}")

    torch.save(sd, out_dir / "model.pt")
    print(f"Wrote {out_dir / 'model.pt'}")

    # Annotate meta.json with the calibration applied.
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["affine_calibration"] = {
            "alpha": args.alpha,
            "beta": args.beta,
            "source_run_dir": str(in_dir.resolve()),
            "note": "Applied to final Linear layer's weight and bias. "
                    "Preserves all relative-rank metrics (SROCC, KROCC).",
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Annotated {meta_path}")


if __name__ == "__main__":
    main()
