#!/usr/bin/env python3
"""Build a v0.2 ↔ v0.3 rough score equivalence table.

v0.2 = linear weighted sum of 228 (basic + peak) features → distance d →
       score = clamp(100 − 18·|d|^0.7, 0, 100)
v0.3 = Tuner v5 MLP bake (372 features → 128 → 128 → tanh-pin →
       PCHIP spline → score)

Approach: load the 68,788 cross-codec equivalence pairs (372 features
per `fa_` row), compute both scores, bin by v0.2 score, report median
v0.3 score per bin.
"""
import re
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path("/home/lilith/work/zen/zensim")
PROFILE_RS = REPO / "zensim/src/profile.rs"
BAKE_V03 = REPO / "zensim/weights/v_tuner_v11_2026-05-24.bin"
PREDICT = REPO / "target/release/predict_features_with_bake"
PARQUET = Path("/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet")


def load_v02_weights() -> np.ndarray:
    """Extract WEIGHTS_PREVIEW_V0_2 from profile.rs."""
    text = PROFILE_RS.read_text()
    m = re.search(r"pub static WEIGHTS_PREVIEW_V0_2: \[f64; 228\] = \[(.*?)\];",
                  text, re.DOTALL)
    if not m:
        raise RuntimeError("WEIGHTS_PREVIEW_V0_2 not found")
    body = m.group(1)
    floats = re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", body)
    weights = np.array([float(f) for f in floats], dtype=np.float64)
    if len(weights) != 228:
        raise RuntimeError(f"expected 228 weights, got {len(weights)}")
    return weights


def score_v02(features_228: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """v0.2 linear path: distance d = dot(weights, features), then
    score = clamp(100 − 18·|d|^0.7, 0, 100)."""
    # Per-row dot product. features shape: (N, 228).
    d = features_228 @ weights
    # Apply score mapping with score_mapping_a=18.0, score_mapping_b=0.7.
    score = 100.0 - 18.0 * np.power(np.abs(d), 0.7)
    return np.clip(score, 0.0, 100.0)


def score_v03(features_372: np.ndarray, bake: Path) -> np.ndarray:
    """v0.3 bake forward pass via predict_features_with_bake."""
    n_rows, n_features = features_372.shape
    buf = bytearray()
    buf += struct.pack("<II", n_features, n_rows)
    buf += features_372.astype(np.float32, copy=False).tobytes(order="C")
    with tempfile.NamedTemporaryFile(suffix=".features.bin", delete=False) as f:
        f.write(buf)
        feats_path = f.name
    try:
        out = subprocess.check_output(
            [str(PREDICT), "--bake", str(bake),
             "--bake-post", "clamp", "--features-file", feats_path],
        )
    finally:
        Path(feats_path).unlink(missing_ok=True)
    return np.array([float(x) for x in out.decode().split()], dtype=np.float64)


def main():
    print(f"[1/4] loading WEIGHTS_PREVIEW_V0_2 from profile.rs ...")
    weights = load_v02_weights()
    print(f"      228 weights, non-zero: {(weights != 0).sum()}")

    print(f"[2/4] loading {PARQUET.name} ...")
    table = pq.read_table(PARQUET)
    n = table.num_rows
    fa_cols = [f"fa_{i}" for i in range(372)]
    fa = np.column_stack([table.column(c).to_numpy(zero_copy_only=False)
                          for c in fa_cols]).astype(np.float32, copy=False)
    print(f"      {n} rows × 372 features")

    print(f"[3/4] scoring v0.2 (linear) + v0.3 (bake) ...")
    fa_228 = fa[:, :228].astype(np.float64)
    v02 = score_v02(fa_228, weights)
    print(f"      v0.2: p5={np.quantile(v02, 0.05):.2f} "
          f"p50={np.median(v02):.2f} p95={np.quantile(v02, 0.95):.2f}")
    v03 = score_v03(fa, BAKE_V03)
    print(f"      v0.3: p5={np.quantile(v03, 0.05):.2f} "
          f"p50={np.median(v03):.2f} p95={np.quantile(v03, 0.95):.2f}")

    print(f"[4/4] building equivalence table ...")
    # Bin v0.2 scores in 10-unit bands.
    bins = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 40), (40, 50),
            (50, 60), (60, 70), (70, 80), (80, 90), (90, 95),
            (95, 99), (99, 101)]
    rows = []
    print(f"\nv0.2 band     n     v0.3 p25 → p50 → p75   (rough equivalence)")
    print("-" * 70)
    for lo, hi in bins:
        mask = (v02 >= lo) & (v02 < hi)
        if mask.sum() == 0:
            continue
        v03_sub = v03[mask]
        p25 = np.quantile(v03_sub, 0.25)
        p50 = np.median(v03_sub)
        p75 = np.quantile(v03_sub, 0.75)
        rows.append((lo, hi, int(mask.sum()), p25, p50, p75))
        print(f"  {lo:3.0f}..{hi:3.0f}  {mask.sum():>6}    "
              f"{p25:5.1f} → {p50:5.1f} → {p75:5.1f}")

    print(f"\nSpearman rank correlation v0.2 ↔ v0.3 (post-clamp):")
    from scipy.stats import spearmanr
    rho, _ = spearmanr(v02, v03)
    print(f"  SROCC = {rho:.4f}")

    # Also emit a more user-facing "if v0.2 user targeted X, v0.3 X' approx" table.
    print(f"\n=== ROUND-NUMBER LOOKUP (v0.2 score → v0.3 median score) ===")
    print(f"  v0.2 target   →  v0.3 median (p25 .. p75 spread)")
    for target in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]:
        # ±2 unit window around the target
        mask = (v02 >= target - 2) & (v02 < target + 2)
        if mask.sum() < 30:
            continue
        v03_sub = v03[mask]
        p25 = np.quantile(v03_sub, 0.25)
        p50 = np.median(v03_sub)
        p75 = np.quantile(v03_sub, 0.75)
        print(f"  v0.2={target:>4}  →  v0.3 median={p50:5.1f}  "
              f"(p25={p25:5.1f}, p75={p75:5.1f}, n={mask.sum()})")


if __name__ == "__main__":
    main()
