#!/usr/bin/env python3
"""DIAL-BUG-AUDIT: Scan every bake in zensim/weights/ for the distance-shape +
clamp/squash dial-bug (#176, #177).

For each bake:
1. Skip if it has 'zentrain.output_calibration_spline' metadata (V9-spline-fixed).
2. Score 1000 random rows from V9 anchor parquet (raw, no post).
3. Apply the profile's actual production clamp/squash.
4. Compute min/max/p5/p50/p95/range of post-clamp distribution.
5. Classify: range < 50 -> DIAL-BROKEN, range >= 80 -> DIAL-OK, else MARGINAL.

Bake -> profile mapping per zensim/src/profile.rs (clamp policy noted):
  v0_18_zerobiased_lz4_2026-05-13.bin       -> V0_3                hard clamp
  v0_20_is_calibrated_2026-05-15.bin        -> V0_4 secondary B3   (paired w/ v0_18, soft clamp)
  v22_mix_cv40_konjnd_002_LARGE_iwssim_*    -> V0_5Balanced        hard clamp
  v_compression_persample_2026-05-18.bin    -> V0_5Compression     soft clamp
  v_tuner_2026-05-18.bin                    -> V0_5Tuner           hard clamp
  v_cross_codec_2026-05-19.bin              -> V0_5CrossCodec      soft clamp
  v_tuner_v6_2026-05-19.bin                 -> V0_5TunerV2         hard clamp (after tanh pin)
  v_tuner_v9_2026-05-20.bin                 -> V0_5TunerV3 (SPLINE FIXED — skip)
  v_balanced_v2_2026-05-20.bin              -> V0_5BalancedV2 (SPLINE FIXED — skip)
  v_compression_v2_2026-05-20.bin           -> V0_5CompressionV2 (SPLINE FIXED — skip)
  v05_ensemble_classifier_2026-05-18.bin    -> classifier only (logit, no dial; skip)
  v_compression_2026-05-18.bin              -> ARCHIVED prior compression ship (372feat)
  v0_22_iw_v2_2026-05-16.bin                -> NOT shipped (research bake)
  v0_22_iw_v2_calibrated_2026-05-16.bin     -> NOT shipped (calibrated variant)
  v0_18_2026-05-13.bin                      -> ARCHIVED uncompressed V0_3 source

DEPRECATED STAT MATH: `srocc_sign_tolerant` here is superseded by the
canonical Rust `panel` (zensim-validate/src/bin/panel.rs), whose SROCC is
already polarity-tolerant (.abs()). For NEW work:
    from scripts.lib.zen_stats import srocc   # shells to Rust `panel`
verified to scipy <= 1e-9 by scripts/verify_panel_parity.py. This script's
dial-distribution scan (min/max/p5/p95/range) is its unique value; the
SROCC helper is what's superseded.
"""

from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path("/home/lilith/work/zen/zensim--cross-codec-v8")
PREDICT = REPO / "target/release/predict_features_with_bake"
ZENPREDICT = Path("/home/lilith/work/zen/zenanalyze/target/release/zenpredict")
PARQUET = Path("/mnt/v/zen/zensim-training/2026-05-20-v9-anchors/anchors_v9_372col.parquet")
WEIGHTS_DIR = REPO / "zensim/weights"
N_SAMPLES = 1000
SEED = 20260520


# Bake -> (profile_variant, post_mode)
# post_mode is what the production runtime applies:
#   "clamp"      -> raw.clamp(0, 100)
#   "soft_clamp" -> 100 / (1 + exp(-(raw - 50) / 20))
#   "mapped"     -> mapped:18,0.7  (V0_1 / V0_2 distance-to-score transform)
#   "tanh_pin+clamp" -> tanh pin handled by predict tool, then hard clamp (V0_5TunerV2)
#   "skip"       -> not scored (spline fixed, or classifier, or non-shipping research bake)

BAKE_TABLE = [
    # bake_file, profile_variant, post_mode, status_note
    ("v0_18_zerobiased_lz4_2026-05-13.bin",   "PreviewV0_3",                   "clamp",      "ship"),
    ("v0_20_is_calibrated_2026-05-15.bin",    "PreviewV0_4 (B3 secondary)",    "soft_clamp", "ship"),
    ("v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin",
                                              "PreviewV0_5 / PreviewV0_5Balanced", "clamp",  "ship"),
    ("v_compression_persample_2026-05-18.bin","PreviewV0_5Compression",        "soft_clamp", "ship"),
    ("v_tuner_2026-05-18.bin",                "PreviewV0_5Tuner",              "clamp",      "ship"),
    ("v_cross_codec_2026-05-19.bin",          "PreviewV0_5CrossCodec",         "soft_clamp", "ship"),
    ("v_tuner_v6_2026-05-19.bin",             "PreviewV0_5TunerV2",            "clamp",      "ship (tanh-pinned)"),
    # SPLINE-FIXED (skip per audit spec)
    ("v_tuner_v9_2026-05-20.bin",             "PreviewV0_5TunerV3",            "skip",       "V9 SPLINE FIXED"),
    ("v_balanced_v2_2026-05-20.bin",          "PreviewV0_5BalancedV2",         "skip",       "V9 SPLINE FIXED"),
    ("v_compression_v2_2026-05-20.bin",       "PreviewV0_5CompressionV2",      "skip",       "V9 SPLINE FIXED"),
    # Auxiliary / archived
    ("v05_ensemble_classifier_2026-05-18.bin","PreviewV0_5Ensemble (classifier only)", "skip", "classifier (logit, not dial)"),
    ("v_compression_2026-05-18.bin",          "(archived prior compression ship)", "clamp", "archived"),
    ("v0_22_iw_v2_2026-05-16.bin",            "(research, not shipped)",       "clamp",      "research bake"),
    ("v0_22_iw_v2_calibrated_2026-05-16.bin", "(research, not shipped)",       "clamp",      "research bake"),
    ("v0_18_2026-05-13.bin",                  "(archived V0_3 uncompressed)",  "clamp",      "archived"),
]


def soft_clamp(raw: np.ndarray) -> np.ndarray:
    return 100.0 / (1.0 + np.exp(-(raw - 50.0) / 20.0))


def mapped(raw: np.ndarray, a: float = 18.0, b: float = 0.7) -> np.ndarray:
    d = np.maximum(raw, 0.0)
    return np.clip(100.0 - a * np.power(d, b), 0.0, 100.0)


def hard_clamp(raw: np.ndarray) -> np.ndarray:
    return np.clip(raw, 0.0, 100.0)


def write_features_file(features: np.ndarray, n_cols: int, path: Path) -> None:
    # features is shape (n_rows, 372). Take first n_cols columns.
    f = features[:, :n_cols].astype(np.float32, order="C")
    n_rows = f.shape[0]
    with open(path, "wb") as out:
        out.write(struct.pack("<II", n_cols, n_rows))
        out.write(f.tobytes(order="C"))


def inspect_bake(bake_path: Path) -> dict:
    r = subprocess.run([str(ZENPREDICT), "inspect", str(bake_path)],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def has_spline(meta: dict) -> bool:
    md = meta.get("metadata") or []
    return any(m.get("key") == "zentrain.output_calibration_spline" for m in md)


def score_bake(bake_path: Path, features: np.ndarray, n_in: int) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        tmp = Path(tf.name)
    try:
        write_features_file(features, n_in, tmp)
        r = subprocess.run(
            [str(PREDICT), "--bake", str(bake_path),
             "--bake-post", "raw",
             "--features-file", str(tmp)],
            capture_output=True, text=True, check=True,
        )
        return np.array([float(x) for x in r.stdout.strip().split("\n")], dtype=np.float64)
    finally:
        tmp.unlink(missing_ok=True)


def percentiles(arr: np.ndarray) -> dict:
    return {
        "min":  float(np.min(arr)),
        "max":  float(np.max(arr)),
        "p5":   float(np.percentile(arr, 5)),
        "p50":  float(np.percentile(arr, 50)),
        "p95":  float(np.percentile(arr, 95)),
        "range": float(np.percentile(arr, 95) - np.percentile(arr, 5)),
        "frac_pinned_lo": float(np.mean(arr <= 0.001)),
        "frac_pinned_hi": float(np.mean(arr >= 99.999)),
        "mean": float(np.mean(arr)),
        "std":  float(np.std(arr)),
    }


def classify(range_: float) -> str:
    if range_ < 50.0:
        return "DIAL-BROKEN"
    if range_ >= 80.0:
        return "DIAL-OK"
    return "DIAL-MARGINAL"


def srocc_sign_tolerant(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation, returning |srocc| so distance-shaped + score-shaped both pass."""
    from scipy.stats import spearmanr  # noqa: PLC0415
    rho, _ = spearmanr(x, y)
    if not np.isfinite(rho):
        return float("nan")
    return abs(float(rho))


def main() -> int:
    print(f"# DIAL-BUG audit — {N_SAMPLES} random rows from {PARQUET.name}")
    print(f"# seed={SEED}")
    print()

    # Sample 1000 random rows
    table = pq.read_table(str(PARQUET))
    df = table.to_pandas()
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), size=min(N_SAMPLES, len(df)), replace=False)
    sample = df.iloc[idx].reset_index(drop=True)
    # Features matrix (N, 372)
    feat_cols = [f"f{i}" for i in range(372)]
    X = sample[feat_cols].to_numpy(dtype=np.float32)
    # Ground-truth proxy for sign-tolerant SROCC: human_score (already 0..100 MCOS-aligned where available)
    Y_truth = sample["human_score"].to_numpy(dtype=np.float64)
    # Filter NaN truth rows for SROCC purposes
    Y_finite_mask = np.isfinite(Y_truth)
    print(f"# Loaded {len(df)} rows; sampled {len(sample)}; truth finite: {Y_finite_mask.sum()}")
    print()

    rows = []
    for fname, profile, post, status in BAKE_TABLE:
        bake_path = WEIGHTS_DIR / fname
        if not bake_path.exists():
            print(f"# MISSING: {fname}")
            continue
        meta = inspect_bake(bake_path)
        n_in = int(meta["n_inputs"])
        spline = has_spline(meta)
        if post == "skip":
            skip_reason = "SPLINE-FIXED" if spline else status.upper()
            rows.append({
                "bake": fname,
                "profile": profile,
                "n_in": n_in,
                "post": post,
                "status": "SKIP",
                "reason": skip_reason,
                "dist": None,
                "dist_raw": None,
                "verdict": "SKIP",
                "abs_srocc_raw": None,
            })
            print(f"  SKIP  {fname}  ({skip_reason})")
            continue
        # Score raw
        raw = score_bake(bake_path, X, n_in)
        # Apply production clamp/squash
        if post == "clamp":
            scored = hard_clamp(raw)
        elif post == "soft_clamp":
            scored = soft_clamp(raw)
        elif post == "mapped":
            scored = mapped(raw)
        elif post == "tanh_pin+clamp":
            # tanh pin already inside predict tool path (V0_5TunerV2)
            scored = hard_clamp(raw)
        else:
            raise ValueError(f"unknown post {post!r}")
        d_raw = percentiles(raw)
        d_post = percentiles(scored)
        abs_srocc = srocc_sign_tolerant(raw[Y_finite_mask], Y_truth[Y_finite_mask])
        verdict = classify(d_post["range"])
        rows.append({
            "bake": fname,
            "profile": profile,
            "n_in": n_in,
            "post": post,
            "status": status,
            "reason": "",
            "dist": d_post,
            "dist_raw": d_raw,
            "verdict": verdict,
            "abs_srocc_raw": abs_srocc,
        })
        print(f"  {verdict:13s}  {fname:55s}  n_in={n_in:3d}  post={post:12s}  "
              f"range={d_post['range']:6.2f}  p5..p95=[{d_post['p5']:6.2f}, {d_post['p95']:6.2f}]  "
              f"|SROCC_raw|={abs_srocc:.3f}")

    # Write structured JSON
    out_json = REPO / "benchmarks/dial_bug_audit_2026-05-20.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
