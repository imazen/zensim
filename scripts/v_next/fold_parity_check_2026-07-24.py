#!/usr/bin/env python3
"""Fold-vs-ext parity analysis for the Folded720 extraction regime.

Compares the folded ONE-pass extraction (`ZENSIM_AB_MODE=fold`) against the
research two-pass ext-720 (`ZENSIM_AB_MODE=ext`) on the same pairs:

  1. structural checks — fold f156..371 all zero, fold v2 block == ext v2
     block exactly;
  2. basic-block (f0..156) per-feature diffs, split by v1's padded-width
     class (`simd_padded_width(w) == w` → bit-exact expected; else the
     documented padded-width divergence);
  3. model-level Δ through the actual latest bakes:
       - ideal clean 720 (BVLS foldable, signed_pow p0.2, mono-reg) —
         3-way: ext raw / ext with pools zeroed / fold, separating the
         bake's pool-block sensitivity (expected 0) from the basic-block
         parity effect;
       - winner_m504 (504 = basic-156 ++ v2-348 MLP);
       - corruption_head_foldable384 (logistic + isotonic, per its own
         embedded deploy formula).

Forwarding goes through `predict_features_with_bake` (the canonical ZNPR
runtime CLI) — this script only packs its wire format and diffs outputs.

Usage:
  python3 scripts/v_next/fold_parity_check_2026-07-24.py \
      --ext ~/tmp/fold-v1/out_ext.csv --fold ~/tmp/fold-v1/out_fold.csv \
      --pairs ~/tmp/fold-v1/aic3_100.tsv \
      --out /mnt/v/output/zensim/fold-parity-2026-07-24
"""

import argparse
import json
import os
import struct
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREDICT_BIN = os.path.join(REPO, "target/release/predict_features_with_bake")

# The TRUE foldable-only clean model (measured pool sensitivity exactly 0)
# + its [0,100] dial-splined variant. NOTE: the mono-regularized RESOLUTION
# artifact `ideal_p0p2_L0p003_F0p005.bin` measured HEAVY pool dependence
# (zeroing f300..372 alone moves raw output ~0.40 — probed 2026-07-24 by
# this script's --probe-pools mode), contradicting its "foldable BVLS"
# doc row; it is NOT a valid fold consumer until refit/explained.
CLEAN_BAKE = "/mnt/v/output/zensim/signedpow-clean-2026-07-24/ideal_smoothpow_p0p2.bin"
CLEAN_DIAL = "/mnt/v/output/zensim/signedpow-clean-2026-07-24/ideal_smoothpow_p0p2_dial.bin"
WINNER_504 = "/mnt/v/output/zensim/bakes/top5/winner_m504.bin"
CORR_HEAD = "/mnt/v/output/zensim/corruption-head-2026-07-24/corruption_head_foldable384.json"


def simd_padded_width(w: int) -> int:
    """Mirror of zensim::blur::simd_padded_width."""
    aligned = (w + 15) & ~15
    if aligned >= 512 and (aligned // 16) % 2 == 0:
        return aligned + 16
    return aligned


def load_feature_csv(path):
    rows = []
    bases = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split(",")
            if parts[2] == "f0":  # header row
                continue
            bases.append(parts[0])
            rows.append(np.array([float(x) for x in parts[2:]], dtype=np.float64))
    return bases, np.vstack(rows)


def png_dims(path):
    with open(path, "rb") as f:
        head = f.read(26)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def forward_bake(bake, matrix):
    blob = struct.pack("<II", matrix.shape[1], matrix.shape[0])
    blob += matrix.astype(np.float32).tobytes()
    tmp = "/tmp_fold_blob.bin" if False else None
    import tempfile

    with tempfile.NamedTemporaryFile(
        dir=os.path.expanduser("~/tmp"), suffix=".blob", delete=False
    ) as tf:
        tf.write(blob)
        tmp = tf.name
    try:
        out = subprocess.run(
            [PREDICT_BIN, "--bake", bake, "--features-file", tmp],
            capture_output=True,
            text=True,
            check=True,
        )
        return np.array([float(x) for x in out.stdout.split()])
    finally:
        os.unlink(tmp)


def corruption_p(head, matrix720):
    idx = np.array(head["feat_idx"], dtype=np.int64)
    x = matrix720[:, idx]
    z = np.clip(
        (x - np.array(head["mean"])) / np.array(head["scale"]),
        -head["clip"],
        head["clip"],
    )
    s = 1.0 / (1.0 + np.exp(-(z @ np.array(head["coef"]) + head["intercept"])))
    cal = head["calibration"]
    return np.interp(s, np.array(cal["x"]), np.array(cal["y"]))


def describe(name, d):
    d = np.abs(d)
    print(
        f"  {name}: mean {d.mean():.3e}  p95 {np.percentile(d, 95):.3e}  max {d.max():.3e}"
    )
    return {"mean": float(d.mean()), "p95": float(np.percentile(d, 95)), "max": float(d.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bases_e, ext = load_feature_csv(args.ext)
    bases_f, fold = load_feature_csv(args.fold)
    assert bases_e == bases_f, "row order mismatch between ext and fold CSVs"
    assert ext.shape == fold.shape and ext.shape[1] == 720, ext.shape
    n = ext.shape[0]

    # Per-row ref dims from the pairs TSV (order preserved by the driver).
    ref_paths = []
    with open(args.pairs) as f:
        next(f)
        for line in f:
            ref_paths.append(line.split("\t")[0])
    dims = [png_dims(p) for p in ref_paths[:n]]
    exact_class = np.array([simd_padded_width(w) == w for (w, h) in dims])
    print(
        f"{n} rows; padded-width classes: exact={int(exact_class.sum())} "
        f"divergent={int((~exact_class).sum())}"
    )

    report = {"n": n, "exact_rows": int(exact_class.sum())}

    # 1. structural checks
    assert np.all(fold[:, 156:372] == 0.0), "fold pool block must be exactly zero"
    v2_equal = np.array_equal(ext[:, 372:], fold[:, 372:])
    print(f"v2 block identical: {v2_equal}")
    assert v2_equal, "v2 block must be identical between ext and fold"

    # 2. basic-block diffs by width class. Rel with a 1e-9 abs floor —
    # near-zero features (both sides ~1e-15) otherwise dominate the rel
    # stats with meaningless ratios.
    basic_e, basic_f = ext[:, :156], fold[:, :156]
    denom = np.maximum(np.abs(basic_e), 1e-9)
    rel = np.abs(basic_f - basic_e) / denom
    print("basic-block rel diffs (per-cell, 1e-9 abs floor):")
    if exact_class.any():
        report["basic_rel_exact"] = describe("exact-width rows  ", rel[exact_class])
    if (~exact_class).any():
        report["basic_rel_padded"] = describe("padded-width rows ", rel[~exact_class])

    # 3a. clean foldable 720 bake, 3-way (raw) + dial-splined delta
    ext_masked = ext.copy()
    ext_masked[:, 156:372] = 0.0
    s_ext = forward_bake(CLEAN_BAKE, ext)
    s_msk = forward_bake(CLEAN_BAKE, ext_masked)
    s_fold = forward_bake(CLEAN_BAKE, fold)
    print("clean foldable 720 bake (smoothpow_p0p2) scores (raw units):")
    report["clean_pool_sensitivity"] = describe("pool sensitivity (ext vs ext-masked)", s_ext - s_msk)
    report["clean_fold_delta"] = describe("fold effect (ext-masked vs fold)     ", s_msk - s_fold)
    report["clean_total_delta"] = describe("total (ext vs fold)                  ", s_ext - s_fold)
    if os.path.exists(CLEAN_DIAL):
        d_ext = forward_bake(CLEAN_DIAL, ext)
        d_fold = forward_bake(CLEAN_DIAL, fold)
        print("clean foldable bake, [0,100] dial-splined:")
        report["clean_dial_delta_pts"] = describe(
            "ext vs fold (dial points)            ", d_ext - d_fold
        )
    if (~exact_class).any() and exact_class.any():
        report["clean_delta_exact"] = describe(
            "total, exact-width rows            ", (s_ext - s_fold)[exact_class]
        )
        report["clean_delta_padded"] = describe(
            "total, padded-width rows           ", (s_ext - s_fold)[~exact_class]
        )

    # 3b. winner_m504
    pack504 = lambda m: np.hstack([m[:, :156], m[:, 372:]])
    w_ext = forward_bake(WINNER_504, pack504(ext))
    w_fold = forward_bake(WINNER_504, pack504(fold))
    print("winner_m504 MLP scores:")
    report["winner504_delta"] = describe("ext vs fold                          ", w_ext - w_fold)

    # 3c. corruption head (foldable-384)
    head = json.load(open(CORR_HEAD))
    p_ext = corruption_p(head, ext)
    p_fold = corruption_p(head, fold)
    t = head["recommended_deadband_T"]
    flips = int(((p_ext > t) != (p_fold > t)).sum())
    print("corruption head P(corruption):")
    report["corruption_delta"] = describe("ext vs fold                          ", p_ext - p_fold)
    report["corruption_gate_flips"] = flips
    print(f"  gate flips at T={t}: {flips}/{n}")

    with open(os.path.join(args.out, "fold_parity_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"report → {os.path.join(args.out, 'fold_parity_report.json')}")


if __name__ == "__main__":
    sys.exit(main())
