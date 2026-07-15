#!/usr/bin/env python3
"""Track down WHY adding bigcodec/imazen-26 costs CID22 (user: 'adding data should usually
help'). A clean diverse-content addition shouldn't reduce photographic rank ability — if it
does, there's a poisoning mechanism (a confound), not an intrinsic trade. Checks the three
prime suspects (DATASET_HISTORY.md §0 families #4 join-integrity + #1 ssim2-target):

  1. SCALER DOMINATION — the trainer's mu/sd is computed over the CONCATENATED set; bigcodec
     (120k) + KADIS (90k) dominate photographic (safesyn 90k + cid22 17k). If bigcodec's
     feature means differ from photographic, CID22 gets standardized by the wrong mu/sd.
     Report: per-feature |mean_bigcodec − mean_photo| / sd_photo  (shift in photo-sd units).
  2. LABEL-SCALE MISMATCH — bigcodec target=ssim2/100 (picker pipeline) vs canonical
     ssim2_gpu (zensim GPU). If they're different ssim2 impls/scales the regression target
     is inconsistent. Report: distribution of each corpus's target on the shared 0..100 scale.
  3. FEATURE-REGIME MISMATCH — is bigcodec the same 372 'with-iw' extraction? Report: NaN/
     constant-feature counts + gross range per block (basic f0..227 vs masked/IW f228..371).
"""
import numpy as np

import importlib.util
from pathlib import Path
REPO = Path.home() / "work/zen/zensim"
_spec = importlib.util.spec_from_file_location("T", REPO / "scripts/v_next/train_mlp_diverse.py")
T = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(T)

CORP = [("safesyn", f"{T.TRAIN}/safesyn.parquet", "ssim2_gpu", 1.0, 90000),
        ("cid22_tr", f"{T.TRAIN}/cid22_train.parquet", "ssim2_gpu", 1.0, None),
        ("bigcodec", T.BIGCODEC, "human_score", 100.0, 120000),
        ("kadis", T.KADIS, "score_ssim2_gpu", 1.0, 90000),
        ("CID22_val", T.CID22_VAL, "human_score", 1.0, None)]


def main():
    data = {}
    for name, path, ycol, scale, cap in CORP:
        X, y = T.load(path, ycol, scale, cap, seed=41)
        data[name] = (X, y)
        print(f"{name:10s} n={len(y):>7d}  target[{ycol}]: "
              f"min={y.min():8.2f} p5={np.percentile(y,5):7.2f} med={np.median(y):7.2f} "
              f"p95={np.percentile(y,95):7.2f} max={y.max():7.2f}  <0:{(y<0).mean()*100:5.1f}%")
    print()

    # ---- (1) SCALER DOMINATION: how far bigcodec/kadis pull mu vs the photographic ref ----
    Xsf = data["safesyn"][0]; mu_p, sd_p = Xsf.mean(0), Xsf.std(0); sd_p[sd_p == 0] = 1
    Xcv = data["CID22_val"][0]
    print("=== (1) feature-mean SHIFT vs safesyn, in safesyn-sd units (scaler-pull risk) ===")
    print(f"{'corpus':10s} {'mean|shift|':>11s} {'p95|shift|':>10s} {'max|shift|':>10s} "
          f"{'#feat>2sd':>9s}  (CID22_val's own shift for scale)")
    for name in ("cid22_tr", "bigcodec", "kadis", "CID22_val"):
        sh = np.abs(data[name][0].mean(0) - mu_p) / sd_p
        print(f"{name:10s} {sh.mean():>11.3f} {np.percentile(sh,95):>10.3f} "
              f"{sh.max():>10.3f} {int((sh>2).sum()):>9d}")
    # the combined-scaler question: with the trainer's row mix, whose distribution wins?
    rows = {n: len(data[n][0]) for n in ("safesyn", "cid22_tr", "bigcodec", "kadis")}
    tot = sum(rows.values())
    print(f"\n  train row mix: " + "  ".join(f"{n}={r}({r/tot*100:.0f}%)" for n, r in rows.items()))
    print(f"  -> photographic (safesyn+cid22)={rows['safesyn']+rows['cid22_tr']} "
          f"({(rows['safesyn']+rows['cid22_tr'])/tot*100:.0f}%) vs "
          f"bigcodec+kadis={rows['bigcodec']+rows['kadis']} "
          f"({(rows['bigcodec']+rows['kadis'])/tot*100:.0f}%)")
    # combined mu/sd (what the trainer actually uses) vs photo-only: how differently is CID22
    # standardized? report mean |z_combined - z_photo| over CID22_val features.
    Xall = np.concatenate([data[n][0] for n in ("safesyn", "cid22_tr", "bigcodec", "kadis")])
    mu_c, sd_c = Xall.mean(0), Xall.std(0); sd_c[sd_c == 0] = 1
    z_c = (Xcv - mu_c) / sd_c
    z_p = (Xcv - mu_p) / sd_p
    dz = np.abs(z_c - z_p)
    print(f"\n  CID22_val standardization drift |z_combined − z_photo|: "
          f"mean={dz.mean():.3f} p95={np.percentile(dz,95):.3f} max={dz.max():.3f} sd-units")
    print("  (large drift => the combined scaler mis-standardizes CID22 => a POISON channel)")

    # ---- (3) FEATURE-REGIME sanity: NaN/const per block, per corpus ----
    print("\n=== (3) feature-regime check (basic f0..227 vs masked/IW f228..371) ===")
    print(f"{'corpus':10s} {'#const(all)':>11s} {'#const basic':>12s} {'#const IW':>10s} "
          f"{'basic|max|':>10s} {'IW|max|':>8s}")
    for name in ("safesyn", "bigcodec", "cid22_tr", "CID22_val"):
        X = data[name][0]
        const = (X.std(0) == 0)
        print(f"{name:10s} {int(const.sum()):>11d} {int(const[:228].sum()):>12d} "
              f"{int(const[228:].sum()):>10d} {np.abs(X[:,:228]).max():>10.1f} "
              f"{np.abs(X[:,228:]).max():>8.1f}")


if __name__ == "__main__":
    main()
