#!/usr/bin/env python3
"""Cross-eval: §8.33 photographic MLP vs the imazen-26 DIVERSE MLP on the SAME held-out
slices. The two clean A/B axes (neither model trained on them, or one never saw them):
  - CID22 (human MOS holdout)  — neither model trained on it.
  - bigcodec-val (imazen-26 diverse, ssim2 target) — §8.33 NEVER saw any bigcodec, so it
    is fully held-out for §8.33; the diverse model held out 25%. Fair for both.
The point: does imazen-26 diversity buy better ranking on NON-photographic content, at the
(measured) small CID22 cost? Reuses train_mlp_diverse.build() so the slices are identical.
"""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path.home() / "work/zen/zensim"
_spec = importlib.util.spec_from_file_location("T", REPO / "scripts/v_next/train_mlp_diverse.py")
T = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(T)

NPZ = {"§8.33 photo": "/mnt/v/output/zensim/reports/b_negatives/mlp_neg_best.npz",
       "diverse":     "/mnt/v/output/zensim/reports/b_negatives/mlp_diverse_best.npz"}


def fwd(P, X):
    z = (X - P["mu"]) / P["sd"]
    h = z @ P["W0"].T + P["b0"]
    h = np.where(h > 0, h, float(P["leaky"]) * h)
    return (h @ P["W1"].T + P["b1"]).ravel()


def main():
    # identical held-out slices from the diverse trainer's seed-41 build
    (_, _, _), val = T.build(41, div_cap=120000, hq_band=85.0, hq_weight=0.3,
                             div_weight=1.0, kadis_weight=0.3)
    Xcv, ycv = T.load(T.CID22_VAL, "human_score")            # MOS holdout (neither trained)
    Xbg, ybg = val["bigcodec"]                                # imazen-26 diverse held-out
    Xkd, ykd = val["kadis"]; dk = ykd < -64                   # deep negatives
    models = {k: {kk: np.load(v)[kk] for kk in np.load(v).files} for k, v in NPZ.items()}

    def srocc(P, X, y):
        return spearmanr(fwd(P, X), y).correlation

    print(f"{'axis':<28s} " + " ".join(f"{k:>13s}" for k in NPZ) + "   winner")
    axes = [("CID22 (photo MOS holdout)", Xcv, ycv, "MOS"),
            ("bigcodec-val (imazen-26)",  Xbg, ybg, "ssim2"),
            ("KADIS deep-neg (<-64)",     Xkd[dk], ykd[dk], "ssim2")]
    for name, X, y, _ in axes:
        vals = {k: srocc(P, X, y) for k, P in models.items()}
        win = max(vals, key=vals.get)
        print(f"{name:<28s} " + " ".join(f"{vals[k]:>+13.4f}" for k in NPZ)
              + f"   {win} (+{abs(vals['diverse']-vals['§8.33 photo']):.4f})")
    print("\nbigcodec-val = held-out imazen-26 REAL-codec content (screen/UI/doc/line-art/AI-gen);")
    print("§8.33 never saw ANY bigcodec -> fully held-out for it. CID22 is photographic only.")


if __name__ == "__main__":
    main()
