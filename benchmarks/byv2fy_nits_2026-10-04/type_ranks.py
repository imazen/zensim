#!/usr/bin/env python3
"""by_v2fy's NITS deficit, by distortion type (descriptive; open external set, cached predictions, no new labels).

For every (fold, seed) cell of by_v2fy and v2 + basic at cv16:cf98, rank the 405 NITS predictions and the human scores; for each
distortion type report the mean (rank(pred) - rank(human)) / n over that type's pairs (positive = the model ranks that type better
than people do), paired by (fold, seed) between the two sets.
  python3 type_ranks.py --root /var/tmp/rev4-featpot/v2c
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
sys.path.insert(0, str(Path.home() / "work/zen/zensim--featbank-potential/scripts/rev4_featpot"))
from v2_common import SOURCE_ORDER  # noqa: E402

ARMS = {"by_v2fy": "sel:59f0bbc2f290@h32:H128:cv16:cf98", "v2basic": "set:v2+basic@h32:H128:cv16:cf98"}
TYPES = {"D1": "gaussian blur", "D2": "chromatic gaussian noise", "D3": "chromatic uniform noise", "D4": "contrast change",
         "D5": "pixelate mosaic", "D6": "motion blur", "D7": "jpeg", "D8": "jpeg2000", "D9": "jpeg-xt"}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); args = ap.parse_args()
    root = Path(args.root); keys = pd.read_parquet(root / "external" / "nits.keys.parquet")
    y = keys.human_score.to_numpy(float); n = len(y); ry = rankdata(y); types = keys.distortion.to_numpy()
    cache = root / "external" / "pred"
    out = {a: [] for a in ARMS}
    for a, spec in ARMS.items():
        for fold in SOURCE_ORDER:
            for i in range(20):
                cell = root / "cells" / f"{spec}__N" / f"without_{fold}_s{i}"
                if not (cell / "result.json").is_file():
                    continue
                bake = Path(json.loads((cell / "result.json").read_text())["selected_bake"])
                p = cache / f"{hashlib.sha256(bake.read_bytes()).hexdigest()[:16]}_nits.tsv"
                if not p.is_file():
                    continue
                pred = pd.read_csv(p, sep="\t").pred.to_numpy(float); rp = rankdata(pred)
                row = {"fold": fold, "seed": i, "srocc": spearmanr(pred, y)[0]}
                for t in TYPES:
                    m = types == t; row[t] = float(np.mean(rp[m] - ry[m]) / n)
                    row[t + "_within"] = spearmanr(pred[m], y[m])[0]
                out[a].append(row)
    A = pd.DataFrame(out["by_v2fy"]).set_index(["fold", "seed"]); B = pd.DataFrame(out["v2basic"]).set_index(["fold", "seed"])
    J = A.join(B, lsuffix="_f", rsuffix="_b", how="inner")
    print(f"paired cells: {len(J)}  overall SROCC by_v2fy {J.srocc_f.mean():.4f}  v2basic {J.srocc_b.mean():.4f}  Δ {np.mean(J.srocc_f - J.srocc_b):+.4f}")
    print("type                       | rank shift by_v2fy  v2basic   Δ (±SE)        | within-type SROCC by_v2fy  v2basic")
    for t, name in TYPES.items():
        d = J[t + "_f"] - J[t + "_b"]
        print(f"{t} {name:24s} | {J[t+'_f'].mean():+.4f}  {J[t+'_b'].mean():+.4f}  {d.mean():+.4f} ±{d.std(ddof=1)/np.sqrt(len(d)):.4f} | {J[t+'_within_f'].mean():.3f}  {J[t+'_within_b'].mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
