#!/usr/bin/env python3
"""Explore natural content clusters in CID22 via k-means on zenanalyze features.

Foundation experiment for cycle-6's image-type-aware MLP dispatch:
- Do CID22 (and synth) images cluster into natural content types
  (photo / screen / line-art / etc.) when k-means is applied to
  zenanalyze's 228-dim feature vector?
- If yes, content-class dispatch is plausible.
- Report cluster sizes + which features distinguish clusters.

Uses scikit-learn if available, else falls back to simple variance-based
clustering.
"""
import csv, pathlib, sys
import numpy as np


def main():
    CLEAN = pathlib.Path("/tmp/zensim_loop/safe_synth_clean_features.csv")
    if not CLEAN.exists():
        print(f"ERROR: {CLEAN} not found", file=sys.stderr); sys.exit(1)

    # Sample up to 10k rows (load all features)
    N_SAMPLE = 10000
    feat_cols = [f"f{i}" for i in range(228)]
    X = []
    refs = []
    with open(CLEAN) as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if len(X) >= N_SAMPLE:
                break
            try:
                vec = [float(row[c]) for c in feat_cols]
                X.append(vec)
                refs.append(row.get("ref_basename", f"row{i}"))
            except (KeyError, ValueError):
                continue
    X = np.array(X, dtype=np.float32)
    print(f"Loaded {len(X)} feature vectors of dim {X.shape[1]}", file=sys.stderr)

    # Normalize: per-column z-score
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    Xn = (X - mean) / std

    try:
        from sklearn.cluster import MiniBatchKMeans
        for k in [3, 4, 6, 8]:
            km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256, n_init=3)
            labels = km.fit_predict(Xn)
            sizes = np.bincount(labels)
            sz_pct = [f"{s} ({s/len(Xn)*100:.1f}%)" for s in sizes]
            print(f"k={k}: cluster sizes = [{', '.join(sz_pct)}]", file=sys.stderr)
            print(f"      inertia = {km.inertia_:.1f}", file=sys.stderr)
    except ImportError:
        print("sklearn not available, falling back to simple variance summary", file=sys.stderr)
        # Show per-feature variance — high-variance features = candidates for separating
        var = X.var(axis=0)
        top10 = np.argsort(-var)[:10]
        print(f"Top 10 highest-variance features:", file=sys.stderr)
        for idx in top10:
            print(f"  f{idx}: var={var[idx]:.4f}, mean={X[:, idx].mean():.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
