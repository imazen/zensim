#!/usr/bin/env python3
"""Dense JXL fleet-sweep chunk builder (2026-05-30).

Per the dense-axis training rule: k-means-cluster the safesyn source universe
on zenanalyze content features, pick the K centroid-nearest representatives
(NOT random), then cross them with a dense butteraugli-distance ladder to form
the omni-worker chunks parquet.

The dense distance ladder (JXL's native axis) is far denser than the old
coarse q5..q100 (16 levels) — emphasis on the near-lossless region where the
dial underscored JXL, plus full low-q coverage per the q5-q60-density rule.

Output chunks schema (omni_backfill_chunk_worker reads this):
    image_path : str   (full source PNG path)
    codec      : str   ("zenjxl")
    q          : int64  (dummy 50 — JXL is driven by the distance knob)
    knob_tuple_json : str  ('{"distance": <d>}')
"""
from __future__ import annotations
import json, sys
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq

SRC_FEAT = "/mnt/v/output/zensim/jxl_dense_2026-05-30/source_features_native.tsv"
OUT      = "/mnt/v/output/zensim/jxl_dense_2026-05-30/jxl_dense_chunks.parquet"
PICKS    = "/mnt/v/output/zensim/jxl_dense_2026-05-30/representative_sources.tsv"
K        = 2000          # representative sources
SEED     = 17

# Dense butteraugli-distance ladder — denser than q5..q100, finest near lossless:
#   near-lossless 0.025 .. 0.5 step 0.025  (20)  ← the underscored region
#   high-fidelity 0.5   .. 1.5 step 0.1    (10)
#   mid           1.5   .. 3.0 step 0.25   (6)
#   low-q tail    3.5 .. 15 (covers q5-q60 equivalent)  (~8)
DISTANCES = sorted({
    round(d, 3) for d in (
        [0.025 * i for i in range(1, 21)]          # 0.025 .. 0.500
        + [0.5 + 0.1 * i for i in range(1, 11)]    # 0.6 .. 1.5
        + [1.5 + 0.25 * i for i in range(1, 7)]    # 1.75 .. 3.0
        + [3.5, 4.0, 5.0, 6.5, 8.0, 10.0, 12.5, 15.0]
    )
})


def main():
    # --- load source features ---
    import csv
    rows = list(csv.DictReader(open(SRC_FEAT), delimiter="\t"))
    if not rows:
        sys.exit("no source-feature rows")
    feat_cols = [c for c in rows[0] if c.startswith("feat") or (c[:1] == "f" and c[1:].isdigit())]
    if not feat_cols:
        # fall back: numeric columns excluding known metadata
        meta = {"image_path", "image_sha", "split", "content_class", "source",
                "size_class", "width", "height"}
        feat_cols = [c for c in rows[0] if c not in meta]
    print(f"{len(rows)} sources, {len(feat_cols)} feature cols", file=sys.stderr)

    X, paths, ids = [], [], []
    for r in rows:
        try:
            v = [float(r[c]) for c in feat_cols]
        except (ValueError, KeyError):
            continue
        if all(np.isfinite(v)):
            X.append(v); paths.append(r.get("image_path", "")); ids.append(r.get("source", ""))
    X = np.array(X, dtype=float)
    print(f"{len(X)} clean rows for clustering", file=sys.stderr)

    # --- standardize + k-means, pick centroid-nearest member per cluster ---
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    k = min(K, len(Xs))
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, random_state=SEED, n_init=4).fit(Xs)
        labels, centers = km.labels_, km.cluster_centers_
    except ImportError:
        # numpy-only Lloyd's fallback (deterministic init: every Nth point)
        rng = np.random.default_rng(SEED)
        idx0 = np.linspace(0, len(Xs) - 1, k).astype(int)
        centers = Xs[idx0].copy()
        for _ in range(25):
            d = ((Xs[:, None, :] - centers[None, :, :]) ** 2).sum(2)
            labels = d.argmin(1)
            for c in range(k):
                m = labels == c
                if m.any(): centers[c] = Xs[m].mean(0)
    # centroid-nearest member of each cluster
    picks = []
    for c in range(k):
        m = np.where(labels == c)[0]
        if len(m) == 0: continue
        d = ((Xs[m] - centers[c]) ** 2).sum(1)
        picks.append(m[d.argmin()])
    picks = sorted(set(picks))
    print(f"picked {len(picks)} representative sources", file=sys.stderr)

    with open(PICKS, "w") as f:
        f.write("source\timage_path\n")
        for i in picks:
            f.write(f"{ids[i]}\t{paths[i]}\n")

    # --- cross with the dense distance ladder → chunks ---
    img, cod, q, knob = [], [], [], []
    for i in picks:
        for d in DISTANCES:
            img.append(paths[i]); cod.append("zenjxl"); q.append(50)
            knob.append(json.dumps({"distance": d}))
    tbl = pa.table({
        "image_path": pa.array(img),
        "codec": pa.array(cod),
        "q": pa.array(q, type=pa.int64()),
        "knob_tuple_json": pa.array(knob),
    })
    pq.write_table(tbl, OUT, compression="zstd")
    print(f"wrote {OUT}: {len(img)} cells "
          f"({len(picks)} sources × {len(DISTANCES)} distances)", file=sys.stderr)
    print(f"distance ladder ({len(DISTANCES)}): {DISTANCES}", file=sys.stderr)


if __name__ == "__main__":
    main()
