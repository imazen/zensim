# Feature-health + leakage sweep — 2026-07-16

User: *'sweep all input parquets for bad data that might be dragging things down'.* Scans f0..f371 in every parquet a depth/psa recipe loads. `nan_rows`=rows with any non-finite feature; `exploded`=features with |max|>1e4 (unbounded IW HF-moment — the 5.8M-on-graphics bug); `const`=zero-variance features; `leak`=human_score byte-identical to a feature/metric column (the kadid/tid iwssim==human_score bug).

| parquet | role | n | nan_rows | n_nan_feat | n_const | worst_feat |max| | exploded | leak |
|---|---|--:|--:|--:|--:|---|--:|---|---|
| safesyn | train | 196086 | 0 | 0 | 0 | f129 | 3.31e+04 | **4: [12, 51, 90, 129]** | clean |
| cid22_train | train | 17611 | 0 | 0 | 0 | f155 | 25.9 | — | **ssim2_gpu** |
| kadid | train | 10125 | 0 | 0 | 0 | f38 | 623 | — | clean |
| tid | train | 3000 | 0 | 0 | 0 | f38 | 882 | — | clean |
| konjnd-dense-norm | train | 20160 | 0 | 0 | 0 | f129 | 27.9 | — | clean |
| multiband_anchor | anchor | 2000 | 0 | 0 | 0 | f12 | 41.3 | — | clean |
| hf_nearlossless | train | 900 | 0 | 0 | 0 | f129 | 0.0837 | — | clean |
| bigcodec_train | train | 120000 | 0 | 0 | 0 | f90 | 1.03e+04 | **1: [90]** | clean |
| kadis_train | train | 60000 | 0 | 0 | 0 | f38 | 3.57e+06 | **8: [12, 38, 51, 77, 90, 116, 129, 155]** | clean |
| bigcodec_val | val | 114871 | 0 | 0 | 0 | f331 | 1.84e+04 | **2: [259, 331]** | clean |
| val/cid22 | holdout | 4292 | 0 | 0 | 0 | f155 | 3.76 | — | clean |
| val/kadid | holdout | 10125 | 0 | 0 | 0 | f38 | 623 | — | clean |
| val/tid | holdout | 3000 | 0 | 0 | 0 | f38 | 882 | — | clean |
| val/konjnd | holdout | 1008 | 0 | 0 | 1 | f77 | 6.18 | — | clean |
| val/aic3 | holdout | 600 | 0 | 0 | 1 | f156 | 1.09 | — | clean |
| val/aic4 | holdout | 300 | 0 | 0 | 0 | f156 | 0.97 | — | clean |
| konfig_triplet_stim | triplet | 1220 | 0 | 0 | 0 | f38 | 98.7 | — | clean |
