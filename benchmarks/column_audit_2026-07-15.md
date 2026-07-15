# Per-column poison audit — 2026-07-15

User: *'dig into every column and how to make it helpful rather than poisoning'.* De-poisoned 372-64-1 leaky MLP (winsor p0.1/p99.9), safesyn+cid22_train, CID22 a PURE holdout, 2 seeds. Verdicts: HELPFUL / NEUTRAL / WEAK / POISON / SPARSE.

## A. Coverage + distribution + ssim2-agreement (per training corpus)

`tail` = (p99.5−p95)/(p50−p5); ≫3 ⇒ log-expanded top (MSE over-weights near-lossless sliver — the cvvdp_log_norm confound). `agree` = |SROCC vs ssim2_gpu|.

### `human_score`

| corpus | cov% | n | min | median | max | tail | agree(ssim2) |
|---|--:|--:|--:|--:|--:|--:|--:|
| safesyn | 100.0 | 196086 | -0.577 | 0.687 | 0.965 | 0.0433 | 1 |
| cid22_train | 100.0 | 17611 | 35.5 | 72.5 | 91.2 | 0.182 | 1 |
| kadid | 100.0 | 10125 | 0.0325 | 0.5 | 0.947 | 0.112 | 0.00981 |
| tid | 100.0 | 3000 | 0.0808 | 0.511 | 0.746 | 0.271 | 0.00729 |
| konjnd-dense | 100.0 | 20160 | -64.2 | 67.7 | 94.6 | 0.0178 | — |
| bigcodec | 100.0 | 2322579 | 0 | 0.632 | 0.961 | 0.0681 | — |

### `ssim2_gpu`

| corpus | cov% | n | min | median | max | tail | agree(ssim2) |
|---|--:|--:|--:|--:|--:|--:|--:|
| safesyn | 100.0 | 196086 | -57.7 | 68.7 | 96.5 | 0.0433 | 1 |
| cid22_train | 100.0 | 17611 | 35.5 | 72.5 | 91.2 | 0.182 | 1 |
| kadid | 100.0 | 10125 | 93.5 | 100 | 100 | 0.49 | 1 |
| tid | 100.0 | 3000 | 63.7 | 70.8 | 77.7 | 0.126 | 1 |
| konjnd-dense | 0.0 | 0 | — | — | — | — | — |

### `ssim2_log_norm`

| corpus | cov% | n | min | median | max | tail | agree(ssim2) |
|---|--:|--:|--:|--:|--:|--:|--:|
| safesyn | 100.0 | 196086 | 0 | 75.9 | 97.3 | 0.0433 | 1 |
| cid22_train | 100.0 | 17611 | 50.4 | 78.8 | 93.2 | 0.182 | 1 |
| kadid | 100.0 | 10125 | 95 | 100 | 100 | 0.49 | 1 |
| tid | 100.0 | 3000 | 72 | 77.5 | 82.8 | 0.126 | 1 |
| konjnd-dense | 0.0 | 0 | — | — | — | — | — |

### `cvvdp_score`

| corpus | cov% | n | min | median | max | tail | agree(ssim2) |
|---|--:|--:|--:|--:|--:|--:|--:|
| safesyn | 100.0 | 196086 | 7.06 | 9.8 | 10 | 0.000435 | 0.984 |
| cid22_train | 100.0 | 17611 | 9.28 | 9.86 | 10 | 0.0585 | 0.941 |
| kadid | 100.0 | 10125 | 4.13 | 8.75 | 10 | 0.018 | 0.003 |
| tid | 100.0 | 3000 | 5.22 | 9.07 | 9.97 | 0.0164 | 0.115 |
| konjnd-dense | 0.0 | 0 | — | — | — | — | — |

### `cvvdp_log_norm`

| corpus | cov% | n | min | median | max | tail | agree(ssim2) |
|---|--:|--:|--:|--:|--:|--:|--:|
| safesyn | 100.0 | 196086 | 6.53 | 23.5 | 100 | 3.09 | 0.984 |
| cid22_train | 100.0 | 17611 | 71.5 | 75.2 | 76 | 0.0585 | 0.941 |
| kadid | 100.0 | 10125 | 3.7 | 13.2 | 100 | 8.67 | 0.003 |
| tid | 100.0 | 3000 | 3.23 | 31.1 | 91.5 | 0.713 | 0.115 |
| konjnd-dense | 0.0 | 0 | — | — | — | — | — |

### `iwssim`

| corpus | cov% | n | min | median | max | tail | agree(ssim2) |
|---|--:|--:|--:|--:|--:|--:|--:|
| safesyn | 100.0 | 196086 | 0.738 | 0.988 | 1 | 0.00173 | 0.963 |
| cid22_train | 100.0 | 17611 | 0.955 | 0.993 | 1 | 0.036 | 0.864 |
| kadid | 100.0 | 10125 | 0.0325 | 0.5 | 0.947 | 0.112 | 0.00981 |
| tid | 100.0 | 3000 | 0.0808 | 0.511 | 0.746 | 0.271 | 0.00729 |
| konjnd-dense | 0.0 | 0 | — | — | — | — | — |

### `iwssim_log_norm`

| corpus | cov% | n | min | median | max | tail | agree(ssim2) |
|---|--:|--:|--:|--:|--:|--:|--:|
| safesyn | 100.0 | 196086 | 9.76 | 32.1 | 81.5 | 1.17 | 0.963 |
| cid22_train | 100.0 | 17611 | 22.6 | 36.6 | 60 | 0.935 | 0.864 |
| kadid | 100.0 | 10125 | 0.241 | 5.05 | 21.4 | 1.04 | 0.00981 |
| tid | 100.0 | 3000 | 0.614 | 5.22 | 10 | 0.574 | 0.00729 |
| konjnd-dense | 0.0 | 0 | — | — | — | — | — |

### `pjnd_target`

| corpus | cov% | n | min | median | max | tail | agree(ssim2) |
|---|--:|--:|--:|--:|--:|--:|--:|
| safesyn | 0.0 | 0 | — | — | — | — | — |
| cid22_train | 0.0 | 0 | — | — | — | — | — |
| kadid | 0.0 | 0 | — | — | — | — | — |
| tid | 0.0 | 0 | — | — | — | — | — |
| konjnd-dense | 100.0 | 20160 | 26.1 | 32.2 | 66.9 | 0.925 | — |

### `active_mix_raw`

| corpus | cov% | n | min | median | max | tail | agree(ssim2) |
|---|--:|--:|--:|--:|--:|--:|--:|
| konjnd-dense | 100.0 | 20160 | -64.2 | 67.7 | 94.6 | 0.0178 | — |

### `score_ssim2_gpu`

| corpus | cov% | n | min | median | max | tail | agree(ssim2) |
|---|--:|--:|--:|--:|--:|--:|--:|
| kadis | 100.0 | 266111 | -175 | -51.8 | 89.2 | 2.52 | — |

### `mix_cv50_iw50`

| corpus | cov% | n | min | median | max | tail | agree(ssim2) |
|---|--:|--:|--:|--:|--:|--:|--:|
| safesyn | 100.0 | 196086 | 8.28 | 27.9 | 90 | 2.06 | 0.983 |
| cid22_train | 0.0 | 0 | — | — | — | — | — |
| kadid | 100.0 | 10125 | 2.34 | 9.27 | 59.8 | 6.05 | 0.00664 |
| tid | 100.0 | 3000 | 2.05 | 18.1 | 49.5 | 0.621 | 0.106 |
| konjnd-dense | 100.0 | 20160 | -64.2 | 67.7 | 94.6 | 0.0178 | — |

### `mix_cv25_iw75`

| corpus | cov% | n | min | median | max | tail | agree(ssim2) |
|---|--:|--:|--:|--:|--:|--:|--:|
| safesyn | 100.0 | 196086 | 9.04 | 30 | 85.4 | 1.57 | 0.974 |
| cid22_train | 0.0 | 0 | — | — | — | — | — |
| kadid | 100.0 | 10125 | 1.43 | 7.18 | 39.7 | 3.76 | 0.0085 |
| tid | 100.0 | 3000 | 1.41 | 11.8 | 28.6 | 0.484 | 0.091 |
| konjnd-dense | 100.0 | 20160 | -64.2 | 67.7 | 94.6 | 0.0178 | — |

### `mix_cv75_iw25`

| corpus | cov% | n | min | median | max | tail | agree(ssim2) |
|---|--:|--:|--:|--:|--:|--:|--:|
| safesyn | 100.0 | 196086 | 7.45 | 25.7 | 95 | 2.56 | 0.987 |
| cid22_train | 0.0 | 0 | — | — | — | — | — |
| kadid | 100.0 | 10125 | 3.1 | 11.3 | 79.9 | 7.63 | 0.00471 |
| tid | 100.0 | 3000 | 2.53 | 24.6 | 70.5 | 0.675 | 0.111 |
| konjnd-dense | 100.0 | 20160 | -64.2 | 67.7 | 94.6 | 0.0178 | — |

### `mix_cv33_iw33_sm33`

| corpus | cov% | n | min | median | max | tail | agree(ssim2) |
|---|--:|--:|--:|--:|--:|--:|--:|
| safesyn | 0.0 | 0 | — | — | — | — | — |
| cid22_train | 0.0 | 0 | — | — | — | — | — |
| kadid | 100.0 | 10125 | 34.8 | 39.5 | 73.2 | 6.04 | 0.0171 |
| tid | 100.0 | 3000 | 26.6 | 37.8 | 60 | 0.725 | 0.227 |
| konjnd-dense | 0.0 | 0 | — | — | — | — | — |

## B. Train-toward-target → CID22 holdout SROCC (is it a usable TARGET?)

| target | CID22(holdout) | train-val | verdict |
|---|--:|--:|---|
| `ssim2_gpu` | 0.8826 | 0.9979 | NEUTRAL (usable, no edge vs ssim2) |
| `ssim2_log_norm` | 0.8801 | 0.9979 | NEUTRAL (usable, no edge vs ssim2) |
| `cvvdp_score` | 0.8486 | 0.9913 | WEAK (usable, below ssim2) |
| `cvvdp_log_norm` | 0.5972 | 0.9625 | POISON (target-shape: rank-fine but MSE-craters) |
| `iwssim` | 0.8333 | 0.9875 | WEAK (usable, below ssim2) |
| `iwssim_log_norm` | 0.7923 | 0.9852 | WEAK (usable, below ssim2) |
| `mix_cv50_iw50` | 0.7048 | 0.9796 | POISON (target-shape: rank-fine but MSE-craters) |

_ssim2_gpu baseline CID22 = 0.8826. A target that rank-agrees with ssim2 (agree≈1) yet craters CID22-as-target is a TARGET-SHAPE poison — fix with raw/rank loss, never MSE on the log-expanded form (§8.36)._
