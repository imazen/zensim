# Baseline-metric SROCC vs human MOS

Max 100000 pairs per dataset.

| Dataset | n | V0_2 | V0_4 (bake) | fast-ssim2 | butteraugli |
|---------|--:|:----:|:-----------:|:----------:|:-----------:|
| KADIK10k | 10125 | 0.8192 | 0.9397 | 0.8133 | 0.6062 |
| TID2013 | 3000 | 0.8427 | 0.9446 | 0.8460 | 0.6696 |
| CID22 | 4292 | 0.8676 | 0.8927 | 0.8895 | 0.7412 |

## KonJND-1k visually-lossless calibration (Lin, Hosu, Saupe 2022)

Pairs at the per-source mean PJND (Probabilistic Just-Noticeable-Difference)
threshold. Each source's pair is the just-barely-perceptible distortion. The
Cloudinary CID22 paper Table 4 publishes these mean ± stdev anchors for
several metrics; comparing our SSIMULACRA 2 / Butteraugli numbers below with
that table cross-validates the pipeline.

### JPEG subset (n = 504)

| metric | mean | stdev | Cloudinary Table 4 (paper) |
|---|--:|--:|---|
| V0_2 raw distance | 2.1120 | 0.4455 | — |
| V0_4 raw distance | -6.0101 | 1.2995 | — |
| fast-ssim2 score | 62.55 | 5.03 | 63.10 ± 4.65 |
| butteraugli 3-norm | 1.6993 | 0.2274 | 1.699 ± 0.229 |

### BPG subset (n = 504)

| metric | mean | stdev | Cloudinary Table 4 (paper) |
|---|--:|--:|---|
| V0_2 raw distance | 1.8835 | 0.4378 | — |
| V0_4 raw distance | -6.1630 | 1.5266 | — |
| fast-ssim2 score | 65.38 | 5.42 | 65.38 ± 5.10 |
| butteraugli 3-norm | 1.5283 | 0.1912 | 1.528 ± 0.192 |
