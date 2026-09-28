# Rev4 E1b — error tables: pairs each metric orders wrong where the best cross peer orders right

Companion to `benchmarks/rev4_e1b_crosscodec_2026-09-23.md`. Raw material for cross-codec supervision design.

Cells are `wrong-where-peer-right / right-where-peer-wrong` (metric ties in parentheses when non-zero), per codec pair. Same-codec rows are `X (same)`. Forced choice: per-question strict majority, split questions excluded. All zensim models and peers per cell are in the JSON.

**CID22-A(25)** — peer ssim2

| class | codec pair | n | B | C | D | R915_fast | R915_rich | V0_2 |
|---|---|---|---|---|---|---|---|---|
| same | AVIF (same) | 4788 | 26/2 | 2/3 | 0/3 | 6/3 | 6/3 | 5/3 |
| same | HEIC (same) | 700 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| same | JP2 (same) | 900 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| same | JPEG (same) | 1355 | 0/0 | 0/0 | 0/0 | 1/0 | 0/0 | 0/0 |
| same | JXL (same) | 1345 | 0/0 | 0/0 | 0/0 | 7/0 | 0/0 | 0/0 |
| same | WebP (same) | 900 | 0/0 | 0/0 | 0/0 | 0/0 | 1/0 | 0/0 |
| mid | AVIF vs AVIF | 14756 | 677/485 | 156/202 | 261/185 | 241/249 | 138/198 | 217/142 |
| cross | AVIF vs HEIC | 7976 | 254/182 | 122/93 | 127/75 | 104/114 | 110/99 | 116/69 |
| cross | AVIF vs JP2 | 8973 | 417/399 | 200/139 | 317/166 | 209/195 | 165/152 | 155/189 |
| cross | AVIF vs JPEG | 10892 | 268/356 | 288/78 | 231/145 | 162/198 | 155/83 | 270/71 |
| cross | AVIF vs JXL | 10841 | 275/341 | 198/105 | 404/64 | 141/262 | 95/164 | 261/77 |
| cross | AVIF vs WebP | 8973 | 299/163 | 176/99 | 184/86 | 156/128 | 107/98 | 161/91 |
| cross | HEIC vs JP2 | 1800 | 52/54 | 23/26 | 48/42 | 34/33 | 27/28 | 18/29 |
| cross | HEIC vs JPEG | 2184 | 32/58 | 50/13 | 38/20 | 29/35 | 30/13 | 70/8 |
| cross | HEIC vs JXL | 2176 | 21/32 | 23/23 | 89/5 | 39/32 | 21/23 | 61/6 |
| cross | HEIC vs WebP | 1800 | 26/36 | 17/28 | 13/35 | 27/35 | 20/34 | 20/35 |
| cross | JP2 vs JPEG | 2457 | 60/34 | 44/29 | 21/52 | 37/46 | 37/22 | 48/26 |
| cross | JP2 vs JXL | 2448 | 73/32 | 45/38 | 40/31 | 32/47 | 45/31 | 55/35 |
| cross | JP2 vs WebP | 2025 | 61/65 | 40/24 | 58/34 | 51/29 | 33/34 | 18/34 |
| cross | JPEG vs JXL | 2970 | 55/41 | 56/24 | 60/59 | 68/40 | 48/22 | 47/24 |
| cross | JPEG vs WebP | 2457 | 33/63 | 72/11 | 44/25 | 32/28 | 28/20 | 71/11 |
| cross | JXL vs WebP | 2448 | 33/47 | 43/37 | 92/16 | 37/49 | 21/37 | 73/20 |

**CSIQ JPEG+JPEG2000** — peer ssim2

| class | codec pair | n | B | C | D |
|---|---|---|---|---|---|
| same | JPEG (same) | 300 | 17/0 (1) | 0/0 (1) | 0/0 (1) |
| same | jpeg2000 (same) | 300 | 20/0 | 0/0 | 0/0 |
| cross | JPEG vs jpeg2000 | 749 | 57/11 | 11/10 | 5/5 |

**AIC-3 CTC (decoded PNG, source res.)** — peer iwssim

| class | codec pair | n | B | C | D | R915_fast | R915_rich | V0_2 |
|---|---|---|---|---|---|---|---|---|
| same | AVIF (same) | 450 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| same | HM (same) | 450 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| same | JPEG-1 (same) | 450 | 3/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| same | JPEG-2000 (same) | 450 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| same | JPEGXL (same) | 450 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 |
| same | VVC (same) | 450 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| cross | AVIF vs HM | 900 | 14/7 | 7/4 | 3/8 | 9/4 | 8/3 | 8/4 |
| cross | AVIF vs JPEG-1 | 900 | 93/28 | 15/24 | 21/14 | 20/15 | 18/10 | 32/19 |
| cross | AVIF vs JPEG-2000 | 900 | 15/7 | 11/7 | 6/6 | 3/2 | 3/4 | 2/3 |
| cross | AVIF vs JPEGXL | 900 | 60/23 | 14/20 | 11/17 | 4/16 | 11/16 | 11/24 |
| cross | AVIF vs VVC | 900 | 29/6 | 8/6 | 3/6 | 5/4 | 19/4 | 2/5 |
| cross | HM vs JPEG-1 | 900 | 85/25 | 11/26 | 17/14 | 17/15 | 16/11 | 28/20 |
| cross | HM vs JPEG-2000 | 900 | 9/8 | 4/12 | 4/7 | 10/8 | 6/11 | 7/9 |
| cross | HM vs JPEGXL | 900 | 68/16 | 20/21 | 16/8 | 12/8 | 18/7 | 23/14 |
| cross | HM vs VVC | 900 | 24/2 | 11/7 | 9/3 | 5/5 | 20/6 | 12/4 |
| cross | JPEG-1 vs JPEG-2000 | 900 | 109/24 | 22/24 | 20/14 | 17/19 | 15/11 | 28/17 |
| cross | JPEG-1 vs JPEGXL | 900 | 10/12 | 17/13 | 12/9 | 14/11 | 14/10 | 14/12 |
| cross | JPEG-1 vs VVC | 900 | 94/29 | 29/19 | 31/16 | 29/10 | 24/14 | 45/20 |
| cross | JPEG-2000 vs JPEGXL | 900 | 89/19 | 33/19 | 18/13 | 10/12 | 21/14 | 22/18 |
| cross | JPEG-2000 vs VVC | 900 | 27/7 | 7/3 | 7/9 | 9/8 | 25/8 | 14/8 |
| cross | JPEGXL vs VVC | 900 | 75/31 | 24/23 | 19/15 | 17/17 | 30/17 | 26/24 |

**AIC-4 crop (620×800, as shown)** — peer cvvdp_fhd

| class | codec pair | n | B | C | D | R915_fast | R915_rich | V0_2 |
|---|---|---|---|---|---|---|---|---|
| same | AVIF (same) | 225 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| same | JPEG-1 (same) | 225 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| same | JPEG-2000 (same) | 225 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| same | JPEG-AI (same) | 225 | 1/0 | 0/0 | 0/0 | 0/0 | 1/0 | 0/0 |
| same | JPEG-XL (same) | 225 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| same | VVC (same) | 225 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| cross | AVIF vs JPEG-1 | 500 | 84/5 | 29/9 | 35/10 | 35/11 | 28/6 | 46/10 |
| cross | AVIF vs JPEG-2000 | 500 | 14/17 | 27/15 | 15/15 | 27/17 | 20/17 | 15/14 |
| cross | AVIF vs JPEG-AI | 500 | 28/40 | 4/23 | 10/33 | 8/26 | 15/26 | 10/30 |
| cross | AVIF vs JPEG-XL | 500 | 61/14 | 27/16 | 38/20 | 27/18 | 32/14 | 36/15 |
| cross | AVIF vs VVC | 500 | 5/17 | 14/7 | 2/13 | 10/10 | 6/7 | 10/8 |
| cross | JPEG-1 vs JPEG-2000 | 500 | 119/4 | 76/6 | 66/6 | 81/4 | 67/5 | 84/6 |
| cross | JPEG-1 vs JPEG-AI | 500 | 8/8 | 2/8 | 2/16 | 9/3 | 5/14 | 3/14 |
| cross | JPEG-1 vs JPEG-XL | 500 | 10/7 | 7/5 | 11/6 | 10/6 | 7/10 | 6/4 |
| cross | JPEG-1 vs VVC | 500 | 89/3 | 48/5 | 38/5 | 53/3 | 41/4 | 60/5 |
| cross | JPEG-2000 vs JPEG-AI | 500 | 39/45 | 8/36 | 13/40 | 17/40 | 23/40 | 16/38 |
| cross | JPEG-2000 vs JPEG-XL | 500 | 89/8 | 57/10 | 54/10 | 55/8 | 58/8 | 59/9 |
| cross | JPEG-2000 vs VVC | 500 | 7/21 | 7/22 | 4/26 | 2/23 | 4/23 | 1/17 |
| cross | JPEG-AI vs JPEG-XL | 500 | 2/21 | 2/10 | 0/22 | 2/18 | 1/21 | 1/22 |
| cross | JPEG-AI vs VVC | 500 | 32/27 | 10/24 | 15/23 | 20/25 | 23/27 | 20/24 |
| cross | JPEG-XL vs VVC | 500 | 61/9 | 39/11 | 32/14 | 38/11 | 41/10 | 47/12 |

**AIC-4 full res. (secondary)** — peer ssim2

| class | codec pair | n | B | C | D | R915_fast | R915_rich | V0_2 |
|---|---|---|---|---|---|---|---|---|
| same | AVIF (same) | 225 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| same | JPEG-1 (same) | 225 | 2/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| same | JPEG-2000 (same) | 225 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| same | JPEG-AI (same) | 225 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| same | JPEG-XL (same) | 225 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| same | VVC (same) | 225 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| cross | AVIF vs JPEG-1 | 500 | 45/2 | 2/21 | 0/24 | 0/18 | 0/24 | 3/8 |
| cross | AVIF vs JPEG-2000 | 500 | 5/22 | 8/11 | 4/9 | 2/2 | 4/4 | 3/8 |
| cross | AVIF vs JPEG-AI | 500 | 33/35 | 11/6 | 4/16 | 5/7 | 11/18 | 1/12 |
| cross | AVIF vs JPEG-XL | 500 | 46/4 | 0/12 | 9/9 | 2/13 | 8/9 | 10/8 |
| cross | AVIF vs VVC | 500 | 23/12 | 10/6 | 2/10 | 9/10 | 13/7 | 0/12 |
| cross | JPEG-1 vs JPEG-2000 | 500 | 54/0 | 8/7 | 2/18 | 0/12 | 3/21 | 10/1 |
| cross | JPEG-1 vs JPEG-AI | 500 | 4/15 | 4/2 | 1/9 | 7/1 | 5/11 | 2/7 |
| cross | JPEG-1 vs JPEG-XL | 500 | 7/17 | 5/9 | 5/14 | 4/6 | 3/13 | 7/11 |
| cross | JPEG-1 vs VVC | 500 | 39/10 | 3/20 | 4/22 | 4/14 | 2/34 | 9/6 |
| cross | JPEG-2000 vs JPEG-AI | 500 | 34/26 | 13/7 | 5/9 | 9/3 | 11/13 | 6/8 |
| cross | JPEG-2000 vs JPEG-XL | 500 | 49/6 | 6/6 | 15/5 | 12/4 | 11/4 | 13/3 |
| cross | JPEG-2000 vs VVC | 500 | 15/5 | 1/13 | 2/9 | 7/8 | 14/7 | 5/6 |
| cross | JPEG-AI vs JPEG-XL | 500 | 4/19 | 6/3 | 1/10 | 3/6 | 4/15 | 2/12 |
| cross | JPEG-AI vs VVC | 500 | 30/12 | 11/13 | 6/20 | 5/16 | 9/7 | 5/11 |
| cross | JPEG-XL vs VVC | 500 | 34/8 | 4/13 | 5/12 | 10/11 | 7/10 | 7/8 |

**TID2013 JPEG+JPEG2000 (TRAIN, descriptive)** — peer butter_p3

| class | codec pair | n | C |
|---|---|---|---|
| same | JPEG (same) | 250 | 0/0 |
| same | JPEG2000 (same) | 249 | 0/0 |
| cross | JPEG vs JPEG2000 | 625 | 15/11 |

**JPEG-AIC forced choice btc_native** — peer ssim2

| class | codec pair | n | B | C |
|---|---|---|---|---|
| same | AVIF (same) | 449 | 0/0 | 0/0 |
| same | JPEG-1 (same) | 449 | 0/0 | 1/1 |
| same | JPEG-2000 (same) | 449 | 0/0 | 0/0 |
| same | JPEG-AI (same) | 450 | 0/0 | 0/0 |
| same | JPEG-XL (same) | 448 | 0/0 | 0/0 |
| same | VVC (same) | 450 | 0/0 | 0/0 |
| cross | AVIF vs JPEG-1 | 54 | 0/0 | 0/0 |
| cross | AVIF vs JPEG-2000 | 52 | 2/4 | 1/1 |
| cross | AVIF vs JPEG-AI | 24 | 4/0 | 0/0 |
| cross | AVIF vs JPEG-XL | 56 | 10/0 | 6/2 |
| cross | AVIF vs VVC | 52 | 1/1 | 1/1 |
| cross | JPEG-1 vs JPEG-2000 | 50 | 7/1 | 2/0 |
| cross | JPEG-1 vs JPEG-AI | 24 | 0/0 | 0/0 |
| cross | JPEG-1 vs JPEG-XL | 36 | 0/0 | 0/0 |
| cross | JPEG-1 vs VVC | 62 | 2/0 | 0/0 |
| cross | JPEG-2000 vs JPEG-AI | 12 | 2/0 | 0/0 |
| cross | JPEG-2000 vs JPEG-XL | 46 | 10/2 | 2/0 |
| cross | JPEG-2000 vs VVC | 50 | 1/1 | 0/0 |
| cross | JPEG-AI vs JPEG-XL | 16 | 0/0 | 0/2 |
| cross | JPEG-AI vs VVC | 34 | 0/0 | 0/2 |
| cross | JPEG-XL vs VVC | 42 | 2/0 | 0/2 |

