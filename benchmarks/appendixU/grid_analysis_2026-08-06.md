# appendix U — grid analysis  (2150 evaluated cells: 1990 LIVE, 158 NULL)

## 1. The MEASURED null (ZERO cells: candidate weights exactly 0.0)

| axis | n | null mean | null sd | null p2.5 | null p97.5 | max abs |
|---|--:|--:|--:|--:|--:|--:|
| b9_signed | 158 | +0.000000 | 0.000000 | +0.000000 | +0.000000 | 0.000000 |
| hfnl_perref | 158 | +0.000000 | 0.000000 | +0.000000 | +0.000000 | 0.000000 |
| cid22 | 158 | +0.000000 | 0.000000 | +0.000000 | +0.000000 | 0.000000 |
| konjnd | 158 | +0.000000 | 0.000000 | +0.000000 | +0.000000 | 0.000000 |
| nonphoto | 158 | +0.000000 | 0.000000 | +0.000000 | +0.000000 | 0.000000 |
| csiq | 158 | +0.000000 | 0.000000 | +0.000000 | +0.000000 | 0.000000 |
| live | 158 | +0.000000 | 0.000000 | +0.000000 | +0.000000 | 0.000000 |
| imazen26 | 158 | +0.000000 | 0.000000 | +0.000000 | +0.000000 | 0.000000 |

The null is the appendix's floor instrument. When it comes out EXACTLY zero, as it does here, that is itself the result: a cell whose candidate coefficients are 0.0 scores bit-identically to base through the whole fit -> pack -> spline -> score chain, so the LIVE/ZERO split is exact and NONE of the spread below is fit noise. It also means the null supplies no usable floor, so the REGISTERED axis floors govern (U.5) and the paired bootstrap owns the remaining eval-sampling noise.

## 2. Primary objective — signed CID22 B9

base (arm A) signed B9 = -0.030504; LIVE cells n=1990
d_b9_signed: mean +0.00148 sd 0.04062 min -0.14905 max +0.23355
cells whose signed B9 reaches the F8 bar (>= 0.15): 15
cells whose ABS B9 reaches 0.15 but are INVERTED: 21  <- these would PASS F8 as implemented

## 3. Shortlist — ranked by signed B9, guards annotated

| # | arm | kind | cell | names | b9_signed | d_b9 | b9_abs | d_hfnl | guard regressions |
|--:|---|---|---|---|--:|--:|--:|--:|---|
| 1 | B | S | 924 | BANDVIS_GAIN@s0Y | +0.2058 | +0.2335 | 0.2058 | -0.4182 | cid22-0.092, nonphoto-0.102, csiq-0.030, live-0.013, imazen26-0.071 |
| 2 | B | Z | 924-929 | BANDVIS_GAIN@s0Y+BANDVIS_GAIN@s1Y | +0.2058 | +0.2335 | 0.2058 | -0.4182 | cid22-0.092, nonphoto-0.102, csiq-0.030, live-0.013, imazen26-0.071 |
| 3 | A | W | 162-163 | ssim_max@s0Y+art_max@s0Y | +0.1882 | +0.2187 | 0.1882 | -0.0281 | cid22-0.021, nonphoto-0.011, csiq-0.021 |
| 4 | A | W | 165-166 | ssim_p95@s0Y+art_p95@s0Y | +0.1882 | +0.2187 | 0.1882 | -0.0502 | cid22-0.032, nonphoto-0.028, csiq-0.037, imazen26-0.023 |
| 5 | A | W | 162-166 | ssim_max@s0Y+art_p95@s0Y | +0.1870 | +0.2175 | 0.1870 | -0.0282 | cid22-0.021, nonphoto-0.012, csiq-0.022 |
| 6 | A | W | 163-165 | art_max@s0Y+ssim_p95@s0Y | +0.1856 | +0.2161 | 0.1856 | -0.0443 | cid22-0.034, nonphoto-0.024, csiq-0.036, imazen26-0.018 |
| 7 | A | W | 164-165 | det_max@s0Y+ssim_p95@s0Y | +0.1844 | +0.2149 | 0.1844 | -0.0510 | cid22-0.033, nonphoto-0.026, csiq-0.039, imazen26-0.021 |
| 8 | B | W | 407-409 | HF_GAIN@s0Y+HF_MAG_LOSS@s0Y | +0.1817 | +0.2095 | 0.1817 | -0.0080 | cid22-0.019, konjnd-0.058, nonphoto-0.030, csiq-0.016, imazen26-0.023 |
| 9 | A | W | 162-164 | ssim_max@s0Y+det_max@s0Y | +0.1755 | +0.2060 | 0.1755 | -0.0301 | cid22-0.021, nonphoto-0.013, csiq-0.023 |
| 10 | A | C | 156-162 | ssim_max@s0X+ssim_max@s0Y | +0.1574 | +0.1879 | 0.1574 | -0.0190 | cid22-0.019, csiq-0.020 |

cells with NO guard regression outside floor: 1218 of 1990

## 4. Secondary objective — HF-NL per-ref

d_hfnl_perref: n=1990 mean -0.00504 sd 0.02705 min -0.41825 max +0.02495
cells clearing the 0.039 axis LSD: 0

rank correlation between d_b9_signed and d_hfnl_perref over 1990 LIVE cells: +0.0092
  -> near zero means the two high-fidelity axes are DIFFERENT problems (registered outcome (b)); strongly positive means one HF factor.

## 5. Did PAIRING beat SINGLETONS?

| axis | pairs with both members measured | pair > max(member) | pair > max(member) by > floor | best excess |
|---|--:|--:|--:|--:|
(the 'by > floor' column uses max(2 x null sd, the registered axis floor) — with a zero null that is the registered floor)
| b9_signed (floor 0.0000) | 875 | 70 | 70 | +0.1102 (HF_GAIN@s0Y+HF_MAG_LOSS@s0Y) |
| hfnl_perref (floor 0.0390) | 875 | 81 | 0 | +0.0159 (PJND_FRAGILITY@s0Y+EDGE_WIDTH_CHANGE@s0Y) |
| cid22 (floor 0.0050) | 875 | 66 | 12 | +0.0201 (GMS_DEV2@s3Y+GMS_DEV2@s3B) |

A pair that never exceeds the better of its two members is an additive combination of effects already available singly — the pairing hypothesis predicts SUPER-additivity, and this table is its test.

## 6. Where the movement lives (LIVE cells, median delta by block)

| arm | block | n LIVE | med d_b9 | max d_b9 | med d_hfnl | med d_cid22 | med d_konjnd |
|---|---|--:|--:|--:|--:|--:|--:|
| A | iw72 | 9 | -0.0063 | -0.0063 | +0.0035 | -0.0054 | -0.0223 |
| A | masked72 | 142 | +0.0000 | +0.1016 | -0.0008 | -0.0003 | -0.0019 |
| A | peak72 | 153 | +0.0222 | +0.2187 | -0.0022 | +0.0001 | +0.0055 |
| B | append204 | 406 | +0.0021 | +0.1012 | -0.0005 | -0.0004 | +0.0011 |
| B | append2_20 | 25 | +0.0246 | +0.2335 | -0.0428 | -0.0127 | -0.0166 |
| B | v2_348 | 1255 | +0.0021 | +0.2095 | -0.0005 | -0.0008 | +0.0004 |

## 7. By candidate family — did the HF-plausibility ranking predict anything?

| family (brief rank) | n LIVE | med dB9 | max dB9 | med dHFNL | max dHFNL | med dCID22 | max dCID22 | n cells with dCID22 > +0.005 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 near-threshold/JND | 372 | +0.0088 | +0.1060 | -0.0005 | +0.0234 | -0.0017 | +0.0603 | 58 |
| 2 artifact: BANDING | 114 | -0.0007 | +0.0624 | -0.0019 | +0.0234 | -0.0002 | +0.0603 | 45 |
| 2 artifact: BLOCKINESS | 100 | +0.0000 | +0.1033 | -0.0001 | +0.0219 | +0.0006 | +0.0531 | 42 |
| 2 artifact: RINGING | 117 | +0.0054 | +0.1060 | -0.0009 | +0.0197 | -0.0019 | +0.0531 | 10 |
| 2 artifact: EDGE_WIDTH | 89 | -0.0015 | +0.1019 | +0.0008 | +0.0197 | -0.0018 | +0.0503 | 9 |
| 3 BANDVIS (append2) | 18 | +0.0413 | +0.2335 | -0.0580 | -0.0145 | -0.0304 | +0.0042 | 0 |
| 4 HF gain/loss (v2) | 281 | +0.0021 | +0.2095 | -0.0012 | +0.0225 | -0.0002 | +0.0532 | 36 |
| 4 CONTRAST gain/loss | 57 | +0.0000 | +0.0068 | -0.0011 | +0.0019 | -0.0000 | +0.0030 | 0 |
| 5 soft-peak | 303 | +0.0000 | +0.1060 | +0.0001 | +0.0191 | -0.0010 | +0.0531 | 15 |
| 6 v1 peak72 | 153 | +0.0222 | +0.2187 | -0.0022 | +0.0047 | +0.0001 | +0.0033 | 0 |
| 6 v1 masked72 | 142 | +0.0000 | +0.1016 | -0.0008 | +0.0107 | -0.0003 | +0.0214 | 28 |
| 6 v1 iw72 | 9 | -0.0063 | -0.0063 | +0.0035 | +0.0035 | -0.0054 | -0.0054 | 0 |
| 7 scale-0 only | 587 | +0.0027 | +0.2335 | -0.0001 | +0.0248 | -0.0019 | +0.0522 | 61 |

The HF-plausibility ranking was frozen in U.3 BEFORE any fit. This table is its scorecard: a family that was ranked high and moves nothing is a falsified prior, and one ranked low that moves an axis is a finding the ranking missed.
