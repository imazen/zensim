# GMSBANK calibration and qualification preregistration (2026-09-23)

Lane `gmsbank`, based on `1881409d`; no labels read and no model fit in this lane.
The candidate is frozen before calibration. Its eventual human-label potential
test is a proposal for the potential owner, not an analysis here.

## Inputs and roles

- CID22 **TRAIN** bank keys: `/var/tmp/rev4-featbank/bank/cid22_train/keys.parquet`, SHA256 `c99a0887705b17bff05bdbfc55b89f9b4f060bc94a009302543f15d9a0127219`.
- SafeSyn **TRAIN** bank keys: `/var/tmp/rev4-featbank/bank/safesyn/keys.parquet`, SHA256 `12d48d7fc02afd5026067a348ea20a3896ea6de9a94bd99a25ec2db071c2b28f`.
- Only `pair_key`, image dimensions, ref/dist paths and pixel digests may be read from those tables. No label sidecar is opened. The bank key is recomputed from decoded RGB8 bytes before a sample is admitted.
- The cited paper corpus records are the GMSD and MS-GMSD converted papers under `/mnt/v/input/papers/03/03d268a2140b4dfb8e86d30c6302db6c8eca0f99e524642a5939de291146c9b6.md` (SHA256 `60ef4689e4f7e2d0695d52bbd8f5b768c0dc5a05460fc8ecb7dde2e787bdaec3`) and `/mnt/v/input/papers/25/25effe3f7937f57333471738f104524caab1014d1a271ee70c03ad48798f9131.md` (SHA256 `2d72d4fd8c31322da09cab0cfee822ee2716980583bc735659e098e0bbbb9289`).

## Frozen feature definition

Append `gmsbank` f1322–f1501, 15 features per X/Y/B channel at each of four scales. For each of five constants `c_k = c_mid * 4^(k-2)`, emit `loss_k = mean[(1-GMS_k) 1(m_d<m_r)]`, `gain_k = mean[(1-GMS_k) 1(m_d>=m_r)]`, and `dev_k = population_std(GMS_k)`. `GMS_k = (2*m_r*m_d+c_k)/(m_r²+m_d²+c_k)`, on the existing central-difference, unit-XYB gradient magnitudes. Divide every sum by true width×height. Row-ordered f64 sums, per-row f64 Welford, and row-ordered Chan merge determine pooling; identity must emit exact zero. This differs from GMSD's gamma-luma, 2× box and Prewitt front end. The exact peer GMSD and its mean GMS are separate controls.

## Pixel-only calibration

The ratio is `R = median(m_Prewitt_gamma_luma / m_XYB_Y_s1)` at co-sited samples, with `m_Prewitt_gamma_luma` on the floor-half-size 2×2 box grid and `m_XYB_Y_s1` from the research engine's scale-1 Y plane. A site is eligible when both magnitudes exceed `1e-6`. Use both reference and distorted planes, weighted equally by pair, and report site counts. The gamma-luma conversion must follow the peer GMSD's RGB8 luma route. Pixel pairing and border treatment are pinned in the calibration implementation record before measurement; any coordinate mismatch is a failure, not a reason to alter the definition after viewing results. `c_mid = 0.0026 / R²`, and the five constants are then source constants. No label-informed adjustment is allowed.

Size strata use `max(width,height)`: tiny <256, small 256–511, medium 512–1023, large >=1024. Content strata (photo, screen, line art, mixed) are assigned to source references using a deterministic RGB8 pixel classifier defined and tested before the ratio sweep; its thresholds, counts and failures must be recorded. It may use image pixels only. For each corpus × class × size stratum, take at most 32 distinct references in ascending SHA256 of `ref_group`, then at most two distortions per reference in ascending `pair_key`; include both planes. Empty strata remain empty and are reported. Equal-weight the per-pair medians for the pooled median, so huge images do not dominate. Report p25/p50/p75 and n by stratum, and pooled p25/p50/p75. No resampling, random seed or bootstrap CI is used for this deterministic unit conversion; the quartile spread is the uncertainty description. No hypothesis test is made from it.

## Qualification gates frozen now

- Existing f0–f1321: bitwise equality on the C1–C4 144-pair corpus × serial/MT8 × native/v3/scalar matrix, on/off and against base `1881409d`. Failure blocks adoption.
- New slots: identity 0 in every tier; `loss+gain` non-increasing in `c`; blur/noise sign on at least four TRAIN refs; strip/stride/thread invariance and tier parity; an independent NumPy reference on >=8 TRAIN pairs with max relative error <=1e-6 and deliberately wrong-`c` negative control.
- Cost: interleaved zenbench 256², 1024², 2048², 4096² at ST/MT8; fit marginal time to `alpha + beta*pixels` by ordinary least squares, report raw repetitions and contention. Goal <=5% of full-rev4; a miss is reported without retuning.
- Peer: all 18 bank sets, keyed by recomputed pixel digests and `legacy-rgb8` pair key; exact row coverage, no labels. If decode hashes disagree, stop and report.
- Measure-first: 2,000 SafeSyn TRAIN rows, f0–f985 exact f32 bank parity, extraction wall time and heaptrack peak. Full-bank C8 extraction is reserved for the extraction owner after review.

The potential-owner proposal will define P0–P3 and size-matched permutations, the D1/D2 roles and D5 adoption bar. This lane does not read human labels, run SROCC, calculate CIs or choose a fitted model.
