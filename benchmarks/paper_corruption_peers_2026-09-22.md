Devin-executed; Opus-reviewed 2026-09-23 (REVIEW_PAPER_MEASURE.md): PROMOTE

# Peer metrics on the canonical corruption packet — zensim-2026 paper lane

Measured 2026-09-23 UTC (run1 05:37–05:50Z, analysis 05:52Z) in
`zensim--paper-corruption` (lane commits `68ce6c10` `527ac623` `5e333548`
`88b51e05` `e7619d7b` + this record). Frame: **these metrics were not designed
as integrity detectors**; the table is informative about which quality signals
double as tripwires. Descriptive only — nothing was fitted or tuned.

Reviewer independently confirmed: 5,353 positives, 326 negatives, 8 origins; butteraugli-max pass@q20 0.6596 and detect@FP1% 0.3831; fast-ssim2 0.4235 / 0.1474; zensim B 0.2793 / 0.0065; zensim D 0.3633 / 0.1480.

## Protocol (owner-reproduced)



## Headline — full packet, VALIDATE (5,353 positives / 326 negatives / 8 origins)

| metric | orient | pass@q20 % [95% CI] | pass@q10 % | detect@FP1% % [95% CI] | detect@FP5% % |
|---|---|---|---|---|---|
| butteraugli-max | damage | 66.0 [62.6–70.1] | 61.2 | 38.3 [34.8–42.4] | 44.2 |
| butteraugli-3norm | damage | 60.2 [57.8–63.3] | 55.2 | 38.3 [34.9–42.2] | 42.3 |
| fast-ssim2 | quality | 42.4 [39.2–46.0] | 34.6 | 14.7 [13.7–16.1] | 19.7 |
| cvvdp-standard_4k | quality | 45.9 [40.3–51.4] | 36.9 | 16.4 [14.4–18.3] | 22.4 |
| cvvdp-standard_fhd | quality | 34.3 [29.4–39.6] | 27.2 | 11.6 [9.9–13.3] | 15.0 |
| gmsd | damage | 36.3 [32.5–40.4] | 26.3 | 8.2 [7.6–8.8] | 11.1 |
| dssim | damage | 36.3 [33.7–39.6] | 30.4 | 14.3 [13.5–15.2] | 18.6 |
| zensim-PreviewV0_2 | quality | 40.4 [36.9–44.4] | 32.7 | 13.9 [13.0–14.9] | 19.7 |
| zensim-B | quality | 27.9 [23.0–33.6] | 15.2 | 0.7 [0.4–0.9] | 2.7 |
| zensim-C | quality | 25.6 [21.2–30.8] | 15.9 | 7.3 [6.5–8.1] | 9.9 |
| zensim-D | quality | 36.3 [33.1–40.2] | 29.9 | 14.8 [12.9–16.8] | 18.3 |
| zensim-Rev3-fast-ens5 | quality | 38.7 [35.8–42.2] | 29.1 | 13.7 [13.4–14.0] | 17.3 |
| zensim-Rev3-rich-ens5 | quality | 38.5 [35.1–41.9] | 30.3 | 12.2 [11.2–13.2] | 16.9 |

## Headline — full packet, TRAIN (7,725 positives / 488 negatives / 12 origins)

| metric | pass@q20 % [95% CI] | pass@q10 % | detect@FP1% % [95% CI] | detect@FP5% % |
|---|---|---|---|---|
| butteraugli-max | 67.9 [64.7–71.0] | 62.0 | 40.0 [36.0–44.5] | 45.1 |
| butteraugli-3norm | 61.4 [58.3–64.8] | 55.7 | 38.8 [35.0–43.0] | 43.0 |
| fast-ssim2 | 43.9 [39.8–48.3] | 37.0 | 15.7 [15.0–16.4] | 21.3 |
| cvvdp-standard_4k | 45.6 [40.5–51.5] | 37.4 | 14.2 [13.6–15.0] | 20.6 |
| cvvdp-standard_fhd | 33.6 [28.0–40.0] | 26.6 | 7.9 [7.3–8.5] | 12.7 |
| gmsd | 36.8 [30.9–43.6] | 26.7 | 9.0 [7.6–10.8] | 11.7 |
| dssim | 37.7 [33.9–42.0] | 31.2 | 13.8 [12.9–14.8] | 18.4 |
| zensim-PreviewV0_2 | 41.7 [36.7–47.4] | 34.2 | 14.3 [13.0–15.9] | 20.6 |
| zensim-B | 27.7 [22.2–34.2] | 17.2 | 0.6 [0.4–0.9] | 2.4 |
| zensim-C | 27.7 [21.4–34.4] | 20.4 | 6.4 [5.7–7.3] | 10.3 |
| zensim-D | 37.2 [32.5–42.5] | 31.1 | 13.8 [12.0–16.1] | 19.7 |
| zensim-Rev3-fast-ens5 | 39.5 [35.4–44.0] | 32.5 | 13.3 [12.4–14.7] | 16.8 |
| zensim-Rev3-rich-ens5 | 39.6 [35.2–44.4] | 33.0 | 13.6 [12.1–15.5] | 17.8 |

## IW-SSIM subset (`iwssim176`; adds iw-ssim)

validate (2,554 pos / 163 neg / 4 origins): iw-ssim pass@q20 36.3 [30.6–40.7],
detect@FP1% 15.6 [13.5–17.5] — between dssim and fast-ssim2 (46.8 / 19.0).
train (5,290 / 324 / 8): iw-ssim pass@q20 30.1 [26.1–34.5], detect@FP1%
9.0 [7.8–10.3]. All other metrics re-measured on the subset; full tables in
`tables.md`.

## Operating point transferred out of sample (extension, labelled)

Threshold = TRAIN negatives' quantile, applied unchanged to VALIDATE
(realised validate FP next to it). Full packet:

| metric | FP1 target: validate detect % [95% CI] | realised FP % | FP5: detect % | realised FP % |
|---|---|---|---|---|
| butteraugli-max | 38.6 [35.2–42.7] | 1.2 | 43.8 | 4.3 |
| butteraugli-3norm | 37.0 [33.8–40.7] | 0.6 | 41.7 | 4.9 |
| fast-ssim2 | 15.6 [14.5–17.0] | 1.8 | 20.8 | 5.5 |
| cvvdp-standard_fhd | 8.5 [6.9–10.3] | 0.0 | 12.8 | 4.0 |
| cvvdp-standard_4k | 14.3 [12.3–16.2] | 0.0 | 21.0 | 4.3 |
| gmsd | 8.1 [7.6–8.7] | 0.6 | 10.0 | 3.7 |
| dssim | 13.6 [12.8–14.5] | 0.6 | 17.8 | 4.3 |
| zensim-PreviewV0_2 | 13.9 [13.0–14.9] | 1.2 | 20.2 | 6.1 |
| zensim-B | 0.5 [0.4–0.7] | 0.9 | 2.5 | 4.6 |
| zensim-C | 6.8 [6.0–7.6] | 0.9 | 10.0 | 5.5 |
| zensim-D | 13.6 [11.8–15.5] | 0.3 | 19.2 | 6.4 |
| zensim-Rev3-fast-ens5 | 13.6 [13.3–14.0] | 0.9 | 16.8 | 4.9 |
| zensim-Rev3-rich-ens5 | 12.4 [11.4–13.4] | 1.8 | 16.7 | 4.6 |

## Paired differences (full/validate, Δ vs reference peers, points)

Same unique positives, origins resampled jointly. butteraugli-max beats
fast-ssim2 by +23.6 pass@q20 [+18.9,+28.0] and +23.6 detect@FP1%
[+20.4,+27.2]; butteraugli-3norm +17.9/+23.6 vs fast-ssim2, −5.8/±0 vs
butteraugli-max. zensim D −6.0 pass@q20 [−7.7,−4.4] and +0.1 detect
[−1.6,+1.6] vs fast-ssim2; Rev3-fast −3.6/−1.0; PreviewV0_2 −2.0/−0.8.
cvvdp-4k +3.5 pass@q20 [+0.1,+6.4] vs fast-ssim2 (the only non-butteraugli
peer ahead on that stat). zensim-B −14.4/−14.1 (worst zensim row).
Full Δ tables for both splits and the iwssim176 set in `tables.md` /
`paired_vs_reference.json`.

## Cross-checks (recorded, reproducible)



## Fairness notes the paper must carry



## Inputs / binaries (sha256 in `/var/tmp/paper-corruption/BINARIES.sha256`, `analysis/RUN.json`)



## Reproduce

```
research/2026-09-paper/corruption_peer_matrix.py run|crosscheck|render   # driver
~/tmp/devin/paper_corruption_run1.sh    # scoring (inputs listed there)
~/tmp/devin/paper_corruption_analyze.sh # analysis -> /var/tmp/paper-corruption/analysis
```

Bulk per-pair scores, audit subsets, per-family/region/severity/content-class
breakdowns and logs: see `paper_corruption_peers_2026-09-22.pointer.md`.
