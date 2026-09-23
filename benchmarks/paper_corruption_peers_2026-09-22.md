Devin-executed; Opus-reviewed 2026-09-23 (REVIEW_PAPER_MEASURE.md): PROMOTE

# Peer metrics on the canonical corruption packet — zensim-2026 paper lane

Measured 2026-09-23 UTC (run1 05:37–05:50Z, analysis 05:52Z) in
`zensim--paper-corruption` (lane commits `68ce6c10` `527ac623` `5e333548`
`88b51e05` `e7619d7b` + this record). Frame: **these metrics were not designed
as integrity detectors**; the table is informative about which quality signals
double as tripwires. Descriptive only — nothing was fitted or tuned.

Reviewer independently confirmed: 5,353 positives, 326 negatives, 8 origins; butteraugli-max pass@q20 0.6596 and detect@FP1% 0.3831; fast-ssim2 0.4235 / 0.1474; zensim B 0.2793 / 0.0065; zensim D 0.3633 / 0.1480.

## Protocol (owner-reproduced)

- Population: canonical corruption packet 2026-09-08 via the DVIFM-ish §6 pair
  lists (`/mnt/v/output/zensim/dvifmish-eval-2026-09-22/pairs/corruption_{validate,train}.tsv`;
  validate 8 origins / 6,024 rows; train 12 origins / 9,036 rows).
- Owner: `corruption_eval.py` (DVIFM-ish §6 gate owner, sha256 `8e8e45d7…`,
  revised 05:35Z). One row per unique (origin, ref pixels, dist pixels);
  positives = non-inert corruptions; negatives = everything else incl. inert
  attempts. pass@q20 = corruption scored worse than its source's q20 anchor
  (strict `>`); detect@FPk% = fraction of positives above the (100−k)
  percentile of the same split's unique negatives. Per-family numbers assert
  equal to the owner's.
- All metrics normalised to higher-is-better before the owner sees them
  (`damage` metrics negated); orientation mapping declared per metric below.
- CIs: origin-clustered bootstrap, n=2000, seed 0
  (`scripts/v_next/corrhead_tests.py::boot_ci`). Paired differences resample
  the same origins in both arms.
- IW-SSIM entered only the `iwssim176` views (origins with min(W,H) ≥ 176:
  validate 4 origins/2,554 positives; train 8/5,290) — its 5-scale
  configuration needs ≥176 px.
- Exposure: `docs/DATA_SPLITS.md` ledger entry "2026-09-22: canonical
  corruption packet, peer scoring for the paper", committed (`68ce6c10`)
  before any score read. No human labels exist in this packet.
- Scoring ran outside the heavy lock (untimed accuracy work, run-heavy
  `--jobs 4 --mem 8G`, cpus 28–31) under the patrol rule; loadavg 9.7→18.0.
  RC=0, 742 s, peak RSS 1.06 GiB; 0 NaN; pair order verified.

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
| dvifmish-talk-faithful-luma | damage | 30.1 [29.0–31.3] | 25.5 | 13.9 [12.4–15.6] | 16.7 |
| dvifmish-serving-gate-ycbcr3 | damage | 22.8 [17.9–28.7] | 18.4 | 8.0 [7.0–9.2] | 10.1 |
| dvifmish-ours-full-luma | damage | 20.5 [15.5–26.0] | 16.8 | 8.9 [6.4–11.5] | 11.0 |
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
| dvifmish-talk-faithful-luma | 31.0 [27.2–35.3] | 26.7 | 14.4 [12.3–16.8] | 17.8 |
| dvifmish-serving-gate-ycbcr3 | 24.1 [18.8–29.8] | 19.6 | 8.7 [7.1–10.6] | 12.0 |
| dvifmish-ours-full-luma | 20.6 [15.1–26.8] | 16.9 | 9.1 [7.3–11.4] | 10.4 |
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
| dvifmish-talk-faithful-luma | 13.4 [11.9–15.0] | 0.9 | 16.9 | 5.8 |
| dvifmish-serving-gate-ycbcr3 | 8.1 [7.1–9.2] | 1.2 | 11.2 | 7.4 |
| dvifmish-ours-full-luma | 9.5 [6.9–12.4] | 3.1 | 11.4 | 6.1 |
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

- Pixel-API `score_pairs_tuner` D reproduces the canonical serving record's
  base D **exactly** (max|Δ| = 0.0) on 5,720 validate / 8,580 train common
  distorted paths → the pixel-API B/D columns are the values used here (the
  DVIFM-ish feature-path bake TSVs differ by up to ~0.1 pt after re-join —
  Rev3-era features; documented in the coordinator's workflow note).
- Rev3-rich-ens5 feature-path TSV vs pixel API under the verified
  reference-basename join: max|Δ| 2.7e-5 (validate) — join confirmed; the
  earlier positional join is mis-paired (max|Δ| ~310) and unused.
- 0 exact ties decided by the strict `>` anywhere (all metrics, both splits,
  both sets) — 6-decimal TSV quantisation decides nothing.
- Equal pass counts that coincided (dssim=gmsd validate, B=C iwssim176 train)
  are different pass sets (294/294, 285/285 discordant) — coincidence, not a
  join bug.

## Fairness notes the paper must carry

- butteraugli-max is the strongest tripwire on this packet (validate pass@q20
  66.0, detect@FP1% 38.3) — report it as a peer win; its max-norm is designed
  to flag the worst local distortion, which is what this packet measures.
- CVVDP's number depends on display geometry: standard_4k (more pixels per
  degree, distortions less visible *per its own model*) beats standard_fhd by
  ~12 pts pass@q20. Both recorded; the AIC studies' standard_fhd is one row.
- DVIFM-ish rows are our reimplementation of the published/talk DVIFM
  configuration (build `49aaf667`, presets talk-faithful-luma /
  serving-gate-ycbcr3 / ours-full-luma), never DVIFM itself.
- zensim-B's near-zero detect@FP1% (0.7%) is honest: B is calibrated for
  codec-quality targeting, not integrity; mark it informative-only.
- validate negatives (326) are fewer than train (488); validate CIs are
  wider. The transferred threshold shows several metrics' realised FP off
  target on validate (0–7.4%) — a per-split operating point is not portable.
- All metrics saw the same decoded pixels (zenpng decode, RGB8) and the same
  pair lists; no per-metric tuning. `detect@FPk%` uses each metric's own
  negative distribution — that is the generous comparison.

## Inputs / binaries (sha256 in `/var/tmp/paper-corruption/BINARIES.sha256`, `analysis/RUN.json`)

- `zenmetrics` @ cvvdpfix tip `d6e5ae96` (this lane's build `0325ac0e`, adds
  `--display-model` for CPU CVVDP) — cvvdp rows.
- `zenmetrics` GMSD-lane build `cdf1418a` (gmsd bit-exact vs libgmsd; also
  dssim, iwssim rows).
- `dvifmish-49aaf667` `69cd2309` — three presets.
- `score_pairs_tuner` `4c5617f0` + frozen R915 bakes (Rev3 rich basic228 ens5:
  seeds s17101/03/07/11/13; Rev3 fast y60 ens5: same seeds) + named profiles
  PreviewV0_2/B/C/D.
- Peer inputs reused from the DVIFM-ish §6 run: fast-ssim2 CSVs and
  butteraugli TSVs under `/var/tmp/dvifmish/peers/` (shas in RUN.json).

## Reproduce

```
research/2026-09-paper/corruption_peer_matrix.py run|crosscheck|render   # driver
~/tmp/devin/paper_corruption_run1.sh    # scoring (inputs listed there)
~/tmp/devin/paper_corruption_analyze.sh # analysis -> /var/tmp/paper-corruption/analysis
```

Bulk per-pair scores, audit subsets, per-family/region/severity/content-class
breakdowns and logs: see `paper_corruption_peers_2026-09-22.pointer.md`.
