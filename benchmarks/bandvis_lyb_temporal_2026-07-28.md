# BANDVIS on LIVE-YT-Banding — FULL-TEMPORAL validation (2026-07-28)

Closes the "frame-sampled analog" caveat of
`benchmarks/bandvis_lyb_validation_2026-07-28.md`: that run scored 8
timestamp-matched frames per video; this run scores **every frame of all
120 videos** (31,422 FR frame pairs; 150–482 frames/video) and evaluates
four temporal poolings per signal. Same dataset, same official splits,
same driver.

Companion docs:
- `benchmarks/bandvis_lyb_validation_2026-07-28.md` — frame-sampled run,
  dataset/alignment details, per-content analysis, confound discussion.
- `benchmarks/bandvis_dither_retest_2026-07-28.md` — dither-robustness
  retest; verdict "dst-masking deferred; texture_dissim_s3 gates dither
  at AUC 0.023". The tex_s3 gate idea tested in the combo below comes
  from there.
- `benchmarks/bandvis_gate_validation_2026-07-27.md` — append2 gates.

## Protocol

- **Videos**: all 40 contents × (3 AV1 CQ renditions vs the H.264
  reference), LIVE-YT-Banding, 1080p yuv420p, MOS higher=better.
  Videos staged one content at a time from Tower
  (`/mnt/tower/input/datasets/live-yt-banding/videos/`), decoded with
  `ffmpeg -vsync 0` to PNG, scored, and deleted before the next content
  (peak working set ~2.4 GB; nothing materialized globally).
- **Pairing**: FR frame-i↔frame-i (both legs decoded with `-vsync 0`;
  frame counts matched exactly for all 120 videos — 0 mismatch WARNs).
- **Features**: `v2_ab_extract` in `foldapp2` mode (944-feature streaming
  walk, append2 ON) for the BANDVIS/tex/mscn signals + `v1stream` for the
  production codec_target score. Signals kept per frame:
  `gain_s0..3` = a2i(s,0), `loss_s0..3` = a2i(s,1), `tex_s3` =
  api(3,1,9), `mscn_s0` = api(0,1,5), `v1score`.
- **Temporal poolings** (per video, per signal over its frame series):
  - `mean` — plain average.
  - `worst1s` — max of sliding 1-second-window means (window = fps
    frames, via cumsum). "Worst second" of the video.
  - `softtopk` — CAMBI-style soft top-k: weights `w = u/(u+c)+1e-6`
    with `u = max(v − p60, 0)`, `c = 0.25·(p95 − p60) + 1e-9`;
    pooled = Σw·v/Σw. Emphasizes the high tail smoothly.
  - `p95` — 95th percentile of the frame series.
- **Stats legs** (per mission spec, methods per
  `zenpapers/docs/iqa-methods/evaluation-statistics.md`):
  - SROCC pooled over all 120 videos (tie-aware).
  - SROCC on the **official 1000 content-aware test folds** (splits
    `.mat`, positions 129–160 per split = 32 test videos), mean ± sd.
  - PLCC after a 4-parameter logistic fit (scipy curve_fit; occasional
    overflow/covariance warnings from the logistic on weak signals are
    expected and harmless).
  - **Krasula different-vs-similar AUC (`AUC_ds`) and better-vs-worse
    accuracy (`BW_acc`)**. The LYB metadata has **no per-video CIs**, so
    pair labels use the significant-difference **proxy: |ΔMOS| > 10**
    (→ 5,045 "different" / 2,095 "similar" pairs). AUC_ds uses
    |Δfeature| and is polarity-free; BW_acc is signed and computed with
    error-feature polarity (lower = better) for every signal except
    `v1score` (a quality score, flipped back). mscn_s0's BW_acc below is
    therefore under the lower-is-better assumption; in its empirical
    (confounded, positive) direction it is 1 − shown.

Pipeline: `~/tmp/lyb_temporal.py` (session scratch; protocol fully
described here). Analysis: `benchmarks/bandvis_lyb_temporal_eval_2026-07-28.py`.

## Results — signal × pooling

| signal | pooling | SROCC | folds mean±sd | PLCC | AUC_ds | BW_acc |
|---|---|---:|---:|---:|---:|---:|
| **loss_s3** | **mean** | **−0.4410** | **−0.4347 ± 0.2502** | **0.4985** | 0.5365 | **0.7033** |
| loss_s3 | worst1s | −0.4090 | −0.4002 ± 0.2526 | 0.4524 | 0.5347 | 0.6866 |
| loss_s3 | softtopk | −0.4060 | −0.3971 ± 0.2511 | 0.4640 | 0.5332 | 0.6842 |
| loss_s3 | p95 | −0.3989 | −0.3901 ± 0.2527 | 0.4543 | 0.5302 | 0.6815 |
| loss_s2 | mean | −0.3915 | −0.3863 ± 0.2625 | 0.4395 | 0.5331 | 0.6755 |
| loss_s2 | worst1s | −0.3897 | −0.3801 ± 0.2665 | 0.4114 | 0.5459 | 0.6751 |
| loss_s2 | softtopk | −0.3827 | −0.3731 ± 0.2647 | 0.4195 | 0.5421 | 0.6727 |
| loss_s2 | p95 | −0.3773 | −0.3649 ± 0.2694 | 0.4124 | 0.5427 | 0.6696 |
| gain_s3 | mean | −0.1750 | −0.1597 ± 0.2501 | 0.2246 | 0.4801 | 0.5788 |
| gain_s3 | worst1s | −0.2006 | −0.1821 ± 0.2521 | 0.2809 | 0.4802 | 0.5883 |
| gain_s3 | softtopk | −0.2258 | −0.2032 ± 0.2439 | 0.2701 | 0.4820 | 0.5992 |
| gain_s3 | p95 | −0.2346 | −0.2073 ± 0.2445 | 0.2775 | 0.4798 | 0.6026 |
| tex_s3 | mean | −0.0196 | −0.0693 ± 0.1867 | 0.1693 | 0.4876 | 0.5112 |
| tex_s3 | worst1s | −0.0575 | −0.1016 ± 0.1802 | 0.1073 | 0.4944 | 0.5277 |
| tex_s3 | softtopk | −0.0360 | −0.0838 ± 0.1860 | 0.1073 | 0.4911 | 0.5195 |
| tex_s3 | p95 | −0.0517 | −0.0986 ± 0.1700 | 0.1073 | 0.4869 | 0.5275 |
| mscn_s0 | mean | +0.3205 | +0.2663 ± 0.2359 | 0.3187 | 0.4949 | 0.3661 |
| mscn_s0 | worst1s | +0.2285 | +0.1833 ± 0.2595 | 0.2599 | 0.4940 | 0.4065 |
| mscn_s0 | softtopk | +0.2392 | +0.1932 ± 0.2623 | 0.2847 | 0.4982 | 0.4014 |
| mscn_s0 | p95 | +0.2204 | +0.1753 ± 0.2679 | 0.2724 | 0.5019 | 0.4103 |
| v1score | mean | +0.1223 | +0.1507 ± 0.1751 | 0.1692 | 0.4990 | 0.5566 |
| v1score | worst1s | +0.1060 | +0.1364 ± 0.1809 | 0.1723 | 0.4923 | 0.5495 |
| v1score | softtopk | +0.1167 | +0.1467 ± 0.1778 | 0.1984 | 0.4942 | 0.5548 |
| v1score | p95 | +0.1065 | +0.1369 ± 0.1837 | 0.1901 | 0.4907 | 0.5500 |

**Best single signal (by |fold mean|): BANDVIS_LOSS s3 × mean —
pooled −0.4410, folds −0.4347 ± 0.2502, PLCC 0.4985, BW_acc 0.7033.**

## Findings

1. **The frame-sampling caveat closes with the number intact.**
   Frame-sampled (8/~240 frames): loss_s3 −0.447 pooled / −0.441 ±
   0.252 folds. Full-temporal (all frames, mean pooling): −0.4410 /
   −0.4347 ± 0.2502. mscn_s0 likewise: +0.322/+0.268±0.237 sampled vs
   +0.3205/+0.2663±0.2359 full. Eight uniformly-spaced frames were
   already a faithful estimate of the mean-pooled full-video statistic
   on this dataset; nothing about the frame-sampled conclusions changes.

2. **Mean pooling wins for LOSS; peak poolings only help GAIN.**
   worst1s/softtopk/p95 all *reduce* loss_s3 (−0.40 vs −0.44) — banding
   visibility loss on these AV1 renditions is sustained, not transient,
   so emphasizing the worst second adds variance without signal. GAIN
   moves the other way (mean −0.175 → p95 −0.235): the oversmoothing that
   GAIN detects is peaky. But GAIN stays a weak signal on this dataset
   regardless of pooling (see the frame-sampled doc's confound analysis:
   AV1 CQ smoothing couples banding with content difficulty).

3. **The pre-registered 3-feature combo does NOT beat the single
   feature.** Combo = [loss_s3×softtopk, tex_s3×mean, mscn_s0×mean]
   (BANDVIS workhorse + the dither-gate texture term from
   `bandvis_dither_retest_2026-07-28.md` + the best pre-existing
   feature), least-squares fit per split on its 72 train videos only,
   evaluated on that split's 32 test videos: **test-fold SROCC +0.3996 ±
   0.2361** vs single loss_s3×mean −0.4347 ± 0.2502. The combo *loses*
   ~0.035 SROCC: tex_s3 carries ≈0 rank signal here (best −0.10 folds)
   and mscn_s0's positive sign is content-difficulty confound, so both
   add noise on 72-video fits. The dither gate earns its keep as a
   *robustness* gate (dither-vs-banding separation, AUC 0.023 in the
   retest), not as an MOS-regression feature on LYB.

4. **Krasula legs are modest and honest.** loss_s3×mean separates
   better-vs-worse at 0.703 accuracy but different-vs-similar at only
   AUC 0.537 — the feature ranks pairs usefully but its magnitude does
   not track *how* different two renditions look. With no CIs in the
   metadata the |ΔMOS|>10 proxy is coarse; treat AUC_ds as indicative
   only.

5. **Production v1 score stays near-uninformative on banding**
   (+0.15 ± 0.18 folds at best) — unchanged from the frame-sampled run;
   banding is exactly the failure mode BANDVIS was added to cover.

## Honest comparison vs CAMBI

CAMBI's recorded SROCC on LIVE-YT-Banding is **0.7143**. This run is now
**protocol-comparable on the temporal axis** — all frames, temporally
pooled (including a CAMBI-style soft-top-k), official content-aware
splits — and BANDVIS_LOSS s3 lands at **0.44** (0.4347 ± 0.2502 on the
folds). CAMBI is clearly stronger on its home dataset, and the remaining
protocol differences do not explain the gap away: CAMBI is a
*purpose-built* NR banding detector (multiscale contrast-aware banding
index with luma-adaptive thresholds and top-percentile temporal
pooling), while BANDVIS_LOSS is **one general-purpose feature of 944**
in an FR metric, mean-pooled, with constants derived from a synthetic
SDR study — not tuned on LYB or on any banding MOS data. The honest
statement: BANDVIS gives zensim a real, fold-robust banding signal where
the production score has none (+0.15), at ~0.6× CAMBI's correlation; it
is not a CAMBI replacement, and closing the gap would take
banding-specific calibration (per-luma thresholds, spatial pooling of
banded *area* rather than mean visibility) that is out of scope for a
single append2 slot.

## Residuals / limitations

1. **FR-vs-NR mismatch stands**: BANDVIS scores banding *introduced
   relative to the H.264 reference*; the references themselves contain
   banding (see the frame-sampled doc), which caps FR headroom. CAMBI
   judges absolute banding and is immune to reference contamination.
2. **|ΔMOS|>10 Krasula proxy** is not the CI-based pair labeling the
   methodology doc prefers; LYB metadata ships no CIs. Stated wherever
   AUC_ds is quoted.
3. **Combo search was pre-registered, not exhaustive** — only the
   mission-specified triple was fit. A wider feature search on LYB would
   be post-hoc tuning on the validation set; not done deliberately.
4. **Fold sd is large (±0.25)** — 32-video test folds on a 40-content
   dataset are intrinsically noisy; the sd reflects content diversity,
   not instability of the extraction (which is deterministic).
5. **Speed**: full-temporal scoring of 120 videos (31,422 1080p FR
   pairs, two feature modes) took ~50 min wall on 14 threads — fine for
   validation, not a per-frame video-QA product path.

## Verdict

Full-temporal evaluation **confirms** the frame-sampled result:
BANDVIS_LOSS s3, mean-pooled, is the strongest single banding signal
zensim has on LIVE-YT-Banding (−0.4347 ± 0.2502 official folds, PLCC
0.4985, BW_acc 0.703), beating the best pre-existing feature (mscn_s0,
|0.2663| folds) and the production score (+0.1507). Mean pooling is the
right default; the mission's acceptance bar (beat the best existing
single feature under the full-temporal protocol) is **met**. CAMBI
remains well ahead on its home turf (0.7143), as a purpose-built NR
banding metric should be.
