# BANDVIS external validation — LIVE-YT-Banding (2026-07-28)

The acceptance gate `append2_bandvis_gates_2026-07-27.md` REMAINDERS #1 owed:
BANDVIS (f924+ append2, curvature band-pass) against the only open
banding-MOS set. Dataset: LIVE-YT-Banding (CBAND paper, TIP 2025) — 40
contents × (1 near-lossless H.264 ref + 3 AV1 CQ), 1080p yuv420p 8-bit,
~8 s; MOS for all 160 (higher = better; refs mean 68.6, distorted 46.3,
range 11.5–88.6). Videos on Tower (sha-verified), labels + official
1000×(24/8/8 by-source) splits local — see
`zenpapers:datasets/live-yt-banding.pointer.md`.

## Protocol (frame-sampled FR analog — NOT CAMBI's full-video temporal protocol)

- fps verified CONSISTENT within every content (metadata sweep, 0
  mismatches; 24/25/30/60 across contents), so timestamp seeking aligns
  frame-for-frame. Alignment probed empirically: ref@2.0s vs cq11@2.0s
  MAD 0.60 codes vs 6.24 at a 0.5 s offset (10×) — aligned.
- 8 frames/video at t_i = duration·(i+0.5)/8, same t across a content's
  4 versions; ffmpeg single-frame seeks; per-content NFS staging to
  `~/tmp` with delete-as-you-go (pipeline log: `~/tmp/lyb_pipeline.log`).
- FR pairs: ref-frame vs each CQ version's frame → 960 pairs (40×3×8);
  scored with `v2_ab_extract` `ZENSIM_AB_MODE=foldapp2` (944 cols) at the
  landed append2 tip. Per-video feature = mean over its 8 frames.
- Analysis: `benchmarks/bandvis_lyb_eval_2026-07-28.py` (committed; own
  tie-aware Spearman; official splits decoded per `data_info_maker.m`:
  positions 129..160 of each split row = the 8-source test fold).
- Refs are MOS-rated too but FR-score identically-0 against themselves;
  all correlations below are on the 120 distorted videos.

## Results (SROCC vs MOS; MOS higher = better, error features expect negative)

### (a) BANDVIS slots

| slot | pooled (120) | official test folds (1000×24) |
|---|--:|--:|
| GAIN s0 | −0.023 | |
| GAIN s1 | −0.094 | |
| GAIN s2 | −0.123 | |
| GAIN s3 (best gain) | −0.163 | −0.154 ± 0.250 |
| GAIN mean-of-scales | −0.113 | |
| LOSS s0 | −0.095 | |
| LOSS s1 | −0.323 | |
| LOSS s2 | −0.391 | |
| **LOSS s3 (best overall)** | **−0.447** | **−0.441 ± 0.252** |

### (b) metric-score baseline

- zensim codec_target v1-streaming score (same frames): SROCC
  **+0.1057** — near-uninformative (and wrong-signed) on this
  banding-specialist set, the same content-difficulty confound as (c);
  independent confirmation that general-quality scores do not cover
  banding and a specialist slot is warranted.

### (c) best existing single features (Y)

| feature | pooled | folds |
|---|--:|--:|
| mscn_s0 (best existing) | +0.322 | +0.268 ± 0.237 |
| art_s0 | +0.295 | |
| ringing_s1 / s0 | +0.291 / +0.288 | |
| mse_s0 [sign-check] | +0.181 | |
| ssim_mean_s0 [sign-check] | +0.187 | |

The POSITIVE signs on error features are the content-difficulty confound:
pooled ranking across contents is dominated by which CONTENTS are hard
(complex/grainy high-MOS content carries larger absolute FR error than
smooth banding-prone content). This is exactly why (d2) matters.

### (d2) within-content 3-point CQ ladders (confound-free direction check)

| feature | mean SROCC over 40 contents |
|---|--:|
| bandvis_loss_s3 | −0.550 ± 0.660 |
| mscn_s0 | −0.550 ± 0.660 |
| mse_s0 | −0.550 ± 0.660 |
| bandvis_gain_s3 | +0.038 ± 0.825 |

Within a content, every error feature (incl. LOSS) tracks the CQ ladder
identically (3-point ladders quantize SROCC coarsely); LOSS's decisive
advantage over mscn/mse is CROSS-CONTENT calibration — its magnitude is
comparable across contents while raw-error magnitudes are content-scaled.
GAIN does not track the ladder within content.

### (e) limitation check (pinned dither/lattice cross-fire, on real data)

Top GAIN among high-MOS (>60) videos: racing_game cq17 (0.045, MOS 81.9),
sky2 cq43 (0.045, 68.1), ship cq43 (0.045, 67.9), sunset cq09 (0.041,
67.8) — sky/gradient/game content where mild smooth-gradient
quantization response fires at SMALL magnitude (u8-floor class; the
low-MOS end reaches similar gains, hence the weak pooled GAIN
discrimination). No large-magnitude false positives observed.

## Honest read vs CAMBI context

CAMBI's recorded 0.7143 SROCC is a FULL-VIDEO, temporally-pooled,
banding-specialist protocol; this run is an 8-frame FR feature probe —
not comparable head-to-head, and no beat/lose claim is made. Within this
protocol:

1. **The acceptance question (best existing single feature) — PASSED by
   the BANDVIS pair, via LOSS, not GAIN:** BANDVIS_LOSS s3 −0.441 ±
   0.252 on the official folds beats mscn_s0's +0.268 ± 0.237 (pooled
   −0.447 vs +0.322) — the strongest single-feature signal measured on
   this set from the whole 944 vector.
2. **GAIN — the banding-INTRODUCTION polarity the slot was designed for —
   is weak here** (−0.15): on AV1-compressed 1080p, banding degradation
   couples with SMOOTHING (the encoder removes the in-band
   micro-structure that masked banding; plateau steps stay shallow), so
   the removal side (LOSS) carries the quality signal while new-step
   detection under-discriminates at frame-sample granularity.
3. Both polarities ship in the slot pair, so the head gets the useful
   one either way; the finding re-ranks which polarity matters for
   AV1-era banding.

## Verdict

**BANDVIS earns its slots on this evidence** — the pair contains the
best single-feature predictor of banding MOS measured here (LOSS s3),
comfortably ahead of every existing 924 feature — with the honest
caveats: (i) the designed GAIN polarity is weak on this set and its
value remains unproven (temporal pooling + the dst-masking fix are the
plausible unlocks); (ii) this is a frame-sampled analog, not the CAMBI
protocol; a full-temporal harness (all-frame scoring + temporal pooling)
is the next tier of evidence; (iii) the LOO-on-944-bake criterion
(REMAINDERS #2) remains the training-side gate. Recommendation: keep
append2 as landed (default-OFF), proceed to the 944 extraction wave, and
let LOO adjudicate with this external evidence attached.

## Residuals

1. Full-temporal protocol (all frames + temporal pooling) for a true
   CAMBI comparison; frame sampling here is 8/~240.
2. The dst-side masking fix (append2 REMAINDERS #3) — retest GAIN after.
3. PLCC/logistic + Krasula AUC panel legs not run (rank-only here).
4. Refs' own MOS (40 rated refs) unused by FR — an NR or
   ref-vs-best-encode protocol could use them.
5. v1stream baseline is codec_target profile (the current production
   default), not the historical B profile.

## Reproduce

```
# pipeline (frame extraction + scoring; ~15 min): ~/tmp/lyb_pipeline.py pattern in this doc's log
# analysis:
python3 benchmarks/bandvis_lyb_eval_2026-07-28.py --out-dir ~/tmp/lyb-out
```
